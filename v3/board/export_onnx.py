#!/usr/bin/env python3
"""ORT baselines (laptop, repo .venv): export FP32 ONNX + ORT INT8 QDQ for one net + checkpoint.
Adapted from v2/board/cpu/export_cpu_models.py @ 28dd2ad (V3: nets from v3/model/nets.py, checkpoint by
path, preprocessing of the checkpoint; the V2 numpy kinds are not carried over).

    .venv/bin/python v3/board/export_onnx.py --net resnet20_b --checkpoint v3/train/runs/<job>/<file>.pt
        [--mean M M M --std S S S] [--limit N]

Writes v3/board/data/<net>/onnx/ (gitignored):
  fp32.onnx         torch.onnx.export of nets.build(net) in eval mode (BN kept; ORT fuses it), opset 17,
                    dynamic batch
  int8_qdq.onnx     onnxruntime.quantization.quantize_static(fp32.onnx): QDQ, S8S8, symmetric activations and
                    weights, per-channel weights, MinMax, calibrated on train[0:1024] (the DPU calibration set),
                    same preprocessing (V2 A5 settings)
  torch_pred.npy, ort_fp32_pred.npy, ort_int8_pred.npy   laptop predictions on test[0:N], batch 1
                    (the board session's agreement references)
  ONNX_INFO.json    sha256s, versions, quantization settings, preprocessing, checkpoint, git state and the
                    laptop verification (accuracies, max |ORT FP32 - torch| logit diff), label "laptop, model"
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

BOARD = Path(__file__).resolve().parent
sys.path.insert(0, str(BOARD))
import laptop_common as lc  # noqa: E402
import baseline_common as bc  # noqa: E402

OPSET = 17
OUT_ROOT = BOARD / "data"


def export_fp32(model, path: Path):
    import torch
    torch.onnx.export(model, torch.zeros(1, 3, 32, 32), str(path), opset_version=OPSET, input_names=["x"],
                      output_names=["logits"], dynamic_axes={"x": {0: "N"}, "logits": {0: "N"}},
                      do_constant_folding=True)


def export_int8(src: Path, dst: Path, calib: np.ndarray) -> dict:
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"x": calib[i:i + 64]} for i in range(0, len(calib), 64)])

        def get_next(self):
            return next(self.it, None)

    pre = dst.parent / "fp32_preproc.onnx"
    preproc = "none"
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(src), str(pre))
        src, preproc = pre, "quant_pre_process"
    except Exception as e:  # noqa: BLE001  recorded, not fatal
        preproc = f"skipped ({type(e).__name__}: {e})"
    opts = {"ActivationSymmetric": True, "WeightSymmetric": True}
    quantize_static(str(src), str(dst), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, extra_options=opts)
    if pre.exists():
        pre.unlink()
    return {"method": "onnxruntime.quantization.quantize_static", "quant_format": "QDQ",
            "activation_type": "QInt8", "weight_type": "QInt8", "per_channel": True, "calibrate_method": "MinMax",
            "extra_options": opts, "pre_process": preproc, "calibration_images": f"train[0:{lc.CALIB_N}]",
            "calibration_batch": 64}


def ort_preds(path: Path, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    lg = np.stack([s.run(None, {"x": x[i:i + 1]})[0][0] for i in range(len(x))]).astype(np.float32)
    return lg, lg.argmax(1).astype(np.int64)


def export(net: str, ckpt: str, pp=None, limit=None, out_root: Path = OUT_ROOT) -> dict:
    import onnx
    import onnxruntime
    import torch
    t0 = time.perf_counter()
    model, ci = lc.load_checkpoint(net, ckpt, pp)
    x_u8, y = lc.cifar_test()
    n = len(y) if limit is None else min(limit, len(y))
    x = bc.preprocess(x_u8[:n], ci["preprocess"])
    calib = bc.preprocess(lc.cifar_train_head(lc.CALIB_N)[0], ci["preprocess"])
    d = out_root / net / "onnx"
    d.mkdir(parents=True, exist_ok=True)
    export_fp32(model, d / "fp32.onnx")
    qinfo = export_int8(d / "fp32.onnx", d / "int8_qdq.onnx", calib)
    lt, pt = lc.predict_torch(model, x)
    lf, pf = ort_preds(d / "fp32.onnx", x)
    _, pq = ort_preds(d / "int8_qdq.onnx", x)
    for name, p in (("torch_pred.npy", pt), ("ort_fp32_pred.npy", pf), ("ort_int8_pred.npy", pq)):
        np.save(d / name, p)
    ver = {"net": net, "n_images": n, "torch_fp32_correct": int((pt == y[:n]).sum()),
           "ort_fp32_correct": int((pf == y[:n]).sum()), "ort_int8_correct": int((pq == y[:n]).sum()),
           "ort_fp32_pred_equal_vs_torch": int((pf == pt).sum()),
           "ort_int8_pred_equal_vs_torch": int((pq == pt).sum()),
           "ort_fp32_max_abs_logit_diff_vs_torch": float(np.abs(lf - lt).max()),
           "label": "laptop verification, model (not board data)"}
    info = {"created_utc": bc.utc_now(), "net": net, **lc.git_state(), "checkpoint": ci["checkpoint"],
            "checkpoint_sha256": ci["checkpoint_sha256"], "checkpoint_kind": ci["checkpoint_kind"],
            "preprocess": ci["preprocess"], "test_set": f"test[0:{n}]", "onnx_opset": OPSET,
            "ort_int8_quantization": qinfo,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                         "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__},
            "verification": ver, "duration_s": round(time.perf_counter() - t0, 1),
            "files": {f.name: {"sha256": bc.sha256_file(f), "bytes": f.stat().st_size}
                      for f in sorted(d.iterdir()) if f.name != "ONNX_INFO.json"}}
    (d / "ONNX_INFO.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps({net: ver}, indent=1))
    print(f"[{net}] wrote {d} ({info['duration_s']} s; checkpoint kind {ci['checkpoint_kind']})")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mean", nargs=3, type=float, default=None)
    ap.add_argument("--std", nargs=3, type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="verification images (default all 10,000)")
    a = ap.parse_args(argv)
    export(a.net, a.checkpoint, lc.parse_preprocess(a.mean, a.std), a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
