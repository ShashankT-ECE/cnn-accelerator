#!/usr/bin/env python3
"""V2 Step 5 (A5): export the CPU-baseline data package. LAPTOP ONLY (needs torch).

Writes, per net, ``<out>/<net>/cpu/`` (default out = v2/board/data, gitignored):

    int8_ref.npz      integer parameters from v2/model (gos_golden.load_net via NET_CONFIGS)
    fp32.npz          FP32 checkpoint weights (NET_CONFIGS[net]["checkpoint"])
    fp32.onnx         torch.onnx.export of the legacy FP32 model (opset 17, dynamic batch)
    int8_qdq.onnx     onnxruntime.quantization.quantize_static(fp32.onnx): QDQ, S8S8,
                      symmetric activations and weights, per-channel weights, MinMax,
                      calibrated on the SAME images as the project INT8 calibration
                      (legacy make_transform, train[0:1024] for both nets)
    raw_test.npz      x_u8 raw test images (MNIST [N,28,28]; CIFAR [N,32,32,3] HWC), y
    CPU_MANIFEST.json sha256 of each file, versions, git state, calibration, verification

and then verifies on the laptop (recorded in the manifest, label "laptop"):
  * numpy preprocessing (cpu_infer.preprocess_fp32) == legacy torchvision transform,
    bitwise, on all 10k test images; numpy quantize_input == gos_golden.quantize_input;
  * cpu_int8_ref raw INT32 logits == gos_golden.run_net "v" and float32 logits bitwise
    equal, on all 10k test images; prediction accuracy;
  * FP32 numpy, ORT FP32, ORT INT8 10k accuracy (own numbers, not the project INT8);
  * ORT FP32 vs torch logits max |diff|.

Refuses a dirty git tree unless --allow-dirty (the manifest records git_dirty).

    .venv/bin/python v2/board/cpu/export_cpu_models.py [--out v2/board/data] [--allow-dirty]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "v2" / "model"))
sys.path.insert(0, str(HERE))

import common  # noqa: E402  (legacy python/ on sys.path)
import gos_golden  # noqa: E402
from net_config import NET_CONFIGS  # noqa: E402

import cpu_infer  # noqa: E402

CALIB_N = 1024
TEST_N = 10000
OPSET = 17


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True,
                          text=True, check=True).stdout.strip()


def git_state() -> dict:
    # Same convention as v2/model/common.git_dirty (generated outputs excluded);
    # the gitignored data package is not part of the tree anyway.
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", ".", *common.OUTPUT_PATHSPECS))}


# --------------------------------------------------------------------------- #
def legacy(net: str):
    from torchvision import datasets
    if net == "lenet5":
        from lenet5.model import LeNet5 as Model
        from lenet5.preprocess import make_transform
        ds_cls = datasets.MNIST
    else:
        from cifar10.model import Cifar10Net as Model
        from cifar10.preprocess import make_transform
        ds_cls = datasets.CIFAR10
    return Model, make_transform, ds_cls


def dataset(net: str, train: bool, transform=True):
    _, make_transform, ds_cls = legacy(net)
    return ds_cls(root=str(REPO_ROOT / "data/raw"), train=train, download=False,
                  transform=make_transform() if transform else None)


def raw_u8(ds) -> tuple[np.ndarray, np.ndarray]:
    x = ds.data
    x = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    y = ds.targets
    y = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    return np.ascontiguousarray(x.astype(np.uint8)), y.astype(np.int64)


def layers_json(net: str) -> str:
    keep = ("name", "IC", "OC", "KH", "KW", "IH", "IW", "OH", "OW", "relu", "pool", "final", "K")
    return json.dumps([{k: L[k] for k in keep} for L in NET_CONFIGS[net]["layers"]])


def export_int8_ref(net: str, d: Path) -> None:
    G = gos_golden.load_net(net)
    arrs = {"layers_json": np.array(layers_json(net)), "S_input": np.float64(G.S_input),
            "B": np.int64(G.B)}
    for P in G.layers:
        n = P.name
        arrs[f"{n}_q_w"] = P.q_w
        arrs[f"{n}_q_b"] = P.q_b
        if P.cfg["final"]:
            arrs[f"{n}_S_a"] = np.float64(P.S_a)
            arrs[f"{n}_S_w"] = np.asarray(P.S_w, dtype=np.float64)
        else:
            arrs[f"{n}_m"] = np.array(P.m, dtype=np.uint64)
            arrs[f"{n}_s"] = np.array(P.s, dtype=np.uint8)
    np.savez(d / "int8_ref.npz", **arrs)


def load_torch_model(net: str):
    import torch
    Model, _, _ = legacy(net)
    m = Model()
    m.load_state_dict(torch.load(REPO_ROOT / NET_CONFIGS[net]["checkpoint"], map_location="cpu"))
    m.eval()
    return m


def export_fp32(net: str, d: Path, model) -> None:
    sd = model.state_dict()
    arrs = {"layers_json": np.array(layers_json(net))}
    for L in NET_CONFIGS[net]["layers"]:
        n = L["name"]
        arrs[f"{n}_weight"] = sd[f"{n}.weight"].detach().numpy().astype(np.float32)
        arrs[f"{n}_bias"] = sd[f"{n}.bias"].detach().numpy().astype(np.float32)
    np.savez(d / "fp32.npz", **arrs)


def export_onnx(net: str, d: Path, model) -> None:
    import torch
    C = NET_CONFIGS[net]["layers"][0]["IC"]
    dummy = torch.zeros(1, C, 32, 32, dtype=torch.float32)
    torch.onnx.export(model, dummy, str(d / "fp32.onnx"), opset_version=OPSET,
                      input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "N"}, "logits": {0: "N"}},
                      do_constant_folding=True)


def calib_images(net: str) -> np.ndarray:
    """Project INT8 calibration set: legacy make_transform, train[0:1024] (both nets:
    python/eval_lenet5_int8.py, v2/model/final_layer.calibrated_params, D3/D9)."""
    tr = dataset(net, train=True)
    return np.stack([tr[i][0].numpy() for i in range(CALIB_N)]).astype(np.float32)


def export_ort_int8(d: Path, calib: np.ndarray) -> dict:
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                          QuantFormat, QuantType, quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"x": calib[i:i + 64]} for i in range(0, len(calib), 64)])

        def get_next(self):
            return next(self.it, None)

    src = d / "fp32.onnx"
    pre = d / "fp32_preproc.onnx"
    preproc = "none"
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(src), str(pre))
        src, preproc = pre, "quant_pre_process"
    except Exception as e:  # recorded, not fatal
        preproc = f"skipped ({type(e).__name__}: {e})"
    opts = {"ActivationSymmetric": True, "WeightSymmetric": True}
    quantize_static(str(src), str(d / "int8_qdq.onnx"), Reader(),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, extra_options=opts)
    if pre.exists():
        pre.unlink()
    return {"method": "onnxruntime.quantization.quantize_static", "quant_format": "QDQ",
            "activation_type": "QInt8", "weight_type": "QInt8", "per_channel": True,
            "calibrate_method": "MinMax", "extra_options": opts, "pre_process": preproc,
            "calibration_images": f"train[0:{CALIB_N}] legacy make_transform (float32 [0,1])",
            "calibration_batch": 64}


# --------------------------------------------------------------------------- #
def verify(net: str, d: Path, x_u8: np.ndarray, y: np.ndarray, model) -> dict:
    import torch
    out = {}
    te = dataset(net, train=False)
    x_leg = np.stack([te[i][0].numpy() for i in range(TEST_N)]).astype(np.float32)
    x_np = cpu_infer.preprocess_fp32(net, x_u8[:TEST_N])
    out["preprocess_fp32_bitexact_vs_legacy_transform"] = int(
        np.all(x_np.view(np.uint32) == x_leg.view(np.uint32), axis=(1, 2, 3)).sum())
    G = gos_golden.load_net(net)
    q_gold = gos_golden.quantize_input(G, x_leg)
    r8 = cpu_infer.Int8RefRunner(net, d.parent.parent, threads=0)
    q_np = cpu_infer.quantize_input(x_np, r8.S_input)
    out["quantize_input_equal_vs_golden"] = int(np.all(q_np == q_gold, axis=(1, 2, 3)).sum())

    # int8_ref vs gos_golden on all 10k
    t0 = time.perf_counter()
    v_eq = l_eq = p_eq = correct8 = 0
    for s in range(0, TEST_N, 500):
        g = gos_golden.run_net(G, q_gold[s:s + 500])
        v = r8.raw_v(q_np[s:s + 500])
        lg = r8.logits_from_raw(v)
        v_eq += int(np.all(v == g["v"], axis=1).sum())
        l_eq += int(np.all(lg.view(np.uint32) == g["logits"].view(np.uint32), axis=1).sum())
        p_eq += int((lg.argmax(1) == g["pred"]).sum())
        correct8 += int((lg.argmax(1) == y[s:s + 500]).sum())
    out["int8_ref_raw_v_bitexact_vs_gos_golden"] = v_eq
    out["int8_ref_logits_bitexact_vs_gos_golden"] = l_eq
    out["int8_ref_pred_equal_vs_gos_golden"] = p_eq
    out["int8_ref_correct"] = correct8
    out["int8_ref_verify_s"] = round(time.perf_counter() - t0, 2)

    # FP32 numpy (batch 1, as in timing) and torch
    f = cpu_infer.Fp32NumpyRunner(net, d.parent.parent, threads=0)
    pred_np = np.array([int(f.compute(x_np[i]).argmax()) for i in range(TEST_N)])
    out["fp32_numpy_correct"] = int((pred_np == y).sum())
    with torch.no_grad():
        lt = model(torch.from_numpy(x_leg)).numpy()
    out["torch_fp32_correct"] = int((lt.argmax(1) == y).sum())
    out["fp32_numpy_pred_equal_vs_torch"] = int((pred_np == lt.argmax(1)).sum())
    lnp = f.compute(x_np)
    out["fp32_numpy_max_abs_logit_diff_vs_torch"] = float(np.abs(lnp - lt).max())

    # ORT
    for kind in ("cpu_ort_fp32", "cpu_ort_int8"):
        r = cpu_infer.OrtRunner(kind, net, d.parent.parent, threads=1)
        pr = np.array([int(r.compute(x_np[i]).argmax()) for i in range(TEST_N)])
        out[f"{kind}_correct"] = int((pr == y).sum())
        if kind == "cpu_ort_fp32":
            out["ort_fp32_max_abs_logit_diff_vs_torch"] = float(np.abs(r.compute(x_np) - lt).max())
        else:
            out["ort_int8_pred_equal_vs_int8_ref"] = None  # filled below
            out["_ort_int8_pred"] = pr
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "v2/board/data"))
    ap.add_argument("--nets", nargs="+", default=list(cpu_infer.NETS))
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    gs = git_state()
    if gs["git_dirty"] and not a.allow_dirty:
        print("refusing: git tree is dirty (use --allow-dirty; the manifest records it)",
              file=sys.stderr)
        return 2
    import onnx
    import onnxruntime
    import torch
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "torch": torch.__version__, "onnx": onnx.__version__,
                "onnxruntime": onnxruntime.__version__, "onnx_opset": OPSET}
    ok = True
    for net in a.nets:
        t0 = time.perf_counter()
        d = Path(a.out) / net / "cpu"
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitignore").write_text("# generated CPU data package (export_cpu_models.py)\n*\n")
        model = load_torch_model(net)
        x_u8, y = raw_u8(dataset(net, train=False, transform=False))
        np.savez(d / "raw_test.npz", x_u8=x_u8, y=y)
        export_int8_ref(net, d)
        export_fp32(net, d, model)
        export_onnx(net, d, model)
        qinfo = export_ort_int8(d, calib_images(net))
        man = {
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "net": net, **gs, "versions": versions,
            "reference_version": NET_CONFIGS[net]["reference_version"],
            "sources": {
                "checkpoint": NET_CONFIGS[net]["checkpoint"],
                "checkpoint_sha256": sha256(REPO_ROOT / NET_CONFIGS[net]["checkpoint"]),
                "quant_params": str(Path(NET_CONFIGS[net]["quant_params"]).relative_to(REPO_ROOT)),
                "quant_params_sha256": sha256(NET_CONFIGS[net]["quant_params"]),
                "hw_requant": str(Path(NET_CONFIGS[net]["hw_requant"]).relative_to(REPO_ROOT)),
                "hw_requant_sha256": sha256(NET_CONFIGS[net]["hw_requant"]),
                "dataset": NET_CONFIGS[net]["dataset"] + " test (torchvision, data/raw)",
            },
            "preprocessing": ("uint8/255 float32, zero pad 2 -> 1x32x32" if net == "lenet5"
                              else "HWC uint8 -> CHW float32 /255") + " (legacy make_transform); "
                             "INT8 input = clip(rint(float64(x)/S_input), -128, 127)",
            "ort_int8_quantization": qinfo,
            "files": {},
        }
        if not a.no_verify:
            v = verify(net, d, x_u8, y, model)
            pr_ort8 = v.pop("_ort_int8_pred")
            r8 = cpu_infer.Int8RefRunner(net, d.parent.parent, 0)
            p8 = r8.logits_from_raw(r8.raw_v(r8.preprocess(x_u8))).argmax(1)
            v["ort_int8_pred_equal_vs_int8_ref"] = int((pr_ort8 == p8).sum())
            v["n_images"] = TEST_N
            v["label"] = "laptop verification (not paper data)"
            man["verification"] = v
            print(json.dumps({net: v}, indent=1))
            ok &= (v["preprocess_fp32_bitexact_vs_legacy_transform"] == TEST_N
                   and v["quantize_input_equal_vs_golden"] == TEST_N
                   and v["int8_ref_raw_v_bitexact_vs_gos_golden"] == TEST_N
                   and v["int8_ref_logits_bitexact_vs_gos_golden"] == TEST_N)
        for f in sorted(d.iterdir()):
            if f.name not in ("CPU_MANIFEST.json", ".gitignore"):
                man["files"][f.name] = {"sha256": sha256(f), "bytes": f.stat().st_size}
        (d / "CPU_MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
        print(f"{net}: wrote {d} ({time.perf_counter() - t0:.1f} s)")
    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
