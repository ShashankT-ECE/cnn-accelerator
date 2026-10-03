#!/usr/bin/env python3
"""DPU baseline step 1 (laptop, repo .venv): export the exact FP32 inputs + weights for Vitis AI.
Adapted from v2/dpu/export_data.py @ 28dd2ad (V3: nets from v3/model/nets.py, checkpoint by path).

    .venv/bin/python v3/dpu/export_data.py --net resnet20_b --checkpoint v3/train/runs/<job>/<file>.pt
        [--mean M M M --std S S S]                  # override the checkpoint's preprocessing

For the net + checkpoint (v3/board/laptop_common.load_checkpoint; strict state_dict load into
v3/model/nets.build(net)):
  1. FP32 accuracy of the checkpoint on CIFAR-10 test[0:10000] (preprocessing =
     v3/board/baseline_common.preprocess with the checkpoint's spec), label "model, FP32".
     V3 has no accuracy of record yet: --expect-correct N stops the export if it differs
     (use it with the training job's reported count once it exists).
  2. Writes v3/dpu/build/<net>/ (gitignored):
        dpu_inputs.npz   x_test float32 [10000,3,32,32], y_test int64, x_calib float32 [1024,3,32,32]
                         (train[0:1024], the calibration subset also used for ORT INT8)
        fp32_state_dict.pt   the checkpoint's state_dict (BN not folded: vai_q folds it)
        torch_pred.npy       FP32 predictions on x_test (agreement reference for the board session)
        export_info.json     sha256s, counts, checkpoint kind, preprocess spec, git state
Nothing here quantizes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DPU = Path(__file__).resolve().parent
V3 = DPU.parent
BUILD = DPU / "build"
sys.path.insert(0, str(V3 / "board"))

import laptop_common as lc  # noqa: E402
import baseline_common as bc  # noqa: E402


def export(net: str, ckpt: str, pp_override=None, expect_correct=None, build: Path = BUILD) -> dict:
    import torch
    t0 = time.perf_counter()
    model, ci = lc.load_checkpoint(net, ckpt, pp_override)
    x_u8, y = lc.cifar_test()
    xc_u8, _ = lc.cifar_train_head(lc.CALIB_N)
    x_test = bc.preprocess(x_u8, ci["preprocess"])
    x_cal = bc.preprocess(xc_u8, ci["preprocess"])
    _, pred = lc.predict_torch(model, x_test)
    correct = int((pred == y).sum())
    print(f"[{net}] FP32 (model) on test[0:{lc.TEST_N}]: {correct}/{len(y)}  checkpoint {ci['checkpoint']} "
          f"kind={ci['checkpoint_kind']} preprocess={ci['preprocess']['source']}", flush=True)
    if ci["checkpoint_kind"] != bc.CHECKPOINT_TRAINED:
        print(f"[{net}] NOTE: {ci['checkpoint_kind']} checkpoint -> everything downstream is DRY RUN only")
    if expect_correct is not None and correct != expect_correct:
        raise SystemExit(f"[{net}] FP32 {correct} != --expect-correct {expect_correct}")
    out = build / net
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "dpu_inputs.npz"
    np.savez(npz, x_test=x_test, y_test=y, x_calib=x_cal)
    sd = out / "fp32_state_dict.pt"
    torch.save(ci["state_dict"], sd)
    np.save(out / "torch_pred.npy", pred)
    info = {"net": net, "checkpoint": ci["checkpoint"], "checkpoint_sha256": ci["checkpoint_sha256"],
            "checkpoint_kind": ci["checkpoint_kind"], "preprocess": ci["preprocess"],
            "state_dict_sha256": bc.sha256_file(sd), "dataset": "CIFAR-10 (data/raw/cifar-10-batches-py)",
            "test_set": f"test[0:{lc.TEST_N}]", "calib_set": f"train[0:{lc.CALIB_N}]",
            "fp32_correct": correct, "fp32_total": int(len(y)),
            "dpu_inputs_sha256": bc.sha256_file(npz), "torch_pred_sha256": bc.sha256_file(out / "torch_pred.npy"),
            "torch_version": torch.__version__, "duration_s": round(time.perf_counter() - t0, 3),
            **lc.git_state(), "label": "model, FP32"}
    (out / "export_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"[{net}] wrote {npz} ({npz.stat().st_size} B), {sd.name}, torch_pred.npy, export_info.json")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--net", required=True)
    ap.add_argument("--checkpoint", required=True, help="e.g. v3/train/runs/<job>/<file>.pt")
    ap.add_argument("--mean", nargs=3, type=float, default=None)
    ap.add_argument("--std", nargs=3, type=float, default=None)
    ap.add_argument("--expect-correct", type=int, default=None)
    a = ap.parse_args(argv)
    export(a.net, a.checkpoint, lc.parse_preprocess(a.mean, a.std), a.expect_correct)
    return 0


if __name__ == "__main__":
    sys.exit(main())
