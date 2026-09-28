#!/usr/bin/env python3
"""DPU baseline step 1 (laptop, repo .venv): export the exact FP32 inputs for Vitis AI.

    .venv/bin/python v2/dpu/export_data.py [--nets lenet5 cifar10]

For each net (v2/model/net_config.NET_CONFIGS, DECISIONS D3: lenet5 = legacy checkpoint
data/checkpoint/lenet5_fp32.pt, cifar10 = r2 v2/model/retrain/cifar10_fp32_r2.pt):

  1. FP32 accuracy with the legacy loaders, exactly as v2/model/reference_accuracy.py
     (train.load_checkpoint + train.evaluate on preprocess.fixed_split_loaders test loader)
     -> must equal the accuracy of record (v2/results/reference_accuracy.csv, FP32 row of the
     net's reference_version); the script stops otherwise.
  2. Writes v2/dpu/build/<net>/dpu_inputs.npz (gitignored):
        x_test  float32 [10000,C,32,32]  test[0:10000], legacy preprocessing
                (MNIST ToTensor + zero-pad 2; CIFAR ToTensor, no mean/std)
        y_test  int64   [10000]
        x_calib float32 [1024,C,32,32]   train[0:1024], same transform (the calibration subset
                used by the legacy INT8 calibrate path for both nets)
     and copies the checkpoint state_dict to v2/dpu/build/<net>/fp32_state_dict.pt.
  3. Re-evaluates the FP32 model on the exported x_test (same count required) and, if the
     board data package v2/board/data/<net>/inputs_act.npz exists, checks its x_f32 is
     bit-identical to x_test (the board session feeds the DPU from the package's x_f32).
  4. Writes v2/dpu/build/<net>/export_info.json (sha256s, counts, git state).

Nothing here quantizes. Label of the accuracy: model, FP32.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DPU = Path(__file__).resolve().parent
V2 = DPU.parent
REPO = V2.parent
BUILD = DPU / "build"
sys.path.insert(0, str(V2 / "model"))

import common  # noqa: E402,F401  (puts legacy python/ on sys.path)
from net_config import NET_CONFIGS, NETS  # noqa: E402

import torch  # noqa: E402

CALIB_N = 1024
TEST_N = 10000


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_state() -> dict:
    g = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,  # noqa: E731
                                  text=True, check=True).stdout.strip()
    return {"git_commit": g("rev-parse", "HEAD"),
            "git_dirty": bool(g("status", "--porcelain", "--", ".", ":!v2/results",
                                ":!v2/vectors/MANIFEST.json"))}


def record_fp32(net: str) -> int:
    ver = NET_CONFIGS[net]["reference_version"]
    with (V2 / "results" / "reference_accuracy.csv").open() as f:
        rows = [r for r in csv.DictReader(f)
                if r["reference_version"] == ver and r["precision"] == "FP32"]
    assert len(rows) == 1, rows
    return int(rows[0]["correct"])


def legacy(net: str):
    if net == "lenet5":
        from lenet5 import preprocess, train
    else:
        from cifar10 import preprocess, train
    return preprocess, train


def export(net: str) -> dict:
    t0 = time.perf_counter()
    cfg = NET_CONFIGS[net]
    preprocess, train = legacy(net)
    ckpt = REPO / cfg["checkpoint"]
    model = train.load_checkpoint(ckpt, device="cpu")
    train_ds, test_ds = preprocess.build_datasets()
    _, _, test_loader = preprocess.fixed_split_loaders(train_ds, test_ds)
    acc, _ = train.evaluate(model, test_loader, "cpu")
    total = len(test_loader.dataset)
    correct = int(round(acc * total))
    rec = record_fp32(net)
    print(f"[{net}] FP32 legacy evaluate: {correct}/{total} (record {rec})", flush=True)
    if total != TEST_N or correct != rec:
        raise SystemExit(f"[{net}] FP32 accuracy {correct}/{total} != record {rec}/{TEST_N}")

    x_test = np.stack([test_ds[i][0].numpy() for i in range(TEST_N)]).astype(np.float32)
    y_test = np.array([test_ds[i][1] for i in range(TEST_N)], dtype=np.int64)
    x_cal = np.stack([train_ds[i][0].numpy() for i in range(CALIB_N)]).astype(np.float32)

    with torch.no_grad():
        pred = np.concatenate([model(torch.from_numpy(x_test[s:s + 500])).argmax(1).numpy()
                               for s in range(0, TEST_N, 500)])
    c2 = int((pred == y_test).sum())
    print(f"[{net}] FP32 on exported x_test: {c2}/{TEST_N}", flush=True)
    if c2 != rec:
        raise SystemExit(f"[{net}] exported-tensor FP32 accuracy {c2} != record {rec}")

    pkg_check = "package not present"
    pkg = V2 / "board" / "data" / net / "inputs_act.npz"
    if pkg.is_file():
        xf = np.load(pkg)["x_f32"]
        same = xf.shape == x_test.shape and np.array_equal(xf, x_test)
        pkg_check = f"x_f32 identical: {same}"
        if not same:
            raise SystemExit(f"[{net}] board package x_f32 != exported x_test")
    print(f"[{net}] board data package check: {pkg_check}")

    out = BUILD / net
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "dpu_inputs.npz"
    np.savez(npz, x_test=x_test, y_test=y_test, x_calib=x_cal)
    sd = out / "fp32_state_dict.pt"
    shutil.copyfile(ckpt, sd)
    info = {"net": net, "reference_version": cfg["reference_version"],
            "checkpoint": cfg["checkpoint"], "checkpoint_sha256": sha256(ckpt),
            "dataset": cfg["dataset"],
            "preprocess": ("MNIST ToTensor [0,1] + zero-pad 2 -> 32x32" if net == "lenet5"
                           else "CIFAR-10 ToTensor [0,1], no mean/std"),
            "test_set": f"test[0:{TEST_N}]", "calib_set": f"train[0:{CALIB_N}]",
            "fp32_correct": correct, "fp32_total": total, "fp32_record_correct": rec,
            "fp32_exported_correct": c2, "board_package_check": pkg_check,
            "dpu_inputs_sha256": sha256(npz), "torch_version": torch.__version__,
            "duration_s": round(time.perf_counter() - t0, 3), **git_state(),
            "label": "model, FP32"}
    (out / "export_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"[{net}] wrote {npz} ({npz.stat().st_size} B), {sd}, export_info.json")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nets", nargs="+", default=list(NETS), choices=NETS)
    a = ap.parse_args(argv)
    for n in a.nets:
        export(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
