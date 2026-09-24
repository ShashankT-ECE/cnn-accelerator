#!/usr/bin/env python3
"""Step 6.3 runner: calibrate the CIFAR-10 INT8 model and measure accuracy.

Run from the repository root:  .venv/bin/python python/eval_cifar10_int8.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from torchvision import datasets  # noqa: E402

from cifar10.int8_model import Int8Cifar10Net, calibrate, load_weights  # noqa: E402
from cifar10.preprocess import make_transform  # noqa: E402

CALIB_N = 1024
TEST_N = 10000
BATCH = 256


def main() -> int:
    ckpt = REPO_ROOT / "data" / "checkpoint" / "cifar10_fp32.pt"
    tr = make_transform()
    train_ds = datasets.CIFAR10(root=str(REPO_ROOT / "data" / "raw"), train=True,
                                download=False, transform=tr)
    test_ds = datasets.CIFAR10(root=str(REPO_ROOT / "data" / "raw"), train=False,
                               download=False, transform=tr)

    print(f"Loading calibration set (train[0:{CALIB_N}])...")
    calib = np.stack([train_ds[i][0].numpy() for i in range(CALIB_N)]).astype(np.float32)
    print("Calibrating per-layer scales...")
    w = load_weights(ckpt)
    params = calibrate(w, calib)
    model = Int8Cifar10Net(params)

    print("Evaluating INT8 top-1 (full test set)...")
    correct = 0
    for s in range(0, TEST_N, BATCH):
        idx = list(range(s, min(s + BATCH, TEST_N)))
        x = np.stack([test_ds[i][0].numpy() for i in idx]).astype(np.float32)
        y = np.array([test_ds[i][1] for i in idx])
        correct += int((model.forward(x).argmax(1) == y).sum())
    acc = correct / TEST_N

    fp32 = 0.6587
    print("=" * 72)
    print("STEP 6.3 — CIFAR-10 INT8 accuracy (calibrated, full test set)")
    print("=" * 72)
    print(f"  INT8 top-1      : {acc*100:.2f}%  ({correct}/{TEST_N})")
    print(f"  FP32 top-1      : {fp32*100:.2f}%  (frozen)")
    print(f"  Δtop-1          : {(acc - fp32)*100:+.2f} pp")
    print(f"  S_input         : {params['S_input']!r}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
