#!/usr/bin/env python3
"""Step 2.2 runner: calibrate the L2 INT8 model and measure end-to-end accuracy.

Run from the repository root:

    .venv/bin/python python/eval_lenet5_int8.py

Calibrates per-layer scales on train[0:1024] (per docs/LENET5_INT8_SPEC.md §13),
then measures INT8 top-1 on the full 10k test set and reports Δtop-1 vs the
frozen L1 FP32 accuracy (98.78%). No RTL, no OS/WS, no sparsity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from torchvision import datasets  # noqa: E402

from lenet5.int8_model import Int8LeNet5, calibrate, load_weights  # noqa: E402
from lenet5.preprocess import make_transform  # noqa: E402

CALIB_N = 1024
TEST_N = 10000
BATCH = 512


def main() -> int:
    ckpt = REPO_ROOT / "data/checkpoint/lenet5_fp32.pt"
    tr = make_transform()
    train_ds = datasets.MNIST(root=str(REPO_ROOT / "data/raw"), train=True,
                              download=False, transform=tr)
    test_ds = datasets.MNIST(root=str(REPO_ROOT / "data/raw"), train=False,
                             download=False, transform=tr)

    print(f"Loading calibration set (train[0:{CALIB_N}])...")
    calib = np.stack([train_ds[i][0].numpy() for i in range(CALIB_N)]).astype(np.float32)

    print("Calibrating per-layer scales...")
    w = load_weights(ckpt)
    params = calibrate(w, calib)
    model = Int8LeNet5(params)

    print("Evaluating INT8 top-1 (full test set)...")
    correct = 0
    for s in range(0, TEST_N, BATCH):
        x = np.stack([test_ds[i][0].numpy() for i in range(s, min(s + BATCH, TEST_N))])
        x = x.astype(np.float32)
        y = np.array([test_ds[i][1] for i in range(s, min(s + BATCH, TEST_N))])
        pred = model.forward(x).argmax(1)
        correct += int((pred == y).sum())
    acc = correct / TEST_N

    l1_acc = 0.9878
    print("=" * 72)
    print("STEP 2.2 — L2 INT8 LeNet-5 accuracy (calibrated, full test set)")
    print("=" * 72)
    print(f"  INT8 top-1      : {acc*100:.4f}%  ({correct}/{TEST_N})")
    print(f"  L1 FP32 top-1   : {l1_acc*100:.4f}%  (frozen)")
    print(f"  Δtop-1          : {(acc - l1_acc)*100:+.4f} pp")
    print(f"  S_input         : {params['S_input']!r}")
    for k in ("conv1", "conv3", "conv5", "fc1"):
        print(f"  S_out[{k:5s}]    : {params[k]['S_out']:.6f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
