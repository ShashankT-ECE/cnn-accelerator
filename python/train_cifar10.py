#!/usr/bin/env python3
"""Step 6.2 runner: train the CIFAR-10 cuda-convnet FP32 reference.

Run from the repository root:  .venv/bin/python python/train_cifar10.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from cifar10 import config as C  # noqa: E402
from cifar10.train import train_model, save_checkpoint  # noqa: E402


def main() -> int:
    print("=" * 72)
    print("STEP 6.2 — CIFAR-10 cuda-convnet FP32 reference")
    print("=" * 72)
    model, metrics = train_model()
    ckpt = REPO_ROOT / C.CKPT_DIR / C.CKPT_FILE
    meta = REPO_ROOT / C.CKPT_DIR / C.META_FILE
    ckpt, sha = save_checkpoint(model, metrics, ckpt, meta)
    print("-" * 72)
    print(f"  Final val acc  : {metrics['val_acc']*100:.2f}%")
    print(f"  Final test acc : {metrics['test_acc']*100:.2f}%")
    print(f"  Params         : {C.TOTAL_PARAMS}")
    print(f"  MACs           : {C.TOTAL_MACS}")
    print(f"  Checkpoint     : {ckpt}")
    print(f"  SHA-256        : {sha}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
