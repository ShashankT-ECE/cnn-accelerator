#!/usr/bin/env python3
"""Step 1B runner: train the L1 FP32 LeNet-5 reference and save the checkpoint.

Run from the repository root:

    .venv/bin/python python/train_lenet5.py

Reproducible: fixed seed, single-threaded deterministic CPU training, explicit
weight init, fixed 50k/10k/10k data split, SHA-256-pinned checkpoint. No
quantization, no RTL, no OS/WS mapping.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from lenet5 import config as C  # noqa: E402
from lenet5.train import train_model, save_checkpoint  # noqa: E402


def main() -> int:
    print("=" * 72)
    print("STEP 1B — L1 FP32 LeNet-5 reference (docs/LENET5_SPEC.md)")
    print("=" * 72)
    model, metrics = train_model()

    ckpt_path = REPO_ROOT / C.CKPT_DIR / C.CKPT_FILE
    meta_path = REPO_ROOT / C.CKPT_DIR / C.META_FILE
    ckpt_path, ckpt_sha = save_checkpoint(model, metrics, ckpt_path, meta_path)

    print("-" * 72)
    print(f"  Final val acc  : {metrics['val_acc']*100:.4f}%")
    print(f"  Final test acc : {metrics['test_acc']*100:.4f}%")
    print(f"  Checkpoint     : {ckpt_path}")
    print(f"  SHA-256        : {ckpt_sha}")
    print(f"  Metadata       : {meta_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
