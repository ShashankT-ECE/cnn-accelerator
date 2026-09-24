#!/usr/bin/env python3
"""Self-checks for the Step 6.3 CIFAR-10 INT8 reference.

Verifies layer shapes, int8 dtypes, int32 accumulator safety, and INT8-vs-FP32
argmax agreement on a sample.

Plain asserts; no pytest dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

import torch  # noqa: E402
from torchvision import datasets  # noqa: E402

from cifar10.int8_model import Int8Cifar10Net, calibrate, load_weights  # noqa: E402
from cifar10.preprocess import make_transform  # noqa: E402
from cifar10.train import load_checkpoint  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def main() -> int:
    print("Step 6.3 CIFAR-10 INT8 reference self-checks")
    ckpt = REPO_ROOT / "data" / "checkpoint" / "cifar10_fp32.pt"
    tr = make_transform()
    train_ds = datasets.CIFAR10(root=str(REPO_ROOT / "data" / "raw"), train=True,
                                download=False, transform=tr)
    calib = np.stack([train_ds[i][0].numpy() for i in range(256)]).astype(np.float32)
    w = load_weights(ckpt)
    params = calibrate(w, calib)
    model = Int8Cifar10Net(params)

    test_ds = datasets.CIFAR10(root=str(REPO_ROOT / "data" / "raw"), train=False,
                               download=False, transform=tr)
    N = 8
    xf = np.stack([test_ds[i][0].numpy() for i in range(N)]).astype(np.float32)
    ys = np.array([test_ds[i][1] for i in range(N)])
    out = model.forward_layers(xf)

    shapes = {"q_input": (N, 3, 32, 32), "pool1": (N, 32, 14, 14), "pool2": (N, 32, 5, 5),
              "c3": (N, 64, 1, 1), "logits": (N, 10)}
    for k, sh in shapes.items():
        check(f"shape {k}", out[k].shape == sh, str(out[k].shape))
    for k in ("q_input", "pool1", "pool2", "c3"):
        check(f"dtype {k} int8", out[k].dtype == np.int8, str(out[k].dtype))
    for k in ("c1_acc", "c2_acc", "c3_acc", "fc_acc"):
        acc = out[k]
        check(f"{k} fits int32", int(acc.min()) >= -2**31 and int(acc.max()) <= 2**31 - 1,
              f"[{int(acc.min())},{int(acc.max())}]")

    # INT8 vs FP32 argmax agreement on a 256-image sample.
    l1 = load_checkpoint(ckpt)
    xs = torch.stack([test_ds[i][0] for i in range(256)])
    with torch.no_grad():
        l1_logits = l1(xs).numpy()
    xf256 = np.stack([test_ds[i][0].numpy() for i in range(256)]).astype(np.float32)
    l2_logits = model.forward(xf256)
    agree = int((l1_logits.argmax(1) == l2_logits.argmax(1)).sum())
    check("INT8 vs FP32 argmax agreement >= 90%", agree >= int(256 * 0.90), f"{agree}/256")

    print("  ALL STEP 6.3 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
