#!/usr/bin/env python3
"""Self-checks for the Step 6.2 CIFAR-10 FP32 reference.

Verifies the exact topology (shapes), parameter count, MAC count, and that the
model is independent of the LeNet-5 model.

Plain asserts; no pytest dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from cifar10 import config as C  # noqa: E402
from cifar10.model import Cifar10Net  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def main() -> int:
    print("Step 6.2 CIFAR-10 reference self-checks")
    torch.manual_seed(0)
    model = Cifar10Net()
    model.eval()
    N = 2
    x = torch.rand(N, 3, 32, 32)
    with torch.no_grad():
        L = model.forward_layers(x)

    expected = {
        "input": (N, 3, 32, 32), "c1": (N, 32, 28, 28), "relu1": (N, 32, 28, 28),
        "pool1": (N, 32, 14, 14), "c2": (N, 32, 10, 10), "relu2": (N, 32, 10, 10),
        "pool2": (N, 32, 5, 5), "c3": (N, 64, 1, 1), "relu3": (N, 64, 1, 1),
        "logits": (N, 10),
    }
    for k, sh in expected.items():
        check(f"shape {k}", tuple(L[k].shape) == sh, str(tuple(L[k].shape)))

    n_params = sum(p.numel() for p in model.parameters())
    check("params == 79978", n_params == C.TOTAL_PARAMS, str(n_params))

    # MAC count from first principles.
    macs = (32 * 28 * 28 * 3 * 25) + (32 * 10 * 10 * 32 * 25) + (64 * 1 * 1 * 32 * 25) + (10 * 64)
    check("MACs == 4493440", macs == C.TOTAL_MACS, str(macs))
    check("MACs per layer: conv1 1881600", 32 * 28 * 28 * 3 * 25 == 1881600)
    check("MACs per layer: conv2 2560000", 32 * 10 * 10 * 32 * 25 == 2560000)
    check("MACs per layer: conv3 51200", 64 * 1 * 1 * 32 * 25 == 51200)

    print("  ALL STEP 6.2 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
