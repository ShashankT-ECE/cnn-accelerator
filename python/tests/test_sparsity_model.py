#!/usr/bin/env python3
"""Self-checks for the Step 5.1 sparsity contract.

Verifies the frozen sparsity statistics are mathematically consistent, derived
from the frozen tensor dimensions, deterministic, and that the L1/L2 artifacts
are unchanged.

Plain asserts; no pytest dependency.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from torchvision import datasets  # noqa: E402
from lenet5.int8_model import Int8LeNet5  # noqa: E402
from lenet5.preprocess import make_transform  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def load_model():
    qp = np.load(REPO_ROOT / "data" / "lenet5_int8" / "quant_params.npz")
    params = {"S_input": float(qp["S_input"])}
    for n in ["conv1", "conv3", "conv5", "fc1", "fc2"]:
        L = {"S_a": float(qp[f"{n}_S_a"]), "S_w": qp[f"{n}_S_w"],
             "q_w": qp[f"{n}_q_w"], "q_b": qp[f"{n}_q_b"]}
        if f"{n}_S_out" in qp.files:
            L["S_out"] = float(qp[f"{n}_S_out"]); L["M"] = qp[f"{n}_M"]
        params[n] = L
    return Int8LeNet5(params)


def main() -> int:
    print("Step 5.1 sparsity contract self-checks")
    r = json.loads((REPO_ROOT / "data" / "benchmark" / "sparsity_results.json").read_text())
    L = r["layers"]

    # Frozen key values.
    check("conv1 zero_pct == 85.28 (structural)", abs(L["conv1"]["zero_pct"] - 85.28) < 0.01)
    check("conv3 zero_pct == 66.04 (ReLU)", abs(L["conv3"]["zero_pct"] - 66.04) < 0.01)
    check("conv3 skippable_pct == 49.31 (boundary-weighted)", abs(L["conv3"]["skippable_pct"] - 49.31) < 0.01)
    check("fc1 zero_pct == 59.85", abs(L["fc1"]["zero_pct"] - 59.85) < 0.01)
    check("total theoretical reduction == 59.08%", abs(r["totals"]["theoretical_mac_reduction_pct"] - 59.08) < 0.01)

    # Mathematical consistency: zero + nonzero == total, percentages consistent.
    for name, d in L.items():
        check(f"{name}: zero_pct + nonzero_pct == 100",
              abs(d["zero_pct"] + d["nonzero_pct"] - 100.0) < 1e-9)
        check(f"{name}: zero_pct == zero_count/total",
              abs(d["zero_pct"] - d["zero_count"] / d["total_elements"] * 100) < 1e-9)
        check(f"{name}: skippable <= total MACs",
              d["skippable_macs_per_image"] <= d["total_macs_per_image"])

    # Total MACs unchanged (frozen mapping).
    check("total MACs == 416520", r["totals"]["total_macs_per_image"] == 416520)

    # L1/L2 artifacts unchanged.
    l1 = hashlib.sha256((REPO_ROOT / "data/checkpoint/lenet5_fp32.pt").read_bytes()).hexdigest()
    qp = hashlib.sha256((REPO_ROOT / "data/lenet5_int8/quant_params.npz").read_bytes()).hexdigest()
    check("L1 checkpoint SHA unchanged", l1 == "9978676be5132f25ba2bbdf93cb3930541bbbc780ecffd49c5560ad03c1af9dd")
    check("L2 quant_params SHA unchanged", qp == "3d39eb008d0e20f934c3d70288d6142d50a55e933129aa285430a06407a6ff84")

    # Determinism: two runs on a fixed subset give identical zero counts.
    model = load_model()
    tr = make_transform()
    test_ds = datasets.MNIST(root=str(REPO_ROOT / "data" / "raw"), train=False,
                             download=False, transform=tr)
    def zero_counts():
        x = np.stack([test_ds[i][0].numpy() for i in range(32)]).astype(np.float32)
        out = model.forward_layers(x)
        return tuple(int((out[k] == 0).sum()) for k in ("q_input", "pool1", "pool2", "c5", "f6"))
    a, b = zero_counts(), zero_counts()
    check("deterministic sparsity (two runs identical)", a == b, str(a))

    print("  ALL STEP 5.1 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
