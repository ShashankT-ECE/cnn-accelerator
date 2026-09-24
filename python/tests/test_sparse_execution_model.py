#!/usr/bin/env python3
"""Self-checks for the Step 5.2 sparse execution model.

Verifies the dense baseline is unchanged, coarse sparse cycles never exceed
dense, the coarse skipped-MAC count stays under the per-MAC bound (246,081),
layer totals sum correctly, L1/L2 artifacts are unchanged, and the result is
deterministic.

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


def main() -> int:
    print("Step 5.2 sparse execution model self-checks")
    r = json.loads((REPO_ROOT / "data" / "benchmark" / "sparse_execution_results.json").read_text())
    C = r["coarse"]
    D = r["dense"]
    T = r["totals"]

    # Dense baseline unchanged (frozen Step 4.3).
    check("dense total == 43021", T["dense_cycles"] == 43021, str(T["dense_cycles"]))
    check("dense layer sum == total", sum(D[n] for n in D) == T["dense_cycles"])

    # Sparse never exceeds dense (per layer + total).
    for n in C:
        check(f"{n}: sparse <= dense", C[n]["sparse_cycles"] <= C[n]["dense_cycles"],
              f"{C[n]['sparse_cycles']} <= {C[n]['dense_cycles']}")
    check("sparse total <= dense total", T["coarse_sparse_cycles"] <= T["dense_cycles"])

    # Key honest values.
    check("conv1 skip rate ~42.28% (structural)", abs(C["conv1"]["skip_rate_pct"] - 42.28) < 0.05)
    check("conv3/conv5/fc skip ~0 (ReLU too fine for group skip)",
          C["conv3"]["skip_rate_pct"] < 0.1 and C["conv5"]["skip_rate_pct"] < 0.1
          and C["fc1"]["skip_rate_pct"] < 0.1 and C["fc2"]["skip_rate_pct"] < 0.1)
    check("coarse speedup == 1.0994x (honest, ~10%)", abs(T["coarse_speedup_vs_dense"] - 1.0994) < 0.0005,
          f"{T['coarse_speedup_vs_dense']:.4f}x")

    # Coarse skipped MACs never exceed the per-MAC bound (246,081/image).
    check("coarse skipped MACs <= 246081", T["coarse_skipped_macs_per_image"] <= 246081,
          f"{T['coarse_skipped_macs_per_image']:.0f}")

    # Fine is an upper bound only (no cycle claim).
    check("fine reduction ~59.08% (upper bound)", abs(T["fine_theoretical_mac_reduction_pct"] - 59.08) < 0.05)
    check("no fine cycle speedup claimed", "no cycle speedup" in T["fine_note"])

    # L1/L2 artifacts unchanged.
    l1 = hashlib.sha256((REPO_ROOT / "data/checkpoint/lenet5_fp32.pt").read_bytes()).hexdigest()
    qp = hashlib.sha256((REPO_ROOT / "data/lenet5_int8/quant_params.npz").read_bytes()).hexdigest()
    check("L1 checkpoint SHA unchanged", l1 == "9978676be5132f25ba2bbdf93cb3930541bbbc780ecffd49c5560ad03c1af9dd")
    check("L2 quant_params SHA unchanged", qp == "3d39eb008d0e20f934c3d70288d6142d50a55e933129aa285430a06407a6ff84")

    # Determinism: two runs of the conv1 group-window check on a fixed subset.
    qp_arr = np.load(REPO_ROOT / "data" / "lenet5_int8" / "quant_params.npz")
    params = {"S_input": float(qp_arr["S_input"])}
    for n in ["conv1", "conv3", "conv5", "fc1", "fc2"]:
        L = {"S_a": float(qp_arr[f"{n}_S_a"]), "S_w": qp_arr[f"{n}_S_w"],
             "q_w": qp_arr[f"{n}_q_w"], "q_b": qp_arr[f"{n}_q_b"]}
        if f"{n}_S_out" in qp_arr.files:
            L["S_out"] = float(qp_arr[f"{n}_S_out"]); L["M"] = qp_arr[f"{n}_M"]
        params[n] = L
    model = Int8LeNet5(params)
    tr = make_transform()
    test_ds = datasets.MNIST(root=str(REPO_ROOT / "data" / "raw"), train=False,
                             download=False, transform=tr)
    def skip_count():
        x = np.stack([test_ds[i][0].numpy() for i in range(32)]).astype(np.float32)
        out = model.forward_layers(x)
        a = out["q_input"]
        c = 0
        for y in range(28):
            for left in range(0, 28, 8):
                right = min(left + 7, 27)
                c += int((a[:, :, y:y + 5, left:right + 5] == 0).all(axis=(1, 2, 3)).sum())
        return c
    a, b = skip_count(), skip_count()
    check("deterministic group skip (two runs identical)", a == b, str(a))

    print("  ALL STEP 5.2 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
