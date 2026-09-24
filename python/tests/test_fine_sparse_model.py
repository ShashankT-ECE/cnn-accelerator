#!/usr/bin/env python3
"""Self-checks for the Step 5.3 fine-grained sparsity model.

Verifies the dense/coarse baselines are unchanged, gating gives no cycle speedup,
compaction is flagged impractical, and the L1/L2 artifacts are unchanged.

Plain asserts; no pytest dependency.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def main() -> int:
    print("Step 5.3 fine-grained sparsity model self-checks")
    r = json.loads((REPO_ROOT / "data" / "benchmark" / "fine_sparse_results.json").read_text())

    check("dense == 43021", r["dense_cycles"] == 43021)
    check("coarse == 39132", r["coarse_cycles"] == 39132)
    check("fine gating == 43021 (no cycle speedup)", r["fine_gating_cycles"] == 43021)
    check("fine gating speedup == 1.0", abs(r["fine_gating_speedup"] - 1.0) < 1e-9)
    check("compaction nonzero MACs == 170439",
          r["fine_compaction"]["nonzero_macs_per_image"] == 170439)
    check("compaction theoretical < coarse (finer granularity wins only if free)",
          r["fine_compaction"]["theoretical_cycles_zero_overhead"] < r["coarse_cycles"])
    check("compaction flagged impractical", "IMPRACTICAL" in r["fine_compaction"]["practicality"])
    check("conclusion: no practical cycle mechanism",
          "No practical fine-grained mechanism" in r["conclusion"])

    l1 = hashlib.sha256((REPO_ROOT / "data/checkpoint/lenet5_fp32.pt").read_bytes()).hexdigest()
    qp = hashlib.sha256((REPO_ROOT / "data/lenet5_int8/quant_params.npz").read_bytes()).hexdigest()
    check("L1 checkpoint SHA unchanged", l1 == "9978676be5132f25ba2bbdf93cb3930541bbbc780ecffd49c5560ad03c1af9dd")
    check("L2 quant_params SHA unchanged", qp == "3d39eb008d0e20f934c3d70288d6142d50a55e933129aa285430a06407a6ff84")

    print("  ALL STEP 5.3 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
