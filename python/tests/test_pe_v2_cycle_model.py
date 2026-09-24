#!/usr/bin/env python3
"""Self-checks for the Step 6.1 PE-v2 architecture study model.

Verifies the recommendation, the option capability matrix, the frozen baseline,
and that L1/L2 artifacts are unchanged.

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
    print("Step 6.1 PE-v2 architecture study self-checks")
    r = json.loads((REPO_ROOT / "data" / "benchmark" / "pe_v2_architecture_results.json").read_text())

    b = r["frozen_baseline"]
    check("baseline OS == 43021", b["OS_dense"] == 43021)
    check("baseline WS == 140412", b["WS_dense"] == 140412)
    check("baseline coarse == 39132", b["coarse_sparse"] == 39132)

    O = r["options"]
    # A/B/C support OS+WS+reconfig but NOT fine-sparsity cycle savings.
    for k in ("A_keep_current_pe", "B_modified_zero_aware_mac", "C_mask_routing"):
        s = O[k]["supports"]
        check(f"{k}: OS+WS+reconfig, no sparse cycles",
              s["OS"] and s["WS"] and s["reconfig"] and not s["fine_sparsity_cycles"])
        check(f"{k}: no cycle change (fine == dense)", O[k]["fine_sparse_cycles"] == 43021)
    # D supports sparse cycles but abandons OS/WS.
    s = O["D_compacted_sparse"]["supports"]
    check("D: sparse cycles, no OS/WS", s["fine_sparsity_cycles"] and not s["OS"] and not s["WS"])

    check("recommendation == KEEP CURRENT PE", r["recommendation"] == "KEEP CURRENT PE")
    check("rationale states incompatibility", "incompatible" in r["rationale"])

    l1 = hashlib.sha256((REPO_ROOT / "data/checkpoint/lenet5_fp32.pt").read_bytes()).hexdigest()
    qp = hashlib.sha256((REPO_ROOT / "data/lenet5_int8/quant_params.npz").read_bytes()).hexdigest()
    check("L1 checkpoint SHA unchanged", l1 == "9978676be5132f25ba2bbdf93cb3930541bbbc780ecffd49c5560ad03c1af9dd")
    check("L2 quant_params SHA unchanged", qp == "3d39eb008d0e20f934c3d70288d6142d50a55e933129aa285430a06407a6ff84")

    print("  ALL STEP 6.1 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
