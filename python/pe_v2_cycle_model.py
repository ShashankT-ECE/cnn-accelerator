#!/usr/bin/env python3
"""PE-architecture study model (Step 6.1).

Compares the current PE (pe_v2) against four candidate PE designs on the frozen
LeNet results (OS/WS cycles + sparsity), to answer whether a PE redesign can
support OS + WS + useful fine-grained activation sparsity on the same datapath.

Honest: cycle numbers are the frozen mapping/sparse results; candidate sparsity
gains are theoretical (from Step 5.3) and are not claimed as implemented.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "benchmark" / "pe_v2_architecture_results.json"

# Frozen LeNet results (per image).
OS_DENSE = 43021
WS_DENSE = 140412
COARSE_SPARSE = 39132
FINE_GATING = 43021            # gating: no cycle change (Step 5.3)
FINE_COMPACT_THEOR = 17604     # compaction zero-overhead bound
FINE_COMPACT_EST = 22885       # compaction + 1.3x metadata overhead


def options():
    return {
        "A_keep_current_pe": {
            "OS_cycles": OS_DENSE, "WS_cycles": WS_DENSE,
            "fine_sparse_cycles": FINE_GATING,
            "supports": {"OS": True, "WS": True, "reconfig": True,
                         "fine_sparsity_cycles": False},
            "note": "verified; zero_skip gates product (coarse skip only)",
        },
        "B_modified_zero_aware_mac": {
            "OS_cycles": OS_DENSE, "WS_cycles": WS_DENSE,
            "fine_sparse_cycles": FINE_GATING,
            "supports": {"OS": True, "WS": True, "reconfig": True,
                         "fine_sparsity_cycles": False},
            "note": "+per-PE zero detect; energy-only, ~0 on DSP48E2, no cycle gain",
        },
        "C_mask_routing": {
            "OS_cycles": OS_DENSE, "WS_cycles": WS_DENSE,
            "fine_sparse_cycles": FINE_GATING,
            "supports": {"OS": True, "WS": True, "reconfig": True,
                         "fine_sparsity_cycles": False},
            "note": "mask alongside activations; same as B unless it compacts (breaks WS)",
        },
        "D_compacted_sparse": {
            "OS_cycles": None, "WS_cycles": None,
            "fine_sparse_cycles": FINE_COMPACT_EST,
            "supports": {"OS": False, "WS": False, "reconfig": False,
                         "fine_sparsity_cycles": True},
            "note": f"SCNN-style crossbar; sparse {OS_DENSE/FINE_COMPACT_THEOR:.2f}x (theor) "
                    f"/ {OS_DENSE/FINE_COMPACT_EST:.2f}x (est); abandons systolic OS/WS",
        },
    }


def main() -> int:
    results = {
        "frozen_baseline": {
            "OS_dense": OS_DENSE, "WS_dense": WS_DENSE,
            "coarse_sparse": COARSE_SPARSE, "fine_gating_cycles": FINE_GATING,
        },
        "options": options(),
        "recommendation": "KEEP CURRENT PE",
        "rationale": (
            "The PE already supports OS+WS+reconfiguration (the project's actual "
            "contribution). Fine-grained activation sparsity requires compaction "
            "(data-dependent timing), which is incompatible with the fixed-timing "
            "systolic OS/WS datapaths; options B/C add no cycle savings (and ~0 "
            "energy on DSP48E2), and option D abandons OS/WS for a different "
            "accelerator. No PE modification produces a meaningful OS+WS+sparsity "
            "win on the same datapath."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    print("=" * 84)
    print("PE-v2 architecture study — frozen LeNet results (per image)")
    print("=" * 84)
    print(f"  baseline: OS dense {OS_DENSE}, WS dense {WS_DENSE}, coarse sparse {COARSE_SPARSE}")
    print()
    hdr = f"{'option':26} {'OS':>6} {'WS':>6} {'fine':>6} {'OS':>4} {'WS':>4} {'sparse-cyc':>9}"
    print(hdr)
    print("-" * 84)
    for k, o in results["options"].items():
        os_ = o["OS_cycles"] if o["OS_cycles"] is not None else "n/a"
        ws_ = o["WS_cycles"] if o["WS_cycles"] is not None else "n/a"
        fine = o["fine_sparse_cycles"]
        s = o["supports"]
        print(f"{k:26} {str(os_):>6} {str(ws_):>6} {fine:>6} "
              f"{'Y' if s['OS'] else 'N':>4} {'Y' if s['WS'] else 'N':>4} "
              f"{'Y' if s['fine_sparsity_cycles'] else 'N':>9}")
    print("=" * 84)
    print(f"Recommendation: {results['recommendation']}")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
