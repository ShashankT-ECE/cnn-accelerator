#!/usr/bin/env python3
"""Fine-grained activation-sparsity analysis model (Step 5.3).

Quantifies three mechanisms against the frozen 10k-image LeNet sparsity, and
models the cycle impact of the most promising one — honestly, without assuming
a speedup.

Mechanisms:
  A. per-PE zero detection (gate the multiplier when activation==0)
  B. zero-aware compaction (feed only nonzero (value,index) pairs)
  C. mask-driven PE gating (propagate a zero mask alongside activations)

Outputs dense (43021), coarse (39132), and the fine-grained estimates.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "benchmark" / "fine_sparse_results.json"

DENSE = 43021          # frozen Step 4.3 OS baseline
COARSE = 39132         # frozen Step 5.2 coarse group skip
TOTAL_MACS = 416520
SKIPPABLE_MACS = 246081          # measured per-MAC (Step 5.1)
NONZERO_MACS = TOTAL_MACS - SKIPPABLE_MACS


def main() -> int:
    gated_pct = SKIPPABLE_MACS / TOTAL_MACS * 100.0

    # Option A/C (per-PE gating / mask gating): gate the multiplier, but the
    # systolic schedule is fixed, so cycles are unchanged. On DSP48E2 the
    # multiplier is a fixed-power hard block, so even energy savings are ~0.
    fine_gating = {
        "cycles": DENSE,
        "speedup": DENSE / DENSE,
        "gated_macs_per_image": SKIPPABLE_MACS,
        "gated_mac_pct": gated_pct,
        "cycle_change": 0,
        "energy_note": "negligible on DSP48E2 (fixed-power hard multiplier, always clocked)",
    }

    # Option B (compaction): theoretical bound = compute only nonzero MACs.
    theoretical_cycles = DENSE * (NONZERO_MACS / TOTAL_MACS)
    theoretical_speedup = DENSE / theoretical_cycles
    # metadata overhead: (value,index) pairs ~2x activation width + crossbar
    # indirection; conservative 1.3x cycle overhead estimate (NOT measured).
    metadata_overhead = 1.3
    estimated_cycles = theoretical_cycles * metadata_overhead
    fine_compaction = {
        "nonzero_macs_per_image": NONZERO_MACS,
        "theoretical_cycles_zero_overhead": theoretical_cycles,
        "theoretical_speedup": theoretical_speedup,
        "metadata_overhead_est": metadata_overhead,
        "estimated_cycles": estimated_cycles,
        "estimated_speedup": DENSE / estimated_cycles,
        "practicality": "IMPRACTICAL — requires crossbar/indirection (non-systolic), "
                        "abandons the frozen 8x8 array; index+value metadata and "
                        "sparse-detection become new bottlenecks",
    }

    results = {
        "dense_cycles": DENSE,
        "coarse_cycles": COARSE,
        "coarse_speedup": DENSE / COARSE,
        "fine_gating_cycles": fine_gating["cycles"],
        "fine_gating_speedup": fine_gating["speedup"],
        "fine_compaction": fine_compaction,
        "conclusion": "No practical fine-grained mechanism converts the 59.08% MAC "
                      "sparsity into cycle savings on the frozen 8x8 DSP48E2 systolic "
                      "array: gating (A/C) saves neither cycles nor (on DSP48E2) energy; "
                      "compaction (B) is a different architecture.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    print("=" * 76)
    print("Fine-grained activation sparsity — model (per image, 10k MNIST)")
    print("=" * 76)
    print(f"  dense OS                    : {DENSE:>7} cycles")
    print(f"  coarse group skip           : {COARSE:>7} cycles  ({DENSE/COARSE:.3f}x)")
    print(f"  fine gating (A/C)           : {fine_gating['cycles']:>7} cycles  "
          f"({fine_gating['speedup']:.3f}x, gated {gated_pct:.1f}% MACs)")
    print(f"  fine compaction (B) theor.  : {theoretical_cycles:>7.0f} cycles  "
          f"({theoretical_speedup:.3f}x, zero overhead)")
    print(f"  fine compaction (B) est.    : {estimated_cycles:>7.0f} cycles  "
          f"({DENSE/estimated_cycles:.3f}x, {metadata_overhead:.1f}x metadata overhead)")
    print("=" * 76)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
