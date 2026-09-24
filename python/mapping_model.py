#!/usr/bin/env python3
"""Cycle-accurate OS/WS mapping model for the frozen LeNet-5 workload (Step 3.2).

Anchored to the two *verified* array schedules (docs/PHASE2_ARCHITECTURE.md,
docs/specs/SYSTOLIC_ARRAY_V2_SPEC.md §9):

  OS group  = 17 + IC·K·(K+8)        cycles   [verified 82 for IC=1, K=5]
  WS group  = 8·ceil(T/8) + 10        cycles   [verified 42 for T=25], T = K·K·IC

Mappings (docs/LENET5_MAPPING_SPEC.md):
  OS rows = output channels (8), columns = 8 pixels, reduction serialized in time.
  WS rows = 8 reduction taps/tile, columns = 8 pixels, channels serialized.

Honest constraints recorded (not hidden):
  * FC maps to the array as a 1×1 conv; it has 1 output "pixel", so BOTH modes
    waste 7/8 columns.
  * WS cannot produce 8 distinct FC outputs per pass (per-row weight broadcast);
    WS-FC is one output neuron per sweep (7/8 columns idle).
  * OS-FC is output-neuron-parallel (rows = 8 neurons) but serializes the whole
    reduction (IC taps) in time.

Run:  .venv/bin/python python/mapping_model.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "benchmark" / "mapping_results.json"

PE_COUNT = 64

LAYERS = [
    {"name": "conv1", "type": "conv", "IC": 1, "OC": 6, "K": 5, "H": 28, "W": 28, "MACs": 117600},
    {"name": "conv3", "type": "conv", "IC": 6, "OC": 16, "K": 5, "H": 10, "W": 10, "MACs": 240000},
    {"name": "conv5", "type": "conv", "IC": 16, "OC": 120, "K": 5, "H": 1, "W": 1, "MACs": 48000},
    {"name": "fc1", "type": "fc", "IC": 120, "OC": 84, "K": 1, "H": 1, "W": 1, "MACs": 10080},
    {"name": "fc2", "type": "fc", "IC": 84, "OC": 10, "K": 1, "H": 1, "W": 1, "MACs": 840},
]


def _ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def os_map(L: dict) -> dict:
    K, IC, OC = L["K"], L["IC"], L["OC"]
    T = K * K * IC
    if K == 1:
        # FC (1x1 conv, 1 pixel) — RTL-verified (Step 4.1): rows = output neurons,
        # IC features serialized. Per group: 1 clear + IC feed + 1 tail; drain
        # 2 cycles/row over all OC rows (the 8-pixel shift-chain lead-in vanishes).
        groups = _ceil(OC, 8)
        cycles = groups * (IC + 2) + 2 * OC
        return {
            "reduction_taps": IC,
            "passes": IC,
            "cycles_per_group": IC + 18,             # full 8-row group (nominal)
            "spatial_groups": 1,
            "channel_groups": groups,
            "groups": groups,
            "cycles": cycles,
        }
    passes = IC * K                                  # row-decomposed passes
    group = 17 + IC * K * (K + 8)                    # 1 clear + passes*(7+K+1) + 16 drain
    spatial = L["H"] * _ceil(L["W"], 8)
    groups = _ceil(OC, 8) * spatial
    return {
        "reduction_taps": T,
        "passes": passes,
        "cycles_per_group": group,
        "spatial_groups": spatial,
        "channel_groups": _ceil(OC, 8),
        "groups": groups,
        "cycles": group * groups,
    }


def ws_map(L: dict) -> dict:
    K, IC, OC = L["K"], L["IC"], L["OC"]
    T = K * K * IC
    if K == 1:
        # FC — one output neuron per sweep, IC taps tiled 8/tile. The conv
        # `+10` overhead (7-cycle 8-pixel lead-in) is absent for 1 pixel; only
        # clear (1) + latch (1) remain (Step 4.1).
        tiles = _ceil(IC, 8)
        group = 8 * tiles + 2
        return {
            "reduction_taps": IC,
            "tiles": tiles,
            "cycles_per_group": group,
            "spatial_groups": 1,
            "channel_sweeps": OC,
            "groups": OC,
            "cycles": OC * group,
        }
    tiles = _ceil(T, 8)
    group = 8 * tiles + 10
    spatial = L["H"] * _ceil(L["W"], 8)
    groups = OC * spatial                             # channels serialized
    return {
        "reduction_taps": T,
        "tiles": tiles,
        "cycles_per_group": group,
        "spatial_groups": spatial,
        "channel_sweeps": OC,
        "groups": groups,
        "cycles": group * groups,
    }


def main() -> int:
    results = {"array": {"PEs": PE_COUNT, "anchor_os": "82 cyc (IC=1,K=5)", "anchor_ws": "42 cyc (T=25)"},
               "layers": {}}
    total_macs = 0
    for L in LAYERS:
        os = os_map(L)
        ws = ws_map(L)
        total_macs += L["MACs"]
        winner = "OS" if os["cycles"] < ws["cycles"] else ("WS" if ws["cycles"] < os["cycles"] else "tie")
        ratio = os["cycles"] / ws["cycles"]
        results["layers"][L["name"]] = {
            "type": L["type"], "IC": L["IC"], "OC": L["OC"], "K": L["K"],
            "H_out": L["H"], "W_out": L["W"], "MACs": L["MACs"],
            "OS": {**os, "PE_util": L["MACs"] / (PE_COUNT * os["cycles"]),
                   "idle_PE_cycles": PE_COUNT * os["cycles"] - L["MACs"],
                   "weight_reloads_per_group": L["IC"] * L["K"],
                   "activation_streams": 1, "psum": "none (local accumulator)"},
            "WS": {**ws, "PE_util": L["MACs"] / (PE_COUNT * ws["cycles"]),
                   "idle_PE_cycles": PE_COUNT * ws["cycles"] - L["MACs"],
                   "weight_reloads_per_group": ws["tiles"],
                   "activation_streams": 8, "psum": "vertical cascade + tile feedback"},
            "winner": winner, "os_over_ws": ratio,
        }
    results["totals"] = {"MACs": total_macs,
                         "OS_cycles": sum(results["layers"][n]["OS"]["cycles"] for n in results["layers"]),
                         "WS_cycles": sum(results["layers"][n]["WS"]["cycles"] for n in results["layers"])}
    results["note"] = (
        "FC layers are 1-pixel 1x1 convs; both modes waste 7/8 columns. WS cannot "
        "emit 8 distinct FC outputs per pass (per-row weight broadcast) — WS-FC is "
        "one neuron/sweep. Cycles are the un-pipelined correctness-first baseline."
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    # human-readable table
    print("=" * 92)
    print("OS vs WS cycle mapping — LeNet-5 (8x8 array, un-pipelined baseline)")
    print("=" * 92)
    print(f"{'layer':6} {'type':4} {'T':>4} {'MACs':>8} | {'OS cyc':>8} {'OS util':>7} | {'WS cyc':>8} {'WS util':>7} | {'winner':6} {'OS/WS':>6}")
    print("-" * 92)
    for n, r in results["layers"].items():
        print(f"{n:6} {r['type']:4} {r['OS']['reduction_taps']:>4} {r['MACs']:>8} | "
              f"{r['OS']['cycles']:>8} {r['OS']['PE_util']*100:>6.1f}% | "
              f"{r['WS']['cycles']:>8} {r['WS']['PE_util']*100:>6.1f}% | "
              f"{r['winner']:6} {r['os_over_ws']:>6.2f}x")
    print("-" * 92)
    t = results["totals"]
    print(f"{'TOTAL':6} {'':4} {'':>4} {t['MACs']:>8} | {t['OS_cycles']:>8} {'':>7} | {t['WS_cycles']:>8} {'':>7} | {'':6} {t['OS_cycles']/t['WS_cycles']:>6.2f}x")
    print("=" * 92)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
