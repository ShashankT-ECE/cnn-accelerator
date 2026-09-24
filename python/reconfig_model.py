#!/usr/bin/env python3
"""Reconfigurable OS/WS dataflow model (Step 3.3).

Extends the cycle model (``mapping_model.py``) with per-layer OS/WS mode
selection and explicit reconfiguration overhead: each OS<->WS switch between
consecutive layers costs the documented 8-cycle FLUSH
(``docs/PHASE2_ARCHITECTURE.md`` §3.3). It does **not** assume reconfiguration is
beneficial — it computes the switch overhead and reports it explicitly.

Run:  .venv/bin/python python/reconfig_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from mapping_model import LAYERS, os_map, ws_map

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "benchmark" / "reconfig_results.json"

SWITCH_OVERHEAD = 8  # cycles per OS<->WS mode switch (FLUSH)


def per_layer_selection() -> list[dict]:
    rows = []
    for L in LAYERS:
        os = os_map(L)
        ws = ws_map(L)
        best = "OS" if os["cycles"] <= ws["cycles"] else "WS"
        best_cyc = os["cycles"] if best == "OS" else ws["cycles"]
        other_cyc = ws["cycles"] if best == "OS" else os["cycles"]
        rows.append({
            "name": L["name"], "type": L["type"],
            "best": best, "best_cycles": best_cyc,
            "os_cycles": os["cycles"], "ws_cycles": ws["cycles"],
            "switch_benefit": other_cyc - best_cyc,   # >0: how much the best mode saves
        })
    return rows


def num_switches(modes: list[str]) -> int:
    return sum(1 for i in range(len(modes) - 1) if modes[i] != modes[i + 1])


def compute() -> dict:
    rows = per_layer_selection()
    names = [r["name"] for r in rows]
    oracle_modes = [r["best"] for r in rows]
    switches = num_switches(oracle_modes)
    overhead = switches * SWITCH_OVERHEAD
    oracle_cycles = sum(r["best_cycles"] for r in rows)      # per-layer best, no overhead
    reconfig_cycles = oracle_cycles + overhead                # realistic: + flush per switch

    fixed_os = sum(r["os_cycles"] for r in rows)
    fixed_ws = sum(r["ws_cycles"] for r in rows)

    # overhead-aware sanity: would any switch be declined? (benefit <= overhead)
    declined = [r["name"] for r in rows if r["switch_benefit"] <= SWITCH_OVERHEAD]

    return {
        "switch_overhead_cycles": SWITCH_OVERHEAD,
        "selected_modes": {n: m for n, m in zip(names, oracle_modes)},
        "layers": rows,
        "num_switches": switches,
        "total_switch_overhead": overhead,
        "oracle_cycles": oracle_cycles,                       # per-layer best, no overhead
        "reconfigurable_cycles": reconfig_cycles,             # per-layer best + overhead
        "fixed_os_cycles": fixed_os,
        "fixed_ws_cycles": fixed_ws,
        "speedup_vs_fixed_os": fixed_os / reconfig_cycles,
        "speedup_vs_fixed_ws": fixed_ws / reconfig_cycles,
        "overhead_fraction": overhead / reconfig_cycles,
        "switches_declined_by_overhead": declined,
        "reconfigurable_beneficial_vs_os": reconfig_cycles < fixed_os,
    }


def main() -> int:
    r = compute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=2))

    print("=" * 76)
    print("Reconfigurable OS/WS model — LeNet-5 (per-layer mode selection)")
    print("=" * 76)
    print(f"{'layer':6} {'type':4} {'best':4} {'OS cyc':>8} {'WS cyc':>8} {'best cyc':>8} {'benefit':>8}")
    print("-" * 76)
    for row in r["layers"]:
        print(f"{row['name']:6} {row['type']:4} {row['best']:4} {row['os_cycles']:>8} "
              f"{row['ws_cycles']:>8} {row['best_cycles']:>8} {row['switch_benefit']:>8}")
    print("-" * 76)
    print(f"Selected modes        : {list(r['selected_modes'].values())}")
    print(f"Num switches          : {r['num_switches']}  (overhead {r['total_switch_overhead']} cycles)")
    print(f"Oracle (per-layer)    : {r['oracle_cycles']} cycles")
    print(f"Reconfigurable (+ovh) : {r['reconfigurable_cycles']} cycles")
    print(f"Fixed OS              : {r['fixed_os_cycles']} cycles")
    print(f"Fixed WS              : {r['fixed_ws_cycles']} cycles")
    print(f"Speedup vs fixed OS   : {r['speedup_vs_fixed_os']:.3f}x")
    print(f"Speedup vs fixed WS   : {r['speedup_vs_fixed_ws']:.3f}x")
    print(f"Overhead fraction     : {r['overhead_fraction']*100:.4f}%")
    print("=" * 76)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
