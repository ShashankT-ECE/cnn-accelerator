#!/usr/bin/env python3
"""Hardware-activity metrics for the LeNet-5 OS/WS mapping (Step 3.4).

Extends the cycle model (``mapping_model.py``) and the reconfigurable model
(``reconfig_model.py``) with per-layer activity accounting, consistent with the
frozen 8x8 array (64 PEs, 1 MAC/PE/cycle, int8xint8->int32).

Defined by the architecture:
  * cycles, MACs, active-PE ops (= MACs, dense), PE utilization, idle PE cycles,
  * weight values (frozen param count) and weight reads (mode-independent:
    each weight is streamed once per spatial position in both OS and WS),
  * psum operations (0 in OS; == MACs in WS, since every WS MAC is `psum_in+product`),
  * reconfiguration overhead (8 cycles per OS<->WS switch).

Explicitly marked UNAVAILABLE (not defined by the frozen architecture — the
line-buffer/weight-store internals were deferred, and no clock x bus-width is
pinned):
  * exact activation-read counts (line-buffer reuse is controller-dependent),
  * byte-level data-movement / bandwidth.

Run:  .venv/bin/python python/activity_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from mapping_model import LAYERS, PE_COUNT, os_map, ws_map
from reconfig_model import SWITCH_OVERHEAD, compute as reconfig_compute, per_layer_selection

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "benchmark" / "activity_results.json"


def activity(L: dict, m: dict, mode: str) -> dict:
    """Activity metrics for one layer + mode (``m`` = os_map/ws_map result)."""
    MACs = L["MACs"]
    IC, OC, K = L["IC"], L["OC"], L["K"]
    total_weights = OC * IC * K * K
    input_tensor = IC * (L["H"] + K - 1) * (L["W"] + K - 1)
    return {
        "cycles": m["cycles"],
        "MACs": MACs,
        "active_PE_operations": MACs,                          # dense: every MAC executes
        "PE_utilization": MACs / (PE_COUNT * m["cycles"]),
        "idle_PE_cycles": PE_COUNT * m["cycles"] - MACs,
        "MAC_utilization": MACs / (PE_COUNT * m["cycles"]),    # useful-work efficiency (== PE util, dense)
        "weight_values": total_weights,
        "weight_reads": total_weights * m["spatial_groups"],   # mode-independent
        "input_tensor_size": input_tensor,
        "activation_reads": None,  # UNAVAILABLE: line-buffer reuse is controller-dependent
        "psum_operations": 0 if mode == "OS" else MACs,        # WS MAC == psum_in + product
        "psum_movement": "none (local accumulator)" if mode == "OS"
                         else "vertical cascade + bottom-to-top feedback",
    }


def main() -> int:
    sel = {r["name"]: r for r in per_layer_selection()}
    rc = reconfig_compute()

    results = {"array": {"PEs": PE_COUNT},
               "unavailable": [
                   "activation_reads (line-buffer reuse is controller-dependent)",
                   "byte-level data movement / bandwidth (no clock x bus-width pinned)",
               ],
               "layers": {}}

    for L in LAYERS:
        n = L["name"]
        os = os_map(L)
        ws = ws_map(L)
        results["layers"][n] = {
            "type": L["type"],
            "OS": activity(L, os, "OS"),
            "WS": activity(L, ws, "WS"),
            "selected_mode": sel[n]["best"],
            "selected": activity(L, os if sel[n]["best"] == "OS" else ws, sel[n]["best"]),
        }

    # totals
    tot = {}
    for mode in ("OS", "WS"):
        tot[mode] = {
            "cycles": sum(results["layers"][n][mode]["cycles"] for n in results["layers"]),
            "MACs": sum(results["layers"][n][mode]["MACs"] for n in results["layers"]),
            "active_PE_operations": sum(results["layers"][n][mode]["active_PE_operations"] for n in results["layers"]),
            "idle_PE_cycles": sum(results["layers"][n][mode]["idle_PE_cycles"] for n in results["layers"]),
            "weight_reads": sum(results["layers"][n][mode]["weight_reads"] for n in results["layers"]),
            "psum_operations": sum(results["layers"][n][mode]["psum_operations"] for n in results["layers"]),
        }
        tot[mode]["PE_utilization"] = tot[mode]["MACs"] / (PE_COUNT * tot[mode]["cycles"])
    results["totals"] = tot
    results["reconfiguration"] = {
        "switch_overhead_cycles": SWITCH_OVERHEAD,
        "num_switches": rc["num_switches"],
        "total_overhead": rc["total_switch_overhead"],
        "reconfigurable_cycles": rc["reconfigurable_cycles"],
        "selected_cycles_no_overhead": sum(results["layers"][n]["selected"]["cycles"] for n in results["layers"]),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    # human-readable table (per layer, selected mode)
    print("=" * 100)
    print("Hardware activity — LeNet-5 (per layer, selected mode; OS vs WS)")
    print("=" * 100)
    hdr = f"{'layer':6} {'mode':4} {'cycles':>8} {'MACs':>8} {'PE util':>7} {'idle PE':>9} {'w_reads':>8} {'psum':>8}"
    print(hdr)
    print("-" * 100)
    for n, r in results["layers"].items():
        for mode in ("OS", "WS"):
            a = r[mode]
            psum = a["psum_operations"]
            print(f"{n:6} {mode:4} {a['cycles']:>8} {a['MACs']:>8} {a['PE_utilization']*100:>6.1f}% "
                  f"{a['idle_PE_cycles']:>9} {a['weight_reads']:>8} {psum if psum is not None else 0:>8}")
    print("-" * 100)
    for mode in ("OS", "WS"):
        t = tot[mode]
        print(f"{'TOTAL':6} {mode:4} {t['cycles']:>8} {t['MACs']:>8} {t['PE_utilization']*100:>6.1f}% "
              f"{t['idle_PE_cycles']:>9} {t['weight_reads']:>8} {t['psum_operations']:>8}")
    print("=" * 100)
    rc = results["reconfiguration"]
    print(f"Reconfiguration: {rc['num_switches']} switch(es), {rc['total_overhead']} cycles overhead, "
          f"reconfigurable = {rc['reconfigurable_cycles']} cycles")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
