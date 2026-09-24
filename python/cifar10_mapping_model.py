#!/usr/bin/env python3
"""OS/WS hardware-mapping model for the frozen CIFAR-10 workload (Step 6.4).

Maps the four learned CIFAR-10 layers to the frozen 8x8 reconfigurable systolic
array (``rtl/common/systolic_array_v2.sv`` + ``pe_v2.sv``) using the **same**
cycle methodology as the frozen LeNet-5 baseline (``mapping_model.py``), i.e. the
formula family anchored to the two RTL-verified schedules and the Step 4.1
corrected FC schedule.

    OS conv group = 17 + IC*K*(K+8)     cycles   [RTL-verified 82 for IC=1,K=5]
    WS conv group = 8*ceil(T/8) + 10    cycles   [RTL-verified 42 for T=25]
    OS FC   = ceil(OC/8)*(IC+2) + 2*OC  cycles   [RTL-verified bit-exact, Step 4.1]
    WS FC   = OC*(8*ceil(IC/8) + 2)     cycles   [analytically derived, Step 4.1]

It is **independent of the LeNet-5 model** (defines its own layer table) but
reads the frozen LeNet baseline (``data/benchmark/mapping_results.json``) for the
side-by-side comparison. It does NOT assume WS should win and does NOT force a
crossover: it computes both dataflows and reports the winner honestly.

Timing verification status is carried per layer/mode (RTL-verified /
analytically-derived / unverified-generalization) — never silently upgraded.

Run:  .venv/bin/python python/cifar10_mapping_model.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "benchmark" / "cifar10_mapping_results.json"
LENET_BASELINE = REPO_ROOT / "data" / "benchmark" / "mapping_results.json"

PE_COUNT = 64
ARRAY_ROWS = 8
ARRAY_COLS = 8

# Frozen CIFAR-10 learned layers (docs/CIFAR10_WORKLOAD_SPEC.md §2, §3).
# Pooling + ReLU are PS-side (as in LeNet-5), so only the 4 learned layers map
# to the array. H/W are the OUTPUT spatial dims (H_out, W_out).
LAYERS = [
    {"name": "conv1", "type": "conv", "IC": 3,  "OC": 32, "K": 5, "H": 28, "W": 28, "MACs": 1881600},
    {"name": "conv2", "type": "conv", "IC": 32, "OC": 32, "K": 5, "H": 10, "W": 10, "MACs": 2560000},
    {"name": "conv3", "type": "conv", "IC": 32, "OC": 64, "K": 5, "H": 1,  "W": 1,  "MACs": 51200},
    {"name": "fc",    "type": "fc",   "IC": 64, "OC": 10, "K": 1, "H": 1,  "W": 1,  "MACs": 640},
]

TOTAL_MACS = 4493440  # frozen (docs/CIFAR10_WORKLOAD_SPEC.md §3)


def _ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def partial_pixels(W: int) -> int:
    """Valid pixels in the last column group of a W-wide output row."""
    return W % ARRAY_COLS if W % ARRAY_COLS else ARRAY_COLS


def os_map(L: dict) -> dict:
    """Output-Stationary mapping for one layer (rows = output channels)."""
    K, IC, OC = L["K"], L["IC"], L["OC"]
    T = K * K * IC
    spatial = L["H"] * _ceil(L["W"], ARRAY_COLS)
    channel_groups = _ceil(OC, ARRAY_ROWS)
    if K == 1:
        # FC (1x1 conv, 1 output pixel) — RTL-verified (Step 4.1): rows = output
        # neurons, IC features serialized; the 8-pixel shift-chain lead-in vanishes.
        cycles = channel_groups * (IC + 2) + 2 * OC
        return {
            "reduction_taps": IC,
            "ic_tiling": "serialized (IC features, 1/cycle)",
            "oc_tiling": f"{channel_groups} channel group(s) of {ARRAY_ROWS} rows",
            "spatial_groups": 1,
            "channel_groups": channel_groups,
            "passes": IC,
            "cycles_per_group": IC + 2 + 2 * ARRAY_ROWS,  # nominal full group
            "groups": channel_groups,
            "cycles": cycles,
        }
    # Conv: reduction serialized as IC*K row-decomposed passes; each pass streams
    # K kernel taps across the 8-column pixel chain (P+K = 8+K cycles incl. lead-in).
    group = 17 + IC * K * (K + 8)
    return {
        "reduction_taps": T,
        "ic_tiling": f"serialized ({IC} IC x {K} rows = {IC * K} passes/group)",
        "oc_tiling": f"{channel_groups} channel group(s) of {ARRAY_ROWS} rows (OC={OC})",
        "spatial_groups": spatial,
        "channel_groups": channel_groups,
        "passes": IC * K,
        "cycles_per_group": group,
        "groups": channel_groups * spatial,
        "cycles": group * channel_groups * spatial,
    }


def ws_map(L: dict) -> dict:
    """Weight-Stationary mapping for one layer (rows = 8 reduction taps)."""
    K, IC, OC = L["K"], L["IC"], L["OC"]
    T = K * K * IC
    spatial = L["H"] * _ceil(L["W"], ARRAY_COLS)
    tiles = _ceil(T, ARRAY_ROWS)
    if K == 1:
        # FC — one output neuron per sweep (per-row weight broadcast prevents 8
        # distinct FC outputs per pass); IC taps tiled 8/tile. The conv "+10"
        # (7-cycle 8-pixel lead-in) is absent for 1 pixel -> clear(1)+latch(1).
        group = 8 * tiles + 2
        return {
            "reduction_taps": IC,
            "ic_tiling": f"folded into {tiles} tile(s) of {ARRAY_ROWS} taps",
            "oc_tiling": f"{OC} serialized sweeps (1 neuron/sweep)",
            "spatial_groups": 1,
            "tiles": tiles,
            "sweeps": OC,
            "cycles_per_group": group,
            "groups": OC,
            "cycles": OC * group,
        }
    group = 8 * tiles + 10
    return {
        "reduction_taps": T,
        "ic_tiling": f"folded into {T} taps -> {tiles} tile(s) of {ARRAY_ROWS}",
        "oc_tiling": f"{OC} serialized sweeps (1 channel/sweep)",
        "spatial_groups": spatial,
        "tiles": tiles,
        "sweeps": OC,
        "cycles_per_group": group,
        "groups": OC * spatial,
        "cycles": group * OC * spatial,
    }


def os_conv_refined(L: dict) -> int:
    """OS conv with per-group partial-column refinement (FC_SCHEDULE_VALIDATION §6).

    The frozen LeNet baseline uses a UNIFORM group cost ``17 + IC*K*(K+8)`` for
    every column group, which overcounts the last (partial) column group of a
    non-multiple-of-8 width. The §6 correction is ``1 + IC*K*(P+K) + 2*rows``
    where P = valid pixels in the group. WS is unaffected (its group cost depends
    only on reduction taps T, not on P), so this refinement *reduces OS only* and
    therefore widens OS's lead. All CIFAR conv layers have OC a multiple of 8, so
    rows = 8 in every channel group.
    """
    K, IC, OC = L["K"], L["IC"], L["OC"]
    H, W = L["H"], L["W"]
    cg = _ceil(OC, ARRAY_ROWS)
    full = W // ARRAY_COLS
    rem = W % ARRAY_COLS
    row = full * (1 + IC * K * (ARRAY_COLS + K) + 2 * ARRAY_ROWS)
    if rem:
        row += (1 + IC * K * (rem + K) + 2 * ARRAY_ROWS)
    return cg * H * row


def timing_status(L: dict, mode: str) -> str:
    """Verification status of the cycle count for one layer/mode (never upgraded)."""
    IC, K, OC = L["IC"], L["K"], L["OC"]
    if L["K"] == 1:
        if mode == "OS":
            return "RTL-verified (general 1-pixel FC schedule, bit-exact)"
        return "analytically derived (conv-WS anchor, FC_SCHEDULE_VALIDATION §3)"
    if mode == "OS":
        if IC == 1 and K == 5:
            return "RTL-verified (82-cycle anchor, IC=1 K=5)"
        return "unverified generalization (IC>1 and/or OC>8 — controller gaps G1/G3)"
    # WS conv
    T = K * K * IC
    if T == 25:
        return "RTL-verified (42-cycle anchor, 25 taps)"
    return "unverified generalization (T != 25 — controller gap G1)"


def layer_result(L: dict) -> dict:
    os = os_map(L)
    ws = ws_map(L)
    winner = "OS" if os["cycles"] < ws["cycles"] else ("WS" if ws["cycles"] < os["cycles"] else "tie")
    ratio = os["cycles"] / ws["cycles"]

    # weight reads are mode-independent: full weight set re-streamed once per
    # spatial group (OS: per channel group; WS: per sweep — same total).
    total_weights = L["OC"] * L["IC"] * L["K"] * L["K"]
    weight_reads = total_weights * max(1, os["spatial_groups"])

    def annotate(m: dict, mode: str) -> dict:
        macs = L["MACs"]
        util = macs / (PE_COUNT * m["cycles"])
        return {
            "cycles": m["cycles"],
            "cycles_per_group": m["cycles_per_group"],
            "groups": m["groups"],
            "reduction_taps": m["reduction_taps"],
            "ic_tiling": m["ic_tiling"],
            "oc_tiling": m["oc_tiling"],
            "spatial_groups": m["spatial_groups"],
            "partial_column_valid_pixels": partial_pixels(L["W"]),
            "weight_reads": weight_reads,
            "PE_utilization": util,
            "idle_PE_cycles": PE_COUNT * m["cycles"] - macs,
            "psum_operations": 0 if mode == "OS" else macs,
            "psum_behavior": "local accumulator (acc += product)" if mode == "OS"
                              else "vertical cascade + bottom-to-top tile feedback",
            "timing_status": timing_status(L, mode),
        }

    return {
        "type": L["type"], "IC": L["IC"], "OC": L["OC"], "K": L["K"],
        "H_out": L["H"], "W_out": L["W"], "MACs": L["MACs"],
        "reduction_taps": os["reduction_taps"],
        "OS": {**os, **annotate(os, "OS")},
        "WS": {**ws, **annotate(ws, "WS")},
        "winner": winner,
        "os_over_ws": ratio,
        "os_faster_by": ws["cycles"] / os["cycles"],
    }


def load_lenet_baseline() -> dict | None:
    if not LENET_BASELINE.exists():
        return None
    return json.loads(LENET_BASELINE.read_text())


def compute() -> dict:
    layers = {L["name"]: layer_result(L) for L in LAYERS}
    names = [L["name"] for L in LAYERS]

    os_total = sum(layers[n]["OS"]["cycles"] for n in names)
    ws_total = sum(layers[n]["WS"]["cycles"] for n in names)

    # Partial-column refinement (FC_SCHEDULE_VALIDATION §6): reduces OS conv
    # partial groups; WS unchanged. Conservative-for-OS robustness check.
    refined_os = {n: (os_conv_refined(L) if L["K"] > 1 else layers[n]["OS"]["cycles"])
                  for n, L in zip(names, LAYERS)}
    refined_os_total = sum(refined_os.values())

    # Reconfiguration: per-layer best (+ 8-cycle FLUSH per OS<->WS switch).
    switch_overhead = 8
    oracle_modes = [("OS" if layers[n]["OS"]["cycles"] <= layers[n]["WS"]["cycles"] else "WS") for n in names]
    switches = sum(1 for i in range(len(oracle_modes) - 1) if oracle_modes[i] != oracle_modes[i + 1])
    oracle_cycles = sum(layers[n][oracle_modes[i]]["cycles"] for i, n in enumerate(names))
    overhead = switches * switch_overhead
    reconfig_cycles = oracle_cycles + overhead

    # Per-layer classification (no arbitrary "30%" threshold): a layer is
    # "near tie" only if the two modes are within a small implementation-noise
    # band (say <5%); otherwise the faster mode "clearly preferred".
    classification = {}
    for n in names:
        r = layers[n]
        os_, ws_ = r["OS"]["cycles"], r["WS"]["cycles"]
        faster, slower = (os_, ws_) if os_ <= ws_ else (ws_, os_)
        spread = (slower - faster) / slower
        if spread < 0.05:
            cls = "near tie / sensitive to overhead"
        else:
            cls = ("OS clearly preferred" if os_ <= ws_ else "WS clearly preferred")
        classification[n] = {"class": cls, "OS": os_, "WS": ws_, "winner": r["winner"]}

    result = {
        "workload": "CIFAR-10 cuda-convnet (frozen, docs/CIFAR10_WORKLOAD_SPEC.md)",
        "array": {
            "PEs": PE_COUNT, "rows": ARRAY_ROWS, "cols": ARRAY_COLS,
            "anchor_os": "82 cyc (IC=1,K=5)", "anchor_ws": "42 cyc (T=25)",
            "fc_os_verified": "IC+2*rows+2 (bit-exact)", "fc_ws_derived": "8*ceil(IC/8)+2",
        },
        "note": (
            "Same frozen formula family as the LeNet-5 baseline. Conv OS/WS timings "
            "for IC>1/OC>8 are unverified generalizations (controller gaps G1/G3); "
            "FC OS is RTL-verified, FC WS analytically derived. No dataflow or "
            "sparsity benefit is assumed."
        ),
        "layers": layers,
        "totals": {
            "MACs": TOTAL_MACS,
            "OS_cycles": os_total,
            "WS_cycles": ws_total,
            "OS_PE_utilization": TOTAL_MACS / (PE_COUNT * os_total),
            "WS_PE_utilization": TOTAL_MACS / (PE_COUNT * ws_total),
            "OS_over_WS": os_total / ws_total,
            "WS_over_OS": ws_total / os_total,
            "refined_OS_cycles": refined_os_total,
            "refined_OS_over_WS": refined_os_total / ws_total,
            "refinement_note": (
                "Uniform per-group cost (frozen baseline) overcounts OS partial "
                "column groups; FC_SCHEDULE_VALIDATION §6 per-group correction "
                "1+IC*K*(P+K)+2*rows lowers OS only (WS unchanged) -> OS lead widens."
            ),
        },
        "per_layer_classification": classification,
        "reconfiguration": {
            "switch_overhead_cycles": switch_overhead,
            "selected_modes": {n: m for n, m in zip(names, oracle_modes)},
            "num_switches": switches,
            "total_overhead": overhead,
            "oracle_cycles": oracle_cycles,
            "reconfigurable_cycles": reconfig_cycles,
            "fixed_OS_cycles": os_total,
            "fixed_WS_cycles": ws_total,
            "speedup_vs_fixed_OS": os_total / reconfig_cycles,
            "speedup_vs_fixed_WS": ws_total / reconfig_cycles,
            "reconfigurable_beneficial_vs_OS": reconfig_cycles < os_total,
            "no_crossover": all(m == "OS" for m in oracle_modes),
        },
        "lenet_comparison": {},
    }

    base = load_lenet_baseline()
    if base:
        ln = base["layers"]
        os_l = sum(ln[k]["OS"]["cycles"] for k in ln)
        ws_l = sum(ln[k]["WS"]["cycles"] for k in ln)
        result["lenet_comparison"] = {
            "source": str(LENET_BASELINE.relative_to(REPO_ROOT)),
            "OS_cycles": os_l, "WS_cycles": ws_l, "OS_over_WS": os_l / ws_l,
            "per_layer": {k: {"OS": ln[k]["OS"]["cycles"], "WS": ln[k]["WS"]["cycles"]}
                          for k in ln},
        }
    return result


def _print_table(r: dict) -> None:
    print("=" * 100)
    print("OS vs WS hardware mapping — CIFAR-10 (8x8 array, un-pipelined baseline)")
    print("=" * 100)
    hdr = (f"{'layer':6} {'IC->OC':>10} {'T':>5} {'MACs':>9} | "
           f"{'OS cyc':>8} {'OS util':>7} | {'WS cyc':>8} {'WS util':>7} | "
           f"{'winner':6} {'OS/WS':>6}")
    print(hdr)
    print("-" * 100)
    for n, lr in r["layers"].items():
        print(f"{n:6} {str(lr['IC'])+'->'+str(lr['OC']):>10} {lr['reduction_taps']:>5} "
              f"{lr['MACs']:>9} | {lr['OS']['cycles']:>8} {lr['OS']['PE_utilization']*100:>6.1f}% | "
              f"{lr['WS']['cycles']:>8} {lr['WS']['PE_utilization']*100:>6.1f}% | "
              f"{lr['winner']:6} {lr['os_over_ws']:>6.2f}x")
    print("-" * 100)
    t = r["totals"]
    print(f"{'TOTAL':6} {'':>10} {'':>5} {t['MACs']:>9} | {t['OS_cycles']:>8} "
          f"{t['OS_PE_utilization']*100:>6.1f}% | {t['WS_cycles']:>8} "
          f"{t['WS_PE_utilization']*100:>6.1f}% | {'':6} {t['OS_over_WS']:>6.2f}x")
    print("=" * 100)
    rc = r["reconfiguration"]
    print(f"Selected modes : {list(rc['selected_modes'].values())}")
    print(f"Switches       : {rc['num_switches']}  (overhead {rc['total_overhead']} cyc)")
    print(f"Reconfigurable : {rc['reconfigurable_cycles']} cyc  vs fixed OS {rc['fixed_OS_cycles']} "
          f"(speedup {rc['speedup_vs_fixed_OS']:.3f}x)  vs fixed WS {rc['fixed_WS_cycles']} "
          f"(speedup {rc['speedup_vs_fixed_WS']:.3f}x)")
    print(f"No crossover   : {rc['no_crossover']}")

    if r["lenet_comparison"]:
        lc = r["lenet_comparison"]
        print("=" * 100)
        print(f"LeNet-5 baseline (frozen): OS {lc['OS_cycles']}  WS {lc['WS_cycles']}  "
              f"OS/WS {lc['OS_over_WS']:.3f}x")


def main() -> int:
    r = compute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=2))
    _print_table(r)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
