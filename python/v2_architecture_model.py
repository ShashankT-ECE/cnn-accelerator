#!/usr/bin/env python3
"""Generalized OS-only datapath cycle model for the proposed V2 architecture.

Resolves the controller gaps that made the frozen V1 baseline's conv cycle
counts an "unverified generalization" (`docs/LENET5_MAPPING_SPEC.md` §4):

  * **IC > 1**   — the reduction IC*K*K is serialized as IC*K row-decomposed
                   passes; input channels are re-fed per pass.
  * **OC > 8**   — output channels tiled into ceil(OC/8) groups; the last group
                   has `rows = OC mod 8` active PE rows (drain = 2*rows).
  * **variable spatial** — the pixel-group lead-in is `P-1` (per-column shift
                   chain), so a partial column group (P < 8) is shorter than a
                   full one; the degenerate W'=1 layer has P=1.
  * **VALID convolution** — output H'=H-K+1, W'=W-K+1, no zero-injection; the
                   right-edge partial group has P = W' mod 8 valid pixels.

This yields the **corrected** per-group schedule
(`docs/FC_SCHEDULE_VALIDATION.md` §6):

    OS group = 1 + IC*K*(P+K) + 2*rows
                 ^ clear + first weight load (1 cycle)
                     ^ IC*K passes x (P-1 lead-in + K taps + 1 drain)
                                   ^ result drain (2 cycles per active row)

vs the frozen baseline's UNIFORM `17 + IC*K*(K+8)` (which forces P=8, rows=8
for every group and overcounts partial groups). The model also makes the
pipeline fill/drain stalls and the weight-BRAM read latency explicit.

Weight-BRAM (gap G5): a synchronous BRAM36 has a 1-cycle read latency; each OS
group issues IC*K*K weight reads (one per reduction tap, 8 weights = 64 bits per
read). With look-ahead addressing (the same technique Phase 2 used for image
reads, `PHASE2_HARDENING.md`) the latency is hidden — `weight_bram_stall = 0`.
A naive controller without look-ahead pays +1 cycle/group (first-tap prefetch).

Timing status is carried per layer and is NOT upgraded silently: the only
RTL-verified conv number is the 82-cycle anchor (IC=1, K=5, P=8, rows=8); the
generalized schedule is analytically derived. FC OS is RTL-verified (bit-exact).

Run:  .venv/bin/python python/v2_architecture_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "benchmark" / "v2_architecture_results.json"

ARRAY = 8          # PE rows == PE columns
PE_COUNT = ARRAY * ARRAY

# Frozen layer tables (H, W are OUTPUT dims H_out, W_out).
# LeNet-5 (docs/LENET5_MAPPING_FROZEN_MANIFEST.md):
LENET = [
    {"name": "conv1", "type": "conv", "IC": 1,  "OC": 6,   "K": 5, "H": 28, "W": 28, "MACs": 117600},
    {"name": "conv3", "type": "conv", "IC": 6,  "OC": 16,  "K": 5, "H": 10, "W": 10, "MACs": 240000},
    {"name": "conv5", "type": "conv", "IC": 16, "OC": 120, "K": 5, "H": 1,  "W": 1,  "MACs": 48000},
    {"name": "fc1",   "type": "fc",   "IC": 120, "OC": 84, "K": 1, "H": 1,  "W": 1,  "MACs": 10080},
    {"name": "fc2",   "type": "fc",   "IC": 84,  "OC": 10, "K": 1, "H": 1,  "W": 1,  "MACs": 840},
]
# CIFAR-10 (docs/CIFAR10_WORKLOAD_SPEC.md):
CIFAR = [
    {"name": "conv1", "type": "conv", "IC": 3,  "OC": 32, "K": 5, "H": 28, "W": 28, "MACs": 1881600},
    {"name": "conv2", "type": "conv", "IC": 32, "OC": 32, "K": 5, "H": 10, "W": 10, "MACs": 2560000},
    {"name": "conv3", "type": "conv", "IC": 32, "OC": 64, "K": 5, "H": 1,  "W": 1,  "MACs": 51200},
    {"name": "fc",    "type": "fc",   "IC": 64, "OC": 10, "K": 1, "H": 1,  "W": 1,  "MACs": 640},
]


def _ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def os_group_breakdown(IC: int, K: int, P: int, rows: int, wbram_stall: int = 0) -> dict:
    """Explicit cycle breakdown of ONE OS group.

    P   = valid output pixels in the group (8 full, W' mod 8 partial, 1 FC).
    rows = active output-channel rows (8 full, OC mod 8 partial).
    """
    passes = IC * K                     # IC channels x K kernel rows
    lead_in = P - 1                     # fill the P-column shift chain
    taps = K                            # kernel columns streamed per pass
    drain_per_pass = 1                  # last product -> accumulator
    compute = passes * (lead_in + taps + drain_per_pass)   # == IC*K*(P+K)
    clear = 1                           # accum_clear + first weight load
    result_drain = 2 * rows             # 2 cycles/row read-out
    return {
        "clear": clear,
        "passes": passes,
        "lead_in_per_pass": lead_in,
        "taps_per_pass": taps,
        "drain_per_pass": drain_per_pass,
        "compute": compute,
        "result_drain": result_drain,
        "weight_bram_stall": wbram_stall,
        "total": clear + compute + result_drain + wbram_stall,
    }


def os_conv_cycles(IC: int, OC: int, K: int, H_out: int, W_out: int, wbram_stall: int = 0) -> dict:
    """Generalized OS cycles for a VALID KxK conv layer (resolves G1/G2/G3/G6)."""
    groups = 0
    total = 0
    channel_groups = _ceil(OC, ARRAY)
    for g in range(channel_groups):
        rows = min(ARRAY, OC - ARRAY * g)          # partial channel group
        full = W_out // ARRAY
        rem = W_out % ARRAY                        # partial column group (VALID edge)
        per_row = full * os_group_breakdown(IC, K, ARRAY, rows, wbram_stall)["total"]
        if rem:
            per_row += os_group_breakdown(IC, K, rem, rows, wbram_stall)["total"]
        total += H_out * per_row
        groups += H_out * (full + (1 if rem else 0))
    return {"cycles": total, "channel_groups": channel_groups,
            "groups": groups, "partial_column_valid_pixels": rem or ARRAY}


def os_fc_cycles(IC: int, OC: int) -> dict:
    """Corrected OS FC schedule (RTL-verified bit-exact, FC_SCHEDULE_VALIDATION §2):
    rows = output neurons, IC features serialized; the 8-pixel lead-in vanishes."""
    channel_groups = _ceil(OC, ARRAY)
    return {"cycles": channel_groups * (IC + 2) + 2 * OC,
            "channel_groups": channel_groups, "groups": channel_groups}


def os_cycles(L: dict, wbram_stall: int = 0) -> dict:
    if L["K"] == 1:
        return os_fc_cycles(L["IC"], L["OC"])
    return os_conv_cycles(L["IC"], L["OC"], L["K"], L["H"], L["W"], wbram_stall)


def weight_bram_analysis(L: dict) -> dict:
    """Weight-BRAM read count, width, and latency for one layer (OS mode)."""
    OC, IC, K = L["OC"], L["IC"], L["K"]
    H_out, W_out = L["H"], L["W"]
    spatial = H_out * _ceil(W_out, ARRAY)
    taps = IC * K * K                            # reduction taps per group
    reads = taps * _ceil(OC, ARRAY) * spatial     # one 64-bit read per tap, per group
    weight_bytes = OC * IC * K * K               # useful int8 weight bytes (mode-indep.)
    return {
        "weight_values": weight_bytes,
        "weight_bytes_total": weight_bytes,       # 1 byte/int8 weight (no bias here)
        "bram_reads_per_layer": reads,
        "bram_read_width_bits": ARRAY * 8,        # 8 weights/read
        "bram_read_latency_cycles": 1,            # synchronous BRAM36
        "latency_hidden_by_lookahead": True,      # Phase-2 look-ahead technique
    }


def compute(wbram_stall: int = 0) -> dict:
    out = {
        "datapath": "OS-only (rows = output channels, cols = output pixels)",
        "array": {"rows": ARRAY, "cols": ARRAY, "PEs": PE_COUNT},
        "schedule": "1 + IC*K*(P+K) + 2*rows  (corrected FC_SCHEDULE_VALIDATION §6)",
        "weight_bram_stall_cycles": wbram_stall,
        "workloads": {},
    }
    for wl_name, layers, total_macs in (("lenet5", LENET, 416520), ("cifar10", CIFAR, 4493440)):
        wl = {"layers": {}, "total": {}}
        tot = 0
        for L in layers:
            n = L["name"]
            cyc = os_cycles(L, wbram_stall)
            bram = weight_bram_analysis(L)
            util = L["MACs"] / (PE_COUNT * cyc["cycles"])
            wl["layers"][n] = {
                "type": L["type"], "IC": L["IC"], "OC": L["OC"], "K": L["K"],
                "H_out": L["H"], "W_out": L["W"], "MACs": L["MACs"],
                "cycles": cyc["cycles"],
                "channel_groups": cyc["channel_groups"],
                "groups": cyc["groups"],
                "PE_utilization": util,
                "idle_PE_cycles": PE_COUNT * cyc["cycles"] - L["MACs"],
                "weight_bram": bram,
                "timing_status": timing_status(L),
            }
            tot += cyc["cycles"]
        wl["total"] = {"MACs": total_macs, "OS_cycles": tot,
                       "PE_utilization": total_macs / (PE_COUNT * tot)}
        out["workloads"][wl_name] = wl
    return out


def timing_status(L: dict) -> str:
    if L["K"] == 1:
        return "RTL-verified (bit-exact 1-pixel FC schedule)"
    if L["IC"] == 1 and L["K"] == 5:
        return "RTL-verified anchor (82 cyc) for full group; partial groups analytical"
    return "analytically derived (IC>1 and/or OC>8 generalization of the 82-cyc anchor)"


def _print(out: dict) -> None:
    print("=" * 92)
    print("Generalized OS-only cycle model — V2 Candidate A (corrected schedule)")
    print("=" * 92)
    for wl_name in ("lenet5", "cifar10"):
        wl = out["workloads"][wl_name]
        print(f"\n{wl_name.upper()}")
        print(f"{'layer':6} {'IC->OC':>9} {'MACs':>9} {'OS cyc':>8} {'util':>6} {'groups':>7}  status")
        print("-" * 92)
        for n, r in wl["layers"].items():
            print(f"{n:6} {str(r['IC'])+'->'+str(r['OC']):>9} {r['MACs']:>9} "
                  f"{r['cycles']:>8} {r['PE_utilization']*100:>5.1f}% {r['groups']:>7}  {r['timing_status'][:40]}")
        print("-" * 92)
        t = wl["total"]
        print(f"{'TOTAL':6} {'':>9} {t['MACs']:>9} {t['OS_cycles']:>8} {t['PE_utilization']*100:>5.1f}%")
    print("=" * 92)


def main() -> int:
    out = compute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    _print(out)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
