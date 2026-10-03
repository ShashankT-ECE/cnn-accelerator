"""V3 Phase 0 array DSE (WS4): cycle model + storage estimate sweep -> v3/results/dse.csv, dse_layers.csv.

Labels: cycles, utilisation and latency are "model" (v3/model/cycle_model.py, projected from the V2
RTL-derived constants); BRAM36/URAM counts are "estimate" (v3/model/mem_model.py). Assumptions P1..P15
are listed in cycle_model.ASSUMPTIONS and v3/analysis/DSE_NOTES.md. Device totals come only from
v3/results/device_resources.csv (empty utilisation columns when it does not exist).

Run (clean committed tree for data; otherwise --out v3/results/dryrun/):
    .venv/bin/python v3/analysis/dse.py [--out DIR]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))

import cycle_model as cm          # noqa: E402
import mem_model as mm            # noqa: E402
from common import RESULTS_DIR, base_meta, write_results_csv   # noqa: E402
from nets import NETS, layer_table                              # noqa: E402

ARRAYS = ((8, 8), (16, 16), (32, 16), (16, 32))          # (R rows = positions, C cols = output channels)
ROW_MAPS = ("x", "xy")
# mode set -> (enabled modes, split-K factors). "best" = os + split-K with any power-of-two S | R (hybrid:
# R/S positions x S k-slices) + dw; "best_pure" = os + split-K only with S = R (one position per tile) + dw.
MODE_SETS = {"os_only": (cm.MODES_OS_ONLY, None), "best": (cm.MODES_BEST, None), "best_pure": (cm.MODES_BEST, "R")}
STRIDE = ("half_rate", "phase_split")
DENSITIES = (1.0, 0.75, 0.5, 0.25)
# Packing variants: (packed, pack_kmax). kmax values are PLACEHOLDERS until WS5 measures the real limit.
PACK = ((False, None), (True, None), (True, 64), (True, 16))
PACK_EXTRACT_CYCLES = 1                                     # PLACEHOLDER (P11)
LABEL = "cycles/util/latency: model (projected); BRAM36/URAM: estimate"
ASSUMPTION_IDS = "P1-P15 (cycle_model.ASSUMPTIONS, v3/analysis/DSE_NOTES.md)"

DSE_FIELDS = [
    "label", "array", "R", "C", "packing", "pack_kmax", "pack_extract_cycles", "row_map", "modes", "splitk_s_allowed",
    "stride2", "residual", "gap", "density", "cycles", "issue_cycles", "stall_cycles", "extra_pass_cycles", "macs",
    "util", "latency_us_249p9975MHz", "latency_us_333p33MHz", "modes_used", "layers_drain_stalled",
    "splitk_wgt_bytes_per_cycle_max", "dw_act_bytes_per_cycle", "pe_dsps", "requant_lanes", "requant_dsps",
    "total_dsps", "dsp_available", "dsp_pct", "weight_raw_bytes", "weight_padded_bytes", "weight_word_bits",
    "weight_uram", "weight_bram36_if_bram", "act_n_buffers", "act_largest_tensor_bytes", "act_pair_max_bytes",
    "act_peak_live_bytes", "skip_max_bytes", "act_bram36", "qparam_bram36", "sparse_meta_bits",
    "sparse_meta_bram36", "bram36_total", "uram_total", "bram36_available", "bram36_pct", "uram_available",
    "uram_pct", "assumptions",
]
LAYER_FIELDS = [
    "label", "array", "R", "C", "packing", "row_map", "modes", "splitk_s_allowed", "stride2", "density", "kind", "ic", "oc", "oh",
    "ow", "k", "stride", "pad", "residual_from", "gap_after", "mode", "split_s", "row_map_used", "T", "issue",
    "stall", "drain", "drain_hidden", "c_pipe", "cycles", "os_cycles", "residual_pass", "gap_pass", "macs",
    "util", "ws_cycles_sketch",
]


def configs():
    """(cfg, density, pack_label) for the sweep; kmax variants only at density 1.0; plus fusion variants."""
    out = []
    for (R, C), (packed, kmax), rm, ms, st, d in itertools.product(
            ARRAYS, PACK, ROW_MAPS, MODE_SETS, STRIDE, DENSITIES):
        if kmax is not None and d != 1.0:
            continue
        if ms == "best_pure" and (d != 1.0 or kmax is not None):
            continue
        modes, sk = MODE_SETS[ms]
        cfg = cm.ArrayConfig(R=R, C=C, row_map=rm, modes=modes, stride2=st, packed=packed,
                             pack_kmax=kmax, pack_extract_cycles=PACK_EXTRACT_CYCLES,
                             splitk_s=(R,) if sk == "R" else None)
        out.append((cfg, d, ms))
    for (R, C) in ARRAYS:     # unfused residual + GAP reference rows
        cfg = cm.ArrayConfig(R=R, C=C, row_map="x", modes=cm.MODES_BEST, stride2="phase_split",
                             residual="separate", gap="separate")
        out.append((cfg, 1.0, "best"))
    return out


def _pct(used, avail):
    return "" if not avail else f"{100.0 * used / avail:.2f}"


def _cfg_cols(cfg: cm.ArrayConfig, d: float, ms: str) -> dict:
    return {"label": LABEL, "array": f"{cfg.R}x{cfg.C}", "R": cfg.R, "C": cfg.C,
            "packing": "yes" if cfg.packed else "no",
            "pack_kmax": "" if not cfg.packed else ("none" if cfg.pack_kmax is None else cfg.pack_kmax),
            "pack_extract_cycles": cfg.pack_extract_cycles if cfg.packed and cfg.pack_kmax else "",
            "row_map": cfg.row_map, "modes": ms,
            "splitk_s_allowed": " ".join(map(str, cm.splitk_factors(cfg))) if "splitk" in cfg.modes else "",
            "stride2": cfg.stride2, "residual": cfg.residual,
            "gap": cfg.gap, "density": d}


def dse_row(net: str, table: list[dict], cfg: cm.ArrayConfig, d: float, ms: str, dev: dict) -> tuple[dict, dict]:
    res = cm.net_cycles(table, cfg, density=d)
    st = mm.storage(table, cfg, density=d)
    w, a = st["weights"], st["acts"]
    used = Counter(r["mode"] for r in res["layers"])
    total_dsps = cfg.pe_dsps + cfg.requant_dsps
    row = {**base_meta(net, "total", source="model"), **_cfg_cols(cfg, d, ms),
           "cycles": res["cycles"], "issue_cycles": sum(r["issue"] for r in res["layers"]),
           "stall_cycles": sum(r["stall"] for r in res["layers"]),
           "extra_pass_cycles": res["extra_pass_cycles"], "macs": res["macs"], "util": f"{res['util']:.6f}",
           "latency_us_249p9975MHz": f"{res['latency_us'][249.9975]:.4f}",
           "latency_us_333p33MHz": f"{res['latency_us'][333.33]:.4f}",
           "modes_used": ";".join(f"{k}:{v}" for k, v in sorted(used.items())),
           "layers_drain_stalled": sum(1 for r in res["layers"] if r["stall"] > 0),
           "splitk_wgt_bytes_per_cycle_max": max((r.get("wgt_bytes_per_cycle_needed", 0) for r in res["layers"]),
                                                 default=0) or "",
           "dw_act_bytes_per_cycle": max((r.get("act_bytes_per_cycle_needed", 0) for r in res["layers"]),
                                         default=0) or "",
           "pe_dsps": cfg.pe_dsps, "requant_lanes": cfg.lanes, "requant_dsps": cfg.requant_dsps,
           "total_dsps": total_dsps, "dsp_available": dev["DSP"] or "", "dsp_pct": _pct(total_dsps, dev["DSP"]),
           "weight_raw_bytes": w["raw_bytes"], "weight_padded_bytes": w["padded_bytes"],
           "weight_word_bits": w["word_bits"], "weight_uram": w["uram"], "weight_bram36_if_bram": w["bram36"],
           "act_n_buffers": a["n_buffers"], "act_largest_tensor_bytes": a["largest_tensor_bytes"],
           "act_pair_max_bytes": a["pair_max_bytes"], "act_peak_live_bytes": a["peak_live_bytes"],
           "skip_max_bytes": a["skip_max_bytes"], "act_bram36": a["bram36"],
           "qparam_bram36": st["qparams"]["bram36"], "sparse_meta_bits": w["meta_bits"],
           "sparse_meta_bram36": w["meta_bram36"], "bram36_total": st["bram36"], "uram_total": st["uram"],
           "bram36_available": dev["BRAM36"] or "", "bram36_pct": _pct(st["bram36"], dev["BRAM36"]),
           "uram_available": dev["URAM"] or "", "uram_pct": _pct(st["uram"], dev["URAM"]),
           "assumptions": ASSUMPTION_IDS}
    return row, res


def layer_rows(net, table, cfg, d, ms, res) -> list[dict]:
    rows = []
    for L, r in zip(table, res["layers"]):
        ws = cm.ws_layer(L, cfg)
        rows.append({**base_meta(net, L["layer"], source="model"), **_cfg_cols(cfg, d, ms),
                     "label": "model (projected)",
                     **{k: L[k] for k in ("kind", "ic", "oc", "oh", "ow", "k", "stride", "pad", "residual_from")},
                     "gap_after": L["gap"], "mode": r["mode"], "split_s": r["split_s"],
                     "row_map_used": r["row_map_used"], "T": r["T"], "issue": r["issue"], "stall": r["stall"],
                     "drain": r["drain"], "drain_hidden": r["drain_hidden"], "c_pipe": r["c_pipe"],
                     "cycles": r["cycles"], "os_cycles": r["os_cycles"], "residual_pass": r["residual_pass"],
                     "gap_pass": r["gap_pass"], "macs": r["macs"], "util": f"{r['util']:.6f}",
                     "ws_cycles_sketch": ws["cycles"] if ws else ""})
    return rows


def is_key(cfg: cm.ArrayConfig, d: float) -> bool:
    return (not cfg.packed and cfg.stride2 == "phase_split" and d == 1.0 and cfg.residual == "fused"
            and cfg.gap == "fused")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=RESULTS_DIR, help="output directory (default v3/results)")
    a = ap.parse_args(argv)
    dev = mm.device_totals()
    tables = {n: layer_table(n) for n in NETS}
    rows, lrows = [], []
    for net in NETS:
        for cfg, d, ms in configs():
            row, res = dse_row(net, tables[net], cfg, d, ms, dev)
            rows.append(row)
            if is_key(cfg, d):
                lrows += layer_rows(net, tables[net], cfg, d, ms, res)
    p1 = write_results_csv(a.out / "dse.csv", rows, ["net"] + DSE_FIELDS)
    p2 = write_results_csv(a.out / "dse_layers.csv", lrows, ["net", "layer"] + LAYER_FIELDS)
    print(f"device totals: {dev} ({'from ' + str(mm.DEVICE_CSV) if any(dev.values()) else 'absent -> None'})")
    print(f"{'net':12} {'array':6} {'pack':4} {'map':3} {'modes':8} {'cycles':>8} {'util':>8} {'us@250':>9}")
    for r in rows:
        if r["density"] == 1.0 and r["stride2"] == "phase_split" and r["pack_kmax"] in ("", "none") \
                and r["residual"] == "fused":
            print(f"{r['net']:12} {r['array']:6} {r['packing']:4} {r['row_map']:3} {r['modes']:8} "
                  f"{r['cycles']:>8} {r['util']:>8} {r['latency_us_249p9975MHz']:>9}")
    print(f"Wrote {p1} ({len(rows)} rows) and {p2} ({len(lrows)} rows); {LABEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
