"""EXPERIMENTS A4 (model side): per-layer PE-array utilization of the frozen 8x8 design.

For every net in NETS and every layer (numbers from v2/model/gos_cycle_model.py and
net_config.py, imported read-only; source=model):

    T              = ceil(OC/8) * OH * ceil(OW/8)             tiles
    MAC_ACTIVE     = T * K                                    cycles with a mac-enable
    LAYER_CYC      = T * K + C_PIPE                           cycle model (RTL-derived C_PIPE)
    macs_useful    = OC * OH * OW * K                         true MACs of the layer
    macs_slots     = 64 * T * K                               array slots while active
    util_spatial   = macs_useful / macs_slots                 (= eff_oc * eff_x, exact)
    util_temporal  = T * K / LAYER_CYC                        (= MAC_ACTIVE / LAYER_CYC)
    util_total     = macs_useful / (64 * LAYER_CYC)           (= spatial * temporal)

Spatial loss decomposition (exact, with Fraction):
    eff_oc = OC / (8*ceil(OC/8))       OC tail: columns (output channels) padded to 8
    eff_x  = OW / (8*ceil(OW/8))       x tail: rows (output x positions) padded to 8
    slots lost to x tail          = macs_slots * (1 - eff_x)
    slots lost to OC tail         = macs_slots * eff_x * (1 - eff_oc)
    slots lost to pipeline fill   = 64 * C_PIPE          (temporal)
For OH = OW = 1 layers eff_x = 1/8 exactly: only 1 of the 8 array rows carries a real
output pixel, so util_spatial <= 1/8 (strictly lower with an OC tail). The bound is
derived from the layer shapes and asserted, not typed.

Also renders UTILIZATION.md (short explanation; every number read back from the CSV).

Run:  .venv/bin/python v2/analysis/utilization.py [--out-dir DIR] [--md PATH]
"""
from __future__ import annotations

import csv
import sys
from fractions import Fraction
from pathlib import Path

import _setup
from _setup import fmt

from common import base_meta, write_results_csv
from net_config import NET_CONFIGS, NETS
import gos_cycle_model as cm

ARRAY = cm.ARRAY
CSV_NAME = "utilization_model.csv"
DEFAULT_MD = _setup.RESULTS_DIR / "utilization_model.md"

FIELDS = ["IC", "OC", "OH", "OW", "K", "oc_tiles", "ow_tiles", "T", "mac_active", "c_pipe",
          "layer_cyc", "cycle_share", "macs_useful", "macs_slots", "util_spatial",
          "util_temporal", "util_total", "eff_oc", "eff_x", "one_by_one_output",
          "spatial_bound", "slots_lost_x_tail", "slots_lost_oc_tail", "slots_lost_pipe",
          "loss_cause", "util_spatial_exact", "util_total_exact"]


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def layer_util(L: dict, n_layers_cycles: int) -> dict:
    r = cm.layer_cycles(L)                       # frozen model (C_PIPE from RTL)
    T, K = r["T"], r["K"]
    mac_active, layer_cyc = r["mac_active"], r["cycles"]
    useful = L["OC"] * L["OH"] * L["OW"] * K
    slots = ARRAY * ARRAY * T * K
    assert useful == r["macs"] and mac_active == T * K

    eff_oc = Fraction(L["OC"], ARRAY * _ceil(L["OC"], ARRAY))
    eff_x = Fraction(L["OW"], ARRAY * _ceil(L["OW"], ARRAY))
    spatial = Fraction(useful, slots)
    temporal = Fraction(mac_active, layer_cyc)
    total = Fraction(useful, ARRAY * ARRAY * layer_cyc)
    assert spatial == eff_oc * eff_x, "spatial utilization must factor into OC and x tails"
    assert total == spatial * temporal
    assert float(spatial) == r["util_theoretical"]

    one_by_one = L["OH"] == 1 and L["OW"] == 1
    bound = Fraction(1, ARRAY) if one_by_one else Fraction(1)
    if one_by_one:
        assert eff_x == Fraction(1, ARRAY)
        assert spatial <= bound and (spatial < bound) == (eff_oc < 1)

    lost_x = slots * (1 - eff_x)
    lost_oc = slots * eff_x * (1 - eff_oc)
    assert lost_x.denominator == 1 and lost_oc.denominator == 1
    assert useful + lost_x + lost_oc == slots
    lost_pipe = ARRAY * ARRAY * (layer_cyc - mac_active)

    causes = []
    if one_by_one:
        causes.append(f"1x1 output: 1 of {ARRAY} rows active")
    elif eff_x < 1:
        causes.append(f"x tail: OW={L['OW']} padded to {ARRAY * _ceil(L['OW'], ARRAY)}")
    if eff_oc < 1:
        causes.append(f"OC tail: OC={L['OC']} padded to {ARRAY * _ceil(L['OC'], ARRAY)}")
    causes.append("pipeline fill C_PIPE")
    return {
        **{k: L[k] for k in ("IC", "OC", "OH", "OW")},
        "K": K, "oc_tiles": r["oc_tiles"], "ow_tiles": r["ow_tiles"], "T": T,
        "mac_active": mac_active, "c_pipe": r["c_pipe"], "layer_cyc": layer_cyc,
        "cycle_share": fmt(Fraction(layer_cyc, n_layers_cycles)),
        "macs_useful": useful, "macs_slots": slots,
        "util_spatial": fmt(spatial), "util_temporal": fmt(temporal), "util_total": fmt(total),
        "eff_oc": fmt(eff_oc), "eff_x": fmt(eff_x), "one_by_one_output": one_by_one,
        "spatial_bound": fmt(bound), "slots_lost_x_tail": int(lost_x),
        "slots_lost_oc_tail": int(lost_oc), "slots_lost_pipe": lost_pipe,
        "loss_cause": "; ".join(causes),
        "util_spatial_exact": str(spatial), "util_total_exact": str(total),
    }


def net_rows(net: str) -> list[dict]:
    layers = NET_CONFIGS[net]["layers"]
    res = cm.net_cycles(layers)
    total_cyc = res["total"]["cycles"]           # includes C_START + C_DONE
    rows = [{**base_meta(net, L["name"], source="model"), **layer_util(L, total_cyc)}
            for L in layers]
    useful = sum(r["macs_useful"] for r in rows)
    slots = sum(r["macs_slots"] for r in rows)
    mac_active = sum(r["mac_active"] for r in rows)
    assert mac_active == res["total"]["mac_active"]
    ctrl = total_cyc - sum(r["layer_cyc"] for r in rows)          # C_START + C_DONE
    assert ctrl == (cm.C_START or 0) + (cm.C_DONE or 0)
    spatial = Fraction(useful, slots)
    total = Fraction(useful, ARRAY * ARRAY * total_cyc)
    rows.append({
        **base_meta(net, "total", source="model"),
        "T": res["total"]["T"], "mac_active": mac_active, "c_pipe": cm.C_PIPE,
        "layer_cyc": total_cyc, "cycle_share": fmt(1), "macs_useful": useful,
        "macs_slots": slots, "util_spatial": fmt(spatial),
        "util_temporal": fmt(Fraction(mac_active, total_cyc)), "util_total": fmt(total),
        "slots_lost_x_tail": sum(r["slots_lost_x_tail"] for r in rows),
        "slots_lost_oc_tail": sum(r["slots_lost_oc_tail"] for r in rows),
        "slots_lost_pipe": ARRAY * ARRAY * (total_cyc - mac_active),
        "loss_cause": f"layer_cyc = TOTAL_CYC (sum of LAYER_CYC + C_START {cm.C_START} "
                      f"+ C_DONE {cm.C_DONE})",
        "util_spatial_exact": str(spatial), "util_total_exact": str(total),
    })
    return rows


def build_rows() -> list[dict]:
    return [r for net in NETS for r in net_rows(net)]


# ---- markdown (numbers rendered from the written CSV) ------------------------------

def _pct(s: str) -> str:
    return f"{100 * float(s):.1f}%"


def render_md(csv_path: Path) -> str:
    rows = list(csv.DictReader(csv_path.open()))
    commits = sorted({r["git_commit"][:8] for r in rows})
    dirty = sorted({r["git_dirty"] for r in rows})
    out = ["# PE-array utilization per layer (model)", "",
           f"Rendered by `v2/analysis/utilization.py` from `{csv_path.name}` "
           f"(source=model; rows git_commit {', '.join(commits)}, git_dirty {', '.join(dirty)})."
           + ("" if dirty == ["False"] else
              " **Rendered from a dirty tree: not paper-valid; regenerate at the freeze.**"),
           "", "Definitions: spatial = OC*OH*OW*K / (64*T*K); temporal = T*K / LAYER_CYC "
           "(MAC_ACTIVE/LAYER_CYC); total = spatial x temporal. All numbers are the analytical "
           "cycle model, not RTL or hardware measurements.", "",
           "| net | layer | OC | OH x OW | K | T | LAYER_CYC | cycle share | eff_oc | eff_x "
           "| spatial | temporal | total | loss |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        if r["layer"] == "total":
            out.append(f"| {r['net']} | **total** | | | | {r['T']} | {r['layer_cyc']} | | | "
                       f"| {_pct(r['util_spatial'])} | {_pct(r['util_temporal'])} "
                       f"| **{_pct(r['util_total'])}** | TOTAL_CYC basis |")
        else:
            out.append(f"| {r['net']} | {r['layer']} | {r['OC']} | {r['OH']}x{r['OW']} | {r['K']} "
                       f"| {r['T']} | {r['layer_cyc']} | {_pct(r['cycle_share'])} "
                       f"| {_pct(r['eff_oc'])} | {_pct(r['eff_x'])} | {_pct(r['util_spatial'])} "
                       f"| {_pct(r['util_temporal'])} | {_pct(r['util_total'])} | {r['loss_cause']} |")
    one = [r for r in rows if r["one_by_one_output"] == "True"]
    bound = sorted({r["spatial_bound"] for r in one})
    out += ["", "## Why the 1x1-output layers are at most 1/8", "",
            "Array rows are 8 consecutive output x positions of one output row and columns are "
            "8 output channels (ARCH_SPEC Datapath). A layer whose output map is 1x1 has one "
            "output pixel per channel, so only row 0 carries a real pixel; the other 7 rows "
            "compute masked outputs. Its spatial utilization is therefore at most "
            f"{_pct(bound[0]) if bound else 'n/a'} (eff_x = 1/8), lower when OC is not a "
            "multiple of 8. Affected layers (derived from the shapes):", ""]
    for r in one:
        out.append(f"- {r['net']} {r['layer']}: spatial {_pct(r['util_spatial'])} "
                   f"(exact {r['util_spatial_exact']}), {_pct(r['cycle_share'])} of the net's "
                   f"cycles.")
    tot = {r["net"]: r for r in rows if r["layer"] == "total"}
    out += ["", "## Where the array slots go (per net, share of 64 x TOTAL_CYC)", ""]
    for net, t in tot.items():
        denom = 64 * int(t["layer_cyc"])
        parts = [("useful MACs", int(t["macs_useful"])),
                 ("x tail (incl. 1x1 outputs)", int(t["slots_lost_x_tail"])),
                 ("OC tail", int(t["slots_lost_oc_tail"])),
                 ("pipeline fill + start", int(t["slots_lost_pipe"]))]
        assert sum(v for _, v in parts) == denom
        one_share = sum(int(r["layer_cyc"]) for r in one if r["net"] == net) / int(t["layer_cyc"])
        out.append(f"- {net}: " + ", ".join(f"{n} {100 * v / denom:.1f}%" for n, v in parts)
                   + f"; 1x1-output layers take {100 * one_share:.1f}% of the cycles.")
    out += ["", "Temporal utilization is close to 1 because the per-layer overhead is the fixed "
            "C_PIPE fill/flush (no per-pass lead-in or drain, DECISIONS D7); the utilization loss "
            "is almost entirely spatial (tile padding).", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = _setup.out_dir_parser(__doc__.splitlines()[0])
    ap.add_argument("--md", type=Path, default=DEFAULT_MD, help="markdown output path")
    a = ap.parse_args(argv)
    rows = build_rows()
    path = write_results_csv(a.out_dir / CSV_NAME, rows, ["net", "layer"] + FIELDS)
    a.md.parent.mkdir(parents=True, exist_ok=True)
    a.md.write_text(render_md(path))
    for r in rows:
        print(f"{r['net']:8} {r['layer']:6} T={r['T']:>4} cyc={r['layer_cyc']:>7} "
              f"spatial={r['util_spatial']} temporal={r['util_temporal']} "
              f"total={r['util_total']}")
    print(f"Wrote {path} and {a.md} (model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
