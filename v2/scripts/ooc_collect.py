#!/usr/bin/env python3
"""Collect v2/build/ooc/<top>/summary.json into v2/results/ooc_synth.csv.

Usage: ooc_collect.py <top> ...   (called by ooc_all.sh). source=post_synth_ooc.
Exit 1 if any module is missing a summary or has WNS < 0.
"""
import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "model"))
from common import RESULTS_DIR, base_meta, write_results_csv  # noqa: E402

FIELDS = ["top", "part", "clk_period_ns", "generics",
          "clb_luts", "lut_as_memory", "clb_registers", "carry8", "dsps", "ramb36", "ramb18",
          "prim_lut", "prim_lutram", "prim_ff", "prim_dsp48e2", "prim_ramb36", "prim_ramb18",
          "wns_ns", "tns_ns", "failing_endpoints", "timing_met"]

# report_utilization rows (Used column) -> csv field
RPT_ROWS = {"CLB LUTs": "clb_luts", "LUT as Memory": "lut_as_memory", "CLB Registers": "clb_registers",
            "CARRY8": "carry8", "DSPs": "dsps", "RAMB36/FIFO": "ramb36", "RAMB18": "ramb18"}


def parse_util(path):
    out = {}
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 2:
            name = cells[1].rstrip("*").strip()
            if name in RPT_ROWS and RPT_ROWS[name] not in out:
                out[RPT_ROWS[name]] = int(float(cells[2]))
    missing = set(RPT_ROWS.values()) - set(out)
    assert not missing, f"{path}: missing {missing}"
    return out
rows, ok = [], True
for top in sys.argv[1:]:
    p = V2 / "build" / "ooc" / top / "summary.json"
    if not p.exists():
        print(f"ooc_collect: missing {p}", file=sys.stderr)
        ok = False
        continue
    s = json.loads(p.read_text())
    met = s["wns_ns"] is not None and s["wns_ns"] >= 0
    ok &= met
    row = base_meta(layer="", source="post_synth_ooc")
    u = parse_util(p.parent / "utilization.rpt")
    row.update(top=s["top"], part=s["part"], clk_period_ns=s["clk_period_ns"], generics=s["generics"],
               prim_lut=s["lut"], prim_lutram=s["lutram"], prim_ff=s["ff"], prim_dsp48e2=s["dsp"],
               prim_ramb36=s["ramb36"], prim_ramb18=s["ramb18"], wns_ns=s["wns_ns"], tns_ns=s["tns_ns"],
               failing_endpoints=s["failing_endpoints"], vivado_version=s["vivado_version"],
               clock_mhz=round(1000 / s["clk_period_ns"], 3), timing_met=met, **u)
    rows.append(row)
    print(f"  {top:16s} LUT {u['clb_luts']:6d} LUTRAM {u['lut_as_memory']:4d} FF {u['clb_registers']:6d} "
          f"CARRY8 {u['carry8']:4d} DSP {u['dsps']:3d} RAMB36 {u['ramb36']:3d} RAMB18 {u['ramb18']:3d} "
          f"WNS {s['wns_ns']}")
out = write_results_csv(RESULTS_DIR / "ooc_synth.csv", rows, FIELDS)
print(f"ooc_collect: wrote {out} ({len(rows)} rows)")
sys.exit(0 if ok else 1)
