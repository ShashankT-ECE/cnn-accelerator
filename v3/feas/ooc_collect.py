#!/usr/bin/env python3
"""Collect V3 WS5 feasibility OOC results and the device query into v3/results/.

  feas_ooc.csv          one row per OOC run (source = 'OOC synth'): top, generics, clk_period_ns, clock_mhz,
                        lut, lutram, ff, carry8, dsp, ramb36, ramb18, uram, wns_ns, tns_ns, failing_endpoints,
                        timing_met
  device_resources.csv  XCK26 resource totals from the Vivado part properties (source = 'Vivado part query')
Reads v3/build/feas_ooc/<run>/summary.json and v3/build/feas_ooc/device_query/part_properties.txt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from common import REPO_ROOT, RESULTS_DIR, base_meta, write_results_csv  # noqa: E402

OOC = REPO_ROOT / "v3" / "build" / "feas_ooc"
PART_KEYS = ("LUT_ELEMENTS", "FLIPFLOPS", "DSP", "BLOCK_RAMS", "ULTRA_RAMS", "SLICES")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    a = ap.parse_args()
    out = Path(a.out_dir)

    dq = OOC / "device_query"
    ver = (dq / "vivado_version.txt").read_text().strip()
    props = {}
    for line in (dq / "part_properties.txt").read_text().splitlines():
        m = re.match(r"^(\w+)\s+\w+\s+\w+\s+(.*)$", line)
        if m:
            props[m.group(1)] = m.group(2).strip()
    rows = []
    for k in PART_KEYS:
        r = base_meta(source="Vivado part query")
        r.update(vivado_version=ver, part="xck26-sfvc784-2LV-c", resource=k, available=int(props[k]))
        rows.append(r)
    write_results_csv(out / "device_resources.csv", rows, ["part", "resource", "available"])

    rows = []
    for sj in sorted(OOC.glob("*/summary.json")):
        s = json.loads(sj.read_text())
        period = float(s["clk_period_ns"])
        r = base_meta(source="OOC synth")
        r.update(vivado_version=s["vivado_version"], clock_mhz=round(1000.0 / period, 3), run=sj.parent.name,
                 top=s["top"], part=s["part"], generics=s["generics"], clk_period_ns=period,
                 **{k: s[k] for k in ("lut", "lutram", "ff", "carry8", "dsp", "ramb36", "ramb18", "uram",
                                      "wns_ns", "tns_ns", "failing_endpoints")},
                 timing_met=(s["wns_ns"] is not None and s["wns_ns"] >= 0 and s["failing_endpoints"] == 0))
        rows.append(r)
    if not rows:
        sys.exit("no summary.json found")
    write_results_csv(out / "feas_ooc.csv", rows, [k for k in rows[0] if k not in base_meta()])
    for r in rows:
        print(f"{r['run']:28s} dsp={r['dsp']:4} lut={r['lut']:6} ff={r['ff']:6} b36={r['ramb36']:3} "
              f"uram={r['uram']:2} wns={r['wns_ns']} met={r['timing_met']}")


if __name__ == "__main__":
    main()
