#!/usr/bin/env python3
"""Collect a bd_shell.tcl build directory into v2/results/impl_<tag>.csv (source=post_impl).

Usage: impl_collect.py <outdir> [--csv <path>]
  <outdir> is the bd_shell.tcl output directory (e.g. v2/vivado/out/shell), holding summary.json,
  utilization.rpt, <name>.bit and <name>.hwh. Default CSV: v2/results/impl_shell.csv.
Exit 1 if the summary is missing, a deliverable is missing, or WNS/WHS < 0.
For paper-grade rows run from a clean, committed tree (git_dirty=False).
"""
import argparse
import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "model"))
from common import RESULTS_DIR, base_meta, sha256_file, write_results_csv  # noqa: E402

FIELDS = ["top", "build_id", "pl_clk0_mhz_requested", "pl_clk0_mhz_actual", "pl_clk0_srcsel",
          "clb_luts", "clb_registers", "lut_as_memory", "dsps", "ramb36", "ramb18",
          "wns_ns", "whs_ns", "tns_ns", "ths_ns", "critical_warnings", "bit_sha256", "hwh_sha256"]

# report_utilization rows (Used column) -> csv field (same parser as ooc_collect.py)
RPT_ROWS = {"CLB LUTs": "clb_luts", "LUT as Memory": "lut_as_memory", "CLB Registers": "clb_registers",
            "DSPs": "dsps", "RAMB36/FIFO": "ramb36", "RAMB18": "ramb18"}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--csv", type=Path, default=RESULTS_DIR / "impl_shell.csv")
    a = ap.parse_args()
    p = a.outdir / "summary.json"
    if not p.exists():
        print(f"impl_collect: missing {p}", file=sys.stderr)
        return 1
    s = json.loads(p.read_text())
    bit, hwh = a.outdir / s["bit"], a.outdir / s["hwh"]
    for f in (bit, hwh):
        if not f.exists():
            print(f"impl_collect: missing {f}", file=sys.stderr)
            return 1
    u = parse_util(a.outdir / "utilization.rpt")
    bit_sha = sha256_file(bit)
    row = base_meta(source="post_impl")
    row.update(top=s["top"], build_id=s["build_id"], pl_clk0_mhz_requested=s["pl_clk0_mhz_requested"],
               pl_clk0_mhz_actual=s["pl_clk0_mhz_actual"], pl_clk0_srcsel=s.get("pl_clk0_srcsel", ""),
               wns_ns=s["wns_ns"], whs_ns=s["whs_ns"], tns_ns=s["tns_ns"], ths_ns=s["ths_ns"],
               critical_warnings=s["critical_warnings"], bit_sha256=bit_sha, hwh_sha256=sha256_file(hwh),
               bitstream_sha256=bit_sha, vivado_version=s["vivado_version"],
               clock_mhz=s["pl_clk0_mhz_actual"], **u)
    out = write_results_csv(a.csv, [row], FIELDS)
    met = s["wns_ns"] is not None and s["whs_ns"] is not None and s["wns_ns"] >= 0 and s["whs_ns"] >= 0
    print(f"  {s['top']} build {s['build_id']} pl_clk0 {s['pl_clk0_mhz_actual']} MHz  LUT {u['clb_luts']} "
          f"FF {u['clb_registers']} LUTRAM {u['lut_as_memory']} DSP {u['dsps']} RAMB36 {u['ramb36']} "
          f"RAMB18 {u['ramb18']}  WNS {s['wns_ns']} WHS {s['whs_ns']} TNS {s['tns_ns']} THS {s['ths_ns']}  "
          f"CW {s['critical_warnings']}  git_dirty={row['git_dirty']}")
    print(f"impl_collect: wrote {out}")
    return 0 if met else 1


if __name__ == "__main__":
    sys.exit(main())
