#!/usr/bin/env python3
"""Collect a bd_shell.tcl build directory into v2/results/impl_<tag>.csv (source=post_impl).

Usage: impl_collect.py <outdir> [<outdir> ...] [--csv <path>] [--allow-timing-fail]
  Each <outdir> is a bd_shell.tcl output directory (e.g. v2/vivado/out/shell, out/gos_200), holding
  summary.json, utilization(.hier).rpt, <name>.bit and <name>.hwh (+ power / static-check reports
  for full-design builds). One row per outdir. Default CSV: v2/results/impl_shell.csv.
Exit 1 if a summary or deliverable is missing, or (without --allow-timing-fail) WNS/WHS < 0.
power_w_estimate is Vivado's report_power estimate (vectorless defaults), not a measurement.
For paper-grade rows run from a clean, committed tree (git_dirty=False).
"""
import argparse
import json
import re
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "model"))
from common import RESULTS_DIR, base_meta, sha256_file, write_results_csv  # noqa: E402

FIELDS = ["top", "build_id", "pl_clk0_mhz_requested", "pl_clk0_mhz_actual", "pl_clk0_srcsel",
          "clb_luts", "clb_registers", "lut_as_memory", "dsps", "ramb36", "ramb18",
          "wns_ns", "whs_ns", "tns_ns", "ths_ns", "critical_warnings", "bit_sha256", "hwh_sha256",
          "timing_met", "strategy", "worst_path", "power_w_estimate",
          "core_luts", "core_registers", "core_lutram", "core_dsps", "core_ramb36", "core_ramb18",
          "methodology_critical", "methodology_warning", "drc_error", "drc_critical", "drc_warning",
          "unconstrained_endpoints", "latches", "comb_loops", "latch_loops", "check_timing_nonzero",
          "synth_latch", "synth_multidriven", "synth_undriven", "synth_removed", "synth_unwaived_groups",
          "synth_truncated", "synth_logs", "synth_scan_pass"]

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


def parse_core_share(path):
    """gos_core row of report_utilization -hierarchical (Total LUTs, LUTRAMs, FFs, RAMB36, RAMB18, DSP)."""
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    hdr = next((l for l in lines if "Instance" in l and "Total LUTs" in l), None)
    # instance u_core; module "gos_core" (OOC) or "<bd>_gos_top_0_0_gos_core" (block design)
    row = next((l for l in lines if l.strip().startswith("|") and l.split("|")[1].strip() == "u_core"
                and re.search(r"(^|_|\()gos_core\)?$", l.split("|")[2].strip())), None)
    if not hdr or not row:
        return {}
    h = [c.strip() for c in hdr.split("|")]
    v = [c.strip() for c in row.split("|")]
    m = dict(zip(h, v))
    get = lambda k: int(float(m.get(k, "0") or 0))
    return {"core_luts": get("Total LUTs"), "core_registers": get("FFs"), "core_lutram": get("LUTRAMs"),
            "core_dsps": get("DSP Blocks"), "core_ramb36": get("RAMB36"), "core_ramb18": get("RAMB18")}


def parse_static(outdir):
    out = {}
    sc = outdir / "static_counts.txt"
    if sc.exists():
        for l in sc.read_text().splitlines():
            kind, sev, n = l.split()
            key = {("methodology", "CRITICAL_WARNING"): "methodology_critical",
                   ("methodology", "WARNING"): "methodology_warning",
                   ("drc", "ERROR"): "drc_error", ("drc", "CRITICAL_WARNING"): "drc_critical",
                   ("drc", "WARNING"): "drc_warning"}.get((kind, sev))
            if key:
                out[key] = int(n)
    ct = outdir / "check_timing.rpt"
    if ct.exists():
        import re
        txt = ct.read_text()
        m = re.findall(r"There (?:are|is) (\d+) (?:register/latch pins with no clock|input ports with no input delay"
                       r"|ports with no output delay|unconstrained internal endpoints)", txt)
        out["unconstrained_endpoints"] = sum(int(x) for x in m) if m else 0
        loops = lambda kind: sum(int(x) for x in re.findall(rf"There (?:are|is) (\d+) {kind} loops?", txt))
        out["comb_loops"], out["latch_loops"] = loops("combinational"), loops("latch")
        nz = sorted({l.strip() for l in txt.splitlines()
                     if re.match(r"\s*There (?:are|is) [1-9]\d* ", l)})
        out["check_timing_nonzero"] = "; ".join(nz)
    logs = [outdir / n for n in ("synth_1.log", "synth_gos_top.log") if (outdir / n).exists()]
    if logs:
        import subprocess
        tot, passed = {}, True
        for sl in logs:
            r = subprocess.run([sys.executable, str(V2 / "scripts" / "synth_scan.py"), str(sl),
                                "--out", str(outdir / f"{sl.stem}_scan.txt")], capture_output=True, text=True)
            last = r.stdout.strip().splitlines()[-1]
            for k, v in (x.split("=") for x in last.split()[2:]):
                tot[k] = tot.get(k, 0) + int(v)
            passed &= r.returncode == 0
        out.update(synth_latch=tot["latch"], synth_multidriven=tot["multidriven"], synth_undriven=tot["undriven"],
                   synth_removed=tot["removed"], synth_unwaived_groups=tot["unwaived_groups"],
                   synth_truncated=tot["truncated"], synth_logs="+".join(l.name for l in logs),
                   synth_scan_pass=passed)
    pw = outdir / "power.rpt"
    if pw.exists():
        for l in pw.read_text().splitlines():
            if "Total On-Chip Power (W)" in l:
                out["power_w_estimate"] = float(l.split("|")[2].strip())
    ut = outdir / "utilization.rpt"
    if ut.exists():
        for l in ut.read_text().splitlines():
            cells = [c.strip() for c in l.split("|")]
            if len(cells) > 2 and cells[1].startswith("Register as Latch"):
                out["latches"] = int(float(cells[2]))
    return out


def collect(outdir, allow_fail):
    p = outdir / "summary.json"
    if not p.exists():
        print(f"impl_collect: missing {p}", file=sys.stderr)
        return None, False
    s = json.loads(p.read_text())
    bit, hwh = outdir / s["bit"], outdir / s["hwh"]
    for f in (bit, hwh):
        if not f.exists():
            print(f"impl_collect: missing {f}", file=sys.stderr)
            return None, False
    u = parse_util(outdir / "utilization.rpt")
    bit_sha = sha256_file(bit)
    met = s["wns_ns"] is not None and s["whs_ns"] is not None and s["wns_ns"] >= 0 and s["whs_ns"] >= 0
    row = base_meta(source="post_impl")
    row.update(top=s["top"], build_id=s["build_id"], pl_clk0_mhz_requested=s["pl_clk0_mhz_requested"],
               pl_clk0_mhz_actual=s["pl_clk0_mhz_actual"], pl_clk0_srcsel=s.get("pl_clk0_srcsel", ""),
               wns_ns=s["wns_ns"], whs_ns=s["whs_ns"], tns_ns=s["tns_ns"], ths_ns=s["ths_ns"],
               critical_warnings=s["critical_warnings"], bit_sha256=bit_sha, hwh_sha256=sha256_file(hwh),
               bitstream_sha256=bit_sha, vivado_version=s["vivado_version"],
               clock_mhz=s["pl_clk0_mhz_actual"], timing_met=met, strategy=s.get("strategy", ""),
               worst_path=s.get("worst_path", ""), **u,
               **parse_core_share(outdir / "utilization_hier.rpt"), **parse_static(outdir))
    print(f"  {s['top']} build {s['build_id']} pl_clk0 {s['pl_clk0_mhz_actual']} MHz  LUT {u['clb_luts']} "
          f"FF {u['clb_registers']} LUTRAM {u['lut_as_memory']} DSP {u['dsps']} RAMB36 {u['ramb36']} "
          f"RAMB18 {u['ramb18']}  WNS {s['wns_ns']} WHS {s['whs_ns']} TNS {s['tns_ns']} THS {s['ths_ns']}  "
          f"CW {s['critical_warnings']}  git_dirty={row['git_dirty']}")
    return row, (met or allow_fail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdirs", type=Path, nargs="+")
    ap.add_argument("--csv", type=Path, default=RESULTS_DIR / "impl_shell.csv")
    ap.add_argument("--allow-timing-fail", action="store_true",
                    help="record variants with WNS/WHS < 0 without failing (timing_met=False)")
    a = ap.parse_args()
    rows, ok = [], True
    for d in a.outdirs:
        row, good = collect(d, a.allow_timing_fail)
        ok &= good
        if row is not None:
            rows.append(row)
    if not rows:
        return 1
    out = write_results_csv(a.csv, rows, FIELDS)
    print(f"impl_collect: wrote {out} ({len(rows)} rows)")
    return 0 if ok and len(rows) == len(a.outdirs) else 1


if __name__ == "__main__":
    sys.exit(main())
