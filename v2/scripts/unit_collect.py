#!/usr/bin/env python3
"""Collect unit-TB results (v2/build/sim/<tb>/run.log) into v2/results/unit_tb.csv.

Usage: unit_collect.py <tb>:<rc>:<seconds> ...   (called by run_unit_all.sh)
Rows are source=rtl_sim with the usual metadata (common.base_meta).
"""
import re
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "model"))
from common import RESULTS_DIR, base_meta, write_results_csv  # noqa: E402

rows = []
for arg in sys.argv[1:]:
    tb, rc, secs = arg.split(":")
    log = V2 / "build" / "sim" / tb / "run.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    hdr = (V2 / "tb" / f"{tb}.sv").read_text().splitlines()[0]
    srcs = hdr.split("GOS_UNIT_TB:", 1)[1].split() if "GOS_UNIT_TB:" in hdr else []
    m = re.findall(r"TEST PASSED checks=(\d+)", text)
    ver = re.search(r"Vivado Simulator v?(\d{4}\.\d)", text) or re.search(r"v(\d{4}\.\d)", text)
    row = base_meta(layer="", source="rtl_sim", duration_s=secs)
    row.update(vivado_version=ver.group(1) if ver else "", tb=tb,
               dut=" ".join(s for s in srcs if s != "gos_pkg.sv"),
               checks=m[-1] if m else "", passed=(rc == "0" and bool(m)))
    rows.append(row)
out = write_results_csv(RESULTS_DIR / "unit_tb.csv", rows, ["tb", "dut", "checks", "passed"])
print(f"unit_collect: wrote {out} ({len(rows)} rows)")
sys.exit(0 if all(r["passed"] for r in rows) else 1)
