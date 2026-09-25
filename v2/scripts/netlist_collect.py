#!/usr/bin/env python3
"""Collect post-synthesis functional-simulation runs (run_netlist_sim.sh) into
v2/results/rtl_netlist.csv (source=post_synth_funcsim): one row per TB with checks, pass,
and the per-job RESULT totals vs model. Exit 1 on any failure."""
import re
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "model"))
from common import RESULTS_DIR, base_meta, write_results_csv  # noqa: E402

rows, ok = [], True
for tb in ("tb_gos_top", "tb_gos_top_backtoback"):
    log = V2 / "build" / "sim" / f"netlist_{tb}" / "run.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    passed = "TEST PASSED" in text and not re.search(r"^ERROR|ERROR:|FATAL|Fatal:|TEST FAILED", text, re.M)
    m = re.findall(r"TEST PASSED checks=(\d+)", text)
    res = [dict(kv.split("=", 1) for kv in l.split()[1:]) for l in text.splitlines() if l.startswith("RESULT ")]
    jobs = [r for r in res if "total" in r]
    exact = sum(r["total"] == r["model_total"] for r in jobs)
    ver = re.search(r"Vivado Simulator v?(\d{4}\.\d)", text)
    row = base_meta(layer="", source="post_synth_funcsim")
    row.update(vivado_version=ver.group(1) if ver else "", tb=tb, checks=m[-1] if m else "",
               passed=passed, jobs_completed=len(jobs), cycles_exact=exact,
               logits_checked=sum("logits_ok" in r for r in jobs),
               logits_ok=sum(r.get("logits_ok") == "1" for r in jobs),
               refused_jobs=sum("refused" in r for r in res), soft_resets=sum("soft_reset" in r for r in res))
    ok &= passed and exact == len(jobs) and row["logits_ok"] == row["logits_checked"]
    rows.append(row)
    print(f"  {tb}: passed={passed} checks={row['checks']} jobs={len(jobs)} cycles_exact={exact}")
out = write_results_csv(RESULTS_DIR / "rtl_netlist.csv", rows,
                        ["tb", "checks", "passed", "jobs_completed", "cycles_exact", "logits_checked", "logits_ok",
                         "refused_jobs", "soft_resets"])
print(f"netlist_collect: wrote {out}")
sys.exit(0 if ok else 1)
