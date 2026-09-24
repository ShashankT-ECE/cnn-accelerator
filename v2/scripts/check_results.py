#!/usr/bin/env python3
"""Verify every v2/results/*.csv row is clean (git_dirty=False) and from one commit.

Usage: check_results.py [--commit SHA]   (default: HEAD). Run after all result scripts
(regen_results.sh, run_unit_all.sh, run_core.sh, ooc_all.sh, vivado/build_shell.sh)
and before committing the CSVs. The r2 training log is exempt (DECISIONS D9).
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
EXEMPT = {"cifar10_retrain_log.csv"}

ap = argparse.ArgumentParser()
ap.add_argument("--commit", default=None)
a = ap.parse_args()
commit = a.commit or subprocess.run(["git", "-C", str(V2), "rev-parse", "HEAD"], capture_output=True,
                                    text=True, check=True).stdout.strip()
ok = True
for p in sorted((V2 / "results").glob("*.csv")):
    if p.name in EXEMPT:
        print(f"  {p.name}: exempt (training artifact)")
        continue
    rows = list(csv.DictReader(p.open()))
    bad = [r for r in rows if r.get("git_dirty") != "False" or r.get("git_commit") != commit]
    status = "clean" if rows and not bad else f"NOT CLEAN ({len(bad)}/{len(rows)} rows)"
    ok &= rows != [] and not bad
    print(f"  {p.name}: {len(rows)} rows, {status} @ {commit[:7]}")
print("check_results:", "ALL CLEAN" if ok else "FAILURES")
sys.exit(0 if ok else 1)
