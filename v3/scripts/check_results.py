#!/usr/bin/env python3
"""Verify every v3/results/*.csv row is clean and its producer's sources equal HEAD.

Adapted from v2/scripts/check_results.py @ 28dd2ad (V2 DECISIONS D16 + amendments), V3 DECISIONS D3.

Usage: check_results.py [--commit SHA] [--verbose]   (default: HEAD)

A row is valid when
  * git_dirty == False, and
  * `git diff --quiet <row git_commit> <commit> -- <sources of that CSV's producer>` succeeds.
Every CSV must be listed in PRODUCER_SOURCES (an unlisted CSV is a failure, so a new producer
cannot slip in unchecked). Aggregate rows with git_commit == "mixed" are checked against every
commit in input_git_commits. Only files directly in v3/results/ are checked
(v3/results/dryrun/ is gitignored and never data).
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
COMMON = ("v3/model/common.py",)
# CSV name -> source paths its producer reads (code + committed inputs). Extend with each new producer.
PRODUCER_SOURCES: dict[str, tuple] = {
    # WS5 feasibility (v3/feas)
    "feas_packing.csv": ("v3/feas/packing_model.py",),
    "feas_tb.csv": ("v3/feas/rtl", "v3/feas/tb", "v3/feas/run_feas_tb.sh", "v3/feas/tb_collect.py"),
    "feas_ooc.csv": ("v3/feas/rtl", "v3/feas/ooc_synth.tcl", "v3/feas/run_feas_ooc.sh", "v3/feas/ooc_collect.py",
                     "v3/scripts/vivado_guard.sh"),
    "device_resources.csv": ("v3/feas/device_query.tcl", "v3/feas/run_feas_ooc.sh", "v3/feas/ooc_collect.py"),
}
# CSVs that are not produced from repo code (e.g. training logs written on an external GPU are
# checked by their own sha256 manifest instead); listed with the reason.
EXEMPT: dict[str, str] = {
}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(V3), *args], capture_output=True, text=True)


ap = argparse.ArgumentParser()
ap.add_argument("--commit", default=None)
ap.add_argument("--verbose", action="store_true", help="list every failing row")
a = ap.parse_args()
commit = git("rev-parse", a.commit or "HEAD").stdout.strip()
top = git("rev-parse", "--show-toplevel").stdout.strip()

_same: dict[tuple, str] = {}


def sources_differ(row_commit: str, paths: tuple) -> str:
    key = (row_commit, paths)
    if key not in _same:
        if not row_commit:
            _same[key] = "no git_commit"
        elif git("cat-file", "-e", f"{row_commit}^{{commit}}").returncode != 0:
            _same[key] = f"unknown commit {row_commit[:8]}"
        else:
            r = subprocess.run(["git", "-C", top, "diff", "--quiet", row_commit, commit, "--", *paths])
            _same[key] = "" if r.returncode == 0 else f"sources differ @ {row_commit[:8]}"
    return _same[key]


ok = True
for p in sorted((V3 / "results").glob("*.csv")):
    if p.name in EXEMPT:
        print(f"  {p.name}: exempt ({EXEMPT[p.name]})")
        continue
    if p.name not in PRODUCER_SOURCES:
        print(f"  {p.name}: NO PRODUCER ENTRY in check_results.PRODUCER_SOURCES")
        ok = False
        continue
    srcs = PRODUCER_SOURCES[p.name] + COMMON
    rows = list(csv.DictReader(p.open()))
    bad = []
    for i, r in enumerate(rows):
        if r.get("git_dirty") != "False":
            why = "git_dirty"
        elif r.get("git_commit") == "mixed" and r.get("input_git_commits"):
            why = next((w for c in r["input_git_commits"].split() if (w := sources_differ(c, srcs))), "")
        else:
            why = sources_differ(r.get("git_commit", ""), srcs)
        if why:
            bad.append((i, r, why))
    commits = sorted({r.get("git_commit", "")[:7] for r in rows})
    ok &= rows != [] and not bad
    status = "clean" if rows and not bad else ("EMPTY" if not rows else f"STALE ({len(bad)}/{len(rows)} rows)")
    print(f"  {p.name}: {len(rows)} rows, {status}; row commits {','.join(commits)}")
    for i, r, why in bad[: None if a.verbose else 5]:
        ident = " ".join(f"{k}={r[k]}" for k in ("net", "layer", "top", "config", "clock_mhz") if r.get(k))
        print(f"      row {i + 2}: {why}  {ident}")
    if len(bad) > 5 and not a.verbose:
        print(f"      ... {len(bad) - 5} more (--verbose)")
print(f"check_results: sources compared against {commit[:8]}")
print("check_results:", "ALL CLEAN" if ok else "FAILURES (stale/dirty/unlisted rows above)")
sys.exit(0 if ok else 1)
