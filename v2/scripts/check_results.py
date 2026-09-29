#!/usr/bin/env python3
"""Verify every v2/results/*.csv row is clean and its RTL/model/Vivado sources equal HEAD (DECISIONS D16).

Usage: check_results.py [--commit SHA]   (default: HEAD). Run after all result scripts
(regen_results.sh, run_unit_all.sh, run_core.sh, ooc_all.sh, vivado/build_*.sh + impl_collect.py)
and before committing the CSVs. The r2 training log is exempt (DECISIONS D9).

A row is valid when
  * git_dirty == False, and
  * `git diff --quiet <row git_commit> <commit> -- v2/rtl v2/vivado v2/model` succeeds
    (the row was produced from RTL, Vivado sources and model identical to <commit>).
Rows may therefore come from different commits. Failing rows are listed as STALE / DIRTY.
Only files directly in v2/results/ are checked (v2/results/dryrun/ is gitignored, never paper data).
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
EXEMPT = {"cifar10_retrain_log.csv"}
SOURCE_PATHS = ("v2/rtl", "v2/vivado", "v2/model")
# CSVs whose producer lives outside SOURCE_PATHS: those sources are compared too (only for these CSVs,
# so older rows of other CSVs are not made stale by a folder that did not exist when they were produced).
EXTRA_SOURCES = {"schedule_ablation.csv": ("v2/analysis",), "projection_16x16.csv": ("v2/analysis",),
                 "utilization_model.csv": ("v2/analysis",), "rtl_full10k.csv": ("v2/fullsim",),
                 "dpu_model_accuracy.csv": ("v2/dpu",)}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(V2), *args], capture_output=True, text=True)


ap = argparse.ArgumentParser()
ap.add_argument("--commit", default=None)
ap.add_argument("--verbose", action="store_true", help="list every failing row")
a = ap.parse_args()
commit = git("rev-parse", a.commit or "HEAD").stdout.strip()
top = git("rev-parse", "--show-toplevel").stdout.strip()

_same: dict[tuple, str] = {}   # (row commit, paths) -> "" (same sources) | reason


def sources_differ(row_commit: str, paths: tuple = SOURCE_PATHS) -> str:
    """'' if the row commit's source paths (default v2/rtl, v2/vivado, v2/model) equal <commit>'s, else a reason."""
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
for p in sorted((V2 / "results").glob("*.csv")):
    if p.name in EXEMPT:
        print(f"  {p.name}: exempt (training artifact)")
        continue
    rows = list(csv.DictReader(p.open()))
    bad = []
    for i, r in enumerate(rows):
        why = "git_dirty" if r.get("git_dirty") != "False" else sources_differ(
            r.get("git_commit", ""), SOURCE_PATHS + EXTRA_SOURCES.get(p.name, ()))
        if why:
            bad.append((i, r, why))
    commits = sorted({r.get("git_commit", "")[:7] for r in rows})
    ok &= rows != [] and not bad
    status = "clean" if rows and not bad else ("EMPTY" if not rows else f"STALE ({len(bad)}/{len(rows)} rows)")
    print(f"  {p.name}: {len(rows)} rows, {status}; row commits {','.join(commits)}")
    for i, r, why in bad[: None if a.verbose else 5]:
        ident = " ".join(f"{k}={r[k]}" for k in ("net", "layer", "top", "module", "tb", "clock_mhz")
                         if r.get(k))
        print(f"      row {i + 2}: {why}  {ident}")
    if len(bad) > 5 and not a.verbose:
        print(f"      ... {len(bad) - 5} more (--verbose)")
extra = "; ".join(f"{n}: +{'+'.join(v)}" for n, v in sorted(EXTRA_SOURCES.items()))
print(f"check_results: sources compared against {commit[:8]} ({', '.join(SOURCE_PATHS)}; {extra})")
print("check_results:", "ALL CLEAN" if ok else "FAILURES (stale/dirty rows above)")
sys.exit(0 if ok else 1)
