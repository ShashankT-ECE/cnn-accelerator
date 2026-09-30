#!/usr/bin/env python3
"""Verify every v2/results/*.csv row is clean and its RTL/model/Vivado sources equal HEAD (DECISIONS D16).

Usage: check_results.py [--commit SHA]   (default: HEAD). Run after all result scripts
(regen_results.sh, run_unit_all.sh, run_core.sh, ooc_all.sh, vivado/build_*.sh + impl_collect.py)
and before committing the CSVs. The r2 training log is exempt (DECISIONS D9).

A row is valid when
  * git_dirty == False, and
  * `git diff --quiet <row git_commit> <commit> -- <sources of that CSV's producer>` succeeds
    (DECISIONS D16 as amended 2026-09-30: each CSV is compared only against the sources its producer
    reads — v2/rtl + v2/vivado for Vivado rows (impl/OOC; their collectors import v2/model/common.py
    for row metadata only), v2/rtl + v2/vivado + v2/model (+ EXTRA_SOURCES) for model and sim rows).
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
# D16 amendment (2026-09-30): producers that do not read v2/model (beyond row metadata) are compared
# against their own sources only, so a model-only change does not invalidate Vivado rows.
PRODUCER_SOURCES = {
    "impl_gos.csv": ("v2/rtl", "v2/vivado"),       # vivado/build_gos.sh + impl_collect.py
    "impl_shell.csv": ("v2/rtl", "v2/vivado"),
    "ooc_synth.csv": ("v2/rtl", "v2/vivado"),      # ooc_all.sh + ooc_collect.py
    # make_dpu_package.py / export_data.py / vai_quantize.py: v2/dpu + net_config, common, r2 checkpoint
    "dpu_model_accuracy.csv": ("v2/model/common.py", "v2/model/net_config.py", "v2/model/retrain"),
}
# dpu_model_accuracy.csv (D16 amendment 2, 2026-09-30): the laptop producer code under v2/dpu
# (make_dpu_package.py, export_data.py, vai_quantize.py, run_docker.sh and their helpers) — NOT the
# documentation (*.md) and NOT the board-side files the producer never executes (dpu_session.py is
# only copied into the package, deploy_dpu.sh only ships it, tests/). A command fix in the docs
# must not make model-accuracy rows stale (it did on 2026-09-30, commit 7bd7173).
DPU_MODEL_SOURCES = ("v2/dpu", ":(exclude)v2/dpu/*.md", ":(exclude)v2/dpu/dpu_session.py",
                     ":(exclude)v2/dpu/deploy_dpu.sh", ":(exclude)v2/dpu/tests")
# CSVs whose producer lives outside SOURCE_PATHS: those sources are compared too (only for these CSVs,
# so older rows of other CSVs are not made stale by a folder that did not exist when they were produced).
EXTRA_SOURCES = {"schedule_ablation.csv": ("v2/analysis",), "projection_16x16.csv": ("v2/analysis",),
                 "utilization_model.csv": ("v2/analysis",), "rtl_full10k.csv": ("v2/fullsim",),
                 "dpu_model_accuracy.csv": DPU_MODEL_SOURCES, "shapes_rtl.csv": ("v2/shapes",),
                 "limits_rtl.csv": ("v2/shapes",)}


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


def sources_for(name: str) -> tuple:
    return PRODUCER_SOURCES.get(name, SOURCE_PATHS) + EXTRA_SOURCES.get(name, ())


ok = True
for p in sorted((V2 / "results").glob("*.csv")):
    if p.name in EXEMPT:
        print(f"  {p.name}: exempt (training artifact)")
        continue
    rows = list(csv.DictReader(p.open()))
    bad = []
    for i, r in enumerate(rows):
        why = "git_dirty" if r.get("git_dirty") != "False" else sources_differ(
            r.get("git_commit", ""), sources_for(p.name))
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
scoped = "; ".join(f"{n}: {'+'.join(v)}" for n, v in sorted(PRODUCER_SOURCES.items()))
print(f"check_results: sources compared against {commit[:8]} (default {', '.join(SOURCE_PATHS)}; {extra}; "
      f"producer-scoped: {scoped})")
print("check_results:", "ALL CLEAN" if ok else "FAILURES (stale/dirty rows above)")
sys.exit(0 if ok else 1)
