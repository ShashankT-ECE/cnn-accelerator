#!/usr/bin/env python3
"""Verification statistics of the V2 accelerator, derived only from v2/results/*.csv.

Usage (repo root, .venv):
    .venv/bin/python v2/scripts/verification_stats.py [--out PATH] [--results-dir DIR] [--no-pytest]

Writes one row per statistic to v2/results/verification_stats.csv (default; --out writes
elsewhere) with the standard metadata (common.base_meta / write_results_csv) and prints a
markdown table to stdout. No number is typed by hand: every value is computed here from a
results CSV (or, for the model test count, from `pytest --collect-only`).

Columns per row:
  activity, metric, value   the statistic, computed over rows with git_dirty=False only
  value_dirty               the same statistic over rows with git_dirty=True ('' if none)
  rows_used, rows_dirty     clean / dirty source rows the statistic was computed from
  label                     model | rtl_sim | post_synth | hw (the label its numbers carry)
  source_csv                the results CSV it comes from
  git_commits               the (8-char) git_commit values of the source rows used
  flag                      DIRTY (rows_dirty > 0: invalid for the paper), PENDING, NOT_RECORDED, FAIL
Provenance against HEAD (per-row source identity, DECISIONS D16) is checked separately by
v2/scripts/check_results.py; paper-grade stats come from a tree where that passes.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2 / "model"))
from common import REPO_ROOT, RESULTS_DIR, base_meta, write_results_csv  # noqa: E402

FIELDS = ["activity", "metric", "value", "value_dirty", "rows_used", "rows_dirty", "label",
          "source_csv", "git_commits", "flag"]
PYTEST_PATHS = ("v2/model", "v2/board/tests", "v2/scripts/tests", "v2/analysis/tests")  # existing ones are used


# ---------------------------------------------------------------- helpers
def is_true(v) -> bool:
    return str(v).strip() in ("True", "true", "1")


def num(v) -> float:
    return float(v) if str(v).strip() not in ("", "None", "nan") else 0.0


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return int(v) if v.is_integer() else round(v, 3)
    return v


def read_csv(results_dir: Path, name: str) -> list[dict] | None:
    p = results_dir / name
    if not p.exists():
        return None
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


class Stats:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, activity, metric, label, source_csv, rows, fn, net="", flag=""):
        """fn(rows) -> value; evaluated separately on clean and dirty rows."""
        clean = [r for r in rows if str(r.get("git_dirty", "")).strip() == "False"]
        dirty = [r for r in rows if str(r.get("git_dirty", "")).strip() != "False"]
        flags = [flag] if flag else []
        if dirty:
            flags.append("DIRTY")
        commits = sorted({r.get("git_commit", "")[:8] for r in rows if r.get("git_commit")})
        self._put(activity, metric, fmt(fn(clean)), fmt(fn(dirty)) if dirty else "", len(clean),
                  len(dirty), label, source_csv, ";".join(commits), ",".join(flags), net)

    def note(self, activity, metric, value, label, source_csv, flag, net="", commits=""):
        self._put(activity, metric, value, "", "", "", label, source_csv, commits, flag, net)

    def _put(self, activity, metric, value, value_dirty, n_clean, n_dirty, label, source_csv,
             commits, flag, net):
        self.rows.append(dict(activity=activity, metric=metric, value=value, value_dirty=value_dirty,
                              rows_used=n_clean, rows_dirty=n_dirty, label=label,
                              source_csv=source_csv, git_commits=commits, flag=flag, net=net))


def count(rows):
    return len(rows)


def count_if(pred):
    return lambda rows: sum(1 for r in rows if pred(r))


def all_of(pred):
    return lambda rows: bool(rows) and all(pred(r) for r in rows)


def total(col):
    return lambda rows: sum(num(r.get(col)) for r in rows)


def nets_of(rows):
    return list(dict.fromkeys(r.get("net", "") for r in rows))


def missing(st: Stats, activity: str, csv_name: str, label: str):
    st.note(activity, "rows", "missing", label, csv_name, "PENDING")


# ---------------------------------------------------------------- activities
def unit_tbs(st: Stats, rd: Path):
    name, lab, act = "unit_tb.csv", "rtl_sim", "unit TBs"
    rows = read_csv(rd, name)
    if rows is None:
        return missing(st, act, name, lab)
    st.add(act, "testbenches", lab, name, rows, count)
    st.add(act, "total checks", lab, name, rows, total("checks"))
    st.add(act, "testbenches passed", lab, name, rows, count_if(lambda r: is_true(r["passed"])))
    ok = all(is_true(r["passed"]) for r in rows)
    st.add(act, "all passed", lab, name, rows, all_of(lambda r: is_true(r["passed"])),
           flag="" if ok else "FAIL")
    st.add(act, "recorded runtime s (sum of duration_s)", lab, name, rows, total("duration_s"))
    # back-to-back jobs (tb_gos_top_backtoback is run as a unit TB)
    btb = [r for r in rows if r.get("tb") == "tb_gos_top_backtoback"]
    act = "back-to-back jobs (RTL)"
    if not btb:
        return st.note(act, "rows", "missing", lab, name, "PENDING")
    st.add(act, "checks", lab, name, btb, total("checks"))
    st.add(act, "passed", lab, name, btb, all_of(lambda r: is_true(r["passed"])))
    st.note(act, "jobs", "n/a", lab, name, "NOT_RECORDED (job count is not a unit_tb.csv column)")


def _cyc_ok(r):
    return (r.get("model_cycles") == r.get("rtl_cycles") and r.get("model_total") == r.get("rtl_total")
            and r.get("model_mac") == r.get("rtl_mac"))


def core_suites(st: Stats, rd: Path):
    lab = "rtl_sim"
    name = "rtl_cycles.csv"
    rows = read_csv(rd, name)
    if rows is None:
        missing(st, "core layer suite", name, lab)
        missing(st, "core fuzz suite", name, lab)
    else:
        for kind, act in (("layer", "core layer suite"), ("fuzz", "core fuzz suite")):
            sub = [r for r in rows if r.get("kind") == kind]
            what = "layer runs" if kind == "layer" else "fuzz shapes"
            st.add(act, what, lab, name, sub, count)
            st.add(act, "cycle matches (layer, total, MAC_ACTIVE = model)", lab, name, sub, count_if(_cyc_ok))
            st.add(act, "output matches", lab, name, sub, count_if(lambda r: is_true(r["out_ok"])))
            st.add(act, "runtime s (duration_s)", lab, name, sub,
                   lambda rs: max((num(r["duration_s"]) for r in rs), default=0.0))
    name, act = "rtl_network.csv", "core net suite"
    rows = read_csv(rd, name)
    if rows is None:
        missing(st, act, name, lab)
    else:
        for net in nets_of(rows):
            sub = [r for r in rows if r.get("net") == net]
            st.add(act, f"{net}: images", lab, name, sub, count, net=net)
            st.add(act, f"{net}: logit matches", lab, name, sub,
                   count_if(lambda r: is_true(r["logits_match"])), net=net)
            st.add(act, f"{net}: prediction matches", lab, name, sub,
                   count_if(lambda r: is_true(r["pred_match"])), net=net)
            st.add(act, f"{net}: cycle matches", lab, name, sub,
                   count_if(lambda r: is_true(r["cycles_ok"]) and r["model_total"] == r["rtl_total"]), net=net)
        st.add(act, "runtime s (duration_s)", lab, name, rows,
               lambda rs: max((num(r["duration_s"]) for r in rs), default=0.0))
    name, act = "rtl_checker.csv", "core checker suite"
    rows = read_csv(rd, name)
    if rows is None:
        missing(st, act, name, lab)
    else:
        st.add(act, "cases", lab, name, rows, count)
        st.add(act, "cases matching expected ERR_CODE", lab, name, rows, count_if(lambda r: is_true(r["ok"])))
        st.add(act, "refused cases", lab, name, rows, count_if(lambda r: not is_true(r["accepted"])))
        st.add(act, "runtime s (duration_s)", lab, name, rows,
               lambda rs: max((num(r["duration_s"]) for r in rs), default=0.0))


def netlist(st: Stats, rd: Path):
    name, lab, act = "rtl_netlist.csv", "post_synth", "netlist sim (post-synthesis funcsim)"
    rows = read_csv(rd, name)
    if rows is None:
        return missing(st, act, name, lab)
    for r0 in rows:
        tb = r0["tb"]
        sub = [r0]
        st.add(act, f"{tb}: passed", lab, name, sub, all_of(lambda r: is_true(r["passed"])))
        st.add(act, f"{tb}: checks", lab, name, sub, total("checks"))
        st.add(act, f"{tb}: jobs completed", lab, name, sub, total("jobs_completed"))
        st.add(act, f"{tb}: cycle-exact jobs", lab, name, sub, total("cycles_exact"))
        st.add(act, f"{tb}: logit-checked jobs ok / checked", lab, name, sub,
               lambda rs: f"{fmt(total('logits_ok')(rs))}/{fmt(total('logits_checked')(rs))}")
        st.add(act, f"{tb}: soft resets", lab, name, sub, total("soft_resets"))


def _adopted_b(net: str):
    try:
        import numpy as np
        from net_config import NET_CONFIGS
        return int(np.load(NET_CONFIGS[net]["hw_requant"])["B"])
    except Exception:  # noqa: BLE001 - missing frozen file / unknown net
        return None


def requant(st: Stats, rd: Path):
    name, lab, act = "requant_equivalence.csv", "model", "requant equivalence"
    rows = read_csv(rd, name)
    if rows is None:
        return missing(st, act, name, lab)
    vals = lambda rs: sum(num(r["values_checked_exact"]) + num(r["values_checked_saturated"]) for r in rs)  # noqa: E731
    for net in nets_of(rows):
        for b in sorted({r["B"] for r in rows if r["net"] == net}, key=int):
            sub = [r for r in rows if r["net"] == net and r["B"] == b]
            adopted = _adopted_b(net) == int(b)
            tag = f"{net} B={b}{' (adopted, hw_requant.npz)' if adopted else ''}"
            st.add(act, f"{tag}: channels", lab, name, sub, count, net=net)
            st.add(act, f"{tag}: values checked", lab, name, sub, vals, net=net)
            st.add(act, f"{tag}: mismatches", lab, name, sub, total("mismatches"), net=net,
                   flag="FAIL" if adopted and total("mismatches")(sub) else "")
            st.add(act, f"{tag}: channels with s out of range", lab, name, sub,
                   count_if(lambda r: not is_true(r.get("s_in_range", "True"))), net=net)


def model_checks(st: Stats, rd: Path):
    lab = "model"
    name, act = "final_layer_check.csv", "final-layer check"
    rows = read_csv(rd, name)
    if rows is None:
        missing(st, act, name, lab)
    else:
        for net in nets_of(rows):
            sub = [r for r in rows if r["net"] == net]
            st.add(act, f"{net}: images", lab, name, sub, total("images"), net=net)
            st.add(act, f"{net}: logits bit-exact", lab, name, sub, total("logits_bitexact"), net=net)
            st.add(act, f"{net}: argmax matches", lab, name, sub, total("argmax_match"), net=net)
    name, act = "golden_crosscheck.csv", "golden cross-check (gos_golden vs legacy)"
    rows = read_csv(rd, name)
    if rows is None:
        missing(st, act, name, lab)
    else:
        for net in nets_of(rows):
            sub = [r for r in rows if r["net"] == net]
            st.add(act, f"{net}: images", lab, name, sub,
                   lambda rs: max((num(r["images"]) for r in rs), default=0.0), net=net)
            st.add(act, f"{net}: tensors compared", lab, name, sub, count, net=net)
            st.add(act, f"{net}: tensors bit-identical on all images", lab, name, sub,
                   count_if(lambda r: r["bit_identical"] == r["images"]), net=net)
            st.add(act, f"{net}: tensors with identical predictions on all images", lab, name, sub,
                   count_if(lambda r: r["predictions_identical"] == r["images"]), net=net)
    name, act = "reference_accuracy.csv", "reference accuracy"
    rows = read_csv(rd, name)
    if rows is None:
        missing(st, act, name, lab)
    else:
        for r0 in rows:
            tag = f"{r0['reference_version']} {r0['precision']}"
            st.add(act, f"{tag}: images", lab, name, [r0], total("total"), net=r0["net"])
            st.add(act, f"{tag}: correct", lab, name, [r0], total("correct"), net=r0["net"])
            st.add(act, f"{tag}: accuracy %", lab, name, [r0], total("accuracy_pct"), net=r0["net"])


def pytest_count(st: Stats, paths=PYTEST_PATHS):
    act, lab, src = "model / driver pytest", "model", "pytest --collect-only"
    use = [p for p in paths if (REPO_ROOT / p).exists()]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *use],
                       cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    if not m or r.returncode not in (0, 5):
        return st.note(act, f"tests collected ({' '.join(use)})", "error", lab, src,
                       f"FAIL (pytest rc={r.returncode})")
    st.note(act, f"tests collected ({' '.join(use)})", int(m.group(1)), lab, src, "")


def board_a1(st: Stats, rd: Path):
    name, lab, act = "hw_a1_accuracy.csv", "hw", "board A1 (KV260)"
    rows = read_csv(rd, name)
    if rows is None:
        return st.note(act, "images", "pending", lab, name, "PENDING")
    hw = [r for r in rows if r.get("source") == "hw"]
    if len(hw) != len(rows):
        st.note(act, "non-hw rows ignored", len(rows) - len(hw), lab, name, "IGNORED")
    for net in nets_of(hw):
        sub = [r for r in hw if r["net"] == net]
        st.add(act, f"{net}: images", lab, name, sub, total("images"), net=net)
        st.add(act, f"{net}: logit-mismatch images", lab, name, sub, total("logit_mismatch_images"), net=net)
        st.add(act, f"{net}: prediction mismatches", lab, name, sub, total("pred_mismatches"), net=net)
        st.add(act, f"{net}: job errors", lab, name, sub, total("job_errors"), net=net)
        st.add(act, f"{net}: accelerator correct", lab, name, sub, total("hw_correct"), net=net)


# ---------------------------------------------------------------- output
def collect(results_dir: Path, with_pytest: bool = True) -> list[dict]:
    st = Stats()
    unit_tbs(st, results_dir)
    core_suites(st, results_dir)
    netlist(st, results_dir)
    requant(st, results_dir)
    model_checks(st, results_dir)
    if with_pytest:
        pytest_count(st)
    board_a1(st, results_dir)
    return st.rows


def to_markdown(stats: list[dict]) -> str:
    out = ["| Activity | Metric | Value | Label | Source CSV | Commits | Dirty rows | Flag |",
           "|---|---|---|---|---|---|---|---|"]
    for s in stats:
        dirty = f"{s['rows_dirty']} (value {s['value_dirty']})" if s["rows_dirty"] else ""
        out.append(f"| {s['activity']} | {s['metric']} | {s['value']} | {s['label']} | {s['source_csv']} "
                   f"| {s['git_commits']} | {dirty} | {s['flag']} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "verification_stats.csv")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="where the source CSVs are")
    ap.add_argument("--no-pytest", action="store_true", help="skip the pytest --collect-only count")
    a = ap.parse_args(argv)
    stats = collect(a.results_dir, with_pytest=not a.no_pytest)
    rows = []
    for s in stats:
        row = base_meta(net=s.pop("net"), source=s["label"])
        row.update(s)
        rows.append(row)
    path = write_results_csv(a.out, rows, FIELDS)
    print(to_markdown(stats))
    print(f"\nverification_stats: wrote {path} ({len(rows)} rows)")
    bad = [s for s in stats if "FAIL" in s["flag"]]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
