#!/usr/bin/env python3
"""3-session repeatability: combine the per-session statistics of session_index 1..3.

    python3 aggregate_sessions.py [--root results/] [--indices 1 2 3]
    python3 aggregate_sessions.py --root ../results/dryrun          # dry-run rows (dryrun_model)

Layout (run_sessions.py --session-index K): K = 1 -> <root>/, K >= 2 -> <root>/rep<K>/.
Per registered metric (METRICS: latency medians, power repeat means, f_meas, accuracy, cycles)
and key (net, phase, kind, ...), the per-session value is the one statistic that session reports
(e.g. the B3 median, the B1 mean over repeats); across sessions: mean of the per-session values,
between-session SD (ddof=1), min, max, range, CV % (stats.between_sessions).
RULES: rows from different bitstreams (bitstream_sha256) or clocks (clock_mhz, 0.1 MHz) are
NEVER combined: they form separate groups; a row's session_index column must equal its
directory's index (else REFUSED); source must be uniform (hw / cpu_board, or dry-run sources);
the output carries git_dirty = any input dirty and paper_grade = every input paper-grade.
Writes <root>/hw_repeatability.csv (one row per metric x key x bitstream x clock).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import board_common as bc
import stats

# (file glob, row filter, key columns, metric columns)
METRICS = [
    ("hw_fclk_cal_s*.csv", None, (), ("f_meas_mhz",)),
    ("hw_a1_accuracy.csv", None, ("net",), ("hw_accuracy_pct", "logit_mismatch_images")),
    ("hw_a2_a3_cycles.csv", lambda r: r.get("layer") == "total", ("net",),
     ("hw_cycles", "hw_us", "wall_us_median")),
    ("hw_b3_breakdown.csv", None, ("net", "phase"), ("median_us", "p95_us")),
    ("hw_b3_breakdown_fast.csv", None, ("net", "phase"), ("median_us", "p95_us")),
    ("hw_b3_breakdown_cpu.csv", None, ("net", "phase"), ("median_us", "p95_us")),
    ("hw_b1_power_ina260_summary_*.csv", lambda r: r.get("row_kind") == "mean", ("net",),
     ("accel_p_idle_w", "accel_dp_w", "control_dp_w", "accel_dp_net_w", "cpu_dp_w",
      "accel_energy_per_image_mj", "accel_e_comp_mj", "cpu_energy_per_image_mj")),
    ("hw_cpu_baseline.csv", lambda r: r.get("mode") in ("compute", "e2e") and r.get("status") == "ok",
     ("net", "kind", "threads", "mode"), ("median_us",)),
]
FIELDS = ["metric_file", "metric", "key", "n_sessions", "session_indices", "per_session",
          "mean", "between_sd", "min", "max", "range", "cv_pct", "input_git_commits",
          "input_paper_grade", "note"]
PAPER_OK = ("True",)


def session_dirs(root: Path, indices) -> dict[int, Path]:
    return {k: (root if k == 1 else root / f"rep{k}") for k in indices}


def load(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def collect(root: Path, indices=(1, 2, 3)) -> tuple[dict, list[str]]:
    """{(file, metric, key, sha, clock): {index: (value, row)}} + problems."""
    groups: dict = {}
    problems = []
    for k, d in session_dirs(root, indices).items():
        if not d.is_dir():
            problems.append(f"session {k}: {d} missing")
            continue
        for pat, filt, keys, mets in METRICS:
            for p in sorted(d.glob(pat)):
                for r in load(p):
                    if filt and not filt(r):
                        continue
                    si = str(r.get("session_index", "")).strip()
                    if si not in ("", str(k)):
                        raise SystemExit(f"REFUSED: {p} row session_index {si} in the directory of "
                                         f"session {k}")
                    key = json.dumps({c: r.get(c, "") for c in keys}, sort_keys=True)
                    try:
                        clk = f"{float(r.get('clock_mhz') or 'nan'):.1f}"
                    except ValueError:
                        clk = ""
                    sha = r.get("bitstream_sha256", "")
                    for m in mets:
                        v = r.get(m, "")
                        if v in ("", None):
                            continue
                        try:
                            fv = float(v)
                        except ValueError:
                            continue
                        g = groups.setdefault((pat, m, key, sha, clk), {})
                        prev = g.get(k)
                        if prev is None or r.get("timestamp", "") >= prev[1].get("timestamp", ""):
                            g[k] = (fv, r)
    return groups, problems


def aggregate(root: Path, indices=(1, 2, 3)) -> tuple[list[dict], str, list[str]]:
    groups, problems = collect(root, indices)
    srcs = {r.get("source") for g in groups.values() for _, r in g.values()}
    board = {bc.SOURCE_HW, "cpu_board"}
    if srcs & board and srcs - board:
        raise SystemExit(f"REFUSED: mixed sources {sorted(srcs)} (board and dry-run rows)")
    out_source = bc.SOURCE_HW if (srcs & board or not srcs) else bc.SOURCE_DRYRUN
    rows = []
    per_metric = {}
    for (pat, m, key, sha, clk) in groups:
        per_metric.setdefault((pat, m, key), set()).add((sha, clk))
    for (pat, m, key, sha, clk), g in sorted(groups.items()):
        ks = sorted(g)
        vals = [g[k][0] for k in ks]
        b = stats.between_sessions(vals)
        rs = [g[k][1] for k in ks]
        commits = sorted({r.get("git_commit", "") for r in rs})
        dirty = any(str(r.get("git_dirty", "")) != "False" for r in rs)
        pg = all(str(r.get("paper_grade", "")) in PAPER_OK for r in rs)
        notes = []
        if len(per_metric[(pat, m, key)]) > 1:
            notes.append("other bitstream/clock groups exist for this metric (never combined)")
        if len(commits) > 1:
            notes.append("sessions from different scripts commits")
        if len(ks) < len(indices):
            notes.append(f"only sessions {ks}")
        r0 = rs[-1]
        row = {"timestamp": bc.utc_now(), "git_commit": commits[0] if len(commits) == 1 else "mixed",
               "git_dirty": dirty, "vivado_version": r0.get("vivado_version", ""),
               "bitstream_sha256": sha, "board_id": r0.get("board_id", ""),
               "net": json.loads(key).get("net", r0.get("net", "")), "layer": r0.get("layer", ""),
               "clock_mhz": clk, "source": out_source, "duration_s": "", "num_inferences": "",
               "session_index": ",".join(str(k) for k in ks), "paper_grade": pg and not dirty,
               "metric_file": pat, "metric": m, "key": key, "n_sessions": b["n"],
               "session_indices": ",".join(str(k) for k in ks),
               "per_session": json.dumps({str(k): g[k][0] for k in ks}),
               "mean": stats.fmt(b["mean"], 6), "between_sd": stats.fmt(b.get("sd"), 6),
               "min": stats.fmt(b["min"], 6), "max": stats.fmt(b["max"], 6),
               "range": stats.fmt(b["range"], 6), "cv_pct": stats.fmt(b.get("cv_pct"), 4),
               "input_git_commits": " ".join(c[:12] for c in commits),
               "input_paper_grade": pg, "note": "; ".join(notes)}
        rows.append(row)
    return rows, out_source, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None, help="default results/ (hw); a dryrun dir for dry runs")
    ap.add_argument("--indices", nargs="+", type=int, default=[1, 2, 3])
    a = ap.parse_args(argv)
    root = Path(a.root).resolve() if a.root else bc.RESULTS_ROOT
    rows, source, problems = aggregate(root, a.indices)
    for p in problems:
        print(f"NOTE: {p}")
    if not rows:
        print("no per-session rows found; nothing written")
        return 1
    bc.write_csv(root / "hw_repeatability.csv", rows, FIELDS, source)
    print(f"{len(rows)} metric groups from sessions {a.indices} ({source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
