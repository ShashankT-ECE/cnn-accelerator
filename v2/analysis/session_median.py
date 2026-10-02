"""Median-over-sessions merge of board result rows (paper freeze, DECISIONS D26).

The board campaign ran the same steps in three independent sessions (session_index 1, 2, 3; results in
results/, results/rep2/, results/rep3/). For the files that exist in every session this module builds one
row per measurement whose numeric cells are the MEDIAN over the sessions, with the across-session spread
(min, max, CV) kept next to it. Nothing is invented: with an odd number of sessions the median of a column
is exactly one session's cell (its original text is kept); the spread comes from the same cells.

  * rows are matched across sessions by identity columns (net, phase, condition, kind, mode, threads,
    row_kind, repeat, ...) plus the occurrence number of the identity; a session with a different row set
    is an error (the sessions must have run the same steps);
  * a numeric column whose cells are identical in every session (PL cycles, clock read-back, counts) is
    kept as is; columns that identify a session (timestamps, seeds, commits, session_index) are taken from
    the first session and never medianed;
  * columns that describe the configuration (host path, workloads, status, ...) must be identical in all
    sessions, else ValueError.

Merged row extras: `_nsess` (number of sessions), `_srcs` (the per-session source rows),
`_spread` {column: {"min", "max", "cv", "min_txt", "max_txt"}} for every numeric column.
Stdlib only (used by v2/paper and v2/analysis/board_efficiency.py).
"""
from __future__ import annotations

import csv
import fnmatch
import math
import re
import statistics
from collections import defaultdict

# Board files that exist in every repeatability session (results/, results/rep2/, results/rep3/). Everything
# else (DPU, A1, A2 wall-clock, A4, B2, soak, layer spread, shapes) was measured in one session only.
SESSION_FILES = ("hw_b1_power_ina260_summary_*.csv", "hw_b3_breakdown.csv", "hw_b3_breakdown_fast.csv",
                 "hw_b3_breakdown_cpu.csv", "hw_cpu_baseline.csv")
SESSION_DIRS = ("rep2", "rep3")

IDENT = ("net", "layer", "phase", "condition", "kind", "mode", "threads", "row_kind", "repeat", "metric",
         "system", "basis", "id")
MUST_MATCH = ("host_path", "write_mode", "fast_store", "accel_workload", "control_workload", "cpu_workload",
              "status", "ci_method", "measurement", "rail", "n_repeats", "env_step", "source")
NOT_MEDIAN = re.compile(r"(seed|epoch|utc|local|timestamp|git|sha|session_index|^_|_ts$|order|hostname|"
                        r"package|versions|affinity|note|method)")


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _key(r: dict, seen: dict) -> tuple:
    base = tuple((c, r[c]) for c in IDENT if c in r)
    n = seen[base]
    seen[base] += 1
    return base + (("#", n),)


def merge_rows(sessions: list[list[dict]], name: str = "?") -> list[dict]:
    """sessions[k] = rows of session k (dicts, any extra keys such as _file/_line are kept from the first
    session's row and the per-session rows are listed in `_srcs`). Returns the merged rows in the first
    session's order."""
    n = len(sessions)
    if n == 1:
        return [dict(r, _nsess=1, _srcs=[r], _spread={}) for r in sessions[0]]
    maps = []
    for rows in sessions:
        seen: dict = defaultdict(int)
        maps.append({_key(r, seen): r for r in rows})
    ref = maps[0]
    for k, m in enumerate(maps[1:], 2):
        if set(m) != set(ref) or len(m) != len(sessions[k - 1]):
            only_a, only_b = sorted(set(ref) - set(m))[:3], sorted(set(m) - set(ref))[:3]
            raise ValueError(f"{name}: session {k} has a different row set than session 1 "
                             f"(rows {len(m)} vs {len(ref)}; only in 1: {only_a}; only in {k}: {only_b})")
    out = []
    for key, r0 in ref.items():
        rs = [m[key] for m in maps]
        row = dict(r0)
        spread = {}
        for col in r0:
            if col.startswith("_") or col in IDENT:
                continue
            cells = [r.get(col, "") for r in rs]
            if col in MUST_MATCH and len(set(cells)) != 1:
                raise ValueError(f"{name} {dict(key[:-1])}: {col} differs between sessions: {sorted(set(cells))}")
            if NOT_MEDIAN.search(col):
                continue
            vals = [_f(c) for c in cells]
            if any(v is None for v in vals):
                continue                                   # empty / text cell: keep the first session's
            if len(set(cells)) == 1:
                spread[col] = {"min": vals[0], "max": vals[0], "cv": 0.0, "min_txt": cells[0], "max_txt": cells[0]}
                continue
            order = sorted(range(n), key=lambda i: vals[i])
            if n % 2:
                row[col] = cells[order[n // 2]]
            else:
                row[col] = repr(statistics.median(vals))
            mean = statistics.fmean(vals)
            sd = statistics.stdev(vals)
            spread[col] = {"min": vals[order[0]], "max": vals[order[-1]],
                           "cv": (sd / abs(mean) * 100.0) if mean else None,
                           "min_txt": cells[order[0]], "max_txt": cells[order[-1]]}
        row.update(_nsess=n, _srcs=rs, _spread=spread)
        out.append(row)
    return out


def is_replicated(name: str) -> bool:
    return any(fnmatch.fnmatch(name, g) for g in SESSION_FILES)


def session_files(res, name: str) -> list:
    """[res/name, res/rep2/name, res/rep3/name] when `name` is a replicated board file and every session
    directory has it; else [res/name]."""
    from pathlib import Path
    p = Path(res) / name
    if not is_replicated(name):
        return [p]
    extra = [Path(res) / d / name for d in SESSION_DIRS]
    return [p, *extra] if p.exists() and all(e.exists() for e in extra) else [p]


def load_median(res, name: str) -> list[dict]:
    """Rows of a board CSV as the median over the sessions (plain csv rows; no cleanliness rules: callers
    that need them, e.g. check_results.py, apply them to the per-session files)."""
    sess = []
    for q in session_files(res, name):
        with open(q, newline="") as f:
            rows = list(csv.DictReader(f))
        for i, r in enumerate(rows):
            r["_file"], r["_line"] = str(q), i + 2
        sess.append(rows)
    return merge_rows(sess, name)
