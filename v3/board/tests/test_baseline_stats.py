"""stats.py (copied from V2): median CI, summarize, block order, between-sessions; D26 aggregation."""
import json
import math

import numpy as np
import pytest

import aggregate_sessions as agg
import baseline_common as bc
import stats


def test_median_ci_brackets_median_and_coverage():
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=1001)
    lo, hi, cov = stats.median_ci(x)
    m = float(np.median(x))
    assert lo <= m <= hi
    assert cov >= 0.95
    assert lo in x and hi in x            # order statistics, not interpolated


def test_median_ci_small_n_has_no_interval():
    assert stats.median_ci([1.0, 2.0, 3.0, 4.0, 5.0]) == (None, None, None)
    assert stats.median_ci(np.arange(6.0))[0] is not None


def test_summarize_fields_and_repeats_flag():
    s = stats.summarize(np.arange(1, 201, dtype=float))
    for k in ("n", "median", "ci_lo", "ci_hi", "ci_coverage", "p5", "p95", "p99", "mean", "sd", "min", "max"):
        assert k in s
    assert s["n"] == 200 and s["median"] == 100.5 and s["repeats_ok"]
    assert not stats.summarize(np.arange(10.0))["repeats_ok"]
    assert math.isnan(stats.summarize([])["median"])


def test_block_order_seeded_and_complete():
    a = stats.block_order(["x", "y", "z"], 4, seed=7)
    assert a == stats.block_order(["x", "y", "z"], 4, seed=7)
    for r in range(4):
        assert sorted(a[3 * r:3 * r + 3]) == ["x", "y", "z"]


def test_between_sessions():
    b = stats.between_sessions([1.0, 2.0, 3.0])
    assert b["mean"] == 2.0 and b["sd"] == 1.0 and b["range"] == 2.0 and b["cv_pct"] == 50.0


def _write(d, name, rows, fields):
    d.mkdir(parents=True, exist_ok=True)
    bc.write_csv(d / name, rows, fields, bc.SOURCE_DRYRUN)


def _row(k, **kw):
    r = {"timestamp": f"2026-10-0{k}T00:00:00+00:00", "git_commit": "c" * 40, "git_dirty": False,
         "source": bc.SOURCE_DRYRUN, "net": "resnet20_b", "session_index": k, "paper_grade": False}
    r.update(kw)
    return r


def test_aggregate_median_rule_and_session_check(tmp_path):
    root = tmp_path / "dryrun"
    for k, p50 in ((1, 100.0), (2, 130.0), (3, 110.0)):
        d = root if k == 1 else root / f"rep{k}"
        _write(d, "hw_baseline_latency.csv",
               [_row(k, config="dpu", metric="e2e", p50=p50, p95=2 * p50, p99=3 * p50, median=p50)],
               ["config", "metric", "p50", "p95", "p99", "median"])
    rows, src, probs = agg.aggregate(root)
    assert src == bc.SOURCE_DRYRUN and not probs
    r = next(r for r in rows if r["metric"] == "p50")
    assert float(r["median"]) == 110.0              # D26: median of the per-session values
    assert float(r["min"]) == 100.0 and float(r["max"]) == 130.0
    assert json.loads(r["per_session"]) == {"1": 100.0, "2": 130.0, "3": 110.0}
    # a row whose session_index disagrees with its directory is refused
    _write(root / "rep2", "hw_baseline_latency.csv",
           [_row(3, config="dpu", metric="e2e", p50=1.0, p95=1.0, p99=1.0, median=1.0)],
           ["config", "metric", "p50", "p95", "p99", "median"])
    with pytest.raises(SystemExit):
        agg.aggregate(root)
