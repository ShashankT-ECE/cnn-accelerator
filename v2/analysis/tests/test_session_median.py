"""session_median.merge_rows: median over board sessions with across-session spread (DECISIONS D26)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import session_median as SM  # noqa: E402


def row(net, phase, med, **kw):
    return {"net": net, "phase": phase, "median_us": med, "images": "1000", "timestamp": "t", "order_seed": str(kw.pop("seed", 1)),
            "host_path": kw.pop("host_path", "fast"), "_file": "f", "_line": 2, **kw}


def sessions(vals):
    return [[row("lenet5", "e2e", v, seed=i), row("lenet5", "pl", "65.745")] for i, v in enumerate(vals)]


def test_median_is_exactly_one_session_cell_and_spread_is_kept():
    m = SM.merge_rows(sessions(["565.730", "560.881", "560.162"]))
    r = m[0]
    assert r["median_us"] == "560.881"                       # original text of the middle session
    sp = r["_spread"]["median_us"]
    assert (sp["min_txt"], sp["max_txt"]) == ("560.162", "565.730")
    assert sp["cv"] == pytest.approx(0.5381, abs=1e-3)       # sample SD / mean * 100
    assert r["_nsess"] == 3 and len(r["_srcs"]) == 3


def test_identical_columns_unchanged_with_zero_spread_and_seeds_not_medianed():
    m = SM.merge_rows(sessions(["1", "2", "3"]))
    pl = m[1]
    assert pl["median_us"] == "65.745" and pl["_spread"]["median_us"]["cv"] == 0.0
    assert m[0]["order_seed"] == "0"                         # session identity columns come from the first session
    assert "order_seed" not in m[0]["_spread"]


def test_row_set_mismatch_and_config_mismatch_raise():
    s = sessions(["1", "2", "3"])
    s[2] = s[2][:1]
    with pytest.raises(ValueError, match="different row set"):
        SM.merge_rows(s)
    s = sessions(["1", "2", "3"])
    s[1][0]["host_path"] = "safe"
    with pytest.raises(ValueError, match="host_path differs"):
        SM.merge_rows(s)


def test_single_session_passthrough_and_even_count():
    one = SM.merge_rows(sessions(["5"])[:1])
    assert one[0]["_nsess"] == 1 and one[0]["median_us"] == "5"
    two = SM.merge_rows(sessions(["1", "3"]))
    assert float(two[0]["median_us"]) == 2.0


def test_duplicate_identities_are_matched_by_occurrence():
    s = [[row("a", "p", v), row("a", "p", w)] for v, w in (("1", "10"), ("2", "20"), ("3", "30"))]
    m = SM.merge_rows(s)
    assert [r["median_us"] for r in m] == ["2", "20"]
