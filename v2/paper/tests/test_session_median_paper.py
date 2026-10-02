"""The paper store returns the median over the board sessions for the replicated files (D26)."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from paperlib import DEFAULT_RESULTS, Artifact, Store  # noqa: E402
from paperdata import ci, pm, power_summaries  # noqa: E402

pytestmark = pytest.mark.skipif(not (DEFAULT_RESULTS / "rep3" / "hw_b3_breakdown.csv").exists(),
                                reason="needs the committed repeatability sessions")


def test_b3_rows_are_session_medians_and_ci_becomes_min_max():
    st = Store(DEFAULT_RESULTS)
    art = Artifact(st, "t", "table")
    rows = art.rows("hw_b3_breakdown_fast.csv", hw=True, net="lenet5", phase="end_to_end")
    r = rows[0]
    assert r["_nsess"] == 3
    per = [float(x["median_us"]) for x in r["_srcs"]]
    assert float(r["median_us"]) == statistics.median(per)
    txt = ci(art, r, "median_us", "median_ci_lo_us", "median_ci_hi_us", "f1")
    assert txt == f"{statistics.median(per):,.1f} [{min(per):,.1f}, {max(per):,.1f}]"
    files = {i["file"] for i in art.inputs.values()}
    assert {"v2/results/hw_b3_breakdown_fast.csv", "v2/results/rep2/hw_b3_breakdown_fast.csv",
            "v2/results/rep3/hw_b3_breakdown_fast.csv"} <= files          # every session is a manifest input


def test_pl_cycles_are_not_session_dependent_and_single_session_stays_single():
    st = Store(DEFAULT_RESULTS)
    pc = Artifact(st, "t", "table").rows("hw_b3_breakdown_fast.csv", hw=True, net="lenet5", phase="pl_compute")[0]
    assert pc["_spread"]["median_us"]["min"] == pc["_spread"]["median_us"]["max"]
    dpu = Artifact(st, "t", "table").rows("hw_dpu_latency.csv", hw=True)
    assert dpu and all(r.get("_nsess", 1) == 1 for r in dpu)
    assert Store(DEFAULT_RESULTS, sessions=False).load("hw_b3_breakdown.csv", True)[0].get("_nsess", 1) == 1


def test_power_cell_is_median_with_cv():
    art = Artifact(Store(DEFAULT_RESULTS), "t", "table")
    e = {x["net"]: x for x in power_summaries(art, "hw_b1_power_ina260")}["lenet5"]
    per = [float(x["accel_dp_w"]) for x in e["mean"]["_srcs"]]
    txt = pm(art, e, "accel_dp_w", "f3")
    assert txt.startswith(f"{statistics.median(per):.3f}") and "CV" in txt
