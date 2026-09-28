"""utilization.py (EXPERIMENTS A4, model side)."""
import csv
from fractions import Fraction

import pytest

import utilization as U
from net_config import NET_CONFIGS, NETS


@pytest.fixture(scope="module")
def rows():
    return {(r["net"], r["layer"]): r for r in U.build_rows()}


def test_cycles_match_committed_cycle_model(rows, cycle_model_csv):
    for net in NETS:
        for name in [L["name"] for L in NET_CONFIGS[net]["layers"]] + ["total"]:
            ref = cycle_model_csv[(net, name)]
            r = rows[(net, name)]
            assert r["layer_cyc"] == int(ref["cycles"]), (net, name)
            assert r["mac_active"] == int(ref["mac_active"]), (net, name)
            assert r["T"] == int(ref["T"]), (net, name)
            assert r["macs_useful"] == int(ref["macs"]), (net, name)
            assert r["util_spatial"] == ref["util_theoretical"], (net, name)


def test_one_by_one_bound_is_one_eighth(rows):
    one = {k: r for k, r in rows.items() if r.get("one_by_one_output") is True}
    # derived from shapes: every OH = OW = 1 layer, and only those
    expect = {(net, L["name"]) for net in NETS for L in NET_CONFIGS[net]["layers"]
              if L["OH"] == 1 and L["OW"] == 1}
    assert set(one) == expect and expect
    for (net, name), r in one.items():
        s = Fraction(r["util_spatial_exact"])
        assert s <= Fraction(1, 8) and r["spatial_bound"] == "0.125000"
        assert (s == Fraction(1, 8)) == (r["OC"] % 8 == 0)
        assert r["eff_x"] == "0.125000"


def test_decomposition_is_exact(rows):
    for (net, name), r in rows.items():
        if name == "total":
            assert (r["macs_useful"] + r["slots_lost_x_tail"] + r["slots_lost_oc_tail"]
                    + r["slots_lost_pipe"]) == 64 * r["layer_cyc"]
            continue
        s = Fraction(r["util_spatial_exact"])
        assert s == Fraction(r["macs_useful"], 64 * r["T"] * r["K"])
        assert s == Fraction(r["OC"], 8 * -(-r["OC"] // 8)) * Fraction(r["OW"], 8 * -(-r["OW"] // 8))
        assert r["macs_useful"] + r["slots_lost_x_tail"] + r["slots_lost_oc_tail"] == r["macs_slots"]
        assert r["slots_lost_pipe"] == 64 * r["c_pipe"]
        assert Fraction(r["util_total_exact"]) == s * Fraction(r["mac_active"], r["layer_cyc"])


def test_main_writes_csv_and_md(tmp_path):
    md = tmp_path / "U.md"
    assert U.main(["--out-dir", str(tmp_path), "--md", str(md)]) == 0
    out = list(csv.DictReader((tmp_path / U.CSV_NAME).open()))
    assert out and all(r["source"] == "model" for r in out)
    text = md.read_text()
    for r in out:   # every total utilization in the md is rendered from the CSV
        if r["layer"] == "total":
            assert f"**{100 * float(r['util_total']):.1f}%**" in text
    assert "not paper-valid" in text or all(r["git_dirty"] == "False" for r in out)
