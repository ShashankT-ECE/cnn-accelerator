"""projection_16x16.py (NxN model projection; not implemented, not EXPERIMENTS C1)."""
import csv

import pytest

import gos_cycle_model as cm
import projection_16x16 as P
from net_config import NET_CONFIGS, NETS


@pytest.fixture(scope="module")
def rows():
    return {(r["array_n"], r["net"], r["layer"]): r for r in P.build_rows()}


def test_n8_equals_committed_cycle_model(rows, cycle_model_csv):
    assert P.c_pipe(8) == cm.C_PIPE
    for net in NETS:
        for name in [L["name"] for L in NET_CONFIGS[net]["layers"]] + ["total"]:
            r, ref = rows[(8, net, name)], cycle_model_csv[(net, name)]
            assert r["layer_cyc"] == int(ref["cycles"]), (net, name)
            assert r["T"] == int(ref["T"]) and r["compute_cycles"] == int(ref["compute_cycles"])
            assert r["util_spatial"] == ref["util_theoretical"]
            assert r["speedup_vs_n8"] == "1.0000" and r["reproduces_8x8_model"] is True


def test_n16_formula(rows):
    assert P.c_pipe(16) == cm.C_PIPE + 8
    for net in NETS:
        for L in NET_CONFIGS[net]["layers"]:
            r = rows[(16, net, L["name"])]
            T = -(-L["OC"] // 16) * L["OH"] * -(-L["OW"] // 16)
            assert r["T"] == T and r["layer_cyc"] == T * L["K"] + P.c_pipe(16)
            assert r["drain_hidden"] == (L["K"] >= 16)
        tot = rows[(16, net, "total")]
        assert tot["layer_cyc"] == sum(rows[(16, net, L["name"])]["layer_cyc"]
                                       for L in NET_CONFIGS[net]["layers"]) + cm.C_START + cm.C_DONE


def test_every_row_labeled(rows):
    for r in rows.values():
        assert r["label"] == "projected (model, not implemented)"
        assert "not EXPERIMENTS C1" in r["c1_note"] and r["assumptions"]
        assert r["source"] == "model"


def test_odd_n_rejected():
    with pytest.raises(ValueError):
        P.layer_proj(NET_CONFIGS["lenet5"]["layers"][0], 9)


def test_main_writes_csv(tmp_path):
    assert P.main(["--out-dir", str(tmp_path)]) == 0
    out = list(csv.DictReader((tmp_path / P.CSV_NAME).open()))
    assert {r["array_n"] for r in out} == {"8", "16"}
