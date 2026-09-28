"""ablation.py (DECISIONS D7 schedule ablation, model)."""
import csv

import pytest

import ablation as A
from net_config import NETS

# DECISIONS D7: LeNet 30,013 = 16,288 + 8,160 + 2,973 + 180 + 2,412 (legacy model spec).
D7_LENET = {"v1_total": 30013, "v1_core": 16288, "v1_lead_in": 8160,
            "v1_pass_drain": 2973, "v1_clear": 180, "v1_result_drain": 2412}


@pytest.fixture(scope="module")
def rows():
    return {(r["net"], r["layer"]): r for r in A.build_rows()}


def test_d7_lenet_decomposition(rows):
    t = rows[("lenet5", "total")]
    assert {k: t[k] for k in D7_LENET} == D7_LENET


def test_both_nets_present_and_consistent(rows, cycle_model_csv):
    for net in NETS:
        layer_rows = [r for (n, l), r in rows.items() if n == net and l != "total"]
        assert layer_rows
        for r in layer_rows + [rows[(net, "total")]]:
            ref = cycle_model_csv[(net, r["layer"])]
            assert r["core_equal"] is True and r["v1_core"] == r["v2_compute"]
            assert r["v1_total"] == sum(r[f"v1_{k}"] for k in A.TERMS)
            assert r["v1_total"] == int(ref["legacy_total"])
            assert r["v2_layer_cyc"] == int(ref["cycles"])
            assert r["removed_cycles"] == r["v1_total"] - r["v2_layer_cyc"]
            assert r["source"] == "model"


def test_main_writes_csv(tmp_path):
    assert A.main(["--out-dir", str(tmp_path)]) == 0
    out = list(csv.DictReader((tmp_path / A.CSV_NAME).open()))
    assert {r["net"] for r in out} == set(NETS)
