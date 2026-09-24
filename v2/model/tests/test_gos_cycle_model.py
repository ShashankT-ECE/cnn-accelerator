"""V2 cycle model (ARCH_SPEC "Cycle model (V2)") and D7 legacy ablation."""
import pytest

import gos_cycle_model as cm
from net_config import NET_CONFIGS

# ARCH_SPEC "Cycle model (V2)" compute-only T*K, and T = ceil(OC/8)*OH*ceil(OW/8).
SPEC_COMPUTE = {
    "lenet5": {"conv1": 2800, "conv3": 6000, "conv5": 6000, "fc1": 1320, "fc2": 168},
    "cifar10": {"conv1": 33600, "conv2": 64000, "conv3": 6400, "fc": 128},
}
SPEC_TOTAL = {"lenet5": 16288, "cifar10": 104128}
SPEC_T = {
    "lenet5": {"conv1": 112, "conv3": 40, "conv5": 15, "fc1": 11, "fc2": 2},
    "cifar10": {"conv1": 448, "conv2": 80, "conv3": 8, "fc": 2},
}
NETS = list(SPEC_COMPUTE)


def _layers(net):
    return NET_CONFIGS[net]["layers"]


@pytest.mark.parametrize("net", NETS)
def test_compute_and_T_match_spec(net):
    res = cm.net_cycles(_layers(net), c_pipe=None, c_start=None)
    assert {n: r["compute_cycles"] for n, r in res["layers"].items()} == SPEC_COMPUTE[net]
    assert {n: r["T"] for n, r in res["layers"].items()} == SPEC_T[net]
    assert res["total"]["compute_cycles"] == SPEC_TOTAL[net]
    assert res["total"]["cycles"] == SPEC_TOTAL[net]


@pytest.mark.parametrize("net", NETS)
def test_layer_invariants(net):
    for L in _layers(net):
        r = cm.layer_cycles(L, c_pipe=None)
        assert r["T"] == r["oc_tiles"] * r["oh_tiles"] * r["ow_tiles"]
        assert r["oh_tiles"] == L["OH"]
        assert r["mac_active"] == r["compute_cycles"] == r["T"] * L["K"]
        assert r["stall"] == 0
        assert r["macs"] == L["OC"] * L["OH"] * L["OW"] * L["K"]
        assert 0 < r["util_theoretical"] <= 1
        assert r["util_theoretical"] == r["macs"] / (64 * r["T"] * L["K"])


def test_full_tile_utilization_is_one():
    L = {"name": "x", "IC": 2, "OC": 16, "OH": 3, "OW": 16, "K": 18}
    r = cm.layer_cycles(L, c_pipe=None)
    assert r["T"] == 2 * 3 * 2 and r["util_theoretical"] == 1.0


def test_defaults_are_unknown():
    assert cm.C_PIPE is None and cm.C_START is None
    res = cm.net_cycles(_layers("lenet5"))
    assert all(r["cycles_basis"] == "compute_only" and r["c_pipe"] is None
               and r["cycles"] == r["compute_cycles"] for r in res["layers"].values())
    assert res["total"]["cycles_basis"] == "compute_only"
    assert res["total"]["cycles"] == SPEC_TOTAL["lenet5"]


@pytest.mark.parametrize("net", NETS)
def test_c_pipe_and_c_start_added(net):
    n = len(_layers(net))
    res = cm.net_cycles(_layers(net), c_pipe=7, c_start=100)
    for name, r in res["layers"].items():
        assert r["cycles"] == SPEC_COMPUTE[net][name] + 7
        assert r["cycles_basis"] == "compute+c_pipe"
    assert res["total"]["cycles"] == SPEC_TOTAL[net] + 7 * n + 100
    assert res["total"]["cycles_basis"] == "compute+c_pipe+c_start"
    assert res["total"]["compute_cycles"] == SPEC_TOTAL[net]

    only_start = cm.net_cycles(_layers(net), c_pipe=None, c_start=100)
    assert only_start["total"]["cycles"] == SPEC_TOTAL[net] + 100
    assert only_start["total"]["cycles_basis"] == "compute+c_start"
    only_pipe = cm.net_cycles(_layers(net), c_pipe=3, c_start=None)
    assert only_pipe["total"]["cycles"] == SPEC_TOTAL[net] + 3 * n
    assert only_pipe["total"]["cycles_basis"] == "compute+c_pipe"


def test_module_parameters_are_used(monkeypatch):
    monkeypatch.setattr(cm, "C_PIPE", 5)
    monkeypatch.setattr(cm, "C_START", 11)
    res = cm.net_cycles(_layers("cifar10"))
    assert res["total"]["cycles"] == SPEC_TOTAL["cifar10"] + 5 * 4 + 11
    assert cm.layer_cycles(_layers("cifar10")[0])["cycles"] == 33600 + 5


@pytest.mark.parametrize("bad", [-1, 1.5, "3", True])
def test_bad_constants_rejected(bad):
    with pytest.raises(ValueError):
        cm.layer_cycles(_layers("lenet5")[0], c_pipe=bad)


# ---- D7 ablation ---------------------------------------------------------------

@pytest.mark.parametrize("net", NETS)
def test_ablation_consistency(net):
    abl = cm.ablation(net)  # itself asserts core == compute and legacy totals
    assert list(abl["layers"]) == [L["name"] for L in _layers(net)]
    for name, t in abl["layers"].items():
        assert t["core"] == SPEC_COMPUTE[net][name]
        assert t["total"] == t["core"] + t["lead_in"] + t["pass_drain"] + t["clear"] + t["result_drain"]
        LL = next(x for x in cm.LEGACY_LAYERS[net] if x["name"] == name)
        assert t["total"] == cm.legacy.os_cycles(LL)["cycles"]
    assert abl["total"]["core"] == SPEC_TOTAL[net]
    assert abl["total"]["total"] == cm.LEGACY_TOTALS_SPEC[net]


def test_lenet_d7_decomposition():
    # DECISIONS.md D7: 30,013 = 16,288 + 8,160 + 2,973 + 180 + 2,412
    t = cm.ablation("lenet5")["total"]
    assert (t["total"], t["core"], t["lead_in"], t["pass_drain"], t["clear"], t["result_drain"]) == \
        (30013, 16288, 8160, 2973, 180, 2412)


def test_build_rows_labels():
    rows = cm.build_rows()
    assert len(rows) == sum(len(_layers(n)) + 1 for n in NETS)
    assert all(r["source"] == "model" and r["legacy_source"] == cm.LEGACY_SOURCE for r in rows)
    totals = {r["net"]: r for r in rows if r["layer"] == "total"}
    assert totals["lenet5"]["compute_cycles"] == 16288
    assert totals["cifar10"]["compute_cycles"] == 104128
    assert totals["lenet5"]["legacy_total"] == 30013
    assert totals["cifar10"]["legacy_total"] == 225584
