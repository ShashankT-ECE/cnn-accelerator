"""Final layer (D2): raw INT32 v -> reference float32 logits -> argmax."""
import numpy as np
import pytest

import final_layer as fl
from common import REPO_ROOT

NETS = list(fl.NETS)


def _assert_exact(r):
    assert r["logits_bitexact"] == r["images"]
    assert r["argmax_match"] == r["images"]
    assert r["int32_fits"]
    assert r["max_abs_v"] <= r["bound_abs_v"] < 2**31


@pytest.mark.parametrize("net", NETS)
def test_final_layer_fast(net):
    _assert_exact(fl.check_net(net, n=64, batch=64))


@pytest.mark.slow
@pytest.mark.parametrize("net", NETS)
def test_final_layer_full(net):
    r = fl.check_net(net, n=fl.TEST_N)
    assert r["images"] == 10000
    _assert_exact(r)


def test_lenet_frozen_params_match_calibration():
    z = np.load(REPO_ROOT / "data/lenet5_int8/quant_params.npz")
    p = fl.calibrated_params("lenet5")
    assert np.array_equal(z["S_input"], p["S_input"])
    for layer in ("conv1", "conv3", "conv5", "fc1", "fc2"):
        for k, val in p[layer].items():
            a = z[f"{layer}_{k}"]
            assert a.dtype == np.asarray(val).dtype, (layer, k)
            assert np.array_equal(a, val), (layer, k)


def test_logits_from_raw_rejects_non_int32():
    p = {"S_a": 1.0, "S_w": np.ones(10)}
    with pytest.raises(AssertionError):
        fl.logits_from_raw(np.zeros((1, 10), np.int64), p)
