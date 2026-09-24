"""Step 2.2 part A: gos_golden (integer-only) vs the legacy INT8 reference.

The legacy model is built from a params dict reconstructed from the
NET_CONFIGS quant_params npz (not via calibrate), so a retrained parameter set
works unchanged. Per image and per layer the golden int8 outputs / final raw
INT32 v / prediction are compared with legacy ``forward_layers``:

  LeNet-5  python/lenet5/int8_model.py : q_input :146, pool1 :154, pool2 :162,
           c5 :168, f6 :175, out_acc :179 (v = out_acc + q_b), logits :181
  CIFAR-10 python/cifar10/int8_model.py: q_input :85, pool1 :93, pool2 :101,
           c3 :108, fc_acc :113 (v = fc_acc + q_b), logits :115
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import gos_golden as gg
from net_config import NET_CONFIGS

from golden_crosscheck import (ACCURACY_OF_RECORD, BATCH, FINAL_KEYS, LEGACY_KEYS,  # noqa: E402,F401
                               NETS, NONFINAL_KEYS, TEST_N, compare, legacy_model,
                               legacy_params)


def _assert_all_exact(r):
    n = r["images"]
    print(r)
    for k, c in r["exact"].items():
        assert c == n, (r["net"], k, c, n)
    assert r["pred_eq"] == n
    assert r["correct"] == r["legacy_correct"]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("net", NETS)
def test_load_net_shapes(net):
    G = gg.load_net(net)
    assert [P.name for P in G.layers] == [L["name"] for L in NET_CONFIGS[net]["layers"]]
    assert G.B == 32
    for P in G.layers:
        L = P.cfg
        assert P.q_w.dtype == np.int8 and P.q_w.shape == (L["OC"], L["IC"], L["KH"], L["KW"])
        assert P.q_b.dtype == np.int32 and P.q_b.shape == (L["OC"],)
        if L["final"]:
            assert P.m == () and P.S_w.shape == (L["OC"],)
        else:
            assert len(P.m) == len(P.s) == L["OC"]


@pytest.mark.parametrize("net", NETS)
def test_legacy_params_complete(net):
    p = legacy_params(net)
    for L in NET_CONFIGS[net]["layers"]:
        assert set(p[L["name"]]) == (FINAL_KEYS if L["final"] else NONFINAL_KEYS)


def test_lenet_legacy_params_equal_calibration():
    """The reconstructed dict equals legacy calibrate() output (LeNet: frozen = calibrated)."""
    import final_layer as fl
    p, c = legacy_params("lenet5"), fl.calibrated_params("lenet5")
    assert p["S_input"] == c["S_input"]
    for n in ("conv1", "conv3", "conv5", "fc1", "fc2"):
        assert set(p[n]) == set(c[n])
        for k in c[n]:
            assert np.array_equal(np.asarray(p[n][k]), np.asarray(c[n][k])), (n, k)
            assert np.asarray(p[n][k]).dtype == np.asarray(c[n][k]).dtype, (n, k)


@pytest.mark.parametrize("net", NETS)
def test_gos_layer_dtypes_and_single_image(net):
    G = gg.load_net(net)
    x, _ = gg.load_test_set(net, 2)
    q = gg.quantize_input(G, x)
    batch = gg.run_net(G, q)
    single = gg.run_net(G, q[1])
    for P in G.layers:
        assert np.array_equal(single[P.name], batch[P.name][1])
        exp = np.int32 if P.cfg["final"] else np.int8
        assert batch[P.name].dtype == exp
    assert single["pred"] == batch["pred"][1]
    # per-layer call with return_v: v int64, y int8
    y, v = gg.gos_layer(q[0], G.layers[0].cfg, G.layers[0], return_v=True)
    assert v.dtype == np.int64 and y.dtype == np.int8
    with pytest.raises(AssertionError):
        gg.gos_layer(q[0].astype(np.int16), G.layers[0].cfg, G.layers[0])


def test_conv_acc_matches_direct_loop():
    rng = np.random.default_rng(1)
    x = rng.integers(-128, 128, (2, 3, 7, 6), dtype=np.int8)
    w = rng.integers(-128, 128, (4, 3, 3, 2), dtype=np.int8)
    acc = gg.gos_conv_acc(x, w)
    ref = np.zeros((2, 4, 5, 5), dtype=np.int64)
    for n in range(2):
        for o in range(4):
            for i in range(5):
                for j in range(5):
                    ref[n, o, i, j] = int((x[n, :, i:i + 3, j:j + 2].astype(np.int64)
                                           * w[o].astype(np.int64)).sum())
    assert np.array_equal(acc, ref)


@pytest.mark.parametrize("net", NETS)
def test_golden_vs_legacy_fast(net):
    _assert_all_exact(compare(net, 32, batch=32))


@pytest.mark.slow
@pytest.mark.parametrize("net", NETS)
def test_golden_vs_legacy_full(net):
    t0 = time.time()
    r = compare(net, TEST_N)
    r["total_s"] = round(time.time() - t0, 2)
    _assert_all_exact(r)
    assert r["correct"] == ACCURACY_OF_RECORD[net], (net, r["correct"])
