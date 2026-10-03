"""Tests for v3/model/cycle_model.py and mem_model.py (model / estimate only).

Regression anchor: R = C = 8, row mapping 'x', VALID / stride 1 / OS reproduces v2/results/cycle_model.csv
(read as DATA, the v2 code is not imported) per layer and in total, for lenet5 and cifar10.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

MODEL = Path(__file__).resolve().parents[1]
REPO = MODEL.parents[1]
sys.path.insert(0, str(MODEL))

import cycle_model as cm  # noqa: E402
import mem_model as mm    # noqa: E402

V2_CSV = REPO / "v2" / "results" / "cycle_model.csv"


def v2_layers():
    """V2 layer shapes rebuilt from v2/results/cycle_model.csv (VALID, stride 1, square kernels)."""
    nets, totals = {}, {}
    with V2_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["layer"] == "total":
                totals[r["net"]] = int(r["cycles"])
                continue
            ic, oc, oh, ow, K = (int(r[k]) for k in ("IC", "OC", "OH", "OW", "K"))
            kk = math.isqrt(K // ic)
            assert kk * kk * ic == K
            L = dict(layer=r["layer"], kind="conv", ic=ic, oc=oc, ih=oh + kk - 1, iw=ow + kk - 1, oh=oh, ow=ow,
                     kh=kk, kw=kk, stride=1, pad=0, groups=1, k=K, macs=oc * oh * ow * K,
                     residual_from="", skip_kind="", gap=0)
            nets.setdefault(r["net"], []).append((L, int(r["T"]), int(r["cycles"])))
    return nets, totals


V2_NETS, V2_TOTALS = v2_layers()


@pytest.mark.parametrize("net", ["lenet5", "cifar10"])
def test_v2_regression(net):
    cfg = cm.ArrayConfig(R=8, C=8, row_map="x", modes=cm.MODES_OS_ONLY)
    assert cfg.c_pipe() == cm.C_PIPE_V2 == 29
    res = cm.net_cycles([L for L, _, _ in V2_NETS[net]], cfg)
    for (L, T, cyc), r in zip(V2_NETS[net], res["layers"]):
        assert r["mode"] == "os" and r["T"] == T and r["cycles"] == cyc, (net, L["layer"])
        assert r["stall"] == 0
    assert res["cycles"] == V2_TOTALS[net]


def test_schedule_drain_stall():
    s = cm.schedule([(9, 10)], drain=16, c_pipe=37)
    assert s["cycles"] == 9 + 9 * 16 + 37 and s["stall"] == 9 * 7
    assert cm.schedule([(64, 3)], 8, 29)["cycles"] == 3 * 64 + 29


def test_c_pipe_projection():
    assert cm.ArrayConfig(R=16, C=16).c_pipe() == 29 - 7 + 15
    assert cm.ArrayConfig(R=32, C=16).drain_len == 16           # drain = C columns
    assert cm.ArrayConfig(R=32, C=16).lanes == 32               # requant lanes = R values per column
    assert cm.ArrayConfig(R=32, C=16, drain_axis="row").drain_len == 32


def _L(**kw):
    base = dict(layer="t", kind="conv", ic=16, oc=32, ih=32, iw=32, oh=16, ow=16, kh=3, kw=3, stride=2,
                pad=1, groups=1, k=144, residual_from="", skip_kind="", gap=0)
    base.update(kw)
    base["macs"] = base["oc"] * base["oh"] * base["ow"] * base["k"]
    return base


def test_stride_strategies():
    L = _L()
    T = 4 * 16 * 2
    half = cm.os_layer(L, cm.ArrayConfig(stride2="half_rate"))
    split = cm.os_layer(L, cm.ArrayConfig(stride2="phase_split"))
    dual = cm.os_layer(L, cm.ArrayConfig(stride2="dual_port"))
    assert half["cycles"] == T * 288 + 29
    assert split["cycles"] == dual["cycles"] == T * 144 + 29


def test_row_mapping():
    L = _L(ic=64, oc=64, ih=8, iw=8, oh=8, ow=8, stride=1, k=576)
    x = cm.os_layer(L, cm.ArrayConfig(R=32, C=16, row_map="x"))
    xy = cm.os_layer(L, cm.ArrayConfig(R=32, C=16, row_map="xy"))
    assert x["T"] == 4 * 8 * 1 and xy["T"] == 4 * 2
    # VALID layer: 'xy' falls back to 'x'
    V = _L(ic=3, oc=32, ih=32, iw=32, oh=28, ow=28, kh=5, kw=5, stride=1, pad=0, k=75)
    r = cm.os_layer(V, cm.ArrayConfig(R=8, C=8, row_map="xy"))
    assert r["row_map_used"] == "x" and r["T"] == 4 * 28 * 4


def test_splitk_fc():
    fc = _L(layer="fc", kind="fc", ic=64, oc=10, ih=1, iw=1, oh=1, ow=1, kh=1, kw=1, stride=1, pad=0, k=64)
    cfg = cm.ArrayConfig(R=16, C=16, modes=cm.MODES_BEST)
    sk = cm.splitk_layer(fc, cfg, 16)
    assert sk["cycles"] == 4 + 37 + 4
    assert cm.os_layer(fc, cfg)["cycles"] == 64 + 37
    best = cm.layer_cycles(fc, cfg)
    assert best["mode"] == "splitk" and best["cycles"] == sk["cycles"]
    assert cm.splitk_factors(cm.ArrayConfig(R=16, C=16, packed=True, modes=cm.MODES_BEST)) == [2, 4, 8]


def test_depthwise_modes():
    L = _L(layer="dw", kind="dw", ic=32, oc=32, ih=32, iw=32, oh=32, ow=32, stride=1, groups=32, k=9)
    cfg = cm.ArrayConfig(R=8, C=8, modes=cm.MODES_BEST)
    assert cm.dw_layer(L, cfg)["cycles"] == 4 * 32 * 4 * 9 + 29
    assert cm.os_layer(L, cfg)["cycles"] == 32 * 32 * 4 * 9 + 29
    assert cm.layer_cycles(L, cfg)["mode"] == "dw"
    # C = 16 > KH*KW: drain-bound tiles
    c16 = cm.ArrayConfig(R=16, C=16, modes=cm.MODES_BEST)
    T = 2 * 32 * 2
    assert cm.dw_layer(L, c16)["cycles"] == 9 + (T - 1) * 16 + 37


def test_packing():
    L = _L(stride=1, oh=32, ow=32, k=144)
    cfg = cm.ArrayConfig(packed=True, pack_kmax=64, pack_extract_cycles=1)
    T = 4 * 32 * 4
    assert cm.os_layer(L, cfg)["cycles"] == T * (144 + 2) + 29
    assert cm.os_layer(L, cm.ArrayConfig(packed=True))["cycles"] == T * 144 + 29
    assert cm.ArrayConfig(R=16, C=16, packed=True).pe_dsps == 128
    assert cm.ArrayConfig(R=8, C=8).requant_dsps == 16


def test_sparsity_density_and_mask():
    L = _L(stride=1, oh=16, ow=16, oc=16, k=144)
    cfg = cm.ArrayConfig(R=8, C=8)
    d = cm.os_layer(L, cfg, density=0.5)
    m = cm.os_layer(L, cfg, nnz=[72, 72])
    assert d["cycles"] == m["cycles"] == 2 * 16 * 2 * 72 + 29
    w = np.zeros((10, 6), dtype=np.int8)
    w[0, 1] = 3
    w[9, 4] = -1
    mask = cm.block_mask(w, 8)
    assert mask.shape == (2, 6) and cm.nnz_from_mask(mask) == [1, 1]
    assert mask[0, 1] and mask[1, 4] and mask.sum() == 2


def test_c2_zero_ksteps_hand():
    # 1x1 conv, 2 input channels, 1x8 map, R = 4 -> two tiles.
    L = _L(ic=2, oc=8, ih=1, iw=8, oh=1, ow=8, kh=1, kw=1, stride=1, pad=0, k=2)
    act = np.zeros((2, 1, 8), dtype=np.int8)
    act[0, 0, 5] = 7            # tile 1, channel 0 nonzero
    zk = cm.zero_ksteps(act, L, R=4)
    assert list(zk) == [2, 1]
    r = cm.layer_cycles_c2(L, cm.ArrayConfig(R=4, C=8), act)
    assert r["skipped_ksteps"] == 3


def test_c2_synthetic_relu():
    rng = np.random.default_rng(0)
    L = _L(ic=16, oc=16, ih=16, iw=16, oh=16, ow=16, stride=1, k=144)
    x = np.maximum(rng.normal(-1.0, 1.0, size=(16, 16, 16)), 0)        # ~84% zeros
    act = np.clip(np.rint(x * 40), 0, 127).astype(np.int8)
    cfg = cm.ArrayConfig(R=8, C=8)
    r = cm.layer_cycles_c2(L, cfg, act)
    # brute-force reference for the zero count
    pad = np.pad(act, ((0, 0), (1, 1), (1, 1)))
    ref = 0
    for oy in range(16):
        for t in range(2):
            for ic in range(16):
                for ky in range(3):
                    for kx in range(3):
                        ref += int(not pad[ic, oy + ky, t * 8 + kx: t * 8 + kx + 8].any())
    assert r["skipped_ksteps"] == ref * 2
    assert 0 < r["saved_cycles"] and r["cycles"] < r["dense_cycles"]
    dense = np.ones_like(act)
    # dense input: only the all-padding rows (oy=0,ky=0 and oy=15,ky=2) skip: 2 rows x 2 tiles x 16 ic x 3 kx x 2 oc tiles
    assert cm.layer_cycles_c2(L, cfg, dense)["skipped_ksteps"] == 2 * 2 * 16 * 3 * 2


def test_residual_and_gap_passes():
    L = _L(stride=1, oh=8, ow=8, oc=64, k=576, residual_from="x", skip_kind="identity", gap=1)
    fused = cm.ArrayConfig(R=8, C=8)
    sep = cm.ArrayConfig(R=8, C=8, residual="separate", gap="separate")
    assert cm.residual_pass_cycles(L, fused) == 0 and cm.gap_pass_cycles(L, fused) == 0
    assert cm.residual_pass_cycles(L, sep) == 64 * 64 // 8 + 29
    assert cm.gap_pass_cycles(L, sep) == 64 * 64 // 8 + 29
    slow = cm.ArrayConfig(R=8, C=8, skip_read_bytes_per_cycle=4)
    assert cm.os_layer(L, slow)["drain"] == 16


def test_ws_sketch():
    L = _L(stride=1, oh=16, ow=16, oc=32, k=288)
    r = cm.ws_layer(L, cm.ArrayConfig(R=16, C=16, modes=("os", "ws")))
    blocks = 18 * 2
    assert r["cycles"] == blocks * 256 + (29 - 7 + 0) + 16 + 16 - 1


def test_mem_blocks_and_nets():
    assert mm.blocks(8, 4096, "BRAM36") == 1
    assert mm.blocks(8, 8192, "BRAM36") == 2
    assert mm.blocks(128, 4096, "URAM288") == 2
    assert mm.blocks(72, 512, "BRAM36") == 1
    import nets
    t = nets.layer_table("resnet20_b")
    cfg = cm.ArrayConfig(R=16, C=16)
    w = mm.weight_storage(t, cfg)
    assert w["raw_bytes"] == sum(L["params"] for L in t)
    assert w["padded_bytes"] >= w["raw_bytes"]
    a = mm.act_storage(t, cfg)
    assert a["n_buffers"] == 3 and a["skip_max_bytes"] > 0
    assert a["peak_live_bytes"] >= a["pair_max_bytes"]
    half = mm.weight_storage(t, cfg, density=0.5)
    assert half["words"] < w["words"] and half["meta_bits"] > 0


def test_best_never_worse_than_os_only():
    import nets
    for net in nets.NETS:
        t = nets.layer_table(net)
        for R, C in ((8, 8), (16, 16), (32, 16), (16, 32)):
            for rm in ("x", "xy"):
                os_only = cm.net_cycles(t, cm.ArrayConfig(R=R, C=C, row_map=rm))
                best = cm.net_cycles(t, cm.ArrayConfig(R=R, C=C, row_map=rm, modes=cm.MODES_BEST))
                assert best["cycles"] <= os_only["cycles"]
                assert all(b["cycles"] <= o["cycles"] for b, o in zip(best["layers"], os_only["layers"]))
