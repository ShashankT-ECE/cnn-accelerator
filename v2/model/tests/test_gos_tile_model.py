"""Step 2.2 part B: memory-level tile model (gos_tile_model) vs the golden model.

* Both nets, 20 test images each: every layer's stored output (unpacked from ACT) and
  the final raw INT32 v in LOGIT are bit-identical to gos_golden.run_net.
* 200 seeded random layer shapes within the envelope, with garbage in every byte of
  every memory not written by the host: output == gos_golden.gos_layer, every ACT byte
  outside the output map (and the whole input buffer) unchanged, LOGIT[OC..15] unchanged.
* Scalar ("loop") executor == golden on the nets (1 image) and on small random shapes.

The random-case helpers here are also used by test_gos_addr_stream.py.
"""
from __future__ import annotations

import math
import time
from collections import Counter

import numpy as np
import pytest

import gos_golden as gg
import gos_pack as gp
import gos_tile_model as tm
from requant_check import select_m_s

NETS = ("lenet5", "cifar10")
N_IMAGES = 20
N_RANDOM = 200
RANDOM_SEED = 20260924
B = 32


from gos_fuzz import (B, draw_accepted, draw_fields, garbage_mem, golden_layer,  # noqa: E402,F401
                      make_params, map_cover, run_case)


def case_label(i, f) -> str:
    return (f"#{i:03d} IC={f['IC']} OC={f['OC']} IH={f['IH']} IW={f['IW']} K{f['KH']}x{f['KW']} "
            f"OH={f['OH']} OW={f['OW']} pool={f['pool_en']} relu={f['relu_en']} "
            f"raw={f['out_raw']} in_sel={f['in_sel']} WGT_BASE={f['WGT_BASE']} "
            f"QP_BASE={f['QP_BASE']} K={f['K']}")


# --------------------------------------------------------------------------- #
# Nets
# --------------------------------------------------------------------------- #
def _net_inputs(net, n):
    x, _ = gg.load_test_set(net, n)
    return gg.quantize_input(net, x)


@pytest.mark.slow
@pytest.mark.parametrize("net", NETS)
def test_nets_bit_identical(net, capsys):
    xq = _net_inputs(net, N_IMAGES)
    G = gg.run_net(net, xq)
    names = [P.name for P in gg.load_net(net).layers]
    t0 = time.time()
    for i in range(N_IMAGES):
        r = tm.run_net_mem(net, xq[i])
        for n in names:
            assert np.array_equal(r["outputs"][n], G[n][i]), (net, i, n)
        assert r["logits"].dtype == np.int32 and np.array_equal(r["logits"], G["v"][i]), (net, i)
    with capsys.disabled():
        print(f"\n[tile model] {net}: {N_IMAGES} images x {len(names)} layers bit-identical "
              f"({time.time() - t0:.2f} s)")


@pytest.mark.slow
@pytest.mark.parametrize("net", NETS)
def test_nets_loop_executor(net):
    xq = _net_inputs(net, 1)
    G = gg.run_net(net, xq)
    r = tm.run_net_mem(net, xq[0], mode="loop")
    for P in gg.load_net(net).layers:
        assert np.array_equal(r["outputs"][P.name], G[P.name][0]), P.name
    assert np.array_equal(r["logits"], G["v"][0])


@pytest.mark.parametrize("net", NETS)
def test_net_layer_counts(net):
    xq = _net_inputs(net, 1)
    r = tm.run_net_mem(net, xq[0])
    for d in gp.make_descriptors(net)[0]:
        info = r["info"][d["name"]]
        assert info["issues"] == info["T_times_K"] == info["cycle_model_compute"]
        if d["out_raw"]:
            assert info["logits_written"] == d["OC"]
        else:
            assert info["act_bytes_written"] == d["OC"] * d["OUT_H"] * d["OUT_W"]


# --------------------------------------------------------------------------- #
# Arbitrary-layer images
# --------------------------------------------------------------------------- #
def test_layer_qparam_packing_matches_format():
    rng = np.random.default_rng(1)
    for net in NETS:
        words, bases = gp.pack_qparam(net)
        p = gp.load_params(net)
        for L, base in zip(gp.layers(net), bases):
            P = p[L["name"]]
            mine = tm.pack_layer_qparam(P["q_b"], P["m"], P["s"])
            assert np.array_equal(mine, words[2 * base:2 * (base + L["OC"])])
    q_b = rng.integers(-2**31, 2**31, size=37).astype(np.int32)
    m = rng.integers(0, 2**32, size=37).astype(np.uint64)
    s = rng.integers(0, 64, size=37).astype(np.uint8)
    b2, m2, s2 = gp.unpack_qparam(tm.pack_layer_qparam(q_b, m, s), 0, 37)
    assert np.array_equal(b2, q_b) and np.array_equal(m2, m) and np.array_equal(s2, s)


def test_place_layer_params_round_trip():
    rng = np.random.default_rng(2)
    (f, _), = draw_accepted(rng, 1)[0]
    p = make_params(rng, f)
    mem = tm.Mem()
    tm.place_layer_params(mem, f, p["q_w"], p["q_b"], p["m"], p["s"])
    assert np.array_equal(gp.unpack_wgt(mem.wgt, f, f["WGT_BASE"]), p["q_w"])
    assert not gp.wgt_tail_bytes(mem.wgt, f, f["WGT_BASE"]).any()
    q_b, m, s = gp.unpack_qparam(np.stack([mem.qp_e, mem.qp_o], 1).ravel(), f["QP_BASE"], f["OC"])
    assert np.array_equal(q_b, p["q_b"]) and list(m) == list(p["m"]) and list(s) == list(p["s"])


# --------------------------------------------------------------------------- #
# Random shapes
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_random_shapes_vs_golden(capsys):
    rng = np.random.default_rng(RANDOM_SEED)
    cases, rejects = draw_accepted(rng, N_RANDOM)
    cov = Counter()
    t0 = time.time()
    lines = []
    for i, (f, w) in enumerate(cases):
        p = make_params(rng, f)
        lines.append(case_label(i, f))
        try:
            info = run_case(f, w, p, "fast", rng)
        except AssertionError as e:
            raise AssertionError(f"{case_label(i, f)}: {e}") from e
        cov["oc_tail"] += f["OC"] % 8 != 0
        cov["ow_tail"] += f["OW"] % 8 != 0
        cov["pool_on"] += f["pool_en"] == 1
        cov["pool_off"] += f["pool_en"] == 0
        cov["k1"] += f["KH"] == 1
        cov["k5"] += f["KH"] == 5
        cov["out_raw"] += f["out_raw"] == 1
        cov["relu_off_nonfinal"] += f["relu_en"] == 0 and f["out_raw"] == 0
        cov["in_sel1"] += f["in_sel"] == 1
        cov["issues"] += info["issues"]
    dt = time.time() - t0
    with capsys.disabled():
        print(f"\n[tile model] random shapes seed={RANDOM_SEED}: {N_RANDOM} pass, "
              f"{rejects['_total']} rejected {dict((k, v) for k, v in rejects.items() if k != '_total')}")
        print(f"[tile model] coverage {dict(cov)}  ({dt:.1f} s)")
        for ln in lines:
            print("   ", ln)
    for k in ("oc_tail", "ow_tail", "pool_on", "pool_off", "k1", "k5", "out_raw", "in_sel1"):
        assert cov[k] > 0, f"random set lacks {k}"
    assert all(f["OC"] <= 16 for f, _ in cases if f["out_raw"])


def test_random_small_loop_vs_fast():
    """Scalar executor == fast executor == golden on small random shapes."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    done = 0
    while done < 6:
        (f, w), = draw_accepted(rng, 1)[0]
        if f["OC_TILES"] * f["OH"] * f["OW_TILES"] * f["K"] > 20000:
            continue
        p = make_params(rng, f)
        seed = int(rng.integers(0, 2**31))
        a = run_case(f, w, p, "loop", np.random.default_rng(seed))
        b = run_case(f, w, p, "fast", np.random.default_rng(seed))
        assert a["act_bytes_written"] == b["act_bytes_written"]
        assert a["logits_written"] == b["logits_written"]
        done += 1


def test_in_end_4095_wraps():
    """IN_END = 4095 (ACT full): masked-row reads of the last x tile address word 4096,
    which wraps to 0 on the 12-bit port; outputs are unaffected."""
    rng = np.random.default_rng(7)
    f = {"IC": 32, "OC": 12, "IH": 32, "IW": 32, "KH": 5, "KW": 5, "OH": 28, "OW": 28,
         "WGT_BASE": 100, "QP_BASE": 3, "relu_en": 1, "pool_en": 1, "out_raw": 0, "in_sel": 1}
    f.update(gp.derive_fields(f))
    w = gp.encode_descriptor(f)
    assert gp.check_descriptor(w)[0] and f["IN_END"] == 4095
    assert gp.max_act_read_word(f) == 4096
    run_case(f, w, make_params(rng, f), "fast", rng)
