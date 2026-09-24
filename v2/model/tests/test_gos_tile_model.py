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


# --------------------------------------------------------------------------- #
# Random layer cases
# --------------------------------------------------------------------------- #
def draw_fields(rng) -> dict:
    """One raw descriptor field set in the envelope: IC 1..64, OC 1..128, input H/W 5..32,
    VALID kernel 1 or 5, pool only with even OH/OW, final layers (out_raw) with OC <= 16
    and a 1x1 output (5x5 kernel on a 5x5 map). Bases and in_sel random."""
    final = rng.random() < 0.1
    IC = int(rng.integers(1, 65))
    if final:
        OC, KH, IH, IW = int(rng.integers(1, 17)), 5, 5, 5
        relu = pool = 0
    else:
        OC = int(rng.integers(1, 129))
        KH = int(rng.choice([1, 5]))
        IH, IW = int(rng.integers(5, 33)), int(rng.integers(5, 33))
        OH, OW = IH - KH + 1, IW - KH + 1
        pool = int(OH % 2 == 0 and OW % 2 == 0 and rng.random() < 0.6)
        relu = int(rng.random() < 0.8)
    KW = KH
    K = IC * KH * KW
    words = math.ceil(OC / 8) * K
    wb = int(rng.integers(0, gp.WGT_DEPTH - words + 1)) if words <= gp.WGT_DEPTH else 0
    qb = int(rng.integers(0, gp.QP_CHANNELS - OC + 1))
    return {"IC": IC, "OC": OC, "IH": IH, "IW": IW, "KH": KH, "KW": KW, "OH": IH - KH + 1,
            "OW": IW - KW + 1, "WGT_BASE": wb, "QP_BASE": qb, "relu_en": relu,
            "pool_en": pool, "out_raw": int(final), "in_sel": int(rng.integers(0, 2))}


def draw_accepted(rng, n: int):
    """n descriptors the config checker accepts; returns (list of (fields, words), rejects)."""
    out, rejects = [], Counter()
    while len(out) < n:
        f = draw_fields(rng)
        f.update(gp.derive_fields(f))
        w = gp.encode_descriptor(f)
        ok, why = gp.check_descriptor(w)
        if not ok:
            rejects.update(why)
            rejects["_total"] += 1
            continue
        out.append((f, w))
    return out, rejects


def make_params(rng, f: dict) -> dict:
    """Random int8 input/weights, q_bias, and valid (m, s) from a random M (select_m_s, B=32)."""
    IC, OC, KH, KW, K = f["IC"], f["OC"], f["KH"], f["KW"], f["K"]
    x = rng.integers(-128, 128, size=(IC, f["IH"], f["IW"])).astype(np.int8)
    q_w = rng.integers(-128, 128, size=(OC, IC, KH, KW)).astype(np.int8)
    std = math.sqrt(K) * 5461.0
    q_b = rng.integers(-int(2 * std), int(2 * std) + 1, size=OC).astype(np.int32)
    if f["out_raw"]:
        m, s = [0] * OC, [0] * OC
    else:
        ms = [select_m_s(float(10 ** rng.uniform(-0.7, 0.7) * 40.0 / std), B) for _ in range(OC)]
        m, s = [a for a, _ in ms], [b for _, b in ms]
    return {"x": x, "q_w": q_w, "q_b": q_b, "m": m, "s": s}


def golden_layer(f: dict, p: dict):
    L = {"name": "rand", "IC": f["IC"], "OC": f["OC"], "KH": f["KH"], "KW": f["KW"],
         "IH": f["IH"], "IW": f["IW"], "OH": f["OH"], "OW": f["OW"], "K": f["K"],
         "relu": bool(f["relu_en"]), "pool": bool(f["pool_en"]), "final": bool(f["out_raw"])}
    if f["out_raw"]:
        P = gg.LayerParams(L, p["q_w"], p["q_b"])
    else:
        P = gg.LayerParams(L, p["q_w"], p["q_b"], m=tuple(p["m"]), s=tuple(p["s"]))
    return gg.gos_layer(p["x"], L, P)


def garbage_mem(rng) -> tm.Mem:
    """Every memory filled with random bytes (the RTL must not rely on unwritten bytes)."""
    mem = tm.Mem()
    mem.act[:] = rng.integers(0, 256, size=mem.act.shape, dtype=np.uint8)
    mem.wgt[:] = rng.integers(0, 2**63, size=mem.wgt.shape, dtype=np.uint64) << np.uint64(1)
    mem.qp_e[:] = rng.integers(0, 2**63, size=mem.qp_e.shape, dtype=np.uint64)
    mem.qp_o[:] = rng.integers(0, 2**63, size=mem.qp_o.shape, dtype=np.uint64)
    mem.logit[:] = rng.integers(-2**31, 2**31, size=mem.logit.shape).astype(np.int32)
    return mem


def map_cover(C, H, W) -> np.ndarray:
    """bool [8, 4096]: bytes occupied by a [C,H,W] map."""
    return gp.pack_act(np.ones((C, H, W), dtype=np.int8)) != 0


def run_case(f: dict, w, p: dict, mode: str, rng) -> dict:
    """Place a random case in garbage-filled memories, run the tile model, compare."""
    mem = garbage_mem(rng)
    ins, outs = f["in_sel"], 1 - f["in_sel"]
    cov_in = map_cover(f["IC"], f["IH"], f["IW"])
    mem.act[ins] = np.where(cov_in, gp.pack_act(p["x"]), mem.act[ins])
    tm.place_layer_params(mem, f, p["q_w"], p["q_b"], p["m"], p["s"])
    before = mem.copy()
    info = tm.run_layer_mem(mem, w, mode)
    gold = golden_layer(f, p)
    assert np.array_equal(mem.act[ins], before.act[ins]), "input buffer modified"
    assert np.array_equal(mem.wgt, before.wgt) and np.array_equal(mem.qp_e, before.qp_e)
    if f["out_raw"]:
        OC = f["OC"]
        assert np.array_equal(mem.logit[:OC], gold[:, 0, 0]), "LOGIT mismatch"
        assert np.array_equal(mem.logit[OC:], before.logit[OC:]), "LOGIT tail written"
        assert np.array_equal(mem.act, before.act), "out_raw wrote ACT"
    else:
        C, H, W = f["OC"], f["OUT_H"], f["OUT_W"]
        got = gp.unpack_act(mem.act[outs], C, H, W)
        assert np.array_equal(got, gold), "output map mismatch"
        cov = map_cover(C, H, W)
        assert np.array_equal(mem.act[outs][~cov], before.act[outs][~cov]), \
            "byte outside the output map written (mask failure)"
        assert info["act_bytes_written"] == C * H * W
        assert np.array_equal(mem.logit, before.logit)
    return info


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
