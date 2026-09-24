"""Random layer cases within the V2 envelope (shared by tests and gen_vectors).

Moved unchanged from tests/test_gos_tile_model.py (Step 2.2) so the Step 4 fuzz
vectors use exactly the same sampler; `draw_accepted` gained an optional T*K cap.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

import gos_golden as gg
import gos_pack as gp
import gos_tile_model as tm
from requant_check import select_m_s

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


def draw_accepted(rng, n: int, max_tk: int | None = None):
    """n descriptors the config checker accepts; returns (list of (fields, words), rejects).

    max_tk: optional cap on T*K (issue cycles) — shapes above it are redrawn and
    counted under "_tk_cap" (keeps RTL simulation time bounded)."""
    out, rejects = [], Counter()
    while len(out) < n:
        f = draw_fields(rng)
        f.update(gp.derive_fields(f))
        if max_tk is not None and f["OC_TILES"] * f["OH"] * f["OW_TILES"] * f["K"] > max_tk:
            rejects["_tk_cap"] += 1
            continue
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


