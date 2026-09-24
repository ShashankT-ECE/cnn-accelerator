"""Step 2.2 part E: controller address stream (gos_addr_stream).

* issue count == T*K == gos_cycle_model compute cycles; drain steps == 8*T.
* every issue / drain field recomputed independently with the closed-form formulas
  (multiplies allowed here) on both nets and on random accepted shapes.
* the written (bank, word) set equals exactly the output map's footprint (gos_pack.act_addr).
* the per-cycle generator iter_issues equals the assembled stream.
* the address path has no '*', '/', '//', '%', '**' (AST inspection).
* replay: the stream replayed through the tile-model memories reproduces golden output.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

import gos_addr_stream as ga
import gos_cycle_model as cm
import gos_pack as gp
from net_config import NET_CONFIGS

from test_gos_tile_model import draw_accepted, make_params, run_case

NETS = ("lenet5", "cifar10")
SPEC_COMPUTE = {"lenet5": [2800, 6000, 6000, 1320, 168], "cifar10": [33600, 64000, 6400, 128]}


def _net_descs(net):
    _, words = gp.make_descriptors(net)
    return [(gp.decode_descriptor(w), w) for w in words]


def _random_descs(seed, n):
    cases, _ = draw_accepted(np.random.default_rng(seed), n)
    return [(gp.decode_descriptor(w), w) for _, w in cases]


# --------------------------------------------------------------------------- #
# Closed-form reference (test helper: multiplication/division allowed)
# --------------------------------------------------------------------------- #
def closed_form_issues(d):
    pool = d["pool_en"]
    ndy = 2 if pool else 1
    nrows = d["OH"] // 2 if pool else d["OH"]
    oct_, row, oxt, dy, ic, ky, kx = [a.ravel() for a in np.indices(
        (d["OC_TILES"], nrows, d["OW_TILES"], ndy, d["IC"], d["KH"], d["KW"]))]
    oy = 2 * row + dy if pool else row
    ox0 = 8 * oxt
    k = (ic * d["KH"] + ky) * d["KW"] + kx
    rowbase = (ic * d["IH"] + oy + ky) * d["IN_WPR"]
    b = np.arange(8)
    rd = rowbase[:, None] + ox0[:, None] // 8 + (b[None, :] < kx[:, None])
    rows_valid = (ox0[:, None] + b[None, :]) < d["OW"]
    mask = (rows_valid * (1 << b)[None, :]).sum(1)
    K = d["K"]
    tile = ((oct_ * nrows + row) * d["OW_TILES"] + oxt) * ndy + dy
    return {"tile": tile, "oc_tile": oct_, "oy": oy, "ox_tile": oxt, "dy": dy, "k": k, "ic": ic,
            "ky": ky, "kx": kx, "first": k == 0, "last": k == K - 1, "rd_addr": rd, "rot": kx,
            "wgt_addr": d["WGT_BASE"] + oct_ * K + k, "row_mask": mask}


def closed_form_drain(d):
    pool = d["pool_en"]
    ndy = 2 if pool else 1
    nrows = d["OH"] // 2 if pool else d["OH"]
    oct_, row, oxt, dy, j = [a.ravel() for a in np.indices(
        (d["OC_TILES"], nrows, d["OW_TILES"], ndy, 8))]
    ch = 8 * oct_ + j
    valid = ch < d["OC"]
    ox0 = 8 * oxt
    oy = 2 * row + dy if pool else row
    b = np.arange(8)
    n = ch.size
    kind = np.full(n, ga.KIND_ACT)
    word = np.zeros(n, dtype=np.int64)
    be = np.zeros(n, dtype=np.int64)
    if d["out_raw"]:
        kind[:] = ga.KIND_LOGIT
        we = valid
    else:
        if pool:
            kind[dy == 0] = ga.KIND_POOL_STORE
            px = ox0[:, None] // 2 + np.arange(4)[None, :]            # pooled x, i = 0..3
            ok = px < d["OUT_W"]
            be_pool = (ok * (1 << (px % 8))).sum(1)
            w_pool = (ch * d["OUT_H"] + row) * d["OUT_WPR"] + (ox0 // 2) // 8
            act = dy == 1
            be = np.where(act & valid, be_pool, 0)
            word = np.where(act, w_pool, 0)
        else:
            rows_valid = (ox0[:, None] + b[None, :]) < d["OW"]
            be = np.where(valid, (rows_valid * (1 << b)[None, :]).sum(1), 0)
            word = (ch * d["OUT_H"] + oy) * d["OUT_WPR"] + ox0 // 8
        we = be != 0
    tile = ((oct_ * nrows + row) * d["OW_TILES"] + oxt) * ndy + dy
    return {"tile": tile, "oc_tile": oct_, "oy": oy, "out_row": row, "ox_tile": oxt, "dy": dy,
            "j": j, "ch": ch, "ch_valid": valid, "qp_idx": d["QP_BASE"] + ch, "kind": kind,
            "we": we, "buf": np.full(n, 1 - d["in_sel"]), "word": word, "be": be,
            "logit_idx": np.where(kind == ga.KIND_LOGIT, ch, 0)}


def _compare(stream, ref, what):
    for k, v in ref.items():
        got = stream[k].astype(np.int64)
        assert np.array_equal(got, np.asarray(v).astype(np.int64)), f"{what}: field {k}"


def _check_all(d, w):
    st = ga.addr_stream(w)
    _compare(st.issues, closed_form_issues(d), "issues")
    _compare(st.drain, closed_form_drain(d), "drain")
    # the ACT footprint written == the output map, each byte exactly once
    if not d["out_raw"]:
        dr = st.drain[st.drain["we"]]
        got = sorted((b, int(r["word"])) for r in dr for b in range(8) if (int(r["be"]) >> b) & 1)
        want = sorted(gp.act_addr(c, y, x, d["OUT_H"], d["OUT_W"])
                      for c in range(d["OC"]) for y in range(d["OUT_H"]) for x in range(d["OUT_W"]))
        assert got == want
    else:
        idx = st.drain["logit_idx"][st.drain["we"]]
        assert sorted(idx.tolist()) == list(range(d["OC"]))
    return st


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("net", NETS)
def test_counts_match_cycle_model(net):
    for (d, w), L, spec in zip(_net_descs(net), NET_CONFIGS[net]["layers"], SPEC_COMPUTE[net]):
        st = ga.addr_stream(w)
        c = st.counts()
        T = cm.layer_tiles(L)["T"]
        assert c["issues"] == T * L["K"] == cm.layer_cycles(L, c_pipe=None)["compute_cycles"] == spec
        assert c["tiles"] == T and c["drain_steps"] == 8 * T
        if d["out_raw"]:
            assert c["logit_writes"] == d["OC"] and c["act_writes"] == 0
        elif d["pool_en"]:
            assert c["pool_stores"] == 4 * T
            assert c["act_writes"] == d["OC"] * d["OUT_H"] * d["OW_TILES"]
        else:
            assert c["act_writes"] == d["OC"] * d["OH"] * d["OW_TILES"]


@pytest.mark.parametrize("net", NETS)
def test_closed_form_nets(net):
    for d, w in _net_descs(net):
        st = _check_all(d, w)
        # highest ACT read word agrees with gos_pack.max_act_read_word
        assert int(st.issues["rd_addr"].max()) == gp.max_act_read_word(d)


def test_closed_form_random_shapes():
    for d, w in _random_descs(11, 60):
        st = _check_all(d, w)
        c = st.counts()
        assert c["issues"] == c["T_times_K"] == c["cycle_model_compute"]
        assert int(st.issues["rd_addr"].max()) == gp.max_act_read_word(d)


def test_decoded_dict_and_words_agree():
    d, w = _net_descs("lenet5")[1]
    a, b = ga.addr_stream(w), ga.addr_stream(d)
    assert np.array_equal(a.issues, b.issues) and np.array_equal(a.drain, b.drain)


def test_first_last_and_tile_order():
    d, w = _net_descs("cifar10")[1]
    iss = ga.addr_stream(w).issues
    K = d["K"]
    assert np.flatnonzero(iss["first"]).tolist() == list(range(0, iss.size, K))
    assert np.flatnonzero(iss["last"]).tolist() == list(range(K - 1, iss.size, K))
    # loop order oc_tile -> oy pair -> ox_tile -> dy -> k (lexicographic, strictly increasing)
    pool_row = iss["oy"].astype(np.int64) >> 1
    key = np.stack([iss["oc_tile"], pool_row, iss["ox_tile"], iss["dy"], iss["k"]], 1).astype(np.int64)
    assert np.all(np.diff(key @ np.array([1 << 40, 1 << 30, 1 << 24, 1 << 22, 1])) > 0)


def _iter_as_arrays(w):
    rows = list(ga.iter_issues(w))
    names = ("tile", "oc_tile", "oy", "ox_tile", "dy", "k", "ic", "ky", "kx", "first", "last",
             "rd_addr", "rot", "wgt_addr", "row_mask")
    return {n: np.array([r[i] for r in rows]) for i, n in enumerate(names)}


@pytest.mark.parametrize("net", NETS)
def test_iter_issues_equals_stream_nets(net):
    for _, w in _net_descs(net):
        _compare(ga.addr_stream(w).issues, _iter_as_arrays(w), "iter_issues")


def test_iter_issues_equals_stream_random():
    for d, w in _random_descs(12, 40):
        if d["OC_TILES"] * d["OH"] * d["OW_TILES"] * d["K"] > 60000:
            continue
        _compare(ga.addr_stream(w).issues, _iter_as_arrays(w), "iter_issues")


ADDR_PATH = ("_row_mask", "_pool_be", "_k_program", "_tile_walk", "_assemble_issues",
             "iter_issues")
FORBIDDEN_OPS = (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult)
FORBIDDEN_CALLS = {"multiply", "divide", "floor_divide", "mod", "remainder", "divmod", "prod",
                   "dot", "matmul", "einsum", "power", "ceil", "floor", "true_divide"}


def test_address_path_has_no_multiply_divide_modulo():
    for name in ADDR_PATH:
        tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(ga, name))))
        bad = [ast.dump(n) for n in ast.walk(tree)
               if isinstance(n, (ast.BinOp, ast.AugAssign)) and isinstance(n.op, FORBIDDEN_OPS)]
        assert not bad, f"{name}: arithmetic operator in the address path: {bad}"
        calls = [n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                 for n in ast.walk(tree) if isinstance(n, ast.Call)]
        assert not FORBIDDEN_CALLS.intersection(calls), (name, calls)


def test_descriptor_refusals():
    d, w = _net_descs("lenet5")[0]
    bad = w.copy()
    bad[3] = (27 << 16) | 28                                      # OW odd with pool
    with pytest.raises(ValueError):
        ga.addr_stream(bad)
    f = {"IC": 8, "OC": 4, "IH": 6, "IW": 6, "KH": 5, "KW": 5, "OH": 2, "OW": 2, "WGT_BASE": 0,
         "QP_BASE": 0, "relu_en": 0, "pool_en": 0, "out_raw": 1, "in_sel": 0}
    f.update(gp.derive_fields(f))
    w2 = gp.encode_descriptor(f)
    assert gp.check_descriptor(w2)[0]                             # the RTL checker accepts it ...
    with pytest.raises(ValueError, match="1x1"):                  # ... the LOGIT contract does not
        ga.addr_stream(w2)


@pytest.mark.slow
def test_replay_random_shapes_through_memories():
    """Standalone replay: the stream drives the scalar tile-model executor on a few random
    shapes (garbage in all unwritten memory) and reproduces gos_golden."""
    rng = np.random.default_rng(13)
    done = 0
    while done < 5:
        (f, w), = draw_accepted(rng, 1)[0]
        if f["OC_TILES"] * f["OH"] * f["OW_TILES"] * f["K"] > 30000:
            continue
        run_case(f, w, make_params(rng, f), "loop", rng)
        done += 1
