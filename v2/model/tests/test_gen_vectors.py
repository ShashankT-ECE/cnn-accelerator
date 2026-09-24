"""Step 2.2 part F: v2/scripts/gen_vectors.py ($readmemh vector generator).

Quick generation into a tmp dir (module fixture), then: hex format of every file, MANIFEST
hashes/sizes/lines, read-back of memory images / descriptors / layer and network files vs the
golden, near-tie presence, requant rows == legacy float64 requantize, independent recomputation
of the PE / array / rotator / pool expectations, stream files == gos_addr_stream, determinism
(quick) and --require-clean. ``slow``: full-size generation twice -> identical hashes.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

import common
import gos_addr_stream as ga
import gos_golden as gg
import gos_pack as gp
from lenet5.int8_quant import requantize
from net_config import NET_CONFIGS, NETS

sys.path.insert(0, str(common.V2_ROOT / "scripts"))
import gen_vectors as gv  # noqa: E402


def _gen(out: Path, quick: bool = True) -> dict:
    gv.main(["--out", str(out)] + (["--quick"] if quick else []))
    return json.loads((out / "MANIFEST.json").read_text())


@pytest.fixture(scope="module")
def gen(tmp_path_factory):
    out = tmp_path_factory.mktemp("vectors_q")
    return out, _gen(out)


def _read(out, rel):
    return gp.read_hex(out / rel)


def _hashes(man):
    return {f["path"]: f["sha256"] for f in man["files"]}


# --------------------------------------------------------------------------- #
# Format / manifest
# --------------------------------------------------------------------------- #
def test_hex_format(gen):
    out, man = gen
    assert man["files"]
    for f in man["files"]:
        nd = f["width_bits"] // 4
        pat = re.compile(rf"^[0-9a-f]{{{nd}}}$")
        text = (out / f["path"]).read_text()
        assert "@" not in text and "//" not in text
        lines = text.split("\n")
        assert lines[-1] == "" and len(lines) - 1 == f["lines"] > 0, f["path"]
        assert all(pat.match(x) for x in lines[:-1]), f["path"]


def test_manifest(gen):
    out, man = gen
    assert man["generator"] == "v2/scripts/gen_vectors.py"
    assert man["git_commit"] == common.git_commit() and man["git_dirty"] == common.git_dirty()
    assert man["mode"] == "quick" and man["seeds"] == gv.SEEDS
    for f in man["files"]:
        p = out / f["path"]
        assert common.sha256_file(p) == f["sha256"] and p.stat().st_size == f["bytes"]
        assert f["description"]
    listed = {f["path"] for f in man["files"]}
    on_disk = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert on_disk - listed == {gv.MARKER, "MANIFEST.json"}
    assert man["total_bytes"] == sum(f["bytes"] for f in man["files"])
    for net in NETS:
        for k in ("quant_params", "hw_requant"):
            assert man["params"][net][k]["sha256"] == common.sha256_file(NET_CONFIGS[net][k])
    assert "timestamp" not in json.dumps(man).lower()


def test_reserved_bits_zero(gen):
    out, _ = gen
    for rel, key in (("unit/pe/pe_cycles.hex", "pe"), ("unit/pool/pool_cols.hex", "pool"),
                     ("unit/requant/lenet5_requant.hex", "requant"),
                     ("lenet5/layers/L0_conv1/issue.hex", "issue"),
                     ("cifar10/layers/L1_conv2/drain.hex", "drain")):
        w, lay = gv.LAYOUTS[key]
        assert gv.reserved_bits_zero(_read(out, rel), w, lay), rel


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(1)
    for w, lay in gv.LAYOUTS.values():
        f = {}
        for name, _, width, signed in lay:
            lo, hi = (-(1 << (width - 1)), 1 << (width - 1)) if signed else (0, 1 << width)
            f[name] = rng.integers(lo, hi, size=50)
        back = gv.unpack_records(gv.pack_records(f, lay), lay)
        for name in f:
            assert np.array_equal(back[name], f[name]), name


# --------------------------------------------------------------------------- #
# Read-back vs golden
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("net", NETS)
def test_net_images(gen, net):
    out, _ = gen
    assert _read(out, f"{net}/wgt.hex") == [int(x) for x in gp.pack_wgt(net)[0]]
    qp, _ = gp.pack_qparam(net)
    e, o = gp.qparam_banks(qp)
    assert _read(out, f"{net}/qparam.hex") == [int(x) for x in qp]
    assert _read(out, f"{net}/qparam_e.hex") == [int(x) for x in e]
    assert _read(out, f"{net}/qparam_o.hex") == [int(x) for x in o]
    _, words = gp.make_descriptors(net)
    assert _read(out, f"{net}/desc.hex") == [int(x) for x in words.reshape(-1)]
    assert _read(out, f"{net}/n_layers.hex") == [len(words)]


@pytest.mark.parametrize("net", NETS)
def test_layer_and_net_files(gen, net):
    out, man = gen
    n_layer = len(man["image_indices"]["layer"])
    n_net = len(man["image_indices"]["network"])
    x_fp, y = gg.load_test_set(net, max(n_layer, n_net))
    x = gg.quantize_input(net, x_fp)
    gold = gg.run_net(net, x)
    descs, words = gp.make_descriptors(net)
    prev = "input"
    for i, (d, w) in enumerate(zip(descs, words)):
        ld = f"{net}/layers/{gv.layer_dir(i, d['name'])}"
        assert _read(out, f"{ld}/desc.hex") == [int(v) for v in w]
        st = ga.addr_stream(w)
        tk = _read(out, f"{ld}/expect_tk.hex")
        assert tk[2] == tk[0] * tk[1] == st.n_issues and tk[0] == st.n_tiles
        iss = gv.unpack_records(_read(out, f"{ld}/issue.hex"), gv.ISSUE_LAYOUT)
        for b in range(8):
            assert np.array_equal(iss[f"rd_addr[{b}]"], st.issues["rd_addr"][:, b])
        for f in ("wgt_addr", "rot", "first", "last", "row_mask", "tile", "k"):
            assert np.array_equal(iss[f], st.issues[f].astype(np.int64)), f
        dr = gv.unpack_records(_read(out, f"{ld}/drain.hex"), gv.DRAIN_LAYOUT)
        for f in ("word", "be", "we", "kind", "qp_idx", "logit_idx", "ch_valid", "buf"):
            assert np.array_equal(dr[f], st.drain[f].astype(np.int64)), f
        for n in range(n_layer):
            a_in = np.array(_read(out, f"{ld}/img{n}/act_in.hex"), dtype=np.uint64)
            assert a_in.size == d["IN_END"] + 1
            assert np.array_equal(gp.unpack_act(a_in, d["IC"], d["IH"], d["IW"]), gold[prev][n])
            if d["out_raw"]:
                lg = np.array(_read(out, f"{ld}/img{n}/logit16.hex"), dtype=np.uint32).view(np.int32)
                assert np.array_equal(lg[:d["OC"]], gold["v"][n]) and not lg[d["OC"]:].any()
            else:
                a_out = np.array(_read(out, f"{ld}/img{n}/act_out.hex"), dtype=np.uint64)
                assert a_out.size == d["OUT_END"] + 1
                assert np.array_equal(gp.unpack_act(a_out, d["OC"], d["OUT_H"], d["OUT_W"]),
                                      gold[d["name"]][n])
        prev = d["name"]
    for n in range(n_net):
        a0 = np.array(_read(out, f"{net}/net/img{n}/act0.hex"), dtype=np.uint64)
        assert np.array_equal(a0, gp.pack_act_words(x[n]))
        lg = np.array(_read(out, f"{net}/net/img{n}/logit10.hex"), dtype=np.uint32).view(np.int32)
        assert np.array_equal(lg, gold["v"][n])
        lg16 = np.array(_read(out, f"{net}/net/img{n}/logit16.hex"), dtype=np.uint32).view(np.int32)
        assert np.array_equal(lg16[:10], lg) and not lg16[10:].any()
    assert _read(out, f"{net}/net/pred.hex") == [int(p) for p in gold["pred"][:n_net]]
    assert _read(out, f"{net}/net/label.hex") == [int(v) for v in y[:n_net]]


# --------------------------------------------------------------------------- #
# Requant
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("net", NETS)
def test_requant_rows(gen, net):
    out, man = gen
    r = gv.unpack_records(_read(out, f"unit/requant/{net}_requant.hex"), gv.REQUANT_LAYOUT)
    params = gp.load_params(net)
    q = np.load(NET_CONFIGS[net]["quant_params"])
    descs, _ = gp.make_descriptors(net)
    ch_map = {}
    for d in descs:
        if d["out_raw"]:
            continue
        P = params[d["name"]]
        M = q[f"{d['name']}_M"]
        for c in range(d["OC"]):
            ch_map[d["QP_BASE"] + c] = (d["name"], c, int(P["q_b"][c]), int(P["m"][c]),
                                        int(P["s"][c]), float(M[c]), d["K"])
    assert len(r["v"]) == man["summary"]["unit"]["requant"][net]["rows"]
    assert np.array_equal(r["v"], r["acc"] + r["q_bias"])
    assert np.array_equal(r["q_relu"], np.maximum(r["q"], 0))
    for i in np.unique(r["qp_ch"]):
        sel = r["qp_ch"] == i
        _, _, qb, m, s, M, K = ch_map[int(i)]
        assert np.all(r["q_bias"][sel] == qb) and np.all(r["m"][sel] == m) and np.all(r["s"][sel] == s)
        assert np.all(np.abs(r["acc"][sel]) <= K * 16384)
        assert np.array_equal(requantize(r["v"][sel], M).astype(np.int64), r["q"][sel])
    ties = {(ch_map[int(c)][0], ch_map[int(c)][1], int(v))
            for c, v, t in zip(r["qp_ch"], r["v"], r["near_tie"]) if t}
    for net_, layer, c, v in gv.REQUIRED_NEAR_TIES:
        if net_ == net:
            assert (layer, c, v) in ties and (layer, c, -v) in ties
    n = man["sizes"]["requant_samples"]
    assert not r["near_tie"][:n].any() and r["near_tie"][n:].all()
    assert r["exact_region"][1:n:2].all()                   # odd sampled rows: exact region


# --------------------------------------------------------------------------- #
# Unit expectations recomputed independently
# --------------------------------------------------------------------------- #
def test_pe(gen):
    out, _ = gen
    r = gv.unpack_records(_read(out, "unit/pe/pe_cycles.hex"), gv.PE_LAYOUT)
    acc, res = 0, []
    for a, w, f, l, e in zip(r["a"], r["w"], r["first"], r["last"], r["acc"]):
        acc = a * w if f else acc + a * w
        assert acc == e
        if l:
            res.append(acc)
    fl = r["first"].astype(bool)
    assert fl[0] and np.array_equal(r["last"].astype(bool)[:-1], fl[1:]) and r["last"][-1]
    got = np.array(_read(out, "unit/pe/pe_results.hex"), dtype=np.uint32).view(np.int32)
    assert np.array_equal(got, res)
    lengths = np.diff(np.append(np.flatnonzero(fl), fl.size))
    assert 1 in lengths and 800 in lengths
    assert max(abs(x) for x in res) == 800 * 128 * 128


def test_array(gen):
    out, _ = gen
    A = gv.word_to_bytes(np.array(_read(out, "unit/array/array_a.hex"), dtype=np.uint64)).astype(np.int64)
    W = gv.word_to_bytes(np.array(_read(out, "unit/array/array_w.hex"), dtype=np.uint64)).astype(np.int64)
    ctl = np.array(_read(out, "unit/array/array_ctl.hex"))
    Ks = _read(out, "unit/array/array_k.hex")
    acc = np.array(_read(out, "unit/array/array_acc.hex"), dtype=np.uint32).view(np.int32).reshape(-1, 8, 8)
    assert {1, 25, 800} <= set(Ks) and len(Ks) == acc.shape[0] and sum(Ks) == A.shape[0]
    k0 = 0
    for t, K in enumerate(Ks):
        assert ctl[k0] & 1 and ctl[k0 + K - 1] & 2
        assert not (ctl[k0 + 1:k0 + K] & 1).any() and not (ctl[k0:k0 + K - 1] & 2).any()
        exp = np.einsum("kr,kj->rj", A[k0:k0 + K], W[k0:k0 + K])
        assert np.array_equal(acc[t], exp), t
        k0 += K
    assert acc.max() == 800 * 128 * 128 and acc.min() == -800 * 127 * 128


def test_rotator(gen):
    out, _ = gen
    banks = gv.word_to_bytes(np.array(_read(out, "unit/rotator/rot_banks.hex"), dtype=np.uint64))
    rows = gv.word_to_bytes(np.array(_read(out, "unit/rotator/rot_rows.hex"), dtype=np.uint64))
    kx = _read(out, "unit/rotator/rot_kx.hex")
    assert set(kx) == set(range(8))
    for bb, rr, k in zip(banks, rows, kx):
        for b in range(8):
            assert rr[(b - k) % 8] == bb[b]


def test_pool(gen):
    out, _ = gen
    r = gv.unpack_records(_read(out, "unit/pool/pool_cols.hex"), gv.POOL_LAYOUT)
    d0 = np.stack([r[f"dy0[{i}]"] for i in range(8)], 1)
    d1 = np.stack([r[f"dy1[{i}]"] for i in range(8)], 1)
    h = np.stack([r[f"h[{i}]"] for i in range(4)], 1)
    q = np.maximum(d0, d1).reshape(-1, 4, 2).max(2)
    assert np.array_equal(h, q)
    for rem, half, be in zip(r["rem"], r["half"], r["be"]):
        assert rem % 2 == 0 and be == ga._pool_be(int(rem), int(half))
    assert set(r["rem"]) >= {2, 4, 6, 8} and set(r["half"]) == {0, 1}
    assert (d0[r["signed_data"] == 0] >= 0).all() and (d0[r["signed_data"] == 1] < 0).any()


# --------------------------------------------------------------------------- #
# Determinism / clean-tree guard
# --------------------------------------------------------------------------- #
def test_determinism_quick(gen, tmp_path):
    _, man = gen
    man2 = _gen(tmp_path / "again")
    assert _hashes(man) == _hashes(man2)


def test_require_clean(tmp_path, monkeypatch):
    monkeypatch.setattr(gv, "git_dirty", lambda: True)
    with pytest.raises(SystemExit):
        gv.main(["--out", str(tmp_path / "x"), "--quick", "--require-clean"])
    assert not (tmp_path / "x").exists()


def test_refuses_foreign_dir(tmp_path):
    (tmp_path / "keep.txt").write_text("user file")
    with pytest.raises(SystemExit):
        gv.main(["--out", str(tmp_path), "--quick"])
    assert (tmp_path / "keep.txt").exists()


@pytest.mark.slow
def test_determinism_full(tmp_path):
    a = _gen(tmp_path / "a", quick=False)
    b = _gen(tmp_path / "b", quick=False)
    assert a["mode"] == "full" and _hashes(a) == _hashes(b)
    assert a["summary"]["unit"]["requant"]["lenet5"]["sampled"] == 100_000
    assert len(a["image_indices"]["layer"]) == 3 and len(a["image_indices"]["network"]) == 10
