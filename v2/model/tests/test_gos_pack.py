"""Tests for gos_pack: the ACT / WGT / QPARAM / DESCRIPTOR contract (v2/docs/FORMATS.md)."""
import numpy as np
import pytest

import gos_pack as gp
from net_config import NET_CONFIGS, NETS

RNG_SEED = 20260924
SHAPES = [(1, 1, 1), (120, 1, 1), (84, 1, 1), (1, 32, 32), (3, 32, 32), (6, 14, 14),
          (16, 5, 5), (32, 28, 28), (2, 3, 9), (4, 7, 15), (5, 2, 17), (1, 1, 8), (2, 1, 7)]


# ---- ACT ---------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
def test_act_round_trip(shape):
    rng = np.random.default_rng(RNG_SEED + sum(shape))
    x = rng.integers(-128, 128, size=shape, dtype=np.int16).astype(np.int8)
    x.flat[0], x.flat[-1] = -128, 127
    banks = gp.pack_act(x)
    assert banks.shape == (8, gp.ACT_DEPTH) and banks.dtype == np.uint8
    np.testing.assert_array_equal(gp.unpack_act(banks, *shape), x)
    words = gp.act_banks_to_words(banks)
    assert words.dtype == np.uint64
    np.testing.assert_array_equal(gp.unpack_act(words, *shape), x)
    np.testing.assert_array_equal(gp.act_words_to_banks(words), banks)
    pw = gp.pack_act_words(x)
    assert pw.size == gp.act_depth(*shape)
    np.testing.assert_array_equal(pw, words[:pw.size])
    # everything outside the map (x tails and words beyond the depth) is 0
    mask = np.zeros_like(banks, dtype=bool)
    C, H, W = shape
    for c in range(C):
        for y in range(H):
            for xx in range(W):
                b, w = gp.act_addr(c, y, xx, H, W)
                assert banks[b, w] == np.uint8(x[c, y, xx].view(np.uint8))
                assert not mask[b, w], "two elements map to one byte"
                mask[b, w] = True
    assert np.all(banks[~mask] == 0)


@pytest.mark.parametrize("shape", SHAPES)
def test_act_addr_formula(shape):
    C, H, W = shape
    wpr = -(-W // 8)
    for c in range(C):
        for y in range(H):
            for x in range(W):
                assert gp.act_addr(c, y, x, H, W) == (x % 8, (c * H + y) * wpr + x // 8)
    assert gp.act_depth(C, H, W) == C * H * wpr


def test_act_1x1_layout_is_fc_input():
    for c in range(256):
        assert gp.act_addr(c, 0, 0, 1, 1) == (0, c)
    x = np.arange(-60, 60, dtype=np.int8).reshape(120, 1, 1)
    words = gp.pack_act_words(x)
    np.testing.assert_array_equal(words, x.ravel().view(np.uint8).astype(np.uint64))


def test_act_ps_byte_lanes():
    x = np.arange(16, dtype=np.int8).reshape(1, 1, 16) - 8     # -8..7
    w = gp.pack_act_words(x)
    assert w.size == 2
    for word in range(2):
        for b in range(8):
            assert (int(w[word]) >> (8 * b)) & 0xFF == int(x[0, 0, 8 * word + b]) & 0xFF


def test_act_overflow_rejected():
    assert gp.act_depth(37, 28, 28) > gp.ACT_DEPTH
    with pytest.raises(AssertionError):
        gp.pack_act(np.zeros((37, 28, 28), dtype=np.int8))
    gp.pack_act(np.zeros((36, 28, 28), dtype=np.int8))            # 4032 words: fits


# ---- WGT -----------------------------------------------------------------------
@pytest.mark.parametrize("net", NETS)
def test_wgt_unpack_reproduces_q_w(net):
    words, bases = gp.pack_wgt(net)
    assert words.dtype == np.uint64
    params = gp.load_params(net)
    Ls = NET_CONFIGS[net]["layers"]
    assert bases[0] == 0
    for i, L in enumerate(Ls):
        n = -(-L["OC"] // 8) * L["K"]
        end = bases[i + 1] if i + 1 < len(Ls) else words.size
        assert end - bases[i] == n
        np.testing.assert_array_equal(gp.unpack_wgt(words, L, bases[i]), params[L["name"]]["q_w"])
        assert np.all(gp.wgt_tail_bytes(words, L, bases[i]) == 0)
        # direct formula spot check on every weight
        w = params[L["name"]]["q_w"]
        OC, IC, KH, KW = w.shape
        for oc in range(OC):
            for ic in range(IC):
                for ky in range(KH):
                    for kx in range(KW):
                        k = (ic * KH + ky) * KW + kx
                        word = int(words[bases[i] + (oc // 8) * L["K"] + k])
                        assert (word >> (8 * (oc % 8))) & 0xFF == int(w[oc, ic, ky, kx]) & 0xFF


def test_wgt_totals_from_spec():
    assert gp.memory_usage("lenet5")["wgt_words"] == 7813
    assert gp.memory_usage("cifar10")["wgt_words"] == 10028
    for net in NETS:
        words, _ = gp.pack_wgt(net)
        assert words.size == gp.memory_usage(net)["wgt_words"] <= gp.WGT_DEPTH


# ---- QPARAM --------------------------------------------------------------------
@pytest.mark.parametrize("net", NETS)
def test_qparam_decode(net):
    words, bases = gp.pack_qparam(net)
    params = gp.load_params(net)
    Ls = NET_CONFIGS[net]["layers"]
    assert words.size == 2 * sum(L["OC"] for L in Ls) <= gp.QP_WORDS
    assert bases[0] == 0
    ev, od = gp.qparam_banks(words)
    assert ev.size == od.size == words.size // 2 <= gp.QP_CHANNELS
    for i, L in enumerate(Ls):
        P = params[L["name"]]
        q_b, m, s = gp.unpack_qparam(words, bases[i], L["OC"])
        np.testing.assert_array_equal(q_b, P["q_b"])
        np.testing.assert_array_equal(m, P["m"])
        np.testing.assert_array_equal(s, P["s"])
        if L["final"]:
            assert np.all(m == 0) and np.all(s == 0)
        else:
            assert np.all(m > 0) and np.all(s > 0)
        for oc in range(L["OC"]):
            ch = bases[i] + oc
            w0, w1 = int(words[2 * ch]), int(words[2 * ch + 1])
            assert w0 >> 32 == int(P["m"][oc])
            assert w0 & 0xFFFFFFFF == int(P["q_b"][oc]) & 0xFFFFFFFF
            assert w1 == int(P["s"][oc])
            assert int(ev[ch]) == w0 and int(od[ch]) == w1
    hw = np.load(NET_CONFIGS[net]["hw_requant"])
    for L in Ls:
        if not L["final"]:
            np.testing.assert_array_equal(params[L["name"]]["m"], hw[L["name"] + "_m"])


def test_qparam_channel_counts():
    # APPROVED: final layer included (m = s = 0)
    for net in NETS:
        u = gp.memory_usage(net)
        assert u["qparam_channels"] == sum(L["OC"] for L in NET_CONFIGS[net]["layers"])
        assert u["qparam_words"] == 2 * u["qparam_channels"] <= gp.QP_WORDS
    assert gp.memory_usage("lenet5")["qparam_words"] == 472
    assert gp.memory_usage("cifar10")["qparam_words"] == 276


# ---- descriptors ---------------------------------------------------------------
@pytest.mark.parametrize("net", NETS)
def test_descriptor_round_trip(net):
    descs, words = gp.make_descriptors(net)
    assert words.dtype == np.uint32 and words.shape == (len(descs), 16) and len(descs) <= 8
    for i, (d, w) in enumerate(zip(descs, words)):
        dec = gp.decode_descriptor(w)
        for k in gp.ALL_FIELDS:
            assert dec[k] == d[k], (d["name"], k)
        assert dec["flags_reserved"] == 0 and dec["w2_reserved"] == 0
        np.testing.assert_array_equal(gp.encode_descriptor({k: dec[k] for k in gp.ALL_FIELDS}), w)
        np.testing.assert_array_equal(
            gp.encode_descriptor({k: dec[k] for k in gp.RAW_FIELDS + gp.FLAG_FIELDS}), w)
        assert d["in_sel"] == i % 2
        L = NET_CONFIGS[net]["layers"][i]
        assert (d["relu_en"], d["pool_en"], d["out_raw"]) == (int(L["relu"]), int(L["pool"]),
                                                             int(L["final"]))


@pytest.mark.parametrize("net", NETS)
def test_descriptor_word_packing(net):
    descs, words = gp.make_descriptors(net)
    for d, w in zip(descs, words):
        w = [int(v) for v in w]
        assert w[0] == d["IC"] | d["OC"] << 16
        assert w[1] == d["IH"] | d["IW"] << 16
        assert w[2] == d["KH"] | d["KW"] << 8
        assert w[3] == d["OH"] | d["OW"] << 16
        assert w[4] == d["WGT_BASE"] and w[5] == d["QP_BASE"]
        assert w[6] == d["relu_en"] | d["pool_en"] << 1 | d["out_raw"] << 2 | d["in_sel"] << 3
        assert w[7] == d["K"]
        assert w[8] == d["IN_WPR"] | d["IN_PLANE"] << 16
        assert w[9] == d["OC_TILES"] | d["OW_TILES"] << 16
        assert w[10] == d["OUT_W"] | d["OUT_H"] << 16
        assert w[11] == d["OUT_WPR"] | d["OUT_PLANE"] << 16
        assert (w[12], w[13], w[14], w[15]) == (d["WGT_END"], d["IN_END"], d["OUT_END"], d["QP_END"])


@pytest.mark.parametrize("net", NETS)
def test_descriptor_derived_values(net):
    descs, _ = gp.make_descriptors(net)
    wbases = gp.pack_wgt(net)[1]
    qbases = gp.pack_qparam(net)[1]
    for i, d in enumerate(descs):
        L = NET_CONFIGS[net]["layers"][i]
        c8 = lambda v: -(-v // 8)  # noqa: E731
        assert d["K"] == L["IC"] * L["KH"] * L["KW"]
        assert d["IN_WPR"] == c8(L["IW"]) and d["IN_PLANE"] == L["IH"] * c8(L["IW"])
        assert d["OC_TILES"] == c8(L["OC"]) and d["OW_TILES"] == c8(L["OW"])
        ow, oh = (L["OW"] // 2, L["OH"] // 2) if L["pool"] else (L["OW"], L["OH"])
        assert (d["OUT_W"], d["OUT_H"]) == (ow, oh)
        assert d["OUT_WPR"] == c8(ow) and d["OUT_PLANE"] == oh * c8(ow)
        assert d["WGT_BASE"] == wbases[i] and d["QP_BASE"] == qbases[i]
        assert d["WGT_END"] == wbases[i] + c8(L["OC"]) * d["K"] - 1
        assert d["IN_END"] + 1 == gp.act_depth(L["IC"], L["IH"], L["IW"])
        assert d["OUT_END"] + 1 == gp.act_depth(L["OC"], oh, ow)
        assert d["QP_END"] == qbases[i] + L["OC"] - 1
    # packed consecutively: each layer starts right after the previous one ends
    for a, b in zip(descs, descs[1:]):
        assert b["WGT_BASE"] == a["WGT_END"] + 1 and b["QP_BASE"] == a["QP_END"] + 1
    last = descs[-1]
    assert last["WGT_END"] + 1 == gp.memory_usage(net)["wgt_words"]
    assert last["QP_END"] + 1 == gp.memory_usage(net)["qparam_channels"]


@pytest.mark.parametrize("field", gp.DERIVED_FIELDS)
def test_encode_rejects_inconsistent_derived(field):
    d = dict(gp.make_descriptors("lenet5")[0][1])
    d.pop("name")
    d[field] += 1
    with pytest.raises(AssertionError):
        gp.encode_descriptor(d)


@pytest.mark.parametrize("field,val", [("IC", 1 << 16), ("OC", 1 << 16), ("KH", 256),
                                       ("KW", 256), ("IW", 1 << 16)])
def test_encode_rejects_field_overflow(field, val):
    d = {k: v for k, v in gp.make_descriptors("lenet5")[0][1].items()
         if k in gp.RAW_FIELDS + gp.FLAG_FIELDS}
    d[field] = val
    with pytest.raises(AssertionError):
        gp.encode_descriptor(d)


# ---- config checker -------------------------------------------------------------
@pytest.mark.parametrize("net", NETS)
def test_checker_accepts_real(net):
    _, words = gp.make_descriptors(net)
    for w in words:
        assert gp.check_descriptor(w) == (True, [])


def _set(words, field, value):
    """Overwrite one field in a copy of the raw words (no re-derivation)."""
    w = [int(v) for v in words]
    if field in gp.FLAG_BITS:
        bit = gp.FLAG_BITS[field]
        w[6] = (w[6] & ~(1 << bit)) | (int(value) << bit)
    else:
        wi, lsb, width = gp.DESC_LAYOUT[field]
        mask = ((1 << width) - 1) << lsb
        assert 0 <= value < (1 << width)
        w[wi] = (w[wi] & ~mask) | (value << lsb)
    return np.array(w, dtype=np.uint32)


def _real(net, name):
    descs, words = gp.make_descriptors(net)
    i = [d["name"] for d in descs].index(name)
    return words[i]


CORRUPTIONS = [
    ("lenet5", "conv1", {"OH": 27}, "pool_en with odd OH"),
    ("lenet5", "conv3", {"OW": 9}, "pool_en with odd OW"),
    ("lenet5", "conv1", {"K": 7}, "K < 8"),
    ("lenet5", "fc2", {"WGT_END": 16384}, "WGT_END > 16383"),
    ("cifar10", "conv2", {"IN_END": 4096}, "IN_END > 4095"),
    ("cifar10", "conv1", {"OUT_END": 4096}, "OUT_END > 4095"),
    ("lenet5", "fc2", {"QP_END": 256}, "QP_END > 255"),
    ("lenet5", "conv5", {"OH": 6}, "OH > IH"),
    ("lenet5", "conv5", {"OW": 6}, "OW > IW"),
    ("lenet5", "fc2", {"OC": 17}, "out_raw with OC > 16"),
    ("cifar10", "fc", {"OC": 17}, "out_raw with OC > 16"),
    ("cifar10", "conv3", {"out_raw": 1}, "out_raw with OC > 16"),   # OC = 64
]


@pytest.mark.parametrize("net,layer,change,reason", CORRUPTIONS)
def test_checker_rejects(net, layer, change, reason):
    w = _real(net, layer)
    for k, v in change.items():
        w = _set(w, k, v)
    ok, reasons = gp.check_descriptor(w)
    assert not ok and reasons == [reason]


def test_checker_boundaries_accepted():
    w = _real("lenet5", "fc2")
    for field, v in [("WGT_END", 16383), ("QP_END", 255), ("IN_END", 4095), ("OUT_END", 4095),
                     ("K", 8), ("OC", 16)]:
        assert gp.check_descriptor(_set(w, field, v)) == (True, [])
    # OH == IH, OW == IW allowed
    w = _real("lenet5", "conv5")
    assert gp.check_descriptor(_set(_set(w, "OH", 5), "OW", 5)) == (True, [])
    # odd OH/OW fine without pooling
    w = _real("lenet5", "conv1")
    w = _set(_set(_set(w, "pool_en", 0), "OH", 27), "OW", 27)
    assert gp.check_descriptor(w) == (True, [])


@pytest.mark.parametrize("field", gp.NONZERO_FIELDS)
def test_checker_rejects_zero(field):
    w = _set(_real("cifar10", "conv2"), field, 0)
    ok, reasons = gp.check_descriptor(w)
    assert not ok and f"{field} == 0" in reasons


def test_checker_uses_no_arithmetic():
    """The checker model contains no *, /, //, % operators (comparator-only RTL)."""
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(gp.check_descriptor)))
    bad = [n for n in ast.walk(tree) if isinstance(n, ast.BinOp)
           and isinstance(n.op, (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow))]
    assert not bad
    for fn in (gp.decode_descriptor,):          # field extraction: shifts and masks only
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.BinOp)
                    and isinstance(n.op, (ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow))]


# ---- memory usage ---------------------------------------------------------------
@pytest.mark.parametrize("net", NETS)
def test_memory_usage(net):
    u = gp.memory_usage(net)
    Ls = NET_CONFIGS[net]["layers"]
    expect = [gp.act_depth(Ls[0]["IC"], Ls[0]["IH"], Ls[0]["IW"])]
    for L in Ls:
        if L["final"]:
            continue
        if L["pool"]:
            expect += [gp.act_depth(L["OC"], L["OH"] // 2, L["OW"] // 2),
                       gp.act_depth(L["OC"], L["OH"], L["OW"])]
        else:
            expect.append(gp.act_depth(L["OC"], L["OH"], L["OW"]))
    assert u["act_max_depth"] == max(expect) <= gp.ACT_DEPTH
    assert u["act_max_input_depth"] == max(gp.act_depth(L["IC"], L["IH"], L["IW"]) for L in Ls)
    assert u["wgt_words"] == sum(-(-L["OC"] // 8) * L["K"] for L in Ls)
    assert u["logits"] == Ls[-1]["OC"] <= gp.N_LOGITS


def test_cifar_unpooled_fallback_fits():
    L = NET_CONFIGS["cifar10"]["layers"][0]
    d = gp.act_depth(L["OC"], L["OH"], L["OW"])
    assert d == 32 * 28 * 4
    assert gp.memory_usage("cifar10")["act_max_depth"] == d <= gp.ACT_DEPTH


# ---- hex files -----------------------------------------------------------------
def test_hex_round_trip(tmp_path):
    words, _ = gp.pack_wgt("lenet5")
    p = gp.write_hex64(tmp_path / "wgt.hex", words)
    lines = p.read_text().splitlines()
    assert len(lines) == words.size and all(len(s) == 16 for s in lines)
    assert gp.read_hex(p) == [int(v) for v in words]
    _, dw = gp.make_descriptors("cifar10")
    p = gp.write_hex32(tmp_path / "desc.hex", dw)
    assert all(len(s) == 8 for s in p.read_text().splitlines())
    assert gp.read_hex(p) == [int(v) for v in dw.ravel()]
    banks = gp.pack_act(np.full((1, 1, 3), -1, dtype=np.int8), depth=4)
    p = gp.write_hex8(tmp_path / "bank0.hex", banks[0])
    assert p.read_text().splitlines() == ["ff", "00", "00", "00"]
    with pytest.raises(AssertionError):
        gp.hex_lines([-1], 8)
    with pytest.raises(AssertionError):
        gp.hex_lines([256], 8)
    assert gp.hex_lines([0x0123456789ABCDEF], 64) == ["0123456789abcdef"]


@pytest.mark.parametrize("net", NETS)
def test_max_act_read_word(net):
    descs, _ = gp.make_descriptors(net)
    for d in descs:
        brute = max(((d["IC"] - 1) * d["IH"] + d["IH"] - 1) * d["IN_WPR"] + t + (1 if b < kx else 0)
                    for t in range(d["OW_TILES"]) for kx in range(d["KW"]) for b in range(8))
        m = gp.max_act_read_word(d)
        assert m == brute and m <= d["IN_END"] + 1 and m < gp.ACT_DEPTH
        print(net, d["name"], "max read word", m, "IN_END", d["IN_END"])
