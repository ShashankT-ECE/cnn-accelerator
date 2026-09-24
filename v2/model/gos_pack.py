"""V2 memory-image and descriptor packing: the host <-> RTL data contract.

This module is the single Python implementation of the formats in
``v2/docs/FORMATS.md`` (ACT, WGT, QPARAM, LOGIT, layer DESCRIPTOR). The golden
model, the cycle model, the vector generators and the PS driver all build
their memory images through it.

Public API (all shapes/dtypes explicit):

  Geometry
    act_wpr(W)                      -> ceil(W/8)  (ACT words per map row)
    act_depth(C, H, W)              -> C*H*ceil(W/8)  (ACT words used by a map)
    act_addr(c, y, x, H, W)         -> (bank, word)

  ACT (8 banks x 8 bit x 4096 words per buffer)
    pack_act(x[C,H,W] int8, depth=4096)   -> banks uint8 [8, depth] (unwritten bytes = 0)
    pack_act_words(x, depth=None)         -> PS view uint64 [depth] (default: used depth)
    act_banks_to_words(banks)             -> uint64 [depth]   (byte lane b = bank b)
    act_words_to_banks(words)             -> uint8 [8, depth]
    unpack_act(mem, C, H, W)              -> int8 [C,H,W]  (mem: banks [8,D] or words [D])

  Parameters (loaded only via net_config.NET_CONFIGS)
    load_params(net)                      -> {layer: {q_w[OC,IC,KH,KW] int8, q_b int32, m, s}}
    pack_wgt(net)                         -> (uint64 [total], WGT_BASE list)
    unpack_wgt(words, layer, base)        -> int8 [OC,IC,KH,KW]
    pack_qparam(net)                      -> (uint64 [2*channels], QP_BASE list)
    qparam_banks(words)                   -> (even uint64 [ch], odd uint64 [ch])
    unpack_qparam(words, base, OC)        -> (q_b int32, m uint64, s uint8)

  Descriptors (16 x 32-bit words per layer)
    make_descriptors(net)                 -> (list of field dicts, uint32 [N_LAYERS, 16])
    encode_descriptor(fields)             -> uint32 [16]   (asserts derived consistency)
    decode_descriptor(words)              -> field dict (raw + derived + flags)
    check_descriptor(words)               -> (ok, reasons)  model of the RTL config checker
    descriptors_json(net)                 -> JSON-able dict

  Reports / files
    memory_usage(net)                     -> dict
    max_act_read_word(desc)               -> highest ACT word the rotator read can address
    layer_table(net)                      -> markdown table (pasted into FORMATS.md)
    write_hex(path, values, width_bits)   -> $readmemh file (one word/line, MSB first)
    read_hex(path)                        -> list[int]

Regenerate the FORMATS.md tables:  ``cd v2/model && ../../.venv/bin/python gos_pack.py tables``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from net_config import NET_CONFIGS, NETS

# ---- memory geometry (ARCH_SPEC "Memory") ------------------------------------
ACT_BANKS = 8
ACT_DEPTH = 4096            # words per bank, per buffer (ACT0 / ACT1)
WGT_DEPTH = 16384           # 64-bit words
WGT_LANES = 8               # output channels per WGT word (= array columns)
QP_CHANNELS = 256           # QPARAM entries (output channels over all layers)
QP_WORDS = 2 * QP_CHANNELS  # 64-bit words in the combined PS view
N_LOGITS = 16
MAX_LAYERS = 8
DESC_WORDS = 16
MIN_K = 8

M_BITS = 32                 # requant multiplier width (D1 outcome: B = 32)
S_BITS = 6                  # requant shift width

# Flag bit positions in descriptor word 6.
FLAG_RELU_EN, FLAG_POOL_EN, FLAG_OUT_RAW, FLAG_IN_SEL = 0, 1, 2, 3

RAW_FIELDS = ("IC", "OC", "IH", "IW", "KH", "KW", "OH", "OW", "WGT_BASE", "QP_BASE")
FLAG_FIELDS = ("relu_en", "pool_en", "out_raw", "in_sel")
DERIVED_FIELDS = ("K", "IN_WPR", "IN_PLANE", "OC_TILES", "OW_TILES", "OUT_W", "OUT_H",
                  "OUT_WPR", "OUT_PLANE", "WGT_END", "IN_END", "OUT_END", "QP_END")
ALL_FIELDS = RAW_FIELDS + FLAG_FIELDS + DERIVED_FIELDS

# (word, lsb, width) for every numeric field. Flags live in word 6.
DESC_LAYOUT = {
    "IC": (0, 0, 16), "OC": (0, 16, 16),
    "IH": (1, 0, 16), "IW": (1, 16, 16),
    "KH": (2, 0, 8), "KW": (2, 8, 8),
    "OH": (3, 0, 16), "OW": (3, 16, 16),
    "WGT_BASE": (4, 0, 32),
    "QP_BASE": (5, 0, 32),
    "K": (7, 0, 32),
    "IN_WPR": (8, 0, 16), "IN_PLANE": (8, 16, 16),
    "OC_TILES": (9, 0, 16), "OW_TILES": (9, 16, 16),
    "OUT_W": (10, 0, 16), "OUT_H": (10, 16, 16),
    "OUT_WPR": (11, 0, 16), "OUT_PLANE": (11, 16, 16),
    "WGT_END": (12, 0, 32),
    "IN_END": (13, 0, 32),
    "OUT_END": (14, 0, 32),
    "QP_END": (15, 0, 32),
}
FLAG_BITS = {"relu_en": FLAG_RELU_EN, "pool_en": FLAG_POOL_EN,
             "out_raw": FLAG_OUT_RAW, "in_sel": FLAG_IN_SEL}

# Fields the RTL checker requires to be nonzero.
NONZERO_FIELDS = ("IC", "OC", "IH", "IW", "KH", "KW", "OH", "OW", "K", "IN_WPR", "IN_PLANE",
                  "OC_TILES", "OW_TILES", "OUT_W", "OUT_H", "OUT_WPR", "OUT_PLANE")


def ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


# ---- ACT ---------------------------------------------------------------------
def act_wpr(W: int) -> int:
    """ACT words per map row: ceil(W/8)."""
    return ceil_div(W, ACT_BANKS)


def act_depth(C: int, H: int, W: int) -> int:
    """ACT words occupied by a [C,H,W] map (from word 0): C*H*ceil(W/8)."""
    return int(C) * int(H) * act_wpr(W)


def act_addr(c: int, y: int, x: int, H: int, W: int) -> tuple[int, int]:
    """(bank, word) of element (c, y, x) of a map of height H, width W.

    bank = x mod 8; word = (c*H + y)*ceil(W/8) + (x >> 3).
    """
    assert 0 <= y < H and 0 <= x < W and c >= 0, (c, y, x, H, W)
    return x % ACT_BANKS, (c * H + y) * act_wpr(W) + (x >> 3)


def _check_int8_map(x) -> np.ndarray:
    x = np.asarray(x)
    assert x.ndim == 3, f"expected [C,H,W], got shape {x.shape}"
    if x.dtype != np.int8:
        assert np.all((x >= -128) & (x <= 127)), "values outside int8"
        x = x.astype(np.int8)
    return x


def pack_act(x, depth: int = ACT_DEPTH) -> np.ndarray:
    """Pack an int8 map [C,H,W] into one ACT buffer image: uint8 [8, depth].

    Bytes not covered by the map (x tail of each row, words >= act_depth) are 0.
    int8 values are stored as two's-complement bytes.
    """
    x = _check_int8_map(x)
    C, H, W = x.shape
    wpr = act_wpr(W)
    used = act_depth(C, H, W)
    assert used <= depth <= ACT_DEPTH, f"map needs {used} words, depth {depth}, max {ACT_DEPTH}"
    banks = np.zeros((ACT_BANKS, depth), dtype=np.uint8)
    xp = np.zeros((C, H, wpr * ACT_BANKS), dtype=np.uint8)
    xp[:, :, :W] = x.view(np.uint8)
    # xp[c, y, 8*j + b] -> bank b, word (c*H + y)*wpr + j
    banks[:, :used] = xp.reshape(C * H * wpr, ACT_BANKS).T
    return banks


def act_banks_to_words(banks) -> np.ndarray:
    """PS view: 64-bit word w, byte lane b (bits 8b+7:8b) = bank b at word w."""
    banks = np.asarray(banks, dtype=np.uint8)
    assert banks.ndim == 2 and banks.shape[0] == ACT_BANKS
    words = np.zeros(banks.shape[1], dtype=np.uint64)
    for b in range(ACT_BANKS):
        words |= banks[b].astype(np.uint64) << np.uint64(8 * b)
    return words


def act_words_to_banks(words) -> np.ndarray:
    words = np.asarray(words, dtype=np.uint64)
    assert words.ndim == 1
    return np.stack([((words >> np.uint64(8 * b)) & np.uint64(0xFF)).astype(np.uint8)
                     for b in range(ACT_BANKS)])


def pack_act_words(x, depth: int | None = None) -> np.ndarray:
    """PS-view image of a map: uint64 [depth]; depth defaults to the used depth."""
    x = _check_int8_map(x)
    d = act_depth(*x.shape) if depth is None else depth
    return act_banks_to_words(pack_act(x, d))


def unpack_act(mem, C: int, H: int, W: int) -> np.ndarray:
    """Read a [C,H,W] int8 map from an ACT image (banks uint8 [8,D] or words uint64 [D])."""
    mem = np.asarray(mem)
    banks = act_words_to_banks(mem) if mem.ndim == 1 else mem.astype(np.uint8)
    assert banks.shape[0] == ACT_BANKS
    wpr = act_wpr(W)
    used = act_depth(C, H, W)
    assert used <= banks.shape[1], f"map needs {used} words, image has {banks.shape[1]}"
    xp = banks[:, :used].T.reshape(C, H, wpr * ACT_BANKS)
    return np.ascontiguousarray(xp[:, :, :W]).view(np.int8)


# ---- parameters ----------------------------------------------------------------
def layers(net: str) -> tuple:
    return NET_CONFIGS[net]["layers"]


def load_params(net: str) -> dict:
    """Per layer: q_w int8 [OC,IC,KH,KW] (FC reshaped), q_b int32 [OC], m uint64, s uint8.

    The final layer gets m = s = 0 (APPROVED: QPARAM entries exist, unused: out_raw).
    """
    cfg = NET_CONFIGS[net]
    q = np.load(cfg["quant_params"])
    h = np.load(cfg["hw_requant"])
    assert int(h["B"]) == M_BITS, f"{net}: hw_requant B={int(h['B'])}, format assumes {M_BITS}"
    out = {}
    for L in cfg["layers"]:
        n = L["name"]
        w = q[f"{n}_q_w"]
        assert w.dtype == np.int8
        w = w.reshape(L["OC"], L["IC"], L["KH"], L["KW"])
        b = q[f"{n}_q_b"]
        assert b.dtype == np.int32 and b.shape == (L["OC"],)
        if L["final"]:
            assert f"{n}_m" not in h.files and f"{n}_s" not in h.files
            m = np.zeros(L["OC"], dtype=np.uint64)
            s = np.zeros(L["OC"], dtype=np.uint8)
        else:
            m = h[f"{n}_m"].astype(np.uint64)
            s = h[f"{n}_s"].astype(np.uint8)
            assert m.shape == (L["OC"],) and s.shape == (L["OC"],)
            # requant layers: s in [1, 63] (6-bit field; DECISIONS OC-2)
            assert int(s.min()) >= 1, f"{net}/{n}: s = 0 on a requant layer"
        assert int(m.max()) < 2 ** M_BITS, f"{net}/{n}: m >= 2^{M_BITS}"
        assert int(s.max()) < 2 ** S_BITS, f"{net}/{n}: s >= 2^{S_BITS}"
        out[n] = {"q_w": w, "q_b": b, "m": m, "s": s}
    return out


# ---- WGT -----------------------------------------------------------------------
def wgt_layer_words(L: dict) -> int:
    return ceil_div(L["OC"], WGT_LANES) * L["K"]


def wgt_bases(net: str) -> list[int]:
    bases, acc = [], 0
    for L in layers(net):
        bases.append(acc)
        acc += wgt_layer_words(L)
    assert acc <= WGT_DEPTH, f"{net}: WGT needs {acc} words > {WGT_DEPTH}"
    return bases


def _pack_wgt_layer(w: np.ndarray) -> np.ndarray:
    OC, IC, KH, KW = w.shape
    K = IC * KH * KW
    tiles = ceil_div(OC, WGT_LANES)
    wk = np.zeros((tiles * WGT_LANES, K), dtype=np.int8)     # OC tail rows stay 0
    wk[:OC] = w.reshape(OC, K)                               # k = (ic*KH + ky)*KW + kx
    lanes = wk.view(np.uint8).reshape(tiles, WGT_LANES, K).transpose(0, 2, 1)  # [tile, k, lane]
    words = np.zeros((tiles, K), dtype=np.uint64)
    for j in range(WGT_LANES):
        words |= lanes[:, :, j].astype(np.uint64) << np.uint64(8 * j)
    return words.reshape(tiles * K)                          # word = oc_tile*K + k


def pack_wgt(net: str) -> tuple[np.ndarray, list[int]]:
    """WGT image uint64 [total] and WGT_BASE per layer (layers packed from 0)."""
    p = load_params(net)
    bases = wgt_bases(net)
    parts = [_pack_wgt_layer(p[L["name"]]["q_w"]) for L in layers(net)]
    for L, base, part in zip(layers(net), bases, parts):
        assert part.size == wgt_layer_words(L)
    words = np.concatenate(parts)
    assert words.size <= WGT_DEPTH
    return words, bases


def unpack_wgt(words, L: dict, base: int) -> np.ndarray:
    """Inverse of the WGT packing for layer L: int8 [OC,IC,KH,KW] (drops OC tail lanes)."""
    words = np.asarray(words, dtype=np.uint64)
    K, OC = L["K"], L["OC"]
    tiles = ceil_div(OC, WGT_LANES)
    seg = words[base:base + tiles * K].reshape(tiles, K)
    lanes = np.stack([((seg >> np.uint64(8 * j)) & np.uint64(0xFF)).astype(np.uint8)
                      for j in range(WGT_LANES)], axis=1)     # [tile, lane, k]
    wk = lanes.reshape(tiles * WGT_LANES, K).view(np.int8)
    return wk[:OC].reshape(OC, L["IC"], L["KH"], L["KW"])


def wgt_tail_bytes(words, L: dict, base: int) -> np.ndarray:
    """int8 bytes of the OC-tail lanes of layer L (must all be 0)."""
    words = np.asarray(words, dtype=np.uint64)
    K, OC = L["K"], L["OC"]
    tiles = ceil_div(OC, WGT_LANES)
    seg = words[base:base + tiles * K].reshape(tiles, K)
    lanes = np.stack([((seg >> np.uint64(8 * j)) & np.uint64(0xFF)).astype(np.uint8)
                      for j in range(WGT_LANES)], axis=1)
    return lanes.reshape(tiles * WGT_LANES, K)[OC:]


# ---- QPARAM --------------------------------------------------------------------
def qp_bases(net: str) -> list[int]:
    bases, acc = [], 0
    for L in layers(net):
        bases.append(acc)
        acc += L["OC"]
    assert acc <= QP_CHANNELS, f"{net}: {acc} QPARAM channels > {QP_CHANNELS}"
    return bases


def pack_qparam(net: str) -> tuple[np.ndarray, list[int]]:
    """Combined QPARAM image uint64 [2*channels] and QP_BASE per layer.

    Channel i = QP_BASE[L] + oc:
      word 2i   = {m[31:0] (bits 63:32), q_bias[31:0] two's complement (bits 31:0)}
      word 2i+1 = {58'b0, s[5:0]}
    """
    p = load_params(net)
    bases = qp_bases(net)
    n = sum(L["OC"] for L in layers(net))
    words = np.zeros(2 * n, dtype=np.uint64)
    for L, base in zip(layers(net), bases):
        P = p[L["name"]]
        qb = P["q_b"].view(np.uint32).astype(np.uint64)
        idx = base + np.arange(L["OC"])
        words[2 * idx] = (P["m"].astype(np.uint64) << np.uint64(32)) | qb
        words[2 * idx + 1] = P["s"].astype(np.uint64) & np.uint64(0x3F)
    return words, bases


def qparam_banks(words) -> tuple[np.ndarray, np.ndarray]:
    """Two 64-bit BRAM images read in parallel: even[i] = word 2i, odd[i] = word 2i+1."""
    words = np.asarray(words, dtype=np.uint64)
    assert words.size % 2 == 0
    return words[0::2].copy(), words[1::2].copy()


def unpack_qparam(words, base: int, OC: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    words = np.asarray(words, dtype=np.uint64)
    ev, od = qparam_banks(words)
    e, o = ev[base:base + OC], od[base:base + OC]
    q_b = (e & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.int32)
    m = e >> np.uint64(32)
    assert np.all((o >> np.uint64(S_BITS)) == 0), "nonzero reserved bits in odd QPARAM word"
    s = (o & np.uint64(0x3F)).astype(np.uint8)
    return q_b, m, s


# ---- descriptors ---------------------------------------------------------------
def derive_fields(f: dict) -> dict:
    """Host-side derived descriptor fields from raw fields + pool flag."""
    IC, OC, IH, IW = f["IC"], f["OC"], f["IH"], f["IW"]
    KH, KW, OH, OW = f["KH"], f["KW"], f["OH"], f["OW"]
    pool = bool(f["pool_en"])
    d = {"K": IC * KH * KW, "IN_WPR": act_wpr(IW)}
    d["IN_PLANE"] = IH * d["IN_WPR"]
    d["OC_TILES"] = ceil_div(OC, WGT_LANES)
    d["OW_TILES"] = ceil_div(OW, ACT_BANKS)
    d["OUT_W"] = OW // 2 if pool else OW
    d["OUT_H"] = OH // 2 if pool else OH
    d["OUT_WPR"] = act_wpr(d["OUT_W"])
    d["OUT_PLANE"] = d["OUT_H"] * d["OUT_WPR"]
    d["WGT_END"] = f["WGT_BASE"] + d["OC_TILES"] * d["K"] - 1
    d["IN_END"] = IC * d["IN_PLANE"] - 1
    d["OUT_END"] = OC * d["OUT_PLANE"] - 1
    d["QP_END"] = f["QP_BASE"] + OC - 1
    return d


def encode_descriptor(fields: dict) -> np.ndarray:
    """Pack a full field dict into uint32 [16].

    Derived fields, if present, must equal derive_fields(); missing ones are filled.
    Every field must fit its bit width.
    """
    f = dict(fields)
    d = derive_fields(f)
    for k, v in d.items():
        if k in f:
            assert int(f[k]) == v, f"derived field {k}={f[k]} inconsistent (expected {v})"
        f[k] = v
    words = [0] * DESC_WORDS
    for k, (w, lsb, width) in DESC_LAYOUT.items():
        v = int(f[k])
        assert 0 <= v < (1 << width), f"field {k}={v} does not fit {width} bits"
        words[w] |= v << lsb
    flags = 0
    for k, bit in FLAG_BITS.items():
        v = int(f[k])
        assert v in (0, 1), f"flag {k}={v}"
        flags |= v << bit
    words[6] = flags
    return np.array(words, dtype=np.uint32)


def decode_descriptor(words) -> dict:
    """uint32 [16] -> dict of all fields (raw, flags, derived) plus 'flags_reserved'."""
    words = [int(w) for w in np.asarray(words, dtype=np.uint64)]
    assert len(words) == DESC_WORDS and all(0 <= w < (1 << 32) for w in words)
    f = {k: (words[w] >> lsb) & ((1 << width) - 1) for k, (w, lsb, width) in DESC_LAYOUT.items()}
    for k, bit in FLAG_BITS.items():
        f[k] = (words[6] >> bit) & 1
    f["flags_reserved"] = words[6] >> 4
    f["w2_reserved"] = words[2] >> 16
    return f


def check_descriptor(words) -> tuple[bool, list[str]]:
    """Python model of the RTL config checker.

    Uses only field extraction (wiring) and comparisons / single-bit tests: no
    multiplication, division or modulo, so it maps 1:1 to comparator logic.
    It does NOT verify that derived fields are consistent with raw fields
    (that needs multipliers); gos_pack asserts that on the host side.
    """
    f = decode_descriptor(words)
    r = []
    if f["pool_en"] and (f["OH"] & 1):
        r.append("pool_en with odd OH")
    if f["pool_en"] and (f["OW"] & 1):
        r.append("pool_en with odd OW")
    if f["K"] < MIN_K:
        r.append("K < 8")
    if f["WGT_END"] > WGT_DEPTH - 1:
        r.append("WGT_END > 16383")
    if f["IN_END"] > ACT_DEPTH - 1:
        r.append("IN_END > 4095")
    if f["OUT_END"] > ACT_DEPTH - 1:
        r.append("OUT_END > 4095")
    if f["QP_END"] > QP_CHANNELS - 1:
        r.append("QP_END > 255")
    for k in NONZERO_FIELDS:
        if f[k] == 0:
            r.append(f"{k} == 0")
    if f["OH"] > f["IH"]:
        r.append("OH > IH")
    if f["OW"] > f["IW"]:
        r.append("OW > IW")
    if f["out_raw"] and f["OC"] > N_LOGITS:
        r.append("out_raw with OC > 16")
    return (not r), r


def make_descriptors(net: str) -> tuple[list[dict], np.ndarray]:
    """Descriptors of every layer of `net`: (field dicts incl. name, uint32 [N_LAYERS, 16]).

    in_sel alternates 0,1,0,... (input image in ACT0); each layer reads ACT[in_sel]
    and writes ACT[!in_sel] from word 0.
    """
    Ls = layers(net)
    assert 1 <= len(Ls) <= MAX_LAYERS
    wb, qb = wgt_bases(net), qp_bases(net)
    descs, words = [], []
    for i, L in enumerate(Ls):
        f = {"IC": L["IC"], "OC": L["OC"], "IH": L["IH"], "IW": L["IW"], "KH": L["KH"],
             "KW": L["KW"], "OH": L["OH"], "OW": L["OW"], "WGT_BASE": wb[i], "QP_BASE": qb[i],
             "relu_en": int(L["relu"]), "pool_en": int(L["pool"]), "out_raw": int(L["final"]),
             "in_sel": i % 2}
        f.update(derive_fields(f))
        assert f["K"] == L["K"]
        assert f["out_raw"] == (i == len(Ls) - 1), "only the last layer is out_raw"
        if f["out_raw"]:
            assert f["OC"] <= N_LOGITS and not f["pool_en"] and not f["relu_en"]
        w = encode_descriptor(f)
        ok, why = check_descriptor(w)
        assert ok, f"{net}/{L['name']}: checker rejects real descriptor: {why}"
        descs.append({"name": L["name"], **f})
        words.append(w)
    # layer chaining: output of layer i is the input of layer i+1, in the other buffer
    for a, b in zip(descs, descs[1:]):
        assert (b["IC"], b["IH"], b["IW"]) == (a["OC"], a["OUT_H"], a["OUT_W"]), (a["name"], b["name"])
        assert b["in_sel"] == 1 - a["in_sel"]
    return descs, np.stack(words)


def max_act_read_word(d: dict) -> int:
    """Highest ACT word the conflict-free read (ARCH_SPEC Memory) can address for layer d:
    bank b reads rowbase + ox0/8 + (b < kx); rowbase of the last row of the last channel.
    Words above IN_END are read only for masked (x-tail) rows."""
    last_row = ((d["IC"] - 1) * d["IH"] + d["IH"] - 1) * d["IN_WPR"]
    extra = 1 if d["KW"] > 1 else 0          # some bank b < kx exists iff kx >= 1 occurs
    return last_row + (d["OW_TILES"] - 1) + extra


def descriptors_json(net: str) -> dict:
    descs, words = make_descriptors(net)
    return {"net": net, "n_layers": len(descs),
            "layers": [{**d, "words": [f"0x{int(x):08X}" for x in w]}
                       for d, w in zip(descs, words)]}


# ---- memory usage / tables -----------------------------------------------------
def stored_maps(net: str) -> list[dict]:
    """Every map stored in ACT: input image, each non-final output (as stored), and the
    unpooled output of each pooling layer (no-pool fallback)."""
    Ls = layers(net)
    L0 = Ls[0]
    maps = [{"what": "input image", "layer": L0["name"], "C": L0["IC"], "H": L0["IH"],
             "W": L0["IW"], "buffer": 0}]
    for i, L in enumerate(Ls):
        if L["final"]:
            continue
        H, W = (L["OH"] // 2, L["OW"] // 2) if L["pool"] else (L["OH"], L["OW"])
        buf = 1 - i % 2
        maps.append({"what": "output", "layer": L["name"], "C": L["OC"], "H": H, "W": W,
                     "buffer": buf})
        if L["pool"]:
            maps.append({"what": "unpooled output (fallback)", "layer": L["name"], "C": L["OC"],
                         "H": L["OH"], "W": L["OW"], "buffer": buf})
    for m in maps:
        m["depth"] = act_depth(m["C"], m["H"], m["W"])
        assert m["depth"] <= ACT_DEPTH, f"{net}: {m} exceeds ACT depth {ACT_DEPTH}"
    return maps


def memory_usage(net: str) -> dict:
    Ls = layers(net)
    maps = stored_maps(net)
    descs, _ = make_descriptors(net)
    in_depths = [d["IN_END"] + 1 for d in descs]
    wgt = sum(wgt_layer_words(L) for L in Ls)
    ch = sum(L["OC"] for L in Ls)
    assert wgt <= WGT_DEPTH and ch <= QP_CHANNELS and max(in_depths) <= ACT_DEPTH
    return {
        "net": net,
        "n_layers": len(Ls),
        "wgt_words": wgt,
        "wgt_depth": WGT_DEPTH,
        "qparam_channels": ch,
        "qparam_words": 2 * ch,
        "qparam_words_max": QP_WORDS,
        "act_depth": ACT_DEPTH,
        "act_max_depth": max(m["depth"] for m in maps),
        "act_max_depth_pooled": max(m["depth"] for m in maps if "fallback" not in m["what"]),
        "act_max_input_depth": max(in_depths),
        "act_maps": maps,
        "logits": Ls[-1]["OC"],
    }


def layer_table(net: str) -> str:
    descs, _ = make_descriptors(net)
    hdr = ("| layer | IC | OC | IH | IW | OH | OW | K | WGT_BASE | WGT words | QP_BASE "
           "| in ACT depth | out ACT depth | unpooled out depth | in_sel |")
    rows = [hdr, "|" + "---|" * 15]
    for d in descs:
        if d["out_raw"]:
            out, unp = f"LOGIT[0..{d['OC'] - 1}]", "-"
        else:
            out = str(d["OUT_END"] + 1)
            unp = str(act_depth(d["OC"], d["OH"], d["OW"])) if d["pool_en"] else "-"
        rows.append(f"| {d['name']} | {d['IC']} | {d['OC']} | {d['IH']} | {d['IW']} | {d['OH']} "
                    f"| {d['OW']} | {d['K']} | {d['WGT_BASE']} | {d['OC_TILES'] * d['K']} "
                    f"| {d['QP_BASE']} | {d['IN_END'] + 1} | {out} | {unp} | {d['in_sel']} |")
    u = memory_usage(net)
    rows.append("")
    rows.append(f"Totals ({net}): WGT {u['wgt_words']} / {WGT_DEPTH} words; QPARAM "
                f"{u['qparam_channels']} / {QP_CHANNELS} channels = {u['qparam_words']} / "
                f"{QP_WORDS} words; max ACT depth {u['act_max_depth']} / {ACT_DEPTH} "
                f"(pooled path {u['act_max_depth_pooled']}); logits {u['logits']} / {N_LOGITS}.")
    return "\n".join(rows)


def descriptor_table(net: str) -> str:
    descs, words = make_descriptors(net)
    rows = ["| layer | " + " | ".join(f"w{i}" for i in range(DESC_WORDS)) + " |",
            "|" + "---|" * (DESC_WORDS + 1)]
    for d, w in zip(descs, words):
        rows.append(f"| {d['name']} | " + " | ".join(f"{int(x):08X}" for x in w) + " |")
    return "\n".join(rows)


# ---- hex files ($readmemh) -----------------------------------------------------
def hex_lines(values, width_bits: int) -> list[str]:
    assert width_bits % 4 == 0
    nd = width_bits // 4
    out = []
    for v in np.asarray(values).ravel():
        v = int(v)
        assert 0 <= v < (1 << width_bits), f"{v} does not fit {width_bits} bits unsigned"
        out.append(f"{v:0{nd}x}")
    return out


def write_hex(path, values, width_bits: int) -> Path:
    """$readmemh file: one word per line, hex, MSB first, fixed width (64->16, 32->8, 8->2 digits).

    Values must be unsigned (pass int8 data as .view(np.uint8)).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(hex_lines(values, width_bits)) + "\n")
    return path


def write_hex64(path, values) -> Path:
    return write_hex(path, values, 64)


def write_hex32(path, values) -> Path:
    return write_hex(path, values, 32)


def write_hex8(path, values) -> Path:
    return write_hex(path, values, 8)


def read_hex(path) -> list[int]:
    return [int(t, 16) for t in Path(path).read_text().split() if not t.startswith("//")]


# ---- CLI -----------------------------------------------------------------------
def _main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "tables"
    if cmd == "tables":
        for net in NETS:
            print(f"### {net}\n\n{layer_table(net)}\n\nDescriptor words (hex, w0..w15):\n\n"
                  f"{descriptor_table(net)}\n")
    elif cmd == "json":
        print(json.dumps({net: descriptors_json(net) for net in (argv[2:] or NETS)}, indent=1))
    elif cmd == "usage":
        for net in NETS:
            u = memory_usage(net)
            maps = u.pop("act_maps")
            print(json.dumps(u))
            for m in maps:
                print("   ", m)
    else:
        print("usage: gos_pack.py [tables|json [net...]|usage]")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
