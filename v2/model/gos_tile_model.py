"""V2 step 2.2 part B: memory-level tile model of the gos_ core.

Executes a layer the way the hardware does, reading ONLY the packed memories
(FORMATS.md) and driven ONLY by the controller streams of ``gos_addr_stream``:

  per issue  : 8 ACT bank bytes at rd_addr[b] (12-bit port, & 0xFFF) of ACT[in_sel]
               -> rotator (bank b -> row (b - kx) mod 8, i.e. row r <- bank (r + kx) mod 8)
               -> a[r] int8;  WGT word at wgt_addr -> w[j] int8 (byte lane j)
               -> acc[r][j] = first ? a[r]*w[j] : acc[r][j] + a[r]*w[j]
  on last    : acc -> shadow (8x8)
  per drain  : column j of the shadow; QPARAM read at qp_idx (8-bit port, & 0xFF) from the
               even/odd banks QP_E/QP_O -> (q_bias, m, s); v = acc + q_bias;
               out_raw: LOGIT[logit_idx] = v[row 0] (int32);
               else 8 requant lanes q = clip(rne_shift(v*m, s)) (requant_check arithmetic),
               ReLU if relu_en;
               pool: dy = 0 column -> 8x8 pool buffer [r][j]; dy = 1 column -> vertical
               max with the stored column j, then horizontal pair max (rows 2i, 2i+1) ->
               4 bytes, replicated onto both bank halves (data[b] = h[b & 3]); the byte
               enable from the stream selects the half and masks the x tail;
               non-pool: data[b] = q[row b];
               write ACT[buf][b][word] = data[b] for every b with be bit b set.

Two executors of the same streams:
  * ``mode="fast"``: the k-stream of each tile is gathered with numpy and reduced with an
    integer einsum; segments are taken from the stream's first/last flags (asserted to be
    contiguous, one per tile). The drain is vectorized over the drain records.
  * ``mode="loop"``: scalar, one issue and one drain step at a time with an explicit
    accumulator, shadow and pool buffer (Python ints; requant via rne_shift_int).
Both are tested bit-identical to gos_golden and to each other.

API:
  Mem                                   memory state (ACT0/ACT1, WGT, QP_E/QP_O, LOGIT)
  net_mem(net, x_int8)                  -> (Mem, descriptor words uint32 [N,16])
  pack_layer_qparam(q_b, m, s)          -> combined QPARAM words for one layer (FORMATS 3)
  place_layer_params(mem, fields, q_w, q_b, m, s)   arbitrary-layer WGT/QPARAM images
  run_layer_mem(mem, desc, mode)        -> info dict (counts, writes, max |acc|)
  run_net_mem(net, x_int8, mode)        -> {"logits", "outputs", "act_images", "mem", ...}
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import gos_pack as gp
import gos_addr_stream as ga
from requant_check import rne_shift, rne_shift_int

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
QP_ADDR_MASK = gp.QP_CHANNELS - 1          # 8-bit QPARAM port
TWO62 = 1 << 62
_LANE_SHIFT = np.arange(0, 64, 8, dtype=np.uint64)
_ROWS = np.arange(ga.LANES)
CHUNK_ISSUES = 1 << 18                     # fast mode: issues gathered per chunk


# --------------------------------------------------------------------------- #
# Memories
# --------------------------------------------------------------------------- #
@dataclass
class Mem:
    act: np.ndarray = field(default_factory=lambda: np.zeros((2, gp.ACT_BANKS, gp.ACT_DEPTH),
                                                             dtype=np.uint8))
    wgt: np.ndarray = field(default_factory=lambda: np.zeros(gp.WGT_DEPTH, dtype=np.uint64))
    qp_e: np.ndarray = field(default_factory=lambda: np.zeros(gp.QP_CHANNELS, dtype=np.uint64))
    qp_o: np.ndarray = field(default_factory=lambda: np.zeros(gp.QP_CHANNELS, dtype=np.uint64))
    logit: np.ndarray = field(default_factory=lambda: np.zeros(gp.N_LOGITS, dtype=np.int32))

    def copy(self) -> "Mem":
        return Mem(self.act.copy(), self.wgt.copy(), self.qp_e.copy(), self.qp_o.copy(),
                   self.logit.copy())


def net_mem(net: str, x_int8) -> tuple[Mem, np.ndarray]:
    """Memory images of a net (gos_pack) with the int8 input image [C,H,W] in ACT0."""
    mem = Mem()
    mem.act[0] = gp.pack_act(x_int8)
    w, _ = gp.pack_wgt(net)
    mem.wgt[:w.size] = w
    q, _ = gp.pack_qparam(net)
    e, o = gp.qparam_banks(q)
    mem.qp_e[:e.size] = e
    mem.qp_o[:o.size] = o
    _, words = gp.make_descriptors(net)
    return mem, words


def pack_layer_qparam(q_b, m, s) -> np.ndarray:
    """Combined QPARAM words [2*OC] of one layer (FORMATS.md section 3):
    word 2i = {m[31:0], q_bias[31:0]}, word 2i+1 = {0, s[5:0]}. gos_pack only packs whole
    nets (pack_qparam(net)); this is the same layout for an arbitrary layer (a test
    round-trips it through gos_pack.unpack_qparam)."""
    q_b = np.asarray(q_b, dtype=np.int32)
    m = np.asarray([int(x) for x in m], dtype=np.uint64)
    s = np.asarray([int(x) for x in s], dtype=np.uint64)
    assert q_b.shape == m.shape == s.shape
    assert np.all(m < np.uint64(1 << gp.M_BITS)) and np.all(s < np.uint64(1 << gp.S_BITS))
    words = np.zeros(2 * q_b.size, dtype=np.uint64)
    words[0::2] = (m << np.uint64(32)) | q_b.view(np.uint32).astype(np.uint64)
    words[1::2] = s & np.uint64(0x3F)
    return words


def place_layer_params(mem: Mem, fields: dict, q_w, q_b, m, s) -> None:
    """Write one arbitrary layer's WGT (gos_pack WGT packing) and QPARAM entries at the
    descriptor's WGT_BASE / QP_BASE."""
    q_w = np.asarray(q_w, dtype=np.int8)
    assert q_w.shape == (fields["OC"], fields["IC"], fields["KH"], fields["KW"])
    w = gp._pack_wgt_layer(q_w)                       # reuse the gos_pack WGT format
    base = int(fields["WGT_BASE"])
    assert base + w.size <= gp.WGT_DEPTH
    mem.wgt[base:base + w.size] = w
    e, o = gp.qparam_banks(pack_layer_qparam(q_b, m, s))
    qb = int(fields["QP_BASE"])
    assert qb + e.size <= gp.QP_CHANNELS
    mem.qp_e[qb:qb + e.size] = e
    mem.qp_o[qb:qb + o.size] = o


# --------------------------------------------------------------------------- #
# Datapath pieces
# --------------------------------------------------------------------------- #
def _wgt_lanes(words: np.ndarray) -> np.ndarray:
    """uint64 WGT words [N] -> int8 lanes [N, 8] (lane j = bits 8j+7:8j)."""
    return ((words[:, None] >> _LANE_SHIFT[None, :]) & np.uint64(0xFF)).astype(np.uint8).view(np.int8)


def _qparam_read(mem: Mem, qp_idx: np.ndarray):
    """QPARAM lane read at channel index (8-bit port): (q_bias int64, m int64, s int64)."""
    a = qp_idx.astype(np.int64) & QP_ADDR_MASK
    e, o = mem.qp_e[a], mem.qp_o[a]
    q_b = (e & np.uint64(0xFFFFFFFF)).astype(np.uint32).view(np.int32).astype(np.int64)
    m = (e >> np.uint64(32)).astype(np.int64)
    s = (o & np.uint64(0x3F)).astype(np.int64)
    return q_b, m, s


def _requant_lanes(v: np.ndarray, m: np.ndarray, s: np.ndarray, relu: bool) -> np.ndarray:
    """8 requant lanes per drain step: v int64 [N,8], per-record m, s [N]."""
    if v.size == 0:
        return np.zeros(v.shape, dtype=np.int8)
    assert int(np.abs(v).max()) * int(m.max()) < TWO62, "p = v*m exceeds the int64 path"
    assert int(s.min()) >= 1 and int(s.max()) <= 62, "s outside the rne_shift range"
    q = rne_shift(v * m[:, None], s[:, None])
    q = np.clip(q, -128, 127).astype(np.int8)
    if relu:
        q = np.maximum(q, np.int8(0))
    return q


def _fast_mac(mem: Mem, st: ga.AddrStream) -> np.ndarray:
    """Accumulate every tile; returns the shadow registers int64 [T, 8(r), 8(j)]."""
    iss = st.issues
    act = mem.act[st.desc["in_sel"]]
    firsts = np.flatnonzero(iss["first"])
    lasts = np.flatnonzero(iss["last"])
    n = iss.size
    assert firsts.size == lasts.size == st.n_tiles and firsts[0] == 0
    assert np.array_equal(lasts[:-1] + 1, firsts[1:]) and lasts[-1] == n - 1, \
        "first/last do not delimit contiguous tiles"
    seg = lasts - firsts + 1
    L = int(seg[0])
    assert np.all(seg == L), "tiles of unequal length"
    T = firsts.size
    shadow = np.empty((T, ga.LANES, ga.LANES), dtype=np.int64)
    tiles_per_chunk = max(1, CHUNK_ISSUES // L)
    for t0 in range(0, T, tiles_per_chunk):
        t1 = min(T, t0 + tiles_per_chunk)
        c = iss[firsts[t0]:lasts[t1 - 1] + 1]
        addr = c["rd_addr"].astype(np.int64) & ga.ACT_ADDR_MASK             # [n, 8] per bank
        banks = act[_ROWS[None, :], addr]                                    # bank b byte
        rot = (_ROWS[None, :] + c["rot"].astype(np.int64)[:, None]) & 7      # row r <- bank r+kx
        a = np.take_along_axis(banks, rot, axis=1).view(np.int8).astype(np.int64)
        w = _wgt_lanes(mem.wgt[c["wgt_addr"].astype(np.int64)]).astype(np.int64)
        nt = t1 - t0
        shadow[t0:t1] = np.einsum("tkr,tkj->trj", a.reshape(nt, L, ga.LANES),
                                  w.reshape(nt, L, ga.LANES))
    return shadow


def _fast_drain(mem: Mem, st: ga.AddrStream, shadow: np.ndarray) -> dict:
    d, dr = st.desc, st.drain
    t = dr["tile"].astype(np.int64)
    j = dr["j"].astype(np.int64)
    col = shadow[t, :, j]                                   # [ND, 8 rows]
    q_b, m, s = _qparam_read(mem, dr["qp_idx"])
    v = col + q_b[:, None]
    we = dr["we"]
    if d["out_raw"]:
        sel = np.flatnonzero(we)
        vv = v[sel, 0]
        assert vv.size == 0 or (vv.min() >= INT32_MIN and vv.max() <= INT32_MAX)
        idx = dr["logit_idx"][sel].astype(np.int64)
        assert np.unique(idx).size == idx.size and (idx.size == 0 or idx.max() < gp.N_LOGITS)
        mem.logit[idx] = vv.astype(np.int32)
        return {"act_bytes_written": 0, "logits_written": int(sel.size)}
    valid = np.flatnonzero(dr["ch_valid"])
    q = np.zeros(v.shape, dtype=np.int8)
    q[valid] = _requant_lanes(v[valid], m[valid], s[valid], bool(d["relu_en"]))
    if d["pool_en"]:
        store = dr["kind"] == ga.KIND_POOL_STORE
        # pool buffer column j holds the most recent store of column j
        src = np.full(dr.size, -1, dtype=np.int64)
        for jj in range(ga.LANES):
            pos = np.flatnonzero(j == jj)
            last_store = np.maximum.accumulate(np.where(store[pos], pos, -1))
            src[pos] = last_store
        wr = np.flatnonzero(dr["kind"] == ga.KIND_ACT)
        assert np.all(src[wr] >= 0), "pool combine without a stored dy=0 column"
        vert = np.maximum(q[src[wr]], q[wr])                             # vertical max
        h = np.maximum(vert[:, 0::2], vert[:, 1::2])                     # horizontal pair max
        data = np.zeros(v.shape, dtype=np.int8)
        data[wr] = h[:, _ROWS & 3]                                       # both halves
    else:
        data = q                                                         # bank b = row b
    return _act_write(mem, dr, data)


def _act_write(mem: Mem, dr: np.ndarray, data: np.ndarray) -> dict:
    """Byte-enabled 8-bank write of every drain record with we."""
    sel = np.flatnonzero(dr["we"] & (dr["kind"] == ga.KIND_ACT))
    be = dr["be"][sel].astype(np.int64)
    bits = ((be[:, None] >> _ROWS[None, :]) & 1).astype(bool)           # [n, 8 banks]
    rec, bank = np.nonzero(bits)
    word = dr["word"][sel][rec].astype(np.int64)
    assert word.size == 0 or word.max() < gp.ACT_DEPTH, "ACT write beyond depth"
    key = bank * gp.ACT_DEPTH + word
    assert np.unique(key).size == key.size, "an ACT byte is written twice in one layer"
    buf = dr["buf"][sel][rec].astype(np.int64)
    mem.act[buf, bank, word] = data[sel][rec, bank].view(np.uint8)
    return {"act_bytes_written": int(key.size), "logits_written": 0}


# --------------------------------------------------------------------------- #
# Scalar executor (one issue / one drain step at a time)
# --------------------------------------------------------------------------- #
def _loop_run(mem: Mem, st: ga.AddrStream) -> dict:
    d = st.desc
    act = mem.act[d["in_sel"]]
    relu = bool(d["relu_en"])
    acc = [[0] * 8 for _ in range(8)]
    shadows = []
    max_acc = 0
    for rec in st.issues:
        rd = [int(x) & ga.ACT_ADDR_MASK for x in rec["rd_addr"]]
        bank_bytes = [int(act[b, rd[b]]) for b in range(8)]
        kx = int(rec["rot"])
        a = [bank_bytes[(r + kx) & 7] for r in range(8)]                 # rotator
        a = [x - 256 if x >= 128 else x for x in a]
        word = int(mem.wgt[int(rec["wgt_addr"])])
        w = [(word >> (8 * jj)) & 0xFF for jj in range(8)]
        w = [x - 256 if x >= 128 else x for x in w]
        first = bool(rec["first"])
        for r in range(8):
            for jj in range(8):
                acc[r][jj] = a[r] * w[jj] if first else acc[r][jj] + a[r] * w[jj]
        if rec["last"]:
            shadows.append([row[:] for row in acc])
            max_acc = max(max_acc, max(abs(x) for row in acc for x in row))
    assert len(shadows) == st.n_tiles
    pool_buf = [[0] * 8 for _ in range(8)]                               # [r][j]
    written = set()
    logits = 0
    for rec in st.drain:
        sh = shadows[int(rec["tile"])]
        jj = int(rec["j"])
        i = int(rec["qp_idx"]) & QP_ADDR_MASK
        e, o = int(mem.qp_e[i]), int(mem.qp_o[i])
        qb = e & 0xFFFFFFFF
        qb = qb - (1 << 32) if qb >= (1 << 31) else qb
        m, s = e >> 32, o & 0x3F
        v = [sh[r][jj] + qb for r in range(8)]
        kind = int(rec["kind"])
        if kind == ga.KIND_LOGIT:
            if rec["we"]:
                assert INT32_MIN <= v[0] <= INT32_MAX
                mem.logit[int(rec["logit_idx"])] = v[0]
                logits += 1
            continue
        if not rec["ch_valid"]:
            assert not rec["we"]
            continue
        q = [min(127, max(-128, rne_shift_int(x * m, s))) for x in v]
        if relu:
            q = [max(0, x) for x in q]
        if kind == ga.KIND_POOL_STORE:
            for r in range(8):
                pool_buf[r][jj] = q[r]
            continue
        if d["pool_en"]:
            vert = [max(pool_buf[r][jj], q[r]) for r in range(8)]
            h = [max(vert[2 * i2], vert[2 * i2 + 1]) for i2 in range(4)]
            data = [h[b & 3] for b in range(8)]
        else:
            data = q
        be = int(rec["be"])
        for b in range(8):
            if (be >> b) & 1:
                key = (b, int(rec["word"]))
                assert key not in written
                written.add(key)
                mem.act[int(rec["buf"]), b, int(rec["word"])] = data[b] & 0xFF
    return {"act_bytes_written": len(written), "logits_written": logits, "max_abs_acc": max_acc}


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
def run_layer_mem(mem: Mem, desc, mode: str = "fast") -> dict:
    """Run one layer on `mem` in place. `desc`: 16 descriptor words (checked by the config
    checker model) or a decoded descriptor dict. Returns counts."""
    st = ga.addr_stream(desc)
    if mode == "fast":
        shadow = _fast_mac(mem, st)
        max_acc = int(np.abs(shadow).max())
        assert max_acc <= INT32_MAX, "accumulator exceeds INT32"
        info = _fast_drain(mem, st, shadow)
        info["max_abs_acc"] = max_acc
    elif mode == "loop":
        info = _loop_run(mem, st)
        assert info["max_abs_acc"] <= INT32_MAX
    else:
        raise ValueError(mode)
    info.update(ga.stream_counts(st))
    return info


def run_net_mem(net: str, x_int8, mode: str = "fast") -> dict:
    """Place x (int8 [C,H,W]) in ACT0, run every layer from its descriptor words, and
    return LOGIT[0..OC-1], each layer's stored output (unpacked from ACT[!in_sel]; final
    layer: LOGIT as int32 [OC,1,1]) and a copy of each layer's output ACT image."""
    x_int8 = np.asarray(x_int8)
    assert x_int8.dtype == np.int8 and x_int8.ndim == 3
    mem, words = net_mem(net, x_int8)
    names = [L["name"] for L in gp.layers(net)]
    outputs, images, infos = {}, {}, {}
    for name, w in zip(names, words):
        d = gp.decode_descriptor(w)
        infos[name] = run_layer_mem(mem, w, mode)
        if d["out_raw"]:
            outputs[name] = mem.logit[:d["OC"]].copy().reshape(d["OC"], 1, 1)
        else:
            buf = 1 - d["in_sel"]
            images[name] = mem.act[buf].copy()
            outputs[name] = gp.unpack_act(mem.act[buf], d["OC"], d["OUT_H"], d["OUT_W"])
    OC = gp.decode_descriptor(words[-1])["OC"]
    return {"logits": mem.logit[:OC].copy(), "outputs": outputs, "act_images": images,
            "info": infos, "mem": mem}
