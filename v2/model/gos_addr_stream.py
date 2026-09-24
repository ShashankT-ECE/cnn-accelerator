"""V2 step 2.2 part E: controller address / issue stream (counter-only).

Model of the gos_ controller's address generation (ARCH_SPEC "Controller",
"Tiling / loop nest", "Memory"; FORMATS.md). Input is a DECODED descriptor
(``gos_pack.decode_descriptor``); only descriptor fields are used (raw fields,
flags and the host-derived fields K, IN_WPR, IN_PLANE, OC_TILES, OW_TILES,
OUT_W, OUT_H, OUT_WPR, OUT_PLANE). No multiplication, division or modulo is
used to form any address: every address is a sum of counters that are advanced
by incremental adds (a test inspects the source with ``ast``). Loop bounds come
straight from descriptor fields. Bit-slicing (``>> 1``, ``& 1``, ``<< 2``) of a
counter is wiring, and ``b < kx`` / ``rem >= 8`` are comparators.

Loop nest (ARCH_SPEC), one ISSUE per cycle:

    for oc_tile in OC_TILES:
      for oy                 (pool: for oy pair p, oy = 2p + dy)
        for ox_tile in OW_TILES          (ox0 = 8*ox_tile)
          for dy in {0,1} if pool else {0}:        -> one TILE
            for ic: for ky: for kx:                -> one issue per k (kx innermost)

Issue record (``ISSUE_DTYPE``), per k of a tile:
    tile, oc_tile, oy, ox_tile, dy, k, ic, ky, kx, first (k == 0), last (k == K-1),
    rd_addr[b] (b = 0..7) = rowbase + ox0/8 + (b < kx), rowbase = (ic*IH + oy + ky)*IN_WPR,
    rot = kx (rotator: bank b -> row (b - kx) mod 8),
    wgt_addr = WGT_BASE + oc_tile*K + k,
    row_mask (bit r set iff ox0 + r < OW).
rd_addr values are the full counter values; the ACT port is 12 bits, so the
memory model applies ``& 0xFFF`` (FORMATS.md section 1: IN_END = 4095 wraps; the
wrapped reads feed only masked rows).

Drain record (``DRAIN_DTYPE``): 8 per tile, one per drain column step j = 0..7
(output channel ch = oc_tile*8 + j), in the order the shadow drains them. Every
drain step of the hardware is represented (T*8 records), including those that
write nothing. Fields: tile, oc_tile, oy, out_row, ox_tile, dy, j, ch, ch_valid
(ch < OC), qp_idx = QP_BASE + oc_tile*8 + j (QPARAM read of the requant lane),
row_mask, kind, we, buf, word, be, logit_idx.

  kind KIND_ACT (non-pool layer, or pool dy = 1): write to ACT[buf], buf = !in_sel,
      at `word` with 8-bit bank byte-enable `be`; `we` = (be != 0).
      Non-pool: out_row = oy, word = (ch*OUT_H + oy)*OUT_WPR + ox_tile,
                bank b holds pixel ox0 + b (row r = b), be = row_mask.
      Pool (dy = 1): out_row = p, pooled px = ox0/2 + i (i = 0..3):
                bank = px mod 8 = 4*(ox_tile & 1) + i, word = (ch*OUT_H + p)*OUT_WPR + (ox_tile >> 1),
                be bit (4*(ox_tile & 1) + i) set iff px < OUT_W (i.e. ox0 + 2i < OW).
  kind KIND_POOL_STORE (pool, dy = 0): the requantized column goes to the 8x8
      pool buffer (row r, column j); no ACT write: we = 0, be = 0.
  kind KIND_LOGIT (out_raw): raw int32 v of row 0 goes to LOGIT[logit_idx],
      logit_idx = ch; we = ch_valid; be = 0, word = 0 (unused).

Masked lanes (precise definition):
  * OC tail: ch >= OC  -> ch_valid = 0, be = 0, we = 0 (the drain step still
    occupies its cycle; nothing is written; its QPARAM read index may point past
    QP_END and its data is discarded).
  * x tail: rows r with ox0 + r >= OW have their be bit cleared (non-pool); for
    pool the pooled byte i is enabled iff ox0 + 2i < OW (OW is even under pool,
    so this equals ox0 + 2i + 1 < OW). Masked rows are computed from whatever
    the banks return and discarded.

out_raw scope: LOGIT[oc] is only defined for a 1x1 output (the nets' FC
layers). A descriptor with out_raw and (OH, OW) != (1, 1), or out_raw with pool
or relu, is rejected here (ValueError) — see the step 2.2 report.

The stream is data independent; write DATA is produced by ``gos_tile_model``.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

import gos_pack as gp

LANES = 8                     # array rows (x positions) == columns (output channels)
ACT_ADDR_MASK = gp.ACT_DEPTH - 1       # 12-bit ACT port
KIND_ACT, KIND_POOL_STORE, KIND_LOGIT = 0, 1, 2

ISSUE_DTYPE = np.dtype([
    ("tile", np.uint32), ("oc_tile", np.uint16), ("oy", np.uint16), ("ox_tile", np.uint16),
    ("dy", np.uint8), ("k", np.uint32), ("ic", np.uint16), ("ky", np.uint8), ("kx", np.uint8),
    ("first", np.bool_), ("last", np.bool_), ("rd_addr", np.uint32, (LANES,)),
    ("rot", np.uint8), ("wgt_addr", np.uint32), ("row_mask", np.uint8),
])
DRAIN_DTYPE = np.dtype([
    ("tile", np.uint32), ("oc_tile", np.uint16), ("oy", np.uint16), ("out_row", np.uint16),
    ("ox_tile", np.uint16), ("dy", np.uint8), ("j", np.uint8), ("ch", np.uint32),
    ("ch_valid", np.bool_), ("qp_idx", np.uint32), ("row_mask", np.uint8), ("kind", np.uint8),
    ("we", np.bool_), ("buf", np.uint8), ("word", np.uint32), ("be", np.uint8),
    ("logit_idx", np.uint32),
])

# Fields the stream is allowed to read from a descriptor.
USED_FIELDS = ("IC", "OC", "IH", "IW", "KH", "KW", "OH", "OW", "K", "IN_WPR", "IN_PLANE",
               "OC_TILES", "OW_TILES", "OUT_W", "OUT_H", "OUT_WPR", "OUT_PLANE", "WGT_BASE",
               "QP_BASE", "relu_en", "pool_en", "out_raw", "in_sel")


# --------------------------------------------------------------------------- #
# Descriptor handling (not the address path)
# --------------------------------------------------------------------------- #
def as_desc(desc) -> dict:
    """Accept a decoded descriptor dict or the 16 descriptor words; return the decoded
    dict restricted to USED_FIELDS. Refuses descriptors the RTL config checker rejects
    and the out_raw cases outside the spec'd LOGIT contract."""
    if not isinstance(desc, dict):
        words = np.asarray(desc, dtype=np.uint64)
        ok, why = gp.check_descriptor(words)
        if not ok:
            raise ValueError(f"config checker rejects descriptor: {why}")
        desc = gp.decode_descriptor(words)
    d = {k: int(desc[k]) for k in USED_FIELDS}
    if d["out_raw"] and (d["OH"] != 1 or d["OW"] != 1 or d["pool_en"] or d["relu_en"]):
        raise ValueError("out_raw is defined only for a 1x1 output without pool/relu "
                         f"(OH={d['OH']}, OW={d['OW']}, pool={d['pool_en']}, relu={d['relu_en']})")
    if d["pool_en"] and ((d["OH"] & 1) or (d["OW"] & 1)):
        raise ValueError("pool_en with odd OH/OW")
    return d


def desc_key(d: dict) -> tuple:
    return tuple(int(d[k]) for k in USED_FIELDS)


# --------------------------------------------------------------------------- #
# Address path: counters and incremental adds only (checked by an AST test)
# --------------------------------------------------------------------------- #
def _row_mask(rem):
    """Row-valid mask from rem = OW - ox0 (a down-counter): bit r set iff r < rem."""
    if rem >= LANES:
        return 0xFF
    return (1 << rem) - 1


def _pool_be(rem, half):
    """Pooled byte enables: byte i (i = 0..3) valid iff 2i < rem; placed at banks
    4*half .. 4*half + 3 (half = ox_tile & 1)."""
    vp = rem >> 1
    pm = 0xF if vp >= 4 else (1 << vp) - 1
    return pm << (half << 2)


def _k_program(d: dict) -> dict:
    """The per-tile k stream (identical for every tile), from the ic/ky/kx counters.

    rel_rd[k][b] = ic*IN_PLANE + ky*IN_WPR + (b < kx) built as ic_base (+= IN_PLANE per
    ic) + ky_off (+= IN_WPR per ky) + comparator; k_rel = k (+= 1 per issue).
    """
    ic_l, ky_l, kx_l, k_l, first_l, last_l, rd_l = [], [], [], [], [], [], []
    ic_last, ky_last, kx_last = d["IC"] - 1, d["KH"] - 1, d["KW"] - 1
    ic_base = 0
    k = 0
    for ic in range(d["IC"]):
        ky_off = ic_base
        for ky in range(d["KH"]):
            for kx in range(d["KW"]):
                ic_l.append(ic)
                ky_l.append(ky)
                kx_l.append(kx)
                k_l.append(k)
                first_l.append(k == 0)
                last_l.append(ic == ic_last and ky == ky_last and kx == kx_last)
                rd_l.append([ky_off + (1 if b < kx else 0) for b in range(LANES)])
                k += 1
            ky_off += d["IN_WPR"]
        ic_base += d["IN_PLANE"]
    return {"ic": np.array(ic_l, dtype=np.int64), "ky": np.array(ky_l, dtype=np.int64),
            "kx": np.array(kx_l, dtype=np.int64), "k": np.array(k_l, dtype=np.int64),
            "first": np.array(first_l, dtype=bool), "last": np.array(last_l, dtype=bool),
            "rd": np.array(rd_l, dtype=np.int64).reshape(-1, LANES)}


def _tile_walk(d: dict) -> tuple[dict, dict]:
    """Walk the tile loops with counters. Returns (per-tile base registers, drain records
    as column lists). Tile-level counters:
      wgt_tile   = WGT_BASE + oc_tile*K     (+= K per oc_tile)
      in_row     = oy*IN_WPR                (+= IN_WPR per oy; pool: += IN_WPR twice per pair)
      dy_off     = dy*IN_WPR                (+= IN_WPR per dy)
      ox_word    = ox0/8 = ox_tile          (+= 1 per ox_tile)
      rem        = OW - ox0                 (-= 8 per ox_tile)
      ch_tile    = oc_tile*8                (+= 8 per oc_tile)
      qp_tile    = QP_BASE + oc_tile*8      (+= 8 per oc_tile)
      out_ch     = ch*OUT_PLANE             (+= OUT_PLANE per drain step j)
      out_row_off= out_row*OUT_WPR          (+= OUT_WPR per output row / pooled row)
    """
    pool = d["pool_en"] == 1
    out_raw = d["out_raw"] == 1
    n_rows = d["OUT_H"] if pool else d["OH"]          # oy pairs (pool) or oy
    dys = (0, 1) if pool else (0,)
    buf = 1 - d["in_sel"]
    OC, OUT_PLANE, OUT_WPR, IN_WPR = d["OC"], d["OUT_PLANE"], d["OUT_WPR"], d["IN_WPR"]

    T = {k: [] for k in ("oc_tile", "oy", "ox_tile", "dy", "rd_base", "wgt_base", "row_mask")}
    D = {k: [] for k in DRAIN_DTYPE.names}

    wgt_tile = d["WGT_BASE"]
    ch_tile = 0
    qp_tile = d["QP_BASE"]
    out_ch_tile = 0                                     # (oc_tile*8)*OUT_PLANE
    tile = 0
    for oc_tile in range(d["OC_TILES"]):
        in_row = 0
        oy_base = 0                                     # oy (non-pool) or 2p (pool)
        out_row_off = 0
        for out_row in range(n_rows):
            ox_word = 0
            rem = d["OW"]
            for ox_tile in range(d["OW_TILES"]):
                rmask = _row_mask(rem)
                dy_off = 0
                oy = oy_base
                for dy in dys:
                    T["oc_tile"].append(oc_tile)
                    T["oy"].append(oy)
                    T["ox_tile"].append(ox_tile)
                    T["dy"].append(dy)
                    T["rd_base"].append(in_row + dy_off + ox_word)
                    T["wgt_base"].append(wgt_tile)
                    T["row_mask"].append(rmask)
                    # ---- drain steps of this tile (8 columns) ----
                    if out_raw:
                        kind = KIND_LOGIT
                    elif pool and dy == 0:
                        kind = KIND_POOL_STORE
                    else:
                        kind = KIND_ACT
                    if pool:
                        col_word = ox_word >> 1                 # ox_tile >> 1 (wiring)
                        be_all = _pool_be(rem, ox_word & 1)
                    else:
                        col_word = ox_word
                        be_all = rmask
                    ch = ch_tile
                    qp = qp_tile
                    out_ch = out_ch_tile
                    for j in range(LANES):
                        valid = ch < OC
                        if kind == KIND_ACT:
                            be = be_all if valid else 0
                            we = be != 0
                            word = out_ch + out_row_off + col_word
                        elif kind == KIND_LOGIT:
                            be, we, word = 0, valid, 0
                        else:
                            be, we, word = 0, False, 0
                        D["tile"].append(tile)
                        D["oc_tile"].append(oc_tile)
                        D["oy"].append(oy)
                        D["out_row"].append(out_row)
                        D["ox_tile"].append(ox_tile)
                        D["dy"].append(dy)
                        D["j"].append(j)
                        D["ch"].append(ch)
                        D["ch_valid"].append(valid)
                        D["qp_idx"].append(qp)
                        D["row_mask"].append(rmask)
                        D["kind"].append(kind)
                        D["we"].append(we)
                        D["buf"].append(buf)
                        D["word"].append(word)
                        D["be"].append(be)
                        D["logit_idx"].append(ch if kind == KIND_LOGIT else 0)
                        ch += 1
                        qp += 1
                        out_ch += OUT_PLANE
                    tile += 1
                    dy_off += IN_WPR
                    oy += 1
                ox_word += 1
                rem -= LANES
            in_row += IN_WPR
            oy_base += 1
            if pool:                                    # a pair advances two input rows
                in_row += IN_WPR
                oy_base += 1
            out_row_off += OUT_WPR
        wgt_tile += d["K"]
        ch_tile += LANES
        qp_tile += LANES
        out_ch_tile = out_ch                            # = (oc_tile+1)*8*OUT_PLANE
    return T, D


def _assemble_issues(T: dict, P: dict) -> np.ndarray:
    """Issue stream = per-tile base registers + the per-tile k stream (adds/broadcast only)."""
    nt = len(T["oc_tile"])
    nk = P["k"].size
    base_rd = np.array(T["rd_base"], dtype=np.int64)
    base_w = np.array(T["wgt_base"], dtype=np.int64)
    rd = (base_rd[:, None, None] + P["rd"][None, :, :]).reshape(-1, LANES)
    s = np.empty(rd.shape[0], dtype=ISSUE_DTYPE)
    s["tile"] = np.repeat(np.arange(nt), nk)
    for f in ("oc_tile", "oy", "ox_tile", "dy", "row_mask"):
        s[f] = np.repeat(np.array(T[f], dtype=np.int64), nk)
    for f in ("k", "ic", "ky", "kx", "first", "last"):
        s[f] = np.tile(P[f], nt)
    s["rot"] = s["kx"]
    s["rd_addr"] = rd
    s["wgt_addr"] = (base_w[:, None] + P["k"][None, :]).ravel()
    return s


def iter_issues(desc):
    """Cycle-by-cycle generator of the same issue stream (pure per-cycle counters, no
    per-tile reuse): yields (tile, oc_tile, oy, ox_tile, dy, k, ic, ky, kx, first, last,
    rd_addr tuple[8], rot, wgt_addr, row_mask). Reference form for small layers."""
    d = as_desc(desc)
    pool = d["pool_en"] == 1
    n_rows = d["OUT_H"] if pool else d["OH"]
    dys = (0, 1) if pool else (0,)
    ic_last, ky_last, kx_last = d["IC"] - 1, d["KH"] - 1, d["KW"] - 1
    IN_WPR, IN_PLANE = d["IN_WPR"], d["IN_PLANE"]
    wgt_tile = d["WGT_BASE"]
    tile = 0
    for oc_tile in range(d["OC_TILES"]):
        in_row = 0
        oy_base = 0
        for _ in range(n_rows):
            ox_word = 0
            rem = d["OW"]
            for ox_tile in range(d["OW_TILES"]):
                rmask = _row_mask(rem)
                dy_off = 0
                oy = oy_base
                for dy in dys:
                    wgt = wgt_tile
                    ic_base = in_row + dy_off + ox_word
                    k = 0
                    for ic in range(d["IC"]):
                        ky_off = ic_base
                        for ky in range(d["KH"]):
                            for kx in range(d["KW"]):
                                rd = tuple(ky_off + (1 if b < kx else 0) for b in range(LANES))
                                yield (tile, oc_tile, oy, ox_tile, dy, k, ic, ky, kx, k == 0,
                                       ic == ic_last and ky == ky_last and kx == kx_last,
                                       rd, kx, wgt, rmask)
                                k += 1
                                wgt += 1
                            ky_off += IN_WPR
                        ic_base += IN_PLANE
                    tile += 1
                    dy_off += IN_WPR
                    oy += 1
                ox_word += 1
                rem -= LANES
            in_row += IN_WPR
            oy_base += 1
            if pool:
                in_row += IN_WPR
                oy_base += 1
        wgt_tile += d["K"]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class AddrStream:
    """Issue + drain streams of one layer (read-only numpy structured arrays)."""

    def __init__(self, desc: dict, issues: np.ndarray, drain: np.ndarray):
        self.desc = desc
        self.issues = issues
        self.drain = drain
        self.issues.flags.writeable = False
        self.drain.flags.writeable = False

    @property
    def n_issues(self) -> int:
        return int(self.issues.size)

    @property
    def n_tiles(self) -> int:
        return int(self.drain.size) >> 3

    @property
    def n_drain(self) -> int:
        return int(self.drain.size)

    @property
    def n_writes(self) -> int:
        return int(np.count_nonzero(self.drain["we"]))

    def counts(self) -> dict:
        return stream_counts(self)


@lru_cache(maxsize=64)
def _build(key: tuple) -> AddrStream:
    d = dict(zip(USED_FIELDS, key))
    P = _k_program(d)
    T, D = _tile_walk(d)
    issues = _assemble_issues(T, P)
    drain = np.empty(len(D["tile"]), dtype=DRAIN_DTYPE)
    for f in DRAIN_DTYPE.names:
        drain[f] = np.array(D[f], dtype=np.int64)
    return AddrStream(d, issues, drain)


def addr_stream(desc) -> AddrStream:
    """Issue + drain streams for a decoded descriptor (dict) or 16 descriptor words."""
    return _build(desc_key(as_desc(desc)))


def stream_counts(st: AddrStream) -> dict:
    """Counts of a stream + the closed-form expectations (multiplication allowed here:
    this is a check helper, not the address path)."""
    import gos_cycle_model as cm
    d = st.desc
    T = d["OC_TILES"] * d["OH"] * d["OW_TILES"]
    cyc = cm.layer_cycles(d, c_pipe=None)
    return {"issues": st.n_issues, "tiles": st.n_tiles, "drain_steps": st.n_drain,
            "writes": st.n_writes,
            "act_writes": int(np.count_nonzero(st.drain["we"] & (st.drain["kind"] == KIND_ACT))),
            "logit_writes": int(np.count_nonzero(st.drain["we"] & (st.drain["kind"] == KIND_LOGIT))),
            "pool_stores": int(np.count_nonzero(st.drain["kind"] == KIND_POOL_STORE)),
            "T": T, "T_times_K": T * d["K"], "cycle_model_compute": cyc["compute_cycles"],
            "cycle_model_T": cyc["T"]}
