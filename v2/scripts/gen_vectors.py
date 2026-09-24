#!/usr/bin/env python3
"""V2 step 2.2 part F: deterministic $readmemh test-vector generator.

Writes (default) v2/vectors/generated/ and v2/vectors/MANIFEST.json. File formats
and bit layouts are documented in v2/vectors/README.md; the layouts below
(``*_LAYOUT``) are the single source for both the writer and the decoders used
by the tests. Unit-vector layouts are PROVISIONAL (may change when the RTL
interfaces are fixed).

Golden data comes only from v2/model (gos_golden, gos_pack, gos_addr_stream,
gos_tile_model, gos_cycle_model, requant_check, final_layer); every generated
expectation is cross-checked inside this script:
  * requant rows: hardware arithmetic == legacy float64 ``requantize`` == Python-int
    formula, for every row; the Step 2.1 near-tie values are asserted present;
  * layer vectors: the tile model is run on memory images READ BACK from the
    written hex files (unwritten memory filled with seeded garbage) and must
    reproduce the expected output words / LOGIT;
  * network vectors: tile model on read-back files and ``run_net_mem`` must
    reproduce LOGIT and the prediction.

Determinism: fixed seeds, fixed image indices (test[0..N-1]), no timestamps or
run-dependent values in any generated file or in MANIFEST.json.

Usage (from anywhere; .venv python):
    gen_vectors.py [--out DIR] [--manifest PATH] [--quick] [--require-clean]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import common  # noqa: E402  (puts legacy python/ on sys.path)
from common import REPO_ROOT, V2_ROOT, git_commit, git_dirty, sha256_file  # noqa: E402
import final_layer  # noqa: E402
import gos_addr_stream as ga  # noqa: E402
import gos_cycle_model as cm  # noqa: E402
import gos_golden as gg  # noqa: E402
import gos_pack as gp  # noqa: E402
import gos_tile_model as tm  # noqa: E402
from lenet5.int8_quant import requantize  # noqa: E402  (legacy float64 reference, read-only)
from net_config import NET_CONFIGS, NETS  # noqa: E402
from requant_check import hw_requant, regions, rne_shift_int  # noqa: E402

GENERATOR = "v2/scripts/gen_vectors.py"
DEFAULT_OUT = V2_ROOT / "vectors" / "generated"
DEFAULT_MANIFEST = V2_ROOT / "vectors" / "MANIFEST.json"
MARKER = ".gen_vectors_output"          # identifies a directory this script may wipe

BASE_SEED = 20260924
SEEDS = {"pe": [BASE_SEED, 1], "array": [BASE_SEED, 2], "rotator": [BASE_SEED, 3],
         "requant_lenet5": [BASE_SEED, 4, 0], "requant_cifar10": [BASE_SEED, 4, 1],
         "pool": [BASE_SEED, 5], "layer_garbage": [BASE_SEED, 6],
         "net_garbage": [BASE_SEED, 7]}

FULL = {"pe_random_seqs": 600, "array_random_tiles": 40, "rot_random_per_kx": 64,
        "requant_samples": 100_000, "pool_cases": 512, "layer_images": 3, "net_images": 10}
QUICK = {"pe_random_seqs": 60, "array_random_tiles": 6, "rot_random_per_kx": 8,
         "requant_samples": 2_000, "pool_cases": 48, "layer_images": 1, "net_images": 2}

NEAR_TIE_TOL = Fraction(1, 10**9)       # exact |v*M - (n + 1/2)| < 1e-9
NEAR_TIE_FLOAT_PREFILTER = 1e-7         # float64 v*M error in the exact region is < 1e-13
# Near-tie values that force an adjusted m at B = 32 (DECISIONS.md OC-1 / D1 outcome): must be
# present. They belong to the adopted parameter set: cifar10 = r2 since Step 2.1c (conv3 ch57,
# v*M = 63.5 - 3.8e-15); the r1 value was conv2 ch0 v = 80582.
REQUIRED_NEAR_TIES = (("lenet5", "conv5", 82, 55930), ("lenet5", "fc1", 54, 31044),
                      ("cifar10", "conv3", 57, 36475))

# --------------------------------------------------------------------------- #
# Record layouts: (field, lsb, width, signed). Unlisted bits are 0.
# --------------------------------------------------------------------------- #
PE_WIDTH = 64
PE_LAYOUT = (("acc", 0, 32, True), ("w", 32, 8, True), ("a", 40, 8, True),
             ("first", 48, 1, False), ("last", 49, 1, False))

REQUANT_WIDTH = 160
REQUANT_LAYOUT = (("acc", 0, 32, True), ("q_bias", 32, 32, True), ("m", 64, 32, False),
                  ("v", 96, 32, True), ("s", 128, 6, False), ("near_tie", 134, 1, False),
                  ("exact_region", 135, 1, False), ("q", 136, 8, True),
                  ("q_relu", 144, 8, True), ("qp_ch", 152, 8, False))

POOL_WIDTH = 192
POOL_LAYOUT = (tuple((f"dy0[{r}]", 8 * r, 8, True) for r in range(8))
               + tuple((f"dy1[{r}]", 64 + 8 * r, 8, True) for r in range(8))
               + tuple((f"h[{i}]", 128 + 8 * i, 8, True) for i in range(4))
               + (("be", 160, 8, False), ("j", 168, 4, False), ("half", 172, 1, False),
                  ("signed_data", 173, 1, False), ("rem", 176, 8, False)))

ISSUE_WIDTH = 256
ISSUE_LAYOUT = (tuple((f"rd_addr[{b}]", 16 * b, 16, False) for b in range(8))
                + (("wgt_addr", 128, 16, False), ("row_mask", 144, 8, False),
                   ("rot", 152, 4, False), ("first", 156, 1, False), ("last", 157, 1, False),
                   ("dy", 158, 1, False), ("k", 160, 16, False), ("ic", 176, 16, False),
                   ("ky", 192, 8, False), ("kx", 200, 8, False), ("ox_tile", 208, 8, False),
                   ("oy", 216, 8, False), ("oc_tile", 224, 8, False), ("tile", 240, 16, False)))

DRAIN_WIDTH = 160
DRAIN_LAYOUT = (("word", 0, 16, False), ("be", 16, 8, False), ("row_mask", 24, 8, False),
                ("qp_idx", 32, 16, False), ("ch", 48, 16, False), ("j", 64, 4, False),
                ("kind", 68, 4, False), ("we", 72, 1, False), ("ch_valid", 73, 1, False),
                ("buf", 74, 1, False), ("dy", 75, 1, False), ("logit_idx", 80, 8, False),
                ("ox_tile", 88, 8, False), ("out_row", 96, 8, False), ("oy", 104, 8, False),
                ("oc_tile", 112, 8, False), ("tile", 128, 16, False))

LAYOUTS = {"pe": (PE_WIDTH, PE_LAYOUT), "requant": (REQUANT_WIDTH, REQUANT_LAYOUT),
           "pool": (POOL_WIDTH, POOL_LAYOUT), "issue": (ISSUE_WIDTH, ISSUE_LAYOUT),
           "drain": (DRAIN_WIDTH, DRAIN_LAYOUT)}


def _check_layout(width, layout):
    used = 0
    for name, lsb, w, _ in layout:
        assert 1 <= w <= 32 and lsb + w <= width, (name, lsb, w, width)
        m = ((1 << w) - 1) << lsb
        assert not used & m, f"layout overlap at {name}"
        used |= m


for _w, _l in LAYOUTS.values():
    _check_layout(_w, _l)


def pack_records(fields: dict, layout) -> np.ndarray:
    """Composite records -> object array of Python ints (each field range-checked)."""
    n = len(np.asarray(fields[layout[0][0]]))
    out = np.zeros(n, dtype=object)
    for name, lsb, w, signed in layout:
        v = np.asarray(fields[name]).astype(np.int64)
        assert v.shape == (n,), (name, v.shape, n)
        if n:
            if signed:
                assert v.min() >= -(1 << (w - 1)) and v.max() < (1 << (w - 1)), (name, v.min(), v.max())
            else:
                assert v.min() >= 0 and v.max() < (1 << w), (name, v.min(), v.max())
        u = (v & np.int64((1 << w) - 1)).astype(object)
        out = out + (u * (1 << lsb) if lsb else u)
    return out


def unpack_records(values, layout) -> dict:
    """Inverse of pack_records: {field: int64 array} (signed fields sign-extended)."""
    arr = np.array([int(x) for x in values], dtype=object)
    out = {}
    for name, lsb, w, signed in layout:
        v = ((arr >> lsb) & ((1 << w) - 1)).astype(np.int64)
        if signed:
            v = np.where(v >= (1 << (w - 1)), v - (1 << w), v)
        out[name] = v
    return out


def reserved_bits_zero(values, width, layout) -> bool:
    used = 0
    for _, lsb, w, _ in layout:
        used |= ((1 << w) - 1) << lsb
    return all((int(x) & ~used) == 0 and int(x) < (1 << width) for x in values)


def u8(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int8).view(np.uint8)


def u32(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    assert x.size == 0 or (x.min() >= -(2**31) and x.max() < 2**31)
    return x.astype(np.int32).view(np.uint32)


def bytes_to_word(b) -> np.ndarray:
    """int8 [..., 8] -> uint64 [...] with byte i at bits 8i+7:8i."""
    b = np.asarray(b, dtype=np.int8).view(np.uint8).astype(np.uint64)
    return (b << (np.arange(8, dtype=np.uint64) * np.uint64(8))).sum(axis=-1, dtype=np.uint64)


def word_to_bytes(w) -> np.ndarray:
    w = np.asarray(w, dtype=np.uint64)
    return ((w[..., None] >> (np.arange(8, dtype=np.uint64) * np.uint64(8)))
            & np.uint64(0xFF)).astype(np.uint8).view(np.int8)


# --------------------------------------------------------------------------- #
# Output bookkeeping
# --------------------------------------------------------------------------- #
class Out:
    def __init__(self, root: Path):
        self.root = root
        self.files = []

    def emit(self, rel: str, values, width: int, desc: str) -> Path:
        path = self.root / rel
        vals = values if isinstance(values, np.ndarray) and values.dtype == object else np.asarray(values)
        gp.write_hex(path, vals, width)
        self.files.append({"path": rel, "width_bits": width, "lines": int(np.asarray(vals).size),
                           "description": desc})
        return path

    def read(self, rel: str) -> list[int]:
        return gp.read_hex(self.root / rel)


def int8_choice(rng, n, corner_p=0.15, nonneg=False):
    """int8 values, uniform, with probability corner_p drawn from the corners."""
    lo = 0 if nonneg else -128
    v = rng.integers(lo, 128, size=n)
    corners = np.array([0, 1, 127] if nonneg else [-128, -127, -1, 0, 1, 127])
    c = rng.random(n) < corner_p
    v[c] = rng.choice(corners, size=int(c.sum()))
    return v.astype(np.int64)


# --------------------------------------------------------------------------- #
# 1. Unit vectors
# --------------------------------------------------------------------------- #
def gen_pe(o: Out, P: dict) -> dict:
    rng = np.random.default_rng(SEEDS["pe"])
    seqs = []
    corners = (-128, -127, -1, 0, 1, 127)
    for a in corners:                                   # length-1 corner products
        for w in corners:
            seqs.append((np.array([a]), np.array([w])))
    for a, w in ((-128, -128), (127, -128), (-128, 127), (127, 127)):   # K=800 extremes
        seqs.append((np.full(800, a), np.full(800, w)))
    seqs.append((np.full(25, -128), np.full(25, -128)))
    for _ in range(P["pe_random_seqs"]):
        u = rng.random()
        n = 1 if u < 0.2 else int(rng.integers(2, 33) if u < 0.7 else
                                   rng.integers(33, 201) if u < 0.95 else rng.integers(201, 801))
        seqs.append((int8_choice(rng, n), int8_choice(rng, n)))
    a = np.concatenate([s[0] for s in seqs]).astype(np.int64)
    w = np.concatenate([s[1] for s in seqs]).astype(np.int64)
    first = np.concatenate([np.arange(len(s[0])) == 0 for s in seqs])
    last = np.concatenate([np.arange(len(s[0])) == len(s[0]) - 1 for s in seqs])
    acc = np.empty(a.size, dtype=np.int64)
    cur = 0
    for i in range(a.size):                             # acc <= first ? a*w : acc + a*w
        p = int(a[i]) * int(w[i])
        cur = p if first[i] else cur + p
        acc[i] = cur
    assert np.abs(acc).max() < 2**31
    results = acc[last]
    assert results.size == len(seqs)
    o.emit("unit/pe/pe_cycles.hex",
           pack_records({"acc": acc, "w": w, "a": a, "first": first, "last": last}, PE_LAYOUT),
           PE_WIDTH, "PE: one cycle per line {a, w, first, last, expected acc after the cycle}")
    o.emit("unit/pe/pe_results.hex", u32(results), 32,
           "PE: expected acc (int32) at the last cycle of each sequence, in order")
    return {"cycles": int(a.size), "sequences": len(seqs), "max_abs_acc": int(np.abs(acc).max())}


def gen_array(o: Out, P: dict) -> dict:
    rng = np.random.default_rng(SEEDS["array"])
    tiles = []                                          # (a [K,8], w [K,8])
    ext_a = np.array([-128, 127, -128, 127, -128, -1, 0, 1])
    ext_w = np.array([-128, -128, 127, 127, -1, 1, 0, -128])
    for K in (1, 25, 800):
        tiles.append((np.tile(ext_a, (K, 1)), np.tile(ext_w, (K, 1))))          # extremes
        tiles.append((np.full((K, 8), -128), np.full((K, 8), -128)))            # max acc
        tiles.append((np.full((K, 8), 127), np.full((K, 8), -128)))             # min acc
        tiles.append((int8_choice(rng, K * 8).reshape(K, 8), int8_choice(rng, K * 8).reshape(K, 8)))
    for _ in range(P["array_random_tiles"]):
        K = int(rng.integers(1, 65)) if rng.random() < 0.85 else int(rng.integers(65, 801))
        tiles.append((int8_choice(rng, K * 8).reshape(K, 8), int8_choice(rng, K * 8).reshape(K, 8)))
    A = np.concatenate([t[0] for t in tiles]).astype(np.int64)
    W = np.concatenate([t[1] for t in tiles]).astype(np.int64)
    Ks = np.array([t[0].shape[0] for t in tiles])
    first = np.concatenate([np.arange(K) == 0 for K in Ks])
    last = np.concatenate([np.arange(K) == K - 1 for K in Ks])
    accs = np.stack([t[0].astype(np.int64).T @ t[1].astype(np.int64) for t in tiles])  # [T, r, j]
    assert np.abs(accs).max() <= 800 * 128 * 128
    o.emit("unit/array/array_a.hex", bytes_to_word(A), 64,
           "array: per cycle, 8 activations; byte r (bits 8r+7:8r) = a[row r]")
    o.emit("unit/array/array_w.hex", bytes_to_word(W), 64,
           "array: per cycle, 8 weights; byte j (bits 8j+7:8j) = w[column j]")
    o.emit("unit/array/array_ctl.hex", first.astype(np.int64) | (last.astype(np.int64) << 1), 8,
           "array: per cycle control, bit0 first, bit1 last")
    o.emit("unit/array/array_k.hex", Ks, 32, "array: K (cycles) of each tile, in order")
    o.emit("unit/array/array_acc.hex", u32(accs.reshape(-1)), 32,
           "array: expected int32 acc at last, 64 lines per tile, line tile*64 + r*8 + j")
    return {"tiles": len(tiles), "cycles": int(Ks.sum()), "K_values": sorted(set(int(k) for k in Ks)),
            "max_abs_acc": int(np.abs(accs).max())}


def gen_rotator(o: Out, P: dict) -> dict:
    rng = np.random.default_rng(SEEDS["rotator"])
    fixed = [0x0706050403020100, 0x8081828384858687, 0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
             0x7F80017F80017F80]
    banks, kxs = [], []
    for kx in range(8):
        words = fixed + [int(x) for x in rng.integers(0, 2**63, size=P["rot_random_per_kx"],
                                                      dtype=np.int64).astype(np.uint64)
                         ^ (rng.integers(0, 2, size=P["rot_random_per_kx"]).astype(np.uint64)
                            << np.uint64(63))]
        banks += words
        kxs += [kx] * len(words)
    banks = np.array(banks, dtype=np.uint64)
    kxs = np.array(kxs, dtype=np.int64)
    bb = word_to_bytes(banks)                           # [n, bank]
    rows = np.take_along_axis(bb, (np.arange(8)[None, :] + kxs[:, None]) & 7, axis=1)
    for kx in range(8):                                 # rule check: bank b -> row (b - kx) mod 8
        s = kxs == kx
        for b in range(8):
            assert np.array_equal(rows[s, (b - kx) % 8], bb[s, b])
    o.emit("unit/rotator/rot_banks.hex", banks, 64, "rotator: input, byte b = ACT bank b")
    o.emit("unit/rotator/rot_kx.hex", kxs, 8, "rotator: kx (0..7) for each input line")
    o.emit("unit/rotator/rot_rows.hex", bytes_to_word(rows), 64,
           "rotator: expected output, byte r = row r = bank (r + kx) mod 8")
    return {"vectors": int(banks.size)}


def requant_channels(net: str) -> list[dict]:
    """Every requantized channel of `net` with K, q_bias, M (float64), m, s, QPARAM index."""
    cfg = NET_CONFIGS[net]
    q = np.load(cfg["quant_params"])
    params = gp.load_params(net)
    descs, _ = gp.make_descriptors(net)
    out = []
    for L, d in zip(cfg["layers"], descs):
        if L["final"]:
            continue
        M = np.asarray(q[f"{L['name']}_M"], dtype=np.float64)
        pr = params[L["name"]]
        for c in range(L["OC"]):
            out.append({"layer": L["name"], "c": c, "K": L["K"], "qb": int(pr["q_b"][c]),
                        "M": float(M[c]), "m": int(pr["m"][c]), "s": int(pr["s"][c]),
                        "qp": d["QP_BASE"] + c})
    return out


def near_ties(ch: dict) -> list[int]:
    """All v in the exact region with exact |v*M - (n + 1/2)| < 1e-9 (Fraction(float M))."""
    vmin, vmax, lo, hi = regions(ch["K"], ch["qb"], ch["M"])
    Mf = Fraction(ch["M"])
    found = []
    step = 1 << 22
    for a in range(lo, hi + 1, step):
        v = np.arange(a, min(hi, a + step - 1) + 1, dtype=np.int64)
        y = v.astype(np.float64) * ch["M"]
        cand = v[np.abs(y - np.floor(y) - 0.5) < NEAR_TIE_FLOAT_PREFILTER]
        for x in cand.tolist():
            e = Fraction(x) * Mf
            n = e.numerator // e.denominator
            if abs(e - n - Fraction(1, 2)) < NEAR_TIE_TOL:
                found.append(int(x))
    return found


def gen_requant(o: Out, P: dict, net: str) -> dict:
    rng = np.random.default_rng(SEEDS[f"requant_{net}"])
    chans = requant_channels(net)
    n = P["requant_samples"]
    ci = rng.integers(0, len(chans), size=n)
    exact_pick = (np.arange(n) & 1).astype(bool)        # odd rows: exact region
    u = rng.random(n)
    v = np.empty(n, dtype=np.int64)
    for i in range(n):
        ch = chans[ci[i]]
        vmin, vmax, lo, hi = regions(ch["K"], ch["qb"], ch["M"])
        a, b = (lo, hi) if exact_pick[i] else (vmin, vmax)
        v[i] = a + min(int(u[i] * (b - a + 1)), b - a)
    tie_rows = []
    for k, ch in enumerate(chans):
        for x in near_ties(ch):
            tie_rows.append((k, x))
    for net_, layer, c, val in REQUIRED_NEAR_TIES:
        if net_ != net:
            continue
        k = next(i for i, ch in enumerate(chans) if ch["layer"] == layer and ch["c"] == c)
        for sgn in (1, -1):
            assert (k, sgn * val) in tie_rows, f"required near-tie {net}/{layer}/{c} v={sgn * val} missing"
    ci = np.concatenate([ci, np.array([t[0] for t in tie_rows], dtype=np.int64)])
    v = np.concatenate([v, np.array([t[1] for t in tie_rows], dtype=np.int64)])
    near = np.concatenate([np.zeros(n, bool), np.ones(len(tie_rows), bool)])
    N = v.size
    qb = np.array([chans[k]["qb"] for k in ci], dtype=np.int64)
    m = np.array([chans[k]["m"] for k in ci], dtype=np.int64)
    s = np.array([chans[k]["s"] for k in ci], dtype=np.int64)
    Mv = np.array([chans[k]["M"] for k in ci], dtype=np.float64)
    qp = np.array([chans[k]["qp"] for k in ci], dtype=np.int64)
    acc = v - qb
    Kv = np.array([chans[k]["K"] for k in ci], dtype=np.int64)
    assert np.all(np.abs(acc) <= Kv * 16384), "v outside the reachable range"
    exact = np.zeros(N, dtype=bool)
    q = np.empty(N, dtype=np.int64)
    ref = np.empty(N, dtype=np.int64)
    for k in np.unique(ci):                              # per channel: hw vs legacy float64
        sel = np.flatnonzero(ci == k)
        ch = chans[k]
        _, _, lo, hi = regions(ch["K"], ch["qb"], ch["M"])
        exact[sel] = (v[sel] >= lo) & (v[sel] <= hi)
        q[sel] = hw_requant(v[sel], ch["m"], ch["s"])
        ref[sel] = requantize(v[sel], ch["M"])
    assert np.array_equal(q, ref), f"{net}: hw requant != legacy float64 requantize"
    assert np.all(exact[near]) and np.all(exact[:n][exact_pick]), "exact-region flag"
    for i in range(N):                                   # Python-int formula, every row
        qi = min(127, max(-128, rne_shift_int(int(v[i]) * int(m[i]), int(s[i]))))
        assert qi == q[i], (net, i)
    q_relu = np.maximum(q, 0)
    recs = pack_records({"acc": acc, "q_bias": qb, "m": m, "v": v, "s": s, "near_tie": near,
                         "exact_region": exact, "q": q, "q_relu": q_relu, "qp_ch": qp},
                        REQUANT_LAYOUT)
    o.emit(f"unit/requant/{net}_requant.hex", recs, REQUANT_WIDTH,
           f"requant lane ({net}): {n} seeded rows + {len(tie_rows)} near-tie rows "
           "{acc, q_bias, m, v, s, flags, q pre-ReLU, q post-ReLU, QPARAM channel}")
    return {"rows": int(N), "sampled": n, "near_tie_rows": len(tie_rows),
            "exact_region_rows": int(exact.sum()), "channels": len(chans),
            "near_ties": [{"layer": chans[k]["layer"], "ch": chans[k]["c"], "v": x}
                          for k, x in tie_rows]}


def gen_pool(o: Out, P: dict) -> dict:
    rng = np.random.default_rng(SEEDS["pool"])
    F = {name: [] for name, *_ in POOL_LAYOUT}
    n_cases = P["pool_cases"]
    rems = (2, 4, 6, 8)
    for c in range(n_cases):
        half = c & 1
        signed = (c >> 1) & 1
        rem = rems[(c >> 2) % 4] if (c >> 2) % 5 != 4 else int(rng.integers(5, 16)) * 2
        kind = c % 11
        if kind == 0:
            d0, d1 = np.full((8, 8), -128 if signed else 0), np.full((8, 8), -128 if signed else 0)
        elif kind == 1:
            d0, d1 = np.full((8, 8), 127), np.full((8, 8), 127)
        elif kind == 2:          # dy0 dominates
            d0 = int8_choice(rng, 64, nonneg=not signed).reshape(8, 8)
            d1 = np.minimum(d0, int8_choice(rng, 64, nonneg=not signed).reshape(8, 8))
        else:
            d0 = int8_choice(rng, 64, nonneg=not signed).reshape(8, 8)
            d1 = int8_choice(rng, 64, nonneg=not signed).reshape(8, 8)
        vp = rem >> 1
        be = (0xF if vp >= 4 else (1 << vp) - 1) << (4 * half)
        assert be == ga._pool_be(rem, half)
        for j in range(8):
            vert = np.maximum(d0[:, j], d1[:, j])                    # vertical max
            h = np.maximum(vert[0::2], vert[1::2])                   # horizontal pair max
            for r in range(8):
                F[f"dy0[{r}]"].append(d0[r, j])
                F[f"dy1[{r}]"].append(d1[r, j])
            for i in range(4):
                F[f"h[{i}]"].append(h[i])
            F["be"].append(be)
            F["j"].append(j)
            F["half"].append(half)
            F["signed_data"].append(signed)
            F["rem"].append(rem)
    o.emit("unit/pool/pool_cols.hex", pack_records(F, POOL_LAYOUT), POOL_WIDTH,
           "pool: one drain column per line {dy0 col, dy1 col, expected 4 pooled bytes, be}")
    return {"cases": n_cases, "columns": n_cases * 8}


# --------------------------------------------------------------------------- #
# 2./3. Net, layer and network vectors
# --------------------------------------------------------------------------- #
def stream_fields(st: ga.AddrStream) -> tuple[dict, dict]:
    iss, dr = st.issues, st.drain
    fi = {f"rd_addr[{b}]": iss["rd_addr"][:, b] for b in range(8)}
    for f in ("wgt_addr", "row_mask", "rot", "first", "last", "dy", "k", "ic", "ky", "kx",
              "ox_tile", "oy", "oc_tile", "tile"):
        fi[f] = iss[f]
    fd = {f: dr[f] for f in ("word", "be", "row_mask", "qp_idx", "ch", "j", "kind", "we",
                             "ch_valid", "buf", "dy", "logit_idx", "ox_tile", "out_row", "oy",
                             "oc_tile", "tile")}
    return fi, fd


def out_byte_mask(st: ga.AddrStream, n_words: int) -> np.ndarray:
    """Per output word 0..OUT_END: OR of the byte enables of every ACT write."""
    dr = st.drain
    sel = dr["we"] & (dr["kind"] == ga.KIND_ACT)
    mask = np.zeros(n_words, dtype=np.int64)
    np.bitwise_or.at(mask, dr["word"][sel].astype(np.int64), dr["be"][sel].astype(np.int64))
    return mask


def mask64(bytemask) -> np.ndarray:
    bm = np.asarray(bytemask, dtype=np.int64)
    bits = (bm[:, None] >> np.arange(8)[None, :]) & 1
    return (bits.astype(np.uint64) * np.uint64(0xFF) << (np.arange(8, dtype=np.uint64) * np.uint64(8))
            ).sum(axis=1, dtype=np.uint64)


def garbage_mem(rng) -> tm.Mem:
    return tm.Mem(act=rng.integers(0, 256, size=(2, 8, gp.ACT_DEPTH), dtype=np.uint8),
                  wgt=rng.integers(0, 2**63, size=gp.WGT_DEPTH, dtype=np.int64).astype(np.uint64),
                  qp_e=rng.integers(0, 2**63, size=gp.QP_CHANNELS, dtype=np.int64).astype(np.uint64),
                  qp_o=rng.integers(0, 2**63, size=gp.QP_CHANNELS, dtype=np.int64).astype(np.uint64),
                  logit=rng.integers(-2**31, 2**31, size=gp.N_LOGITS, dtype=np.int64).astype(np.int32))


def load_net_files(o: Out, net: str, mem: tm.Mem) -> np.ndarray:
    """Place the read-back net WGT/QPARAM files into `mem`; return descriptor words [N,16]."""
    w = np.array(o.read(f"{net}/wgt.hex"), dtype=np.uint64)
    mem.wgt[:w.size] = w
    e = np.array(o.read(f"{net}/qparam_e.hex"), dtype=np.uint64)
    od = np.array(o.read(f"{net}/qparam_o.hex"), dtype=np.uint64)
    mem.qp_e[:e.size] = e
    mem.qp_o[:od.size] = od
    n = o.read(f"{net}/n_layers.hex")[0]
    return np.array(o.read(f"{net}/desc.hex"), dtype=np.uint32).reshape(n, gp.DESC_WORDS)


def gen_net_common(o: Out, net: str) -> dict:
    wgt, wb = gp.pack_wgt(net)
    qp, qb = gp.pack_qparam(net)
    e, od = gp.qparam_banks(qp)
    descs, words = gp.make_descriptors(net)
    assert [d["WGT_BASE"] for d in descs] == wb and [d["QP_BASE"] for d in descs] == qb
    o.emit(f"{net}/wgt.hex", wgt, 64, f"{net}: full WGT image from word 0 (all layers, FORMATS 2)")
    o.emit(f"{net}/qparam.hex", qp, 64,
           f"{net}: QPARAM combined PS view, word 2i = {{m, q_bias}}, 2i+1 = {{0, s}} (FORMATS 3)")
    o.emit(f"{net}/qparam_e.hex", e, 64, f"{net}: QPARAM bank QP_E[i] = combined word 2i")
    o.emit(f"{net}/qparam_o.hex", od, 64, f"{net}: QPARAM bank QP_O[i] = combined word 2i+1")
    o.emit(f"{net}/desc.hex", words.reshape(-1), 32,
           f"{net}: descriptors, line 16*layer + w = DESC[layer] word w (FORMATS 5)")
    o.emit(f"{net}/n_layers.hex", [len(descs)], 32, f"{net}: N_LAYERS")
    # read-back equals the packers
    assert o.read(f"{net}/wgt.hex") == [int(x) for x in wgt]
    assert o.read(f"{net}/qparam.hex") == [int(x) for x in qp]
    return {"descs": descs, "words": words}


def layer_dir(i: int, name: str) -> str:
    return f"L{i}_{name}"


def gen_layers(o: Out, P: dict, net: str, common_: dict, x_int8: np.ndarray) -> list[dict]:
    """Layer-level vectors for images x_int8 [N,...] = test[0..N-1]."""
    G = gg.load_net(net)
    gold = gg.run_net(G, x_int8)
    descs, words = common_["descs"], common_["words"]
    cfg_layers = NET_CONFIGS[net]["layers"]
    rng = np.random.default_rng(SEEDS["layer_garbage"] + [NETS.index(net)])
    summary = []
    prev = "input"
    for i, (d, w, L) in enumerate(zip(descs, words, cfg_layers)):
        ld = f"{net}/layers/{layer_dir(i, d['name'])}"
        o.emit(f"{ld}/desc.hex", w, 32, f"{net}/{d['name']}: descriptor (16 words); "
                                        "WGT_BASE/QP_BASE index the net-level wgt/qparam files")
        wb = np.array(o.read(f"{ld}/desc.hex"), dtype=np.uint32)
        st = ga.addr_stream(wb)                                         # from the read-back desc
        cyc = cm.layer_cycles(L, c_pipe=None)
        T, K, TK = cyc["T"], cyc["K"], cyc["compute_cycles"]
        assert st.n_issues == TK and st.n_tiles == T and st.n_drain == 8 * T
        o.emit(f"{ld}/expect_tk.hex", [T, K, TK], 32,
               f"{net}/{d['name']}: expected T, K, T*K (gos_cycle_model, compute only)")
        fi, fd = stream_fields(st)
        o.emit(f"{ld}/issue.hex", pack_records(fi, ISSUE_LAYOUT), ISSUE_WIDTH,
               f"{net}/{d['name']}: issue stream, one record per cycle (T*K = {TK})")
        o.emit(f"{ld}/drain.hex", pack_records(fd, DRAIN_LAYOUT), DRAIN_WIDTH,
               f"{net}/{d['name']}: drain/write event stream, 8 records per tile ({8 * T})")
        # round trip of the stream files
        ri = unpack_records(o.read(f"{ld}/issue.hex"), ISSUE_LAYOUT)
        rd = unpack_records(o.read(f"{ld}/drain.hex"), DRAIN_LAYOUT)
        for f, a in fi.items():
            assert np.array_equal(ri[f], np.asarray(a).astype(np.int64)), f
        for f, a in fd.items():
            assert np.array_equal(rd[f], np.asarray(a).astype(np.int64)), f
        n_in = d["IN_END"] + 1
        out_raw = bool(d["out_raw"])
        if not out_raw:
            n_out = d["OUT_END"] + 1
            bm = out_byte_mask(st, n_out)
            ones = np.full((d["OC"], d["OUT_H"], d["OUT_W"]), -1, dtype=np.int8)
            geo = word_to_bytes(gp.pack_act_words(ones, n_out)).view(np.uint8)
            assert np.array_equal(bm, (geo != 0).astype(np.int64) @ (1 << np.arange(8))), \
                "stream write mask != output map geometry"
            o.emit(f"{ld}/act_out_mask.hex", bm, 8,
                   f"{net}/{d['name']}: byte-enable mask of the written bytes of each output word")
        for n in range(x_int8.shape[0]):
            x_in = gold[prev][n]
            img = f"{ld}/img{n}"
            o.emit(f"{img}/act_in.hex", gp.pack_act_words(x_in, n_in), 64,
                   f"{net}/{d['name']} test[{n}]: input ACT[in_sel={d['in_sel']}] words 0..IN_END")
            y = gold[d["name"]][n]
            if out_raw:
                v = np.zeros(gp.N_LOGITS, dtype=np.int64)
                v[:d["OC"]] = y.reshape(-1)
                o.emit(f"{img}/logit16.hex", u32(v), 32,
                       f"{net}/{d['name']} test[{n}]: expected LOGIT[0..15] int32 (unused = 0)")
            else:
                o.emit(f"{img}/act_out.hex", gp.pack_act_words(y, n_out), 64,
                       f"{net}/{d['name']} test[{n}]: expected ACT[{1 - d['in_sel']}] words "
                       "0..OUT_END (unwritten bytes 0; compare under act_out_mask)")
            # ---- cross-check: tile model on the read-back files, garbage elsewhere ----
            mem = garbage_mem(rng)
            dw = load_net_files(o, net, mem)
            assert np.array_equal(dw[i], wb)
            a_in = np.array(o.read(f"{img}/act_in.hex"), dtype=np.uint64)
            mem.act[d["in_sel"], :, :n_in] = gp.act_words_to_banks(a_in)
            before = mem.copy()
            tm.run_layer_mem(mem, wb)
            ob = 1 - d["in_sel"]
            assert np.array_equal(mem.act[d["in_sel"]], before.act[d["in_sel"]])
            if out_raw:
                exp = np.array(o.read(f"{img}/logit16.hex"), dtype=np.uint32).view(np.int32)
                assert np.array_equal(mem.logit[:d["OC"]], exp[:d["OC"]]), (net, d["name"], n)
                assert np.array_equal(mem.logit[d["OC"]:], before.logit[d["OC"]:])
            else:
                exp = np.array(o.read(f"{img}/act_out.hex"), dtype=np.uint64)
                mk = mask64(o.read(f"{ld}/act_out_mask.hex"))
                got = gp.act_banks_to_words(mem.act[ob])
                old = gp.act_banks_to_words(before.act[ob])
                assert np.all(((got[:n_out] ^ exp) & mk) == 0), (net, d["name"], n)
                assert np.all(((got[:n_out] ^ old[:n_out]) & ~mk) == 0)
                assert np.array_equal(got[n_out:], old[n_out:])
                assert np.all((exp & ~mk) == 0)
                assert np.array_equal(gp.unpack_act(exp, d["OC"], d["OUT_H"], d["OUT_W"]), y)
        summary.append({"layer": d["name"], "dir": ld, "in_sel": d["in_sel"], "T": T, "K": K,
                        "T_times_K": TK, "issues": st.n_issues, "drain_records": st.n_drain,
                        "writes": st.n_writes, "in_words": n_in,
                        "out_words": None if out_raw else d["OUT_END"] + 1, "out_raw": out_raw})
        prev = d["name"]
    return summary


def gen_network(o: Out, P: dict, net: str, x_int8: np.ndarray, labels: np.ndarray) -> dict:
    G = gg.load_net(net)
    gold = gg.run_net(G, x_int8)
    d0 = gp.decode_descriptor(gp.make_descriptors(net)[1][0])
    n_in = d0["IN_END"] + 1
    OC = G.final.cfg["OC"]
    rng = np.random.default_rng(SEEDS["net_garbage"] + [NETS.index(net)])
    params = {"S_a": G.final.S_a, "S_w": G.final.S_w}
    preds = []
    for n in range(x_int8.shape[0]):
        img = f"{net}/net/img{n}"
        o.emit(f"{img}/act0.hex", gp.pack_act_words(x_int8[n], n_in), 64,
               f"{net} test[{n}]: initial ACT0 image, words 0..IN_END of layer 0")
        v = gold["v"][n].astype(np.int64)
        v16 = np.zeros(gp.N_LOGITS, dtype=np.int64)
        v16[:OC] = v
        o.emit(f"{img}/logit10.hex", u32(v), 32, f"{net} test[{n}]: expected LOGIT[0..{OC - 1}] int32")
        o.emit(f"{img}/logit16.hex", u32(v16), 32,
               f"{net} test[{n}]: expected LOGIT[0..15] int32 (unused = 0)")
        # tile model on the read-back files (garbage elsewhere)
        mem = garbage_mem(rng)
        dw = load_net_files(o, net, mem)
        mem.act[0, :, :n_in] = gp.act_words_to_banks(np.array(o.read(f"{img}/act0.hex"), dtype=np.uint64))
        lg0 = mem.logit.copy()
        for w in dw:
            tm.run_layer_mem(mem, w)
        exp = np.array(o.read(f"{img}/logit16.hex"), dtype=np.uint32).view(np.int32)
        assert np.array_equal(mem.logit[:OC], exp[:OC]), (net, n)
        assert np.array_equal(mem.logit[OC:], lg0[OC:])
        r = tm.run_net_mem(net, x_int8[n])
        assert np.array_equal(r["logits"], exp[:OC])
        pred = int(final_layer.predict_from_raw(r["logits"][None].astype(np.int32), params)[0])
        assert pred == int(gold["pred"][n])
        preds.append(pred)
    o.emit(f"{net}/net/pred.hex", preds, 8, f"{net}: expected prediction per image (PS float32 argmax)")
    o.emit(f"{net}/net/label.hex", [int(x) for x in labels], 8, f"{net}: dataset label per image")
    return {"images": int(x_int8.shape[0]), "preds": preds, "labels": [int(x) for x in labels],
            "correct": int(sum(p == int(lbl) for p, lbl in zip(preds, labels)))}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def prepare_out(out: Path) -> None:
    if out.exists():
        entries = list(out.iterdir())
        if entries and not (out / MARKER).exists():
            raise SystemExit(f"refusing to overwrite non-empty {out} (no {MARKER} marker)")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / MARKER).write_text("generated by v2/scripts/gen_vectors.py; this directory is wiped on regeneration\n")


def rel_repo(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def generate(out: Path, manifest: Path, quick: bool = False, require_clean: bool = False,
             verbose: bool = True) -> dict:
    t0 = time.time()
    dirty = git_dirty()
    if dirty:
        if require_clean:
            raise SystemExit("gen_vectors: REFUSED (--require-clean): working tree is dirty")
        print("!" * 78 + "\n!! WARNING: git tree is DIRTY - these vectors are NOT reproducible from "
              "a commit.\n!! MANIFEST records git_dirty=true. Regenerate from a clean tree.\n"
              + "!" * 78, file=sys.stderr)
    P = QUICK if quick else FULL
    out = Path(out)
    prepare_out(out)
    o = Out(out)
    info = {"unit": {}, "nets": {}}
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    info["unit"]["pe"] = gen_pe(o, P)
    info["unit"]["array"] = gen_array(o, P)
    info["unit"]["rotator"] = gen_rotator(o, P)
    info["unit"]["pool"] = gen_pool(o, P)
    log(f"  unit pe/array/rotator/pool done ({time.time() - t0:.1f}s)")
    info["unit"]["requant"] = {}
    for net in NETS:
        info["unit"]["requant"][net] = gen_requant(o, P, net)
        log(f"  unit requant {net} done ({time.time() - t0:.1f}s)")
    for net in NETS:
        n_img = max(P["layer_images"], P["net_images"])
        x_fp, y = gg.load_test_set(net, n_img)
        x_int8 = gg.quantize_input(net, x_fp)
        c = gen_net_common(o, net)
        layers = gen_layers(o, P, net, c, x_int8[:P["layer_images"]])
        netv = gen_network(o, P, net, x_int8[:P["net_images"]], y[:P["net_images"]])
        info["nets"][net] = {"n_layers": len(c["descs"]), "layers": layers, "network": netv}
        log(f"  {net} layer + network vectors done ({time.time() - t0:.1f}s)")

    files = []
    for f in sorted(o.files, key=lambda f: f["path"]):
        p = out / f["path"]
        text = p.read_text()
        assert text.count("\n") == f["lines"]
        files.append({"path": f["path"], "sha256": sha256_file(p), "bytes": p.stat().st_size,
                      "lines": f["lines"], "width_bits": f["width_bits"],
                      "description": f["description"]})
    assert len({f["path"] for f in files}) == len(files)
    on_disk = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != MARKER)
    assert on_disk == [f["path"] for f in files], "unlisted files in the output directory"
    man = {
        "generator": GENERATOR,
        "git_commit": git_commit(),
        "git_dirty": dirty,
        "mode": "quick" if quick else "full",
        "generated_dir": rel_repo(out),
        "sizes": P,
        "seeds": {k: v for k, v in SEEDS.items()},
        "image_indices": {"layer": list(range(P["layer_images"])),
                          "network": list(range(P["net_images"])), "split": "test"},
        "params": {net: {k: {"path": rel_repo(NET_CONFIGS[net][k]),
                             "sha256": sha256_file(NET_CONFIGS[net][k])}
                         for k in ("quant_params", "hw_requant")} for net in NETS},
        "layouts": {k: {"width_bits": w, "fields": [list(f) for f in lay]}
                    for k, (w, lay) in LAYOUTS.items()},
        "summary": info,
        "n_files": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(man, indent=1) + "\n")
    log(f"gen_vectors: {len(files)} files, {man['total_bytes']:,} bytes -> {out}; "
        f"manifest {manifest}; git {man['git_commit'][:7]} dirty={dirty}; "
        f"{time.time() - t0:.1f}s")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="V2 $readmemh vector generator (step 2.2 F)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (wiped)")
    ap.add_argument("--manifest", default=None,
                    help="manifest path (default: v2/vectors/MANIFEST.json for the default "
                         "--out, else OUT/MANIFEST.json)")
    ap.add_argument("--quick", action="store_true", help="reduced sizes (tests)")
    ap.add_argument("--require-clean", action="store_true", help="refuse to run on a dirty tree")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if a.manifest:
        manifest = Path(a.manifest)
    elif out == DEFAULT_OUT.resolve():
        manifest = DEFAULT_MANIFEST
    else:
        manifest = out / "MANIFEST.json"
    generate(out, manifest, quick=a.quick, require_clean=a.require_clean)
    return 0


if __name__ == "__main__":
    sys.exit(main())
