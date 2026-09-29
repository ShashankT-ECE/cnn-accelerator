#!/usr/bin/env python3
"""A3-general random-shape set: seeded random multi-layer gos_ jobs, golden outputs, model cycles.

Generates a set of jobs (default 300, seed DEFAULT_SEED) inside the V2 architecture envelope
(ARCH_SPEC.md, FORMATS.md) and writes them to v2/build/shapes/<seed>/:

  shapes.json          manifest: format, seed, generator/model hashes, git state, limits,
                       category counts, per-file sha256 + bytes, per-job summary, shapeset_sha256
  jobs/J0000.npz ...   one deterministic npz per job (layout: v2/shapes/README.md, loader
                       v2/board/shapeset.py)
  hex/                 $readmemh files for the RTL harness v2/shapes/tb_shapes.sv
                       (hex/jobs.hex, hex/xcheck.hex, hex/J0000/{meta,desc,wgt,qparam,act_in,
                       exp_cyc,out_exp,out_mask,logit}.hex)

Everything numeric comes from v2/model (imported read-only): descriptors from
gos_pack.derive_fields/encode_descriptor (host-side consistency asserts), checker verdicts from
gos_pack.check_descriptor / job_err_code, golden outputs from gos_golden.gos_layer (integer
path), requant (m, s) from requant_check.select_m_s (B = 32), cycles from
gos_cycle_model.net_cycles (RTL-derived C_PIPE / C_START / C_DONE). Every accepted job is
also re-run through the address-level tile model (gos_tile_model.run_layer_mem) on
garbage-filled memories and must equal the golden (--no-tile-check skips this).

Label of every number produced here: model.

Run:  .venv/bin/python v2/shapes/gen_shapes.py [--seed S] [--n-jobs N] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

SHAPES_DIR = Path(__file__).resolve().parent
V2 = SHAPES_DIR.parent
MODEL_DIR = V2 / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import common  # noqa: E402  (legacy python/ on sys.path; git helpers)
import gos_cycle_model as cm  # noqa: E402
import gos_fuzz  # noqa: E402
import gos_golden as gg  # noqa: E402
import gos_pack as gp  # noqa: E402
import gos_tile_model as tm  # noqa: E402
from requant_check import ShiftOutOfRange, select_m_s  # noqa: E402

FORMAT = "gos_shapeset/1"
GENERATOR = "v2/shapes/gen_shapes.py"
DEFAULT_SEED = 20260929
DEFAULT_OUT = V2 / "build" / "shapes"
B = gp.M_BITS                                  # 32 (DECISIONS D1)

# ---- envelope limits (FORMATS.md / ARCH_SPEC; the ones marked "harness" are sim-time caps) ----
LIMITS = {
    "max_layers": gp.MAX_LAYERS,               # 8
    "act_depth": gp.ACT_DEPTH,                 # 4096 words per ACT buffer
    "wgt_depth": gp.WGT_DEPTH,                 # 16384 words
    "qp_channels": gp.QP_CHANNELS,             # 256
    "min_k": gp.MIN_K,                         # 8
    "max_logits": gp.N_LOGITS,                 # 16 (out_raw OC)
    # conflict-free read rd[b] = rowbase + ox0/8 + (b < kx), rot = kx[2:0] is exact only for
    # kx <= 8 (KW <= 9); the checker does not test KW. The generator stays at KW <= 8.
    "max_kw": 8,
    "max_kh": 8,
    # requant multiply uses a signed V_MUL_W = 26-bit v: K*16384 + |q_bias| < 2^25 -> K <= 2047
    "max_k": 1800,
    "max_hw": 96,                              # input map H, W
    "layer_tk_max": 150_000,                   # harness: T*K per layer
    "job_tk_max": 400_000,                     # harness: sum of T*K per job
    "large_layer_tk_max": 800_000,             # harness, category "large" only
    "large_job_tk_max": 1_200_000,
    "large_job_tk_min": 100_000,
}

# Category plan for the default 300 jobs (scaled for other --n-jobs, each category >= 1).
PLAN = [
    ("conv", 26),          # 1 layer, KxK / KHxKW (K > 1) conv, no pool
    ("conv_pool", 30),     # 1 layer conv + fused 2x2/2 max pool
    ("pointwise", 20),     # 1 layer 1x1 conv on a spatial map
    ("fc", 24),            # 1-3 FC layers (1x1 maps, 1x1 kernels), final out_raw when OC <= 16
    ("head_raw", 20),      # 1 layer out_raw: global conv to 1x1, OC <= 16 (LOGIT)
    ("tails", 30),         # 1-2 layers, forced OC % 8 != 0 and OW % 8 != 0, narrow maps
    ("tiny", 18),          # 1-2 layers: minimal K (8..16) or 1x1 outputs (non-raw)
    ("depth2_3", 32),      # random chains of 2-3 layers
    ("depth4_6", 36),      # random chains of 4-6 layers
    ("depth7_8", 24),      # random chains of 7-8 layers
    ("netlike", 14),       # LeNet/CIFAR-like: conv+pool, conv+pool, conv->1x1, FC..., FC raw
    ("large", 12),         # 1-2 big layers (T*K 1e5..8e5 per layer; CIFAR-conv scale and above)
    ("refuse", 14),        # EXPECTED REFUSAL: checker must refuse (ERR_CODE from job_err_code)
]
CATEGORIES = [c for c, _ in PLAN]
XCHECK_PER_CATEGORY = 2                        # cheapest jobs per category (+ the median one) -> xsim set

KERNELS = np.array([1, 2, 3, 4, 5, 6, 7, 8])
KERNEL_P = np.array([0.18, 0.10, 0.25, 0.08, 0.20, 0.05, 0.08, 0.06])

MODEL_SOURCES = ("gos_pack.py", "gos_golden.py", "gos_cycle_model.py", "gos_tile_model.py",
                 "gos_addr_stream.py", "gos_fuzz.py", "requant_check.py", "final_layer.py")


class Reject(Exception):
    """Structure outside the envelope; the sampler redraws."""


# --------------------------------------------------------------------------- helpers
def plan_counts(n_jobs: int) -> dict[str, int]:
    tot = sum(c for _, c in PLAN)
    if n_jobs == tot:
        return dict(PLAN)
    return {k: max(1, round(c * n_jobs / tot)) for k, c in PLAN}


def ri(rng, lo: int, hi: int) -> int:
    """Uniform int in [lo, hi] (inclusive)."""
    return int(rng.integers(lo, hi + 1))


def pick_kernel(rng, maxk: int, mink: int = 1) -> int:
    ok = (KERNELS <= maxk) & (KERNELS >= mink)
    if not ok.any():
        raise Reject("no kernel fits")
    p = KERNEL_P * ok
    return int(rng.choice(KERNELS, p=p / p.sum()))


def layer(IC, IH, IW, OC, KH, KW, pool=0, relu=1, raw=0) -> dict:
    """Raw descriptor fields of one layer (bases / in_sel assigned later)."""
    return {"IC": int(IC), "OC": int(OC), "IH": int(IH), "IW": int(IW), "KH": int(KH),
            "KW": int(KW), "OH": int(IH) - int(KH) + 1, "OW": int(IW) - int(KW) + 1,
            "WGT_BASE": 0, "QP_BASE": 0, "relu_en": int(relu and not raw),
            "pool_en": int(pool and not raw), "out_raw": int(raw), "in_sel": 0}


def out_dims(f: dict) -> tuple[int, int, int]:
    if f["pool_en"]:
        return f["OC"], f["OH"] // 2, f["OW"] // 2
    return f["OC"], f["OH"], f["OW"]


def tk(f: dict) -> int:
    return gp.ceil_div(f["OC"], 8) * f["OH"] * gp.ceil_div(f["OW"], 8) * f["IC"] * f["KH"] * f["KW"]


def conv_step(rng, C, H, W, *, oc, p_pool=0.0, p_rect=0.25, kmin=1, relu_p=0.85,
              kmax=None) -> dict:
    """One random non-raw layer on a [C,H,W] input."""
    kmax_h = min(H, LIMITS["max_kh"], kmax or 99)
    kmax_w = min(W, LIMITS["max_kw"], kmax or 99)
    if rng.random() < p_rect:
        KH, KW = pick_kernel(rng, kmax_h, min(kmin, kmax_h)), pick_kernel(rng, kmax_w, min(kmin, kmax_w))
    else:
        KH = KW = pick_kernel(rng, min(kmax_h, kmax_w), min(kmin, kmax_h, kmax_w))
    OH, OW = H - KH + 1, W - KW + 1
    pool = int(OH % 2 == 0 and OW % 2 == 0 and OH >= 2 and OW >= 2 and rng.random() < p_pool)
    return layer(C, H, W, oc, KH, KW, pool=pool, relu=int(rng.random() < relu_p))


def raw_head(C, H, W, OC) -> dict:
    """Final out_raw layer: global conv to 1x1 (KH = H, KW = W), OC <= 16."""
    if H > LIMITS["max_kh"] or W > LIMITS["max_kw"] or OC > LIMITS["max_logits"]:
        raise Reject("raw head does not fit")
    return layer(C, H, W, OC, H, W, raw=1)


def chain(rng, n, C, H, W, *, oc_hi, p_pool, end_raw, kmin=1, p_rect=0.2) -> list[dict]:
    Ls = []
    for i in range(n):
        if i == n - 1 and end_raw:
            Ls.append(raw_head(C, H, W, ri(rng, 1, 16)))
            break
        f = conv_step(rng, C, H, W, oc=ri(rng, 1, oc_hi), p_pool=p_pool, p_rect=p_rect, kmin=kmin)
        Ls.append(f)
        C, H, W = out_dims(f)
    return Ls


# --------------------------------------------------------------------------- categories
def draw_structure(rng, cat: str, idx: int = 0) -> dict:
    """Layer fields (no bases) for one job of category `cat` (may raise Reject). `idx` = index of
    the job within its category (refuse jobs cycle through REFUSE_MODES with it)."""
    if cat == "conv":
        C, H, W = ri(rng, 1, 48), ri(rng, 2, 48), ri(rng, 2, 80)
        f = conv_step(rng, C, H, W, oc=ri(rng, 1, 96), kmin=2)
        if f["KH"] * f["KW"] < 2:
            raise Reject("conv needs K>1 kernel")
        return {"layers": [f]}
    if cat == "conv_pool":
        C, H, W = ri(rng, 1, 48), ri(rng, 3, 64), ri(rng, 3, 80)
        f = conv_step(rng, C, H, W, oc=ri(rng, 1, 96), p_pool=1.0, kmin=1)
        if not f["pool_en"]:
            raise Reject("pool needs even OH/OW")
        return {"layers": [f]}
    if cat == "pointwise":
        C, H, W = ri(rng, 8, 128), ri(rng, 1, 48), ri(rng, 2, 80)
        f = layer(C, H, W, ri(rng, 1, 128), 1, 1, relu=int(rng.random() < 0.8))
        f["pool_en"] = int(H % 2 == 0 and W % 2 == 0 and rng.random() < 0.3)
        return {"layers": [f]}
    if cat == "fc":
        n = ri(rng, 1, 3)
        C = ri(rng, 8, 512)
        Ls = []
        for i in range(n):
            last = i == n - 1
            raw = last and rng.random() < 0.6
            OC = ri(rng, 1, 16) if raw else ri(rng, 8 if not last else 1, 128)
            Ls.append(layer(C, 1, 1, OC, 1, 1, relu=int(rng.random() < 0.85), raw=int(raw)))
            C = OC
        return {"layers": Ls}
    if cat == "head_raw":
        KH, KW = ri(rng, 1, 8), ri(rng, 1, 8)
        lo = max(1, gp.ceil_div(8, KH * KW))
        C = ri(rng, lo, max(lo, min(128, LIMITS["max_k"] // (KH * KW))))
        return {"layers": [raw_head(C, KH, KW, ri(rng, 1, 16))]}
    if cat == "tails":
        n = ri(rng, 1, 2)
        C, H, W = ri(rng, 1, 32), ri(rng, 1, 24), ri(rng, 1, 23)
        Ls = []
        for _ in range(n):
            oc = int(rng.choice([x for x in range(1, 64) if x % 8]))
            f = conv_step(rng, C, H, W, oc=oc, p_pool=0.3)
            if f["OW"] % 8 == 0:
                raise Reject("tails: OW multiple of 8")
            Ls.append(f)
            C, H, W = out_dims(f)
        return {"layers": Ls}
    if cat == "tiny":
        n = ri(rng, 1, 2)
        Ls = []
        if rng.random() < 0.5:                         # minimal K (8..16)
            KH, KW = ri(rng, 1, 4), ri(rng, 1, 4)
            C = max(1, gp.ceil_div(ri(rng, 8, 16), KH * KW))
            H, W = ri(rng, KH, KH + 12), ri(rng, KW, KW + 20)
            f = layer(C, H, W, ri(rng, 1, 24), KH, KW, relu=int(rng.random() < 0.7))
        else:                                          # 1x1 output, non-raw
            H, W = ri(rng, 1, 8), ri(rng, 1, 8)
            C = ri(rng, max(1, gp.ceil_div(8, H * W)), 64)
            f = layer(C, H, W, ri(rng, 1, 40), H, W, relu=int(rng.random() < 0.7))
        Ls.append(f)
        if n == 2:
            C2, H2, W2 = out_dims(f)
            if H2 == 1 and W2 == 1 and rng.random() < 0.5 and C2 >= 8:
                Ls.append(raw_head(C2, 1, 1, ri(rng, 1, 16)))
            else:
                Ls.append(conv_step(rng, C2, H2, W2, oc=ri(rng, 1, 24)))
        return {"layers": Ls}
    if cat in ("depth2_3", "depth4_6", "depth7_8"):
        lo, hi, oc_hi, hw_hi = {"depth2_3": (2, 3, 96, 48), "depth4_6": (4, 6, 56, 48),
                                "depth7_8": (7, 8, 32, 40)}[cat]
        n = ri(rng, lo, hi)
        end_raw = rng.random() < 0.45
        C, H, W = ri(rng, 1, 32), ri(rng, 4, hw_hi), ri(rng, 4, hw_hi + 16)
        return {"layers": chain(rng, n, C, H, W, oc_hi=oc_hi, p_pool=0.35, end_raw=end_raw)}
    if cat == "netlike":
        C = int(rng.choice([1, 3, ri(rng, 1, 8)]))
        H = W = int(rng.choice([28, 32, ri(rng, 16, 40)]))
        Ls = []
        for _ in range(2):                             # conv + pool, twice
            k = int(rng.choice([3, 5]))
            if (H - k + 1) % 2:
                k += 1 if k + 1 <= min(H, 8) else -1
            f = layer(C, H, W, ri(rng, 4, 48), k, k, pool=1, relu=1)
            Ls.append(f)
            C, H, W = out_dims(f)
        if H > 8 or W > 8:
            raise Reject("netlike: map too large for the 1x1 conv")
        f = layer(C, H, W, ri(rng, 16, 96), H, W, relu=1)        # conv to 1x1 (conv5-like)
        Ls.append(f)
        C = f["OC"]
        for _ in range(ri(rng, 0, 2)):                            # hidden FCs
            f = layer(C, 1, 1, ri(rng, 16, 96), 1, 1, relu=1)
            Ls.append(f)
            C = f["OC"]
        Ls.append(raw_head(C, 1, 1, ri(rng, 2, 16)))            # classifier (LOGIT)
        return {"layers": Ls}
    if cat == "large":
        C, H, W = ri(rng, 8, 64), ri(rng, 20, 64), ri(rng, 20, 64)
        k = int(rng.choice([3, 5]))
        f = layer(C, H, W, ri(rng, 24, 128), k, k, relu=1)
        f["pool_en"] = int(f["OH"] % 2 == 0 and f["OW"] % 2 == 0 and rng.random() < 0.6)
        Ls = [f]
        if rng.random() < 0.4:
            C2, H2, W2 = out_dims(f)
            Ls.append(conv_step(rng, C2, H2, W2, oc=ri(rng, 16, 96), p_pool=0.5, kmin=3))
        return {"layers": Ls, "large": True}
    if cat == "refuse":
        return draw_refuse(rng, REFUSE_MODES[idx % len(REFUSE_MODES)])
    raise ValueError(cat)


# refuse mode -> the checker rule id that must be reported first (FORMATS.md section 5)
REFUSE_RULE = {"pool_odd_OH": 1, "pool_odd_OW": 2, "K_lt_8": 3, "WGT_END": 4, "IN_END": 5,
               "OUT_END": 6, "QP_END": 7, "raw_OC_gt_16": 27, "n_layers_0": gp.RULE_N_LAYERS,
               "n_layers_9": gp.RULE_N_LAYERS}
REFUSE_MODES = tuple(REFUSE_RULE)


def draw_refuse(rng, mode: str) -> dict:
    """A job whose descriptors pass the host-side field checks (derive/encode consistent)
    but that the RTL config checker must refuse. The bad layer is the last one of a short
    valid chain (so ERR_CODE carries a layer index > 0 for n > 1), or N_LAYERS is 0 / 9."""
    n = ri(rng, 1, 3)
    C, H, W = ri(rng, 1, 16), ri(rng, 8, 24), ri(rng, 8, 24)
    pre = chain(rng, n - 1, C, H, W, oc_hi=24, p_pool=0.3, end_raw=False) if n > 1 else []
    if pre:
        C, H, W = out_dims(pre[-1])
    ref = {"mode": mode, "n_layers_override": None, "wgt_base_override": None,
           "qp_base_override": None}
    if mode == "pool_odd_OH":
        KH = 1 if (H % 2 == 1) else 2 if H >= 2 else 1               # OH = H-KH+1 odd
        if (H - KH + 1) % 2 == 0:
            raise Reject("pool_odd_OH")
        bad = layer(C, H, W, ri(rng, 1, 16), KH, 1, pool=1)
        bad["pool_en"] = 1
    elif mode == "pool_odd_OW":
        KH = 1 if H % 2 == 0 else 2
        KW = 1 if W % 2 == 1 else 2
        bad = layer(C, H, W, ri(rng, 1, 16), KH, KW, pool=1)
        bad["pool_en"] = 1
        if bad["OH"] % 2 or bad["OW"] % 2 == 0:
            raise Reject("pool_odd_OW")
    elif mode == "K_lt_8":
        pre = []
        KH, KW = int(rng.choice([1, 2])), int(rng.choice([1, 2, 3]))
        C = ri(rng, 1, max(1, 7 // (KH * KW)))
        if C * KH * KW >= 8:
            raise Reject("K_lt_8")
        bad = layer(C, ri(rng, 4, 16), ri(rng, 4, 16), ri(rng, 1, 16), KH, KW)
    elif mode == "WGT_END":
        bad = conv_step(rng, C, H, W, oc=ri(rng, 1, 16))
        words = gp.ceil_div(bad["OC"], 8) * bad["IC"] * bad["KH"] * bad["KW"]
        ref["wgt_base_override"] = gp.WGT_DEPTH - ri(rng, 0, max(0, words - 1))
    elif mode == "IN_END":
        pre = []
        C, W = ri(rng, 8, 32), ri(rng, 57, 64)                     # IC*IH*8 > 4096
        Hh = gp.ceil_div(4097, C * 8) + ri(rng, 0, 4)
        bad = layer(C, Hh, W, ri(rng, 1, 8), 1, 1)
    elif mode == "OUT_END":
        pre = []
        C, H, W = ri(rng, 1, 4), ri(rng, 40, 64), ri(rng, 57, 64)
        OC = gp.ceil_div(4097, H * 8) + ri(rng, 0, 8)
        bad = layer(C, H, W, OC, 1, 1)
        bad["IC"] = max(8, C)
    elif mode == "QP_END":
        bad = conv_step(rng, C, H, W, oc=ri(rng, 1, 32))
        ref["qp_base_override"] = gp.QP_CHANNELS - ri(rng, 1, bad["OC"]) if bad["OC"] > 1 else gp.QP_CHANNELS
    elif mode == "raw_OC_gt_16":
        if H > 8 or W > 8:
            raise Reject("raw_OC_gt_16 needs a map <= 8x8")
        bad = layer(C, H, W, ri(rng, 17, 64), H, W, raw=1)
    else:                                                    # n_layers 0 / 9: valid descriptors
        bad = conv_step(rng, C, H, W, oc=ri(rng, 1, 16))
        ref["n_layers_override"] = 0 if mode == "n_layers_0" else 9
    return {"layers": pre + [bad], "refuse": ref}


# --------------------------------------------------------------------------- structure checks
def finalize_structure(rng, s: dict) -> dict:
    """Assign bases / in_sel, derive, encode, check. Valid jobs must satisfy every envelope
    limit and be accepted by the checker; refuse jobs must pass the host-side encode and be
    refused with job_err_code != 0."""
    Ls = [dict(f) for f in s["layers"]]
    refuse = s.get("refuse")
    n = len(Ls)
    if not 1 <= n <= LIMITS["max_layers"]:
        raise Reject("n_layers")
    for i, f in enumerate(Ls):
        f["in_sel"] = i % 2
        if f["out_raw"]:
            f["relu_en"] = f["pool_en"] = 0
        f.update(gp.derive_fields(f))
    # chaining (the output map of layer i is the input of layer i+1)
    for a, b in zip(Ls, Ls[1:]):
        if (b["IC"], b["IH"], b["IW"]) != (a["OC"], a["OUT_H"], a["OUT_W"]):
            raise Reject("chain")
        if a["out_raw"]:
            raise Reject("out_raw not last")
    wgt_words = [f["OC_TILES"] * f["K"] for f in Ls]
    qp_ch = [f["OC"] for f in Ls]
    if refuse is None:
        for f in Ls:
            if f["K"] < LIMITS["min_k"] or f["K"] > LIMITS["max_k"]:
                raise Reject("K")
            if f["KW"] > LIMITS["max_kw"] or f["KH"] > LIMITS["max_kh"]:
                raise Reject("kernel")
            if f["IH"] > LIMITS["max_hw"] or f["IW"] > LIMITS["max_hw"]:
                raise Reject("H/W")
            if f["IN_END"] + 1 > LIMITS["act_depth"] or (not f["out_raw"] and f["OUT_END"] + 1 > LIMITS["act_depth"]):
                raise Reject("ACT depth")
            if f["pool_en"] and (f["OH"] % 2 or f["OW"] % 2):
                raise Reject("pool parity")
            if f["out_raw"] and (f["OC"] > LIMITS["max_logits"] or f["OH"] != 1 or f["OW"] != 1):
                raise Reject("raw")
            if tk(f) > LIMITS["large_layer_tk_max" if s.get("large") else "layer_tk_max"]:
                raise Reject("layer T*K")
        job_tk = sum(tk(f) for f in Ls)
        if job_tk > LIMITS["large_job_tk_max" if s.get("large") else "job_tk_max"]:
            raise Reject("job T*K")
        if s.get("large") and job_tk < LIMITS["large_job_tk_min"]:
            raise Reject("large: T*K too small")
        if sum(wgt_words) > LIMITS["wgt_depth"] or sum(qp_ch) > LIMITS["qp_channels"]:
            raise Reject("WGT / QPARAM")
        wgt0 = 0 if rng.random() < 0.4 else ri(rng, 0, LIMITS["wgt_depth"] - sum(wgt_words))
        qp0 = 0 if rng.random() < 0.4 else ri(rng, 0, LIMITS["qp_channels"] - sum(qp_ch))
    else:
        wgt0 = qp0 = 0
    acc_w, acc_q = wgt0, qp0
    for i, f in enumerate(Ls):
        f["WGT_BASE"], f["QP_BASE"] = acc_w, acc_q
        acc_w += wgt_words[i]
        acc_q += qp_ch[i]
    if refuse is not None:
        if refuse["wgt_base_override"] is not None:
            Ls[-1]["WGT_BASE"] = int(refuse["wgt_base_override"])
        if refuse["qp_base_override"] is not None:
            Ls[-1]["QP_BASE"] = int(refuse["qp_base_override"])
    words = []
    for f in Ls:
        f.update(gp.derive_fields(f))
        try:
            words.append(gp.encode_descriptor(f))      # host-side field widths + consistency
        except AssertionError as e:
            raise Reject(f"encode: {e}") from None
    words = np.stack(words).astype(np.uint32)
    n_layers = n if refuse is None or refuse["n_layers_override"] is None else refuse["n_layers_override"]
    if refuse is None:
        for w in words:
            ok, why = gp.check_descriptor(w)
            assert ok, f"checker rejects a generated valid descriptor: {why}"
        err = gp.job_err_code(n_layers, words)
        assert err == 0
    else:
        full = np.zeros((gp.MAX_LAYERS, gp.DESC_WORDS), dtype=np.uint32)
        full[:len(words)] = words
        err = gp.job_err_code(n_layers, full)
        n_ok = n if refuse["n_layers_override"] is not None else n - 1
        if not all(gp.check_descriptor(w)[0] for w in words[:n_ok]):
            raise Reject("refuse job: a layer other than the bad one fails")
        exp_layer = 0 if refuse["n_layers_override"] is not None else n - 1
        if err == 0 or (err >> 8, err & 7) != (REFUSE_RULE[refuse["mode"]], exp_layer):
            raise Reject("refuse job: first failing rule/layer differs from the mode")
    return {"layers": Ls, "words": words, "n_layers": int(n_layers), "err_code": int(err),
            "refuse": refuse}


# --------------------------------------------------------------------------- data + golden
def golden_L(f: dict, name: str) -> dict:
    return {"name": name, "IC": f["IC"], "OC": f["OC"], "KH": f["KH"], "KW": f["KW"],
            "IH": f["IH"], "IW": f["IW"], "OH": f["OH"], "OW": f["OW"], "K": f["K"],
            "relu": bool(f["relu_en"]), "pool": bool(f["pool_en"]), "final": bool(f["out_raw"])}


def make_layer_params(rng, f: dict, x: np.ndarray) -> dict:
    """Random int8 weights; q_bias and per-channel (m, s) calibrated on this layer's own
    accumulator statistics so outputs are neither all-saturated nor all-zero."""
    q_w = rng.integers(-128, 128, size=(f["OC"], f["IC"], f["KH"], f["KW"])).astype(np.int8)
    acc = gg.gos_conv_acc(x[None], q_w)[0]
    sd = float(acc.std()) + 1.0
    OC = f["OC"]
    q_b = np.round(rng.uniform(-1.5, 1.5, size=OC) * sd).astype(np.int64)
    if f["out_raw"]:
        q_b = np.clip(q_b, -(2**30), 2**30).astype(np.int32)
        return {"q_w": q_w, "q_b": q_b, "m": [0] * OC, "s": [0] * OC}
    qb_max = 2 ** (gp.V_MUL_W - 1) - 1 - f["K"] * gp.ACT_PROD_MAX       # V_MUL_W bound
    assert qb_max > 0, f"K={f['K']} too large for V_MUL_W"
    q_b = np.clip(q_b, -qb_max, qb_max).astype(np.int32)
    sdv = float((acc + q_b.astype(np.int64)[:, None, None]).std()) + 1.0
    m, s = [], []
    for _ in range(OC):
        M = 10 ** rng.uniform(-0.45, 0.45) * 48.0 / sdv
        try:
            mi, si = select_m_s(M, B)
        except ShiftOutOfRange:
            mi, si = select_m_s(min(max(M, 2.0 ** -30), 2.0 ** 30), B)
        m.append(mi)
        s.append(si)
    assert all(1 <= x_ <= 63 for x_ in s) and all((1 << (B - 1)) <= x_ < (1 << B) for x_ in m)
    assert gp.v_abs_bound({"K": f["K"]}, q_b) < 2 ** (gp.V_MUL_W - 1)
    return {"q_w": q_w, "q_b": q_b, "m": m, "s": s}


def run_golden(rng, st: dict) -> dict:
    """Input image, per-layer params, golden output of the whole chain (gos_golden)."""
    L0 = st["layers"][0]
    shape = (L0["IC"], L0["IH"], L0["IW"])
    if rng.random() < 0.3:
        x0 = rng.integers(0, 128, size=shape).astype(np.int8)      # ReLU-like input
    else:
        x0 = rng.integers(-128, 128, size=shape).astype(np.int8)
    x = x0
    params, stats = [], []
    for i, f in enumerate(st["layers"]):
        p = make_layer_params(rng, f, x)
        Lg = golden_L(f, f"L{i}")
        if f["out_raw"]:
            P = gg.LayerParams(Lg, p["q_w"], p["q_b"])
        else:
            P = gg.LayerParams(Lg, p["q_w"], p["q_b"], m=tuple(p["m"]), s=tuple(p["s"]))
        y = gg.gos_layer(x, Lg, P)
        params.append(p)
        if not f["out_raw"]:
            stats.append({"frac_zero": float((y == 0).mean()),
                          "frac_sat": float(((y == 127) | (y == -128)).mean())})
        x = y
    return {"x0": x0, "params": params, "y": x, "stats": stats}


def tile_model_check(st: dict, g: dict, rng) -> None:
    """Independent address-level check: gos_tile_model on garbage memories == golden."""
    mem = gos_fuzz.garbage_mem(rng)
    L0 = st["layers"][0]
    cov = gos_fuzz.map_cover(L0["IC"], L0["IH"], L0["IW"])
    mem.act[0] = np.where(cov, gp.pack_act(g["x0"]), mem.act[0])
    for f, p in zip(st["layers"], g["params"]):
        tm.place_layer_params(mem, f, p["q_w"], p["q_b"], p["m"], p["s"])
    for w in st["words"]:
        tm.run_layer_mem(mem, w, "fast")
    fl = st["layers"][-1]
    if fl["out_raw"]:
        assert np.array_equal(mem.logit[:fl["OC"]], g["y"][:, 0, 0]), "tile model LOGIT != golden"
    else:
        got = gp.unpack_act(mem.act[1 - fl["in_sel"]], fl["OC"], fl["OUT_H"], fl["OUT_W"])
        assert np.array_equal(got, g["y"]), "tile model output != golden"


def model_cycles(layers: list[dict]) -> dict:
    Ls = [{"name": f"L{i}", **{k: f[k] for k in ("IC", "OC", "OH", "OW", "K")}}
          for i, f in enumerate(layers)]
    r = cm.net_cycles(Ls)
    assert None not in (cm.C_PIPE, cm.C_START, cm.C_DONE)
    return {"layer": [int(r["layers"][L["name"]]["cycles"]) for L in Ls],
            "T": [int(r["layers"][L["name"]]["T"]) for L in Ls],
            "total": int(r["total"]["cycles"]), "mac": int(r["total"]["mac_active"])}


def shape_str(f: dict) -> str:
    s = f"{f['IC']}x{f['IH']}x{f['IW']}-k{f['KH']}x{f['KW']}->{f['OC']}"
    if f["pool_en"]:
        s += "p"
    if f["out_raw"]:
        s += "raw"
    elif not f["relu_en"]:
        s += "n"
    return s


def be_bits(cover_words: np.ndarray) -> np.ndarray:
    """bool [8, n] -> per-word byte-enable bits (uint64 values 0..255)."""
    return (cover_words.astype(np.uint64) << np.arange(8, dtype=np.uint64)[:, None]).sum(
        axis=0, dtype=np.uint64)


def build_job(rng, job_id: int, cat: str, st: dict, tile_check: bool, rng_tile) -> dict:
    Ls, words = st["layers"], st["words"]
    L0, fl = Ls[0], Ls[-1]
    arr: dict = {"job": np.int64(job_id), "category": np.str_(cat), "n_layers": np.int64(st["n_layers"]),
                 "desc": words.astype(np.uint32), "expect_refuse": np.int64(st["refuse"] is not None),
                 "err_code": np.int64(st["err_code"]),
                 "refuse_mode": np.str_(st["refuse"]["mode"] if st["refuse"] else ""),
                 "shape": np.str_(";".join(shape_str(f) for f in Ls)),
                 "fields": np.array([[f[k] for k in gp.ALL_FIELDS] for f in Ls], dtype=np.int64),
                 "field_names": np.array(gp.ALL_FIELDS)}
    empty64 = np.zeros(0, dtype=np.uint64)
    if st["refuse"] is not None:
        # Refused job: busy only for the config check (S_CHK1..S_CHK3 = C_START, gos_cycle_model),
        # then idle (C_DONE); no layer runs, so MAC_ACTIVE = 0 and no LAYER_CYC is defined.
        arr.update(wgt=empty64, wgt_base=np.int64(0), qparam=empty64, qp_base=np.int64(0),
                   act_in=empty64, out_buf=np.int64(-1), out_raw=np.int64(0), out_expected=empty64,
                   out_mask=empty64, logits_expected=np.zeros(gp.N_LOGITS, dtype=np.int32),
                   n_logits=np.int64(0), model_layer_cycles=np.zeros(0, dtype=np.int64),
                   model_T=np.zeros(0, dtype=np.int64), model_total=np.int64(cm.C_START + cm.C_DONE),
                   model_mac=np.int64(0), tk_total=np.int64(0), macs=np.int64(0))
        return arr
    g = run_golden(rng, st)
    if tile_check:
        tile_model_check(st, g, rng_tile)
    wgt = np.concatenate([gp._pack_wgt_layer(p["q_w"]) for p in g["params"]])
    qp = np.concatenate([tm.pack_layer_qparam(p["q_b"], p["m"], p["s"]) for p in g["params"]])
    assert wgt.size == sum(f["OC_TILES"] * f["K"] for f in Ls)
    assert qp.size == 2 * sum(f["OC"] for f in Ls)
    act_in = gp.pack_act_words(g["x0"], L0["IN_END"] + 1)
    mc = model_cycles(Ls)
    logits = np.zeros(gp.N_LOGITS, dtype=np.int32)
    if fl["out_raw"]:
        logits[:fl["OC"]] = g["y"][:, 0, 0]
        out_exp, out_mask, out_buf = empty64, empty64, 1 - fl["in_sel"]
    else:
        n_out = fl["OUT_END"] + 1
        out_exp = gp.pack_act_words(g["y"], n_out)
        cov = gos_fuzz.map_cover(fl["OC"], fl["OUT_H"], fl["OUT_W"])[:, :n_out]
        out_mask = be_bits(cov)
        out_buf = 1 - fl["in_sel"]
        assert int(np.count_nonzero(cov)) == fl["OC"] * fl["OUT_H"] * fl["OUT_W"]
    arr.update(wgt=wgt, wgt_base=np.int64(L0["WGT_BASE"]), qparam=qp, qp_base=np.int64(L0["QP_BASE"]),
               act_in=act_in, out_buf=np.int64(out_buf), out_raw=np.int64(fl["out_raw"]),
               out_expected=out_exp, out_mask=out_mask, logits_expected=logits,
               n_logits=np.int64(fl["OC"] if fl["out_raw"] else 0),
               model_layer_cycles=np.array(mc["layer"], dtype=np.int64),
               model_T=np.array(mc["T"], dtype=np.int64), model_total=np.int64(mc["total"]),
               model_mac=np.int64(mc["mac"]), tk_total=np.int64(sum(tk(f) for f in Ls)),
               macs=np.int64(sum(f["OC"] * f["OH"] * f["OW"] * f["K"] for f in Ls)))
    arr["_stats"] = g["stats"]
    return arr


# --------------------------------------------------------------------------- output files
def save_npz_det(path: Path, arrays: dict) -> None:
    """Deterministic npz (fixed zip timestamps, sorted members): same content -> same bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[k]), allow_pickle=False)
            zi = zipfile.ZipInfo(f"{k}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def write_hex_job(hdir: Path, a: dict) -> None:
    """$readmemh files of one job for tb_shapes.sv (formats: FORMATS.md section 7)."""
    hdir.mkdir(parents=True, exist_ok=True)
    nl_desc = int(a["desc"].shape[0])
    cyc = [int(x) for x in a["model_layer_cycles"]]
    meta = [int(a["n_layers"]), nl_desc, int(a["expect_refuse"]), int(a["err_code"]),
            int(a["wgt_base"]), int(a["wgt"].size), 2 * int(a["qp_base"]), int(a["qparam"].size),
            int(a["act_in"].size), int(a["out_raw"]), int(max(a["out_buf"], 0)),
            int(a["out_expected"].size), int(a["n_logits"]), len(cyc),
            int(a["model_total"]), int(a["model_mac"])]
    gp.write_hex(hdir / "meta.hex", meta, 32)
    gp.write_hex(hdir / "desc.hex", a["desc"].reshape(-1), 32)
    if a["expect_refuse"]:
        return
    gp.write_hex(hdir / "wgt.hex", a["wgt"], 64)
    gp.write_hex(hdir / "qparam.hex", a["qparam"], 64)
    gp.write_hex(hdir / "act_in.hex", a["act_in"], 64)
    gp.write_hex(hdir / "exp_cyc.hex", cyc + [int(a["model_total"]), int(a["model_mac"])], 32)
    if a["out_raw"]:
        gp.write_hex(hdir / "logit.hex", a["logits_expected"].view(np.uint32), 32)
    else:
        gp.write_hex(hdir / "out_exp.hex", a["out_expected"], 64)
        gp.write_hex(hdir / "out_mask.hex", a["out_mask"], 8)


def model_sources_sha256() -> str:
    h = hashlib.sha256()
    for n in MODEL_SOURCES:
        h.update(n.encode() + b"\0" + (MODEL_DIR / n).read_bytes())
    return h.hexdigest()


def shapeset_sha256_of(files: dict) -> str:
    """Identity of a shape set: sha256 over the sorted job-file sha256 list (npz content is
    deterministic, so the same seed + model code gives the same value)."""
    return sha256_bytes(json.dumps(sorted((k, v["sha256"]) for k, v in files.items() if k.startswith("jobs/")),
                                   separators=(",", ":")).encode())


# --------------------------------------------------------------------------- driver
def draw_valid(rng, cat: str, idx: int = 0, max_tries: int = 20000) -> tuple[dict, int]:
    for t in range(1, max_tries + 1):
        try:
            return finalize_structure(rng, draw_structure(rng, cat, idx)), t
        except Reject:
            continue
    raise RuntimeError(f"category {cat}: no valid structure in {max_tries} tries")


def generate(seed: int = DEFAULT_SEED, n_jobs: int | None = None, out: Path | None = None,
             tile_check: bool = True, hex_files: bool = True, log=print) -> dict:
    t0 = time.time()
    counts = plan_counts(n_jobs if n_jobs is not None else sum(c for _, c in PLAN))
    out = Path(out) if out is not None else DEFAULT_OUT / str(seed)
    rng = np.random.default_rng([seed, 1])               # structures + data
    rng_order = np.random.default_rng([seed, 2])         # job order
    rng_tile = np.random.default_rng([seed, 3])          # garbage memories of the tile check
    cats = [c for c in CATEGORIES for _ in range(counts[c])]
    order = rng_order.permutation(len(cats))
    cats = [cats[i] for i in order]                     # interleaved: contiguous shards mix categories
    jobs_dir = out / "jobs"
    for d in (jobs_dir, out / "hex"):
        if d.exists():
            shutil.rmtree(d)
    if (out / "shapes.json").exists():
        (out / "shapes.json").unlink()
    files, summary, tries = {}, [], {}
    zero_frac = []
    seen: dict[str, int] = {}
    for j, cat in enumerate(cats):
        st, t = draw_valid(rng, cat, seen.get(cat, 0))
        seen[cat] = seen.get(cat, 0) + 1
        tries[cat] = tries.get(cat, 0) + t
        a = build_job(rng, j, cat, st, tile_check, rng_tile)
        stats = a.pop("_stats", [])
        zero_frac += [s["frac_zero"] for s in stats]
        name = f"J{j:04d}"
        p = jobs_dir / f"{name}.npz"
        save_npz_det(p, a)
        files[f"jobs/{name}.npz"] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        if hex_files:
            write_hex_job(out / "hex" / name, a)
        Ls = st["layers"]
        summary.append({
            "job": j, "name": name, "category": cat, "n_layers": int(a["n_layers"]),
            "n_desc": int(a["desc"].shape[0]), "expect_refuse": int(a["expect_refuse"]),
            "err_code": int(a["err_code"]), "refuse_mode": str(a["refuse_mode"]),
            "shape": str(a["shape"]), "out_raw": int(a["out_raw"]),
            "has_pool": int(any(f["pool_en"] for f in Ls)),
            "has_oc_tail": int(any(f["OC"] % 8 for f in Ls)),
            "has_x_tail": int(any(f["OW"] % 8 for f in Ls)),
            "has_1x1_out": int(any(f["OH"] == 1 and f["OW"] == 1 for f in Ls)),
            "K": [int(f["K"]) for f in Ls], "T": [int(x) for x in a["model_T"]],
            "tk_total": int(a["tk_total"]), "macs": int(a["macs"]),
            "model_layer_cycles": [int(x) for x in a["model_layer_cycles"]],
            "model_total": int(a["model_total"]), "model_mac": int(a["model_mac"]),
            "wgt_words": int(a["wgt"].size), "qp_channels": int(a["qparam"].size) // 2,
            "act_in_words": int(a["act_in"].size), "out_words": int(a["out_expected"].size),
            "file": f"jobs/{name}.npz"})
        if (j + 1) % 50 == 0:
            log(f"  {j + 1}/{len(cats)} jobs ({time.time() - t0:.1f} s)")
    # xsim cross-check subset: per category the XCHECK_PER_CATEGORY cheapest jobs + the median job
    xcheck = set()
    for c in CATEGORIES:
        js = sorted((s for s in summary if s["category"] == c), key=lambda s: (s["tk_total"], s["job"]))
        xcheck |= {s["job"] for s in js[:XCHECK_PER_CATEGORY]}
        if js:
            xcheck.add(js[len(js) // 2]["job"])
    xcheck = sorted(xcheck)
    for s in summary:
        s["xcheck"] = int(s["job"] in xcheck)
    if hex_files:
        gp.write_hex(out / "hex" / "jobs.hex", [len(summary)], 32)
        gp.write_hex(out / "hex" / "xcheck.hex", [len(xcheck)] + xcheck, 32)
    man = {
        "format": FORMAT, "seed": seed, "n_jobs": len(summary),
        "generator": GENERATOR, "generator_sha256": sha256_file(Path(__file__)),
        "model_sources_sha256": model_sources_sha256(),
        "git_commit": common.git_commit(), "git_dirty": common.git_dirty(),
        "label": "model (gos_golden / gos_pack / gos_cycle_model); not RTL, not hardware",
        "limits": LIMITS, "plan": counts,
        "categories": {c: sum(s["category"] == c for s in summary) for c in CATEGORIES},
        "coverage": {k: sum(s[k] for s in summary if not s["expect_refuse"])
                     for k in ("out_raw", "has_pool", "has_oc_tail", "has_x_tail", "has_1x1_out")},
        "n_layers_hist": {str(n): sum(s["n_layers"] == n and not s["expect_refuse"] for s in summary)
                          for n in range(1, 9)},
        "refuse_modes": sorted({s["refuse_mode"] for s in summary if s["expect_refuse"]}),
        "sampler_tries": tries, "tile_model_checked": tile_check,
        "mean_frac_zero_outputs": float(np.mean(zero_frac)) if zero_frac else None,
        "model_constants": {"C_PIPE": cm.C_PIPE, "C_START": cm.C_START, "C_DONE": cm.C_DONE},
        "xcheck_jobs": xcheck,
        "total_model_cycles": int(sum(s["model_total"] for s in summary)),
        "files": files,
        "jobs": summary,
    }
    man["shapeset_sha256"] = shapeset_sha256_of(files)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shapes.json").write_text(json.dumps(man, indent=1) + "\n")
    log(f"gen_shapes: {len(summary)} jobs (seed {seed}) -> {out}  shapeset_sha256 "
        f"{man['shapeset_sha256'][:16]}  ({time.time() - t0:.1f} s) [model]")
    return man


def is_current(out: Path, seed: int, n_jobs: int | None) -> tuple[bool, str]:
    """True if out/shapes.json was made by this generator + model code at this commit / dirty
    state for (seed, n_jobs) and its files verify (run_shapes.sh regenerates otherwise)."""
    mp = Path(out) / "shapes.json"
    if not mp.is_file():
        return False, "no shapes.json"
    try:
        m = json.loads(mp.read_text())
    except ValueError:
        return False, "unreadable shapes.json"
    want_n = sum(plan_counts(n_jobs if n_jobs is not None else sum(c for _, c in PLAN)).values())
    checks = {"format": (m.get("format"), FORMAT), "seed": (m.get("seed"), seed),
              "n_jobs": (m.get("n_jobs"), want_n),
              "generator_sha256": (m.get("generator_sha256"), sha256_file(Path(__file__))),
              "model_sources_sha256": (m.get("model_sources_sha256"), model_sources_sha256()),
              "git_commit": (m.get("git_commit"), common.git_commit()),
              "git_dirty": (m.get("git_dirty"), common.git_dirty())}
    for k, (have, want) in checks.items():
        if have != want:
            return False, f"{k} differs"
    sys.path.insert(0, str(V2 / "board"))
    import shapeset  # noqa: PLC0415
    try:
        shapeset.verify_manifest(out)
    except shapeset.ShapesetError as e:
        return False, str(e)
    if not (Path(out) / "hex" / "jobs.hex").is_file():
        return False, "no hex/"
    return True, "current"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--n-jobs", type=int, default=None, help="default: the PLAN total (300)")
    ap.add_argument("--out", default=None, help="default v2/build/shapes/<seed>")
    ap.add_argument("--no-tile-check", action="store_true")
    ap.add_argument("--if-stale", action="store_true",
                    help="generate only if the set is missing or not current (is_current)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else DEFAULT_OUT / str(a.seed)
    if a.if_stale:
        ok, why = is_current(out, a.seed, a.n_jobs)
        if ok:
            print(f"gen_shapes: {out} is current (seed {a.seed})")
            return 0
        print(f"gen_shapes: regenerating {out}: {why}")
    m = generate(a.seed, a.n_jobs, out, tile_check=not a.no_tile_check)
    print(json.dumps({k: m[k] for k in ("n_jobs", "categories", "coverage", "n_layers_hist",
                                        "refuse_modes", "total_model_cycles", "shapeset_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
