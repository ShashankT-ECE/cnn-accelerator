"""V3 on-chip storage estimate (Phase 0 DSE; every number it produces is labelled "estimate").

Given a layer table (v3/model/nets.layer_table) and an ArrayConfig (v3/model/cycle_model.py):
  * weights: OS layout as V2 (one C-byte word = C output channels at one k, OC padded to C, packed
    oc_tile -> k per layer, all layers in one memory); dw layers: one word per (oc_tile, kh, kw);
    C1: only nonzero k-steps stored, plus index metadata (min of a K-bit mask or a ceil(log2 K)-bit
    index per nonzero k-step, per OC tile).
  * activations: tensors stored R-bank interleaved (bank = x mod R for 'x', flat index mod R for 'xy'),
    each row (resp. each map) padded to a multiple of R. Liveness: a tensor lives from its producer until
    its last consumer (main-path input or residual skip). Buffers: 2 (ping-pong) + 1 skip buffer when the
    net has residual adds (3-buffer rotation), each sized to the largest stored tensor - an assumption.
  * qparams: qparam_bits_per_oc per output channel (default 128 = V2 FORMATS 2 x 64-bit words).
  * block counts: the cheapest aspect ratio of each primitive (BRAM36 TDP aspects up to 36 bits, 72 in SDP;
    URAM288 = 72 x 4096). Capacity only; ports, banking granularity and cascades are not modelled beyond
    one memory per bank.
Device totals come only from v3/results/device_resources.csv (Vivado part query); never typed by hand.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Sequence

from cycle_model import ArrayConfig, ceil_div, log2_ceil

# (width_bits, depth) aspect ratios
PRIMS = {
    "BRAM36": ((1, 32768), (2, 16384), (4, 8192), (9, 4096), (18, 2048), (36, 1024), (72, 512)),
    "BRAM18": ((1, 16384), (2, 8192), (4, 4096), (9, 2048), (18, 1024), (36, 512)),
    "URAM288": ((72, 4096),),
}
QPARAM_BITS_PER_OC = 128     # V2 FORMATS: 2 x 64-bit words per output channel (copied layout, not a V3 decision)
DEVICE_CSV = Path(__file__).resolve().parents[1] / "results" / "device_resources.csv"


def blocks(width_bits: int, depth: int, prim: str) -> int:
    """Primitives of type `prim` needed for a width_bits x depth memory (capacity, best aspect)."""
    if width_bits <= 0 or depth <= 0:
        return 0
    return min(ceil_div(width_bits, w) * ceil_div(depth, d) for w, d in PRIMS[prim])


# ---------------------------------------------------------------- weights
def weight_storage(layers: Sequence[dict], cfg: ArrayConfig, density: float | dict = 1.0,
                   masks: dict | None = None) -> dict:
    per, words_total, meta_bits = [], 0, 0
    for L in layers:
        C = cfg.C
        n_oc = ceil_div(int(L["oc"]), C)
        if L["kind"] == "dw":
            K = int(L["kh"]) * int(L["kw"])
            nnz = [K] * n_oc
        else:
            K = int(L["k"])
            d = density.get(L["layer"], 1.0) if isinstance(density, dict) else density
            if masks and L["layer"] in masks:
                nnz = list(masks[L["layer"]])
            else:
                nnz = [K if d >= 1.0 else math.ceil(d * K)] * n_oc
        words = sum(nnz)
        sparse = any(z < K for z in nnz)
        mbits = sum(min(K, z * log2_ceil(K)) for z in nnz) if sparse else 0
        per.append({"layer": L["layer"], "raw_bytes": int(L["params"]), "words": words,
                    "padded_bytes": words * C, "meta_bits": mbits})
        words_total += words
        meta_bits += mbits
    width = 8 * cfg.C
    return {"layers": per, "raw_bytes": sum(p["raw_bytes"] for p in per), "words": words_total,
            "word_bits": width, "padded_bytes": words_total * cfg.C, "meta_bits": meta_bits,
            "uram": blocks(width, words_total, "URAM288"), "bram36": blocks(width, words_total, "BRAM36"),
            "meta_bram36": blocks(36, ceil_div(meta_bits, 36), "BRAM36")}


# ---------------------------------------------------------------- activations
def tensor_bytes(ch: int, h: int, w: int, cfg: ArrayConfig) -> int:
    """Stored bytes of a (ch, h, w) INT8 tensor in the R-bank layout of cfg.row_map."""
    if h * w == 1:
        return ceil_div(ch, cfg.R) * cfg.R     # 1x1 maps: channels interleaved over banks
    if cfg.row_map == "xy":
        return ch * ceil_div(h * w, cfg.R) * cfg.R
    return ch * h * ceil_div(w, cfg.R) * cfg.R


def _input_from(layers: Sequence[dict]) -> list[str]:
    """Producer of each layer's main input ('input' = image). A layer that only feeds a residual add
    (referenced by residual_from, relu == 0, i.e. a projection shortcut) passes its own input on."""
    skip_only = {L["residual_from"] for L in layers if L.get("residual_from")}
    src, prev = [], "input"
    for i, L in enumerate(layers):
        src.append(prev)
        if not (L["layer"] in skip_only and int(L.get("relu", 1)) == 0):
            prev = L["layer"]
    return src


def act_storage(layers: Sequence[dict], cfg: ArrayConfig, input_shape=(3, 32, 32)) -> dict:
    srcs = _input_from(layers)
    size = {"input": tensor_bytes(*input_shape, cfg)}
    produced = {"input": -1}
    last_use: dict[str, int] = {}
    for i, (L, s) in enumerate(zip(layers, srcs)):
        if L.get("gap") and cfg.gap == "fused":
            out = tensor_bytes(int(L["oc"]), 1, 1, cfg)              # only the pooled vector is stored
        else:
            out = tensor_bytes(int(L["oc"]), int(L["oh"]), int(L["ow"]), cfg)
        size[L["layer"]] = out
        produced[L["layer"]] = i
        last_use[s] = max(last_use.get(s, -1), i)
        if L.get("residual_from"):
            last_use[L["residual_from"]] = max(last_use.get(L["residual_from"], -1), i)
    # with a separate GAP the fc pair is counted against the full last-conv tensor (conservative)
    pair_max, live_max, skip_max = 0, 0, 0
    for i, (L, s) in enumerate(zip(layers, srcs)):
        out = size[L["layer"]]
        pair_max = max(pair_max, size[s] + out)
        live = sum(size[t] for t, p in produced.items() if p < i and last_use.get(t, -1) >= i) + out
        if L.get("residual_from") and cfg.residual == "separate":
            live += out                     # pre-add conv output stored before the element-wise pass
        live_max = max(live_max, live)
        if L.get("residual_from"):
            skip_max = max(skip_max, size[L["residual_from"]])
    largest = max(size.values())
    n_buf = 2 + (1 if skip_max else 0) + (1 if skip_max and cfg.residual == "separate" else 0)
    bank_depth = ceil_div(largest, cfg.R)
    per_bank = blocks(8, bank_depth, "BRAM36")
    return {"largest_tensor_bytes": largest, "pair_max_bytes": pair_max, "skip_max_bytes": skip_max,
            "peak_live_bytes": live_max, "n_buffers": n_buf, "bank_depth": bank_depth,
            "bram36": n_buf * cfg.R * per_bank, "uram": n_buf * blocks(8 * cfg.R, bank_depth, "URAM288")}


# ---------------------------------------------------------------- qparams + totals
def qparam_storage(layers: Sequence[dict], bits_per_oc: int = QPARAM_BITS_PER_OC) -> dict:
    n = sum(int(L["oc"]) for L in layers)
    return {"channels": n, "bits": n * bits_per_oc, "bram36": blocks(bits_per_oc, n, "BRAM36")}


def device_totals(path: Path = DEVICE_CSV) -> dict:
    """{'DSP','BRAM36','URAM','LUT','FF'} -> available, from the Vivado part query CSV; None if absent."""
    keys = {"DSP": None, "BRAM36": None, "URAM": None, "LUT": None, "FF": None}
    if not Path(path).exists():
        return keys
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            name = (r.get("resource") or "").upper()
            try:
                avail = int(float(r.get("available", "")))
            except ValueError:
                continue
            if "DSP" in name:
                keys["DSP"] = avail
            elif "URAM" in name:
                keys["URAM"] = avail
            elif "RAMB36" in name or name in ("BRAM", "BRAM36", "BLOCK RAM TILE", "RAMB36E2"):
                keys["BRAM36"] = avail
            elif "LUT" in name and "RAM" not in name:
                keys["LUT"] = keys["LUT"] or avail
            elif name in ("FF", "REGISTER", "CLB REGISTERS", "FDRE"):
                keys["FF"] = avail
    return keys


def storage(layers: Sequence[dict], cfg: ArrayConfig, density: float | dict = 1.0, masks: dict | None = None,
            weights_in: str = "URAM288") -> dict:
    """Summary: weights in URAM (default) or BRAM36, activations/qparams/metadata in BRAM36."""
    w = weight_storage(layers, cfg, density, masks)
    a = act_storage(layers, cfg)
    q = qparam_storage(layers)
    w_uram = w["uram"] if weights_in == "URAM288" else 0
    w_bram = w["bram36"] if weights_in == "BRAM36" else 0
    return {"weights": w, "acts": a, "qparams": q, "weights_in": weights_in,
            "uram": w_uram, "bram36": w_bram + a["bram36"] + q["bram36"] + w["meta_bram36"]}
