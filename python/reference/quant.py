"""Symmetric INT8 quantization primitives.

Convention (Jacob et al. 2018 "Quantization and Training of Neural Networks for
Efficient Integer-Arithmetic-Only Inference"; Krishnamoorthi 2018), symmetric
zero-point 0, full-range:

    scale   S  = max(|x|) / 127
    quant   q  = clip(round_half_to_even(x / S), -128, 127)
    dequant x' = q * S

RNE = round-half-to-even (``numpy.rint``).  All arithmetic is performed in
float64 for bit-reproducibility.  The integer MAC that the hardware computes is
``sum(q_a * q_w)`` with no scale involvement; scales appear only when
(de)quantising values, which lives outside the PE (see PHASE1_EXPERIMENT_SPEC).
"""
from __future__ import annotations

import numpy as np

INT8_MIN = -128
INT8_MAX = 127
SYM_LEVELS = 127


def rne(x: np.ndarray) -> np.ndarray:
    """Round-half-to-even over float64."""
    return np.rint(np.asarray(x, dtype=np.float64))


def tensor_scale_abs_max(x: np.ndarray) -> float:
    """Per-tensor symmetric scale S = max(|x|) / 127."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    m = float(a.max()) if a.size else 0.0
    if m == 0.0:
        return 1.0  # degenerate (all-zero tensor); caller should avoid
    return m / SYM_LEVELS


def per_channel_scales(w: np.ndarray) -> np.ndarray:
    """Per-output-channel symmetric scales S[c] = max(|w[c]|) / 127.

    ``w`` has shape (C, ...) with the output channel on axis 0.
    """
    w = np.asarray(w, dtype=np.float64)
    max_abs = np.abs(w.reshape(w.shape[0], -1)).max(axis=1)
    max_abs = np.where(max_abs == 0.0, 1.0, max_abs)  # guard dead channels
    return max_abs / SYM_LEVELS


def quantize_tensor(x: np.ndarray, scale: float) -> np.ndarray:
    """Quantize a tensor with a single (per-tensor) scale -> int8."""
    q = rne(np.asarray(x, dtype=np.float64) / scale)
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)


def quantize_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel quantize ``w`` (shape (C, ...)) -> (int8, scales)."""
    w = np.asarray(w, dtype=np.float64)
    s = per_channel_scales(w)
    shape = (-1,) + (1,) * (w.ndim - 1)
    q = rne(w / s.reshape(shape))
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8), s


def quantize_bias_int32(b: np.ndarray, s_a: float, s_w: np.ndarray) -> np.ndarray:
    """Quantize a per-channel float bias to int32 with scale S_a * S_w[c].

    q_b[c] = rne(b[c] / (S_a * S_w[c])), clipped to int32 (safe here).
    """
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    s_w = np.asarray(s_w, dtype=np.float64).reshape(-1)
    q = rne(b / (s_a * s_w))
    return q.astype(np.int32)


def clip_count(x: np.ndarray, scale) -> int:
    """Number of values actually clamped to the int8 boundary.

    A value is "clipped" only if its rounded ratio falls outside [-128, 127].
    With symmetric full-range quantisation (scale = max|x|/127) this is 0 by
    construction: the max-magnitude value maps to exactly +/-127, and rounding
    never pushes it past the boundary.
    """
    x = np.asarray(x, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    rounded = rne(x / scale)
    return int((rounded > INT8_MAX).sum() + (rounded < INT8_MIN).sum())


def dequantize_int32(q: np.ndarray, s_a: float, s_w: np.ndarray) -> np.ndarray:
    """Dequantize an int32 per-channel tensor back to float: y' = (q) * S_a * S_w[c]."""
    q = np.asarray(q, dtype=np.int64)
    s_w = np.asarray(s_w, dtype=np.float64)
    scale = (float(s_a) * s_w).reshape((-1,) + (1,) * (q.ndim - 1))
    return (q * scale).astype(np.float32)
