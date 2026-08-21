"""Independent integer golden model (L2) for the accelerated Conv1 layer.

This is the *definition* of the operation, not a description of the systolic
schedule.  It is deliberately written so that someone who has never read the PE
geometry, the tile counters, or the input-feed schedule can verify it:

    O[c, y, x] = sum_{ky=0..4} sum_{kx=0..4}
                    q_a[y + ky - pad, x + kx - pad] * q_w[c, ky, kx]

with zero padding (out-of-range reads contribute 0).  Operands are int8; the
accumulation is exact integer arithmetic in int64, cast to int32 at the boundary
(the hardware accumulator is int32, and |acc| <= 25 * 127 * 127 = 403,225, far
below 2**31).

Two implementations are provided:
  * ``conv1_golden_loop``   — pure-Python nested loops (the definition itself).
  * ``conv1_golden_einsum`` — vectorised NumPy, validated bit-identical to the
    loop in ``python/tests/test_pipeline.py``.

Neither reads the RTL, the schedule, or the PE/array geometry.
"""
from __future__ import annotations

import numpy as np

from . import constants as C


def conv1_golden_loop(q_a: np.ndarray, q_w: np.ndarray) -> np.ndarray:
    """Pure-Python definition of the zero-padded 5x5 integer convolution.

    q_a: int8 (H, W)  (H = W = 28)
    q_w: int8 (C, 5, 5)
    returns int64 (C, 28, 28) in ch/y/x order.
    """
    Cc, K, _ = q_w.shape
    H, W = q_a.shape
    pad = C.CONV1_PAD
    out = np.zeros((Cc, H, W), dtype=np.int64)
    for c in range(Cc):
        for y in range(H):
            for x in range(W):
                acc = 0
                for ky in range(K):
                    iy = y + ky - pad
                    if iy < 0 or iy >= H:
                        continue
                    for kx in range(K):
                        ix = x + kx - pad
                        if ix < 0 or ix >= W:
                            continue
                        acc += int(q_a[iy, ix]) * int(q_w[c, ky, kx])
                out[c, y, x] = acc
    return out


def conv1_golden_einsum(q_a: np.ndarray, q_w: np.ndarray) -> np.ndarray:
    """Vectorised NumPy equivalent of ``conv1_golden_loop`` (int64)."""
    pad = C.CONV1_PAD
    qp = np.pad(q_a.astype(np.int64), ((pad, pad), (pad, pad)), mode="constant")
    win = np.lib.stride_tricks.sliding_window_view(qp, (C.CONV1_K, C.CONV1_K))
    qw = q_w.astype(np.int64)
    return np.einsum("yxkh,ckh->cyx", win, qw)


def conv1_golden(q_a: np.ndarray, q_w: np.ndarray) -> np.ndarray:
    """Return the int64 golden (ch/y/x); assert it fits int32."""
    out = conv1_golden_einsum(q_a, q_w)
    assert int(out.min()) >= -(2 ** 31) and int(out.max()) <= 2 ** 31 - 1
    return out


def conv1_golden_batch(q_a_batch: np.ndarray, q_w: np.ndarray) -> np.ndarray:
    """Batched zero-padded integer convolution.

    q_a_batch: int8 (N, H, W)
    q_w:       int8 (C, 5, 5)
    returns:   int64 (N, C, 28, 28), same definition as ``conv1_golden``.
    """
    pad = C.CONV1_PAD
    qp = np.pad(q_a_batch.astype(np.int64), ((0, 0), (pad, pad), (pad, pad)),
                mode="constant")
    win = np.lib.stride_tricks.sliding_window_view(
        qp, (C.CONV1_K, C.CONV1_K), axis=(1, 2))
    qw = q_w.astype(np.int64)
    return np.einsum("nyxkh,ckh->ncyx", win, qw)
