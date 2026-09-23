"""L2 INT8 quantization and requantization primitives (Step 2.2).

Extends the frozen Phase-1 primitives (``reference.quant``) with the full-network
requantization required by ``docs/LENET5_INT8_SPEC.md`` §9–§11. The base
symmetric-INT8 primitives are re-exported from ``reference.quant`` (single frozen
source of truth); the L2 additions here are requantization only.
"""
from __future__ import annotations

import numpy as np

from reference import quant  # frozen Phase-1 symmetric INT8 primitives

# Re-export the base primitives.
INT8_MIN = quant.INT8_MIN          # -128
INT8_MAX = quant.INT8_MAX          # 127
SYM_LEVELS = quant.SYM_LEVELS      # 127
rne = quant.rne
tensor_scale_abs_max = quant.tensor_scale_abs_max
per_channel_scales = quant.per_channel_scales
quantize_tensor = quant.quantize_tensor
quantize_per_channel = quant.quantize_per_channel
quantize_bias_int32 = quant.quantize_bias_int32


def requantize(y, M):
    """Canonical requantization (float64 reference): ``q = clip(RNE(y·M), -128, 127)``.

    ``y`` is the int32 accumulator (bias added); ``M = S_a·S_w[c]/S_out`` is the
    per-channel requantization scale (broadcastable against ``y``). RNE =
    round-half-to-even.
    """
    q = rne(np.asarray(y, dtype=np.float64) * np.asarray(M, dtype=np.float64))
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)


def requantize_fixed(y, M):
    """Integer-only fixed-point requantization (Jacob et al. 2018 §2.2).

    ``M = M0 · 2^-n`` with ``M0 ∈ [0.5, 1)``; int32 multiplier ``m = RNE(M0·2^31)``;
    ``q = clip(((y·m + 2^(shift-1)) >> shift), -128, 127)`` with ``shift = 31 + n``.
    Realizes ``requantize`` within ±1 LSB.
    """
    y = np.asarray(y, dtype=np.int64)
    M = np.asarray(M, dtype=np.float64)
    M0, exp = np.frexp(M)                    # M = M0·2^exp, M0 ∈ [0.5, 1)
    n = -exp
    m = np.rint(M0 * 2.0 ** 31).astype(np.int64)
    shift = (31 + n).astype(np.int64)
    rounding = np.int64(1) << (shift - 1)
    q = (y * m + rounding) >> shift
    return np.clip(q, INT8_MIN, INT8_MAX).astype(np.int8)
