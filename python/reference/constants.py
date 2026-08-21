"""Phase 1 experiment constants — single source of truth.

Every value is cited to its origin: the ONNX checkpoint architecture (from
`data/checkpoint/mnist-12.onnx`, inspected with `onnx.shape_inference`), the
frozen RTL geometry (`rtl/common/input_feed_v2.sv`), and the quantization
convention (Jacob et al. 2018). Do not edit these without updating the
documentation in `docs/PHASE1_EXPERIMENT_SPEC.md`.
"""
from __future__ import annotations

# --- Dataset / input geometry -------------------------------------------
# MNIST: 28x28 greyscale, 10 classes. The ONNX mnist-12 model consumes the raw
# image scaled to [0, 1] (NO mean/std normalization) — see the model-zoo README.
IMG_H = 28
IMG_W = 28
NUM_CLASSES = 10

# --- Accelerated layer: Conv1 (Convolution28) ----------------------------
# Conv1: 1 -> 8 channels, 5x5 kernel, stride 1, SAME_UPPER padding.
# SAME_UPPER with k=5, s=1, in=28 -> total_pad = 4 -> pad 2 top/left + 2 bottom/right,
# so the output is 28x28 (the input is zero-padded to 32x32, then a valid 5x5 conv).
CONV1_K = 5
CONV1_IC = 1
CONV1_OC = 8
CONV1_PAD = 2
CONV1_STRIDE = 1
CONV1_OUT_H = 28
CONV1_OUT_W = 28
CONV1_MACS = CONV1_OC * CONV1_OUT_H * CONV1_OUT_W * CONV1_K * CONV1_K  # 156,800
CONV1_WEIGHTS = CONV1_OC * CONV1_K * CONV1_K  # 200
CONV1_OUTPUTS = CONV1_OC * CONV1_OUT_H * CONV1_OUT_W  # 6,272

# --- Quantization convention ---------------------------------------------
# Symmetric, zero-point 0, full-range: q in [-127, 127] (level -128 unused).
# scale = max(|x|) / 127.  Weights per-output-channel; activations per-tensor.
INT8_MIN = -128
INT8_MAX = 127
SYM_LEVELS = 127

# --- Layout / canonical order -------------------------------------------
# Weights:  ch-major, then ky, then kx  ->  flat index ch*25 + ky*5 + kx.
# Golden:   ch / y / x (row-major spatial)  ->  flat index ch*784 + y*28 + x.
