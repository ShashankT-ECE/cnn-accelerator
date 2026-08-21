"""Reproducible MNIST preprocessing.

The ONNX mnist-12 checkpoint (CNTK-trained) consumes a 28x28 greyscale float
image in [0, 1], black background / white foreground, with **no** mean/std
normalisation — see the model-zoo README (validated/vision/classification/mnist).

Preprocessing is therefore a single step — ``uint8 / 255 -> float32`` — followed,
for the integer pipeline, by the int8 activation quantisation
(``quant.quantize_tensor`` with the per-tensor activation scale S_a = 1/127).
"""
from __future__ import annotations

import numpy as np

from . import quant


def pil_to_input(pil_image) -> np.ndarray:
    """PIL image (0..255) -> float32 [0,1], shape (1, 1, H, W)."""
    arr = np.asarray(pil_image, dtype=np.float32) / 255.0
    return arr.reshape(1, 1, arr.shape[0], arr.shape[1])


def to_int8(x_fp: np.ndarray, scale_a: float) -> np.ndarray:
    """Quantize a [0,1] float tensor to int8 with a single activation scale."""
    return quant.quantize_tensor(x_fp, scale_a).astype(np.int8)
