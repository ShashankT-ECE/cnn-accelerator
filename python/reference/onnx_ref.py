"""FP32 reference model (L1): exact semantics of the ONNX mnist-12 checkpoint.

Implements the graph read from `data/checkpoint/mnist-12.onnx`:

    Input [1,1,28,28]
      Conv1  1->8  k5 s1 SAME_UPPER(pad2) + bias   -> [1,8,28,28]
      ReLU
      MaxPool 2x2/2                                -> [1,8,14,14]
      Conv2  8->16 k5 s1 SAME_UPPER(pad2) + bias   -> [1,16,14,14]
      ReLU
      MaxPool 3x3/3                                -> [1,16,4,4]
      Flatten -> 256
      Linear 256->10 + bias                        -> [1,10] (logits)

The forward pass uses ``torch.nn.functional`` (pinned torch 2.1.2+cpu), which is
the battle-tested reference for these primitives.  The INT8 golden model is a
separate, independent NumPy implementation, so the two do not share code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import constants as C

REPO_ROOT = Path(__file__).resolve().parents[2]
NPZ_PATH = REPO_ROOT / "data" / "checkpoint" / "mnist_weights.npz"


def load_weights(npz_path: Path = NPZ_PATH) -> dict[str, np.ndarray]:
    """Load the extracted weights, with the 1x1 bias axes squeezed."""
    d = np.load(npz_path)
    w = {
        "w_conv1": d["w_conv1"].astype(np.float32),
        "b_conv1": d["b_conv1"].astype(np.float32).reshape(-1),
        "w_conv2": d["w_conv2"].astype(np.float32),
        "b_conv2": d["b_conv2"].astype(np.float32).reshape(-1),
        "w_fc": d["w_fc"].astype(np.float32),
        "b_fc": d["b_fc"].astype(np.float32).reshape(-1),
    }
    return w


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x)


def fp32_conv1(x_fp32: torch.Tensor, w: dict[str, np.ndarray]) -> torch.Tensor:
    """Conv1 + bias (pre-ReLU), SAME_UPPER padding=2 -> [N, 8, 28, 28]."""
    return F.conv2d(x_fp32, _t(w["w_conv1"]), _t(w["b_conv1"]), padding=C.CONV1_PAD)


def fp32_forward(x_fp32: torch.Tensor, w: dict[str, np.ndarray]) -> torch.Tensor:
    """Full network -> logits [N, 10]."""
    x = fp32_conv1(x_fp32, w)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    x = F.conv2d(x, _t(w["w_conv2"]), _t(w["b_conv2"]), padding=C.CONV1_PAD)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=3)
    x = x.flatten(1)
    # ONNX MatMul: [N,256] @ [256,10] + [10]  (w_fc stored in ONNX layout [256,10])
    x = x @ _t(w["w_fc"]) + _t(w["b_fc"])
    return x


def predict_logits(x_fp32: torch.Tensor, w: dict[str, np.ndarray]) -> np.ndarray:
    with torch.no_grad():
        return fp32_forward(x_fp32, w).numpy()
