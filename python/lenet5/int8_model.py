"""L2 INT8 hardware-reference model (Step 2.2).

Implements the full LeNet INT8 path per ``docs/LENET5_INT8_SPEC.md``:

    conv1(1->6, 5x5 valid) -> ReLU -> pool -> conv3(6->16, 5x5 valid) -> ReLU ->
    pool -> conv5(16->120, 5x5 valid) -> ReLU -> fc1(120->84) -> ReLU ->
    fc2(84->10)  (logits, dequantized per-channel).

Integer-arithmetic-only: int8 activations/weights, int64 accumulation (asserted
to fit int32), int32 bias, canonical float64 requantization. Independent of the
L1 ``model.LeNet5.forward`` — weights are read from the frozen checkpoint and
quantized here, and the forward uses its own integer (NumPy) path.
"""
from __future__ import annotations

import numpy as np
import torch

from reference import quant
from .int8_quant import requantize

_LAYERS = ("conv1", "conv3", "conv5", "fc1", "fc2")


def int_conv2d(q_a: np.ndarray, q_w: np.ndarray) -> np.ndarray:
    """Exact integer 2-D convolution (VALID, stride 1).

    ``q_a``: int8 (N, IC, H, W); ``q_w``: int8 (OC, IC, K, K).
    Returns int64 (N, OC, OH, OW) with OH = H-K+1, OW = W-K+1.
    """
    N, IC, H, W = q_a.shape
    OC, ICw, K, K = q_w.shape
    assert IC == ICw
    OH, OW = H - K + 1, W - K + 1
    win = np.lib.stride_tricks.sliding_window_view(
        q_a.astype(np.int64), (K, K), axis=(2, 3))          # (N, IC, OH, OW, K, K)
    win = win.transpose(0, 2, 3, 1, 4, 5).reshape(N, OH, OW, IC * K * K)
    wf = q_w.astype(np.int64).reshape(OC, IC * K * K)
    out = np.einsum("npwd,od->npwo", win, wf)              # (N, OH, OW, OC)
    return out.transpose(0, 3, 1, 2)                      # (N, OC, OH, OW)


def int_maxpool2d(x: np.ndarray, k: int = 2, s: int = 2) -> np.ndarray:
    """Exact max-pool over (k, k) windows (H, W must be divisible by k)."""
    N, C, H, W = x.shape
    OH, OW = H // k, W // k
    return x.reshape(N, C, OH, k, OW, k).max(axis=(3, 5))


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def load_weights(ckpt_path) -> dict:
    """Load the frozen L1 checkpoint weights as a float32 NumPy dict."""
    sd = torch.load(ckpt_path, map_location="cpu")
    w = {}
    for name in _LAYERS:
        w[name] = {
            "weight": sd[name + ".weight"].detach().cpu().numpy().astype(np.float32),
            "bias": sd[name + ".bias"].detach().cpu().numpy().astype(np.float32),
        }
    return w


def calibrate(w: dict, calib_x: np.ndarray) -> dict:
    """Per-layer MinMax calibration (``docs/LENET5_INT8_SPEC.md`` §13).

    ``calib_x``: float32 (N, 1, 32, 32) in [0, 1]. Returns the per-layer
    quantization parameters (S_a, S_w, q_w, q_b, S_out, M).
    """
    p = {}
    S_in = quant.tensor_scale_abs_max(calib_x)
    p["S_input"] = S_in
    q = quant.quantize_tensor(calib_x, S_in).astype(np.int8)      # (N,1,32,32)

    # ---- C1 (conv 1->6, pool) ----
    q_w, S_w = quant.quantize_per_channel(w["conv1"]["weight"])
    q_b = quant.quantize_bias_int32(w["conv1"]["bias"], S_in, S_w)
    acc = int_conv2d(q, q_w)                                      # (N,6,28,28)
    real = (acc + q_b.reshape(1, -1, 1, 1)) * (S_in * S_w).reshape(1, -1, 1, 1)
    real = int_maxpool2d(_relu(real))                             # (N,6,14,14)
    S_out = quant.tensor_scale_abs_max(real)
    p["conv1"] = {"S_a": S_in, "S_w": S_w, "q_w": q_w, "q_b": q_b,
                  "S_out": S_out, "M": S_in * S_w / S_out}
    q = int_maxpool2d(_relu(requantize(acc + q_b.reshape(1, -1, 1, 1),
                                       p["conv1"]["M"].reshape(1, -1, 1, 1))))

    # ---- C3 (conv 6->16, pool) ----
    S_a = S_out
    q_w, S_w = quant.quantize_per_channel(w["conv3"]["weight"])
    q_b = quant.quantize_bias_int32(w["conv3"]["bias"], S_a, S_w)
    acc = int_conv2d(q, q_w)                                      # (N,16,10,10)
    real = (acc + q_b.reshape(1, -1, 1, 1)) * (S_a * S_w).reshape(1, -1, 1, 1)
    real = int_maxpool2d(_relu(real))                             # (N,16,5,5)
    S_out = quant.tensor_scale_abs_max(real)
    p["conv3"] = {"S_a": S_a, "S_w": S_w, "q_w": q_w, "q_b": q_b,
                  "S_out": S_out, "M": S_a * S_w / S_out}
    q = int_maxpool2d(_relu(requantize(acc + q_b.reshape(1, -1, 1, 1),
                                       p["conv3"]["M"].reshape(1, -1, 1, 1))))

    # ---- C5 (conv 16->120, no pool) ----
    S_a = S_out
    q_w, S_w = quant.quantize_per_channel(w["conv5"]["weight"])
    q_b = quant.quantize_bias_int32(w["conv5"]["bias"], S_a, S_w)
    acc = int_conv2d(q, q_w)                                      # (N,120,1,1)
    real = (acc + q_b.reshape(1, -1, 1, 1)) * (S_a * S_w).reshape(1, -1, 1, 1)
    real = _relu(real)
    S_out = quant.tensor_scale_abs_max(real)
    p["conv5"] = {"S_a": S_a, "S_w": S_w, "q_w": q_w, "q_b": q_b,
                  "S_out": S_out, "M": S_a * S_w / S_out}
    q = _relu(requantize(acc + q_b.reshape(1, -1, 1, 1),
                         p["conv5"]["M"].reshape(1, -1, 1, 1))).reshape(calib_x.shape[0], -1)

    # ---- F6 (fc 120->84) ----
    S_a = S_out
    q_w, S_w = quant.quantize_per_channel(w["fc1"]["weight"])
    q_b = quant.quantize_bias_int32(w["fc1"]["bias"], S_a, S_w)
    acc = q.astype(np.int64) @ q_w.astype(np.int64).T            # (N,84)
    real = (acc + q_b) * (S_a * S_w)
    real = _relu(real)
    S_out = quant.tensor_scale_abs_max(real)
    p["fc1"] = {"S_a": S_a, "S_w": S_w, "q_w": q_w, "q_b": q_b,
                "S_out": S_out, "M": S_a * S_w / S_out}
    q = _relu(requantize(acc + q_b, p["fc1"]["M"].reshape(1, -1)))

    # ---- Output (fc 84->10, no ReLU) ----
    S_a = S_out
    q_w, S_w = quant.quantize_per_channel(w["fc2"]["weight"])
    q_b = quant.quantize_bias_int32(w["fc2"]["bias"], S_a, S_w)
    p["fc2"] = {"S_a": S_a, "S_w": S_w, "q_w": q_w, "q_b": q_b}   # no S_out needed

    return p


class Int8LeNet5:
    """The L2 INT8 reference model; ``params`` from :func:`calibrate`."""

    def __init__(self, params: dict):
        self.p = params

    def forward_layers(self, x_fp: np.ndarray) -> dict:
        """Run the integer path; return layer tensors + dequantized logits."""
        p = self.p
        q = quant.quantize_tensor(x_fp, p["S_input"]).astype(np.int8)
        out = {"q_input": q}

        L = p["conv1"]
        acc = int_conv2d(q, L["q_w"])                            # (N,6,28,28)
        out["c1_acc"] = acc
        r = _relu(requantize(acc + L["q_b"].reshape(1, -1, 1, 1), L["M"].reshape(1, -1, 1, 1)))
        out["c1"] = r
        q = int_maxpool2d(r)                                     # (N,6,14,14)
        out["pool1"] = q

        L = p["conv3"]
        acc = int_conv2d(q, L["q_w"])                            # (N,16,10,10)
        out["c3_acc"] = acc
        r = _relu(requantize(acc + L["q_b"].reshape(1, -1, 1, 1), L["M"].reshape(1, -1, 1, 1)))
        out["c3"] = r
        q = int_maxpool2d(r)                                     # (N,16,5,5)
        out["pool2"] = q

        L = p["conv5"]
        acc = int_conv2d(q, L["q_w"])                            # (N,120,1,1)
        out["c5_acc"] = acc
        r = _relu(requantize(acc + L["q_b"].reshape(1, -1, 1, 1), L["M"].reshape(1, -1, 1, 1)))
        out["c5"] = r
        flat = r.reshape(r.shape[0], -1)                        # (N,120)

        L = p["fc1"]
        acc = flat.astype(np.int64) @ L["q_w"].astype(np.int64).T  # (N,84)
        out["f6_acc"] = acc
        r = _relu(requantize(acc + L["q_b"], L["M"].reshape(1, -1)))
        out["f6"] = r                                           # (N,84) int8

        L = p["fc2"]
        acc = r.astype(np.int64) @ L["q_w"].astype(np.int64).T   # (N,10)
        out["out_acc"] = acc
        logits = (acc + L["q_b"]) * (L["S_a"] * L["S_w"]).reshape(1, -1)  # float64 (N,10)
        out["logits"] = logits.astype(np.float32)
        return out

    def forward(self, x_fp: np.ndarray) -> np.ndarray:
        """Return float32 logits [N, 10]."""
        return self.forward_layers(x_fp)["logits"]
