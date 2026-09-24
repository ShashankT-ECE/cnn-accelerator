"""L2 INT8 hardware-reference model for the CIFAR-10 workload (Step 6.3).

Mirrors the frozen convention (docs/LENET5_INT8_SPEC.md): symmetric signed INT8,
zero-point 0, per-channel weight scales, per-tensor activation scales, int32
accumulation/bias, canonical RNE requantization. Independent of the CIFAR-10 FP32
forward (reads the frozen checkpoint weights and runs its own integer path).
"""
from __future__ import annotations

import numpy as np
import torch

from reference import quant
from lenet5.int8_model import int_conv2d, int_maxpool2d
from lenet5.int8_quant import requantize

_LAYERS = ("conv1", "conv2", "conv3", "fc")


def load_weights(ckpt_path) -> dict:
    sd = torch.load(ckpt_path, map_location="cpu")
    return {n: {"weight": sd[n + ".weight"].detach().cpu().numpy().astype(np.float32),
                "bias": sd[n + ".bias"].detach().cpu().numpy().astype(np.float32)}
            for n in _LAYERS}


def calibrate(w: dict, calib_x: np.ndarray) -> dict:
    p = {}
    S_in = quant.tensor_scale_abs_max(calib_x)
    p["S_input"] = S_in
    q = quant.quantize_tensor(calib_x, S_in).astype(np.int8)          # (N,3,32,32)

    # Conv1 (3->32, 5x5 valid, pool)
    q_w, S_w = quant.quantize_per_channel(w["conv1"]["weight"])
    q_b = quant.quantize_bias_int32(w["conv1"]["bias"], S_in, S_w)
    acc = int_conv2d(q, q_w)                                          # (N,32,28,28)
    real = int_maxpool2d(np.maximum((acc + q_b.reshape(1, -1, 1, 1)) *
                                    (S_in * S_w).reshape(1, -1, 1, 1), 0))
    S_out = quant.tensor_scale_abs_max(real)
    p["conv1"] = {"S_a": S_in, "S_w": S_w, "q_w": q_w, "q_b": q_b,
                  "S_out": S_out, "M": S_in * S_w / S_out}
    q = int_maxpool2d(np.maximum(requantize(acc + q_b.reshape(1, -1, 1, 1),
                                            p["conv1"]["M"].reshape(1, -1, 1, 1)), 0))

    # Conv2 (32->32, 5x5 valid, pool)
    S_a = S_out
    q_w, S_w = quant.quantize_per_channel(w["conv2"]["weight"])
    q_b = quant.quantize_bias_int32(w["conv2"]["bias"], S_a, S_w)
    acc = int_conv2d(q, q_w)                                          # (N,32,10,10)
    real = int_maxpool2d(np.maximum((acc + q_b.reshape(1, -1, 1, 1)) *
                                    (S_a * S_w).reshape(1, -1, 1, 1), 0))
    S_out = quant.tensor_scale_abs_max(real)
    p["conv2"] = {"S_a": S_a, "S_w": S_w, "q_w": q_w, "q_b": q_b,
                  "S_out": S_out, "M": S_a * S_w / S_out}
    q = int_maxpool2d(np.maximum(requantize(acc + q_b.reshape(1, -1, 1, 1),
                                            p["conv2"]["M"].reshape(1, -1, 1, 1)), 0))

    # Conv3 (32->64, 5x5 valid, no pool)
    S_a = S_out
    q_w, S_w = quant.quantize_per_channel(w["conv3"]["weight"])
    q_b = quant.quantize_bias_int32(w["conv3"]["bias"], S_a, S_w)
    acc = int_conv2d(q, q_w)                                          # (N,64,1,1)
    real = np.maximum((acc + q_b.reshape(1, -1, 1, 1)) * (S_a * S_w).reshape(1, -1, 1, 1), 0)
    S_out = quant.tensor_scale_abs_max(real)
    p["conv3"] = {"S_a": S_a, "S_w": S_w, "q_w": q_w, "q_b": q_b,
                  "S_out": S_out, "M": S_a * S_w / S_out}
    q = np.maximum(requantize(acc + q_b.reshape(1, -1, 1, 1),
                              p["conv3"]["M"].reshape(1, -1, 1, 1)), 0).reshape(calib_x.shape[0], -1)

    # FC (64->10, no ReLU)
    S_a = S_out
    q_w, S_w = quant.quantize_per_channel(w["fc"]["weight"])
    q_b = quant.quantize_bias_int32(w["fc"]["bias"], S_a, S_w)
    p["fc"] = {"S_a": S_a, "S_w": S_w, "q_w": q_w, "q_b": q_b}
    return p


class Int8Cifar10Net:
    def __init__(self, params: dict):
        self.p = params

    def forward_layers(self, x_fp: np.ndarray) -> dict:
        p = self.p
        q = quant.quantize_tensor(x_fp, p["S_input"]).astype(np.int8)
        out = {"q_input": q}

        L = p["conv1"]
        acc = int_conv2d(q, L["q_w"])                                   # (N,32,28,28)
        out["c1_acc"] = acc
        r = np.maximum(requantize(acc + L["q_b"].reshape(1, -1, 1, 1),
                                  L["M"].reshape(1, -1, 1, 1)), 0)
        q = int_maxpool2d(r)                                            # (N,32,14,14)
        out["pool1"] = q

        L = p["conv2"]
        acc = int_conv2d(q, L["q_w"])                                   # (N,32,10,10)
        out["c2_acc"] = acc
        r = np.maximum(requantize(acc + L["q_b"].reshape(1, -1, 1, 1),
                                  L["M"].reshape(1, -1, 1, 1)), 0)
        q = int_maxpool2d(r)                                            # (N,32,5,5)
        out["pool2"] = q

        L = p["conv3"]
        acc = int_conv2d(q, L["q_w"])                                   # (N,64,1,1)
        out["c3_acc"] = acc
        r = np.maximum(requantize(acc + L["q_b"].reshape(1, -1, 1, 1),
                                  L["M"].reshape(1, -1, 1, 1)), 0)
        out["c3"] = r
        flat = r.reshape(r.shape[0], -1)                                # (N,64)

        L = p["fc"]
        acc = flat.astype(np.int64) @ L["q_w"].astype(np.int64).T       # (N,10)
        out["fc_acc"] = acc
        logits = (acc + L["q_b"]) * (L["S_a"] * L["S_w"]).reshape(1, -1)  # float64 (N,10)
        out["logits"] = logits.astype(np.float32)
        return out

    def forward(self, x_fp: np.ndarray) -> np.ndarray:
        return self.forward_layers(x_fp)["logits"]
