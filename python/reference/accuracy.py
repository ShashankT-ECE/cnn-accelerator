"""Calibration, INT8 layer emulation, and FP32-vs-INT8 accuracy measurement.

Produces the numbers the research paper needs:

  * the calibrated per-tensor activation scale S_a and per-channel weight scales S_w,
  * the layer-level error of the INT8 Conv1 vs the FP32 Conv1 (max/mean/RMSE/SQNR),
  * the end-to-end top-1 accuracy of the network with Conv1 run in INT8.

The INT8 layer is emulated exactly as the hardware + PS software will: the
integer MAC is the independent golden model (``int8_ref``); the bias is added in
int32; the result is dequantised by S_a * S_w[c] and fed into the FP32 remainder
of the network (ReLU, pool, Conv2, FC).  Only Conv1 is quantised — matching the
scope of the accelerator.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import constants as C
from . import int8_ref, onnx_ref, quant


def calibrate(w: dict[str, np.ndarray], train_imgs: np.ndarray,
              n_calib: int = 1024) -> dict:
    """Calibrate the INT8 parameters for Conv1.

    train_imgs: float32 [N, 1, 28, 28] in [0,1], train split (calibration set).

    Returns a dict with S_a, S_w, q_w, q_b, and the zero-points (all 0).
    """
    # Per-tensor activation scale over the calibration set (MinMax observer).
    calib = train_imgs[:n_calib]
    S_a = quant.tensor_scale_abs_max(calib)

    # Per-channel weight scales + quantized weights.
    w_conv1 = w["w_conv1"]  # (8,1,5,5)
    q_w, S_w = quant.quantize_per_channel(w_conv1)

    # int32 bias with scale S_a * S_w[c].
    q_b = quant.quantize_bias_int32(w["b_conv1"], S_a, S_w)

    return {
        "S_a": float(S_a),
        "S_w": S_w,
        "q_w": q_w,
        "q_b": q_b,
        "zp_a": 0,      # symmetric: activation zero-point 0
        "zp_w": 0,      # symmetric: weight zero-point 0 (per-channel, all 0)
        "n_calib": n_calib,
    }


def int8_conv1_outputs(x_fp: np.ndarray, params: dict) -> np.ndarray:
    """Quantized Conv1 output (pre-ReLU, bias included) as float32 [N,8,28,28].

    Equivalent to:  y = (acc_int + q_b) * S_a * S_w[c]
    """
    S_a = params["S_a"]
    S_w = params["S_w"]
    q_b = params["q_b"]
    q_w = params["q_w"]

    # Input -> int8 (per-tensor).
    q_a = quant.quantize_tensor(x_fp, S_a).astype(np.int8)  # [N,1,28,28] -> [N,28,28]
    q_a = q_a.reshape(q_a.shape[0], q_a.shape[2], q_a.shape[3])
    q_w2 = q_w.reshape(q_w.shape[0], q_w.shape[2], q_w.shape[3])  # (8,5,5)

    acc = int8_ref.conv1_golden_batch(q_a, q_w2)  # int64 [N,8,28,28]
    acc_plus_bias = acc + q_b.reshape(1, -1, 1, 1).astype(np.int64)
    scale = (S_a * S_w).astype(np.float64).reshape(1, -1, 1, 1)
    return (acc_plus_bias * scale).astype(np.float32)


def layer_error(y_fp: np.ndarray, y_q: np.ndarray) -> dict:
    """Elementwise error metrics between FP32 and INT8 Conv1 outputs."""
    y_fp = y_fp.astype(np.float64)
    y_q = y_q.astype(np.float64)
    err = y_q - y_fp
    sig = y_fp
    p_sig = float(np.sum(sig ** 2))
    p_err = float(np.sum(err ** 2))
    sqnr = 10.0 * np.log10(p_sig / p_err) if p_err > 0 else float("inf")
    # per-channel SQNR
    ch_p_sig = np.sum(sig ** 2, axis=(0, 2, 3))
    ch_p_err = np.sum(err ** 2, axis=(0, 2, 3))
    ch_sqnr = 10.0 * np.log10(ch_p_sig / np.where(ch_p_err == 0, 1e-30, ch_p_err))
    return {
        "max_abs_err": float(np.max(np.abs(err))),
        "mean_abs_err": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "sqnr_db": sqnr,
        "ch_sqnr_db": ch_sqnr.tolist(),
        "n_elements": int(err.size),
    }


def fp32_end_to_end(x_fp: np.ndarray, w: dict[str, np.ndarray]) -> np.ndarray:
    """FP32 logits [N,10]."""
    with torch.no_grad():
        return onnx_ref.fp32_forward(torch.from_numpy(x_fp), w).numpy()


def int8_hybrid_end_to_end(x_fp: np.ndarray, w: dict[str, np.ndarray],
                           params: dict) -> np.ndarray:
    """Logits [N,10] with Conv1 replaced by the INT8 emulation."""
    y_q = int8_conv1_outputs(x_fp, params)  # [N,8,28,28] float32
    y_t = torch.from_numpy(y_q)
    x = F.relu(y_t)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    x = F.conv2d(x, torch.from_numpy(w["w_conv2"]), torch.from_numpy(w["b_conv2"]),
                 padding=C.CONV1_PAD)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=3, stride=3)
    x = x.flatten(1)
    x = x @ torch.from_numpy(w["w_fc"]) + torch.from_numpy(w["b_fc"])
    return x.numpy()


def evaluate(x_fp: np.ndarray, labels: np.ndarray, w: dict[str, np.ndarray],
             params: dict) -> dict:
    """Full FP32-vs-INT8 comparison; returns the report dict."""
    # Layer-level error over the full evaluation set.
    with torch.no_grad():
        y_fp = onnx_ref.fp32_conv1(torch.from_numpy(x_fp), w).numpy()  # [N,8,28,28]
    y_q = int8_conv1_outputs(x_fp, params)
    err = layer_error(y_fp, y_q)

    # End-to-end accuracy.
    logits_fp = fp32_end_to_end(x_fp, w)
    logits_q = int8_hybrid_end_to_end(x_fp, w, params)
    pred_fp = logits_fp.argmax(1)
    pred_q = logits_q.argmax(1)
    acc_fp = float((pred_fp == labels).mean())
    acc_q = float((pred_q == labels).mean())
    flips = int((pred_fp != pred_q).sum())

    # Quantization sanity: true clamping counts (symmetric full-range -> 0).
    w_conv1 = w["w_conv1"]  # (8,1,5,5)
    clip_a = quant.clip_count(x_fp, params["S_a"])
    clip_w = quant.clip_count(
        w_conv1, params["S_w"].reshape(-1, 1, 1, 1))

    return {
        "layer_error": err,
        "acc_fp32": acc_fp,
        "acc_int8": acc_q,
        "delta_top1_pp": (acc_q - acc_fp) * 100.0,
        "n_flipped": flips,
        "clip_activation": clip_a,
        "clip_weight": clip_w,
    }
