#!/usr/bin/env python3
"""Phase 1 runner: calibrate, evaluate FP32-vs-INT8, export RTL test vectors.

Run from the repository root:

    .venv/bin/python python/run_phase1.py

Deterministic: fixed calibration split (train[0:1024]), full test split for
evaluation, fixed test indices for export.  No RNG anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from torchvision import datasets

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from reference import accuracy, constants as C, export_vectors, int8_ref, onnx_ref, preprocess  # noqa: E402

N_CALIB = 1024
EXPORT_INDICES = list(range(10))


def load_split(split: str, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    ds = datasets.MNIST(root="data/raw", train=(split == "train"), download=False)
    n = len(ds) if n is None else n
    xs = np.stack([preprocess.pil_to_input(ds[i][0]) for i in range(n)])
    ys = np.array([ds[i][1] for i in range(n)])
    return xs.reshape(-1, 1, C.IMG_H, C.IMG_W), ys


def main() -> int:
    w = onnx_ref.load_weights()

    print("Loading calibration set (train[0:%d])..." % N_CALIB)
    X_cal, _ = load_split("train", N_CALIB)
    print("Loading full test set...")
    X_te, Y_te = load_split("test")

    print("Calibrating...")
    params = accuracy.calibrate(w, X_cal, n_calib=N_CALIB)

    print("Evaluating FP32 vs INT8 (full test set)...")
    report = accuracy.evaluate(X_te, Y_te, w, params)

    print("Exporting test vectors + manifest...")
    images_int8 = []
    goldens = []
    labels = []
    for idx in EXPORT_INDICES:
        img_fp = X_te[idx:idx + 1]
        q_a = preprocess.to_int8(img_fp, params["S_a"])
        q_w2 = params["q_w"].reshape(C.CONV1_OC, C.CONV1_K, C.CONV1_K)
        gold = int8_ref.conv1_golden(q_a[0, 0], q_w2)
        images_int8.append(q_a[0, 0])  # (28,28) int8
        goldens.append(gold)           # (8,28,28) int64
        labels.append(int(Y_te[idx]))
    manifest_path = export_vectors.export(
        params, w, images_int8, goldens, EXPORT_INDICES, labels)

    # ---- print report ----
    le = report["layer_error"]
    print("\n" + "=" * 72)
    print("PHASE 1 RESULTS — ONNX MNIST-12 Conv1, calibrated symmetric INT8")
    print("=" * 72)
    print(f"  Activation scale  S_a        = {params['S_a']!r}  (max|a|=1.0 -> 1/127)")
    print(f"  Weight scales     S_w        = {[f'{s:.6f}' for s in params['S_w']]}")
    print(f"  Bias (int32)      q_b        = {params['q_b'].tolist()}")
    print(f"  Zero-points       (a,w)      = ({params['zp_a']}, {params['zp_w']})")
    print("-" * 72)
    print("  Layer error (FP32 vs INT8 Conv1, pre-ReLU, full test set):")
    print(f"    max |err|   = {le['max_abs_err']:.5f}")
    print(f"    mean |err|  = {le['mean_abs_err']:.5f}")
    print(f"    RMSE        = {le['rmse']:.5f}")
    print(f"    SQNR        = {le['sqnr_db']:.2f} dB")
    print(f"    ch SQNR dB  = {[f'{v:.1f}' for v in le['ch_sqnr_db']]}")
    print(f"    elements    = {le['n_elements']}")
    print("-" * 72)
    print("  End-to-end top-1 (full test set):")
    print(f"    FP32        = {report['acc_fp32']*100:.4f}%")
    print(f"    INT8 Conv1  = {report['acc_int8']*100:.4f}%")
    print(f"    delta       = {report['delta_top1_pp']:+.4f} pp")
    print(f"    flips       = {report['n_flipped']}")
    print("-" * 72)
    print(f"  Clipping (true out-of-range): activation={report['clip_activation']} "
          f"weight={report['clip_weight']}")
    print(f"  Manifest written to: {manifest_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
