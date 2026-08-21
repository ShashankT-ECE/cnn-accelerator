#!/usr/bin/env python3
"""Obtain and verify the pretrained checkpoint, then extract weights.

Source: ONNX Model Zoo "MNIST-12"
    https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
    model/mnist-12.onnx  (MIT licence, ONNX opset 12, validated TOP-1 error 1.1%)

This script is idempotent and reproducibility-first:
  * downloads the checkpoint only if it is absent,
  * verifies its SHA-256 against a pinned digest,
  * extracts the weight tensors to a plain NumPy .npz for the reference model.

The checkpoint binary itself is committed at data/checkpoint/mnist-12.onnx so a
clean clone needs no network access to reproduce the experiment.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "data" / "checkpoint"
CKPT_PATH = CKPT_DIR / "mnist-12.onnx"
NPZ_PATH = CKPT_DIR / "mnist_weights.npz"

# Pinned source and digest (recorded at acquisition time).
SOURCE_URL = (
    "https://media.githubusercontent.com/media/onnx/models/"
    "main/validated/vision/classification/mnist/model/mnist-12.onnx"
)
SHA256 = "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"

# ONNX tensor names for the accelerated Conv1 (Convolution28) and the rest.
W_CONV1 = "Parameter5"    # [8, 1, 5, 5]
B_CONV1 = "Parameter6"    # [8]
W_CONV2 = "Parameter87"   # [16, 8, 5, 5]
B_CONV2 = "Parameter88"   # [16]
W_FC = "Parameter193"     # [16, 4, 4, 10] -> reshape [256, 10]
B_FC = "Parameter194"     # [10]


def _download() -> bytes:
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def ensure_checkpoint() -> Path:
    """Return the checkpoint path, downloading and verifying if needed."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    if CKPT_PATH.exists():
        digest = hashlib.sha256(CKPT_PATH.read_bytes()).hexdigest()
        if digest != SHA256:
            raise RuntimeError(
                f"{CKPT_PATH} SHA-256 mismatch: got {digest}, expected {SHA256}"
            )
        print(f"[fetch] checkpoint present, SHA-256 OK ({CKPT_PATH.name})")
        return CKPT_PATH

    print(f"[fetch] downloading {SOURCE_URL}")
    data = _download()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise RuntimeError(
            f"downloaded checkpoint SHA-256 mismatch: got {digest}, expected {SHA256}"
        )
    CKPT_PATH.write_bytes(data)
    print(f"[fetch] wrote {CKPT_PATH.name} ({len(data)} bytes), SHA-256 {digest}")
    return CKPT_PATH


def extract_weights(ckpt_path: Path) -> Path:
    """Extract the weight tensors to data/checkpoint/mnist_weights.npz."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(ckpt_path))
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}

    w_conv1 = np.asarray(init[W_CONV1], dtype=np.float32)
    b_conv1 = np.asarray(init[B_CONV1], dtype=np.float32)
    w_conv2 = np.asarray(init[W_CONV2], dtype=np.float32)
    b_conv2 = np.asarray(init[B_CONV2], dtype=np.float32)
    w_fc = np.asarray(init[W_FC], dtype=np.float32).reshape(256, 10)
    b_fc = np.asarray(init[B_FC], dtype=np.float32).reshape(10)

    np.savez(
        NPZ_PATH,
        w_conv1=w_conv1, b_conv1=b_conv1,
        w_conv2=w_conv2, b_conv2=b_conv2,
        w_fc=w_fc, b_fc=b_fc,
    )
    print(
        f"[fetch] extracted {NPZ_PATH.name}: "
        f"w_conv1 {w_conv1.shape}, b_conv1 {b_conv1.shape}, "
        f"w_conv2 {w_conv2.shape}, b_conv2 {b_conv2.shape}, "
        f"w_fc {w_fc.shape}, b_fc {b_fc.shape}"
    )
    return NPZ_PATH


def main() -> int:
    ckpt = ensure_checkpoint()
    extract_weights(ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
