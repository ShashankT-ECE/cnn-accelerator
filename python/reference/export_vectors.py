"""Export deterministic INT8 RTL test vectors + reproducibility manifest.

Produces, under data/vectors/:

  * weights.hex          — the 200 Conv1 weights, int8 two's-complement hex,
                           ch-major, then ky*5+kx (200 lines).
  * input_img.hex        — the primary test image (784 int8 values), row-major.
  * golden_canonical.txt — the primary image's integer golden, decimal int32,
                           ch/y/x order (6272 lines).
  * multi/               — a small deterministic regression set (10 images).
  * quant_params.npz     — S_a, S_w, q_b, q_w, zero-points (machine-readable).
  * manifest.json        — full reproducibility metadata.

Hex format is ``$readmemh``-compatible: 2-digit lowercase two's-complement hex,
one value per line (there is no ``$readmemd``).  Negative int8 is emitted as its
8-bit two's-complement pattern (e.g. -1 -> "ff").

The HW emission order (reverse-index / group order) is a Phase 2 deliverable:
the controller for the padded OC=8 layer does not exist yet, so only the
canonical ch/y/x golden is emitted here.  Phase 2's ``compare.py`` will reorder
this canonical golden to the controller's emission order.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import constants as C

VECTORS_DIR = Path(__file__).resolve().parents[2] / "data" / "vectors"


def fmt_int8(v: int) -> str:
    """int8 -> 2-digit lowercase two's-complement hex."""
    return f"{int(v) & 0xff:02x}"


def flatten_weights_ch_major(q_w: np.ndarray) -> np.ndarray:
    """q_w (8,1,5,5) -> 1-D in ch*25 + ky*5 + kx order."""
    return q_w.reshape(C.CONV1_OC, C.CONV1_K * C.CONV1_K).ravel()


def flatten_golden_ch_y_x(golden: np.ndarray) -> np.ndarray:
    """golden (8,28,28) -> 1-D in ch*784 + y*28 + x order."""
    return golden.ravel()  # C-order already yields ch/y/x


def write_hex(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in values:
            f.write(fmt_int8(int(v)) + "\n")


def write_golden(path: Path, golden: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = flatten_golden_ch_y_x(golden)
    with open(path, "w") as f:
        for v in flat:
            f.write(f"{int(v)}\n")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=VECTORS_DIR.parents[1],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _env_versions() -> dict[str, str]:
    import torch
    import torchvision
    import numpy as _np
    import onnx
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": _np.__version__,
        "onnx": onnx.__version__,
    }


def build_manifest(params: dict, test_indices: list[int], labels: list[int]) -> dict:
    manifest = {
        "experiment": "Phase 1 — INT8 golden pipeline for ONNX MNIST-12 Conv1",
        "cnn": {
            "name": "LeNet-5 variant (CNTK 103D), ONNX Model Zoo 'MNIST-12'",
            "checkpoint": "data/checkpoint/mnist-12.onnx",
            "source_url": (
                "https://github.com/onnx/models/tree/main/"
                "validated/vision/classification/mnist"
            ),
            "license": "MIT",
            "sha256": "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd",
            "stated_top1_error": "1.1%",
        },
        "dataset": {
            "name": "MNIST",
            "source": "http://yann.lecun.com/exdb/mnist/ (mirror: ossci-datasets S3)",
            "split": {"train": 60000, "test": 10000},
            "image": "28x28 greyscale, [0,1], no mean/std normalisation",
        },
        "accelerated_layer": {
            "layer": "Conv1 (Convolution28)",
            "in_channels": C.CONV1_IC,
            "out_channels": C.CONV1_OC,
            "kernel": C.CONV1_K,
            "stride": C.CONV1_STRIDE,
            "padding": f"SAME_UPPER, pad={C.CONV1_PAD}",
            "input_shape": [1, C.CONV1_IC, C.IMG_H, C.IMG_W],
            "output_shape": [1, C.CONV1_OC, C.CONV1_OUT_H, C.CONV1_OUT_W],
            "maccs": C.CONV1_MACS,
            "weights": C.CONV1_WEIGHTS,
            "outputs": C.CONV1_OUTPUTS,
        },
        "quantization": {
            "scheme": "symmetric, zero-point 0, full-range (q in [-127,127])",
            "rounding": "round-half-to-even (RNE), float64",
            "weights": "per-output-channel, S_w[c] = max|w_c|/127",
            "activations": "per-tensor, S_a = max|a|/127",
            "bias": "int32, scale S_a * S_w[c]",
            "accumulator": "int32",
            "S_a": params["S_a"],
            "S_w": params["S_w"].tolist(),
            "q_b": params["q_b"].tolist(),
            "zero_point_activation": params["zp_a"],
            "zero_point_weight": params["zp_w"],
            "n_calibration_samples": params["n_calib"],
        },
        "tensor_layout": {
            "input": "row-major y*28 + x, single channel",
            "weights": "ch*25 + ky*5 + kx (ch-major)",
            "golden_canonical": "ch*784 + y*28 + x (ch/y/x order)",
            "note": "HW emission order is a Phase 2 deliverable (see docs).",
        },
        "test_vectors": {
            "primary_image": test_indices[0],
            "indices": test_indices,
            "labels": labels,
            "deterministic": "test split, fixed indices, no RNG",
        },
        "environment": _env_versions(),
        "git_sha": _git_sha(),
    }
    return manifest


def export(params: dict, weights: dict, images_int8: list[np.ndarray],
           goldens: list[np.ndarray], test_indices: list[int],
           labels: list[int]) -> Path:
    """Write all vector artifacts; returns the manifest path."""
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    # Shared weights (one file for all images).
    q_w = params["q_w"]
    write_hex(VECTORS_DIR / "weights.hex", flatten_weights_ch_major(q_w))

    # Primary vector (image 0) under the canonical names.
    write_hex(VECTORS_DIR / "input_img.hex", images_int8[0].ravel())
    write_golden(VECTORS_DIR / "golden_canonical.txt", goldens[0])

    # Small regression set under multi/.
    for img, gold, idx in zip(images_int8, goldens, test_indices):
        write_hex(VECTORS_DIR / "multi" / f"input_img_{idx:05d}.hex", img.ravel())
        write_golden(VECTORS_DIR / "multi" / f"golden_{idx:05d}.txt", gold)

    # Machine-readable quantisation parameters.
    np.savez(
        VECTORS_DIR / "quant_params.npz",
        S_a=np.float64(params["S_a"]),
        S_w=params["S_w"].astype(np.float64),
        q_b=params["q_b"],
        q_w=params["q_w"],
        zp_a=np.int32(params["zp_a"]),
        zp_w=np.int32(params["zp_w"]),
    )

    # Manifest.
    manifest = build_manifest(params, test_indices, labels)
    manifest_path = VECTORS_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path
