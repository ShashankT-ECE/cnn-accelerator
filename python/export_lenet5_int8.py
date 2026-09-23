#!/usr/bin/env python3
"""Step 2.3: generate frozen L2 INT8 artifacts (reproducible).

Run from the repository root:

    .venv/bin/python python/export_lenet5_int8.py            # -> data/lenet5_int8/
    .venv/bin/python python/export_lenet5_int8.py --out DIR   # -> DIR/

Generates, deterministically (fixed indices, fixed ordering, no RNG):

  * quant_params.npz    — per-layer int8 weights, int32 biases, activation /
                          weight / output scales, and requantization scales M.
  * layer_vectors.npz   — representative int8 input / int32 accumulator /
                          int8 output vectors for every learned layer, plus the
                          dequantized logits (10 fixed test images).
  * manifest.json       — tensor shapes/dtypes, architecture, versions, hashes.
  * SHA256SUMS          — SHA-256 of every generated artifact.
  * L2_FROZEN_MANIFEST.json — frozen reference linking to L1 + the INT8 convention.

No RTL, no L1 files modified, no change to the quantization convention.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

import torch  # noqa: E402
from torchvision import datasets  # noqa: E402

from lenet5.int8_model import Int8LeNet5, calibrate, load_weights  # noqa: E402
from lenet5.preprocess import make_transform  # noqa: E402

CALIB_N = 1024
VEC_INDICES = list(range(10))          # fixed representative test images
LAYERS = ("conv1", "conv3", "conv5", "fc1", "fc2")

L1_CKPT = "data/checkpoint/lenet5_fp32.pt"
L1_SHA = "9978676be5132f25ba2bbdf93cb3930541bbbc780ecffd49c5560ad03c1af9dd"
L1_CANON = "202a965f849849a6e96061a6649098cfe28eb7d21b5a5f8ddb456f88f0558fb5"
TOTAL_PARAMS = 61706
TOTAL_MACS = 416520


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _versions() -> dict:
    import torchvision
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
    }


def _savez(path: Path, **arrays) -> None:
    np.savez(path, **arrays)


def generate(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load L1 weights + calibration set (fixed train[0:1024]) ----
    w = load_weights(REPO_ROOT / L1_CKPT)
    tr = make_transform()
    train_ds = datasets.MNIST(root=str(REPO_ROOT / "data" / "raw"), train=True,
                              download=False, transform=tr)
    calib = np.stack([train_ds[i][0].numpy() for i in range(CALIB_N)]).astype(np.float32)

    # ---- calibrate + build model ----
    params = calibrate(w, calib)
    model = Int8LeNet5(params)

    # ---- representative vectors (fixed test images) ----
    test_ds = datasets.MNIST(root=str(REPO_ROOT / "data" / "raw"), train=False,
                             download=False, transform=tr)
    xf = np.stack([test_ds[i][0].numpy() for i in VEC_INDICES]).astype(np.float32)
    labels = np.array([test_ds[i][1] for i in VEC_INDICES], dtype=np.int64)
    out = model.forward_layers(xf)

    # ---- quantization params npz ----
    qp = {"S_input": np.float64(params["S_input"])}
    for name in LAYERS:
        L = params[name]
        qp[f"{name}_q_w"] = L["q_w"]
        qp[f"{name}_q_b"] = L["q_b"]
        qp[f"{name}_S_w"] = L["S_w"].astype(np.float64)
        qp[f"{name}_S_a"] = np.float64(L["S_a"])
        if "S_out" in L:
            qp[f"{name}_S_out"] = np.float64(L["S_out"])
            qp[f"{name}_M"] = L["M"].astype(np.float64)
    _savez(out_dir / "quant_params.npz", **qp)

    # ---- layer vectors npz ----
    flat5 = out["c5"].reshape(out["c5"].shape[0], -1)
    lv = {
        "labels": labels,
        "input": out["q_input"],
        "conv1_in": out["q_input"], "conv1_acc": out["c1_acc"], "conv1_out": out["pool1"],
        "conv3_in": out["pool1"], "conv3_acc": out["c3_acc"], "conv3_out": out["pool2"],
        "conv5_in": out["pool2"], "conv5_acc": out["c5_acc"], "conv5_out": out["c5"],
        "fc1_in": flat5, "fc1_acc": out["f6_acc"], "fc1_out": out["f6"],
        "fc2_in": out["f6"], "fc2_acc": out["out_acc"], "fc2_out": out["logits"],
    }
    _savez(out_dir / "layer_vectors.npz", **lv)

    # ---- manifest.json ----
    tensor_shapes = {k: list(v.shape) for k, v in lv.items()}
    tensor_dtypes = {k: str(v.dtype) for k, v in lv.items()}
    qp_shapes = {k: list(v.shape) for k, v in qp.items()}

    manifest = {
        "experiment": "Step 2.3 — frozen L2 INT8 LeNet-5 artifacts",
        "int8_convention": "docs/LENET5_INT8_SPEC.md",
        "l1": {
            "checkpoint": L1_CKPT,
            "sha256": L1_SHA,
            "canonical_weights_sha256": L1_CANON,
        },
        "calibration": {"split": "train[0:1024]", "n": CALIB_N},
        "architecture": {
            "layers": ["conv1(1->6,5x5 valid)", "maxpool(2x2/2)",
                       "conv3(6->16,5x5 valid)", "maxpool(2x2/2)",
                       "conv5(16->120,5x5 valid)", "fc1(120->84)", "fc2(84->10)"],
            "params": TOTAL_PARAMS,
            "macs": TOTAL_MACS,
        },
        "representative_images": {"indices": VEC_INDICES, "labels": labels.tolist()},
        "tensor_shapes": tensor_shapes,
        "tensor_dtypes": tensor_dtypes,
        "quant_param_shapes": qp_shapes,
        "environment": _versions(),
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # ---- SHA256SUMS ----
    artifacts = [out_dir / "quant_params.npz", out_dir / "layer_vectors.npz",
                 manifest_path]
    with open(out_dir / "SHA256SUMS", "w") as f:
        for a in sorted(artifacts):
            f.write(f"{_sha256(a)}  {a.name}\n")

    # ---- L2 frozen manifest ----
    frozen = {
        "freeze_statement": "These are the frozen L2 INT8 hardware-reference artifacts "
                            "for all subsequent dataflow, sparsity, RTL, synthesis, and "
                            "hardware experiments.",
        "frozen_date": "2026-09-23",
        "status": "FROZEN",
        "spec": "docs/LENET5_INT8_SPEC.md (FROZEN, Step 2.1)",
        "l1_checkpoint_sha256": L1_SHA,
        "l1_canonical_weights_sha256": L1_CANON,
        "int8_convention": "symmetric signed INT8, zero-point 0, per-tensor activation "
                           "scales, per-channel weight scales, int32 accumulation/bias, "
                           "canonical RNE requantization",
        "calibration_set": "train[0:1024]",
        "architecture": {"params": TOTAL_PARAMS, "macs": TOTAL_MACS},
        "environment": _versions(),
        "artifacts": {
            a.name: _sha256(a) for a in artifacts
        },
    }
    with open(out_dir / "L2_FROZEN_MANIFEST.json", "w") as f:
        json.dump(frozen, f, indent=2, sort_keys=True)

    print(f"Wrote L2 INT8 artifacts to {out_dir}/")
    for a in sorted(out_dir.glob("*")):
        if a.is_file():
            print(f"  {a.name}  sha256={_sha256(a)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "lenet5_int8")
    args = ap.parse_args()
    generate(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
