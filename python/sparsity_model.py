#!/usr/bin/env python3
"""Activation-sparsity measurement for the frozen LeNet-5 INT8 workload (Step 5.1).

Measures, over the frozen 10k MNIST test split with the L1/L2 preprocessing and
the calibrated INT8 model, the zero-activation statistics of the tensors actually
fed into MAC operations — and the per-MAC "theoretically skippable" count.

Distinguishes:
  * structural zeros — Conv1's raw-image input (background + zero-pad-2), which
    are NOT ReLU sparsity,
  * ReLU zeros — the post-ReLU activations feeding Conv3/Conv5/FC1/FC2.

Deterministic (fixed indices, no RNG). Machine-readable: data/benchmark/sparsity_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from torchvision import datasets  # noqa: E402

from lenet5.int8_model import Int8LeNet5, calibrate, load_weights  # noqa: E402
from lenet5.preprocess import make_transform  # noqa: E402

OUT = REPO_ROOT / "data" / "benchmark" / "sparsity_results.json"
CALIB_N = 1024
TEST_N = 10000
BATCH = 512

# Tensors fed into MAC operations, and their sparsity provenance.
LAYERS = [
    ("conv1", "q_input", "structural", 6),   # raw image (no ReLU before Conv1)
    ("conv3", "pool1",    "relu",       16),
    ("conv5", "pool2",    "relu",       120),
    ("fc1",  "c5",        "relu",       84),   # Conv5 post-ReLU (flattened)
    ("fc2",  "f6",        "relu",       10),   # FC1 post-ReLU
]


def num_uses(H: int, W: int, K: int) -> np.ndarray:
    """(H,W) array: number of (y,x,ky,kx) MAC taps that read A[i,j] in a valid KxK conv."""
    Hp, Wp = H - K + 1, W - K + 1
    ny = np.array([min(i, Hp - 1) - max(0, i - K + 1) + 1 for i in range(H)], dtype=np.int64)
    nx = np.array([min(j, Wp - 1) - max(0, j - K + 1) + 1 for j in range(W)], dtype=np.int64)
    return ny[:, None] * nx[None, :]


def main() -> int:
    ckpt = REPO_ROOT / "data" / "checkpoint" / "lenet5_fp32.pt"
    tr = make_transform()
    train_ds = datasets.MNIST(root=str(REPO_ROOT / "data" / "raw"), train=True,
                              download=False, transform=tr)
    calib = np.stack([train_ds[i][0].numpy() for i in range(CALIB_N)]).astype(np.float32)
    params = calibrate(load_weights(ckpt), calib)
    model = Int8LeNet5(params)

    test_ds = datasets.MNIST(root=str(REPO_ROOT / "data" / "raw"), train=False,
                             download=False, transform=tr)

    # num_uses for the three conv input geometries.
    nu = {"conv1": num_uses(32, 32, 5), "conv3": num_uses(14, 14, 5), "conv5": num_uses(5, 5, 5)}

    zero = {n: 0 for n, _, _, _ in LAYERS}
    total = {n: 0 for n, _, _, _ in LAYERS}
    skip = {n: 0 for n, _, _, _ in LAYERS}

    for s in range(0, TEST_N, BATCH):
        idx = list(range(s, min(s + BATCH, TEST_N)))
        x = np.stack([test_ds[i][0].numpy() for i in idx]).astype(np.float32)
        out = model.forward_layers(x)

        for name, key, kind, OC in LAYERS:
            t = out[key]
            if name.startswith("conv"):
                # conv input tensor (N, IC, H, W); skippable = OC * sum(zero * num_uses)
                z = (t == 0).astype(np.int64)
                zero[name] += int(z.sum())
                total[name] += int(t.size)
                skip[name] += int(np.einsum('nihw,hw->ni', z, nu[name]).sum() * OC)
            else:
                # FC input tensor (N, IC); skippable = OC * zero_count
                t = t.reshape(t.shape[0], -1)
                z = (t == 0).astype(np.int64)
                zero[name] += int(z.sum())
                total[name] += int(t.size)
                skip[name] += int(z.sum() * OC)

    total_macs = {"conv1": 117600, "conv3": 240000, "conv5": 48000, "fc1": 10080, "fc2": 840}
    results = {"layers": {}, "totals": {}}
    for name, key, kind, OC in LAYERS:
        results["layers"][name] = {
            "tensor": key, "kind": kind, "output_channels": OC,
            "zero_count": zero[name], "total_elements": total[name],
            "zero_pct": zero[name] / total[name] * 100.0,
            "nonzero_pct": (total[name] - zero[name]) / total[name] * 100.0,
            "total_macs_per_image": total_macs[name],
            "skippable_macs_per_image": skip[name] / TEST_N,
            "skippable_pct": skip[name] / (total_macs[name] * TEST_N) * 100.0,
        }
    t_macs = sum(total_macs.values())
    t_skip = sum(skip.values())
    results["totals"] = {
        "total_macs_per_image": t_macs,
        "skippable_macs_per_image": t_skip / TEST_N,
        "theoretical_mac_reduction_pct": t_skip / (t_macs * TEST_N) * 100.0,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    print("=" * 86)
    print("Activation sparsity (frozen MNIST test set, 10k, INT8, per-layer input)")
    print("=" * 86)
    print(f"{'layer':6} {'kind':10} {'zero_cnt':>9} {'total':>9} {'zero%':>7} {'skipMACs':>9} {'skip%':>7}")
    print("-" * 86)
    for name, key, kind, OC in LAYERS:
        r = results["layers"][name]
        print(f"{name:6} {kind:10} {r['zero_count']:>9} {r['total_elements']:>9} "
              f"{r['zero_pct']:>6.2f}% {r['skippable_macs_per_image']:>9.0f} {r['skippable_pct']:>6.2f}%")
    print("-" * 86)
    t = results["totals"]
    print(f"{'TOTAL':6} {'':10} {'':>9} {'':>9} {'':>7} {t['skippable_macs_per_image']:>9.0f} "
          f"{t['theoretical_mac_reduction_pct']:>6.2f}%  (of {t['total_macs_per_image']})")
    print("=" * 86)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
