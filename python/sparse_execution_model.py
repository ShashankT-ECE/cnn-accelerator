#!/usr/bin/env python3
"""Sparse execution / cycle model (Step 5.2).

Models three cases for the frozen LeNet-5 OS baseline (43,021 cycles):

  1. dense   — frozen Step 4.3 OS execution (mapping_model).
  2. coarse  — group-level skip: a whole schedulable OS *spatial* group is
               skipped only when its entire activation window (K rows x
               (P+K-1) cols x IC channels) is zero — the zero information the
               scheduler actually has at the group boundary. Partial groups are
               not assumed skippable. FC layers are all-or-nothing (the whole
               feature vector is shared), so they are effectively never skipped.
  3. fine    — per-MAC upper bound only (59.08% theoretical MAC reduction from
               sparsity_model); NOT claimed achievable on the current PE.

Honest: does not inflate the coarse skip. Machine-readable:
data/benchmark/sparse_execution_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from torchvision import datasets  # noqa: E402

from mapping_model import LAYERS, os_map  # noqa: E402
from lenet5.int8_model import Int8LeNet5  # noqa: E402
from lenet5.preprocess import make_transform  # noqa: E402

OUT = REPO_ROOT / "data" / "benchmark" / "sparse_execution_results.json"
TEST_N = 10000
BATCH = 512

# layer -> (input tensor key, K, IC, OC, output W) for the conv group-window test.
CONV = {
    "conv1": ("q_input", 5, 1, 6, 28),
    "conv3": ("pool1", 5, 6, 16, 10),
    "conv5": ("pool2", 5, 16, 120, 1),
}
FC = {"fc1": ("c5", 120, 84), "fc2": ("f6", 84, 10)}


def spatial_groups(W_out: int) -> list[tuple[int, int]]:
    """(left, right) inclusive output-pixel ranges, 8 per group, last partial."""
    return [(l, min(l + 7, W_out - 1)) for l in range(0, W_out, 8)]


def load_model():
    qp = np.load(REPO_ROOT / "data" / "lenet5_int8" / "quant_params.npz")
    params = {"S_input": float(qp["S_input"])}
    for n in ["conv1", "conv3", "conv5", "fc1", "fc2"]:
        L = {"S_a": float(qp[f"{n}_S_a"]), "S_w": qp[f"{n}_S_w"],
             "q_w": qp[f"{n}_q_w"], "q_b": qp[f"{n}_q_b"]}
        if f"{n}_S_out" in qp.files:
            L["S_out"] = float(qp[f"{n}_S_out"]); L["M"] = qp[f"{n}_M"]
        params[n] = L
    return Int8LeNet5(params)


def main() -> int:
    model = load_model()
    tr = make_transform()
    test_ds = datasets.MNIST(root=str(REPO_ROOT / "data" / "raw"), train=False,
                             download=False, transform=tr)

    dense = {L["name"]: os_map(L) for L in LAYERS}
    dense_total = sum(dense[n]["cycles"] for n in dense)

    # per-layer skip bookkeeping
    conv_groups = {n: spatial_groups(CONV[n][4]) for n in CONV}
    skip_groups = {n: 0 for n in CONV}     # count of all-zero-window spatial groups
    skip_fc = {n: 0 for n in FC}           # count of fully-zero feature vectors
    skip_macs = {n: 0 for n in CONV}       # MACs in skipped conv groups
    skip_macs_fc = {n: 0 for n in FC}

    for s in range(0, TEST_N, BATCH):
        idx = list(range(s, min(s + BATCH, TEST_N)))
        x = np.stack([test_ds[i][0].numpy() for i in idx]).astype(np.float32)
        out = model.forward_layers(x)
        for name, (key, K, IC, OC, W_out) in CONV.items():
            A = out[key]                        # (N, IC, H, W)
            H = A.shape[2]
            for y in range(H - K + 1):
                for left, right in conv_groups[name]:
                    win = A[:, :, y:y + K, left:right + K]   # (N, IC, K, P+K-1)
                    n_skip = int((win == 0).all(axis=(1, 2, 3)).sum())
                    skip_groups[name] += n_skip
                    P = right - left + 1
                    skip_macs[name] += n_skip * P * OC * K * K * IC
        for name, (key, IC, OC) in FC.items():
            A = out[key].reshape(out[key].shape[0], -1)       # (N, IC)
            n_skip = int((A == 0).all(axis=1).sum())
            skip_fc[name] += n_skip
            skip_macs_fc[name] += n_skip * IC * OC

    # cycle accounting (per image)
    results = {"dense": {}, "coarse": {}, "fine": {}, "totals": {}}
    for name in ["conv1", "conv3", "conv5"]:
        L = LAYERS[[l["name"] for l in LAYERS].index(name)]
        H_out, W_out = L["H"], L["W"]
        n_spatial = H_out * ((W_out + 7) // 8)
        per_group_cycles = dense[name]["cycles"] / n_spatial       # fixed group cost
        skipped = skip_groups[name] / TEST_N                        # avg groups skipped/img
        results["dense"][name] = dense[name]["cycles"]
        results["coarse"][name] = {
            "spatial_groups_per_image": n_spatial,
            "skipped_groups_per_image": skipped,
            "skip_rate_pct": skipped / n_spatial * 100.0,
            "dense_cycles": dense[name]["cycles"],
            "sparse_cycles": dense[name]["cycles"] - skipped * per_group_cycles,
            "skipped_cycles": skipped * per_group_cycles,
            "skipped_macs_per_image": skip_macs[name] / TEST_N,
        }
    for name in ["fc1", "fc2"]:
        skipped = skip_fc[name] / TEST_N
        results["dense"][name] = dense[name]["cycles"]
        results["coarse"][name] = {
            "spatial_groups_per_image": 1,
            "skipped_groups_per_image": skipped,
            "skip_rate_pct": skipped * 100.0,
            "dense_cycles": dense[name]["cycles"],
            "sparse_cycles": dense[name]["cycles"] - skipped * dense[name]["cycles"],
            "skipped_cycles": skipped * dense[name]["cycles"],
            "skipped_macs_per_image": skip_macs_fc[name] / TEST_N,
        }

    # fine (per-MAC) upper bound — no cycle speedup claimed.
    spr = json.loads((REPO_ROOT / "data" / "benchmark" / "sparsity_results.json").read_text())
    fine_reduction_pct = spr["totals"]["theoretical_mac_reduction_pct"]

    coarse_sparse_total = sum(results["coarse"][n]["sparse_cycles"] for n in results["coarse"])
    coarse_skipped_macs = sum(results["coarse"][n]["skipped_macs_per_image"] for n in results["coarse"])
    results["totals"] = {
        "dense_cycles": dense_total,
        "coarse_sparse_cycles": coarse_sparse_total,
        "coarse_speedup_vs_dense": dense_total / coarse_sparse_total,
        "coarse_skipped_macs_per_image": coarse_skipped_macs,
        "fine_theoretical_mac_reduction_pct": fine_reduction_pct,
        "fine_note": "per-MAC upper bound only; no cycle speedup claimed on the current PE",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    print("=" * 80)
    print("Sparse execution model — coarse group skip vs dense OS (per image)")
    print("=" * 80)
    print(f"{'layer':6} {'dense':>7} {'groups':>7} {'skipped':>8} {'skip%':>7} {'sparse':>7} {'saved':>7}")
    print("-" * 80)
    for n in results["coarse"]:
        r = results["coarse"][n]
        print(f"{n:6} {r['dense_cycles']:>7} {r['spatial_groups_per_image']:>7.0f} "
              f"{r['skipped_groups_per_image']:>8.1f} {r['skip_rate_pct']:>6.2f}% "
              f"{r['sparse_cycles']:>7.0f} {r['skipped_cycles']:>7.0f}")
    print("-" * 80)
    t = results["totals"]
    print(f"{'TOTAL':6} {t['dense_cycles']:>7} {'':>7} {'':>8} {'':>7} "
          f"{t['coarse_sparse_cycles']:>7.0f} {t['dense_cycles']-t['coarse_sparse_cycles']:>7.0f}")
    print("=" * 80)
    print(f"Coarse speedup vs dense OS : {t['coarse_speedup_vs_dense']:.4f}x")
    print(f"Fine per-MAC reduction (upper bound) : {t['fine_theoretical_mac_reduction_pct']:.2f}%  [no cycle speedup claimed]")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
