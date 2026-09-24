#!/usr/bin/env python3
"""CIFAR-10 activation-sparsity + coarse-skip measurement (Risk R7).

Answers, empirically over the frozen CIFAR-10 test split, whether the V1 coarse
group-level zero-skip mechanism provides a meaningful cycle reduction on
CIFAR-10, or is pure overhead.

Why this cannot be assumed from MNIST: CIFAR-10 uses dense, unpadded RGB input
(no background, no zero-pad), which removes the *structural* zeros that drove
the MNIST coarse-skip win; CIFAR's ReLU zeros live in the deep layers and are
spatially scattered (the LeNet-5 finding, `docs/SPARSE_EXECUTION_RESULTS.md`).

Measures, per layer, the tensor actually fed into the MACs:
  * zero% of the activation operand (structural for conv1's raw RGB; ReLU for
    conv2/conv3/fc),
  * the fine per-MAC "skippable" fraction (theoretical upper bound only),
  * the *coarse* group-level skip — the only mechanism the frozen 8x8 array can
    realize — and its cycle reduction against the dense OS baseline.

Deterministic (fixed indices, no RNG). Machine-readable:
data/benchmark/cifar10_sparsity_results.json

Run from the repo root:
  .venv/bin/python python/measure_cifar_sparsity.py            # full 10k test set
  .venv/bin/python python/measure_cifar_sparsity.py 512        # smoke test (N images)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from torchvision import datasets  # noqa: E402

from cifar10.int8_model import Int8Cifar10Net, calibrate, load_weights  # noqa: E402
from cifar10.preprocess import make_transform  # noqa: E402
from cifar10_mapping_model import LAYERS, os_map  # noqa: E402

OUT = REPO_ROOT / "data" / "benchmark" / "cifar10_sparsity_results.json"
CALIB_N = 1024     # frozen calibration split (matches eval_cifar10_int8.py)
TEST_N = 10000     # frozen CIFAR-10 test split
BATCH = 128

# Layer -> (input tensor key, kind, OC, H_in, W_in, K) for the activation operand.
CONV = [
    ("conv1", "q_input", "structural", 32, 32, 32, 5),
    ("conv2", "pool1",    "relu",       32, 14, 14, 5),
    ("conv3", "pool2",    "relu",       64, 5,  5,  5),
]
FC = [("fc", "c3", "relu", 10, 64)]


def num_uses(H: int, W: int, K: int) -> np.ndarray:
    """(H,W): number of (y,x,ky,kx) MAC taps that read A[i,j] in a valid KxK conv."""
    Hp, Wp = H - K + 1, W - K + 1
    ny = np.array([min(i, Hp - 1) - max(0, i - K + 1) + 1 for i in range(H)], dtype=np.int64)
    nx = np.array([min(j, Wp - 1) - max(0, j - K + 1) + 1 for j in range(W)], dtype=np.int64)
    return ny[:, None] * nx[None, :]


def spatial_groups(W_out: int) -> list[tuple[int, int]]:
    """(left, right) inclusive output-pixel ranges, 8 per group, last partial."""
    return [(l, min(l + 7, W_out - 1)) for l in range(0, W_out, 8)]


def main() -> int:
    test_n = int(sys.argv[1]) if len(sys.argv) > 1 else TEST_N

    # Dense OS baseline (frozen CIFAR mapping, OS-only).
    dense = {L["name"]: os_map(L) for L in LAYERS}
    dense_total = sum(dense[n]["cycles"] for n in dense)

    # ---- Calibrate the frozen CIFAR INT8 reference ------------------------
    ckpt = REPO_ROOT / "data" / "checkpoint" / "cifar10_fp32.pt"
    tr = make_transform()
    train_ds = datasets.CIFAR10(root=str(REPO_ROOT / "data" / "raw"), train=True,
                                download=False, transform=tr)
    test_ds = datasets.CIFAR10(root=str(REPO_ROOT / "data" / "raw"), train=False,
                               download=False, transform=tr)
    print(f"Calibrating on train[0:{CALIB_N}] ...")
    calib = np.stack([train_ds[i][0].numpy() for i in range(CALIB_N)]).astype(np.float32)
    params = calibrate(load_weights(ckpt), calib)
    model = Int8Cifar10Net(params)

    # ---- Bookkeeping ------------------------------------------------------
    nu = {n: num_uses(H, W, K) for n, _, _, OC, H, W, K in CONV}
    W_out = {n: W - K + 1 for n, _, _, OC, H, W, K in CONV}
    conv_groups = {n: spatial_groups(W_out[n]) for n, *_ in CONV}

    zero = {n: 0 for n, *_ in CONV}
    total = {n: 0 for n, *_ in CONV}
    skip_macs = {n: 0 for n, *_ in CONV}          # fine per-MAC skippable (sum over imgs)
    skip_groups = {n: 0 for n, *_ in CONV}        # all-zero-window spatial groups (sum)
    skip_group_macs = {n: 0 for n, *_ in CONV}    # MACs inside skipped groups (sum)

    zero_fc = {n: 0 for n, *_ in FC}
    total_fc = {n: 0 for n, *_ in FC}
    skip_macs_fc = {n: 0 for n, *_ in FC}
    skip_groups_fc = {n: 0 for n, *_ in FC}

    # ---- Forward over the test set ----------------------------------------
    print(f"Forward over test[0:{test_n}] ...")
    for s in range(0, test_n, BATCH):
        idx = list(range(s, min(s + BATCH, test_n)))
        x = np.stack([test_ds[i][0].numpy() for i in idx]).astype(np.float32)
        out = model.forward_layers(x)

        for n, key, kind, OC, H, W, K in CONV:
            A = out[key].astype(np.int8)                       # (N, IC, H, W)
            z = (A == 0).astype(np.int64)
            zero[n] += int(z.sum())
            total[n] += int(A.size)
            skip_macs[n] += int(np.einsum('nihw,hw->ni', z, nu[n]).sum() * OC)
            # coarse: whole OS spatial group skipped iff its Kx(P+K-1)xIC window is 0
            for y in range(H - K + 1):
                for left, right in conv_groups[n]:
                    win = A[:, :, y:y + K, left:right + K]      # (N, IC, K, P+K-1)
                    n_skip = int((win == 0).all(axis=(1, 2, 3)).sum())
                    skip_groups[n] += n_skip
                    P = right - left + 1
                    skip_group_macs[n] += n_skip * P * OC * K * K * (win.shape[1])

        for n, key, kind, OC, IC in FC:
            A = out[key].reshape(out[key].shape[0], -1).astype(np.int8)  # (N, IC)
            z = (A == 0).astype(np.int64)
            zero_fc[n] += int(z.sum())
            total_fc[n] += int(A.size)
            skip_macs_fc[n] += int(z.sum() * OC)
            skip_groups_fc[n] += int((A == 0).all(axis=1).sum())

    # ---- Per-layer results ------------------------------------------------
    results = {"calibration": CALIB_N, "test_images": test_n,
               "dense_os": {}, "coarse": {}, "fine": {}, "layers": {}}

    total_macs = {L["name"]: L["MACs"] for L in LAYERS}
    for n, key, kind, OC, H, W, K in CONV:
        L = LAYERS[[l["name"] for l in LAYERS].index(n)]
        H_out, W_out_l = L["H"], L["W"]
        n_spatial = H_out * ((W_out_l + 7) // 8)
        per_group = dense[n]["cycles"] / n_spatial
        skipped = skip_groups[n] / test_n
        results["layers"][n] = {
            "tensor": key, "kind": kind, "output_channels": OC,
            "zero_pct": zero[n] / total[n] * 100.0,
            "fine_skippable_pct": skip_macs[n] / (total_macs[n] * test_n) * 100.0,
            "coarse_skipped_groups_per_image": skipped,
            "coarse_skip_rate_pct": skipped / n_spatial * 100.0,
        }
        results["dense_os"][n] = dense[n]["cycles"]
        results["coarse"][n] = {
            "spatial_groups_per_image": n_spatial,
            "dense_cycles": dense[n]["cycles"],
            "skipped_cycles": skipped * per_group,
            "sparse_cycles": dense[n]["cycles"] - skipped * per_group,
            "skipped_macs_per_image": skip_group_macs[n] / test_n,
        }
        results["fine"][n] = {
            "skippable_macs_per_image": skip_macs[n] / test_n,
            "skippable_pct": skip_macs[n] / (total_macs[n] * test_n) * 100.0,
        }
    for n, key, kind, OC, IC in FC:
        L = LAYERS[[l["name"] for l in LAYERS].index(n)]
        skipped = skip_groups_fc[n] / test_n
        results["layers"][n] = {
            "tensor": key, "kind": kind, "output_channels": OC,
            "zero_pct": zero_fc[n] / total_fc[n] * 100.0,
            "fine_skippable_pct": skip_macs_fc[n] / (total_macs[n] * test_n) * 100.0,
            "coarse_skipped_groups_per_image": skipped,
            "coarse_skip_rate_pct": skipped * 100.0,
        }
        results["dense_os"][n] = dense[n]["cycles"]
        results["coarse"][n] = {
            "spatial_groups_per_image": 1,
            "dense_cycles": dense[n]["cycles"],
            "skipped_cycles": skipped * dense[n]["cycles"],
            "sparse_cycles": dense[n]["cycles"] - skipped * dense[n]["cycles"],
            "skipped_macs_per_image": skip_groups_fc[n] / test_n * IC * OC,
        }
        results["fine"][n] = {
            "skippable_macs_per_image": skip_macs_fc[n] / test_n,
            "skippable_pct": skip_macs_fc[n] / (total_macs[n] * test_n) * 100.0,
        }

    coarse_total = sum(results["coarse"][n]["sparse_cycles"] for n in results["coarse"])
    coarse_skipped_macs = sum(results["coarse"][n]["skipped_macs_per_image"] for n in results["coarse"])
    fine_skipped_macs = sum(results["fine"][n]["skippable_macs_per_image"] for n in results["fine"])
    results["totals"] = {
        "dense_os_cycles": dense_total,
        "coarse_sparse_cycles": coarse_total,
        "coarse_speedup_vs_dense": dense_total / coarse_total,
        "coarse_cycle_reduction_pct": (dense_total - coarse_total) / dense_total * 100.0,
        "coarse_skipped_macs_per_image": coarse_skipped_macs,
        "fine_skippable_macs_per_image": fine_skipped_macs,
        "fine_theoretical_mac_reduction_pct": fine_skipped_macs / sum(total_macs.values()) * 100.0,
    }

    # ---- Definitive verdict ----------------------------------------------
    red_pct = results["totals"]["coarse_cycle_reduction_pct"]
    results["verdict"] = {
        "threshold_pct": 5.0,
        "coarse_cycle_reduction_pct": red_pct,
        "meaningful": red_pct > 5.0,
        "conclusion": (
            f"coarse-skip provides a {red_pct:.2f}% cycle reduction (> 5% => meaningful)"
            if red_pct > 5.0
            else f"coarse-skip provides only {red_pct:.2f}% cycle reduction "
                 f"(<= 5% => pure overhead on CIFAR-10)"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    # ---- Human-readable table ---------------------------------------------
    print("=" * 94)
    print(f"CIFAR-10 activation sparsity + coarse-skip ({test_n} test images)")
    print("=" * 94)
    print(f"{'layer':6} {'kind':10} {'zero%':>7} {'fine skip%':>10} "
          f"{'groups':>7} {'skipped':>8} {'skip%':>7} | {'dense':>8} {'sparse':>9}")
    print("-" * 94)
    for n in results["layers"]:
        lr = results["layers"][n]
        cr = results["coarse"][n]
        print(f"{n:6} {lr['kind']:10} {lr['zero_pct']:>6.2f}% "
              f"{lr['fine_skippable_pct']:>9.2f}% "
              f"{cr['spatial_groups_per_image']:>7.0f} "
              f"{lr['coarse_skipped_groups_per_image']:>8.2f} "
              f"{lr['coarse_skip_rate_pct']:>6.2f}% | {cr['dense_cycles']:>8.0f} {cr['sparse_cycles']:>9.1f}")
    print("-" * 94)
    t = results["totals"]
    print(f"{'TOTAL':6} {'':10} {'':>7} {t['fine_theoretical_mac_reduction_pct']:>9.2f}% "
          f"{'':>7} {'':>8} {'':>7} | {t['dense_os_cycles']:>8.0f} {t['coarse_sparse_cycles']:>9.1f}")
    print("=" * 94)
    print(f"Coarse-skip cycle reduction : {t['coarse_cycle_reduction_pct']:.2f}%")
    print(f"Coarse speedup vs dense OS  : {t['coarse_speedup_vs_dense']:.4f}x")
    print(f"Fine per-MAC reduction (upper bound) : {t['fine_theoretical_mac_reduction_pct']:.2f}%")
    print(f"VERDICT: {results['verdict']['conclusion']}")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
