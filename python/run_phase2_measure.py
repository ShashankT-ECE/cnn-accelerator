"""Phase 2 research measurements (Part G) — machine-readable results.

Emits ``data/vectors/phase2_results.json`` with, per experiment, the mode,
sparsity, MAC accounting, cycles, throughput, speedup, reconfiguration
overhead, and (once filled in from synthesis) resource/timing figures.

All cycle/latency figures are [SIMULATION] (xsim, Vivado ML 2023.1) from the
verified schedule; no figure here is a fabricated speedup. The dense-vs-sparse
comparison is the genuine result of the coarse zero-group skip on the REAL
MNIST Conv1 vectors (data/vectors/input_img.hex).

The zero-activation MAC count (skipped) is mode-independent and computed here
independently of the RTL (it depends only on the image + geometry).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VECTORS = Path(__file__).resolve().parents[1] / "data" / "vectors"

OC, K, PAD = 8, 5, 2
H = W = 28
TOTAL_MAC = OC * H * W * K * K          # 156,800
CLK_MHZ = 200.0


def load_image() -> np.ndarray:
    img = np.array([int(x, 16) for x in (VECTORS / "input_img.hex").read_text().split()])
    return np.where(img >= 128, img - 256, img).reshape(H, W)


def zero_activation_macs(img: np.ndarray) -> int:
    """Number of MACs whose activation operand is zero (mode-independent)."""
    skip = 0
    for y in range(H):
        for x in range(W):
            for ky in range(K):
                for kx in range(K):
                    iy, ix = y + ky - PAD, x + kx - PAD
                    if iy < 0 or iy >= H or ix < 0 or ix >= W or img[iy, ix] == 0:
                        skip += 1
    return skip * OC


def main() -> None:
    img = load_image()
    skipped = zero_activation_macs(img)
    executed = TOTAL_MAC - skipped

    # Verified cycle counts [SIMULATION] (RTL, xsim, Vivado ML 2023.1).
    # OS: 112 groups x 82 cycles. WS dense: 896 groups x 43 cycles.
    # WS sparse: coarse zero-group skip (60/112 positions all-zero -> skip all 8
    # channel groups; 480 skipped x 1 cycle + 416 computed x 43 cycles).
    os_dense = 112 * 82
    ws_dense = 896 * 43
    ws_sparse = 18368                      # measured, with 1-cycle/skip emit overhead
    flush_cycles = 8                       # reconfiguration flush (mode switch)

    results = {
        "experiment": "Phase 2 — runtime OS/WS reconfiguration + sparsity on real MNIST Conv1",
        "layer": {"in": "28x28x1", "kernel": "5x5", "pad": "SAME_UPPER pad=2",
                  "out": "28x28x8", "macs": TOTAL_MAC},
        "sparsity": {
            "total_mac_opportunities": TOTAL_MAC,
            "zero_activation_macs": int(skipped),
            "executed_macs": int(executed),
            "skip_percent": round(100.0 * skipped / TOTAL_MAC, 2),
            "note": "zero-activation = activation operand == 0 (incl. SAME padding zeros)",
        },
        "experiments": [
            {
                "id": "os_dense", "mode": "OS", "sparse": False,
                "cycles": os_dense,
                "latency_us": round(os_dense / CLK_MHZ, 3),
                "sustained_mac_per_cycle": round(TOTAL_MAC / os_dense, 2),
                "sustained_GOPS": round(TOTAL_MAC * 2 * CLK_MHZ / os_dense / 1e3, 2),
            },
            {
                "id": "ws_dense", "mode": "WS", "sparse": False,
                "cycles": ws_dense,
                "latency_us": round(ws_dense / CLK_MHZ, 3),
                "sustained_mac_per_cycle": round(TOTAL_MAC / ws_dense, 2),
                "sustained_GOPS": round(TOTAL_MAC * 2 * CLK_MHZ / ws_dense / 1e3, 2),
                "provenance": "[ANALYTICAL] no-skip bound (896 groups x 43): the coarse "
                              "zero-skip is always-on in WS mode, so this is not measured on a sparse input",
            },
            {
                "id": "ws_sparse", "mode": "WS", "sparse": True,
                "cycles": ws_sparse,
                "latency_us": round(ws_sparse / CLK_MHZ, 3),
                "sustained_mac_per_cycle": round(TOTAL_MAC / ws_sparse, 2),
                "sustained_GOPS": round(TOTAL_MAC * 2 * CLK_MHZ / ws_sparse / 1e3, 2),
                "speedup_vs_ws_dense": round(ws_dense / ws_sparse, 3),
                "skip_method": "coarse zero-group skip (60/112 output positions all-zero)",
            },
        ],
        "reconfiguration": {
            "flush_cycles": flush_cycles,
            "os_to_ws": "verified bit-exact (no reset)",
            "ws_to_os": "verified bit-exact (no reset)",
            "mode_switch_latency_cycles": flush_cycles,
            "mode_switch_latency_ns": flush_cycles * (1000.0 / CLK_MHZ),
        },
        "synthesis": {
            "label": "[POST-ROUTE SYNTHESIS ESTIMATE]",
            "device": "xck26-sfvc784-2LV-c",
            "tool": "Vivado ML 2023.1",
            "clock_mhz": 200.0,
            "wns_ns": 0.189, "tns_ns": 0.0, "whs_ns": 0.073, "ths_ns": 0.0,
            "lut": 9372, "ff": 5795, "dsp48e2": 64, "carry8": 20,
            "bram": 0, "uram": 0,
            "critical_path": "s counter -> address look-ahead -> image dist-RAM -> stream reg",
            "note": "HARDENED: WNS +0.030 -> +0.189 ns (6.3x); -3 DSP, -1873 LUT; ~208 MHz max",
        },
        "labels": "all figures [SIMULATION] except where marked; no fabricated speedup",
    }

    out = VECTORS / "phase2_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
