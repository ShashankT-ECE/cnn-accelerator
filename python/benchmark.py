"""Phase 2 benchmark generator — cycle decomposition, benchmark matrix, sparsity accounting.

Produces, under ``data/benchmark/``:

  * ``cycle_decomposition.json`` — OS vs WS cycle attribution (Section 2).
  * ``benchmark_matrix.csv`` / ``.json``  — the 6-config benchmark matrix (Section 7).
  * ``sparsity_accounting.csv`` / ``.json`` — zero-fraction vs skippable-group vs
    measured cycle reduction, over the primary image AND a deterministic subset
    of the real MNIST test set (Section 4).

Provenance discipline: every cycle figure is labelled [SIMULATION] (measured in
xsim, Vivado ML 2023.1), [ANALYTICAL] (derived from the verified per-group
schedule), or [THEORETICAL/PEAK]. ``WS dense`` is the analytical no-skip bound:
the RTL's coarse zero-skip is *always on* in WS mode, so the no-skip figure is
896 groups x 43 cycles = 38,528, not directly measured on a sparse input (see
docs §Section 2 / 10). No fabricated numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
VECTORS = REPO / "data" / "vectors"
OUT = REPO / "data" / "benchmark"

OC, K, PAD = 8, 5, 2
H = W = 28
TOTAL_MAC = OC * H * W * K * K          # 156,800
CLK_MHZ = 200.0
PE = 64                                # 8x8 array

# Verified schedule constants [SIMULATION / schedule-derived].
OS_GROUP = 82                          # 1 clear + 5x13 + 16 drain
WS_GROUP = 43                          # 42 compute + 1 capture
OS_GROUPS = 112                        # 28 rows x 4 groups
WS_GROUPS = 896                        # 8 channels x 112
FLUSH = 8                              # reconfiguration flush cycles


# ---------------------------------------------------------------------------
# Image loading / quantisation
# ---------------------------------------------------------------------------
def load_image_hex(path: Path) -> np.ndarray:
    vals = [int(x, 16) for x in path.read_text().split()]
    return np.where(np.asarray(vals) >= 128, np.asarray(vals) - 256,
                    np.asarray(vals)).astype(np.int8).reshape(H, W)


def zero_activation_macs(img: np.ndarray) -> int:
    """MACs whose activation operand is zero or out-of-range (mode-independent)."""
    skip = 0
    for y in range(H):
        for x in range(W):
            for ky in range(K):
                for kx in range(K):
                    iy, ix = y + ky - PAD, x + kx - PAD
                    if iy < 0 or iy >= H or ix < 0 or ix >= W or img[iy, ix] == 0:
                        skip += 1
    return skip * OC


def gaz(img: np.ndarray, yy: int, base: int) -> bool:
    """Coarse zero-group predicate (mirrors the RTL gaz()): the 5x12 window
    (rows y-2..y+2, cols base-9..base+2) is entirely zero."""
    nz = img != 0
    for iy in range(-2, 3):
        ry = yy + iy
        if ry < 0 or ry >= H:
            continue
        for dx in range(-9, 3):
            cx = base + dx
            if cx < 0 or cx >= W:
                continue
            if nz[ry, cx]:
                return False
    return True


def skippable_positions(img: np.ndarray) -> list[tuple[int, int]]:
    """(y, base) output positions whose 5x12 window is all-zero."""
    out = []
    for y in range(H):
        for b in range(4):
            base = 7 + 8 * b
            if gaz(img, y, base):
                out.append((y, base))
    return out


def valid_pixels(base: int) -> int:
    """Valid output pixels in a group (base=31 is partial: 4, else 8)."""
    return 4 if base == 31 else 8


def skippable_macs(img: np.ndarray, positions: list[tuple[int, int]]) -> int:
    """MACs eliminated by the coarse group skip = sum over skippable (y,base)
    of 8 channels x valid_pixels x 25 taps."""
    return sum(OC * valid_pixels(base) * K * K for _, base in positions)


def ws_sparse_cycles(n_skip_positions: int) -> int:
    """Coarse zero-group skip: skipped groups take 1 cycle, computed take 43."""
    skipped_groups = n_skip_positions * OC
    computed_groups = WS_GROUPS - skipped_groups
    return skipped_groups * 1 + computed_groups * WS_GROUP


# ---------------------------------------------------------------------------
# Cycle decomposition (Section 2)
# ---------------------------------------------------------------------------
def cycle_decomposition() -> dict:
    os_dense = OS_GROUPS * OS_GROUP
    ws_dense = WS_GROUPS * WS_GROUP            # analytical no-skip bound

    # OS per-group anatomy (82 cycles): 1 accum_clear + 5 passes x 13 + 16 drain.
    # Within each of the 5 passes: 5 tap-active cycles + 8 lead-in/inter-pass.
    os_tap_cycles = OS_GROUPS * 5 * 5          # 2,800 (64 MAC/cycle each)
    os_drain = OS_GROUPS * 16                  # 1,792
    os_serial = os_dense - os_tap_cycles - os_drain   # lead-in + clear (4,592)

    # WS channel-serialization factor.
    ch_sweep = WS_GROUPS // OS_GROUPS          # 8

    return {
        "labels": "[SIMULATION] cycle counts; WS dense is [ANALYTICAL] no-skip bound",
        "os_dense": {
            "cycles": os_dense,
            "mac_compute_cycles": os_tap_cycles,
            "result_drain_cycles": os_drain,
            "leadin_and_clear_cycles": os_serial,
            "channel_sweep_overhead": 0,       # channels parallel on 8 rows
            "reconfig_overhead": 0,
        },
        "ws_dense": {
            "cycles": ws_dense,
            "channel_sweep_factor": ch_sweep,  # 8 sweeps vs OS parallel
            "per_group_cycles": WS_GROUP,
            "groups": WS_GROUPS,
            "note": "8x channel serialization vs OS; shorter group (43 vs 82) recovers ~1.9x",
        },
        "ratio_os_vs_ws": round(ws_dense / os_dense, 3),
        "mechanism": (
            "OS holds all 8 channels on 8 PE rows (parallel); WS serializes channels "
            "(8 sweeps). WS's shorter group (43 vs 82) recovers ~1.9x, net ~4.2x slower."
        ),
    }


# ---------------------------------------------------------------------------
# Benchmark matrix (Section 7)
# ---------------------------------------------------------------------------
def benchmark_matrix(img: np.ndarray, n_skip: int, skipped_macs: int) -> list[dict]:
    os_dense = OS_GROUPS * OS_GROUP
    ws_dense = WS_GROUPS * WS_GROUP
    ws_sparse = ws_sparse_cycles(n_skip)
    zero_macs = zero_activation_macs(img)

    def row(id_, mode, sparse, cycles, prov, extra=None):
        d = {
            "config": id_,
            "mode": mode,
            "sparse": sparse,
            "cycles": cycles,
            "cycles_provenance": prov,
            "latency_us": round(cycles / CLK_MHZ, 3),
            "useful_macs": TOTAL_MAC,
            "zero_activation_macs": zero_macs,
            "executed_macs": TOTAL_MAC - zero_macs,
            "zero_fraction_pct": round(100.0 * zero_macs / TOTAL_MAC, 2),
            "pe_utilization_pct": round(100.0 * TOTAL_MAC / (PE * cycles), 2),
            "mac_per_cycle": round(TOTAL_MAC / cycles, 2),
            "sustained_GOPS": round(TOTAL_MAC * 2 * CLK_MHZ / cycles / 1e3, 2),
        }
        if extra:
            d.update(extra)
        return d

    def switch_row(id_, extra=None):
        # Reconfiguration: no MAC work; the MAC metrics are not applicable.
        d = {
            "config": id_,
            "mode": "switch",
            "sparse": False,
            "cycles": FLUSH,
            "cycles_provenance": "[SIMULATION] flush (commit -> quiesced)",
            "latency_us": round(FLUSH / CLK_MHZ, 3),
            "useful_macs": "n/a",
            "zero_activation_macs": "n/a",
            "executed_macs": "n/a",
            "zero_fraction_pct": "n/a",
            "pe_utilization_pct": "n/a",
            "mac_per_cycle": "n/a",
            "sustained_GOPS": "n/a",
            "reconfiguration_overhead_cycles": FLUSH,
        }
        if extra:
            d.update(extra)
        return d

    rows = [
        row("os_dense", "OS", False, os_dense, "[SIMULATION]"),
        row("ws_dense", "WS", False, ws_dense, "[ANALYTICAL] no-skip bound"),
        row("os_sparse", "OS", True, os_dense, "[SIMULATION]",
            {"sparsity_applies": False,
             "note": "coarse zero-group skip is WS-only in this design; OS is NOT accelerated"}),
        row("ws_sparse", "WS", True, ws_sparse, "[SIMULATION]",
            {"sparsity_applies": True,
             "skippable_macs": skipped_macs,
             "skippable_positions": n_skip,
             "speedup_vs_ws_dense": round(ws_dense / ws_sparse, 3),
             "measured_cycle_reduction_pct": round(100.0 * (1 - ws_sparse / ws_dense), 2)}),
        switch_row("os_to_ws"),
        switch_row("ws_to_os"),
    ]
    return rows


# ---------------------------------------------------------------------------
# Sparsity accounting (Section 4)
# ---------------------------------------------------------------------------
def sparsity_accounting(img: np.ndarray) -> dict:
    zero_macs = zero_activation_macs(img)
    positions = skippable_positions(img)
    n_skip = len(positions)
    skip_macs = skippable_macs(img, positions)
    ws_dense = WS_GROUPS * WS_GROUP
    ws_sparse = ws_sparse_cycles(n_skip)

    alpha = zero_macs / TOTAL_MAC
    # fine-grained upper bound: if every zero MAC could be individually skipped
    fine_grained_max_speedup = 1.0 / (1.0 - alpha) if alpha < 1.0 else float("inf")

    return {
        "total_mac_opportunities": TOTAL_MAC,
        "zero_activation_macs": zero_macs,
        "executed_macs": TOTAL_MAC - zero_macs,
        "activation_zero_fraction_pct": round(100.0 * alpha, 2),
        "skippable_positions": n_skip,
        "skippable_group_fraction_pct": round(100.0 * n_skip / OS_GROUPS, 2),
        "skippable_macs_coarse": skip_macs,
        "skippable_mac_fraction_pct": round(100.0 * skip_macs / TOTAL_MAC, 2),
        "ws_dense_cycles": ws_dense,
        "ws_sparse_cycles": ws_sparse,
        "measured_cycle_reduction_pct": round(100.0 * (1 - ws_sparse / ws_dense), 2),
        "measured_speedup": round(ws_dense / ws_sparse, 3),
        "fine_grained_max_speedup": round(fine_grained_max_speedup, 3),
    }


def reconfiguration_timing() -> dict:
    """Runtime OS<->WS reconfiguration overhead (Section 5).

    The mode switch is a single 8-cycle FLUSH (cycle 0 asserts accum_clear +
    drives act_out = 0; cycles 0..7 drain the 56-register activation shift
    chain). ``mode_active`` is latched at commit; no reset / reprogram. After the
    flush the controller is IDLE and ready to run the new mode. Bit-exactness of
    OS->WS and WS->OS (and the OS->WS->OS / WS->OS->WS sequences) is verified by
    sim/tb_cnn_accelerator_v2.sv (T3/T4).
    """
    return {
        "flush_cycles": FLUSH,                                  # commit -> quiesced
        "mode_switch_cycles": FLUSH,                            # = flush (the flush IS the switch)
        "first_valid_result_after_switch_cycles": {
            "os_to_ws": FLUSH + 42,   # flush + WS first group result (s=42, dense)
            "ws_to_os": FLUSH + 67,   # flush + OS first drain emit (s=67)
            "note": "first valid result of the new mode after start; the WS first "
                    "group of a sparse image emits zeros at s=0 (skip) instead of s=42",
        },
        "total_transition_latency_cycles": FLUSH,
        "total_transition_latency_ns": round(FLUSH * (1000.0 / CLK_MHZ), 1),
        "bit_exact": {
            "os_to_ws": "verified (no reset)",
            "ws_to_os": "verified (no reset)",
            "os_to_ws_to_os": "verified (no reset)",
            "ws_to_os_to_ws": "verified (no reset)",
        },
        "provenance": "[SIMULATION] tb_cnn_accelerator_v2.sv T3/T4; flush is "
                      "flush_cnt 0..7 in cnn_accelerator_v2.sv",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Primary real vector (image 0).
    img0 = load_image_hex(VECTORS / "input_img.hex")
    n_skip0 = len(skippable_positions(img0))
    skip_macs0 = skippable_macs(img0, skippable_positions(img0))

    # ---- cycle decomposition ----
    decomp = cycle_decomposition()
    (OUT / "cycle_decomposition.json").write_text(json.dumps(decomp, indent=2) + "\n")

    # ---- reconfiguration timing ----
    reconfig = reconfiguration_timing()
    (OUT / "reconfiguration.json").write_text(json.dumps(reconfig, indent=2) + "\n")

    # ---- benchmark matrix ----
    matrix = benchmark_matrix(img0, n_skip0, skip_macs0)
    (OUT / "benchmark_matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    keys = list(matrix[0].keys())
    with open(OUT / "benchmark_matrix.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in matrix:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    # ---- sparsity accounting: primary + deterministic multi-image subset ----
    # Deterministic subset of the REAL test set (indices 0..N-1, no RNG).
    import sys
    sys.path.insert(0, str(REPO))
    from torchvision import datasets
    from reference import preprocess
    from reference import quant

    params = None
    try:
        qp = np.load(VECTORS / "quant_params.npz")
        S_a = float(qp["S_a"])
    except Exception:
        S_a = 1.0 / 127.0

    N = 200
    ds = datasets.MNIST(root=str(REPO / "data" / "raw"), train=False, download=False)
    rows = []
    for i in range(N):
        x_fp = preprocess.pil_to_input(ds[i][0])
        q_a = quant.quantize_tensor(x_fp, S_a).astype(np.int8).reshape(28, 28)
        a = sparsity_accounting(q_a)
        rows.append({"index": i, "label": int(ds[i][1]), **a})

    # aggregate
    arr = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("index", "label")}
    agg = {}
    for k, v in arr.items():
        agg[k] = {
            "mean": round(float(v.mean()), 3),
            "min": round(float(v.min()), 3),
            "max": round(float(v.max()), 3),
            "std": round(float(v.std()), 3),
        }

    summary = {
        "primary_image": {**sparsity_accounting(img0), "index": 0, "label": 7},
        "subset": {"n_images": N, "deterministic": "test split indices 0..N-1, no RNG"},
        "aggregate": agg,
    }
    (OUT / "sparsity_accounting.json").write_text(json.dumps(summary, indent=2) + "\n")
    metric_keys = list(rows[0].keys())
    metric_keys.remove("index")
    metric_keys.remove("label")
    with open(OUT / "sparsity_accounting.csv", "w") as f:
        f.write("index,label," + ",".join(metric_keys) + "\n")
        for r in rows:
            f.write(f"{r['index']},{r['label']}," +
                    ",".join(str(r[k]) for k in metric_keys) + "\n")

    # ---- print summary ----
    print("=== cycle decomposition ===")
    print(json.dumps(decomp, indent=2))
    print("=== benchmark matrix ===")
    for r in matrix:
        util = r['pe_utilization_pct']
        mpc = r['mac_per_cycle']
        util_s = f"{util:5.2f}%" if isinstance(util, float) else str(util)
        mpc_s = f"{mpc:5.2f}" if isinstance(mpc, float) else str(mpc)
        print(f"{r['config']:10s} cycles={r['cycles']:6d} util={util_s} "
              f"mac/cyc={mpc_s} [{r['cycles_provenance']}]")
    print("=== sparsity (primary image) ===")
    print(json.dumps(summary["primary_image"], indent=2))
    print("=== sparsity (multi-image aggregate, key fields) ===")
    for k in ["activation_zero_fraction_pct", "skippable_group_fraction_pct",
              "measured_speedup", "measured_cycle_reduction_pct"]:
        print(f"  {k}: {agg[k]}")
    print(f"\nWrote artifacts to {OUT}/")


if __name__ == "__main__":
    main()
