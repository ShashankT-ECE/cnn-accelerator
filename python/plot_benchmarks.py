"""Paper-ready figures from the benchmark CSV/JSON (source of truth = CSV).

Reads the artifacts produced by ``python/benchmark.py`` under ``data/benchmark/``
and writes PNG figures to ``data/benchmark/figures/``:

  * fig1_os_vs_ws_cycles.png     — OS vs WS total cycles (dense + sparse).
  * fig2_dense_vs_sparse.png     — WS dense (analytical) vs sparse (measured).
  * fig3_sparsity_accounting.png — activation-zero vs skippable-group vs cycle reduction.
  * fig4_resource_utilization.png— LUT/FF/DSP/CARRY8/BRAM (from synthesis JSON).
  * fig5_reconfig_overhead.png   — reconfiguration flush latency.
  * fig6_multi_image_sparsity.png— per-image speedup distribution (200 real images).

Figures are deliberately plain and honest: integer axes where the quantity is a
count, labelled provenance, no misleading truncation. The CSV is the source of
truth; this script only renders it.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
B = REPO / "data" / "benchmark"
FIG = B / "figures"


def load(name: str):
    return json.loads((B / name).read_text())


def style_ax(ax):
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    matrix = load("benchmark_matrix.json")
    decomp = load("cycle_decomposition.json")
    spars = load("sparsity_accounting.json")
    reconfig = load("reconfiguration.json")

    by_cfg = {r["config"]: r for r in matrix}

    # ---- fig 1: OS vs WS cycles ----
    fig, ax = plt.subplots(figsize=(6, 4))
    cfgs = ["os_dense", "ws_dense", "ws_sparse"]
    labels = ["OS dense\n(measured)", "WS dense\n(no-skip bound)", "WS sparse\n(measured)"]
    cycles = [by_cfg[c]["cycles"] for c in cfgs]
    colors = ["#3b6ea5", "#c94f4f", "#3b8a63"]
    bars = ax.bar(labels, cycles, color=colors)
    for b, v in zip(bars, cycles):
        ax.text(b.get_x() + b.get_width() / 2, v + 800, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Cycles (lower is better)")
    ax.set_title("OS vs WS total cycles (Conv1, 156,800 MACs)")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_os_vs_ws_cycles.png", dpi=150)
    plt.close(fig)

    # ---- fig 2: dense vs sparse (WS) ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ws_d = by_cfg["ws_dense"]["cycles"]
    ws_s = by_cfg["ws_sparse"]["cycles"]
    ax.bar(["WS dense\n(no-skip bound)", "WS sparse\n(coarse zero-skip)"], [ws_d, ws_s],
           color=["#c94f4f", "#3b8a63"])
    for i, v in enumerate([ws_d, ws_s]):
        ax.text(i, v + 800, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Cycles")
    ax.set_title(f"WS dense vs sparse (speedup {ws_d / ws_s:.2f}x)")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_dense_vs_sparse.png", dpi=150)
    plt.close(fig)

    # ---- fig 3: sparsity accounting (three distinct quantities) ----
    p = spars["primary_image"]
    fig, ax = plt.subplots(figsize=(6, 4))
    cats = [
        ("activation\nzero fraction", p["activation_zero_fraction_pct"]),
        ("skippable\nMAC fraction", p["skippable_mac_fraction_pct"]),
        ("measured\ncycle reduction", p["measured_cycle_reduction_pct"]),
    ]
    ax.bar([c[0] for c in cats], [c[1] for c in cats], color=["#8a8a8a", "#3b6ea5", "#3b8a63"])
    for i, (_, v) in enumerate(cats):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 100)
    ax.set_title("Sparsity: three distinct quantities (not interchangeable)")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_sparsity_accounting.png", dpi=150)
    plt.close(fig)

    # ---- fig 4: resource utilization (if synthesis data present) ----
    synth_path = B / "synthesis.json"
    if synth_path.exists():
        s = load("synthesis.json")
        fig, ax = plt.subplots(figsize=(6, 4))
        res = ["LUT", "FF", "DSP48E2", "CARRY8", "BRAM", "URAM"]
        vals = [s.get(k.lower(), 0) for k in res]
        ax.bar(res, vals, color="#3b6ea5")
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title("Post-route resource utilization (hardened)")
        style_ax(ax)
        fig.tight_layout()
        fig.savefig(FIG / "fig4_resource_utilization.png", dpi=150)
        plt.close(fig)

    # ---- fig 5: reconfiguration overhead ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["flush\n(mode switch)"], [reconfig["flush_cycles"]], color="#3b6ea5")
    ax.text(0, reconfig["flush_cycles"] + 0.2, f"{reconfig['flush_cycles']} cycles\n"
             f"({reconfig['total_transition_latency_ns']} ns @ 200 MHz)",
            ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Cycles")
    ax.set_title("Runtime OS<->WS reconfiguration overhead")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_reconfig_overhead.png", dpi=150)
    plt.close(fig)

    # ---- fig 6: multi-image speedup distribution ----
    import csv
    rows = []
    with open(B / "sparsity_accounting.csv") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    speedups = [float(r["measured_speedup"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(speedups, bins=20, color="#3b8a63", alpha=0.8)
    ax.axvline(sum(speedups) / len(speedups), color="#222", linestyle="--",
               label=f"mean {sum(speedups)/len(speedups):.2f}x")
    ax.set_xlabel("WS speedup (dense / sparse)")
    ax.set_ylabel("Images")
    ax.set_title(f"Per-image coarse-skip speedup over {len(speedups)} real MNIST images")
    ax.legend()
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_multi_image_sparsity.png", dpi=150)
    plt.close(fig)

    print(f"Wrote figures to {FIG}/")


if __name__ == "__main__":
    main()
