#!/usr/bin/env python3
"""Self-checks for the CIFAR-10 OS/WS mapping model (Step 6.4).

Verifies the frozen CIFAR tensor dims/MACs, the exact OS/WS cycle counts (same
formula family as the frozen LeNet baseline + corrected FC schedule), that OS
wins every layer, and that no OS/WS crossover exists.

Plain asserts; no pytest dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from cifar10_mapping_model import (  # noqa: E402
    LAYERS, TOTAL_MACS, compute, os_map, ws_map,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def main() -> int:
    print("Step 6.4 CIFAR-10 mapping model self-checks")

    # ---- Frozen tensor dims / MACs (docs/CIFAR10_WORKLOAD_SPEC.md §2-§3) ----
    dims = {"conv1": (3, 32, 5, 28, 28), "conv2": (32, 32, 5, 10, 10),
            "conv3": (32, 64, 5, 1, 1), "fc": (64, 10, 1, 1, 1)}
    for L in LAYERS:
        IC, OC, K, H, W = dims[L["name"]]
        check(f"{L['name']} dims (IC={IC} OC={OC} K={K} H'={H} W'={W})",
              (L["IC"], L["OC"], L["K"], L["H"], L["W"]) == (IC, OC, K, H, W),
              str((L["IC"], L["OC"], L["K"], L["H"], L["W"])))
    check("total MACs == 4493440", TOTAL_MACS == 4493440, str(TOTAL_MACS))
    check("sum(layer MACs) == 4493440",
          sum(L["MACs"] for L in LAYERS) == 4493440,
          str(sum(L["MACs"] for L in LAYERS)))

    # ---- Exact OS/WS cycles (same formula family as frozen LeNet baseline) ----
    expected_os = {"conv1": 94976, "conv2": 167760, "conv3": 16776, "fc": 152}
    expected_ws = {"conv1": 322560, "conv2": 518400, "conv3": 51840, "fc": 660}
    for L in LAYERS:
        n = L["name"]
        o, w = os_map(L)["cycles"], ws_map(L)["cycles"]
        check(f"{n} OS == {expected_os[n]}", o == expected_os[n], str(o))
        check(f"{n} WS == {expected_ws[n]}", w == expected_ws[n], str(w))

    # ---- OS wins every layer; no crossover ----
    for L in LAYERS:
        n = L["name"]
        o, w = os_map(L)["cycles"], ws_map(L)["cycles"]
        check(f"{n}: OS clearly preferred (OS < WS)", o < w, f"OS={o} WS={w}")

    # ---- Corrected FC schedule applied (Step 4.1, not the old overcount) ----
    fc = next(L for L in LAYERS if L["name"] == "fc")
    check("FC OS uses corrected schedule (152, not ~8x overcount)",
          os_map(fc)["cycles"] == 152, str(os_map(fc)["cycles"]))

    # ---- Totals ----
    r = compute()
    check("Total OS == 279664", r["totals"]["OS_cycles"] == 279664,
          str(r["totals"]["OS_cycles"]))
    check("Total WS == 893460", r["totals"]["WS_cycles"] == 893460,
          str(r["totals"]["WS_cycles"]))
    check("OS_over_WS == 279664/893460", abs(r["totals"]["OS_over_WS"] - 279664/893460) < 1e-12,
          f"{r['totals']['OS_over_WS']:.6f}")

    # ---- Reconfiguration: OS everywhere -> 0 switches, no benefit vs fixed OS ----
    rc = r["reconfiguration"]
    check("no crossover (all modes OS)", rc["no_crossover"] is True)
    check("selected modes == all OS", list(rc["selected_modes"].values()) == ["OS"] * 4)
    check("num_switches == 0", rc["num_switches"] == 0)
    check("reconfigurable == fixed OS", rc["reconfigurable_cycles"] == rc["fixed_OS_cycles"])
    check("speedup_vs_fixed_OS == 1.0", abs(rc["speedup_vs_fixed_OS"] - 1.0) < 1e-12)

    # ---- Honest timing-status labels (never upgraded to 'verified') ----
    for L in LAYERS:
        n = L["name"]
        lr = r["layers"][n]
        if L["K"] > 1:
            check(f"{n} conv OS labeled 'unverified generalization'",
                  "unverified" in lr["OS"]["timing_status"],
                  lr["OS"]["timing_status"])
            check(f"{n} conv WS labeled 'unverified generalization'",
                  "unverified" in lr["WS"]["timing_status"],
                  lr["WS"]["timing_status"])
        else:
            check(f"{n} FC OS labeled 'RTL-verified'",
                  "RTL-verified" in lr["OS"]["timing_status"],
                  lr["OS"]["timing_status"])

    # ---- Structural ratio sanity ----
    # Asymptotically (large IC, T a multiple of 8) OS/WS -> (K+8)/(8K) = 13/40.
    # This holds for conv2/conv3 (IC=32, T=800 = 8*100 exactly); conv1 (IC=3)
    # carries non-negligible fixed overhead (17/+10) and a non-exact tile (75 = 9*8+3),
    # so its ratio (0.2944) is checked via the exact-cycle assertions above instead.
    for n in ("conv2", "conv3"):
        L = next(x for x in LAYERS if x["name"] == n)
        ratio = os_map(L)["cycles"] / ws_map(L)["cycles"]
        check(f"{n} conv OS/WS within 1% of 13/40",
              abs(ratio - 13 / 40) / (13 / 40) < 0.01, f"{ratio:.4f}")

    print("  ALL STEP 6.4 CIFAR-10 MAPPING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
