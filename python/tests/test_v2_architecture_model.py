#!/usr/bin/env python3
"""Self-checks for the generalized OS-only V2 architecture model (Step: V2 profiling).

Verifies the corrected per-group schedule (1 + IC*K*(P+K) + 2*rows), that the
controller gaps (IC>1, OC>8, variable spatial, VALID) are resolved, that partial
groups are correctly shorter than full groups, and the exact refined totals for
LeNet-5 and CIFAR-10. Also checks the weight-BRAM latency accounting.

Plain asserts; no pytest dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from v2_architecture_model import (  # noqa: E402
    LENET, CIFAR, compute,
    os_conv_cycles, os_fc_cycles, os_group_breakdown, weight_bram_analysis,
)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def main() -> int:
    print("V2 generalized OS model self-checks")

    # ---- RTL-verified anchor: full group (IC=1, K=5, P=8, rows=8) = 82 ----
    b = os_group_breakdown(1, 5, 8, 8)
    check("anchor clear == 1", b["clear"] == 1, str(b["clear"]))
    check("anchor compute == 65 (5 passes x 13)", b["compute"] == 65, str(b["compute"]))
    check("anchor result_drain == 16 (2 x 8 rows)", b["result_drain"] == 16, str(b["result_drain"]))
    check("anchor total == 82", b["total"] == 82, str(b["total"]))

    # ---- Corrected schedule form: 1 + IC*K*(P+K) + 2*rows ----
    check("formula 1+IC*K*(P+K)+2*rows", b["total"] == 1 + 1 * 5 * (8 + 5) + 2 * 8, str(b["total"]))

    # ---- Partial groups are SHORTER than full groups (resolves R2 overcount) ----
    full = os_group_breakdown(1, 5, 8, 8)["total"]
    partial_px = os_group_breakdown(1, 5, 4, 8)["total"]      # P=4 pixel group
    partial_oc = os_group_breakdown(1, 5, 8, 6)["total"]      # rows=6 channel group
    check("partial pixel group (P=4) < full", partial_px < full, f"{partial_px} < {full}")
    check("partial channel group (rows=6) < full", partial_oc < full, f"{partial_oc} < {full}")
    check("partial pixel group == 62 (P=4, rows=8)", partial_px == 62, str(partial_px))
    check("partial channel group == 78 (P=8, rows=6)", partial_oc == 78, str(partial_oc))

    # ---- Corrected FC schedule (RTL-verified, not the old overcount) ----
    check("FC1 OS == 1510", os_fc_cycles(120, 84)["cycles"] == 1510)
    check("FC2 OS == 192", os_fc_cycles(84, 10)["cycles"] == 192)
    check("CIFAR FC OS == 152", os_fc_cycles(64, 10)["cycles"] == 152)

    # ---- Controller gaps resolved: IC>1, OC>8, variable spatial, VALID ----
    check("IC>1 (CIFAR conv1 IC=3) runs", os_conv_cycles(3, 32, 5, 28, 28)["cycles"] > 0)
    check("OC>8 (CIFAR conv3 OC=64) -> 8 channel groups",
          os_conv_cycles(32, 64, 5, 1, 1)["channel_groups"] == 8)
    check("variable spatial W'=1 (degenerate) -> P=1",
          os_conv_cycles(16, 120, 5, 1, 1)["partial_column_valid_pixels"] == 1)
    check("variable spatial W'=28 -> partial P=4",
          os_conv_cycles(1, 6, 5, 28, 28)["partial_column_valid_pixels"] == 4)

    # ---- Exact refined per-layer + totals ----
    def cyc(L):
        return os_conv_cycles(L["IC"], L["OC"], L["K"], L["H"], L["W"])["cycles"] if L["K"] > 1 \
            else os_fc_cycles(L["IC"], L["OC"])["cycles"]

    lenet_exp = {"conv1": 8176, "conv3": 12680, "conv5": 7455, "fc1": 1510, "fc2": 192}
    cifar_exp = {"conv1": 88256, "conv2": 129360, "conv3": 7816, "fc": 152}
    for L in LENET:
        n = L["name"]
        check(f"LeNet {n} OS == {lenet_exp[n]}", cyc(L) == lenet_exp[n], str(cyc(L)))
    for L in CIFAR:
        n = L["name"]
        check(f"CIFAR {n} OS == {cifar_exp[n]}", cyc(L) == cifar_exp[n], str(cyc(L)))

    out = compute()
    check("LeNet total OS == 30013", out["workloads"]["lenet5"]["total"]["OS_cycles"] == 30013)
    check("CIFAR total OS == 225584", out["workloads"]["cifar10"]["total"]["OS_cycles"] == 225584)

    # ---- Refined (corrected) never exceeds the frozen uniform baseline ----
    # Frozen uniform OS totals (docs/LENET5_MAPPING_FROZEN_MANIFEST / CIFAR mapping):
    uniform = {"lenet5": 43021, "cifar10": 279664}
    for wl in ("lenet5", "cifar10"):
        refined = out["workloads"][wl]["total"]["OS_cycles"]
        check(f"{wl} refined <= uniform baseline", refined <= uniform[wl],
              f"{refined} <= {uniform[wl]}")

    # ---- Weight-BRAM latency: 1-cycle read, hidden by look-ahead ----
    for L in CIFAR + LENET:
        wb = weight_bram_analysis(L)
        check(f"{L['name']} BRAM latency == 1, hidden", wb["bram_read_latency_cycles"] == 1
              and wb["latency_hidden_by_lookahead"] is True)
    check("CIFAR conv2 weight values == 25600",
          weight_bram_analysis([L for L in CIFAR if L["name"] == "conv2"][0])["weight_values"] == 25600)

    # ---- weight_bram_stall sensitivity: +1 cycle per group ----
    conv1 = os_conv_cycles(3, 32, 5, 28, 28)
    conv1_stall = os_conv_cycles(3, 32, 5, 28, 28, wbram_stall=1)
    check("wbram stall adds groups cycles",
          conv1_stall["cycles"] == conv1["cycles"] + conv1["groups"],
          f"{conv1_stall['cycles']} == {conv1['cycles']} + {conv1['groups']}")

    print("  ALL V2 GENERALIZED OS MODEL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
