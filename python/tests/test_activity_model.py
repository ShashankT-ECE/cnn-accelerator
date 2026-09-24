#!/usr/bin/env python3
"""Self-checks for the Step 3.4 hardware-activity model.

Verifies the activity accounting is self-consistent and that the Step 3.2/3.3
cycle results are unchanged (the activity model only *reads* them).

Plain asserts; no pytest dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from mapping_model import LAYERS, PE_COUNT, os_map, ws_map  # noqa: E402
from reconfig_model import compute as reconfig_compute  # noqa: E402
from activity_model import activity  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def main() -> int:
    print("Step 3.4 hardware-activity model self-checks")

    # Corrected (Step 4.2) cycle totals.
    os_tot = sum(os_map(L)["cycles"] for L in LAYERS)
    ws_tot = sum(ws_map(L)["cycles"] for L in LAYERS)
    check("OS total == 43021 (corrected)", os_tot == 43021, str(os_tot))
    check("WS total == 140412 (corrected)", ws_tot == 140412, str(ws_tot))

    # Corrected reconfiguration (no benefit).
    rc = reconfig_compute()
    check("reconfigurable == 43021 (corrected)", rc["reconfigurable_cycles"] == 43021,
          str(rc["reconfigurable_cycles"]))
    check("overhead == 0 (corrected)", rc["total_switch_overhead"] == 0)

    # Activity accounting per layer.
    for L in LAYERS:
        n = L["name"]
        for mode, m in (("OS", os_map(L)), ("WS", ws_map(L))):
            a = activity(L, m, mode)
            check(f"{n} {mode} active_PE_ops == MACs", a["active_PE_operations"] == L["MACs"])
            check(f"{n} {mode} PE_util == MACs/(64*cycles)",
                  abs(a["PE_utilization"] - L["MACs"] / (PE_COUNT * m["cycles"])) < 1e-12)
            check(f"{n} {mode} idle == 64*cycles - MACs",
                  a["idle_PE_cycles"] == PE_COUNT * m["cycles"] - L["MACs"])
            # psum: OS none, WS == MACs
            expect_psum = 0 if mode == "OS" else L["MACs"]
            check(f"{n} {mode} psum_ops", a["psum_operations"] == expect_psum)

    # weight_reads are mode-independent (OS == WS per layer).
    for L in LAYERS:
        n = L["name"]
        os_a = activity(L, os_map(L), "OS")
        ws_a = activity(L, ws_map(L), "WS")
        check(f"{n} weight_reads mode-independent", os_a["weight_reads"] == ws_a["weight_reads"],
              f"{os_a['weight_reads']} vs {ws_a['weight_reads']}")

    # activation_reads is explicitly marked unavailable.
    check("activation_reads marked unavailable",
          activity(LAYERS[0], os_map(LAYERS[0]), "OS")["activation_reads"] is None)

    print("  ALL STEP 3.4 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
