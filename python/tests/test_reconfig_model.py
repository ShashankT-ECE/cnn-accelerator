#!/usr/bin/env python3
"""Self-checks for the Step 3.3 reconfigurable OS/WS model.

Verifies mode selection, switch counting, overhead accounting, and the honest
finding that reconfiguration is (marginally) beneficial vs fixed OS and strongly
beneficial vs fixed WS.

Plain asserts; no pytest dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from reconfig_model import (compute, num_switches, per_layer_selection,  # noqa: E402
                            SWITCH_OVERHEAD)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def main() -> int:
    print("Step 3.3 reconfigurable OS/WS model self-checks")

    rows = {r["name"]: r for r in per_layer_selection()}

    # Mode selection matches the measured crossover (Step 3.2).
    check("conv1 -> OS", rows["conv1"]["best"] == "OS")
    check("conv3 -> OS", rows["conv3"]["best"] == "OS")
    check("conv5 -> OS", rows["conv5"]["best"] == "OS")
    check("fc1 -> OS", rows["fc1"]["best"] == "OS")
    check("fc2 -> OS", rows["fc2"]["best"] == "OS")

    # Switch counting.
    check("num_switches == 0 (all-OS)", num_switches(["OS"] * 5) == 0)
    check("num_switches == 1 (one OS->WS transition)",
          num_switches(["OS", "OS", "OS", "WS", "WS"]) == 1)

    r = compute()

    # Overhead accounting: reconfigurable == oracle + switches * overhead.
    check("overhead == switches * SWITCH_OVERHEAD",
          r["total_switch_overhead"] == r["num_switches"] * SWITCH_OVERHEAD)
    check("reconfigurable == oracle + overhead",
          r["reconfigurable_cycles"] == r["oracle_cycles"] + r["total_switch_overhead"],
          f"{r['reconfigurable_cycles']} vs {r['oracle_cycles']} + {r['total_switch_overhead']}")

    # Honest: OS is optimal everywhere, so reconfiguration == fixed OS (no benefit).
    check("reconfigurable == fixed OS (no reconfiguration benefit)",
          r["reconfigurable_cycles"] == r["fixed_os_cycles"],
          f"{r['reconfigurable_cycles']} == {r['fixed_os_cycles']}")
    check("fixed WS is worst", r["fixed_ws_cycles"] > r["fixed_os_cycles"])
    check("zero switches, zero overhead", r["num_switches"] == 0 and r["total_switch_overhead"] == 0)
    check("speedup vs fixed OS == 1.0 (no benefit)",
          abs(r["speedup_vs_fixed_os"] - 1.0) < 1e-9, f"{r['speedup_vs_fixed_os']:.4f}x")
    check("speedup vs fixed WS == 3.264x",
          abs(r["speedup_vs_fixed_ws"] - 140412 / 43021) < 1e-9, f"{r['speedup_vs_fixed_ws']:.4f}x")

    print("  ALL STEP 3.3 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
