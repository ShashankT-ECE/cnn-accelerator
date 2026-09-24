#!/usr/bin/env python3
"""Self-checks for the corrected mapping model (Step 4.2).

Verifies the RTL-derived FC schedule (Step 4.1) is applied, the Conv1 anchor is
unchanged, and OS is optimal for every LeNet layer.

Plain asserts; no pytest dependency.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from mapping_model import LAYERS, os_map, ws_map  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(f"FAILED: {name}")


def main() -> int:
    print("Step 4.2 corrected mapping model self-checks")
    os = {L["name"]: os_map(L) for L in LAYERS}
    ws = {L["name"]: ws_map(L) for L in LAYERS}

    # Conv1 anchor unchanged (RTL-verified schedule).
    check("Conv1 OS == 9184 (anchor)", os["conv1"]["cycles"] == 9184, str(os["conv1"]["cycles"]))
    check("Conv1 WS == 28224 (anchor)", ws["conv1"]["cycles"] == 28224, str(ws["conv1"]["cycles"]))

    # Corrected FC cycle counts (RTL-derived, Step 4.1).
    check("FC1 OS == 1510", os["fc1"]["cycles"] == 1510, str(os["fc1"]["cycles"]))
    check("FC1 WS == 10248", ws["fc1"]["cycles"] == 10248, str(ws["fc1"]["cycles"]))
    check("FC2 OS == 192", os["fc2"]["cycles"] == 192, str(os["fc2"]["cycles"]))
    check("FC2 WS == 900", ws["fc2"]["cycles"] == 900, str(ws["fc2"]["cycles"]))

    # OS is optimal for every layer (no crossover).
    for L in LAYERS:
        n = L["name"]
        check(f"{n}: OS optimal (OS <= WS)", os[n]["cycles"] <= ws[n]["cycles"],
              f"OS={os[n]['cycles']} WS={ws[n]['cycles']}")

    # Corrected totals.
    check("Total OS == 43021", sum(os[n]["cycles"] for n in os) == 43021)
    check("Total WS == 140412", sum(ws[n]["cycles"] for n in ws) == 140412)

    print("  ALL STEP 4.2 MAPPING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
