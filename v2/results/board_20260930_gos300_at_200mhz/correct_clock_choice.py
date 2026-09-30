#!/usr/bin/env python3
"""Correct hw_clock_choice.json of the 2026-09-30 Session 1 (DECISIONS D19).

The session recorded "highest candidate 299.997 MHz passed the smoke test", but pl_clk0 was never
set to the closed clock: the gos_300 bitstream ran at the read-back clock stored in
session_state.json (provenance.clock_readback_mhz). Every number written here is read from
hw_clock_choice.original.json / session_state.json in this directory; nothing is typed.

    python3 correct_clock_choice.py        # writes hw_clock_choice.json
"""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
orig = json.loads((here / "hw_clock_choice.original.json").read_text())
prov = json.loads((here / "session_state.json").read_text())["provenance"]
rb, closed = float(prov["clock_readback_mhz"]), float(prov["closed_clock_mhz"])
at_closed = abs(rb - closed) <= 0.1
note = (f"gos_300 bitstream at {rb:g} MHz: pl_clk0 read back {rb:.6f} MHz, closed clock "
        f"{closed:.6f} MHz — the smoke test passed at {rb:g} MHz; {closed:g} MHz was NOT exercised")
out = dict(orig)
out.update(ok=at_closed, valid_as_clock_choice=at_closed, clock_readback_mhz=rb,
           ran_at_closed_clock=at_closed, reason=note,
           attempts=[dict(a, clock_readback_mhz=rb, ran_at_closed_clock=at_closed,
                          reason=f"smoke PASS at {rb:g} MHz read-back (not at the closed clock)")
                     for a in orig["attempts"]],
           correction={"corrected_utc_date": "2026-09-30", "decision": "DECISIONS D19",
                       "readback_source": "session_state.json provenance.clock_readback_mhz",
                       "original_file": "hw_clock_choice.original.json",
                       "original_reason": orig["reason"], "original_ok": orig["ok"]})
(here / "hw_clock_choice.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
