#!/usr/bin/env python3
"""Collect the feasibility xsim TB results (v3/build/feas_sim/*/xsim.txt) into v3/results/feas_tb.csv
(source = 'RTL sim'). One row per TB run: tb, the key=value fields of its FEAS_TB line, verdict."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from common import REPO_ROOT, RESULTS_DIR, base_meta, write_results_csv  # noqa: E402

SIM = REPO_ROOT / "v3" / "build" / "feas_sim"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_DIR / "feas_tb.csv"))
    a = ap.parse_args()
    rows, fields = [], ["run", "tb", "verdict"]
    for x in sorted(SIM.glob("*/xsim.txt")):
        line = next((l for l in x.read_text().splitlines() if l.startswith("FEAS_TB")), None)
        r = base_meta(source="RTL sim")
        r.update(run=x.parent.name, tb="", verdict="FAIL")
        if line:
            parts = line.split()
            r.update(tb=parts[1], verdict=parts[-1])
            for k, v in re.findall(r"(\w+)=(\S+)", line):
                r[k] = v
                if k not in fields:
                    fields.append(k)
        rows.append(r)
    write_results_csv(a.out, rows, fields)
    for r in rows:
        print(r["run"], r["verdict"], {k: r.get(k) for k in fields[3:] if r.get(k)})
    sys.exit(0 if rows and all(r["verdict"] == "PASS" for r in rows) else 1)


if __name__ == "__main__":
    main()
