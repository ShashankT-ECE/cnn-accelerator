"""Put v2/model on sys.path (read-only import of the frozen model code).

v2/analysis never writes under v2/model (DECISIONS D16: any change there makes
every results row stale). Bytecode writing is disabled so importing the model
leaves no files behind.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ANALYSIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = ANALYSIS_DIR.parent / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import common  # noqa: E402  (also puts legacy python/ on sys.path)

RESULTS_DIR = common.RESULTS_DIR


def out_dir_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR,
                    help="directory for the CSV (default v2/results/; paper CSVs only "
                         "from a clean committed tree via v2/scripts/regen_results.sh)")
    return ap


def fmt(x, nd: int = 6) -> str:
    return f"{float(x):.{nd}f}"
