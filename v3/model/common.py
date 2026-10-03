"""Shared helpers for v3: repo paths and results metadata (copied from v2/model/common.py @ 28dd2ad).

Results CSVs use ``write_results_csv`` so every row carries the metadata
columns (V3 keeps the V2 CSV rule). V3 does not import v2/ or python/.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = REPO_ROOT / "v3"
MODEL_DIR = V3_ROOT / "model"
FROZEN_DIR = MODEL_DIR / "frozen"
RESULTS_DIR = V3_ROOT / "results"
DRYRUN_DIR = RESULTS_DIR / "dryrun"
CIFAR_RAW = REPO_ROOT / "data" / "raw"   # torchvision CIFAR-10 root (cifar-10-batches-py)

# Metadata columns required on every results row (V2 EXPERIMENTS.md "CSV rule", kept for V3).
META_COLUMNS = (
    "timestamp", "git_commit", "git_dirty", "vivado_version", "bitstream_sha256",
    "board_id", "net", "layer", "clock_mhz", "source", "duration_s", "num_inferences",
)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def git_commit() -> str:
    return _git("rev-parse", "HEAD")


# Generated outputs are excluded from the dirty check so that a regeneration
# run (which rewrites them) keeps reporting the state of the code that produced
# them. Everything else - code, frozen params, docs - counts.
OUTPUT_PATHSPECS = (":!v3/results",)


def git_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--", ".", *OUTPUT_PATHSPECS))


def base_meta(net: str = "", layer: str = "", source: str = "model",
              duration_s: float | str = "", num_inferences: int | str = "") -> dict:
    """Metadata for one row. Hardware-only fields are left empty."""
    return {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "vivado_version": "",
        "bitstream_sha256": "",
        "board_id": "",
        "net": net,
        "layer": layer,
        "clock_mhz": "",
        "source": source,
        "duration_s": duration_s,
        "num_inferences": num_inferences,
    }


def write_results_csv(path, rows: list[dict], fields: list[str]) -> Path:
    """Write rows with META_COLUMNS first, then ``fields`` (measured columns)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(META_COLUMNS) + [f for f in fields if f not in META_COLUMNS]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    return path
