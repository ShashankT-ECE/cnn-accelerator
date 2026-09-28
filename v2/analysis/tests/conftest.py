import csv
import sys
from pathlib import Path

import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))
import _setup  # noqa: E402,F401  (puts v2/model and legacy python/ on sys.path, read-only)


@pytest.fixture(scope="session")
def cycle_model_csv():
    """Committed frozen cycle model results: {(net, layer): row}."""
    path = _setup.RESULTS_DIR / "cycle_model.csv"
    return {(r["net"], r["layer"]): r for r in csv.DictReader(path.open())}
