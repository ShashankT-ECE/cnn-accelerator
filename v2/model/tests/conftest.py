import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
import common  # noqa: E402,F401  (puts legacy python/ on sys.path)
