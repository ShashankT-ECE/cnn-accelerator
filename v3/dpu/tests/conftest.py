import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                     # v3/dpu
sys.path.insert(0, str(HERE.parents[1] / "board"))       # v3/board
sys.path.insert(0, str(HERE.parents[1] / "model"))       # v3/model
