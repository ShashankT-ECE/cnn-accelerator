#!/usr/bin/env bash
# Analyses of the committed board results (run after a results commit, from a clean tree):
#   v2/analysis/dpu_overlay_params.py -> v2/results/dpu_overlay_params.csv  (DPU overlay .hwh parameters)
#   v2/analysis/board_efficiency.py   -> v2/results/board_efficiency.csv    (GOPS, images/s per MAC lane / DSP)
# Then commit the two CSVs separately and run v2/scripts/check_results.py.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- v2 ':!v2/results')" ] || { echo "REFUSED: dirty tree (v2/)"; exit 2; }
.venv/bin/python v2/analysis/dpu_overlay_params.py
.venv/bin/python v2/analysis/board_efficiency.py
.venv/bin/python v2/scripts/check_results.py | tail -1
