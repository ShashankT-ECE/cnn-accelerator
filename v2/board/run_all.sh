#!/usr/bin/env bash
# SUPERSEDED: thin wrapper that forwards to `./session.sh 2` (Session 2: A1 -> A2/A3 -> A4 -> B3 ->
# CPU baselines, with pre-flight, resume, time budget and per-step timeouts; see README.md).
#
#   sudo -E ./run_all.sh [session.sh options]        ==  sudo -E ./session.sh 2 [options]
#   ./run_all.sh --backend model --allow-dirty       # laptop dry run -> v2/results/dryrun/
#
# B1 / B2 power are NOT part of Session 2: they live in Session 3 (`./session.sh 3`), measured
# only with the on-board INA260 SOM-rail logger (power_log.py). The old run_all.sh options
# --limit / --skip-cpu no longer exist (use --quick, --steps, --budget-min; `./session.sh --help`).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "run_all.sh is superseded: forwarding to ./session.sh 2 $*"
echo "NOTE: B1/B2 power live in Session 3 (INA260 SOM-rail power, power_log.py): ./session.sh 3"
exec "$HERE/session.sh" 2 "$@"
