#!/usr/bin/env bash
# copied from v2/board/session.sh @ 28dd2ad (V3: runs baseline_session.py; dry run = --backend model)
# V3 baseline board sessions -- one command per session (thin wrapper around baseline_session.py).
#
#   ./session.sh 1|2|3 [--plan] [--resume|--fresh] [...]     # on the KV260 (cd ~/gos3)
#   ./session.sh py <script.py> [args]                        # any board script with the same environment,
#                                                             #   e.g. ./session.sh py power_log.py --list-sensors
#   v3/board/session.sh 1 --backend model                     # laptop DRY RUN -> v3/results/dryrun/baseline/
#
# On the board the script sets up its own environment, so this ONE command works from a plain login shell:
#   1. not root -> re-executes itself with `sudo -E` (overlay loading / cpufreq need root);
#   2. sources /etc/profile.d/pynq_venv.sh AS ROOT (PYNQ venv python3 with pynq, pynq_dpu, onnxruntime,
#      XILINX_XRT; `sudo -E python3` alone loses the venv from PATH).
# PY overrides the interpreter and skips the board setup (dry run: the repo .venv if present).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYNQ_PROFILE="${GOS_PYNQ_PROFILE:-/etc/profile.d/pynq_venv.sh}"
MODEL=0
for a in "$@"; do [ "$a" = model ] && MODEL=1; done
if [ -z "${PY:-}" ]; then
  if [ "$MODEL" = 1 ]; then
    PY=python3
    [ -x "$HERE/../../.venv/bin/python" ] && PY="$HERE/../../.venv/bin/python"
  elif [ -f "$PYNQ_PROFILE" ]; then
    if [ "$(id -u)" != 0 ]; then
      exec sudo -E "$0" "$@"
    fi
    set +u
    # shellcheck disable=SC1090
    . "$PYNQ_PROFILE" || { echo "session.sh: could not source $PYNQ_PROFILE" >&2; exit 3; }
    set -u
    PY=python3
    "$PY" -c 'import pynq' 2>/dev/null || {
      echo "session.sh: $(command -v "$PY") cannot import pynq after sourcing $PYNQ_PROFILE" >&2; exit 3; }
  else
    PY=python3
  fi
fi
if [ "${1:-}" = py ]; then
  shift
  [ $# -ge 1 ] || { echo "usage: session.sh py <script.py> [args]" >&2; exit 2; }
  exec "$PY" "$@"
fi
exec "$PY" "$HERE/baseline_session.py" "$@"
