#!/usr/bin/env bash
# V2 board sessions — one command per session (thin wrapper around run_sessions.py).
#
#   ./session.sh 1|2|3|all [--budget-min N] [--resume|--fresh] [--quick] [--plan]     # on the KV260
#   ./session.sh py <script.py> [args]      # any board script with the same environment, e.g.
#                                           #   ./session.sh py power_log.py --list-sensors
#   ./session.sh all --backend model --allow-dirty       # laptop dry run -> v2/results/dryrun/
#
# `./session.sh --help` lists every option.
# On the board the script sets up its own environment, so this ONE command works from a plain
# login shell (with or without `sudo`, with or without the PYNQ venv sourced):
#   1. not root -> re-executes itself with `sudo -E` (MMIO / overlay loading need root);
#   2. sources /etc/profile.d/pynq_venv.sh AS ROOT: the PYNQ venv python3, XILINX_XRT, BOARD and
#      the venv bin directory on PATH (pynq calls `xclbinutil` from there when it loads an overlay).
# `sudo -E python3 ...` alone does NOT work: sudo's secure_path drops the venv from PATH, so
# python3 is /usr/bin/python3 (no pynq) and xclbinutil is not found.
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
    command -v xclbinutil >/dev/null || {
      echo "session.sh: xclbinutil not on PATH after sourcing $PYNQ_PROFILE (pynq needs it to load an overlay)" >&2; exit 3; }
  else
    PY=python3
  fi
fi
if [ "${1:-}" = py ]; then
  shift
  [ $# -ge 1 ] || { echo "usage: session.sh py <script.py> [args]" >&2; exit 2; }
  exec "$PY" "$@"
fi
exec "$PY" "$HERE/run_sessions.py" "$@"
