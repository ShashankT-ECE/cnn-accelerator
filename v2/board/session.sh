#!/usr/bin/env bash
# V2 board sessions — one command per session (thin wrapper around run_sessions.py).
#
#   sudo -E ./session.sh 1|2|3|all [--budget-min N] [--resume|--fresh] [--quick] [--plan]
#   ./session.sh all --backend model --allow-dirty       # laptop dry run -> v2/results/dryrun/
#
# `./session.sh --help` lists every option. PY overrides the interpreter (board: python3 = the
# PYNQ venv after `source /etc/profile.d/pynq_venv.sh`; dry run: the repo .venv if present).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${PY:-}" ]; then
  PY=python3
  for a in "$@"; do
    if [ "$a" = model ] && [ -x "$HERE/../../.venv/bin/python" ]; then PY="$HERE/../../.venv/bin/python"; fi
  done
fi
exec "$PY" "$HERE/run_sessions.py" "$@"
