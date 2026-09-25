#!/usr/bin/env bash
# V2 Step 5 Session 2: A1 -> A2/A3 -> A4 -> B3 -> CPU baselines (A5).
#
#   sudo -E ./run_all.sh [--backend pynq] [--allow-dirty] [--limit N] [--quick] [--skip-cpu]
#   ./run_all.sh --backend model            # laptop dry run -> v2/results/dryrun/ (dryrun_model)
#
# Every step runs even if an earlier one fails; the exit code is nonzero if any failed.
# PY overrides the interpreter (board default python3; dry run default = repo .venv if present).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

BACKEND=pynq
PASS=()
QUICK=0
SKIP_CPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --limit) PASS+=(--limit "$2"); shift 2 ;;
    --allow-dirty) PASS+=(--allow-dirty); shift ;;
    --quick) QUICK=1; shift ;;
    --skip-cpu) SKIP_CPU=1; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
case "$BACKEND" in pynq|model) ;; *) echo "--backend must be pynq or model"; exit 2 ;; esac

if [ -z "${PY:-}" ]; then
  PY=python3
  if [ "$BACKEND" = model ] && [ -x "$HERE/../../.venv/bin/python" ]; then PY="$HERE/../../.venv/bin/python"; fi
fi
if [ -f "$HERE/../model/gos_pack.py" ]; then RES="$HERE/../results"; else RES="$HERE/results"; fi
if [ "$BACKEND" = model ]; then OUT="$RES/dryrun"; TAG=laptop; else OUT="$RES"; TAG=board; fi
mkdir -p "$OUT"
B3N=()
if [ "$QUICK" = 1 ]; then PASS+=(--limit 200); B3N=(--n 200); fi

echo "run_all: backend=$BACKEND python=$PY out=$OUT $(date -u +%FT%TZ)"
FAILED=()
step() {
  local name="$1"; shift
  echo; echo "########## $name: $* ##########"
  local t0=$SECONDS
  if "$@"; then echo "########## $name: OK ($((SECONDS - t0)) s)"
  else echo "########## $name: FAILED (exit $?, $((SECONDS - t0)) s)"; FAILED+=("$name"); fi
}

step A1    "$PY" exp_a1_accuracy.py  --backend "$BACKEND" "${PASS[@]}"
step A2A3  "$PY" exp_a2_a3_cycles.py --backend "$BACKEND" "${PASS[@]}"
step A4    "$PY" exp_a4_util.py      --backend "$BACKEND" "${PASS[@]}"
step B3    "$PY" exp_b3_breakdown.py --backend "$BACKEND" "${PASS[@]}" "${B3N[@]}"
if [ "$SKIP_CPU" = 1 ]; then
  echo "CPU baselines skipped (--skip-cpu)"
elif [ -f cpu/run_cpu_baselines.py ]; then
  CPU=(--data-dir "$HERE/data" --out-dir "$OUT" --tag "$TAG")
  [ "$QUICK" = 1 ] && CPU+=(--quick)
  for a in "${PASS[@]}"; do [ "$a" = --allow-dirty ] && CPU+=(--allow-dirty); done
  step CPU "$PY" cpu/run_cpu_baselines.py "${CPU[@]}"
else
  echo "WARNING: cpu/run_cpu_baselines.py not found; CPU baselines not run"; FAILED+=(CPU-missing)
fi

echo
if [ ${#FAILED[@]} -eq 0 ]; then echo "run_all: ALL STEPS OK -> $OUT"; exit 0; fi
echo "run_all: FAILED: ${FAILED[*]} -> $OUT"; exit 1
