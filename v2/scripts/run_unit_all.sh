#!/usr/bin/env bash
# Run every unit TB (v2/tb/tb_*.sv with a "// GOS_UNIT_TB:" header) after checking
# the vectors against MANIFEST.json; write v2/results/unit_tb.csv (source=rtl_sim).
# Exit nonzero if any TB fails. For paper-grade rows run from a clean, committed tree.
set -uo pipefail
V2="$(cd "$(dirname "$0")/.." && pwd)"
PY="$V2/../.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1

"$PY" "$V2/scripts/check_vectors.py" || exit 1

RESULTS=()
fail=0
for tbf in "$V2"/tb/tb_*.sv; do
    head -1 "$tbf" | grep -q "GOS_UNIT_TB:" || continue
    tb="$(basename "$tbf" .sv)"
    t0=$(date +%s.%N)
    "$V2/scripts/run_xsim.sh" "$tb"; rc=$?
    t1=$(date +%s.%N)
    RESULTS+=("$tb:$rc:$(printf '%.1f' "$(echo "$t1 - $t0" | bc)")")
    [[ $rc -eq 0 ]] || fail=1
done
[[ ${#RESULTS[@]} -gt 0 ]] || { echo "run_unit_all.sh: no unit TBs found" >&2; exit 1; }
"$PY" "$V2/scripts/unit_collect.py" "${RESULTS[@]}" || fail=1
[[ $fail -eq 0 ]] && echo "run_unit_all.sh: ALL PASS (${#RESULTS[@]} TBs)" || echo "run_unit_all.sh: FAILURES" >&2
exit $fail
