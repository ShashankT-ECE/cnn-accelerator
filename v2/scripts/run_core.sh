#!/usr/bin/env bash
# Core RTL verification (V2 step 4): runs tb_gos_core with SUITE=layer|net|fuzz|checker
# (in parallel, one xsim run each, in v2/build/sim/tb_gos_core_<suite>/), then
# core_collect.py writes v2/results/rtl_cycles.csv, rtl_network.csv, rtl_checker.csv
# (source=rtl_sim). Exit nonzero on any failure. Paper-grade rows need a clean tree.
set -uo pipefail
V2="$(cd "$(dirname "$0")/.." && pwd)"
PY="$V2/../.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
"$PY" "$V2/scripts/check_vectors.py" || exit 1

FILES="$(sed -n 's#^// GOS_CORE_TB:##p' "$V2/tb/tb_gos_core.sv" | head -1)"
SUITES=(${CORE_SUITES:-layer net fuzz checker})
declare -A RC T0 T1
for s in "${SUITES[@]}"; do
    (
        t0=$(date +%s.%N)
        # shellcheck disable=SC2086
        SIM_TAG="$s" XSIM_PLUSARGS="SUITE=$s" "$V2/scripts/run_xsim.sh" tb_gos_core $FILES
        rc=$?
        t1=$(date +%s.%N)
        echo "$rc $(echo "$t1 - $t0" | bc)" > "$V2/build/sim/tb_gos_core_$s.status"
    ) &
done
wait
ARGS=()
fail=0
for s in "${SUITES[@]}"; do
    read -r rc secs < "$V2/build/sim/tb_gos_core_$s.status"
    ARGS+=("$s:$rc:$secs")
    [[ $rc -eq 0 ]] || fail=1
done
"$PY" "$V2/scripts/core_collect.py" "${ARGS[@]}" || fail=1
[[ $fail -eq 0 ]] && echo "run_core.sh: ALL PASS (${SUITES[*]})" || echo "run_core.sh: FAILURES" >&2
exit $fail
