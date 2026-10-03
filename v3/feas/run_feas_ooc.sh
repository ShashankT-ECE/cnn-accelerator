#!/usr/bin/env bash
# run_feas_ooc.sh — V3 Phase 0 WS5 feasibility OOC synthesis + device query (ONE Vivado job at a time;
# vivado_guard.sh before each job). Outputs under v3/build/feas_ooc/<run>/ ; then
# v3/feas/ooc_collect.py writes v3/results/feas_ooc.csv and device_resources.csv.
#   v3/feas/run_feas_ooc.sh [run-name-filter]
set -euo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO"
source ~/Xilinx/Vivado/2023.1/settings64.sh > /dev/null
OUT="$REPO/v3/build/feas_ooc"
mkdir -p "$OUT"
FILTER="${1:-}"
RTL=v3/feas/rtl

vivado_job() {   # <name> <args...>
    local name="$1"; shift
    [[ -n "$FILTER" && "$name" != *$FILTER* ]] && return 0
    until v3/scripts/vivado_guard.sh; do echo "guard refused; retry in 60 s"; sleep 60; done
    mkdir -p "$OUT/$name"
    ( cd "$OUT/$name" && vivado -mode batch -nojournal -log vivado.log "$@" > stdout.txt 2>&1 ) \
        || { echo "FAILED: $name (see $OUT/$name/vivado.log)"; return 1; }
    grep -h "OOC_SUMMARY" "$OUT/$name/stdout.txt" || true
}

vivado_job device_query -source "$REPO/v3/feas/device_query.tcl" -tclargs "$OUT/device_query"
for P in 4.000 3.000; do
    for W in 16 32; do
        for PK in 0 1; do
            vivado_job "feas_row_w${W}_p${PK}_${P}" -source "$REPO/v3/feas/ooc_synth.tcl" -tclargs \
                feas_row "$OUT/feas_row_w${W}_p${PK}_${P}" "$P" "$REPO/$RTL/feas_row.sv" -- "W=$W" "PACK=$PK"
        done
        vivado_job "feas_mem_path_w${W}_${P}" -source "$REPO/v3/feas/ooc_synth.tcl" -tclargs \
            feas_mem_path "$OUT/feas_mem_path_w${W}_${P}" "$P" "$REPO/$RTL/feas_mem_path.sv" -- "W=$W"
    done
done
echo "run_feas_ooc: done"
