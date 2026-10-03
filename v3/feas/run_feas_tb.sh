#!/usr/bin/env bash
# run_feas_tb.sh — xsim self-checking TBs of the exploratory feasibility RTL (not Vivado jobs).
# Logs under v3/build/feas_sim/<tb>/ ; v3/feas/tb_collect.py writes v3/results/feas_tb.csv.
set -euo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source ~/Xilinx/Vivado/2023.1/settings64.sh > /dev/null
SIM="$REPO/v3/build/feas_sim"
RTL="$REPO/v3/feas/rtl"; TB="$REPO/v3/feas/tb"
run() {   # <name> <top> <rtl> <generics...>
    local name="$1" top="$2" rtl="$3"; shift 3
    local d="$SIM/$name"; rm -rf "$d"; mkdir -p "$d"; cd "$d"
    local g=(); for x in "$@"; do g+=(-generic_top "$x"); done
    xvlog -sv "$rtl" "$TB/$top.sv" > xvlog.txt 2>&1
    xelab -s snap "$top" "${g[@]}" > xelab.txt 2>&1
    xsim snap -R > xsim.txt 2>&1 || true
    grep -h "FEAS_TB" xsim.txt || echo "FEAS_TB $top $name NO-RESULT FAIL"
}
for W in 16 32; do
    for PK in 0 1; do run "row_w${W}_p${PK}" tb_feas_row "$RTL/feas_row.sv" "W=$W" "PACK=$PK" "NSEQ=400"; done
    run "mem_w${W}" tb_feas_mem_path "$RTL/feas_mem_path.sv" "W=$W" "NREQ=20000"
done
