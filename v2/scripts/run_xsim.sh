#!/usr/bin/env bash
# run_xsim.sh <tb_name> [rtl files...]
# Compile + elaborate + run one self-checking TB with Vivado 2023.1 xsim in
# v2/build/sim/<tb_name>/ (never the repo root). RTL files are relative to v2/rtl/
# (or absolute); the TB is v2/tb/<tb_name>.sv. If no RTL files are given, the
# "// GOS_UNIT_TB:" header line of the TB lists them.
# Exits 0 only if the log contains "TEST PASSED" and no ERROR/FATAL/"TEST FAILED".
set -uo pipefail

[[ $# -ge 1 ]] || { echo "usage: $0 <tb_name> [rtl files...]" >&2; exit 2; }
TB="$1"; shift
V2="$(cd "$(dirname "$0")/.." && pwd)"
TB_SRC="$V2/tb/$TB.sv"
[[ -f "$TB_SRC" ]] || { echo "run_xsim.sh: missing $TB_SRC" >&2; exit 2; }

if [[ $# -eq 0 ]]; then
    read -r -a FILES <<< "$(sed -n 's#^// GOS_UNIT_TB:##p' "$TB_SRC" | head -1)"
else
    FILES=("$@")
fi
SRCS=()
for f in "${FILES[@]}"; do
    [[ "$f" = /* ]] && SRCS+=("$f") || SRCS+=("$V2/rtl/$f")
done
SRCS+=("$TB_SRC")

WORK="$V2/build/sim/$TB"
rm -rf "$WORK"; mkdir -p "$WORK"; cd "$WORK"
# shellcheck disable=SC1090
source "$HOME/Xilinx/Vivado/2023.1/settings64.sh" >/dev/null 2>&1
VEC_DIR="${VEC_DIR:-$V2/vectors/generated}"

{
    xvlog -sv "${SRCS[@]}" &&
    xelab "$TB" -s "${TB}_snap" -timescale 1ns/1ps --debug off &&
    xsim "${TB}_snap" -R -testplusarg "VEC_DIR=$VEC_DIR"
} > run.log 2>&1
rc=$?

if [[ $rc -eq 0 ]] && grep -q "TEST PASSED" run.log \
   && ! grep -qE "^ERROR|ERROR:|FATAL|Fatal:|TEST FAILED" run.log; then
    grep "TEST PASSED" run.log | tail -1
    echo "run_xsim.sh: $TB PASS (log: $WORK/run.log)"
    exit 0
fi
tail -25 run.log >&2
echo "run_xsim.sh: $TB FAIL (rc=$rc, log: $WORK/run.log)" >&2
exit 1
