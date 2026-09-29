#!/usr/bin/env bash
# run_shard.sh <sim> <build_dir> <hex_dir> <sel> <shard_dir>
#   sel = "range:<start>:<end>" (jobs [start, end)) or "list:<file.hex>" (count, then job ids)
# Runs tb_shapes with one simulator process (sim = xsim | verilator) in <shard_dir>/. Writes
# <shard_dir>/run.log (first line "SIMULATOR name=.. version=..") and <shard_dir>/status
# ("rc t_start t_end", epoch seconds). Called by v2/scripts/run_shapes.sh (never edits RTL,
# never runs Vivado).
set -uo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 sim build_dir hex_dir sel shard_dir" >&2; exit 2; }
SIM="$1"; BUILD="$2"; HEX="$3"; SEL="$4"; DIR="$5"
ulimit -c 0                                    # Verilator $fatal aborts (SIGABRT): no core files
case "$SEL" in
    range:*) IFS=: read -r _ S E <<< "$SEL"; ARGS=("DATA=$HEX" "JOB_START=$S" "JOB_END=$E") ;;
    list:*)  ARGS=("DATA=$HEX" "JOBLIST=${SEL#list:}") ;;
    *) echo "run_shard.sh: bad sel $SEL" >&2; exit 2 ;;
esac
rm -rf "$DIR"; mkdir -p "$DIR"; cd "$DIR" || exit 2
VER="$(cat "$BUILD/sim_version.txt" 2>/dev/null || echo unknown)"
t0=$(date +%s.%N)
{
    echo "SIMULATOR name=$SIM version=$VER"
    case "$SIM" in
        xsim)
            # shellcheck disable=SC1090
            source "$HOME/Xilinx/Vivado/2023.1/settings64.sh" >/dev/null 2>&1
            XA=(); for a in "${ARGS[@]}"; do XA+=(-testplusarg "$a"); done
            cp -r "$BUILD/xsim.dir" . && xsim snap -R "${XA[@]}"
            ;;
        verilator)
            VA=(); for a in "${ARGS[@]}"; do VA+=("+$a"); done
            "$BUILD/obj/Vtb_shapes" "${VA[@]}"
            ;;
        *) echo "run_shard.sh: unknown simulator $SIM"; false ;;
    esac
} > run.log 2>&1
rc=$?
t1=$(date +%s.%N)
rm -rf xsim.dir                                # the per-shard snapshot copy is not needed afterwards
echo "$rc $t0 $t1" > status
if [[ $rc -eq 0 ]] && grep -q "^TEST PASSED" run.log && grep -q "^SHARD_DONE" run.log; then
    echo "shard $SEL $SIM PASS ($(echo "$t1 - $t0" | bc) s)"
else
    echo "shard $SEL $SIM FAIL rc=$rc (log: $DIR/run.log)" >&2
fi
exit 0                                         # failures are judged by the collector, not xargs
