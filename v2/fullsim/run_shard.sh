#!/usr/bin/env bash
# run_shard.sh <sim> <build_dir> <data_net_dir> <net> <start> <end> <shard_dir>
# Runs tb_full10k on images [start, end) of <net> with one simulator process
# (sim = xsim | verilator), in <shard_dir>/. Writes <shard_dir>/run.log (first line
# "SIMULATOR name=.. version=..") and <shard_dir>/status ("rc t_start t_end" epoch seconds).
# Called by v2/scripts/run_full10k.sh (never edits RTL; never runs Vivado).
set -uo pipefail
[[ $# -eq 7 ]] || { echo "usage: $0 sim build_dir data_net_dir net start end shard_dir" >&2; exit 2; }
SIM="$1"; BUILD="$2"; DATA="$3"; NET="$4"; S="$5"; E="$6"; DIR="$7"
ulimit -c 0                                    # Verilator $fatal aborts (SIGABRT): no core files
rm -rf "$DIR"; mkdir -p "$DIR"; cd "$DIR" || exit 2
VER="$(cat "$BUILD/sim_version.txt" 2>/dev/null || echo unknown)"
t0=$(date +%s.%N)
{
    echo "SIMULATOR name=$SIM version=$VER"
    case "$SIM" in
        xsim)
            # shellcheck disable=SC1090
            source "$HOME/Xilinx/Vivado/2023.1/settings64.sh" >/dev/null 2>&1
            cp -r "$BUILD/xsim.dir" . &&
            xsim snap -R -testplusarg "DATA=$DATA" -testplusarg "NET=$NET" \
                -testplusarg "IMG_START=$S" -testplusarg "IMG_END=$E"
            ;;
        verilator)
            "$BUILD/obj/Vtb_full10k" "+DATA=$DATA" "+NET=$NET" "+IMG_START=$S" "+IMG_END=$E"
            ;;
        *) echo "run_shard.sh: unknown simulator $SIM"; false ;;
    esac
} > run.log 2>&1
rc=$?
t1=$(date +%s.%N)
rm -rf xsim.dir                                # the per-shard snapshot copy is not needed afterwards
echo "$rc $t0 $t1" > status
if [[ $rc -eq 0 ]] && grep -q "^TEST PASSED" run.log && grep -q "^SHARD_DONE" run.log; then
    echo "shard $NET [$S,$E) $SIM PASS ($(echo "$t1 - $t0" | bc) s)"
else
    echo "shard $NET [$S,$E) $SIM FAIL rc=$rc (log: $DIR/run.log)" >&2
fi
exit 0                                         # failures are judged by the collector, not xargs
