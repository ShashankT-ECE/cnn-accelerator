#!/usr/bin/env bash
# run_limits.sh — boundary-case ("limits") RTL simulation: every case of v2/shapes/limit_cases_*.py
# (gen_limits.py; e.g. KW / KH beyond the random-shape envelope, DECISIONS OC-3) through gos_core
# with the unchanged tb_shapes.sv, on BOTH Verilator and xsim, one simulator process per case.
# Label: RTL sim. Mismatches / refusals / timeouts are recorded as data (column "observed").
#
#   v2/scripts/run_limits.sh [--cases kw,fields] [--seed S] [--max-par P] [--timeout-s T]
#                            [--force] [--allow-dirty] [--ignore-vivado] [--csv PATH]
#
# Steps (details: v2/shapes/README.md, "Boundary cases"):
#   1. data     gen_limits.py gen -> v2/build/limits/<all|cases>/ (always regenerated; < 1 s)
#   2. build    tb_shapes + RTL once per commit / source hash and simulator
#               (v2/build/limits/builds/<tag>/<sim>/)
#   3. run      one shard per case ("range:j:j+1") on xsim and on Verilator, P processes in
#               parallel, each under `timeout T` (default 1800 s) so a hang cannot stop the others
#               (tb_shapes also has its own per-job timeout + soft reset). Complete shards are
#               reused on a rerun unless --force.
#   4. collect  gen_limits.py collect -> v2/results/limits_rtl.csv + v2/results/limits_rtl_logs_<key>.tar.xz
#               (sha256 in the CSV) ONLY when the tree is clean, every case module ran (no --cases)
#               and the seed is the default; otherwise <run dir>/limits_rtl.csv (or --csv PATH).
# Exit 0 iff every case has a RESULT from both simulators and they agree on every RESULT field.
# The run refuses a dirty tree unless --allow-dirty (rows then carry git_dirty=True).
set -uo pipefail
V2="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$V2/.." && pwd)"
PY="$REPO/.venv/bin/python"
SH="$V2/shapes"
OUT="$V2/build/limits"
export PYTHONDONTWRITEBYTECODE=1
VERILATOR_ROOT="${VERILATOR_ROOT:-$HOME/tools/verilator-src}"

CASES=""; SEED=""; MAXPAR=8; TMO=1800; FORCE=0; ALLOW_DIRTY=0; IGNORE_VIVADO=0; CSV_OUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cases) CASES="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --max-par) MAXPAR="$2"; shift ;;
        --timeout-s) TMO="$2"; shift ;;
        --force) FORCE=1 ;;
        --allow-dirty) ALLOW_DIRTY=1 ;;
        --ignore-vivado) IGNORE_VIVADO=1 ;;
        --csv) CSV_OUT="$2"; shift ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "run_limits.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
DEFAULT_SEED="$("$PY" -c "import sys; sys.path.insert(0, '$SH'); import gen_limits as g; print(g.DEFAULT_SEED)")" || exit 1
DEFAULT_SET=1
[[ -n "$SEED" && "$SEED" != "$DEFAULT_SEED" ]] && DEFAULT_SET=0
[[ -n "$CASES" ]] && DEFAULT_SET=0
SEED="${SEED:-$DEFAULT_SEED}"
T_START=$(date +%s)

# ---------------------------------------------------------------- git state
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
DIRTY="$("$PY" -c "import sys; sys.path.insert(0, '$V2/model'); import common; print(common.git_dirty())")"
if [[ "$DIRTY" == True && $ALLOW_DIRTY -eq 0 ]]; then
    echo "run_limits.sh: REFUSED — git tree is dirty (commit first, or --allow-dirty for a test run" \
         "whose rows are marked git_dirty=True and never go to v2/results)" >&2
    exit 1
fi
SRC_HASH="$(cat "$SH/tb_shapes.sv" "$SH/run_shard.sh" "$V2"/rtl/*.sv | sha256sum | cut -c1-12)"
TAG="${COMMIT:0:12}"; [[ "$DIRTY" == True ]] && TAG="$TAG-dirty-$SRC_HASH"
echo "run_limits.sh: commit $TAG, seed $SEED, cases ${CASES:-all}"

# ---------------------------------------------------------------- resources
free -h
if [[ $IGNORE_VIVADO -eq 0 ]] && pgrep -x vivado > /dev/null 2>&1; then
    echo "run_limits.sh: REFUSED — a Vivado job is running (one heavy job at a time):" >&2
    pgrep -ax vivado >&2; exit 1
fi
NPROC=$(nproc)
AVAIL_MB=$(free -m | awk '/^Mem:/ {print $7}')
if (( AVAIL_MB < 8192 )); then
    echo "run_limits.sh: REFUSED — available memory ${AVAIL_MB} MB < 8 GB" >&2; exit 1
fi
PROC_MB=800                                    # xsim per process (Verilator ~120 MB)
P=$MAXPAR
(( NPROC - 4 < P )) && P=$(( NPROC - 4 ))
MEM_P=$(( (AVAIL_MB - 8192) / PROC_MB ))
(( MEM_P < P )) && P=$MEM_P
(( P < 1 )) && P=1
echo "run_limits.sh: parallel processes P=$P (nproc $NPROC, avail ${AVAIL_MB} MB, ${PROC_MB} MB/process, max-par $MAXPAR)"

# ---------------------------------------------------------------- 1. data
CTAG="$("$PY" -c "import sys; sys.path.insert(0, '$SH'); import gen_limits as g; print(g.cases_tag(g._names('$CASES')))")" || exit 1
DATA="$OUT/$CTAG${SEED:+-s$SEED}"
"$PY" "$SH/gen_limits.py" gen --seed "$SEED" ${CASES:+--cases "$CASES"} --out "$DATA" || exit 1
read -r N SHASH < <("$PY" -c "import json; m = json.load(open('$DATA/shapes.json')); print(m['n_jobs'], m['shapeset_sha256'])")
KEY="$TAG-${SHASH:0:12}"
RUNS="$DATA/runs/$KEY"
HEX="$DATA/hex"
echo "run_limits.sh: set $DATA ($N cases, shapeset_sha256 ${SHASH:0:16}), run dir $RUNS"

# ---------------------------------------------------------------- 2. build (as run_shapes.sh)
build_sim() {   # build_sim <sim> -> $OUT/builds/$TAG/<sim>
    local sim="$1" B="$OUT/builds/$TAG/$1"
    [[ -f "$B/BUILD_OK" && "$(cat "$B/BUILD_OK")" == "$SRC_HASH" ]] && { echo "run_limits.sh: $sim build cached ($B)"; return 0; }
    rm -rf "$B"; mkdir -p "$B"
    local SRCS=() f
    for f in $(sed -n 's#^// GOS_SHAPES_TB:##p' "$SH/tb_shapes.sv" | head -1); do SRCS+=("$V2/rtl/$f"); done
    SRCS+=("$SH/tb_shapes.sv")
    echo "run_limits.sh: building $sim in $B"
    if [[ "$sim" == xsim ]]; then
        ( cd "$B" && source "$HOME/Xilinx/Vivado/2023.1/settings64.sh" >/dev/null 2>&1 &&
          xvlog -sv -i "$V2/rtl" "${SRCS[@]}" && xelab tb_shapes -s snap -timescale 1ns/1ps --debug off &&
          xsim -version | head -1 | awk '{print $NF}' > sim_version.txt ) > "$B/build.log" 2>&1
    else
        [[ -x "$VERILATOR_ROOT/bin/verilator" ]] || { echo "Verilator not found at $VERILATOR_ROOT" >&2; return 1; }
        ( cd "$B" && export VERILATOR_ROOT &&
          nice -n 5 "$VERILATOR_ROOT/bin/verilator" --binary --timing -O3 -j 4 --top-module tb_shapes \
              -I"$V2/rtl" "${SRCS[@]}" -Mdir obj &&
          "$VERILATOR_ROOT/bin/verilator" --version | awk '{print $2}' > sim_version.txt ) > "$B/build.log" 2>&1
    fi
    local rc=$?
    [[ $rc -eq 0 ]] && echo "$SRC_HASH" > "$B/BUILD_OK" || { tail -20 "$B/build.log" >&2; echo "build $sim FAILED" >&2; }
    return $rc
}
build_sim verilator || exit 1
build_sim xsim || exit 1

# ---------------------------------------------------------------- 3. run (one shard per case, both sims)
SELS=(); for (( j = 0; j < N; j++ )); do SELS+=("range:$j:$((j + 1))"); done
LINES=()
for sim in xsim verilator; do
    R="$RUNS/$sim"; B="$OUT/builds/$TAG/$sim"
    mkdir -p "$R"
    plan="$(printf "%s\n" "${SELS[@]}")"
    if [[ -f "$R/plan.txt" && "$(cat "$R/plan.txt")" != "$plan" ]]; then
        echo "run_limits.sh: [$sim] shard plan changed — discarding old shards"; rm -rf "$R"; mkdir -p "$R"
    fi
    printf "%s\n" "$plan" > "$R/plan.txt"
    n_skip=0
    for sel in "${SELS[@]}"; do
        SD="$R/$("$PY" -c "import sys; sys.path.insert(0, '$SH'); import collect_shapes as c; print(c.shard_dir_name('$sel'))")"
        if [[ $FORCE -eq 0 ]] && "$PY" "$SH/collect_shapes.py" check-shard "$SD" "$sel" > /dev/null 2>&1; then
            n_skip=$((n_skip + 1)); continue
        fi
        LINES+=("$TMO $sim $B $HEX $sel $SD")
    done
    echo "run_limits.sh: [$sim] $N cases, $n_skip complete (reused)"
done
# xsim first in the list (slower start-up); a shard killed by `timeout` leaves no RESULT -> observed=no_result
printf "%s\n" "${LINES[@]}" | grep -v '^$' | \
    xargs -r -P "$P" -L 1 sh -c 'timeout --kill-after=30 "$0" bash "'"$SH"'/run_shard.sh" "$@" || echo "shard $4 $1 killed/failed rc=$?" >&2'

# ---------------------------------------------------------------- 4. collect
CSV="${CSV_OUT:-$RUNS/limits_rtl.csv}"; ARCH=()
if [[ -z "$CSV_OUT" && "$DIRTY" == False && $DEFAULT_SET -eq 1 ]]; then
    CSV="$V2/results/limits_rtl.csv"
    ARCH=(--archive "$V2/results/limits_rtl_logs_$KEY.tar.xz")
fi
"$PY" "$SH/gen_limits.py" collect --data "$DATA" --runs "$RUNS" --csv "$CSV" "${ARCH[@]}"
rc=$?
echo "run_limits.sh: $([[ $rc -eq 0 ]] && echo "COMPLETE, simulators agree" || echo "INCOMPLETE or simulators DISAGREE") ($(( $(date +%s) - T_START )) s this invocation)"
exit $rc
