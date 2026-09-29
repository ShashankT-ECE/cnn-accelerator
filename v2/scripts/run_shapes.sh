#!/usr/bin/env bash
# run_shapes.sh — A3-general random-shape RTL simulation: ~300 seeded random multi-layer gos_ jobs
# (v2/shapes/gen_shapes.py) through gos_core, outputs bit-exact + LAYER_CYC/TOTAL_CYC/MAC_ACTIVE
# exact vs the model, refusal jobs vs the expected ERR_CODE. Label: RTL sim.
#
#   v2/scripts/run_shapes.sh [--seed S] [--n-jobs N] [--sim verilator|xsim] [--jobs-per-shard M]
#                            [--limit K] [--max-par P] [--xcheck-min X] [--force] [--allow-dirty]
#                            [--no-xval] [--ignore-vivado] [--csv PATH]
#
# Steps (details: v2/shapes/README.md):
#   1. data     gen_shapes.py --if-stale -> v2/build/shapes/<seed>[-n<N>]/ (regenerated unless made
#               by the same generator + model code at this commit / dirty state)
#   2. build    tb_shapes + RTL once per commit and simulator (v2/build/shapes/builds/<tag>/<sim>/)
#   3. xval     (--sim verilator only) the shape set's xsim cross-check jobs (hex/xcheck.hex, the 2
#               cheapest per category, >= --xcheck-min, default 20) on BOTH xsim and Verilator:
#               identical RESULT fields per job and every job exact. REFUSES to continue otherwise.
#               Cached per (commit, source hash, shapeset).
#   4. shards   jobs [0, K) in shards of M (default 20), P shard processes in parallel
#               (P = min(--max-par [8], nproc - 4, (available MB - 8192) / per-process MB), >= 1);
#               complete shards are skipped on a rerun (resume); --force reruns everything.
#   5. collect  v2/shapes/collect_shapes.py collect -> v2/results/shapes_rtl.csv + compressed logs
#               v2/results/shapes_rtl_logs_<key>.tar.xz (sha256 in the CSV) ONLY when the tree is
#               clean, every job ran (no --limit) and the seed / job count are the defaults;
#               otherwise <run dir>/shapes_rtl.csv (or --csv PATH).
# The run refuses a dirty tree unless --allow-dirty (rows then carry git_dirty=True and never go to
# v2/results). Exit 0 only if every job is exact.
set -uo pipefail
V2="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$V2/.." && pwd)"
PY="$REPO/.venv/bin/python"
SH="$V2/shapes"
OUT="$V2/build/shapes"
export PYTHONDONTWRITEBYTECODE=1
VERILATOR_ROOT="${VERILATOR_ROOT:-$HOME/tools/verilator-src}"

SEED=""; NJOBS=""; SIM=verilator; JPS=20; LIMIT=""; MAXPAR=8; XMIN=20
FORCE=0; ALLOW_DIRTY=0; XVAL=1; IGNORE_VIVADO=0; CSV_OUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift ;;
        --n-jobs) NJOBS="$2"; shift ;;
        --sim) SIM="$2"; shift ;;
        --jobs-per-shard) JPS="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --max-par) MAXPAR="$2"; shift ;;
        --xcheck-min) XMIN="$2"; shift ;;
        --force) FORCE=1 ;;
        --allow-dirty) ALLOW_DIRTY=1 ;;
        --no-xval) XVAL=0 ;;
        --ignore-vivado) IGNORE_VIVADO=1 ;;
        --csv) CSV_OUT="$2"; shift ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "run_shapes.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
[[ "$SIM" == xsim || "$SIM" == verilator ]] || { echo "--sim must be xsim or verilator" >&2; exit 2; }
(( JPS >= 1 )) || { echo "--jobs-per-shard must be >= 1" >&2; exit 2; }
DEFAULT_SEED="$("$PY" -c "import sys; sys.path.insert(0, '$SH'); import gen_shapes as g; print(g.DEFAULT_SEED)")" || exit 1
DEFAULT_SET=1
[[ -n "$SEED" && "$SEED" != "$DEFAULT_SEED" ]] && DEFAULT_SET=0
[[ -n "$NJOBS" ]] && DEFAULT_SET=0
SEED="${SEED:-$DEFAULT_SEED}"
T_START=$(date +%s)

# ---------------------------------------------------------------- git state
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
DIRTY="$("$PY" -c "import sys; sys.path.insert(0, '$V2/model'); import common; print(common.git_dirty())")"
if [[ "$DIRTY" == True && $ALLOW_DIRTY -eq 0 ]]; then
    echo "run_shapes.sh: REFUSED — git tree is dirty (commit first, or --allow-dirty for a test run" \
         "whose rows are marked git_dirty=True and never go to v2/results)" >&2
    exit 1
fi
SRC_HASH="$(cat "$SH/tb_shapes.sv" "$SH/run_shard.sh" "$V2"/rtl/*.sv | sha256sum | cut -c1-12)"
TAG="${COMMIT:0:12}"; [[ "$DIRTY" == True ]] && TAG="$TAG-dirty-$SRC_HASH"
echo "run_shapes.sh: commit $TAG, sim $SIM, seed $SEED${NJOBS:+, n-jobs $NJOBS}"

# ---------------------------------------------------------------- resources
free -h
if [[ $IGNORE_VIVADO -eq 0 ]] && pgrep -x vivado > /dev/null 2>&1; then
    echo "run_shapes.sh: REFUSED — a Vivado job is running (one heavy job at a time):" >&2
    pgrep -ax vivado >&2; exit 1
fi
NPROC=$(nproc)
AVAIL_MB=$(free -m | awk '/^Mem:/ {print $7}')
if (( AVAIL_MB < 8192 )); then
    echo "run_shapes.sh: REFUSED — available memory ${AVAIL_MB} MB < 8 GB" >&2; exit 1
fi
case "$SIM" in xsim) PROC_MB=800 ;; verilator) PROC_MB=128 ;; esac   # xsim: as fullsim; Verilator RSS ~120 MB
P=$MAXPAR
(( NPROC - 4 < P )) && P=$(( NPROC - 4 ))
MEM_P=$(( (AVAIL_MB - 8192) / PROC_MB ))
(( MEM_P < P )) && P=$MEM_P
(( P < 1 )) && P=1
echo "run_shapes.sh: parallel shards P=$P (nproc $NPROC, avail ${AVAIL_MB} MB, ${PROC_MB} MB/process, max-par $MAXPAR)"

# ---------------------------------------------------------------- 1. data
DATA="$OUT/$SEED${NJOBS:+-n$NJOBS}"
"$PY" "$SH/gen_shapes.py" --seed "$SEED" ${NJOBS:+--n-jobs "$NJOBS"} --out "$DATA" --if-stale || exit 1
read -r N SHASH < <("$PY" -c "import json; m = json.load(open('$DATA/shapes.json')); print(m['n_jobs'], m['shapeset_sha256'])")
LIMIT="${LIMIT:-$N}"
(( LIMIT >= 1 && LIMIT <= N )) || { echo "--limit must be 1..$N" >&2; exit 2; }
KEY="$TAG-${SHASH:0:12}"
RUNS="$DATA/runs/$KEY"
RUN="$RUNS/$SIM"
HEX="$DATA/hex"
echo "run_shapes.sh: shape set $DATA ($N jobs, shapeset_sha256 ${SHASH:0:16}), limit $LIMIT, run dir $RUN"

# ---------------------------------------------------------------- 2. build
build_sim() {   # build_sim <sim> -> $OUT/builds/$TAG/<sim>
    local sim="$1" B="$OUT/builds/$TAG/$1"
    [[ -f "$B/BUILD_OK" && "$(cat "$B/BUILD_OK")" == "$SRC_HASH" ]] && return 0
    rm -rf "$B"; mkdir -p "$B"
    local SRCS=() f
    for f in $(sed -n 's#^// GOS_SHAPES_TB:##p' "$SH/tb_shapes.sv" | head -1); do SRCS+=("$V2/rtl/$f"); done
    SRCS+=("$SH/tb_shapes.sv")
    echo "run_shapes.sh: building $sim in $B"
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

# run_set <sim> <run_dir> <force> <sel...>: plan + resume + parallel shards
run_set() {
    local sim="$1" R="$2" force="$3"; shift 3
    local B="$OUT/builds/$TAG/$sim" jobs=() sel n_skip=0 plan
    mkdir -p "$R"
    plan="$(printf "%s\n" "$@")"
    if [[ -f "$R/plan.txt" && "$(cat "$R/plan.txt")" != "$plan" ]]; then
        echo "run_shapes.sh: shard plan changed — discarding old shards in $R"
        rm -rf "$R"; mkdir -p "$R"
    fi
    printf "%s\n" "$plan" > "$R/plan.txt"
    for sel in "$@"; do
        local SD; SD="$R/$("$PY" -c "import sys; sys.path.insert(0, '$SH'); import collect_shapes as c; print(c.shard_dir_name('$sel'))")"
        if [[ $force -eq 0 ]] && "$PY" "$SH/collect_shapes.py" check-shard "$SD" "$sel" > /dev/null 2>&1; then
            n_skip=$((n_skip + 1)); continue
        fi
        jobs+=("$sim $B $HEX $sel $SD")
    done
    echo "run_shapes.sh: [$sim] $# shards, $n_skip complete (skipped) in $R"
    printf "%s\n" "${jobs[@]}" | grep -v '^$' | xargs -r -P "$P" -L 1 bash "$SH/run_shard.sh"
}

build_sim "$SIM" || exit 1

# ---------------------------------------------------------------- 3. xval (Verilator only)
XVJ=""
if [[ "$SIM" == verilator && $XVAL -eq 1 ]]; then
    XV="$RUNS/xval"
    XVJ="$XV/xval.json"
    if [[ -f "$XV/PASS" && "$(cat "$XV/PASS")" == "$SRC_HASH" ]]; then
        echo "run_shapes.sh: xval already PASS for $KEY"
    else
        build_sim xsim || exit 1
        mkdir -p "$XV/lists"
        # split the cross-check list into up to 4 chunks (same chunks for both simulators)
        "$PY" - "$HEX/xcheck.hex" "$XV/lists" "$(( P < 4 ? P : 4 ))" <<'EOF' || exit 1
import sys
from pathlib import Path
vals = [int(t, 16) for t in Path(sys.argv[1]).read_text().split()]
ids, k = vals[1:1 + vals[0]], max(1, int(sys.argv[3]))
for p in Path(sys.argv[2]).glob("x*.hex"):
    p.unlink()
for i in range(k):
    part = ids[i::k]
    if part:
        Path(sys.argv[2], f"x{i}.hex").write_text("".join(f"{v:08x}\n" for v in [len(part)] + part))
EOF
        SELS=(); for f in "$XV"/lists/x*.hex; do SELS+=("list:$f"); done
        run_set xsim "$XV/xsim" 1 "${SELS[@]}"
        run_set verilator "$XV/verilator" 1 "${SELS[@]}"
        if "$PY" "$SH/collect_shapes.py" xval --a "$XV/xsim" --b "$XV/verilator" --data "$DATA" \
               --min-jobs "$XMIN" --out "$XVJ"; then
            echo "$SRC_HASH" > "$XV/PASS"
        else
            echo "run_shapes.sh: REFUSED — Verilator does not reproduce xsim on the cross-check jobs (see $XVJ)" >&2
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------- 4. shards
SELS=()
for (( s = 0; s < LIMIT; s += JPS )); do
    e=$(( s + JPS < LIMIT ? s + JPS : LIMIT )); SELS+=("range:$s:$e")
done
run_set "$SIM" "$RUN" "$FORCE" "${SELS[@]}"

# ---------------------------------------------------------------- 5. collect
CSV="${CSV_OUT:-$RUN/shapes_rtl.csv}"; ARCH=()
if [[ -z "$CSV_OUT" && "$DIRTY" == False && $LIMIT -eq $N && $DEFAULT_SET -eq 1 ]]; then
    CSV="$V2/results/shapes_rtl.csv"
    ARCH=(--archive "$V2/results/shapes_rtl_logs_$KEY.tar.xz")
fi
"$PY" "$SH/collect_shapes.py" collect --run-dir "$RUN" --data "$DATA" --csv "$CSV" \
    --shard-csv "$RUN/shards.csv" ${XVJ:+--xval-json "$XVJ"} "${ARCH[@]}"
rc=$?
echo "run_shapes.sh: $([[ $rc -eq 0 ]] && echo ALL EXACT || echo FAILURES) ($(( $(date +%s) - T_START )) s this invocation)"
exit $rc
