#!/usr/bin/env bash
# run_full10k.sh — full-test-set RTL simulation (all 10,000 test images per net) of gos_core,
# bit-exact LOGIT + exact LAYER_CYC/TOTAL_CYC vs the frozen golden model. Label: RTL sim.
#
#   v2/scripts/run_full10k.sh [--nets lenet5 cifar10] [--sim verilator|xsim]
#                             [--images-per-shard M | --shards N] [--limit K] [--max-par P]
#                             [--force] [--allow-dirty] [--no-xval] [--ignore-vivado]
#
# Steps (details: v2/fullsim/README.md):
#   1. data     v2/fullsim/gen_full10k_data.py -> v2/build/fullsim/data/<net>/ (regenerated when
#               missing or made at another commit; images 0..9 are checked byte-identical to
#               the gen_vectors.py net vectors)
#   2. build    compile tb_full10k + RTL once per commit and simulator
#               (v2/build/fullsim/runs/<commit12>[-dirty]/<sim>/build/)
#   3. xval     (--sim verilator only) images 0..9 per net on BOTH xsim and Verilator: per-image
#               identical LOGIT / LAYER_CYC / TOTAL_CYC / MAC / STALL, and cycles identical to
#               v2/results/rtl_network.csv (xsim tb_gos_core SUITE=net). The run refuses if xval
#               fails. Cached per commit.
#   4. shards   images [0, K) per net split into shards; P shard processes in parallel
#               (P = min(--max-par, nproc - 4, (available MB - 8192) / per-process MB), >= 1).
#               A shard whose log is COMPLETE (collect_full10k.py check-shard) is skipped on a
#               rerun (resume); --force reruns everything.
#   5. collect  v2/fullsim/collect_full10k.py collect -> v2/results/rtl_full10k.csv when the tree
#               is clean and K = 10000 for every net; otherwise <run dir>/rtl_full10k.csv only.
#               Per-shard CSV: <run dir>/shards.csv.
# Resource rule: xsim/Verilator are not Vivado jobs, but this is a heavy multi-process run:
# it prints `free -h`, refuses while a vivado process is running (unless --ignore-vivado) and
# keeps 4 cores and 8 GB free. Exit 0 only if every image of every net is exact.
set -uo pipefail
V2="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$V2/.." && pwd)"
PY="$REPO/.venv/bin/python"
FS="$V2/fullsim"
OUT="$V2/build/fullsim"
export PYTHONDONTWRITEBYTECODE=1
VERILATOR_ROOT="${VERILATOR_ROOT:-$HOME/tools/verilator-src}"

NETS=(lenet5 cifar10); SIM=verilator; IPS=500; NSHARDS=""; LIMIT=10000; MAXPAR=12
FORCE=0; ALLOW_DIRTY=0; XVAL=1; IGNORE_VIVADO=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nets) shift; NETS=(); while [[ $# -gt 0 && "$1" != --* ]]; do NETS+=("$1"); shift; done; continue ;;
        --sim) SIM="$2"; shift ;;
        --images-per-shard) IPS="$2"; shift ;;
        --shards) NSHARDS="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --max-par) MAXPAR="$2"; shift ;;
        --force) FORCE=1 ;;
        --allow-dirty) ALLOW_DIRTY=1 ;;
        --no-xval) XVAL=0 ;;
        --ignore-vivado) IGNORE_VIVADO=1 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "run_full10k.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done
[[ "$SIM" == xsim || "$SIM" == verilator ]] || { echo "--sim must be xsim or verilator" >&2; exit 2; }
(( LIMIT >= 1 && LIMIT <= 10000 )) || { echo "--limit must be 1..10000" >&2; exit 2; }
T_START=$(date +%s)

# ---------------------------------------------------------------- git state
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
DIRTY="$("$PY" -c "import sys; sys.path.insert(0, '$V2/model'); import common; print(common.git_dirty())")"
if [[ "$DIRTY" == True && $ALLOW_DIRTY -eq 0 ]]; then
    echo "run_full10k.sh: REFUSED — git tree is dirty (commit first, or --allow-dirty for a" \
         "test run whose rows are marked git_dirty=True and never go to v2/results)" >&2
    exit 1
fi
# hash of everything a shard result depends on besides v2/model (data carries its own check)
SRC_HASH="$(cat "$FS/tb_full10k.sv" "$FS/run_shard.sh" "$V2"/rtl/*.sv | sha256sum | cut -c1-12)"
GEN_SHA="$(sha256sum "$FS/gen_full10k_data.py" | cut -d' ' -f1)"
TAG="${COMMIT:0:12}"; [[ "$DIRTY" == True ]] && TAG="$TAG-dirty-$SRC_HASH"
RUN="$OUT/runs/$TAG/$SIM"
echo "run_full10k.sh: commit $TAG, sim $SIM, nets ${NETS[*]}, limit $LIMIT, run dir $RUN"

# ---------------------------------------------------------------- resources
free -h
if [[ $IGNORE_VIVADO -eq 0 ]] && pgrep -x vivado > /dev/null 2>&1; then
    echo "run_full10k.sh: REFUSED — a Vivado job is running (one heavy job at a time):" >&2
    pgrep -ax vivado >&2; exit 1
fi
NPROC=$(nproc)
AVAIL_MB=$(free -m | awk '/^Mem:/ {print $7}')
case "$SIM" in xsim) PROC_MB=800 ;; verilator) PROC_MB=64 ;; esac   # measured RSS + margin (README)
P=$MAXPAR
(( NPROC - 4 < P )) && P=$(( NPROC - 4 ))
MEM_P=$(( (AVAIL_MB - 8192) / PROC_MB ))
(( MEM_P < P )) && P=$MEM_P
(( P < 1 )) && P=1
echo "run_full10k.sh: parallel shards P=$P (nproc $NPROC, avail ${AVAIL_MB} MB, ${PROC_MB} MB/process, max-par $MAXPAR)"

# ---------------------------------------------------------------- 1. data
DATA="$OUT/data"
need_data=()
for net in "${NETS[@]}"; do
    ok="$("$PY" - "$DATA/$net/meta.json" "$COMMIT" "$GEN_SHA" <<'EOF'
import json, sys
try:
    m = json.load(open(sys.argv[1]))
    print(int(m["git_commit"] == sys.argv[2] and m["n_images"] == 10000
              and m["generator_sha256"] == sys.argv[3]))
except Exception:
    print(0)
EOF
)"
    [[ "$ok" == 1 ]] || need_data+=("$net")
done
if [[ ${#need_data[@]} -gt 0 ]]; then
    echo "run_full10k.sh: generating data for ${need_data[*]} (10000 images each)"
    "$PY" "$FS/gen_full10k_data.py" --nets "${need_data[@]}" --out "$DATA" || exit 1
fi

# ---------------------------------------------------------------- 2. build
build_sim() {   # build_sim <sim> -> $OUT/runs/$TAG/<sim>/build
    local sim="$1" B="$OUT/runs/$TAG/$1/build"
    [[ -f "$B/BUILD_OK" && "$(cat "$B/BUILD_OK")" == "$SRC_HASH" ]] && return 0
    rm -rf "$B"; mkdir -p "$B"
    local SRCS=() f
    for f in $(sed -n 's#^// GOS_FULL10K_TB:##p' "$FS/tb_full10k.sv" | head -1); do SRCS+=("$V2/rtl/$f"); done
    SRCS+=("$FS/tb_full10k.sv")
    echo "run_full10k.sh: building $sim in $B"
    if [[ "$sim" == xsim ]]; then
        ( cd "$B" && source "$HOME/Xilinx/Vivado/2023.1/settings64.sh" >/dev/null 2>&1 &&
          xvlog -sv -i "$V2/rtl" "${SRCS[@]}" && xelab tb_full10k -s snap -timescale 1ns/1ps --debug off &&
          xsim -version | head -1 | awk '{print $NF}' > sim_version.txt ) > "$B/build.log" 2>&1
    else
        [[ -x "$VERILATOR_ROOT/bin/verilator" ]] || { echo "Verilator not found at $VERILATOR_ROOT" >&2; return 1; }
        ( cd "$B" && export VERILATOR_ROOT &&
          nice -n 5 "$VERILATOR_ROOT/bin/verilator" --binary --timing -O3 -j 4 --top-module tb_full10k \
              -I"$V2/rtl" "${SRCS[@]}" -Mdir obj &&
          "$VERILATOR_ROOT/bin/verilator" --version | awk '{print $2}' > sim_version.txt ) > "$B/build.log" 2>&1
    fi
    local rc=$?
    [[ $rc -eq 0 ]] && echo "$SRC_HASH" > "$B/BUILD_OK" || { tail -20 "$B/build.log" >&2; echo "build $sim FAILED" >&2; }
    return $rc
}

# run_set <sim> <run_dir> <ips> <limit> <force> <nets...>: plan + resume + parallel shards
run_set() {
    local sim="$1" R="$2" ips="$3" lim="$4" force="$5"; shift 5
    local B="$OUT/runs/$TAG/$sim/build" jobs=() net s e n_skip
    for net in "$@"; do
        mkdir -p "$R/$net"
        local plan="" nsh
        if [[ -n "$NSHARDS" && "$R" == "$RUN" ]]; then
            nsh=$NSHARDS; ips=$(( (lim + nsh - 1) / nsh ))
        fi
        for (( s = 0; s < lim; s += ips )); do
            e=$(( s + ips < lim ? s + ips : lim )); plan+="$s $e"$'\n'
        done
        if [[ -f "$R/$net/plan.txt" && "$(cat "$R/$net/plan.txt")" != "${plan%$'\n'}" ]]; then
            echo "run_full10k.sh: $net shard plan changed — discarding old shards in $R/$net"
            rm -rf "$R/$net"; mkdir -p "$R/$net"
        fi
        printf "%s" "$plan" > "$R/$net/plan.txt"
        n_skip=0
        while read -r s e; do
            [[ -z "$s" ]] && continue
            local SD; SD="$R/$net/$(printf 'shard_%05d_%05d' "$s" "$e")"
            if [[ $force -eq 0 ]] && "$PY" "$FS/collect_full10k.py" check-shard "$SD" "$net" "$s" "$e" > /dev/null 2>&1; then
                n_skip=$((n_skip + 1)); continue
            fi
            jobs+=("$sim $B $DATA/$net $net $s $e $SD")
        done < "$R/$net/plan.txt"
        echo "run_full10k.sh: $net [$sim]: $(wc -l < "$R/$net/plan.txt") shards, $n_skip complete (skipped)"
    done
    # cifar10 shards first (longest), then the rest; P at a time
    printf "%s\n" "${jobs[@]}" | grep -v '^$' | sort -t' ' -k4,4 -s |
        xargs -r -P "$P" -L 1 bash "$FS/run_shard.sh"
}

build_sim "$SIM" || exit 1

# ---------------------------------------------------------------- 3. xval (Verilator only)
if [[ "$SIM" == verilator && $XVAL -eq 1 ]]; then
    XV="$OUT/runs/$TAG/xval"
    if [[ -f "$XV/PASS" && "$(cat "$XV/PASS")" == "$SRC_HASH" ]]; then
        echo "run_full10k.sh: xval already PASS for $TAG"
    else
        build_sim xsim || exit 1
        mkdir -p "$XV"
        run_set xsim "$XV/xsim" 10 10 1 lenet5 cifar10
        run_set verilator "$XV/verilator" 10 10 1 lenet5 cifar10
        if "$PY" "$FS/collect_full10k.py" xval --a "$XV/xsim" --b "$XV/verilator" \
               --nets lenet5 cifar10 --out "$XV/xval.json"; then
            echo "$SRC_HASH" > "$XV/PASS"
        else
            echo "run_full10k.sh: REFUSED — Verilator does not reproduce xsim bit-exactly (see $XV/xval.json)" >&2
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------- 4. shards
run_set "$SIM" "$RUN" "$IPS" "$LIMIT" "$FORCE" "${NETS[@]}"

# ---------------------------------------------------------------- 5. collect
CSV="$RUN/rtl_full10k.csv"
if [[ "$DIRTY" == False && $LIMIT -eq 10000 ]]; then CSV="$V2/results/rtl_full10k.csv"; fi
"$PY" "$FS/collect_full10k.py" collect --run-dir "$RUN" --data-dir "$DATA" --nets "${NETS[@]}" \
    --csv "$CSV" --shard-csv "$RUN/shards.csv"
rc=$?
echo "run_full10k.sh: $([[ $rc -eq 0 ]] && echo ALL EXACT || echo FAILURES) ($(( $(date +%s) - T_START )) s this invocation)"
exit $rc
