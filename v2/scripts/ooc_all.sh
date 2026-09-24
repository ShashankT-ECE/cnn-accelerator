#!/usr/bin/env bash
# Out-of-context synthesis of every V2 leaf module (5.000 ns, xck26-sfvc784-2LV-c).
# Runs each in v2/build/ooc/<top>/ (up to OOC_JOBS in parallel, default 4) and
# writes v2/results/ooc_synth.csv (source=post_synth_ooc) via ooc_collect.py.
# For paper-grade rows run from a clean, committed tree.
set -uo pipefail
V2="$(cd "$(dirname "$0")/.." && pwd)"
PY="$V2/../.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
source "$HOME/Xilinx/Vivado/2023.1/settings64.sh" >/dev/null 2>&1
JOBS=1   # one Vivado job at a time (user rule after the OOM freeze); OOC_JOBS is ignored

# top : rtl sources (relative to v2/rtl, gos_pkg.sv first). PS_RD_LAT uses its default (1).
MODULES=(
    "gos_array:gos_pkg.sv gos_array.sv"
    "gos_act_buf:gos_pkg.sv gos_act_buf.sv"
    "gos_wgt_mem:gos_pkg.sv gos_wgt_mem.sv"
    "gos_qparam_mem:gos_pkg.sv gos_qparam_mem.sv"
    "gos_rotator:gos_pkg.sv gos_rotator.sv"
    "gos_pool:gos_pkg.sv gos_pool.sv"
    "gos_requant:gos_pkg.sv gos_requant.sv"
    "gos_csr:gos_pkg.sv gos_csr.sv"
    "gos_core:gos_pkg.sv gos_core.sv"
)
[[ $# -gt 0 ]] && { SEL=" $* "; } || SEL=""

run_one() {
    local top="$1"; shift
    local out="$V2/build/ooc/$top"
    "$V2/scripts/vivado_guard.sh" || { echo "ooc_all.sh: $top not started (guard)" >&2; return 1; }
    rm -rf "$out"; mkdir -p "$out"
    local srcs=()
    for f in "$@"; do srcs+=("$V2/rtl/$f"); done
    ( cd "$out" && vivado -mode batch -nojournal -log vivado.log -source "$V2/vivado/ooc_synth.tcl" \
        -tclargs "$top" "$out" "${srcs[@]}" > stdout.log 2>&1 )
    if [[ -f "$out/summary.json" ]]; then grep OOC_SUMMARY "$out/stdout.log" | tail -1
    else echo "ooc_all.sh: $top FAILED (see $out/vivado.log)" >&2; fi
}

TOPS=()
for entry in "${MODULES[@]}"; do
    top="${entry%%:*}"; files="${entry#*:}"
    [[ -n "$SEL" && "$SEL" != *" $top "* ]] && continue
    # additional sources from a "// GOS_OOC_DEPS:" header line of the top file
    deps="$(sed -n 's#^// GOS_OOC_DEPS:##p' "$V2/rtl/$top.sv" 2>/dev/null | head -1)"
    TOPS+=("$top")
    # shellcheck disable=SC2086
    run_one "$top" $files $deps &
    while [[ $(jobs -rp | wc -l) -ge $JOBS ]]; do sleep 2; done
done
wait
"$PY" "$V2/scripts/ooc_collect.py" "${TOPS[@]}"
