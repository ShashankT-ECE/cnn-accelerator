#!/usr/bin/env bash
# build_gos.sh — full V2 design (gos_top_wrapper) in the Step 4 block design (bd_shell.tcl, unchanged BD).
#   v2/vivado/build_gos.sh <MHz> [--strategy default|explore] [--tol <pct>] [--bd-only]
# Outputs (gitignored): v2/vivado/out/gos_<MHz>[_<strategy>]/{gos_<MHz>.bit, gos_<MHz>.hwh, *.sha256,
#   utilization*.rpt, timing_summary.rpt, worst_path.rpt, power.rpt, methodology.rpt, drc.rpt,
#   check_timing.rpt, static_counts.txt, address_map.txt, critical_warnings.txt, summary.json, proj/}
# Results rows: v2/scripts/impl_collect.py --csv v2/results/impl_gos.csv <outdirs...> (run after all variants).
# For paper-grade results run from a clean, committed tree (BUILD_ID = 8-hex short commit).
set -euo pipefail
V2="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/Xilinx/Vivado/2023.1/settings64.sh" >/dev/null 2>&1

[[ $# -ge 1 ]] || { echo "usage: $0 <MHz> [--strategy default|explore] [--tol pct] [--bd-only]" >&2; exit 2; }
MHZ="$1"; shift
STRATEGY=default; TOL=1.0; BD_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --strategy) STRATEGY="$2"; shift 2 ;;
        --tol) TOL="$2"; shift 2 ;;
        --bd-only) BD_ONLY=1; shift ;;
        *) echo "build_gos.sh: unknown option $1" >&2; exit 2 ;;
    esac
done
NAME="gos_${MHZ}"
OUT="$V2/vivado/out/gos_${MHZ}"
[[ "$STRATEGY" == default ]] || OUT="${OUT}_${STRATEGY}"
BUILD_ID="$(git -C "$V2" rev-parse --short=8 HEAD)"
mkdir -p "$OUT"
rm -f "$OUT"/*.bit "$OUT"/*.hwh "$OUT"/*.sha256 "$OUT"/summary.json

cat > "$OUT/filelist.txt" <<FL
$V2/rtl/gos_pkg.sv
$V2/rtl/gos_cfg_check.sv
$V2/rtl/gos_ctrl.sv
$V2/rtl/gos_act_buf.sv
$V2/rtl/gos_wgt_mem.sv
$V2/rtl/gos_qparam_mem.sv
$V2/rtl/gos_rotator.sv
$V2/rtl/gos_pe.sv
$V2/rtl/gos_array.sv
$V2/rtl/gos_requant.sv
$V2/rtl/gos_pool.sv
$V2/rtl/gos_core.sv
$V2/rtl/gos_csr.sv
$V2/rtl/gos_top.sv
$V2/rtl/gos_top_ports.vh
$V2/rtl/gos_top_wrapper.v
FL

echo "build_gos.sh: top=gos_top_wrapper ${MHZ} MHz strategy=$STRATEGY build_id=$BUILD_ID out=$OUT"
( cd "$OUT" && vivado -mode batch -nojournal -log vivado.log -source "$V2/vivado/bd_shell.tcl" \
    -tclargs top=gos_top_wrapper build_id=$BUILD_ID outdir="$OUT" filelist="$OUT/filelist.txt" \
    name=$NAME pl_mhz=$MHZ freq_tol_pct=$TOL strategy=$STRATEGY bd_only=$BD_ONLY jobs=6 \
    > stdout.log 2>&1 ) || { echo "build_gos.sh: vivado FAILED (see $OUT/vivado.log)" >&2; exit 1; }
if [[ $BD_ONLY -eq 1 ]]; then echo "build_gos.sh: BD-only check done"; exit 0; fi
( cd "$OUT" && sha256sum "$NAME.bit" > "$NAME.bit.sha256" && sha256sum "$NAME.hwh" > "$NAME.hwh.sha256" )
grep BD_SHELL_SUMMARY "$OUT/stdout.log" | tail -1
cat "$OUT/$NAME.bit.sha256"
