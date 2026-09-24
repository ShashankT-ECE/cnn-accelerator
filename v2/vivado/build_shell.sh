#!/usr/bin/env bash
# build_shell.sh — build the V2 Step 4 empty-shell bitstream for the KV260 and collect results.
#   v2/vivado/build_shell.sh [--bd-only] [--no-collect]
# Outputs (gitignored): v2/vivado/out/shell/{gos_shell.bit, gos_shell.hwh, gos_shell.bit.sha256,
#   utilization*.rpt, timing_summary.rpt, address_map.txt, critical_warnings.txt, summary.json, proj/}
# Then v2/scripts/impl_collect.py -> v2/results/impl_shell.csv (source=post_impl).
# For paper-grade rows run from a clean, committed tree.
set -euo pipefail
V2="$(cd "$(dirname "$0")/.." && pwd)"
PY="$V2/../.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
source "$HOME/Xilinx/Vivado/2023.1/settings64.sh" >/dev/null 2>&1

BD_ONLY=0; COLLECT=1
for a in "$@"; do
    case "$a" in
        --bd-only) BD_ONLY=1 ;;
        --no-collect) COLLECT=0 ;;
        *) echo "build_shell.sh: unknown option $a" >&2; exit 2 ;;
    esac
done

OUT="$V2/vivado/out/shell"
TOP=gos_shell_top
NAME=gos_shell
BUILD_ID="$(git -C "$V2" rev-parse --short=8 HEAD)"
mkdir -p "$OUT"
rm -f "$OUT"/*.bit "$OUT"/*.hwh "$OUT"/*.sha256 "$OUT"/summary.json

# Sources in compile order (package first).
cat > "$OUT/filelist.txt" <<EOF
$V2/rtl/gos_pkg.sv
$V2/rtl/gos_act_buf.sv
$V2/rtl/gos_wgt_mem.sv
$V2/rtl/gos_qparam_mem.sv
$V2/rtl/gos_top_ports.vh
$V2/rtl/gos_shell_scratch.v
$V2/rtl/gos_shell_top.v
EOF

"$V2/scripts/vivado_guard.sh" || exit 1
echo "build_shell.sh: top=$TOP build_id=$BUILD_ID out=$OUT"
( cd "$OUT" && vivado -mode batch -nojournal -log vivado.log -source "$V2/vivado/bd_shell.tcl" \
    -tclargs top=$TOP build_id=$BUILD_ID outdir="$OUT" filelist="$OUT/filelist.txt" name=$NAME \
    bd_only=$BD_ONLY > stdout.log 2>&1 ) || { echo "build_shell.sh: vivado FAILED (see $OUT/vivado.log)" >&2; exit 1; }
if [[ $BD_ONLY -eq 1 ]]; then echo "build_shell.sh: BD-only check done"; exit 0; fi

( cd "$OUT" && sha256sum "$NAME.bit" > "$NAME.bit.sha256" && sha256sum "$NAME.hwh" > "$NAME.hwh.sha256" )
grep BD_SHELL_SUMMARY "$OUT/stdout.log" | tail -1
cat "$OUT/$NAME.bit.sha256"
if [[ $COLLECT -eq 1 ]]; then "$PY" "$V2/scripts/impl_collect.py" "$OUT"; fi
