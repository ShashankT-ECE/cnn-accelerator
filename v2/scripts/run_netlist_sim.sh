#!/usr/bin/env bash
# run_netlist_sim.sh — post-synthesis functional simulation of gos_top (V2 step 4.5, Part B2).
# 1) netlist_synth.tcl: synth_design (OOC) + write_verilog -mode funcsim -> v2/build/netlist/
# 2) xsim with the UNISIM library on the netlist: tb_gos_top (1 LeNet + 1 CIFAR image) and
#    tb_gos_top_backtoback with +NJOBS=${NETLIST_NJOBS:-10} (both compiled with -d NETLIST).
# Runs in v2/build/sim/netlist_<tb>/; PASS iff "TEST PASSED" and no ERROR/FATAL/"TEST FAILED".
set -uo pipefail
V2="$(cd "$(dirname "$0")/.." && pwd)"
source "$HOME/Xilinx/Vivado/2023.1/settings64.sh" >/dev/null 2>&1
NL="$V2/build/netlist"
BID=1EB08730                          # the TBs expect this BUILD_ID
RTL=(gos_pkg.sv gos_csr.sv gos_cfg_check.sv gos_ctrl.sv gos_act_buf.sv gos_wgt_mem.sv gos_qparam_mem.sv
     gos_rotator.sv gos_pe.sv gos_array.sv gos_requant.sv gos_pool.sv gos_core.sv gos_top.sv)
SRCS=(); for f in "${RTL[@]}"; do SRCS+=("$V2/rtl/$f"); done

if [[ "${SKIP_SYNTH:-0}" != 1 ]]; then
    "$V2/scripts/vivado_guard.sh" || exit 1
    rm -rf "$NL"; mkdir -p "$NL"
    ( cd "$NL" && vivado -mode batch -nojournal -log synth.log -source "$V2/vivado/netlist_synth.tcl" \
        -tclargs "$NL" "$BID" "${SRCS[@]}" > stdout.log 2>&1 )
    grep -q NETLIST_DONE "$NL/stdout.log" || { echo "run_netlist_sim.sh: synthesis FAILED ($NL/synth.log)" >&2; exit 1; }
fi

fail=0
run_tb() {
    local tb="$1"; shift
    local W="$V2/build/sim/netlist_$tb"
    rm -rf "$W"; mkdir -p "$W"
    ( cd "$W" &&
      xvlog -sv -d NETLIST "$V2/rtl/gos_pkg.sv" "$NL/gos_top_funcsim.v" "$V2/tb/$tb.sv" > run.log 2>&1 &&
      xvlog "$XILINX_VIVADO/data/verilog/src/glbl.v" >> run.log 2>&1 &&
      xelab "$tb" glbl -s snap -L unisims_ver -L secureip -timescale 1ns/1ps --debug off >> run.log 2>&1 &&
      xsim snap -R -testplusarg "VEC_DIR=$V2/vectors/generated" "$@" >> run.log 2>&1 )
    if grep -q "TEST PASSED" "$W/run.log" && ! grep -qE "^ERROR|ERROR:|FATAL|Fatal:|TEST FAILED" "$W/run.log"; then
        echo "run_netlist_sim.sh: $tb PASS ($(grep 'TEST PASSED' "$W/run.log" | tail -1))"
    else
        tail -20 "$W/run.log" >&2; echo "run_netlist_sim.sh: $tb FAIL ($W/run.log)" >&2; fail=1
    fi
}
run_tb tb_gos_top &
run_tb tb_gos_top_backtoback -testplusarg "NJOBS=${NETLIST_NJOBS:-10}" &
wait
# re-evaluate from logs (background subshells cannot set fail)
for tb in tb_gos_top tb_gos_top_backtoback; do
    grep -q "TEST PASSED" "$V2/build/sim/netlist_$tb/run.log" 2>/dev/null || fail=1
done
PYTHONDONTWRITEBYTECODE=1 "$V2/../.venv/bin/python" "$V2/scripts/netlist_collect.py" || fail=1
exit $fail
