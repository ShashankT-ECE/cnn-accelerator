# bd_shell.tcl — KV260 block design + full build (synth, impl, bitstream, .hwh) for a V2 PL top.
#
#   vivado -mode batch -source v2/vivado/bd_shell.tcl -tclargs \
#       top=<module> build_id=<8 hex> outdir=<dir> filelist=<file> [name=<basename>] [jobs=<n>] [bd_only=1]
#
#   top       PL top module (plain Verilog, port list = v2/rtl/gos_top_ports.vh, parameter
#             [31:0] BUILD_ID). Default gos_shell_top; Step 6 passes gos_top_wrapper.
#   build_id  8 hex digits (git rev-parse --short=8 HEAD); set as the top's BUILD_ID parameter.
#   outdir    output directory (project in <outdir>/proj, deliverables in <outdir>).
#   filelist  text file, one RTL source per line (absolute or relative to the list file; '#'
#             comments). .sv -> SystemVerilog, .v -> Verilog, .vh/.svh -> Verilog Header (its
#             directory is added to the include path). Compile order: as listed.
#   name      basename of the deliverables (default gos_shell): <name>.bit, <name>.hwh.
#   bd_only=1 stop after BD validation + wrapper generation (quick check).
#
# The block design is design-independent: everything design-specific comes from the top module
# and its sources. BD cell names are fixed (the top instance is always "gos_top_0").
#
# Outputs in <outdir>: <name>.bit, <name>.hwh, <name>.bit.sha256, utilization.rpt,
# utilization_hier.rpt, timing_summary.rpt, address_map.txt, critical_warnings.txt, summary.json.

set_param general.maxThreads 8   ;# user rule after the OOM freeze: one Vivado job, <= 8 threads

# ------------------------------------------------------------------ arguments
array set A {top gos_shell_top build_id "" outdir "" filelist "" name gos_shell jobs 8 bd_only 0
             pl_mhz 200 freq_tol_pct 1.0 strategy default}
foreach a $argv {
    set kv [split $a =]
    if {[llength $kv] < 2} { error "bd_shell.tcl: bad argument '$a' (expected key=value)" }
    set k [lindex $kv 0]
    if {![info exists A($k)]} { error "bd_shell.tcl: unknown argument '$k'" }
    set A($k) [join [lrange $kv 1 end] =]
}
set A(jobs) 1   ;# forced after argv: one run at a time (synth, IP OOC and impl runs are serialized)
foreach k {build_id outdir filelist} {
    if {$A($k) eq ""} { error "bd_shell.tcl: missing $k=..." }
}
if {![regexp {^[0-9a-fA-F]{8}$} $A(build_id)]} { error "bd_shell.tcl: build_id must be 8 hex digits" }
set top     $A(top)
set outdir  [file normalize $A(outdir)]
set name    $A(name)
set part    xck26-sfvc784-2LV-c
set bdname  gos_system
set PL0_MHZ_REQ $A(pl_mhz)
if {[lsearch -exact {default explore} $A(strategy)] < 0} { error "bd_shell.tcl: strategy must be default|explore" }
file mkdir $outdir
set projdir [file join $outdir proj]
file delete -force $projdir

# ------------------------------------------------------------------ project + board
set bps [get_board_parts -quiet -latest_file_version *kv260_som*]
if {[llength $bps] == 0} {
    puts "bd_shell: KV260 board files not found, installing from the Xilinx board store"
    xhub::refresh_catalog [xhub::get_xstores xilinx_board_store]
    xhub::install [xhub::get_xitems *kv260*]
    set bps [get_board_parts -quiet -latest_file_version *kv260_som*]
}
if {[llength $bps] == 0} { error "bd_shell: no kv260_som board part available" }
set board_part [lindex [lsort $bps] end]

create_project -force proj $projdir -part $part
set_property board_part $board_part [current_project]
# Carrier connection (KV260 carrier 1.3 on SOM connector 1). No PL pins are used by this design,
# so this only records the board pairing; a failure is reported but not fatal.
if {[catch {set_property board_connections \
        {som240_1_connector xilinx.com:kv260_carrier:som240_1_connector:1.3} [current_project]} e]} {
    puts "bd_shell: board_connections NOT set: $e"
} else {
    puts "bd_shell: board_part $board_part, board_connections [get_property board_connections [current_project]]"
}
set_property target_language Verilog [current_project]

# ------------------------------------------------------------------ sources
set lf [file normalize $A(filelist)]
set fh [open $lf r]; set lines [split [read $fh] "\n"]; close $fh
set incdirs {}
set srcs {}
foreach l $lines {
    set l [string trim $l]
    if {$l eq "" || [string index $l 0] eq "#"} continue
    if {[file pathtype $l] ne "absolute"} { set l [file normalize [file join [file dirname $lf] $l]] }
    if {![file exists $l]} { error "bd_shell: source not found: $l" }
    lappend srcs $l
}
add_files -norecurse -fileset sources_1 $srcs
foreach f $srcs {
    set fo [get_files $f]
    switch -glob -- $f {
        *.sv            { set_property file_type SystemVerilog $fo }
        *.v             { set_property file_type Verilog $fo }
        *.vh - *.svh    { set_property file_type {Verilog Header} $fo
                          lappend incdirs [file dirname $f] }
    }
}
if {[llength $incdirs]} { set_property include_dirs [lsort -unique $incdirs] [get_filesets sources_1] }
update_compile_order -fileset sources_1

# ------------------------------------------------------------------ block design
create_bd_design $bdname

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
# Only M_AXI_HPM0_FPD (the KV260 preset also enables HPM1_FPD and pl_clk1: both disabled);
# pl_clk0 requested at 200 MHz. With the carrier connection the preset sources PL0 from IOPLL
# (1500 MHz -> /8 = 187.5 MHz), so the source PLL (IOPLL or RPLL; DPLL is tied to DDR) is
# chosen here as the one whose ACTUAL frequency is closest to the request.
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} \
    CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} \
    CONFIG.PSU__FPGA_PL1_ENABLE {0} \
] $ps
set best_src ""; set best_err 1e9
foreach src {IOPLL RPLL} {
    set_property -dict [list CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL $src \
        CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $PL0_MHZ_REQ] $ps
    set act [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]
    puts "bd_shell: pl_clk0 source $src -> actual $act MHz"
    set err [expr {abs($act - $PL0_MHZ_REQ)}]
    # ties go to RPLL (the source of the 200 MHz baseline), so all variants share one PL0 source
    if {$err <= $best_err + 1e-6} { set best_err $err; set best_src $src }
}
set_property -dict [list CONFIG.PSU__CRL_APB__PL0_REF_CTRL__SRCSEL $best_src \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $PL0_MHZ_REQ] $ps
set pl0_act [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__ACT_FREQMHZ $ps]
set pl0_div [list [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__DIVISOR0 $ps] \
                  [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__DIVISOR1 $ps]]
puts "bd_shell: pl_clk0 requested $PL0_MHZ_REQ MHz, source $best_src, divisors $pl0_div, actual $pl0_act MHz"
if {abs($pl0_act - $PL0_MHZ_REQ) > 0.01 * $A(freq_tol_pct) * $PL0_MHZ_REQ} {
    error "bd_shell: pl_clk0 actual $pl0_act MHz is more than $A(freq_tol_pct)% off the requested $PL0_MHZ_REQ MHz"
}

set rstc [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_pl0]
set sc   [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect axi_sc]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {5} CONFIG.NUM_CLKS {1}] $sc

# name  window_bytes  offset  top BRAM interface
set mems {
    act0_ctrl   32K  0xA0020000 BRAM_ACT0
    act1_ctrl   32K  0xA0040000 BRAM_ACT1
    wgt_ctrl    128K 0xA0080000 BRAM_WGT
    qparam_ctrl 4K   0xA0010000 BRAM_QPARAM
}
foreach {cn rng off bif} $mems {
    set c [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl $cn]
    set_property -dict [list CONFIG.DATA_WIDTH {64} CONFIG.SINGLE_PORT_BRAM {1} \
        CONFIG.PROTOCOL {AXI4} CONFIG.ECC_TYPE {0} CONFIG.READ_LATENCY {1}] $c
}

set t [create_bd_cell -type module -reference $top gos_top_0]
set_property CONFIG.BUILD_ID "0x[string toupper $A(build_id)]" $t

# clocks / resets
set clk [get_bd_pins ps/pl_clk0]
connect_bd_net $clk [get_bd_pins ps/maxihpm0_fpd_aclk] [get_bd_pins rst_pl0/slowest_sync_clk] \
    [get_bd_pins axi_sc/aclk] [get_bd_pins gos_top_0/clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst_pl0/ext_reset_in]
connect_bd_net [get_bd_pins rst_pl0/peripheral_aresetn] [get_bd_pins axi_sc/aresetn] [get_bd_pins gos_top_0/rstn]

# AXI
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins axi_sc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_sc/M00_AXI] [get_bd_intf_pins gos_top_0/S_AXI_CSR]
set mi 1
foreach {cn rng off bif} $mems {
    connect_bd_intf_net [get_bd_intf_pins axi_sc/M0${mi}_AXI] [get_bd_intf_pins $cn/S_AXI]
    connect_bd_net $clk [get_bd_pins $cn/s_axi_aclk]
    connect_bd_net [get_bd_pins rst_pl0/peripheral_aresetn] [get_bd_pins $cn/s_axi_aresetn]
    connect_bd_intf_net [get_bd_intf_pins $cn/BRAM_PORTA] [get_bd_intf_pins gos_top_0/$bif]
    incr mi
}

# address map
assign_bd_address -offset 0xA0000000 -range 4K -target_address_space [get_bd_addr_spaces ps/Data] \
    [get_bd_addr_segs gos_top_0/S_AXI_CSR/*]
foreach {cn rng off bif} $mems {
    assign_bd_address -offset $off -range $rng -target_address_space [get_bd_addr_spaces ps/Data] \
        [get_bd_addr_segs $cn/S_AXI/*]
}

validate_bd_design
save_bd_design

# address map + interface widths for the record
set fh [open [file join $outdir address_map.txt] w]
foreach s [lsort [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data]]] {
    puts $fh [format "%-40s offset %s range %s" $s [get_property OFFSET $s] [get_property RANGE $s]]
}
foreach {cn rng off bif} $mems {
    puts $fh [format "%-12s DATA_WIDTH %s READ_LATENCY %s MEM_DEPTH %s BRAM_ADDR_WIDTH(pin) %s" $cn \
        [get_property CONFIG.DATA_WIDTH [get_bd_cells $cn]] \
        [get_property CONFIG.READ_LATENCY [get_bd_cells $cn]] \
        [get_property CONFIG.MEM_DEPTH [get_bd_cells $cn]] \
        [expr {[get_property LEFT [get_bd_pins $cn/bram_addr_a]] + 1}]]
}
puts $fh "pl_clk0 requested $PL0_MHZ_REQ MHz source $best_src divisors $pl0_div actual $pl0_act MHz"
puts $fh "HPM0_FPD data width [get_property CONFIG.PSU__MAXIGP0__DATA_WIDTH $ps]"
puts $fh "top $top BUILD_ID [get_property CONFIG.BUILD_ID $t]"
close $fh

make_wrapper -files [get_files $bdname.bd] -top
set wrap [glob [file join $projdir proj.gen sources_1 bd $bdname hdl ${bdname}_wrapper.v]]
add_files -norecurse $wrap
set_property top ${bdname}_wrapper [current_fileset]
update_compile_order -fileset sources_1
generate_target all [get_files $bdname.bd]

if {$A(bd_only)} { puts "bd_shell: bd_only done"; exit 0 }

# ------------------------------------------------------------------ build
if {$A(strategy) eq "explore"} {
    set_property strategy Performance_Explore [get_runs impl_1]
    set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
    set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
}
puts "bd_shell: impl_1 strategy [get_property STRATEGY [get_runs impl_1]]"
launch_runs impl_1 -to_step write_bitstream -jobs $A(jobs)
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "bd_shell: impl_1 did not complete: [get_property STATUS [get_runs impl_1]]"
}
open_run impl_1

report_utilization -file [file join $outdir utilization.rpt]
report_utilization -hierarchical -file [file join $outdir utilization_hier.rpt]
report_timing_summary -max_paths 10 -file [file join $outdir timing_summary.rpt]
report_timing -max_paths 1 -nworst 1 -delay_type max -file [file join $outdir worst_path.rpt]
report_power -file [file join $outdir power.rpt]
# static checks (Step 4.5)
report_methodology -file [file join $outdir methodology.rpt]
report_drc -file [file join $outdir drc.rpt]
check_timing -verbose -file [file join $outdir check_timing.rpt]
set fh [open [file join $outdir static_counts.txt] w]
foreach sev {{CRITICAL WARNING} WARNING ERROR {ADVISORY}} {
    puts $fh "methodology [string map {{ } _} $sev] [llength [get_methodology_violations -quiet -filter "SEVERITY == \"$sev\""]]"
    puts $fh "drc [string map {{ } _} $sev] [llength [get_drc_violations -quiet -filter "SEVERITY == \"$sev\""]]"
}
close $fh
set wp [lindex [get_timing_paths -quiet -delay_type max -max_paths 1] 0]
set wp_desc "none"
if {$wp ne ""} {
    set wp_desc "[get_property STARTPOINT_PIN $wp] -> [get_property ENDPOINT_PIN $wp] levels [get_property LOGIC_LEVELS $wp] datapath [get_property DATAPATH_DELAY $wp]"
}
set wp_desc [string map [list "\"" "'" "\\" "/"] $wp_desc]

proc slack_sum {type} {
    set wns "null"; set tot 0.0; set n 0
    set p [get_timing_paths -quiet -delay_type $type -max_paths 1]
    if {[llength $p] && [get_property SLACK $p] ne ""} { set wns [get_property SLACK $p] }
    foreach q [get_timing_paths -quiet -delay_type $type -max_paths 100000 -slack_lesser_than 0] {
        set tot [expr {$tot + [get_property SLACK $q]}]; incr n
    }
    return [list $wns $tot $n]
}
lassign [slack_sum max] wns tns nfs
lassign [slack_sum min] whs ths nfh

# deliverables
set rundir [get_property DIRECTORY [get_runs impl_1]]
set bit [glob [file join $rundir ${bdname}_wrapper.bit]]
set hwh [glob [file join $projdir proj.gen sources_1 bd $bdname hw_handoff ${bdname}.hwh]]
file copy -force $bit [file join $outdir $name.bit]
file copy -force $hwh [file join $outdir $name.hwh]

# critical warnings from the run logs
set cw {}
foreach lg [list [file join [get_property DIRECTORY [get_runs synth_1]] runme.log] [file join $rundir runme.log]] {
    if {[file exists $lg]} {
        set fh [open $lg r]
        foreach l [split [read $fh] "\n"] { if {[string match "CRITICAL WARNING:*" $l]} { lappend cw $l } }
        close $fh
    }
}
set fh [open [file join $outdir critical_warnings.txt] w]
foreach l $cw { puts $fh $l }
close $fh

# clock period actually constrained on pl_clk0
set clkobj [lindex [get_clocks -quiet -of_objects [get_pins -quiet -hierarchical -filter {NAME =~ */ps/*PLCLK[0]}]] 0]
set per "null"
if {$clkobj ne ""} { set per [get_property PERIOD $clkobj] }

set fh [open [file join $outdir summary.json] w]
puts $fh "{\"top\": \"$top\", \"build_id\": \"$A(build_id)\", \"board_part\": \"$board_part\", \"part\": \"$part\","
puts $fh " \"pl_clk0_mhz_requested\": $PL0_MHZ_REQ, \"pl_clk0_mhz_actual\": $pl0_act, \"pl_clk0_period_ns_constrained\": $per,"
puts $fh " \"pl_clk0_srcsel\": \"$best_src\", \"pl_clk0_divisors\": \"$pl0_div\","
puts $fh " \"wns_ns\": $wns, \"tns_ns\": $tns, \"failing_setup_endpoints\": $nfs,"
puts $fh " \"whs_ns\": $whs, \"ths_ns\": $ths, \"failing_hold_endpoints\": $nfh,"
puts $fh " \"critical_warnings\": [llength $cw], \"bit\": \"$name.bit\", \"hwh\": \"$name.hwh\","
puts $fh " \"strategy\": \"[get_property STRATEGY [get_runs impl_1]]\", \"worst_path\": \"$wp_desc\","
puts $fh " \"vivado_version\": \"[version -short]\"}"
close $fh
puts "BD_SHELL_SUMMARY top=$top pl_clk0=$pl0_act MHz wns=$wns whs=$whs tns=$tns ths=$ths cw=[llength $cw]"
