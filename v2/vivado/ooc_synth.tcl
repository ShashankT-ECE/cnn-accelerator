# ooc_synth.tcl — out-of-context synthesis of one V2 leaf module.
#   vivado -mode batch -source v2/vivado/ooc_synth.tcl -tclargs <top> <outdir> <rtl files...> [-- generic=value ...]
# Part xck26-sfvc784-2LV-c, 5.000 ns clock on port clk; all other ports get 0 ns
# input/output delay relative to clk (full-cycle budget). Writes to <outdir>:
#   utilization.rpt, utilization_hier.rpt, timing_summary.rpt, summary.json
# summary.json: top, part, clk_period_ns, lut, lutram, ff, carry8, dsp, ramb36, ramb18, wns_ns, tns_ns,
#               failing_endpoints, vivado_version
set top    [lindex $argv 0]
set outdir [lindex $argv 1]
set files  {}
set generics {}
set in_generics 0
foreach a [lrange $argv 2 end] {
    if {$a eq "--"} { set in_generics 1; continue }
    if {$in_generics} { lappend generics $a } else { lappend files $a }
}
set part xck26-sfvc784-2LV-c
file mkdir $outdir

foreach f $files { read_verilog -sv $f }
set xdc [file join $outdir ooc_clk.xdc]
set fh [open $xdc w]
puts $fh "create_clock -name clk -period 5.000 \[get_ports clk\]"
# Time every port against clk with 0 ns external delay, so port-to-register and
# register-to-port (and port-to-port) paths of leaf modules are budgeted a full
# cycle (in integration they are driven by / drive registers). Without this a
# module with no reg-to-reg path (e.g. gos_rotator) would report no WNS at all.
puts $fh "set_input_delay -clock clk 0.000 \[get_ports -filter {DIRECTION == IN && NAME != clk}\]"
puts $fh "set_output_delay -clock clk 0.000 \[get_ports -filter {DIRECTION == OUT}\]"
close $fh
read_xdc -mode out_of_context $xdc

set gopts {}
foreach g $generics { lappend gopts -generic $g }
synth_design -top $top -part $part -mode out_of_context {*}$gopts

report_utilization -file [file join $outdir utilization.rpt]
report_utilization -hierarchical -file [file join $outdir utilization_hier.rpt]
report_timing_summary -max_paths 10 -file [file join $outdir timing_summary.rpt]

proc cnt {filter} { return [llength [get_cells -quiet -hierarchical -filter $filter]] }
# Primitive counts by REF_NAME (a DSP48E2 macro counts once; its DSP_* sub-cells are not matched).
set lut    [cnt {REF_NAME =~ LUT*}]
set lutram [cnt {REF_NAME =~ RAM32* || REF_NAME =~ RAM64* || REF_NAME =~ RAM128* || REF_NAME =~ RAM256* || REF_NAME =~ RAMD* || REF_NAME =~ RAMS* || REF_NAME =~ SRL*}]
set ff     [cnt {REF_NAME =~ FD*}]
set carry  [cnt {REF_NAME == CARRY8}]
set dsp    [cnt {REF_NAME == DSP48E2}]
set r36    [cnt {REF_NAME =~ RAMB36*}]
set r18    [cnt {REF_NAME =~ RAMB18*}]
set paths  [get_timing_paths -quiet -delay_type max -max_paths 1]
set wns "null"
if {[llength $paths]} {
    set sl [get_property SLACK [lindex $paths 0]]
    if {$sl ne ""} { set wns $sl }
}
set tns 0.0; set nfail 0
foreach p [get_timing_paths -quiet -delay_type max -max_paths 100000 -slack_lesser_than 0] {
    set tns [expr {$tns + [get_property SLACK $p]}]; incr nfail
}
set ver [version -short]
set fh [open [file join $outdir summary.json] w]
puts $fh "{\"top\": \"$top\", \"part\": \"$part\", \"clk_period_ns\": 5.000, \"generics\": \"$generics\","
puts $fh " \"lut\": $lut, \"lutram\": $lutram, \"ff\": $ff, \"carry8\": $carry, \"dsp\": $dsp,"
puts $fh " \"ramb36\": $r36, \"ramb18\": $r18, \"wns_ns\": $wns, \"tns_ns\": $tns, \"failing_endpoints\": $nfail,"
puts $fh " \"vivado_version\": \"$ver\"}"
close $fh
puts "OOC_SUMMARY $top lut=$lut lutram=$lutram ff=$ff carry8=$carry dsp=$dsp ramb36=$r36 ramb18=$r18 wns=$wns"
