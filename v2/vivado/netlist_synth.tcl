# netlist_synth.tcl — synthesize gos_top (out of context, xck26-sfvc784-2LV-c, 5.000 ns) and write
# the post-synthesis functional-simulation netlist (write_verilog -mode funcsim) for gate-level sim.
#   vivado -mode batch -source netlist_synth.tcl -tclargs <outdir> <build_id_hex8> <rtl files...>
set_param general.maxThreads 8   ;# user rule after the OOM freeze: one Vivado job, <= 8 threads
set outdir [lindex $argv 0]
set bid    [lindex $argv 1]
file mkdir $outdir
foreach f [lrange $argv 2 end] { read_verilog -sv $f }
set fh [open [file join $outdir clk.xdc] w]
puts $fh "create_clock -name clk -period 5.000 \[get_ports clk\]"
close $fh
read_xdc -mode out_of_context [file join $outdir clk.xdc]
synth_design -top gos_top -part xck26-sfvc784-2LV-c -mode out_of_context \
    -generic BUILD_ID=32'h$bid -generic PS_RD_LAT=1
write_verilog -mode funcsim -force [file join $outdir gos_top_funcsim.v]
report_utilization -file [file join $outdir utilization.rpt]
puts "NETLIST_DONE [file join $outdir gos_top_funcsim.v]"
