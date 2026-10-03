# device_query.tcl — dump the XCK26 part properties (resource totals) for the V3 DSE (no design, no synthesis).
#   vivado -mode batch -source v3/feas/device_query.tcl -tclargs <outdir>
set_param general.maxThreads 8
set outdir [lindex $argv 0]
file mkdir $outdir
set part [get_parts xck26-sfvc784-2LV-c]
report_property -all $part -file [file join $outdir part_properties.txt]
set fh [open [file join $outdir vivado_version.txt] w]
puts $fh [version -short]
close $fh
