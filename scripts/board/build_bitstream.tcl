#==============================================================================
# build_bitstream.tcl — KV260 block design + bitstream generation (BOARD-ONLY).
#
# This script documents the exact SoC integration. It requires the KV260 board
# files (Xilinx Board Store "k26c" / "kv260") and the Zynq UltraScale+ PS IP,
# which are NOT present in this development environment. It is prepared now so
# the only remaining board work is: install board files -> source this script
# -> generate the .xsa -> build the platform/app -> load the bitstream.
#
# Architecture (matches docs/PHASE3_BOARD_INTEGRATION.md):
#   PS (A53) M_AXI_HPM0_FPD @0xA400_0000 -> AXI SmartConnect -> cnn_top AXI slave
#   Clocking Wizard: PS pl_clk0 (100 MHz) -> 200 MHz core clock (s_axi_aclk)
#   proc_sys_reset x2: peripheral_aresetn -> cnn_top s_axi_aresetn
#
# Before sourcing, add the RTL as a packaged IP (or use "add_files" +
# "import_files" with cnn_top.sv as the top), then run this script inside a
# Vivado project targeting the KV260 board.
#==============================================================================

set part xck26-sfvc784-2LV-c
set board kv260-som

# --- 1. Processing System (Zynq UltraScale+ MPSoC) --------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 ps8_0
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset 1} [get_bd_cells ps8_0]

# --- 2. cnn_top (packaged RTL IP) ------------------------------------------
create_bd_cell -type module -reference cnn_top cnn_accel_0

# --- 3. Clocking: pl_clk0 (100 MHz) -> 200 MHz ------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz:6.0 clk_wiz_0
set_property -dict [list \
    CONFIG.PRIM_IN_FREQ {100.000} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {200.000} \
    CONFIG.RESET_TYPE {ACTIVE_LOW} \
] [get_bd_cells clk_wiz_0]
connect_bd_net [get_bd_pins ps8_0/pl_clk0] [get_bd_pins clk_wiz_0/clk_in1]

# --- 4. Reset (proc_sys_reset) ----------------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_axi
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_core
connect_bd_net [get_bd_pins clk_wiz_0/locked] [get_bd_pins rst_core/dcm_locked]
connect_bd_net [get_bd_pins ps8_0/pl_resetn0] [get_bd_pins rst_axi/ext_reset_in]
connect_bd_net [get_bd_pins ps8_0/pl_resetn0] [get_bd_pins rst_core/ext_reset_in]

# --- 5. AXI SmartConnect + connections --------------------------------------
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells smartconnect_0]
connect_bd_intf_net [get_bd_intf_pins ps8_0/M_AXI_HPM0_FPD] \
                    [get_bd_intf_pins smartconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_0/M00_AXI] \
                    [get_bd_intf_pins cnn_accel_0/s_axi]
connect_bd_net [get_bd_pins rst_axi/peripheral_aresetn] \
               [get_bd_pins cnn_accel_0/s_axi_aresetn]
connect_bd_net [get_bd_pins rst_core/peripheral_aresetn] \
               [get_bd_pins cnn_accel_0/s_axi_aresetn]
connect_bd_net [get_bd_pins clk_wiz_0/clk_out1] [get_bd_pins cnn_accel_0/s_axi_aclk]
connect_bd_net [get_bd_pins cnn_accel_0/irq] [get_bd_pins ps8_0/pl_ps_irq0]

# --- 6. Address + validation ------------------------------------------------
assign_bd_address
set_property offset 0xA4000000 [get_bd_addr_segs {ps8_0/Data/SEG_cnn_accel_0_reg}]
validate_bd_design
save_bd_design

# --- 7. Generate bitstream + XSA --------------------------------------------
generate_target all [get_files *.bd]
synth_design -top [get_bd_design].bd_wrapper
write_bitstream -force cnn_top.bit
write_hw_platform -force -include_bit cnn_top.xsa
