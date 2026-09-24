//==============================================================================
// cnn_top.sv — KV260 board-facing top level
//
// Thin wrapper that adapts the AXI-conventional active-low reset (s_axi_aresetn,
// as emitted by proc_sys_reset / the PS) to the active-high synchronous reset
// used throughout the accelerator (Decision 5). All accelerator/control logic
// lives in rtl/common/cnn_axi_ctrl.sv; this module only converts reset polarity
// and re-exposes the AXI4-Lite slave + interrupt.
//
// The PS side (ARM Cortex-A53) drives this slave over M_AXI_HPM0_FPD through an
// AXI SmartConnect; the core and AXI share s_axi_aclk (one clock domain — the
// block design's Clocking Wizard derives it from PS pl_clk0). See
// docs/PHASE3_BOARD_INTEGRATION.md.
//==============================================================================

module cnn_top #(
    parameter int AXI_ADDR_W = 32,
    parameter int AXI_DATA_W = 32
) (
    input  logic                      s_axi_aclk,
    input  logic                      s_axi_aresetn,

    // ---- AXI4-Lite slave ------------------------------------------------
    input  logic [AXI_ADDR_W-1:0]     s_axi_awaddr,
    input  logic                      s_axi_awvalid,
    output logic                      s_axi_awready,
    input  logic [AXI_DATA_W-1:0]     s_axi_wdata,
    input  logic [AXI_DATA_W/8-1:0]   s_axi_wstrb,
    input  logic                      s_axi_wvalid,
    output logic                      s_axi_wready,
    output logic [1:0]                s_axi_bresp,
    output logic                      s_axi_bvalid,
    input  logic                      s_axi_bready,
    input  logic [AXI_ADDR_W-1:0]     s_axi_araddr,
    input  logic                      s_axi_arvalid,
    output logic                      s_axi_arready,
    output logic [AXI_DATA_W-1:0]     s_axi_rdata,
    output logic [1:0]                s_axi_rresp,
    output logic                      s_axi_rvalid,
    input  logic                      s_axi_rready,

    // ---- Interrupt ------------------------------------------------------
    output logic                      irq
);

    logic rst;
    assign rst = ~s_axi_aresetn;

    cnn_axi_ctrl #(
        .AXI_ADDR_W (AXI_ADDR_W),
        .AXI_DATA_W (AXI_DATA_W)
    ) u_ctrl (
        .clk               (s_axi_aclk),
        .rst               (rst),
        .s_axi_awaddr      (s_axi_awaddr),
        .s_axi_awvalid     (s_axi_awvalid),
        .s_axi_awready     (s_axi_awready),
        .s_axi_wdata       (s_axi_wdata),
        .s_axi_wstrb       (s_axi_wstrb),
        .s_axi_wvalid      (s_axi_wvalid),
        .s_axi_wready      (s_axi_wready),
        .s_axi_bresp       (s_axi_bresp),
        .s_axi_bvalid      (s_axi_bvalid),
        .s_axi_bready      (s_axi_bready),
        .s_axi_araddr      (s_axi_araddr),
        .s_axi_arvalid     (s_axi_arvalid),
        .s_axi_arready     (s_axi_arready),
        .s_axi_rdata       (s_axi_rdata),
        .s_axi_rresp       (s_axi_rresp),
        .s_axi_rvalid      (s_axi_rvalid),
        .s_axi_rready      (s_axi_rready),
        .irq               (irq)
    );

endmodule
