`timescale 1ns/1ps
// gos_top_wrapper — V2 PL top for the KV260 block design (plain Verilog, BD module reference).
// Same port list as the empty shell (gos_top_ports.vh, FORMATS.md "PL top ports and address map"),
// so v2/vivado/bd_shell.tcl builds the identical block design with top=gos_top_wrapper.
// Contents: gos_top (gos_csr AXI4-Lite + gos_core). Adaptation of the AXI BRAM Controller ports
// (64-bit, single port, READ_LATENCY 1) to the core PS memory ports (PS_RD_LAT = 1):
//   ps_addr = byte addr >> 3, ps_en = en, ps_we = |we, ps_be = we, ps_wdata = din, dout = ps_rdata.
// Everything runs on clk (pl_clk0); the BRAM-side clk/rst pins are unused. Internal synchronous
// active-high rst = registered !rstn. BUILD_ID = 8-hex-digit short git commit (set by bd_shell.tcl).
// VERSION (CSR 0x0F8) = 32'h474F_5302 ("GOS", 2) — the empty shell reads 32'h474F_5300.
`define GOS_TOP_PORTS
module gos_top_wrapper #(
  parameter [31:0] BUILD_ID = 32'h0000_0000
) (
`include "gos_top_ports.vh"
);
`undef GOS_TOP_PORTS

  reg rst;
  always @(posedge clk) rst <= ~rstn;

  gos_top #(
    .BUILD_ID  (BUILD_ID),
    .PS_RD_LAT (1)
  ) u_top (
    .clk           (clk),
    .rst           (rst),
    .s_axi_awaddr  (s_axi_csr_awaddr),
    .s_axi_awprot  (3'b000),
    .s_axi_awvalid (s_axi_csr_awvalid),
    .s_axi_awready (s_axi_csr_awready),
    .s_axi_wdata   (s_axi_csr_wdata),
    .s_axi_wstrb   (s_axi_csr_wstrb),
    .s_axi_wvalid  (s_axi_csr_wvalid),
    .s_axi_wready  (s_axi_csr_wready),
    .s_axi_bresp   (s_axi_csr_bresp),
    .s_axi_bvalid  (s_axi_csr_bvalid),
    .s_axi_bready  (s_axi_csr_bready),
    .s_axi_araddr  (s_axi_csr_araddr),
    .s_axi_arprot  (3'b000),
    .s_axi_arvalid (s_axi_csr_arvalid),
    .s_axi_arready (s_axi_csr_arready),
    .s_axi_rdata   (s_axi_csr_rdata),
    .s_axi_rresp   (s_axi_csr_rresp),
    .s_axi_rvalid  (s_axi_csr_rvalid),
    .s_axi_rready  (s_axi_csr_rready),
    .act0_en       (bram_act0_en),
    .act0_we       (|bram_act0_we),
    .act0_be       (bram_act0_we),
    .act0_addr     (bram_act0_addr[14:3]),
    .act0_wdata    (bram_act0_din),
    .act0_rdata    (bram_act0_dout),
    .act1_en       (bram_act1_en),
    .act1_we       (|bram_act1_we),
    .act1_be       (bram_act1_we),
    .act1_addr     (bram_act1_addr[14:3]),
    .act1_wdata    (bram_act1_din),
    .act1_rdata    (bram_act1_dout),
    .wgt_en        (bram_wgt_en),
    .wgt_we        (|bram_wgt_we),
    .wgt_be        (bram_wgt_we),
    .wgt_addr      (bram_wgt_addr[16:3]),
    .wgt_wdata     (bram_wgt_din),
    .wgt_rdata     (bram_wgt_dout),
    .qp_en         (bram_qparam_en),
    .qp_we         (|bram_qparam_we),
    .qp_be         (bram_qparam_we),
    .qp_addr       (bram_qparam_addr[11:3]),
    .qp_wdata      (bram_qparam_din),
    .qp_rdata      (bram_qparam_dout)
  );
endmodule
