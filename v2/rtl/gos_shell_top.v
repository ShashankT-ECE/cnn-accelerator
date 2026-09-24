`timescale 1ns/1ps
// gos_shell_top — V2 Step 4 EMPTY SHELL PL top (plain Verilog, BD module reference).
// Same port list as the final gos_top_wrapper (gos_top_ports.vh). Contents:
//   * the real memories gos_act_buf x2 (ACT0, ACT1), gos_wgt_mem (WGT), gos_qparam_mem (QPARAM),
//     PS_RD_LAT = 1, PS port A driven by the AXI BRAM Controllers:
//       ps_addr = byte addr >> 3, ps_en = en, ps_we = |we, ps_be = we, ps_wdata = din,
//       dout = ps_rdata (valid 1 cycle after en = controller READ_LATENCY 1);
//     accel_busy = 0 and all accelerator ports tied off (port B unused);
//   * gos_shell_scratch: AXI4-Lite VERSION (0x0F8 = 32'h474F_5300), BUILD_ID (0x0FC), SCRATCH0..3.
// Everything runs on clk; the BRAM-side clk/rst pins are unused (the controllers are clocked by the
// same pl_clk0). Internal sync active-high rst = registered !rstn.
// Sources: gos_pkg.sv gos_act_buf.sv gos_wgt_mem.sv gos_qparam_mem.sv gos_shell_scratch.v
//          gos_top_ports.vh (include) gos_shell_top.v
`define GOS_TOP_PORTS
module gos_shell_top #(
  parameter [31:0] BUILD_ID = 32'h0000_0000
) (
`include "gos_top_ports.vh"
);
`undef GOS_TOP_PORTS

  localparam [31:0] SHELL_VERSION = 32'h474F_5300;   // "GOS" + 0x00 = empty shell
  localparam integer TW = 16;

  // ------------------------------------------------------------------ reset
  reg rst;
  always @(posedge clk) rst <= ~rstn;

  // ------------------------------------------------------------------ AXI-Lite scratch block
  gos_shell_scratch #(
    .VERSION  (SHELL_VERSION),
    .BUILD_ID (BUILD_ID)
  ) u_scratch (
    .clk     (clk),
    .rst     (rst),
    .awaddr  (s_axi_csr_awaddr),
    .awvalid (s_axi_csr_awvalid),
    .awready (s_axi_csr_awready),
    .wdata   (s_axi_csr_wdata),
    .wstrb   (s_axi_csr_wstrb),
    .wvalid  (s_axi_csr_wvalid),
    .wready  (s_axi_csr_wready),
    .bresp   (s_axi_csr_bresp),
    .bvalid  (s_axi_csr_bvalid),
    .bready  (s_axi_csr_bready),
    .araddr  (s_axi_csr_araddr),
    .arvalid (s_axi_csr_arvalid),
    .arready (s_axi_csr_arready),
    .rdata   (s_axi_csr_rdata),
    .rresp   (s_axi_csr_rresp),
    .rvalid  (s_axi_csr_rvalid),
    .rready  (s_axi_csr_rready)
  );

  // ------------------------------------------------------------------ ACT0 / ACT1
  wire act0_busy_violation, act1_busy_violation;       // unused in the shell
  wire act0_out_valid, act1_out_valid;
  wire [TW-1:0] act0_out_tok, act1_out_tok;
  wire [63:0]   act0_rd_data, act1_rd_data;

  gos_act_buf #(.TW(TW), .PS_RD_LAT(1)) u_act0 (
    .clk               (clk),
    .rst               (rst),
    .accel_busy        (1'b0),
    .ps_en             (bram_act0_en),
    .ps_we             (|bram_act0_we),
    .ps_be             (bram_act0_we),
    .ps_addr           (bram_act0_addr[14:3]),
    .ps_wdata          (bram_act0_din),
    .ps_rdata          (bram_act0_dout),
    .ps_busy_violation (act0_busy_violation),
    .acc_wr_en         (1'b0),
    .acc_wr_addr       (12'd0),
    .acc_wr_data       (64'd0),
    .acc_wr_be         (8'd0),
    .in_valid          (1'b0),
    .in_tok            ({TW{1'b0}}),
    .rd_addr           (96'd0),
    .out_valid         (act0_out_valid),
    .out_tok           (act0_out_tok),
    .rd_data           (act0_rd_data)
  );

  gos_act_buf #(.TW(TW), .PS_RD_LAT(1)) u_act1 (
    .clk               (clk),
    .rst               (rst),
    .accel_busy        (1'b0),
    .ps_en             (bram_act1_en),
    .ps_we             (|bram_act1_we),
    .ps_be             (bram_act1_we),
    .ps_addr           (bram_act1_addr[14:3]),
    .ps_wdata          (bram_act1_din),
    .ps_rdata          (bram_act1_dout),
    .ps_busy_violation (act1_busy_violation),
    .acc_wr_en         (1'b0),
    .acc_wr_addr       (12'd0),
    .acc_wr_data       (64'd0),
    .acc_wr_be         (8'd0),
    .in_valid          (1'b0),
    .in_tok            ({TW{1'b0}}),
    .rd_addr           (96'd0),
    .out_valid         (act1_out_valid),
    .out_tok           (act1_out_tok),
    .rd_data           (act1_rd_data)
  );

  // ------------------------------------------------------------------ WGT
  wire          wgt_out_valid;
  wire [TW-1:0] wgt_out_tok;
  wire [63:0]   wgt_rd_data;

  gos_wgt_mem #(.TW(TW), .PS_RD_LAT(1)) u_wgt (
    .clk       (clk),
    .rst       (rst),
    .ps_en     (bram_wgt_en),
    .ps_we     (|bram_wgt_we),
    .ps_be     (bram_wgt_we),
    .ps_addr   (bram_wgt_addr[16:3]),
    .ps_wdata  (bram_wgt_din),
    .ps_rdata  (bram_wgt_dout),
    .in_valid  (1'b0),
    .in_tok    ({TW{1'b0}}),
    .rd_addr   (14'd0),
    .out_valid (wgt_out_valid),
    .out_tok   (wgt_out_tok),
    .rd_data   (wgt_rd_data)
  );

  // ------------------------------------------------------------------ QPARAM
  wire          qp_out_valid;
  wire [TW-1:0] qp_out_tok;
  wire [31:0]   qp_q_bias, qp_m;
  wire [5:0]    qp_s;

  gos_qparam_mem #(.TW(TW), .PS_RD_LAT(1)) u_qparam (
    .clk       (clk),
    .rst       (rst),
    .ps_en     (bram_qparam_en),
    .ps_we     (|bram_qparam_we),
    .ps_be     (bram_qparam_we),
    .ps_addr   (bram_qparam_addr[11:3]),
    .ps_wdata  (bram_qparam_din),
    .ps_rdata  (bram_qparam_dout),
    .in_valid  (1'b0),
    .in_tok    ({TW{1'b0}}),
    .rd_ch     (8'd0),
    .out_valid (qp_out_valid),
    .out_tok   (qp_out_tok),
    .q_bias    (qp_q_bias),
    .m         (qp_m),
    .s         (qp_s)
  );

endmodule
