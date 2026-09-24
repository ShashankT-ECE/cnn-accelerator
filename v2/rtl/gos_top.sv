`timescale 1ns/1ps
// GOS_OOC_DEPS: gos_csr.sv gos_core.sv gos_ctrl.sv gos_cfg_check.sv gos_act_buf.sv gos_wgt_mem.sv gos_qparam_mem.sv gos_rotator.sv gos_pe.sv gos_array.sv gos_requant.sv gos_pool.sv
// gos_top — gos_csr (AXI4-Lite, FORMATS.md "CSR register map") + gos_core.
// The four PS memory ports are passed through unchanged (word addresses, byte
// enables; PS_RD_LAT). The Step 6 plain-Verilog gos_top_wrapper adapts them to the
// AXI BRAM Controller ports declared once in gos_top_ports.vh and converts rstn.
module gos_top
  import gos_pkg::*;
#(
  parameter logic [31:0] BUILD_ID  = 32'h0000_0000,
  parameter int          PS_RD_LAT = 1
) (
  input  logic               clk,
  input  logic               rst,
  // AXI4-Lite slave (CSR)
  input  logic [11:0]        s_axi_awaddr,
  input  logic [2:0]         s_axi_awprot,
  input  logic               s_axi_awvalid,
  output logic               s_axi_awready,
  input  logic [31:0]        s_axi_wdata,
  input  logic [3:0]         s_axi_wstrb,
  input  logic               s_axi_wvalid,
  output logic               s_axi_wready,
  output logic [1:0]         s_axi_bresp,
  output logic               s_axi_bvalid,
  input  logic               s_axi_bready,
  input  logic [11:0]        s_axi_araddr,
  input  logic [2:0]         s_axi_arprot,
  input  logic               s_axi_arvalid,
  output logic               s_axi_arready,
  output logic [31:0]        s_axi_rdata,
  output logic [1:0]         s_axi_rresp,
  output logic               s_axi_rvalid,
  input  logic               s_axi_rready,
  // PS memory ports
  input  logic               act0_en, act0_we,
  input  logic [7:0]         act0_be,
  input  logic [ACT_AW-1:0]  act0_addr,
  input  logic [63:0]        act0_wdata,
  output logic [63:0]        act0_rdata,
  input  logic               act1_en, act1_we,
  input  logic [7:0]         act1_be,
  input  logic [ACT_AW-1:0]  act1_addr,
  input  logic [63:0]        act1_wdata,
  output logic [63:0]        act1_rdata,
  input  logic               wgt_en, wgt_we,
  input  logic [7:0]         wgt_be,
  input  logic [WGT_AW-1:0]  wgt_addr,
  input  logic [63:0]        wgt_wdata,
  output logic [63:0]        wgt_rdata,
  input  logic               qp_en, qp_we,
  input  logic [7:0]         qp_be,
  input  logic [QP_AW:0]     qp_addr,
  input  logic [63:0]        qp_wdata,
  output logic [63:0]        qp_rdata
);
  logic                   start, soft_reset, busy, done, error;
  logic [3:0]             n_layers;
  logic [7:0][15:0][31:0] desc;
  logic [31:0]            err_code;
  logic [63:0]            total_cyc, mac_active, stall;
  logic [7:0][31:0]       layer_cyc;
  logic [15:0][31:0]      logit;
  logic [7:0]             err_flags;

  gos_csr #(.BUILD_ID(BUILD_ID)) u_csr (
    .clk, .rst,
    .s_axi_awaddr, .s_axi_awprot, .s_axi_awvalid, .s_axi_awready,
    .s_axi_wdata, .s_axi_wstrb, .s_axi_wvalid, .s_axi_wready,
    .s_axi_bresp, .s_axi_bvalid, .s_axi_bready,
    .s_axi_araddr, .s_axi_arprot, .s_axi_arvalid, .s_axi_arready,
    .s_axi_rdata, .s_axi_rresp, .s_axi_rvalid, .s_axi_rready,
    .start_pulse(start), .soft_reset_pulse(soft_reset), .n_layers, .desc,
    .busy, .done, .error, .err_code, .total_cyc, .mac_active, .stall,
    .ps_busy_violation(err_flags), .layer_cyc, .logit
  );

  gos_core #(.PS_RD_LAT(PS_RD_LAT)) u_core (
    .clk, .rst, .start, .soft_reset, .n_layers, .desc,
    .busy, .done, .error, .err_code, .total_cyc, .mac_active, .stall, .layer_cyc, .logit, .err_flags,
    .act0_en, .act0_we, .act0_be, .act0_addr, .act0_wdata, .act0_rdata,
    .act1_en, .act1_we, .act1_be, .act1_addr, .act1_wdata, .act1_rdata,
    .wgt_en, .wgt_we, .wgt_be, .wgt_addr, .wgt_wdata, .wgt_rdata,
    .qp_en, .qp_we, .qp_be, .qp_addr, .qp_wdata, .qp_rdata
  );
endmodule
