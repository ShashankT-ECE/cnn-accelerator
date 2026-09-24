`timescale 1ns/1ps
// gos_wgt_mem — weight memory WGT, 64 bit x 16384 words (ARCH_SPEC "Memory", FORMATS.md §2).
// One inferred UG901 true-dual-port block RAM with byte-write enables on port A (expected
// 32 RAMB36). No XPM.
//
// Port A (PS): ps_en (access), ps_we (write), ps_be[7:0] (byte enables; byte lane j = bits 8j+7:8j
//   written iff ps_en & ps_we & ps_be[j]), ps_addr[13:0], ps_wdata[63:0].
//   ps_rdata[63:0] valid PS_RD_LAT cycles after a cycle with ps_en = 1; READ_FIRST (a write
//   returns the old word). Undefined otherwise.
// Port B (accelerator read): rd_addr[13:0], in_valid/in_tok -> out_valid/out_tok,
//   rd_data[63:0] (byte lane j = output channel oc_tile*8 + j). L_WGT_MEM = gos_pkg::L_MEM_ACC = 2
//   (BRAM read register + output register). rd_data is defined only when out_valid = 1.
//
// Collision (same word, same clock edge): a port-A write and a port-B read of the same word give
//   UNDEFINED port-B data (don't care). The PS loads weights before start; tests do not rely on it.
//   No accel_busy lock-out on this memory (not part of its contract).
//
// Parameters: TW (token width), PS_RD_LAT in {1, 2} (default 1; 2 adds a PS output register).
//   Elaboration fails for any other PS_RD_LAT.
// Reset: synchronous active-high rst clears only the valid pipeline.
module gos_wgt_mem
  import gos_pkg::*;
#(
  parameter int TW        = 16,
  parameter int PS_RD_LAT = 1
) (
  input  logic              clk,
  input  logic              rst,
  // port A: PS
  input  logic              ps_en,
  input  logic              ps_we,
  input  logic [7:0]        ps_be,
  input  logic [WGT_AW-1:0] ps_addr,
  input  logic [63:0]       ps_wdata,
  output logic [63:0]       ps_rdata,
  // port B: accelerator read
  input  logic              in_valid,
  input  logic [TW-1:0]     in_tok,
  input  logic [WGT_AW-1:0] rd_addr,
  output logic              out_valid,
  output logic [TW-1:0]     out_tok,
  output logic [63:0]       rd_data
);

  localparam int L_WGT_MEM = L_MEM_ACC;   // = 2

  if (PS_RD_LAT != 1 && PS_RD_LAT != 2) begin : g_bad_ps_rd_lat
    $fatal(1, "gos_wgt_mem: PS_RD_LAT must be 1 or 2 (got %0d)", PS_RD_LAT);
  end
  if (L_WGT_MEM != 2) begin : g_bad_lat
    $fatal(1, "gos_wgt_mem: accelerator read latency is fixed at 2");
  end

  (* ram_style = "block" *) logic [63:0] mem [0:WGT_DEPTH-1];

  // ---------------------------------------------------------------- port A (read-first, byte write)
  logic [63:0] a_q;
  always_ff @(posedge clk) begin
    if (ps_en) begin
      for (int j = 0; j < 8; j++) begin
        if (ps_we && ps_be[j]) mem[ps_addr][8*j +: 8] <= ps_wdata[8*j +: 8];
      end
      a_q <= mem[ps_addr];
    end
  end

  if (PS_RD_LAT == 1) begin : g_ps_lat1
    assign ps_rdata = a_q;
  end else begin : g_ps_lat2
    logic [63:0] ps_q;
    always_ff @(posedge clk) ps_q <= a_q;
    assign ps_rdata = ps_q;
  end

  // ---------------------------------------------------------------- port B (read + output register)
  logic          v1;
  logic [TW-1:0] tok1;
  logic [63:0]   b_q;

  always_ff @(posedge clk) begin
    if (rst) begin
      v1        <= 1'b0;
      out_valid <= 1'b0;
    end else begin
      v1        <= in_valid;
      out_valid <= v1;
    end
  end

  always_ff @(posedge clk) begin
    tok1    <= in_tok;
    out_tok <= tok1;
  end

  always_ff @(posedge clk) begin
    if (in_valid) b_q <= mem[rd_addr];
  end
  always_ff @(posedge clk) begin
    if (v1) rd_data <= b_q;
  end

endmodule
