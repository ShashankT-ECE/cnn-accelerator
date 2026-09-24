`timescale 1ns/1ps
// gos_act_buf — ONE activation buffer (ACT0 or ACT1; instantiated twice at integration).
// ARCH_SPEC "Memory", FORMATS.md §1, DECISIONS D8-7/D8-8, D11-1.
//
// Storage: 8 banks x 8 bit x 4096 words. Each bank is its own inferred UG901 true-dual-port
// block RAM (8 x 4096 -> 1 RAMB36 each), so every bank has an independent port-B read address
// and its own port-A write enable. No XPM.
//
// Port A (one 64-bit word, byte lane b = bits 8b+7:8b = bank b), muxed by accel_busy:
//   accel_busy = 0 : PS side. ps_en (access), ps_we (write), ps_be[7:0] (byte enables; a byte is
//                    written iff ps_en & ps_we & ps_be[b]), ps_addr[11:0], ps_wdata[63:0].
//                    ps_rdata[63:0] is valid PS_RD_LAT cycles after a cycle with ps_en = 1 and is
//                    read-first (a write returns the old word); undefined otherwise.
//                    The accelerator write inputs are ignored.
//   accel_busy = 1 : accelerator write side. acc_wr_en, acc_wr_addr[11:0] (common word address),
//                    acc_wr_data[63:0], acc_wr_be[7:0]; byte b written iff acc_wr_en & acc_wr_be[b].
//                    PS accesses are ignored (APPROVED D11-1): PS writes do not land, a PS read
//                    returns 0 (PS_RD_LAT cycles later), and any ps_en = 1 sets the sticky flag
//                    ps_busy_violation (cleared only by rst).
//   accel_busy is sampled in the same cycle as the port-A request (combinational mux in front of
//   the BRAM port A pins).
// Port B (accelerator read): rd_addr[b][11:0] per bank b (independent; 12-bit, so addresses wrap
//   at 4096, D8-8), in_valid/in_tok -> out_valid/out_tok, rd_data[b] = bank b byte (bank order).
//   L_ACT_BUF = gos_pkg::L_MEM_ACC = 2 (BRAM read register + output register). rd_data is
//   defined only when out_valid = 1.
//
// Collision behavior (same bank, same word, same clock edge):
//   * port A itself: READ_FIRST (UG901 read-first template): ps_rdata of a PS write = old data.
//   * port-A write vs port-B read of the same bank word in the same cycle: port-B data is
//     UNDEFINED (don't care). The system never does this (the core reads ACT[in_sel] and writes
//     ACT[!in_sel]; the PS is locked out while busy). Tests do not rely on it.
//
// Parameters: TW (token width), PS_RD_LAT in {1, 2} (default 1: no PS output register, matching
// the AXI BRAM Controller default read latency 1; 2 adds an output register). Elaboration fails
// for any other PS_RD_LAT.
// Reset: synchronous active-high rst clears only out_valid (pipeline valids) and ps_busy_violation.
module gos_act_buf
  import gos_pkg::*;
#(
  parameter int TW        = 16,
  parameter int PS_RD_LAT = 1
) (
  input  logic                      clk,
  input  logic                      rst,
  input  logic                      accel_busy,
  // PS side of port A (active when accel_busy = 0)
  input  logic                      ps_en,
  input  logic                      ps_we,
  input  logic [7:0]                ps_be,
  input  logic [ACT_AW-1:0]         ps_addr,
  input  logic [63:0]               ps_wdata,
  output logic [63:0]               ps_rdata,
  output logic                      ps_busy_violation,
  // accelerator write side of port A (active when accel_busy = 1)
  input  logic                      acc_wr_en,
  input  logic [ACT_AW-1:0]         acc_wr_addr,
  input  logic [63:0]               acc_wr_data,
  input  logic [7:0]                acc_wr_be,
  // port B: accelerator read, 8 independent bank addresses
  input  logic                      in_valid,
  input  logic [TW-1:0]             in_tok,
  input  logic [N-1:0][ACT_AW-1:0]  rd_addr,
  output logic                      out_valid,
  output logic [TW-1:0]             out_tok,
  output logic [N-1:0][7:0]         rd_data
);

  localparam int L_ACT_BUF = L_MEM_ACC;   // = 2

  // ---------------------------------------------------------------- parameter check
  if (PS_RD_LAT != 1 && PS_RD_LAT != 2) begin : g_bad_ps_rd_lat
    $fatal(1, "gos_act_buf: PS_RD_LAT must be 1 or 2 (got %0d)", PS_RD_LAT);
  end
  if (L_ACT_BUF != 2) begin : g_bad_lat
    $fatal(1, "gos_act_buf: accelerator read latency is fixed at 2");
  end

  // ---------------------------------------------------------------- port A mux
  logic [ACT_AW-1:0] a_addr;
  logic [63:0]       a_din;
  logic [7:0]        a_we;

  always_comb begin
    if (accel_busy) begin
      a_addr = acc_wr_addr;
      a_din  = acc_wr_data;
      a_we   = acc_wr_en ? acc_wr_be : 8'h00;
    end else begin
      a_addr = ps_addr;
      a_din  = ps_wdata;
      a_we   = (ps_en && ps_we) ? ps_be : 8'h00;
    end
  end

  // ---------------------------------------------------------------- port B valid/token pipeline
  logic          v1;
  logic [TW-1:0] tok1;

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

  // ---------------------------------------------------------------- banks (UG901 TDP, read-first A)
  logic [63:0] a_rd;   // port A read latch, all banks

  for (genvar gb = 0; gb < N; gb++) begin : g_bank
    (* ram_style = "block" *) logic [7:0] mem [0:ACT_DEPTH-1];
    logic [7:0] a_q;
    logic [7:0] b_q;
    logic [7:0] b_q2;

    // port A: read-first, write with per-bank enable. The read latch is not enabled by a_en:
    // with an enable Vivado ties ENA = 1 and emulates the hold in fabric (64 FF + 64 LUT); the PS
    // read data is only defined PS_RD_LAT cycles after a request, so no hold is needed.
    always_ff @(posedge clk) begin
      if (a_we[gb]) mem[a_addr] <= a_din[8*gb +: 8];
      a_q <= mem[a_addr];
    end

    // port B: read register (enable = in_valid) + output register (enable = stage-1 valid)
    always_ff @(posedge clk) begin
      if (in_valid) b_q <= mem[rd_addr[gb]];
    end
    always_ff @(posedge clk) begin
      if (v1) b_q2 <= b_q;
    end

    assign a_rd[8*gb +: 8] = a_q;
    assign rd_data[gb]     = b_q2;
  end

  // ---------------------------------------------------------------- PS read data / busy lock
  // ps_ok: the last PS access (ps_en = 1) happened while not busy. A read while busy returns 0.
  logic ps_ok;
  always_ff @(posedge clk) begin
    if (ps_en) ps_ok <= ~accel_busy;
  end

  always_ff @(posedge clk) begin
    if (rst)                     ps_busy_violation <= 1'b0;
    else if (ps_en && accel_busy) ps_busy_violation <= 1'b1;
  end

  if (PS_RD_LAT == 1) begin : g_ps_lat1
    assign ps_rdata = ps_ok ? a_rd : 64'd0;
  end else begin : g_ps_lat2
    logic [63:0] ps_q;
    always_ff @(posedge clk) begin
      if (!ps_ok) ps_q <= 64'd0;
      else        ps_q <= a_rd;
    end
    assign ps_rdata = ps_q;
  end

endmodule
