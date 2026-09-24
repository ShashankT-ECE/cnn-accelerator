`timescale 1ns/1ps
// gos_qparam_mem — requant parameter memory QPARAM (ARCH_SPEC "Memory", FORMATS.md §3,
// DECISIONS D8-10, D11-6).
//
// Two inferred UG901 true-dual-port block RAMs read in parallel at channel index i:
//   QP_E[i] (256 x 64) = combined word 2i   = {m[31:0] (63:32), q_bias[31:0] (31:0)}
//   QP_O[i] (256 x  6) = combined word 2i+1 = {58'b0, s[5:0]}
// QP_O stores only the 6 defined bits s[5:0]; FORMATS.md fixes bits 63:6 of odd words to 0, so
// a PS read of an odd word returns {58'b0, s} whatever was written to bits 63:6.
// No XPM.
//
// Port A (PS, combined 512-word view): ps_en, ps_we, ps_be[7:0], ps_addr[8:0], ps_wdata[63:0].
//   ps_addr[0] selects the bank (0: QP_E, 1: QP_O), word index = ps_addr[8:1] (APPROVED D11-6).
//   QP_E: byte lane j written iff ps_en & ps_we & ps_be[j] & !ps_addr[0].
//   QP_O: s <= ps_wdata[5:0] iff ps_en & ps_we & ps_be[0] & ps_addr[0] (ps_be[7:1] ignored).
//   ps_rdata[63:0] valid PS_RD_LAT cycles after a cycle with ps_en = 1; READ_FIRST; undefined
//   otherwise.
// Port B (accelerator read): rd_ch[7:0] = channel index i (8-bit, wraps at 256, D8-10),
//   in_valid/in_tok -> out_valid/out_tok, and in one access: q_bias (int32), m (uint32), s (uint6).
//   L_QPARAM_MEM = gos_pkg::L_MEM_ACC = 2 (BRAM read register + output register). Outputs are
//   defined only when out_valid = 1.
//
// Collision (same entry, same clock edge): a port-A write and a port-B read of the same channel
//   give UNDEFINED port-B data (don't care). The PS loads QPARAM before start; tests do not rely on it.
//   No accel_busy lock-out on this memory (not part of its contract).
//
// Parameters: TW (token width), PS_RD_LAT in {1, 2} (default 1; 2 adds a PS output register).
//   Elaboration fails for any other PS_RD_LAT.
// Reset: synchronous active-high rst clears only the valid pipeline.
module gos_qparam_mem
  import gos_pkg::*;
#(
  parameter int TW        = 16,
  parameter int PS_RD_LAT = 1
) (
  input  logic               clk,
  input  logic               rst,
  // port A: PS (combined view, 512 x 64)
  input  logic               ps_en,
  input  logic               ps_we,
  input  logic [7:0]         ps_be,
  input  logic [QP_AW:0]     ps_addr,
  input  logic [63:0]        ps_wdata,
  output logic [63:0]        ps_rdata,
  // port B: accelerator read
  input  logic               in_valid,
  input  logic [TW-1:0]      in_tok,
  input  logic [QP_AW-1:0]   rd_ch,
  output logic               out_valid,
  output logic [TW-1:0]      out_tok,
  output logic signed [31:0] q_bias,
  output logic [M_W-1:0]     m,
  output logic [S_W-1:0]     s
);

  localparam int L_QPARAM_MEM = L_MEM_ACC;   // = 2

  if (PS_RD_LAT != 1 && PS_RD_LAT != 2) begin : g_bad_ps_rd_lat
    $fatal(1, "gos_qparam_mem: PS_RD_LAT must be 1 or 2 (got %0d)", PS_RD_LAT);
  end
  if (L_QPARAM_MEM != 2) begin : g_bad_lat
    $fatal(1, "gos_qparam_mem: accelerator read latency is fixed at 2");
  end
  if (M_W != 32 || S_W != 6) begin : g_bad_fmt
    $fatal(1, "gos_qparam_mem: FORMATS.md QPARAM layout assumes M_W = 32, S_W = 6");
  end

  (* ram_style = "block" *) logic [63:0]    mem_e [0:QP_CH-1];
  (* ram_style = "block" *) logic [S_W-1:0] mem_o [0:QP_CH-1];

  logic [QP_AW-1:0] ps_idx;
  logic             ps_odd;
  assign ps_idx = ps_addr[QP_AW:1];
  assign ps_odd = ps_addr[0];

  // ---------------------------------------------------------------- port A (read-first)
  logic [63:0]    ea_q;
  logic [S_W-1:0] oa_q;
  logic           sel_odd;

  always_ff @(posedge clk) begin
    if (ps_en) begin
      for (int j = 0; j < 8; j++) begin
        if (ps_we && !ps_odd && ps_be[j]) mem_e[ps_idx][8*j +: 8] <= ps_wdata[8*j +: 8];
      end
      ea_q <= mem_e[ps_idx];
    end
  end

  always_ff @(posedge clk) begin
    if (ps_en) begin
      if (ps_we && ps_odd && ps_be[0]) mem_o[ps_idx] <= ps_wdata[S_W-1:0];
      oa_q <= mem_o[ps_idx];
    end
  end

  always_ff @(posedge clk) begin
    if (ps_en) sel_odd <= ps_odd;
  end

  logic [63:0] ps_mux;
  assign ps_mux = sel_odd ? {{(64-S_W){1'b0}}, oa_q} : ea_q;

  if (PS_RD_LAT == 1) begin : g_ps_lat1
    assign ps_rdata = ps_mux;
  end else begin : g_ps_lat2
    logic [63:0] ps_q;
    always_ff @(posedge clk) ps_q <= ps_mux;
    assign ps_rdata = ps_q;
  end

  // ---------------------------------------------------------------- port B (read + output register)
  logic           v1;
  logic [TW-1:0]  tok1;
  logic [63:0]    eb_q,  eb_q2;
  logic [S_W-1:0] ob_q,  ob_q2;

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
    if (in_valid) eb_q <= mem_e[rd_ch];
  end
  always_ff @(posedge clk) begin
    if (in_valid) ob_q <= mem_o[rd_ch];
  end
  always_ff @(posedge clk) begin
    if (v1) begin
      eb_q2 <= eb_q;
      ob_q2 <= ob_q;
    end
  end

  assign q_bias = signed'(eb_q2[31:0]);
  assign m      = eb_q2[63:32];
  assign s      = ob_q2;

endmodule
