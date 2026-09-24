`timescale 1ns/1ps
// gos_rotator — byte rotator aligning the 8 ACT banks to the 8 array rows
// (ARCH_SPEC "Memory: conflict-free read"; vectors/README.md §3.3).
//
// Rule: bank b supplies row r = (b - kx) mod 8, i.e.
//         out_rows[r] = in_banks[(r + kx) mod 8]      (r = 0..7)
//
// Interface
//   in_valid, in_tok[TW-1:0]   token (opaque, passed through)
//   in_banks[b]  (b = 0..7)    byte of ACT bank b (from the BRAM read, bank order)
//   in_kx[2:0]                 rotation amount (issue-stream field `rot` = kx)
//   out_valid, out_tok         token, same latency as the data
//   out_rows[r]  (r = 0..7)    byte for array row r
//
// Latency: L_ROT = 1 (one output register stage; the 8:1 byte mux is the only
// logic). Only out_valid is reset; data/token registers have no reset.
module gos_rotator
  import gos_pkg::*;
#(
  parameter int TW = 16
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic                  in_valid,
  input  logic [TW-1:0]         in_tok,
  input  logic [N-1:0][7:0]     in_banks,
  input  logic [2:0]            in_kx,
  output logic                  out_valid,
  output logic [TW-1:0]         out_tok,
  output logic [N-1:0][7:0]     out_rows
);

  localparam int L_ROT = 1;

  logic [N-1:0][7:0] rows_c;

  always_comb begin
    for (int r = 0; r < N; r++) begin
      rows_c[r] = in_banks[3'(r) + in_kx];   // 3-bit add: (r + kx) mod 8
    end
  end

  always_ff @(posedge clk) begin
    if (rst) out_valid <= 1'b0;
    else     out_valid <= in_valid;
  end

  always_ff @(posedge clk) begin
    out_tok  <= in_tok;
    out_rows <= rows_c;
  end

endmodule
