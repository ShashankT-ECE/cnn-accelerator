`timescale 1ns/1ps
// gos_pool — fused 2x2/2 max-pool on the drain stream (ARCH_SPEC Datapath "Pool";
// DECISIONS D8-11, D8-14; vectors/README.md §3.5; reference gos_tile_model.py /
// gos_addr_stream.py drain kinds).
//
// One input per drain cycle: one output channel (column j = in_col) x 8 pixels
// (in_q[r] = row r = pixel ox0 + r), post-requant/ReLU int8 bytes.
//
// Interface
//   in_valid, in_tok[TW-1:0]  token (opaque, passed through with the data)
//   in_q[r]   (r = 0..7)      int8 value of row r
//   in_col[2:0]               drain column j (pool buffer index)
//   in_dy                     pool phase (0: store, 1: combine); ignored if !in_pool_en
//   in_pool_en                layer pool_en
//   in_half                   pooled byte half select (= ox_tile & 1): 0 -> banks 0-3, 1 -> banks 4-7
//   in_row_mask[7:0]          bit r: row r valid (ox0 + r < OW), from the token
//   out_valid, out_tok        one output per input (1:1), latency L_POOL
//   out_data[b] (b = 0..7)    write data, bank order
//   out_be[7:0]               bank byte enables (x-tail masked)
//   out_we                    = (out_be != 0)
//
// Output convention (per input cycle, 1:1 — every drain step produces an output,
// matching the drain record kinds of gos_addr_stream):
//   pool_en = 0          (kind ACT, non-pool): out_data[b] = in_q[b] (all 8 lanes
//                         unchanged), out_be = in_row_mask.
//   pool_en = 1, dy = 0  (kind POOL STORE): column stored into buffer[in_col];
//                         out_valid = 1 with out_be = 0, out_we = 0 (out_data = don't-care).
//   pool_en = 1, dy = 1  (kind ACT, pooled): v[r] = max(buffer[in_col][r], in_q[r]),
//                         h[i] = max(v[2i], v[2i+1]) (i = 0..3), signed int8 compare (D8-14);
//                         out_data[b] = h[b & 3] (same 4 bytes on both halves, D8-11);
//                         out_be[4*half + i] = in_row_mask[2i] (pooled byte i valid iff
//                         ox0 + 2i < OW; OW even under pool), other half 0.
//   The OC-tail (ch_valid) and out_raw/LOGIT gating are NOT applied here: the
//   integration ANDs out_be with {8{ch_valid & !out_raw}} (from out_tok) to obtain
//   the drain record's be. A dy = 0 store with ch_valid = 0 is harmless (its dy = 1
//   write is masked the same way).
//
// Buffer: 8 columns x 8 bytes, written at the input edge when in_valid & pool_en & !dy;
//   read synchronously at the input edge (address in_col). A dy = 1 for column j directly
//   following the dy = 0 of column j (back-to-back) sees the new value. In the drain order
//   the dy = 1 tile follows the dy = 0 tile of the same (oc_tile, oy pair, ox_tile)
//   directly (dy is the innermost tile loop), so the buffer holds exactly one tile.
//   Vivado infers the 8 x 64-bit buffer as LUTRAM (async read + the stage-1 register).
//   D11-5 (approved): dy = 1 for a column with no prior dy = 0 store gives an undefined
//   result (whatever the buffer holds); no error flag.
//
// Latency: L_POOL = 2
//   stage 1: input registers + registered buffer read
//   stage 2: vertical max, pair max, output select -> output registers
// Only valid bits are reset; data/token/buffer registers have no reset (out_be/out_we
// are meaningful only with out_valid).
module gos_pool
  import gos_pkg::*;
#(
  parameter int TW = GOS_TOK_W
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic                  in_valid,
  input  logic [TW-1:0]         in_tok,
  input  logic [N-1:0][7:0]     in_q,
  input  logic [2:0]            in_col,
  input  logic                  in_dy,
  input  logic                  in_pool_en,
  input  logic                  in_half,
  input  logic [N-1:0]          in_row_mask,
  output logic                  out_valid,
  output logic [TW-1:0]         out_tok,
  output logic [N-1:0][7:0]     out_data,
  output logic [N-1:0]          out_be,
  output logic                  out_we
);

  localparam int L_POOL = 2;

  // ---------------- pool buffer (dy = 0 columns) ----------------
  logic [8*N-1:0] pbuf [N];

  always_ff @(posedge clk) begin
    if (in_valid && in_pool_en && !in_dy) pbuf[in_col] <= in_q;
  end

  // ---------------- stage 1 ----------------
  logic              s1_valid;
  logic [TW-1:0]     s1_tok;
  logic [N-1:0][7:0] s1_q;
  logic [8*N-1:0]    s1_stored;
  logic              s1_dy, s1_pool_en, s1_half;
  logic [N-1:0]      s1_mask;

  always_ff @(posedge clk) begin
    if (rst) s1_valid <= 1'b0;
    else     s1_valid <= in_valid;
  end

  always_ff @(posedge clk) begin
    s1_tok     <= in_tok;
    s1_q       <= in_q;
    s1_stored  <= pbuf[in_col];
    s1_dy      <= in_dy;
    s1_pool_en <= in_pool_en;
    s1_half    <= in_half;
    s1_mask    <= in_row_mask;
  end

  // ---------------- stage 2 (combinational part) ----------------
  logic signed [7:0] vmax [N];
  logic signed [7:0] hmax [N/2];
  logic [N-1:0][7:0] data_c;
  logic [N-1:0]      be_c;
  logic [N/2-1:0]    pmask;

  always_comb begin
    for (int r = 0; r < N; r++) begin
      vmax[r] = ($signed(s1_q[r]) > $signed(s1_stored[8*r +: 8])) ? $signed(s1_q[r])
                                                                  : $signed(s1_stored[8*r +: 8]);
    end
    for (int i = 0; i < N/2; i++) begin
      hmax[i]  = (vmax[2*i+1] > vmax[2*i]) ? vmax[2*i+1] : vmax[2*i];
      pmask[i] = s1_mask[2*i];
    end

    if (!s1_pool_en) begin
      data_c = s1_q;
      be_c   = s1_mask;
    end else begin
      for (int b = 0; b < N; b++) data_c[b] = hmax[b % (N/2)];
      if (!s1_dy)       be_c = '0;
      else if (s1_half) be_c = {pmask, {(N/2){1'b0}}};
      else              be_c = {{(N/2){1'b0}}, pmask};
    end
  end

  // ---------------- stage 2 registers ----------------
  always_ff @(posedge clk) begin
    if (rst) out_valid <= 1'b0;
    else     out_valid <= s1_valid;
  end

  always_ff @(posedge clk) begin
    out_tok  <= s1_tok;
    out_data <= data_c;
    out_be   <= be_c;
    out_we   <= |be_c;
  end

endmodule
