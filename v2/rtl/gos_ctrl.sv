`timescale 1ns/1ps
// gos_ctrl — issue stage of the V2 core (ARCH_SPEC "Tiling / loop nest", FORMATS.md).
//
// On `load` (one cycle) it latches the 16-word descriptor and resets the loop
// counters; from the next cycle it walks
//     oc_tile -> oy (oy pair when pooling) -> ox_tile -> dy -> ic -> ky -> kx
// issuing one k per cycle with no gaps (T*K issues), then drops `issuing`.
// Every address is produced by counters and incremental adds only (no
// multiply / divide / modulo; checked by v2/model/tests/test_rtl_no_mul.py). The
// counter structure mirrors v2/model/gos_addr_stream.py (_tile_walk / iter_issues).
//
// Issue register (L_ISSUE = 1): the state of cycle c is registered into iss_* and
// appears at cycle c+1. Per issue: 8 ACT bank read addresses
// rd[b] = rowbase + ox0/8 + (b < kx) (full 16-bit counters; the ACT port uses
// bits 11:0), rotate = kx, WGT address, first/last, and the tile token.
module gos_ctrl
  import gos_pkg::*;
(
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    load,        // latch desc, start issuing next cycle
  input  logic [15:0][31:0]       desc,        // descriptor words w0..w15
  input  logic [2:0]              layer,       // layer index for the token
  output logic                    issuing,     // loop nest still has k to issue
  output logic                    iss_valid,
  output logic                    iss_first,
  output logic                    iss_last,
  output logic [N-1:0][15:0]      iss_rd,      // full read-address counters, bank b
  output logic [2:0]              iss_rot,
  output logic [15:0]             iss_wgt,
  output gos_tok_t                iss_tok,
  // debug view of the issued k (compared with the Step 2.2 issue vectors; pruned in synthesis)
  output logic [15:0]             iss_k,
  output logic [15:0]             iss_ic,
  output logic [7:0]              iss_ky,
  output logic [7:0]              iss_kx,
  output logic [7:0]              iss_ox_tile,
  output logic [7:0]              iss_oy,
  output logic [7:0]              iss_oc_tile,
  output logic                    iss_dy,
  output logic [15:0]             iss_tile,
  // layer constants for the drain / writer side (stable while the layer runs)
  output logic [15:0]             out_plane,
  output logic                    in_sel
);
  localparam int L_ISSUE = 1;

  // ---- latched descriptor fields (FORMATS.md descriptor layout) ----
  logic [15:0] f_ic, f_oc, f_ow, f_out_h, f_oh, f_in_wpr, f_in_plane, f_oc_tiles, f_ow_tiles;
  logic [15:0] f_out_wpr;
  logic [7:0]  f_kh, f_kw;
  logic [31:0] f_wgt_base, f_qp_base, f_k;
  logic        f_relu, f_pool, f_raw;
  // loop bounds minus one (subtractions only)
  logic [15:0] ic_m1, rows_m1, owt_m1, oct_m1;
  logic [7:0]  kh_m1, kw_m1;
  logic [2:0]  f_layer;

  // ---- loop counters and incremental bases ----
  logic [7:0]  kx, ky;
  logic [15:0] ic, k;
  logic        dy;
  logic [15:0] ox_tile, out_row, oc_tile, tile;
  logic [15:0] oy, oy_base;
  logic [15:0] ky_off, ic_base, in_row, ox_word, rem, ch_rem;
  logic [15:0] wgt, wgt_tile, qp_tile, out_ch_tile, out_row_off;

  wire kx_last  = (kx == kw_m1);
  wire ky_last  = (ky == kh_m1);
  wire ic_last  = (ic == ic_m1);
  wire k_last   = kx_last & ky_last & ic_last;
  wire dy_last  = ~f_pool | dy;
  wire ox_last  = (ox_tile == owt_m1);
  wire row_last = (out_row == rows_m1);
  wire oc_last  = (oc_tile == oct_m1);

  // row / channel validity masks from down-counters: bit r set iff r < rem
  function automatic logic [N-1:0] lt_mask(input logic [15:0] r);
    logic [N-1:0] m;
    for (int i = 0; i < N; i++) m[i] = (r > 16'(i));
    return m;
  endfunction

  // next-tile bases (adds only)
  wire [15:0] in_row_step = f_pool ? (f_in_wpr + f_in_wpr) : f_in_wpr;   // IN_WPR per oy (x2 per pair)
  wire [15:0] col_word    = f_pool ? {1'b0, ox_word[15:1]} : ox_word;     // pooled: ox_tile >> 1

  always_ff @(posedge clk) begin
    if (rst) begin
      issuing   <= 1'b0;
      iss_valid <= 1'b0;
    end else begin
      iss_valid <= issuing;
      if (load) begin
        issuing <= 1'b1;
      end else if (issuing && k_last && dy_last && ox_last && row_last && oc_last) begin
        issuing <= 1'b0;
      end
    end
  end

  // ---- descriptor latch + counter advance (data path, no reset needed) ----
  always_ff @(posedge clk) begin
    if (load) begin
      f_ic       <= desc[0][15:0];   f_oc       <= desc[0][31:16];
      f_kh       <= desc[2][7:0];    f_kw       <= desc[2][15:8];
      f_oh       <= desc[3][15:0];   f_ow       <= desc[3][31:16];
      f_wgt_base <= desc[4];         f_qp_base  <= desc[5];
      f_relu     <= desc[6][0];      f_pool     <= desc[6][1];
      f_raw      <= desc[6][2];      in_sel     <= desc[6][3];
      f_k        <= desc[7];
      f_in_wpr   <= desc[8][15:0];   f_in_plane <= desc[8][31:16];
      f_oc_tiles <= desc[9][15:0];   f_ow_tiles <= desc[9][31:16];
      f_out_h    <= desc[10][31:16];
      f_out_wpr  <= desc[11][15:0];  out_plane  <= desc[11][31:16];
      f_layer    <= layer;
      ic_m1      <= desc[0][15:0] - 16'd1;
      kh_m1      <= desc[2][7:0] - 8'd1;
      kw_m1      <= desc[2][15:8] - 8'd1;
      owt_m1     <= desc[9][31:16] - 16'd1;
      oct_m1     <= desc[9][15:0] - 16'd1;
      rows_m1    <= (desc[6][1] ? desc[10][31:16] : desc[3][15:0]) - 16'd1;
      // counters
      kx <= '0; ky <= '0; ic <= '0; k <= '0; dy <= 1'b0; tile <= '0;
      ox_tile <= '0; out_row <= '0; oc_tile <= '0;
      oy <= '0; oy_base <= '0;
      ky_off <= '0; ic_base <= '0; in_row <= '0; ox_word <= '0;
      rem <= desc[3][31:16];                 // OW - ox0
      ch_rem <= desc[0][31:16];              // OC - oc_tile*8
      wgt <= desc[4][15:0]; wgt_tile <= desc[4][15:0];
      qp_tile <= desc[5][15:0];
      out_ch_tile <= '0; out_row_off <= '0;
    end else if (issuing) begin
      if (!k_last) begin
        k   <= k + 16'd1;
        wgt <= wgt + 16'd1;
        if (!kx_last) begin
          kx <= kx + 8'd1;
        end else begin
          kx <= '0;
          if (!ky_last) begin
            ky     <= ky + 8'd1;
            ky_off <= ky_off + f_in_wpr;
          end else begin
            ky      <= '0;
            ic      <= ic + 16'd1;
            ic_base <= ic_base + f_in_plane;
            ky_off  <= ic_base + f_in_plane;
          end
        end
      end else begin
        // tile finished: reset the k counters, advance the tile loops
        kx <= '0; ky <= '0; ic <= '0; k <= '0;
        tile <= tile + 16'd1;
        wgt  <= wgt_tile;
        if (!dy_last) begin
          dy      <= 1'b1;
          oy      <= oy + 16'd1;
          ic_base <= in_row + f_in_wpr + ox_word;
          ky_off  <= in_row + f_in_wpr + ox_word;
        end else if (!ox_last) begin
          dy      <= 1'b0;
          ox_tile <= ox_tile + 16'd1;
          ox_word <= ox_word + 16'd1;
          rem     <= rem - 16'd8;
          oy      <= oy_base;
          ic_base <= in_row + ox_word + 16'd1;
          ky_off  <= in_row + ox_word + 16'd1;
        end else if (!row_last) begin
          dy          <= 1'b0;
          ox_tile     <= '0;
          ox_word     <= '0;
          rem         <= f_ow;
          out_row     <= out_row + 16'd1;
          out_row_off <= out_row_off + f_out_wpr;
          in_row      <= in_row + in_row_step;
          oy_base     <= oy_base + (f_pool ? 16'd2 : 16'd1);
          oy          <= oy_base + (f_pool ? 16'd2 : 16'd1);
          ic_base     <= in_row + in_row_step;
          ky_off      <= in_row + in_row_step;
        end else if (!oc_last) begin
          dy          <= 1'b0;
          ox_tile     <= '0;
          ox_word     <= '0;
          rem         <= f_ow;
          out_row     <= '0;
          out_row_off <= '0;
          in_row      <= '0;
          oy_base     <= '0;
          oy          <= '0;
          ic_base     <= '0;
          ky_off      <= '0;
          oc_tile     <= oc_tile + 16'd1;
          wgt_tile    <= wgt_tile + f_k[15:0];
          wgt         <= wgt_tile + f_k[15:0];
          ch_rem      <= ch_rem - 16'd8;
          qp_tile     <= qp_tile + 16'd8;
          out_ch_tile <= out_ch_tile + {out_plane[12:0], 3'b000};   // + 8*OUT_PLANE (shift)
        end
      end
    end
  end

  // ---- issue register ----
  always_ff @(posedge clk) begin
    for (int b = 0; b < N; b++)
      iss_rd[b] <= ky_off + ((8'(b) < kx) ? 16'd1 : 16'd0);   // (b < kx), as gos_addr_stream
    iss_rot   <= kx[2:0];
    iss_wgt   <= wgt;
    iss_first <= (k == 16'd0);
    iss_last  <= k_last;
    iss_tok.layer     <= f_layer;
    iss_tok.oc_tile   <= oc_tile[7:0];
    iss_tok.dy        <= dy;
    iss_tok.pool_en   <= f_pool;
    iss_tok.relu_en   <= f_relu;
    iss_tok.out_raw   <= f_raw;
    iss_tok.row_mask  <= lt_mask(rem);
    iss_tok.ch_valid  <= lt_mask(ch_rem);
    iss_tok.word_base <= ACT_AW'(out_ch_tile + out_row_off + col_word);
    iss_tok.half      <= ox_word[0];
    iss_tok.qp_base   <= qp_tile[QP_AW-1:0];
    iss_k       <= k;
    iss_ic      <= ic;
    iss_ky      <= ky;
    iss_kx      <= kx;
    iss_ox_tile <= ox_tile[7:0];
    iss_oy      <= oy[7:0];
    iss_oc_tile <= oc_tile[7:0];
    iss_dy      <= dy;
    iss_tile    <= tile;
  end

  // unused descriptor fields (IH, IW, OUT_W and the END words are checked by gos_cfg_check)
  logic unused;
  assign unused = ^{desc[1], desc[10][15:0], desc[12], desc[13], desc[14], desc[15], f_ic, f_oc,
                    f_out_h, f_oh, f_oc_tiles, f_ow_tiles, f_wgt_base[31:16], f_qp_base[31:16],
                    f_k[31:16], rows_m1};
endmodule
