`timescale 1ns/1ps
// GOS_OOC_DEPS: gos_ctrl.sv gos_cfg_check.sv gos_act_buf.sv gos_wgt_mem.sv gos_qparam_mem.sv gos_rotator.sv gos_pe.sv gos_array.sv gos_requant.sv gos_pool.sv
// gos_core — V2 generalized INT8 8x8 output-stationary core (ARCH_SPEC, FORMATS.md,
// DECISIONS D8/D10/D11/D12). Memories + datapath + control; PS memory ports and
// the descriptor/CSR side are plain ports (gos_top adds the AXI-Lite CSR).
//
// Datapath (alignment only through valid + token pipelines, never by counting):
//   gos_ctrl issue reg (1)
//     -> ACT[in_sel] port B (2)  || WGT read (2)   [same issue, equal documented latency]
//     -> gos_rotator (1)  (token carries the WGT word, first/last, gos_tok_t)
//     -> gos_array (L_ARRAY = 6 to drain column 0; 8 drain columns)
//     -> drain stage: word = word_base + col*OUT_PLANE (incremental), qp = qp_base + col
//     -> QPARAM read (2)  (token carries the 8 accumulators + drain token)
//     -> gos_requant (6) -> gos_pool (2) -> writer (combinational gating)
//   writer: ACT[!in_sel] port A, be & {8{ch_valid[col] & !out_raw}};
//           out_raw: LOGIT[oc_tile*8 + col] <= v_raw of row 0.
//
// Sequencer: IDLE -start-> CHK1 -> CHK2 -> (error: IDLE) LOAD(l) -> RUN(l) -> ... -> IDLE
//   A layer ends when issue is finished and no tile is in flight (tiles issued vs.
//   tiles retired at the writer — event counts, not cycle counts). Cycle model:
//   LAYER_CYC = T*K + C_PIPE, TOTAL_CYC = C_START + sum(LAYER_CYC) + C_DONE
//   (derivation in v2/model/gos_cycle_model.py).
module gos_core
  import gos_pkg::*;
#(
  parameter int PS_RD_LAT = 1
) (
  input  logic                    clk,
  input  logic                    rst,
  // control / status (CSR side)
  input  logic                    start,         // 1-cycle pulse
  input  logic                    soft_reset,    // 1-cycle pulse
  input  logic [3:0]              n_layers,
  input  logic [7:0][15:0][31:0]  desc,
  output logic                    busy,
  output logic                    done,
  output logic                    error,
  output logic [31:0]             err_code,      // {16'b0, rule_id[7:0], 5'b0, layer[2:0]}
  output logic [63:0]             total_cyc,
  output logic [63:0]             mac_active,
  output logic [63:0]             stall,
  output logic [7:0][31:0]        layer_cyc,
  output logic [15:0][31:0]       logit,
  output logic [7:0]              err_flags,     // sticky: b0 ACT0 PS-busy, b1 ACT1 PS-busy, b4 array overrun
  // PS memory ports (AXI BRAM Controller side; PS_RD_LAT)
  input  logic                    act0_en, act0_we,
  input  logic [7:0]              act0_be,
  input  logic [ACT_AW-1:0]       act0_addr,
  input  logic [63:0]             act0_wdata,
  output logic [63:0]             act0_rdata,
  input  logic                    act1_en, act1_we,
  input  logic [7:0]              act1_be,
  input  logic [ACT_AW-1:0]       act1_addr,
  input  logic [63:0]             act1_wdata,
  output logic [63:0]             act1_rdata,
  input  logic                    wgt_en, wgt_we,
  input  logic [7:0]              wgt_be,
  input  logic [WGT_AW-1:0]       wgt_addr,
  input  logic [63:0]             wgt_wdata,
  output logic [63:0]             wgt_rdata,
  input  logic                    qp_en, qp_we,
  input  logic [7:0]              qp_be,
  input  logic [QP_AW:0]          qp_addr,
  input  logic [63:0]             qp_wdata,
  output logic [63:0]             qp_rdata
);
  // Structural latencies of the sequencer (used by the cycle model, gos_cycle_model.py):
  localparam int L_LOAD     = 1;  // S_LOAD cycle before issue state k0
  localparam int L_RETIRE   = 1;  // tile retire -> inflight update seen by the layer-end test
  localparam int N_CHK      = 2;  // S_CHK1 + S_CHK2 before layer 0 (C_START)
  localparam int N_DONE     = 0;  // busy falls on the edge after the last layer-end cycle (C_DONE)

  // ------------------------------------------------------------------ reset
  logic crst;
  assign crst = rst | soft_reset;

  // ------------------------------------------------------------------ sequencer
  typedef enum logic [2:0] {S_IDLE, S_CHK1, S_CHK2, S_LOAD, S_RUN} state_t;
  state_t      state;
  logic [2:0]  layer;
  logic [31:0] lcyc;              // cycles of the current layer (LOAD .. end)
  logic [7:0][26:0] fail_r;       // registered per-layer rule failures (CHK1)
  logic        nl_bad_r;
  logic [7:0][26:0] fail_c;

  for (genvar l = 0; l < 8; l++) begin : g_chk
    gos_cfg_check u_chk (.d(desc[l]), .fail(fail_c[l]));
  end

  // first failing (layer, rule): lowest layer, then lowest rule id
  logic        chk_fail;
  logic [7:0]  chk_rule;
  logic [2:0]  chk_layer;
  always_comb begin
    chk_fail = 1'b0; chk_rule = '0; chk_layer = '0;
    if (nl_bad_r) begin
      chk_fail = 1'b1; chk_rule = 8'd32; chk_layer = '0;
    end else begin
      for (int l = 7; l >= 0; l--) begin
        if (|fail_r[l]) begin
          chk_fail  = 1'b1;
          chk_layer = 3'(l);
          for (int i = 26; i >= 0; i--)
            if (fail_r[l][i]) chk_rule = 8'(i + 1);
        end
      end
    end
  end

  // ------------------------------------------------------------------ issue
  logic               ctrl_load, issuing, iss_valid, iss_first, iss_last;
  logic [N-1:0][15:0] iss_rd;
  logic [2:0]         iss_rot;
  logic [15:0]        iss_wgt, out_plane;
  gos_tok_t           iss_tok;
  logic               in_sel;

  gos_ctrl u_ctrl (
    .clk, .rst(crst), .load(ctrl_load), .desc(desc[layer]), .layer(layer),
    .issuing, .iss_valid, .iss_first, .iss_last, .iss_rd, .iss_rot, .iss_wgt, .iss_tok,
    .iss_k(), .iss_ic(), .iss_ky(), .iss_kx(), .iss_ox_tile(), .iss_oy(), .iss_oc_tile(),
    .iss_dy(), .iss_tile(), .out_plane, .in_sel
  );

  logic        iss_fire;          // issue advances this cycle
  logic        arr_in_valid;      // valid array input (MAC_ACTIVE)

  // tiles in flight: +1 per issued last k, -1 per retired tile (drain column 7 at the writer)
  logic        retire;
  logic [15:0] inflight;
  wire         layer_end = (state == S_RUN) && !issuing && !iss_valid && (inflight == '0);

  always_ff @(posedge clk) begin
    if (crst) begin
      state <= S_IDLE; layer <= '0; lcyc <= '0;
      busy <= 1'b0; done <= 1'b0; error <= 1'b0; err_code <= '0;
      total_cyc <= '0; mac_active <= '0; stall <= '0; layer_cyc <= '0;
      inflight <= '0; nl_bad_r <= 1'b0; fail_r <= '0;
    end else begin
      inflight <= inflight + ((iss_valid && iss_last) ? 16'd1 : 16'd0) - (retire ? 16'd1 : 16'd0);
      if (busy) total_cyc <= total_cyc + 64'd1;
      if (arr_in_valid) mac_active <= mac_active + 64'd1;
      // STALL: issue-idle cycles inside the issue phase. gos_ctrl issues every cycle
      // while `issuing` (no back-pressure, D10), so this reads 0 by construction.
      if (issuing && !iss_fire) stall <= stall + 64'd1;
      case (state)
        S_IDLE: if (start) begin
          state <= S_CHK1; busy <= 1'b1; done <= 1'b0; error <= 1'b0; err_code <= '0;
          total_cyc <= '0; mac_active <= '0; stall <= '0; layer_cyc <= '0; layer <= '0;
        end
        S_CHK1: begin
          nl_bad_r <= (n_layers == 4'd0) || (n_layers > 4'd8);
          for (int l = 0; l < 8; l++)
            fail_r[l] <= (4'(l) < n_layers) ? fail_c[l] : '0;
          state <= S_CHK2;
        end
        S_CHK2: begin
          if (chk_fail) begin
            error <= 1'b1; busy <= 1'b0; state <= S_IDLE;
            err_code <= {16'b0, chk_rule, 5'b0, chk_layer};
          end else begin
            state <= S_LOAD; lcyc <= '0;
          end
        end
        S_LOAD: begin
          lcyc <= lcyc + 32'd1; state <= S_RUN;
        end
        S_RUN: begin
          lcyc <= lcyc + 32'd1;
          if (layer_end) begin
            layer_cyc[layer] <= lcyc + 32'd1;           // LOAD .. this cycle inclusive
            if (4'(layer) == n_layers - 4'd1) begin
              state <= S_IDLE; busy <= 1'b0; done <= 1'b1;
            end else begin
              layer <= layer + 3'd1; state <= S_LOAD; lcyc <= '0;
            end
          end
        end
        default: state <= S_IDLE;
      endcase
    end
  end
  assign ctrl_load = (state == S_LOAD);
  assign iss_fire  = issuing;     // every issuing cycle issues one k

  // ------------------------------------------------------------------ ACT / WGT read
  localparam int TWA = 2 + 3 + GOS_TOK_W;              // {first, last, rot, tok}
  logic [TWA-1:0]      a_tok_in, a0_tok, a1_tok, a_tok;
  logic                a0_valid, a1_valid, a_valid;
  logic [N-1:0][7:0]   a0_data, a1_data, a_data;
  logic [N-1:0][ACT_AW-1:0] rd12;
  logic                w_valid;
  logic [63:0]         w_data;
  logic                wr_en;
  logic [ACT_AW-1:0]   wr_addr;
  logic [63:0]         wr_data;
  logic [7:0]          wr_be;
  logic                viol0, viol1;

  assign a_tok_in = {iss_first, iss_last, iss_rot, iss_tok};
  for (genvar b = 0; b < N; b++) begin : g_rd
    assign rd12[b] = iss_rd[b][ACT_AW-1:0];            // 12-bit port: wraps (D8-8)
  end

  gos_act_buf #(.TW(TWA), .PS_RD_LAT(PS_RD_LAT)) u_act0 (
    .clk, .rst(crst), .accel_busy(busy),
    .ps_en(act0_en), .ps_we(act0_we), .ps_be(act0_be), .ps_addr(act0_addr),
    .ps_wdata(act0_wdata), .ps_rdata(act0_rdata), .ps_busy_violation(viol0),
    .acc_wr_en(wr_en && in_sel), .acc_wr_addr(wr_addr), .acc_wr_data(wr_data), .acc_wr_be(wr_be),
    .in_valid(iss_valid && !in_sel), .in_tok(a_tok_in), .rd_addr(rd12),
    .out_valid(a0_valid), .out_tok(a0_tok), .rd_data(a0_data)
  );
  gos_act_buf #(.TW(TWA), .PS_RD_LAT(PS_RD_LAT)) u_act1 (
    .clk, .rst(crst), .accel_busy(busy),
    .ps_en(act1_en), .ps_we(act1_we), .ps_be(act1_be), .ps_addr(act1_addr),
    .ps_wdata(act1_wdata), .ps_rdata(act1_rdata), .ps_busy_violation(viol1),
    .acc_wr_en(wr_en && !in_sel), .acc_wr_addr(wr_addr), .acc_wr_data(wr_data), .acc_wr_be(wr_be),
    .in_valid(iss_valid && in_sel), .in_tok(a_tok_in), .rd_addr(rd12),
    .out_valid(a1_valid), .out_tok(a1_tok), .rd_data(a1_data)
  );
  assign a_valid = a0_valid | a1_valid;                 // only ACT[in_sel] is read
  assign a_tok   = a1_valid ? a1_tok  : a0_tok;
  assign a_data  = a1_valid ? a1_data : a0_data;

  gos_wgt_mem #(.TW(1), .PS_RD_LAT(PS_RD_LAT)) u_wgt (
    .clk, .rst(crst),
    .ps_en(wgt_en), .ps_we(wgt_we), .ps_be(wgt_be), .ps_addr(wgt_addr),
    .ps_wdata(wgt_wdata), .ps_rdata(wgt_rdata),
    .in_valid(iss_valid), .in_tok(1'b0), .rd_addr(iss_wgt[WGT_AW-1:0]),
    .out_valid(w_valid), .out_tok(), .rd_data(w_data)
  );

  // ------------------------------------------------------------------ rotator
  localparam int TWR = 64 + TWA;                        // {wgt word, first, last, rot, tok}
  logic                r_valid;
  logic [TWR-1:0]      r_tok;
  logic [N-1:0][7:0]   r_rows;
  gos_rotator #(.TW(TWR)) u_rot (
    .clk, .rst(crst), .in_valid(a_valid), .in_tok({w_data, a_tok}), .in_banks(a_data),
    .in_kx(a_tok[GOS_TOK_W +: 3]),
    .out_valid(r_valid), .out_tok(r_tok), .out_rows(r_rows)
  );

  // ------------------------------------------------------------------ array
  logic                       arr_first, arr_last;
  logic [N-1:0][7:0]          arr_w;
  gos_tok_t                   arr_tok_in, d_tok;
  logic                       d_valid, overrun;
  logic [2:0]                 d_col;
  logic [N-1:0][ACC_W-1:0]    d_acc;
  assign arr_in_valid = r_valid;
  assign arr_tok_in   = r_tok[GOS_TOK_W-1:0];
  assign arr_last     = r_tok[GOS_TOK_W + 3];
  assign arr_first    = r_tok[GOS_TOK_W + 4];
  assign arr_w        = r_tok[TWA +: 64];               // byte j = weight of output channel j

  gos_array #(.TW(GOS_TOK_W)) u_array (
    .clk, .rst(crst), .in_valid(arr_in_valid), .in_first(arr_first), .in_last(arr_last),
    .in_tok(arr_tok_in), .a_vec(r_rows), .w_vec(arr_w),
    .out_valid(d_valid), .out_col(d_col), .out_acc(d_acc), .out_tok(d_tok), .err_overrun(overrun)
  );

  // ------------------------------------------------------------------ drain stage
  // Output word of drain column j = word_base + j*OUT_PLANE, built incrementally
  // (column 0 loads word_base, each following column adds OUT_PLANE); QPARAM channel
  // = qp_base + j. Columns always drain 0..7 consecutively (gos_array, D11-3).
  typedef struct packed {
    gos_tok_t          tok;
    logic [2:0]        col;
    logic [ACT_AW-1:0] word;
  } dtok_t;
  localparam int DTW = $bits(dtok_t);
  dtok_t             d_dtok;
  logic [ACT_AW-1:0] word_prev;
  always_comb begin
    d_dtok.tok  = d_tok;
    d_dtok.col  = d_col;
    d_dtok.word = (d_col == 3'd0) ? d_tok.word_base : (word_prev + out_plane[ACT_AW-1:0]);
  end
  always_ff @(posedge clk) if (d_valid) word_prev <= d_dtok.word;

  logic                     q_valid;
  logic [N*ACC_W+DTW-1:0]   q_tok;
  logic signed [31:0]       q_bias;
  logic [M_W-1:0]           q_m;
  logic [S_W-1:0]           q_s;
  gos_qparam_mem #(.TW(N*ACC_W + DTW), .PS_RD_LAT(PS_RD_LAT)) u_qp (
    .clk, .rst(crst),
    .ps_en(qp_en), .ps_we(qp_we), .ps_be(qp_be), .ps_addr(qp_addr),
    .ps_wdata(qp_wdata), .ps_rdata(qp_rdata),
    .in_valid(d_valid), .in_tok({d_acc, d_dtok}), .rd_ch(d_tok.qp_base + QP_AW'(d_col)),
    .out_valid(q_valid), .out_tok(q_tok), .q_bias, .m(q_m), .s(q_s)
  );
  dtok_t q_dtok;
  assign q_dtok = q_tok[DTW-1:0];

  // ------------------------------------------------------------------ requant
  logic                    rq_valid, rq_raw;
  logic [DTW-1:0]          rq_tok;
  logic [N-1:0][7:0]       rq_q;
  logic [N-1:0][ACC_W-1:0] rq_v;
  gos_requant #(.TW(DTW)) u_rq (
    .clk, .rst(crst), .in_valid(q_valid), .in_tok(q_tok[DTW-1:0]),
    .acc(q_tok[DTW +: N*ACC_W]), .q_bias(q_bias), .m(q_m), .s(q_s),
    .relu_en(q_dtok.tok.relu_en), .out_raw(q_dtok.tok.out_raw),
    .out_valid(rq_valid), .out_tok(rq_tok), .q_int8(rq_q), .v_raw(rq_v), .out_raw_d(rq_raw)
  );
  dtok_t rq_dtok;
  assign rq_dtok = rq_tok;

  // ------------------------------------------------------------------ pool
  logic                 p_valid, p_we_unused;
  logic [ACC_W+DTW-1:0] p_tok;
  logic [N-1:0][7:0]    p_data;
  logic [N-1:0]         p_be;
  gos_pool #(.TW(ACC_W + DTW)) u_pool (
    .clk, .rst(crst), .in_valid(rq_valid), .in_tok({rq_v[0], rq_tok}), .in_q(rq_q),
    .in_col(rq_dtok.col), .in_dy(rq_dtok.tok.dy), .in_pool_en(rq_dtok.tok.pool_en),
    .in_half(rq_dtok.tok.half), .in_row_mask(rq_dtok.tok.row_mask),
    .out_valid(p_valid), .out_tok(p_tok), .out_data(p_data), .out_be(p_be), .out_we(p_we_unused)
  );
  dtok_t             p_dtok;
  logic [ACC_W-1:0]  p_v0;
  assign p_dtok = p_tok[DTW-1:0];
  assign p_v0   = p_tok[DTW +: ACC_W];

  // ------------------------------------------------------------------ writer
  logic ch_ok;
  assign ch_ok   = p_dtok.tok.ch_valid[p_dtok.col];
  assign wr_be   = p_be & {N{ch_ok & ~p_dtok.tok.out_raw}};
  assign wr_en   = p_valid && (wr_be != '0);
  assign wr_addr = p_dtok.word;
  assign wr_data = p_data;
  assign retire  = p_valid && (p_dtok.col == 3'd7);

  always_ff @(posedge clk) begin
    if (crst) begin
      logit <= '0;
    end else if (state == S_IDLE && start) begin
      logit <= '0;
    end else if (p_valid && p_dtok.tok.out_raw && ch_ok) begin
      logit[{p_dtok.tok.oc_tile[0], p_dtok.col}] <= p_v0;   // oc = oc_tile*8 + col (OC <= 16)
    end
  end

  // sticky error flags
  always_ff @(posedge clk) begin
    if (crst) err_flags <= '0;
    else begin
      err_flags[0] <= err_flags[0] | viol0;
      err_flags[1] <= err_flags[1] | viol1;
      err_flags[4] <= err_flags[4] | overrun;
    end
  end

`ifndef SYNTHESIS
  // ACT and WGT reads are issued together with equal documented latency (L_MEM_ACC).
  always @(posedge clk) if (!crst && (a_valid !== w_valid))
    $error("gos_core: ACT/WGT read valid misaligned");
  always @(posedge clk) if (!crst && a0_valid && a1_valid)
    $error("gos_core: both ACT buffers read");
`endif
endmodule
