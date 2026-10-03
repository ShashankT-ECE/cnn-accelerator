`timescale 1ns/1ps
// feas_mem_path — EXPLORATORY (V3 Phase 0, WS5, user addition 2026-10-03). Not design RTL.
//
// Timing/resource probe of the memory side of a W-wide array, at 4.000 ns and 3.000 ns:
//   (1) weight bank in URAM: W x INT8 per word (one OC tile per k), 4096 words, read -> 2 pipeline
//       registers -> REP broadcast replica registers (keep) per weight byte -> w_out.
//   (2) banked ACT read path: W banks (bank = x mod W) of INT8 x 4096 in BRAM, per-bank address
//       word + (b < sh), registered BRAM output, a log2(W)-level rotator (registered halfway and at the
//       end), REP broadcast replica registers per activation byte -> a_out.
//   (3) residual buffer: W x INT8 per word in BRAM, read aligned with W accumulators entering requant:
//       skip term in the accumulator domain (candidate residual option R2, an OPEN decision):
//         sk  = skip * m_skip (INT8 x INT16), skr = (sk + 2^(s_skip-1)) >>> s_skip (s_skip 1..15),
//         v   = acc + q_bias + (res_en ? skr : 0)
//       then the V2 requant lane pipeline (copied: feas_requant) -> q_out (INT8, RNE, clip, ReLU).
// Each stream carries its own valid; no consumer counts cycles. Weight/act/res memories have write
// ports for loading (the TB preloads them).
module feas_mem_path #(
    parameter int W     = 16,          // lanes (array width); power of two
    parameter int REP   = 2,           // broadcast replicas per byte (fanout per replica = sinks / REP)
    parameter int DEPTH = 4096
) (
    input  logic                     clk,
    input  logic                     rst,
    // weight URAM
    input  logic                     wgt_we,
    input  logic [11:0]              wgt_waddr,
    input  logic [W*8-1:0]           wgt_wdata,
    input  logic                     wgt_re,
    input  logic [11:0]              wgt_raddr,
    output logic                     w_valid,
    output logic [REP-1:0][W*8-1:0]  w_out,
    // ACT banks
    input  logic                     act_we,
    input  logic [11:0]              act_waddr,
    input  logic [W*8-1:0]           act_wdata,      // byte b -> bank b
    input  logic                     act_re,
    input  logic [11:0]              act_word,       // word holding output position x0
    input  logic [$clog2(W)-1:0]     act_sh,         // x0 mod W
    output logic                     a_valid,
    output logic [REP-1:0][W*8-1:0]  a_out,          // lane r = activation at x0 + r
    // residual buffer + requant
    input  logic                     res_we,
    input  logic [11:0]              res_waddr,
    input  logic [W*8-1:0]           res_wdata,
    input  logic                     rq_valid,
    input  logic [11:0]              res_raddr,
    input  logic [W-1:0][31:0]       acc,
    input  logic [31:0]              q_bias,
    input  logic [31:0]              m,
    input  logic [5:0]               s,
    input  logic signed [15:0]       m_skip,
    input  logic [3:0]               s_skip,
    input  logic                     res_en,
    input  logic                     relu_en,
    output logic                     q_valid,
    output logic [W-1:0][7:0]        q_out
);
    localparam int SHW = $clog2(W);

    // ================================================================ (1) weight URAM
    (* ram_style = "ultra" *) logic [W*8-1:0] wmem [DEPTH];
    logic [W*8-1:0] wr0, wr1;
    logic [2:0]     wv;
    always_ff @(posedge clk) begin
        if (wgt_we) wmem[wgt_waddr] <= wgt_wdata;
        if (wgt_re) wr0 <= wmem[wgt_raddr];
        wr1 <= wr0;
    end
    (* keep = "true" *) logic [REP-1:0][W*8-1:0] w_rep;
    always_ff @(posedge clk) begin
        for (int r = 0; r < REP; r++) w_rep[r] <= wr1;
        if (rst) wv <= '0;
        else     wv <= {wv[1:0], wgt_re};
    end
    assign w_out   = w_rep;
    assign w_valid = wv[2];

    // ================================================================ (2) banked ACT read
    logic [W-1:0][11:0] baddr;
    logic [SHW-1:0]     sh0, sh1, sh2;
    logic [W-1:0][7:0]  bq0, bq1;
    logic [5:0]         av;
    always_ff @(posedge clk) begin
        for (int b = 0; b < W; b++) baddr[b] <= act_word + 12'((b < int'(act_sh)) ? 1 : 0);
        sh0 <= act_sh;
        sh1 <= sh0;
        sh2 <= sh1;
    end
    for (genvar b = 0; b < W; b++) begin : g_bank
        (* ram_style = "block" *) logic [7:0] amem [DEPTH];
        logic [7:0] q0;
        always_ff @(posedge clk) begin
            if (act_we) amem[act_waddr] <= act_wdata[b*8 +: 8];
            q0 <= amem[baddr[b]];
        end
        always_ff @(posedge clk) bq1[b] <= q0;      // BRAM output register
        assign bq0[b] = q0;
    end
    // rotator: lane r <- bank (sh + r) mod W ; log2(W) levels, registered after level SHW/2 and at the end
    localparam int HALF = SHW / 2;
    logic [W-1:0][7:0] rot_mid, rot_out;
    logic [SHW-1:0]    sh_mid;
    always_ff @(posedge clk) begin
        logic [W-1:0][7:0] t;
        t = bq1;
        for (int l = 0; l < HALF; l++) begin
            logic [W-1:0][7:0] u;
            for (int r = 0; r < W; r++) u[r] = sh2[l] ? t[(r + (1 << l)) % W] : t[r];
            t = u;
        end
        rot_mid <= t;
        sh_mid  <= sh2;
    end
    always_ff @(posedge clk) begin
        logic [W-1:0][7:0] t;
        t = rot_mid;
        for (int l = HALF; l < SHW; l++) begin
            logic [W-1:0][7:0] u;
            for (int r = 0; r < W; r++) u[r] = sh_mid[l] ? t[(r + (1 << l)) % W] : t[r];
            t = u;
        end
        rot_out <= t;
    end
    (* keep = "true" *) logic [REP-1:0][W*8-1:0] a_rep;
    always_ff @(posedge clk) begin
        for (int r = 0; r < REP; r++) a_rep[r] <= rot_out;
        if (rst) av <= '0;
        else     av <= {av[4:0], act_re};
    end
    assign a_out   = a_rep;
    assign a_valid = av[5];
    // act_re latency 6: baddr (1) -> q0 (2) -> bq1 (3) -> rot_mid (4) -> rot_out (5) -> a_rep (6)

    // ================================================================ (3) residual read + requant
    logic [W*8-1:0] rmem_q0, rmem_q1;
    (* ram_style = "block" *) logic [W*8-1:0] rmem [DEPTH];
    always_ff @(posedge clk) begin
        if (res_we) rmem[res_waddr] <= res_wdata;
        rmem_q0 <= rmem[res_raddr];
        rmem_q1 <= rmem_q0;
    end
    // align acc / params with the 2-cycle residual read
    logic [W-1:0][31:0] acc_d0, acc_d1;
    logic [31:0]        qb_d0, qb_d1, m_d0, m_d1, m_d2, m_d3, m_d4;
    logic [5:0]         s_d0, s_d1, s_d2, s_d3, s_d4;
    logic signed [15:0] ms_d0, ms_d1;
    logic [3:0]         ss_d0, ss_d1, ss_d2;
    logic [3:0]         re_d, rl_d;
    logic [3:0]         vld_d;
    always_ff @(posedge clk) begin
        acc_d0 <= acc;     acc_d1 <= acc_d0;
        qb_d0  <= q_bias;  qb_d1  <= qb_d0;
        m_d0   <= m;       m_d1   <= m_d0;   m_d2 <= m_d1;   m_d3 <= m_d2;  m_d4 <= m_d3;
        s_d0   <= s;       s_d1   <= s_d0;   s_d2 <= s_d1;   s_d3 <= s_d2;  s_d4 <= s_d3;
        ms_d0  <= m_skip;  ms_d1  <= ms_d0;
        ss_d0  <= s_skip;  ss_d1  <= ss_d0;  ss_d2 <= ss_d1;
        re_d   <= {re_d[2:0], res_en};
        rl_d   <= {rl_d[2:0], relu_en};
        if (rst) vld_d <= '0;
        else     vld_d <= {vld_d[2:0], rq_valid};
    end
    // R1: skip * m_skip (DSP), acc + q_bias
    logic signed [23:0] sk1 [W];
    logic [W-1:0][31:0] vb1, vb2;
    always_ff @(posedge clk) begin
        for (int l = 0; l < W; l++) begin
            sk1[l] <= $signed(rmem_q1[l*8 +: 8]) * ms_d1;
            vb1[l] <= acc_d1[l] + qb_d1;
        end
    end
    // R2: rounding shift of the skip term
    logic signed [23:0] skr2 [W];
    always_ff @(posedge clk) begin
        for (int l = 0; l < W; l++) begin
            logic signed [24:0] t;
            t = 25'(sk1[l]) + ((ss_d2 == 4'd0) ? 25'sd0 : (25'sd1 <<< (ss_d2 - 4'd1)));
            skr2[l] <= 24'(t >>> ss_d2);
        end
        vb2 <= vb1;
    end
    // R3: v = acc + q_bias + skip term
    logic [W-1:0][31:0] v3;
    always_ff @(posedge clk) begin
        for (int l = 0; l < W; l++) v3[l] <= vb2[l] + (re_d[3] ? 32'(skr2[l]) : 32'd0);
    end
    logic vld3;
    logic rl3;
    always_ff @(posedge clk) begin
        if (rst) vld3 <= 1'b0;
        else     vld3 <= vld_d[3];
        rl3 <= rl_d[3];
    end
    // requant (copied V2 lane pipeline), q_bias already added -> pass 0
    feas_requant #(.N(W)) u_rq (
        .clk(clk), .rst(rst), .in_valid(vld3), .v_in(v3), .m(m_d4), .s(s_d4), .relu_en(rl3),
        .out_valid(q_valid), .q_int8(q_out));
endmodule


// feas_requant — copy of the lane pipeline of v2/rtl/gos_requant.sv @ 28dd2ad (S1..S6), with the token,
// out_raw and v_raw paths removed and q_bias pre-added by the caller (v_in = acc + q_bias + skip).
// q = clip(RNE(v[25:0] * m / 2^s)), ReLU after the clip; s in 1..63 (s = 0 forces 0). Latency 6.
module feas_requant #(
    parameter int N = 16
) (
    input  logic                 clk,
    input  logic                 rst,
    input  logic                 in_valid,
    input  logic [N-1:0][31:0]   v_in,
    input  logic [31:0]          m,
    input  logic [5:0]           s,
    input  logic                 relu_en,
    output logic                 out_valid,
    output logic [N-1:0][7:0]    q_int8
);
    localparam int L_RQ = 6;
    localparam int P_W  = 58;
    localparam int MP_W = 26 + 32 + 1;
    logic [L_RQ-1:0] vld_q, relu_q, sz_q;
    always_ff @(posedge clk) begin
        if (rst) vld_q <= '0;
        else     vld_q <= {vld_q[L_RQ-2:0], in_valid};
        relu_q <= {relu_q[L_RQ-2:0], relu_en};
        sz_q   <= {sz_q[L_RQ-2:0], (s == 6'd0)};
    end
    assign out_valid = vld_q[L_RQ-1];

    logic [5:0] k1;
    logic [31:0] m1;
    logic [N-1:0][31:0] v1;
    always_ff @(posedge clk) begin
        v1 <= v_in;
        m1 <= m;
        k1 <= s - 6'd1;
    end
    logic signed [MP_W-1:0] mp2 [N];
    logic [5:0]             k2;
    logic [P_W-1:0]         lomask2, himask2, lomask_c, himask_c;
    always_comb begin
        for (int i = 0; i < P_W; i++) begin
            lomask_c[i] = (7'(i) <  7'(k1));
            himask_c[i] = (7'(i) >= (7'(k1) + 7'd8));
        end
    end
    always_ff @(posedge clk) begin
        for (int l = 0; l < N; l++) mp2[l] <= $signed(v1[l][25:0]) * $signed({1'b0, m1});
        k2 <= k1; lomask2 <= lomask_c; himask2 <= himask_c;
    end
    logic signed [MP_W-1:0] mp3 [N];
    logic signed [MP_W-1:0] mp4 [N];
    logic [5:0]             k3, k4;
    logic [P_W-1:0]         lomask3, himask3, lomask4, himask4;
    always_ff @(posedge clk) begin
        for (int l = 0; l < N; l++) begin
            mp3[l] <= mp2[l];
            mp4[l] <= mp3[l];
        end
        k3 <= k2; lomask3 <= lomask2; himask3 <= himask2;
        k4 <= k3; lomask4 <= lomask3; himask4 <= himask3;
    end
    logic [N-1:0][8:0] w4;
    logic [N-1:0]      sticky4, inr4, sign4;
    always_ff @(posedge clk) begin
        for (int l = 0; l < N; l++) begin
            logic signed [P_W-1:0] p, sh;
            p  = mp4[l][P_W-1:0];
            sh = p >>> k4;
            w4[l]      <= sh[8:0];
            sticky4[l] <= |(p & lomask4);
            inr4[l]    <= ~|((p ^ {P_W{p[P_W-1]}}) & himask4);
            sign4[l]   <= p[P_W-1];
        end
    end
    always_ff @(posedge clk) begin
        for (int l = 0; l < N; l++) begin
            logic       rnd;
            logic [7:0] t8, q8;
            t8  = w4[l][8:1];
            rnd = w4[l][0] & (sticky4[l] | w4[l][1]);
            if (!inr4[l])                q8 = sign4[l] ? 8'h80 : 8'h7F;
            else if (rnd && t8 == 8'h7F) q8 = 8'h7F;
            else                         q8 = t8 + {7'd0, rnd};
            if (sz_q[4])                 q8 = 8'h00;
            if (relu_q[4] && q8[7])      q8 = 8'h00;
            q_int8[l] <= q8;
        end
    end
endmodule
