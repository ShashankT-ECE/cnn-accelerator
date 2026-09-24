`timescale 1ns/1ps
// gos_requant — 8-lane requantization (ARCH_SPEC "Numeric contract" / "Requant",
// FORMATS.md QPARAM, DECISIONS D1 outcome B = 32, OC-2 s in 1..63, D11-4 ReLU after clip).
//
// One output channel's (q_bias, m, s) per cycle, shared by the N lanes (drain column).
// Per lane:
//   v     = acc + q_bias                      (32-bit; wrap-free by host proof)
//   v_raw = v                                 (independent of relu_en / out_raw)
//   p     = v[25:0] (signed, V_MUL_W) * m (unsigned 32-bit)   exact, 58-bit signed
//           (|v| < 2^25 for every reachable v: gos_pack.v_mul_margin, Phase-0 proof)
//   q     = (p + 2^(s-1) - 1 + ((p >> s) & 1)) >>> s          RNE, s in 1..63
//   q     = clip(q, -128, 127);  if relu_en: q = max(q, 0)     (ReLU after the clip)
//
// Rounding is implemented without a wide adder, with k = s - 1:
//   half = p[k], t = p >>> s, t0 = t[0], sticky = |p[k-1:0]
//   q    = t + (half & (sticky | t0))          (identical to the formula above)
// and clip uses a range flag: t in [-128, 127] iff p[57 : s+7] all equal the sign bit.
//
// s = 0 (final layer, m = 0, out_raw = 1): q_int8 is forced to 0 (the RNE formula is
// undefined for s = 0). Irrelevant in operation: the out_raw path uses v_raw.
// out_raw is carried as a passthrough (out_raw_d), aligned with the data; the downstream
// mux selects v_raw vs q_int8. Masked lanes/channels are handled at integration level.
//
// Latency L_RQ = 6 (out_valid/out_tok = in_valid/in_tok delayed by exactly 6):
//   S1  v = acc + q_bias (fabric CARRY8)  -> DSP AREG ; m -> BREG ; k = s-1, s==0 flag
//   S2  multiply (MREG) ; shift masks from k
//   S3  PREG (low DSP of the 2-DSP pair)
//   S4  PREG (high DSP, cascade PCIN>>17) + low-17-bit alignment register -> p
//   S5  w = (p >>> k)[8:0] (64:1 mux per bit), sticky = |(p & lomask),
//       in_range = ~|((p ^ sign) & himask)
//   S6  round increment, clip, ReLU, s==0 mask -> q_int8 ; v_raw
// Multiply: 26 x 33 signed ({1'b0, m}) inferred; Vivado decomposes it into 2 DSP48E2
// per lane (m split into a 17-bit low part and the high part, combined via the
// PCIN >> 17 cascade), with AREG/BREG, MREG and PREG packed from S1..S4.
module gos_requant
  import gos_pkg::*;
#(
  parameter int TW = GOS_TOK_W
) (
  input  logic                         clk,
  input  logic                         rst,
  input  logic                         in_valid,
  input  logic [TW-1:0]                in_tok,
  input  logic [N-1:0][ACC_W-1:0]      acc,        // INT32 per lane (two's complement)
  input  logic [ACC_W-1:0]             q_bias,     // INT32, shared
  input  logic [M_W-1:0]               m,          // UINT32, shared
  input  logic [S_W-1:0]               s,          // UINT6, shared (0 only with out_raw)
  input  logic                         relu_en,
  input  logic                         out_raw,
  output logic                         out_valid,
  output logic [TW-1:0]                out_tok,
  output logic [N-1:0][7:0]            q_int8,     // INT8 per lane
  output logic [N-1:0][ACC_W-1:0]      v_raw,      // INT32 v = acc + q_bias per lane
  output logic                         out_raw_d   // out_raw delayed by L_RQ
);

  localparam int L_RQ = 6;
  localparam int P_W  = 58;                  // |v| < 2^25, m < 2^32 -> |p| < 2^57
  localparam int MP_W = V_MUL_W + M_W + 1;   // 59-bit signed product of 26 x 33

  // ---------------------------------------------------------------- control pipe
  logic [L_RQ-1:0]         vld_q;
  logic [TW-1:0]           tok_q  [L_RQ];
  logic [L_RQ-1:0]         relu_q;
  logic [L_RQ-1:0]         raw_q;
  logic [L_RQ-1:0]         sz_q;             // s == 0

  always_ff @(posedge clk) begin
    if (rst) vld_q <= '0;
    else     vld_q <= {vld_q[L_RQ-2:0], in_valid};
  end

  always_ff @(posedge clk) begin
    tok_q[0] <= in_tok;
    for (int i = 1; i < L_RQ; i++) tok_q[i] <= tok_q[i-1];
    relu_q <= {relu_q[L_RQ-2:0], relu_en};
    raw_q  <= {raw_q[L_RQ-2:0],  out_raw};
    sz_q   <= {sz_q[L_RQ-2:0],   (s == '0)};
  end

  assign out_valid = vld_q[L_RQ-1];
  assign out_tok   = tok_q[L_RQ-1];
  assign out_raw_d = raw_q[L_RQ-1];

  // ---------------------------------------------------------------- S1: v, m, k
  logic [N-1:0][ACC_W-1:0] v1;
  logic [M_W-1:0]          m1;
  logic [S_W-1:0]          k1;

  always_ff @(posedge clk) begin
    for (int l = 0; l < N; l++) v1[l] <= acc[l] + q_bias;   // 32-bit modular add
    m1 <= m;
    k1 <= s - S_W'(1);                                       // s = 0 -> 63 (masked in S6)
  end

  // ---------------------------------------------------------------- S2: multiply, masks
  logic signed [MP_W-1:0]  mp2 [N];
  logic [N-1:0][ACC_W-1:0] v2;
  logic [S_W-1:0]          k2;
  logic [P_W-1:0]          lomask2, himask2;
  logic [P_W-1:0]          lomask_c, himask_c;

  always_comb begin
    for (int i = 0; i < P_W; i++) begin
      lomask_c[i] = (7'(i) <  7'(k1));              // bits below p[k]        (sticky)
      himask_c[i] = (7'(i) >= (7'(k1) + 7'd8));     // bits at/above p[s+7]   (range)
    end
  end

  always_ff @(posedge clk) begin
    for (int l = 0; l < N; l++)
      mp2[l] <= $signed(v1[l][V_MUL_W-1:0]) * $signed({1'b0, m1});
    v2      <= v1;
    k2      <= k1;
    lomask2 <= lomask_c;
    himask2 <= himask_c;
  end

  // ---------------------------------------------------------------- S3, S4: DSP P regs
  logic signed [MP_W-1:0]  mp3 [N];
  logic signed [MP_W-1:0]  mp4 [N];
  logic [N-1:0][ACC_W-1:0] v3, v4;
  logic [S_W-1:0]          k3, k4;
  logic [P_W-1:0]          lomask3, himask3, lomask4, himask4;

  always_ff @(posedge clk) begin
    for (int l = 0; l < N; l++) begin
      mp3[l] <= mp2[l];
      mp4[l] <= mp3[l];
    end
    v3 <= v2;  v4 <= v3;
    k3 <= k2;  k4 <= k3;
    lomask3 <= lomask2;  lomask4 <= lomask3;
    himask3 <= himask2;  himask4 <= himask3;
  end

  // ---------------------------------------------------------------- S5: shift, sticky, range
  logic [N-1:0][8:0]       w5;         // (p >>> k)[8:0]: w[0] = half, w[8:1] = t[7:0]
  logic [N-1:0]            sticky5, inr5, sign5;
  logic [N-1:0][ACC_W-1:0] v5;

  always_ff @(posedge clk) begin
    for (int l = 0; l < N; l++) begin
      logic signed [P_W-1:0] p;
      logic signed [P_W-1:0] sh;
      p          = mp4[l][P_W-1:0];
      sh         = p >>> k4;
      w5[l]      <= sh[8:0];
      sticky5[l] <= |(p & lomask4);
      inr5[l]    <= ~|((p ^ {P_W{p[P_W-1]}}) & himask4);
      sign5[l]   <= p[P_W-1];
    end
    v5 <= v4;
  end

  // ---------------------------------------------------------------- S6: round, clip, ReLU
  always_ff @(posedge clk) begin
    for (int l = 0; l < N; l++) begin
      logic       rnd;
      logic [7:0] t8, q8;
      t8  = w5[l][8:1];
      rnd = w5[l][0] & (sticky5[l] | w5[l][1]);
      if (!inr5[l])                    q8 = sign5[l] ? 8'h80 : 8'h7F;   // saturate
      else if (rnd && t8 == 8'h7F)     q8 = 8'h7F;                      // 127 + 1 -> 127
      else                             q8 = t8 + {7'd0, rnd};
      if (sz_q[4])                     q8 = 8'h00;                      // s == 0
      if (relu_q[4] && q8[7])          q8 = 8'h00;                      // ReLU after clip
      q_int8[l] <= q8;
    end
    v_raw <= v5;
  end

endmodule
