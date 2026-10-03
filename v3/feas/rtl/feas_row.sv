`timescale 1ns/1ps
// feas_row — EXPLORATORY (V3 Phase 0, WS5). Not design RTL; never reused unreviewed (v3/CLAUDE.md).
//
// One output-stationary PE row of W columns. One INT8 activation `a` is broadcast to all W columns;
// each column c has its own INT8 weight w[c]. Per column: acc[c] = sum_k a_k * w_k[c] (INT32).
// Purpose: OOC timing/resources of the row + broadcast at W = 16 / 32, with and without INT8 DSP packing.
//
// Broadcast stage (as V2 gos_array): a is registered into W/8 replica registers (keep), each feeding
// 8 columns; w[c] registered once per column (keep) — so fanout per broadcast register is <= 8.
//
// PACK = 0: one DSP48E2 per column, multiply-accumulate inside the DSP (V2 gos_pe UG901 template):
//           A/B reg -> MREG -> PREG accumulator.
// PACK = 1: one DSP48E2 per column PAIR (2j, 2j+1), sharing the activation (AMD WP486 style):
//           ad = (w[2j+1] <<< 18) + w[2j]      (pre-adder, AD register)
//           p  = ad * a                         (MREG, PREG; no accumulation in the DSP)
//           lo = signed p[17:0]                 = a * w[2j]       (|a*w| <= 2^14 < 2^17: exact)
//           hi = (p >>> 18) + p[17]             = a * w[2j+1]     (borrow correction)
//           two INT32 accumulators per pair in fabric, updated every cycle.
//           Accumulating inside the DSP is NOT used: the 18-bit low field overflows after
//           floor((2^17-1)/2^14) = 7 products (feas_packing.csv), far below K of the target nets.
//
// Control (valid/first/last) enters with a/w and is delayed to the accumulator stage. On `last`
// the W accumulators are copied into a shadow register chain that is shifted out one column per
// cycle on `drain_out` (drain_valid marks the W valid cycles). A new `first` may arrive while the
// shadow is draining only if the previous drain is done (the testbench guarantees K >= W).
module feas_row #(
    parameter int W     = 16,      // columns (even)
    parameter int PACK  = 0,       // 0: 1 MAC/DSP, 1: 2 MACs/DSP
    parameter int ACC_W = 32
) (
    input  logic                    clk,
    input  logic                    rst,
    input  logic signed [7:0]       a,
    input  logic [W-1:0][7:0]       w,
    input  logic                    in_valid,
    input  logic                    in_first,
    input  logic                    in_last,
    output logic                    drain_valid,
    output logic signed [ACC_W-1:0] drain_out
);
    localparam int G   = (W + 7) / 8;                 // activation replicas
    // input -> accumulator-update stage (number of register stages before the acc edge)
    localparam int L_ACC = (PACK == 0) ? 3 : 5;       // PACK0: bcast, A/B, M | PACK1: bcast, A/D/B, AD, M, P

    // ---------------------------------------------------------------- broadcast stage
    (* keep = "true" *) logic signed [7:0] a_b [G];
    (* keep = "true" *) logic [W-1:0][7:0] w_b;
    always_ff @(posedge clk) begin
        for (int g = 0; g < G; g++) a_b[g] <= a;
        w_b <= w;
    end

    // ---------------------------------------------------------------- control delay
    logic [L_ACC:0] vld_d, fst_d, lst_d;
    always_ff @(posedge clk) begin
        if (rst) begin
            vld_d <= '0;
            lst_d <= '0;
        end else begin
            vld_d <= {vld_d[L_ACC-1:0], in_valid};
            lst_d <= {lst_d[L_ACC-1:0], in_valid & in_last};
        end
        fst_d <= {fst_d[L_ACC-1:0], in_first};
    end
    logic ce, load, cap;
    always_comb begin
        ce   = vld_d[L_ACC-1];          // aligned with the product register feeding the accumulator
        load = fst_d[L_ACC-1];
        cap  = lst_d[L_ACC];            // one cycle after the last accumulate
    end

    logic signed [ACC_W-1:0] acc [W];

    generate
        if (PACK == 0) begin : g_unpacked
            for (genvar c = 0; c < W; c++) begin : g_pe
                feas_pe_mac #(.ACC_W(ACC_W)) u_pe (
                    .clk(clk), .a(a_b[c / 8]), .w($signed(w_b[c])), .ce(ce), .load(load), .acc(acc[c]));
            end
        end else begin : g_packed
            for (genvar j = 0; j < W / 2; j++) begin : g_pair
                logic signed [7:0]  a_r, a_r2, wl_r, wh_r;
                logic signed [26:0] ad_r;
                logic signed [34:0] m_r, p_r;
                logic signed [17:0] lo;
                logic signed [16:0] hi;
                logic signed [ACC_W-1:0] acc_lo, acc_hi;
                always_ff @(posedge clk) begin
                    a_r  <= a_b[(2 * j) / 8];
                    wl_r <= $signed(w_b[2 * j]);
                    wh_r <= $signed(w_b[2 * j + 1]);
                    a_r2 <= a_r;
                    ad_r <= (27'(wh_r) <<< 18) + 27'(wl_r);
                    m_r  <= ad_r * a_r2;
                    p_r  <= m_r;
                end
                always_comb begin
                    lo = p_r[17:0];
                    hi = p_r[34:18] + 17'(p_r[17]);
                end
                always_ff @(posedge clk) begin
                    if (ce) begin
                        acc_lo <= (load ? '0 : acc_lo) + ACC_W'(lo);
                        acc_hi <= (load ? '0 : acc_hi) + ACC_W'(hi);
                    end
                end
                assign acc[2 * j]     = acc_lo;
                assign acc[2 * j + 1] = acc_hi;
            end
        end
    endgenerate

    // ---------------------------------------------------------------- shadow drain
    logic signed [ACC_W-1:0] shadow [W];
    logic [$clog2(W+1)-1:0]  dcnt;
    always_ff @(posedge clk) begin
        if (cap) begin
            for (int c = 0; c < W; c++) shadow[c] <= acc[c];
        end else begin
            for (int c = 0; c < W - 1; c++) shadow[c] <= shadow[c + 1];
        end
    end
    always_ff @(posedge clk) begin
        if (rst)        dcnt <= '0;
        else if (cap)   dcnt <= ($clog2(W+1))'(W);
        else if (dcnt != '0) dcnt <= dcnt - 1'b1;
    end
    always_comb begin
        drain_valid = (dcnt != '0);
        drain_out   = shadow[0];
    end
endmodule

// feas_pe_mac — copy of v2/rtl/gos_pe.sv @ 28dd2ad (UG901 MAC-with-load template, one DSP48E2,
// AREG/BREG = 1, MREG = 1, PREG = 1). ce/load are aligned with m_r (2 cycles after a/w).
(* use_dsp = "yes" *)
module feas_pe_mac #(
    parameter int ACC_W = 32
) (
    input  logic                    clk,
    input  logic signed [7:0]       a,
    input  logic signed [7:0]       w,
    input  logic                    ce,
    input  logic                    load,
    output logic signed [ACC_W-1:0] acc
);
    logic signed [7:0]       a_r, w_r;
    logic signed [15:0]      m_r;
    logic signed [ACC_W-1:0] base, m_ext;
    always_ff @(posedge clk) begin
        a_r <= a;
        w_r <= w;
        m_r <= a_r * w_r;
    end
    always_comb m_ext = {{(ACC_W-16){m_r[15]}}, m_r};
    always_comb base  = load ? '0 : acc;
    always_ff @(posedge clk) begin
        if (ce) acc <= base + m_ext;
    end
endmodule
