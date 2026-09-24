`timescale 1ns/1ps
// gos_pe — one output-stationary MAC processing element (ARCH_SPEC "Datapath", PE).
//
//   acc <= first ? a*w : acc + a*w          (INT32, no clear cycle)
//
// Written after the UG901 multiply-accumulate-with-load template so that the whole PE packs into ONE
// DSP48E2 with AREG/BREG = 1, MREG = 1, PREG = 1:
//   a_r / w_r  (AREG/BREG)  <- a, w                       every cycle (no enable, no reset)
//   m_r        (MREG)       <- a_r * w_r                  every cycle
//   acc        (PREG)       <- (load ? 0 : acc) + m_r     when ce          (OPMODE W/Z mux + CEP)
// The caller supplies `load` (= first) and `ce` (= valid) already delayed by L_PE_M = 2 cycles, i.e.
// aligned with the product in m_r. No reset on the datapath (keeps it inside the DSP48E2).
//
// Latency: an (a, w) pair presented in cycle t is in acc from cycle t + L_PE (= 3) on.
(* use_dsp = "yes" *)
module gos_pe #(
    parameter int ACC_W = gos_pkg::ACC_W
) (
    input  logic                    clk,
    input  logic signed [7:0]       a,
    input  logic signed [7:0]       w,
    input  logic                    ce,      // valid, aligned with m_r (2 cycles after a/w)
    input  logic                    load,    // first, aligned with m_r (2 cycles after a/w)
    output logic signed [ACC_W-1:0] acc
);
    localparam int L_PE_M = 2;   // a/w input -> m_r
    localparam int L_PE   = 3;   // a/w input -> acc

    logic signed [7:0]       a_r, w_r;
    logic signed [15:0]      m_r;
    logic signed [ACC_W-1:0] base, m_ext;

    always_ff @(posedge clk) begin
        a_r <= a;
        w_r <= w;
        m_r <= a_r * w_r;
    end

    always_comb m_ext = {{(ACC_W-16){m_r[15]}}, m_r};   // explicit sign extension
    always_comb base  = load ? '0 : acc;

    always_ff @(posedge clk) begin
        if (ce) acc <= base + m_ext;
    end
endmodule
