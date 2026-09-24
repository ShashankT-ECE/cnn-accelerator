// GOS_OOC_DEPS: gos_pe.sv
`timescale 1ns/1ps
// gos_array — 8x8 output-stationary outer-product MAC array with shadow drain
// (ARCH_SPEC "Datapath": Array, Drain; DECISIONS D8, D10, D11-3).
//
// Rows r = 8 consecutive output x pixels (a_vec[r]); columns c = 8 output channels (w_vec[c]).
// One k = (ic, ky, kx) per valid input cycle; PE(r,c): acc <= first ? a[r]*w[c] : acc + a[r]*w[c]
// (INT32, no clear cycle). One DSP48E2 per PE (gos_pe), 64 in total.
//
// Pipeline (input cycle t = the cycle in_valid/in_* are presented on the ports):
//   edge t+1  broadcast register: a_b, w_b, control stage 1 (valid, first, last, tok)
//   edge t+2  DSP AREG/BREG                                    control stage 2
//   edge t+3  DSP MREG (a*w)                                   control stage 3 (drives PE ce/load)
//   edge t+4  DSP PREG: acc = (first ? 0 : acc) + a*w          control stage 4 (valid & last, tok)
//   edge t+5  if stage-4 valid & last: shadow <= all 64 accs, tok_cap <= tok   ("capture")
//   edge t+6+j (j = 0..7): drain column j -> out_valid/out_col=j/out_acc/out_tok
// So for the in_last cycle t: first drain output (column 0) is visible in cycle t + L_ARRAY,
// column j in cycle t + L_ARRAY + j. Cycles with in_valid = 0 are ignored (acc holds; first/last
// and data are don't-care). No back-pressure (D10).
//
// Overrun: a capture at edge e overwrites the shadow. That is safe iff the previous drain is idle
// or is reading its last column (7) at the same edge, i.e. captures >= L_DRAIN = 8 cycles apart,
// i.e. K >= K_MIN = 8 for back-to-back tiles (zero gap). A capture while column 0..6 is still to be
// read sets the sticky err_overrun (cleared only by rst); the new tile then overwrites the shadow and
// its drain restarts at column 0 (the rest of the old tile is lost).
//
// Reset: only valid bits (control stages, drain active, out_valid) and err_overrun. Data/token
// registers and the DSP datapath have no reset.
module gos_array #(
    parameter int TW = gos_pkg::GOS_TOK_W
) (
    input  logic                                clk,
    input  logic                                rst,
    // input stream: one k per valid cycle
    input  logic                                in_valid,
    input  logic                                in_first,   // first k of a tile
    input  logic                                in_last,    // last k of a tile
    input  logic [TW-1:0]                       in_tok,     // captured at in_last, output with the drain
    input  logic [gos_pkg::N-1:0][7:0]          a_vec,      // a_vec[r] = activation of row r (int8)
    input  logic [gos_pkg::N-1:0][7:0]          w_vec,      // w_vec[c] = weight of column c (int8)
    // drain stream: one column (output channel) per cycle, 8 cycles per tile
    output logic                                out_valid,
    output logic [$clog2(gos_pkg::N)-1:0]       out_col,    // 0..7: column j = output channel j of the tile
    output logic [gos_pkg::N-1:0][gos_pkg::ACC_W-1:0] out_acc, // out_acc[r] = acc[r][out_col] (int32)
    output logic [TW-1:0]                       out_tok,    // captured token, repeated on all 8 drain cycles
    // sticky error
    output logic                                err_overrun
);
    import gos_pkg::*;

    localparam int CW = $clog2(N);

    // ---- latencies (cycles, relative to the input cycle) -------------------------------------
    localparam int L_BCAST      = 1;                        // input -> broadcast register
    localparam int L_PE         = 3;                        // PE: AREG/BREG, MREG, PREG (gos_pe)
    localparam int L_ARRAY_FILL = L_BCAST + L_PE;           // 4: input -> product in the accumulator
    localparam int L_CAPTURE    = L_ARRAY_FILL + 1;         // 5: in_last cycle -> shadow/tok_cap loaded
    localparam int L_ARRAY      = L_CAPTURE + 1;            // 6: in_last cycle -> drain column 0 on out_*
    localparam int L_DRAIN      = N;                        // 8: drain cycles per tile (1 column/cycle)
    localparam int K_MIN        = L_DRAIN;                  // 8: minimum K for zero-gap tiles w/o overrun

    // ---- broadcast register (operands registered once before the fanout of 8) ----------------
    // keep: stays a fabric register (one per operand byte); without it Vivado replicates it into
    // every DSP48E2 as A1/B1 (AREG = BREG = 2), which moves the fanout of 8 in front of the register.
    (* keep = "true" *) logic [N-1:0][7:0] a_b, w_b;
    always_ff @(posedge clk) begin
        a_b <= a_vec;
        w_b <= w_vec;
    end

    // ---- control / token pipeline (stages 1..4) -----------------------------------------------
    logic          v1, v2, v3, v4;
    logic          f1, f2, f3;
    logic          l1, l2, l3, l4;
    logic [TW-1:0] t1, t2, t3, t4;

    always_ff @(posedge clk) begin
        if (rst) begin
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; v4 <= 1'b0;
        end else begin
            v1 <= in_valid; v2 <= v1; v3 <= v2; v4 <= v3;
        end
    end

    always_ff @(posedge clk) begin
        f1 <= in_first; f2 <= f1; f3 <= f2;
        l1 <= in_last;  l2 <= l1; l3 <= l2; l4 <= l3;
        t1 <= in_tok;   t2 <= t1; t3 <= t2; t4 <= t3;
    end

    // ---- 8x8 PE grid ---------------------------------------------------------------------------
    logic [N-1:0][N-1:0][ACC_W-1:0] acc;   // acc[r][c]

    for (genvar r = 0; r < N; r++) begin : g_row
        for (genvar c = 0; c < N; c++) begin : g_col
            logic signed [ACC_W-1:0] acc_rc;
            gos_pe #(.ACC_W(ACC_W)) u_pe (
                .clk  (clk),
                .a    (a_b[r]),
                .w    (w_b[c]),
                .ce   (v3),
                .load (f3),
                .acc  (acc_rc)
            );
            assign acc[r][c] = acc_rc;
        end
    end

    // ---- capture + shadow ---------------------------------------------------------------------
    logic                            cap;       // capture at the next edge
    logic [N-1:0][N-1:0][ACC_W-1:0]  shadow;    // shadow[c][r] (column-major for the drain mux)
    logic [TW-1:0]                   tok_cap;

    always_comb cap = v4 & l4;

    always_ff @(posedge clk) begin
        if (cap) begin
            for (int c = 0; c < N; c++)
                for (int r = 0; r < N; r++)
                    shadow[c][r] <= acc[r][c];
            tok_cap <= t4;
        end
    end

    // ---- drain: one column per cycle, starting the cycle after capture -----------------------
    logic          drn_act;   // shadow holds columns still to be drained
    logic [CW-1:0] drn_col;   // column moved to out_* at the next edge

    always_ff @(posedge clk) begin
        if (rst) begin
            drn_act     <= 1'b0;
            err_overrun <= 1'b0;
        end else begin
            if (cap) begin
                drn_act <= 1'b1;
                if (drn_act && (drn_col != CW'(N - 1))) err_overrun <= 1'b1;
            end else if (drn_act && (drn_col == CW'(N - 1))) begin
                drn_act <= 1'b0;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (cap)          drn_col <= '0;
        else if (drn_act) drn_col <= drn_col + CW'(1);
    end

    // ---- output register --------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) out_valid <= 1'b0;
        else     out_valid <= drn_act;
    end

    always_ff @(posedge clk) begin
        out_col <= drn_col;
        out_acc <= shadow[drn_col];
        out_tok <= tok_cap;
    end
endmodule
