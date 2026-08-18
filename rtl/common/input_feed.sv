//==============================================================================
// input_feed.sv — V1 Input-Feed / Controller
//
// Orchestrates the frozen 8x8 systolic array (rtl/common/systolic_array.sv)
// through the finalized row-decomposed 5-pass 2-D convolution schedule
// (docs/specs/INPUT_FEED_SPEC.md, Decision 9). It turns the array's verified
// 1-D 5-tap correlation into the true 5x5 convolution for LeNet-5 Conv1
// (1x28x28 -> 6x24x24, valid, no padding).
//
// This module is the *controller*: it generates the per-cycle activation and
// weight streams, the array control signals, and captures the drained results.
// It does NOT instantiate the array and does NOT pick a storage technology —
// the image and weight data arrive via simple combinational read ports (direct
// external addressing, INPUT_FEED_SPEC.md §10/§11 "OPEN DECISION" left open),
// and captured results are exposed on a registered strobe port.
//
// Canonical (non-overlapped) group schedule — 82 cycles/group (§7):
//    1   accum_clear           (before pass 0 only)
//    65  5 passes x 13         (each pass = 7 lead-in + 5 taps + 1 drain)
//    16  column-sequential     result_req drain (8 columns x 2 cycles)
//    ---
//    82  total per group
//
// Result drain protocol (§7): result_out is registered (1-cycle capture) and
// the array's result_out bus is a combinational one-hot mux on the live
// result_req, so each column is asserted for 2 cycles (latch on cycle 1, read
// on cycle 2 while still asserted) — 16 cycles total, NOT 8.
//
// The proven-safe 78-cycle overlap (§6.3) is NOT implemented (CANDIDATE only).
// pe.sv and systolic_array.sv are unchanged.
//
// Authoritative contract: docs/specs/INPUT_FEED_SPEC.md (V1).
//==============================================================================

module input_feed (
    input  logic                clk,
    input  logic                rst,             // synchronous, active-high

    // Image memory interface (combinational read, direct addressing)
    output logic [9:0]          img_addr,        // 0..783 = row*28 + col
    input  logic signed [7:0]   img_data,        // input[row][col]

    // Weight memory interface (combinational read: all 6 channels per tap)
    output logic [4:0]          wgt_addr,        // 0..24 = ky*5 + kx
    input  logic signed [7:0]   wgt_data [0:5],  // weight[ch][ky][kx], ch=0..5

    // Array control (drives systolic_array.sv)
    output logic signed [7:0]   act_in      [0:7], // per-row activation (all identical)
    output logic signed [7:0]   w_in        [0:7], // per-row weight (rows 6..7 = 0)
    output logic                weight_load,       // asserted every lead-in/tap cycle
    output logic                accum_clear,       // once per group, before pass 0
    output logic                zero_skip,         // tied 0 in V1
    output logic                result_req  [0:7], // one-hot, 2 cycles per column

    // Array data (from systolic_array.sv)
    input  logic signed [31:0]  result_out  [0:7], // selected column's 8 results

    // Captured results (registered; rows 0..5 valid, 6..7 idle)
    output logic                result_valid,      // 1-cycle pulse per column capture
    output logic [4:0]          result_y,          // output row (0..23)
    output logic [4:0]          result_x,          // output column (0..23)
    output logic signed [31:0]  result_data [0:7]  // captured result_out snapshot
);

    //--------------------------------------------------------------------------
    // Fixed V1 geometry (INPUT_FEED_SPEC.md §2)
    //--------------------------------------------------------------------------
    localparam int IMG_H  = 28;   // input height
    localparam int IMG_W  = 28;   // input width
    localparam int OC     = 6;    // output channels
    localparam int K      = 5;    // kernel size
    localparam int GROUPS = 3;    // groups per output row (24 = 3 x 8)

    //--------------------------------------------------------------------------
    // FSM phase encoding
    //--------------------------------------------------------------------------
    localparam logic [1:0] PH_CLEAR = 2'd0;   // 1 cycle: assert accum_clear
    localparam logic [1:0] PH_PASS  = 2'd1;   // 13 cycles: lead-in + taps + drain
    localparam logic [1:0] PH_DRAIN = 2'd2;   // 16 cycles: result_req drain
    localparam logic [1:0] PH_IDLE  = 2'd3;   // frame complete; all outputs quiet

    //--------------------------------------------------------------------------
    // FSM state
    //--------------------------------------------------------------------------
    logic [1:0] phase;   // current phase
    logic [4:0] y;       // output row  (0..23)
    logic [1:0] g;       // group        (0..2)
    logic [2:0] ky;      // kernel-row pass (0..4)
    logic [3:0] pc;      // pass cycle   (0..12)
    logic [3:0] dc;      // drain cycle  (0..15)

    //--------------------------------------------------------------------------
    // Derived controls
    //--------------------------------------------------------------------------
    logic in_tap;     // PASS && pc<12 : present an activation (and weight-load)
    logic in_weight;  // PASS && pc in [6,10] : weight stream valid
    logic capture;    // DRAIN && dc odd : read result_out (2nd cycle of a column)

    assign in_tap    = (phase == PH_PASS)  && (pc < 4'd12);
    assign in_weight = (phase == PH_PASS)  && (pc >= 4'd6) && (pc <= 4'd10);
    assign capture   = (phase == PH_DRAIN) && dc[0];

    //--------------------------------------------------------------------------
    // FSM (self-timed; 72 groups = 24 rows x 3 groups, then idle)
    //--------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            phase <= PH_CLEAR;
            y     <= '0;
            g     <= '0;
            ky    <= '0;
            pc    <= '0;
            dc    <= '0;
        end else begin
            case (phase)
                PH_CLEAR: begin
                    // 1-cycle accum_clear, then pass 0
                    phase <= PH_PASS;
                    ky    <= '0;
                    pc    <= '0;
                    // y, g unchanged
                end

                PH_PASS: begin
                    if (pc < 4'd12)
                        pc <= pc + 4'd1;
                    else if (ky < 3'd4) begin
                        // next kernel-row pass
                        ky <= ky + 3'd1;
                        pc <= '0;
                    end else begin
                        // pass 4 complete -> drain
                        phase <= PH_DRAIN;
                        dc    <= '0;
                    end
                end

                PH_DRAIN: begin
                    if (dc < 4'd15)
                        dc <= dc + 4'd1;
                    else if (g < 2'd2) begin
                        g     <= g + 2'd1;
                        phase <= PH_CLEAR;
                    end else if (y < 5'd23) begin
                        g     <= '0;
                        y     <= y + 5'd1;
                        phase <= PH_CLEAR;
                    end else begin
                        // all 72 groups complete
                        phase <= PH_IDLE;
                    end
                end

                PH_IDLE: begin
                    // hold; all outputs deasserted
                end

                default: phase <= PH_CLEAR;
            endcase
        end
    end

    //--------------------------------------------------------------------------
    // Address generation (combinational, direct addressing)
    //--------------------------------------------------------------------------
    // Pass pc (0..11) presents input pixel (row=y+ky, col=8g+pc):
    //   pc 0..6  -> lead-in input[y+ky][8g .. 8g+6]
    //   pc 7..11 -> taps    input[y+ky][8g+7 .. 8g+11]
    // pc 12 is the drain cycle (act_in = 0, no read).
    always_comb begin
        img_addr = 10'd0;
        if (in_tap)
            img_addr = 10'((y + ky) * IMG_W + (8 * g) + pc);
    end

    // Weight stream: during pc 6..10, present weight[ch][ky][pc-6] (the
    // weight-lead of the next tap); zero otherwise. Tap index = ky*5 + (pc-6).
    always_comb begin
        wgt_addr = 5'd0;
        if (in_weight)
            wgt_addr = 5'(ky * K + (pc - 4'd6));
    end

    //--------------------------------------------------------------------------
    // Array control decode (combinational)
    //--------------------------------------------------------------------------
    // All 8 rows receive the identical (single-channel) activation.
    always_comb begin
        for (int r = 0; r < 8; r++)
            act_in[r] = in_tap ? img_data : 8'sd0;
    end

    // Per-row weight: rows 0..5 are the 6 output channels; rows 6..7 idle (0).
    always_comb begin
        for (int r = 0; r < 8; r++) begin
            if (in_weight && (r < OC))
                w_in[r] = wgt_data[r];
            else
                w_in[r] = 8'sd0;
        end
    end

    // weight_load asserted every lead-in/tap cycle (pc 0..11), deasserted on the
    // pass drain cycle (pc 12) and during accum_clear / result drain.
    assign weight_load = (phase == PH_PASS) && (pc < 4'd12);

    // accum_clear once per group (before pass 0).
    assign accum_clear = (phase == PH_CLEAR);

    // No sparsity in V1.
    assign zero_skip = 1'b0;

    // Column-sequential result drain: one-hot, 2 cycles per column.
    always_comb begin
        for (int c = 0; c < 8; c++)
            result_req[c] = (phase == PH_DRAIN) && ((dc >> 1) == c);
    end

    //--------------------------------------------------------------------------
    // Result capture (registered). On the 2nd cycle of column c (dc odd), the
    // array result_out bus routes column c's freshly-latched values; latch them
    // and pulse result_valid on the following cycle.
    //--------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            result_valid <= 1'b0;
            result_y     <= '0;
            result_x     <= '0;
            for (int r = 0; r < 8; r++)
                result_data[r] <= '0;
        end else begin
            result_valid <= capture;
            if (capture) begin
                result_y <= y;
                result_x <= (8 * g + 4'd7) - (dc >> 1);   // pixel x = base - c
                for (int r = 0; r < 8; r++)
                    result_data[r] <= result_out[r];
            end
        end
    end

endmodule
