//==============================================================================
// tb_systolic_array.sv — 8x8 Systolic Array (V1) integration testbench
//
// Self-checking, directed, non-UVM testbench for rtl/common/systolic_array.sv
// (which instantiates 64 copies of rtl/common/pe.sv). Verifies the finalized V1
// array contract (docs/specs/SYSTOLIC_ARRAY_SPEC.md §4-§19 and
// docs/PROJECT_STATE.md Decisions 7-8) WITHOUT modifying any RTL or the specs.
//
// Run (Vivado ML 2023.1):
//   source ~/Xilinx/Vivado/2023.1/settings64.sh
//   xvlog -sv rtl/common/pe.sv rtl/common/systolic_array.sv sim/tb_systolic_array.sv
//   xelab -debug typical -s tb_systolic_array_snapshot tb_systolic_array
//   xsim tb_systolic_array_snapshot -runall
//
// Reference model (independent of the DUT):
//   PE(r,c) accumulates the dot product of the activation stream with the
//   weight stream shifted by c:
//       ref(r,c) = SUM_t a_r[t] * w_r[t+c]
//   where a_r[t]=0 outside [0,T-1] (shift registers reset to 0) and
//   w_r[k]=0 outside [0,T-1]. This is the spec's "reverse index" behavior
//   (SYSTOLIC_ARRAY_SPEC.md §10.4): column c produces pixel (base - c).
//
// Timing contract under test (SYSTOLIC_ARRAY_SPEC.md §9, §16):
//   - weight_load captures w_in into the PE weight register (available 1 cycle later)
//   - the weight stream leads the activation stream by one cycle
//   - a MAC is NOT suppressed on a weight_load cycle (the old weight is used)
//   - activation shifts one column per cycle via array-level registers
//   - the driver presents the weight stream for T+7 tap cycles (zero-padded) so
//     a delayed column c pairs a[t] with w[t+c] (not a frozen w[T-1]).
//==============================================================================

module tb_systolic_array;

    //--------------------------------------------------------------------------
    // DUT signals
    //--------------------------------------------------------------------------
    logic               clk         = 0;
    logic               rst         = 0;
    logic signed [7:0]  act_in      [0:7];
    logic signed [7:0]  w_in        [0:7];
    logic               weight_load = 0;
    logic               accum_clear = 0;
    logic               zero_skip   = 0;
    logic               result_req  [0:7];
    logic signed [31:0] result_out  [0:7];

    //--------------------------------------------------------------------------
    // Scoreboard
    //--------------------------------------------------------------------------
    int n_pass = 0;
    int n_fail = 0;

    //--------------------------------------------------------------------------
    // Stimulus state for the current group + captured results
    //--------------------------------------------------------------------------
    logic signed [7:0]  g_act     [0:7][0:63];  // activation per row per tap
    logic signed [7:0]  g_wgt     [0:7][0:63];  // weight per row per tap
    int                 g_ntaps   = 0;          // number of taps in current group
    logic signed [31:0] captured  [0:7][0:7];   // captured[r][c] = PE(r,c) result

    //--------------------------------------------------------------------------
    // Clock (10 ns period)
    //--------------------------------------------------------------------------
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    //--------------------------------------------------------------------------
    // DUT instantiation
    //--------------------------------------------------------------------------
    systolic_array dut (
        .clk         (clk),
        .rst         (rst),
        .act_in      (act_in),
        .w_in        (w_in),
        .weight_load (weight_load),
        .accum_clear (accum_clear),
        .zero_skip   (zero_skip),
        .result_req  (result_req),
        .result_out  (result_out)
    );

    //--------------------------------------------------------------------------
    // Check helper: exact bit-match, explicit PASS/FAIL.
    //--------------------------------------------------------------------------
    task automatic check(string name, longint got, longint expected);
        if (got == expected) begin
            n_pass++;
            $display("[PASS] %s  (got=%0d, expected=%0d)", name, got, expected);
        end else begin
            n_fail++;
            $display("[FAIL] %s  (got=%0d, expected=%0d)", name, got, expected);
        end
    endtask

    //--------------------------------------------------------------------------
    // Clear all stimulus arrays (so no stale tap data leaks between groups).
    //--------------------------------------------------------------------------
    task automatic zero_stimulus();
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 64; t++) begin
                g_act[r][t] = '0;
                g_wgt[r][t] = '0;
            end
    endtask

    //--------------------------------------------------------------------------
    // One array-wide accum_clear pulse (all other controls deasserted).
    //--------------------------------------------------------------------------
    task automatic clear_group();
        @(negedge clk);
        rst = 0;
        for (int r = 0; r < 8; r++) begin act_in[r] = '0; w_in[r] = '0; end
        weight_load = 0; zero_skip = 0; accum_clear = 1;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        @(posedge clk); #1;
        @(negedge clk);
        accum_clear = 0;
        @(posedge clk); #1;
    endtask

    //--------------------------------------------------------------------------
    // Drive one complete output group per the finalized schedule (§13, §16),
    // then leave all 64 accumulators stable. Uses module-level g_act/g_wgt/
    // g_ntaps. Optional zero_skip_tap asserts zero_skip during that tap cycle.
    //
    //   clear -> preload w[0] -> T+7 tap cycles -> one idle flush cycle
    //
    // The T+7 tap cycles keep the weight stream zero-padded past the kernel end
    // so every column c pairs a[t] with w[t+c] (reverse index), never with a
    // frozen w[T-1].
    //--------------------------------------------------------------------------
    task automatic drive_group(input int zero_skip_tap = -1);
        int T     = g_ntaps;
        int N_TAP = T + 7;

        clear_group();

        // weight preload of tap 0 (weight register <= w[0])
        @(negedge clk);
        for (int r = 0; r < 8; r++) begin act_in[r] = '0; w_in[r] = g_wgt[r][0]; end
        weight_load = 1; zero_skip = 0; accum_clear = 0;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        @(posedge clk); #1;

        // tap cycles: weight leads activation by one cycle; both zero-padded
        for (int t = 0; t < N_TAP; t++) begin
            @(negedge clk);
            for (int r = 0; r < 8; r++) begin
                if (t < T)     act_in[r] = g_act[r][t];
                else           act_in[r] = '0;
                if (t + 1 < T) w_in[r] = g_wgt[r][t+1];
                else           w_in[r] = '0;
            end
            weight_load = 1;             // asserted every tap cycle
            zero_skip   = (t == zero_skip_tap);
            accum_clear = 0;
            for (int c = 0; c < 8; c++) result_req[c] = 0;
            @(posedge clk); #1;
        end

        // one idle cycle: flush the final in-flight product into the accumulator
        @(negedge clk);
        for (int r = 0; r < 8; r++) begin act_in[r] = '0; w_in[r] = '0; end
        weight_load = 0; zero_skip = 0; accum_clear = 0;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        @(posedge clk); #1;
    endtask

    //--------------------------------------------------------------------------
    // Column-sequential result drain: assert result_req[c] one column at a time,
    // capture the selected column's result_out into captured[r][c].
    //--------------------------------------------------------------------------
    task automatic drain_all();
        for (int c = 0; c < 8; c++) begin
            @(negedge clk);
            for (int r = 0; r < 8; r++) begin act_in[r] = '0; w_in[r] = '0; end
            weight_load = 0; zero_skip = 0; accum_clear = 0;
            for (int cc = 0; cc < 8; cc++) result_req[cc] = (cc == c);
            @(posedge clk); #1;
            for (int r = 0; r < 8; r++) captured[r][c] = result_out[r];
            @(negedge clk);
            for (int cc = 0; cc < 8; cc++) result_req[cc] = 0;
            @(posedge clk); #1;
        end
    endtask

    //--------------------------------------------------------------------------
    // Drain the shift chain and any in-flight products (8 idle cycles), leaving
    // the shift chain empty and all accumulators stable. Used after a manual
    // (non-drive_group) stimulus sequence before starting a fresh group.
    //--------------------------------------------------------------------------
    task automatic flush_shift();
        repeat (8) begin
            @(negedge clk);
            for (int r = 0; r < 8; r++) begin act_in[r] = '0; w_in[r] = '0; end
            weight_load = 0; zero_skip = 0; accum_clear = 0;
            for (int c = 0; c < 8; c++) result_req[c] = 0;
            @(posedge clk); #1;
        end
    endtask

    //--------------------------------------------------------------------------
    // Independent reference: PE(r,c) = SUM_t a_r[t] * w_r[t+c], both streams
    // zero-padded. Matches the PE arithmetic (signed 8x8 product, 32-bit sum).
    //--------------------------------------------------------------------------
    function automatic int ref_dot(input int r, input int c);
        int s;
        s = 0;
        for (int t = 0; t < g_ntaps; t++) begin
            if (t + c < g_ntaps)
                s = s + $signed(g_act[r][t]) * $signed(g_wgt[r][t+c]);
        end
        return s;
    endfunction

    //==========================================================================
    // T0 — Synchronous active-high reset (SYSTOLIC_ARRAY_SPEC §13, §19.1)
    //==========================================================================
    task automatic test_reset();
        $display("[---] T0: synchronous active-high reset");

        // Establish a non-zero state across all 64 accumulators.
        zero_stimulus();
        g_ntaps = 1;
        for (int r = 0; r < 8; r++) begin g_act[r][0] = 3; g_wgt[r][0] = 5; end
        drive_group();
        drain_all();
        check("T0 pre-reset PE(0,0)=15", $signed(captured[0][0]), 15);
        check("T0 pre-reset PE(7,0)=15", $signed(captured[7][0]), 15);

        // Assert reset with every control high and non-zero inputs.
        @(negedge clk);
        rst = 1;
        for (int r = 0; r < 8; r++) begin act_in[r] = 8'sd37; w_in[r] = 8'sd7; end
        weight_load = 1; zero_skip = 1; accum_clear = 1;
        for (int c = 0; c < 8; c++) result_req[c] = 1;
        @(posedge clk); #1;
        @(negedge clk);
        @(posedge clk); #1;   // hold reset a second cycle

        // Deassert reset.
        @(negedge clk);
        rst = 0;
        for (int r = 0; r < 8; r++) begin act_in[r] = '0; w_in[r] = '0; end
        weight_load = 0; zero_skip = 0; accum_clear = 0;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        @(posedge clk); #1;

        // Accumulators cleared by reset (drain reads zeros).
        drain_all();
        check("T0 accumulator cleared by reset: PE(0,0)=0", $signed(captured[0][0]), 0);
        check("T0 accumulator cleared by reset: PE(7,7)=0", $signed(captured[7][7]), 0);

        // Weight cleared by reset: a MAC with no loaded weight contributes 0.
        @(negedge clk);
        for (int r = 0; r < 8; r++) begin act_in[r] = 8'sd7; w_in[r] = '0; end
        weight_load = 0; zero_skip = 0; accum_clear = 0;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        @(posedge clk); #1;   // product <= 7*0 = 0
        @(negedge clk);
        for (int r = 0; r < 8; r++) act_in[r] = '0;
        @(posedge clk); #1;   // accumulator <= 0 + 0 = 0
        @(negedge clk);
        result_req[0] = 1;
        @(posedge clk); #1;
        check("T0 weight cleared by reset (MAC with no weight = 0)", $signed(result_out[0]), 0);
        @(negedge clk);
        result_req[0] = 0;
        @(posedge clk); #1;

        // Clean recovery: a fresh group computes correctly after reset.
        zero_stimulus();
        g_ntaps = 1;
        for (int r = 0; r < 8; r++) begin g_act[r][0] = 3; g_wgt[r][0] = 5; end
        drive_group();
        drain_all();
        check("T0 clean recovery after reset: PE(0,0)=15", $signed(captured[0][0]), 15);
    endtask

    //==========================================================================
    // T1 — Activation shift + boundary columns 0 and 7 (§7, §14, §19.3/§19.7)
    //==========================================================================
    task automatic test_shift();
        $display("[---] T1: activation shift + boundary columns");

        // Single weight at tap 7 (w[7]=1), ramp activation a[t]=t+1.
        // Column c accumulates a[7-c] = 8-c  (col0=8 ... col7=1), which directly
        // exposes the c-cycle activation delay across the row.
        zero_stimulus();
        g_ntaps = 8;
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 8; t++) begin
                g_act[r][t] = t + 1;
                g_wgt[r][t] = (t == 7) ? 1 : 0;
            end
        drive_group();
        drain_all();
        for (int c = 0; c < 8; c++)
            check($sformatf("T1 shift col %0d = %0d", c, 8 - c),
                  $signed(captured[0][c]), 8 - c);
    endtask

    //==========================================================================
    // T2 — 64-PE structure + per-row weight broadcast (§4, §5, §8, §19.2)
    //==========================================================================
    task automatic test_broadcast();
        int f [0:7];
        $display("[---] T2: 64-PE structure + per-row weight broadcast");

        // Constant per-row weight w[r]=r+1, ramp activation a[t]=t+1.
        // PE(r,c) = (r+1) * f(c), f(c) = SUM_{t=0}^{7-c}(t+1).
        // All 64 values are distinct, proving 64 independent PEs and per-row
        // broadcast (same weight across all 8 columns of a row).
        f[0] = 36; f[1] = 28; f[2] = 21; f[3] = 15;
        f[4] = 10; f[5] = 6;  f[6] = 3;  f[7] = 1;

        zero_stimulus();
        g_ntaps = 8;
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 8; t++) begin
                g_act[r][t] = t + 1;
                g_wgt[r][t] = r + 1;
            end
        drive_group();
        drain_all();
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                check($sformatf("T2 PE(%0d,%0d)", r, c),
                      $signed(captured[r][c]), (r + 1) * f[c]);
    endtask

    //==========================================================================
    // T3 — Weight-lead skew: MAC uses the previous weight, never suppressed
    //      on a weight_load cycle (§9, §16, §19.4)
    //==========================================================================
    task automatic test_skew();
        $display("[---] T3: one-cycle weight-leads-activation skew");

        // T=2, w=[5,7], a=[3,2]. During tap 0, weight_load=1 with w_in=7, but the
        // product must use the OLD weight 5 (skew), and the MAC must still occur:
        //   col0 = a[0]*w[0] + a[1]*w[1] = 3*5 + 2*7 = 29
        // Wrong answers: 35 (new weight used immediately) or 14 (MAC suppressed).
        zero_stimulus();
        g_ntaps = 2;
        for (int r = 0; r < 8; r++) begin
            g_act[r][0] = 3; g_act[r][1] = 2;
            g_wgt[r][0] = 5; g_wgt[r][1] = 7;
        end
        drive_group();
        drain_all();
        check("T3 skew col0 = 29 (old weight, MAC kept)", $signed(captured[0][0]), 29);
        check("T3 skew col1 = 21 (a[0] paired with w[1])", $signed(captured[0][1]), 21);
        check("T3 skew col7 = 0 (weight zero-padded)", $signed(captured[0][7]), 0);
    endtask

    //==========================================================================
    // T4 — Signed 8x8 arithmetic incl. corner cases (§19.10)
    //==========================================================================
    task automatic test_signed();
        $display("[---] T4: signed 8x8 arithmetic (mixed sign + corners)");

        // Mixed-sign streams including -128 and 127 in both operands.
        zero_stimulus();
        g_ntaps = 8;
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 8; t++) begin
                case (t)
                    0: begin g_act[r][t] =  127; g_wgt[r][t] = -128; end
                    1: begin g_act[r][t] = -128; g_wgt[r][t] =  127; end
                    2: begin g_act[r][t] =    3; g_wgt[r][t] =   -1; end
                    3: begin g_act[r][t] =   -5; g_wgt[r][t] =    3; end
                    4: begin g_act[r][t] =    7; g_wgt[r][t] =   -5; end
                    5: begin g_act[r][t] =   -1; g_wgt[r][t] =    7; end
                    6: begin g_act[r][t] =  127; g_wgt[r][t] = -128; end
                    7: begin g_act[r][t] = -128; g_wgt[r][t] =  127; end
                endcase
            end
        drive_group();
        drain_all();
        check("T4 signed col0 = -65084 (hand-computed anchor)", $signed(captured[0][0]), -65084);
        for (int c = 0; c < 8; c++)
            check($sformatf("T4 signed col %0d", c), $signed(captured[0][c]), ref_dot(0, c));
    endtask

    //==========================================================================
    // T5 — Local 32-bit accumulation across 25 taps (§11.1, §19.10)
    //==========================================================================
    task automatic test_accum32();
        $display("[---] T5: 32-bit accumulation across 25 taps");

        // Positive: 25 taps of 127*127 = 16129 -> 403225 (exceeds 16-bit range).
        zero_stimulus();
        g_ntaps = 25;
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 25; t++) begin g_act[r][t] = 127; g_wgt[r][t] = 127; end
        drive_group();
        drain_all();
        check("T5 32-bit positive col0 = 403225", $signed(captured[0][0]), 403225);
        check("T5 32-bit positive col7 = 290322", $signed(captured[0][7]), 290322);
        for (int c = 0; c < 8; c++)
            check($sformatf("T5 32-bit positive col %0d", c), $signed(captured[0][c]), ref_dot(0, c));

        // Negative: 25 taps of -128*127 = -16256 -> -406400.
        zero_stimulus();
        g_ntaps = 25;
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 25; t++) begin g_act[r][t] = 127; g_wgt[r][t] = -128; end
        drive_group();
        drain_all();
        check("T5 32-bit negative col0 = -406400", $signed(captured[0][0]), -406400);
        for (int c = 0; c < 8; c++)
            check($sformatf("T5 32-bit negative col %0d", c), $signed(captured[0][c]), ref_dot(0, c));
    endtask

    //==========================================================================
    // T6 — zero_skip gates the MAC; activation forwarding continues (§12, §19.9)
    //==========================================================================
    task automatic test_zero_skip();
        int exp [0:7];
        $display("[---] T6: zero_skip gates MAC, forwarding continues");

        // T=8, w[t]=a[t]=1, zero_skip asserted during tap 3.
        // Column c loses the tap-3 term a[3-c]*w[3] only if 0<=3-c<=7 (c<=3);
        // later columns still receive the shifted activation after the skip.
        // Expected: [7,6,5,4,4,3,2,1].
        exp[0] = 7; exp[1] = 6; exp[2] = 5; exp[3] = 4;
        exp[4] = 4; exp[5] = 3; exp[6] = 2; exp[7] = 1;

        zero_stimulus();
        g_ntaps = 8;
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 8; t++) begin g_act[r][t] = 1; g_wgt[r][t] = 1; end
        drive_group(/* zero_skip_tap = */ 3);
        drain_all();
        for (int c = 0; c < 8; c++)
            check($sformatf("T6 zero_skip col %0d", c), $signed(captured[0][c]), exp[c]);
    endtask

    //==========================================================================
    // T7 — accum_clear flushes accumulator AND product register (§13, §19.1)
    //==========================================================================
    task automatic test_accum_clear();
        $display("[---] T7: accum_clear flushes accumulator + product");

        // Manual mid-pipeline clear: preload weight 3, present a=2 (product=6 in
        // flight), then accum_clear, then present a=5. Column 0 must be 15, not
        // 21 (which would indicate a stale +6 product was not flushed).
        clear_group();

        @(negedge clk);   // preload weight 3
        for (int r = 0; r < 8; r++) begin act_in[r] = '0; w_in[r] = 8'sd3; end
        weight_load = 1; zero_skip = 0; accum_clear = 0;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        @(posedge clk); #1;

        @(negedge clk);   // present a=2 -> product <= 6 (in flight)
        for (int r = 0; r < 8; r++) begin act_in[r] = 8'sd2; w_in[r] = '0; end
        weight_load = 0; zero_skip = 0; accum_clear = 0;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        @(posedge clk); #1;

        @(negedge clk);   // accum_clear -> accumulator <= 0, product <= 0
        for (int r = 0; r < 8; r++) begin act_in[r] = '0; w_in[r] = '0; end
        weight_load = 0; zero_skip = 0; accum_clear = 1;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        @(posedge clk); #1;
        @(negedge clk);
        accum_clear = 0;
        @(posedge clk); #1;

        @(negedge clk);   // present a=5 -> product <= 15, accumulator <= 0
        for (int r = 0; r < 8; r++) begin act_in[r] = 8'sd5; w_in[r] = '0; end
        weight_load = 0; zero_skip = 0; accum_clear = 0;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        @(posedge clk); #1;

        @(negedge clk);   // flush -> accumulator <= 15
        for (int r = 0; r < 8; r++) act_in[r] = '0;
        @(posedge clk); #1;

        @(negedge clk);
        result_req[0] = 1;
        @(posedge clk); #1;
        check("T7 accum_clear flush: col0 = 15 (no stale +6)", $signed(result_out[0]), 15);
        @(negedge clk);
        result_req[0] = 0;
        @(posedge clk); #1;

        // Drain the shift chain left dirty by the manual sequence above.
        flush_shift();

        // Array-wide accum_clear zeroes all 64 accumulators (and product regs).
        // Establish a clean non-zero state across every PE first (T=8 so column 7
        // is also non-zero), then clear and re-drain.
        zero_stimulus();
        g_ntaps = 8;
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 8; t++) begin g_act[r][t] = 2; g_wgt[r][t] = 3; end
        drive_group();
        drain_all();
        check("T7 pre-clear non-zero PE(0,0)=48", $signed(captured[0][0]), 48);
        check("T7 pre-clear non-zero PE(7,7)=6",  $signed(captured[7][7]), 6);

        clear_group();
        drain_all();
        check("T7 array-wide clear: PE(0,0)=0", $signed(captured[0][0]), 0);
        check("T7 array-wide clear: PE(7,7)=0", $signed(captured[7][7]), 0);
    endtask

    //==========================================================================
    // T8 — Column-sequential result drain + read-without-clear (§11.3, §19.6)
    //==========================================================================
    task automatic test_drain();
        logic signed [31:0] first [0:7][0:7];
        int mismatches = 0;
        $display("[---] T8: column-sequential drain + read-without-clear");

        zero_stimulus();
        g_ntaps = 8;
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 8; t++) begin g_act[r][t] = t + 1; g_wgt[r][t] = r + 1; end
        drive_group();
        drain_all();
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                first[r][c] = captured[r][c];

        // Re-drain: read-without-clear -> results unchanged.
        drain_all();
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                if (first[r][c] !== captured[r][c])
                    mismatches++;
        check("T8 read-without-clear: re-drain identical (64/64)", 64 - mismatches, 64);
    endtask

    //==========================================================================
    // T9 — Idle rows 6-7 for the 6-output-channel Conv1 case (§14, §17)
    //==========================================================================
    task automatic test_idle_rows();
        $display("[---] T9: idle rows 6-7 (6 output channels)");

        // 6 active rows (0-5) with per-row weights; rows 6-7 driven w=0.
        zero_stimulus();
        g_ntaps = 8;
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 8; t++) begin
                g_act[r][t] = t + 1;
                g_wgt[r][t] = (r < 6) ? (r + 1) : 0;
            end
        drive_group();
        drain_all();
        for (int r = 0; r < 6; r++)
            for (int c = 0; c < 8; c++)
                check($sformatf("T9 active row %0d col %0d", r, c),
                      $signed(captured[r][c]), ref_dot(r, c));
        for (int r = 6; r < 8; r++)
            for (int c = 0; c < 8; c++)
                check($sformatf("T9 idle row %0d col %0d = 0", r, c),
                      $signed(captured[r][c]), 0);
    endtask

    //==========================================================================
    // T10 — Back-to-back groups + clear between independent groups (§13, §16)
    //==========================================================================
    task automatic test_back_to_back();
        $display("[---] T10: back-to-back groups + clear between groups");

        // Group A
        zero_stimulus();
        g_ntaps = 5;
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 5; t++) begin g_act[r][t] = t + 1; g_wgt[r][t] = r + 1; end
        drive_group();
        drain_all();
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                check($sformatf("T10 group A PE(%0d,%0d)", r, c),
                      $signed(captured[r][c]), ref_dot(r, c));

        // Group B (different, immediately after; accum_clear inside drive_group
        // isolates it from group A)
        zero_stimulus();
        g_ntaps = 6;
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 6; t++) begin g_act[r][t] = 7 - t; g_wgt[r][t] = r + 3; end
        drive_group();
        drain_all();
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                check($sformatf("T10 group B PE(%0d,%0d)", r, c),
                      $signed(captured[r][c]), ref_dot(r, c));
    endtask

    //==========================================================================
    // T11 — Complete small deterministic convolution (6 channels, signed kernel),
    //       independently calculated expected results (§15, §19.5)
    //==========================================================================
    task automatic test_convolution();
        $display("[---] T11: complete small deterministic convolution");

        // 6 output channels (rows 0-5), 2 idle (rows 6-7).
        // 5-tap signed kernel base = [1,-1,2,-2,1], scaled per channel (r+1);
        // 8-tap input ramp img[t] = t+1. Every PE result is checked against the
        // independent reference ref_dot(r,c).
        zero_stimulus();
        g_ntaps = 8;
        for (int r = 0; r < 8; r++)
            for (int t = 0; t < 8; t++) begin
                int base;
                case (t)
                    0: base =  1;
                    1: base = -1;
                    2: base =  2;
                    3: base = -2;
                    4: base =  1;
                    default: base = 0;
                endcase
                g_act[r][t] = t + 1;
                g_wgt[r][t] = (r < 6) ? ((r + 1) * base) : 0;
            end
        drive_group();
        drain_all();
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                check($sformatf("T11 conv PE(%0d,%0d)", r, c),
                      $signed(captured[r][c]), ref_dot(r, c));
    endtask

    //==========================================================================
    // Test runner
    //==========================================================================
    initial begin
        // Global synchronous reset before any stimulus.
        rst = 1;
        for (int r = 0; r < 8; r++) begin act_in[r] = '0; w_in[r] = '0; end
        weight_load = 0; accum_clear = 0; zero_skip = 0;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        repeat (2) @(posedge clk);
        #1;
        rst = 0;
        @(posedge clk); #1;

        test_reset();
        test_shift();
        test_broadcast();
        test_skew();
        test_signed();
        test_accum32();
        test_zero_skip();
        test_accum_clear();
        test_drain();
        test_idle_rows();
        test_back_to_back();
        test_convolution();

        $display("----------------------------------------");
        $display("TOTAL checks : %0d", n_pass + n_fail);
        $display("PASS         : %0d", n_pass);
        $display("FAIL         : %0d", n_fail);
        $display("----------------------------------------");
        if (n_fail == 0) begin
            $display("RESULT: PASS");
            $finish;
        end else begin
            $display("RESULT: FAIL");
            $fatal(1, "tb_systolic_array: %0d of %0d checks failed", n_fail, n_pass + n_fail);
        end
    end

endmodule
