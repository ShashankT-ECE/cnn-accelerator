//==============================================================================
// tb_systolic_array_v2.sv — 8x8 Systolic Array V2 (WS) integration testbench
//
// Self-checking, directed, non-UVM testbench for
// rtl/common/systolic_array_v2.sv (which instantiates 64 copies of
// rtl/common/pe_v2.sv in the V2 Weight-Stationary dataflow). Verifies the
// resolved V2 WS contract (docs/specs/SYSTOLIC_ARRAY_V2_SPEC.md §3-§9,
// docs/PROJECT_STATE.md Decisions 10-11) WITHOUT modifying pe_v2.sv or any V1
// module/testbench.
//
// Run (Vivado ML 2023.1):
//   source ~/Xilinx/Vivado/2023.1/settings64.sh
//   xvlog -sv rtl/common/pe_v2.sv rtl/common/systolic_array_v2.sv sim/tb_systolic_array_v2.sv
//   xelab -debug typical -s tb_systolic_array_v2_snapshot tb_systolic_array_v2
//   xsim tb_systolic_array_v2_snapshot -runall
//
// Reference model (independent of the DUT — never derived from the V2 cycle
// schedule, PE rows/columns, tile counters, activation delays, or psum
// propagation). It is the mathematical 5x5 VALID convolution:
//
//   expected[y][x] = SUM_{ky=0..4} SUM_{kx=0..4}
//                    input[y+ky][x+kx] * weight[ky][kx]
//
// with input = 28x28 signed 8-bit, weight = 5x5 signed 8-bit, output 24x24.
// Column c produces output pixel (base - c) — the V1 reverse index.
//
// Timing contract under test (SYSTOLIC_ARRAY_V2_SPEC.md §9):
//   - 42 cycles/group; accum_clear at cycle 0 only
//   - weight_load at cycles 0/15/23/31 (tiles 0/1/2/3)
//   - per-row diagonal-skewed activation streams (a_t = 8t+1)
//   - vertical psum cascade + always-on row7->row0 feedback
//   - tile partials reach accum[7] at cycles 17/25/33; full result at 41
//   - result_request on row 7 at cycle 41; result_out captured at cycle 42
//==============================================================================

module tb_systolic_array_v2;

    //--------------------------------------------------------------------------
    // DUT signals
    //--------------------------------------------------------------------------
    logic               clk           = 0;
    logic               rst           = 0;
    logic signed [7:0]  act_in        [0:7];
    logic signed [7:0]  w_in          [0:7];
    logic               weight_load   = 0;
    logic               accum_clear   = 0;
    logic               zero_skip     = 0;
    logic               dataflow_mode = 0;
    logic               result_req    [0:7];
    logic signed [31:0] result_out    [0:7];

    //--------------------------------------------------------------------------
    // Scoreboard
    //--------------------------------------------------------------------------
    int n_pass = 0;
    int n_fail = 0;

    //--------------------------------------------------------------------------
    // Stimulus state (input feature map, kernel, current group)
    //--------------------------------------------------------------------------
    logic signed [7:0]  img    [0:27][0:27];  // 28x28 input feature map
    logic signed [7:0]  kernel [0:4][0:4];    // 5x5 Conv1 weights
    int                 g_y;                   // output row of the current group
    int                 g_base;                // rightmost output pixel (base)

    // Captured results (multi-cycle partial + final)
    logic signed [31:0] captured [0:7];        // final result (pixel base-c)
    logic signed [31:0] partial  [0:3][0:7];   // [0]=tile0, [1]=+tile1, [2]=+tile2, [3]=full

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
    systolic_array_v2 dut (
        .clk           (clk),
        .rst           (rst),
        .act_in        (act_in),
        .w_in          (w_in),
        .weight_load   (weight_load),
        .accum_clear   (accum_clear),
        .zero_skip     (zero_skip),
        .dataflow_mode (dataflow_mode),
        .result_req    (result_req),
        .result_out    (result_out)
    );

    //--------------------------------------------------------------------------
    // Hierarchical probe of the internal vertical psum cascade, for the
    // structural row-to-row propagation check (T3). psum is a module-level
    // unpacked array inside systolic_array_v2 (one entry per PE = its
    // registered accumulator). This is observation only — no DUT behaviour is
    // modified.
    //--------------------------------------------------------------------------
    wire signed [31:0] psum_probe [0:7][0:7];
    genvar pr, pc;
    generate
        for (pr = 0; pr < 8; pr++) begin : probe_r
            for (pc = 0; pc < 8; pc++) begin : probe_c
                assign psum_probe[pr][pc] = dut.psum[pr][pc];
            end
        end
    endgenerate

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
    // Clear all stimulus arrays.
    //--------------------------------------------------------------------------
    task automatic zero_img();
        for (int r = 0; r < 28; r++)
            for (int c = 0; c < 28; c++) img[r][c] = 0;
    endtask

    task automatic zero_kernel();
        for (int r = 0; r < 5; r++)
            for (int c = 0; c < 5; c++) kernel[r][c] = 0;
    endtask

    //--------------------------------------------------------------------------
    // Reset the DUT (2-cycle synchronous reset, all controls/idle inputs).
    //--------------------------------------------------------------------------
    task automatic reset_dut();
        @(negedge clk);
        rst = 1;
        for (int r = 0; r < 8; r++) begin act_in[r] = 0; w_in[r] = 0; end
        weight_load = 0; accum_clear = 0; zero_skip = 0; dataflow_mode = 1;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        @(posedge clk); #1;
        @(negedge clk);
        @(posedge clk); #1;
        @(negedge clk);
        rst = 0;
        @(posedge clk); #1;
    endtask

    //--------------------------------------------------------------------------
    // Independent golden model: 5x5 VALID convolution (28x28 -> 24x24).
    //   expected[y][x] = SUM_ky SUM_kx input[y+ky][x+kx] * weight[ky][kx]
    // Not derived from the V2 schedule / PE geometry / tile counters.
    //--------------------------------------------------------------------------
    function automatic longint golden(input int y, input int x);
        longint acc = 0;
        for (int ky = 0; ky < 5; ky++)
            for (int kx = 0; kx < 5; kx++)
                acc = acc + $signed(img[y+ky][x+kx]) * $signed(kernel[ky][kx]);
        return acc;
    endfunction

    //--------------------------------------------------------------------------
    // Partial golden: first n_taps taps (row-major k = 5*ky + kx), for the
    // tile-partial-sum checks. n_taps = 8/16/24/25 -> tile 0 / +1 / +2 / full.
    //--------------------------------------------------------------------------
    function automatic longint golden_partial(input int y, input int x, input int n_taps);
        longint acc = 0;
        for (int k = 0; k < n_taps; k++) begin
            int ky = k / 5;
            int kx = k % 5;
            acc = acc + $signed(img[y+ky][x+kx]) * $signed(kernel[ky][kx]);
        end
        return acc;
    endfunction

    //--------------------------------------------------------------------------
    // Per-row activation value at cycle s, per SYSTOLIC_ARRAY_V2_SPEC.md §9.5.
    //
    //   tile t streams starting at a_t = 8t+1; row r drives for
    //   s in [a_t+r, a_t+r+8):
    //     act_in[r](s) = input[y+k_y][ base-7 + (s-a_t) - r + k_x ]  (0 if OOB)
    //
    // This is the array's external input-feed responsibility: it produces the
    // eight distinct per-row diagonal-skewed streams that §9.5 requires. The
    // array itself only shifts them horizontally (unchanged from V1).
    //--------------------------------------------------------------------------
    function automatic logic signed [7:0] act_value(input int r, input int s);
        int t, k, ky, kx, a_t, px, py;
        t = -1;
        for (int tt = 0; tt < 4; tt++)
            if (s >= 8*tt + 1 + r && s <= 8*tt + 8 + r) t = tt;
        if (t < 0) return 0;                 // row r idle at cycle s
        if (t == 3 && r > 0) return 0;       // tile 3 has only tap 24 (row 0)

        k   = 8*t + r;                       // flat tap index
        ky  = k / 5;
        kx  = k % 5;
        a_t = 8*t + 1;
        px  = g_base - 7 + (s - a_t) - r + kx;   // input column
        py  = g_y + ky;                           // input row

        if (px < 0 || px > 27 || py < 0 || py > 27) return 0;
        return img[py][px];
    endfunction

    //--------------------------------------------------------------------------
    // Drive w_in[0:7] with tile t's tap weights (per-row, held per tile).
    //--------------------------------------------------------------------------
    task automatic set_tile_weights(input int t);
        for (int r = 0; r < 8; r++) begin
            if (t < 3) begin
                int k  = 8*t + r;
                int ky = k / 5;
                int kx = k % 5;
                w_in[r] = kernel[ky][kx];
            end else begin
                // tile 3: tap 24 (=(4,4)) on row 0 only; rows 1-7 pass +0
                w_in[r] = (r == 0) ? kernel[4][4] : 8'sd0;
            end
        end
    endtask

    //--------------------------------------------------------------------------
    // Drive one complete WS group (42 cycles, cycles 0..41) per the resolved
    // schedule (§9.3, §9.5), capturing:
    //   partial[0..3][c] = accum[7][c] at cycles 17 / 25 / 33 / 41
    //   (tile 0 / +tile 1 / +tile 2 / full 25-tap result)
    // and the final result into captured[0..7].
    //--------------------------------------------------------------------------
    task automatic drive_group();
        int cap_cycle [0:3];
        cap_cycle[0] = 17;
        cap_cycle[1] = 25;
        cap_cycle[2] = 33;
        cap_cycle[3] = 41;

        for (int s = 0; s < 42; s++) begin
            @(negedge clk);
            rst = 0;
            weight_load = 0;
            accum_clear = 0;
            zero_skip   = 0;
            dataflow_mode = 1;
            for (int c = 0; c < 8; c++) result_req[c] = 0;
            for (int r = 0; r < 8; r++) begin
                w_in[r]    = 8'sd0;
                act_in[r]  = act_value(r, s);
            end

            // control timeline (§9.3)
            if (s == 0) begin
                accum_clear = 1;
                weight_load = 1;
                set_tile_weights(0);
            end else if (s == 15) begin
                weight_load = 1;
                set_tile_weights(1);
            end else if (s == 23) begin
                weight_load = 1;
                set_tile_weights(2);
            end else if (s == 31) begin
                weight_load = 1;
                set_tile_weights(3);
            end

            // partial/final result strobes on row 7
            for (int i = 0; i < 4; i++)
                if (s == cap_cycle[i]) result_req[7] = 1;

            @(posedge clk); #1;

            // capture result_out one cycle after each strobe
            for (int i = 0; i < 4; i++)
                if (s == cap_cycle[i])
                    for (int c = 0; c < 8; c++) partial[i][c] = result_out[c];
        end

        // deassert result_req (read-without-clear: result_out is retained)
        @(negedge clk);
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        for (int r = 0; r < 8; r++) begin act_in[r] = 8'sd0; w_in[r] = 8'sd0; end
        weight_load = 0; accum_clear = 0; zero_skip = 0; dataflow_mode = 1;
        @(posedge clk); #1;

        for (int c = 0; c < 8; c++) captured[c] = partial[3][c];
    endtask

    //--------------------------------------------------------------------------
    // Check all 8 captured columns against the independent golden model.
    //--------------------------------------------------------------------------
    task automatic check_columns(string tag);
        for (int c = 0; c < 8; c++) begin
            int x = g_base - c;
            check($sformatf("%s col %0d (pixel %0d)", tag, c, x),
                  $signed(captured[c]), golden(g_y, x));
        end
    endtask

    //--------------------------------------------------------------------------
    // Single-tap arithmetic probe: kernel[ky][kx] = W, input[y+ky][x0+kx] = A,
    // so expected = A*W at output pixel x0 and 0 elsewhere.
    //--------------------------------------------------------------------------
    task automatic single_tap_check(input int ky, input int kx, input int x0,
                                    input logic signed [7:0] W,
                                    input logic signed [7:0] A,
                                    input string tag);
        zero_img(); zero_kernel();
        kernel[ky][kx] = W;
        img[g_y + ky][x0 + kx] = A;
        drive_group();
        for (int c = 0; c < 8; c++) begin
            int x = g_base - c;
            longint exp = (x == x0) ? $signed(A) * $signed(W) : 0;
            check($sformatf("%s col %0d (pixel %0d)", tag, c, x),
                  $signed(captured[c]), exp);
        end
    endtask

    //==========================================================================
    // T0 — Synchronous active-high reset
    //==========================================================================
    task automatic test_reset();
        $display("[---] T0: synchronous active-high reset");

        // Establish a non-zero state across all 8 columns.
        zero_kernel();
        for (int ky = 0; ky < 5; ky++)
            for (int kx = 0; kx < 5; kx++) kernel[ky][kx] = 2;
        zero_img();
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++) img[y][x] = 3;
        g_y = 12; g_base = 15;
        drive_group();
        check("T0 pre-reset non-zero result (col 0)", $signed(captured[0]),
              golden(g_y, 15));
        check("T0 pre-reset result is non-zero", (captured[0] != 0), 1);

        // Assert reset with controls high and non-zero inputs.
        @(negedge clk);
        rst = 1;
        for (int r = 0; r < 8; r++) begin act_in[r] = 37; w_in[r] = 7; end
        weight_load = 1; zero_skip = 1; accum_clear = 1; dataflow_mode = 1;
        for (int c = 0; c < 8; c++) result_req[c] = 1;
        @(posedge clk); #1;
        @(negedge clk);
        @(posedge clk); #1;   // hold reset a second cycle

        // Deassert reset; all registers (acc, weight, product, result_out,
        // shift) are cleared -> result_out reads 0.
        @(negedge clk);
        rst = 0;
        for (int r = 0; r < 8; r++) begin act_in[r] = 0; w_in[r] = 0; end
        weight_load = 0; zero_skip = 0; accum_clear = 0; dataflow_mode = 1;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        @(posedge clk); #1;
        check("T0 result_out cleared by reset (col 0)", $signed(result_out[0]), 0);
        check("T0 result_out cleared by reset (col 7)", $signed(result_out[7]), 0);

        // Clean recovery: a fresh group computes correctly after reset.
        drive_group();
        check("T0 clean recovery after reset (col 0)", $signed(captured[0]),
              golden(g_y, 15));
        check_columns("T0 clean recovery");
    endtask

    //==========================================================================
    // T1 — Weight loading and holding (single tap, weight held per tile)
    //==========================================================================
    task automatic test_weight_load();
        $display("[---] T1: weight loading and holding (single tap)");
        g_y = 12; g_base = 15;
        // tap 0 = (0,0): output pixel 11 gets A*W = 3*7 = 21.
        single_tap_check(0, 0, 11, 8'sd7, 8'sd3, "T1 weight load");
    endtask

    //==========================================================================
    // T2 — Activation diagonal skew (single tap on the deepest row)
    //==========================================================================
    task automatic test_activation_skew();
        $display("[---] T2: activation diagonal skew (tap on row 7)");
        g_y = 12; g_base = 15;
        // tap 7 = (1,2): row 7, delayed 7 cycles. Output pixel 12 gets 5*(-4).
        single_tap_check(1, 2, 12, 8'sd5, -8'sd4, "T2 row-7 skew");
        // tap 24 = (4,4): the single tap of tile 3 (row 0, deepest latency).
        single_tap_check(4, 4, 9, 8'sd6, -8'sd5, "T2 tile-3 tap-24");
    endtask

    //==========================================================================
    // T3 — Vertical psum cascade: row-to-row / 2-PE / 8-row propagation
    //==========================================================================
    task automatic test_cascade();
        $display("[---] T3: vertical psum cascade (row-to-row, 2-PE, 8-row)");

        reset_dut();

        // cycle 0: accum_clear + weight_load w[0]=5 (rows 1-7 weight 0)
        @(negedge clk);
        for (int r = 0; r < 8; r++) begin
            act_in[r] = 0;
            w_in[r]   = (r == 0) ? 8'sd5 : 8'sd0;
        end
        weight_load = 1; accum_clear = 1; zero_skip = 0; dataflow_mode = 1;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        @(posedge clk); #1;    // weight[0] <= 5, acc <= 0, product <= 0

        // cycle 1: present act_in[0]=3 for exactly one cycle
        @(negedge clk);
        for (int r = 0; r < 8; r++) begin
            act_in[r] = (r == 0) ? 8'sd3 : 8'sd0;
            w_in[r]   = 0;
        end
        weight_load = 0; accum_clear = 0;
        @(posedge clk); #1;    // product[0] <= 3*5 = 15

        // cycles 2..9: idle; the product 15 walks down one row per cycle:
        //   acc[r] == 15 during cycle (3+r), for r = 0..7.
        for (int cyc = 2; cyc <= 9; cyc++) begin
            @(negedge clk);
            for (int r = 0; r < 8; r++) begin act_in[r] = 0; w_in[r] = 0; end
            weight_load = 0; accum_clear = 0; zero_skip = 0; dataflow_mode = 1;
            for (int c = 0; c < 8; c++) result_req[c] = 0;
            @(posedge clk); #1;
            check($sformatf("T3 psum[%0d][0] = 15 (row-to-row)", cyc-2),
                  $signed(psum_probe[cyc-2][0]), 15);
        end
    endtask

    //==========================================================================
    // T4 — Tile decomposition + bottom-to-top feedback + single-tap tile 3
    //      (partial sums at cycles 17/25/33/41 vs independent partial golden)
    //==========================================================================
    task automatic test_tiles_feedback();
        $display("[---] T4: tile decomposition + feedback (17/25/33/41 partials)");

        // Deterministic mixed-sign kernel and input so every tap and every
        // partial sum is non-trivial.
        zero_kernel();
        for (int ky = 0; ky < 5; ky++)
            for (int kx = 0; kx < 5; kx++) begin
                int v;
                if (((ky + kx) % 3) == 0) v = -((ky*2 + kx) % 5) - 1;
                else                       v = ((ky + kx) % 5) + 1;
                kernel[ky][kx] = v;
            end
        zero_img();
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++)
                img[y][x] = ((x * 7 + y * 11) % 256) - 128;

        g_y = 12; g_base = 15;
        drive_group();

        // tile 0 partial (8 taps) at cycle 17
        for (int c = 0; c < 8; c++) begin
            int x = g_base - c;
            check($sformatf("T4 tile0 partial col %0d", c),
                  $signed(partial[0][c]), golden_partial(g_y, x, 8));
        end
        // tile 0 + 1 (16 taps) at cycle 25
        for (int c = 0; c < 8; c++) begin
            int x = g_base - c;
            check($sformatf("T4 tile0+1 feedback col %0d", c),
                  $signed(partial[1][c]), golden_partial(g_y, x, 16));
        end
        // tile 0 + 1 + 2 (24 taps) at cycle 33
        for (int c = 0; c < 8; c++) begin
            int x = g_base - c;
            check($sformatf("T4 tile0+1+2 feedback col %0d", c),
                  $signed(partial[2][c]), golden_partial(g_y, x, 24));
        end
        // full 25-tap result (tile 3 single tap) at cycle 41
        for (int c = 0; c < 8; c++) begin
            int x = g_base - c;
            check($sformatf("T4 full 25-tap (tile3) col %0d", c),
                  $signed(partial[3][c]), golden_partial(g_y, x, 25));
            check($sformatf("T4 full == golden col %0d", c),
                  $signed(captured[c]), golden(g_y, x));
        end
    endtask

    //==========================================================================
    // T5 — Result capture timing
    //      (result ready at accum[7] cycle 41; result_out latched at edge 42)
    //==========================================================================
    task automatic test_result_capture();
        $display("[---] T5: result capture timing (cycle 41 ready, edge 42 latched)");

        zero_kernel();
        for (int ky = 0; ky < 5; ky++)
            for (int kx = 0; kx < 5; kx++) kernel[ky][kx] = (ky + 1) * (kx - 2);
        zero_img();
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++)
                img[y][x] = ((x * 5 + y * 3) % 256) - 128;
        g_y = 12; g_base = 15;

        drive_group();

        // (1) accum[7] holds the full 25-tap result during cycle 41 (before the
        //     capture edge), i.e. the result is READY at the specified cycle.
        for (int c = 0; c < 8; c++) begin
            int x = g_base - c;
            check($sformatf("T5 accum[7] ready at cycle 41 col %0d", c),
                  $signed(partial[3][c]), golden(g_y, x));
        end

        // (2) result_out is latched at the edge ending cycle 41 (into cycle 42),
        //     matching the §9.8 capture timing. The result is also read-without-
        //     clear: the result_req strobes at cycles 17/25/33 (partial[0..2])
        //     did not clear the accumulator, since the final result is the full
        //     25-tap sum, not just tile 3's single tap.
        for (int c = 0; c < 8; c++) begin
            int x = g_base - c;
            check($sformatf("T5 result_out latched at cycle 42 col %0d", c),
                  $signed(captured[c]), golden(g_y, x));
        end
    endtask

    //==========================================================================
    // T6 — Signed 8x8 arithmetic incl. corner cases
    //==========================================================================
    task automatic test_signed();
        $display("[---] T6: signed 8x8 arithmetic (mixed sign + corners)");
        g_y = 12; g_base = 15;
        single_tap_check(0, 0, 11, 8'sd3,      8'sd5,     "T6 +*+");
        single_tap_check(0, 0, 11, 8'sd3,     -8'sd5,     "T6 +*-");
        single_tap_check(0, 0, 11, -8'sd3,     8'sd5,     "T6 -*+");
        single_tap_check(0, 0, 11, -8'sd3,    -8'sd5,     "T6 -*-");
        single_tap_check(0, 0, 11, -8'sd128,  -8'sd128,   "T6 -128*-128");
        single_tap_check(0, 0, 11, -8'sd128,   8'sd127,   "T6 -128*127");
        single_tap_check(0, 0, 11,  8'sd127,  -8'sd128,   "T6 127*-128");
        single_tap_check(0, 0, 11,  8'sd127,   8'sd127,   "T6 127*127");
    endtask

    //==========================================================================
    // T7 — Zero operands
    //==========================================================================
    task automatic test_zero();
        $display("[---] T7: zero operands");
        g_y = 12; g_base = 15;
        single_tap_check(0, 0, 11, 8'sd5, 8'sd0, "T7 a=0");
        single_tap_check(0, 0, 11, 8'sd0, 8'sd5, "T7 w=0");
        single_tap_check(0, 0, 11, 8'sd0, 8'sd0, "T7 both=0");
    endtask

    //==========================================================================
    // T8 — 32-bit accumulation (exceeds 16-bit product range)
    //==========================================================================
    task automatic test_accum32();
        $display("[---] T8: 32-bit accumulation across 25 taps");

        // Positive: 25 taps of 127*127 = 16129 -> 403225 (needs 19 bits).
        zero_kernel();
        for (int ky = 0; ky < 5; ky++)
            for (int kx = 0; kx < 5; kx++) kernel[ky][kx] = 127;
        zero_img();
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++) img[y][x] = 127;
        g_y = 12; g_base = 15;
        drive_group();
        for (int c = 0; c < 8; c++) begin
            int x = g_base - c;
            check($sformatf("T8 positive 32-bit col %0d", c),
                  $signed(captured[c]), golden(g_y, x));
        end
        check("T8 positive 32-bit anchor (403225)", $signed(captured[4]), 403225);

        // Negative: 25 taps of -128*127 = -16256 -> -406400.
        zero_kernel();
        for (int ky = 0; ky < 5; ky++)
            for (int kx = 0; kx < 5; kx++) kernel[ky][kx] = -128;
        zero_img();
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++) img[y][x] = 127;
        drive_group();
        for (int c = 0; c < 8; c++) begin
            int x = g_base - c;
            check($sformatf("T8 negative 32-bit col %0d", c),
                  $signed(captured[c]), golden(g_y, x));
        end
        check("T8 negative 32-bit anchor (-406400)", $signed(captured[4]), -406400);
    endtask

    //==========================================================================
    // T9 — Randomized cases (deterministic LCG, multiple seeds)
    //==========================================================================
    task automatic test_randomized();
        int rng;
        $display("[---] T9: randomized cases (deterministic LCG)");
        rng = 20260819;
        for (int trial = 0; trial < 4; trial++) begin
            // random 5x5 kernel
            for (int ky = 0; ky < 5; ky++)
                for (int kx = 0; kx < 5; kx++) begin
                    rng = (rng * 1103515245 + 12345) & 32'h7fffffff;
                    kernel[ky][kx] = (rng % 256) - 128;
                end
            // random 28x28 input
            for (int y = 0; y < 28; y++)
                for (int x = 0; x < 28; x++) begin
                    rng = (rng * 1103515245 + 12345) & 32'h7fffffff;
                    img[y][x] = (rng % 256) - 128;
                end
            g_y    = 12;
            g_base = 15;
            drive_group();
            check_columns($sformatf("T9 randomized trial %0d", trial));
        end
    endtask

    //==========================================================================
    // T10 — Conv1 full mapping: all 8 output columns vs independent golden
    //==========================================================================
    task automatic test_conv_full();
        $display("[---] T10: Conv1 full mapping (all 8 columns, golden)");

        // Deterministic mixed-sign kernel + input; interior group.
        zero_kernel();
        for (int ky = 0; ky < 5; ky++)
            for (int kx = 0; kx < 5; kx++)
                kernel[ky][kx] = ((ky * kx) % 7) - 3;
        zero_img();
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++)
                img[y][x] = ((x * 13 + y * 7) % 256) - 128;
        g_y = 12; g_base = 15;
        drive_group();
        check_columns("T10 conv full");
    endtask

    //==========================================================================
    // T11 — Output-column boundary cases (left edge base=7, right edge base=23)
    //==========================================================================
    task automatic test_boundaries();
        $display("[---] T11: output-column boundary cases");

        zero_kernel();
        for (int ky = 0; ky < 5; ky++)
            for (int kx = 0; kx < 5; kx++)
                kernel[ky][kx] = ((ky + kx) % 5) - 2;
        zero_img();
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++)
                img[y][x] = ((x * 3 + y * 17) % 256) - 128;

        // Left boundary: output pixels 0..7 (base = 7).
        g_y = 0; g_base = 7;
        drive_group();
        check_columns("T11 left edge (base 7)");

        // Right boundary: output pixels 16..23 (base = 23).
        g_y = 23; g_base = 23;
        drive_group();
        check_columns("T11 right edge (base 23)");

        // Top-left corner and bottom-right corner.
        g_y = 0; g_base = 7;
        drive_group();
        check_columns("T11 top-left corner");
        g_y = 23; g_base = 23;
        drive_group();
        check_columns("T11 bottom-right corner");
    endtask

    //==========================================================================
    // T12 — Multiple output rows
    //==========================================================================
    task automatic test_multi_rows();
        $display("[---] T12: multiple output rows");

        zero_kernel();
        for (int ky = 0; ky < 5; ky++)
            for (int kx = 0; kx < 5; kx++)
                kernel[ky][kx] = ((ky * 3) - kx) % 6 - 1;
        zero_img();
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++)
                img[y][x] = ((x * 11 + y * 5) % 256) - 128;

        g_base = 15;
        for (int i = 0; i < 4; i++) begin
            int y;
            case (i)
                0: y = 0;
                1: y = 5;
                2: y = 12;
                3: y = 23;
            endcase
            g_y = y;
            drive_group();
            check_columns($sformatf("T12 output row %0d", y));
        end
    endtask

    //==========================================================================
    // T13 — Multiple kernels / channels (independent golden per channel)
    //==========================================================================
    task automatic test_multi_kernels();
        int rng;
        $display("[---] T13: multiple kernels/channels");
        rng = 987654321;
        for (int ch = 0; ch < 3; ch++) begin
            // A distinct random kernel = one output channel.
            for (int ky = 0; ky < 5; ky++)
                for (int kx = 0; kx < 5; kx++) begin
                    rng = (rng * 1103515245 + 12345) & 32'h7fffffff;
                    kernel[ky][kx] = (rng % 256) - 128;
                end
            zero_img();
            for (int y = 0; y < 28; y++)
                for (int x = 0; x < 28; x++)
                    img[y][x] = ((x * 7 + y * 9) % 256) - 128;
            g_y = 12; g_base = 15;
            drive_group();
            check_columns($sformatf("T13 channel %0d", ch));
        end
    endtask

    //==========================================================================
    // Test runner
    //==========================================================================
    initial begin
        // Global synchronous reset before any stimulus.
        rst = 1;
        for (int r = 0; r < 8; r++) begin act_in[r] = 0; w_in[r] = 0; end
        weight_load = 0; accum_clear = 0; zero_skip = 0; dataflow_mode = 1;
        for (int c = 0; c < 8; c++) result_req[c] = 0;
        repeat (2) @(posedge clk);
        #1;
        rst = 0;
        @(posedge clk); #1;

        test_reset();
        test_weight_load();
        test_activation_skew();
        test_cascade();
        test_tiles_feedback();
        test_result_capture();
        test_signed();
        test_zero();
        test_accum32();
        test_randomized();
        test_conv_full();
        test_boundaries();
        test_multi_rows();
        test_multi_kernels();

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
            $fatal(1, "tb_systolic_array_v2: %0d of %0d checks failed", n_fail, n_pass + n_fail);
        end
    end

endmodule
