//==============================================================================
// tb_input_feed_v2.sv — Self-checking testbench for the V2 WS input-feed
//
// Instantiates input_feed_v2 (controller) + systolic_array_v2 (frozen array)
// and checks every Conv1 output against an INDEPENDENT golden model:
//
//   expected[ch][y][x] = SUM_{ky=0..4} SUM_{kx=0..4}
//                        img[y+ky][x+kx] * wgt[ch][ky][kx]
//
// The image is modelled as a flat 28x28 register array addressed by the
// controller's img_addr port (img_data = img[img_addr], combinational read).
//
// The golden model uses none of: controller counters, tile numbers, PE
// rows/columns, activation delays, weight-load timing, psum timing, or the V2
// schedule itself. Output ordering is tracked by a simple external counter
// (ch-major, then row, then group), not by reading controller internals.
//
// Verified: all 6 x 24 x 24 = 3456 outputs, all three groups, the right-edge
// group, all rows, all channels, signed +/-/0 data, reset, channel/group/tile
// transitions, result_req/capture timing, the 42-cycle group, row-7 result
// source, weight_load @0/15/23/31, accum_clear only at group start, and that
// phantom (off-diagonal) products do not corrupt captured results.
//
// Authoritative contract: docs/specs/INPUT_FEED_V2_SPEC.md.
//==============================================================================

module tb_input_feed_v2;

    logic               clk = 1'b0;
    logic               rst = 1'b0;
    logic               start = 1'b0;

    // External image memory (flat 28x28, read combinationally via img_addr)
    logic signed [7:0]  img [0:783];
    logic signed [7:0]  wgt [0:5][0:4][0:4];

    logic [9:0]         img_addr;
    logic signed [7:0]  img_data;

    // Controller <-> array interconnect
    logic signed [7:0]  act_out [0:7];
    logic signed [7:0]  w_out   [0:7];
    logic               weight_load;
    logic               accum_clear;
    logic               zero_skip;
    logic               dataflow_mode;
    logic               result_req [0:7];
    logic signed [31:0] result_in  [0:7];
    logic               done;
    logic               result_valid;
    logic signed [31:0] result_out [0:7];

    //--------------------------------------------------------------------------
    // DUTs
    //--------------------------------------------------------------------------
    input_feed_v2 u_if (
        .clk          (clk),
        .rst          (rst),
        .start        (start),
        .img_addr     (img_addr),
        .img_data     (img_data),
        .wgt          (wgt),
        .act_out      (act_out),
        .w_out        (w_out),
        .weight_load  (weight_load),
        .accum_clear  (accum_clear),
        .zero_skip    (zero_skip),
        .dataflow_mode(dataflow_mode),
        .result_req   (result_req),
        .result_in    (result_in),
        .done         (done),
        .result_valid (result_valid),
        .result_out   (result_out)
    );

    systolic_array_v2 u_arr (
        .clk          (clk),
        .rst          (rst),
        .act_in       (act_out),
        .w_in         (w_out),
        .weight_load  (weight_load),
        .accum_clear  (accum_clear),
        .zero_skip    (zero_skip),
        .dataflow_mode(dataflow_mode),
        .result_req   (result_req),
        .result_out   (result_in)
    );

    // Combinational image memory (flat 28x28).
    assign img_data = img[img_addr];

    //--------------------------------------------------------------------------
    // Clock
    //--------------------------------------------------------------------------
    always #5 clk = ~clk;

    //--------------------------------------------------------------------------
    // Independent golden model: 5x5 valid convolution (28x28 -> 24x24).
    // No schedule / tile / PE / delay knowledge.
    //--------------------------------------------------------------------------
    function automatic longint golden(input int ch_i, input int y_i, input int x_i);
        longint acc = 0;
        for (int ky = 0; ky < 5; ky++) begin
            for (int kx = 0; kx < 5; kx++) begin
                int a = $signed(img[(y_i + ky) * 28 + (x_i + kx)]);
                int b = $signed(wgt[ch_i][ky][kx]);
                acc = acc + a * b;
            end
        end
        return acc;
    endfunction

    //--------------------------------------------------------------------------
    // Result checking
    //--------------------------------------------------------------------------
    int n_pass = 0;
    int n_fail = 0;
    int result_count = 0;
    int ch_i, rem, y_i, b_i, base_i;
    longint got, exp;

    always @(posedge clk) begin
        if (result_valid) begin
            ch_i   = result_count / 72;
            rem    = result_count % 72;
            y_i    = rem / 3;
            b_i    = rem % 3;
            base_i = 7 + 8 * b_i;
            for (int c = 0; c < 8; c++) begin
                got = result_out[c];
                exp = golden(ch_i, y_i, base_i - c);
                if (got === exp) begin
                    n_pass = n_pass + 1;
                end else begin
                    n_fail = n_fail + 1;
                    if (n_fail <= 20)
                        $display("[FAIL] ch=%0d y=%0d x=%0d (grp=%0d col=%0d): got=%0d exp=%0d",
                                 ch_i, y_i, base_i - c, result_count, c, got, exp);
                end
            end
            result_count = result_count + 1;
        end
    end

    //--------------------------------------------------------------------------
    // Fixed-schedule assertions (first 42-cycle group after start)
    //--------------------------------------------------------------------------
    logic [5:0] sched_cyc;
    logic       sched_started;
    logic       exp_wl, exp_rq;

    always @(posedge clk) begin
        if (rst) begin
            sched_started <= 1'b0;
            sched_cyc     <= 6'd0;
        end else begin
            if (!sched_started) begin
                if (accum_clear) begin
                    // compute cycle 0: accum_clear, weight_load, result_req[7] all high
                    sched_started <= 1'b1;
                    sched_cyc     <= 6'd1;
                    if (!(weight_load && accum_clear && result_req[7]))
                        $fatal(1, "[SCHED] cycle 0: wl=%b ac=%b rq=%b",
                               weight_load, accum_clear, result_req[7]);
                end
            end else begin
                exp_wl = (sched_cyc == 6'd15 || sched_cyc == 6'd23 || sched_cyc == 6'd31);
                exp_rq = (sched_cyc == 6'd41);
                if (weight_load !== exp_wl)
                    $fatal(1, "[SCHED] weight_load=%b exp=%b @cyc%0d", weight_load, exp_wl, sched_cyc);
                if (accum_clear !== 1'b0)
                    $fatal(1, "[SCHED] accum_clear=%b @cyc%0d", accum_clear, sched_cyc);
                if (result_req[7] !== exp_rq)
                    $fatal(1, "[SCHED] result_req[7]=%b exp=%b @cyc%0d", result_req[7], exp_rq, sched_cyc);

                sched_cyc <= (sched_cyc == 6'd41) ? 6'd0 : sched_cyc + 6'd1;
                if (sched_cyc == 6'd41)
                    sched_started <= 1'b0;
            end
        end
    end

    //--------------------------------------------------------------------------
    // Main stimulus
    //--------------------------------------------------------------------------
    int timeout;

    initial begin
        // Load deterministic mixed-sign image and weights (positive, negative,
        // and zero values are all exercised by the modular patterns).
        for (int yy = 0; yy < 28; yy++)
            for (int xx = 0; xx < 28; xx++)
                img[yy * 28 + xx] = ((yy * 37 + xx * 17) % 256) - 128;

        for (int cc = 0; cc < 6; cc++)
            for (int ky = 0; ky < 5; ky++)
                for (int kx = 0; kx < 5; kx++)
                    wgt[cc][ky][kx] = ((cc * 101 + ky * 31 + kx * 13) % 256) - 128;

        // Reset
        rst   = 1'b1;
        start = 1'b0;
        repeat (3) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        // Start the full Conv1 frame (assert for one full cycle, stable at
        // the posedge so the controller samples it cleanly)
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        // Wait for done (with timeout: LOAD 784 + compute 18144 + margin)
        timeout = 0;
        while (done !== 1'b1 && timeout < 30000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (done !== 1'b1) begin
            $display("[FAIL] TIMEOUT: done never asserted");
            $fatal(1, "timeout");
        end

        // Final report
        $display("=== INPUT_FEED_V2 TB SUMMARY ===");
        $display("result_valid pulses : %0d (expect 432)", result_count);
        $display("outputs checked    : %0d (expect 3456)", n_pass + n_fail);
        $display("PASS               : %0d", n_pass);
        $display("FAIL               : %0d", n_fail);
        if (n_fail == 0 && result_count == 432 && (n_pass + n_fail) == 3456) begin
            $display("=== SIMULATION PASS ===");
        end else begin
            $display("=== SIMULATION FAIL ===");
        end
        $finish;
    end

endmodule
