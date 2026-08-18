//==============================================================================
// tb_input_feed.sv — V1 Input-Feed / Controller integration testbench
//
// Self-checking, directed, non-UVM testbench for rtl/common/input_feed.sv
// driving rtl/common/systolic_array.sv (which instantiates 64 copies of
// rtl/common/pe.sv). Verifies the finalized V1 input-feed contract
// (docs/specs/INPUT_FEED_SPEC.md) WITHOUT modifying any RTL or the specs.
//
// Run (Vivado ML 2023.1):
//   source ~/Xilinx/Vivado/2023.1/settings64.sh
//   xvlog -sv rtl/common/pe.sv rtl/common/systolic_array.sv \
//          rtl/common/input_feed.sv sim/tb_input_feed.sv
//   xelab -debug typical -s tb_input_feed_snapshot tb_input_feed
//   xsim tb_input_feed_snapshot -runall
//
// Golden model (independent of the 5-pass hardware schedule, §16):
//   expected[ch][y][x] = SUM_ky SUM_kx input[y+ky][x+kx] * weight[ch][ky][kx]
// computed directly as a 5x5 2-D dot product from (y,x). It is NOT derived from
// the per-pass activation/weight streams, so it independently validates the
// controller's schedule.
//
// The DUT pair (input_feed + systolic_array) is exercised over the full frame:
// 24 output rows x 3 groups = 72 groups x 82 cycles = 5,904 cycles, producing
// 6 x 24 x 24 = 3,456 results, each compared bit-exact against the golden
// model. Output columns 16-23 (group 2) are covered and are the highest-value
// boundary check (INPUT_FEED_SPEC.md §16 item 5).
//==============================================================================

module tb_input_feed;

    //--------------------------------------------------------------------------
    // Clock (10 ns period)
    //--------------------------------------------------------------------------
    logic clk = 0;
    initial forever #5 clk = ~clk;

    logic rst = 0;

    //--------------------------------------------------------------------------
    // Behavioral memories (direct-addressed by the DUT)
    //--------------------------------------------------------------------------
    logic signed [7:0] image  [0:783];   // 28 x 28 input, flat row-major
    logic signed [7:0] weight [0:149];   // 6 x 5 x 5 weights, flat ch*25+ky*5+kx

    //--------------------------------------------------------------------------
    // input_feed <-> array wiring
    //--------------------------------------------------------------------------
    logic signed [7:0]  act_in      [0:7];
    logic signed [7:0]  w_in        [0:7];
    logic               weight_load;
    logic               accum_clear;
    logic               zero_skip;
    logic               result_req  [0:7];
    logic signed [31:0] result_out  [0:7];

    //--------------------------------------------------------------------------
    // input_feed memory / result interface
    //--------------------------------------------------------------------------
    logic [9:0]         img_addr;
    logic signed [7:0]  img_data;
    logic [4:0]         wgt_addr;
    logic signed [7:0]  wgt_data [0:5];
    logic               result_valid;
    logic [4:0]         result_y;
    logic [4:0]         result_x;
    logic signed [31:0] result_data [0:7];

    //--------------------------------------------------------------------------
    // Scoreboard
    //--------------------------------------------------------------------------
    logic signed [31:0] expected [0:5][0:23][0:23];
    logic               seen     [0:5][0:23][0:23];
    int n_pass = 0;
    int n_fail = 0;
    int n_captured = 0;
    int missing = 0;

    //--------------------------------------------------------------------------
    // Combinational memory reads (address -> data in the same cycle)
    //--------------------------------------------------------------------------
    always_comb begin
        img_data = image[img_addr];
    end

    always_comb begin
        for (int ch = 0; ch < 6; ch++)
            wgt_data[ch] = weight[ch*25 + wgt_addr];
    end

    //--------------------------------------------------------------------------
    // DUT instantiation
    //--------------------------------------------------------------------------
    input_feed u_feed (
        .clk          (clk),
        .rst          (rst),
        .img_addr     (img_addr),
        .img_data     (img_data),
        .wgt_addr     (wgt_addr),
        .wgt_data     (wgt_data),
        .act_in       (act_in),
        .w_in         (w_in),
        .weight_load  (weight_load),
        .accum_clear  (accum_clear),
        .zero_skip    (zero_skip),
        .result_req   (result_req),
        .result_out   (result_out),
        .result_valid (result_valid),
        .result_y     (result_y),
        .result_x     (result_x),
        .result_data  (result_data)
    );

    systolic_array u_array (
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
    // Golden model: direct 5x5 2-D dot product, independent of the schedule.
    //--------------------------------------------------------------------------
    task automatic compute_expected();
        for (int ch = 0; ch < 6; ch++) begin
            for (int y = 0; y < 24; y++) begin
                for (int x = 0; x < 24; x++) begin
                    logic signed [31:0] acc = 0;
                    for (int ky = 0; ky < 5; ky++) begin
                        for (int kx = 0; kx < 5; kx++) begin
                            int a = $signed(image[(y+ky)*28 + (x+kx)]);
                            int b = $signed(weight[ch*25 + ky*5 + kx]);
                            acc = acc + a * b;
                        end
                    end
                    expected[ch][y][x] = acc;
                end
            end
        end
    endtask

    //--------------------------------------------------------------------------
    // Result capture checker: sample the registered capture port.
    //--------------------------------------------------------------------------
    always @(posedge clk) begin
        if (result_valid) begin
            for (int ch = 0; ch < 6; ch++) begin
                n_captured++;
                if (seen[ch][result_y][result_x]) begin
                    n_fail++;
                    $display("[FAIL] duplicate capture ch=%0d y=%0d x=%0d",
                             ch, result_y, result_x);
                end
                seen[ch][result_y][result_x] = 1'b1;
                if (result_data[ch] !== expected[ch][result_y][result_x]) begin
                    n_fail++;
                    $display("[FAIL] ch=%0d y=%0d x=%0d got=%0d expected=%0d",
                             ch, result_y, result_x, result_data[ch],
                             expected[ch][result_y][result_x]);
                end else begin
                    n_pass++;
                end
            end
        end
    end

    //--------------------------------------------------------------------------
    // Stimulus
    //--------------------------------------------------------------------------
    initial begin
        // Deterministic pseudo-random (in-range signed 8-bit) stimulus.
        for (int r = 0; r < 28; r++)
            for (int c = 0; c < 28; c++)
                image[r*28+c] = 8'(((r*29 + c*13 + 7) % 256) - 128);

        for (int ch = 0; ch < 6; ch++)
            for (int ky = 0; ky < 5; ky++)
                for (int kx = 0; kx < 5; kx++)
                    weight[ch*25 + ky*5 + kx] = 8'(((ch*43 + ky*17 + kx*11 + 5) % 256) - 128);

        compute_expected();

        // Reset for a few cycles, then release.
        rst = 1;
        repeat (4) @(posedge clk);
        @(negedge clk);
        rst = 0;

        // Run the full frame (72 groups x 82 cycles = 5,904) plus margin. The
        // controller goes IDLE after the frame, so overrunning is harmless.
        repeat (72 * 82 + 32) @(posedge clk);

        //------------------------------------------------------------------------
        // Report
        //------------------------------------------------------------------------
        $display("----------------------------------------");
        $display("Results captured : %0d (expected %0d)", n_captured, 6*24*24);
        $display("PASS             : %0d", n_pass);
        $display("FAIL             : %0d", n_fail);
        $display("----------------------------------------");

        // Verify no output pixel was missed (bit-exact full coverage).
        missing = 0;
        for (int ch = 0; ch < 6; ch++)
            for (int y = 0; y < 24; y++)
                for (int x = 0; x < 24; x++)
                    if (!seen[ch][y][x]) begin
                        if (missing < 10)
                            $display("[MISS] ch=%0d y=%0d x=%0d", ch, y, x);
                        missing++;
                    end
        $display("Missing pixels   : %0d", missing);
        $display("----------------------------------------");

        if (n_fail == 0 && n_captured == 6*24*24 && missing == 0) begin
            $display("RESULT: PASS");
            $finish;
        end else begin
            $display("RESULT: FAIL");
            $fatal(1, "tb_input_feed: %0d failures, %0d missing",
                   n_fail, missing);
        end
    end

endmodule
