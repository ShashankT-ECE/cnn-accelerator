//==============================================================================
// tb_cnn_accelerator_v2.sv — Phase 2 research accelerator integration testbench
//
// Drives rtl/common/cnn_accelerator_v2.sv (which wraps the FROZEN
// systolic_array_v2 + pe_v2) with the REAL Phase-1 vectors and verifies:
//
//   1. OS dense == independent integer golden (golden_canonical.hex)
//   2. WS dense == golden
//   3. OS -> WS runtime transition (no reset / reprogram)
//   4. WS -> OS runtime transition
//   5. SAME-padding boundaries (edges + 4 corners, directed synthetic vectors
//      against an independent padded-convolution golden)
//   6. all 8 output channels, all 28x28 positions (6,272 results)
//   7. result counts (896 result words / 6,272 valid pixels, no dup/missing)
//   8. sparsity counters (skipped/executed/total MACs) vs independent count
//   9. mode interlock: mode_commit rejected while busy, flush before switch
//
// The PRIMARY golden (golden_canonical.hex) is produced by the independent
// Python integer model (python/reference/int8_ref.py) — never by this TB or by
// the schedule. The directed boundary tests use a separate padded-convolution
// golden function written directly from the layer's mathematical definition.
//
// Run (Vivado ML 2023.1), from the repo root:
//   source ~/Xilinx/Vivado/2023.1/settings64.sh
//   xvlog -sv rtl/common/pe_v2.sv rtl/common/systolic_array_v2.sv \
//         rtl/common/cnn_accelerator_v2.sv sim/tb_cnn_accelerator_v2.sv
//   xelab -debug typical -s tb_cnn_acc_snapshot tb_cnn_accelerator_v2
//   xsim tb_cnn_acc_snapshot -runall
//==============================================================================

module tb_cnn_accelerator_v2;

    //--------------------------------------------------------------------------
    // DUT signals
    //--------------------------------------------------------------------------
    logic               clk = 1'b0;
    logic               rst = 1'b0;

    logic               dataflow_mode;
    logic               mode_commit;
    logic               start;
    logic               busy;
    logic               done;
    logic               mode_active;
    logic               mode_error;
    logic               switching;

    logic [9:0]         img_wr_addr;
    logic signed [7:0]  img_wr_data;
    logic               img_wr_en;

    logic [7:0]         wgt_wr_addr;
    logic signed [7:0]  wgt_wr_data;
    logic               wgt_wr_en;

    logic               result_valid;
    logic signed [31:0] result_data [0:7];
    logic [12:0]        result_base;
    logic               result_last;

    logic [31:0]        total_macs;
    logic [31:0]        executed_macs;
    logic [31:0]        skipped_macs;
    logic [31:0]        zero_skip_cycles;
    logic [31:0]        cycle_count;

    cnn_accelerator_v2 dut (
        .clk           (clk),
        .rst           (rst),
        .dataflow_mode (dataflow_mode),
        .mode_commit   (mode_commit),
        .start         (start),
        .busy          (busy),
        .done          (done),
        .mode_active   (mode_active),
        .mode_error    (mode_error),
        .switching     (switching),
        .img_wr_addr   (img_wr_addr),
        .img_wr_data   (img_wr_data),
        .img_wr_en     (img_wr_en),
        .wgt_wr_addr   (wgt_wr_addr),
        .wgt_wr_data   (wgt_wr_data),
        .wgt_wr_en     (wgt_wr_en),
        .result_valid  (result_valid),
        .result_data   (result_data),
        .result_base   (result_base),
        .result_last   (result_last),
        .total_macs    (total_macs),
        .executed_macs (executed_macs),
        .skipped_macs  (skipped_macs),
        .zero_skip_cycles(zero_skip_cycles),
        .cycle_count   (cycle_count)
    );

    //--------------------------------------------------------------------------
    // Test vectors (loaded from Phase-1 exports)
    //--------------------------------------------------------------------------
    logic signed [7:0]  img   [0:783];   // 28x28 row-major
    logic signed [7:0]  wgt   [0:199];   // ch*25 + ky*5 + kx
    logic signed [31:0] golden[0:6271];  // ch*784 + y*28 + x

    //--------------------------------------------------------------------------
    // Clock
    //--------------------------------------------------------------------------
    always #5 clk = ~clk;

    //--------------------------------------------------------------------------
    // Scoreboard
    //--------------------------------------------------------------------------
    int n_pass  = 0;
    int n_fail  = 0;
    int frame_result_words = 0;
    int frame_pixels       = 0;
    int frame_pix_fail     = 0;
    int frame_first_base   = -1;
    logic seen [0:6271];
    logic frame_active = 1'b0;
    logic use_sv_golden = 1'b0;   // 0: compare vs Python golden[]; 1: vs conv_golden()

    task automatic check(string name, longint got, longint expected);
        if (got === expected) begin
            n_pass++;
            $display("[PASS] %s", name);
        end else begin
            n_fail++;
            $display("[FAIL] %s  (got=%0d expected=%0d)", name, got, expected);
        end
    endtask

    //--------------------------------------------------------------------------
    // Result monitor: every result word is checked bit-exact against the golden,
    // with duplicate/missing detection via a seen bitmap.
    //--------------------------------------------------------------------------
    always @(posedge clk) begin
        if (result_valid) begin
            logic [12:0] y;
            logic [5:0]  base;
            logic [2:0]  ch;
            logic [12:0] rem;
            logic [4:0]  left;

            // result_base = ch*784 + y*28 + (base-7)  (leftmost pixel of the group)
            ch   = result_base / 784;
            rem  = result_base % 784;
            y    = rem / 28;
            left = rem % 28;       // 0 / 8 / 16 / 24
            base = left + 7;       // 7 / 15 / 23 / 31

            frame_result_words++;
            for (int c = 0; c < 8; c++) begin
                // column c -> pixel base-c; valid iff base-c <= 27
                if (base - c <= 27) begin
                    logic [12:0] gidx = result_base + 7 - c;   // ch*784 + y*28 + (base-c)
                    longint expv;
                    if (use_sv_golden)
                        expv = conv_golden(gidx / 784, (gidx % 784) / 28, gidx % 28);
                    else
                        expv = $signed(golden[gidx]);
                    if (gidx < 6272 && !seen[gidx]) begin
                        seen[gidx] = 1'b1;
                        frame_pixels++;
                        if ($signed(result_data[c]) === expv)
                            n_pass++;
                        else begin
                            n_fail++;
                            frame_pix_fail++;
                            if (frame_first_base < 0) frame_first_base = result_base;
                            if (n_fail <= 25)
                                $display("[FAIL] pixel ch=%0d y=%0d x=%0d base=%0d rbase=%0d gidx=%0d c=%0d: got=%0d exp=%0d",
                                         gidx / 784, (gidx % 784) / 28, gidx % 28, base,
                                         result_base, gidx, c,
                                         $signed(result_data[c]), expv);
                        end
                    end else begin
                        n_fail++;
                        if (n_fail <= 25)
                            $display("[FAIL] duplicate/oor result idx %0d", gidx);
                    end
                end
            end
        end
    end

    //--------------------------------------------------------------------------
    // DUT stimulus helpers
    //--------------------------------------------------------------------------
    task automatic write_image();
        for (int i = 0; i < 784; i++) begin
            @(negedge clk);
            img_wr_en   = 1'b1;
            img_wr_addr = i[9:0];
            img_wr_data = img[i];
            @(posedge clk); #1;
        end
        @(negedge clk);
        img_wr_en = 1'b0;
        @(posedge clk); #1;
    endtask

    task automatic write_weights();
        for (int i = 0; i < 200; i++) begin
            @(negedge clk);
            wgt_wr_en   = 1'b1;
            wgt_wr_addr = i[7:0];
            wgt_wr_data = wgt[i];
            @(posedge clk); #1;
        end
        @(negedge clk);
        wgt_wr_en = 1'b0;
        @(posedge clk); #1;
    endtask

    // Commit a mode and wait for the flush to finish.
    task automatic set_mode(input logic m);
        @(negedge clk);
        dataflow_mode = m;
        mode_commit   = 1'b1;
        @(negedge clk);
        mode_commit = 1'b0;
        // wait until flush completes and DUT is idle again
        while (switching !== 1'b0 || busy !== 1'b0) @(posedge clk);
        @(posedge clk); #1;
        check($sformatf("mode_active == %0d after commit", m),
              mode_active, m);
    endtask

    // Begin a frame and reset per-frame scoreboard state.
    task automatic run_frame();
        for (int i = 0; i < 6272; i++) seen[i] = 1'b0;
        frame_result_words = 0;
        frame_pixels       = 0;
        frame_pix_fail     = 0;
        frame_first_base   = -1;
        frame_active       = 1'b1;
        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;
        // wait for done
        while (done !== 1'b1) @(posedge clk);
        @(posedge clk); #1;
        frame_active = 1'b0;
        if (frame_pix_fail > 0)
            $display("  [frame] pixel fails=%0d first_rbase=%0d", frame_pix_fail, frame_first_base);
    endtask

    //--------------------------------------------------------------------------
    // Independent padded-convolution golden (for directed boundary vectors).
    // The mathematical definition of the SAME-padded 5x5 conv, not the schedule.
    //--------------------------------------------------------------------------
    function automatic longint conv_golden(input int ch, input int y, input int x);
        longint acc = 0;
        for (int ky = 0; ky < 5; ky++) begin
            int iy = y + ky - 2;
            if (iy < 0 || iy >= 28) continue;
            for (int kx = 0; kx < 5; kx++) begin
                int ix = x + kx - 2;
                if (ix < 0 || ix >= 28) continue;
                acc = acc + $signed(img[iy*28 + ix]) * $signed(wgt[ch*25 + ky*5 + kx]);
            end
        end
        return acc;
    endfunction

    //--------------------------------------------------------------------------
    // Independent zero-activation MAC count (skipped MACs), used to cross-check
    // the RTL sparsity counter. Mode-independent: depends only on image.
    //--------------------------------------------------------------------------
    function automatic longint expected_skipped();
        longint sk = 0;
        for (int y = 0; y < 28; y++)
            for (int x = 0; x < 28; x++)
                for (int ky = 0; ky < 5; ky++)
                    for (int kx = 0; kx < 5; kx++) begin
                        int iy = y + ky - 2;
                        int ix = x + kx - 2;
                        if (iy < 0 || iy >= 28 || ix < 0 || ix >= 28 ||
                            img[iy*28 + ix] == 8'sd0)
                            sk++;
                    end
        return sk * 8;
    endfunction

    //--------------------------------------------------------------------------
    // Directed boundary test: run one frame in `m` mode, compare all 6,272
    // outputs against the independent padded-convolution golden.
    //--------------------------------------------------------------------------
    task automatic boundary_frame(input string tag, input logic m);
        write_image();
        write_weights();
        set_mode(m);
        run_frame();
        check($sformatf("%s: result words == 896", tag), frame_result_words, 896);
        check($sformatf("%s: valid pixels == 6272", tag), frame_pixels, 6272);
    endtask

    //==========================================================================
    // Test flow
    //==========================================================================
    initial begin
        int rng;
        $display("=== Phase 2 research accelerator testbench ===");

        // ---- reset ----
        rst = 1'b1;
        dataflow_mode = 1'b0; mode_commit = 1'b0; start = 1'b0;
        img_wr_en = 1'b0; wgt_wr_en = 1'b0;
        img_wr_addr = 0; img_wr_data = 0; wgt_wr_addr = 0; wgt_wr_data = 0;
        repeat (3) @(posedge clk);
        rst = 1'b0;
        repeat (2) @(posedge clk);

        // ---- load real Phase-1 vectors ----
        $readmemh("data/vectors/weights.hex", wgt);
        $readmemh("data/vectors/input_img.hex", img);
        $readmemh("data/vectors/golden_canonical.hex", golden);
        $display("loaded real vectors: weights=200 img=784 golden=6272");

        // Sanity: the Python golden must equal the independent SV padded-conv
        // definition for the primary image (confirms no loading mismatch).
        begin
            int bad = 0;
            for (int gg = 0; gg < 6272; gg++) begin
                int c = gg / 784, yy = (gg % 784) / 28, xx = gg % 28;
                if ($signed(golden[gg]) !== conv_golden(c, yy, xx)) bad++;
            end
            check("Sanity: Python golden == SV conv_golden (6272/6272)", 6272 - bad, 6272);
        end

        //==================================================================
        // T1 — OS dense vs independent golden (real MNIST vector)
        //==================================================================
        $display("[---] T1: OS dense vs golden");
        write_image();
        write_weights();
        set_mode(1'b0);
        run_frame();
        check("T1 OS: result words == 896", frame_result_words, 896);
        check("T1 OS: valid pixels == 6272", frame_pixels, 6272);
        check("T1 OS: total_macs == 156800", total_macs, 156800);
        check("T1 OS: skipped_macs == expected", skipped_macs, expected_skipped());
        check("T1 OS: executed+skipped == total", executed_macs + skipped_macs, 156800);
        $display("  OS cycles=%0d skipped=%0d executed=%0d",
                 cycle_count, skipped_macs, executed_macs);

        //==================================================================
        // T2 — WS dense vs golden
        //==================================================================
        $display("[---] T2: WS dense vs golden");
        write_image();
        write_weights();
        set_mode(1'b1);
        run_frame();
        check("T2 WS: result words == 896", frame_result_words, 896);
        check("T2 WS: valid pixels == 6272", frame_pixels, 6272);
        check("T2 WS: skipped_macs == expected", skipped_macs, expected_skipped());
        check("T2 WS: executed+skipped == total", executed_macs + skipped_macs, 156800);
        $display("  WS cycles=%0d skipped=%0d executed=%0d",
                 cycle_count, skipped_macs, executed_macs);

        //==================================================================
        // T3 — OS -> WS runtime transition (no reset)
        //==================================================================
        $display("[---] T3: OS -> WS runtime transition (no reset)");
        write_image();
        write_weights();
        set_mode(1'b0);
        run_frame();
        check("T3 pre: OS dense ok", frame_pixels, 6272);
        // switch to WS without reset
        set_mode(1'b1);
        run_frame();
        check("T3 post: WS dense ok after OS->WS", frame_pixels, 6272);
        check("T3: no reset — WS skipped == expected", skipped_macs, expected_skipped());

        //==================================================================
        // T4 — WS -> OS runtime transition (no reset)
        //==================================================================
        $display("[---] T4: WS -> OS runtime transition (no reset)");
        set_mode(1'b0);
        run_frame();
        check("T4 post: OS dense ok after WS->OS", frame_pixels, 6272);
        check("T4: no reset — OS skipped == expected", skipped_macs, expected_skipped());

        //==================================================================
        // T5 — mode interlock: mode_commit rejected while busy
        //==================================================================
        $display("[---] T5: mode interlock (mode_commit while busy rejected)");
        write_image();
        write_weights();
        set_mode(1'b1);
        // start a frame, then try to commit mid-run
        for (int i = 0; i < 6272; i++) seen[i] = 1'b0;
        frame_result_words = 0;
        frame_pixels       = 0;
        @(negedge clk); start = 1'b1; @(negedge clk); start = 1'b0;
        // wait until busy is high
        while (busy !== 1'b1) @(posedge clk);
        // attempt mode_commit while busy
        @(negedge clk);
        dataflow_mode = 1'b0;
        mode_commit = 1'b1;
        @(negedge clk);
        mode_commit = 1'b0;
        // mode must not change mid-run
        if (mode_active !== 1'b1) begin
            n_fail++;
            $display("[FAIL] T5: mode_active changed while busy");
        end else begin
            n_pass++;
            $display("[PASS] T5: mode_active unchanged while busy");
        end
        while (done !== 1'b1) @(posedge clk);
        @(posedge clk); #1;
        // the in-flight frame results must still be correct
        check("T5: in-flight WS frame still correct", frame_pixels, 6272);

        //==================================================================
        // T6 — SAME-padding boundaries (directed synthetic vectors)
        //==================================================================
        $display("[---] T6: SAME-padding boundaries (directed vectors)");

        // (a) all-ones image: every output = sum over its valid 5x5 taps.
        for (int i = 0; i < 784; i++) img[i] = 8'sd1;
        use_sv_golden = 1'b1;
        boundary_frame("T6a all-ones OS", 1'b0);
        boundary_frame("T6a all-ones WS", 1'b1);

        // (b) single corner pixel: top-left input pixel 5, everything else 0.
        for (int i = 0; i < 784; i++) img[i] = 8'sd0;
        img[0] = 8'sd5;
        boundary_frame("T6b corner-pixel OS", 1'b0);
        boundary_frame("T6b corner-pixel WS", 1'b1);

        // (c) deterministic gradient image (all positions non-trivial).
        for (int yy = 0; yy < 28; yy++)
            for (int xx = 0; xx < 28; xx++)
                img[yy*28 + xx] = ((xx*3 + yy*5) % 128) - 64;
        boundary_frame("T6c gradient OS", 1'b0);
        boundary_frame("T6c gradient WS", 1'b1);
        use_sv_golden = 1'b0;

        //==================================================================
        // Final report
        //==================================================================
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
            $fatal(1, "tb_cnn_accelerator_v2: %0d of %0d checks failed",
                   n_fail, n_pass + n_fail);
        end
    end

endmodule
