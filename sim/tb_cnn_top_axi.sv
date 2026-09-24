//==============================================================================
// tb_cnn_top_axi.sv — AXI4-Lite board-integration testbench
//
// Drives rtl/top/cnn_top.sv (AXI4-Lite slave + cnn_axi_ctrl + cnn_accelerator_v2)
// through the AXI interface with the REAL Phase-1 vectors and verifies the full
// board-facing path:
//
//   1. image + weight loading through the AXI auto-increment write windows
//   2. OS frame: START -> DONE, cycle count == 9184, results bit-exact vs golden
//   3. WS frame (sparse): cycle count == 18368, results bit-exact
//   4. WS dense (sparsity_disable): cycle count == 38528, results bit-exact
//   5. runtime OS->WS->OS switch through DATAFLOW_MODE + MODE_COMMIT (no reset)
//   6. MAC/zero-skip counters vs independent expected count
//
// The golden (golden_canonical.hex) is produced by the independent Python
// integer model (python/reference/int8_ref.py), never by this TB or the DUT.
//
// Run (Vivado ML 2023.1), from the repo root:
//   source ~/Xilinx/Vivado/2023.1/settings64.sh
//   xvlog -sv rtl/common/pe_v2.sv rtl/common/systolic_array_v2.sv \
//         rtl/common/cnn_accelerator_v2.sv rtl/common/cnn_axi_ctrl.sv \
//         rtl/top/cnn_top.sv sim/tb_cnn_top_axi.sv
//   xelab -debug typical -s tb_cnn_top_axi_snap tb_cnn_top_axi
//   xsim tb_cnn_top_axi_snap -runall
//==============================================================================

module tb_cnn_top_axi;

    //--------------------------------------------------------------------------
    // DUT signals
    //--------------------------------------------------------------------------
    logic          clk     = 1'b0;
    logic          aresetn = 1'b0;

    logic [31:0]   s_axi_awaddr;
    logic          s_axi_awvalid = 1'b0;
    logic          s_axi_awready;
    logic [31:0]   s_axi_wdata;
    logic [3:0]    s_axi_wstrb = 4'hF;
    logic          s_axi_wvalid = 1'b0;
    logic          s_axi_wready;
    logic [1:0]    s_axi_bresp;
    logic          s_axi_bvalid;
    logic          s_axi_bready = 1'b0;
    logic [31:0]   s_axi_araddr;
    logic          s_axi_arvalid = 1'b0;
    logic          s_axi_arready;
    logic [31:0]   s_axi_rdata;
    logic [1:0]    s_axi_rresp;
    logic          s_axi_rvalid;
    logic          s_axi_rready = 1'b0;
    logic          irq;

    cnn_top dut (
        .s_axi_aclk    (clk),
        .s_axi_aresetn (aresetn),
        .s_axi_awaddr  (s_axi_awaddr),
        .s_axi_awvalid (s_axi_awvalid),
        .s_axi_awready (s_axi_awready),
        .s_axi_wdata   (s_axi_wdata),
        .s_axi_wstrb   (s_axi_wstrb),
        .s_axi_wvalid  (s_axi_wvalid),
        .s_axi_wready  (s_axi_wready),
        .s_axi_bresp   (s_axi_bresp),
        .s_axi_bvalid  (s_axi_bvalid),
        .s_axi_bready  (s_axi_bready),
        .s_axi_araddr  (s_axi_araddr),
        .s_axi_arvalid (s_axi_arvalid),
        .s_axi_arready (s_axi_arready),
        .s_axi_rdata   (s_axi_rdata),
        .s_axi_rresp   (s_axi_rresp),
        .s_axi_rvalid  (s_axi_rvalid),
        .s_axi_rready  (s_axi_rready),
        .irq           (irq)
    );

    //--------------------------------------------------------------------------
    // Register map (byte offsets)
    //--------------------------------------------------------------------------
    localparam [31:0] ADDR_CONTROL       = 32'h0004;
    localparam [31:0] ADDR_STATUS        = 32'h0008;
    localparam [31:0] ADDR_DATAFLOW_MODE = 32'h000C;
    localparam [31:0] ADDR_MODE_STATUS   = 32'h0010;
    localparam [31:0] ADDR_SPARSITY_DIS  = 32'h0014;
    localparam [31:0] ADDR_CYCLE_RUN     = 32'h0018;
    localparam [31:0] ADDR_TOTAL_MACS    = 32'h001C;
    localparam [31:0] ADDR_EXECUTED      = 32'h0020;
    localparam [31:0] ADDR_SKIPPED       = 32'h0024;
    localparam [31:0] ADDR_IMAGE_ADDR    = 32'h002C;
    localparam [31:0] ADDR_IMAGE_DATA    = 32'h0030;
    localparam [31:0] ADDR_WEIGHT_ADDR   = 32'h0034;
    localparam [31:0] ADDR_WEIGHT_DATA   = 32'h0038;
    localparam [31:0] ADDR_RESULT_BASE   = 32'h1000;

    localparam [31:0] CTRL_START       = 32'h1;
    localparam [31:0] CTRL_SOFT_RESET  = 32'h2;
    localparam [31:0] CTRL_MODE_COMMIT = 32'h4;

    //--------------------------------------------------------------------------
    // Clock
    //--------------------------------------------------------------------------
    always #5 clk = ~clk;   // 100 MHz sim clock; cycle counts are clock-agnostic

    //--------------------------------------------------------------------------
    // Test vectors (loaded from Phase-1 exports)
    //--------------------------------------------------------------------------
    logic signed [7:0]  img    [0:783];
    logic signed [7:0]  wgt    [0:199];
    logic signed [31:0] golden [0:6271];

    //--------------------------------------------------------------------------
    // Scoreboard
    //--------------------------------------------------------------------------
    int n_pass = 0;
    int n_fail = 0;

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
    // AXI4-Lite bus-functional model
    //
    // Control signals are driven at negedge and held stable across the posedge
    // (where the slave's always_ff samples them), then changed only at the next
    // negedge. This avoids the classic posedge race where deasserting a control
    // signal on the same edge the slave samples it causes the slave to miss the
    // handshake.
    //--------------------------------------------------------------------------
    task automatic axi_write(input logic [31:0] addr, input logic [31:0] data);
        @(negedge clk);
        s_axi_awaddr  = addr;
        s_axi_awvalid = 1'b1;
        s_axi_wdata   = data;
        s_axi_wvalid  = 1'b1;
        @(negedge clk);   // hold valid high across the posedge
        s_axi_awvalid = 1'b0;
        s_axi_wvalid  = 1'b0;
        while (s_axi_bvalid !== 1'b1) @(posedge clk);
        @(negedge clk);
        s_axi_bready = 1'b1;
        @(negedge clk);
        s_axi_bready = 1'b0;
    endtask

    task automatic axi_read(input logic [31:0] addr, output logic [31:0] data);
        @(negedge clk);
        s_axi_araddr  = addr;
        s_axi_arvalid = 1'b1;
        @(negedge clk);   // hold valid high across the posedge
        s_axi_arvalid = 1'b0;
        while (s_axi_rvalid !== 1'b1) @(posedge clk);
        data = s_axi_rdata;
        @(negedge clk);
        s_axi_rready = 1'b1;
        @(negedge clk);
        s_axi_rready = 1'b0;
    endtask

    //--------------------------------------------------------------------------
    // High-level driver tasks
    //--------------------------------------------------------------------------
    task automatic load_image();
        axi_write(ADDR_IMAGE_ADDR, 32'd0);
        for (int i = 0; i < 784; i++)
            axi_write(ADDR_IMAGE_DATA, {24'b0, img[i]});
    endtask

    task automatic load_weights();
        axi_write(ADDR_WEIGHT_ADDR, 32'd0);
        for (int i = 0; i < 200; i++)
            axi_write(ADDR_WEIGHT_DATA, {24'b0, wgt[i]});
    endtask

    // poll a STATUS bit until set (with a timeout guard)
    task automatic wait_status(input int bitidx);
        int guard = 0;
        logic [31:0] st;
        while (guard < 200000) begin
            axi_read(ADDR_STATUS, st);
            if (st[bitidx] === 1'b1) return;
            guard++;
        end
        n_fail++;
        $display("[FAIL] wait_status bit=%0d timed out", bitidx);
    endtask

    task automatic set_mode(input logic m);
        axi_write(ADDR_DATAFLOW_MODE, {31'b0, m});
        axi_write(ADDR_CONTROL, CTRL_MODE_COMMIT);
        wait_status(0);   // wait IDLE
    endtask

    task automatic start_and_wait_done();
        axi_write(ADDR_CONTROL, CTRL_START);
        wait_status(2);   // wait DONE
    endtask

    // Read back all 6272 results and compare bit-exact vs golden.
    task automatic check_results(string tag);
        int bad = 0;
        for (int ch = 0; ch < 8; ch++)
            for (int y = 0; y < 28; y++)
                for (int x = 0; x < 28; x++) begin
                    int flat  = ch*784 + y*28 + x;
                    int bank  = x & 7;
                    int dense = flat >> 3;
                    logic [31:0] addr = ADDR_RESULT_BASE + 4*(bank*1024 + dense);
                    logic [31:0] v;
                    axi_read(addr, v);
                    if ($signed(v) !== golden[flat]) begin
                        if (bad < 10)
                            $display("  [%s] mismatch ch=%0d y=%0d x=%0d got=%0d exp=%0d",
                                     tag, ch, y, x, $signed(v), golden[flat]);
                        bad++;
                    end
                end
        check($sformatf("%s: results bit-exact (6272)", tag), 6272 - bad, 6272);
    endtask

    // Run one full frame in a given mode/sparsity config and check cycle + results.
    task automatic run_config(string tag, input logic m, input logic sp,
                              input longint cycle_expect);
        axi_write(ADDR_DATAFLOW_MODE, {31'b0, m});
        axi_write(ADDR_SPARSITY_DIS, {31'b0, sp});
        axi_write(ADDR_CONTROL, CTRL_MODE_COMMIT);
        wait_status(0);   // IDLE
        axi_write(ADDR_CONTROL, CTRL_START);
        wait_status(2);   // DONE
        begin
            logic [31:0] cyc;
            axi_read(ADDR_CYCLE_RUN, cyc);
            check($sformatf("%s: cycle_count == %0d", tag, cycle_expect), cyc, cycle_expect);
        end
        check_results(tag);
    endtask

    //--------------------------------------------------------------------------
    // Test flow
    //--------------------------------------------------------------------------
    initial begin
        logic [31:0] rd;

        $display("=== AXI board-integration testbench (cnn_top) ===");

        // reset
        aresetn = 1'b0;
        repeat (4) @(posedge clk);
        aresetn = 1'b1;
        repeat (2) @(posedge clk);

        // load real vectors
        $readmemh("data/vectors/weights.hex", wgt);
        $readmemh("data/vectors/input_img.hex", img);
        $readmemh("data/vectors/golden_canonical.hex", golden);
        $display("loaded real vectors: weights=200 img=784 golden=6272");

        // version register sanity
        axi_read(32'h0000, rd);
        check("VERSION == 0x00020001", rd, 32'h00020001);

        // load image + weights through AXI once (persist across frames)
        load_image();
        load_weights();
        check("image/weight AXI load complete", 1, 1);

        //==================================================================
        // T1 — OS dense
        //==================================================================
        $display("[---] T1: OS dense via AXI");
        run_config("T1 OS", 1'b0, 1'b0, 9184);

        //==================================================================
        // T2 — WS sparse (coarse zero-group skip enabled)
        //==================================================================
        $display("[---] T2: WS sparse via AXI");
        run_config("T2 WS-sparse", 1'b1, 1'b0, 18368);

        //==================================================================
        // T3 — WS dense (sparsity disabled) — measured no-skip bound
        //==================================================================
        $display("[---] T3: WS dense (sparsity_disable=1) via AXI");
        run_config("T3 WS-dense", 1'b1, 1'b1, 38528);
        axi_write(ADDR_SPARSITY_DIS, 32'd0);   // restore

        //==================================================================
        // T4 — runtime OS -> WS -> OS switch (no reset, no reprogram)
        //==================================================================
        $display("[---] T4: runtime OS->WS->OS via AXI (no reset)");
        run_config("T4 OS-first", 1'b0, 1'b0, 9184);
        run_config("T4 WS-after-switch", 1'b1, 1'b0, 18368);
        run_config("T4 OS-after-switch", 1'b0, 1'b0, 9184);

        //==================================================================
        // T5 — counters
        //==================================================================
        $display("[---] T5: MAC / zero-skip counters via AXI");
        begin
            logic [31:0] total, exec, skip;
            axi_read(ADDR_TOTAL_MACS, total);
            axi_read(ADDR_EXECUTED,  exec);
            axi_read(ADDR_SKIPPED,   skip);
            check("T5: total_macs == 156800", total, 156800);
            check("T5: executed+skipped == total", exec + skip, 156800);
            check("T5: skipped_macs == 133960", skip, 133960);
        end

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
            $fatal(1, "tb_cnn_top_axi: %0d of %0d checks failed", n_fail, n_pass + n_fail);
        end
    end

endmodule
