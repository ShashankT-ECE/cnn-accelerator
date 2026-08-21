//==============================================================================
// tb_pe_v2.sv — Processing Element V2 unit testbench
//
// Self-checking, directed, non-UVM testbench for rtl/common/pe_v2.sv.
// Verifies the finalized PE-v2 contract (docs/specs/PE_SPEC.md §5-§8 + §13 and
// docs/PROJECT_STATE.md Decision 10) WITHOUT modifying the frozen V1 PE
// (rtl/common/pe.sv) or the V1 testbench (sim/tb_pe.sv).
//
// Run (Vivado ML 2023.1):
//   source ~/Xilinx/Vivado/2023.1/settings64.sh
//   xvlog -sv rtl/common/pe.sv rtl/common/pe_v2.sv sim/tb_pe_v2.sv
//   xelab -debug typical -s tb_pe_v2_snapshot tb_pe_v2
//   xsim tb_pe_v2_snapshot -runall
//
// Reference model is independent of the DUT: expected values are computed from
// the documented arithmetic (OS: acc += a*w; WS: acc = psum_in + a*w; chain:
// psum_out[i] = SUM_{j<=i} a_j*w_j), never from DUT-internal signals. The V1 PE
// (rtl/common/pe.sv) is instantiated alongside pe_v2 and driven with identical
// OS-mode stimuli to verify bit-exact OS equivalence.
//==============================================================================

module tb_pe_v2;

    //--------------------------------------------------------------------------
    // Clock (10 ns period)
    //--------------------------------------------------------------------------
    logic clk = 0;
    initial forever #5 clk = ~clk;

    //--------------------------------------------------------------------------
    // Single-DUT signals
    //--------------------------------------------------------------------------
    logic               rst            = 0;
    logic signed [7:0]  activation_in  = 0;
    logic signed [7:0]  weight_in      = 0;
    logic               weight_load    = 0;
    logic               zero_skip      = 0;
    logic               result_request = 0;
    logic               accum_clear    = 0;
    logic               dataflow_mode  = 0;
    logic signed [31:0] psum_in        = 0;
    logic signed [31:0] result_out;
    logic signed [7:0]  act_out;
    logic signed [31:0] psum_out;

    //--------------------------------------------------------------------------
    // V1 reference PE (rtl/common/pe.sv) signals — for OS-equivalence
    //--------------------------------------------------------------------------
    wire                v1_rst;
    wire signed [7:0]   v1_activation_in;
    wire signed [7:0]   v1_weight_in;
    wire                v1_weight_load;
    wire                v1_zero_skip;
    wire                v1_result_request;
    wire                v1_accum_clear;
    logic signed [31:0] v1_result_out;
    logic signed [7:0]  v1_act_out;

    //--------------------------------------------------------------------------
    // 8-PE WS chain signals (chained psum cascade)
    //--------------------------------------------------------------------------
    localparam int CHAIN_N = 8;
    logic               c_rst            = 0;
    logic signed [7:0]  c_activation_in [0:CHAIN_N-1];
    logic signed [7:0]  c_weight_in     [0:CHAIN_N-1];
    logic               c_weight_load    = 0;
    logic               c_zero_skip      = 0;
    logic               c_result_request = 0;
    logic               c_accum_clear    = 0;
    logic               c_dataflow_mode  = 1;   // WS
    logic signed [31:0] c_psum_in       [0:CHAIN_N-1];
    logic signed [31:0] c_result_out    [0:CHAIN_N-1];
    logic signed [7:0]  c_act_out       [0:CHAIN_N-1];
    logic signed [31:0] c_psum_out      [0:CHAIN_N-1];

    // Chain weight / activation stimulus (set by the chain tests)
    logic signed [7:0]  chain_w [0:CHAIN_N-1];
    logic signed [7:0]  chain_a [0:CHAIN_N-1];

    //--------------------------------------------------------------------------
    // Scoreboard
    //--------------------------------------------------------------------------
    int n_pass = 0;
    int n_fail = 0;

    //--------------------------------------------------------------------------
    // DUT instantiation (single pe_v2)
    //--------------------------------------------------------------------------
    pe_v2 dut (
        .clk           (clk),
        .rst           (rst),
        .activation_in (activation_in),
        .weight_in     (weight_in),
        .weight_load   (weight_load),
        .zero_skip     (zero_skip),
        .result_request(result_request),
        .accum_clear   (accum_clear),
        .dataflow_mode (dataflow_mode),
        .psum_in       (psum_in),
        .result_out    (result_out),
        .act_out       (act_out),
        .psum_out      (psum_out)
    );

    //--------------------------------------------------------------------------
    // V1 reference PE (frozen module) — mirror the single-DUT stimuli so the
    // two PEs see identical inputs and can be compared bit-for-bit.
    //--------------------------------------------------------------------------
    pe v1_ref (
        .clk           (clk),
        .rst           (v1_rst),
        .activation_in (v1_activation_in),
        .weight_in     (v1_weight_in),
        .weight_load   (v1_weight_load),
        .zero_skip     (v1_zero_skip),
        .result_request(v1_result_request),
        .accum_clear   (v1_accum_clear),
        .result_out    (v1_result_out),
        .act_out       (v1_act_out)
    );

    assign v1_rst            = rst;
    assign v1_activation_in  = activation_in;
    assign v1_weight_in      = weight_in;
    assign v1_weight_load    = weight_load;
    assign v1_zero_skip      = zero_skip;
    assign v1_result_request = result_request;
    assign v1_accum_clear    = accum_clear;

    //--------------------------------------------------------------------------
    // 8-PE WS chain: psum_in[0] = 0; psum_in[i] = psum_out[i-1]
    //--------------------------------------------------------------------------
    assign c_psum_in[0] = 32'sd0;
    genvar ci;
    generate
        for (ci = 1; ci < CHAIN_N; ci++) begin : psum_link
            assign c_psum_in[ci] = c_psum_out[ci-1];
        end
    endgenerate

    genvar cj;
    generate
        for (cj = 0; cj < CHAIN_N; cj++) begin : chain
            pe_v2 chain_pe (
                .clk           (clk),
                .rst           (c_rst),
                .activation_in (c_activation_in[cj]),
                .weight_in     (c_weight_in[cj]),
                .weight_load   (c_weight_load),
                .zero_skip     (c_zero_skip),
                .result_request(c_result_request),
                .accum_clear   (c_accum_clear),
                .dataflow_mode (c_dataflow_mode),
                .psum_in       (c_psum_in[cj]),
                .result_out    (c_result_out[cj]),
                .act_out       (c_act_out[cj]),
                .psum_out      (c_psum_out[cj])
            );
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
    // Single-DUT low-level helpers (drive at negedge, sample at posedge).
    // dataflow_mode and psum_in are left untouched by these helpers — each test
    // sets dataflow_mode explicitly and the helpers drive psum_in = 0 where the
    // addend source is irrelevant (OS).
    //--------------------------------------------------------------------------

    // Clear accumulator + product register (one-cycle accum_clear pulse).
    task automatic clear_acc();
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 1; psum_in = 0;
        @(posedge clk); #1;
        @(negedge clk);
        accum_clear = 0;
        @(posedge clk); #1;
    endtask

    // Load weight (one-cycle weight_load pulse). Available for MAC one cycle later.
    task automatic load_weight(input logic signed [7:0] w);
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = w; weight_load = 1;
        zero_skip = 0; result_request = 0; accum_clear = 0; psum_in = 0;
        @(posedge clk); #1;
        @(negedge clk);
        weight_load = 0;
        @(posedge clk); #1;
    endtask

    // Present one activation for a normal MAC cycle (OS: local accumulate).
    task automatic feed(input logic signed [7:0] a);
        @(negedge clk);
        rst = 0; activation_in = a; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0; psum_in = 0;
        @(posedge clk); #1;
    endtask

    // Present one WS cycle: psum_in = p, activation = a (mode must be 1).
    task automatic ws_cycle(input logic signed [31:0] p, input logic signed [7:0] a);
        @(negedge clk);
        rst = 0; activation_in = a; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0; psum_in = p;
        @(posedge clk); #1;
    endtask

    // Advance n idle cycles (flush in-flight product; OS holds, WS passes 0).
    task automatic drain(input int n = 1);
        repeat (n) begin
            @(negedge clk);
            rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
            zero_skip = 0; result_request = 0; accum_clear = 0; psum_in = 0;
            @(posedge clk); #1;
        end
    endtask

    // Read the accumulator via a one-cycle result_request pulse.
    task automatic read_acc(output logic signed [31:0] val);
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 1; accum_clear = 0; psum_in = 0;
        @(posedge clk); #1;
        val = result_out;
        @(negedge clk);
        result_request = 0;
        @(posedge clk); #1;
    endtask

    // Seed a known OS state: clear + load weight + one activation + drain.
    // Afterwards: accumulator = a*w, product = 0, weight = w.
    task automatic seed(input logic signed [7:0] w, input logic signed [7:0] a);
        clear_acc();
        load_weight(w);
        feed(a);
        drain(1);
    endtask

    // Single-MAC convenience (OS): verify result_out == w*a.
    task automatic single_mac(input logic signed [7:0] w,
                              input logic signed [7:0] a,
                              longint expected);
        logic signed [31:0] v;
        seed(w, a);
        read_acc(v);
        check($sformatf("single MAC %0d x %0d", $signed(w), $signed(a)),
              $signed(v), expected);
    endtask

    //--------------------------------------------------------------------------
    // Chain helper: reset, accum_clear, load chain_w into the chain PEs, then
    // drive chain_a for `cycles` cycles to let the cascade settle. Assumes
    // chain_w[] and chain_a[] are set by the caller. Leaves c_dataflow_mode = 1.
    //--------------------------------------------------------------------------
    task automatic chain_run(input int cycles = 20);
        // reset
        @(negedge clk);
        c_rst = 1;
        for (int i = 0; i < CHAIN_N; i++) begin c_activation_in[i] = '0; c_weight_in[i] = '0; end
        c_weight_load = 0; c_zero_skip = 0; c_result_request = 0; c_accum_clear = 0;
        c_dataflow_mode = 1;
        @(posedge clk); #1;
        @(negedge clk);
        @(posedge clk); #1;
        @(negedge clk);
        c_rst = 0;
        @(posedge clk); #1;

        // accum_clear (clears all chain accumulators + products)
        @(negedge clk);
        c_accum_clear = 1;
        @(posedge clk); #1;
        @(negedge clk);
        c_accum_clear = 0;
        @(posedge clk); #1;

        // load weights
        @(negedge clk);
        for (int i = 0; i < CHAIN_N; i++) c_weight_in[i] = chain_w[i];
        c_weight_load = 1;
        @(posedge clk); #1;
        @(negedge clk);
        c_weight_load = 0;
        @(posedge clk); #1;

        // drive activations (constant) to fill the cascade
        repeat (cycles) begin
            @(negedge clk);
            for (int i = 0; i < CHAIN_N; i++) c_activation_in[i] = chain_a[i];
            @(posedge clk); #1;
        end
    endtask

    //==========================================================================
    // T0 — Synchronous active-high reset (PE_SPEC §10.1, §5.7 priority 1)
    //==========================================================================
    task automatic test_reset();
        logic signed [31:0] v;
        $display("[---] T0: synchronous active-high reset");

        // Establish a non-zero state (OS).
        dataflow_mode = 0;
        seed(5, 37);            // accumulator = 185
        read_acc(v);
        check("T0 pre-reset accumulator = 185", $signed(v), 185);

        // Assert reset with all controls high and non-zero inputs.
        @(negedge clk);
        rst = 1; activation_in = 37; weight_in = 7; weight_load = 1;
        zero_skip = 1; result_request = 1; accum_clear = 1; psum_in = 99;
        @(posedge clk); #1;
        check("T0 result_out = 0 after reset (rst > result_request)", $signed(result_out), 0);
        check("T0 psum_out = 0 after reset", $signed(psum_out), 0);

        // Deassert reset, clear all controls.
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0; psum_in = 0;
        @(posedge clk); #1;

        // Weight cleared by reset: a MAC with no loaded weight yields 0.
        feed(37);
        drain(1);
        read_acc(v);
        check("T0 weight cleared by reset (MAC = 0)", $signed(v), 0);

        // Clean recovery.
        seed(5, 37);
        read_acc(v);
        check("T0 clean MAC after reset = 185", $signed(v), 185);
    endtask

    //==========================================================================
    // T1 — Weight loading and retention (PE_SPEC §5.5, §10.4)
    //==========================================================================
    task automatic test_weight_load();
        logic signed [31:0] v;
        $display("[---] T1: weight loading and retention");

        dataflow_mode = 0;
        seed(3, 2);             // accum = 6
        feed(4);
        feed(6);
        drain(1);
        read_acc(v);
        check("T1 weight retained: 3*(2+4+6) = 36", $signed(v), 36);

        clear_acc();
        load_weight(7);
        feed(1);
        feed(2);
        drain(1);
        read_acc(v);
        check("T1 weight reload: 7*(1+2) = 21", $signed(v), 21);

        // Weight loading not gated by zero_skip.
        clear_acc();
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 9; weight_load = 1;
        zero_skip = 1; result_request = 0; accum_clear = 0; psum_in = 0;
        @(posedge clk); #1;    // weight <= 9 despite zero_skip
        @(negedge clk);
        weight_load = 0; zero_skip = 0;
        @(posedge clk); #1;
        feed(5);
        drain(1);
        read_acc(v);
        check("T1 weight load during zero_skip: 9*5 = 45", $signed(v), 45);
    endtask

    //==========================================================================
    // T2 — OS mode basic MAC (P = P + A*B)
    //==========================================================================
    task automatic test_os_basic();
        $display("[---] T2: OS mode basic MAC");
        dataflow_mode = 0;
        single_mac(3, 5, 15);
        single_mac(-3, 5, -15);
    endtask

    //==========================================================================
    // T3 — OS mode accumulation (P = P + A*B over many cycles)
    //==========================================================================
    task automatic test_os_accum();
        logic signed [31:0] v;
        $display("[---] T3: OS mode accumulation");
        dataflow_mode = 0;
        clear_acc();
        load_weight(2);
        for (int i = 1; i <= 8; i++) feed(i);
        drain(1);
        read_acc(v);
        check("T3 OS accum: 2*(1+..+8) = 72", $signed(v), 72);
    endtask

    //==========================================================================
    // T4 — WS mode with psum_in (P = psum_in + A*B)
    //==========================================================================
    task automatic test_ws_basic();
        $display("[---] T4: WS mode with psum_in");
        dataflow_mode = 1;
        clear_acc();
        load_weight(3);
        ws_cycle(100, 5);       // product <= 15, acc <= 100 + 0 = 100
        ws_cycle(100, 5);       // product <= 15, acc <= 100 + 15 = 115
        check("T4 WS: psum_out = 100 + 5*3 = 115", $signed(psum_out), 115);
    endtask

    //==========================================================================
    // T5 — psum_out forwarding (= registered accumulator, tracks psum_in)
    //==========================================================================
    task automatic test_psum_forward();
        $display("[---] T5: psum_out forwarding");
        dataflow_mode = 1;
        clear_acc();
        load_weight(2);
        ws_cycle(10, 3);        // acc <= 10 + 0 = 10
        ws_cycle(10, 3);        // acc <= 10 + 6 = 16
        check("T5 psum_out = 10 + 3*2 = 16", $signed(psum_out), 16);
        ws_cycle(50, 3);        // acc <= 50 + 6 = 56
        check("T5 psum_out forwards new psum_in = 50 + 3*2 = 56", $signed(psum_out), 56);
    endtask

    //==========================================================================
    // T6 — Two-PE WS chain (psum_in[1] = psum_out[0])
    //==========================================================================
    task automatic test_ws_chain2();
        $display("[---] T6: two-PE WS chain");
        for (int i = 0; i < CHAIN_N; i++) begin
            chain_w[i] = i + 1;
            chain_a[i] = i + 1;
        end
        chain_run();
        // expected: psum_out[i] = SUM_{j<=i} (j+1)^2
        check("T6 chain psum_out[0] = 1",        $signed(c_psum_out[0]), 1);
        check("T6 chain psum_out[1] = 1+4 = 5",  $signed(c_psum_out[1]), 5);
    endtask

    //==========================================================================
    // T7 — Multi-PE chained accumulation (8 stages, signed mix)
    //==========================================================================
    task automatic test_ws_chain8();
        logic signed [31:0] run;
        $display("[---] T7: 8-PE chained accumulation (signed mix)");
        chain_w = '{ 2, -3,  4, -5,  1,  7, -2,  6 };
        chain_a = '{ 3,  1, -2,  4, -1,  2,  5, -3 };
        chain_run();
        run = 0;
        for (int i = 0; i < CHAIN_N; i++) begin
            run = run + $signed(chain_a[i]) * $signed(chain_w[i]);
            check($sformatf("T7 chain psum_out[%0d]", i),
                  $signed(c_psum_out[i]), $signed(run));
        end
    endtask

    //==========================================================================
    // T8 — Signed 8x8 arithmetic incl. corner cases (PE_SPEC §10.7/§10.8)
    //==========================================================================
    task automatic test_signed();
        $display("[---] T8: signed 8x8 arithmetic (mixed sign + corners)");
        dataflow_mode = 0;
        single_mac(  3,   5,      15);
        single_mac(  3,  -5,     -15);
        single_mac( -3,   5,     -15);
        single_mac( -3,  -5,      15);
        single_mac(-128, -128,  16384);
        single_mac(-128,  127, -16256);
        single_mac( 127, -128, -16256);
        single_mac( 127,  127,  16129);
        single_mac(   5,    0,      0);
        single_mac(   0,    5,      0);
        single_mac(   0,    0,      0);
    endtask

    //==========================================================================
    // T9 — 32-bit accumulation (exceeds 16-bit product range)
    //==========================================================================
    task automatic test_accum32();
        logic signed [31:0] v;
        $display("[---] T9: 32-bit accumulation");
        dataflow_mode = 0;
        clear_acc();
        load_weight(127);
        repeat (10) feed(127);
        drain(1);
        read_acc(v);
        check("T9 32-bit accum: 10*16129 = 161290", $signed(v), 161290);

        clear_acc();
        load_weight(-128);
        repeat (3) feed(127);
        drain(1);
        read_acc(v);
        check("T9 32-bit negative: 3*(-16256) = -48768", $signed(v), -48768);
    endtask

    //==========================================================================
    // T10 — accum_clear flushes accumulator AND product (both modes)
    //==========================================================================
    task automatic test_accum_clear();
        logic signed [31:0] v;
        $display("[---] T10: accum_clear flushes accumulator + product");

        // OS mode: no stale +6 add-back.
        dataflow_mode = 0;
        clear_acc();
        load_weight(3);
        feed(2);                // product = 6 (in flight)
        feed(4);                // product = 12, accumulator = 6
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 1; psum_in = 0;
        @(posedge clk); #1;     // accumulator <= 0, product <= 0
        @(negedge clk);
        accum_clear = 0;
        @(posedge clk); #1;
        read_acc(v);
        check("T10 OS clear -> accumulator = 0", $signed(v), 0);
        feed(5);
        drain(1);
        read_acc(v);
        check("T10 OS pipeline flush: next MAC = 15 (no stale +12)", $signed(v), 15);

        // WS mode: clear resets psum_out to 0.
        dataflow_mode = 1;
        load_weight(4);
        ws_cycle(20, 3);
        ws_cycle(20, 3);
        check("T10 WS pre-clear psum_out = 32", $signed(psum_out), 32);
        clear_acc();
        check("T10 WS clear -> psum_out = 0", $signed(psum_out), 0);
    endtask

    //==========================================================================
    // T11 — zero_skip gates the MAC; forwarding continues (both modes)
    //==========================================================================
    task automatic test_zero_skip();
        logic signed [31:0] v;
        $display("[---] T11: zero_skip gates MAC, forwarding continues");

        // OS mode: poison activation 100 skipped.
        dataflow_mode = 0;
        clear_acc();
        load_weight(3);
        feed(2);                // contributes 6
        feed(4);                // contributes 12
        @(negedge clk);
        rst = 0; activation_in = 100; weight_in = 0; weight_load = 0;
        zero_skip = 1; result_request = 0; accum_clear = 0; psum_in = 0;
        #1;
        check("T11 act_out forwarding during zero_skip (=100)", $signed(act_out), 100);
        @(posedge clk); #1;
        @(negedge clk);
        zero_skip = 0; activation_in = 0;
        @(posedge clk); #1;
        feed(5);                // contributes 15
        drain(1);
        read_acc(v);
        check("T11 OS zero_skip excludes poison: 3*(2+4+5) = 33", $signed(v), 33);

        // WS mode: zero_skip -> psum passes through unchanged (psum_in + 0).
        dataflow_mode = 1;
        clear_acc();
        load_weight(3);
        @(negedge clk);
        rst = 0; activation_in = 100; weight_in = 0; weight_load = 0;
        zero_skip = 1; result_request = 0; accum_clear = 0; psum_in = 44;
        @(posedge clk); #1;     // product <= 0 (skip)
        @(negedge clk);
        zero_skip = 0; activation_in = 0;
        @(posedge clk); #1;     // accumulator <= psum_in(44) + product(0) = 44
        check("T11 WS zero_skip passes psum through = 44", $signed(psum_out), 44);
    endtask

    //==========================================================================
    // T12 — result_request/read: read-without-clear, level-sensitive
    //==========================================================================
    task automatic test_result_request();
        logic signed [31:0] v;
        $display("[---] T12: result_request/read behavior");

        dataflow_mode = 0;
        clear_acc();
        load_weight(2);
        feed(1); feed(2); feed(3);
        drain(1);
        read_acc(v);
        check("T12 result_request captures 12", $signed(v), 12);

        read_acc(v);
        check("T12 read-without-clear (re-read = 12)", $signed(v), 12);

        feed(4);
        drain(1);
        read_acc(v);
        check("T12 accumulator preserved after reads: 12+8 = 20", $signed(v), 20);
    endtask

    //==========================================================================
    // T13 — Back-to-back operations
    //==========================================================================
    task automatic test_back_to_back();
        logic signed [31:0] v;
        $display("[---] T13: back-to-back operations");

        dataflow_mode = 0;
        clear_acc();
        load_weight(2);
        for (int i = 1; i <= 8; i++) feed(i);
        drain(1);
        read_acc(v);
        check("T13 burst1: 2*(1+..+8) = 72", $signed(v), 72);

        clear_acc();
        load_weight(-3);
        feed(2); feed(-1); feed(4); feed(-2); feed(1);
        drain(1);
        read_acc(v);
        check("T13 burst2: -3*(2-1+4-2+1) = -12", $signed(v), -12);
    endtask

    //==========================================================================
    // T14 — OS-mode equivalence with the V1 PE (bit-for-bit)
    //==========================================================================
    task automatic test_os_equiv();
        logic signed [31:0] v;
        $display("[---] T14: OS-mode equivalence with V1 PE");
        dataflow_mode = 0; psum_in = 0;

        // Reset both.
        @(negedge clk);
        rst = 1; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;
        @(negedge clk);
        rst = 0;
        @(posedge clk); #1;
        check("T14 reset: result_out == v1", $signed(result_out), $signed(v1_result_out));
        check("T14 reset: act_out == v1", $signed(act_out), $signed(v1_act_out));

        // Load + MAC.
        load_weight(5);
        feed(37);
        drain(1);
        check("T14 after MAC: result_out == v1", $signed(result_out), $signed(v1_result_out));
        read_acc(v);
        check("T14 read 5*37 = 185", $signed(v), 185);

        // Signed back-to-back.
        clear_acc();
        load_weight(-3);
        feed(2); feed(-1); feed(4);
        drain(1);
        check("T14 signed: result_out == v1", $signed(result_out), $signed(v1_result_out));
        read_acc(v);
        check("T14 -3*(2-1+4) = -15", $signed(v), -15);

        // zero_skip poison.
        clear_acc();
        load_weight(3);
        feed(2); feed(4);
        @(negedge clk);
        activation_in = 100; zero_skip = 1;
        @(posedge clk); #1;
        @(negedge clk);
        zero_skip = 0; activation_in = 0;
        @(posedge clk); #1;
        feed(5);
        drain(1);
        check("T14 zero_skip: result_out == v1", $signed(result_out), $signed(v1_result_out));
        read_acc(v);
        check("T14 skip poison: 3*(2+4+5) = 33", $signed(v), 33);

        // accum_clear flush.
        clear_acc();
        load_weight(3);
        feed(2); feed(4);       // accumulator = 6, product = 12 in flight
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 1; psum_in = 0;
        @(posedge clk); #1;     // accumulator <= 0, product <= 0
        @(negedge clk);
        accum_clear = 0;
        @(posedge clk); #1;
        read_acc(v);
        check("T14 clear: accumulator = 0", $signed(v), 0);
        check("T14 after clear: result_out == v1", $signed(result_out), $signed(v1_result_out));
    endtask

    //==========================================================================
    // T15 — Safe OS<->WS mode switching at a group/boundary
    //==========================================================================
    task automatic test_mode_switch();
        logic signed [31:0] v;
        $display("[---] T15: safe OS<->WS mode switching at a boundary");

        // OS phase.
        dataflow_mode = 0;
        clear_acc();
        load_weight(3);
        feed(2); feed(4);
        drain(1);
        read_acc(v);
        check("T15 OS phase = 3*(2+4) = 18", $signed(v), 18);

        // Boundary: clear + switch to WS.
        dataflow_mode = 1;
        clear_acc();
        load_weight(2);
        ws_cycle(7, 5);
        ws_cycle(7, 5);
        check("T15 WS phase = 7 + 5*2 = 17 (no stale OS state)", $signed(psum_out), 17);

        // Boundary: clear + switch back to OS.
        dataflow_mode = 0;
        clear_acc();
        load_weight(3);
        feed(2);
        drain(1);
        read_acc(v);
        check("T15 OS again = 6 (no stale WS state)", $signed(v), 6);
    endtask

    //==========================================================================
    // Test runner
    //==========================================================================
    initial begin
        // Global synchronous reset before any stimulus.
        rst = 1;
        activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        dataflow_mode = 0; psum_in = 0;
        repeat (2) @(posedge clk);
        #1;
        rst = 0;
        @(posedge clk); #1;

        test_reset();
        test_weight_load();
        test_os_basic();
        test_os_accum();
        test_ws_basic();
        test_psum_forward();
        test_ws_chain2();
        test_ws_chain8();
        test_signed();
        test_accum32();
        test_accum_clear();
        test_zero_skip();
        test_result_request();
        test_back_to_back();
        test_os_equiv();
        test_mode_switch();

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
            $fatal(1, "tb_pe_v2: %0d of %0d checks failed", n_fail, n_pass + n_fail);
        end
    end

endmodule
