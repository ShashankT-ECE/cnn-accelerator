//==============================================================================
// tb_pe.sv — Processing Element (PE) V1 unit testbench
//
// Self-checking, directed, non-UVM testbench for rtl/common/pe.sv.
// Verifies the finalized V1 PE contract (docs/specs/PE_SPEC.md §5-§10 and
// docs/PROJECT_STATE.md Decisions 1-6) WITHOUT modifying the PE or the spec.
//
// Run (Vivado ML 2023.1):
//   source ~/Xilinx/Vivado/2023.1/settings64.sh
//   xvlog -sv rtl/common/pe.sv sim/tb_pe.sv
//   xelab -debug typical -s pe_tb_snapshot tb_pe
//   xsim pe_tb_snapshot -runall
//
// Timing contract under test (PE_SPEC.md §7.8, Decision 6):
//   - 2-cycle MAC latency (activation_in -> accumulator update)
//   - weight_load -> weight available for MAC = 1 cycle
//   - result_request -> result_out valid = 1 cycle (level-sensitive)
//   - accum_clear -> accumulator/product = 0 = 1 cycle
//   - act_out = activation_in (combinational, 0-cycle)
//   - control priority: rst > weight_load > accum_clear > result_request >
//     zero_skip > normal MAC (PE_SPEC.md §5.7)
//==============================================================================

module tb_pe;

    //--------------------------------------------------------------------------
    // DUT signals
    //--------------------------------------------------------------------------
    logic               clk             = 0;
    logic               rst             = 0;
    logic signed [7:0]  activation_in   = 0;
    logic signed [7:0]  weight_in       = 0;
    logic               weight_load     = 0;
    logic               zero_skip       = 0;
    logic               result_request  = 0;
    logic               accum_clear     = 0;
    logic signed [31:0] result_out;
    logic signed [7:0]  act_out;

    //--------------------------------------------------------------------------
    // Scoreboard
    //--------------------------------------------------------------------------
    int n_pass = 0;
    int n_fail = 0;

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
    pe dut (
        .clk            (clk),
        .rst            (rst),
        .activation_in  (activation_in),
        .weight_in      (weight_in),
        .weight_load    (weight_load),
        .zero_skip      (zero_skip),
        .result_request (result_request),
        .accum_clear    (accum_clear),
        .result_out     (result_out),
        .act_out        (act_out)
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
    // Low-level helpers. Each drives all controls at the negedge and advances
    // one posedge, leaving all DUT inputs in a known (deasserted) state.
    //--------------------------------------------------------------------------

    // Clear accumulator + product register (one-cycle accum_clear pulse).
    task automatic clear_acc();
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 1;
        @(posedge clk); #1;
        @(negedge clk);
        accum_clear = 0;
        @(posedge clk); #1;
    endtask

    // Load weight (one-cycle weight_load pulse). New weight is available for
    // MAC one cycle later (nonblocking semantics).
    task automatic load_weight(input logic signed [7:0] w);
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = w; weight_load = 1;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;
        @(negedge clk);
        weight_load = 0;
        @(posedge clk); #1;
    endtask

    // Present one activation for a normal MAC cycle.
    task automatic feed(input logic signed [7:0] a);
        @(negedge clk);
        rst = 0; activation_in = a; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;
    endtask

    // Advance n cycles with no new activation (flushes in-flight product into
    // the accumulator without adding a new contribution).
    task automatic drain(input int n = 1);
        repeat (n) begin
            @(negedge clk);
            rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
            zero_skip = 0; result_request = 0; accum_clear = 0;
            @(posedge clk); #1;
        end
    endtask

    // Read the accumulator via a one-cycle result_request pulse
    // (read-without-clear; the accumulator is left unchanged).
    task automatic read_acc(output logic signed [31:0] val);
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 1; accum_clear = 0;
        @(posedge clk); #1;
        val = result_out;
        @(negedge clk);
        result_request = 0;
        @(posedge clk); #1;
    endtask

    // Seed a known state: clear + load weight + one activation + drain.
    // Afterwards: accumulator = a*w, product = 0, weight = w.
    task automatic seed(input logic signed [7:0] w, input logic signed [7:0] a);
        clear_acc();
        load_weight(w);
        feed(a);
        drain(1);
    endtask

    // Single-MAC convenience: verify result_out == w*a after clear/load/feed/drain.
    task automatic single_mac(input logic signed [7:0] w,
                              input logic signed [7:0] a,
                              longint expected);
        logic signed [31:0] v;
        seed(w, a);
        read_acc(v);
        check($sformatf("single MAC %0d x %0d", $signed(w), $signed(a)),
              $signed(v), expected);
    endtask

    //==========================================================================
    // T0 — Synchronous active-high reset (PE_SPEC §10.1, §5.7 priority 1)
    //==========================================================================
    task automatic test_reset();
        logic signed [31:0] v;

        // Establish a non-zero state, then reset over it.
        seed(5, 37);            // accumulator = 185, weight = 5
        read_acc(v);
        check("T0 pre-reset accumulator = 185", $signed(v), 185);

        // Assert reset for one cycle with ALL controls high and non-zero inputs.
        @(negedge clk);
        rst = 1; activation_in = 37; weight_in = 7; weight_load = 1;
        zero_skip = 1; result_request = 1; accum_clear = 1;
        #1;
        check("T0 act_out during reset (combinational, =37)", $signed(act_out), 37);
        @(posedge clk); #1;
        check("T0 result_out = 0 after reset (rst > result_request)", $signed(result_out), 0);

        // Deassert reset, clear all controls.
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;

        // Weight was cleared by reset: a MAC with no loaded weight yields 0.
        feed(37);
        drain(1);
        read_acc(v);
        check("T0 weight cleared by reset (MAC = 0)", $signed(v), 0);

        // Accumulator cleared by reset: load 5, feed 37 -> exactly 185.
        seed(5, 37);
        read_acc(v);
        check("T0 clean MAC after reset = 185", $signed(v), 185);
    endtask

    //==========================================================================
    // T1 — Weight loading and retention (PE_SPEC §5.5, §10.4)
    //==========================================================================
    task automatic test_weight_load_retain();
        logic signed [31:0] v;

        // Load W=3, feed three activations -> weight held stationary.
        seed(3, 2);             // accum = 6
        feed(4);                // product = 12 (in flight)
        feed(6);                // accum = 18, product = 18
        drain(1);               // accum = 36
        read_acc(v);
        check("T1 weight retained: 3*(2+4+6) = 36", $signed(v), 36);

        // Reload W=7: accumulation switches to the new weight.
        clear_acc();
        load_weight(7);
        feed(1);
        feed(2);
        drain(1);
        read_acc(v);
        check("T1 weight reload: 7*(1+2) = 21", $signed(v), 21);

        // Weight loading is not gated by zero_skip (PE_SPEC §5.5).
        clear_acc();
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 9; weight_load = 1;
        zero_skip = 1; result_request = 0; accum_clear = 0;
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
    // T2 — Signed 8-bit x signed 8-bit arithmetic + corner cases
    //      (PE_SPEC §10.7, §10.8)
    //==========================================================================
    task automatic test_signed_corner();
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
    // T3 — Full 16-bit product accumulation into 32-bit accumulator (PE_SPEC §7.4-§7.5)
    //==========================================================================
    task automatic test_32bit_accum();
        logic signed [31:0] v;

        // 10 x (127*127 = 16129) = 161290, exceeding 16-bit range, held in 32-bit.
        clear_acc();
        load_weight(127);
        repeat (10) feed(127);
        drain(1);
        read_acc(v);
        check("T3 32-bit positive accum: 10*16129 = 161290", $signed(v), 161290);

        // Negative products accumulate below -32768 (sign extension into 32-bit).
        clear_acc();
        load_weight(-128);
        repeat (3) feed(127);   // 3 * (127*-128 = -16256) = -48768
        drain(1);
        read_acc(v);
        check("T3 32-bit negative accum: 3*(-16256) = -48768", $signed(v), -48768);
    endtask

    //==========================================================================
    // T4 — Repeated MAC operations (PE_SPEC §10.3)
    //==========================================================================
    task automatic test_repeated_mac();
        logic signed [31:0] v;
        clear_acc();
        load_weight(2);
        for (int i = 1; i <= 8; i++) feed(i);
        drain(1);
        read_acc(v);
        check("T4 repeated MAC: 2*(1+..+8) = 72", $signed(v), 72);
    endtask

    //==========================================================================
    // T5 — Combinational activation forwarding (PE_SPEC §10.6, §8.4)
    //==========================================================================
    task automatic test_activation_forwarding();
        int vals[6];
        vals[0] = 0; vals[1] = 1; vals[2] = -1; vals[3] = 127; vals[4] = -128; vals[5] = 5;
        for (int i = 0; i < 6; i++) begin
            @(negedge clk);
            rst = 0; activation_in = vals[i]; weight_in = 0; weight_load = 0;
            zero_skip = 0; result_request = 0; accum_clear = 0;
            #1;
            check($sformatf("T5 act_out == activation_in (%0d)", vals[i]),
                  $signed(act_out), vals[i]);
            @(posedge clk); #1;
        end
    endtask

    //==========================================================================
    // T6 — zero_skip: prevents the current MAC contribution, forwarding continues
    //      (PE_SPEC §9, §10.10)
    //==========================================================================
    task automatic test_zero_skip();
        logic signed [31:0] v;
        clear_acc();
        load_weight(3);
        feed(2);            // contributes 6
        feed(4);            // contributes 12

        // zero_skip cycle with a "poison" activation 100.
        @(negedge clk);
        rst = 0; activation_in = 100; weight_in = 0; weight_load = 0;
        zero_skip = 1; result_request = 0; accum_clear = 0;
        #1;
        check("T6 act_out forwarding during zero_skip (=100)", $signed(act_out), 100);
        @(posedge clk); #1;

        // Normal accumulation resumes.
        feed(5);            // contributes 15
        drain(1);
        read_acc(v);
        check("T6 zero_skip excludes poison: 3*(2+4+5) = 33", $signed(v), 33);
    endtask

    //==========================================================================
    // T7 — result_request: capture without clear, level-sensitive (PE_SPEC §5.2, §10.5)
    //==========================================================================
    task automatic test_result_request();
        logic signed [31:0] v;

        clear_acc();
        load_weight(2);
        feed(1); feed(2); feed(3);   // sum = 2*(1+2+3) = 12
        drain(1);
        read_acc(v);
        check("T7 result_request captures 12", $signed(v), 12);

        // Read-without-clear: re-read returns the same value.
        read_acc(v);
        check("T7 read-without-clear (re-read = 12)", $signed(v), 12);

        // Accumulator unchanged by reads: continue accumulating from 12.
        feed(4);
        drain(1);
        read_acc(v);
        check("T7 accumulator preserved after reads: 12+8 = 20", $signed(v), 20);

        // Level-sensitive: hold result_request and verify result_out tracks the
        // accumulator across cycles. State now: accumulator = 20, product = 0, weight = 2.
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 1; accum_clear = 0;
        @(posedge clk); #1;
        check("T7 level-sensitive hold (cycle1 = 20)", $signed(result_out), 20);
        @(negedge clk); // result_request still 1
        @(posedge clk); #1;
        check("T7 level-sensitive hold (cycle2 = 20)", $signed(result_out), 20);

        // While held, feed 4 -> accumulator becomes 28 two cycles later; result_out tracks.
        @(negedge clk);
        activation_in = 4;      // result_request still 1
        @(posedge clk); #1;     // product <= 8, accumulator stays 20
        @(negedge clk);
        activation_in = 0;      // result_request still 1
        @(posedge clk); #1;     // accumulator <= 28, product <= 0
        @(negedge clk);         // result_request still 1
        @(posedge clk); #1;     // result_out <= 28
        check("T7 level-sensitive tracks to 28", $signed(result_out), 28);

        @(negedge clk);
        result_request = 0;
        @(posedge clk); #1;
    endtask

    //==========================================================================
    // T8 — accum_clear: clears accumulator AND product register (pipeline flush)
    //      (PE_SPEC §7.8, §10.13-§10.14)
    //==========================================================================
    task automatic test_accum_clear();
        logic signed [31:0] v;
        clear_acc();
        load_weight(3);
        feed(2);            // product = 6, accumulator = 0
        feed(4);            // product = 12 (in flight), accumulator = 6

        // One-cycle accum_clear.
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 1;
        @(posedge clk); #1;    // accumulator <= 0, product <= 0
        @(negedge clk);
        accum_clear = 0;
        @(posedge clk); #1;

        read_acc(v);
        check("T8 accum_clear -> accumulator = 0", $signed(v), 0);

        // Product register flushed: next MAC is exactly 5*3 = 15 (no stale +12).
        feed(5);
        drain(1);
        read_acc(v);
        check("T8 pipeline flush: next MAC = 15 (no stale add-back)", $signed(v), 15);
    endtask

    //==========================================================================
    // T9 — Control priority: the 8 simultaneous combinations from PE_SPEC §5.7
    //==========================================================================
    task automatic test_control_priority();
        logic signed [31:0] v;

        // Case 1: rst overrides everything (from a non-zero state).
        seed(5, 37);            // accumulator = 185
        @(negedge clk);
        rst = 1; activation_in = 37; weight_in = 7; weight_load = 1;
        zero_skip = 1; result_request = 1; accum_clear = 1;
        @(posedge clk); #1;
        check("T9 rst overrides all (result_out = 0)", $signed(result_out), 0);
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;

        // Case 2: weight_load + accum_clear -> weight updates AND accumulator clears.
        seed(3, 5);             // accumulator = 15, weight = 3
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 7; weight_load = 1;
        zero_skip = 0; result_request = 0; accum_clear = 1;
        @(posedge clk); #1;    // weight <= 7, accumulator <= 0, product <= 0
        @(negedge clk);
        activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;
        read_acc(v);
        check("T9 weight_load+accum_clear: accumulator = 0", $signed(v), 0);
        feed(2);
        drain(1);
        read_acc(v);
        check("T9 weight_load+accum_clear: new weight used (7*2 = 14)", $signed(v), 14);

        // Case 3: weight_load + zero_skip -> weight updates, MAC skipped.
        seed(3, 5);             // accumulator = 15, weight = 3
        @(negedge clk);
        rst = 0; activation_in = 99; weight_in = 7; weight_load = 1;
        zero_skip = 1; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;    // weight <= 7, product <= 0 (skip), accumulator <= 15
        @(negedge clk);
        activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;
        read_acc(v);
        check("T9 weight_load+zero_skip: poison skipped (accum = 15)", $signed(v), 15);
        feed(2);
        drain(1);
        read_acc(v);
        check("T9 weight_load+zero_skip: new weight (15 + 7*2 = 29)", $signed(v), 29);

        // Case 4: weight_load + result_request -> weight updates, result_out = current accum.
        seed(3, 5);             // accumulator = 15, weight = 3
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 7; weight_load = 1;
        zero_skip = 0; result_request = 1; accum_clear = 0;
        @(posedge clk); #1;    // result_out <= 15, weight <= 7
        check("T9 weight_load+result_request: result_out = 15", $signed(result_out), 15);
        @(negedge clk);
        activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;

        // Case 5: accum_clear + zero_skip -> clear wins (accumulator + product = 0).
        seed(3, 5);             // accumulator = 15
        feed(2);                // product = 6 in flight, accumulator = 15
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 1; result_request = 0; accum_clear = 1;
        @(posedge clk); #1;    // accumulator <= 0, product <= 0 (clear wins)
        @(negedge clk);
        activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;
        read_acc(v);
        check("T9 accum_clear+zero_skip: clear wins (accum = 0)", $signed(v), 0);
        feed(3);
        drain(1);
        read_acc(v);
        check("T9 accum_clear+zero_skip: clean next MAC (3*3 = 9)", $signed(v), 9);

        // Case 6: accum_clear + result_request -> result_out = 0 (clear first).
        seed(3, 5);             // accumulator = 15
        @(negedge clk);
        rst = 0; activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 1; accum_clear = 1;
        @(posedge clk); #1;    // result_out <= 0, accumulator <= 0
        check("T9 accum_clear+result_request: result_out = 0", $signed(result_out), 0);
        @(negedge clk);
        activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;

        // Case 7: result_request + zero_skip -> result_out = accum (unchanged), MAC skipped.
        seed(3, 5);             // accumulator = 15
        @(negedge clk);
        rst = 0; activation_in = 99; weight_in = 0; weight_load = 0;
        zero_skip = 1; result_request = 1; accum_clear = 0;
        @(posedge clk); #1;    // result_out <= 15, accumulator <= 15, product <= 0
        check("T9 result_request+zero_skip: result_out = 15", $signed(result_out), 15);
        @(negedge clk);
        activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;

        // Case 8: result_request + MAC -> result_out = accumulator BEFORE this cycle's MAC.
        seed(3, 5);             // accumulator = 15
        feed(2);                // product = 6 in flight, accumulator = 15
        @(negedge clk);
        rst = 0; activation_in = 4; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 1; accum_clear = 0;
        @(posedge clk); #1;    // result_out <= 15 (pre-update); accumulator <= 21; product <= 12
        check("T9 result_request+MAC: captures pre-update accum = 15", $signed(result_out), 15);
        @(negedge clk);
        activation_in = 0; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        @(posedge clk); #1;
    endtask

    //==========================================================================
    // T10 — Back-to-back MAC operations (PE_SPEC §10.11)
    //==========================================================================
    task automatic test_back_to_back();
        logic signed [31:0] v;

        // Burst 1: fixed weight 2, 8 back-to-back activations.
        clear_acc();
        load_weight(2);
        for (int i = 1; i <= 8; i++) feed(i);
        drain(1);
        read_acc(v);
        check("T10 back-to-back burst1: 2*(1+..+8) = 72", $signed(v), 72);

        // Burst 2: reload a negative weight, mixed-sign back-to-back.
        clear_acc();
        load_weight(-3);
        feed(2); feed(-1); feed(4); feed(-2); feed(1);
        drain(1);
        read_acc(v);
        check("T10 back-to-back burst2: -3*(2-1+4-2+1) = -12", $signed(v), -12);
    endtask

    //==========================================================================
    // T11 — Result/output timing per the finalized pipeline contract (PE_SPEC §7.8)
    //==========================================================================
    task automatic test_output_timing();
        logic signed [31:0] v;
        clear_acc();
        load_weight(2);     // weight = 2, accumulator = 0, product = 0

        // act_out is combinational (0-cycle): drive and check in the same cycle.
        @(negedge clk);
        rst = 0; activation_in = 9; weight_in = 0; weight_load = 0;
        zero_skip = 0; result_request = 0; accum_clear = 0;
        #1;
        check("T11 act_out 0-cycle (=9)", $signed(act_out), 9);
        @(posedge clk); #1;    // product <= 9*2 = 18, accumulator <= 0

        // 2-cycle MAC latency: one cycle after presenting 9, accumulator is still 0.
        @(negedge clk);
        activation_in = 0; result_request = 1;
        @(posedge clk); #1;    // result_out <= 0 (pre-update), accumulator <= 18
        check("T11 2-cycle latency: accumulator = 0 @ +1 cycle", $signed(result_out), 0);

        // Two cycles after presenting 9, the accumulator reflects the product.
        @(negedge clk);         // result_request still 1
        @(posedge clk); #1;     // result_out <= 18
        check("T11 2-cycle latency: accumulator = 18 @ +2 cycles", $signed(result_out), 18);

        @(negedge clk);
        result_request = 0; activation_in = 0;
        @(posedge clk); #1;

        // accum_clear -> 0 in one cycle.
        @(negedge clk);
        accum_clear = 1; result_request = 1;
        @(posedge clk); #1;    // accumulator <= 0, result_out <= 0
        check("T11 accum_clear -> 0 in 1 cycle", $signed(result_out), 0);
        @(negedge clk);
        accum_clear = 0; result_request = 0;
        @(posedge clk); #1;
    endtask

    //==========================================================================
    // Test runner
    //==========================================================================
    initial begin
        // Global synchronous reset before any stimulus.
        rst = 1;
        repeat (2) @(posedge clk);
        #1;
        rst = 0;
        @(posedge clk); #1;

        test_reset();
        test_weight_load_retain();
        test_signed_corner();
        test_32bit_accum();
        test_repeated_mac();
        test_activation_forwarding();
        test_zero_skip();
        test_result_request();
        test_accum_clear();
        test_control_priority();
        test_back_to_back();
        test_output_timing();

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
            $fatal(1, "tb_pe: %0d of %0d checks failed", n_fail, n_pass + n_fail);
        end
    end

endmodule
