`timescale 1ns/1ps
// tb_feas_row — self-checking xsim testbench for the EXPLORATORY feas_row (V3 Phase 0, WS5).
// Drives NSEQ accumulation sequences (corner sequences first, then random K in [W, KMAX] with random
// bubbles), computes the expected INT32 sums in the TB, and checks every drained column.
// Prints "FEAS_TB tb_feas_row W=.. PACK=.. macs=.. seqs=.. errors=.. PASS|FAIL"; $fatal on failure.
module tb_feas_row;
    parameter int W     = 16;
    parameter int PACK  = 0;
    parameter int NSEQ  = 400;
    parameter int KMAX  = 600;      // >= largest K of the V3 nets (ResNet-20: 64*3*3 = 576)
    parameter int SEED  = 1;
    localparam int ACC_W = 32;

    logic clk = 0, rst = 1;
    always #2 clk = ~clk;

    logic signed [7:0]       a;
    logic [W-1:0][7:0]       w;
    logic                    in_valid, in_first, in_last;
    logic                    drain_valid;
    logic signed [ACC_W-1:0] drain_out;

    feas_row #(.W(W), .PACK(PACK), .ACC_W(ACC_W)) dut (.*);

    longint exp_q [$];             // expected values, column order, all sequences
    longint macs = 0;
    int     errors = 0, checked = 0;

    // checker
    always @(posedge clk) begin
        if (!rst && drain_valid) begin
            longint e;
            if (exp_q.size() == 0) begin
                errors++;
                $display("ERROR: unexpected drain value %0d", drain_out);
            end else begin
                e = exp_q.pop_front();
                if (longint'(drain_out) !== e) begin
                    errors++;
                    if (errors < 20) $display("ERROR: drain %0d got %0d exp %0d", checked, drain_out, e);
                end
                checked++;
            end
        end
    end

    function automatic logic signed [7:0] pick(int mode, int k, int c);
        case (mode)
            1: return -8'sd128;
            2: return (c % 2) ? 8'sd127 : -8'sd128;
            3: return ((k + c) % 2) ? 8'sd127 : -8'sd128;
            default: return 8'($urandom());
        endcase
    endfunction

    task automatic run_seq(int K, int amode, int wmode, int bubble_pct);
        longint sum [W];
        for (int c = 0; c < W; c++) sum[c] = 0;
        for (int k = 0; k < K; k++) begin
            while (bubble_pct > 0 && ($urandom() % 100) < bubble_pct) begin
                @(negedge clk);
                in_valid = 0; in_first = 0; in_last = 0;
                a = 8'($urandom()); w = {W{8'($urandom())}};
            end
            @(negedge clk);
            a = pick(amode, k, 0);
            for (int c = 0; c < W; c++) begin
                w[c] = pick(wmode, k, c);
                sum[c] += longint'(a) * longint'($signed(w[c]));
            end
            in_valid = 1; in_first = (k == 0); in_last = (k == K - 1);
            macs += W;
        end
        for (int c = 0; c < W; c++) exp_q.push_back(longint'($signed(32'(sum[c]))));
    endtask

    initial begin
        process::self().srandom(SEED);
        a = 0; w = '0; in_valid = 0; in_first = 0; in_last = 0;
        repeat (5) @(negedge clk);
        rst = 0;
        // corner sequences: extreme products, longest K
        run_seq(KMAX, 1, 1, 0);       // (-128)*(-128) every step
        run_seq(KMAX, 1, 2, 0);       // -128 * {-128, 127}
        run_seq(KMAX, 3, 3, 0);       // alternating extremes
        run_seq(W, 0, 0, 0);          // shortest supported K
        for (int i = 0; i < NSEQ; i++)
            run_seq(W + ($urandom() % (KMAX - W + 1)), 0, 0, (i % 4 == 0) ? 20 : 0);
        @(negedge clk);
        in_valid = 0; in_first = 0; in_last = 0;
        repeat (W + 20) @(negedge clk);
        if (exp_q.size() != 0) begin
            errors++;
            $display("ERROR: %0d expected values never drained", exp_q.size());
        end
        $display("FEAS_TB tb_feas_row W=%0d PACK=%0d macs=%0d seqs=%0d checked=%0d errors=%0d %s",
                 W, PACK, macs, NSEQ + 4, checked, errors, (errors == 0) ? "PASS" : "FAIL");
        if (errors != 0) $fatal(1, "tb_feas_row FAIL");
        $finish;
    end
endmodule
