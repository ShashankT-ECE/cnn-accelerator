`timescale 1ns/1ps
// tb_feas_mem_path — self-checking xsim testbench for the EXPLORATORY feas_mem_path (V3 Phase 0, WS5).
// Preloads the weight URAM, the W ACT banks and the residual buffer with random data, then issues
// NREQ random reads on each stream (back-to-back with random gaps) and checks every output against
// a behavioural reference (weights: word; ACT: lane r = bank (sh+r) mod W at word + (bank < sh);
// requant: v = acc + q_bias + skr, skr = (skip*m_skip + 2^(s_skip-1)) >>> s_skip, q = clip(RNE(v26*m/2^s)),
// ReLU after clip, s = 0 -> 0). Prints "FEAS_TB tb_feas_mem_path W=.. ... PASS|FAIL"; $fatal on failure.
module tb_feas_mem_path;
    parameter int W     = 16;
    parameter int REP   = 2;
    parameter int NREQ  = 20000;
    parameter int NWORD = 512;       // preloaded words per memory
    parameter int SEED  = 7;
    localparam int SHW = $clog2(W);

    logic clk = 0, rst = 1;
    always #2 clk = ~clk;

    logic                     wgt_we, wgt_re, act_we, act_re, res_we, rq_valid, res_en, relu_en;
    logic [11:0]              wgt_waddr, wgt_raddr, act_waddr, act_word, res_waddr, res_raddr;
    logic [W*8-1:0]           wgt_wdata, act_wdata, res_wdata;
    logic [SHW-1:0]           act_sh;
    logic                     w_valid, a_valid, q_valid;
    logic [REP-1:0][W*8-1:0]  w_out, a_out;
    logic [W-1:0][31:0]       acc;
    logic [31:0]              q_bias, m;
    logic [5:0]               s;
    logic signed [15:0]       m_skip;
    logic [3:0]               s_skip;
    logic [W-1:0][7:0]        q_out;

    feas_mem_path #(.W(W), .REP(REP)) dut (.*);

    logic [W*8-1:0] wm [NWORD], am [NWORD], rm [NWORD];
    logic [W*8-1:0] w_exp [$], a_exp [$], q_exp [$];
    int errors = 0, nw = 0, na = 0, nq = 0;

    function automatic logic [7:0] ref_q(longint accv, longint qb, logic signed [7:0] skip, longint ms,
                                         int ss, logic re, longint mm, int sv, logic relu);
        longint sk, skr, v, v26, p, q;
        sk  = longint'(skip) * ms;
        skr = (sk + ((ss == 0) ? 0 : (longint'(1) <<< (ss - 1)))) >>> ss;
        v   = longint'($signed(32'(accv + qb + (re ? skr : 0))));
        v26 = longint'($signed(26'(v)));
        p   = v26 * mm;
        if (sv == 0) return 8'h00;
        q = (p + (longint'(1) <<< (sv - 1)) - 1 + ((p >>> sv) & 1)) >>> sv;
        if (q > 127)  q = 127;
        if (q < -128) q = -128;
        if (relu && q < 0) q = 0;
        return 8'(q);
    endfunction

    // checkers
    always @(posedge clk) if (!rst) begin
        if (w_valid) begin
            logic [W*8-1:0] e = w_exp.pop_front();
            for (int r = 0; r < REP; r++) if (w_out[r] !== e) begin
                errors++; if (errors < 20) $display("ERROR wgt %0d rep %0d", nw, r);
            end
            nw++;
        end
        if (a_valid) begin
            logic [W*8-1:0] e = a_exp.pop_front();
            for (int r = 0; r < REP; r++) if (a_out[r] !== e) begin
                errors++; if (errors < 20) $display("ERROR act %0d rep %0d got %h exp %h", na, r, a_out[r], e);
            end
            na++;
        end
        if (q_valid) begin
            logic [W*8-1:0] e = q_exp.pop_front();
            if (q_out !== e) begin
                errors++; if (errors < 20) $display("ERROR rq %0d got %h exp %h", nq, q_out, e);
            end
            nq++;
        end
    end

    function automatic logic [W*8-1:0] rnd_word();
        logic [W*8-1:0] x;
        for (int i = 0; i < W; i++) x[i*8 +: 8] = 8'($urandom());
        return x;
    endfunction

    initial begin
        process::self().srandom(SEED);
        {wgt_we, wgt_re, act_we, act_re, res_we, rq_valid, res_en, relu_en} = '0;
        {wgt_waddr, wgt_raddr, act_waddr, act_word, res_waddr, res_raddr} = '0;
        {wgt_wdata, act_wdata, res_wdata} = '0;
        act_sh = '0; acc = '0; q_bias = '0; m = '0; s = '0; m_skip = '0; s_skip = '0;
        repeat (4) @(negedge clk);
        rst = 0;
        // preload
        for (int i = 0; i < NWORD; i++) begin
            @(negedge clk);
            wm[i] = rnd_word(); am[i] = rnd_word(); rm[i] = rnd_word();
            wgt_we = 1; wgt_waddr = 12'(i); wgt_wdata = wm[i];
            act_we = 1; act_waddr = 12'(i); act_wdata = am[i];
            res_we = 1; res_waddr = 12'(i); res_wdata = rm[i];
        end
        @(negedge clk);
        wgt_we = 0; act_we = 0; res_we = 0;
        repeat (4) @(negedge clk);
        // random reads, all three streams in parallel
        for (int n = 0; n < NREQ; n++) begin
            @(negedge clk);
            wgt_re = ($urandom() % 8) != 0;
            act_re = ($urandom() % 8) != 0;
            rq_valid = ($urandom() % 8) != 0;
            if (wgt_re) begin
                wgt_raddr = 12'($urandom() % NWORD);
                w_exp.push_back(wm[wgt_raddr]);
            end
            if (act_re) begin
                logic [W*8-1:0] e;
                act_word = 12'($urandom() % (NWORD - 1));
                act_sh   = SHW'($urandom());
                for (int r = 0; r < W; r++) begin
                    int b = (int'(act_sh) + r) % W;
                    int ad = int'(act_word) + ((b < int'(act_sh)) ? 1 : 0);
                    e[r*8 +: 8] = am[ad][b*8 +: 8];
                end
                a_exp.push_back(e);
            end
            if (rq_valid) begin
                logic [W*8-1:0] e;
                int mode;
                mode = $urandom() % 4;
                res_raddr = 12'($urandom() % NWORD);
                m      = {1'b1, 31'($urandom())};
                if (mode == 0) begin                             // fine scale: q sensitive to +-1 in v
                    q_bias = 32'(int'($urandom() % 512) - 256);
                    s      = 6'(32 + ($urandom() % 2));
                end else begin
                    q_bias = 32'(int'($urandom() % (1 << 21)) - (1 << 20));
                    s      = (mode == 1) ? 6'($urandom()) : 6'(30 + ($urandom() % 20));   // 0..63 / 30..49
                end
                if (mode == 0) begin                             // small skip term, rounding visible
                    m_skip = 16'(int'($urandom() % 512) - 256);
                    s_skip = 4'(8 + ($urandom() % 8));
                end else begin
                    m_skip = 16'($urandom());
                    s_skip = 4'($urandom());
                end
                res_en  = $urandom() % 2;
                relu_en = $urandom() % 2;
                for (int l = 0; l < W; l++) begin
                    acc[l] = (mode == 0) ? 32'(int'($urandom() % 512) - 256)
                                         : 32'(int'($urandom() % (1 << 25)) - (1 << 24));
                    e[l*8 +: 8] = ref_q(longint'($signed(acc[l])), longint'($signed(q_bias)),
                                        $signed(rm[res_raddr][l*8 +: 8]), longint'(m_skip), int'(s_skip),
                                        res_en, longint'(m), int'(s), relu_en);
                end
                q_exp.push_back(e);
            end
        end
        @(negedge clk);
        wgt_re = 0; act_re = 0; rq_valid = 0;
        repeat (40) @(negedge clk);
        if (w_exp.size() || a_exp.size() || q_exp.size()) begin
            errors++;
            $display("ERROR: undrained expectations w=%0d a=%0d q=%0d", w_exp.size(), a_exp.size(), q_exp.size());
        end
        $display("FEAS_TB tb_feas_mem_path W=%0d REP=%0d wgt_reads=%0d act_reads=%0d rq_vectors=%0d lanes=%0d errors=%0d %s",
                 W, REP, nw, na, nq, nq * W, errors, (errors == 0) ? "PASS" : "FAIL");
        if (errors != 0) $fatal(1, "tb_feas_mem_path FAIL");
        $finish;
    end
endmodule
