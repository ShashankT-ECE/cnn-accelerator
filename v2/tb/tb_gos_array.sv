// GOS_UNIT_TB: gos_pkg.sv gos_pe.sv gos_array.sv
`timescale 1ns/1ps
// tb_gos_array — self-checking TB for gos_array (non-UVM).
//
// Phase A: v2/model unit vectors unit/array/ (array_a/w/ctl/k/acc.hex). Tiles are driven in order;
//          tiles with K < 8 are followed by (8 - K) idle cycles so no capture overruns a drain.
// Phase B: v2/model unit vectors unit/pe/ (pe_cycles/pe_results.hex): every sequence broadcast to all
//          64 PEs (a on every row, w on every column); every PE must end at pe_results; same padding.
// Phase C: TB reference model, random tiles back-to-back (zero gap) with K in [8, 800], incl. bursts
//          of K = 8 tiles, int8 data with ~15 % extremes (-128 / 127), random tokens; then the same
//          with random in_valid gaps inside and between tiles. Garbage on a/w/first/last/tok while
//          in_valid = 0.
// Phase D: K < 8 back-to-back must raise the sticky err_overrun (stays set until rst); after rst the
//          array works again (short random run).
// Every drained column is checked: value (8 rows), out_col order, token, exact timing
// (column j at in_last cycle + L_ARRAY + j, contiguous), and the number of drained tiles equals the
// number of issued tiles. err_overrun is checked 0 during phases A-C.
module tb_gos_array;
    import gos_pkg::*;

    localparam int TW      = GOS_TOK_W;
    localparam int L_ARRAY = 6;      // must equal gos_array.L_ARRAY
    localparam int K_MIN   = 8;

    logic                       clk = 1'b0;
    logic                       rst;
    logic                       in_valid, in_first, in_last;
    logic [TW-1:0]              in_tok;
    logic [N-1:0][7:0]          a_vec, w_vec;
    logic                       out_valid;
    logic [2:0]                 out_col;
    logic [N-1:0][ACC_W-1:0]    out_acc;
    logic [TW-1:0]              out_tok;
    logic                       err_overrun;

    always #2.5 clk = ~clk;

    gos_array #(.TW(TW)) dut (
        .clk, .rst, .in_valid, .in_first, .in_last, .in_tok, .a_vec, .w_vec,
        .out_valid, .out_col, .out_acc, .out_tok, .err_overrun
    );

    // ---------------------------------------------------------------- expected-tile queue
    typedef struct {
        logic [N-1:0][N-1:0][ACC_W-1:0] acc;   // [r][c]
        logic [TW-1:0]                  tok;
        longint                         t_last; // cycle of in_last
    } tile_t;

    tile_t  exp_q[$];
    longint cyc = 0;
    longint checks = 0;
    longint tiles_issued = 0, tiles_drained = 0;
    int     drain_j = 0;          // next expected column of the front tile
    bit     check_en = 1'b1;      // output checking enabled (off during phase D overrun)
    bit     ovr_must_be_0 = 1'b1;

    task automatic fail(string msg);
        $display("TEST FAILED %s (cycle %0d)", msg, cyc);
        $fatal(1, "TEST FAILED");
    endtask

    // checker (samples just before the rising edge's NBA updates: use negedge sampling)
    always @(negedge clk) begin
        cyc++;
        if (!rst && ovr_must_be_0 && err_overrun !== 1'b0) fail("err_overrun set on legal traffic");
        if (!rst && check_en) begin
            if (out_valid === 1'b1) begin
                tile_t e;
                if (exp_q.size() == 0) fail("out_valid with no tile outstanding");
                e = exp_q[0];
                if (cyc != e.t_last + L_ARRAY + drain_j)
                    fail($sformatf("drain timing: col %0d at cycle %0d, expected %0d",
                                   drain_j, cyc, e.t_last + L_ARRAY + drain_j));
                if (out_col !== 3'(drain_j)) fail($sformatf("out_col %0d expected %0d", out_col, drain_j));
                if (out_tok !== e.tok) fail($sformatf("out_tok %h expected %h", out_tok, e.tok));
                for (int r = 0; r < N; r++) begin
                    if (out_acc[r] !== e.acc[r][drain_j])
                        fail($sformatf("tile %0d col %0d row %0d: acc %0d expected %0d", tiles_drained,
                             drain_j, r, $signed(out_acc[r]), $signed(e.acc[r][drain_j])));
                    checks++;
                end
                checks += 2;   // col + tok
                if (drain_j == N - 1) begin
                    void'(exp_q.pop_front());
                    drain_j = 0;
                    tiles_drained++;
                end else drain_j++;
            end else if (out_valid !== 1'b0) begin
                fail("out_valid is X");
            end else begin
                if (drain_j != 0) fail("drain not contiguous");
                if (exp_q.size() != 0 && cyc >= exp_q[0].t_last + L_ARRAY)
                    fail("expected drain missing");
            end
        end
    end

    // ---------------------------------------------------------------- driver helpers
    // All drives happen right after the rising edge (#0.5); the input cycle number is the cycle
    // counted by the checker at the following negedge.
    logic [N-1:0][N-1:0][ACC_W-1:0] model_acc;

    function automatic logic [7:0] rnd_i8();
        int x = $urandom_range(0, 99);
        if (x < 8)       return 8'h80;   // -128
        else if (x < 15) return 8'h7f;   //  127
        else             return 8'($urandom);
    endfunction

    task automatic idle_cycle();
        @(posedge clk); #0.5;
        in_valid = 1'b0;
        in_first = 1'($urandom); in_last = 1'($urandom); in_tok = TW'($urandom);
        for (int i = 0; i < N; i++) begin a_vec[i] = 8'($urandom); w_vec[i] = 8'($urandom); end
    endtask

    // one valid k; updates the reference model; on last pushes the expected tile
    task automatic drive_k(logic [N-1:0][7:0] a, logic [N-1:0][7:0] w, bit first, bit last,
                           logic [TW-1:0] tok);
        @(posedge clk); #0.5;
        in_valid = 1'b1; in_first = first; in_last = last; in_tok = tok;
        a_vec = a; w_vec = w;
        for (int r = 0; r < N; r++)
            for (int c = 0; c < N; c++) begin
                logic signed [ACC_W-1:0] p;
                p = ACC_W'($signed(a[r])) * ACC_W'($signed(w[c]));
                model_acc[r][c] = first ? p : model_acc[r][c] + p;
            end
        if (last) begin
            tile_t t;
            t.acc = model_acc; t.tok = tok; t.t_last = cyc + 1;   // counted at the next negedge
            exp_q.push_back(t);
            tiles_issued++;
        end
    endtask

    task automatic random_tile(int K, int gap_pct);
        logic [TW-1:0] tok = TW'($urandom);
        for (int k = 0; k < K; k++) begin
            logic [N-1:0][7:0] a, w;
            for (int i = 0; i < N; i++) begin a[i] = rnd_i8(); w[i] = rnd_i8(); end
            if (k > 0) while ($urandom_range(0, 99) < gap_pct) idle_cycle();
            drive_k(a, w, k == 0, k == K - 1, tok);
        end
        while ($urandom_range(0, 99) < gap_pct) idle_cycle();
    endtask

    task automatic drain_all();
        int guard = 0;
        while (exp_q.size() != 0) begin idle_cycle(); if (++guard > 1000) fail("drain timeout"); end
        repeat (4) idle_cycle();
    endtask

    function automatic int random_K_sel();
        int x = $urandom_range(0, 99);
        if (x < 30)      return 8;
        else if (x < 50) return $urandom_range(9, 16);
        else if (x < 80) return $urandom_range(17, 100);
        else             return $urandom_range(101, 800);
    endfunction

    // ---------------------------------------------------------------- vectors
    string vec_dir;
    logic [63:0] va[], vw[], pe_cyc[];
    logic [7:0]  vctl[];
    logic [31:0] vk[], vacc[], pe_res[];

    initial begin
        int n_cyc, n_tiles, idx, n_pe_seq;
        longint base_checks;

        rst = 1'b1; in_valid = 1'b0; in_first = 1'b0; in_last = 1'b0; in_tok = '0;
        a_vec = '0; w_vec = '0;
        if (!$value$plusargs("VEC_DIR=%s", vec_dir)) fail("missing +VEC_DIR");

        va = new[5069]; vw = new[5069]; vctl = new[5069]; vk = new[52]; vacc = new[52 * 64];
        $readmemh($sformatf("%s/unit/array/array_a.hex", vec_dir),   va);
        $readmemh($sformatf("%s/unit/array/array_w.hex", vec_dir),   vw);
        $readmemh($sformatf("%s/unit/array/array_ctl.hex", vec_dir), vctl);
        $readmemh($sformatf("%s/unit/array/array_k.hex", vec_dir),   vk);
        $readmemh($sformatf("%s/unit/array/array_acc.hex", vec_dir), vacc);
        pe_cyc = new[43708]; pe_res = new[641];
        $readmemh($sformatf("%s/unit/pe/pe_cycles.hex", vec_dir),  pe_cyc);
        $readmemh($sformatf("%s/unit/pe/pe_results.hex", vec_dir), pe_res);
        if ($isunknown(va[5068]) || $isunknown(vacc[52*64-1]) || $isunknown(pe_cyc[43707]) ||
            $isunknown(pe_res[640]) || $isunknown(vk[51]))
            fail("vector files shorter than expected");

        repeat (5) @(posedge clk);
        #0.5 rst = 1'b0;
        repeat (3) idle_cycle();

        // ---------------- Phase A: array vectors
        base_checks = checks;
        idx = 0;
        for (int t = 0; t < 52; t++) begin
            int K;
            K = vk[t];
            for (int p = K; p < K_MIN; p++) idle_cycle();   // K < 8: pad before the tile so
                                                             // captures stay >= 8 cycles apart
            if (vctl[idx][0] !== 1'b1) fail($sformatf("array vector tile %0d: first not set", t));
            for (int k = 0; k < K; k++) begin
                if (vctl[idx + k][0] !== (k == 0) || vctl[idx + k][1] !== (k == K - 1))
                    fail($sformatf("array vector tile %0d: ctl mismatch at k %0d", t, k));
                drive_k(va[idx + k], vw[idx + k], k == 0, k == K - 1, TW'(t));
            end
            // model (TB) vs golden vector acc
            for (int r = 0; r < N; r++)
                for (int c = 0; c < N; c++)
                    if (model_acc[r][c] !== vacc[t * 64 + r * 8 + c])
                        fail($sformatf("TB model disagrees with array_acc.hex tile %0d r%0d c%0d", t, r, c));
            idx += K;
        end
        if (idx != 5069) fail("array vector cycle count mismatch");
        drain_all();
        $display("phase A (unit/array vectors): 52 tiles, checks=%0d", checks - base_checks);

        // ---------------- Phase B: PE vectors broadcast to all 64 PEs
        base_checks = checks;
        n_pe_seq = 0;
        idx = 0;
        while (idx < 43708) begin
            int K;
            logic [TW-1:0] tok;
            K = 0;
            begin   // sequence length -> pad before short sequences (captures >= 8 cycles apart)
                int len;
                len = 1;
                while (pe_cyc[idx + len - 1][49] !== 1'b1) len++;
                for (int p = len; p < K_MIN; p++) idle_cycle();
            end
            tok = TW'($urandom);
            do begin
                logic [N-1:0][7:0] a, w;
                logic [63:0] L;
                L = pe_cyc[idx + K];
                for (int i = 0; i < N; i++) begin a[i] = L[47:40]; w[i] = L[39:32]; end
                if (L[48] !== (K == 0)) fail($sformatf("pe vector: first mismatch at line %0d", idx + K));
                drive_k(a, w, L[48], L[49], tok);
                if (model_acc[0][0] !== L[31:0]) fail($sformatf("TB model vs pe_cycles line %0d", idx + K));
                K++;
            end while (pe_cyc[idx + K - 1][49] !== 1'b1);
            if (model_acc[0][0] !== pe_res[n_pe_seq]) fail("TB model vs pe_results");
            n_pe_seq++;
            idx += K;
        end
        if (n_pe_seq != 641) fail($sformatf("pe sequences %0d != 641", n_pe_seq));
        drain_all();
        $display("phase B (unit/pe vectors): %0d sequences x 64 PEs, checks=%0d", n_pe_seq, checks - base_checks);

        // ---------------- Phase C1: random tiles, zero gap, incl. K = 8 bursts
        base_checks = checks;
        for (int i = 0; i < 120; i++) begin
            if (i % 20 == 0) for (int b = 0; b < 24; b++) random_tile(8, 0);   // K = 8 burst
            random_tile(random_K_sel(), 0);
        end
        drain_all();
        $display("phase C1 (random, zero gap): checks=%0d", checks - base_checks);

        // ---------------- Phase C2: random tiles with in_valid gaps inside and between tiles
        base_checks = checks;
        for (int i = 0; i < 120; i++) begin
            if (i % 30 == 0) for (int b = 0; b < 16; b++) random_tile(8, 0);
            random_tile(random_K_sel(), $urandom_range(1, 40));
        end
        drain_all();
        $display("phase C2 (random, gaps): checks=%0d", checks - base_checks);

        if (tiles_drained != tiles_issued)
            fail($sformatf("tiles drained %0d != issued %0d", tiles_drained, tiles_issued));
        if (err_overrun !== 1'b0) fail("err_overrun set after legal traffic");
        checks += 2;

        // ---------------- Phase D: K < 8 must raise sticky err_overrun
        ovr_must_be_0 = 1'b0;
        check_en = 1'b0;
        random_tile(8, 0);
        random_tile(7, 0);          // capture 7 cycles after the previous one -> overrun
        repeat (L_ARRAY + 2) idle_cycle();
        if (err_overrun !== 1'b1) fail("err_overrun not raised by K = 7");
        repeat (50) idle_cycle();
        if (err_overrun !== 1'b1) fail("err_overrun not sticky");
        checks += 2;
        // reset clears it
        @(posedge clk); #0.5 rst = 1'b1;
        @(posedge clk); #0.5 rst = 1'b0;
        exp_q.delete(); drain_j = 0;
        repeat (L_ARRAY + 10) idle_cycle();
        if (err_overrun !== 1'b0) fail("err_overrun not cleared by rst");
        if (out_valid !== 1'b0) fail("out_valid after rst");
        checks += 2;
        // K = 1 streams (both first and last each cycle) also overrun
        random_tile(1, 0); random_tile(1, 0);
        repeat (L_ARRAY + 2) idle_cycle();
        if (err_overrun !== 1'b1) fail("err_overrun not raised by K = 1");
        checks++;
        @(posedge clk); #0.5 rst = 1'b1;
        @(posedge clk); #0.5 rst = 1'b0;
        exp_q.delete(); drain_j = 0;
        repeat (L_ARRAY + 10) idle_cycle();
        // after rst: legal traffic works again and keeps err_overrun = 0
        tiles_issued = 0; tiles_drained = 0;
        check_en = 1'b1; ovr_must_be_0 = 1'b1;
        for (int b = 0; b < 10; b++) random_tile(8, 0);
        for (int i = 0; i < 10; i++) random_tile(random_K_sel(), 5);
        drain_all();
        if (tiles_drained != tiles_issued || tiles_issued != 20)
            fail($sformatf("post-reset tiles drained %0d != issued %0d", tiles_drained, tiles_issued));
        checks++;
        $display("phase D (overrun + reset) done");

        $display("TEST PASSED checks=%0d", checks);
        $finish;
    end
endmodule
