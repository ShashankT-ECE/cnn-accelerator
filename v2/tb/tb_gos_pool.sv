// GOS_UNIT_TB: gos_pkg.sv gos_pool.sv
`timescale 1ns/1ps
// tb_gos_pool — self-checking TB for gos_pool (L_POOL = 2, 1:1 token pipeline).
//  1. unit vectors $VEC_DIR/unit/pool/pool_cols.hex: per case dy=0 columns 0..7 then dy=1
//     columns 0..7, back-to-back across all cases; checks the 4 pooled bytes on both halves,
//     the 8-bit be (half select + x-tail mask from row_mask = rem), be = 0 / we = 0 on stores.
//  2. real drain streams $VEC_DIR/<net>/layers/<L>/drain.hex (+ desc.hex for pool_en), every
//     non-LOGIT layer of both nets, back-to-back, random column data: checks
//     (out_be & {8{ch_valid}}) == drain.be and out_we & ch_valid == drain.we for every record,
//     the data against a TB model, and that every dy=1 column was stored by the directly
//     preceding tile (drain order: dy innermost).
//  3. random: gaps, pool_en 0/1, dy, col, half, row_mask, full signed int8, back-to-back
//     dy=0 col j -> dy=1 col j; dy=1 without a prior store (D11-5, undefined) is not
//     data-checked (be/we still checked).
//  Token passthrough and out_valid latency are checked on every cycle.
module tb_gos_pool;
  import gos_pkg::*;

  localparam int TW     = 24;
  localparam int L_POOL = 2;
  localparam int MAXP   = 8192;
  localparam int MAXD   = 16384;
  localparam int N_RAND = 60000;

  logic clk = 1'b0;
  logic rst;
  logic in_valid;
  logic [TW-1:0] in_tok;
  logic [N-1:0][7:0] in_q;
  logic [2:0] in_col;
  logic in_dy, in_pool_en, in_half;
  logic [N-1:0] in_row_mask;
  logic out_valid;
  logic [TW-1:0] out_tok;
  logic [N-1:0][7:0] out_data;
  logic [N-1:0] out_be;
  logic out_we;

  gos_pool #(.TW(TW)) dut (.*);

  always #2.5 clk = ~clk;

  typedef struct {
    logic              v;
    logic [TW-1:0]     tok;
    logic              chk_data;
    logic [N-1:0][7:0] data;
    logic [N-1:0]      be;
    logic              we;
    logic              gate;       // ch_valid (stream test): compare be/we after gating
    logic [N-1:0]      gbe;        // expected be after gating (stream test)
    logic              gwe;
    logic              use_g;
  } exp_t;

  exp_t exq[L_POOL+1];

  // TB model of the pool buffer
  logic [N-1:0][7:0] mbuf [N];
  logic [N-1:0]      mstored;

  int checks = 0;
  int errors = 0;

  task automatic fail(input string msg);
    errors++;
    $display("ERROR %s (t=%0t)", msg, $time);
    if (errors > 20) begin
      $display("TEST FAILED too many errors");
      $fatal(1, "TEST FAILED");
    end
  endtask

  function automatic logic [7:0] smax(input logic [7:0] a, input logic [7:0] b);
    return ($signed(a) > $signed(b)) ? a : b;
  endfunction

  // Model: expected output for one input, updates the model buffer.
  function automatic exp_t model(input logic [N-1:0][7:0] q, input int col, input logic dy,
                                 input logic pool_en, input logic half,
                                 input logic [N-1:0] rmask);
    exp_t e;
    logic [63:0] st, qq, dd;
    logic [7:0]  hb;
    logic [N-1:0] bb;
    e.v = 1'b1; e.chk_data = 1'b1; e.use_g = 1'b0; e.gate = 1'b1; e.gbe = '0; e.gwe = 1'b0;
    e.tok = '0;
    qq = q;
    if (!pool_en) begin
      e.data = q; e.be = rmask;
    end else if (!dy) begin
      mbuf[col] = q; mstored[col] = 1'b1;
      e.chk_data = 1'b0; e.data = '0; e.be = '0;
    end else begin
      e.chk_data = mstored[col];              // D11-5: undefined without a prior store
      st = mbuf[col];
      dd = '0; bb = '0;
      for (int i = 0; i < N/2; i++) begin
        hb = smax(smax(st[16*i +: 8], qq[16*i +: 8]), smax(st[16*i+8 +: 8], qq[16*i+8 +: 8]));
        dd[8*i +: 8]      = hb;
        dd[8*(i+4) +: 8]  = hb;
        bb[4*half + i]    = rmask[2*i];
      end
      e.data = dd; e.be = bb;
    end
    e.we = (e.be != '0);
    return e;
  endfunction

  // One clock: drive (or idle), shift expected pipeline, check the output.
  task automatic cycle(input logic v, input logic [N-1:0][7:0] q, input int col, input logic dy,
                       input logic pool_en, input logic half, input logic [N-1:0] rmask,
                       input exp_t e);
    in_valid = v; in_q = q; in_col = 3'(col); in_dy = dy; in_pool_en = pool_en;
    in_half = half; in_row_mask = rmask; in_tok = TW'($urandom);
    e.v = v; e.tok = in_tok;
    exq[0] = e;
    @(posedge clk);
    for (int s = L_POOL; s > 0; s--) exq[s] = exq[s-1];
    #1;
    if (out_valid !== exq[L_POOL].v)
      fail($sformatf("out_valid %b exp %b", out_valid, exq[L_POOL].v));
    else if (exq[L_POOL].v) begin
      exp_t x;
      x = exq[L_POOL];
      checks++;
      if (out_tok !== x.tok) fail($sformatf("tok %h exp %h", out_tok, x.tok));
      if (out_be !== x.be) fail($sformatf("be %b exp %b", out_be, x.be));
      if (out_we !== x.we) fail($sformatf("we %b exp %b", out_we, x.we));
      if (x.use_g) begin
        if ((out_be & {N{x.gate}}) !== x.gbe)
          fail($sformatf("gated be %b exp (drain.hex) %b", out_be & {N{x.gate}}, x.gbe));
        if ((out_we & x.gate) !== x.gwe)
          fail($sformatf("gated we %b exp (drain.hex) %b", out_we & x.gate, x.gwe));
      end
      if (x.chk_data && out_data !== x.data)
        fail($sformatf("data %h exp %h", out_data, x.data));
    end
  endtask

  task automatic idle(input int n);
    exp_t e;
    e = '{default: '0};
    for (int i = 0; i < n; i++) cycle(1'b0, '0, 0, 1'b0, 1'b0, 1'b0, '0, e);
  endtask

  function automatic logic [N-1:0] mask_of_rem(input int rem);
    return (rem >= N) ? '1 : N'((1 << rem) - 1);
  endfunction

  function automatic logic [N-1:0][7:0] rand_q(input bit nonneg);
    logic [N-1:0][7:0] q;
    for (int r = 0; r < N; r++) begin
      case ($urandom % 8)
        0: q[r] = nonneg ? 8'd0 : 8'h80;
        1: q[r] = 8'h7f;
        2: q[r] = nonneg ? 8'd1 : 8'hff;
        default: q[r] = nonneg ? 8'($urandom % 128) : 8'($urandom);
      endcase
    end
    return q;
  endfunction

  // ---------------- vectors ----------------
  logic [191:0] pv [MAXP];
  logic [159:0] dv [MAXD];
  logic [31:0]  descv [16];
  string vec_dir, fname;

  string layers [9] = '{"lenet5/layers/L0_conv1", "lenet5/layers/L1_conv3",
                         "lenet5/layers/L2_conv5", "lenet5/layers/L3_fc1",
                         "lenet5/layers/L4_fc2",
                         "cifar10/layers/L0_conv1", "cifar10/layers/L1_conv2",
                         "cifar10/layers/L2_conv3", "cifar10/layers/L3_fc"};

  initial begin
    int np, nd, n_stream, n_stream_pool, n_layers_run;
    if (!$value$plusargs("VEC_DIR=%s", vec_dir)) begin
      $display("TEST FAILED no +VEC_DIR");
      $fatal(1, "TEST FAILED");
    end

    for (int s = 0; s <= L_POOL; s++) exq[s] = '{default: '0};
    mstored = '0;
    rst = 1'b1;
    in_valid = 1'b0; in_q = '0; in_col = '0; in_dy = 1'b0; in_pool_en = 1'b0;
    in_half = 1'b0; in_row_mask = '0; in_tok = '0;
    repeat (3) @(posedge clk);
    #1;
    if (out_valid !== 1'b0) fail("out_valid not 0 in reset");
    rst = 1'b0;

    // ================= 1. unit vectors =================
    for (int i = 0; i < MAXP; i++) pv[i] = 'x;
    fname = {vec_dir, "/unit/pool/pool_cols.hex"};
    $readmemh(fname, pv);
    np = 0;
    while (np < MAXP && !$isunknown(pv[np])) np++;
    if (np == 0 || np % 8 != 0) begin
      $display("TEST FAILED pool_cols.hex: %0d lines", np);
      $fatal(1, "TEST FAILED");
    end
    for (int c = 0; c < np / 8; c++) begin
      logic half;
      int rem;
      logic [N-1:0] rmask;
      half  = pv[8*c][172];
      rem   = int'(pv[8*c][183:176]);
      rmask = mask_of_rem(rem);
      for (int j = 0; j < N; j++) begin               // dy = 0 tile
        exp_t e;
        if (int'(pv[8*c+j][171:168]) != j) fail($sformatf("case %0d line %0d: j", c, j));
        e = model(pv[8*c+j][63:0], j, 1'b0, 1'b1, half, rmask);
        cycle(1'b1, pv[8*c+j][63:0], j, 1'b0, 1'b1, half, rmask, e);
      end
      for (int j = 0; j < N; j++) begin               // dy = 1 tile
        exp_t e;
        logic [N-1:0][7:0] vd;
        e = model(pv[8*c+j][127:64], j, 1'b1, 1'b1, half, rmask);
        for (int b = 0; b < N; b++) vd[b] = pv[8*c+j][128 + 8*(b % 4) +: 8];
        if (e.data !== vd) fail($sformatf("case %0d col %0d: TB model %h != vector h %h", c, j, e.data, vd));
        if (e.be !== pv[8*c+j][167:160])
          fail($sformatf("case %0d col %0d: TB be %b != vector be %b", c, j, e.be, pv[8*c+j][167:160]));
        e.data = vd; e.be = pv[8*c+j][167:160]; e.we = (e.be != '0);
        cycle(1'b1, pv[8*c+j][127:64], j, 1'b1, 1'b1, half, rmask, e);
      end
    end
    $display("unit vectors: %0d cases (%0d lines)", np / 8, np);

    // ================= 2. real drain streams =================
    n_stream = 0; n_stream_pool = 0; n_layers_run = 0;
    foreach (layers[li]) begin
      logic pool_en, out_raw;
      int last_store_tile [N];
      for (int i = 0; i < 16; i++) descv[i] = 'x;
      fname = {vec_dir, "/", layers[li], "/desc.hex"};
      $readmemh(fname, descv);
      if ($isunknown(descv[6])) begin
        $display("TEST FAILED missing %s", fname);
        $fatal(1, "TEST FAILED");
      end
      pool_en = descv[6][1];
      out_raw = descv[6][2];
      if (out_raw) continue;                          // LOGIT layer: pool not in the path
      for (int i = 0; i < MAXD; i++) dv[i] = 'x;
      fname = {vec_dir, "/", layers[li], "/drain.hex"};
      $readmemh(fname, dv);
      nd = 0;
      while (nd < MAXD && !$isunknown(dv[nd])) nd++;
      if (nd == 0 || nd == MAXD) begin
        $display("TEST FAILED %s: %0d lines", fname, nd);
        $fatal(1, "TEST FAILED");
      end
      for (int j = 0; j < N; j++) last_store_tile[j] = -2;
      mstored = '0;
      for (int i = 0; i < nd; i++) begin
        exp_t e;
        logic [N-1:0][7:0] q;
        logic [7:0] be, rmask;
        int j, kind, tile;
        logic we, chv, dy, half;
        be    = dv[i][23:16];
        rmask = dv[i][31:24];
        j     = int'(dv[i][67:64]);
        kind  = int'(dv[i][71:68]);
        we    = dv[i][72];
        chv   = dv[i][73];
        dy    = dv[i][75];
        half  = dv[i][88];                            // ox_tile & 1
        tile  = int'(dv[i][143:128]);
        if (kind != (pool_en ? (dy ? 0 : 1) : 0))
          fail($sformatf("%s rec %0d: kind %0d (pool_en %b dy %b)", layers[li], i, kind, pool_en, dy));
        if (pool_en && dy) begin
          if (last_store_tile[j] != tile - 1)
            fail($sformatf("%s rec %0d: dy=1 col %0d not preceded by its dy=0 tile", layers[li], i, j));
        end
        if (pool_en && !dy) last_store_tile[j] = tile;
        q = rand_q(($urandom % 4) != 0);
        e = model(q, j, dy, pool_en, half, rmask);
        e.use_g = 1'b1; e.gate = chv; e.gbe = be; e.gwe = we;
        cycle(1'b1, q, j, dy, pool_en, half, rmask, e);
        n_stream++;
        if (pool_en) n_stream_pool++;
      end
      n_layers_run++;
      if ($urandom % 2) idle($urandom % 3);
    end
    $display("drain streams: %0d layers, %0d records (%0d pooled)", n_layers_run, n_stream, n_stream_pool);

    // ================= 3. random =================
    mstored = '0;
    for (int n = 0; n < N_RAND; n++) begin
      exp_t e;
      logic [N-1:0][7:0] q;
      logic v, dy, pe, half;
      int col, rem;
      logic [N-1:0] rmask;
      v    = ($urandom % 6) != 0;
      pe   = ($urandom % 4) != 0;
      dy   = $urandom % 2;
      half = $urandom % 2;
      col  = $urandom % N;
      rem  = ($urandom % 3 == 0) ? 2 * ($urandom % 5) : 8;   // 0,2,4,6,8 (even) or full
      rmask = ($urandom % 8 == 0) ? N'($urandom) : mask_of_rem(rem);
      q = rand_q(($urandom % 2) == 0);
      if (v) e = model(q, col, dy, pe, half, rmask);
      else   e = '{default: '0};
      cycle(v, q, col, dy, pe, half, rmask, e);
      // back-to-back dy=0 col -> dy=1 same col
      if (v && pe && !dy && ($urandom % 3 == 0)) begin
        q = rand_q(1'b0);
        rmask = mask_of_rem(8);
        e = model(q, col, 1'b1, 1'b1, half, rmask);
        cycle(1'b1, q, col, 1'b1, 1'b1, half, rmask, e);
      end
    end
    idle(L_POOL + 2);

    // reset clears valid
    in_valid = 1'b1; rst = 1'b1;
    @(posedge clk); #1;
    if (out_valid !== 1'b0) fail("out_valid not cleared by rst");
    @(posedge clk); #1;
    if (out_valid !== 1'b0) fail("out_valid not cleared by rst (2)");
    rst = 1'b0; in_valid = 1'b0;

    if (errors == 0) $display("TEST PASSED checks=%0d", checks);
    else begin
      $display("TEST FAILED errors=%0d", errors);
      $fatal(1, "TEST FAILED");
    end
    $finish;
  end
endmodule
