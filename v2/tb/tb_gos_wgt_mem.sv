// GOS_UNIT_TB: gos_pkg.sv gos_wgt_mem.sv
`timescale 1ns/1ps
// tb_gos_wgt_mem — self-checking TB for gos_wgt_mem. Two DUTs (PS_RD_LAT = 1 and 2) share every
// input; both are checked against one TB shadow model (16384 x 64 bit, per-byte known tracking).
//  1. reset; fill the whole memory through the PS port (random data, all byte enables)
//  2. directed: port-B wrap (14-bit field of a counter crossing 16384), sequential sweep
//  3. random mix: random PS reads/writes with random byte enables, random port-B reads with
//     random valid gaps and tokens, occasional rst
//  4. real images: $VEC_DIR/<net>/wgt.hex (lenet5, cifar10) written through the PS port, read back
//     through port B (compared to the file words directly) and through the PS port
// Checked every cycle: ps_rdata (latency PS_RD_LAT, read-first), out_valid / out_tok / rd_data
// (latency 2). A port-B read of the word the PS writes in the same cycle is don't-care (undefined).
module tb_gos_wgt_mem;
  import gos_pkg::*;

  localparam int TW     = 16;
  localparam int N_RAND = 30000;

  logic clk = 1'b0;
  always #2.5 clk = ~clk;

  logic              rst;
  logic              ps_en, ps_we;
  logic [7:0]        ps_be;
  logic [WGT_AW-1:0] ps_addr;
  logic [63:0]       ps_wdata;
  logic              in_valid;
  logic [TW-1:0]     in_tok;
  logic [WGT_AW-1:0] rd_addr;

  logic [63:0]   ps_rdata  [2];
  logic          out_valid [2];
  logic [TW-1:0] out_tok   [2];
  logic [63:0]   rd_data   [2];

  gos_wgt_mem #(.TW(TW), .PS_RD_LAT(1)) dut1 (
    .clk, .rst, .ps_en, .ps_we, .ps_be, .ps_addr, .ps_wdata, .ps_rdata(ps_rdata[0]),
    .in_valid, .in_tok, .rd_addr, .out_valid(out_valid[0]), .out_tok(out_tok[0]), .rd_data(rd_data[0]));

  gos_wgt_mem #(.TW(TW), .PS_RD_LAT(2)) dut2 (
    .clk, .rst, .ps_en, .ps_we, .ps_be, .ps_addr, .ps_wdata, .ps_rdata(ps_rdata[1]),
    .in_valid, .in_tok, .rd_addr, .out_valid(out_valid[1]), .out_tok(out_tok[1]), .rd_data(rd_data[1]));

  // ---------------------------------------------------------------- shadow model
  logic [63:0] sh    [WGT_DEPTH];
  logic [7:0]  known [WGT_DEPTH];   // per-byte known mask

  bit            ps_chk [4];
  logic [63:0]   ps_exp [4];
  logic [7:0]    ps_kn  [4];
  bit            b_v    [4];
  logic [TW-1:0] b_tok  [4];
  logic [63:0]   b_dat  [4];
  logic [7:0]    b_kn   [4];

  int cyc = 0, checks = 0, errors = 0, n_dc = 0;
  bit prev_rst = 1'b0;

  task automatic fail(input string msg);
    errors++;
    if (errors <= 20) $display("ERROR cyc=%0d: %s", cyc, msg);
  endtask

  task automatic idle_inputs();
    ps_en = 0; ps_we = 0; ps_be = '0; ps_addr = '0; ps_wdata = '0;
    in_valid = 0; in_tok = '0; rd_addr = '0;
  endtask

  task automatic cmp_bytes(input string what, input logic [63:0] got, input logic [63:0] exp,
                           input logic [7:0] kn);
    for (int j = 0; j < 8; j++) if (kn[j]) begin
      if (got[8*j +: 8] !== exp[8*j +: 8])
        fail($sformatf("%s byte %0d = %h exp %h", what, j, got[8*j +: 8], exp[8*j +: 8]));
      checks++;
    end
  endtask

  task automatic step();
    int s0, sm1;
    s0  = cyc % 4;
    sm1 = (cyc + 3) % 4;
    ps_chk[s0] = ps_en;
    ps_exp[s0] = sh[ps_addr];
    ps_kn[s0]  = known[ps_addr];
    b_v[s0]    = in_valid && !rst;
    b_tok[s0]  = in_tok;
    b_dat[s0]  = sh[rd_addr];
    b_kn[s0]   = known[rd_addr];
    if (in_valid && ps_en && ps_we && ps_be != 0 && ps_addr == rd_addr) begin
      b_kn[s0] = '0; n_dc++;
    end
    if (ps_en && ps_we) for (int j = 0; j < 8; j++) if (ps_be[j]) begin
      sh[ps_addr][8*j +: 8] = ps_wdata[8*j +: 8];
      known[ps_addr][j] = 1'b1;
    end

    @(posedge clk); #1;

    if (ps_chk[s0])             cmp_bytes("dut1 ps_rdata", ps_rdata[0], ps_exp[s0],  ps_kn[s0]);
    if (cyc > 0 && ps_chk[sm1]) cmp_bytes("dut2 ps_rdata", ps_rdata[1], ps_exp[sm1], ps_kn[sm1]);
    for (int d = 0; d < 2; d++) begin
      bit ev;
      ev = (cyc > 0) && b_v[sm1] && !rst && !prev_rst;
      if (out_valid[d] !== ev) fail($sformatf("dut%0d out_valid=%b exp %b", d+1, out_valid[d], ev));
      checks++;
      if (ev) begin
        if (out_tok[d] !== b_tok[sm1]) fail($sformatf("dut%0d out_tok=%h exp %h", d+1, out_tok[d], b_tok[sm1]));
        checks++;
        cmp_bytes($sformatf("dut%0d rd_data", d+1), rd_data[d], b_dat[sm1], b_kn[sm1]);
      end
    end
    prev_rst = rst;
    cyc++;
    #1;
  endtask

  // ---------------------------------------------------------------- real image
  logic [63:0] img [0:WGT_DEPTH-1];

  task automatic image_test(input string vec_dir, input string net);
    string fn;
    int n;
    fn = {vec_dir, "/", net, "/wgt.hex"};
    for (int i = 0; i < WGT_DEPTH; i++) img[i] = 'x;
    $readmemh(fn, img);
    n = 0;
    while (n < WGT_DEPTH && !$isunknown(img[n])) n++;
    if (n == 0) begin
      $display("TEST FAILED cannot read %s", fn);
      $fatal(1, "missing vector file");
    end
    $display("image %s: %0d words", fn, n);
    idle_inputs();
    for (int w = 0; w < n; w++) begin
      ps_en = 1; ps_we = 1; ps_be = 8'hff; ps_addr = WGT_AW'(w); ps_wdata = img[w];
      step();
    end
    idle_inputs();
    // port B back-to-back sweep, compared to the file words directly (pipeline of 2)
    // (after step() of request w, a latency-2 output shows request w-1, latency-1 shows request w)
    for (int w = 0; w < n + 1; w++) begin
      in_valid = (w < n); in_tok = TW'(w); rd_addr = WGT_AW'(w < n ? w : 0);
      step();
      if (w >= 1) for (int d = 0; d < 2; d++) begin
        if (out_valid[d] !== 1'b1 || out_tok[d] !== TW'(w - 1) || rd_data[d] !== img[w - 1])
          fail($sformatf("dut%0d %s wgt word %0d = %h exp %h", d+1, net, w-1, rd_data[d], img[w-1]));
        checks++;
      end
    end
    idle_inputs();
    // PS read-back, also compared directly
    for (int w = 0; w < n + 1; w++) begin
      ps_en = (w < n); ps_we = 0; ps_addr = WGT_AW'(w < n ? w : 0);
      step();
      if (w < n) begin
        if (ps_rdata[0] !== img[w]) fail($sformatf("dut1 PS %s word %0d", net, w));
        checks++;
      end
      if (w >= 1) begin
        if (ps_rdata[1] !== img[w - 1]) fail($sformatf("dut2 PS %s word %0d", net, w-1));
        checks++;
      end
    end
    idle_inputs(); step(); step();
  endtask

  function automatic logic [WGT_AW-1:0] rnd_addr();
    int r;
    r = $urandom_range(0, 99);
    if (r < 10) return WGT_AW'($urandom_range(0, 7));
    if (r < 20) return WGT_AW'(WGT_DEPTH - 1 - $urandom_range(0, 7));
    if (r < 50) return WGT_AW'($urandom_range(0, 31));
    return WGT_AW'($urandom);
  endfunction

  function automatic logic [7:0] rnd_be();
    int r;
    r = $urandom_range(0, 9);
    if (r == 0) return 8'hff;
    if (r == 1) return 8'h00;
    return 8'($urandom);
  endfunction

  string vec_dir;

  initial begin
    process::self().srandom(32'h0B6E_77A1);
    if (!$value$plusargs("VEC_DIR=%s", vec_dir)) vec_dir = "../../../vectors/generated";
    for (int w = 0; w < WGT_DEPTH; w++) known[w] = '0;
    idle_inputs();

    rst = 1; repeat (3) step(); rst = 0;
    for (int w = 0; w < WGT_DEPTH; w++) begin
      ps_en = 1; ps_we = 1; ps_be = 8'hff; ps_addr = WGT_AW'(w); ps_wdata = {$urandom, $urandom};
      step();
    end
    idle_inputs(); step();
    for (int w = 0; w < WGT_DEPTH; w++) if (known[w] !== 8'hff) fail("fill incomplete");

    // wrap: counter values 16376..16391, only bits 13:0 reach the port
    for (int full = WGT_DEPTH - 8; full < WGT_DEPTH + 8; full++) begin
      in_valid = 1; in_tok = TW'(full); rd_addr = full[WGT_AW-1:0];
      step();
    end
    in_valid = 1; begin int full; full = WGT_DEPTH + 5; rd_addr = full[WGT_AW-1:0]; end
    step(); in_valid = 0; step();
    for (int d = 0; d < 2; d++) begin
      if (rd_data[d] !== sh[5]) fail($sformatf("dut%0d wrap", d+1));
      checks++;
    end
    idle_inputs(); step(); step();

    for (int t = 0; t < N_RAND; t++) begin
      ps_en    = $urandom_range(0, 9) < 7;
      ps_we    = $urandom_range(0, 1);
      ps_be    = rnd_be();
      ps_addr  = rnd_addr();
      ps_wdata = {$urandom, $urandom};
      in_valid = $urandom_range(0, 9) < 7;
      in_tok   = TW'($urandom);
      rd_addr  = rnd_addr();
      rst      = ($urandom_range(0, 1999) == 0);
      step();
      rst = 0;
    end
    idle_inputs(); step(); step();

    image_test(vec_dir, "lenet5");
    image_test(vec_dir, "cifar10");

    $display("info: port-B collisions skipped (don't care) = %0d", n_dc);
    if (errors == 0 && checks > 0) $display("TEST PASSED checks=%0d", checks);
    else begin
      $display("TEST FAILED errors=%0d checks=%0d", errors, checks);
      $fatal(1, "tb_gos_wgt_mem failed");
    end
    $finish;
  end

endmodule
