// GOS_UNIT_TB: gos_pkg.sv gos_qparam_mem.sv
`timescale 1ns/1ps
// tb_gos_qparam_mem — self-checking TB for gos_qparam_mem. Two DUTs (PS_RD_LAT = 1 and 2) share
// every input; both are checked against one TB shadow model: QP_E[256] (64 bit, per-byte known)
// and QP_O[256] (s, 6 bit). PS view: 9-bit word address, bit 0 = bank (0: QP_E, 1: QP_O),
// index = addr[8:1]; odd words read {58'b0, s}; an odd write stores wdata[5:0] iff be[0].
//  1. reset; fill all 512 PS words (random data incl. nonzero bits 63:6 on odd words)
//  2. directed: odd-word upper bits read as 0; odd write with be[0] = 0 is ignored; channel-index
//     wrap (8-bit field of a counter crossing 256)
//  3. random mix: random PS reads/writes/byte enables, random accelerator channel reads with random
//     valid gaps and tokens, occasional rst
//  4. real images: $VEC_DIR/<net>/qparam.hex (combined view) written through the PS port; the
//     accelerator port is compared field by field to qparam_e.hex / qparam_o.hex (read directly
//     from the files), the PS port to qparam.hex (lenet5, cifar10)
// Checked every cycle: ps_rdata (latency PS_RD_LAT, read-first), out_valid / out_tok /
// {q_bias, m, s} (latency 2). A channel read colliding with a PS write of that channel is don't-care.
module tb_gos_qparam_mem;
  import gos_pkg::*;

  localparam int TW     = 16;
  localparam int N_RAND = 30000;

  logic clk = 1'b0;
  always #2.5 clk = ~clk;

  logic             rst;
  logic             ps_en, ps_we;
  logic [7:0]       ps_be;
  logic [QP_AW:0]   ps_addr;
  logic [63:0]      ps_wdata;
  logic             in_valid;
  logic [TW-1:0]    in_tok;
  logic [QP_AW-1:0] rd_ch;

  logic [63:0]        ps_rdata  [2];
  logic               out_valid [2];
  logic [TW-1:0]      out_tok   [2];
  logic signed [31:0] q_bias    [2];
  logic [M_W-1:0]     m         [2];
  logic [S_W-1:0]     s         [2];

  gos_qparam_mem #(.TW(TW), .PS_RD_LAT(1)) dut1 (
    .clk, .rst, .ps_en, .ps_we, .ps_be, .ps_addr, .ps_wdata, .ps_rdata(ps_rdata[0]),
    .in_valid, .in_tok, .rd_ch, .out_valid(out_valid[0]), .out_tok(out_tok[0]),
    .q_bias(q_bias[0]), .m(m[0]), .s(s[0]));

  gos_qparam_mem #(.TW(TW), .PS_RD_LAT(2)) dut2 (
    .clk, .rst, .ps_en, .ps_we, .ps_be, .ps_addr, .ps_wdata, .ps_rdata(ps_rdata[1]),
    .in_valid, .in_tok, .rd_ch, .out_valid(out_valid[1]), .out_tok(out_tok[1]),
    .q_bias(q_bias[1]), .m(m[1]), .s(s[1]));

  // ---------------------------------------------------------------- shadow model
  logic [63:0]    she  [QP_CH];
  logic [7:0]     kne  [QP_CH];
  logic [S_W-1:0] sho  [QP_CH];
  bit             kno  [QP_CH];

  // PS view of the model
  function automatic logic [63:0] ps_word(input logic [QP_AW:0] a);
    return a[0] ? {{(64-S_W){1'b0}}, sho[a[QP_AW:1]]} : she[a[QP_AW:1]];
  endfunction
  function automatic logic [7:0] ps_known(input logic [QP_AW:0] a);
    return a[0] ? (kno[a[QP_AW:1]] ? 8'hff : 8'hfe) : kne[a[QP_AW:1]];  // odd bits 63:8 always 0
  endfunction

  bit            ps_chk [4];
  logic [63:0]   ps_exp [4];
  logic [7:0]    ps_kn  [4];
  bit            b_v    [4];
  logic [TW-1:0] b_tok  [4];
  logic [63:0]   b_e    [4];
  logic [7:0]    b_ke   [4];
  logic [S_W-1:0] b_o   [4];
  bit            b_ko   [4];

  int cyc = 0, checks = 0, errors = 0, n_dc = 0;
  bit prev_rst = 1'b0;

  task automatic fail(input string msg);
    errors++;
    if (errors <= 20) $display("ERROR cyc=%0d: %s", cyc, msg);
  endtask

  task automatic idle_inputs();
    ps_en = 0; ps_we = 0; ps_be = '0; ps_addr = '0; ps_wdata = '0;
    in_valid = 0; in_tok = '0; rd_ch = '0;
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
    logic [QP_AW-1:0] idx;
    s0  = cyc % 4;
    sm1 = (cyc + 3) % 4;
    idx = ps_addr[QP_AW:1];
    ps_chk[s0] = ps_en;
    ps_exp[s0] = ps_word(ps_addr);
    ps_kn[s0]  = ps_known(ps_addr);
    b_v[s0]    = in_valid && !rst;
    b_tok[s0]  = in_tok;
    b_e[s0]    = she[rd_ch];
    b_ke[s0]   = kne[rd_ch];
    b_o[s0]    = sho[rd_ch];
    b_ko[s0]   = kno[rd_ch];
    if (in_valid && ps_en && ps_we && idx == rd_ch) begin
      if (!ps_addr[0] && ps_be != 0) begin b_ke[s0] = '0; n_dc++; end
      if ( ps_addr[0] && ps_be[0])   begin b_ko[s0] = 0;  n_dc++; end
    end
    if (ps_en && ps_we) begin
      if (!ps_addr[0]) begin
        for (int j = 0; j < 8; j++) if (ps_be[j]) begin
          she[idx][8*j +: 8] = ps_wdata[8*j +: 8]; kne[idx][j] = 1'b1;
        end
      end else if (ps_be[0]) begin
        sho[idx] = ps_wdata[S_W-1:0]; kno[idx] = 1'b1;
      end
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
        cmp_bytes($sformatf("dut%0d {m,q_bias}", d+1), {m[d], q_bias[d]}, b_e[sm1], b_ke[sm1]);
        if (b_ko[sm1]) begin
          if (s[d] !== b_o[sm1]) fail($sformatf("dut%0d s=%0d exp %0d", d+1, s[d], b_o[sm1]));
          checks++;
        end
      end
    end
    prev_rst = rst;
    cyc++;
    #1;
  endtask

  // ---------------------------------------------------------------- real images
  logic [63:0] img_c [0:2*QP_CH-1];
  logic [63:0] img_e [0:QP_CH-1];
  logic [63:0] img_o [0:QP_CH-1];

  task automatic image_test(input string vec_dir, input string net);
    string fc, fe, fo;
    int nc, ne, no;
    fc = {vec_dir, "/", net, "/qparam.hex"};
    fe = {vec_dir, "/", net, "/qparam_e.hex"};
    fo = {vec_dir, "/", net, "/qparam_o.hex"};
    for (int i = 0; i < 2*QP_CH; i++) img_c[i] = 'x;
    for (int i = 0; i < QP_CH; i++) begin img_e[i] = 'x; img_o[i] = 'x; end
    $readmemh(fc, img_c); $readmemh(fe, img_e); $readmemh(fo, img_o);
    nc = 0; while (nc < 2*QP_CH && !$isunknown(img_c[nc])) nc++;
    ne = 0; while (ne < QP_CH && !$isunknown(img_e[ne])) ne++;
    no = 0; while (no < QP_CH && !$isunknown(img_o[no])) no++;
    if (nc == 0 || ne == 0 || nc != 2*ne || no != ne) begin
      $display("TEST FAILED bad qparam files for %s (%0d/%0d/%0d lines)", net, nc, ne, no);
      $fatal(1, "bad vector files");
    end
    $display("image %s: %0d combined words, %0d channels", fc, nc, ne);
    for (int i = 0; i < ne; i++) begin
      if (img_o[i][63:S_W] !== '0) fail($sformatf("%s qparam_o[%0d] bits 63:6 not 0", net, i));
      checks++;
    end
    idle_inputs();
    for (int w = 0; w < nc; w++) begin
      ps_en = 1; ps_we = 1; ps_be = 8'hff; ps_addr = (QP_AW+1)'(w); ps_wdata = img_c[w];
      step();
    end
    idle_inputs();
    // accelerator port: after step() of request i, the latency-2 output shows request i-1
    for (int i = 0; i < ne + 1; i++) begin
      in_valid = (i < ne); in_tok = TW'(i); rd_ch = QP_AW'(i < ne ? i : 0);
      step();
      if (i >= 1) for (int d = 0; d < 2; d++) begin
        if (out_valid[d] !== 1'b1 || out_tok[d] !== TW'(i - 1) ||
            q_bias[d] !== signed'(img_e[i-1][31:0]) || m[d] !== img_e[i-1][63:32] ||
            s[d] !== img_o[i-1][S_W-1:0])
          fail($sformatf("dut%0d %s ch %0d: q_bias=%0d m=%h s=%0d exp e=%h o=%h", d+1, net, i-1,
                         q_bias[d], m[d], s[d], img_e[i-1], img_o[i-1]));
        checks++;
      end
    end
    idle_inputs();
    // PS read-back of the combined view
    for (int w = 0; w < nc + 1; w++) begin
      ps_en = (w < nc); ps_we = 0; ps_addr = (QP_AW+1)'(w < nc ? w : 0);
      step();
      if (w < nc) begin
        if (ps_rdata[0] !== img_c[w]) fail($sformatf("dut1 PS %s word %0d", net, w));
        checks++;
      end
      if (w >= 1) begin
        if (ps_rdata[1] !== img_c[w - 1]) fail($sformatf("dut2 PS %s word %0d", net, w-1));
        checks++;
      end
    end
    idle_inputs(); step(); step();
  endtask

  function automatic logic [7:0] rnd_be();
    int r;
    r = $urandom_range(0, 9);
    if (r == 0) return 8'hff;
    if (r == 1) return 8'h00;
    return 8'($urandom);
  endfunction

  string vec_dir;

  initial begin
    process::self().srandom(32'h0C9A_3E11);
    if (!$value$plusargs("VEC_DIR=%s", vec_dir)) vec_dir = "../../../vectors/generated";
    for (int i = 0; i < QP_CH; i++) begin kne[i] = '0; kno[i] = 0; end
    idle_inputs();

    rst = 1; repeat (3) step(); rst = 0;
    for (int w = 0; w < 2*QP_CH; w++) begin
      ps_en = 1; ps_we = 1; ps_be = 8'hff; ps_addr = (QP_AW+1)'(w); ps_wdata = {$urandom, $urandom};
      step();
    end
    idle_inputs(); step();
    for (int i = 0; i < QP_CH; i++) if (kne[i] !== 8'hff || !kno[i]) fail("fill incomplete");

    // odd word: bits 63:6 are not stored and read as 0; be[0] = 0 leaves s unchanged
    ps_en = 1; ps_we = 1; ps_be = 8'hff; ps_addr = 9'd7; ps_wdata = 64'hffff_ffff_ffff_ffea; step();
    ps_en = 1; ps_we = 1; ps_be = 8'hfe; ps_addr = 9'd7; ps_wdata = 64'h0000_0000_0000_0001; step();
    ps_en = 1; ps_we = 0; ps_addr = 9'd7; step();
    if (ps_rdata[0] !== 64'h2a) fail($sformatf("odd word read %h exp 2a", ps_rdata[0]));
    checks++;
    idle_inputs(); step();
    if (ps_rdata[1] !== 64'h2a) fail($sformatf("dut2 odd word read %h exp 2a", ps_rdata[1]));
    checks++;
    in_valid = 1; rd_ch = 8'd3; step(); idle_inputs(); step();
    for (int d = 0; d < 2; d++) begin
      if (s[d] !== 6'h2a) fail("s of channel 3");
      checks++;
    end

    // channel-index wrap: counter values 248..263 -> 8-bit field
    for (int full = QP_CH - 8; full < QP_CH + 8; full++) begin
      in_valid = 1; in_tok = TW'(full); rd_ch = full[QP_AW-1:0];
      step();
    end
    in_valid = 1; begin int full; full = QP_CH + 9; rd_ch = full[QP_AW-1:0]; end
    step(); in_valid = 0; step();
    for (int d = 0; d < 2; d++) begin
      if ({m[d], q_bias[d]} !== she[9] || s[d] !== sho[9]) fail($sformatf("dut%0d wrap", d+1));
      checks++;
    end
    idle_inputs(); step(); step();

    for (int t = 0; t < N_RAND; t++) begin
      ps_en    = $urandom_range(0, 9) < 7;
      ps_we    = $urandom_range(0, 1);
      ps_be    = rnd_be();
      ps_addr  = (QP_AW+1)'($urandom);
      ps_wdata = {$urandom, $urandom};
      in_valid = $urandom_range(0, 9) < 7;
      in_tok   = TW'($urandom);
      rd_ch    = ($urandom_range(0, 3) == 0) ? ps_addr[QP_AW:1] : QP_AW'($urandom);
      rst      = ($urandom_range(0, 1999) == 0);
      step();
      rst = 0;
    end
    idle_inputs(); step(); step();

    image_test(vec_dir, "lenet5");
    image_test(vec_dir, "cifar10");

    $display("info: channel-read collisions skipped (don't care) = %0d", n_dc);
    if (errors == 0 && checks > 0) $display("TEST PASSED checks=%0d", checks);
    else begin
      $display("TEST FAILED errors=%0d checks=%0d", errors, checks);
      $fatal(1, "tb_gos_qparam_mem failed");
    end
    $finish;
  end

endmodule
