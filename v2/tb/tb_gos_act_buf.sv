// GOS_UNIT_TB: gos_pkg.sv gos_act_buf.sv
`timescale 1ns/1ps
// tb_gos_act_buf — self-checking TB for gos_act_buf. Two DUTs (PS_RD_LAT = 1 and 2) share every
// input; each is checked against one TB shadow model (8 banks x 4096 bytes, with known-bit tracking).
//  1. reset; fill the whole buffer through the PS port (random data, all byte enables)
//  2. directed: port-B wrap (12-bit field of a counter crossing 4096) and 8 independent addresses
//  3. directed busy mux: PS write / read while busy (ignored / returns 0, sticky flag in both DUTs),
//     accel writes while not busy (ignored), accel writes while busy (land), rst clears the flag
//  4. random mix: busy runs, random PS ops + byte enables, random accel writes (also while idle),
//     random port-B reads with random per-bank addresses, random valid gaps, random tokens
//  5. real images: $VEC_DIR/<net>/net/img0/act0.hex (lenet5, cifar10) written through the PS port,
//     read back through port B (same address per bank and rotated per-bank addresses) and the PS port
// Checked every cycle: ps_rdata (latency PS_RD_LAT, read-first; 0 for reads while busy),
// ps_busy_violation, out_valid / out_tok / rd_data (latency 2). Port-B reads that collide with a
// port-A write to the same bank word in the same cycle are don't-care (documented as undefined).
module tb_gos_act_buf;
  import gos_pkg::*;

  localparam int TW     = 16;
  localparam int N_RAND = 30000;

  logic clk = 1'b0;
  always #2.5 clk = ~clk;

  logic                     rst;
  logic                     accel_busy;
  logic                     ps_en, ps_we;
  logic [7:0]               ps_be;
  logic [ACT_AW-1:0]        ps_addr;
  logic [63:0]              ps_wdata;
  logic                     acc_wr_en;
  logic [ACT_AW-1:0]        acc_wr_addr;
  logic [63:0]              acc_wr_data;
  logic [7:0]               acc_wr_be;
  logic                     in_valid;
  logic [TW-1:0]            in_tok;
  logic [N-1:0][ACT_AW-1:0] rd_addr;

  logic [63:0]       ps_rdata  [2];
  logic              viol      [2];
  logic              out_valid [2];
  logic [TW-1:0]     out_tok   [2];
  logic [N-1:0][7:0] rd_data   [2];

  gos_act_buf #(.TW(TW), .PS_RD_LAT(1)) dut1 (
    .clk, .rst, .accel_busy, .ps_en, .ps_we, .ps_be, .ps_addr, .ps_wdata,
    .ps_rdata(ps_rdata[0]), .ps_busy_violation(viol[0]),
    .acc_wr_en, .acc_wr_addr, .acc_wr_data, .acc_wr_be,
    .in_valid, .in_tok, .rd_addr,
    .out_valid(out_valid[0]), .out_tok(out_tok[0]), .rd_data(rd_data[0]));

  gos_act_buf #(.TW(TW), .PS_RD_LAT(2)) dut2 (
    .clk, .rst, .accel_busy, .ps_en, .ps_we, .ps_be, .ps_addr, .ps_wdata,
    .ps_rdata(ps_rdata[1]), .ps_busy_violation(viol[1]),
    .acc_wr_en, .acc_wr_addr, .acc_wr_data, .acc_wr_be,
    .in_valid, .in_tok, .rd_addr,
    .out_valid(out_valid[1]), .out_tok(out_tok[1]), .rd_data(rd_data[1]));

  // ---------------------------------------------------------------- shadow model
  logic [7:0] sh    [N][ACT_DEPTH];
  bit         known [N][ACT_DEPTH];
  bit         flag_m;

  // expectation slots, indexed by request cycle mod 4
  bit                ps_chk  [4];
  logic [63:0]       ps_exp  [4];
  bit                ps_kn   [4][8];   // byte known (else don't care)
  bit                b_v     [4];
  logic [TW-1:0]     b_tok   [4];
  logic [N-1:0][7:0] b_dat   [4];
  bit                b_chk   [4][8];   // per-bank data check (known and no collision)

  int  cyc    = 0;
  int  checks = 0;
  int  errors = 0;
  int  n_dc   = 0;                     // port-B collisions skipped (don't care)
  int  n_psbusy_rd = 0;
  bit  prev_rst = 1'b0;

  task automatic fail(input string msg);
    errors++;
    if (errors <= 20) $display("ERROR cyc=%0d: %s", cyc, msg);
  endtask

  task automatic idle_inputs();
    ps_en = 0; ps_we = 0; ps_be = '0; ps_addr = '0; ps_wdata = '0;
    acc_wr_en = 0; acc_wr_addr = '0; acc_wr_data = '0; acc_wr_be = '0;
    in_valid = 0; in_tok = '0;
    for (int b = 0; b < N; b++) rd_addr[b] = '0;
  endtask

  // Apply the inputs currently driven for one clock and check all outputs.
  task automatic step();
    int s0, sm1;
    logic [7:0] a_we;
    logic [ACT_AW-1:0] a_addr;
    logic [63:0] a_din;
    s0  = cyc % 4;
    sm1 = (cyc + 3) % 4;
    // --- expectations of this request (pre-write state: read-first / reads before writes)
    ps_chk[s0] = ps_en && !rst;
    if (ps_en) begin
      for (int b = 0; b < N; b++) begin
        if (accel_busy) begin
          ps_exp[s0][8*b +: 8] = 8'h00; ps_kn[s0][b] = 1;
        end else begin
          ps_exp[s0][8*b +: 8] = sh[b][ps_addr]; ps_kn[s0][b] = known[b][ps_addr];
        end
      end
      if (accel_busy && !ps_we) n_psbusy_rd++;
    end
    if (accel_busy) begin a_addr = acc_wr_addr; a_din = acc_wr_data; a_we = acc_wr_en ? acc_wr_be : '0; end
    else            begin a_addr = ps_addr;     a_din = ps_wdata;    a_we = (ps_en && ps_we) ? ps_be : '0; end
    b_v[s0]   = in_valid && !rst;
    b_tok[s0] = in_tok;
    for (int b = 0; b < N; b++) begin
      b_dat[s0][b] = sh[b][rd_addr[b]];
      b_chk[s0][b] = known[b][rd_addr[b]];
      if (in_valid && a_we[b] && a_addr == rd_addr[b]) begin
        b_chk[s0][b] = 0; n_dc++;
      end
    end
    // --- apply writes / flag to the model
    for (int b = 0; b < N; b++) begin
      if (a_we[b]) begin sh[b][a_addr] = a_din[8*b +: 8]; known[b][a_addr] = 1; end
    end
    if (rst) flag_m = 0;
    else if (ps_en && accel_busy) flag_m = 1;

    @(posedge clk); #1;

    // --- checks
    for (int d = 0; d < 2; d++) begin
      if (viol[d] !== flag_m) fail($sformatf("dut%0d ps_busy_violation=%b exp %b", d+1, viol[d], flag_m));
      checks++;
    end
    // PS latency 1 (request s0), latency 2 (request sm1)
    if (ps_chk[s0]) begin
      for (int b = 0; b < N; b++) if (ps_kn[s0][b]) begin
        if (ps_rdata[0][8*b +: 8] !== ps_exp[s0][8*b +: 8])
          fail($sformatf("dut1 ps_rdata byte %0d = %h exp %h", b, ps_rdata[0][8*b +: 8], ps_exp[s0][8*b +: 8]));
        checks++;
      end
    end
    if (cyc > 0 && ps_chk[sm1] && !rst) begin
      for (int b = 0; b < N; b++) if (ps_kn[sm1][b]) begin
        if (ps_rdata[1][8*b +: 8] !== ps_exp[sm1][8*b +: 8])
          fail($sformatf("dut2 ps_rdata byte %0d = %h exp %h", b, ps_rdata[1][8*b +: 8], ps_exp[sm1][8*b +: 8]));
        checks++;
      end
    end
    // port B, latency 2 (request sm1); a reset in this or the previous cycle kills the valid
    for (int d = 0; d < 2; d++) begin
      bit ev;
      ev = (cyc > 0) && b_v[sm1] && !rst && !prev_rst;
      if (out_valid[d] !== ev) fail($sformatf("dut%0d out_valid=%b exp %b", d+1, out_valid[d], ev));
      checks++;
      if (ev) begin
        if (out_tok[d] !== b_tok[sm1]) fail($sformatf("dut%0d out_tok=%h exp %h", d+1, out_tok[d], b_tok[sm1]));
        checks++;
        for (int b = 0; b < N; b++) if (b_chk[sm1][b]) begin
          if (rd_data[d][b] !== b_dat[sm1][b])
            fail($sformatf("dut%0d rd_data bank %0d = %h exp %h", d+1, b, rd_data[d][b], b_dat[sm1][b]));
          checks++;
        end
      end
    end
    prev_rst = rst;
    cyc++;
    #1;   // leave the posedge region before the caller drives the next inputs
  endtask

  // ---------------------------------------------------------------- real image load
  logic [63:0] img [0:ACT_DEPTH-1];

  task automatic image_test(input string vec_dir, input string net);
    string fn;
    int n;
    fn = {vec_dir, "/", net, "/net/img0/act0.hex"};
    for (int i = 0; i < ACT_DEPTH; i++) img[i] = 'x;
    $readmemh(fn, img);
    n = 0;
    while (n < ACT_DEPTH && !$isunknown(img[n])) n++;
    if (n == 0) begin
      $display("TEST FAILED cannot read %s", fn);
      $fatal(1, "missing vector file");
    end
    $display("image %s: %0d words", fn, n);
    // PS write (not busy), full byte enables
    idle_inputs(); accel_busy = 0;
    for (int w = 0; w < n; w++) begin
      ps_en = 1; ps_we = 1; ps_be = 8'hff; ps_addr = w[ACT_AW-1:0]; ps_wdata = img[w];
      step();
    end
    idle_inputs();
    // the model now holds the image: compare it to the file directly
    for (int w = 0; w < n; w++) for (int b = 0; b < N; b++) begin
      if (sh[b][w] !== img[w][8*b +: 8] || !known[b][w]) fail("TB shadow != image");
    end
    // accelerator side (busy, as during a layer): same address per bank, then per-bank rotated
    accel_busy = 1;
    for (int w = 0; w < n; w++) begin
      in_valid = 1; in_tok = TW'(w);
      for (int b = 0; b < N; b++) rd_addr[b] = w[ACT_AW-1:0];
      step();
    end
    for (int w = 0; w < n; w++) begin
      in_valid = 1; in_tok = TW'(w ^ 16'h5a5a);
      for (int b = 0; b < N; b++) rd_addr[b] = ACT_AW'((w + 3*b) % n);
      step();
    end
    // explicit direct compare: one read, then compare the DUT output to the image words
    for (int w = 0; w < n; w += 17) begin
      in_valid = 1; in_tok = '0;
      for (int b = 0; b < N; b++) rd_addr[b] = ACT_AW'((w + b) % n);
      step(); in_valid = 0; step(); // output of the read is visible now
      for (int d = 0; d < 2; d++) for (int b = 0; b < N; b++) begin
        if (rd_data[d][b] !== img[(w + b) % n][8*b +: 8]) fail($sformatf("dut%0d image %s word %0d bank %0d", d+1, net, (w+b)%n, b));
        checks++;
      end
    end
    idle_inputs(); accel_busy = 0;
    // PS read-back
    for (int w = 0; w < n; w++) begin
      ps_en = 1; ps_we = 0; ps_be = '0; ps_addr = w[ACT_AW-1:0];
      step();
    end
    idle_inputs(); step(); step();
  endtask

  // ---------------------------------------------------------------- stimulus
  function automatic logic [ACT_AW-1:0] rnd_addr();
    int r;
    r = $urandom_range(0, 99);
    if (r < 10) return ACT_AW'($urandom_range(0, 7));
    if (r < 20) return ACT_AW'(ACT_DEPTH - 1 - $urandom_range(0, 7));
    if (r < 50) return ACT_AW'($urandom_range(0, 31));   // hot region: many read-after-write hits
    return ACT_AW'($urandom);
  endfunction

  function automatic logic [7:0] rnd_be();
    int r;
    r = $urandom_range(0, 9);
    if (r == 0) return 8'hff;
    if (r == 1) return 8'h00;
    return 8'($urandom);
  endfunction

  string vec_dir;
  bit busy_run;
  int run_left;

  initial begin
    process::self().srandom(32'h0AC7_B0F1);
    if (!$value$plusargs("VEC_DIR=%s", vec_dir)) vec_dir = "../../../vectors/generated";
    for (int b = 0; b < N; b++) for (int w = 0; w < ACT_DEPTH; w++) known[b][w] = 0;
    flag_m = 0;
    idle_inputs(); accel_busy = 0;

    // 1. reset, fill
    rst = 1; repeat (3) step(); rst = 0;
    for (int w = 0; w < ACT_DEPTH; w++) begin
      ps_en = 1; ps_we = 1; ps_be = 8'hff; ps_addr = w[ACT_AW-1:0];
      ps_wdata = {$urandom, $urandom};
      step();
    end
    idle_inputs(); step();
    for (int b = 0; b < N; b++) for (int w = 0; w < ACT_DEPTH; w++)
      if (!known[b][w]) fail("fill incomplete");

    // 2a. wrap: counter base 4095 + offset b -> 12-bit field wraps to 0..6 (banks 1..7)
    for (int base = ACT_DEPTH - 8; base < ACT_DEPTH + 8; base++) begin
      in_valid = 1; in_tok = TW'(base);
      for (int b = 0; b < N; b++) begin
        int full;
        full = base + b + (b < 3 ? 1 : 0);        // as the rotator counter (rowbase + ox0/8 + (b<kx))
        rd_addr[b] = full[ACT_AW-1:0];              // only bits 11:0 reach the port
      end
      step();
    end
    // wrap direct compare: bank b reads counter value 4095 + b
    in_valid = 1; in_tok = '0;
    for (int b = 0; b < N; b++) begin int full; full = 4095 + b; rd_addr[b] = full[ACT_AW-1:0]; end
    step(); in_valid = 0; step();
    for (int d = 0; d < 2; d++) for (int b = 0; b < N; b++) begin
      if (rd_data[d][b] !== sh[b][(4095 + b) % ACT_DEPTH]) fail($sformatf("dut%0d wrap bank %0d", d+1, b));
      checks++;
    end
    // 2b. independence: 8 distinct addresses per cycle; also each bank at a different address from
    //     a pattern whose bytes differ from what the same address of other banks holds
    for (int t = 0; t < 512; t++) begin
      in_valid = 1; in_tok = TW'(t);
      for (int b = 0; b < N; b++) rd_addr[b] = ACT_AW'(t * 7 + b * 523);
      step();
    end
    idle_inputs(); step(); step();

    // 3. busy mux, directed
    accel_busy = 1;
    ps_en = 1; ps_we = 1; ps_be = 8'hff; ps_addr = 12'd100; ps_wdata = 64'hdead_beef_cafe_f00d;
    step();                                            // PS write while busy: ignored, flag set
    idle_inputs();
    if (viol[0] !== 1'b1 || viol[1] !== 1'b1) fail("flag not set by PS write while busy");
    ps_en = 1; ps_we = 0; ps_addr = 12'd100; step();   // PS read while busy: 0
    idle_inputs(); step(); step();
    rst = 1; step(); rst = 0;                          // rst clears the sticky flag
    if (viol[0] !== 1'b0 || viol[1] !== 1'b0) fail("flag not cleared by rst");
    accel_busy = 0;
    ps_en = 1; ps_we = 0; ps_addr = 12'd100; step();   // PS read of word 100 not modified
    idle_inputs(); step(); step();
    accel_busy = 1;                                    // PS read while busy alone sets the flag
    ps_en = 1; ps_we = 0; ps_addr = 12'd5; step(); idle_inputs(); step();
    if (viol[0] !== 1'b1 || viol[1] !== 1'b1) fail("flag not set by PS read while busy");
    rst = 1; step(); rst = 0;
    accel_busy = 0;                                    // accel writes while not busy: ignored
    acc_wr_en = 1; acc_wr_be = 8'hff; acc_wr_addr = 12'd200; acc_wr_data = 64'h0123_4567_89ab_cdef;
    step(); idle_inputs();
    ps_en = 1; ps_we = 0; ps_addr = 12'd200; step(); idle_inputs(); step(); step();
    accel_busy = 1;                                    // accel writes while busy: land (byte enables)
    acc_wr_en = 1; acc_wr_be = 8'b1010_0101; acc_wr_addr = 12'd200; acc_wr_data = 64'h0123_4567_89ab_cdef;
    step(); idle_inputs(); step();
    in_valid = 1; for (int b = 0; b < N; b++) rd_addr[b] = 12'd200; step();
    idle_inputs(); step(); step();
    accel_busy = 0;
    ps_en = 1; ps_we = 0; ps_addr = 12'd200; step(); idle_inputs(); step(); step();

    // 4. random mix
    busy_run = 0; run_left = 0;
    for (int t = 0; t < N_RAND; t++) begin
      if (run_left == 0) begin
        busy_run = $urandom_range(0, 1);
        run_left = $urandom_range(1, 60);
      end
      run_left--;
      accel_busy = busy_run;
      // PS: mostly idle while busy (occasional violations), active while idle
      ps_en    = busy_run ? ($urandom_range(0, 49) == 0) : ($urandom_range(0, 9) < 7);
      ps_we    = $urandom_range(0, 1);
      ps_be    = rnd_be();
      ps_addr  = rnd_addr();
      ps_wdata = {$urandom, $urandom};
      acc_wr_en   = busy_run ? ($urandom_range(0, 9) < 6) : ($urandom_range(0, 9) == 0);
      acc_wr_be   = rnd_be();
      acc_wr_addr = rnd_addr();
      acc_wr_data = {$urandom, $urandom};
      in_valid = $urandom_range(0, 9) < 7;
      in_tok   = TW'($urandom);
      for (int b = 0; b < N; b++) rd_addr[b] = rnd_addr();
      rst = ($urandom_range(0, 1999) == 0);
      step();
      rst = 0;
    end
    idle_inputs(); accel_busy = 0;
    rst = 1; step(); rst = 0; step(); step();

    // 5. real images
    image_test(vec_dir, "lenet5");
    image_test(vec_dir, "cifar10");

    $display("info: port-B collisions skipped (don't care) = %0d, PS reads while busy = %0d", n_dc, n_psbusy_rd);
    if (errors == 0 && checks > 0) $display("TEST PASSED checks=%0d", checks);
    else begin
      $display("TEST FAILED errors=%0d checks=%0d", errors, checks);
      $fatal(1, "tb_gos_act_buf failed");
    end
    $finish;
  end

endmodule
