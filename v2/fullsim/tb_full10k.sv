// GOS_FULL10K_TB: gos_pkg.sv gos_cfg_check.sv gos_ctrl.sv gos_act_buf.sv gos_wgt_mem.sv gos_qparam_mem.sv gos_rotator.sv gos_pe.sv gos_array.sv gos_requant.sv gos_pool.sv gos_core.sv
`timescale 1ns/1ps
// tb_full10k — full-test-set whole-network RTL simulation of gos_core (harness, not a unit TB).
//
// Same procedure as tb_gos_core SUITE=net (v2/tb/tb_gos_core.sv), over an image range inside ONE
// simulation: memories filled with random garbage through the PS ports (PS_RD_LAT = 1), WGT /
// QPARAM / descriptors loaded once, then per image i in [IMG_START, IMG_END): write ACT0 words
// 0..IN_END, pulse start, wait !busy, check done && !error, LOGIT[0..15] bit-exact vs the golden
// (unused = 0), LAYER_CYC[0..NL-1] / TOTAL_CYC / MAC_ACTIVE == model and STALL == 0.
// Mismatches do not stop the run (the shard summary counts them).
//
// Plusargs: +DATA=<v2/build/fullsim/data/<net>> +NET=<name> +IMG_START=<a> +IMG_END=<b>
// Data files (v2/fullsim/gen_full10k_data.py): sizes.hex, wgt.hex, qparam.hex, desc.hex,
// expect_cyc.hex, act0_b<k>.hex, logit16_b<k>.hex (block k = images [k*BLK, (k+1)*BLK)).
// Output: one "RESULT kind=full ..." line per image, then
//   "SHARD_DONE net=.. start=.. end=.. images=.. logits_ok=.. cycles_ok=.. first_bad=.. errors=.."
// and "TEST PASSED ..." / "TEST FAILED ..." (v2/fullsim/collect_full10k.py parses these).
// Simulator-portable: no reliance on 4-state 'x (file sizes come from sizes.hex).
module tb_full10k;
  import gos_pkg::*;

  localparam int ACTB_MAX = 65536;       // block * n_in words
  localparam int LGB_MAX  = 16384;       // block * 16 words

  logic clk = 1'b0;
  always #2.5 clk = ~clk;
  logic rst;

  // ---------------------------------------------------------------- DUT
  logic                   start, soft_reset;
  logic [3:0]             n_layers;
  logic [7:0][15:0][31:0] desc;
  logic                   busy, done, error;
  logic [31:0]            err_code;
  logic [63:0]            total_cyc, mac_active, stall;
  logic [7:0][31:0]       layer_cyc;
  logic [15:0][31:0]      logit;
  logic [7:0]             err_flags;
  logic                   act_en[2], act_we[2];
  logic [7:0]             act_be[2];
  logic [ACT_AW-1:0]      act_addr[2];
  logic [63:0]            act_wdata[2], act_rdata[2];
  logic                   wgt_en, wgt_we, qp_en, qp_we;
  logic [7:0]             wgt_be, qp_be;
  logic [WGT_AW-1:0]      wgt_addr;
  logic [QP_AW:0]         qp_addr;
  logic [63:0]            wgt_wdata, wgt_rdata, qp_wdata, qp_rdata;

  gos_core #(.PS_RD_LAT(1)) dut (
    .clk, .rst, .start, .soft_reset, .n_layers, .desc,
    .busy, .done, .error, .err_code, .total_cyc, .mac_active, .stall, .layer_cyc, .logit, .err_flags,
    .act0_en(act_en[0]), .act0_we(act_we[0]), .act0_be(act_be[0]), .act0_addr(act_addr[0]),
    .act0_wdata(act_wdata[0]), .act0_rdata(act_rdata[0]),
    .act1_en(act_en[1]), .act1_we(act_we[1]), .act1_be(act_be[1]), .act1_addr(act_addr[1]),
    .act1_wdata(act_wdata[1]), .act1_rdata(act_rdata[1]),
    .wgt_en, .wgt_we, .wgt_be, .wgt_addr, .wgt_wdata, .wgt_rdata,
    .qp_en, .qp_we, .qp_be, .qp_addr, .qp_wdata, .qp_rdata
  );

  // ---------------------------------------------------------------- files
  string data, net;
  logic [63:0] f64  [0:16383];
  logic [31:0] f32  [0:255];
  logic [63:0] actb [0:ACTB_MAX-1];
  logic [31:0] lgb  [0:LGB_MAX-1];
  int n_in, n_wgt, n_qp, nl, oc, blk, n_img;
  int cur_blk = -1;

  function automatic string p(input string rel);
    return $sformatf("%s/%s", data, rel);
  endfunction

  // ---------------------------------------------------------------- PS port tasks (as tb_gos_core)
  task automatic ps_idle();
    for (int b = 0; b < 2; b++) begin
      act_en[b] = 0; act_we[b] = 0; act_be[b] = 0; act_addr[b] = 0; act_wdata[b] = 0;
    end
    wgt_en = 0; wgt_we = 0; wgt_be = 0; wgt_addr = 0; wgt_wdata = 0;
    qp_en = 0; qp_we = 0; qp_be = 0; qp_addr = 0; qp_wdata = 0;
  endtask
  task automatic act_write(input int b, input int addr, input logic [63:0] d);
    @(negedge clk);
    act_en[b] = 1; act_we[b] = 1; act_be[b] = 8'hFF; act_addr[b] = addr[ACT_AW-1:0]; act_wdata[b] = d;
    @(negedge clk);
    act_en[b] = 0; act_we[b] = 0; act_be[b] = 0;
  endtask
  task automatic wgt_write(input int addr, input logic [63:0] d);
    @(negedge clk);
    wgt_en = 1; wgt_we = 1; wgt_be = 8'hFF; wgt_addr = addr[WGT_AW-1:0]; wgt_wdata = d;
    @(negedge clk);
    wgt_en = 0; wgt_we = 0; wgt_be = 0;
  endtask
  task automatic qp_write(input int addr9, input logic [63:0] d);
    @(negedge clk);
    qp_en = 1; qp_we = 1; qp_be = 8'hFF; qp_addr = addr9[QP_AW:0]; qp_wdata = d;
    @(negedge clk);
    qp_en = 0; qp_we = 0; qp_be = 0;
  endtask
  task automatic fill_garbage();
    for (int b = 0; b < 2; b++)
      for (int a = 0; a < ACT_DEPTH; a++) act_write(b, a, {$urandom, $urandom});
    for (int a = 0; a < WGT_DEPTH; a++) wgt_write(a, {$urandom, $urandom});
    for (int a = 0; a < 2 * QP_CH; a++) qp_write(a, {$urandom, $urandom});
  endtask

  task automatic run_job();
    int wait_limit;
    @(negedge clk); start = 1;
    @(negedge clk); start = 0;
    wait_limit = 2_000_000;
    while (busy && wait_limit > 0) begin @(negedge clk); wait_limit--; end
    if (wait_limit == 0) begin
      $display("TEST FAILED: job timeout"); $fatal(1, "job timeout");
    end
  endtask

  // ---------------------------------------------------------------- main
  logic [31:0] exp_cyc [0:15];
  int img_start, img_end, n_done, n_log_ok, n_cyc_ok, first_bad, errors;

  initial begin
    string s, lc;
    bit lg_ok, cy_ok;
    int base, lbase, nb;
    if (!$value$plusargs("DATA=%s", data)) begin
      $display("TEST FAILED: no +DATA"); $fatal(1, "no DATA");
    end
    if (!$value$plusargs("NET=%s", net)) net = "?";
    if (!$value$plusargs("IMG_START=%d", img_start)) img_start = 0;
    if (!$value$plusargs("IMG_END=%d", img_end)) img_end = img_start + 1;
    $readmemh(p("sizes.hex"), f32, 0, 6);
    n_in = f32[0]; n_wgt = f32[1]; n_qp = f32[2]; nl = f32[3]; oc = f32[4]; blk = f32[5]; n_img = f32[6];
    if (img_start < 0 || img_end > n_img || img_start >= img_end || blk * n_in > ACTB_MAX
        || blk * 16 > LGB_MAX || n_wgt > WGT_DEPTH || n_qp > 2 * QP_CH || nl < 1 || nl > 8) begin
      $display("TEST FAILED: bad range/sizes start=%0d end=%0d n_img=%0d blk=%0d n_in=%0d",
               img_start, img_end, n_img, blk, n_in);
      $fatal(1, "bad sizes");
    end
    $display("FULL10K_START net=%s start=%0d end=%0d n_in=%0d nl=%0d oc=%0d blk=%0d",
             net, img_start, img_end, n_in, nl, oc, blk);

    ps_idle();
    start = 0; soft_reset = 0; n_layers = 0; desc = '0;
    rst = 1;
    repeat (5) @(negedge clk);
    rst = 0;
    fill_garbage();

    // net-level memories, descriptors, expected cycles (once per shard)
    $readmemh(p("wgt.hex"), f64, 0, n_wgt - 1);
    for (int a = 0; a < n_wgt; a++) wgt_write(a, f64[a]);
    $readmemh(p("qparam.hex"), f64, 0, n_qp - 1);           // combined PS view: word 2i / 2i+1
    for (int a = 0; a < n_qp; a++) qp_write(a, f64[a]);
    $readmemh(p("desc.hex"), f32, 0, 16 * nl - 1);
    for (int l = 0; l < 8; l++)
      for (int w = 0; w < 16; w++)
        desc[l][w] = (l < nl) ? f32[16 * l + w] : $urandom;  // unused slots: garbage
    n_layers = nl[3:0];
    $readmemh(p("expect_cyc.hex"), f32, 0, nl + 1);
    for (int i = 0; i < nl + 2; i++) exp_cyc[i] = f32[i];

    n_done = 0; n_log_ok = 0; n_cyc_ok = 0; first_bad = -1; errors = 0;
    for (int img = img_start; img < img_end; img++) begin
      if (img / blk != cur_blk) begin
        cur_blk = img / blk;
        nb = (n_img - cur_blk * blk < blk) ? n_img - cur_blk * blk : blk;   // images in this block
        $readmemh(p($sformatf("act0_b%0d.hex", cur_blk)), actb, 0, nb * n_in - 1);
        $readmemh(p($sformatf("logit16_b%0d.hex", cur_blk)), lgb, 0, nb * 16 - 1);
      end
      base  = (img - cur_blk * blk) * n_in;
      lbase = (img - cur_blk * blk) * 16;
      for (int a = 0; a < n_in; a++) act_write(0, a, actb[base + a]);
      run_job();
      lg_ok = done && !error;
      for (int i = 0; i < 16; i++) if (logit[i] !== lgb[lbase + i]) lg_ok = 0;
      cy_ok = (stall == 0) && (total_cyc == 64'(exp_cyc[nl])) && (mac_active == 64'(exp_cyc[nl + 1]));
      for (int l = 0; l < nl; l++) if (layer_cyc[l] != exp_cyc[l]) cy_ok = 0;
      n_done++;
      if (lg_ok) n_log_ok++;
      if (cy_ok) n_cyc_ok++;
      if (!(lg_ok && cy_ok)) begin
        errors++;
        if (first_bad < 0) first_bad = img;
        if (errors <= 20)
          $display("MISMATCH: %s img %0d done=%0d error=%0d code=%h logits_ok=%0d cycles_ok=%0d total %0d/%0d",
                   net, img, done, error, err_code, lg_ok, cy_ok, total_cyc, exp_cyc[nl]);
      end
      s = "";
      for (int i = 0; i < 16; i++) s = {s, (i != 0 ? "," : ""), $sformatf("%0d", $signed(logit[i]))};
      lc = "";
      for (int l = 0; l < nl; l++) lc = {lc, (l != 0 ? "," : ""), $sformatf("%0d", layer_cyc[l])};
      $display("RESULT kind=full net=%s img=%0d logits_ok=%0d cycles_ok=%0d rtl_total=%0d model_total=%0d mac=%0d stall=%0d logits=%s layer_cycles=%s",
               net, img, lg_ok, cy_ok, total_cyc, exp_cyc[nl], mac_active, stall, s, lc);
    end
    if (err_flags != 0) begin
      errors++;
      $display("MISMATCH: sticky error flags %h", err_flags);
    end
    $display("SHARD_DONE net=%s start=%0d end=%0d images=%0d logits_ok=%0d cycles_ok=%0d first_bad=%0d errors=%0d",
             net, img_start, img_end, n_done, n_log_ok, n_cyc_ok, first_bad, errors);
    if (errors == 0) $display("TEST PASSED images=%0d net=%s", n_done, net);
    else begin
      $display("TEST FAILED net=%s errors=%0d images=%0d", net, errors, n_done);
      $fatal(1, "tb_full10k failed");
    end
    $finish;
  end
endmodule
