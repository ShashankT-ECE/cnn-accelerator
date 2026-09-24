// GOS_CORE_TB: gos_pkg.sv gos_cfg_check.sv gos_ctrl.sv gos_act_buf.sv gos_wgt_mem.sv gos_qparam_mem.sv gos_rotator.sv gos_pe.sv gos_array.sv gos_requant.sv gos_pool.sv gos_core.sv
`timescale 1ns/1ps
// tb_gos_core — whole-core verification (V2 step 4). Self-checking, vector-driven.
//
// +SUITE=layer|net|fuzz|checker   (one suite per run; run_core.sh runs all four)
//   layer   : every layer of both nets x 3 images as a 1-layer job. Output ACT (masked)
//             or LOGIT bit-exact; LAYER_CYC / TOTAL_CYC / MAC_ACTIVE == model; STALL == 0;
//             issue stream == <layer>/issue.hex cycle by cycle; ACT write events ==
//             <layer>/drain.hex (kind ACT, we) in order.
//   net     : 10 images per net, all layers from one start. LOGIT[0..15] bit-exact;
//             per-layer + total cycles == model. Logits printed for the Python prediction check.
//   fuzz    : 100 random single-layer shapes (tile-model expected), bit-exact, cycles == model.
//   checker : config-checker cases: ERR_CODE == expected; accepted jobs start (then soft_reset).
// Memories are loaded only through the PS ports (PS_RD_LAT = 1); every memory is first
// filled with random garbage. Prints "RESULT ..." lines (parsed by v2/scripts/core_collect.py)
// and "TEST PASSED checks=<n>" / "TEST FAILED ...".
module tb_gos_core;
  import gos_pkg::*;

  localparam int ISSUE_W = 256;
  localparam int DRAIN_W = 160;
  localparam int MAX_ISS = 65536;
  localparam int MAX_DRN = 65536;

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

  // ---------------------------------------------------------------- bookkeeping
  string  vec, suite;
  longint checks = 0;
  int     errors = 0;

  task automatic fail(input string msg);
    errors++;
    if (errors <= 20) $display("MISMATCH: %s", msg);
  endtask
  task automatic chk(input bit ok, input string msg);
    checks++;
    if (!ok) fail(msg);
  endtask

  // ---------------------------------------------------------------- hex loading
  logic [63:0]  f64 [0:16383];
  logic [31:0]  f32 [0:65535];
  logic [7:0]   f8  [0:4095];
  logic [ISSUE_W-1:0] iss_mem [0:MAX_ISS-1];
  logic [DRAIN_W-1:0] drn_mem [0:MAX_DRN-1];

  function automatic string p(input string rel);
    return $sformatf("%s/%s", vec, rel);
  endfunction
  task automatic rd64(input string rel);
    for (int i = 0; i < 16384; i++) f64[i] = 'x;
    $readmemh(p(rel), f64);
  endtask
  task automatic rd32(input string rel);
    for (int i = 0; i < 65536; i++) f32[i] = 'x;
    $readmemh(p(rel), f32);
  endtask
  task automatic rd8(input string rel);
    for (int i = 0; i < 4096; i++) f8[i] = 'x;
    $readmemh(p(rel), f8);
  endtask

  // ---------------------------------------------------------------- PS port tasks
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
  task automatic act_read(input int b, input int addr, output logic [63:0] d);
    @(negedge clk);
    act_en[b] = 1; act_we[b] = 0; act_be[b] = 0; act_addr[b] = addr[ACT_AW-1:0];
    @(negedge clk);                       // PS_RD_LAT = 1: data valid after the enabled edge
    act_en[b] = 0;
    d = act_rdata[b];
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

  // net-level images
  task automatic load_net_mem(input string net);
    rd64($sformatf("%s/wgt.hex", net));
    for (int a = 0; a < WGT_DEPTH && f64[a] !== 'x; a++) wgt_write(a, f64[a]);
    rd64($sformatf("%s/qparam.hex", net));             // combined view: PS word address = 2i / 2i+1
    for (int a = 0; a < 2 * QP_CH && f64[a] !== 'x; a++) qp_write(a, f64[a]);
  endtask
  task automatic load_act(input int b, input string rel, input int n);
    rd64(rel);
    for (int a = 0; a < n; a++) act_write(b, a, f64[a]);
  endtask
  task automatic set_desc_from(input string rel, input int nl);
    rd32(rel);
    for (int l = 0; l < 8; l++)
      for (int w = 0; w < 16; w++)
        desc[l][w] = (l < nl) ? f32[16 * l + w] : $urandom;   // unused slots: garbage
    n_layers = nl[3:0];
  endtask

  // ---------------------------------------------------------------- run a job
  int wait_limit;
  task automatic run_job();
    @(negedge clk); start = 1;
    @(negedge clk); start = 0;
    wait_limit = 2_000_000;
    while (busy && wait_limit > 0) begin @(negedge clk); wait_limit--; end
    if (wait_limit == 0) begin
      $display("TEST FAILED: job timeout"); $fatal(1, "job timeout");
    end
  endtask

  // ---------------------------------------------------------------- stream monitors
  bit mon_en = 0;
  int n_iss, n_wr_exp, iss_i, wr_i, iss_bad, wr_bad;
  logic [15:0] wr_word_exp [0:MAX_DRN-1];
  logic [7:0]  wr_be_exp   [0:MAX_DRN-1];

  task automatic arm_monitors(input string ld);
    rd32($sformatf("%s/expect_tk.hex", ld));
    n_iss = f32[2];
    $readmemh(p($sformatf("%s/issue.hex", ld)), iss_mem);
    $readmemh(p($sformatf("%s/drain.hex", ld)), drn_mem);
    n_wr_exp = 0;
    for (int i = 0; i < 8 * f32[0]; i++) begin
      if (drn_mem[i][71:68] == 4'd0 && drn_mem[i][72]) begin
        wr_word_exp[n_wr_exp] = drn_mem[i][15:0];
        wr_be_exp[n_wr_exp]   = drn_mem[i][23:16];
        n_wr_exp++;
      end
    end
    iss_i = 0; wr_i = 0; iss_bad = 0; wr_bad = 0; mon_en = 1;
  endtask

  always @(posedge clk) if (mon_en) begin
    if (dut.u_ctrl.iss_valid) begin
      automatic logic [ISSUE_W-1:0] e = iss_mem[iss_i];
      automatic bit ok = 1;
      for (int b = 0; b < N; b++) ok &= (dut.u_ctrl.iss_rd[b] == e[16*b +: 16]);
      ok &= (dut.u_ctrl.iss_wgt == e[143:128]);
      ok &= (dut.u_ctrl.iss_tok.row_mask == e[151:144]);
      ok &= ({1'b0, dut.u_ctrl.iss_rot} == e[155:152]);
      ok &= (dut.u_ctrl.iss_first == e[156]) && (dut.u_ctrl.iss_last == e[157]);
      ok &= (dut.u_ctrl.iss_dy == e[158]);
      ok &= (dut.u_ctrl.iss_k == e[175:160]) && (dut.u_ctrl.iss_ic == e[191:176]);
      ok &= (dut.u_ctrl.iss_ky == e[199:192]) && (dut.u_ctrl.iss_kx == e[207:200]);
      ok &= (dut.u_ctrl.iss_ox_tile == e[215:208]) && (dut.u_ctrl.iss_oy == e[223:216]);
      ok &= (dut.u_ctrl.iss_oc_tile == e[231:224]) && (dut.u_ctrl.iss_tile == e[255:240]);
      if (iss_i >= n_iss) ok = 0;
      if (!ok) begin
        iss_bad++;
        if (iss_bad <= 3) $display("ISSUE MISMATCH at %0d: rtl rd0=%h wgt=%h k=%0d tile=%0d exp=%h",
                                   iss_i, dut.u_ctrl.iss_rd[0], dut.u_ctrl.iss_wgt,
                                   dut.u_ctrl.iss_k, dut.u_ctrl.iss_tile, e);
      end
      iss_i++;
    end
    if (dut.wr_en) begin
      if (wr_i >= n_wr_exp || dut.wr_addr != wr_word_exp[wr_i][ACT_AW-1:0] || dut.wr_be != wr_be_exp[wr_i]) begin
        wr_bad++;
        if (wr_bad <= 3) $display("WRITE MISMATCH at %0d: rtl word=%h be=%h exp word=%h be=%h",
                                  wr_i, dut.wr_addr, dut.wr_be, wr_word_exp[wr_i], wr_be_exp[wr_i]);
      end
      wr_i++;
    end
  end

  // ---------------------------------------------------------------- compare helpers
  bit out_ok;
  task automatic compare_act(input int b, input string exp_rel, input string mask_rel, input int n);
    logic [63:0] got, mk;
    rd64(exp_rel);
    rd8(mask_rel);
    out_ok = 1;
    for (int a = 0; a < n; a++) begin
      act_read(b, a, got);
      for (int i = 0; i < 8; i++) mk[8*i +: 8] = f8[a][i] ? 8'hFF : 8'h00;
      checks++;
      if (((got ^ f64[a]) & mk) != 0) begin
        out_ok = 0;
        fail($sformatf("%s word %0d: got %h exp %h mask %h", exp_rel, a, got, f64[a], mk));
      end
    end
  endtask
  task automatic compare_logit(input string exp_rel, input int n);
    rd32(exp_rel);
    out_ok = 1;
    for (int i = 0; i < n; i++) begin
      checks++;
      if (logit[i] !== f32[i]) begin
        out_ok = 0;
        fail($sformatf("%s LOGIT[%0d]: got %h exp %h", exp_rel, i, logit[i], f32[i]));
      end
    end
  endtask

  // expected cycles file: LAYER_CYC[0..nl-1], TOTAL, MAC
  logic [31:0] exp_cyc [0:15];
  bit cyc_ok;
  task automatic check_cycles(input string rel, input int nl, input string tag);
    rd32(rel);
    for (int i = 0; i < nl + 2; i++) exp_cyc[i] = f32[i];
    cyc_ok = 1;
    for (int l = 0; l < nl; l++) if (layer_cyc[l] != exp_cyc[l]) cyc_ok = 0;
    if (total_cyc != 64'(exp_cyc[nl])) cyc_ok = 0;
    if (mac_active != 64'(exp_cyc[nl + 1])) cyc_ok = 0;
    if (stall != 0) cyc_ok = 0;
    chk(cyc_ok, $sformatf("%s cycles: layer0 %0d/%0d total %0d/%0d mac %0d/%0d stall %0d",
                          tag, layer_cyc[0], exp_cyc[0], total_cyc, exp_cyc[nl], mac_active,
                          exp_cyc[nl + 1], stall));
  endtask

  function automatic string net_layer_dir(input string net, input int l);
    string lenet[5] = '{"L0_conv1", "L1_conv3", "L2_conv5", "L3_fc1", "L4_fc2"};
    string cifar[4] = '{"L0_conv1", "L1_conv2", "L2_conv3", "L3_fc"};
    return (net == "lenet5") ? lenet[l] : cifar[l];
  endfunction

  // ---------------------------------------------------------------- suites
  task automatic suite_layer();
    string nets[2] = '{"lenet5", "cifar10"};
    int nls[2] = '{5, 4};
    foreach (nets[ni]) begin
      string net = nets[ni];
      load_net_mem(net);
      for (int l = 0; l < nls[ni]; l++) begin
        string ld = $sformatf("%s/layers/%s", net, net_layer_dir(net, l));
        int in_sel, out_raw, in_end, out_end;
        set_desc_from($sformatf("%s/desc.hex", ld), 1);
        in_sel  = desc[0][6][3];
        out_raw = desc[0][6][2];
        in_end  = desc[0][13];
        out_end = desc[0][14];
        for (int img = 0; img < 3; img++) begin
          string id = $sformatf("%s/img%0d", ld, img);
          load_act(in_sel, $sformatf("%s/act_in.hex", id), in_end + 1);
          arm_monitors(ld);
          run_job();
          mon_en = 0;
          chk(done && !error, $sformatf("%s: done=%0d error=%0d code=%h", id, done, error, err_code));
          chk(iss_bad == 0 && iss_i == n_iss, $sformatf("%s: issue stream bad=%0d n=%0d/%0d", id, iss_bad, iss_i, n_iss));
          chk(wr_bad == 0 && wr_i == n_wr_exp, $sformatf("%s: write stream bad=%0d n=%0d/%0d", id, wr_bad, wr_i, n_wr_exp));
          if (out_raw) compare_logit($sformatf("%s/logit16.hex", id), 16);
          else compare_act(1 - in_sel, $sformatf("%s/act_out.hex", id), $sformatf("%s/act_out_mask.hex", ld), out_end + 1);
          check_cycles($sformatf("%s/expect_cyc.hex", ld), 1, id);
          $display("RESULT kind=layer net=%s layer=%s img=%0d out_ok=%0d issue_ok=%0d write_ok=%0d rtl_cycles=%0d model_cycles=%0d rtl_total=%0d model_total=%0d mac=%0d model_mac=%0d stall=%0d",
                   net, net_layer_dir(net, l), img, out_ok, (iss_bad == 0 && iss_i == n_iss),
                   (wr_bad == 0 && wr_i == n_wr_exp), layer_cyc[0], exp_cyc[0], total_cyc, exp_cyc[1],
                   mac_active, exp_cyc[2], stall);
        end
      end
    end
  endtask

  task automatic suite_net();
    string nets[2] = '{"lenet5", "cifar10"};
    int nls[2] = '{5, 4};
    foreach (nets[ni]) begin
      string net = nets[ni];
      string s;
      int in_end;
      load_net_mem(net);
      set_desc_from($sformatf("%s/desc.hex", net), nls[ni]);
      in_end = desc[0][13];
      for (int img = 0; img < 10; img++) begin
        string id = $sformatf("%s/net/img%0d", net, img);
        load_act(0, $sformatf("%s/act0.hex", id), in_end + 1);
        run_job();
        chk(done && !error, $sformatf("%s: done=%0d error=%0d", id, done, error));
        compare_logit($sformatf("%s/logit16.hex", id), 16);
        check_cycles($sformatf("%s/net/expect_cyc.hex", net), nls[ni], id);
        s = "";
        for (int i = 0; i < 16; i++) s = {s, (i ? "," : ""), $sformatf("%0d", $signed(logit[i]))};
        $display("RESULT kind=net net=%s img=%0d logits_ok=%0d cycles_ok=%0d rtl_total=%0d model_total=%0d mac=%0d stall=%0d logits=%s layer_cycles=%0d,%0d,%0d,%0d,%0d model_layer_cycles=%0d,%0d,%0d,%0d,%0d",
                 net, img, out_ok, cyc_ok, total_cyc, exp_cyc[nls[ni]], mac_active, stall, s,
                 layer_cyc[0], layer_cyc[1], layer_cyc[2], layer_cyc[3], (nls[ni] > 4) ? layer_cyc[4] : 0,
                 exp_cyc[0], exp_cyc[1], exp_cyc[2], exp_cyc[3], (nls[ni] > 4) ? exp_cyc[4] : 0);
      end
    end
  endtask

  task automatic suite_fuzz();
    int n;
    rd32("fuzz/n_cases.hex");
    n = f32[0];
    for (int c = 0; c < n; c++) begin
      string d = $sformatf("fuzz/F%03d", c);
      int in_sel, out_raw, in_end, out_end, wbase, qbase, oc;
      set_desc_from($sformatf("%s/desc.hex", d), 1);
      in_sel = desc[0][6][3]; out_raw = desc[0][6][2];
      in_end = desc[0][13];   out_end = desc[0][14];
      wbase  = desc[0][4];    qbase = desc[0][5];   oc = desc[0][0][31:16];
      rd64($sformatf("%s/wgt.hex", d));
      for (int a = 0; a < WGT_DEPTH && f64[a] !== 'x; a++) wgt_write(wbase + a, f64[a]);
      rd64($sformatf("%s/qp_e.hex", d));
      for (int i = 0; i < oc; i++) qp_write(2 * (qbase + i), f64[i]);
      rd64($sformatf("%s/qp_o.hex", d));
      for (int i = 0; i < oc; i++) qp_write(2 * (qbase + i) + 1, f64[i]);
      load_act(in_sel, $sformatf("%s/act_in.hex", d), in_end + 1);
      run_job();
      chk(done && !error, $sformatf("%s: done=%0d error=%0d code=%h", d, done, error, err_code));
      if (out_raw) compare_logit($sformatf("%s/logit16.hex", d), oc);
      else compare_act(1 - in_sel, $sformatf("%s/act_out.hex", d), $sformatf("%s/act_out_mask.hex", d), out_end + 1);
      check_cycles($sformatf("%s/expect_cyc.hex", d), 1, d);
      $display("RESULT kind=fuzz case=%0d out_ok=%0d rtl_cycles=%0d model_cycles=%0d rtl_total=%0d model_total=%0d mac=%0d model_mac=%0d stall=%0d",
               c, out_ok, layer_cyc[0], exp_cyc[0], total_cyc, exp_cyc[1], mac_active, exp_cyc[2], stall);
    end
  endtask

  task automatic suite_checker();
    int n, base, exp_code;
    bit ok;
    rd32("checker/n_cases.hex");
    n = f32[0];
    rd32("checker/cases.hex");
    for (int c = 0; c < n; c++) begin
      base = c * 130;
      n_layers = f32[base][3:0];
      for (int l = 0; l < 8; l++)
        for (int w = 0; w < 16; w++) desc[l][w] = f32[base + 1 + 16 * l + w];
      exp_code = f32[base + 129];
      @(negedge clk); start = 1;
      @(negedge clk); start = 0;
      // wait for the checker's decision: refused (busy falls) or accepted (first S_LOAD)
      for (int w = 0; w < 16 && busy && !dut.ctrl_load; w++) @(negedge clk);
      @(negedge clk);
      if (exp_code != 0) ok = !busy && error && (err_code == exp_code);
      else               ok = busy && !error;
      chk(ok, $sformatf("checker case %0d: busy=%0d error=%0d code=%h exp=%h", c, busy, error, err_code, exp_code));
      $display("RESULT kind=checker case=%0d exp_code=%0d rtl_code=%0d rtl_error=%0d rtl_busy=%0d ok=%0d",
               c, exp_code, err_code, error, busy, ok);
      if (busy) begin                      // accepted: abort with soft_reset
        @(negedge clk); soft_reset = 1;
        @(negedge clk); soft_reset = 0;
        @(negedge clk);
        chk(!busy && !error && !done, $sformatf("checker case %0d: soft_reset did not idle", c));
      end
    end
  endtask

  // ---------------------------------------------------------------- main
  initial begin
    if (!$value$plusargs("VEC_DIR=%s", vec)) begin
      $display("TEST FAILED: no VEC_DIR"); $fatal(1, "no VEC_DIR");
    end
    if (!$value$plusargs("SUITE=%s", suite)) suite = "layer";
    ps_idle();
    start = 0; soft_reset = 0; n_layers = 0; desc = '0;
    rst = 1;
    repeat (5) @(negedge clk);
    rst = 0;
    fill_garbage();
    case (suite)
      "layer":   suite_layer();
      "net":     suite_net();
      "fuzz":    suite_fuzz();
      "checker": suite_checker();
      default: begin $display("TEST FAILED: unknown SUITE %s", suite); $fatal(1, "bad suite"); end
    endcase
    chk(err_flags == 0, $sformatf("sticky error flags %h", err_flags));
    if (errors == 0) $display("TEST PASSED checks=%0d suite=%s", checks, suite);
    else begin
      $display("TEST FAILED suite=%s errors=%0d checks=%0d", suite, errors, checks);
      $fatal(1, "tb_gos_core failed");
    end
    $finish;
  end
endmodule
