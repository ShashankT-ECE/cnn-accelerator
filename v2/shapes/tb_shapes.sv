// GOS_SHAPES_TB: gos_pkg.sv gos_cfg_check.sv gos_ctrl.sv gos_act_buf.sv gos_wgt_mem.sv gos_qparam_mem.sv gos_rotator.sv gos_pe.sv gos_array.sv gos_requant.sv gos_pool.sv gos_core.sv
`timescale 1ns/1ps
// tb_shapes — A3-general random-shape RTL harness (not a unit TB): runs a range (or list) of
// random multi-layer jobs from v2/shapes/gen_shapes.py through gos_core in ONE simulation.
//
// Once: rst, every memory filled with random garbage through the PS ports (PS_RD_LAT = 1).
// Per job (nothing is cleared between jobs: earlier jobs' data stays as garbage):
//   valid job : write WGT words at wgt_base.., QPARAM combined words at 2*QP_BASE.., ACT0 words
//               0..IN_END(layer 0), DESC slots 0..nd-1 (other slots: $urandom), N_LAYERS; pulse
//               start; wait !busy (timeout -> soft_reset, job marked bad). Check done && !error;
//               out_raw: LOGIT[0..OC-1] bit-exact and LOGIT[OC..15] == 0 (LOGIT is cleared at
//               start, DECISIONS); else ACT[out_buf] words 0..n_out-1 bit-exact under the byte
//               mask and LOGIT[0..15] == 0;
//               LAYER_CYC[0..nl-1], TOTAL_CYC, MAC_ACTIVE == model, STALL == 0, err_flags == 0.
//   refuse job: DESC + N_LAYERS only; start; the checker must refuse within 64 cycles: !busy,
//               error, ERR_CODE == expected (else soft_reset, job marked bad); TOTAL_CYC ==
//               model (C_START + C_DONE), MAC_ACTIVE == 0, STALL == 0.
// Mismatches never stop the run. Output per job:
//   RESULT kind=shape job=.. refuse=.. done=.. error=.. err_code=.. exp_code=.. err_ok=.. out_ok=..
//          out_sig=.. n_bad=.. lg_zero=.. cycles_ok=.. rtl_total=.. model_total=.. mac=.. model_mac=..
//          stall=.. flags=.. timeout=.. layer_cycles=a,b,.. logits=a,b,..   (logits: out_raw only)
// out_sig = FNV-style fold over the masked output words (the collector recomputes it from the npz).
// Then "SHARD_DONE sel=<range|list> start=.. end=.. jobs=.. ok=.. first_bad=.. errors=.." and
// "TEST PASSED ..." / "TEST FAILED ...".
//
// Plusargs: +DATA=<shapeset>/hex  and either +JOB_START=a +JOB_END=b  or  +JOBLIST=<file.hex>
// (count, then job ids). Portable to xsim and Verilator: sizes come from meta.hex, no 'x tests.
module tb_shapes;
  import gos_pkg::*;

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
  string data;
  logic [63:0] f64  [0:16383];
  logic [31:0] f32  [0:255];
  logic [31:0] meta [0:15];
  logic [63:0] oexp [0:4095];
  logic [7:0]  omsk [0:4095];
  logic [31:0] jl   [0:4096];
  logic [31:0] exp_cyc [0:9];         // module scope: xsim cannot $readmemh into automatic arrays

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

  // ---------------------------------------------------------------- one job
  localparam logic [63:0] SIG_INIT  = 64'hcbf29ce484222325;
  localparam logic [63:0] SIG_PRIME = 64'h00000100000001b3;
  int n_ok, n_bad_jobs, first_bad, n_run;

  task automatic run_one(input int job);
    string jd, lc, lg;
    int nl, nd, exp_code, wbase, n_wgt, qbase, n_qp, n_in, out_buf, n_out, n_lg, n_cyc;
    int n_bad, limit;
    bit refuse, out_raw, timeout;
    logic [63:0] got, mk, sig;
    bit err_ok, out_ok, lg_zero, cyc_ok, ok;
    jd = $sformatf("J%04d", job);
    $readmemh(p({jd, "/meta.hex"}), meta, 0, 15);
    nl = meta[0]; nd = meta[1]; refuse = (meta[2] != 0); exp_code = meta[3];
    wbase = meta[4]; n_wgt = meta[5]; qbase = meta[6]; n_qp = meta[7]; n_in = meta[8];
    out_raw = (meta[9] != 0); out_buf = meta[10]; n_out = meta[11]; n_lg = meta[12]; n_cyc = meta[13];
    if (nd < 1 || nd > 8 || n_wgt > WGT_DEPTH || n_qp > 2 * QP_CH || n_in > ACT_DEPTH
        || n_out > ACT_DEPTH || n_lg > 16 || (!refuse && (n_cyc != nl || nl < 1 || nl > 8))) begin
      $display("TEST FAILED: bad meta for job %0d", job);
      $fatal(1, "bad meta");
    end
    // descriptors (slots >= nd: garbage, ignored by the RTL for l >= N_LAYERS)
    $readmemh(p({jd, "/desc.hex"}), f32, 0, 16 * nd - 1);
    for (int l = 0; l < 8; l++)
      for (int w = 0; w < 16; w++)
        desc[l][w] = (l < nd) ? f32[16 * l + w] : $urandom;
    n_layers = nl[3:0];
    err_ok = 0; out_ok = 0; lg_zero = 0; cyc_ok = 0; sig = SIG_INIT; n_bad = 0; timeout = 0;
    if (refuse) begin
      @(negedge clk); start = 1;
      @(negedge clk); start = 0;
      for (int w = 0; w < 64 && busy; w++) @(negedge clk);
      if (busy) begin
        timeout = 1;
        @(negedge clk); soft_reset = 1;
        @(negedge clk); soft_reset = 0;
        @(negedge clk);
      end
      err_ok = !timeout && !busy && error && (err_code == 32'(exp_code));
      // model: busy only for the config check -> TOTAL_CYC = C_START + C_DONE, MAC_ACTIVE = 0
      cyc_ok = !timeout && (total_cyc == 64'(meta[14])) && (mac_active == 64'(meta[15])) && (stall == 0);
      ok = err_ok && cyc_ok && (err_flags == 0);
    end else begin
      $readmemh(p({jd, "/wgt.hex"}), f64, 0, n_wgt - 1);
      for (int a = 0; a < n_wgt; a++) wgt_write(wbase + a, f64[a]);
      $readmemh(p({jd, "/qparam.hex"}), f64, 0, n_qp - 1);   // combined view, from word 2*QP_BASE
      for (int a = 0; a < n_qp; a++) qp_write(qbase + a, f64[a]);
      $readmemh(p({jd, "/act_in.hex"}), f64, 0, n_in - 1);
      for (int a = 0; a < n_in; a++) act_write(0, a, f64[a]);
      $readmemh(p({jd, "/exp_cyc.hex"}), exp_cyc, 0, nl + 1);
      @(negedge clk); start = 1;
      @(negedge clk); start = 0;
      limit = 2 * int'(exp_cyc[nl]) + 100_000;
      while (busy && limit > 0) begin @(negedge clk); limit--; end
      if (busy) begin
        timeout = 1;
        @(negedge clk); soft_reset = 1;
        @(negedge clk); soft_reset = 0;
        @(negedge clk);
      end
      err_ok = !timeout && done && !error;
      lg_zero = 1;
      if (out_raw) begin
        $readmemh(p({jd, "/logit.hex"}), f32, 0, 15);
        out_ok = 1;
        for (int i = 0; i < 16; i++) begin
          if (i < n_lg && logit[i] !== f32[i]) out_ok = 0;
          if (i >= n_lg && logit[i] !== 32'd0) lg_zero = 0;
        end
      end else begin
        $readmemh(p({jd, "/out_exp.hex"}), oexp, 0, n_out - 1);
        $readmemh(p({jd, "/out_mask.hex"}), omsk, 0, n_out - 1);
        for (int a = 0; a < n_out; a++) begin
          act_read(out_buf, a, got);
          for (int i = 0; i < 8; i++) mk[8*i +: 8] = omsk[a][i] ? 8'hFF : 8'h00;
          if (((got ^ oexp[a]) & mk) != 0) n_bad++;
          sig = (sig * SIG_PRIME) ^ (got & mk);
        end
        out_ok = (n_bad == 0);
        for (int i = 0; i < 16; i++) if (logit[i] !== 32'd0) lg_zero = 0;
      end
      cyc_ok = (stall == 0) && (total_cyc == 64'(exp_cyc[nl])) && (mac_active == 64'(exp_cyc[nl + 1]));
      for (int l = 0; l < nl; l++) if (layer_cyc[l] != exp_cyc[l]) cyc_ok = 0;
      ok = err_ok && out_ok && lg_zero && cyc_ok && (err_flags == 0);
    end
    lc = "";
    for (int l = 0; l < (refuse ? 0 : nl); l++) lc = {lc, (l != 0 ? "," : ""), $sformatf("%0d", layer_cyc[l])};
    lg = "";
    if (!refuse && out_raw)
      for (int i = 0; i < n_lg; i++) lg = {lg, (i != 0 ? "," : ""), $sformatf("%0d", $signed(logit[i]))};
    if (lc == "") lc = "-";                // (xsim: no string ternaries inside $display)
    if (lg == "") lg = "-";
    n_run++;
    if (ok) n_ok++;
    else begin
      n_bad_jobs++;
      if (first_bad < 0) first_bad = job;
      if (n_bad_jobs <= 20)
        $display("MISMATCH: job %0d refuse=%0d done=%0d error=%0d code=%h exp=%h out_ok=%0d n_bad=%0d lg_zero=%0d cycles_ok=%0d total %0d/%0d timeout=%0d flags=%h",
                 job, refuse, done, error, err_code, exp_code, out_ok, n_bad, lg_zero, cyc_ok,
                 total_cyc, refuse ? meta[14] : exp_cyc[nl], timeout, err_flags);
    end
    $display("RESULT kind=shape job=%0d refuse=%0d done=%0d error=%0d err_code=%0d exp_code=%0d err_ok=%0d out_ok=%0d out_sig=%016h n_bad=%0d lg_zero=%0d cycles_ok=%0d rtl_total=%0d model_total=%0d mac=%0d model_mac=%0d stall=%0d flags=%0d timeout=%0d layer_cycles=%s logits=%s",
             job, refuse, done, error, err_code, exp_code, err_ok, out_ok, sig, n_bad, lg_zero, cyc_ok,
             total_cyc, refuse ? meta[14] : exp_cyc[nl], mac_active, refuse ? meta[15] : exp_cyc[nl + 1], stall,
             err_flags, timeout, lc, lg);
    if (err_flags != 0) begin                // sticky: clear so later jobs are judged on their own
      @(negedge clk); soft_reset = 1;
      @(negedge clk); soft_reset = 0;
      @(negedge clk);
    end
  endtask

  // ---------------------------------------------------------------- main
  initial begin
    string jlist, sel;
    int js, je, n_jobs, cnt;
    if (!$value$plusargs("DATA=%s", data)) begin
      $display("TEST FAILED: no +DATA"); $fatal(1, "no DATA");
    end
    $readmemh(p("jobs.hex"), f32, 0, 0);
    n_jobs = f32[0];
    if ($value$plusargs("JOBLIST=%s", jlist)) begin
      sel = "list";
      $readmemh(jlist, jl);                  // count, then job ids (shorter than jl: no warning)
      cnt = jl[0];
      if (cnt < 1 || cnt > 4096) begin $display("TEST FAILED: bad JOBLIST"); $fatal(1, "bad list"); end
      js = 0; je = cnt;
    end else begin
      sel = "range";
      if (!$value$plusargs("JOB_START=%d", js)) js = 0;
      if (!$value$plusargs("JOB_END=%d", je)) je = js + 1;
      if (js < 0 || je > n_jobs || js >= je) begin
        $display("TEST FAILED: bad range start=%0d end=%0d n_jobs=%0d", js, je, n_jobs);
        $fatal(1, "bad range");
      end
    end
    $display("SHAPES_START sel=%s start=%0d end=%0d n_jobs=%0d", sel, js, je, n_jobs);

    ps_idle();
    start = 0; soft_reset = 0; n_layers = 0; desc = '0;
    rst = 1;
    repeat (5) @(negedge clk);
    rst = 0;
    fill_garbage();

    n_ok = 0; n_bad_jobs = 0; first_bad = -1; n_run = 0;
    for (int i = js; i < je; i++) run_one(sel == "list" ? int'(jl[i + 1]) : i);
    $display("SHARD_DONE sel=%s start=%0d end=%0d jobs=%0d ok=%0d first_bad=%0d errors=%0d",
             sel, js, je, n_run, n_ok, first_bad, n_bad_jobs);
    if (n_bad_jobs == 0) $display("TEST PASSED jobs=%0d", n_run);
    else begin
      $display("TEST FAILED errors=%0d jobs=%0d", n_bad_jobs, n_run);
      $fatal(1, "tb_shapes failed");
    end
    $finish;
  end
endmodule
