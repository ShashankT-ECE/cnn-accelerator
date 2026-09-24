// GOS_UNIT_TB: gos_pkg.sv gos_csr.sv gos_cfg_check.sv gos_ctrl.sv gos_act_buf.sv gos_wgt_mem.sv gos_qparam_mem.sv gos_rotator.sv gos_pe.sv gos_array.sv gos_requant.sv gos_pool.sv gos_core.sv gos_top.sv
`timescale 1ns/1ps
// tb_gos_top_backtoback — robustness (V2 step 4.5, Part B1). NJOBS consecutive jobs (default 100)
// through the AXI-Lite CSR only, alternating LeNet-5 / CIFAR-10 (test images cycle 0..9), exactly as
// the board driver will run them: when the net changes, WGT and QPARAM are reloaded through the
// PS memory ports; ACT0 gets the job's input image; DESC[0..N-1] and N_LAYERS are rewritten; no
// soft_reset between jobs. Special jobs:
//   job NJOBS*37/100 : refused by the config checker (layer 1 K = 7) -> STATUS.error, ERR_CODE
//                      = {rule 3, layer 1}; the next job runs normally without any reset.
//   job NJOBS*63/100 : soft_reset (CTRL bit1) while the job is running -> STATUS idle; then the
//                      same job is started again and must complete bit-exact.
// Every completed job: LOGIT[0..15] bit-exact, LAYER_CYC[l], TOTAL_CYC, MAC_ACTIVE == model,
// STALL = 0, ERR_CODE = 0, PS_BUSY_VIOLATION = 0.
// Define NETLIST to instantiate the post-synthesis funcsim netlist (no parameters).
module tb_gos_top_backtoback;
  import gos_pkg::*;
  localparam logic [31:0] BID = 32'h1EB0_8730;

  logic clk = 1'b0;
  always #2.5 clk = ~clk;
  logic rst;

  logic [11:0] awaddr, araddr;
  logic        awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;
  logic [31:0] wdata, rdata;
  logic [3:0]  wstrb;
  logic [1:0]  bresp, rresp;
  logic        act0_en, act0_we;
  logic [7:0]  act0_be;
  logic [ACT_AW-1:0] act0_addr;
  logic [63:0] act0_wdata, act0_rdata, act1_rdata;
  logic        wgt_en, wgt_we, qp_en, qp_we;
  logic [7:0]  wgt_be, qp_be;
  logic [WGT_AW-1:0] wgt_addr;
  logic [QP_AW:0]    qp_addr;
  logic [63:0] wgt_wdata, wgt_rdata, qp_wdata, qp_rdata;

`ifdef NETLIST
  gos_top dut (
`else
  gos_top #(.BUILD_ID(BID), .PS_RD_LAT(1)) dut (
`endif
    .clk, .rst,
    .s_axi_awaddr(awaddr), .s_axi_awprot(3'b0), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arprot(3'b0), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
    .act0_en(act0_en), .act0_we(act0_we), .act0_be(act0_be), .act0_addr(act0_addr),
    .act0_wdata(act0_wdata), .act0_rdata(act0_rdata),
    .act1_en(1'b0), .act1_we(1'b0), .act1_be(8'h0), .act1_addr('0), .act1_wdata(64'h0), .act1_rdata(act1_rdata),
    .wgt_en, .wgt_we, .wgt_be, .wgt_addr, .wgt_wdata, .wgt_rdata,
    .qp_en, .qp_we, .qp_be, .qp_addr, .qp_wdata, .qp_rdata
  );

  string vec;
  int njobs = 100;
  longint checks = 0;
  int errors = 0;
  task automatic chk(input bit ok, input string msg);
    checks++;
    if (!ok) begin errors++; if (errors <= 20) $display("MISMATCH: %s", msg); end
  endtask

  // ---- AXI-Lite BFM ----
  task automatic axi_wr(input logic [11:0] a, input logic [31:0] d, output logic [1:0] r);
    @(negedge clk);
    awaddr = a; awvalid = 1; wdata = d; wstrb = 4'hF; wvalid = 1; bready = 1;
    while (awvalid || wvalid) begin
      @(posedge clk);
      if (awready) awvalid <= 0;
      if (wready)  wvalid  <= 0;
      @(negedge clk);
    end
    while (!bvalid) @(negedge clk);
    r = bresp;
    @(posedge clk); @(negedge clk); bready = 0;
  endtask
  task automatic axi_rd(input logic [11:0] a, output logic [31:0] d, output logic [1:0] r);
    @(negedge clk);
    araddr = a; arvalid = 1; rready = 1;
    while (arvalid) begin
      @(posedge clk);
      if (arready) arvalid <= 0;
      @(negedge clk);
    end
    while (!rvalid) @(negedge clk);
    d = rdata; r = rresp;
    @(posedge clk); @(negedge clk); rready = 0;
  endtask
  task automatic wr(input logic [11:0] a, input logic [31:0] d);
    logic [1:0] r;
    axi_wr(a, d, r);
    chk(r == 2'b00, $sformatf("write %h resp %b", a, r));
  endtask
  logic [31:0] rv; logic [1:0] rr;

  // ---- PS memory loading ----
  logic [63:0] f64 [0:16383];
  logic [31:0] desc_w [0:127];
  logic [31:0] exp_logit [0:15];
  logic [31:0] exp_cyc [0:15];
  task automatic ps_idle();
    act0_en = 0; act0_we = 0; act0_be = 0; act0_addr = 0; act0_wdata = 0;
    wgt_en = 0; wgt_we = 0; wgt_be = 0; wgt_addr = 0; wgt_wdata = 0;
    qp_en = 0; qp_we = 0; qp_be = 0; qp_addr = 0; qp_wdata = 0;
  endtask
  task automatic load_params(input string net);
    for (int i = 0; i < 16384; i++) f64[i] = 'x;
    $readmemh($sformatf("%s/%s/wgt.hex", vec, net), f64);
    for (int a = 0; a < WGT_DEPTH && f64[a] !== 'x; a++) begin
      @(negedge clk); wgt_en = 1; wgt_we = 1; wgt_be = 8'hFF; wgt_addr = a; wgt_wdata = f64[a];
    end
    @(negedge clk); wgt_en = 0; wgt_we = 0;
    for (int i = 0; i < 16384; i++) f64[i] = 'x;
    $readmemh($sformatf("%s/%s/qparam.hex", vec, net), f64);
    for (int a = 0; a < 2 * QP_CH && f64[a] !== 'x; a++) begin
      @(negedge clk); qp_en = 1; qp_we = 1; qp_be = 8'hFF; qp_addr = a; qp_wdata = f64[a];
    end
    @(negedge clk); qp_en = 0; qp_we = 0;
  endtask
  task automatic load_input(input string net, input int img, input int in_end);
    $readmemh($sformatf("%s/%s/net/img%0d/act0.hex", vec, net, img), f64);
    for (int a = 0; a <= in_end; a++) begin
      @(negedge clk); act0_en = 1; act0_we = 1; act0_be = 8'hFF; act0_addr = a; act0_wdata = f64[a];
    end
    @(negedge clk); act0_en = 0; act0_we = 0;
  endtask
  task automatic write_desc(input int nl);
    for (int l = 0; l < nl; l++)
      for (int w = 0; w < 16; w++) wr(12'h100 + 12'h40 * l + 4 * w, desc_w[16 * l + w]);
    wr(12'h008, nl);
  endtask
  task automatic wait_idle(input int lim);
    do begin axi_rd(12'h004, rv, rr); lim--; end while (rv[0] && lim > 0);
  endtask

  task automatic check_job(input string net, input int nl, input int img, input int job);
    logic [63:0] tot, mac, stl;
    bit ok = 1;
    chk(rv[2:0] == 3'b010, $sformatf("job %0d %s STATUS %b", job, net, rv[2:0]));
    $readmemh($sformatf("%s/%s/net/img%0d/logit16.hex", vec, net, img), exp_logit);
    for (int i = 0; i < 16; i++) begin
      axi_rd(12'h080 + 4 * i, rv, rr);
      if (rv !== exp_logit[i]) ok = 0;
      chk(rv === exp_logit[i], $sformatf("job %0d %s img %0d LOGIT[%0d] %h exp %h", job, net, img, i, rv, exp_logit[i]));
    end
    $readmemh($sformatf("%s/%s/net/expect_cyc.hex", vec, net), exp_cyc);
    for (int l = 0; l < nl; l++) begin
      axi_rd(12'h040 + 4 * l, rv, rr);
      chk(rv == exp_cyc[l], $sformatf("job %0d %s LAYER_CYC[%0d] %0d exp %0d", job, net, l, rv, exp_cyc[l]));
    end
    axi_rd(12'h010, tot[31:0], rr);  axi_rd(12'h014, tot[63:32], rr);
    axi_rd(12'h018, mac[31:0], rr);  axi_rd(12'h01C, mac[63:32], rr);
    axi_rd(12'h020, stl[31:0], rr);  axi_rd(12'h024, stl[63:32], rr);
    chk(tot == 64'(exp_cyc[nl]), $sformatf("job %0d %s TOTAL_CYC %0d exp %0d", job, net, tot, exp_cyc[nl]));
    chk(mac == 64'(exp_cyc[nl + 1]), $sformatf("job %0d %s MAC_ACTIVE %0d exp %0d", job, net, mac, exp_cyc[nl + 1]));
    chk(stl == 0, $sformatf("job %0d STALL %0d", job, stl));
    axi_rd(12'h00C, rv, rr); chk(rv == 0, $sformatf("job %0d ERR_CODE %h", job, rv));
    axi_rd(12'h028, rv, rr); chk(rv == 0, $sformatf("job %0d PS_BUSY_VIOLATION %h", job, rv));
    $display("RESULT kind=backtoback job=%0d net=%s img=%0d logits_ok=%0d total=%0d model_total=%0d",
             job, net, img, ok, tot, exp_cyc[nl]);
  endtask

  initial begin
    string cur_net, net;
    int nl, in_end, img, j_refuse, j_reset, completed;
    if (!$value$plusargs("VEC_DIR=%s", vec)) begin $display("TEST FAILED: no VEC_DIR"); $fatal(1, "no VEC_DIR"); end
    void'($value$plusargs("NJOBS=%d", njobs));
    j_refuse = njobs * 37 / 100;
    j_reset  = njobs * 63 / 100;
    awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0; awaddr = 0; araddr = 0; wdata = 0; wstrb = 0;
    ps_idle();
    rst = 1; repeat (5) @(negedge clk); rst = 0;
    axi_rd(12'h0F8, rv, rr); chk(rv == 32'h474F_5302, $sformatf("VERSION %h", rv));
    cur_net = "";
    completed = 0;
    for (int job = 0; job < njobs; job++) begin
      net = (job % 2 == 0) ? "lenet5" : "cifar10";
      nl  = (net == "lenet5") ? 5 : 4;
      img = (job / 2) % 10;
      $readmemh($sformatf("%s/%s/desc.hex", vec, net), desc_w);
      in_end = desc_w[13];
      if (net != cur_net) begin load_params(net); cur_net = net; end
      load_input(net, img, in_end);
      if (job == j_refuse) begin
        desc_w[16 * 1 + 7] = 32'd7;                       // layer 1: K = 7 -> rule 3
        write_desc(nl);
        wr(12'h000, 32'h1);
        wait_idle(1000);
        chk(rv[2:0] == 3'b100, $sformatf("refused job %0d STATUS %b", job, rv[2:0]));
        axi_rd(12'h00C, rv, rr);
        chk(rv == 32'h0000_0301, $sformatf("refused job %0d ERR_CODE %h exp 00000301", job, rv));
        $display("RESULT kind=backtoback job=%0d net=%s refused=1 err_code=%h", job, net, rv);
        continue;                                         // next job, no reset
      end
      write_desc(nl);
      wr(12'h000, 32'h1);                                 // CTRL.start
      if (job == j_reset) begin
        repeat (3000) @(negedge clk);
        axi_rd(12'h004, rv, rr); chk(rv[0] == 1'b1, $sformatf("job %0d not busy before soft_reset", job));
        wr(12'h000, 32'h2);                               // CTRL.soft_reset while running
        repeat (4) @(negedge clk);
        axi_rd(12'h004, rv, rr); chk(rv[2:0] == 3'b000, $sformatf("job %0d STATUS after soft_reset %b", job, rv[2:0]));
        $display("RESULT kind=backtoback job=%0d net=%s soft_reset=1 status=%b", job, net, rv[2:0]);
        load_input(net, img, in_end);                     // input buffer may have been read; reload
        wr(12'h000, 32'h1);                               // same job again
      end
      wait_idle(400000);
      check_job(net, nl, img, job);
      completed++;
    end
    chk(completed == njobs - 1, $sformatf("completed %0d of %0d", completed, njobs - 1));
    if (errors == 0) $display("TEST PASSED checks=%0d jobs=%0d", checks, njobs);
    else begin $display("TEST FAILED errors=%0d", errors); $fatal(1, "tb_gos_top_backtoback failed"); end
    $finish;
  end
endmodule
