// GOS_UNIT_TB: gos_pkg.sv gos_csr.sv gos_cfg_check.sv gos_ctrl.sv gos_act_buf.sv gos_wgt_mem.sv gos_qparam_mem.sv gos_rotator.sv gos_pe.sv gos_array.sv gos_requant.sv gos_pool.sv gos_core.sv gos_top.sv
`timescale 1ns/1ps
// tb_gos_top — AXI-Lite smoke test of gos_top (gos_csr + gos_core), V2 step 4.
// Everything the PS will do on the board, through the CSR register map only
// (FORMATS.md): VERSION/BUILD_ID, DESC/N_LAYERS writes, CTRL.start, STATUS poll,
// LOGIT / TOTAL_CYC / MAC_ACTIVE / STALL / LAYER_CYC reads, the error path
// (N_LAYERS = 0 -> ERR_CODE), and soft_reset. Memories via the PS ports.
// One LeNet-5 image (test[0]) and one CIFAR-10 image: LOGIT bit-exact, cycles == model.
module tb_gos_top;
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
  logic        act_en[2], act_we[2];
  logic [7:0]  act_be[2];
  logic [ACT_AW-1:0] act_addr[2];
  logic [63:0] act_wdata[2], act_rdata[2];
  logic        wgt_en, wgt_we, qp_en, qp_we;
  logic [7:0]  wgt_be, qp_be;
  logic [WGT_AW-1:0] wgt_addr;
  logic [QP_AW:0]    qp_addr;
  logic [63:0] wgt_wdata, wgt_rdata, qp_wdata, qp_rdata;

  gos_top #(.BUILD_ID(BID), .PS_RD_LAT(1)) dut (
    .clk, .rst,
    .s_axi_awaddr(awaddr), .s_axi_awprot(3'b0), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arprot(3'b0), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
    .act0_en(act_en[0]), .act0_we(act_we[0]), .act0_be(act_be[0]), .act0_addr(act_addr[0]),
    .act0_wdata(act_wdata[0]), .act0_rdata(act_rdata[0]),
    .act1_en(act_en[1]), .act1_we(act_we[1]), .act1_be(act_be[1]), .act1_addr(act_addr[1]),
    .act1_wdata(act_wdata[1]), .act1_rdata(act_rdata[1]),
    .wgt_en, .wgt_we, .wgt_be, .wgt_addr, .wgt_wdata, .wgt_rdata,
    .qp_en, .qp_we, .qp_be, .qp_addr, .qp_wdata, .qp_rdata
  );

  string vec;
  longint checks = 0;
  int errors = 0;
  task automatic chk(input bit ok, input string msg);
    checks++;
    if (!ok) begin errors++; if (errors <= 20) $display("MISMATCH: %s", msg); end
  endtask

  // ---- AXI-Lite BFM (single outstanding, AW and W together) ----
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

  // ---- PS memory tasks ----
  logic [63:0] f64 [0:16383];
  logic [31:0] f32 [0:4095];
  task automatic ps_idle();
    for (int b = 0; b < 2; b++) begin act_en[b] = 0; act_we[b] = 0; act_be[b] = 0; act_addr[b] = 0; act_wdata[b] = 0; end
    wgt_en = 0; wgt_we = 0; wgt_be = 0; wgt_addr = 0; wgt_wdata = 0;
    qp_en = 0; qp_we = 0; qp_be = 0; qp_addr = 0; qp_wdata = 0;
  endtask
  task automatic load_mem(input string net, input int img, input int in_end);
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
    $readmemh($sformatf("%s/%s/net/img%0d/act0.hex", vec, net, img), f64);
    for (int a = 0; a <= in_end; a++) begin
      @(negedge clk); act_en[0] = 1; act_we[0] = 1; act_be[0] = 8'hFF; act_addr[0] = a; act_wdata[0] = f64[a];
    end
    @(negedge clk); act_en[0] = 0; act_we[0] = 0;
  endtask

  task automatic run_net(input string net, input int nl, input int img);
    int in_end, lim;
    logic [31:0] exp_logit [0:15];
    logic [31:0] exp_cyc [0:15];
    logic [63:0] tot, mac, stl;
    $readmemh($sformatf("%s/%s/desc.hex", vec, net), f32);
    in_end = f32[13];
    load_mem(net, img, in_end);
    for (int l = 0; l < nl; l++)
      for (int w = 0; w < 16; w++) wr(12'h100 + 12'h40 * l + 4 * w, f32[16 * l + w]);
    wr(12'h008, nl);
    wr(12'h000, 32'h1);                                  // CTRL.start
    lim = 400000;
    do begin axi_rd(12'h004, rv, rr); lim--; end while (rv[0] && lim > 0);
    chk(rv[2:0] == 3'b010, $sformatf("%s STATUS after run = %b", net, rv[2:0]));
    $readmemh($sformatf("%s/%s/net/img%0d/logit16.hex", vec, net, img), exp_logit);
    for (int i = 0; i < 16; i++) begin
      axi_rd(12'h080 + 4 * i, rv, rr);
      chk(rv == exp_logit[i] && rr == 2'b00, $sformatf("%s LOGIT[%0d] %h exp %h", net, i, rv, exp_logit[i]));
    end
    $readmemh($sformatf("%s/%s/net/expect_cyc.hex", vec, net), exp_cyc);
    for (int l = 0; l < nl; l++) begin
      axi_rd(12'h040 + 4 * l, rv, rr);
      chk(rv == exp_cyc[l], $sformatf("%s LAYER_CYC[%0d] %0d exp %0d", net, l, rv, exp_cyc[l]));
    end
    axi_rd(12'h010, tot[31:0], rr);  axi_rd(12'h014, tot[63:32], rr);
    axi_rd(12'h018, mac[31:0], rr);  axi_rd(12'h01C, mac[63:32], rr);
    axi_rd(12'h020, stl[31:0], rr);  axi_rd(12'h024, stl[63:32], rr);
    chk(tot == 64'(exp_cyc[nl]), $sformatf("%s TOTAL_CYC %0d exp %0d", net, tot, exp_cyc[nl]));
    chk(mac == 64'(exp_cyc[nl + 1]), $sformatf("%s MAC_ACTIVE %0d exp %0d", net, mac, exp_cyc[nl + 1]));
    chk(stl == 0, $sformatf("%s STALL %0d", net, stl));
    axi_rd(12'h00C, rv, rr); chk(rv == 0, $sformatf("%s ERR_CODE %h", net, rv));
    axi_rd(12'h028, rv, rr); chk(rv == 0, $sformatf("%s PS_BUSY_VIOLATION %h", net, rv));
    $display("RESULT kind=top net=%s img=%0d total=%0d model_total=%0d", net, img, tot, exp_cyc[nl]);
  endtask

  initial begin
    if (!$value$plusargs("VEC_DIR=%s", vec)) begin $display("TEST FAILED: no VEC_DIR"); $fatal(1, "no VEC_DIR"); end
    awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0; awaddr = 0; araddr = 0; wdata = 0; wstrb = 0;
    ps_idle();
    rst = 1; repeat (5) @(negedge clk); rst = 0;

    axi_rd(12'h0F8, rv, rr); chk(rv == 32'h474F_5302 && rr == 0, $sformatf("VERSION %h", rv));
    axi_rd(12'h0FC, rv, rr); chk(rv == BID, $sformatf("BUILD_ID %h", rv));

    // error path: N_LAYERS = 0 -> STATUS.error, ERR_CODE = rule 32
    wr(12'h008, 0);
    wr(12'h000, 32'h1);
    repeat (6) @(negedge clk);
    axi_rd(12'h004, rv, rr); chk(rv[2:0] == 3'b100, $sformatf("STATUS after bad job = %b", rv[2:0]));
    axi_rd(12'h00C, rv, rr); chk(rv == 32'h0000_2000, $sformatf("ERR_CODE %h exp 00002000", rv));
    wr(12'h000, 32'h2);                                  // soft_reset clears error
    repeat (3) @(negedge clk);
    axi_rd(12'h004, rv, rr); chk(rv[2:0] == 3'b000, $sformatf("STATUS after soft_reset = %b", rv[2:0]));

    run_net("lenet5", 5, 0);
    run_net("cifar10", 4, 0);

    if (errors == 0) $display("TEST PASSED checks=%0d", checks);
    else begin $display("TEST FAILED errors=%0d", errors); $fatal(1, "tb_gos_top failed"); end
    $finish;
  end
endmodule
