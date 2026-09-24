// GOS_UNIT_TB: gos_pkg.sv gos_csr.sv
`timescale 1ns/1ps
// tb_gos_csr — self-checking TB for gos_csr (AXI4-Lite CSR slave; FORMATS.md "CSR register map").
// AXI-Lite BFM (drives at posedge+1, handshake when valid & ready at the next posedge) with
// random AW/W ordering (AW first, W first, simultaneous, random delays), random bready/rready
// back-pressure, random addr[1:0]. Checked against a TB model (desc, n_layers, hi shadows) and
// the randomly driven core-side inputs:
//  1. reset values; VERSION, BUILD_ID (non-default parameter)
//  2. exhaustive read sweep of all 1024 words (mapped: data + OKAY; unmapped: 0 + SLVERR)
//  3. exhaustive write sweep of every unmapped / RO word: SLVERR, no state change
//  4. DESC full write (128 words, random data) + read-back + desc port check; N_LAYERS
//  5. wstrb partial writes on DESC / N_LAYERS
//  6. CTRL start / soft_reset: exactly one-cycle pulses, counted, CTRL reads 0
//  7. busy lock: N_LAYERS/DESC writes while busy -> SLVERR, no effect; CTRL still OKAY
//  8. 64-bit lo/hi consistency (counter changes between lo and hi reads)
//  9. random mix (concurrent read + write, random core inputs, random busy)
// Protocol monitors: bvalid/rvalid held (with stable resp/data) until ready; no B before both AW
// and W handshakes; start/soft_reset pulses never last more than one cycle.
module tb_gos_csr;
  import gos_pkg::*;

  localparam logic [31:0] BID     = 32'h5186_BDC3;
  localparam logic [31:0] VERSION = 32'h474F_5302;
  localparam int          N_RAND  = 20000;

  logic clk = 1'b0;
  always #2.5 clk = ~clk;
  logic rst;

  logic [11:0] awaddr, araddr;
  logic        awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;
  logic [31:0] wdata, rdata;
  logic [3:0]  wstrb;
  logic [1:0]  bresp, rresp;

  logic                   start_pulse, soft_reset_pulse;
  logic [3:0]             n_layers;
  logic [7:0][15:0][31:0] desc;
  logic                   busy, done, error;
  logic [31:0]            err_code;
  logic [63:0]            total_cyc, mac_active, stall;
  logic [7:0]             psbv;
  logic [7:0][31:0]       layer_cyc;
  logic [15:0][31:0]      logit;

  gos_csr #(.BUILD_ID(BID)) dut (
    .clk, .rst,
    .s_axi_awaddr(awaddr), .s_axi_awprot(3'b0), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arprot(3'b0), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
    .start_pulse, .soft_reset_pulse, .n_layers, .desc,
    .busy, .done, .error, .err_code, .total_cyc, .mac_active, .stall,
    .ps_busy_violation(psbv), .layer_cyc, .logit);

  // ---------------------------------------------------------------- bookkeeping
  int checks = 0, errors = 0;
  task automatic chk(input bit ok, input string msg);
    checks++;
    if (!ok) begin
      errors++;
      if (errors <= 20) $display("ERROR_CHECK t=%0t %s", $time, msg);
    end
  endtask

  // ---------------------------------------------------------------- model
  logic [7:0][15:0][31:0] m_desc;
  logic [3:0]             m_nl;
  logic [31:0]            m_sh [3];   // TOTAL / MAC / STALL hi shadows
  int                     exp_start = 0, exp_srst = 0, got_start = 0, got_srst = 0;

  function automatic bit is_desc(input logic [9:0] wi);
    return wi >= 10'h040 && wi <= 10'h0BF;
  endfunction
  function automatic bit is_rw(input logic [9:0] wi);
    return wi == 10'h002 || is_desc(wi);
  endfunction
  function automatic bit is_mapped(input logic [9:0] wi);
    return wi <= 10'h00A || (wi >= 10'h010 && wi <= 10'h017) || (wi >= 10'h020 && wi <= 10'h02F) ||
           wi == 10'h03E || wi == 10'h03F || is_desc(wi);
  endfunction

  // expected read (data, resp); updates the shadow model on lo reads
  task automatic exp_read(input logic [9:0] wi, output logic [31:0] d, output logic [1:0] r);
    d = '0; r = 2'b00;
    if (wi >= 10'h010 && wi <= 10'h017) d = layer_cyc[wi[2:0]];
    else if (wi >= 10'h020 && wi <= 10'h02F) d = logit[wi[3:0]];
    else if (is_desc(wi)) d = m_desc[(wi - 10'h040) >> 4][wi[3:0]];
    else case (wi)
      10'h000: d = '0;
      10'h001: d = {29'b0, error, done, busy};
      10'h002: d = {28'b0, m_nl};
      10'h003: d = err_code;
      10'h004: begin d = total_cyc[31:0];  m_sh[0] = total_cyc[63:32];  end
      10'h005: d = m_sh[0];
      10'h006: begin d = mac_active[31:0]; m_sh[1] = mac_active[63:32]; end
      10'h007: d = m_sh[1];
      10'h008: begin d = stall[31:0];      m_sh[2] = stall[63:32];      end
      10'h009: d = m_sh[2];
      10'h00A: d = {24'b0, psbv};
      10'h03E: d = VERSION;
      10'h03F: d = BID;
      default: r = 2'b10;
    endcase
  endtask

  // expected write effect + resp
  task automatic exp_write(input logic [9:0] wi, input logic [31:0] d, input logic [3:0] s,
                           output logic [1:0] r);
    r = 2'b10;
    if (wi == 10'h000) begin
      r = 2'b00;
      if (s[0] && d[0]) exp_start++;
      if (s[0] && d[1]) exp_srst++;
    end else if (is_rw(wi) && !busy) begin
      r = 2'b00;
      if (wi == 10'h002) begin if (s[0]) m_nl = d[3:0]; end
      else for (int b = 0; b < 4; b++)
        if (s[b]) m_desc[(wi - 10'h040) >> 4][wi[3:0]][8*b +: 8] = d[8*b +: 8];
    end
  endtask

  task automatic chk_ports(input string tag);
    chk(desc === m_desc, $sformatf("%s desc port mismatch", tag));
    chk(n_layers === m_nl, $sformatf("%s n_layers port %0h exp %0h", tag, n_layers, m_nl));
  endtask

  // ---------------------------------------------------------------- core-side inputs
  task automatic rand_core();
    done = $urandom; error = $urandom; err_code = $urandom;
    total_cyc = {$urandom, $urandom}; mac_active = {$urandom, $urandom}; stall = {$urandom, $urandom};
    psbv = $urandom;
    for (int l = 0; l < 8; l++)  layer_cyc[l] = $urandom;
    for (int i = 0; i < 16; i++) logit[i] = $urandom;
  endtask

  // ---------------------------------------------------------------- BFM
  int bp_pct = 50;  // back-pressure probability (ready low) in percent
  int aw_done_cnt = 0, w_done_cnt = 0, b_cnt = 0;

  task automatic aw_send(input logic [11:0] a, input int dly);
    repeat (dly) begin @(posedge clk); #1; end
    awvalid = 1'b1; awaddr = a;
    while (!awready) begin @(posedge clk); #1; end
    @(posedge clk); #1; awvalid = 1'b0; awaddr = 'x; aw_done_cnt++;
  endtask
  task automatic w_send(input logic [31:0] d, input logic [3:0] s, input int dly);
    repeat (dly) begin @(posedge clk); #1; end
    wvalid = 1'b1; wdata = d; wstrb = s;
    while (!wready) begin @(posedge clk); #1; end
    @(posedge clk); #1; wvalid = 1'b0; wdata = 'x; wstrb = 'x; w_done_cnt++;
  endtask
  task automatic b_recv(output logic [1:0] r);
    forever begin
      bready = ($urandom_range(99) >= bp_pct);
      if (bvalid && bready) begin r = bresp; @(posedge clk); #1; bready = 1'b0; b_cnt++; return; end
      @(posedge clk); #1;
    end
  endtask

  // mode: 0 AW first, 1 W first, 2 simultaneous, 3 random delays
  task automatic axi_write(input logic [11:0] a, input logic [31:0] d, input logic [3:0] s,
                           input int mode, output logic [1:0] r);
    int da, dw;
    case (mode)
      0: begin da = 0; dw = $urandom_range(1, 4); end
      1: begin dw = 0; da = $urandom_range(1, 4); end
      2: begin da = 0; dw = 0; end
      default: begin da = $urandom_range(0, 3); dw = $urandom_range(0, 3); end
    endcase
    fork
      aw_send(a, da);
      w_send(d, s, dw);
      b_recv(r);
    join
  endtask

  task automatic axi_read(input logic [11:0] a, output logic [31:0] d, output logic [1:0] r);
    repeat ($urandom_range(0, 2)) begin @(posedge clk); #1; end
    arvalid = 1'b1; araddr = a;
    while (!arready) begin @(posedge clk); #1; end
    @(posedge clk); #1; arvalid = 1'b0; araddr = 'x;
    forever begin
      rready = ($urandom_range(99) >= bp_pct);
      if (rvalid && rready) begin d = rdata; r = rresp; @(posedge clk); #1; rready = 1'b0; return; end
      @(posedge clk); #1;
    end
  endtask

  // checked transactions (core inputs must stay constant from call to return)
  task automatic do_write(input logic [9:0] wi, input logic [31:0] d, input logic [3:0] s,
                          input int mode, input string tag);
    logic [1:0] r, er;
    exp_write(wi, d, s, er);
    axi_write({wi, 2'($urandom)}, d, s, mode, r);
    chk(r === er, $sformatf("%s write wi=%03h bresp %0b exp %0b", tag, wi, r, er));
  endtask
  task automatic do_read(input logic [9:0] wi, input string tag);
    logic [31:0] d, ed; logic [1:0] r, er;
    axi_read({wi, 2'($urandom)}, d, r);
    exp_read(wi, ed, er);
    chk(r === er && d === ed,
        $sformatf("%s read wi=%03h got %08h/%0b exp %08h/%0b", tag, wi, d, r, ed, er));
  endtask

  // ---------------------------------------------------------------- monitors
  logic       sp_q = 0, sr_q = 0;
  logic       bv_q = 0, rv_q = 0;
  logic [1:0] bresp_q; logic [1:0] rresp_q; logic [31:0] rdata_q;
  bit         bhs_q = 0, rhs_q = 0;
  always @(posedge clk) begin
    if (!rst) begin
      if (start_pulse) got_start++;
      if (soft_reset_pulse) got_srst++;
      chk(!(start_pulse && sp_q), "start_pulse longer than one cycle");
      chk(!(soft_reset_pulse && sr_q), "soft_reset_pulse longer than one cycle");
      // B / R stability: once valid, hold valid/resp/data until the handshake
      if (bv_q && !bhs_q) chk(bvalid && bresp === bresp_q, "bvalid/bresp dropped or changed before bready");
      if (rv_q && !rhs_q) chk(rvalid && rresp === rresp_q && rdata === rdata_q,
                              "rvalid/rresp/rdata dropped or changed before rready");
      if (bvalid && !bv_q) chk(aw_done_cnt + (awvalid && awready) > b_cnt &&
                               w_done_cnt + (wvalid && wready) > b_cnt, "bvalid before AW and W handshakes");
    end
    sp_q <= start_pulse; sr_q <= soft_reset_pulse;
    bv_q <= bvalid; bresp_q <= bresp; bhs_q <= bvalid && bready;
    rv_q <= rvalid; rresp_q <= rresp; rdata_q <= rdata; rhs_q <= rvalid && rready;
  end

  // ---------------------------------------------------------------- stimulus
  initial begin
    logic [31:0] d; logic [1:0] r;
    awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0;
    awaddr = 'x; wdata = 'x; wstrb = 'x; araddr = 'x;
    busy = 0; rand_core();
    m_desc = '0; m_nl = '0; m_sh = '{default: '0};
    rst = 1'b1;
    repeat (5) @(posedge clk);
    #1 rst = 1'b0;

    // 1. reset values, VERSION, BUILD_ID
    chk_ports("reset");
    chk(awready === 1'b1 && wready === 1'b1 && arready === 1'b1 && bvalid === 1'b0 && rvalid === 1'b0,
        "ready/valid after reset");
    do_read(10'h03E, "VERSION");
    do_read(10'h03F, "BUILD_ID");
    do_read(10'h005, "TOT_HI reset shadow");

    // 2. exhaustive read sweep (mapped + unmapped)
    for (int wi = 0; wi < 1024; wi++) begin
      if (wi % 64 == 0) rand_core();
      do_read(wi[9:0], "sweep");
    end

    // 3. writes to every unmapped / RO word: SLVERR and no effect
    for (int wi = 0; wi < 1024; wi++)
      if (!is_rw(wi[9:0]) && wi != 0) do_write(wi[9:0], $urandom, 4'hF, $urandom_range(0, 3), "ro/unmapped");
    chk_ports("after ro/unmapped sweep");
    chk(got_start == 0 && got_srst == 0, "spurious pulse during write sweep");

    // 4. DESC full write + read-back, N_LAYERS
    for (int wi = 'h040; wi <= 'h0BF; wi++) do_write(wi[9:0], $urandom, 4'hF, wi % 4, "desc fill");
    chk_ports("desc fill");
    for (int wi = 'h040; wi <= 'h0BF; wi++) do_read(wi[9:0], "desc readback");
    for (int k = 0; k < 16; k++) begin
      do_write(10'h002, $urandom, 4'hF, k % 4, "n_layers"); chk_ports("n_layers");
      do_read(10'h002, "n_layers");
    end

    // 5. wstrb partial writes
    for (int k = 0; k < 400; k++) begin
      logic [9:0] wi;
      wi = ($urandom_range(0, 8) == 0) ? 10'h002 : 10'h040 + $urandom_range(0, 127);
      do_write(wi, $urandom, 4'($urandom), $urandom_range(0, 3), "wstrb");
      chk_ports("wstrb");
      do_read(wi, "wstrb readback");
    end

    // 6. CTRL pulses
    for (int k = 0; k < 64; k++) begin
      do_write(10'h000, $urandom, 4'($urandom), $urandom_range(0, 3), "ctrl");
      repeat (2) @(posedge clk); #1;
      chk(got_start == exp_start && got_srst == exp_srst,
          $sformatf("pulse count start %0d/%0d srst %0d/%0d", got_start, exp_start, got_srst, exp_srst));
      do_read(10'h000, "ctrl reads 0");
    end
    do_write(10'h000, 32'h3, 4'h1, 2, "ctrl both");   // both pulses in the same cycle
    repeat (2) @(posedge clk); #1;
    chk(got_start == exp_start && got_srst == exp_srst, "pulse count after both");

    // 7. busy lock
    busy = 1'b1;
    for (int k = 0; k < 64; k++) begin
      logic [9:0] wi;
      wi = (k % 8 == 0) ? 10'h002 : 10'h040 + $urandom_range(0, 127);
      do_write(wi, $urandom, 4'hF, $urandom_range(0, 3), "busy lock");
      chk_ports("busy lock");
      do_read(wi, "busy lock readback");
    end
    do_write(10'h000, 32'h1, 4'hF, 2, "ctrl while busy");
    repeat (2) @(posedge clk); #1;
    chk(got_start == exp_start, "start pulse while busy");
    do_read(10'h001, "status busy");
    busy = 1'b0;

    // 8. 64-bit lo/hi consistency: counter changes between lo and hi
    for (int k = 0; k < 60; k++) begin
      logic [9:0] lo; logic [63:0] v0;
      lo = 10'h004 + 2 * (k % 3);
      rand_core();
      v0 = (k % 3 == 0) ? total_cyc : (k % 3 == 1) ? mac_active : stall;
      do_read(lo, "lo");
      rand_core();                              // counter advances (hi differs)
      do_read(lo + 1, "hi after lo");           // model returns shadow = v0[63:32]
      d = m_sh[k % 3];
      chk(d === v0[63:32], "shadow model sanity");
      if (k % 5 == 0) do_read(lo + 1, "hi again (still shadow)");
    end

    // 9. random mix: concurrent write + read, random busy / core inputs / back-pressure
    for (int k = 0; k < N_RAND; k++) begin
      logic [9:0] wwi, rwi; logic [31:0] wd, rd, ed; logic [3:0] ws; logic [1:0] br, rr, ebr, err;
      int mode;
      bp_pct = $urandom_range(0, 80);
      if ($urandom_range(0, 15) == 0) busy = $urandom;
      if ($urandom_range(0, 7) == 0) rand_core();
      wwi = ($urandom_range(0, 1)) ? 10'h040 + $urandom_range(0, 127) :
            ($urandom_range(0, 3) == 0) ? 10'($urandom) : 10'($urandom_range(0, 'h03F));
      rwi = ($urandom_range(0, 1)) ? 10'h040 + $urandom_range(0, 127) :
            ($urandom_range(0, 3) == 0) ? 10'($urandom) : 10'($urandom_range(0, 'h03F));
      if (rwi == wwi) rwi = 10'h03F;            // same-word concurrent read/write order is not defined
      wd = $urandom; ws = ($urandom_range(0, 3) == 0) ? 4'($urandom) : 4'hF;
      if (wwi == 10'h000 && $urandom_range(0, 1)) wd[1:0] = 2'b00;
      mode = $urandom_range(0, 3);
      exp_write(wwi, wd, ws, ebr);
      exp_read(rwi, ed, err);
      fork
        axi_write({wwi, 2'($urandom)}, wd, ws, mode, br);
        axi_read({rwi, 2'($urandom)}, rd, rr);
      join
      chk(br === ebr, $sformatf("mix write wi=%03h bresp %0b exp %0b", wwi, br, ebr));
      chk(rr === err && rd === ed, $sformatf("mix read wi=%03h got %08h/%0b exp %08h/%0b",
                                             rwi, rd, rr, ed, err));
      chk_ports("mix");
    end
    repeat (3) @(posedge clk); #1;
    chk(got_start == exp_start && got_srst == exp_srst,
        $sformatf("final pulse count start %0d/%0d srst %0d/%0d", got_start, exp_start, got_srst, exp_srst));

    // final: read everything back once more
    bp_pct = 50; busy = 0; rand_core();
    for (int wi = 0; wi < 1024; wi++) do_read(wi[9:0], "final sweep");

    if (errors == 0) begin
      $display("TEST PASSED checks=%0d (starts=%0d soft_resets=%0d)", checks, got_start, got_srst);
    end else begin
      $display("TEST FAILED errors=%0d checks=%0d", errors, checks);
      $fatal(1, "tb_gos_csr failed");
    end
    $finish;
  end

  initial begin
    #50ms;
    $display("TEST FAILED timeout");
    $fatal(1, "tb_gos_csr timeout");
  end

endmodule
