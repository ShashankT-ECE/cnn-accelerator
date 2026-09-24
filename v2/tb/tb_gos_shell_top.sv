// GOS_UNIT_TB: gos_pkg.sv gos_act_buf.sv gos_wgt_mem.sv gos_qparam_mem.sv gos_shell_scratch.v gos_shell_top.v
// tb_gos_shell_top — empty-shell PL top (Step 4): VERSION/BUILD_ID, scratch registers, all four memories through the BRAM-controller ports.
`timescale 1ns/1ps
module tb_gos_shell_top;
  logic clk = 0; always #2.5 clk = ~clk;
  logic rstn = 0;
  logic [11:0] awaddr, araddr; logic awvalid=0, wvalid=0, bready=1, arvalid=0, rready=1;
  logic [31:0] wdata; logic [3:0] wstrb;
  wire awready, wready, bvalid, arready, rvalid; wire [1:0] bresp, rresp; wire [31:0] rdata;
  logic [14:0] a0_addr=0, a1_addr=0; logic [16:0] w_addr=0; logic [11:0] q_addr=0;
  logic a0_en=0,a1_en=0,w_en=0,q_en=0; logic [7:0] a0_we=0,a1_we=0,w_we=0,q_we=0;
  logic [63:0] a0_din=0,a1_din=0,w_din=0,q_din=0; wire [63:0] a0_dout,a1_dout,w_dout,q_dout;
  int errs = 0, checks = 0;
  gos_shell_top #(.BUILD_ID(32'h1EB0873E)) dut (
    .clk(clk), .rstn(rstn),
    .s_axi_csr_awaddr(awaddr), .s_axi_csr_awvalid(awvalid), .s_axi_csr_awready(awready),
    .s_axi_csr_wdata(wdata), .s_axi_csr_wstrb(wstrb), .s_axi_csr_wvalid(wvalid), .s_axi_csr_wready(wready),
    .s_axi_csr_bresp(bresp), .s_axi_csr_bvalid(bvalid), .s_axi_csr_bready(bready),
    .s_axi_csr_araddr(araddr), .s_axi_csr_arvalid(arvalid), .s_axi_csr_arready(arready),
    .s_axi_csr_rdata(rdata), .s_axi_csr_rresp(rresp), .s_axi_csr_rvalid(rvalid), .s_axi_csr_rready(rready),
    .bram_act0_addr(a0_addr), .bram_act0_clk(clk), .bram_act0_rst(1'b0), .bram_act0_en(a0_en), .bram_act0_we(a0_we), .bram_act0_din(a0_din), .bram_act0_dout(a0_dout),
    .bram_act1_addr(a1_addr), .bram_act1_clk(clk), .bram_act1_rst(1'b0), .bram_act1_en(a1_en), .bram_act1_we(a1_we), .bram_act1_din(a1_din), .bram_act1_dout(a1_dout),
    .bram_wgt_addr(w_addr), .bram_wgt_clk(clk), .bram_wgt_rst(1'b0), .bram_wgt_en(w_en), .bram_wgt_we(w_we), .bram_wgt_din(w_din), .bram_wgt_dout(w_dout),
    .bram_qparam_addr(q_addr), .bram_qparam_clk(clk), .bram_qparam_rst(1'b0), .bram_qparam_en(q_en), .bram_qparam_we(q_we), .bram_qparam_din(q_din), .bram_qparam_dout(q_dout));

  task automatic chk(input string what, input logic [63:0] got, exp);
    checks++; if (got !== exp) begin errs++; $display("MISMATCH %s got %h exp %h", what, got, exp); end
  endtask
  task automatic axw(input [11:0] a, input [31:0] d, input [3:0] s);
    @(negedge clk); awaddr=a; wdata=d; wstrb=s; awvalid=1; wvalid=1;
    do @(posedge clk); while (!(awready && wready));
    @(negedge clk); awvalid=0; wvalid=0;
    while (!bvalid) @(negedge clk);
    chk("bresp", bresp, 0);
  endtask
  task automatic axr(input [11:0] a, output [31:0] d);
    @(negedge clk); araddr=a; arvalid=1;
    do @(posedge clk); while (!arready);
    @(negedge clk); arvalid=0;
    while (!rvalid) @(negedge clk);
    d = rdata; chk("rresp", rresp, 0);
  endtask
  // BRAM port: 1-cycle request, data sampled 1 cycle later (READ_LATENCY 1)
  task automatic bw(input int m, input int word, input [63:0] d, input [7:0] we);
    @(negedge clk);
    case (m) 0: begin a0_addr=word<<3; a0_en=1; a0_we=we; a0_din=d; end
             1: begin a1_addr=word<<3; a1_en=1; a1_we=we; a1_din=d; end
             2: begin w_addr=word<<3;  w_en=1;  w_we=we;  w_din=d;  end
             3: begin q_addr=word<<3;  q_en=1;  q_we=we;  q_din=d;  end endcase
    @(negedge clk); {a0_en,a1_en,w_en,q_en}=0; {a0_we,a1_we,w_we,q_we}=0;
  endtask
  task automatic br(input int m, input int word, output [63:0] d);
    @(negedge clk);
    case (m) 0: begin a0_addr=word<<3; a0_en=1; end 1: begin a1_addr=word<<3; a1_en=1; end
             2: begin w_addr=word<<3; w_en=1; end   3: begin q_addr=word<<3; q_en=1; end endcase
    @(posedge clk); #0.1 {a0_en,a1_en,w_en,q_en}=0;
    @(posedge clk); #0.1;
    case (m) 0: d=a0_dout; 1: d=a1_dout; 2: d=w_dout; 3: d=q_dout; endcase
  endtask

  logic [31:0] r; logic [63:0] q;
  int depth[4] = '{4096, 4096, 16384, 512};
  logic [63:0] pat;
  initial begin
    repeat (5) @(posedge clk); rstn = 1; repeat (3) @(posedge clk);
    axr(12'h0F8, r); chk("VERSION", r, 32'h474F5300);
    axr(12'h0FC, r); chk("BUILD_ID", r, 32'h1EB0873E);
    for (int i = 0; i < 4; i++) begin axr(12'h0E0 + 4*i, r); chk("scratch reset", r, 0); end
    for (int i = 0; i < 4; i++) axw(12'h0E0 + 4*i, 32'hA5A50000 + i, 4'hF);
    axw(12'h0E4, 32'h0000_7700, 4'b0010);      // byte 1 only
    axw(12'h0F8, 32'hFFFFFFFF, 4'hF);          // RO: ignored
    axw(12'h100, 32'hFFFFFFFF, 4'hF);          // unmapped: ignored
    for (int i = 0; i < 4; i++) begin axr(12'h0E0 + 4*i, r); chk("scratch", r, (i==1) ? 32'hA5A57701 : 32'hA5A50000 + i); end
    axr(12'h0F8, r); chk("VERSION after", r, 32'h474F5300);
    axr(12'h100, r); chk("unmapped", r, 0);
    for (int m = 0; m < 4; m++) begin
      for (int w = 0; w < depth[m]; w += (depth[m]/64 + 1)) begin
        pat = {$urandom, $urandom};
        bw(m, w, pat, 8'hFF);
        br(m, w, q); chk($sformatf("mem%0d w%0d full", m, w), q, (m==3 && w[0]) ? (pat & 64'h3F) : pat);
      end
      // last word + two narrow (32-bit) writes as the controller does for 32-bit AXI writes
      bw(m, depth[m]-1, 64'h1111_2222_3333_4444, 8'hFF);
      bw(m, depth[m]-1, 64'hDEAD_BEEF_0000_0000, 8'hF0);
      bw(m, depth[m]-1, 64'h0000_0000_CAFE_F00D, 8'h0F);
      br(m, depth[m]-1, q); chk($sformatf("mem%0d last narrow", m), q, (m==3) ? 64'h0D & 64'h3F : 64'hDEADBEEF_CAFEF00D);
      bw(m, 2, 64'hFFFF_FFFF_FFFF_FFFF, 8'hFF); bw(m, 2, 64'h0, 8'b0100_0001);
      br(m, 2, q); chk($sformatf("mem%0d be", m), q, 64'hFF00_FFFF_FFFF_FF00);
    end
    if (errs == 0) $display("TEST PASSED checks=%0d", checks);
    else begin $display("TEST FAILED errors=%0d checks=%0d", errs, checks); $fatal(1, "tb_gos_shell_top failed"); end
    $finish;
  end
endmodule
