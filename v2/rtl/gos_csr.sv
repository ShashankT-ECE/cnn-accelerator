`timescale 1ns/1ps
// gos_csr — AXI4-Lite slave control/status registers (ARCH_SPEC "CSR"; FORMATS.md
// "CSR register map" is the authoritative offset table; descriptor words FORMATS.md §5).
//
// Clock/reset: project clock clk, synchronous active-high rst (the top wrapper converts
// the AXI aresetn). All registers below are reset by rst; soft_reset does NOT reset the CSR
// (it is only forwarded to the core as soft_reset_pulse).
//
// AXI4-Lite side (32-bit data, 12-bit byte address = 4 KB window)
//   AW/W are independent channels: either may arrive first, or both in the same cycle.
//   Each is accepted into a one-entry holding register (awready = !aw_have & !bvalid,
//   wready = !w_have & !bvalid); once both are held the write is committed in the next cycle
//   and bvalid is raised and held until bready. One outstanding write, one outstanding read.
//   Read: araddr is registered on the AR handshake (arready = !ar_pend & !rvalid); the next
//   cycle the register mux is sampled into rdata and rvalid is raised and held until rready.
//   Addresses are word-aligned: awaddr[1:0]/araddr[1:0] are ignored. awprot/arprot ignored.
//   Responses: OKAY (2'b00) / SLVERR (2'b10).
//
// Register map (byte offsets; word index = addr[11:2])
//   0x000 CTRL        W: bit0 start, bit1 soft_reset (write-1 -> 1-cycle pulse, requires
//                     wstrb[0]); reads 0 (OKAY). Writes are always OKAY (also while busy).
//   0x004 STATUS      RO {29'b0, error, done, busy}
//   0x008 N_LAYERS    RW [3:0] (wstrb[0]); reads {28'b0, n_layers}
//   0x00C ERR_CODE    RO err_code (core format {16'b0, rule_id[7:0], 5'b0, layer[2:0]})
//   0x010/0x014 TOTAL_CYC  lo / hi   RO
//   0x018/0x01C MAC_ACTIVE lo / hi   RO
//   0x020/0x024 STALL      lo / hi   RO
//                     64-bit read rule: a read of lo latches the live hi word of the same
//                     counter into a shadow in the same cycle; a read of hi returns the
//                     shadow (0 after rst). Read lo then hi for a consistent 64-bit value.
//   0x028 PS_BUSY_VIOLATION RO {24'b0, ps_busy_violation[7:0]} (bit0 ACT0, bit1 ACT1)
//   0x040 + 4*l  LAYER_CYC[l]  RO, l < 8
//   0x080 + 4*i  LOGIT[i]      RO, i < 16
//   0x0F8 VERSION     RO 32'h474F_5302 ("GOS", 2)
//   0x0FC BUILD_ID    RO parameter BUILD_ID
//   0x100 + 0x40*l + 4*w  DESC[l][w]  RW, l < 8, w < 16 (0x100..0x2FC), byte strobes honoured
//   Everything else: read -> SLVERR, rdata 0; write -> SLVERR, no effect.
//   Write to an RO register (STATUS..PS_BUSY_VIOLATION, LAYER_CYC, LOGIT, VERSION, BUILD_ID):
//     SLVERR, no effect.
//   Write to N_LAYERS or DESC while busy = 1: SLVERR, no effect (the core reads these
//     registers live during a run; analogous to the ACT busy lock-out, DECISIONS D11-1).
//   A write with wstrb = 0 to a writable register is OKAY and changes nothing.
//
// Core side
//   outputs: start_pulse, soft_reset_pulse (registered, exactly 1 cycle, the cycle after the
//            write commit), n_layers[3:0], desc[l][w] (packed, [7:0][15:0][31:0]).
//   inputs : busy, done, error, err_code, total_cyc, mac_active, stall, ps_busy_violation,
//            layer_cyc[l], logit[i] — sampled only when a read of them is served (no CDC).
//
// Latency: write = AW/W handshakes -> +1 commit cycle -> bvalid; read = AR handshake -> +1 ->
// rvalid. Not a token-pipeline module.
module gos_csr #(
  parameter logic [31:0] BUILD_ID = 32'h0000_0000
) (
  input  logic                    clk,
  input  logic                    rst,

  // AXI4-Lite slave
  input  logic [11:0]             s_axi_awaddr,
  input  logic [2:0]              s_axi_awprot,
  input  logic                    s_axi_awvalid,
  output logic                    s_axi_awready,
  input  logic [31:0]             s_axi_wdata,
  input  logic [3:0]              s_axi_wstrb,
  input  logic                    s_axi_wvalid,
  output logic                    s_axi_wready,
  output logic [1:0]              s_axi_bresp,
  output logic                    s_axi_bvalid,
  input  logic                    s_axi_bready,
  input  logic [11:0]             s_axi_araddr,
  input  logic [2:0]              s_axi_arprot,
  input  logic                    s_axi_arvalid,
  output logic                    s_axi_arready,
  output logic [31:0]             s_axi_rdata,
  output logic [1:0]              s_axi_rresp,
  output logic                    s_axi_rvalid,
  input  logic                    s_axi_rready,

  // Core side: control outputs
  output logic                    start_pulse,
  output logic                    soft_reset_pulse,
  output logic [3:0]              n_layers,
  output logic [7:0][15:0][31:0]  desc,

  // Core side: status inputs
  input  logic                    busy,
  input  logic                    done,
  input  logic                    error,
  input  logic [31:0]             err_code,
  input  logic [63:0]             total_cyc,
  input  logic [63:0]             mac_active,
  input  logic [63:0]             stall,
  input  logic [7:0]              ps_busy_violation,
  input  logic [7:0][31:0]        layer_cyc,
  input  logic [15:0][31:0]       logit
);

  localparam logic [31:0] VERSION = 32'h474F_5302;
  localparam logic [1:0]  RESP_OKAY   = 2'b00;
  localparam logic [1:0]  RESP_SLVERR = 2'b10;

  // Word indices (byte offset >> 2)
  localparam logic [9:0] W_CTRL     = 10'h000;
  localparam logic [9:0] W_STATUS   = 10'h001;
  localparam logic [9:0] W_NLAYERS  = 10'h002;
  localparam logic [9:0] W_ERRCODE  = 10'h003;
  localparam logic [9:0] W_TOT_LO   = 10'h004;
  localparam logic [9:0] W_TOT_HI   = 10'h005;
  localparam logic [9:0] W_MAC_LO   = 10'h006;
  localparam logic [9:0] W_MAC_HI   = 10'h007;
  localparam logic [9:0] W_STL_LO   = 10'h008;
  localparam logic [9:0] W_STL_HI   = 10'h009;
  localparam logic [9:0] W_PSBV     = 10'h00A;
  localparam logic [9:0] W_VERSION  = 10'h03E;
  localparam logic [9:0] W_BUILD_ID = 10'h03F;

  // --------------------------------------------------------------- address decode
  // LAYER_CYC: word 0x10..0x17; LOGIT: word 0x20..0x2F;
  // DESC: word 0x040..0x0BF -> desc index {wi[7], wi[5:0]} = {l[2:0], w[3:0]}.
  function automatic logic is_layer_cyc(input logic [9:0] wi);
    return wi[9:3] == 7'b0000_010;
  endfunction
  function automatic logic is_logit(input logic [9:0] wi);
    return wi[9:4] == 6'b00_0010;
  endfunction
  function automatic logic is_desc(input logic [9:0] wi);
    return (wi[9:8] == 2'b00) && (wi[7] ^ wi[6]);
  endfunction

  // --------------------------------------------------------------- registers
  logic [3:0]              n_layers_q;
  logic [7:0][15:0][31:0]  desc_q;
  logic [31:0]             tot_hi_sh, mac_hi_sh, stl_hi_sh;

  assign n_layers = n_layers_q;
  assign desc     = desc_q;

  // --------------------------------------------------------------- write channel
  logic        aw_have, w_have;
  logic [9:0]  aw_wi;
  logic [31:0] w_data;
  logic [3:0]  w_strb;
  logic        wr_commit;

  assign s_axi_awready = !aw_have && !s_axi_bvalid;
  assign s_axi_wready  = !w_have  && !s_axi_bvalid;
  assign wr_commit     = aw_have && w_have;   // implies !bvalid (ready were low while held)

  logic wr_ctrl, wr_nl, wr_desc, wr_lock, wr_ok;
  always_comb begin
    wr_ctrl = (aw_wi == W_CTRL);
    wr_nl   = (aw_wi == W_NLAYERS);
    wr_desc = is_desc(aw_wi);
    wr_lock = busy && (wr_nl || wr_desc);
    wr_ok   = wr_ctrl || ((wr_nl || wr_desc) && !busy);
  end

  logic [2:0] wr_l;
  logic [3:0] wr_w;
  assign wr_l = {aw_wi[7], aw_wi[5:4]};
  assign wr_w = aw_wi[3:0];

  always_ff @(posedge clk) begin
    if (rst) begin
      aw_have          <= 1'b0;
      w_have           <= 1'b0;
      s_axi_bvalid     <= 1'b0;
      s_axi_bresp      <= RESP_OKAY;
      start_pulse      <= 1'b0;
      soft_reset_pulse <= 1'b0;
      n_layers_q       <= '0;
      desc_q           <= '0;
    end else begin
      start_pulse      <= 1'b0;
      soft_reset_pulse <= 1'b0;

      if (s_axi_awvalid && s_axi_awready) begin
        aw_have <= 1'b1;
        aw_wi   <= s_axi_awaddr[11:2];
      end
      if (s_axi_wvalid && s_axi_wready) begin
        w_have <= 1'b1;
        w_data <= s_axi_wdata;
        w_strb <= s_axi_wstrb;
      end

      if (wr_commit) begin
        aw_have      <= 1'b0;
        w_have       <= 1'b0;
        s_axi_bvalid <= 1'b1;
        s_axi_bresp  <= wr_ok ? RESP_OKAY : RESP_SLVERR;
        if (wr_ctrl && w_strb[0]) begin
          start_pulse      <= w_data[0];
          soft_reset_pulse <= w_data[1];
        end
        if (wr_nl && !wr_lock && w_strb[0])
          n_layers_q <= w_data[3:0];
        if (wr_desc && !wr_lock)
          for (int b = 0; b < 4; b++)
            if (w_strb[b]) desc_q[wr_l][wr_w][8*b +: 8] <= w_data[8*b +: 8];
      end else if (s_axi_bvalid && s_axi_bready) begin
        s_axi_bvalid <= 1'b0;
      end
    end
  end

  // --------------------------------------------------------------- read channel
  logic       ar_pend;
  logic [9:0] ar_wi;

  assign s_axi_arready = !ar_pend && !s_axi_rvalid;

  logic [31:0] rd_mux;
  logic        rd_hit;
  always_comb begin
    rd_mux = '0;
    rd_hit = 1'b1;
    if (is_layer_cyc(ar_wi))      rd_mux = layer_cyc[ar_wi[2:0]];
    else if (is_logit(ar_wi))     rd_mux = logit[ar_wi[3:0]];
    else if (is_desc(ar_wi))      rd_mux = desc_q[{ar_wi[7], ar_wi[5:4]}][ar_wi[3:0]];
    else begin
      unique case (ar_wi)
        W_CTRL:     rd_mux = '0;
        W_STATUS:   rd_mux = {29'b0, error, done, busy};
        W_NLAYERS:  rd_mux = {28'b0, n_layers_q};
        W_ERRCODE:  rd_mux = err_code;
        W_TOT_LO:   rd_mux = total_cyc[31:0];
        W_TOT_HI:   rd_mux = tot_hi_sh;
        W_MAC_LO:   rd_mux = mac_active[31:0];
        W_MAC_HI:   rd_mux = mac_hi_sh;
        W_STL_LO:   rd_mux = stall[31:0];
        W_STL_HI:   rd_mux = stl_hi_sh;
        W_PSBV:     rd_mux = {24'b0, ps_busy_violation};
        W_VERSION:  rd_mux = VERSION;
        W_BUILD_ID: rd_mux = BUILD_ID;
        default:    rd_hit = 1'b0;
      endcase
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      ar_pend      <= 1'b0;
      s_axi_rvalid <= 1'b0;
      s_axi_rresp  <= RESP_OKAY;
      s_axi_rdata  <= '0;
      tot_hi_sh    <= '0;
      mac_hi_sh    <= '0;
      stl_hi_sh    <= '0;
    end else begin
      if (s_axi_arvalid && s_axi_arready) begin
        ar_pend <= 1'b1;
        ar_wi   <= s_axi_araddr[11:2];
      end
      if (ar_pend) begin
        ar_pend      <= 1'b0;
        s_axi_rvalid <= 1'b1;
        s_axi_rdata  <= rd_mux;
        s_axi_rresp  <= rd_hit ? RESP_OKAY : RESP_SLVERR;
        if (ar_wi == W_TOT_LO) tot_hi_sh <= total_cyc[63:32];
        if (ar_wi == W_MAC_LO) mac_hi_sh <= mac_active[63:32];
        if (ar_wi == W_STL_LO) stl_hi_sh <= stall[63:32];
      end else if (s_axi_rvalid && s_axi_rready) begin
        s_axi_rvalid <= 1'b0;
      end
    end
  end

  // awprot/arprot are ignored (documented).
  logic unused_ok;
  assign unused_ok = ^{s_axi_awprot, s_axi_arprot};

endmodule
