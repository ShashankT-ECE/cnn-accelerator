// gos_top_ports.vh — THE port list of the V2 PL top (plain Verilog, `include'd inside the
// module port list of gos_shell_top (Step 4 empty shell) and gos_top_wrapper (Step 6)).
// Contract: v2/docs/FORMATS.md "PL top ports and address map". Do not edit one top without
// the other: the block design (v2/vivado/bd_shell.tcl) is built against exactly these ports.
//
// Usage (the body is guarded by GOS_TOP_PORTS so that Vivado's IP packager, which parses every
// project header on its own when a BD module reference is created, sees an empty file instead of
// a bare port list — without the guard it reports CRITICAL WARNING [HDL 9-1206]):
//   `define GOS_TOP_PORTS
//   module gos_shell_top #(parameter [31:0] BUILD_ID = 32'h0) (
//   `include "gos_top_ports.vh"
//   );
//   `undef GOS_TOP_PORTS
//
// Clocking: everything runs on clk (= PS pl_clk0). The BRAM-side clk/rst pins of the four
// BRAM ports are driven by the AXI BRAM Controllers (their s_axi_aclk / reset) and are present
// only so the BD bram interface is complete; the top uses clk for all logic.
// Reset: rstn is active low (proc_sys_reset peripheral_aresetn). Internal modules use a
// synchronous active-high rst = registered !rstn.
//
// BRAM ports (AXI BRAM Controller, 64-bit data, single port, read latency 1): *_addr is the
// BYTE address emitted by the controller (bits [2:0] always 0 for 64-bit data); word = addr >> 3.
//   ACT0 / ACT1 : 32 KB window -> addr[14:0], word address addr[14:3] (12 bit, 4096 words)
//   WGT         : 128 KB window -> addr[16:0], word address addr[16:3] (14 bit, 16384 words)
//   QPARAM      : 4 KB window  -> addr[11:0], word address addr[11:3] (9 bit, 512 words)
//   *_we[7:0]   : byte write enables (write iff en & |we); *_din = write data (controller -> PL),
//                 *_dout = read data (PL -> controller), valid 1 cycle after en.

`ifdef GOS_TOP_PORTS
  // ------------------------------------------------------------------ clock / reset
  (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
  (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI_CSR, ASSOCIATED_RESET rstn" *)
  input  wire         clk,
  (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rstn RST" *)
  (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
  input  wire         rstn,

  // ------------------------------------------------------------------ AXI4-Lite slave (CSR, 4 KB)
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR AWADDR" *)
  (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 12, READ_WRITE_MODE READ_WRITE" *)
  input  wire [11:0]  s_axi_csr_awaddr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR AWVALID" *)
  input  wire         s_axi_csr_awvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR AWREADY" *)
  output wire         s_axi_csr_awready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR WDATA" *)
  input  wire [31:0]  s_axi_csr_wdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR WSTRB" *)
  input  wire [3:0]   s_axi_csr_wstrb,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR WVALID" *)
  input  wire         s_axi_csr_wvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR WREADY" *)
  output wire         s_axi_csr_wready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR BRESP" *)
  output wire [1:0]   s_axi_csr_bresp,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR BVALID" *)
  output wire         s_axi_csr_bvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR BREADY" *)
  input  wire         s_axi_csr_bready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR ARADDR" *)
  input  wire [11:0]  s_axi_csr_araddr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR ARVALID" *)
  input  wire         s_axi_csr_arvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR ARREADY" *)
  output wire         s_axi_csr_arready,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR RDATA" *)
  output wire [31:0]  s_axi_csr_rdata,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR RRESP" *)
  output wire [1:0]   s_axi_csr_rresp,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR RVALID" *)
  output wire         s_axi_csr_rvalid,
  (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_CSR RREADY" *)
  input  wire         s_axi_csr_rready,

  // ------------------------------------------------------------------ BRAM port ACT0 (32 KB)
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT0 ADDR" *)
  (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_SIZE 32768, MEM_WIDTH 64, MEM_ECC NONE, READ_WRITE_MODE READ_WRITE, READ_LATENCY 1" *)
  input  wire [14:0]  bram_act0_addr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT0 CLK" *)
  input  wire         bram_act0_clk,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT0 RST" *)
  input  wire         bram_act0_rst,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT0 EN" *)
  input  wire         bram_act0_en,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT0 WE" *)
  input  wire [7:0]   bram_act0_we,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT0 DIN" *)
  input  wire [63:0]  bram_act0_din,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT0 DOUT" *)
  output wire [63:0]  bram_act0_dout,

  // ------------------------------------------------------------------ BRAM port ACT1 (32 KB)
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT1 ADDR" *)
  (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_SIZE 32768, MEM_WIDTH 64, MEM_ECC NONE, READ_WRITE_MODE READ_WRITE, READ_LATENCY 1" *)
  input  wire [14:0]  bram_act1_addr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT1 CLK" *)
  input  wire         bram_act1_clk,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT1 RST" *)
  input  wire         bram_act1_rst,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT1 EN" *)
  input  wire         bram_act1_en,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT1 WE" *)
  input  wire [7:0]   bram_act1_we,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT1 DIN" *)
  input  wire [63:0]  bram_act1_din,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_ACT1 DOUT" *)
  output wire [63:0]  bram_act1_dout,

  // ------------------------------------------------------------------ BRAM port WGT (128 KB)
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_WGT ADDR" *)
  (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_SIZE 131072, MEM_WIDTH 64, MEM_ECC NONE, READ_WRITE_MODE READ_WRITE, READ_LATENCY 1" *)
  input  wire [16:0]  bram_wgt_addr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_WGT CLK" *)
  input  wire         bram_wgt_clk,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_WGT RST" *)
  input  wire         bram_wgt_rst,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_WGT EN" *)
  input  wire         bram_wgt_en,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_WGT WE" *)
  input  wire [7:0]   bram_wgt_we,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_WGT DIN" *)
  input  wire [63:0]  bram_wgt_din,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_WGT DOUT" *)
  output wire [63:0]  bram_wgt_dout,

  // ------------------------------------------------------------------ BRAM port QPARAM (4 KB)
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_QPARAM ADDR" *)
  (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_SIZE 4096, MEM_WIDTH 64, MEM_ECC NONE, READ_WRITE_MODE READ_WRITE, READ_LATENCY 1" *)
  input  wire [11:0]  bram_qparam_addr,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_QPARAM CLK" *)
  input  wire         bram_qparam_clk,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_QPARAM RST" *)
  input  wire         bram_qparam_rst,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_QPARAM EN" *)
  input  wire         bram_qparam_en,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_QPARAM WE" *)
  input  wire [7:0]   bram_qparam_we,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_QPARAM DIN" *)
  input  wire [63:0]  bram_qparam_din,
  (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_QPARAM DOUT" *)
  output wire [63:0]  bram_qparam_dout
`endif
