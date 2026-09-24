`timescale 1ns/1ps
// gos_shell_scratch — minimal AXI4-Lite slave for the Step 4 empty shell (gos_shell_top only;
// NOT part of the core). Register map (byte offsets in the 4 KB CSR window, FORMATS.md
// "PL top ports and address map"):
//   0x0E0..0x0EC  SCRATCH0..3  RW, reset 0, byte writes honour WSTRB     (shell-only)
//   0x0F8         VERSION      RO = VERSION parameter (shell: 32'h474F_5300)
//   0x0FC         BUILD_ID     RO = BUILD_ID parameter (8-hex-digit short git commit)
//   any other     reads return 0, writes are ignored; response is always OKAY.
// Address bits [1:0] are ignored (word access). One outstanding read and one outstanding write
// (AW and W are accepted together in the same cycle). Synchronous active-high rst.
module gos_shell_scratch #(
  parameter [31:0] VERSION  = 32'h474F_5300,
  parameter [31:0] BUILD_ID = 32'h0000_0000
) (
  input  wire         clk,
  input  wire         rst,
  input  wire [11:0]  awaddr,
  input  wire         awvalid,
  output wire         awready,
  input  wire [31:0]  wdata,
  input  wire [3:0]   wstrb,
  input  wire         wvalid,
  output wire         wready,
  output wire [1:0]   bresp,
  output reg          bvalid,
  input  wire         bready,
  input  wire [11:0]  araddr,
  input  wire         arvalid,
  output wire         arready,
  output reg  [31:0]  rdata,
  output wire [1:0]   rresp,
  output reg          rvalid,
  input  wire         rready
);

  reg [31:0] scratch [0:3];

  // write: accept AW and W together when no response is pending
  wire wr_go = awvalid && wvalid && !bvalid;
  assign awready = wr_go;
  assign wready  = wr_go;
  assign bresp   = 2'b00;

  wire       wr_hit = (awaddr[11:4] == 8'h0E);   // 0x0E0..0x0EF
  wire [1:0] wr_idx = awaddr[3:2];

  integer i, b;
  always @(posedge clk) begin
    if (rst) begin
      bvalid <= 1'b0;
      for (i = 0; i < 4; i = i + 1) scratch[i] <= 32'd0;
    end else begin
      if (wr_go) begin
        bvalid <= 1'b1;
        if (wr_hit) begin
          for (b = 0; b < 4; b = b + 1)
            if (wstrb[b]) scratch[wr_idx][8*b +: 8] <= wdata[8*b +: 8];
        end
      end else if (bready) begin
        bvalid <= 1'b0;
      end
    end
  end

  // read: one outstanding read
  wire rd_go = arvalid && !rvalid;
  assign arready = rd_go;
  assign rresp   = 2'b00;

  reg [31:0] rd_mux;
  always @(*) begin
    case (araddr[11:2])
      10'h038: rd_mux = scratch[0];   // 0x0E0
      10'h039: rd_mux = scratch[1];   // 0x0E4
      10'h03A: rd_mux = scratch[2];   // 0x0E8
      10'h03B: rd_mux = scratch[3];   // 0x0EC
      10'h03E: rd_mux = VERSION;      // 0x0F8
      10'h03F: rd_mux = BUILD_ID;     // 0x0FC
      default: rd_mux = 32'd0;
    endcase
  end

  always @(posedge clk) begin
    if (rst) begin
      rvalid <= 1'b0;
      rdata  <= 32'd0;
    end else if (rd_go) begin
      rvalid <= 1'b1;
      rdata  <= rd_mux;
    end else if (rready) begin
      rvalid <= 1'b0;
    end
  end

endmodule
