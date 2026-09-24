`timescale 1ns/1ps
// gos_cfg_check — combinational config checker for ONE 16-word descriptor
// (FORMATS.md "config checker"; Python model: gos_pack.check_descriptor /
// check_descriptor_code). Comparisons and single-bit tests only — no multiply,
// divide or modulo. Derived-field consistency is the host's job (DECISIONS D8-1).
//
// fail[i-1] = 1 iff rule id i fails. Rule ids (priority = ascending id):
//   1 pool_en with odd OH      2 pool_en with odd OW     3 K < 8
//   4 WGT_END > 16383          5 IN_END > 4095           6 OUT_END > 4095
//   7 QP_END > 255
//   8..24 field == 0, in order: IC OC IH IW KH KW OH OW K IN_WPR IN_PLANE
//                              OC_TILES OW_TILES OUT_W OUT_H OUT_WPR OUT_PLANE
//   25 OH > IH                 26 OW > IW                27 out_raw with OC > 16
// (Rule 32, N_LAYERS not in 1..8, is checked by gos_core.)
module gos_cfg_check (
  input  logic [15:0][31:0] d,
  output logic [26:0]       fail
);
  logic [15:0] IC, OC, IH, IW, OH, OW, IN_WPR, IN_PLANE, OC_TILES, OW_TILES;
  logic [15:0] OUT_W, OUT_H, OUT_WPR, OUT_PLANE;
  logic [7:0]  KH, KW;
  logic [31:0] K, WGT_END, IN_END, OUT_END, QP_END;
  logic        pool_en, out_raw;

  always_comb begin
    IC = d[0][15:0];   OC = d[0][31:16];
    IH = d[1][15:0];   IW = d[1][31:16];
    KH = d[2][7:0];    KW = d[2][15:8];
    OH = d[3][15:0];   OW = d[3][31:16];
    pool_en = d[6][1]; out_raw = d[6][2];
    K = d[7];
    IN_WPR = d[8][15:0];     IN_PLANE = d[8][31:16];
    OC_TILES = d[9][15:0];   OW_TILES = d[9][31:16];
    OUT_W = d[10][15:0];     OUT_H = d[10][31:16];
    OUT_WPR = d[11][15:0];   OUT_PLANE = d[11][31:16];
    WGT_END = d[12]; IN_END = d[13]; OUT_END = d[14]; QP_END = d[15];

    fail[0]  = pool_en & OH[0];
    fail[1]  = pool_en & OW[0];
    fail[2]  = (K < 32'd8);
    fail[3]  = (WGT_END > 32'd16383);
    fail[4]  = (IN_END  > 32'd4095);
    fail[5]  = (OUT_END > 32'd4095);
    fail[6]  = (QP_END  > 32'd255);
    fail[7]  = (IC == '0);
    fail[8]  = (OC == '0);
    fail[9]  = (IH == '0);
    fail[10] = (IW == '0);
    fail[11] = (KH == '0);
    fail[12] = (KW == '0);
    fail[13] = (OH == '0);
    fail[14] = (OW == '0);
    fail[15] = (K == '0);
    fail[16] = (IN_WPR == '0);
    fail[17] = (IN_PLANE == '0);
    fail[18] = (OC_TILES == '0);
    fail[19] = (OW_TILES == '0);
    fail[20] = (OUT_W == '0);
    fail[21] = (OUT_H == '0);
    fail[22] = (OUT_WPR == '0);
    fail[23] = (OUT_PLANE == '0);
    fail[24] = (OH > IH);
    fail[25] = (OW > IW);
    fail[26] = out_raw & (OC > 16'd16);
  end
endmodule
