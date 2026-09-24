`timescale 1ns/1ps
// gos_pkg — shared parameters and token convention for the V2 generalized
// output-stationary accelerator (ARCH_SPEC.md, FORMATS.md).
//
// Token convention (v2/rtl/README.md): every pipelined leaf module takes a
// generic token width parameter TW and carries in_tok to out_tok with exactly
// the same fixed latency as its data (localparam L_<MODULE>). gos_tok_t below
// is the integration token; it is PROVISIONAL (DECISIONS D11) and will be
// extended in Step 4 (e.g. output word address, QPARAM channel index).
package gos_pkg;

  // Array / datapath
  localparam int N         = 8;       // array rows (x pixels) = columns (output channels)
  localparam int ACC_W     = 32;      // PE accumulator (INT32)
  localparam int V_MUL_W   = 26;      // signed v width used by the requant multiply (host-proved |v| < 2^25)
  localparam int M_W       = 32;      // requant multiplier m (unsigned, B = 32)
  localparam int S_W       = 6;       // requant shift s (1..63)

  // Memories (FORMATS.md)
  localparam int ACT_DEPTH = 4096;    // words per ACT bank
  localparam int ACT_AW    = 12;
  localparam int WGT_DEPTH = 16384;   // 64-bit words
  localparam int WGT_AW    = 14;
  localparam int QP_CH     = 256;     // QPARAM channels (2 x 64-bit words each)
  localparam int QP_AW     = 8;       // channel index width

  // Accelerator-side memory read latency (fixed; UG901 template with output register)
  localparam int L_MEM_ACC = 2;

  // Integration token (PROVISIONAL, extended in Step 4).
  typedef struct packed {
    logic [7:0]   oc_tile;     // output-channel tile index
    logic [2:0]   col;         // drain column j (output channel oc_tile*8 + j)
    logic         dy;          // pool phase (0: store, 1: combine)
    logic         pool_en;
    logic         relu_en;
    logic         out_raw;     // final layer: raw INT32 v to LOGIT
    logic [N-1:0] row_mask;    // row r valid iff ox0 + r < OW
    logic         ch_valid;    // oc_tile*8 + col < OC
    logic         half;        // pooled bytes go to banks 0-3 (0) or 4-7 (1)
    logic         layer_last;  // last drain step of the layer
  } gos_tok_t;

  localparam int GOS_TOK_W = $bits(gos_tok_t);

endpackage
