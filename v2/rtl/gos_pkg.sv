`timescale 1ns/1ps
// gos_pkg — shared parameters and token convention for the V2 generalized
// output-stationary accelerator (ARCH_SPEC.md, FORMATS.md).
//
// Token convention (v2/rtl/README.md): every pipelined leaf module takes a
// generic token width parameter TW and carries in_tok to out_tok with exactly
// the same fixed latency as its data (localparam L_<MODULE>). gos_tok_t below
// is the integration token (final in Step 4, DECISIONS D12).
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

  // Integration token (final, Step 4; DECISIONS D12). Carried from the issue stage
  // with every k, captured by gos_array at `last` and repeated on the 8 drain cycles.
  // Tile-constant: all fields are fixed for the whole tile.
  typedef struct packed {
    logic [2:0]        layer;      // layer index l (descriptor slot)
    logic [7:0]        oc_tile;    // output-channel tile (channel base = oc_tile*8, a shift)
    logic              dy;         // pool phase (0: store, 1: combine)
    logic              pool_en;
    logic              relu_en;
    logic              out_raw;    // final layer: raw INT32 v of row 0 -> LOGIT[oc]
    logic [N-1:0]      row_mask;   // row r valid iff ox0 + r < OW
    logic [N-1:0]      ch_valid;   // bit j: oc_tile*8 + j < OC
    logic [ACT_AW-1:0] word_base;  // output ACT word of drain column 0 (column j adds j*OUT_PLANE)
    logic              half;       // pooled bytes to banks 0-3 (0) or 4-7 (1)
    logic [QP_AW-1:0]  qp_base;    // QPARAM channel of drain column 0 (= QP_BASE + oc_tile*8)
  } gos_tok_t;

  localparam int GOS_TOK_W = $bits(gos_tok_t);

endpackage
