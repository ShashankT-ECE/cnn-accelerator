//==============================================================================
// systolic_array_v2.sv — 8x8 Systolic Array (V2 Reconfigurable Dataflow, WS)
//
// Structural datapath that instantiates 64 processing elements
// (rtl/common/pe_v2.sv) in an 8x8 grid and coordinates their interconnect,
// control fan-out, and result collection for the V2 Weight-Stationary (WS)
// dataflow (Decision 10/11; docs/specs/SYSTOLIC_ARRAY_V2_SPEC.md §3-§9).
//
// V2 WS mapping (frozen — do not change):
//   - PE rows 0-7     = kernel-tap / reduction dimension (8 taps per tile)
//   - PE columns 0-7  = 8 adjacent output pixels (reverse index: base - c)
//   - output channels serialized in time (one channel per sweep)
//   - 25 taps = 4 tiles (8 + 8 + 8 + 1)
//   - vertical psum reduction: PE(r,c).psum_in = PE(r-1,c).psum_out
//   - bottom-to-top tile feedback: PE(0,c).psum_in = PE(7,c).psum_out
//   - accum_clear once per group (cycle 0); weight_load at cycles 0/15/23/31
//   - final result reaches row 7 at cycle 41; captured at cycle 42
//
// The array is a pure datapath (same boundary as V1's systolic_array.sv):
//   - per-row activation inputs `act_in[0:7]`, shifted left-to-right one column
//     per cycle via array-level registers (identical shift chain to V1).
//   - per-row weight inputs `w_in[0:7]`, broadcast to all 8 columns of a row,
//     loaded via the single array-wide `weight_load` (held per tile).
//   - the per-row DIAGONAL SKEW and per-tap activation CONTENT required by
//     §9.5 are realized by the external input-feed/controller, which drives the
//     8 `act_in` lines with one skewed stream per tap. The array interface is
//     unchanged from V1; only the content differs (§9.5). No FSM / input-feed /
//     BRAM is instantiated here.
//   - result collection is a per-ROW read, the transpose of V1's per-column
//     drain: `result_req[r]` strobes row r's 8 PEs and `result_out[c]` returns
//     that row's column-c result. For WS the controller asserts `result_req[7]`
//     (row 7) at cycle 41 (§8, §9.8, §9.10).
//
// Authoritative contract: docs/specs/SYSTOLIC_ARRAY_V2_SPEC.md (V2 WS),
// docs/specs/PE_SPEC.md §13 (PE-v2), docs/PROJECT_STATE.md Decisions 10-11.
// The V1 modules (pe.sv / systolic_array.sv / input_feed.sv) and pe_v2.sv are
// FROZEN and untouched by this module.
//==============================================================================

module systolic_array_v2 (
    input  logic               clk,             // synchronous design clock
    input  logic               rst,             // synchronous, active-high
    input  logic signed [7:0]  act_in      [0:7], // per-row activation input
    input  logic signed [7:0]  w_in        [0:7], // per-row weight (held per tile)
    input  logic               weight_load,     // array-wide weight load (per tile)
    input  logic               accum_clear,     // array-wide ring flush (per group)
    input  logic               zero_skip,       // array-wide (tied 0 in WS)
    input  logic               dataflow_mode,   // 0 = OS, 1 = WS (tied 1 for WS)
    input  logic               result_req  [0:7], // per-row result capture (row 7)
    output logic signed [31:0] result_out  [0:7]  // requested row's 8 column results
);

    //--------------------------------------------------------------------------
    // Internal interconnect
    //--------------------------------------------------------------------------
    logic signed [7:0]  shift_reg   [0:7][0:6];  // 7 inter-column shift regs per row
    logic signed [7:0]  act_delayed [0:7][0:7];  // activation delivered to PE(r,c)
    logic signed [31:0] pe_result   [0:7][0:7];  // PE(r,c).result_out
    logic signed [31:0] psum        [0:7][0:7];  // PE(r,c).psum_out (= accumulator)
    logic signed [31:0] psum_in     [0:7][0:7];  // PE(r,c).psum_in

    //--------------------------------------------------------------------------
    // Activation shift chain (SYSTOLIC_ARRAY_V2_SPEC.md §7; V1-identical).
    //
    // PE-v2 act_out is combinational (= activation_in), so the one-cycle-per-
    // column shift is implemented with array-level registers, exactly as in V1
    // (systolic_array.sv). shift_reg[r][k] holds act_in[r] delayed by (k+1)
    // cycles; 7 registers per row (56 total), all reset to 0 on rst.
    //--------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        for (int r = 0; r < 8; r++) begin
            if (rst) begin
                for (int k = 0; k < 7; k++)
                    shift_reg[r][k] <= '0;
            end else begin
                shift_reg[r][0] <= act_in[r];
                for (int k = 1; k < 7; k++)
                    shift_reg[r][k] <= shift_reg[r][k-1];
            end
        end
    end

    //--------------------------------------------------------------------------
    // Per-PE activation delivery (combinational): column 0 sees act_in[r]
    // directly; column c>0 sees the c-cycle-delayed value from the shift chain.
    //--------------------------------------------------------------------------
    always_comb begin
        for (int r = 0; r < 8; r++) begin
            act_delayed[r][0] = act_in[r];
            for (int c = 1; c < 8; c++)
                act_delayed[r][c] = shift_reg[r][c-1];
        end
    end

    //--------------------------------------------------------------------------
    // Partial-sum cascade + bottom-to-top tile feedback
    // (SYSTOLIC_ARRAY_V2_SPEC.md §6).
    //
    // Within a tile: PE(r,c).psum_in = PE(r-1,c).psum_out (vertical reduction).
    // Across tiles:  PE(0,c).psum_in = PE(7,c).psum_out (always-on feedback).
    //
    // psum_out is the registered accumulator (pe_v2.sv), so the ring is a closed
    // 8-stage systolic loop (8 accumulator registers around the loop, one
    // psum_in + product adder between each). There is no combinational cycle;
    // the initial 0 comes from accum_clear once per group (cycle 0), never
    // between tiles (§6.2, §9.3-§9.4).
    //--------------------------------------------------------------------------
    genvar pc;
    generate
        for (pc = 0; pc < 8; pc++) begin : psum_link
            assign psum_in[0][pc] = psum[7][pc];          // row 0 <- row 7 feedback
            for (genvar r = 1; r < 8; r++) begin : psum_vert
                assign psum_in[r][pc] = psum[r-1][pc];    // row r <- row r-1
            end
        end
    endgenerate

    //--------------------------------------------------------------------------
    // 8x8 PE-v2 grid (SYSTOLIC_ARRAY_V2_SPEC.md §4). Instanced row-major as
    // pe_v2[r*8 + c]. Weights broadcast per row; weight_load / accum_clear /
    // zero_skip / dataflow_mode fan out array-wide; result_request is per-row
    // (WS reads the bottom row). act_out is unconnected (shift chain driven
    // from act_in above, matching V1).
    //--------------------------------------------------------------------------
    genvar r, c;
    generate
        for (r = 0; r < 8; r++) begin : row
            for (c = 0; c < 8; c++) begin : col
                pe_v2 pe_inst (
                    .clk           (clk),
                    .rst           (rst),
                    .activation_in (act_delayed[r][c]),
                    .weight_in     (w_in[r]),
                    .weight_load   (weight_load),
                    .zero_skip     (zero_skip),
                    .result_request(result_req[r]),
                    .accum_clear   (accum_clear),
                    .dataflow_mode (dataflow_mode),
                    .psum_in       (psum_in[r][c]),
                    .result_out    (pe_result[r][c]),
                    .act_out       (),                 // unconnected: shift chain
                    .psum_out      (psum[r][c])
                );
            end
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Per-row result drain (SYSTOLIC_ARRAY_V2_SPEC.md §8, §9.8) — the transpose
    // of V1's per-column drain.
    //
    // result_req is one-hot over ROWS: result_req[r] strobes all 8 PEs of row r.
    // result_out[c] returns the requested row's column-c result (32-bit). For
    // WS the controller asserts result_req[7] (row 7) at cycle 41 and captures
    // result_out at cycle 42 (read-without-clear). When no row is selected,
    // result_out is driven to 0. The PEs' result_out is already registered; no
    // additional array-level register is added.
    //--------------------------------------------------------------------------
    always_comb begin
        for (int c = 0; c < 8; c++) begin
            result_out[c] = '0;
            for (int r = 0; r < 8; r++) begin
                if (result_req[r])
                    result_out[c] = pe_result[r][c];
            end
        end
    end

endmodule
