//==============================================================================
// systolic_array.sv — 8x8 Systolic Array (V1 Baseline)
//
// Structural datapath that instantiates 64 processing elements (rtl/common/pe.sv)
// in an 8x8 grid and coordinates their interconnect, control fan-out, and result
// collection. The V1 dataflow is Output-Stationary: weights broadcast per row and
// reloaded every tap, activations shift left-to-right across columns (one-cycle
// register per hop), and each PE accumulates its own output locally. There is no
// partial-sum cascade, no FSM, and no skew logic — the weight-leads-activation
// skew (SYSTOLIC_ARRAY_SPEC.md §9) and all sequencing (§13, §16) are realized by
// the external controller's schedule, not inside this module.
//
// Authoritative contract: docs/specs/SYSTOLIC_ARRAY_SPEC.md (V1) and
// docs/PROJECT_STATE.md (Decisions 7-8). PE contract: docs/specs/PE_SPEC.md.
//==============================================================================

module systolic_array (
    input  logic               clk,             // synchronous design clock
    input  logic               rst,             // synchronous, active-high
    input  logic signed [7:0]  act_in      [0:7], // per-row activation input
    input  logic signed [7:0]  w_in        [0:7], // per-row weight (current tap)
    input  logic               weight_load,     // array-wide weight load (every tap)
    input  logic               accum_clear,     // array-wide pipeline flush
    input  logic               zero_skip,       // array-wide (tied 0 in V1)
    input  logic               result_req  [0:7], // per-column result capture (one-hot)
    output logic signed [31:0] result_out  [0:7]  // selected column's 8 results
);

    //--------------------------------------------------------------------------
    // Internal interconnect
    //--------------------------------------------------------------------------
    logic signed [7:0]  shift_reg   [0:7][0:6];  // 7 inter-column shift regs per row
    logic signed [7:0]  act_delayed [0:7][0:7];  // activation delivered to PE(r,c)
    logic signed [31:0] pe_result   [0:7][0:7];  // PE(r,c).result_out

    //--------------------------------------------------------------------------
    // Activation shift chain (SYSTOLIC_ARRAY_SPEC.md §6, §7).
    //
    // PE act_out is combinational (= activation_in, PE_SPEC.md §7.8/§8.4), so the
    // one-cycle-per-column shift is implemented with array-level registers, not in
    // the PE. shift_reg[r][k] holds act_in[r] delayed by (k+1) cycles. Seven 8-bit
    // registers per row (56 total), all reset to 0 on rst.
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
    // directly; column c>0 sees the (c)-cycle-delayed value from the shift chain.
    //--------------------------------------------------------------------------
    always_comb begin
        for (int r = 0; r < 8; r++) begin
            act_delayed[r][0] = act_in[r];
            for (int c = 1; c < 8; c++)
                act_delayed[r][c] = shift_reg[r][c-1];
        end
    end

    //--------------------------------------------------------------------------
    // 8x8 PE grid (SYSTOLIC_ARRAY_SPEC.md §4, §6). Instanced row-major as
    // pe[r*8 + c]. Weights broadcast per row; weight_load / accum_clear /
    // zero_skip fan out array-wide; result_request is per-column.
    //--------------------------------------------------------------------------
    genvar r, c;
    generate
        for (r = 0; r < 8; r++) begin : row
            for (c = 0; c < 8; c++) begin : col
                pe pe_inst (
                    .clk           (clk),
                    .rst           (rst),
                    .activation_in (act_delayed[r][c]),
                    .weight_in     (w_in[r]),
                    .weight_load   (weight_load),
                    .zero_skip     (zero_skip),
                    .result_request(result_req[c]),
                    .accum_clear   (accum_clear),
                    .result_out    (pe_result[r][c]),
                    .act_out       ()                // unconnected: shift chain
                                                    // driven from act_in (see above)
                );
            end
        end
    endgenerate

    //--------------------------------------------------------------------------
    // Column-sequential result drain (SYSTOLIC_ARRAY_SPEC.md §11.3).
    //
    // result_req is one-hot: at most one column is selected per cycle. Route the
    // selected column's registered result_out to the array's result_out bus
    // (8 x 32-bit). When no column is selected, result_out is driven to 0. The
    // PEs' result_out is already registered; no additional array-level register
    // is added (§18.1).
    //--------------------------------------------------------------------------
    always_comb begin
        for (int r = 0; r < 8; r++) begin
            result_out[r] = '0;
            for (int c = 0; c < 8; c++) begin
                if (result_req[c])
                    result_out[r] = pe_result[r][c];
            end
        end
    end

endmodule
