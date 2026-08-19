//==============================================================================
// pe_v2.sv — Processing Element V2 (Reconfigurable Dataflow)
//
// Superset of the V1 PE (rtl/common/pe.sv). Adds the V2 reconfigurable-dataflow
// partial-sum cascade (Decision 10; docs/specs/PE_SPEC.md §13):
//   psum_in       — vertical partial-sum cascade input (WS mode)
//   psum_out      — registered partial sum for vertical forwarding
//   dataflow_mode — 0 = Output-Stationary (OS, V1 behaviour), 1 = Weight-Stationary (WS)
//
// Accumulator behaviour (mode-selected addend, PE_SPEC.md §13.3):
//   OS (dataflow_mode=0): accumulator <= accumulator + product   (V1-identical)
//   WS (dataflow_mode=1): accumulator <= psum_in     + product   (partial-sum cascade)
//
// Everything else (weight/product/result/activation blocks, control priority
// §5.7/§13.4) is V1-identical. The V1 PE (rtl/common/pe.sv) is FROZEN and
// untouched; this module is the V2 PE variant.
//
// DSP48E2 inference (1 DSP/PE, BREG=1, MREG=1, PREG=1) verified under
// build/dsp_probe/ and build/v2_synth/ (Vivado ML 2023.1, xck26-sfvc784-2LV-c).
// In WS mode the psum_in addend is implemented through the DSP48E2 C input
// (OPMODE "C or P"), NOT the PCIN/PCOUT cascade: the dataflow_mode mux prevents
// PCIN/PCOUT inference. This is an implementation detail only — the WS
// accumulator behaviour (accumulator <= psum_in + product) is functionally
// equivalent either way and is verified at array level (335/335,
// sim/tb_systolic_array_v2.sv). PCIN/PCOUT is not a V2 functional requirement.
//
// Authoritative contract: docs/specs/PE_SPEC.md (V1 §5-§8, V2 §13) and
// docs/PROJECT_STATE.md (Decision 10).
//==============================================================================

module pe_v2 (
    input  logic               clk,
    input  logic               rst,             // synchronous, active-high
    input  logic signed [7:0]  activation_in,   // signed 8-bit (Q8, scale 2^-8)
    input  logic signed [7:0]  weight_in,       // signed 8-bit (Q8, scale 2^-8)
    input  logic               weight_load,     // load weight_in into weight reg
    input  logic               zero_skip,       // gate MAC accumulation
    input  logic               result_request,  // capture accumulator -> result_out
    input  logic               accum_clear,     // clear accumulator + product
    input  logic               dataflow_mode,   // 0 = OS (V1), 1 = WS (psum cascade)
    input  logic signed [31:0] psum_in,         // WS partial-sum cascade input
    output logic signed [31:0] result_out,      // registered accumulator snapshot
    output logic signed [7:0]  act_out,         // forwarded activation (combinational)
    output logic signed [31:0] psum_out         // registered partial sum (= accumulator)
);

    // Internal state
    logic signed [7:0]  weight;       // weight register     (DSP48E2 BREG)
    // use_dsp forces DSP48E2 inference: 8x8 operands fall below Vivado's
    // auto-DSP threshold, so the MAC would otherwise map to fabric LUTs/CARRY8.
    (* use_dsp = "yes" *) logic signed [15:0] product;
                                      // registered product  (DSP48E2 MREG)
    logic signed [31:0] accumulator;  // accumulator/psum    (DSP48E2 PREG)

    //--------------------------------------------------------------------------
    // Activation forwarding: combinational, 0-cycle (PE_SPEC.md §7.8, §8.4).
    // Not gated by any control signal, so it stays cycle-aligned during
    // zero_skip.
    //--------------------------------------------------------------------------
    assign act_out = activation_in;

    //--------------------------------------------------------------------------
    // Partial-sum forwarding: expose the registered accumulator for vertical
    // cascade (PE_SPEC.md §13.3). In OS mode this is a passive tap of the local
    // accumulator; in WS mode it carries the cascading partial sum down a column
    // (PE(row n).psum_in = PE(row n-1).psum_out). The accumulator register
    // (PREG) provides the one-cycle-per-row cascade latency.
    //--------------------------------------------------------------------------
    assign psum_out = accumulator;

    //--------------------------------------------------------------------------
    // Weight register (BREG=1). Held stationary; updated only on weight_load.
    // Not gated by zero_skip (PE_SPEC.md §5.5). The new weight is available to
    // the MAC one cycle after weight_load is asserted (nonblocking semantics).
    //--------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst)
            weight <= '0;
        else if (weight_load)
            weight <= weight_in;
    end

    //--------------------------------------------------------------------------
    // Product register (MREG=1). Full-precision signed 16-bit product
    // (PE_SPEC.md §7.4). zero_skip zeroes this register so the *skipped*
    // activation's MAC contributes 0 to the accumulator one cycle later.
    // Cleared by accum_clear to flush the pipeline (PE_SPEC.md §7.8, §10.14).
    //--------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst)
            product <= '0;
        else if (accum_clear)
            product <= '0;
        else if (zero_skip)
            product <= '0;
        else
            product <= activation_in * weight;
    end

    //--------------------------------------------------------------------------
    // Accumulator register (PREG=1). 32-bit signed (PE_SPEC.md §7.5). The
    // addend source is selected by dataflow_mode (PE_SPEC.md §13.3):
    //   OS: local accumulation (V1-identical)
    //   WS: partial-sum cascade (psum_in + product)
    // rst and accum_clear clear the accumulator (and product) in both modes,
    // independent of dataflow_mode (priority unchanged, §13.4). zero_skip
    // zeroes product, so in WS a skipped MAC passes psum_in + 0 through.
    //--------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst)
            accumulator <= '0;
        else if (accum_clear)
            accumulator <= '0;
        else if (dataflow_mode)
            accumulator <= psum_in + product;      // WS: cascade
        else
            accumulator <= accumulator + product;  // OS: hold (V1)
    end

    //--------------------------------------------------------------------------
    // Result output register (fabric). Level-sensitive: while result_request is
    // asserted, result_out re-captures the accumulator every cycle
    // (read-without-clear; the accumulator is untouched). accum_clear has
    // higher priority than result_request, so a simultaneous clear + request
    // captures 0 (PE_SPEC.md §5.7).
    //--------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst)
            result_out <= '0;
        else if (result_request)
            result_out <= accum_clear ? '0 : accumulator;
    end

endmodule
