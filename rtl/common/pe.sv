//==============================================================================
// pe.sv — Processing Element (V1 Baseline)
//
// Atomic multiply-accumulate unit of the CNN accelerator's 8x8 systolic array.
// V1 dataflow is Weight-Stationary: the weight is held stationary in a
// register while activations shift across rows (act_out = activation_in,
// combinational, 0-cycle). One DSP48E2 is inferred per PE (MREG=1, PREG=1,
// BREG=1, AREG=0).
//
// Authoritative contract: docs/specs/PE_SPEC.md (V1) and
// docs/PROJECT_STATE.md (Decisions 1-6).
//==============================================================================

module pe (
    input  logic               clk,
    input  logic               rst,             // synchronous, active-high
    input  logic signed [7:0]  activation_in,   // signed 8-bit (Q8, scale 2^-8)
    input  logic signed [7:0]  weight_in,       // signed 8-bit (Q8, scale 2^-8)
    input  logic               weight_load,     // load weight_in into weight reg
    input  logic               zero_skip,       // gate MAC accumulation
    input  logic               result_request,  // capture accumulator -> result_out
    input  logic               accum_clear,     // clear accumulator + product
    output logic signed [31:0] result_out,      // registered accumulator snapshot
    output logic signed [7:0]  act_out          // forwarded activation (combinational)
);

    // Internal state
    logic signed [7:0]  weight;       // weight register     (DSP48E2 BREG)
    // use_dsp forces DSP48E2 inference: 8x8 operands fall below Vivado's
    // auto-DSP threshold, so the MAC would otherwise map to fabric LUTs/CARRY8.
    (* use_dsp = "yes" *) logic signed [15:0] product;
                                      // registered product  (DSP48E2 MREG)
    logic signed [31:0] accumulator;  // accumulator         (DSP48E2 PREG)

    //--------------------------------------------------------------------------
    // Activation forwarding: combinational, 0-cycle (PE_SPEC.md §7.8, §8.4).
    // Not gated by any control signal, so it stays cycle-aligned during
    // zero_skip.
    //--------------------------------------------------------------------------
    assign act_out = activation_in;

    //--------------------------------------------------------------------------
    // Weight register (BREG=1). Held stationary; updated only on weight_load.
    // Not gated by zero_skip (PE_SPEC.md §5.5). The new weight is available to
    // the MAC one cycle after weight_load is asserted (nonblocking semantics:
    // the assertion cycle still uses the old weight).
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
    // activation's MAC contributes 0 to the accumulator one cycle later. It
    // leaves the previous cycle's product untouched (that value has already
    // been drained by the accumulator below). Cleared by accum_clear to flush
    // the pipeline (PE_SPEC.md §7.8, §10.14).
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
    // Accumulator register (PREG=1). 32-bit signed (PE_SPEC.md §7.5).
    // Always consumes the registered product so a valid product from the
    // previous cycle is never suppressed when zero_skip is asserted. The
    // skipped activation's 0 (zeroed above) is consumed one cycle later, which
    // is a no-op. accum_clear clears the accumulator (PE_SPEC.md §5.7).
    //--------------------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst)
            accumulator <= '0;
        else if (accum_clear)
            accumulator <= '0;
        else
            accumulator <= accumulator + product;
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
