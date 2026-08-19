# PE Specification — Processing Element (V1 Baseline)

> **Status:** IMPLEMENTATION-READY — all architectural decisions resolved.
> This document is the implementation contract for `rtl/common/pe.sv`.
> An RTL implementer can write the PE module from this specification without
> making architectural decisions.

---

## 1. Purpose and Scope

The Processing Element (PE) is the atomic computational unit of the CNN
hardware accelerator. Every multiply-accumulate operation in the 8×8 systolic
array is performed by a PE instance.

This document defines the **V1 shared baseline PE**. V1 is the foundation both
teams build upon:

- **Team A** extends the PE with sparsity-skip functionality in V2.
- **Team B** extends the PE with reconfigurable dataflow support in V2.

This document:

- Records what the architecture roadmap **explicitly establishes** about the PE.
- Summarises **research-backed hardware facts** relevant to PE implementation
  on the XCK26 / UltraScale+ platform.
- Presents **candidate** functional models, interfaces, and microarchitectures.
- Lists **open architectural decisions** the team must resolve before RTL.

This document does **not**:

- Define AXI interfaces, register maps, or memory layouts.
- Replace `docs/PROJECT_STATE.md` as the master project decisions record.
- Specify the convolution mapping strategy or array-level schedule.

---

## 2. Source and Specification Status

### 2.1 Documentation Hierarchy

Project documentation is ranked by authority:

| Tier | Source | Authority for PE Design |
|------|--------|------------------------|
| 1 | Explicitly agreed project requirements | Binding |
| 2 | Detailed specifications (this document, once approved) | Binding within scope |
| 3 | External technical research (UG579, DS925, Gemini PE report) | Informative |
| 4 | Engineering recommendations | Advisory |
| 5 | Unresolved assumptions / open decisions | No authority |

### 2.2 Architecture Roadmap

The primary architecture source is:

> **`docs/reference/architecture_major_project.pdf`**
> "CNN Hardware Accelerator on AMD Kria KV260 — Complete Project Reference
> v3.0"

It is a **high-level project roadmap**, not a detailed RTL implementation
specification. Per `docs/PROJECT_STATE.md` §Architecture Status:

> "The architecture/overview PDF is a high-level roadmap and initial reference.
> It is not the authoritative source for detailed implementation decisions."

When this document cites the roadmap, it cites **specific sections** of the
PDF. The roadmap's PE definition in §3.2 is the authoritative baseline for
what the PE must do. How the PE achieves it (pipeline depth, signal naming,
microarchitecture) is left to this specification and to the implementing
engineer, within the constraints established by agreed project requirements.

### 2.3 Gemini PE Research Report

The Gemini Pro research report on the PE is treated as **Tier 3 — external
research input**. Its findings inform Sections 4, 8, and 9 but do not
automatically create project requirements.

### 2.4 Label Convention

Every factual claim carries a status label:

| Label | Meaning |
|-------|---------|
| **REQUIREMENT** | Explicitly established by the architecture roadmap or an agreed project decision |
| **RESEARCH FACT** | Established by external technical reference (DS925, UG579, etc.) |
| **DECISION** | Resolved by team agreement — recorded in `docs/PROJECT_STATE.md` |
| **CANDIDATE** | Proposed design choice — not yet agreed |
| **OPEN DECISION** | Unresolved — team must decide before RTL |

---

## 3. Confirmed PE Requirements from the Architecture Roadmap

### 3.1 Requirements Table

All requirements are sourced to specific sections of
`docs/reference/architecture_major_project.pdf`.

| # | Requirement | Roadmap Source | Scope |
|---|-------------|---------------|-------|
| **R1** | The PE accepts one **activation input**. | §3.2: "Takes one activation…" | V1 |
| **R2** | The PE accepts one **weight input**. | §3.2: "…and one weight." | V1 |
| **R3** | The PE computes **activation × weight**. | §3.2: "Multiplies them." | V1 |
| **R4** | The PE adds the product to an **internal accumulator**. | §3.2: "Adds to internal accumulator." | V1 |
| **R5** | The PE **outputs the accumulated result when the FSM signals**. | §3.2: "Outputs accumulated result when the FSM signals." | V1 |
| **R6** | The PE **output is registered**. | §4 V1 table: `pe.sv` — "Multiply + accumulate + registered output." | V1 |
| **R7** | The PE contains a **zero_skip input port**. When asserted, **multiplication is bypassed** and the **accumulator does not update**. The port is wired only in Team A's build; Team B leaves it unconnected. | §3.2: "Contains a zero_skip input port — when asserted by the Sparsity Manager, the multiply is bypassed and the accumulator does not update. This port is wired only in Team A's build; Team B leaves it unconnected." Also §4 V2 Team A table. | V1 (port), Team A V2 (functional connection) |
| **R8** | The PE is instantiated in an **8×8 systolic grid** (64 PE instances). **Activations shift across rows; weights shift down columns.** | §3.2: "Systolic Array — 8×8 PE Grid. 64 PE instances wired in a grid. Every clock cycle, all 64 PEs execute a MAC simultaneously — 64 MACs per cycle. Activations shift across rows; weights shift down columns." Also §4 V1 table: `systolic_array.sv` — "8×8 PE grid, fixed weight-stationary dataflow." | V1 |
| **R9** | The **V1 baseline dataflow is Weight-Stationary**: weights are held in place in the PE, activations shift across rows. | §3.2: "In Team A's build, the dataflow is fixed weight-stationary." §4 V1 goal: "Fixed weight-stationary dataflow." §4 V1 table: `systolic_array.sv` — "fixed weight-stationary dataflow." | V1 |

### 3.2 What the Roadmap Explicitly Does NOT Establish

The following are **not** established by the roadmap and must not be treated
as V1 requirements:

| Item | Roadmap Evidence |
|------|-----------------|
| Final numerical format | **RESOLVED (2026-08-10).** Q8 signed 8-bit formally adopted as V1 numerical format. See §7.1. |
| Accumulator bit-width | §12 provides **guidance**: "8-bit weight × 8-bit activation = 16-bit product. Summing 64 products needs at least 22 bits minimum." Skill 1 provides the formula: `log2(N_PEs) + weight_bits + activation_bits`. This is **advisory**, not a fixed requirement — it depends on the numerical format. |
| Pipeline depth / number of register stages | Not mentioned in roadmap. |
| Clock frequency target | §4 V1 says "Timing closes at target freq" but does not specify the target. |
| Overflow behaviour (wrap vs. saturate) | Not mentioned in roadmap. |
| Read-and-clear accumulator semantics | §3.2 says "outputs accumulated result when the FSM signals" — does not state that reading clears the accumulator. |
| Specific SystemVerilog signal names | The roadmap describes functional behaviour, not port names. |
| AXI interfaces or register maps | Outside PE scope. Defined in `axi_stream_if.sv` and `axi_lite_ctrl.sv` per §3.2, §4. |
| Output Stationary for V1 | OS is a **Team B V2 extension**. §3.2: "Mode 1 — Output Stationary: partial sums held in PEs, both activations and weights shift." §4 V2 Team B. V1 is Weight-Stationary only. |
| Sparsity Manager for V1 | Sparsity is a **Team A V2 extension**. §4 V1 goal: "No sparsity." The zero_skip **port** exists in V1 but is functionally connected only in Team A's V2 build. |

---

## 4. Research-Backed Hardware Facts

This section records **external technical facts** relevant to PE
implementation on the target platform (XCK26 / Zynq UltraScale+ MPSoC).
These are informative, not prescriptive.

### 4.1 Target DSP Slice

**RESEARCH FACT** — The XCK26 contains **DSP48E2** slices.
Sources: DS925 (Zynq UltraScale+ MPSoC Data Sheet), UG579 (UltraScale
Architecture DSP48E2 Slice User Guide).

### 4.2 DSP48E2 Multiplier Capability

**RESEARCH FACT** — Each DSP48E2 contains a **27 × 18 two's-complement
multiplier**. It supports:

- One `27 × 18` signed multiplication per cycle.
- Dynamically configured signed/unsigned operand selection.
- 8-bit × 8-bit multiplication (the roadmap's working assumption) fits
  comfortably within a single DSP48E2 without decomposition.

### 4.3 DSP48E2 Accumulator

**RESEARCH FACT** — The DSP48E2 includes a **48-bit accumulator** with
dedicated cascade routing between adjacent DSP slices. The accumulator can
add/subtract the multiplier product on every cycle.

### 4.4 DSP48E2 Pipeline Registers

**RESEARCH FACT** — The DSP48E2 offers optional pipeline registers:

| Stage   | Registers | Description |
|---------|-----------|-------------|
| M       | 0–2       | Multiplier input registers (A, B, D inputs) |
| M-pipe  | 0–1       | Internal multiplier pipeline register |
| C       | 0–1       | Accumulator input register |
| P       | 0–1       | Output register |

These registers are inferred by Vivado from behavioural RTL pipeline stages
or can be explicitly instantiated. A 3-stage configuration (M → M-pipe → P)
is a common high-frequency choice but is **not the only valid configuration**.

### 4.5 Behavioural Inference to DSP48E2

**RESEARCH FACT** — Vivado ML 2023.1 infers DSP48E2 usage from behavioural
SystemVerilog `*` (multiply) and `+` (add/accumulate) operators when operand
widths and coding style permit. The synthesis attribute
`(* use_dsp = "yes" *)` guides inference.

**RESEARCH FACT (verified 2026-08-17, Vivado ML 2023.1)** — For 8×8 signed
operands, default synthesis (`use_dsp = "auto"`) does **not** infer a DSP48E2:
the multiplier is small enough that Vivado implements it in fabric LUTs/CARRY8.
The explicit `(* use_dsp = "yes" *)` directive is therefore **required** to
obtain the intended DSP48E2 mapping for this project's PE (§8.1, `PROJECT_STATE.md`).

### 4.6 Synchronous Reset on DSP48E2

**RESEARCH FACT** — DSP48E2 registers support **synchronous reset only**
(`RSTALLCARRYIN`, `RSTALUMODE`, `RSTCTRL`, `RSTM`, `RSTP`, etc., per
UG579). There are no asynchronous reset inputs on DSP48E2 primitives.

**DECISION** — Per UG949 and DSP48E2 constraints, the V1 PE uses
**synchronous reset** for all sequential logic (Decision 5).

If the PE uses DSP48E2-inferred registers, reset on those paths must be
synchronous for direct DSP mapping. Synthesis tools can map asynchronous-reset
RTL to DSPs by adding fabric-based reset synchronisation, at potential cost in
additional LUTs/FFs (UG949 example: async reset on 16×16 multiplier costs
~65 extra FFs + ~32 extra LUTs vs synchronous reset).

### 4.7 Signed Arithmetic and Width Handling

**RESEARCH FACT** — For two's-complement operands of widths `W_a` and `W_w`:

- Product width: `W_product = W_a + W_w` (full precision).
- Accumulator width to avoid overflow across `N_acc` terms:
  `W_acc ≥ W_product + ceil(log2(N_acc))`.

For the roadmap's working assumption of 8-bit signed operands, using the
roadmap's §12 example of 64 accumulated terms:

- `W_product = 8 + 8 = 16`
- `W_acc ≥ 16 + ceil(log2(64)) = 16 + 6 = 22`

This matches the roadmap's §12 guidance of "at least 22 bits minimum."

The DSP48E2's 48-bit accumulator provides ample headroom for this working
assumption.

### 4.8 Systolic Data Movement

**RESEARCH FACT** — In a systolic array, each PE forwards operands to
downstream neighbours every cycle. The forwarding direction (horizontal,
vertical, diagonal) depends on the dataflow. In Weight-Stationary dataflow
(the V1 baseline per R9):

- **Weights** are preloaded and held stationary in each PE.
- **Activations** flow horizontally across rows.
- **Partial sums** (if any) flow vertically down columns.

In the V1 baseline, each PE accumulates partial products locally; partial-sum
forwarding between PEs is not required in pure WS mode for a 2D convolution
mapped to a 1D or 2D systolic array. Whether partial sums are forwarded
depends on the convolution mapping strategy, which is outside PE scope.

---

## 5. PE Functional Model — V1 Baseline

### 5.1 Normal Operation (Weight-Stationary)

**REQUIREMENT** (R1–R4, R9) — In the V1 Weight-Stationary dataflow:

1. A **weight** is loaded into the PE and **held stationary**.
2. Each cycle, a new **activation** arrives from the left neighbour
   (or from the input buffer for the leftmost column).
3. The PE computes: `product = activation × weight`
4. The PE accumulates: `accum = accum + product`
5. The activation is **forwarded** to the right neighbour (R8).

This repeats every cycle. All 64 PEs operate simultaneously — 64 MACs per
cycle (R8).

**DECISION** — Pipeline architecture (Decision 6): registered MAC with
combinational activation forwarding. The multiply-accumulate uses a 2-cycle
pipeline (DSP48E2 MREG + PREG): product computed in cycle N, accumulated
in cycle N+1. Activation forwarding is combinational (`act_out = act_in`,
0-cycle delay) — no pipeline skew between columns. Steady-state throughput:
1 MAC/cycle/PE after 2-cycle pipeline fill.

```
          weight_in
              │
              ▼
         [weight reg] (8b)
              │
activation_in ──► [ × ] ──► [MREG] ──► [ + ] ──► accum (32b)
  │                          (pipeline)      ▲
  ▼                                          │
act_out (= act_in, combinational)    product (registered)
```

### 5.2 Result Exposure

**REQUIREMENT** (R5, R6) — The accumulated result is:

- Exposed on the PE's output port **when the top-level FSM signals**.
- **Registered** at the output (R6).

**DECISION** — The FSM↔PE protocol is: **result_request strobe with
read-without-clear, separate accum_clear** (2026-08-10, recorded in
`docs/PROJECT_STATE.md`).

**PE-level mechanism:**
- `result_request` (input, 1 bit): asserted by FSM during OUTPUT state
- `result_out` (output, 32 bits): registered, captures accumulator on
  `result_request` assertion
- `accum_clear` (input, 1 bit): separate signal to reset accumulator to zero
- The accumulator is NOT cleared by reading (read-without-clear)

**Behavior:**
```systemverilog
// Conceptual — not final RTL. Implements Decisions 4, 5, and 6.
always_ff @(posedge clk) begin
    if (rst) begin
        accumulator <= '0;
        weight      <= '0;
        result_out  <= '0;
        product     <= '0;
    end else begin
        if (weight_load)     weight      <= weight_in;
        if (accum_clear) begin
            accumulator <= '0;
            product     <= '0;   // flush pipeline to prevent stale add-back
        end
        if (result_request)  result_out  <= accumulator;
        if (!zero_skip && !accum_clear) begin
            product     <= activation_in * weight;
            accumulator <= accumulator + product;
        end
    end
end
```

Note: `product` is shown here as a single register for clarity. In the
recommended multi-stage RTL (§7.8), product and accumulator updates are in
separate `always_ff` blocks to infer MREG=1, PREG=1 in DSP48E2.

**Rationale:**
- Read-without-clear is universal across surveyed accelerators (TPU, Eyeriss,
  ShiDianNao, SAURIA, VRM21) — no production accelerator uses read-and-clear
- Separate `accum_clear` follows standard practice; preserves debugging
  capability (result can be re-read)
- No flow control (valid/ready) in the PE — FSM controls timing globally
- One-cycle response: FSM asserts request, result appears next cycle
- Array-level result collection (64 PEs → AXI-Stream) is outside PE scope;
  recommended column-sequential drain (8 PEs/cycle, 8 cycles to drain array)
  is implemented in `systolic_array.sv`, not in the PE

**The roadmap does not state that reading the result clears the accumulator.**
This decision adopts read-without-clear, consistent with the roadmap.

### 5.3 zero_skip Operation

**REQUIREMENT** (R7) — The PE has a `zero_skip` input port. When asserted:

```
product  = (bypassed — not accumulated)
accum    = accum                         // unchanged
```

Multiplication may or may not occur internally; the **functional requirement**
is that the accumulator does not update.

When `zero_skip` is deasserted (or when the port is unconnected, as in Team
B's build), normal accumulation resumes.

**DECISION** (§11.9) — The PE is agnostic to the trigger condition. The
roadmap states the signal is "asserted by the Sparsity Manager" (§3.2),
which implies external control. The Sparsity Manager's detection logic
(Team A V2, `sparsity_manager.sv`) determines the condition. The PE responds
to the signal — it does not generate or interpret it. The V1 PE port exists
and is functional; in Team B's build it is tied LOW.

### 5.4 Activation Forwarding

**REQUIREMENT** (R8, R9) — In V1 Weight-Stationary, the activation input is
forwarded to `act_out` for the next PE in the row. The forwarding path may be
combinational (passthrough) or registered, depending on the pipeline depth
chosen (§11.5).

**IMPLIED** — The forwarding path is expected to remain cycle-aligned during
zero_skip so that a skipped MAC does not disrupt downstream systolic data
movement. This is an architectural implication to be verified during
array-level scheduling and is not stated explicitly by the roadmap.

### 5.5 Weight Handling

**REQUIREMENT** (R9) — In V1 Weight-Stationary:

- Weights are **held in place** within the PE. The PE must retain its weight
  value across cycles until a new weight is explicitly loaded.
- Weight loading is controlled via the `weight_load` input port (1 bit).

**DECISION** — `weight_load` adopted as a V1 PE port. When asserted:
- `weight_in` is captured into the weight register on the next posedge clk.
- The new weight is available for MAC starting one cycle after assertion.
- During the assertion cycle, the MAC uses the OLD weight value.
- Weight register retains its stored value when `weight_load` is deasserted.
- Weight loading is NOT gated by `zero_skip` — weights can be loaded during
  zero_skip cycles.
- Priority: below `rst`, above all other controls (see §5.7).

### 5.6 V2 Extensibility — Team B Output Stationary

**Not a V1 requirement.** For Team B's V2 Output Stationary mode (§3.2):

- Both activations and weights shift every cycle.
- Partial sums are held in PEs.
- The Dataflow Controller generates `shift_enable`, `load_enable`, and
  `accumulate_enable` signals per PE per cycle.

**DECISION** — The V1 PE does NOT expose V2 extensibility ports
(`wgt_out`, `shift_enable`, `accumulate_enable`). Per the roadmap's V2
description, `systolic_array.sv` (not `pe.sv`) is updated to wire controller
outputs. These ports will be added to the PE in V2 if needed. Keeping the
V1 PE interface minimal per §11.10 and §11.11.

### 5.7 Control Signal Priority

**DECISION** — When multiple control signals are asserted simultaneously, the
PE resolves them in this priority order (highest to lowest):

| Priority | Signal | Behavior |
|----------|--------|----------|
| 1 (highest) | `rst` | All registers → 0. All other controls ignored. |
| 2 | `weight_load` | Captures weight_in into weight register. |
| 3 | `accum_clear` | Resets accumulator and product register to 0. |
| 4 | `result_request` | Captures accumulator into result_out. Level-sensitive: updates result_out every cycle while asserted. |
| 5 | `zero_skip` | Gates accumulator update. Product register still updates. Activation forwarding unaffected. |
| 6 (lowest) | Normal MAC | Default when no other control is asserted. |

**Simultaneous assertion resolution:**

| Combination | Behavior |
|------------|----------|
| `rst` + any | Reset wins. All registers → 0. |
| `weight_load` + `accum_clear` | Both take effect (non-conflicting: different registers). |
| `weight_load` + `zero_skip` | Weight updates. MAC skipped (accumulator unchanged). |
| `weight_load` + `result_request` | Weight updates. result_out captures current accumulator. |
| `accum_clear` + `zero_skip` | accum_clear wins. Accumulator and product register → 0. |
| `accum_clear` + `result_request` | accum_clear takes effect first. result_out captures accumulator = 0. |
| `result_request` + `zero_skip` | Both take effect. result_out captures accumulator (unchanged by zero_skip). |
| `result_request` + MAC | Both take effect. result_out captures accumulator BEFORE this cycle's MAC update (registered pipeline timing). |

---

## 6. V1 PE Interface

The following table defines the complete V1 PE port list. All architectural
decisions affecting the interface are resolved. This is the implementation
contract for the `pe.sv` module port declaration.

| Signal Concept      | Direction | Width     | Status        |
|---------------------|-----------|-----------|---------------|
| `clk`               | input     | 1         | **IMPLIED** — synchronous design throughout the project |
| `rst`               | input     | 1         | **DECISION** (Decision 5) — synchronous, active-high reset. Resets accumulator, weight, result_out. |
| `activation_in`     | input     | 8         | **REQUIREMENT** (R1) — width from Decision 1 |
| `weight_in`         | input     | 8         | **REQUIREMENT** (R2) — width from Decision 1 |
| `weight_load`       | input     | 1         | **DECISION** — load-enable for weight register in WS mode. Weight is available for MAC 1 cycle after assertion. |
| `zero_skip`         | input     | 1         | **REQUIREMENT** (R7) — port exists in V1; connected in Team A, unconnected in Team B |
| `result_request`    | input     | 1         | **DECISION** (Decision 4) — FSM strobe to expose accumulated result (R5). Level-sensitive: updates result_out every cycle while asserted. |
| `result_out`        | output    | 32        | **REQUIREMENT** (R5, R6) — registered output; width from Decision 2 |
| `accum_clear`       | input     | 1         | **DECISION** (Decision 4) — resets accumulator AND product register to zero (pipeline flush). Read-without-clear semantics. |
| `act_out`           | output    | 8         | **REQUIREMENT** (R8) — activation forwarding across rows; combinational (= activation_in). Width matches activation_in. |
| `wgt_out`           | output    | OPEN      | **NOT IN V1** — V2 Team B concern (§11.10). Omitted from V1 PE. |
| `shift_enable`      | input     | 1         | **NOT IN V1** — V2 Team B concern (§11.11). Omitted from V1 PE. |
| `accumulate_enable` | input     | 1         | **NOT IN V1** — V2 Team B concern (§11.11). Omitted from V1 PE. |

### 6.1 Notes on Candidate Signals

- **`clk` / `rst`:** Synchronous active-high reset per Decision 5. All
  sequential logic uses `always_ff @(posedge clk)`. Reset initializes
  accumulator, weight register, and result_out to zero.

- **Data widths:** `activation_in`, `weight_in`: 8 bits (Decision 1).
  `result_out`: 32 bits (Decision 2). `act_out`: 8 bits (matches activation_in).
  `wgt_out`: width = 8 bits if/once adopted.

- **`weight_load`:** In WS mode, weights are held stationary but must be
  loaded initially and may be reloaded between layers. A load-enable signal
  is the natural mechanism. **DECISION** — adopted as a V1 PE port. When
  asserted, `weight_in` is captured into the weight register on the next
  clock edge. The new weight is available for MAC one cycle after assertion.
  Weight register retains its value when weight_load=0. Weight loading is
  not gated by zero_skip.

- **`result_request` / `accum_clear`:** The FSM↔PE protocol is resolved
  (Decision 4, §5.2). `result_request` strobes the registered `result_out`.
  `accum_clear` separately resets the accumulator AND product register to
  zero (pipeline flush). No `result_valid` signal in the V1 PE — FSM
  controls timing globally.

- **Control priority (highest to lowest):** rst > weight_load > accum_clear >
  result_request > zero_skip > normal MAC. See §5.7 for full priority table.

- **`shift_enable` / `accumulate_enable` / `wgt_out`:** These are V2
  Team B extensibility hooks. Explicitly excluded from V1 PE. Per the
  roadmap, `systolic_array.sv` (not `pe.sv`) is updated for V2. Added
  to PE in V2 if needed (§11.10, §11.11, §5.6).

---

## 7. Candidate Arithmetic Architecture

### 7.1 Numerical Format — V1 Decision

**DECISION** — The V1 numerical format is formally adopted (2026-08-10,
recorded in `docs/PROJECT_STATE.md`):

| Parameter | Value |
|-----------|-------|
| **Container** | Signed 8-bit two's-complement integer |
| **Scale factor (implicit)** | 1/256 = 2^(-8) |
| **Integer range** | [-128, 127] |
| **Real range** | [-0.5, +0.49609375] |
| **Resolution** | 0.00390625 (1/256) |
| **Weight extraction** | `w_fixed = clip(round(w × 256), -128, 127)` (roadmap §5.3) |
| **Activation encoding** | Same format: signed 8-bit, scale 1/256 |
| **Product** | 16-bit signed, full precision (scale 2^(-16)) |
| **Fixed-point interpretation** | External to PE; handled at array output / drain path |

The roadmap uses the term **"Q8"** for this format, meaning 8 fractional bits
in the short-form Qn convention. The format is not expressible in standard TI
or ARM Qm.n notation (an 8-bit signed container with 8 fractional bits
requires negative `m`).

**The PE treats all operands as raw signed 8-bit integers.** The PE does not
interpret binary-point position, apply scaling, or perform saturation. The
scale factor (1/256) is applied externally — in the weight extraction script,
the golden model, and the array drain path. This keeps the PE datapath
minimal: integer multiply-accumulate only.

**Relationship to INT8:** The roadmap distinguishes Q8 (fixed power-of-two
scale, this decision) from "INT8 quantization" (per-layer calibrated scale
with optional zero-point), which is listed as future work (§10). The PE
datapath is identical for both; only the weight extraction script and
drain-path rescaling differ.

**Known risk:** Weights with magnitude ≥ 0.5 are clipped by the extraction
script. The team should measure actual LeNet-5 weight distributions to assess
whether this clipping introduces meaningful accuracy loss. If it does,
per-layer scale calibration can be adopted without changing the PE.

**Rationale for adoption:**
- Consistent with roadmap §5.3 weight extraction script
- Minimal PE datapath complexity (integer MAC only)
- Optimal DSP48E2 mapping (one DSP/PE with 27×18 multiplier headroom)
- Efficient BRAM usage (8 bits/weight)
- LeNet-5/MNIST accuracy loss is negligible with 8-bit quantization
- Preserves Q8→INT8 future-work pathway (roadmap §10)

### 7.2 Signedness

**DECISION** — Both operands are **signed (two's-complement)** per the
adopted V1 numerical format (§7.1). The weight extraction script clips to
`[-128, 127]` (roadmap §5.3).

The DSP48E2 supports per-operand signed/unsigned selection via INMODE[4:3].
For V1, both operands are signed (INMODE[4]=0, INMODE[3]=0).

### 7.3 Operand Width

**DECISION** — **8-bit activation, 8-bit weight** per the adopted V1
numerical format (§7.1).

Both fit within a single DSP48E2 multiplier (27×18) with substantial headroom.

### 7.4 Product Width

**DECISION** — **16-bit signed, full precision** (`W_a + W_w = 8 + 8`).

For two's-complement signed multiplication, this preserves full precision.
The PE retains all 16 product bits; any truncation or rounding is applied
at the drain path, not inside the PE.

### 7.5 Accumulator Width

**DECISION** — The V1 PE uses a **32-bit signed two's-complement accumulator**
(2026-08-10, recorded in `docs/PROJECT_STATE.md`).

**Derivation:**
- Operand width: 8 bits (Decision 1)
- Product width: 16 bits, full precision (Decision 1)
- Per-PE accumulation depth for LeNet-5 conv1: N_max_acc = 25 (one K×K×IC
  kernel dot product per output pixel)
- Minimum accumulator width: W_acc ≥ 16 + ceil(log2(25)) = 21 bits
- Roadmap example (N=64): W_acc ≥ 22 bits
- LeNet-5 worst case (FC1, N=400): W_acc ≥ 25 bits

**32 bits was selected because:**
1. Industry standard for 8-bit MAC arrays (Google TPU, FINN, Jacob et al. 2018)
2. Provides 16 guard bits → handles up to 65,536 worst-case accumulations
3. 328× margin over LeNet-5 worst case (~400 accumulations)
4. Natural power-of-two width — clean SystemVerilog, clean verification
5. DSP48E2 uses 48 bits internally regardless; 32-bit RTL width governs
   behavioral simulation and PE output routing

**Accumulator register:** `logic signed [31:0] accumulator;`

**DSP48E2 mapping:** The 32-bit RTL accumulator infers a DSP48E2 with its
native 48-bit internal accumulator. Vivado connects only the lower 32 bits to
fabric routing; upper bits are optimized away.

### 7.6 Overflow Behaviour

**DECISION** — **No overflow handling logic in the V1 PE. Overflow is
mathematically impossible by construction** (2026-08-10, recorded in
`docs/PROJECT_STATE.md`).

**Proof:**
- Max |product| = 128 × 128 = 16,384 (8-bit signed operands)
- 32-bit signed accumulator range: [-2^31, 2^31-1] ≈ [-2.15×10^9, 2.15×10^9]
- Max accumulations before overflow: floor(2^31 / 16,384) ≈ 131,072
- LeNet-5 worst-case accumulations per PE: ~400 (FC1 layer)
- Safety margin: 131,072 / 400 ≈ **328×**

**DSP48E2 internal path:** The DSP48E2's native 48-bit accumulator provides
additional headroom internally (would require >1.7×10^10 accumulations to
overflow). The 32-bit RTL width governs behavioral simulation wrapping; the
48-bit internal path means synthesis-level overflow is also impossible.

**Policies not adopted:**
- **Wrap:** Natural two's-complement behavior but produces incorrect results
- **Saturate:** Requires ~20-30 LUTs/PE (comparator + mux); non-deterministic
  accumulation order effects (A2Q, 2023); unjustified when overflow is impossible
- **Flag:** Requires detection logic + status register; unnecessary complexity

If a future workload genuinely risks overflow, the fix is wider accumulation
or per-layer scale calibration, not per-PE saturation logic.

### 7.8 Pipeline / Cycle Architecture

**DECISION** — The V1 PE uses a **registered MAC with combinational
activation forwarding, fixed (non-parameterized) depth** (2026-08-10,
recorded in `docs/PROJECT_STATE.md`).

**Architecture:**
- MAC: DSP48E2-inferred, MREG=1 (registered product), PREG=1 (registered
  accumulator). 2-cycle latency from activation-in to accumulator-update.
- Activation forwarding: combinational (`act_out = act_in`). Zero forwarding
  delay. All PEs see identical pipeline timing — no systolic skew.
- Weight: registered (BREG=1), loaded via `weight_load`.
- Result output: separately registered in fabric, loaded via `result_request`.
  result_request is level-sensitive: result_out tracks accumulator every
  cycle while result_request is asserted.
- accum_clear resets BOTH accumulator AND product register to 0 (pipeline
  flush). This prevents stale product from being added to the zeroed
  accumulator in the cycle after accum_clear deasserts.
- Throughput: 1 MAC/cycle/PE in steady state (after 2-cycle pipeline fill).
- Pipeline depth: **fixed** for V1. Not parameterized. If Vivado synthesis
  shows timing violations, the pipeline can be deepened in a subsequent
  iteration.

**Latency summary:**

| Path | Latency (cycles) |
|------|-----------------|
| activation_in → accumulator update | 2 |
| weight_load → weight available for MAC | 1 |
| result_request → result_out valid | 1 |
| accum_clear → accumulator = 0 | 1 |
| activation_in → act_out (forwarding) | 0 (combinational) |
| Pipeline fill (first result after weight load) | 2 |

**DSP48E2 mapping:** Multi-stage behavioral RTL infers one DSP48E2 per PE
with MREG=1, PREG=1:
```systemverilog
// Multi-stage coding to infer MREG=1 (single-line acc <= acc + a*b infers MREG=0)
logic signed [15:0] product;  // registered product (MREG)
always_ff @(posedge clk) begin
    if (!zero_skip) product <= activation_in * weight;
end
always_ff @(posedge clk) begin
    if (accum_clear)    accumulator <= '0;
    else if (!zero_skip) accumulator <= accumulator + product;
end
```
The `weight <= weight_in` infers BREG=1. AREG is bypassed (combinational
activation input). Expected resource: 1 DSP48E2 per PE (64 total for the
8×8 array); this requires the `(* use_dsp = "yes" *)` attribute on the product
datapath, because Vivado 2023.1 does not auto-infer a DSP for 8×8 operands
(§4.5, §8.1). At -2 speed grade: ~460 MHz achievable (2-stage) vs ~660 MHz
for full 3-stage pipeline (AREG+BREG+MREG+PREG); V1 accepts 2-stage for
simplicity; deepen if synthesis requires it.

### 7.7 Fixed-Point / Quantisation

**DECISION** — The fixed-point convention is formalised as part of the V1
numerical format (§7.1):

- **Format:** signed 8-bit two's-complement integer
- **Implicit scale factor:** 1/256 = 2^(-8)
- **Real value:** `v_real = v_int / 256`
- **Range:** [-0.5, +0.49609375]
- **Resolution:** 0.00390625

The PE performs integer multiply-accumulate. The interpretation of those
integers as fixed-point values (scaling, binary-point position,
requantisation) is external to the PE and is handled at the array output or
in the accumulation drain path. Post-accumulation rescaling to return from
Q16 scale (product) to Q8 scale is an array-level concern.

**Q8 vs INT8:** The roadmap's "Q8" uses a fixed power-of-two scale (1/256).
Standard "INT8 quantization" uses per-layer calibrated scales with optional
zero-points. These are different conventions. The roadmap lists "INT8
quantization path" as future work (§10). If adopted later, the PE datapath
does not change — only the weight extraction script and drain-path rescaling
change.

---

## 8. DSP / Pipeline Architecture

### 8.1 Behavioural RTL and DSP Inference

Write behavioural SystemVerilog using `*` and `+` operators. Vivado ML
2023.1 infers DSP48E2 slices. Multi-stage RTL is required to infer MREG=1
(single-line `acc <= acc + a*b` infers MREG=0). See §7.8 for the recommended
multi-stage coding pattern.

The synthesis attribute `(* use_dsp = "yes" *)` is **required** for this
project's signed 8×8 PE: under default synthesis (`use_dsp = "auto"`) Vivado
ML 2023.1 maps an 8×8 MAC to fabric LUTs/CARRY8 rather than a DSP48E2 (the
27×18 multiplier would be ~94% idle). `rtl/common/pe.sv` therefore applies the
attribute to the `product` datapath. (For wider operands, e.g. 16×16, Vivado
auto-infers a DSP without the attribute — the 8×8 case sits below the auto-DSP
threshold.)

### 8.2 Pipeline Registers

**DECISION** — Pipeline architecture is resolved (§7.8, Decision 6):
registered MAC (MREG + PREG), combinational activation forwarding, fixed
depth. The DSP48E2 provides optional pipeline registers (§4.4); the V1 PE
uses MREG=1, PREG=1, BREG=1 (weight register), AREG=0 (combinational
activation input).

### 8.3 Latency and Throughput

- **Latency:** 2 cycles activation-in → accumulator-update (§7.8).
- **Throughput:** 1 MAC/cycle/PE in steady state after 2-cycle pipeline fill.

### 8.4 Forwarding Pipeline Alignment

**DECISION** — Combinational activation forwarding (§7.8). `act_out =
activation_in` (continuous assignment). Zero forwarding delay. No pipeline
skew between columns. All 8 PEs in a row see the same activation in the
same cycle.

---

## 9. zero_skip Extension

### 9.1 Roadmap Requirement

**REQUIREMENT** (R7) — The PE has a `zero_skip` input port. When asserted:

1. **Multiplication is bypassed** (functionally — the product is not added to
   the accumulator).
2. **The accumulator does not update.**

The port exists in the V1 PE (§4, §7 repository structure). It is:

- **Connected** in Team A's V2 build — driven by `sparsity_manager.sv`.
- **Unconnected** (tied LOW or left floating with a pulldown) in Team B's
  V1/V2 build — the PE operates as if zero_skip is never asserted.

### 9.2 Functional Behaviour

**REQUIREMENT** (R7) — When `zero_skip` is asserted:
- The accumulator does NOT update (holds its previous value).
- The product register still updates (multiplication proceeds; per §5.7,
  `zero_skip` gates only the accumulator update, not the multiply).
- Activation forwarding continues normally (§9.3).
- Priority: below `accum_clear` (§5.7). If both asserted, `accum_clear` wins.

```systemverilog
// Conceptual — consistent with §5.7 priority and §7.8 pipeline
// zero_skip gates accumulator update only; product register still updates
```

### 9.3 Forwarding During zero_skip

**IMPLIED** — Activation forwarding (`act_out`) is expected to continue during
zero_skip cycles. If a PE in the middle of a row skips a MAC, the activation
must still reach downstream PEs to maintain systolic alignment. This is an
architectural implication of the systolic data movement requirement (R8) and
is not stated explicitly by the roadmap for the zero_skip case. The forwarding
behaviour during zero_skip should be verified during array-level scheduling.

### 9.4 Candidate Implementations

**CANDIDATE** — Two approaches have been identified for evaluation:

1. **Mux-based hold:** A multiplexer selects between `accum + product` and
   `accum` based on `zero_skip`. The multiplier still operates but its output
   is discarded.

2. **Clock-enable / gating:** Disable the accumulator register update. The
   feasibility and synthesis behaviour of this approach on DSP48E2-inferred
   accumulators should be evaluated during implementation.

The final implementation should be selected after DSP inference and synthesis
evaluation on the target device.

### 9.5 PE Bypass vs. Upstream Sparsity Throughput

**Important distinction** — The PE-level `zero_skip` is a **functional bypass**
mechanism. It prevents the PE from corrupting its accumulator when the
Sparsity Manager signals a skipped activation.

It does **not**, by itself, create cycle-level throughput improvement. Any
throughput gain from sparsity comes from the **upstream Sparsity Manager**
(Team A, `sparsity_manager.sv`), which:

1. Detects zero activations in the input stream.
2. Drops them from the FIFO.
3. Compacts the non-zero stream.
4. Presents a denser stream to the PE array, reducing the number of cycles
   needed to process a layer.

The PE's `zero_skip` port is the **interface** that makes this possible. The
throughput gain is an **array-level** effect, not a PE-level effect.

---

## 10. Verification Requirements

This section defines the verification categories the V1 PE must satisfy.
It references the roadmap's V1 testbench: `tb_pe.sv` — "Unit testbench —
compare PE output against golden" (§4 V1 table, §6.1).

### 10.1 Reset Behaviour

- After reset deassertion, accumulator = 0 (or a defined initial value).
- Control state is in a known initial condition.
- Output is in a defined state.

### 10.2 Single Multiply-Accumulate

- Present one (activation, weight) pair.
- Verify: `result_out = activation × weight` after the configured pipeline
  latency (exact latency to be verified after pipeline architecture is
  selected).

### 10.3 Repeated Accumulation (No Drain)

- Present a sequence of N (activation, weight) pairs without draining.
- Verify: `accum = Σ(activation_i × weight_i)` for i = 1..N.

### 10.4 Weight Loading (WS Mode)

- Load a weight. Verify it is held stationary across multiple activation
  cycles.
- Load a new weight. Verify the PE switches to using the new weight.

### 10.5 Result Exposure

- Verify that `result_out` reflects the internal accumulator when the FSM
  signals (or when `result_request` is asserted, depending on the chosen
  protocol).
- Verify that result exposure does not corrupt accumulator state (unless
  read-and-clear is adopted).

### 10.6 Activation Forwarding

- Verify `act_out` = `activation_in` (or appropriately delayed by the
  configured pipeline) on every cycle.
- Verify forwarding during both normal and zero_skip cycles.

### 10.7 Signed Arithmetic (Conditional on Q8 Adoption)

These test cases assume the roadmap's Q8 signed 8-bit working assumption.
They must be updated if a different numerical format is adopted.

- Verify correct product sign: `{+, -} × {+, -}`.
- Verify correct behaviour at min-negative: `-128 × -128 = +16384` (fits
  within 16-bit product).
- Verify `-128 × 127` and `127 × -128` — asymmetric two's-complement range.

### 10.8 Corner Cases (Conditional on Q8 Adoption)

These test cases assume the roadmap's Q8 signed 8-bit working assumption.
They must be updated if a different numerical format is adopted.

- Activation = 0, weight ≠ 0.
- Activation ≠ 0, weight = 0.
- Both operands = 0.
- Maximum positive: `127 × 127 = 16129`.
- Maximum negative: `-128 × -128 = 16384`.
- Mixed extremes: `127 × -128 = -16256`.

### 10.9 Overflow Behaviour

**DECISION** — Overflow is mathematically impossible by construction (§7.6).
Verification confirms that with 8-bit operands and 32-bit accumulator, the
accumulator never overflows for the LeNet-5 workload (328× safety margin).
No overflow handling logic to verify.

### 10.10 zero_skip Behaviour

- Verify: `zero_skip` asserted → accumulator unchanged after the cycle.
- Verify: `zero_skip` deasserted → normal accumulation resumes.
- Verify: activation forwarding continues correctly during zero_skip.
- Verify: back-to-back zero_skip → normal → zero_skip transitions.

### 10.11 Back-to-Back Operations

- Verify correct behaviour with new operands presented on consecutive cycles.
- Verify no pipeline bubbles or data hazards between successive operations.
- Verify that after the initial pipeline fill, new operands can be accepted
  every cycle and results emerge at the expected steady-state rate (candidate
  target: one MAC operation initiated per cycle per PE, to be confirmed after
  pipeline architecture is selected).

### 10.12 Pipeline Latency

**DECISION** — Pipeline latency is 2 cycles from activation-in to
accumulator-update (§7.8). Verify:
- Results emerge after 2 pipeline stages.
- Combinational forwarding incurs 0-cycle delay (act_out = act_in).
- Steady-state throughput = 1 MAC/cycle/PE after pipeline fill.

### 10.13 Control Priority

Verify the priority order defined in §5.7:
- rst overrides all other controls.
- accum_clear wins over zero_skip.
- weight_load + MAC: MAC uses OLD weight in assertion cycle.
- Simultaneous result_request + MAC: result_out captures pre-update accumulator.
- accum_clear clears both accumulator AND product register.

### 10.14 accum_clear Pipeline Flush

Verify that accum_clear prevents stale-product add-back:
- Assert accum_clear for one cycle.
- Next cycle with accum_clear=0 and zero_skip=0: accumulator starts from 0,
  product register was also cleared, so first accumulation is correct.
- Verify that accum_clear does NOT require a 2-cycle assertion.

---

## 11. Open Architectural Decisions

### 11.1 Numerical Format Finalisation

**DECISION** — **Resolved (2026-08-10).** The V1 numerical format is signed
8-bit two's-complement with implicit scale factor 1/256. Recorded in
`docs/PROJECT_STATE.md` and specified in `docs/specs/PE_SPEC.md` §7.1.

**Blocking for RTL?** **Resolved** — operand width (8 bits), product width
(16 bits), and signedness established.

### 11.2 Accumulator Width

**DECISION** — **Resolved (2026-08-10).** The V1 PE uses a 32-bit signed
two's-complement accumulator. Recorded in `docs/PROJECT_STATE.md` and
specified in `docs/specs/PE_SPEC.md` §7.5.

**Blocking for RTL?** **Resolved** — accumulator register width is 32 bits.

### 11.3 Product Representation

**DECISION** — **Resolved (consequence of Decision 1).** Full 16-bit product,
no truncation inside the PE. Recorded in `docs/specs/PE_SPEC.md` §7.4.

**Blocking for RTL?** **Resolved** — product width is 16 bits.

### 11.4 Overflow / Saturation Behaviour

**DECISION** — **Resolved (2026-08-10).** Overflow is mathematically
impossible by construction with 32-bit accumulator and 8-bit operands
(328× margin over LeNet-5 worst case). No overflow handling logic in the
V1 PE. Recorded in `docs/PROJECT_STATE.md` and specified in
`docs/specs/PE_SPEC.md` §7.6.

**Blocking for RTL?** **Resolved** — no overflow logic required.

### 11.5 Pipeline Depth

**DECISION** — **Resolved (2026-08-10).** Pipeline architecture: registered
MAC with combinational activation forwarding, fixed depth. DSP48E2-inferred
MREG=1, PREG=1 (2-cycle MAC latency). Recorded in `docs/PROJECT_STATE.md`
and specified in `docs/specs/PE_SPEC.md` §7.8.

**Blocking for RTL?** **Resolved** — cycle-level behaviour defined.

### 11.6 Clock Frequency Target

**Why it matters:** Drives pipeline-depth selection and determines whether
additional fabric pipeline registers are needed beyond DSP48E2 internal
registers. Listed as undecided in `docs/PROJECT_STATE.md`. §4 V1 says
"Timing closes at target freq" without specifying the target.

**Blocking for RTL?** **Partially** — the PE can be designed with a
parameterisable pipeline; the target frequency determines the minimum viable
depth.

### 11.7 Result-Read Protocol (FSM↔PE Handshake)

**DECISION** — **Resolved (2026-08-10).** The protocol is: FSM asserts
`result_request` → PE captures accumulator into registered `result_out`
next cycle → accumulator unchanged (read-without-clear). Separate
`accum_clear` signal resets accumulator. No flow control (valid/ready) in
PE. Array-level column-sequential drain is outside PE scope. Recorded in
`docs/PROJECT_STATE.md` and specified in `docs/specs/PE_SPEC.md` §5.2.

**Blocking for RTL?** **Resolved** — PE control interface defined.

### 11.8 Reset Semantics

**DECISION** — **Resolved (2026-08-10).** The V1 PE uses synchronous reset,
active-high polarity (`rst`). All sequential logic uses
`always_ff @(posedge clk)` with `if (rst)` for reset. Resets: accumulator ← 0,
weight ← 0, result_out ← 0. Recorded in `docs/PROJECT_STATE.md` and
specified in `docs/specs/PE_SPEC.md` §4.6.

The roadmap's Skill 1 `negedge rst_n` pattern is explicitly a code-generation
example, not a project requirement. Synchronous reset is mandatory for DSP48E2
inference (DSP48E2 has no async reset inputs; async reset forces registers
into fabric at ~65 extra FFs + ~32 extra LUTs per PE per UG949).

**Blocking for RTL?** **Resolved** — reset polarity and synchronicity
established.

### 11.9 zero_skip Trigger Granularity

**DECISION** — Not a PE concern. The PE is agnostic to the trigger condition.
The Sparsity Manager (Team A, `sparsity_manager.sv`) determines when
zero_skip is asserted. The PE responds to the signal regardless of the
condition (activation==0, weight==0, either, or external control). The V1
PE port exists and is functional; in Team B's build it is tied LOW.

**Blocking for RTL?** **Resolved** — PE behavior under zero_skip is fully
specified in §5.3, §5.7, and §9.

### 11.10 Weight Forwarding Port (`wgt_out`)

**DECISION** — Not in V1. `wgt_out` is a V2 Team B concern (Output
Stationary mode, §3.2). The V1 PE omits this port per §5.6. It will be
added in V2 if needed.

**Blocking for RTL?** **Resolved** — omitted from V1 PE.

### 11.11 V2 Extensibility Hooks

**DECISION** — Not in V1. `shift_enable`, `load_enable`, and
`accumulate_enable` are V2 Team B Dataflow Controller signals (§3.2).
The V1 PE omits these ports per §5.6. The roadmap's V2 description
explicitly says `systolic_array.sv` (not `pe.sv`) is updated to wire
controller outputs.

**Blocking for RTL?** **Resolved** — omitted from V1 PE.

---

## 12. RTL Entry Criteria

### 12.1 Blocking Decisions (Must Resolve Before RTL)

These decisions affect observable PE behaviour or the RTL module interface.
They must be resolved before `pe.sv` is written.

| Decision | Reference | Impact |
|----------|-----------|--------|
| Numerical format finalisation | §11.1 | **RESOLVED** — 8-bit signed operands |
| Accumulator width | §11.2 | **RESOLVED** — 32-bit signed |
| Product representation | §11.3 | **RESOLVED** — 16-bit full precision |
| Result-read protocol | §11.7 | **RESOLVED** — result_request strobe, read-without-clear |
| Overflow behaviour | §11.4 | **RESOLVED** — impossible by construction |
| Reset semantics | §11.8 | **RESOLVED** — synchronous, active-high (rst) |
| Pipeline depth (cycle-level behaviour) | §11.5 | **RESOLVED** — 2-stage registered MAC, combinational forwarding |

### 12.2 Non-Blocking Decisions (Can Be Deferred or Are V2 Concerns)

| Decision | Reference | Deferral Strategy |
|----------|-----------|-------------------|
| Clock frequency target | §11.6 | Defer to synthesis; PE is parameterised for pipeline depth |
| Weight forwarding port | §11.10 | V2 Team B concern; may be omitted or included as optional in V1 |
| V2 extensibility hooks | §11.11 | V2 Team B concern; optional V1 ports to be decided by implementer |

### 12.3 Readiness Gate

All §12.1 decisions are RESOLVED. Before `pe.sv` is written:

1. This specification (§5–§8) is the implementation contract.
2. `docs/PROJECT_STATE.md` records all six decisions.
3. The weight extraction script (§5.3 of roadmap) must use signed 8-bit,
   scale 1/256, range [-128, 127].
4. The golden model (`golden_model.py`, §4 V1 table) must match the PE's
   arithmetic behaviour exactly: 2-cycle registered MAC, 32-bit accumulator,
   read-without-clear, no overflow handling.
5. The implementing engineer reads §5.7 (control priority) and §7.8
   (pipeline architecture) as the primary functional contract.

---

## 13. PE-v2 Specification Delta (V2 Reconfigurable Dataflow)

> **Status:** RESOLVED — interface and WS accumulator semantics recorded as
> **Decision 10** (`docs/PROJECT_STATE.md`, 2026-08-18). This is the PE change
> required for Team B's V2 reconfigurable-dataflow mode. It does **not** modify
> the V1 PE: `rtl/common/pe.sv` and `sim/tb_pe.sv` remain frozen and untouched.

### 13.1 Scope

PE-v2 is a **superset** of the V1 PE. The V1 contract (§5–§8, Decisions 1–6) is
unchanged. The V2 dataflow is **true spatial Weight-Stationary** (Decision 10):
weights held in PEs, activations shift, partial sums cascade vertically. PE-v2
adds exactly **three ports and one mux**; the weight, product, result, and
activation-forwarding blocks are V1-identical.

This supersedes the roadmap's §3.2/§5.6 wording that named
`wgt_out`/`shift_enable`/`load_enable`/`accumulate_enable` as the V2 PE hooks:
Decision 10 defines the actual V2 PE change as
`psum_in`/`psum_out`/`dataflow_mode`. §11.10/§11.11 remain historical/roadmap
context only.

### 13.2 Added Ports

| Signal | Direction | Width | Status |
|--------|-----------|-------|--------|
| `psum_in` | input | 32 | **DECISION** (Decision 10) — WS cascade addend; ignored in OS mode |
| `psum_out` | output | 32 | **DECISION** (Decision 10) — registered accumulator value (`= accumulator`) |
| `dataflow_mode` | input | 1 | **DECISION** (Decision 10) — `0` = OS (V1), `1` = WS |

Note: the internal `dataflow_mode` encoding (`0`=OS, `1`=WS) is a PE-level
convention chosen so that the reset default (`0`) yields V1 OS behaviour. Its
mapping to the ARM-visible `DATAFLOW_MODE` register (roadmap §3.2 encodes
`0`=WS, `1`=OS) is deferred to the register-map definition and is not resolved
here.

### 13.3 Accumulator Semantics (WS vs OS)

**DECISION** — `dataflow_mode` selects only the accumulator addend source:

```systemverilog
assign psum_out = accumulator;

always_ff @(posedge clk) begin
    if (rst)                 accumulator <= '0;
    else if (accum_clear)    accumulator <= '0;
    else if (dataflow_mode)  accumulator <= psum_in + product;   // WS: cascade
    else                     accumulator <= accumulator + product; // OS: hold (V1)
end
```

- **OS (`dataflow_mode = 0`):** `accumulator <= accumulator + product` —
  bit-identical to the V1 accumulator block. `psum_in` is ignored.
- **WS (`dataflow_mode = 1`):** `accumulator <= psum_in + product` — the
  accumulator register is reused as a partial-sum pass-through; `psum_out`
  feeds the PE below.
- `rst` and `accum_clear` clear `accumulator` (and `product`) in both modes,
  independent of `dataflow_mode`.
- `zero_skip` still zeroes `product` (§5.3/§9), so in WS a skipped MAC passes
  `psum_in + 0` through unchanged — no new gating.

### 13.4 Control Priority

**DECISION** — The V1 control priority (§5.7) is **unchanged**:
`rst` > `weight_load` > `accum_clear` > `result_request` > `zero_skip` > MAC.
`dataflow_mode` is a static configuration input, stable throughout a layer's
COMPUTE phase; it is not a per-cycle control and only selects the accumulator
addend source. `rst`/`accum_clear` override it in both modes.

### 13.5 OS Equivalence Requirement

**REQUIREMENT** — PE-v2 in OS mode must be behaviourally equivalent to the V1
PE. Acceptance criterion: the frozen `sim/tb_pe.sv` (55/55) and
`sim/tb_systolic_array.sv` (379/379) pass unchanged against PE-v2 (and
array-v2) with `dataflow_mode = 0` and `psum_in = 0`. `psum_out` is a passive
tap of `accumulator` and does not affect OS behaviour.

### 13.6 Open V2 Array-Level Items (not resolved in this delta)

The PE-v2 contract above is complete for the PE change. The following V2
array-level items remain open and are **not** resolved here:

| # | Item | Status |
|---|------|--------|
| 1 | WS weight-loading mechanism | **OPEN DECISION** |
| 2 | WS activation delivery (per-row streams vs diagonal skew) | **OPEN DECISION** |
| 3 | WS result collection (reuse `result_req` reading row 7 vs dedicated bottom-row bus) | **OPEN DECISION** |
| 4 | Reconfiguration flush sequence (full `rst` vs lighter flush) | **OPEN DECISION** |
| 5 | DSP48E2 inference of the mode mux / PCIN–PCOUT cascade | **RESOLVED (2026-08-19)** — Vivado ML 2023.1 implements the muxed WS addend (`psum_in`) through the DSP48E2 **C input** (OPMODE "C or P"), not PCIN/PCOUT. Functionally equivalent (array 335/335 PASS, 64 DSP48E2 / 0 CARRY8); PCIN/PCOUT is an implementation detail, not a V2 functional requirement |

---

## Appendix A: References

| Reference | Description |
|-----------|-------------|
| `docs/reference/architecture_major_project.pdf` | Architecture roadmap v3.0 — primary PE requirements source |
| `CLAUDE.md` | Project instructions and coding conventions |
| `docs/PROJECT_STATE.md` | Master project decisions and status |
| `docs/DEVELOPMENT_SETUP.md` | Development environment and HW workflow |
| DS925 | Zynq UltraScale+ MPSoC Data Sheet (XCK26) |
| UG579 | UltraScale Architecture DSP48E2 Slice User Guide |
| Gemini Pro PE Research Report | External research input (summarised in spec brief) |

---

> **Next Step:** Resolve the blocking decisions in §11 and §12.1. Record them
> in `docs/PROJECT_STATE.md`. Return to this specification to finalise the PE
> interface, then begin `rtl/common/pe.sv` implementation.
