# CNN Accelerator — Project State

## Current Status

The repository baseline and shared development environment documentation are
established.

The project targets the AMD Kria KV260 / XCK26 using SystemVerilog and
Vivado ML 2023.1.

The repository is hosted on GitHub and is the shared source of truth for the
development team.

**Current engineering phase:** V1 systolic array implemented
(`rtl/common/systolic_array.sv`) and verified by its integration testbench
(`sim/tb_systolic_array.sv`, 379/379 checks PASS in Vivado ML 2023.1). Array
compile/elaboration is verified (`xvlog`/`xelab` clean), and the V1 PE remains
verified at 55/55 PASS. The systolic-array RTL + verification milestone is
complete, and DSP48E2 inference is established (1 DSP/PE, 64/array). The
input-feed/line-buffer module is the next major module; full array-level
integration (controller/BRAM/input-feed wiring) remains future work.

## Verified Development Environment

- Host OS: Ubuntu 22.04.5 LTS 64-bit
- Python: 3.10.12
- Vivado ML: 2023.1
- Target device: `xck26-sfvc784-2LV-c`
- PyTorch: 2.1.2+cpu
- torchvision: 0.16.2+cpu
- NumPy: 1.26.4
- Claude Code: 2.1.207
- Graphify: 0.8.39

## Repository Setup

Project-scoped Claude Code configuration is present under `.claude/`.

The shared development setup is documented in `docs/DEVELOPMENT_SETUP.md`.

The Research skill is included under `.claude/skills/research/`.

Generated build and local data must remain outside the committed source where
specified by `.gitignore`.

## Architecture Status

### Architecture Roadmap

The architecture roadmap is available locally at:

`docs/reference/architecture_major_project.pdf`
("CNN Hardware Accelerator on AMD Kria KV260 — Complete Project Reference v3.0")

It is a **local reference only** and has **not** been committed to Git. It is
the high-level project roadmap and initial reference. It is **not** an
authoritative detailed implementation specification.

Do not treat assumptions from the roadmap as finalized requirements.

### PE Specification

The first technical specification has been created and reviewed:

`docs/specs/PE_SPEC.md` — PE Specification (V1 Baseline)

This document records what the roadmap explicitly establishes, what external
research informs, candidate architectures, and open decisions. It uses a label
system: REQUIREMENT, RESEARCH FACT, CANDIDATE, IMPLIED, OPEN DECISION.

### PE Implementation and Verification

The V1 PE is implemented and functionally verified:

- **RTL:** `rtl/common/pe.sv` — implements the finalized V1 contract
  (`docs/specs/PE_SPEC.md` §5–§8; Decisions 1–6).
- **Testbench:** `sim/tb_pe.sv` — self-checking, non-UVM unit testbench.
- **Simulation:** Vivado ML 2023.1 (`xvlog`/`xelab`/`xsim`) — **55/55 checks
  PASS, 0 FAIL**.
- **Verified scenarios:** synchronous reset, weight load/retention/reload,
  signed 8×8 arithmetic and corner cases, 16→32-bit accumulation, repeated and
  back-to-back MAC, combinational activation forwarding, `zero_skip`,
  `result_request` read-without-clear, `accum_clear` pipeline flush, the §5.7
  control-priority combinations, and the finalized pipeline timing.

Synthesis (DSP48E2 inference) is **resolved** — see "Synthesis and DSP48E2
Inference" below. Array-level (`systolic_array.sv`) integration with the
controller/BRAM/input-feed remains future work.

### Systolic Array Specification

The V1 systolic array specification has been drafted:

`docs/specs/SYSTOLIC_ARRAY_SPEC.md` — Systolic Array Specification (V1 Baseline)

It records the resolved V1 dataflow (Decision 7): **Output-Stationary** —
weight-broadcast + activation-shift + local PE accumulation. It defines the 8×8
topology, the row/column mapping, the cycle-level schedule, the array interface,
and the verification requirements. It amends the roadmap's "fixed
weight-stationary" wording as historical/source context.

The five array scheduling decisions (Decision 8 — padding, skew, drain, tap
order, peak-vs-sustained) are also recorded, leaving `systolic_array.sv`
unblocked. The input-feed / line-buffer unit is a separate upcoming module.

### Systolic Array Implementation and Verification

The V1 systolic array is implemented and functionally verified:

- **RTL:** `rtl/common/systolic_array.sv` — an 8×8 grid of 64 verified PEs with
  array-level activation-shift interconnect (7 registers per row), per-row weight
  broadcast, array-wide control fan-out (`weight_load`/`accum_clear`/`zero_skip`),
  and column-sequential result drain. No FSM, no partial-sum cascade, no sparsity
  (per `docs/specs/SYSTOLIC_ARRAY_SPEC.md`).
- **Testbench:** `sim/tb_systolic_array.sv` — self-checking, directed, non-UVM,
  with an independent reference model (`PE(r,c) = Σ_t a_r[t]·w_r[t+c]`, streams
  zero-padded).
- **Simulation:** Vivado ML 2023.1 (`xvlog`/`xelab`/`xsim`) — **379/379 checks
  PASS, 0 FAIL**. Covers synchronous reset, 64-PE structure, activation shift and
  boundary columns, per-row weight broadcast, the one-cycle weight-lead skew,
  signed 8×8 arithmetic, 32-bit accumulation, `zero_skip`, `accum_clear`,
  column-sequential drain + read-without-clear, idle rows 6–7, back-to-back
  groups, and a complete deterministic convolution.
- **Compile/elaboration:** `xvlog -sv` and `xelab` on `pe.sv` + `systolic_array.sv`
  (+ testbench) are clean (0 errors).

The input-feed/line-buffer module is the next major module; full array-level
integration (controller/BRAM/input-feed wiring) remains future work.

### Synthesis and DSP48E2 Inference

DSP48E2 inference for the V1 PE and 8×8 array is established and verified
(Vivado ML 2023.1, target `xck26-sfvc784-2LV-c`):

- **Baseline (default synthesis):** Vivado's default `use_dsp = "auto"` mapped
  the signed 8×8 PE MAC entirely to fabric — **0 DSP48E2**, 97 LUT / 12 CARRY8
  per PE (array: 0 DSP, 7,054 LUT / 5,632 FF / 768 CARRY8). The 8×8 operands
  fall below Vivado's auto-DSP threshold (a DSP48E2 27×18 multiplier would be
  ~94% idle).
- **Root cause (controlled probes):** an 8×8 multiply and an 8×8 MAC each infer
  0 DSP48E2 under default synthesis; an explicit `(* use_dsp = "yes" *)` (or
  ≥16-bit operands) is required to obtain DSP48E2 inference. The PE's control
  gating (`rst`/`accum_clear`/`zero_skip`) is **not** the blocker.
- **Fix:** `rtl/common/pe.sv` adds a targeted `(* use_dsp = "yes" *)` attribute
  on the `product` datapath. No other behavioural change.
- **Single PE synthesis:** exactly **1 DSP48E2** — `AREG=0, BREG=1, MREG=1,
  PREG=1` (weight→BREG, product→MREG, accumulator→PREG, combinational
  activation→AREG=0). Fabric drops to 2 LUT / 0 CARRY8 / 32 FF (result_out
  only).
- **8×8 array synthesis:** exactly **64 DSP48E2** (1 per PE), 1,039 LUT /
  2,048 FF / 0 CARRY8 / 0 BRAM / 0 URAM. All DSPs use BREG=1, MREG=1, PREG=1;
  in the array Vivado may absorb the activation shift-chain registers into the
  DSP AREG inputs (observed AREG = 0/1/2 across columns), so AREG is not
  uniformly 0 at array level — functionally equivalent.
- **Functional verification unchanged:** `tb_pe` 55/55 PASS and
  `tb_systolic_array` 379/379 PASS (the attribute is a synthesis hint only;
  simulated behaviour is identical).

**RESOLVED:** DSP48E2 inference for the V1 PE/array is confirmed (1 DSP/PE,
64/array). Numerical format, accumulator width, and cycle-level behaviour are
unchanged.

### Post-Route Implementation Baseline

A baseline post-synthesis/post-route implementation of the V1 8×8 array was run
(Vivado ML 2023.1, top `systolic_array`, target `xck26-sfvc784-2LV-c`). This is
a timing/implementation baseline only — no RTL, pipeline, or constraint-source
changes were made.

- **Clock target (analysis only):** 200 MHz (5.000 ns), applied as a baseline
  constraint in the generated build flow only. This is **not** a finalized
  project clock requirement — the clock frequency target remains an open
  decision (`PE_SPEC.md` §11.6) until the controller/BRAM/input-feed
  integration is timed.
- **Implementation mode:** out-of-context (OOC). The flat `systolic_array`
  top exposes 397 I/O ports (256-bit `result_out` + 128 data inputs + control),
  which exceeds the ~189 bonded I/O sites available on the KV260/XCK26 package.
  This is expected: the array is a datapath sub-block intended to be driven by
  an on-chip controller/BRAM/PS-AXI interface, not by package pins. OOC isolates
  the internal register-to-register timing of the array core.
- **Post-route DSP count:** 64/64 DSP48E2 remain present after
  placement/routing.
- **DSP register configuration:** BREG=1, MREG=1, PREG=1 for all 64 DSPs.
- **AREG distribution:** AREG=0 × 8, AREG=1 × 40, AREG=2 × 16. The AREG=1/2
  DSPs reflect the array's activation shift-chain registers being absorbed into
  the DSP A-input registers (AREG absorption); the inter-column activation
  movement is realized via the DSP48E2 ACOUT→ACIN cascade.
- **Critical setup path:** DSP48E2_X11Y11 → DSP48E2_X11Y12, the DSP
  ACOUT→ACIN activation cascade between adjacent columns. The path is
  routing-dominated (~66% route vs ~34% logic of the 1.471 ns data delay, 0
  logic levels) and currently closes at the 200 MHz baseline target.
- **Post-route timing (200 MHz baseline):** WNS +2.772 ns, TNS 0 ns,
  WHS +0.115 ns. Setup, hold, and pulse-width constraints are all met (0 failing
  endpoints). 200 MHz closes with margin (~449 MHz setup-limited capability,
  with OOC clock skew estimated).
- **Resources (post-route):** 64 DSP48E2, 1,032 LUT, 2,048 FF, 0 CARRY8,
  0 BRAM, 0 URAM. The 2,048 FFs are the 64×32 `result_out` registers only; all
  other datapath registers (weight/product/accumulator/shift) live inside the
  DSPs.

**RESOLVED:** the V1 array datapath is timing-clean at the 200 MHz baseline and
DSP48E2 inference (64/64) is preserved through implementation.

## Decisions

### Established V1 PE Requirements (from Roadmap)

The architecture roadmap (§3.2, §4) establishes these V1 PE requirements:

- activation input
- weight input
- activation × weight multiplication
- internal accumulation
- FSM-controlled result exposure
- registered output
- zero_skip input port with functional bypass behaviour (port in V1;
  functional connection is Team A V2)
- 8×8 systolic grid (64 PE instances); activations shift across rows,
  weights shift down columns
- V1 baseline dataflow: Weight-Stationary (weights held in place,
  activations shift)

These are recorded in `docs/specs/PE_SPEC.md` §3.1 as R1–R9.

> Note: the "Weight-Stationary" dataflow label above is the roadmap's original
> wording. The V1 array dataflow was subsequently reclassified to
> Output-Stationary — see Decision 7.

### Numerical Format Status

**Decision 1 — Numerical Format — RESOLVED (2026-08-10).**

The V1 PE uses **signed 8-bit two's-complement integers** for both activation
and weight operands. The implicit scale factor is 1/256 (2^(-8)), making the
representable real range [-0.5, +0.49609375] with resolution 0.00390625.

This is the roadmap's "Q8" working assumption (§5.3), formally adopted as the
V1 numerical format.

| Parameter | Value |
|-----------|-------|
| Activation operand | signed 8-bit two's-complement |
| Weight operand | signed 8-bit two's-complement |
| Operand width | 8 bits |
| Product width | 16 bits (full precision, not truncated inside PE) |
| Fixed-point interpretation | external to PE (array output / drain path) |
| Weight extraction | `w_fixed = clip(round(w × 256), -128, 127)` per roadmap §5.3 |

**Rationale:** consistent with roadmap weight extraction; minimal PE datapath
complexity (integer MAC only); optimal DSP48E2 mapping (one DSP/PE with 27×18
headroom); efficient BRAM usage (8 bits/weight); LeNet-5/MNIST accuracy loss
negligible with 8-bit quantization; preserves Q8→INT8 future-work pathway.

**Known risk:** weights with magnitude ≥ 0.5 are clipped. The team should
measure actual LeNet-5 weight distributions to assess accuracy impact.

### Decision 2 — Accumulator Width — RESOLVED (2026-08-10)

The V1 PE uses a **32-bit signed two's-complement accumulator**.

| Parameter | Value |
|-----------|-------|
| Accumulator width | 32 bits signed |
| Minimum required (LeNet-5 conv1, N=25) | 21 bits |
| Industry standard (TPU, FINN, Jacob 2018) | 32 bits |
| Overflow threshold (worst-case products) | >131,000 |
| LeNet-5 worst case (FC1, N=400) | ~400 |
| DSP48E2 internal | 48 bits (native, "free") |

**Rationale:** Industry standard for 8-bit MAC arrays. 328× safety margin over
LeNet-5 worst case. Natural power-of-two width. DSP48E2 uses 48 bits
internally regardless; 32-bit RTL width governs behavioral simulation and PE
output routing. The exact per-PE accumulation depth (N_max_acc = 25 for conv1)
is not required to justify 32 bits — the margin is so large that any plausible
V1 mapping is safe.

### Decision 3 — Overflow Behaviour — RESOLVED (2026-08-10)

**No overflow handling logic in the V1 PE. Overflow is mathematically
impossible by construction.**

With 32-bit signed accumulator and 8-bit signed operands (max |product| =
16,384), the accumulator permits >131,000 worst-case accumulations before
overflow. LeNet-5 requires at most ~400 accumulations per PE. The safety
margin is 328×.

No wrap, saturation, or overflow-flag logic is implemented. The DSP48E2's
native 48-bit internal accumulator provides additional headroom internally.

### Decision 4 — Result-Read Protocol — RESOLVED (2026-08-10)

**FSM-driven strobe protocol with read-without-clear semantics.**

| Parameter | Value |
|-----------|-------|
| FSM→PE signal | `result_request` (1 bit) |
| PE→FSM signal | `result_out` (32 bits, registered) |
| Accumulator clear | `accum_clear` (1 bit, separate from result_request) |
| Read semantics | Read-without-clear |
| Flow control | None in PE — FSM controls timing globally |
| Array-level drain | Column-sequential (8 PEs/cycle, 8 cycles), outside PE scope |

**Rationale:** Simplest mechanism satisfying R5. Read-without-clear is
universal across all surveyed systolic accelerators. Separate `accum_clear`
preserves debugging capability. No valid/ready handshake in PE — consistent
with the roadmap's FSM-driven architecture.

### Decision 5 — Reset Semantics — RESOLVED (2026-08-10)

**Synchronous reset, active-high polarity (`rst`).**

| Parameter | Value |
|-----------|-------|
| Synchronicity | Synchronous (`always_ff @(posedge clk)`) |
| Polarity | Active-high (`if (rst)`) |
| Registers reset | accumulator, weight, result_out |
| DSP48E2 mapping | Direct (native synchronous reset) |

**Rationale:** DSP48E2 registers support synchronous reset only (UG579, UG949).
Async reset would force MAC registers into fabric (~65 extra FFs + ~32 extra
LUTs per PE per UG949 example). Active-high matches DSP48E2 native reset
polarity. The roadmap's Skill 1 `negedge rst_n` is explicitly a code-gen
example, not a project requirement (per PE_SPEC.md §11.8).

### Decision 6 — Pipeline / Cycle Architecture — RESOLVED (2026-08-10)

**Registered MAC with combinational activation forwarding. Fixed depth.**

| Parameter | Value |
|-----------|-------|
| MAC pipeline | DSP48E2-inferred, MREG=1, PREG=1 |
| MAC latency | 2 cycles (activation-in → accumulator-update) |
| Activation forwarding | Combinational (`act_out = act_in`, 0-cycle delay) |
| Weight register | 1 stage (`weight <= weight_in` on `weight_load`) |
| Result output | 1 stage (`result_out <= accumulator` on `result_request`) |
| Throughput | 1 MAC/cycle/PE (steady state) |
| Parameterization | Fixed (not parameterized for V1) |

**Rationale:** Simplest architecture satisfying all requirements (R1–R9).
MREG=1 is essential for timing and power (Xcell87). Combinational forwarding
eliminates pipeline alignment complexity. 2-cycle latency acceptable for CNN
inference (throughput matters, not latency). DSP48E2 inferred from behavioral
RTL. Fixed depth avoids premature parameterization.

### Decision 7 — V1 Dataflow Reclassification — RESOLVED (2026-08-17)

**The V1 array dataflow is Output-Stationary, not fixed spatial
Weight-Stationary.**

The architecture roadmap (§3.2, §4) describes V1 as "fixed weight-stationary:
weights held in place, activations shift." The finalized PE
(`rtl/common/pe.sv`) has a pinned local accumulator and no `psum_in`/`psum_out`
partial-sum path, so a literal fixed-weight dataflow would compute
`weight × Σ activations`, which is not a convolution. The completed array
specification (`docs/specs/SYSTOLIC_ARRAY_SPEC.md` §3) therefore establishes:

| Parameter | Value |
|-----------|-------|
| V1 dataflow | Output-Stationary (tap-serial) |
| Weight movement | broadcast per row, reloaded every tap |
| Activation movement | shift across columns (array-level registers) |
| Partial-sum path | none in V1 |
| PE status | unchanged, verified (55/55) |
| True spatial Weight-Stationary | deferred to V2 / PE v1.1 (psum cascade) |

This **replaces** the earlier interpretation of V1 as fixed spatial
Weight-Stationary. The roadmap's "weight-stationary" wording is preserved as
historical/source context; the roadmap did not originally specify
Output-Stationary. This is an explicit architectural decision, not a silent
relabeling.

**Rationale:** the Output-Stationary mapping (weight-broadcast +
activation-shift + local accumulation) is the only dataflow that is both
mathematically correct for convolution and compatible with the locked PE; it
sustains one MAC/cycle/PE after pipeline fill. See
`docs/specs/SYSTOLIC_ARRAY_SPEC.md` §3 and §9.

### Decision 8 — V1 Array Scheduling Decisions — RESOLVED (2026-08-17)

Five array scheduling decisions are finalized (see
`docs/specs/SYSTOLIC_ARRAY_SPEC.md` §10, §11, §14–§17). `systolic_array.sv` is
now unblocked.

| # | Decision | Resolution | Rationale |
|---|----------|-----------|-----------|
| 1 | Conv1 output dimension / padding | 24×24 valid (no padding) | Standard LeNet-5/MNIST; 24 = 3×8 → no partial tile |
| 2 | Weight/activation skew | weight stream leads activation by one cycle | Absorbs the PE BREG stage; §16 schedule |
| 3 | Result drain | column-sequential, 8 cycles | Matches `result_request` protocol; 256-bit bus |
| 4 | Tap serialization | row-major `t = K·k_y + k_x` | Deterministic; shared by weight ROM and input feed |
| 5 | "64 MACs/cycle" | peak capability, not sustained | Conv1 sustains 48 (6 rows); rows 6–7 idle (75%) |

**Note:** the input-feed / line-buffer unit is a **separate upcoming module**
with its own specification. Its interface to the array (`act_in` sequence, tap
order, skew) is fully defined by the array spec; only its internals remain.

### Unresolved PE Decisions

The following decisions block `rtl/common/pe.sv` implementation (see
`docs/specs/PE_SPEC.md` §11–§12 for full context):

1. ~~Numerical format finalisation~~ **RESOLVED (2026-08-10)**
2. ~~Accumulator width~~ **RESOLVED (2026-08-10) — 32-bit signed**
3. ~~Product representation~~ **RESOLVED — 16-bit full precision (consequence of Decision 1)**
4. ~~Overflow behaviour (or explicit waiver)~~ **RESOLVED (2026-08-10) — overflow impossible by construction**
5. ~~Result-read protocol (FSM↔PE handshake)~~ **RESOLVED (2026-08-10) — result_request strobe, read-without-clear**
6. ~~Reset semantics (polarity, synchronicity)~~ **RESOLVED (2026-08-10) — synchronous, active-high (rst)**
7. ~~Pipeline depth / cycle-level behaviour~~ **RESOLVED (2026-08-10) — registered MAC, combinational forwarding, fixed depth**

These are **not** finalized:
- clock frequency target
- exact RTL signal names
- `wgt_out` port
- V2 dataflow-controller extensibility hooks

### V1 / V2 Distinction

- **V1** is the shared baseline: **Output-Stationary** dataflow
  (weight-broadcast + activation-shift + local accumulation), no sparsity
  (Decision 7).
- **True spatial Weight-Stationary** (partial-sum cascade, `psum_in`/`psum_out`)
  is deferred to V2 / PE v1.1.
- **Team B reconfigurable dataflow** (per-layer WS/OS mode switching) is a V2
  extension; its Weight-Stationary mode requires the deferred psum cascade.
- **Team A sparsity** is a V2 extension. The V1 PE has the `zero_skip`
  port, but its functional connection to the Sparsity Manager is a V2
  Team A concern.

### Previously Undecided — Now Established

| Parameter | Previous Status | Current Status |
|-----------|----------------|----------------|
| PE count | Undecided | **Established:** 64 (8×8 grid) per roadmap §3.2 |
| V1 dataflow mode | Undecided | **Established:** Output-Stationary (weight-broadcast + activation-shift + local accumulation) per Decision 7; roadmap §3.2 "weight-stationary" wording superseded |
| Numerical format | Undecided | **Established:** Signed 8-bit two's-complement, scale 1/256 (Decision 1, 2026-08-10) |
| Accumulator width | Undecided | **Established:** 32-bit signed two's-complement (Decision 2, 2026-08-10) |
| Overflow behaviour | Undecided | **Established:** Overflow impossible by construction; no handling logic (Decision 3, 2026-08-10) |
| Result-read protocol | Undecided | **Established:** result_request strobe, read-without-clear, separate accum_clear (Decision 4, 2026-08-10) |
| Reset semantics | Undecided | **Established:** synchronous, active-high (rst) (Decision 5, 2026-08-10) |
| Pipeline architecture | Undecided | **Established:** registered MAC (MREG+PREG), combinational forwarding, fixed depth (Decision 6, 2026-08-10) |
| PE forwarding direction | Undecided | **Established:** activations shift across rows per roadmap §3.2 |
| Conv1 output dimension | Undecided | **Established:** 24×24 valid, no padding (Decision 8, 2026-08-17) |
| Weight/activation skew | Undecided | **Established:** weight stream leads activation by one cycle (Decision 8, 2026-08-17) |
| Result drain | Undecided | **Established:** column-sequential, 8 cycles (Decision 8, 2026-08-17) |
| Tap serialization order | Undecided | **Established:** row-major `t = K·k_y + k_x` (Decision 8, 2026-08-17) |
| "64 MACs/cycle" meaning | Undecided | **Established:** peak capability; Conv1 sustains 48 MACs/cycle (Decision 8, 2026-08-17) |

## Open Questions

Record unresolved architectural or implementation questions here as they
arise.

## Milestones

1. Repository baseline and dev environment documentation — **complete**
2. PE specification (`docs/specs/PE_SPEC.md`) — **complete** (draft, reviewed)
3. Numerical format decision — **complete** (2026-08-10)
4. Accumulator width, overflow, result-read decisions — **complete** (2026-08-10)
5. Reset semantics, pipeline architecture decisions — **complete** (2026-08-10)
6. **All six PE-blocking decisions resolved.** RTL entry criteria satisfied.
7. V1 PE RTL implementation (`rtl/common/pe.sv`) — **complete**
8. V1 PE unit testbench (`sim/tb_pe.sv`) — **complete** (55/55 PASS in simulation)
9. V1 systolic array specification (`docs/specs/SYSTOLIC_ARRAY_SPEC.md`) and
   dataflow decision (Decision 7) — **complete** (2026-08-17)
10. V1 array scheduling decisions (Decision 8) — **complete** (2026-08-17);
    `systolic_array.sv` unblocked
11. Systolic array RTL (`rtl/common/systolic_array.sv`) — **complete**
12. V1 systolic-array testbench (`sim/tb_systolic_array.sv`) — **complete**
    (379/379 PASS in Vivado ML 2023.1)
13. Input-feed / line-buffer module (separate spec + RTL) — **pending**
14. Synthesis/DSP48E2 inference — **complete** (1 DSP/PE, 64/array); full
    array-level integration (controller/BRAM/input-feed) — **pending**
15. Team A / Team B V2 extensions — **pending**

## Next Planned Work

1. ~~**Decision 1: numerical format.**~~ **COMPLETE (2026-08-10).**
2. ~~**Decisions 2–4: accumulator width, overflow, result-read.**~~ **COMPLETE (2026-08-10).**
3. ~~**Decisions 5–6: reset semantics, pipeline architecture.**~~ **COMPLETE (2026-08-10).**
4. ~~**V1 PE RTL implementation.**~~ **COMPLETE** — `rtl/common/pe.sv` implemented.
5. ~~**V1 PE unit testbench.**~~ **COMPLETE** — `sim/tb_pe.sv`; Vivado 2023.1
   simulation passes 55/55.
6. ~~**V1 systolic array specification + dataflow decision.**~~ **COMPLETE** —
   `docs/specs/SYSTOLIC_ARRAY_SPEC.md` (draft); Decision 7 recorded (2026-08-17).
7. ~~**Resolve the array's scheduling decisions.**~~ **COMPLETE** — Decision 8
   (padding, skew, drain, tap order, peak-vs-sustained) recorded (2026-08-17).
8. ~~**Systolic array implementation + array-level verification.**~~ **COMPLETE** —
   `rtl/common/systolic_array.sv` + `sim/tb_systolic_array.sv`; Vivado 2023.1
   simulation passes 379/379.
9. Input-feed / line-buffer module specification and RTL.
10. ~~**Vivado synthesis and DSP48E2 inference check on the target device.**~~
    **COMPLETE** — 1 DSP48E2/PE, 64/array (`use_dsp` attribute required for 8×8).

## Research Discipline

- Distinguish roadmap requirements from external research facts.
- Distinguish research facts from engineering recommendations.
- Do not allow common CNN accelerator practice to become a project
  requirement without an explicit team decision.
- Do not over-engineer infrastructure before it is needed; prioritise
  accelerator architecture and RTL development.
