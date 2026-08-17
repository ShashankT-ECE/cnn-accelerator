# CNN Accelerator — Project State

## Current Status

The repository baseline and shared development environment documentation are
established.

The project targets the AMD Kria KV260 / XCK26 using SystemVerilog and
Vivado ML 2023.1.

The repository is hosted on GitHub and is the shared source of truth for the
development team.

**Current engineering phase:** V1 PE implemented (`rtl/common/pe.sv`) and
functionally verified by its unit testbench (`sim/tb_pe.sv`, 55/55 checks PASS
in Vivado ML 2023.1). Synthesis/DSP inference and array-level integration remain.

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

Synthesis (DSP48E2 inference) and array-level (`systolic_array.sv`) integration
remain future work.

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

- **V1** is the shared baseline: Weight-Stationary dataflow, no sparsity.
- **Team B Output-Stationary** is a V2 extension.
- **Team A sparsity** is a V2 extension. The V1 PE has the `zero_skip`
  port, but its functional connection to the Sparsity Manager is a V2
  Team A concern.

### Previously Undecided — Now Established

| Parameter | Previous Status | Current Status |
|-----------|----------------|----------------|
| PE count | Undecided | **Established:** 64 (8×8 grid) per roadmap §3.2 |
| V1 dataflow mode | Undecided | **Established:** Weight-Stationary per roadmap §3.2, §4 |
| Numerical format | Undecided | **Established:** Signed 8-bit two's-complement, scale 1/256 (Decision 1, 2026-08-10) |
| Accumulator width | Undecided | **Established:** 32-bit signed two's-complement (Decision 2, 2026-08-10) |
| Overflow behaviour | Undecided | **Established:** Overflow impossible by construction; no handling logic (Decision 3, 2026-08-10) |
| Result-read protocol | Undecided | **Established:** result_request strobe, read-without-clear, separate accum_clear (Decision 4, 2026-08-10) |
| Reset semantics | Undecided | **Established:** synchronous, active-high (rst) (Decision 5, 2026-08-10) |
| Pipeline architecture | Undecided | **Established:** registered MAC (MREG+PREG), combinational forwarding, fixed depth (Decision 6, 2026-08-10) |
| PE forwarding direction | Undecided | **Established:** activations shift across rows per roadmap §3.2 |

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
9. Synthesis/DSP inference and array integration — **pending**
10. Team A / Team B V2 extensions — **pending**

## Next Planned Work

1. ~~**Decision 1: numerical format.**~~ **COMPLETE (2026-08-10).**
2. ~~**Decisions 2–4: accumulator width, overflow, result-read.**~~ **COMPLETE (2026-08-10).**
3. ~~**Decisions 5–6: reset semantics, pipeline architecture.**~~ **COMPLETE (2026-08-10).**
4. ~~**V1 PE RTL implementation.**~~ **COMPLETE** — `rtl/common/pe.sv` implemented.
5. ~~**V1 PE unit testbench.**~~ **COMPLETE** — `sim/tb_pe.sv`; Vivado 2023.1
   simulation passes 55/55.
6. Vivado synthesis and DSP48E2 inference check on the target device.
7. Systolic array (`systolic_array.sv`) integration and array-level verification.

## Research Discipline

- Distinguish roadmap requirements from external research facts.
- Distinguish research facts from engineering recommendations.
- Do not allow common CNN accelerator practice to become a project
  requirement without an explicit team decision.
- Do not over-engineer infrastructure before it is needed; prioritise
  accelerator architecture and RTL development.
