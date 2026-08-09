# CNN Accelerator — Project State

## Current Status

The repository baseline and shared development environment documentation are
established.

The project targets the AMD Kria KV260 / XCK26 using SystemVerilog and
Vivado ML 2023.1.

The repository is hosted on GitHub and is the shared source of truth for the
development team.

**Current engineering phase:** PE specification complete; architectural
decisions pending; no RTL has been written.

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

The roadmap uses **signed 8-bit Q8** as its V1 working assumption (weight
extraction script §5.3, accumulator guidance §12). Q8 has **not** been
formally adopted as the project's final numerical format. CLAUDE.md and this
document both list numerical format as undecided.

Do not silently convert Q8 into an INT8 or finalized-format requirement.

### Unresolved PE Decisions

The following decisions block `rtl/common/pe.sv` implementation (see
`docs/specs/PE_SPEC.md` §11–§12 for full context):

1. Numerical format finalisation
2. Accumulator width
3. Product representation
4. Overflow behaviour (or explicit waiver)
5. Result-read protocol (FSM↔PE handshake)
6. Reset semantics (polarity, synchronicity)
7. Pipeline depth / cycle-level behaviour

These are **not** finalized:
- accumulator width (roadmap provides guidance, not a fixed width)
- pipeline depth
- clock frequency target
- exact RTL signal names
- `wgt_out` port
- V2 dataflow-controller extensibility hooks
- overflow policy
- reset polarity/synchronicity
- result-read handshake mechanism

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
| PE forwarding direction | Undecided | **Established:** activations shift across rows per roadmap §3.2 |

## Open Questions

Record unresolved architectural or implementation questions here as they
arise.

## Milestones

1. Repository baseline and dev environment documentation — **complete**
2. PE specification (`docs/specs/PE_SPEC.md`) — **complete** (draft, reviewed)
3. Numerical format decision — **next**
4. Remaining PE blocking decisions — **pending**
5. `rtl/common/pe.sv` implementation — **blocked** on decisions 3–4
6. V1 shared baseline complete — **pending**
7. Team A / Team B V2 extensions — **pending**

## Next Planned Work

1. **Decision 1: numerical format.** Research the Q8 / numerical-format
   decision using the architecture roadmap, AMD DSP48E2 documentation
   (UG579), and relevant CNN accelerator / fixed-point research. Evaluate
   alternatives and tradeoffs before formally updating this document.
2. Resolve the remaining PE blocking decisions one at a time, recording
   each in this document.
3. Only after the RTL entry criteria in `docs/specs/PE_SPEC.md` §12 are
   satisfied should `rtl/common/pe.sv` be written.

## Research Discipline

- Distinguish roadmap requirements from external research facts.
- Distinguish research facts from engineering recommendations.
- Do not allow common CNN accelerator practice to become a project
  requirement without an explicit team decision.
- Do not over-engineer infrastructure before it is needed; prioritise
  accelerator architecture and RTL development.
