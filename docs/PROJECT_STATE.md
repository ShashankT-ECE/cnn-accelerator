# CNN Accelerator — Project State

## Current Status

The repository baseline and shared development environment documentation are
established.

The project targets the AMD Kria KV260 / XCK26 using SystemVerilog and
Vivado ML 2023.1.

The repository is hosted on GitHub and is the shared source of truth for the
development team.

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

The shared development setup is documented in:

`docs/DEVELOPMENT_SETUP.md`

The Research skill is included under:

`.claude/skills/research/`

Generated build and local tool data must remain outside the committed source
where specified by `.gitignore`.

## Architecture Status

The architecture/overview PDF is a high-level roadmap and initial reference.
It is not the authoritative source for detailed implementation decisions.

Detailed hardware decisions are still being established.

Do not treat assumptions from the roadmap as finalized requirements.

## Decisions

No finalized project-specific decisions are recorded here yet for:

- fixed-point representation
- accumulator widths
- PE count
- AXI register map
- AXI interface details
- memory layout
- dataflow configuration
- target clock frequency
- resource utilization targets
- detailed module interfaces

When these decisions are agreed, record them here or move detailed contracts
into dedicated files under `docs/specs/`.

## Open Questions

Record unresolved architectural or implementation questions here as they
arise.

## Milestones

No formal hardware implementation milestone has been completed yet.

Update this section as major project milestones are reached.
