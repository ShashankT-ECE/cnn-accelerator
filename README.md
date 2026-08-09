# CNN Accelerator

A CNN hardware accelerator targeting the AMD Kria KV260 / XCK26 FPGA.
The design is developed in SystemVerilog and built with Vivado ML 2023.1.

## Repository

- `rtl/` — synthesizable SystemVerilog RTL
- `sim/` — simulation sources and testbenches
- `python/` — Python reference/model code
- `scripts/` — project scripts
- `data/` — project data
- `docs/` — project documentation
- `build/` — generated build artifacts

## Development Setup

See [docs/DEVELOPMENT_SETUP.md](docs/DEVELOPMENT_SETUP.md) for the
reproducible development environment (Vivado ML 2023.1, Python environment,
and tooling).

The repository also contains the shared Claude Code project configuration
(`CLAUDE.md` and `.claude/`).

## Current Status

The shared development environment and project documentation baseline are
established. Detailed accelerator implementation decisions are still being
developed.
