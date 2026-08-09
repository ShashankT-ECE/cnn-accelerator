# CNN Accelerator — Claude Code Project Instructions

## Project

This project develops a CNN hardware accelerator targeting the AMD Kria KV260.

- Board: AMD Kria KV260
- FPGA: XCK26
- Device: `xck26-sfvc784-2LV-c`
- HDL: SystemVerilog
- FPGA toolchain: Vivado ML 2023.1
- Host OS: Ubuntu 22.04 LTS 64-bit

## Repository Structure

Use the existing project directories:

- `rtl/` — synthesizable SystemVerilog RTL
- `sim/` — simulation sources and testbenches
- `python/` — Python reference/model code
- `scripts/` — project scripts
- `data/` — project data
- `docs/` — project documentation
- `build/` — generated build artifacts

Do not create unnecessary new top-level directories.

## Architecture and Design Decisions

The architecture/overview PDF is a high-level roadmap and initial reference.
It is not automatically authoritative for detailed implementation decisions.

Do not invent:

- hardware interfaces
- register maps
- numeric formats
- PE counts
- memory layouts
- timing targets
- resource targets
- other architectural parameters

When something is ambiguous:

1. Check the current project documentation.
2. Check the relevant technical specification if one exists.
3. If it is still unresolved, clearly identify it as an open design decision.
4. Do not silently turn an assumption into a requirement.

When an important design decision is agreed, document it in the appropriate
project documentation or specification.

## Vivado and Simulation

Use Vivado ML 2023.1 for FPGA development.

Target device:

`xck26-sfvc784-2LV-c`

Use the Vivado command-line simulator:

- `xvlog`
- `xelab`
- `xsim`

Do not substitute another simulator for the project's Vivado simulation flow.

Normal simulation should use the command line rather than requiring the
Vivado GUI.

Before using Vivado tools from a new shell:

```bash
source ~/Xilinx/Vivado/2023.1/settings64.sh
SystemVerilog Coding Rules

For synthesizable RTL:

Use always_ff for sequential logic.
Use always_comb for combinational logic.
Do not use initial blocks in synthesizable modules.
Do not use # delays in synthesizable modules.
Do not use simulation-only constructs in synthesizable datapaths.
Use explicit port directions and types.
Keep signal widths and signedness explicit.
Avoid implicit truncation or extension.
Keep reset behavior explicit and consistent with the module specification.

These are project coding conventions unless an explicit documented
requirement requires otherwise.

Python Reference Models

Keep Python reference/model code consistent with the implemented hardware.

Do not assume a quantization format, numeric representation, saturation rule,
accumulator width, or other numerical convention unless it has been specified.

When RTL and Python behavior are compared, make arithmetic and edge-case
behavior explicit and testable.

Development Workflow

For RTL work:

Understand the current requirements and design decisions.
Identify the exact module/interface being implemented.
Define interfaces and parameters before writing substantial RTL.
Write synthesizable SystemVerilog.
Create or update the appropriate testbench.
Run Vivado simulation.
Check functional correctness.
Proceed to synthesis/implementation after correctness is established.
Optimize only after correctness is established.

For Python/model work, keep the reference implementation aligned with the
hardware behavior and verify important numerical behavior with tests.

Git and Collaboration

GitHub is the shared source of truth.

Use feature branches for development rather than directly modifying main
unless explicitly agreed by the team.

Before destructive Git operations such as reset, rebase, force push, deleting
branches, or rewriting history:

explain the consequence
verify the user's intention

Never expose or request passwords, tokens, API keys, or other secrets.

Communication and Execution

Work practically and sequentially.

When performing setup, debugging, or development tasks:

give one actionable step at a time
verify the result before proceeding
do not repeat dependencies that are already verified
do not make unnecessary version changes
diagnose failures before moving forward

Do not over-engineer project infrastructure before it is needed.

Documentation

CLAUDE.md contains stable project instructions and development rules.

docs/PROJECT_STATE.md contains current project status, agreed decisions,
milestones, ownership, and open questions.

Detailed technical contracts should live in dedicated specification files
under docs/specs/ when they are established.

Do not duplicate detailed specifications unnecessarily.

Graphify

This project has a knowledge graph at graphify-out/ with god nodes,
community structure, and cross-file relationships.

Rules:

For codebase questions, first run graphify query "<question>" when
graphify-out/graph.json exists.
Use graphify path "<A>" "<B>" for relationships.
Use graphify explain "<concept>" for focused concepts.
If graphify-out/wiki/index.md exists, use it for broad navigation instead
of raw source browsing.
Read graphify-out/GRAPH_REPORT.md only for broad architecture review or
when query/path/explain do not surface enough context.
After modifying code, run graphify update . to keep the graph current.

Generated Graphify data is local project state and must not be committed.
