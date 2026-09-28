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
```

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

## V2 accelerator work

All V2 work lives in `v2/` on branch `v2-dev`. For any file under `v2/`,
`v2/CLAUDE.md` takes precedence over this file. `v2/` is an approved top-level
directory. Legacy files (rtl/, sim/, python/, data/, docs/, scripts/,
software/) are read-only references for V2 work.

## Where the project stands (snapshot 2026-09-28)

Update this section whenever a step finishes. It is a pointer summary. The
records of truth are `v2/docs/DECISIONS.md` (decisions D1–D17, open
conflicts, TODO) and the script-generated `v2/results/*.csv`. Every number
below is copied from those sources and labelled model / RTL sim /
post-impl. **None of them are KV260 measurements yet.**

### What the project is

- **V1 / legacy (frozen):** `rtl/`, `sim/`, `python/`, `docs/` on `main`,
  preserved on branch `v1-snapshot` / tag `v1-baseline`. This work covered the
  PE, the systolic array, the Phase 2 OS/WS reconfigurable research
  accelerator, Phase 3 AXI top, and the LeNet-5/CIFAR-10 mapping and sparsity
  studies. Those studies concluded that output-stationary wins, and that
  motivated V2. Status is in `docs/PROJECT_STATE.md`.
- **V2 (active, branch `v2-dev`, directory `v2/`):** a generalized INT8 8×8
  output-stationary accelerator (`gos_` modules) that runs LeNet-5 and
  CIFAR-10 end to end, with all weights resident in BRAM. The thesis is
  bit-exact inference plus cycle-exact agreement between model, RTL sim and
  board. The architecture is frozen in `v2/docs/ARCH_SPEC.md`, the formats and
  register map are in `v2/docs/FORMATS.md`, and the experiment plan (A1–A6,
  B1–B3) is in `v2/docs/EXPERIMENTS.md`.

### V2 layout

`v2/model` Python golden/pack/cycle model (`.venv`) · `v2/rtl` gos_ RTL +
`gos_top_wrapper.v` · `v2/tb` self-checking xsim TBs · `v2/vectors` hex
vectors · `v2/scripts` sim/collect/check scripts · `v2/vivado` Tcl, block
design, `build_gos.sh` (outputs in `v2/vivado/out/gos_<MHz>/`, gitignored) ·
`v2/board` KV260 PYNQ driver + experiment scripts + CPU baselines ·
`v2/results` CSVs (`dryrun/` is gitignored and never counts as paper data).

### How to build and test V2 (repo root, clean committed tree for paper rows)

```bash
source ~/Xilinx/Vivado/2023.1/settings64.sh
.venv/bin/python -m pytest v2/model v2/board/tests   # model + driver unit tests
v2/scripts/regen_results.sh          # model CSVs (refuses a dirty tree)
v2/scripts/run_unit_all.sh           # unit TBs      -> unit_tb.csv
v2/scripts/run_core.sh               # core/net/fuzz -> rtl_cycles/rtl_network/rtl_checker.csv
v2/scripts/run_netlist_sim.sh        # post-synth netlist sim (Vivado job) -> rtl_netlist.csv
v2/scripts/ooc_all.sh                # leaf OOC synth (Vivado)              -> ooc_synth.csv
v2/vivado/build_gos.sh <200|250|300> # full bitstream (Vivado)              -> impl_gos.csv via impl_collect.py
.venv/bin/python v2/scripts/check_results.py   # provenance check (read-only)
```

Run only one Vivado job at a time, through `vivado_guard.sh` (the resource
rule in `v2/CLAUDE.md`). xsim may run alongside it. The board workflow
(`make_board_data.py` → `deploy.sh <ip>` → Sessions 1–3) is in
`v2/board/README.md`.

### Done (V2 steps, each committed on v2-dev)

- **1–2.2:** scaffold, frozen spec, INT8 references. CIFAR uses the retrained
  r2, and the accuracies of record (model) are LeNet-5 INT8 98.79% and CIFAR-10
  INT8 78.52%. Requant B=32 is bit-exact to float64 (0 mismatches). The golden
  model, formats, cycle model and vectors are done.
- **3:** leaf RTL + unit TBs all PASS; OOC at 5 ns met.
- **4:** core/ctrl/cfg_check/CSR/top. RTL-sim cycles equal the model exactly
  (C_PIPE=29, C_DONE=0).
- **4.5:** checker pipelined, so C_START=3 (D13). Job totals from RTL sim
  match the model: LeNet 16,436 and CIFAR 104,247 cycles. RTL-sim logits are
  bit-exact on 10 LeNet + 10 CIFAR images (`rtl_network.csv`). Back-to-back
  100 jobs PASS. Post-synth netlist sim PASS (reduced scope, D15-7). Static
  checks are clean.
- **4.5 impl (post-impl):** the 200 MHz build c88e71a0 met timing (WNS +0.291).
- **5 part 0/1 (fd880d4):** D16 per-row provenance rule; D17 CSR DESC staging
  register (cycle-neutral) for higher clocks.
- **5 part 2 (a660a04, d8fedfd):** board software (`gos_driver` with
  Pynq/Model backends, data package, A1–A4/B1–B3 scripts, smoke test,
  `run_all.sh`, `deploy.sh`) + CPU baselines. Laptop dry runs
  (`source=dryrun_model`) pass end to end on 10k images per net. These are a
  software check only, not measurements.

### State of the working tree and builds (as of 2026-09-28)

- `v2-dev` is **3 commits ahead of origin** (fd880d4, a660a04, d8fedfd are
  not pushed).
- **12 regenerated results CSVs are uncommitted** (git_dirty=False rows from
  a660a04 / fd880d4). They include `impl_gos.csv` with two post-impl rows
  (build fd880d43, 0 critical warnings each):
  - **200 MHz:** WNS +0.757 ns; 16,513 LUT / 22,977 FF / 80 DSP / 50 RAMB36
    + 1 RAMB18.
  - **250 MHz:** WNS +0.207 ns; 16,538 LUT / 22,985 FF.
- Bitstreams on disk: `v2/vivado/out/gos_200/` and `gos_250/` (build
  fd880d43), plus the older `gos_200_c88e71a0/`. **No 300 MHz build exists.**
- `check_results.py` currently FAILS. The stale rows are `impl_shell.csv`,
  `ooc_synth.csv` (from c6ff5d9) and `rtl_netlist.csv` (from fc13c08). All
  three predate the D17 CSR change and must be re-run before the final
  freeze.

### Pending (in order)

1. Commit the 12 regenerated CSVs as a separate results commit, then push
   `v2-dev`. This needs the user's go-ahead.
2. Decide on the performance variant (D14). 250 MHz passes the D14 rule.
   300 MHz has not been tried; it runs only on user request.
3. Re-run the stale producers (netlist sim, OOC, shell) so that
   `check_results.py` passes.
4. **Board bring-up on the KV260** (never done yet): `make_board_data.py` →
   `deploy.sh` → Session 1 (`test_shell.py`, `test_core_smoke.py`). Confirm
   the assumptions in `v2/board/README.md`: `--write-mode slice`, runtime
   fclk0 change, INA260 path.
5. Session 2 (A1–A4, B3, CPU A5 on the Cortex-A53) and Session 3 (B1 power
   with inline meter, B2 clock sweep).
6. Open tool: a script that joins the meter log to the B1/B2 windows and
   computes ΔP and energy per inference.
7. Final results freeze from one commit (DECISIONS TODO), then the paper
   (`v2/paper/`). C1 (16×16) and C2 are optional.
