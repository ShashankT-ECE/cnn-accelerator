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

## Where the project stands (updated 2026-09-30, after OC-3 / limits audit)

Update this section whenever a step finishes. It is a pointer summary. The
records of truth are `v2/docs/DECISIONS.md` (decisions D1–D18, open
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
.venv/bin/python v2/scripts/verification_stats.py   # verification_stats.csv (after all producers)
.venv/bin/python v2/paper/scripts/make_all.py       # paper tables/figures -> v2/paper/generated/
.venv/bin/python -m pytest v2/board/tests v2/paper/tests v2/analysis v2/scripts/tests
# board (on the KV260, cd ~/gos): sudo -E ./session.sh 1|2|3|all [--budget-min N]   (see v2/board/README.md)
```
Full step-by-step reproduction: `v2/REPRODUCE.md`. Analyses (`v2/analysis/*.py`) run
inside `regen_results.sh`.

Run only one Vivado job at a time, through `vivado_guard.sh` (the resource
rule in `v2/CLAUDE.md`). xsim may run alongside it. The board workflow
(`make_board_data.py` → `deploy.sh <ip>` → Sessions 1–3) is in
`v2/board/README.md`.

### Done (V2 steps, each committed on v2-dev)

- **1–4.5:** frozen INT8 references (model accuracies of record: LeNet-5 INT8 98.79%,
  CIFAR-10 r2 INT8 78.52%), requant B=32 bit-exact, full RTL, unit/core/fuzz/checker/
  network/back-to-back/netlist verification, C_START=3 (D13), static checks (D15).
- **5:** D16 per-row provenance, D17 CSR staging (cycle-neutral), board software.
- **Build audit + 300 MHz (2026-09-28, D18):** the parallel 200/250/300 runs in
  `~/gos_build_wt` (b5fbcd6) were killed in synthesis — INCOMPLETE, not used. Builds
  of record are fd880d43 in `~/gos-build` (copied to `v2/vivado/out/gos_<MHz>/`, no proj/).
- **Steps 9/10 (2026-09-28):** code 3527f97; interim results 63261ad; **freeze results
  8b8acce** — every producer re-run clean, `check_results.py` ALL CLEAN, untruncated
  netlist synth scan PASS, netlist sim now reports logits (2/2, 3/3). New: `v2/paper/`
  (make_all.py, tables/figures from CSVs, board placeholders), `v2/analysis/`
  (utilization, V1-vs-V2 ablation, 16x16 "projected" — not C1), `v2/board/session.sh`
  (resumable sessions, budget, pre-flight), `v2/board/power_log.py` (INA260 SOM rail =
  primary power, label "SOM-rail power (INA260)"), `v2/REPRODUCE.md`,
  `verification_stats.csv`.

- **Reviewer-gap work (2026-09-29, laptop only, all pushed):**
  - Full-dataset RTL sim `v2/results/rtl_full10k.csv`: 10,000/10,000 images per net
    bit-exact logits, prediction match, cycle-exact (Verilator 5.028 in `~/tools`,
    cross-validated vs xsim before every run; `v2/scripts/run_full10k.sh`, ~7 min on 8 shards).
  - Board software: fast host path (`--host-path fast`, gated by the `s1.fast` bring-up
    check), B1 control phase, energy two ways (E_sys, E_comp, duty cycle), VCC_SOM =
    SOM +5 V input per UG1089 v1.3 (per-rail split unconfirmed), external meter removed
    (INA260 only), pre-flight (governor/pinning/apt guard/AMS temp, `paper_grade`), measured
    PL clock (`exp_fclk_cal.py`), median + 95% CI, interleaving, 3-session repeatability,
    soak, per-layer spread, B2 cycle identity + P = P_static + k*f fit, 300->250 MHz
    auto-fallback. Step priority: A3, A1, latency, baselines, energy, sweep, soak, A4.
  - DPU baseline prep (`v2/dpu/`): Vitis AI 2.5.0 matched to the prebuilt pynq-dpu 2.5
    KV260 overlay (fingerprint 0x101000016010407), xmodels in gitignored `v2/dpu/build/`
    (Docker image `xilinx/vitis-ai-cpu:2.5.0` loaded locally), `dpu_session.py`;
    model-level vai_q accuracy in `v2/results/dpu_model_accuracy.csv`.
  - Fixes: CIFAR per-layer cycle list trimmed to 4 layers; D2/D3 r2 logit bound.
  - Worktrees: `~/gos-build` (fd880d4, bitstream builds of record + proj/), `~/gos-10k`
    (full-10k run dir incl. shard logs), `~/gos_build_wt` (killed b5fbcd6 runs, unused).

- **OC-3 + limits audit (2026-09-30, pushed 626a341):** RTL sim (Verilator + xsim) KW ≤ 9 exact, KW ≥ 10
  wrong → host contract `gos_pack.assert_host_limits` (MAX_KW = 9) + host asserts for every limit the
  checker does not enforce (`v2/docs/LIMITS.md`, `run_limits.sh` → `limits_rtl.csv`, 49 cases). D16 amended:
  rows checked only against their producer's sources (Vivado rows: v2/rtl + v2/vivado). All v2/model
  producers re-run @ e9cbb88, results identical, `check_results.py` ALL CLEAN. Post-ROCS checker rules: DECISIONS TODO.

### Final implementation table (post-impl, build fd880d43, from `v2/results/impl_gos.csv`)

| Clock | Complete? | Current RTL? | WNS (ns) | WHS (ns) | CW | LUT / FF / DSP / BRAM36+18 | bit sha256 | Verdict |
|---|---|---|---|---|---|---|---|---|
| 200 | yes | yes | +0.757 | +0.010 | 0 | 16,513 / 22,977 / 80 / 50+1 | 87fcae36… | VALID |
| 250 | yes | yes | +0.207 | +0.010 | 0 | 16,538 / 22,985 / 80 / 50+1 | 7b2dac60… | VALID (fallback) |
| 300 | yes | yes | +0.122 | +0.011 | 0 | 16,563 / 23,528 / 80 / 50+1 | 2d40e2c8… | VALID (performance) |

Superseded: `out/gos_200_c88e71a0` (STALE, pre-D17 CSR), `out/shell_c6ff5d96` (old shell).
`synth_scan_pass=False` on these rows = truncated BD synth log only (D18-3); the
untruncated netlist scan passes.

### Pending (in order)

1. **KV260 board sessions** (nothing measured on the board yet). Laptop: commit-clean
   tree → `make_board_data.py` → `v2/board/deploy.sh <ip>` (ships gos_300 + gos_250) and
   `v2/dpu/deploy_dpu.sh <ip>`. Board (`cd ~/gos`, tmux, stop packagekit/unattended-upgrades):
   `power_log.py --list-sensors` / `--sample-only` first, then `sudo -E ./session.sh 1`
   (smoke at 300, auto-fallback 250), `2`, `3`; repeat with `--session-index 2|3` for
   repeatability; `dpu_session.py` for the DPU baseline.
2. Copy results back (`rsync ... v2/results/`), `check_results.py`, separate results commit,
   `aggregate_sessions.py`, `v2/paper/scripts/make_all.py`.
3. Paper text (ROCS 2026 short paper, 4 pages + refs, IEEE 2-col, deadline 2026-10-09 AoE).
4. Open: confirm INA260 update rate, VCC_SOM per-rail coverage (carrier schematic U14),
   DPU input fix_point at bring-up; optional RTL-sim per-layer spread from `~/gos-10k`
   shard logs (`exp_layer_spread.py --rtl-sim`, dry-run output only); C1/C2 optional.
