# V2 rules (take precedence under v2/)

- **Scope:** edit only files under `v2/`. Legacy files outside `v2/` are read-only references. Never modify them unless the user explicitly says so.
- **Branch:** `v2-dev`. Commit after every step that passes its checks, with a message naming the step (e.g. "V2 step 2.1: ...").
- **The architecture is FROZEN** in `docs/ARCH_SPEC.md`. Do not redesign. If an implementation or synthesis problem conflicts with the spec, stop, append the evidence to `docs/DECISIONS.md` under "Open conflicts", and report to the user.
- **RTL:** synthesizable SystemVerilog, module prefix `gos_`. Single clock `clk`, synchronous active-high `rst` inside the core. No runtime multipliers in address generation (counters/incremental adds only). Data and control travel together via the token pipeline (valid/first/last/ids/masks); no module infers data timing by counting cycles.
- Every RTL module has a self-checking xsim testbench in `v2/tb/` that prints PASS/FAIL and exits nonzero on failure. Simulations run in `v2/build/sim/<tb_name>/`, never in the repo root.
- Golden data comes only from `v2/model` (Python, `.venv`). Vectors are hex files loaded with `$readmemh`.
- **Honesty:** never type a result number by hand. All results come from scripts that write `v2/results/*.csv` with metadata. Never present simulation or model numbers as hardware measurements. Label every number: model / RTL sim / post-implementation / measured on KV260.
  Results CSVs are generated only from a clean, committed tree (git_dirty=False). Workflow: commit code → run result scripts → commit CSVs in a separate commit. Rows with git_dirty=True are invalid for the paper.
- Graphify is not required for `v2/` work.
- Use parallel subagents for independent work (e.g. separate leaf modules and their testbenches), then integrate and run all checks in the main session. Subagents never launch Vivado (see the resource rule).
- **Resource rule (permanent, after the 2026-09-24 OOM freeze):** at most ONE Vivado job at a time (synthesis, OOC, netlist or implementation — never in parallel, never from concurrent subagents); every Vivado Tcl sets `set_param general.maxThreads 8` and launches runs with `-jobs 1`; before each Vivado run print `free -h` and do not start if available memory < 8 GB — `scripts/vivado_guard.sh` does both checks and every Vivado entry script calls it. xsim runs (xvlog/xelab/xsim) are not Vivado jobs.
