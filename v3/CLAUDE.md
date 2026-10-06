# V3 rules (take precedence under v3/)

V3 = college major project: a timing-predictable INT8 CNN accelerator on the KV260 for ResNet-20 and
a MobileNet-style CIFAR-10 net (milestones M1 -> M5). **Source of truth for scope and direction:
`docs/V3_PLAN.md`.** Records of truth: `docs/DECISIONS.md` (D4–D13 record the plan; the D1 level scheme
and the D2 3.000 ns target are superseded) and the script-generated `results/*.csv`. Detailed contracts
go in `docs/V3_SPEC.md`.

- **Scope:** edit only files under `v3/` (plus the V3 pointer section in the root CLAUDE.md).
  `v2/`, its tags (`v2-paper-freeze`, `v2-paper-final`) and the legacy dirs are read-only
  references. V2 code that V3 needs is COPIED into `v3/` with a header `copied from v2/<path> @ <commit>`;
  nothing in `v3/` imports `v2/` at runtime.
- **Branch / worktree:** `v3-dev`, worked in the worktree `~/gos-v3` (the main checkout
  `~/cnn-accelerator` stays on `v2-dev`). Commit after every step that passes its checks
  ("V3 phase 0: ...", later "V3 A1: ..."). Subagents never commit; the main session integrates and commits.
- **Phase 0 = decisions only.** No design RTL until the user approves `docs/V3_SPEC.md`. Exploratory RTL
  for feasibility lives only in `v3/feas/`, is labelled exploratory, and is never reused unreviewed.
- **Do not invent** interfaces, formats, sizes, targets: unresolved choices are written as open decisions
  in DECISIONS.md and presented to the user; agreed ones are recorded there (V3 Dn numbering).
- **Clock (V3 D2):** the KV260 PL clock comes from the 999.99 MHz IOPLL of the boot image: 249.9975 MHz
  (/4) and 333.33 MHz (/3) are reachable, 300 MHz is not (V2 D19). **Design target: 4.000 ns / 250 MHz
  (V3 D5).** 3.000 ns / 333.33 MHz is a feasibility data point only, not a design option.
  Board clock settings must request the PLL-reachable value and verify the read-back (V2 D19/D20).
- **RTL (from Phase 1):** synthesizable SystemVerilog, single clock `clk`, synchronous active-high `rst`,
  always_ff/always_comb, no initial/# in synthesizable code, explicit widths/signedness, data and control
  travel together (no cycle-counting inference). Module prefix: decided in V3_SPEC.
- Every RTL module (feasibility included) has a self-checking xsim testbench that prints PASS/FAIL and exits
  nonzero on failure; sims run under `v3/build/`, never in the repo root.
- Golden data comes only from `v3/model` (Python, repo `.venv`).
- **Honesty:** never type a result number by hand. Results come only from scripts writing `v3/results/*.csv`
  with the metadata columns of `v3/model/common.py`. Label every number: literature / model / estimate /
  RTL sim / OOC synth / post-impl / measured on KV260. CSVs only from a clean, committed tree
  (git_dirty=False): commit code -> run producers -> commit CSVs separately. `v3/scripts/check_results.py`
  must report ALL CLEAN before a results commit.
- **External GPU training** (Kaggle/Colab, user-run): scripts are committed first; each job writes a CSV with
  the script git commit, torch/CUDA versions, GPU name and checkpoint sha256. Local CPU smoke runs
  (1-2 epochs) only verify scripts and write to `v3/results/dryrun/` (gitignored, never data).
- **Literature numbers** live in `v3/results/literature.csv`, each with DOI + table/page and a verified flag.
- **Resource rule (permanent):** at most ONE Vivado job at a time; every Vivado Tcl sets
  `set_param general.maxThreads 8` and `-jobs 1`; every Vivado entry script calls
  `v3/scripts/vivado_guard.sh` (refuses if another vivado runs or < 8 GB available). The Vitis AI Docker
  container also waits for the guard. Subagents never launch Vivado. xsim is not a Vivado job.
- Graphify is not required for `v3/` work.
