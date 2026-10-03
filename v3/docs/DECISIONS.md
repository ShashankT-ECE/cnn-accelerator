# V3 decisions

Format as in V2: `## Dn — title (date, phase, who decided)`; outcomes and amendments are separate
headings; open conflicts at the end as `### OC-n`. Numbers quoted here are copied from
`v3/results/*.csv` (with the CSV named) or from `v2/` records (named), never typed from memory.

## D1 — V3 scope, branch and rules (2026-10-03, Phase 0, user decision)

- V3 is the college major project. It lives in `v3/` on branch `v3-dev` (worktree `~/gos-v3`, branched from
  `v2-dev` @ 28dd2ad). `v2/` and its tags are never modified.
- Level definitions (user, verbatim summary): A = better CNNs (A1 SAME padding, stride 2, residual add, GAP,
  URAM weights → ResNet-20 CIFAR-10 bit-exact + cycle-exact on board; A2 depthwise → small MobileNet-style
  CIFAR-10 net). B = reconfigurable dataflow (B1 split-K; B2 depthwise mode; B3 per-layer mode choice by the
  compiler using the cycle model; B4 weight-stationary mode, stretch, dropped if it threatens timing).
  C = sparsity skipping (C1 structured weight sparsity in array-width-OC × K-step blocks, static schedule;
  C2 activation skipping of all-zero broadcast K-steps, data-dependent; C3 accuracy/sparsity/speed curve on
  board). Order A1 → B1 → C1 → A2/B2 → B3 → C2 → B4.
- Array size and INT8 DSP packing are Phase 0 decisions (from the DSE), not levels.
- Phase 0 = decisions only; no design RTL before the user approves `V3_SPEC.md`.
- V2 rules carry over (`v3/CLAUDE.md`): CSV-only numbers, clean-tree provenance, one Vivado job at a time.

## D2 — Clock targets follow the board PLL (2026-10-03, Phase 0, user decision)

- The KV260 boot image derives the PL clock from a 999.99 MHz IOPLL; 249.9975 MHz and 333.33 MHz are
  reachable, 300 MHz is not (V2 D19, measured in V2 Session 1).
- V3 feasibility OOC runs at 4.000 ns and 3.000 ns. Full builds request the PLL-reachable frequency and
  the board driver verifies the read-back (V2 D19/D20 carry over).

## D3 — Results provenance (2026-10-03, Phase 0)

- `v3/model/common.py` (copied from V2) writes the metadata columns on every row; `v3/scripts/check_results.py`
  (adapted from V2 D16) requires every CSV to have a producer entry and every row to be clean with
  unchanged producer sources.
- Training on an external GPU (user decision 2026-10-03): scripts committed first; each job's CSV records the
  script commit, torch/CUDA versions, GPU name and checkpoint sha256; local CPU smoke runs go only to
  `v3/results/dryrun/`.
- Golden model vs PyTorch fake-quant agreement is checked on all 10,000 CIFAR-10 test images per net
  (user requirement 2026-10-03).

## TODO

- Phase 0 workstreams WS1–WS6 (plan 2026-10-03); V3_SPEC.md; decisions D4+ below after the report.

## Open decisions (Phase 0, to be resolved with data, then approved by the user)

- Array R×C and INT8 DSP packing; row mapping (x-only vs flattened x·y); stride-2 handling under banked
  ACT reads; residual-add placement; ResNet-20 shortcut option A/B; MobileNet-style net widths;
  PTQ→QAT threshold; descriptor word count and storage; B4 keep/drop; TARGETS values; RTL module prefix.

## Open conflicts

(none)
