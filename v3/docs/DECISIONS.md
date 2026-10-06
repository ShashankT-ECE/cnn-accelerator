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

## D4 — Thesis (2026-10-06, Phase 0, user decision)

- V3 is a timing-predictable CNN accelerator on the KV260. The latency of an inference is known exactly
  before it runs (model = RTL = board, cycle-exact). Reconfigurable dataflow and sparsity speed it up
  without losing that guarantee. `docs/V3_PLAN.md` is the source of truth for scope and direction.

## D5 — Milestones M1–M5 (2026-10-06, Phase 0, user decision)

- Each milestone is fully verified (D8) before the next one starts.
- **M1 Foundation:** R × C INT8 output-stationary array (16 × 16 expected, to be confirmed by the DSE),
  250 MHz, SAME padding, stride 2, residual add, global average pooling, weights in URAM; ResNet-20 CIFAR-10.
- **M2:** split-K mode + a compiler that picks the mode per layer and emits the exact predicted cycles.
- **M3:** structured weight sparsity (block = C OCs × 1 K-step, static skip; latency stays exact).
- **M4:** depthwise mode + a small MobileNet-style CIFAR-10 net.
- **M5:** activation zero-step skipping; the dense schedule is the guaranteed worst-case bound.
- Clock: 250 MHz is the design target. 333.33 MHz (the other PLL-reachable value, D2) is only a
  feasibility data point, not a design option.
- Supersedes the D1 level scheme and order (see "D1/D2 amendment" below).

## D6 — Dropped and conditional features (2026-10-06, Phase 0, user decision)

- Dropped: weight-stationary mode (former level B4), DRAM / ImageNet, batching, 300 MHz.
- Conditional: INT8 DSP packing, only if proven bit-exact and timing-clean in Phase 0.

## D7 — One bitstream, per-layer mode descriptor (2026-10-06, Phase 0, user decision)

- One final bitstream supports all modes, selected per layer by a descriptor, so one board session can
  measure the full ablation.
- The descriptor format reserves fields for every milestone (M1–M5) from day one. The field layout is
  still an open decision (V3_SPEC).

## D8 — Verification per milestone (2026-10-06, Phase 0, user decision)

- On the laptop, for every milestone: cycle model, bit-exact golden model, full 10,000-image Verilator
  sims (bit-exact and cycle-exact), random shapes, post-implementation timing, board-script dry-run.

## D9 — Three board sessions (2026-10-06, Phase 0, user decision)

- The board is not ours, so there are three planned sessions:
  - S1 at the end of Phase 0: DPU + ORT baselines, V3 shell smoke test.
  - S2 after M2: ResNet-20 dense + split-K.
  - S3 after M5: full ablation, MobileNet, sparsity curves, 3-session repeatability.

## D10 — Evaluation (2026-10-06, Phase 0, user decision)

- Predicted = measured latency per image; per-feature speedup ablation; accuracy vs sparsity vs speed;
  energy per image (SOM-rail power); vs DPU and ORT on the same nets, including latency jitter;
  vs literature.

## D11 — Training (2026-10-06, Phase 0, user decision)

- GPU jobs are run by the user on Kaggle; CPU smoke tests run locally (`v3/results/dryrun/`).
  The D3 provenance rules for training jobs apply.

## D12 — Rules inherited from V2 (2026-10-06, Phase 0, user decision)

- Numbers only from scripts/CSVs; labels (model / RTL sim / post-impl / measured on KV260); one Vivado
  job at a time; self-checking TBs; commit after each finished piece. Details in `v3/CLAUDE.md`.

## D13 — Phase 0 status and next step (2026-10-06, Phase 0, user decision)

- Phase 0 (literature/novelty, networks, cycle model + DSE, baselines, OOC feasibility, V3_SPEC.md) is in
  progress. First versions are committed in e5bb077, 99f0018, b827ae0, c36414f and c713da5 (listed in
  V3_PLAN §10). Pending: final novelty check, training, V3_SPEC.md.
- Work that is still valid is kept. Work that conflicts with D4–D12 is marked superseded, not deleted.

## D1/D2 amendment (2026-10-06, user decision)

- **D1 superseded in part:** the level scheme (A1/A2, B1–B4, C1–C3) and the order A1 → B1 → C1 → A2/B2 →
  B3 → C2 → B4 are replaced by milestones M1–M5 (D5). B4 (weight-stationary) is dropped (D6), so the
  `DSE_NOTES.md` P14 WS sketch and the `ws_cycles_sketch` column are no longer design inputs. The rest of
  D1 (directory, branch, worktree, V2 frozen, Phase 0 = decisions only) stays valid.
- **D2 superseded in part:** 3.000 ns / 333.33 MHz is not a timing target. The 3.000 ns OOC runs
  (b827ae0) and the 333.33 MHz DSE column remain feasibility data points only. The design target is
  4.000 ns / 250 MHz (D5). The PLL facts and the D19/D20 read-back rule stay valid.

## TODO

- Phase 0 remainder (D13): final novelty check, training, V3_SPEC.md; further decisions as D14+.

## Open decisions (Phase 0, to be resolved with data, then approved by the user)

- Array R×C (16×16 expected, D5) and INT8 DSP packing (conditional, D6); row mapping (x-only vs flattened
  x·y); stride-2 handling under banked ACT reads; residual-add placement; ResNet-20 shortcut option A/B;
  MobileNet-style net widths; PTQ→QAT threshold; descriptor word count, field layout and storage (D7);
  TARGETS values; RTL module prefix. (B4 keep/drop: resolved, dropped, D6.)

## Open conflicts

(none)
