# V3 plan

Source of truth for V3 scope and direction (user decision, 2026-10-06). Decisions are recorded in
`DECISIONS.md` as D4–D13; detailed technical contracts will live in `V3_SPEC.md` (not yet written).
`v2/` and its tags are frozen and never touched.

## 1. Thesis

A timing-predictable CNN accelerator on the KV260. The latency of an inference is known exactly before
it runs (model = RTL = board, cycle-exact). Reconfigurable dataflow and sparsity make it faster without
losing that guarantee.

## 2. Milestones

Each milestone is fully verified (section 5) before the next one starts.

| # | Milestone | Content |
|---|---|---|
| M1 | Foundation | R × C INT8 output-stationary array (16 × 16 expected, to be confirmed by the DSE), 250 MHz, SAME padding, stride 2, residual add, global average pooling, weights in URAM. Network: ResNet-20, CIFAR-10. |
| M2 | Split-K + compiler | Split-K mode, plus a compiler that picks the mode per layer and emits the exact predicted cycle count. |
| M3 | Structured weight sparsity | Block = C OCs × 1 K-step, skipped statically. Latency stays exact. |
| M4 | Depthwise | Depthwise mode, plus a small MobileNet-style CIFAR-10 net. |
| M5 | Activation zero-step skipping | Skip all-zero activation K-steps. The dense schedule is the guaranteed worst-case bound. |

Clock: the KV260 PL PLL gives 250 or 333.33 MHz, never 300 MHz (V2 D19). **250 MHz is the design
target.** 333.33 MHz is only a feasibility data point, not a design option.

## 3. Dropped and conditional

- **Dropped (decided):** weight-stationary mode (former level B4), DRAM / ImageNet, batching, 300 MHz.
- **Conditional:** INT8 DSP packing, only if Phase 0 proves it bit-exact and timing-clean.

## 4. Design rule: one bitstream, all modes

One final bitstream supports every mode, and a descriptor selects the mode per layer. One board session
can therefore measure the full ablation. The descriptor format reserves fields for every milestone
(M1–M5) from day one.

## 5. Verification (per milestone, on the laptop)

- cycle model
- bit-exact golden model
- full 10,000-image Verilator sims, bit-exact and cycle-exact
- random shapes
- post-implementation timing
- board-script dry-run

## 6. Board sessions

There are three sessions. The board is not ours, so they are planned in advance.

| Session | When | Content |
|---|---|---|
| S1 | End of Phase 0 | DPU + ORT baselines, V3 shell smoke test |
| S2 | After M2 | ResNet-20 dense + split-K |
| S3 | After M5 | Full ablation, MobileNet, sparsity curves, 3-session repeatability |

## 7. Evaluation

- predicted = measured latency per image
- speedup ablation per feature
- accuracy vs sparsity vs speed
- energy per image (SOM-rail power)
- vs DPU and ONNX Runtime (ORT) on the same nets, including latency jitter
- vs literature

## 8. Training

GPU jobs are run by the user on Kaggle. CPU smoke tests run locally (`v3/results/dryrun/`, never data).

## 9. Rules inherited from V2

- Numbers come only from scripts/CSVs.
- Every number is labelled: model / RTL sim / post-impl / measured on KV260.
- One Vivado job at a time.
- Testbenches are self-checking.
- Commit after each finished piece.

The full rule set is in `v3/CLAUDE.md`.

## 10. Next step: Phase 0 (in progress)

Phase 0 covers literature/novelty, networks, the cycle model + DSE, baselines, OOC feasibility and
`V3_SPEC.md`. It is partly done. First versions of these commits are on `v3-dev`:

| Commit | Covers |
|---|---|
| e5bb077 | Scaffold: `v3/CLAUDE.md` rules, `common.py` + `check_results.py` + `vivado_guard.sh` copied from V2 @ 28dd2ad, DECISIONS D1–D3, root CLAUDE.md V3 pointer |
| 99f0018 | Networks: shared net definitions (ResNet-20 shortcut A/B, proposed MobileNet-style net) + per-layer shape table (`v3/model/nets.py`) |
| b827ae0 | OOC feasibility (WS5, exploratory): PE row W=16/32 with/without INT8 DSP packing; URAM weight + banked ACT + residual-into-requant memory path; self-checking xsim TBs; exhaustive packing model; OOC flow at 4.000/3.000 ns + device query |
| c36414f | Literature (WS1: `refs.bib`, `LITERATURE.md`, `literature.csv`); DPU/ORT baseline flow + board baseline session, dry-run ready (WS3); extended cycle model + memory estimate + DSE driver (WS4) |
| c713da5 | Novelty check, interim: t1–t6 findings, merged related-work list, independent reference re-check (`docs/research/work/`) |

**Pending:** final novelty check (NOVELTY/VENUES), training, `V3_SPEC.md`.

**Superseded by this plan.** These are kept in the tree, not deleted:

- DECISIONS D1 level scheme (A1/A2, B1–B4, C1–C3) and its order A1 → B1 → C1 → A2/B2 → B3 → C2 → B4.
  The milestones M1–M5 (section 2) replace them.
- DECISIONS D2, where it lists 3.000 ns / 333.33 MHz as a timing target. The b827ae0 OOC runs at 3.000 ns
  and the `latency_us_333p33MHz` DSE column remain feasibility data points only.
- Former level B4 (weight-stationary): the `DSE_NOTES.md` P14 WS sketch and the `ws_cycles_sketch` DSE
  column are no longer a design input.
- The open decision "B4 keep/drop" is resolved: dropped.

Still valid: everything else above. That includes the scaffold and provenance rules (D3), the net
definitions, the feasibility RTL in `v3/feas/` (exploratory, never reused unreviewed), the literature,
the baseline flow and the cycle/memory model.
