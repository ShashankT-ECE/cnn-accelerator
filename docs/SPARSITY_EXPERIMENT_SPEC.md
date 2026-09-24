# Activation Sparsity Experiment Specification (Step 5.1)

> **Status: FROZEN (2026-09-23).** This defines and freezes the *experiment
> contract* for activation zero-skipping on the LeNet-5 workload — what is
> measured, how, and what the hardware can and cannot support. It does **not**
> implement zero-skipping, does not modify RTL/mapping/L1/L2, and does **not**
> claim a hardware speedup. Measurement is deterministic (fixed 10k MNIST test
> split, L1/L2 preprocessing, calibrated INT8 model); results in
> `data/benchmark/sparsity_results.json` via `python/sparsity_model.py`.

## 1. What constitutes a skippable zero

- **Skippable = runtime activation zeros produced by ReLU.** A MAC
  `acc += a · w` is skippable when its **activation operand `a == 0`** (the
  product contributes 0, so the MAC can be omitted). Only post-ReLU activations
  are in scope.
- **Excluded:** structural zeros in Conv1's raw-image input (MNIST background +
  the 2-pixel zero pad) are *not* ReLU sparsity; they are measured separately and
  reported as `kind=structural`.
- **Excluded:** zero *weights* are **not** the primary sparsity mechanism (per
  the frozen symmetric INT8 scheme, weights are dense int8; weight pruning is
  out of scope for this experiment).

## 2. Tensors measured (distinguished: fed into MACs)

The activation operand of each layer's MAC is that layer's **input** tensor:

| Layer | Input tensor | Shape (per image) | Provenance |
|---|---|---|---|
| conv1 | `q_input` | (1, 32, 32) | **structural** (raw image) |
| conv3 | `pool1` (Conv1 ReLU+pool) | (6, 14, 14) | **ReLU** |
| conv5 | `pool2` (Conv3 ReLU+pool) | (16, 5, 5) | **ReLU** |
| fc1 | `c5` (Conv5 ReLU) | (120,) | **ReLU** |
| fc2 | `f6` (FC1 ReLU) | (84,) | **ReLU** |

Only these *input* tensors are fed into MAC operations. Layer *outputs* (pre/ post
ReLU) are recorded for context, but the skippable-MAC accounting is defined on
the inputs above.

## 3. How sparsity is measured

For each tensor in §2, over the full 10k test split:

- **zero_count** — number of elements equal to 0 (int8).
- **total_elements** — tensor element count.
- **zero_pct** = `zero_count / total_elements × 100`.
- **nonzero_pct** = `100 − zero_pct`.
- **skippable_macs (per image)** — the number of MACs whose activation operand is
  zero. For a conv layer: `OC × Σ_{zero A[ic,i,j]} num_uses(i,j)`, where
  `num_uses(i,j) = ny(i)·nx(j)` is how many `(y,x,ky,kx)` taps read that
  activation in a valid K×K conv. For an FC layer: `OC × zero_count`.
- **skippable_pct** = `skippable_macs / total_MACs_per_image × 100`.

`num_uses` accounts for boundary effects: interior activations are read K·K times,
boundary activations fewer, so the skippable-MAC fraction can differ from the raw
zero fraction (notably conv3: 66.0% zeros but 49.3% skippable MACs).

## 4. Evaluation dataset

- **Frozen MNIST test split**, 10,000 samples, `torchvision.datasets.MNIST`
  (torchvision 0.16.2), zero-pad-2 → 32×32, `[0,1]`, no mean/std — identical to
  the L1/L2 preprocessing (`docs/LENET5_SPEC.md` §4).
- **Deterministic inference** through the calibrated INT8 model
  (`data/lenet5_int8/quant_params.npz`), fixed indices, no RNG.

## 5. Measurement scope

- **Globally** (full 10k test set) and **per layer**, as above.
- **Per sample** is available from the same tensors but is **not** a frozen
  deliverable in this step. No per-sample thresholds are defined.

## 6. Dense-vs-sparse cycle model (conceptual — not implemented)

- **Dense baseline** = the frozen Step 4.3 mapping result (OS 43,021 cycles;
  `docs/LENET5_MAPPING_FROZEN_MANIFEST.md`).
- **Sparse execution** omits MAC work whose activation is zero.
- **Theoretical MAC reduction** = the per-MAC skippable fraction (§3, §9) —
  **59.08%** of the 416,520 MACs/image, but this is the *upper bound*.
- **Actual cycle reduction** depends on the skip *granularity* (whole-group skip
  skips far fewer MACs than per-MAC skip) and is **not** assumed to equal the
  theoretical reduction — no linear-speedup assumption is made.

## 7. What the current RTL can / cannot support (gaps)

| Capability | Status |
|---|---|
| Coarse zero-**group** skip in WS (skip a whole group whose 5×12 window is all-zero; `PHASE2_ARCHITECTURE.md` §6) | **already supported** (MNIST-12 Conv1 only) |
| Per-MAC `zero_skip` gating | **datapath limitation** — the frozen array's `zero_skip` gates the *shifted* product (`act_delayed`), not the fed stream; gating on the fed stream corrupts results, so it is tied 0 |
| OS-mode sparsity (the coarse skip is WS-only; OS is channel-parallel and needs 8-word emission) | **new sparsity-control requirement** |
| Generalizing the zero-detection mask + skip predicates + per-layer MAC counters from MNIST-12 Conv1 to the five LeNet-5 layers | **controller/storage gap** |

**Classification summary:** (1) already supported — WS coarse group skip;
(2) controller/storage gap — per-layer skip predicates + sparsity counters;
(3) datapath limitation — per-MAC `zero_skip` (gates shifted product);
(4) new sparsity-control requirement — OS-mode skip, and a LeNet-5 skip
granularity that is honest about the gap between per-MAC (59.08%) and
group-level skipping.

## 8. Frozen measured results (10k test set, deterministic)

| Layer | kind | zero_count | total | zero% | skip MACs/img | skip% |
|---|---:|---:|---:|---:|---:|
| conv1 | structural | 8,732,613 | 10,240,000 | 85.28% | 95,003 | 80.79% |
| conv3 | relu | 7,765,910 | 11,760,000 | 66.04% | 118,351 | 49.31% |
| conv5 | relu | 2,186,396 | 4,000,000 | 54.66% | 26,237 | 54.66% |
| fc1 | relu | 718,248 | 1,200,000 | 59.85% | 6,033 | 59.85% |
| fc2 | relu | 456,667 | 840,000 | 54.37% | 457 | 54.37% |
| **TOTAL** | | | | | **246,081** | **59.08%** |

**Key fact:** the ReLU (skippable) sparsity is **49–60% in the deep layers**;
the 85% figure at Conv1 is structural (background + padding), not ReLU. The
theoretical per-MAC reduction is **59.08%** of 416,520 MACs/image; the achievable
cycle reduction under the coarse group skip is expected to be lower and is a
Step 5.2 measurement, not an assumption here.

## 9. Deliberately not defined here

- The skip *granularity* and the resulting sparse cycle model (Step 5.2).
- Any hardware speedup claim (requires the sparse RTL/controller).
- Per-sample statistics and any sparsity threshold.
