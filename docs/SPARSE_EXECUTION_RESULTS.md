# Sparse Execution Results (Step 5.2)

> **Status:** MEASURED (2026-09-23). Determines what zero-skipping granularity
> the frozen 8×8 architecture can realistically exploit, without implementing
> RTL. Three cases are modeled on the frozen OS baseline (43,021 cycles).
> Machine-readable: `data/benchmark/sparse_execution_results.json`.

## Three cases

1. **Dense** — frozen Step 4.3 OS execution: **43,021 cycles/image**.
2. **Coarse / group-level** — a whole schedulable OS *spatial* group (8 pixels,
   all channels) is skipped only when its entire activation window
   (`K × (P+K−1) × IC`) is zero — the zero information the scheduler has at the
   group boundary. Partial groups are not assumed skippable; FC layers are
   all-or-nothing (their feature vector is shared across all output neurons).
3. **Fine / per-MAC** — the 59.08% theoretical MAC reduction (Step 5.1), an
   **upper bound only**; no cycle speedup is claimed (the frozen `zero_skip`
   gates the *shifted* product, so per-MAC gating is unsafe).

## Results (per image)

| Layer | Dense cyc | Groups | Skipped | skip% | Sparse cyc | Saved |
|---|---:|---:|---:|---:|---:|---:|
| conv1 | 9,184 | 112 | 47.4 | **42.28%** | 5,301 | 3,883 |
| conv3 | 16,280 | 20 | 0.0 | 0.04% | 16,274 | 6 |
| conv5 | 15,855 | 1 | 0.0 | 0.00% | 15,855 | 0 |
| fc1 | 1,510 | 1 | 0.0 | 0.00% | 1,510 | 0 |
| fc2 | 192 | 1 | 0.0 | 0.00% | 192 | 0 |
| **TOTAL** | **43,021** | | | | **39,132** | **3,889** |

- **Coarse speedup vs dense OS: 1.0994× (≈10%).**
- **Coarse skipped MACs: 42,801/image** (10.3% of 416,520) vs the **59.08%
  (246,081/image) per-MAC** upper bound.

## Honest findings

**F1 — Coarse group skip helps only Conv1, and only from structural zeros.**
42.28% of Conv1's 112 spatial groups have an entirely-zero 5×12 raw-image window
(background + padding). This is *structural* sparsity, not ReLU sparsity.

**F2 — ReLU sparsity is invisible to group-level skipping.** The deep layers'
ReLU zeros are spatially scattered, and their schedulable windows are large
(conv3: 5×12×6 = 360 pixels; conv5/FC: the entire shared feature vector), so an
entire group is essentially never all-zero. Their skip rates are ≈0%.

**F3 — The granularity gap is the whole story.** The theoretical per-MAC
reduction (59.08%) is 5.7× larger than the achievable group-level reduction
(10.3%). Capturing ReLU sparsity requires *finer-than-group* granularity
(per-MAC / per-tap / per-feature), which the frozen datapath cannot do safely.

## What coarse skipping requires (gap classification)

| Item | Classification |
|---|---|
| WS coarse zero-group skip (Phase 2, `PHASE2_ARCHITECTURE.md` §6) | **already supported** (MNIST-12 Conv1 only) |
| OS-mode coarse group skip (the frozen 43,021 baseline is OS; Phase 2 skip is WS-only) | **new sparsity-control requirement** |
| Per-MAC `zero_skip` gating (gates the shifted product) | **datapath limitation** — unsafe, tied 0 |
| Per-layer skip predicates + zero-detection masks + MAC counters for the 5 LeNet-5 layers | **controller/storage gap** |

## Conclusion

Zero-skipping on the current 8×8 architecture provides **≈10% speedup at group
granularity, almost entirely from Conv1's structural (background/padding) zeros
— not from ReLU sparsity.** The ReLU sparsity that motivated the experiment
(49–60% in deep layers) is only exploitable at per-MAC granularity, which the
frozen datapath does not support. Any paper claim of a large sparsity speedup on
this workload would be an artifact of fine-grain compaction that the current PE
cannot realize; the honest achievable number is **1.0994× (coarse)**, with the
per-MAC 59.08% reported strictly as a theoretical upper bound.
