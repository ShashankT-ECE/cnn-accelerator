# Fine-Grained Activation-Sparse Datapath — Design Investigation (Step 5.3)

> **Status:** INVESTIGATED (2026-09-23). Determines whether a *practical*
> fine-grained mechanism can convert the measured 59.08% ReLU MAC sparsity into
> a meaningful improvement on the frozen 8×8 array — without modifying RTL.
> Honest finding: **no practical mechanism does.**

## The problem

59.08% of MACs are skippable from ReLU zeros (Step 5.1), but coarse group
skipping achieves only ~10.3% cycle reduction (Step 5.2) because the zeros are
spatially scattered. This step asks: can finer granularity recover the gap?

## Root causes that dominate the answer

1. **The array is a fixed-timing systolic array.** The per-row shift chain
   (7 registers) and the WS psum cascade assume a fixed cycle rhythm. Any
   mechanism that *drops* a cycle (compaction) desynchronizes the pipeline.
2. **The MAC is a DSP48E2 hard multiplier** (fixed dynamic power, always
   clocked). Gating the multiplier on a zero operand saves neither cycles nor
   meaningful energy.

## Option comparison

| Dimension | A. Per-PE zero detect | B. Zero-aware compaction | C. Mask-driven gating |
|---|---|---|---|
| Extra state/registers | 64× (8-bit ==0) comparators | compaction buffers + crossbar + index regs | 64× mask regs |
| Control complexity | low | **high** (crossbar routing) | low |
| Metadata overhead | 0 | **value+index ≈ 2–2.75×** activation width | 1 bit/activation |
| PE change | 1 comparator (or reuse `zero_skip`) | **major** (compacted input) | none (mask→`zero_skip`) |
| Array change | per-PE `zero_skip` wiring | **major** (crossbar replaces shift chain) | mask fan-out |
| Controller change | none | **major** (sparse scheduling) | none |
| OS/WS impact | both, energy only | **abandons systolic (both)** | both, energy only |
| Accumulation correct? | yes (a·0 = 0) | yes but index-based | yes |
| II=1 maintained? | yes | **hard** (crossbar contention) | yes |
| Detection a bottleneck? | no | **yes** (detect + route) | no |
| Exploits 59.08%? | energy only (≈0 on DSP48E2) | cycles (theoretical) | energy only |

## Cycle-level model (10k MNIST, per image)

| Mode | Cycles | Speedup |
|---|---:|---:|
| Dense OS (frozen) | 43,021 | 1.000× |
| Coarse group skip (frozen) | 39,132 | 1.099× |
| **Fine gating (A/C)** | **43,021** | **1.000×** (no cycle change) |
| Fine compaction (B) — theoretical, zero overhead | 17,604 | 2.444× |
| Fine compaction (B) — est. 1.3× metadata overhead | 22,885 | 1.880× |

## Findings

**F1 — Gating (A/C) saves no cycles.** The systolic schedule is fixed; gating
the multiplier only zeroes the product for one cycle, it does not remove the
cycle. On a DSP48E2 the multiplier is a fixed-power hard block, so the energy
saving is also ≈0. Gating is therefore **not** a fine-grained speedup mechanism.

**F2 — Compaction (B) is the only cycle-saving mechanism, and it is a different
architecture.** Feeding only nonzero `(value,index)` pairs requires a crossbar /
indirection network that replaces the shift chain and the fixed schedule
(SCNN-style, Parashar et al. ISCA'17). It also adds `(value,index)` metadata
(≈2–2.75× the activation width) and makes sparse detection + routing a new
bottleneck. This is an **unreasonable architectural cost** for the frozen 8×8
array, not a practical extension.

**F3 — The 59.08% is unrecoverable on this datapath.** The gap between coarse
(10.3%) and fine (59.08%) MAC reduction is *real but unreachable*: it lives in
the per-MAC granularity that a fixed-timing, fixed-power systolic array cannot
exploit. A positive result here would have to come from compaction, which is a
different accelerator, not this one.

## Conclusion

**No practical fine-grained activation-sparsity mechanism produces a meaningful
improvement on the existing 8×8 DSP48E2 systolic array.** Per-PE/mask gating
(A/C) is cheap but yields zero cycle speedup (and ≈0 energy on DSP48E2);
compaction (B) would yield ~2.4× in theory but requires abandoning the systolic
architecture for a crossbar-based sparse datapath — an unreasonable cost. The
honest sparsity result for this accelerator remains the **coarse group skip
(1.099×)**, with the per-MAC 59.08% reported strictly as an unreachable
theoretical bound.
