# PE Architecture Study — OS + WS + Fine-Grained Sparsity (Step 6.1)

> **Status:** INVESTIGATED (2026-09-23). Answers whether the 8×8 PE/datapath can
> be redesigned to support output-stationary, weight-stationary, and *useful*
> fine-grained activation zero-skipping on the same hardware — without a full
> SCNN-style architecture. No RTL modified; recommendation only.

## The research question

Can one PE design support, simultaneously and efficiently:
1. OS execution, 2. WS execution, 3. fine-grained activation zero-skipping
(cycle savings), 4. IC/OC tiling, 5. INT8×INT8→INT32, 6. dataflow reconfiguration?

## The fundamental tension

OS and WS are **fixed-timing systolic** dataflows: the per-row shift chain
(7 registers) and the WS psum cascade assume a fixed cycle rhythm. Useful
fine-grained sparsity (cycle savings) requires **data-dependent compaction** —
skipping a cycle when an activation is zero. These two properties conflict:
compaction desynchronizes the systolic pipeline. Separately, the MAC is a
**DSP48E2 hard multiplier** (fixed power), so gating a zero operand saves no
energy either.

## Option comparison

| Dimension | A. Keep current PE | B. Modified (zero-aware MAC) | C. Mask routing | D. Compacted (SCNN-style) |
|---|---|---|---|---|
| PE change | none | +1 comparator, gate | +mask reg | **major** (compacted in) |
| Array change | none | per-PE `zero_skip` wire | mask fan-out | **crossbar** replaces shift chain |
| Controller change | none | none | none | **sparse scheduler** |
| Extra regs/DSP/LUT/BRAM | 0 | 64 cmp (LUT) | 64 mask regs | buffers + crossbar + index BRAM |
| Routing complexity | low | low | low | **high** |
| II=1 maintained | yes | yes | yes | **hard** (contention) |
| OS capable | yes | yes | yes | no |
| WS capable | yes | yes | yes | no |
| Fine sparsity → cycles | no | no | no | yes (theoretical) |
| Utilization impact | 15.1% OS / 4.6% WS | unchanged | unchanged | higher, at cost of OS/WS |
| XCK26 implementation risk | none | low | low | **high** |
| Verification effort | done (verified) | low | low | **very high** |

## Cycle results on the frozen LeNet workload

| Design | OS cycles | WS cycles | fine-sparse cycles |
|---|---:|---:|---:|
| A/B/C (systolic) | 43,021 | 140,412 | 43,021 (no gain) |
| D (compacted) | n/a | n/a | ~17,604–22,885 (2.44×–1.88×) |

## Findings

**F1 — No single PE design achieves OS + WS + *useful* sparsity.** Options B/C
add a zero-aware gate that yields no cycle savings (fixed systolic timing) and
≈0 energy (DSP48E2 fixed power). Option D achieves the sparsity speedup but
*abandons* the systolic OS/WS datapaths (crossbar replaces the shift chain and
psum cascade). The three objectives are mutually incompatible on one datapath.

**F2 — The current PE already does the project's actual contribution.** `pe_v2`
cleanly supports OS ↔ WS reconfiguration (the verified, frozen feature), IC/OC
tiling (controller-level), INT8×INT8→INT32, and a `zero_skip` port that already
serves the coarse group skip. A redesign would add risk without adding the
sparsity capability.

**F3 — Fine-grained sparsity is a different accelerator.** Capturing the 59.08%
per-MAC reduction requires a compaction-based (SCNN-style) datapath, which is a
separate project, not a PE modification.

## Recommendation

**KEEP CURRENT PE.** The PE is correctly designed for the project's real
contribution — like-for-like OS/WS dataflow measurement with runtime
reconfiguration — and is already implemented and verified. Fine-grained
activation sparsity is not a PE-modification problem; it is incompatible with the
systolic dataflow and would require a different architecture. Any paper claim of
fine-grained sparsity on *this* accelerator would be unsupported by the hardware;
the honest sparsity result remains the coarse group skip (1.099×), with the
per-MAC 59.08% reported as an unreachable upper bound.
