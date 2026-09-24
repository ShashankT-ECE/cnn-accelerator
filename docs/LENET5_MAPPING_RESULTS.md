# LeNet-5 OS/WS Mapping Results (Step 3.2–3.4, corrected Step 4.2)

> **Status:** MEASURED + CORRECTED (2026-09-23). The FC cycle schedule was
> corrected in Step 4.1 (`docs/FC_SCHEDULE_VALIDATION.md`): the earlier model
> reused the 8-pixel-group lead-in for 1-pixel FC layers, overcounting OS-FC by
> ~8×. This document records the **corrected** results. `python/mapping_model.py`,
> `python/reconfig_model.py`, `python/activity_model.py`; machine-readable results
> in `data/benchmark/{mapping,reconfig,activity}_results.json`. No RTL modified.

## 1. Model, anchors, and verification status

| Timing | Status | Source |
|---|---|---|
| OS conv group (8 px, IC=1, K=5) = 82 cycles | **RTL-verified** | `PHASE2_ARCHITECTURE.md` §3.1 |
| WS conv group (25 taps = 4 tiles) = 42 cycles | **RTL-verified** | `SYSTOLIC_ARRAY_V2_SPEC.md` §9.9 |
| OS FC group = `IC + 2·rows + 2` | **RTL-verified** (bit-exact sim) | `FC_SCHEDULE_VALIDATION.md` |
| WS FC group = `8·ceil(IC/8) + 2`/neuron | **analytically derived** (conv anchor) | `FC_SCHEDULE_VALIDATION.md` |
| Conv3/Conv5 (IC>1, OC>8, 32×32) | **unverified generalization** | controller gap (G1–G3) |

Formulas:
- **OS conv** `group = 17 + IC·K·(K+8)`, groups `= ceil(OC/8)·H'·ceil(W'/8)`.
- **OS FC** `cycles = ceil(OC/8)·(IC+2) + 2·OC`.
- **WS conv** `group = 8·ceil(T/8)+10`, `T = K·K·IC`, groups `= OC·H'·ceil(W'/8)`.
- **WS FC** `cycles = OC·(8·ceil(IC/8)+2)`.

## 2. Results (corrected)

| Layer | T | MACs | OS cycles | OS util | WS cycles | WS util | Winner | OS/WS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conv1 (1→6) | 25 | 117,600 | 9,184 | 20.0% | 28,224 | 6.5% | OS | 0.33× |
| conv3 (6→16) | 150 | 240,000 | 16,280 | 23.0% | 51,840 | 7.2% | OS | 0.31× |
| conv5 (16→120) | 400 | 48,000 | 15,855 | 4.7% | 49,200 | 1.5% | OS | 0.32× |
| fc1 (120→84) | 120 | 10,080 | **1,510** | 10.4% | **10,248** | 1.5% | OS | 0.15× |
| fc2 (84→10) | 84 | 840 | **192** | 6.8% | **900** | 1.5% | OS | 0.21× |
| **TOTAL** | | **416,520** | **43,021** | 15.1% | **140,412** | 4.6% | OS | 0.31× |

## 3. Findings (corrected)

**F1 — OS wins every layer; there is no OS/WS crossover.** Convolutions favor OS
~3×; FC layers favor OS by ~6.8× (fc1) and ~4.7× (fc2) once the FC schedule is
corrected. The Step 3.2/3.3 "WS wins FC" crossover was an **artifact** of the
overcounted OS-FC formula.

**F2 — Mechanism.** OS parallelizes output channels on rows (cheap for small
reductions and many channels) *and* serializes the FC reduction efficiently (one
feature per cycle, weight pipelined). WS's per-row weight broadcast forces FC to
one neuron per sweep with 7/8 columns idle, and its tile/sweep overhead is never
competitive. Net: OS is **3.26× faster** than WS overall (43,021 vs 140,412).

**F3 — There is no reconfiguration benefit on LeNet-5.** Because OS is optimal
for all five layers, the per-layer-optimal sequence is OS/OS/OS/OS/OS, i.e.
**identical to fixed OS**. Reconfiguration buys nothing (0 switches, 0 overhead,
speedup 1.000×).

**F4 — FC is low-utilization in both modes (honest).** FC maps as a 1×1 conv
with one output pixel, wasting 7/8 columns in both OS (10.4%/6.8%) and WS
(1.5%). OS's FC win is large but is still a low-utilization regime.

**F5 — Absolute utilization is low everywhere** (15.1% OS, 4.6% WS overall) —
the correctness-first un-pipelined baseline on a small 8×8 array for a
416,520-MAC workload. Group-boundary pipelining would raise utilization without
changing the ranking.

## 4. Reconfigurable dataflow (corrected)

| Mode | Cycles | vs fixed OS |
|---|---:|---:|
| Reconfigurable (per-layer best + overhead) | **43,021** | 1.000× |
| Fixed OS | 43,021 | — |
| Fixed WS | 140,412 | 3.264× |

Selected sequence = `OS → OS → OS → OS → OS` (**0 switches, 0 overhead**).
**Reconfiguration provides no benefit on this workload** — the honest conclusion
is that dynamic OS/WS selection has no measurable basis on LeNet-5 because OS
dominates every layer. A crossover would require a more FC/pointwise-dominated
workload (e.g. the CIFAR-10 candidate).

## 5. Hardware activity (corrected totals)

| Layer | Mode | Cycles | MACs | PE util | Psum ops |
|---|---:|---:|---:|---:|
| conv1 | OS | 9,184 | 117,600 | 20.0% | 0 |
| conv3 | OS | 16,280 | 240,000 | 23.0% | 0 |
| conv5 | OS | 15,855 | 48,000 | 4.7% | 0 |
| fc1 | OS | 1,510 | 10,080 | 10.4% | 0 |
| fc2 | OS | 192 | 840 | 6.8% | 0 |
| **TOTAL** | OS | **43,021** | 416,520 | **15.1%** | 0 |
| **TOTAL** | WS | **140,412** | 416,520 | **4.6%** | 416,520 |

Weight reads are mode-independent (123,720 in both). Psum work is WS-only (each
WS MAC is `psum_in + product`). Activation-read byte counts and bandwidth remain
marked unavailable (line-buffer reuse and clock×bus-width are not pinned).
