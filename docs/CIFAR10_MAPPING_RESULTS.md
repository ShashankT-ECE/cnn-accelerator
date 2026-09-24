# CIFAR-10 OS/WS Hardware-Mapping Analysis (Step 6.4)

> **Status:** COMPLETE (2026-09-23). Maps the four learned CIFAR-10 layers to the
> frozen 8×8 reconfigurable systolic array using the **same** cycle methodology as
> the frozen LeNet-5 baseline (`docs/LENET5_MAPPING_FROZEN_MANIFEST.md`), with the
> corrected FC schedule (`docs/FC_SCHEDULE_VALIDATION.md`). This is the decision
> experiment for whether the current reconfigurable-dataflow architecture should
> be retained. **No RTL is modified, no PE redesign, no sparsity, no frozen LeNet
> artifact is touched, and no OS/WS crossover is assumed or forced.**

Artifacts: `python/cifar10_mapping_model.py`,
`python/tests/test_cifar10_mapping_model.py`,
`data/benchmark/cifar10_mapping_results.json`, this document.

---

## 1. Tensor-dimension verification (CIFAR INT8 vs frozen workload)

The frozen CIFAR-10 workload (`docs/CIFAR10_WORKLOAD_SPEC.md` §2) and the
INT8/FP32 reference (`python/cifar10/model.py`, `python/cifar10/int8_model.py`,
`python/cifar10/config.py`) agree exactly. Quantization is
dimension-preserving, so the INT8 tensors carry the same shapes as FP32.
`python/tests/test_cifar10_reference.py` (Step 6.2) and
`python/tests/test_cifar10_int8.py` (Step 6.3) confirm the shapes, dtypes, and
the 4,493,440 MAC / 79,978-param totals.

| Layer | Input | Output | Learned? | Maps to array? |
|---|---|---|---|---|
| Conv1 | (N,3,32,32) | (N,32,28,28) | yes | **yes** |
| Pool1 | (N,32,28,28) | (N,32,14,14) | no (PS) | no |
| Conv2 | (N,32,14,14) | (N,32,10,10) | yes | **yes** |
| Pool2 | (N,32,10,10) | (N,32,5,5) | no (PS) | no |
| Conv3 | (N,32,5,5) | (N,64,1,1) | yes | **yes** |
| FC | (N,64) | (N,10) | yes | **yes** |

Pooling and ReLU are PS-side (identical convention to LeNet-5), so **four
learned layers map to the array**. Layer MAC counts match the frozen spec §3:
1,881,600 / 2,560,000 / 51,200 / 640 = **4,493,440**.

> Note: no `data/cifar10_int8/` frozen golden directory exists yet (Step 6.3 INT8
> quantization is in progress — only `int8_model.py` + the eval runner are
> present). This is immaterial to Step 6.4: the mapping uses the *frozen tensor
> shapes*, which INT8 does not change.

## 2. Model, anchors, and verification status

The cycle formulas are the **frozen LeNet-5 baseline family**, re-applied to the
CIFAR shapes (identical, not re-derived):

| Formula | Value | CIFAR applicability |
|---|---|---|
| OS conv group = `17 + IC·K·(K+8)` | 82 cyc anchor (IC=1,K=5) | all CIFAR convs have IC>1 → **unverified generalization** |
| WS conv group = `8·ceil(T/8)+10`, T=K·K·IC | 42 cyc anchor (T=25) | all CIFAR convs have T≠25 → **unverified generalization** |
| OS FC = `ceil(OC/8)·(IC+2) + 2·OC` | 152 cyc (IC=64,OC=10) | **RTL-verified** (bit-exact, general 1-pixel schedule) |
| WS FC = `OC·(8·ceil(IC/8)+2)` | 660 cyc | **analytically derived** (conv-WS anchor) |

**The corrected FC schedule is used throughout** (Step 4.1): OS-FC is
`ceil(OC/8)·(IC+2) + 2·OC`, **not** the old `17 + IC·K·(K+8)` overcount that
reused the 8-pixel lead-in. The old formula would have given FC ≈ 1,242 cycles
(~8× too high); the corrected value is 152.

**Verification-status discipline (never upgraded silently):**

| Timing | Status |
|---|---|
| OS FC (CIFAR fc) | **RTL-verified** (bit-exact) |
| WS FC (CIFAR fc) | analytically derived |
| OS conv (all 3 CIFAR convs) | **unverified generalization** (IC>1 and/or OC>8) |
| WS conv (all 3 CIFAR convs) | **unverified generalization** (T≠25) |

The `82`/`42` anchors are RTL-verified only for IC=1/K=5 and T=25 respectively.
CIFAR exercises exactly the IC>1/OC>8 regime that the current controller cannot
yet drive (gaps G1–G3 below), so **every CIFAR conv cycle count is an unverified
generalization** of a verified anchor. This is the same status the frozen LeNet
manifest assigns to its own Conv3/Conv5 layers.

## 3. Results (primary — frozen-baseline formula family)

| Layer | IC→OC | T | MACs | OS cyc | OS util | WS cyc | WS util | Winner | OS/WS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| conv1 | 3→32 | 75 | 1,881,600 | **94,976** | 31.0% | 322,560 | 9.1% | **OS** | 0.294× |
| conv2 | 32→32 | 800 | 2,560,000 | **167,760** | 23.8% | 518,400 | 7.7% | **OS** | 0.324× |
| conv3 | 32→64 | 800 | 51,200 | **16,776** | 4.8% | 51,840 | 1.5% | **OS** | 0.324× |
| fc | 64→10 | 64 | 640 | **152** | 6.6% | 660 | 1.5% | **OS** | 0.230× |
| **TOTAL** | | | **4,493,440** | **279,664** | 25.1% | **893,460** | 7.9% | **OS** | **0.313×** |

**OS wins every CIFAR layer. There is no OS/WS crossover.**

### 3.1 Per-layer tiling

| Layer | OS: IC tiling | OS: OC tiling | OS: spatial | WS: IC tiling | WS: OC tiling | WS: spatial |
|---|---|---|---|---|---|---|
| conv1 | serialized, 3×5=15 passes | 4 groups of 8 (OC=32) | 28×4=112 (last col-group 4 px) | 75 taps → 10 tiles | 32 sweeps | 112 |
| conv2 | serialized, 32×5=160 passes | 4 groups of 8 (OC=32) | 10×2=20 (last col-group 2 px) | 800 taps → 100 tiles | 32 sweeps | 20 |
| conv3 | serialized, 160 passes | 8 groups of 8 (OC=64) | 1×1=1 (1 px) | 800 taps → 100 tiles | 64 sweeps | 1 |
| fc | serialized, 64 features | 2 groups (8+2 rows) | 1 | 64 taps → 8 tiles | 10 sweeps (1 neuron/sweep) | 1 |

### 3.2 Exact traffic

| Layer | Weight reads (mode-independent) | Psum ops (OS) | Psum ops (WS) |
|---|---:|---:|---:|
| conv1 | 32·3·25 × 112 = 268,800 | 0 | 1,881,600 |
| conv2 | 32·32·25 × 20 = 512,000 | 0 | 2,560,000 |
| conv3 | 64·32·25 × 1 = 51,200 | 0 | 51,200 |
| fc | 10·64 × 1 = 640 | 0 | 640 |
| **TOTAL** | **832,640** | **0** | **4,493,440** |

Weight reads are mode-independent (the full weight set is re-streamed once per
spatial group in both modes). Psum traffic is **WS-only**: every WS MAC is a
`psum_in + product` cascade op; OS uses the local `acc += product` accumulator.

### 3.3 The structural reason OS wins (not an artifact)

Ignoring fixed overhead and tile rounding, the OS/WS cycle ratio for a K×K conv
collapses to a **closed form independent of IC and OC**:

```
OS  per (8 channels, 8 pixels) ≈ IC·K·(K+8)
WS  per (8 channels, 8 pixels) ≈ 8 · IC·K·K          (8 serialized sweeps)
   ⇒  OS/WS  ≈  (K+8) / (8K)  =  13/40  =  0.325   for K=5
```

This is `(K+8)/(8K)`, which is **< 1 for every K ≥ 2** (K=5 → 0.325, K=3 →
0.458). The mechanism is symmetric parallelism: OS's 8 output-channel rows and
WS's 8 reduction-tap rows both expose 8× parallelism, but WS *also* serializes
the output channels (one sweep each), so OS's channel parallelism is pure win.
WS can only theoretically compete at K=1 (pointwise), and there the per-row
weight-broadcast constraint caps WS-FC to **one output neuron per sweep**
(7/8 columns idle) — so the corrected FC schedule still favors OS (0.23×).

conv2/conv3 (IC=32, T=800 = 8×100 exactly) land within 0.5% of the 0.325 limit;
conv1 (IC=3) carries a larger fixed-overhead fraction and a non-exact tile
(75 = 9×8+3), so it lands at 0.294 — still decisively OS.

## 4. CIFAR assumptions vs actual current RTL

The frozen datapath (`pe_v2.sv`, `systolic_array_v2.sv`) needs **no change**;
all gaps are controller/input-feed scope. CIFAR exercises every one of them:

| Requirement (CIFAR) | Current RTL | Gap | Status for CIFAR timing |
|---|---|---|---|
| **IC > 1** (3, 32, 32) | controller drives IC=1 only | **G1 (blocking)** | unverified generalization |
| **OC > 8** (32, 32, 64) | controller drives OC=8 only | **G3 (blocking)** | unverified generalization |
| **variable spatial** (28, 10, 1) | hardcoded 28×28 | **G2 (blocking)** | unverified generalization |
| **32×32 input** | hardcoded 28×28 | **G2** | unverified generalization |
| **weight storage ~79,840 int8 (≈78 KB)** | ~1.2 KB LUTROM | **G5 (blocking)** | — (needs BRAM; K26 has ~648 KB) |
| **VALID convolution** | SAME-pad zero-injection | **G6 (moderate)** | — (re-parameterize predicates) |
| **weight broadcast behavior** | per-row broadcast | none (conv) | fine for conv OS/WS |
| weight broadcast (FC in WS) | per-row broadcast | architectural | caps WS-FC to 1 neuron/sweep |
| **local accumulator (OS) / psum cascade (WS)** | both in `pe_v2.sv` | none | both implemented + verified |

**Nothing in the CIFAR mapping silently assumes an unsupported datapath
capability.** The conv OS/WS formulas generalize the two RTL-verified anchors
across IC>1/OC>8, and are labeled "unverified generalization" — exactly the
gaps (G1–G3) the LeNet mapping spec already records as blocking. Weight storage
(G5) is an additional blocker for CIFAR's ~78 KB of weights.

## 5. Direct comparison with LeNet-5

| LeNet layer | LeNet OS | LeNet WS | CIFAR layer | CIFAR OS | CIFAR WS |
|---|---:|---:|---:|---:|---:|
| conv1 (1→6) | 9,184 | 28,224 | conv1 (3→32) | 94,976 | 322,560 |
| conv3 (6→16) | 16,280 | 51,840 | conv2 (32→32) | 167,760 | 518,400 |
| conv5 (16→120) | 15,855 | 49,200 | conv3 (32→64) | 16,776 | 51,840 |
| fc1 (120→84) | 1,510 | 10,248 | — | — | — |
| fc2 (84→10) | 192 | 900 | fc (64→10) | 152 | 660 |
| **TOTAL** | **43,021** | **140,412** | **TOTAL** | **279,664** | **893,460** |

LeNet has 5 learned layers (conv1/conv3/conv5/fc1/fc2); CIFAR has 4
(conv1/conv2/conv3/fc). Rows are aligned by *position in the network*, not by
identical shape.

**Per-layer OS/WS speed ratio (OS is faster by):**

| Layer | LeNet OS/WS | CIFAR OS/WS | CIFAR WS÷OS |
|---|---:|---:|---:|
| Conv1 | 0.33× | 0.294× | 3.40× |
| Conv2/Conv3 | 0.31× | 0.324× | 3.09× |
| Conv3/Conv5 | 0.32× | 0.324× | 3.09× |
| FC (last) | 0.21× | 0.230× | 4.34× |
| **Overall** | **0.306×** | **0.313×** | **3.195×** |

(`OS/WS` = OS cycles ÷ WS cycles, <1 means OS faster; `WS÷OS` = the inverse,
i.e. how many times slower WS is on CIFAR.)

**Utilization:**

| | LeNet | CIFAR |
|---|---:|---:|
| OS | 15.1% | 25.1% |
| WS | 4.6% | 7.9% |

**IC/OC pressure:** CIFAR is the more *reduction-heavy* workload — IC reaches 32
(vs LeNet's 16), reduction depth T reaches 800 (vs LeNet's 400). But its OC
maxes at 64 (vs LeNet's 120). The extra IC pressure does **not** rescue WS: WS's
tiling overhead scales with the reduction depth it was meant to exploit, while
OS parallelizes the (large) OC count that WS serializes.

**Tiling pressure:** CIFAR conv2/conv3 need 100 WS tiles / 160 OS passes
(vs LeNet's 50/80 max). Deeper reductions make WS's per-tile `weight_load`
overhead *more*, not less, prominent — the opposite of the "reduction-heavy
favors WS" hypothesis.

## 6. Per-layer classification (no arbitrary threshold)

Classification uses "the faster mode by a margin larger than plausible
implementation noise" — not a fixed 30% gate. Every layer's margin is far larger
than the 8-cycle reconfiguration flush and any tiling-reload overhead:

| Layer | OS cyc | WS cyc | Margin | Classification |
|---|---:|---:|---|---|
| conv1 | 94,976 | 322,560 | WS 3.40× slower | **OS clearly preferred** |
| conv2 | 167,760 | 518,400 | WS 3.09× slower | **OS clearly preferred** |
| conv3 | 16,776 | 51,840 | WS 3.09× slower | **OS clearly preferred** |
| fc | 152 | 660 | WS 4.34× slower | **OS clearly preferred** |

**No layer is a near-tie; no layer favors WS.**

## 7. Reconfiguration policy analysis

Because OS is the per-layer optimum for all four layers, the oracle sequence is
`OS → OS → OS → OS`, i.e. **identical to fixed OS**.

| Schedule | Cycles | vs fixed OS |
|---|---:|---:|
| Fixed OS | 279,664 | — |
| Fixed WS | 893,460 | 3.195× slower |
| Per-layer adaptive (oracle) | 279,664 | 1.000× |
| Adaptive + reconfig overhead (0 switches × 8 cyc) | 279,664 | 1.000× |

- **0 switches, 0 reconfiguration overhead.**
- Reconfiguration buys **nothing** on CIFAR: `reconfigurable == fixed OS`.
- Speedup vs fixed WS = 3.195× (this is OS's win, **not** reconfiguration's).

**Exact fixed-OS vs fixed-WS difference:** 893,460 − 279,664 = **613,796 cycles**
(WS is 3.195× slower).

### 7.1 Robustness to the partial-group refinement

The frozen baseline (and this model's primary numbers) uses a **uniform** per-group
cost, which `FC_SCHEDULE_VALIDATION.md §6` notes overcounts the last (partial)
column group of a non-multiple-of-8 output width. Applying that per-group
correction (`1 + IC·K·(P+K) + 2·rows`) **reduces OS only** — WS's group cost
depends only on reduction taps T, not on the pixel count — so it *widens* OS's
lead rather than threatening the conclusion:

| | Primary (uniform) | Refined (§6 partial groups) |
|---|---:|---:|
| OS total | 279,664 | **225,584** |
| WS total | 893,460 | 893,460 (unchanged) |
| OS/WS | 0.313× | 0.2525× |
| WS÷OS | 3.195× | **3.961×** |

The uniform simplification is therefore **conservative for OS**: the true OS
advantage on CIFAR is *at least* 3.2×, and the §6 refinement pushes it toward
4.0×. No winner changes.

## 8. Decision

**B — the current architecture is effectively OS-dominant, and the
reconfigurable-dataflow claim should be reconsidered.**

The evidence:

1. **OS wins every CIFAR layer** (3.09–4.34× per layer; 3.20× overall), including
   the reduction-heavy conv2/conv3 (T=800) that were the strongest *a priori*
   case for WS. **No dataflow crossover was found.**
2. **The OS/WS ratio is structural**, not workload-specific: `(K+8)/(8K) ≈ 0.325`
   for K=5, independent of IC/OC. OS's 8 output-channel rows and WS's 8
   reduction-tap rows are symmetric parallelism, but WS *additionally* serializes
   output channels — so OS wins for every real (K≥2) convolution on this array.
3. **FC also favors OS** (4.34×), because the array's per-row weight broadcast
   caps WS-FC at one output neuron per sweep. There is no pointwise crossover
   either.
4. **Reconfiguration buys nothing**: the per-layer-optimal sequence is `OS×4`,
   identical to fixed OS. Speedup vs fixed WS (3.2×) is OS's win alone, not a
   reconfiguration benefit.

This mirrors the LeNet-5 finding (OS wins all five layers, 3.26× overall) and
closes the open question the CIFAR workload was designed to test: a genuinely
more IC-heavy, reduction-heavy, 3-channel workload does **not** produce an OS/WS
crossover. The reconfigurable-dataflow claim currently rests on a hypothetical
workload regime that this 8×8 array's datapath structure does not reward.

**Scope discipline:** this step establishes evidence only. It does **not**
recommend a new architecture, redesign the PE, or implement sparsity. The
question of whether to drop the WS path / reconfiguration (or what would
actually justify it) is the subject of a later step.

## 9. Reproducibility

```bash
.venv/bin/python python/cifar10_mapping_model.py            # -> data/benchmark/cifar10_mapping_results.json
.venv/bin/python python/tests/test_cifar10_mapping_model.py # 37 checks, all PASS
.venv/bin/python python/tests/test_mapping_model.py         # LeNet baseline (unchanged)
.venv/bin/python python/tests/test_reconfig_model.py        # LeNet reconfig (unchanged)
.venv/bin/python python/tests/test_activity_model.py        # LeNet activity (unchanged)
```

All five tests PASS. No RTL, PE, sparsity, or frozen LeNet/CIFAR artifact was
modified.
