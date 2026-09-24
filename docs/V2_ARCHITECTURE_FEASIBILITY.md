# V2 Architecture Feasibility — Trade Study (Phase 1)

> **Status:** ANALYTICAL STUDY (2026-09-23). Compares three candidate V2
> architectures against the empirical V1 baseline. This document **derives
> mathematics and resource estimates only** — no RTL is written, no Python model
> is written, no existing V1 RTL/testbench or frozen L1/L2 artifact is modified.
> It makes **no publishability claim and no speedup claim**; every number is
> either a measured V1 fact, a frozen cycle-model value, or an explicitly labeled
> analytical estimate.

---

## 1. Scope, guardrails, and evidence base

**Guardrails honored throughout:** (i) no RTL written; (ii) no V1 RTL or
testbench modified; (iii) no frozen L1/L2 reference artifact or V1 benchmark
data modified; (iv) no publishability claim; (v) the OS/WS crossover question is
answered from the mathematics, not engineered toward a desired result.

**Evidence base (all read, none modified):**

| Source | Provides |
|---|---|
| `docs/LENET5_MAPPING_SPEC.md`, `LENET5_MAPPING_RESULTS.md`, `LENET5_MAPPING_FROZEN_MANIFEST.md` | frozen OS/WS mapping contract + corrected cycle models |
| `docs/FC_SCHEDULE_VALIDATION.md` | RTL-verified FC schedule (the corrected formula) |
| `docs/CIFAR10_WORKLOAD_SPEC.md`, `CIFAR10_MAPPING_RESULTS.md` | frozen CIFAR topology + Step 6.4 OS/WS mapping |
| `docs/SPARSITY_EXPERIMENT_SPEC.md`, `SPARSE_EXECUTION_RESULTS.md` | frozen 59.08% ReLU-sparsity + coarse-skip result |
| `docs/FINE_GRAINED_SPARSITY_DESIGN.md`, `PE_V2_ARCHITECTURE_STUDY.md` | why fine sparsity is a different architecture |
| `docs/PHASE2_ARCHITECTURE.md`, `PHASE2_SYNTHESIS.md`, `PHASE2_HARDENING.md` | implemented V2 array + measured synthesis |
| `rtl/common/pe_v2.sv`, `systolic_array_v2.sv`, `cnn_accelerator_v2.sv` | the actual V1 datapath |
| `data/benchmark/synthesis.json`, `build/phase2/baseline.xdc` | measured XCK26 resource/timing |

### 1.1 The empirical V1 baseline (frozen)

| Workload | OS cycles | WS cycles | OS advantage | OS/WS crossover? |
|---|---:|---:|---|---|
| LeNet-5 (5 layers) | **43,021** | 140,412 | 3.26× | **none — OS wins all 5** |
| CIFAR-10 (4 layers) | **279,664** | 893,460 | 3.20× | **none — OS wins all 4** |
| Reconfiguration benefit | — | — | 1.000× | OS→OS→… (0 switches) |

Sparsity: **59.08%** theoretical per-MAC ReLU sparsity (246,081 MACs/image on
LeNet/MNIST); coarse group skip recovers only **1.10×** (39,132 cycles), almost
entirely from Conv1 *structural* zeros, not ReLU zeros.

---

## 2. The three candidates (one-paragraph definition)

- **Candidate A — V1 Refined.** Keep the rigid `pe_v2` + `systolic_array_v2`
  datapath *unchanged*; complete the controller gaps (IC>1, OC>8, variable
  spatial, VALID, weight-BRAM) so the *same* array runs both LeNet-5 and
  CIFAR-10, and add the OS-mode coarse group skip. No datapath change.
- **Candidate B — True Reconfigurable OS/WS Array.** Explicit spatial psum
  routing (`psum_in`/`psum_out`), stationary weights, activation
  broadcast + systolic shifting, and a controller that selects OS vs WS per
  layer. **This is already built:** it is exactly the frozen V1
  `systolic_array_v2.sv` + `pe_v2.sv` datapath (Decision 10/11), with a
  generalized (multi-layer) WS controller.
- **Candidate C — Decoupled Sparse-Aware Array.** Zero-detection with operand
  compaction and metadata (value,index) indexing, a crossbar/indirection network
  replacing the shift chain, and a dynamic matching buffer (SCNN-style,
  Parashar et al. ISCA'17). This is the *only* candidate that can convert the
  59.08% ReLU sparsity into cycle savings.

---

## 3. Candidate A — V1 Refined

### 3.1 Spatial mapping (unchanged from frozen contract)

`docs/LENET5_MAPPING_SPEC.md` §3 is the authoritative mapping and is unchanged:

- **OS:** PE **rows = output channels** (8/group, `ceil(OC/8)` groups), PE
  **columns = 8 output pixels** (reverse index `base−c`), the reduction
  `IC·K·K` serialized in time as `IC·K` row-decomposed passes. Local
  `acc += product` accumulator. Weight per-row broadcast; activation per-column
  shift chain.
- **WS:** PE **rows = 8 reduction taps** (tiled `ceil(IC·K·K/8)`), PE
  **columns = 8 output pixels**, output channels serialized (OC sweeps).
  Vertical psum cascade + bottom-to-top tile feedback; stationary weights.

The `(N, H, W, Cin, Cout, Kh, Kw)` loop nest is tiled as:
`Cout → (H', W' pixels) → Cin·Kh·Kw` in OS, and
`Cout → (H', W' pixels) → (Cin·Kh·Kw tiled by 8)` in WS — identical to the
frozen baseline.

### 3.2 Cycle models (frozen — identical to V1)

**LeNet-5** (`docs/LENET5_MAPPING_FROZEN_MANIFEST.md`):

| Layer | MACs | OS cycles | WS cycles | Optimal |
|---|---:|---:|---:|---|
| conv1 (1→6) | 117,600 | 9,184 | 28,224 | OS |
| conv3 (6→16) | 240,000 | 16,280 | 51,840 | OS |
| conv5 (16→120) | 48,000 | 15,855 | 49,200 | OS |
| fc1 (120→84) | 10,080 | 1,510 | 10,248 | OS |
| fc2 (84→10) | 840 | 192 | 900 | OS |
| **TOTAL** | 416,520 | **43,021** | 140,412 | **OS** |

**CIFAR-10** (`docs/CIFAR10_MAPPING_RESULTS.md`):

| Layer | MACs | OS cycles | WS cycles | Optimal |
|---|---:|---:|---:|---|
| conv1 (3→32) | 1,881,600 | 94,976 | 322,560 | OS |
| conv2 (32→32) | 2,560,000 | 167,760 | 518,400 | OS |
| conv3 (32→64) | 51,200 | 16,776 | 51,840 | OS |
| fc (64→10) | 640 | 152 | 660 | OS |
| **TOTAL** | 4,493,440 | **279,664** | 893,460 | **OS** |

Formulas (frozen, corrected FC): OS conv `17 + IC·K·(K+8)`/group; WS conv
`8·ceil(T/8)+10`; OS FC `ceil(OC/8)·(IC+2)+2·OC`; WS FC
`OC·(8·ceil(IC/8)+2)`. Pipeline fill/drain and tail cycles are the `+17`/`+10`
/`+2·rows` constant terms; dimension-mismatch tails are the `ceil()` partial
groups. These are the same models as V1, so Candidate A's cycle count is
**byte-for-byte the V1 baseline**.

### 3.3 Hardware cost (XCK26)

Measured V1 (hardened, `synthesis.json`): **64 DSP48E2, 9,372 LUT, 5,795 FF,
0 BRAM, 0 URAM**, WNS **+0.189 ns @ 200 MHz**. Candidate A adds controller
generality and weight storage (gaps G1–G8) but no datapath change:

| Resource | V1 measured | Candidate A estimate (analytical) | Basis |
|---|---:|---:|---|
| DSP48E2 | 64 | **64** (≤ 5.2% of 1,248) | datapath unchanged |
| LUT | 9,372 | **~13–16 K** (≤ 13%) | + IC/OC-generic addr gen, VALID predicates, 4-D weight decode, OS skip |
| FF | 5,795 | **~8–10 K** (≤ 4%) | + weight-BRAM address regs, IC line-buffer pointers |
| BRAM36 | 0 | **~3 (MNIST) / ~22 (CIFAR)** | 200×8 weights vs 79,840×8 ≈ 638 Kb ≈ 18 BRAM36 + line buffers |
| URAM | 0 | 0 | BRAM simpler for 8-bit weight streams |

**Timing:** closes 200 MHz today with +0.189 ns margin; the generalization adds
moderate address logic. The conservative research-plan estimate of **150 MHz**
remains comfortable. **Timing risk: LOW.**

### 3.4 Sparsity viability

Candidate A keeps the frozen `zero_skip` datapath. Per
`docs/SPARSE_EXECUTION_RESULTS.md`, the *only* safe skip granularity is the
**whole spatial group**, which recovers **1.10×** on LeNet (Conv1 structural
zeros only) and **≈1.00×** on CIFAR (its 3-channel RGB input has **no**
structural-zero background, so no all-zero window exists). The 59.08% ReLU
sparsity is **not** convertible at this granularity (F2: ReLU zeros are spatially
scattered). **Verdict: Candidate A does not exploit ReLU sparsity — it exploits
≈10% of Conv1's structural zeros on MNIST only.**

---

## 4. Candidate B — True Reconfigurable OS/WS Array

### 4.1 **Candidate B is already built (key finding)**

Every feature Candidate B is asked to provide is **already present in the frozen
V1 datapath**:

| Required feature | Where it already exists |
|---|---|
| Explicit spatial psum routing (`psum_in`, `psum_out`) | `pe_v2.sv` (ports + `psum_out = accumulator`) and `systolic_array_v2.sv` §6 (vertical cascade + bottom-to-top feedback) |
| Stationary weights, per-tile `weight_load` | `pe_v2.sv` weight register; array-wide `weight_load` at cycles 0/15/23/31 |
| Activation broadcasting + systolic shifting | `systolic_array_v2.sv` per-row `act_in[r]` + 7-register shift chain |
| Spatial reduction vs output-channel parallelism | OS: rows = output channels; WS: rows = reduction taps (Decision 10/11) |
| Runtime OS↔WS reconfiguration | `dataflow_mode` + `mode_commit` + FLUSH (`cnn_accelerator_v2.sv`) |

The only thing Candidate B would add over V1 is a **multi-layer WS controller**
(V1's WS controller drives only MNIST-12 Conv1). That is a controller change, not
a datapath change.

### 4.2 Cycle models

Identical to §3.2 — the datapath is the same. **Candidate B's cycle counts equal
Candidate A's and equal the V1 baseline**, because the OS/WS ranking is a property
of the *datapath topology*, not the controller. A generalized WS controller does
not move the OS/WS ratio.

### 4.3 Hardware cost

Identical to Candidate A **plus** the WS datapath and reconfig FSM that
Candidate-A (if OS-only) could drop. The WS psum addend uses the DSP48E2 **C
input** (not an extra DSP), so DSP count stays 64. The mode-muxed weight path is
a measured minor cost (it forces `BREG=0`, `PHASE2_SYNTHESIS.md` §2.1) and
contributes to the routing-dominated critical path. **Net: Candidate B ≥
Candidate A area/timing, with zero additional speedup.**

### 4.4 Sparsity viability

Same as Candidate A (coarse group skip only). The WS path's only sparsity note is
that Phase-2's coarse zero-group skip was implemented **WS-only**
(`PHASE2_ARCHITECTURE.md` §6) — but that skip operates at group granularity and
delivers the same ~1.10× (structural) result. **Verdict: no ReLU-sparsity
conversion.**

### 4.5 **Verdict: does Candidate B create an OS/WS crossover?**

**No. It cannot, and the hardware already proves it.** See §6 for the exact
mathematics; the short form is that on any R×R array with per-row weight
broadcast and per-column spatial shift, the OS/WS cycle ratio is

```
        OS/WS  ≈  (K + R) / (R·K)
```

which is **< 1 for every kernel K ≥ 2**, i.e. OS wins every *spatial*
convolution. For R = 8 and K = 5, OS/WS = 13/40 = 0.325. The reconfiguration
mechanism (`dataflow_mode`) is real and verified, but it has **nothing to select
to** — OS is optimal for every LeNet-5 and CIFAR-10 layer. Candidate B is
therefore Candidate A **with unjustified area, timing, and verification cost**.

---

## 5. Candidate C — Decoupled Sparse-Aware Array

### 5.1 Spatial mapping

Candidate C **abandons the systolic OS/WS mapping**. Instead of a fixed shift
chain and psum cascade, nonzero activations are detected, compacted into
`(value, index)` pairs, and routed to PEs through a **crossbar / indirection
network** with a dynamic matching buffer. There is no OS-vs-WS distinction: the
reduction is performed by matching nonzero input indices against the weight
indices (SCNN-style Cartesian product of nonzero terms). The
`(N,H,W,Cin,Cout,Kh,Kw)` loop nest is re-expressed as a **sparse dot-product
gather** rather than a dense convolution.

### 5.2 Cycle model (LeNet/MNIST — CIFAR sparsity is unmeasured)

From `docs/FINE_GRAINED_SPARSITY_DESIGN.md` §Cycle-level model (the only honest
numbers available):

| Mode | LeNet cycles/image | Speedup vs dense OS |
|---|---:|---:|
| Dense OS (frozen) | 43,021 | 1.000× |
| Coarse group skip | 39,132 | 1.099× |
| Fine compaction — zero overhead (upper bound) | 17,604 | **2.444×** |
| Fine compaction — ~1.3× metadata overhead | 22,885 | **1.880×** |

The 2.444× figure is the **theoretical upper bound** (per-MAC 59.08% skippable,
perfectly compacted, zero routing/index cost). The 1.880× figure includes the
`(value,index)` metadata width penalty (~2–2.75× the activation width). It does
**not** include crossbar routing delay, matching-buffer contention, or the
irregular (data-dependent) control path, all of which depress it further on FPGA.
**CIFAR-10 sparsity has not been measured** — its ReLU zero-rate and per-MAC
skippable fraction are unknown, so no CIFAR speedup is claimed here.

### 5.3 Hardware cost (XCK26)

| Resource | Estimate (analytical, order-of-magnitude) | Risk |
|---|---|---|
| DSP48E2 | **64** (compaction does not reduce multiplier count) | low |
| LUT | **~30–50 K** — the crossbar + comparator + index-decode + sparse scheduler dominate | **high** |
| FF | **~10–15 K** — metadata FIFOs (`value+index` ≈ 12 bits × depth × PEs) | medium |
| BRAM36 | **~25–35** — activation + weight + index + output buffering | medium |
| Timing | arbitrary high-fanout crossbar routing; closing 150–200 MHz is **unproven** on FPGA (SCNN's ~1.2 GHz is a 40 nm full-custom ASIC, not transferable) | **high** |

### 5.4 Sparsity viability

Candidate C is the **only** candidate that converts the 59.08% ReLU sparsity into
cycles, but the conversion is conditional and costly:

1. It requires abandoning the fixed-timing systolic array (crossbar replaces the
   shift chain) — a *different accelerator*, not a PE modification
   (`PE_V2_ARCHITECTURE_STUDY.md` F1/F3).
2. The `(value,index)` metadata and the detect+route path are a **new
   bottleneck** (F2) — the 2.44× upper bound drops to ~1.88× before any routing
   cost, and irregular sparse control adds further FPGA-specific degradation.
3. DSP48E2 is fixed-power, so gating (not compaction) saves **≈0 energy and 0
   cycles** (F1) — the cheaper "gate the zero" idea is a dead end.

**Verdict: Candidate C is the only sparsity-viable candidate, but at high
architectural and timing cost, and its headline number is unproven and
CIFAR-unquantified.**

---

## 6. The OS/WS crossover — exact mathematics

This is the decision-critical question. Define an **R×R** array (R = 8 here),
per-row weight broadcast, per-column spatial shift, OS rows = output channels,
WS rows = reduction taps.

**OS group (R output channels, R pixels):** clear (1) + `IC·K` passes ×
`(R+K)` cycles (R−1 lead-in + K taps + 1 drain) + drain (2R):
`1 + IC·K·(R+K) + 2R`.

**WS group (1 output channel, R pixels):** `R·ceil(IC·K·K/R) + c₀`, where `c₀`
≈ R+2 is the clear/lead-in/capture constant. Per R output channels, WS runs R
sweeps: `R·(R·ceil(IC·K·K/R) + c₀)`.

Taking the dominant terms (large IC, `IC·K·K` a multiple of R, c₀ negligible):

```
OS  ≈  IC·K·(R+K)
WS  ≈  R·IC·K·K
────────────────────────────
OS/WS  =  (K + R) / (R·K)          (1)
```

**Crossover condition (WS faster than OS):** `(K+R)/(R·K) > 1  ⇔  K + R > R·K
⇔  R > K·(R−1)  ⇔  K < R/(R−1)`.

For R = 8, `K < 8/7 ≈ 1.143`. Since K is an integer, **the only kernel size that
can favor WS is K = 1 (pointwise)**. Every spatial convolution (K ≥ 2) has
`OS/WS < 1`, so **OS wins for all K ≥ 2 on any R ≥ 2 square array** of this
topology. This is the structural result behind the measured 3.26×/3.20× OS
margins; it is not workload-dependent and it is not an artifact of the specific
IC/OC counts.

**The one regime where WS wins (K = 1, pointwise, with spatial extent).** For
K = 1, Eq. (1) gives OS/WS = 9/8 = 1.125 (WS nominally faster), but the
*exact* formulas bound it further. For a pointwise conv with spatial extent
`H'·W' > 1`, per 8×8 block:

```
OS  =  17 + 9·IC
WS  =  8·(8·ceil(IC/8) + 10)
```

WS < OS ⇔ `64·ceil(IC/8) + 80 < 17 + 9·IC` ⇔ `64·ceil(IC/8) < 9·IC − 63`.
For IC = 8m: `64m < 72m − 63 ⇔ m ≥ 8 ⇔ **IC ≥ 64**`.

So a crossover exists **only** at: `K = 1`, `IC ≥ 64`, `H'·W' > 1` (a
depthwise-separable / MobileNet-style **pointwise** layer). Two further
constraints close the door on the actual workloads:

1. **FC layers are K=1 with H'·W' = 1** (no spatial extent). There, the
   per-row weight broadcast prevents WS from emitting 8 distinct output neurons
   per pass (weights vary along the *column* axis), so WS-FC is one neuron/sweep
   with 7/8 columns idle: OS-FC wins by ~8×, not 1.125×. The pointwise crossover
   formula does **not** apply to FC.
2. **Neither LeNet-5 nor CIFAR-10 contains a pointwise conv with spatial extent.**
   LeNet: conv1/3/5 are K=5; fc1/fc2 are K=1, H'=W'=1. CIFAR: conv1/2/3 are K=5;
   fc is K=1, H'=W'=1.

**Definitive verdict:** an 8×8 array (or any R×R array) with per-row weight
broadcast and per-column spatial shift **cannot produce an OS/WS crossover on
LeNet-5 or CIFAR-10**. The mathematical crossover lives in a regime — pointwise
(1×1) convolution with `IC ≥ 64` and nonzero spatial extent — that is structurally
absent from both workloads (which have only K=5 spatial convs and K=1 non-spatial
FC layers). Candidate B, being exactly this array, inherits the same result and
adds nothing.

---

## 7. Candidate comparison

| Dimension | **A — V1 Refined** | **B — Reconfigurable OS/WS** | **C — Sparse-Aware (SCNN)** |
|---|---|---|---|
| Datapath change | none (rigid PE) | none (already built) | **full redesign** (crossbar) |
| DSP48E2 (of 1,248) | ~64 | ~64 | ~64 |
| BRAM36 (of 144) | ~3 (MNIST) / ~22 (CIFAR) | same | ~25–35 |
| LUT (of 117 K) | ~13–16 K | ≥ A (mode-mux + reconfig) | ~30–50 K |
| FF (of 234 K) | ~8–10 K | ≥ A | ~10–15 K |
| **Area** | LOW | LOW (slightly > A) | **HIGH** |
| **Timing risk** | LOW (closes 200 MHz, +0.189 ns) | LOW (same) | **HIGH** (crossbar routing) |
| **Real speedup (LeNet)** | 1.00× dense; **1.10×** coarse-skip | **1.00×** (reconfig buys 0) | 1.88–2.44× (theoretical, unproven) |
| **Real speedup (CIFAR)** | 1.00× (no structural zeros) | 1.00× | unmeasured (sparsity unknown) |
| **Sparsity exploitation** | structural zeros only (~10%) | structural zeros only | **ReLU sparsity (theoretical)** |
| **OS/WS crossover** | n/a (OS-only) | **none (proven)** | n/a (no OS/WS) |
| Path to bitstream | short (exists + generalize) | short (exists) | long (new everything) |

"Real speedup" is honest: A's dense speedup over the V1 baseline is **1.00×**
(identical datapath); its only gain is the coarse group skip (1.10× on MNIST,
≈1.00× on CIFAR) plus the *ability to run CIFAR*, which V1's controller cannot
yet do. B's reconfiguration gain is **1.000×** (Step 6.4 measured 0 switches).
C's numbers are theoretical upper bounds with unproven FPGA closure.

---

## 8. Selected architecture recommendation

**Recommend Candidate A — V1 Refined, OS-only.**

Rationale:

1. **The reconfigurable-dataflow claim has no measured basis.** OS wins every
   LeNet-5 and CIFAR-10 layer; the crossover is mathematically excluded (§6).
   Retaining the WS path (Candidate B) adds area, a mode-mux timing cost, and a
   reconfiguration FSM for a mode that is **never selected** — pure overhead.
2. **Candidate A is the shortest, lowest-risk path to a Kria KV260 bitstream.**
   The datapath is frozen and bit-exact-verified; the array alone closes 200 MHz
   at WNS +2.861 ns; the full system closes at +0.189 ns; an AXI4-Lite top
   (`cnn_top_axi`) already synthesizes (WNS +0.202 ns). The remaining work is
   controller generality (gaps G1–G8) and weight-BRAM, both localized to the
   controller/input-feed.
3. **Candidate C's sparsity speedup is real but belongs to a different project.**
   It requires abandoning the systolic datapath for a crossbar (SCNN-style) at
   high FPGA timing risk, and its CIFAR sparsity is unquantified. It is not a
   reasonable *next* step for this accelerator; it is a separate research
   question, and committing to it now would forfeit the working, verified V1
   baseline.

**Concrete path to bitstream (deferred to a later step, not executed here):**

1. **Freeze an OS-only datapath** — remove the WS psum cascade/mode-mux and the
   reconfig FSM from the *controller* (the frozen `pe_v2`/`systolic_array_v2`
   remain untouched as evidence). This recovers the DSP BREG and reduces the
   routing-critical control fan-out.
2. **Close controller gaps G1–G8** (IC>1, OC>8, variable spatial, VALID
   predicates, 4-D weight decode, weight BRAM) so both LeNet-5 and CIFAR-10 run.
3. **Generalize the OS coarse group skip** (currently WS-only) for MNIST's
   structural zeros; skip it for CIFAR (no benefit).
4. **Re-verify** the generalized cycle model (`1 + IC·K·(P+K) + 2·rows` per
   `FC_SCHEDULE_VALIDATION.md` §6) against the RTL at IC>1/OC>8 — this is the
   open "unverified generalization" the mapping flagged.
5. **Re-synthesize** at a conservative 150 MHz and re-run the AXI integration.

**Rejected:** Candidate B (reconfiguration with no crossover to exploit);
Candidate C *now* (sparsity speedup is real but architecturally incompatible and
unproven on FPGA; revisit only as a standalone sparse-accelerator question after
the OS-only V2 is delivered and CIFAR sparsity is measured).

---

## 9. Major technical risks to resolve before RTL freezing

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **IC>1 / OC>8 cycle model is an unverified generalization** (the only RTL-verified conv number is 82 cyc at IC=1/K=5) | high | cycle-accurate sim of the generalized schedule before any claim |
| R2 | **Partial-column-group timing** (P<8) is overcounted by the uniform formula | medium | apply §6 per-group formula; confirm against RTL |
| R3 | **Weight storage (G5)**: CIFAR needs ~78 KB weights vs V1's ~1.2 KB register file | medium | BRAM36 (≈18) + 4-D decode; verify stream bandwidth |
| R4 | **VALID-conv predicates (G6)** replacing SAME-pad zero-injection | medium | re-parameterize row/col predicates; directed boundary tests |
| R5 | **Timing margin erosion** at 200 MHz as address-gen generality grows | medium | target 150 MHz; pipeline controller→array control fan-out |
| R6 | **OS coarse-skip generalization** (Phase-2 skip is WS-only) | medium | OS needs 8-word group emission; verify skip correctness |
| R7 | **CIFAR sparsity is unmeasured** — any CIFAR sparsity argument is currently unsupported | high | measure CIFAR ReLU sparsity before any sparse claim |
| R8 | **FC OS weight transpose (G7)** — 4-D→2-D reshape for FC | low | index reshape, not a datapath change |

R7 is the most consequential open question for any future sparsity work: the
59.08% figure is LeNet/MNIST-specific, and CIFAR's RGB input removes the
structural-zero component that made coarse skipping work at all on MNIST.

---

## 10. Deliberately not done (scope)

- No RTL written, no V1 RTL/testbench modified, no frozen artifact modified.
- No publishability or speedup claim (all "speedups" are measured or
  explicitly-labeled theoretical bounds).
- No Python model written in this step.
- No new architecture recommended beyond "Candidate A OS-only"; the WS-drop and
  sparse-accelerator questions are flagged for later steps, not decided here.

**Headline:** the reconfigurable-dataflow hypothesis is **mathematically closed**
— an 8×8 per-row-broadcast / per-column-shift array cannot favor WS for any
spatial (K≥2) convolution, and neither workload contains the pointwise
(IC≥64, spatial) regime where a crossover could exist. The evidence directs the
project toward an **OS-only V1 refinement (Candidate A)**, with fine-grained
sparsity (Candidate C) correctly scoped as a separate, higher-risk accelerator
problem.
