# CNN Accelerator — Research Readiness Plan

> **Status:** PLAN — RECOMMENDATIONS PENDING TEAM ADOPTION (2026-08-20).
>
> This document is a **research and engineering plan**, not a specification. It
> does not by itself change any recorded decision. Where it recommends
> superseding an existing decision (notably Decision 1), that change takes effect
> only when it is recorded in `docs/PROJECT_STATE.md` by explicit team agreement.
>
> **No RTL, testbench, specification, or Git history was modified in producing
> this document.** All repository facts are cited to `file:line`. All external
> facts carry URLs. Derived arithmetic is shown so it can be independently
> checked.

Label convention follows `docs/specs/PE_SPEC.md` §2.4: **DECISION** (recorded in
`PROJECT_STATE.md`), **RESEARCH FACT** (external reference), **RECOMMENDATION**
(this document's proposal, not yet binding), **OPEN DECISION** (unresolved).

---

## Context — why this document exists

The project has a verified V1 (Output-Stationary) and V2 (Weight-Stationary) 8×8
INT8 systolic array on `xck26-sfvc784-2LV-c`, closing 200 MHz. It has **no** real
data, no pretrained weights, no quantization pipeline, no Python reference, no
AXI interface, no PS integration, and no actual runtime reconfiguration.

The objective is a technically defensible, reproducible, real-data CNN
accelerator experiment ready to execute on the KV260 with minimal board-time
debugging and sufficient evidence for a serious major-project / research-paper
submission.

Eight parallel investigations were run to produce this plan: CNN/dataset
selection, hardware/architecture mapping, runtime reconfiguration, KV260 system
integration, quantization/numerics, verification/golden model, benchmark design,
and an adversarial reviewer audit. This document is the reconciled result.

---

## 1. Executive recommendation

Four findings drive everything. The first three are bad news; the fourth is the
paper.

### F1 — The novelty claim as currently framed is prior art. FATAL.

**RESEARCH FACT.** "Runtime-reconfigurable OS/WS systolic array" is published:

- **ReSA** (ACM TACO 2024) — a reconfigurable systolic array supporting input-,
  weight-, and output-stationary dataflows on one array, with per-tensor dataflow
  switching.
- **RIFT** (DATE 2026) — a **single-bitstream, runtime-adaptive FPGA
  accelerator** whose PE array morphs between weight-stationary and
  output-stationary modes by updating control registers. This is our exact claim,
  on FPGA, this year.
- Also: MAERI (ASPLOS'18), Eyeriss v2 (JETCAS'19), FlexFlow (HPCA'17),
  Planaria (MICRO'20), AdaFlow (DATE'22), TRINE (2026).

**The "we invented reconfigurable dataflow" framing must be deleted.** A paper
whose title claim is already in the literature is rejected before the first
figure is read.

### F2 — Runtime reconfiguration does not exist today.

**REPOSITORY FACT.** `rtl/common/input_feed_v2.sv:245` is
`assign dataflow_mode = 1'b1;`. There is no register, no AXI interface, no
handshake, and no flush protocol anywhere in `rtl/`. V1's OS schedule
(`input_feed.sv`, 82 cyc/group, per-column drain) and V2's WS schedule
(`input_feed_v2.sv`, 42 cyc/group, per-row drain) live in two separate,
non-coexisting controllers.

A mode mux that no software can reach is not a reconfigurable accelerator.

### F3 — The numerical format is broken for real data.

**REPOSITORY FACT + derived arithmetic.** Decision 1's fixed scale 1/256 gives a
real range of [−0.5, +0.49609375]. Standard MNIST normalization (mean 0.1307,
std 0.3081) maps input [0,1] to [−0.424, +2.822].

Clip point: `(x − 0.1307) / 0.3081 = 0.4961` → `x = 0.2835`.

**Every pixel with raw intensity ≥ 73/255 saturates to +127 — 183 of 256
intensity levels collapse to a single value.** MNIST digits are anti-aliased with
saturated stroke cores, so the informative stroke pixels are flattened. The
network is effectively fed a thresholded binary image.

The good news: **fixing this requires zero RTL change** (§5).

### F4 — THE RESULT: V1 (OS) is 3.07× faster than V2 (WS) on Conv1.

**SIMULATION-DERIVED.** This inverts the project's expectation and is the
strongest asset the project has.

| | Groups | Results/group | Cyc/group | Total cycles | Latency @200 MHz | Sustained MAC/cyc | Array util |
|---|---|---|---|---|---|---|---|
| **V1 (OS)** | 72 = 24×3 | 48 = 6 ch × 8 px | 82 | **5,904** | 29.5 µs | 14.63 | 22.9% |
| **V2 (WS)** | 432 = 72×6 | 8 = 1 ch × 8 px | 42 | **18,144** | 90.7 µs | 4.76 | 7.4% |

Cross-check: 72 × 48 = 432 × 8 = 3,456 Conv1 outputs. Both schedules close
exactly against the layer's output count.

**Mechanism:** V1 holds the 6 output channels **in parallel on PE rows**; V2
**serializes** them (6 sweeps, `SYSTOLIC_ARRAY_V2_SPEC.md` §4.2). WS pays 6× for
channel serialization and wins back only ~2× from its shorter group.

### Recommendation

**RECOMMENDATION.** Stop presenting reconfiguration as the novelty. Present the
**like-for-like dataflow measurement** — same 64 PEs, same DSP48E2 primitives,
same clock target, same workload, same independent golden model — which the
literature genuinely lacks because published OS/WS figures are cross-chip and
confounded by different processes, array sizes, and clocks.

Reconfiguration becomes the *apparatus* that makes the measurement fair, not the
claim itself.

**Working title:** *"Output-Stationary vs. Weight-Stationary on Identical
Silicon: A Verified Reconfigurable 8×8 INT8 Systolic Array on AMD Kria KV260."*

### The integrity condition (non-negotiable)

V2's 42-cycle group is an explicitly **un-pipelined correctness-first baseline**
(`SYSTOLIC_ARRAY_V2_SPEC.md` §10 item 6: "group-boundary pipelining — NOT
ADOPTED"). V1's 82-cycle group is likewise un-pipelined (`INPUT_FEED_SPEC.md` §7;
the proven-safe 78-cycle overlap is CANDIDATE only and not implemented).

Reporting 3.07× without disclosing this compares two *controller
implementations*, not two *dataflows*. **Ablation A1 (pipelined V2) is therefore
BLOCKING, not optional.** If pipelining V2 closes the gap, that is the finding
and it is reported as such.

---

## 2. Selected CNN + justification

**RECOMMENDATION — PRIMARY: LeNet-5 (28×28 variant, 6/16/120/84/10).**

The RTL does not merely suit LeNet-5 — it *is* LeNet-5 Conv1. Both controllers
hardcode `IMG_H = IMG_W = 28`, `K = 5`, `IC = 1`, `OC = 6`, output 24×24 valid
(`input_feed.sv:68-72`, `input_feed_v2.sv:63-65`). No other common CNN has a
first layer of "1-channel 28×28, 5×5 valid, 6 output channels."

| # | Layer | Type | IC | OC | K | In H×W | Out H×W | MACs | Params |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Conv1 | conv | 1 | 6 | 5 | 28×28 | 24×24 | 86,400 | 156 |
| 2 | Pool1 | maxpool 2×2 | 6 | 6 | — | 24×24 | 12×12 | 0 | 0 |
| 3 | Conv2 | conv | 6 | 16 | 5 | 12×12 | 8×8 | 153,600 | 2,416 |
| 4 | Pool2 | maxpool 2×2 | 16 | 16 | — | 8×8 | 4×4 | 0 | 0 |
| 5 | FC1 | linear | 256 | 120 | — | — | — | 30,720 | 30,840 |
| 6 | FC2 | linear | 120 | 84 | — | — | — | 10,080 | 10,164 |
| 7 | FC3 | linear | 84 | 10 | — | — | — | 840 | 850 |

**Total ≈ 281,640 MACs, 44,426 params. Conv1 = 30.7%; Conv1+Conv2 = 85.2%.**

**Why not CIFAR-10 or ImageNet models.** For ResNet-20 (CIFAR-10), Conv1 is
**1.08%** of total MACs; for MobileNetV2 (ImageNet), **3.6%**. A "we accelerate
the convolution layers" claim is only non-vacuous on LeNet-5. The alternatives
also cost real RTL work (IC=3, 3×3 kernels, padding, 32×32 or 224×224 inputs —
days to weeks per §6) and abandon the exact-fit geometry the project already has.

**Is LeNet-5/MNIST defensible in 2026?**

- As an **accuracy / ML result: no.** MNIST is saturated (~99%) and LeNet-5 is a
  1998 network. Framed as a classifier result it reads as a toy.
- As an **FPGA micro-architecture workload: yes.** FINN (FPGA'17, top-tier) used
  MNIST/CIFAR/SVHN, and the systolic-array literature explicitly benchmarks
  LeNet-5's first layer (Yang et al., Springer 2018). Reviewers of FPGA
  *accelerator* papers accept small real networks when the contribution is the
  datapath, not the model.

**The framing carries it, not the model.** The paper never claims an ML result.

**RECOMMENDATION — BACKUP / SECOND DATASET: identical geometry on
Fashion-MNIST.** Zero RTL change, MIT-licensed, harder (LeNet-5 ≈ 90%), and it
pre-empts both the "MNIST is solved" and the "MNIST has no licence" objections.
Report it **alongside** MNIST, not instead of it.

---

## 3. Dataset + justification

| | MNIST | Fashion-MNIST |
|---|---|---|
| URL | http://yann.lecun.com/exdb/mnist/ | https://github.com/zalandoresearch/fashion-mnist |
| Licence | none stated (NIST SD-1/SD-3 derived; universal research use) | **MIT** |
| Split | 60k train / 10k test, 28×28 greyscale, 10 classes | identical |
| Preprocessing | `Normalize(0.1307, 0.3081)` | `Normalize(0.2860, 0.3530)` |
| LeNet-5 accuracy | ~99% | ~90% |

Both are loaded via `torchvision.datasets` (pinned 0.16.2, per
`python/requirements.txt`).

**Data hygiene:**

- Raw archives → `data/raw/`, which **must be added to `.gitignore`** (it
  currently has no `data/` rule at all).
- Generated vectors and golden outputs → `data/vectors/`, `data/golden/` —
  **committed** (a few hundred bytes to ~14 KB each, text, and required for a
  clean-clone reproduction).
- Calibration samples are drawn from the **train** split; evaluation from the
  **test** split. The two never mix.

---

## 4. Pretrained checkpoint + authoritative source

**RESEARCH FACT — honest negative finding, verified:** *no authoritative,
publicly downloadable pretrained LeNet-5 checkpoint exists.*

- `torchvision` ships **no LeNet** at all (only ResNet/VGG/MobileNet/SqueezeNet
  and similar). https://docs.pytorch.org/vision/stable/models.html
- **The BVLC Caffe model zoo does NOT host LeNet-MNIST.**
  `lenet_iter_10000.caffemodel` is *generated* by running
  `examples/mnist/train_lenet.sh`; it is not downloadable from Berkeley.
  https://github.com/BVLC/caffe/blob/master/docs/model_zoo.md
- `activatedgeek/LeNet-5` (a faithful LeCun reimplementation) contains **no
  checkpoint file** — training code only — and uses 32×32 padded input, not
  28×28. https://github.com/activatedgeek/LeNet-5
- Community checkpoints exist (`SunnyHaze/LeNet5-MNIST-Pytorch`,
  `ChawDoe/LeNet5-MNIST-PyTorch`) but none is authoritative or versioned.

**RECOMMENDATION — resolution, which is *more* reproducible than any third-party
`.pth`:**

1. **Train in-repo** with a pinned, seed-fixed, deterministic script
   (`python/model/train_lenet5.py`; ~1–2 min on CPU; 98.8–99.2% test accuracy).
   Publish the checkpoint, its **SHA-256**, the exact script, and the environment
   lock.
2. **Cross-check against a public checkpoint:** ONNX Model Zoo MNIST (**MIT**,
   maintained by the ONNX community / LF AI & Data, validated top-1 error 1.1%),
   https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
   (`model/mnist-12.onnx`).

   **Caveat that must be stated, not glossed:** its Conv1 has **OC = 8**, not 6,
   so it is a **secondary reference**, not a drop-in replacement.

A reviewer demanding "a pretrained checkpoint" is really demanding
*reproducibility*. A seed-pinned script plus a published hash delivers that more
strongly than an unversioned third-party blob.

---

## 5. Quantization plan

**RECOMMENDATION — Decision 1 (fixed scale 1/256) is superseded. The PE RTL does
not change.**

Adopt symmetric per-channel-weight / per-tensor-activation INT8 quantization,
zero-point 0 (Jacob et al. 2018; gemmlowp; Krishnamoorthi 2018).

**Why no RTL change is needed.** The integer dot product our hardware computes is
*already* the exact kernel of standard INT8 quantization. Jacob et al. §2.1/§3
and the gemmlowp quantization document both establish that the multiplier
`M = S₁S₂/S₃` and the zero-point corrections live **outside** the accumulation
loop. Only the *scale interpretation* — which lives outside the PE — was wrong.

| Item | Value |
|---|---|
| Weights | per-output-channel symmetric, `S_w[c] = max\|w_c\| / 127`, Z = 0 |
| Activations | per-tensor symmetric, `S_a = max\|a\| / 127`, Z = 0 |
| Quantize | `q = clip(RNE(x / S), −128, 127)`; RNE = round-half-to-even |
| Product / accumulator | 16-bit / 32-bit signed — **unchanged** |
| Bias | int32, scale `S_b = S_a · S_w[c]` |
| Requantize | `M = S_a·S_w[c]/S_out = 2^−n · M0`, `M0 ∈ [0.5,1)`; int32 multiplier `m = RNE(M0 · 2³¹)`; arithmetic shift `31+n`; round-half-away-from-zero; saturate to int8 |
| Where | **PS software** — no RTL requantization unit |
| Calibration | 1,024 samples from the **train** split, MinMax observer |
| Tooling | NumPy primary (bit-exactness); `torch.ao.quantization` FX cross-check |

**Environment compatibility.** All of the above works on the pinned
PyTorch 2.1.2+cpu / torchvision 0.16.2 / NumPy 1.26.4 environment recorded in
`PROJECT_STATE.md`. **No upgrade is required.** (Avoid `quantize_pt2e`, which is
still prototype in 2.1.)

**Per-channel quantization is free here.** V2 WS serializes output channels (one
per sweep, `SYSTOLIC_ARRAY_V2_SPEC.md` §4.2), so a per-channel scale is a simple
lookup in the software drain. The architecture is *ideally* suited to per-channel
weight quantization.

**Signed-only PE.** Post-ReLU activations encoded symmetrically use `q ∈ [0,127]`
— 7 effective bits, ~6 dB lower SQNR than uint8. Negligible at MNIST/CIFAR scale.
Full-range uint8 remains available at zero RTL cost by storing `q − 128` and
folding `−128 · Σ q_w` into the bias (the standard gemmlowp offset fold); the
correction term is a per-output-channel constant, hence precomputable.

**Accumulator width — Decisions 2/3 CONFIRMED.**
`max |product| = 128 × 128 = 16,384 = 2¹⁴`;
`N_max = floor((2³¹ − 1) / 16,384) = 131,071` accumulation terms.

| Layer | N = K·K·IC | Worst-case sum | Bits |
|---|---|---|---|
| LeNet-5 Conv1 | 25 | 4.10 × 10⁵ | 19 |
| LeNet-5 FC1 | 400 | 6.55 × 10⁶ | 23 |
| 3×3×512 | 4,608 | 7.55 × 10⁷ | 27 |

32 bits is ample in every realistic case. **RECOMMENDATION:** sharpen the
wording in `PE_SPEC.md` §7.6 from "overflow is impossible" to "overflow is
impossible up to 131,071 accumulation terms per PE, which no realistic layer
approaches."

---

## 6. V1 / V2 mapping

**Verdict: (C) a useful subset of layers today — and with controller-only work,
(B) all convolution layers.** The PE and both arrays stay frozen in every
scenario below.

| Layer | V1 cycles | V2 cycles | Change needed |
|---|---|---|---|
| Conv1 (1→6, 5×5, 28²) | **5,904** | **18,144** | none (hardcoded) |
| Conv2 (6→16, 5×5, 12²) | ~16,280 | ~20,736 | IC>1 + OC>8: controller-only |
| FC (N ≤ 400) | ~54,255 | ~49,200 | controller-only; 7 of 8 columns idle |
| 3×3 pad=1, IC=3 | — | — | + padding (see the bug below) |

### The IC > 1 question — definitive

**V2's 8-row cascade plus the always-on bottom-to-top ring
(`systolic_array_v2.sv:108-116`) reduces `K·K·IC` terms by chaining
`ceil(K·K·IC / 8)` tiles. There is no architectural correctness limit, and the
array and PE need zero change.**

Evidence: the array contains **no tile counter**. Tile count is purely a
controller constant (`weight_load` at cycles 0/15/23/31,
`input_feed_v2.sv:247-249`; `load_tile` decode at `:262-269`). The general result
lands at `accum[7]` at cycle `8T + 9`. Partial final tiles already work — Conv1's
tile 3 is 1 tap plus 7 zero-weight pass-through rows (`input_feed_v2.sv:313`).

V1 with IC > 1 is likewise controller-only: the PE accumulator already sums
across passes (`pe.sv:81-88`), so IC > 1 simply lengthens the pass sequence from
`K` to `K·IC`.

**Caveat on "controller-only":** the array and PE are untouched, but
`input_feed_v2.sv` itself needs real rework (variable tile count, 4-D weight
decode `W[oc][ic][ky][kx]`, IC-channel buffering). Estimated 1.5–2.5 weeks
including verification.

### ⚠ Latent padding bug — record it now

`input_feed_v2.sv:226-229` **clamps** out-of-range columns to [0,27] and reads the
edge pixel. The only zero-gating is the temporal `active` window (`:323`); there
is **no column-validity gate**.

- **For the current valid Conv1 this is harmless** — verified: during every
  *active* cycle the accessed column `px = base − 8 + s − 5·k_y` lies within
  [0,27] (min `base−7+k_x`, max `base+k_x`, `base ∈ {7,15,23}`, `k_x ∈ 0..4`).
  The clamp only fires on inactive cycles whose read is discarded, and exists to
  prevent out-of-range BRAM addressing.
- **For any padded convolution it becomes a real bug:** out-of-range columns
  would occur during *active* cycles, and the RTL would silently emit
  edge-replicated pixels instead of zeros — wrong results with no error
  indication.

**Fix (when padding is added):** a `col_valid` / `row_valid` predicate, then
`act_out = (active && col_valid && row_valid) ? stream : 0`. Estimated 0.5–1 week
including testbench. **This must be documented even though padding is not
currently in scope.**

### Post-processing → PS software

**RECOMMENDATION.** Bias add, requantization, ReLU, pooling, and the FC layers
run on the ARM A53. Rationale: zero RTL risk, preserves the frozen V1/V2/PE-v2
boundaries, keeps per-channel scales flexible, and still yields a real end-to-end
top-1 accuracy number.

An on-chip post-processing unit (estimated ~1 DSP + ~200 LUT + ~150 FF, plus
~1 BRAM18 for a pooling line buffer) is **deferred**. The current +0.600 ns WNS
does not invite optional logic on the critical path.

---

## 7. Runtime reconfiguration architecture

**RECOMMENDATION — Option A: one dual-mode controller driving ONE
`systolic_array_v2`.**

**Rejected alternatives:**

- *Two arrays muxed* (128 DSP48E2 = 10.3% of the 1,248 available). This is
  duplication, not reconfiguration, and a reviewer will say so in one sentence.
- *Reusing the frozen `input_feed.sv` against array-v2.* Its **per-column**
  `result_req` (`input_feed.sv:214-217`) would be misinterpreted as a **per-row**
  select by array-v2 (`systolic_array_v2.sv:136`). It cannot drive array-v2
  verbatim.

**No frozen module changes.** `pe.sv`, `systolic_array.sv`, `input_feed.sv`,
`pe_v2.sv`, and `systolic_array_v2.sv` are all untouched. `input_feed_v2.sv`'s
hardcoded `dataflow_mode` output is simply left unused — the new controller
drives the array's `dataflow_mode` directly.

```
cnn_top                        rtl/top/cnn_top.sv              NEW
├── axi_lite_ctrl              rtl/common/axi_lite_ctrl.sv     NEW
├── reconfig_ctrl              rtl/common/reconfig_ctrl.sv     NEW
│   ├── os_schedule            rtl/common/os_schedule.sv       NEW (per-ROW drain)
│   └── input_feed_v2          rtl/common/input_feed_v2.sv     EXISTING, frozen
├── systolic_array_v2          rtl/common/systolic_array_v2.sv EXISTING, frozen
└── weight_store               rtl/common/weight_store.sv      NEW
```

**OS-on-array-v2 correctness.** In OS mode `pe_v2` is bit-identical to the V1 PE
(`pe_v2.sv:114-117` vs `pe.sv:87`), the shift chain is V1-identical
(`systolic_array_v2.sv:70-93`), and `psum_in` is never read (only the `else`
branch is taken), so the always-on ring at `systolic_array_v2.sv:111` is
functionally dead in OS mode. All 64 accumulators therefore hold correct OS
results; they must simply be read **per row** (`result_req[r]`, r = 0..5 for six
channels) instead of per column. A useful side effect: the OS drain shrinks from
16 cycles to ~7, so an OS group becomes ~72–74 cycles rather than 82. **This
changes the V1 baseline and must be re-measured, not assumed.**

### Flush analysis (register by register)

| State | Location | Cleared by `accum_clear`? | Action at switch |
|---|---|---|---|
| accumulator ×64 (PREG) | `pe_v2.sv:109-118` | **yes** (`:112`) | 1 cycle, all 8 ring stages **in parallel** |
| product ×64 (MREG) | `pe_v2.sv:89-98` | **yes** (`:92`) | same cycle |
| weight ×64 (BREG) | `pe_v2.sv:76-81` | no | not needed — `weight_load` overwrites before first use |
| result_out ×64 | `pe_v2.sv:127-132` | no | not needed — overwritten by the new mode's `result_request` |
| **shift chain ×56** | `systolic_array_v2.sv:56,70-80` | **NO** | **7 cycles of `act_in = 0`** |
| input_feed BRAM / counters | `input_feed_v2.sv:172-182, 92-158` | no | leave BRAM; counters reset by `start` |

**Key correction to a natural assumption:** the psum ring does **not** need 8
drain cycles. `accum_clear` fans out array-wide (`systolic_array_v2.sv:46`) and
every PE zeroes its PREG on the same edge. What `accum_clear` *misses* is the
56-register activation shift chain, which needs 7 cycles of zero input.

**Total flush = 8 cycles ≈ 40 ns @ 200 MHz.** A full `rst` is **not** required
and must **not** be used — it would destroy the loaded image BRAM and the weight
store, forcing a costly reload, and `rst` is a power-on-only event per
`INPUT_FEED_V2_SPEC.md` §11.

### Switch protocol

1. PS polls `STATUS.IDLE` (or waits for the DONE interrupt).
2. PS writes `DATAFLOW_MODE`.
3. PS writes `CONTROL.MODE_COMMIT`.
4. HW: if not IDLE → set `STATUS.ERROR` and **reject**; else latch
   `mode_pending → mode_active`, assert `MODE_SWITCHING`, enter FLUSH.
5. FLUSH (8 cycles): cycle 0 asserts `accum_clear` and drives `act_out = 0`;
   cycles 1–7 hold `act_out = 0`. Both schedule generators held idle.
6. HW: clear `MODE_SWITCHING`, set IDLE, pulse SWITCH_DONE IRQ.
7. PS writes `CONTROL.START`.

**Hardware interlock (structural, not software discipline):** `MODE_COMMIT` is
honoured only in IDLE; the array sees only the registered `mode_active`, so a
mid-run AXI write can touch only `mode_pending` and can never glitch the array;
and the two schedule generators are mutually reset by `mode_active`, so at most
one can drive `weight_load` / `result_req` at any time.

**Weight reload.** OS and WS read the *same* 150-weight store with different
address decode — OS: `w_in[r] = W[ch=r][ky][kx]`, reloaded every tap; WS:
`w_in[r] = W[ch][k = 8t+r]`, reloaded per tile. No pre-reordering and no separate
memory are needed.

### Encoding conflict — resolved

**RECOMMENDATION.** Adopt **0 = OS, 1 = WS**, matching `PE_SPEC.md` §13.2 and
wired 1:1 to the array's `dataflow_mode`. The reset default of 0 then yields V1 OS
behaviour, and there is no inversion gate to get wrong. The architecture
roadmap's opposite encoding (`0 = WS, 1 = OS`) is a PDF artifact and is
explicitly not authoritative for detail decisions per `CLAUDE.md`. Document this
in the register description.

### ⚠ Timing risk — honest assessment

The current V2 full-system worst setup path
(`build/v2_input_feed/06_worst_setup_path.rpt`) is already:

```
RAMB18E2 (1.266 ns) → LUT6 (0.220) → LUT4 (0.056) → fo=70 net (0.914)
→ DSP_A_B_DATA (0.258) → DSP_PREADD (0.114) → DSP_MULT (0.700)
= 4.315 ns, 5 logic levels, WNS +0.600 ns, WHS +0.067 ns
```

This is exactly the activation path. A dataflow mux adds **one more LUT level to
that same bus** and may raise the already-hot fanout-70 net. Honest estimate:
new WNS ≈ **+0.3 to +0.4 ns** — it closes, but with no safety margin, and hold is
already only +0.067 ns.

**RECOMMENDATION (adopted mitigation):**

1. **Register `mode_active`** (already required for the interlock).
2. **Pipeline `act_out` / `w_out` at the controller→array boundary.** This is
   consistent with the controller's existing one-cycle address look-ahead
   (`input_feed_v2.sv:194-239`), breaks the path into `BRAM→LUT→REG` and
   `REG→DSP`, costs ~64+ FFs, and leaves the frozen array untouched.
3. **Target 150 MHz for the reconfigurable system**, while continuing to report
   200 MHz for the frozen V1/V2 blocks. At 150 MHz, V2 Conv1 is ~121 µs — which
   is irrelevant for a correctness-and-overhead paper.
4. **Re-check hold** after any pipeline change.

**Do not ship "+0.3 ns at 200 MHz" and call it closed.**

### AXI4-Lite register map

| Offset | Name | Access | Fields |
|---|---|---|---|
| 0x00 | `VERSION` | RO | [31:16] ID, [15:8] major, [7:0] minor |
| 0x04 | `CONTROL` | W1S | [0] START, [1] SOFT_RESET, [2] MODE_COMMIT, [3] ABORT |
| 0x08 | `STATUS` | RO/W1C | [0] IDLE, [1] BUSY, [2] DONE, [3] ERROR, [4] MODE_SWITCHING |
| 0x0C | `DATAFLOW_MODE` | RW | [0] 0 = OS, 1 = WS (pending) |
| 0x10 | `MODE_STATUS` | RO | [0] ACTIVE_MODE, [1] PENDING_VALID |
| 0x14 | `GEOMETRY` | RW | K_H, K_W, IMG_H, IMG_W |
| 0x28 | `IRQ_ENABLE` | RW | [0] DONE, [1] ERROR, [2] SWITCH_DONE |
| 0x2C | `IRQ_STATUS` | RO/W1C | pending interrupts |
| **0x30** | **`CYCLE_RUN`** | RO | **per-run counter, START→DONE, frozen at DONE** |
| 0x34/0x38 | `CYCLE_FREERUN_LO/HI` | RO | free-running 64-bit cycle counter |
| 0x3C | `FLUSH_CYCLES` | RW | quiescence cycles (default 8) |
| 0x40 | `WEIGHT_LOAD_CTRL` | RW | [0] RELOAD, [1] AUTO_ON_START |
| 0x1000 | `IMAGE_BASE` | W | 784-byte auto-increment window |

`CYCLE_RUN` is **the paper's latency instrument** — a per-run counter, not a
free-running one, so measured per-mode latency carries no software timing noise.

---

## 8. KV260 system architecture

**RECOMMENDATION: AXI4-Lite (control + image + weights) + AXI4 BRAM Controller
(results). No DMA, no DDR data buffers.**

The workload is 784 B in, 150 B of weights, 13,824 B out. AXI DMA (PG021) exists
to move megabyte-scale video streams; here it would add a Linux DMA driver,
scatter-gather descriptors, and `Xil_DCacheFlushRange` / `dma_map_single`
coherency correctness burden for no benefit. Because all data lives in PL BRAM,
there is **zero cache-coherency surface** — the single biggest risk removal
available.

```
PS (A53 ×4, Ubuntu 22.04)   M_AXI_HPM0_FPD @0xA400_0000   pl_clk0=100MHz   pl_ps_irq0
        │                                                        │
        ▼                     AXI SmartConnect (PG247)           ▼
   ┌────┴────────────┬──────────────────┐      Clocking Wizard (PG065) → 200/150 MHz
   ▼                 ▼                  │      proc_sys_reset ×2 (PG164)
 AXI BRAM Ctrl   cnn_accel_wrapper      │        aresetn (active-low)  → AXI
 (PG078, AXI4)   (AXI4-Lite slave)      │        reset   (active-high) → our rst
   │                 │                  │
 BMG 4096×32     weight regfile 150×8 (combinational read)
 (PG058)         image dist-RAM 1024×8 (ASYNCHRONOUS read — required)
   ▲                 │
   │                 ▼
   └──── cnn_top: reconfig_ctrl → systolic_array_v2 (64 × pe_v2)
```

**⚠ Critical constraint.** The image store **must be distributed RAM with
asynchronous read**. `input_feed_v2.sv:174-182` presents `img_addr` and samples
`img_data` in the *same* cycle. A BRAM there would silently break the design.

**Reset conversion.** Our RTL uses synchronous **active-high** `rst` (Decision 5);
AXI uses active-low `aresetn`. `proc_sys_reset` (PG164) emits **both**
`peripheral_aresetn` and `peripheral_reset`, already clock-synchronized. No extra
synchronizer is needed. The 200/150 MHz core domain gets its own `proc_sys_reset`
gated by the MMCM `locked` signal.

**Clocking.** AXI/control domain = 100 MHz (PS `pl_clk0`; lowest risk). Core =
200 MHz nominal, **150 MHz for the reconfigurable system** per §7. Note the KV260
carrier has **no discrete PL oscillator** — the PL clock must come from the PS
`pl_clk0`, which is the portable choice.

**Address map** (UG1085 Ch. 10): control at `0xA400_0000` (64 KB), results at
`0xA401_0000` (16 KB), both inside `M_AXI_HPM0_FPD`
(`0xA400_0000`–`0xAFFF_FFFF`).

**Software stack — RECOMMENDATION: Kria native Ubuntu 22.04 + device-tree
overlay + `xmutil` firmware package + userspace C application over UIO.** It is
the AMD-certified path for KV260, provides `xlnx-platformstats` for power
measurement, and every artifact (`.bit`, `.dtbo`, `shell.json`, cross-compiled
app) is buildable before the board is touched.

- *Fallback:* bare-metal Vitis standalone — deterministic timing, ideal for the
  very first `done`-bit bring-up.
- *Demos only:* PYNQ v3.0.1 via `Xilinx/Kria-PYNQ` (install-on-Ubuntu, not a
  standalone image). Excellent for figures; Python overhead disqualifies it as
  the timing primary.

**Power measurement — verified reality.** The KV260 carrier has a **single INA260
at I²C address `0x40`, in the SOM power path**. The K26 SOM PMICs
(DA9062/DA9130/DA9131) provide **no per-rail current monitoring**, so **PS and PL
power cannot be separated in hardware.**

The honest method is **differential**: SOM total (accelerator running) − SOM total
(idle), with everything else held constant, sampled via `xlnx-platformstats`. The
pre-board estimate comes from Vivado `report_power` with a real `.saif`
(vector-based ≈ ±10%; vectorless ≈ ±15%, per UG907). Report **energy per
inference**, not watts, and state exactly what the rail covers.

---

## 9. Data movement architecture

| Data | Size | Location | Transport | When |
|---|---|---|---|---|
| Weights (Conv1) | 150 B | PL register file, **combinational** read | AXI4-Lite writes | once per model |
| Input image | 784 B | PL distributed RAM, **async** read | AXI4-Lite writes | per inference |
| Line buffers | 5 × 784 B | 5 × RAMB18E2 inside `input_feed_v2` | self-loaded, 784 cycles | on `start` |
| Results | 13,824 B | PL dual-port BRAM 4096×32 | AXI4 INCR bursts (14 × 256-beat) | on `done` |
| Intermediates (Conv2+) | — | **OPEN DECISION** — PS DDR memcpy initially | — | future |

**Result layout:** flat index `k = ((ch·24 + y)·3 + b)·8 + c`.

**⚠ Software must apply the reverse column index:** `x = base − c` where
`base = 7 + 8b`. This is the single most likely source of a *false* PASS in the
whole project. It is owned by `compare.py` (§10) and **must be confirmed in
block-design simulation, not on the board**.

**Estimated end-to-end:** 94.6 µs compute (V2 @ 200 MHz, including the 784-cycle
load) + ~40 µs burst read-back ≈ 135 µs → ~7,400 Conv1-layer inferences/s,
compute-dominated (~70%).

---

## 10. Verification strategy

**Ladder: L1 → L2 → L5 → L6 → L7. L3/L4 (cycle-accurate Python schedule models)
are explicitly REJECTED.**

| L | Layer | Purpose | Pass criterion |
|---|---|---|---|
| L1 | FP32 PyTorch reference | ground-truth CNN semantics | top-1 ≥ 98.5% (harness gate, not a result) |
| L2 | **INT8 integer reference (NumPy int32/int64)** | **the single golden model** | deterministic; no overflow |
| L5 | RTL simulation with real vectors (xsim) | implemented RTL matches mathematics | **bit-exact** vs L2, 3456/3456 |
| L6 | KV260 execution | silicon matches mathematics | **bit-exact** vs L2 |
| L7 | SW-vs-HW compare + accuracy | close the loop | 0 mismatches; Δtop-1 ≤ 0.5% |

### Why L3/L4 are rejected — the most important methodological decision here

A Python model of the OS/WS *schedule* is a re-implementation of the same mapping
the RTL implements. That is precisely the **Decision 9** failure mode: the
withdrawn single-pass 2-D mapping passed `sim/tb_systolic_array.sv` T11 for weeks
because the golden model was expressed in the *same shift-chain language* as the
wrong schedule. The testbench and the DUT agreed with each other, not with
mathematics.

**A golden model is independent if and only if it can be written by someone who
has never read the schedule, the PE geometry, or the tile counters** — i.e. if
and only if it is the *definition* of the operation
(`O[ch][y][x] = Σ_{ky,kx} I[y+ky][x+kx] · K[ch][ky][kx]`), not a description of
the datapath.

Cycle-level evidence already exists, independently, in the frozen synthetic
testbenches (379/379, 335/335) and their hierarchical probes. A Python schedule
model would add no new evidence — only a new place to agree with a wrong mapping.

**Corollary:** the new real-data testbenches must be **dumb drivers** —
`$readmemh` in, `$fwrite` out, **no golden model in SystemVerilog**. The
comparison happens in Python against L2. The stimulus that would have caught
Decision 9's bug (a 2-D image with all 25 kernel taps non-zero) becomes a
permanent regression vector.

### Synthetic / real separation

The seven existing testbenches (`tb_pe`, `tb_pe_v2`, `tb_systolic_array`,
`tb_systolic_array_v2`, `tb_input_feed`, `tb_input_feed_v2`, `tb_v2_full_system`)
are **frozen regression evidence**. They are not touched, renamed, or "upgraded."

Real-data tests are **new and additive** under `sim/real/` with a `_real` suffix.
The paper can then state plainly: *seven synthetic regression testbenches
(frozen, N/N PASS) + real-data testbenches (bit-exact vs an independent integer
golden model) + on-board execution.*

### Vector formats

- **Inputs** (must be `$readmemh`-compatible — there is no `$readmemd`): 2-digit
  lowercase two's-complement hex, one value per line.
  `input_img.hex` (784, row-major `y·28+x`), `weights.hex`
  (150, `ch·25 + ky·5 + kx`).
- **Golden**: decimal signed INT32. `golden_canonical.txt` (3456 lines, ch/y/x
  order, human-auditable) and `golden_hw_order.txt` (432 lines × 8, emission
  order including the reverse index, marked as generated).
- **Manifest**: `data/vectors/manifest.json` pins dataset, test indices, seed,
  package versions, git SHA, and the quantization scale factors — one file that
  fully specifies reproducibility.

### Tolerances

| Comparison | Rule | Justification |
|---|---|---|
| L2 vs L5 (RTL) | **BIT-EXACT** | Both are exact integer arithmetic; max \|acc\| = 409,600 ≪ 2³¹; no rounding occurs. A tolerance would only conceal bugs. |
| L2 vs L6 (board) | **BIT-EXACT** | The board runs the identical RTL. |
| L1 vs L2 | **NOT bit-exact** | This is quantization error — report it, don't mask it. |

Metrics to report for L1 vs L2: max absolute error, mean absolute error, RMSE,
per-channel and overall SQNR, weight-clip count, and Δtop-1 with the identities
of flipped predictions.

### Corner cases

All-zero image; all-`7f` and all-`80`; negated weights; valid-convolution edges
(the right-edge group `base = 23`, which reaches `px = 27`); the V2 tile-3
single-tap path (tap 24); the first and last group of a frame; channel 0 vs
channel 5 isolation; the OS↔WS switch boundary; and the permanent **Decision 9
regression vector**.

### Reconfiguration-specific tests (`sim/tb_reconfig_switch.sv`)

1. **OS ≡ WS bit-identity** — the same image and weights run once in each mode
   must produce **bit-identical** 3,456 results. *This single assertion fails if
   and only if the mode is not actually switched*, because a stuck-WS array
   drained per-row-0..5 would return garbage.
2. **Stale-ring independence** — run WS (leaving the ring non-zero), switch to OS
   **without** `rst`, run OS, assert bit-exact. Proves `psum_in` is truly ignored
   in OS mode.
3. **First-group-after-flush** — assert the *first* group of the new mode is
   bit-exact, proving the 7-cycle shift-chain drain worked.
4. **Interlock negative test** — attempt `MODE_COMMIT` while BUSY; assert
   `STATUS.ERROR`, `mode_active` unchanged, and the in-flight results still
   bit-exact.
5. **Schedule fingerprinting** — assert `CYCLE_RUN` matches the expected
   per-mode total.

### Runtime budget

A V2 frame is 784 + 18,144 = 18,928 cycles ≈ 30–90 s in xsim. **Rule: RTL
simulation is correctness sampling (1 image per mode plus the corner vectors);
the board is exhaustive accuracy (the full 10k test set, ~1 s of silicon time).**

**Entry point:** `scripts/run_verification.sh` runs the whole ladder and prints a
single PASS/FAIL table; exit code 0 only if every row passes. It asserts that
`xsim` resolves to Vivado ML 2023.1 (no substitute simulator, per `CLAUDE.md`).

---

## 11. Benchmark methodology

**Metric conventions, fixed now.** 1 MAC = 2 OPs, stated explicitly, because
papers differ and this is a classic unfair-comparison trap. Peak =
64 MAC/cycle × 200 MHz = 12.8 GMAC/s = **25.6 GOPS**. Sustained is the headline;
peak is a ceiling only.

| | Cycles | Latency | Sustained MAC/cyc | Sustained GOPS | Util | GOPS/DSP |
|---|---|---|---|---|---|---|
| V1 OS | 5,904 | 29.5 µs | 14.63 | 5.85 | 22.9% | 0.091 |
| V2 WS | 18,144 | 90.7 µs | 4.76 | 1.90 | 7.4% | 0.030 |

### Experiment set

| ID | Experiment | Priority |
|---|---|---|
| E1 | V1 OS vs V2 WS, identical layer / clock / data — cycles, latency, resources, power | **ESSENTIAL** |
| E2 | Runtime OS↔WS switch demonstration + overhead (flush cycles, weight reload, PS µs) | **ESSENTIAL** |
| E3 | Accuracy: FP32 vs INT8 software vs hardware-in-the-loop | **ESSENTIAL** |
| E4 | Resource / timing characterization (**must reconcile the LUT anomaly, §16 D-J**) | **ESSENTIAL** |
| E5 | Power / energy (INA260 differential + Vivado `report_power` with `.saif`) | **ESSENTIAL** |
| E6 | ARM Cortex-A53 INT8 baseline, same layer, same board | **ESSENTIAL** |
| E7 | End-to-end LeNet-5 (HW convolution + PS post-processing): top-1, FPS | **ESSENTIAL** |
| A1 | **Pipelined-V2 ablation** — is the 3.07× a dataflow effect or an implementation gap? | **BLOCKING** |
| A2 | Fixed-OS vs fixed-WS vs reconfigurable: the cost of the mux (reuses `build/dsp_probe/`) | HIGH |
| A3 | Switch cost vs layer size (amortization curve) | HIGH |
| A4 | Utilization vs layer shape (where each dataflow wins) | HIGH |

Repetitions: ≥5 for any timing or power measurement, discarding warm-up.

### Reconfiguration overhead — report five separate numbers, never one blended figure

| Quantity | Instrument | Expected |
|---|---|---|
| Flush cycles (commit → quiesced) | `MODE_SWITCHING` window | **8 cycles** |
| Commit → first valid compute cycle | `CYCLE_FREERUN` delta | ~12 cycles ≈ 60 ns @200 MHz |
| Weight-store reload (layer change) | `CYCLE_FREERUN` delta | ~40–100 cycles |
| PS software overhead | A53 global timer around the AXI transaction | ~1–5 µs |
| Per-run compute latency | `CYCLE_RUN` | per §11 table |

Never fold µs-scale PS latency into the ns-scale hardware number.

### Reporting discipline

Every number carries exactly one label: **[MEASURED ON HARDWARE]**,
**[POST-ROUTE SYNTHESIS ESTIMATE]**, **[SIMULATION]**, **[THEORETICAL/PEAK]**, or
**[LITERATURE — cited]** — with method, conditions (device, frequency, workload),
and source.

Table header format:
`Metric | Value | Label | Method/Instrument | Conditions | Source/URL`

Conflating these is the single most common rejection cause, and our ~13×
peak-to-sustained gap makes it this project's top reporting risk.

---

## 12. Literature comparison strategy

| Baseline | Verdict | Notes |
|---|---|---|
| **ARM Cortex-A53 INT8, same layer, same board** | **FAIREST** | Same silicon, same workload, same data, same power rail. **Must be measured, not estimated.** |
| Vitis AI DPUCZDX8G B4096 | **UNFAIR raw** | 2,048 MACs/cycle = **32× our peak**, and largely LUT-based. Usable only per-LUT / per-watt with the 32× gap stated explicitly. |
| Published FPGA CNN accelerators | Fair with normalization | See §20. |
| Desktop CPU / GPU | **UNFAIR** | Different process node, 65–350 W envelope. Accuracy parity only, never performance. |

Positioning against reconfigurable-dataflow prior art (MAERI, Eyeriss v2,
FlexFlow, Planaria, ReSA, AdaFlow, RIFT/TRINE) is **mandatory**, since that is
where the original novelty claim lived. Note that most flexible-dataflow work is
ASIC or simulation; the closest FPGA priors are AdaFlow (DATE'22) and RIFT/TRINE
(2026).

**Honest note the paper must print:** our per-DSP efficiency (0.030–0.091
GOPS/DSP) is roughly 8–23× below Winograd-class designs (~0.69 GOPS/DSP),
because we run an 86,400-MAC layer on an idle-dominated array. **Our GOPS must
never appear head-to-head with full-network accelerators as if on equal footing.**
The comparable quantity is the *relative* OS-vs-WS ratio, which is
workload-matched and self-normalizing.

---

## 13. Publication / reviewer risk assessment

| # | Objection | Severity | Fix |
|---|---|---|---|
| R1 | Novelty is prior art (RIFT / ReSA / AdaFlow) | **FATAL** | Reframe to the like-for-like measurement (§1) |
| R2 | Synthetic-only; no real data or pretrained weights | **FATAL** | W1–W3 |
| R3 | No runtime reconfiguration (`input_feed_v2.sv:245`) | **FATAL** | W6–W8 |
| R4 | Workload triviality (one layer of a 1998 network) | MAJOR | Full LeNet-5 via HW conv + PS post-processing (W5, W11, E7) |
| R5 | No end-to-end inference (raw INT32 out) | MAJOR | W5, E7 |
| R6 | "Q8" is not INT8 | MAJOR | §5 |
| R7 | 7.4% sustained utilization | MAJOR | Report honestly; A1 |
| R8 | No power, energy, accuracy, or FPS | MAJOR | E3, E5, E7 |
| R9 | No reproducibility artifacts | MINOR | manifest + `run_verification.sh` + committed top-level |
| R10 | Missing baselines | MAJOR | E6 (A53) |
| R11 | Missing ablations | MINOR | A1–A4 |
| R12 | V1 has no full-system testbench (V2 does) | MAJOR | W9 — the OS side of the comparison must be verified to the same standard as the WS side |

### Documentation red flags to correct

1. **`PROJECT_STATE.md:306-307`** — *"LeNet-5/MNIST accuracy loss negligible with
   8-bit quantization."* An unverified quantitative accuracy claim, with zero
   measured accuracy, zero real weights, and a format that is not standard 8-bit
   quantization. **The single most challengeable sentence in the repository.**
2. **`PROJECT_STATE.md:28`** quotes the **array-only OOC** WNS **+2.861 ns**,
   while the integrated full system closes at **+0.600 ns**
   (`build/v2_input_feed/04_postroute_timing_summary_max.rpt`). Quoting the
   flattering figure alone is cherry-picking. Print both; label both.
3. **`PE_SPEC.md:1071`**, **`SYSTOLIC_ARRAY_SPEC.md:482,634,649`** reference
   **`golden_model.py`** and **`tb_v1_integration.sv`** — **neither file exists.**
   Verification is claimed against absent artifacts.
4. **`README.md:10`** describes `python/` as "Python reference/model code"; it
   contains only `requirements.txt`.
5. **`SYSTOLIC_ARRAY_V2_SPEC.md:42-44,488-489`** and `PROJECT_STATE.md`
   milestone 17 say the V2 controller "remains future work," but
   `input_feed_v2.sv` exists and is verified (commit `318e7ed`). The docs lag the
   code.
6. **`SYSTOLIC_ARRAY_V2_SPEC.md:282`** — *"structurally identical to the
   PCIN→PCOUT cascade"* sits in the body while Decision 12 concludes the opposite
   (C input, not PCIN/PCOUT). Internally inconsistent as written.

### Venue reality check

Not competitive at FPGA / FCCM / MICRO / ISCA / DAC on novelty grounds.

**Realistic targets:** **HEART** (https://www.isheart.org/) or **ARC**
(https://www.reconfigarch.org/), plus an **arXiv preprint** to stake the
measurement early. FPL/OSDA workshop and an FCCM poster become plausible *if* the
board results land. Deadlines are annual — check the current CFP.

---

## 14. Work breakdown

Dependency-ordered. Tags: **[PRE]** completable without the board · **[BOARD]**
requires hardware · **[BLOCK]** blocking · **[OPT]** optional.

| # | Work item | Tags |
|---|---|---|
| W0 | Correct the §13 documentation red flags; supersede Decision 1; record the padding-clamp bug; add `data/raw/` to `.gitignore` | [PRE][BLOCK] |
| W1 | `python/` skeleton: environment lock, `model/train_lenet5.py` (seed-pinned), checkpoint + SHA-256 | [PRE][BLOCK] |
| W2 | `python/reference/`: `fp32_ref.py` (L1), `quant.py` (calibration), `int8_ref.py` (L2, integer-exact) | [PRE][BLOCK] |
| W3 | `gen_vectors.py` + `manifest.json` → `data/vectors/`, `data/golden/`, corner vectors | [PRE][BLOCK] |
| W4 | `sim/real/tb_conv1_v2_real.sv` — dumb driver, `$readmemh`/`$fwrite`; bit-exact vs L2 | [PRE][BLOCK] |
| W5 | `compare.py` + `accuracy.py`; PS post-processing reference (bias/requant/ReLU/pool/FC) | [PRE][BLOCK] |
| W6 | `weight_store.sv`, `os_schedule.sv` (OS schedule with **per-row drain**) | [PRE][BLOCK] |
| W7 | `reconfig_ctrl.sv` — mode_pending/mode_active, 8-cycle FLUSH FSM, interlock, `CYCLE_RUN` | [PRE][BLOCK] |
| W8 | `axi_lite_ctrl.sv` + `cnn_top.sv`; AXI VIP (PG267) protocol simulation | [PRE][BLOCK] |
| W9 | `sim/real/tb_conv1_v1_real.sv` — V1 full-system real-data testbench (closes R12) | [PRE][BLOCK] |
| W10 | `sim/tb_reconfig_switch.sv` — the five §10 reconfiguration tests | [PRE][BLOCK] |
| W11 | Controller generalization: Conv2 (IC=6, OC=16) + FC; 4-D weight store | [PRE][OPT → BLOCK for E7] |
| W12 | Vivado block design, `.xsa`, bitstream, timing closure at 150/200 MHz | [PRE][BLOCK] |
| W13 | Device-tree overlay, `shell.json`, firmware package, cross-compiled UIO application | [PRE][BLOCK] |
| W14 | `scripts/run_verification.sh`, `scripts/run_board.sh`, benchmark automation | [PRE][BLOCK] |
| W15 | Vivado `report_power` with a real `.saif` from W4/W10 | [PRE] |
| W16 | Board bring-up: boot, load bitstream, first `done` bit | [BOARD][BLOCK] |
| W17 | E1/E2/E3/E7 on hardware; bit-exactness vs L2 across the 10k test set | [BOARD][BLOCK] |
| W18 | E5 power (INA260 differential), E6 A53 baseline | [BOARD][BLOCK] |
| W19 | A1 pipelined-V2 ablation | [PRE sim + BOARD confirm][BLOCK] |
| W20 | A2–A4 ablations; paper tables and plots | [PRE/BOARD][OPT] |
| W21 | **Sparsity (Team A `zero_skip`)** — see §16 D-H; gated, droppable | [PRE][OPT] |

**Roughly 85–90% of the remaining engineering is board-independent.** The board
adds physical truth only: power, boot, DRAM behaviour, thermal behaviour, and
*measured* (not simulated) latency.

---

## 15. KV260 DAY-1 READINESS GATE

The goal: when board time begins, nothing architectural is still being decided.
Day 1 should be *connect → boot → load → feed a real MNIST sample → execute →
collect → repeat*.

**Hard gates — all must be GREEN before board time is spent:**

- [ ] **G1** Decision 1 superseded by calibrated INT8; PE RTL confirmed unchanged
- [ ] **G2** LeNet-5 trained; checkpoint + SHA-256 committed; FP32 top-1 ≥ 98.5%
- [ ] **G3** `data/vectors/`, `data/golden/`, `manifest.json` committed
- [ ] **G4** L2 bit-exact vs `sim/real/tb_conv1_v2_real.sv` (3456/3456)
- [ ] **G5** L2 bit-exact vs `sim/real/tb_conv1_v1_real.sv` (V1 verified to the V2 standard)
- [ ] **G6** `tb_reconfig_switch.sv` passes, **including OS≡WS bit-identity and the stale-ring test**
- [ ] **G7** `cnn_top` AXI4-Lite verified against AXI VIP (PG267)
- [ ] **G8** Bitstream generated; timing closed at the committed target; DRC clean; **WNS and target frequency recorded**
- [ ] **G9** Overlay + `shell.json` + firmware package built; UIO application cross-compiled
- [ ] **G10** `run_verification.sh` green end-to-end in simulation
- [ ] **G11** `run_board.sh` written and dry-run against a simulated register model
- [ ] **G12** Power method rehearsed: `xlnx-platformstats` invocation, differential protocol, sample windows
- [ ] **G13** A53 INT8 baseline written and profiled **on x86** (the port to A53 is recompile-only)
- [ ] **G14** Result ordering (reverse index `x = base − c`) confirmed **in block-design simulation, not on the board**
- [ ] **G15** Expected numbers pre-registered: V1 5,904 cycles, V2 18,144 cycles, 3,456 outputs, bit-exact match expected

**Known Day-1 blockers if not cleared beforehand:** the asynchronous-read image
RAM requirement (§8); the reverse column index (§9); the AXI `aresetn` ↔ `rst`
polarity conversion (§8); and the 150-vs-200 MHz decision (§7).

---

## 16. Exact remaining decisions

| # | Decision | Recommendation | Status |
|---|---|---|---|
| D-A | Numerical format | **Adopt calibrated symmetric INT8; retire the fixed 1/256 scale.** No RTL change | OPEN — team must record |
| D-B | Clock target for the reconfigurable system | **150 MHz** with pipelined `act_out`; keep 200 MHz for the frozen V1/V2 blocks | OPEN |
| D-C | Post-processing location | **PS software** | Resolved by analysis (§6) |
| D-D | Experimental scope | **Conv1 in both dataflows (core) + full LeNet-5 end-to-end via PS (E7)** | Resolved by analysis |
| D-E | Second dataset | **Yes — Fashion-MNIST, zero RTL cost** | OPEN |
| D-F | `DATAFLOW_MODE` encoding | **0 = OS, 1 = WS** (hardware-aligned) | OPEN |
| D-G | Result read-back | AXI4 BRAM burst; DMA only if reviewers object | OPEN |
| D-H | **Sparsity scope** | **Gate it** — see below | OPEN |
| D-I | Multi-layer intermediate storage | PS DDR memcpy initially; revisit if E7 latency disappoints | OPEN |
| D-J | LUT anomaly: 875 (full system) vs 1,032 (array alone) | **Must be reconciled before publication** — do not print both unexplained | OPEN |

### On sparsity (D-H) — a scoping caution

The project scope includes both reconfigurable dataflow *and* Team A's sparsity
extension. **This planning pass did not include a sparsity investigation, so this
document offers no researched sparsity plan and deliberately does not invent
one.**

What is known from the repository:

- `zero_skip` exists as a port on `pe.sv` and `pe_v2.sv` and is tied to 0
  everywhere (`input_feed_v2.sv:244`).
- **`PE_SPEC.md` §9.5 is explicit that PE-level `zero_skip` provides no
  throughput gain by itself.** The gain must come from an upstream Sparsity
  Manager that detects zeros, drops them from a FIFO, and compacts the activation
  stream.

That is a substantial new datapath (zero detection, FIFO compaction, index
bookkeeping, and re-alignment against the systolic schedule). Post-ReLU MNIST
activations are genuinely sparse enough to make it interesting.

**RECOMMENDATION:** treat sparsity as **W21, a gated second axis**. Commission a
dedicated investigation, then decide. Do **not** let it block W0–W19. The
dataflow-measurement result must be able to stand alone; sparsity either lands as
a second contribution or is described as future work. Attempting both novelty
axes simultaneously is the most likely way to finish neither.

---

## 17. Recommended implementation order

1. **Phase 0 — Truth (W0).** Fix the documentation before building on it.
2. **Phase 1 — Software ground truth (W1–W3, W5).** Nothing hardware-side is
   meaningful until L2 exists. **This is the critical path.**
3. **Phase 2 — Real-data RTL (W4, W9).** The first moment real pretrained weights
   meet the array. Expect surprises here, not on the board.
4. **Phase 3 — Reconfiguration (W6–W8, W10).** The apparatus. Gate on OS≡WS
   bit-identity.
5. **Phase 4 — SoC integration (W12–W14) + power estimate (W15).**
6. **Phase 5 — Scope extension (W11)** if E7 is pursued.
7. **Phase 6 — Board (W16–W18).**
8. **Phase 7 — Ablations and writing (W19–W20).**
9. **Phase 8 — Sparsity (W21)** only once Phases 0–7 are complete.

Phases 1–2 and Phase 3 can proceed in parallel across two people; Phase 4 depends
on Phase 3.

---

## 18. Definition of "paper-experiment ready"

- Real pretrained LeNet-5, calibrated INT8, real MNIST (and Fashion-MNIST)
  samples
- Hardware bit-exact against the integer reference in **both** dataflows, in
  simulation **and** on the KV260
- A genuine software-commanded OS↔WS switch with measured overhead, verified by a
  test that **fails if the mode bit is ignored**
- Measured latency, power, energy, and end-to-end top-1 accuracy — every number
  labelled with its provenance
- **A1 complete**, so the OS-vs-WS gap is attributed to dataflow rather than to
  differential un-pipelining
- Every objection in §13 either fixed or explicitly scoped out in the text

## 19. Definition of "KV260 ready"

All fifteen §15 gates green. Concretely: nothing architectural remains undecided,
the bitstream and overlay exist and are timing-clean, the host application is
compiled, the expected numbers are pre-registered, and the only unknown remaining
is what the silicon actually measures.

---

## 20. References and source URLs

### Datasets and models

- MNIST — http://yann.lecun.com/exdb/mnist/
- Fashion-MNIST (MIT) — https://github.com/zalandoresearch/fashion-mnist
- ONNX Model Zoo MNIST (MIT) — https://github.com/onnx/models/tree/main/validated/vision/classification/mnist
- BVLC Caffe model zoo (negative finding — no hosted LeNet) — https://github.com/BVLC/caffe/blob/master/docs/model_zoo.md
- `activatedgeek/LeNet-5` (no checkpoint) — https://github.com/activatedgeek/LeNet-5
- CIFAR-10 — https://www.cs.toronto.edu/~kriz/cifar.html
- `chenyaofo/pytorch-cifar-models` (BSD-3) — https://github.com/chenyaofo/pytorch-cifar-models
- `akamaster/pytorch_resnet_cifar10` (BSD-2) — https://github.com/akamaster/pytorch_resnet_cifar10
- torchvision models — https://docs.pytorch.org/vision/stable/models.html

### Quantization

- Jacob et al. 2018, *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* — https://arxiv.org/abs/1712.05877
- Krishnamoorthi 2018, *Quantizing deep convolutional networks for efficient inference* — https://arxiv.org/abs/1806.08342
- gemmlowp quantization — https://github.com/google/gemmlowp/blob/master/doc/quantization.md
- gemmlowp low-precision (zero-point folding) — https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
- PyTorch quantization (v2.1) — https://pytorch.org/docs/2.1/quantization.html
- ONNX Runtime quantization — https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- Brevitas — https://github.com/Xilinx/brevitas
- Vitis AI `vai_q_pytorch` — https://github.com/Xilinx/Vitis-AI/tree/master/src/vai_quantizer/vai_q_pytorch

### AMD / Xilinx documentation

- UG1085 — Zynq UltraScale+ MPSoC TRM (Ch. 10 address map) — https://docs.amd.com/v/u/en-US/ug1085-zynq-ultrascale-trm
- UG1089 — Kria KV260 Vision AI Starter Kit User Guide — https://docs.amd.com/v/u/en-US/ug1089-kv260-starter-kit
- UG907 — Vivado Power Analysis and Optimization — https://docs.amd.com/v/u/en-US/ug907-vivado-power-analysis-optimization
- PG021 — AXI DMA — https://docs.amd.com/v/u/en-US/pg021_axi_dma
- PG058 — Block Memory Generator — https://docs.amd.com/v/u/en-US/pg058-blk-mem-gen
- PG065 — Clocking Wizard — https://docs.xilinx.com/r/en-US/pg065-clk-wiz
- PG078 — AXI BRAM Controller — https://docs.amd.com/v/u/en-US/pg078-axi-bram-ctrl
- PG164 — Processor System Reset — https://www.xilinx.com/support/documentation/ip_documentation/proc_sys_reset/v5_0/pg164-proc-sys-reset.pdf
- PG201 — Zynq UltraScale+ MPSoC Processing System — https://docs.amd.com/v/u/en-US/pg201-zynq-ultrascale-plus-processing-system
- PG247 — AXI SmartConnect — https://www.xilinx.com/support/documents/ip_documentation/smartconnect/v1_0/pg247-smartconnect.pdf
- PG267 — AXI Verification IP — https://www.xilinx.com/support/documentation/ip_documentation/axi_vip/v1_1/pg267-axi-vip.pdf
- DS941 — Zynq UltraScale+ MPSoC VIP — https://japan.xilinx.com/support/documentation/ip_documentation/zynq_ultra_ps_e_vip/v1_0/ds941-zynq-ultra-ps-e-vip.pdf
- `xlnx-platformstats` — https://github.com/Xilinx/xlnx_platformstats
- Kria-PYNQ — https://github.com/Xilinx/Kria-PYNQ
- Kria custom firmware generation — https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/generating_custom_firmware.html

### Comparison and prior art

- FINN (FPGA'17) — https://dl.acm.org/doi/10.1145/3020078.3021744 · https://arxiv.org/abs/1612.07119
- Scaling BNNs on Reconfigurable Logic — https://arxiv.org/abs/1701.03400
- Angel-Eye (IEEE TCAD 2018) — https://www.semanticscholar.org/paper/196b2611ea462ffbe4e0d9124656eeb6677e3312
- Systolic Array Accelerator for LeNet-5 (Yang et al.) — https://link.springer.com/chapter/10.1007/978-3-030-05677-3_16
- LeNet-5 on ZCU102 — https://ieeexplore.ieee.org/abstract/document/9095722
- Winograd VGG-16 on ZCU104 — https://cjlcd.lightpublishing.cn/en/article/doi/10.37188/CJLCD.2023-0013/
- fpgaConvNet — https://github.com/AlexMontgomerie/fpgaConvNet
- **MAERI (ASPLOS'18)** — https://dl.acm.org/doi/10.1145/3297858.3304048 · http://synergy.ece.gatech.edu/tools/maeri/
- **Eyeriss v2 (JETCAS'19)** — https://www.rle.mit.edu/eyeriss-v2-a-flexible-accelerator-for-emerging-deep-neural-networks-on-mobile-devices/ · https://arxiv.org/abs/1807.07928
- **FlexFlow (HPCA'17)** — https://www.semanticscholar.org/paper/ce6403e99465e5e8a48d5c2017fc23976e29fe59
- **Planaria (MICRO'20)** — https://ieeexplore.ieee.org/abstract/document/9251939
- **ReSA (ACM TACO 2024)** — https://dl.acm.org/doi/full/10.1145/3653363
- **ReDas (IEEE 2024)** — https://ieeexplore.ieee.org/abstract/document/10527420
- **AdaFlow (DATE'22)** — https://ieeexplore.ieee.org/document/9774727
- **RIFT (DATE 2026)** — https://ieeexplore.ieee.org/document/11539726
- **TRINE (2026)** — https://arxiv.org/pdf/2603.22867
- Edge-AI FPGA dataflow survey (2025) — https://arxiv.org/pdf/2505.08992

### Venues

- HEART — https://www.isheart.org/
- ARC (Applied Reconfigurable Computing) — https://www.reconfigarch.org/
- FCCM — https://www.fccm.org/

---

## Appendix A — Internal references

| Reference | Description |
|---|---|
| `docs/PROJECT_STATE.md` | Master project decisions and status (Decisions 1–13) |
| `docs/specs/PE_SPEC.md` | PE contract (V1 §5–§8; PE-v2 §13) |
| `docs/specs/SYSTOLIC_ARRAY_SPEC.md` | V1 array contract (§20.4 withdrawn mapping) |
| `docs/specs/SYSTOLIC_ARRAY_V2_SPEC.md` | V2 WS array contract (§9 cycle schedule) |
| `docs/specs/INPUT_FEED_SPEC.md` | V1 controller contract (§7 group schedule) |
| `docs/specs/INPUT_FEED_V2_SPEC.md` | V2 controller contract (§12 open transport) |
| `build/v2_input_feed/` | V2 full-system post-route reports (WNS +0.600 ns) |
| `build/dsp_probe/` | DSP mapping probes (`pe_ws` vs `pe_dual`) — reusable for ablation A2 |

---

> **Next step:** adopt or amend the §16 decisions, then execute W0 (documentation
> corrections) before any new code is written.
