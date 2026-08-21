# Phase 1 — Real-Data Experiment Specification

> **Status:** LOCKED (2026-08-21). This is the authoritative record of the real
> CNN/dataset/checkpoint decision and the INT8 golden-model pipeline built in
> Phase 1. It supersedes the "valid-conv LeNet-5" assumption in
> `docs/RESEARCH_READINESS_PLAN.md` §2 on one specific point (see §3): the
> authoritative checkpoint is **SAME-padded**, so the accelerated Conv1 is a
> **padded** convolution, not a valid one. No RTL was modified.

---

## 1. Executive summary

| Item | Decision |
|---|---|
| CNN | **LeNet-5 variant ("MNIST-12"), ONNX Model Zoo** (MIT, validated) |
| Dataset | **MNIST** (60k train / 10k test, 28×28 greyscale) |
| Checkpoint | `data/checkpoint/mnist-12.onnx`, SHA-256 `5c688690…171bdd` |
| Accelerated layer | **Conv1** (1→8, 5×5, SAME_UPPER pad 2, 28×28→28×28) |
| Quantization | symmetric per-channel-weight / per-tensor-activation INT8, Z=0 |
| FP32 top-1 | **98.90%** (top-1 error 1.10%, matches the published 1.1%) |
| INT8 top-1 (Conv1 quantized) | **98.90%** (Δ = +0.00 pp, 0/10,000 flips) |
| Conv1 layer SQNR | **46.44 dB** |

The pipeline is end-to-end verified against real downloaded weights and real
MNIST data. All vectors are deterministic (fixed splits/indices, no RNG).

---

## 2. Selected CNN — and why

**A LeNet-5-family network is the only defensible choice for this hardware.** The
frozen 8×8 INT8 systolic array hardcodes 28×28 single-channel input, 5×5 kernels
(`rtl/common/input_feed_v2.sv:63-65`). For CIFAR/ImageNet models (ResNet-20,
MobileNetV2) the first convolution is 1–4% of total MACs and needs IC=3, 3×3
kernels, padding, and 32×32/224×224 inputs — a "we accelerate the conv layers"
claim is vacuous there. LeNet-5/MNIST is accepted as an FPGA *micro-architecture*
workload (FINN FPGA'17 used MNIST/CIFAR/SVHN); the contribution is the datapath,
not the model. See `docs/RESEARCH_READINESS_PLAN.md` §2 for the full argument.

The specific network is the **ONNX Model Zoo "MNIST-12"** model, a compact
LeNet-5 variant (trained in CNTK, alternating conv/maxpool):

| Layer | Type | Shape |
|---|---|---|
| Conv1 | conv 1→8, 5×5, SAME_UPPER | 28×28 → 28×28 |
| +bias, ReLU | | |
| Pool1 | maxpool 2×2/2 | 28×28 → 14×14 |
| Conv2 | conv 8→16, 5×5, SAME_UPPER | 14×14 → 14×14 |
| +bias, ReLU | | |
| Pool2 | maxpool 3×3/3 | 14×14 → 4×4 |
| FC | linear 256→10 | logits |

Total ≈ 320k MACs, ~6k parameters. **Conv1 = 156,800 MACs, 200 weights, 8 biases.**

---

## 3. Dataset — and why

**MNIST.** It is the canonical LeNet-5 dataset and the one the pretrained
checkpoint was trained on (a cross-dataset claim would be invalid). 28×28
greyscale, 10 classes, 60k/10k split. Downloaded via `torchvision.datasets.MNIST`
(pinned 0.16.2), which mirrors to `ossci-datasets.s3.amazonaws.com` (the
original `yann.lecun.com` host 404s).

Note: Fashion-MNIST is a zero-RTL-cost second dataset (identical 28×28 1-channel
geometry) that pre-empts the "MNIST is solved" objection; it is **not** part of
Phase 1 because the pretrained checkpoint is MNIST-trained. See
`docs/RESEARCH_READINESS_PLAN.md` §3.

---

## 4. Pretrained checkpoint — source and provenance

| Field | Value |
|---|---|
| Model | ONNX Model Zoo `MNIST-12` |
| URL | <https://github.com/onnx/models/tree/main/validated/vision/classification/mnist> |
| File | `model/mnist-12.onnx` (26,143 bytes, ONNX opset 12) |
| Licence | **MIT** |
| Stated top-1 error | 1.1% |
| SHA-256 | `5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd` |
| Local copy | `data/checkpoint/mnist-12.onnx` (committed) |
| Extracted weights | `data/checkpoint/mnist_weights.npz` (committed) |

**Why this checkpoint and not a `.pth`.** There is no authoritative,
publicly-downloadable pretrained LeNet-5 `.pth` (BVLC Caffe does not host
`lenet_iter_10000.caffemodel`; torchvision ships no LeNet). The ONNX Model Zoo is
MIT-licensed, maintained by LF AI & Data, versioned, and carries a validated
accuracy number. It is the most defensible "pretrained checkpoint" available.
`python/model/fetch_checkpoint.py` pins the SHA-256 and reproduces the download +
extraction; the committed `.onnx`/`.npz` make the experiment reproducible without
network access.

**Important consequence — SAME padding.** This checkpoint uses
`auto_pad=SAME_UPPER` on both convolutions (verified via
`onnx.shape_inference`). Its Conv1 is therefore a **padded** convolution
(28×28 → 28×28), *not* the valid 28×28 → 24×24 convolution the V1/V2 controllers
currently hardcode. This is a deliberate, honest consequence of using the
authoritative checkpoint rather than a hand-tuned valid-conv network; it makes
padding support a Phase 2 requirement (§12).

---

## 5. Accelerated layer — exact contract

**Conv1 (`Convolution28`) of MNIST-12.**

| Parameter | Value |
|---|---|
| Input | [1, 1, 28, 28], float [0,1] (post INT8 quant: int8) |
| Weights | [8, 1, 5, 5] |
| Bias | [8] |
| Kernel / stride / dilation | 5×5 / 1 / 1 |
| Padding | SAME_UPPER ⇒ pad 2 on all sides (input zero-padded to 32×32, then valid 5×5) |
| Output | [1, 8, 28, 28] = 6,272 values |
| MACs | 8 · 28 · 28 · 25 = **156,800** |

Equivalent to a **valid 5×5 convolution over a 2-pixel zero-padded 28×28 input**,
which is the exact operation the PE datapath already performs (one MAC/cycle/PE).
Only the *input-feed's* out-of-range handling must change to emit zero instead of
edge-replicated pixels (§12).

---

## 6. Preprocessing (reproducible)

The checkpoint consumes the raw greyscale image **scaled to [0,1], no mean/std
normalisation** (model-zoo README: "Color value is scaled to [0.0, 1.0]"). The
only preprocessing is:

```
x_fp = uint8_pixel / 255.0        # [0,1], float32
x_int8 = clip(RNE(x_fp / S_a), -128, 127)   # S_a = 1/127
```

No augmentation, no resizing (MNIST is already 28×28), no normalisation. The
integer pipeline operates on `x_int8`.

---

## 7. Quantization — exact convention

Symmetric, zero-point 0, full-range INT8 (Jacob et al. 2018; Krishnamoorthi
2018). The integer dot product the hardware computes is the exact kernel of this
scheme — the scales live *outside* the PE, so **no RTL change** is needed.

| Item | Convention |
|---|---|
| Weights | per-output-channel, `S_w[c] = max\|w_c\| / 127`, Z=0 |
| Activations | per-tensor, `S_a = max\|a\| / 127`, Z=0 |
| Quantize | `q = clip(round_half_to_even(x / S), -128, 127)` (RNE, float64) |
| Product / accumulator | 16-bit / **32-bit** signed |
| Bias | int32, `q_b[c] = RNE(b[c] / (S_a · S_w[c]))` |
| Dequantize | `y = (Σ q_a·q_w + q_b) · S_a · S_w[c]` |
| Calibration | 1,024 train samples, MinMax observer |

Measured parameters (full precision in `data/vectors/manifest.json` /
`quant_params.npz`):

| Parameter | Value |
|---|---|
| `S_a` | `1/127 = 0.007874015748031496` (max\|a\|=1.0 exactly) |
| `S_w[0..7]` | 0.008023, 0.004470, 0.007659, 0.003754, 0.005380, 0.005774, 0.004405, 0.004466 |
| `q_b[0..7]` (int32) | -2557, -12326, 1520, -570, -1535, -2898, 589, -3444 |
| zero-points | activation 0, weights 0 (per channel) |

Symmetric full-range quantisation clips nothing by construction (verified: 0
clipped activations, 0 clipped weights). `S_a = 1/127` because the input is
[0,1], so the activation maps to `q ∈ [0,127]` (7 effective bits) — negligible at
MNIST scale (measured Δtop-1 = 0.00 pp).

---

## 8. Tensor layout

| Tensor | Shape | Layout (flat index) |
|---|---|---|
| Input | 28×28 | row-major `y·28 + x` |
| Weights | 8×1×5×5 | `ch·25 + ky·5 + kx` (ch-major) |
| Golden (canonical) | 8×28×28 | `ch·784 + y·28 + x` (ch/y/x) |
| Bias | 8 | `ch` |
| Scales | 8 | `ch` (per-channel `S_w`; `S_a` scalar) |

The **hardware emission order** (reverse-column index / group order) is a Phase 2
deliverable. The controller for the padded OC=8 layer does not exist yet, so
Phase 1 emits only the canonical ch/y/x golden; Phase 2's `compare.py` reorders
it to the controller's emission order and confirms the mapping in block-design
simulation (the single most likely source of a false PASS, per
`RESEARCH_READINESS_PLAN.md` §9).

---

## 9. Golden model (independent, integer-exact)

`python/reference/int8_ref.py` implements the *definition* of the layer, written
so someone who has never read the PE geometry, tile counters, or input-feed
schedule can verify it:

```
O[c, y, x] = Σ_{ky=0..4} Σ_{kx=0..4} q_a[y+ky-2, x+kx-2] · q_w[c, ky, kx]
             (out-of-range reads contribute 0)
```

Operands int8, accumulation int64 (asserted to fit int32; real-image max
|acc| = 81,911, theoretical worst 25·127·127 = 403,225 ≪ 2³¹). Two
implementations — a pure-Python nested loop and a vectorised `einsum` — are
asserted bit-identical in `python/tests/test_pipeline.py` (T1). This is the
Decision 9 guard: the golden is expressed in the language of the operation, not
of the datapath.

---

## 10. Measured accuracy (verified end-to-end)

Full 10,000-image test set, deterministic:

| Metric | Value |
|---|---|
| FP32 top-1 | 98.90% (1.10% error) |
| INT8-Conv1 top-1 | 98.90% |
| Δ top-1 | +0.00 pp |
| Predictions flipped | 0 / 10,000 |
| Conv1 max \|err\| | 0.0360 |
| Conv1 mean \|err\| | 0.00163 |
| Conv1 RMSE | 0.00356 |
| Conv1 SQNR | 46.44 dB |
| Per-channel SQNR | 42.8–51.1 dB |

The FP32 number reproduces the checkpoint's published 1.1% error exactly,
validating the reference implementation. INT8 quantization of Conv1 is accuracy-
neutral for this workload.

---

## 11. Files created

```
data/checkpoint/mnist-12.onnx          # pretrained checkpoint (committed)
data/checkpoint/mnist_weights.npz      # extracted weights (committed)
data/raw/                              # MNIST archives (gitignored)
data/vectors/weights.hex               # 200 int8 weights ($readmemh hex)
data/vectors/input_img.hex             # primary test image (784 int8)
data/vectors/golden_canonical.txt      # 6272 int32 golden (ch/y/x)
data/vectors/multi/                    # 10-image regression set
data/vectors/quant_params.npz          # S_a, S_w, q_b, q_w, zero-points
data/vectors/manifest.json             # full reproducibility metadata
python/model/fetch_checkpoint.py       # download + SHA-256 + extraction
python/reference/constants.py          # experiment constants
python/reference/quant.py              # symmetric INT8 primitives
python/reference/preprocess.py         # [0,1] preprocessing + int8 quant
python/reference/onnx_ref.py           # FP32 reference (L1)
python/reference/int8_ref.py           # independent integer golden (L2)
python/reference/accuracy.py           # calibration + FP32-vs-INT8 metrics
python/reference/export_vectors.py     # hex/golden/manifest writer
python/run_phase1.py                   # end-to-end runner
python/tests/test_pipeline.py          # self-checks (T1–T5)
docs/PHASE1_EXPERIMENT_SPEC.md         # this document
```

`.gitignore` now ignores `data/raw/` (regenerable); checkpoints and vectors are
committed (small, required for clean-clone reproduction).

---

## 12. Requirements for Phase 2 (from the selected layer)

The PE (`pe.sv`/`pe_v2.sv`) and array (`systolic_array.sv`/`systolic_array_v2.sv`)
remain **frozen**. All Phase 2 changes are controller/input-feed level:

1. **Zero-padding support (BLOCKING).** The input-feed must emit `0` for
   out-of-range rows/columns during *active* cycles. Today
   `input_feed_v2.sv` clamps out-of-range columns to the edge pixel
   (`RESEARCH_READINESS_PLAN.md` §6 "latent padding bug"); that is harmless for
   valid conv but silently wrong for SAME padding. Fix: a `col_valid`/`row_valid`
   predicate gating `act_out`. Estimate ~0.5–1 week.
2. **OC = 8 (not 6).** The controller hardcodes 6 output channels. V1 OS maps
   channels onto PE rows 0–5 (rows 6–7 idle); OC=8 uses **all 8 rows** (better
   utilisation). V2 WS serialises channels — 8 sweeps instead of 6.
3. **Output 28×28 (not 24×24).** 6,272 results vs 3,456; the group/row schedule
   changes (28 output rows, no longer 3 clean 8-pixel groups).
4. **Weight store = 200 weights** (8×25) with 8 per-channel scales (V2 WS reads
   one channel per sweep, so per-channel scale is a trivial lookup — free).
5. **Input is unsigned [0,1]; no mean/std normalisation.** Symmetric Z=0 with
   `S_a = 1/127`; the optional full-range-uint8 offset fold (`q−128`, fold
   `−128·Σq_w` into bias) is available at zero RTL cost if the 7-bit input range
   ever matters.
6. **Bias-add + dequantisation stay in PS software** (unchanged from
   `RESEARCH_READINESS_PLAN.md` §6). The RTL emits the pure int32 MAC; the PS
   adds `q_b` and multiplies by `S_a·S_w[c]`.
7. **Accumulator width confirmed.** int32 is safe: worst case 403,225 ≪ 2³¹.

---

## 13. Reproducibility procedure

From a clean clone (network required only for MNIST, which mirrors to S3):

```bash
# 1. Environment (pinned): see docs/DEVELOPMENT_SETUP.md
.venv/bin/python -c "import torch,torchvision,numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)"
#    -> 2.1.2+cpu 0.16.2+cpu 1.26.4   (onnx 1.16.1 added for checkpoint reading)

# 2. Checkpoint is already committed; re-verify its SHA-256:
sha256sum data/checkpoint/mnist-12.onnx
#    -> 5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd

# 3. Re-extract weights (idempotent, verifies SHA):
.venv/bin/python python/model/fetch_checkpoint.py

# 4. Run the pipeline (downloads MNIST to data/raw/ on first run):
.venv/bin/python python/run_phase1.py

# 5. Self-checks:
.venv/bin/python python/tests/test_pipeline.py
```

Determinism: calibration = train[0:1024], evaluation = full test set, export
indices = test[0..9]. No RNG, no augmentation. `manifest.json` pins the
checkpoint SHA-256, scales, environment versions, and git SHA.

---

## 14. Remaining risks

1. **HW emission order unverified.** The canonical golden is mathematical truth,
   but the controller's exact per-mode emission order (and the reverse-column
   index) is not pinned until Phase 2. Mitigation: `compare.py` reorders the
   canonical golden and is confirmed in block-design simulation (never on the
   board first).
2. **SAME-padding controller change is new RTL** (the padding bug fix + OC=8 +
   28×28 scheduling). The PE/array are frozen, but this is real controller work
   and the first place real data meets the array.
3. **Per-channel weight scales require the PS to apply 8 distinct multipliers** —
   trivial, but it must not be flattened to a single scale by mistake.
4. **Only Conv1 is accelerated**; Conv2/FC run on the PS (FP32 in this Phase 1
   accuracy measurement). End-to-end *fully-quantised* accuracy (all layers INT8)
   is a later, separate measurement; Phase 1 establishes the Conv1 layer is
   accuracy-neutral.
5. **MNIST licensing** (no explicit licence; NIST-derived) is noted; Fashion-MNIST
   (MIT) is the drop-in fallback with zero RTL cost if reviewers object.
