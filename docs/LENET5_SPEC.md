# LeNet-5 Workload Specification (Step 1A)

> **Status: LOCKED (2026-09-23).** This is the authoritative specification of the
> primary workload for the publication experiment, per the approved Step-0 and
> Step-1A decisions. The A1–A6 decisions below are final. No RTL, Python, or
> other documentation is changed by this document.

---

## 1. Purpose

Freeze the exact LeNet-5-based workload that replaces "MNIST-12" as the primary
CNN. The definition is the canonical LeCun et al. 1998 architecture, **in its
explicitly-documented INT8/ReLU hardware-inference variant** — it is **not**
claimed to be bit-for-bit identical to the 1998 network (which used tanh,
average-pooling, and an RBF output). See §8 "Reference Definitions" for the
precise tier separation.

---

## 2. Architecture — exact layer table

| # | Layer | Type | IC | OC | K / pool | In H×W | Out H×W | Params | MACs |
|---|---|---|---|---|---|---|---|---|---|
| — | Input | — | 1 | — | — | 32×32 | — | — | — |
| 1 | C1 | conv 5×5, valid | 1 | 6 | 5×5 s1 | 32×32 | 28×28 | 156 | 117,600 |
| | +bias, ReLU | | | | | | | | |
| 2 | Pool1 | maxpool | 6 | 6 | 2×2 / 2 | 28×28 | 14×14 | 0 | 0 |
| 3 | C3 | conv 5×5, valid, **fully-connected** | 6 | 16 | 5×5 s1 | 14×14 | 10×10 | 2,416 | 240,000 |
| | +bias, ReLU | | | | | | | | |
| 4 | Pool2 | maxpool | 16 | 16 | 2×2 / 2 | 10×10 | 5×5 | 0 | 0 |
| 5 | C5 | conv 5×5, valid | 16 | 120 | 5×5 s1 | 5×5 | 1×1 | 48,120 | 48,000 |
| | +bias, ReLU | | | | | | | | |
| 6 | F6 | FC | 120 | 84 | — | 120 | 84 | 10,164 | 10,080 |
| | +bias, ReLU | | | | | | | | |
| 7 | Output | FC | 84 | 10 | — | 84 | 10 | 850 | 840 |
| — | softmax → argmax | | | | | 10 | 1 | — | — |

**Totals: 61,706 params, 416,520 MACs.** (Params include biases.) Layer shape
derivation is deterministic: C1 `32−5+1=28`; Pool `28/2=14`; C3 `14−5+1=10`;
Pool `10/2=5`; C5 `5−5+1=1`.

**C3 is fully-connected** (6→16, 96 kernels = 6×16×25 = 2,400 weights + 16
biases). The canonical LeCun-5 **sparse C3 connection table** (60 kernels) is
**explicitly out of the primary experiment scope** — see §7 (Decision A2).

---

## 3. Operations & exact semantics

- **Convolutions (C1, C3, C5):** 5×5, stride 1, **VALID (no padding)**. Every
  conv has a per-output-channel bias. Activation = **ReLU** after each conv
  (`y = max(0, conv + bias)`).
- **Pooling (Pool1, Pool2):** **max-pool**, 2×2, stride 2, non-overlapping
  (Decision A1).
- **Fully-connected (F6, Output):** matrix-vector with bias; **ReLU after F6**,
  **no** ReLU after Output (Output feeds softmax directly).
- **Output semantics:** 10-class **softmax** over the Output logits; prediction
  = argmax. Loss = cross-entropy (training only) (Decision A4).
- **Numerical format:** symmetric **INT8** throughout (see §5). The integer
  dot-product the array computes is the exact kernel of this scheme; scales live
  outside the PE.

---

## 4. Preprocessing (input path)

The model consumes a **32×32×1** greyscale image, values `[0, 1]` (uint8/255),
no mean/std normalisation — consistent with the existing pipeline
(`python/reference/preprocess.py`). MNIST is natively **28×28**; the locked
transformation is **deterministic zero-pad-2 on each side (28×28 → 32×32)**
(Decision A3). No resampling, no per-image normalisation.

**Sparsity provenance — explicit distinction (Decision A3).** C1's input zeros
are **structural**: MNIST background pixels plus the 2-px zero border. C3/C5/F6
input zeros are **ReLU activation sparsity** produced by the network itself.
These two sources are distinct and must be reported separately in the paper;
the activation-sparsity (zero-skipping) contribution is carried by the deeper
layers (C3/C5/F6), not by C1's structural zeros.

---

## 5. Quantization — locked convention, extended to the full network

Symmetric, zero-point 0, per-channel weight / per-tensor activation INT8 (Jacob
et al. 2018; Krishnamoorthi 2018), identical to `docs/PHASE1_EXPERIMENT_SPEC.md`
§7: per-channel weight scale `S_w[c] = max|w_c|/127`, per-tensor activation scale
`S_a = max|a|/127`, RNE, int32 accumulator, bias `q_b[c] = RNE(b[c]/(S_a·S_w[c]))`.

**Full-network scope (Decision A6):** all **five learned layers** (C1, C3, C5,
F6, Output) are INT8-quantized, each with its own per-layer `S_a` and per-channel
`S_w`. Between layers the PS applies bias-add → dequantize → ReLU → (pool) →
requantize.

Accumulator safety (int32) holds for every layer: worst-case `|acc| = N·127²`,
with `N` = 25 (C1), 150 (C3), 400 (C5), 120 (F6), 84 (Output) → max ≈ 6.45 M ≪ 2³¹.

---

## 6. Hardware mapping scope (for orientation only — detailed mapping is a later step)

- **On the 8×8 array (as convolutions):** C1 (1→6), C3 (6→16), C5 (16→120),
  F6 (120→84, as a 1×1 conv), Output (84→10, as a 1×1 conv).
- **On the PS (software):** bias-add, dequantize, ReLU, maxpool, requantize,
  softmax, argmax.
- **New hardware dimensions vs the frozen controller** (currently 28×28, IC=1,
  OC=8, SAME): the 32×32 input, VALID convs, and IC>1 (C3, C5) all require
  controller changes. C1 (1→6, 5×5 valid, 32→28) is the closest to current scope;
  C3/C5 introduce IC>1 (uniform channel buffering + variable tile count); F6/Output
  are 1×1-conv FC mappings.

---

## 7. Resolved decisions (Step 1A lock)

| # | Decision | Resolution |
|---|---|---|
| A1 | Pooling type | **2×2/2 max-pool** (non-overlapping). Zero RTL cost (PS-side); no effect on OS/WS or on the zero/nonzero sparsity pattern. |
| A2 | C3 connectivity | **Fully-connected** (6→16, 96 kernels). The canonical **sparse C3 table is out of scope**: it introduces non-uniform IC/reduction schedules (75/100/150 taps per channel) that would confound the clean OS/WS comparison, and it represents historical structural connectivity rather than our activation zero-skipping mechanism. It is documented, not implemented. |
| A3 | MNIST 28→32 | **Deterministic zero-pad-2** (2 px each side). C1 structural input zeros are distinguished from ReLU activation sparsity in deeper layers (§4). |
| A4 | Output layer | **Softmax + cross-entropy** (training); argmax prediction. |
| A5 | Checkpoint | **Train the FP32 reference model in-repo** (32×32, seed-pinned, deterministic) with **SHA-256 checkpoint hashing** for reproducibility. Exact optimizer / learning rate / epochs / seed are frozen in **Step 1B**. |
| A6 | Quantization scope | **Full-network INT8** for all five learned layers, per-layer activation scales + per-channel weight scales. |

---

## 8. Reference Definitions: L1 FP32 and L2 INT8

Three distinct reference tiers; only L1 and L2 are implemented. They are kept
conceptually separate and asserted consistent at the layer boundaries.

1. **Historical canonical LeNet-5 (1998)** — tanh activations, average pooling
   with trainable coefficients, sparse C3 connectivity, Euclidean RBF output.
   **Reference only; NOT implemented.** This document's network is an explicitly
   documented modernized *variant* of it.
2. **L1 — FP32 reference.** Our modernized network definition (ReLU, max-pool,
   fully-connected C3, softmax) in FP32, trained in-repo per A5. This is the
   **accuracy ground truth** and the mathematical definition of the workload:
   the shapes, connectivity, and layer semantics that everything else is judged
   against.
3. **L2 — INT8 hardware reference.** The L1 network quantized per §5: int8
   activations and weights, int32 accumulation, int32 bias, per-layer activation
   scales and per-channel weight scales, with PS requantization between layers.
   This represents the **exact integer computation targeted by the accelerator**
   and is the bit-exact golden the RTL is verified against.

**Relationship:** L1 and L2 have **identical tensor shapes and C3 connectivity**;
they differ only in that L2 applies the defined quantization convention. The
FP32 definition (L1) and the INT8 hardware reference (L2) are two independent
code paths (mirroring `onnx_ref.py` vs `int8_ref.py`), linked by the shared
quantization convention and compared via the INT8-vs-FP32 Δtop-1 metric — they
are never collapsed into a single definition.

---

## 9. What this spec deliberately does NOT fix

- Controller/input-feed internals, tile counts, weight-store sizing, or
  per-mode emission order (later steps, per the existing spec discipline).
- Training hyperparameters (optimizer / lr / epochs / seed) — frozen in Step 1B.
- The second CNN (CIFAR-10 cuda-convnet) — separate spec, later.
