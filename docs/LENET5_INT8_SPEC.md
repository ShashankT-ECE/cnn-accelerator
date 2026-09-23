# LeNet-5 L2 INT8 Quantization Convention (Step 2.1)

> **Status: FROZEN (2026-09-23).** This is the authoritative INT8 (L2)
> quantization convention for the LeNet-5 workload. It extends the locked
> Phase-1 convention (`docs/PHASE1_EXPERIMENT_SPEC.md` §7) to full-network
> integer-arithmetic-only inference and is consistent with the L1 spec
> (`docs/LENET5_SPEC.md` §5, §8). L1 (FP32) and L2 (INT8) remain conceptually
> separate throughout (§15).

---

## 1. Purpose and relationship to L1

- **L1** (`docs/LENET5_SPEC.md`) is the frozen FP32 accuracy reference: the
  ground truth and the mathematical definition of the workload.
- **L2** (this document) is the INT8 hardware reference: L1's network with
  identical tensor shapes and connectivity, quantized per this convention.
- L2 is **integer-arithmetic-only**: the layer-to-layer inference path uses int8
  activations/weights, int32 accumulation, and integer requantization. No FP32
  arithmetic appears between layers — only the initial input quantization and the
  final softmax/argmax are FP32 (and even those are outside the accelerator).
- L1 and L2 are two independent code paths (mirroring `onnx_ref.py` vs
  `int8_ref.py`), compared via the INT8-vs-FP32 Δtop-1 metric.

## 2. Number formats

| Quantity | Format |
|---|---|
| Activation | signed int8, two's complement |
| Weight | signed int8, two's complement |
| Product | 16-bit full precision (max 127² = 16,129) |
| Accumulator | signed int32 |
| Bias | signed int32 |
| Scale / requantization arithmetic | float64 (bit-reproducible reference) |

## 3. INT8 signed range and symmetric quantization

- int8 range is **[-128, 127]** (two's complement).
- Quantization is **symmetric, zero-point 0, full-range**: scale `S = max|x| / 127`,
  so the maximum-magnitude value maps to exactly ±127 and the level **-128 is
  never produced** (the `clip` to [-128, 127] is a defensive bound that never
  triggers by construction).
- Quantize: `q = clip(RNE(x / S), -128, 127)`, where RNE = round-half-to-even.

## 4. Activation quantization (per-tensor)

- Each layer `L` has **one** activation scale `S_a^L` for its **input** activations.
- `S_a^L = max|a^L| / 127`, determined by MinMax calibration (§12).
- `q_a^L = clip(RNE(a^L / S_a^L), -128, 127)`, zero-point 0.
- Post-ReLU activations are non-negative, so `q_a ∈ [0, 127]` (7 effective bits)
  for every layer after the input — an accepted, documented cost of symmetric
  quantization (negligible at MNIST scale; Phase 1 measured Δtop-1 = 0.00 pp).

## 5. Weight quantization (per-output-channel)

- Each layer `L` has one scale **per output channel**:
  `S_w^L[c] = max|w_c^L| / 127`.
- `q_w^L[c] = clip(RNE(w_c^L / S_w^L[c]), -128, 127)`, zero-point 0.

## 6. Zero-points

- Activation zero-point: **0**. Weight zero-point: **0** (per channel).
- Symmetric zero-point-0 removes the `z_a·Σq_w` and `z_w·Σq_a` correction terms
  from the integer dot product, so the MAC is the pure `Σ q_a·q_w`.

## 7. Bias representation

- Bias is **int32**, added **after** the MAC (the array emits the pure int32 MAC;
  the PS adds the bias — unchanged from `RESEARCH_READINESS_PLAN.md` §5).
- `q_b^L[c] = RNE(b^L[c] / (S_a^L · S_w^L[c]))`, rounded to int32.

## 8. int8×int8 → int32 accumulation

- `acc = Σ q_a · q_w`, summed over the reduction dimension `K·K·IC`, accumulated
  in int32. The product is full 16-bit precision inside the PE.
- Overflow is impossible by construction: `max|acc| ≤ N·127²` with `N` = 25 (C1),
  150 (C3), 400 (C5), 120 (F6), 84 (Output) → max ≈ 6.45×10⁶ ≪ 2³¹
  (see `docs/LENET5_SPEC.md` §5).

## 9. Requantization (integer-only, between layers)

The int32 accumulator is requantized to the next layer's int8 activation.

- Requantization scale (per output channel): `M^L[c] = S_a^L · S_w^L[c] / S_out^L`.
- Canonical (float64 reference): `q_out = clip(RNE((acc + q_b) · M^L[c]), -128, 127)`.
- `S_out^L` is the calibrated per-tensor **output** activation scale of layer `L`
  (it becomes `S_a^(L+1)`; see §12).

**Integer-only fixed-point realization** (to be pinned in Step 2.2):
`M = 2^-n · M0` with `M0 ∈ [0.5, 1)`; int32 multiplier `m = RNE(M0 · 2^31)`;
`q_out = clip(((acc + q_b) · m + 2^30) >> (31 + n), -128, 127)` with
round-half-away-from-zero. This realizes the canonical formula within ±1 LSB.

## 10. Rounding

- Quantization (float → int8): **RNE** (round-half-to-even), identical to Phase 1.
- Requantization (canonical float64 reference): **RNE**.
- Requantization (integer-only fixed-point, Step 2.2): **round-half-away-from-zero**
  (standard add-and-shift); equivalent to RNE within ±1 LSB.

## 11. Saturation / clamping

- Every quantized value clips to **[-128, 127]**.
- Symmetric full-range quantization clips nothing by construction; the clip is a
  defensive bound that never triggers (Phase 1 verified 0 clipped activations
  and 0 clipped weights).

## 12. ReLU placement

- ReLU is applied to the requantized output of every layer **except the final
  Output layer**: `q_out = max(0, q_out)` (equivalently, clamp to [0, 127]).
- ReLU and requantization commute (rounding is monotonic and 0 is a fixed
  point), so ReLU may be applied before or after the requantization rounding
  without changing the result.

## 13. Scale propagation between layers

Per-layer MinMax calibration over the fixed calibration split `train[0:1024]`
(matching Phase 1's `N_CALIB = 1024`; the exact procedure is Step 2.2):

1. `S_a^C1 = max|input| / 127` (the input is [0,1], so `S_a^C1 = 1/127`).
2. For each layer `L` in (C1, C3, C5, F6, Output): quantize weights/bias per
   §5/§7, run the quantized layer on the calibration set, and set
   `S_out^L = max|post-ReLU output| / 127` (symmetric over the post-ReLU,
   post-pool activations; for the Output layer, over the raw logits).
3. `S_a^(L+1) = S_out^L`.

The output scale of each layer becomes the input scale of the next — no global
scale is shared across layers.

## 14. Final FC layer (Output, 84→10)

- Quantized like every other FC layer: per-channel weight scales, per-tensor
  activation scale, int32 bias, int32 accumulation.
- **No ReLU, no pooling** on the output.
- For top-1 accuracy: dequantize the int32 logits per channel,
  `logit_fp32[c] = (acc[c] + q_b[c]) · S_a^Out · S_w^Out[c]`, then softmax/argmax
  in FP32.
- Note: argmax is invariant to a *common* positive scale, but **not** to the
  per-channel weight scale `S_w^Out[c]`, so the dequantization must remain
  per-channel (never flattened to a single scale).

## 15. Conceptual separation (L1 vs L2)

- L1 (FP32) is the accuracy ground truth; L2 (INT8) is its quantized realization
  with identical tensor shapes and C3 connectivity, differing only by this
  quantization convention.
- L2 is validated by (a) an independent integer golden model (Step 2.2),
  (b) layer-wise bit-exactness against that golden, and (c) INT8-vs-FP32 Δtop-1
  against the frozen L1 checkpoint (`data/checkpoint/lenet5_fp32.pt`,
  SHA-256 `9978676b…`).

## 16. Deliberately deferred to Step 2.2

- The exact integer-only fixed-point `m`/shift realization (bit-widths, rounding).
- The exact calibration procedure and indices.
- The independent integer golden model (`int8_ref`-equivalent) and its tests.
- Layer-wise and end-to-end INT8 accuracy numbers.
