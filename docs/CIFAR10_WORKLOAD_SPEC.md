# CIFAR-10 Secondary Workload Specification (Step 6.2)

> **Status: FROZEN (2026-09-23).** Defines the second, substantially-different CNN
> workload used to test whether OS/WS dataflow selection and reconfiguration
> provide benefits not visible on LeNet-5. This workload is reproducible and
> **independent of the LeNet-5 model**. No RTL, sparsity, or dataflow mapping is
> performed here; no dataflow/sparsity benefit is claimed.

## 1. Network topology (cuda-convnet-style)

Input `32×32×3` → **Conv1** 5×5 valid `3→32` → ReLU → MaxPool 2×2/2 →
**Conv2** 5×5 valid `32→32` → ReLU → MaxPool 2×2/2 → **Conv3** 5×5 valid
`32→64` → ReLU → Flatten → **FC** `64→10` → softmax.

Valid (no-padding) convolution geometry is preserved throughout, matching the
LeNet-5 convention (`docs/LENET5_SPEC.md` §3). Max-pool 2×2/2, ReLU after each
conv, softmax on the logits — identical activation/pooling conventions to
LeNet-5.

## 2. Tensor shapes (per layer)

| Layer | Input | Output |
|---|---|---|
| Conv1 | (N,3,32,32) | (N,32,28,28) |
| Pool1 | (N,32,28,28) | (N,32,14,14) |
| Conv2 | (N,32,14,14) | (N,32,10,10) |
| Pool2 | (N,32,10,10) | (N,32,5,5) |
| Conv3 | (N,32,5,5) | (N,64,1,1) |
| FC | (N,64) | (N,10) |

Shape derivation: `32−5+1=28`, `28/2=14`, `14−5+1=10`, `10/2=5`, `5−5+1=1`.

## 3. MAC count

| Layer | MACs |
|---|---:|
| Conv1 (3→32) | 32·28·28·(3·25) = 1,881,600 |
| Conv2 (32→32) | 32·10·10·(32·25) = 2,560,000 |
| Conv3 (32→64) | 64·1·1·(32·25) = 51,200 |
| FC (64→10) | 10·64 = 640 |
| **Total** | **4,493,440** |

## 4. Parameter count

| Layer | Params |
|---|---:|
| Conv1 | 32·(3·25+1) = 2,432 |
| Conv2 | 32·(32·25+1) = 25,632 |
| Conv3 | 64·(32·25+1) = 51,264 |
| FC | 10·(64+1) = 650 |
| **Total** | **79,978** |

## 5. Dataset

CIFAR-10: 32×32×3 RGB, 10 classes, 50,000 train / 10,000 test.
`torchvision.datasets.CIFAR10` (torchvision 0.16.2).

## 6. Preprocessing

`uint8 [0,255] → float [0,1]` via `ToTensor`. No mean/std normalisation, no
augmentation — reproducible and consistent with the LeNet-5 preprocessing.

## 7. Train / validation / test split

Fixed indices, deterministic (no random split):
- train = CIFAR-10 train `[0:45000]` (45k)
- val   = CIFAR-10 train `[45000:50000]` (5k)
- test  = CIFAR-10 test `[0:10000]` (10k)

## 8. FP32 training configuration (frozen)

| Choice | Value |
|---|---|
| seed | 42 |
| optimizer | SGD, lr=0.01, momentum=0.9, weight_decay=0 |
| batch size | 128 |
| epochs | 10 |
| LR schedule | none |
| loss | CrossEntropyLoss |
| weight init | kaiming_uniform(a=√5) + bias uniform(±1/√fan_in) |
| device | CPU |

## 9. FP32 accuracy (measured)

Test top-1 = **65.87%** (val 67.18% at epoch 10; checkpoint SHA-256
`febc11d8…`). This modest accuracy is an honest, documented consequence of the
design: **no data augmentation**, **valid (no-padding) convolutions** (aggressive
spatial shrinkage 32→28→14→10→5→1), and **smaller channels** (32/32/64 vs the
classic 64/64/64 cuda-convnet). The workload is defined by its *architecture*
(topology/shapes/MACs/params), not by accuracy — the reference is reproducible
and sufficient for a dataflow micro-architecture experiment. Recorded in
`data/checkpoint/cifar10_fp32_meta.json`.

## 10. INT8 quantization convention

Identical to the frozen LeNet-5 L2 convention (`docs/LENET5_INT8_SPEC.md`):
symmetric signed INT8, zero-point 0, per-channel weight scales
`S_w[c]=max|w_c|/127`, per-tensor activation scales `S_a=max|a|/127`, RNE,
int32 accumulator, int32 bias `q_b[c]=RNE(b[c]/(S_a·S_w[c]))`, full-network
integer-arithmetic-only requantization. Accumulator safety: max reduction
N = 32·25 = 800 → |acc| ≤ 800·127² ≈ 12.9 M ≪ 2³¹.

## 11. INT8 accuracy validation requirement

INT8 top-1 within **Δ ≤ 1 pp** of the FP32 reference (measured, not assumed),
evaluated on the same 10k test split. This is the Step 6.3 deliverable.

## 12. Artifacts that will be frozen

- `data/checkpoint/cifar10_fp32.pt` + `cifar10_fp32_meta.json` (FP32 reference).
- `python/cifar10/` (config, model, preprocess, train) — independent of `lenet5/`.
- `python/tests/test_cifar10_reference.py` (topology/params/MACs checks).
- (Step 6.3) INT8 quantized weights + golden, and the L2 CIFAR-10 reference.

## 13. Why this workload differs from LeNet-5 (for OS/WS)

CIFAR-10 has **IC=3/32/32** (vs LeNet-5's IC=1/6/16) and **OC=32/32/64** (vs
6/16/120), i.e. a far more *IC-heavy, reduction-heavy* shape, ~10.8× the MACs,
and a 3-channel input. This is exactly the regime (deep reductions, many
channels) where WS is hypothesized to compete with OS — the open question the
mapping (Step 6.3+) will test, without assuming a result.
