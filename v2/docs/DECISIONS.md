# V2 Decisions

## D1 — Requant (2026-09-24)
Hardware integer (m, s, RNE shift) proven equivalent to the float64 reference by exhaustive check; B ∈ {32, 40, 48}; fallback defined.

## D2 — Final layer (2026-09-24)
Raw INT32 logits exported; PS runs the reference float32 dequant + argmax.

## D3 — Reference freeze and accuracies of record (2026-09-24)
CIFAR INT8 reference to be frozen in Step 2. Accuracies of record: LeNet INT8 98.79% (9879/10000), CIFAR INT8 65.76% on 10,000 (from the recon runs; re-confirmed in Step 2). FP32: 98.78% / 65.87%.

## D4 — Git (2026-09-24)
v1-snapshot branch + v1-baseline tag preserve the pre-V2 state; V2 on v2-dev; main untouched.

## D5 — Naming (2026-09-24)
Legacy `rtl/common/*_v2.sv` is the reconfigurable OS/WS prototype (a baseline). New V2 modules use the prefix `gos_` and live in `v2/`.

## D6 — Recon gaps resolved by the spec (2026-09-24)
The recon-identified gaps (8 activations per cycle, clear handling, fill/skew) are resolved by the spec: banked ACT + rotator, first-flag load, broadcast (no skew), fill once per layer.

## D7 — Paper ablation (model-labeled) (2026-09-24)
The legacy generalized model (`python/v2_architecture_model.py`, 40 checks) = V2 core T\*K + per-pass lead-in + pass drain + clear + result drain (LeNet 30,013 = 16,288 + 8,160 + 2,973 + 180 + 2,412). V2 removes lead-in and pass drain via the banked continuous stream. Note: "41 tests" in earlier notes is incorrect; the count is 40.

## Open conflicts

