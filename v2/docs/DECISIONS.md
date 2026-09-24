# V2 Decisions

## D1 — Requant (2026-09-24)
Hardware integer (m, s, RNE shift) proven equivalent to the float64 reference by exhaustive check; B ∈ {32, 40, 48}; fallback defined.

## D2 — Final layer (2026-09-24)
Raw INT32 logits exported; PS runs the reference float32 dequant + argmax.

## D3 — Reference freeze and accuracies of record (2026-09-24; CIFAR updated in Step 2.1c)
**CIFAR-10 reference = r2** (Step 2.1c, see "D3 update" and D9): retrained checkpoint `v2/model/retrain/cifar10_fp32_r2.pt` (sha256 0e69ed90b744f8c9bf1557c4ad9d49439957ff544f58020aae93e0f8f5eb819f, commit 0c14bd0), frozen INT8 set `v2/model/frozen/cifar10_int8_r2/` (`NET_CONFIGS["cifar10"]`). Accuracies of record (model, 10,000-image test sets): LeNet-5 FP32 98.78% (9878), INT8 98.79% (9879); **CIFAR-10 r2 FP32 78.56% (7856), INT8 78.52% (7852)**.
History (superseded, kept for the record as `NET_CONFIGS["cifar10_r1"]`): CIFAR r1 INT8 reference frozen in Step 2.1 from `data/checkpoint/cifar10_fp32.pt`; FP32 65.87% (6587), INT8 65.76% (6576) (from the recon runs; re-confirmed in Step 2.1).

## D4 — Git (2026-09-24)
v1-snapshot branch + v1-baseline tag preserve the pre-V2 state; V2 on v2-dev; main untouched.

## D5 — Naming (2026-09-24)
Legacy `rtl/common/*_v2.sv` is the reconfigurable OS/WS prototype (a baseline). New V2 modules use the prefix `gos_` and live in `v2/`.

## D6 — Recon gaps resolved by the spec (2026-09-24)
The recon-identified gaps (8 activations per cycle, clear handling, fill/skew) are resolved by the spec: banked ACT + rotator, first-flag load, broadcast (no skew), fill once per layer.

## D7 — Paper ablation (model-labeled) (2026-09-24)
The legacy generalized model (`python/v2_architecture_model.py`, 40 checks) = V2 core T\*K + per-pass lead-in + pass drain + clear + result drain (LeNet 30,013 = 16,288 + 8,160 + 2,973 + 180 + 2,412). V2 removes lead-in and pass drain via the banked continuous stream. Note: "41 tests" in earlier notes is incorrect; the count is 40.

## D1 outcome — Requant B = 32 (2026-09-24, Step 2.1)
Selected **B = 32** (smallest passing B; 32, 40 and 48 all pass with the search below). Source: `v2/model/requant_check.py` → `v2/results/requant_equivalence.csv` (model).
- Per-channel m selection rule (resolves OC-1): s from m_rne = RNE(Fraction(M)·2^s) ∈ [2^31, 2^32); try m = m_rne + d for d = 0, +1, −1, +2, −2, … (|d| ≤ 16, s fixed); keep the first m with 0 mismatches over the full check. Shift/rounding formula unchanged.
- At B = 32 only 2 of 354 channels needed an adjusted m, each by 1 LSB: lenet5 conv5 ch82 (d = −1) and cifar10 conv2 ch0 (d = +1). All other channels use m_rne.
- s range at B = 32: lenet5 40–44, cifar10 41–44 (fits the 6-bit field).
- Values checked per B: lenet5 291,507,758 (65,506,854 exact-region + 226,000,904 saturated boundary/samples); cifar10 183,973,136 (55,972,624 + 128,000,512). **0 mismatches** vs the legacy float64 `requantize` for the selected (m, s).
- Frozen per-channel (m, s): `v2/model/frozen/lenet5_int8/hw_requant.npz` (sha256 0c94d949…cbc05), `v2/model/frozen/cifar10_int8/hw_requant.npz` (sha256 23b58b5d…493e).
- Note: the float64 reference differs from exact rational RNE at 2 values (lenet5 fc1 ch54, v = ±31044); the hardware reproduces the float64 reference, which is the V2 golden.

## D2 outcome — Final layer verified (2026-09-24, Step 2.1)
`v2/model/final_layer.py` `logits_from_raw` mirrors `python/lenet5/int8_model.py:180-181` and `python/cifar10/int8_model.py:114-115`. On all 10,000 test images of each net: float32 logits bit-identical, argmax 10000/10000 equal to the legacy INT8 prediction. max |v|: lenet5 fc2 73,231; cifar10 fc 45,529 (bounds 1,376,611 / 1,050,146; int32 fits). An argmax on the raw INT32 v would differ on 55 / 1,422 images, so the PS float32 dequant is required. Source: `v2/results/final_layer_check.csv` (model).

## D3 outcome — CIFAR-10 INT8 frozen; accuracies of record (2026-09-24, Step 2.1)
- Frozen: `v2/model/frozen/cifar10_int8/quant_params.npz` (sha256 d2401ff8…801397), `manifest.json` (sha256 48acd07d…25db0), `SHA256SUMS`. Source commit v1-baseline a824e98; calibration CIFAR-10 train[0:1024], no RNG; export deterministic (byte-identical across runs/processes).
- LeNet-5: pointer only (`v2/model/frozen/lenet5_int8/POINTER.md` → `data/lenet5_int8/quant_params.npz`, sha256 verified).
- Accuracies of record (model, full 10,000-image test sets, `v2/results/reference_accuracy.csv`): LeNet-5 FP32 98.78% (9878), INT8 98.79% (9879); CIFAR-10 FP32 65.87% (6587), INT8 65.76% (6576). Confirmed equal to D3 values. *(Superseded for CIFAR-10 by the D3 update below; this set is now r1.)*

## D3 update — CIFAR-10 reference switched to r2 (2026-09-24, Step 2.1c)
- `net_config.NET_CONFIGS["cifar10"]` → `v2/model/frozen/cifar10_int8_r2/` (`quant_params.npz` sha256 7869babf…4ace12, `manifest.json` d25e9165…f9f7f0, `hw_requant.npz` 0cfc1802…219f49 at B = 32). All V2 scripts (golden, pack, vectors, cycle model, requant check, final layer, accuracy) take CIFAR parameters from there. The r1 set stays in `frozen/cifar10_int8/` as `NET_CONFIGS["cifar10_r1"]` (record-only, not in `NETS`).
- `v2/results/reference_accuracy.csv` has a `reference_version` column: `lenet5_v1`, `cifar10_r1`, `cifar10_r2`.
- r2 numeric checks (model): requant equivalence at B = 32 with the feasible-m search: 0 mismatches over 195,138,270 values; one adjusted channel (conv3 ch57, d = −1); s range 40–48, which fits the 6-bit s field (s < 64, FORMATS.md QPARAM; `gos_golden` asserts 1 ≤ s ≤ 63). Final layer: float32 logits from raw INT32 bit-identical and argmax identical on 10,000/10,000; max |v| 36,603 (int32 fits); raw-INT32 argmax would differ on 1,276 images, so the PS float32 dequant stays required. Sources: cifar10 rows of `requant_equivalence.csv`, `final_layer_check.csv`.
- ARCH_SPEC unchanged: layer shapes are identical, so cycle counts and memory sizes (WGT/ACT/QPARAM depth asserts in `gos_pack`) are unaffected.

## D8 — Step 2.2 format and dataflow resolutions (2026-09-24)
Approved at the Step 2.2 plan (user) unless marked "(Step 2.2 impl)" — those were resolved during implementation, reported to the user, and are open to override.
1. **Descriptor = 16 x 32-bit words per layer** with host-precomputed derived fields (K, IN_WPR, IN_PLANE, OC_TILES, OW_TILES, OUT_W, OUT_H, OUT_WPR, OUT_PLANE, WGT_END, IN_END, OUT_END, QP_END) so the RTL needs no multipliers, even at configuration time. Packing and checker rules: `docs/FORMATS.md`. The RTL config checker only compares; derived-field consistency is the host's responsibility (asserted in `gos_pack`). CSR byte offsets are deferred to the CSR step.
2. **Final layer has QPARAM entries** (q_bias, m=0, s=0; m/s unused because out_raw). LeNet 236 channels, CIFAR 138 (≤ 256).
3. **The address stream is data-independent**; vectors emit it once per layer.
4. **Requant-lane vectors:** 100k seeded (channel, v) pairs per net plus every near-tie v (exact |v·M − (n+½)| < 1e-9 in the exact region), including the Step 2.1 values.
5. **Unit-vector layouts are provisional** until the RTL interfaces are fixed (`v2/vectors/README.md`).
6. **`regen_results.sh` regenerates every results CSV** (reference_accuracy, requant_equivalence, final_layer_check, cycle_model, golden_crosscheck). `common.git_dirty()` excludes the generated outputs (`v2/results/`, `v2/vectors/MANIFEST.json`) so a regeneration run reports the state of the code that produced them; the script itself refuses a dirty tree at start.
7. (Step 2.2 impl) **Unwritten ACT bytes:** the model initializes buffers to 0; the RTL must not rely on unwritten bytes and writes with per-bank byte enables. Garbage may only reach masked x-tail rows / OC-tail lanes.
8. (Step 2.2 impl) **Rotator over-read:** masked rows may read word IN_END+1 (< 4096 for both nets; at IN_END = 4095 the 12-bit address wraps to 0, still harmless) — the checker rule IN_END ≤ 4095 suffices.
9. (Step 2.2 impl) **out_raw requires a 1x1 output** (OH = OW = 1, no pool, no ReLU); the address stream refuses other out_raw layers. The config checker (comparisons only) also requires out_raw → OC ≤ 16.
10. (Step 2.2 impl) **Masked output channels (≥ OC) still take their drain cycle** with byte-enable 0; their QPARAM read index may point past QP_END (wraps on the 8-bit port) and the data is discarded.
11. (Step 2.2 impl) **Pool write data:** the 4 pooled bytes are presented on both bank halves; the byte enables select the half (banks 0–3 for even ox_tile, 4–7 for odd). The dy=0 buffer holds post-requant/ReLU bytes; pooled = horizontal pair max of the vertical max.
12. (Step 2.2 impl) Drain/next-tile overlap and back-pressure are not modeled in the tile model (functional order only); cycle timing is the cycle model's and later the RTL's job.

13. (Step 2.2 impl, vectors) Layer-level vectors use the full-net WGT/QPARAM images (so each descriptor's WGT_BASE/QP_BASE is used as is); QPARAM is emitted as combined, even (QP_E) and odd (QP_O) views; expected ACT outputs come with a byte mask (`act_out_mask.hex`) — unmasked bytes are don't-care; LOGIT[OC..15] is 0-padded and don't-care.
14. (Step 2.2 impl, vectors) ARCH_SPEC does not fix where ReLU sits in the requant lane: requant vectors carry both the pre-ReLU and post-ReLU q. Max-pool compares signed int8 (after ReLU all values are 0..127; unit pool vectors also cover full signed int8). Placement of the 4 pooled bytes inside the 64-bit write word is left to the RTL; vectors give the 4 bytes + byte enable.
15. (Step 2.2 impl, vectors) Address-stream records carry the full read-address counter in 16 bits; the ACT port uses bits 11:0. Unit PE/array vectors include K < 8 (the per-layer checker's K ≥ 8 rule does not apply to the PE/array units).

## D9 — CIFAR-10 retrain r2 (2026-09-24, Steps 2.1b/2.1c)
(Requested as "D8"; D8 was already taken by the Step 2.2 resolutions.)
- **Same architecture:** legacy `cifar10.model.Cifar10Net` unchanged (32x32x3 → conv5x5 3→32 VALID, ReLU, pool 2x2/2 → conv5x5 32→32, ReLU, pool → conv5x5 32→64, ReLU → FC 64→10); no BatchNorm, no added layers. Legacy preprocessing (ToTensor [0,1], no mean/std), legacy per-tensor INT8 input scale, legacy `calibrate` on train[0:1024].
- **Protocol:** train = train[0:45000]; val = train[45000:50000] used only for selection; test (10k) evaluated once, at the end, on the chosen checkpoint — never used for any selection. Augmentation: random crop 32 with zero padding 4 + horizontal flip. SGD momentum 0.9, Nesterov, weight decay 5e-4, cosine LR from 0.05 (per epoch, eta_min 0), batch 128, 60 epochs; epoch chosen = argmax FP32 val accuracy → **epoch 59** (val FP32 78.54%). Fixed seeds (python, numpy, torch, data generator; `torch.use_deterministic_algorithms`). Script `v2/model/retrain/train_cifar10_r2.py`; training 1,835 s on CPU (16 threads).
- **Acceptance rule (set before evaluation):** accept only if INT8 test accuracy improves by ≥ 2.0 pp AND the requant check passes at B = 32. **Result: accept** — INT8 test 65.76% → 78.52% (+12.76 pp), FP32 test 65.87% → 78.56%; val FP32 67.18% → 78.54%, INT8 67.42% → 78.30%; requant B = 32: 0 mismatches. Adopted by the user in Step 2.1c. Record: `v2/results/cifar10_r2_accuracy.csv`, `cifar10_r2_summary.csv` (regenerated by `retrain/check_r2.py`).
- **Training log provenance:** `v2/results/cifar10_retrain_log.csv` is a training artifact tied to the r2 checkpoint SHA256 (0e69ed90…eb819f, `cifar10_fp32_r2_meta.json`). It is not regenerable without retraining, so it keeps its training-time metadata (git_dirty=True at bd28a72, the training code being uncommitted then) and is the only results CSV exempt from the clean-tree rule; `regen_results.sh` checks it against the checkpoint SHA256 and the selected epoch instead.
- **Every paper accuracy comes from evaluating the frozen checkpoint** (`reference_accuracy.py` / `check_r2.py` on the committed checkpoint and frozen npz), never from the training log.

## Open conflicts

### OC-2 — RESOLVED (option 1, user decision 2026-09-24) — CIFAR r2: B = 48 is infeasible with the 6-bit s field (2026-09-24, Step 2.1c)
- Evidence (model, `requant_check.py`): r2 conv1 ch7 has M = 8.1999e-06 (r1 min M over all CIFAR layers 1.5060e-04; LeNet 2.0568e-04). s = B − E(M) gives s_max = 48 / 56 / **64** at B = 32 / 40 / 48; s = 64 does not fit s[5:0] (ARCH_SPEC QPARAM, FORMATS.md "s < 64"), so `select_m_s` hits its guard: `STOP: s=64 outside [1, 63] for M=8.199850688629206e-06, B=48 (spec contradiction)`. `requant_check.py --nets lenet5 cifar10` (default B ∈ {32, 40, 48}, as run by `regen_results.sh`) therefore aborts before writing any CSV.
- Not affected: the selected B = 32 (D1 outcome) — r2 at B = 32: 0 mismatches over 195,138,270 values, s 40–48; B = 40 also passes (0 mismatches, s 48–56). LeNet-5 and CIFAR r1 at B = 48 stay within s ≤ 60.
- **Resolution (option 1):** `requant_check.py` keeps B ∈ {32, 40, 48}. `select_m_s` raises `ShiftOutOfRange` instead of stopping; `check_channel` records that (channel, B) as infeasible (`s_in_range = False`, required s in `s`, no m, nothing checked), and a B passes only if every channel is in range with 0 mismatches. Result: passing B = [32, 40] (lenet5 + cifar10 r2); cifar10 B = 48 has 2 out-of-range channels (s = 64); **selected B = 32 unchanged**, both `hw_requant.npz` byte-identical on regeneration.
- Guards so an out-of-range s can never reach the packed QPARAM: `gos_pack.load_params` asserts 1 ≤ s ≤ 63 on every requant layer (and s < 64 as before); test `test_adopted_shifts_in_range` asserts, for every net in `NETS`, that every adopted (selected-B) s is in [1, 63] in `hw_requant.npz` and round-trips identically through `pack_qparam`/`unpack_qparam`; `test_out_of_range_shift_recorded_not_fatal` and the slow `test_full_exhaustive_cifar10_decision` pin the recording and the decision.

### OC-1 — RESOLVED (option 2, user decision 2026-09-24; see D1 outcome) — D1 requant: no B in {32, 40, 48} is bit-exact with m = RNE(M·2^s) (2026-09-24, Step 2.1)
Source: `v2/model/requant_check.py` → `v2/results/requant_equivalence.csv` (model). Exhaustive exact region + saturated boundaries + 1M seeded samples per channel.

| net | B | values checked | mismatches |
|---|---|---|---|
| lenet5 | 32 / 40 / 48 | 291,507,758 each | 2 / 4 / 4 |
| cifar10 | 32 / 40 / 48 | 183,973,136 each | 2 / 0 / 2 |

All mismatches are ±v pairs at near-ties within ~1e-14 of a half-integer (saturated region: 0 mismatches). Independently re-verified in the main session:

| net / layer / ch | v | exact v·M − (−63.5) | float64 v·M | ref q | hw q (B=32/40/48) |
|---|---|---|---|---|---|
| lenet5 / conv5 / 82 | −55930 | +4.46e-15 | −63.49999999999999 | −63 | −64 / −64 / −64 |
| lenet5 / fc1 / 54 | −31044 | +2.95e-15 | −63.5 (float64 rounds to an exact tie → RNE −64; exact math gives −63) | −64 | −64 / −63 / −63 |
| cifar10 / conv2 / 0 | −80582 | −8.14e-15 | −63.50000000000001 | −64 | −63 / −64 / −63 |

Cause: the error of m/2^s vs M (≤ 2^−(s+1)), times |v| ≈ 5e4, exceeds these margins even at B=48; and for lenet5 fc1 54 the float64 reference itself departs from exact rational math. Any exact multiplier fails fc1/54, and every RNE m fails conv5/82, so the conflict is in the m-selection rule, not the shift/rounding formula.

Options considered (user chose 2):
1. Spec-defined fallback: the integer contract becomes the V2 golden; re-measure accuracy and record it.
2. Keep bit-exactness to float64 by changing only the per-channel m selection: choose m within the feasible interval (nearest to RNE(M·2^s)), then verify exhaustively. The Part B agent found feasible m for the 3 failing channels at every B, 1–3 LSB from RNE (scratch check only, not a deliverable); feasibility for all channels together is unverified.
