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
- Accuracies of record (model, full 10,000-image test sets, `v2/results/reference_accuracy.csv`): LeNet-5 FP32 98.78% (9878), INT8 98.79% (9879); CIFAR-10 FP32 65.87% (6587), INT8 65.76% (6576). Confirmed equal to D3 values.

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

## Open conflicts

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
