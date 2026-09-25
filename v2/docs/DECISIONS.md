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

## D10 — Array drain: no back-pressure path (2026-09-24, Step 3)
User-specified for Step 3; **supersedes the ARCH_SPEC Datapath sentence "Back-pressure stall if the drain is busy"**. `gos_array` has no stall/back-pressure path: the config checker guarantees K ≥ 8 (D8-1), and at K ≥ 8 a new capture can coincide at the latest with the last drain cycle (zero-gap K = 8 tiles verified in `tb_gos_array`). The STALL counter reads 0 by construction. A capture that would overwrite undrained data (K < 8) sets the sticky `err_overrun` (cleared only by rst); the new tile then overwrites the shadow and restarts the drain (old tile's remaining columns lost).

## D11 — Step 3 leaf-RTL resolutions (2026-09-24)
Items 1–6 approved at the Step 3 plan (user; item 2 as revised by the user). Items marked "(Step 3 impl)" were resolved during implementation and reported; open to override.
1. **PS access to ACT while busy:** PS writes are ignored, PS reads return 0; either sets the sticky `ps_busy_violation` flag (cleared by rst). Accelerator writes while not busy are ignored silently.
2. **PS-side read latency** is a parameter `PS_RD_LAT` ∈ {1, 2} on `gos_act_buf`, `gos_wgt_mem`, `gos_qparam_mem`, default **1** (no output register; the AXI BRAM Controller expects latency 1 by default). Both values verified in the TBs; OOC uses the default. The accelerator-side read latency is fixed at 2.
3. **Array drain order:** out_col 0→7 (output channel j = col) on consecutive cycles starting the cycle after capture; the captured token is repeated on all 8 drain cycles. `L_ARRAY` = 6 (in_last cycle → column 0 on the outputs; column j at +6+j).
4. **Requant ReLU placement:** clip, then ReLU when relu_en (resolves D8-14; equivalent to ReLU-then-clip). `v_raw` = full 32-bit acc + q_bias, independent of relu_en/out_raw; `out_raw` is carried as `out_raw_d`. Masked lanes are handled at integration.
5. **Pool dy=1 without a prior dy=0 for that column:** undefined result, no error flag (the loop nest guarantees dy=1 immediately follows the dy=0 tile of the same (oc_tile, oy pair, ox_tile); asserted on the real drain streams in `tb_gos_pool`).
6. **`gos_qparam_mem` PS port:** 9-bit word address, bit 0 selects the bank (even {m, q_bias} / odd {58'b0, s}), word index = addr[8:1]; accelerator channel index wraps at 8 bits.
7. **`gos_tok_t` (gos_pkg.sv) is PROVISIONAL** and will be extended in Step 4 (e.g. output word address, QPARAM channel index).
8. (Step 3 impl) **Requant latency L_RQ = 6** (user brief: target 4–6). ARCH_SPEC's "3–4 pipeline stages" is exceeded using the ARCH_SPEC fallback "extra requant pipeline stages" (not a redesign). Multiply `$signed(v[25:0]) × {0, m}` = 2 DSP48E2 per lane (16 total); RNE realized as shift-by-(s−1) + sticky + half bit (bit-exact on all vectors and 1.2M random lanes incl. 337k exact ties). s = 0 → q forced to 0 (final layer uses v_raw).
9. (Step 3 impl) **Memory collisions:** port A is read-first (UG901); a port-A write and a port-B read of the same word in the same cycle give undefined port-B data. Cannot occur on ACT in operation (core reads ACT[in_sel], writes ACT[!in_sel]; PS locked out while busy). WGT/QPARAM have no busy lock-out (not in their contract). PS ports use separate `ps_en`, `ps_we`, `ps_be[7:0]` (AXI BRAM Controller `we[7:0]` → `ps_we = |we`, `ps_be = we`).
10. (Step 3 impl) **QPARAM storage:** odd bank stores only s[5:0] (PS reads return {58'b0, s}); result 2 RAMB36 + 1 RAMB18 (one accelerator access must return 70 bits; a PS-writable BRAM port is ≤ 36 bits wide). WGT byte lanes are built as 16K×8 cascades (Vivado Synth 8-6841), still 32 RAMB36.
11. (Step 3 impl) **Array:** the operand broadcast register is kept in fabric (`(* keep *)` on a_b/w_b, +128 FF) so every PE DSP uses AREG=BREG=1 (fanout after the register, as specified). **Pool:** one output per input (pool-store cycles emit out_valid with be=0); `out_be = row_mask` (no pool) / `row_mask[2i]` on the selected half (pool); ch_valid/out_raw gating at integration (`be & {8{ch_valid & !out_raw}}`); the 8×8 buffer maps to LUTRAM. Rotator L_ROT = 1, pool L_POOL = 2.
12. (Step 3 impl) **OOC timing method:** all non-clock ports get 0 ns input/output delay relative to clk (full-cycle budget), so port-to-register paths (e.g. `gos_rotator`) are timed. Numbers are post-synthesis OOC estimates, not post-implementation.
13. (Step 3 impl) **xsim 2023.1:** a bare `$fatal(1);` is silently ignored and xsim exits 0 on `$fatal`; TBs call `$fatal(1, "<msg>")` after printing `TEST FAILED`, and `run_xsim.sh` decides PASS/FAIL from the log.

## D12 — Step 4 core integration (2026-09-24)
Items 1–6 approved at the Step 4 plan (user). Items marked "(Step 4 impl)" were resolved during implementation and reported; open to override.
1. **`gos_core` contains the memories** (ACT0/ACT1, WGT, QPARAM) + datapath + control; PS memory ports and the descriptor/CSR side are plain ports. `gos_top` = `gos_csr` + `gos_core`; the Step 6 plain-Verilog wrapper adapts `gos_top` to the shell port list (`gos_top_ports.vh`).
2. **Cycle definitions:** LAYER_CYC[l] counts the cycles from layer l's S_LOAD to its layer-end cycle inclusive (issue finished and no tile in flight: tiles issued − tiles retired at the writer = 0, an event count). TOTAL_CYC counts every cycle with STATUS.busy = 1 = C_START + Σ LAYER_CYC + C_DONE. MAC_ACTIVE counts cycles with a valid array input (= Σ T·K). STALL counts issue-idle cycles inside the issue phase; gos_ctrl issues every cycle while `issuing` (no back-pressure, D10), so it reads 0 by construction.
3. **Config check:** all descriptors l < N_LAYERS in parallel (S_CHK1 registers the rule vectors, S_CHK2 priority-encodes); N_LAYERS must be 1..8 (rule 32). ERR_CODE = {16'b0, rule_id[7:0], 5'b0, layer[2:0]} for the first failing layer, lowest rule id. Rule ids / priority: FORMATS.md §5 (mirrored by `gos_pack.check_descriptor_rules`); rule 16 (K == 0) is never reported first because rule 3 (K < 8) precedes it.
4. **STATUS:** busy = running; done and error sticky until the next start or soft_reset; start while busy ignored; soft_reset clears core state, status, counters and LOGIT (memory contents and CSR DESC/N_LAYERS kept); LOGIT[0..15] cleared to 0 at start.
5. **BUILD_ID** = 8-hex-digit short commit (`git rev-parse --short=8`); `impl_shell.csv` records git_dirty; the artifact of record must come from a clean tree.
6. **Fuzz vectors:** 100 seeded random single-layer shapes with T·K ≤ 60,000 (`gen_vectors.py`, `gos_fuzz.py`).
7. **Final `gos_tok_t`** (gos_pkg.sv): layer[2:0], oc_tile[7:0], dy, pool_en, relu_en, out_raw, row_mask[7:0], ch_valid[7:0], word_base[11:0] (output ACT word of drain column 0), half, qp_base[7:0] (QPARAM channel of drain column 0) — tile-constant, captured by gos_array at `last`. The drain stage adds col (3 bits) and the column's word = word_base + col·OUT_PLANE, built incrementally (column 0 loads word_base, each next column adds OUT_PLANE); QPARAM channel = qp_base + col.
8. **Cycle model (RTL-derived, committed before any core simulation, `1eb0873`):** LAYER_CYC = T·K + C_PIPE, TOTAL_CYC = Σ LAYER_CYC + C_START + C_DONE with
   C_PIPE = L_LOAD 1 + L_ISSUE 1 + L_MEM_ACC 2 + L_ROT 1 + L_ARRAY 6 + (L_DRAIN − 1) 7 + L_QPARAM 2 + L_RQ 6 + L_POOL 2 + L_RETIRE 1 = **29**, C_START = S_CHK1 + S_CHK2 = **2**, C_DONE = **0** (busy falls on the edge after the last layer-end cycle). Derivation in `gos_cycle_model.py`; `tests/test_cycle_constants.py` checks every term against the RTL localparams. RTL simulation matched without any correction (net layers, network jobs, 100 fuzz shapes): `rtl_cycles.csv`, `rtl_network.csv`.
9. **Register map and shell address map:** FORMATS.md "CSR register map" and "PL top ports and address map". VERSION = 0x474F5302 (core) / 0x474F5300 (empty shell), both at 0x0F8; BUILD_ID at 0x0FC.
10. (Step 4 impl) **CSR busy lock:** writes to N_LAYERS or DESC while STATUS.busy return SLVERR and have no effect (the core reads them live during a run; modelled on D11-1). CTRL writes are always OKAY. Reading a 64-bit counter's lo word latches its hi word for a consistent hi read. Unmapped / read-only accesses return SLVERR.
11. (Step 4 impl) **Error flags register (0x028):** bit0/bit1 ACT0/ACT1 PS access while busy, bit4 array overrun; sticky, cleared by rst / soft_reset.
12. (Step 4 impl) **ACT / WGT alignment:** both reads are issued in the same cycle with the same documented latency (L_MEM_ACC); the WGT word then rides in the rotator token to the array. A simulation assertion checks the two valids are identical every cycle.
13. (Step 4 impl) **Core OOC timing:** the worst path (+0.923 ns at 5 ns, post-synthesis OOC) is the config checker's priority encoder (fail_r → err_code, 14 LUT levels), not the datapath. A third check stage would fix it but would change C_START; left as is unless implementation timing requires it (then documented as a structural C_START change).

## D13 — Config checker pipelined: C_START 2 → 3 (2026-09-24, Step 4.5)
User decision. Reason: the Step 4 OOC worst path of `gos_core` (+0.923 ns at 5 ns, post-synthesis) was the checker's priority encoder (fail_r → err_code, 14 LUT levels), a timing risk for the 250/300 MHz variants. Structure: S_CHK1 registers the per-layer rule comparisons (fail_r); S_CHK2 registers, per layer, the OR-reduce (any failure) and the first failing rule id; S_CHK3 selects the first failing layer (8-way) and sets ERR_CODE / refuses the job or enters S_LOAD. One extra cycle per job: **C_START = 3** (was 2); C_PIPE and C_DONE unchanged. The model (`gos_cycle_model.py`) and the test tying C_START to the RTL (`N_CHK`, and the number of S_CHK states) were updated and committed before any simulation of the new RTL. Expected job totals from the model: LeNet-5 16,288 + 5·29 + 3 = 16,436; CIFAR-10 104,128 + 4·29 + 3 = 104,247 (to be verified by RTL simulation).

## D14 — Implementation variant selection (2026-09-25, Step 4.5, user rule)
Full design (gos_top_wrapper in the bd_shell.tcl block design), pl_clk0 from the RPLL at the nearest achievable frequency, reported exactly. **Baseline = 200 MHz** (mandatory). **Performance bitstream = the highest variant with WNS ≥ 0, WHS ≥ 0 and 0 critical warnings.** Per variant: default strategy first, one retry with Performance_Explore + phys_opt_design if it fails. Allowed fixes for a failing higher clock: only register insertion that does not change cycle counts (AXI/CSR side, fanout duplication, CSR→core descriptor path); anything that would change cycles → stop and report. Records: `v2/results/impl_gos.csv` (source=post_impl); .bit/.hwh in `v2/vivado/out/gos_<MHz>/` (gitignored), SHA256 in the CSV. 250/300 MHz run only on user request after the 200 MHz report.
## D15 — Step 4.5 static checks: synthesis-log waivers (2026-09-25)
Source: `scripts/synth_scan.py` on the gos_top synthesis log (`run_netlist_sim.sh`, `build/netlist/synth_scan.txt`) with no per-ID message limit (`messaging.defaultLimit 100000`; the default limit of 100 had truncated Synth 8-3332 and 8-7129 in the first run). Result: **0 latches, 0 multi-driven, 0 undriven** nets; 666 "removed logic" messages in 26 groups, each reviewed against the RTL and waived below (`vivado/synth_waivers.txt`, which the scan matches; any new group fails the scan). None is a real defect.
1. **gos_requant product registers (8-3332 mp4[l][47:17] ×8 lanes, 8-6014 mp2[l], 8-3936 mp2/mp3/mp4 59 → 58 bits).** Each lane's 26×33 product is two cascaded DSP48E2 (DSP Report: `mp3_reg[l]` = A2*B2, `mp4_reg[l]` = (PCIN>>17)+A*B, with v1/mp2/mp3/mp4 absorbed as A/B/M/P registers). The fabric copies of the absorbed bits are removed; the product is 58-bit signed (|p| < 2^57, P_W = 58), so bit 58 of the 59-bit declaration is redundant.
2. **gos_requant relu_q / sz_q (8-3936, 6 → 5 bits).** The L_RQ = 6 control shift registers are read only at tap [4] (S6 register stage); tap [5] has no load.
3. **gos_ctrl descriptor latches (8-6014 f_ic, f_oc, f_kh, f_kw, f_oh, f_out_h, f_oc_tiles, f_ow_tiles, f_wgt_base, f_qp_base; 8-3936 f_k 32 → 16).** These fields are consumed at S_LOAD through their derived registers (ic_m1, kh_m1, kw_m1, owt_m1, oct_m1, rows_m1, rem, ch_rem, wgt, wgt_tile, qp_tile). f_k is compared against the 16-bit k counter; K ≥ 65536 is impossible for an accepted job because the checker bounds WGT_END ≤ 16383 and WGT_END = WGT_BASE + OC_TILES·K − 1 is host-derived (D8-1).
4. **gos_ctrl issue stage (8-6014 iss_oc_tile, iss_tok[dy]).** iss_* debug outputs are left open in gos_core; the named token flop disappearing is normal synthesis (register mapping/renaming), not investigated further — the functional evidence is the post-synthesis netlist simulation (`rtl_netlist.csv`).
5. **Unused descriptor input bits (8-7129 gos_ctrl desc, gos_cfg_check d).** By FORMATS.md §5: gos_ctrl ignores IH/IW (w1), KH/KW bits 31:16 (w2), WGT_BASE/QP_BASE bits 31:16 (addresses are 14/8 bits; bounded by the checker's END rules), flags bits 31:4 (w6), OUT_W (w10[15:0]) and the END words (w12–w15), which only the checker reads; gos_cfg_check reads only the fields its rules compare (it does not read WGT_BASE/QP_BASE, relu_en, in_sel or the zero bits).
6. **gos_csr AXI inputs (8-7129 s_axi_awaddr/araddr[1:0], s_axi_awprot/arprot).** Registers are 32-bit words (byte-offset address bits unused); gos_csr ignores AXI protection (`unused_ok = ^{s_axi_awprot, s_axi_arprot}`, gos_csr.sv).
7. **Post-synthesis netlist simulation setup.** In the funcsim netlist every flop is held by `glbl.GSR` for the first 100 ns; the netlist TBs released reset and issued the first AXI read (VERSION) inside that window, so the CSR accepted the address and never answered (the TB hung; not an RTL defect — the RTL has no GSR and on the board GSR is released at configuration). The TBs now wait for `glbl.GSR` release under `NETLIST`. Scope (user decision 2026-09-25, gate-level xsim runs ≈ 500 cycles/s): `tb_gos_top` (1 LeNet + 1 CIFAR image) and `tb_gos_top_backtoback` with NJOBS=3, J_RESET=1, J_REFUSE=−1 (LeNet; CIFAR with a mid-job soft_reset, then the same job again; LeNet). The checker-refused job is covered by the RTL back-to-back run (100 jobs).
8. **Full-design synthesis logs.** In the BD build `gos_top_wrapper` is synthesized in its own OOC run (`gos_system_gos_top_0_0_synth_1`), not in synth_1. From commit after c88e71a the no-limit hook is set on every synthesis run and that log is kept as `synth_gos_top.log` (scanned by `impl_collect.py`). The 200 MHz build of c88e71a predates this: its gos_top OOC log is capped at 100 messages per ID; the untruncated scan of the same RTL is the gos_top netlist synthesis above.

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
