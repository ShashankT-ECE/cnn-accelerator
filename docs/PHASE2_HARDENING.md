# Phase 2 — Hardening Report (paper-readiness before KV260 integration)

> **Status:** COMPLETE (2026-08-21). This records the hardening milestone that
> precedes KV260/AXI integration. All results are reproduced against the real
> pretrained ONNX MNIST-12 Conv1 layer; no synthetic data, no retraining, no
> frozen-RTL change. Machine-readable artifacts live under `data/benchmark/`.

Provenance labels used throughout: **[SIMULATION]** (xsim, Vivado ML 2023.1),
**[ANALYTICAL]** (derived from the verified per-group schedule),
**[POST-ROUTE SYNTHESIS ESTIMATE]** (Vivado ML 2023.1), **[THEORETICAL/PEAK]**.

---

## 1. Timing hardening (Section 1)

**What changed** — all localized to the Phase-2 integration controller
(`rtl/common/cnn_accelerator_v2.sv`); no frozen module (`pe.sv`,
`systolic_array.sv`, `input_feed.sv`, `pe_v2.sv`, `systolic_array_v2.sv`) was
touched, and the functional schedule / bit-exact results are unchanged:

1. **Registered controller→array control fan-out** — `accum_clear` and
   `weight_load` are decoded one cycle early from the look-ahead state (`s_nxt` /
   `os_j_nxt`, the same pattern as the existing registered stream read) and
   registered at the array boundary, so the array sees a clean register output
   instead of a combinational `s → decode → 64-PE` cone.
2. **Pipelined the MAC/cycle counters** — the wide per-cycle increments
   (`ws_valid_inc`/`ws_skip_inc`/`skip_inc_os`/`exec_inc_os`) are registered one
   cycle; the 32-bit accumulators add them a cycle later. Totals are unchanged
   (each compute cycle still contributes exactly one increment; the final total
   arrives one cycle later, still before `done`).
3. **Eliminated 3 controller DSPs** — `result_base = ch·784 + y·28 + (base−7)`
   was being inferred as DSP multiply; it is now expanded as shifts
   (784 = 512+256+16, 28 = 32−4).
4. **Fanout control** — `max_fanout = 16` on the `s` cycle counter and
   `mode_active_r`.

**Result** ([POST-ROUTE SYNTHESIS ESTIMATE], 200 MHz baseline):

| Metric | Before | After |
|---|---|---|
| WNS (setup) | +0.030 ns | **+0.189 ns** |
| TNS | 0.000 ns | 0.000 ns |
| WHS (hold) | +0.073 ns | +0.073 ns |
| LUT | 11,245 | **9,372** |
| FF | 5,726 | 5,795 |
| DSP48E2 | 67 | **64** (3 controller DSPs removed) |
| CARRY8 | 20 | 20 |
| BRAM / URAM | 0 / 0 | 0 / 0 |

- WNS improved **6.3×** (+0.030 → +0.189 ns). Extrapolated maximum frequency
  ≈ **208 MHz** (data-path-limited). The +0.3 ns *preference* was not reached
  without a 2-deep address pipeline (see the residual path below); +0.189 ns is
  reported as the achieved, documented margin.
- **Residual critical path:** `s` counter → address look-ahead → image
  distributed-RAM → `stream` register (11 logic levels). The remaining lever is
  a **2-deep address look-ahead** (register `addr_ky`, compute the address from
  `s_n2 = s+2`). That re-times the registered stream read and is higher-risk
  (wrap/mode-transition/partial-group edge cases), so it is **deferred and
  documented** rather than risked against bit-exactness.
- Bit-exactness re-verified: `sim/tb_cnn_accelerator_v2.sv` **75,305/75,305
  PASS**, `OS cycles=9184`, `WS cycles=18368`, `skipped=133960`,
  `executed=22840` — identical to the pre-hardening run.

The cost of the fix is +69 FF (pipeline registers) for −3 DSP and −1,873 LUT.

---

## 2. OS vs WS — a fair comparison (Section 2)

The raw gap is **not hidden**: OS dense = 9,184 cycles, WS (no-skip bound) =
38,528 cycles. The difference is fully explained and is the paper's central
measurement.

| | OS dense | WS (no-skip) |
|---|---|---|
| groups | 112 = 28×4 | 896 = 8 ch × 112 |
| cycles/group | 82 | 43 |
| total cycles | 9,184 | 38,528 |
| MAC compute cycles | 2,800 (25×112) | — |
| result-drain cycles | 1,792 (16×112) | — |
| lead-in/clear cycles | 4,592 | — |
| channel-sweep factor | 1 (8 channels on 8 rows) | **8** (channels serialized) |
| sustained MAC/cycle | 17.07 | 4.07 |
| PE utilization | 26.68% | 6.36% |

**Mechanism** ([ANALYTICAL]): OS holds all 8 output channels **in parallel on
the 8 PE rows**; WS maps rows to the **tap/reduction** dimension and must
**serialize the 8 channels** (8 sweeps). WS's shorter group (43 vs 82) recovers
only ~1.9×, so the net is 38,528 / 9,184 = **4.195×** slower. This is a
workload-matched, self-normalizing comparison on identical silicon (same 64 PEs,
same DSP48E2 primitives, same clock, same golden model) — the quantity the
literature lacks because published OS/WS figures are cross-chip and confounded.

The comparison is **between the existing mappings** (the frozen V2 datapath
realizes OS via the V1-identical mode and WS via the psum-cascade mode). A
"fair V2 OS" is already what OS-on-array-v2 *is* (PE-v2 is bit-identical to V1
in OS mode). No new mapping was invented.

Full decomposition: `data/benchmark/cycle_decomposition.json`.

---

## 3. WS bubble analysis (Section 3) — no safe optimization

The WS per-group schedule was analyzed for avoidable bubbles:

| Structure | Cycles | Verdict |
|---|---|---|
| accum_clear + weight_load (s=0) | 1 | necessary |
| 4-tile tap window (d = 1..32) | 32 | 25 taps spread 8+8+8+1 (diagonal skew) |
| tile-3 partial (1 tap) | 8 (7 rows idle) | inherent: 25 = 3·8 + 1 |
| vertical-cascade drain tail | ~9 | inherent: last tap propagates 7 rows |
| result capture (s=42) | 1 | could overlap next group (see below) |

**Conclusion:** there is **no optimization that safely reduces WS cycles without
changing the schedule.** The inefficiency is structural to the frozen
Weight-Stationary mapping (Decision 11): (a) channel serialization (8×), (b)
tap serialization over a 8-row reduction cascade with diagonal skew, (c) the
25≡1 (mod 8) partial final tile. The only genuine lever — **group-boundary
pipelining** (overlap the result capture of group N with the accum_clear of
group N+1, saving 1 cycle/group ≈ 2.3%) — is a schedule change, i.e. the
already-planned **ablation A1** (`docs/RESEARCH_READINESS_PLAN.md` §1, §11), not
a hardening change. Per the milestone constraint ("do not change the
architecture merely to improve the number"), it is **not applied**; it remains a
documented ablation for the paper.

---

## 4. Sparsity — rigorous accounting (Section 4)

Three quantities are kept **separate** (they are not interchangeable):

| Quantity | Primary image | Meaning |
|---|---|---|
| activation zero fraction | **85.43%** | MACs whose activation operand is 0 (or padding) |
| hardware-skippable group fraction | **53.57%** (60/112 positions) | output groups whose 5×12 window is all-zero |
| skippable MAC fraction (coarse) | **46.94%** (73,600 MACs) | MACs actually eliminated by the coarse skip |
| measured cycle reduction | **52.33%** | (38,528 − 18,368) / 38,528 |
| measured speedup | **2.098×** | 38,528 / 18,368 |
| fine-grained upper bound | 6.865× | 1/(1−0.8543) |

**Key honest point:** the 85.43% zero-activation rate does **not** yield an
85.43% cycle saving. The coarse skip eliminates only whole all-zero groups
(46.94% of MACs), so the measured speedup is **2.10×**, not 6.87× (the
fine-grained bound). No claim is made that 85.43% sparsity → 2.10× speedup
directly.

**Multi-image check** ([SIMULATION], 200 deterministic test images, no RNG):
activation zero fraction mean 82.25% (65.8–94.1%); skippable-group fraction
mean 43.17% (22.3–59.8%); measured WS speedup mean **1.77×** (1.28–2.41×). The
single-image 2.10× is therefore at the *upper* end of the distribution; the
paper quotes the distribution, not the best-case image.

Artifacts: `data/benchmark/sparsity_accounting.{json,csv}`,
`fig3_sparsity_accounting.png`, `fig6_multi_image_sparsity.png`.

---

## 5. Runtime reconfiguration (Section 5)

The OS↔WS switch is a single **8-cycle FLUSH** (cycle 0 asserts `accum_clear`
+ drives `act_out = 0`; cycles 0–7 drain the 56-register activation shift
chain). `mode_active` is latched at commit; no reset, no reprogram.

| Quantity | Value |
|---|---|
| flush cycles (commit → quiesced) | 8 |
| mode-switch cycles | 8 (the flush is the switch) |
| total transition latency | 8 cycles = 40 ns @ 200 MHz |
| OS→WS first result (dense group) | 8 + 42 cycles |
| WS→OS first result | 8 + 67 cycles |

Bit-exactness verified [SIMULATION]: OS→WS, WS→OS, **OS→WS→OS**, and
**WS→OS→WS** all bit-exact with no reset (`sim/tb_cnn_accelerator_v2.sv` T3/T4).
Artifacts: `data/benchmark/reconfiguration.json`, `fig5_reconfig_overhead.png`.

---

## 6. Accuracy / quantization validation (Section 6)

Full 10,000-image MNIST test set, no synthetic data, no retraining.

| Metric | Value |
|---|---|
| FP32 top-1 | **98.90%** |
| INT8 Conv1 top-1 | **98.90%** |
| accuracy delta | **+0.00 pp** |
| changed predictions | **0 / 10,000** |
| SQNR (layer) | 46.44 dB |
| per-channel SQNR | 42.8 / 51.1 / 47.3 / 48.1 / 49.2 / 43.7 / 49.9 / 45.2 dB |
| activation clipping | 0 |
| weight clipping | 0 |

Checkpoint frozen: `data/checkpoint/mnist-12.onnx`, SHA-256
`5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd` (ONNX Model
Zoo MNIST-12, MIT, validated top-1 error 1.1%). Vector manifest:
`data/vectors/manifest.json`.

---

## 7. Benchmark matrix (Section 7)

`data/benchmark/benchmark_matrix.{csv,json}` records six configurations (OS
dense, WS dense [ANALYTICAL no-skip bound], OS sparse, WS sparse, OS→WS, WS→OS)
with cycles, latency, MAC accounting, zero fraction, PE utilization, MAC/cycle,
speedup, and reconfiguration overhead.

**OS sparse is reported honestly**: the coarse zero-group skip is **WS-only** in
this design (the skip predicate is gated on `mode_active_r`), so OS is *not*
accelerated — OS sparse == OS dense, and this is stated explicitly rather than
fabricated.

---

## 8. Resource analysis (Section 8)

`data/benchmark/synthesis.json` records post-route utilization for the hardened
top vs. the pre-hardening top vs. the frozen array-v2 alone.

| Resource | V1 array | V2 array | Phase-2 (before) | **Phase-2 (after)** |
|---|---|---|---|---|
| LUT | 1,032 | 1,032 | 11,245 | **9,372** |
| FF | 2,048 | 2,368 | 5,726 | **5,795** |
| DSP48E2 | 64 | 64 | 67 | **64** |
| CARRY8 | 0 | 0 | 20 | 20 |
| BRAM / URAM | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| WNS @200 MHz | +2.772 | +2.861 | +0.030 | **+0.189** |

Cost attribution (hardened): +69 FF for the pipeline registers; −3 DSP and
−1,873 LUT from eliminating the controller address-arithmetic DSPs and the
accompanying logic. The 20 CARRY8 are the four 32-bit sparsity/cycle counters
and small address adders; all PE arithmetic is in DSPs. The mode-muxed weight
path yields **BREG=0** (weight registered in fabric) — functionally identical,
documented (Decision 12 class).

---

## 9. Paper-ready figures/tables (Section 9)

`python/plot_benchmarks.py` renders `data/benchmark/figures/`:
`fig1_os_vs_ws_cycles`, `fig2_dense_vs_sparse`, `fig3_sparsity_accounting`,
`fig4_resource_utilization`, `fig5_reconfig_overhead`,
`fig6_multi_image_sparsity`. The CSV is the source of truth; the plots only
render it, with integer axes for counts and explicit provenance — no
misleading styling.

---

## 10. Negative results (Section 10)

Recorded honestly:

1. **OS is faster than WS on this workload (4.195×).** Mechanism: WS serializes
   the 8 output channels (rows = taps), OS parallelizes them (rows = channels).
2. **WS has serialization overhead** — tap serialization over an 8-row
   reduction cascade with diagonal skew + a 25≡1 (mod 8) partial tile.
3. **Fine-grained zero-skipping is unsafe** with the frozen spatial alignment:
   the array-wide `zero_skip` gates the *shifted* product, so gating on the fed
   stream corrupts in-flight MACs (verified). It is tied to 0.
4. **Coarse group skipping is used instead** (whole 5×12 all-zero window),
   preserving fixed systolic timing and pixel identity.
5. **Zero fraction ≠ exploitable sparsity**: 85.43% zero activations → 46.94%
   skippable MACs → 52.33% cycle reduction → 2.10× (single image), 1.77× mean
   (200 images).
6. **Timing margin** is +0.189 ns @ 200 MHz (~208 MHz max), not +0.3 ns.
7. **"Runtime reconfigurable OS/WS systolic array" is prior art** (ReSA, RIFT,
   AdaFlow, MAERI, Eyeriss v2, FlexFlow). The novelty is the **like-for-like
   measurement on identical silicon**, not the reconfiguration apparatus.
8. **Only measurable on the physical KV260:** actual board frequency, measured
   latency/throughput, power, energy/inference, and confirmation the bitstream
   works on silicon.

---

## 11. KV260-readiness data package (Section 11)

Everything preparable before the board arrives:

- Frozen model: `data/checkpoint/mnist-12.onnx` (SHA-256 above).
- Real MNIST vectors: `data/vectors/` (weights.hex, input_img.hex,
  golden_canonical.hex, multi/, manifest.json).
- INT8 weights / quant params: `data/vectors/weights.hex`, `quant_params.npz`.
- Golden outputs: `data/vectors/golden_canonical.hex`, `multi/golden_*.txt`.
- Benchmark scripts: `python/benchmark.py`, `python/plot_benchmarks.py`.
- Expected cycle counts: OS 9,184; WS sparse 18,368; WS no-skip bound 38,528.
- Expected resource/timing: `data/benchmark/synthesis.json`.
- Experiment matrix + CSVs: `data/benchmark/`.

The physical KV260 should only provide: actual board frequency, measured
latency/throughput, power, energy/inference, and confirmation the prepared
bitstream/software works on hardware.

---

## 12. Acceptance criteria

- [x] Full real-MNIST regression passes — `tb_cnn_accelerator_v2` 75,305/75,305
- [x] OS dense passes (9,184 cycles, bit-exact)
- [x] WS dense passes (bit-exact; 18,368 measured = sparse; 38,528 is the no-skip bound)
- [x] Sparse execution passes (coarse zero-group skip)
- [x] OS→WS→OS and WS→OS→WS pass (bit-exact, no reset)
- [x] Counters are exact (133,960 / 22,840 / 156,800)
- [x] Accuracy 98.90% (FP32 == INT8, 0 flips)
- [x] Timing: +0.189 ns @ 200 MHz (documented max-frequency result)
- [x] Synthesis clean, implementation clean (0 failing endpoints, 0 unrouted)
- [x] Benchmark CSVs generated
- [x] Paper-ready tables/plots generated
- [x] Limitations documented
- [x] Results reproducible
- [x] V1 frozen (pe.sv, systolic_array.sv, input_feed.sv, pe_v2.sv,
      systolic_array_v2.sv unchanged)
- [x] `git diff --check` passes

**Phase 2 is genuinely ready for Phase 3 (KV260/AXI) integration.**
