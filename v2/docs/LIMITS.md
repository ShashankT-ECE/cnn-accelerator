# V2 descriptor limits: hardware vs. config checker vs. host

**Status:** audit of every descriptor field (2026-09-30). RTL is frozen; nothing here changes it.
Descriptor layout, checker rules 1–27 / 32 and the host-responsibility list: [FORMATS.md §5](FORMATS.md#5-layer-descriptor-16-x-32-bit-words-per-layer).

Columns:
- **hardware limit**: what the frozen RTL actually supports, with `file:line` evidence
  (`v2/rtl/…`). "analysis" = derived from RTL widths / the rules, not simulated.
- **RTL checker**: rule id(s) of `gos_cfg_check.sv` that bound the field. "via END" means the
  bound holds only if the host keeps the END field consistent with the raw fields (the checker
  cannot multiply, and the datapath never reads END fields).
- **host assert**: `v2/model/gos_pack.py`. `encode_descriptor` = field widths + derived-field
  consistency (also used for expected-refusal jobs). `assert_host_limits` / `assert_job_limits` /
  `assert_descriptor_words` = the limits the checker does not enforce, called from
  `make_descriptors` (nets, board data, full-10k sim) and from `gen_shapes.finalize_structure`
  for valid jobs only (expected-refusal jobs still encode). Tests: `v2/model/tests/test_gos_pack.py`
  (`test_assert_host_limits`, `test_assert_job_limits`, `test_assert_descriptor_words_reserved`, …).
- **RTL-sim boundary test**: case ids of `v2/shapes/limit_cases_fields.py` (prefix `F_`, run by
  `v2/scripts/run_limits.sh`, Verilator + xsim, `tb_shapes.sv`). Observed = RTL sim,
  **dirty-tree run, final in `v2/results/limits_rtl.csv`** (the main session re-runs clean).
  The golden is always that of the *intended* layer; `desc_patch` cases bypass the host on purpose.

| field | width | hardware limit (evidence) | RTL checker | host assert | RTL-sim boundary test (observed) | status |
|---|---|---|---|---|---|---|
| IC | 16 | `ic`/`ic_m1` 16-bit (gos_ctrl.sv:61,117). Effective: IC·IN_PLANE ≤ 4096, IC·KH·KW = K ≤ 16384 (WGT), K ≤ 2047 on requant layers (V_MUL_W row) | 8; 5 via IN_END; 4 via WGT_END | encode_descriptor; assert_host_limits (K on requant layers) | F_IC_2047_req exact; F_IC_4096_raw exact; F_IC_4097_raw refuse (rule 5) | enforced (via END, given consistency) |
| OC | 16 | `ch_rem`, `oc_tile` 16-bit (gos_ctrl.sv:63,129); QPARAM channel 8-bit, masked tail channels wrap at 256 (gos_core.sv:310, gos_qparam_mem.sv:18); token `oc_tile[7:0]` feeds only the LOGIT index `oc_tile[0]` (gos_ctrl.sv:213, gos_core.sv:361) → OC ≤ 256, out_raw OC ≤ 16 | 9; 7 via QP_END; 6 via OUT_END; 27 | encode_descriptor; assert_host_limits (out_raw OC ≤ 16, duplicate) | F_OC_256 exact; F_OC_257 refuse (rule 7) | enforced (via END) |
| IH | 16 | never read by the datapath (gos_ctrl.sv:236); only via IN_PLANE (IN_END) → ≤ 4096; `oy`/`out_row` 16-bit (gos_ctrl.sv:63-64) | 10, 25 | encode_descriptor (IN_PLANE) | F_IH_4096 exact; F_IH_4097 refuse (rule 5); F_IH_field_ignored exact (IH patched: no effect) | enforced (via END); field itself checker-only |
| IW | 16 | never read by the datapath; via IN_WPR → IW ≤ 32768; `ox_tile`/`rem` 16-bit. At IN_END = 4095 the rotator over-read hits word 4096 and wraps to 0 (gos_core.sv:221), masked rows only | 11, 26 | encode_descriptor; assert_host_limits (`max_act_read_word ≤ IN_END+1`) | F_IW_32768 exact (4096 x tiles, wrap exercised); F_IW_32776 refuse (rule 5) | enforced (via END) |
| KH | 8 | `ky`/`kh_m1` 8-bit (gos_ctrl.sv:60,118); ky only steps `ky_off += IN_WPR` (gos_ctrl.sv:143), no rotation → full 1..255 | 12 | encode_descriptor (< 256); assert_host_limits (KH ≤ IH, OH = IH−KH+1) | F_KH_255 exact. KH 9/10/16: `limit_cases_kw.py` (other agent) | OK, no limit below the field width |
| KW | 8 | **KW ≤ 9**: `iss_rd[b] = ky_off + (b < kx)`, `iss_rot = kx[2:0]` (gos_ctrl.sv:207-208) exact only for kx ≤ 8 — [DECISIONS OC-3](DECISIONS.md) | 13 (KW = 0 only) | assert_host_limits: KW ≤ MAX_KW = 9 (new, `test_kw_limit`) | KW08_*, KW09_* (6 cases incl. chain, raw head) exact; KW10/11/12/16_* mismatch (cycles exact) | host-only (now asserted) |
| OH | 16 | non-pool row loop bound `rows_m1` (gos_ctrl.sv:122). Must equal IH−KH+1: a larger OH reads past the channel plane and writes over the next output channel | 14, 25 (OH ≤ IH only), 1 (pool parity) | **assert_host_limits: OH == IH−KH+1 (new; `derive_fields` never checked it although FORMATS.md said so)** | F_OH_over mismatch | host-only (now asserted) |
| OW | 16 | row mask `rem` (gos_ctrl.sv:128,218). Must equal IW−KW+1 | 15, 26, 2 | **assert_host_limits: OW == IW−KW+1 (new)** | F_OW_over mismatch | host-only (now asserted) |
| WGT_BASE | 32 | ctrl uses bits 15:0 (gos_ctrl.sv:130), WGT address `iss_wgt[13:0]` (gos_core.sv:248) | 4 via WGT_END | encode_descriptor; assert_job_limits (no WGT overlap between layers) | F_bases_top exact (WGT_END = 16383) | enforced (via END) |
| QP_BASE | 32 | bits 15:0 → `qp_tile`, channel bits 7:0 (gos_ctrl.sv:131,222) | 7 via QP_END | encode_descriptor; assert_job_limits (no QPARAM overlap) | F_bases_top exact (QP_END = 255, tail reads wrap) | enforced (via END) |
| relu_en | 1 | ignored on out_raw: LOGIT takes `v_raw`, independent of relu_en (gos_requant.sv:8,170) | — | assert_host_limits (relu_en = 0 with out_raw) | F_raw_relu exact | harmless; asserted as contract |
| pool_en | 1 | with out_raw both dy tiles write LOGIT (gos_core.sv:360) → wrong | 1, 2 (parity only) | assert_host_limits (no pool with out_raw) | F_raw_pool mismatch | host-only |
| out_raw | 1 | `LOGIT[{oc_tile[0], col}] <= v` of row 0 for **every** tile (gos_core.sv:360-361): only a 1x1 output is defined; a non-final out_raw layer writes no ACT, so the next layer reads stale data | 27 (OC > 16) | **assert_host_limits: OH = OW = 1 (new)**; assert_job_limits (last layer only) | F_raw_OH2 mismatch; F_raw_OW9 mismatch; F_raw_OW2 exact (LOGIT = ox 0 = what is compared; ox 1 silently lost) | host-only (now asserted) |
| in_sel | 1 | selects the read buffer, writes the other (gos_core.sv:228-237); any value runs | — | assert_job_limits (in_sel = i mod 2) | F_in_sel_same mismatch | host-only |
| K | 32 | field used only as `f_k[15:0]`, the WGT step between OC tiles (gos_ctrl.sv:194); tile length comes from the IC/KH/KW counters; 16-bit `k` only drives `iss_first` (gos_ctrl.sv:210). Real IC·KH·KW ≥ 8 (drain K_MIN, gos_array.sv:21-25) | 3, 16; 4 via WGT_END | encode_descriptor (K = IC·KH·KW); assert_host_limits (requant K ≤ 2047) | F_K_16384_raw exact; F_K_16384_oc9 refuse (rule 4); F_K_field_oc8 exact; F_K_field_oc16 mismatch; **F_K_real_4 timeout** (real K = 4 with K field 8: core never finishes) | consistency host-only; see finding 1 |
| IN_WPR, IN_PLANE | 16 | `ky_off`/`ic_base` steps (gos_ctrl.sv:143,147) | 17, 18 | encode_descriptor | (IW/IH/IC cases) | derived, host |
| OC_TILES, OW_TILES | 16 | loop bounds (gos_ctrl.sv:120-121) | 19, 20 | encode_descriptor | F_OC_256, F_IW_32768; F_OW_over (patched) | derived, host |
| OUT_W | 16 | never read by the datapath (gos_ctrl.sv:236) | 21 | encode_descriptor | F_OUT_W_field_ignored exact | checker-only field |
| OUT_H, OUT_WPR | 16 | pool row bound (gos_ctrl.sv:122), `out_row_off` step (gos_ctrl.sv:175) | 22, 23 | encode_descriptor | F_raw_pool (patched) | derived, host |
| OUT_PLANE | 16 | used as `[12:0]<<3` (gos_ctrl.sv:198) and `[11:0]` (gos_core.sv:297): exact mod 4096 since ≤ 4096 via OUT_END | 24; 6 | encode_descriptor | F_IH_4096 exact (OUT_PLANE = 4096) | enforced (via END) |
| WGT_END / IN_END / OUT_END / QP_END | 32 | read only by the checker (gos_ctrl.sv:236) | 4 / 5 / 6 / 7 | encode_descriptor | F_WGT_END_0 exact, F_IN_END_0 exact (understated END: no effect); at the bound: F_IW_32768, F_IH_4096, F_bases_top exact | bounds are only as good as host consistency |
| reserved w2[31:16], w6[31:4] | 16 / 28 | ignored by checker and datapath (gos_cfg_check.sv:28,30; gos_ctrl.sv:106-110) | — | **assert_descriptor_words (new)** | F_w2_reserved exact; F_w6_reserved exact | harmless; host keeps 0 |
| N_LAYERS | 4 (CSR) | CSR keeps `w_data[3:0]` (gos_csr.sv:205): 16 reads as 0 (refused), 17 as 1 (runs layer 0) | 32 (0, 9..15) | **assert_job_limits: 1..8 (new)** | not simulated (tb_shapes drives the core port) | host-only for ≥ 16 |
| requant v (V_MUL_W) | 26 | multiply uses `v[25:0]` (gos_requant.sv:112): every reachable \|acc+q_bias\| < 2^25; worst case K·16384 + \|q_bias\| → K ≤ 2047 | — | load_params (with q_bias); **assert_host_limits: K·16384 < 2^25 on requant layers (new)** | F_IC_2047_req exact (random data, bound not stressed). K ≥ 2048 not buildable by gen_limits (make_layer_params asserts) | host-only; worst-case, data-dependent |
| raw v (out_raw) | 32 | 32-bit modular add (gos_requant.sv:91); K ≤ 16384 → \|acc\| ≤ 2^28 (analysis) | 4 via WGT_END | gos_golden asserts INT32 | F_K_16384_raw exact | OK |
| cycle counters | LAYER_CYC 32, TOTAL/MAC/STALL 64, inflight 16 | per layer T·K = (OC_TILES·K ≤ 16384) × (rows·OW_TILES·(pool ? 2 : 1) ≤ 4·OUT_PLANE ≤ 16384) ≤ 2^28 (analysis, from rules 4 and 6) plus C_PIPE, so `lcyc`/LAYER_CYC (gos_core.sv:81,190) cannot overflow; TOTAL_CYC 64-bit (gos_core.sv:154) | 4, 6 (implicitly) | not needed | — | no overflow reachable |

## Findings (RTL sim, dirty-tree run; final in `v2/results/limits_rtl.csv`)

1. **Inconsistent K field hangs the core.** A descriptor whose K field says ≥ 8 while IC·KH·KW < 8
   passes the checker (rule 3 tests the field); tiles then arrive faster than the 8-cycle drain,
   the array overruns, and in both simulators the job never finished (`F_K_real_4`: timeout,
   recovered by soft_reset). Reachable only if the host violates K = IC·KH·KW
   (`encode_descriptor` asserts it).
2. **Inside the checker-accepted space, silently wrong results** (all need a host contract
   violation; all now asserted by the host): OH/OW ≠ IH−KH+1 / IW−KW+1 (`F_OH_over`,
   `F_OW_over`), out_raw with OH or OW > 1 (`F_raw_OH2`, `F_raw_OW9`; `F_raw_OW2` compares equal
   only because the harness checks ox 0), out_raw with pool_en (`F_raw_pool`), K field ≠ IC·KH·KW
   with > 1 OC tile (`F_K_field_oc16`), in_sel not alternating (`F_in_sel_same`). KW ≥ 10: OC-3 (resolved: host assert KW ≤ 9).
3. Every field with a real width limit (IC, OC, IH, IW, K, OUT_PLANE, bases) is bounded by an
   END rule once the END fields are consistent; at-limit shapes ran exact and the just-over shapes
   were refused with the predicted ERR_CODE.
4. Fields the datapath never reads (IH, IW, OUT_W, END fields, reserved bits) have no effect
   when patched: they matter only for the checker.

Not simulated (analysis only): K ≥ 2048 on a requant layer with adversarial data, out_raw on a
non-final layer, N_LAYERS ≥ 16, WGT/QPARAM overlap.
