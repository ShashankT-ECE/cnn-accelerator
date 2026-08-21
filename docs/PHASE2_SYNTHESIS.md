# Phase 2 — Synthesis / Implementation / Timing

> **Status:** VERIFIED (2026-08-21). Vivado ML 2023.1, target
> `xck26-sfvc784-2LV-c` (AMD Kria KV260 / XCK26). Top = `cnn_accelerator_v2`,
> OOC (out-of-context), 200 MHz (5.000 ns) baseline constraint
> (`build/phase2/baseline.xdc`).

---

## 1. Result summary

| Metric | Value |
|---|---|
| Clock | 200 MHz (5.000 ns) |
| **WNS (setup)** | **+0.030 ns** (0 failing endpoints) |
| **TNS** | **0.000 ns** |
| **WHS (hold)** | **+0.073 ns** (0 failing endpoints) |
| THS | 0.000 ns |
| Pulse width | 0 failing (worst slack +1.738 ns) |
| Unrouted nets | 0 (route completed) |
| Congestion | none above level 5 |

The reconfigurable research system **closes 200 MHz**, but with essentially no
setup margin (+0.030 ns) — the critical path is the high-fanout cycle-counter
`→` per-PE control decode (routing-dominated). The frozen V2 array alone closes
200 MHz at WNS +2.861 ns (`PROJECT_STATE.md` Decision 12); the reconfigurable
controller + padding + sparsity logic consumes that margin. This is consistent
with the timing risk identified in `docs/RESEARCH_READINESS_PLAN.md` §7, which
anticipated ~150 MHz for the reconfigurable system; Phase 2 meets 200 MHz but
with zero comfort margin (see §5).

---

## 2. Resource utilization

| Resource | Used | Available | % |
|---|---|---|---|
| CLB LUTs | 11,245 | 117,120 | 9.60 |
| — LUT as Logic | 10,595 | — | 9.05 |
| — LUT as Memory (distributed RAM) | 650 | 57,600 | 1.13 |
| Flip-flops (FDRE) | 5,726 | 234,240 | 2.44 |
| DSP48E2 | 67 | 1,248 | 5.37 |
| CARRY8 | 20 | 14,640 | 0.14 |
| BRAM | 0 | 144 | 0 |
| URAM | 0 | 64 | 0 |

Notes:

- **DSP48E2 = 67** = 64 PE MACs (1/PE, `use_dsp` attribute in the frozen
  `pe_v2`) + **3 address-arithmetic DSPs** from the controller's constant-factor
  arithmetic (`y*28`, `ch*784`/`r*784`, and the `img_wr_addr/28` decode in the
  non-zero-mask write). These 3 are a controller cost, not a datapath cost, and
  could be moved to LUTs (28 = 4·7) if DSP budget mattered.
- **BRAM/URAM = 0**: the 784×8 image is distributed RAM (650 LUTs as LUTRAM);
  the 200×8 weight store is a register file (fabric FFs).
- **CARRY8 = 20**: the 32-bit sparsity/cycle counters (4× 32-bit adders ≈ 16)
  plus small address adders. The PE arithmetic is all in DSPs (0 CARRY8 in the
  MAC datapath).

### 2.1 DSP register configuration (post-route)

| Count | AREG | BREG | MREG | PREG | USE_MULT | Role |
|---|---|---|---|---|---|---|
| 48 | 2 | 0 | 1 | 1 | MULTIPLY | PE MAC (activation shift-chain absorbed) |
| 8 | 1 | 0 | 1 | 1 | MULTIPLY | PE MAC |
| 8 | 0 | 0 | 1 | 1 | MULTIPLY | PE MAC |
| 3 | 0–1 | 0 | 0 | 0 | MULTIPLY/NONE | controller address arithmetic |

**Notable difference from the frozen V2 baseline (Decision 12):** the PE MACs
here show **BREG=0** (weight register in fabric), whereas the frozen
`systolic_array_v2` OOC synthesis showed **BREG=1** (weight in the DSP B
register). The reconfigurable controller's **mode-muxed weight path**
(`w_out = dataflow_mode ? w_ws : w_os`) prevents Vivado from inferring the DSP
BREG — the same class of mux-interference recorded for PCIN/PCOUT in Decision 12.
This is an implementation detail, not a functional change: the weight is still
registered (in fabric) and the simulation is bit-exact. MREG=1/PREG=1 are
preserved for all 64 PE MACs.

---

## 3. Critical path

```
Slack (MET)  : +0.030 ns
Source       : s_reg[2]/C                      (cycle counter)
Destination  : u_array/.../pe_inst/accumulator_reg_i_..._psdsp/D   (PE accumulator DSP input)
Data Path    : 4.966 ns  (logic 1.251 ns [25.2%]  route 3.715 ns [74.8%])
Logic Levels : 8  (LUT4=1 LUT6=6 MUXF7=1)
```

The path is **routing-dominated** (74.8%) through the high-fanout cycle counter
`→` per-PE control decode (weight/accumulate control into the DSP). This is the
residual cost of fanning the controller's `s`/`weight_load`/`accum_clear` out to
64 PEs.

---

## 4. Timing closure journey (diagnosis, not clock-lowering)

Three architectural fixes took the integrated design from failing to closing:

| Iteration | WNS | Fix |
|---|---|---|
| baseline | −3.531 ns | (combinational distributed-RAM image read → sparsity counter) |
| + registered stream read | −2.581 ns | look-ahead address + register the 5 image-read streams |
| + registered `ws_skip` | −0.207 ns | latch the 60-bit zero-window reduction at group boundary |
| + pass/sub-cycle counters | **+0.030 ns** | remove divide-by-13 (`(s-1)/13`,`%13`) from the address/activation path |

Each fix was diagnosed from the actual worst path and, where possible, applied
architecturally (registering long combinational paths, removing non-power-of-two
division) rather than by lowering the clock. The final path is routing-limited,
not logic-limited.

---

## 5. Honest assessment

- 200 MHz **closes**, but at WNS +0.030 ns — no safety margin. Any subsequent
  controller change, or a different PVT corner, could tip it negative.
- The architecture is timing-clean at **~190 MHz** with comfortable margin; the
  research plan's §7 estimate of **150 MHz** for the reconfigurable system is
  therefore comfortably conservative.
- If a solid 200 MHz margin is required in Phase 3 (AXI integration adds
  control-path logic), the next lever is to **pipeline the controller→array
  control fan-out** (register `weight_load`/`accum_clear`/`act_out`) or reduce
  the `s`-counter fanout (a gray/one-hot cycle counter). Both are localized to
  the controller and leave the frozen array untouched.

---

## 6. Reproducibility

```bash
source ~/Xilinx/Vivado/2023.1/settings64.sh
vivado -mode batch -source build/phase2/synth_impl.tcl
# reports -> build/phase2/00..11_*.rpt / *.txt
```
