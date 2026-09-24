# FC Schedule Validation (Step 4.1)

> **Status:** RESOLVED (2026-09-23). Verifies the exact OS/WS cycle schedule for
> 1-pixel (FC) layers against the frozen 8×8 array RTL (`pe_v2.sv`,
> `systolic_array_v2.sv`, `cnn_accelerator_v2.sv`), resolving Step 3.5
> **discrepancy A**. Backed by a cycle-accurate simulation
> (`python/fc_schedule_sim.py`) that is bit-exact against the frozen L2 golden
> for OS. No RTL modified.

## 1. RTL timing facts (from the frozen array)

1. **Shift chain is per-column, not a fixed lead-in.** `systolic_array_v2.sv:70-93`:
   column 0 sees `act_in[r]` **combinational (0-cycle)**; column `c` sees it
   delayed `c` cycles (`shift_reg[r][c-1]`). The "7-cycle lead-in" is the time to
   fill the 8-column chain for **8 pixels** — a 1-pixel FC layer does not need it.
2. **PE accumulate latency.** `pe_v2.sv`: weight = BREG, product = MREG,
   accumulator = PREG. `product <= activation_in × weight` (1 cycle),
   `accumulator <= accumulator + product` in OS / `psum_in + product` in WS
   (1 cycle). So OS has a 2-cycle product→accumulate latency, pipelined.
3. **Result drain.** OS reads one row per 2 cycles (`cnn_accelerator_v2.sv:575`,
   `result_req` held 2 cycles/row); WS reads row 7 in 1 cycle.

## 2. OS FC schedule (derived + verified bit-exact)

Rows = 8 output neurons, 1 pixel (column 0), IC features serialized.

| Phase | Cycles |
|---|---:|
| accum_clear + first weight load | 1 |
| feed x[ic] (IC features, weight pipelined) | IC |
| tail (last product → accumulator) | 1 |
| drain (2 cycles × active rows) | 2·rows |

**OS FC group = IC + 2·rows + 2**, where `rows = min(8, OC − 8·g)`.

The model's `17 + IC·9` incorrectly reused the 8-pixel pass (`K+8` = 8 pixels +
K taps). For FC (1 pixel), the pass is `P + K = 2`, not `9` — an **~8× overcount**.

## 3. WS FC schedule

Rows = 8 input-feature taps (tiled), 1 pixel, OC serialized (1 neuron/sweep).

**WS FC group = 8·ceil(IC/8) + 2 per neuron.** The conv formula
`8·ceil(T/8)+10` carries a `+10` whose dominant term is the 7-cycle **8-pixel**
lead-in; for FC (1 pixel) that term vanishes, leaving clear (1) + latch (1).

## 4. Worked examples (frozen L2 geometry)

**FC1 (IC=120, OC=84):**
- OS: 10 full groups (rows 8) + 1 partial (rows 4)
  = 10·(120+18) + (120+10) = **1,510 cycles** (bit-exact vs golden).
- WS: 84 sweeps × (8·15+2) = 84 × 122 = **10,248 cycles**.

**FC2 (IC=84, OC=10):**
- OS: 1 full group (rows 8) + 1 partial (rows 2) = (84+18) + (84+6) = **192 cycles**.
- WS: 10 sweeps × (8·11+2) = 10 × 90 = **900 cycles**.

## 5. Comparison vs mapping_model.py

| Layer | Model OS | Corrected OS | Model WS | Corrected WS |
|---|---:|---:|---:|---:|
| fc1 | 12,067 | **1,510** | 10,920 | 10,248 |
| fc2 | 1,546 | **192** | 980 | 900 |

The OS-FC overcount was the error (discrepancy A); the WS-FC model was within
~7% (its `+10` included the conv's pixel lead-in).

## 6. Resolution and correction

- **Discrepancy A is RESOLVED:** the OS FC formula reused the 8-pixel-group
  lead-in; the correct OS FC is `IC + 2·rows + 2` per group, **~8× smaller**.
- **Consequence for the crossover:** with the correction, **OS wins FC** by
  6.8× (fc1) and 4.7× (fc2). The Step 3.2/3.3 "WS wins FC" crossover was an
  **artifact of the overcounted OS FC**. **OS wins all five layers; there is no
  OS/WS crossover on LeNet-5.**
- **Required mapping-model correction:** replace the OS group formula
  `17 + IC·K·(K+8)` with `1 + IC·K·(P+K) + 2·rows`, where `P` is the pixels per
  group (8 for full conv groups, fewer for partial, 1 for FC) and `rows =
  min(8, OC − 8·g)`. This fixes FC (and partial conv groups) without changing the
  verified Conv1 anchor (82 cycles).

## 7. Simulation result

`python/fc_schedule_sim.py` models the exact PE+array register semantics and:
- **OS FC is bit-exact** against `layer_vectors.npz` (`fc1_acc`, `fc2_acc`).
- The WS driving was traced; its result lands at row 7 one cycle after the last
  skewed tap (the `+2` latch), confirming the WS formula's structure. (The full
  WS golden-equality check was not completed in this step — the WS cycle count is
  anchored analytically to the verified conv-WS schedule, not to a full WS
  bit-exact run.)

**Verdict:** PASS for the OS-FC schedule (bit-exact); the WS-FC count is derived
and consistent with the conv anchor. The required correction to the mapping
model is identified but **not applied** here (per Step 4.1 scope).
