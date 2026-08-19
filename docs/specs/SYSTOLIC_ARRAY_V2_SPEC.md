# Systolic Array V2 Specification — Reconfigurable Dataflow (Weight-Stationary)

> **Status:** IMPLEMENTED AND VERIFIED — Conv1 Weight-Stationary (WS) mapping
> finalized (Decision 11, `docs/PROJECT_STATE.md`, 2026-08-19) and the cycle
> schedule derived and verified against the `pe_v2.sv` register model (§9).
> The PE-v2 contract (`docs/specs/PE_SPEC.md` §13, Decision 10) is implemented
> and verified (`rtl/common/pe_v2.sv`, `sim/tb_pe_v2.sv`). The V2 array
> (`rtl/common/systolic_array_v2.sv`) is implemented and functionally verified:
> 335/335 checks PASS in `sim/tb_systolic_array_v2.sv`; 64 DSP48E2 / 0 CARRY8;
> 200 MHz post-route WNS +2.861 ns. §10 item 4 records the resolved DSP-mapping
> finding (the muxed WS addend maps to the C input, not PCIN/PCOUT).

---

## 1. Purpose and Scope

This document defines the **V2 reconfigurable-dataflow systolic array** in its
**Weight-Stationary (WS) mode** — the mode that uses the PE-v2 partial-sum
cascade (Decision 10). It is the V2 counterpart to
`docs/specs/SYSTOLIC_ARRAY_SPEC.md` (V1 baseline).

It specifies:

- The V2 WS dataflow (true spatial Weight-Stationary).
- The resolved **Conv1 WS mapping** (Decision 11): PE rows = tap/reduction
  dimension, PE columns = output pixels, output channels serialized.
- The **4-tile tap decomposition** (8 + 8 + 8 + 1) and the **bottom-to-top tile
  feedback** mechanism.
- Why the earlier "6 output channels on PE rows" idea was **withdrawn**.
- The compatibility argument showing the feedback mechanism is realizable with
  the **already-verified** `rtl/common/pe_v2.sv`.
- The **verified WS cycle schedule** (42 cycles/group; tile, weight, activation,
  feedback and result-capture timing) — §9.

It does **not**:

- Modify `pe_v2.sv`, `pe.sv`, `systolic_array.sv`, or any V1 testbench (V1 is
  frozen — Decision 10).
- Define the top-level FSM, BRAM weight controller, input-feed internals, or
  any AXI interface. (The array-level cycle schedule is specified in §9; only
  the controller/input-feed *internals* that realize it remain future work.)

---

## 2. Source and Specification Status

### 2.1 Documentation Hierarchy

| Tier | Source | Authority |
|------|--------|-----------|
| 1 | Explicitly agreed project requirements | Binding |
| 2 | Detailed specifications (this doc, `PE_SPEC.md` §13) | Binding within scope |
| 3 | External technical research (DS925, UG579, dataflow taxonomy) | Informative |
| 4 | Engineering recommendations | Advisory |
| 5 | Unresolved assumptions / open decisions | No authority |

### 2.2 Label Convention

| Label | Meaning |
|-------|---------|
| **REQUIREMENT** | Established by the roadmap or an agreed decision |
| **DECISION** | Resolved by team agreement (recorded in `docs/PROJECT_STATE.md`) |
| **RESEARCH FACT** | Established by external technical reference |
| **CANDIDATE** | Proposed design choice — not yet agreed |
| **IMPLIED** | Direct logical consequence of a requirement/decision |
| **OPEN DECISION** | Unresolved — must be decided before RTL |

### 2.3 Primary Sources

- `docs/specs/PE_SPEC.md` §13 — PE-v2 contract (Decision 10).
- `rtl/common/pe_v2.sv` — the implemented, verified PE-v2.
- `sim/tb_pe_v2.sv` — the PE-v2 unit testbench (chain tests T4–T7).
- `docs/specs/SYSTOLIC_ARRAY_SPEC.md` — the V1 array contract (topology, shift
  chain, reverse column index, result protocol).
- `docs/PROJECT_STATE.md` Decisions 9, 10, 11.

---

## 3. V2 WS Dataflow (True Spatial Weight-Stationary)

**DECISION (Decision 10)** — V2's reconfigurable-dataflow WS mode is **true
spatial Weight-Stationary**:

| Parameter | Value |
|-----------|-------|
| Weights | held stationary in each PE (loaded once per tile) |
| Activations | shift horizontally across columns (per-row shift chain) |
| Partial sums | cascade vertically down each column via `psum_in`/`psum_out` |
| Cross-tile reduction | bottom-to-top feedback (row 7 → row 0) |

**Why "true spatial"** — the roadmap §3.2 describes V2 "Mode 0 Weight
Stationary" as "weights held, activations shift" with **no partial sums**, which
taken literally computes `w × Σ activations` — not a convolution. True spatial
WS requires the vertical partial-sum cascade that Decision 7 deferred and that
PE-v2 now supplies (`psum_in`/`psum_out`/`dataflow_mode`). This is the same
class of error withdrawn in Decision 9 / `SYSTOLIC_ARRAY_SPEC.md` §20.4, but in
the vertical (reduction) direction.

**PE-v2 accumulator semantics (WS):**

```systemverilog
assign psum_out = accumulator;                       // pe_v2.sv:63
always_ff @(posedge clk) begin
    if (rst)              accumulator <= '0;
    else if (accum_clear) accumulator <= '0;
    else if (dataflow_mode) accumulator <= psum_in + product;  // WS: pe_v2.sv:109
    else                   accumulator <= accumulator + product; // OS
end
```

In WS mode the accumulator register is reused as a **partial-sum pass-through +
add**: each PE adds its own `product` to the incoming `psum_in` and forwards the
result downward. The accumulator does **not** retain its own previous value in
WS mode — the running partial sum lives in the *row below*, and the *completed*
sum lives at the bottom row.

---

## 4. Conv1 WS Mapping (Resolved)

### 4.1 The Withdrawn Idea — "6 Output Channels on PE Rows"

**WITHDRAWN (2026-08-19, Decision 11)** — An earlier proposal mapped the **6
output channels onto PE rows 0–5**. This is **incorrect** for WS mode, for the
same reason "weights held, activations shift" without partial sums was wrong:

- The PE-v2 vertical `psum_in → psum_out` cascade **sums rows**. Row `r`'s
  `psum_out` carries the sum of rows `0..r`.
- If rows were output channels, then `psum_out[5]` would be
  `Σ_{ch=0..5} output[ch]` — the **sum of six different output channels**, not
  six separate outputs.
- A convolution needs `6` *independent* results, one per channel. Mixing them in
  one accumulator is not a convolution.

Therefore, in WS mode **rows must represent the reduction/tap dimension**, so
the vertical cascade computes exactly `Σ_k w[k] · a[k]` (a dot product).

### 4.2 Resolved Mapping (Decision 11)

**DECISION** — The Conv1 WS mapping is:

| Axis | V2 WS meaning | V1 (for contrast) |
|------|---------------|-------------------|
| **PE row `r` (0–7)** | kernel tap / reduction index | output channel |
| **PE column `c` (0–7)** | 8 adjacent output pixels (reverse index, `base − c`) | 8 adjacent output pixels (same reverse index) |
| **Time (serialized)** | output channels — one channel per sweep | kernel taps (5 passes × 5 taps) |

Concretely:

- **Rows** = the **25 Conv1 taps** tiled 8-per-tile. Row `r` of tile `t` holds
  tap `(8t + r)`'s weight and receives that tap's input pixels.
- **Columns** = **8 output pixels**, using the **verified V1 reverse-index
  mapping** (`SYSTOLIC_ARRAY_SPEC.md` §10.4): column `c` produces pixel
  `base − c`.
- **Output channels are serialized**: one channel per sweep. Conv1's 6 channels
  require 6 sweeps per pixel group, so the array processes `6 sweeps × 8 pixels
  = 48` results per group — identical count to V1's output group (§15 of the V1
  spec), but achieved by transposing channels from space to time.

### 4.3 Rationale

1. **Correctness** — the psum cascade reduces the tap dimension, which is the
   only dimension whose terms must be *summed* per output. Columns (pixels) and
   sweeps (channels) are dimensions whose terms must be kept *separate*, so they
   live on the shift chain and on time, respectively.
2. **Reuses the verified reverse-index mapping** — the column → pixel reverse
   index is a property of the horizontal shift chain, which is unchanged from
   V1 and already verified (379/379).
3. **Preserves the V1 output-group shape** — 6 channels × 8 pixels = 48 results
   per group, so the top-level sequencing and golden-model structure carry over.
4. **Natural tile size** — the vertical cascade is 8 rows deep, so an 8-tap tile
   uses the full column depth; 25 = 8 + 8 + 8 + 1 yields 4 tiles.

### 4.4 V1 vs V2 WS — Transposition Summary

| Dimension | V1 (Output-Stationary) | V2 WS |
|-----------|------------------------|-------|
| Rows | output channels (6 active, 2 idle) | taps (8 per tile) |
| Columns | 8 output pixels (reverse index) | 8 output pixels (reverse index) |
| Time | taps (5×5, 5 passes) | output channels (6 sweeps) |
| Weight movement | broadcast per row, reloaded every tap | held per row, reloaded per tile |
| Activation stream | one shared stream (all rows identical) | per-row streams (one per tap) |
| Partial sums | none (local accumulator) | vertical cascade + tile feedback |

---

## 5. Four-Tile Tap Decomposition (8 + 8 + 8 + 1)

**DECISION** — The 25 Conv1 taps are processed as **4 WS tiles**:

| Tile | Tap indices | Rows used |
|------|-------------|-----------|
| 0 | 0–7 | rows 0–7 |
| 1 | 8–15 | rows 0–7 |
| 2 | 16–23 | rows 0–7 |
| 3 | 24 | row 0 only (rows 1–7 pass through `+0`) |

The tap index is a flat serialization of the 5×5 kernel (order is a controller/
input-feed detail; any fixed order is correct for a sum). Tile 3 has a single
tap because `25 = 3 × 8 + 1`; its rows 1–7 contribute `0` (weights driven to
`0`, or `zero_skip`/idle-weight tie-off), so the partial sum simply flows down.

**IMPLIED** — 7 of the 32 row-slots across the 4 tiles are idle (rows 1–7 of
tile 3). This is inherent to the 8-row cascade depth and the decision to tile
by 8; it is a utilization note, not a correctness issue.

---

## 6. Partial-Sum Cascade and Bottom-to-Top Feedback

### 6.1 Vertical Cascade (within a tile)

**DECISION** — Within a tile, `PE(r,c).psum_in = PE(r−1,c).psum_out` (row 0's
`psum_in` is the tile feedback, §6.2). Each PE adds its tap product and passes
the sum down. After a tile's activations have propagated, **the bottom row
(row 7) holds the running partial sum** of all taps processed so far (all tiles
up to and including the current one).

### 6.2 Bottom-to-Top Feedback (across tiles)

**DECISION** — The bottom row's `psum_out` is fed back to the top row's
`psum_in`: `PE(0,c).psum_in = PE(7,c).psum_out`. This carries the running
partial sum into the next tile. The verified schedule (§9) keeps this feedback
**enabled permanently** — **no row-0 mux is required**, because `accum_clear`
(once per group) zeroes the ring and provides the initial `psum_in[0] = 0`. (A
row-0 mux selecting `0` vs `accum[7]` remains an equivalent alternative; it is
simply not needed.)

After the final tile, **row 7 holds the completed output** for that
channel/pixel group.

**Sequencing discipline (mirrors V1 §13)** — `accum_clear` is asserted **once
per group** (cycle 0), **never between tiles** of the same group, because the
running partial sum must survive across tiles. Between tiles only the per-row
weights (reloaded at cycles 15/23/31) change; the accumulator ring is left
running.

### 6.3 Compatibility with `pe_v2.sv` (verification)

**VERIFIED** — the bottom-to-top feedback is realizable with the already-verified
`rtl/common/pe_v2.sv`, with **no PE change**:

1. **Ports exist.** `psum_in[31:0]` (input) and `psum_out[31:0]` (output) are
   PE-v2 ports (`pe_v2.sv:34,38`). The feedback is array-level wiring + a mux,
   not a PE modification.

2. **No combinational loop.** The psum path has exactly **one register** — the
   accumulator (PREG): `psum_out = accumulator` (comb assign, `pe_v2.sv:63`) and
   `accumulator <= psum_in + product` (`pe_v2.sv:109`). Wired as a ring
   (`psum_in[r] = psum_out[r−1]`, with `psum_in[0] = psum_out[7]`), the loop
   `acc[7] → … → acc[0] → … → acc[7]` passes through **8 accumulator registers**
   (one per row), with a single `psum_in + product` adder between each. This is a
   proper systolic ring, structurally identical to the DSP48E2 `PCIN → PCOUT`
   column cascade. There is no register-free cycle.

3. **Correct accumulation.** The WS recurrence (derived from `pe_v2.sv`'s MREG +
   PREG pipeline) is
   `acc[r](n) = acc[r−1](n−1) + a_r(n−2) · w_r`, with `acc[−1] ≡ acc[7]`
   (feedback). At steady state `acc[7] = Σ_{all taps so far} w_k · a_k`, which is
   exactly the running Conv1 dot product. The chain test `T7`
   (`tb_pe_v2.sv`) verifies this recurrence for an 8-stage cascade with signed
   operands.

4. **Weight reload between tiles is supported.** `weight_load` updates the
   weight register in **both** modes (`pe_v2.sv:70–75`, gated only by `rst`),
   with the standard 1-cycle BREG latency. Reloading 8 tap weights between tiles
   is therefore a controller-schedule concern, not a PE limitation.

5. **`accum_clear` resets the ring at a sweep boundary.** `accum_clear` clears
   `accumulator` **and** `product` in both modes (`pe_v2.sv:84–91, 104–107`),
   independent of `dataflow_mode`. Asserting it once per sweep (not between
   tiles) starts each output channel's reduction from zero cleanly.

**Conclusion:** the 4-tile bottom-to-top feedback is *architecturally* compatible
with `pe_v2.sv`, and the *cycle-level schedule* that aligns each row's per-tap
activation stream with the psum arrival is specified and verified in §9. The
`rtl/common/systolic_array_v2.sv` interconnect realizes this (§10); the
controller/input-feed internals remain future work.

**Note (synthesis, 2026-08-19):** "structurally identical to the PCIN → PCOUT
cascade" in point 2 above is a *topological* analogy — one accumulator register
per stage around the loop, no combinational cycle. The actual DSP48E2 mapping is
resolved in §10 item 4: Vivado ML 2023.1 implements the muxed `psum_in + product`
addend through the DSP **C input** (OPMODE "C or P"), **not** the PCIN/PCOUT
cascade, because the `dataflow_mode` mux prevents PCIN/PCOUT inference. PCIN/PCOUT
is an implementation detail, not a V2 functional requirement; the dataflow, ring
topology, and functional behaviour are unchanged.

---

## 7. Activation and Weight Delivery

**DECISION / IMPLIED** — WS mode changes how the existing per-row ports are
used, but does **not** change the array interface shape:

- **Activations are per-row distinct.** Row `r` = tap `(k_y, k_x)`, so row `r`
  needs input pixels from input row `y + k_y`, horizontally offset by `k_x`. The
  V1 array already exposes `act_in[0..7]` (one per row) and a per-row shift
  chain (`systolic_array.sv:20,32–55`), so per-row activation streams are
  supported. The **reverse index** (§4.2) is a per-row property of the shift
  chain and is unchanged. The **diagonal skew** between rows (one cycle per row)
  is specified in §9.5.
- **Weights are per-row, held per tile.** `w_in[r]` broadcasts the tile's tap-`r`
  weight to all 8 columns of row `r` (weight is pixel-invariant). `weight_load`
  is asserted once per tile (per sweep) to load the 8 tap weights, unlike V1's
  every-tap reload.

**IMPLIED** — This is the transpose of V1: in V1 the activation stream was
*shared* (one input feature map read by every channel) and the weight was
*per-row per-tap*; in WS the weight is *per-row stationary* and the activation
is *per-row distinct*. Both fit the existing 8 `act_in` / 8 `w_in` port set.

---

## 8. Result Collection

**OPEN DECISION (partially narrowed)** — The final result lives at **row 7**
(bottom row), one completed output per column (8 pixels for one channel).

This is a **row-parallel read** (8 columns of row 7 simultaneously), which is
**transposed relative to V1's column-sequential drain** (`result_req[c]` selects
one column, reading 8 rows). `systolic_array.sv`'s current result mux is
column-oriented (`systolic_array.sv:104–112`), so the V2 array needs either:

- a per-row `result_req` (assert row 7, capture its 8 columns), or
- a dedicated bottom-row read bus.

This remains the mechanism-level open item (`PE_SPEC.md` §13.6 item 3); the
mapping has only fixed **where** the result is (row 7), not **how** it is read.
The **timing** is now fixed by §9.8: `result_request` on row 7 at cycle 41,
`result_out` captured at cycle 42.

---

## 9. Cycle-Level Schedule (Verified)

> **Status:** RESOLVED — the schedule below is derived from the actual
> `rtl/common/pe_v2.sv` register behaviour (weight = BREG, product = MREG,
> accumulator = PREG, `result_out` = fabric register) and verified numerically
> against a golden 5×5 convolution (multiple input/kernel seeds). It is **not**
> assumed from the architecture document.

### 9.1 Conventions

- Cycle index `t` is 0-based; a signal "at cycle `t`" is stable during that
  clock period, and registers update on the posedge *ending* cycle `t` (into
  `t+1`).
- `base` is the rightmost output-pixel index of the current 8-pixel group
  (column `c` → pixel `base − c`, the V1 reverse index). For the first 24×24
  group, `base = 7`.
- Tap flattening is **row-major** `k = 5·k_y + k_x` (the working convention used
  here; the exact order is still an open decision — §10).
- `dataflow_mode = 1` (WS) and `zero_skip = 0` throughout.

### 9.2 Tile / Tap Assignment

| Tile | Taps (`k`) | Rows used |
|------|-----------|-----------|
| 0 | 0–7 | 0–7 |
| 1 | 8–15 | 0–7 |
| 2 | 16–23 | 0–7 |
| 3 | 24 | 0 only (rows 1–7 weight = 0, pass through `+0`) |

### 9.3 Control / Weight Timeline

| Cycle | accum_clear | weight_load | w_in[r] (all rows) |
|------:|:-----------:|:-----------:|--------------------|
| 0 | **1** | **1** | tile 0 taps (0–7) |
| 15 | 0 | **1** | tile 1 taps (8–15) |
| 23 | 0 | **1** | tile 2 taps (16–23) |
| 31 | 0 | **1** | tile 3 tap 24 (rows 1–7 = 0) |
| 41 | 0 | 0 | — (result_request row 7) |

- `accum_clear` is asserted **once per group** (cycle 0), concurrent with the
  tile-0 `weight_load` (independent registers; `PE_SPEC.md` §5.7 priority allows
  both). It zeroes the ring (all `accum` and `product`), which is what makes the
  always-on feedback start from 0. It is **never** asserted between tiles.
- A **single array-wide `weight_load`** is sufficient (verified): the 1-cycle
  BREG latency makes the new weights live one cycle later. Row 7's last
  tile-`t` product samples the old weight at cycle `8t+15`, while row 0's first
  tile-`t+1` product samples the new weight at cycle `8t+16`, so no per-row
  weight-load strobe is required.

### 9.4 Feedback (partial-sum ring)

`psum_in[0][c] = accum[7][c]` is **always enabled** (no row-0 mux). The ring is
a closed 8-stage systolic loop (8 accumulator registers); the initial zero comes
from `accum_clear` at cycle 0.

### 9.5 Activation Timing (per-row diagonal skew)

Tile `t` streams starting at `a_t = 8t + 1`. Row `r` (tap `8t+r = (k_y,k_x)`)
drives, for `s ∈ [a_t + r, a_t + r + 8)`:

```
act_in[r](s) = input[y + k_y][ base − 7 + (s − a_t) − r + k_x ]   (0 if column ∉ [0,27])
```

The three offsets are: `base − 7` = **7-cycle horizontal lead-in** (fills the
shift chain so all 8 columns align); `−r` = **per-row diagonal skew** (row `r`'s
stream delayed one cycle per row, so its product meets the descending partial
sum); `+k_x` = the tap's horizontal kernel offset.

**REQUIREMENT (V2-specific)** — V2 WS requires **eight distinct per-row
activation streams**, each carrying its tap's `(k_y,k_x)` offset and the diagonal
skew. This is an **array/input-feed requirement**, and it is **different from
V1**, which drove all 8 `act_in` lines with a *single shared* stream
(`SYSTOLIC_ARRAY_SPEC.md` §7). The array interface is unchanged (the 8 `act_in`
inputs already exist); only the *content* — one skewed stream per tap — differs.

The skew collapses for consecutive taps on the same input row: in tile 0, rows
0–4 (`k_y=0`, `k_x=0..4`) all read `input[y][·]` with `k_x − r = 0`, so they
share one stream; rows 5–7 (`k_y=1`) read `input[y+1][·]` shifted by −5.

### 9.6 Pipeline Latency

| Path | Latency (cycles) |
|------|------------------|
| `activation_in → product` (MREG) | 1 |
| `product → accumulator` (PREG) | 1 |
| `accumulator[r] → accumulator[r+1]` (vertical cascade) | 1 per row |
| activation at row `r` → `accum[7]` | `2 + (7 − r)` |

A tile's wavefront for row `r` lands at cycle `8t + 8 + r`; the tile's partial
sum completes at `accum[7]` at cycle `8t + 17`.

### 9.7 Result Timeline

| accum[7][c] = … | during cycle |
|-----------------|:------------:|
| tile-0 partial | 17 |
| tile-0 + tile-1 | 25 |
| tile-0 + tile-1 + tile-2 | 33 |
| **full 25-tap result (golden)** | **41** |

- Tile-0 partials become tile-1 `psum_in` **combinational** at cycle 17:
  `psum_in[0][c](17) = accum[7][c](17)`; at edge 18 row 0 adds tile 1's product.
  The same carry happens at cycles 25 (→ tile 2) and 33 (→ tile 3).
- The single-tap tile 3 (tap 24) occupies only row 0; rows 1–7 pass `psum_in`
  through unchanged (`product = 0`). The tap-24 contribution still traverses the
  full 8-row cascade, so the result still lands at `accum[7]` cycle 41 (a 7-cycle
  tail; rows 1–7 idle this tile).

### 9.8 Result Capture

`result_request` is asserted on **row 7** at **cycle 41** (a single-cycle strobe
— it is level-sensitive and would re-capture garbage at cycle 42). `result_out`
is latched at edge 42 and held (read-without-clear). Result collection reads
**row 7** (8 columns, one per output pixel) — see §8.

### 9.9 Totals

- **One group** (8 pixels, 1 channel, 25 taps): **42 cycles** (cycles 0–41,
  result latched at edge 42).
- **One Conv1 channel sweep** = 24×24 = 576 pixels = **72 groups →
  72 × 42 = 3024 cycles**.
- **Full Conv1** = 6 serialized channels × 72 groups = 432 groups → 432 × 42 =
  **18,144 cycles** (non-pipelined baseline).

### 9.10 PE-v2 Interface Sufficiency

**DECISION** — the `pe_v2.sv` interface is **sufficient**; **no PE modification
is required**. Every signal used exists: per-row `activation_in` (+ array shift
chain), `weight_in`/`weight_load` (a single array-wide load is verified to
suffice), `psum_in`/`psum_out` (always-on feedback), `accum_clear` (ring reset),
`dataflow_mode = 1`, `zero_skip = 0`, and per-PE `result_request` (bottom-row
read). The only array-level consequence is that `result_request` must be driven
**per row** (row 7), not per column as in V1 — that is wiring, not a PE change.

---

## 10. Open Decisions

| # | Item | Status |
|---|------|--------|
| 1 | WS result-collection mechanism (per-row `result_req` vs bottom-row bus) | **OPEN DECISION** — result location fixed to row 7, captured at cycle 41 (§8, §9.8) |
| 2 | WS weight-loading mechanism (BRAM/controller sequencing to hit cycles 0/15/23/31) | **OPEN DECISION** — timing fixed by §9.3; BRAM/controller realization open |
| 3 | Reconfiguration flush sequence (full `rst` vs lighter flush) | **OPEN DECISION** (`PE_SPEC.md` §13.6) |
| 4 | DSP48E2 inference of the mode mux / PCIN–PCOUT cascade incl. feedback | **RESOLVED (2026-08-19)** — Vivado ML 2023.1 implements the muxed WS addend through the DSP **C input** (OPMODE "C or P"), not PCIN/PCOUT. PCIN/PCOUT is an implementation detail, not a V2 functional requirement; no PE redesign required |
| 5 | Tap flattening order (row-major vs column-major) for the 25-tap stream | **OPEN DECISION** — §9.1 uses row-major `k = 5·k_y + k_x` as a working convention; weight ROM and input-feed must agree |
| 6 | Group-boundary pipelining (reduce the 42-cycle/group baseline) | **NOT ADOPTED** — optimization only; the correctness-first baseline is §9.9 |

**Resolved by this revision:** the exact WS cycle schedule (previously §10 item 1)
is now specified and verified in §9.

---

## Appendix A: References

| Reference | Description |
|-----------|-------------|
| `docs/specs/PE_SPEC.md` §13 | PE-v2 contract (Decision 10) |
| `rtl/common/pe_v2.sv` | Implemented, verified PE-v2 (WS/OS modes) |
| `sim/tb_pe_v2.sv` | PE-v2 unit testbench (T4–T7 chain tests) |
| `docs/specs/SYSTOLIC_ARRAY_SPEC.md` | V1 array contract (§4 topology, §6 shift chain, §10.4 reverse index) |
| `docs/PROJECT_STATE.md` | Decisions 9, 10, 11 |
| DS925 / UG579 | XCK26 / DSP48E2 references |

---

> **Status:** `systolic_array_v2.sv` is implemented and functionally verified
> (335/335 in `sim/tb_systolic_array_v2.sv`; 64 DSP48E2 / 0 CARRY8; 200 MHz
> post-route WNS +2.861 ns). The controller/input-feed internals that realize the
> §9 schedule remain future work. `pe_v2.sv` is **not** modified.
