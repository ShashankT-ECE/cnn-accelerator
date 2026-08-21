# Input-Feed / Controller V2 Specification — Weight-Stationary (WS) Schedule

> **Status:** SPECIFICATION — the controller/input-feed is derived directly from
> the verified 42-cycle/group Weight-Stationary schedule
> (`docs/specs/SYSTOLIC_ARRAY_V2_SPEC.md` §9) and the frozen interfaces of
> `rtl/common/pe_v2.sv` / `rtl/common/systolic_array_v2.sv`. It records the four
> controller decisions (Decision 13, `docs/PROJECT_STATE.md`, 2026-08-20) and the
> derived activation/weight addressing. No RTL or testbench is defined here;
> physical memory technology (BRAM/URAM/distributed RAM) is deliberately kept
> separate from these architectural requirements.

---

## 1. Purpose and Scope

This document defines the controller / input-feed unit that drives
`systolic_array_v2` to realize the verified Conv1 Weight-Stationary schedule. It
specifies the controller state, the cycle-by-cycle control and activation/weight
schedule, the tile/group/channel transitions, result capture, reset behaviour,
and the derived data-span requirements.

It is the V2 counterpart to `docs/specs/INPUT_FEED_SPEC.md` (V1 input-feed),
but is a **new** module — the V1 input-feed (`rtl/common/input_feed.sv`) is
**frozen** (Decision 10) and is not reused or modified.

It does **not**:

- Modify `pe_v2.sv`, `systolic_array_v2.sv`, `pe.sv`, `systolic_array.sv`, or
  `input_feed.sv` (V1 and V2 datapath RTL are frozen).
- Prescribe a physical memory primitive (BRAM/URAM/distributed RAM) for the
  activation line buffer or weight store — that is an implementation detail.
- Define the PS/ARM host interface (AXI/DMA register map) that loads the image
  and weights — that is deferred to integration.

## 2. Labels and Conventions

| Label | Meaning |
|-------|---------|
| **REQUIREMENT** | Established by the roadmap or an agreed decision |
| **DECISION** | Resolved by team agreement (recorded in `docs/PROJECT_STATE.md`) |
| **DERIVED** | Computed directly from the verified schedule / frozen interfaces |
| **IMPLIED** | Direct logical consequence of a requirement/decision |
| **CANDIDATE** | Proposed design choice — not yet agreed |
| **OPEN** | Unresolved — must be decided before RTL |

Conventions (from `SYSTOLIC_ARRAY_V2_SPEC.md` §9.1):

- Cycle index `s` is 0-based within a group; a signal "at cycle `s`" is stable
  during that clock period, and registers update on the posedge *ending* cycle
  `s` (into `s+1`).
- `I[y][x]` — the 28×28 input activation image, signed 8-bit (Q8, scale 2⁻⁸),
  indices `0..27`.
- `K[ch][ky][kx]` — Conv1 weights, 6 output channels × 5×5 kernel, signed 8-bit,
  `ky,kx ∈ 0..4`.
- `O[ch][y][x] = Σ_{ky,kx} I[y+ky][x+kx] · K[ch][ky][kx]` — the 24×24 valid
  convolution (no padding), `y,x ∈ 0..23`.
- Tap flattening is **row-major** `k = 5·k_y + k_x` (Decision 13 item 4).

## 3. Reference Schedule (Verified)

The controller realizes, without change, the verified schedule
(`SYSTOLIC_ARRAY_V2_SPEC.md` §9):

- 8×8 array; PE rows 0–7 = kernel **tap** dimension; PE columns 0–7 = 8 output
  pixels (reverse index: column `c` → pixel `base − c`).
- 25 taps = **4 tiles** of `8 + 8 + 8 + 1` (§9.2).
- **42 cycles/group**; `accum_clear` once at cycle 0; `weight_load` at cycles
  0/15/23/31; `result_req[7]` at cycle 41; `result_out` captured at cycle 42
  (§9.3, §9.8).
- Vertical WS psum reduction + always-on bottom-to-top tile feedback
  (`psum_in[0][c] = psum_out[7][c]`) — §9.4.
- `dataflow_mode = 1`, `zero_skip = 0` throughout (§9.1).

## 4. Controller State and Counters

**DERIVED** — the minimal controller state that generates the schedule is three
counters plus derived values:

| Register | Range | Advances |
|----------|-------|----------|
| `ch` — channel | `0..5` | after 72 groups (one full 24×24 sweep) |
| `g` — group | `0..71` | after 42 cycles |
| `s` — cycle-in-group | `0..41` | every cycle (mod 42) |

Derived from `g`:

| Value | Derivation |
|-------|-----------|
| output row `y` | `y = g / 3` (integer div) — `0..23` |
| group-in-row `b` | `b = g % 3` — `0..2` |
| `base` (rightmost output pixel) | `base = 7 + 8·b` — `7, 15, 23` |

**IMPLIED** — 24 output columns = 3 groups of 8, so `base ∈ {7, 15, 23}`. Group
`g` produces output pixels `x = base − c` for `c = 0..7`, i.e. the 8 pixels
`[base−7, base]`.

All control and address signals are combinational functions of `ch`, `g` (via
`y`, `base`), and `s`; no per-PE state is required. The controller does **not**
need to track the psum ring — the array's always-on feedback and the single
cycle-0 `accum_clear` maintain it (§3, `SYSTOLIC_ARRAY_V2_SPEC.md` §9.4).

## 5. Cycle-by-Cycle Control Schedule

**DERIVED** from `SYSTOLIC_ARRAY_V2_SPEC.md` §9.3/§9.8. For every group (all
channels, all `y`/`base`), the 42-cycle control timeline is fixed:

| Cycle `s` | `accum_clear` | `weight_load` | tile loaded | `result_req[7]` |
|----------:|:-------------:|:-------------:|:-----------:|:---------------:|
| 0 | **1** | **1** | 0 | 0 |
| 1..14 | 0 | 0 | — | 0 |
| 15 | 0 | **1** | 1 | 0 |
| 16..22 | 0 | 0 | — | 0 |
| 23 | 0 | **1** | 2 | 0 |
| 24..30 | 0 | 0 | — | 0 |
| 31 | 0 | **1** | 3 | 0 |
| 32..40 | 0 | 0 | — | 0 |
| 41 | 0 | 0 | — | **1** |

Equivalently, as combinational decode:

```systemverilog
accum_clear = (s == 0);
weight_load = (s == 0) | (s == 15) | (s == 23) | (s == 31);
result_req  = (s == 41) ? 8'b1000_0000 : 8'b0;   // row 7 only
tile        = (s == 0) ? 0 : (s == 15) ? 1 : (s == 23) ? 2 : 3;  // load source
zero_skip   = 0;
dataflow_mode = 1;                               // WS
```

**REQUIREMENT** — `result_req[7]` must be a **single-cycle strobe** at cycle 41
only. It is level-sensitive (read-without-clear); holding it into cycle 42 would
re-capture garbage (`SYSTOLIC_ARRAY_V2_SPEC.md` §9.8).

**REQUIREMENT** — `accum_clear` is asserted **only** at cycle 0 (concurrent with
the tile-0 `weight_load`; the PE priority `PE_SPEC.md` §5.7 permits both). It is
**never** asserted between tiles (Decision 13 item 3) — doing so would corrupt
the running partial sum carried by the feedback ring.

## 6. Weight Addressing and Loading

**DECISION** (Decision 13 item 2) — reuse `w_in[0:7]` + the single array-wide
`weight_load`; no per-row strobe; per-tap weights from a small deterministic
weight store.

**DERIVED** — at `weight_load` cycle `s` the controller presents tile `t`'s
8 tap weights on `w_in[0:7]`, where tap `k = 8·t + r` maps to `(ky, kx)` by
row-major `k = 5·ky + kx`:

| Tile `t` | `w_in[r]` for `r = 0..7` |
|:--------:|--------------------------|
| 0 | `K[ch][0][0..4]`, `K[ch][1][0..2]` (taps 0–7) |
| 1 | `K[ch][1][3..4]`, `K[ch][2][0..4]`, `K[ch][3][0]` (taps 8–15) |
| 2 | `K[ch][3][1..4]`, `K[ch][4][0..3]` (taps 16–23) |
| 3 | `w_in[0] = K[ch][4][4]`; `w_in[1..7] = 0` (tap 24 + pass-through) |

Weight address is therefore `(ch, ky, kx)` (or flat `(ch, k)` with `k = 8t+r`).
The store holds **150 weights** total (`6 ch × 25`), 25 live per channel. This
fits a LUTROM/register file — no BRAM/URAM is required (Decision 13 item 2).

**REQUIREMENT** — `w_in[r]` must be **stable during the `weight_load` cycle**
(cycle `s`), so the BREG stage captures it at edge `s→s+1` and it is live at
cycle `s+1`. The single-strobe timing is sufficient because the 1-cycle BREG
latency makes row 7's last tile-`t` product sample the old weight at `8t+15`
while row 0's first tile-`t+1` product samples the new weight at `8t+16`
(`SYSTOLIC_ARRAY_V2_SPEC.md` §9.3).

## 7. Activation Addressing and Stream Generation

**REQUIREMENT** (`SYSTOLIC_ARRAY_V2_SPEC.md` §9.5) — V2 WS requires **eight
distinct per-row activation streams**, each carrying its tap's `(ky,kx)` offset
and the diagonal skew. The array interface (8 `act_in` lines) is unchanged; only
the *content* differs from V1.

### 7.1 General per-row formula (authoritative)

**DERIVED** from `SYSTOLIC_ARRAY_V2_SPEC.md` §9.5 and verified by
`sim/tb_systolic_array_v2.sv` (`act_value`). For row `r` of tile `t`
(`a_t = 8t+1`), active for `s ∈ [a_t + r, a_t + r + 7]`:

```
act_in[r](s) = I[ y + ky ][ base − 7 + (s − a_t) − r + kx ]   (0 if column ∉ [0,27])
               where k = 8t + r,  ky = k / 5,  kx = k % 5
```

Row `r` is idle (drives 0) outside its 8-cycle window, and tile-3 rows 1–7 are
driven 0 (pass-through `+0`).

### 7.2 Collapsed stream form (DERIVED simplification)

Within tile `t`, for all rows `r` whose tap `k = 8t + r` shares a kernel row
`ky`, the identity `kx − r = 8t − 5·ky` makes the per-row `−r` skew cancel the
`+kx` offset. The activation value therefore depends only on `ky` and `s`:

```
act_in[r](s) = stream_ky(s) = I[ y + ky ][ base − 8 + s − 5·ky ]
```

where `ky = floor((8t + r) / 5)` for the row's active tile `t`.

**Consequence (DERIVED):** the input-feed's underlying data is **five continuous
streams** `stream_0 .. stream_4` (one per kernel row `ky = 0..4`), each a
left-to-right column sweep. The 8 array rows tap these 5 streams at their skewed
windows; the "8 distinct streams" required by §9.5 collapse to 5 data streams
routed to 8 `act_in` lines. For `base = 23`:

| `ky` | `stream_ky(s)` | array rows that use it |
|:----:|----------------|------------------------|
| 0 | `I[y+0][15+s]` | rows 0–4 (tile 0) |
| 1 | `I[y+1][10+s]` | rows 5–7 (tile 0), rows 0–1 (tile 1) |
| 2 | `I[y+2][ 5+s]` | rows 2–6 (tile 1) |
| 3 | `I[y+3][ s ]`   | row 7 (tile 1), rows 0–3 (tile 2) |
| 4 | `I[y+4][ s−5]`  | rows 4–7 (tile 2), row 0 (tile 3) |

**REQUIREMENT** — the controller must source these 5 streams with the correct
column sweep, and route each `act_in[r]` to the stream of its current tap's
`ky`, applying the per-row active-window (start/stop) gating. The column access
is **monotonic left-to-right** in `s` for every stream (favourable for a simple
streaming line buffer), but the physical buffer is not specified here (§12).

**REQUIREMENT** — out-of-bounds columns are driven 0. For the valid 24×24
mapping (`base ∈ {7,15,23}`) the column window `px = base − 8 + s − 5·ky` is
always `⊆ [0,27]`, so this guard is **defensive** for the valid mapping; it
becomes active only for non-valid `base` values or a padded mapping.

## 8. Tile Transitions and Feedback

**DERIVED** from `SYSTOLIC_ARRAY_V2_SPEC.md` §9.3–§9.7.

- **Tile boundaries are weight-load events, not control events.** The psum ring
  is never flushed between tiles; `accum_clear` is asserted only at cycle 0.
- The always-on bottom-to-top feedback `psum_in[0][c] = psum[7][c]` carries the
  running partial sum across tiles: tile-0 partial → tile-1 → tile-2 → tile-3.
- **Row-7 wavefront convergence:** every tap of a tile lands at `accum[7][c]` at
  the same cycle `8t + 17` (regardless of row `r`), because a deeper row's
  earlier sample time exactly offsets its longer cascade (§9.6). This is what the
  per-row diagonal skew in §7 aligns.

| Tile `t` | `a_t` | `accum[7][c]` completes at cycle |
|:--------:|:-----:|:--------------------------------:|
| 0 | 1 | 17 |
| 1 | 9 | 25 |
| 2 | 17 | 33 |
| 3 | 25 | 41 |

**IMPLIED** — the controller emits no signal at tile boundaries beyond
`weight_load`; the feedback is entirely inside the frozen array. The 7-cycle
tile-3 tail (rows 1–7 pass `+0`) means cycles 33–41 are drain-only; the
controller simply continues its schedule and asserts `result_req[7]` at 41.

## 9. Group and Channel Transitions

**DERIVED** — no group-boundary or channel-boundary pipelining (correctness-first
baseline; `SYSTOLIC_ARRAY_V2_SPEC.md` §10 item 6 **NOT ADOPTED**).

- **Group → group:** after cycle 41 (and latching `result_out` at edge 42), the
  controller resets `s` to 0 and advances `g`. The new group's cycle-0
  `accum_clear` flushes the ring. Groups are fully serial (42 cycles each).
- **Group addressing:** `g` increments through `y = 0..23`, and within each row
  through `b = 0..2` (`base = 7, 15, 23`). Equivalently `base` sweeps
  left→right across each row.
- **Channel → channel:** after 72 groups, `g` wraps and `ch` increments. The
  weights for the new channel are presented on the next `weight_load` cycles.
  **No reset is required** between channels — the cycle-0 `accum_clear` of the
  first group of the new channel flushes the ring (Decision 13 item 3).

## 10. Result Capture Interface

**DECISION** (Decision 13 item 1) — reuse the per-row `result_req`; assert
`result_req[7]` at cycle 41; capture `result_out[0:7]` at cycle 42. No dedicated
bottom-row bus.

**DERIVED:**

- At cycle 41, `accum[7][c]` holds the full 25-tap result for output pixel
  `base − c` (`c = 0..7`). The controller asserts `result_req[7] = 1` for that
  one cycle.
- At edge 41→42, the PEs' `result_out` registers latch the accumulator
  (read-without-clear), and the controller latches `result_out[0:7]`
  (**8 × 32 = 256 bits**, one 32-bit result per output pixel).
- **REQUIREMENT** — the controller must hold its own 256-bit result register:
  after `result_req[7]` deasserts, the array's combinational output mux drives
  `result_out = 0` (no row selected), so the array port does **not** retain the
  value. The captured 256-bit value is the group's result; hand-off to the output
  buffer/PS is deferred (§12).

## 11. Reset and Reconfiguration Behaviour

**DECISION** (Decision 13 item 3) — `accum_clear` at the group/sweep boundary is
the reconfiguration flush; full `rst` is the power-on reset only.

**DERIVED:**

- **Power-on / global reset:** the controller asserts `rst` once at start-up
  (synchronous, active-high; `PE_SPEC.md` §13.2/§5.7). This clears weight,
  product, accumulator, and result across all 64 PEs and the shift chain.
- **Normal reconfiguration (OS↔WS, or channel change):** **no `rst`.** The
  controller holds `dataflow_mode` static per layer (`PE_SPEC.md` §13.4) and
  asserts `accum_clear` at the boundary group's cycle 0; weights are re-loaded by
  the normal `weight_load` schedule, and results are re-captured by
  `result_req`. `accum_clear` clears exactly the state a mode switch corrupts
  (accumulator + product), and nothing else needs flushing.
- **Between tiles: never `accum_clear`** (§5).

## 12. Required Image and Weight Interfaces

**REQUIREMENT** — the controller/input-feed must be able to source, per group:

- **Activations:** the five streams `stream_ky(s) = I[y+ky][base−8+s−5·ky]` for
  `ky = 0..4`, i.e. read access to input rows `y..y+4` (5 rows) over the column
  window `[base−7, base+4]` (12 columns) — §13. Any source that can deliver these
  columns left-to-right per row satisfies the requirement.
- **Weights:** `K[ch][ky][kx]` for the current channel, 25 values, presented as
  §6 at the four tile-load cycles.

**Array-facing interface (DERIVED, fixed by `systolic_array_v2.sv`):** the
controller drives `act_in[0:7]`, `w_in[0:7]`, `weight_load`, `accum_clear`,
`zero_skip` (tie 0), `dataflow_mode` (tie 1), `result_req[0:7]`, and `rst`;
it reads `result_out[0:7]` and `clk`. No other array ports exist or are needed.

**OPEN (deferred to integration, not blocking this spec):** the physical
transport that delivers the 28×28 image and 150 weights to the controller from
the PS (AXI/DMA, pre-loaded line buffer, or streaming), and the result hand-off
to the output buffer/PS. These are protocol/SoC choices, not architectural
requirements of the schedule.

## 13. Derived Storage / Data-Span Requirements

**DERIVED** — the minimum *data availability* the schedule requires, independent
of physical memory technology:

- **Weight data:** 150 weights total (`6 ch × 25`), 8-bit; 25 live per channel.
  Addressable by `(ch, ky, kx)`. ≈ 1.2 kbit — far below any BRAM/URAM threshold
  (Decision 13 item 2).
- **Activation data-span per group:** 5 rows (`y..y+4`) × 12 columns
  (`base−7 .. base+4`) = **60 pixels** live per group. The row-major flattening
  keeps the live kernel-row count at ≤3 per tile (≤5 across the whole group).
- **Horizontal overlap:** adjacent groups in a row overlap 4 columns
  (`base_i + 4` vs `base_{i+1} − 7 = base_i + 1`), so a sliding 12-column window
  advances 8 per group.
- **Row reuse across output rows:** successive `y` share 4 of 5 input rows
  (`y+1..y+4` of group `y` = `y'..y'+3` of group `y+1`), so a 5-row line buffer
  needs only one new row per output-row step.

These are **data-availability bounds, not a prescribed buffer size or type.** The
choice of BRAM / URAM / distributed RAM / streaming is an implementation detail
and is explicitly not specified here.

## 14. Full Conv1 Schedule and Cycle Count

**DERIVED** (matches `SYSTOLIC_ARRAY_V2_SPEC.md` §9.9):

| Quantity | Value |
|----------|-------|
| One group | 42 cycles (cycles 0–41, result latched at 42) |
| Output pixels per group | 8 |
| Output per channel | 24×24 = 576 px = **72 groups** |
| Channel sweep | 72 × 42 = **3,024 cycles** |
| Full Conv1 (6 channels) | 6 × 72 = **432 groups** = **18,144 cycles** |
| Total MACs | 432 × 8 × 25 = **86,400** |
| Sustained throughput | 86,400 / 18,144 = **4.76 MACs/cycle** (vs 64 peak; ~7.4% util.) |
| At 200 MHz (5 ns) | **90.72 µs** |

**IMPLIED** — the low sustained utilization (4.76/64) is the known cost of the
correctness-first, non-pipelined baseline; group-boundary pipelining is an
explicitly non-adopted optimization (`SYSTOLIC_ARRAY_V2_SPEC.md` §10 item 6).

## 15. Worked Example — Right-Edge Group (base = 23)

The hardest group: `b = 2`, `base = 23`, producing output pixels `x = 16..23`.
Fix output row `y = 12` and one channel `ch`. Column `c` of the array produces
pixel `x = 23 − c`; the trace below follows **column 0 (pixel 23)**, whose taps
reach the image's right boundary `px = 27`.

### 15.1 Tap assignment (row-major, all groups)

| Row `r` | Tile 0 (k) | Tile 1 (k) | Tile 2 (k) | Tile 3 (k) |
|:-------:|:----------:|:----------:|:----------:|:----------:|
| 0 | 0 = (0,0) | 8 = (1,3) | 16 = (3,1) | 24 = (4,4) |
| 1 | 1 = (0,1) | 9 = (1,4) | 17 = (3,2) | — (0) |
| 2 | 2 = (0,2) | 10 = (2,0) | 18 = (3,3) | — (0) |
| 3 | 3 = (0,3) | 11 = (2,1) | 19 = (3,4) | — (0) |
| 4 | 4 = (0,4) | 12 = (2,2) | 20 = (4,0) | — (0) |
| 5 | 5 = (1,0) | 13 = (2,3) | 21 = (4,1) | — (0) |
| 6 | 6 = (1,1) | 14 = (2,4) | 22 = (4,2) | — (0) |
| 7 | 7 = (1,2) | 15 = (3,0) | 23 = (4,3) | — (0) |

(`(ky,kx)` shown; tile 3 has only tap 24 = (4,4) on row 0.)

### 15.2 The five streams (`base = 23`, so `base − 8 = 15`)

| `ky` | `stream_ky(s)` |
|:----:|----------------|
| 0 | `I[12][15+s]` |
| 1 | `I[13][10+s]` |
| 2 | `I[14][ 5+s]` |
| 3 | `I[15][ s ]` |
| 4 | `I[16][ s−5]` |

### 15.3 Per-cycle control and activation (`S_k ≡ stream_k(s)`)

| `s` | ctrl | `act_in[0..7]` |
|:---:|------|----------------|
| 0 | **AC,WL(0)** | 0 0 0 0 0 0 0 0 |
| 1 | | S0 · · · · · · · |
| 2 | | S0 S0 · · · · · · |
| 3 | | S0 S0 S0 · · · · · |
| 4 | | S0 S0 S0 S0 · · · · |
| 5 | | S0 S0 S0 S0 S0 · · · |
| 6 | | S0 S0 S0 S0 S0 S1 · · |
| 7 | | S0 S0 S0 S0 S0 S1 S1 · |
| 8 | | S0 S0 S0 S0 S0 S1 S1 S1 |
| 9 | | S1 S0 S0 S0 S0 S1 S1 S1 |
| 10 | | S1 S1 S0 S0 S0 S1 S1 S1 |
| 11 | | S1 S1 S2 S0 S0 S1 S1 S1 |
| 12 | | S1 S1 S2 S2 S0 S1 S1 S1 |
| 13 | | S1 S1 S2 S2 S2 S1 S1 S1 |
| 14 | | S1 S1 S2 S2 S2 S2 S1 S1 |
| 15 | **WL(1)** | S1 S1 S2 S2 S2 S2 S2 S1 |
| 16 | | S1 S1 S2 S2 S2 S2 S2 S3 |
| 17 | | S3 S1 S2 S2 S2 S2 S2 S3 |
| 18 | | S3 S3 S2 S2 S2 S2 S2 S3 |
| 19 | | S3 S3 S3 S2 S2 S2 S2 S3 |
| 20 | | S3 S3 S3 S3 S2 S2 S2 S3 |
| 21 | | S3 S3 S3 S3 S4 S2 S2 S3 |
| 22 | | S3 S3 S3 S3 S4 S4 S2 S3 |
| 23 | **WL(2)** | S3 S3 S3 S3 S4 S4 S4 S3 |
| 24 | | S3 S3 S3 S3 S4 S4 S4 S4 |
| 25 | | S4 S3 S3 S3 S4 S4 S4 S4 |
| 26 | | S4 · S3 S3 S4 S4 S4 S4 |
| 27 | | S4 · · S3 S4 S4 S4 S4 |
| 28 | | S4 · · · S4 S4 S4 S4 |
| 29 | | S4 · · · · S4 S4 S4 |
| 30 | | S4 · · · · · S4 S4 |
| 31 | **WL(3)** | S4 · · · · · · S4 |
| 32 | | S4 · · · · · · · |
| 33–41 | | 0 0 0 0 0 0 0 0 (tile-3 tail + drain) |
| 41 | **RQ(7)** | 0 0 0 0 0 0 0 0 |

(`·` = 0/idle; AC = accum_clear; WL(t) = weight_load tile t; RQ(7) = result_req[7].
Row `r`'s active window is visible as the diagonal band of `S_k` values; rows
1–7 of tile 3 are pass-through 0.)

### 15.4 Vertical cascade and tile feedback (column 0, pixel 23)

For column 0 the correct activation sample for every row of a tile is
`stream_ky` at `s = 8t + 8 + r` (the last active cycle), and each tile's partial
sum converges at `accum[7][0]` at cycle `8t + 17`:

- **Cycle 17 — tile-0 partial** `= Σ I[12][23+j]·K[0][j] (j=0..4) + I[13][23..25]·K[1][0..2]`
- **Cycle 25 — + tile-1** (via bottom-to-top feedback) `= + I[13][26..27]·K[1][3..4] + I[14][23..27]·K[2][0..4] + I[15][23]·K[3][0]`
- **Cycle 33 — + tile-2** `= + I[15][24..27]·K[3][1..4] + I[16][23..26]·K[4][0..3]`
- **Cycle 41 — + tile-3** `= + I[16][27]·K[4][4]`

The cycle-41 value is exactly `O[ch][12][23] = Σ_{ky,kx} I[12+ky][23+kx]·K[ky][kx]`
— the golden valid-convolution value, with the maximum column index `px = 27`
reached by taps `(0,4)` and `(4,4)`. `result_req[7] = 1` at cycle 41 latches
`result_out[0] = O[ch][12][23]` at edge 42; the other 7 columns simultaneously
latch `O[ch][12][16..22]` (one per column `c`, pixel `23 − c`).

## 16. Open Items / Out of Scope

| # | Item | Status |
|---|------|--------|
| 1 | Physical memory primitive for the activation line buffer and weight store (BRAM/URAM/distributed RAM) | **OPEN (deferred)** — implementation detail; architectural data-span is §13 |
| 2 | PS/ARM image + weight load transport and result hand-off (AXI/DMA register map) | **OPEN (deferred to integration)** |
| 3 | Group-boundary pipelining to raise the 4.76 MAC/cycle sustained rate | **NOT ADOPTED** (correctness-first baseline; `SYSTOLIC_ARRAY_V2_SPEC.md` §10 item 6) |

---

## Appendix A: References

| Reference | Description |
|-----------|-------------|
| `docs/specs/SYSTOLIC_ARRAY_V2_SPEC.md` §9 | Verified 42-cycle WS schedule (source of truth) |
| `docs/specs/PE_SPEC.md` §13 | PE-v2 contract (ports, accumulator semantics, control priority) |
| `docs/PROJECT_STATE.md` Decision 13 | The four controller/input-feed decisions (2026-08-20) |
| `rtl/common/systolic_array_v2.sv` | Frozen array interface (ports, per-row result drain, feedback) |
| `rtl/common/pe_v2.sv` | Frozen PE-v2 (BREG/MREG/PREG latency, psum cascade) |
| `sim/tb_systolic_array_v2.sv` | Verified driver (`act_value`, `set_tile_weights`, `drive_group`) |
| `docs/specs/INPUT_FEED_SPEC.md` | V1 input-feed (frozen; V2 analog, not reused) |
