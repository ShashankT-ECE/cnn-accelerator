# Input-Feed / Line-Buffer Specification — V1 Baseline

> **Status:** IMPLEMENTATION-READY for the input-feed/line-buffer module.
> This document is the contract for the V1 input-feed/line-buffer unit, which
> turns the array's verified 1-D correlation into the true 2-D 5×5 convolution.
>
> It is governed by **Decision 9** (row-decomposed 5-pass mapping, recorded in
> `docs/PROJECT_STATE.md`) and by the frozen PE / systolic-array contracts
> (`PE_SPEC.md`, `SYSTOLIC_ARRAY_SPEC.md`). It does **not** define RTL, and it
> does **not** choose BRAM/URAM/distributed-RAM, AXI, DMA, or PS integration.

> **AMENDED (2026-08-17):** the canonical group schedule's result-drain figure
> is corrected from **8 cycles (74 total)** to **16 cycles (82 total)**. The
> frozen `pe.sv`/`systolic_array.sv` expose results with a 1-cycle registered
> `result_out` latency and a combinational one-hot `result_req` mux, so each of
> the 8 columns requires 2 cycles (assert `result_req[c]`, then read the bus on
> the next cycle while it is still asserted). Derived throughput figures
> (70→78, 5,328→5,904, 5,040→5,616) are corrected accordingly. `pe.sv` and
> `systolic_array.sv` are **unchanged**; this corrects only the schedule
> arithmetic in this spec (§7, §12, §13, §15).

---

## 1. Purpose and Scope

**REQUIREMENT** — V1 must compute LeNet-5 Conv1 (1×28×28 → 6×24×24, valid,
no padding) correctly.

**DECISION (Decision 9)** — The 5×5 convolution is computed as **five sequential
1-D horizontal 5-tap passes**, one per kernel row `k_y = 0..4`. The array itself
performs only the verified 1-D 5-tap correlation; this module orchestrates the
five passes into one true 2-D 5×5 convolution.

This document defines:

- the exact per-pass activation and weight streams required by
  `rtl/common/systolic_array.sv`;
- the 7-cycle lead-in and the proof that it is required **before every pass**;
- the complete five-pass group schedule;
- the derived storage and throughput;
- the minimum control interface;
- the verification requirements, including an independent 2-D golden model.

This document does **not**:

- modify `pe.sv`, `systolic_array.sv`, either testbench, or any existing spec;
- reintroduce the withdrawn single-pass 25-tap 2-D schedule
  (`SYSTOLIC_ARRAY_SPEC.md` §20.4);
- choose a physical memory technology (BRAM/URAM/distributed RAM);
- define AXI/DMA/PS interfaces;
- write RTL.

---

## 2. Conv1 Geometry

**DECISION** — Fixed V1 Conv1 geometry:

| Parameter | Value |
|-----------|-------|
| Input | 1 × 28 × 28 (grayscale), `input[r][c]`, `r,c ∈ [0,27]` |
| Kernel | 5 × 5, `weight[ch][ky][kx]`, `ky,kx ∈ [0,4]`, 6 output channels |
| Output | 6 × 24 × 24 valid (no padding), `output[ch][y][x]`, `y,x ∈ [0,23]` |
| Output width | 24 = 28 − 5 + 1 (valid) |
| Output groups per row | 3 (24 = 3 × 8) |
| MACs per output pixel | 25 (= 5 passes × 5 taps) |

**DECISION** — Each 24-pixel output row is processed as **three 8-pixel groups**:

| Group `g` | Output columns | `base` (= 8g+7) |
|-----------|----------------|-----------------|
| 0 | 0–7 | 7 |
| 1 | 8–15 | 15 |
| 2 | 16–23 | 23 |

**IMPLIED** — Column `c` of the array (0..7) produces output pixel `base − c`
(reverse index, `SYSTOLIC_ARRAY_SPEC.md` §10.4). For group 2: column 0 → pixel
23, column 7 → pixel 16.

**IMPLIED** — No padding means no out-of-range reads: the window for output
pixel `(y, x)` is `input[y..y+4][x..x+4]`, all indices within `[0,27]`.

---

## 3. Five-Pass Mapping

**DECISION (Decision 9)** — Each output group is computed by five sequential
passes:

| Pass | Kernel row `k_y` | Input row read | Weights broadcast |
|------|------------------|----------------|-------------------|
| 0 | 0 | `y + 0` | `weight[ch][0][0..4]` |
| 1 | 1 | `y + 1` | `weight[ch][1][0..4]` |
| 2 | 2 | `y + 2` | `weight[ch][2][0..4]` |
| 3 | 3 | `y + 3` | `weight[ch][3][0..4]` |
| 4 | 4 | `y + 4` | `weight[ch][4][0..4]` |

**DECISION** — Each pass performs a **1-D 5-tap horizontal correlation** along
one input row. The PE accumulator is **preserved across all five passes**;
`accum_clear` is asserted **only before pass 0** of a new output group.

**IMPLIED** — After pass 4, each `PE(ch, c)` accumulator holds the full 2-D dot
product for output channel `ch`, output pixel `base − c`:

```
accum(ch, c) = Σ_{k_y=0}^{4} Σ_{k_x=0}^{4} weight[ch][k_y][k_x] · input[y+k_y][base−c+k_x]
```

**IMPLIED** — The array is unaware of the 2-D structure. It only ever performs a
1-D 5-tap correlation per pass. The input-feed/controller owns the 5-pass
orchestration and supplies the correct input row and five weights per pass.

---

## 4. Array Contract (derived from the frozen RTL)

This section restates, with RTL references, the exact behavior the input-feed
must satisfy. These are **not** new requirements; they are the frozen
`systolic_array.sv` / `pe.sv` behavior.

**RESEARCH FACT (verified against `systolic_array.sv`)** — Activation shift:
seven per-row registers; column `c` sees `act_in` delayed by `c` cycles
(`systolic_array.sv` `shift_reg`, `act_delayed`; `SYSTOLIC_ARRAY_SPEC.md` §6–§7).
`act_delayed[r][0] = act_in[r]` (combinational); `act_delayed[r][c] =
shift_reg[r][c-1]` for `c ≥ 1`.

**RESEARCH FACT (verified against `systolic_array.sv`)** — The shift chain
resets to `0` **only on `rst`**, never on `accum_clear`. This is the key fact
for the lead-in analysis (§6).

**RESEARCH FACT (verified against `pe.sv`)** — Weight register (`BREG=1`) loads
`weight_in` one cycle after `weight_load`; the MAC uses the *old* weight on the
assertion cycle (weight-lead skew, `PE_SPEC.md` §5.5, §9). Product register
(`MREG=1`) and accumulator (`PREG=1`) give a 2-cycle MAC latency
(`PE_SPEC.md` §7.8).

**RESEARCH FACT (verified against `pe.sv`)** — Per-row weight broadcast: `w_in[r]`
fans out to all 8 PEs of row `r`. Rows `0..5` are the 6 output channels; rows
`6..7` are idle (`w_in = 0`).

**IMPLIED (the array's exact compute)** — With `weight_load` asserted every
cycle and `w_in[r]` carrying `w_r[s+1]` during activation cycle `s`, the weight
register holds `w_r[s]` during cycle `s`. Column `c` then computes, over the
cycles of a pass,

```
PE(r,c) = Σ_s a[s−c] · w_r[s]
```

where `a[s]` is the activation presented at `act_in` during cycle `s` and
`w_r[s]` the weight in row `r`'s weight register during cycle `s`. This is the
reverse-index 1-D correlation (`SYSTOLIC_ARRAY_SPEC.md` §10.4), verified by
`tb_systolic_array.sv` (379/379).

---

## 5. Exact Per-Pass Stream

**DECISION** — For pass `k_y` (input row `R = y + k_y`) of a group with base
`b = 8g+7`, the activation stream is:

```
a[s] = input[R][b + s],   s = −7 .. 4     (7 lead-in + 5 taps)
a[s] = 0                   otherwise
```

and the per-channel weight stream is:

```
w_ch[s] = weight[ch][k_y][s],   s = 0 .. 4   (5 taps)
w_ch[s] = 0                      otherwise
```

The 13-cycle canonical pass schedule (non-overlapped), for every row `ch`:

| Cycle | `s` | `act_in` (all rows) | weight reg `w_ch[s]` | `weight_load` | `w_in[ch]` | column `c` product (`input[R][b+s−c]·w_ch[s]`) |
|------:|:---:|:---|:---:|:---:|:---:|:---|
| L1 | −7 | `input[R][b−7]` | 0 | 1 | 0 | (lead-in, weight 0) |
| L2 | −6 | `input[R][b−6]` | 0 | 1 | 0 | (lead-in) |
| L3 | −5 | `input[R][b−5]` | 0 | 1 | 0 | (lead-in) |
| L4 | −4 | `input[R][b−4]` | 0 | 1 | 0 | (lead-in) |
| L5 | −3 | `input[R][b−3]` | 0 | 1 | 0 | (lead-in) |
| L6 | −2 | `input[R][b−2]` | 0 | 1 | 0 | (lead-in) |
| L7 | −1 | `input[R][b−1]` | 0 | 1 | `w_ch[0]` | (lead-in; preloads tap-0 weight) |
| T0 | 0 | `input[R][b]` | `w_ch[0]` | 1 | `w_ch[1]` | `input[R][b−c]·w_ch[0]` |
| T1 | 1 | `input[R][b+1]` | `w_ch[1]` | 1 | `w_ch[2]` | `input[R][b+1−c]·w_ch[1]` |
| T2 | 2 | `input[R][b+2]` | `w_ch[2]` | 1 | `w_ch[3]` | `input[R][b+2−c]·w_ch[2]` |
| T3 | 3 | `input[R][b+3]` | `w_ch[3]` | 1 | `w_ch[4]` | `input[R][b+3−c]·w_ch[3]` |
| T4 | 4 | `input[R][b+4]` | `w_ch[4]` | 1 | 0 | `input[R][b+4−c]·w_ch[4]` |
| D | — | 0 | 0 | 0 | 0 | `acc += input[R][b+4−c]·w_ch[4]` |

**IMPLIED** — After cycle D, the pass partial sum
`Σ_{k_x} input[R][b−c+k_x]·weight[ch][k_y][k_x]` has been added to each PE's
accumulator.

**DECISION** — The reverse index means column `c` produces output pixel `b − c`;
the mapping above is what realizes it (each column's window is shifted left by
`c`, absorbed by the lead-in + shift).

---

## 6. Lead-In and Pass Transitions (proved from RTL)

### 6.1 The 7-cycle lead-in is required before every pass

**DECISION** — The 7-cycle lead-in is required **before every kernel-row pass**,
not only before pass 0.

**Proof (from `systolic_array.sv`):**

1. `accum_clear` clears only the PE accumulator and product registers
   (`pe.sv`), **not** the activation shift chain. The shift chain
   (`systolic_array.sv` `shift_reg`) resets to `0` only on `rst` (§4).
2. Therefore, at the start of pass `k_y+1`, the 7 shift registers still hold the
   tail of pass `k_y`'s stream: pixels of input row `y + k_y`, not row
   `y + k_y + 1`.
3. Pass `k_y+1` must have its own row's left-edge pixels
   `input[y+k_y+1][b−7 .. b−1]` sitting in the shift chain at its first tap
   cycle (T0), so column `c` sees `input[y+k_y+1][b−c]` and multiplies it by
   `weight[ch][k_y+1][0]`.
4. The stale pixels (row `y+k_y`) cannot serve this purpose — they are from the
   wrong input row.
5. Hence seven new-row pixels must be shifted in **before each pass's taps**:
   the 7-cycle lead-in is mandatory for every pass.

**IMPLIED** — The lead-in is semantically required (it supplies the left edge of
each column's 5-pixel window), not a mere "reset convenience." Flushing the
chain with zeros would truncate every column's window and produce wrong results.

### 6.2 Pass transition is uniquely determined

**DECISION** — The safe pass-transition schedule is **uniquely determined** by
the RTL: after pass `k_y`'s last tap (T4), present pass `k_y+1`'s 7 lead-in
cycles, then its 5 taps. There is **no** open decision on correctness.

### 6.3 Proven-safe overlap (CANDIDATE, not required)

**CANDIDATE (proven safe)** — The drain cycle D of pass `k_y` may present the
first lead-in activation `input[y+k_y+1][b−7]` instead of `0`. This is safe
because:

1. During D the weight register is already `0` (it was zero-loaded at T4 by the
   zero-padded weight-lead), so the lead-in value multiplies by `0` and adds
   nothing to the accumulator.
2. D still drains the previous pass's last product
   (`input[R][b+4−c]·w_ch[4]`) into the accumulator via the product register.

This folds one cycle per internal pass boundary (4 cycles per group). It is an
optimization and is **not** required for correctness; the canonical
non-overlapped schedule (§5, §7) is the baseline.

---

## 7. Complete Five-Pass Group Schedule

**DECISION** — Canonical (non-overlapped) schedule for one output group:

| Component | Cycles |
|-----------|-------:|
| `accum_clear` (before pass 0 only) | 1 |
| Pass 0: 7 lead-in + 5 taps + 1 drain | 13 |
| Pass 1: 7 lead-in + 5 taps + 1 drain | 13 |
| Pass 2: 7 lead-in + 5 taps + 1 drain | 13 |
| Pass 3: 7 lead-in + 5 taps + 1 drain | 13 |
| Pass 4: 7 lead-in + 5 taps + 1 drain | 13 |
| column-sequential `result_req` drain (8 columns × 2 cycles) | 16 |
| **Total per group** | **82** |

**DECISION** — `accum_clear` is asserted exactly once per group (before pass 0).
It clears accumulators and product registers but **not** the shift chain, so pass
0 still needs its full 7-cycle lead-in.

**DECISION** — After pass 4's drain cycle, all 25 MACs per output pixel have
accumulated; the controller then asserts `result_req[0..7]` one column at a time,
**2 cycles per column (16 cycles total)**, to capture the 8 output pixels
(column `c` → pixel `base − c`).

**IMPLIED (derived from the frozen RTL)** — Each column needs 2 cycles because
`result_out` is registered (1-cycle capture, `PE_SPEC.md` §7.8) and the array's
`result_out` bus is a combinational one-hot mux on the **live** `result_req`
(`systolic_array.sv`). The controller asserts `result_req[c]` for one cycle (the
PE latches `accumulator → result_out` at the end of that cycle), then reads the
bus on the next cycle **while `result_req[c]` is still asserted** so the mux
still routes column c. Advancing to column c+1 on that next cycle would route
the wrong (stale) column, so 8 columns cannot be drained in 8 cycles. This is the
frozen array's read protocol, verified by `tb_systolic_array.sv` (2 cycles/column).

**CANDIDATE (proven safe, §6.3)** — Overlapping each pass's drain with the next
pass's first lead-in reduces the group to **78 cycles** (passes 0–3 each 12
cycles, pass 4 13 cycles, plus clear and the 16-cycle read). Not required for V1.

---

## 8. Output Groups and Worked Example (columns 16–23)

### 8.1 Group boundaries

| Group | Output columns | `base` | per-pass lead-in columns | per-pass tap columns |
|-------|----------------|--------|--------------------------|----------------------|
| 0 | 0–7 | 7 | `[0..6]` | `[7..11]` |
| 1 | 8–15 | 15 | `[8..14]` | `[15..19]` |
| 2 | 16–23 | 23 | `[16..22]` | `[23..27]` |

Adjacent groups overlap by 4 input columns (group `g`'s taps `[8g+7..8g+11]`
overlap group `g+1`'s lead-in `[8g+8..8g+14]`).

### 8.2 Worked example — group 2 (output columns 16–23)

`base = 23`. Column `c` → output pixel `23 − c`. Per pass, the 12 input columns
are `input[row][16..27]` (lead-in `16..22`, taps `23..27`).

| Pass `k_y` | Input row | Weights (`weight[ch][k_y][kx]`) | Lead-in (`input[row][16..22]`) | Taps (`input[row][23..27]`) |
|:---:|:---:|:---|:---|:---|
| 0 | `y+0` | `w0[0..4]` | `input[y][16..22]` | `input[y][23..27]` |
| 1 | `y+1` | `w1[0..4]` | `input[y+1][16..22]` | `input[y+1][23..27]` |
| 2 | `y+2` | `w2[0..4]` | `input[y+2][16..22]` | `input[y+2][23..27]` |
| 3 | `y+3` | `w3[0..4]` | `input[y+3][16..22]` | `input[y+3][23..27]` |
| 4 | `y+4` | `w4[0..4]` | `input[y+4][16..22]` | `input[y+4][23..27]` |

Concrete trace for one PE (output pixel 20, i.e. column `c = 23 − 20 = 3`):
over the 5 passes it accumulates, in order, the 25 terms
`input[y+k_y][20+k_x]·weight[ch][k_y][k_x]`, `k_y,k_x ∈ [0,4]` — the complete
5×5 window anchored at `(y, 20)`.

The 8 columns together compute pixels 16–23, each as the full 5×5 window at its
own horizontal anchor, with all windows overlapping by 4 columns.

---

## 9. Input Stream / Window Generation

**DECISION** — For output row `y`, the 5 passes read input rows `y..y+4` in
full (each pass reads its row's 28 columns across the 3 groups, with the 4-column
inter-group overlap). As `y` advances, the working set of input rows slides by
one: output row `y+1` reads rows `y+1..y+5`.

**IMPLIED** — Horizontal ordering within a pass is deterministic: for each group
in turn (g = 0,1,2), present the 7 lead-in columns then the 5 tap columns, one
pixel per cycle, on all 8 `act_in` lines identically (single-channel input).

**IMPLIED** — Vertical reuse: input row `R` is read once for each of the 5
output rows `y = R−4 .. R` that uses it (i.e., 5 times total). This is the
reason a multi-row buffer exists; the depth is derived in §10.

**DECISION** — The exact row-major `act_in` order is:
for each output row `y` (0..23), for each pass `k_y` (0..4), for each group `g`
(0..2): emit `input[y+k_y][8g .. 8g+11]` as the 7 lead-in + 5 taps, with the
inter-group/pass gaps per §5/§7. This order is the input-feed's contract and is
shared with the weight controller so weights stay aligned with activations.

---

## 10. Storage Requirement

**DECISION** — The 117-pixel figure is retained as a **mathematical data-span
lower bound** only:

```
(kernel_height − 1) · input_width + kernel_width = 4 · 28 + 5 = 117
```

This is the span of a single 5×5 window sliding one column at a time
(4 full rows + 5 columns of the current row). It is **not** the physical buffer
depth for this schedule.

**DECISION (derived from the 5-pass schedule)** — The actual data-availability
requirement is **five full input rows = 5 × 28 = 140 pixels**:

- Each pass reads one input row **in full** (all 28 columns, because the 3
  groups of an output row span `[0,27]`).
- The 5 passes of one output row read 5 distinct rows (`y..y+4`).
- The "4 rows + 5 columns" trick that yields 117 assumes column-granular
  streaming; the group-based schedule reads whole rows, so it requires the
  full 5-row working set.

**IMPLIED** — The working set is a sliding 5-row window (140 pixels) over the
28×28 input. Whether this is realized as a 5-row line buffer, a frame buffer,
or direct external addressing is a **memory-interface decision** (§11, OPEN
DECISION). 117 is retained only as the mathematical lower bound, not as the
implemented depth.

---

## 11. Memory Interface (architectural assumptions only)

**DECISION** — V1 assumes the 28×28 input pixels are available to the input-feed
in row order, one pixel per cycle when read. No further arrival contract is
imposed here.

**OPEN DECISION** — The physical storage technology (BRAM vs URAM vs distributed
RAM), the presence or absence of a frame buffer vs a 5-row line buffer, and any
AXI/DMA/PS transport are **not** chosen by this spec. They are deferred to a
subsequent implementation decision document. This spec constrains only the
logical data-availability (§10) and the deterministic `act_in` stream (§9).

**DECISION** — The weight stream is supplied by the BRAM weight controller (a
separate module); the input-feed's job is the activation stream and the shared
schedule timing. `w_in[ch] = weight[ch][k_y][k_x]` during pass `k_y`, tap `k_x`.

---

## 12. Control Interface

**DECISION** — The array interface is already fixed (`SYSTOLIC_ARRAY_SPEC.md`
§18) and is not extended. The input-feed/controller drives it with the schedule
of §5/§7:

| Signal | Driven by | V1 behavior |
|--------|-----------|-------------|
| `act_in[0..7]` | input-feed | per-pass activation stream (§5), all rows identical |
| `w_in[0..7]` | weight controller | `weight[ch][k_y][k_x]` per pass/tap; rows 6–7 = 0 |
| `weight_load` | controller | asserted every lead-in and tap cycle (§5) |
| `accum_clear` | controller | once, before pass 0 of a group |
| `zero_skip` | controller | tied `0` (V1) |
| `result_req[0..7]` | controller | one-hot, one column for 2 cycles (latch + read), after pass 4 |
| `result_out[0..7]` | array → controller | captured results |

**DECISION** — The minimum internal sequencing state is four counters
(output row `y`, group `g`, pass `k_y`, activation index `s`). No external
`start`/`done`/`valid`/`group` signals are required for V1 correctness: the
schedule is deterministic and self-timed.

**OPEN DECISION** — A top-level `start` (frame trigger) or `done` (frame
complete) signal, if any, is an AXI/PS-integration concern and is **not**
invented here. It is only needed when the accelerator is driven by an external
host, which is out of V1 scope.

---

## 13. Reset and Transitions

**DECISION** — `rst` (synchronous, active-high) resets all PE registers
(accumulator, weight, result) and the shift chain to `0`. It is the only signal
that resets the shift chain.

**DECISION** — `accum_clear` resets PE accumulators and product registers but
**not** the shift chain. It is asserted once per group (before pass 0).

**DECISION** — Pass transition: after pass `k_y`'s T4, present pass `k_y+1`'s
7 lead-in cycles (§6). No `accum_clear`, no flush, no weight register reset is
needed between passes (the weight register is zero-loaded by the zero-padded
weight-lead at T4, so the next pass's lead-in multiplies by 0).

**DECISION** — Group transition: after pass 4's drain, drain the 8 columns via
`result_req` (16 cycles, 2 per column), then assert `accum_clear` and begin the
next group's pass 0. The shift chain is not cleared between groups; pass 0's
lead-in overwrites the stale pixels.

**DECISION** — Output-row transition: output row `y+1` reuses input rows
`y+1..y+4`; the input-feed advances its 5-row window by one row. No reset is
needed.

---

## 14. Boundaries

**DECISION** — Valid convolution, no padding:

- **First valid window** — output `(0,0)` reads `input[0..4][0..4]`; group 0's
  lead-in `input[y][0..6]` begins at column 0, in-bounds.
- **Final valid window** — output `(23,23)` reads `input[23..27][23..27]`; group
  2's taps `input[y+4][23..27]` end at column 27, in-bounds.
- **No out-of-range reads** — every lead-in and tap column lies in `[0,27]` for
  every group (see §8.1). No negative columns, no columns ≥ 28.
- **Left edge** — group 0 needs no pixels left of column 0 (lead-in is the
  window's own left edge, not padding).
- **Right edge** — group 2 needs no pixels right of column 27 (taps end at the
  input's right edge).

---

## 15. Throughput

**DECISION (derived after the schedule is established)** — Cycles per group:

| Variant | Cycles/group |
|---------|-------------:|
| Canonical (non-overlapped, §7) | **82** |
| Proven-safe overlap (§6.3, CANDIDATE) | **78** |

**IMPLIED** — Full-frame cycles = 72 groups × cycles/group, i.e. **5,904**
(canonical) or **5,616** (overlapped), plus any frame-level overhead the
controller adds. These are **not** asserted as final performance figures.

**DECISION** — The withdrawn "48 MACs/cycle sustained" figure
(`SYSTOLIC_ARRAY_SPEC.md` §17/§20.4) is **not** reused. Sustained MACs/cycle is
reduced by the 7 lead-in cycles (weight = 0) per 5 taps per pass, and its final
value depends on the overlap decision and the controller microarchitecture. It
is an **OPEN DECISION**, re-derived from the final controller schedule, not
asserted here.

---

## 16. Verification Requirements

**REQUIREMENT** — Directed checks must cover, at minimum:

1. **All 5 kernel rows** — each pass contributes exactly one `k_y` row.
2. **All 5 taps** — each pass performs the full 5-tap correlation.
3. **Accumulation across passes** — the PE accumulator carries the running sum
   across the 5 passes.
4. **`accum_clear` only before pass 0** — no clear between passes.
5. **All three output groups** — 0–7, 8–15, and **especially 16–23** (right
   edge).
6. **Row/group transitions** — no stale add-back, correct sliding-window reuse.
7. **One-cycle weight/activation skew** — product uses the correct pass tap
   weight.
8. **First/final valid windows** — no padding, no out-of-range reads.
9. **Lead-in correctness** — every pass's lead-in, and that `accum_clear` does
   **not** reset the shift chain.
10. **True 2-D Conv1 correctness** — against the independent golden model.

**CRITICAL GOLDEN-MODEL REQUIREMENT** — The 2-D golden model **must be
independent of the five-pass hardware schedule**. It must directly compute:

```
output[ch][y][x] = Σ_{k_y=0}^{4} Σ_{k_x=0}^{4}
                     input[y + k_y][x + k_x] · weight[ch][k_y][k_x]
```

It must **not** be derived from the per-pass activation/weight streams being
verified (that would make it tautological). The model computes the 2-D dot
product directly from `(y, x)` and the 5×5 kernel; the hardware result is
compared bit-exact against it.

**REQUIREMENT** — The golden model must match the array arithmetic exactly:
signed 8-bit operands, 32-bit accumulation, reverse column index
(`result_out` column `c` ↔ pixel `base − c`), read-without-clear.

---

## 17. Open Decisions

| # | Item | Status |
|---|------|--------|
| 1 | Physical storage technology (BRAM / URAM / distributed RAM) | **OPEN** — later implementation decision |
| 2 | Line-buffer vs frame-buffer vs external addressing | **OPEN** — later implementation decision |
| 3 | Pass-transition overlap (drain ↔ lead-in) | **CANDIDATE** — proven safe (§6.3), not yet adopted |
| 4 | Sustained throughput / final cycles-per-layer | **OPEN** — depends on #3 and controller microarchitecture |
| 5 | Controller microarchitecture (FSM state encoding, counter widths) | **OPEN** — implementation detail |
| 6 | AXI/DMA/PS input transport and any `start`/`done` signaling | **OPEN** — out of V1 scope, deferred |
| 7 | Clock frequency target | **OPEN** — deferred to synthesis (`PE_SPEC.md` §11.6) |
| 8 | Decision 7 sign-off (governance) | **OPEN** — not blocking |

**RESOLVED / NOT OPEN:** the lead-in requirement (before every pass) and the
pass-transition correctness are proven from the frozen RTL (§6); the storage
data-availability (140 pixels) and canonical cycles/group (82) are derived
(§10, §15).

---

## Appendix A: References

| Reference | Description |
|-----------|-------------|
| `docs/PROJECT_STATE.md` | Decision 9 (row-decomposed 5-pass mapping) |
| `docs/specs/SYSTOLIC_ARRAY_SPEC.md` | Array contract; §20.4 (withdrawn single-pass mapping) |
| `docs/specs/PE_SPEC.md` | PE contract (§5.5 weight, §7.8 pipeline) |
| `rtl/common/pe.sv` | PE RTL (weight/product/accumulator, `accum_clear` scope) |
| `rtl/common/systolic_array.sv` | Array RTL (shift chain, `rst`-only shift reset) |
| `sim/tb_systolic_array.sv` | Verified 1-D reference model (`ref = Σ a[t]·w[t+c]`) |
| `docs/reference/architecture_major_project.pdf` | Roadmap (§3.2, §4) — reference only |
