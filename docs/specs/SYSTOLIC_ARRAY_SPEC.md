# Systolic Array Specification — 8×8 PE Array (V1 Baseline)

> **Status:** IMPLEMENTATION-READY — dataflow and scheduling decisions resolved;
> interface and schedule defined (Decision 7 sign-off pending).
> This document is the implementation contract for `rtl/common/systolic_array.sv`.
> It records the resolved V1 dataflow, the 8×8 topology, the cycle-level
> schedule, and the array interface. It does **not** define RTL signal names
> beyond the PE contract, and it does **not** specify the top-level FSM, the
> BRAM weight controller, the input buffer, or any AXI interface.

---

## 1. Purpose and Scope

The systolic array is the compute fabric of the CNN accelerator. It instantiates
**64 Processing Elements (`rtl/common/pe.sv`) in an 8×8 grid** and coordinates
their data movement and result collection.

This document defines the **V1 shared baseline systolic array**:

- **Dataflow (V1):** Output-Stationary — *weight-broadcast + activation-shift +
  local PE accumulation* (see §3). This is **not** the roadmap's literal
  "fixed weight-stationary" wording; the distinction is a formal decision
  recorded here and flagged for `docs/PROJECT_STATE.md`.
- **Topology:** 8 rows × 8 columns = 64 PEs (roadmap §3.2).
- **Compute rate:** up to 64 MACs/cycle (peak); one MAC/cycle/PE in steady
  state where the schedule permits (see §16, §17).
- **No partial-sum cascade in V1.** True spatial Weight-Stationary and any
  `psum_in`/`psum_out` PE extension are deferred to V2.

This document does **not**:

- Modify `pe.sv` or `tb_pe.sv` (both remain locked).
- Define AXI4-Stream / AXI4-Lite interfaces or register maps.
- Specify the top-level FSM, the BRAM weight controller, the input buffer, or
  the input-feed/line-buffer unit (their *interfaces* to the array are
  defined in §18; their internals are out of scope).
- Include sparsity (the `zero_skip` path is present but tied inactive in V1).
- Specify the clock target or synthesis/implementation constraints.

---

## 2. Source and Specification Status

### 2.1 Documentation Hierarchy

Authority tiers, consistent with `docs/specs/PE_SPEC.md` §2.1:

| Tier | Source | Authority for the Array |
|------|--------|-------------------------|
| 1 | Explicitly agreed project requirements | Binding |
| 2 | Detailed specifications (this document, `PE_SPEC.md`) | Binding within scope |
| 3 | External technical research (DS925, UG579, dataflow taxonomy) | Informative |
| 4 | Engineering recommendations | Advisory |
| 5 | Unresolved assumptions / open decisions | No authority |

### 2.2 Label Convention

Every claim in this document carries one of the following labels:

| Label | Meaning |
|-------|---------|
| **REQUIREMENT** | Established by the architecture roadmap or an agreed project decision |
| **DECISION** | Resolved by team agreement (recorded in `docs/PROJECT_STATE.md` or this spec) |
| **RESEARCH FACT** | Established by external technical reference |
| **CANDIDATE** | Proposed design choice — not yet agreed |
| **IMPLIED** | Direct logical consequence of a requirement/decision, not explicitly stated in source |
| **OPEN DECISION** | Unresolved — must be decided before RTL |

### 2.3 Primary Sources

- `docs/reference/architecture_major_project.pdf` §3.2 (Systolic Array,
  Dataflow Controller), §4 (V1 table) — roadmap.
- `docs/specs/PE_SPEC.md` §3.1 (R1–R9), §5, §6, §7.8, §8.4 — PE contract.
- `docs/PROJECT_STATE.md` Decisions 1–6 — numeric format, accumulator width,
  overflow, result-read protocol, reset, pipeline.
- `rtl/common/pe.sv` — the implemented, verified PE.

---

## 3. Resolved V1 Dataflow

### 3.1 The Roadmap Conflict and Its Resolution

**REQUIREMENT (roadmap §3.2, §4)** — V1 is described as *"fixed
weight-stationary: weights held in place, activations shift."*

**DECISION (this spec; supersedes the roadmap wording)** — The finalized PE
has a pinned local accumulator and **no `psum_in`/`psum_out`/`wgt_out`**
(`PE_SPEC.md` §6, `pe.sv:14-25`). A literal fixed-weight dataflow would compute
`w × Σ activations`, which is **not** a convolution (`Σ w_k · a_k`). Classical
spatial Weight-Stationary requires a partial-sum path that the PE does not have.

Therefore V1 adopts the only correct dataflow compatible with the locked PE:

> **V1 dataflow = Output-Stationary, tap-serial.**
> Each PE holds its own **output** stationary in its local accumulator across the
> whole tap sequence; **weights are broadcast per row and reloaded every tap**;
> **activations shift across columns**. This is taxonomically Output-Stationary,
> not Weight-Stationary. True spatial Weight-Stationary (partial-sum cascade)
> is deferred to V2.

This decision **amends** `PE_SPEC.md` §3.1 R9's literal "weight-stationary"
label and the roadmap §3.2 wording. It must also be recorded in
`docs/PROJECT_STATE.md` as **Decision 7** (see §20.1).

### 3.2 Terminology Used in This Spec

| Term | Meaning in this spec |
|------|----------------------|
| **Tap** | One `(k_x, k_y, ic)` term of the K×K×IC convolution kernel. Tap `t` is one serialized index into the kernel (see §10). |
| **Tap cycle** | One clock cycle in which the array presents one tap's weight and activation and every PE performs one MAC. |
| **Output group** | One pass through the array that produces one tile of results (up to 8 rows × 8 columns). |
| **Output tile** | The 8 consecutive output pixels (one per column) for each active row. |

---

## 4. 8×8 Topology

**REQUIREMENT (roadmap §3.2)** — 64 PE instances wired in a grid; every cycle,
all 64 PEs execute a MAC simultaneously — 64 MACs/cycle peak.

**DECISION** — The array is 8 rows × 8 columns. PE `(r, c)` is the instance at
row `r` (0..7) and column `c` (0..7), instanced row-major as `pe[r*8 + c]`.

```
             column 0   column 1   ...   column 7
  row 0      PE(0,0) ──► PE(0,1) ──► ... ──► PE(0,7)
  row 1      PE(1,0) ──► PE(1,1) ──► ... ──► PE(1,7)
   ...          ...        ...             ...
  row 7      PE(7,0) ──► PE(7,1) ──► ... ──► PE(7,7)
             (──► = activation shift, one 1-cycle register per hop)
```

Each PE is the verified `pe.sv` with its fixed port list (`pe.sv:14-25`). The
array adds **interconnect and control only**; it does not change the PE.

---

## 5. Row / Column Mapping

**DECISION** — For V1 (LeNet-5 Conv1):

- **Row `r` = output channel `r`** (a tile of up to 8 output channels).
  Rows beyond the layer's actual output-channel count are idle (see §14).
- **Column `c` = one output pixel**, with columns covering **8 adjacent output
  pixels** of the same output row `y`. Column `c` produces pixel index
  `base - c` (reverse index; see §10.4).

**IMPLIED** — All 8 columns of a row share the **same weight** at a given tap
(one output channel), and all 8 rows share the **same activation stream** at a
given tap (one input feature map read by every output channel). Only the weight
differs across rows; only the (shifted) activation differs across columns.

**DECISION** — This mapping is Conv1-specific; a general mapping for Conv2/FC
layers (different input-channel counts) is **not** part of V1 and is deferred.
V1 targets Conv1 only (roadmap §4: "correctly computes LeNet conv1").

---

## 6. PE-to-PE Connections

**DECISION** — The only PE-to-PE wire is the **activation shift**. All other PE
ports are driven by array-level buses (weight, control) or collected into the
result bus.

| Connection | Wiring | Source of truth |
|------------|--------|-----------------|
| `PE(r,c).activation_in` | `act_in[r]` for c=0; else the 1-cycle-registered `PE(r,c-1).act_out` | §7 |
| `PE(r,c).act_out` | feeds the register to `PE(r,c+1).activation_in`; unconnected at c=7 | `pe.sv:37` (combinational) |
| `PE(r,c).weight_in` | `w_in[r]` (per-row broadcast) | §8 |
| `PE(r,c).weight_load` | array-wide `weight_load` | §8 |
| `PE(r,c).accum_clear` | array-wide `accum_clear` | §13 |
| `PE(r,c).zero_skip` | tied `0` in V1 | §12 |
| `PE(r,c).result_request` | `result_req[c]` (per-column) | §11 |
| `PE(r,c).result_out` | collected into the array result bus | §11 |

**RESEARCH FACT** — `act_out = activation_in` is combinational (`pe.sv:37`,
`PE_SPEC.md §8.4`). It produces a **broadcast within a cycle**, not a shift.
The one-cycle-per-column shift is therefore implemented with **array-level
registers** on each inter-column hop — it is the array's responsibility, not the
PE's.

**DECISION** — Seven 8-bit registers per row implement the shift (56 registers
total). All shift registers reset to `0` on `rst`.

---

## 7. Activation Movement

**REQUIREMENT (roadmap §3.2)** — Activations move across rows (horizontally,
along each row).

**DECISION** — Each row receives an 8-bit activation per tap cycle and shifts it
left-to-right, one column per cycle:

```
act_in[r] ──► [PE(r,0)] ──(reg)──► [PE(r,1)] ──(reg)──► ... ──(reg)──► [PE(r,7)]
```

`PE(r,c)` at cycle `t` sees the activation value that entered row `r` at cycle
`t - c`. Because `act_out` is combinational, the array inserts the register; the
PE does not delay the activation itself.

**DECISION (V1 Conv1)** — All 8 rows receive the **same** activation stream
(every output channel reads the same input feature map). The array exposes 8
`act_in` inputs; for Conv1 the controller drives all 8 identically.

**IMPLIED** — The activation value for tap `t` is presented at the array input
`act_in`; the per-column delay comes from the shift chain, not from the input
feed. The input-feed/line-buffer unit (out of scope, §1) is responsible for
producing the correct 2-D activation sequence for the current tap (see §10.3).

---

## 8. Weight Distribution and Loading

**REQUIREMENT (roadmap §3.2, BRAM Weight Controller)** — Weights are supplied to
the array at the correct rate each cycle.

**DECISION** — Weights are **broadcast per row and reloaded every tap**:

- `w_in[r]` (8-bit, signed) carries the weight for output channel `r` at the
  current tap.
- `w_in[r]` fans out to all 8 PEs of row `r` (`PE(r,0..7).weight_in`).
- `weight_load` is a single array-wide signal, asserted **every tap cycle**, so
  every PE loads its row's current-tap weight each cycle.

**IMPLIED** — There is **no temporal weight reuse** per PE (the weight changes
every tap). There is **8× spatial reuse** (one `w_in[r]` value feeds 8 columns).
This is the defining property of the Output-Stationary dataflow (§3.1).

**RESEARCH FACT** — Weight traffic is one byte per active row per tap cycle:
up to 8 B/cycle, trivially served by on-chip BRAM on the XCK26.

---

## 9. PE Timing and the One-Cycle Weight/Activation Skew

**REQUIREMENT / DECISION** — The PE's cycle-level timing is fixed by
`PE_SPEC.md` §7.8 and Decision 6, and is verified in `tb_pe.sv`:

| Path | Latency (cycles) |
|------|------------------|
| `activation_in` → accumulator update | 2 |
| `weight_load` → weight available for MAC | 1 |
| `result_request` → `result_out` valid | 1 |
| `accum_clear` → accumulator = 0 | 1 |
| `activation_in` → `act_out` | 0 (combinational) |
| Pipeline fill | 2 |

**RESEARCH FACT (verified against `pe.sv`)** — `weight_load` does **not**
suppress the MAC. The product and accumulator `always_ff` blocks
(`pe.sv:60-69`, `pe.sv:78-85`) are gated only by `rst`, `accum_clear`, and
`zero_skip` — **not** by `weight_load`. On a `weight_load=1` cycle:

1. the weight register captures `weight_in` (new weight visible next cycle);
2. the product register still computes `activation_in × weight` using the
   **old** weight (pre-edge, nonblocking semantics);
3. the accumulator still adds the prior cycle's product.

**DECISION (the skew)** — Because the weight path has one more register stage
(BREG) than the combinational activation path, the **weight stream leads the
activation stream by one cycle**. Concretely, during the tap cycle in which
`act_in` carries tap `t`'s activation, `w_in[r]` must carry tap `(t+1)`'s weight
so that the weight register holds tap `t`'s weight. The first tap's weight is
**preloaded** one cycle before the first tap's activation.

Equivalently (and interchangeably): the activation stream may be **delayed by
one cycle** relative to the weight stream. The controller must adopt exactly one
of these two conventions; §16 uses "weight leads by one cycle."

**IMPLIED** — With `weight_load` asserted every tap cycle, the array sustains
**one MAC/cycle/PE** after the 2-cycle pipeline fill. The skew is a constant
offset absorbed by the schedule; it does not reduce throughput.

---

## 10. Convolution / Tap Scheduling

### 10.1 Tap Serialization

**DECISION** — The K×K×IC kernel is serialized into `T = K·K·IC` taps in a fixed,
documented order. For Conv1 (K=5, IC=1): `T = 25` taps.

**DECISION** — Row-major serialization of the 2-D kernel: `t = K·k_y + k_x`
(`k_y = t / K`, `k_x = t mod K`). For Conv1 (K=5): `t=0 → (0,0)`, `t=4 → (4,0)`,
`t=5 → (0,1)`, …, `t=24 → (4,4)`. This fixed order is shared by the BRAM weight
controller and the input-feed unit so weight generation and activation feeding
stay aligned.

### 10.2 Per-PE Computation

**DECISION** — `PE(r,c)` accumulates, over the `T` tap cycles, the dot product
for output channel `r`, output pixel `(base - c)`:

```
accum(r,c) = Σ_{t=0}^{T-1} w_r[t] · a_pixel(r,c)[t]
```

where `w_r[t]` is the tap-`t` weight of channel `r`, and `a_pixel(r,c)[t]` is
the input activation the shift chain delivers to column `c` at tap `t`.

### 10.3 Input Feed Responsibility

**DECISION (boundary)** — Producing the correct `a_pixel(r,c)[t]` sequence from
the input feature map (line buffering, `im2col` vs. sliding window, the
`(W − K)` row jump) is the **input-feed unit's** job — a separate module with its
own specification, not part of `systolic_array.sv`. This spec defines only the
array's contract: `act_in` carries the current tap's activation each cycle.

### 10.4 Column → Pixel Index (reverse order)

**DECISION** — Because the shift chain delays each column by one cycle,
`PE(r,c)` accumulates the output pixel whose activations are shifted by `c`
relative to column 0. The result is a **reverse index**: column `c` produces
pixel `base − c`. This is a constant offset absorbed by the schedule (lead-in
activations); it must be reflected in the result-ordering convention (§11.3)
and in the golden model.

---

## 11. Accumulation and Result Collection

### 11.1 Accumulation

**DECISION** — Each PE accumulates locally in its own 32-bit signed accumulator
(Decision 2). There is **no cross-PE partial-sum path** in V1 (§3.1). The
32-bit width is mapping-independent and safe: `Σ|a_i·w_i|` over the 25 Conv1
taps is bounded far below 2³¹ (see `PE_SPEC.md` §7.5, generalized).

### 11.2 Result Exposure

**DECISION (from `PE_SPEC.md` §5.2, Decision 4)** — Results are exposed via the
per-PE `result_request`/`result_out` protocol: **read-without-clear**, with
`accum_clear` separate. `result_out` is registered (1-cycle capture).

### 11.3 Drain Scheme

**DECISION** — **Column-sequential drain** (per `PE_SPEC.md` §5.2 recommendation):
assert `result_req[c]` for one column at a time, capture that column's 8
`result_out` values (8 × 32-bit = 256-bit bus), and step through the 8 columns.
**8 cycles to drain the full array** (48 active results for Conv1).

**CANDIDATE (deferred)** — Parallel capture of all 64 results in one cycle
(2048-bit bus) would cut the drain to 1 cycle at the cost of wide routing. Not
adopted for V1 (do not over-optimize before it is needed).

**DECISION** — Result ordering: column `c`'s captured value corresponds to
output pixel `base − c` (§10.4). The drain order (column 0 → 7) and the
pixel-index map must match the golden model bit-for-bit.

---

## 12. Array-Level Control Signals

**DECISION** — The array is a **dumb datapath**. The top-level FSM, BRAM weight
controller, and input-feed unit drive these array-level controls (full port list
in §18):

| Signal | Purpose | V1 behavior |
|--------|---------|-------------|
| `clk`, `rst` | global clock and synchronous active-high reset | per `PE_SPEC.md` §11.8 |
| `act_in[0..7]` | per-row activation input | Conv1: all 8 driven identically |
| `w_in[0..7]` | per-row weight input (per tap) | weight-broadcast, §8 |
| `weight_load` | load the current `w_in` into all 64 PEs | asserted every tap cycle |
| `accum_clear` | clear all 64 accumulators + products | between output groups, §13 |
| `zero_skip` | gate MAC accumulation | **tied `0` in V1** (no sparsity) |
| `result_req[0..7]` | per-column result capture (one-hot) | asserted one column per drain cycle |
| `result_out[0..7]` | 8 × 32-bit results of the selected column | §11 |

**DECISION** — `zero_skip` is a single array-wide input tied `0` in V1. Per-PE
zero-detection is Team A V2 and is not present here.

---

## 13. Reset and Accumulator-Clear Sequencing

**DECISION** — Reset is **synchronous, active-high** (`rst`), fanning out to all
64 PEs. Reset clears every PE's accumulator, weight, and result registers
(`PE_SPEC.md` §11.8, `pe.sv:45-99`).

**DECISION** — `accum_clear` is a **single array-wide signal** that clears every
PE's accumulator **and** product register in one cycle (pipeline flush,
`PE_SPEC.md` §7.8/§10.14). It is asserted **once at the start of each output
group** — after the previous group's results have been fully drained and before
the next group's tap cycles begin.

**IMPLIED** — Because `accum_clear` flushes the product register, no multi-cycle
clear is needed, and the next group starts from a clean pipeline.

**DECISION (sequencing)** — For each output group, the canonical order is:

1. `accum_clear` (1 cycle),
2. weight preload of tap 0 (1 cycle),
3. `T` tap cycles (weight and activation streams, skew per §9),
4. one pipeline-drain cycle for the last product,
5. column-sequential `result_req` drain (8 cycles),
6. return to step 1 for the next group.

See §16 for the cycle-by-cycle table.

---

## 14. Boundary PE Behavior

**DECISION** — The PE has no boundary concept; all boundary behavior lives in the
array interconnect:

- **Column 0** — `activation_in` driven by `act_in[r]` (array input).
- **Column 7** — `act_out` unconnected (no downstream PE).
- **Idle rows** (Conv1 rows 6–7, since OC=6) — their `w_in` is tied to `0`,
  their activations shift (or are tied to `0`), and their `result_out` values
  are **ignored**. Their accumulators hold `0 × act = 0` and are never read.
- **Idle columns** (when the output-pixel count is not a multiple of 8) — the
  corresponding PE results are ignored; inputs are tied off so no `x` propagates.

**IMPLIED** — No floating inputs anywhere. Idle-row weights are driven to `0`
(not left undriven). This avoids `x`/`z` propagation into the signed MAC
datapath.

**DECISION** — Conv1 output is **24×24 valid (no padding)**, the standard
LeNet-5/MNIST convention (§15). Because 24 = 3 × 8, each output row is exactly
3 full output groups — **no partial tile**. The "idle columns" case above is
retained only for generality; Conv1 never triggers it.

---

## 15. LeNet-5 Conv1 Mapping (concrete V1 example)

**REQUIREMENT (roadmap §4)** — V1 must correctly compute LeNet-5 Conv1 on a test
input.

**DECISION** — Conv1 parameters used throughout this spec:

| Parameter | Value |
|-----------|-------|
| Input | 1 × 28 × 28 (grayscale) |
| Kernel | 5 × 5 |
| Output channels | 6 |
| Taps per output | 25 (5·5·1) |
| Output (no padding) | 6 × 24 × 24 (standard LeNet-5) |

**Mapping:**

- **Rows 0–5** = output channels 0–5; **rows 6–7 idle**.
- **Columns 0–7** = 8 adjacent output pixels of one output row `y`.
- One **output group** = 6 channels × 8 pixels = **48 results**, computed in 25
  tap cycles (plus overhead, §16).

**DECISION** — Conv1 uses **valid convolution (no padding)**: 28×28 input →
24×24 output. This matches the standard LeNet-5/MNIST reference
(`nn.Conv2d(1, 6, 5)` with no padding) and keeps `golden_model.py` aligned with
it. The 24×24 choice also makes each output row exactly 3 output groups
(24 = 3 × 8), eliminating partial-tile handling. The roadmap does not specify
padding, so this is a V1 design choice.

**Conv1 totals (24×24 output):**

| Quantity | Value |
|----------|-------|
| Output pixels per channel | 576 |
| Total outputs | 6 × 576 = 3,456 |
| Total MACs | 3,456 × 25 = 86,400 |
| Output groups | 3,456 ÷ 48 = 72 |
| Compute cycles (floor) | 86,400 ÷ 48 = 1,800 |

---

## 16. Cycle-Level Schedule

**DECISION** — The canonical schedule for **one output group** (48 results),
using the "weight leads activation by one cycle" convention (§9). `a_r[t]`
denotes tap `t`'s activation; `w_r[t]` denotes channel `r`'s tap-`t` weight.

| Cycle | `weight_load` | `w_in[r]` | `act_in[r]` | Notes |
|------:|:---:|:---:|:---:|:---|
| 0 | 0 | — | — | `accum_clear=1`: zero all accumulators + products |
| 1 | 1 | `w_r[0]` | 0 | weight preload (tap 0) |
| 2 | 1 | `w_r[1]` | `a_r[0]` | product ← `a_r[0]·w_r[0]` |
| 3 | 1 | `w_r[2]` | `a_r[1]` | product ← `a_r[1]·w_r[1]`; acc += `a_r[0]·w_r[0]` |
| … | 1 | `w_r[t+1]` | `a_r[t]` | product ← `a_r[t]·w_r[t]`; acc += `a_r[t−1]·w_r[t−1]` |
| 26 | 0 | — | `a_r[24]` | product ← `a_r[24]·w_r[24]` (last tap) |
| 27 | 0 | — | 0 | acc += `a_r[24]·w_r[24]` → **accumulator complete** |
| 27–34 | 0 | — | — | `result_req[0..7]` one column per cycle (drain) |

**Correctness note** — Every product uses the correct tap weight: because the
weight register lags `w_in` by one cycle, presenting `w_r[t+1]` while
`act_in` carries `a_r[t]` makes the product register see `w_r[t]` (loaded one
cycle earlier). All 25 terms accumulate into the pinned local accumulator with
no partial-sum movement.

**Throughput note** — Every tap cycle (cycles 2–26) performs one MAC per active
PE: **1 MAC/cycle/PE after the 2-cycle fill**, 48 active PEs for Conv1.

**Steady-state accounting (non-overlapped):**

| Component | Cycles |
|-----------|-------:|
| `accum_clear` | 1 |
| weight preload | 1 |
| tap cycles (25 MACs) | 25 |
| pipeline drain (last product) | 1 |
| result drain (8 columns) | 8 |
| **Total per group** | **36** |

**CANDIDATE (deferred)** — The next group's weight preload can overlap the
current group's result drain (they touch different registers). Not required for
V1 correctness; noted to avoid premature optimization.

---

## 17. Throughput and Latency Definitions

**DECISION** — The following quantities are defined for this spec:

| Term | Definition | V1 (Conv1) |
|------|-----------|-----------|
| **Peak MAC rate** | 64 PEs × 1 MAC/cycle | 64 MACs/cycle |
| **Active-PE MAC rate** | (active rows) × 1 MAC/cycle during tap cycles | 48 MACs/cycle (6 rows) |
| **Array utilization** | active rows ÷ 8 | 6/8 = 75% |
| **Group latency** | cycles to produce one output group | 36 cycles / 48 results |
| **Group compute efficiency** | tap cycles ÷ group latency | 25/36 ≈ 69% |
| **Pipeline latency** | `act_in` → first accumulator contribution | 2 cycles |
| **Conv1 total** | groups × group latency | 72 × 36 = 2,592 cycles |

**RESEARCH FACT / IMPLIED** — "64 MACs/cycle" (roadmap §3.2) is the array's
**peak capability**, not a per-layer sustained throughput requirement. Conv1 has
only 6 output channels, so a channel-per-row mapping sustains 48 MACs/cycle.

**DECISION** — "64 MACs/cycle" is the array's **peak** capability, not a
sustained per-layer requirement. For Conv1 the channel-per-row mapping sustains
48 MACs/cycle (6 rows) at 75% array utilization; rows 6–7 are idle and their
results are ignored (§14). Filling the idle rows would require splitting the
pixel or tap dimension across rows — deferred (not needed for V1 correctness).

---

## 18. Array Interface

**DECISION** — The `systolic_array.sv` port list (signal concepts; exact widths
follow the PE contract). Directions are relative to the array module.

| Signal | Direction | Width | Status | Source |
|--------|-----------|-------|--------|--------|
| `clk` | input | 1 | **IMPLIED** | synchronous design |
| `rst` | input | 1 | **DECISION** (Decision 5) | synchronous, active-high |
| `act_in[0..7]` | input | 8 × 8b signed | **REQUIREMENT** (R1/R8) | per-row activation |
| `w_in[0..7]` | input | 8 × 8b signed | **REQUIREMENT** (R2) | per-row weight, per tap |
| `weight_load` | input | 1 | **DECISION** | array-wide weight load |
| `accum_clear` | input | 1 | **DECISION** (Decision 4) | array-wide pipeline flush |
| `zero_skip` | input | 1 | **REQUIREMENT** (R7, port); tied `0` in V1 | array-wide |
| `result_req[0..7]` | input | 8 × 1b | **DECISION** (Decision 4) | per-column result capture |
| `result_out[0..7]` | output | 8 × 32b signed | **REQUIREMENT** (R5/R6) | selected column's results |

### 18.1 Notes

- **`act_in`, `w_in`:** signed 8-bit two's-complement (Decision 1). For Conv1,
  all 8 `act_in` lines carry the same value each cycle (§7); `w_in[r]` is
  channel `r`'s current-tap weight (§8).
- **`weight_load`:** asserted every tap cycle; also during the tap-0 preload
  cycle. Deasserted during `accum_clear` and result drain.
- **`result_req`:** one-hot per column. When `result_req[c]=1`, column `c`'s 8
  PEs capture their accumulators into `result_out[0..7]` on the next cycle.
- **`result_out`:** registered (PE `result_out` is registered). Values hold
  while `result_req[c]` remains asserted (level-sensitive, read-without-clear).
- **`zero_skip`:** tied `0` in V1. The port exists for V2 Team A; it is not
  used by V1.
- **No AXI, no valid/ready, no flow control** in the array — the FSM/controller
  times everything globally (per `PE_SPEC.md` §5.2).

---

## 19. Verification Requirements

**REQUIREMENT (roadmap §4)** — Array-level integration testbench
(`tb_v1_integration.sv`) and `golden_model.py` must confirm the array matches
the golden convolution.

**DECISION** — The array testbench (`tb_systolic_array.sv`, or the roadmap's
`tb_v1_integration.sv`) must cover, at minimum:

1. **Reset / clear at scale** — `rst` and `accum_clear` zero all 64 accumulators
   and product registers; no stale add-back across groups.
2. **Weight broadcast** — `w_in[r]` reaches all 8 PEs of row `r`; a reloaded
   weight takes effect one cycle later (skew per §9).
3. **Activation shift** — a value at `act_in[r]` appears at column `c` exactly
   `c` cycles later (1-cycle register per hop).
4. **One-cycle skew** — weight leads activation by one cycle; verify the product
   uses the correct tap weight, not the previous one.
5. **Full Conv1 tile vs. golden** — a complete output group matches
   `golden_model.py` bit-exact (signed 8-bit, 32-bit accumulator,
   read-without-clear, reverse column index).
6. **Drain ordering** — `result_req` captures the correct column, and the
   (column → pixel) map matches the golden model.
7. **Boundary behavior** — column 7 `act_out` unconnected without `x`
   propagation; idle rows (6–7) ignored; no floating inputs.
8. **Pipeline fill** — first accumulator contribution 2 cycles after `act_in`;
   steady-state 1 MAC/PE/cycle during tap cycles.
9. **`zero_skip` tied low** — dense baseline matches the no-skip result.
10. **Signed/corner arithmetic at scale** — the `tb_pe` corner cases
    (min-negative, mixed sign) hold across the array.

**DECISION** — The golden model must implement the exact array arithmetic:
25-tap dot product per PE, signed 8-bit operands, 32-bit accumulation, the
reverse column index, and read-without-clear. Any mismatch is a bug in either
the RTL or the model.

---

## 20. Open Decisions and Assumptions

### 20.1 Dataflow Reclassification Record (Decision 7)

**DECISION (recorded)** — The Output-Stationary dataflow (§3.1) is recorded in
`docs/PROJECT_STATE.md` as **Decision 7 — V1 Dataflow Reclassification**. The
technical decision is settled; the only remaining item is **roadmap-owner +
Team B sign-off** (V1 ships the dataflow the roadmap reserved for Team B V2).

### 20.2 Resolved Scheduling Decisions (Decision 8)

The five scheduling decisions are finalized (recorded in
`docs/PROJECT_STATE.md` as **Decision 8 — V1 Array Scheduling Decisions**) and
no longer block the array:

| # | Decision | Resolution | Reference |
|---|----------|-----------|-----------|
| 1 | Conv1 output dimension / padding | 24×24 valid, no padding (24 = 3×8 → no partial tile) | §14, §15 |
| 2 | Weight/activation skew convention | weight stream leads activation by one cycle | §9, §16 |
| 3 | Result drain | column-sequential, 8 cycles | §11.3 |
| 4 | Tap serialization order | row-major, `t = K·k_y + k_x` | §10.1 |
| 5 | "64 MACs/cycle" | peak capability; Conv1 sustains 48 (75%), rows 6–7 idle | §17 |

**Remaining items (not blocking `systolic_array.sv`):**

- **Decision 7 sign-off** — roadmap-owner + Team B approval of the dataflow
  reclassification (governance; the technical content is settled).
- **Input-feed / line-buffer module** — a separate module with its own spec,
  now unblocked; its array interface is defined (§10.3, §18).

### 20.3 Non-Blocking / Out-of-Scope

| Item | Disposition |
|------|-------------|
| Input-feed / line-buffer unit internals | Separate module; only its array interface is defined (§10.3) |
| BRAM weight controller, top FSM, input buffer | Separate modules; referenced, not specified here |
| Partial-sum cascade (`psum_in`/`psum_out`) | V2 Team B (PE v1.1) — explicitly out of V1 |
| Sparsity (`zero_skip` functional connection) | V2 Team A — tied `0` in V1 |
| Clock frequency target | Deferred to synthesis (`PE_SPEC.md` §11.6) |
| Conv2 / FC / general layer mapping | Out of V1 scope (§5) |

---

## Appendix A: References

| Reference | Description |
|-----------|-------------|
| `docs/reference/architecture_major_project.pdf` | Architecture roadmap v3.0 (§3.2, §4) |
| `docs/specs/PE_SPEC.md` | PE contract (§3.1, §5, §6, §7.8, §8.4, §10) |
| `docs/PROJECT_STATE.md` | Decisions 1–8 (PE contract + V1 dataflow + array scheduling) |
| `rtl/common/pe.sv` | Implemented, verified PE (fixed port list) |
| `sim/tb_pe.sv` | PE unit testbench (55/55 PASS) |
| DS925 / UG579 | XCK26 / DSP48E2 references (`PE_SPEC.md` §4) |
| Sze–Chen–Emer dataflow taxonomy | Weight-/Output-/Row-Stationary classification (`PE_SPEC.md` §4.8) |

---

> **Next step:** Implement `rtl/common/systolic_array.sv` per this contract and
> the `pe.sv` interface. The input-feed/line-buffer module is the next separate
> module to specify.
