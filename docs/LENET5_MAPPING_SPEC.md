# LeNet-5 Hardware Mapping Contract (Step 3.1)

> **Status: FROZEN (2026-09-23).** This is the authoritative mapping contract
> between the five learned LeNet-5 layers (`docs/LENET5_SPEC.md`) and the frozen
> 8×8 reconfigurable systolic array (`rtl/common/systolic_array_v2.sv` +
> `pe_v2.sv`). It defines *what* maps where and *what the array can and cannot
> do*; it does **not** fix cycle counts (Step 3.2) or implement OS/WS (Step 3.3).
> No RTL is modified by this document.

---

## 1. Frozen datapath summary (the array being mapped)

| Property | Value |
|---|---|
| Topology | 8×8 grid, 64× `pe_v2` (int8×int8 → int32 MAC, DSP48E2) |
| OS mode (`dataflow_mode=0`) | rows = **output channels** (8), columns = **8 output pixels** (reverse index `base−c`), local per-PE accumulator, activation shift chain, weight broadcast per row |
| WS mode (`dataflow_mode=1`) | rows = **8 reduction taps/tile**, columns = **8 output pixels**, vertical `psum` cascade (sums rows) + bottom-to-top tile feedback, output channels serialized in time |
| Weight delivery | `w_in[r]` broadcast to all 8 columns of row `r` (pixel-invariant); single array-wide `weight_load` |
| Activation delivery | 8 per-row `act_in[r]` streams, per-row horizontal shift chain |
| Result | OS: per-column drain (16 cycles); WS: per-row drain (read row 7, 1 cycle) |
| Precision | int8 activations/weights, int32 accumulator; bias/requantize on PS |

**Invariant that drives everything below:** the array's weight is **per-row
broadcast** (identical across all 8 columns) and its activation is **per-column
shifted** (a spatial stream). This means the column axis is always a *spatial
pixel* axis, and the weight can only vary along the *row* axis. Convolutions fit
this; fully-connected layers do **not** (their weights vary along the output
axis, i.e. the column axis) — see §3.4 and §3.5.

## 2. Common conventions (all layers)

- **Tensor layout (INT8, from `docs/LENET5_INT8_SPEC.md`):** activations
  `(IC, H, W)` flat `ic·H·W + y·W + x`; weights `(OC, IC, K, K)` flat
  `oc·(IC·K·K) + ic·(K·K) + ky·K + kx` (ch-major); outputs `(OC, H', W')`
  flat `oc·H'·W' + y·W' + x`.
- **Padding:** every layer is **VALID** (no padding) — output `H' = H − K + 1`.
  No zero-injection is required for the LeNet-5 workload (the Phase-2 SAME-pad
  logic is bypassed, not exercised).
- **Precision:** int8 activations and weights; int32 accumulator; int32 bias and
  requantization are PS-side (`docs/LENET5_INT8_SPEC.md` §7–§9). ReLU and pooling
  are PS-side and are **not** part of the array mapping.
- **Reduction depth** `N = K·K·IC` per output: Conv1 = 25, Conv3 = 150,
  Conv5 = 400, FC1 = 120, FC2 = 84. All `≤ 400`, so `|acc| ≤ 400·127² ≈ 6.45 M`,
  comfortably within int32.

## 3. Per-layer mapping

### 3.1 Conv1 — 1→6, 5×5 valid, 32×32 → 28×28

| Element | OS (`dataflow_mode=0`) | WS (`dataflow_mode=1`) |
|---|---|---|
| Tensor layout | in (1,32,32); w (6,1,5,5); out (6,28,28) | same |
| Tiling | rows 0–5 = 6 channels (**rows 6–7 idle**); cols = 8 pixels; 28 rows × 4 groups (last group 4 valid) | rows = 8 taps; **25 taps = 4 tiles** (8+8+8+1); cols = 8 pixels; **6 channel sweeps** |
| PE utilization | 6/8 rows × 8/8 cols = 75% (partial group lowers it) | 8/8 rows × 8/8 cols; 7/32 row-slots idle (tile 3) |
| Accumulator ownership | per-PE (local, `acc += product`) | bottom row (row 7) holds running sum |
| Weight movement | broadcast per row, reloaded per tap (5 passes) | held per row, reloaded per tile (cycles 0/15/23/31) |
| Input movement | one shared stream, shifts across cols | 8 per-row streams (diagonal skew, §SYSTOLIC_ARRAY_V2_SPEC §9.5) |
| Partial sums | none | vertical cascade + bottom-to-top feedback |
| Padding/boundary | VALID, no OOB | VALID, no OOB |
| OS constraint | OC=6 ≤ 8 → single OS group | — |
| WS constraint | — | 4-tile decomposition (25 = 3·8 + 1) |

**Notes:** closest to the existing Phase-2 mapping (MNIST-12 Conv1). Deltas vs
Phase 2: OC=6 (not 8), **32×32 valid input** (not 28×28 SAME), so the controller
re-parameterizes but the datapath is unchanged.

### 3.2 Conv3 — 6→16, 5×5 valid, 14×14 → 10×10

| Element | OS | WS |
|---|---|---|
| Tensor layout | in (6,14,14); w (16,6,5,5); out (16,10,10) | same |
| Tiling | rows = 8 of 16 channels → **2 OS groups**; cols = 8 pixels; 10 rows × 2 groups (last group 2 valid); **IC=6 → 30 passes** (6 IC × 5) | rows = 8 taps; **150 taps = 19 tiles** (18·8 + 6); **16 channel sweeps** |
| PE utilization | 8/8 rows; 2/8 cols on partial group | 8/8 rows × 8/8 cols; 2/152 row-slots idle (tile 18 partial) |
| Accumulator ownership | per-PE | row 7 |
| Weight movement | per-row per-tap; **4-D decode `W[oc][ic][ky][kx]`** | per-row per-tile; same 4-D decode |
| Input movement | shared stream per IC pass (6 channel-major passes) | per-row streams; **IC channel buffering** |
| Partial sums | none (acc sums 150 taps in time) | vertical cascade + feedback (150 taps / 19 tiles) |
| Padding | VALID | VALID |
| OS constraint | IC>1 → input must be re-fed per channel; OC=16 → 2 groups | — |
| WS constraint | — | IC>1 → tile count scales with IC (19 tiles) |

**Notes:** first IC>1 layer. Requires input-channel buffering and variable tile
count (§4 gap G1).

### 3.3 Conv5 — 16→120, 5×5 valid, 5×5 → 1×1

| Element | OS | WS |
|---|---|---|
| Tensor layout | in (16,5,5); w (120,16,5,5); out (120,1,1) | same |
| Tiling | rows = 8 of 120 → **15 OS groups**; cols = 8 (**1 valid pixel, 7 idle**); **400 taps** (16 IC × 25) | rows = 8 taps; **400 taps = 50 tiles**; cols = 1 valid pixel; **120 sweeps** |
| PE utilization | 1/8 columns (degenerate 1×1 output) | 1/8 columns |
| Accumulator ownership | per-PE | row 7 |
| Weight movement | per-row per-tap (4-D) | per-row per-tile |
| Input movement | shared per-IC stream | per-row streams (IC buffering) |
| Partial sums | none | cascade + feedback |
| Padding | VALID | VALID |

**Notes:** this is a **degenerate convolution** (`5×5` on `5×5 → 1×1`): a single
output pixel, mathematically a fully-connected layer of 400 → 120. Both modes
waste 7/8 columns (§4 gap G4).

### 3.4 FC1 — 120→84 (mapped as a 1×1 conv)

| Element | OS | WS |
|---|---|---|
| Tensor layout | in (120,) → (120,1,1); w (84,120) → (84,120,1,1); out (84,) | same |
| Tiling | rows = 8 of 84 → **11 OS groups**; cols = **1 valid pixel**; **120 taps** (IC serialized) | rows = 8 taps; **120 taps = 15 tiles**; cols = 1 valid pixel; **84 sweeps** |
| PE utilization | 1/8 columns | 1/8 columns |
| Accumulator ownership | per-PE | row 7 |
| Weight movement | per-row per-tap | per-row per-tile |
| Input movement | shared scalar stream | per-row scalar streams |
| Partial sums | none | cascade + feedback |
| Padding | n/a (1×1) | n/a |

### 3.5 FC2 — 84→10 (mapped as a 1×1 conv)

| Element | OS | WS |
|---|---|---|
| Tensor layout | in (84,) → (84,1,1); w (10,84) → (10,84,1,1); out (10,) | same |
| Tiling | rows = 8 of 10 → **2 OS groups**; cols = 1; **84 taps** | rows = 8 taps; **84 taps = 11 tiles**; cols = 1; **10 sweeps** |
| PE utilization | 1/8 columns | 1/8 columns |
| Accumulator ownership | per-PE | row 7 |
| Weight movement | per-row per-tap | per-row per-tile |
| Input movement | shared scalar stream | per-row scalar streams |
| Partial sums | none | cascade + feedback |
| Padding | n/a | n/a |

**FC → WS caveat (important):** the array's weight is **per-row broadcast** and
its columns are a *spatial* (shifted) axis. A fully-connected layer needs
per-*output-column* weights (`W[oc][ic]`), which the array cannot supply in one
pass. Consequently:

- **OS maps FC directly** — rows = output neurons (8 in parallel), input features
  serialized in time. This is the natural FC mapping.
- **WS does not map FC to 8 distinct output neurons.** With rows = input features,
  all 8 columns compute the *same* dot product (weight is pixel-invariant and the
  input is a scalar), so WS can only produce **one output neuron per sweep**
  (7/8 columns idle, output serialized).

This **supersedes** the earlier `RESEARCH_READINESS_PLAN.md` §6 assumption that
"WS wins on FC" — that assumption did not account for the per-row weight-broadcast
constraint. The OS-vs-WS comparison on FC (and whether a crossover exists) is
re-measured in Step 3.2; this contract records only that **FC is OS-native and
WS-inefficient**, and that **no conv→FC crossover is guaranteed** by the array.

## 4. RTL gaps (what the current controller cannot do)

The frozen datapath (`pe_v2`, `systolic_array_v2`) needs **no change**. All gaps
are controller/input-feed level. The current `cnn_accelerator_v2.sv` is
specialized to MNIST-12 Conv1 (IC=1, OC=8, 5×5, SAME, 28×28).

| # | Gap | Affects | Severity |
|---|---|---|---|
| G1 | **IC > 1** (input-channel buffering + variable tile count + 4-D weight decode) | Conv3, Conv5, FC1, FC2 | **blocking** — the largest gap |
| G2 | **32×32 input** (controller hardcodes 28×28) | Conv1 | blocking (re-parameterization) |
| G3 | **OC > 8** (arbitrary output-channel tiling: 15 groups / 120 sweeps) | Conv5, FC1, Conv3 | blocking |
| G4 | **1-pixel layers** (7/8 columns idle) — inherent, not a fixable gap | Conv5, FC1, FC2 | utilization note (documented) |
| G5 | **Weight-store capacity** — ~61,706 int8 weights (~60 KB) vs the current ~1.2 KB LUTROM | all | blocking (needs BRAM; K26 has ~648 KB BRAM) |
| G6 | **VALID-conv boundary** (controller has SAME-pad zero-injection; valid needs re-parameterized row/col predicates) | Conv1, Conv3, Conv5 | moderate (simpler than SAME) |
| G7 | **FC weight transpose for OS** — OS-FC needs `W[oc][ic]` indexed by output-neuron-row and input-feature-tap (a 4-D-to-2-D reshape, not a transpose) | FC1, FC2 | low |
| G8 | **Result-address encoding** — current `result_base` assumes `ch*784 + y*28 + x`; must generalize to per-layer `(H', W')` and OC | all | moderate |

**No datapath limitation is found for the conv layers.** The array computes every
convolution (OS and WS) correctly; the gaps are controller scope (IC/OC/input
generality) and storage (weight BRAM). The one *architectural* constraint is the
FC-in-WS limitation (§3.4/§3.5), which is inherent to the per-row weight
broadcast and is documented as a property, not a bug.

## 5. Deliberately deferred

- Exact OS/WS cycle counts per layer (Step 3.2).
- The OS/WS reconfiguration sequence across layers and the "selected mode"
  decision (Step 3.3).
- Weight-store (BRAM) sizing and layout, and the IC-buffering line-buffer sizing
  (implementation).
- Zero-skipping on the LeNet-5 layers (Step 4).
