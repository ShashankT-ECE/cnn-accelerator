# Phase 2 — Research Accelerator Architecture

> **Status:** IMPLEMENTED AND VERIFIED (2026-08-21). This records the Phase 2
> research accelerator (`rtl/common/cnn_accelerator_v2.sv`) that executes the
> real MNIST-12 Conv1 layer with runtime OS/WS reconfiguration, SAME padding,
> OC=8, and a genuine coarse zero-skip. Simulation is bit-exact against the
> independent Phase-1 golden model; synthesis/implementation results are in
> `docs/PHASE2_SYNTHESIS.md`.

---

## 1. Scope and deliverable

Phase 2 builds the complete research accelerator around the **frozen** V2
datapath (`rtl/common/systolic_array_v2.sv` + `rtl/common/pe_v2.sv`). No frozen
module was modified. All Phase 2 logic is new integration logic in one
synthesizable top-level wrapper:

| File | Role |
|---|---|
| `rtl/common/cnn_accelerator_v2.sv` | top-level research accelerator (new) |
| `rtl/common/systolic_array_v2.sv` | FROZEN 8×8 array (64× `pe_v2`) |
| `rtl/common/pe_v2.sv` | FROZEN reconfigurable PE |
| `sim/tb_cnn_accelerator_v2.sv` | self-checking integration testbench (new) |

The target layer is fixed (`docs/PHASE1_EXPERIMENT_SPEC.md`):

| Parameter | Value |
|---|---|
| Input | 28×28×1, signed INT8 (`S_a = 1/127`, no normalization) |
| Kernel | 5×5, 8 output channels, 200 weights |
| Padding | SAME_UPPER, pad = 2 |
| Output | 28×28×8 = 6,272 results |
| MACs | 156,800 |

---

## 2. Top-level interface

```
cnn_accelerator_v2 (
  clk, rst,
  dataflow_mode, mode_commit, start, busy, done, mode_active, mode_error, switching,
  img_wr_addr[9:0], img_wr_data[7:0], img_wr_en,        // image write port
  wgt_wr_addr[7:0], wgt_wr_data[7:0], wgt_wr_en,        // weight write port
  result_valid, result_data[0:7][31:0], result_base[12:0], result_last,
  total_macs, executed_macs, skipped_macs, zero_skip_cycles, cycle_count
)
```

The image (784 B) and weights (200 B) are written through simple write ports
(populated before `start`); the PS/AXI transport is deferred to Phase 3. Results
are emitted one **word** per `result_valid` pulse — 8 × 32-bit pixels plus a
`result_base` flat index (the group's leftmost pixel, `ch*784 + y*28 + (base-7)`).

---

## 3. Dual-mode controller (OS and WS on one datapath)

The same `systolic_array_v2` is driven in two dataflows selected by
`dataflow_mode` (0 = Output-Stationary, 1 = Weight-Stationary). No bitstream
reload, no FPGA reprogramming.

### 3.1 OS mode (`dataflow_mode = 0`)

V1-identical PE behaviour (local accumulation; `psum_in` ignored). **Rows =
output channels** (all 8 used), **columns = 8 output pixels** (reverse index).
The 5×5 conv is computed as five 1-D horizontal passes (one per kernel row),
mirroring Decision 9's row-decomposed schedule:

- 112 groups = 28 output rows × 4 groups (`base = 7, 15, 23, 31`).
- Per group: 1 `accum_clear` + 5 passes × 13 cycles (7 lead-in + 5 taps + 1
  drain) + 16-cycle per-row result drain = **82 cycles/group**.
- Total = **9,184 cycles**; sustained 17.07 MAC/cycle (26.7% of 64 peak).

### 3.2 WS mode (`dataflow_mode = 1`)

True spatial Weight-Stationary (Decision 10/11): **rows = kernel tap** (8 per
tile), **columns = 8 output pixels**, **channels serialized** (8 sweeps).

- 896 groups = 8 channels × 112 groups.
- Per group: 4 tiles (8+8+8+1) over 42 compute cycles + 1 result-capture cycle
  = **43 cycles/group**.
- Total = **38,528 cycles**; sustained 4.07 MAC/cycle (6.4% of 64 peak).

### 3.3 Runtime reconfiguration sequence

```
IDLE → configure mode (dataflow_mode + mode_commit) → FLUSH (8 cycles) → IDLE
     → load weights/data → start → execute → done → (repeat)
```

- `mode_commit` is honoured only in IDLE; a commit while busy is **rejected**
  (`mode_error` pulse), and `mode_active` never changes mid-run.
- FLUSH = 8 cycles: cycle 0 asserts `accum_clear` (clears all 64 accumulators +
  products) and cycles 0–7 drive `act_out = 0` (drains the 56-register shift
  chain). No `rst` — image/weight stores are preserved.
- `mode_active` (latched) is the only signal the array sees, so a mid-run write
  can never glitch the datapath.
- Verified OS→WS and WS→OS transitions are **bit-exact** with no reset.

---

## 4. SAME padding (zero injection)

The input is zero-padded to 32×32 (pad 2), equivalently every out-of-range
activation read injects **exact 0** into the MAC stream. The stream address
generation applies a `-2` offset to both the row (`y + ky − 2`) and column
(`base − 9 + j` in OS; `base − 10 + s − 5·ky` in WS), and a `row_ok`/`col_ok`
predicate gates the value to 0 when out of `[0,27]`. This is the Phase-1
"latent padding bug" fix: the previous controller **clamped** OOB columns to
edge pixels; this controller emits 0.

Directed tests cover top/bottom/left/right edges, all four corners, and centre
against an independent padded-convolution golden.

---

## 5. OC = 8 and 28×28 mapping

- **OS**: 8 channels on the 8 PE rows (no idle rows — V1 used only 6). The last
  group of each row (`base = 31`) is partial: 4 valid pixels (24–27), 4
  discarded (28–31).
- **WS**: 8 channel sweeps. The partial group is handled by the same validity
  gating (4 valid columns).
- The result base index encodes the group's **leftmost** pixel (`base − 7`) so
  the flat index never overflows into the next row (unambiguous for `base=31`).

---

## 6. Sparsity — genuine coarse zero-skip

Measured on the real MNIST vector: **85.43%** of the 156,800 MACs have a zero
activation operand (the `S_a = 1/127` no-normalization quantisation maps the
background to `q_a = 0`). Two mechanisms are implemented and distinguished:

1. **Fine-grained `zero_skip` port — tied to 0.** The frozen array's `zero_skip`
   is array-wide and gates the *product* register, which samples the *shifted*
   activation (`act_delayed`), not the fed stream. Gating it on the fed stream
   (`act_out == 0`) zeroes legitimate in-flight products and corrupts results
   (verified: sparse MNIST fails bit-exact). Per-MAC zero-skipping is therefore
   unsafe on the frozen array; it is documented as a limitation, not silently
   used.

2. **Coarse zero-group skip — the genuine speedup.** A whole output group whose
   5×12 activation window (rows `y−2..y+2`, cols `base−9..base+2`) is entirely
   zero has all-zero outputs for every channel, so its compute is skipped.
   Detection uses a 784-bit non-zero mask maintained during image write (not 60
   combinational RAM reads). 60 of 112 output positions are all-zero on MNIST,
   so WS drops from 38,528 to **18,368 cycles (2.10× speedup)**.

**Honest accounting.** The skip is *structured and coarse* (whole group), not
fine-grained compaction — it preserves the fixed systolic timing and pixel
identity/order. The 85.4% zero-activation rate is the *capability*; the 2.10×
WS speedup is the *measured* result. No speedup is claimed for OS mode, and no
cycle saving is claimed for the `zero_skip` gate itself.

### 6.1 Counters

| Counter | Meaning | MNIST value |
|---|---|---|
| `total_macs` | layer MAC opportunities | 156,800 |
| `skipped_macs` | zero-activation MACs | 133,960 |
| `executed_macs` | non-zero-activation MACs | 22,840 |
| `cycle_count` | per-run cycles (start→done) | mode-dependent |
| `zero_skip_cycles` | cycles `zero_skip` asserted | 0 (tied off) |

`skipped_macs` is counted exactly (mode-independent) and cross-checked against
an independent TB computation (8 × count of zero activations over
`(y,x,ky,kx)`).

---

## 7. Verification

`sim/tb_cnn_accelerator_v2.sv` loads the **real** Phase-1 vectors
(`data/vectors/weights.hex`, `input_img.hex`, `golden_canonical.hex`) and checks:

1. OS dense == golden (6,272/6,272 bit-exact).
2. WS dense == golden.
3. OS → WS and WS → OS transitions (no reset).
4. SAME-padding boundaries (all-ones / corner-pixel / gradient directed vectors
   against an independent padded-conv golden).
5. All 8 channels, all 28×28 positions, result word/pixel counts (896 words /
   6,272 pixels, no duplicate/missing via a seen bitmap).
6. Sparsity counters vs independent count.
7. Mode interlock (commit-while-busy rejected; in-flight results unchanged).

Result: **75,305/75,305 checks PASS, 0 FAIL**.

The primary golden (`golden_canonical.hex`) is produced by the independent
Python integer model (`python/reference/int8_ref.py`) — never by this TB or by
the schedule. The directed boundary tests use a separate padded-convolution
function written from the layer's definition.

---

## 8. Reproducibility

```bash
source ~/Xilinx/Vivado/2023.1/settings64.sh

# Simulation (bit-exact vs golden)
xvlog -sv rtl/common/pe_v2.sv rtl/common/systolic_array_v2.sv \
       rtl/common/cnn_accelerator_v2.sv sim/tb_cnn_accelerator_v2.sv
xelab -debug typical -s tb_cnn_acc_snapshot tb_cnn_accelerator_v2
xsim tb_cnn_acc_snapshot -runall          # expect RESULT: PASS

# Synthesis + implementation (200 MHz)
vivado -mode batch -source build/phase2/synth_impl.tcl

# Measurements
.venv/bin/python python/run_phase2_measure.py   # -> data/vectors/phase2_results.json
```

---

## 9. Limitations and open items

1. **`zero_skip` per-MAC gating is unsafe** on the frozen array (array-wide,
   gates the shifted product) — documented, tied to 0.
2. **Coarse skip is WS-only.** OS groups are channel-parallel (8 words/group),
   so the same skip would need 8-word emission; left as a documented extension.
3. **Fine-grained activation compaction** (drop zeros, re-align) is explicitly
   rejected — it would break the fixed systolic timing.
4. **Partial group (base=31)** wastes 4 of 8 columns (inherent to 28 = 3×8+4).
5. **PS/AXI transport** (image/weight/result) is deferred to Phase 3; the
   wrapper exposes write/read ports suitable for AXI4-Lite wrapping.
6. **Bias add + dequantisation** remain in PS software (unchanged from Phase 1).

---

## 10. Files created/modified

- `rtl/common/cnn_accelerator_v2.sv` — new (Phase 2 accelerator).
- `sim/tb_cnn_accelerator_v2.sv` — new (integration testbench).
- `data/vectors/golden_canonical.hex` — new (hex golden for `$readmemh`).
- `data/vectors/phase2_results.json` — new (machine-readable measurements).
- `python/run_phase2_measure.py` — new (measurement generator).
- `build/phase2/synth_impl.tcl`, `baseline.xdc` — new (synthesis flow).
- `docs/PHASE2_ARCHITECTURE.md`, `docs/PHASE2_SYNTHESIS.md` — new.

No frozen file (`pe.sv`, `systolic_array.sv`, `input_feed.sv`, `pe_v2.sv`,
`systolic_array_v2.sv`, any V1 testbench) was modified.
