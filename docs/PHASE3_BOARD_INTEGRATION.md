# Phase 3 — KV260 Board Integration (AXI / control / data movement)

> **Status:** IMPLEMENTED + VERIFIED IN SIMULATION (2026-08-21). Only physical
> measurement and bitstream generation remain BOARD-ONLY.
>
> This documents the board-facing architecture that wraps the verified Phase 2
> reconfigurable accelerator (`cnn_accelerator_v2`) behind an AXI4-Lite slave so
> a host CPU can load the real MNIST-12 Conv1 workload, run it in OS/WS modes at
> runtime, and read back results. No frozen module (`pe.sv`, `systolic_array.sv`,
> `input_feed.sv`, `pe_v2.sv`, `systolic_array_v2.sv`) was modified.

## 1. What was added

| File | Role |
|---|---|
| `rtl/common/cnn_axi_ctrl.sv` | AXI4-Lite slave + register file + result BRAM; instantiates the accelerator |
| `rtl/top/cnn_top.sv` | thin top: `aresetn`→`rst` conversion + re-expose AXI slave + IRQ |
| `sim/tb_cnn_top_axi.sv` | AXI4-Lite bus-functional-model testbench (bit-exact vs golden) |
| `software/cnn_host.c` | PS userspace driver/benchmark (UIO), cross-compilable |
| `scripts/board/*` | device-tree overlay, block-design template, firmware shell template |
| `scripts/run_verification.sh` | single PASS/FAIL gate (both regressions + Python checks) |

## 2. Architecture

```
PS (A53, Ubuntu 22.04)  M_AXI_HPM0_FPD @0xA400_0000   pl_clk0=100MHz   pl_ps_irq0
        │                                                        │
        ▼                 AXI SmartConnect                       ▼
   cnn_top (AXI4-Lite slave)  ◄──  Clocking Wizard: 100 → 200 MHz
        │  rst = ~aresetn  ◄──────── proc_sys_reset (peripheral_aresetn)
        ▼
   cnn_axi_ctrl ──► cnn_accelerator_v2 ──► systolic_array_v2 (64×pe_v2, FROZEN)
        │                 │
        ▼                 ▼
   8×result BRAM      image dist-RAM + weight regfile
```

- **One clock domain** (`s_axi_aclk` = core clock). The PS `pl_clk0` (100 MHz)
  is multiplied to the core frequency by the block-design Clocking Wizard; the
  AXI SmartConnect handles the PS↔PL clock conversion.
- **Reset conversion:** AXI `aresetn` (active-low) is inverted to the active-high
  synchronous `rst` used throughout the accelerator (Decision 5). `proc_sys_reset`
  provides the already-synchronized reset; no extra synchronizer is needed.
- **No DMA, no DDR data buffers, no cache-coherency surface** — all data lives in
  PL BRAM/distributed-RAM and is written/read through AXI4-Lite. The workload is
  784 B in, 200 B weights, 25 KB results; AXI DMA (PG021) adds a driver and
  coherency burden for no benefit (per the research plan §8).

## 3. Register map (32-bit, byte offsets)

| Offset | Name | Access | Meaning |
|---|---|---|---|
| 0x0000 | VERSION | RO | 0x0002_0001 |
| 0x0004 | CONTROL | WO | [0] START, [1] SOFT_RESET, [2] MODE_COMMIT |
| 0x0008 | STATUS | RO | [0] IDLE, [1] BUSY, [2] DONE, [3] ERROR, [4] SWITCHING |
| 0x000C | DATAFLOW_MODE | RW | [0] 0=OS, 1=WS |
| 0x0010 | MODE_STATUS | RO | [0] ACTIVE_MODE |
| 0x0014 | SPARSITY_DISABLE | RW | [0] 1 = force WS dense (disable coarse skip) |
| 0x0018 | CYCLE_RUN | RO | per-run cycles (START→DONE, frozen at DONE) |
| 0x001C | TOTAL_MACS | RO | 156,800 |
| 0x0020 | EXECUTED_MACS | RO | non-zero-activation MACs |
| 0x0024 | SKIPPED_MACS | RO | zero-activation MACs |
| 0x0028 | ZERO_SKIP_CYCLES | RO | cycles the (tied-0) gate is asserted |
| 0x002C | IMAGE_ADDR | RW | next image write address (0..783) |
| 0x0030 | IMAGE_DATA | WO | write img[addr], auto-increment |
| 0x0034 | WEIGHT_ADDR | RW | next weight write address (0..199) |
| 0x0038 | WEIGHT_DATA | WO | write wgt[addr], auto-increment |
| 0x0040 | IRQ_ENABLE | RW | [0] DONE IRQ enable |
| 0x0044 | IRQ_STATUS | RO/W1C | [0] DONE pending |
| 0x1000..0x8FFF | RESULT | RO | 8 banks × 1024 × 32-bit result window |

**`CYCLE_RUN` is the paper's latency instrument** — a hardware per-run counter,
so measured per-mode latency carries no software timing noise. Wall-clock (AXI
load/poll/readback) is reported separately and never folded into it.

## 4. Result layout (canonical flat order)

The accelerator emits one result word (8 pixels + `result_base`) per cycle. Pixel
`c` maps to the canonical flat index `result_base + 7 - c` (i.e. `ch*784+y*28+x`),
which is stored in bank `x & 7` at dense address `flat >> 3`. 8 banks give 8
parallel single-write ports (one per column residue), so the 1-word/cycle
emission rate needs **no FIFO and no backpressure**.

Host readback (see `software/cnn_host.c`):

```
for ch, y, x:
    flat = ch*784 + y*28 + x
    val  = read(0x1000 + 4*((x&7)*1024 + (flat>>3)))
```

This is verified bit-exact in `tb_cnn_top_axi.sv` (17/17 PASS, 6,272/6,272 per
mode). The reverse-column-index hazard flagged in the research plan (§9) is
absorbed by the hardware here: the bank/dense write re-orders results into
canonical order, so the software reads straight through — no reverse-index decode
in software.

## 5. Runtime OS↔WS reconfiguration

Software writes `DATAFLOW_MODE`, then `CONTROL.MODE_COMMIT`. The hardware honours
the commit only in IDLE (rejects with ERROR while BUSY), runs the 8-cycle flush,
and latches `mode_active` — the only signal the frozen array sees. Verified
OS→WS→OS and WS→OS→WS bit-exact with no reset and no reprogram.

## 6. Verification status

- `sim/tb_cnn_top_axi.sv` — **17/17 PASS**: OS 9,184 / WS 18,368 / WS-dense
  38,528 cycles, 6,272/6,272 bit-exact per mode, runtime switch, counters.
- `sim/tb_cnn_accelerator_v2.sv` — **81,583/81,583 PASS** (incl. the new
  measured WS-dense test).
- Synthesis/implementation of `cnn_top` — see `docs/PHASE3_SYNTHESIS.md`.

## 7. Board-only remainder

| Item | Status |
|---|---|
| KV260 board files + Zynq PS block design | **BOARD-ONLY** (scripted in `scripts/board/build_bitstream.tcl`) |
| Bitstream + `.xsa` + firmware package | **BOARD-ONLY** |
| Device-tree overlay load | **BOARD-ONLY** (`scripts/board/kv260_cnn.dts`) |
| Cross-compile + run `cnn_host` on A53 | **BOARD-ONLY** |
| Power/energy (INA260 differential + `report_power` `.saif`) | **BOARD-ONLY** |
| Measured latency/throughput/accuracy on silicon | **BOARD-ONLY** |
| ARM Cortex-A53 INT8 baseline (same layer) | **BOARD-ONLY** |
