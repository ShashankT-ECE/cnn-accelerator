# KV260 host software

`cnn_host.c` is the ARM Cortex-A53 userspace application that drives the PL-side
accelerator (`rtl/top/cnn_top.sv` → `cnn_axi_ctrl` → `cnn_accelerator_v2`)
through its AXI4-Lite slave.

## Build

Cross-compile for the Kria KV260 (aarch64):

```bash
sudo apt-get install gcc-aarch64-linux-gnu   # once
make CROSS_COMPILE=aarch64-linux-gnu-
# -> cnn_host (aarch64 ELF)
```

Native x86 compilation smoke test (does **not** run — there is no UIO device):

```bash
make native
```

## Run on the KV260

The AXI slave must first be exposed as a UIO device by a device-tree overlay
(see `scripts/board/kv260_cnn.dts`). After loading the bitstream + overlay, the
slave appears as `/dev/uio<N>` with one memory map. Then, from the repo root:

```bash
./cnn_host /dev/uio0 data/vectors/input_img.hex data/vectors/weights.hex data/vectors/golden_canonical.hex
```

It loads the real MNIST-12 Conv1 vectors, runs the layer in OS / WS / WS-dense
modes, checks every result bit-exact against the committed integer golden, and
prints per-mode cycle counts (the hardware `CYCLE_RUN` instrument), MAC/cycle,
and PS wall-clock overhead.

## Measurement discipline

- **Compute latency** comes from the hardware `CYCLE_RUN` register (per-run,
  START→DONE, frozen at DONE) — no software timing noise.
- **PS wall-clock** (AXI load + poll + readback) is reported separately and is
  never folded into the hardware number.
- **Power/energy** is collected out-of-band on the board (INA260 differential,
  `xlnx-platformstats`) — see `docs/PHASE3_BOARD_INTEGRATION.md`. This
  application does not report power; it leaves that to the board harness.
