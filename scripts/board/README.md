# Board build artifacts (BOARD-ONLY)

Everything needed to take the verified PL design to the KV260 is prepared here;
the only remaining work requires the physical board / board files.

| File | Purpose | Verified? |
|---|---|---|
| `build_bitstream.tcl` | Vivado block-design + bitstream + `.xsa` flow (Zynq PS + Clocking Wizard + SmartConnect + `cnn_top`) | **BOARD-ONLY** (needs KV260 board files, not installed) |
| `kv260_cnn.dts` | device-tree overlay exposing `cnn_top` as a `generic-uio` device | **BOARD-ONLY** (compile with `dtc`) |
| `shell.json` | Kria firmware shell manifest template | **BOARD-ONLY** (finalize after block design) |

## Day-1 board sequence (pre-registered)

1. Install the KV260 board files (Xilinx Board Store).
2. `vivado -mode batch -source scripts/board/build_bitstream.tcl` → `cnn_top.bit` + `.xsa`.
3. Package firmware (`shell.json` + `.pdi`) and copy `cnn_top.bit.bin` + `kv260_cnn.dtbo` to the KV260.
4. Cross-compile the host app: `make -C software CROSS_COMPILE=aarch64-linux-gnu-` (install `gcc-aarch64-linux-gnu` on the dev machine).
5. On the KV260: `xmutil loadapp kv260_cnn`, then
   `./cnn_host /dev/uio<N> data/vectors/input_img.hex data/vectors/weights.hex data/vectors/golden_canonical.hex`.
6. Expected (pre-registered): OS 9,184 / WS 18,368 / WS-dense 38,528 cycles;
   6,272/6,272 results bit-exact; accuracy 98.90% (FP32 == INT8).

## Power/energy (board-only, rehearsed method)

Differential INA260 (SOM total) at I²C 0x40 via `xlnx-platformstats`:
`P_accel = P_SOM(running) − P_SOM(idle)`, sampled over the run window; report
**energy/inference**, not watts, and state exactly what the rail covers. Pre-board
estimate from Vivado `report_power` with a real `.saif`.
