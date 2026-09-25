# board — KV260 PYNQ host code and on-board measurement scripts (V2 Step 5)

Everything the KV260 runs lives here and is deployed by `deploy.sh` to `~/gos/` on the board.
Board-side code needs only **numpy + pynq** and the files in this directory plus the data package;
`v2/model` (torch, legacy code) is used only on the laptop (`make_board_data.py`, dry runs).

| file | role |
|---|---|
| `gos_driver.py` | `GosDevice` register-level driver; `PynqBackend` (KV260) and `ModelBackend` (dry run), same API |
| `gos_sim.py` | simulated CSR + memories (FORMATS.md semantics) used by the dry run and the tests |
| `gos_model_backend.py` | dry-run job model: gos_pack unpack + gos_golden + gos_cycle_model (laptop only) |
| `board_common.py` | data-package loading + SHA256 verification, PS dequant, provenance, CSV rules, `infer_loop` |
| `make_board_data.py` | builds `data/<net>/` on the laptop (below) |
| `test_shell.py` | Step 4 shell / memory smoke test (also usable on the real design with `--skip-scratch`) |
| `test_core_smoke.py` | core bring-up: VERSION/BUILD_ID, one LeNet + one CIFAR image bit- and cycle-exact, refused job + soft_reset |
| `exp_a1_accuracy.py` | A1 correctness on all 10k images per net |
| `exp_a2_a3_cycles.py` | A2 latency + A3 model / RTL / accelerator cycles per layer, determinism over images |
| `exp_a4_util.py` | A4 MAC_ACTIVE / cycles vs theoretical |
| `exp_b3_breakdown.py` | B3 host-side phase times (≥1000 images, warm-up discarded) |
| `exp_b1_power.py` | B1 idle / fpga / cpu power windows with meter banners |
| `exp_b2_clock.py` | B2 pl_clk0 sweep (≤ closed clock): cycles, latency, power window |
| `run_all.sh` | Session 2: A1 → A2/A3 → A4 → B3 → CPU baselines (`cpu/run_cpu_baselines.py`) |
| `deploy.sh` | rsync this directory + data + bitstream to `<user>@<host>:~/gos/`, write `DEPLOY_INFO.json` |
| `tests/` | pytest of the driver against the simulated register map (no hardware) |
| `cpu/` | CPU baselines (A5), separate owner; contract below |

## Prerequisites

- **Board:** KV260 with Ubuntu 22.04 + Kria-PYNQ (pynq 3.x, its Python with numpy). MMIO needs
  root: run every board script as `sudo -E python3 ...` (`-E` keeps the PYNQ environment, e.g.
  `XILINX_XRT`; on Kria-PYNQ you may need `source /etc/profile.d/pynq_venv.sh` first so that
  `python3` is the PYNQ venv). The board needs no git checkout.
- **Laptop:** the repo venv `~/cnn-accelerator/.venv` (numpy, torch, torchvision, pytest), the
  MNIST / CIFAR-10 test sets under `data/raw/`, the built bitstream (`v2/vivado/out/gos_200/`),
  `rsync` + `ssh` access to the board as `ubuntu`.

## Workflow (clean tree → deploy → sessions → copy back → commit results)

Results CSVs must come from a clean, committed tree (v2/CLAUDE.md). The board scripts refuse to
write hardware rows if `DEPLOY_INFO.json` or the data manifest says dirty (override
`--allow-dirty` marks the rows `git_dirty=True`, invalid for the paper).

```bash
# laptop, repo root, after committing the code
.venv/bin/python v2/board/make_board_data.py            # ~1 min; refuses a dirty tree
v2/board/deploy.sh <board-ip> [--bit v2/vivado/out/gos_200/gos_200.bit]   # prints the run commands
```

### Session 1 — bring-up (on the board, `cd ~/gos`)

```bash
sudo -E python3 test_shell.py --bit bit/gos_200.bit --expect-version 0x474F5302 --skip-scratch
sudo -E python3 test_core_smoke.py                     # PASS/FAIL, exit code
```
`test_shell.py` loads the overlay, prints pl_clk0 (read back), VERSION/BUILD_ID and fills/reads
every BRAM. `test_core_smoke.py` reloads the overlay and checks one LeNet-5 and one CIFAR-10 image
(logits bit-exact vs golden, LAYER_CYC/TOTAL_CYC/MAC_ACTIVE exactly the model, STALL 0, no
PS-busy violation), then a refused job (layer 1 K = 4 → ERR_CODE rule 3, layer 1, TOTAL_CYC =
C_START), soft_reset, and a good job again. Optional, to validate the fast memory write path:
`sudo -E python3 test_core_smoke.py --write-mode slice` (see Assumptions); use
`--write-mode slice` in Session 2 only if it passes.

### Session 2 — A1–A4, B3, CPU baselines

```bash
sudo -E ./run_all.sh 2>&1 | tee results/run_all_$(date +%Y%m%d_%H%M%S).log
```
Individual scripts take `--nets`, `--limit N`, `--timeout-s`, `--write-mode`, `--out-dir`.
Expected board time is dominated by Python MMIO (per image: input writes 256 / 768 32-bit
stores, ~25 CSR reads); `--quick` limits every step to 200 images for a first pass.

### Session 3 — power (B1) and clock sweep (B2), with the inline meter

```bash
sudo -E python3 exp_b1_power.py --modes idle fpga cpu --net lenet5 --window-s 60
sudo -E python3 exp_b1_power.py --modes cpu --cpu-kind cpu_int8_ref --cpu-threads 4 --net cifar10
sudo -E python3 exp_b2_clock.py --net lenet5            # clocks <= DEPLOY_INFO bit_clock_mhz
```
Each window prints `===== START ... | UTC ... | local ... | epoch ... =====` and a matching STOP
line; write the meter readings with those timestamps into a meter log. **The meter measures
board-level input power; the on-board INA260 value is SOM power; neither is accelerator power.**
Energy per inference (ΔP × time) must be computed by a script from the meter log joined on the
window timestamps (open item, see below), never typed by hand.

### Where results land and how to copy them back

- Board: `~/gos/results/hw_*.csv` (+ `hw_logits_<net>.npz`, `hw_cycles_<net>.npz`,
  `hw_b3_times_<net>.npz`, CPU `hw_cpu_baseline.csv`, run logs).
- Laptop: `rsync -av ubuntu@<board-ip>:gos/results/ v2/results/`, then
  `v2/scripts/check_results.py` (every CSV row must be clean at HEAD), then commit the results in
  a separate commit.
- Dry runs (`--backend model`) write only to `v2/results/dryrun/` (gitignored); rows carry
  `source=dryrun_model`. The code refuses model output outside a `dryrun` directory and hardware
  output inside one.

| file | rows | measured on the board |
|---|---|---|
| `hw_a1_accuracy.csv` | per net | images, logit/prediction mismatches vs golden, golden (model) and accelerator accuracy |
| `hw_a2_a3_cycles.csv` | per net × layer + total | model / RTL-sim / accelerator cycles, min/max/distinct over images, errors %, µs at the read-back clock, wall-clock per image |
| `hw_a4_util.csv` | per net × layer + total | MAC_ACTIVE/cycles vs theoretical (per-layer MAC_ACTIVE is **model T·K**, only the total is a HW counter) |
| `hw_b3_breakdown.csv` | per net × phase | input write, start→done, logit read, PS dequant, end-to-end, counter read, PL compute (median, p5, p95, p99) |
| `hw_b1_power.csv`, `hw_b1_power_samples.csv` | per window / per sample | window timestamps, inferences, INA260 samples |
| `hw_b2_clock.csv`, `hw_b2_power*.csv` | per clock | requested/read-back clock, cycles (must equal the model), latency, power window |

Metadata columns on every row: EXPERIMENTS.md "CSV rule" (`timestamp, git_commit, git_dirty,
vivado_version, bitstream_sha256, board_id, net, layer, clock_mhz, source, duration_s,
num_inferences`) + `build_id_hw` (BUILD_ID register), `board_hostname`, `clock_source`
(`clock_mhz` = pynq `Clocks.fclk0_mhz` read back on the board), `scripts_commit`,
`data_manifest_sha256`, `data_git_commit`, `data_git_dirty`, `backend`. `git_commit` /
`scripts_commit` come from `DEPLOY_INFO.json` on the board (git on the laptop); `source` is `hw`
or `dryrun_model`; `bitstream_sha256` is computed from the loaded `.bit`.

### Dry run on the laptop (no board)

```bash
cd v2/board
../../.venv/bin/python test_core_smoke.py --backend model
./run_all.sh --backend model --allow-dirty          # all 10k images; ~4 min without CPU
../../.venv/bin/python exp_b1_power.py --backend model --window-s 2 --gap-s 1
../../.venv/bin/python exp_b2_clock.py --backend model --window-s 2 --gap-s 1
../../.venv/bin/python -m pytest tests -q
```
`ModelBackend` runs the same driver code against `gos_sim` (FORMATS.md register semantics); on
CTRL.start it decodes the descriptors actually written, runs the config-checker model, unpacks
the ACT/WGT/QPARAM images actually written, computes every layer with `gos_golden.gos_layer`
and the counters with `gos_cycle_model`. Dry-run times (B3, B1 inference rates) are the laptop
simulator's and mean nothing.

## Driver (`gos_driver.py`)

```python
import board_common as bc, gos_driver as D
dev = D.GosDevice(D.PynqBackend("bit/gos_200.bit"))     # VERSION must be 0x474F5302
pkg = bc.load_package("data", "lenet5")                  # SHA256-verified
dev.load_net(pkg)                                        # WGT, QPARAM, DESC[0..L-1], N_LAYERS (+ readback)
r = dev.infer(pkg.x_act[0])                              # InferResult
r.logits, r.layer_cyc, r.total_cyc, r.mac_active, r.stall, r.violation, r.t_run_ns
pred = bc.predict(r.logits, pkg.dequant)                 # PS float32 dequant + argmax (D2)
```
- The input is always written to **ACT0** (FORMATS §1: layer i reads ACT[i mod 2]); the final
  layer writes **LOGIT** only, so no ACT buffer is read back. ACT0 is overwritten by layer 1's
  output, so the input is rewritten for every inference (it is anyway).
- 64-bit counters are read lo then hi. DESC / N_LAYERS / ACT are written only while
  STATUS.busy = 0. Refused jobs raise `GosJobError` (ERR_CODE decoded: rule id, name, layer);
  poll timeouts raise `GosTimeout`; `dev.recover()` soft-resets and rewrites the descriptors.
- `dev.set_fclk0(MHz)` sets pl_clk0 and returns the read-back value.

## CPU baseline contract (`cpu/`, other owner)

- `python3 cpu/run_cpu_baselines.py --data-dir <data> --out-dir <dir> --tag {board,laptop}
  [--nets lenet5 cifar10] [--quick] [--allow-dirty]` writes `hw_cpu_baseline.csv` into out-dir
  (called by `run_all.sh` with `--tag board`, or `--tag laptop` + `results/dryrun/` in a dry run).
- `cpu/cpu_infer.make_runner(kind, net, data_dir, threads)` returns `runner(x)` used by
  `exp_b1_power.py --modes cpu`: x = `x_nchw[i]` (int8) for `cpu_int8_ref`, else `x_f32[i]`
  (float32); BLAS/OpenMP thread variables are set from `--cpu-threads` before numpy is imported.
- The CPU scripts read this data package (and their own `data/<net>/cpu/` files).

## Assumptions (to confirm at bring-up) and open items

- **PYNQ API:** `pynq.Overlay(bit)` with the `.hwh` beside it; `pynq.MMIO(base, size)` with
  `.read/.write` (32-bit) and `.array` (numpy uint32 view); `pynq.ps.Clocks.fclk0_mhz` get/set.
  These are what `test_shell.py` already uses; `.array` is used for the default `elem` write mode
  (one 32-bit store per word, as `MMIO.write` does). `--write-mode slice` (numpy block copy into
  the mapped window) is faster but the store width/alignment is up to numpy/libc on Device
  memory: validate with `test_core_smoke.py --write-mode slice` before using it.
- **Stale done bit:** AXI does not order a write (CTRL.start) before a later read (STATUS)
  through the interconnect, and done/error are sticky. The driver therefore clears a set
  done/error with soft_reset (and waits for STATUS = 0) before every start; the cost is inside
  the B3 `start_done` phase and counted in `soft_reset_clears`.
- **Power sensor:** on-board power is read from the INA260 hwmon node (`/sys/class/hwmon/*/name`
  starting with `ina260`, `power1_input` in µW — the source platformstats uses), else parsed from
  `platformstats -p` output (format assumed), else recorded "unavailable". It is SOM power.
- **board_id:** `--board-id` / `$GOS_BOARD_ID`, else device-tree model + first 8 chars of
  `/etc/machine-id`.
- **Clock sweep:** runtime pl_clk0 changes via PYNQ are assumed to work (EXPERIMENTS B2 says to
  verify at bring-up; fallback: one bitstream per clock). After each change the driver
  soft-resets and reloads + reads back WGT/QPARAM/DESC.
- **Open:** a script that joins the manual meter log with the B1/B2 window timestamps and
  computes ΔP and energy per inference; per-layer MAC_ACTIVE is not measurable (no HW counter).

## Data package (`v2/board/data/<net>/`, gitignored)

Produced on the laptop by `make_board_data.py` (needs the repo `.venv`: numpy, torch/torchvision
for the datasets, `v2/model`). Consumed on the board by the experiment scripts here and by the CPU
baselines in `cpu/` — **board-side code needs only numpy (+ pynq for the FPGA)**. `<net>` ∈
{`lenet5`, `cifar10`} (`net_config.NETS`; `cifar10` = the r2 reference of record, DECISIONS D3).
N = 10,000 (full test set) unless `--limit` was used (then recorded in `net.json` / MANIFEST).

| file | contents |
|---|---|
| `inputs_act.npz` | `x`: int8 `[N, 8*D]` — the ACT0 byte image of each input (D = input ACT depth in 64-bit words: LeNet 128, CIFAR 384). Byte `8*w + b` = byte lane / bank `b` of ACT word `w` (FORMATS.md §1, little-endian), so `x[i].view(np.uint32)` is the exact sequence of 32-bit MMIO words to write from ACT0 offset 0 (lo word of 64-bit word w at `8*w`, hi at `8*w+4`). Unused bytes (x tail) are 0. `x_nchw`: int8 `[N, C, H, W]` — the same quantized input as a tensor (`= gos_pack.unpack_act(x)`). `x_f32`: float32 `[N, C, H, W]` — the legacy preprocessed FP32 model input (MNIST: ToTensor [0,1] + zero-pad 2 → 32x32; CIFAR-10: ToTensor [0,1], no mean/std). `x_nchw = clip(rint_float64(x_f32 / S_input), -128, 127)` (legacy `reference.quant.quantize_tensor`). |
| `labels.npy` | int64 `[N]` test labels |
| `golden_logits.npy` | int32 `[N, OC]` golden raw final-layer `v = acc + q_bias` (what the HW writes to `LOGIT[0..OC-1]`), from `gos_golden.run_net` |
| `golden_logits_f32.npy` | float32 `[N, OC]` PS float32 logits = `final_dequant(golden_logits)` |
| `golden_pred.npy` | int64 `[N]` golden prediction = argmax of the float32 logits (DECISIONS D2) |
| `wgt.npy` | uint64 `[words]` WGT image (FORMATS.md §2), written from WGT word 0 |
| `qparam.npy` | uint64 `[2*channels]` combined QPARAM image (FORMATS.md §3), written from QPARAM word 0 |
| `desc.npy` | uint32 `[N_LAYERS, 16]` layer descriptors (FORMATS.md §5); `N_LAYERS = desc.shape[0]` |
| `net.json` | net name, reference_version, dataset, N, N_LAYERS, OC (classes), input shape `[C,H,W]`, `act_in_words` (D), `S_input`, requant `B`, and per layer: name, IC, OC, IH, IW, KH, KW, OH, OW, K, relu, pool, final, in_sel, WGT_BASE, QP_BASE |
| `final_dequant.json` | final layer name, `S_a` (float), `S_w` (list, float64, OC), `scale` = `S_a*S_w` (float64, OC). PS dequant (legacy expression, D2): `logits = (v.astype(int64) * scale.reshape(1,-1)).astype(float32)`; `pred = logits.argmax(1)`. JSON floats round-trip exactly (Python repr). Implemented by `board_common.final_dequant` / `predict` and checked bit-identical against `v2/model/final_layer.py` when the package is built. |
| `quant_params.npz` | copy of the frozen legacy INT8 parameter file (`NET_CONFIGS[net]["quant_params"]`; keys `{layer}_q_w` int8, `{layer}_q_b` int32, `{layer}_S_a/_S_w/_S_out/_M`, `S_input`) — for CPU INT8 baselines |
| `hw_requant.npz` | copy of the frozen hardware requant set (`{layer}_m` uint64, `{layer}_s` uint8, `B`) |
| `fp32_params.npz` | FP32 checkpoint state_dict as float32 numpy (keys as in the checkpoint, e.g. `conv1.weight` `[OC,IC,KH,KW]`, `fc1.weight` `[OUT,IN]`, `*.bias`) — for CPU FP32 baselines |
| `model_cycles.json` | cycle model (`gos_cycle_model`, label **model**): per layer T, K, compute (= T·K = MAC_ACTIVE), cycles (= T·K + C_PIPE), macs, util_theoretical; constants C_PIPE/C_START/C_DONE; total cycles and MAC_ACTIVE |
| `rtl_cycles.csv`, `rtl_network.csv` | copies of `v2/results/rtl_cycles.csv` / `rtl_network.csv` (RTL simulation, label **rtl_sim**) |
| `refuse_test.json` | a descriptor corruption the RTL checker must refuse (layer 1, w7 = K := 4) with the expected ERR_CODE (rule 3, layer 1, from `gos_pack.job_err_code`) and TOTAL_CYC = C_START; used by `test_core_smoke.py` |
| `MANIFEST.json` | `files: {name: {sha256, bytes}}` for every file above, `git_commit`, `git_dirty`, `created_utc`, `generator`, `net`, `n_images`, golden accuracy. Board scripts verify every SHA256 before use (`board_common.load_package`). |

Subdirectory `cpu/` (written by the CPU-baseline exporter, own `CPU_MANIFEST.json`) is not part of
this manifest.

`v2/board/data/PACKAGE.json` lists the nets and the SHA256 of each `MANIFEST.json`
(`data_manifest_sha256` in every results row).

Python access (board or laptop): `from board_common import load_package; pkg = load_package(data_dir, "lenet5")`
→ `pkg.x_act`, `pkg.x_nchw`, `pkg.x_f32`, `pkg.labels`, `pkg.golden_logits`, `pkg.golden_pred`,
`pkg.wgt`, `pkg.qparam`, `pkg.desc`, `pkg.net` (net.json), `pkg.dequant`, `pkg.model_cycles`,
`pkg.manifest_sha256`.
