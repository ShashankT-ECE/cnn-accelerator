# dpu — AMD DPU (DPUCZDX8G) baseline on the KV260 (Vitis AI 2.5.0, prebuilt pynq_dpu overlay)

The same FP32 networks as our accelerator (DECISIONS D3: LeNet-5 = legacy checkpoint
`data/checkpoint/lenet5_fp32.pt`, CIFAR-10 = r2 `v2/model/retrain/cifar10_fp32_r2.pt`; legacy
classes `python/lenet5`, `python/cifar10`, unchanged) quantized with **Vitis AI vai_q_pytorch**
and compiled for the KV260 DPU, plus a board session measuring accuracy, latency and power.

**Label rule.** DPU numbers come from a *different* quantization (vai_q PTQ, power-of-two
scales) of the same FP32 networks. They are never our INT8 reference and never our accelerator.
Every row carries `DPU (Vitis AI 2.5.0, DPUCZDX8G_ISA1_B4096, fingerprint <fp>)`; container
accuracies are labelled `DPU quantized (vai_q), model`; power is `SOM-rail power (INA260)`.

## Version mapping (prebuilt pynq_dpu overlay -> Vitis AI 2.5.0)

We use the **prebuilt** KV260 DPU overlay shipped with pynq_dpu (no custom DPU platform build).
The Vitis AI version and the compile target follow from it:

| item | value | evidence |
|---|---|---|
| board runtime | Kria-PYNQ installs `pynq-dpu==2.5` (with pynq 3.0.1) | Kria-PYNQ `install.sh` (`python3 -m pip install pynq-dpu==2.5 --no-build-isolation`) |
| pynq-dpu 2.5 | DPU-PYNQ v2.5: "supports PYNQ 3.0 and Vitis AI 2.5.0"; KV260 = B4096, 1 core | DPU-PYNQ README; PyPI sdist `pynq_dpu-2.5.tar.gz` sha256 `29110dc7…3486f43` |
| overlay files | `dpu.bit/.hwh/.xclbin` = `pynqdpu.dpu.kv260_som.2.5.0.*`; md5 bit `424cfe2e9e5e247982b984e4dcc69006`, hwh `dc81379b8d742b8ac6700f2bc45d6fd1` | `dpu.*.link` files in the sdist; downloaded + md5-checked by `fetch_overlay_ref.sh` |
| overlay DPU config | `.hwh`: ARCH_ICP 16 × ARCH_OCP 16 × ARCH_PP 8 (= 4096 ops/cycle, B4096), UBANK_IMG_N 5, UBANK_WGT_N 17, LOAD/SAVE_PARALLEL 2, ALU_PARALLEL 4, CONV_LEAKYRELU 1 | `build/overlay_arch_info.json` |
| **overlay fingerprint** | **`0x101000016010407`** | DPU subgraph of Xilinx's KV260 example xmodel for this overlay (`pynqdpu.tf2_mnist_classifier.DPUCZDX8G_ISA1_B4096.2.5.0.xmodel`, md5 per the `.link` file), read with xir by `overlay_arch.py` |
| compile target | `build/arch_kv260_pynqdpu25.json` = `{"fingerprint": "0x101000016010407"}` (DPU-PYNQ's custom arch.json convention, cf. its `arch_ultra96.json`) | written by `overlay_arch.py`; pynq-dpu 2.5 ships no arch.json |
| cross-check | the docker's `/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json` (`{"target": "DPUCZDX8G_ISA1_B4096"}`) resolves to type 0x01 / ISA 1 / feature code 0x16010407 = the same fingerprint | `overlay_arch_info.json` `docker_target_matches_overlay: true` |
| docker image | `xilinx/vitis-ai-cpu:2.5.0` (= `xilinx/vitis-ai:2.5.0`, the image DPU-PYNQ's host README uses), manifest `sha256:eaa85efb06924995ebdb973546e7f69169b003b8cc525764bd9524ad554dddbe`, 6.40 GB compressed | `docker image inspect` |
| container tools | Python 3.7.12, PyTorch 1.10.1 (CPU), vai_q_pytorch (pytorch_nndct) 0.1.0+09b3f3d, xir / xcompiler / vart / unilog 2.5.0 | `xmodel_info.json` `versions` |

`xilinx/vitis-ai-pytorch-cpu:3.0/3.5` exist, but they target the VAI 3.x runtime; the Kria-PYNQ
overlay/VART are 2.5. Every compiled xmodel's DPU fingerprint is checked against the overlay's
(`inspect_xmodel.py --expect-overlay`; `make_dpu_package.py` refuses a mismatch), and the board
session refuses a `dpu.bit` whose md5 is not the prebuilt KV260 overlay's (`--allow-overlay-mismatch`
to override). The on-board DPU fingerprint is also recorded from `xdputil query` when available.

## Files

| file | where it runs | role |
|---|---|---|
| `export_data.py` | laptop, repo `.venv` | FP32 accuracy with the legacy loaders == accuracy of record (else stops); exports `build/<net>/dpu_inputs.npz` (test[0:10000], calib train[0:1024], legacy preprocessing), state_dict copy, checks x_test == board package `x_f32` |
| `fetch_image.sh` | laptop | resumable fallback for `docker pull` on slow links: registry blobs via curl (resume + stall restart), every sha256 verified, `docker load` |
| `fetch_overlay_ref.sh` | laptop | pynq-dpu 2.5 sdist + the prebuilt KV260 overlay `.bit/.hwh` + KV260 example xmodel (md5 vs `.link` files) → `build/pynq_dpu_2.5/` |
| `run_docker.sh` | laptop | container steps (`arch`, `quantize`, `compile`, `all`, `shell`) with `--cpus 8 --memory 12g`, repo mounted read-only, as your uid; resource guard (no vivado/xsim/xsimk/xelab, ≥ 8 GB available) before each container |
| `overlay_arch.py` | container | overlay fingerprint → `build/arch_kv260_pynqdpu25.json` + `build/overlay_arch_info.json` |
| `vai_quantize.py` | container | `--mode calib`: FP32 check + PTQ calibration; `--mode test`: vai_q accuracy on 10k + predictions; `--mode deploy`: fresh quantizer, one batch-1 forward, `export_xmodel` |
| `inspect_xmodel.py` | container | subgraphs (DPU/CPU/USER + ops), I/O tensors + fix_point, fingerprint == overlay check, tool versions → `compiled/xmodel_info.json` |
| `xmodel_diff.py` | container | compares two xmodels' DPU code, register/parameter maps and constants (recompile check) |
| `make_dpu_package.py` | laptop | `build/package/` (xmodels, xmodel_info, vaiq_pred, `DPU_INFO.json`, `dpu_session.py`) + `build/dpu_model_accuracy.csv`; `--results` writes `v2/results/dpu_model_accuracy.csv` (clean tree only) |
| `deploy_dpu.sh` | laptop | rsync `build/package/` → `~/gos/dpu/` (needs the V2 board deployment in `~/gos/`) |
| `dpu_session.py` | KV260 (or `--dry-run` on the laptop) | accuracy, latency, INA260 power (below) |
| `tests/` | laptop | pytest, no hardware (fake runner + mock sensor) |

Everything generated goes to `v2/dpu/build/` (gitignored: exported tensors, calibration data,
vai_q output, xmodels, overlay reference files, logs, package). No Docker image or dataset is committed.

## Reproduction

```bash
# 0. laptop, repo root, clean committed tree; resource rule: no Vivado/xsim running, >= 8 GB RAM
#    available; >= 60 GB free on the Docker root (docker info | grep "Docker Root Dir"; df -h)
free -h; df -h /var/lib/docker; docker info | grep "Docker Root Dir"
docker pull xilinx/vitis-ai-cpu:2.5.0      # or, on a slow/stalling link: v2/dpu/fetch_image.sh
v2/dpu/fetch_overlay_ref.sh                # prebuilt pynq-dpu 2.5 KV260 overlay references

# 1. export (FP32 must equal v2/results/reference_accuracy.csv: 9878 / 7856 at the time of writing)
.venv/bin/python v2/dpu/export_data.py

# 2.+3. overlay arch.json, quantize (calib, test, deploy), compile; one container at a time
v2/dpu/run_docker.sh all          # = arch; quantize lenet5; compile lenet5; quantize cifar10; compile cifar10
#     logs: v2/dpu/build/logs/{arch,quantize_<net>_{calib,test,deploy},compile_<net>}.log

# 4. package + model-labelled accuracy CSV
.venv/bin/python v2/dpu/make_dpu_package.py            # build/ only
.venv/bin/python v2/dpu/make_dpu_package.py --results  # + v2/results/dpu_model_accuracy.csv (clean tree)

# 5. laptop dry run of the board session (fake runner, mock sensor -> v2/results/dryrun/dpu/)
.venv/bin/python -m pytest v2/dpu/tests -q
.venv/bin/python v2/dpu/dpu_session.py --dry-run --pkg-dir v2/dpu/build/package --limit 200

# 6. deploy (after v2/board/deploy.sh <board-ip>)
v2/dpu/deploy_dpu.sh <board-ip>
```

On the board (`ssh ubuntu@<board-ip>`, inside tmux):

```bash
source /etc/profile.d/pynq_venv.sh
python3 -c "import pynq_dpu; print(pynq_dpu.__file__)"   # Kria-PYNQ installs pynq-dpu 2.5
sudo -E python3 ~/gos/power_log.py --list-sensors
cd ~/gos/dpu
sudo -E python3 dpu_session.py --limit 200 --no-power --out-dir ~/gos/results/dpu_quick   # smoke
sudo -E python3 dpu_session.py            # both nets: 10k images each + INA260 power (~2 x 9.5 min)
```
Copy back with `rsync -av ubuntu@<board-ip>:gos/results/hw_dpu_* v2/results/`, then
`v2/scripts/check_results.py` and a separate results commit.

## Board session (`dpu_session.py`)

Per net, batch 1, input = the data package's `x_f32` (legacy preprocessing; checked
bit-identical to the tensors vai_q saw), NCHW → NHWC once before the loop:

- **Accuracy** on all images (default 10,000) vs labels; agreement with our golden INT8
  predictions (`golden_pred.npy`, comparison only) and with the vai_q PyTorch model's
  predictions from the container (`vaiq_pred.npy`: how faithfully the DPU executes the vai_q model).
- **Latency** after `--warmup` (default 50) discarded images, per image with
  `perf_counter_ns`: `dpu_runner` (VART `execute_async` + `wait`), `pre` (float → int8 into the
  preallocated input buffer), `post` (argmax), `end_to_end` (all three); p50/p95/p99/mean/min/max µs.
- **Power:** `power_log.run_power_protocol` with the end-to-end DPU loop (images cycled) as the
  "accel" workload, idle/accel/idle × `--repeats` (default 3) × `--phase-s` (default 60 s), no
  CPU phases (the CPU baseline is B1's). Same sensor discovery, CSV layout, P_idle rule, ΔP and
  energy/image arithmetic as B1.
- **Outputs** (`~/gos/results/`): `hw_dpu_accuracy.csv`, `hw_dpu_latency.csv`,
  `hw_dpu_preds_<net>.npz` (per-image predictions, raw int8 outputs, times),
  `hw_dpu_power_ina260_{samples,phases,summary}_<net>.csv`, `hw_dpu_session_info.json`
  (arguments, overlay path + sha256, pynq_dpu version, `xdputil query` output, pl_clk0 read back).
  Row metadata = EXPERIMENTS.md CSV rule via `board_common.write_csv`; `source=hw` only with
  pynq_dpu on the board; `--dry-run` rows are `dryrun_model` and are written only under a
  `dryrun` directory.
- **Provenance:** xmodel SHA256 vs `DPU_INFO.json`; the data package MANIFEST is verified by
  `board_common.load_package`; a dirty DPU or data package is refused unless `--allow-dirty`
  (rows then `git_dirty=True`, invalid for the paper). `bitstream_sha256` = the pynq_dpu
  overlay `dpu.bit`; `clock_mhz` = pl_clk0 read back (the DPU core clock is generated inside the
  overlay and is not read — recorded in `clock_source`).

## Caveats / assumptions to confirm at bring-up

- **Input/output dtype.** The session reads dtype and `fix_point` from the runner tensors: int8
  input → `clip(floor(x·2^fp + 0.5), -128, 127)` (inputs are ≥ 0; round half up — matches the
  vai_q input quantizer for non-negative values except possibly at exact ties); float tensors are
  copied unchanged. The dtype/fix_point used is written into every row (`io` column).
- **Overlay/fingerprint.** `DpuOverlay("dpu.bit")` = the prebuilt KV260 overlay installed by
  pynq-dpu 2.5 (md5-checked). The xmodels carry its fingerprint `0x101000016010407`. If the board
  runs a different pynq-dpu (e.g. a MakarenaLabs/3.5 fork), runner creation fails or the md5
  check refuses; then put an arch.json with the board's fingerprint (`xdputil query`) in
  `v2/dpu/build/` and run `DPU_ARCH_JSON=/workspace/build/<file>.json v2/dpu/run_docker.sh compile <net>`.
- **Subgraphs.** Each net compiles to 3 subgraphs: USER (input), **one DPU subgraph holding every
  layer** (conv/FC as conv2d-fix, ReLU fused, max-pool as pool-fix), and one CPU subgraph with
  only `fix2float` (output conversion). pynq_dpu runs the DPU subgraph; the session does the
  int8 → argmax itself. No compute falls back to the CPU (`cpu_compute_ops: []`).
- **Rounding at the input.** Inputs are k/255 (k = 0..255) and the input fix_point is 6, so
  x·64 is never exactly at a .5 tie; the round-half-up conversion equals any round-to-nearest.
- **Export batch.** vai_q requires batch 1 for `export_xmodel`; the `deploy` mode builds a fresh
  quantizer with a single batch-1 forward (a batch-500 export printed `VAIQ_ERROR`); the
  prediction of image 0 is checked against the test-mode run.
- **Recompiles are not byte-identical** (xmodel sha256 changes between `vai_c_xir` runs), but
  `xmodel_diff.py` shows identical DPU machine code, register/parameter maps, fingerprint and
  constants. The SHA256 recorded in `DPU_INFO.json` identifies the deployed file.
- **Energy.** power_log's `E_comp`/`t_PL`/`duty` need a PL cycle counter (our accelerator); the
  DPU has none exposed here, so those columns stay empty. The DPU energy per image is
  `E_sys` = ΔP × time/image (end-to-end batch-1 loop).
- **Latency** is PS wall-clock around VART calls (Python), not a DPU cycle counter; for these tiny
  nets it is dominated by runtime overhead. DPU numbers are INT8 with Vitis AI's power-of-two
  scales; the vai_q container accuracy is a model (PyTorch simulation), not a DPU measurement.
- **Power** is SOM-rail (VCC_SOM) power incl. the PS running Python + VART; not DPU-only power.
- `check_results.py` reads every `v2/results/*.csv`; the hw_dpu_* CSVs follow the same metadata
  rule, but whether it accepts their extra columns is checked when they are copied in.
