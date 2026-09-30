# cpu — same-board CPU baseline (EXPERIMENTS.md A5)

Cortex-A53 baselines for the KV260 comparison. Board side needs **numpy only**; onnxruntime is
optional (missing → rows with `status = unavailable: ...`, the run still succeeds).

| file | runs on | purpose |
|---|---|---|
| `export_cpu_models.py` | laptop (repo `.venv`, torch) | writes the CPU package `v2/board/data/<net>/cpu/` (gitignored) and verifies it |
| `cpu_infer.py` | board / laptop | `make_runner(kind, net, data_dir, threads)` → runner with `.preprocess`, `.compute`, `.e2e` |
| `run_cpu_baselines.py` | board / laptop | A5 protocol, writes `hw_cpu_baseline.csv` + `hw_cpu_baseline_env.json` |

## Kinds

| kind | what | label |
|---|---|---|
| `cpu_int8_ref` | vendored numpy copy of `v2/model/gos_golden.py` (int64 einsum conv, exact per-channel requant, D2 float32 dequant); bit-exact raw INT32 logits | **unoptimized reference** |
| `cpu_fp32_numpy` | FP32 reference net (LeNet `data/checkpoint/lenet5_fp32.pt`, CIFAR r2 `v2/model/retrain/cifar10_fp32_r2.pt`), im2col + float32 matmul (numpy BLAS) | numpy FP32 |
| `cpu_ort_fp32` | ONNX Runtime CPU EP, `fp32.onnx` (torch export, opset 17) | ORT FP32 |
| `cpu_ort_int8` | ONNX Runtime CPU EP, `int8_qdq.onnx` = ORT static QDQ quantization (S8S8, symmetric, per-channel weights, MinMax), calibrated on the project INT8 calibration images (legacy transform, train[0:1024], both nets) | ORT INT8 — **its own quantization, not the project INT8 numerics**; own accuracy |

Inputs: `compute` = preprocessed tensor already in memory (int8 NCHW for `cpu_int8_ref`, float32
NCHW otherwise) → float32 logits. `e2e` = raw stored dataset image (uint8: MNIST 28x28, CIFAR
32x32x3 HWC) → legacy preprocessing (/255, LeNet zero-pad 2; `cpu_int8_ref` also the float64-RNE
input quantization) → logits → argmax. One image per call.

## CPU package (`<data_dir>/<net>/cpu/`)

`int8_ref.npz`, `fp32.npz`, `fp32.onnx`, `int8_qdq.onnx`, `raw_test.npz` (`x_u8`, `y`),
`CPU_MANIFEST.json` (SHA256 of every file, source checkpoint/npz SHA256s, versions, git state,
ORT quantization settings, laptop verification counts). `run_cpu_baselines.py` re-checks every
SHA256 before running. It lives beside the board data package of `make_board_data.py`
(`../README.md` "Data package"), which only replaces files in `<net>/`, not the `cpu/` subdir.
When that package is present, the accuracy pass cross-checks `labels.npy`, `inputs_act.npz`
(`x_nchw` for `cpu_int8_ref`, `x_f32` otherwise), and `golden_logits.npy` (int32, bit-exact vs
`cpu_int8_ref` raw logits). Any mismatch → `status = FAIL: package mismatch`, exit 1.

## Laptop

```bash
.venv/bin/python v2/board/cpu/export_cpu_models.py            # refuses a dirty tree; --allow-dirty to override
.venv/bin/python v2/board/cpu/run_cpu_baselines.py --data-dir v2/board/data \
    --out-dir v2/results/dryrun/cpu_laptop --tag laptop [--quick]
```
`--tag laptop` writes only under `v2/results/dryrun/` (refused elsewhere), `source = cpu_laptop`.
Laptop numbers are not paper data.

## Board

```bash
pip install onnxruntime==1.19.2        # installed on the KV260 2026-09-30 (DECISIONS D22: packages added, none changed)
python3 run_cpu_baselines.py --data-dir ~/gos/data --out-dir ~/gos/results --tag board
```
`--tag board` requires an aarch64 host and refuses (unless `--allow-dirty`) when DEPLOY_INFO.json
says dirty or a CPU package was exported from a dirty tree. Provenance: `DEPLOY_INFO.json` (looked
up in the data dir, its parent, this dir and its parent — `~/gos/DEPLOY_INFO.json` after deploy),
else git. `board_id` = `--board-id`, else `GOS_BOARD_ID`, else device-tree model + machine-id
(as `board_common`). `bitstream_sha256`, `vivado_version`, `clock_mhz` stay empty (no PL used).

Protocol: each net × kind × thread count (default 1 and 4) runs in its own subprocess with
`OPENBLAS_NUM_THREADS = OMP_NUM_THREADS = MKL_NUM_THREADS` set before numpy is imported (plus
threadpoolctl limits if installed; ORT `intra_op_num_threads`); `--warmup` (10) calls discarded,
`--runs` (100, ≥ 100 for paper data) timed with `perf_counter_ns`, cycling over the first `runs`
test images; median / p5 / p95 / mean. Accuracy pass (mode `accuracy`, not timed) at the first
thread count: 10,000 images (`--quick`: 500). Environment (numpy `show_config`, threadpoolctl,
ORT version, `/proc/cpuinfo` model, hostname) goes to the CSV columns and `hw_cpu_baseline_env.json`.

## onnxruntime on the board (aarch64)

PyPI has `cp310` aarch64 wheels: `manylinux2014` up to 1.16.3 and `manylinux_2_27/2_28` for
1.17.0–1.23.2 (checked with `pip download --platform ... --python-version 3.10`). Ubuntu 22.04
(glibc 2.35, pip 22.x) can install them; 1.19.2 needs numpy ≥ 1.21.6. The ONNX files are IR 8 /
opset 17 with standard ops only, so any ORT ≥ 1.14 loads them. If the board has no internet:
on the laptop `pip download --no-deps onnxruntime==1.19.2 coloredlogs flatbuffers humanfriendly
packaging protobuf sympy mpmath --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64
--python-version 3.10 --only-binary=:all: -d wheels/` (numpy is not fetched: the board's own numpy
must be ≥ 1.21.6), copy `wheels/` and `pip install --no-index --find-links wheels onnxruntime==1.19.2`.
(Resolution of this command checked on the laptop; the install itself is untested on the board.) If it cannot be
installed, the ORT rows are recorded as unavailable.
