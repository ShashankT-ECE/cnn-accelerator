# V3 baselines: AMD DPU + ONNX Runtime on the KV260 (ResNet-20, MobileNet-style net)

Board baselines for the V3 nets (`v3/model/nets.py`: `resnet20_b`, `mobilenet_s`; `resnet20_a` compiles too).
The flow is the V2 flow (`v2/dpu/`, `v2/board/` @ 28dd2ad). The V2 code is copied into `v3/dpu/` and `v3/board/`
with a `copied from` / `adapted from` header, and nothing here imports `v2/`.

| system | what it is | label |
|---|---|---|
| `dpu` | AMD DPU DPUCZDX8G_ISA1_B4096, prebuilt pynq-dpu 2.5 KV260 overlay, Vitis AI 2.5.0 vai_q PTQ (power-of-2, calib train[0:1024]), VART batch 1 | its own quantization, not our INT8 reference |
| `ort_int8_tN` | ONNX Runtime CPU EP, `int8_qdq.onnx` (static QDQ S8S8, symmetric, per-channel weights, MinMax, calib train[0:1024]), N threads | its own quantization |
| `ort_fp32_tN` | ONNX Runtime CPU EP, `fp32.onnx` (opset 17), N threads | FP32 |

All three start from the same FP32 checkpoint and use the same preprocessing (`baseline_common.preprocess`).
The default preprocessing is u8/255 with no mean/std, which is the V2 convention. A checkpoint can carry
`"preprocess": {"mean", "std"}` instead, or the export commands take `--mean/--std`. **OPEN:** the preprocessing
must equal the v3/train recipe once that is committed.

## Files

| file | runs on | role |
|---|---|---|
| `v3/dpu/make_dryrun_ckpt.py` | laptop | THROWAWAY random-init checkpoint (BN statistics from train[0:1024] + 1 SGD step) for dry runs. Output goes to `v3/dpu/build/dryrun_ckpt/` and is never data |
| `v3/dpu/export_data.py` | laptop | checkpoint → `v3/dpu/build/<net>/{dpu_inputs.npz, fp32_state_dict.pt, torch_pred.npy, export_info.json}`, plus the FP32 test accuracy (model) |
| `v3/dpu/fetch_overlay_ref.sh` | laptop | pynq-dpu 2.5 sdist + prebuilt KV260 overlay `.bit/.hwh` + example xmodel (md5 checked) |
| `v3/dpu/run_docker.sh` | laptop | Vitis AI 2.5.0 container steps `arch`, `quantize <net>` (calib/test/deploy), `compile <net>` (vai_c_xir + `inspect_xmodel.py --expect-overlay`), `all`. It waits for `v3/scripts/vivado_guard.sh` |
| `v3/dpu/overlay_arch.py`, `vai_quantize.py`, `inspect_xmodel.py`, `xmodel_diff.py` | container | overlay fingerprint → arch.json; vai_q; xmodel subgraphs / ops / fingerprint check; recompile diff |
| `v3/board/export_onnx.py` | laptop | checkpoint → `v3/board/data/<net>/onnx/{fp32.onnx, int8_qdq.onnx, *_pred.npy, ONNX_INFO.json}`, plus the laptop accuracies |
| `v3/board/make_baseline_package.py` | laptop | → `v3/board/data/package/` (scripts, CIFAR-10 test u8, models, `MANIFEST.json` sha256). It refuses a dirty tree and a throwaway checkpoint unless the matching flags are given. `--results` writes `v3/results/baseline_model_accuracy.csv` |
| `v3/board/deploy_baseline.sh <ip>` | laptop | rsync the package to `~/gos3/` (never `~/gos/`, which holds V2). `~/gos3/results*/` is kept |
| `baseline_session.py` (+ `session.sh`) | board / laptop dry run | accuracy, latency and power per net × config; resumable; 3 sessions |
| `power_log.py`, `stats.py`, `board_env.py`, `login_spikes.py`, `aggregate_sessions.py`, `baseline_common.py` | board / laptop | INA260 logger + protocol, statistics, environment pre-flight, ssh / apt overlap check, 3-session median (V2 D26) |
| `v3/board/tests`, `v3/dpu/tests` | laptop | pytest: stats, CSV schema, power arithmetic, DPU I/O, Python 3.7 container syntax, full dry run end to end |

Generated files are all gitignored: `v3/dpu/build/`, `v3/board/data/`, `v3/results/dryrun/`.

## Laptop: from a checkpoint to a board package

Run these from the repo root of the `~/gos-v3` worktree. Repo `.venv`. Only one Vivado job at a time: the container waits for the guard.

```bash
NET=resnet20_b; CK=v3/train/runs/<job>/<file>.pt          # the external GPU training checkpoint
.venv/bin/python v3/dpu/export_data.py   --net $NET --checkpoint $CK [--expect-correct N]
.venv/bin/python v3/board/export_onnx.py --net $NET --checkpoint $CK
v3/dpu/fetch_overlay_ref.sh                                 # once
DPU_NETS="resnet20_b mobilenet_s" v3/dpu/run_docker.sh all  # arch + quantize + compile (logs v3/dpu/build/logs/)
#   the compile step fails unless the xmodel fingerprint == the overlay's (0x101000016010407)
.venv/bin/python -m pytest v3/dpu/tests v3/board/tests -q
git commit ...                                              # clean tree: the package records the commit
.venv/bin/python v3/board/make_baseline_package.py [--results]
v3/board/session.sh 1 --backend model --pkg-dir v3/board/data/package   # laptop dry run of that package
v3/board/deploy_baseline.sh <board-ip>
```

Dry run with a throwaway checkpoint (pipeline check only; every row is `dryrun_model`, `paper_grade=False`):

```bash
.venv/bin/python v3/dpu/make_dryrun_ckpt.py
for n in resnet20_b mobilenet_s; do
  .venv/bin/python v3/dpu/export_data.py   --net $n --checkpoint v3/dpu/build/dryrun_ckpt/$n.pt
  .venv/bin/python v3/board/export_onnx.py --net $n --checkpoint v3/dpu/build/dryrun_ckpt/$n.pt
done
v3/dpu/run_docker.sh all
.venv/bin/python v3/board/make_baseline_package.py --allow-dirty --allow-throwaway
v3/board/session.sh 1 --backend model [--limit 500] [--fresh]     # -> v3/results/dryrun/baseline/
v3/board/session.sh 2 --backend model; v3/board/session.sh 3 --backend model
.venv/bin/python v3/board/aggregate_sessions.py --root v3/results/dryrun/baseline
```

## Board (KV260, Kria-PYNQ with pynq-dpu 2.5 and onnxruntime in the PYNQ venv, as in V2)

```bash
ssh ubuntu@<board-ip>; tmux; cd ~/gos3
./session.sh py power_log.py --list-sensors        # INA260 (hwmon ina260_u14 in V2)
./session.sh 1 --plan                              # pre-flight + steps, runs nothing
./session.sh 1 --limit 200 --no-power --allow-non-paper-grade --results-dir ~/gos3/results_smoke   # smoke
./session.sh 1                                     # session 1; rerun the same command to resume
./session.sh 2        # another day
./session.sh 3        # another day
python3 aggregate_sessions.py                      # -> results/hw_baseline_repeatability.csv
```

`session.sh` handles sudo and the PYNQ venv. A session runs per net: 5 accuracy+latency steps (10,000 images +
50 warm-up each, batch 1), then the power protocol. The power protocol runs 5 configurations × 3 repeats with
60 s idle and 60 s run phases, which is 31 phases = 31 min per net. `--configs`, `--power-configs`, `--threads`,
`--no-power` and `--power-only` narrow a session.

Copy the results back from the laptop:
`rsync -av ubuntu@<board-ip>:gos3/results/ v3/results/`. Then run `v3/scripts/check_results.py` and make a separate results commit.

## What a session measures (see the `baseline_session.py` docstring for details)

- **Accuracy** is computed on all 10,000 CIFAR-10 test images, starting from the raw u8 image. It is compared
  with the laptop reference predictions of the same model: the vai_q PyTorch model for the DPU, ORT INT8 on the
  laptop for `ort_int8`, and PyTorch FP32 for `ort_fp32`.
- **Latency** is measured per image with `perf_counter_ns`, after 50 warm-up images that are discarded. It is split
  into `pre` (u8 → preprocessing → runner input; DPU int8 = clip(floor(x·2^fp + 0.5))), `infer` (VART
  execute_async+wait / `session.run`), `post` (argmax) and `e2e`. Each part reports p50/p95/p99/mean/min/max and
  the median with its distribution-free 95 % CI.
- **Energy** uses the INA260 SOM rail sampled by a separate process at 10 Hz. Run phases come in a seeded random
  order per repeat, and each is bracketed by idle phases. P_idle is the mean of the two bracketing idle phases,
  and E_sys = (P_run − P_idle) × time/image. The label is "SOM-rail power (INA260)". This is not DPU-only power.
- **Pinning:** the DPU and 1-thread ORT run on core 3 (`--meas-cores`). N-thread ORT runs on cores 0-3
  (`--cpun-cores`); its pool threads are created there. The sampler process runs on the other cores.
- **Pre-flight** (as V2 `run_sessions.py`): no apt/dpkg/unattended-upgrades/packagekitd process may be running;
  the apt/PackageKit unit states are recorded (they have been masked on the board since the V2 campaign, D27);
  the governor must be `performance` at a fixed frequency (read back, restored on exit); pinning is read back; the
  die temperature and free disk are checked. After the run, ssh logins and apt/PackageKit windows that overlap a
  power phase are flagged.
- **paper_grade** requires the pre-flight to pass, a clean package, a trained (not throwaway) checkpoint and a
  board source. The DPU rows also carry the overlay `dpu.bit` sha256 and the pl_clk0 read-back.

Outputs go to `~/gos3/results/` for session 1 and `results/rep<K>/` for session K ≥ 2:
`hw_baseline_accuracy.csv`, `hw_baseline_latency.csv`, `hw_baseline_power_ina260_{samples,phases,summary}_<net>.csv`,
`hw_baseline_preds_<net>_<cfg>.npz`, `hw_baseline_session_info.json`, `baseline_state.json` and `logs/`.
The row source is `measured on KV260`.

## Caveats (to confirm at bring-up)

- The DPU input/output dtype and fix_point are read from the runner. The dry run uses the xmodel_info values.
- The DPU core clock is generated inside the overlay. Only the pl_clk0 read-back is recorded.
- ORT INT8 with 4 threads gave 9987/10000 agreement with the 1-thread laptop reference in the x86 dry run. The
  agreement is reported and never forced.
- The vai_q calibration tolerates a 0.1 % FP32 difference between the container's torch 1.10 and the laptop's
  torch 2.x. The difference is recorded in `quant_info_calib.json`; V2 required equality.
