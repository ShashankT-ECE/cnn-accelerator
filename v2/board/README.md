# board — KV260 PYNQ host code and on-board measurement scripts (V2 Step 5)

Everything the KV260 runs lives here and is deployed by `deploy.sh` to `~/gos/` on the board.
Board-side code needs only **numpy + pynq** and the files in this directory plus the data package;
`v2/model` (torch, legacy code) is used only on the laptop (`make_board_data.py`, dry runs).

| file | role |
|---|---|
| `gos_driver.py` | `GosDevice` register-level driver; `PynqBackend` (KV260), `ModelBackend` (dry run) and `MockMmioBackend` (laptop host overhead), same API; host path `safe` (default) or `fast` (below) |
| `mock_mmio.py` | MockMMIO: numpy arrays standing in for the mapped PL windows (pynq-MMIO-like read/write + fast views), golden-logit job lookup; **laptop host-overhead measurements only** |
| `exp_host_overhead.py` | safe vs fast host path per-image phases on the MockMMIO (laptop), → `results/dryrun/host_overhead_mock.csv` |
| `gos_sim.py` | simulated CSR + memories (FORMATS.md semantics) used by the dry run and the tests |
| `gos_model_backend.py` | dry-run job model: gos_pack unpack + gos_golden + gos_cycle_model (laptop only) |
| `board_common.py` | data-package loading + SHA256 verification, PS dequant, provenance, CSV rules, `infer_loop` |
| `make_board_data.py` | builds `data/<net>/` on the laptop (below) |
| `test_shell.py` | Step 4 shell / memory smoke test (also usable on the real design with `--skip-scratch`) |
| `test_core_smoke.py` | core bring-up: VERSION/BUILD_ID, one LeNet + one CIFAR image bit- and cycle-exact, refused job + soft_reset |
| `exp_a1_accuracy.py` | A1 correctness on all 10k images per net |
| `exp_a2_a3_cycles.py` | A2 latency + A3 model / RTL / accelerator cycles per layer, determinism over images |
| `exp_a4_util.py` | A4 MAC_ACTIVE / cycles vs theoretical |
| `exp_b3_breakdown.py` | B3 host-side phase times, **interleaved conditions** (`--conditions safe fast cpu`: randomized blocks, seed recorded; ≥1000 kept per condition, warm-up discarded; median + 95 % CI) → `hw_b3_breakdown.csv` (safe), `hw_b3_breakdown_fast.csv` (fast, after the fast-path check), `hw_b3_breakdown_cpu.csv` (CPU INT8 reference, 1 thread) |
| `power_log.py` | **B1/B2 power (the only power source):** on-board INA260 SOM-rail (VCC_SOM) logger + protocol (B1: accel / control / cpu, each bracketed by idle, seeded random order per repeat, ×3 + idle), energy two ways (E_sys, E_comp at the read-back clock) + duty cycle, sensor check, sensor probe |
| `exp_b1_power.py` | B1 entry point: runs the `power_log.py` protocol for each net (thin wrapper) |
| `exp_b2_clock.py` | B2 pl_clk0 sweep (≤ closed clock): cycles == model, latency, INA260 idle/accel/idle per clock, cycle identity + power-vs-clock fit (other owner; see "Publication extras") |
| `exp_fclk_cal.py` | **PL clock cross-check** f_meas (not the clock of record): cycle counter vs CLOCK_MONOTONIC_RAW, differential LeNet/CIFAR lower-envelope estimator with dithered poll phase, fold t-interval, flag if > 0.1 % off the read-back → `hw_fclk_cal_s<N>.csv` |
| `stats.py` | warm-up discard, median + distribution-free 95 % CI, p95, randomized block order, between-session statistics |
| `board_env.py` | environment pre-flight: cpufreq governor + fixed frequency, pinning, package-manager check, AMS die temperature; fake sysfs/proc trees for laptop tests / dry runs |
| `aggregate_sessions.py` | 3-session repeatability → `hw_repeatability.csv` |
| `clock_fallback.py`, `session_extra_steps.py`, `exp_soak.py`, `exp_layer_spread.py` | publication extras (other owner): 300 → 250 MHz bitstream choice, extra session steps, soak, per-layer cycle spread |
| `session.sh` / `run_sessions.py` | **one command per session** (`1`, `2`, `3`, `all`): pre-flight (+ environment), clock fallback, resumable state, priority order, time budget, per-step timeout (below) |
| `run_all.sh` | superseded: thin wrapper for `session.sh 2` (B1/B2 are in Session 3) |
| `deploy.sh` | rsync this directory + data + bitstream to `<user>@<host>:~/gos/`, write `DEPLOY_INFO.json` |
| `tests/` | pytest: driver vs the simulated register map; session orchestrator (resume, interrupted step, budget, pre-flight, B2 sweep, power steps); power logger (sensor backends with fake sysfs / smbus, read-only I2C, ΔP / energy arithmetic, dry-run rules); measurement rigor (`test_rigor.py`: statistics, governor / pinning / package manager / temperature on fake trees, f_meas estimator on a fake device + clock, priority, extra steps, clock-fallback wiring, `$GOS_RUN_ENV` in rows, aggregation) — no hardware |
| `cpu/` | CPU baselines (A5), separate owner; contract below |

## Prerequisites

- **Board:** KV260 with Ubuntu 22.04 + Kria-PYNQ (pynq 3.x, its Python with numpy); user with
  passwordless sudo. **One command: `./session.sh ...`** from a plain login shell in `~/gos` — the
  script re-executes itself under `sudo -E` if it is not root and sources
  `/etc/profile.d/pynq_venv.sh` as root (PYNQ venv `python3`, `XILINX_XRT`, and the venv `bin` on
  PATH, where pynq finds `xclbinutil` when it loads an overlay). Single scripts run the same way:
  `./session.sh py <script.py> [args]`. Do **not** use `sudo -E python3 ...` / `sudo -E ./...`
  with a sourced venv: sudo's `secure_path` drops the venv from PATH, so `python3` is the system
  one ("No module named 'pynq'") and, with only the interpreter fixed, the overlay load fails on a
  missing `t.xclbin` (found at bring-up, 2026-09-30). The board needs no git checkout.
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
v2/board/deploy.sh <board-ip>            # ships gos_300 + gos_250 (+ .hwh, .sha256, summary.json); prints the run commands
```

### On the board: one command per session (`ssh ubuntu@<board-ip>`, `cd ~/gos`)

Run inside `tmux` (an ssh drop then does not stop the run; if the orchestrator does receive
SIGHUP/SIGTERM it stops the running step cleanly and saves the state). `./session.sh` sets up
sudo and the PYNQ environment itself (Prerequisites); nothing has to be sourced first.

```bash
./session.sh 1 --plan                       # pre-flight + plan only, runs nothing
./session.sh 1                              # Session 1: bring-up           (~5 min)
./session.sh 2 --quick --results-dir results/quick   # optional first pass (200 images)
./session.sh 2 --budget-min 120             # Session 2: A1-A4, B3, CPU
./session.sh py power_log.py --list-sensors        # before Session 3: which INA260 path exists
./session.sh py power_log.py --sample-only --seconds 20   # real sensor update rate (value changes)
./session.sh 3                              # Session 3: B1 INA260 power (both nets) + B2 sweep + soak
./session.sh all --budget-min 240           # or everything in one go
# repeatability: the whole campaign again on two other days, then combine
./session.sh all --session-index 2          # -> results/rep2/
./session.sh all --session-index 3          # -> results/rep3/
python3 aggregate_sessions.py                       # -> results/hw_repeatability.csv
# interrupted / out of time?  run the SAME command again: verified steps are skipped
# new bitstream / data / scripts?  ./session.sh all --fresh   (old outputs archived)
```

| session | steps (ids) | what |
|---|---|---|
| 1 | (clock fallback) `s1.shell`, `s1.smoke`, `s1.smoke_slice`, `s1.fast`, `s1.fcal` | before the steps, with several deployed bitstreams: `clock_fallback.choose` (pre-flight — including pl_clk0 set to the closed clock and read back == it — + core smoke of the 300 MHz build, else the 250 MHz build; choice, per-attempt read-back clock and failure reason in `hw_clock_choice.json` and the state, used by every later step and session); `s1.fcal` = PL clock calibration; `test_shell.py --skip-scratch` (pl_clk0 read back, VERSION/BUILD_ID, every BRAM filled + read back); `test_core_smoke.py` (one LeNet-5 + one CIFAR-10 image bit- and cycle-exact, refused job → ERR_CODE rule 3, soft_reset, good job); the same with `--write-mode slice` (**informational**: a failure does not fail the session; use `--write-mode slice` for Sessions 2/3 only if it passed); `s1.fast` = the same smoke on the **fast host path** preceded and followed by `GosDevice.check_fast_path()` (**informational**; its PASS is the bring-up condition for `--host-path fast`, see "Host paths") |
| 2 | `s2.fcal`, `s2.A2A3`, `s2.layerspread`, `s2.A1`, `s2.B3`, `s2.shapes`, `s2.CPU`, `s2.DPU`, `s2.A4` (priority order) | f_meas calibration, A2/A3 all images, A3b per-layer spread, A1 all 10k images per net, B3 1000 kept per condition, **interleaved** safe + CPU (+ fast host path only if `s1.fast` passed), A3-general random shape jobs (only if `data/shapes/` is shipped; hard 15 min cap), CPU baselines `--tag board` (workers pinned), DPU baseline (only if `dpu/dpu_session.py` is deployed; informational), A4 1000 images |
| 3 | `s3.fcal`, `s3.B1.lenet5`, `s3.B1.cifar10`, `s3.B2`, `s3.soak` | B1 = INA260 SOM-rail protocol per net (accel/**control**/cpu each bracketed by idle, seeded random order per repeat, × `--power-repeats` 3 + final idle = 19 phases of `--window-s` 60 s ≈ 20.5 min per net incl. setup; `--no-power-control`: 13 phases ≈ 14.5 min; CPU = `cpu_int8_ref`, 1 thread); B2 clock sweep (LeNet-5): cycles == model + INA260 idle/accel/idle × `--b2-power-repeats` 3 per clock ≈ 9.5 min per clock (+ cycle identity, power-vs-clock fit); soak 30 min |

Options: `--budget-min N`, `--resume` (default) / `--fresh`, `--quick` (200 images per step, B3
`--n 200`, CPU `--quick`, 30 s windows), `--plan`, `--steps s2.A1 s3.B2` (subset),
`--step-timeout-min M`, `--write-mode {elem,slice}`, `--host-path {safe,fast}` (A1-A4, B1, B2; default
safe; fast requires `s1.fast` OK in the same state, else those steps are BLOCKED = failed, nothing
run), `--fast-store {block,words32}`, `--no-power-control` (B1 without control phases), `--window-s/--gap-s` (B1/B2 INA260 phase
length / B2 gap, default 60/10 s; `--quick` 30 s), `--power-repeats`, `--b2-power-repeats`
(default 3 each), `--power-rate-hz` (default 10), `--b2-images`, `--fcal-s` (calibration wall time,
default 8 s), `--no-b3-cpu`, `--session-index {1,2,3}` (results of 2/3 in `results/rep<K>/`),
`--meas-cores` (default 3), `--cpu1-cores` (3), `--cpun-cores` (0-3), `--cpu-freq-khz` (default the
highest available), `--allow-non-paper-grade` (environment FAILs become warnings, rows
paper_grade=False), `--bit` (one bitstream, no fallback), `--allow-dirty` (rows git_dirty=True,
invalid for the paper), `--no-bringup-check`, `--results-dir` (default `results/`). Exit codes: 0 all OK, 1 a step failed, 3 pre-flight failed,
4 provenance changed / Session 1 not recorded / another run holds the lock, 5 incomplete (deferred),
130 interrupted.

**Pre-flight** (every invocation, before anything runs, also before every resume; any failure →
exit 3, nothing run): `DEPLOY_INFO.json` present; SHA256 of `bit/<name>.bit` == DEPLOY_INFO
`bit_sha256` == the shipped `.bit.sha256`, `.hwh` == `hwh_sha256`; `bit/summary.json` build id and
closed clock == DEPLOY_INFO, timing met (WNS ≥ 0, no failing endpoints); overlay loads; VERSION ==
0x474F5302; BUILD_ID register == DEPLOY_INFO `build_id`; pl_clk0 **set to the closed clock after
the overlay load and read back == it within ±0.1 MHz, in every session** (DECISIONS D19; a clock
the board's PLLs cannot reach fails the pre-flight, nothing is written); every data file vs `MANIFEST.json`, each
`MANIFEST.json` vs `PACKAGE.json`, `PACKAGE.json` vs DEPLOY_INFO; ≥ 1000 MB free; clean-tree flags
(scripts + data). Sessions 2/3 also require Session 1 recorded OK in the same state.
**Environment pre-flight** (`board_env.py`, before the device checks): no package manager
running (offenders listed; `sudo systemctl stop unattended-upgrades packagekit`), cpufreq governor
`performance` + one fixed frequency on every policy with `scaling_cur_freq` read back (restored on
exit), the orchestrator pinned to the housekeeping cores and every step to `--meas-cores` (read
back per step), AMS die temperature at the start/end of every step (`logs/env_log.jsonl`, state,
`die_temp_start_c` column). A FAIL aborts (exit 3) unless `--allow-non-paper-grade`. Details and
the paper-grade rule: EXPERIMENTS.md "Measurement rigor".

**State / resume** (`results/session_state.json`): provenance (backend, BUILD_ID, bit SHA256,
closed + read-back clock, data package SHA256, scripts commit) + per step: status, command,
parameters, start/end, duration, exit code, every output file with SHA256, the step log
(`results/logs/<step>.log`; orchestrator log `results/logs/run_sessions_<ts>.log`). A step is
skipped only if it finished OK with the same parameters and all its outputs still verify.
Anything else (failed, timed out, interrupted, killed by a power cut, output missing or edited,
parameters changed) is **rerun from scratch** — there is no per-image-chunk resume, so every CSV
is one uninterrupted run with one bitstream and one clock. A different bitstream / BUILD_ID /
clock / data package / scripts commit refuses to resume (exit 4) until `--fresh`. Steps write to
`results/.staging/<step>/` and are moved into `results/` when the step process exits; timed-out or
interrupted steps never leave partial files among the results. `--fresh`, replaced outputs and
partial outputs are **archived** into `results/archive/<timestamp>_<why>/`, never deleted
(`check_results.py` only reads `v2/results/*.csv`, not the archive). One run per results dir at a
time (`results/.session.lock`, also held by an orphaned step process).

**Budget / timeouts:** before each step its duration is estimated — the recorded duration of the
same step and parameters, else the per-image rate learned from steps already run in this state,
else a default (`run_sessions.DEFAULT_RATE_S`: board 20 ms/image **assumed**; power steps: phases ×
window exact + a fixed setup allowance — B1 (4R+1) × window + 60 s per net, B2 3R × window + 20 s per
clock)
— and a step that does not fit in the remaining `--budget-min` is **deferred** (reported, recorded
as `deferred`, picked up by the next run; a later, smaller step may still run). Hard per-step
timeout: `--step-timeout-min` or max(5 min, 3 × estimate + 2 min), capped by the remaining budget;
the step's process group gets SIGINT (B2 restores pl_clk0 in its `finally`), then SIGTERM, then
SIGKILL.

**B2 sweep:** 100–300 MHz in 25 MHz steps (9 points on the 300 MHz bitstream, pl_clk0 set at runtime via PYNQ; the per-clock INA260 protocol makes B2 ≈ 9 × 9.3 min ≈ 84 min at default settings)
(`board_common.b2_sweep_clocks`), capped at the bitstream's closed clock (DEPLOY_INFO
`bit_clock_mhz` = summary.json `pl_clk0_mhz_actual`): the top point is requested at the closed
clock itself (e.g. 199.998001, 249.997498), never above; a read-back above closed + 0.5 MHz aborts
the sweep. Each row records requested, read-back and closed clock.

**Session 3 power (INA260 SOM-rail, the only power source):** `run_sessions.power_hook_steps` (block
`POWER HOOK`) schedules one `power_log.py --protocol --net <net> --tag _<net>` step per net
(`s3.B1.lenet5`, `s3.B1.cifar10`); `s3.B2` calls the same logger per clock
(`power_log.run_power_protocol`, reduced protocol idle/accel/idle without CPU or control phases —
the CPU baseline does not depend on pl_clk0). Label of every number: **"SOM-rail power (INA260)"**
— the SOM rail VCC_SOM as reported by the on-board INA260; it is **not** accelerator-only power and
**not** board input power (it includes the PS running the Python driver / CPU baseline; coverage
below). B1 phases per repeat: idle_pre → **accel** → idle_mid → **control** → idle_ctl → **cpu**,
× 3, + idle_post (19 × 60 s). Accel loop per image: `infer` (host path as selected) + TOTAL_CYC
lo/hi read + PS dequant/argmax. **Control** = the same host loop with the accelerator **not
started** (`GosDevice.control_step`): input write, the soft_reset + clear poll that `infer` issues
before every start, a STATUS poll spin for the median start→done time of the preceding accel phase
(calibrated on 20 warm-up images before the first phase), LOGIT read, TOTAL_CYC read, dequant;
CTRL.start is never written and STATUS ≠ 0 / TOTAL_CYC ≠ 0 aborts the step. Per phase: mean, std,
n samples, achieved rate, max gap, duration, images (+ TOTAL_CYC median/min/max and start→done
median for accel, pacing for control); per repeat and mean/std over repeats: P_idle (mean of the
two idle phases bracketing the run phase), ΔP = P_run − P_idle, time/image = phase duration /
images, energy/image = ΔP × time/image; accelerator energy two ways: **E_sys** = ΔP_accel ×
time/image (host loop included), **E_comp** = ΔP_accel × TOTAL_CYC / f (TOTAL_CYC = hardware
counter median, f = the pl_clk0 PLL read-back), **duty** = (TOTAL_CYC / f) / time/image, and the
control-subtracted **ΔP_accel − ΔP_control** with E_sys,net / E_comp,net — all computed by
`power_log.py` (`energy_rule` column), never typed. CPU phases use `cpu_int8_ref` with 1 thread (same INT8 arithmetic as the
accelerator, A5's single-thread configuration). Sensor path (`--sensor auto`): hwmon `ina260*`
→ `platformstats -p` → read-only I2C (smbus2, ID-checked; never writes a register); none found →
the step fails (no fallback). **First run `power_log.py --sample-only`** on the board: it prints
the achieved rate and how often the value actually changed — hwmon `update_interval` and the
INA260 averaging/conversion time limit the real bandwidth, so a 10 Hz sample stream may repeat
values. Dry runs use a seeded mock sensor (source=dryrun_model; synthetic, not measurements).

**Clock of record and statistics (DECISIONS D20):** every µs value is converted with the pl_clk0
**PLL read-back** frequency (`f_used_mhz` = `f_readback_mhz`, `f_used_source` = f_readback).
`sN.fcal` (`exp_fclk_cal.py`) measures f_meas once per session as a **cross-check only**; the
orchestrator passes it to every later step (`$GOS_RUN_ENV`), which records it beside the read-back
(`f_meas_mhz` columns) without using it. If |f_meas − read-back| > 0.1 % the step prints a FLAG,
sets `fcal_flag=True` in `hw_fclk_cal_s<N>.csv` and the session summary shows it. Latencies are reported as median
with the distribution-free order-statistic 95 % CI and p95 after warm-up discard (`stats.py`);
compared conditions run interleaved in randomized blocks with a recorded seed.

**Extra steps / priority:** `session_extra_steps.extra_steps(opts)` (if present) adds steps into
their priority group (A3, A1, latency, shapes, baselines, energy, sweep, soak, A4 — after bring-up
and the calibration); a step with a built-in id replaces it. The time budget defers the lowest-priority
steps first.

The individual scripts remain usable by hand (`--nets`, `--limit N`, `--timeout-s`,
`--write-mode`, `--out-dir`), e.g. `./session.sh py test_core_smoke.py --write-mode slice`.
Expected board time is dominated by Python MMIO (per image: input writes 256 / 768 32-bit stores,
~25 CSR reads); the first run records the real per-step durations for the next estimates.

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
| `hw_b3_breakdown.csv` / `hw_b3_breakdown_fast.csv` | per net × phase | safe / fast host path: input write, status clear, start→done (= start write + poll), logit read, PS dequant, end-to-end, counter read, PL compute (median, p5, p95, p99), `host_path`, `fast_store` |
| `hw_b1_power_ina260_{samples,phases,summary}_{lenet5,cifar10}.csv` | per sample / per phase / per repeat + mean + std | **B1 primary**, "SOM-rail power (INA260)": P_idle, P_accel, P_control, P_cpu, ΔP, time/image, energy/image (J, mJ); accel E_sys, t_PL = TOTAL_CYC/f, E_comp, duty, ΔP_accel − ΔP_control, E_sys,net, E_comp,net; host path; sensor backend/device/limits, requested + achieved rate, max gap |
| `hw_b2_clock.csv` | per clock | requested/read-back clock, cycles (must equal the model), latency, INA260 P_idle/P_accel/ΔP/energy per image (mean/std over repeats) |
| `hw_b2_power_ina260_{samples,phases,summary}_lenet5_<NNN>mhz.csv` | per clock (NNN = rounded requested clock, e.g. 200mhz) | B2 INA260 idle/accel/idle protocol at that clock |
| `hw_fclk_cal_s<N>.csv` (+ `.npz`) | one per session | f_req, f_readback (clock of record), f_meas cross-check + 95 % fold t-interval, fold min/max, rel. difference, `fcal_flag` (> 0.1 %), former median estimator, f_simple, per-net jobs / minima / medians / intercepts, poll period, dither, seed |
| `hw_b3_breakdown_cpu.csv` | per net × phase | B3 CPU condition (cpu_int8_ref, 1 thread) interleaved with the accelerator |
| `hw_clock_choice.json` | one | clock_fallback decision (bit, closed clock, fell_back, attempts) |
| `hw_repeatability.csv` | per metric × key × bitstream × clock | `aggregate_sessions.py`: mean ± between-session SD, range, CV, n sessions |

Metadata columns on every row: EXPERIMENTS.md "CSV rule" (`timestamp, git_commit, git_dirty,
vivado_version, bitstream_sha256, board_id, net, layer, clock_mhz, source, duration_s,
num_inferences`) + `build_id_hw` (BUILD_ID register), `board_hostname`, `clock_source`
(`clock_mhz` = pynq `Clocks.fclk0_mhz` read back on the board), `scripts_commit`,
`data_manifest_sha256`, `data_git_commit`, `data_git_dirty`, `backend`, and the measurement
environment `session_index`, `paper_grade`, `env_step`, `cpu_governor`, `cpu_freq_khz`,
`cpu_affinity` (read back in the step process), `die_temp_start_c`, `env_note`. `git_commit` /
`scripts_commit` come from `DEPLOY_INFO.json` on the board (git on the laptop); `source` is `hw`
or `dryrun_model`; `bitstream_sha256` is computed from the loaded `.bit`.

### Dry run on the laptop (no board)

```bash
cd v2/board
../../.venv/bin/python -m pytest tests -q
./session.sh all --backend model --allow-dirty      # sessions 1-3, all images, 2 s windows,
                                                    # fake sysfs/proc tree, gos_300 -> gos_250 choice
./session.sh all --backend model --allow-dirty      # again: every step verified and skipped
./session.sh all --backend model --allow-dirty --fresh --dryrun-fail-smoke-mhz 300   # fallback drill
./session.sh 3 --backend model --allow-dirty --bit ../vivado/out/gos_250/gos_250.bit --fresh \
    --results-dir ../results/dryrun/b250 --no-bringup-check    # sweep as a 250 MHz build would
```
The dry-run pre-flight uses the ModelBackend equivalents (bitstream SHA256 vs its `.bit.sha256`,
summary.json timing, simulated VERSION/BUILD_ID set from summary.json, nominal clock = the
closed clock, data package, disk, clean flags) and prints DRY RUN on every line.
`ModelBackend` runs the same driver code against `gos_sim` (FORMATS.md register semantics); on
CTRL.start it decodes the descriptors actually written, runs the config-checker model, unpacks
the ACT/WGT/QPARAM images actually written, computes every layer with `gos_golden.gos_layer`
and the counters with `gos_cycle_model`. Dry-run times (B3, B1 inference rates) are the laptop
simulator's and mean nothing.

## Host paths (`--host-path safe|fast`)

| | safe (default, verified) | fast |
|---|---|---|
| windows | pynq `MMIO` per access | mapped once: pynq `MMIO.array` numpy views (or an own `/dev/mem` mmap, `DevMemWindow`, if a pynq version has no `.array`); `backend.fast_windows()` |
| input image → ACT0 | one 32-bit numpy element store per word (`--write-mode elem`; `mmio` = `MMIO.write`; `slice` = numpy block copy) | ONE vectorized numpy copy of the whole image (`--fast-store block`: contiguous copy, store width chosen by numpy/libc; `words32`: lo-word and hi-word strided copies, numpy's strided 4-byte loop = 32-bit stores) |
| LOGIT | `OC` × `MMIO.read` | ONE vectorized read of all 16 LOGIT words (0x080-0x0BF: 64 B, 64-B aligned, no read side effects), sliced to `OC` |
| CSR scalars (CTRL, STATUS, counters) | `MMIO.read/write` | `memoryview` item access on the mapped CSR window (CPython packs/unpacks a fixed 4-byte item: one 32-bit load/store) |
| STATUS poll | `MMIO.read` + clock read per poll | tight loop, clock read every 64 polls |
| counters | lo then hi (latch) | unchanged: ordered scalar reads, lo then hi (never a block read: a block copy does not guarantee the load order); LAYER_CYC[0..7] as one block read (no side effects) |
| WGT/QPARAM/DESC (once per net) | per word + readback | unchanged (safe path) |

Protocol, error handling and API are identical (`GosDevice(..., host_path="fast")`, same `infer()`,
`InferResult` gains `t_start_ns` = the CTRL.start write alone). **The default stays `safe`** until
the board check passes: a numpy block copy / vectorized read on Device memory (`/dev/mem`, mapped
uncached) is only safe if every access is naturally aligned and the AXI slaves accept the widths
numpy/libc choose — an unaligned or unsupported access can raise SIGBUS or corrupt data. Therefore:

1. Session 1 step **`s1.fast`** (`test_core_smoke.py --host-path fast`) runs
   `GosDevice.check_fast_path()`: 3 patterns (seeded random, its complement, index hash) over the
   **whole** ACT0 window (4096 × 64 bit) written with the fast copy and **every word read back**
   through the safe per-word `MMIO.read` and through a fast block read; LOGIT[0..15] and
   LAYER_CYC[0..7] block reads compared with per-word reads (again after a real job, when they are
   non-zero); memoryview scalar reads == `MMIO.read`; then both nets bit- and cycle-exact on the
   fast path. Any mismatch or exception → FAIL (informational step: the session goes on, safe).
2. Only with `s1.fast` recorded OK in the same state do `--host-path fast` steps run
   (`requires=("s1.fast",)`); otherwise they are **BLOCKED** (recorded, exit 1, nothing run) — rerun
   with the default safe path. `--no-bringup-check` overrides (not for paper runs).
3. `exp_b3_breakdown.py --host-path fast` repeats the ACT0/CSR check before measuring and writes
   nothing if it fails; `s2.B3fast` always measures the fast path (when `s1.fast` passed) next to
   the safe `s2.B3`, so B3 reports both.
4. If `block` fails but you want to try `words32`: `./session.sh py test_core_smoke.py --host-path
   fast --fast-store words32`, then `session.sh ... --fast-store words32` (the steps' parameters
   change, so `s1.fast` reruns with it).

**Laptop measurement (MockMMIO, NOT the KV260):** `python3 exp_host_overhead.py [--fast-store
block|words32]` runs both paths interleaved per image on `mock_mmio.MockMmioBackend` (numpy arrays
as windows; pynq-like `read/write` for the safe path; the fast path's CSR scalars go through Python
methods on the mock, memoryview on the board) → `results/dryrun/host_overhead_mock.csv`, label
"host overhead, laptop mock (MockMMIO), not KV260", `source=dryrun_model`. It shows the Python/
numpy cost only (zero bus latency); start_write/poll/status_clear also include the mock's register
emulation (`mock_emulation_us_*`, identical on both paths). Board numbers come from B3.

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
  (called by `session.sh 2` (step `s2.CPU`) with `--tag board`, or `--tag laptop` +
  `results/dryrun/` in a dry run).
- `cpu/cpu_infer.make_runner(kind, net, data_dir, threads)` returns `runner(x)` used by
  `power_log.py` (B1 CPU phases) and `exp_b1_power.py --modes cpu`: x = `x_nchw[i]` (int8) for
  `cpu_int8_ref`, else `x_f32[i]`
  (float32); BLAS/OpenMP thread variables are set from `--cpu-threads` before numpy is imported.
- The CPU scripts read this data package (and their own `data/<net>/cpu/` files).

## Assumptions (to confirm at bring-up) and open items

- **pl_clk0 (confirmed at bring-up 2026-09-30, DECISIONS D19):** PYNQ does not apply the Vivado PS
  PLL settings; after an overlay load pl_clk0 is whatever the `.hwh` dividers give on the boot
  image's PLLs (gos_300 came up at 199.998 MHz). The driver therefore sets pl_clk0 to the
  bitstream's closed clock (`summary.json`) and verifies the read-back within ±0.1 MHz
  (`gos_driver.set_fclk0_exact`; never pynq's `Clocks.fclk0_mhz = x`, which picks the closest
  reachable frequency — 333.33 MHz for a 300 MHz request). With the boot image's PLLs (PL0 source
  IOPLL = 999.99 MHz) 250 MHz is reachable and 300 MHz is not: gos_300 fails its pre-flight and
  Session 1 falls back to gos_250. The B2 sweep grid is limited the same way (open, before Session 3).
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
- **Power sensor (`power_log.py`):** the INA260 hwmon node (`/sys/class/hwmon/*/name` starting
  with `ina260`, `power1_input` µW, `curr1_input` mA, `in1_input` mV, `update_interval` ms), else
  `platformstats -p` (format assumed; matched raw line recorded), else read-only I2C via smbus2
  (0x40 first, then 0x41–0x4F; Manufacturer ID 0xFE = 0x5449, Die ID 0xFF = 0x2270; power 0x03
  LSB 10 mW, current 0x01 1.25 mA signed, bus voltage 0x02 1.25 mV; config 0x00 decoded, never
  written). Rail coverage: see "VCC_SOM coverage (INA260)" below.
- **board_id:** `--board-id` / `$GOS_BOARD_ID`, else device-tree model + first 8 chars of
  `/etc/machine-id`.
- **Clock sweep:** runtime pl_clk0 changes via PYNQ are assumed to work (EXPERIMENTS B2 says to
  verify at bring-up; fallback: one bitstream per clock). After each change the driver
  soft-resets and reloads + reads back WGT/QPARAM/DESC.
- **cpufreq / AMS paths (to confirm at bring-up):** `/sys/devices/system/cpu/cpufreq/policy*`
  (cpufreq-dt; the target is the highest `scaling_available_frequencies` entry), AMS temperatures
  under `/sys/bus/iio/devices/iio:device*/` (name containing "ams": `in_temp*_raw` × scale + offset)
  or hwmon; the pre-flight prints what it found.
- **Open:** per-layer MAC_ACTIVE is not measurable (no HW counter).

## Publication extras (other owner: `session_extra_steps.py`, `clock_fallback.py`, `exp_soak.py`, `exp_layer_spread.py`, `exp_b2_clock.py`)

- **`s2.shapes` — A3-general** (`exp_shapes.py` → `hw_shapes.csv`, group `shapes`, scheduled right
  after `latency` (A2 + B3); requires `s1.smoke` (+ `s1.fast` with `--host-path fast`); registered
  only when the data package has `data/shapes/SHIP.json`). The seeded random multi-layer shape set
  (`v2/shapes/gen_shapes.py` → `v2/build/shapes/<seed>/`, loader `shapeset.py`, other owner) is
  run job by job: soft_reset; WGT from `wgt_base`, QPARAM from combined word `2*qp_base`, ACT0 =
  `act_in` (host path as selected), each read back word by word (`--no-verify-writes` skips the
  memory readback); DESC + N_LAYERS written and read back; start; STATUS poll (`--timeout-s`);
  LAYER_CYC / TOTAL_CYC / MAC_ACTIVE / STALL / PS_BUSY_VIOLATION / ERR_CODE; output read back per
  word from ACT`out_buf` and compared bit-exact on the bytes enabled by `out_mask` (bit b = byte
  lane b; cross-checked with `Shape.compare_output`), or LOGIT[0..n_logits-1] for `out_raw` jobs.
  Refuse jobs (N_LAYERS may be 0 or 9): STATUS.error, ERR_CODE == expected, TOTAL_CYC == C_START.
  Cycles vs the model (per layer, total, MAC_ACTIVE; STALL == 0) and vs the RTL simulation row of
  the same `job` in the shipped `shapes_rtl.csv` (compared only when its `shapeset_sha256` equals
  the shipped set's). **Time cap:** `--budget-s` (default = hard maximum 900 s, from script start;
  `--quick` 300 s): before each job the script stops if elapsed + the longest job so far would
  exceed it and records `capped=True`, `jobs_completed` / `jobs_total` in the summary row. **Job
  order:** seeded shuffle within each category, round-robin over the categories (`--order-seed`,
  the session passes state seed + 30), so a capped run still covers every category. Step
  estimate = 60 s + jobs × 1.5 s (board, **assumed**; the recorded duration replaces it), capped
  at the budget; step timeout = budget + 180 s. Rows: one per job + `layer=summary` (jobs,
  outputs exact, refuse OK, cycles == model, cycles == RTL / compared, max |err|, categories
  covered); label "A3-general random shapes, measured on KV260" (dry run: model backend).
  Dry run: `python3 exp_shapes.py --backend model --out-dir ../results/dryrun/shapes`.

- **`s2.layerspread`** (`exp_layer_spread.py` → `hw_layer_spread.csv`, group A3, scheduled right after
  `s2.A2A3`): per net × layer, min / max / distinct / spread of LAYER_CYC (and TOTAL_CYC) over every
  image; reuses `hw_cycles_<net>.npz` from `s2.A2A3` when it has the same source and clock and covers
  all images, else runs every image itself.
- **`s3.soak`** (`exp_soak.py`, group soak): 30 min (`--quick` 5 min), LeNet-5 / CIFAR-10 alternating
  every 60 s, every job bit- and cycle-exact; counts errors, timeouts, STATUS errors and
  PS_BUSY_VIOLATION flags; per-minute throughput; AMS die temperature via `board_env.read_die_temp`
  → `hw_soak.csv`, `hw_soak_minutes.csv`; aborts after > 100 errors; not resumable (one run).
- **Performance-clock choice:** `clock_fallback.py --bits bit/gos_300/gos_300.bit bit/gos_250/gos_250.bit
  --out results/hw_clock_choice.json` (standalone; `session.sh 1` calls `clock_fallback.choose` itself):
  300 MHz first, 250 MHz as fallback, 200 MHz only with `--allow-lower`.
- **B2 additions:** `hw_b2_cycles.csv` (cycle identity across clocks, PASS/FAIL) and `hw_b2_fit.csv`
  (least squares P = P_static + k·f on the per-clock mean SOM-rail power for P_accel, ΔP_accel and
  P_idle, with SE, 95 % CI (t, n − 2), R², n); `exp_b2_clock.py --fit-only` recomputes the fit.

## VCC_SOM coverage (INA260) — what the B1/B2 numbers include

Sources read on 2026-09-29 (text extracted with `pdftotext`; quotes verbatim):

- **AMD UG1089 "Kria KV260 Vision AI Starter Kit User Guide", v1.3, May 14, 2024**, Chapter 2
  "Initial Setup", section "Powering the Starter Kit and Power Budgets" (PDF p. 9-10; copy fetched
  from https://docs.xibif.ch/_downloads/480ec1217d1ac2bd520e695148cfe86f/KV260_Userguide.pdf,
  SHA256 f2bb7a37…8e1763; the same section is on
  https://docs.amd.com/r/en-US/ug1089-kv260-starter-kit/Powering-the-Starter-Kit-and-Power-Budgets,
  whose page header reads v1.4, 2025-06-25 — only the summary of that page could be read, its
  wording matches v1.3):
  - "Powering the K26 SOM: • The KV260 Starter Kit carrier card on-board regulator generates a 5V
    supply and provides power to other voltage regulators. • The SOM power rail (VCC_SOM) is
    powered by the 5V supply. • Next, the SOM on-board power-on sequencing starts. • The carrier
    card provides the programmable logic (PL) the VCCO voltage rails after the SOM asserts the
    VCCOEN_S_M2C and VCCOEN_PL_M2C signals"
  - "Power Telemetry: A power monitor device is available on the SOM power rail (VCC_SOM). You can
    access the total power consumed by the SOM module through the I2C bus and AMD provided
    utilities."
  - UG1089 does **not** name the device, its I2C address, or list the loads on VCC_SOM.
- **AMD/Xilinx DS987 "Kria K26 SOM Data Sheet", v1.0, April 20, 2021 (Preliminary)** (copy
  https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/7171/122_SM-K26-XCL2GC.pdf, SHA256
  3d6da9a7…4e00a1; current revision not accessible here):
  - Overview, "Integrated and flexible power design: ○ SOC power supplies derived from a single +5V
    input ○ PL I/O supplies customized through carrier card defined power rails"
  - "Power Management Signals": "VCCOEN_PS_M2C … to enable the PS VCCO rails that are supplied by
    the carrier card"; "VCCOEN_PL_M2C … to enable the PL VCCO rails that are supplied by the carrier
    card"; "PWROFF_C2M_L is … pulled High to the +5V SOM input power rail"; "The VCCO for MIO banks
    501 and 502 is fixed at 1.8V and is supplied by the K26 SOM."
  - Table 8 (connector legend): "VCC_SOM — Both SOM240_1 and SOM240_2 — Power connection pins";
    Table 4 "SOM I2C Interface Addresses" lists on the SOM: DA9062 PMIC (0x30, 0x31), DA9130 (0x32),
    DA9131 (0x33), "PL power domain monitor" (0x68), "PS power domain monitor" (0x70).
  - A search-engine snippet of DS987 "Power Sequencing" (docs.amd.com, not opened) reads "Your
    carrier card supplies the +5V SOM power rail (VCC_SOM)."
- **Upstream Linux device tree of the KV260 carrier card**
  (`arch/arm64/boot/dts/xilinx/zynqmp-sck-kv-g-revB.dtso` and `-revA.dtso`, torvalds/linux master,
  last commit bc7b1759, 2026-06-03): on `&i2c1 /* I2C_SCK C23/C24 - MIO from SOM */`:
  `u14: ina260@40 { compatible = "ti,ina260"; label = "ina260-u14"; reg = <0x40>; };` — the only
  power monitor on the carrier I2C bus.

**Conclusion (what is sourced, what is inferred):**
- *Sourced:* VCC_SOM is the SOM's single +5 V input rail (UG1089, DS987); the monitor on it gives
  "the total power consumed by the SOM module" (UG1089). The PL and PS **VCCO** (I/O bank) rails for
  carrier-side I/O are supplied **by the carrier card**, not from VCC_SOM (UG1089, DS987); MIO banks
  501/502 VCCO (1.8 V) come from the SOM.
- *Inferred, not stated in the documents:* that the "power monitor device" of UG1089 is the INA260
  U14 at 0x40 (the device tree names no other monitor on the carrier bus); that VCC_SOM therefore
  covers every on-SOM load — PS (APU running the Python host and the CPU baseline, LPD/FPD), PL core
  rails (the accelerator), the on-SOM 4 GB DDR4, eMMC/QSPI/TPM, and the PMIC/regulator conversion
  losses — since the SOM has no other supply input apart from VCC_BATT (RTC) and the carrier VCCO
  rails. DS987 v1.0 does not break VCC_SOM down per load.
- *Not covered:* carrier-side regulators and their losses (12 V → 5 V), carrier peripherals (USB,
  Ethernet PHY, fan, …), PL/PS VCCO supplied by the carrier (this design drives no PL I/O banks),
  so VCC_SOM ≠ board input power.
- **Open questions:** (1) confirm U14 = the VCC_SOM monitor from the KV260 carrier schematic
  (not accessed); (2) the current DS987 revision / UG1091 per-rail budget (not accessible here);
  (3) the SOM's "PL power domain monitor" (0x68) / "PS power domain monitor" (0x70) could split PL
  from PS power — device type and access are undocumented here; not used.

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
(`data_manifest_sha256` in every results row), and `shapes.ship_sha256` when a shape set is shipped.

**`data/shapes/` (A3-general, optional):** `make_board_data.py --shapes auto|none|<seed>|<dir>`
(default auto = the only `v2/build/shapes/<digits>/` holding `shapes.json`; several → pass the
seed; `--shapes-only` ships the set and rewrites PACKAGE.json without rebuilding the nets) copies
`shapes.json` + every listed `jobs/*.npz` (not `hex/` or `runs/`) and, if present,
`v2/results/shapes_rtl.csv` (RTL simulation, label rtl_sim), and writes `SHIP.json` (SHA256 + size
of every shipped file, seed, n_jobs, categories, the set's `shapeset_sha256`, set and ship git
provenance; `git_dirty` = ship tree dirty or set generated from a dirty tree). `exp_shapes.py`
refuses a set whose files differ from SHIP.json, whose SHIP.json SHA256 is not the one in
PACKAGE.json, or that fails `shapeset.verify_manifest`; `deploy.sh` runs `exp_shapes.py verify`
before copying (no `data/shapes/` → warning, `s2.shapes` not scheduled). `data_manifest_sha256`
of `hw_shapes.csv` rows = the SHIP.json SHA256.

Python access (board or laptop): `from board_common import load_package; pkg = load_package(data_dir, "lenet5")`
→ `pkg.x_act`, `pkg.x_nchw`, `pkg.x_f32`, `pkg.labels`, `pkg.golden_logits`, `pkg.golden_pred`,
`pkg.wgt`, `pkg.qparam`, `pkg.desc`, `pkg.net` (net.json), `pkg.dequant`, `pkg.model_cycles`,
`pkg.manifest_sha256`.
