# Reproducing the V2 results

Exact commands, in order, from a fresh clone, to regenerate every V2 number. This file holds
**no result numbers**. Every number comes from a script that writes `v2/results/*.csv` with
metadata (v2/CLAUDE.md "Honesty"). Runtimes are given as "see `duration_s` in X.csv" or as the
`runtime` rows of `v2/results/verification_stats.csv`, which are computed from those columns.

Each number carries a label: **model**, **RTL sim** (`source=rtl_sim`), **post-synth**
(`post_synth_ooc`, `post_synth_funcsim`), **post-impl** (`post_impl`), or **hw**, meaning
measured on the KV260 (`source=hw`, `cpu_board`). The `source` column of each row gives its label.
`dryrun_model` and `cpu_laptop` rows under `v2/results/dryrun/` are never paper data.

## 0. Rules that apply to every step

- **Clean tree.** Commit code, run the result scripts, then commit the CSVs in a separate
  commit. Rows with `git_dirty=True` are invalid for the paper. `regen_results.sh`,
  `gen_vectors.py --require-clean`, `make_board_data.py` and the board scripts refuse to run on a
  dirty tree (the board scripts accept `--allow-dirty`, which marks the rows invalid).
- **Provenance (DECISIONS D16).** A row is valid when `git_dirty=False` and `v2/rtl`,
  `v2/vivado` and `v2/model` at the row's `git_commit` are identical to HEAD.
  `v2/scripts/check_results.py` checks this (step 9).
- **Vivado resource rule (v2/CLAUDE.md).** Run **one Vivado job at a time**: synthesis, OOC,
  netlist or implementation. Never run two in parallel or from concurrent shells or agents.
  Every Vivado entry script (`ooc_all.sh`, `run_netlist_sim.sh`, `vivado/build_gos.sh`,
  `vivado/build_shell.sh`) first calls `v2/scripts/vivado_guard.sh`. The guard prints
  `free -h` and refuses to start if another `vivado` process is running or if available memory
  is below `GOS_MIN_AVAIL_GB` (default 8 GB). Every Tcl sets `general.maxThreads 8`, and
  `bd_shell.tcl` forces `-jobs 1`. xsim runs (`xvlog`/`xelab`/`xsim`) are not Vivado jobs.

## 1. Environment

| item | how |
|---|---|
| OS | Ubuntu 22.04 LTS 64-bit (project CLAUDE.md) |
| Vivado | Vivado ML 2023.1 at `~/Xilinx/Vivado/2023.1`. Every sim/Vivado script sources `~/Xilinx/Vivado/2023.1/settings64.sh` itself. For manual commands run `source ~/Xilinx/Vivado/2023.1/settings64.sh`. Part `xck26-sfvc784-2LV-c`. |
| Python | CPython 3.10 venv at the repo root: `.venv` (every script uses `<repo>/.venv/bin/python`) |
| board | KV260, Ubuntu 22.04 + Kria-PYNQ (pynq 3.x, numpy). See `v2/board/README.md` "Prerequisites". |

```bash
git clone <repo-url> cnn-accelerator && cd cnn-accelerator
git checkout v2-dev

python3.10 -m venv .venv
.venv/bin/python -m pip install numpy==1.26.4
.venv/bin/python -m pip install torch==2.1.2+cpu torchvision==0.16.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install onnx==1.16.1 onnxruntime==1.19.2 matplotlib==3.10.9 pytest==9.1.1
.venv/bin/python -m pip freeze        # record; compare with the pins above
```

The pins match the working `.venv` (`pip freeze`) and `python/requirements.txt` (numpy, torch,
torchvision, onnx, matplotlib). onnxruntime is needed only for the CPU-baseline export
(`v2/board/cpu/`), and pytest only for the tests. See "Gaps" below for the missing V2
requirements file.

**Data sets.** MNIST and CIFAR-10 go under `data/raw/`, which is gitignored. Every V2 script
loads them with `download=False`, so fetch them once:

```bash
.venv/bin/python - <<'EOF'
from torchvision import datasets
for tr in (True, False):
    datasets.MNIST("data/raw", train=tr, download=True)
    datasets.CIFAR10("data/raw", train=tr, download=True)
EOF
```

This creates `data/raw/MNIST/raw/` and `data/raw/cifar-10-batches-py/`, the same layout as the
legacy `python/lenet5/preprocess.py` and `python/cifar10/preprocess.py`. The checkpoints and
frozen INT8 sets are tracked: `data/checkpoint/lenet5_fp32.pt`, `data/checkpoint/cifar10_fp32.pt`
(r1), `v2/model/retrain/cifar10_fp32_r2.pt` (r2, the reference of record; sha256 in
`cifar10_fp32_r2.pt.sha256`), and `v2/model/frozen/*/`.

Optional: retrain CIFAR-10 r2 and refreeze it. This is **not** needed to reproduce the results.
The commands are in `v2/model/retrain/README.md`. `cifar10_retrain_log.csv` is a training
artifact tied to the checkpoint SHA256 (DECISIONS D9) and cannot be regenerated without
retraining.

## 2. Model results: `v2/scripts/regen_results.sh` (label: model)

```bash
v2/scripts/regen_results.sh           # refuses a dirty tree; run from anywhere
```

Runs, in `v2/model`: `reference_accuracy.py`, `requant_check.py --nets lenet5 cifar10`,
`final_layer.py`, `gos_cycle_model.py`, `golden_crosscheck.py`, `retrain/check_r2.py`. Then it
runs the model analyses in `v2/analysis/` (see `v2/analysis/README.md`). It stops if a frozen
`hw_requant.npz` changes, and it checks that every regenerated CSV is clean at HEAD.

| inputs | outputs (`v2/results/`) | runtime |
|---|---|---|
| `data/raw/`, checkpoints, `v2/model/frozen/` | `reference_accuracy.csv`, `requant_equivalence.csv`, `final_layer_check.csv`, `cycle_model.csv`, `golden_crosscheck.csv`, `cifar10_r2_accuracy.csv`, `cifar10_r2_summary.csv`, plus the `v2/analysis/` CSVs | see `duration_s` in each CSV (empty in `cycle_model.csv` and `cifar10_r2_summary.csv`) |

**Test vectors** (needed by steps 3, 4 and 5; `v2/vectors/generated/` is gitignored and
`MANIFEST.json` is tracked):

```bash
.venv/bin/python v2/scripts/gen_vectors.py --require-clean   # options: --out DIR --manifest PATH --quick
.venv/bin/python v2/scripts/check_vectors.py                 # SHA256 vs v2/vectors/MANIFEST.json
```

**Model tests** (the collected count is a row of `verification_stats.csv`):

```bash
.venv/bin/python -m pytest -q v2/model v2/board/tests v2/scripts/tests v2/analysis/tests
```

## 3. Unit testbenches: `v2/scripts/run_unit_all.sh` (label: RTL sim)

```bash
v2/scripts/run_unit_all.sh
```

This runs `check_vectors.py`, then every `v2/tb/tb_*.sv` whose first line is a
`// GOS_UNIT_TB:` header, one after another, through `run_xsim.sh` in
`v2/build/sim/<tb>/`. `unit_collect.py` then writes **`unit_tb.csv`**. Among these TBs are
`tb_gos_top` and the RTL back-to-back job test `tb_gos_top_backtoback`. Runtime: see
`duration_s` in `unit_tb.csv` (per TB) or the `unit TBs` runtime row of
`verification_stats.csv`. To run one TB: `v2/scripts/run_xsim.sh <tb_name> [rtl files...]`
(env `SIM_TAG`, `XSIM_PLUSARGS`, `VEC_DIR`).

## 4. Core RTL suites: `v2/scripts/run_core.sh` (label: RTL sim)

```bash
v2/scripts/run_core.sh                          # CORE_SUITES="layer net fuzz checker" by default
```

Runs `tb_gos_core` with `SUITE=layer|net|fuzz|checker`, four xsim runs in parallel in
`v2/build/sim/tb_gos_core_<suite>/`. `core_collect.py` then writes **`rtl_cycles.csv`**
(kind = layer / fuzz), **`rtl_network.csv`** (per network image: LOGIT, prediction, cycles) and
**`rtl_checker.csv`**. Runtime: see `duration_s` in each CSV. The suites run in parallel, so
wall time is roughly the largest value (the `runtime` rows of `verification_stats.csv`).

## 5. Post-synthesis netlist simulation: `v2/scripts/run_netlist_sim.sh` (label: post-synth; Vivado job)

```bash
v2/scripts/run_netlist_sim.sh                   # SKIP_SYNTH=1 reuses v2/build/netlist/
                                                # NETLIST_BTB="NJOBS=3 J_RESET=1 J_REFUSE=-1" (default)
```

This runs `vivado_guard.sh` and `vivado/netlist_synth.tcl`, which writes the funcsim netlist
to `v2/build/netlist/`. `synth_scan.py` scans the synthesis log and writes
`build/netlist/synth_scan.txt`. xsim with UNISIM then runs `tb_gos_top` and
`tb_gos_top_backtoback`, and `netlist_collect.py` writes **`rtl_netlist.csv`**
(`source=post_synth_funcsim`). Runtime: not recorded (`duration_s` is empty). Gate-level xsim is
slow; see DECISIONS D15-7.

## 6. Out-of-context synthesis: `v2/scripts/ooc_all.sh` (label: post-synth; Vivado jobs, run serially)

```bash
v2/scripts/ooc_all.sh [gos_array gos_core ...]  # default: all 9 leaf/core modules
```

This runs one `vivado/ooc_synth.tcl` per module in `v2/build/ooc/<top>/`, one at a time and
each behind the guard. `ooc_collect.py` then writes **`ooc_synth.csv`**
(`source=post_synth_ooc`). Runtime: not recorded (`duration_s` is empty).

## 7. Full-design implementation (label: post-impl; Vivado jobs, run serially)

Run one build at a time. Wait for each to finish before starting the next.

```bash
v2/vivado/build_gos.sh 200                      # baseline (mandatory, D14)
v2/vivado/build_gos.sh 250
v2/vivado/build_gos.sh 300
#   retry of a failing variant (D14): v2/vivado/build_gos.sh <MHz> --strategy explore  -> out/gos_<MHz>_explore
#   other options: --tol <pct> (pl_clk0 tolerance, default 1.0), --bd-only
.venv/bin/python v2/scripts/impl_collect.py --csv v2/results/impl_gos.csv \
    v2/vivado/out/gos_200 v2/vivado/out/gos_250 v2/vivado/out/gos_300
#   add --allow-timing-fail to record a variant with WNS/WHS < 0 (timing_met=False) instead of failing
```

Outputs (gitignored) go to `v2/vivado/out/gos_<MHz>[_<strategy>]/`: `.bit`, `.hwh`, `.sha256`,
utilization, timing, power, methodology, DRC and check_timing reports, `summary.json`, and the
synthesis logs. `impl_collect.py` writes one row per outdir to **`impl_gos.csv`**
(`source=post_impl`, including the bit/hwh SHA256). **`--csv` is required**, because the
default is `impl_shell.csv`. BUILD_ID is the 8-hex short commit. The performance bitstream is
the highest variant with WNS ≥ 0, WHS ≥ 0 and 0 critical warnings (D14).
`power_w_estimate` is a Vivado vectorless estimate, not a measurement. Runtime: not recorded
(`duration_s` is empty).

Step 4 empty shell (history, `impl_shell.csv`): `v2/vivado/build_shell.sh [--bd-only] [--no-collect]`.

## 8. Synthesis-log scan: `v2/scripts/synth_scan.py` (static check, post-synth)

```bash
.venv/bin/python v2/scripts/synth_scan.py <synth.log> [--out report.txt] [--waivers v2/vivado/synth_waivers.txt]
```

Called automatically by `run_netlist_sim.sh` (netlist log) and by `impl_collect.py`, which scans
`synth_1.log` / `synth_gos_top.log` of each outdir. The result goes into the `synth_*` columns
of `impl_gos.csv`. The scan refuses a truncated log (it needs no per-ID message limit), and it
fails on any latch, multi-driven or undriven message and on any unwaived removed-logic group
(DECISIONS D15).

## 9. Provenance check: `v2/scripts/check_results.py`

```bash
.venv/bin/python v2/scripts/check_results.py [--commit SHA] [--verbose]
```

Run this after all producers and before committing the CSVs. It lists each CSV as clean or
STALE/DIRTY, with its row commits, and exits nonzero on any failure.

## 10. Verification statistics: `v2/scripts/verification_stats.py`

```bash
.venv/bin/python v2/scripts/verification_stats.py            # -> v2/results/verification_stats.csv + markdown on stdout
.venv/bin/python v2/scripts/verification_stats.py --out /tmp/vs.csv [--results-dir DIR] [--no-pytest]
```

This writes one row per statistic, computed only from the CSVs above, plus the model test count
from `pytest --collect-only`. The statistics cover:

- unit TBs: count, checks, pass status
- core suites: layer runs, fuzz shapes, images per net, logit and cycle matches, checker cases
- the RTL back-to-back TB
- netlist sim
- requant equivalence: values checked and mismatches per net and B, with the adopted B read from
  `hw_requant.npz`
- the final-layer check, the golden cross-check and the reference accuracy
- board A1, shown as "pending" until `hw_a1_accuracy.csv` exists

Each row carries its label, its source CSV and the `git_commit`s it used. Rows with
`git_dirty=True` are counted separately (`value_dirty`, `rows_dirty`) and flagged `DIRTY`. The
script exits 1 if any statistic is flagged FAIL. Run it **last** (after step 11 has copied the
board CSVs back) from a clean tree.

## 11. Board (label: hw)

Everything is in `v2/board/README.md`: prerequisites, the workflow, and where the results land.
In order:

```bash
.venv/bin/python v2/board/make_board_data.py            # data package v2/board/data/<net>/ (refuses a dirty tree; --nets --limit --out)
.venv/bin/python v2/board/cpu/export_cpu_models.py      # CPU package v2/board/data/<net>/cpu/ (A5; v2/board/cpu/README.md)
v2/board/deploy.sh <board-ip> [--bit v2/vivado/out/gos_<MHz>/gos_<MHz>.bit] [--user ubuntu] [--print-only]
```

Then run the sessions on the board. Session 1 is bring-up (`test_shell.py`,
`test_core_smoke.py`). Session 2 is A1–A4, B3 and the CPU baselines (`run_all.sh`). Session 3 is
power B1 and the clock sweep B2 with the inline meter. Use **the session runner in
`v2/board/README.md`**, or the individual commands listed there. Copy the results back and check
them:

```bash
rsync -av ubuntu@<board-ip>:gos/results/ v2/results/
.venv/bin/python v2/scripts/check_results.py
```

Dry run on the laptop, which is never paper data: `v2/board/run_all.sh --backend model
--allow-dirty` writes to `v2/results/dryrun/`.

## 12. Model analyses and paper figures

- Model analyses (label: model): `v2/analysis/` (see `v2/analysis/README.md`). They run inside
  `regen_results.sh`.
- Paper figures and tables: `.venv/bin/python v2/paper/scripts/make_all.py`. It reads only
  `v2/results/*.csv`. See `v2/paper/README.md`.

## 13. Number → producing script → CSV

| category (EXPERIMENTS.md) | number | producing script | CSV (`v2/results/`) | label |
|---|---|---|---|---|
| Accuracies of record (D3) | FP32 / INT8 test accuracy, LeNet-5, CIFAR-10 r1 and r2 | `v2/model/reference_accuracy.py` (regen) | `reference_accuracy.csv` | model |
| Accuracies of record | r1 vs r2 val/test and acceptance rule | `v2/model/retrain/check_r2.py` (regen) | `cifar10_r2_accuracy.csv`, `cifar10_r2_summary.csv` | model |
| Accuracies of record | r2 training curve | `v2/model/retrain/train_cifar10_r2.py` (not regenerated, D9) | `cifar10_retrain_log.csv` | model |
| A1 correctness | images, logit/prediction mismatches vs golden, golden and accelerator accuracy | `v2/board/exp_a1_accuracy.py` (Session 2) | `hw_a1_accuracy.csv` (+ `hw_logits_<net>.npz`) | hw |
| A1 (pre-board evidence) | RTL network images: LOGIT and prediction match | `run_core.sh` (net suite) | `rtl_network.csv` | RTL sim |
| A1 (pre-board evidence) | golden = legacy, final layer bit-exact, requant equivalence | `golden_crosscheck.py`, `final_layer.py`, `requant_check.py` (regen) | `golden_crosscheck.csv`, `final_layer_check.csv`, `requant_equivalence.csv` | model |
| A2 latency | per-layer and total PL cycles, µs at the read-back clock, wall-clock per image | `v2/board/exp_a2_a3_cycles.py` | `hw_a2_a3_cycles.csv` (+ `hw_cycles_<net>.npz`) | hw |
| A3 three-way cycles | model_cycles | `v2/model/gos_cycle_model.py` (regen) | `cycle_model.csv` | model |
| A3 | rtl_cycles | `run_core.sh` | `rtl_cycles.csv`, `rtl_network.csv` | RTL sim |
| A3 | hw_cycles, rtl_err_pct, hw_err_pct | `v2/board/exp_a2_a3_cycles.py` | `hw_a2_a3_cycles.csv` | hw |
| A4 utilization | MAC_ACTIVE / TOTAL vs theoretical (per-layer MAC_ACTIVE is model T·K; only the total is a HW counter) | `v2/board/exp_a4_util.py` | `hw_a4_util.csv` | hw (per-layer: model) |
| A4 (model side) | spatial/temporal utilization decomposition | `v2/analysis/utilization.py` (regen) | `utilization_model.csv` | model |
| A5 CPU baseline | INT8 ref, FP32 numpy, ORT FP32/INT8, 1 and 4 threads, compute and e2e | `v2/board/cpu/run_cpu_baselines.py --tag board` (run by `run_all.sh`) | `hw_cpu_baseline.csv` (+ `hw_cpu_baseline_env.json`) | hw (`cpu_board`) |
| A5 FPGA side | compute-only (PL counter) vs end-to-end | `v2/board/exp_b3_breakdown.py` | `hw_b3_breakdown.csv` | hw |
| A6 implementation | LUT/FF/DSP/BRAM, WNS/WHS, achieved clock, static checks | `vivado/build_gos.sh` + `impl_collect.py --csv …impl_gos.csv` | `impl_gos.csv` | post-impl |
| A6 (supporting) | per-module OOC resources / WNS | `ooc_all.sh` | `ooc_synth.csv` | post-synth |
| A6 (supporting) | empty shell | `vivado/build_shell.sh` | `impl_shell.csv` | post-impl |
| B1 power | P_idle / P_fpga / P_cpu windows, INA260 samples | `v2/board/exp_b1_power.py` (Session 3) | `hw_b1_power.csv`, `hw_b1_power_samples.csv` | hw (board-level input power, never accelerator power) |
| B1 energy/inference | ΔP × time from the inline meter log | **no script yet** (see Gaps) | — | hw |
| B2 clock sweep | cycles, latency, power per clock | `v2/board/exp_b2_clock.py` | `hw_b2_clock.csv`, `hw_b2_power.csv`, `hw_b2_power_samples.csv` | hw |
| B3 breakdown | input write, compute, logit read, host overhead | `v2/board/exp_b3_breakdown.py` | `hw_b3_breakdown.csv` (+ `hw_b3_times_<net>.npz`) | hw |
| C1 (optional) | 16x16 post-implementation | **no producer** (`projection_16x16.csv` is a model projection, explicitly not C1) | — | post-impl |
| C2 (optional) | normalized prior-work comparison | **no producer** | — | — |
| Model motivation | schedule ablation (D7) | `v2/analysis/ablation.py` (regen) | `schedule_ablation.csv` | model |
| Model motivation | 16x16 projection | `v2/analysis/projection_16x16.py` (regen) | `projection_16x16.csv` | model |
| Model motivation | OS vs WS, coarse sparsity | legacy V1 models (read-only; cited from `data/benchmark/*.json`) | not a `v2/results` CSV | model |
| Verification summary | all verification counts | `v2/scripts/verification_stats.py` | `verification_stats.csv` | per row |

## Gaps and inconsistencies (as of writing)

1. **No V2 requirements file.** `python/requirements.txt` pins numpy, torch, torchvision, onnx
   and matplotlib, but not onnxruntime or pytest, which the V2 tests and the CPU export use.
   The pins in section 1 come from `pip freeze` of the working `.venv`.
2. **Data-set download** is not scripted for V2 (all V2 loaders use `download=False`); section 1
   gives the command.
3. **`regen_results.sh` producer registry.** The inline check asserts that every
   `v2/results/*.csv` is known. `verification_stats.csv` and the board CSVs (`hw_*.csv`,
   `hw_cpu_baseline.csv`) are not in `REGEN` or `OTHER_PRODUCERS`. Once they are copied into
   `v2/results/`, the assertion fails, unless the script under edit adds them.
4. **Runtimes not recorded** (`duration_s` empty): `impl_gos.csv`, `impl_shell.csv`,
   `ooc_synth.csv`, `rtl_netlist.csv`, `cycle_model.csv`, `cifar10_r2_summary.csv`.
5. **Back-to-back RTL job count** is not in any CSV. `unit_tb.csv` records only checks and pass
   status for `tb_gos_top_backtoback`. The job count stated in DECISIONS comes from the TB
   default and its log.
6. **`rtl_netlist.csv`, `tb_gos_top` row:** `logits_checked` / `logits_ok` are 0, although the
   TB compares LOGIT[0..15] with `chk` (its checks count covers them). `netlist_collect.py` counts `logits_ok` only
   from `RESULT` lines, and the `RESULT kind=top` line of `tb_gos_top` has no `logits_ok` key.
7. **`impl_gos.csv` provenance.** `git_commit` / `git_dirty` are taken when `impl_collect.py`
   runs, not when the build runs. BUILD_ID (`build_id`) holds the build commit. Collect at the
   build commit, or check `build_id` against `git_commit`.
8. **`ooc_all.sh` header comment** says "up to OOC_JOBS in parallel, default 4". The code forces
   `JOBS=1` and ignores `OOC_JOBS`, which is the correct behaviour under the resource rule.
9. **B1 energy/inference:** there is no script that joins the meter log with the window
   timestamps (open item in `v2/board/README.md`).
10. **C1 and C2** have no producers.
11. **`run_netlist_sim.sh`** hard-codes `BID=1EB08730`, the BUILD_ID the netlist TBs expect.
    It is not the current commit.
