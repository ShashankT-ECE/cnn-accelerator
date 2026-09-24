# V2 Experiment Plan

## Thesis
A small INT8 output-stationary accelerator with bit-exact end-to-end inference and cycle-exact agreement between the analytical model, RTL simulation, and measured KV260 execution.

## A — Core results
- **A1 Correctness:** full 10k test sets, both nets. Record images tested, output mismatches vs golden (target 0), golden accuracy, hardware accuracy.
- **A2 Latency:** per-layer and total PL cycles; µs at the implemented clock; wall-clock per image.
- **A3 Three-way cycle agreement per layer:** model_cycles, rtl_cycles, hw_cycles, rtl_err_pct, hw_err_pct.
- **A4 Utilization per layer:** MAC_ACTIVE/TOTAL, compared with the theoretical value.
- **A5 Same-board CPU baseline (Cortex-A53):** INT8 numpy and FP32. Protocol: warm-up discarded, median of ≥100 runs, 1 thread and 4 threads reported separately, numpy/BLAS version recorded. Split compute-only vs end-to-end on both CPU and FPGA (FPGA compute-only = PL counter; FPGA end-to-end = input write + start + poll + logit read).
- **A6 Implementation:** LUT/FF/DSP/BRAM, WNS, achieved clock.

## B — Power and system
- **B1 Board-level input power** (inline 12 V meter; platformstats if available): P_idle, P_fpga, P_cpu, ΔP_fpga, ΔP_cpu, duration ≥30 s, number of inferences, meter resolution. Energy/inference = ΔP x time. Label as board-level input power, never accelerator power.
- **B2 Clock sweep 100/150/200 MHz:** latency, power, energy; cycle counts recorded as a consistency check. Runtime clock switching to be verified at bring-up; fallback one bitstream per clock.
- **B3 End-to-end breakdown:** input write, compute, logit read, host overhead.

## C — Optional
- **C1** (optional) 16x16 post-implementation only.
- **C2** (optional) normalized comparison with prior work.

## Model-based motivation (labeled model)
OS vs WS (V1 model), coarse sparsity 1.0001x (CIFAR-10, `data/benchmark/cifar10_sparsity_results.json:99`; LeNet-5 is 1.099x, `data/benchmark/sparse_execution_results.json:60`), schedule ablation (D7).

## CSV rule
Every results CSV row carries timestamp, git_commit, git_dirty, vivado_version, bitstream_sha256 (board runs), board_id, net, layer, clock_mhz, source (model/rtl_sim/post_impl/hw), duration_s, num_inferences, plus the measured fields.
