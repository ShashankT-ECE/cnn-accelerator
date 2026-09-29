#!/usr/bin/env python3
"""PL clock calibration: measured pl_clk0 (f_meas) = hardware cycle counter vs a monotonic clock.

    sudo -E python3 exp_fclk_cal.py [--seconds 8] [--rounds 2] [--tag _s2]
    python3 exp_fclk_cal.py --backend model --seconds 1    # dry run: f_meas := nominal (see below)

Why: f = TOTAL_CYC / (wall time of one job) per step is noisy and biased (the wall time includes
the CTRL.start write and the STATUS-poll detection latency). Every µs value of the board runs
(A2 PL latency, B3 pl_compute, B1 E_comp / t_PL) is instead converted with ONE calibrated f_meas
per bitstream + clock, measured here; the pynq read-back f_readback is kept alongside.

METHOD (differential two-length estimator, EXPERIMENTS.md "Measured PL clock"):
  per job j (GosDevice.timed_job): dt_j = t(done seen) - t(before CTRL.start), t =
  CLOCK_MONOTONIC_RAW (never slewed by NTP, unlike CLOCK_MONOTONIC), and C_j = TOTAL_CYC of that
  job (hardware counter). Model: dt_j = C_j / f + d_j, d_j >= 0 = start-write latency + done-
  detection latency (one STATUS poll period), independent of the job length. Two job lengths
  with deterministic cycle counts (LeNet-5 ~16k, CIFAR-10 ~104k cycles; identical on every image)
  give
        f_meas = (C_B - C_A) / (median dt_B - median dt_A)
  which cancels the constant part of d. Jobs run back to back in blocks (each net reloaded,
  untimed, at the start of its block; --warmup jobs per block discarded); the block order is a
  randomized ABAB design (stats.block_order, seed recorded), --rounds rounds, the whole run
  >= --seconds of wall time (default 8 s: >= 2 s per net per round at 2 rounds -> CIFAR-10 back
  to back for >= 4 s).
UNCERTAINTY: 95 % percentile bootstrap of the estimator (the kept jobs of each net resampled
  independently, --n-boot 2000, seed recorded) -> f_meas_ci_lo/hi_mhz. Systematic (documented,
  not in the CI): (1) CLOCK_MONOTONIC_RAW (ARM generic timer) and pl_clk0 (PS PLL) both derive
  from the PS reference oscillator, so its ppm error is common-mode: f_meas verifies the PLL
  configuration / the pynq read-back, not the crystal's absolute accuracy; (2) timer granularity
  <= 10 ns per time stamp, averaged out by the medians; (3) d assumed independent of the job
  length: the per-net intercepts d_A, d_B = median dt - C / f_meas are reported
  (intercept_*_ns) and should agree within a poll period.
Also reported: f_simple = C_B / median dt_B (the naive per-job estimate, biased low by d), the
ppm difference f_meas vs f_readback and whether f_readback lies inside the CI.

DRY RUN (--backend model): the ModelBackend has no timing model (a job is computed inside the
CTRL.start write), so the estimator output is meaningless; it is recorded as estimator_mhz and
f_meas_mhz is set to the nominal clock with f_meas_source "dryrun_nominal" (never data).

Writes hw_fclk_cal<tag>.csv (one row) and hw_fclk_cal<tag>.npz (raw dt, C, net, block, kept).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import board_common as bc
import stats

FIELDS = ["f_req_mhz", "f_readback_mhz", "f_meas_mhz", "f_meas_ci_lo_mhz", "f_meas_ci_hi_mhz",
          "f_meas_source", "estimator_mhz", "f_simple_mhz", "ppm_vs_readback",
          "readback_in_ci", "method", "clock_ts", "net_a", "net_b", "cycles_a", "cycles_b",
          "jobs_a", "jobs_b", "median_dt_a_ns", "median_dt_b_ns", "intercept_a_ns",
          "intercept_b_ns", "seconds", "rounds", "warmup_per_block", "block_order", "seed",
          "n_boot", "repeats_ok", "cycles_constant", "host_path"]
METHOD = ("differential: f = (C_B - C_A) / (median dt_B - median dt_A); dt = CLOCK_MONOTONIC_RAW "
          "start->done; 95% CI percentile bootstrap")


def raw_clock_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def estimate(c_a: float, dt_a, c_b: float, dt_b) -> float:
    """f in MHz from the two nets' cycle counts and start->done times (ns)."""
    d = float(np.median(dt_b)) - float(np.median(dt_a))
    return (c_b - c_a) / d * 1e3 if d > 0 else float("nan")


def calibrate(dev, pkgs: dict, seconds: float = 8.0, rounds: int = 2, warmup: int = 5,
              seed: int = 1, n_boot: int = 2000, clock_ns=raw_clock_ns, say=print) -> dict:
    """Run the calibration on the loaded bitstream at its current clock. pkgs: {net: Package}
    (exactly two nets with different cycle counts). Returns the estimate + raw arrays."""
    if len(pkgs) != 2:
        raise ValueError("calibration needs exactly two nets (two job lengths)")
    nets = list(pkgs)
    order = stats.block_order(nets, rounds, seed, "random")
    block_ns = int(seconds * 1e9 / len(order))
    rec = {"dt": [], "cyc": [], "net": [], "block": [], "kept": []}
    for b, net in enumerate(order):
        pkg = pkgs[net]
        dev.load_net(pkg)
        x = pkg.x_act
        n_img = x.shape[0]
        k = 0
        t_end = clock_ns() + block_ns
        while True:
            dt, cyc, _ = dev.timed_job(x[k % n_img], clock_ns=clock_ns)
            rec["dt"].append(dt)
            rec["cyc"].append(cyc)
            rec["net"].append(net)
            rec["block"].append(b)
            rec["kept"].append(k >= warmup)
            k += 1
            if k > warmup and clock_ns() >= t_end:
                break
        say(f"  [fclk cal] block {b + 1}/{len(order)} {net}: {k} jobs ({warmup} warm-up discarded)")
    arr = {k: np.asarray(v) for k, v in rec.items()}
    sel = {n: (arr["net"] == n) & arr["kept"] for n in nets}
    cyc = {n: arr["cyc"][sel[n]] for n in nets}
    c_med = {n: float(np.median(cyc[n])) for n in nets}
    a, b = sorted(nets, key=lambda n: c_med[n])            # a = shorter job
    dt_a, dt_b = arr["dt"][sel[a]].astype(np.float64), arr["dt"][sel[b]].astype(np.float64)
    f = estimate(c_med[a], dt_a, c_med[b], dt_b)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = estimate(c_med[a], dt_a[rng.integers(0, dt_a.size, dt_a.size)],
                            c_med[b], dt_b[rng.integers(0, dt_b.size, dt_b.size)])
    boots = boots[np.isfinite(boots)]
    lo, hi = ((float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
              if boots.size else (float("nan"), float("nan")))
    return {"f_mhz": f, "ci_lo_mhz": lo, "ci_hi_mhz": hi, "net_a": a, "net_b": b,
            "cycles_a": c_med[a], "cycles_b": c_med[b], "jobs_a": int(dt_a.size),
            "jobs_b": int(dt_b.size), "median_dt_a_ns": float(np.median(dt_a)),
            "median_dt_b_ns": float(np.median(dt_b)),
            "intercept_a_ns": float(np.median(dt_a) - c_med[a] / f * 1e3) if f == f else float("nan"),
            "intercept_b_ns": float(np.median(dt_b) - c_med[b] / f * 1e3) if f == f else float("nan"),
            "f_simple_mhz": c_med[b] / float(np.median(dt_b)) * 1e3,
            "cycles_constant": bool(all(np.unique(cyc[n]).size == 1 for n in nets)),
            "order": order, "seed": seed, "n_boot": n_boot, "raw": arr}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap)
    ap.add_argument("--seconds", type=float, default=8.0, help="total timed wall (>= 4 for the paper)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5, help="jobs discarded at the start of each block")
    ap.add_argument("--seed", type=int, default=None, help="block order + bootstrap seed (recorded)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tag", default="", help="output hw_fclk_cal<tag>.csv (e.g. _s2)")
    a = ap.parse_args(argv)
    if len(a.nets) != 2:
        raise SystemExit("--nets: two nets (two job lengths) are needed")
    seed = a.seed if a.seed is not None else stats.new_seed()
    dev, be = bc.open_device(a)
    ctx = bc.RunContext(a, dev, be, "FCLK-CAL")
    ctx.check_clean()
    ctx.banner()
    dry = ctx.source != bc.SOURCE_HW
    f_rb = dev.fclk0_mhz()
    f_req = ctx.run_env.get("f_req_mhz") or ctx.info.get("bit_clock_requested_mhz") or ""
    pkgs = {n: ctx.package(n) for n in a.nets}
    print(f"[fclk cal] {METHOD}; {a.seconds:g} s, {a.rounds} rounds, seed {seed}"
          + ("  DRY RUN: model backend has no timing -> f_meas := nominal" if dry else ""))
    t0 = time.perf_counter()
    r = calibrate(dev, pkgs, a.seconds, a.rounds, a.warmup, seed, a.n_boot)
    dur = time.perf_counter() - t0
    est = r["f_mhz"]
    if dry:
        f_meas, lo, hi = f_rb, f_rb, f_rb
        src = "dryrun_nominal (model backend: no timing model; estimator meaningless)"
    else:
        f_meas, lo, hi = est, r["ci_lo_mhz"], r["ci_hi_mhz"]
        src = "calibration (differential, CLOCK_MONOTONIC_RAW)"
    ok_rep = min(r["jobs_a"], r["jobs_b"]) >= stats.MIN_REPEATS
    row = ctx.meta("+".join(a.nets), "all", dur, r["jobs_a"] + r["jobs_b"], clock_mhz=f_rb)
    f = stats.fmt
    row.update(f_req_mhz=f_req, f_readback_mhz=f"{f_rb:.6f}", f_meas_mhz=f(f_meas, 6),
               f_meas_ci_lo_mhz=f(lo, 6), f_meas_ci_hi_mhz=f(hi, 6), f_meas_source=src,
               estimator_mhz=f(est, 6), f_simple_mhz=f(r["f_simple_mhz"], 6),
               ppm_vs_readback=f((f_meas - f_rb) / f_rb * 1e6, 1),
               readback_in_ci=(lo <= f_rb <= hi) if lo == lo else "",
               method=METHOD, clock_ts="CLOCK_MONOTONIC_RAW", net_a=r["net_a"], net_b=r["net_b"],
               cycles_a=f(r["cycles_a"], 1), cycles_b=f(r["cycles_b"], 1), jobs_a=r["jobs_a"],
               jobs_b=r["jobs_b"], median_dt_a_ns=f(r["median_dt_a_ns"], 1),
               median_dt_b_ns=f(r["median_dt_b_ns"], 1), intercept_a_ns=f(r["intercept_a_ns"], 1),
               intercept_b_ns=f(r["intercept_b_ns"], 1), seconds=a.seconds, rounds=a.rounds,
               warmup_per_block=a.warmup, block_order=" ".join(r["order"]), seed=seed,
               n_boot=a.n_boot, repeats_ok=ok_rep, cycles_constant=r["cycles_constant"],
               host_path=a.host_path)
    ctx.csv(f"hw_fclk_cal{a.tag}.csv", [row], FIELDS)
    raw = r["raw"]
    np.savez_compressed(ctx.path(f"hw_fclk_cal{a.tag}.npz"), dt_ns=raw["dt"], cycles=raw["cyc"],
                        net=raw["net"].astype(str), block=raw["block"], kept=raw["kept"],
                        f_readback_mhz=f_rb, seed=seed, source=np.array(ctx.source))
    print(f"[fclk cal] ({ctx.source}) f_readback {f_rb:.6f} MHz; estimator {est:.6f} MHz "
          f"[{r['ci_lo_mhz']:.6f}, {r['ci_hi_mhz']:.6f}] (95% bootstrap); f_simple "
          f"{r['f_simple_mhz']:.6f} MHz; jobs {r['net_a']} {r['jobs_a']} / {r['net_b']} {r['jobs_b']}; "
          f"intercepts {r['intercept_a_ns']:.0f} / {r['intercept_b_ns']:.0f} ns")
    print(f"[fclk cal] f_meas = {f_meas:.6f} MHz ({src})")
    if not r["cycles_constant"]:
        print("[fclk cal] WARNING: TOTAL_CYC not identical over the jobs of a net (median used)")
    if not ok_rep:
        print(f"[fclk cal] NOTE: fewer than {stats.MIN_REPEATS} kept jobs for a net")
    return 0 if (est == est or dry) else 1


if __name__ == "__main__":
    sys.exit(main())
