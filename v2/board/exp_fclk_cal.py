#!/usr/bin/env python3
"""PL clock cross-check: measured pl_clk0 (f_meas) = hardware cycle counter vs a monotonic clock.

    sudo ./session.sh py exp_fclk_cal.py [--seconds 8] [--rounds 2] [--tag _s2]
    python3 exp_fclk_cal.py --backend model --seconds 1    # dry run: f_meas := nominal (see below)

ROLE (DECISIONS D20): the clock of record for every µs value is the pl_clk0 PLL READ-BACK
(f_readback, exact divider arithmetic on the PS reference). f_meas is a CROSS-CHECK that the PL
really runs at that frequency; it is recorded beside the read-back and never used for a
conversion. |f_meas - f_readback| / f_readback > AGREE_TOL (0.1 %) is FLAGGED (fcal_flag column,
FLAG line in the log and in the session summary).

METHOD (differential lower-envelope estimator with a dithered poll phase):
  per job j (GosDevice.timed_job): dt_j = t(done seen) - t(before CTRL.start), t =
  CLOCK_MONOTONIC_RAW (never slewed by NTP), and C_j = TOTAL_CYC of that job (hardware counter).
  Model: dt_j = C_j / f + d_j, d_j >= d_min = the fastest start write + the fastest done read;
  d_j - d_min = how long after the true end the next STATUS poll happened (0 .. one poll period,
  several µs) plus interrupts / cache misses. Two job lengths with deterministic cycle counts
  (LeNet-5 ~16k, CIFAR-10 ~104k cycles; identical on every image) give
        f_meas = (C_B - C_A) / (min dt_B - min dt_A)
  The minimum removes every additive delay; d_min is the same for both nets and cancels. For the
  minimum to reach the true end, the poll grid must not be phase-locked to the job: timed_job
  waits a seeded random time in [0, dither) (dither = 2 x the measured STATUS poll period) between
  CTRL.start and the first poll, so the detection latency is uniform over a poll period and the
  minimum over n jobs is within ~period / n of the true end.
  Jobs run back to back in blocks (each net reloaded, untimed, at the start of its block; --warmup
  jobs per block discarded); block order = randomized ABAB (stats.block_order, seed recorded),
  --rounds rounds, the whole run >= --seconds of wall time.
WHY NOT THE MEDIAN (the method until 2026-09-30, kept as f_median_est_mhz for the record): on
  the KV260 the first run gave median-based 201.992 MHz at a 199.998 MHz read-back (+1.0 %).
  Cause: the cycle counts are deterministic and one STATUS poll takes ~3.5 µs, so each net's
  done-detection latency is a fixed phase of the poll grid, not a random variable: LeNet-5 jobs
  were mostly seen ~7 µs late, CIFAR-10 jobs ~3 µs late (raw dt histograms in the .npz), a 4.3 µs
  error on the 439 µs difference = 1 %. The medians do not cancel it; the minima of the same raw
  data give 199.959 MHz (-0.02 %). f_simple = C_B / median dt_B (-4 % on that run) is the naive
  single-job estimate: it contains the whole start-write + detection latency d (~19-27 µs on a
  521 µs job) and is reported only to show that bias.
UNCERTAINTY: the kept jobs of each net are split into N_FOLDS interleaved folds (job index mod
  N_FOLDS); the estimator is computed per fold; f_meas_ci_lo/hi_mhz = the 95 % t-interval of the
  fold estimates (N_FOLDS - 1 dof); fold min / max are reported too. Systematic (documented, not
  in the interval): (1) CLOCK_MONOTONIC_RAW (ARM generic timer) and pl_clk0 (PS PLL) both derive
  from the PS reference oscillator, so its ppm error is common-mode: f_meas verifies the PLL
  configuration / the pynq read-back, not the crystal's absolute accuracy; (2) timer granularity
  <= 10 ns per time stamp; (3) d_min assumed independent of the job length: the per-net
  intercepts min dt - C / f_meas are reported (intercept_*_ns).

DRY RUN (--backend model): the ModelBackend has no timing model (a job is computed inside the
CTRL.start write), so the estimator output is meaningless; it is recorded as estimator_mhz and
f_meas_mhz is set to the nominal clock with f_meas_source "dryrun_nominal" (never data).

Writes hw_fclk_cal<tag>.csv (one row) and hw_fclk_cal<tag>.npz (raw dt, C, net, block, kept).
"""
from __future__ import annotations

import argparse
import inspect
import sys
import time

import numpy as np

import board_common as bc
import stats

FIELDS = ["f_req_mhz", "f_readback_mhz", "f_meas_mhz", "f_meas_ci_lo_mhz", "f_meas_ci_hi_mhz",
          "f_meas_source", "estimator_mhz", "f_fold_min_mhz", "f_fold_max_mhz", "n_folds",
          "f_median_est_mhz", "f_simple_mhz", "ppm_vs_readback", "rel_diff_pct", "agree_tol_pct",
          "fcal_flag", "readback_in_ci", "method", "clock_ts", "net_a", "net_b", "cycles_a",
          "cycles_b", "jobs_a", "jobs_b", "min_dt_a_ns", "min_dt_b_ns", "median_dt_a_ns",
          "median_dt_b_ns", "intercept_a_ns", "intercept_b_ns", "poll_period_ns", "dither_ns",
          "seconds", "rounds", "warmup_per_block", "block_order", "seed", "repeats_ok",
          "cycles_constant", "host_path", "clock_of_record"]
METHOD = ("differential lower envelope: f = (C_B - C_A) / (min dt_B - min dt_A); dt = "
          "CLOCK_MONOTONIC_RAW start->done, poll phase dithered over 2 poll periods; 95% "
          "t-interval over interleaved folds; cross-check only (clock of record = PLL read-back)")
AGREE_TOL = 1e-3             # |f_meas - f_readback| / f_readback above this is flagged (0.1 %)
N_FOLDS = 8
T975 = {7: 2.364624}         # Student t, 0.975 quantile, N_FOLDS - 1 dof
MAX_DITHER_NS = 50_000


def raw_clock_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def estimate(c_a: float, dt_a, c_b: float, dt_b, stat=np.min) -> float:
    """f in MHz from the two nets' cycle counts and start->done times (ns); stat = np.min (the
    lower-envelope estimator) or np.median (the former method, biased by the poll grid)."""
    d = float(stat(dt_b)) - float(stat(dt_a))
    return (c_b - c_a) / d * 1e3 if d > 0 else float("nan")


def agreement(f_meas: float, f_readback: float, tol: float = AGREE_TOL) -> tuple[float, bool]:
    """(relative difference, flag): flag = |f_meas - f_readback| / f_readback > tol (or NaN)."""
    rel = (f_meas - f_readback) / f_readback
    return rel, not (abs(rel) <= tol)


def calibrate(dev, pkgs: dict, seconds: float = 8.0, rounds: int = 2, warmup: int = 5,
              seed: int = 1, n_folds: int = N_FOLDS, clock_ns=raw_clock_ns, say=print,
              dither_ns: int | None = None) -> dict:
    """Run the cross-check on the loaded bitstream at its current clock. pkgs: {net: Package}
    (exactly two nets with different cycle counts). Returns the estimate + raw arrays."""
    if len(pkgs) != 2:
        raise ValueError("calibration needs exactly two nets (two job lengths)")
    nets = list(pkgs)
    order = stats.block_order(nets, rounds, seed, "random")
    block_ns = int(seconds * 1e9 / len(order))
    rng = np.random.default_rng(seed)
    period = float(dev.poll_period_ns(clock_ns)) if hasattr(dev, "poll_period_ns") else float("nan")
    if dither_ns is None:
        dither_ns = int(min(MAX_DITHER_NS, max(1000.0, 2.0 * period))) if period == period else 0
    kw = ({"dither_ns": dither_ns, "rng": rng}
          if "dither_ns" in inspect.signature(dev.timed_job).parameters else {})
    rec = {"dt": [], "cyc": [], "net": [], "block": [], "kept": []}
    for b, net in enumerate(order):
        pkg = pkgs[net]
        dev.load_net(pkg)
        x = pkg.x_act
        n_img = x.shape[0]
        k = 0
        t_end = clock_ns() + block_ns
        while True:
            dt, cyc, _ = dev.timed_job(x[k % n_img], clock_ns=clock_ns, **kw)
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
    folds = np.array([estimate(c_med[a], dt_a[i::n_folds], c_med[b], dt_b[i::n_folds])
                      for i in range(n_folds)
                      if dt_a[i::n_folds].size and dt_b[i::n_folds].size])
    folds = folds[np.isfinite(folds)]
    if folds.size == n_folds and (n_folds - 1) in T975:
        half = T975[n_folds - 1] * float(np.std(folds, ddof=1)) / np.sqrt(n_folds)
        lo, hi = float(np.mean(folds)) - half, float(np.mean(folds)) + half
    else:
        lo = hi = float("nan")
    return {"f_mhz": f, "ci_lo_mhz": lo, "ci_hi_mhz": hi,
            "fold_min_mhz": float(folds.min()) if folds.size else float("nan"),
            "fold_max_mhz": float(folds.max()) if folds.size else float("nan"),
            "n_folds": int(folds.size), "f_median_est_mhz": estimate(c_med[a], dt_a, c_med[b], dt_b, np.median),
            "net_a": a, "net_b": b,
            "cycles_a": c_med[a], "cycles_b": c_med[b], "jobs_a": int(dt_a.size),
            "jobs_b": int(dt_b.size), "median_dt_a_ns": float(np.median(dt_a)),
            "median_dt_b_ns": float(np.median(dt_b)),
            "min_dt_a_ns": float(dt_a.min()), "min_dt_b_ns": float(dt_b.min()),
            "intercept_a_ns": float(dt_a.min() - c_med[a] / f * 1e3) if f == f else float("nan"),
            "intercept_b_ns": float(dt_b.min() - c_med[b] / f * 1e3) if f == f else float("nan"),
            "f_simple_mhz": c_med[b] / float(np.median(dt_b)) * 1e3,
            "cycles_constant": bool(all(np.unique(cyc[n]).size == 1 for n in nets)),
            "poll_period_ns": period, "dither_ns": dither_ns,
            "order": order, "seed": seed, "raw": arr}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap)
    ap.add_argument("--seconds", type=float, default=8.0, help="total timed wall (>= 4 for the paper)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=5, help="jobs discarded at the start of each block")
    ap.add_argument("--seed", type=int, default=None, help="block order + bootstrap seed (recorded)")
    ap.add_argument("--dither-ns", type=int, default=None,
                    help="random wait before the first poll, ns (default 2 x the measured poll period)")
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
    r = calibrate(dev, pkgs, a.seconds, a.rounds, a.warmup, seed, dither_ns=a.dither_ns)
    dur = time.perf_counter() - t0
    est = r["f_mhz"]
    if dry:
        f_meas, lo, hi = f_rb, f_rb, f_rb
        src = "dryrun_nominal (model backend: no timing model; estimator meaningless)"
    else:
        f_meas, lo, hi = est, r["ci_lo_mhz"], r["ci_hi_mhz"]
        src = "cross-check (differential lower envelope, CLOCK_MONOTONIC_RAW)"
    ok_rep = min(r["jobs_a"], r["jobs_b"]) >= stats.MIN_REPEATS
    row = ctx.meta("+".join(a.nets), "all", dur, r["jobs_a"] + r["jobs_b"], clock_mhz=f_rb)
    f = stats.fmt
    rel, flag = agreement(f_meas, f_rb)
    row.update(f_req_mhz=f_req, f_readback_mhz=f"{f_rb:.6f}", f_meas_mhz=f(f_meas, 6),
               f_meas_ci_lo_mhz=f(lo, 6), f_meas_ci_hi_mhz=f(hi, 6), f_meas_source=src,
               estimator_mhz=f(est, 6), f_fold_min_mhz=f(r["fold_min_mhz"], 6),
               f_fold_max_mhz=f(r["fold_max_mhz"], 6), n_folds=r["n_folds"],
               f_median_est_mhz=f(r["f_median_est_mhz"], 6), f_simple_mhz=f(r["f_simple_mhz"], 6),
               ppm_vs_readback=f(rel * 1e6, 1), rel_diff_pct=f(rel * 100, 4),
               agree_tol_pct=f(AGREE_TOL * 100, 3), fcal_flag=flag,
               min_dt_a_ns=f(r["min_dt_a_ns"], 1), min_dt_b_ns=f(r["min_dt_b_ns"], 1),
               poll_period_ns=f(r["poll_period_ns"], 1), dither_ns=r["dither_ns"],
               clock_of_record="f_readback (PLL read-back); f_meas is a cross-check only",
               readback_in_ci=(lo <= f_rb <= hi) if lo == lo else "",
               method=METHOD, clock_ts="CLOCK_MONOTONIC_RAW", net_a=r["net_a"], net_b=r["net_b"],
               cycles_a=f(r["cycles_a"], 1), cycles_b=f(r["cycles_b"], 1), jobs_a=r["jobs_a"],
               jobs_b=r["jobs_b"], median_dt_a_ns=f(r["median_dt_a_ns"], 1),
               median_dt_b_ns=f(r["median_dt_b_ns"], 1), intercept_a_ns=f(r["intercept_a_ns"], 1),
               intercept_b_ns=f(r["intercept_b_ns"], 1), seconds=a.seconds, rounds=a.rounds,
               warmup_per_block=a.warmup, block_order=" ".join(r["order"]), seed=seed,
               repeats_ok=ok_rep, cycles_constant=r["cycles_constant"],
               host_path=a.host_path)
    ctx.csv(f"hw_fclk_cal{a.tag}.csv", [row], FIELDS)
    raw = r["raw"]
    np.savez_compressed(ctx.path(f"hw_fclk_cal{a.tag}.npz"), dt_ns=raw["dt"], cycles=raw["cyc"],
                        net=raw["net"].astype(str), block=raw["block"], kept=raw["kept"],
                        f_readback_mhz=f_rb, seed=seed, source=np.array(ctx.source))
    print(f"[fclk cal] ({ctx.source}) f_readback {f_rb:.6f} MHz; estimator {est:.6f} MHz "
          f"[{r['ci_lo_mhz']:.6f}, {r['ci_hi_mhz']:.6f}] (95% t over {r['n_folds']} folds; fold range "
          f"{r['fold_min_mhz']:.6f}..{r['fold_max_mhz']:.6f}); median-based (former method) "
          f"{r['f_median_est_mhz']:.6f} MHz; f_simple {r['f_simple_mhz']:.6f} MHz; poll period "
          f"{r['poll_period_ns']:.0f} ns, dither {r['dither_ns']} ns; jobs {r['net_a']} {r['jobs_a']} / {r['net_b']} {r['jobs_b']}; "
          f"intercepts {r['intercept_a_ns']:.0f} / {r['intercept_b_ns']:.0f} ns")
    print(f"[fclk cal] f_meas = {f_meas:.6f} MHz ({src})")
    print(f"[fclk cal] clock of record = f_readback {f_rb:.6f} MHz; f_meas - f_readback = "
          f"{rel * 100:+.4f} % (tolerance {AGREE_TOL * 100:g} %)")
    if flag and not dry:
        print(f"[fclk cal] FLAG: |f_meas - f_readback| > {AGREE_TOL * 100:g} % — the PL clock does "
              "not match the PLL read-back; do not trust µs values until this is explained")
    if not r["cycles_constant"]:
        print("[fclk cal] WARNING: TOTAL_CYC not identical over the jobs of a net (median used)")
    if not ok_rep:
        print(f"[fclk cal] NOTE: fewer than {stats.MIN_REPEATS} kept jobs for a net")
    return 0 if (est == est or dry) else 1


if __name__ == "__main__":
    sys.exit(main())
