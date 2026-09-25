#!/usr/bin/env python3
"""B3 end-to-end breakdown: host-side time per phase with time.perf_counter_ns.

    sudo -E python3 exp_b3_breakdown.py [--nets ...] [--n 1000] [--warmup 50]
    python3 exp_b3_breakdown.py --backend model        # dry run -> results/dryrun/ (times are
                                                         the laptop simulator's, meaningless)

Per image (images 0..n-1 cycled; the first --warmup inferences are discarded):
  input_write   input ACT image -> ACT0 (MMIO writes, --write-mode)
  start_done    pre-start STATUS check/clear + CTRL.start + STATUS polling until done
  logit_read    LOGIT[0..OC-1] reads
  ps_dequant    PS float32 dequant + argmax (board_common.predict, D2)
  end_to_end    sum of the four phases above
  counter_read  TOTAL/MAC/STALL/LAYER_CYC reads (measurement overhead; NOT in end_to_end)
  pl_compute    TOTAL_CYC / clock (PL compute only, from the counter; for comparison)
Statistics: median, p5, p95, p99, mean, min, max in µs. Logits are checked against golden.
Writes hw_b3_breakdown.csv and hw_b3_times_<net>.npz (raw ns per phase).
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import board_common as bc

PHASES = ("input_write", "status_clear", "start_done", "logit_read", "ps_dequant", "end_to_end",
          "counter_read", "pl_compute")
FIELDS = ["phase", "images", "warmup_discarded", "median_us", "p5_us", "p95_us", "p99_us",
          "mean_us", "min_us", "max_us", "in_end_to_end", "write_mode", "logit_mismatches",
          "polls_median", "soft_reset_clears"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap)
    ap.add_argument("--n", type=int, default=1000, help="measured inferences (>= 1000 for the paper)")
    ap.add_argument("--warmup", type=int, default=50)
    a = ap.parse_args(argv)
    dev, be = bc.open_device(a)
    ctx = bc.RunContext(a, dev, be, "B3")
    ctx.check_clean()
    ctx.banner()
    if a.n < 1000:
        print(f"NOTE: --n {a.n} < 1000 (the EXPERIMENTS protocol asks for >= 1000)")
    rows = []
    ok = True
    for net in a.nets:
        pkg = ctx.package(net)
        dev.load_net(pkg)
        clk = dev.fclk0_mhz()
        n_img = pkg.n if a.limit is None else min(a.limit, pkg.n)
        x_act = pkg.x_act
        tot = a.warmup + a.n
        t = {p: np.zeros(tot, np.int64) for p in PHASES}
        polls = np.zeros(tot, np.int64)
        cleared = np.zeros(tot, bool)
        mism = 0
        t_start = time.perf_counter()
        for k in range(tot):
            i = k % n_img
            r = dev.infer(x_act[i], read_counters=True)
            t0 = time.perf_counter_ns()
            bc.predict(r.logits, pkg.dequant)
            t_ps = time.perf_counter_ns() - t0
            t["input_write"][k], t["start_done"][k] = r.t_write_ns, r.t_run_ns
            t["status_clear"][k] = r.t_clear_ns
            t["logit_read"][k], t["ps_dequant"][k] = r.t_logit_ns, t_ps
            t["counter_read"][k] = r.t_counter_ns
            t["end_to_end"][k] = r.t_write_ns + r.t_clear_ns + r.t_run_ns + r.t_logit_ns + t_ps
            t["pl_compute"][k] = round(r.total_cyc / clk * 1e3)          # ns
            polls[k], cleared[k] = r.polls, r.cleared
            if k >= a.warmup and not np.array_equal(r.logits, pkg.golden_logits[i]):
                mism += 1
        dur = time.perf_counter() - t_start
        ok &= mism == 0
        keep = slice(a.warmup, tot)
        print(f"[B3 {net}] ({ctx.source}) {a.n} inferences after {a.warmup} warm-up; "
              f"clock {clk:.3f} MHz; logit mismatches {mism}")
        print(f"  {'phase':13} {'median':>9} {'p5':>9} {'p95':>9} {'p99':>9}  (µs)")
        for p in PHASES:
            s = bc.percentiles(t[p][keep] / 1e3)
            row = ctx.meta(net, "all", dur, tot, clock_mhz=clk)
            row.update(phase=p, images=a.n, warmup_discarded=a.warmup,
                       median_us=f"{s['p50']:.3f}", p5_us=f"{s['p5']:.3f}",
                       p95_us=f"{s['p95']:.3f}", p99_us=f"{s['p99']:.3f}",
                       mean_us=f"{s['mean']:.3f}", min_us=f"{s['min']:.3f}",
                       max_us=f"{s['max']:.3f}",
                       in_end_to_end=p in ("input_write", "status_clear", "start_done", "logit_read", "ps_dequant"),
                       write_mode=a.write_mode, logit_mismatches=mism,
                       polls_median=float(np.median(polls[keep])),
                       soft_reset_clears=int(cleared[keep].sum()))
            rows.append(row)
            print(f"  {p:13} {s['p50']:>9.2f} {s['p5']:>9.2f} {s['p95']:>9.2f} {s['p99']:>9.2f}")
        np.savez_compressed(ctx.path(f"hw_b3_times_{net}.npz"), warmup=a.warmup, clock_mhz=clk,
                            polls=polls, cleared=cleared, source=np.array(ctx.source),
                            **{f"{p}_ns": v for p, v in t.items()})
    ctx.csv("hw_b3_breakdown.csv", rows, FIELDS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
