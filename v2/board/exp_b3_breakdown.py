#!/usr/bin/env python3
"""B3 end-to-end breakdown: host-side time per phase, interleaved conditions, median + 95 % CI.

    ./session.sh py exp_b3_breakdown.py [--nets ...] [--n 1000] [--conditions safe fast cpu]
                                        [--block-n 50] [--block-warmup 5] [--seed S]
    ./session.sh py exp_b3_breakdown.py --host-path fast        # one condition (old usage)
    python3 exp_b3_breakdown.py --backend model                 # dry run -> results/dryrun/
    python3 exp_b3_breakdown.py --backend mock --conditions safe fast   # laptop MockMMIO

CONDITIONS (interleaved in one process, EXPERIMENTS.md "Measurement rigor"):
  safe  accelerator, per-word MMIO host path            -> hw_b3_breakdown.csv
  fast  accelerator, fast host path (mapped numpy windows) -> hw_b3_breakdown_fast.csv; first
        GosDevice.check_fast_path() (whole ACT0 window fast-written + read back word by word, CSR
        block reads == per-word reads); a failed check drops the condition (safe still measured)
  cpu   CPU baseline cpu_int8_ref, 1 thread (cpu/cpu_infer.make_runner; same INT8 arithmetic),
        compute + argmax per image                        -> hw_b3_breakdown_cpu.csv
Design: per net, blocks of --block-warmup + --block-n inferences; each round runs every condition
once in a seeded random order (stats.block_order: randomized ABAB), rounds = ceil(n / block-n);
the first --warmup inferences of each condition (once) and the first --block-warmup of every
block are DISCARDED. --conditions-add appends conditions (run_sessions adds 'fast' only when the
s1.fast bring-up check passed).

Per accelerator image (images 0..N-1 cycled):
  input_write   input ACT image -> ACT0 (MMIO writes, --write-mode)
  status_clear  pre-start STATUS read + soft_reset of the previous job's sticky done + clear poll
  start_done    CTRL.start + STATUS polling until done (= start_write + poll)
  start_write   CTRL.start write alone (part of start_done)
  poll          STATUS polling until done (start_done - start_write)
  logit_read    LOGIT[0..OC-1] reads
  ps_dequant    PS float32 dequant + argmax (board_common.predict, D2)
  end_to_end    sum of the four phases above
  counter_read  TOTAL/MAC/STALL/LAYER_CYC reads (measurement overhead; NOT in end_to_end)
  pl_compute    TOTAL_CYC / f_used (f_used = the pl_clk0 PLL read-back, the clock of record;
                f_meas is a cross-check column only; columns f_readback_mhz / f_meas_mhz /
                f_used_mhz / f_used_source)
CPU condition phases: cpu_compute (runner) and end_to_end (runner + argmax).
Statistics (stats.summarize): median with the distribution-free order-statistic 95 % CI
(median_ci_lo_us / median_ci_hi_us, achieved coverage), p5, p95, p99, mean, min, max in µs;
repeats_ok = kept >= 100. Logits are checked against golden. Raw ns per phase + condition /
block / kept flags in hw_b3_times_<net><suffix>.npz.
"""
from __future__ import annotations

import os
import sys

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")          # CPU condition: 1 thread (before numpy import)

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import board_common as bc  # noqa: E402
import stats  # noqa: E402

PHASES = ("input_write", "status_clear", "start_done", "start_write", "poll", "logit_read",
          "ps_dequant", "end_to_end", "counter_read", "pl_compute")
CPU_PHASES = ("cpu_compute", "end_to_end")
IN_E2E = ("input_write", "status_clear", "start_done", "logit_read", "ps_dequant")
FIELDS = ["phase", "condition", "images", "warmup_discarded", "median_us", "median_ci_lo_us",
          "median_ci_hi_us", "ci_coverage", "ci_method", "repeats_ok", "p5_us", "p95_us", "p99_us",
          "mean_us", "min_us", "max_us", "in_end_to_end", "write_mode", "host_path", "fast_store",
          "measurement", "logit_mismatches", "polls_median", "soft_reset_clears", "interleave",
          "conditions", "order_seed", "rounds", "block_n", "block_warmup", "block_median_min_us",
          "block_median_max_us", *bc.CLOCK_COLS]
OUT_NAME = {"safe": "hw_b3_breakdown.csv", "fast": "hw_b3_breakdown_fast.csv",
            "cpu": "hw_b3_breakdown_cpu.csv"}
SUFFIX = {"safe": "", "fast": "_fast", "cpu": "_cpu"}
CPU_KIND = "cpu_int8_ref"


def measure(dev, pkg, n: int, warmup: int, n_img: int | None = None, read_counters: bool = True,
            f_mhz: float | None = None):
    """Single-condition B3 loop (kept for callers / tests): warmup + n accelerator inferences.
    Returns (t, polls, cleared, mismatches, duration_s); t[phase] = int64 ns per inference."""
    n_img = pkg.n if n_img is None else n_img
    clk = dev.fclk0_mhz() if f_mhz is None else f_mhz
    tot = warmup + n
    t = {p: np.zeros(tot, np.int64) for p in PHASES}
    polls = np.zeros(tot, np.int64)
    cleared = np.zeros(tot, bool)
    mism = 0
    t_start = time.perf_counter()
    for k in range(tot):
        i = k % n_img
        r = _accel_one(dev, pkg, i, read_counters, clk, t, polls, cleared, k)
        if k >= warmup and not np.array_equal(r.logits, pkg.golden_logits[i]):
            mism += 1
    return t, polls, cleared, mism, time.perf_counter() - t_start


def _accel_one(dev, pkg, i, read_counters, clk, t, polls, cleared, k):
    r = dev.infer(pkg.x_act[i], read_counters=read_counters)
    t0 = time.perf_counter_ns()
    bc.predict(r.logits, pkg.dequant)
    t_ps = time.perf_counter_ns() - t0
    t["input_write"][k], t["start_done"][k] = r.t_write_ns, r.t_run_ns
    t["start_write"][k], t["poll"][k] = r.t_start_ns, r.t_run_ns - r.t_start_ns
    t["status_clear"][k] = r.t_clear_ns
    t["logit_read"][k], t["ps_dequant"][k] = r.t_logit_ns, t_ps
    t["counter_read"][k] = r.t_counter_ns
    t["end_to_end"][k] = r.t_write_ns + r.t_clear_ns + r.t_run_ns + r.t_logit_ns + t_ps
    if r.total_cyc is not None:
        t["pl_compute"][k] = round(r.total_cyc / clk * 1e3)          # ns
    polls[k], cleared[k] = r.polls, r.cleared
    return r


def make_cpu_runner(net: str, data_dir: str):
    sys.path.insert(0, str(bc.BOARD_DIR))
    from cpu.cpu_infer import make_runner
    return make_runner(CPU_KIND, net, data_dir, 1)


def run_interleaved(dev, pkg, conds, n, warmup, block_n, block_warmup, seed, n_img, clk,
                    runner=None, fast_store="block"):
    """Interleaved blocks over conds. Returns per condition dict of arrays + bookkeeping."""
    rounds = max(1, math.ceil(n / block_n))
    order = stats.block_order(conds, rounds, seed, "random")
    per_block = block_warmup + block_n
    out = {}
    for c in conds:
        tot = warmup + rounds * per_block
        ph = CPU_PHASES if c == "cpu" else PHASES
        out[c] = {"t": {p: np.zeros(tot, np.int64) for p in ph}, "polls": np.zeros(tot, np.int64),
                  "cleared": np.zeros(tot, bool), "kept": np.zeros(tot, bool),
                  "block": np.full(tot, -1, np.int64), "k": 0, "mism": 0, "img": 0}
    xs = pkg.x_nchw if "cpu" in conds else None

    def run_n(c, count, keep, block):
        d = out[c]
        for _ in range(count):
            k, i = d["k"], d["img"] % n_img
            if c == "cpu":
                t0 = time.perf_counter_ns()
                lg = runner(xs[i])
                t1 = time.perf_counter_ns()
                np.argmax(lg)
                t2 = time.perf_counter_ns()
                d["t"]["cpu_compute"][k], d["t"]["end_to_end"][k] = t1 - t0, t2 - t0
            else:
                dev.set_host_path(c, fast_store)
                r = _accel_one(dev, pkg, i, True, clk, d["t"], d["polls"], d["cleared"], k)
                if keep and not np.array_equal(r.logits, pkg.golden_logits[i]):
                    d["mism"] += 1
            d["kept"][k], d["block"][k] = keep, block
            d["k"] += 1
            d["img"] += 1

    t_start = time.perf_counter()
    for c in conds:                                   # global warm-up, once per condition
        run_n(c, warmup, False, -1)
    for b, c in enumerate(order):
        run_n(c, block_warmup, False, b)
        run_n(c, block_n, True, b)
    dev.set_host_path("safe")
    return out, order, rounds, time.perf_counter() - t_start


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap)
    ap.add_argument("--n", type=int, default=1000, help="kept inferences per condition (>= 1000 paper)")
    ap.add_argument("--warmup", type=int, default=50, help="discarded once per condition")
    ap.add_argument("--conditions", nargs="+", default=None, choices=("safe", "fast", "cpu"),
                    help="default: --host-path alone")
    ap.add_argument("--conditions-add", nargs="+", default=[], choices=("safe", "fast", "cpu"))
    ap.add_argument("--block-n", type=int, default=50, help="kept inferences per block")
    ap.add_argument("--block-warmup", type=int, default=5, help="discarded at every block start")
    ap.add_argument("--seed", type=int, default=None, help="block order seed (recorded)")
    a = ap.parse_args(argv)
    conds = list(dict.fromkeys((a.conditions or [a.host_path]) + a.conditions_add))
    seed = a.seed if a.seed is not None else stats.new_seed()
    dev, be = bc.open_device(a)
    ctx = bc.RunContext(a, dev, be, "B3")
    ctx.check_clean()
    ctx.banner()
    if a.n < 1000:
        print(f"NOTE: --n {a.n} < 1000 (the EXPERIMENTS protocol asks for >= 1000)")
    label = "measured on KV260" if ctx.source == bc.SOURCE_HW else (
        "host overhead, laptop mock (MockMMIO), not KV260" if be.kind == "mock_mmio"
        else "dry run (model backend), not KV260")
    ok = True
    if "fast" in conds:
        try:
            dev.set_host_path("fast", a.fast_store)
            chk = dev.check_fast_path()
            print(f"[B3] fast path check PASSED: {chk}")
        except Exception as e:  # noqa: BLE001 - any failure (mismatch, MemoryError) = not validated
            print(f"[B3] FAST PATH CHECK FAILED: {type(e).__name__}: {e}\n"
                  "[B3] condition 'fast' dropped; the safe host path is still measured")
            conds.remove("fast")
            ok = False
        finally:
            dev.set_host_path("safe")
    if not conds:
        return 2
    print(f"[B3] conditions {conds}, interleaved blocks of {a.block_warmup}+{a.block_n}, seed {seed}")
    rows = {c: [] for c in conds}
    for net in a.nets:
        pkg = ctx.package(net)
        dev.load_net(pkg)
        rb = dev.fclk0_mhz()
        f_used, _ = ctx.f_used(rb)
        ccols = ctx.clock_cols(rb)
        n_img = pkg.n if a.limit is None else min(a.limit, pkg.n)
        runner = make_cpu_runner(net, a.data_dir) if "cpu" in conds else None
        res, order, rounds, dur = run_interleaved(dev, pkg, conds, a.n, a.warmup, a.block_n,
                                                  a.block_warmup, seed, n_img, f_used, runner,
                                                  a.fast_store)
        print(f"[B3 {net}] ({ctx.source}; {label}) {rounds} rounds, order {' '.join(order[:8])}"
              f"{' ...' if len(order) > 8 else ''}; f_used {f_used:.6f} MHz ({ccols['f_used_source']})")
        for c in conds:
            d = res[c]
            keep = d["kept"]
            ok &= d["mism"] == 0
            print(f"  [{c}] kept {int(keep.sum())}, logit mismatches {d['mism']}")
            print(f"  {'phase':13} {'median':>9} {'95% CI':>21} {'p95':>9}  (µs)")
            for p, arr in d["t"].items():
                v = arr[keep] / 1e3
                s = stats.summarize(v)
                bm = [float(np.median(arr[(d["block"] == b) & keep])) / 1e3
                      for b in sorted(set(d["block"][keep].tolist()))]
                row = ctx.meta(net, "all", dur, int(d["k"]), clock_mhz=rb)
                hp = {"safe": "safe", "fast": "fast", "cpu": ""}[c]
                row.update(phase=p, condition=c if c != "cpu" else f"cpu ({CPU_KIND} x1)",
                           images=int(keep.sum()), warmup_discarded=int((~keep).sum()),
                           median_us=stats.fmt(s["median"]), median_ci_lo_us=stats.fmt(s["ci_lo"]),
                           median_ci_hi_us=stats.fmt(s["ci_hi"]), ci_coverage=stats.fmt(s["ci_coverage"], 4),
                           ci_method=s["ci_method"], repeats_ok=s["repeats_ok"],
                           p5_us=stats.fmt(s["p5"]), p95_us=stats.fmt(s["p95"]), p99_us=stats.fmt(s["p99"]),
                           mean_us=stats.fmt(s["mean"]), min_us=stats.fmt(s["min"]),
                           max_us=stats.fmt(s["max"]),
                           in_end_to_end=(p in IN_E2E) if c != "cpu" else (p == "cpu_compute"),
                           write_mode=a.write_mode, host_path=hp,
                           fast_store=a.fast_store if c == "fast" else "",
                           measurement=label if c != "cpu" else label.replace("KV260", "KV260 A53 CPU"),
                           logit_mismatches=d["mism"] if c != "cpu" else "",
                           polls_median=float(np.median(d["polls"][keep])) if c != "cpu" else "",
                           soft_reset_clears=int(d["cleared"][keep].sum()) if c != "cpu" else "",
                           interleave="randomized blocks (stats.block_order)", conditions=" ".join(conds),
                           order_seed=seed, rounds=rounds, block_n=a.block_n, block_warmup=a.block_warmup,
                           block_median_min_us=stats.fmt(min(bm)) if bm else "",
                           block_median_max_us=stats.fmt(max(bm)) if bm else "", **ccols)
                rows[c].append(row)
                lo, hi = stats.fmt(s["ci_lo"], 2), stats.fmt(s["ci_hi"], 2)
                print(f"  {p:13} {s['median']:>9.2f} {('[' + lo + ', ' + hi + ']'):>21} {s['p95']:>9.2f}")
            np.savez_compressed(ctx.path(f"hw_b3_times_{net}{SUFFIX[c]}.npz"), warmup=a.warmup,
                                f_used_mhz=f_used, f_readback_mhz=rb, polls=d["polls"],
                                cleared=d["cleared"], kept=keep, block=d["block"], seed=seed,
                                source=np.array(ctx.source), **{f"{p}_ns": v for p, v in d["t"].items()})
    for c in conds:
        ctx.csv(OUT_NAME[c], rows[c], FIELDS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
