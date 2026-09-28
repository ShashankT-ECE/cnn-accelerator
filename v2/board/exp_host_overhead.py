#!/usr/bin/env python3
"""Host overhead of the driver's safe vs fast host path on the LAPTOP (MockMMIO): per-image phases.

    python3 exp_host_overhead.py [--n 1000] [--warmup 100] [--busy-polls 1] [--fast-store block]
                                 [--backend mock|model]      -> results/dryrun/host_overhead_mock.csv

LABEL: "host overhead, laptop mock (MockMMIO), not KV260". source=dryrun_model; output only under
results/dryrun/ (board_common guards). This measures the Python/numpy cost of the two host code
paths on the laptop CPU with zero bus latency (mock_mmio.py). It is NOT a KV260 measurement: on
the board every MMIO access adds AXI/interconnect latency and the Cortex-A53 is slower; the board
numbers come from exp_b3_breakdown.py (hw_b3_breakdown.csv / hw_b3_breakdown_fast.csv).

Both paths run on the SAME backend instance, interleaved per image (safe image k, fast image k,
...) so that laptop frequency drift affects both alike; the first --warmup inferences of each
path are discarded. Every logit vector is compared with golden (the mock returns the golden
logits of the image it finds byte-exact in ACT0, so a wrong input write shows up as an error).
Phases per image (time.perf_counter_ns, µs): input_write, status_clear, start_write, poll,
logit_read, ps_dequant, end_to_end (= input_write + status_clear + start_write + poll +
logit_read + ps_dequant). Per net x path x phase: median, p5, p95, p99, mean; for the fast rows
speedup_median = safe median / fast median and speedup_p95 likewise (computed here).
start_write, poll and status_clear also contain the mock's own register emulation (job lookup by
the ACT0 bytes, LOGIT/counter update, clear) - identical on both paths, so those phases'
speedups understate the host-code difference; its cost per job, timed without the driver, is
reported in mock_emulation_us_median / _p95.
--backend model runs the same comparison on the ModelBackend (golden compute inside the poll
phase; CSR through the gos_sim adapter) - a correctness run, its times mean nothing.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np

import board_common as bc
import gos_driver as D
from mock_mmio import LABEL as MOCK_LABEL

PHASES = ("input_write", "status_clear", "start_write", "poll", "logit_read", "ps_dequant",
          "end_to_end")
FIELDS = ["measurement", "host_path", "fast_store", "phase", "images", "warmup_discarded",
          "median_us", "p5_us", "p95_us", "p99_us", "mean_us", "speedup_median", "speedup_p95",
          "logit_mismatches", "job_errors", "busy_polls", "polls_median", "interleaved",
          "mock_emulation_us_median", "mock_emulation_us_p95",
          "cpu_model", "python_version", "numpy_version"]


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def one(dev, pkg, i: int) -> tuple[dict, np.ndarray, int]:
    r = dev.infer(pkg.x_act[i], read_counters=False)
    t0 = time.perf_counter_ns()
    bc.predict(r.logits, pkg.dequant)
    t_ps = time.perf_counter_ns() - t0
    ph = {"input_write": r.t_write_ns, "status_clear": r.t_clear_ns, "start_write": r.t_start_ns,
          "poll": r.t_run_ns - r.t_start_ns, "logit_read": r.t_logit_ns, "ps_dequant": t_ps}
    ph["end_to_end"] = sum(ph.values())
    return ph, r.logits, r.polls


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap)
    ap.set_defaults(backend="mock")
    ap.add_argument("--n", type=int, default=1000, help="measured inferences per path and net")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--busy-polls", type=int, default=1, help="mock: STATUS reads that see busy")
    a = ap.parse_args(argv)
    if a.backend == "pynq":
        raise SystemExit("exp_host_overhead.py is the laptop mock comparison; on the KV260 run "
                         "exp_b3_breakdown.py (--host-path safe / fast)")
    clk = a.clock_mhz or 200.0
    if a.backend == "mock":
        be = D.MockMmioBackend(a.data_dir, clock_mhz=clk, busy_polls=a.busy_polls)
        label = MOCK_LABEL
    else:
        be = D.ModelBackend(clock_mhz=clk)
        label = "dry run (model backend: golden compute inside poll), not KV260; times meaningless"
    devs = {"safe": D.GosDevice(be, timeout_s=a.timeout_s, write_mode=a.write_mode),
            "fast": D.GosDevice(be, timeout_s=a.timeout_s, host_path="fast",
                                fast_store=a.fast_store)}
    ctx = bc.RunContext(a, devs["fast"], be, "HOST-OVERHEAD")
    ctx.check_clean()
    ctx.banner()
    print(f"[HOST-OVERHEAD] {label}; safe = {devs['safe'].host_path_desc}; fast = "
          f"{devs['fast'].host_path_desc}")
    chk = devs["fast"].check_fast_path()
    print(f"[HOST-OVERHEAD] fast path check (mock windows): {chk}")
    rows, ok = [], True
    for net in a.nets:
        pkg = ctx.package(net)
        for d in devs.values():
            d.load_net(pkg)
        n_img = pkg.n if a.limit is None else min(a.limit, pkg.n)
        tot = a.warmup + a.n
        t = {hp: {p: np.zeros(tot, np.int64) for p in PHASES} for hp in devs}
        polls = {hp: np.zeros(tot, np.int64) for hp in devs}
        mism = {hp: 0 for hp in devs}
        errs = {hp: 0 for hp in devs}
        t0 = time.perf_counter()
        for k in range(tot):
            i = k % n_img
            for hp, dev in devs.items():
                try:
                    ph, logits, pl = one(dev, pkg, i)
                except D.GosJobError:
                    errs[hp] += 1
                    dev.recover()
                    continue
                for p in PHASES:
                    t[hp][p][k] = ph[p]
                polls[hp][k] = pl
                if k >= a.warmup and not np.array_equal(logits, pkg.golden_logits[i]):
                    mism[hp] += 1
        dur = time.perf_counter() - t0
        emu = be.emulation_cost_ns() if a.backend == "mock" else None
        if emu:
            print(f"[HOST-OVERHEAD {net}] mock register emulation per job (in start_write/poll/"
                  f"status_clear on both paths): median {emu['median_ns'] / 1e3:.2f} µs")
        keep = slice(a.warmup, tot)
        stats = {hp: {p: bc.percentiles(t[hp][p][keep] / 1e3) for p in PHASES} for hp in devs}
        print(f"[HOST-OVERHEAD {net}] {label}: {a.n} images per path after {a.warmup} warm-up "
              f"(interleaved); mismatches safe {mism['safe']} fast {mism['fast']}; job errors "
              f"safe {errs['safe']} fast {errs['fast']}")
        print(f"  {'phase':13} {'safe med':>10} {'safe p95':>10} {'fast med':>10} {'fast p95':>10} "
              f"{'speedup':>8}  (µs)")
        for p in PHASES:
            s, f = stats["safe"][p], stats["fast"][p]
            sp = s["p50"] / f["p50"] if f["p50"] > 0 else float("nan")
            print(f"  {p:13} {s['p50']:>10.2f} {s['p95']:>10.2f} {f['p50']:>10.2f} {f['p95']:>10.2f} "
                  f"{sp:>7.2f}x")
        for hp in devs:
            ok &= mism[hp] == 0 and errs[hp] == 0
            for p in PHASES:
                s = stats[hp][p]
                row = ctx.meta(net, "all", dur, 2 * tot, clock_mhz=clk)
                sp = sp95 = ""
                if hp == "fast":
                    ref = stats["safe"][p]
                    sp = f"{ref['p50'] / s['p50']:.3f}" if s["p50"] > 0 else ""
                    sp95 = f"{ref['p95'] / s['p95']:.3f}" if s["p95"] > 0 else ""
                row.update(measurement=label, host_path=hp,
                           fast_store=a.fast_store if hp == "fast" else "", phase=p,
                           images=a.n, warmup_discarded=a.warmup, median_us=f"{s['p50']:.3f}",
                           p5_us=f"{s['p5']:.3f}", p95_us=f"{s['p95']:.3f}",
                           p99_us=f"{s['p99']:.3f}", mean_us=f"{s['mean']:.3f}",
                           speedup_median=sp, speedup_p95=sp95, logit_mismatches=mism[hp],
                           job_errors=errs[hp],
                           busy_polls=a.busy_polls if a.backend == "mock" else "",
                           polls_median=float(np.median(polls[hp][keep])), interleaved=True,
                           mock_emulation_us_median=f"{emu['median_ns'] / 1e3:.3f}" if emu else "",
                           mock_emulation_us_p95=f"{emu['p95_ns'] / 1e3:.3f}" if emu else "",
                           cpu_model=cpu_model(), python_version=platform.python_version(),
                           numpy_version=np.__version__)
                rows.append(row)
    name = "host_overhead_mock.csv" if a.backend == "mock" else "host_overhead_model.csv"
    ctx.csv(name, rows, FIELDS)
    print("HOST-OVERHEAD:", "PASS (logits == golden on both paths)" if ok else "MISMATCH / ERRORS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
