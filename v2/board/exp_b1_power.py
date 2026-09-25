#!/usr/bin/env python3
"""B1 power windows: idle / fpga / cpu, for the inline 12 V meter (board-level input power).

    sudo -E python3 exp_b1_power.py --modes idle fpga cpu [--net lenet5] [--window-s 60]
                                    [--gap-s 10] [--cpu-kind cpu_int8_ref] [--cpu-threads 1]
    python3 exp_b1_power.py --backend model --window-s 2   # dry run -> results/dryrun/

For each window the script prints START / STOP banners with wall-clock timestamps (ISO UTC,
ISO local, epoch seconds) so the operator can align the manual meter log, runs the workload for
--window-s seconds and counts inferences:
  idle  nothing runs (the overlay stays loaded, PL clock running)
  fpga  back-to-back accelerator inferences (GosDevice.infer, counters not read)
  cpu   back-to-back CPU inferences via cpu/cpu_infer.make_runner(kind, net, data_dir, threads)
        (CPU-baseline scripts; runner(x) = runner.compute: x_nchw[i] int8 [C,H,W] for
        cpu_int8_ref, x_f32[i] float32 [C,H,W] for the other kinds; BLAS/OpenMP threads are
        fixed from --cpu-threads before numpy is imported)
On-board sensor: sampled every --sample-s (0.5 s) from the INA260 hwmon sysfs node (what
platformstats reads) or, if absent, the `platformstats` CLI; else recorded as "unavailable".
LABELS: the meter measures BOARD-LEVEL INPUT POWER; the INA260 value is SOM power as reported by
the sensor. Neither is accelerator power. Energy/inference = delta-P x time is computed later
from the meter log joined on the window timestamps — never typed by hand.
Writes hw_b1_power.csv (one row per window) and hw_b1_power_samples.csv (every sensor sample).
"""
from __future__ import annotations

import os
import sys


def _preset_blas_threads(argv):
    """cpu/cpu_infer requires OPENBLAS/OMP thread counts fixed BEFORE numpy is imported."""
    n = "1"
    for i, t in enumerate(argv):
        if t == "--cpu-threads" and i + 1 < len(argv):
            n = argv[i + 1]
        elif t.startswith("--cpu-threads="):
            n = t.split("=", 1)[1]
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = n


_preset_blas_threads(sys.argv[1:])

import argparse  # noqa: E402
import datetime as _dt
import glob
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

import board_common as bc

FIELDS = ["mode", "window_id", "cpu_kind", "cpu_threads", "start_utc", "stop_utc", "start_local",
          "stop_local", "start_epoch", "stop_epoch", "window_s", "inf_per_s", "sensor_source",
          "sensor_label", "sensor_power_mean_w", "sensor_power_median_w", "sensor_power_min_w",
          "sensor_power_max_w", "sensor_samples", "meter_label"]
SAMPLE_FIELDS = ["mode", "window_id", "t_epoch", "t_rel_s", "sensor_power_w", "sensor_source"]
METER_LABEL = "board-level input power (inline 12 V meter; manual log, joined by timestamps)"
SENSOR_LABEL = "INA260 SOM power as reported on-board (not accelerator power, not board input)"


# ---- on-board power sensor -------------------------------------------------------------------
class PowerSensor:
    """INA260 via hwmon sysfs (power1_input in µW), else `platformstats -p` output parsing."""

    def __init__(self):
        self.kind, self.path, self.name = "unavailable", None, ""
        for d in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
            try:
                name = Path(d, "name").read_text().strip()
            except OSError:
                continue
            p = Path(d, "power1_input")
            if name.startswith("ina260") and p.is_file():
                self.kind, self.path, self.name = "hwmon", p, f"hwmon {name} ({d})"
                return
        exe = shutil.which("platformstats")
        if exe:
            self.kind, self.path, self.name = "platformstats_cli", exe, f"{exe} -p"

    def read_w(self):
        try:
            if self.kind == "hwmon":
                return int(self.path.read_text().strip()) / 1e6
            if self.kind == "platformstats_cli":
                out = subprocess.run([self.path, "-p"], capture_output=True, text=True,
                                     timeout=2).stdout
                # ASSUMPTION: a line mentioning total/SOM power followed by a number + unit
                for line in out.splitlines():
                    m = re.search(r"(total|som).*?power.*?([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\b", line,
                                  re.IGNORECASE)
                    if m:
                        v = float(m.group(2))
                        return v / 1000 if m.group(3) == "mW" else v
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        return None

    @property
    def source(self) -> str:
        return self.name or "unavailable"


class Sampler(threading.Thread):
    def __init__(self, sensor: PowerSensor, period_s: float):
        super().__init__(daemon=True)
        self.sensor, self.period = sensor, period_s
        self.samples: list[tuple[float, float]] = []
        self._stop_ev = threading.Event()

    def run(self):
        if self.sensor.kind == "unavailable":
            return
        nxt = time.time()
        while not self._stop_ev.is_set():
            v = self.sensor.read_w()
            if v is not None:
                self.samples.append((time.time(), v))
            nxt += self.period
            self._stop_ev.wait(max(0.0, nxt - time.time()))

    def stop(self):
        self._stop_ev.set()
        self.join(timeout=5)


# ---- windows ---------------------------------------------------------------------------------
def stamp() -> dict:
    now = time.time()
    return {"epoch": f"{now:.3f}",
            "utc": _dt.datetime.fromtimestamp(now, _dt.timezone.utc).isoformat(timespec="milliseconds"),
            "local": _dt.datetime.fromtimestamp(now).astimezone().isoformat(timespec="milliseconds"),
            "t": now}


def banner(what: str, s: dict, extra: str = ""):
    print(f"\n===== {what} {extra} | UTC {s['utc']} | local {s['local']} | epoch {s['epoch']} =====",
          flush=True)


def make_cpu_runner(kind: str, net: str, data_dir: str, threads: int):
    sys.path.insert(0, str(bc.BOARD_DIR))
    try:
        from cpu.cpu_infer import make_runner
    except ImportError as e:
        raise SystemExit(f"cpu mode needs cpu/cpu_infer.py (CPU-baseline scripts): {e}")
    return make_runner(kind, net, data_dir, threads)


def power_window(ctx, dev, pkg, mode: str, window_s: float, sensor: PowerSensor,
                 sample_s: float = 0.5, window_id: int = 0, cpu_kind: str = "", cpu_threads: int = 1,
                 clock_mhz=None, label: str = "B1") -> tuple[dict, list[dict]]:
    """One measurement window. Returns (row, sample rows)."""
    net = pkg.name if pkg is not None else ""
    work = None
    if mode == "fpga":
        dev.load_net(pkg)
        x = pkg.x_act
        n_img = x.shape[0]
        work = lambda k: dev.infer(x[k % n_img], read_counters=False)  # noqa: E731
    elif mode == "cpu":
        runner = make_cpu_runner(cpu_kind, net, ctx.args.data_dir, cpu_threads)
        xs = pkg.x_nchw if cpu_kind == "cpu_int8_ref" else pkg.x_f32
        n_img = xs.shape[0]
        work = lambda k: runner(xs[k % n_img])  # noqa: E731
    elif mode != "idle":
        raise ValueError(mode)
    if work is not None:        # one untimed call (lazy init, caches) before the window
        work(0)
    samp = Sampler(sensor, sample_s)
    extra = f"{label} mode={mode} net={net if mode != 'idle' else '-'} window={window_s:g}s" + \
            (f" cpu={cpu_kind}x{cpu_threads}" if mode == "cpu" else "")
    s0 = stamp()
    banner("START", s0, extra)
    samp.start()
    count = 0
    t_end = s0["t"] + window_s
    if work is None:
        while time.time() < t_end:
            time.sleep(min(0.2, max(0.0, t_end - time.time())))
    else:
        while time.time() < t_end:
            work(count)
            count += 1
    s1 = stamp()
    samp.stop()
    banner("STOP ", s1, f"{extra} inferences={count}")
    dur = s1["t"] - s0["t"]
    pw = np.array([v for _, v in samp.samples], dtype=np.float64)
    row = ctx.meta(net, "all", dur, count if mode != "idle" else 0, clock_mhz=clock_mhz)
    row.update(mode=mode, window_id=window_id, cpu_kind=cpu_kind if mode == "cpu" else "",
               cpu_threads=cpu_threads if mode == "cpu" else "",
               start_utc=s0["utc"], stop_utc=s1["utc"], start_local=s0["local"],
               stop_local=s1["local"], start_epoch=s0["epoch"], stop_epoch=s1["epoch"],
               window_s=f"{dur:.3f}", inf_per_s=f"{count / dur:.3f}" if mode != "idle" else "",
               sensor_source=sensor.source, sensor_label=SENSOR_LABEL,
               sensor_power_mean_w=f"{pw.mean():.4f}" if pw.size else "",
               sensor_power_median_w=f"{np.median(pw):.4f}" if pw.size else "",
               sensor_power_min_w=f"{pw.min():.4f}" if pw.size else "",
               sensor_power_max_w=f"{pw.max():.4f}" if pw.size else "",
               sensor_samples=int(pw.size), meter_label=METER_LABEL)
    srows = []
    for t, v in samp.samples:
        r = ctx.meta(net, "all", "", "", clock_mhz=clock_mhz)
        r.update(mode=mode, window_id=window_id, t_epoch=f"{t:.3f}", t_rel_s=f"{t - s0['t']:.3f}",
                 sensor_power_w=f"{v:.4f}", sensor_source=sensor.source)
        srows.append(r)
    print(f"  [{label}] {mode}: {count} inferences in {dur:.2f} s; sensor ({sensor.source}): "
          f"{row['sensor_power_mean_w'] or 'unavailable'} W mean over {pw.size} samples", flush=True)
    return row, srows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap, nets=False)
    ap.add_argument("--modes", nargs="+", default=["idle", "fpga", "cpu"],
                    choices=("idle", "fpga", "cpu"))
    ap.add_argument("--net", default="lenet5", choices=bc.NETS)
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--gap-s", type=float, default=10.0, help="pause between windows (settling)")
    ap.add_argument("--sample-s", type=float, default=0.5)
    ap.add_argument("--cpu-kind", default="cpu_int8_ref",
                    help="cpu/cpu_infer kind: cpu_int8_ref, cpu_fp32_numpy, cpu_ort_fp32, cpu_ort_int8")
    ap.add_argument("--cpu-threads", type=int, default=1)
    a = ap.parse_args(argv)
    a.nets = [a.net]
    dev, be = bc.open_device(a)
    ctx = bc.RunContext(a, dev, be, "B1")
    ctx.check_clean()
    ctx.banner()
    if a.window_s < 30:
        print(f"NOTE: window {a.window_s} s < 30 s (EXPERIMENTS B1 asks for >= 30 s)")
    sensor = PowerSensor()
    print(f"[B1] on-board sensor: {sensor.source}; meter: {METER_LABEL}")
    pkg = ctx.package(a.net)
    clk = dev.fclk0_mhz()
    rows, samples = [], []
    for w, mode in enumerate(a.modes):
        if w:
            time.sleep(a.gap_s)
        r, s = power_window(ctx, dev, pkg, mode, a.window_s, sensor, a.sample_s, w, a.cpu_kind,
                            a.cpu_threads, clock_mhz=clk)
        rows.append(r)
        samples += s
    ctx.csv("hw_b1_power.csv", rows, FIELDS)
    ctx.csv("hw_b1_power_samples.csv", samples, SAMPLE_FIELDS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
