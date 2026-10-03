#!/usr/bin/env python3
"""V3 baseline SOM-rail power logger: on-board INA260 (VCC_SOM) sampled on the KV260.
Adapted from v2/board/power_log.py @ 28dd2ad: sensors (hwmon / platformstats / read-only I2C / mock),
the sampler PROCESS, rate checks and the B1 P_idle / energy rule are copied unchanged; the protocol is
generalized from the fixed accel/control/cpu kinds to N named workloads (one per baseline system
configuration) and the summary is written in long format (one row per repeat x configuration).

    ./session.sh py power_log.py --list-sensors                    # probe only, writes nothing
    ./session.sh py power_log.py --sample-only --seconds 10        # sensor check (CSV + stats)
    (the protocol itself is run by baseline_session.py)

LABEL: every number is "SOM-rail power (INA260)" -- the SOM rail (VCC_SOM) as reported by the
on-board INA260 (V2 UG1089 reading: SOM +5 V input; per-rail coverage unconfirmed, V2 open item). It is
NOT accelerator-only power and NOT board input power.

Protocol (per repeat, default 3 repeats, 60 s phases): the run phases of the configurations come in
a seeded random permutation per repeat (--order random, the default of baseline_session.py; the order
and seed are recorded), each bracketed by idle phases:
    idle_1, <cfg a>, idle_2, <cfg b>, ..., idle_k, <cfg k>     ... then one idle_post
P_idle (V2 B1 rule, fixed): for each run phase the equal-weight mean of the phase means of the two idle
phases that bracket it in time (cancels a linear drift); one bracketing idle -> used alone.
    dP = P_run - P_idle;  time/image = phase duration / images completed;
    E_sys = energy/image = dP x time/image   (the whole e2e loop incl. host preprocessing)
mean / std rows = mean and sample std (ddof=1) of the per-repeat values.

Outputs (out_dir): <prefix>_samples<tag>.csv (every sample), <prefix>_phases<tag>.csv (one row per
phase), <prefix>_summary<tag>.csv (row_kind repeat/mean/std x config).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import baseline_common as bc

LABEL = "SOM-rail power (INA260)"
RAIL = "VCC_SOM"
PREFIX = "hw_baseline_power_ina260"
DEFAULT_RATE_HZ = 10.0
DEFAULT_PHASE_S = 60.0
DEFAULT_REPEATS = 3
SENSOR_ORDER = ("hwmon", "platformstats", "i2c")

# INA260 (TI datasheet SBOSA27): 16-bit big-endian registers
REG_CONFIG, REG_CURRENT, REG_VBUS, REG_POWER = 0x00, 0x01, 0x02, 0x03
REG_MFR_ID, REG_DIE_ID = 0xFE, 0xFF
MFR_ID, DIE_ID = 0x5449, 0x2270
I2C_ADDRS = tuple(range(0x40, 0x50))
POWER_LSB_W, CURRENT_LSB_A, VBUS_LSB_V = 0.010, 0.00125, 0.00125
INA260_AVG = (1, 4, 16, 64, 128, 256, 512, 1024)
INA260_CT_US = (140, 204, 332, 588, 1100, 2116, 4156, 8244)


class SensorUnavailable(RuntimeError):
    pass


@dataclass
class Reading:
    power_w: float
    current_a: float | None = None
    voltage_v: float | None = None
    raw: str = ""


# ---- sensors (copied unchanged) ----------------------------------------------------------------
class Sensor:
    backend = "abstract"
    is_hw = True

    def __init__(self):
        self.device = ""
        self.limits: dict = {}

    def read(self, phase: str = "") -> Reading | None:
        raise NotImplementedError

    def describe(self) -> str:
        lim = f" limits={json.dumps(self.limits, sort_keys=True)}" if self.limits else ""
        return f"{self.backend} {self.device}{lim}"


class HwmonSensor(Sensor):
    backend = "hwmon"

    def __init__(self, hwmon_dir):
        super().__init__()
        self.dir = Path(hwmon_dir)
        self.name = (self.dir / "name").read_text().strip()
        self.device = f"{self.dir} name={self.name}"
        self.p_power = self.dir / "power1_input"
        self.p_curr = self.dir / "curr1_input"
        self.p_volt = self.dir / "in1_input"
        ui = self.dir / "update_interval"
        if ui.is_file():
            try:
                self.limits["update_interval_ms"] = int(ui.read_text().strip())
            except (OSError, ValueError):
                self.limits["update_interval_ms"] = "unreadable"
        else:
            self.limits["update_interval_ms"] = "not exposed"

    @staticmethod
    def _int(p: Path):
        try:
            return int(p.read_text().strip())
        except (OSError, ValueError):
            return None

    def read(self, phase: str = "") -> Reading | None:
        uw = self._int(self.p_power)
        if uw is None:
            return None
        ma = self._int(self.p_curr) if self.p_curr.is_file() else None
        mv = self._int(self.p_volt) if self.p_volt.is_file() else None
        return Reading(uw / 1e6, None if ma is None else ma / 1e3, None if mv is None else mv / 1e3,
                       f"power1_input={uw}uW curr1_input={ma}mA in1_input={mv}mV")


_PS_NUM = re.compile(r"([-+]?[0-9]+(?:\.[0-9]+)?)\s*(uW|µW|mW|W)\b")


def parse_platformstats(text: str):
    """(power_w, raw_line) from `platformstats -p` output, or None (format ASSUMED, see V2)."""
    cands = []
    for line in text.splitlines():
        low = line.lower()
        if "power" not in low:
            continue
        m = _PS_NUM.search(line)
        if not m:
            continue
        v = float(m.group(1))
        unit = m.group(2)
        w = v / 1e6 if unit in ("uW", "µW") else v / 1e3 if unit == "mW" else v
        prio = 0 if "som" in low else 1 if "total" in low else 2
        cands.append((prio, w, line.strip()))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    return cands[0][1], cands[0][2]


class PlatformstatsSensor(Sensor):
    backend = "platformstats"

    def __init__(self, exe: str, run=subprocess.run):
        super().__init__()
        self.exe, self.run = exe, run
        self.device = f"{exe} -p"
        self.limits["note"] = "one subprocess per sample; update rate of the underlying sensor unknown"

    def read(self, phase: str = "") -> Reading | None:
        try:
            out = self.run([self.exe, "-p"], capture_output=True, text=True, timeout=2).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        p = parse_platformstats(out or "")
        if p is None:
            return None
        return Reading(p[0], None, None, p[1])


class ReadOnlyBus:
    """Only SMBus register READS; every write attribute raises (INA260 config is never written)."""
    _ALLOWED = ("read_word_data", "read_byte_data")

    def __init__(self, bus):
        object.__setattr__(self, "_bus", bus)

    def __getattr__(self, name):
        if name in self._ALLOWED or name == "close":
            return getattr(self._bus, name)
        raise PermissionError(f"power_log I2C is read-only: {name} is not allowed")

    def __setattr__(self, name, value):
        raise PermissionError("power_log I2C is read-only")


def _swap16(w: int) -> int:
    return ((w & 0xFF) << 8) | ((w >> 8) & 0xFF)


def _s16(u: int) -> int:
    return u - 0x10000 if u & 0x8000 else u


def ina260_read_reg(bus: ReadOnlyBus, addr: int, reg: int) -> int:
    return _swap16(bus.read_word_data(addr, reg))


def decode_ina260_config(cfg: int) -> dict:
    avg = INA260_AVG[(cfg >> 9) & 7]
    vct = INA260_CT_US[(cfg >> 6) & 7]
    ict = INA260_CT_US[(cfg >> 3) & 7]
    mode = cfg & 7
    d = {"config_reg": f"0x{cfg:04X}", "averages": avg, "vbus_ct_us": vct, "ishunt_ct_us": ict, "mode": mode}
    if mode == 7:
        d["update_period_ms_derived"] = round(avg * (vct + ict) / 1000.0, 3)
    return d


def _default_smbus(busnum: int):
    try:
        from smbus2 import SMBus
    except ImportError as e:
        raise SensorUnavailable(f"smbus2 not installed: {e}") from None
    return SMBus(busnum)


class I2cSensor(Sensor):
    backend = "i2c"

    def __init__(self, bus, busnum: int, addr: int):
        super().__init__()
        self.bus = bus if isinstance(bus, ReadOnlyBus) else ReadOnlyBus(bus)
        self.busnum, self.addr = busnum, addr
        self.device = f"/dev/i2c-{busnum} addr=0x{addr:02X}"
        try:
            self.limits.update(decode_ina260_config(ina260_read_reg(self.bus, addr, REG_CONFIG)))
        except OSError as e:
            self.limits["config_reg"] = f"unreadable: {e}"

    def read(self, phase: str = "") -> Reading | None:
        try:
            p = ina260_read_reg(self.bus, self.addr, REG_POWER)
            c = _s16(ina260_read_reg(self.bus, self.addr, REG_CURRENT))
            v = ina260_read_reg(self.bus, self.addr, REG_VBUS)
        except OSError:
            return None
        return Reading(p * POWER_LSB_W, c * CURRENT_LSB_A, v * VBUS_LSB_V,
                       f"reg03=0x{p:04X} reg01=0x{c & 0xFFFF:04X} reg02=0x{v:04X}")


def is_idle_phase(phase: str) -> bool:
    return phase == "" or phase.startswith("idle") or phase == "sample_only"


class MockSensor(Sensor):
    """Dry-run sensor. levels_w: {'idle': W, <config>: W, 'run': default W for other run phases}
    (synthetic, NOT measured); noise_w: std of seeded gaussian noise."""
    backend = "mock"
    is_hw = False

    def __init__(self, levels_w: dict | None = None, noise_w: float = 0.0, seed: int = 0):
        super().__init__()
        import random
        self.levels = dict(idle=1.0, run=1.4)
        self.levels.update(levels_w or {})
        self.noise_w = float(noise_w)
        self.rng = random.Random(seed)
        self.device = f"mock levels={self.levels} noise_w={self.noise_w} seed={seed}"
        self.limits["note"] = "synthetic dry-run sensor, not a measurement"

    def read(self, phase: str = "") -> Reading | None:
        w = self.levels["idle"] if is_idle_phase(phase) else self.levels.get(phase, self.levels["run"])
        if self.noise_w:
            w += self.rng.gauss(0.0, self.noise_w)
        return Reading(w, None, None, f"mock phase={phase}")


# ---- discovery (copied unchanged) --------------------------------------------------------------
def find_hwmon(root="/sys/class/hwmon") -> list[Path]:
    out = []
    for d in sorted(glob.glob(os.path.join(str(root), "*"))):
        try:
            name = Path(d, "name").read_text().strip()
        except OSError:
            continue
        if name.startswith("ina260") and Path(d, "power1_input").is_file():
            out.append(Path(d))
    return out


def probe_i2c(dev_glob="/dev/i2c-*", smbus_factory=None, addrs=I2C_ADDRS, log=None,
              stop_at_first=True) -> list[dict]:
    factory = smbus_factory or _default_smbus
    log = log if log is not None else []
    buses = []
    for p in sorted(glob.glob(dev_glob), key=lambda s: (len(s), s)):
        m = re.search(r"i2c-(\d+)$", p)
        if m:
            buses.append(int(m.group(1)))
    if not buses:
        log.append(f"i2c: no device matches {dev_glob}")
    opened, found = {}, []
    passes = ([[0x40], [a for a in addrs if a != 0x40]] if 0x40 in addrs else [list(addrs)])
    for k, pas in enumerate(passes):
        if found and k > 0:
            break
        for addr in pas:
            for n in buses:
                if n not in opened:
                    try:
                        opened[n] = ReadOnlyBus(factory(n))
                    except SensorUnavailable as e:
                        log.append(f"i2c: {e}")
                        return found
                    except OSError as e:
                        log.append(f"i2c-{n}: cannot open: {e}")
                        opened[n] = None
                bus = opened[n]
                if bus is None:
                    continue
                try:
                    mfr = ina260_read_reg(bus, addr, REG_MFR_ID)
                    die = ina260_read_reg(bus, addr, REG_DIE_ID)
                except OSError as e:
                    log.append(f"i2c-{n} 0x{addr:02X}: no read ({e.__class__.__name__})")
                    continue
                ok = mfr == MFR_ID and die == DIE_ID
                log.append(f"i2c-{n} 0x{addr:02X}: mfr=0x{mfr:04X} die=0x{die:04X} {'INA260' if ok else 'not INA260'}")
                if ok:
                    found.append({"busnum": n, "addr": addr, "bus": bus})
                    if stop_at_first:
                        return found
    return found


def discover(name: str = "auto", *, allow_hw: bool = True, allow_mock: bool = False,
             hwmon_root="/sys/class/hwmon", which=shutil.which, run=subprocess.run,
             i2c_glob="/dev/i2c-*", smbus_factory=None, mock_kw: dict | None = None):
    """(sensor, probe_log). auto: hwmon -> platformstats -> i2c on hardware; the mock only in a dry run."""
    log: list[str] = []
    if name == "mock":
        if not allow_mock:
            raise SensorUnavailable("mock sensor refused: hardware run")
        return MockSensor(**(mock_kw or {})), ["mock requested"]
    if name != "auto" and not allow_hw:
        raise SensorUnavailable(f"real sensor '{name}' refused in a dry run")
    tries = SENSOR_ORDER if name == "auto" else (name,)
    if name == "auto" and not allow_hw:
        tries = ()
    for t in tries:
        if t == "hwmon":
            d = find_hwmon(hwmon_root)
            log.append(f"hwmon: {[str(x) for x in d] or 'no ina260* node with power1_input'}")
            if d:
                return HwmonSensor(d[0]), log
        elif t == "platformstats":
            exe = which("platformstats")
            log.append(f"platformstats: {exe or 'not on PATH'}")
            if exe:
                s = PlatformstatsSensor(exe, run=run)
                r = s.read()
                log.append(f"platformstats: test read -> {r.raw if r else 'no power line parsed'}")
                if r is not None:
                    return s, log
        elif t == "i2c":
            c = probe_i2c(i2c_glob, smbus_factory, log=log)
            if c:
                return I2cSensor(c[0]["bus"], c[0]["busnum"], c[0]["addr"]), log
        else:
            raise ValueError(f"unknown sensor backend {t!r}")
    if name == "auto" and allow_mock:
        log.append("auto: dry run -> mock sensor")
        return MockSensor(**(mock_kw or {})), log
    raise SensorUnavailable("no INA260 sensor found: " + "; ".join(log))


def list_sensors() -> list[str]:
    log = []
    d = find_hwmon()
    log.append(f"hwmon: {[str(x) for x in d] or 'none'}")
    for x in d:
        s = HwmonSensor(x)
        r = s.read()
        log.append(f"  {s.describe()} -> {r.raw if r else 'read failed'}")
    exe = shutil.which("platformstats")
    log.append(f"platformstats: {exe or 'not on PATH'}")
    if exe:
        r = PlatformstatsSensor(exe).read()
        log.append(f"  -> {r.raw if r else 'no power line parsed'}")
    for x in probe_i2c(log=log, stop_at_first=False):
        s = I2cSensor(x["bus"], x["busnum"], x["addr"])
        r = s.read()
        log.append(f"  {s.describe()} -> {r.raw if r else 'read failed'}")
    return log


# ---- sampler (copied unchanged: a PROCESS, the V2 GIL-starvation fix) --------------------------
def stamp(mono=None, wall=None) -> dict:
    now = time.time() if wall is None else wall
    return {"mono": time.monotonic() if mono is None else mono, "epoch": now,
            "utc": _dt.datetime.fromtimestamp(now, _dt.timezone.utc).isoformat(timespec="milliseconds"),
            "local": _dt.datetime.fromtimestamp(now).astimezone().isoformat(timespec="milliseconds")}


def banner(what: str, s: dict, extra: str = ""):
    print(f"\n===== {what} {extra} | UTC {s['utc']} | local {s['local']} | epoch {s['epoch']:.3f} =====",
          flush=True)


MIN_RUN_RATE_FRACTION = 0.5


class UndersampledError(RuntimeError):
    """A run phase got far fewer INA260 samples than the idle phases: the sampler was starved."""


def _sampler_main(sensor, rate_hz: float, q, stop_ev, phase_buf, err_val, avoid_cores):
    try:
        if avoid_cores:
            os.sched_setaffinity(0, avoid_cores)
    except (AttributeError, OSError):
        pass
    period = 1.0 / rate_hz
    nxt = time.monotonic()
    while not stop_ev.is_set():
        t, te = time.monotonic(), time.time()
        ph = phase_buf.value.decode("ascii", "replace")
        try:
            r = sensor.read(ph)
        except Exception:  # noqa: BLE001
            r = None
        if r is None:
            with err_val.get_lock():
                err_val.value += 1
        else:
            q.put({"t_mono": t, "t_epoch": te, "live_phase": ph, "r": r})
        nxt += period
        now = time.monotonic()
        if nxt < now:
            nxt = now
        stop_ev.wait(nxt - now)
    q.put(None)


class PowerLogger:
    """Sampling PROCESS at a fixed rate; samples carry monotonic + UTC epoch time; a sample with
    start <= t < stop belongs to that phase window."""

    def __init__(self, sensor: Sensor, rate_hz: float = DEFAULT_RATE_HZ):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        self.sensor, self.rate_hz = sensor, float(rate_hz)
        self.samples: list[dict] = []
        self.windows: list[dict] = []
        self._read_errors = 0
        self._proc = None
        self._err = None
        self.t_start = self.t_stop = None

    @property
    def read_errors(self) -> int:
        return self._err.value if self._err is not None and self._proc is not None else self._read_errors

    def _set_phase(self, name: str):
        self._phase_buf.value = name.encode("ascii", "replace")[:63]

    def start(self):
        import multiprocessing as mp
        if self._proc is not None:
            raise RuntimeError("PowerLogger already started")
        mpx = mp.get_context("fork")
        self._q, self._stop_ev = mpx.Queue(), mpx.Event()
        self._phase_buf, self._err = mpx.Array("c", 64), mpx.Value("i", 0)
        try:
            parent = set(os.sched_getaffinity(0))
            allc = set(range(os.cpu_count() or 1))
            others = allc - parent if len(parent) < len(allc) else set()
        except (AttributeError, OSError):
            others = set()
        self.t_start = time.monotonic()
        self._proc = mpx.Process(target=_sampler_main, name="power_log_sampler", daemon=True,
                                 args=(self.sensor, self.rate_hz, self._q, self._stop_ev, self._phase_buf,
                                       self._err, others))
        self._proc.start()

    def _drain(self) -> bool:
        import queue
        if self._proc is None:
            return True
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return False
            if item is None:
                return True
            self.samples.append(item)

    def stop(self):
        if self._proc is None:
            return
        self._stop_ev.set()
        deadline = time.monotonic() + 15
        while not self._drain() and time.monotonic() < deadline:
            time.sleep(0.01)
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
        self.t_stop = time.monotonic()
        self._read_errors = self._err.value
        self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def begin(self, phase: str, extra: str = "") -> dict:
        self._set_phase(phase)
        s = stamp()
        banner("START", s, extra)
        return s

    def end(self, phase: str, s0: dict, extra: str = "", **info) -> dict:
        s1 = stamp()
        self._set_phase("")
        banner("STOP ", s1, extra)
        w = {"phase": phase, "start": s0, "stop": s1, **info}
        self.windows.append(w)
        return w

    def window_samples(self, w: dict) -> list[dict]:
        self._drain()
        a, b = w["start"]["mono"], w["stop"]["mono"]
        return [s for s in self.samples if a <= s["t_mono"] < b]

    def overall(self) -> dict:
        self._drain()
        t1 = self.t_stop if self.t_stop is not None else time.monotonic()
        return rate_stats([s["t_mono"] for s in self.samples], self.t_start, t1)


def undersampled_phases(logger: PowerLogger, frac: float = MIN_RUN_RATE_FRACTION) -> list[str]:
    def rate(w):
        n = len(logger.window_samples(w))
        d = w["stop"]["mono"] - w["start"]["mono"]
        return n / d if d > 0 else 0.0
    idle = sorted(rate(w) for w in logger.windows if w.get("kind") == "idle")
    if not idle:
        return []
    ref = idle[len(idle) // 2]
    return [f"{w['phase']}#{w.get('repeat')} ({rate(w):.2f} Hz vs idle {ref:.2f} Hz)"
            for w in logger.windows if w.get("kind") != "idle" and rate(w) < frac * ref]


def rate_stats(ts: list[float], t0: float, t1: float) -> dict:
    el = max(t1 - t0, 0.0)
    edges = [t0] + sorted(ts) + [t1]
    gap = max((b - a for a, b in zip(edges, edges[1:])), default=el)
    return {"n": len(ts), "elapsed_s": el, "rate_hz": len(ts) / el if el > 0 else 0.0, "max_gap_s": gap}


def mean_std(v: list[float]):
    n = len(v)
    if n == 0:
        return math.nan, math.nan
    m = sum(v) / n
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else math.nan
    return m, s


def loop_until(step, deadline: float) -> int:
    k = 0
    while time.monotonic() < deadline:
        step(k)
        k += 1
    return k


def idle_until(deadline: float) -> int:
    while True:
        r = deadline - time.monotonic()
        if r <= 0:
            return 0
        time.sleep(min(0.2, r))


def die_temps(read=None) -> dict:
    """Zynq AMS die temperatures {'pl','ps','remote'} (deg C) or {} (laptop / dry run)."""
    try:
        if read is None:
            import board_env
            read = board_env.read_die_temp
        ch = read().get("channels_c", {})
    except Exception:  # noqa: BLE001
        return {}
    return {k: ch[n] for k, n in (("pl", "pl_temp"), ("ps", "ps_temp"), ("remote", "remote_temp")) if n in ch}


# ---- protocol (generalized: N named configurations) --------------------------------------------
def build_schedule(repeats: int, names: list[str], idle_s: float, run_s: float,
                   order_seed: int | None = None) -> list[dict]:
    """Per repeat: the configurations in the given order (order_seed None) or a seeded random permutation,
    each preceded by an idle phase idle_<j>; one closing idle_post."""
    import random
    rng = random.Random(order_seed) if order_seed is not None else None
    sch = []
    for r in range(1, repeats + 1):
        perm = list(names)
        if rng is not None:
            rng.shuffle(perm)
        for j, n in enumerate(perm, 1):
            sch.append({"repeat": r, "phase": f"idle_{j}", "kind": "idle", "dur": idle_s})
            sch.append({"repeat": r, "phase": n, "kind": "run", "dur": run_s})
    sch.append({"repeat": repeats, "phase": "idle_post", "kind": "idle", "dur": idle_s})
    return sch


def n_phases(repeats: int, n_configs: int) -> int:
    return repeats * 2 * n_configs + 1


def _f(x, nd=6):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _num_or_nan(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _mean_or_nan(vals) -> float:
    v = [_num_or_nan(x) for x in vals]
    v = [x for x in v if x == x]
    return sum(v) / len(v) if v else math.nan


NUM_SUMMARY = ["p_idle_w", "p_run_w", "dp_w", "images", "run_duration_s", "time_per_image_s",
               "energy_per_image_j", "energy_per_image_mj", "temp_pl_c", "idle_temp_pl_c"]


def summarize(phases: list[dict], repeats: int, names: list[str]) -> list[dict]:
    """phases in schedule order (repeat, phase, kind, mean_w, duration_s, images, temp_pl_c). Returns
    per (repeat, config) dicts + per config 'mean' and 'std' dicts (floats / NaN)."""
    per = []
    for i, run in enumerate(phases):
        if run["kind"] != "run":
            continue
        ref = [phases[j] for j in (i - 1, i + 1) if 0 <= j < len(phases) and phases[j]["kind"] == "idle"]
        p_idle = (sum(p["mean_w"] for p in ref) / len(ref)) if ref else math.nan
        dp = run["mean_w"] - p_idle
        imgs = run["images"]
        tpi = run["duration_s"] / imgs if imgs else math.nan
        e = dp * tpi
        per.append({"row_kind": "repeat", "repeat": run["repeat"], "config": run["phase"],
                    "p_idle_ref": "+".join(f"{p['phase']}#{p['repeat']}" for p in ref),
                    "p_idle_w": p_idle, "p_run_w": run["mean_w"], "dp_w": dp, "images": imgs,
                    "run_duration_s": run["duration_s"], "time_per_image_s": tpi, "energy_per_image_j": e,
                    "energy_per_image_mj": e * 1e3, "temp_pl_c": _num_or_nan(run.get("temp_pl_c")),
                    "idle_temp_pl_c": _mean_or_nan([p.get("temp_pl_c") for p in ref])})
    agg = []
    for n in names:
        rows = [p for p in per if p["config"] == n]
        for stat in ("mean", "std"):
            row = {"row_kind": stat, "repeat": "all", "config": n, "p_idle_ref": ""}
            for k in NUM_SUMMARY:
                vals = [float(p[k]) for p in rows if not (isinstance(p[k], float) and math.isnan(p[k]))]
                if vals:
                    m, s = mean_std(vals)
                    row[k] = m if stat == "mean" else s
            agg.append(row)
    return per + agg


SENSOR_FIELDS = ["measurement", "rail", "sensor_backend", "sensor_device", "sensor_limits",
                 "rate_requested_hz", "rate_achieved_hz", "max_gap_s", "read_errors"]
SAMPLE_FIELDS = SENSOR_FIELDS + ["repeat", "phase", "config", "t_mono_s", "t_rel_s", "t_epoch", "t_utc",
                                 "power_w", "current_a", "voltage_v", "raw"]
PHASE_FIELDS = SENSOR_FIELDS + ["repeat", "phase", "kind", "config", "workload", "start_utc", "stop_utc",
                                "start_local", "stop_local", "start_epoch", "stop_epoch", "phase_s", "images",
                                "power_mean_w", "power_std_w", "n_samples", "phase_rate_hz", "phase_max_gap_s",
                                "phase_order", "order_seed", "temp_pl_start_c", "temp_pl_end_c",
                                "temp_ps_start_c", "temp_ps_end_c", "temp_remote_start_c", "temp_remote_end_c"]
SUMMARY_FIELDS = SENSOR_FIELDS + ["row_kind", "repeat", "n_repeats", "config", "workload", "p_idle_rule",
                                  "energy_rule", "p_idle_ref", "phase_order", "order_seed"] + NUM_SUMMARY
P_IDLE_RULE = "mean of the phase means of the two idle phases bracketing the run phase"
ENERGY_RULE = "E_sys = (P_run - P_idle) x phase duration / images completed (batch 1, e2e loop)"


def _check_source(ctx, sensor: Sensor):
    if ctx.source == bc.SOURCE_HW and not sensor.is_hw:
        raise SystemExit("REFUSED: mock sensor in a hardware run")
    if ctx.source != bc.SOURCE_HW and sensor.is_hw:
        raise SystemExit("REFUSED: real INA260 samples in a dry run")


def _sensor_cols(sensor: Sensor, logger: PowerLogger, ov: dict) -> dict:
    return {"measurement": LABEL, "rail": RAIL, "sensor_backend": sensor.backend,
            "sensor_device": sensor.device, "sensor_limits": json.dumps(sensor.limits, sort_keys=True),
            "rate_requested_hz": _f(logger.rate_hz, 3), "rate_achieved_hz": _f(ov["rate_hz"], 3),
            "max_gap_s": _f(ov["max_gap_s"], 4), "read_errors": logger.read_errors}


def _phase_stats(logger: PowerLogger, w: dict) -> dict:
    ss = logger.window_samples(w)
    m, sd = mean_std([s["r"].power_w for s in ss])
    rs = rate_stats([s["t_mono"] for s in ss], w["start"]["mono"], w["stop"]["mono"])
    return {"mean_w": m, "std_w": sd, "n": len(ss), "rate_hz": rs["rate_hz"],
            "max_gap_s": rs["max_gap_s"], "duration_s": w["stop"]["mono"] - w["start"]["mono"]}


def run_power_protocol(ctx, net: str, workloads: dict, *, sensor="auto", phase_s: float = DEFAULT_PHASE_S,
                       idle_s: float | None = None, repeats: int = DEFAULT_REPEATS,
                       rate_hz: float = DEFAULT_RATE_HZ, prefix: str = PREFIX, tag: str = "",
                       order_seed: int | None = None, temp_fn=None, sensor_kw: dict | None = None,
                       label: str = "baseline") -> dict:
    """Run the protocol and write the three CSVs. Returns a summary dict.

    ctx        .source, .out_dir, .meta(net, config, duration_s, num_inferences) -> row metadata
    workloads  {config: (fn(deadline_monotonic) -> images, workload label)} in their default order
    sensor     'auto' | 'hwmon' | 'platformstats' | 'i2c' | 'mock' | a Sensor instance
    """
    dry = ctx.source != bc.SOURCE_HW
    if not isinstance(sensor, Sensor):
        sensor, plog = discover(sensor, allow_hw=not dry, allow_mock=dry, mock_kw=sensor_kw)
        for line in plog:
            print(f"[{label} power] probe: {line}")
    _check_source(ctx, sensor)
    out = Path(ctx.out_dir)
    names = {k: out / f"{prefix}_{k}{tag}.csv" for k in ("samples", "phases", "summary")}
    for p in names.values():
        bc.check_output_path(p, ctx.source)
    cfgs = list(workloads)
    sch = build_schedule(repeats, cfgs, idle_s if idle_s is not None else phase_s, phase_s, order_seed)
    order_txt = " | ".join(",".join(p["phase"] for p in sch if p["repeat"] == r and p["kind"] == "run")
                           for r in range(1, repeats + 1))
    order_cols = {"phase_order": ("fixed: " if order_seed is None else "random: ") + order_txt,
                  "order_seed": "" if order_seed is None else order_seed}
    tot = sum(p["dur"] for p in sch)
    print(f"[{label} power] {LABEL} on {RAIL}; sensor {sensor.describe()}; rate {rate_hz:g} Hz; "
          f"{len(sch)} phases, {tot:.0f} s; net {net}; configs {cfgs}; source={ctx.source}")
    if dry:
        print(f"[{label} power] DRY RUN: mock sensor -- NOT measurements.")
    if temp_fn is None:
        temp_fn = (lambda: {}) if dry else die_temps
    temp_rec = []
    logger = PowerLogger(sensor, rate_hz)
    with logger:
        for p in sch:
            tmp0 = temp_fn()
            extra = (f"{label} {LABEL} phase={p['phase']} repeat={p['repeat']}/{repeats} net={net} "
                     f"window={p['dur']:g}s")
            s0 = logger.begin(p["phase"], extra)
            fn = idle_until if p["kind"] == "idle" else workloads[p["phase"]][0]
            res = fn(s0["mono"] + p["dur"])
            imgs = res["images"] if isinstance(res, dict) else res
            w = logger.end(p["phase"], s0, f"{extra} images={imgs}", repeat=p["repeat"], kind=p["kind"],
                           images=imgs)
            temp_rec.append((tmp0, temp_fn()))
            st = _phase_stats(logger, w)
            print(f"  [{label} power] {p['phase']}#{p['repeat']}: mean {_f(st['mean_w'], 4) or 'n/a'} W "
                  f"(std {_f(st['std_w'], 4) or 'n/a'}, n={st['n']}, {st['rate_hz']:.2f} Hz) images={imgs}",
                  flush=True)
    ov = logger.overall()
    base = _sensor_cols(sensor, logger, ov)
    starved = undersampled_phases(logger)
    phases, prow = [], []
    for w, (tmp0, tmp1) in zip(logger.windows, temp_rec):
        st = _phase_stats(logger, w)
        cfg = w["phase"] if w["kind"] == "run" else ""
        phases.append({"repeat": w["repeat"], "phase": w["phase"], "kind": w["kind"], "mean_w": st["mean_w"],
                       "duration_s": st["duration_s"], "images": w["images"],
                       "temp_pl_c": _mean_or_nan([tmp0.get("pl"), tmp1.get("pl")])})
        r = ctx.meta(net, cfg, st["duration_s"], w["images"])
        r.update(base, repeat=w["repeat"], phase=w["phase"], kind=w["kind"], config=cfg,
                 workload=workloads[cfg][1] if cfg else "none (sleep)",
                 start_utc=w["start"]["utc"], stop_utc=w["stop"]["utc"], start_local=w["start"]["local"],
                 stop_local=w["stop"]["local"], start_epoch=f"{w['start']['epoch']:.3f}",
                 stop_epoch=f"{w['stop']['epoch']:.3f}", phase_s=f"{st['duration_s']:.3f}", images=w["images"],
                 power_mean_w=_f(st["mean_w"]), power_std_w=_f(st["std_w"]), n_samples=st["n"],
                 phase_rate_hz=_f(st["rate_hz"], 3), phase_max_gap_s=_f(st["max_gap_s"], 4), **order_cols,
                 **{f"temp_{k}_{e}_c": _f(float(t[k]), 3) for k in ("pl", "ps", "remote")
                    for e, t in (("start", tmp0), ("end", tmp1)) if k in t})
        prow.append(r)
    summ = summarize(phases, repeats, cfgs)
    srows = []
    for s in summ:
        r = ctx.meta(net, s["config"], "", s["images"] if s["row_kind"] == "repeat" else "")
        r.update(base, row_kind=s["row_kind"], repeat=s["repeat"], n_repeats=repeats, config=s["config"],
                 workload=workloads[s["config"]][1], p_idle_rule=P_IDLE_RULE, energy_rule=ENERGY_RULE,
                 p_idle_ref=s.get("p_idle_ref", ""), **order_cols)
        for k in NUM_SUMMARY:
            if k in s:
                v = s[k]
                r[k] = v if isinstance(v, int) else _f(v, 9 if ("_j" in k or "time" in k) else 6)
        srows.append(r)
    samples = []
    for smp in logger.samples:
        w = next((w for w in logger.windows if w["start"]["mono"] <= smp["t_mono"] < w["stop"]["mono"]), None)
        cfg = w["phase"] if w and w["kind"] == "run" else ""
        r = ctx.meta(net, cfg, "", "")
        rd = smp["r"]
        r.update(base, repeat=w["repeat"] if w else "", phase=w["phase"] if w else "", config=cfg,
                 t_mono_s=f"{smp['t_mono']:.6f}", t_rel_s=f"{smp['t_mono'] - logger.t_start:.6f}",
                 t_epoch=f"{smp['t_epoch']:.6f}",
                 t_utc=_dt.datetime.fromtimestamp(smp["t_epoch"], _dt.timezone.utc).isoformat(timespec="milliseconds"),
                 power_w=f"{rd.power_w:.6f}", current_a=_f(rd.current_a, 6), voltage_v=_f(rd.voltage_v, 6),
                 raw=rd.raw)
        samples.append(r)
    bc.write_csv(names["samples"], samples, SAMPLE_FIELDS, ctx.source)
    bc.write_csv(names["phases"], prow, PHASE_FIELDS, ctx.source)
    bc.write_csv(names["summary"], srows, SUMMARY_FIELDS, ctx.source)
    print(f"[{label} power] achieved {ov['rate_hz']:.2f} Hz of {rate_hz:g} Hz ({ov['n']} samples / "
          f"{ov['elapsed_s']:.2f} s), max gap {ov['max_gap_s']:.3f} s, read errors {logger.read_errors}")
    for s in summ:
        if s["row_kind"] != "std" and "dp_w" in s:
            print(f"  [{label} power]{' DRY RUN (mock sensor)' if dry else ''} {s['row_kind']} {s['repeat']} "
                  f"{s['config']}: P_idle {_f(s['p_idle_w'], 4)} W, P_run {_f(s['p_run_w'], 4)} W, "
                  f"dP {_f(s['dp_w'], 4)} W, time/image {_f(s['time_per_image_s'], 9)} s, "
                  f"E_sys {_f(s['energy_per_image_mj'], 6)} mJ/image ({LABEL})")
    if starved and not dry:
        raise UndersampledError("run phase(s) sampled far below the idle rate: " + "; ".join(starved))
    return {"label": LABEL, "rail": RAIL, "source": ctx.source, "sensor": sensor.describe(),
            "rate_achieved_hz": ov["rate_hz"], "max_gap_s": ov["max_gap_s"], "read_errors": logger.read_errors,
            "phases": phases, "summary": summ, "files": {k: str(v) for k, v in names.items()},
            "undersampled": starved}


# ---- sensor check CLI ---------------------------------------------------------------------------
class SensorOnlyContext:
    def __init__(self, source: str, out_dir=None):
        self.source = source
        self.out_dir = bc.resolve_out_dir(source, out_dir)
        self.board_id = bc.board_id_default() if source == bc.SOURCE_HW else ""

    def meta(self, net="", config="", duration_s="", num_inferences="") -> dict:
        return {"timestamp": bc.utc_now(), "git_commit": "", "git_dirty": True, "board_id": self.board_id,
                "net": net, "layer": "", "source": self.source,
                "duration_s": duration_s if duration_s == "" else f"{float(duration_s):.3f}",
                "num_inferences": num_inferences, "board_hostname": socket.gethostname(),
                "backend": "sensor_only", **bc.env_meta(self.source, True, {})}


def sample_only(ctx, seconds: float, sensor: Sensor, rate_hz: float = DEFAULT_RATE_HZ,
                tag: str = "_sensorcheck") -> dict:
    _check_source(ctx, sensor)
    logger = PowerLogger(sensor, rate_hz)
    with logger:
        s0 = logger.begin("sample_only", f"sensor check {LABEL} {seconds:g}s")
        idle_until(s0["mono"] + seconds)
        w = logger.end("sample_only", s0, "sensor check", repeat=0, kind="idle", images=0)
    st = _phase_stats(logger, w)
    vals = [s["r"].power_w for s in logger.samples]
    n_changes = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
    r = ctx.meta("", "", st["duration_s"], 0)
    r.update(_sensor_cols(sensor, logger, logger.overall()), repeat=0, phase="sample_only", kind="idle",
             workload="none (sleep)", start_utc=w["start"]["utc"], stop_utc=w["stop"]["utc"],
             phase_s=f"{st['duration_s']:.3f}", images=0, power_mean_w=_f(st["mean_w"]),
             power_std_w=_f(st["std_w"]), n_samples=st["n"], phase_rate_hz=_f(st["rate_hz"], 3),
             phase_max_gap_s=_f(st["max_gap_s"], 4))
    bc.write_csv(Path(ctx.out_dir) / f"{PREFIX}_phases{tag}.csv", [r], PHASE_FIELDS, ctx.source)
    print(f"[sensor check] {LABEL}: mean {_f(st['mean_w'], 4)} W std {_f(st['std_w'], 4)} W, {st['n']} samples, "
          f"{st['rate_hz']:.2f} Hz of {rate_hz:g}, max gap {st['max_gap_s']:.3f} s, value changes {n_changes}, "
          f"read errors {logger.read_errors}")
    return {"mean_w": st["mean_w"], "n": st["n"], "value_changes": n_changes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-sensors", action="store_true")
    mode.add_argument("--sample-only", action="store_true")
    ap.add_argument("--sensor", default="auto", choices=("auto",) + SENSOR_ORDER + ("mock",))
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    if a.list_sensors:
        for line in list_sensors():
            print(line)
        return 0
    try:
        sensor, plog = discover(a.sensor, allow_hw=True, allow_mock=a.sensor == "mock")
    except SensorUnavailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    for line in plog:
        print(f"[sensor check] probe: {line}")
    src = bc.SOURCE_HW if sensor.is_hw else bc.SOURCE_DRYRUN
    sample_only(SensorOnlyContext(src, a.out_dir), a.seconds, sensor, a.rate_hz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
