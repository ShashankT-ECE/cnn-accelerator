#!/usr/bin/env python3
"""B1/B2 SOM-rail power logger: on-board INA260 (VCC_SOM), sampled remotely on the KV260.

    ./session.sh py power_log.py --list-sensors                    # probe only, writes nothing
    ./session.sh py power_log.py --sample-only --seconds 10        # sensor check (CSV + stats)
    ./session.sh py power_log.py --protocol --net lenet5 [--phase-s 60] [--repeats 3]
                                 [--rate-hz 10] [--cpu-kind cpu_int8_ref] [--cpu-threads 1]
    python3 power_log.py --protocol --backend model --phase-s 2    # dry run -> results/dryrun/

LABEL: every number is "SOM-rail power (INA260)" — the SOM rail (VCC_SOM) as reported by the
on-board INA260. It is NOT accelerator-only power and NOT board input power. It is the only power
measurement of the V2 board sessions (EXPERIMENTS.md B1).

Sensor backends (--sensor auto tries a-c in order on the board; a dry run uses d only):
  a hwmon          /sys/class/hwmon/*/name starting with "ina260": power1_input (µW),
                   curr1_input (mA), in1_input (mV) if present; update_interval (ms) recorded
  b platformstats  `platformstats -p`, parsed defensively; the matched raw line is recorded
  c i2c            smbus2 on /dev/i2c-*: address 0x40 first, then 0x41-0x4F; identified by
                   Manufacturer ID (0xFE) == 0x5449 and Die ID (0xFF) == 0x2270; reads power
                   (0x03, LSB 10 mW), current (0x01, signed, 1.25 mA), bus voltage (0x02, 1.25 mV)
                   and configuration (0x00, averaging / conversion times). STRICTLY READ-ONLY: the
                   bus object is wrapped so that only register reads exist (no config/mask
                   writes; the kernel ina2xx driver may own the device). Note an SMBus register
                   read itself sets the INA260 register pointer (part of every read transaction).
  d mock           deterministic, seeded; per phase-kind levels (idle / accel / cpu) + optional
                   noise. Dry runs only: rows source=dryrun_model, written only under a
                   `dryrun` directory (board_common guards). Mock numbers are NOT measurements.

Protocol (per repeat, default 3 repeats, all durations configurable; B1 default with control):
    idle_pre -> accel -> idle_mid -> control -> idle_ctl -> cpu   ... then one idle_post
    (--order fixed, the default of build_schedule). --order random --order-seed S (what
    run_sessions.py uses): in every repeat the run phases accel / control / cpu come in a seeded
    random permutation (each still bracketed by idle phases; the idle phases keep their positional
    names idle_pre / idle_mid / idle_ctl), so a slow drift does not always hit the same kind;
    the order and seed are recorded in the phase and summary rows (phase_order, order_seed).
    (60 s phases: 6R+1 = 19 phases = 19 min per net; --no-control: idle_pre/accel/idle_mid/cpu,
    4R+1 = 13 min; B2 per clock: idle_pre/accel/idle_mid only)
Every phase prints START / STOP banners (same style as exp_b1_power.py). Workloads are callables
fn(deadline_monotonic) -> images completed, or a dict {"images": n, ...extras} (AccelWorkload /
ControlWorkload / cpu_workload). Per phase: mean, std (ddof=1), n samples, achieved rate, max gap,
duration, images (+ TOTAL_CYC stats and start->done median for accel, pacing for control).

Accelerator loop (AccelWorkload, both accel and control use the SAME host loop): per image
GosDevice.infer (input write, pre-start clear, CTRL.start, STATUS poll, LOGIT read; host path
safe or fast), TOTAL_CYC lo/hi read, PS dequant + argmax (board_common.predict).
CONTROL (ControlWorkload): the same host loop with the accelerator NOT started
(GosDevice.control_step): input write, soft_reset write + clear poll (what infer does before
every start), a STATUS poll spin for the median start->done time of the most recent accel phase
(calibrated before the first phase), LOGIT read, TOTAL_CYC read, PS dequant + argmax. CTRL.start
is never written; STATUS != 0 or TOTAL_CYC != 0 during the control aborts the step. So
dP_accel - dP_control isolates what the PL compute adds on top of the host-loop activity.

P_idle choice (documented, fixed): for each run phase (accel, control or cpu) P_idle is the
equal-weight mean of the phase means of the two idle phases that bracket it in time (accel r:
idle_pre r + idle_mid r; control r: idle_mid r + idle_ctl r; cpu r: idle_ctl r (idle_mid r without
control) + idle_pre r+1, or idle_post after the last repeat). This cancels a linear drift of the
idle level. If only one bracketing idle exists it is used alone (recorded in p_idle_ref).
    dP = P_run - P_idle;  time/image = phase duration / images;
    energy/image = dP x time/image (J and mJ)                          (every run kind)
Accelerator energy two ways (all computed here, per repeat; f = f_used = the pl_clk0 PLL
read-back, the clock of record (DECISIONS D20); f_meas (exp_fclk_cal.py) is a cross-check only;
f_readback_mhz / f_used_mhz / f_used_source recorded):
    E_sys  = dP_accel x time/image (host loop included)       accel_e_sys_mj (= energy_per_image)
    t_PL   = TOTAL_CYC / f  (TOTAL_CYC = hardware counter, median over the phase's images)
    E_comp = dP_accel x t_PL (compute-only)                   accel_e_comp_mj
    duty   = t_PL / time/image                                accel_duty
  control-subtracted variant (only with a control phase):
    dP_net = dP_accel - dP_control                            accel_dp_net_w
    E_sys,net = dP_net x time/image;  E_comp,net = dP_net x t_PL   accel_e_sys_net_mj / _e_comp_net_mj
The mean / std rows are the mean and sample std (ddof=1) of the per-repeat values (energy mean =
mean of per-repeat energies).

Outputs (in out_dir; dry run -> results/dryrun/):
  <prefix>_samples<tag>.csv   every sample (monotonic + UTC time, phase tag, W / A / V, raw)
  <prefix>_phases<tag>.csv    one row per phase
  <prefix>_summary<tag>.csv   one row per repeat + mean + std
with <prefix> = hw_b1_power_ina260 (B2 passes e.g. hw_b2_power_ina260 and tag _200mhz).

Session wiring (one call):
    summary = power_log.run_power_protocol(ctx, "lenet5", cpu_fn=power_log.cpu_workload(runner, xs))
"""
from __future__ import annotations

import os
import sys


def _preset_blas_threads(argv):
    """cpu/cpu_infer requires OPENBLAS/OMP thread counts fixed BEFORE numpy is imported
    (only when run as a script; an importing session runner sets them itself)."""
    n = "1"
    for i, t in enumerate(argv):
        if t == "--cpu-threads" and i + 1 < len(argv):
            n = argv[i + 1]
        elif t.startswith("--cpu-threads="):
            n = t.split("=", 1)[1]
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = n


if __name__ == "__main__":
    _preset_blas_threads(sys.argv[1:])

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import board_common as bc  # noqa: E402

LABEL = "SOM-rail power (INA260)"
RAIL = "VCC_SOM"
PREFIX_B1 = "hw_b1_power_ina260"
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


# ---- sensors ---------------------------------------------------------------------------------
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
    """(power_w, raw_line) from `platformstats -p` output, or None. The exact format is not
    specified here (ASSUMPTION): a line mentioning power with a number + unit (uW/mW/W);
    preference: 'SOM' + 'power', then 'total' + 'power', then any 'power' line."""
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
    """Exposes only SMBus register READS of a bus object; every other attribute (write_*,
    process_call, block writes, ...) raises. The INA260 config/mask registers are never written."""
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
    """SMBus word reads are little-endian; INA260 registers are big-endian."""
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
    d = {"config_reg": f"0x{cfg:04X}", "averages": avg, "vbus_ct_us": vct, "ishunt_ct_us": ict,
         "mode": mode}
    if mode == 7:   # continuous shunt + bus: one power update every avg * (vct + ict)
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


def clock_tag(mhz: float) -> str:
    """B2 per-clock output tag: 199.998001 -> '200mhz' (= run_sessions.clock_tag)."""
    return f"{int(round(float(mhz)))}mhz"


PREFIX_B2 = "hw_b2_power_ina260"
PHASE_KINDS = ("idle", "accel", "control", "cpu")
RUN_KINDS = ("accel", "control", "cpu")


def phase_kind(phase: str) -> str:
    for k in PHASE_KINDS:
        if phase.startswith(k):
            return k
    return "idle"


class MockSensor(Sensor):
    """Dry-run sensor. levels_w: {'idle','accel','control','cpu'} -> W (synthetic, NOT measured);
    noise_w: std of seeded gaussian noise (0 = exact levels)."""
    backend = "mock"
    is_hw = False

    def __init__(self, levels_w: dict | None = None, noise_w: float = 0.0, seed: int = 0):
        super().__init__()
        import random
        self.levels = dict(idle=1.0, accel=1.25, control=1.05, cpu=1.5)
        self.levels.update(levels_w or {})
        self.noise_w = float(noise_w)
        self.rng = random.Random(seed)
        self.device = f"mock levels={self.levels} noise_w={self.noise_w} seed={seed}"
        self.limits["note"] = "synthetic dry-run sensor, not a measurement"

    def read(self, phase: str = "") -> Reading | None:
        w = self.levels[phase_kind(phase)]
        if self.noise_w:
            w += self.rng.gauss(0.0, self.noise_w)
        return Reading(w, None, None, f"mock phase={phase}")


# ---- discovery ---------------------------------------------------------------------------------
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
    """INA260 candidates: [{busnum, addr, bus}] (bus = ReadOnlyBus, left open). 0x40 is probed on
    every bus first; the other addresses only if no 0x40 match was found."""
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
        if found and k > 0:          # other addresses only if nothing answered as INA260 at 0x40
            break
        for addr in pas:
            for n in buses:
                if n not in opened:
                    try:
                        opened[n] = ReadOnlyBus(factory(n))
                    except SensorUnavailable as e:      # e.g. smbus2 missing: same for every bus
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
                log.append(f"i2c-{n} 0x{addr:02X}: mfr=0x{mfr:04X} die=0x{die:04X} "
                           f"{'INA260' if ok else 'not INA260'}")
                if ok:
                    found.append({"busnum": n, "addr": addr, "bus": bus})
                    if stop_at_first:
                        return found
    return found


def discover(name: str = "auto", *, allow_hw: bool = True, allow_mock: bool = False,
             hwmon_root="/sys/class/hwmon", which=shutil.which, run=subprocess.run,
             i2c_glob="/dev/i2c-*", smbus_factory=None, mock_kw: dict | None = None):
    """Return (sensor, probe_log). auto: hwmon -> platformstats -> i2c (if allow_hw), else mock
    (only if allow_mock). A real sensor is refused when not allow_hw, the mock when not
    allow_mock (a mock sensor must never produce source=hw rows)."""
    log: list[str] = []
    if name == "mock":
        if not allow_mock:
            raise SensorUnavailable("mock sensor refused: hardware run (rows would be source=hw)")
        return MockSensor(**(mock_kw or {})), ["mock requested"]
    if name != "auto" and not allow_hw:
        raise SensorUnavailable(f"real sensor '{name}' refused in a dry run (model accelerator)")
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


def list_sensors(**kw) -> list[str]:
    """Probe every backend (reads only) and return the probe log."""
    log = []
    d = find_hwmon(kw.get("hwmon_root", "/sys/class/hwmon"))
    log.append(f"hwmon: {[str(x) for x in d] or 'none'}")
    for x in d:
        s = HwmonSensor(x)
        r = s.read()
        log.append(f"  {s.describe()} -> {r.raw if r else 'read failed'}")
    exe = kw.get("which", shutil.which)("platformstats")
    log.append(f"platformstats: {exe or 'not on PATH'}")
    if exe:
        r = PlatformstatsSensor(exe, run=kw.get("run", subprocess.run)).read()
        log.append(f"  -> {r.raw if r else 'no power line parsed'}")
    c = probe_i2c(kw.get("i2c_glob", "/dev/i2c-*"), kw.get("smbus_factory"), log=log,
                  stop_at_first=False)
    for x in c:
        s = I2cSensor(x["bus"], x["busnum"], x["addr"])
        r = s.read()
        log.append(f"  {s.describe()} -> {r.raw if r else 'read failed'}")
    return log


# ---- sampler -----------------------------------------------------------------------------------
def stamp(mono=None, wall=None) -> dict:
    now = time.time() if wall is None else wall
    return {"mono": time.monotonic() if mono is None else mono, "epoch": now,
            "utc": _dt.datetime.fromtimestamp(now, _dt.timezone.utc).isoformat(timespec="milliseconds"),
            "local": _dt.datetime.fromtimestamp(now).astimezone().isoformat(timespec="milliseconds")}


def banner(what: str, s: dict, extra: str = ""):
    print(f"\n===== {what} {extra} | UTC {s['utc']} | local {s['local']} | epoch {s['epoch']:.3f} "
          f"=====", flush=True)


class PowerLogger:
    """Background sampling thread at a fixed rate. Samples carry monotonic + UTC epoch time.
    Phase windows are opened/closed atomically with the sampler (same lock), so a sample with
    start <= t < stop was taken while that phase was active."""

    def __init__(self, sensor: Sensor, rate_hz: float = DEFAULT_RATE_HZ):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        self.sensor, self.rate_hz = sensor, float(rate_hz)
        self.samples: list[dict] = []
        self.windows: list[dict] = []
        self.read_errors = 0
        self._phase = ""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.t_start = self.t_stop = None

    def start(self):
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("PowerLogger already started")
            self._stop.clear()
            self.t_start = time.monotonic()
            self._thread = threading.Thread(target=self._run, name="power_log", daemon=True)
            self._thread.start()

    def stop(self):
        th = self._thread
        if th is None:
            return
        self._stop.set()
        th.join(timeout=10)
        self.t_stop = time.monotonic()
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def _run(self):
        period = 1.0 / self.rate_hz
        nxt = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                t, te, ph = time.monotonic(), time.time(), self._phase
                try:
                    r = self.sensor.read(ph)
                except Exception:  # noqa: BLE001 - a sensor glitch must not kill the logger
                    r = None
                if r is None:
                    self.read_errors += 1
                else:
                    self.samples.append({"t_mono": t, "t_epoch": te, "live_phase": ph, "r": r})
            nxt += period
            now = time.monotonic()
            if nxt < now:          # fell behind: do not burst, restart the schedule (gap recorded)
                nxt = now
            self._stop.wait(nxt - now)

    def begin(self, phase: str, extra: str = "") -> dict:
        with self._lock:
            self._phase = phase
            s = stamp()
        banner("START", s, extra)
        return s

    def end(self, phase: str, s0: dict, extra: str = "", **info) -> dict:
        with self._lock:
            s1 = stamp()
            self._phase = ""
        banner("STOP ", s1, extra)
        w = {"phase": phase, "start": s0, "stop": s1, **info}
        self.windows.append(w)
        return w

    def window_samples(self, w: dict) -> list[dict]:
        a, b = w["start"]["mono"], w["stop"]["mono"]
        return [s for s in self.samples if a <= s["t_mono"] < b]

    def overall(self) -> dict:
        t0 = self.t_start
        t1 = self.t_stop if self.t_stop is not None else time.monotonic()
        return rate_stats([s["t_mono"] for s in self.samples], t0, t1)


def rate_stats(ts: list[float], t0: float, t1: float) -> dict:
    """achieved rate = samples / elapsed; max gap includes the window edges."""
    el = max(t1 - t0, 0.0)
    edges = [t0] + sorted(ts) + [t1]
    gap = max((b - a for a, b in zip(edges, edges[1:])), default=el)
    return {"n": len(ts), "elapsed_s": el, "rate_hz": len(ts) / el if el > 0 else 0.0,
            "max_gap_s": gap}


def mean_std(v: list[float]):
    n = len(v)
    if n == 0:
        return math.nan, math.nan
    m = sum(v) / n
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else math.nan
    return m, s


# ---- workloads ---------------------------------------------------------------------------------
def loop_until(step, deadline: float) -> int:
    """step(k) back to back until time.monotonic() >= deadline; returns completed calls."""
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


class AccelWorkload:
    """Accelerator phase callable. Per image: GosDevice.infer (counters not read), TOTAL_CYC
    lo/hi read (hardware counter, for E_comp / duty), PS dequant + argmax. Loads the net and runs
    `warmup` untimed inferences now (before any phase); their median start->done time is the
    initial control pacing. After each phase pace_ns = that phase's median start->done."""

    def __init__(self, dev, pkg, warmup: int = 20):
        import gos_driver as D
        self.dev, self.pkg = dev, pkg
        self._off_tc = D.OFF_TOTAL_CYC
        dev.load_net(pkg)
        self.x = pkg.x_act
        self.n = self.x.shape[0]
        runs = []
        for k in range(max(1, warmup)):
            r, _ = self._step(k)
            runs.append(r.t_run_ns)
        self.pace_ns = int(sorted(runs)[len(runs) // 2])
        self.pace_source = f"calibration: median start->done of {len(runs)} warm-up images"

    def _step(self, k: int):
        r = self.dev.infer(self.x[k % self.n], read_counters=False)
        tc = self.dev.read64(self._off_tc)
        bc.predict(r.logits, self.pkg.dequant)
        return r, tc

    def __call__(self, deadline: float) -> dict:
        runs, tcs = [], []
        k = 0
        while time.monotonic() < deadline:
            r, tc = self._step(k)
            runs.append(r.t_run_ns)
            tcs.append(tc)
            k += 1
        out = {"images": k}
        if k:
            runs.sort()
            tcs.sort()
            self.pace_ns = int(runs[k // 2])
            self.pace_source = "median start->done of the preceding accel phase"
            med = tcs[k // 2] if k % 2 else (tcs[k // 2 - 1] + tcs[k // 2]) / 2
            out.update(total_cyc_median=int(med) if float(med).is_integer() else med,
                       total_cyc_min=tcs[0], total_cyc_max=tcs[-1],
                       start_done_median_us=(runs[k // 2] if k % 2 else
                                             (runs[k // 2 - 1] + runs[k // 2]) / 2) / 1e3)
        return out


class ControlWorkload:
    """B1 control phase callable: the accelerator loop's host activity WITHOUT starting the
    accelerator (GosDevice.control_step + PS dequant), paced by accel.pace_ns per image."""

    def __init__(self, dev, pkg, accel: AccelWorkload):
        self.dev, self.pkg, self.accel = dev, pkg, accel
        self.x = pkg.x_act
        self.n = self.x.shape[0]

    def __call__(self, deadline: float) -> dict:
        pace = int(self.accel.pace_ns)
        k = 0
        polls = 0
        while time.monotonic() < deadline:
            logits, tc, pl_ = self.dev.control_step(self.x[k % self.n], pace)
            bc.predict(logits, self.pkg.dequant)
            polls += pl_
            k += 1
        return {"images": k, "pace_us": pace / 1e3, "pace_source": self.accel.pace_source,
                "total_cyc_max": 0 if k else "", "control_polls_mean": polls / k if k else ""}


def gos_workload(dev, pkg, warmup: int = 20) -> AccelWorkload:
    """Accelerator callable (AccelWorkload; kept for exp_b2_clock / older callers)."""
    return AccelWorkload(dev, pkg, warmup)


def cpu_workload(runner, xs, warmup: int = 1, phase_cores=None):
    """CPU callable from a cpu/cpu_infer runner and its input array (x_nchw or x_f32).

    phase_cores (set of core ids, multi-thread CPU kinds): the calling thread is allowed on these
    cores for the duration of the CPU phase only and its previous affinity is restored after it
    (the accelerator / control / idle phases stay on the step's pinned measurement core)."""
    n = xs.shape[0]
    step = lambda k: runner(xs[k % n])  # noqa: E731
    for k in range(warmup):
        step(k)
    if not phase_cores:
        return lambda deadline: loop_until(step, deadline)

    def phase(deadline):
        prev = os.sched_getaffinity(0)
        os.sched_setaffinity(0, phase_cores)
        try:
            return loop_until(step, deadline)
        finally:
            os.sched_setaffinity(0, prev)
    return phase


# ---- protocol ----------------------------------------------------------------------------------
def build_schedule(repeats: int, durations: dict, with_cpu: bool = True,
                   final_idle: bool = True, with_control: bool = False,
                   order_seed: int | None = None) -> list[dict]:
    """Phase list. order_seed None = the fixed order; else the run phases of every repeat in a
    seeded random permutation, each bracketed by idle phases."""
    if order_seed is not None:
        import random
        rng = random.Random(order_seed)
        runs = ["accel"] + (["control"] if with_control else []) + (["cpu"] if with_cpu else [])
        idle_names = ["idle_pre", "idle_mid", "idle_ctl"]
        sch = []
        for r in range(1, repeats + 1):
            perm = runs[:]
            rng.shuffle(perm)
            for j, kind in enumerate(perm):
                sch.append({"repeat": r, "phase": idle_names[j], "kind": "idle",
                            "dur": durations["idle"]})
                sch.append({"repeat": r, "phase": kind, "kind": kind, "dur": durations[kind]})
        # the last run phase needs its closing idle (always appended in random order)
        sch.append({"repeat": repeats, "phase": "idle_post", "kind": "idle", "dur": durations["idle"]})
        return sch
    sch = []
    for r in range(1, repeats + 1):
        sch.append({"repeat": r, "phase": "idle_pre", "kind": "idle", "dur": durations["idle"]})
        sch.append({"repeat": r, "phase": "accel", "kind": "accel", "dur": durations["accel"]})
        sch.append({"repeat": r, "phase": "idle_mid", "kind": "idle", "dur": durations["idle"]})
        if with_control:
            sch.append({"repeat": r, "phase": "control", "kind": "control",
                        "dur": durations["control"]})
            if with_cpu:
                sch.append({"repeat": r, "phase": "idle_ctl", "kind": "idle",
                            "dur": durations["idle"]})
        if with_cpu:
            sch.append({"repeat": r, "phase": "cpu", "kind": "cpu", "dur": durations["cpu"]})
    if (with_cpu or with_control) and final_idle:
        sch.append({"repeat": repeats, "phase": "idle_post", "kind": "idle",
                    "dur": durations["idle"]})
    return sch


def n_phases(repeats: int, with_cpu: bool = True, with_control: bool = False,
             final_idle: bool = True) -> int:
    per = 3 + (1 if with_control else 0) + (1 if with_control and with_cpu else 0) + (1 if with_cpu else 0)
    return repeats * per + (1 if (with_cpu or with_control) and final_idle else 0)


def _f(x, nd=6):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def summarize(phases: list[dict], repeats: int, clock_mhz=None) -> list[dict]:
    """phases (in schedule order): dicts with repeat, phase, kind, mean_w, duration_s, images
    (accel optionally total_cyc_median). clock_mhz: pl_clk0 (read back) for t_PL = TOTAL_CYC / f.
    Returns per-repeat dicts + 'mean' + 'std' dicts (numbers as floats / NaN)."""
    try:
        f_hz = float(clock_mhz) * 1e6 if clock_mhz not in (None, "") else math.nan
    except (TypeError, ValueError):
        f_hz = math.nan
    per = []
    for r in range(1, repeats + 1):
        row = {"row_kind": "repeat", "repeat": r}
        for kind in RUN_KINDS:
            idx = [i for i, p in enumerate(phases) if p["repeat"] == r and p["kind"] == kind]
            if not idx:
                continue
            i = idx[0]
            run = phases[i]
            ref = [phases[j] for j in (i - 1, i + 1)
                   if 0 <= j < len(phases) and phases[j]["kind"] == "idle"]
            p_idle = (sum(p["mean_w"] for p in ref) / len(ref)) if ref else math.nan
            dp = run["mean_w"] - p_idle
            imgs = run["images"]
            tpi = run["duration_s"] / imgs if imgs else math.nan
            e = dp * tpi
            row.update({f"{kind}_p_idle_ref": "+".join(f"{p['phase']}#{p['repeat']}" for p in ref),
                        f"{kind}_p_idle_w": p_idle, f"{kind}_p_run_w": run["mean_w"],
                        f"{kind}_dp_w": dp, f"{kind}_images": imgs,
                        f"{kind}_duration_s": run["duration_s"],
                        f"{kind}_time_per_image_s": tpi, f"{kind}_energy_per_image_j": e,
                        f"{kind}_energy_per_image_mj": e * 1e3})
            if kind == "accel":
                tc = run.get("total_cyc_median")
                t_pl = (float(tc) / f_hz) if tc not in (None, "") and f_hz == f_hz else math.nan
                row.update(accel_e_sys_mj=e * 1e3, accel_total_cyc=float(tc) if tc not in (None, "") else math.nan,
                           accel_t_pl_s=t_pl, accel_e_comp_mj=dp * t_pl * 1e3,
                           accel_duty=t_pl / tpi if tpi == tpi and tpi else math.nan)
        if "accel_dp_w" in row and "control_dp_w" in row:
            net = row["accel_dp_w"] - row["control_dp_w"]
            row.update(accel_dp_net_w=net,
                       accel_e_sys_net_mj=net * row["accel_time_per_image_s"] * 1e3,
                       accel_e_comp_net_mj=net * row["accel_t_pl_s"] * 1e3)
        per.append(row)
    agg = []
    for stat in ("mean", "std"):
        row = {"row_kind": stat, "repeat": "all"}
        for k in NUM_SUMMARY:
            vals = [p[k] for p in per if k in p and not (isinstance(p[k], float) and math.isnan(p[k]))]
            if vals:
                m, s = mean_std([float(v) for v in vals])
                row[k] = m if stat == "mean" else s
        agg.append(row)
    return per + agg


NUM_SUMMARY = [f"{k}_{f}" for k in RUN_KINDS for f in
               ("p_idle_w", "p_run_w", "dp_w", "images", "duration_s", "time_per_image_s",
                "energy_per_image_j", "energy_per_image_mj")] + [
    "accel_e_sys_mj", "accel_total_cyc", "accel_t_pl_s", "accel_e_comp_mj", "accel_duty",
    "accel_dp_net_w", "accel_e_sys_net_mj", "accel_e_comp_net_mj"]
SENSOR_FIELDS = ["measurement", "rail", "sensor_backend", "sensor_device", "sensor_limits",
                 "rate_requested_hz", "rate_achieved_hz", "max_gap_s", "read_errors"]
SAMPLE_FIELDS = SENSOR_FIELDS + ["repeat", "phase", "t_mono_s", "t_rel_s", "t_epoch", "t_utc",
                                 "power_w", "current_a", "voltage_v", "raw"]
PHASE_FIELDS = SENSOR_FIELDS + ["repeat", "phase", "kind", "workload", "start_utc", "stop_utc",
                                "start_local", "stop_local", "start_epoch", "stop_epoch",
                                "phase_s", "images", "power_mean_w", "power_std_w", "n_samples",
                                "phase_rate_hz", "phase_max_gap_s", "host_path", "total_cyc_median",
                                "total_cyc_min", "total_cyc_max", "start_done_median_us",
                                "pace_us", "pace_source", "phase_order", "order_seed"]
SUMMARY_FIELDS = SENSOR_FIELDS + ["row_kind", "repeat", "n_repeats", "p_idle_rule", "energy_rule",
                                  "host_path", "accel_p_idle_ref", "control_p_idle_ref",
                                  "cpu_p_idle_ref", "accel_workload", "control_workload",
                                  "cpu_workload", "phase_order", "order_seed",
                                  *bc.CLOCK_COLS] + NUM_SUMMARY
P_IDLE_RULE = "mean of the phase means of the two idle phases bracketing the run phase"
ENERGY_RULE = ("E_sys = dP_accel x time/image; E_comp = dP_accel x TOTAL_CYC/f (f = f_used_mhz: "
               "the pl_clk0 PLL read-back = clock of record, TOTAL_CYC median of the phase); duty = (TOTAL_CYC/f)/(time/image); "
               "_net: dP_accel - dP_control")


class SensorOnlyContext:
    """Row metadata for runs without a GosDevice (--sample-only, tests). Same columns as
    board_common.RunContext.meta; source = hw for a real sensor, dryrun_model for the mock."""

    def __init__(self, source: str, out_dir=None, board_id: str | None = None,
                 allow_dirty: bool = False):
        self.source = source
        self.info = bc.deploy_info()
        self.scripts_commit = self.info.get("commit", "")
        self.scripts_dirty = bool(self.info.get("dirty", True))
        self.dirty = self.scripts_dirty
        self.hostname = socket.gethostname()
        self.board_id = board_id or (bc.board_id_default() if source == bc.SOURCE_HW else "")
        self.out_dir = bc.resolve_out_dir(source, out_dir)
        self.packages: dict = {}
        self.args = argparse.Namespace(allow_dirty=allow_dirty)

    def meta(self, net: str = "", layer: str = "", duration_s="", num_inferences="",
             clock_mhz=None) -> dict:
        return {"timestamp": bc.utc_now(), "git_commit": self.scripts_commit,
                "git_dirty": self.dirty, "vivado_version": self.info.get("vivado_version", ""),
                "bitstream_sha256": "", "board_id": self.board_id, "net": net, "layer": layer,
                "clock_mhz": "" if clock_mhz is None else clock_mhz, "source": self.source,
                "duration_s": duration_s if duration_s == "" else f"{float(duration_s):.3f}",
                "num_inferences": num_inferences, "build_id_hw": "",
                "board_hostname": self.hostname, "clock_source": "not read (sensor only)",
                "scripts_commit": self.scripts_commit, "data_manifest_sha256": "",
                "data_git_commit": "", "data_git_dirty": "", "backend": "sensor_only",
                **bc.env_meta(self.source, self.dirty)}


def _check_source(ctx, sensor: Sensor):
    if ctx.source == bc.SOURCE_HW and not sensor.is_hw:
        raise SystemExit("REFUSED: mock sensor in a hardware run (rows would be source=hw)")
    if ctx.source != bc.SOURCE_HW and sensor.is_hw:
        raise SystemExit("REFUSED: real INA260 samples with a dry-run (model) accelerator")


def _sensor_cols(sensor: Sensor, logger: PowerLogger, ov: dict) -> dict:
    return {"measurement": LABEL, "rail": RAIL, "sensor_backend": sensor.backend,
            "sensor_device": sensor.device, "sensor_limits": json.dumps(sensor.limits, sort_keys=True),
            "rate_requested_hz": _f(logger.rate_hz, 3), "rate_achieved_hz": _f(ov["rate_hz"], 3),
            "max_gap_s": _f(ov["max_gap_s"], 4), "read_errors": logger.read_errors}


def _sample_rows(ctx, logger, sensor, net, clk, base) -> list[dict]:
    t_ref = logger.t_start
    rows = []
    wins = logger.windows
    for s in logger.samples:
        w = next((w for w in wins if w["start"]["mono"] <= s["t_mono"] < w["stop"]["mono"]), None)
        r = ctx.meta(net, "all", "", "", clock_mhz=clk)
        rd: Reading = s["r"]
        r.update(base, repeat=w["repeat"] if w else "", phase=w["phase"] if w else "",
                 t_mono_s=f"{s['t_mono']:.6f}", t_rel_s=f"{s['t_mono'] - t_ref:.6f}",
                 t_epoch=f"{s['t_epoch']:.6f}",
                 t_utc=_dt.datetime.fromtimestamp(s["t_epoch"], _dt.timezone.utc)
                 .isoformat(timespec="milliseconds"),
                 power_w=f"{rd.power_w:.6f}", current_a=_f(rd.current_a, 6),
                 voltage_v=_f(rd.voltage_v, 6), raw=rd.raw)
        rows.append(r)
    return rows


def _phase_stats(logger: PowerLogger, w: dict) -> dict:
    ss = logger.window_samples(w)
    m, sd = mean_std([s["r"].power_w for s in ss])
    rs = rate_stats([s["t_mono"] for s in ss], w["start"]["mono"], w["stop"]["mono"])
    return {"mean_w": m, "std_w": sd, "n": len(ss), "rate_hz": rs["rate_hz"],
            "max_gap_s": rs["max_gap_s"], "duration_s": w["stop"]["mono"] - w["start"]["mono"]}


def run_power_protocol(ctx, net: str, cpu_fn=None, *, accel_fn=None, sensor="auto",
                       out_dir=None, durations: dict | None = None, phase_s: float = DEFAULT_PHASE_S,
                       repeats: int = DEFAULT_REPEATS, rate_hz: float = DEFAULT_RATE_HZ,
                       final_idle: bool = True, prefix: str = PREFIX_B1, tag: str = "",
                       clock_mhz=None, accel_label: str = "", cpu_label: str = "",
                       label: str = "B1", sensor_kw: dict | None = None, control: bool = False,
                       control_fn=None, control_label: str = "",
                       order_seed: int | None = None) -> dict:
    """Run the SOM-rail power protocol and write the three CSVs. Returns a summary dict.

    ctx       board_common.RunContext (or SensorOnlyContext): provenance, source, out_dir
    net       net name ('lenet5' / 'cifar10'); accel_fn defaults to gos_workload(ctx.dev,
              ctx.package(net)) (loads the net + one warm-up inference before the first phase)
    cpu_fn    fn(deadline_monotonic) -> images, e.g. cpu_workload(runner, xs); None = no CPU phases
    control   True: control phases (ControlWorkload(ctx.dev, pkg, accel_fn) unless control_fn is
              given; accel_fn must then be an AccelWorkload for the pacing)
    sensor    'auto' | 'hwmon' | 'platformstats' | 'i2c' | 'mock' | a Sensor instance. auto = the
              real sensors in order on hardware, the mock in a dry run (never mixed)
    durations {'idle','accel','cpu'} seconds (default phase_s each); repeats; rate_hz
    prefix/tag  output names <prefix>_{samples,phases,summary}<tag>.csv
    """
    dry = ctx.source != bc.SOURCE_HW
    if not isinstance(sensor, Sensor):
        sensor, plog = discover(sensor, allow_hw=not dry, allow_mock=dry, mock_kw=sensor_kw)
        for line in plog:
            print(f"[{label} power] probe: {line}")
    _check_source(ctx, sensor)
    out = bc.resolve_out_dir(ctx.source, out_dir if out_dir is not None else ctx.out_dir)
    names = {k: out / f"{prefix}_{k}{tag}.csv" for k in ("samples", "phases", "summary")}
    for p in names.values():
        bc.check_output_path(p, ctx.source)
    dur = {k: float(phase_s) for k in PHASE_KINDS}
    dur.update(durations or {})
    if accel_fn is None:
        accel_fn = AccelWorkload(ctx.dev, ctx.package(net))
    if isinstance(accel_fn, AccelWorkload):
        accel_label = accel_label or (f"GosDevice.infer {net} + TOTAL_CYC read + PS dequant "
                                      f"({ctx.dev.host_path_desc})")
    if control and control_fn is None:
        if not isinstance(accel_fn, AccelWorkload):
            raise ValueError("control phases need an AccelWorkload accel_fn (pacing) or control_fn")
        control_fn = ControlWorkload(accel_fn.dev, accel_fn.pkg, accel_fn)
    with_control = control_fn is not None
    if with_control:
        control_label = control_label or (f"control: same host loop, accelerator NOT started, "
                                          f"STATUS poll spin paced to the accel start->done median "
                                          f"({ctx.dev.host_path_desc if getattr(ctx, 'dev', None) else ''})")
    host_path = getattr(getattr(ctx, "dev", None), "host_path", "")
    if clock_mhz is None and getattr(ctx, "dev", None) is not None:
        clock_mhz = ctx.dev.fclk0_mhz()
    clk = f"{clock_mhz:.6f}" if isinstance(clock_mhz, float) else clock_mhz
    # clock for t_PL / E_comp: the pl_clk0 PLL read-back (clock of record)
    if hasattr(ctx, "clock_cols") and clock_mhz not in (None, ""):
        ccols = ctx.clock_cols(float(clock_mhz))
        f_comp = float(ccols["f_used_mhz"])
    else:
        ccols = {"f_readback_mhz": clk if clk is not None else "", "f_used_mhz": clk if clk is not None else "",
                 "f_used_source": "f_readback" if clk not in (None, "") else ""}
        f_comp = clock_mhz
    work = {"idle": idle_until, "accel": accel_fn, "cpu": cpu_fn, "control": control_fn}
    wl_name = {"idle": "none (sleep)", "accel": accel_label or "accel callable",
               "cpu": cpu_label or "cpu callable", "control": control_label or "control callable"}
    sch = build_schedule(repeats, dur, with_cpu=cpu_fn is not None, final_idle=final_idle,
                         with_control=with_control, order_seed=order_seed)
    order_txt = " | ".join(",".join(p["kind"] for p in sch if p["repeat"] == r and p["kind"] != "idle")
                           for r in range(1, repeats + 1))
    order_cols = {"phase_order": ("fixed: " if order_seed is None else "random: ") + order_txt,
                  "order_seed": "" if order_seed is None else order_seed}
    tot = sum(p["dur"] for p in sch)
    print(f"[{label} power] {LABEL} on {RAIL}; sensor {sensor.describe()}; rate {rate_hz:g} Hz; "
          f"{len(sch)} phases, {tot:.0f} s; source={ctx.source}")
    if dry:
        print(f"[{label} power] DRY RUN: mock sensor + model accelerator — NOT measurements.")
    logger = PowerLogger(sensor, rate_hz)
    with logger:
        for p in sch:
            extra = (f"{label} {LABEL} phase={p['phase']} repeat={p['repeat']}/{repeats} "
                     f"net={net if p['kind'] != 'idle' else '-'} window={p['dur']:g}s")
            s0 = logger.begin(p["phase"], extra)
            res_w = work[p["kind"]](s0["mono"] + p["dur"])
            info = dict(res_w) if isinstance(res_w, dict) else {"images": res_w}
            imgs = info.pop("images")
            w = logger.end(p["phase"], s0, f"{extra} images={imgs}", repeat=p["repeat"],
                           kind=p["kind"], images=imgs, extras=info)
            st = _phase_stats(logger, w)
            print(f"  [{label} power] {p['phase']}#{p['repeat']}: {LABEL} mean "
                  f"{_f(st['mean_w'], 4) or 'n/a'} W (std {_f(st['std_w'], 4) or 'n/a'}, "
                  f"n={st['n']}, {st['rate_hz']:.2f} Hz) images={imgs}", flush=True)
    ov = logger.overall()
    base = _sensor_cols(sensor, logger, ov)
    phases, prow = [], []
    for w in logger.windows:
        st = _phase_stats(logger, w)
        ex = w.get("extras", {})
        phases.append({"repeat": w["repeat"], "phase": w["phase"], "kind": w["kind"],
                       "mean_w": st["mean_w"], "duration_s": st["duration_s"],
                       "images": w["images"], **ex})
        r = ctx.meta(net, "all", st["duration_s"],
                     w["images"], clock_mhz=clk)
        r.update(base, repeat=w["repeat"], phase=w["phase"], kind=w["kind"],
                 workload=wl_name[w["kind"]],
                 start_utc=w["start"]["utc"], stop_utc=w["stop"]["utc"],
                 start_local=w["start"]["local"], stop_local=w["stop"]["local"],
                 start_epoch=f"{w['start']['epoch']:.3f}", stop_epoch=f"{w['stop']['epoch']:.3f}",
                 phase_s=f"{st['duration_s']:.3f}", images=w["images"],
                 power_mean_w=_f(st["mean_w"]), power_std_w=_f(st["std_w"]), n_samples=st["n"],
                 phase_rate_hz=_f(st["rate_hz"], 3), phase_max_gap_s=_f(st["max_gap_s"], 4),
                 host_path=host_path if w["kind"] in ("accel", "control") else "", **order_cols,
                 **{k: (_f(v, 3) if isinstance(v, float) else v) for k, v in ex.items()
                    if k in PHASE_FIELDS})
        prow.append(r)
    summ = summarize(phases, repeats, f_comp)
    srows = []
    for s in summ:
        imgs = sum(int(s.get(f"{k}_images", 0) or 0) for k in RUN_KINDS) \
            if s["row_kind"] == "repeat" else ""
        r = ctx.meta(net, "all", "", imgs, clock_mhz=clk)
        r.update(base, row_kind=s["row_kind"], repeat=s["repeat"], n_repeats=repeats,
                 p_idle_rule=P_IDLE_RULE, energy_rule=ENERGY_RULE, host_path=host_path,
                 accel_p_idle_ref=s.get("accel_p_idle_ref", ""),
                 control_p_idle_ref=s.get("control_p_idle_ref", ""),
                 cpu_p_idle_ref=s.get("cpu_p_idle_ref", ""), accel_workload=wl_name["accel"],
                 control_workload=wl_name["control"] if with_control else "",
                 cpu_workload=wl_name["cpu"] if cpu_fn is not None else "", **order_cols, **ccols)
        for k in NUM_SUMMARY:
            if k in s:
                v = s[k]
                r[k] = (_f(v, 3) if (k.endswith("images") or k == "accel_total_cyc")
                        and s["row_kind"] != "repeat"
                        else v if isinstance(v, int) else
                        _f(v, 12 if k == "accel_t_pl_s" else 9 if "_j" in k or "time" in k else 6))
        srows.append(r)
    bc.write_csv(names["samples"], _sample_rows(ctx, logger, sensor, net, clk, base),
                 SAMPLE_FIELDS, ctx.source)
    bc.write_csv(names["phases"], prow, PHASE_FIELDS, ctx.source)
    bc.write_csv(names["summary"], srows, SUMMARY_FIELDS, ctx.source)
    print(f"[{label} power] {LABEL}: achieved {ov['rate_hz']:.2f} Hz of {rate_hz:g} Hz requested "
          f"({ov['n']} samples / {ov['elapsed_s']:.2f} s), max gap {ov['max_gap_s']:.3f} s, "
          f"read errors {logger.read_errors}")
    for s in summ:
        for k in RUN_KINDS:
            if f"{k}_dp_w" in s:
                print(f"  [{label} power]{' DRY RUN (mock sensor, not measured)' if dry else ''} "
                      f"{s['row_kind']} {s['repeat']}: {k}: P_idle "
                      f"{_f(s[f'{k}_p_idle_w'], 4)} W, P_{k} {_f(s[f'{k}_p_run_w'], 4)} W, "
                      f"dP {_f(s[f'{k}_dp_w'], 4)} W, time/image "
                      f"{_f(s[f'{k}_time_per_image_s'], 9)} s, energy/image "
                      f"{_f(s[f'{k}_energy_per_image_mj'], 6)} mJ  ({LABEL})")
        if "accel_e_comp_mj" in s:
            print(f"  [{label} power]{' DRY RUN' if dry else ''} {s['row_kind']} {s['repeat']}: "
                  f"E_sys {_f(s['accel_e_sys_mj'], 6)} mJ, E_comp {_f(s['accel_e_comp_mj'], 6)} mJ "
                  f"(t_PL {_f(s['accel_t_pl_s'] * 1e6 if s['accel_t_pl_s'] == s['accel_t_pl_s'] else math.nan, 3)} µs), "
                  f"duty {_f(s['accel_duty'], 6)}"
                  + (f"; dP_accel-dP_control {_f(s['accel_dp_net_w'], 4)} W, E_sys,net "
                     f"{_f(s['accel_e_sys_net_mj'], 6)} mJ, E_comp,net {_f(s['accel_e_comp_net_mj'], 6)} mJ"
                     if "accel_dp_net_w" in s else ""))
    return {"label": LABEL, "rail": RAIL, "source": ctx.source, "sensor": sensor.describe(),
            "sensor_backend": sensor.backend, "rate_requested_hz": rate_hz,
            "rate_achieved_hz": ov["rate_hz"], "max_gap_s": ov["max_gap_s"],
            "read_errors": logger.read_errors, "phases": phases, "summary": summ,
            "files": {k: str(v) for k, v in names.items()}}


def sample_only(ctx, seconds: float, sensor: Sensor, rate_hz: float = DEFAULT_RATE_HZ,
                prefix: str = PREFIX_B1, tag: str = "_sensorcheck") -> dict:
    """Sensor check: one 'sample_only' window; writes <prefix>_samples/phases<tag>.csv."""
    _check_source(ctx, sensor)
    out = ctx.out_dir
    logger = PowerLogger(sensor, rate_hz)
    with logger:
        s0 = logger.begin("sample_only", f"sensor check {LABEL} {seconds:g}s")
        idle_until(s0["mono"] + seconds)
        w = logger.end("sample_only", s0, "sensor check", repeat=0, kind="idle", images=0)
    ov = logger.overall()
    base = _sensor_cols(sensor, logger, ov)
    st = _phase_stats(logger, w)
    r = ctx.meta("", "", st["duration_s"], 0)
    r.update(base, repeat=0, phase="sample_only", kind="idle", workload="none (sleep)",
             start_utc=w["start"]["utc"], stop_utc=w["stop"]["utc"], start_local=w["start"]["local"],
             stop_local=w["stop"]["local"], start_epoch=f"{w['start']['epoch']:.3f}",
             stop_epoch=f"{w['stop']['epoch']:.3f}", phase_s=f"{st['duration_s']:.3f}", images=0,
             power_mean_w=_f(st["mean_w"]), power_std_w=_f(st["std_w"]), n_samples=st["n"],
             phase_rate_hz=_f(st["rate_hz"], 3), phase_max_gap_s=_f(st["max_gap_s"], 4))
    bc.write_csv(out / f"{prefix}_samples{tag}.csv",
                 _sample_rows(ctx, logger, sensor, "", "", base), SAMPLE_FIELDS, ctx.source)
    bc.write_csv(out / f"{prefix}_phases{tag}.csv", [r], PHASE_FIELDS, ctx.source)
    vals = [s["r"].power_w for s in logger.samples]
    n_changes = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
    print(f"[sensor check] {LABEL}: mean {_f(st['mean_w'], 4)} W std {_f(st['std_w'], 4)} W, "
          f"{st['n']} samples, achieved {st['rate_hz']:.2f} Hz of {rate_hz:g}, max gap "
          f"{st['max_gap_s']:.3f} s, value changes {n_changes} (a sensor updating slower than the "
          f"sample rate repeats values), read errors {logger.read_errors}")
    return {"mean_w": st["mean_w"], "std_w": st["std_w"], "n": st["n"], "rate_hz": st["rate_hz"],
            "max_gap_s": st["max_gap_s"], "value_changes": n_changes}


# ---- CLI ---------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap, nets=False)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--protocol", action="store_true",
                      help="idle/accel/idle/control/idle/cpu x repeats + idle (--no-control: no control)")
    mode.add_argument("--sample-only", action="store_true", help="sensor check, no workload")
    mode.add_argument("--list-sensors", action="store_true", help="probe + print, write nothing")
    ap.add_argument("--sensor", default="auto", choices=("auto",) + SENSOR_ORDER + ("mock",),
                    help="power sensor backend (--backend selects the accelerator: pynq|model)")
    ap.add_argument("--net", default="lenet5", choices=bc.NETS)
    ap.add_argument("--seconds", type=float, default=10.0, help="--sample-only duration")
    ap.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    ap.add_argument("--phase-s", type=float, default=DEFAULT_PHASE_S)
    ap.add_argument("--idle-s", type=float, default=None)
    ap.add_argument("--accel-s", type=float, default=None)
    ap.add_argument("--cpu-s", type=float, default=None)
    ap.add_argument("--control-s", type=float, default=None)
    ap.add_argument("--no-control", action="store_true",
                    help="omit the control phases (host loop without starting the accelerator)")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--order", choices=("fixed", "random"), default="fixed",
                    help="run-phase order per repeat (random: seeded permutation, recorded)")
    ap.add_argument("--order-seed", type=int, default=None, help="with --order random (recorded)")
    ap.add_argument("--no-final-idle", action="store_true")
    ap.add_argument("--cpu-kind", default="cpu_int8_ref",
                    help="cpu/cpu_infer kind, or 'none' to skip the CPU phases")
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--cpu-cores", default="",
                    help="cores for the CPU phases (e.g. 0-3): the ORT thread pool is created on "
                         "them (pool threads inherit the creating thread's affinity) and the "
                         "calling thread moves onto them during CPU phases only; '' = stay pinned")
    ap.add_argument("--prefix", default=PREFIX_B1)
    ap.add_argument("--tag", default=None, help="output tag (default _<net>; --sample-only _sensorcheck)")
    ap.add_argument("--mock-idle-w", type=float, default=None)
    ap.add_argument("--mock-accel-w", type=float, default=None)
    ap.add_argument("--mock-cpu-w", type=float, default=None)
    ap.add_argument("--mock-control-w", type=float, default=None)
    ap.add_argument("--mock-noise-w", type=float, default=0.0)
    ap.add_argument("--mock-seed", type=int, default=0)
    a = ap.parse_args(argv)
    mock_kw = {"levels_w": {k: v for k, v in (("idle", a.mock_idle_w), ("accel", a.mock_accel_w),
                                               ("cpu", a.mock_cpu_w), ("control", a.mock_control_w))
                            if v is not None},
               "noise_w": a.mock_noise_w, "seed": a.mock_seed}

    if a.list_sensors:
        for line in list_sensors():
            print(line)
        return 0

    if a.sample_only:
        try:
            sensor, plog = discover(a.sensor, allow_hw=True, allow_mock=a.sensor == "mock",
                                    mock_kw=mock_kw)
        except SensorUnavailable as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3
        for line in plog:
            print(f"[sensor check] probe: {line}")
        src = bc.SOURCE_HW if sensor.is_hw else bc.SOURCE_DRYRUN
        ctx = SensorOnlyContext(src, a.out_dir, a.board_id, a.allow_dirty)
        print(f"[sensor check] {sensor.describe()} source={src} out={ctx.out_dir}")
        sample_only(ctx, a.seconds, sensor, a.rate_hz, a.prefix, a.tag or "_sensorcheck")
        return 0

    a.nets = [a.net]
    dev, be = bc.open_device(a)
    ctx = bc.RunContext(a, dev, be, "B1-INA260")
    ctx.check_clean()
    ctx.banner()
    if min(x for x in (a.phase_s, a.idle_s, a.accel_s, a.cpu_s, a.control_s) if x is not None) < 30:
        print("NOTE: a phase < 30 s (EXPERIMENTS B1 protocol uses 60 s phases)")
    pkg = ctx.package(a.net)
    cpu_fn, cpu_label = None, ""
    if a.cpu_kind != "none":
        sys.path.insert(0, str(bc.BOARD_DIR))
        try:
            from cpu.cpu_infer import make_runner
        except ImportError as e:
            raise SystemExit(f"CPU phases need cpu/cpu_infer.py (or --cpu-kind none): {e}")
        import board_env as benv
        cores = benv.parse_cores(a.cpu_cores) if a.cpu_cores else None
        home = os.sched_getaffinity(0)
        if cores:
            os.sched_setaffinity(0, cores)       # ORT creates its pool threads here
        try:
            runner = make_runner(a.cpu_kind, a.net, a.data_dir, a.cpu_threads)
        finally:
            os.sched_setaffinity(0, home)
        xs = pkg.x_nchw if a.cpu_kind == "cpu_int8_ref" else pkg.x_f32
        cpu_fn = cpu_workload(runner, xs, phase_cores=cores)
        cpu_label = (f"{a.cpu_kind} x{a.cpu_threads} threads {a.net}"
                     + (f" (cores {a.cpu_cores} during CPU phases; ORT {runner.version})"
                        if hasattr(runner, "version") else ""))
    durations = {k: v for k, v in (("idle", a.idle_s), ("accel", a.accel_s), ("cpu", a.cpu_s),
                                   ("control", a.control_s)) if v is not None}
    try:
        run_power_protocol(ctx, a.net, cpu_fn, sensor=a.sensor, durations=durations,
                           phase_s=a.phase_s, repeats=a.repeats, rate_hz=a.rate_hz,
                           final_idle=not a.no_final_idle, prefix=a.prefix,
                           tag=a.tag if a.tag is not None else f"_{a.net}",
                           cpu_label=cpu_label, sensor_kw=mock_kw, control=not a.no_control,
                           order_seed=(None if a.order == "fixed" else
                                       (a.order_seed if a.order_seed is not None else
                                        __import__("stats").new_seed())))
    except SensorUnavailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
