# copied from v2/board/board_env.py @ 28dd2ad (unchanged; V3 baseline package)
"""Board measurement environment: CPU governor / fixed frequency, core pinning, package-manager
check, die temperature (Zynq MPSoC AMS). Used by run_sessions.py's pre-flight and per step.

All paths are injectable (sysfs_root, proc_root, iio_root, hwmon_root) so the laptop tests and dry
runs use FAKE trees (make_fake_tree); every dry-run line says DRY RUN and nothing on the laptop's
real cpufreq is touched.

  CpuFreqControl   sysfs cpufreq: every policy -> governor 'performance' and scaling_min_freq =
                   scaling_max_freq = the target frequency (default: the highest available);
                   read back scaling_governor and scaling_cur_freq of every policy and online CPU;
                   the original governor / min / max are saved and restored (restore()).
  pin / affinity   os.sched_setaffinity (what `taskset -c` does) on a process; read back with
                   os.sched_getaffinity. parse_cores("0-2,3") -> {0, 1, 2, 3}.
  pkg_manager_busy /proc scan (what `pgrep` does): apt, apt-get, aptitude, dpkg,
                   unattended-upgrades (not the idle unattended-upgrade-shutdown helper),
                   packagekitd, apt.systemd.daily -> list of offenders (pid, comm, cmdline).
  read_die_temp    iio (/sys/bus/iio/devices/iio:device*/ with name ~ 'ams': in_temp*_raw with
                   in_temp*_scale / _offset (or the shared in_temp_scale / in_temp_offset), value
                   = (raw + offset) * scale m°C; or in_temp*_input m°C), else hwmon (name ~ 'ams':
                   temp*_input m°C); else {"source": "unavailable"}. All channels + the max.
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

SYSFS_CPU = "/sys/devices/system/cpu"
PROC = "/proc"
IIO = "/sys/bus/iio/devices"
HWMON = "/sys/class/hwmon"
FREQ_TOL = 0.005                 # relative: scaling_cur_freq vs target

BUSY_COMM = {"apt", "apt-get", "aptitude", "dpkg", "unattended-upgr", "unattended-upgrade",
             "packagekitd", "apt.systemd.dai", "apt.systemd.daily"}
BUSY_EXCLUDE_CMDLINE = ("unattended-upgrade-shutdown",)


def _rd(p) -> str | None:
    try:
        return Path(p).read_text().strip()
    except OSError:
        return None


def _wr(p, v) -> str | None:
    """Write; returns an error string or None."""
    try:
        Path(p).write_text(f"{v}\n")
        return None
    except OSError as e:
        return f"{p}: {e.strerror or e}"


def parse_cores(spec) -> set[int]:
    if spec is None or spec == "":
        return set()
    if isinstance(spec, (set, list, tuple)):
        return {int(c) for c in spec}
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def cores_str(cores) -> str:
    c = sorted(parse_cores(cores))
    if not c:
        return ""
    runs, start, prev = [], c[0], c[0]
    for x in c[1:] + [None]:
        if x is not None and x == prev + 1:
            prev = x
            continue
        runs.append(f"{start}-{prev}" if prev > start else f"{start}")
        if x is not None:
            start = prev = x
    return ",".join(runs)


# ---- cpufreq ---------------------------------------------------------------------------------
@dataclass
class CpuFreqControl:
    root: str = SYSFS_CPU
    fake: bool = False                 # dry run: a fake tree (emulated kernel), label DRY RUN
    saved: dict = field(default_factory=dict)
    applied: bool = False

    def policies(self) -> list[Path]:
        return sorted(Path(p) for p in glob.glob(os.path.join(self.root, "cpufreq", "policy*")))

    def online_cpus(self) -> list[int]:
        out = []
        for d in sorted(glob.glob(os.path.join(self.root, "cpu[0-9]*"))):
            m = re.fullmatch(r"cpu(\d+)", os.path.basename(d))
            if not m:
                continue
            on = _rd(os.path.join(d, "online"))
            if on is None or on == "1":
                out.append(int(m.group(1)))
        return out

    def snapshot(self) -> dict:
        pol = {}
        for p in self.policies():
            pol[p.name] = {k: _rd(p / k) for k in (
                "affected_cpus", "scaling_governor", "scaling_min_freq", "scaling_max_freq",
                "scaling_cur_freq", "cpuinfo_min_freq", "cpuinfo_max_freq",
                "scaling_available_governors", "scaling_available_frequencies", "scaling_driver")}
        cur = {}
        for c in self.online_cpus():
            v = _rd(os.path.join(self.root, f"cpu{c}", "cpufreq", "scaling_cur_freq"))
            if v is not None:
                cur[c] = v
        return {"root": self.root, "fake": self.fake, "policies": pol, "cpu_cur_freq_khz": cur}

    def target_khz(self, want_khz: int | None = None) -> int | None:
        if want_khz:
            return int(want_khz)
        best = None
        for p in self.policies():
            av = _rd(p / "scaling_available_frequencies")
            vals = [int(x) for x in av.split()] if av else []
            mx = _rd(p / "cpuinfo_max_freq")
            if mx:
                vals.append(int(mx))
            if vals:
                best = max(vals) if best is None else min(best, max(vals))
        return best

    def apply(self, want_khz: int | None = None, governor: str = "performance",
              settle=None) -> dict:
        """Set governor + fixed frequency on every policy; returns {ok, errors, target_khz,
        readback}. settle(): called after the writes (the fake tree's kernel emulation)."""
        errors = []
        pols = self.policies()
        if not pols:
            return {"ok": False, "errors": [f"no cpufreq policies under {self.root}/cpufreq "
                                            "(cpufreq driver absent: frequency cannot be fixed "
                                            "or verified)"],
                    "target_khz": None, "readback": self.snapshot()}
        tgt = self.target_khz(want_khz)
        if tgt is None:
            errors.append("target frequency unknown (no scaling_available_frequencies / "
                          "cpuinfo_max_freq)")
        for p in pols:
            if p.name not in self.saved:
                self.saved[p.name] = {k: _rd(p / k) for k in
                                      ("scaling_governor", "scaling_min_freq", "scaling_max_freq")}
            avg = (_rd(p / "scaling_available_governors") or "").split()
            if avg and governor not in avg:
                errors.append(f"{p.name}: governor {governor} not available ({avg})")
                continue
            e = _wr(p / "scaling_governor", governor)
            if e:
                errors.append(e)
            if tgt is None:
                continue
            cur_min = int(_rd(p / "scaling_min_freq") or 0)
            order = (("scaling_max_freq", "scaling_min_freq") if tgt >= cur_min
                     else ("scaling_min_freq", "scaling_max_freq"))
            for k in order:
                e = _wr(p / k, tgt)
                if e:
                    errors.append(e)
        self.applied = True
        if settle is not None:
            settle()
        rb = self.verify(tgt, governor)
        errors += rb["errors"]
        return {"ok": not errors, "errors": errors, "target_khz": tgt, "readback": rb}

    def verify(self, tgt: int | None, governor: str = "performance") -> dict:
        errs, gov, cur = [], {}, {}
        for p in self.policies():
            g = _rd(p / "scaling_governor")
            gov[p.name] = g
            if g != governor:
                errs.append(f"{p.name}: scaling_governor reads {g!r}, not {governor!r}")
            c = _rd(p / "scaling_cur_freq")
            cur[p.name] = c
            if tgt is not None:
                if c is None:
                    errs.append(f"{p.name}: scaling_cur_freq unreadable")
                elif abs(int(c) - tgt) > FREQ_TOL * tgt:
                    errs.append(f"{p.name}: scaling_cur_freq {c} kHz != target {tgt} kHz")
        per_cpu = {}
        for c in self.online_cpus():
            v = _rd(os.path.join(self.root, f"cpu{c}", "cpufreq", "scaling_cur_freq"))
            per_cpu[c] = v
            if tgt is not None and v is not None and abs(int(v) - tgt) > FREQ_TOL * tgt:
                errs.append(f"cpu{c}: scaling_cur_freq {v} kHz != target {tgt} kHz")
        covered = set()
        for p in self.policies():
            covered |= parse_cores((_rd(p / "affected_cpus") or "").replace(" ", ","))
        missing = [c for c in self.online_cpus() if covered and c not in covered]
        if missing:
            errs.append(f"online CPUs {missing} not covered by any cpufreq policy")
        return {"errors": errs, "governor": gov, "policy_cur_freq_khz": cur,
                "cpu_cur_freq_khz": per_cpu}

    def restore(self) -> list[str]:
        """Write back the saved governor / min / max (min/max order chosen so both are valid)."""
        errs = []
        for name, s in self.saved.items():
            p = Path(self.root) / "cpufreq" / name
            if s.get("scaling_governor"):
                e = _wr(p / "scaling_governor", s["scaling_governor"])
                if e:
                    errs.append(e)
            mn, mx = s.get("scaling_min_freq"), s.get("scaling_max_freq")
            if mn and mx:
                cur_max = int(_rd(p / "scaling_max_freq") or 0)
                # raising max first keeps min <= max valid; lowering: min first
                order = ((("scaling_max_freq", mx), ("scaling_min_freq", mn)) if int(mx) >= cur_max
                         else (("scaling_min_freq", mn), ("scaling_max_freq", mx)))
                for k, v in order:
                    e = _wr(p / k, v)
                    if e:
                        errs.append(e)
        self.saved = {}
        self.applied = False
        return errs


# ---- affinity --------------------------------------------------------------------------------
def available_cores() -> set[int]:
    """Every online CPU of the machine (not this process's current affinity)."""
    on = _rd("/sys/devices/system/cpu/online")
    if on:
        try:
            return parse_cores(on)
        except ValueError:
            pass
    return set(range(os.cpu_count() or 1))


def pin(pid: int, cores) -> dict:
    """sched_setaffinity(pid, cores) + read back. {ok, requested, readback, error}."""
    req = parse_cores(cores)
    out = {"requested": cores_str(req), "readback": "", "ok": False, "error": ""}
    try:
        os.sched_setaffinity(pid, req)
        rb = set(os.sched_getaffinity(pid))
        out["readback"] = cores_str(rb)
        out["ok"] = rb == req
        if not out["ok"]:
            out["error"] = f"affinity reads back {cores_str(rb)}, requested {cores_str(req)}"
    except (AttributeError, OSError, ValueError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def affinity_of(pid: int = 0) -> str:
    try:
        return cores_str(os.sched_getaffinity(pid))
    except (AttributeError, OSError):
        return ""


# ---- package managers ------------------------------------------------------------------------
def pkg_manager_busy(proc_root: str = PROC, self_pid: int | None = None) -> list[dict]:
    self_pid = os.getpid() if self_pid is None else self_pid
    out = []
    for d in glob.glob(os.path.join(proc_root, "[0-9]*")):
        pid = int(os.path.basename(d))
        if pid == self_pid:
            continue
        comm = _rd(os.path.join(d, "comm"))
        if comm is None:
            continue
        try:
            raw = Path(d, "cmdline").read_bytes()
        except OSError:
            raw = b""
        cmd = raw.replace(b"\0", b" ").decode(errors="replace").strip()
        base = [os.path.basename(t) for t in cmd.split()[:2]]
        hit = comm in BUSY_COMM or any(b in BUSY_COMM for b in base)
        if hit and not any(x in cmd for x in BUSY_EXCLUDE_CMDLINE):
            out.append({"pid": pid, "comm": comm, "cmdline": cmd[:160]})
    return sorted(out, key=lambda r: r["pid"])


# ---- die temperature -------------------------------------------------------------------------
def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_die_temp(iio_root: str = IIO, hwmon_root: str = HWMON) -> dict:
    for d in sorted(glob.glob(os.path.join(iio_root, "iio:device*"))):
        name = _rd(os.path.join(d, "name")) or ""
        if "ams" not in name.lower():
            continue
        ch = {}
        for f in sorted(glob.glob(os.path.join(d, "in_temp*_input"))):
            v = _num(_rd(f))
            if v is not None:
                pre = os.path.basename(f)[:-len("_input")]
                ch[_rd(os.path.join(d, pre + "_label")) or pre] = v / 1000.0
        for f in sorted(glob.glob(os.path.join(d, "in_temp*_raw"))):
            pre = os.path.basename(f)[:-len("_raw")]
            lab = _rd(os.path.join(d, pre + "_label")) or pre
            if lab in ch:
                continue
            raw = _num(_rd(f))
            scale = _num(_rd(os.path.join(d, pre + "_scale")) or _rd(os.path.join(d, "in_temp_scale")))
            off = _num(_rd(os.path.join(d, pre + "_offset")) or _rd(os.path.join(d, "in_temp_offset"))) or 0.0
            if raw is None or scale is None:
                continue
            ch[lab] = (raw + off) * scale / 1000.0
        if ch:
            return {"source": f"iio {os.path.basename(d)} ({name})", "channels_c":
                    {k: round(v, 3) for k, v in ch.items()}, "max_c": round(max(ch.values()), 3)}
    for d in sorted(glob.glob(os.path.join(hwmon_root, "hwmon*"))):
        name = _rd(os.path.join(d, "name")) or ""
        if "ams" not in name.lower():
            continue
        ch = {}
        for f in sorted(glob.glob(os.path.join(d, "temp*_input"))):
            v = _num(_rd(f))
            if v is not None:
                pre = os.path.basename(f)[:-len("_input")]
                ch[_rd(os.path.join(d, pre + "_label")) or pre] = v / 1000.0
        if ch:
            return {"source": f"hwmon {os.path.basename(d)} ({name})", "channels_c":
                    {k: round(v, 3) for k, v in ch.items()}, "max_c": round(max(ch.values()), 3)}
    return {"source": "unavailable", "channels_c": {}, "max_c": None}


# ---- fake trees (laptop tests / dry runs) ----------------------------------------------------
KV260_FREQS_KHZ = (333333, 444444, 666666, 1333333)    # fake tree only (typical cpufreq-dt OPPs)


def make_fake_tree(root, n_cpus: int = 4, freqs=KV260_FREQS_KHZ, governor: str = "schedutil",
                   temp_raw: int = 41000, busy: list | None = None) -> dict:
    """Create a fake sysfs/proc tree under root: cpufreq (one policy for all CPUs), an AMS iio
    device, a /proc with an idle system (+ `busy` comm names as offending processes). Returns the
    paths {sysfs_root, proc_root, iio_root, hwmon_root}. DRY RUN / tests only."""
    root = Path(root)
    cpu = root / "sys" / "devices" / "system" / "cpu"
    pol = cpu / "cpufreq" / "policy0"
    pol.mkdir(parents=True, exist_ok=True)
    lo = min(freqs)
    vals = {"affected_cpus": " ".join(str(c) for c in range(n_cpus)), "scaling_governor": governor,
            "scaling_available_governors": "conservative ondemand userspace powersave performance schedutil",
            "scaling_available_frequencies": " ".join(str(f) for f in freqs),
            "scaling_min_freq": lo, "scaling_max_freq": max(freqs), "scaling_cur_freq": lo,
            "cpuinfo_min_freq": lo, "cpuinfo_max_freq": max(freqs), "scaling_driver": "cpufreq-dt (FAKE)"}
    for k, v in vals.items():
        (pol / k).write_text(f"{v}\n")
    for c in range(n_cpus):
        d = cpu / f"cpu{c}"
        (d / "cpufreq").mkdir(parents=True, exist_ok=True)
        (d / "online").write_text("1\n")
        (d / "cpufreq" / "scaling_cur_freq").write_text(f"{lo}\n")
    iio = root / "sys" / "bus" / "iio" / "devices" / "iio:device0"
    iio.mkdir(parents=True, exist_ok=True)
    (iio / "name").write_text("xilinx-ams\n")
    # (raw + offset) * scale m°C; AMS-like scale/offset (fake numbers, dry run only)
    for ch, lab, dr in ((7, "ps_temp", 0), (8, "remote_temp", 150), (20, "pl_temp", -200)):
        (iio / f"in_temp{ch}_raw").write_text(f"{temp_raw + dr}\n")
        (iio / f"in_temp{ch}_scale").write_text("7.771514892\n")
        (iio / f"in_temp{ch}_offset").write_text("-36058\n")
        (iio / f"in_temp{ch}_label").write_text(f"{lab}\n")
    hw = root / "sys" / "class" / "hwmon"
    hw.mkdir(parents=True, exist_ok=True)
    proc = root / "proc"
    proc.mkdir(parents=True, exist_ok=True)
    procs = [("1", "systemd", "/sbin/init"),
             ("700", "unattended-upgr", "/usr/bin/python3 /usr/share/unattended-upgrades/"
                                         "unattended-upgrade-shutdown --wait-for-signal")]
    for i, name in enumerate(busy or []):
        procs.append((str(5000 + i), name[:15], f"/usr/bin/{name}"))
    for pid, comm, cmd in procs:
        d = proc / pid
        d.mkdir(exist_ok=True)
        (d / "comm").write_text(comm + "\n")
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode() + b"\0")
    return {"sysfs_root": str(cpu), "proc_root": str(proc),
            "iio_root": str(root / "sys" / "bus" / "iio" / "devices"), "hwmon_root": str(hw)}


def fake_kernel_settle(sysfs_root: str):
    """Emulate cpufreq for a fake tree: cur = clamp(max for 'performance', [min, max])."""
    for p in glob.glob(os.path.join(sysfs_root, "cpufreq", "policy*")):
        p = Path(p)
        mn, mx = int(_rd(p / "scaling_min_freq")), int(_rd(p / "scaling_max_freq"))
        gov = _rd(p / "scaling_governor")
        cur = mx if gov == "performance" else mn
        (p / "scaling_cur_freq").write_text(f"{cur}\n")
        for c in parse_cores((_rd(p / "affected_cpus") or "").replace(" ", ",")):
            f = Path(sysfs_root, f"cpu{c}", "cpufreq", "scaling_cur_freq")
            if f.parent.is_dir():
                f.write_text(f"{cur}\n")


def summary_line(d: dict) -> str:
    return json.dumps(d, sort_keys=True)
