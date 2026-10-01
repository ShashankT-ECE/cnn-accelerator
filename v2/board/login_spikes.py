"""Flag INA260 sample spikes that coincide with an ssh login.

A new ssh login costs the board about 2 s of CPU (sshd, PAM, snapd-desktop-i, tailscale) and shows up
as a +0.1 .. +0.4 W burst on the SOM rail. Logins during a power phase therefore contaminate that
phase's mean. Hygiene rule (v2/board/README.md): one persistent ssh ControlMaster connection, polling
>= 60 s apart, no new login while a power phase runs. This tool is the check.

Inputs
  * `hw_*power_ina260_samples*.csv` (columns repeat, phase, t_epoch, power_w) under --results-dir;
  * the board's sshd login times: lines containing "sshd[...]: Accepted ..." from `journalctl`
    (`--output=short-iso` or the default short format), read from --journal-file or, on the board,
    from `journalctl _COMM=sshd --since .. --until ..` (--since/--until).

A spike = a sample more than --spike-w above its phase median; consecutive spikes (<= 1 s apart) form
one burst. A burst coincides with a login if it starts within [login - 1 s, login + 10 s]. Every login
that falls inside a power phase is also listed, spike or not (the rule is "no login during a phase").
Exit status 0 = nothing flagged, 1 = at least one burst coincides with a login or a login fell inside
a phase.

Run:  python3 login_spikes.py --results-dir results --journal-file logins.txt
      python3 login_spikes.py --results-dir results --since "2026-09-30 18:30" --until "2026-09-30 21:30"
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SPIKE_W = 0.12
BURST_GAP_S = 1.0
LOGIN_BEFORE_S = 1.0
LOGIN_AFTER_S = 10.0

_ISO = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})([+-]\d{2}):?(\d{2})?\s")
_SYSLOG = re.compile(r"^([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{2}:\d{2}:\d{2})\s")
_MONTHS = {m: i for i, m in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)}


def parse_logins(text: str, assume_year: int | None = None) -> list[float]:
    """Epoch seconds of every `sshd[...]: Accepted` line. short-iso lines carry their own UTC
    offset; classic syslog lines ("Sep 30 20:24:01") are taken as UTC in assume_year."""
    out = []
    for line in text.splitlines():
        if "sshd" not in line or "Accepted" not in line:
            continue
        m = _ISO.match(line)
        if m:
            off = f"{m.group(2)}:{m.group(3) or '00'}"
            out.append(datetime.fromisoformat(m.group(1) + off).timestamp())
            continue
        m = _SYSLOG.match(line)
        if m and assume_year:
            h, mi, s = (int(x) for x in m.group(3).split(":"))
            out.append(datetime(assume_year, _MONTHS[m.group(1)], int(m.group(2)), h, mi, s,
                                tzinfo=timezone.utc).timestamp())
    return sorted(out)


def read_phases(results_dir: Path) -> dict:
    """{(file, repeat, phase): [(t_epoch, power_w), ...]} over every INA260 samples CSV."""
    ph = defaultdict(list)
    for f in sorted(Path(results_dir).glob("hw_*power_ina260_samples*.csv")):
        with f.open() as fh:
            for r in csv.DictReader(fh):
                if not r["phase"]:          # inter-phase filler rows are not part of a power phase
                    continue
                ph[(f.name, r["repeat"], r["phase"])].append((float(r["t_epoch"]), float(r["power_w"])))
    return ph


def bursts(samples: list[tuple[float, float]], spike_w: float = SPIKE_W) -> list[dict]:
    med = statistics.median(p for _, p in samples)
    out: list[dict] = []
    for t, p in samples:
        if p - med <= spike_w:
            continue
        if out and t - out[-1]["t_end"] <= BURST_GAP_S:
            out[-1].update(t_end=t, n=out[-1]["n"] + 1, peak_w=max(out[-1]["peak_w"], p))
        else:
            out.append(dict(t_start=t, t_end=t, n=1, peak_w=p, median_w=med))
    return out


def analyse(phases: dict, logins: list[float], spike_w: float = SPIKE_W) -> list[dict]:
    """One finding per (phase, login): kind 'spike' (a burst starts within the login window) or
    'in_phase' (the login falls inside the phase, no burst attributable)."""
    found = []
    for (fname, rep, phase), s in sorted(phases.items()):
        t0, t1 = min(t for t, _ in s), max(t for t, _ in s)
        bs = bursts(s, spike_w)
        for lg in logins:
            hit = [b for b in bs if lg - LOGIN_BEFORE_S <= b["t_start"] <= lg + LOGIN_AFTER_S]
            if hit:
                b = hit[0]
                found.append(dict(file=fname, repeat=rep, phase=phase, login=lg, kind="spike",
                                  burst_start=b["t_start"], n=b["n"], peak_w=b["peak_w"],
                                  median_w=b["median_w"], in_phase=t0 <= lg <= t1))
            elif t0 <= lg <= t1:
                found.append(dict(file=fname, repeat=rep, phase=phase, login=lg, kind="in_phase",
                                  burst_start=None, n=0, peak_w=None, median_w=None, in_phase=True))
    return found


def fmt(f: dict) -> str:
    ts = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    s = f"{f['file']} rep{f['repeat']} {f['phase']}: login {ts(f['login'])} UTC"
    if f["kind"] == "spike":
        s += (f" -> burst at {ts(f['burst_start'])} ({f['n']} samples, peak {f['peak_w']:.2f} W "
              f"vs median {f['median_w']:.3f} W)")
    else:
        s += " fell inside the phase (no burst)"
    return s


_UNIT = re.compile(r"systemd\[1\]: (Starting|Finished|Stopped|Failed to start) (.+?)(?:\.\.\.|\.)?$")


def service_windows(text: str, pattern: str = r"apt|PackageKit|unattended") -> list[tuple[float, float]]:
    """(start, end) epoch pairs of systemd units whose description matches `pattern`, from journal
    lines `Starting X...` ... `Finished X.` (short-iso format). An unterminated start ends at the last
    line of the text. Used for the daily apt upgrade (apt-daily-upgrade.service) that raised the SOM-rail
    power and its noise for 5.6 minutes in the 2026-10-01 B2 run."""
    out, open_, last = [], {}, None
    for line in text.splitlines():
        m = _ISO.match(line)
        if not m:
            continue
        t = datetime.fromisoformat(m.group(1) + f"{m.group(2)}:{m.group(3) or '00'}").timestamp()
        last = t
        u = _UNIT.search(line.rstrip())
        if not u or not re.search(pattern, u.group(2), re.I):
            continue
        if u.group(1) == "Starting":
            open_[u.group(2)] = t
        elif u.group(2) in open_:
            out.append((open_.pop(u.group(2)), t))
    out += [(t0, last) for t0 in open_.values() if last]
    return sorted(out)


def analyse_windows(phases: dict, windows: list[tuple[float, float]]) -> list[dict]:
    """Power phases overlapping a system-activity window (apt upgrade etc.)."""
    found = []
    merged: list[list[float]] = []
    for a, b in sorted(windows):                      # overlapping windows count once
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for (fname, rep, phase), s in sorted(phases.items()):
        t0, t1 = min(t for t, _ in s), max(t for t, _ in s)
        for a, b in merged:
            if a <= t1 and b >= t0:
                found.append(dict(file=fname, repeat=rep, phase=phase, win=(a, b), overlap_s=min(b, t1) - max(a, t0)))
    return found


def fmt_window(f: dict) -> str:
    ts = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%H:%M:%S")
    return (f"{f['file']} rep{f['repeat']} {f['phase']}: overlaps system activity {ts(f['win'][0])}-"
            f"{ts(f['win'][1])} UTC by {f['overlap_s']:.0f} s")


def fetch_units(since: str, until: str | None = None) -> str:
    cmd = ["journalctl", "-u", "apt-daily.service", "-u", "apt-daily-upgrade.service", "-u", "packagekit.service",
           "-u", "unattended-upgrades.service", "--since", since, "--output=short-iso", "--no-pager"]
    if until:
        cmd += ["--until", until]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def fetch_journal(since: str, until: str | None) -> str:
    cmd = ["journalctl", "_COMM=sshd", "--since", since, "--output=short-iso", "--no-pager"]
    if until:
        cmd += ["--until", until]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--journal-file")
    src.add_argument("--since", help="journalctl --since (on the board)")
    ap.add_argument("--until", default=None)
    ap.add_argument("--assume-year", type=int, default=None, help="for classic syslog lines")
    ap.add_argument("--spike-w", type=float, default=SPIKE_W)
    a = ap.parse_args(argv)
    text = Path(a.journal_file).read_text() if a.journal_file else fetch_journal(a.since, a.until)
    logins = parse_logins(text, a.assume_year)
    found = analyse(read_phases(Path(a.results_dir)), logins, a.spike_w)
    print(f"login_spikes: {len(logins)} ssh login(s), {len(found)} finding(s)")
    for f in found:
        print("  FLAG " + fmt(f))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
