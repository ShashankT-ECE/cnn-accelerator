#!/usr/bin/env python3
"""V2 board sessions orchestrator: one command per session, resumable, time-budgeted, pre-flighted.

    sudo -E ./session.sh 1|2|3|all [--budget-min N] [--resume|--fresh] [--quick] [--plan]
    ./session.sh all --backend model --allow-dirty          # laptop dry run -> v2/results/dryrun/

Sessions (v2/board/README.md):
  1  bring-up       test_shell (--skip-scratch), test_core_smoke (+ informational --write-mode slice)
  2  A1-A4, B3, CPU A1 accuracy, A2/A3 cycles, A4 utilization, B3 breakdown, CPU baselines (A5)
  3  B1, B2         B1 INA260 SOM-rail power protocol per net (power_log.py: idle/accel/idle/cpu x3
                    + final idle), B2 clock sweep (cycles == model + INA260 idle/accel/idle per clock);
                    --with-meter adds the optional external-meter cross-check windows (exp_b1_power.py)

Every invocation first runs the PRE-FLIGHT (fails loudly, exit 3): DEPLOY_INFO present; .bit/.hwh
SHA256 vs DEPLOY_INFO and the shipped .bit.sha256; summary.json build id / closed clock / timing
met; VERSION == 0x474F5302; BUILD_ID register == DEPLOY_INFO build_id; pl_clk0 read back <= the
closed clock (+tol) and, for sessions 2/3, equal to it (within tol); every data-package SHA256
(MANIFEST.json) and the PACKAGE.json chain; free disk; clean-tree flags (--allow-dirty marks
rows git_dirty=True, invalid for the paper). With --backend model everything is the ModelBackend
equivalent and every line says DRY RUN.

State: <results>/session_state.json records the provenance (backend, BUILD_ID, bit SHA256, closed
and read-back clock, data package, scripts commit) and every step (status, command, duration,
output files + SHA256, log). On rerun (default --resume) a step is skipped only if it finished OK
with the same parameters and all its outputs (and log) still verify; otherwise it is rerun FROM
SCRATCH (no per-image-chunk resume: one CSV = one uninterrupted run = one bitstream + clock).
A provenance change (other bitstream / BUILD_ID / clock / data / scripts commit) refuses to resume
(exit 4): start over with --fresh. --fresh and replaced outputs are ARCHIVED into
<results>/archive/<timestamp>_<why>/, never deleted.

Steps write into <results>/.staging/<step>/ and are promoted into <results>/ only when the step
process exits (interrupted / timed-out steps never leave partial files among the results; their
staging is archived). Budget: --budget-min; each step's duration is estimated (previous recorded
duration of the same step and parameters, else a per-image rate learned in this state, else the
default table below, labelled "assumed") and a step that would not fit in the remaining budget is
DEFERRED (reported, recorded; the next run picks it up). Every step has a hard timeout
(--step-timeout-min, default max(5 min, 3 x estimate + 2 min), capped by the remaining budget):
SIGINT (scripts restore pl_clk0 in their finally blocks), then SIGTERM, then SIGKILL.

Exit: 0 all requested steps OK; 1 a step failed; 2 usage; 3 pre-flight failed; 4 provenance
changed / bring-up missing / lock held; 5 incomplete (deferred / not run); 130 interrupted.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import board_common as bc

STATE_NAME = "session_state.json"
STATE_VERSION = 1
VERSION_CORE = 0x474F5302
OFF_VERSION, OFF_BUILD_ID = 0x0F8, 0x0FC        # FORMATS.md CSR map (= gos_driver)

# Default per-inference wall time (s/image) and fixed costs used before anything was measured in
# this state. pynq values are ASSUMPTIONS (Python MMIO bound; README), replaced by the recorded
# durations after the first run; model values were measured by the laptop dry run.
DEFAULT_RATE_S = {"pynq": 0.02, "model": 0.006}
DEFAULT_FIXED_S = {  # kind -> backend -> seconds
    "bringup": {"pynq": 120.0, "model": 30.0},
    "cpu": {"pynq": 2400.0, "model": 600.0},
    "cpu_quick": {"pynq": 600.0, "model": 120.0},
}
STEP_OVERHEAD_S = 30.0          # overlay load, package verify, net loads
KILL_GRACE_S = (20.0, 5.0)      # SIGINT -> SIGTERM, SIGTERM -> SIGKILL
RESULT_PATTERNS = ("hw_*.csv", "hw_*.npz", "hw_*.json")


class PreflightError(RuntimeError):
    pass


class Interrupted(BaseException):
    """SIGINT / SIGTERM / SIGHUP received by the orchestrator."""


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---- steps ------------------------------------------------------------------------------------
@dataclass
class Step:
    id: str                         # "s2.A1"
    session: int
    title: str
    argv: list                      # script (relative to the board dir) + args
    kind: str = "infer"             # infer | fixed | bringup | cpu | cpu_quick
    images: int = 0                 # accelerator inferences (estimate)
    fixed_s: float = 0.0            # fixed part (windows, gaps)
    out_flag: str | None = "--out-dir"
    expect: tuple = ()              # outputs that must exist after a successful run
    required: bool = True

    @property
    def params_key(self) -> str:
        return json.dumps(self.argv)


@dataclass
class Config:
    backend: str                    # pynq | model
    board_dir: Path
    results_dir: Path
    data_dir: Path
    deploy: dict
    bit: Path
    allow_dirty: bool = False
    clock_tol: float = bc.CLOCK_TOL_MHZ
    min_free_mb: float = 1000.0
    require_clock_equal: bool = True
    python: str = sys.executable
    extra: dict = field(default_factory=dict)

    @property
    def dry(self) -> bool:
        return self.backend == "model"

    @property
    def closed_mhz(self):
        v = self.deploy.get("bit_clock_mhz")
        return None if v is None else float(v)


def n_images(data_dir: Path, net: str) -> int:
    try:
        return int(json.loads((Path(data_dir) / net / "MANIFEST.json").read_text())["n_images"])
    except (OSError, KeyError, ValueError):
        return 10000


def build_steps(cfg: Config, sessions, quick: bool = False, window_s: float = 60.0,
                gap_s: float = 10.0, write_mode: str = "elem", b2_images: int = 100,
                board_id: str | None = None, power_repeats: int = 3, b2_power_repeats: int = 3,
                power_rate_hz: float = 10.0, with_meter: bool = False) -> list[Step]:
    be, dd = cfg.backend, str(cfg.data_dir)
    closed = cfg.closed_mhz
    maxf = f"{closed + cfg.clock_tol:.6f}"
    dev = ["--backend", be, "--data-dir", dd]
    if cfg.dry:
        dev += ["--clock-mhz", f"{closed:.6f}"]
    else:
        dev += ["--bit", str(cfg.bit)]
    if board_id:
        dev += ["--board-id", board_id]
    dirty = ["--allow-dirty"] if cfg.allow_dirty else []
    wm = ["--write-mode", write_mode]
    nets = bc.NETS
    n_all = sum(n_images(cfg.data_dir, n) for n in nets)
    lim = 200 if quick else None
    n_lim = sum(min(lim, n_images(cfg.data_dir, n)) for n in nets) if lim else n_all
    steps: list[Step] = []
    if 1 in sessions:
        shell = ["test_shell.py", "--bit", str(cfg.bit), "--expect-version", f"0x{VERSION_CORE:08X}",
                 "--skip-scratch", "--max-fclk0", maxf]
        if cfg.dry:
            shell += ["--backend", "model", "--clock-mhz", f"{closed:.6f}"]
        steps += [
            Step("s1.shell", 1, "shell/memory smoke (test_shell --skip-scratch)", shell,
                 kind="bringup", out_flag=None),
            Step("s1.smoke", 1, "core smoke (VERSION/BUILD_ID, LeNet+CIFAR bit/cycle-exact, refused job)",
                 ["test_core_smoke.py", *dev, "--max-fclk0", maxf], kind="bringup", out_flag=None),
            Step("s1.smoke_slice", 1, "core smoke with --write-mode slice (informational)",
                 ["test_core_smoke.py", *dev, "--max-fclk0", maxf, "--write-mode", "slice"],
                 kind="bringup", out_flag=None, required=False),
        ]
    if 2 in sessions:
        lim_a = ["--limit", str(lim)] if lim else []
        a4_lim = lim or 1000
        b3_n = 200 if quick else 1000
        steps += [
            Step("s2.A1", 2, "A1 accuracy", ["exp_a1_accuracy.py", *dev, *wm, *dirty, *lim_a],
                 images=n_lim, expect=("hw_a1_accuracy.csv",)),
            Step("s2.A2A3", 2, "A2/A3 cycles", ["exp_a2_a3_cycles.py", *dev, *wm, *dirty, *lim_a],
                 images=n_lim, expect=("hw_a2_a3_cycles.csv",)),
            Step("s2.A4", 2, "A4 utilization", ["exp_a4_util.py", *dev, *wm, *dirty,
                                                "--limit", str(a4_lim)],
                 images=sum(min(a4_lim, n_images(cfg.data_dir, n)) for n in nets),
                 expect=("hw_a4_util.csv",)),
            Step("s2.B3", 2, "B3 breakdown", ["exp_b3_breakdown.py", *dev, *wm, *dirty,
                                              "--n", str(b3_n)],
                 images=len(nets) * (b3_n + 50), expect=("hw_b3_breakdown.csv",)),
        ]
        cpu = ["cpu/run_cpu_baselines.py", "--data-dir", dd, "--tag",
               "laptop" if cfg.dry else "board", *(["--quick"] if quick else []), *dirty]
        if (cfg.board_dir / "cpu" / "run_cpu_baselines.py").is_file():
            steps.append(Step("s2.CPU", 2, "CPU baselines (A5)", cpu,
                              kind="cpu_quick" if quick else "cpu", expect=("hw_cpu_baseline.csv",)))
    if 3 in sessions:
        w = ["--window-s", f"{window_s:g}", "--gap-s", f"{gap_s:g}"]
        clocks = bc.b2_sweep_clocks(closed, cfg.clock_tol)
        # B1 primary: INA260 SOM-rail protocol, one step per net
        steps += power_hook_steps(cfg, dev, dirty, window_s, power_repeats, power_rate_hz, wm)
        if with_meter:      # optional cross-check with the external inline 12 V meter
            steps += [
                Step("s3.B1meter", 3, f"B1 cross-check windows idle/fpga/cpu (LeNet-5): {METER_LABEL}",
                     ["exp_b1_power.py", *dev, *wm, *dirty, "--modes", "idle", "fpga", "cpu",
                      "--net", "lenet5", *w], kind="fixed", fixed_s=3 * window_s + 2 * gap_s + 15,
                     expect=("hw_b1_meter_windows.csv",)),
                Step("s3.B1meter_cpu4", 3, f"B1 cross-check window cpu int8 x4 threads (CIFAR-10): "
                     f"{METER_LABEL}",
                     ["exp_b1_power.py", *dev, *wm, *dirty, "--modes", "cpu", "--cpu-kind",
                      "cpu_int8_ref", "--cpu-threads", "4", "--net", "cifar10", *w,
                      "--csv-suffix", "cpu4_cifar10"], kind="fixed", fixed_s=window_s + 15,
                     expect=("hw_b1_meter_windows_cpu4_cifar10.csv",)),
            ]
        b2_net = "lenet5"
        b2_power = [f"hw_b2_power_ina260_{k}_{b2_net}_{clock_tag(c)}.csv"
                    for c in clocks for k in ("samples", "phases", "summary")]
        per_clock = 3 * b2_power_repeats * window_s + B2_PER_CLOCK_OVERHEAD_S
        if with_meter:
            per_clock += window_s + gap_s
        steps.append(
            Step("s3.B2", 3, f"B2 clock sweep {clocks} MHz (closed {closed}): cycles == model + "
                 f"{POWER_LABEL} idle/accel/idle x{b2_power_repeats} per clock",
                 ["exp_b2_clock.py", *dev, *wm, *dirty, "--net", b2_net, "--clocks",
                  *[f"{c:.6f}" for c in clocks], "--max-mhz", f"{closed:.6f}",
                  "--images", str(b2_images), *w, "--power-repeats", str(b2_power_repeats),
                  "--rate-hz", f"{power_rate_hz:g}",
                  *(["--sensor", "mock"] if cfg.dry else []),
                  *(["--with-meter"] if with_meter else [])], kind="infer",
                 images=len(clocks) * b2_images, fixed_s=len(clocks) * per_clock,
                 expect=("hw_b2_clock.csv", *b2_power)))
    return steps


# ==== POWER HOOK (Session 3) ====================================================================
# B1 PRIMARY = the INA260 SOM-rail power protocol (power_log.py), one step per net:
#   python3 power_log.py --backend {pynq,model} --data-dir D (--bit B | --clock-mhz C) [--allow-dirty]
#       --write-mode M --protocol --net <net> --tag _<net> --phase-s W --repeats R --rate-hz H
#       --cpu-kind cpu_int8_ref --cpu-threads 1 [dry run: --sensor mock] --out-dir <staging dir>
# Phases per repeat idle/accel/idle/cpu (W s each) x R + one final idle -> 4R+1 phases.
# CPU phases = the INT8 numpy reference (cpu_int8_ref), 1 thread: the same INT8 arithmetic as the
# accelerator and A5's single-thread configuration, so dP_cpu is a like-for-like single-core
# software baseline on the same rail (4-thread CPU power only via the optional meter step).
# Rows are labelled "SOM-rail power (INA260)"; model backend -> mock sensor, dryrun_model rows.
POWER_HOOK_SCRIPT = "power_log.py"
POWER_LABEL = "SOM-rail power (INA260)"          # = power_log.LABEL
METER_LABEL = "board input power (external meter, cross-check)"   # = exp_b1_power.METER_LABEL
B1_CPU_KIND, B1_CPU_THREADS = "cpu_int8_ref", 1
B1_POWER_OVERHEAD_S = 60.0     # package verify + load_net + CPU runner build + warm-ups
B2_PER_CLOCK_OVERHEAD_S = 20.0  # set clock, soft_reset, reload + readback, sensor probe, CSVs


def clock_tag(mhz: float) -> str:
    """B2 per-clock file tag (= power_log.clock_tag): 199.998001 -> '200mhz'."""
    return f"{int(round(float(mhz)))}mhz"


def power_phases(repeats: int) -> int:
    return 4 * repeats + 1


def power_hook_steps(cfg: Config, dev: list, dirty: list, window_s: float, repeats: int = 3,
                     rate_hz: float = 10.0, wm: list | None = None) -> list[Step]:
    extra = ["--sensor", "mock"] if cfg.dry else []
    out = []
    for net in bc.NETS:
        tag = f"_{net}"
        out.append(Step(
            f"s3.B1.{net}", 3, f"B1 {POWER_LABEL} protocol {net} (idle/accel/idle/cpu x{repeats} "
            f"+ idle, {window_s:g} s phases, CPU {B1_CPU_KIND} x{B1_CPU_THREADS})",
            [POWER_HOOK_SCRIPT, *dev, *(wm or []), *dirty, "--protocol", "--net", net, "--tag", tag,
             "--phase-s", f"{window_s:g}", "--repeats", str(repeats), "--rate-hz", f"{rate_hz:g}",
             "--cpu-kind", B1_CPU_KIND, "--cpu-threads", str(B1_CPU_THREADS), *extra],
            kind="fixed", fixed_s=power_phases(repeats) * window_s + B1_POWER_OVERHEAD_S,
            expect=tuple(f"hw_b1_power_ina260_{k}{tag}.csv" for k in ("samples", "phases", "summary"))))
    return out
# ==== end POWER HOOK ============================================================================


# ---- state -----------------------------------------------------------------------------------
def load_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    s = json.loads(path.read_text())
    if s.get("version") != STATE_VERSION:
        raise SystemExit(f"{path}: state version {s.get('version')} != {STATE_VERSION}; use --fresh")
    return s


def save_state(path: Path, state: dict):
    state["updated_utc"] = utc_now()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=False) + "\n")
    os.replace(tmp, path)


PROV_KEYS = ("backend", "build_id_hw", "bit_sha256", "closed_clock_mhz", "data_package_sha256",
             "scripts_commit", "scripts_dirty")


def provenance_diff(old: dict, new: dict, tol: float) -> list[str]:
    diffs = [f"{k}: state {old.get(k)!r} != now {new.get(k)!r}"
             for k in PROV_KEYS if old.get(k) != new.get(k)]
    a, b = old.get("clock_readback_mhz"), new.get("clock_readback_mhz")
    if a is None or b is None or abs(float(a) - float(b)) > tol:
        diffs.append(f"clock_readback_mhz: state {a} != now {b}")
    return diffs


class Archiver:
    """Moves files into <results>/archive/<ts>_<label>/ keeping their relative path. Never deletes."""

    def __init__(self, results_dir: Path, label: str):
        self.root = results_dir / "archive" / f"{stamp()}_{label}"
        self.results_dir = results_dir
        self.moved: list[str] = []

    def move(self, p: Path, rel: Path | None = None):
        if not p.exists():
            return
        rel = rel or p.relative_to(self.results_dir)
        dst = self.root / rel
        k = 1
        while dst.exists():
            dst = dst.with_name(f"{dst.name}.{k}")
            k += 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst))
        self.moved.append(str(rel))


def archive_everything(results_dir: Path, label: str, keep=()) -> Archiver:
    """--fresh / first state: state file, hw_* results, logs/*, .staging/ -> archive
    (except the paths in keep, e.g. the session log of this invocation)."""
    ar = Archiver(results_dir, label)
    keep = {Path(k).resolve() for k in keep}
    ar.move(results_dir / STATE_NAME)
    for pat in RESULT_PATTERNS:
        for p in sorted(results_dir.glob(pat)):
            ar.move(p)
    for p in sorted((results_dir / "logs").rglob("*")) if (results_dir / "logs").is_dir() else []:
        if p.is_file() and p.resolve() not in keep:
            ar.move(p)
    ar.move(results_dir / ".staging")
    return ar


def new_state(prov: dict, timing: dict | None = None) -> dict:
    return {"version": STATE_VERSION, "created_utc": utc_now(), "updated_utc": "",
            "provenance": prov, "steps": {}, "invocations": [], "timing": timing or {}}


def verify_step(step: Step, rec: dict | None, results_dir: Path) -> tuple[bool, str]:
    """Completed step whose outputs still verify?"""
    if not rec:
        return False, "never run"
    if rec.get("status") != "ok":
        return False, f"last status {rec.get('status')}"
    if rec.get("params_key") != step.params_key:
        return False, "parameters changed"
    files = dict(rec.get("outputs", {}))
    if rec.get("log"):
        files[rec["log"]] = rec.get("log_sha256", "")
    for name, sha in files.items():
        p = results_dir / name
        if not p.is_file():
            return False, f"output {name} missing"
        if bc.sha256_file(p) != sha:
            return False, f"output {name} changed (SHA256)"
    return True, f"verified {len(files)} files"


# ---- estimates -------------------------------------------------------------------------------
def learned_rate(state: dict, backend: str):
    rates = [t["rate_s"] for t in state.get("timing", {}).values()
             if t.get("backend") == backend and t.get("rate_s")]
    return max(rates) if rates else None


def estimate(step: Step, state: dict, backend: str) -> tuple[float, str]:
    t = state.get("timing", {}).get(step.id)
    if t and t.get("params_key") == step.params_key and t.get("backend") == backend:
        return 1.1 * float(t["duration_s"]), "previous run"
    if step.kind in DEFAULT_FIXED_S:
        if t and t.get("backend") == backend:
            return 1.1 * float(t["duration_s"]), "previous run (other params)"
        return DEFAULT_FIXED_S[step.kind][backend], "assumed default"
    rate, src = learned_rate(state, backend), "learned rate"
    if rate is None:
        rate, src = DEFAULT_RATE_S[backend], "assumed default rate"
    if not step.images:
        return step.fixed_s + STEP_OVERHEAD_S, "fixed windows + overhead"
    return step.images * rate + step.fixed_s + STEP_OVERHEAD_S, f"{src} {rate * 1e3:.2f} ms/img"


# ---- pre-flight ------------------------------------------------------------------------------
def open_backend_default(cfg: Config):
    import gos_driver as D
    if cfg.dry:
        bid = int(str(cfg.deploy.get("build_id") or "0"), 16)
        return D.ModelBackend(clock_mhz=cfg.closed_mhz or 200.0, build_id=bid)
    return D.PynqBackend(cfg.bit, download=True)


def read_summary(bit: Path) -> dict | None:
    p = bit.parent / "summary.json"
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except (OSError, ValueError):
        return None


def preflight(cfg: Config, open_backend=open_backend_default, verify_data=bc.verify_manifest,
              say=print) -> dict:
    """All checks; prints each; raises PreflightError listing every failure. Returns provenance."""
    tag = "DRY RUN " if cfg.dry else ""
    fails, warns = [], []

    def ok(msg):
        say(f"  [{tag}preflight] OK    {msg}")

    def bad(msg):
        say(f"  [{tag}preflight] FAIL  {msg}")
        fails.append(msg)

    def warn(msg):
        say(f"  [{tag}preflight] WARN  {msg}")
        warns.append(msg)

    info = cfg.deploy
    say(f"[{tag}preflight] backend={cfg.backend} deploy info: {info.get('origin')}")
    if cfg.dry:
        say(f"  [{tag}preflight] DRY RUN: ModelBackend equivalents; VERSION/BUILD_ID/pl_clk0 are the "
            "simulated register map / nominal clock (set from the deploy info), NOT hardware.")
    elif info.get("origin") != "DEPLOY_INFO.json":
        bad("DEPLOY_INFO.json missing next to the scripts: deploy with deploy.sh (hardware rows need it)")

    # -- bitstream files ----------------------------------------------------------------
    bit_sha = hwh_sha = ""
    bit = cfg.bit
    hwh = bit.with_suffix(".hwh")
    if not bit.is_file():
        bad(f"bitstream {bit} missing")
    else:
        bit_sha = bc.sha256_file(bit)
        want = info.get("bit_sha256")
        if want:
            (ok if want == bit_sha else bad)(f"{bit.name} SHA256 {bit_sha[:16]} == DEPLOY_INFO "
                                             f"bit_sha256 {str(want)[:16]}")
        elif cfg.dry:
            warn("no bit_sha256 in the (laptop) deploy info; checked against .bit.sha256 only")
        else:
            bad("DEPLOY_INFO has no bit_sha256")
        side = Path(str(bit) + ".sha256")
        if side.is_file():
            s = side.read_text().split()[0] if side.read_text().split() else ""
            (ok if s == bit_sha else bad)(f"{bit.name} SHA256 == shipped {side.name} {s[:16]}")
        else:
            warn(f"{side.name} not shipped; SHA256 checked against DEPLOY_INFO only")
    if not hwh.is_file():
        bad(f"{hwh} missing (the .hwh must sit beside the .bit)")
    else:
        hwh_sha = bc.sha256_file(hwh)
        if info.get("hwh_sha256"):
            (ok if info["hwh_sha256"] == hwh_sha else bad)(
                f"{hwh.name} SHA256 {hwh_sha[:16]} == DEPLOY_INFO hwh_sha256")

    # -- closed clock / timing -------------------------------------------------------------
    closed = cfg.closed_mhz
    if closed is None:
        bad("closed pl_clk0 unknown (DEPLOY_INFO bit_clock_mhz / summary.json pl_clk0_mhz_actual)")
    summ = read_summary(bit)
    if summ is not None:
        if info.get("build_id") and str(summ.get("build_id", "")).lower() != str(info["build_id"]).lower():
            bad(f"summary.json build_id {summ.get('build_id')} != DEPLOY_INFO build_id {info['build_id']}")
        sc = summ.get("pl_clk0_mhz_actual")
        if closed is not None and sc is not None and abs(float(sc) - closed) > 1e-3:
            bad(f"summary.json closed clock {sc} != DEPLOY_INFO bit_clock_mhz {closed}")
        wns, fs, fh = summ.get("wns_ns"), summ.get("failing_setup_endpoints"), summ.get("failing_hold_endpoints")
        met = wns is not None and float(wns) >= 0 and not fs and not fh and float(summ.get("whs_ns", 0)) >= 0
        (ok if met else bad)(f"bitstream timing met at {sc} MHz (WNS {wns} ns, failing setup {fs}, hold {fh})")
    else:
        wns = info.get("timing_wns_ns")
        if wns is None:
            warn("no summary.json beside the bitstream and no DEPLOY_INFO timing_wns_ns: timing "
                 "closure not re-checked")
        else:
            (ok if float(wns) >= 0 else bad)(f"DEPLOY_INFO timing_wns_ns {wns} >= 0")

    # -- device: VERSION, BUILD_ID, clock read-back -------------------------------------------
    version = build_id = clk = None
    be_bit_sha = ""
    try:
        be = open_backend(cfg)
        version = be.csr.read(OFF_VERSION)
        build_id = be.csr.read(OFF_BUILD_ID)
        clk = float(be.fclk0_mhz())
        be_bit_sha = getattr(be, "bit_sha256", "")
    except Exception as e:  # noqa: BLE001 - reported as a pre-flight failure
        bad(f"could not open the {cfg.backend} backend / overlay: {type(e).__name__}: {e}")
    if version is not None:
        (ok if version == VERSION_CORE else bad)(
            f"VERSION 0x{version:08X} == 0x{VERSION_CORE:08X}"
            + ("" if version == VERSION_CORE else " (0x474F5300 = empty shell bitstream?)"))
        want = str(info.get("build_id") or "").lower()
        if not want:
            bad("DEPLOY_INFO build_id unknown: cannot check the BUILD_ID register")
        else:
            (ok if f"{build_id:08x}" == want else bad)(
                f"BUILD_ID register {build_id:08x} == DEPLOY_INFO build_id {want}")
        if be_bit_sha and bit_sha and be_bit_sha != bit_sha:
            bad(f"loaded overlay SHA256 {be_bit_sha[:16]} != {bit.name} {bit_sha[:16]}")
    if clk is not None and closed is not None:
        src = "nominal (dry run)" if cfg.dry else "read back (pynq Clocks.fclk0_mhz)"
        (ok if clk <= closed + cfg.clock_tol else bad)(
            f"pl_clk0 {clk:.6f} MHz {src} <= closed clock {closed:.6f} MHz (+{cfg.clock_tol})"
            + ("" if clk <= closed + cfg.clock_tol else " — ABOVE the timing-closed clock"))
        if cfg.require_clock_equal:
            (ok if abs(clk - closed) <= cfg.clock_tol else bad)(
                f"pl_clk0 {clk:.6f} MHz == closed clock {closed:.6f} MHz (±{cfg.clock_tol}) for "
                "the main runs")

    # -- data package ------------------------------------------------------------------------
    pkg_path = cfg.data_dir / "PACKAGE.json"
    pkg_sha, man_sha, dirty_data = "", {}, []
    if not pkg_path.is_file():
        bad(f"{pkg_path} missing (make_board_data.py + deploy)")
    else:
        pkg_sha = bc.sha256_file(pkg_path)
        pkg = json.loads(pkg_path.read_text())
        want = info.get("data_package_sha256")
        if want:
            (ok if want == pkg_sha else bad)(f"PACKAGE.json SHA256 {pkg_sha[:16]} == DEPLOY_INFO "
                                             f"data_package_sha256 {want[:16]}")
        elif not cfg.dry:
            bad("DEPLOY_INFO has no data_package_sha256")
        if pkg.get("git_dirty", True):
            dirty_data.append("PACKAGE.json")
        for net in bc.NETS:
            try:
                man = verify_data(cfg.data_dir / net)
                ms = bc.sha256_file(cfg.data_dir / net / "MANIFEST.json")
                man_sha[net] = ms
                exp = pkg.get("nets", {}).get(net, {}).get("manifest_sha256")
                (ok if exp == ms else bad)(f"{net}: every file matches MANIFEST.json; MANIFEST "
                                           f"SHA256 {ms[:16]} == PACKAGE.json")
                if man.get("git_dirty", True):
                    dirty_data.append(net)
            except (bc.ManifestError, OSError, ValueError) as e:
                bad(f"{net}: data package: {e}")

    # -- disk --------------------------------------------------------------------------------
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    free_mb = shutil.disk_usage(cfg.results_dir).free / 2**20
    (ok if free_mb >= cfg.min_free_mb else bad)(
        f"free disk at {cfg.results_dir}: {free_mb:.0f} MB >= {cfg.min_free_mb:.0f} MB")

    # -- clean tree --------------------------------------------------------------------------
    sdirty = bool(info.get("dirty", True))
    ddirty = bool(dirty_data) or bool(info.get("data_git_dirty", False))
    if sdirty or ddirty:
        msg = (f"dirty tree: scripts dirty={sdirty} (commit {str(info.get('commit', ''))[:8]}), "
               f"data dirty={dirty_data or info.get('data_git_dirty')}")
        if cfg.allow_dirty:
            warn(msg + " — --allow-dirty: rows are git_dirty=True, INVALID for the paper")
        else:
            bad(msg + " — commit, rebuild data, redeploy (or --allow-dirty, rows invalid for the paper)")
    else:
        ok(f"clean tree: scripts {str(info.get('commit', ''))[:8]}, data package clean")

    prov = {"backend": cfg.backend, "source": bc.SOURCE_DRYRUN if cfg.dry else bc.SOURCE_HW,
            "version_hw": None if version is None else f"0x{version:08X}",
            "build_id_hw": None if build_id is None else f"{build_id:08x}",
            "build_id_deploy": info.get("build_id"), "bit": str(bit), "bit_sha256": bit_sha,
            "hwh_sha256": hwh_sha, "closed_clock_mhz": closed, "clock_readback_mhz": clk,
            "data_package_sha256": pkg_sha, "manifest_sha256": man_sha,
            "scripts_commit": info.get("commit", ""), "scripts_dirty": sdirty,
            "data_dirty": ddirty, "allow_dirty": cfg.allow_dirty,
            "deploy_origin": info.get("origin"), "preflight_utc": utc_now(), "warnings": warns}
    if fails:
        raise PreflightError(f"{tag}PRE-FLIGHT FAILED ({len(fails)}):\n  - " + "\n  - ".join(fails))
    say(f"[{tag}preflight] PASSED ({len(warns)} warnings)")
    return prov


# ---- step execution ---------------------------------------------------------------------------
def _kill_group(p: subprocess.Popen, say):
    for sig, grace in ((signal.SIGINT, KILL_GRACE_S[0]), (signal.SIGTERM, KILL_GRACE_S[1]),
                       (signal.SIGKILL, 5.0)):
        if p.poll() is not None:
            return
        say(f"  sending {sig.name} to step process group {p.pid}")
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        try:
            p.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def run_step(step: Step, cfg: Config, state: dict, state_path: Path, timeout_s: float, say,
             lock_fd: int | None = None) -> dict:
    rd = cfg.results_dir
    staging = rd / ".staging" / step.id
    prev = state["steps"].get(step.id)
    ar = Archiver(rd, f"{step.id}_replaced")
    if staging.exists():
        ar.move(staging)
    if prev:
        for name in list(prev.get("outputs", {})) + ([prev["log"]] if prev.get("log") else []):
            ar.move(rd / name)
    log_rel = f"logs/{step.id}.log"
    ar.move(rd / log_rel)
    if ar.moved:
        say(f"  archived previous/partial files of {step.id} -> {ar.root.relative_to(rd)}: {ar.moved}")
    staging.mkdir(parents=True)
    (rd / "logs").mkdir(exist_ok=True)
    cmd = [cfg.python, str(cfg.board_dir / step.argv[0]), *step.argv[1:]]
    if step.out_flag:
        cmd += [step.out_flag, str(staging)]
    rec = {"status": "running", "title": step.title, "session": step.session,
           "params_key": step.params_key, "cmd": cmd, "started_utc": utc_now(),
           "timeout_s": round(timeout_s, 1), "provenance": state["provenance"], "outputs": {},
           "log": log_rel, "required": step.required}
    state["steps"][step.id] = rec
    save_state(state_path, state)
    say(f"  $ {' '.join(cmd)}")
    t0 = time.monotonic()
    status, rc = "failed", None
    with (rd / log_rel).open("w") as log:
        log.write(f"# {step.id} {step.title}\n# {utc_now()} cmd: {cmd}\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=str(cfg.board_dir), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             start_new_session=True, pass_fds=(lock_fd,) if lock_fd else (),
                             env=dict(os.environ, PYTHONUNBUFFERED="1"))

        def pump():
            for line in p.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            log.flush()
        th = threading.Thread(target=pump, daemon=True)
        th.start()
        try:
            try:
                rc = p.wait(timeout=timeout_s)
                status = "ok" if rc == 0 else "failed"
            except subprocess.TimeoutExpired:
                say(f"  TIMEOUT: {step.id} exceeded {timeout_s:.0f} s")
                _kill_group(p, say)
                rc, status = p.returncode, "timeout"
        except (KeyboardInterrupt, Interrupted):
            say(f"  INTERRUPTED: stopping {step.id}")
            _kill_group(p, say)
            rc, status = p.returncode, "interrupted"
        th.join(timeout=10)
        log.write(f"# exit {rc} status {status} after {time.monotonic() - t0:.1f} s\n")
    dur = time.monotonic() - t0
    rec.update(finished_utc=utc_now(), duration_s=round(dur, 3), exit_code=rc)
    if status in ("timeout", "interrupted"):
        ar2 = Archiver(rd, f"{step.id}_{status}")
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                ar2.move(f, Path(f.relative_to(staging)))
        _rmdir_empty(staging)
        if ar2.moved:
            say(f"  partial outputs of {step.id} NOT promoted; archived -> {ar2.root.relative_to(rd)}")
    else:
        ar3 = Archiver(rd, f"{step.id}_overwritten")
        outs = {}
        for f in sorted(staging.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(staging)
            dst = rd / rel
            if dst.exists():
                ar3.move(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(f, dst)
            outs[str(rel)] = bc.sha256_file(dst)
        _rmdir_empty(staging)
        rec["outputs"] = outs
        missing = [e for e in step.expect if e not in outs]
        if status == "ok" and missing:
            status = "failed"
            rec["note"] = f"expected outputs missing: {missing}"
        if status == "ok":
            state.setdefault("timing", {})[step.id] = {
                "backend": cfg.backend, "params_key": step.params_key, "duration_s": round(dur, 3),
                "images": step.images, "fixed_s": step.fixed_s,
                "rate_s": (max(0.0, dur - step.fixed_s - STEP_OVERHEAD_S) / step.images
                           if step.kind == "infer" and step.images else None)}
    rec["status"] = status
    rec["log_sha256"] = bc.sha256_file(rd / log_rel)
    save_state(state_path, state)
    if status == "interrupted":
        raise Interrupted(f"during {step.id}")
    return rec


def _rmdir_empty(d: Path):
    for p in sorted(d.rglob("*"), reverse=True) + [d]:
        try:
            p.rmdir()
        except OSError:
            pass
    try:
        d.parent.rmdir()          # .staging itself, if empty
    except OSError:
        pass


# ---- main --------------------------------------------------------------------------------------
def deploy_info_for(backend: str, bit_arg: str | None, board_dir: Path, data_dir: Path) -> tuple[dict, Path]:
    """(deploy info, bit path). Hardware: DEPLOY_INFO.json (required). Dry run: a laptop
    equivalent from the bitstream's summary.json + git (clearly labelled)."""
    info = bc.deploy_info()
    if backend == "pynq":
        rel = bit_arg or info.get("bit")
        if not rel:
            return info, board_dir / "bit" / "MISSING.bit"
        bit = Path(rel)
        return info, bit if bit.is_absolute() else (board_dir / bit)
    bit = Path(bit_arg) if bit_arg else (Path(info["bit"]) if info.get("bit") else None)
    if bit is None:
        raise SystemExit("dry run: no bitstream known; pass --bit v2/vivado/out/<build>/<name>.bit")
    bit = bit.resolve()
    s = read_summary(bit) or {}
    d = {k: v for k, v in info.items() if k in ("commit", "dirty")}
    d.update(origin=f"dry run: laptop git + {bit.parent / 'summary.json'}", bit=str(bit),
             build_id=s.get("build_id", info.get("build_id", "")),
             vivado_version=s.get("vivado_version", ""),
             bit_clock_mhz=s.get("pl_clk0_mhz_actual", info.get("bit_clock_mhz")),
             timing_wns_ns=s.get("wns_ns"))
    if (data_dir / "PACKAGE.json").is_file():
        p = json.loads((data_dir / "PACKAGE.json").read_text())
        d.update(data_git_commit=p.get("git_commit"), data_git_dirty=p.get("git_dirty"))
    return d, bit


def parse_sessions(s: str) -> list[int]:
    if s == "all":
        return [1, 2, 3]
    out = sorted({int(x) for x in s.split(",")})
    if any(x not in (1, 2, 3) for x in out):
        raise argparse.ArgumentTypeError("sessions: 1, 2, 3, all (or e.g. 2,3)")
    return out


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n", 2)[2])
    ap.add_argument("sessions", type=parse_sessions, help="1 | 2 | 3 | all | e.g. 2,3")
    ap.add_argument("--backend", choices=("pynq", "model"), default="pynq")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true", help="(default) skip verified completed steps")
    g.add_argument("--fresh", action="store_true", help="archive state + outputs, start over")
    ap.add_argument("--budget-min", type=float, default=None, help="time budget for this invocation")
    ap.add_argument("--step-timeout-min", type=float, default=None,
                    help="hard per-step timeout (default max(5, 3 x estimate + 2) min)")
    ap.add_argument("--quick", action="store_true",
                    help="first pass: 200 images per step, B3 --n 200, CPU --quick, B1/B2 30 s windows")
    ap.add_argument("--plan", action="store_true", help="pre-flight + print the plan, run nothing")
    ap.add_argument("--steps", nargs="+", default=None, help="only these step ids (e.g. s2.A1)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--bit", default=None, help="hw: default DEPLOY_INFO bit; dry run: a built .bit "
                    "(its summary.json gives the closed clock), default the laptop gos_200 build")
    ap.add_argument("--data-dir", default=str(bc.DEFAULT_DATA_DIR))
    ap.add_argument("--results-dir", default=None,
                    help="default results/ (hw) or results/dryrun/ (model)")
    ap.add_argument("--window-s", type=float, default=None,
                    help="B1/B2 INA260 phase and meter window (default 60; --quick 30; model 2)")
    ap.add_argument("--gap-s", type=float, default=None, help="B1/B2 gap (default 10; model 1)")
    ap.add_argument("--b2-images", type=int, default=100)
    ap.add_argument("--power-repeats", type=int, default=3, help="B1 INA260 protocol repeats")
    ap.add_argument("--b2-power-repeats", type=int, default=3,
                    help="B2 INA260 idle/accel/idle repeats per clock")
    ap.add_argument("--power-rate-hz", type=float, default=10.0, help="INA260 sample rate request")
    ap.add_argument("--with-meter", action="store_true",
                    help="also run the optional external-meter cross-check windows (B1 + B2)")
    ap.add_argument("--write-mode", choices=("elem", "mmio", "slice"), default="elem")
    ap.add_argument("--board-id", default=None)
    ap.add_argument("--clock-tol-mhz", type=float, default=bc.CLOCK_TOL_MHZ)
    ap.add_argument("--min-free-mb", type=float, default=1000.0)
    ap.add_argument("--no-bringup-check", action="store_true",
                    help="run sessions 2/3 although Session 1 is not recorded OK in this state")
    return ap


def main(argv=None, open_backend=open_backend_default, verify_data=bc.verify_manifest,
         steps_hook=None) -> int:
    a = make_parser().parse_args(argv)
    board_dir = bc.BOARD_DIR
    data_dir = Path(a.data_dir).resolve()
    source = bc.SOURCE_DRYRUN if a.backend == "model" else bc.SOURCE_HW
    rd = bc.resolve_out_dir(source, a.results_dir).resolve()
    dry = a.backend == "model"
    window_s = a.window_s if a.window_s is not None else (2.0 if dry else (30.0 if a.quick else 60.0))
    gap_s = a.gap_s if a.gap_s is not None else (1.0 if dry else 10.0)

    # one orchestrator (and its orphaned step) per results dir
    lock_f = open(rd / ".session.lock", "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"REFUSED: another session run (or its step process) holds {rd / '.session.lock'}")
        return 4

    sess_log = rd / "logs" / f"run_sessions_{stamp()}.log"
    sess_log.parent.mkdir(parents=True, exist_ok=True)
    slog = sess_log.open("a")

    def say(msg=""):
        print(msg, flush=True)
        slog.write(msg + "\n")
        slog.flush()

    def on_signal(signum, frame):
        raise Interrupted(signal.Signals(signum).name)
    old = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        return _main(a, board_dir, data_dir, rd, window_s, gap_s, say, lock_f.fileno(),
                     open_backend, verify_data, steps_hook, sess_log)
    except (KeyboardInterrupt, Interrupted) as e:
        say(f"[session] INTERRUPTED ({type(e).__name__} {e}); state saved; rerun the same command to resume")
        return 130
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        slog.close()
        lock_f.close()


def _main(a, board_dir, data_dir, rd, window_s, gap_s, say, lock_fd, open_backend, verify_data,
          steps_hook, sess_log) -> int:
    dry = a.backend == "model"
    tag = "DRY RUN " if dry else ""
    t_start = time.monotonic()
    info, bit = deploy_info_for(a.backend, a.bit, board_dir, data_dir)
    cfg = Config(backend=a.backend, board_dir=board_dir, results_dir=rd, data_dir=data_dir,
                 deploy=info, bit=bit, allow_dirty=a.allow_dirty, clock_tol=a.clock_tol_mhz,
                 min_free_mb=a.min_free_mb, require_clock_equal=any(s >= 2 for s in a.sessions))
    say(f"[session] {tag}sessions {a.sessions} backend={a.backend} results={rd} "
        f"budget={a.budget_min} min quick={a.quick} {utc_now()}")
    if dry:
        say("[session] DRY RUN: every number is model output (source=dryrun_model), NOT a hardware "
            "measurement; outputs only under results/dryrun/.")
    state_path = rd / STATE_NAME
    old_state = load_state(state_path)
    timing = (old_state or {}).get("timing", {})     # durations survive --fresh (not results)

    try:
        prov = preflight(cfg, open_backend, verify_data, say)
    except PreflightError as e:
        say(str(e))
        say("[session] nothing was run.")
        return 3

    if old_state is not None and not a.fresh:
        diff = provenance_diff(old_state["provenance"], prov, cfg.clock_tol)
        if diff:
            say("[session] REFUSED to resume: provenance differs from " + str(state_path) + ":\n  - "
                + "\n  - ".join(diff) + "\n  Results of one state must come from one bitstream, "
                "clock, data package and scripts commit. Use --fresh (old outputs are archived).")
            return 4
        state = old_state
        say(f"[session] resuming state {state_path.name} (created {state['created_utc']}); "
            "provenance identical")
    else:
        why = "fresh" if a.fresh else "pre_state"
        if a.plan:
            say(f"[session] --plan: would archive existing results ({why}) and start a new state")
        elif state_path.exists() or any(rd.glob("hw_*")) or (rd / ".staging").exists():
            ar = archive_everything(rd, why, keep=[sess_log])
            if ar.moved:
                say(f"[session] {why}: archived {len(ar.moved)} entries -> {ar.root.relative_to(rd)}")
        state = new_state(prov, timing)
    if not a.plan:
        state["invocations"].append({"utc": utc_now(), "args": vars(a),
                                     "session_log": str(sess_log.relative_to(rd)),
                                     "preflight": prov})
        save_state(state_path, state)

    steps = build_steps(cfg, a.sessions, a.quick, window_s, gap_s, a.write_mode, a.b2_images,
                        a.board_id, a.power_repeats, a.b2_power_repeats, a.power_rate_hz,
                        a.with_meter)
    if 3 in a.sessions:
        say(f"[session] Session 3 power: primary {POWER_LABEL} via {POWER_HOOK_SCRIPT} "
            f"(B1 per net, B2 per clock); external meter cross-check "
            + ("ON (--with-meter)" if a.with_meter else "off (enable with --with-meter)"))
    if steps_hook:
        steps = steps_hook(steps, cfg)
    if a.steps:
        unknown = set(a.steps) - {s.id for s in steps}
        if unknown:
            say(f"[session] unknown step ids {sorted(unknown)}; known: {[s.id for s in steps]}")
            return 2
        steps = [s for s in steps if s.id in a.steps]

    if (2 in a.sessions or 3 in a.sessions) and 1 not in a.sessions and not a.no_bringup_check:
        s1 = [k for k in ("s1.shell", "s1.smoke")
              if state["steps"].get(k, {}).get("status") != "ok"]
        if s1:
            say(f"[session] REFUSED: Session 1 bring-up not recorded OK in this state ({s1}). Run "
                "session 1 first (same bitstream), or --no-bringup-check.")
            return 4

    budget_s = None if a.budget_min is None else 60.0 * a.budget_min
    report, any_fail, blocked, incomplete = [], False, False, False
    say(f"[session] plan ({len(steps)} steps):")
    for st in steps:
        est, src = estimate(st, state, a.backend)
        done, why = verify_step(st, state["steps"].get(st.id), rd)
        say(f"  {st.id:15} est {est / 60:7.1f} min ({src}); {'DONE, skip' if done else 'run'} ({why})"
            f" — {st.title}")
    if a.plan:
        return 0

    for st in steps:
        rec = state["steps"].get(st.id)
        done, why = verify_step(st, rec, rd)
        est, src = estimate(st, state, a.backend)
        if done:
            say(f"\n[session] SKIP {st.id}: completed {rec.get('finished_utc')} ({why})")
            report.append((st.id, "skipped (done)", rec.get("duration_s"), est, why))
            continue
        if blocked:
            report.append((st.id, "not run", None, est, "Session 1 bring-up failed"))
            incomplete = True
            continue
        elapsed = time.monotonic() - t_start
        remaining = None if budget_s is None else budget_s - elapsed
        if remaining is not None and est > remaining:
            say(f"\n[session] DEFER {st.id}: estimate {est / 60:.1f} min ({src}) > remaining budget "
                f"{remaining / 60:.1f} min")
            state["steps"].setdefault(st.id, {}).update(
                status="deferred", deferred_utc=utc_now(), deferred_estimate_s=round(est, 1))
            save_state(state_path, state)
            report.append((st.id, "DEFERRED", None, est, f"est > remaining {remaining / 60:.1f} min"))
            incomplete = True
            continue
        if a.step_timeout_min is not None:
            timeout = 60.0 * a.step_timeout_min
        else:
            timeout = max(300.0, 3.0 * est + 120.0)
        if remaining is not None:
            timeout = min(timeout, max(remaining, 1.0))
        say(f"\n[session] ===== {tag}{st.id}: {st.title} | est {est / 60:.1f} min ({src}) | "
            f"timeout {timeout / 60:.1f} min | {utc_now()} =====")
        if rec and rec.get("status") in ("running", "interrupted", "timeout"):
            say(f"  previous attempt {rec.get('status')} ({rec.get('started_utc')}): rerun from scratch")
        r = run_step(st, cfg, state, state_path, timeout, say, lock_fd)
        say(f"[session] ===== {st.id}: {r['status'].upper()} (exit {r.get('exit_code')}, "
            f"{r['duration_s']:.1f} s; {len(r['outputs'])} outputs) =====")
        report.append((st.id, r["status"], r["duration_s"], est, r.get("note", "")))
        if r["status"] != "ok":
            if st.required:
                any_fail = True
                if st.session == 1:
                    blocked = True
            else:
                say(f"  ({st.id} is informational: not counted as a failure)")

    say(f"\n[session] {tag}SUMMARY ({(time.monotonic() - t_start) / 60:.1f} min, results {rd})")
    say(f"  {'step':15} {'status':16} {'dur_s':>9} {'est_s':>9}  note")
    for sid, status, dur, est, note in report:
        say(f"  {sid:15} {status:16} {('' if dur is None else f'{dur:.1f}'):>9} {est:9.0f}  {note}")
    deferred = [x[0] for x in report if x[1] == "DEFERRED"]
    if deferred:
        say(f"  deferred (rerun the same command to continue): {deferred}")
    if any_fail:
        return 1
    return 5 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
