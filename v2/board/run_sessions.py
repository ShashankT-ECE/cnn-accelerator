#!/usr/bin/env python3
"""V2 board sessions orchestrator: one command per session, resumable, time-budgeted, pre-flighted.

    ./session.sh 1|2|3|all [--session-index 1|2|3] [--budget-min N] [--resume|--fresh]
                                    [--quick] [--plan]
    ./session.sh all --backend model --allow-dirty          # laptop dry run -> v2/results/dryrun/

Sessions (v2/board/README.md):
  1  bring-up       [clock fallback: 300 MHz build, else 250 MHz] test_shell (--skip-scratch),
                    test_core_smoke (+ informational --write-mode slice, + informational s1.fast =
                    fast host path bring-up check), s1.fcal = PL clock calibration (f_meas)
  2  A1-A4, B3, CPU s2.fcal, A2/A3 cycles, A1 accuracy, B3 latency breakdown (interleaved safe /
                    fast host path / CPU), CPU baselines (A5), DPU baseline (only if
                    dpu/dpu_session.py exists), A4 utilization
  3  B1, B2         s3.fcal, B1 INA260 SOM-rail power protocol per net (idle/accel/idle/control/
                    idle/cpu x3 + final idle, run phases in a seeded random order per repeat), B2
                    clock sweep, (+ extra steps, e.g. soak)
Power: the on-board INA260 ("SOM-rail power (INA260)") is the ONLY power source.

PRIORITY (steps run in this order; the time budget therefore defers the tail first):
  bring-up, f_meas calibration, then A3, A1, latency (A2 + B3), shapes (A3-general random shape
  jobs, s2.shapes, 15 min cap), baselines (CPU, DPU), energy (B1), sweep (B2), soak, A4.
  Session 1 always first; otherwise the order is global across sessions.
EXTRA STEPS: if session_extra_steps.py is importable, session_extra_steps.extra_steps(opts) ->
  list[dict] (id, session, group, cmd, outputs, est_s, timeout_s, requires, params; optional
  title, out_flag (default "--out-dir"; the orchestrator appends `out_flag <staging dir>`),
  required, kind, images) become normal resumable steps slotted into their group; an extra step
  with the id of a built-in one replaces it. `opts` (extra_opts()) is a SimpleNamespace with:
  backend, dry, board_dir, data_dir, results_dir, bit, closed_mhz, clock_tol, deploy, allow_dirty,
  quick, sessions, session_index, write_mode, host_path, fast_store, board_id, window_s, gap_s,
  power_repeats, b2_power_repeats, power_rate_hz, b2_images, order_seed, python, clock_choice
  (path of the clock_fallback decision JSON), sensor_args (["--sensor", "mock"] in a dry run).
CLOCK FALLBACK: with several deployed bitstreams (DEPLOY_INFO "bits", highest clock first) and
  clock_fallback.py importable, Session 1 calls clock_fallback.choose(ctx, bits, run_smoke) with
  bits = [{"bit", "clock_mhz"}] and run_smoke(entry) = pre-flight of that bitstream + core smoke
  (test_core_smoke.py) on it; the chosen bitstream is recorded in the state provenance
  (bit_choice) and <results>/hw_clock_choice.json, and every later step / session uses it.
  ctx = SimpleNamespace(say, dry, cfg, results_dir, board_dir).

ENVIRONMENT PRE-FLIGHT (board_env.py; recorded in the state provenance and, through $GOS_RUN_ENV,
in every results row: session_index, paper_grade, env_step, cpu_governor, cpu_freq_khz,
cpu_affinity, die_temp_start_c):
  * no package manager running (apt, apt-get, dpkg, unattended-upgrades, packagekitd, ...;
    offenders listed) -> else FAIL;
  * cpufreq: governor 'performance' + scaling_min = scaling_max = --cpu-freq-khz (default the
    highest available) on every policy, scaling_cur_freq read back == target -> else FAIL;
    restored to the saved values when the orchestrator exits;
  * pinning: the orchestrator runs on the housekeeping cores, every step process is pinned to
    --meas-cores (default 3) with sched_setaffinity (= taskset) and read back -> else FAIL;
    the CPU baseline step runs on --cpun-cores (0-3) and pins its 1-thread workers to
    --cpu1-cores (3) and multi-thread workers to --cpun-cores;
  * die temperature (Zynq MPSoC AMS via iio, else hwmon) at the start and end of every step
    (non-fatal: "unavailable" recorded).
  FAILs abort (exit 3) unless --allow-non-paper-grade, which turns them into warnings and marks
  every row paper_grade=False (so do --allow-dirty and every dry run). Dry runs use a FAKE sysfs/
  proc tree under <results>/.dryrun_env (DRY RUN on every line); the laptop's cpufreq is never
  touched.
MEASURED PL CLOCK: sN.fcal (exp_fclk_cal.py) measures f_meas per session; later steps get it in
  $GOS_RUN_ENV and record it beside the read-back clock as a CROSS-CHECK; every µs value uses the
  pl_clk0 PLL read-back (clock of record, DECISIONS D20). |f_meas - read-back| > 0.1 % is FLAGGED.
SESSION INDEX (3-session repeatability): --session-index K tags every row; results of K = 1 go to
  <results>/, of K >= 2 to <results>/rep<K>/; aggregate_sessions.py combines them.

Pre-flight (fails loudly, exit 3) also checks: DEPLOY_INFO present; .bit/.hwh SHA256 vs
DEPLOY_INFO and the shipped .bit.sha256; summary.json build id / closed clock / timing met;
VERSION == 0x474F5302; BUILD_ID register == DEPLOY_INFO build_id; pl_clk0 SET to the closed clock
after the overlay load and read back == it within 0.1 MHz in every session (2026-09-30: PYNQ left
the 300 MHz build at 199.998 MHz); pl_clk0 read back <= the closed
clock (+tol) and, for sessions 2/3, equal to it (within tol); every data-package SHA256; free disk;
clean-tree flags (--allow-dirty marks rows git_dirty=True, invalid for the paper).

State: <results>/session_state.json records the provenance and every step (status, command,
duration, outputs + SHA256, log, pinning read-back, die temperature start/end). On rerun (default
--resume) a step is skipped only if it finished OK with the same parameters and all its outputs
still verify; otherwise it is rerun FROM SCRATCH. A provenance change (bitstream / BUILD_ID / clock
/ data / scripts commit / session index / paper grade / CPU frequency) refuses to resume (exit 4):
--fresh. --fresh and replaced outputs are ARCHIVED into <results>/archive/<ts>_<why>/, never
deleted. Steps write into <results>/.staging/<step>/ and are promoted when the step exits.
Budget: --budget-min; a step whose estimate does not fit is DEFERRED. Hard per-step timeout.

Exit: 0 all requested steps OK; 1 a step failed; 2 usage; 3 pre-flight failed; 4 provenance
changed / bring-up missing / lock held; 5 incomplete (deferred / not run); 130 interrupted.
"""
from __future__ import annotations

import argparse
import csv
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
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import board_common as bc
import board_env as benv
import stats

STATE_NAME = "session_state.json"
STATE_VERSION = 1
VERSION_CORE = 0x474F5302
OFF_VERSION, OFF_BUILD_ID = 0x0F8, 0x0FC        # FORMATS.md CSR map (= gos_driver)
CHOICE_FILE = "hw_clock_choice.json"            # = clock_fallback.CHOICE_FILE

DEFAULT_RATE_S = {"pynq": 0.02, "model": 0.006}
DEFAULT_FIXED_S = {  # kind -> backend -> seconds
    "bringup": {"pynq": 120.0, "model": 30.0},
    "cpu": {"pynq": 2400.0, "model": 600.0},
    "cpu_quick": {"pynq": 600.0, "model": 120.0},
}
STEP_OVERHEAD_S = 30.0          # overlay load, package verify, net loads
KILL_GRACE_S = (20.0, 5.0)      # SIGINT -> SIGTERM, SIGTERM -> SIGKILL
RESULT_PATTERNS = ("hw_*.csv", "hw_*.npz", "hw_*.json")

# Priority (EXPERIMENTS.md "Measurement rigor"): infrastructure first, then these groups.
GROUP_ORDER = ("A3", "A1", "latency", "shapes", "baselines", "energy", "sweep", "soak", "A4")
INFRA_RANK = {"bringup": -2, "fcal": -1}

DEFAULT_MEAS_CORES = "3"
DEFAULT_CPU1_CORES = "3"
DEFAULT_CPUN_CORES = "0-3"


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
    requires: tuple = ()            # step ids that must be recorded ok before this step runs
    group: str = ""                 # priority group (GROUP_ORDER / INFRA_RANK)
    est_s: float | None = None      # explicit estimate (extra steps)
    timeout_s: float | None = None  # explicit hard timeout (extra steps)
    params: dict = field(default_factory=dict)   # extra resume-key parameters
    cores: str = "meas"             # meas | cpun: which core set the step process is pinned to
    cond_argv: tuple = ()           # ((step id, [args]), ...): args appended only if that step is OK

    @property
    def params_key(self) -> str:
        if self.params:
            return json.dumps({"argv": self.argv, "params": self.params}, sort_keys=True)
        return json.dumps(self.argv)


def resolve_step(step: Step, state: dict) -> Step:
    """Append the conditional arguments whose prerequisite step is recorded OK."""
    if not step.cond_argv:
        return step
    extra = []
    for sid, args in step.cond_argv:
        if state.get("steps", {}).get(sid, {}).get("status") == "ok":
            extra += list(args)
    return replace(step, argv=[*step.argv, *extra], cond_argv=())


def group_rank(g: str) -> int:
    if g in INFRA_RANK:
        return INFRA_RANK[g]
    return GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER)


def prioritize(steps: list[Step]) -> list[Step]:
    """Session 1 first; then global priority by group; stable within a group."""
    return [s for _, s in sorted(enumerate(steps), key=lambda t: (
        0 if t[1].session == 1 else 1, group_rank(t[1].group), t[0]))]


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


def dpu_session_path(board_dir: Path) -> Path | None:
    """v2/dpu/dpu_session.py (other owner): ~/gos/dpu/ on the board, v2/dpu/ in the repo."""
    for p in (Path(board_dir) / "dpu" / "dpu_session.py", Path(board_dir).parent / "dpu" / "dpu_session.py"):
        if p.is_file():
            return p
    return None


def _b2_extra_outputs() -> tuple:
    try:
        import session_extra_steps as sx
        return tuple(getattr(sx, "B2_EXTRA_OUTPUTS", ()))
    except ImportError:
        return ()


def build_steps(cfg: Config, sessions, quick: bool = False, window_s: float = 60.0,
                gap_s: float = 10.0, write_mode: str = "elem", b2_images: int = 100,
                board_id: str | None = None, power_repeats: int = 3, b2_power_repeats: int = 3,
                power_rate_hz: float = 10.0, host_path: str = "safe",
                fast_store: str = "block", power_control: bool = True, order_seed: int = 1,
                fcal_s: float | None = None, cpu1_cores: str = DEFAULT_CPU1_CORES,
                cpun_cores: str = DEFAULT_CPUN_CORES, b3_cpu: bool = True) -> list[Step]:
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
    fs = ["--fast-store", fast_store]
    hp = ["--host-path", host_path, *fs]
    need = ("s1.fast",) if host_path == "fast" else ()
    nets = bc.NETS
    n_all = sum(n_images(cfg.data_dir, n) for n in nets)
    lim = 200 if quick else None
    n_lim = sum(min(lim, n_images(cfg.data_dir, n)) for n in nets) if lim else n_all
    fcal_s = fcal_s if fcal_s is not None else (1.0 if cfg.dry else (4.0 if quick else 8.0))
    steps: list[Step] = []

    def fcal(s: int) -> Step:
        return Step(f"s{s}.fcal", s, f"PL clock cross-check: f_meas vs the read-back clock (cycle counter vs "
                    f"CLOCK_MONOTONIC_RAW, {fcal_s:g} s)",
                    ["exp_fclk_cal.py", *dev, *wm, *dirty, "--seconds", f"{fcal_s:g}", "--tag",
                     f"_s{s}", "--seed", str(order_seed + 100 * s)],
                    kind="fixed", fixed_s=fcal_s + 10.0, expect=(f"hw_fclk_cal_s{s}.csv",),
                    group="fcal")

    if 1 in sessions:
        shell = ["test_shell.py", "--bit", str(cfg.bit), "--expect-version", f"0x{VERSION_CORE:08X}",
                 "--skip-scratch", "--max-fclk0", maxf]
        if cfg.dry:
            shell += ["--backend", "model", "--clock-mhz", f"{closed:.6f}"]
        else:
            shell += ["--set-fclk0", f"{closed:.6f}"]
        steps += [
            Step("s1.shell", 1, "shell/memory smoke (test_shell --skip-scratch)", shell,
                 kind="bringup", out_flag=None, group="bringup"),
            Step("s1.smoke", 1, "core smoke (VERSION/BUILD_ID, LeNet+CIFAR bit/cycle-exact, refused job)",
                 ["test_core_smoke.py", *dev, "--max-fclk0", maxf], kind="bringup", out_flag=None,
                 group="bringup"),
            Step("s1.smoke_slice", 1, "core smoke with --write-mode slice (informational)",
                 ["test_core_smoke.py", *dev, "--max-fclk0", maxf, "--write-mode", "slice"],
                 kind="bringup", out_flag=None, required=False, group="bringup"),
            Step("s1.fast", 1, "fast host path bring-up check (ACT0 fast write + per-word readback, "
                 "CSR block reads; core smoke on the fast path) (informational)",
                 ["test_core_smoke.py", *dev, "--max-fclk0", maxf, "--host-path", "fast", *fs],
                 kind="bringup", out_flag=None, required=False, group="bringup"),
            fcal(1),
        ]
    if 2 in sessions:
        lim_a = ["--limit", str(lim)] if lim else []
        a4_lim = lim or 1000
        b3_n = 200 if quick else 1000
        cpu_ok = (cfg.board_dir / "cpu" / "cpu_infer.py").is_file() and b3_cpu
        b3 = ["exp_b3_breakdown.py", *dev, *wm, *dirty, "--n", str(b3_n), "--conditions", "safe",
              *(["cpu"] if cpu_ok else []), *fs, "--seed", str(order_seed + 2)]
        steps += [
            fcal(2),
            Step("s2.A2A3", 2, "A2/A3 cycles", ["exp_a2_a3_cycles.py", *dev, *wm, *hp, *dirty,
                                                *lim_a],
                 images=n_lim, expect=("hw_a2_a3_cycles.csv",), requires=need, group="A3"),
            Step("s2.A1", 2, "A1 accuracy", ["exp_a1_accuracy.py", *dev, *wm, *hp, *dirty, *lim_a],
                 images=n_lim, expect=("hw_a1_accuracy.csv",), requires=need, group="A1"),
            Step("s2.B3", 2, "B3 latency breakdown, interleaved conditions (safe host path"
                 + (", CPU cpu_int8_ref x1" if cpu_ok else "") + "; + fast host path if s1.fast OK)",
                 b3, images=len(nets) * (b3_n + 100) * (3 if cpu_ok else 2),
                 expect=("hw_b3_breakdown.csv",) + (("hw_b3_breakdown_cpu.csv",) if cpu_ok else ()),
                 group="latency", cond_argv=(("s1.fast", ["--conditions-add", "fast"]),)),
        ]
        cpu = ["cpu/run_cpu_baselines.py", "--data-dir", dd, "--tag",
               "laptop" if cfg.dry else "board", *(["--quick"] if quick else []), *dirty,
               "--pin-1t", cpu1_cores, "--pin-nt", cpun_cores]
        if (cfg.board_dir / "cpu" / "run_cpu_baselines.py").is_file():
            steps.append(Step("s2.CPU", 2, "CPU baselines (A5)", cpu,
                              kind="cpu_quick" if quick else "cpu", expect=("hw_cpu_baseline.csv",),
                              group="baselines", cores="cpun"))
        dpu = dpu_session_path(cfg.board_dir)
        if dpu is not None:
            argv = [str(dpu), "--data-dir", dd, "--phase-s", f"{window_s:g}", "--repeats",
                    str(power_repeats), *dirty]
            if cfg.dry:
                argv += ["--dry-run", "--limit", "200"]
                pk = dpu.parent / "build" / "package"
                if pk.is_dir():
                    argv += ["--pkg-dir", str(pk)]
            steps.append(Step("s2.DPU", 2, f"DPU baseline ({dpu.name}, other owner; informational)",
                              argv, kind="fixed",
                              fixed_s=(120.0 if cfg.dry else 3 * power_repeats * 2 * window_s + 1200.0),
                              required=False, group="baselines"))
        steps.append(Step("s2.A4", 2, "A4 utilization", ["exp_a4_util.py", *dev, *wm, *hp, *dirty,
                                                         "--limit", str(a4_lim)],
                          images=sum(min(a4_lim, n_images(cfg.data_dir, n)) for n in nets),
                          expect=("hw_a4_util.csv",), requires=need, group="A4"))
    if 3 in sessions:
        w = ["--window-s", f"{window_s:g}", "--gap-s", f"{gap_s:g}"]
        clocks = bc.b2_sweep_clocks(closed)
        steps.append(fcal(3))
        steps += power_hook_steps(cfg, dev, dirty, window_s, power_repeats, power_rate_hz,
                                  [*wm, *hp], control=power_control, requires=need,
                                  order_seed=order_seed)
        b2_net = "lenet5"
        b2_power = [f"hw_b2_power_ina260_{k}_{b2_net}_{clock_tag(c)}.csv"
                    for c in clocks for k in ("samples", "phases", "summary")]
        per_clock = 3 * b2_power_repeats * window_s + B2_PER_CLOCK_OVERHEAD_S
        steps.append(
            Step("s3.B2", 3, f"B2 clock sweep {clocks} MHz (closed {closed}): cycles == model + "
                 f"{POWER_LABEL} idle/accel/idle x{b2_power_repeats} per clock",
                 ["exp_b2_clock.py", *dev, *wm, *dirty, "--net", b2_net, "--clocks",
                  *[f"{c:.6f}" for c in clocks], "--max-mhz", f"{closed:.6f}",
                  *hp, "--images", str(b2_images), *w, "--power-repeats", str(b2_power_repeats),
                  "--rate-hz", f"{power_rate_hz:g}",
                  *(["--sensor", "mock"] if cfg.dry else [])], kind="infer",
                 images=len(clocks) * b2_images, fixed_s=len(clocks) * per_clock,
                 expect=("hw_b2_clock.csv", *b2_power, *_b2_extra_outputs()), requires=need,
                 group="sweep"))
    return steps


# ==== POWER HOOK (Session 3) ====================================================================
# B1 = the INA260 SOM-rail power protocol (power_log.py), one step per net; the ONLY power source.
# Phases per repeat idle + {accel, control, cpu} each bracketed by idle phases, the three run
# phases in a seeded random order per repeat (--order random --order-seed S, recorded), x R + one
# final idle -> 6R+1 phases (--no-power-control: 4R+1). CPU phases = onnxruntime INT8, 4 threads on
# cores 0-3 (user decision 2026-09-30: the CPU condition is the fastest CPU baseline; in the board
# probe of 2026-09-30 ORT INT8 x4 had the lowest end-to-end time on both nets, see hw_cpu_baseline.csv).
POWER_HOOK_SCRIPT = "power_log.py"
POWER_LABEL = "SOM-rail power (INA260)"          # = power_log.LABEL
B1_CPU_KIND, B1_CPU_THREADS, B1_CPU_CORES = "cpu_ort_int8", 4, "0-3"
B1_POWER_OVERHEAD_S = 60.0     # package verify + load_net + CPU runner build + warm-ups
B2_PER_CLOCK_OVERHEAD_S = 20.0  # set clock, soft_reset, reload + readback, sensor probe, CSVs


def clock_tag(mhz: float) -> str:
    """B2 per-clock file tag (= power_log.clock_tag): 199.998001 -> '200mhz'."""
    return f"{int(round(float(mhz)))}mhz"


def power_phases(repeats: int, control: bool = True) -> int:
    return (6 if control else 4) * repeats + 1


def power_hook_steps(cfg: Config, dev: list, dirty: list, window_s: float, repeats: int = 3,
                     rate_hz: float = 10.0, wm: list | None = None, control: bool = True,
                     requires: tuple = (), order_seed: int = 1) -> list[Step]:
    extra = ["--sensor", "mock"] if cfg.dry else []
    out = []
    for i, net in enumerate(bc.NETS):
        tag = f"_{net}"
        out.append(Step(
            f"s3.B1.{net}", 3, f"B1 {POWER_LABEL} protocol {net} "
            f"({'accel/control/cpu' if control else 'accel/cpu'} bracketed by idle, random order, "
            f"x{repeats} + idle, {window_s:g} s phases, CPU {B1_CPU_KIND} x{B1_CPU_THREADS})",
            [POWER_HOOK_SCRIPT, *dev, *(wm or []), *dirty, "--protocol", "--net", net, "--tag", tag,
             "--phase-s", f"{window_s:g}", "--repeats", str(repeats), "--rate-hz", f"{rate_hz:g}",
             "--cpu-kind", B1_CPU_KIND, "--cpu-threads", str(B1_CPU_THREADS),
             "--cpu-cores", B1_CPU_CORES,
             "--order", "random", "--order-seed", str(order_seed + 10 + i),
             *([] if control else ["--no-control"]), *extra],
            kind="fixed", fixed_s=power_phases(repeats, control) * window_s + B1_POWER_OVERHEAD_S,
            requires=requires, group="energy",
            expect=tuple(f"hw_b1_power_ina260_{k}{tag}.csv" for k in ("samples", "phases", "summary"))))
    return out
# ==== end POWER HOOK ============================================================================


# ---- extra steps (session_extra_steps.py, other owner) -----------------------------------------
def extra_opts(cfg: Config, a, sessions, window_s, gap_s, order_seed) -> SimpleNamespace:
    return SimpleNamespace(
        backend=cfg.backend, dry=cfg.dry, board_dir=cfg.board_dir, data_dir=str(cfg.data_dir),
        results_dir=str(cfg.results_dir), bit=str(cfg.bit), closed_mhz=cfg.closed_mhz,
        clock_tol=cfg.clock_tol, deploy=cfg.deploy, allow_dirty=cfg.allow_dirty, quick=a.quick,
        sessions=list(sessions), session_index=a.session_index, write_mode=a.write_mode,
        host_path=a.host_path, fast_store=a.fast_store, board_id=a.board_id, window_s=window_s,
        gap_s=gap_s, power_repeats=a.power_repeats, b2_power_repeats=a.b2_power_repeats,
        power_rate_hz=a.power_rate_hz, b2_images=a.b2_images, order_seed=order_seed,
        python=cfg.python, clock_choice=str(cfg.results_dir / CHOICE_FILE),
        sensor_args=["--sensor", "mock"] if cfg.dry else [],
        **({"soak_block_s": 10.0} if cfg.dry else {}))   # dry-run soak (60 s): both nets alternate


def step_from_dict(d: dict) -> Step:
    g = d.get("group", "")
    return Step(id=str(d["id"]), session=int(d["session"]), title=str(d.get("title") or d["id"]),
                argv=[str(x) for x in d["cmd"]], kind=d.get("kind", "fixed"),
                images=int(d.get("images") or 0), fixed_s=float(d.get("est_s") or 0.0),
                out_flag=d.get("out_flag", "--out-dir"), expect=tuple(d.get("outputs", ())),
                required=bool(d.get("required", True)), requires=tuple(d.get("requires", ())),
                group=g, est_s=float(d["est_s"]) if d.get("est_s") is not None else None,
                timeout_s=float(d["timeout_s"]) if d.get("timeout_s") is not None else None,
                params=dict(d.get("params") or {}))


def merge_extra_steps(steps: list[Step], opts, sessions, say=print) -> list[Step]:
    """Import session_extra_steps (absent -> unchanged) and merge its steps (same id replaces)."""
    try:
        import session_extra_steps as sx
    except ImportError:
        say("[session] extra steps: session_extra_steps.py not present (none added)")
        return steps
    try:
        extra = [step_from_dict(d) for d in sx.extra_steps(opts)]
    except Exception as e:  # noqa: BLE001 - a broken extra module must not break the sessions
        say(f"[session] extra steps: session_extra_steps.extra_steps failed: {type(e).__name__}: {e}"
            " (none added)")
        return steps
    by = {s.id: i for i, s in enumerate(steps)}
    out = list(steps)
    added = []
    for s in extra:
        if s.session not in sessions:
            continue
        if s.group not in GROUP_ORDER:
            say(f"[session] extra step {s.id}: unknown group {s.group!r}; scheduled last")
        if s.id in by:
            out[by[s.id]] = s
            added.append(f"{s.id} (replaces built-in)")
        else:
            out.append(s)
            added.append(s.id)
    say(f"[session] extra steps from session_extra_steps.py: {added or 'none for these sessions'}")
    return out


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
    tmp.write_text(json.dumps(state, indent=1, sort_keys=False, default=str) + "\n")
    os.replace(tmp, path)


PROV_KEYS = ("backend", "build_id_hw", "bit_sha256", "closed_clock_mhz", "data_package_sha256",
             "scripts_commit", "scripts_dirty", "session_index", "paper_grade", "cpu_freq_khz")


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
    (except the paths in keep, e.g. the session log of this invocation and the
    idle-reference files measured before the archive step)."""
    ar = Archiver(results_dir, label)
    keep = {Path(k).resolve() for k in keep}
    ar.move(results_dir / STATE_NAME)
    for pat in RESULT_PATTERNS:
        for p in sorted(results_dir.glob(pat)):
            if p.resolve() not in keep:
                ar.move(p)
    for p in sorted((results_dir / "logs").rglob("*")) if (results_dir / "logs").is_dir() else []:
        if p.is_file() and p.resolve() not in keep:
            ar.move(p)
    ar.move(results_dir / ".staging")
    return ar


def new_state(prov: dict, timing: dict | None = None) -> dict:
    return {"version": STATE_VERSION, "created_utc": utc_now(), "updated_utc": "",
            "provenance": prov, "steps": {}, "invocations": [], "timing": timing or {},
            "order_seed": stats.new_seed()}


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
    if step.est_s is not None:
        return float(step.est_s), "step estimate (extra step)"
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
    return D.PynqBackend(cfg.bit, download=True, clock_mhz=cfg.closed_mhz)   # set + verified


def read_summary(bit: Path) -> dict | None:
    p = bit.parent / "summary.json"
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except (OSError, ValueError):
        return None


def preflight(cfg: Config, open_backend=open_backend_default, verify_data=bc.verify_manifest,
              say=print) -> dict:
    """All device/bitstream/data checks; prints each; raises PreflightError listing every failure.
    Returns provenance."""
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
    say(f"[{tag}preflight] backend={cfg.backend} bitstream={cfg.bit} deploy info: {info.get('origin')}")
    if cfg.dry:
        say(f"  [{tag}preflight] DRY RUN: ModelBackend equivalents; VERSION/BUILD_ID/pl_clk0 are the "
            "simulated register map / nominal clock (set from the deploy info), NOT hardware.")
    elif info.get("origin") != "DEPLOY_INFO.json":
        bad("DEPLOY_INFO.json missing next to the scripts: deploy with deploy.sh (hardware rows need it)")

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
            tol = min(cfg.clock_tol, bc.FCLK_SET_TOL_MHZ)
            (ok if abs(clk - closed) <= tol else bad)(
                f"pl_clk0 {clk:.6f} MHz == closed clock {closed:.6f} MHz (±{tol}): the design runs "
                "at the clock it was timed for"
                + ("" if abs(clk - closed) <= tol else " — NOT at the closed clock"))

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

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    free_mb = shutil.disk_usage(cfg.results_dir).free / 2**20
    (ok if free_mb >= cfg.min_free_mb else bad)(
        f"free disk at {cfg.results_dir}: {free_mb:.0f} MB >= {cfg.min_free_mb:.0f} MB")

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
            "clock_requested_mhz": info.get("bit_clock_requested_mhz"),
            "data_package_sha256": pkg_sha, "manifest_sha256": man_sha,
            "scripts_commit": info.get("commit", ""), "scripts_dirty": sdirty,
            "data_dirty": ddirty, "allow_dirty": cfg.allow_dirty,
            "deploy_origin": info.get("origin"), "preflight_utc": utc_now(), "warnings": warns}
    if fails:
        raise PreflightError(f"{tag}PRE-FLIGHT FAILED ({len(fails)}):\n  - " + "\n  - ".join(fails))
    say(f"[{tag}preflight] PASSED ({len(warns)} warnings)")
    return prov


# ---- environment pre-flight ---------------------------------------------------------------------
@dataclass
class EnvPaths:
    sysfs_root: str = benv.SYSFS_CPU
    proc_root: str = benv.PROC
    iio_root: str = benv.IIO
    hwmon_root: str = benv.HWMON
    fake: bool = False


def dryrun_env_paths(rd: Path) -> EnvPaths:
    root = rd / ".dryrun_env"
    if root.exists():
        shutil.rmtree(root)
    p = benv.make_fake_tree(root)
    return EnvPaths(**p, fake=True)


def env_preflight(paths: EnvPaths, *, dry: bool, plan: bool, allow_non_paper: bool,
                  allow_dirty: bool, meas_cores: str, cpu1_cores: str, cpun_cores: str,
                  cpu_freq_khz: int | None, say=print) -> tuple[dict, benv.CpuFreqControl]:
    """Package managers, governor + fixed frequency, pinning, die temperature. Raises
    PreflightError on a paper-grade FAIL (unless allow_non_paper). Returns (env provenance,
    the cpufreq controller holding the saved settings for restore())."""
    tag = "DRY RUN " if dry else ""
    fails, warns = [], []
    downgrade = allow_non_paper or dry

    def ok(msg):
        say(f"  [{tag}env] OK    {msg}")

    def fail(msg):
        if downgrade:
            say(f"  [{tag}env] WARN  {msg} — {'dry run' if dry else '--allow-non-paper-grade'}: "
                "rows are NOT paper-grade")
            warns.append(msg)
        else:
            say(f"  [{tag}env] FAIL  {msg}")
            fails.append(msg)

    def warn(msg):
        say(f"  [{tag}env] WARN  {msg}")
        warns.append(msg)

    say(f"[{tag}env] measurement environment pre-flight"
        + (f" on a FAKE sysfs/proc tree ({Path(paths.sysfs_root).parents[3]}) — DRY RUN, the laptop's "
           "cpufreq is not touched" if paths.fake else ""))
    # package managers
    busy = benv.pkg_manager_busy(paths.proc_root)
    if busy:
        fail("package manager running: " + "; ".join(f"pid {b['pid']} {b['comm']} ({b['cmdline']})"
                                                     for b in busy)
             + " — wait or `sudo systemctl stop unattended-upgrades packagekit`")
    else:
        ok("no apt/dpkg/unattended-upgrades/packagekitd process running")
    # governor + frequency
    ctl = benv.CpuFreqControl(root=paths.sysfs_root, fake=paths.fake)
    before = ctl.snapshot()
    gov_res = {"ok": None, "target_khz": ctl.target_khz(cpu_freq_khz), "errors": [], "readback": {}}
    if plan:
        say(f"  [{tag}env] --plan: governor not changed; now "
            f"{ {k: v['scaling_governor'] for k, v in before['policies'].items()} }, "
            f"cur kHz {before['cpu_cur_freq_khz']}; would fix {gov_res['target_khz']} kHz")
    else:
        gov_res = ctl.apply(cpu_freq_khz, settle=(lambda: benv.fake_kernel_settle(paths.sysfs_root))
                            if paths.fake else None)
        if gov_res["ok"]:
            ok(f"cpufreq: governor performance, fixed {gov_res['target_khz']} kHz on every policy; "
               f"scaling_cur_freq read back {gov_res['readback']['cpu_cur_freq_khz']} (restored on exit)")
        else:
            fail("cpufreq governor/frequency: " + "; ".join(gov_res["errors"]))
    # pinning
    avail = benv.available_cores()
    pins = {}
    for name, spec in (("meas", meas_cores), ("cpu1", cpu1_cores), ("cpun", cpun_cores)):
        c = benv.parse_cores(spec)
        if not c or not c <= avail:
            fail(f"--{name}-cores {spec!r} not a subset of the available cores {benv.cores_str(avail)}")
        pins[name] = benv.cores_str(c)
    hk = (avail - benv.parse_cores(meas_cores)) or avail
    orig_aff = benv.affinity_of(0)
    pin_self = benv.pin(0, hk)
    pin_self["original"] = orig_aff
    if pin_self["ok"]:
        ok(f"orchestrator pinned to housekeeping cores {pin_self['readback']}; steps -> {pins['meas']}, "
           f"CPU baseline -> {pins['cpun']} (1-thread workers {pins['cpu1']})")
    else:
        fail(f"pinning (sched_setaffinity = taskset): {pin_self['error']}")
    # temperature
    t = benv.read_die_temp(paths.iio_root, paths.hwmon_root)
    if t["max_c"] is None:
        warn("die temperature unavailable (no AMS iio / hwmon node): recorded as 'unavailable'")
    else:
        ok(f"die temperature {t['max_c']:.1f} °C max ({t['source']}: {t['channels_c']})"
           + (" [FAKE]" if paths.fake else ""))
    paper = not dry and not allow_dirty and not allow_non_paper and not fails
    env = {"paper_grade": paper, "pkg_manager_offenders": busy, "cpufreq_before": before,
           "cpufreq_apply": {k: v for k, v in gov_res.items()},
           "cpu_freq_khz": gov_res.get("target_khz"), "cpu_governor": "performance" if gov_res["ok"] else
           ",".join(sorted({v["scaling_governor"] or "" for v in before["policies"].values()})),
           "meas_cores": pins.get("meas"), "cpu1_cores": pins.get("cpu1"), "cpun_cores": pins.get("cpun"),
           "housekeeping_cores": benv.cores_str(hk), "orchestrator_pin": pin_self,
           "die_temp": t, "fake_tree": paths.fake, "paths": vars(paths), "warnings": warns,
           "allow_non_paper_grade": allow_non_paper}
    if fails:
        raise PreflightError(f"{tag}ENVIRONMENT PRE-FLIGHT FAILED ({len(fails)}; override only with "
                             "--allow-non-paper-grade, rows then NOT paper-grade):\n  - "
                             + "\n  - ".join(fails))
    say(f"[{tag}env] {'PAPER-GRADE' if paper else 'NOT paper-grade'} environment "
        f"({len(warns)} warnings)")
    return env, ctl


# ---- f_meas ------------------------------------------------------------------------------------
def fmeas_for(state: dict, rd: Path, session: int) -> dict:
    """f_meas fields for $GOS_RUN_ENV: the calibration of this session if OK, else the latest OK
    calibration in the state; {} if none."""
    cands = []
    for sid, rec in state.get("steps", {}).items():
        if sid.endswith(".fcal") and rec.get("status") == "ok":
            cands.append((sid == f"s{session}.fcal", rec.get("finished_utc", ""), sid, rec))
    for _, _, sid, rec in sorted(cands, reverse=True):
        name = next((n for n in rec.get("outputs", {}) if n.startswith("hw_fclk_cal") and n.endswith(".csv")), None)
        if not name or not (rd / name).is_file():
            continue
        with (rd / name).open() as f:
            rows = list(csv.DictReader(f))
        if not rows or not rows[-1].get("f_meas_mhz"):
            continue
        r = rows[-1]
        return {"f_meas_mhz": float(r["f_meas_mhz"]), "f_meas_ci_lo_mhz": r.get("f_meas_ci_lo_mhz", ""),
                "f_meas_ci_hi_mhz": r.get("f_meas_ci_hi_mhz", ""),
                "f_meas_readback_mhz": float(r["f_readback_mhz"]),
                "f_meas_source": r.get("f_meas_source", ""), "f_meas_step": sid,
                "f_meas_rel_diff_pct": r.get("rel_diff_pct", ""),
                "f_meas_flag": str(r.get("fcal_flag", "")).lower() == "true"}
    return {}


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


def _env_log(rd: Path, rec: dict):
    (rd / "logs").mkdir(exist_ok=True)
    with (rd / "logs" / "env_log.jsonl").open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def run_step(step: Step, cfg: Config, state: dict, state_path: Path, timeout_s: float, say,
             lock_fd: int | None = None, run_env: dict | None = None, cores=None,
             env_paths: EnvPaths | None = None, strict_pin: bool = False) -> dict:
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
    a0 = step.argv[0]
    if a0.endswith(".py"):
        cmd = [cfg.python, str(cfg.board_dir / a0), *step.argv[1:]]
    else:
        cmd = [a0 if os.path.isabs(a0) else str(cfg.board_dir / a0), *step.argv[1:]]
    if step.out_flag:
        cmd += [step.out_flag, str(staging)]
    temp0 = benv.read_die_temp(env_paths.iio_root, env_paths.hwmon_root) if env_paths else {}
    renv = dict(run_env or {})
    renv.update(step_id=step.id, die_temp_start_c=temp0.get("max_c") if temp0.get("max_c") is not None
                else "unavailable")
    rec = {"status": "running", "title": step.title, "session": step.session, "group": step.group,
           "params_key": step.params_key, "cmd": cmd, "started_utc": utc_now(),
           "timeout_s": round(timeout_s, 1), "provenance": state["provenance"], "outputs": {},
           "log": log_rel, "required": step.required, "die_temp_start": temp0,
           "run_env": renv}
    state["steps"][step.id] = rec
    save_state(state_path, state)
    say(f"  $ {' '.join(cmd)}")
    t0 = time.monotonic()
    status, rc = "failed", None
    want = benv.parse_cores(cores) if cores else set()
    preexec = (lambda: os.sched_setaffinity(0, want)) if want else None
    with (rd / log_rel).open("w") as log:
        log.write(f"# {step.id} {step.title}\n# {utc_now()} cmd: {cmd}\n# run env: {json.dumps(renv, default=str)}\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=str(cfg.board_dir), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             start_new_session=True, pass_fds=(lock_fd,) if lock_fd else (),
                             preexec_fn=preexec,
                             env=dict(os.environ, PYTHONUNBUFFERED="1",
                                      **{bc.RUN_ENV_VAR: json.dumps(renv, default=str)}))
        aff = benv.affinity_of(p.pid) if want else ""
        rec["affinity"] = {"requested": benv.cores_str(want), "readback": aff,
                           "ok": True if not want else (None if not aff and p.poll() is not None
                                                        else aff == benv.cores_str(want))}

        def pump():
            for line in p.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            log.flush()
        th = threading.Thread(target=pump, daemon=True)
        th.start()
        try:
            if want and rec["affinity"]["ok"] is False and p.poll() is None and strict_pin:
                say(f"  PINNING FAILED for {step.id}: {rec['affinity']} — stopping the step")
                _kill_group(p, say)
                rc, status = p.returncode, "failed"
                rec["note"] = f"pinning failed {rec['affinity']}"
            else:
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
    temp1 = benv.read_die_temp(env_paths.iio_root, env_paths.hwmon_root) if env_paths else {}
    rec.update(finished_utc=utc_now(), duration_s=round(dur, 3), exit_code=rc, die_temp_end=temp1)
    _env_log(rd, {"step": step.id, "start_utc": rec["started_utc"], "end_utc": rec["finished_utc"],
                  "die_temp_start": temp0, "die_temp_end": temp1, "affinity": rec["affinity"],
                  "paper_grade": renv.get("paper_grade"), "session_index": renv.get("session_index")})
    if temp0 or temp1:
        say(f"  die temperature {step.id}: start {temp0.get('max_c', 'n/a')} °C, end "
            f"{temp1.get('max_c', 'n/a')} °C ({temp1.get('source', '')}); pinned "
            f"{rec['affinity']['readback'] or '-'}")
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


# ---- deploy info / bitstreams ----------------------------------------------------------------------
DEPLOY_BIT_KEYS = ("bit", "hwh", "bit_sha256", "hwh_sha256", "build_id", "vivado_version",
                   "bit_clock_mhz", "bit_clock_requested_mhz", "timing_wns_ns")


def deploy_info_for(backend: str, bit_arg: str | None, board_dir: Path, data_dir: Path,
                    info: dict | None = None) -> tuple[dict, Path]:
    """(deploy info, bit path) for ONE bitstream. Hardware: DEPLOY_INFO.json (required; with a
    "bits" list the matching entry's fields override the top level). Dry run: a laptop equivalent
    from the bitstream's summary.json + git (clearly labelled)."""
    info = bc.deploy_info() if info is None else info
    if backend == "pynq":
        rel = bit_arg or info.get("bit")
        if not rel:
            return info, board_dir / "bit" / "MISSING.bit"
        bit = Path(rel)
        bit = bit if bit.is_absolute() else (board_dir / bit)
        d = dict(info)
        for e in info.get("bits", []) or []:
            eb = Path(e.get("bit", ""))
            eb = eb if eb.is_absolute() else board_dir / eb
            if eb.resolve() == bit.resolve():
                d.update({k: e[k] for k in DEPLOY_BIT_KEYS if k in e})
        return d, bit
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
             bit_clock_requested_mhz=s.get("pl_clk0_mhz_requested"),
             timing_wns_ns=s.get("wns_ns"))
    if (data_dir / "PACKAGE.json").is_file():
        p = json.loads((data_dir / "PACKAGE.json").read_text())
        d.update(data_git_commit=p.get("git_commit"), data_git_dirty=p.get("git_dirty"))
    return d, bit


def bit_candidates(a, board_dir: Path, info: dict) -> list[str]:
    """Bitstreams, highest closed clock first. Explicit --bit = that one only."""
    if a.bit:
        return [a.bit]
    if a.backend == "pynq":
        bits = [e["bit"] for e in (info.get("bits") or []) if e.get("bit")]
        return bits or ([info["bit"]] if info.get("bit") else [])
    if a.fallback_bits:
        return list(a.fallback_bits)
    out = []
    if bc.IN_REPO:
        for name in ("gos_300", "gos_250"):
            p = board_dir.parent / "vivado" / "out" / name / f"{name}.bit"
            if p.is_file():
                out.append(str(p))
    return out or ([info["bit"]] if info.get("bit") else [])


# ---- main --------------------------------------------------------------------------------------
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
    ap.add_argument("--idle-ref-s", type=float, default=0.0,
                    help="no-overlay idle reference: INA260 SOM-rail idle power sampled for this many seconds "
                         "right after the environment pre-flight and BEFORE any bitstream is loaded (run it "
                         "first after a fresh boot: a loaded PL configuration cannot be unloaded) -> "
                         "hw_nooverlay_idle_{samples,phases}.csv, paper-grade environment rules as every step")
    ap.add_argument("--session-index", type=int, choices=(1, 2, 3), default=1,
                    help="repeatability campaign index (rows tagged; K >= 2 -> <results>/rep<K>/)")
    ap.add_argument("--budget-min", type=float, default=None, help="time budget for this invocation")
    ap.add_argument("--step-timeout-min", type=float, default=None,
                    help="hard per-step timeout (default max(5, 3 x estimate + 2) min)")
    ap.add_argument("--quick", action="store_true",
                    help="first pass: 200 images per step, B3 --n 200, CPU --quick, B1/B2 30 s windows")
    ap.add_argument("--plan", action="store_true", help="pre-flight + print the plan, run nothing")
    ap.add_argument("--steps", nargs="+", default=None, help="only these step ids (e.g. s2.A1)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-non-paper-grade", action="store_true",
                    help="environment pre-flight FAILs (governor, pinning, package manager) become "
                    "warnings; every row is marked paper_grade=False")
    ap.add_argument("--bit", default=None, help="one bitstream only (no clock fallback); hw default: "
                    "DEPLOY_INFO bits (300 then 250 MHz build)")
    ap.add_argument("--fallback-bits", nargs="+", default=None,
                    help="dry run: candidate bitstreams, highest first (default gos_300 + gos_250)")
    ap.add_argument("--dryrun-fail-smoke-mhz", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--data-dir", default=str(bc.DEFAULT_DATA_DIR))
    ap.add_argument("--results-dir", default=None,
                    help="default results/ (hw) or results/dryrun/ (model)")
    ap.add_argument("--window-s", type=float, default=None,
                    help="B1/B2 INA260 phase length (default 60; --quick 30; model 2)")
    ap.add_argument("--gap-s", type=float, default=None, help="B2 gap (default 10; model 1)")
    ap.add_argument("--fcal-s", type=float, default=None,
                    help="PL clock calibration wall time per session (default 8; --quick 4; model 1)")
    ap.add_argument("--b2-images", type=int, default=100)
    ap.add_argument("--power-repeats", type=int, default=3, help="B1 INA260 protocol repeats")
    ap.add_argument("--b2-power-repeats", type=int, default=3,
                    help="B2 INA260 idle/accel/idle repeats per clock")
    ap.add_argument("--power-rate-hz", type=float, default=10.0, help="INA260 sample rate request")
    ap.add_argument("--write-mode", choices=("elem", "mmio", "slice"), default="elem")
    ap.add_argument("--host-path", choices=("safe", "fast"), default="safe",
                    help="host path of A1-A4/B1/B2 (fast only after s1.fast passed; B3 interleaves both)")
    ap.add_argument("--fast-store", choices=("block", "words32"), default="block")
    ap.add_argument("--no-power-control", action="store_true",
                    help="B1 without the control phases (4R+1 instead of 6R+1 phases)")
    ap.add_argument("--no-b3-cpu", action="store_true", help="B3 without the interleaved CPU condition")
    ap.add_argument("--meas-cores", default=DEFAULT_MEAS_CORES,
                    help="cores every step process is pinned to (default 3)")
    ap.add_argument("--cpu1-cores", default=DEFAULT_CPU1_CORES,
                    help="CPU baseline 1-thread workers (default 3)")
    ap.add_argument("--cpun-cores", default=DEFAULT_CPUN_CORES,
                    help="CPU baseline step + multi-thread workers (default 0-3)")
    ap.add_argument("--cpu-freq-khz", type=int, default=None,
                    help="fixed CPU frequency (default: the highest available)")
    ap.add_argument("--sysfs-root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--proc-root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--iio-root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--hwmon-root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--board-id", default=None)
    ap.add_argument("--clock-tol-mhz", type=float, default=bc.CLOCK_TOL_MHZ)
    ap.add_argument("--min-free-mb", type=float, default=1000.0)
    ap.add_argument("--no-bringup-check", action="store_true",
                    help="run sessions 2/3 although Session 1 is not recorded OK in this state")
    return ap


def results_dir_for(source: str, results_dir, index: int) -> Path:
    root = bc.resolve_out_dir(source, results_dir).resolve()
    if index > 1:
        return bc.resolve_out_dir(source, str(root / f"rep{index}")).resolve()
    return root


def main(argv=None, open_backend=open_backend_default, verify_data=bc.verify_manifest,
         steps_hook=None) -> int:
    a = make_parser().parse_args(argv)
    board_dir = bc.BOARD_DIR
    data_dir = Path(a.data_dir).resolve()
    source = bc.SOURCE_DRYRUN if a.backend == "model" else bc.SOURCE_HW
    rd = results_dir_for(source, a.results_dir, a.session_index)
    dry = a.backend == "model"
    window_s = a.window_s if a.window_s is not None else (2.0 if dry else (30.0 if a.quick else 60.0))
    gap_s = a.gap_s if a.gap_s is not None else (1.0 if dry else 10.0)

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
    holder = {}
    try:
        return _main(a, board_dir, data_dir, rd, window_s, gap_s, say, lock_f.fileno(),
                     open_backend, verify_data, steps_hook, sess_log, holder)
    except (KeyboardInterrupt, Interrupted) as e:
        say(f"[session] INTERRUPTED ({type(e).__name__} {e}); state saved; rerun the same command to resume")
        return 130
    finally:
        env_prov = holder.get("env") or {}
        orig = (env_prov.get("orchestrator_pin") or {}).get("original")
        if orig:
            benv.pin(0, orig)                    # the orchestrator's own affinity back
        ctl = holder.get("cpufreq")
        if ctl is not None and ctl.applied:
            errs = ctl.restore()
            say(f"[session] {'DRY RUN (fake tree) ' if ctl.fake else ''}cpufreq restored to the saved "
                f"governor/min/max" + (f" — ERRORS {errs}" if errs else ""))
        for s, h in old.items():
            signal.signal(s, h)
        slog.close()
        lock_f.close()


def run_idle_ref(a, rd: Path, envp: dict, say) -> list:
    """No-overlay idle reference (before any bitstream is loaded in this boot): power_log.sample_only
    with the same environment rules as a step (paper_grade from the environment pre-flight)."""
    import power_log as pl
    try:
        sensor, plog = pl.discover("auto", allow_hw=True, allow_mock=False)
    except pl.SensorUnavailable as e:
        say(f"[session] idle reference: no sensor ({e}); skipped")
        return []
    os.environ[bc.RUN_ENV_VAR] = json.dumps({
        "session_index": a.session_index, "paper_grade": envp["paper_grade"],
        "cpu_governor": envp["cpu_governor"], "cpu_freq_khz": envp["cpu_freq_khz"],
        "note": "no-overlay idle reference (before any bitstream load)"}, default=str)
    try:
        ctx = pl.SensorOnlyContext(bc.SOURCE_HW, rd, getattr(a, "board_id", None), a.allow_dirty)
        say(f"[session] no-overlay idle reference: {a.idle_ref_s:g} s, {pl.LABEL}, {sensor.describe()}")
        pl.sample_only(ctx, a.idle_ref_s, sensor, getattr(a, "power_rate_hz", 10.0), "hw_nooverlay_idle", "")
    finally:
        os.environ.pop(bc.RUN_ENV_VAR, None)
    return [rd / "hw_nooverlay_idle_samples.csv", rd / "hw_nooverlay_idle_phases.csv"]


def _smoke_runner(a, cfg0: Config, board_dir, data_dir, rd, open_backend, verify_data, say,
                  info, meas_cores):
    """run_smoke(entry) for clock_fallback.choose: pre-flight of that bitstream + core smoke."""
    def run(entry: dict) -> bool:
        d, bit = deploy_info_for(a.backend, str(entry["bit"]), board_dir, data_dir, info)
        cfg = replace(cfg0, deploy=d, bit=bit)
        try:
            prov = preflight(cfg, open_backend, verify_data, lambda m: say("    " + m))
            entry["clock_readback_mhz"] = prov.get("clock_readback_mhz")
        except PreflightError as e:
            say(f"    [fallback] pre-flight of {bit.name} failed: {e}")
            entry["detail"] = "pre-flight failed: " + " | ".join(str(e).split("\n")[1:] or [str(e)]).strip()
            return False
        if (dry := cfg.dry) and a.dryrun_fail_smoke_mhz is not None and \
                abs(float(entry["clock_mhz"]) - a.dryrun_fail_smoke_mhz) <= 1.0:
            say(f"    [fallback] DRY RUN: SIMULATED smoke failure at {entry['clock_mhz']} MHz "
                "(--dryrun-fail-smoke-mhz)")
            return False
        dev = ["--backend", a.backend, "--data-dir", str(data_dir)]
        dev += ["--clock-mhz", f"{cfg.closed_mhz:.6f}"] if dry else ["--bit", str(bit)]
        cmd = [cfg.python, str(board_dir / "test_core_smoke.py"), *dev,
               "--max-fclk0", f"{cfg.closed_mhz + cfg.clock_tol:.6f}"]
        logp = rd / "logs" / f"fallback_smoke_{clock_tag(cfg.closed_mhz)}.log"
        logp.parent.mkdir(exist_ok=True)
        cores = benv.parse_cores(meas_cores)
        with logp.open("w") as log:
            p = subprocess.run(cmd, cwd=str(board_dir), stdout=log, stderr=subprocess.STDOUT,
                               timeout=600, preexec_fn=(lambda: os.sched_setaffinity(0, cores)))
        say(f"    [fallback] smoke {bit.name}: exit {p.returncode} (log {logp.relative_to(rd)})")
        return p.returncode == 0
    return run


def _main(a, board_dir, data_dir, rd, window_s, gap_s, say, lock_fd, open_backend, verify_data,
          steps_hook, sess_log, holder) -> int:
    dry = a.backend == "model"
    tag = "DRY RUN " if dry else ""
    t_start = time.monotonic()
    info = bc.deploy_info()
    say(f"[session] {tag}sessions {a.sessions} session_index {a.session_index} backend={a.backend} "
        f"results={rd} budget={a.budget_min} min quick={a.quick} {utc_now()}")
    if dry:
        say("[session] DRY RUN: every number is model output (source=dryrun_model), NOT a hardware "
            "measurement; outputs only under results/dryrun/.")
    state_path = rd / STATE_NAME
    old_state = load_state(state_path)
    timing = (old_state or {}).get("timing", {})

    # -- environment pre-flight (before anything touches the board) -----------------------------
    if dry and not a.sysfs_root:
        paths = dryrun_env_paths(rd)
    else:
        paths = EnvPaths(sysfs_root=a.sysfs_root or benv.SYSFS_CPU, proc_root=a.proc_root or benv.PROC,
                         iio_root=a.iio_root or benv.IIO, hwmon_root=a.hwmon_root or benv.HWMON,
                         fake=False)
    try:
        envp, ctl = env_preflight(paths, dry=dry, plan=a.plan, allow_non_paper=a.allow_non_paper_grade,
                                  allow_dirty=a.allow_dirty, meas_cores=a.meas_cores,
                                  cpu1_cores=a.cpu1_cores, cpun_cores=a.cpun_cores,
                                  cpu_freq_khz=a.cpu_freq_khz, say=say)
        holder["cpufreq"] = ctl
        holder["env"] = envp
    except PreflightError as e:
        say(str(e))
        say("[session] nothing was run.")
        return 3

    idle_ref_files = []
    if a.idle_ref_s > 0 and not a.plan and not dry:
        idle_ref_files = run_idle_ref(a, rd, envp, say)

    # -- bitstream: recorded choice / clock fallback / single --------------------------------
    cands = bit_candidates(a, board_dir, info)
    if not cands:
        raise SystemExit("no bitstream known (DEPLOY_INFO bits / --bit)")
    entries = []
    for b in cands:
        d, bp = deploy_info_for(a.backend, b, board_dir, data_dir, info)
        entries.append({"bit": bp, "clock_mhz": d.get("bit_clock_mhz"), "deploy": d})
    entries.sort(key=lambda e: -(float(e["clock_mhz"]) if e["clock_mhz"] is not None else 0.0))
    rec_choice = None if (old_state is None or a.fresh) else old_state["provenance"].get("bit_choice")
    chosen = entries[0]
    choice = {"ok": True, "bit": str(chosen["bit"]), "clock_mhz": chosen["clock_mhz"],
              "fell_back": False, "attempts": [],
              "reason": "single bitstream" if len(entries) == 1 else "primary (highest clock)"}
    if rec_choice and rec_choice.get("bit"):
        m = [e for e in entries if str(e["bit"]) == str(rec_choice["bit"])]
        if m:
            chosen, choice = m[0], rec_choice
            say(f"[session] bitstream: recorded choice {chosen['bit']} ({rec_choice.get('reason')})")
    elif len(entries) > 1 and 1 in a.sessions and not a.plan:
        cfg0 = Config(backend=a.backend, board_dir=board_dir, results_dir=rd, data_dir=data_dir,
                      deploy=entries[0]["deploy"], bit=entries[0]["bit"], allow_dirty=a.allow_dirty,
                      clock_tol=a.clock_tol_mhz, min_free_mb=a.min_free_mb)
        try:
            import clock_fallback as cf
        except ImportError:
            cf = None
            choice["reason"] = "clock_fallback.py absent: primary (highest clock) bitstream, no fallback"
            say(f"[session] {choice['reason']}")
        if cf is not None:
            say(f"[session] {tag}clock fallback over {[e['clock_mhz'] for e in entries]} MHz "
                "(clock_fallback.choose: pre-flight + core smoke per bitstream)")
            ctx = SimpleNamespace(say=say, dry=dry, cfg=cfg0, results_dir=rd, board_dir=board_dir)
            res = cf.choose(ctx, [{"bit": str(e["bit"]), "clock_mhz": float(e["clock_mhz"])}
                                  for e in entries],
                            _smoke_runner(a, cfg0, board_dir, data_dir, rd, open_backend, verify_data,
                                          say, info, envp["meas_cores"]))
            if not res.get("ok", res.get("bit") is not None) or not res.get("bit"):
                say(f"[session] CLOCK FALLBACK FAILED: {res.get('reason')} — nothing else run")
                return 3
            chosen = next(e for e in entries if str(e["bit"]) == str(res["bit"]))
            choice = res
    cfg = Config(backend=a.backend, board_dir=board_dir, results_dir=rd, data_dir=data_dir,
                 deploy=chosen["deploy"], bit=chosen["bit"], allow_dirty=a.allow_dirty,
                 clock_tol=a.clock_tol_mhz, min_free_mb=a.min_free_mb)
    try:
        prov = preflight(cfg, open_backend, verify_data, say)
    except PreflightError as e:
        say(str(e))
        say("[session] nothing was run.")
        return 3
    prov.update(session_index=a.session_index, paper_grade=envp["paper_grade"],
                cpu_freq_khz=envp["cpu_freq_khz"], bit_choice=choice, env=envp)

    if old_state is not None and not a.fresh:
        diff = provenance_diff(old_state["provenance"], prov, cfg.clock_tol)
        if diff:
            say("[session] REFUSED to resume: provenance differs from " + str(state_path) + ":\n  - "
                + "\n  - ".join(diff) + "\n  Results of one state must come from one bitstream, "
                "clock, data package, scripts commit, session index and environment. Use --fresh "
                "(old outputs are archived).")
            return 4
        state = old_state
        state["env_last"] = envp
        say(f"[session] resuming state {state_path.name} (created {state['created_utc']}); "
            "provenance identical")
    else:
        why = "fresh" if a.fresh else "pre_state"
        if a.plan:
            say(f"[session] --plan: would archive existing results ({why}) and start a new state")
        elif state_path.exists() or any(rd.glob("hw_*")) or (rd / ".staging").exists():
            keep = [sess_log, *(rd / "logs").glob("fallback_smoke_*.log"), *idle_ref_files]
            ar = archive_everything(rd, why, keep=keep)
            if ar.moved:
                say(f"[session] {why}: archived {len(ar.moved)} entries -> {ar.root.relative_to(rd)}")
        state = new_state(prov, timing)
    state.setdefault("order_seed", stats.new_seed())
    if not a.plan:
        state["invocations"].append({"utc": utc_now(), "args": vars(a),
                                     "session_log": str(sess_log.relative_to(rd)),
                                     "preflight": prov})
        save_state(state_path, state)
        if len(entries) > 1 or choice.get("attempts"):
            (rd / CHOICE_FILE).write_text(json.dumps(choice, indent=1, default=str) + "\n")
    say(f"[session] bitstream for every step: {cfg.bit} (closed {cfg.closed_mhz} MHz; "
        f"{choice.get('reason')})")

    seed = int(state["order_seed"])
    steps = build_steps(cfg, a.sessions, a.quick, window_s, gap_s, a.write_mode, a.b2_images,
                        a.board_id, a.power_repeats, a.b2_power_repeats, a.power_rate_hz,
                        a.host_path, a.fast_store, not a.no_power_control, order_seed=seed,
                        fcal_s=a.fcal_s, cpu1_cores=envp["cpu1_cores"], cpun_cores=envp["cpun_cores"],
                        b3_cpu=not a.no_b3_cpu)
    steps = merge_extra_steps(steps, extra_opts(cfg, a, a.sessions, window_s, gap_s, seed),
                              a.sessions, say)
    steps = prioritize(steps)
    say(f"[session] host path for A1-A4/B1/B2: {a.host_path}"
        + (f" (store {a.fast_store}; requires s1.fast OK)" if a.host_path == "fast" else
           " (default; B3 adds the fast path, interleaved, if s1.fast passed)"))
    if 3 in a.sessions:
        say(f"[session] Session 3 power: {POWER_LABEL} via {POWER_HOOK_SCRIPT} (B1 per net, B2 per "
            "clock); the INA260 is the only power source")
    say(f"[session] priority: bring-up, f_meas calibration, {', '.join(GROUP_ORDER)}; seed {seed}")
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

    base_env = {"session_index": a.session_index, "paper_grade": envp["paper_grade"],
                "cpu_governor": envp["cpu_governor"], "cpu_freq_khz": envp["cpu_freq_khz"],
                "f_req_mhz": cfg.deploy.get("bit_clock_requested_mhz") or "",
                "note": ("paper-grade environment" if envp["paper_grade"] else
                         "NOT paper-grade: " + "; ".join(
                             (["dry run"] if dry else []) + (["--allow-dirty"] if a.allow_dirty else [])
                             + (["--allow-non-paper-grade"] if a.allow_non_paper_grade else [])
                             + envp["warnings"])[:400])}
    budget_s = None if a.budget_min is None else 60.0 * a.budget_min
    report, any_fail, blocked, incomplete = [], False, False, False
    say(f"[session] plan ({len(steps)} steps, priority order):")
    for st0 in steps:
        st = resolve_step(st0, state)
        est, src = estimate(st, state, a.backend)
        done, why = verify_step(st, state["steps"].get(st.id), rd)
        say(f"  {st.id:16} [{st.group or '-':9}] est {est / 60:7.1f} min ({src}); "
            f"{'DONE, skip' if done else 'run'} ({why}) — {st.title}")
    if a.plan:
        return 0

    for st0 in steps:
        st = resolve_step(st0, state)
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
        unmet = [q for q in st.requires if state["steps"].get(q, {}).get("status") != "ok"]
        if unmet and not a.no_bringup_check:
            why = f"prerequisite {unmet} not recorded OK in this state"
            say(f"\n[session] BLOCKED {st.id}: {why}"
                + ("; rerun with --host-path safe" if st.required else " (informational step)"))
            state["steps"].setdefault(st.id, {}).update(status="blocked", blocked_utc=utc_now(),
                                                        blocked_reason=why)
            save_state(state_path, state)
            report.append((st.id, "BLOCKED", None, est, why))
            if st.required:
                any_fail = True
            continue
        elapsed = time.monotonic() - t_start
        remaining = None if budget_s is None else budget_s - elapsed
        if remaining is not None and est > remaining:
            say(f"\n[session] DEFER {st.id} [{st.group}]: estimate {est / 60:.1f} min ({src}) > "
                f"remaining budget {remaining / 60:.1f} min")
            state["steps"].setdefault(st.id, {}).update(
                status="deferred", deferred_utc=utc_now(), deferred_estimate_s=round(est, 1))
            save_state(state_path, state)
            report.append((st.id, "DEFERRED", None, est, f"est > remaining {remaining / 60:.1f} min"))
            incomplete = True
            continue
        if a.step_timeout_min is not None:
            timeout = 60.0 * a.step_timeout_min
        elif st.timeout_s is not None:
            timeout = st.timeout_s
        else:
            timeout = max(300.0, 3.0 * est + 120.0)
        if remaining is not None:
            timeout = min(timeout, max(remaining, 1.0))
        say(f"\n[session] ===== {tag}{st.id}: {st.title} | est {est / 60:.1f} min ({src}) | "
            f"timeout {timeout / 60:.1f} min | {utc_now()} =====")
        if rec and rec.get("status") in ("running", "interrupted", "timeout"):
            say(f"  previous attempt {rec.get('status')} ({rec.get('started_utc')}): rerun from scratch")
        renv = dict(base_env, **fmeas_for(state, rd, st.session))
        cores = envp["cpun_cores"] if st.cores == "cpun" else envp["meas_cores"]
        r = run_step(st, cfg, state, state_path, timeout, say, lock_fd, run_env=renv, cores=cores,
                     env_paths=paths, strict_pin=envp["paper_grade"])
        say(f"[session] ===== {st.id}: {r['status'].upper()} (exit {r.get('exit_code')}, "
            f"{r['duration_s']:.1f} s; {len(r['outputs'])} outputs) =====")
        note = r.get("note", "")
        if st.group == "fcal" and r["status"] == "ok":
            fm = fmeas_for(state, rd, st.session)
            if fm.get("f_meas_step") == st.id:
                note = (f"cross-check f_meas {fm['f_meas_mhz']:.6f} vs read-back "
                        f"{fm['f_meas_readback_mhz']:.6f} MHz: {fm['f_meas_rel_diff_pct']} %"
                        + (" — FLAG: > 0.1 %" if fm["f_meas_flag"] and not dry else ""))
                say(f"  [fcal] {note} (µs values use the read-back clock)")
        report.append((st.id, r["status"], r["duration_s"], est, note))
        if r["status"] != "ok":
            if st.required:
                any_fail = True
                if st.session == 1:
                    blocked = True
            else:
                say(f"  ({st.id} is informational: not counted as a failure)")

    say(f"\n[session] {tag}SUMMARY ({(time.monotonic() - t_start) / 60:.1f} min, results {rd}, "
        f"session_index {a.session_index}, paper_grade {envp['paper_grade']})")
    say(f"  {'step':16} {'status':16} {'dur_s':>9} {'est_s':>9}  note")
    for sid, status, dur, est, note in report:
        say(f"  {sid:16} {status:16} {('' if dur is None else f'{dur:.1f}'):>9} {est:9.0f}  {note}")
    deferred = [x[0] for x in report if x[1] == "DEFERRED"]
    if deferred:
        say(f"  deferred (rerun the same command to continue): {deferred}")
    if any_fail:
        return 1
    return 5 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
