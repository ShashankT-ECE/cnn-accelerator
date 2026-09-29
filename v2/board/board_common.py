"""Shared helpers for the V2 board scripts (numpy only; pynq / v2/model imported lazily).

  load_package(data_dir, net)   verify MANIFEST.json SHA256s, load the data package (README)
  final_dequant / predict        PS float32 dequant + argmax (DECISIONS D2, legacy expression)
  add_common_args / open_device  --backend {pynq,model}, bitstream, data dir, timeouts
  RunContext                     provenance + results-row metadata + output-dir rules
  run_env / env_meta             measurement environment handed down by run_sessions.py
                                 ($GOS_RUN_ENV: session_index, paper_grade, governor/frequency,
                                 pinning, die temperature at step start, measured PL clock f_meas)
  RunContext.f_used / clock_cols the clock used for every µs value: f_meas (PL clock calibration,
                                 exp_fclk_cal.py) when it was measured at the current read-back
                                 clock, else the read-back clock (labelled)
  write_csv                      results CSV (EXPERIMENTS.md "CSV rule" columns first)

Layouts: on the laptop this file lives in v2/board/ (results -> v2/results/, dry runs ->
v2/results/dryrun/); on the board deploy.sh mirrors v2/board/ to ~/gos/ (results -> ~/gos/results/,
DEPLOY_INFO.json -> ~/gos/DEPLOY_INFO.json, bitstream -> ~/gos/bit/).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BOARD_DIR = Path(__file__).resolve().parent
IN_REPO = (BOARD_DIR.parent / "model" / "gos_pack.py").is_file()
REPO_ROOT = BOARD_DIR.parents[1] if IN_REPO else None
DEFAULT_DATA_DIR = BOARD_DIR / "data"
RESULTS_ROOT = (BOARD_DIR.parent / "results") if IN_REPO else (BOARD_DIR / "results")
DEPLOY_INFO_PATH = BOARD_DIR / "DEPLOY_INFO.json"
IMPL_SUMMARY = (BOARD_DIR.parent / "vivado" / "out" / "gos_200" / "summary.json") if IN_REPO else None

NETS = ("lenet5", "cifar10")                       # = v2/model/net_config.NETS
# Accuracies of record (model, full 10,000-image test sets; DECISIONS D3, cifar10 = r2).
EXPECTED_GOLDEN_CORRECT = {"lenet5": 9879, "cifar10": 7852}

META_COLUMNS = (
    "timestamp", "git_commit", "git_dirty", "vivado_version", "bitstream_sha256",
    "board_id", "net", "layer", "clock_mhz", "source", "duration_s", "num_inferences",
)
EXTRA_META = ("build_id_hw", "board_hostname", "clock_source", "scripts_commit",
              "data_manifest_sha256", "data_git_commit", "data_git_dirty", "backend",
              # measurement environment (run_sessions.py pre-flight, $GOS_RUN_ENV):
              "session_index", "paper_grade", "env_step", "cpu_governor", "cpu_freq_khz",
              "cpu_affinity", "die_temp_start_c", "env_note")
RUN_ENV_VAR = "GOS_RUN_ENV"
CLOCK_COLS = ("f_readback_mhz", "f_meas_mhz", "f_meas_ci_lo_mhz", "f_meas_ci_hi_mhz", "f_used_mhz",
              "f_used_source")
SOURCE_HW = "hw"
SOURCE_DRYRUN = "dryrun_model"

PACKAGE_FILES = ("inputs_act.npz", "labels.npy", "golden_logits.npy", "golden_logits_f32.npy",
                 "golden_pred.npy", "wgt.npy", "qparam.npy", "desc.npy", "net.json",
                 "final_dequant.json", "quant_params.npz", "hw_requant.npz", "fp32_params.npz",
                 "model_cycles.json", "rtl_cycles.csv", "rtl_network.csv", "refuse_test.json")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---- data package -------------------------------------------------------------------------
class ManifestError(RuntimeError):
    pass


def verify_manifest(net_dir: Path) -> dict:
    """Check every file listed in MANIFEST.json (SHA256 + size). Returns the manifest."""
    net_dir = Path(net_dir)
    mpath = net_dir / "MANIFEST.json"
    if not mpath.is_file():
        raise ManifestError(f"{mpath} missing (run make_board_data.py on the laptop, then deploy)")
    man = json.loads(mpath.read_text())
    missing = [f for f in PACKAGE_FILES if f not in man["files"]]
    if missing:
        raise ManifestError(f"{mpath}: files not in manifest: {missing}")
    for name, meta in man["files"].items():
        p = net_dir / name
        if not p.is_file():
            raise ManifestError(f"{p} missing")
        if p.stat().st_size != meta["bytes"]:
            raise ManifestError(f"{p}: size {p.stat().st_size} != manifest {meta['bytes']}")
        if sha256_file(p) != meta["sha256"]:
            raise ManifestError(f"{p}: SHA256 differs from MANIFEST.json")
    return man


@dataclass
class Package:
    net_dir: Path
    manifest: dict
    manifest_sha256: str
    net: dict
    dequant: dict
    model_cycles: dict
    wgt: np.ndarray
    qparam: np.ndarray
    desc: np.ndarray
    labels: np.ndarray
    golden_logits: np.ndarray
    golden_pred: np.ndarray
    _inputs: object = None

    @property
    def name(self) -> str:
        return self.net["net"]

    @property
    def n(self) -> int:
        return int(self.labels.shape[0])

    def _npz(self):
        if self._inputs is None:
            self._inputs = np.load(self.net_dir / "inputs_act.npz")
        return self._inputs

    @property
    def x_act(self) -> np.ndarray:
        if not hasattr(self, "_x_act"):
            self._x_act = self._npz()["x"]
        return self._x_act

    @property
    def x_nchw(self) -> np.ndarray:
        return self._npz()["x_nchw"]

    @property
    def x_f32(self) -> np.ndarray:
        return self._npz()["x_f32"]

    def rtl_cycles_path(self) -> Path:
        return self.net_dir / "rtl_cycles.csv"


def load_package(data_dir, net: str, verify: bool = True) -> Package:
    net_dir = Path(data_dir) / net
    man = verify_manifest(net_dir) if verify else json.loads((net_dir / "MANIFEST.json").read_text())
    rd = lambda n: json.loads((net_dir / n).read_text())  # noqa: E731
    return Package(
        net_dir=net_dir, manifest=man, manifest_sha256=sha256_file(net_dir / "MANIFEST.json"),
        net=rd("net.json"), dequant=rd("final_dequant.json"), model_cycles=rd("model_cycles.json"),
        wgt=np.load(net_dir / "wgt.npy"), qparam=np.load(net_dir / "qparam.npy"),
        desc=np.load(net_dir / "desc.npy"), labels=np.load(net_dir / "labels.npy"),
        golden_logits=np.load(net_dir / "golden_logits.npy"),
        golden_pred=np.load(net_dir / "golden_pred.npy"))


# ---- PS final layer (D2) --------------------------------------------------------------------
def final_dequant(v_int32, dequant: dict) -> np.ndarray:
    """Reference float32 logits from raw INT32 v (N, OC) or (OC,).
    Same expression/dtypes as v2/model/final_layer.logits_from_raw:
    (v int64 * (S_a*S_w) float64) -> float32."""
    v = np.asarray(v_int32)
    assert v.dtype == np.int32, v.dtype
    single = v.ndim == 1
    v2 = v.reshape(1, -1) if single else v
    scale = (np.float64(dequant["S_a"]) * np.asarray(dequant["S_w"], dtype=np.float64)).reshape(1, -1)
    out = (v2.astype(np.int64) * scale).astype(np.float32)
    return out[0] if single else out


def predict(v_int32, dequant: dict):
    return final_dequant(v_int32, dequant).argmax(-1)


# ---- provenance ----------------------------------------------------------------------------
def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def git_state() -> tuple[str, bool]:
    """(commit, dirty) of the repo on the laptop; generated outputs excluded like
    v2/model/common.git_dirty (v2/results, v2/vectors/MANIFEST.json)."""
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--", ".", ":!v2/results",
                      ":!v2/vectors/MANIFEST.json"))
    return commit, dirty


def deploy_info() -> dict:
    """DEPLOY_INFO.json on the board; on the laptop a git-derived equivalent."""
    if DEPLOY_INFO_PATH.is_file():
        d = json.loads(DEPLOY_INFO_PATH.read_text())
        d["origin"] = "DEPLOY_INFO.json"
        return d
    d = {"origin": "none"}
    if IN_REPO:
        try:
            c, dirty = git_state()
            d.update(commit=c, dirty=dirty, origin="git (laptop)")
        except Exception as e:  # noqa: BLE001
            d.update(commit="", dirty=True, origin=f"git failed: {e}")
        if IMPL_SUMMARY and IMPL_SUMMARY.is_file():
            s = json.loads(IMPL_SUMMARY.read_text())
            d.update(vivado_version=s.get("vivado_version", ""), build_id=s.get("build_id", ""),
                     bit_clock_mhz=s.get("pl_clk0_mhz_actual"),
                     bit=str(IMPL_SUMMARY.parent / s.get("bit", "")))
    return d


def run_env() -> dict:
    """The measurement environment run_sessions.py passes to every step ($GOS_RUN_ENV, JSON).
    {} when a script is run by hand (then rows are not paper-grade: no environment pre-flight)."""
    try:
        d = json.loads(os.environ.get(RUN_ENV_VAR) or "{}")
        return d if isinstance(d, dict) else {}
    except ValueError:
        return {}


def env_meta(source: str, dirty: bool, env: dict | None = None) -> dict:
    """Row columns describing the measurement environment. paper_grade is True only for a
    hardware row from a clean tree whose step passed the environment pre-flight (governor fixed,
    process pinned, no package manager running) without --allow-non-paper-grade."""
    env = run_env() if env is None else env
    try:
        aff = ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        aff = ""
    pg = bool(env.get("paper_grade")) and source == SOURCE_HW and not dirty
    if not env:
        note = "no environment pre-flight (not run by run_sessions.py): not paper-grade"
    elif source != SOURCE_HW:
        note = "DRY RUN environment (fake sysfs), not paper data"
    else:
        note = env.get("note", "")
    return {"session_index": env.get("session_index", ""), "paper_grade": pg,
            "env_step": env.get("step_id", ""), "cpu_governor": env.get("cpu_governor", ""),
            "cpu_freq_khz": env.get("cpu_freq_khz", ""), "cpu_affinity": aff,
            "die_temp_start_c": env.get("die_temp_start_c", ""), "env_note": note}


def board_id_default() -> str:
    env = os.environ.get("GOS_BOARD_ID")
    if env:
        return env
    try:
        model = Path("/proc/device-tree/model").read_bytes().rstrip(b"\0").decode()
    except OSError:
        model = platform.machine()
    try:
        mid = Path("/etc/machine-id").read_text().strip()[:8]
    except OSError:
        mid = "nomachineid"
    return f"{model} machine-id:{mid}"


# ---- CLI / device ---------------------------------------------------------------------------
def add_common_args(ap: argparse.ArgumentParser, nets: bool = True):
    ap.add_argument("--backend", choices=("pynq", "model", "mock"), default="pynq",
                    help="pynq = KV260 hardware (default); model = laptop dry run (dryrun_model); "
                    "mock = laptop MockMMIO host-overhead backend (dryrun_model, golden logits)")
    ap.add_argument("--bit", default=None, help="<name>.bit (with .hwh beside it); default from "
                    "DEPLOY_INFO.json")
    ap.add_argument("--no-download", action="store_true",
                    help="attach to the already-loaded overlay (Overlay(download=False))")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir", default=None, help="default: results/ (hw) or results/dryrun/ (model)")
    if nets:
        ap.add_argument("--nets", nargs="+", default=list(NETS), choices=NETS)
    ap.add_argument("--limit", type=int, default=None, help="first N images only (default: all)")
    ap.add_argument("--timeout-s", type=float, default=1.0, help="per-job STATUS poll timeout")
    ap.add_argument("--write-mode", choices=("elem", "mmio", "slice"), default="elem",
                    help="MMIO write path for memories (slice = numpy block copy, validate first)")
    ap.add_argument("--host-path", choices=("safe", "fast"), default="safe",
                    help="safe (default) = per-word MMIO; fast = mapped numpy windows, vectorized "
                    "input write / LOGIT read (board: only after the s1.fast bring-up check passed)")
    ap.add_argument("--fast-store", choices=("block", "words32"), default="block",
                    help="fast path copy style: block = one contiguous numpy copy; words32 = "
                    "lo/hi strided copies (32-bit element stores)")
    ap.add_argument("--clock-mhz", type=float, default=None,
                    help="model backend only: nominal clock for µs conversion")
    ap.add_argument("--board-id", default=None)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="run although scripts/data are from a dirty tree (rows marked git_dirty=True)")
    return ap


def open_device(args):
    """GosDevice for args.backend; returns (dev, backend)."""
    import gos_driver as D
    info = deploy_info()
    if args.backend == "pynq":
        bit = args.bit or info.get("bit")
        if not bit:
            raise SystemExit("no --bit given and no DEPLOY_INFO.json 'bit'")
        bit = Path(bit)
        if not bit.is_absolute() and not bit.exists():
            bit = BOARD_DIR / bit
        try:
            be = D.PynqBackend(bit, download=not args.no_download)
        except D.GosError as e:
            raise SystemExit(f"ERROR: {e}") from None
    else:
        clk = args.clock_mhz or info.get("bit_clock_mhz") or 200.0
        if args.backend == "mock":
            be = D.MockMmioBackend(args.data_dir, clock_mhz=float(clk))
        else:
            be = D.ModelBackend(clock_mhz=float(clk))
    dev = D.GosDevice(be, timeout_s=args.timeout_s, write_mode=args.write_mode,
                      host_path=getattr(args, "host_path", "safe"),
                      fast_store=getattr(args, "fast_store", "block"))
    return dev, be


class RunContext:
    """Provenance of one script run + results-row metadata + output rules."""

    def __init__(self, args, dev, be, script: str):
        self.args, self.dev, self.be, self.script = args, dev, be, script
        self.source = be.source
        self.info = deploy_info()
        self.scripts_commit = self.info.get("commit", "")
        self.scripts_dirty = bool(self.info.get("dirty", True))
        self.hostname = socket.gethostname()
        self.board_id = args.board_id or (board_id_default() if be.kind == "pynq" else "")
        if be.kind == "pynq":
            self.bitstream_sha256 = be.bit_sha256
            dep_sha = self.info.get("bit_sha256")
            if dep_sha and dep_sha != be.bit_sha256:
                print(f"WARNING: loaded bitstream SHA256 {be.bit_sha256[:12]} != DEPLOY_INFO "
                      f"{dep_sha[:12]}", file=sys.stderr)
            self.vivado_version = self.info.get("vivado_version", "")
            self.clock_source = "pynq Clocks.fclk0_mhz readback"
        else:
            self.bitstream_sha256 = ""
            self.vivado_version = ""
            self.clock_source = "nominal (dry run, not measured)"
        self.packages: dict[str, Package] = {}
        self.out_dir = resolve_out_dir(be.source, args.out_dir)
        self.run_env = run_env()

    def f_used(self, readback_mhz=None) -> tuple[float, str]:
        """(clock in MHz for every µs value, its source). f_meas from the PL clock calibration
        (exp_fclk_cal.py, passed down in $GOS_RUN_ENV) if it was measured at this read-back clock
        (within CLOCK_TOL_MHZ), else the read-back clock itself."""
        rb = self.dev.fclk0_mhz() if readback_mhz is None else float(readback_mhz)
        e = self.run_env
        fm, fr = e.get("f_meas_mhz"), e.get("f_meas_readback_mhz")
        if fm not in (None, "") and fr not in (None, "") and abs(float(fr) - rb) <= CLOCK_TOL_MHZ:
            return float(fm), f"f_meas ({e.get('f_meas_source', 'calibration')}; {e.get('f_meas_step', '')})"
        return rb, "f_readback (no PL clock calibration at this clock)"

    def clock_cols(self, readback_mhz=None) -> dict:
        rb = self.dev.fclk0_mhz() if readback_mhz is None else float(readback_mhz)
        f, src = self.f_used(rb)
        e = self.run_env
        meas = src.startswith("f_meas")
        return {"f_readback_mhz": f"{rb:.6f}", "f_meas_mhz": f"{float(e['f_meas_mhz']):.6f}" if meas else "",
                "f_meas_ci_lo_mhz": e.get("f_meas_ci_lo_mhz", "") if meas else "",
                "f_meas_ci_hi_mhz": e.get("f_meas_ci_hi_mhz", "") if meas else "",
                "f_used_mhz": f"{f:.6f}", "f_used_source": src}

    def package(self, net: str) -> Package:
        if net not in self.packages:
            self.packages[net] = load_package(self.args.data_dir, net)
        return self.packages[net]

    @property
    def dirty(self) -> bool:
        data_dirty = any(bool(p.manifest.get("git_dirty", True)) for p in self.packages.values())
        return self.scripts_dirty or data_dirty

    def check_clean(self):
        """Hardware results must come from a clean, committed tree (v2/CLAUDE.md)."""
        for net in getattr(self.args, "nets", []):
            self.package(net)
        if self.source == SOURCE_HW and self.dirty and not self.args.allow_dirty:
            raise SystemExit("REFUSED: scripts or data package come from a dirty tree "
                             f"(scripts dirty={self.scripts_dirty}; data dirty="
                             f"{[p.manifest.get('git_dirty') for p in self.packages.values()]}). "
                             "Commit, rebuild data, redeploy — or pass --allow-dirty "
                             "(rows are then invalid for the paper).")
        if self.source == SOURCE_HW:
            for p in self.packages.values():
                if p.manifest.get("git_commit") != self.scripts_commit:
                    print(f"NOTE: data package {p.name} built at {p.manifest.get('git_commit', '')[:8]}"
                          f", scripts at {self.scripts_commit[:8]}", file=sys.stderr)

    def meta(self, net: str = "", layer: str = "", duration_s="", num_inferences="",
             clock_mhz=None) -> dict:
        pkg = self.packages.get(net)
        clk = self.dev.fclk0_mhz() if clock_mhz is None else clock_mhz
        return {
            "timestamp": utc_now(),
            "git_commit": self.scripts_commit,
            "git_dirty": self.dirty,
            "vivado_version": self.vivado_version,
            "bitstream_sha256": self.bitstream_sha256,
            "board_id": self.board_id,
            "net": net, "layer": layer,
            "clock_mhz": f"{clk:.6f}" if isinstance(clk, float) else clk,
            "source": self.source,
            "duration_s": duration_s if duration_s == "" else f"{float(duration_s):.3f}",
            "num_inferences": num_inferences,
            "build_id_hw": self.dev.build_id_hex,
            "board_hostname": self.hostname,
            "clock_source": self.clock_source,
            "scripts_commit": self.scripts_commit,
            "data_manifest_sha256": pkg.manifest_sha256 if pkg else "",
            "data_git_commit": pkg.manifest.get("git_commit", "") if pkg else "",
            "data_git_dirty": pkg.manifest.get("git_dirty", "") if pkg else "",
            "backend": self.be.kind,
            **env_meta(self.source, self.dirty, self.run_env),
        }

    def csv(self, name: str, rows: list[dict], fields: list[str]) -> Path:
        return write_csv(self.out_dir / name, rows, fields, self.source)

    def path(self, name: str) -> Path:
        p = self.out_dir / name
        check_output_path(p, self.source)
        return p

    def banner(self):
        print(f"[{self.script}] host path: {self.dev.host_path_desc}")
        em = env_meta(self.source, self.dirty, self.run_env)
        print(f"[{self.script}] environment: session_index={em['session_index'] or '-'} "
              f"paper_grade={em['paper_grade']} governor={em['cpu_governor'] or '-'} "
              f"cpu_freq_khz={em['cpu_freq_khz'] or '-'} affinity={em['cpu_affinity']} "
              f"die_temp_start_c={em['die_temp_start_c'] or '-'} {em['env_note']}")
        f, fsrc = self.f_used()
        print(f"[{self.script}] µs values use f = {f:.6f} MHz ({fsrc})")
        print(f"[{self.script}] backend={self.be.kind} source={self.source} "
              f"VERSION=0x{self.dev.version:08X} BUILD_ID={self.dev.build_id_hex} "
              f"clock={self.dev.fclk0_mhz():.3f} MHz ({self.clock_source}) "
              f"scripts={self.scripts_commit[:8]} dirty={self.scripts_dirty} out={self.out_dir}")
        if self.source == SOURCE_DRYRUN:
            print(f"[{self.script}] DRY RUN: every number below is model output "
                  "(gos_golden + gos_cycle_model), NOT a hardware measurement.")


def resolve_out_dir(source: str, out_dir) -> Path:
    if out_dir is None:
        out = RESULTS_ROOT / "dryrun" if source == SOURCE_DRYRUN else RESULTS_ROOT
    else:
        out = Path(out_dir)
    check_output_path(out / "x.csv", source)
    out.mkdir(parents=True, exist_ok=True)
    return out


def check_output_path(path: Path, source: str):
    """Dry-run output only under a 'dryrun' directory; hardware output never there."""
    parts = Path(os.path.abspath(path)).parent.parts
    in_dryrun = "dryrun" in parts
    if source == SOURCE_DRYRUN and not in_dryrun:
        raise SystemExit(f"REFUSED: model-backend (dryrun_model) output must go under a "
                         f"'dryrun' directory, not {path}")
    if source != SOURCE_DRYRUN and in_dryrun:
        raise SystemExit(f"REFUSED: hardware output must not go under a 'dryrun' directory: {path}")


def write_csv(path, rows: list[dict], fields: list[str], source: str) -> Path:
    path = Path(path)
    check_output_path(path, source)
    for r in rows:
        if r.get("source") != source:
            raise SystemExit(f"REFUSED: row source {r.get('source')!r} != run source {source!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(META_COLUMNS) + [c for c in EXTRA_META] + [f for f in fields
                                                          if f not in META_COLUMNS + EXTRA_META]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {path} ({len(rows)} rows, source={source})")
    return path


def percentiles(a, ps=(5, 50, 95, 99)) -> dict:
    a = np.asarray(a, dtype=np.float64)
    out = {f"p{p}": float(np.percentile(a, p)) for p in ps}
    out.update(mean=float(a.mean()), min=float(a.min()), max=float(a.max()), n=int(a.size))
    return out


def rtl_cycles_for(pkg: Package) -> dict:
    """RTL-simulation cycles (label rtl_sim) for the net from the packaged copies of
    v2/results/rtl_network.csv (network jobs: per-layer + total) and rtl_cycles.csv (single-layer
    jobs). Returns {"layers": {i: set}, "total": set, "layer_single": {name: set}}."""
    net = pkg.name
    out = {"layers": {}, "total": set(), "layer_single": {}}
    with (pkg.net_dir / "rtl_network.csv").open() as f:
        for r in csv.DictReader(f):
            if r["net"] != net:
                continue
            out["total"].add(int(r["rtl_total"]))
            for i, c in enumerate(r["rtl_layer_cycles"].split(",")):
                out["layers"].setdefault(i, set()).add(int(c))
    with (pkg.net_dir / "rtl_cycles.csv").open() as f:
        for r in csv.DictReader(f):
            if r["net"] == net and r["kind"] == "layer":
                out["layer_single"].setdefault(r["layer"], set()).add(int(r["rtl_cycles"]))
    return out


CLOCK_TOL_MHZ = 0.5          # read-back vs closed-clock tolerance (PLL rounding, e.g. 199.998001)
B2_BASE_CLOCKS = (100.0, 150.0, 200.0)      # EXPERIMENTS.md B2
B2_EXTRA_CLOCKS = (250.0, 300.0)            # only if the deployed bitstream closed timing there


def b2_sweep_clocks(closed_mhz: float, tol: float = CLOCK_TOL_MHZ) -> list[float]:
    """B2 sweep = {100, 150, 200} ∪ {250 if closed >= 250} ∪ {300 if closed >= 300}, each capped
    at the bitstream's closed pl_clk0 (a nominal point within tol of the closed clock is requested
    AT the closed clock, e.g. 249.997498 instead of 250; points above it are dropped)."""
    closed = float(closed_mhz)
    out = []
    for c in B2_BASE_CLOCKS + B2_EXTRA_CLOCKS:
        if c in B2_EXTRA_CLOCKS and closed < c - tol:
            continue
        if c > closed + tol:
            continue
        v = round(min(c, closed), 6)
        if v not in out:
            out.append(v)
    return out


def one(s):
    """The single value of a set, or '' (with 'varies' flagged by the caller)."""
    return next(iter(s)) if len(s) == 1 else ""


def image_range(pkg: Package, limit) -> range:
    return range(pkg.n if limit is None else min(limit, pkg.n))


def infer_loop(dev, pkg: Package, indices, read_counters: bool = True, max_errors: int = 10,
               progress_every: int = 1000, tag: str = "") -> dict:
    """Run dev.infer on pkg.x_act[i] for i in indices. Job errors / timeouts are recorded
    (logits row = INT32_MIN sentinel, counters -1), the device is recovered (soft_reset) and the
    loop continues; more than max_errors aborts. Returns numpy arrays + error list."""
    import time as _t
    import gos_driver as D
    idx = np.asarray(list(indices), dtype=np.int64)
    n, oc, nl = idx.size, int(pkg.net["OC"]), int(pkg.net["n_layers"])
    out = {"idx": idx, "logits": np.full((n, oc), np.iinfo(np.int32).min, dtype=np.int32),
           "total_cyc": np.full(n, -1, np.int64), "mac_active": np.full(n, -1, np.int64),
           "stall": np.full(n, -1, np.int64), "violation": np.full(n, -1, np.int64),
           "layer_cyc": np.full((n, nl), -1, np.int64), "polls": np.zeros(n, np.int64),
           "t_write_ns": np.zeros(n, np.int64), "t_clear_ns": np.zeros(n, np.int64), "t_run_ns": np.zeros(n, np.int64),
           "t_logit_ns": np.zeros(n, np.int64), "t_counter_ns": np.zeros(n, np.int64),
           "ok": np.zeros(n, bool), "errors": []}
    x_act = pkg.x_act
    t0 = _t.perf_counter()
    for k, i in enumerate(idx):
        try:
            r = dev.infer(x_act[i], read_counters=read_counters)
        except (D.GosJobError, D.GosTimeout) as e:
            out["errors"].append({"img": int(i), "error": type(e).__name__, "msg": str(e)})
            print(f"  [{tag}] img {int(i)}: {e}", file=sys.stderr)
            if len(out["errors"]) > max_errors:
                raise SystemExit(f"[{tag}] more than {max_errors} job errors; aborting")
            dev.recover()
            continue
        out["ok"][k] = True
        out["logits"][k] = r.logits
        out["polls"][k] = r.polls
        out["t_write_ns"][k], out["t_run_ns"][k] = r.t_write_ns, r.t_run_ns
        out["t_clear_ns"][k] = r.t_clear_ns
        out["t_logit_ns"][k], out["t_counter_ns"][k] = r.t_logit_ns, r.t_counter_ns
        if read_counters:
            out["total_cyc"][k], out["mac_active"][k] = r.total_cyc, r.mac_active
            out["stall"][k], out["violation"][k] = r.stall, r.violation
            out["layer_cyc"][k] = r.layer_cyc
        if progress_every and (k + 1) % progress_every == 0:
            print(f"  [{tag}] {k + 1}/{n} images, {_t.perf_counter() - t0:.1f} s", flush=True)
    out["duration_s"] = _t.perf_counter() - t0
    return out


def main(argv=None) -> int:
    """`python3 board_common.py verify [data_dir]`: check every package manifest."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] != "verify":
        print("usage: board_common.py verify [data_dir]")
        return 2
    data_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_DATA_DIR
    ok = True
    for net in NETS:
        try:
            m = verify_manifest(data_dir / net)
            print(f"{net}: OK ({len(m['files'])} files, commit {m['git_commit'][:8]}, "
                  f"dirty={m['git_dirty']}, n={m['n_images']})")
        except ManifestError as e:
            ok = False
            print(f"{net}: FAIL {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
