#!/usr/bin/env python3
"""V3 baseline board session: AMD DPU + ONNX Runtime (INT8 / FP32) on the KV260, per net:
accuracy on the CIFAR-10 test set, batch-1 latency (p50/p95/p99, median 95 % CI) and SOM-rail energy.

    ./session.sh 1|2|3 [--plan] [--resume|--fresh] [--nets ...] [--configs ...] [--no-power] ...
    ./session.sh 1 --backend model                 # LAPTOP DRY RUN -> v3/results/dryrun/baseline/

Adapted from v2/dpu/dpu_session.py (DPU runner I/O, overlay checks, rows) and v2/board/run_sessions.py
(environment pre-flight, resumable state, session index, login / apt-window check) @ 28dd2ad; V3 runs
everything in one process (no PL accelerator of our own here).

SYSTEMS / CONFIGURATIONS (--configs; default every one the package has):
  dpu          AMD DPU (DPUCZDX8G_ISA1_B4096, prebuilt pynq-dpu 2.5 KV260 overlay, Vitis AI 2.5.0 vai_q PTQ
               of the same FP32 checkpoint); VART runner execute_async + wait, batch 1; pinned to --meas-cores
  ort_int8_tN  ONNX Runtime CPU EP, int8_qdq.onnx (ORT static QDQ S8S8, per-channel weights, MinMax, calibrated
               on train[0:1024]) with intra_op_num_threads = N; N = 1 pinned to --meas-cores, N > 1 to --cpun-cores
  ort_fp32_tN  ONNX Runtime CPU EP, fp32.onnx (opset 17), same threading
  None of these is our accelerator or our INT8 reference; each is its own quantization / precision.

PER NET x CONFIGURATION (step acc.<net>.<cfg>): batch 1, one image per call, e2e from the raw stored
uint8 image: pre (u8 -> preprocessing of the checkpoint (baseline_common) -> runner input buffer; DPU:
int8 = clip(floor(x * 2^fix_point + 0.5), -128, 127)), infer (DPU: execute_async + wait; ORT: session.run),
post (argmax). --warmup images (default 50) first, discarded; then every image of the test set (default
all 10,000; --limit N) timed with perf_counter_ns. Accuracy vs labels and agreement with the laptop
reference predictions of the same model (DPU: vai_q PyTorch model; ORT INT8: ORT INT8 on the laptop;
ORT FP32: PyTorch FP32). Latency per metric (pre / infer / post / e2e): p50 / p95 / p99 / mean / min /
max and the median with its distribution-free 95 % CI (stats.summarize).
POWER (step power.<net>): power_log.run_power_protocol with one e2e batch-1 loop per configuration
(images cycled), idle / run phases x --repeats, run phases in a seeded random order per repeat,
P_idle = bracketing idle phases, E_sys = dP x time/image. Label "SOM-rail power (INA260)".

ENVIRONMENT PRE-FLIGHT (board_env.py, as V2 run_sessions.py): no package manager running (apt/dpkg/
unattended-upgrades/packagekitd), cpufreq governor 'performance' + fixed frequency read back, pinning
read back, die temperature; the apt/PackageKit timer state is recorded (masked during the V2 campaign).
FAIL -> exit 3 unless --allow-non-paper-grade. After the run: ssh logins and apt/PackageKit unit windows
overlapping a power phase are flagged (login_spikes.py).
paper_grade = pre-flight passed AND clean package (git_dirty False) AND trained checkpoint (not the
dry-run throwaway) AND source 'measured on KV260'.

STATE: <results>/baseline_state.json; a step is skipped on rerun if it finished OK with the same
parameters and its outputs still verify (sha256). Provenance change -> exit 4 (use --fresh, which
archives the old outputs to <results>/archive/<ts>_fresh/). SESSION INDEX K: K = 1 -> <results>/,
K >= 2 -> <results>/rep<K>/ (aggregate_sessions.py combines them, V2 D26 median rule).

OUTPUTS: hw_baseline_accuracy.csv, hw_baseline_latency.csv, hw_baseline_power_ina260_{samples,phases,
summary}_<net>.csv, hw_baseline_preds_<net>_<cfg>.npz, hw_baseline_session_info.json, logs/.
source = 'measured on KV260' (board) / 'dryrun_model' (laptop, only under a dryrun directory).
Exit: 0 OK; 1 a step failed; 2 usage; 3 pre-flight failed; 4 provenance changed / lock held.
"""
from __future__ import annotations

import os
import sys

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # before numpy is imported
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import baseline_common as bc  # noqa: E402
import board_env as benv  # noqa: E402
import login_spikes  # noqa: E402
import power_log as pl  # noqa: E402
import stats  # noqa: E402

STATE_NAME = "baseline_state.json"
STATE_VERSION = 1
SYSTEMS = ("dpu", "ort_int8", "ort_fp32")
DEFAULT_THREADS = (1, 4)
DEFAULT_WARMUP = 50
DEFAULT_MEAS_CORES = "3"
DEFAULT_CPUN_CORES = "0-3"
METRICS = ("pre", "infer", "post", "e2e")
REF_NAME = {"dpu": "vai_q PyTorch model (container)", "ort_int8": "ORT INT8 on the laptop",
            "ort_fp32": "PyTorch FP32 on the laptop"}

ACC_FIELDS = ["config", "system", "threads", "measurement", "model_file", "model_sha256", "runtime_version",
              "io", "preprocess", "images", "correct", "accuracy_pct", "ref_name", "ref_images", "agree_ref", "agree_ref_pct",
              "warmup_images", "batch", "cores", "status"]
LAT_FIELDS = ["config", "system", "threads", "measurement", "model_sha256", "runtime_version", "metric", "unit",
              "n", "warmup_images", "batch", "p50", "p95", "p99", "mean", "min", "max", "median",
              "median_ci_lo_us", "median_ci_hi_us", "ci_coverage", "ci_method", "repeats_ok", "cores", "status"]


class PreflightError(RuntimeError):
    pass


def utc_now() -> str:
    return bc.utc_now()


def stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(p) -> str:
    return bc.sha256_file(p)


# ---- configurations -----------------------------------------------------------------------------
def all_configs(systems=SYSTEMS, threads=DEFAULT_THREADS) -> list[str]:
    out = []
    for s in systems:
        out += ["dpu"] if s == "dpu" else [f"{s}_t{t}" for t in threads]
    return out


def parse_config(cfg: str) -> tuple[str, int | None]:
    if cfg == "dpu":
        return "dpu", None
    for s in ("ort_int8", "ort_fp32"):
        if cfg.startswith(s + "_t") and cfg[len(s) + 2:].isdigit() and int(cfg[len(s) + 2:]) >= 1:
            return s, int(cfg[len(s) + 2:])
    raise ValueError(f"unknown configuration {cfg!r} (dpu, ort_int8_t<N>, ort_fp32_t<N>)")


# ---- DPU runner I/O (copied from v2/dpu/dpu_session.py @ 28dd2ad) ---------------------------------
def _attr(t, name):
    try:
        return t.get_attr(name) if t.has_attr(name) else None
    except Exception:  # noqa: BLE001
        return None


class IoSpec:
    """Input/output buffer layout for one runner (dims NHWC, dtype, fix_point)."""

    def __init__(self, in_dims, in_dtype: str, in_fp, out_dims, out_dtype: str, out_fp):
        self.in_dims, self.out_dims = tuple(int(v) for v in in_dims), tuple(int(v) for v in out_dims)
        self.in_int8 = "int8" in in_dtype.lower()
        self.out_int8 = "int8" in out_dtype.lower()
        if self.in_int8 and in_fp is None:
            raise SystemExit("int8 DPU input without fix_point")
        self.in_dtype, self.out_dtype = in_dtype, out_dtype
        self.in_fp, self.out_fp = in_fp, out_fp
        self.in_scale = float(2.0 ** in_fp) if self.in_int8 else 1.0
        if self.in_dims[0] != 1:
            raise SystemExit(f"xmodel batch {self.in_dims[0]} != 1 (batch-1 measurement)")

    @classmethod
    def from_runner(cls, r):
        ti, to = r.get_input_tensors()[0], r.get_output_tensors()[0]
        return cls(ti.dims, str(getattr(ti, "dtype", "xint8")), _attr(ti, "fix_point"),
                   to.dims, str(getattr(to, "dtype", "xint8")), _attr(to, "fix_point"))

    @classmethod
    def from_info(cls, xi: dict):
        i, o = xi["dpu_input"][0], xi["dpu_output"][0]
        return cls(i["dims"], i.get("dtype") or "xint8", i["fix_point"],
                   o["dims"], o.get("dtype") or "xint8", o["fix_point"])

    def buffers(self):
        return ([np.empty(self.in_dims, dtype=np.int8 if self.in_int8 else np.float32, order="C")],
                [np.empty(self.out_dims, dtype=np.int8 if self.out_int8 else np.float32, order="C")])

    def describe(self) -> dict:
        return {"in_dims": list(self.in_dims), "in_dtype": self.in_dtype, "in_fix_point": self.in_fp,
                "out_dims": list(self.out_dims), "out_dtype": self.out_dtype, "out_fix_point": self.out_fp,
                "input_conversion": (f"int8 clip(floor(x*2^{self.in_fp}+0.5),-128,127)"
                                     if self.in_int8 else "float32 copy")}


def fill_input(buf: np.ndarray, x_hwc: np.ndarray, io: IoSpec):
    """One preprocessed image (H,W,C float32) into the preallocated input buffer (1,H,W,C)."""
    if io.in_int8:
        q = np.floor(x_hwc * np.float32(io.in_scale) + np.float32(0.5))
        buf[0] = np.clip(q, -128, 127).astype(np.int8)
    else:
        buf[0] = x_hwc


class FakeRunner:
    """DRY RUN stand-in for the VART runner: one-hot int8 output of a given prediction vector
    (pipeline test only; numbers are meaningless)."""

    def __init__(self, io: IoSpec, preds: np.ndarray):
        self.io, self.preds, self.k, self.jobs = io, preds, 0, 0

    def execute_async(self, inp, out):
        o = out[0].reshape(-1)
        o[:] = 0
        o[int(self.preds[self.k % len(self.preds)])] = 100
        self.k += 1
        self.jobs += 1
        return (self.jobs, 0)

    def wait(self, job):
        return 0


# ---- runners (one interface for DPU and ORT) ------------------------------------------------------
class DpuSystem:
    system = "dpu"

    def __init__(self, sess, net: str):
        pkg, n = sess.pkg, sess.pkg.net(net)
        d = n["dpu"]
        self.net, self.threads = net, None
        self.model_file = d["xmodel"]
        self.model_sha256 = d["xmodel_sha256"]
        self.xinfo = json.loads(pkg.path(d["xmodel_info"]).read_text())
        self.ref_pred = np.load(pkg.path(d["vaiq_pred"]))
        self.spec = pkg.preprocess_spec(net)
        self.fingerprint = ",".join(self.xinfo.get("dpu_fingerprint_hex") or []) or "unknown"
        if sess.dry:
            self.io = IoSpec.from_info(self.xinfo)
            self.runner = FakeRunner(self.io, self.ref_pred)
            self.runtime_version = "fake runner (dry run)"
        else:
            ov = sess.overlay()
            ov.load_model(str(pkg.path(self.model_file)))
            self.runner = ov.runner
            self.io = IoSpec.from_runner(self.runner)
            self.runtime_version = f"pynq_dpu {sess.ovinfo.get('pynq_dpu_version', '')}"
        exp = tuple(self.io.in_dims[1:])
        if exp != (32, 32, 3):
            raise SystemExit(f"[{net}] DPU input {exp} != (32, 32, 3)")
        self.inp, self.out = self.io.buffers()
        info = pkg.manifest.get("dpu", {})
        self.label = (f"AMD DPU ({info.get('dpu_arch', 'DPUCZDX8G')}, Vitis AI {info.get('vitis_ai_version', '?')} "
                      f"vai_q PTQ, fingerprint {self.fingerprint}), batch 1")

    def io_desc(self) -> str:
        return json.dumps(self.io.describe(), sort_keys=True)

    def reset_dry(self):
        if isinstance(self.runner, FakeRunner):
            self.runner.k = 0

    def step(self, x_u8, pc=time.perf_counter_ns):
        t0 = pc()
        fill_input(self.inp[0], bc.preprocess_hwc(x_u8, self.spec), self.io)
        t1 = pc()
        self.runner.wait(self.runner.execute_async(self.inp, self.out))
        t2 = pc()
        p = int(np.argmax(self.out[0].reshape(-1)))
        t3 = pc()
        return p, t0, t1, t2, t3


class OrtSystem:
    def __init__(self, sess, net: str, system: str, threads: int):
        import onnxruntime as ort  # noqa: PLC0415  (RunnerUnavailable if missing: caught by the caller)
        pkg, n = sess.pkg, sess.pkg.net(net)
        key = "int8_qdq" if system == "ort_int8" else "fp32"
        o = n["onnx"]
        self.system, self.net, self.threads = system, net, threads
        self.model_file = o[key]
        self.model_sha256 = o[key + "_sha256"]
        self.ref_pred = np.load(pkg.path(o["ref_pred_" + key]))
        self.spec = pkg.preprocess_spec(net)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(pkg.path(self.model_file)), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.runtime_version = f"onnxruntime {ort.__version__}"
        q = o.get("ort_int8_quantization", {})
        what = (f"INT8 QDQ ({q.get('activation_type', 'QInt8')}/{q.get('weight_type', 'QInt8')}, per-channel "
                f"{q.get('per_channel', True)}, {q.get('calibrate_method', 'MinMax')})" if system == "ort_int8"
                else "FP32")
        self.label = (f"ONNX Runtime {ort.__version__} CPU EP {what}, {threads} thread(s), "
                      f"{platform.machine()} CPU, batch 1")

    def io_desc(self) -> str:
        return ""

    def reset_dry(self):
        pass

    def step(self, x_u8, pc=time.perf_counter_ns):
        t0 = pc()
        x = bc.preprocess(x_u8, self.spec)[None]
        t1 = pc()
        y = self.sess.run(None, {self.in_name: x})[0]
        t2 = pc()
        p = int(np.argmax(y.reshape(-1)))
        t3 = pc()
        return p, t0, t1, t2, t3


# ---- session context ----------------------------------------------------------------------------
class Session:
    """Provenance + row metadata + pinning for one baseline_session.py run."""

    def __init__(self, a, pkg: bc.Package, rd: Path, say):
        self.a, self.pkg, self.rd, self.say = a, pkg, rd, say
        self.dry = a.backend == "model"
        self.source = bc.SOURCE_DRYRUN if self.dry else bc.SOURCE_HW
        self.out_dir = rd
        self.env: dict = {}
        self.step_env: dict = {}
        self.board_id = a.board_id or ("" if self.dry else bc.board_id_default())
        self.hostname = socket.gethostname()
        self._ov = None
        self.ovinfo: dict = {"overlay": "none (dry run)"} if self.dry else {}
        self.meas = benv.parse_cores(a.meas_cores)
        self.cpun = benv.parse_cores(a.cpun_cores)

    # pinning: DPU and 1-thread ORT on --meas-cores, multi-thread ORT on --cpun-cores
    def cores_for(self, cfg: str) -> set:
        _, t = parse_config(cfg)
        return self.cpun if (t or 1) > 1 else self.meas

    def overlay(self):
        if self._ov is None:
            self._ov, self.ovinfo = open_overlay(self.a, self.pkg.manifest.get("dpu", {}).get(
                "overlay_bit_md5_expected", ""))
        return self._ov

    @property
    def dirty(self) -> bool:
        return self.pkg.dirty

    def meta(self, net: str = "", config: str = "", duration_s="", num_inferences="") -> dict:
        sysname = parse_config(config)[0] if config else ""
        dpu = sysname == "dpu"
        n = self.pkg.net(net) if net else {}
        ck = n.get("checkpoint_kind", bc.CHECKPOINT_TRAINED) if net else bc.CHECKPOINT_TRAINED
        clk = self.ovinfo.get("fclk0_mhz", "") if dpu and not self.dry else ""
        return {"timestamp": utc_now(), "git_commit": self.pkg.manifest.get("git_commit", ""),
                "git_dirty": self.dirty, "vivado_version": "",
                "bitstream_sha256": self.ovinfo.get("bit_sha256", "") if dpu else "",
                "board_id": self.board_id, "net": net, "layer": "all",
                "clock_mhz": f"{clk:.6f}" if isinstance(clk, float) else clk, "source": self.source,
                "duration_s": duration_s if duration_s == "" else f"{float(duration_s):.3f}",
                "num_inferences": num_inferences, "board_hostname": self.hostname,
                "clock_source": ("pynq Clocks.fclk0_mhz read-back (pl_clk0 of the DPU overlay)" if dpu and not self.dry
                                 else ""),
                "scripts_commit": self.pkg.manifest.get("git_commit", ""),
                "package_manifest_sha256": self.pkg.manifest_sha256,
                "checkpoint_sha256": n.get("checkpoint_sha256", ""), "checkpoint_kind": ck,
                "backend": ("fake DPU runner + laptop ORT (dry run)" if self.dry else "pynq_dpu / onnxruntime"),
                **bc.env_meta(self.source, self.dirty, {**self.env, **self.step_env}, ck),
                # the cores this configuration runs on (the session itself stays on --meas-cores)
                **({"cpu_affinity": benv.cores_str(self.cores_for(config))} if config else {})}


def open_overlay(a, expected_md5: str = ""):
    """(DpuOverlay, info) -- copied from v2/dpu/dpu_session.open_overlay @ 28dd2ad."""
    from pynq_dpu import DpuOverlay  # noqa: PLC0415
    import pynq_dpu  # noqa: PLC0415
    ov = DpuOverlay(a.overlay)
    bit = Path(a.overlay)
    if not bit.is_absolute():
        bit = Path(pynq_dpu.__file__).resolve().parent / a.overlay
    md5 = hashlib.md5(bit.read_bytes()).hexdigest() if bit.is_file() else ""
    info = {"overlay": str(bit), "bit_sha256": sha256_file(bit) if bit.is_file() else "", "bit_md5": md5,
            "bit_md5_expected": expected_md5, "pynq_dpu_version": getattr(pynq_dpu, "__version__", "")}
    if expected_md5 and md5 != expected_md5:
        msg = f"overlay {bit} md5 {md5} != prebuilt pynq-dpu 2.5 KV260 overlay {expected_md5}"
        if not a.allow_overlay_mismatch:
            raise SystemExit("REFUSED: " + msg + " (--allow-overlay-mismatch to override)")
        print("WARNING: " + msg, file=sys.stderr)
    try:
        from pynq.ps import Clocks  # noqa: PLC0415
        info["fclk0_mhz"] = float(Clocks.fclk0_mhz)
    except Exception as e:  # noqa: BLE001
        info["fclk0_mhz"] = ""
        info["fclk0_error"] = str(e)
    try:
        q = subprocess.run(["xdputil", "query"], capture_output=True, text=True, timeout=30)
        info["xdputil_query"] = q.stdout[-4000:]
        try:
            j = json.loads(q.stdout)
            info["hw_fingerprint"] = ",".join(str(k.get("fingerprint", "")) for k in j.get("kernels", [])
                                              if k.get("fingerprint"))
        except ValueError:
            pass
    except Exception as e:  # noqa: BLE001
        info["xdputil_error"] = str(e)
    return ov, info


def with_cores(cores: set):
    """Context manager: process affinity = cores for the block, restored after."""
    class _C:
        def __enter__(self):
            self.prev = os.sched_getaffinity(0)
            if cores:
                os.sched_setaffinity(0, cores)
            return self

        def __exit__(self, *exc):
            os.sched_setaffinity(0, self.prev)
    return _C()


def make_system(sess: Session, net: str, cfg: str):
    system, threads = parse_config(cfg)
    if system == "dpu":
        if "dpu" not in sess.pkg.net(net):
            raise RuntimeError("unavailable: package has no DPU xmodel for this net")
        return DpuSystem(sess, net)
    if "onnx" not in sess.pkg.net(net):
        raise RuntimeError("unavailable: package has no ONNX models for this net")
    try:
        return OrtSystem(sess, net, system, threads)
    except ImportError as e:
        raise RuntimeError(f"unavailable: onnxruntime not importable ({e})") from None


# ---- steps ----------------------------------------------------------------------------------------
def pct(c, n):
    return f"{100.0 * c / n:.2f}" if n else ""


def run_acc_step(sess: Session, net: str, cfg: str) -> dict:
    """Accuracy + latency of one configuration. Returns {rows_acc, rows_lat, outputs}."""
    a = sess.a
    system, threads = parse_config(cfg)
    cores = sess.cores_for(cfg)
    x_u8, y = sess.pkg.x_u8, sess.pkg.labels
    n = len(y) if a.limit is None else min(a.limit, len(y))
    with with_cores(cores):
        aff = benv.affinity_of(0)
        s = make_system(sess, net, cfg)
        sess.say(f"[{net}/{cfg}] {s.label}; images {n}, warm-up {a.warmup}, cores {aff}"
                 + ("  (DRY RUN: laptop / fake runner -- NOT measurements)" if sess.dry else ""))
        for k in range(a.warmup):                      # discarded
            s.step(x_u8[k % n])
        s.reset_dry()
        preds = np.empty(n, dtype=np.int64)
        t = np.empty((4, n), dtype=np.int64)
        t_start = time.perf_counter()
        for i in range(n):
            p, t0, t1, t2, t3 = s.step(x_u8[i])
            preds[i] = p
            t[0, i], t[1, i], t[2, i], t[3, i] = t1 - t0, t2 - t1, t3 - t2, t3 - t0
            if a.progress and (i + 1) % a.progress == 0:
                sess.say(f"  [{net}/{cfg}] {i + 1}/{n} images, {time.perf_counter() - t_start:.1f} s")
        dur = time.perf_counter() - t_start
    correct = int((preds == y[:n]).sum())
    m_ref = min(n, len(s.ref_pred))                    # laptop reference may cover fewer images (export --limit)
    agree = int((preds[:m_ref] == s.ref_pred[:m_ref]).sum())
    common = {"config": cfg, "system": system, "threads": threads or "", "measurement": s.label,
              "model_sha256": s.model_sha256, "runtime_version": s.runtime_version, "cores": aff, "status": "ok"}
    acc = sess.meta(net, cfg, dur, n)
    acc.update(common, model_file=s.model_file, io=s.io_desc(),
               preprocess=json.dumps(s.spec, sort_keys=True), images=n, correct=correct,
               accuracy_pct=pct(correct, n), ref_name=REF_NAME[system], ref_images=m_ref, agree_ref=agree,
               agree_ref_pct=pct(agree, m_ref), warmup_images=a.warmup, batch=1)
    lat = []
    for j, m in enumerate(METRICS):
        st = stats.summarize(t[j] / 1e3)
        r = sess.meta(net, cfg, dur, n)
        r.update(common, metric=m, unit="us", n=st["n"], warmup_images=a.warmup, batch=1,
                 p50=stats.fmt(float(np.percentile(t[j] / 1e3, 50))),
                 **{k: stats.fmt(st[k]) for k in ("p95", "p99", "mean", "min", "max", "median")},
                 **stats.ci_fields(st, "us"))
        lat.append(r)
    e2e = stats.summarize(t[3] / 1e3)
    sess.say(f"[{net}/{cfg}] accuracy {correct}/{n} = {pct(correct, n)} %, agreement with {REF_NAME[system]} "
             f"{agree}/{m_ref}; e2e median {e2e['median']:.1f} us [{stats.fmt(e2e['ci_lo'], 1)}, "
             f"{stats.fmt(e2e['ci_hi'], 1)}] p95 {e2e['p95']:.1f} p99 {e2e['p99']:.1f} us"
             + (" (DRY RUN)" if sess.dry else ""))
    npz = sess.rd / f"hw_baseline_preds_{net}_{cfg}.npz"
    bc.check_output_path(npz, sess.source)
    np.savez(npz, pred=preds, labels=y[:n], ref_pred=s.ref_pred[:m_ref],
             **{f"t_{m}_ns": t[j] for j, m in enumerate(METRICS)})
    return {"rows_acc": [acc], "rows_lat": lat, "outputs": [npz.name]}


def run_power_step(sess: Session, net: str, cfgs: list[str]) -> dict:
    a = sess.a
    x_u8 = sess.pkg.x_u8
    nimg = len(x_u8) if a.limit is None else min(a.limit, len(x_u8))
    workloads, keep = {}, []
    for cfg in cfgs:
        cores = sess.cores_for(cfg)
        with with_cores(cores):          # ORT creates its thread pool here: pool threads inherit the affinity
            try:
                s = make_system(sess, net, cfg)
            except RuntimeError as e:
                sess.say(f"[{net}/{cfg}] power: skipped ({e})")
                continue
            for k in range(max(1, a.warmup)):
                s.step(x_u8[k % nimg])
        keep.append(s)

        def phase(deadline, s=s, cores=cores):
            with with_cores(cores):
                return pl.loop_until(lambda k: s.step(x_u8[k % nimg]), deadline)
        workloads[cfg] = (phase, f"{s.label}; e2e loop, images cycled, cores {benv.cores_str(cores)}")
    if not workloads:
        raise RuntimeError("no configuration available for the power protocol")
    seed = a.order_seed if a.order_seed is not None else stats.new_seed()
    res = pl.run_power_protocol(sess, net, workloads, sensor=a.sensor, phase_s=a.phase_s, repeats=a.repeats,
                                rate_hz=a.rate_hz, tag=f"_{net}", order_seed=seed,
                                sensor_kw={"noise_w": 0.01, "seed": 1} if sess.dry else None)
    return {"rows_acc": [], "rows_lat": [], "outputs": [Path(p).name for p in res["files"].values()],
            "order_seed": seed, "undersampled": res["undersampled"]}


# ---- pre-flight (environment part copied from v2/board/run_sessions.env_preflight @ 28dd2ad) -------
def env_preflight(a, rd: Path, dry: bool, say) -> tuple[dict, benv.CpuFreqControl]:
    tag = "DRY RUN " if dry else ""
    if dry:
        root = rd / ".dryrun_env"
        if root.exists():
            shutil.rmtree(root)
        paths = benv.make_fake_tree(root)
    else:
        paths = {"sysfs_root": benv.SYSFS_CPU, "proc_root": benv.PROC, "iio_root": benv.IIO,
                 "hwmon_root": benv.HWMON}
    fails, warns = [], []
    downgrade = a.allow_non_paper_grade or dry

    def ok(m):
        say(f"  [{tag}env] OK    {m}")

    def fail(m):
        if downgrade:
            say(f"  [{tag}env] WARN  {m} -- {'dry run' if dry else '--allow-non-paper-grade'}: NOT paper-grade")
            warns.append(m)
        else:
            say(f"  [{tag}env] FAIL  {m}")
            fails.append(m)

    say(f"[{tag}env] measurement environment pre-flight" + (" on a FAKE sysfs/proc tree" if dry else ""))
    busy = benv.pkg_manager_busy(paths["proc_root"])
    if busy:
        fail("package manager running: " + "; ".join(f"pid {b['pid']} {b['comm']}" for b in busy))
    else:
        ok("no apt/dpkg/unattended-upgrades/packagekitd process running")
    timers = apt_timer_state(dry)
    (ok if timers.get("all_inactive") else (lambda m: (say(f"  [{tag}env] WARN  {m}"), warns.append(m))))(
        f"apt/PackageKit units: {timers.get('state')}")
    ctl = benv.CpuFreqControl(root=paths["sysfs_root"], fake=dry)
    before = ctl.snapshot()
    if a.plan:
        gov = {"ok": None, "target_khz": ctl.target_khz(a.cpu_freq_khz), "errors": []}
        say(f"  [{tag}env] --plan: governor not changed (would fix {gov['target_khz']} kHz)")
    else:
        gov = ctl.apply(a.cpu_freq_khz, settle=(lambda: benv.fake_kernel_settle(paths["sysfs_root"])) if dry else None)
        if gov["ok"]:
            ok(f"cpufreq: governor performance, fixed {gov['target_khz']} kHz, read back "
               f"{gov['readback']['cpu_cur_freq_khz']} (restored on exit)")
        else:
            fail("cpufreq governor/frequency: " + "; ".join(gov["errors"]))
    avail = benv.available_cores()
    for name, spec in (("meas", a.meas_cores), ("cpun", a.cpun_cores)):
        c = benv.parse_cores(spec)
        if not c or not c <= avail:
            fail(f"--{name}-cores {spec!r} not a subset of the available cores {benv.cores_str(avail)}")
    pin = benv.pin(0, benv.parse_cores(a.meas_cores) & avail or avail)
    if pin["ok"]:
        ok(f"session pinned to {pin['readback']} (DPU + 1-thread ORT); multi-thread ORT on {a.cpun_cores}")
    else:
        fail(f"pinning: {pin['error']}")
    t = benv.read_die_temp(paths["iio_root"], paths["hwmon_root"])
    if t["max_c"] is None:
        say(f"  [{tag}env] WARN  die temperature unavailable")
        warns.append("die temperature unavailable")
    else:
        ok(f"die temperature {t['max_c']:.1f} C max ({t['source']})" + (" [FAKE]" if dry else ""))
    if not dry:
        du = shutil.disk_usage(rd)
        (ok if du.free > 2e9 else fail)(f"free disk {du.free / 1e9:.1f} GB in {rd}")
    paper = not dry and not a.allow_dirty and not a.allow_non_paper_grade and not fails
    env = {"paper_grade": paper, "pkg_manager_offenders": busy, "apt_units": timers, "cpufreq_before": before,
           "cpufreq_apply": gov, "cpu_freq_khz": gov.get("target_khz"),
           "cpu_governor": "performance" if gov.get("ok") else ",".join(
               sorted({v["scaling_governor"] or "" for v in before["policies"].values()})),
           "pin": pin, "die_temp": t, "fake_tree": dry, "warnings": warns, "session_index": a.session,
           "note": "" if paper else "pre-flight downgraded: not paper-grade"}
    if fails:
        raise PreflightError("ENVIRONMENT PRE-FLIGHT FAILED (override only with --allow-non-paper-grade):\n  - "
                             + "\n  - ".join(fails))
    say(f"[{tag}env] {'PAPER-GRADE' if paper else 'NOT paper-grade'} environment ({len(warns)} warnings)")
    return env, ctl


APT_UNITS = ("apt-daily.timer", "apt-daily-upgrade.timer", "packagekit.service", "unattended-upgrades.service")


def apt_timer_state(dry: bool) -> dict:
    """systemctl is-enabled / is-active of the apt/PackageKit units (V2 D27: masked for the campaign)."""
    if dry:
        return {"state": "not checked (dry run)", "all_inactive": True}
    out = {}
    for u in APT_UNITS:
        try:
            en = subprocess.run(["systemctl", "is-enabled", u], capture_output=True, text=True, timeout=10).stdout.strip()
            ac = subprocess.run(["systemctl", "is-active", u], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as e:  # noqa: BLE001
            en, ac = f"error {type(e).__name__}", ""
        out[u] = f"{en}/{ac}"
    timers_off = all(not out[u].startswith("enabled") for u in APT_UNITS[:2])
    active = any(v.endswith("/active") for k, v in out.items() if k != "unattended-upgrades.service")
    return {"state": ", ".join(f"{k}={v}" for k, v in out.items()), "units": out,
            "all_inactive": timers_off and not active}


def login_check(rd: Path, since_utc: str, say) -> list:
    """Copied from v2/board/run_sessions.login_check @ 28dd2ad: ssh logins and apt/PackageKit unit windows
    overlapping an INA260 power phase are flagged (informational)."""
    try:
        text = login_spikes.fetch_journal(since_utc.replace("T", " ")[:19], None)
    except Exception as e:  # noqa: BLE001
        say(f"[session] login check skipped: {type(e).__name__}: {e}")
        return []
    logins = login_spikes.parse_logins(text)
    (rd / "logs").mkdir(parents=True, exist_ok=True)
    (rd / "logs" / f"ssh_logins_{stamp()}.txt").write_text(text)
    found = login_spikes.analyse(login_spikes.read_phases(rd), logins)
    say(f"[session] ssh login check: {len(logins)} login(s) since {since_utc}, {len(found)} flagged")
    for f in found:
        say("  FLAG " + login_spikes.fmt(f))
    try:
        wins = login_spikes.service_windows(login_spikes.fetch_units(since_utc.replace("T", " ")[:19]))
    except Exception as e:  # noqa: BLE001
        say(f"[session] system-activity check skipped: {type(e).__name__}: {e}")
        wins = []
    sysf = login_spikes.analyse_windows(login_spikes.read_phases(rd), wins)
    say(f"[session] apt/PackageKit check: {len(wins)} window(s), {len(sysf)} power phase(s) overlap")
    for f in sysf:
        say("  FLAG " + login_spikes.fmt_window(f))
    return [login_spikes.fmt(f) for f in found] + [login_spikes.fmt_window(f) for f in sysf]


# ---- state ----------------------------------------------------------------------------------------
def load_state(p: Path) -> dict | None:
    if not p.is_file():
        return None
    s = json.loads(p.read_text())
    return s if s.get("version") == STATE_VERSION else None


def save_state(p: Path, s: dict):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1, default=str) + "\n")
    tmp.replace(p)


def verify_outputs(rd: Path, rec: dict) -> bool:
    return all((rd / n).is_file() and sha256_file(rd / n) == h for n, h in rec.get("outputs", {}).items())


def archive(rd: Path, why: str, say) -> Path | None:
    files = [p for p in rd.iterdir() if p.is_file() and (p.name.startswith("hw_baseline_") or p.name == STATE_NAME)]
    if not files:
        return None
    dst = rd / "archive" / f"{stamp()}_{why}"
    dst.mkdir(parents=True, exist_ok=True)
    for p in files:
        shutil.move(str(p), dst / p.name)
    say(f"[session] archived {len(files)} file(s) to {dst}")
    return dst


def results_dir_for(source: str, results_dir, index: int) -> Path:
    root = bc.resolve_out_dir(source, results_dir).resolve()
    return bc.resolve_out_dir(source, str(root / f"rep{index}")).resolve() if index > 1 else root


def default_pkg_dir() -> Path:
    return HERE if (HERE / bc.MANIFEST).is_file() else bc.PACKAGE_DIR_REPO


# ---- CLI ------------------------------------------------------------------------------------------
def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", type=int, choices=(1, 2, 3), help="session index (repeatability, V2 D26)")
    ap.add_argument("--backend", choices=("board", "model"), default="board",
                    help="board = KV260 (default); model = laptop DRY RUN (fake DPU runner, laptop ORT, mock sensor)")
    ap.add_argument("--pkg-dir", default=None, help="baseline package (default: this dir if it holds MANIFEST.json, "
                    "else v3/board/data/package)")
    ap.add_argument("--results-dir", default=None,
                    help="default: results/ (board) or v3/results/dryrun/baseline/ (model)")
    ap.add_argument("--nets", nargs="+", default=None, help="default: every net of the package")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="dpu, ort_int8_t<N>, ort_fp32_t<N> (default: dpu + ORT INT8/FP32 at --threads)")
    ap.add_argument("--threads", nargs="+", type=int, default=list(DEFAULT_THREADS))
    ap.add_argument("--power-configs", nargs="+", default=None, help="default: the --configs")
    ap.add_argument("--limit", type=int, default=None, help="first N test images (default: all 10,000)")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--progress", type=int, default=2000, help="progress line every N images (0 = off)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-power", action="store_true")
    g.add_argument("--power-only", action="store_true")
    ap.add_argument("--sensor", default="auto", choices=("auto",) + pl.SENSOR_ORDER + ("mock",))
    ap.add_argument("--phase-s", type=float, default=None, help=f"default {pl.DEFAULT_PHASE_S:g} (dry run 1)")
    ap.add_argument("--repeats", type=int, default=None, help=f"default {pl.DEFAULT_REPEATS} (dry run 1)")
    ap.add_argument("--rate-hz", type=float, default=pl.DEFAULT_RATE_HZ)
    ap.add_argument("--order-seed", type=int, default=None, help="power run-phase order seed (default: fresh)")
    ap.add_argument("--overlay", default="dpu.bit", help="pynq_dpu overlay (default: its prebuilt KV260 dpu.bit)")
    ap.add_argument("--allow-overlay-mismatch", action="store_true")
    ap.add_argument("--meas-cores", default=DEFAULT_MEAS_CORES)
    ap.add_argument("--cpun-cores", default=DEFAULT_CPUN_CORES)
    ap.add_argument("--cpu-freq-khz", type=int, default=None, help="default: highest available")
    ap.add_argument("--board-id", default=None)
    ap.add_argument("--allow-dirty", action="store_true", help="run a dirty package (rows git_dirty=True)")
    ap.add_argument("--allow-non-paper-grade", action="store_true")
    r = ap.add_mutually_exclusive_group()
    r.add_argument("--resume", action="store_true", default=True)
    r.add_argument("--fresh", action="store_true", help="archive this session's outputs and start over")
    ap.add_argument("--plan", action="store_true", help="pre-flight (no governor change) + step list, run nothing")
    return ap


def main(argv=None) -> int:
    a = make_parser().parse_args(argv)
    dry = a.backend == "model"
    if a.phase_s is None:
        a.phase_s = 1.0 if dry else pl.DEFAULT_PHASE_S
    if a.repeats is None:
        a.repeats = 1 if dry else pl.DEFAULT_REPEATS
    if (a.limit is not None and a.limit < 1) or a.warmup < 0 or a.repeats < 1 or a.phase_s <= 0:
        print("usage: --limit >= 1, --warmup >= 0, --repeats >= 1, --phase-s > 0", file=sys.stderr)
        return 2
    if not dry and platform.machine() not in ("aarch64", "arm64"):
        print(f"REFUSED: --backend board on a {platform.machine()} host (laptop: --backend model)", file=sys.stderr)
        return 2
    source = bc.SOURCE_DRYRUN if dry else bc.SOURCE_HW
    rd = results_dir_for(source, a.results_dir, a.session)
    (rd / "logs").mkdir(parents=True, exist_ok=True)
    lock_f = open(rd / ".baseline.lock", "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"REFUSED: another baseline session holds {rd / '.baseline.lock'}")
        return 4
    log = (rd / "logs" / f"baseline_session_{stamp()}.log").open("w")

    def say(m=""):
        print(m, flush=True)
        log.write(m + "\n")
        log.flush()

    t_inv = utc_now()
    pkg_dir = Path(a.pkg_dir).resolve() if a.pkg_dir else default_pkg_dir()
    try:
        pkg = bc.load_package(pkg_dir)
    except bc.ManifestError as e:
        say(f"PRE-FLIGHT FAILED: {e}")
        return 3
    nets = a.nets or pkg.nets
    bad = [n for n in nets if n not in pkg.nets]
    if bad:
        say(f"usage: nets {bad} not in the package ({pkg.nets})")
        return 2
    try:
        cfgs = a.configs or all_configs(SYSTEMS, a.threads)
        pcfgs = a.power_configs or cfgs
        for c in cfgs + pcfgs:
            parse_config(c)
    except ValueError as e:
        say(f"usage: {e}")
        return 2
    sess = Session(a, pkg, rd, say)
    say(f"[baseline] session {a.session} source={source} results={rd} package={pkg_dir} "
        f"(commit {pkg.manifest.get('git_commit', '')[:8]} dirty={pkg.dirty}) nets={nets} configs={cfgs}")
    kinds = {n: pkg.checkpoint_kind(n) for n in nets}
    if any(k != bc.CHECKPOINT_TRAINED for k in kinds.values()):
        say(f"[baseline] NOTE: checkpoint kinds {kinds}: rows with a {bc.CHECKPOINT_THROWAWAY} checkpoint are "
            "never paper-grade")
    if pkg.dirty and not dry and not a.allow_dirty:
        say("REFUSED: package built from a dirty tree (commit, rebuild with make_baseline_package.py, redeploy; "
            "or --allow-dirty: rows git_dirty=True, not paper data)")
        return 3
    try:
        env, ctl = env_preflight(a, rd, dry, say)
    except PreflightError as e:
        say(str(e))
        return 3
    sess.env = env
    steps = []
    if not a.power_only:
        steps += [(f"acc.{n}.{c}", "acc", n, c) for n in nets for c in cfgs]
    if not a.no_power:
        steps += [(f"power.{n}", "power", n, None) for n in nets]
    per_net_power = pl.n_phases(a.repeats, len(pcfgs)) * a.phase_s
    say(f"[baseline] {len(steps)} steps; images per acc step {a.limit or len(pkg.labels)} (+{a.warmup} warm-up); "
        f"power per net {pl.n_phases(a.repeats, len(pcfgs))} phases = {per_net_power / 60:.1f} min")
    for sid, *_ in steps:
        say(f"  step {sid}")
    if a.plan:
        say("[baseline] --plan: nothing run")
        return 0
    prov = {"package_manifest_sha256": pkg.manifest_sha256, "backend": a.backend, "session_index": a.session,
            "paper_grade_env": env["paper_grade"], "cpu_freq_khz": env.get("cpu_freq_khz"),
            "allow_dirty": a.allow_dirty}
    sp = rd / STATE_NAME
    state = load_state(sp)
    if a.fresh:
        archive(rd, "fresh", say)
        state = None
    if state is not None and state["provenance"] != prov:
        diff = [k for k in prov if state["provenance"].get(k) != prov[k]]
        say(f"REFUSED: provenance changed since the last run ({diff}); use --fresh (archives the old outputs)")
        return 4
    state = state or {"version": STATE_VERSION, "provenance": prov, "steps": {}}
    save_state(sp, state)
    failures = 0
    try:
        for sid, kind, net, cfg in steps:
            params = {"limit": a.limit, "warmup": a.warmup, "overlay": a.overlay, "meas_cores": a.meas_cores,
                      "cpun_cores": a.cpun_cores}
            if kind == "power":
                params.update(configs=pcfgs, phase_s=a.phase_s, repeats=a.repeats, rate_hz=a.rate_hz,
                              sensor=a.sensor)
            rec = state["steps"].get(sid)
            if rec and rec.get("status") == "ok" and rec.get("params") == params and verify_outputs(rd, rec):
                say(f"[baseline] {sid}: done earlier, outputs verify -> skipped")
                continue
            temp = benv.read_die_temp(*(([str(rd / ".dryrun_env/sys/bus/iio/devices"), str(rd / ".dryrun_env/sys/class/hwmon")])
                                        if dry else []))
            sess.step_env = {"step_id": sid, "die_temp_start_c": temp.get("max_c") or ""}
            say(f"\n[baseline] ===== {sid} ===== {utc_now()}")
            t0 = time.perf_counter()
            rec = {"status": "running", "params": params, "started_utc": utc_now()}
            state["steps"][sid] = rec
            save_state(sp, state)
            try:
                out = run_acc_step(sess, net, cfg) if kind == "acc" else run_power_step(sess, net, pcfgs)
                rec.update(status="ok", rows_acc=out["rows_acc"], rows_lat=out["rows_lat"],
                           outputs={n: sha256_file(rd / n) for n in out["outputs"]},
                           **{k: v for k, v in out.items() if k not in ("rows_acc", "rows_lat", "outputs")})
            except RuntimeError as e:
                if str(e).startswith("unavailable"):
                    rec.update(status=str(e))
                    say(f"[baseline] {sid}: {e}")
                else:
                    failures += 1
                    rec.update(status=f"FAIL: {type(e).__name__}: {e}")
                    say(f"[baseline] {sid}: FAIL {type(e).__name__}: {e}")
            except (pl.UndersampledError, pl.SensorUnavailable, Exception) as e:  # noqa: BLE001  one step fails, not the session
                failures += 1
                rec.update(status=f"FAIL: {type(e).__name__}: {e}")
                say(f"[baseline] {sid}: FAIL {e}")
            rec.update(finished_utc=utc_now(), duration_s=round(time.perf_counter() - t0, 3),
                       die_temp_end_c=benv.read_die_temp(*(([str(rd / ".dryrun_env/sys/bus/iio/devices"),
                                                             str(rd / ".dryrun_env/sys/class/hwmon")]) if dry else []))
                       .get("max_c"))
            save_state(sp, state)
    except KeyboardInterrupt:
        say("[baseline] interrupted: state saved, rerun the same command to resume")
        return 130
    finally:
        if not dry:
            errs = ctl.restore()
            say(f"[baseline] cpufreq restored{': ' + '; '.join(errs) if errs else ''}")
    # assemble the per-session CSVs from every OK step of the state (in step order)
    acc = [r for sid, *_ in steps for r in state["steps"].get(sid, {}).get("rows_acc", [])]
    lat = [r for sid, *_ in steps for r in state["steps"].get(sid, {}).get("rows_lat", [])]
    if acc:
        bc.write_csv(rd / "hw_baseline_accuracy.csv", acc, ACC_FIELDS, source)
        bc.write_csv(rd / "hw_baseline_latency.csv", lat, LAT_FIELDS, source)
    flags = [] if dry else login_check(rd, t_inv, say)
    info = {"source": source, "utc": utc_now(), "argv": sys.argv, "args": vars(a), "package_dir": str(pkg_dir),
            "package_manifest_sha256": pkg.manifest_sha256, "package_git_commit": pkg.manifest.get("git_commit"),
            "package_git_dirty": pkg.dirty, "checkpoint_kinds": kinds, "environment": env,
            "overlay": sess.ovinfo, "host": platform.platform(), "python": platform.python_version(),
            "numpy": np.__version__, "ort_version": _ort_version(),
            "steps": {sid: {k: v for k, v in state["steps"].get(sid, {}).items() if k not in ("rows_acc", "rows_lat")}
                      for sid, *_ in steps}, "login_apt_flags": flags}
    p = rd / "hw_baseline_session_info.json"
    bc.check_output_path(p, source)
    p.write_text(json.dumps(info, indent=1, default=str) + "\n")
    say(f"wrote {p}")
    bad = {sid: state["steps"].get(sid, {}).get("status") for sid, *_ in steps
           if state["steps"].get(sid, {}).get("status") != "ok"}
    say(f"[baseline] {'DRY RUN ' if dry else ''}session {a.session} done: {len(steps) - len(bad)}/{len(steps)} steps OK"
        + (f"; not OK: {bad}" if bad else ""))
    return 1 if failures else 0


def _ort_version() -> str:
    try:
        import onnxruntime as ort  # noqa: PLC0415
        return ort.__version__
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"


if __name__ == "__main__":
    sys.exit(main())
