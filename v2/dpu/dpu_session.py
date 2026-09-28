#!/usr/bin/env python3
"""DPU baseline board session (KV260 + Kria-PYNQ + pynq_dpu 2.5): accuracy, latency, power.

On the board (after v2/board/deploy.sh AND v2/dpu/deploy_dpu.sh; files in ~/gos/dpu/):

    source /etc/profile.d/pynq_venv.sh
    cd ~/gos/dpu && sudo -E python3 dpu_session.py                  # both nets, all 10k, + power
    sudo -E python3 dpu_session.py --nets lenet5 --no-power          # accuracy + latency only
    sudo -E python3 dpu_session.py --power-only --phase-s 60 --repeats 3

Laptop dry run (no pynq_dpu; fake runner + mock sensor; writes only under results/dryrun/dpu/):

    .venv/bin/python v2/dpu/dpu_session.py --dry-run --pkg-dir v2/dpu/build/package --limit 200

WHAT IS MEASURED (label on every row): "DPU (Vitis AI 2.5.0, DPUCZDX8G <arch>, fingerprint
<fp>)" -- the AMD DPU running a Vitis AI vai_q_pytorch PTQ quantization of the SAME FP32
networks (DECISIONS D3). It is NOT our INT8 reference and NOT our accelerator; agreement with
our golden INT8 predictions is reported as a comparison only.

Per net (batch 1, x = data package x_f32 = legacy preprocessing, NCHW -> NHWC):
  * accuracy on all N images (default 10,000) vs labels; agreement with our golden predictions
    (golden_pred.npy, INT8 reference) and with the vai_q PyTorch model predictions from the
    container (vaiq_pred.npy)
  * latency per image, warm-up images (default 50) run first and discarded:
      dpu_runner  = execute_async + wait            (VART runner time, PS-measured)
      pre         = float32 -> DPU input dtype into the preallocated input buffer
      post        = output -> argmax
      end_to_end  = pre + dpu_runner + post          (one perf_counter_ns span)
    p50 / p95 / p99 / mean / min / max (ns -> µs)
  * power: power_log.run_power_protocol with the DPU loop (end-to-end, batch 1, images cycled)
    as the "accel" workload; schedule idle/accel/idle x repeats (no CPU phases: the CPU
    baseline is B1's). Label "SOM-rail power (INA260)"; prefix hw_dpu_power_ina260, tag _<net>.

Input conversion: if the DPU input tensor is int8 with fix_point fp, q = clip(floor(x * 2^fp
+ 0.5), -128, 127) (round half up; inputs are >= 0); if it is float, x is copied unchanged.
Output: int8 -> logits = q * 2^-fp_out (argmax is taken on the raw values; identical).
The tensor dtype / fix_point actually used are recorded in every row (assumption to confirm at
bring-up: pynq_dpu 2.5 exposes XINT8 tensors of the compiled xmodel).

Outputs (hw: ~/gos/results/, dry run: results/dryrun/dpu/):
  hw_dpu_accuracy.csv, hw_dpu_latency.csv, hw_dpu_preds_<net>.npz, hw_dpu_session_info.json,
  hw_dpu_power_ina260_{samples,phases,summary}_<net>.csv
Rows follow EXPERIMENTS.md "CSV rule" (board_common.write_csv). source = hw only on the board
with pynq_dpu; dry runs are source = dryrun_model and mean nothing (fake runner, mock sensor).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _find_board_dir() -> Path:
    """Directory holding board_common.py + power_log.py: ~/gos on the board, v2/board on the laptop."""
    env = os.environ.get("GOS_BOARD_DIR")
    cands = [Path(env)] if env else []
    for d in [HERE, *HERE.parents]:
        cands += [d, d / "board", d / "v2" / "board"]
    for c in cands:
        if (c / "board_common.py").is_file() and (c / "power_log.py").is_file():
            return c
    raise SystemExit("cannot find board_common.py + power_log.py (set GOS_BOARD_DIR)")


BOARD_DIR = _find_board_dir()
sys.path.insert(0, str(BOARD_DIR))
import board_common as bc  # noqa: E402
import power_log as pl  # noqa: E402

PREFIX_POWER = "hw_dpu_power_ina260"
DEFAULT_WARMUP = 50
PACKAGE_INFO = "DPU_INFO.json"
DRY_OUT = Path("dryrun") / "dpu"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---- DPU package (built on the laptop by make_dpu_package.py) --------------------------------
class DpuNet:
    """One net of the DPU package: xmodel (sha256 checked), xmodel_info.json, vaiq_pred.npy."""

    def __init__(self, pkg_dir: Path, net: str, info: dict):
        self.net = net
        d = pkg_dir / net
        ent = info["nets"][net]
        self.xmodel = d / ent["xmodel"]
        if not self.xmodel.is_file():
            raise SystemExit(f"missing {self.xmodel}")
        got = sha256_file(self.xmodel)
        if got != ent["xmodel_sha256"]:
            raise SystemExit(f"REFUSED: {self.xmodel} sha256 {got[:12]} != {PACKAGE_INFO} "
                             f"{ent['xmodel_sha256'][:12]}")
        self.xmodel_sha256 = got
        self.xinfo = json.loads((d / "xmodel_info.json").read_text())
        vp = d / "vaiq_pred.npy"
        self.vaiq_pred = np.load(vp) if vp.is_file() else None
        self.fingerprint = ",".join(self.xinfo.get("dpu_fingerprint_hex")
                                    or self.xinfo.get("dpu_fingerprint", [])) or "unknown"
        self.n_dpu = self.xinfo.get("n_dpu_subgraphs")
        self.n_cpu = self.xinfo.get("n_cpu_subgraphs")


def load_dpu_package(pkg_dir: Path, nets) -> tuple[dict, dict]:
    p = pkg_dir / PACKAGE_INFO
    if not p.is_file():
        raise SystemExit(f"missing {p} (build it with make_dpu_package.py, deploy with deploy_dpu.sh)")
    info = json.loads(p.read_text())
    return info, {n: DpuNet(pkg_dir, n, info) for n in nets}


# ---- runner I/O ---------------------------------------------------------------------------
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
        self.out_scale = float(2.0 ** -out_fp) if (self.out_int8 and out_fp is not None) else 1.0
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
                "out_dims": list(self.out_dims), "out_dtype": self.out_dtype,
                "out_fix_point": self.out_fp,
                "input_conversion": (f"int8 clip(floor(x*2^{self.in_fp}+0.5),-128,127)"
                                     if self.in_int8 else "float32 copy")}


def to_nhwc(x_nchw: np.ndarray) -> np.ndarray:
    """(N,C,H,W) float32 -> contiguous (N,H,W,C) float32 (done once, outside the timed loop)."""
    return np.ascontiguousarray(np.transpose(x_nchw, (0, 2, 3, 1)))


def fill_input(buf: np.ndarray, x_hwc: np.ndarray, io: IoSpec):
    """One image (H,W,C float32) into the preallocated input buffer (1,H,W,C)."""
    if io.in_int8:
        q = np.floor(x_hwc * np.float32(io.in_scale) + np.float32(0.5))
        buf[0] = np.clip(q, -128, 127).astype(np.int8)
    else:
        buf[0] = x_hwc


class FakeRunner:
    """Dry-run stand-in for the VART runner: 'computes' a one-hot int8 output of a given
    prediction vector (pipeline test only; numbers are meaningless)."""

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


# ---- measurement ----------------------------------------------------------------------------
def run_images(runner, io: IoSpec, x_nhwc: np.ndarray, indices, record: bool = True):
    """Batch-1 loop. Returns preds, raw outputs and per-image ns arrays (pre, dpu, post, e2e)."""
    inp, out = io.buffers()
    n = len(indices)
    preds = np.empty(n, dtype=np.int64)
    raw = np.empty((n,) + io.out_dims[1:], dtype=out[0].dtype) if record else None
    t = np.empty((4, n), dtype=np.int64)
    pc = time.perf_counter_ns
    for j, i in enumerate(indices):
        t0 = pc()
        fill_input(inp[0], x_nhwc[i], io)
        t1 = pc()
        job = runner.execute_async(inp, out)
        runner.wait(job)
        t2 = pc()
        p = int(np.argmax(out[0].reshape(-1)))
        t3 = pc()
        preds[j] = p
        if record:
            raw[j] = out[0][0]
        t[0, j], t[1, j], t[2, j], t[3, j] = t1 - t0, t2 - t1, t3 - t2, t3 - t0
    return preds, raw, {"pre": t[0], "dpu_runner": t[1], "post": t[2], "end_to_end": t[3]}


def dpu_workload(runner, io: IoSpec, x_nhwc: np.ndarray, warmup: int = 1):
    """power_log accel callable: back-to-back end-to-end batch-1 inferences, images cycled."""
    inp, out = io.buffers()
    n = x_nhwc.shape[0]

    def step(k):
        fill_input(inp[0], x_nhwc[k % n], io)
        runner.wait(runner.execute_async(inp, out))
        return int(np.argmax(out[0].reshape(-1)))

    for k in range(warmup):
        step(k)
    return lambda deadline: pl.loop_until(step, deadline)


# ---- provenance / rows ---------------------------------------------------------------------
class DpuContext:
    """Row metadata (same columns as board_common.RunContext.meta) for DPU runs; also the ctx
    object power_log.run_power_protocol expects (source, out_dir, meta)."""

    def __init__(self, a, dpu_info: dict, overlay_info: dict):
        self.args = a
        self.source = bc.SOURCE_DRYRUN if a.dry_run else bc.SOURCE_HW
        out = a.out_dir
        if out is None:
            out = bc.RESULTS_ROOT / DRY_OUT if a.dry_run else bc.RESULTS_ROOT
        self.out_dir = bc.resolve_out_dir(self.source, out)
        self.dpu_info = dpu_info
        self.ov = overlay_info
        self.scripts_commit = dpu_info.get("git_commit", "")
        self.scripts_dirty = bool(dpu_info.get("git_dirty", True))
        self.hostname = socket.gethostname()
        self.board_id = a.board_id or (bc.board_id_default() if not a.dry_run else "")
        self.packages: dict = {}
        self.clock_source = ("not read (dry run)" if a.dry_run else
                             "pynq Clocks.fclk0_mhz readback (pl_clk0; the DPU core clock is "
                             "generated inside the overlay and not read)")

    def package(self, net: str):
        if net not in self.packages:
            self.packages[net] = bc.load_package(self.args.data_dir, net)
        return self.packages[net]

    @property
    def dirty(self) -> bool:
        return self.scripts_dirty or any(bool(p.manifest.get("git_dirty", True))
                                         for p in self.packages.values())

    def check_clean(self):
        if self.source == bc.SOURCE_HW and self.dirty and not self.args.allow_dirty:
            raise SystemExit("REFUSED: DPU package or data package from a dirty tree "
                             f"(DPU_INFO git_dirty={self.scripts_dirty}); rebuild from a clean "
                             "commit or pass --allow-dirty (rows then invalid for the paper)")

    def clock(self):
        return "" if self.args.dry_run else self.ov.get("fclk0_mhz", "")

    def meta(self, net: str = "", layer: str = "", duration_s="", num_inferences="",
             clock_mhz=None) -> dict:
        pkg = self.packages.get(net)
        clk = self.clock() if clock_mhz is None else clock_mhz
        return {"timestamp": bc.utc_now(), "git_commit": self.scripts_commit,
                "git_dirty": self.dirty, "vivado_version": "",
                "bitstream_sha256": self.ov.get("bit_sha256", ""), "board_id": self.board_id,
                "net": net, "layer": layer,
                "clock_mhz": f"{clk:.6f}" if isinstance(clk, float) else clk,
                "source": self.source,
                "duration_s": duration_s if duration_s == "" else f"{float(duration_s):.3f}",
                "num_inferences": num_inferences, "build_id_hw": "",
                "board_hostname": self.hostname, "clock_source": self.clock_source,
                "scripts_commit": self.scripts_commit,
                "data_manifest_sha256": pkg.manifest_sha256 if pkg else "",
                "data_git_commit": pkg.manifest.get("git_commit", "") if pkg else "",
                "data_git_dirty": pkg.manifest.get("git_dirty", "") if pkg else "",
                "backend": "fake_runner (dry run)" if self.args.dry_run else "pynq_dpu"}


def dpu_label(info: dict, dn: DpuNet) -> str:
    return (f"DPU (Vitis AI {info.get('vitis_ai_version', '?')}, "
            f"{info.get('dpu_arch', 'DPUCZDX8G')}, fingerprint {dn.fingerprint})")


def dpu_cols(ctx: DpuContext, dn: DpuNet, io: IoSpec) -> dict:
    i = ctx.dpu_info
    return {"measurement": dpu_label(i, dn), "vitis_ai_version": i.get("vitis_ai_version", ""),
            "docker_image": i.get("docker_image", ""), "dpu_arch": i.get("dpu_arch", ""),
            "dpu_fingerprint": dn.fingerprint, "hw_dpu_fingerprint": ctx.ov.get("hw_fingerprint", ""),
            "quantizer": i.get("quantizer", ""), "xmodel_sha256": dn.xmodel_sha256,
            "dpu_subgraphs": dn.n_dpu, "cpu_subgraphs": dn.n_cpu,
            "overlay": ctx.ov.get("overlay", ""), "pynq_dpu_version": ctx.ov.get("pynq_dpu_version", ""),
            "io": json.dumps(io.describe(), sort_keys=True)}


DPU_FIELDS = ["measurement", "vitis_ai_version", "docker_image", "dpu_arch", "dpu_fingerprint",
              "hw_dpu_fingerprint", "quantizer", "xmodel_sha256", "dpu_subgraphs", "cpu_subgraphs",
              "overlay", "pynq_dpu_version", "io"]
ACC_FIELDS = DPU_FIELDS + ["images", "correct", "accuracy_pct", "golden_correct",
                           "golden_accuracy_pct", "agree_golden", "agree_golden_pct",
                           "agree_vaiq_model", "agree_vaiq_model_pct", "vaiq_model_correct",
                           "warmup_images", "batch"]
LAT_FIELDS = DPU_FIELDS + ["metric", "unit", "n", "warmup_images", "batch", "p50", "p95", "p99",
                           "mean", "min", "max"]


# ---- overlay ---------------------------------------------------------------------------------
def open_overlay(a, expected_md5: str = ""):
    """(overlay object, info dict). pynq_dpu only on the board."""
    from pynq_dpu import DpuOverlay  # noqa: PLC0415
    import pynq_dpu  # noqa: PLC0415
    ov = DpuOverlay(a.overlay)
    bit = Path(a.overlay)
    if not bit.is_absolute():
        bit = Path(pynq_dpu.__file__).resolve().parent / a.overlay
    md5 = hashlib.md5(bit.read_bytes()).hexdigest() if bit.is_file() else ""
    info = {"overlay": str(bit), "bit_sha256": sha256_file(bit) if bit.is_file() else "",
            "bit_md5": md5, "bit_md5_expected": expected_md5,
            "pynq_dpu_version": getattr(pynq_dpu, "__version__", "")}
    if expected_md5 and md5 != expected_md5:
        msg = (f"overlay {bit} md5 {md5} != prebuilt pynq-dpu 2.5 KV260 overlay {expected_md5}")
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
            fps = [str(k.get("fingerprint", "")) for k in j.get("kernels", [])]
            info["hw_fingerprint"] = ",".join(f for f in fps if f)
        except ValueError:
            pass
    except Exception as e:  # noqa: BLE001
        info["xdputil_error"] = str(e)
    return ov, info


def make_runner(a, ov, dn: DpuNet, pkg):
    if a.dry_run:
        io = IoSpec.from_info(dn.xinfo)
        preds = dn.vaiq_pred if dn.vaiq_pred is not None else pkg.golden_pred
        return FakeRunner(io, preds), io
    ov.load_model(str(dn.xmodel))
    r = ov.runner
    return r, IoSpec.from_runner(r)


# ---- per-net steps -------------------------------------------------------------------------
def pct(c, n):
    return f"{100.0 * c / n:.2f}" if n else ""


def measure_net(a, ctx: DpuContext, ov, dn: DpuNet):
    pkg = ctx.package(dn.net)
    n = pkg.n if a.limit is None else min(a.limit, pkg.n)
    x = to_nhwc(pkg.x_f32[:n])
    runner, io = make_runner(a, ov, dn, pkg)
    exp = tuple(io.in_dims[1:])
    if x.shape[1:] != exp:
        raise SystemExit(f"[{dn.net}] input NHWC {x.shape[1:]} != DPU input {exp}")
    lab = dpu_label(ctx.dpu_info, dn)
    print(f"[{dn.net}] {lab}; io {io.describe()}; images {n}, warm-up {a.warmup}", flush=True)
    if a.dry_run:
        print(f"[{dn.net}] DRY RUN: fake runner — accuracy/latency below are NOT measurements.")
    run_images(runner, io, x, [k % n for k in range(a.warmup)], record=False)   # discarded
    if isinstance(runner, FakeRunner):
        runner.k = 0                  # dry run: replay predictions from image 0
    t0 = time.perf_counter()
    preds, raw, t = run_images(runner, io, x, range(n))
    dur = time.perf_counter() - t0
    y, g = pkg.labels[:n], pkg.golden_pred[:n]
    correct, gcorrect = int((preds == y).sum()), int((g == y).sum())
    agree_g = int((preds == g).sum())
    vq = dn.vaiq_pred[:n] if dn.vaiq_pred is not None else None
    agree_v = int((preds == vq).sum()) if vq is not None else ""
    base = dpu_cols(ctx, dn, io)
    acc = ctx.meta(dn.net, "all", dur, n)
    acc.update(base, images=n, correct=correct, accuracy_pct=pct(correct, n),
               golden_correct=gcorrect, golden_accuracy_pct=pct(gcorrect, n),
               agree_golden=agree_g, agree_golden_pct=pct(agree_g, n),
               agree_vaiq_model=agree_v, agree_vaiq_model_pct=pct(agree_v, n) if vq is not None else "",
               vaiq_model_correct=int((vq == y).sum()) if vq is not None else "",
               warmup_images=a.warmup, batch=1)
    lat = []
    for metric in ("dpu_runner", "pre", "post", "end_to_end"):
        s = bc.percentiles(t[metric] / 1e3, ps=(50, 95, 99))
        r = ctx.meta(dn.net, "all", dur, n)
        r.update(base, metric=metric, unit="us", n=s["n"], warmup_images=a.warmup, batch=1,
                 **{k: f"{s[k]:.3f}" for k in ("p50", "p95", "p99", "mean", "min", "max")})
        lat.append(r)
        print(f"[{dn.net}] {metric:10s} p50 {s['p50']:.1f} us  p95 {s['p95']:.1f} us  "
              f"({lab}{', DRY RUN' if a.dry_run else ''})")
    print(f"[{dn.net}] accuracy {correct}/{n} = {pct(correct, n)}%; agreement with our golden "
          f"INT8 {agree_g}/{n}; with vai_q model {agree_v}/{n if vq is not None else '-'}")
    npz = ctx.out_dir / f"hw_dpu_preds_{dn.net}.npz"
    bc.check_output_path(npz, ctx.source)
    np.savez(npz, pred=preds, raw_out=raw, labels=y, golden_pred=g,
             **{f"t_{k}_ns": v for k, v in t.items()})
    return [acc], lat, (runner, io, x)


def power_net(a, ctx: DpuContext, dn: DpuNet, runner, io, x):
    wl = dpu_workload(runner, io, x, warmup=max(1, a.warmup))
    return pl.run_power_protocol(
        ctx, dn.net, None, accel_fn=wl, sensor=a.sensor, durations={}, phase_s=a.phase_s,
        repeats=a.repeats, rate_hz=a.rate_hz, final_idle=False, prefix=PREFIX_POWER,
        tag=f"_{dn.net}", clock_mhz=ctx.clock(),
        accel_label=f"{dpu_label(ctx.dpu_info, dn)} execute_async+wait, batch 1, end-to-end, "
                    f"{dn.net} images cycled", label="DPU")


# ---- CLI -------------------------------------------------------------------------------------
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="laptop: no pynq_dpu (fake runner + mock sensor), output under dryrun/")
    ap.add_argument("--pkg-dir", default=str(HERE),
                    help=f"DPU package dir ({PACKAGE_INFO} + <net>/*.xmodel); default: this dir")
    ap.add_argument("--data-dir", default=str(bc.DEFAULT_DATA_DIR), help="board data package")
    ap.add_argument("--nets", nargs="+", default=list(bc.NETS), choices=bc.NETS)
    ap.add_argument("--overlay", default="dpu.bit", help="pynq_dpu overlay (default: the "
                    "package's dpu.bit for the KV260)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N images (default: all)")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-power", action="store_true", help="accuracy + latency only")
    g.add_argument("--power-only", action="store_true", help="power protocol only")
    ap.add_argument("--sensor", default="auto", choices=("auto",) + pl.SENSOR_ORDER + ("mock",))
    ap.add_argument("--phase-s", type=float, default=None,
                    help=f"power phase seconds (default {pl.DEFAULT_PHASE_S:g}; dry run 1)")
    ap.add_argument("--repeats", type=int, default=None,
                    help=f"power repeats (default {pl.DEFAULT_REPEATS}; dry run 1)")
    ap.add_argument("--rate-hz", type=float, default=pl.DEFAULT_RATE_HZ)
    ap.add_argument("--board-id", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--allow-overlay-mismatch", action="store_true",
                    help="run although dpu.bit is not the prebuilt pynq-dpu 2.5 KV260 overlay")
    a = ap.parse_args(argv)
    if a.phase_s is None:
        a.phase_s = 1.0 if a.dry_run else pl.DEFAULT_PHASE_S
    if a.repeats is None:
        a.repeats = 1 if a.dry_run else pl.DEFAULT_REPEATS
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be >= 1")
    if a.warmup < 0 or a.repeats < 1 or a.phase_s <= 0 or a.rate_hz <= 0:
        ap.error("--warmup >= 0, --repeats >= 1, --phase-s > 0, --rate-hz > 0 required")
    return a


def main(argv=None) -> int:
    a = parse(argv)
    info, nets = load_dpu_package(Path(a.pkg_dir), a.nets)
    if a.dry_run:
        ov, ovinfo = None, {"overlay": "none (dry run)"}
    else:
        ov, ovinfo = open_overlay(a, info.get("overlay_bit_md5_expected", ""))
    ctx = DpuContext(a, info, ovinfo)
    for n in a.nets:
        ctx.package(n)
    ctx.check_clean()
    print(f"[dpu_session] source={ctx.source} out={ctx.out_dir} package={a.pkg_dir} "
          f"VAI {info.get('vitis_ai_version')} arch {info.get('dpu_arch')} commit "
          f"{ctx.scripts_commit[:8]} dirty={ctx.dirty} overlay={ovinfo.get('overlay')}")
    acc, lat, power = [], [], {}
    for n in a.nets:
        dn = nets[n]
        if a.power_only:
            pkg = ctx.package(n)
            runner, io = make_runner(a, ov, dn, pkg)
            ctx_x = to_nhwc(pkg.x_f32[: (a.limit or pkg.n)])
            power[n] = power_net(a, ctx, dn, runner, io, ctx_x)["files"]
            continue
        r_acc, r_lat, (runner, io, x) = measure_net(a, ctx, ov, dn)
        acc += r_acc
        lat += r_lat
        if not a.no_power:
            power[n] = power_net(a, ctx, dn, runner, io, x)["files"]
    if acc:
        bc.write_csv(ctx.out_dir / "hw_dpu_accuracy.csv", acc, ACC_FIELDS, ctx.source)
        bc.write_csv(ctx.out_dir / "hw_dpu_latency.csv", lat, LAT_FIELDS, ctx.source)
    sess = {"source": ctx.source, "utc": bc.utc_now(), "argv": sys.argv, "args": vars(a),
            "dpu_info": info, "overlay": ovinfo, "host": platform.platform(),
            "xmodel_sha256": {n: nets[n].xmodel_sha256 for n in a.nets}, "power_files": power}
    p = ctx.out_dir / "hw_dpu_session_info.json"
    bc.check_output_path(p, ctx.source)
    p.write_text(json.dumps(sess, indent=1, default=str) + "\n")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
