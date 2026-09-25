#!/usr/bin/env python3
"""V2 Step 5, EXPERIMENTS.md A5: same-board CPU baseline (latency + accuracy).

Runs on the KV260 (``--tag board``, numpy only; onnxruntime optional) or on the
laptop as a dry run (``--tag laptop``, output only under v2/results/dryrun/).

    python3 run_cpu_baselines.py --data-dir <dir with <net>/cpu/> --out-dir <dir> \
        --tag {board,laptop} [--nets lenet5 cifar10] [--kinds ...] [--threads 1 4] \
        [--runs 100] [--warmup 10] [--quick]

Protocol (A5): per net x kind x threads, in its own subprocess with
OPENBLAS_NUM_THREADS = OMP_NUM_THREADS = MKL_NUM_THREADS = threads set before numpy
is imported (ORT: intra_op_num_threads = threads):
  * mode=compute : pre-loaded preprocessed tensor (int8 NCHW for cpu_int8_ref,
                   float32 NCHW otherwise) -> float32 logits, one image per call;
  * mode=e2e     : raw stored dataset image (uint8) -> preprocessing (and, for
                   cpu_int8_ref, input quantization) -> logits -> argmax prediction;
  * warm-up calls discarded, then ``--runs`` timed calls (>= 100 for paper data),
    cycling over the first ``--runs`` test images; median, p5, p95 reported;
  * mode=accuracy: separate pass (not timed) over the test set (10k; --quick: 500),
    at threads = 1; cpu_int8_ref is also compared bit-exactly with the package's
    golden logits when available.
A kind that cannot run (e.g. onnxruntime missing) gets rows with status
"unavailable: <reason>" instead of failing the script.

Output: <out-dir>/hw_cpu_baseline.csv (EXPERIMENTS.md CSV-rule metadata columns
first) and <out-dir>/hw_cpu_baseline_env.json (full environment record).
source = cpu_board (--tag board) / cpu_laptop (--tag laptop). Laptop numbers are
not paper data.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KINDS = ("cpu_int8_ref", "cpu_fp32_numpy", "cpu_ort_fp32", "cpu_ort_int8")
NETS = ("lenet5", "cifar10")

META_COLUMNS = ("timestamp", "git_commit", "git_dirty", "vivado_version", "bitstream_sha256",
                "board_id", "net", "layer", "clock_mhz", "source", "duration_s",
                "num_inferences")
FIELDS = ("kind", "label", "threads", "mode", "median_us", "p5_us", "p95_us", "mean_us",
          "runs", "warmup", "accuracy", "correct", "n_images", "acc_batch",
          "golden_bitexact", "package_crosscheck", "status", "numpy_version", "blas", "ort_version",
          "cpu_model", "hostname", "python_version", "package_sha256",
          "data_manifest_sha256", "versions")


# --------------------------------------------------------------------------- #
# Worker (one net x kind x threads, fresh process)
# --------------------------------------------------------------------------- #
def _percentiles(ts_ns):
    import numpy as np
    a = np.asarray(ts_ns, dtype=np.float64) / 1e3
    return {"median_us": float(np.median(a)), "p5_us": float(np.percentile(a, 5)),
            "p95_us": float(np.percentile(a, 95)), "mean_us": float(a.mean())}


def _time_calls(fn, xs, runs: int, warmup: int):
    n = len(xs)
    for i in range(warmup):
        fn(xs[i % n])
    ts = []
    pc = time.perf_counter_ns
    for i in range(runs):
        x = xs[i % n]
        t0 = pc()
        fn(x)
        ts.append(pc() - t0)
    return ts


def _load_golden(data_dir: Path, net: str):
    """Golden raw INT32 / float32 logits from the board data package (make_board_data.py),
    if present. Returns (array, description) or (None, reason)."""
    import numpy as np
    p = data_dir / net / "golden_logits.npy"
    if not p.exists():
        return None, "golden_logits.npy not in package"
    return np.load(p), str(p.name)


def _package_crosscheck(data_dir: Path, net: str, r, x_u8, y) -> dict:
    """Compare the CPU package's raw images / labels with the board data package
    (make_board_data.py: labels.npy, x_nchw int8, x_f32 float32), when present."""
    import numpy as np
    n = len(y)
    out, ok = [], True
    lp = data_dir / net / "labels.npy"
    if lp.exists():
        lab = np.load(lp)[:n]
        k = int((lab == y[:len(lab)]).sum())
        ok &= k == len(lab)
        out.append(f"labels {k}/{len(lab)}")
    ip = data_dir / net / "inputs_act.npz"
    if ip.exists():
        z = np.load(ip)
        pre = r.preprocess(x_u8)
        key = "x_nchw" if pre.dtype == np.int8 else "x_f32"
        if key in z.files:
            ref = z[key][:n]
            m = len(ref)
            eq = np.all(pre[:m].view(np.uint8).reshape(m, -1) ==
                        np.ascontiguousarray(ref).view(np.uint8).reshape(m, -1), axis=1)
            ok &= bool(eq.all())
            out.append(f"{key} {int(eq.sum())}/{m}")
    return {"package_crosscheck": "; ".join(out) if out else "board data package not present",
            "_ok": ok}


def worker(args) -> dict:
    import numpy as np
    sys.path.insert(0, str(HERE))
    import cpu_infer
    data_dir = Path(args.data_dir)
    res = {"net": args.net, "kind": args.kind, "threads": args.threads, "rows": []}
    try:
        r = cpu_infer.make_runner(args.kind, args.net, data_dir, args.threads)
    except cpu_infer.RunnerUnavailable as e:
        res["status"] = f"unavailable: {e}"
        return res
    res["status"] = "ok"
    res["label"] = r.label
    raw = np.load(cpu_infer.cpu_dir(data_dir, args.net) / "raw_test.npz")
    x_u8, y = raw["x_u8"], raw["y"]
    nt = min(args.runs, len(x_u8))
    pre = [np.ascontiguousarray(r.preprocess(x_u8[i])) for i in range(nt)]
    raws = [np.ascontiguousarray(x_u8[i]) for i in range(nt)]
    if args.timing:
        for mode, fn, xs in (("compute", r.compute, pre), ("e2e", r.e2e, raws)):
            t0 = time.perf_counter()
            ts = _time_calls(fn, xs, args.runs, args.warmup)
            res["rows"].append({"mode": mode, "runs": args.runs, "warmup": args.warmup,
                                "duration_s": round(time.perf_counter() - t0, 3),
                                "num_inferences": args.runs + args.warmup, **_percentiles(ts)})
    if args.accuracy:
        n = min(args.acc_n, len(x_u8))
        t0 = time.perf_counter()
        # cpu_int8_ref is exact integer arithmetic: batching cannot change results.
        # Float kinds run one image per call, as timed.
        bs = args.acc_batch if args.kind == "cpu_int8_ref" else 1
        preds = np.empty(n, dtype=np.int64)
        extra = {}
        if args.kind == "cpu_int8_ref":
            gold, gdesc = _load_golden(data_dir, args.net)
            eq = 0
            for s in range(0, n, bs):
                xq = r.preprocess(x_u8[s:s + bs])
                v = r.raw_v(xq)
                lg = r.logits_from_raw(v)
                preds[s:s + bs] = lg.argmax(1)
                if gold is not None and s < len(gold):
                    g = gold[s:s + bs]
                    m = len(g)
                    if g.dtype == np.int32:
                        eq += int(np.all(v[:m] == g, axis=1).sum())
                    else:
                        g = g.astype(np.float32)
                        eq += int(np.all(lg[:m].view(np.uint32) == g.view(np.uint32), axis=1).sum())
            if gold is not None:
                extra["golden_bitexact"] = f"{eq}/{min(n, len(gold))} ({gdesc})"
                extra["_ok"] = eq == min(n, len(gold))
            else:
                extra["golden_bitexact"] = gdesc
        else:
            for i in range(n):
                preds[i] = int(r.compute(r.preprocess(x_u8[i])).argmax())
        gold_ok = extra.pop("_ok", True)
        extra.update(_package_crosscheck(data_dir, args.net, r, x_u8[:n], y[:n]))
        extra["status"] = "ok" if (gold_ok and extra.pop("_ok")) else "FAIL: package mismatch"
        correct = int((preds == y[:n]).sum())
        res["rows"].append({"mode": "accuracy", "duration_s": round(time.perf_counter() - t0, 3),
                            "num_inferences": n, "n_images": n, "correct": correct,
                            "accuracy": f"{100.0 * correct / n:.2f}", "acc_batch": bs, **extra})
    return res


# --------------------------------------------------------------------------- #
# Environment / provenance
# --------------------------------------------------------------------------- #
def cpu_model(txt: str | None = None) -> str:
    if txt is None:
        try:
            txt = Path("/proc/cpuinfo").read_text()
        except OSError:
            return platform.processor()
    for key in ("model name", "Model", "Hardware"):
        for line in txt.splitlines():
            if line.split(":")[0].strip() == key:
                return line.split(":", 1)[1].strip()
    parts = {}
    for line in txt.splitlines():  # arm64: no model name -> CPU implementer/part
        if ":" in line:
            k, v = (s.strip() for s in line.split(":", 1))
            if k in ("CPU implementer", "CPU part", "CPU architecture") and k not in parts:
                parts[k] = v
    ncpu = txt.count("processor\t:") or txt.count("processor :")
    names = {("0x41", "0xd03"): "ARM Cortex-A53"}
    nm = names.get((parts.get("CPU implementer"), parts.get("CPU part")), "")
    return f"{nm} ({', '.join(f'{k}={v}' for k, v in parts.items())}; {ncpu} cpus)".strip()


def env_record() -> dict:
    import io
    import contextlib
    import numpy as np
    rec = {"numpy": np.__version__, "python": platform.python_version(),
           "platform": platform.platform(), "machine": platform.machine(),
           "hostname": socket.gethostname(), "cpu_model": cpu_model(),
           "nproc": os.cpu_count()}
    buf = io.StringIO()
    import warnings
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            np.show_config()
        except Exception as e:
            print(f"show_config failed: {e}")
    rec["numpy_show_config"] = buf.getvalue()
    blas = ""
    try:
        from threadpoolctl import threadpool_info
        info = threadpool_info()
        rec["threadpoolctl"] = info
        blas = "; ".join(f"{i.get('internal_api')} {i.get('version')} {i.get('architecture', '')}"
                         f" threads={i.get('num_threads')}" for i in info if i.get("user_api") == "blas")
    except Exception:
        rec["threadpoolctl"] = "unavailable"
    if not blas:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cfg = np.show_config(mode="dicts")
            b = cfg.get("Build Dependencies", {}).get("blas", {})
            blas = f"{b.get('name', '')} {b.get('version', '')}".strip()
        except Exception:
            for line in rec["numpy_show_config"].splitlines():
                if "libraries" in line and ("blas" in line.lower() or "lapack" in line.lower()):
                    blas = line.strip()
                    break
    rec["blas"] = blas or "unknown"
    try:
        import onnxruntime as ort
        rec["onnxruntime"] = ort.__version__
    except Exception as e:
        rec["onnxruntime"] = f"unavailable ({type(e).__name__})"
    return rec


def find_deploy_info(data_dir: Path):
    for p in (data_dir / "DEPLOY_INFO.json", data_dir.parent / "DEPLOY_INFO.json",
              HERE / "DEPLOY_INFO.json", HERE.parent / "DEPLOY_INFO.json"):
        if p.exists():
            return p
    return None


def _first(d: dict, *keys, default=""):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def board_id_default() -> str:
    """Same rule as v2/board/board_common.board_id_default (GOS_BOARD_ID, else
    device-tree model + machine-id prefix)."""
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


def provenance(data_dir: Path, board_id: str | None) -> dict:
    """Commit of the deployed code: DEPLOY_INFO.json (board) if present, else git.
    bitstream_sha256 stays empty: the CPU baseline does not use the PL."""
    p = find_deploy_info(data_dir)
    if p is not None:
        di = json.loads(p.read_text())
        return {"git_commit": _first(di, "git_commit", "commit"),
                "git_dirty": _first(di, "git_dirty", "dirty"),
                "board_id": board_id or _first(di, "board_id") or board_id_default(),
                "commit_source": str(p)}
    try:
        g = lambda *a: subprocess.run(["git", "-C", str(HERE), *a], capture_output=True,  # noqa: E731
                                      text=True, check=True).stdout.strip()
        # generated outputs excluded, as v2/model/common.OUTPUT_PATHSPECS
        dirty = bool(g("status", "--porcelain", "--", ":(top)",
                       ":(top,exclude)v2/results", ":(top,exclude)v2/vectors/MANIFEST.json"))
        return {"git_commit": g("rev-parse", "HEAD"), "git_dirty": dirty,
                "board_id": board_id or board_id_default(), "commit_source": "git"}
    except Exception as e:
        return {"git_commit": "", "git_dirty": "", "board_id": board_id or board_id_default(),
                "commit_source": f"none ({type(e).__name__})"}


def package_sha(data_dir: Path, net: str) -> str:
    """sha256 of the net's CPU_MANIFEST.json (which lists every package file's sha256)."""
    p = data_dir / net / "cpu" / "CPU_MANIFEST.json"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def data_manifest_sha(data_dir: Path, net: str) -> str:
    """sha256 of the board data package MANIFEST.json (make_board_data.py), if present."""
    p = data_dir / net / "MANIFEST.json"
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def check_package(data_dir: Path, net: str) -> list[str]:
    """Verify file sha256s against CPU_MANIFEST.json. Returns problems."""
    d = data_dir / net / "cpu"
    mp = d / "CPU_MANIFEST.json"
    if not mp.exists():
        return [f"{mp} missing (run export_cpu_models.py on the laptop and copy it)"]
    man = json.loads(mp.read_text())
    bad = []
    for name, info in man.get("files", {}).items():
        f = d / name
        if not f.exists():
            bad.append(f"{f} missing")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != info["sha256"]:
            bad.append(f"{f} sha256 mismatch")
    return bad


# --------------------------------------------------------------------------- #
def run_worker_subprocess(a, net, kind, threads, timing, accuracy) -> dict:
    env = dict(os.environ)
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = str(threads)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           "--data-dir", str(a.data_dir), "--net", net, "--kind", kind,
           "--threads", str(threads), "--runs", str(a.runs), "--warmup", str(a.warmup),
           "--acc-n", str(a.acc_n), "--acc-batch", str(a.acc_batch)]
    cmd += ["--timing"] if timing else []
    cmd += ["--accuracy"] if accuracy else []
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        return {"status": f"error: rc={p.returncode}: {p.stderr.strip().splitlines()[-1:] }",
                "rows": [], "stderr": p.stderr}
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(HERE.parent / "data"),
                    help="directory holding <net>/cpu/ (default: ../data next to this script)")
    ap.add_argument("--out-dir")
    ap.add_argument("--tag", choices=("board", "laptop"))
    ap.add_argument("--nets", nargs="+", default=list(NETS), choices=NETS)
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--quick", action="store_true",
                    help="accuracy pass on 500 images only (timing protocol unchanged)")
    ap.add_argument("--acc-batch", type=int, default=100,
                    help="batch of the cpu_int8_ref accuracy pass (exact integers: result "
                         "independent of batching); float kinds always run 1 image per call")
    ap.add_argument("--no-accuracy", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="--tag board: run although the deployed code or a CPU package is from a "
                         "dirty tree (rows are then invalid for the paper)")
    ap.add_argument("--board-id", default=None, help="default: GOS_BOARD_ID or device-tree model + machine-id")
    # worker-only
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--net", help=argparse.SUPPRESS)
    ap.add_argument("--kind", help=argparse.SUPPRESS)
    ap.add_argument("--timing", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--accuracy", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--acc-n", type=int, default=10000, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker:
        a.threads = a.threads[0]
        print(json.dumps(worker(a)))
        return 0

    if not a.out_dir or not a.tag:
        ap.error("--out-dir and --tag are required")
    data_dir = Path(a.data_dir).resolve()
    out_dir = Path(a.out_dir).resolve()
    results = (HERE.parents[1] / "results").resolve()
    dryrun = results / "dryrun"
    if a.tag == "laptop" and not (out_dir == dryrun or dryrun in out_dir.parents):
        print(f"refusing: --tag laptop writes only under {dryrun} (got {out_dir})", file=sys.stderr)
        return 2
    if a.tag == "board" and platform.machine() not in ("aarch64", "arm64"):
        print(f"refusing: --tag board on a {platform.machine()} host", file=sys.stderr)
        return 2
    if a.runs < 100:
        print(f"WARNING: --runs {a.runs} < 100 (A5 protocol requires >= 100)", file=sys.stderr)
    a.acc_n = 500 if a.quick else 10000
    a.data_dir = data_dir
    source = "cpu_board" if a.tag == "board" else "cpu_laptop"

    env = env_record()
    prov = provenance(data_dir, a.board_id)
    print(f"[{source}] {env['hostname']} | {env['cpu_model']} | numpy {env['numpy']} | "
          f"BLAS {env['blas']} | ORT {env['onnxruntime']} | commit {prov['git_commit'][:12]} "
          f"dirty={prov['git_dirty']} ({prov['commit_source']})", flush=True)

    if a.tag == "board" and not a.allow_dirty:
        dirty = [f"deployed code ({prov['commit_source']})"] if prov["git_dirty"] in (True, "True", "") else []
        for net in a.nets:
            mp = data_dir / net / "cpu" / "CPU_MANIFEST.json"
            if mp.exists() and json.loads(mp.read_text()).get("git_dirty", True):
                dirty.append(f"{mp} (exported from a dirty tree)")
        if dirty:
            print("refusing (--allow-dirty to override; rows would be invalid for the paper): "
                  + "; ".join(dirty), file=sys.stderr)
            return 2

    common = {"vivado_version": "", "clock_mhz": "", "layer": "all", "source": source,
              "git_commit": prov["git_commit"], "git_dirty": prov["git_dirty"],
              "bitstream_sha256": "", "board_id": prov["board_id"],
              "numpy_version": env["numpy"], "blas": env["blas"],
              "ort_version": env["onnxruntime"], "cpu_model": env["cpu_model"],
              "hostname": env["hostname"], "python_version": env["python"]}
    rows, failures = [], 0
    t_all = time.perf_counter()
    for net in a.nets:
        probs = check_package(data_dir, net)
        if probs:
            print(f"ERROR {net}: " + "; ".join(probs), file=sys.stderr)
            failures += 1
            continue
        man = json.loads((data_dir / net / "cpu" / "CPU_MANIFEST.json").read_text())
        psha = package_sha(data_dir, net)
        for kind in a.kinds:
            for th in a.threads:
                acc = (not a.no_accuracy) and th == a.threads[0]
                res = run_worker_subprocess(a, net, kind, th, timing=True, accuracy=acc)
                label = res.get("label", "")
                ver = {"export": man.get("versions", {}), "reference_version":
                       man.get("reference_version", "")}
                base = {**common, "net": net, "kind": kind, "label": label, "threads": th,
                        "package_sha256": psha,
                        "data_manifest_sha256": data_manifest_sha(data_dir, net), "versions": json.dumps(ver, sort_keys=True)}
                if res["status"] != "ok":
                    print(f"  {net:8s} {kind:15s} t={th}: {res['status']}", flush=True)
                    if res["status"].startswith("error"):
                        failures += 1
                        print(res.get("stderr", ""), file=sys.stderr)
                    for mode in ("compute", "e2e"):
                        rows.append({**base, "mode": mode, "status": res["status"],
                                     "timestamp": now()})
                    continue
                for r in res["rows"]:
                    row = {**base, "status": "ok", **r, "timestamp": now()}
                    if row["status"] != "ok":
                        failures += 1
                        print(f"  {net:8s} {kind:15s} t={th}: {row['status']}", flush=True)
                    if r["mode"] == "accuracy":
                        row["threads"] = th
                        print(f"  {net:8s} {kind:15s} t={th} accuracy  {r['correct']}/{r['n_images']}"
                              f" = {r['accuracy']}%  golden: {r.get('golden_bitexact', '-')}  "
                              f"pkg: {r.get('package_crosscheck', '')}", flush=True)
                    else:
                        print(f"  {net:8s} {kind:15s} t={th} {r['mode']:8s} median "
                              f"{r['median_us']:10.1f} us  p5 {r['p5_us']:10.1f}  "
                              f"p95 {r['p95_us']:10.1f}  (runs {r['runs']})", flush=True)
                    rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(META_COLUMNS) + [f for f in FIELDS if f not in META_COLUMNS]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else v  # noqa: E731
    csv_path = out_dir / "hw_cpu_baseline.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: fmt(r.get(c, "")) for c in cols})
    env_path = out_dir / "hw_cpu_baseline_env.json"
    env_path.write_text(json.dumps({"provenance": prov, "env": env, "args": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "total_s": round(time.perf_counter() - t_all, 1)}, indent=1) + "\n")
    print(f"wrote {csv_path} ({len(rows)} rows), {env_path}")
    return 1 if failures else 0


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
