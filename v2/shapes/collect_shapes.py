#!/usr/bin/env python3
"""Collect tb_shapes shard logs into shapes_rtl.csv (one row per job, source=rtl_sim).

Subcommands (called by v2/scripts/run_shapes.sh; usable by hand):
  check-shard <shard_dir> <sel> --data D
        exit 0 iff the shard is COMPLETE: run.log has a SHARD_DONE line for this selection with
        jobs == number of selected jobs, and exactly one RESULT line per selected job.
        sel = "range:<a>:<b>" or "list:<file.hex>". (Complete is not passing: a complete shard
        with mismatches is kept and reported, not re-run.)
  xval --a <run_dir_sim_a> --b <run_dir_sim_b> --data D [--out xval.json]
        simulator cross-check on the same job list: per job identical RESULT fields (outputs
        signature, logits, LAYER_CYC, TOTAL_CYC, MAC_ACTIVE, STALL, ERR_CODE, flags, verdicts),
        both complete, and every job exact against the shape set (collector evaluation).
  collect --run-dir R --data D [--csv PATH] [--shard-csv PATH] [--archive PATH] [--xval-json J]
        every planned shard complete; every job re-evaluated independently from the npz:
        valid jobs: done && !error, outputs (ACT: TB byte-masked compare + out_sig recomputed from
        the npz; out_raw: LOGIT[0..OC-1] == npz, unused LOGIT == 0), LAYER_CYC / TOTAL_CYC /
        MAC_ACTIVE == model (npz), STALL == 0, sticky flags == 0; refuse jobs: error and ERR_CODE ==
        expected. One row per job via common.write_results_csv (source=rtl_sim).
  archive --run-dir R --out PATH
Exit 1 on any incomplete shard or mismatch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import lzma
import re
import sys
import tarfile
from pathlib import Path

import numpy as np

SHAPES = Path(__file__).resolve().parent
V2 = SHAPES.parent
sys.path.insert(0, str(V2 / "board"))
import shapeset as ss  # noqa: E402  (numpy-only loader, shared with the board)

SIG_INIT = 0xCBF29CE484222325
SIG_PRIME = 0x00000100000001B3
M64 = (1 << 64) - 1

FIELDS = ["job_id", "category", "n_layers", "expect_refuse", "refuse_mode", "shape", "in_shape",
          "out_shape", "out_raw", "has_pool", "has_oc_tail", "has_x_tail", "has_1x1_out", "K_list",
          "T_list", "T_total", "tk_total", "macs", "model_layer_cycles", "model_total", "model_mac",
          "rtl_layer_cycles", "rtl_total", "rtl_mac", "rtl_stall", "rtl_flags", "total_err",
          "abs_total_err", "max_abs_layer_err", "err_code_expected", "err_code_rtl", "err_ok",
          "outputs_match", "cycles_exact", "all_ok", "timeout", "simulator", "simulator_version",
          "xsim_crosschecked", "seed", "shapeset_sha256", "data_git_commit", "data_git_dirty",
          "shard", "run_dir", "logs_archive", "logs_archive_sha256"]

ARCHIVE_SUFFIXES = (".log", ".txt", ".csv", ".json", ".hex")
ARCHIVE_NAMES = ("status", "BUILD_OK", "PASS")


# --------------------------------------------------------------------------- parsing
def _kv(line: str) -> dict:
    return dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)


def parse_log(text: str) -> dict:
    out = {"sim": "", "version": "", "results": {}, "done": None, "passed": False, "failed": False,
           "dup_jobs": []}
    for line in text.splitlines():
        if line.startswith("SIMULATOR "):
            kv = _kv(line)
            out["sim"], out["version"] = kv.get("name", ""), kv.get("version", "")
        elif line.startswith("RESULT kind=shape "):
            kv = _kv(line)
            j = int(kv["job"])
            if j in out["results"]:
                out["dup_jobs"].append(j)
            out["results"][j] = kv
        elif line.startswith("SHARD_DONE "):
            out["done"] = _kv(line)
        elif line.startswith("TEST PASSED"):
            out["passed"] = True
        elif line.startswith("TEST FAILED") or re.match(r"^(ERROR|FATAL|Fatal|%Error)", line):
            out["failed"] = True
    return out


def read_joblist(path) -> list[int]:
    vals = [int(t, 16) for t in Path(path).read_text().split()]
    return vals[1:1 + vals[0]]


def sel_jobs(sel: str) -> list[int]:
    if sel.startswith("range:"):
        _, a, b = sel.split(":")
        return list(range(int(a), int(b)))
    if sel.startswith("list:"):
        return read_joblist(sel[5:])
    raise ValueError(sel)


def shard_complete(parsed: dict, jobs: list[int]) -> tuple[bool, str]:
    d = parsed["done"]
    if d is None:
        return False, "no SHARD_DONE line"
    try:
        if int(d["jobs"]) != len(jobs):
            return False, f"SHARD_DONE jobs {d['jobs']} != {len(jobs)}"
    except (KeyError, ValueError):
        return False, f"bad SHARD_DONE {d}"
    if parsed["dup_jobs"]:
        return False, f"duplicate RESULT jobs {parsed['dup_jobs'][:5]}"
    if set(parsed["results"]) != set(jobs):
        return False, "RESULT jobs != selection"
    return True, "complete"


def read_status(shard_dir: Path):
    try:
        rc, t0, t1 = (shard_dir / "status").read_text().split()
        return int(rc), float(t0), float(t1)
    except (OSError, ValueError):
        return None


def load_shard(shard_dir: Path, sel: str) -> dict:
    log = shard_dir / "run.log"
    p = parse_log(log.read_text(errors="replace")) if log.is_file() else parse_log("")
    jobs = sel_jobs(sel)
    ok, why = shard_complete(p, jobs)
    p.update(complete=ok, why=why, status=read_status(shard_dir), sel=sel, jobs=jobs, dir=str(shard_dir))
    return p


def shard_dir_name(sel: str) -> str:
    if sel.startswith("range:"):
        _, a, b = sel.split(":")
        return f"shard_{int(a):04d}_{int(b):04d}"
    return "shard_" + Path(sel[5:]).stem


def read_plan(run_dir: Path) -> list[str]:
    return [ln.strip() for ln in (run_dir / "plan.txt").read_text().splitlines() if ln.strip()]


# --------------------------------------------------------------------------- evaluation
def out_signature(words, mask64) -> int:
    sig = SIG_INIT
    for w, m in zip(np.asarray(words, dtype=np.uint64).tolist(), np.asarray(mask64, dtype=np.uint64).tolist()):
        sig = ((sig * SIG_PRIME) & M64) ^ (w & m)
    return sig


def _ints(s: str) -> list[int]:
    return [] if s in ("", "-") else [int(x) for x in s.split(",")]


def evaluate_job(r: dict | None, s: "ss.Shape", exp_sig: int | None = None) -> dict:
    """Independent verdict for one job from its RESULT fields and the npz (Shape)."""
    if r is None:
        return {"present": False, "err_ok": False, "outputs_match": False, "cycles_exact": False,
                "all_ok": False, "why": "no RESULT"}
    g = lambda k: r.get(k, "")  # noqa: E731
    why = []
    timeout = g("timeout") == "1"
    code = int(g("err_code") or -1)
    lc = _ints(g("layer_cycles"))
    total, mac = int(g("rtl_total") or -1), int(g("mac") or -1)
    stall, flags = int(g("stall") or -1), int(g("flags") or -1)
    if s.expect_refuse:
        err_ok = (g("error") == "1" and code == s.err_code and not timeout and g("err_ok") == "1")
        if not err_ok:
            why.append(f"refusal: error={g('error')} code={code} exp={s.err_code} timeout={timeout}")
        cyc_ok = (total == s.model_total and mac == s.model_mac and stall == 0 and not timeout
                  and g("cycles_ok") == "1" and int(g("model_total") or -1) == s.model_total)
        if not cyc_ok:
            why.append(f"refusal cycles: rtl {total} model {s.model_total} mac {mac}")
        if flags != 0:
            why.append(f"sticky flags {flags}")
        return {"present": True, "err_ok": err_ok, "outputs_match": "", "cycles_exact": cyc_ok,
                "all_ok": err_ok and cyc_ok and flags == 0, "why": "; ".join(why),
                "timeout": int(timeout), "err_code": code, "layer_cycles": lc, "total": total,
                "mac": mac, "stall": stall, "flags": flags}
    err_ok = g("done") == "1" and g("error") == "0" and code == 0 and not timeout
    if not err_ok:
        why.append(f"status: done={g('done')} error={g('error')} code={code} timeout={timeout}")
    if s.out_raw:
        lg = _ints(g("logits"))
        out_ok = (lg == [int(x) for x in s.logits_expected[:s.n_logits]] and g("out_ok") == "1"
                  and g("lg_zero") == "1")
    else:
        sig = exp_sig if exp_sig is not None else out_signature(s.out_expected, s.out_mask64)
        try:
            got_sig = int(g("out_sig"), 16)
        except ValueError:
            got_sig = -1
        out_ok = (got_sig == sig and g("n_bad") == "0" and g("out_ok") == "1" and g("lg_zero") == "1")
    if not out_ok:
        why.append("outputs")
    cyc_ok = (lc == [int(x) for x in s.model_layer_cycles] and total == s.model_total
              and mac == s.model_mac and stall == 0 and g("cycles_ok") == "1"
              and int(g("model_total") or -1) == s.model_total)
    if not cyc_ok:
        why.append(f"cycles: rtl {total} model {s.model_total}")
    if flags != 0:
        why.append(f"sticky flags {flags}")
    return {"present": True, "err_ok": err_ok, "outputs_match": out_ok, "cycles_exact": cyc_ok,
            "all_ok": err_ok and out_ok and cyc_ok and flags == 0, "why": "; ".join(why),
            "timeout": int(timeout), "err_code": code, "layer_cycles": lc, "total": total, "mac": mac,
            "stall": stall, "flags": flags}


def job_row(s: "ss.Shape", ev: dict, man_job: dict) -> dict:
    L = [s.layer(i) for i in range(s.fields.shape[0])]
    f0, fl = L[0], L[-1]
    row = {"job_id": s.job, "category": s.category, "n_layers": s.n_layers,
           "expect_refuse": int(s.expect_refuse), "refuse_mode": s.refuse_mode, "shape": s.shape,
           "in_shape": f"{f0['IC']}x{f0['IH']}x{f0['IW']}",
           "out_shape": (f"LOGIT[{fl['OC']}]" if fl["out_raw"] else f"{fl['OC']}x{fl['OUT_H']}x{fl['OUT_W']}"),
           "out_raw": int(s.out_raw), "has_pool": man_job.get("has_pool", ""),
           "has_oc_tail": man_job.get("has_oc_tail", ""), "has_x_tail": man_job.get("has_x_tail", ""),
           "has_1x1_out": man_job.get("has_1x1_out", ""),
           "K_list": ",".join(str(f["K"]) for f in L),
           "T_list": ",".join(str(t) for t in man_job.get("T", [])),
           "T_total": sum(man_job.get("T", [])) if not s.expect_refuse else "",
           "tk_total": s.tk_total if not s.expect_refuse else "",
           "macs": s.macs if not s.expect_refuse else "",
           "err_code_expected": s.err_code, "timeout": ev.get("timeout", ""),
           "err_code_rtl": ev.get("err_code", ""), "err_ok": ev["err_ok"],
           "outputs_match": ev["outputs_match"], "cycles_exact": ev["cycles_exact"], "all_ok": ev["all_ok"]}
    lc = ev.get("layer_cycles", [])
    mlc = [int(x) for x in s.model_layer_cycles]
    row.update({"model_layer_cycles": ",".join(map(str, mlc)), "model_total": s.model_total,
                "model_mac": s.model_mac, "rtl_layer_cycles": ",".join(map(str, lc)),
                "rtl_total": ev.get("total", ""), "rtl_mac": ev.get("mac", ""),
                "rtl_stall": ev.get("stall", ""), "rtl_flags": ev.get("flags", "")})
    if ev["present"]:
        row["total_err"] = ev["total"] - s.model_total
        row["abs_total_err"] = abs(row["total_err"])
        row["max_abs_layer_err"] = (max((abs(a - b) for a, b in zip(lc, mlc)), default=0)
                                    if len(lc) == len(mlc) else "")
    return row


# --------------------------------------------------------------------------- archive
def archive_members(run_dir: Path) -> list[Path]:
    out = []
    for f in sorted(run_dir.rglob("*")):
        if not f.is_file() or {"obj", "xsim.dir"} & set(f.relative_to(run_dir).parts):
            continue
        if f.suffix in ARCHIVE_SUFFIXES or f.name in ARCHIVE_NAMES:
            out.append(f)
    return out


def write_archive(run_dir: Path, out: Path) -> str:
    """Deterministic .tar.xz of a run's text logs (sorted, fixed mtime/owner); returns sha256."""
    run_dir, out = Path(run_dir), Path(out)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for f in archive_members(run_dir):
            data = f.read_bytes()
            ti = tarfile.TarInfo(f"{run_dir.name}/{f.relative_to(run_dir)}")
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
            tf.addfile(ti, io.BytesIO(data))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(lzma.compress(buf.getvalue(), preset=9))
    return hashlib.sha256(out.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- subcommands
def _load_set(data: Path) -> tuple[dict, dict]:
    man = ss.verify_manifest(data)
    shapes = {s.job: s for s in ss.load_shapeset(data, verify=False)}
    return man, shapes


def cmd_check_shard(a) -> int:
    s = load_shard(Path(a.shard_dir), a.sel)
    print(f"{a.shard_dir}: {s['why']}")
    return 0 if s["complete"] else 1


XVAL_KEYS = ("refuse", "done", "error", "err_code", "exp_code", "err_ok", "out_ok", "out_sig", "n_bad",
             "lg_zero", "cycles_ok", "rtl_total", "model_total", "mac", "model_mac", "stall", "flags",
             "timeout", "layer_cycles", "logits")


def compare_runs(A: dict[int, dict], B: dict[int, dict]) -> list[str]:
    errs = []
    if set(A) != set(B):
        errs.append(f"job sets differ: {sorted(set(A) ^ set(B))[:5]}")
    for j in sorted(set(A) & set(B)):
        for k in XVAL_KEYS:
            if A[j].get(k) != B[j].get(k):
                errs.append(f"job {j} {k}: {A[j].get(k)} != {B[j].get(k)}")
    return errs


def _run_results(run_dir: Path) -> tuple[dict, list[dict]]:
    res, shards = {}, []
    for sel in read_plan(run_dir):
        sh = load_shard(run_dir / shard_dir_name(sel), sel)
        shards.append(sh)
        res.update(sh["results"])
    return res, shards


def cmd_xval(a) -> int:
    man, shapes = _load_set(Path(a.data))
    runs = {}
    errs = []
    for tag, d in (("a", a.a), ("b", a.b)):
        res, shards = _run_results(Path(d))
        runs[tag] = {"res": res, "sims": sorted({f"{s['sim']} {s['version']}" for s in shards})}
        if not all(s["complete"] for s in shards):
            errs.append(f"{tag}: a shard is incomplete")
        for j, r in res.items():
            ev = evaluate_job(r, shapes[j])
            if not ev["all_ok"]:
                errs.append(f"{tag}: job {j} not exact ({ev['why']})")
    errs += compare_runs(runs["a"]["res"], runs["b"]["res"])
    n = len(runs["a"]["res"])
    rep = {"ok": not errs and n >= 1, "jobs": sorted(runs["a"]["res"]), "n_jobs": n,
           "a": runs["a"]["sims"], "b": runs["b"]["sims"], "shapeset_sha256": man["shapeset_sha256"],
           "errors": errs[:50]}
    if a.min_jobs and n < a.min_jobs:
        rep["ok"] = False
        rep["errors"].append(f"only {n} jobs < {a.min_jobs}")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(f"xval: {n} jobs, {rep['a']} vs {rep['b']}: {'IDENTICAL + EXACT' if rep['ok'] else rep['errors'][:3]}")
    print(f"xval: {'PASS' if rep['ok'] else 'FAIL'}")
    return 0 if rep["ok"] else 1


def cmd_collect(a) -> int:
    sys.path.insert(0, str(V2 / "model"))
    from common import RESULTS_DIR, base_meta, write_results_csv  # noqa: PLC0415

    data, run_dir = Path(a.data), Path(a.run_dir)
    man, shapes = _load_set(data)
    mj = {e["job"]: e for e in man["jobs"]}
    xv = json.loads(Path(a.xval_json).read_text()) if a.xval_json and Path(a.xval_json).is_file() else None
    xjobs = set(xv["jobs"]) if xv and xv.get("ok") and xv.get("shapeset_sha256") == man["shapeset_sha256"] else set()
    rows, shard_rows = [], []
    res, shards = _run_results(run_dir)
    planned = sorted({j for s in shards for j in s["jobs"]})
    owner = {j: s for s in shards for j in s["jobs"]}
    n_ok = 0
    bad = []
    for j in planned:
        s, sh = shapes[j], owner[j]
        ev = evaluate_job(res.get(j), s)
        row = job_row(s, ev, mj[j])
        st = sh["status"]
        m = base_meta(net="shapes", layer=f"J{j:04d}", source="rtl_sim", num_inferences=1)
        m["vivado_version"] = "2023.1" if sh["sim"] == "xsim" else ""
        row.update({"simulator": sh["sim"], "simulator_version": sh["version"],
                    "xsim_crosschecked": int(j in xjobs), "seed": man["seed"],
                    "shapeset_sha256": man["shapeset_sha256"], "data_git_commit": man["git_commit"],
                    "data_git_dirty": man["git_dirty"], "shard": Path(sh["dir"]).name,
                    "run_dir": str(run_dir.relative_to(V2)) if run_dir.is_relative_to(V2) else str(run_dir)})
        rows.append({**m, **row})
        n_ok += bool(ev["all_ok"])
        if not ev["all_ok"]:
            bad.append((j, ev["why"]))
    complete = all(s["complete"] for s in shards)
    for s in shards:
        d, st = s["done"] or {}, s["status"]
        shard_rows.append({"sel": s["sel"], "complete": s["complete"], "why": s["why"], "passed": s["passed"],
                           "jobs": d.get("jobs", ""), "ok": d.get("ok", ""), "errors": d.get("errors", ""),
                           "rc": st[0] if st else "", "seconds": round(st[2] - st[1], 1) if st else "",
                           "simulator": s["sim"], "dir": s["dir"]})
    if a.shard_csv:
        p = Path(a.shard_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(shard_rows[0]) if shard_rows else ["sel"])
            w.writeheader()
            w.writerows(shard_rows)
        print(f"wrote {p}")
    arch = {"logs_archive": "", "logs_archive_sha256": ""}
    if a.archive:
        ap_ = Path(a.archive)
        arch = {"logs_archive": ap_.name, "logs_archive_sha256": write_archive(run_dir.parent, ap_)}
        print(f"wrote {ap_} ({ap_.stat().st_size} B, sha256 {arch['logs_archive_sha256']})")
    for r in rows:
        r.update(arch)
    csv_path = Path(a.csv) if a.csv else RESULTS_DIR / "shapes_rtl.csv"
    write_results_csv(csv_path, rows, FIELDS)
    st = [s["status"] for s in shards if s["status"]]
    wall = round(max(t[2] for t in st) - min(t[1] for t in st), 1) if st else ""
    valid = [r for r in rows if not r["expect_refuse"]]
    print(f"  jobs {len(rows)} (valid {len(valid)}, refuse {len(rows) - len(valid)}), all exact {n_ok}, "
          f"outputs match {sum(r['outputs_match'] is True for r in valid)}, cycles exact "
          f"{sum(r['cycles_exact'] is True for r in valid)}, refusals ok "
          f"{sum(r['err_ok'] is True for r in rows if r['expect_refuse'])}, shards "
          f"{sum(s['complete'] for s in shards)}/{len(shards)}, xsim cross-checked {len(xjobs & set(planned))}, "
          f"wall {wall} s [RTL sim]")
    for j, why in bad[:20]:
        print(f"    MISMATCH job {j}: {why}")
    print(f"wrote {csv_path}")
    ok = complete and n_ok == len(planned) and len(planned) > 0
    print(f"collect_shapes: {'ALL EXACT' if ok else 'FAILURES / INCOMPLETE'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-shard")
    c.add_argument("shard_dir")
    c.add_argument("sel")
    x = sub.add_parser("xval")
    x.add_argument("--a", required=True)
    x.add_argument("--b", required=True)
    x.add_argument("--data", required=True)
    x.add_argument("--min-jobs", type=int, default=0)
    x.add_argument("--out", default="")
    k = sub.add_parser("collect")
    k.add_argument("--run-dir", required=True)
    k.add_argument("--data", required=True)
    k.add_argument("--csv", default="", help="default v2/results/shapes_rtl.csv")
    k.add_argument("--shard-csv", default="")
    k.add_argument("--archive", default="")
    k.add_argument("--xval-json", default="")
    r = sub.add_parser("archive")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "archive":
        print(f"wrote {a.out} (sha256 {write_archive(Path(a.run_dir).parent, Path(a.out))})")
        return 0
    return {"check-shard": cmd_check_shard, "xval": cmd_xval, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
