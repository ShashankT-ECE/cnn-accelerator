#!/usr/bin/env python3
"""Collect tb_full10k shard logs into v2/results/rtl_full10k.csv (source=rtl_sim).

Subcommands (called by v2/scripts/run_full10k.sh; usable by hand):
  check-shard <shard_dir> <net> <start> <end>
        exit 0 iff the shard is COMPLETE: run.log has the SHARD_DONE line for exactly this
        net/range with images == end-start, and one RESULT line per image of the range.
        (Complete is not the same as passing: a complete shard with mismatches is kept and
        reported, not re-run.)
  xval --a <run_dir_sim_a> --b <run_dir_sim_b> --nets ... [--network-csv v2/results/rtl_network.csv]
        cross-validation of two simulators on the same images: per image identical LOGIT[0..15],
        LAYER_CYC, TOTAL_CYC, MAC_ACTIVE, STALL, logits_ok, cycles_ok; both all-pass; plus
        rtl_total / rtl_layer_cycles identical to the committed xsim tb_gos_core SUITE=net rows.
  collect --run-dir R --data-dir D --nets ... [--csv PATH] [--shard-csv PATH]
        per net (plan = R/<net>/plan.txt, lines "start end"): every shard complete, every image
        RESULT: logits bit-exact (TB compare vs logit16 hex AND collector compare vs golden.npz v,
        unused LOGIT[OC..15] == 0), prediction (final_layer PS float32 argmax of the RTL logits)
        == golden prediction, cycles exact (TB). One row per net via common.write_results_csv.
Exit 1 on any incomplete shard or mismatch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import lzma
import tarfile
import json
import re
import sys
from pathlib import Path

import numpy as np

FULLSIM = Path(__file__).resolve().parent
V2 = FULLSIM.parent
sys.path.insert(0, str(V2 / "model"))

N_LOGITS = 16
FIELDS = ["images", "images_expected", "logits_bitexact", "pred_match", "cycles_exact",
          "all_exact", "first_mismatch_img", "n_mismatch", "rtl_correct", "golden_correct",
          "simulator", "simulator_version", "shards", "shards_complete", "wall_s",
          "shard_seconds_sum", "model_total_cyc", "data_commit", "run_dir", "all_ok",
          "logs_archive", "logs_archive_sha256", "record_archives"]

# Text logs kept in the archive (compiled simulator objects under build/obj are excluded).
ARCHIVE_SUFFIXES = (".log", ".txt", ".csv", ".json")
ARCHIVE_NAMES = ("status", "BUILD_OK")


def archive_members(run_dir: Path) -> list[Path]:
    """Every text log of a run directory (shard run.log/status, plan, shards.csv, xval, build logs)."""
    out = []
    for f in sorted(run_dir.rglob("*")):
        if not f.is_file() or "obj" in f.relative_to(run_dir).parts:
            continue
        if f.suffix in ARCHIVE_SUFFIXES or f.name in ARCHIVE_NAMES:
            out.append(f)
    return out


def write_archive(run_dir: Path, out: Path) -> str:
    """Deterministic .tar.xz of the run's text logs (sorted names, fixed mtime/owner); returns sha256."""
    run_dir, out = Path(run_dir), Path(out)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for f in archive_members(run_dir):
            data = f.read_bytes()
            ti = tarfile.TarInfo(f"{run_dir.parent.name}/{run_dir.name}/{f.relative_to(run_dir)}")
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
            tf.addfile(ti, io.BytesIO(data))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(lzma.compress(buf.getvalue(), preset=9))
    return hashlib.sha256(out.read_bytes()).hexdigest()


def record_archives(out: Path) -> str:
    """'name:sha256;...' of the other rtl_full10k_logs_*.tar.xz next to `out` (earlier runs kept as a record)."""
    out = Path(out)
    others = sorted(p for p in out.parent.glob("rtl_full10k_logs_*.tar.xz") if p.name != out.name)
    return ";".join(f"{p.name}:{hashlib.sha256(p.read_bytes()).hexdigest()}" for p in others)


# --------------------------------------------------------------------------- parsing
def parse_log(text: str) -> dict:
    """run.log -> {sim, version, results {img: fields}, done (SHARD_DONE fields) | None,
    passed, failed, dup_imgs}."""
    out = {"sim": "", "version": "", "results": {}, "done": None, "passed": False,
           "failed": False, "dup_imgs": []}
    for line in text.splitlines():
        if line.startswith("SIMULATOR "):
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            out["sim"], out["version"] = kv.get("name", ""), kv.get("version", "")
        elif line.startswith("RESULT kind=full "):
            kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
            img = int(kv["img"])
            if img in out["results"]:
                out["dup_imgs"].append(img)
            out["results"][img] = kv
        elif line.startswith("SHARD_DONE "):
            out["done"] = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
        elif line.startswith("TEST PASSED"):
            out["passed"] = True
        elif line.startswith("TEST FAILED") or re.match(r"^(ERROR|FATAL|Fatal)", line):
            out["failed"] = True
    return out


def shard_complete(parsed: dict, net: str, start: int, end: int) -> tuple[bool, str]:
    d = parsed["done"]
    if d is None:
        return False, "no SHARD_DONE line"
    try:
        if (d.get("net"), int(d["start"]), int(d["end"]), int(d["images"])) != (net, start, end,
                                                                                 end - start):
            return False, f"SHARD_DONE mismatch {d}"
    except (KeyError, ValueError):
        return False, f"bad SHARD_DONE {d}"
    if parsed["dup_imgs"]:
        return False, f"duplicate RESULT images {parsed['dup_imgs'][:5]}"
    if set(parsed["results"]) != set(range(start, end)):
        return False, "RESULT images != range"
    if any(r.get("net") != net for r in parsed["results"].values()):
        return False, "RESULT net mismatch"
    return True, "complete"


def read_status(shard_dir: Path):
    try:
        rc, t0, t1 = (shard_dir / "status").read_text().split()
        return int(rc), float(t0), float(t1)
    except (OSError, ValueError):
        return None


def load_shard(shard_dir: Path, net: str, start: int, end: int) -> dict:
    log = shard_dir / "run.log"
    p = parse_log(log.read_text(errors="replace")) if log.is_file() else parse_log("")
    ok, why = shard_complete(p, net, start, end)
    p.update(complete=ok, why=why, status=read_status(shard_dir), start=start, end=end,
             dir=str(shard_dir))
    return p


def read_plan(net_dir: Path) -> list[tuple[int, int]]:
    plan = []
    for line in (net_dir / "plan.txt").read_text().splitlines():
        if line.strip():
            a, b = line.split()
            plan.append((int(a), int(b)))
    return plan


def shard_dir_name(start: int, end: int) -> str:
    return f"shard_{start:05d}_{end:05d}"


def logits_of(r: dict) -> list[int]:
    return [int(x) for x in r["logits"].split(",")]


# --------------------------------------------------------------------------- per-net evaluation
def evaluate_net(net: str, shards: list[dict], golden: dict, predict) -> dict:
    """Pure evaluation (unit-tested with synthetic logs). golden = {v [n,OC], pred [n], labels [n]};
    predict(v_int32 [k,OC]) -> preds [k]."""
    v_gold = np.asarray(golden["v"])
    OC = v_gold.shape[1]
    expected = sum(s["end"] - s["start"] for s in shards)
    imgs, rows_v, bad = [], [], []
    n_log = n_cyc = n_complete = 0
    for s in shards:
        n_complete += s["complete"]
        for img in sorted(s["results"]):
            r = s["results"][img]
            lg = logits_of(r)
            tb_ok = r.get("logits_ok") == "1"
            ok_v = (len(lg) == N_LOGITS and lg[:OC] == [int(x) for x in v_gold[img]]
                    and all(x == 0 for x in lg[OC:]))
            lok = tb_ok and ok_v
            cok = r.get("cycles_ok") == "1"
            n_log += lok
            n_cyc += cok
            imgs.append(img)
            rows_v.append(lg[:OC] if len(lg) >= OC else [0] * OC)
            if not (lok and cok):
                bad.append(img)
    if imgs:
        idx = np.array(imgs)
        pred_rtl = np.asarray(predict(np.array(rows_v, dtype=np.int32)))
        pm = pred_rtl == np.asarray(golden["pred"])[idx]
        n_pred = int(pm.sum())
        rtl_correct = int((pred_rtl == np.asarray(golden["labels"])[idx]).sum())
        golden_correct = int((np.asarray(golden["pred"])[idx] == np.asarray(golden["labels"])[idx]).sum())
        bad += [int(i) for i in idx[~pm]]
    else:
        n_pred = rtl_correct = golden_correct = 0
    bad = sorted(set(bad))
    st = [s["status"] for s in shards if s["status"]]
    n = len(imgs)
    all_ok = (n == expected and n_complete == len(shards) and n_log == n and n_pred == n
              and n_cyc == n and not bad and len(set(imgs)) == n)
    return {"net": net, "images": n, "images_expected": expected, "logits_bitexact": n_log,
            "pred_match": n_pred, "cycles_exact": n_cyc, "all_exact": n - len(bad),
            "first_mismatch_img": bad[0] if bad else "", "n_mismatch": len(bad),
            "mismatch_imgs": bad,
            "rtl_correct": rtl_correct, "golden_correct": golden_correct,
            "shards": len(shards), "shards_complete": n_complete,
            "wall_s": round(max(t[2] for t in st) - min(t[1] for t in st), 1) if st else "",
            "shard_seconds_sum": round(sum(t[2] - t[1] for t in st), 1) if st else "",
            "simulator": ",".join(sorted({s["sim"] for s in shards if s["sim"]})),
            "simulator_version": ",".join(sorted({s["version"] for s in shards if s["version"]})),
            "all_ok": all_ok}


# --------------------------------------------------------------------------- subcommands
def cmd_check_shard(a) -> int:
    s = load_shard(Path(a.shard_dir), a.net, a.start, a.end)
    print(f"{a.shard_dir}: {s['why']}")
    return 0 if s["complete"] else 1


def same_layer_cycles(a: str, b: str) -> bool:
    """LAYER_CYC lists equal up to trailing zero padding (rtl_network.csv rows before 668e8f6
    carry a 0-padded 5th entry for 4-layer nets)."""
    x, y = [int(t) for t in a.split(",")], [int(t) for t in b.split(",")]
    n = max(len(x), len(y))
    return x + [0] * (n - len(x)) == y + [0] * (n - len(y))


def compare_runs(A: dict[int, dict], B: dict[int, dict]) -> list[str]:
    """Per-image identity of two simulators' RESULT fields."""
    keys = ("logits", "layer_cycles", "rtl_total", "model_total", "mac", "stall", "logits_ok",
            "cycles_ok")
    errs = []
    if set(A) != set(B):
        errs.append(f"image sets differ: {sorted(set(A) ^ set(B))[:5]}")
    for img in sorted(set(A) & set(B)):
        for k in keys:
            if A[img].get(k) != B[img].get(k):
                errs.append(f"img {img} {k}: {A[img].get(k)} != {B[img].get(k)}")
        if A[img].get("logits_ok") != "1" or A[img].get("cycles_ok") != "1":
            errs.append(f"img {img}: not passing")
    return errs


def cmd_xval(a) -> int:
    report = {"nets": {}, "ok": True}
    for net in a.nets:
        runs = {}
        for tag, d in (("a", a.a), ("b", a.b)):
            sd = Path(d) / net
            plan = read_plan(sd)
            res, sims = {}, set()
            complete = True
            for s, e in plan:
                sh = load_shard(sd / shard_dir_name(s, e), net, s, e)
                complete &= sh["complete"] and sh["passed"] and not sh["failed"]
                res.update(sh["results"])
                sims.add(f"{sh['sim']} {sh['version']}")
            runs[tag] = {"res": res, "sims": sorted(sims), "complete": complete}
        errs = compare_runs(runs["a"]["res"], runs["b"]["res"])
        if not (runs["a"]["complete"] and runs["b"]["complete"]):
            errs.append("a shard is incomplete or failing")
        # committed xsim tb_gos_core SUITE=net rows (same images, same RTL per D16)
        ref_n = 0
        if a.network_csv and Path(a.network_csv).is_file():
            for row in csv.DictReader(open(a.network_csv)):
                if row["net"] != net:
                    continue
                img = int(row["img"])
                if img in runs["a"]["res"]:
                    r = runs["a"]["res"][img]
                    ref_n += 1
                    if (r["rtl_total"] != row["rtl_total"]
                            or not same_layer_cycles(r["layer_cycles"], row["rtl_layer_cycles"])):
                        errs.append(f"img {img}: cycles differ from rtl_network.csv")
        n = len(runs["a"]["res"])
        report["nets"][net] = {"images": n, "a": runs["a"]["sims"], "b": runs["b"]["sims"],
                               "rtl_network_csv_rows_compared": ref_n, "errors": errs[:20],
                               "ok": not errs and n > 0}
        report["ok"] &= report["nets"][net]["ok"]
        print(f"  xval {net}: {n} images, {runs['a']['sims']} vs {runs['b']['sims']}, "
              f"rtl_network.csv rows compared {ref_n}: {'IDENTICAL' if not errs else errs[:3]}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"xval: {'PASS' if report['ok'] else 'FAIL'}")
    return 0 if report["ok"] else 1


def cmd_collect(a) -> int:
    from common import RESULTS_DIR, base_meta, write_results_csv  # noqa: E402
    import final_layer  # noqa: E402
    import gos_golden as gg  # noqa: E402

    run_dir, data_dir = Path(a.run_dir), Path(a.data_dir)
    rows, shard_rows, ok = [], [], True
    for net in a.nets:
        meta = json.loads((data_dir / net / "meta.json").read_text())
        z = np.load(data_dir / net / "golden.npz")
        golden = {"v": z["v"], "pred": z["pred"], "labels": z["labels"]}
        G = gg.load_net(net)
        params = {"S_a": G.final.S_a, "S_w": G.final.S_w}
        nd = run_dir / net
        shards = [load_shard(nd / shard_dir_name(s, e), net, s, e) for s, e in read_plan(nd)]
        r = evaluate_net(net, shards, golden, lambda v: final_layer.predict_from_raw(v, params))
        mt = sorted({x["results"][i]["model_total"] for x in shards for i in x["results"]})
        r["model_total_cyc"] = ",".join(mt)
        r["data_commit"] = meta["git_commit"]
        r["run_dir"] = str(run_dir.relative_to(V2)) if run_dir.is_relative_to(V2) else str(run_dir)
        m = base_meta(net=net, layer="all", source="rtl_sim", duration_s=r["wall_s"],
                      num_inferences=r["images"])
        m["vivado_version"] = "2023.1" if r["simulator"] == "xsim" else ""
        rows.append({**m, **r})
        ok &= r["all_ok"]
        for s in shards:
            d = s["done"] or {}
            st = s["status"]
            shard_rows.append({"net": net, "start": s["start"], "end": s["end"],
                               "complete": s["complete"], "why": s["why"], "passed": s["passed"],
                               "images": d.get("images", ""), "logits_ok": d.get("logits_ok", ""),
                               "cycles_ok": d.get("cycles_ok", ""), "errors": d.get("errors", ""),
                               "first_bad": d.get("first_bad", ""), "rc": st[0] if st else "",
                               "seconds": round(st[2] - st[1], 1) if st else "",
                               "simulator": s["sim"], "dir": s["dir"]})
        print(f"  {net}: images {r['images']}/{r['images_expected']}, logits bit-exact "
              f"{r['logits_bitexact']}, pred match {r['pred_match']}, cycles exact "
              f"{r['cycles_exact']}, first mismatch {r['first_mismatch_img'] or '-'}, shards "
              f"{r['shards_complete']}/{r['shards']}, {r['simulator']} {r['simulator_version']}, "
              f"wall {r['wall_s']} s [RTL sim]")
        if r["mismatch_imgs"]:
            print(f"    mismatching images (first 20): {r['mismatch_imgs'][:20]}")
    if a.shard_csv:
        p = Path(a.shard_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(shard_rows[0]) if shard_rows else ["net"])
            w.writeheader()
            w.writerows(shard_rows)
        print(f"wrote {p}")
    arch = {"logs_archive": "", "logs_archive_sha256": "", "record_archives": ""}
    if a.archive:          # after shards.csv so the archive contains it
        ap_ = Path(a.archive)
        arch = {"logs_archive": ap_.name, "logs_archive_sha256": write_archive(run_dir.parent, ap_),
                "record_archives": record_archives(ap_)}
        print(f"wrote {ap_} ({ap_.stat().st_size} B, sha256 {arch['logs_archive_sha256']})")
    for r in rows:
        r.update(arch)
    csv_path = Path(a.csv) if a.csv else RESULTS_DIR / "rtl_full10k.csv"
    write_results_csv(csv_path, rows, FIELDS)
    print(f"wrote {csv_path}")
    print(f"collect_full10k: {'ALL OK' if ok else 'FAILURES / INCOMPLETE'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-shard")
    c.add_argument("shard_dir")
    c.add_argument("net")
    c.add_argument("start", type=int)
    c.add_argument("end", type=int)
    x = sub.add_parser("xval")
    x.add_argument("--a", required=True)
    x.add_argument("--b", required=True)
    x.add_argument("--nets", nargs="+", required=True)
    x.add_argument("--network-csv", default=str(V2 / "results" / "rtl_network.csv"))
    x.add_argument("--out", default="")
    k = sub.add_parser("collect")
    k.add_argument("--run-dir", required=True)
    k.add_argument("--data-dir", required=True)
    k.add_argument("--nets", nargs="+", required=True)
    k.add_argument("--csv", default="", help="default v2/results/rtl_full10k.csv")
    k.add_argument("--shard-csv", default="")
    k.add_argument("--archive", default="",
                   help="also write a .tar.xz of the run's text logs here (sha256 into the CSV)")
    r = sub.add_parser("archive", help="archive an existing run's text logs (e.g. an earlier run as a record)")
    r.add_argument("--run-dir", required=True, help="<runs>/<commit>/<sim> directory")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "archive":
        sha = write_archive(Path(a.run_dir).parent, Path(a.out))
        print(f"wrote {a.out} (sha256 {sha})")
        return 0
    return {"check-shard": cmd_check_shard, "xval": cmd_xval, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
