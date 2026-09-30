#!/usr/bin/env python3
"""Per-layer cycle spread over all 10k test images per net (A3 determinism, expected spread 0).

Board / dry run (per net, every image: LAYER_CYC[l] + TOTAL_CYC):
    ./session.sh py exp_layer_spread.py [--nets ...] [--npz-dir results] [--rerun] [--limit N]
    python3 exp_layer_spread.py --backend model [--npz-dir ../results/dryrun]   # -> results/dryrun/
RTL-sim equivalent (laptop analysis only, read-only on the full-10k Verilator shard logs):
    .venv/bin/python v2/board/exp_layer_spread.py --rtl-sim [--run-dir v2/build/fullsim/runs/<tag>/verilator]
        -> v2/results/dryrun/rtl_layer_spread.csv (never a committed result)

Board source of the per-image counters (no duplicate 10k run when it is not needed):
  1. <npz-dir>/hw_cycles_<net>.npz written by exp_a2_a3_cycles.py (session step s2.A2A3; it runs
     every image with counters and stores layer_cyc [N, L], total_cyc [N], ok [N], idx, clock_mhz,
     source); then hw_logits_<net>.npz (A1) if it carries layer_cyc. Accepted only if its source
     equals this run's source, it covers the whole selected image range with every job ok, and its
     clock equals the current read-back clock (+-0.5 MHz); the npz SHA256 is recorded in each row.
  2. otherwise (or with --rerun) this script runs the images itself and saves
     hw_layer_spread_<net>.npz.
Per net x layer (+ total): images, model cycles, min, max, distinct values, spread = max - min,
images equal to the model, PASS/FAIL (PASS = one distinct value equal to the model on every image).
Writes hw_layer_spread.csv.

RTL-sim mode: the run directory defaults to run_dir of the clean rows of v2/results/rtl_full10k.csv
(the committed full-10k run); every shard_*/run.log under <run_dir>/<net>/ is parsed with
v2/fullsim/collect_full10k.parse_log (RESULT lines carry layer_cycles=... and rtl_total=...). If
the logs are not present (the build dir is not kept), the CSV says so (status column) instead of
numbers. Rows: source=rtl_sim, label "RTL sim (Verilator shard logs), laptop analysis".
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import board_common as bc

FIELDS = ["images", "model_cycles", "cyc_min", "cyc_max", "cyc_distinct", "cyc_spread",
          "images_equal_model", "cycles_origin", "cycles_origin_sha256", "job_errors", "result"]
RTL_FIELDS = ["timestamp", "git_commit", "git_dirty", "net", "layer", "source", "label", "status",
              "run_dir", "run_dir_git_commit", "shards", "images", "images_expected", "model_cycles",
              "cyc_min", "cyc_max", "cyc_distinct", "cyc_spread", "images_equal_model", "result"]
RTL_LABEL = "RTL sim (Verilator shard logs), laptop analysis"
NPZ_CANDIDATES = ("hw_cycles_{net}.npz", "hw_logits_{net}.npz")


# ---- shared statistics ---------------------------------------------------------------------------
def spread_stats(values: np.ndarray, model: int) -> dict:
    v = np.asarray(values, dtype=np.int64)
    if v.size == 0:
        return dict(images=0, model_cycles=model, cyc_min="", cyc_max="", cyc_distinct=0,
                    cyc_spread="", images_equal_model=0, result="FAIL")
    u = np.unique(v)
    eq = int((v == model).sum())
    return dict(images=int(v.size), model_cycles=int(model), cyc_min=int(v.min()), cyc_max=int(v.max()),
                cyc_distinct=int(u.size), cyc_spread=int(v.max() - v.min()), images_equal_model=eq,
                result="PASS" if u.size == 1 and eq == v.size else "FAIL")


def layer_table(names: list[str], model_layers: list[int], model_total: int,
                layer_cyc: np.ndarray, total_cyc: np.ndarray) -> list[dict]:
    out = []
    for l, name in enumerate(names):
        out.append({"layer": name, **spread_stats(layer_cyc[:, l], model_layers[l])})
    out.append({"layer": "total", **spread_stats(total_cyc, model_total)})
    return out


# ---- board / dry run -----------------------------------------------------------------------------
def load_npz_counters(npz_dir: Path, net: str, source: str, n_expected: int, clk: float,
                      tol: float = bc.CLOCK_TOL_MHZ):
    """(arrays dict, path, reason) of the first acceptable npz; (None, None, reasons) otherwise."""
    reasons = []
    for pat in NPZ_CANDIDATES:
        p = Path(npz_dir) / pat.format(net=net)
        if not p.is_file():
            reasons.append(f"{p.name}: absent")
            continue
        try:
            d = np.load(p)
        except (OSError, ValueError) as e:
            reasons.append(f"{p.name}: unreadable ({e})")
            continue
        if "layer_cyc" not in d.files or "total_cyc" not in d.files:
            reasons.append(f"{p.name}: no per-image layer_cyc")
            continue
        src = str(d["source"]) if "source" in d.files else ""
        if src != source:
            reasons.append(f"{p.name}: source {src!r} != {source!r}")
            continue
        idx = d["idx"] if "idx" in d.files else np.arange(d["total_cyc"].shape[0])
        ok = d["ok"] if "ok" in d.files else np.ones(idx.shape[0], bool)
        if idx.shape[0] != n_expected or not np.array_equal(np.sort(idx), np.arange(n_expected)):
            reasons.append(f"{p.name}: covers {idx.shape[0]} images, need 0..{n_expected - 1}")
            continue
        if not ok.all():
            reasons.append(f"{p.name}: {int((~ok).sum())} failed jobs")
            continue
        if "clock_mhz" in d.files:
            c = float(d["clock_mhz"])
            if abs(c - clk) > tol:
                reasons.append(f"{p.name}: clock {c:.6f} != current {clk:.6f} MHz")
                continue
        else:
            reasons.append(f"{p.name}: no clock_mhz recorded")
            continue
        order = np.argsort(idx)
        return ({"layer_cyc": d["layer_cyc"][order], "total_cyc": d["total_cyc"][order],
                 "errors": 0}, p, "; ".join(reasons + [f"{p.name}: accepted"]))
    return None, None, "; ".join(reasons)


def main_board(a) -> int:
    dev, be = bc.open_device(a)
    ctx = bc.RunContext(a, dev, be, "LAYER_SPREAD")
    ctx.check_clean()
    ctx.banner()
    npz_dir = Path(a.npz_dir) if a.npz_dir else ctx.out_dir
    rows, ok = [], True
    for net in a.nets:
        pkg = ctx.package(net)
        clk = dev.fclk0_mhz()
        idx = bc.image_range(pkg, a.limit)
        n = len(idx)
        got, path, why = (None, None, "--rerun") if a.rerun else load_npz_counters(
            npz_dir, net, ctx.source, n, clk)
        if got is not None:
            origin, sha, dur = f"{path.name} (reused, {why.split('; ')[-1]})", bc.sha256_file(path), 0.0
            print(f"[spread {net}] reusing {path} ({n} images): {why}")
        else:
            print(f"[spread {net}] running {n} images ({why})")
            dev.load_net(pkg)
            res = bc.infer_loop(dev, pkg, idx, read_counters=True, max_errors=a.max_errors,
                                tag=f"spread {net}")
            good = res["ok"]
            got = {"layer_cyc": res["layer_cyc"][good], "total_cyc": res["total_cyc"][good],
                   "errors": len(res["errors"])}
            out = ctx.path(f"hw_layer_spread_{net}.npz")
            np.savez_compressed(out, idx=res["idx"], ok=good, layer_cyc=res["layer_cyc"],
                                total_cyc=res["total_cyc"], clock_mhz=clk, source=np.array(ctx.source))
            origin, sha, dur = f"own run -> {out.name}", bc.sha256_file(out), res["duration_s"]
        mc = pkg.model_cycles
        names = [L["name"] for L in pkg.net["layers"]]
        tab = layer_table(names, [int(mc["layers"][l]["cycles"]) for l in range(len(names))],
                          int(mc["total"]["cycles"]), got["layer_cyc"], got["total_cyc"])
        for t in tab:
            row = ctx.meta(net, t["layer"], dur, n, clock_mhz=clk)
            row.update({k: v for k, v in t.items() if k != "layer"}, cycles_origin=origin,
                       cycles_origin_sha256=sha, job_errors=got["errors"])
            if got["errors"] or t["images"] != n:
                row["result"] = "FAIL"
            ok &= row["result"] == "PASS"
            rows.append(row)
            print(f"  {net:8} {t['layer']:8} model {t['model_cycles']:>8} min {t['cyc_min']:>8} "
                  f"max {t['cyc_max']:>8} distinct {t['cyc_distinct']:>3} spread {t['cyc_spread']:>4} "
                  f"{row['result']}")
    ctx.csv("hw_layer_spread.csv", rows, FIELDS)
    print("LAYER SPREAD:", "PASS (one value per layer, equal to the model, on every image)" if ok else "FAIL")
    return 0 if ok else 1


# ---- RTL-sim equivalent (laptop) -----------------------------------------------------------------
def _v2() -> Path:
    return bc.BOARD_DIR.parent


def default_run_dir() -> tuple[Path | None, str]:
    """(run dir, git commit) from the clean rows of v2/results/rtl_full10k.csv."""
    p = _v2() / "results" / "rtl_full10k.csv"
    if not p.is_file():
        return None, ""
    with p.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("git_dirty") == "False" and r.get("run_dir")]
    if not rows:
        return None, ""
    r = max(rows, key=lambda r: r["timestamp"])
    rd = Path(r["run_dir"])
    return (rd if rd.is_absolute() else _v2() / rd), r.get("git_commit", "")


def rtl_spread(run_dir: Path, net: str, n_expected: int = 10000):
    """(table rows, status, shards, images) from <run_dir>/<net>/shard_*/run.log."""
    sys.path.insert(0, str(_v2() / "fullsim"))
    from collect_full10k import parse_log  # noqa: E402  (read-only reuse)
    logs = sorted((Path(run_dir) / net).glob("shard_*/run.log")) if run_dir else []
    if not logs:
        return None, f"shard logs not present under {run_dir}/{net}/shard_*/run.log", 0, 0
    per = {}
    for lg in logs:
        for img, kv in parse_log(lg.read_text(errors="replace"))["results"].items():
            if kv.get("net") == net:
                per[img] = kv
    if not per:
        return None, "no RESULT lines with layer_cycles", len(logs), 0
    imgs = sorted(per)
    lc = np.array([[int(x) for x in per[i]["layer_cycles"].split(",")] for i in imgs], np.int64)
    tc = np.array([int(per[i]["rtl_total"]) for i in imgs], np.int64)
    status = "ok" if len(imgs) == n_expected else f"partial: {len(imgs)} of {n_expected} images"
    return (lc, tc), status, len(logs), len(imgs)


def main_rtl(a) -> int:
    run_dir, commit = (Path(a.run_dir), "") if a.run_dir else default_run_dir()
    out_dir = Path(a.out_dir) if a.out_dir else bc.RESULTS_ROOT / "dryrun"
    if "dryrun" not in out_dir.resolve().parts:
        raise SystemExit(f"REFUSED: the RTL-sim spread is a laptop analysis; output only under a "
                         f"'dryrun' directory, not {out_dir}")
    try:
        head, dirty = bc.git_state()
    except Exception:  # noqa: BLE001
        head, dirty = "", True
    rows = []
    for net in a.nets:
        pkg = bc.load_package(a.data_dir, net)
        mc = pkg.model_cycles
        names = [L["name"] for L in pkg.net["layers"]]
        n_exp = a.limit or pkg.n
        got, status, shards, n = rtl_spread(run_dir, net, n_exp)
        if run_dir is not None and "dirty" in Path(run_dir).as_posix():
            status += "; run dir built from a dirty tree (harness test run, not the committed run)"
        base = dict(timestamp=bc.utc_now(), git_commit=head, git_dirty=dirty, net=net, source="rtl_sim",
                    label=RTL_LABEL, status=status, run_dir=str(run_dir), run_dir_git_commit=commit,
                    shards=shards, images_expected=n_exp)
        print(f"[rtl spread {net}] {run_dir}: {status} ({shards} shard logs, {n} images)")
        if got is None:
            rows.append({**base, "layer": "all", "result": "N/A"})
            continue
        tab = layer_table(names, [int(mc["layers"][l]["cycles"]) for l in range(len(names))],
                          int(mc["total"]["cycles"]), *got)
        for t in tab:
            if status != "ok":
                t["result"] = f"{t['result']} ({status})"
            rows.append({**base, **t})
            print(f"  {net:8} {t['layer']:8} model {t['model_cycles']:>8} min {t['cyc_min']:>8} "
                  f"max {t['cyc_max']:>8} distinct {t['cyc_distinct']:>3} {t['result']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "rtl_layer_spread.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RTL_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in RTL_FIELDS})
    print(f"wrote {p} ({len(rows)} rows, {RTL_LABEL}; not a committed result)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap)
    ap.add_argument("--npz-dir", default=None,
                    help="where A2/A3's hw_cycles_<net>.npz lives (default: the output dir)")
    ap.add_argument("--rerun", action="store_true", help="ignore existing npz, run every image")
    ap.add_argument("--max-errors", type=int, default=10)
    ap.add_argument("--rtl-sim", action="store_true",
                    help="laptop: RTL-sim spread from the full-10k shard logs -> results/dryrun/")
    ap.add_argument("--run-dir", default=None, help="--rtl-sim: <runs>/<tag>/verilator (default: "
                    "run_dir of v2/results/rtl_full10k.csv)")
    a = ap.parse_args(argv)
    return main_rtl(a) if a.rtl_sim else main_board(a)


if __name__ == "__main__":
    sys.exit(main())
