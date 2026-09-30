#!/usr/bin/env python3
"""Soak test: back-to-back inference for --duration-s (default 30 min) at the deployed clock.

    ./session.sh py exp_soak.py [--duration-s 1800] [--nets lenet5 cifar10] [--block-s 60]
                                [--clock-choice results/hw_clock_choice.json] [--expect-clock-mhz F]
    python3 exp_soak.py --backend model --duration-s 20 --bucket-s 5    # dry run -> results/dryrun/

Runs at whatever pl_clk0 the loaded bitstream runs at: the performance clock is chosen before the
session by clock_fallback.py (300 MHz build, else 250 MHz; the decision JSON is recorded in every
row when --clock-choice is given). --expect-clock-mhz refuses to start if the read-back clock is
not within 0.5 MHz of it. The net alternates every --block-s seconds (several nets) -- each switch
reloads WGT/QPARAM/DESC with read-back (a load error is counted); images cycle through the test
set of each net. Every job is checked:
  logits      LOGIT[0..OC-1] bit-exact vs golden_logits (v2/model golden)
  cycles      TOTAL_CYC == model total, LAYER_CYC[l] == model layer l, MAC_ACTIVE == model, STALL == 0
  errors      GosJobError (STATUS.error, ERR_CODE decoded), GosTimeout (poll timeout), other GosError
  STATUS      STATUS read after every job must be exactly DONE (0x2)
  err_flags   PS_BUSY_VIOLATION != 0
Per net and layer (+ total): min / max / number of distinct cycle values over all jobs (constant
memory: distinct values are tracked up to 64). Throughput per --bucket-s bucket (default 60 s) per
net. Die temperature every --temp-every-s if board_common provides an AMS reader (looked up by
board_env.read_die_temp (max over the AMS channels), else board_common by name,
TEMP_READER_NAMES, or --temp-reader NAME), else "unavailable"; never read in a dry run.
Not resumable (a soak is one uninterrupted run; a rerun restarts). More than --max-errors errors
abort the soak (FAIL).
Writes hw_soak.csv (row_kind net / layer / all) and hw_soak_minutes.csv (row_kind bucket).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import board_common as bc
import gos_driver as D

TEMP_READER_NAMES = ("read_die_temp_c", "read_die_temp", "read_ams_temp", "read_ams_temps",
                     "ams_temperature", "read_temperature")
DISTINCT_CAP = 64
FIELDS = ["row_kind", "duration_target_s", "images", "jobs_ok", "logit_mismatch", "cycle_mismatch",
          "layer_cycle_mismatch", "mac_mismatch", "stall_nonzero", "job_errors", "timeouts",
          "other_errors", "status_errors", "err_flags", "load_errors", "net_loads",
          "model_cycles", "cyc_min", "cyc_max", "cyc_distinct", "cyc_spread", "inf_per_s",
          "temp_source", "temp_c_start", "temp_c_end", "temp_c_min", "temp_c_max", "temp_samples",
          "clock_expected_mhz", "clock_choice_file", "clock_choice_bit", "clock_choice_mhz",
          "clock_fell_back", "clock_choice_reason", "aborted", "first_errors", "result"]
BUCKET_FIELDS = ["row_kind", "bucket", "t_start_s", "t_end_s", "images", "inf_per_s",
                 "errors", "mismatches", "temp_c_mean", "temp_c_max", "temp_samples", "temp_source"]


# ---- temperature -----------------------------------------------------------------------------
def find_temp_reader(name: str | None = None, dry: bool = False):
    """(callable or None, source text). The AMS reader belongs to board_common (other owner);
    it may return a float (deg C) or a dict {sensor: deg C} (the maximum is used)."""
    if dry:
        return None, "unavailable (dry run: no die temperature)"
    if name == "none":
        return None, "unavailable (disabled by --temp-reader none)"
    if not name:          # the AMS reader of the session environment module (other owner)
        try:
            import board_env
            f = getattr(board_env, "read_die_temp", None)
            if callable(f):
                probe = f()
                src = probe.get("source", "") if isinstance(probe, dict) else ""
                if isinstance(probe, dict) and probe.get("max_c") is None:
                    return None, f"unavailable (board_env.read_die_temp: {src or 'no AMS channel'})"
                return f, f"board_env.read_die_temp ({src})" if src else "board_env.read_die_temp"
        except ImportError:
            pass
    for n in ([name] if name else TEMP_READER_NAMES):
        f = getattr(bc, n, None)
        if callable(f):
            return f, f"board_common.{n}"
    return None, "unavailable (no AMS reader in board_common)"


def read_temp(fn) -> float | None:
    try:
        v = fn()
    except Exception as e:  # noqa: BLE001 - temperature is informational
        print(f"[soak] temperature read failed: {e}", file=sys.stderr)
        return None
    if isinstance(v, dict) and "max_c" in v:          # board_env.read_die_temp
        return None if v["max_c"] is None else float(v["max_c"])
    if isinstance(v, dict):
        vals = [float(x) for x in v.values() if isinstance(x, (int, float))]
        return max(vals) if vals else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---- statistics ----------------------------------------------------------------------------------
class NetStats:
    def __init__(self, pkg):
        self.pkg = pkg
        mc = pkg.model_cycles
        self.nl = int(pkg.net["n_layers"])
        self.names = [L["name"] for L in pkg.net["layers"]]
        self.model_layers = np.array([int(mc["layers"][l]["cycles"]) for l in range(self.nl)], np.int64)
        self.model_total = int(mc["total"]["cycles"])
        self.model_mac = int(mc["total"]["mac_active"])
        self.lmin = np.full(self.nl, np.iinfo(np.int64).max, np.int64)
        self.lmax = np.full(self.nl, -1, np.int64)
        self.ldist = [set() for _ in range(self.nl)]
        self.tmin, self.tmax, self.tdist = np.iinfo(np.int64).max, -1, set()
        self.c = dict(images=0, jobs_ok=0, logit_mismatch=0, cycle_mismatch=0, layer_cycle_mismatch=0,
                      mac_mismatch=0, stall_nonzero=0, job_errors=0, timeouts=0, other_errors=0,
                      status_errors=0, err_flags=0, load_errors=0, net_loads=0)
        self.first_errors: list[str] = []
        self.next_img = 0
        self.busy_s = 0.0

    def note(self, msg: str):
        if len(self.first_errors) < 10:
            self.first_errors.append(msg)

    def job(self, img: int, r, status: int) -> int:
        """Record one completed job; returns the number of mismatches/flags found (0 = clean)."""
        c = self.c
        c["jobs_ok"] += 1
        bad = 0
        if not np.array_equal(np.asarray(r.logits), self.pkg.golden_logits[img]):
            c["logit_mismatch"] += 1
            bad += 1
            self.note(f"img {img}: logits != golden")
        lc = np.asarray(r.layer_cyc, np.int64)
        np.minimum(self.lmin, lc, out=self.lmin)
        np.maximum(self.lmax, lc, out=self.lmax)
        for l in range(self.nl):
            if len(self.ldist[l]) <= DISTINCT_CAP:
                self.ldist[l].add(int(lc[l]))
        t = int(r.total_cyc)
        self.tmin, self.tmax = min(self.tmin, t), max(self.tmax, t)
        if len(self.tdist) <= DISTINCT_CAP:
            self.tdist.add(t)
        if t != self.model_total:
            c["cycle_mismatch"] += 1
            bad += 1
            self.note(f"img {img}: TOTAL_CYC {t} != model {self.model_total}")
        if not np.array_equal(lc, self.model_layers):
            c["layer_cycle_mismatch"] += 1
            bad += 1
            self.note(f"img {img}: LAYER_CYC {lc.tolist()} != model {self.model_layers.tolist()}")
        if int(r.mac_active) != self.model_mac:
            c["mac_mismatch"] += 1
            bad += 1
            self.note(f"img {img}: MAC_ACTIVE {r.mac_active} != model {self.model_mac}")
        if int(r.stall) != 0:
            c["stall_nonzero"] += 1
            bad += 1
            self.note(f"img {img}: STALL {r.stall}")
        if int(r.violation or 0) != 0:
            c["err_flags"] += 1
            bad += 1
            self.note(f"img {img}: PS_BUSY_VIOLATION 0x{r.violation:X} {D.decode_violation(r.violation)}")
        if status != D.ST_DONE:
            c["status_errors"] += 1
            bad += 1
            self.note(f"img {img}: STATUS after job 0x{status:X} != DONE 0x{D.ST_DONE:X}")
        return bad

    def errors(self) -> int:
        c = self.c
        return sum(c[k] for k in ("logit_mismatch", "cycle_mismatch", "layer_cycle_mismatch",
                                  "mac_mismatch", "stall_nonzero", "job_errors", "timeouts",
                                  "other_errors", "status_errors", "err_flags", "load_errors"))


def _dist(s: set) -> str:
    return f">{DISTINCT_CAP}" if len(s) > DISTINCT_CAP else str(len(s))


def _temps_summary(samples: list[tuple[float, float]]) -> dict:
    v = [x for _, x in samples]
    if not v:
        return dict(temp_c_start="", temp_c_end="", temp_c_min="", temp_c_max="", temp_samples=0)
    return dict(temp_c_start=f"{v[0]:.2f}", temp_c_end=f"{v[-1]:.2f}", temp_c_min=f"{min(v):.2f}",
                temp_c_max=f"{max(v):.2f}", temp_samples=len(v))


# ---- run -------------------------------------------------------------------------------------------
def soak(dev, pkgs: list, duration_s: float, block_s: float, bucket_s: float, temp_fn=None,
         temp_every_s: float = 10.0, max_errors: int = 100, clock=time.monotonic) -> dict:
    """The soak loop (no I/O besides the device). Returns stats, buckets, temperature samples."""
    stats = {p.name: NetStats(p) for p in pkgs}
    buckets: dict[tuple[int, str], dict] = {}
    temps: list[tuple[float, float]] = []
    t0 = clock()
    deadline = t0 + duration_s
    k = 0
    cur = None
    next_switch = t0
    next_temp = t0
    aborted = ""
    total_err = 0

    def load(p):
        st = stats[p.name]
        for attempt in (1, 2):
            try:
                dev.soft_reset()
                dev.load_net(p, verify=True)
                return True
            except D.GosError as e:
                st.c["load_errors"] += 1
                st.note(f"load_net {p.name} attempt {attempt}: {type(e).__name__}: {e}")
        return False

    while True:
        now = clock()
        if now >= deadline:
            break
        if temp_fn is not None and now >= next_temp:
            tv = read_temp(temp_fn)
            if tv is not None:
                temps.append((now - t0, tv))
            next_temp = now + temp_every_s
        if cur is None or (len(pkgs) > 1 and now >= next_switch):
            nxt = pkgs[k % len(pkgs)]
            k += 1
            if cur is None or nxt.name != cur.name:
                if not load(nxt):
                    total_err += 1
                    if total_err > max_errors:
                        aborted = f"more than {max_errors} errors"
                        break
                    continue
                stats[nxt.name].c["net_loads"] += 1
            cur = nxt
            next_switch = now + block_s
        st = stats[cur.name]
        img = st.next_img
        st.next_img = (img + 1) % cur.n
        st.c["images"] += 1
        tj = clock()
        bad = 0
        try:
            r = dev.infer(cur.x_act[img], read_counters=True)
            status = dev.status()
            bad = st.job(img, r, status)
        except D.GosJobError as e:
            st.c["job_errors"] += 1
            st.c["status_errors"] += 1
            st.note(f"img {img}: {e}")
            bad = 1
            dev.recover()
        except D.GosTimeout as e:
            st.c["timeouts"] += 1
            st.note(f"img {img}: {e}")
            bad = 1
            dev.recover()
        except D.GosError as e:
            st.c["other_errors"] += 1
            st.note(f"img {img}: {type(e).__name__}: {e}")
            bad = 1
            dev.recover()
        te = clock()
        st.busy_s += te - tj
        b = int((te - t0) // bucket_s)
        bk = buckets.setdefault((b, cur.name), {"images": 0, "errors": 0, "mismatches": 0})
        bk["images"] += 1
        bk["mismatches"] += bad
        if bad:
            total_err += 1
            bk["errors"] += 1
            if total_err > max_errors:
                aborted = f"more than {max_errors} errors"
                break
    elapsed = clock() - t0
    return {"stats": stats, "buckets": buckets, "temps": temps, "elapsed_s": elapsed,
            "aborted": aborted}


def build_rows(ctx, res: dict, a, clk: float, temp_src: str, choice: dict | None) -> tuple[list, list]:
    stats, elapsed = res["stats"], res["elapsed_s"]
    tsum = _temps_summary(res["temps"])
    ch = choice or {}
    common = dict(duration_target_s=f"{a.duration_s:g}", temp_source=temp_src, **tsum,
                  clock_expected_mhz="" if a.expect_clock_mhz is None else f"{a.expect_clock_mhz:.6f}",
                  clock_choice_file=a.clock_choice or "",
                  clock_choice_bit=ch.get("bit", "") if choice else "none recorded",
                  clock_choice_mhz=ch.get("clock_mhz", ""), clock_fell_back=ch.get("fell_back", ""),
                  clock_choice_reason=ch.get("reason", ""), aborted=res["aborted"])
    rows = []
    tot = {k: 0 for k in next(iter(stats.values())).c}
    all_ok = not res["aborted"]
    for net, st in stats.items():
        for kk, v in st.c.items():
            tot[kk] += v
        n_ok = st.c["jobs_ok"]
        ok = st.errors() == 0 and st.c["images"] > 0 and not res["aborted"]
        all_ok &= ok
        for l, name in enumerate(st.names + ["total"]):
            if name == "total":
                mn, mx, ds, model = st.tmin, st.tmax, st.tdist, st.model_total
            else:
                mn, mx, ds, model = int(st.lmin[l]), int(st.lmax[l]), st.ldist[l], int(st.model_layers[l])
            has = n_ok > 0
            ok_l = has and mn == mx == model and len(ds) == 1
            row = ctx.meta(net, name, elapsed, st.c["images"], clock_mhz=clk)
            row.update(common, row_kind="layer", images=st.c["images"], jobs_ok=n_ok, model_cycles=model,
                       cyc_min=mn if has else "", cyc_max=mx if has else "",
                       cyc_distinct=_dist(ds), cyc_spread=(mx - mn) if has else "",
                       result="PASS" if ok_l else "FAIL")
            rows.append(row)
        row = ctx.meta(net, "all", elapsed, st.c["images"], clock_mhz=clk)
        row.update(common, **st.c, row_kind="net", model_cycles=st.model_total,
                   cyc_min=st.tmin if n_ok else "", cyc_max=st.tmax if n_ok else "",
                   cyc_distinct=_dist(st.tdist), cyc_spread=(st.tmax - st.tmin) if n_ok else "",
                   inf_per_s=f"{st.c['images'] / elapsed:.3f}" if elapsed > 0 else "",
                   first_errors=" | ".join(st.first_errors), result="PASS" if ok else "FAIL")
        rows.append(row)
    n_all = tot["images"]
    row = ctx.meta("all", "all", elapsed, n_all, clock_mhz=clk)
    row.update(common, **tot, row_kind="all",
               inf_per_s=f"{n_all / elapsed:.3f}" if elapsed > 0 else "",
               result="PASS" if all_ok and n_all > 0 else "FAIL")
    rows.append(row)
    brows = []
    temps = res["temps"]
    last_b = max((b for b, _ in res["buckets"]), default=-1)
    for (b, net) in sorted(res["buckets"]):
        bk = res["buckets"][(b, net)]
        t_s, t_e = b * a.bucket_s, min((b + 1) * a.bucket_s, elapsed)
        tb = [v for t, v in temps if t_s <= t < (b + 1) * a.bucket_s]
        # throughput of the net over the bucket's wall time (with several nets alternating in
        # --block-s blocks, a net may occupy only part of a bucket; see the images column)
        dur = t_e - t_s if b == last_b else a.bucket_s
        row = ctx.meta(net, "all", dur, bk["images"], clock_mhz=clk)
        row.update(row_kind="bucket", bucket=b, t_start_s=f"{t_s:.3f}", t_end_s=f"{t_e:.3f}",
                   images=bk["images"], inf_per_s=f"{bk['images'] / dur:.3f}" if dur > 0 else "",
                   errors=bk["errors"], mismatches=bk["mismatches"],
                   temp_c_mean=f"{sum(tb) / len(tb):.2f}" if tb else "",
                   temp_c_max=f"{max(tb):.2f}" if tb else "", temp_samples=len(tb), temp_source=temp_src)
        brows.append(row)
    return rows, brows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap)
    ap.add_argument("--duration-s", type=float, default=1800.0)
    ap.add_argument("--block-s", type=float, default=60.0,
                    help="alternate the nets every block (s); one net = no switching")
    ap.add_argument("--bucket-s", type=float, default=60.0, help="throughput bucket (s)")
    ap.add_argument("--temp-every-s", type=float, default=10.0)
    ap.add_argument("--temp-reader", default=None,
                    help=f"board_common function name (default: first of {TEMP_READER_NAMES}); "
                    "'none' disables")
    ap.add_argument("--max-errors", type=int, default=100)
    ap.add_argument("--expect-clock-mhz", type=float, default=None,
                    help="refuse to start unless pl_clk0 reads back within 0.5 MHz of this")
    ap.add_argument("--clock-choice", default=None,
                    help="clock_fallback decision JSON to record (absent file -> 'none recorded')")
    a = ap.parse_args(argv)
    dev, be = bc.open_device(a)
    ctx = bc.RunContext(a, dev, be, "SOAK")
    ctx.check_clean()
    ctx.banner()
    clk = dev.fclk0_mhz()
    if a.expect_clock_mhz is not None and abs(clk - a.expect_clock_mhz) > bc.CLOCK_TOL_MHZ:
        raise SystemExit(f"[soak] REFUSED: pl_clk0 reads back {clk:.6f} MHz, expected "
                         f"{a.expect_clock_mhz:.6f} (+-{bc.CLOCK_TOL_MHZ})")
    choice = None
    if a.clock_choice:
        import clock_fallback as cf
        choice = cf.read_choice(a.clock_choice)
    dry = ctx.source != bc.SOURCE_HW
    temp_fn, temp_src = find_temp_reader(a.temp_reader, dry=dry)
    pkgs = [ctx.package(n) for n in a.nets]
    print(f"[soak] {a.duration_s:g} s at {clk:.6f} MHz ({ctx.clock_source}); nets {a.nets} "
          f"alternating every {a.block_s:g} s; temperature: {temp_src}; clock choice: "
          f"{json.dumps({k: choice.get(k) for k in ('bit', 'clock_mhz', 'fell_back')}) if choice else 'none recorded'}")
    res = soak(dev, pkgs, a.duration_s, a.block_s, a.bucket_s, temp_fn, a.temp_every_s, a.max_errors)
    rows, brows = build_rows(ctx, res, a, clk, temp_src, choice)
    ctx.csv("hw_soak.csv", rows, FIELDS)
    ctx.csv("hw_soak_minutes.csv", brows, BUCKET_FIELDS)
    allr = rows[-1]
    print(f"[soak] {allr['images']} jobs in {res['elapsed_s']:.1f} s ({allr['inf_per_s']} inf/s): "
          f"logit mismatches {allr['logit_mismatch']}, cycle mismatches {allr['cycle_mismatch']} "
          f"(layer {allr['layer_cycle_mismatch']}), job errors {allr['job_errors']}, timeouts "
          f"{allr['timeouts']}, STATUS errors {allr['status_errors']}, err_flags {allr['err_flags']}, "
          f"load errors {allr['load_errors']}{', ABORTED: ' + res['aborted'] if res['aborted'] else ''}")
    for r in rows:
        if r["row_kind"] == "layer":
            print(f"  {r['net']:8} {r['layer']:14} model {r['model_cycles']:>8} min {r['cyc_min']:>8} "
                  f"max {r['cyc_max']:>8} distinct {r['cyc_distinct']:>3} {r['result']}")
    print("SOAK:", allr["result"])
    return 0 if allr["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
