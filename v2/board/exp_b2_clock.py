#!/usr/bin/env python3
"""B2 clock sweep: pl_clk0 = 100..300 MHz in 25 MHz steps (9 points), set at runtime, <= closed clock.

    ./session.sh py exp_b2_clock.py [--clocks 100 150 200] [--max-mhz 200] [--net lenet5]
                                    [--images 100] [--window-s 60] [--power-repeats 3]
                                    [--rate-hz 10]
    python3 exp_b2_clock.py --backend model --window-s 2 --sensor mock   # dry run -> results/dryrun/

The timing-closed frequency comes from --max-mhz or DEPLOY_INFO.json 'bit_clock_mhz' (the
Vivado-reported pl_clk0 of the deployed build). Default sweep (board_common.b2_sweep_clocks):
100, 125, ..., 300 MHz (25 MHz steps), capped at the closed clock (the
top point is requested at the closed clock itself). A requested clock above it (+0.5 MHz
tolerance) is skipped and logged, never run; a read-back above it aborts the sweep. Per clock: set pl_clk0 (pynq Clocks.fclk0_mhz), read it back,
soft_reset + reload the net (WGT/QPARAM/DESC with readback), A2 on --images images (cycles must
equal the model at every clock: consistency check), then the power at this clock with the SAME
INA260 SOM-rail logger as B1 (power_log.run_power_protocol) in a REDUCED protocol: accelerator
phases only, idle / accel / idle (--window-s each) x --power-repeats (no CPU phases: the CPU
baseline does not depend on pl_clk0; B1 has it). P_idle, dP, time/image, energy/image as in B1,
computed by power_log. Label "SOM-rail power (INA260)" (not accelerator-only, not board input).
The original clock is restored at the end.
Cycle identity (explicit PASS/FAIL): per clock and layer (+ total) the accelerator LAYER_CYC /
TOTAL_CYC over the --images images must have one value, equal to the model, and that value must be
identical at every clock of the sweep (hw_b2_cycles.csv, one row per clock x layer; summary
columns cycle_check / cycles_identical_across_clocks in hw_b2_clock.csv).
Power-vs-clock fit (script only, never typed): P = P_static + k*f by ordinary least squares on
the per-clock MEAN power (f = read-back pl_clk0 in MHz), for the accelerator-phase absolute power
P_accel, for dP_accel = P_accel - P_idle, and for P_idle; P_static (W), k (W/MHz), standard
errors, 95 % confidence intervals (Student t, n-2 dof), R^2, n points; a supplementary fit on the
per-repeat values (basis per_repeat). Label "SOM-rail power (INA260)". -> hw_b2_fit.csv.
Recompute the fit from existing summary CSVs (no device): --fit-only --in-dir <dir>.
Dry run only: --mock-slope-w-per-mhz S (+ --mock-noise-w) makes the mock sensor clock-dependent
(synthetic levels, NOT measurements) so that the fit path is exercised.
Writes hw_b2_clock.csv (one row per clock, incl. the INA260 mean/std over repeats),
hw_b2_power_ina260_{samples,phases,summary}_<net>_<NNN>mhz.csv per clock, hw_b2_cycles.csv,
hw_b2_fit.csv. The INA260 is the only power source (no external meter, user decision 2026-09-29).
"""
from __future__ import annotations

import os
import sys

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import board_common as bc  # noqa: E402
import exp_a2_a3_cycles as a2  # noqa: E402
import power_log as pl  # noqa: E402

FIELDS = ["clock_requested_mhz", "clock_readback_mhz", "max_closed_mhz", "images",
          "model_total_cycles", "hw_total_cycles", "hw_min", "hw_max", "cycles_equal_model",
          "latency_us", "wall_us_median", "wall_us_p95",
          "power_label", "ina260_backend", "ina260_repeats", "ina260_p_idle_w", "ina260_p_accel_w",
          "ina260_dp_w", "ina260_dp_std_w", "ina260_time_per_image_s",
          "ina260_energy_per_image_mj", "ina260_energy_per_image_std_mj", "ina260_rate_achieved_hz",
          "ina260_max_gap_s", "ina260_summary_csv", "model_layer_cycles", "hw_layer_cycles",
          "layers_equal_model", "cycles_identical_across_clocks", "cycle_check", "skipped_reason"]
CYC_FIELDS = ["clock_requested_mhz", "clock_readback_mhz", "images", "model_cycles", "hw_cycles",
              "hw_min", "hw_max", "hw_distinct", "hw_images_equal_model", "clocks_compared",
              "identical_across_clocks", "cycle_check"]
FIT_FIELDS = ["power_label", "quantity", "basis", "model", "fit_method", "n_points", "dof",
              "clocks_mhz", "p_static_w", "p_static_se_w", "p_static_ci95_lo_w", "p_static_ci95_hi_w",
              "k_w_per_mhz", "k_se_w_per_mhz", "k_ci95_lo_w_per_mhz", "k_ci95_hi_w_per_mhz",
              "k_mw_per_mhz", "r2", "resid_std_w", "t_crit_95", "intercept_meaning", "inputs", "note"]
FIT_METHOD = ("ordinary least squares y = P_static + k*f (numpy lstsq); f = read-back pl_clk0 (MHz); "
              "SE from the residual variance (n-2 dof); 95 % CI = estimate +- t(0.975, n-2) * SE")
FIT_QUANTITIES = {   # quantity -> (summary column, meaning of the intercept)
    "p_accel_w": ("accel_p_run_w", "P_static = extrapolated SOM-rail power at f -> 0 with the accelerator loop running"),
    "dp_accel_w": ("accel_dp_w", "intercept of dP_accel = P_accel - P_idle (clock-independent part of the loop's extra power)"),
    "p_idle_w": ("accel_p_idle_w", "P_static = extrapolated idle SOM-rail power at f -> 0"),
}
# two-sided 95 % Student t critical values t(0.975, dof); dof above the table -> the next lower
# tabulated dof (conservative), >= 1000 -> normal 1.960
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
         18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
         26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
         1000: 1.960}


def t_crit_95(dof: int) -> float:
    if dof < 1:
        return math.nan
    return _T975[max(k for k in _T975 if k <= dof)]


def fit_linear(xs, ys) -> dict:
    """OLS y = a + k*x. Returns a, k, their SEs and 95 % CIs, R^2, n, dof, residual std
    (NaN where undefined: n < 2 -> no fit; n == 2 -> exact line, no SE / CI)."""
    import numpy as np
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    n = int(x.size)
    nan = math.nan
    out = dict(n=n, dof=n - 2, a=nan, k=nan, a_se=nan, k_se=nan, a_lo=nan, a_hi=nan, k_lo=nan,
               k_hi=nan, r2=nan, resid_std=nan, t=nan)
    if n < 2 or np.ptp(x) == 0:
        return out
    A = np.column_stack([np.ones(n), x])
    (a, k), *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - (a + k * x)
    sst = float(((y - y.mean()) ** 2).sum())
    sse = float((res ** 2).sum())
    out.update(a=float(a), k=float(k), r2=(1.0 - sse / sst) if sst > 0 else nan)
    if n > 2:
        s2 = sse / (n - 2)
        sxx = float(((x - x.mean()) ** 2).sum())
        k_se = math.sqrt(s2 / sxx)
        a_se = math.sqrt(s2 * (1.0 / n + x.mean() ** 2 / sxx))
        t = t_crit_95(n - 2)
        out.update(k_se=k_se, a_se=a_se, t=t, a_lo=a - t * a_se, a_hi=a + t * a_se,
                   k_lo=k - t * k_se, k_hi=k + t * k_se, resid_std=math.sqrt(s2))
    return out


def fit_points(summaries: list[tuple[float, list[dict], str]]) -> dict:
    """summaries: [(read-back clock MHz, summary rows of that clock, file name)] ->
    {(quantity, basis): (xs, ys, files)}."""
    pts: dict = {}
    for clk, rows, fname in summaries:
        for q, (col, _) in FIT_QUANTITIES.items():
            for basis, kind in (("per_clock_mean", "mean"), ("per_repeat", "repeat")):
                for r in rows:
                    if r.get("row_kind") != kind or r.get(col) in ("", None):
                        continue
                    v = float(r[col])
                    if math.isnan(v):
                        continue
                    xs, ys, fs = pts.setdefault((q, basis), ([], [], []))
                    xs.append(float(clk))
                    ys.append(v)
                    if fname not in fs:
                        fs.append(fname)
    return pts


def fit_rows(pts: dict, base_meta) -> list[dict]:
    """hw_b2_fit.csv rows; base_meta(quantity, basis, n) -> metadata dict (CSV rule columns)."""
    g = lambda v, nd=6: "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"  # noqa: E731
    rows = []
    for q in FIT_QUANTITIES:
        for basis in ("per_clock_mean", "per_repeat"):
            xs, ys, files = pts.get((q, basis), ([], [], []))
            f = fit_linear(xs, ys)
            note = ("" if f["n"] > 2 and len(set(xs)) > 2 else
                    "fewer than 3 clocks: no standard error / CI" if len(set(xs)) >= 2 else
                    "fewer than 2 clocks: no fit")
            if basis == "per_repeat":
                note = ("supplementary: repeats of one clock are not independent clock points; "
                        "primary fit = per_clock_mean" + (f"; {note}" if note else ""))
            row = base_meta(q, basis, len(xs))
            row.update(power_label=pl.LABEL, quantity=q, basis=basis, model="P = P_static + k*f",
                       fit_method=FIT_METHOD, n_points=f["n"], dof=f["dof"] if f["n"] >= 2 else "",
                       clocks_mhz=" ".join(f"{c:.6f}" for c in sorted(set(xs))),
                       p_static_w=g(f["a"]), p_static_se_w=g(f["a_se"]),
                       p_static_ci95_lo_w=g(f["a_lo"]), p_static_ci95_hi_w=g(f["a_hi"]),
                       k_w_per_mhz=g(f["k"], 9), k_se_w_per_mhz=g(f["k_se"], 9),
                       k_ci95_lo_w_per_mhz=g(f["k_lo"], 9), k_ci95_hi_w_per_mhz=g(f["k_hi"], 9),
                       k_mw_per_mhz=g(f["k"] * 1e3 if not math.isnan(f["k"]) else math.nan, 6),
                       r2=g(f["r2"]), resid_std_w=g(f["resid_std"]), t_crit_95=g(f["t"], 3),
                       intercept_meaning=FIT_QUANTITIES[q][1], inputs=" ".join(files), note=note)
            rows.append(row)
    return rows


def cycle_identity(per_clock: list[tuple[float, float, list[dict]]]) -> tuple[list[dict], dict, bool]:
    """per_clock: [(requested, read-back, a2.cycle_rows rows)] -> (hw_b2_cycles.csv rows without
    metadata, {requested: (identical, check)}, all PASS). A layer passes at a clock when its
    counter had one value over the images, equal to the model, and that value is the same at
    every clock of the sweep."""
    vals: dict = {}
    for _, _, crows in per_clock:
        for r in crows:
            vals.setdefault(r["layer"], set()).add(r["hw_cycles"])
    out, per_req, all_ok = [], {}, True
    for req, act, crows in per_clock:
        ok_clk, ident_clk = True, True
        for r in crows:
            ident = len(vals[r["layer"]]) == 1
            ok = (ident and r["hw_distinct"] == 1 and r["images"] > 0
                  and r["hw_images_equal_model"] == r["images"] and r["hw_cycles"] == r["model_cycles"])
            ok_clk &= ok
            ident_clk &= ident
            out.append({"_meta_from": r, "clock_requested_mhz": req, "clock_readback_mhz": f"{act:.6f}",
                        "images": r["images"], "model_cycles": r["model_cycles"],
                        "hw_cycles": r["hw_cycles"], "hw_min": r["hw_min"], "hw_max": r["hw_max"],
                        "hw_distinct": r["hw_distinct"], "hw_images_equal_model": r["hw_images_equal_model"],
                        "clocks_compared": len(per_clock), "identical_across_clocks": ident,
                        "cycle_check": "PASS" if ok else "FAIL"})
        per_req[req] = (ident_clk, "PASS" if ok_clk else "FAIL")
        all_ok &= ok_clk
    return out, per_req, all_ok


def _read_csv(p: Path) -> list[dict]:
    with Path(p).open(newline="") as f:
        return list(csv.DictReader(f))


def fit_only(a) -> int:
    """Recompute hw_b2_fit.csv from <in-dir>/hw_b2_power_ina260_summary_*.csv (no device)."""
    ind = Path(a.in_dir)
    files = sorted(ind.glob(f"{pl.PREFIX_B2}_summary_*.csv"))
    if not files:
        raise SystemExit(f"[B2 fit] no {pl.PREFIX_B2}_summary_*.csv in {ind}")
    summ, meta0 = [], None
    for p in files:
        rows = _read_csv(p)
        m = [r for r in rows if r.get("row_kind") == "mean"]
        if not m:
            print(f"[B2 fit] {p.name}: no mean row, skipped")
            continue
        if a.net and m[0].get("net") != a.net:
            continue
        srcs = {r.get("source") for r in rows}
        if len(srcs) != 1:
            raise SystemExit(f"[B2 fit] {p.name}: mixed sources {srcs}")
        meta0 = meta0 or m[0]
        if m[0].get("source") != meta0.get("source"):
            raise SystemExit(f"[B2 fit] {p.name}: source {m[0].get('source')} != {meta0.get('source')}")
        summ.append((float(m[0]["clock_mhz"]), rows, p.name))
    if meta0 is None:
        raise SystemExit("[B2 fit] no usable summary")
    source = meta0["source"]
    out = bc.resolve_out_dir(source, a.out_dir)

    def base(q, basis, n):
        b = {k: meta0.get(k, "") for k in bc.META_COLUMNS + bc.EXTRA_META}
        b.update(timestamp=bc.utc_now(), layer="fit", clock_mhz="", duration_s="", num_inferences=n)
        return b
    rows = fit_rows(fit_points(summ), base)
    bc.write_csv(out / "hw_b2_fit.csv", rows, FIT_FIELDS, source)
    _print_fit(rows)
    return 0


def _print_fit(rows):
    print(f"[B2 fit] {pl.LABEL}: P = P_static + k*f (least squares; computed by this script)")
    for r in rows:
        print(f"  {r['quantity']:10} {r['basis']:15} n={r['n_points']:<3} P_static={r['p_static_w'] or '-':>10} W "
              f"(+-{r['p_static_se_w'] or '-'}) k={r['k_mw_per_mhz'] or '-':>10} mW/MHz "
              f"(+-{r['k_se_w_per_mhz'] or '-'} W/MHz) R2={r['r2'] or '-'} {r['note']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap, nets=False)
    ap.add_argument("--clocks", nargs="+", type=float, default=None,
                    help="default: board_common.b2_sweep_clocks(closed clock)")
    ap.add_argument("--max-mhz", type=float, default=None)
    ap.add_argument("--net", default="lenet5", choices=bc.NETS)
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--gap-s", type=float, default=10.0)
    ap.add_argument("--sample-s", type=float, default=0.5)
    ap.add_argument("--power-repeats", type=int, default=3,
                    help="INA260 idle/accel/idle repeats per clock")
    ap.add_argument("--rate-hz", type=float, default=pl.DEFAULT_RATE_HZ)
    ap.add_argument("--sensor", default="auto", choices=("auto",) + pl.SENSOR_ORDER + ("mock",))
    ap.add_argument("--fit-only", action="store_true",
                    help="no device: recompute hw_b2_fit.csv from --in-dir summary CSVs")
    ap.add_argument("--in-dir", default=None, help="--fit-only: dir with hw_b2_power_ina260_summary_*.csv")
    ap.add_argument("--mock-slope-w-per-mhz", type=float, default=0.0,
                    help="dry run (--sensor mock) only: synthetic clock-dependent mock levels")
    ap.add_argument("--mock-noise-w", type=float, default=0.0, help="dry run mock sensor noise std")
    a = ap.parse_args(argv)
    if a.fit_only:
        if not a.in_dir:
            raise SystemExit("--fit-only needs --in-dir")
        return fit_only(a)
    a.nets = [a.net]
    dev, be = bc.open_device(a)
    ctx = bc.RunContext(a, dev, be, "B2")
    ctx.check_clean()
    ctx.banner()
    max_mhz = a.max_mhz if a.max_mhz is not None else ctx.info.get("bit_clock_mhz")
    if max_mhz is None:
        raise SystemExit("closed clock unknown: pass --max-mhz (or deploy with DEPLOY_INFO.json)")
    max_mhz = float(max_mhz)
    if a.clocks is None:
        a.clocks = bc.b2_sweep_clocks(max_mhz)
    print(f"[B2] sweep {a.clocks} MHz (closed clock {max_mhz} MHz)")
    dry = ctx.source != bc.SOURCE_HW
    try:
        ina, plog = pl.discover(a.sensor, allow_hw=not dry, allow_mock=dry)
    except pl.SensorUnavailable as e:
        raise SystemExit(f"[B2] ERROR: {e}")
    for line in plog:
        print(f"[B2 power] probe: {line}")
    if (a.mock_slope_w_per_mhz or a.mock_noise_w) and (not dry or a.sensor != "mock"):
        raise SystemExit("[B2] --mock-slope-w-per-mhz / --mock-noise-w: dry run with --sensor mock only")
    print(f"[B2] closed clock {max_mhz} MHz; {pl.LABEL}: {ina.describe()}")
    pkg = ctx.package(a.net)
    clk0 = dev.fclk0_mhz()
    rows = []
    per_clock, summaries = [], []
    ok = True
    try:
        for req in a.clocks:
            row = None
            if req > max_mhz + bc.CLOCK_TOL_MHZ:
                print(f"[B2] SKIP {req:g} MHz: above the closed clock {max_mhz} MHz")
                row = ctx.meta(a.net, "all", "", 0, clock_mhz="")
                row.update(clock_requested_mhz=req, max_closed_mhz=max_mhz,
                           skipped_reason=f"above closed clock {max_mhz} MHz")
                rows.append(row)
                continue
            actual = dev.set_fclk0(req)
            print(f"[B2] pl_clk0 requested {req:g} MHz -> read back {actual:.6f} MHz")
            if actual > max_mhz + bc.CLOCK_TOL_MHZ:
                raise SystemExit(f"[B2] ABORT: read-back pl_clk0 {actual:.6f} MHz > closed clock "
                                 f"{max_mhz} MHz (+{bc.CLOCK_TOL_MHZ}); nothing run at this clock")
            dev.soft_reset()
            dev.load_net(pkg)
            t0 = time.perf_counter()
            crows, _, cyc_ok = a2.cycle_rows(ctx, dev, pkg, range(min(a.images, pkg.n)), actual,
                                             tag=f"B2 {req:g}")
            ok &= cyc_ok
            per_clock.append((req, actual, crows))
            tot = crows[-1]
            sens = ina
            if a.mock_slope_w_per_mhz or a.mock_noise_w:     # dry run only (checked above)
                sl = a.mock_slope_w_per_mhz
                sens = pl.MockSensor(levels_w={"idle": 1.0 + 0.25 * sl * actual,
                                               "accel": 1.25 + sl * actual,
                                               "control": 1.05 + 0.3 * sl * actual},
                                     noise_w=a.mock_noise_w, seed=int(round(req)))
            res = pl.run_power_protocol(
                ctx, a.net, None, accel_fn=pl.gos_workload(dev, pkg), sensor=sens,
                phase_s=a.window_s, repeats=a.power_repeats, rate_hz=a.rate_hz,
                prefix=pl.PREFIX_B2, tag=f"_{a.net}_{pl.clock_tag(req)}", clock_mhz=actual,
                accel_label=f"GosDevice.infer {a.net} + TOTAL_CYC read + PS dequant ({dev.host_path_desc}) @ {actual:.6f} MHz",
                label=f"B2 {req:g}MHz")
            # fit input = the written summary CSV (same numbers a later --fit-only recomputation reads)
            summaries.append((actual, _read_csv(res["files"]["summary"]), Path(res["files"]["summary"]).name))
            mean = next(x for x in res["summary"] if x["row_kind"] == "mean")
            std = next(x for x in res["summary"] if x["row_kind"] == "std")
            n_ina = sum(int(p["images"]) for p in res["phases"])
            row = ctx.meta(a.net, "all", time.perf_counter() - t0,
                           tot["images"] + n_ina,
                           clock_mhz=actual)
            f = pl._f
            row.update(clock_requested_mhz=req, clock_readback_mhz=f"{actual:.6f}",
                       max_closed_mhz=max_mhz, images=tot["images"],
                       model_total_cycles=tot["model_cycles"], hw_total_cycles=tot["hw_cycles"],
                       hw_min=tot["hw_min"], hw_max=tot["hw_max"], cycles_equal_model=cyc_ok,
                       latency_us=tot["hw_us"], wall_us_median=tot["wall_us_median"],
                       wall_us_p95=tot["wall_us_p95"],
                       power_label=pl.LABEL, ina260_backend=res["sensor_backend"],
                       ina260_repeats=a.power_repeats,
                       ina260_p_idle_w=f(mean.get("accel_p_idle_w")),
                       ina260_p_accel_w=f(mean.get("accel_p_run_w")),
                       ina260_dp_w=f(mean.get("accel_dp_w")), ina260_dp_std_w=f(std.get("accel_dp_w")),
                       ina260_time_per_image_s=f(mean.get("accel_time_per_image_s"), 9),
                       ina260_energy_per_image_mj=f(mean.get("accel_energy_per_image_mj")),
                       ina260_energy_per_image_std_mj=f(std.get("accel_energy_per_image_mj")),
                       ina260_rate_achieved_hz=f(res["rate_achieved_hz"], 3),
                       ina260_max_gap_s=f(res["max_gap_s"], 4),
                       ina260_summary_csv=Path(res["files"]["summary"]).name,
                       model_layer_cycles=",".join(str(x["model_cycles"]) for x in crows[:-1]),
                       hw_layer_cycles=",".join(str(x["hw_cycles"]) for x in crows[:-1]),
                       layers_equal_model=all(x["hw_images_equal_model"] == x["images"] for x in crows))
            rows.append(row)
    finally:
        back = dev.set_fclk0(clk0)
        print(f"[B2] restored pl_clk0 to {back:.6f} MHz (was {clk0:.6f})")
    cyc, per_req, ident_ok = cycle_identity(per_clock)
    ok &= ident_ok
    for r in rows:
        if r.get("clock_requested_mhz") in per_req:
            r["cycles_identical_across_clocks"], r["cycle_check"] = per_req[r["clock_requested_mhz"]]
    crow_out = []
    for c in cyc:
        src = c.pop("_meta_from")
        m = {k: src.get(k, "") for k in bc.META_COLUMNS + bc.EXTRA_META}
        crow_out.append({**m, **c})
    ctx.csv("hw_b2_clock.csv", rows, FIELDS)
    ctx.csv("hw_b2_cycles.csv", crow_out, CYC_FIELDS)
    print(f"[B2] per-layer cycles identical to the model and across {len(per_clock)} clocks: "
          f"{'PASS' if ident_ok else 'FAIL'}")
    for c in crow_out:
        print(f"  {float(c['clock_requested_mhz']):>11.6f} MHz {c['layer']:8} model {c['model_cycles']:>8} "
              f"hw {str(c['hw_cycles']):>8} distinct {c['hw_distinct']} {c['cycle_check']}")

    def base(q, basis, n):
        return ctx.meta(a.net, "fit", "", n, clock_mhz="")
    frows = fit_rows(fit_points(summaries), base)
    ctx.csv("hw_b2_fit.csv", frows, FIT_FIELDS)
    _print_fit(frows)
    print("B2 cycles:", "equal to model at every clock" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
