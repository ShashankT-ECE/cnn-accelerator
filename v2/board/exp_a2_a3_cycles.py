#!/usr/bin/env python3
"""A2 latency + A3 three-way cycle agreement (model / RTL simulation / accelerator).

    ./session.sh py exp_a2_a3_cycles.py [--nets ...] [--limit N]
    python3 exp_a2_a3_cycles.py --backend model       # dry run -> results/dryrun/

Runs every image (default) and reads LAYER_CYC[l] and TOTAL_CYC after each job. Per net and layer
(+ a 'total' row): model_cycles (gos_cycle_model, from model_cycles.json), rtl_cycles (RTL
simulation network jobs, packaged rtl_network.csv; single-layer jobs from rtl_cycles.csv as a
cross-check), hw_cycles (the value if identical on every image, else the median) with min/max and
the number of distinct values over images (determinism), rtl_err_pct and hw_err_pct vs the model,
µs at f_used (= the pl_clk0 PLL read-back, the clock of record (DECISIONS D20); f_meas from exp_fclk_cal.py is
a cross-check column only; f_readback_mhz / f_meas_mhz / f_used_mhz / f_used_source columns,
hw_us_readback = hw_us), and for the total row MAC_ACTIVE/STALL
ranges and the wall-clock time per image (input write + clear + start/poll + logit read + counter
read, perf_counter_ns) after discarding the first --wall-warmup images: median with the
distribution-free order-statistic 95 % CI (stats.py), p5, p95, p99.
Writes hw_a2_a3_cycles.csv and hw_cycles_<net>.npz (per-image counters and phase times).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import board_common as bc
import stats

FIELDS = ["images", "model_cycles", "rtl_cycles", "rtl_single_layer_cycles", "hw_cycles",
          "hw_min", "hw_max", "hw_distinct", "hw_images_equal_model", "rtl_err_pct",
          "hw_err_pct", "model_us", "hw_us", "hw_us_readback", "hw_mac_active_min",
          "hw_mac_active_max", "hw_stall_max", "wall_us_median", "wall_us_ci_lo", "wall_us_ci_hi",
          "wall_ci_coverage", "wall_ci_method", "wall_us_p5", "wall_us_p50", "wall_us_p95", "wall_us_p99",
          "wall_images", "wall_warmup_discarded", "wall_repeats_ok", "job_errors", *bc.CLOCK_COLS]
WALL_WARMUP = 50


def _err(x, ref):
    return "" if x in ("", None) else f"{100.0 * (x - ref) / ref:.6f}"


def cycle_rows(ctx, dev, pkg, idx, clk: float, tag: str = "A2",
               wall_warmup: int = WALL_WARMUP) -> tuple[list[dict], dict, bool]:
    """Run the images, return (rows, raw per-image arrays, all-equal-to-model flag).
    clk = the read-back clock; µs use ctx.f_used(clk) = clk (clock of record)."""
    res = bc.infer_loop(dev, pkg, idx, read_counters=True, tag=f"{tag} {pkg.name}")
    good = res["ok"]
    n = int(good.sum())
    mc = pkg.model_cycles
    rtl = bc.rtl_cycles_for(pkg)
    names = [L["name"] for L in pkg.net["layers"]]
    rows, all_ok = [], bool(good.all())
    wall_all = (res["t_write_ns"] + res.get("t_clear_ns", 0) + res["t_run_ns"] + res["t_logit_ns"]
                + res["t_counter_ns"]) / 1e3
    wmask = good.copy()
    wmask[:min(wall_warmup, max(0, len(wmask) - 1))] = False
    wall = wall_all[wmask]
    ws = stats.summarize(wall) if wall.size else None
    f_used, _ = ctx.f_used(clk)
    ccols = ctx.clock_cols(clk)
    for l, name in enumerate(names + ["total"]):
        if name == "total":
            hw = res["total_cyc"][good]
            model = mc["total"]["cycles"]
            rtl_v, rtl_single = bc.one(rtl["total"]), ""
        else:
            hw = res["layer_cyc"][good, l]
            model = mc["layers"][l]["cycles"]
            rtl_v = bc.one(rtl["layers"].get(l, set()))
            rtl_single = bc.one(rtl["layer_single"].get(f"L{l}_{name}", set()))
        uniq = np.unique(hw)
        hw_v = int(uniq[0]) if uniq.size == 1 else (float(np.median(hw)) if n else "")
        eq = int((hw == model).sum())
        all_ok &= eq == n and rtl_v == model
        row = ctx.meta(pkg.name, name, res["duration_s"], len(idx), clock_mhz=clk)
        row.update(images=n, model_cycles=model, rtl_cycles=rtl_v,
                   rtl_single_layer_cycles=rtl_single, hw_cycles=hw_v,
                   hw_min=int(hw.min()) if n else "", hw_max=int(hw.max()) if n else "",
                   hw_distinct=int(uniq.size), hw_images_equal_model=eq,
                   rtl_err_pct=_err(rtl_v, model), hw_err_pct=_err(hw_v, model),
                   model_us=f"{model / f_used:.4f}",
                   hw_us=f"{hw_v / f_used:.4f}" if hw_v != "" else "",
                   hw_us_readback=f"{hw_v / clk:.4f}" if hw_v != "" else "",
                   job_errors=len(res["errors"]), **ccols)
        if name == "total" and n:
            row.update(hw_mac_active_min=int(res["mac_active"][good].min()),
                       hw_mac_active_max=int(res["mac_active"][good].max()),
                       hw_stall_max=int(res["stall"][good].max()))
            if ws is not None:
                row.update(wall_us_median=stats.fmt(ws["median"]), wall_us_ci_lo=stats.fmt(ws["ci_lo"]),
                           wall_us_ci_hi=stats.fmt(ws["ci_hi"]),
                           wall_ci_coverage=stats.fmt(ws["ci_coverage"], 4), wall_ci_method=ws["ci_method"],
                           wall_us_p5=stats.fmt(ws["p5"]), wall_us_p50=stats.fmt(ws["median"]),
                           wall_us_p95=stats.fmt(ws["p95"]),
                           wall_us_p99=stats.fmt(ws["p99"]), wall_images=ws["n"],
                           wall_warmup_discarded=int(good.sum()) - ws["n"],
                           wall_repeats_ok=ws["repeats_ok"])
            all_ok &= (row["hw_mac_active_min"] == row["hw_mac_active_max"]
                       == mc["total"]["mac_active"] and row["hw_stall_max"] == 0)
        rows.append(row)
    return rows, res, all_ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap)
    ap.add_argument("--wall-warmup", type=int, default=WALL_WARMUP,
                    help="images discarded from the wall-clock statistics (cycles use every image)")
    a = ap.parse_args(argv)
    dev, be = bc.open_device(a)
    ctx = bc.RunContext(a, dev, be, "A2/A3")
    ctx.check_clean()
    ctx.banner()
    rows, ok = [], True
    for net in a.nets:
        pkg = ctx.package(net)
        dev.load_net(pkg)
        clk = dev.fclk0_mhz()
        r, res, nok = cycle_rows(ctx, dev, pkg, bc.image_range(pkg, a.limit), clk,
                                 wall_warmup=a.wall_warmup)
        rows += r
        ok &= nok
        np.savez_compressed(ctx.path(f"hw_cycles_{net}.npz"),
                            **{k: v for k, v in res.items() if isinstance(v, np.ndarray)},
                            clock_mhz=clk, source=np.array(ctx.source))
        print(f"[A2/A3 {net}] ({ctx.source}, clock read back {clk:.3f} MHz {ctx.clock_source}; "
              f"µs at f_used {r[-1]['f_used_mhz']} MHz = {r[-1]['f_used_source']})")
        print(f"  {'layer':8} {'model':>8} {'rtl':>8} {'hw':>10} {'min':>8} {'max':>8} "
              f"{'distinct':>8} {'hw_err%':>9} {'hw_us':>10}")
        for x in r:
            print(f"  {x['layer']:8} {x['model_cycles']:>8} {str(x['rtl_cycles']):>8} "
                  f"{str(x['hw_cycles']):>10} {str(x['hw_min']):>8} {str(x['hw_max']):>8} "
                  f"{x['hw_distinct']:>8} {x['hw_err_pct']:>9} {x['hw_us']:>10}")
    ctx.csv("hw_a2_a3_cycles.csv", rows, FIELDS)
    print("A2/A3:", "PASS (model == RTL == accelerator on every image)" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
