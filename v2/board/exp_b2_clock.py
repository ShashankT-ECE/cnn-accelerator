#!/usr/bin/env python3
"""B2 clock sweep: pl_clk0 in {100, 150, 200, 250, 300} MHz, limited to the bitstream's closed clock.

    sudo -E python3 exp_b2_clock.py [--clocks 100 150 200] [--max-mhz 200] [--net lenet5]
                                    [--images 100] [--window-s 60] [--with-idle]
    python3 exp_b2_clock.py --backend model --window-s 2    # dry run -> results/dryrun/

The timing-closed frequency comes from --max-mhz or DEPLOY_INFO.json 'bit_clock_mhz' (the
Vivado-reported pl_clk0 of the deployed build); a requested clock above it (+0.5 MHz tolerance)
is skipped and logged, never run. Per clock: set pl_clk0 (pynq Clocks.fclk0_mhz), read it back,
soft_reset + reload the net (WGT/QPARAM/DESC with readback), A2 on --images images (cycles must
equal the model at every clock: consistency check), then a --window-s fpga power window (and an
idle window with --with-idle) with START/STOP banners for the meter. The original clock is
restored at the end. Power labels as in exp_b1_power.py (board-level input via the meter; on-board
sensor = SOM power; never accelerator power).
Writes hw_b2_clock.csv (one row per clock), hw_b2_power.csv and hw_b2_power_samples.csv.
"""
from __future__ import annotations

import os
import sys

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import time  # noqa: E402

import board_common as bc  # noqa: E402
import exp_a2_a3_cycles as a2  # noqa: E402
import exp_b1_power as b1  # noqa: E402

CLOCKS = (100, 150, 200, 250, 300)
FIELDS = ["clock_requested_mhz", "clock_readback_mhz", "max_closed_mhz", "images",
          "model_total_cycles", "hw_total_cycles", "hw_min", "hw_max", "cycles_equal_model",
          "latency_us", "wall_us_median", "wall_us_p95", "fpga_window_s", "fpga_inferences",
          "fpga_inf_per_s", "fpga_sensor_power_mean_w", "idle_sensor_power_mean_w",
          "fpga_start_utc", "fpga_stop_utc", "sensor_source", "skipped_reason"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap, nets=False)
    ap.add_argument("--clocks", nargs="+", type=float, default=list(CLOCKS))
    ap.add_argument("--max-mhz", type=float, default=None)
    ap.add_argument("--net", default="lenet5", choices=bc.NETS)
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--gap-s", type=float, default=10.0)
    ap.add_argument("--sample-s", type=float, default=0.5)
    ap.add_argument("--with-idle", action="store_true", help="also an idle window per clock")
    a = ap.parse_args(argv)
    a.nets = [a.net]
    dev, be = bc.open_device(a)
    ctx = bc.RunContext(a, dev, be, "B2")
    ctx.check_clean()
    ctx.banner()
    max_mhz = a.max_mhz if a.max_mhz is not None else ctx.info.get("bit_clock_mhz")
    if max_mhz is None:
        raise SystemExit("closed clock unknown: pass --max-mhz (or deploy with DEPLOY_INFO.json)")
    max_mhz = float(max_mhz)
    sensor = b1.PowerSensor()
    print(f"[B2] closed clock {max_mhz} MHz; sensor {sensor.source}")
    pkg = ctx.package(a.net)
    clk0 = dev.fclk0_mhz()
    rows, prow, psamp = [], [], []
    ok = True
    wid = 0
    try:
        for req in a.clocks:
            row = None
            if req > max_mhz + 0.5:
                print(f"[B2] SKIP {req:g} MHz: above the closed clock {max_mhz} MHz")
                row = ctx.meta(a.net, "all", "", 0, clock_mhz="")
                row.update(clock_requested_mhz=req, max_closed_mhz=max_mhz,
                           skipped_reason=f"above closed clock {max_mhz} MHz")
                rows.append(row)
                continue
            actual = dev.set_fclk0(req)
            print(f"[B2] pl_clk0 requested {req:g} MHz -> read back {actual:.6f} MHz")
            dev.soft_reset()
            dev.load_net(pkg)
            t0 = time.perf_counter()
            crows, _, cyc_ok = a2.cycle_rows(ctx, dev, pkg, range(min(a.images, pkg.n)), actual,
                                             tag=f"B2 {req:g}")
            ok &= cyc_ok
            tot = crows[-1]
            idle_w = ""
            if a.with_idle:
                time.sleep(a.gap_s)
                r, s = b1.power_window(ctx, dev, pkg, "idle", a.window_s, sensor, a.sample_s, wid,
                                       clock_mhz=actual, label=f"B2 {req:g}MHz")
                r["clock_requested_mhz"] = req
                prow.append(r)
                psamp += s
                wid += 1
                idle_w = r["sensor_power_mean_w"]
            time.sleep(a.gap_s)
            r, s = b1.power_window(ctx, dev, pkg, "fpga", a.window_s, sensor, a.sample_s, wid,
                                   clock_mhz=actual, label=f"B2 {req:g}MHz")
            r["clock_requested_mhz"] = req
            prow.append(r)
            psamp += s
            wid += 1
            row = ctx.meta(a.net, "all", time.perf_counter() - t0, tot["images"] + int(r["num_inferences"]),
                           clock_mhz=actual)
            row.update(clock_requested_mhz=req, clock_readback_mhz=f"{actual:.6f}",
                       max_closed_mhz=max_mhz, images=tot["images"],
                       model_total_cycles=tot["model_cycles"], hw_total_cycles=tot["hw_cycles"],
                       hw_min=tot["hw_min"], hw_max=tot["hw_max"], cycles_equal_model=cyc_ok,
                       latency_us=tot["hw_us"], wall_us_median=tot["wall_us_median"],
                       wall_us_p95=tot["wall_us_p95"], fpga_window_s=r["window_s"],
                       fpga_inferences=r["num_inferences"], fpga_inf_per_s=r["inf_per_s"],
                       fpga_sensor_power_mean_w=r["sensor_power_mean_w"],
                       idle_sensor_power_mean_w=idle_w, fpga_start_utc=r["start_utc"],
                       fpga_stop_utc=r["stop_utc"], sensor_source=sensor.source)
            rows.append(row)
    finally:
        back = dev.set_fclk0(clk0)
        print(f"[B2] restored pl_clk0 to {back:.6f} MHz (was {clk0:.6f})")
    ctx.csv("hw_b2_clock.csv", rows, FIELDS)
    ctx.csv("hw_b2_power.csv", prow, b1.FIELDS + ["clock_requested_mhz"])
    ctx.csv("hw_b2_power_samples.csv", psamp, b1.SAMPLE_FIELDS)
    print("B2 cycles:", "equal to model at every clock" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
