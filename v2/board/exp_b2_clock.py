#!/usr/bin/env python3
"""B2 clock sweep: pl_clk0 in {100, 150, 200} (+ 250 / 300 if the bitstream closed there), <= closed clock.

    sudo -E python3 exp_b2_clock.py [--clocks 100 150 200] [--max-mhz 200] [--net lenet5]
                                    [--images 100] [--window-s 60] [--power-repeats 3]
                                    [--rate-hz 10] [--with-meter [--with-idle]]
    python3 exp_b2_clock.py --backend model --window-s 2 --sensor mock   # dry run -> results/dryrun/

The timing-closed frequency comes from --max-mhz or DEPLOY_INFO.json 'bit_clock_mhz' (the
Vivado-reported pl_clk0 of the deployed build). Default sweep (board_common.b2_sweep_clocks):
{100, 150, 200} + {250 if closed >= 250} + {300 if closed >= 300}, capped at the closed clock (the
top point is requested at the closed clock itself). A requested clock above it (+0.5 MHz
tolerance) is skipped and logged, never run; a read-back above it aborts the sweep. Per clock: set pl_clk0 (pynq Clocks.fclk0_mhz), read it back,
soft_reset + reload the net (WGT/QPARAM/DESC with readback), A2 on --images images (cycles must
equal the model at every clock: consistency check), then the power at this clock with the SAME
INA260 SOM-rail logger as B1 (power_log.run_power_protocol) in a REDUCED protocol: accelerator
phases only, idle / accel / idle (--window-s each) x --power-repeats (no CPU phases: the CPU
baseline does not depend on pl_clk0; B1 has it). P_idle, dP, time/image, energy/image as in B1,
computed by power_log. Label "SOM-rail power (INA260)" (not accelerator-only, not board input).
--with-meter adds the old fpga window (and an idle one with --with-idle) with START/STOP banners
for the optional external meter ("board input power (external meter, cross-check)").
The original clock is restored at the end.
Writes hw_b2_clock.csv (one row per clock, incl. the INA260 mean/std over repeats),
hw_b2_power_ina260_{samples,phases,summary}_<net>_<NNN>mhz.csv per clock, and with --with-meter
hw_b2_meter_windows.csv / hw_b2_meter_samples.csv.
"""
from __future__ import annotations

import os
import sys

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import board_common as bc  # noqa: E402
import exp_a2_a3_cycles as a2  # noqa: E402
import exp_b1_power as b1  # noqa: E402
import power_log as pl  # noqa: E402

FIELDS = ["clock_requested_mhz", "clock_readback_mhz", "max_closed_mhz", "images",
          "model_total_cycles", "hw_total_cycles", "hw_min", "hw_max", "cycles_equal_model",
          "latency_us", "wall_us_median", "wall_us_p95", "fpga_window_s", "fpga_inferences",
          "fpga_inf_per_s", "fpga_sensor_power_mean_w", "idle_sensor_power_mean_w",
          "fpga_start_utc", "fpga_stop_utc", "sensor_source", "meter_label",
          "power_label", "ina260_backend", "ina260_repeats", "ina260_p_idle_w", "ina260_p_accel_w",
          "ina260_dp_w", "ina260_dp_std_w", "ina260_time_per_image_s",
          "ina260_energy_per_image_mj", "ina260_energy_per_image_std_mj", "ina260_rate_achieved_hz",
          "ina260_max_gap_s", "ina260_summary_csv", "skipped_reason"]


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
    ap.add_argument("--with-idle", action="store_true",
                    help="with --with-meter: also an idle meter window per clock")
    ap.add_argument("--with-meter", action="store_true",
                    help="also the external-meter cross-check window(s) per clock")
    ap.add_argument("--power-repeats", type=int, default=3,
                    help="INA260 idle/accel/idle repeats per clock")
    ap.add_argument("--rate-hz", type=float, default=pl.DEFAULT_RATE_HZ)
    ap.add_argument("--sensor", default="auto", choices=("auto",) + pl.SENSOR_ORDER + ("mock",))
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
    sensor = b1.PowerSensor() if a.with_meter else None
    print(f"[B2] closed clock {max_mhz} MHz; {pl.LABEL}: {ina.describe()}"
          + (f"; meter cross-check sensor {sensor.source}" if sensor else ""))
    pkg = ctx.package(a.net)
    clk0 = dev.fclk0_mhz()
    rows, prow, psamp = [], [], []
    ok = True
    wid = 0
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
            tot = crows[-1]
            idle_w = ""
            r = {}
            if a.with_meter:
                if a.with_idle:
                    time.sleep(a.gap_s)
                    r, s = b1.power_window(ctx, dev, pkg, "idle", a.window_s, sensor, a.sample_s,
                                           wid, clock_mhz=actual, label=f"B2 {req:g}MHz meter")
                    r["clock_requested_mhz"] = req
                    prow.append(r)
                    psamp += s
                    wid += 1
                    idle_w = r["sensor_power_mean_w"]
                time.sleep(a.gap_s)
                r, s = b1.power_window(ctx, dev, pkg, "fpga", a.window_s, sensor, a.sample_s, wid,
                                       clock_mhz=actual, label=f"B2 {req:g}MHz meter")
                r["clock_requested_mhz"] = req
                prow.append(r)
                psamp += s
                wid += 1
            res = pl.run_power_protocol(
                ctx, a.net, None, accel_fn=pl.gos_workload(dev, pkg), sensor=ina,
                phase_s=a.window_s, repeats=a.power_repeats, rate_hz=a.rate_hz,
                prefix=pl.PREFIX_B2, tag=f"_{a.net}_{pl.clock_tag(req)}", clock_mhz=actual,
                accel_label=f"GosDevice.infer {a.net} (counters not read) @ {actual:.6f} MHz",
                label=f"B2 {req:g}MHz")
            mean = next(x for x in res["summary"] if x["row_kind"] == "mean")
            std = next(x for x in res["summary"] if x["row_kind"] == "std")
            n_ina = sum(int(p["images"]) for p in res["phases"])
            row = ctx.meta(a.net, "all", time.perf_counter() - t0,
                           tot["images"] + n_ina + int(r.get("num_inferences") or 0),
                           clock_mhz=actual)
            f = pl._f
            row.update(clock_requested_mhz=req, clock_readback_mhz=f"{actual:.6f}",
                       max_closed_mhz=max_mhz, images=tot["images"],
                       model_total_cycles=tot["model_cycles"], hw_total_cycles=tot["hw_cycles"],
                       hw_min=tot["hw_min"], hw_max=tot["hw_max"], cycles_equal_model=cyc_ok,
                       latency_us=tot["hw_us"], wall_us_median=tot["wall_us_median"],
                       wall_us_p95=tot["wall_us_p95"], fpga_window_s=r.get("window_s", ""),
                       fpga_inferences=r.get("num_inferences", ""),
                       fpga_inf_per_s=r.get("inf_per_s", ""),
                       fpga_sensor_power_mean_w=r.get("sensor_power_mean_w", ""),
                       idle_sensor_power_mean_w=idle_w, fpga_start_utc=r.get("start_utc", ""),
                       fpga_stop_utc=r.get("stop_utc", ""),
                       sensor_source=sensor.source if sensor else "",
                       meter_label=b1.METER_LABEL if a.with_meter else "",
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
                       ina260_summary_csv=Path(res["files"]["summary"]).name)
            rows.append(row)
    finally:
        back = dev.set_fclk0(clk0)
        print(f"[B2] restored pl_clk0 to {back:.6f} MHz (was {clk0:.6f})")
    ctx.csv("hw_b2_clock.csv", rows, FIELDS)
    if a.with_meter:
        ctx.csv("hw_b2_meter_windows.csv", prow, b1.FIELDS + ["clock_requested_mhz"])
        ctx.csv("hw_b2_meter_samples.csv", psamp, b1.SAMPLE_FIELDS)
    print("B2 cycles:", "equal to model at every clock" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
