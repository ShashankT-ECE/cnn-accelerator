"""Throughput and efficiency of the accelerator vs the DPU vs the CPU, from committed board results.

Every input is a committed CSV in v2/results (nothing typed):
  cycle_model.csv         macs per image (layer == total)                      -> ops = 2 * macs
  impl_gos.csv            DSP48E2 of the 250 MHz build (the performance clock, DECISIONS D21)
  hw_a2_a3_cycles.csv     PL compute time = TOTAL_CYC / f (hw_us, total row)
  hw_b3_breakdown*.csv    end-to-end p50, safe and fast host path
  hw_dpu_latency.csv      DPU dpu_runner and end_to_end p50
  dpu_overlay_params.csv  DPU MAC lanes and clock (hwh; resource counts are NOT available there)
  hw_cpu_baseline.csv     best CPU baseline (lowest e2e median) compute / e2e

Per net and system (rows):
  ours_pl          PL counter time           ours_e2e_fast  fast host path end-to-end (primary)
  ours_e2e_safe    safe host path end-to-end (footnote)
  dpu_runner       VART runner time          dpu_e2e        pre + runner + post
  cpu_compute / cpu_e2e   best CPU baseline
Exact formulas (also written to the `formula` column of every row):
  ops_per_image        = 2 * macs                      (macs = cycle_model.csv, layer total; the same MACs
                                                        for every system: same architecture, the DPU runs
                                                        the vai_q quantization of the same FP32 net)
  gops                 = ops_per_image / (time_us * 1e3)   (ops per ns = Gop/s)
  images_per_s         = 1e6 / time_us
  mac_lanes            = ours 64 (8 x 8 PEs, one DSP48E2 MAC each); DPU ARCH_PP * ARCH_ICP * ARCH_OCP
                         = 8 * 16 * 16 = 2048 (hwh parameters, dpu_overlay_params.csv); CPU empty
  gops_per_mac_lane    = gops / mac_lanes                 images_per_s_per_mac_lane = images_per_s / mac_lanes
  peak_gops            = 2 * mac_lanes * clock_mhz / 1e3  (1 MAC = 2 ops, every lane busy every cycle)
  pct_of_peak          = 100 * gops / peak_gops
Lane clock (`clock_mhz`, source in `clock_source`): ours = the pl_clk0 PLL read-back of the A2 run
(hw_a2_a3_cycles.csv f_used_mhz, the clock of record, D20); DPU = `aclk` of DPUCZDX8G_1 = 300 MHz, the
PORT CLKFREQUENCY in the overlay file pynqdpu.dpu.kv260_som.2.5.0.hwh (via dpu_overlay_params.csv
hwh_clock row aclk; `ap_clk_2` = 600 MHz is the DSP double-rate clock and is NOT the lane clock used here).
Columns: dsps (ours only, from impl_gos.csv) with gops_per_dsp and images_per_s_per_dsp (ours only).
There is NO DPU per-DSP figure: the overlay's .hwh/.xclbin carry no resource counts and a MAC lane is
not a DSP (DPU INT8 MACs are packed in DSP48E2 slices that run at 2x aclk).

Run:  .venv/bin/python v2/analysis/board_efficiency.py [--out-dir DIR] [--results-dir DIR]
"""
from __future__ import annotations

import csv
from pathlib import Path

import _setup
from _setup import fmt

from common import base_meta, write_results_csv

CSV_NAME = "board_efficiency.csv"
NETS = ("lenet5", "cifar10")
OUR_LANES = 64            # 8x8 PEs, one DSP48E2 MAC each (ARCH_SPEC: PE = one DSP48E2)
PERF_CLOCK = 250
FORMULA = ("ops_per_image = 2*macs; gops = ops_per_image/(time_us*1e3); gops_per_mac_lane = gops/mac_lanes; "
           "mac_lanes: ours 8x8 = 64, DPU ARCH_PP*ARCH_ICP*ARCH_OCP = 2048; peak_gops = 2*mac_lanes*clock_mhz/1e3; "
           "pct_of_peak = 100*gops/peak_gops")
FIELDS = ["system", "time_basis", "ops_per_image", "time_us", "images_per_s", "gops", "mac_lanes",
          "clock_mhz", "clock_source", "peak_gops", "pct_of_peak",
          "gops_per_mac_lane", "images_per_s_per_mac_lane", "dsps", "gops_per_dsp",
          "images_per_s_per_dsp", "formula", "inputs", "note"]


def rows_of(res: Path, name: str) -> list[dict]:
    p = res / name
    with p.open(newline="") as f:
        return list(csv.DictReader(f))


def latest(rows, key):
    out = {}
    for r in rows:
        k = key(r)
        if k not in out or r.get("timestamp", "") >= out[k].get("timestamp", ""):
            out[k] = r
    return out


def compute(res: Path) -> list[dict]:
    cm = {r["net"]: r for r in rows_of(res, "cycle_model.csv") if r["layer"] == "total"}
    impl = [r for r in rows_of(res, "impl_gos.csv")
            if abs(float(r["pl_clk0_mhz_requested"]) - PERF_CLOCK) < 0.5 and r.get("top") == "gos_top_wrapper"]
    dsps = int(latest(impl, lambda r: r["pl_clk0_mhz_requested"])[str(PERF_CLOCK)]["dsps"]) if impl else None
    a2 = {r["net"]: r for r in rows_of(res, "hw_a2_a3_cycles.csv") if r["layer"] == "total"}
    b3 = {(hp, r["net"]): r for hp, f in (("safe", "hw_b3_breakdown.csv"), ("fast", "hw_b3_breakdown_fast.csv"))
          for r in rows_of(res, f) if r["phase"] == "end_to_end"}
    dl = latest(rows_of(res, "hw_dpu_latency.csv"), lambda r: (r["net"], r["metric"]))
    prow = {r["name"]: r for r in rows_of(res, "dpu_overlay_params.csv")}
    params = {k: r["value"] for k, r in prow.items()}
    dpu_lanes = int(params["mac_lanes"])
    dpu_mhz = float(params["aclk"])
    dpu_clk_src = f"{Path(prow['aclk']['hwh_file']).name} PORT CLKFREQUENCY aclk (dpu_overlay_params.csv)"
    cpu = [r for r in rows_of(res, "hw_cpu_baseline.csv") if r.get("status") == "ok"]
    out = []
    for net in NETS:
        ops = 2 * int(cm[net]["macs"])
        systems = [
            ("ours_pl", "PL counter TOTAL_CYC / pl_clk0 read-back", float(a2[net]["hw_us"]), OUR_LANES,
             dsps, "hw_a2_a3_cycles.csv hw_us"),
            ("ours_e2e_fast", "fast host path end-to-end p50 (primary)", float(b3[("fast", net)]["p50_us"]),
             OUR_LANES, dsps, "hw_b3_breakdown_fast.csv end_to_end p50_us"),
            ("ours_e2e_safe", "safe host path end-to-end p50 (footnote)", float(b3[("safe", net)]["p50_us"]),
             OUR_LANES, dsps, "hw_b3_breakdown.csv end_to_end p50_us"),
            ("dpu_runner", "VART runner execute_async + wait p50", float(dl[(net, "dpu_runner")]["p50"]),
             dpu_lanes, None, "hw_dpu_latency.csv dpu_runner p50"),
            ("dpu_e2e", "DPU pre + runner + post p50", float(dl[(net, "end_to_end")]["p50"]),
             dpu_lanes, None, "hw_dpu_latency.csv end_to_end p50"),
        ]
        e2e = latest([r for r in cpu if r["net"] == net and r["mode"] == "e2e" and r.get("median_us")],
                     lambda r: (r["kind"], r["threads"]))
        if e2e:
            (kind, th), best = min(e2e.items(), key=lambda kv: float(kv[1]["median_us"]))
            comp = latest([r for r in cpu if r["net"] == net and r["mode"] == "compute"],
                          lambda r: (r["kind"], r["threads"]))[(kind, th)]
            tag = f"{kind} x{th} (lowest e2e median)"
            systems += [("cpu_compute", f"best CPU baseline compute median, {tag}", float(comp["median_us"]),
                         None, None, "hw_cpu_baseline.csv compute median_us"),
                        ("cpu_e2e", f"best CPU baseline e2e median, {tag}", float(best["median_us"]),
                         None, None, "hw_cpu_baseline.csv e2e median_us")]
        our_mhz = float(a2[net]["f_used_mhz"])
        for sysname, basis, t_us, lanes, nd, inp in systems:
            gops = ops / (t_us * 1e3)            # ops / ns = Gop/s
            ips = 1e6 / t_us
            mhz, clk_src = ((our_mhz, "hw_a2_a3_cycles.csv f_used_mhz (pl_clk0 PLL read-back, clock of record)")
                            if sysname.startswith("ours") else (dpu_mhz, dpu_clk_src)
                            if sysname.startswith("dpu") else (None, ""))
            r = {**base_meta(net, sysname, source="derived_from_hw"), "system": sysname, "time_basis": basis,
                 "ops_per_image": ops, "time_us": fmt(t_us, 3), "images_per_s": fmt(ips, 1), "gops": fmt(gops, 4),
                 "mac_lanes": lanes or "", "dsps": nd or "", "formula": FORMULA,
                 "inputs": inp + "; cycle_model.csv macs"}
            if lanes:
                peak = 2 * lanes * mhz / 1e3
                r.update(gops_per_mac_lane=fmt(gops / lanes, 5), images_per_s_per_mac_lane=fmt(ips / lanes, 3),
                         clock_mhz=fmt(mhz, 4), clock_source=clk_src, peak_gops=fmt(peak, 3),
                         pct_of_peak=fmt(100 * gops / peak, 3))
            if nd:
                r.update(gops_per_dsp=fmt(gops / nd, 5), images_per_s_per_dsp=fmt(ips / nd, 3))
            r["note"] = ("no per-DSP figure for the DPU (no resource counts in the overlay metadata); "
                         "lane clock = aclk, ap_clk_2 (600 MHz) is the DSP double-rate clock"
                         if sysname.startswith("dpu") else
                         "no MAC-lane / DSP figure for the CPU" if sysname.startswith("cpu") else "")
            out.append(r)
    return out


def main(argv=None) -> int:
    ap = _setup.out_dir_parser(__doc__.split("\n")[0])
    ap.add_argument("--results-dir", type=Path, default=_setup.RESULTS_DIR)
    a = ap.parse_args(argv)
    rows = compute(a.results_dir)
    p = write_results_csv(a.out_dir / CSV_NAME, rows, FIELDS)
    print(f"wrote {p} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['net']:8} {r['system']:14} {r['time_us']:>10} us {r['images_per_s']:>9} img/s "
              f"{r['gops']:>9} GOPS  /lane {r.get('gops_per_mac_lane') or '-':>8}  "
              f"@{r.get('clock_mhz') or '-':>9} MHz  {r.get('pct_of_peak') or '-':>7} % of peak  "
              f"/DSP {r.get('gops_per_dsp') or '-':>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
