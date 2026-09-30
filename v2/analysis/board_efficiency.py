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
Columns: ops_per_image = 2 * macs (the same MACs for every system: the nets are the same
architecture; the DPU runs the vai_q quantization of the same FP32 net), time_us, images_per_s,
gops = ops_per_image / time, mac_lanes (ours 64 = 8x8 PEs, DPU ARCH_PP*ARCH_ICP*ARCH_OCP,
CPU empty), gops_per_mac_lane, images_per_s_per_mac_lane, dsps (ours from impl_gos.csv, DPU and
CPU EMPTY: the overlay's .hwh/.xclbin carry no resource counts), gops_per_dsp and
images_per_s_per_dsp (ours only). A MAC lane is not a DSP: the DPU packs INT8 MACs in DSP48E2 and
runs its DSPs at 2x aclk; per-DSP numbers for the DPU need its utilization report.

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
FIELDS = ["system", "time_basis", "ops_per_image", "time_us", "images_per_s", "gops", "mac_lanes",
          "gops_per_mac_lane", "images_per_s_per_mac_lane", "dsps", "gops_per_dsp",
          "images_per_s_per_dsp", "inputs", "note"]


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
    params = {r["name"]: r["value"] for r in rows_of(res, "dpu_overlay_params.csv")}
    dpu_lanes = int(params["mac_lanes"])
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
        for sysname, basis, t_us, lanes, nd, inp in systems:
            gops = ops / (t_us * 1e3)            # ops / ns = Gop/s
            ips = 1e6 / t_us
            r = {**base_meta(net, sysname, source="derived_from_hw"), "system": sysname, "time_basis": basis,
                 "ops_per_image": ops, "time_us": fmt(t_us, 3), "images_per_s": fmt(ips, 1), "gops": fmt(gops, 4),
                 "mac_lanes": lanes or "", "dsps": nd or "", "inputs": inp + "; cycle_model.csv macs"}
            if lanes:
                r.update(gops_per_mac_lane=fmt(gops / lanes, 5), images_per_s_per_mac_lane=fmt(ips / lanes, 3))
            if nd:
                r.update(gops_per_dsp=fmt(gops / nd, 5), images_per_s_per_dsp=fmt(ips / nd, 3))
            r["note"] = ("per-DSP value unavailable: no DPU resource counts in the overlay metadata"
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
              f"{r['gops']:>9} GOPS  /lane {r.get('gops_per_mac_lane') or '-':>8}  /DSP {r.get('gops_per_dsp') or '-':>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
