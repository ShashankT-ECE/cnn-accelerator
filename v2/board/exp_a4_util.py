#!/usr/bin/env python3
"""A4 utilization: MAC_ACTIVE / cycles per layer and per net vs the theoretical (model) value.

    sudo -E python3 exp_a4_util.py [--nets ...] [--limit N]      (default 1000 images)
    python3 exp_a4_util.py --backend model                       # dry run -> results/dryrun/

What is measured and what is derived (be explicit in the paper):
  * The hardware has ONE MAC_ACTIVE counter for the whole job (FORMATS.md CSR map 0x018) and
    per-layer LAYER_CYC counters. There is no per-layer MAC_ACTIVE counter.
  * Net row: mac_active = measured MAC_ACTIVE, cycles = measured TOTAL_CYC ->
    active_frac = MAC_ACTIVE / TOTAL_CYC (measured / measured).
  * Layer rows: mac_active = model T*K (NOT a hardware counter; mac_active_source says so),
    cycles = measured LAYER_CYC[l] -> active_frac = T*K / LAYER_CYC (derived). The net row also
    checks measured MAC_ACTIVE == sum over layers of T*K.
  * theoretical_active_frac = T*K / (T*K + C_PIPE) per layer, sum(T*K) / model TOTAL per net.
  * pe_util = MACs / (64 * cycles): useful-MAC utilization of the 8x8 array including OC-tail
    and x-tail lane masking (model MACs over measured cycles); pe_util_theoretical uses model
    cycles. util_array_active = MACs / (64 * T*K) is the lane occupancy while active (model).
Cycle counters are identical on every image (A2/A3), so the median over the images is used and
hw_distinct reports how many distinct values occurred.
Writes hw_a4_util.csv.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import board_common as bc

FIELDS = ["images", "mac_active", "mac_active_source", "cycles", "cycles_source", "hw_distinct",
          "active_frac", "theoretical_active_frac", "active_frac_err_pct", "macs",
          "pe_util", "pe_util_theoretical", "util_array_active", "mac_active_equals_sum_TK"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    bc.add_common_args(ap)
    a = ap.parse_args(argv)
    if a.limit is None:
        a.limit = 1000
    dev, be = bc.open_device(a)
    ctx = bc.RunContext(a, dev, be, "A4")
    ctx.check_clean()
    ctx.banner()
    rows = []
    ok = True
    for net in a.nets:
        pkg = ctx.package(net)
        dev.load_net(pkg)
        clk = dev.fclk0_mhz()
        idx = bc.image_range(pkg, a.limit)
        res = bc.infer_loop(dev, pkg, idx, read_counters=True, tag=f"A4 {net}")
        g = res["ok"]
        n = int(g.sum())
        mc = pkg.model_cycles
        pe = mc["PE_COUNT"]
        sum_tk = 0
        print(f"[A4 {net}] ({ctx.source})  {'layer':6} {'mac_active':>10} {'cycles':>8} "
              f"{'active':>8} {'theory':>8} {'pe_util':>8} {'pe_theo':>8}")
        for l, L in enumerate(mc["layers"]):
            cyc = res["layer_cyc"][g, l]
            u = np.unique(cyc)
            c = float(np.median(cyc))
            tk = L["mac_active"]
            sum_tk += tk
            row = ctx.meta(net, L["name"], res["duration_s"], len(idx), clock_mhz=clk)
            af, th = tk / c, tk / L["cycles"]
            row.update(images=n, mac_active=tk,
                       mac_active_source="model T*K (no per-layer HW counter)",
                       cycles=int(c) if c.is_integer() else c,
                       cycles_source=f"{'hw' if ctx.source == bc.SOURCE_HW else ctx.source} LAYER_CYC",
                       hw_distinct=int(u.size), active_frac=f"{af:.6f}",
                       theoretical_active_frac=f"{th:.6f}",
                       active_frac_err_pct=f"{100.0 * (af - th) / th:.6f}", macs=L["macs"],
                       pe_util=f"{L['macs'] / (pe * c):.6f}",
                       pe_util_theoretical=f"{L['macs'] / (pe * L['cycles']):.6f}",
                       util_array_active=f"{L['util_theoretical']:.6f}")
            rows.append(row)
            print(f"  {'':15} {L['name']:6} {tk:>10} {row['cycles']:>8} {af:>8.4f} {th:>8.4f} "
                  f"{L['macs'] / (pe * c):>8.4f} {L['macs'] / (pe * L['cycles']):>8.4f}")
        mac = res["mac_active"][g]
        tot = res["total_cyc"][g]
        m, t = float(np.median(mac)), float(np.median(tot))
        af, th = m / t, mc["total"]["mac_active"] / mc["total"]["cycles"]
        eq = bool(np.all(mac == sum_tk))
        ok &= eq
        row = ctx.meta(net, "total", res["duration_s"], len(idx), clock_mhz=clk)
        row.update(images=n, mac_active=int(m) if m.is_integer() else m,
                   mac_active_source=f"{'hw' if ctx.source == bc.SOURCE_HW else ctx.source} MAC_ACTIVE counter",
                   cycles=int(t) if t.is_integer() else t,
                   cycles_source=f"{'hw' if ctx.source == bc.SOURCE_HW else ctx.source} TOTAL_CYC",
                   hw_distinct=int(max(np.unique(mac).size, np.unique(tot).size)),
                   active_frac=f"{af:.6f}", theoretical_active_frac=f"{th:.6f}",
                   active_frac_err_pct=f"{100.0 * (af - th) / th:.6f}",
                   macs=mc["total"]["macs"], pe_util=f"{mc['total']['macs'] / (pe * t):.6f}",
                   pe_util_theoretical=f"{mc['total']['macs'] / (pe * mc['total']['cycles']):.6f}",
                   util_array_active=f"{mc['total']['util_theoretical']:.6f}",
                   mac_active_equals_sum_TK=eq)
        rows.append(row)
        print(f"  {'':15} {'total':6} {row['mac_active']:>10} {row['cycles']:>8} {af:>8.4f} "
              f"{th:>8.4f}   MAC_ACTIVE == sum(T*K) on every image: {eq}")
    ctx.csv("hw_a4_util.csv", rows, FIELDS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
