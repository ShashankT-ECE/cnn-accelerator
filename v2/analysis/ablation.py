"""DECISIONS D7 schedule ablation (model): V1-style generalized OS schedule vs the V2 schedule.

V1-style terms come from the legacy generalized OS model ``python/v2_architecture_model.py``
(imported read-only through ``gos_cycle_model.ablation``; its ``main()`` is never called
because it writes data/benchmark/). Per layer they are decomposed into

    core         = T*K               (the same reduction stream as V2; asserted equal)
    lead_in      = passes * (P-1)    per-pass shift-chain fill
    pass_drain   = passes * 1        per-pass drain (FC: one tail cycle per group)
    clear        = 1 per group
    result_drain = 2 * active rows per group

and compared with the V2 cycle model LAYER_CYC = T*K + C_PIPE (C_PIPE RTL-derived) of
``gos_cycle_model``. The legacy model covers both LeNet-5 and CIFAR-10 (its CIFAR layer
table is checked shape-for-shape against net_config). The legacy per-layer timing status
(RTL-verified vs analytical) is carried in the CSV. Everything is source=model: neither
column is an RTL or hardware measurement.

Run:  .venv/bin/python v2/analysis/ablation.py [--out-dir DIR]
"""
from __future__ import annotations

import sys
from fractions import Fraction

import _setup
from _setup import fmt

from common import base_meta, write_results_csv
from net_config import NET_CONFIGS, NETS
import gos_cycle_model as cm
import v2_architecture_model as legacy   # read-only; main() never called

CSV_NAME = "schedule_ablation.csv"
TERMS = ("core", "lead_in", "pass_drain", "clear", "result_drain")
FIELDS = (["IC", "OC", "OH", "OW", "K", "T"]
          + [f"v1_{t}" for t in TERMS] + ["v1_overhead", "v1_total", "v1_timing_status",
                                          "v1_source"]
          + ["v2_compute", "v2_c_pipe", "v2_c_start_done", "v2_overhead", "v2_layer_cyc",
             "core_equal", "removed_cycles", "speedup_v1_over_v2", "v1_overhead_frac",
             "v2_overhead_frac"])


def net_rows(net: str) -> list[dict]:
    abl = cm.ablation(net)          # asserts: layer map, core == T*K, legacy total == D7 spec
    v2 = cm.net_cycles(NET_CONFIGS[net]["layers"])
    legacy_layers = {LL["name"]: LL for LL in cm.LEGACY_LAYERS[net]}
    rows = []
    for L in NET_CONFIGS[net]["layers"]:
        n = L["name"]
        t, r = abl["layers"][n], v2["layers"][n]
        assert t["core"] == r["compute_cycles"]
        v1_over = t["total"] - t["core"]
        v2_over = r["cycles"] - r["compute_cycles"]
        rows.append({
            **base_meta(net, n, source="model"),
            **{k: L[k] for k in ("IC", "OC", "OH", "OW", "K")}, "T": r["T"],
            **{f"v1_{k}": t[k] for k in TERMS}, "v1_overhead": v1_over, "v1_total": t["total"],
            "v1_timing_status": legacy.timing_status(legacy_layers[n]),
            "v1_source": cm.LEGACY_SOURCE,
            "v2_compute": r["compute_cycles"], "v2_c_pipe": r["c_pipe"], "v2_c_start_done": 0,
            "v2_overhead": v2_over, "v2_layer_cyc": r["cycles"],
            "core_equal": t["core"] == r["compute_cycles"],
            "removed_cycles": t["total"] - r["cycles"],
            "speedup_v1_over_v2": fmt(Fraction(t["total"], r["cycles"]), 4),
            "v1_overhead_frac": fmt(Fraction(v1_over, t["total"])),
            "v2_overhead_frac": fmt(Fraction(v2_over, r["cycles"])),
        })
    t, tot = abl["total"], v2["total"]
    start_done = (cm.C_START or 0) + (cm.C_DONE or 0)
    assert tot["cycles"] == sum(r["v2_layer_cyc"] for r in rows) + start_done
    assert t["total"] == cm.LEGACY_TOTALS_SPEC[net] == sum(t[k] for k in TERMS)
    v1_over, v2_over = t["total"] - t["core"], tot["cycles"] - tot["compute_cycles"]
    rows.append({
        **base_meta(net, "total", source="model"), "T": tot["T"],
        **{f"v1_{k}": t[k] for k in TERMS}, "v1_overhead": v1_over, "v1_total": t["total"],
        "v1_timing_status": "mixed (see layer rows)", "v1_source": cm.LEGACY_SOURCE,
        "v2_compute": tot["compute_cycles"], "v2_c_pipe": cm.C_PIPE,
        "v2_c_start_done": start_done, "v2_overhead": v2_over, "v2_layer_cyc": tot["cycles"],
        "core_equal": t["core"] == tot["compute_cycles"],
        "removed_cycles": t["total"] - tot["cycles"],
        "speedup_v1_over_v2": fmt(Fraction(t["total"], tot["cycles"]), 4),
        "v1_overhead_frac": fmt(Fraction(v1_over, t["total"])),
        "v2_overhead_frac": fmt(Fraction(v2_over, tot["cycles"])),
    })
    return rows


def build_rows() -> list[dict]:
    return [r for net in NETS for r in net_rows(net)]


def main(argv=None) -> int:
    a = _setup.out_dir_parser(__doc__.splitlines()[0]).parse_args(argv)
    rows = build_rows()
    path = write_results_csv(a.out_dir / CSV_NAME, rows, ["net", "layer"] + FIELDS)
    for r in rows:
        print(f"{r['net']:8} {r['layer']:6} v1={r['v1_total']:>7} (core {r['v1_core']} + lead "
              f"{r['v1_lead_in']} + pdrain {r['v1_pass_drain']} + clear {r['v1_clear']} + rdrain "
              f"{r['v1_result_drain']})  v2={r['v2_layer_cyc']:>7}  x{r['speedup_v1_over_v2']}")
    print(f"Wrote {path} (model)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
