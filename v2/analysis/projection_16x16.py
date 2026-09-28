"""NxN array projection of the V2 cycle model (model only; N=16 is NOT implemented).

This is NOT EXPERIMENTS C1: C1 ("optional 16x16") is defined as post-implementation only.
Nothing here is a resource, fmax, power or hardware claim; every row carries
label = "projected (model, not implemented)".

The frozen 8x8 cycle model (gos_cycle_model) is parameterized by the array size N with
the ARCH_SPEC loop nest unchanged (rows = N consecutive output x positions of one output
row, columns = N output channels, one k per cycle, pool dy pairs inside the oy loop):

    T(N)        = ceil(OC/N) * OH * ceil(OW/N)
    LAYER_CYC   = T(N) * K + C_PIPE(N)
    C_PIPE(N)   = C_PIPE - (L_DRAIN - 1) + (N - 1)       (drain = N columns per tile)
    TOTAL_CYC   = sum(LAYER_CYC) + C_START + C_DONE      (unchanged)

N=8 is asserted to reproduce gos_cycle_model exactly (per layer and total).

Assumptions (also in the CSV 'assumptions' column):
  P1 Only the drain length depends on N explicitly in the model (L_DRAIN = ARRAY columns);
     every other latency (L_LOAD, L_ISSUE, L_MEM_ACC, L_ROT, L_ARRAY, L_QPARAM, L_RQ,
     L_POOL, L_RETIRE) and C_START/C_DONE are kept at their 8x8 RTL-derived values.
     Each extra pipeline stage a 16x16 build might need (e.g. a deeper broadcast tree for
     fanout 16, a wider rotator) adds exactly 1 cycle per layer (column
     extra_cycles_per_added_stage).
  P2 The drain of N columns overlaps the next tile's K issue cycles with no stall iff
     K >= N (the 8x8 argument, ARCH_SPEC Datapath "Drain"); checked per layer
     (drain_hidden). If False the projection would be optimistic for that layer.
  P3 Memories scale to N banks / N-byte weight words so that one k per cycle is sustained
     (conflict-free banked read with an N-way rotator); capacity, BRAM count and the
     config checker's buffer limits are not modelled.
  P4 Requant has N lanes, one output channel per drain cycle (same per-tile schedule).
  P5 Pooling: N is even and ox0 = N*ox_tile is even, so horizontal pool pairs never
     straddle a tile (same as 8x8); T is unchanged by pooling.

Run:  .venv/bin/python v2/analysis/projection_16x16.py [--out-dir DIR] [--sizes 8 16]
"""
from __future__ import annotations

import sys
from fractions import Fraction

import _setup
from _setup import fmt

from common import base_meta, write_results_csv
from net_config import NET_CONFIGS, NETS
import gos_cycle_model as cm

CSV_NAME = "projection_16x16.csv"
LABEL = "projected (model, not implemented)"
NOT_C1 = "not EXPERIMENTS C1 (C1 = 16x16 post-implementation only)"
ASSUMPTIONS = ("P1 only drain length scales with N (C_PIPE(N)=C_PIPE-(L_DRAIN-1)+(N-1)); "
               "other latencies, C_START, C_DONE at 8x8 RTL values; "
               "P2 drain hidden iff K>=N (see drain_hidden); "
               "P3 memories sustain one k/cycle at N banks, capacity/BRAM not modelled; "
               "P4 N requant lanes, one channel per drain cycle; "
               "P5 pool pairs never straddle a tile (N even); no resources/fmax claimed")
DEFAULT_SIZES = (8, 16)
FIELDS = ["label", "array_n", "pe_count", "IC", "OC", "OH", "OW", "K", "oc_tiles",
          "ow_tiles", "T", "compute_cycles", "c_pipe", "c_start_done", "layer_cyc",
          "speedup_vs_n8", "macs_useful", "util_spatial", "util_temporal", "util_total",
          "drain_hidden", "extra_cycles_per_added_stage", "reproduces_8x8_model", "c1_note",
          "assumptions"]


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def c_pipe(n: int) -> int:
    """C_PIPE with the drain length set to n columns (P1)."""
    assert cm.L_DRAIN == cm.ARRAY, "the 8x8 model's drain length is the array width"
    return cm.C_PIPE - (cm.L_DRAIN - 1) + (n - 1)


def layer_proj(L: dict, n: int) -> dict:
    if n < 2 or n % 2:
        raise ValueError("array size must be even (pool pairs, P5)")
    oc_t, ow_t = _ceil(L["OC"], n), _ceil(L["OW"], n)
    T = oc_t * L["OH"] * ow_t
    K = L["K"]
    compute = T * K
    cp = c_pipe(n)
    macs = L["OC"] * L["OH"] * L["OW"] * K
    return {"array_n": n, "pe_count": n * n, "K": K, "oc_tiles": oc_t, "ow_tiles": ow_t,
            "T": T, "compute_cycles": compute, "c_pipe": cp, "layer_cyc": compute + cp,
            "macs_useful": macs, "drain_hidden": K >= n,
            "util_spatial": Fraction(macs, n * n * compute),
            "util_temporal": Fraction(compute, compute + cp),
            "util_total": Fraction(macs, n * n * (compute + cp))}


def net_proj(net: str, n: int) -> dict:
    layers = NET_CONFIGS[net]["layers"]
    per = {L["name"]: layer_proj(L, n) for L in layers}
    start_done = (cm.C_START or 0) + (cm.C_DONE or 0)
    compute = sum(r["compute_cycles"] for r in per.values())
    cycles = sum(r["layer_cyc"] for r in per.values()) + start_done
    macs = sum(r["macs_useful"] for r in per.values())
    total = {"array_n": n, "pe_count": n * n, "T": sum(r["T"] for r in per.values()),
             "compute_cycles": compute, "c_pipe": c_pipe(n), "c_start_done": start_done,
             "layer_cyc": cycles, "macs_useful": macs,
             "drain_hidden": all(r["drain_hidden"] for r in per.values()),
             "util_spatial": Fraction(macs, n * n * compute),
             "util_temporal": Fraction(compute, cycles),
             "util_total": Fraction(macs, n * n * cycles)}
    return {"layers": per, "total": total}


def check_n8_reproduces_model() -> None:
    """N = cm.ARRAY must equal the frozen cycle model exactly (per layer and total)."""
    for net in NETS:
        ref = cm.net_cycles(NET_CONFIGS[net]["layers"])
        p = net_proj(net, cm.ARRAY)
        for name, r in ref["layers"].items():
            q = p["layers"][name]
            assert (q["T"], q["compute_cycles"], q["c_pipe"], q["layer_cyc"]) == \
                (r["T"], r["compute_cycles"], r["c_pipe"], r["cycles"]), (net, name)
            assert float(q["util_spatial"]) == r["util_theoretical"]
        assert p["total"]["layer_cyc"] == ref["total"]["cycles"], net
        assert p["total"]["compute_cycles"] == ref["total"]["compute_cycles"], net


def build_rows(sizes=DEFAULT_SIZES) -> list[dict]:
    check_n8_reproduces_model()
    assert cm.ARRAY in sizes, "N=8 rows are the reference for speedup_vs_n8"
    rows = []
    for net in NETS:
        base = net_proj(net, cm.ARRAY)
        n_layers = len(NET_CONFIGS[net]["layers"])
        for n in sizes:
            p = net_proj(net, n)
            items = [(L["name"], L, p["layers"][L["name"]], base["layers"][L["name"]])
                     for L in NET_CONFIGS[net]["layers"]]
            items.append(("total", None, p["total"], base["total"]))
            for name, L, r, b in items:
                row = {**base_meta(net, name, source="model"), "label": LABEL,
                       **({k: L[k] for k in ("IC", "OC", "OH", "OW")} if L else {}),
                       **{k: v for k, v in r.items()
                          if k not in ("util_spatial", "util_temporal", "util_total")},
                       "c_start_done": r.get("c_start_done", 0),
                       "speedup_vs_n8": fmt(Fraction(b["layer_cyc"], r["layer_cyc"]), 4),
                       "util_spatial": fmt(r["util_spatial"]),
                       "util_temporal": fmt(r["util_temporal"]),
                       "util_total": fmt(r["util_total"]),
                       "extra_cycles_per_added_stage": n_layers if L is None else 1,
                       "reproduces_8x8_model": n == cm.ARRAY,
                       "c1_note": NOT_C1, "assumptions": ASSUMPTIONS}
                rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = _setup.out_dir_parser(__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    a = ap.parse_args(argv)
    rows = build_rows(tuple(a.sizes))
    path = write_results_csv(a.out_dir / CSV_NAME, rows, ["net", "layer"] + FIELDS)
    for r in rows:
        print(f"N={r['array_n']:>2} {r['net']:8} {r['layer']:6} T={r['T']:>4} "
              f"cyc={r['layer_cyc']:>7} x{r['speedup_vs_n8']} util={r['util_total']} "
              f"drain_hidden={r['drain_hidden']}")
    print(f"Wrote {path} ({LABEL}; {NOT_C1})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
