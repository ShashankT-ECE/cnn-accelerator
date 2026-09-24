"""V2 cycle model (ARCH_SPEC "Cycle model (V2)", "Tiling / loop nest"; D7 ablation).

Per layer (OH, OW = conv output before pooling; K = IC*KH*KW):

    T        = ceil(OC/8) * OH * ceil(OW/8)       (tiles)
    compute  = T * K                              (one k per cycle per tile)
    cycles   = T * K + C_PIPE                     (C_PIPE: per-layer fill/flush)
    total    = sum(cycles) + C_START              (C_START: once per inference)

C_PIPE and C_START are set by the final RTL pipeline depth and are unknown
until then (default None). With None they are EXCLUDED and the row is labeled
``cycles_basis = "compute_only"`` — they are never guessed.

D7 ablation: the legacy generalized OS model (``python/v2_architecture_model.py``,
imported read-only; its ``main()`` is never called because it writes
data/benchmark/) is decomposed per layer into core + lead-in + pass drain (FC
tail) + clear + result drain, and its core is checked equal to the V2 compute.

Every number produced here is labeled "model" (analytical, not RTL / hardware).

Run:  ../../.venv/bin/python gos_cycle_model.py   (from v2/model)
"""
from __future__ import annotations

import sys

from common import RESULTS_DIR, base_meta, write_results_csv  # also puts python/ on sys.path
from net_config import NET_CONFIGS, NETS

import v2_architecture_model as legacy  # noqa: E402  (legacy, read-only)

ARRAY = 8                    # PE rows (output channels) == PE columns (output x)
PE_COUNT = ARRAY * ARRAY

# Unknown until the RTL pipeline is final. None -> excluded, labeled compute_only.
C_PIPE: int | None = None    # per-layer fill/flush constant
C_START: int | None = None   # once-per-inference start overhead

_MODULE = object()           # sentinel: "use the module-level parameter"

LEGACY_SOURCE = "python/v2_architecture_model.py, model"
LEGACY_LAYERS = {"lenet5": legacy.LENET, "cifar10": legacy.CIFAR}
# Legacy totals spec'd in DECISIONS.md D7 / ARCH_SPEC (asserted, not substituted).
LEGACY_TOTALS_SPEC = {"lenet5": 30013, "cifar10": 225584}


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def _resolve(value, module_value):
    v = module_value if value is _MODULE else value
    if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 0):
        raise ValueError(f"cycle constant must be None or a non-negative int, got {v!r}")
    return v


def layer_tiles(L: dict) -> dict:
    """Tile counts for one layer dict (needs OC, OH, OW)."""
    oc_tiles = _ceil(L["OC"], ARRAY)
    oh_tiles = L["OH"]
    ow_tiles = _ceil(L["OW"], ARRAY)
    return {"oc_tiles": oc_tiles, "oh_tiles": oh_tiles, "ow_tiles": ow_tiles,
            "T": oc_tiles * oh_tiles * ow_tiles}


def layer_cycles(L: dict, c_pipe=_MODULE) -> dict:
    """V2 cycle model for one layer dict (needs IC, OC, OH, OW, K).

    ``c_pipe`` defaults to the module-level ``C_PIPE``; None -> compute only.
    """
    c_pipe = _resolve(c_pipe, C_PIPE)
    t = layer_tiles(L)
    K = L["K"]
    compute = t["T"] * K
    macs = L["OC"] * L["OH"] * L["OW"] * K
    return {
        **t,
        "K": K,
        "compute_cycles": compute,
        "mac_active": compute,   # one mac-enable per tile per k (masked lanes included)
        "stall": 0,              # banked continuous stream: no stall cycles in the model
        "macs": macs,
        "util_theoretical": macs / (PE_COUNT * compute),
        "c_pipe": c_pipe,
        "cycles": compute if c_pipe is None else compute + c_pipe,
        "cycles_basis": "compute_only" if c_pipe is None else "compute+c_pipe",
    }


def net_cycles(layers, c_pipe=_MODULE, c_start=_MODULE) -> dict:
    """V2 cycle model for a list of layer dicts: per-layer rows + total.

    total cycles = sum(layer cycles) + C_START (excluded when None).
    """
    c_pipe = _resolve(c_pipe, C_PIPE)
    c_start = _resolve(c_start, C_START)
    rows = {L["name"]: layer_cycles(L, c_pipe) for L in layers}
    compute = sum(r["compute_cycles"] for r in rows.values())
    macs = sum(r["macs"] for r in rows.values())
    cycles = sum(r["cycles"] for r in rows.values())
    terms = (["c_pipe"] if c_pipe is not None else []) + (["c_start"] if c_start is not None else [])
    total = {
        "T": sum(r["T"] for r in rows.values()),
        "compute_cycles": compute,
        "mac_active": sum(r["mac_active"] for r in rows.values()),
        "stall": sum(r["stall"] for r in rows.values()),
        "macs": macs,
        "util_theoretical": macs / (PE_COUNT * compute),
        "c_pipe": c_pipe,
        "c_start": c_start,
        "cycles": cycles if c_start is None else cycles + c_start,
        "cycles_basis": "compute_only" if not terms else "+".join(["compute"] + terms),
    }
    return {"layers": rows, "total": total}


# ---- D7 ablation (legacy generalized OS model, labeled "model") --------------

def legacy_layer_terms(LL: dict) -> dict:
    """Decompose one legacy-model layer into D7 terms using the legacy functions.

    Conv: summed legacy ``os_group_breakdown`` per group — core = taps (IC*K*K),
    lead-in = passes*(P-1), pass drain = passes*1, clear = 1, result drain = 2*rows.
    FC (legacy ``os_fc_cycles`` = groups*(IC+2) + 2*OC): core = groups*IC,
    clear = groups, FC tail (counted as pass drain, per D7) = groups,
    result drain = 2*OC. Both are checked to sum to the legacy ``os_cycles``.
    """
    total = legacy.os_cycles(LL)["cycles"]
    if LL["K"] == 1:
        groups = legacy.os_fc_cycles(LL["IC"], LL["OC"])["channel_groups"]
        terms = {"core": groups * LL["IC"], "lead_in": 0, "pass_drain": groups,
                 "clear": groups, "result_drain": 2 * LL["OC"]}
    else:
        IC, OC, K, H, W = LL["IC"], LL["OC"], LL["K"], LL["H"], LL["W"]
        terms = dict.fromkeys(("core", "lead_in", "pass_drain", "clear", "result_drain"), 0)
        for g in range(_ceil(OC, legacy.ARRAY)):
            rows = min(legacy.ARRAY, OC - legacy.ARRAY * g)
            Ps = [legacy.ARRAY] * (W // legacy.ARRAY) + ([W % legacy.ARRAY] if W % legacy.ARRAY else [])
            for P in Ps:
                b = legacy.os_group_breakdown(IC, K, P, rows)
                terms["core"] += H * b["passes"] * b["taps_per_pass"]
                terms["lead_in"] += H * b["passes"] * b["lead_in_per_pass"]
                terms["pass_drain"] += H * b["passes"] * b["drain_per_pass"]
                terms["clear"] += H * b["clear"]
                terms["result_drain"] += H * (b["result_drain"] + b["weight_bram_stall"])
    if sum(terms.values()) != total:
        raise AssertionError(f"legacy decomposition {terms} != legacy total {total}")
    return {"total": total, **terms}


def _check_layer_map(net: str) -> None:
    """Verify legacy layer names/shapes map 1:1 onto net_config layers."""
    ours = NET_CONFIGS[net]["layers"]
    theirs = LEGACY_LAYERS[net]
    if [L["name"] for L in ours] != [LL["name"] for LL in theirs]:
        raise AssertionError(f"{net}: layer names differ between legacy model and net_config")
    for L, LL in zip(ours, theirs):
        same = (L["IC"] == LL["IC"] and L["OC"] == LL["OC"] and L["KH"] == L["KW"] == LL["K"]
                and L["OH"] == LL["H"] and L["OW"] == LL["W"]
                and L["OC"] * L["OH"] * L["OW"] * L["K"] == LL["MACs"])
        if not same:
            raise AssertionError(f"{net} {L['name']}: shape mismatch legacy {LL} vs ours {L}")


def ablation(net: str) -> dict:
    """Per-layer legacy terms next to V2 compute; asserts core == compute, spec totals."""
    _check_layer_map(net)
    v2 = net_cycles(NET_CONFIGS[net]["layers"], c_pipe=None, c_start=None)
    layers = {}
    for LL in LEGACY_LAYERS[net]:
        t = legacy_layer_terms(LL)
        if t["core"] != v2["layers"][LL["name"]]["compute_cycles"]:
            raise AssertionError(f"{net} {LL['name']}: legacy core {t['core']} != V2 compute")
        layers[LL["name"]] = t
    total = {k: sum(t[k] for t in layers.values()) for k in next(iter(layers.values()))}
    legacy_reported = legacy.compute()["workloads"][net]["total"]["OS_cycles"]  # no file I/O
    if not (total["total"] == legacy_reported == LEGACY_TOTALS_SPEC[net]):
        raise AssertionError(f"{net}: legacy total {total['total']} / {legacy_reported} "
                             f"!= spec {LEGACY_TOTALS_SPEC[net]}")
    return {"layers": layers, "total": total}


# ---- results ------------------------------------------------------------------

FIELDS = ["IC", "OC", "OH", "OW", "K", "oc_tiles", "oh_tiles", "ow_tiles", "T",
          "compute_cycles", "mac_active", "stall", "macs", "util_theoretical",
          "c_pipe", "c_start", "cycles", "cycles_basis",
          "legacy_total", "legacy_core", "legacy_lead_in", "legacy_pass_drain",
          "legacy_clear", "legacy_result_drain", "legacy_source"]


def _legacy_cols(t: dict) -> dict:
    return {"legacy_total": t["total"], "legacy_core": t["core"], "legacy_lead_in": t["lead_in"],
            "legacy_pass_drain": t["pass_drain"], "legacy_clear": t["clear"],
            "legacy_result_drain": t["result_drain"], "legacy_source": LEGACY_SOURCE}


def build_rows() -> list[dict]:
    rows = []
    for net in NETS:
        res = net_cycles(NET_CONFIGS[net]["layers"])
        abl = ablation(net)
        for L in NET_CONFIGS[net]["layers"]:
            r = res["layers"][L["name"]]
            row = {**base_meta(net, L["name"], source="model"),
                   **{k: L[k] for k in ("IC", "OC", "OH", "OW")},
                   **{k: r[k] for k in FIELDS if k in r},
                   **_legacy_cols(abl["layers"][L["name"]])}
            rows.append(row)
        tot = res["total"]
        rows.append({**base_meta(net, "total", source="model"),
                     **{k: tot[k] for k in FIELDS if k in tot},
                     **_legacy_cols(abl["total"])})
    for row in rows:
        for k in ("c_pipe", "c_start"):
            if row.get(k) is None:
                row[k] = ""
        row["util_theoretical"] = f"{row['util_theoretical']:.6f}"
    return rows


def _print(rows: list[dict]) -> None:
    hdr = (f"{'net':8} {'layer':6} {'T':>5} {'K':>5} {'compute':>8} {'util':>7} {'cycles':>8} "
           f"{'basis':>12} | {'leg_tot':>8} {'core':>7} {'lead':>6} {'pdrain':>6} "
           f"{'clear':>5} {'rdrain':>6}")
    print("V2 cycle model (model; C_PIPE/C_START unknown until RTL -> compute_only)")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['net']:8} {r['layer']:6} {r['T']:>5} {str(r.get('K', '')):>5} "
              f"{r['compute_cycles']:>8} {r['util_theoretical']:>7} {r['cycles']:>8} "
              f"{r['cycles_basis']:>12} | {r['legacy_total']:>8} {r['legacy_core']:>7} "
              f"{r['legacy_lead_in']:>6} {r['legacy_pass_drain']:>6} {r['legacy_clear']:>5} "
              f"{r['legacy_result_drain']:>6}")
    print(f"legacy columns: {LEGACY_SOURCE}")


def main() -> int:
    rows = build_rows()
    path = write_results_csv(RESULTS_DIR / "cycle_model.csv", rows, ["net", "layer"] + FIELDS)
    _print(rows)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
