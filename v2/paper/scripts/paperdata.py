"""Data assembly shared by tables and figures. Every value is a V(value, origin) whose origin
names the CSV file, line(s) and column (or the formula over such values)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from paperlib import Artifact, _rel, fnum, inum


@dataclass(frozen=True)
class V:
    value: float
    origin: str

    def fmt(self, art: Artifact, fmt: str) -> str:
        return art.num(self.value, fmt, origin=self.origin)


def at(row: dict, col: str, conv=fnum) -> V:
    v = row.get(col, "")
    if v in ("", None):
        raise ValueError(f"empty {col} at {row['_file']}:{row['_line']}")
    return V(conv(v), f"{_rel(Path(row['_file']))}:{row['_line']}:{col}")


def err_pct(x: V, ref: V) -> V:
    return V((x.value - ref.value) / ref.value * 100.0,
             f"computed (x-ref)/ref*100, x={x.origin}, ref={ref.origin}")


def div(a: V, b: V, what: str = "a/b") -> V:
    return V(a.value / b.value, f"computed {what}, a={a.origin}, b={b.origin}")


def latest(rows: list[dict], key) -> dict:
    """Latest-timestamp row per key (duplicates from re-runs)."""
    best: dict = {}
    for r in rows:
        k = key(r)
        if k not in best or r.get("timestamp", "") >= best[k].get("timestamp", ""):
            best[k] = r
    return best


def layer_key(name: str) -> str:
    return re.sub(r"^L\d+_", "", name)


def lines_of(rows: list[dict]) -> str:
    ls = sorted(r["_line"] for r in rows)
    return f"{_rel(Path(rows[0]['_file']))}:{ls[0]}-{ls[-1]}" if len(ls) > 1 else \
        f"{_rel(Path(rows[0]['_file']))}:{ls[0]}"


# --------------------------------------------------------------------------------------------
# A3: per-layer model / RTL-sim / board cycles
# --------------------------------------------------------------------------------------------
def a3_net(art: Artifact, net: str) -> list[dict] | None:
    """Rows: {layer, model, rtl, rtl_images, hw, hw_images, hw_distinct, mac_model, util_th,
    rtl_mac, is_total}. Values are V or None."""
    mrows = art.rows("cycle_model.csv", net=net)
    if not mrows:
        return None
    by = latest(mrows, lambda r: r["layer"])
    if len(by) != len(mrows):
        art.check(f"cycle_model.csv {net}: duplicate layer rows, latest timestamp used")
    layers = [r for r in mrows if r["layer"] != "total" and by[r["layer"]] is r]
    total = by.get("total")
    out = []
    for r in layers:
        out.append({"layer": r["layer"], "is_total": False, "model": at(r, "cycles", inum),
                    "mac_model": at(r, "mac_active", inum), "util_th": at(r, "util_theoretical"),
                    "model_row": r})
    if total is not None:
        out.append({"layer": "total", "is_total": True, "model": at(total, "cycles", inum),
                    "mac_model": at(total, "mac_active", inum),
                    "util_th": at(total, "util_theoretical"), "model_row": total})
    nl = len(layers)

    # RTL sim, whole-network jobs (same LAYER_CYC semantics as the board run)
    nrows = art.rows("rtl_network.csv", net=net) or []
    for d in out:
        d.update(rtl=None, rtl_images=0, rtl_mac=None)
    if nrows:
        lists = [[int(x) for x in r["rtl_layer_cycles"].split(",")] for r in nrows]
        mlists = [[int(x) for x in r["model_layer_cycles"].split(",")] for r in nrows]
        src = lines_of(nrows)
        for i, d in enumerate(out[:nl]):
            vals = {lst[i] for lst in lists if len(lst) > i}
            if len(vals) != 1:
                art.check(f"rtl_network.csv {net} {d['layer']}: LAYER_CYC differs over images {sorted(vals)}; "
                          "min shown")
            d["rtl"] = V(min(vals), f"{src}:rtl_layer_cycles[{i}] (identical over {len(nrows)} images)"
                         if len(vals) == 1 else f"{src}:rtl_layer_cycles[{i}] min over images")
            d["rtl_images"] = len(nrows)
            mv = {m[i] for m in mlists}
            if mv != {d["model"].value}:
                art.check(f"rtl_network.csv {net} {d['layer']}: model_layer_cycles {sorted(mv)} != "
                          f"cycle_model.csv {d['model'].value}")
        extra = {tuple(lst[nl:]) for lst in lists}
        if any(len(e) for e in extra):
            nz = [e for e in extra if any(e)]
            art.check(f"rtl_network.csv {net}: rtl_layer_cycles has {len(next(iter(extra)))} extra entr"
                      f"{'y' if len(next(iter(extra))) == 1 else 'ies'} beyond the {nl} layers "
                      f"({'all 0' if not nz else 'NONZERO ' + str(nz)})")
        if out and out[-1]["is_total"]:
            tv = {inum(r["rtl_total"]) for r in nrows}
            macs = {inum(r["mac"]) for r in nrows}
            out[-1]["rtl"] = V(min(tv), f"{src}:rtl_total (identical over {len(nrows)} images)"
                               if len(tv) == 1 else f"{src}:rtl_total min")
            out[-1]["rtl_images"] = len(nrows)
            out[-1]["rtl_mac"] = V(min(macs), f"{src}:mac")
            if len(tv) != 1:
                art.check(f"rtl_network.csv {net}: rtl_total differs over images {sorted(tv)}")
            if {inum(r["model_total"]) for r in nrows} != {out[-1]["model"].value}:
                art.check(f"rtl_network.csv {net}: model_total != cycle_model.csv total")

    # RTL sim, single-layer jobs: cross-check + per-layer MAC_ACTIVE
    crows = art.rows("rtl_cycles.csv", net=net, kind="layer") or []
    for d in out[:nl]:
        rs = [r for r in crows if layer_key(r["layer"]) == d["layer"]]
        if not rs:
            continue
        cyc = {inum(r["rtl_cycles"]) for r in rs}
        macs = {inum(r["rtl_mac"]) for r in rs}
        d["rtl_mac"] = V(min(macs), f"{lines_of(rs)}:rtl_mac (single-layer jobs, {len(rs)} images)")
        d["rtl_single"] = V(min(cyc), f"{lines_of(rs)}:rtl_cycles")
        if d["rtl"] is not None and cyc != {d["rtl"].value}:
            art.check(f"rtl_cycles.csv {net} {d['layer']}: single-layer rtl_cycles {sorted(cyc)} != "
                      f"network LAYER_CYC {d['rtl'].value}")
        if d["rtl"] is None:
            d["rtl"] = d["rtl_single"]
            d["rtl_images"] = len(rs)

    # Board
    hrows = art.rows("hw_a2_a3_cycles.csv", hw=True, net=net)
    for d in out:
        d.update(hw=None, hw_images=None, hw_distinct=None, hw_row=None)
    if hrows:
        hb = latest(hrows, lambda r: r["layer"])
        for d in out:
            r = hb.get(d["layer"])
            if r is None or r.get("hw_cycles", "") == "":
                continue
            d["hw"] = at(r, "hw_cycles", inum)
            d["hw_images"] = at(r, "images", inum)
            d["hw_distinct"] = at(r, "hw_distinct", inum) if r.get("hw_distinct") else None
            d["hw_row"] = r
            if d["hw_distinct"] is not None and d["hw_distinct"].value != 1:
                art.check(f"hw_a2_a3_cycles.csv {net} {d['layer']}: {d['hw_distinct'].value} distinct cycle "
                          f"counts over images (min {r.get('hw_min')}, max {r.get('hw_max')})")
            if r.get("model_cycles") and inum(r["model_cycles"]) != d["model"].value:
                art.check(f"hw_a2_a3_cycles.csv {net} {d['layer']}: model_cycles {r['model_cycles']} != "
                          f"cycle_model.csv {d['model'].value}")
            if r.get("hw_err_pct") not in (None, ""):
                mine = (d["hw"].value - d["model"].value) / d["model"].value * 100
                if abs(mine - fnum(r["hw_err_pct"])) > 1e-6:
                    art.check(f"hw_a2_a3_cycles.csv {net} {d['layer']}: hw_err_pct {r['hw_err_pct']} != "
                              f"recomputed {mine:.6f}")
    for d in out:
        d["rtl_err"] = err_pct(d["rtl"], d["model"]) if d["rtl"] is not None else None
        d["hw_err"] = err_pct(d["hw"], d["model"]) if d["hw"] is not None else None
    return out


# --------------------------------------------------------------------------------------------
# Implemented clocks (impl_gos.csv)
# --------------------------------------------------------------------------------------------
# DECISIONS D21 (user, 2026-09-30): the 250 MHz build is the performance clock, the build the
# KV260 runs. A variant above it met timing in Vivado but is POST-IMPLEMENTATION ONLY: PYNQ does
# not reprogram the PS PLLs and the boot image's PL clock source has no integer divider to it (D19).
PERFORMANCE_CLOCK_MHZ = 250
PERF_MARK, POSTIMPL_ONLY_MARK = "$^\\star$", "$^\\ddagger$"


def is_performance(r: dict) -> bool:
    return abs(fnum(r["pl_clk0_mhz_requested"]) - PERFORMANCE_CLOCK_MHZ) < 0.5


def is_postimpl_only(r: dict) -> bool:
    """True for a timing-met variant that cannot be run on the board (above the performance clock)."""
    return fnum(r["pl_clk0_mhz_requested"]) > PERFORMANCE_CLOCK_MHZ + 0.5


def clock_marks(r: dict) -> str:
    return PERF_MARK if is_performance(r) else POSTIMPL_ONLY_MARK if is_postimpl_only(r) else ""


def clock_notes(art: "Artifact", vs: list[dict]) -> list[str]:
    """Footnotes for the marks above (the clock numbers are impl_gos.csv cells)."""
    perf = [r for r in vs if is_performance(r)]
    only = [r for r in vs if is_postimpl_only(r)]
    out = []
    mhz = lambda r: art.cell(r, "pl_clk0_mhz_requested", "int")  # noqa: E731
    if perf:
        out.append(f"{PERF_MARK}Performance clock: the {mhz(perf[0])}~MHz build is the one run on the KV260.")
    if only:
        out.append(f"{POSTIMPL_ONLY_MARK}Post-implementation only ({', '.join(mhz(r) for r in only)}~MHz): "
                   "timing met in Vivado, never run on the KV260. PYNQ does not reprogram the PS PLLs, "
                   "and the PL clock source of the board's boot image has no integer divider to this "
                   "frequency" + (f"; the board runs the {mhz(perf[0])}~MHz build." if perf else "."))
    return out


def impl_variants(art: Artifact) -> list[dict]:
    """Latest clean row per requested clock that met timing (D14: WNS >= 0, WHS >= 0,
    0 critical warnings, timing_met). Sorted by clock."""
    rows = art.rows("impl_gos.csv", top="gos_top_wrapper") or []
    ok, failed = [], []
    for r in rows:
        try:
            met = (r.get("timing_met") == "True" and fnum(r["wns_ns"]) >= 0 and fnum(r["whs_ns"]) >= 0
                   and inum(r.get("critical_warnings") or 0) == 0)
        except ValueError:
            met = False
        (ok if met else failed).append(r)
    for r in failed:
        art.check(f"impl_gos.csv line {r['_line']} ({r.get('pl_clk0_mhz_requested')} MHz): excluded, "
                  f"timing not met / critical warnings (WNS {r.get('wns_ns')}, WHS {r.get('whs_ns')})")
    best = latest(ok, lambda r: r["pl_clk0_mhz_requested"])
    for r in ok:
        if best[r["pl_clk0_mhz_requested"]] is not r:
            art.check(f"impl_gos.csv line {r['_line']}: older build at {r['pl_clk0_mhz_requested']} MHz "
                      "superseded by a later row")
    return sorted(best.values(), key=lambda r: fnum(r["pl_clk0_mhz_requested"]))


# --------------------------------------------------------------------------------------------
# B1 / B2 SOM-rail power (INA260), v2/board/power_log.py summary files
# --------------------------------------------------------------------------------------------
INA_LABEL = "SOM-rail power (INA260)"      # power_log.LABEL; checked against the CSV column; the only
                                           # power source of the board sessions


def ci(art: Artifact, row: dict, med: str, lo: str, hi: str, fmt: str, scale: float = 1.0) -> str:
    """'median [lo, hi]' (95 % CI) text from one CSV row; just the median if the CI columns are
    absent or empty (e.g. rows written before the CI columns existed)."""
    def one(col):
        return V(fnum(row[col]) * scale, f"{_rel(Path(row['_file']))}:{row['_line']}:{col}"
                 + (f"*{scale:g}" if scale != 1 else "")).fmt(art, fmt)
    if not row.get(med):
        return "--"
    txt = one(med)
    if row.get(lo) not in (None, "") and row.get(hi) not in (None, ""):
        txt += " [" + one(lo) + ", " + one(hi) + "]"
    return txt


def power_summaries(art: Artifact, prefix: str) -> list[dict]:
    """One entry per (net, clock) from <prefix>_summary*.csv: {net, clock, mean, std, repeats, file}.
    Files are found by glob (the tag -- per net or per clock -- is not relied upon; net and clock come
    from the rows). The latest run (git_commit/timestamp of its mean row) wins per (net, clock)."""
    out: dict = {}
    for name in art.store.glob(f"{prefix}_summary*.csv"):
        rows = art.rows(name, hw=True) or []
        groups: dict = {}
        for r in rows:
            groups.setdefault((r["net"], r.get("clock_mhz", "")), []).append(r)
        for (net, clk), rs in groups.items():
            mean = [r for r in rs if r.get("row_kind") == "mean"]
            std = [r for r in rs if r.get("row_kind") == "std"]
            rep = [r for r in rs if r.get("row_kind") == "repeat"]
            if not mean:
                art.check(f"{name} {net} @ {clk}: no mean row")
                continue
            m = max(mean, key=lambda r: r["timestamp"])
            if len(mean) > 1:
                art.check(f"{name} {net} @ {clk}: {len(mean)} mean rows (several runs); latest used")
                rep = [r for r in rep if r["timestamp"][:16] <= m["timestamp"][:16]][-int(m["n_repeats"] or 0):]
                std = [max(std, key=lambda r: r["timestamp"])] if std else []
            for r in (m, *std):
                if r.get("measurement") and r["measurement"] != INA_LABEL:
                    art.check(f"{name} line {r['_line']}: measurement label {r['measurement']!r} != {INA_LABEL!r}")
            if rep:                                   # mean row consistent with the repeats?
                for k in ("accel_dp_w", "accel_energy_per_image_mj", "cpu_dp_w", "cpu_energy_per_image_mj"):
                    vals = [fnum(r[k]) for r in rep if r.get(k)]
                    if vals and m.get(k) and abs(sum(vals) / len(vals) - fnum(m[k])) > 1e-5 * max(1, abs(fnum(m[k]))):
                        art.check(f"{name} {net} @ {clk}: mean {k} {m[k]} != mean of repeats {sum(vals)/len(vals):.6f}")
                if len(rep) != inum(m["n_repeats"]):
                    art.check(f"{name} {net} @ {clk}: {len(rep)} repeat rows, n_repeats={m['n_repeats']}")
            key = (net, clk)
            if key in out and out[key]["mean"]["timestamp"] >= m["timestamp"]:
                continue
            out[key] = {"net": net, "clock": clk, "mean": m, "std": std[0] if std else None,
                        "repeats": rep, "file": name}
    return sorted(out.values(), key=lambda e: (e["net"], fnum(e["clock"]) if e["clock"] else 0))


def pm(art: Artifact, e: dict, col: str, fmt: str, scale: float = 1.0) -> str:
    """'mean ± std' text for a summary column (std row if present)."""
    m = e["mean"]
    if not m.get(col):
        return "--"
    txt = V(fnum(m[col]) * scale, f"{_rel(Path(m['_file']))}:{m['_line']}:{col}" + (f"*{scale:g}" if scale != 1 else "")
            ).fmt(art, fmt)
    s = e["std"]
    if s is not None and s.get(col):
        txt += r"\,$\pm$\," + V(fnum(s[col]) * scale, f"{_rel(Path(s['_file']))}:{s['_line']}:{col}"
                                + (f"*{scale:g}" if scale != 1 else "")).fmt(art, fmt)
    return txt
