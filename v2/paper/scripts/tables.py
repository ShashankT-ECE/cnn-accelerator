"""Table generators. Each returns an Artifact after writing generated/<name>.tex."""
from __future__ import annotations

import statistics
from pathlib import Path

from paperdata import (INA_LABEL, V, a3_net, at, ci, clock_marks, clock_notes, div, impl_variants, is_performance,
                       is_postimpl_only, latest, lines_of, pm, power_summaries)
from paperlib import (NETS, PH_BOARD, Artifact, Store, _rel, fnum, inum, latex_table, pretty_net,
                      row_origin, tex_escape, write_text)

META_COLS = {"timestamp", "git_commit", "git_dirty", "vivado_version", "bitstream_sha256", "board_id",
             "net", "layer", "clock_mhz", "source", "duration_s", "num_inferences", "_file", "_line"}

NUMPY_KINDS = {"cpu_int8_ref", "cpu_fp32_numpy"}
NUMPY_MARK = "$^\\dagger$"
CPU_KINDS = {"cpu_int8_ref": "INT8 numpy (ref.)", "cpu_fp32_numpy": "FP32 numpy",
             "cpu_ort_fp32": "ORT FP32", "cpu_ort_int8": "ORT INT8"}
B3_PHASES = ["input_write", "status_clear", "start_done", "start_write", "poll", "logit_read", "ps_dequant",
             "end_to_end", "counter_read", "pl_compute"]


class Ctx:
    """Table context (also passed to tables_extra.make by make_all.py):
    store     paperlib.Store: .load(name, hw), .glob(pattern, hw), rules (git_dirty, source, paper_grade)
    out       output directory (generated/ or generated/dryrun/)
    booktabs  bool; dryrun: bool (dry-run rows, watermarked)
    art(name) -> paperlib.Artifact (rows(), cell(), num(), label(), placeholder(), check())
    table(art, colspec, header, body, caption, label, notes=None, wide=False) -> Artifact (writes .tex)
    emit(art, tex) -> Artifact;  board_label -> "KV260" / "KV260 (DRY RUN)"
    fctx      figures.FCtx for figures (set by make_all.py)
    Helpers for extra tables: paperdata.V / at / div / latest / pm (mean +- std) / ci (median [95 % CI]) /
    power_summaries; paperlib.PH_BOARD placeholder text "TBD (board)"."""

    def __init__(self, store: Store, out: Path, booktabs: bool, dryrun: bool):
        self.store, self.out, self.booktabs, self.dryrun = store, out, booktabs, dryrun

    def art(self, name: str) -> Artifact:
        return Artifact(self.store, name, "table")

    def emit(self, art: Artifact, tex: str) -> Artifact:
        p = self.out / f"{art.name}.tex"
        write_text(p, tex)
        art.outputs.append(_rel(p))
        return art

    def table(self, art, colspec, header, body, caption, label, notes=None, wide=False):
        return self.emit(art, latex_table(art, colspec, header, body, caption, label, notes,
                                          booktabs=self.booktabs, wide=wide, dryrun=self.dryrun))

    @property
    def board_label(self) -> str:
        return "KV260 (DRY RUN)" if self.dryrun else "KV260"


_NUMW = {2: "two", 3: "three", 4: "four", 5: "five"}


def _nsess(c: Ctx) -> int:
    """Number of board sessions behind the replicated board rows (1 = first session only)."""
    return len(c.store.session_paths("hw_b3_breakdown.csv", True))


def _sessw(c: Ctx) -> str:
    n = _nsess(c)
    return _NUMW.get(n, str(n))


def _bracket(c: Ctx) -> str:
    """Header text of a 'median [..]' cell: [min, max] over the sessions, else the 95 % CI of one session."""
    return "median [min, max]" if _nsess(c) > 1 else "median [95\\% CI]"


SINGLE = "first session only"        # board rows that exist for one session only (no replicates)


def _lab(art: Artifact, r: dict, col: str, n: int | None = None) -> str:
    v = r[col][:n] if n else r[col]
    return art.label(tex_escape(v), f"{r['_file']}:{r['_line']}:{col}" + (f"[:{n}]" if n else ""))


def _mhz(art: Artifact, r: dict) -> str:
    return art.cell(r, "pl_clk0_mhz_requested", "int")


# --------------------------------------------------------------------------------------------
def t1_impl(c: Ctx) -> Artifact:
    """A6 / T1: post-implementation results, one column per implemented clock."""
    art = c.art("tab_t1_impl")
    vs = impl_variants(art)
    if not vs:
        body = [["Implementation", art.placeholder("impl_gos.csv: no clean timing-met row", "TBD (impl)")]]
        return c.table(art, "@{}ll@{}", ["Quantity & Value"], body,
                       "Implementation results (post-implementation).", "tab:impl")
    n = len(vs)
    head = "Quantity & " + " & ".join(f"{_mhz(art, r)}~MHz{clock_marks(r)}" for r in vs)
    status = lambda r: art.label("post-impl.\\ only" if is_postimpl_only(r) else  # noqa: E731
                                 "post-impl.; run on KV260" if is_performance(r) else "post-impl.",
                                 "DECISIONS D21 (performance clock) / D19 (PLL constraint)")
    grp = lambda t: f"\\multicolumn{{{n + 1}}}{{@{{}}l}}{{\\emph{{{t}}}}} \\\\"
    row = lambda lab, col, fmt="int": [lab] + [art.cell(r, col, fmt) for r in vs]
    body = [grp("Full design (gos\\_top + AXI shell)"),
            row("CLB LUT", "clb_luts"), row("CLB FF", "clb_registers"), row("LUT as memory", "lut_as_memory"),
            row("DSP48E2", "dsps"), row("RAMB36", "ramb36"), row("RAMB18", "ramb18"),
            grp("gos\\_core only"),
            row("CLB LUT", "core_luts"), row("CLB FF", "core_registers"), row("DSP48E2", "core_dsps"),
            row("RAMB36", "core_ramb36"), row("RAMB18", "core_ramb18"),
            grp("Timing and build"),
            ["Status"] + [status(r) for r in vs],
            row("WNS (ns)", "wns_ns", "f3"), row("WHS (ns)", "whs_ns", "f3"),
            row("pl\\_clk0 requested (MHz)", "pl_clk0_mhz_requested", "int"),
            row("pl\\_clk0 actual (MHz)", "pl_clk0_mhz_actual", "f3"),
            ["BUILD\\_ID"] + [f"\\texttt{{{_lab(art, r, 'build_id')}}}" for r in vs],
            ["Bitstream SHA-256"] + [f"\\texttt{{{_lab(art, r, 'bit_sha256', 8)}}}" for r in vs]]
    vv = {r["vivado_version"] for r in vs}
    vtxt = " ".join(art.label(tex_escape(v), "impl_gos.csv:vivado_version") for v in sorted(vv))
    for r in vs:
        if r.get("synth_scan_pass") not in ("True", None, ""):
            art.check(f"impl_gos.csv line {r['_line']} ({r['pl_clk0_mhz_requested']} MHz): synth_scan_pass="
                      f"{r['synth_scan_pass']} (synth_truncated={r.get('synth_truncated')}, "
                      f"synth_unwaived_groups={r.get('synth_unwaived_groups')})")
    return c.table(art, "@{}l" + "r" * n + "@{}", [head], body,
                   f"Implementation results (post-implementation, Vivado {vtxt}, "
                   "xck26-sfvc784-2LV-c). One column per implemented clock that met timing; every "
                   "number is a post-implementation result.",
                   "tab:impl",
                   notes=["Source: impl\\_gos.csv (post\\_impl). Resources are Vivado utilization "
                          "report counts; no percentages (device totals not in a results CSV).",
                          *clock_notes(art, vs)])


# --------------------------------------------------------------------------------------------
def a1_accuracy(c: Ctx) -> Artifact:
    art = c.art("tab_a1_accuracy")
    refver = {"lenet5": "lenet5_v1", "cifar10": "cifar10_r2"}
    body = []
    for net in NETS:
        acc = art.rows("reference_accuracy.csv", net=net, reference_version=refver[net]) or []
        by = latest(acc, lambda r: r["precision"])
        fp = art.cell(by["FP32"], "accuracy_pct", "f2") if "FP32" in by else "--"
        i8 = art.cell(by["INT8"], "accuracy_pct", "f2") if "INT8" in by else "--"
        if net == "cifar10" and "INT8" in by:      # cross-check the r2 record
            r2 = art.rows("cifar10_r2_accuracy.csv", reference_version="cifar10_r2", split="test") or []
            for r in r2:
                ref = by.get(r["precision"])
                if ref and inum(ref["correct"]) != inum(r["correct"]):
                    art.check(f"cifar10 r2 {r['precision']} test: reference_accuracy.csv {ref['correct']} != "
                              f"cifar10_r2_accuracy.csv {r['correct']}")
        nr = art.rows("rtl_network.csv", net=net) or []
        if nr:
            ok = sum(r["logits_match"] == "True" for r in nr)
            rtl = (art.num(ok, "int", origin=f"{lines_of(nr)}: count logits_match==True") + "/" +
                   art.num(len(nr), "int", origin=f"{lines_of(nr)}: row count"))
        else:
            rtl = "--"
        hw = art.rows("hw_a1_accuracy.csv", hw=True, net=net)
        if hw:
            h = latest(hw, lambda r: r["net"])[net]
            mism = (art.cell(h, "logit_mismatch_images", "int") + "/" + art.cell(h, "images", "int"))
            hacc = art.cell(h, "hw_accuracy_pct", "f2")
            if "INT8" in by and inum(h["golden_correct"]) != inum(by["INT8"]["correct"]):
                art.check(f"hw_a1_accuracy.csv {net}: golden_correct {h['golden_correct']} != "
                          f"reference_accuracy.csv INT8 {by['INT8']['correct']}")
        else:
            mism = art.placeholder(f"A1 {net}: hw_a1_accuracy.csv")
            hacc = art.placeholder(f"A1 {net}: hw_a1_accuracy.csv")
        body.append([pretty_net(net), fp, i8, rtl, mism, hacc])
    hdr = [r"& \multicolumn{2}{c}{Model acc.\ (\%)} & RTL sim & \multicolumn{2}{c}{" + c.board_label + "} \\\\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(l){5-6}",
           r"Net & FP32 & INT8 & bit-exact & mism. & acc.\ (\%)"]
    return c.table(art, "@{}lrrrrr@{}", hdr, body,
                   "A1 correctness: accuracy of record (model, full test sets) and bit-exact INT32 logits "
                   "(RTL sim, sampled images; KV260, full test sets).", "tab:accuracy",
                   notes=["Sources: reference\\_accuracy.csv (model; CIFAR-10 = r2 reference), "
                          "rtl\\_network.csv (RTL sim), hw\\_a1\\_accuracy.csv (measured on KV260). "
                          "``mism.'' = images whose logits differ from the golden model."])


# --------------------------------------------------------------------------------------------
def a2_latency(c: Ctx) -> Artifact:
    art = c.art("tab_a2_latency")
    vs = impl_variants(art)
    data = {net: a3_net(art, net) for net in NETS}
    tot = {net: (d[-1] if d and d[-1]["is_total"] else None) for net, d in data.items()}
    rows = []

    def line(label, fn):
        rows.append([label] + [fn(net) for net in NETS])

    line("Cycles, model", lambda n: tot[n]["model"].fmt(art, "int") if tot[n] else "--")
    line("Cycles, RTL sim", lambda n: tot[n]["rtl"].fmt(art, "int") if tot[n] and tot[n]["rtl"] else "--")
    line(f"Cycles, {c.board_label}", lambda n: tot[n]["hw"].fmt(art, "int") if tot[n] and tot[n]["hw"]
         else art.placeholder(f"A2 {n}: hw_a2_a3_cycles.csv total"))
    for r in vs:
        f = at(r, "pl_clk0_mhz_actual")
        lab = f"Latency (\\textmu s) @ {_mhz(art, r)}~MHz$^\\dagger${clock_marks(r)}"
        line(lab, lambda n, f=f: div(tot[n]["rtl"], f, "cycles/MHz = us").fmt(art, "f2")
             if tot[n] and tot[n]["rtl"] else "--")
    hw = {n: latest(art.rows("hw_a2_a3_cycles.csv", hw=True, net=n) or [], lambda r: r["layer"]).get("total")
          for n in NETS}
    line(f"PL latency (\\textmu s) at the read-back clock, {c.board_label}", lambda n: art.cell(hw[n], "hw_us", "f2")
         if hw[n] and hw[n].get("hw_us") else art.placeholder(f"A2 {n}: hw_us"))
    line(f"$f_\\mathrm{{meas}}$ cross-check (MHz), {c.board_label} ({SINGLE})", lambda n: art.cell(hw[n], "f_meas_mhz", "f3")
         if hw[n] and hw[n].get("f_meas_mhz") else art.placeholder(f"A2 {n}: f_meas_mhz (exp_fclk_cal)"))
    line(f"Clock read back (MHz), {c.board_label}", lambda n: art.cell(hw[n], "clock_mhz", "f3")
         if hw[n] and hw[n].get("clock_mhz") else art.placeholder(f"A2 {n}: clock_mhz"))
    line(f"Wall-clock/image (\\textmu s), median [95\\% CI], {c.board_label} ({SINGLE})",
         lambda n: ci(art, hw[n], "wall_us_median", "wall_us_ci_lo", "wall_us_ci_hi", "f1")
         if hw[n] and hw[n].get("wall_us_median") else art.placeholder(f"A2 {n}: wall_us_median"))
    return c.table(art, "@{}lrr@{}", ["& LeNet-5 & CIFAR-10"], rows,
                   "A2 latency per image (single-image jobs).", "tab:latency",
                   notes=["$^\\dagger$RTL sim cycles at post-impl clock (cycles / pl\\_clk0 actual from "
                          "impl\\_gos.csv), computed, not measured. Sources: cycle\\_model.csv (model), "
                          "rtl\\_network.csv (RTL sim), hw\\_a2\\_a3\\_cycles.csv (measured on KV260; PL "
                          "latency = cycle counter / the pl\\_clk0 PLL read-back, the clock of record; "
                          "$f_\\mathrm{meas}$ = cross-check of that clock (cycle counter vs "
                          "CLOCK\\_MONOTONIC\\_RAW, hw\\_fclk\\_cal\\_s*.csv), not used for any value; "
                          "wall-clock = input write to counter read per image, warm-up discarded, median "
                          "with the distribution-free order-statistic 95\\% CI).",
                          *clock_notes(art, vs)])


# --------------------------------------------------------------------------------------------
def a3_cycles(c: Ctx) -> Artifact:
    art = c.art("tab_a3_cycles")
    body = []
    for net in NETS:
        d = a3_net(art, net)
        if not d:
            continue
        body.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{pretty_net(net)}}}}} \\\\")
        for x in d:
            lab = "Total" if x["is_total"] else art.label(tex_escape(x["layer"]), "cycle_model.csv:layer")
            hw = x["hw"].fmt(art, "int") if x["hw"] else art.placeholder(f"A3 {net}: hw_cycles")
            he = x["hw_err"].fmt(art, "pct2") if x["hw_err"] else art.placeholder(f"A3 {net}: hw_err")
            body.append([lab, x["model"].fmt(art, "int"), x["rtl"].fmt(art, "int") if x["rtl"] else "--", hw,
                         x["rtl_err"].fmt(art, "pct2") if x["rtl_err"] else "--", he])
    hdr = [r"& \multicolumn{3}{c}{Cycles} & \multicolumn{2}{c}{Error vs model (\%)} \\",
           r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
           f"Layer & Model & RTL sim & {c.board_label} & RTL & {c.board_label}"]
    notes = ["Per-layer LAYER\\_CYC; Total = TOTAL\\_CYC (includes C\\_START). Sources: cycle\\_model.csv "
             "(model), rtl\\_network.csv (RTL sim, whole-network jobs), hw\\_a2\\_a3\\_cycles.csv "
             "(measured on KV260)."]
    if art.placeholders:
        notes.append("TBD = board data pending.")
    return c.table(art, "@{}lrrrrr@{}", hdr, body, "A3 three-way per-layer cycle agreement.",
                   "tab:cycles", notes=notes)


# --------------------------------------------------------------------------------------------
def a4_util(c: Ctx) -> Artifact:
    art = c.art("tab_a4_util")
    body = []
    for net in NETS:
        d = a3_net(art, net)
        if not d:
            continue
        hw = latest(art.rows("hw_a4_util.csv", hw=True, net=net) or [], lambda r: r["layer"])
        body.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{pretty_net(net)}}}}} \\\\")
        for x in d:
            lab = "Total" if x["is_total"] else art.label(tex_escape(x["layer"]), "cycle_model.csv:layer")
            am = div(x["mac_model"], x["model"], "MAC_ACTIVE/cycles*100")
            am = V(am.value * 100, am.origin)
            if x.get("rtl_mac") is not None and x.get("rtl") is not None:
                ar = div(x["rtl_mac"], x["rtl"], "rtl MAC_ACTIVE/rtl cycles*100")
                ar = V(ar.value * 100, ar.origin).fmt(art, "f2")
            else:
                ar = "--"
            h = hw.get(x["layer"])
            if h and h.get("active_frac"):
                ah = V(fnum(h["active_frac"]) * 100, f"{_rel(Path(h['_file']))}:{h['_line']}:active_frac*100")
                ahs = ah.fmt(art, "f2") + ("" if x["is_total"] else "$^\\ast$")
            else:
                ahs = art.placeholder(f"A4 {net}: hw_a4_util.csv")
            ut = V(x["util_th"].value * 100, x["util_th"].origin + "*100")
            body.append([lab, x["mac_model"].fmt(art, "int"), am.fmt(art, "f2"), ar, ahs, ut.fmt(art, "f2")])
    hdr = [r"& MAC\_ACTIVE & \multicolumn{3}{c}{Array busy fraction (\%)} & Lane util. \\",
           r"\cmidrule(lr){3-5}",
           f"Layer & (model) & Model & RTL sim & {c.board_label} & model (\\%)"]
    return c.table(art, "@{}lrrrrr@{}", hdr, body,
                   "A4 array busy fraction (MAC\\_ACTIVE / cycles) and model lane utilization per layer.",
                   "tab:util",
                   notes=["Array busy fraction = MAC\\_ACTIVE / cycles: the share of cycles with a valid array "
                          "input ($\\sum T\\cdot K$ over cycles; the KV260 value is the hardware MAC\\_ACTIVE "
                          "counter). It is not lane utilization: lane utilization (model) = fraction of the "
                          "array PE lanes doing useful MACs while the array is busy (cycle\\_model.csv "
                          "util\\_theoretical), a model quantity that no hardware counter measures. "
                          "RTL sim per layer from single-layer jobs (rtl\\_cycles.csv), "
                          "total from rtl\\_network.csv. $^\\ast$Per-layer KV260 values use model $T\\cdot K$ over "
                          "measured LAYER\\_CYC (no per-layer MAC counter); the total uses the MAC\\_ACTIVE "
                          "counter (hw\\_a4\\_util.csv)."])


# --------------------------------------------------------------------------------------------
def _best_cpu(cpu: list[dict]) -> dict:
    """{net: (kind, threads, e2e row, compute row, accuracy row)} of the CPU configuration with the
    lowest e2e median over every kind / thread count in hw_cpu_baseline.csv."""
    ok = [r for r in cpu if r.get("status", "ok") == "ok"]
    e2e = latest([r for r in ok if r.get("mode") == "e2e" and r.get("median_us")],
                 lambda r: (r["net"], r["kind"], r["threads"]))
    comp = latest([r for r in ok if r.get("mode") == "compute"], lambda r: (r["net"], r["kind"], r["threads"]))
    acc = latest([r for r in ok if r.get("mode") == "accuracy"], lambda r: (r["net"], r["kind"]))
    out = {}
    for net in NETS:
        cand = [(fnum(r["median_us"]), k, r) for k, r in e2e.items() if k[0] == net]
        if cand:
            _, (_, kind, th), r = min(cand, key=lambda t: t[0])
            out[net] = (kind, th, r, comp.get((net, kind, th)), acc.get((net, kind)))
    return out


def _dpu_lat(art: Artifact) -> dict:
    """{(net, metric): latest hw_dpu_latency.csv row} (metric: dpu_runner / pre / post / end_to_end)."""
    return latest(art.rows("hw_dpu_latency.csv", hw=True) or [], lambda r: (r["net"], r["metric"]))


def _spread(art: Artifact, r: dict | None, hi: str = "max_us", lo: str = "min_us") -> str:
    """max - min of a row (computed from two CSV cells); '--' if a column is absent."""
    if not r or r.get(hi) in ("", None) or r.get(lo) in ("", None):
        return "--"
    return div_sub(at(r, hi), at(r, lo)).fmt(art, "f1")


def div_sub(a: V, b: V) -> V:
    return V(a.value - b.value, f"computed max-min, max={a.origin}, min={b.origin}")


def _best_cpu_rows(art: Artifact, c: Ctx, cpu: list[dict], b3l: dict, fast: dict) -> list:
    """A5 rows: the best CPU baseline per net, its accuracy, and the accelerator speedups against
    it (computed from CSV cells); fast host path = primary accelerator number, the safe path is
    returned as a footnote string. Also ours-vs-DPU ratios."""
    mc = lambda txt: f"\\multicolumn{{2}}{{c}}{{{txt}}}"  # noqa: E731
    if not cpu:
        return [], ""
    best = _best_cpu(cpu)
    dl = _dpu_lat(art)
    rows = [r"\midrule"]
    lab, acc, sp_e, sp_c, vd_e, vd_c = [], [], [], [], [], []
    safe_txt = []
    for net in NETS:
        if net not in best:
            ph = art.placeholder(f"A5 best CPU {net}: hw_cpu_baseline.csv")
            for l in (lab, acc, sp_e, sp_c, vd_e, vd_c):
                l.append(mc(ph))
            continue
        kind, th, re2e, rcomp, racc = best[net]
        lab.append(mc(art.label(f"{CPU_KINDS.get(kind, tex_escape(kind))}, {th}~thr.",
                                f"hw_cpu_baseline.csv:{re2e['_line']}:kind,threads")))
        acc.append(mc(art.cell(racc, "accuracy", "f2") if racc and racc.get("accuracy") else "--"))
        a_fast, a_safe = fast.get((net, "end_to_end")), b3l.get((net, "end_to_end"))
        a_pl = b3l.get((net, "pl_compute")) or fast.get((net, "pl_compute"))
        cpu_e2e = at(re2e, "median_us")
        ph = lambda what: mc(art.placeholder(what))  # noqa: E731
        sp_e.append(mc(div(cpu_e2e, at(a_fast, "median_us"), "best CPU e2e / accel e2e (fast)").fmt(art, "f2")
                       + "$\\times$") if a_fast else ph(f"A5 {net}: accelerator e2e fast"))
        sp_c.append(mc(div(at(rcomp, "median_us"), at(a_pl, "median_us"), "best CPU compute / PL compute").fmt(art, "f2")
                       + "$\\times$") if (rcomp and a_pl) else ph(f"A5 {net}: compute speedup"))
        if a_safe:
            safe_txt.append(f"{pretty_net(net)} "
                            + div(cpu_e2e, at(a_safe, "median_us"), "best CPU e2e / accel e2e (safe)").fmt(art, "f2")
                            + "$\\times$")
        d_e, d_r = dl.get((net, "end_to_end")), dl.get((net, "dpu_runner"))
        vd_e.append(mc(div(at(d_e, "p50"), at(a_fast, "median_us"), "DPU e2e p50 / accel e2e (fast)").fmt(art, "f2")
                       + "$\\times$") if (d_e and a_fast) else ph(f"A5 {net}: DPU e2e"))
        vd_c.append(mc(div(at(d_r, "p50"), at(a_pl, "median_us"), "DPU runner p50 / PL compute").fmt(art, "f2")
                       + "$\\times$") if (d_r and a_pl) else ph(f"A5 {net}: DPU compute"))
    rows += [["Best CPU baseline", "", *lab], ["\\quad accuracy (\\%)", "", *acc],
             ["Speedup vs best CPU, e2e", "", *sp_e],
             ["Speedup vs best CPU, comp.\\ (PL counter)", "", *sp_c],
             r"\midrule",
             ["DPU time / accelerator time, e2e", "", *vd_e],
             ["DPU time / accelerator time, comp.", "", *vd_c]]
    return rows, "; ".join(safe_txt)


def a5_cpu(c: Ctx) -> Artifact:
    art = c.art("tab_a5_cpu")
    cpu = art.rows("hw_cpu_baseline.csv", hw=True) or []
    b3 = art.rows("hw_b3_breakdown.csv", hw=True) or []
    b3l = latest(b3, lambda r: (r["net"], r["phase"]))
    body = []
    # ORT rows are the primary CPU baselines; the plain numpy rows are unoptimized reference code (footnote)
    ORT_FIRST = ["cpu_ort_int8", "cpu_ort_fp32", "cpu_fp32_numpy", "cpu_int8_ref"]
    kinds = sorted({r["kind"] for r in cpu}, key=lambda k: ORT_FIRST.index(k) if k in ORT_FIRST else 99)
    if cpu:
        cl = latest([r for r in cpu if r.get("mode") in ("compute", "e2e") and r.get("status", "ok") == "ok"],
                    lambda r: (r["net"], r["kind"], r["threads"], r["mode"]))
        for k in kinds:
            if k in NUMPY_KINDS and k == next(x for x in kinds if x in NUMPY_KINDS):
                body.append(r"\midrule")
            for th in sorted({r["threads"] for r in cpu if r["kind"] == k}, key=lambda t: inum(t)):
                cells = []
                for net in NETS:
                    for mode in ("compute", "e2e"):
                        r = cl.get((net, k, th, mode))
                        cells.append(ci(art, r, "median_us", "median_ci_lo_us", "median_ci_hi_us", "f1")
                                     if r else "--")
                trow = next(r for r in cpu if r["kind"] == k and r["threads"] == th)
                mark = NUMPY_MARK if k in NUMPY_KINDS else ""
                body.append([CPU_KINDS.get(k, tex_escape(k)) + mark, art.cell(trow, "threads", "int")] + cells)
    else:
        for k, lab in CPU_KINDS.items():
            body.append(f"{lab} & -- & \\multicolumn{{4}}{{c}}"
                        f"{{{art.placeholder(f'A5 {k}: hw_cpu_baseline.csv (cpu_board)')}}} \\\\")
    fast = latest(art.rows("hw_b3_breakdown_fast.csv", hw=True) or [], lambda r: (r["net"], r["phase"]))
    fp = [f"Accelerator, fast host path ({c.board_label})", "--"]
    for net in NETS:
        for ph in ("pl_compute", "end_to_end"):
            r = fast.get((net, ph)) or b3l.get((net, ph))
            fp.append(ci(art, r, "median_us", "median_ci_lo_us", "median_ci_hi_us", "f1") if r
                      else art.placeholder(f"A5 FPGA {net} {ph}"))
    dl = _dpu_lat(art)
    dp = [f"DPU (Vitis AI, vai\\_q INT8), p50 ({SINGLE})", "--"]
    for net in NETS:
        for met in ("dpu_runner", "end_to_end"):
            r = dl.get((net, met))
            dp.append(art.cell(r, "p50", "f1") if r else art.placeholder(f"A5 DPU {net} {met}"))
    body.append(r"\midrule")
    body.append(fp)
    body.append(dp)
    extra_rows, safe_note = _best_cpu_rows(art, c, cpu, b3l, fast)
    body += extra_rows
    hdr = [r"& & \multicolumn{2}{c}{LeNet-5 (\textmu s)} & \multicolumn{2}{c}{CIFAR-10 (\textmu s)} \\",
           r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
           r"Implementation & thr. & comp. & e2e & comp. & e2e"]
    return c.table(art, "@{}lrrrrr@{}", hdr, body,
                   "A5 same-board baselines: "
                   + (f"median over {_sessw(c)} board sessions [min, max]" if _nsess(c) > 1 else "median [95\\% CI]")
                   + " time per image on the KV260 Cortex-A53, the accelerator (fast host path) and the DPU.",
                   "tab:cpu",
                   notes=[_sess_note(c) + "CPU: hw\\_cpu\\_baseline.csv (source cpu\\_board; median over runs, warm-up "
                          "discarded; workers pinned; p50/p95/p99 in Table~\\ref{tab:pct}). Accelerator: hw\\_b3\\_breakdown.csv, comp.\\ = PL "
                          "counter / the pl\\_clk0 PLL read-back (pl\\_compute), e2e = input write + start + "
                          "poll + logit read + PS dequant (end\\_to\\_end, safe host path; the fast host path "
                          "is given in the speedup rows). CPU e2e additionally includes the input "
                          "preprocessing from the stored raw image (the accelerator e2e starts from the "
                          "stored INT8 activation image), so comp.\\ is the like-for-like comparison. "
                          + ("Safe host path (per-word MMIO; footnote): e2e speedup vs best CPU " + safe_note + ". "
                             if safe_note else "")
                          + "The DPU rows are the AMD DPU (DPUCZDX8G, prebuilt pynq-dpu overlay) running a vai\\_q "
                          "quantization of the same FP32 nets (hw\\_dpu\\_latency.csv, one session only; comp. = VART runner "
                          "time, e2e = input conversion + runner + argmax); DPU/accelerator ratios above "
                          "unity mean the accelerator is faster. "
                          "\\emph{Best CPU baseline} = the CPU configuration with the lowest e2e median per "
                          "net over all kinds and thread counts of the table; every speedup is computed "
                          "against it only (a value below unity means the CPU is faster). ORT INT8 is "
                          "onnxruntime's own static quantization, not the project INT8 numerics "
                          "(accuracy row). "
                          + NUMPY_MARK + "Plain numpy implementations (the INT8 one is the bit-exact reference), not "
                          "optimized CPU runtimes; they are listed for completeness and are not the basis of any "
                          "speedup. Their brackets are the minimum and maximum over the "
                          "sessions as for every row, and are wide for several of them (a four-thread numpy run "
                          "in one session was far slower than in the others), so read them as indicative only."])


# --------------------------------------------------------------------------------------------
def _power_label(art: Artifact, entries: list[dict]) -> str:
    labs = sorted({e["mean"].get("measurement") or INA_LABEL for e in entries}) or [INA_LABEL]
    return " / ".join(art.label(tex_escape(l), "power summary CSV: measurement") for l in labs)


B1_LINES = [   # (label, summary column, fmt, scale); energy/duty columns written by power_log.py
    ("$P_\\mathrm{idle}$ (W)", "accel_p_idle_w", "f3", 1.0),
    ("$\\Delta P$ accelerator (W)", "accel_dp_w", "f3", 1.0),
    ("$\\Delta P$ control (W)", "control_dp_w", "f3", 1.0),
    ("$\\Delta P$ accel.\\ $-$ control (W)", "accel_dp_net_w", "f3", 1.0),
    ("$\\Delta P$ CPU (W)", "cpu_dp_w", "f3", 1.0),
    ("Time/image accel.\\ (ms)", "accel_time_per_image_s", "f3", 1000.0),
    ("Time/image control (ms)", "control_time_per_image_s", "f3", 1000.0),
    ("Time/image CPU (ms)", "cpu_time_per_image_s", "f3", 1000.0),
    ("$t_\\mathrm{PL}$ = TOTAL\\_CYC$/f$ (\\textmu s)", "accel_t_pl_s", "f3", 1e6),
    ("Duty cycle $t_\\mathrm{PL}/$time/image (\\%)", "accel_duty", "f3", 100.0),
    ("$E_\\mathrm{sys}$ accel.\\ (mJ)", "accel_energy_per_image_mj", "f3", 1.0),
    ("$E_\\mathrm{comp}$ accel.\\ (mJ)", "accel_e_comp_mj", "f4", 1.0),
    ("$E_\\mathrm{sys}$ accel.\\ $-$ control (mJ)", "accel_e_sys_net_mj", "f3", 1.0),
    ("$E_\\mathrm{comp}$ accel.\\ $-$ control (mJ)", "accel_e_comp_net_mj", "f4", 1.0),
    ("Energy/image CPU (mJ)", "cpu_energy_per_image_mj", "f3", 1.0),
]


def b1_power(c: Ctx) -> Artifact:
    """B1: SOM-rail power (INA260) protocol summary, mean +- std over repeats (power_log.py):
    dP accel / control / CPU, E_sys and E_comp (+ control-subtracted variants), duty cycle."""
    art = c.art("tab_b1_power")
    ents = power_summaries(art, "hw_b1_power_ina260")
    by = {}
    for e in ents:                                    # one entry per net (latest run if several clocks)
        if e["net"] not in by or e["mean"]["timestamp"] > by[e["net"]]["mean"]["timestamp"]:
            by[e["net"]] = e
    nets = list(NETS)
    body = []
    lab = _power_label(art, list(by.values())) if by else INA_LABEL

    def line(text, fn):
        body.append([text] + [fn(by[n]) if n in by else art.placeholder(f"B1 {n}: INA260 summary")
                              for n in nets])
    line("Clock read back (MHz)", lambda e: art.cell(e["mean"], "clock_mhz", "f1") if e["mean"].get("clock_mhz") else "--")
    line("Host path", lambda e: art.label(tex_escape(e["mean"].get("host_path") or "--"), f"{e['file']}:host_path"))
    for text, col, fmt, scale in B1_LINES:
        line(text, lambda e, col=col, fmt=fmt, scale=scale: pm(art, e, col, fmt, scale))
    line("Repeats", lambda e: art.cell(e["mean"], "n_repeats", "int"))
    line("Sample rate achieved (Hz)", lambda e: art.cell(e["mean"], "rate_achieved_hz", "f1"))
    line("CPU workload", lambda e: art.label(tex_escape(e["mean"].get("cpu_workload") or "--"),
                                              f"{e['file']}:cpu_workload"))
    if not by:
        art.placeholder("B1: hw_b1_power_ina260_summary*.csv (source=hw)")
    hdr = ["Quantity & " + " & ".join(pretty_net(n) for n in nets)]
    return c.table(art, "@{}l" + "r" * len(nets) + "@{}", hdr, body,
                   f"B1 power and energy per image: {tex_escape(lab) if lab == INA_LABEL else lab}, "
                   + (f"median over {_sessw(c)} board sessions (coefficient of variation CV across sessions)."
                      if _nsess(c) > 1 else "mean $\\pm$ std over repeats."), "tab:power",
                   notes=[_power_sess_note(c) + "Measured on the KV260 by the on-board INA260 on the SOM rail (VCC\\_SOM, the "
                          "SOM input): not accelerator-only power and not board input power; PL I/O (VCCO) rails "
                          "and carrier peripherals are outside it. $P_\\mathrm{idle}$ = mean of the two "
                          "idle phases bracketing each run phase; $\\Delta P = P_\\mathrm{run}-P_\\mathrm{idle}$. "
                          "Control = the same host loop (input write, clear, STATUS poll paced to the accelerator's "
                          "start-to-done time, LOGIT read, PS dequant) with the accelerator not started. "
                          "$E_\\mathrm{sys}=\\Delta P_\\mathrm{accel}\\times$ time/image (host loop included); "
                          "$E_\\mathrm{comp}=\\Delta P_\\mathrm{accel}\\times t_\\mathrm{PL}$, "
                          "$t_\\mathrm{PL}$ = TOTAL\\_CYC (hardware counter) over the pl\\_clk0 PLL read-back "
                          "(clock of record); run phases in a seeded random order "
                          "per repeat; "
                          "``$-$ control'' rows use $\\Delta P_\\mathrm{accel}-\\Delta P_\\mathrm{control}$. "
                          "All computed by power\\_log.py. Source: hw\\_b1\\_power\\_ina260\\_summary*.csv."])


CLK_TOL = 0.5      # MHz: a read-back / nominal pair of the same sweep point (points are >= 50 MHz apart)


def b2_clock(c: Ctx) -> Artifact:
    """B2: per-clock SOM-rail power (INA260) dP and energy/image + PL latency."""
    art = c.art("tab_b2_clock")
    ents = power_summaries(art, "hw_b2_power_ina260")
    clk = [r for r in (art.rows("hw_b2_clock.csv", hw=True) or []) if not r.get("skipped_reason")
           and r.get("clock_readback_mhz")]

    def lat_row(net, f):
        cand = [r for r in clk if r["net"] == net and abs(fnum(r["clock_readback_mhz"]) - f) < CLK_TOL]
        return max(cand, key=lambda r: r["timestamp"]) if cand else None

    keys = {(e["net"], fnum(e["clock"])): e for e in ents if e["clock"]}
    for r in clk:                                     # clocks within CLK_TOL are the same setting
        f = fnum(r["clock_readback_mhz"])
        if not any(n == r["net"] and abs(g - f) < CLK_TOL for n, g in keys):
            keys[(r["net"], f)] = None
    ncol = 5
    body, last_net = [], None
    for (net, f) in sorted(keys, key=lambda k: (NETS.index(k[0]) if k[0] in NETS else 9, k[1])):
        e = keys[(net, f)]
        if net != last_net:
            body.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\emph{{{pretty_net(net)}}}}} \\\\")
            last_net = net
        lr = lat_row(net, f)
        fcell = (art.cell(e["mean"], "clock_mhz", "f1") if e else art.cell(lr, "clock_readback_mhz", "f1"))
        row = [fcell,
               pm(art, e, "accel_dp_w", "f3") if e else art.placeholder(f"B2 {net}: INA260 summary"),
               pm(art, e, "accel_energy_per_image_mj", "f3") if e else art.placeholder(f"B2 {net}: INA260 summary"),
               pm(art, e, "accel_time_per_image_s", "f3", 1000) if e else art.placeholder(f"B2 {net}: INA260 summary"),
               art.cell(lr, "latency_us", "f2") if lr and lr.get("latency_us")
               else art.placeholder(f"B2 {net}: hw_b2_clock.csv latency")]
        body.append(row)
    if not keys:
        body.append(f"\\multicolumn{{{ncol}}}{{c}}{{{art.placeholder('B2: hw_b2_power_ina260_summary*.csv / hw_b2_clock.csv (source=hw)')}}} \\\\")
    lab = _power_label(art, ents) if ents else INA_LABEL
    hdr = [r"Clock & $\Delta P$ & Energy/img & Time/img & PL lat.",
           r"(MHz) & (W) & (mJ) & (ms) & (\textmu s)"]
    return c.table(art, "@{}l" + "r" * (ncol - 1) + "@{}", hdr, body,
                   f"B2 clock sweep: {tex_escape(lab) if lab == INA_LABEL else lab} (mean $\\pm$ std over "
                   "repeats) and PL latency.", "tab:clock",
                   notes=["Measured on the KV260. $\\Delta P$ and energy/image: on-board INA260, SOM rail "
                          "(hw\\_b2\\_power\\_ina260\\_summary*.csv); time/img = host inference loop; PL latency = "
                          "cycle counter at the clock of each point (hw\\_b2\\_clock.csv; $f_\\mathrm{meas}$ where "
                          "calibrated at that clock, else the read-back clock)."])


B3_FILES = {"safe": "hw_b3_breakdown.csv", "fast": "hw_b3_breakdown_fast.csv"}


def b3_breakdown(c: Ctx) -> Artifact:
    """B3: host-side phases per image, safe (per-word MMIO) vs fast (mapped numpy windows) host
    path, median and p95 per net, + end-to-end speedup safe/fast (computed from the medians)."""
    art = c.art("tab_b3_breakdown")
    b = {}
    for hp, name in B3_FILES.items():
        rows = art.rows(name, hw=True) or []
        for k, r in latest(rows, lambda r: (r["net"], r["phase"])).items():
            b[(hp,) + k] = r
    phases = [p for p in B3_PHASES if any(k[2] == p for k in b)] or B3_PHASES
    body = []
    for p in phases:
        cells = [tex_escape(p)]
        for net in NETS:
            for hp in B3_FILES:
                r = b.get((hp, net, p))
                cells += ([ci(art, r, "median_us", "median_ci_lo_us", "median_ci_hi_us", "f1"),
                           art.cell(r, "p95_us", "f1")] if r else
                          [art.placeholder(f"B3 {net} {hp}: {B3_FILES[hp]}")] * 2)
        body.append(cells)
    sp = ["speedup e2e (safe/fast)"]
    for net in NETS:
        rs, rf = b.get(("safe", net, "end_to_end")), b.get(("fast", net, "end_to_end"))
        if rs and rf:
            v = div(at(rs, "median_us"), at(rf, "median_us"), "safe median / fast median")
            sp += [v.fmt(art, "f2") + "$\\times$", "", "", ""]
        else:
            sp += [art.placeholder(f"B3 {net}: speedup needs both host paths"), "", "", ""]
    body.append(r"\midrule")
    body.append(sp)
    hdr = [r"& \multicolumn{4}{c}{LeNet-5 (\textmu s)} & \multicolumn{4}{c}{CIFAR-10 (\textmu s)} \\",
           r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
           r"& \multicolumn{2}{c}{safe} & \multicolumn{2}{c}{fast} & \multicolumn{2}{c}{safe} & \multicolumn{2}{c}{fast} \\",
           "Phase & " + " & ".join([f"{_bracket(c)} & p95"] * 4)]
    return c.table(art, "@{}l" + "r" * 8 + "@{}", hdr, body,
                   "B3 end-to-end breakdown per image (host side), safe vs fast host path.",
                   "tab:breakdown", wide=True,
                   notes=[_sess_note(c) + "Measured on the KV260 (Python host). safe: per-word MMIO (hw\\_b3\\_breakdown.csv); "
                          "fast: PL windows mapped once, input written with one vectorized numpy copy, LOGIT "
                          "read with one vectorized read, tight STATUS poll (hw\\_b3\\_breakdown\\_fast.csv; used "
                          "only after the bring-up fast-path check passed). end\\_to\\_end = input\\_write + "
                          "status\\_clear + start\\_done + logit\\_read + ps\\_dequant; start\\_done = start\\_write "
                          "+ poll; pl\\_compute = PL cycle counter / $f_\\mathrm{meas}$. Safe and fast "
                          "conditions interleaved in randomized blocks in one run (warm-up and block warm-up "
                          "discarded)."])


# --------------------------------------------------------------------------------------------
def verification(c: Ctx) -> Artifact:
    art = c.art("tab_verification")
    rows = art.rows("verification_stats.csv")
    if rows is None:
        ph = art.placeholder("verification_stats.csv absent", "TBD (verification\\_stats.csv not yet produced)")
        return c.table(art, "@{}l@{}", ["Verification statistics"], [[ph]],
                       "Verification statistics (pending).", "tab:verif")
    cols = [k for k in (rows[0].keys() if rows else []) if k not in META_COLS]
    keep_src = any(r.get("source") for r in rows)
    shown = cols[:7]
    if len(cols) > len(shown):
        art.check(f"verification_stats.csv: {len(cols)} data columns, first {len(shown)} shown")
    body = []
    for r in rows:
        cells = []
        for k in shown:
            v = r.get(k, "")
            cells.append(art.label(tex_escape(v), f"{r['_file']}:{r['_line']}:{k}") if v != "" else "--")
        if keep_src:
            cells.append(tex_escape(r.get("source", "")))
        body.append(cells)
    hdr = [" & ".join(tex_escape(k) for k in shown) + (" & source" if keep_src else "")]
    n = len(shown) + keep_src
    return c.table(art, "@{}l" + "r" * (n - 1) + "@{}", hdr, body, "Verification statistics.",
                   "tab:verif", notes=["Source: verification\\_stats.csv."], wide=n > 5)


REPEAT_METRICS = [   # (metric file glob, metric, key filter, label, fmt)
    ("hw_fclk_cal_s*.csv", "f_meas_mhz", {}, "$f_\\mathrm{meas}$ (MHz)", "f3"),
    ("hw_a2_a3_cycles.csv", "wall_us_median", {}, "A2 wall-clock/image, median (\\textmu s)", "f1"),
    ("hw_b3_breakdown.csv", "median_us", {"phase": "end_to_end"}, "B3 end-to-end safe, median (\\textmu s)", "f1"),
    ("hw_b3_breakdown_fast.csv", "median_us", {"phase": "end_to_end"}, "B3 end-to-end fast, median (\\textmu s)", "f1"),
    ("hw_cpu_baseline.csv", "median_us", {"kind": "cpu_ort_int8", "threads": "4", "mode": "e2e"},
     "ORT INT8, four threads, e2e, median (\\textmu s)", "f1"),
    ("hw_cpu_baseline.csv", "p99_us", {"kind": "cpu_ort_int8", "threads": "4", "mode": "e2e"},
     "ORT INT8, four threads, e2e, p99 (\\textmu s)", "f1"),
    ("hw_dpu_latency.csv", "p50", {"metric": "end_to_end"}, "DPU e2e, p50 (\\textmu s)", "f1"),
    ("hw_b1_power_ina260_summary_*.csv", "accel_dp_w", {}, "B1 $\\Delta P$ accel.\\ (W)", "f3"),
    ("hw_b1_power_ina260_summary_*.csv", "accel_energy_per_image_mj", {}, "B1 $E_\\mathrm{sys}$ accel.\\ (mJ)", "f3"),
    ("hw_b1_power_ina260_summary_*.csv", "accel_e_comp_mj", {}, "B1 $E_\\mathrm{comp}$ accel.\\ (mJ)", "f4"),
]


def repeatability(c: Ctx) -> Artifact:
    """3-session repeatability (aggregate_sessions.py -> hw_repeatability.csv): per metric and net
    median [min, max] of the per-session values and the number of sessions. Groups are per
    bitstream + clock (never mixed); the latest-timestamp group per metric/net is shown."""
    import json as _json
    art = c.art("tab_repeatability")
    rows = art.rows("hw_repeatability.csv", hw=True) or []
    body = []
    for pat, metric, filt, label, fmt in REPEAT_METRICS:
        cells = [label]
        for net in NETS:
            cand = []
            for r in rows:
                if r.get("metric_file") != pat or r.get("metric") != metric:
                    continue
                k = _json.loads(r.get("key") or "{}")
                if "net" in k and k["net"] != net:       # f_meas has no net key: same for both
                    continue
                if all(k.get(a) == b for a, b in filt.items()):
                    cand.append(r)
            r = max(cand, key=lambda x: x["timestamp"]) if cand else None
            if r is None:
                cells.append(art.placeholder(f"repeatability {metric} {net}: hw_repeatability.csv"))
                continue
            per = sorted(fnum(v) for v in _json.loads(r["per_session"]).values())
            txt = V(statistics.median(per), f"{row_origin(r, 'per_session')} median of the per-session values").fmt(art, fmt)
            if r.get("min") and r.get("max") and fnum(r["n_sessions"]) > 1:
                txt += " [" + art.cell(r, "min", fmt) + ", " + art.cell(r, "max", fmt) + "]"
            cells.append(txt + " (" + art.cell(r, "n_sessions", "int") + ")")
        body.append(cells)
    return c.table(art, "@{}lrr@{}", ["Metric & LeNet-5 & CIFAR-10"], body,
                   "Repeatability over independent board sessions: median [min, max] of the per-session "
                   "values (number of sessions).", "tab:repeat",
                   notes=["Source: hw\\_repeatability.csv (aggregate\\_sessions.py): per session the "
                          "session's own statistic (latency median, power mean over repeats), combined "
                          "across session\\_index values; rows from different bitstreams or clocks are "
                          "never combined. Measured on the KV260."])


# --------------------------------------------------------------------------------------------
def percentiles(c: Ctx) -> Artifact:
    """p50 / p95 / p99 and spread (max - min) of the per-image time of every measured condition
    (p50 = median): accelerator (fast host path first, safe path below), DPU, CPU baselines."""
    art = c.art("tab_percentiles")
    body = []

    def add(label, rowmap, cols, hi="max_us", lo="min_us"):
        line = [label]
        for net in NETS:
            r = rowmap(net)
            if not r:
                line += [art.placeholder(f"percentiles {label} {net}")] * 4
                continue
            line += [art.cell(r, col, "f1") if r.get(col) else "--" for col in cols] + [_spread(art, r, hi, lo)]
        body.append(line)
    cols = ("p50_us", "p95_us", "p99_us")
    safe = latest(art.rows("hw_b3_breakdown.csv", hw=True) or [], lambda r: (r["net"], r["phase"]))
    fast = latest(art.rows("hw_b3_breakdown_fast.csv", hw=True) or [], lambda r: (r["net"], r["phase"]))
    b3c = latest(art.rows("hw_b3_breakdown_cpu.csv", hw=True) or [], lambda r: (r["net"], r["phase"]))
    a2 = {n: latest(art.rows("hw_a2_a3_cycles.csv", hw=True, net=n) or [], lambda r: r["layer"]).get("total")
          for n in NETS}
    dl = _dpu_lat(art)
    cpu = latest([r for r in (art.rows("hw_cpu_baseline.csv", hw=True) or [])
                  if r.get("mode") == "e2e" and r.get("status", "ok") == "ok"],
                 lambda r: (r["net"], r["kind"], r["threads"]))
    add(f"Accelerator e2e, fast host path ({c.board_label})", lambda n: fast.get((n, "end_to_end")), cols)
    add("DPU e2e (pre + runner + post)", lambda n: dl.get((n, "end_to_end")), ("p50", "p95", "p99"), "max", "min")
    add("DPU runner only", lambda n: dl.get((n, "dpu_runner")), ("p50", "p95", "p99"), "max", "min")
    best = _best_cpu(list(cpu.values())) if cpu else {}
    if best:
        (kind, th) = next(iter(best.values()))[:2]
        r0 = next((r for k, r in cpu.items() if k[1] == kind and k[2] == th), None)
        add(art.label(f"Best CPU: {CPU_KINDS[kind]}, {th}~thr.\\ e2e", f"hw_cpu_baseline.csv:{r0['_line']}:kind,threads")
            if r0 else "Best CPU", lambda n, kind=kind, th=th: cpu.get((n, kind, th)), cols)
    body.append(r"\midrule")
    add(f"Accelerator e2e, safe host path ({c.board_label})", lambda n: safe.get((n, "end_to_end")), cols)
    add("Accelerator wall-clock (A2 run, safe)", lambda n: a2[n], ("wall_us_p50", "wall_us_p95", "wall_us_p99"),
        "wall_us_max", "wall_us_min")
    for kind in CPU_KINDS:
        for th in sorted({k[2] for k in cpu if k[1] == kind}, key=inum):
            r0 = next(r for k, r in cpu.items() if k[1] == kind and k[2] == th)
            add(art.label(f"{CPU_KINDS[kind]}, {th}~thr.\\ e2e", f"hw_cpu_baseline.csv:{r0['_line']}:kind,threads"),
                lambda n, kind=kind, th=th: cpu.get((n, kind, th)), cols)
    rb = next(iter(b3c.values()), None)
    add(art.label(f"CPU INT8 ref., {rb['condition'].split()[-1].lstrip('x').rstrip(')')}~thr.\\ (B3 interleaved)",
                  f"hw_b3_breakdown_cpu.csv:{rb['_line']}:condition") if rb else "CPU (B3 interleaved)",
        lambda n: b3c.get((n, "end_to_end")), cols)
    rs0 = next(iter(fast.values()), None)
    rc0 = next(iter(cpu.values()), None)
    rd0 = next(iter(dl.values()), None)
    hdr = [r"& \multicolumn{4}{c}{LeNet-5 (\textmu s)} & \multicolumn{4}{c}{CIFAR-10 (\textmu s)} \\",
           r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
           r"Condition & p50 & p95 & p99 & spread & p50 & p95 & p99 & spread"]
    return c.table(art, "@{}l" + "r" * 8 + "@{}", hdr, body,
                   "Per-image time percentiles p50 / p95 / p99 and spread (max $-$ min) of every measured condition.",
                   "tab:pct",
                   notes=[_sess_note(c, pct=True) + "p50 = median; spread = slowest minus fastest timed image. The fast host path is the primary "
                          "accelerator number, the safe host path is listed below the rule. Sources: "
                          "hw\\_b3\\_breakdown*.csv (kept images per condition: "
                          + (art.cell(rs0, "images", "int") if rs0 and rs0.get("images") else "--")
                          + ", interleaved), hw\\_a2\\_a3\\_cycles.csv (A2 wall-clock, one image per call), "
                          "hw\\_dpu\\_latency.csv (DPU, images per net: "
                          + (art.cell(rd0, "n", "int") if rd0 and rd0.get("n") else "--") + "), "
                          "hw\\_cpu\\_baseline.csv (timed runs per configuration: "
                          + (art.cell(rc0, "runs", "int") if rc0 and rc0.get("runs") else "--")
                          + "; p99 is then close to the maximum and only indicative)."], wide=True)


# --------------------------------------------------------------------------------------------
def power_compare(c: Ctx) -> Artifact:
    """B1 vs DPU SOM-rail power: absolute power and delta above each system's OWN-overlay idle, energy/image;
    the no-overlay (fresh boot) idle reference as a single row."""
    art = c.art("tab_power_compare")
    ours = {e["net"]: e for e in power_summaries(art, "hw_b1_power_ina260")}
    dpu = {e["net"]: e for e in power_summaries(art, "hw_dpu_power_ina260")}
    ref = latest(art.rows("hw_nooverlay_idle_phases.csv", hw=True) or [], lambda r: "ref").get("ref")
    body = []

    def block(title, ents, cpu_kind=False):
        body.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for text, col, fmt, scale in (("$P_\\mathrm{idle}$, own overlay loaded (W)", "accel_p_idle_w", "f3", 1.0),
                                      ("$P$ running, absolute (W)", "accel_p_run_w", "f3", 1.0),
                                      ("$\\Delta P$ above own idle (W)", "accel_dp_w", "f3", 1.0),
                                      ("Time/image (ms)", "accel_time_per_image_s", "f3", 1000.0),
                                      ("Energy/image (mJ)", "accel_energy_per_image_mj", "f3", 1.0)):
            body.append([text] + [pm(art, ents[n], col, fmt, scale) if n in ents
                                  else art.placeholder(f"power compare {title} {n}") for n in NETS])
    block(f"Accelerator ({c.board_label}, gos build loaded, host loop included)", ours)
    block("DPU (prebuilt pynq-dpu overlay loaded, end-to-end loop)", dpu)
    body.append(r"\midrule")
    body.append(["$P_\\mathrm{idle}$, no overlay (fresh boot) (W)"]
                + [f"\\multicolumn{{2}}{{c}}{{{art.cell(ref, 'power_mean_w', 'f3') if ref and ref.get('power_mean_w') else art.placeholder('no-overlay idle reference: hw_nooverlay_idle_phases.csv')}}}"])
    return c.table(art, "@{}lrr@{}", ["Quantity & LeNet-5 & CIFAR-10"], body,
                   f"SOM-rail power ({INA_LABEL}) of the accelerator and the DPU: absolute power and $\\Delta P$ above "
                   "each system's own-overlay idle.", "tab:powercmp",
                   notes=[_power_sess_note(c) + "The DPU values come from one session (mean $\\pm$ std over its repeats). "
                          "Each $\\Delta P$ is above the idle power measured with that system's own bitstream loaded "
                          "(mean of the idle phases bracketing the run phases); the two idle levels differ because the "
                          "loaded PL configuration differs, so compare absolute power and $\\Delta P$ together. The "
                          "no-overlay reference is the idle SOM-rail power measured "
                          + art.label("15.6", "board journal: boot 2026-10-02 08:12:52 UTC (uptime -s); reference window start "
                                      "08:28:30 UTC (hw_nooverlay_idle_phases.csv start_utc), 600 s")
                          + "~min after boot, before any bitstream was loaded, over a long window; the board had settled "
                          "(no drift across the window). "
                          "$P_\\mathrm{run}$ / time and energy are per image of the measured loop (accelerator: "
                          "host loop of the selected host path; DPU: input conversion + runner + argmax). Source: "
                          "hw\\_b1\\_power\\_ina260\\_summary*.csv, hw\\_dpu\\_power\\_ina260\\_summary*.csv, "
                          "hw\\_nooverlay\\_idle\\_phases.csv. Measured on the KV260."])


# --------------------------------------------------------------------------------------------
def efficiency(c: Ctx) -> Artifact:
    """Throughput and efficiency: images/s, GOPS, GOPS per MAC lane, GOPS per DSP (board_efficiency.csv)."""
    art = c.art("tab_efficiency")
    rows = latest(art.rows("board_efficiency.csv") or [], lambda r: (r["net"], r["system"]))
    params = {r["name"]: r for r in (art.rows("dpu_overlay_params.csv") or [])}
    body = []
    lab = {"ours_pl": "Accelerator, PL counter", "ours_e2e_fast": "Accelerator, e2e fast host path",
           "dpu_runner": "DPU, runner only", "dpu_e2e": "DPU, e2e",
           "cpu_compute": "Best CPU, compute", "cpu_e2e": "Best CPU, e2e"}

    def cell(net, sysname, col, fmt):
        r = rows.get((net, sysname))
        if not r:
            return art.placeholder(f"efficiency {net} {sysname}: board_efficiency.csv")
        return art.cell(r, col, fmt) if r.get(col) else "n/a"
    for sysname, text in lab.items():
        for colname, title, fmt in (("images_per_s", "images/s", "f0"), ("gops", "GOPS", "f2"),
                                    ("gops_per_mac_lane", "GOPS / MAC lane", "f4"), ("gops_per_dsp", "GOPS / DSP", "f3"),
                                    ("images_per_s_per_dsp", "images/s / DSP", "f1")):
            if sysname.startswith("cpu") and colname not in ("images_per_s", "gops"):
                continue
            if sysname.startswith("dpu") and colname.endswith("_dsp"):
                continue                                   # no DPU per-DSP figure (no resource counts)
            body.append([f"{text}: {title}"] + [cell(n, sysname, colname, fmt) for n in NETS])
    body.append(r"\midrule")
    body.append(["MAC lanes (ours / DPU)"] + [f"{cell(n, 'ours_pl', 'mac_lanes', 'int')} / {cell(n, 'dpu_runner', 'mac_lanes', 'int')}"
                                              for n in NETS])
    body.append(["Lane clock, MHz (ours / DPU)"] + [f"{cell(n, 'ours_pl', 'clock_mhz', 'f1')} / {cell(n, 'dpu_runner', 'clock_mhz', 'f1')}"
                                                   for n in NETS])
    body.append(["Peak GOPS (ours / DPU)"] + [f"{cell(n, 'ours_pl', 'peak_gops', 'f1')} / {cell(n, 'dpu_runner', 'peak_gops', 'f1')}"
                                             for n in NETS])
    body.append(["PL / runner time as \\% of peak (ours / DPU)"] + [
        f"{cell(n, 'ours_pl', 'pct_of_peak', 'f1')} / {cell(n, 'dpu_runner', 'pct_of_peak', 'f2')}" for n in NETS])
    hw = params.get("mac_lanes")
    prow = params
    two = art.label("2", "formula: 1 MAC = 2 ops (board_efficiency.csv formula column)")
    hund = art.label("100", "formula: percent (board_efficiency.csv formula column)")
    aclk_mhz = (art.label(prow["aclk"]["value"], f"dpu_overlay_params.csv:{prow['aclk']['_line']}:value")
                if prow.get("aclk") else "?")
    ap2_mhz = (art.label(prow["ap_clk_2"]["value"], f"dpu_overlay_params.csv:{prow['ap_clk_2']['_line']}:value")
               if prow.get("ap_clk_2") else "?")
    return c.table(art, "@{}lrr@{}", ["Quantity & LeNet-5 & CIFAR-10"], body,
                   "Throughput and efficiency per image: accelerator vs DPU (vs best CPU).", "tab:eff",
                   notes=["GOPS = twice the MACs per image (cycle\\_model.csv) divided by the time, the same operation count for "
                          "every system (same network architecture; the DPU runs the vai\\_q quantization of the "
                          "FP32 net). MAC lanes: accelerator eight-by-eight PEs (one DSP48E2 MAC each), DPU = "
                          "ARCH\\_PP $\\times$ ARCH\\_ICP $\\times$ ARCH\\_OCP from the overlay's "
                          + (art.label(tex_escape(hw["hwh_file"].split("/")[-1]), f"dpu_overlay_params.csv:{hw['_line']}:hwh_file")
                             if hw else "hwh")
                          + " (dpu\\_overlay\\_params.csv). Formulas: ops/image $=" + two + "\\times$MACs; GOPS $=$ ops/image "
                          "$/$ time; GOPS per MAC lane $=$ GOPS $/$ MAC lanes; peak GOPS $=" + two + "\\times$ MAC lanes $\\times$ "
                          "lane clock; \\% of peak $=" + hund + "\\times$ GOPS $/$ peak GOPS (PL / runner time rows). Lane clock: "
                          "accelerator = pl\\_clk0 PLL read-back of the A2 run (hw\\_a2\\_a3\\_cycles.csv); DPU = "
                          "\\texttt{aclk} of the DPU module, " + aclk_mhz + "~MHz, from "
                          + (art.label(tex_escape(prow["aclk"]["hwh_file"].split("/")[-1]), f"dpu_overlay_params.csv:{prow['aclk']['_line']}:hwh_file")
                             if prow.get("aclk") else "hwh")
                          + " (PORT CLKFREQUENCY; the " + ap2_mhz + "~MHz ap\\_clk\\_2 is the DSP double-rate clock, not the lane clock). "
                          "DSP: accelerator only (impl\\_gos.csv, post-implementation); there is no DPU per-DSP figure "
                          "because the overlay's hwh and xclbin carry no resource counts. A MAC lane is not a DSP. "
                          "Source: board\\_efficiency.csv (derived from committed board results)."])


def _sess_note(c: Ctx, pct: bool = False) -> str:
    """Note sentence for tables whose board cells are session medians (empty for a single session)."""
    n = _nsess(c)
    if n < 2:
        return ""
    what = "each percentile" if pct else "each cell"
    return (f"Board cells of the accelerator and CPU rows: {what} is the median over the {_sessw(c)} independent board "
            "sessions of that session's own statistic"
            + (" (the spread is the median over the sessions of the slowest image minus the median of the fastest)"
               if pct else "; the bracket is the minimum and maximum over the sessions")
            + ". The DPU, A1, A2 wall-clock, A4, B2, soak, layer-spread and shape rows were measured in one session only. ")


def _power_sess_note(c: Ctx) -> str:
    n = _nsess(c)
    if n < 2:
        return ""
    return (f"Accelerator cells: median over the {_sessw(c)} board sessions of each session's repeat-mean; "
            "CV = standard deviation / mean of those per-session values. ")


ALL = [t1_impl, a1_accuracy, a2_latency, a3_cycles, a4_util, a5_cpu, percentiles, efficiency, b1_power, power_compare,
       b2_clock, b3_breakdown,
       repeatability, verification]
