"""Table generators. Each returns an Artifact after writing generated/<name>.tex."""
from __future__ import annotations

from pathlib import Path

from paperdata import INA_LABEL, V, a3_net, at, ci, div, impl_variants, latest, lines_of, pm, power_summaries
from paperlib import (NETS, PH_BOARD, Artifact, Store, _rel, fnum, inum, latex_table, pretty_net,
                      tex_escape, write_text)

META_COLS = {"timestamp", "git_commit", "git_dirty", "vivado_version", "bitstream_sha256", "board_id",
             "net", "layer", "clock_mhz", "source", "duration_s", "num_inferences", "_file", "_line"}

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
    head = "Quantity & " + " & ".join(f"{_mhz(art, r)}~MHz" for r in vs)
    grp = lambda t: f"\\multicolumn{{{n + 1}}}{{@{{}}l}}{{\\emph{{{t}}}}} \\\\"
    row = lambda lab, col, fmt="int": [lab] + [art.cell(r, col, fmt) for r in vs]
    body = [grp("Full design (gos\\_top + AXI shell)"),
            row("CLB LUT", "clb_luts"), row("CLB FF", "clb_registers"), row("LUT as memory", "lut_as_memory"),
            row("DSP48E2", "dsps"), row("RAMB36", "ramb36"), row("RAMB18", "ramb18"),
            grp("gos\\_core only"),
            row("CLB LUT", "core_luts"), row("CLB FF", "core_registers"), row("DSP48E2", "core_dsps"),
            row("RAMB36", "core_ramb36"), row("RAMB18", "core_ramb18"),
            grp("Timing and build"),
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
                   "xck26-sfvc784-2LV-c). One column per implemented clock that met timing.",
                   "tab:impl",
                   notes=["Source: impl\\_gos.csv (post\\_impl). Resources are Vivado utilization "
                          "report counts; no percentages (device totals not in a results CSV)."])


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
        lab = f"Latency (\\textmu s) @ {_mhz(art, r)}~MHz$^\\dagger$"
        line(lab, lambda n, f=f: div(tot[n]["rtl"], f, "cycles/MHz = us").fmt(art, "f2")
             if tot[n] and tot[n]["rtl"] else "--")
    hw = {n: latest(art.rows("hw_a2_a3_cycles.csv", hw=True, net=n) or [], lambda r: r["layer"]).get("total")
          for n in NETS}
    line(f"PL latency (\\textmu s) at $f_\\mathrm{{meas}}$, {c.board_label}", lambda n: art.cell(hw[n], "hw_us", "f2")
         if hw[n] and hw[n].get("hw_us") else art.placeholder(f"A2 {n}: hw_us"))
    line(f"$f_\\mathrm{{meas}}$ (MHz), {c.board_label}", lambda n: art.cell(hw[n], "f_meas_mhz", "f3")
         if hw[n] and hw[n].get("f_meas_mhz") else art.placeholder(f"A2 {n}: f_meas_mhz (exp_fclk_cal)"))
    line(f"Clock read back (MHz), {c.board_label}", lambda n: art.cell(hw[n], "clock_mhz", "f3")
         if hw[n] and hw[n].get("clock_mhz") else art.placeholder(f"A2 {n}: clock_mhz"))
    line(f"Wall-clock/image (\\textmu s), median [95\\% CI], {c.board_label}",
         lambda n: ci(art, hw[n], "wall_us_median", "wall_us_ci_lo", "wall_us_ci_hi", "f1")
         if hw[n] and hw[n].get("wall_us_median") else art.placeholder(f"A2 {n}: wall_us_median"))
    return c.table(art, "@{}lrr@{}", ["& LeNet-5 & CIFAR-10"], rows,
                   "A2 latency per image (single-image jobs).", "tab:latency",
                   notes=["$^\\dagger$RTL sim cycles at post-impl clock (cycles / pl\\_clk0 actual from "
                          "impl\\_gos.csv), computed, not measured. Sources: cycle\\_model.csv (model), "
                          "rtl\\_network.csv (RTL sim), hw\\_a2\\_a3\\_cycles.csv (measured on KV260; PL "
                          "latency = cycle counter / $f_\\mathrm{meas}$, the pl\\_clk0 measured by the "
                          "calibration (cycle counter vs CLOCK\\_MONOTONIC\\_RAW, hw\\_fclk\\_cal\\_s*.csv); "
                          "wall-clock = input write to counter read per image, warm-up discarded, median "
                          "with the distribution-free order-statistic 95\\% CI)."])


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
    hdr = [r"& MAC\_ACTIVE & \multicolumn{3}{c}{MAC\_ACTIVE / cycles (\%)} & PE util. \\",
           r"\cmidrule(lr){3-5}",
           f"Layer & (model) & Model & RTL sim & {c.board_label} & theor.\\ (\\%)"]
    return c.table(art, "@{}lrrrrr@{}", hdr, body, "A4 array utilization per layer.", "tab:util",
                   notes=["MAC\\_ACTIVE = cycles with a valid array input ($\\sum T\\cdot K$). PE util.\\ "
                          "theor.\\ = fraction of the array PEs doing useful MACs (cycle\\_model.csv "
                          "util\\_theoretical). RTL sim per layer from single-layer jobs (rtl\\_cycles.csv), "
                          "total from rtl\\_network.csv. $^\\ast$Per-layer KV260 values use model $T\\cdot K$ over "
                          "measured LAYER\\_CYC (no per-layer MAC counter); the total uses the MAC\\_ACTIVE "
                          "counter (hw\\_a4\\_util.csv)."])


# --------------------------------------------------------------------------------------------
def a5_cpu(c: Ctx) -> Artifact:
    art = c.art("tab_a5_cpu")
    cpu = art.rows("hw_cpu_baseline.csv", hw=True) or []
    b3 = art.rows("hw_b3_breakdown.csv", hw=True) or []
    b3l = latest(b3, lambda r: (r["net"], r["phase"]))
    body = []
    kinds = sorted({r["kind"] for r in cpu}, key=lambda k: list(CPU_KINDS).index(k) if k in CPU_KINDS else 99)
    if cpu:
        cl = latest([r for r in cpu if r.get("mode") in ("compute", "e2e") and r.get("status", "ok") == "ok"],
                    lambda r: (r["net"], r["kind"], r["threads"], r["mode"]))
        for k in kinds:
            for th in sorted({r["threads"] for r in cpu if r["kind"] == k}, key=lambda t: inum(t)):
                cells = []
                for net in NETS:
                    for mode in ("compute", "e2e"):
                        r = cl.get((net, k, th, mode))
                        cells.append(ci(art, r, "median_us", "median_ci_lo_us", "median_ci_hi_us", "f1")
                                     if r else "--")
                trow = next(r for r in cpu if r["kind"] == k and r["threads"] == th)
                body.append([CPU_KINDS.get(k, tex_escape(k)), art.cell(trow, "threads", "int")] + cells)
    else:
        for k, lab in CPU_KINDS.items():
            body.append(f"{lab} & -- & \\multicolumn{{4}}{{c}}"
                        f"{{{art.placeholder(f'A5 {k}: hw_cpu_baseline.csv (cpu_board)')}}} \\\\")
    fp = [f"Accelerator ({c.board_label} PL)", "--"]
    for net in NETS:
        for ph in ("pl_compute", "end_to_end"):
            r = b3l.get((net, ph))
            fp.append(ci(art, r, "median_us", "median_ci_lo_us", "median_ci_hi_us", "f1") if r
                      else art.placeholder(f"A5 FPGA {net} {ph}"))
    body.append(r"\midrule")
    body.append(fp)
    hdr = [r"& & \multicolumn{2}{c}{LeNet-5 (\textmu s)} & \multicolumn{2}{c}{CIFAR-10 (\textmu s)} \\",
           r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
           r"Implementation & thr. & comp. & e2e & comp. & e2e"]
    return c.table(art, "@{}lrrrrr@{}", hdr, body,
                   "A5 same-board baseline: median [95\\% CI] time per image on the KV260 Cortex-A53 vs "
                   "the accelerator.",
                   "tab:cpu",
                   notes=["CPU: hw\\_cpu\\_baseline.csv (source cpu\\_board; median over runs, warm-up "
                          "discarded, distribution-free order-statistic 95\\% CI; workers pinned). "
                          "Accelerator: hw\\_b3\\_breakdown.csv, comp.\\ = PL counter / $f_\\mathrm{meas}$ "
                          "(pl\\_compute), e2e = input write + start + poll + logit read + PS dequant "
                          "(end\\_to\\_end)."])


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
                   "mean $\\pm$ std over repeats.", "tab:power",
                   notes=["Measured on the KV260 by the on-board INA260 on the SOM rail (VCC\\_SOM, the "
                          "SOM input): not accelerator-only power and not board input power; PL I/O (VCCO) rails "
                          "and carrier peripherals are outside it. $P_\\mathrm{idle}$ = mean of the two "
                          "idle phases bracketing each run phase; $\\Delta P = P_\\mathrm{run}-P_\\mathrm{idle}$. "
                          "Control = the same host loop (input write, clear, STATUS poll paced to the accelerator's "
                          "start-to-done time, LOGIT read, PS dequant) with the accelerator not started. "
                          "$E_\\mathrm{sys}=\\Delta P_\\mathrm{accel}\\times$ time/image (host loop included); "
                          "$E_\\mathrm{comp}=\\Delta P_\\mathrm{accel}\\times t_\\mathrm{PL}$, "
                          "$t_\\mathrm{PL}$ = TOTAL\\_CYC (hardware counter) over $f_\\mathrm{meas}$ (measured "
                          "pl\\_clk0; read-back clock if no calibration); run phases in a seeded random order "
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
           r"Phase & median [95\% CI] & p95 & median [95\% CI] & p95 & median [95\% CI] & p95 & median [95\% CI] & p95"]
    return c.table(art, "@{}l" + "r" * 8 + "@{}", hdr, body,
                   "B3 end-to-end breakdown per image (host side), safe vs fast host path.",
                   "tab:breakdown", wide=True,
                   notes=["Measured on the KV260 (Python host). safe: per-word MMIO (hw\\_b3\\_breakdown.csv); "
                          "fast: PL windows mapped once, input written with one vectorized numpy copy, LOGIT "
                          "read with one vectorized read, tight STATUS poll (hw\\_b3\\_breakdown\\_fast.csv; used "
                          "only after the bring-up fast-path check passed). end\\_to\\_end = input\\_write + "
                          "status\\_clear + start\\_done + logit\\_read + ps\\_dequant; start\\_done = start\\_write "
                          "+ poll; pl\\_compute = PL cycle counter / $f_\\mathrm{meas}$. Safe and fast "
                          "conditions interleaved in randomized blocks in one run (warm-up and block warm-up "
                          "discarded); median with the distribution-free order-statistic 95\\% CI."])


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
    ("hw_b1_power_ina260_summary_*.csv", "accel_dp_w", {}, "B1 $\\Delta P$ accel.\\ (W)", "f3"),
    ("hw_b1_power_ina260_summary_*.csv", "accel_energy_per_image_mj", {}, "B1 $E_\\mathrm{sys}$ accel.\\ (mJ)", "f3"),
    ("hw_b1_power_ina260_summary_*.csv", "accel_e_comp_mj", {}, "B1 $E_\\mathrm{comp}$ accel.\\ (mJ)", "f4"),
]


def repeatability(c: Ctx) -> Artifact:
    """3-session repeatability (aggregate_sessions.py -> hw_repeatability.csv): per metric and net
    mean of the per-session values +- between-session SD and the number of sessions. Groups are per
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
            txt = art.cell(r, "mean", fmt)
            if r.get("between_sd"):
                txt += r"\,$\pm$\," + art.cell(r, "between_sd", fmt)
            cells.append(txt + " (" + art.cell(r, "n_sessions", "int") + ")")
        body.append(cells)
    return c.table(art, "@{}lrr@{}", ["Metric & LeNet-5 & CIFAR-10"], body,
                   "Repeatability over independent board sessions: mean $\\pm$ between-session SD "
                   "(number of sessions).", "tab:repeat",
                   notes=["Source: hw\\_repeatability.csv (aggregate\\_sessions.py): per session the "
                          "session's own statistic (latency median, power mean over repeats), combined "
                          "across session\\_index values; rows from different bitstreams or clocks are "
                          "never combined. Measured on the KV260."])


ALL = [t1_impl, a1_accuracy, a2_latency, a3_cycles, a4_util, a5_cpu, b1_power, b2_clock, b3_breakdown,
       repeatability, verification]
