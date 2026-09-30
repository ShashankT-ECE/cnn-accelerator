"""Extra paper artifacts for the publication board experiments (soak, B2 power-vs-clock fit,
per-layer cycle spread). Called by make_all.py (guarded import):

    import tables_extra
    arts = tables_extra.make(ctx[, failures])   # ctx: anything with .store, .out, .dryrun (.booktabs)

Same honesty machinery as tables.py / figures.py (paperlib): every number comes from a CSV row and
is registered with its origin (Artifact.num / cell / label); board rows only with source=hw (dry
run: v2/results/dryrun/, watermarked); missing board data -> "TBD (board)" placeholders and a
hatched "board data pending" figure. Nothing is typed.

Artifacts:
  tab_soak          hw_soak.csv (row_kind net / all): duration, jobs, throughput, mismatches, errors,
                    distinct TOTAL_CYC values, die temperature, PASS/FAIL
  tab_b2_fit        hw_b2_fit.csv (basis per_clock_mean): P = P_static + k*f for P_accel, dP_accel,
                    P_idle; P_static and k with SE and 95 % CI, R^2, n  ("SOM-rail power (INA260)")
  fig_b2_fit        P vs f: per-clock means +- std (hw_b2_power_ina260_summary*.csv) and the fitted
                    lines from hw_b2_fit.csv, (a) absolute P_accel / P_idle, (b) dP_accel
  tab_layer_spread  hw_layer_spread.csv per net x layer: model cycles, KV260 min / max / distinct /
                    spread over all images; RTL-sim reference = rtl_full10k.csv cycles_exact/images
  tab_shapes        A3-general random shapes: per source (RTL sim = shapes_rtl.csv, KV260 = hw_shapes.csv):
                    jobs (refuse jobs), outputs bit-exact, cycle-exact, refusals correct, all exact,
                    max |measured - model| cycles (TOTAL / per layer)
  fig_shapes_cycles predicted (model) vs measured cycles per random-shape job, log-log with y = x:
                    RTL-sim TOTAL_CYC (+ per-layer LAYER_CYC) from shapes_rtl.csv, KV260 TOTAL_CYC from
                    hw_shapes.csv (source=hw) when present, else a "board data pending" note
"""
from __future__ import annotations

import traceback
from pathlib import Path

import figures as F
import tables as T
from paperdata import INA_LABEL, PERFORMANCE_CLOCK_MHZ, latest, power_summaries
from paperlib import NETS, Artifact, _rel, fnum, pretty_net, tex_escape

FIT_Q = [("p_accel_w", r"$P_\mathrm{accel}$ (absolute)"), ("dp_accel_w", r"$\Delta P_\mathrm{accel}$"),
         ("p_idle_w", r"$P_\mathrm{idle}$")]
FIT_BASIS = "per_clock_mean"


def _ctxs(ctx):
    tc = ctx if isinstance(ctx, T.Ctx) else T.Ctx(ctx.store, ctx.out, booktabs=getattr(ctx, "booktabs", True),
                                                   dryrun=ctx.dryrun)
    fc = F.FCtx(ctx.store, ctx.out, dryrun=ctx.dryrun)
    return tc, fc


def _latest_run(rows: list[dict]) -> list[dict]:
    """Rows of the most recent run (same timestamp minute as the newest row) -- reruns append."""
    if not rows:
        return rows
    t = max(r["timestamp"] for r in rows)[:16]
    return [r for r in rows if r["timestamp"][:16] == t]


# --------------------------------------------------------------------------------------------
def soak_table(c: T.Ctx) -> Artifact:
    art = c.art("tab_soak")
    rows = _latest_run(art.rows("hw_soak.csv", hw=True) or [])
    per = {r["net"]: r for r in rows if r.get("row_kind") == "net"}
    allr = next((r for r in rows if r.get("row_kind") == "all"), None)
    body = []
    cols = ["duration_s", "images", "inf_per_s", "logit_mismatch", "cycle_mismatch", "job_errors",
            "timeouts", "status_errors", "err_flags", "cyc_distinct", "temp_c_max", "result"]
    labels = ["Duration (s)", "Jobs", "Throughput (inf/s)", "Logit mismatches", "Cycle mismatches",
              "Job errors (refused)", "Timeouts", "STATUS errors", "Error flags (PS\\_BUSY\\_VIOLATION)",
              "Distinct TOTAL\\_CYC values", "Die temperature max ($^\\circ$C)", "Result"]
    fmt = {"duration_s": "f1", "inf_per_s": "f1", "temp_c_max": "f1", "result": "raw"}
    order = [n for n in NETS if n in per]
    if not rows:
        ph = art.placeholder("soak: hw_soak.csv (source=hw)")
        for lab in labels:
            body.append([lab, ph, ph, ph])
        head = "Quantity & " + " & ".join(pretty_net(n) for n in NETS) + " & All"
        return c.table(art, "@{}lrrr@{}", [head], body, f"Soak test on the {c.board_label}.", "tab:soak")
    keys = order + (["all"] if allr else [])
    src = {**per, **({"all": allr} if allr else {})}
    for col, lab in zip(cols, labels):
        cells = []
        for k in keys:
            r = src[k]
            v = r.get(col, "")
            if v in ("", None):
                cells.append("--")
            elif col == "cyc_distinct" and not str(v).isdigit():
                cells.append(art.label(tex_escape(v), f"{r['_file']}:{r['_line']}:{col}"))
            else:
                cells.append(art.cell(r, col, fmt.get(col, "int")))
        body.append([lab] + cells)
    for k in keys:
        if src[k].get("result") != "PASS":
            art.check(f"hw_soak.csv line {src[k]['_line']} ({k}): result {src[k].get('result')}")
    head = "Quantity & " + " & ".join(pretty_net(k) if k != "all" else "All" for k in keys)
    clk = next(iter(src.values()))
    cap = (f"Soak test on the {c.board_label}: back-to-back inference at "
           f"{art.cell(clk, 'clock_mhz', 'f1')}~MHz, every job checked bit-exact (logits) and "
           "cycle-exact (LAYER\\_CYC, TOTAL\\_CYC, MAC\\_ACTIVE) against the model.")
    notes = ["Source: hw\\_soak.csv (per-minute throughput in hw\\_soak\\_minutes.csv). Die temperature: "
             + art.label(tex_escape(clk.get("temp_source", "") or "not recorded"),
                         f"{clk['_file']}:{clk['_line']}:temp_source") + "."]
    on_perf = abs(fnum(clk["clock_mhz"]) - PERFORMANCE_CLOCK_MHZ) < 0.5
    if clk.get("clock_fell_back") == "True":
        notes.append("The clock fallback recorded in hw\\_soak.csv (clock\\_choice\\_reason) is expected: "
                     "the higher-clock build is post-implementation only (PLL constraint of the board), "
                     "the soak ran on the performance build." if on_perf else
                     "The bitstream fell back to a clock below the performance clock "
                     "(clock\\_fallback.py, hw\\_soak.csv clock\\_choice\\_reason).")
    if not on_perf:
        art.check(f"soak ran at {clk['clock_mhz']} MHz, not at the performance clock "
                  f"{PERFORMANCE_CLOCK_MHZ} MHz (DECISIONS D21)")
    return c.table(art, "@{}l" + "r" * len(keys) + "@{}", [head], body, cap, "tab:soak", notes=notes)


# --------------------------------------------------------------------------------------------
def _fit_rows(art: Artifact) -> dict:
    rows = art.rows("hw_b2_fit.csv", hw=True, basis=FIT_BASIS) or []
    by = latest(rows, lambda r: r["quantity"])
    for r in by.values():
        if r.get("power_label") != INA_LABEL:
            art.check(f"hw_b2_fit.csv line {r['_line']}: power_label {r.get('power_label')!r} != {INA_LABEL!r}")
    return by


def _pm_ci(art: Artifact, r: dict, est: str, se: str, lo: str, hi: str, fmt: str, scale: float = 1.0) -> str:
    def n(col):
        return art.num(fnum(r[col]) * scale, fmt, row=r, col=col + (f"*{scale:g}" if scale != 1 else ""))
    txt = n(est)
    if r.get(se):
        txt += r"\,$\pm$\," + n(se)
    if r.get(lo) and r.get(hi):
        txt += r" [" + n(lo) + ", " + n(hi) + "]"
    return txt


def b2_fit_table(c: T.Ctx) -> Artifact:
    art = c.art("tab_b2_fit")
    by = _fit_rows(art)
    body = []
    for q, lab in FIT_Q:
        r = by.get(q)
        if r is None or not r.get("k_w_per_mhz"):
            ph = art.placeholder(f"B2 fit {q}: hw_b2_fit.csv (source=hw)")
            body.append([lab, ph, ph, ph, ph])
            continue
        body.append([lab,
                     _pm_ci(art, r, "p_static_w", "p_static_se_w", "p_static_ci95_lo_w", "p_static_ci95_hi_w", "f3"),
                     _pm_ci(art, r, "k_w_per_mhz", "k_se_w_per_mhz", "k_ci95_lo_w_per_mhz",
                            "k_ci95_hi_w_per_mhz", "f3", 1e3),
                     art.cell(r, "r2", "f4") if r.get("r2") else "--",
                     art.cell(r, "n_points", "int")])
    r2 = art.label("$R^2$", "notation: coefficient of determination (hw_b2_fit.csv column r2)")
    ci = art.label("95", "hw_b2_fit.csv fit_method: 95 % CI (Student t, n-2 dof)")
    hdr = [f"Quantity & $P_\\mathrm{{static}}$ (W) & $k$ (mW/MHz) & {r2} & $n$"]
    return c.table(art, "@{}lllrr@{}", hdr, body,
                   f"B2 power vs.\\ clock: least-squares fit $P = P_\\mathrm{{static}} + k f$ on the per-clock "
                   f"mean {tex_escape(INA_LABEL)} ($f$ = read-back pl\\_clk0).", "tab:b2fit",
                   notes=[f"Estimate $\\pm$ standard error [{ci}\\,\\% confidence interval, Student $t$, $n-$2 dof]"
                          .replace("$n-$2", "$n-$" + art.label("2", "hw_b2_fit.csv fit_method: n-2 dof")) +
                          ". Computed by exp\\_b2\\_clock.py (hw\\_b2\\_fit.csv, basis per\\_clock\\_mean); "
                          "SOM rail (VCC\\_SOM), not accelerator-only and not board input power."])


def b2_fit_figure(fc: F.FCtx) -> Artifact:
    plt = F.plt
    art = fc.art("fig_b2_fit")
    by = _fit_rows(art)
    ents = [e for e in power_summaries(art, "hw_b2_power_ina260") if e["clock"]]
    fit_inputs = {f for r in by.values() for f in (r.get("inputs") or "").split()}
    if fit_inputs:          # plot exactly the per-clock summaries the fit was computed from
        dropped = sorted({e["file"] for e in ents if e["file"] not in fit_inputs})
        if dropped:
            art.check(f"fig_b2_fit: summaries not used by hw_b2_fit.csv, not plotted: {dropped}")
        ents = [e for e in ents if e["file"] in fit_inputs]
    fig, axs = plt.subplots(1, 2, figsize=(F.COL_W, 1.9), layout="constrained")
    data = []
    panels = [(axs[0], [("accel_p_run_w", "p_accel_w", "o", "-", r"$P_\mathrm{accel}$"),
                        ("accel_p_idle_w", "p_idle_w", "s", "--", r"$P_\mathrm{idle}$")], "SOM-rail P (W)"),
              (axs[1], [("accel_dp_w", "dp_accel_w", "^", "-", r"$\Delta P_\mathrm{accel}$")], "ΔP (W)")]
    for ax, series, title in panels:
        any_pt = False
        for col, q, mk, ls, lab in series:
            for net in NETS:
                en = sorted([e for e in ents if e["net"] == net and e["mean"].get(col)], key=lambda e: fnum(e["clock"]))
                if not en:
                    continue
                any_pt = True
                xs = [fnum(e["clock"]) for e in en]
                ys = [fnum(e["mean"][col]) for e in en]
                es = [fnum(e["std"][col]) if e["std"] is not None and e["std"].get(col) else 0.0 for e in en]
                ax.errorbar(xs, ys, yerr=es, marker=mk, ls="none", color="black", ms=3.2, mfc="white", mew=0.7,
                            capsize=1.5, elinewidth=0.6, zorder=3, label=f"{lab} {fc.hw_name}")
                data += [{"panel": col, "net": net, "clock_mhz": x, "value": y, "std": s, "unit": "W",
                          "kind": "hw INA260 per-clock mean", "src": f"{e['mean']['_file']}:{e['mean']['_line']}"}
                         for x, y, s, e in zip(xs, ys, es, en)]
                r = by.get(q)
                if r is not None and r.get("k_w_per_mhz") and r.get("p_static_w"):
                    a, k = fnum(r["p_static_w"]), fnum(r["k_w_per_mhz"])
                    lo, hi = min(xs), max(xs)
                    ax.plot([lo, hi], [a + k * lo, a + k * hi], ls=ls, color="0.35", lw=0.8, zorder=2,
                            label=f"fit {lab} = $P_\\mathrm{{static}}$ + $k f$")
                    data.append({"panel": col, "net": net, "kind": "fit line", "p_static_w": a, "k_w_per_mhz": k,
                                 "r2": r.get("r2", ""), "n_points": r.get("n_points", ""),
                                 "src": f"{r['_file']}:{r['_line']}"})
        if not any_pt:
            art.placeholder(f"B2 fit figure {title}: hw_b2_power_ina260_summary*.csv / hw_b2_fit.csv (source=hw)")
            F._placeholder_box(ax, 0.0, 1.0)
            ax.set_xlim(0, 1)
            ax.text(0.5, 0.5, "board data\npending", transform=ax.transAxes, ha="center", va="center",
                    fontsize=F.FS - 1.5, style="italic", color="0.2",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.55", lw=0.5, ls="--"))
            ax.set_yticks([])
            ax.set_xticks([])
        ax.set_title(title, fontsize=F.FS - 1, pad=3)
        ax.set_xlabel("pl_clk0 (MHz)", fontsize=F.FS - 1)
        ax.yaxis.grid(True, color="0.88", lw=0.4, zorder=0)
        ax.tick_params(labelsize=F.FS - 2)
    h, l = [], []
    for ax in axs:
        hh, ll = ax.get_legend_handles_labels()
        h += hh
        l += ll
    seen = dict(zip(l, h))
    if seen:
        fig.legend(list(seen.values()), list(seen.keys()), loc="outside lower center", ncol=2,
                   fontsize=F.FS - 2.5, handletextpad=0.3, columnspacing=0.8)
    fig.suptitle(f"P vs. f, least-squares fit: {INA_LABEL}", fontsize=F.FS - 1.5, color="0.25")
    return fc.save(art, fig, data)


# --------------------------------------------------------------------------------------------
def layer_spread_table(c: T.Ctx) -> Artifact:
    art = c.art("tab_layer_spread")
    rows = _latest_run(art.rows("hw_layer_spread.csv", hw=True) or [])
    rtl = {r["net"]: r for r in (art.rows("rtl_full10k.csv") or [])}
    body = []
    for net in NETS:
        nr = [r for r in rows if r["net"] == net]
        body.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{pretty_net(net)}}}"
                    + (f" (RTL sim: {art.cell(rtl[net], 'cycles_exact', 'int')}/{art.cell(rtl[net], 'images', 'int')}"
                       " images cycle-exact)" if net in rtl and rtl[net].get("cycles_exact") else "")
                    + "} \\\\")
        if not nr:
            ph = art.placeholder(f"layer spread {net}: hw_layer_spread.csv (source=hw)")
            body.append(["all layers", ph, ph, ph, ph, ph])
            continue
        for r in nr:
            lay = art.label(tex_escape(r["layer"]), f"{r['_file']}:{r['_line']}:layer")
            body.append([lay, art.cell(r, "model_cycles", "int"), art.cell(r, "images", "int"),
                         f"{art.cell(r, 'cyc_min', 'int')} / {art.cell(r, 'cyc_max', 'int')}",
                         art.cell(r, "cyc_distinct", "int"), art.cell(r, "cyc_spread", "int")])
            if r.get("result") != "PASS":
                art.check(f"hw_layer_spread.csv line {r['_line']} {net} {r['layer']}: result {r.get('result')}")
    hdr = ["Layer & Model & Images & KV260 min / max & Distinct & Spread",
           "& (cycles) & & (cycles) & values & (cycles)"]
    return c.table(art, "@{}lrrrrr@{}", hdr, body,
                   f"Per-layer cycle determinism on the {c.board_label}: LAYER\\_CYC and TOTAL\\_CYC over every "
                   "test image.", "tab:layerspread",
                   notes=["Source: hw\\_layer\\_spread.csv (cycles origin per row: A2/A3 per-image counters or "
                          "an own run); RTL-sim reference: rtl\\_full10k.csv. Spread = max $-$ min."])


# --------------------------------------------------------------------------------------------
# A3-general random shapes (v2/shapes: RTL sim; v2/board/exp_shapes.py: KV260)
# Column aliases: shapes_rtl.csv (collect_shapes.py) and hw_shapes.csv (exp_shapes.py) name the
# same quantities differently; the first present column wins.
SH_RTL, SH_HW = "shapes_rtl.csv", "hw_shapes.csv"
SH_COLS = {
    "id": ("job_id", "job", "id"),
    "refuse": ("expect_refuse",),
    "model_total": ("model_total",),
    "model_layers": ("model_layer_cycles",),
    "rtl_total": ("rtl_total",),
    "rtl_layers": ("rtl_layer_cycles",),
    "hw_total": ("hw_total", "total_cyc"),
    "hw_layers": ("hw_layer_cycles", "layer_cyc"),
    "out_ok": ("outputs_match", "output_exact"),
    "cyc_ok": ("cycles_exact", "cycles_eq_model"),
    "ref_ok": ("err_ok", "refuse_ok"),
    "all_ok": ("all_ok", "pass"),
}
TRUE = ("True", "true", "1")


def _sc(r: dict, key: str) -> str | None:
    """Name of the first present, non-empty alias column of `key` in row r."""
    for c in SH_COLS[key]:
        if r.get(c) not in ("", None):
            return c
    return None


def _is_refuse(r: dict) -> bool:
    c = _sc(r, "refuse")
    return bool(c) and r[c] in TRUE


def _ilist(s: str) -> list[int]:
    return [int(float(x)) for x in str(s).split(",") if x.strip() not in ("", "-")]


def _shape_rows(art: Artifact) -> tuple[list[dict], list[dict]]:
    """(RTL rows, board rows): per-job rows only; board rows latest per job id."""
    rtl = [r for r in (art.rows(SH_RTL) or []) if r.get("source") == "rtl_sim" and _sc(r, "id")]
    hw_all = [r for r in (art.rows(SH_HW, hw=True) or []) if _sc(r, "id") and r.get("layer") != "summary"
              and r.get("label", "") != "summary"]
    hw = sorted(latest(hw_all, lambda r: r[_sc(r, "id")]).values(), key=lambda r: r["_line"])
    shas = {r.get("shapeset_sha256") for r in rtl} | {r.get("shapeset_sha256") for r in hw}
    shas.discard(None)
    shas.discard("")
    if len(shas) > 1:
        art.check(f"random shapes: shapeset_sha256 differs between/within {SH_RTL} and {SH_HW}: "
                  f"{sorted(x[:12] for x in shas)}")
    return rtl, hw


def _lines(rows: list[dict]) -> str:
    ls = sorted(r["_line"] for r in rows)
    return f"lines {ls[0]}-{ls[-1]}" if ls else "no lines"


def _shape_summary(art: Artifact, rows: list[dict], meas: str, fname: str) -> dict:
    """Counts and max |measured - model| over per-job rows, each registered with its origin."""
    valid = [r for r in rows if not _is_refuse(r)]
    refuse = [r for r in rows if _is_refuse(r)]
    src = f"{fname} {_lines(rows)}"

    def cnt(rs, key, what):
        n = sum(1 for r in rs if (c := _sc(r, key)) and r[c] in TRUE)
        return n, art.num(n, "int", origin=f"computed count({what}) over {src}")

    tot_err, lay_err, n_cmp = [], [], 0
    mt_col = f"{meas}_total"
    for r in valid:
        cm_, cr = _sc(r, "model_total"), _sc(r, mt_col)
        if cm_ and cr:
            n_cmp += 1
            tot_err.append(abs(int(float(r[cr])) - int(float(r[cm_]))))
            ml, rl = _sc(r, "model_layers"), _sc(r, f"{meas}_layers")
            if ml and rl:
                a, b = _ilist(r[ml]), _ilist(r[rl])
                if len(a) == len(b):
                    lay_err += [abs(x - y) for x, y in zip(a, b)]
                else:
                    art.check(f"{fname} line {r['_line']}: layer-cycle list lengths differ")
    out = {"n": len(rows), "n_valid": len(valid), "n_refuse": len(refuse), "n_cmp": n_cmp}
    out["jobs"] = art.num(len(rows), "int", origin=f"computed count(rows) over {src}")
    out["refuse"] = art.num(len(refuse), "int", origin=f"computed count(expect_refuse) over {src}")
    out["valid_s"] = art.num(len(valid), "int", origin=f"computed count(not expect_refuse) over {src}")
    out["out_ok_n"], out["out_ok"] = cnt(valid, "out_ok", "outputs bit-exact, valid jobs")
    out["cyc_ok_n"], out["cyc_ok"] = cnt(valid, "cyc_ok", "cycle-exact, valid jobs")
    out["ref_ok_n"], out["ref_ok"] = cnt(refuse, "ref_ok", "refusal ERR_CODE correct")
    out["all_ok_n"], out["all_ok"] = cnt(rows, "all_ok", "all exact")
    out["max_tot"] = (art.num(max(tot_err), "int", origin=f"computed max|{meas}_total-model_total| over {src}")
                      if tot_err else "--")
    out["max_lay"] = (art.num(max(lay_err), "int", origin=f"computed max|{meas} layer - model layer| over {src}")
                      if lay_err else "--")
    for k, n in (("all_ok_n", len(rows)),):
        if out[k] != n:
            art.check(f"random shapes {fname}: {n - out[k]} of {n} jobs not exact")
    return out


def shapes_table(c: T.Ctx) -> Artifact:
    art = c.art("tab_shapes")
    rtl, hw = _shape_rows(art)
    body = []
    specs = [(rtl, "rtl", SH_RTL, "RTL sim", f"random shapes: {SH_RTL} (source=rtl_sim)", "TBD (RTL sim)"),
             (hw, "hw", SH_HW, f"{c.board_label} (measured)", f"random shapes: {SH_HW} (source=hw)", None)]
    for rows, meas, fname, lab, what, ph_text in specs:
        if not rows:
            ph = art.placeholder(what, ph_text) if ph_text else art.placeholder(what)
            body.append([lab, ph, ph, ph, ph, ph, ph])
            continue
        s_ = _shape_summary(art, rows, meas, fname)
        if meas == "rtl":
            sims = sorted({f"{r.get('simulator', '')} {r.get('simulator_version', '')}".strip() for r in rows})
            lab = lab + " (" + art.label(tex_escape(", ".join(sims)), f"{SH_RTL}:simulator,simulator_version") + ")"
        body.append([lab, f"{s_['jobs']} ({s_['refuse']})", f"{s_['out_ok']}/{s_['valid_s']}",
                     f"{s_['cyc_ok']}/{s_['valid_s']}", f"{s_['ref_ok']}/{s_['refuse']}",
                     f"{s_['all_ok']}/{s_['jobs']}", f"{s_['max_tot']} / {s_['max_lay']}"])
    hdr = ["Source & Jobs & Outputs & Cycle- & Refusals & All & max $|\\Delta|$ cycles",
           "& (refuse) & bit-exact & exact & correct & exact & total / layer"]
    notes = ["Random multi-layer jobs within the architecture envelope (gen\\_shapes.py, seeded). "
             "Model = gos\\_cycle\\_model.py; $\\Delta$ = measured $-$ model. Refuse jobs: the config checker "
             "must refuse with the expected ERR\\_CODE."]
    if rtl:
        seeds = sorted({r.get("seed", "") for r in rtl})
        nx = sum(1 for r in rtl if r.get("xsim_crosschecked") in TRUE)
        notes.append("Shape set seed " + art.label(tex_escape(",".join(seeds)), f"{SH_RTL}:seed")
                     + "; RTL jobs also cross-checked on xsim: "
                     + art.num(nx, "int", origin=f"computed count(xsim_crosschecked) over {SH_RTL} {_lines(rtl)}")
                     + ".")
    return c.table(art, "@{}lrrrrrr@{}", hdr, body,
                   "A3-general: random-shape jobs, predicted (model) vs.\\ measured cycles and bit-exact outputs.",
                   "tab:shapes", notes=notes)


def shapes_figure(fc: F.FCtx) -> Artifact:
    plt = F.plt
    art = fc.art("fig_shapes_cycles")
    rtl, hw = _shape_rows(art)
    fig, ax = plt.subplots(figsize=(F.COL_W, 2.6), layout="constrained")
    data: list[dict] = []

    def pts(rows, meas, fname, per_layer):
        xs, ys, lx, ly = [], [], [], []
        for r in rows:
            if _is_refuse(r):
                continue
            cm_, cr = _sc(r, "model_total"), _sc(r, f"{meas}_total")
            if not (cm_ and cr):
                continue
            x, y = int(float(r[cm_])), int(float(r[cr]))
            xs.append(x)
            ys.append(y)
            src = f"{_rel(Path(r['_file']))}:{r['_line']}"
            data.append({"source": meas, "kind": "total", "job": r[_sc(r, "id")], "model_cycles": x,
                         "measured_cycles": y, "src": src})
            ml, rl = _sc(r, "model_layers"), _sc(r, f"{meas}_layers")
            if per_layer and ml and rl:
                a, b = _ilist(r[ml]), _ilist(r[rl])
                if len(a) == len(b) and len(a) > 1:
                    lx += a
                    ly += b
                    data.extend([{"source": meas, "kind": f"layer{i}", "job": r[_sc(r, "id")], "model_cycles": p,
                              "measured_cycles": q, "src": src} for i, (p, q) in enumerate(zip(a, b))])
        return xs, ys, lx, ly

    rx, ry, rlx, rly = pts(rtl, "rtl", SH_RTL, True)
    hx, hy, _, _ = pts(hw, "hw", SH_HW, False)
    if rlx:
        ax.scatter(rlx, rly, s=4, marker=".", color="0.6", lw=0, zorder=2, label="RTL sim, per layer (LAYER_CYC)")
    if rx:
        ax.scatter(rx, ry, s=14, marker="o", facecolor="white", edgecolor="0.3", lw=0.6, zorder=3,
                   label="RTL sim, job (TOTAL_CYC)")
    if hx:
        ax.scatter(hx, hy, s=16, marker="x", color="black", lw=0.7, zorder=4, label=f"{fc.hw_name}, job (TOTAL_CYC)")
    allv = rx + ry + rlx + rly + hx + hy
    if allv:
        lo, hi = max(1, min(allv)) / 1.8, max(allv) * 1.8
    else:
        lo, hi = 1e2, 1e6
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.45", lw=0.6, zorder=1, label="y = x (model)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("predicted cycles (model)")
    ax.set_ylabel("measured cycles")
    ax.grid(True, which="major", color="0.9", lw=0.4, zorder=0)
    ann = []
    if rx:
        s_ = _shape_summary(art, rtl, "rtl", SH_RTL)
        ann.append(f"RTL sim: {s_['cyc_ok']}/{s_['valid_s']} jobs cycle-exact")
    else:
        art.placeholder(f"random shapes figure: {SH_RTL} (source=rtl_sim)", "TBD (RTL sim)")
    if hx:
        s_ = _shape_summary(art, hw, "hw", SH_HW)
        ann.append(f"{fc.hw_name}: {s_['cyc_ok']}/{s_['valid_s']} jobs cycle-exact")
    else:
        art.placeholder(f"random shapes figure: {SH_HW} (source=hw)")
        F._pending_label(ax, "board data pending" if rx else "RTL sim + board data pending")
    if ann:
        ax.text(0.03, 0.97 if hx else 0.86, "\n".join(ann), transform=ax.transAxes, ha="left", va="top",
                fontsize=F.FS - 1.5, color="0.15")
    ax.legend(loc="lower right", fontsize=F.FS - 2.5, handletextpad=0.3, borderaxespad=0.3)
    return fc.save(art, fig, data)


ALL_TABLES = [soak_table, b2_fit_table, layer_spread_table, shapes_table]
ALL_FIGURES = [b2_fit_figure, shapes_figure]


def make(ctx, failures: list | None = None) -> list[Artifact]:
    """All extra artifacts. With a failures list: record failures and continue (make_all style);
    without: raise on the first failure."""
    tc, fc = _ctxs(ctx)
    F.style()
    arts = []
    for fn, c_ in [(f, tc) for f in ALL_TABLES] + [(f, fc) for f in ALL_FIGURES]:
        try:
            arts.append(fn(c_))
        except Exception as e:
            if failures is None:
                raise
            failures.append({"artifact": fn.__name__, "error": f"{type(e).__name__}: {e}",
                             "trace": traceback.format_exc()})
    return arts


__all__ = ["make", "soak_table", "b2_fit_table", "b2_fit_figure", "layer_spread_table", "shapes_table",
           "shapes_figure"]
