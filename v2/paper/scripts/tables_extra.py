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
"""
from __future__ import annotations

import traceback

import figures as F
import tables as T
from paperdata import INA_LABEL, latest, power_summaries
from paperlib import NETS, Artifact, fnum, pretty_net, tex_escape

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
    if clk.get("clock_fell_back") == "True":
        notes.append("The performance bitstream fell back to a lower clock (clock\\_fallback.py, "
                     "hw\\_soak.csv clock\\_choice\\_reason).")
        art.check("soak ran on the fallback bitstream (clock_fell_back=True)")
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


ALL_TABLES = [soak_table, b2_fit_table, layer_spread_table]
ALL_FIGURES = [b2_fit_figure]


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


__all__ = ["make", "soak_table", "b2_fit_table", "b2_fit_figure", "layer_spread_table"]
