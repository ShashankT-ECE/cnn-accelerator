"""Tests of the paper table/figure pipeline (v2/paper/scripts/make_all.py).

Run: .venv/bin/python -m pytest v2/paper/tests -q
Every test works on a temporary copy of v2/results/*.csv; nothing in the repo is modified.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import make_all  # noqa: E402
from paperlib import (DEFAULT_RESULTS, PH_BOARD, Artifact, ProvenanceError, Store, latex_table,  # noqa: E402
                      num_tokens, sha256_file, strip_for_scan)

META = ["timestamp", "git_commit", "git_dirty", "vivado_version", "bitstream_sha256", "board_id", "net",
        "layer", "clock_mhz", "source", "duration_s", "num_inferences"]


# ---------------------------------------------------------------------------------- helpers
def read_csv(p: Path):
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write_csv(p: Path, fields, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def edit_csv(p: Path, match: dict, **changes):
    fields, rows = read_csv(p)
    n = 0
    for r in rows:
        if all(r.get(k) == v for k, v in match.items()):
            r.update({k: str(v) for k, v in changes.items()})
            n += 1
    assert n, f"no row matched {match} in {p.name}"
    write_csv(p, fields, rows)


def run(res: Path, out: Path, *extra) -> dict:
    rc = make_all.main(["--results-dir", str(res), "--out-dir", str(out), "--quiet", *extra])
    m = json.loads((out / "MANIFEST.json").read_text())
    assert rc == 0, m["failures"]
    return m


def tex(out: Path, name: str) -> str:
    return (out / f"{name}.tex").read_text()


def model_rows(res: Path):
    return [r for r in read_csv(res / "cycle_model.csv")[1]]


def synth_hw_a2a3(res: Path, dirty=False, source="hw", delta=None, where=None):
    """Synthetic board file with the documented hw_a2_a3_cycles.csv columns (board/README.md),
    built from cycle_model.csv so that hw = model (+delta on one layer)."""
    head = read_csv(res / "cycle_model.csv")[1][0]
    fields = META + ["images", "model_cycles", "rtl_cycles", "hw_cycles", "hw_min", "hw_max", "hw_distinct",
                     "rtl_err_pct", "hw_err_pct", "hw_us", "wall_us_median"]
    rows = []
    for m in model_rows(res):
        hw = int(m["cycles"]) + (delta if delta and (m["net"], m["layer"]) == where else 0)
        rows.append({"timestamp": "2026-10-01T00:00:00+00:00", "git_commit": head["git_commit"],
                     "git_dirty": str(dirty), "net": m["net"], "layer": m["layer"], "clock_mhz": "199.998001",
                     "source": source, "images": "10000", "model_cycles": m["cycles"], "rtl_cycles": m["cycles"],
                     "hw_cycles": str(hw), "hw_min": str(hw), "hw_max": str(hw), "hw_distinct": "1",
                     "rtl_err_pct": "0", "hw_err_pct": str((hw - int(m["cycles"])) / int(m["cycles"]) * 100),
                     "hw_us": f"{hw / 199.998001:.4f}", "wall_us_median": "4321.5"})
    return fields, rows


@pytest.fixture()
def res(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    for p in DEFAULT_RESULTS.glob("*.csv"):          # top level only: never the dryrun dir
        if not p.name.startswith("hw_"):             # board data absent: tests add synthetic hw files
            shutil.copy(p, d / p.name)
    return d


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    t = tmp_path_factory.mktemp("base")
    d = t / "results"
    d.mkdir()
    for p in DEFAULT_RESULTS.glob("*.csv"):
        if not p.name.startswith("hw_"):
            shutil.copy(p, d / p.name)
    out = t / "out"
    return d, out, run(d, out)


# ------------------------------------------------------------------------------------ tests
def test_all_artifacts_and_manifest(baseline):
    res, out, m = baseline
    for n in ["tab_t1_impl", "tab_a1_accuracy", "tab_a2_latency", "tab_a3_cycles", "tab_a4_util", "tab_a5_cpu",
              "tab_b1_power", "tab_b2_clock", "tab_b3_breakdown", "tab_verification"]:
        assert (out / f"{n}.tex").exists(), n
    for n in ["fig_a3_cycles", "fig_a3_cycles_wide", "fig_a4_util", "fig_b2_clock", "fig_b3_breakdown"]:
        assert (out / f"{n}.pdf").exists(), n
    # manifest sha256 = the files actually read; row commits listed
    for f in m["files"]:
        assert f["sha256"] == sha256_file(Path(f["path"]) if Path(f["path"]).is_absolute()
                                          else Path(make_all.__file__).resolve().parents[3] / f["path"])
    a3 = m["artifacts"]["fig_a3_cycles"]
    assert a3["row_git_commits"] and all(len(c) == 40 for c in a3["row_git_commits"])
    assert {"model", "rtl_sim"} <= set(a3["sources"])


def test_every_table_number_is_registered(baseline):
    """No numeric token in any table without a registered CSV origin (re-checked from the manifest)."""
    res, out, m = baseline
    for name, a in m["artifacts"].items():
        if a["kind"] != "table":
            continue
        ok = set()
        for n in a["numbers"]:
            ok |= set(num_tokens(n["text"]))
            assert n["origin"], f"{name}: number {n['text']} without origin"
        bad = [t for t in num_tokens(strip_for_scan(tex(out, name))) if t not in ok]
        assert not bad, f"{name}: unregistered numbers {bad}"


def test_hand_typed_number_is_refused():
    art = Artifact(Store(DEFAULT_RESULTS), "t", "table")
    with pytest.raises(ProvenanceError):
        latex_table(art, "ll", ["a & b"], [["x", "42"]], "cap", "tab:x")
    art.num(42, "int", origin="test")
    latex_table(art, "ll", ["a & b"], [["x", "42"]], "cap", "tab:x")


def test_tampered_cycle_model_changes_outputs(baseline, res, tmp_path):
    _, bout, _ = baseline
    edit_csv(res / "cycle_model.csv", {"net": "lenet5", "layer": "conv1"}, cycles=2833)
    out = tmp_path / "out"
    run(res, out)
    t = tex(out, "tab_a3_cycles")
    assert "conv1 & 2,833 & 2,829" in t                 # model changed, RTL unchanged
    assert "$-$0.14" in t                                # (2829-2833)/2833 -> -0.14 %
    assert (out / "fig_a3_cycles.pdf").read_bytes() != (bout / "fig_a3_cycles.pdf").read_bytes()
    _, rows = read_csv(out / "fig_a3_cycles.data.csv")
    r = next(r for r in rows if r["net"] == "lenet5" and r["layer"] == "conv1")
    assert float(r["model_cycles"]) == 2833 and float(r["rtl_err_pct"]) != 0
    m = json.loads((out / "MANIFEST.json").read_text())
    assert any("model_layer_cycles" in c for c in m["artifacts"]["tab_a3_cycles"]["checks"])


def test_tampered_impl_changes_t1(res, tmp_path):
    fields, rows = read_csv(res / "impl_gos.csv")
    old = rows[0]["clb_luts"]
    rows[0]["clb_luts"] = "12345"
    write_csv(res / "impl_gos.csv", fields, rows)
    out = tmp_path / "out"
    run(res, out)
    t = tex(out, "tab_t1_impl")
    assert "12,345" in t and f"{int(old):,}" not in t.split("CLB LUT", 1)[1].split("\\\\", 1)[0]


def test_placeholders_when_board_data_absent(baseline):
    res, out, m = baseline
    assert not list(res.glob("hw_*.csv"))
    for n in ["tab_a1_accuracy", "tab_a2_latency", "tab_a3_cycles", "tab_a4_util", "tab_a5_cpu",
              "tab_b1_power", "tab_b2_clock", "tab_b3_breakdown"]:
        assert PH_BOARD in tex(out, n), n
    for n in ["fig_a3_cycles", "fig_a3_cycles_wide", "fig_a4_util", "fig_b2_clock", "fig_b3_breakdown"]:
        assert m["artifacts"][n]["placeholders"], n
    _, rows = read_csv(out / "fig_a3_cycles.data.csv")
    assert all(r["hw_cycles"] == "" and r["hw_src"] == "board data pending" for r in rows)


def test_switches_to_board_data_automatically(res, tmp_path):
    fields, rows = synth_hw_a2a3(res, delta=1, where=("cifar10", "fc"))
    write_csv(res / "hw_a2_a3_cycles.csv", fields, rows)
    out = tmp_path / "out"
    m = run(res, out)
    t = tex(out, "tab_a3_cycles")
    assert PH_BOARD not in t
    assert "fc & 157 & 157 & 158" in t and "+0.64" in t    # (158-157)/157
    assert not m["artifacts"]["fig_a3_cycles"]["placeholders"]
    assert "hw" in m["artifacts"]["fig_a3_cycles"]["sources"]
    assert "4,321.5" in tex(out, "tab_a2_latency")


def test_dirty_board_rows_rejected(res, tmp_path):
    fields, rows = synth_hw_a2a3(res, dirty=True)
    write_csv(res / "hw_a2_a3_cycles.csv", fields, rows)
    out = tmp_path / "out"
    m = run(res, out)
    assert PH_BOARD in tex(out, "tab_a3_cycles")
    assert m["artifacts"]["fig_a3_cycles"]["placeholders"]
    f = next(f for f in m["files"] if f["path"].endswith("hw_a2_a3_cycles.csv"))
    assert f["rows_clean"] == 0 and all("git_dirty" in r["reason"] for r in f["rejected"])


def test_dirty_result_row_rejected(res, tmp_path):
    fields, rows = read_csv(res / "impl_gos.csv")
    assert len(rows) >= 2
    clk = rows[-1]["pl_clk0_mhz_requested"]
    rows[-1]["git_dirty"] = "True"
    write_csv(res / "impl_gos.csv", fields, rows)
    out = tmp_path / "out"
    m = run(res, out)
    assert f"{clk}~MHz" not in tex(out, "tab_t1_impl")
    f = next(f for f in m["files"] if f["path"].endswith("impl_gos.csv"))
    assert f["rejected"] and f["rejected"][0]["reason"].startswith("git_dirty")


def test_non_hw_source_rejected_as_board_data(res, tmp_path):
    fields, rows = synth_hw_a2a3(res, source="dryrun_model")
    write_csv(res / "hw_a2_a3_cycles.csv", fields, rows)
    out = tmp_path / "out"
    run(res, out)
    assert PH_BOARD in tex(out, "tab_a3_cycles")


def test_dryrun_off_by_default_and_watermarked(res, tmp_path):
    fields, rows = synth_hw_a2a3(res, source="dryrun_model")
    write_csv(res / "dryrun" / "hw_a2_a3_cycles.csv", fields, rows)
    out = tmp_path / "out"
    m = run(res, out)
    assert PH_BOARD in tex(out, "tab_a3_cycles")
    assert not any("/dryrun/" in f["path"] for f in m["files"])
    assert "DRY RUN" not in "".join(tex(out, n.stem) for n in out.glob("*.tex"))
    dout = tmp_path / "dry"
    md = run(res, dout, "--dryrun")
    assert md["dryrun"] is True
    for n in dout.glob("tab_*.tex"):
        assert "DRY RUN -- NOT DATA" in n.read_text(), n.name
    assert PH_BOARD not in tex(dout, "tab_a3_cycles")


def test_300mhz_column_added_only_if_timing_met(res, tmp_path):
    fields, rows = read_csv(res / "impl_gos.csv")
    # since step 9 the real impl_gos.csv has a timing-met 300 MHz build: test the rule on a temp
    # copy without it (the synthetic row below is the only 300 MHz candidate)
    rows = [r for r in rows if r["pl_clk0_mhz_requested"] not in ("300", "300.0")]
    new = dict(rows[-1], pl_clk0_mhz_requested="300", pl_clk0_mhz_actual="299.997", wns_ns="0.011",
               timestamp="2026-10-02T00:00:00+00:00")
    write_csv(res / "impl_gos.csv", fields, rows + [new])
    out = tmp_path / "out"
    run(res, out)
    t = tex(out, "tab_t1_impl")
    assert "300~MHz" in t and "299.997" in t
    assert "300~MHz" in tex(out, "tab_a2_latency")
    new["wns_ns"] = "-0.050"
    write_csv(res / "impl_gos.csv", fields, rows + [new])
    out2 = tmp_path / "out2"
    m = run(res, out2)
    assert "300~MHz" not in tex(out2, "tab_t1_impl")
    assert any("timing not met" in c for c in m["artifacts"]["tab_t1_impl"]["checks"])


def test_retrain_log_exempt_from_dirty_rule():
    s = Store(DEFAULT_RESULTS)
    rows = s.load("cifar10_retrain_log.csv")
    assert rows and any(r["git_dirty"] == "True" for r in rows)


def test_verification_stats_rendered_when_present(res, tmp_path):
    head = read_csv(res / "cycle_model.csv")[1][0]
    fields = META + ["item", "count", "passed"]
    rows = [{"timestamp": head["timestamp"], "git_commit": head["git_commit"], "git_dirty": "False",
             "source": "rtl_sim", "item": "unit_checks", "count": "7777", "passed": "True"},
            {"timestamp": head["timestamp"], "git_commit": head["git_commit"], "git_dirty": "True",
             "source": "rtl_sim", "item": "dirty_row", "count": "9999", "passed": "True"}]
    write_csv(res / "verification_stats.csv", fields, rows)
    out = tmp_path / "out"
    run(res, out)
    t = tex(out, "tab_verification")
    assert "7777" in t and "9999" not in t and "pending" not in t


# ------------------------------------------------------------------ INA260 power (power_log.py)
INA_COLS = ["measurement", "rail", "sensor_backend", "rate_requested_hz", "rate_achieved_hz", "row_kind", "repeat",
            "n_repeats", "cpu_workload"] + [f"{k}_{f}" for k in ("accel", "cpu") for f in
                                            ("p_idle_w", "p_run_w", "dp_w", "images", "duration_s",
                                             "time_per_image_s", "energy_per_image_j", "energy_per_image_mj")]


def synth_ina(res: Path, name: str, net: str, clock: str, dp: float, dirty=False, source="hw"):
    head = read_csv(res / "cycle_model.csv")[1][0]
    rows = []
    reps = [dp - 0.01, dp, dp + 0.01]
    for kind, rep, v in [("repeat", str(i + 1), x) for i, x in enumerate(reps)] + \
                        [("mean", "all", sum(reps) / 3), ("std", "all", 0.01)]:
        r = {"timestamp": "2026-10-01T00:00:00+00:00", "git_commit": head["git_commit"], "git_dirty": str(dirty),
             "net": net, "layer": "all", "clock_mhz": clock, "source": source,
             "measurement": "SOM-rail power (INA260)", "rail": "VCC_SOM", "sensor_backend": "hwmon",
             "rate_requested_hz": "10.000", "rate_achieved_hz": "9.876", "row_kind": kind, "repeat": rep,
             "n_repeats": "3", "cpu_workload": "cpu_int8_ref x1 threads " + net,
             "accel_p_idle_w": "3.210000" if kind != "std" else "0.004", "accel_dp_w": f"{v:.6f}",
             "cpu_dp_w": f"{v * 2:.6f}" if kind != "std" else "0.02",
             "accel_time_per_image_s": "0.004000000" if kind != "std" else "0.0001",
             "cpu_time_per_image_s": "0.003000000" if kind != "std" else "0.0001",
             "accel_energy_per_image_mj": f"{v * 4:.6f}" if kind != "std" else "0.04",
             "cpu_energy_per_image_mj": f"{v * 6:.6f}" if kind != "std" else "0.06"}
        rows.append(r)
    write_csv(res / name, META + INA_COLS, rows)


def test_b1_b2_use_ina260_summaries(res, tmp_path):
    out0 = tmp_path / "out0"
    run(res, out0)
    for n in ("tab_b1_power", "tab_b2_clock"):
        t = tex(out0, n)
        assert PH_BOARD in t and "meter" not in t.lower().split("\\end{tabular}")[0] and "TBD (meter)" not in t
    synth_ina(res, "hw_b1_power_ina260_summary_lenet5.csv", "lenet5", "199.998001", 0.3456)
    synth_ina(res, "hw_b2_power_ina260_summary_200mhz.csv", "lenet5", "199.998001", 0.3456)
    synth_ina(res, "hw_b2_power_ina260_summary_100mhz.csv", "lenet5", "100.000000", 0.1234)
    synth_ina(res, "hw_b1_power_ina260_summary_sensorcheck.csv", "lenet5", "199.998001", 9.99)  # ignored
    out = tmp_path / "out"
    m = run(res, out)
    b1, b2 = tex(out, "tab_b1_power"), tex(out, "tab_b2_clock")
    assert "SOM-rail power (INA260)" in b1 and "SOM-rail power (INA260)" in b2
    assert "0.346\\,$\\pm$\\,0.010" in b1 and "1.382" in b1          # dP accel; energy = 4 x dP
    assert "0.123\\,$\\pm$\\,0.010" in b2 and "0.346\\,$\\pm$\\,0.010" in b2
    assert "9.990" not in b1 and "external meter" not in b1         # sensorcheck ignored; no meter rows
    assert PH_BOARD in b1                                           # CIFAR-10 column still pending
    assert "hw" in m["artifacts"]["fig_b2_clock"]["sources"]
    # no external meter: a stray meter summary file is never read
    synth_ina(res, "hw_b1_power_meter_summary_lenet5.csv", "lenet5", "199.998001", 0.5)
    out2 = tmp_path / "out2"
    m2 = run(res, out2)
    assert "meter" not in tex(out2, "tab_b1_power").lower()
    assert not any("meter" in f["path"] for f in m2["files"])


def test_dirty_ina260_rows_rejected(res, tmp_path):
    synth_ina(res, "hw_b1_power_ina260_summary_lenet5.csv", "lenet5", "199.998001", 0.3456, dirty=True)
    out = tmp_path / "out"
    run(res, out)
    assert "0.346" not in tex(out, "tab_b1_power")


GOS_BUILD_IMPL = Path("/home/shashankt/gos-build/v2/results/impl_gos.csv")


@pytest.mark.skipif(not GOS_BUILD_IMPL.exists(), reason="gos-build impl_gos.csv (3 clocks) not present")
def test_t1_gets_300mhz_column_from_build_copy(res, tmp_path):
    shutil.copy(GOS_BUILD_IMPL, res / "impl_gos.csv")               # temp copy only
    out = tmp_path / "out"
    run(res, out)
    t = tex(out, "tab_t1_impl")
    # D21: 250 MHz = performance clock (star), 300 MHz = post-implementation only (double dagger)
    assert "200~MHz & 250~MHz$^\\star$ & 300~MHz$^\\ddagger$" in t
    assert "post-impl.\\ only" in t and "post-impl.; run on KV260" in t
    assert "Post-implementation only (300~MHz)" in t and "PLL" in t and "Performance clock" in t
    a2 = tex(out, "tab_a2_latency")
    assert "300~MHz$^\\dagger$$^\\ddagger$" in a2 and "250~MHz$^\\dagger$$^\\star$" in a2
    assert "Post-implementation only (300~MHz)" in a2
    assert "$^\\ddagger$" not in tex(out, "tab_b2_clock")


# ------------------------------------------------ B1 energy two ways / control; B3 safe vs fast
def test_b1_control_energy_rows_and_placeholders(res, tmp_path):
    out0 = tmp_path / "out0"
    run(res, out0)
    t0 = tex(out0, "tab_b1_power")
    for lab in ("$\\Delta P$ control", "$E_\\mathrm{comp}$ accel.", "Duty cycle",
                "$E_\\mathrm{sys}$ accel.\\ $-$ control"):
        assert lab in t0, lab                      # rows present, cells pending
    assert PH_BOARD in t0.split("Duty cycle", 1)[1].split("\\\\", 1)[0]
    synth_ina(res, "hw_b1_power_ina260_summary_lenet5.csv", "lenet5", "199.998001", 0.3456)
    f, rows = read_csv(res / "hw_b1_power_ina260_summary_lenet5.csv")
    extra = ["host_path", "control_dp_w", "accel_dp_net_w", "accel_t_pl_s", "accel_duty",
             "accel_e_comp_mj", "accel_e_sys_net_mj", "accel_e_comp_net_mj"]
    for r in rows:
        std = r["row_kind"] == "std"
        r.update(host_path="fast", control_dp_w="0.001" if std else "0.0456",
                 accel_dp_net_w="0.001" if std else f"{float(r['accel_dp_w']) - 0.0456:.6f}",
                 accel_t_pl_s="0" if std else "0.000411235", accel_duty="0" if std else "0.102809",
                 accel_e_comp_mj="0" if std else "0.142129", accel_e_sys_net_mj="0" if std else "1.2",
                 accel_e_comp_net_mj="0" if std else "0.123")
    write_csv(res / "hw_b1_power_ina260_summary_lenet5.csv", f + extra, rows)
    out = tmp_path / "out"
    run(res, out)
    t = tex(out, "tab_b1_power")
    assert "0.046\\,$\\pm$\\,0.001" in t                  # dP control
    assert "411.235" in t and "10.281" in t              # t_PL in us; duty in %
    assert "0.1421" in t and "0.1230" in t               # E_comp, E_comp net (f4)
    assert "fast" in t


def synth_b3(res: Path, name: str, host_path: str, e2e: float):
    head = read_csv(res / "cycle_model.csv")[1][0]
    rows = []
    for net in ("lenet5", "cifar10"):
        for ph, v in (("input_write", e2e * 0.5), ("status_clear", 1.0), ("start_done", e2e * 0.3),
                      ("logit_read", 2.0), ("ps_dequant", 3.0), ("end_to_end", e2e), ("pl_compute", 80.0)):
            rows.append({"timestamp": "2026-10-01T00:00:00+00:00", "git_commit": head["git_commit"],
                         "git_dirty": "False", "net": net, "layer": "all", "clock_mhz": "199.998001",
                         "source": "hw", "phase": ph, "median_us": f"{v:.3f}", "p95_us": f"{v * 1.1:.3f}",
                         "host_path": host_path})
    write_csv(res / name, META + ["phase", "median_us", "p95_us", "host_path"], rows)


def test_b3_safe_vs_fast_columns_and_speedup(res, tmp_path):
    synth_b3(res, "hw_b3_breakdown.csv", "safe", 900.0)
    out = tmp_path / "out"
    m = run(res, out)
    t = tex(out, "tab_b3_breakdown")
    assert "900.0" in t and PH_BOARD in t                # fast columns still pending
    assert m["artifacts"]["fig_b3_breakdown"]["placeholders"]
    synth_b3(res, "hw_b3_breakdown_fast.csv", "fast", 300.0)
    out2 = tmp_path / "out2"
    m2 = run(res, out2)
    t2 = tex(out2, "tab_b3_breakdown")
    assert "300.0" in t2 and "3.00$\\times$" in t2 and PH_BOARD not in t2
    assert not m2["artifacts"]["fig_b3_breakdown"]["placeholders"]


# ---------------------------------------------------------------------- measurement rigor
def test_non_paper_grade_board_rows_rejected(res, tmp_path):
    fields, rows = synth_hw_a2a3(res)
    for r in rows:
        r["paper_grade"] = "False"
    write_csv(res / "hw_a2_a3_cycles.csv", fields + ["paper_grade"], rows)
    out = tmp_path / "out"
    m = run(res, out)
    assert PH_BOARD in tex(out, "tab_a3_cycles")
    f = next(f for f in m["files"] if f["path"].endswith("hw_a2_a3_cycles.csv"))
    assert f["rows_clean"] == 0 and all("paper_grade" in r["reason"] for r in f["rejected"])
    for r in rows:
        r["paper_grade"] = "True"
    write_csv(res / "hw_a2_a3_cycles.csv", fields + ["paper_grade"], rows)
    out2 = tmp_path / "out2"
    run(res, out2)
    assert PH_BOARD not in tex(out2, "tab_a3_cycles")


def test_b3_median_with_ci_cells(res, tmp_path):
    synth_b3(res, "hw_b3_breakdown.csv", "safe", 900.0)
    fields, rows = read_csv(res / "hw_b3_breakdown.csv")
    for r in rows:
        v = float(r["median_us"])
        r["median_ci_lo_us"], r["median_ci_hi_us"] = f"{v - 1.25:.3f}", f"{v + 2.5:.3f}"
    write_csv(res / "hw_b3_breakdown.csv", fields + ["median_ci_lo_us", "median_ci_hi_us"], rows)
    out = tmp_path / "out"
    run(res, out)
    t = tex(out, "tab_b3_breakdown")
    assert "900.0 [898.8, 902.5]" in t and "95\\% CI" in t


def test_repeatability_table_placeholder_then_data(res, tmp_path):
    out = tmp_path / "out"
    run(res, out)
    assert PH_BOARD in tex(out, "tab_repeatability")
    head = read_csv(res / "cycle_model.csv")[1][0]
    rows = []
    for net in ("lenet5", "cifar10"):
        rows.append({"timestamp": "2026-10-02T00:00:00+00:00", "git_commit": head["git_commit"],
                     "git_dirty": "False", "net": net, "layer": "", "clock_mhz": "300.0",
                     "source": "hw", "paper_grade": "True", "metric_file": "hw_b3_breakdown.csv",
                     "metric": "median_us", "key": json.dumps({"net": net, "phase": "end_to_end"}),
                     "n_sessions": "3", "mean": "123.456", "between_sd": "1.5"})
    write_csv(res / "hw_repeatability.csv", META + ["paper_grade", "metric_file", "metric", "key",
                                                   "n_sessions", "mean", "between_sd"], rows)
    out2 = tmp_path / "out2"
    run(res, out2)
    t = tex(out2, "tab_repeatability")
    assert "123.5\\,$\\pm$\\,1.5 (3)" in t


def test_tables_extra_hook_guarded(res, tmp_path, monkeypatch):
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "tables_extra", None)          # absent -> no failure
    out = tmp_path / "out"
    m = run(res, out)
    assert not m["failures"]


# ---- best CPU baseline / speedups (user decision 2026-09-30: claims only vs the best CPU baseline) ----------
def _cpu_row(net, kind, th, mode, med, line):
    return {"net": net, "kind": kind, "threads": str(th), "mode": mode, "median_us": str(med),
            "status": "ok", "timestamp": "2026-10-01T00:00:00+00:00", "_file": "v2/results/hw_cpu_baseline.csv",
            "_line": line, "accuracy": ""}


def test_best_cpu_is_the_fastest_e2e_configuration():
    sys.path.insert(0, str(SCRIPTS))
    import tables
    rows = []
    ln = 2
    for net, vals in (("lenet5", {("cpu_int8_ref", 1): 44800, ("cpu_fp32_numpy", 1): 2170, ("cpu_ort_fp32", 4): 760,
                                  ("cpu_ort_int8", 1): 880, ("cpu_ort_int8", 4): 700}),
                      ("cifar10", {("cpu_fp32_numpy", 4): 6800, ("cpu_ort_int8", 4): 1200, ("cpu_ort_fp32", 4): 2000})):
        for (k, th), v in vals.items():
            rows.append(_cpu_row(net, k, th, "e2e", v, ln))
            rows.append(_cpu_row(net, k, th, "compute", v - 100, ln + 1))
            ln += 2
    rows.append(dict(_cpu_row("lenet5", "cpu_ort_int8", 1, "accuracy", "", ln), accuracy="98.5"))
    best = tables._best_cpu(rows)
    assert best["lenet5"][:2] == ("cpu_ort_int8", "4") and best["cifar10"][:2] == ("cpu_ort_int8", "4")
    assert best["lenet5"][2]["median_us"] == "700" and best["lenet5"][3]["median_us"] == "600"
    # a failed / unavailable row never wins
    bad = dict(_cpu_row("lenet5", "cpu_ort_fp32", 1, "e2e", 1, 99), status="unavailable: onnxruntime not installed")
    assert tables._best_cpu(rows + [bad])["lenet5"][:2] == ("cpu_ort_int8", "4")


def test_a4_labels_board_counter_as_array_busy_fraction(baseline):
    res, out, m = baseline
    t = tex(out, "tab_a4_util")
    assert "Array busy fraction" in t and "Lane util." in t and "lane utilization" in t
    assert "PE util" not in t


def test_real_board_results_generate_without_failures(tmp_path):
    """The committed board results (whatever is there) go through the whole pipeline."""
    d = tmp_path / "results"
    d.mkdir()
    for p in DEFAULT_RESULTS.glob("*.csv"):
        shutil.copy(p, d / p.name)
    if not (d / "hw_cpu_baseline.csv").exists():
        pytest.skip("no board results committed")
    out = tmp_path / "out"
    run(d, out)
    a5 = tex(out, "tab_a5_cpu")
    assert "Best CPU baseline" in a5 and "Speedup vs best CPU" in a5
    for n in ("tab_percentiles", "tab_efficiency", "tab_power_compare"):
        assert (out / f"{n}.tex").exists(), n
    pct = tex(out, "tab_percentiles")
    assert "spread" in pct and "DPU" in pct and pct.index("fast host path") < pct.index("safe host path")
    eff = tex(out, "tab_efficiency")
    assert "MAC lane" in eff
    assert "no DPU per-DSP figure" in eff or "TBD" in eff            # DPU per-DSP rows dropped, stated in the notes
    assert "DPU, runner only: GOPS / DSP" not in eff and "DPU, e2e: images/s / DSP" not in eff
    assert "Lane clock" in eff and "pynqdpu.dpu.kv260" in eff or "TBD" in eff
