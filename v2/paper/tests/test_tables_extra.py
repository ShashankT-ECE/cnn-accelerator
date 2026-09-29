"""tables_extra (soak / B2 fit / layer spread): placeholders, provenance, dry-run, fit-input consistency.
Works on temporary results dirs only; board-shaped rows are generated synthetically here."""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import tables  # noqa: E402
import tables_extra as TX  # noqa: E402
from paperlib import DEFAULT_RESULTS, PH_BOARD, Store, unregistered_numbers  # noqa: E402

META = ["timestamp", "git_commit", "git_dirty", "vivado_version", "bitstream_sha256", "board_id", "net",
        "layer", "clock_mhz", "source", "duration_s", "num_inferences", "data_git_dirty"]


def wcsv(p: Path, rows: list[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def meta(net="lenet5", layer="all", clk="299.997009", source="hw", dirty="False"):
    return dict(timestamp="2026-10-01T10:00:00+00:00", git_commit="abc", git_dirty=dirty, vivado_version="2023.1",
                bitstream_sha256="f" * 64, board_id="kv260", net=net, layer=layer, clock_mhz=clk, source=source,
                duration_s="1800.0", num_inferences="100", data_git_dirty="False")


def board_rows(d: Path, source="hw", dirty="False"):
    soak = []
    for net in ("lenet5", "cifar10"):
        soak.append({**meta(net, source=source, dirty=dirty), "row_kind": "net", "images": "1000", "inf_per_s": "55.5",
                     "logit_mismatch": "0", "cycle_mismatch": "0", "job_errors": "0", "timeouts": "0",
                     "status_errors": "0", "err_flags": "0", "cyc_distinct": "1", "temp_c_max": "48.25",
                     "temp_source": "board_env.read_die_temp (iio ams)", "result": "PASS"})
    soak.append({**meta("all", source=source, dirty=dirty), "row_kind": "all", "images": "2000", "inf_per_s": "111.0",
                 "logit_mismatch": "0", "cycle_mismatch": "0", "job_errors": "0", "timeouts": "0",
                 "status_errors": "0", "err_flags": "0", "temp_c_max": "48.25", "result": "PASS"})
    wcsv(d / "hw_soak.csv", soak)
    spread = []
    for net, layers in (("lenet5", [("conv1", 2829), ("total", 16436)]), ("cifar10", [("conv1", 33629), ("total", 104247)])):
        for lay, c in layers:
            spread.append({**meta(net, lay, source=source, dirty=dirty), "images": "10000", "model_cycles": str(c),
                           "cyc_min": str(c), "cyc_max": str(c), "cyc_distinct": "1", "cyc_spread": "0",
                           "result": "PASS"})
    wcsv(d / "hw_layer_spread.csv", spread)
    files = []
    for clk, p in ((100.0, 1.3), (200.0, 1.5), (299.997009, 1.7)):
        name = f"hw_b2_power_ina260_summary_lenet5_{int(round(clk))}mhz.csv"
        files.append(name)
        base = {**meta(clk=f"{clk:.6f}", source=source, dirty=dirty), "measurement": "SOM-rail power (INA260)",
                "n_repeats": "1"}
        wcsv(d / name, [{**base, "row_kind": "repeat", "repeat": "1", "accel_p_run_w": f"{p:.6f}",
                         "accel_p_idle_w": "1.100000", "accel_dp_w": f"{p - 1.1:.6f}"},
                        {**base, "row_kind": "mean", "repeat": "all", "accel_p_run_w": f"{p:.6f}",
                         "accel_p_idle_w": "1.100000", "accel_dp_w": f"{p - 1.1:.6f}"},
                        {**base, "row_kind": "std", "repeat": "all", "accel_p_run_w": "0.010000",
                         "accel_p_idle_w": "0.001000", "accel_dp_w": "0.010000"}])
    fit = []
    for q, a, k in (("p_accel_w", 1.1, 0.002), ("dp_accel_w", 0.0, 0.002), ("p_idle_w", 1.1, 0.0)):
        fit.append({**meta(layer="fit", clk="", source=source, dirty=dirty), "power_label": "SOM-rail power (INA260)",
                    "quantity": q, "basis": "per_clock_mean", "n_points": "3", "p_static_w": f"{a:.6f}",
                    "p_static_se_w": "0.001000", "p_static_ci95_lo_w": f"{a - 0.0127:.6f}",
                    "p_static_ci95_hi_w": f"{a + 0.0127:.6f}", "k_w_per_mhz": f"{k:.9f}",
                    "k_se_w_per_mhz": "0.000004000", "k_ci95_lo_w_per_mhz": f"{k - 5e-5:.9f}",
                    "k_ci95_hi_w_per_mhz": f"{k + 5e-5:.9f}", "r2": "0.999900" if k else "",
                    "inputs": " ".join(files)})
    wcsv(d / "hw_b2_fit.csv", fit)


@pytest.fixture
def res(tmp_path):
    r = tmp_path / "results"
    r.mkdir()
    shutil.copy(DEFAULT_RESULTS / "rtl_full10k.csv", r / "rtl_full10k.csv")
    return r


def make(res: Path, out: Path, dryrun=False):
    st = Store(res, dryrun=dryrun)
    fails: list = []
    arts = TX.make(tables.Ctx(st, out, booktabs=True, dryrun=dryrun), fails)
    assert not fails, fails
    return {a.name: a for a in arts}


def test_placeholders_without_board_data(res, tmp_path):
    arts = make(res, tmp_path / "gen")
    assert set(arts) == {"tab_soak", "tab_b2_fit", "tab_layer_spread", "fig_b2_fit"}
    for n in arts:
        assert arts[n].placeholders, n
    for n in ("tab_soak", "tab_b2_fit", "tab_layer_spread"):
        assert PH_BOARD in (tmp_path / "gen" / f"{n}.tex").read_text()
    assert (tmp_path / "gen" / "fig_b2_fit.pdf").is_file()
    # the committed RTL-sim reference is used even without board data
    assert "RTL sim: 10,000/10,000" in (tmp_path / "gen" / "tab_layer_spread.tex").read_text()


def test_board_data_switches_automatically(res, tmp_path):
    board_rows(res)
    arts = make(res, tmp_path / "gen")
    for n, a in arts.items():
        assert not a.placeholders, (n, a.placeholders)
        assert a.sources == ({"hw", "rtl_sim"} if n == "tab_layer_spread" else {"hw"})
    t = (tmp_path / "gen" / "tab_b2_fit.tex").read_text()
    assert "1.100" in t and "2.000" in t and "[1.087, 1.113]" in t        # k in mW/MHz, CI
    assert not unregistered_numbers(t, arts["tab_b2_fit"])
    s = (tmp_path / "gen" / "tab_soak.tex").read_text()
    assert "48.2" in s or "48.3" in s
    assert "PASS" in s and "300.0~MHz" in s
    d = list(csv.DictReader(open(tmp_path / "gen" / "fig_b2_fit.data.csv")))
    assert sum(r["kind"] == "fit line" for r in d) == 3 and sum(r["kind"].startswith("hw") for r in d) == 9


def test_dirty_or_dryrun_rows_are_not_board_data(res, tmp_path):
    board_rows(res, dirty="True")
    assert all(a.placeholders for a in make(res, tmp_path / "g1").values())
    board_rows(res, source="dryrun_model")
    assert all(a.placeholders for a in make(res, tmp_path / "g2").values())


def test_dryrun_mode_reads_dryrun_dir_and_watermarks(res, tmp_path):
    board_rows(res / "dryrun", source="dryrun_model", dirty="True")
    arts = make(res, tmp_path / "gen", dryrun=True)
    assert all(not a.placeholders for a in arts.values())
    assert "DRY RUN -- NOT DATA" in (tmp_path / "gen" / "tab_soak.tex").read_text()


def test_figure_plots_only_fit_inputs(res, tmp_path):
    board_rows(res)
    extra = res / "hw_b2_power_ina260_summary_lenet5_150mhz.csv"       # not a fit input
    shutil.copy(res / "hw_b2_power_ina260_summary_lenet5_100mhz.csv", extra)
    rows = list(csv.DictReader(open(extra)))
    for r in rows:
        r["clock_mhz"] = "150.000000"
    wcsv(extra, rows)
    arts = make(res, tmp_path / "gen")
    assert any("not used by hw_b2_fit.csv" in c for c in arts["fig_b2_fit"].checks)
    d = list(csv.DictReader(open(tmp_path / "gen" / "fig_b2_fit.data.csv")))
    assert not any(r.get("clock_mhz") == "150.0" for r in d)


def test_failed_rows_raise_checks(res, tmp_path):
    board_rows(res)
    rows = list(csv.DictReader(open(res / "hw_layer_spread.csv")))
    rows[0]["result"], rows[0]["cyc_max"], rows[0]["cyc_distinct"], rows[0]["cyc_spread"] = "FAIL", "2830", "2", "1"
    wcsv(res / "hw_layer_spread.csv", rows)
    arts = make(res, tmp_path / "gen")
    assert any("result FAIL" in c for c in arts["tab_layer_spread"].checks)


def test_make_raises_without_failures_list(res, tmp_path, monkeypatch):
    monkeypatch.setattr(TX, "ALL_TABLES", [lambda c: 1 / 0])
    with pytest.raises(ZeroDivisionError):
        TX.make(tables.Ctx(Store(res), tmp_path, booktabs=True, dryrun=False))
    fails: list = []
    TX.make(tables.Ctx(Store(res), tmp_path, booktabs=True, dryrun=False), fails)
    assert fails and "ZeroDivisionError" in fails[0]["error"]
