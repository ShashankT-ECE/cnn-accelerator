"""exp_b2_clock: P = P_static + k*f least-squares fit, cycle identity across clocks, dry run."""
import csv
import math

import numpy as np
import pytest

import exp_b2_clock as B


def test_fit_exact_line_two_points():
    f = B.fit_linear([100, 200], [1.2, 1.4])
    assert f["a"] == pytest.approx(1.0) and f["k"] == pytest.approx(0.002) and f["r2"] == pytest.approx(1.0)
    assert math.isnan(f["k_se"]) and f["dof"] == 0


def test_fit_matches_numpy_polyfit_cov():
    rng = np.random.default_rng(1)
    x = np.array([100, 150, 200, 250, 300.0])
    y = 1.1 + 0.0021 * x + rng.normal(0, 0.01, x.size)
    f = B.fit_linear(x, y)
    (k, a), cov = np.polyfit(x, y, 1, cov=True)          # cov scaled by SSE/(n-2)
    assert f["k"] == pytest.approx(k) and f["a"] == pytest.approx(a)
    assert f["k_se"] == pytest.approx(math.sqrt(cov[0, 0])) and f["a_se"] == pytest.approx(math.sqrt(cov[1, 1]))
    assert f["t"] == 3.182 and f["k_lo"] == pytest.approx(k - 3.182 * f["k_se"])
    r2 = 1 - ((y - (a + k * x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert f["r2"] == pytest.approx(r2)


def test_fit_degenerate():
    assert math.isnan(B.fit_linear([100], [1.0])["k"])
    assert math.isnan(B.fit_linear([100, 100, 100], [1, 2, 3])["k"])
    f = B.fit_linear([100, 200, 300], [1, 1, 1])          # flat: k = 0, R^2 undefined
    assert f["k"] == pytest.approx(0) and math.isnan(f["r2"])


def test_t_table():
    assert B.t_crit_95(1) == 12.706 and B.t_crit_95(35) == 2.042 and B.t_crit_95(5000) == 1.96
    assert math.isnan(B.t_crit_95(0))


def _summ(clk, dp, reps=(0.0, 0.0)):
    rows = [{"row_kind": "repeat", "accel_p_run_w": f"{1 + dp + e:.6f}", "accel_dp_w": f"{dp + e:.6f}",
             "accel_p_idle_w": "1.000000"} for e in reps]
    rows.append({"row_kind": "mean", "accel_p_run_w": f"{1 + dp:.6f}", "accel_dp_w": f"{dp:.6f}",
                 "accel_p_idle_w": "1.000000"})
    rows.append({"row_kind": "std", "accel_dp_w": "0.1"})
    return (clk, rows, f"s_{int(clk)}.csv")


def test_fit_points_and_rows():
    pts = B.fit_points([_summ(100, 0.2), _summ(200, 0.4), _summ(300, 0.6, (0.01, -0.01))])
    xs, ys, files = pts[("dp_accel_w", "per_clock_mean")]
    assert xs == [100, 200, 300] and ys == pytest.approx([0.2, 0.4, 0.6]) and len(files) == 3
    assert len(pts[("dp_accel_w", "per_repeat")][0]) == 6
    rows = B.fit_rows(pts, lambda q, b, n: {"num_inferences": n})
    assert len(rows) == 6
    r = next(r for r in rows if r["quantity"] == "dp_accel_w" and r["basis"] == "per_clock_mean")
    assert float(r["k_w_per_mhz"]) == pytest.approx(0.002) and float(r["p_static_w"]) == pytest.approx(0.0, abs=1e-6)
    assert float(r["k_mw_per_mhz"]) == pytest.approx(2.0) and r["n_points"] == 3 and r["power_label"] == "SOM-rail power (INA260)"
    assert r["clocks_mhz"] == "100.000000 200.000000 300.000000" and r["note"] == ""
    ri = next(r for r in rows if r["quantity"] == "p_idle_w" and r["basis"] == "per_clock_mean")
    assert float(ri["k_w_per_mhz"]) == pytest.approx(0) and ri["r2"] == ""
    rows2 = B.fit_rows(B.fit_points([_summ(100, 0.2), _summ(200, 0.4)]), lambda q, b, n: {})
    assert "no standard error" in rows2[0]["note"] and rows2[0]["p_static_se_w"] == ""


def _crows(layers, model, distinct=1, eq=None, n=10):
    out = [{"layer": f"L{i}", "images": n, "model_cycles": m, "hw_cycles": v, "hw_min": v, "hw_max": v,
            "hw_distinct": distinct, "hw_images_equal_model": n if eq is None else eq}
           for i, (v, m) in enumerate(zip(layers, model))]
    return out


def test_cycle_identity_pass_and_fail():
    m = [10, 20]
    rows, per, ok = B.cycle_identity([(100.0, 100.0, _crows(m, m)), (200.0, 199.998, _crows(m, m))])
    assert ok and per == {100.0: (True, "PASS"), 200.0: (True, "PASS")}
    assert all(r["cycle_check"] == "PASS" and r["clocks_compared"] == 2 for r in rows)
    # one layer differs at one clock -> that layer FAILs at every clock (not identical across clocks)
    rows, per, ok = B.cycle_identity([(100.0, 100.0, _crows(m, m)), (200.0, 200.0, _crows([10, 21], m))])
    assert not ok and per[100.0] == (False, "FAIL") and per[200.0] == (False, "FAIL")
    assert [r["cycle_check"] for r in rows] == ["PASS", "FAIL", "PASS", "FAIL"]
    # nondeterministic within a clock
    _, per, ok = B.cycle_identity([(100.0, 100.0, _crows(m, m, distinct=2, eq=9))])
    assert not ok and per[100.0][1] == "FAIL"


def test_dry_run_sweep_fit_and_refit(tmp_path):
    out = tmp_path / "dryrun"
    rc = B.main(["--backend", "model", "--allow-dirty", "--clock-mhz", "249.997498", "--max-mhz", "249.997498",
                 "--images", "3", "--window-s", "0.3", "--power-repeats", "1", "--sensor", "mock",
                 "--mock-slope-w-per-mhz", "0.002", "--out-dir", str(out)])
    assert rc == 0
    clk = list(csv.DictReader(open(out / "hw_b2_clock.csv")))
    assert [float(r["clock_requested_mhz"]) for r in clk] == [99.999, 111.11, 124.99875, 142.855714,
                                                                166.665, 199.998, 249.9975]
    assert all(r["clock_readback_equal"] == "True" for r in clk)      # read-back == requested
    assert all(r["cycle_check"] == "PASS" and r["cycles_identical_across_clocks"] == "True" for r in clk)
    cyc = list(csv.DictReader(open(out / "hw_b2_cycles.csv")))
    assert len(cyc) == 7 * 6 and all(r["cycle_check"] == "PASS" for r in cyc)
    fit = list(csv.DictReader(open(out / "hw_b2_fit.csv")))
    r = next(r for r in fit if r["quantity"] == "p_accel_w" and r["basis"] == "per_clock_mean")
    assert float(r["k_w_per_mhz"]) == pytest.approx(0.002, rel=1e-3) and r["n_points"] == "7"
    assert r["source"] == "dryrun_model" and r["layer"] == "fit"
    rc = B.main(["--fit-only", "--in-dir", str(out), "--out-dir", str(out / "refit")])
    fit2 = list(csv.DictReader(open(out / "refit" / "hw_b2_fit.csv")))
    assert rc == 0 and [x["k_w_per_mhz"] for x in fit2] == [x["k_w_per_mhz"] for x in fit]


def test_mock_slope_refused_on_hardware_paths(tmp_path):
    with pytest.raises(SystemExit):   # model backend but not the mock sensor
        B.main(["--backend", "model", "--allow-dirty", "--clock-mhz", "200", "--max-mhz", "200",
                "--sensor", "auto", "--mock-slope-w-per-mhz", "0.1", "--window-s", "0.1",
                "--out-dir", str(tmp_path / "dryrun")])
    with pytest.raises(SystemExit):
        B.main(["--fit-only", "--in-dir", str(tmp_path / "empty"), "--out-dir", str(tmp_path / "dryrun")])


def test_sweep_aborts_when_a_clock_is_not_reached(tmp_path, monkeypatch):
    """The divider-computing setter refuses / the read-back differs -> abort, nothing run there."""
    import gos_driver as D
    import gos_model_backend as MB

    def stuck(self, mhz, tol=0.1):
        raise D.GosError(f"pl_clk0 {mhz:.6f} MHz is not reachable on this board")
    monkeypatch.setattr(MB.ModelBackend, "set_fclk0_exact", stuck, raising=False)
    with pytest.raises(SystemExit, match="ABORT at .*not reachable"):
        B.main(["--backend", "model", "--allow-dirty", "--clock-mhz", "249.997498", "--max-mhz", "249.997498",
                "--images", "2", "--window-s", "0.2", "--power-repeats", "1", "--sensor", "mock",
                "--out-dir", str(tmp_path / "dryrun")])


def test_fit_with_temp_recovers_known_coefficients():
    f = np.array([100.0, 111.0, 125.0, 143.0, 167.0, 200.0, 250.0])
    t = np.array([33.0, 36.0, 34.0, 38.0, 35.0, 37.0, 39.0])           # not collinear with f
    y = 3.0 + 0.001 * f + 0.01 * (t - t.mean())
    g = B.fit_with_temp(f, t, y)
    assert g["a"] == pytest.approx(3.0) and g["k"] == pytest.approx(0.001) and g["c"] == pytest.approx(0.01)
    assert g["dof"] == 4 and g["t"] == 2.776 and g["r2"] == pytest.approx(1.0)
    assert g["t_mean"] == pytest.approx(t.mean()) and g["t_range"] == pytest.approx(6.0)
    assert -1 <= g["corr"] <= 1


def test_fit_with_temp_removes_a_pure_temperature_confound():
    """Ascending sweep: T rises with f. Power depends on T only (k = 0). The plain fit reports k > 0,
    the covariate fit k ~ 0 once T varies independently of f at all."""
    f = np.array([100.0, 111.0, 125.0, 143.0, 167.0, 200.0, 250.0])
    t = np.array([30.0, 33.0, 31.0, 34.0, 32.0, 35.0, 36.5])
    y = 3.0 + 0.02 * (t - 30.0)
    plain = B.fit_linear(f, y)
    cov = B.fit_with_temp(f, t, y)
    assert plain["k"] > 5e-5 and cov["k"] == pytest.approx(0.0, abs=1e-9) and cov["c"] == pytest.approx(0.02)


def test_fit_with_temp_degenerate():
    f = [100.0, 150.0, 200.0]
    assert math.isnan(B.fit_with_temp(f, [30, 31, 32], [1, 2, 3])["k"])               # n < 4
    assert math.isnan(B.fit_with_temp(f + [250.0], [30, 30, 30, 30], [1, 2, 3, 4])["k"])   # T constant


def _summ_t(clk, temp):
    rows = [{"row_kind": "repeat", "accel_p_run_w": "1.2", "accel_dp_w": "0.2", "accel_p_idle_w": "1.0",
             "accel_temp_pl_c": f"{temp}", "accel_idle_temp_pl_c": f"{temp - 1}"}]
    rows.append({**rows[0], "row_kind": "mean"})
    return (clk, rows, f"s_{int(clk)}.csv")


def test_fit_rows_add_temperature_basis_only_with_temps():
    clks = (100, 140, 180, 220, 260)
    temps = (30.0, 33.5, 31.0, 35.0, 32.0)
    rows = B.fit_rows(B.fit_points([_summ_t(c, t) for c, t in zip(clks, temps)]), lambda q, b, n: {})
    tr = [r for r in rows if r["basis"] == "per_clock_mean_temp"]
    assert len(tr) == 3 and all(r["dof"] == 2 and r["temp_covariate"] for r in tr)
    assert {r["temp_covariate"] for r in tr} == {"accel_temp_pl_c", "accel_idle_temp_pl_c"}
    assert all(r["c_w_per_c"] != "" and r["corr_f_temp"] != "" and r["temp_range_c"] == "5.000" for r in tr)
    assert len(rows) == 9
    # no temperature columns -> only the 6 plain rows
    assert len(B.fit_rows(B.fit_points([_summ(100, 0.2), _summ(200, 0.4), _summ(300, 0.6)]), lambda q, b, n: {})) == 6


def test_dry_run_records_seeded_random_order(tmp_path):
    def run(seed, order="random"):
        out = tmp_path / "dryrun" / f"{seed}_{order}"
        assert B.main(["--backend", "model", "--allow-dirty", "--clock-mhz", "249.997498", "--max-mhz", "249.997498",
                       "--images", "2", "--window-s", "0.2", "--power-repeats", "1", "--sensor", "mock",
                       "--order", order, "--order-seed", str(seed), "--out-dir", str(out)]) == 0
        return list(csv.DictReader(open(out / "hw_b2_clock.csv")))
    a, b, asc = run(11), run(11), run(11, "ascending")
    assert [r["sweep_order"] for r in a] == [r["sweep_order"] for r in b]          # same seed, same order
    assert a[0]["order_seed"] == "11" and sorted(int(r["sweep_pos"]) for r in a) == list(range(1, 8))
    assert [float(r["clock_requested_mhz"]) for r in a] == sorted(float(r["clock_requested_mhz"]) for r in a)
    assert a[0]["sweep_order"] != asc[0]["sweep_order"]
    assert [int(r["sweep_pos"]) for r in asc] == list(range(1, 8))
