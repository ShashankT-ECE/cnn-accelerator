"""Measurement rigor: stats.py, board_env.py (fake sysfs/proc trees), exp_fclk_cal (fake device +
fake clock), run_sessions environment pre-flight / pinning / $GOS_RUN_ENV / priority / extra steps
/ clock fallback wiring, aggregate_sessions.py, power_log random phase order. No hardware."""
import csv
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import aggregate_sessions as AG
import board_common as bc
import board_env as benv
import exp_fclk_cal as FC
import power_log as pl
import run_sessions as RS
import stats
from test_sessions import FakeBE, env, run, state  # noqa: F401 (fixture)


# ---- stats -------------------------------------------------------------------------------------
@pytest.mark.parametrize("n, lo, hi", [(6, 1, 6), (10, 2, 9), (100, 40, 61), (1000, 469, 532)])
def test_median_ci_order_statistics_textbook(n, lo, hi):
    l_, h_, cov = stats.order_stat_ci_indices(n)
    assert (l_, h_) == (lo, hi) and cov >= 0.95


def test_median_ci_small_n_and_coverage():
    assert stats.order_stat_ci_indices(5) is None
    assert stats.median_ci([1, 2, 3]) == (None, None, None)
    rng = np.random.default_rng(3)
    hit = sum(lo <= math.log(2) < hi for lo, hi, _ in
              (stats.median_ci(rng.exponential(size=100)) for _ in range(1000)))
    assert 0.93 <= hit / 1000 <= 0.99            # nominal >= 0.95 (discrete: 0.965)


def test_summarize_and_warmup():
    x = np.r_[np.full(10, 1e6), np.arange(200.0)]
    s = stats.summarize(stats.discard_warmup(x, 10))
    assert s["n"] == 200 and s["median"] == pytest.approx(99.5) and s["repeats_ok"]
    assert s["ci_lo"] <= s["median"] <= s["ci_hi"] and s["p95"] < 200
    assert not stats.summarize(np.arange(50.0))["repeats_ok"]


def test_block_order_randomized_abab_reproducible():
    o = stats.block_order(["safe", "fast", "cpu"], 20, seed=5)
    assert o == stats.block_order(["safe", "fast", "cpu"], 20, seed=5)
    for r in range(20):
        assert sorted(o[3 * r:3 * r + 3]) == ["cpu", "fast", "safe"]
    assert len({tuple(o[3 * r:3 * r + 3]) for r in range(20)}) > 1
    ab = stats.block_order(["a", "b"], 3, seed=1, mode="abab")
    assert ab in (["a", "b"] * 3, ["b", "a"] * 3)


def test_between_sessions():
    b = stats.between_sessions([10.0, 11.0, 12.0])
    assert b["mean"] == 11.0 and b["sd"] == pytest.approx(1.0) and b["range"] == 2.0
    assert b["cv_pct"] == pytest.approx(100 / 11)


# ---- board_env ---------------------------------------------------------------------------------
def test_parse_cores_roundtrip():
    assert benv.parse_cores("0-2,5") == {0, 1, 2, 5} and benv.cores_str({0, 1, 2, 5}) == "0-2,5"


def test_governor_apply_verify_restore(tmp_path):
    p = benv.make_fake_tree(tmp_path)
    ctl = benv.CpuFreqControl(root=p["sysfs_root"], fake=True)
    r = ctl.apply(settle=lambda: benv.fake_kernel_settle(p["sysfs_root"]))
    assert r["ok"] and r["target_khz"] == max(benv.KV260_FREQS_KHZ)
    pol = Path(p["sysfs_root"]) / "cpufreq" / "policy0"
    assert (pol / "scaling_governor").read_text().strip() == "performance"
    assert (pol / "scaling_min_freq").read_text().strip() == str(r["target_khz"])
    assert ctl.restore() == []
    assert (pol / "scaling_governor").read_text().strip() == "schedutil"
    assert (pol / "scaling_min_freq").read_text().strip() == str(min(benv.KV260_FREQS_KHZ))


def test_governor_readback_mismatch_fails(tmp_path):
    p = benv.make_fake_tree(tmp_path)
    r = benv.CpuFreqControl(root=p["sysfs_root"]).apply()      # no kernel emulation: cur stays low
    assert not r["ok"] and any("scaling_cur_freq" in e for e in r["errors"])
    r2 = benv.CpuFreqControl(root=str(tmp_path / "none")).apply()
    assert not r2["ok"] and "cpufreq driver absent" in r2["errors"][0]


def test_pkg_manager_busy_detection(tmp_path):
    p = benv.make_fake_tree(tmp_path, busy=["apt-get", "packagekitd"])
    busy = benv.pkg_manager_busy(p["proc_root"])
    assert {b["comm"] for b in busy} == {"apt-get", "packagekitd"}   # shutdown helper ignored
    q = benv.make_fake_tree(tmp_path / "idle")
    assert benv.pkg_manager_busy(q["proc_root"]) == []


def test_die_temp_iio_hwmon_unavailable(tmp_path):
    p = benv.make_fake_tree(tmp_path)
    t = benv.read_die_temp(p["iio_root"], p["hwmon_root"])
    assert t["source"].startswith("iio") and set(t["channels_c"]) == {"ps_temp", "remote_temp", "pl_temp"}
    assert t["channels_c"]["ps_temp"] == pytest.approx((41000 - 36058) * 7.771514892 / 1000, abs=1e-3)
    hw = tmp_path / "hw" / "hwmon3"
    hw.mkdir(parents=True)
    (hw / "name").write_text("ams\n")
    (hw / "temp1_input").write_text("52500\n")
    t2 = benv.read_die_temp(str(tmp_path / "noiio"), str(tmp_path / "hw"))
    assert t2["max_c"] == 52.5 and t2["source"].startswith("hwmon")
    assert benv.read_die_temp(str(tmp_path / "x"), str(tmp_path / "y"))["source"] == "unavailable"


def test_pin_readback():
    orig = benv.affinity_of(0)
    try:
        r = benv.pin(0, {0})
        assert r["ok"] and r["readback"] == "0"
    finally:
        benv.pin(0, orig)


def test_env_preflight_fail_and_override(tmp_path):
    p = benv.make_fake_tree(tmp_path, busy=["dpkg"])
    paths = RS.EnvPaths(**p, fake=True)
    kw = dict(plan=False, allow_dirty=False, meas_cores="0", cpu1_cores="0", cpun_cores="0",
              cpu_freq_khz=None, say=lambda m: None)
    orig = benv.affinity_of(0)
    try:
        with pytest.raises(RS.PreflightError, match="package manager running"):
            RS.env_preflight(paths, dry=False, allow_non_paper=False, **kw)
        envp, ctl = RS.env_preflight(paths, dry=False, allow_non_paper=True, **kw)
        assert envp["paper_grade"] is False and envp["pkg_manager_offenders"]
        ctl.restore()
        q = benv.make_fake_tree(tmp_path / "ok")
        envp, ctl = RS.env_preflight(RS.EnvPaths(**q, fake=True), dry=False, allow_non_paper=False, **kw)
        assert envp["paper_grade"] is True and envp["cpu_governor"] == "performance"
        ctl.restore()
        with pytest.raises(RS.PreflightError, match="not a subset"):
            RS.env_preflight(RS.EnvPaths(**q, fake=True), dry=False, allow_non_paper=False,
                             **{**kw, "meas_cores": "4096"})
    finally:
        benv.pin(0, orig)


# ---- f_meas calibration ----------------------------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        self.t += 50                     # every time stamp costs 50 ns
        return self.t


class FakeCalDev:
    """KV260-like timing: the job ends C / f after the start write; done is only seen on the
    STATUS poll grid (period poll_ns), optionally phase-shifted by the dither wait. Deterministic
    cycle counts + a fixed grid = a fixed, net-dependent detection latency (the 2026-09-30 bug)."""

    def __init__(self, clock, f_mhz=249.9975, start_ns=1500.0, poll_ns=3500.0, read_ns=900.0,
                 jitter_ns=40.0, seed=0):
        self.clock, self.f, self.start, self.poll, self.read, self.j = \
            clock, f_mhz, start_ns, poll_ns, read_ns, jitter_ns
        self.rng = np.random.default_rng(seed)
        self.cyc = None

    def load_net(self, pkg):
        self.cyc = pkg.cycles

    def poll_period_ns(self, clock_ns=None):
        return self.poll

    def timed_job(self, x, clock_ns=None, dither_ns=0, rng=None):
        t0 = clock_ns()
        end = self.start + self.cyc / self.f * 1e3               # true end, relative to t0
        first = self.start + (rng.integers(0, dither_ns) if dither_ns else 0) + self.rng.uniform(0, self.j)
        n = max(0, int(np.ceil((end - first) / self.poll)))      # polls that still see busy
        seen = first + n * self.poll + self.read
        self.clock.t += int(seen)
        t1 = clock_ns()
        self.clock.t += 20000            # host work between jobs
        return t1 - t0, self.cyc, n + 1


def _pkg(c):
    return types.SimpleNamespace(cycles=c, x_act=np.zeros((4, 8), np.int8))


def _cal(dither_ns=None, seed=7):
    clk = FakeClock()
    return FC.calibrate(FakeCalDev(clk), {"lenet5": _pkg(16436), "cifar10": _pkg(104247)},
                        seconds=1.0, rounds=2, warmup=3, seed=seed, clock_ns=clk,
                        say=lambda m: None, dither_ns=dither_ns)


def test_fclk_cross_check_recovers_clock_with_dither():
    r = _cal()
    assert r["net_a"] == "lenet5" and r["jobs_a"] > 100 and r["jobs_b"] > 100
    assert r["dither_ns"] == 7000 and r["n_folds"] == FC.N_FOLDS
    assert r["f_mhz"] == pytest.approx(249.9975, rel=3e-4)
    assert r["ci_lo_mhz"] < r["ci_hi_mhz"] and r["fold_min_mhz"] <= r["fold_max_mhz"]
    rel, flag = FC.agreement(r["f_mhz"], 249.9975)
    assert abs(rel) < FC.AGREE_TOL and flag is False
    assert r["f_simple_mhz"] < r["f_mhz"]                         # naive estimate biased low
    assert r["intercept_a_ns"] == pytest.approx(r["intercept_b_ns"], abs=150)
    assert r["cycles_constant"] and len(r["order"]) == 4


def test_fclk_median_estimator_is_biased_by_the_poll_grid():
    """The former method: without dither the detection latency is a fixed phase of the poll grid
    per net, the medians do not cancel it and the estimate is off by ~1 % (KV260, 2026-09-30)."""
    r = _cal(dither_ns=0)
    assert abs(r["f_median_est_mhz"] / 249.9975 - 1) > 2e-3
    rel, flag = FC.agreement(r["f_median_est_mhz"], 249.9975)
    assert flag is True


def test_fclk_agreement_flag_threshold():
    assert FC.agreement(200.19, 199.998)[1] is False              # +0.096 %
    assert FC.agreement(200.21, 199.998)[1] is True               # +0.106 %
    assert FC.agreement(201.992064, 199.998)[1] is True           # the first board run (+1.0 %)
    assert FC.agreement(float("nan"), 199.998)[1] is True


def test_readback_is_the_clock_of_record(monkeypatch):
    monkeypatch.setenv(bc.RUN_ENV_VAR, json.dumps({"f_meas_mhz": 199.9, "f_meas_readback_mhz": 199.998001,
                                                   "f_meas_source": "cross-check", "f_meas_step": "s2.fcal",
                                                   "session_index": 2, "paper_grade": True}))
    ctx = bc.RunContext.__new__(bc.RunContext)
    ctx.run_env = bc.run_env()
    assert ctx.f_used(199.998001) == (199.998001, bc.F_USED_SOURCE)      # never f_meas
    f, src = ctx.f_used(100.0)                              # B2 at another clock
    assert f == 100.0 and src.startswith("f_readback")
    cols = ctx.clock_cols(199.998001)
    assert cols["f_used_mhz"] == "199.998001" and cols["f_readback_mhz"] == "199.998001"
    assert cols["f_meas_mhz"] == "199.900000"               # recorded beside it as the cross-check
    assert ctx.clock_cols(100.0)["f_meas_mhz"] == ""        # no cross-check at that clock


# ---- power_log random order --------------------------------------------------------------------
def test_power_random_order_bracketed_and_recorded():
    d = {k: 1.0 for k in pl.PHASE_KINDS}
    sch = pl.build_schedule(3, d, with_cpu=True, with_control=True, order_seed=11)
    assert len(sch) == pl.n_phases(3, True, True) == 19
    for i, p in enumerate(sch):
        if p["kind"] != "idle":
            assert sch[i - 1]["kind"] == "idle" and sch[i + 1]["kind"] == "idle"
    orders = {tuple(p["kind"] for p in sch if p["repeat"] == r and p["kind"] != "idle") for r in (1, 2, 3)}
    assert all(sorted(o) == ["accel", "control", "cpu"] for o in orders)
    assert sch == pl.build_schedule(3, d, with_cpu=True, with_control=True, order_seed=11)
    fixed = pl.build_schedule(3, d, with_cpu=True, with_control=True)
    assert [p["phase"] for p in fixed[:6]] == ["idle_pre", "accel", "idle_mid", "control", "idle_ctl", "cpu"]


# ---- orchestrator: env, pinning, run env, priority, extras, fallback ---------------------------
ENV_STEP = """
import json, os, pathlib, sys
a = sys.argv[1:]
out = pathlib.Path(a[a.index("--out-dir") + 1])
e = json.loads(os.environ["GOS_RUN_ENV"])
e["affinity"] = sorted(os.sched_getaffinity(0))
(out / f"hw_{a[0]}.csv").write_text("source\\ndryrun_model\\n")
(out / f"hw_{a[0]}_env.json").write_text(json.dumps(e))
"""


def _env_hook(env, names=("x",), group="A1"):
    script = env["tmp"] / "env_step.py"
    script.write_text(ENV_STEP)

    def hook(steps, cfg):
        return [RS.Step(f"s2.{n}", 2, n, [str(script), n], kind="fixed", fixed_s=1,
                        expect=(f"hw_{n}.csv",), group=group) for n in names]
    return hook


def test_run_env_pinning_and_temperature_recorded(env):
    assert run(env, "--session-index", "1", "--meas-cores", "1", hook=_env_hook(env)) == 0
    e = json.loads((env["rd"] / "hw_x_env.json").read_text())
    assert e["affinity"] == [1] and e["session_index"] == 1 and e["paper_grade"] is False
    assert e["cpu_governor"] == "performance" and e["step_id"] == "s2.x"
    assert isinstance(e["die_temp_start_c"], float)
    st = state(env)
    rec = st["steps"]["s2.x"]
    assert rec["affinity"]["requested"] == "1" and rec["die_temp_end"]["max_c"] is not None
    assert st["provenance"]["session_index"] == 1 and st["provenance"]["env"]["fake_tree"]
    pol = env["rd"] / ".dryrun_env" / "sys" / "devices" / "system" / "cpu" / "cpufreq" / "policy0"
    assert (pol / "scaling_governor").read_text().strip() == "schedutil"      # restored
    assert (env["rd"] / "logs" / "env_log.jsonl").is_file()


def test_session_index_goes_to_rep_dir_and_refuses_mixing(env):
    assert run(env, "--session-index", "2", hook=_env_hook(env)) == 0
    assert (env["rd"] / "rep2" / "hw_x.csv").is_file() and not (env["rd"] / "hw_x.csv").exists()
    e = json.loads((env["rd"] / "rep2" / "hw_x_env.json").read_text())
    assert e["session_index"] == 2


def test_priority_order():
    mk = lambda i, s, g: RS.Step(i, s, i, ["x.py"], group=g)  # noqa: E731
    steps = [mk("s3.soak", 3, "soak"), mk("s2.A4", 2, "A4"), mk("s3.B1", 3, "energy"),
             mk("s2.CPU", 2, "baselines"), mk("s2.B3", 2, "latency"), mk("s2.A1", 2, "A1"),
             mk("s2.A2A3", 2, "A3"), mk("s2.layerspread", 2, "A3"), mk("s2.fcal", 2, "fcal"),
             mk("s3.B2", 3, "sweep"), mk("s1.smoke", 1, "bringup"), mk("s1.fcal", 1, "fcal")]
    got = [s.id for s in RS.prioritize(steps)]
    assert got == ["s1.smoke", "s1.fcal", "s2.fcal", "s2.A2A3", "s2.layerspread", "s2.A1", "s2.B3",
                   "s2.CPU", "s3.B1", "s3.B2", "s3.soak", "s2.A4"]


def test_extra_steps_merged_with_absent_and_present_module(env, monkeypatch):
    cfg = RS.Config(backend="model", board_dir=env["tmp"], results_dir=env["rd"], data_dir=env["data"],
                    deploy={"bit_clock_mhz": 199.998001}, bit=env["bit"])
    base = RS.build_steps(cfg, [2, 3])
    monkeypatch.setitem(sys.modules, "session_extra_steps", None)          # absent
    assert RS.merge_extra_steps(base, {}, [2, 3], say=lambda m: None) == base
    fake = types.ModuleType("session_extra_steps")
    fake.extra_steps = lambda opts: [
        {"id": "s3.soak", "session": 3, "group": "soak", "cmd": ["exp_soak.py", "--x"],
         "outputs": ["hw_soak.csv"], "est_s": 70.0, "timeout_s": 90.0, "requires": [],
         "params": {"d": 60}},
        {"id": "s3.B2", "session": 3, "group": "sweep", "cmd": ["exp_b2_clock.py"], "outputs": [],
         "est_s": 5.0, "timeout_s": 9.0, "requires": [], "params": {}},
        {"id": "s1.x", "session": 1, "group": "A1", "cmd": ["y.py"], "outputs": [], "est_s": 1,
         "timeout_s": 1, "requires": [], "params": {}}]
    monkeypatch.setitem(sys.modules, "session_extra_steps", fake)
    out = {s.id: s for s in RS.merge_extra_steps(base, {}, [2, 3], say=lambda m: None)}
    assert "s1.x" not in out and out["s3.B2"].argv == ["exp_b2_clock.py"]     # replaced
    soak = out["s3.soak"]
    assert soak.est_s == 70.0 and soak.timeout_s == 90.0 and soak.params == {"d": 60}
    assert RS.estimate(soak, {"timing": {}}, "model") == (70.0, "step estimate (extra step)")
    assert "params" in soak.params_key


def test_real_extra_steps_module_if_present(env):
    try:
        import session_extra_steps  # noqa: F401
    except ImportError:
        pytest.skip("session_extra_steps.py not present")
    cfg = RS.Config(backend="model", board_dir=env["tmp"], results_dir=env["rd"], data_dir=env["data"],
                    deploy={"bit_clock_mhz": 199.998001}, bit=env["bit"])
    a = RS.make_parser().parse_args(["all", "--backend", "model"])
    steps = RS.merge_extra_steps(RS.build_steps(cfg, [1, 2, 3]),
                                 RS.extra_opts(cfg, a, [1, 2, 3], 2.0, 1.0, 1), [1, 2, 3],
                                 say=lambda m: None)
    ids = [s.id for s in RS.prioritize(steps)]
    assert ids.index("s2.A2A3") < ids.index("s2.layerspread") < ids.index("s2.A1")
    assert ids.index("s3.B2") < ids.index("s3.soak") < ids.index("s2.A4")
    b2 = next(s for s in steps if s.id == "s3.B2")
    assert "hw_b2_cycles.csv" in b2.expect and "hw_b2_fit.csv" in b2.expect


def _second_bit(env, name="gos_300", clk=299.997009):
    import hashlib
    d = env["tmp"] / name
    d.mkdir()
    b = d / f"{name}.bit"
    b.write_bytes(b"fake " + name.encode())
    (d / f"{name}.hwh").write_text("<hwh/>")
    (d / f"{name}.bit.sha256").write_text(hashlib.sha256(b.read_bytes()).hexdigest() + "\n")
    (d / "summary.json").write_text(json.dumps({
        "build_id": "c88e71a0", "pl_clk0_mhz_actual": clk, "pl_clk0_mhz_requested": round(clk),
        "wns_ns": 0.1, "whs_ns": 0.01, "failing_setup_endpoints": 0, "failing_hold_endpoints": 0}))
    return b


@pytest.mark.parametrize("fail300, want", [(True, "gos_200"), (False, "gos_300")])
def test_clock_fallback_wiring(env, monkeypatch, fail300, want):
    b300 = _second_bit(env)
    fake = types.ModuleType("clock_fallback")
    calls = []

    def choose(ctx, bits, run_smoke):
        assert [sorted(b) for b in bits] == [["bit", "clock_mhz"]] * 2
        assert bits[0]["clock_mhz"] > bits[1]["clock_mhz"]
        for b in bits:
            calls.append(b["clock_mhz"])
            if run_smoke(b):
                return {"ok": True, "bit": b["bit"], "clock_mhz": b["clock_mhz"],
                        "fell_back": b is not bits[0], "reason": "test", "attempts": list(calls)}
        return {"ok": False, "bit": None, "reason": "all failed", "attempts": calls}
    fake.choose = choose
    monkeypatch.setitem(sys.modules, "clock_fallback", fake)
    monkeypatch.setattr(RS, "_smoke_runner", lambda *a, **k: (lambda e: not (fail300 and e["clock_mhz"] > 250)))
    hook = _env_hook(env)
    argv = ["1", "--backend", "model", "--fallback-bits", str(b300), str(env["bit"]), "--data-dir",
            str(env["data"]), "--results-dir", str(env["rd"]), "--min-free-mb", "1"]
    kw = dict(open_backend=lambda cfg: FakeBE(clk=cfg.closed_mhz),
              verify_data=lambda d: json.loads((Path(d) / "MANIFEST.json").read_text()))
    assert RS.main(argv, steps_hook=lambda s, c: [RS.Step("s1.x", 1, "x", hook(s, c)[0].argv,
                                                          kind="fixed", expect=("hw_x.csv",))], **kw) == 0
    st = state(env)
    assert Path(st["provenance"]["bit"]).stem == want
    assert st["provenance"]["bit_choice"]["fell_back"] is fail300
    ch = json.loads((env["rd"] / RS.CHOICE_FILE).read_text())
    assert Path(ch["bit"]).stem == want
    # a later invocation (session 2) uses the recorded choice without re-running the smoke
    calls.clear()
    argv2 = ["2", *argv[1:], "--no-bringup-check"]
    assert RS.main(argv2, steps_hook=lambda s, c: [], **kw) in (0, 5)
    assert calls == [] and Path(state(env)["provenance"]["bit"]).stem == want


def test_fallback_module_absent_uses_primary(env, monkeypatch):
    b300 = _second_bit(env)
    monkeypatch.setitem(sys.modules, "clock_fallback", None)
    argv = ["1", "--backend", "model", "--fallback-bits", str(env["bit"]), str(b300), "--data-dir",
            str(env["data"]), "--results-dir", str(env["rd"]), "--min-free-mb", "1"]
    rc = RS.main(argv, open_backend=lambda cfg: FakeBE(clk=cfg.closed_mhz),
                 verify_data=lambda d: json.loads((Path(d) / "MANIFEST.json").read_text()),
                 steps_hook=lambda s, c: [])
    assert rc == 0
    pv = state(env)["provenance"]
    assert Path(pv["bit"]).stem == "gos_300" and "absent" in pv["bit_choice"]["reason"]


# ---- aggregation ---------------------------------------------------------------------------------
def _write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def test_aggregate_sessions_never_mixes_bitstreams(tmp_path):
    root = tmp_path / "dryrun"
    base = {"source": "dryrun_model", "git_dirty": "False", "git_commit": "abc", "paper_grade": "False",
            "net": "lenet5", "clock_mhz": "199.998001", "bitstream_sha256": "A", "timestamp": "t"}
    for k, v in ((1, 100.0), (2, 102.0), (3, 104.0)):
        d = root if k == 1 else root / f"rep{k}"
        rows = [{**base, "session_index": k, "phase": "end_to_end", "median_us": v, "p95_us": v + 5}]
        if k == 3:
            rows.append({**base, "session_index": k, "phase": "end_to_end", "median_us": 50.0,
                         "p95_us": 55.0, "bitstream_sha256": "B", "clock_mhz": "299.997009"})
        _write(d / "hw_b3_breakdown.csv", rows)
    rows, src, _ = AG.aggregate(root)
    med = [r for r in rows if r["metric"] == "median_us"]
    a = next(r for r in med if r["bitstream_sha256"] == "A")
    assert a["n_sessions"] == 3 and float(a["mean"]) == pytest.approx(102.0)
    assert float(a["between_sd"]) == pytest.approx(2.0) and src == bc.SOURCE_DRYRUN
    b = next(r for r in med if r["bitstream_sha256"] == "B")
    assert b["n_sessions"] == 1 and "never combined" in b["note"]
    assert AG.main(["--root", str(root)]) == 0 and (root / "hw_repeatability.csv").is_file()


def test_aggregate_refuses_wrong_session_index(tmp_path):
    root = tmp_path / "dryrun"
    _write(root / "rep2" / "hw_b3_breakdown.csv",
           [{"source": "dryrun_model", "session_index": 3, "net": "lenet5", "phase": "x", "median_us": 1}])
    with pytest.raises(SystemExit, match="session_index"):
        AG.aggregate(root)


def test_deploy_info_follows_the_loaded_bitstream(capsys):
    """Fallback to the 250 MHz build: no SHA warning, per-bitstream fields of the loaded one."""
    info = {"bit": "bit/gos_300/gos_300.bit", "bit_sha256": "a" * 64, "bit_clock_mhz": 299.997009,
            "bit_clock_requested_mhz": 300, "commit": "abc",
            "bits": [{"bit": "bit/gos_300/gos_300.bit", "bit_sha256": "a" * 64, "bit_clock_mhz": 299.997009,
                      "bit_clock_requested_mhz": 300},
                     {"bit": "bit/gos_250/gos_250.bit", "bit_sha256": "b" * 64, "bit_clock_mhz": 249.997498,
                      "bit_clock_requested_mhz": 250}]}
    got = bc.deploy_info_for_sha(info, "b" * 64)
    assert got["bit_sha256"] == "b" * 64 and got["bit_clock_mhz"] == 249.997498
    assert got["bit_clock_requested_mhz"] == 250 and got["commit"] == "abc"
    assert bc.deploy_info_for_sha(info, "c" * 64) is info          # unknown bitstream: still warned
