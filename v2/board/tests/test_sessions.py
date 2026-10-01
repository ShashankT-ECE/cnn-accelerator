"""run_sessions.py: resume / interrupted rerun / budget deferral / pre-flight / B2 sweep list.

No hardware and no real experiment: the orchestrator runs tiny fake step scripts (tmp_path) and
talks to a fake backend (VERSION / BUILD_ID / pl_clk0 only). Everything writes under tmp dirs
named 'dryrun' (the model-output rule)."""
import hashlib
import json
import sys
import textwrap
from types import SimpleNamespace
from pathlib import Path

import pytest

import board_common as bc
import gos_driver as D
import run_sessions as RS

BUILD = "c88e71a0"
CLOSED = 199.998001


# ---- fixtures ----------------------------------------------------------------------------------
class FakeCsr:
    def __init__(self, version, build_id):
        self.regs = {RS.OFF_VERSION: version, RS.OFF_BUILD_ID: build_id}

    def read(self, off):
        return self.regs[off]


class FakeBE:
    def __init__(self, version=RS.VERSION_CORE, build_id=int(BUILD, 16), clk=CLOSED, bit_sha256=""):
        self.csr = FakeCsr(version, build_id)
        self.clk = clk
        self.bit_sha256 = bit_sha256

    def fclk0_mhz(self):
        return self.clk


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Fake build dir (bit/hwh/sha256/summary.json), data package, results dir, deploy info."""
    build = tmp_path / "build"
    build.mkdir()
    bit = build / "gos_200.bit"
    bit.write_bytes(b"fake bitstream")
    (build / "gos_200.hwh").write_text("<hwh/>")
    (build / "gos_200.bit.sha256").write_text(f"{sha(bit)}  gos_200.bit\n")
    (build / "summary.json").write_text(json.dumps({
        "build_id": BUILD, "pl_clk0_mhz_actual": CLOSED, "wns_ns": 0.29, "whs_ns": 0.01,
        "failing_setup_endpoints": 0, "failing_hold_endpoints": 0, "vivado_version": "2023.1"}))
    data = tmp_path / "data"
    nets = {}
    for net in bc.NETS:
        (data / net).mkdir(parents=True)
        m = data / net / "MANIFEST.json"
        m.write_text(json.dumps({"git_dirty": False, "n_images": 10000, "net": net}))
        nets[net] = {"manifest_sha256": sha(m)}
    (data / "PACKAGE.json").write_text(json.dumps({"nets": nets, "git_commit": "abc",
                                                   "git_dirty": False}))
    rd = tmp_path / "dryrun"
    monkeypatch.setattr(RS.bc, "deploy_info", lambda: {"commit": "abc", "dirty": False,
                                                        "origin": "test"})
    return {"tmp": tmp_path, "bit": bit, "data": data, "rd": rd}


FAKE_STEP = textwrap.dedent("""
    import json, pathlib, sys, time
    a = sys.argv[1:]
    out = pathlib.Path(a[a.index("--out-dir") + 1])
    name, counter, ctl = a[0], pathlib.Path(a[1]), pathlib.Path(a[2])
    c = json.loads(ctl.read_text()) if ctl.is_file() else {}
    with counter.open("a") as f:
        f.write(name + "\\n")
    (out / f"hw_{name}.csv").write_text(f"source,x\\ndryrun_model,{name}\\n")
    print("fake step", name, flush=True)
    time.sleep(c.get(name, {}).get("sleep", 0))
    sys.exit(c.get(name, {}).get("rc", 0))
""")


def fake_steps(env, names=("a", "b"), images=None):
    script = env["tmp"] / "fake_step.py"
    script.write_text(FAKE_STEP)
    counter, ctl = env["tmp"] / "counter.txt", env["tmp"] / "ctl.json"
    images = images or {}

    def hook(steps, cfg):
        return [RS.Step(f"s2.{n}", 2, f"fake {n}", [str(script), n, str(counter), str(ctl)],
                        images=images.get(n, 10), expect=(f"hw_{n}.csv",)) for n in names]
    return hook, counter, ctl


def run(env, *args, hook=None, be=None, sessions="2"):
    argv = [sessions, "--backend", "model", "--bit", str(env["bit"]), "--data-dir", str(env["data"]),
            "--results-dir", str(env["rd"]), "--min-free-mb", "1", "--no-bringup-check", *args]
    return RS.main(argv, open_backend=lambda cfg: be or FakeBE(),
                   verify_data=lambda d: json.loads((Path(d) / "MANIFEST.json").read_text()),
                   steps_hook=hook)


def runs(counter: Path) -> list:
    return counter.read_text().split() if counter.is_file() else []


def state(env) -> dict:
    return json.loads((env["rd"] / RS.STATE_NAME).read_text())


# ---- resume ------------------------------------------------------------------------------------
def test_resume_skips_verified_steps(env):
    hook, counter, _ = fake_steps(env)
    assert run(env, hook=hook) == 0
    assert runs(counter) == ["a", "b"]
    st = state(env)
    assert st["steps"]["s2.a"]["status"] == "ok"
    assert st["steps"]["s2.a"]["outputs"]["hw_a.csv"] == sha(env["rd"] / "hw_a.csv")
    assert st["provenance"]["build_id_hw"] == BUILD and st["provenance"]["bit_sha256"] == sha(env["bit"])
    assert not (env["rd"] / ".staging").exists()
    assert run(env, hook=hook) == 0               # second run: everything verified -> skipped
    assert runs(counter) == ["a", "b"]


def test_tampered_output_is_rerun_and_old_file_archived(env):
    hook, counter, _ = fake_steps(env)
    assert run(env, hook=hook) == 0
    (env["rd"] / "hw_b.csv").write_text("edited by hand\n")
    assert run(env, hook=hook) == 0
    assert runs(counter) == ["a", "b", "b"]
    arch = list((env["rd"] / "archive").rglob("hw_b.csv"))
    assert arch and arch[0].read_text() == "edited by hand\n"      # archived, not deleted
    assert "dryrun_model,b" in (env["rd"] / "hw_b.csv").read_text()


def test_timed_out_step_not_promoted_then_rerun_from_scratch(env):
    hook, counter, ctl = fake_steps(env)
    ctl.write_text(json.dumps({"b": {"sleep": 30}}))
    assert run(env, "--step-timeout-min", "0.03", hook=hook) == 1
    st = state(env)
    assert st["steps"]["s2.a"]["status"] == "ok"
    assert st["steps"]["s2.b"]["status"] == "timeout"
    assert not (env["rd"] / "hw_b.csv").exists()                   # partial output not promoted
    assert list((env["rd"] / "archive").glob("*_s2.b_timeout/hw_b.csv"))
    ctl.write_text("{}")
    assert run(env, hook=hook) == 0
    assert runs(counter) == ["a", "b", "b"]                        # a skipped, b rerun
    assert state(env)["steps"]["s2.b"]["status"] == "ok"


def test_hard_killed_step_left_running_is_rerun(env):
    hook, counter, _ = fake_steps(env)
    assert run(env, hook=hook) == 0
    st = state(env)                                  # simulate a power cut during step b
    st["steps"]["s2.b"]["status"] = "running"
    (env["rd"] / RS.STATE_NAME).write_text(json.dumps(st))
    stg = env["rd"] / ".staging" / "s2.b"
    stg.mkdir(parents=True)
    (stg / "hw_b.csv").write_text("half written")
    assert run(env, hook=hook) == 0
    assert runs(counter) == ["a", "b", "b"]
    assert any(p.read_text() == "half written" for p in (env["rd"] / "archive").rglob("hw_b.csv"))


def test_failed_step_outputs_kept_and_rerun(env):
    hook, counter, ctl = fake_steps(env)
    ctl.write_text(json.dumps({"a": {"rc": 1}}))
    assert run(env, hook=hook) == 1
    assert state(env)["steps"]["s2.a"]["status"] == "failed"
    assert (env["rd"] / "hw_a.csv").exists()          # a FAIL result is still data (e.g. A1 mismatch)
    ctl.write_text("{}")
    assert run(env, hook=hook) == 0
    assert runs(counter) == ["a", "b", "a"]


def test_fresh_archives_everything(env):
    hook, counter, _ = fake_steps(env)
    assert run(env, hook=hook) == 0
    assert run(env, "--fresh", hook=hook) == 0
    assert runs(counter) == ["a", "b", "a", "b"]
    fresh = list((env["rd"] / "archive").glob("*_fresh"))
    assert fresh and (fresh[0] / RS.STATE_NAME).is_file() and (fresh[0] / "hw_a.csv").is_file()
    assert state(env)["timing"]["s2.a"]["duration_s"] >= 0          # durations carried over


def test_provenance_change_refuses_resume(env):
    hook, counter, _ = fake_steps(env)
    assert run(env, hook=hook) == 0
    st_before = state(env)
    rc = run(env, hook=hook, be=FakeBE(clk=CLOSED - 0.05))           # within the 0.1 MHz set tolerance: same provenance
    assert rc == 0
    s = json.loads((env["bit"].parent / "summary.json").read_text())
    s["build_id"] = "deadbeef"
    (env["bit"].parent / "summary.json").write_text(json.dumps(s))
    assert run(env, hook=hook, be=FakeBE(build_id=0xDEADBEEF)) == 4
    assert state(env)["provenance"] == st_before["provenance"]


def test_sessions_2_3_need_bringup_recorded(env):
    hook, _, _ = fake_steps(env)
    argv = ["2", "--backend", "model", "--bit", str(env["bit"]), "--data-dir", str(env["data"]),
            "--results-dir", str(env["rd"]), "--min-free-mb", "1"]
    rc = RS.main(argv, open_backend=lambda cfg: FakeBE(),
                 verify_data=lambda d: json.loads((Path(d) / "MANIFEST.json").read_text()),
                 steps_hook=hook)
    assert rc == 4


# ---- budget ------------------------------------------------------------------------------------
def test_budget_defers_steps_that_do_not_fit(env):
    # a: 10 images; b: 100k images x assumed 6 ms = 600 s > 1 min budget
    hook, counter, _ = fake_steps(env, images={"b": 100000})
    assert run(env, "--budget-min", "1", hook=hook) == 5              # incomplete
    assert runs(counter) == ["a"]
    st = state(env)
    assert st["steps"]["s2.b"]["status"] == "deferred"
    assert st["steps"]["s2.b"]["deferred_estimate_s"] > 60
    assert run(env, "--budget-min", "30", hook=hook) == 0             # next run picks it up
    assert runs(counter) == ["a", "b"]


def test_estimate_uses_previous_duration_then_learned_rate():
    st = RS.Step("s2.x", 2, "x", ["x.py"], images=1000)
    s = {"timing": {}}
    est, src = RS.estimate(st, s, "pynq")
    assert src.startswith("assumed") and est == pytest.approx(1000 * RS.DEFAULT_RATE_S["pynq"] + 30)
    s["timing"]["s2.y"] = {"backend": "pynq", "rate_s": 0.05, "params_key": "[]", "duration_s": 1}
    est, src = RS.estimate(st, s, "pynq")
    assert src.startswith("learned") and est == pytest.approx(1000 * 0.05 + 30)
    s["timing"]["s2.x"] = {"backend": "pynq", "params_key": st.params_key, "duration_s": 100.0}
    est, src = RS.estimate(st, s, "pynq")
    assert src == "previous run" and est == pytest.approx(110.0)


# ---- pre-flight --------------------------------------------------------------------------------
def hw_cfg(env, **deploy):
    bit = env["bit"]
    info = {"origin": "DEPLOY_INFO.json", "commit": "abc", "dirty": False, "bit": str(bit),
            "bit_sha256": sha(bit), "hwh_sha256": sha(bit.with_suffix(".hwh")), "build_id": BUILD,
            "bit_clock_mhz": CLOSED, "data_package_sha256": sha(env["data"] / "PACKAGE.json"),
            "data_git_dirty": False}
    info.update(deploy)
    return RS.Config(backend="pynq", board_dir=env["tmp"], results_dir=env["rd"],
                     data_dir=env["data"], deploy=info, bit=bit, min_free_mb=1)


def pf(cfg, be):
    return RS.preflight(cfg, open_backend=lambda c: be,
                        verify_data=lambda d: json.loads((Path(d) / "MANIFEST.json").read_text()),
                        say=lambda m: None)


def test_preflight_passes_on_consistent_deploy(env):
    prov = pf(hw_cfg(env), FakeBE(bit_sha256=sha(env["bit"])))
    assert prov["build_id_hw"] == BUILD and prov["source"] == bc.SOURCE_HW
    assert prov["closed_clock_mhz"] == CLOSED


@pytest.mark.parametrize("case, cfg_kw, be_kw, needle", [
    ("wrong BUILD_ID", {}, {"build_id": 0x12345678}, "BUILD_ID register 12345678"),
    ("wrong VERSION (shell bit)", {}, {"version": 0x474F5300}, "VERSION 0x474F5300"),
    ("DEPLOY_INFO bit sha", {"bit_sha256": "0" * 64}, {}, "DEPLOY_INFO bit_sha256"),
    ("clock above closed", {}, {"clk": 250.0}, "ABOVE the timing-closed clock"),
    ("clock below closed", {}, {"clk": 150.0}, "NOT at the closed clock"),
    ("clock 0.2 MHz off the closed clock", {}, {"clk": CLOSED - 0.2}, "NOT at the closed clock"),
    ("data package sha", {"data_package_sha256": "f" * 64}, {}, "data_package_sha256"),
    ("dirty scripts", {"dirty": True}, {}, "dirty tree"),
    ("no DEPLOY_INFO", {"origin": "none"}, {}, "DEPLOY_INFO.json missing"),
])
def test_preflight_failures(env, case, cfg_kw, be_kw, needle):
    with pytest.raises(RS.PreflightError) as e:
        pf(hw_cfg(env, **cfg_kw), FakeBE(**be_kw))
    assert needle in str(e.value), (case, str(e.value))


def test_preflight_shipped_sha256_mismatch(env):
    Path(str(env["bit"]) + ".sha256").write_text("ab" * 32 + "  gos_200.bit\n")
    with pytest.raises(RS.PreflightError, match="shipped gos_200.bit.sha256"):
        pf(hw_cfg(env), FakeBE())


def test_preflight_timing_not_met(env):
    s = json.loads((env["bit"].parent / "summary.json").read_text())
    s["wns_ns"] = -0.1
    (env["bit"].parent / "summary.json").write_text(json.dumps(s))
    with pytest.raises(RS.PreflightError, match="timing met"):
        pf(hw_cfg(env), FakeBE())


def test_preflight_dirty_with_flag_and_session1_needs_the_closed_clock(env):
    cfg = hw_cfg(env, dirty=True)
    cfg.allow_dirty = True
    prov = pf(cfg, FakeBE())
    assert prov["scripts_dirty"] and any("INVALID for the paper" in w for w in prov["warnings"])
    # 2026-09-30: the 300 MHz build ran Session 1 at 199.998 MHz because only "<=" was required
    assert RS.Config.__dataclass_fields__["require_clock_equal"].default is True
    with pytest.raises(RS.PreflightError, match="NOT at the closed clock"):
        pf(cfg, FakeBE(clk=100.0))


def test_preflight_failure_runs_nothing(env):
    hook, counter, _ = fake_steps(env)
    assert run(env, hook=hook, be=FakeBE(build_id=1)) == 3
    assert runs(counter) == [] and not (env["rd"] / RS.STATE_NAME).exists()


def test_dry_run_preflight_says_dry_run(env):
    lines = []
    cfg = RS.Config(backend="model", board_dir=env["tmp"], results_dir=env["rd"],
                    data_dir=env["data"], deploy={"origin": "dry", "commit": "abc", "dirty": False,
                                                   "build_id": BUILD, "bit_clock_mhz": CLOSED},
                    bit=env["bit"], min_free_mb=1)
    RS.preflight(cfg, open_backend=lambda c: FakeBE(), say=lines.append,
                 verify_data=lambda d: json.loads((Path(d) / "MANIFEST.json").read_text()))
    assert all("DRY RUN" in ln for ln in lines if "preflight" in ln)


# ---- B2 sweep ---------------------------------------------------------------------------------
B2_ALL = [99.999, 111.11, 124.99875, 142.855714, 166.665, 199.998, 249.9975]


@pytest.mark.parametrize("closed, expect", [
    (199.998001, B2_ALL[:6]),
    (249.997498, B2_ALL),                       # the performance build: 100 ... 250 (7 points)
    (240.0, B2_ALL[:6]),
    (299.997009, B2_ALL),                       # 300 MHz is not reachable: never in the sweep
    (150.0, B2_ALL[:4]),
])
def test_b2_sweep_clocks(closed, expect):
    got = bc.b2_sweep_clocks(closed)
    assert got == expect
    assert max(got) <= closed + bc.FCLK_SET_TOL_MHZ
    for c in got:                               # every point is an exact divider of the source PLL
        assert D.best_pl_dividers(bc.B2_SRC_PLL_MHZ, c)[2] == pytest.approx(c, abs=1e-6)


def test_b2_step_uses_sweep_and_closed_clock(env):
    cfg = hw_cfg(env, bit_clock_mhz=249.997498)
    steps = {s.id: s for s in RS.build_steps(cfg, [3])}
    argv = steps["s3.B2"].argv
    i = argv.index("--clocks")
    assert argv[i + 1:argv.index("--max-mhz")] == [f"{c:.6f}" for c in B2_ALL]
    assert argv[argv.index("--max-mhz") + 1] == "249.997498"
    assert argv[argv.index("--power-repeats") + 1] == "3" and "--with-meter" not in argv
    assert steps["s3.B2"].group == "sweep"
    exp = steps["s3.B2"].expect
    assert "hw_b2_clock.csv" in exp
    for tag in ("100mhz", "111mhz", "125mhz", "143mhz", "167mhz", "200mhz", "250mhz"):
        for k in ("samples", "phases", "summary"):
            assert f"hw_b2_power_ina260_{k}_lenet5_{tag}.csv" in exp
    assert "mock" not in argv                                      # hw: real sensor only


def test_session3_power_steps_ina260_primary(env):
    cfg = hw_cfg(env)
    steps = [s for s in RS.build_steps(cfg, [3]) if s.group != "fcal"]
    assert [s.id for s in steps] == ["s3.B1.lenet5", "s3.B1.cifar10", "s3.B2"]   # INA260 only
    for s, net in zip(steps, ("lenet5", "cifar10")):
        a = s.argv
        assert a[0] == RS.POWER_HOOK_SCRIPT and "--protocol" in a and "--bit" in a
        assert a[a.index("--net") + 1] == net and a[a.index("--tag") + 1] == f"_{net}"
        assert a[a.index("--cpu-kind") + 1] == "cpu_ort_int8"          # best CPU baseline (ORT INT8)
        assert a[a.index("--cpu-threads") + 1] == "4" and a[a.index("--cpu-cores") + 1] == "0-3"
        assert a[a.index("--repeats") + 1] == "3" and a[a.index("--phase-s") + 1] == "60"
        assert "mock" not in a and s.out_flag == "--out-dir"
        assert set(s.expect) == {f"hw_b1_power_ina260_{k}_{net}.csv"
                                 for k in ("samples", "phases", "summary")}
        assert RS.POWER_LABEL in s.title
        assert a[a.index("--order") + 1] == "random" and "--order-seed" in a
    dry = RS.Config(backend="model", board_dir=env["tmp"], results_dir=env["rd"],
                    data_dir=env["data"], deploy={"bit_clock_mhz": CLOSED}, bit=env["bit"])
    for s in RS.build_steps(dry, [3], window_s=2.0):
        if s.group != "fcal":
            assert s.argv[s.argv.index("--sensor") + 1] == "mock"


def test_ina260_is_the_only_power_source(env):
    """User decision: the INA260 is the only power source; no meter steps / options remain."""
    steps = RS.build_steps(hw_cfg(env), [1, 2, 3])
    for s in steps:
        assert "meter" not in " ".join(s.argv[1:]).replace(str(env["tmp"]), "").lower() \
            and "meter" not in s.title.lower(), s.id
    with pytest.raises(SystemExit):
        RS.make_parser().parse_args(["3", "--with-meter"])


def test_power_step_estimates_realistic(env):
    steps = {s.id: s for s in RS.build_steps(hw_cfg(env), [3])}
    est, src = RS.estimate(steps["s3.B1.lenet5"], {"timing": {}}, "pynq")
    # 3 repeats x 6 phases (idle/accel/idle/control/idle/cpu) x 60 s + final idle 60 s = 1140 s,
    # + setup 60 s + step overhead 30 s
    assert est == pytest.approx(19 * 60 + RS.B1_POWER_OVERHEAD_S + RS.STEP_OVERHEAD_S)
    assert 19 * 60 < est < 22 * 60
    nc = {s.id: s for s in RS.build_steps(hw_cfg(env), [3], power_control=False)}
    assert "--no-control" in nc["s3.B1.lenet5"].argv
    assert RS.estimate(nc["s3.B1.lenet5"], {"timing": {}}, "pynq")[0] == pytest.approx(
        13 * 60 + RS.B1_POWER_OVERHEAD_S + RS.STEP_OVERHEAD_S)
    est2, _ = RS.estimate(steps["s3.B2"], {"timing": {}}, "pynq")
    per_clock = 3 * 3 * 60 + RS.B2_PER_CLOCK_OVERHEAD_S
    b2 = steps["s3.B2"].argv
    n_clk = b2.index("--max-mhz") - b2.index("--clocks") - 1       # reachable clocks capped at the closed clock
    assert n_clk == len(bc.b2_sweep_clocks(float(b2[b2.index("--max-mhz") + 1])))
    assert est2 == pytest.approx(n_clk * per_clock + n_clk * 100 * RS.DEFAULT_RATE_S["pynq"]
                                 + RS.STEP_OVERHEAD_S)
    q = {s.id: s for s in RS.build_steps(hw_cfg(env), [3], window_s=30.0, power_repeats=1)}
    assert RS.estimate(q["s3.B1.cifar10"], {"timing": {}}, "pynq")[0] == pytest.approx(
        7 * 30 + RS.B1_POWER_OVERHEAD_S + RS.STEP_OVERHEAD_S)


def test_budget_defers_power_steps(env):
    counter = env["tmp"] / "never.txt"

    def hook(steps, cfg):   # the real Session 3 power steps, pointed at a script that must not run
        steps = [s for s in steps if s.group in ("energy", "sweep")]
        for s in steps:
            s.argv = [str(env["tmp"] / "must_not_run.py"), *s.argv[1:]]
        return steps
    (env["tmp"] / "must_not_run.py").write_text(f"open({str(counter)!r}, 'a').write('ran')\n")
    rc = run(env, "--budget-min", "10", "--window-s", "60", hook=hook, sessions="3")
    assert rc == 5 and not counter.exists()
    st = state(env)
    for sid in ("s3.B1.lenet5", "s3.B1.cifar10", "s3.B2"):
        assert st["steps"][sid]["status"] == "deferred"
        assert st["steps"][sid]["deferred_estimate_s"] > 600


def test_power_constants_match_power_log():
    import power_log as pl
    import exp_b1_power as b1
    assert RS.POWER_LABEL == pl.LABEL == b1.SENSOR_LABEL
    assert not hasattr(RS, "METER_LABEL") and not hasattr(b1, "METER_LABEL")
    for c in (100.0, 150.0, 199.998001, 249.997498, 299.997):
        assert RS.clock_tag(c) == pl.clock_tag(c)


def test_model_output_refused_outside_dryrun(tmp_path):
    with pytest.raises(SystemExit):
        RS.main(["1", "--backend", "model", "--results-dir", str(tmp_path / "results")])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---- host path / fast-path bring-up ----------------------------------------------------------
def test_host_path_steps(env):
    cfg = hw_cfg(env)
    st = {s.id: s for s in RS.build_steps(cfg, [1, 2, 3])}
    f = st["s1.fast"]
    assert not f.required and f.argv[f.argv.index("--host-path") + 1] == "fast"
    b3 = st["s2.B3"]                    # interleaved: safe always, fast only if s1.fast is OK
    assert b3.argv[b3.argv.index("--conditions") + 1] == "safe" and "fast" not in b3.argv
    assert b3.cond_argv == (("s1.fast", ["--conditions-add", "fast"]),) and b3.requires == ()
    assert "fast" in RS.resolve_step(b3, {"steps": {"s1.fast": {"status": "ok"}}}).argv
    assert "fast" not in RS.resolve_step(b3, {"steps": {"s1.fast": {"status": "failed"}}}).argv
    assert "s2.B3fast" not in st
    for sid in ("s2.A1", "s2.A2A3", "s2.A4", "s3.B1.lenet5", "s3.B1.cifar10", "s3.B2"):
        a = st[sid].argv
        assert a[a.index("--host-path") + 1] == "safe" and st[sid].requires == (), sid
    st = {s.id: s for s in RS.build_steps(cfg, [2, 3], host_path="fast", fast_store="words32")}
    for sid in ("s2.A1", "s2.A2A3", "s2.A4", "s3.B1.lenet5", "s3.B1.cifar10", "s3.B2"):
        a = st[sid].argv
        assert a[a.index("--host-path") + 1] == "fast" and st[sid].requires == ("s1.fast",), sid
        assert a[a.index("--fast-store") + 1] == "words32"
    assert st["s2.B3"].argv[st["s2.B3"].argv.index("--conditions") + 1] == "safe"


def _req_hook(env, fast_rc):
    script = env["tmp"] / "fake_step.py"
    script.write_text(FAKE_STEP)
    counter, ctl = env["tmp"] / "counter.txt", env["tmp"] / "ctl.json"
    ctl.write_text(json.dumps({"fast": {"rc": fast_rc}}))

    def hook(steps, cfg):
        mk = lambda sid, n, **kw: RS.Step(sid, int(sid[1]), n, [str(script), n, str(counter), str(ctl)],  # noqa: E731
                                          expect=(f"hw_{n}.csv",), **kw)
        return [mk("s1.fast", "fast", required=False),
                mk("s2.main", "main", requires=("s1.fast",)),
                mk("s2.info", "info", required=False, requires=("s1.fast",))]
    return hook, counter


def test_fast_path_steps_blocked_until_bringup_passes(env):
    hook, counter = _req_hook(env, fast_rc=1)
    argv = ["1,2", "--backend", "model", "--bit", str(env["bit"]), "--data-dir", str(env["data"]),
            "--results-dir", str(env["rd"]), "--min-free-mb", "1"]
    kw = dict(open_backend=lambda cfg: FakeBE(),
              verify_data=lambda d: json.loads((Path(d) / "MANIFEST.json").read_text()))
    rc = RS.main(argv, steps_hook=hook, **kw)
    assert rc == 1 and runs(counter) == ["fast"]             # main + info never ran
    st = state(env)
    assert st["steps"]["s2.main"]["status"] == "blocked"
    assert st["steps"]["s2.info"]["status"] == "blocked"
    hook, counter = _req_hook(env, fast_rc=0)                # bring-up passes -> both run
    rc = RS.main(argv, steps_hook=hook, **kw)
    assert rc == 0 and runs(counter)[-3:] == ["fast", "main", "info"]


def test_archive_everything_honours_keep_for_result_files(tmp_path):
    rd = tmp_path / "dryrun_results"
    (rd / "logs").mkdir(parents=True)
    for n in ("hw_nooverlay_idle_samples.csv", "hw_nooverlay_idle_phases.csv", "hw_old.csv"):
        (rd / n).write_text("x\n")
    (rd / "logs" / "keep.log").write_text("log\n")
    keep = [rd / "hw_nooverlay_idle_samples.csv", rd / "hw_nooverlay_idle_phases.csv",
            rd / "logs" / "keep.log"]
    ar = RS.archive_everything(rd, "fresh", keep=keep)
    assert all(p.exists() for p in keep)                  # kept files stay in results/
    assert not (rd / "hw_old.csv").exists()               # everything else is archived
    assert (ar.root / "hw_old.csv").exists()


def _samples_csv(path, start, phases):
    import csv as _csv
    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["repeat", "phase", "t_epoch", "power_w"])
        t = start
        for rep, phase, spike_at in phases:
            for i in range(600):
                p = 3.30 + (0.4 if spike_at is not None and spike_at <= i < spike_at + 5 else 0.0)
                w.writerow([rep, phase, f"{t:.3f}", f"{p:.3f}"])
                t += 0.1
        w.writerow(["", "", f"{t:.3f}", "9.9"])        # inter-phase filler row is ignored


def test_login_spikes_flags_burst_after_login_and_login_in_phase(tmp_path):
    import login_spikes as LS
    t0 = 1_790_000_000.0
    # phase 1 has a burst at +10.0 s (sample 100), phase 2 is clean
    _samples_csv(tmp_path / "hw_x_power_ina260_samples_a.csv", t0, [("1", "accel", 100), ("1", "idle", None)])
    iso = lambda t: __import__("datetime").datetime.fromtimestamp(t, __import__("datetime").timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S+0000")
    journal = (f"{iso(t0 + 9.5)} kria sshd[1]: Accepted publickey for ubuntu from 1.2.3.4\n"      # -> burst
               f"{iso(t0 + 70.0)} kria sshd[2]: Accepted publickey for ubuntu from 1.2.3.4\n"     # in phase 2, no burst
               f"{iso(t0 + 500.0)} kria sshd[3]: Accepted publickey for ubuntu from 1.2.3.4\n"    # outside every phase
               f"{iso(t0 + 20.0)} kria sshd[4]: Failed password for x\n")                         # not a login
    logins = LS.parse_logins(journal)
    assert len(logins) == 3
    found = LS.analyse(LS.read_phases(tmp_path), logins)
    kinds = [(f["phase"], f["kind"]) for f in found]
    assert ("accel", "spike") in kinds and ("idle", "in_phase") in kinds
    assert not any(f["login"] == logins[2] for f in found)
    assert LS.main(["--results-dir", str(tmp_path), "--journal-file", _w(tmp_path, journal)]) == 1


def _w(tmp_path, text):
    p = tmp_path / "journal.txt"
    p.write_text(text)
    return str(p)


def test_login_check_writes_log_and_reports(tmp_path):
    t0 = 1_790_000_000.0
    rd = tmp_path / "dryrun_results"
    rd.mkdir()
    _samples_csv(rd / "hw_x_power_ina260_samples_a.csv", t0, [("1", "accel", None)])
    out = []
    journal = "2026-09-26T00:00:00+0000 kria sshd[1]: Accepted publickey for ubuntu from 1.2.3.4\n"
    found = RS.login_check(rd, "2026-09-26T00:00:00+00:00", out.append, journal_text=journal)
    assert list(rd.glob("logs/ssh_logins_*.txt")) and any("ssh login check" in m for m in out)
    assert isinstance(found, list)


def test_idle_reference_refused_when_the_pl_was_loaded_this_boot(tmp_path):
    loaded = "[  392.58] fpga_manager fpga0: writing gos_300.bin to Xilinx ZynqMP FPGA Manager\n"
    assert RS.fpga_loads_this_boot("boot noise\n") == 0 and RS.fpga_loads_this_boot(loaded * 3) == 3
    out = []
    a = SimpleNamespace(session_index=1, idle_ref_s=1.0, allow_dirty=True)
    assert RS.run_idle_ref(a, tmp_path, {"paper_grade": True, "cpu_governor": "", "cpu_freq_khz": ""},
                           out.append, kmsg_text=loaded) == []
    assert any("REFUSED" in m and "1 bitstream load" in m for m in out)


def test_system_activity_windows_flag_overlapping_phases(tmp_path):
    import login_spikes as LS
    t0 = 1_790_000_000.0
    _samples_csv(tmp_path / "hw_x_power_ina260_samples_a.csv", t0, [("1", "accel", None), ("1", "idle", None)])
    iso = lambda t: __import__("datetime").datetime.fromtimestamp(t, __import__("datetime").timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S+0000")
    j = (f"{iso(t0 + 70)} kria systemd[1]: Starting Daily apt upgrade and clean activities...\n"
         f"{iso(t0 + 100)} kria systemd[1]: Finished Daily apt upgrade and clean activities.\n"
         f"{iso(t0 + 200)} kria systemd[1]: Starting Refresh fwupd metadata.\n")           # not apt/PackageKit
    w = LS.service_windows(j)
    assert w == [(t0 + 70, t0 + 100)]
    f = LS.analyse_windows(LS.read_phases(tmp_path), w)
    assert [(x["phase"], round(x["overlap_s"])) for x in f] == [("idle", 30)]
    out = []
    rd = tmp_path / "dryrun_r"; rd.mkdir()
    _samples_csv(rd / "hw_x_power_ina260_samples_a.csv", t0, [("1", "accel", None), ("1", "idle", None)])
    found = RS.login_check(rd, "2026-09-26T00:00:00+00:00", out.append, journal_text="", units_text=j)
    assert any("system-activity check" in m and "1 power phase" in m for m in out) and len(found) == 1
