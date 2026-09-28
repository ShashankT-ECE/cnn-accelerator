"""power_log.py tests (laptop, no hardware): sensor backends, sampler rate, protocol arithmetic,
dry-run output rules, labels + metadata columns."""
import csv
import math
import time
from pathlib import Path

import pytest

import board_common as bc
import power_log as pl


# ---- fakes -------------------------------------------------------------------------------------
class FakeSMBus:
    """INA260 at `addr` on this bus; any write raises (read-only guarantee)."""

    def __init__(self, regs_by_addr: dict):
        self.regs = regs_by_addr
        self.reads = []

    def read_word_data(self, addr, reg):
        self.reads.append((addr, reg))
        if addr not in self.regs:
            raise OSError(121, "Remote I/O error")
        v = self.regs[addr].get(reg, 0)
        return ((v & 0xFF) << 8) | (v >> 8)          # SMBus little-endian word

    def read_byte_data(self, addr, reg):
        raise AssertionError("not used")

    def write_word_data(self, *a):
        raise AssertionError("WRITE attempted on I2C")

    def write_byte_data(self, *a):
        raise AssertionError("WRITE attempted on I2C")

    def write_i2c_block_data(self, *a):
        raise AssertionError("WRITE attempted on I2C")

    def close(self):
        pass


INA_REGS = {pl.REG_MFR_ID: 0x5449, pl.REG_DIE_ID: 0x2270, pl.REG_CONFIG: 0x6127,
            pl.REG_POWER: 350,            # 3.50 W
            pl.REG_CURRENT: 0xFF38,       # -200 -> -0.25 A
            pl.REG_VBUS: 4000}            # 5.000 V


def make_i2c(tmp_path, buses: dict):
    for n in buses:
        (tmp_path / f"i2c-{n}").write_text("")
    return str(tmp_path / "i2c-*"), (lambda n: buses[n])


def make_hwmon(root: Path, name="ina260_u14", uw=4200000, ma=850, mv=4950, ui=None, idx=0):
    d = root / f"hwmon{idx}"
    d.mkdir(parents=True)
    (d / "name").write_text(name + "\n")
    (d / "power1_input").write_text(f"{uw}\n")
    (d / "curr1_input").write_text(f"{ma}\n")
    (d / "in1_input").write_text(f"{mv}\n")
    if ui is not None:
        (d / "update_interval").write_text(f"{ui}\n")
    return d


class FakeRun:
    def __init__(self, text):
        self.text = text

    def __call__(self, cmd, **kw):
        class R:
            stdout = self.text
        return R()


NO_PS = dict(which=lambda n: None)


# ---- backends ----------------------------------------------------------------------------------
def test_hwmon_read_and_limits(tmp_path):
    make_hwmon(tmp_path / "hw", name="ams", idx=0)              # other sensor ignored
    make_hwmon(tmp_path / "hw", idx=1, ui=35)
    s, log = pl.discover("auto", hwmon_root=tmp_path / "hw", **NO_PS)
    assert s.backend == "hwmon" and "hwmon1" in s.device
    r = s.read()
    assert r.power_w == pytest.approx(4.2) and r.current_a == pytest.approx(0.85)
    assert r.voltage_v == pytest.approx(4.95)
    assert s.limits["update_interval_ms"] == 35


def test_order_hwmon_before_platformstats_before_i2c(tmp_path):
    fake = FakeSMBus({0x40: INA_REGS})
    g, fac = make_i2c(tmp_path, {1: fake})
    run = FakeRun("SOM total power    :     3417 mW\n")
    make_hwmon(tmp_path / "hw")
    s, _ = pl.discover("auto", hwmon_root=tmp_path / "hw", which=lambda n: "/usr/bin/platformstats",
                       run=run, i2c_glob=g, smbus_factory=fac)
    assert s.backend == "hwmon"
    s, _ = pl.discover("auto", hwmon_root=tmp_path / "none", which=lambda n: "/usr/bin/platformstats",
                       run=run, i2c_glob=g, smbus_factory=fac)
    assert s.backend == "platformstats" and s.read().power_w == pytest.approx(3.417)
    assert fake.reads == []                                   # i2c never touched
    s, _ = pl.discover("auto", hwmon_root=tmp_path / "none", which=lambda n: None,
                       i2c_glob=g, smbus_factory=fac)
    assert s.backend == "i2c"


def test_platformstats_parse_defensive():
    assert pl.parse_platformstats("nothing here\n") is None
    txt = "Power Utilization\nPS power : 1.5 W\nSOM total power : 3417 mW\n"
    w, line = pl.parse_platformstats(txt)
    assert w == pytest.approx(3.417) and "SOM" in line
    assert pl.parse_platformstats("total power: 2500000 uW")[0] == pytest.approx(2.5)


def test_platformstats_unparsable_falls_through(tmp_path):
    fake = FakeSMBus({0x40: INA_REGS})
    g, fac = make_i2c(tmp_path, {0: fake})
    s, log = pl.discover("auto", hwmon_root=tmp_path / "none", which=lambda n: "/x/platformstats",
                         run=FakeRun("garbage"), i2c_glob=g, smbus_factory=fac)
    assert s.backend == "i2c"
    assert any("no power line parsed" in x for x in log)


def test_i2c_id_check_scaling_and_read_only(tmp_path):
    wrong = {**INA_REGS, pl.REG_DIE_ID: 0x2260}          # INA226-like die id: rejected
    bus0 = FakeSMBus({0x40: wrong})
    bus1 = FakeSMBus({0x45: INA_REGS})
    g, fac = make_i2c(tmp_path, {0: bus0, 1: bus1})
    log = []
    c = pl.probe_i2c(g, fac, log=log)
    assert [(x["busnum"], x["addr"]) for x in c] == [(1, 0x45)]
    assert any("not INA260" in x for x in log)
    s = pl.I2cSensor(c[0]["bus"], 1, 0x45)
    r = s.read()
    assert r.power_w == pytest.approx(3.50)
    assert r.current_a == pytest.approx(-0.25)
    assert r.voltage_v == pytest.approx(5.0)
    assert s.limits["averages"] == 1 and s.limits["mode"] == 7 and s.limits["vbus_ct_us"] == 1100
    assert s.limits["update_period_ms_derived"] == pytest.approx(2.2)
    # read-only wrapper: write methods are not reachable at all
    with pytest.raises(PermissionError):
        s.bus.write_word_data(0x45, 0, 0)
    with pytest.raises(PermissionError):
        s.bus.write_byte_data(0x45, 6, 0)
    with pytest.raises(PermissionError):
        s.bus._bus = None
    # only ID / config / measurement registers were read
    regs = {reg for _, reg in bus0.reads + bus1.reads}
    assert regs <= {pl.REG_MFR_ID, pl.REG_DIE_ID, pl.REG_CONFIG, pl.REG_POWER, pl.REG_CURRENT,
                    pl.REG_VBUS}


def test_i2c_prefers_0x40_and_stops(tmp_path):
    bus = FakeSMBus({0x40: INA_REGS, 0x41: INA_REGS})
    g, fac = make_i2c(tmp_path, {2: bus})
    c = pl.probe_i2c(g, fac)
    assert c[0]["addr"] == 0x40
    assert {a for a, _ in bus.reads} == {0x40}


def test_i2c_open_error_logged(tmp_path):
    g, _ = make_i2c(tmp_path, {3: None})

    def fac(n):
        raise OSError(16, "Device or resource busy")
    log = []
    assert pl.probe_i2c(g, fac, log=log) == []
    assert any("cannot open" in x for x in log)


def test_list_sensors_no_writes(tmp_path):
    bus = FakeSMBus({0x40: INA_REGS})
    g, fac = make_i2c(tmp_path, {1: bus})
    make_hwmon(tmp_path / "hw", ui=35)
    out = pl.list_sensors(hwmon_root=tmp_path / "hw", which=lambda n: None, i2c_glob=g,
                          smbus_factory=fac)
    txt = "\n".join(out)
    assert "hwmon" in txt and "INA260" in txt and "addr=0x40" in txt


def test_mock_and_real_never_mixed(tmp_path):
    with pytest.raises(pl.SensorUnavailable):
        pl.discover("mock", allow_hw=True, allow_mock=False)
    with pytest.raises(pl.SensorUnavailable):
        pl.discover("hwmon", allow_hw=False, allow_mock=True)
    s, _ = pl.discover("auto", allow_hw=False, allow_mock=True)       # dry run: no probing
    assert s.backend == "mock" and not s.is_hw
    with pytest.raises(pl.SensorUnavailable):                         # hw run, nothing found
        pl.discover("auto", hwmon_root=tmp_path / "none", which=lambda n: None,
                    i2c_glob=str(tmp_path / "i2c-*"), allow_hw=True, allow_mock=False)
    hw_ctx = type("C", (), {"source": bc.SOURCE_HW})()
    with pytest.raises(SystemExit):
        pl._check_source(hw_ctx, pl.MockSensor())


def test_mock_deterministic_seeded():
    a = pl.MockSensor(noise_w=0.1, seed=7)
    b = pl.MockSensor(noise_w=0.1, seed=7)
    assert [a.read("accel").power_w for _ in range(5)] == [b.read("accel").power_w for _ in range(5)]
    assert pl.MockSensor().read("idle_mid").power_w == 1.0


# ---- sampler -----------------------------------------------------------------------------------
def test_rate_stats_exact():
    st = pl.rate_stats([0.1, 0.2, 0.3, 0.7], 0.0, 1.0)
    assert st["n"] == 4 and st["rate_hz"] == pytest.approx(4.0)
    assert st["max_gap_s"] == pytest.approx(0.4)


def test_logger_achieved_rate():
    lg = pl.PowerLogger(pl.MockSensor(), rate_hz=50)
    with lg:
        time.sleep(0.5)
    ov = lg.overall()
    assert ov["elapsed_s"] == pytest.approx(0.5, abs=0.2)
    assert ov["rate_hz"] == pytest.approx(ov["n"] / ov["elapsed_s"])
    assert 20 <= ov["rate_hz"] <= 60
    assert ov["max_gap_s"] < 0.25


def test_logger_phase_windows_atomic():
    lg = pl.PowerLogger(pl.MockSensor(), rate_hz=200)
    with lg:
        for ph in ("idle_pre", "accel", "idle_mid"):
            s0 = lg.begin(ph)
            time.sleep(0.1)
            lg.end(ph, s0, repeat=1, kind=pl.phase_kind(ph), images=0)
    for w in lg.windows:
        for s in lg.window_samples(w):
            assert s["live_phase"] == w["phase"]


# ---- arithmetic --------------------------------------------------------------------------------
def test_summarize_exact_known_step():
    ph = []
    for r in (1, 2):
        ph += [dict(repeat=r, phase="idle_pre", kind="idle", mean_w=1.0, duration_s=60.0, images=0),
               dict(repeat=r, phase="accel", kind="accel", mean_w=1.5, duration_s=60.0, images=3000),
               dict(repeat=r, phase="idle_mid", kind="idle", mean_w=1.5 - 0.25 * r, duration_s=60.0,
                    images=0),
               dict(repeat=r, phase="cpu", kind="cpu", mean_w=2.0, duration_s=60.0, images=600)]
    ph.append(dict(repeat=2, phase="idle_post", kind="idle", mean_w=1.0, duration_s=60.0, images=0))
    s = pl.summarize(ph, 2)
    r1, r2, mean, std = s
    # repeat 1: accel idle = (1.0 + 1.25)/2 = 1.125 ; cpu idle = (1.25 + 1.0)/2 = 1.125
    assert r1["accel_p_idle_w"] == 1.125 and r1["accel_dp_w"] == 0.375
    assert r1["accel_time_per_image_s"] == 0.02
    assert r1["accel_energy_per_image_j"] == pytest.approx(0.375 * 0.02, rel=1e-15)
    assert r1["accel_energy_per_image_mj"] == pytest.approx(7.5, rel=1e-15)
    assert r1["accel_p_idle_ref"] == "idle_pre#1+idle_mid#1"
    assert r1["cpu_p_idle_ref"] == "idle_mid#1+idle_pre#2"
    assert r1["cpu_dp_w"] == 0.875 and r1["cpu_time_per_image_s"] == 0.1
    # repeat 2: idle_mid 1.0 -> accel idle 1.0 ; cpu brackets idle_mid#2 + idle_post
    assert r2["accel_dp_w"] == 0.5 and r2["cpu_p_idle_ref"] == "idle_mid#2+idle_post#2"
    assert mean["accel_dp_w"] == pytest.approx(0.4375)
    assert std["accel_dp_w"] == pytest.approx(math.sqrt(2 * 0.0625 ** 2))
    assert mean["accel_energy_per_image_mj"] == pytest.approx((7.5 + 10.0) / 2)


def test_summarize_zero_images_and_single_idle():
    ph = [dict(repeat=1, phase="idle_pre", kind="idle", mean_w=1.0, duration_s=1.0, images=0),
          dict(repeat=1, phase="accel", kind="accel", mean_w=2.0, duration_s=1.0, images=0)]
    r = pl.summarize(ph, 1)[0]
    assert r["accel_dp_w"] == 1.0 and r["accel_p_idle_ref"] == "idle_pre#1"
    assert math.isnan(r["accel_time_per_image_s"]) and math.isnan(r["accel_energy_per_image_j"])


def counting(rate_per_s):
    def fn(deadline):
        n = 0
        while time.monotonic() < deadline:
            time.sleep(1.0 / rate_per_s)
            n += 1
        return n
    return fn


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


@pytest.fixture
def dry_ctx(tmp_path):
    return pl.SensorOnlyContext(bc.SOURCE_DRYRUN, tmp_path / "results" / "dryrun")


def test_protocol_end_to_end_mock(dry_ctx, tmp_path):
    sensor = pl.MockSensor({"idle": 2.0, "accel": 2.5, "cpu": 3.25})
    res = pl.run_power_protocol(dry_ctx, "lenet5", counting(200), accel_fn=counting(400),
                                sensor=sensor, phase_s=0.3, repeats=3, rate_hz=100,
                                accel_label="fake accel", cpu_label="fake cpu")
    assert res["source"] == bc.SOURCE_DRYRUN and res["label"] == "SOM-rail power (INA260)"
    reps = [s for s in res["summary"] if s["row_kind"] == "repeat"]
    assert len(reps) == 3
    for s in reps:
        assert s["accel_dp_w"] == 0.5 and s["cpu_dp_w"] == 1.25        # exact: noise 0
        assert s["accel_time_per_image_s"] == s["accel_duration_s"] / s["accel_images"]
        assert s["accel_energy_per_image_j"] == 0.5 * s["accel_time_per_image_s"]
        assert s["cpu_energy_per_image_mj"] == pytest.approx(1.25 * s["cpu_time_per_image_s"] * 1e3)
    files = res["files"]
    for k in ("samples", "phases", "summary"):
        p = Path(files[k])
        assert "dryrun" in p.parts and p.name.startswith("hw_b1_power_ina260_")
        rows = read_csv(p)
        assert rows
        for r in rows:
            assert r["measurement"] == "SOM-rail power (INA260)" and r["rail"] == "VCC_SOM"
            assert r["source"] == bc.SOURCE_DRYRUN and r["sensor_backend"] == "mock"
            for c in bc.META_COLUMNS + bc.EXTRA_META:
                assert c in r
            assert r["rate_requested_hz"] == "100.000" and r["rate_achieved_hz"]
    phases = read_csv(files["phases"])
    assert [p["phase"] for p in phases][:4] == ["idle_pre", "accel", "idle_mid", "cpu"]
    assert phases[-1]["phase"] == "idle_post" and len(phases) == 13
    assert all(int(p["n_samples"]) > 0 for p in phases)
    summ = read_csv(files["summary"])
    assert [r["row_kind"] for r in summ] == ["repeat"] * 3 + ["mean", "std"]
    assert float(summ[3]["accel_dp_w"]) == 0.5


def test_dryrun_output_refused_outside_dryrun(tmp_path):
    ctx = pl.SensorOnlyContext(bc.SOURCE_DRYRUN, tmp_path / "dryrun")
    with pytest.raises(SystemExit):
        pl.run_power_protocol(ctx, "lenet5", None, accel_fn=counting(100), sensor=pl.MockSensor(),
                              out_dir=tmp_path / "results", phase_s=0.05, repeats=1)
    with pytest.raises(SystemExit):   # SensorOnlyContext itself refuses a non-dryrun dir
        pl.SensorOnlyContext(bc.SOURCE_DRYRUN, tmp_path / "results")
    with pytest.raises(SystemExit):   # hardware output never under dryrun
        pl.SensorOnlyContext(bc.SOURCE_HW, tmp_path / "dryrun")


def test_protocol_without_cpu(dry_ctx):
    res = pl.run_power_protocol(dry_ctx, "lenet5", None, accel_fn=counting(100),
                                sensor=pl.MockSensor(), phase_s=0.1, repeats=1, rate_hz=50,
                                tag="_nocpu")
    assert [p["phase"] for p in res["phases"]] == ["idle_pre", "accel", "idle_mid"]
    assert "cpu_dp_w" not in res["summary"][0]


def test_sample_only_mock(dry_ctx):
    st = pl.sample_only(dry_ctx, 0.3, pl.MockSensor(), rate_hz=50)
    assert st["n"] > 0 and st["mean_w"] == 1.0 and st["value_changes"] == 0
    rows = read_csv(dry_ctx.out_dir / "hw_b1_power_ina260_samples_sensorcheck.csv")
    assert rows and all(r["measurement"] == "SOM-rail power (INA260)" for r in rows)


def test_b2_reduced_protocol_per_clock(dry_ctx):
    """B2: idle/accel/idle x repeats per clock, prefix hw_b2_power_ina260, tag per clock, clock_mhz."""
    assert pl.clock_tag(199.998001) == "200mhz" and pl.clock_tag(100.0) == "100mhz"
    res = pl.run_power_protocol(dry_ctx, "lenet5", None, accel_fn=counting(200),
                                sensor=pl.MockSensor({"idle": 1.0, "accel": 1.5}), phase_s=0.15,
                                repeats=2, rate_hz=50, prefix=pl.PREFIX_B2,
                                tag=f"_lenet5_{pl.clock_tag(150.0)}", clock_mhz=149.999)
    assert [p["phase"] for p in res["phases"]] == ["idle_pre", "accel", "idle_mid"] * 2
    assert Path(res["files"]["summary"]).name == "hw_b2_power_ina260_summary_lenet5_150mhz.csv"
    mean = next(s for s in res["summary"] if s["row_kind"] == "mean")
    assert mean["accel_dp_w"] == 0.5
    rows = read_csv(res["files"]["phases"])
    assert all(r["clock_mhz"] == "149.999000" for r in rows)
