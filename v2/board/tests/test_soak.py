"""exp_soak: per-job checks, counters, buckets, temperature, abort; session step; dry run."""
import csv
from types import SimpleNamespace

import numpy as np
import pytest

import board_common as bc
import exp_soak as S
import gos_driver as D
import session_extra_steps as X


# ---- fakes --------------------------------------------------------------------------------------
def fake_pkg(name="netA", n=4, layers=(10, 20, 5), oc=3):
    gl = np.arange(n * oc, dtype=np.int32).reshape(n, oc)
    return SimpleNamespace(
        name=name, n=n, x_act=np.arange(n), golden_logits=gl,
        net={"n_layers": len(layers), "layers": [{"name": f"L{i}"} for i in range(len(layers))], "OC": oc},
        model_cycles={"layers": [{"cycles": c} for c in layers], "total": {"cycles": sum(layers) + 1,
                                                                            "mac_active": 7}})


class FakeClock:
    def __init__(self, step=0.01):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


class FakeDev:
    """infer returns the golden result; `faults` maps job number -> fault kind."""

    def __init__(self, faults=None):
        self.faults = faults or {}
        self.pkg = None
        self.jobs = 0
        self.loads = []
        self.recovers = 0
        self.last_status = D.ST_DONE

    def soft_reset(self):
        pass

    def load_net(self, pkg, verify=True):
        if self.faults.get(("load", len(self.loads))) == "load":
            self.loads.append(None)
            raise D.GosError("readback mismatch")
        self.loads.append(pkg.name)
        self.pkg = pkg

    def recover(self):
        self.recovers += 1

    def status(self):
        return self.last_status

    def infer(self, x, read_counters=True):
        j = self.jobs
        self.jobs += 1
        f = self.faults.get(j)
        p = self.pkg
        self.last_status = D.ST_DONE
        if f == "refuse":
            raise D.GosJobError(0x0300, D.ST_ERROR | D.ST_DONE)
        if f == "timeout":
            raise D.GosTimeout("job not done", D.ST_BUSY, 5)
        logits = p.golden_logits[int(x)].copy()
        lc = [int(L["cycles"]) for L in p.model_cycles["layers"]]
        tot, mac, stall, viol = p.model_cycles["total"]["cycles"], p.model_cycles["total"]["mac_active"], 0, 0
        if f == "logit":
            logits[0] += 1
        if f == "layer":
            lc[1] += 1
        if f == "total":
            tot += 3
        if f == "viol":
            viol = 0x1
        if f == "status":
            self.last_status = D.ST_DONE | 0x8
        if f == "stall":
            stall = 2
        return D.InferResult(logits=logits, total_cyc=tot, layer_cyc=lc, mac_active=mac, stall=stall,
                             violation=viol)


def run(dev, pkgs, dur=1.0, block=0.25, bucket=0.5, **kw):
    return S.soak(dev, pkgs, dur, block, bucket, clock=FakeClock(0.01), **kw)


# ---- tests --------------------------------------------------------------------------------------
def test_clean_soak_alternates_and_counts():
    a, b = fake_pkg("netA"), fake_pkg("netB", layers=(3, 4))
    dev = FakeDev()
    res = run(dev, [a, b])
    sa, sb = res["stats"]["netA"], res["stats"]["netB"]
    assert sa.errors() == 0 and sb.errors() == 0
    assert sa.c["images"] > 0 and sb.c["images"] > 0
    assert sa.c["images"] + sb.c["images"] == dev.jobs
    assert dev.loads[:3] == ["netA", "netB", "netA"]            # alternation reloads the net
    assert sa.lmin.tolist() == sa.lmax.tolist() == [10, 20, 5]
    assert sa.tmin == sa.tmax == 36 and sa.tdist == {36}
    assert sum(bk["images"] for bk in res["buckets"].values()) == dev.jobs
    assert not res["aborted"]


def test_each_fault_is_counted():
    faults = {0: "logit", 1: "layer", 2: "total", 3: "viol", 4: "status", 5: "stall", 6: "refuse",
              7: "timeout"}
    a = fake_pkg()
    dev = FakeDev(faults)
    st = run(dev, [a], dur=0.5)["stats"]["netA"]
    c = st.c
    assert c["logit_mismatch"] == 1 and c["layer_cycle_mismatch"] == 1 and c["cycle_mismatch"] == 1
    assert c["err_flags"] == 1 and c["stall_nonzero"] == 1
    assert c["status_errors"] == 2                   # bad STATUS + the refused job
    assert c["job_errors"] == 1 and c["timeouts"] == 1
    assert dev.recovers == 2
    assert st.lmax[1] == 21 and st.tmax == 39 and len(st.tdist) == 2 and len(st.ldist[1]) == 2
    assert st.first_errors and any("LAYER_CYC" in e for e in st.first_errors)


def test_abort_after_max_errors():
    dev = FakeDev({j: "logit" for j in range(1000)})
    res = run(dev, [fake_pkg()], dur=10.0, max_errors=5)
    assert res["aborted"] and res["stats"]["netA"].c["logit_mismatch"] == 6


def test_load_error_counted_and_retried():
    dev = FakeDev({("load", 0): "load"})
    st = run(dev, [fake_pkg()], dur=0.3)["stats"]["netA"]
    assert st.c["load_errors"] == 1 and st.c["images"] > 0 and st.errors() == 1


def test_temperature_sampling_dict_and_failure():
    vals = iter([{"ps": 40.0, "pl": 45.5}, 41.0, RuntimeError("no ams")] + [42.0] * 100)

    def reader():
        v = next(vals)
        if isinstance(v, Exception):
            raise v
        return v
    res = run(FakeDev(), [fake_pkg()], dur=1.0, temp_fn=reader, temp_every_s=0.1)
    t = [v for _, v in res["temps"]]
    assert t[0] == 45.5 and t[1] == 41.0 and 42.0 in t
    assert S._temps_summary(res["temps"])["temp_c_max"] == "45.50"


def test_find_temp_reader(monkeypatch):
    assert S.find_temp_reader(dry=True)[0] is None
    import board_env
    monkeypatch.setattr(board_env, "read_die_temp", lambda: {"source": "unavailable", "channels_c": {}, "max_c": None})
    assert S.find_temp_reader()[0] is None and "unavailable" in S.find_temp_reader()[1]
    monkeypatch.setattr(board_env, "read_die_temp", lambda: {"source": "iio x (ams)", "channels_c": {"a": 51.0}, "max_c": 51.0})
    fn, src = S.find_temp_reader()
    assert S.read_temp(fn) == 51.0 and "board_env" in src
    monkeypatch.delattr(board_env, "read_die_temp")
    monkeypatch.setattr(bc, "read_ams_temp", lambda: 50.0, raising=False)
    for n in S.TEMP_READER_NAMES:
        if n != "read_ams_temp" and hasattr(bc, n):
            monkeypatch.delattr(bc, n)
    fn, src = S.find_temp_reader()
    assert fn() == 50.0 and src == "board_common.read_ams_temp"
    assert S.find_temp_reader("none")[0] is None


def test_dry_run_end_to_end(tmp_path):
    out = tmp_path / "dryrun"
    rc = S.main(["--backend", "model", "--allow-dirty", "--duration-s", "3", "--block-s", "1",
                 "--bucket-s", "1", "--out-dir", str(out), "--clock-choice", str(tmp_path / "none.json")])
    assert rc == 0
    rows = list(csv.DictReader(open(out / "hw_soak.csv")))
    assert {r["source"] for r in rows} == {bc.SOURCE_DRYRUN}
    allr = [r for r in rows if r["row_kind"] == "all"][0]
    assert allr["result"] == "PASS" and int(allr["images"]) > 0
    assert allr["clock_choice_bit"] == "none recorded"
    assert "unavailable" in allr["temp_source"]
    lay = [r for r in rows if r["row_kind"] == "layer"]
    assert lay and all(r["cyc_distinct"] == "1" and r["cyc_spread"] == "0" for r in lay)
    b = list(csv.DictReader(open(out / "hw_soak_minutes.csv")))
    assert b and sum(int(r["images"]) for r in b) == int(allr["images"])


def test_dry_run_refused_outside_dryrun_and_clock_check(tmp_path):
    with pytest.raises(SystemExit):
        S.main(["--backend", "model", "--allow-dirty", "--duration-s", "1", "--out-dir", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        S.main(["--backend", "model", "--allow-dirty", "--duration-s", "1", "--clock-mhz", "200",
                "--expect-clock-mhz", "300", "--out-dir", str(tmp_path / "dryrun")])


def test_session_soak_step():
    opts = {"backend": "pynq", "data_dir": "data", "bit": "bit/gos_300.bit", "closed_mhz": 299.997009,
            "results_dir": "results", "host_path": "safe"}
    st = X.soak_step(opts)
    assert st["id"] == "s3.soak" and st["session"] == 3 and st["group"] == "soak"
    assert st["cmd"][0] == "exp_soak.py" and "--bit" in st["cmd"]
    assert st["params"]["duration_s"] == 1800.0 and st["timeout_s"] > st["est_s"] >= 1800.0
    assert "--expect-clock-mhz" in st["cmd"] and "results/hw_clock_choice.json" in st["cmd"]
    assert set(st["outputs"]) == {"hw_soak.csv", "hw_soak_minutes.csv"} and st["requires"] == []
    dry = X.soak_step(SimpleNamespace(backend="model", data_dir="d", closed_mhz=200.0, quick=False,
                                      host_path="fast", allow_dirty=True))
    assert dry["params"]["duration_s"] == 60.0 and "--clock-mhz" in dry["cmd"]
    assert "--allow-dirty" in dry["cmd"] and dry["requires"] == ["s1.fast"]
    assert X.soak_step({"quick": True})["params"]["duration_s"] == 300.0
    assert X.soak_step({"soak_s": 20})["params"]["duration_s"] == 20.0
    ids = [s["id"] for s in X.extra_steps(opts)]
    assert ids == ["s2.layerspread", "s3.soak"]
    for s in X.extra_steps(opts):
        assert {"id", "session", "group", "cmd", "outputs", "est_s", "timeout_s", "requires",
                "params"} <= set(s)
        assert s["group"] in ("A3", "A1", "latency", "baselines", "energy", "sweep", "soak", "A4")
