"""Driver unit tests against the simulated register map (gos_sim) — no hardware, no pynq."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import board_common as bc
import gos_driver as D
from gos_sim import JobOutcome, SimDevice, SimSlvErr

BOARD = Path(__file__).resolve().parents[1]


class FakeBackend(D.MmioBackend):
    kind, source = "fake", bc.SOURCE_DRYRUN

    def __init__(self, sim):
        self.sim = sim
        self.clk = 200.0
        super().__init__(sim.csr, sim.mems)

    def fclk0_mhz(self):
        return self.clk

    def set_fclk0(self, mhz):
        self.clk = mhz
        return mhz


class Pkg:
    """Minimal net package: 2 layers, OC 3."""
    def __init__(self):
        self.wgt = np.arange(1, 11, dtype=np.uint64) * np.uint64(0x0101010101010101)
        self.qparam = np.array([0xAABBCCDD11223344, 0x2A, 0x1, 0x3F], dtype=np.uint64)
        self.desc = (np.arange(32, dtype=np.uint32) * 7).reshape(2, 16)
        self.net = {"OC": 3, "act_in_words": 4, "net": "fake", "n_layers": 2}


def ok_job(logits=(-5, 7, 2**31 - 1), total=0x1_0000_0005):
    return lambda sim: JobOutcome(logits=list(logits), total_cyc=total, mac_active=12,
                                  stall=0, layer_cyc=[3, 4])


def make(job=None, **kw):
    sim = SimDevice(job or ok_job(), **kw)
    dev = D.GosDevice(FakeBackend(sim), timeout_s=0.2)
    return sim, dev


def test_import_without_pynq():
    code = ("import sys; sys.modules['pynq'] = None; sys.path.insert(0, %r); import gos_driver as D;"
            "\ntry:\n  D.PynqBackend('x.bit')\nexcept D.GosError as e:\n  print('GOSERR', e)"
            % str(BOARD))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "GOSERR" in out.stdout and "pynq" in out.stdout


def test_version_check():
    with pytest.raises(D.GosVersionError):
        make(version=D.VERSION_SHELL)
    sim, dev = make(build_id=0xC88E71A0)
    assert dev.version == D.VERSION_CORE and dev.build_id_hex == "c88e71a0"


def test_counter_read_order_lo_then_hi():
    sim, dev = make()
    dev.load_net(Pkg())
    r = dev.infer(np.zeros(32, np.int8))
    assert r.total_cyc == 0x1_0000_0005
    # hi without a preceding lo read returns the last latched shadow
    sim.total_cyc = 0x2_0000_0009
    assert sim.csr.read(D.OFF_TOTAL_CYC + 4) == 1          # stale shadow from the lo read above
    assert dev.read64(D.OFF_TOTAL_CYC) == 0x2_0000_0009


def test_counter_tear_free_with_live_counter():
    sim, dev = make()
    sim.total_cyc = 0xFFFF_FFFF

    def hook(s, name, is_lo):            # the counter rolls over right after the lo read
        if name == "total_cyc" and not is_lo:
            s.total_cyc = 0x1_0000_0000
    sim.live_counter_hook = hook
    assert dev.read64(D.OFF_TOTAL_CYC) == 0xFFFF_FFFF      # lo + latched hi: consistent
    # a second read sees the rolled-over counter consistently as well
    sim.live_counter_hook = None
    assert dev.read64(D.OFF_TOTAL_CYC) == 0x1_0000_0000


def test_logits_signed_and_layers():
    sim, dev = make()
    dev.load_net(Pkg())
    r = dev.infer(np.zeros(32, np.int8))
    assert r.logits.dtype == np.int32 and r.logits.tolist() == [-5, 7, 2**31 - 1]
    assert r.layer_cyc == [3, 4] and r.mac_active == 12 and r.stall == 0 and r.violation == 0


def test_poll_timeout():
    sim, dev = make()
    dev.load_net(Pkg())
    sim.hang = True
    with pytest.raises(D.GosTimeout) as e:
        dev.infer(np.zeros(32, np.int8), timeout_s=0.02)
    assert e.value.status & D.ST_BUSY and e.value.polls > 0
    # while busy: DESC writes are refused by the driver before they reach the bus (SLVERR)
    with pytest.raises(D.GosBusyError):
        dev.write_descriptors(Pkg().desc)
    sim.hang = False
    dev.recover()
    assert dev.status() == 0
    assert dev.infer(np.zeros(32, np.int8)).logits.tolist() == [-5, 7, 2**31 - 1]


def test_error_path_and_soft_reset():
    code = (3 << 8) | 1
    state = {"refuse": True}
    good = ok_job()

    def job(sim):
        return JobOutcome(err_code=code, total_cyc=3) if state["refuse"] else good(sim)
    sim, dev = make(job)
    dev.load_net(Pkg())
    with pytest.raises(D.GosJobError) as e:
        dev.infer(np.zeros(32, np.int8))
    assert e.value.err_code == code
    assert e.value.info["rule_id"] == 3 and e.value.info["layer"] == 1
    assert e.value.info["rule"] == "K < 8"
    assert dev.read64(D.OFF_TOTAL_CYC) == 3
    state["refuse"] = False
    r = dev.infer(np.zeros(32, np.int8))       # stale error bit is cleared before the start
    assert r.cleared and r.logits[0] == -5
    dev.soft_reset()
    assert dev.status() == 0 and dev.err_code() == 0


def test_stale_done_is_cleared_before_start():
    sim, dev = make()
    dev.load_net(Pkg())
    r1 = dev.infer(np.zeros(32, np.int8))
    assert not r1.cleared                     # first job: STATUS was 0
    r2 = dev.infer(np.zeros(32, np.int8))
    assert r2.cleared                         # done from job 1 cleared by soft_reset first
    assert sim.jobs == 2


def test_load_net_readback_and_qparam_mask():
    sim, dev = make()
    p = Pkg()
    dev.load_net(p)
    assert np.array_equal(dev.read_words("WGT", p.wgt.size), p.wgt)
    q = dev.read_words("QPARAM", 4)
    assert q[0] == p.qparam[0] and q[1] == 0x2A and q[3] == 0x3F
    assert sim.n_layers == 2 and np.array_equal(sim.desc[:2], p.desc)


def test_input_validation():
    sim, dev = make()
    dev.load_net(Pkg())
    with pytest.raises(D.GosError):
        dev.infer(np.zeros((1, 4, 8), np.int8))       # NCHW is not accepted
    with pytest.raises(D.GosError):
        dev.infer(np.zeros(40, np.int8))              # wrong ACT depth


def test_sim_slverr_on_unmapped_and_ro():
    sim, dev = make()
    with pytest.raises(SimSlvErr):
        sim.csr.read(0x0E0)
    with pytest.raises(SimSlvErr):
        sim.csr.write(D.OFF_VERSION, 1)


def test_decode_err_code():
    d = D.decode_err_code((32 << 8) | 0)
    assert d["rule_id"] == 32 and "N_LAYERS" in d["rule"]


def test_output_dir_rules(tmp_path):
    with pytest.raises(SystemExit):
        bc.check_output_path(tmp_path / "results" / "hw_a1.csv", bc.SOURCE_DRYRUN)
    with pytest.raises(SystemExit):
        bc.check_output_path(tmp_path / "dryrun" / "hw_a1.csv", bc.SOURCE_HW)
    bc.check_output_path(tmp_path / "dryrun" / "hw_a1.csv", bc.SOURCE_DRYRUN)
    with pytest.raises(SystemExit):
        bc.write_csv(tmp_path / "dryrun" / "x.csv", [{"source": "hw"}], [], bc.SOURCE_DRYRUN)


def test_model_backend_bit_exact_if_available():
    data = bc.DEFAULT_DATA_DIR
    if not (data / "lenet5" / "MANIFEST.json").is_file():
        pytest.skip("data package not built")
    pytest.importorskip("gos_model_backend")
    dev = D.GosDevice(D.ModelBackend(clock_mhz=200.0))
    for net in bc.NETS:
        pkg = bc.load_package(data, net)
        dev.load_net(pkg)
        r = dev.infer(pkg.x_act[1])
        assert np.array_equal(r.logits, pkg.golden_logits[1])
        assert r.total_cyc == pkg.model_cycles["total"]["cycles"]
