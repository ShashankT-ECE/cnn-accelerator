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


# ---- fast host path ------------------------------------------------------------------------------
from gos_sim import SimCsrArray  # noqa: E402


class FastFakeBackend(FakeBackend):
    """gos_sim behind the fast path: ACT0 = the SimMem numpy array, CSR via SimCsrArray."""

    def fast_windows(self):
        ca = SimCsrArray(self.sim.csr)
        return D.FastWindows(ca, ca.rd, ca.wr, self.sim.mems["ACT0"].array, note="test")


def make_fast(job=None, fast_store="block", **kw):
    sim = SimDevice(job or ok_job(), **kw)
    return sim, D.GosDevice(FastFakeBackend(sim), timeout_s=0.2, host_path="fast",
                            fast_store=fast_store)


@pytest.mark.parametrize("store", D.FAST_STORES)
def test_fast_path_same_results_as_safe(store):
    x = (np.arange(32) * 37 % 251 - 120).astype(np.int8)
    sim_s, safe = make()
    sim_f, fast = make_fast(fast_store=store)
    for dev in (safe, fast):
        dev.load_net(Pkg())
    rs, rf = safe.infer(x), fast.infer(x)
    assert rf.logits.tolist() == rs.logits.tolist() == [-5, 7, 2**31 - 1]
    assert (rf.total_cyc, rf.mac_active, rf.stall, rf.layer_cyc, rf.violation) == \
           (rs.total_cyc, rs.mac_active, rs.stall, rs.layer_cyc, rs.violation)
    assert np.array_equal(sim_f.mems["ACT0"].array[:8], sim_s.mems["ACT0"].array[:8])
    assert np.array_equal(sim_f.mems["ACT0"].array[:8], x.view(np.uint32))
    assert rf.t_start_ns >= 0 and rf.t_run_ns >= rf.t_start_ns
    r2 = fast.infer(x)
    assert r2.cleared and sim_f.jobs == 2          # stale done cleared on the fast path too


def test_fast_path_errors_and_timeout():
    code = (3 << 8) | 1
    sim, dev = make_fast(lambda s: JobOutcome(err_code=code, total_cyc=3))
    dev.load_net(Pkg())
    with pytest.raises(D.GosJobError) as e:
        dev.infer(np.zeros(32, np.int8))
    assert e.value.err_code == code
    sim2, dev2 = make_fast()
    dev2.load_net(Pkg())
    sim2.hang = True
    with pytest.raises(D.GosTimeout):
        dev2.infer(np.zeros(32, np.int8), timeout_s=0.02)


@pytest.mark.parametrize("store", D.FAST_STORES)
def test_check_fast_path_passes_and_detects_bad_window(store):
    sim, dev = make_fast(fast_store=store)
    chk = dev.check_fast_path()
    assert chk["act0_words_checked"] == 3 * D.MEMS["ACT0"][1] // 4 and chk["fast_store"] == store

    class Lossy(FastFakeBackend):              # a window that drops the top byte of every store
        def fast_windows(self):
            fw = super().fast_windows()
            real = fw.act0

            class View(np.ndarray):
                def __setitem__(self, k, v):
                    real[k] = np.asarray(v, dtype=np.uint32) & np.uint32(0x00FF_FFFF)
            fw.act0 = real.view(View)
            return fw
    bad = D.GosDevice(Lossy(SimDevice(ok_job())), host_path="fast", fast_store=store)
    with pytest.raises(D.GosError, match="per-word readback"):
        bad.check_fast_path()
    with pytest.raises(D.GosError):
        make()[1].check_fast_path()             # safe device: no fast path to check


def test_fast_windows_from_numpy_and_devmem(tmp_path):
    csr = np.zeros(1024, np.uint32)
    act = np.zeros(8, np.uint32)
    fw = D.FastWindows.from_arrays(csr, act)
    fw.csr_wr(1, 0xDEAD_BEEF)
    assert csr[1] == 0xDEAD_BEEF and fw.csr_rd(1) == 0xDEAD_BEEF and type(fw.csr_rd(1)) is int
    with pytest.raises(D.GosError):
        D.FastWindows.from_arrays(np.zeros(4, np.uint64), act)
    f = tmp_path / "mem"
    f.write_bytes(bytes(8192))
    w = D.DevMemWindow(4096, 4096, path=str(f))         # same mmap code as /dev/mem, on a file
    w.write(8, 0x1234_5678)
    assert w.read(8) == 0x1234_5678 and w.array.dtype == np.uint32 and w.array.size == 1024
    w._mm.flush()
    assert f.read_bytes()[4096 + 8:4096 + 12] == (0x1234_5678).to_bytes(4, "little")


@pytest.mark.parametrize("fast", [False, True])
def test_control_step_never_starts_the_core(fast):
    sim, dev = make_fast() if fast else make()
    dev.load_net(Pkg())
    dev.infer(np.zeros(32, np.int8))                    # previous job: done is set
    jobs = sim.jobs
    logits, tc, polls = dev.control_step(np.ones(32, np.int8), spin_ns=200_000)
    assert sim.jobs == jobs and tc == 0 and polls >= 1  # no start, counters cleared
    assert logits.tolist() == [0, 0, 0]                 # LOGIT cleared by the soft_reset
    assert np.array_equal(sim.mems["ACT0"].array[:8], np.ones(32, np.int8).view(np.uint32))
    r = dev.infer(np.zeros(32, np.int8))                # normal job afterwards
    assert r.logits.tolist() == [-5, 7, 2**31 - 1] and not r.cleared


def test_control_step_refuses_a_running_job():
    sim, dev = make()
    dev.load_net(Pkg())
    sim.hang = True
    with pytest.raises(D.GosTimeout):
        dev.infer(np.zeros(32, np.int8), timeout_s=0.01)
    with pytest.raises(D.GosError):                     # busy core: STATUS != 0
        dev.control_step(np.zeros(32, np.int8), spin_ns=1000)


def test_mock_mmio_backend_bit_exact_both_paths():
    data = bc.DEFAULT_DATA_DIR
    if not (data / "lenet5" / "MANIFEST.json").is_file():
        pytest.skip("data package not built")
    import mock_mmio as MM
    be = MM.MockMmioBackend(data, busy_polls=2)
    assert be.source == bc.SOURCE_DRYRUN
    devs = [D.GosDevice(be), D.GosDevice(be, host_path="fast"),
            D.GosDevice(be, host_path="fast", fast_store="words32")]
    devs[1].check_fast_path()
    for net in bc.NETS:
        pkg = bc.load_package(data, net)
        for dev in devs:
            dev.load_net(pkg)
            for i in (0, 7):
                r = dev.infer(pkg.x_act[i])
                assert np.array_equal(r.logits, pkg.golden_logits[i])
                assert r.total_cyc == pkg.model_cycles["total"]["cycles"] and r.polls == 3
        bad = pkg.x_act[0].copy()
        bad[0] ^= 1                                      # an input that is not in the package
        with pytest.raises(D.GosJobError):
            devs[1].infer(bad)
        devs[1].recover()
    assert be.emulation_cost_ns(reps=10)["median_ns"] > 0


def test_model_backend_fast_path_bit_exact_if_available():
    data = bc.DEFAULT_DATA_DIR
    if not (data / "lenet5" / "MANIFEST.json").is_file():
        pytest.skip("data package not built")
    pytest.importorskip("gos_model_backend")
    dev = D.GosDevice(D.ModelBackend(clock_mhz=200.0), host_path="fast")
    dev.check_fast_path()
    pkg = bc.load_package(data, "cifar10")
    dev.load_net(pkg)
    r = dev.infer(pkg.x_act[3])
    assert np.array_equal(r.logits, pkg.golden_logits[3])
    assert r.layer_cyc == [L["cycles"] for L in pkg.model_cycles["layers"]]
