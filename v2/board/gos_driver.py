"""V2 gos_ accelerator host driver (register level) with two backends and one API.

    dev = GosDevice(PynqBackend("bit/gos_200.bit"))        # KV260, PYNQ overlay
    dev = GosDevice(ModelBackend(clock_mhz=200.0))          # laptop dry run (gos_model_backend)
    dev.load_net(pkg)                                       # WGT, QPARAM, DESC, N_LAYERS (once per net)
    r = dev.infer(pkg.x_act[i])                             # one image -> InferResult

Contract: v2/docs/FORMATS.md ("CSR register map", "PL top ports and address map", §1-§5) and
DECISIONS D2 (raw INT32 logits), D10 (STALL = 0), D12 (counters, STATUS, soft_reset, busy lock),
D13 (C_START = 3). This module needs only numpy (pynq is imported lazily inside PynqBackend);
it does no packing — images come ready-made from the data package (make_board_data.py, which
uses v2/model/gos_pack.py).

Register access rules followed here:
  * every CSR access is one 32-bit read/write at a mapped offset (unmapped / RO writes answer
    SLVERR on the hardware, which on the PS can abort the process: never touch them);
  * 64-bit counters: read lo, then hi (reading lo latches hi into a shadow);
  * N_LAYERS / DESC are written only when STATUS.busy = 0 (busy lock: SLVERR while busy);
  * memories are written as 64-bit words = two 32-bit MMIO words, little-endian
    (bits 31:0 at 8*w, bits 63:32 at 8*w + 4); ACT is written only while the core is idle.

Job start protocol (driver choice, see README "Assumptions"): STATUS.done / .error are sticky
until the next start or soft_reset. AXI gives no ordering between a write (start) and a later read
(STATUS) on the interconnect, so a STATUS read could overtake the start and see the previous job's
done bit. Before each start the driver therefore makes sure STATUS reads 0 (issuing a soft_reset,
which keeps memories and DESC/N_LAYERS, and polling until STATUS == 0 when a done/error bit is
set); after that a done/error bit can only come from the new job.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# ---- address map (FORMATS.md "Address map (PS view, HPM0_FPD)") -------------------------
CSR_BASE, CSR_SIZE = 0xA000_0000, 4 * 1024
MEMS = {  # name: (base, size in bytes); 64-bit words
    "QPARAM": (0xA001_0000, 4 * 1024),
    "ACT0":   (0xA002_0000, 32 * 1024),
    "ACT1":   (0xA004_0000, 32 * 1024),
    "WGT":    (0xA008_0000, 128 * 1024),
}

# ---- CSR offsets (FORMATS.md "CSR register map") -------------------------------------------
OFF_CTRL = 0x000
OFF_STATUS = 0x004
OFF_N_LAYERS = 0x008
OFF_ERR_CODE = 0x00C
OFF_TOTAL_CYC = 0x010      # lo; hi at +4
OFF_MAC_ACTIVE = 0x018
OFF_STALL = 0x020
OFF_VIOLATION = 0x028      # PS_BUSY_VIOLATION
OFF_LAYER_CYC = 0x040      # + 4*l, l = 0..7
OFF_LOGIT = 0x080          # + 4*i, i = 0..15
OFF_VERSION = 0x0F8
OFF_BUILD_ID = 0x0FC
OFF_DESC = 0x100           # + 0x40*l + 4*w

CTRL_START = 0x1
CTRL_SOFT_RESET = 0x2
ST_BUSY, ST_DONE, ST_ERROR = 0x1, 0x2, 0x4

VERSION_CORE = 0x474F5302  # "GOS", 2
VERSION_SHELL = 0x474F5300
MAX_LAYERS = 8
DESC_WORDS = 16
N_LOGITS = 16

# ERR_CODE rule ids (FORMATS.md §5 "Rule ids and ERR_CODE"; priority = ascending id)
RULE_NAMES = {
    1: "pool_en and OH odd", 2: "pool_en and OW odd", 3: "K < 8", 4: "WGT_END > 16383",
    5: "IN_END > 4095", 6: "OUT_END > 4095", 7: "QP_END > 255", 8: "IC == 0", 9: "OC == 0",
    10: "IH == 0", 11: "IW == 0", 12: "KH == 0", 13: "KW == 0", 14: "OH == 0", 15: "OW == 0",
    16: "K == 0", 17: "IN_WPR == 0", 18: "IN_PLANE == 0", 19: "OC_TILES == 0",
    20: "OW_TILES == 0", 21: "OUT_W == 0", 22: "OUT_H == 0", 23: "OUT_WPR == 0",
    24: "OUT_PLANE == 0", 25: "OH > IH", 26: "OW > IW", 27: "out_raw and OC > 16",
    32: "N_LAYERS not in 1..8",
}
VIOLATION_BITS = {0: "ACT0 PS access while busy", 1: "ACT1 PS access while busy",
                  4: "array overrun"}


def decode_err_code(code: int) -> dict:
    """ERR_CODE = {16'b0, rule_id[7:0], 5'b0, layer[2:0]}."""
    rule = (code >> 8) & 0xFF
    return {"err_code": code, "rule_id": rule, "layer": code & 0x7,
            "rule": RULE_NAMES.get(rule, f"unknown rule {rule}")}


def decode_violation(flags: int) -> list[str]:
    return [name for bit, name in VIOLATION_BITS.items() if flags >> bit & 1]


def to_int32(u: int) -> int:
    u &= 0xFFFF_FFFF
    return u - (1 << 32) if u & 0x8000_0000 else u


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- errors ------------------------------------------------------------------------------
class GosError(RuntimeError):
    pass


class GosVersionError(GosError):
    pass


class GosBusyError(GosError):
    pass


class GosTimeout(GosError):
    def __init__(self, msg, status=None, polls=0):
        super().__init__(msg)
        self.status, self.polls = status, polls


class GosJobError(GosError):
    """The core refused the job (config checker): STATUS.error with ERR_CODE."""

    def __init__(self, err_code: int, status: int, violation: int = 0):
        self.info = decode_err_code(err_code)
        self.err_code, self.status, self.violation = err_code, status, violation
        super().__init__(f"job refused: ERR_CODE=0x{err_code:08X} rule {self.info['rule_id']} "
                         f"({self.info['rule']}) at layer {self.info['layer']}; "
                         f"STATUS=0x{status:X}")


# ---- results -----------------------------------------------------------------------------
@dataclass
class InferResult:
    logits: np.ndarray                  # int32 [OC] raw v = acc + q_bias (LOGIT[0..OC-1])
    total_cyc: int | None = None        # TOTAL_CYC (64-bit)
    layer_cyc: list = field(default_factory=list)   # LAYER_CYC[0..N_LAYERS-1]
    mac_active: int | None = None       # MAC_ACTIVE (64-bit, whole job; no per-layer counter)
    stall: int | None = None            # STALL (64-bit; 0 by construction, D10)
    violation: int | None = None        # PS_BUSY_VIOLATION flags
    polls: int = 0                      # STATUS reads until done
    cleared: bool = False               # a soft_reset was issued before start
    t_write_ns: int = 0                 # input write into ACT0
    t_clear_ns: int = 0                 # pre-start clear of a stale done/error (soft_reset), if any
    t_run_ns: int = 0                   # CTRL.start write + poll until done
    t_logit_ns: int = 0                 # LOGIT[0..OC-1] reads
    t_counter_ns: int = 0               # counter reads (measurement overhead, not end-to-end)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["logits"] = [int(v) for v in self.logits]
        return d


# ---- backends ----------------------------------------------------------------------------
class MmioBackend:
    """A set of MMIO-like windows: .csr and .mems[name], each with read(off) -> int,
    write(off, int) (32-bit, byte offsets) and optionally .array (numpy uint32 view)."""

    kind = "abstract"
    source = ""

    def __init__(self, csr, mems: dict):
        self.csr = csr
        self.mems = mems

    def fclk0_mhz(self) -> float:
        raise NotImplementedError

    def set_fclk0(self, mhz: float) -> float:
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.kind}


class PynqBackend(MmioBackend):
    """KV260 via PYNQ: Overlay(<name>.bit) with <name>.hwh side by side; pynq.MMIO windows.

    pynq is imported here (lazily) so that importing this module never needs it.
    """

    kind = "pynq"
    source = "hw"

    def __init__(self, bit: str | Path, download: bool = True):
        try:
            from pynq import MMIO, Overlay  # noqa: WPS433 (lazy on purpose)
            from pynq.ps import Clocks
        except ImportError as e:  # pragma: no cover - board only
            raise GosError(f"PynqBackend needs the pynq package (run on the KV260): {e}") from e
        bit = Path(bit)
        hwh = bit.with_suffix(".hwh")
        if not bit.is_file() or not hwh.is_file():
            raise GosError(f"overlay needs {bit} and {hwh} side by side")
        self.bit = bit
        self.bit_sha256 = sha256_file(bit)
        self.hwh_sha256 = sha256_file(hwh)
        self._Clocks = Clocks
        self.overlay = Overlay(str(bit), download=download)
        csr = MMIO(CSR_BASE, CSR_SIZE)
        mems = {n: MMIO(b, s) for n, (b, s) in MEMS.items()}
        super().__init__(csr, mems)

    def fclk0_mhz(self) -> float:
        return float(self._Clocks.fclk0_mhz)

    def set_fclk0(self, mhz: float) -> float:
        self._Clocks.fclk0_mhz = float(mhz)
        return self.fclk0_mhz()

    def info(self) -> dict:
        return {"backend": self.kind, "bit": str(self.bit), "bit_sha256": self.bit_sha256,
                "hwh_sha256": self.hwh_sha256}


def ModelBackend(*args, **kwargs):  # noqa: N802 - factory with class-like name
    """Laptop dry-run backend (gos_golden + gos_cycle_model behind a simulated register map).
    Lazily imported so that the board never needs v2/model."""
    from gos_model_backend import ModelBackend as _MB
    return _MB(*args, **kwargs)


def open_backend(kind: str, **kw) -> MmioBackend:
    if kind == "pynq":
        return PynqBackend(kw["bit"], download=kw.get("download", True))
    if kind == "model":
        return ModelBackend(clock_mhz=kw.get("clock_mhz", 200.0))
    raise ValueError(f"unknown backend {kind!r}")


# ---- device ------------------------------------------------------------------------------
WRITE_MODES = ("elem", "mmio", "slice")


class GosDevice:
    """Register-level driver of the gos_ core. Same code for every backend."""

    def __init__(self, backend: MmioBackend, expect_version=(VERSION_CORE,),
                 timeout_s: float = 1.0, write_mode: str = "elem"):
        if write_mode not in WRITE_MODES:
            raise ValueError(f"write_mode must be one of {WRITE_MODES}")
        self.be = backend
        self.csr = backend.csr
        self.timeout_s = float(timeout_s)
        self.write_mode = write_mode
        self.version = self.csr.read(OFF_VERSION)
        self.build_id = self.csr.read(OFF_BUILD_ID)
        if self.version not in tuple(expect_version):
            raise GosVersionError(
                f"VERSION 0x{self.version:08X} not in "
                f"{[f'0x{v:08X}' for v in expect_version]} (shell = 0x{VERSION_SHELL:08X})")
        self.n_layers = 0
        self.oc = 0
        self.act_in_words = 0
        self._desc = None                 # uint32 [n_layers, 16] of the loaded net
        self._desc_valid = False          # CSR DESC/N_LAYERS hold self._desc
        self.net_name = None

    # -- identification / clock -------------------------------------------------------
    @property
    def build_id_hex(self) -> str:
        return f"{self.build_id:08x}"

    def fclk0_mhz(self) -> float:
        return self.be.fclk0_mhz()

    def set_fclk0(self, mhz: float) -> float:
        """Request pl_clk0 = mhz; returns the read-back actual frequency."""
        self._require_idle()
        return self.be.set_fclk0(mhz)

    # -- CSR helpers ------------------------------------------------------------------
    def status(self) -> int:
        return self.csr.read(OFF_STATUS)

    def read64(self, lo_off: int) -> int:
        """Tear-free 64-bit counter read: lo first (latches hi), then hi."""
        lo = self.csr.read(lo_off)
        hi = self.csr.read(lo_off + 4)
        return (hi << 32) | lo

    def err_code(self) -> int:
        return self.csr.read(OFF_ERR_CODE)

    def violation(self) -> int:
        return self.csr.read(OFF_VIOLATION)

    def _require_idle(self):
        st = self.status()
        if st & ST_BUSY:
            raise GosBusyError(f"core busy (STATUS=0x{st:X}); DESC/N_LAYERS/ACT writes refused")
        return st

    # -- memories ---------------------------------------------------------------------
    def _write_u32(self, name: str, u32: np.ndarray, word32_off: int = 0):
        mm = self.be.mems[name]
        u32 = np.ascontiguousarray(u32, dtype=np.uint32)
        size = MEMS[name][1] // 4
        if word32_off < 0 or word32_off + u32.size > size:
            raise GosError(f"{name}: write of {u32.size} words at {word32_off} exceeds window")
        mode = self.write_mode if hasattr(mm, "array") else "mmio"
        if mode == "slice":            # numpy slice copy: store width chosen by numpy/libc
            mm.array[word32_off:word32_off + u32.size] = u32
        elif mode == "elem":           # one 32-bit store per word (what pynq MMIO.write does)
            arr = mm.array
            for i, v in enumerate(u32.tolist(), word32_off):
                arr[i] = v
        else:                          # MMIO.write per word (reference path)
            for i, v in enumerate(u32.tolist(), word32_off):
                mm.write(4 * i, v)

    def write_words(self, name: str, words, word_off: int = 0, verify: bool = False):
        """Write 64-bit words (uint64) from 64-bit word offset word_off."""
        w = np.ascontiguousarray(words, dtype=np.uint64)
        self._write_u32(name, w.view(np.uint32), 2 * word_off)
        if verify:
            got = self.read_words(name, w.size, word_off)
            if name == "QPARAM":       # odd combined words keep only s[5:0] (FORMATS §3)
                exp = w.copy()
                odd = (np.arange(word_off, word_off + w.size) % 2) == 1
                exp[odd] &= np.uint64(0x3F)
            else:
                exp = w
            bad = np.flatnonzero(got != exp)
            if bad.size:
                i = int(bad[0])
                raise GosError(f"{name}: readback mismatch in {bad.size}/{w.size} words; first "
                               f"word {word_off + i}: got 0x{int(got[i]):016X} "
                               f"expected 0x{int(exp[i]):016X}")

    def read_words(self, name: str, n: int, word_off: int = 0) -> np.ndarray:
        mm = self.be.mems[name]
        out = np.empty(2 * n, dtype=np.uint32)
        for i in range(2 * n):
            out[i] = mm.read(4 * (2 * word_off + i))
        return out.view(np.uint64)

    # -- descriptors ------------------------------------------------------------------
    def write_descriptors(self, desc, n_layers: int | None = None, verify: bool = True):
        """Write DESC[l][w] for the rows of desc (uint32 [L, 16]) and N_LAYERS (default L).
        Only while idle (busy lock). Marks the loaded net's descriptors as overwritten unless
        desc equals them."""
        desc = np.asarray(desc, dtype=np.uint32)
        assert desc.ndim == 2 and desc.shape[1] == DESC_WORDS and desc.shape[0] <= MAX_LAYERS
        n = desc.shape[0] if n_layers is None else int(n_layers)
        self._require_idle()
        for l in range(desc.shape[0]):
            for w in range(DESC_WORDS):
                self.csr.write(OFF_DESC + 0x40 * l + 4 * w, int(desc[l, w]))
        self.csr.write(OFF_N_LAYERS, n & 0xF)
        if verify:
            for l in range(desc.shape[0]):
                for w in range(DESC_WORDS):
                    got = self.csr.read(OFF_DESC + 0x40 * l + 4 * w)
                    if got != int(desc[l, w]):
                        raise GosError(f"DESC[{l}][{w}] readback 0x{got:08X} != 0x{int(desc[l, w]):08X}")
            got = self.csr.read(OFF_N_LAYERS)
            if got != (n & 0xF):
                raise GosError(f"N_LAYERS readback {got} != {n & 0xF}")
        self._desc_valid = (self._desc is not None and n == self.n_layers
                            and desc.shape == self._desc.shape and np.array_equal(desc, self._desc))

    # -- network ----------------------------------------------------------------------
    def load_net(self, pkg, verify: bool = True):
        """Write the WGT and QPARAM images and the descriptors + N_LAYERS of one net
        (pkg: board_common.Package or any object with wgt, qparam, desc, net)."""
        self._require_idle()
        self.write_words("WGT", pkg.wgt, 0, verify=verify)
        self.write_words("QPARAM", pkg.qparam, 0, verify=verify)
        self._desc = np.asarray(pkg.desc, dtype=np.uint32).copy()
        self.n_layers = self._desc.shape[0]
        self.oc = int(pkg.net["OC"])
        self.act_in_words = int(pkg.net["act_in_words"])
        self.net_name = pkg.net["net"]
        self.write_descriptors(self._desc, self.n_layers, verify=verify)
        assert self._desc_valid

    # -- job control ------------------------------------------------------------------
    def soft_reset(self, timeout_s: float | None = None):
        """CTRL.soft_reset; waits until STATUS reads 0 (clears core state, status, counters,
        LOGIT and error flags; memories and DESC/N_LAYERS are kept, D12-4)."""
        self.csr.write(OFF_CTRL, CTRL_SOFT_RESET)
        self._wait_status_clear(timeout_s)

    def _wait_status_clear(self, timeout_s=None):
        deadline = time.perf_counter() + (self.timeout_s if timeout_s is None else timeout_s)
        polls = 0
        while True:
            st = self.status()
            polls += 1
            if st & 0x7 == 0:
                return polls
            if time.perf_counter() > deadline:
                raise GosTimeout(f"STATUS did not clear after soft_reset (0x{st:X})", st, polls)

    def _clear_for_start(self) -> bool:
        st = self.status()
        if st & ST_BUSY:
            raise GosBusyError(f"start requested while busy (STATUS=0x{st:X})")
        if st & (ST_DONE | ST_ERROR):
            self.csr.write(OFF_CTRL, CTRL_SOFT_RESET)
            self._wait_status_clear()
            return True
        return False

    def start_and_wait(self, timeout_s: float | None = None, clear: bool = True) -> tuple[int, bool]:
        """Start a job with the current DESC/N_LAYERS and memories; poll STATUS.
        Returns (polls, cleared). Raises GosJobError (STATUS.error) or GosTimeout.
        clear=False: the caller already ran _clear_for_start() (infer times it separately)."""
        cleared = self._clear_for_start() if clear else False
        timeout = self.timeout_s if timeout_s is None else timeout_s
        self.csr.write(OFF_CTRL, CTRL_START)
        t_end = time.perf_counter_ns() + int(timeout * 1e9)
        polls = 0
        read = self.csr.read
        while True:
            st = read(OFF_STATUS)
            polls += 1
            if st & ST_ERROR:
                raise GosJobError(self.err_code(), st, self.violation())
            if st & ST_DONE:
                return polls, cleared
            if time.perf_counter_ns() > t_end:
                raise GosTimeout(f"job not done after {timeout} s (STATUS=0x{st:X}, "
                                 f"{polls} polls)", st, polls)

    def read_logits(self, oc: int | None = None) -> np.ndarray:
        oc = self.oc if oc is None else oc
        read = self.csr.read
        return np.array([to_int32(read(OFF_LOGIT + 4 * i)) for i in range(oc)], dtype=np.int32)

    def read_counters(self, n_layers: int | None = None) -> dict:
        n = self.n_layers if n_layers is None else n_layers
        return {"total_cyc": self.read64(OFF_TOTAL_CYC),
                "mac_active": self.read64(OFF_MAC_ACTIVE),
                "stall": self.read64(OFF_STALL),
                "layer_cyc": [self.csr.read(OFF_LAYER_CYC + 4 * l) for l in range(n)],
                "violation": self.violation()}

    def write_input(self, x):
        """Input image -> ACT0 from word 0. x: the ACT0 byte image (int8/uint8 [8*D]), or
        uint64 [D] / uint32 [2*D] words (data package key 'x')."""
        x = np.ascontiguousarray(x)
        if x.dtype in (np.int8, np.uint8):
            if x.size % 8:
                raise GosError("ACT byte image length must be a multiple of 8")
            u32 = x.view(np.uint32)
        elif x.dtype == np.uint64:
            u32 = x.view(np.uint32)
        elif x.dtype == np.uint32:
            u32 = x
        else:
            raise GosError(f"input must be the packed ACT image (int8/uint8/uint32/uint64), "
                           f"got {x.dtype} {x.shape}")
        if x.ndim != 1:
            raise GosError(f"input must be 1-D (one packed ACT image), got shape {x.shape}")
        if self.act_in_words and u32.size != 2 * self.act_in_words:
            raise GosError(f"input has {u32.size // 2} ACT words, net expects {self.act_in_words}")
        self._write_u32("ACT0", u32, 0)

    def infer(self, x, read_counters: bool = True, timeout_s: float | None = None) -> InferResult:
        """One inference: input -> ACT0, (descriptors if overwritten), start, poll, LOGIT read,
        counters (optional). Phase times use time.perf_counter_ns."""
        if self._desc is None:
            raise GosError("load_net first")
        if not self._desc_valid:
            self.write_descriptors(self._desc, self.n_layers)
        t0 = time.perf_counter_ns()
        self.write_input(x)
        tc = time.perf_counter_ns()
        cleared = self._clear_for_start()      # stale done/error -> soft_reset; own phase, not compute
        t1 = time.perf_counter_ns()
        polls, _ = self.start_and_wait(timeout_s, clear=False)
        t2 = time.perf_counter_ns()
        logits = self.read_logits()
        t3 = time.perf_counter_ns()
        r = InferResult(logits=logits, polls=polls, cleared=cleared,
                        t_write_ns=tc - t0, t_clear_ns=t1 - tc, t_run_ns=t2 - t1, t_logit_ns=t3 - t2)
        if read_counters:
            c = self.read_counters()
            r.total_cyc, r.mac_active, r.stall = c["total_cyc"], c["mac_active"], c["stall"]
            r.layer_cyc, r.violation = c["layer_cyc"], c["violation"]
            r.t_counter_ns = time.perf_counter_ns() - t3
        return r

    def recover(self):
        """After a timeout / refused job: soft_reset and mark DESC for rewrite."""
        self.soft_reset()
        self._desc_valid = False
