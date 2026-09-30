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
    t_start_ns: int = 0                 # CTRL.start write alone (part of t_run_ns)
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

    def fast_windows(self) -> "FastWindows":
        """Numpy views of the mapped windows for the fast host path (default: the .array of
        every window, CSR scalars through a memoryview)."""
        if not hasattr(self.csr, "array") or not hasattr(self.mems["ACT0"], "array"):
            raise GosError(f"backend {self.kind}: no mapped numpy windows for the fast host path")
        return FastWindows.from_arrays(self.csr.array, self.mems["ACT0"].array,
                                       note=f"{self.kind}: .array views")


@dataclass
class FastWindows:
    """What the fast host path needs: the CSR window as a numpy uint32 array (block reads of
    side-effect-free registers: LOGIT, LAYER_CYC), scalar CSR accessors rd(i)/wr(i, v) with i a
    32-bit word index (one 32-bit access each), and the ACT0 window as a writable numpy uint32
    array."""
    csr_arr: object
    csr_rd: object
    csr_wr: object
    act0: np.ndarray
    note: str = ""

    @classmethod
    def from_arrays(cls, csr_arr: np.ndarray, act0: np.ndarray, note: str = "") -> "FastWindows":
        mv = memoryview(csr_arr)
        if mv.format not in ("I", "=I", "<I") or mv.itemsize != 4:
            raise GosError(f"CSR window view has format {mv.format!r}, expected uint32")
        return cls(csr_arr, mv.__getitem__, mv.__setitem__, act0, note)


class DevMemWindow:
    """A /dev/mem mmap of one PL window (fallback when pynq.MMIO has no .array); .array is a
    numpy uint32 view, read()/write() are the per-word 32-bit accesses of the safe path."""

    def __init__(self, base: int, size: int, path: str = "/dev/mem"):
        import mmap
        import os
        fd = os.open(path, os.O_RDWR | os.O_SYNC)
        try:
            self._mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                                 offset=base)
        finally:
            os.close(fd)
        self.base_addr, self.length = base, size
        self.array = np.frombuffer(self._mm, dtype=np.uint32)

    def read(self, off: int) -> int:
        return int(self.array[off >> 2])

    def write(self, off: int, val: int):
        self.array[off >> 2] = val & 0xFFFF_FFFF


FCLK_SET_TOL_MHZ = 0.1       # pl_clk0 read back vs the target (the bitstream's closed clock)
_CRL_APB_BASE, _OFF_PL0_REF_CTRL = 0xFF5E0000, 0xC0     # ZynqMP CRL_APB.PL0_REF_CTRL (read only here)


def best_pl_dividers(src_mhz: float, target_mhz: float, width: int = 6) -> tuple[int, int, float]:
    """(div0, div1, f) with f = src / (div0 * div1) closest to target; two width-bit dividers
    (1 .. 2^width - 1). A tie goes to the lower frequency."""
    best = None
    for d0 in range(1, 1 << width):
        for d1 in range(1, d0 + 1):
            if d1 >= (1 << width):
                break
            f = src_mhz / (d0 * d1)
            key = (abs(f - target_mhz), f, d1)            # same product: div1 as small as possible
            if best is None or key < best[0]:
                best = (key, d0, d1, f)
    return best[1], best[2], best[3]


def set_fclk0_exact(clocks, mmio_cls, mhz: float, tol: float = FCLK_SET_TOL_MHZ) -> float:
    """Set pl_clk0 to mhz and verify the read-back within tol, else raise GosError.

    pynq's `Clocks.fclk0_mhz = x` silently programs the CLOSEST frequency the current source PLL
    can divide down to (e.g. 333.33 MHz for a 300 MHz request on a 1000 MHz PLL), which may be
    above the timing-closed clock. Here the dividers are computed first from the source-PLL
    frequency (= read-back x the dividers in CRL_APB.PL0_REF_CTRL) and NOTHING is written unless
    the result is within tol of the target. Returns the read-back clock."""
    v = mmio_cls(_CRL_APB_BASE, 0x100).read(_OFF_PL0_REF_CTRL)
    d0, d1 = (v >> 8) & 0x3F, (v >> 16) & 0x3F
    now = float(clocks.fclk0_mhz)
    if not d0 or not d1:
        raise GosError(f"pl_clk0: PL0_REF_CTRL 0x{v:08X} has a zero divider")
    src = now * d0 * d1
    n0, n1, f = best_pl_dividers(src, float(mhz))
    if abs(f - float(mhz)) > tol:
        raise GosError(f"pl_clk0 {float(mhz):.6f} MHz is not reachable on this board: source PLL "
                       f"{src:.3f} MHz (PL0_REF_CTRL SRCSEL {v & 7}), nearest {f:.6f} MHz "
                       f"(div {n0} x {n1}), tolerance +-{tol} MHz; pl_clk0 left at {now:.6f} MHz")
    clocks.set_pl_clk(0, div0=n0, div1=n1)
    rb = float(clocks.fclk0_mhz)
    if abs(rb - float(mhz)) > tol:
        raise GosError(f"pl_clk0 read back {rb:.6f} MHz != target {float(mhz):.6f} MHz "
                       f"(+-{tol} MHz) after setting div {n0} x {n1}")
    return rb


class PynqBackend(MmioBackend):
    """KV260 via PYNQ: Overlay(<name>.bit) with <name>.hwh side by side; pynq.MMIO windows.

    pynq is imported here (lazily) so that importing this module never needs it.

    clock_mhz: the bitstream's closed pl_clk0. PYNQ on the KV260 does not program the PS PLLs of
    the Vivado design (it only copies the .hwh dividers onto the boot image's PLLs), so after the
    overlay is loaded pl_clk0 is set to clock_mhz and the read-back must be within
    FCLK_SET_TOL_MHZ, else GosError (set_fclk0_exact). None: the clock is left as PYNQ set it.
    """

    kind = "pynq"
    source = "hw"

    def __init__(self, bit: str | Path, download: bool = True, clock_mhz: float | None = None):
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
        self._MMIO = MMIO
        self.overlay = Overlay(str(bit), download=download)
        self.clock_loaded_mhz = float(Clocks.fclk0_mhz)      # as PYNQ left it after the load
        self.clock_target_mhz = None if clock_mhz is None else float(clock_mhz)
        if clock_mhz is not None:
            self.set_fclk0_exact(clock_mhz)
        csr = MMIO(CSR_BASE, CSR_SIZE)
        mems = {n: MMIO(b, s) for n, (b, s) in MEMS.items()}
        super().__init__(csr, mems)

    def fclk0_mhz(self) -> float:
        return float(self._Clocks.fclk0_mhz)

    def set_fclk0(self, mhz: float) -> float:
        self._Clocks.fclk0_mhz = float(mhz)
        return self.fclk0_mhz()

    def set_fclk0_exact(self, mhz: float, tol: float = FCLK_SET_TOL_MHZ) -> float:
        return set_fclk0_exact(self._Clocks, self._MMIO, mhz, tol)

    def info(self) -> dict:
        return {"backend": self.kind, "bit": str(self.bit), "bit_sha256": self.bit_sha256,
                "hwh_sha256": self.hwh_sha256, "clock_target_mhz": self.clock_target_mhz,
                "clock_after_load_mhz": self.clock_loaded_mhz}

    def fast_windows(self) -> FastWindows:
        """pynq MMIO.array views (mapped once by pynq at construction); if a pynq version has no
        .array, the windows are mapped once more from /dev/mem (DevMemWindow)."""
        csr, act0 = self.csr, self.mems["ACT0"]
        if hasattr(csr, "array") and hasattr(act0, "array"):
            return FastWindows.from_arrays(csr.array, act0.array, note="pynq MMIO.array views")
        self._devmem = {"csr": DevMemWindow(CSR_BASE, CSR_SIZE),
                        "ACT0": DevMemWindow(*MEMS["ACT0"])}
        return FastWindows.from_arrays(self._devmem["csr"].array, self._devmem["ACT0"].array,
                                       note="own /dev/mem mmap (pynq MMIO has no .array)")


def ModelBackend(*args, **kwargs):  # noqa: N802 - factory with class-like name
    """Laptop dry-run backend (gos_golden + gos_cycle_model behind a simulated register map).
    Lazily imported so that the board never needs v2/model."""
    from gos_model_backend import ModelBackend as _MB
    return _MB(*args, **kwargs)


def MockMmioBackend(*args, **kwargs):  # noqa: N802 - factory with class-like name
    """Laptop host-overhead backend (mock_mmio.py): numpy arrays stand in for the mapped windows;
    jobs finish after a fixed number of STATUS polls with the golden logits. dryrun_model only."""
    from mock_mmio import MockMmioBackend as _MM
    return _MM(*args, **kwargs)


def open_backend(kind: str, **kw) -> MmioBackend:
    if kind == "pynq":
        return PynqBackend(kw["bit"], download=kw.get("download", True))
    if kind == "model":
        return ModelBackend(clock_mhz=kw.get("clock_mhz", 200.0))
    if kind == "mock":
        return MockMmioBackend(kw["data_dir"], clock_mhz=kw.get("clock_mhz", 200.0))
    raise ValueError(f"unknown backend {kind!r}")


# ---- device ------------------------------------------------------------------------------
WRITE_MODES = ("elem", "mmio", "slice")
HOST_PATHS = ("safe", "fast")
FAST_STORES = ("block", "words32")   # fast path: one contiguous numpy copy / lo+hi strided copies
_I_CTRL, _I_STATUS = OFF_CTRL >> 2, OFF_STATUS >> 2
_I_LOGIT, _I_LAYER_CYC = OFF_LOGIT >> 2, OFF_LAYER_CYC >> 2
_POLL_CLOCK_EVERY = 64                # fast poll loop: read the clock every 64 STATUS polls


class GosDevice:
    """Register-level driver of the gos_ core. Same code for every backend."""

    def __init__(self, backend: MmioBackend, expect_version=(VERSION_CORE,),
                 timeout_s: float = 1.0, write_mode: str = "elem", host_path: str = "safe",
                 fast_store: str = "block"):
        if write_mode not in WRITE_MODES:
            raise ValueError(f"write_mode must be one of {WRITE_MODES}")
        if host_path not in HOST_PATHS:
            raise ValueError(f"host_path must be one of {HOST_PATHS}")
        if fast_store not in FAST_STORES:
            raise ValueError(f"fast_store must be one of {FAST_STORES}")
        self.be = backend
        self.csr = backend.csr
        self.timeout_s = float(timeout_s)
        self.write_mode = write_mode
        self.host_path = host_path
        self.fast_store = fast_store
        self._fast = host_path == "fast"
        self._fw = backend.fast_windows() if self._fast else None
        self.version = self.csr.read(OFF_VERSION)
        self.build_id = self.csr.read(OFF_BUILD_ID)
        if self.version not in tuple(expect_version):
            raise GosVersionError(
                f"VERSION 0x{self.version:08X} not in "
                f"{[f'0x{v:08X}' for v in expect_version]} (shell = 0x{VERSION_SHELL:08X})")
        self.n_layers = 0
        self.oc = 0
        self.act_in_words = 0
        self._last_t_start_ns = 0
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
    @property
    def host_path_desc(self) -> str:
        if not self._fast:
            return f"safe (write-mode {self.write_mode})"
        return f"fast (store {self.fast_store}; {self._fw.note})"

    def status(self) -> int:
        if self._fast:
            return self._fw.csr_rd(_I_STATUS)
        return self.csr.read(OFF_STATUS)

    def read64(self, lo_off: int) -> int:
        """Tear-free 64-bit counter read: lo first (latches hi), then hi (two ordered scalar
        reads on both host paths; never a block read)."""
        if self._fast:
            rd = self._fw.csr_rd
            lo = rd(lo_off >> 2)
            hi = rd((lo_off >> 2) + 1)
        else:
            lo = self.csr.read(lo_off)
            hi = self.csr.read(lo_off + 4)
        return (hi << 32) | lo

    def err_code(self) -> int:
        if self._fast:
            return self._fw.csr_rd(OFF_ERR_CODE >> 2)
        return self.csr.read(OFF_ERR_CODE)

    def violation(self) -> int:
        if self._fast:
            return self._fw.csr_rd(OFF_VIOLATION >> 2)
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
        status = self.status
        while True:
            st = status()
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
            if self._fast:
                self._fw.csr_wr(_I_CTRL, CTRL_SOFT_RESET)
            else:
                self.csr.write(OFF_CTRL, CTRL_SOFT_RESET)
            self._wait_status_clear()
            return True
        return False

    def _start_and_poll_fast(self, timeout: float) -> tuple[int, int]:
        """CTRL.start + tight STATUS poll (fast path). Returns (polls, t_start_ns)."""
        rd, wr = self._fw.csr_rd, self._fw.csr_wr
        pc = time.perf_counter_ns
        t0 = pc()
        wr(_I_CTRL, CTRL_START)
        t1 = pc()
        t_end = t0 + int(timeout * 1e9)
        polls = 0
        while True:
            st = rd(_I_STATUS)
            polls += 1
            if st & 0x6:
                break
            if not polls % _POLL_CLOCK_EVERY and pc() > t_end:
                raise GosTimeout(f"job not done after {timeout} s (STATUS=0x{st:X}, "
                                 f"{polls} polls)", st, polls)
        if st & ST_ERROR:
            raise GosJobError(self.err_code(), st, self.violation())
        self._last_t_start_ns = t1 - t0
        return polls, t1 - t0

    def start_and_wait(self, timeout_s: float | None = None, clear: bool = True) -> tuple[int, bool]:
        """Start a job with the current DESC/N_LAYERS and memories; poll STATUS.
        Returns (polls, cleared). Raises GosJobError (STATUS.error) or GosTimeout.
        clear=False: the caller already ran _clear_for_start() (infer times it separately)."""
        cleared = self._clear_for_start() if clear else False
        timeout = self.timeout_s if timeout_s is None else timeout_s
        if self._fast:
            polls, _ = self._start_and_poll_fast(timeout)
            return polls, cleared
        t0 = time.perf_counter_ns()
        self.csr.write(OFF_CTRL, CTRL_START)
        self._last_t_start_ns = time.perf_counter_ns() - t0
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

    def _csr_block(self, i0: int, n: int) -> np.ndarray:
        """Fast path: n consecutive side-effect-free CSR words from word index i0 as ONE
        vectorized read (block: contiguous numpy copy; words32: even/odd strided copies)."""
        arr = self._fw.csr_arr
        if self.fast_store == "block":
            return np.array(arr[i0:i0 + n], dtype=np.uint32)
        out = np.empty(n, dtype=np.uint32)
        out[0::2] = arr[i0:i0 + n:2]
        out[1::2] = arr[i0 + 1:i0 + n:2]
        return out

    def read_logits(self, oc: int | None = None) -> np.ndarray:
        oc = self.oc if oc is None else oc
        if self._fast:        # all 16 LOGIT words (64 B at 0x080, 64-B aligned) in one read
            return self._csr_block(_I_LOGIT, N_LOGITS).view(np.int32)[:oc]
        read = self.csr.read
        return np.array([to_int32(read(OFF_LOGIT + 4 * i)) for i in range(oc)], dtype=np.int32)

    def read_counters(self, n_layers: int | None = None) -> dict:
        n = self.n_layers if n_layers is None else n_layers
        if self._fast:        # LAYER_CYC[0..7] (32 B at 0x040) in one read; counters ordered
            lc = [int(v) for v in self._csr_block(_I_LAYER_CYC, MAX_LAYERS)[:n]]
        else:
            lc = [self.csr.read(OFF_LAYER_CYC + 4 * l) for l in range(n)]
        return {"total_cyc": self.read64(OFF_TOTAL_CYC),
                "mac_active": self.read64(OFF_MAC_ACTIVE),
                "stall": self.read64(OFF_STALL),
                "layer_cyc": lc,
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
        if self._fast:
            self._fast_store_act0(u32, 0)
        else:
            self._write_u32("ACT0", u32, 0)

    def _fast_store_act0(self, u32: np.ndarray, off: int):
        """Fast path: the whole image into ACT0 as one vectorized numpy copy (block: contiguous
        copy, store width chosen by numpy/libc; words32: lo words then hi words, numpy strided
        4-byte loops = 32-bit stores). Validated at bring-up by check_fast_path()."""
        act0 = self._fw.act0
        n = u32.size
        if off < 0 or off + n > act0.size:
            raise GosError(f"ACT0: write of {n} words at {off} exceeds window")
        if self.fast_store == "block":
            act0[off:off + n] = u32
        else:
            act0[off:off + n:2] = u32[0::2]
            act0[off + 1:off + n:2] = u32[1::2]

    # -- fast-path bring-up check ------------------------------------------------------
    def check_fast_path(self, seed: int = 20260929, patterns: int = 3) -> dict:
        """Bring-up validation of the fast host path (board Session 1, step s1.fast; B3 fast).
        Only while idle. For `patterns` patterns over the WHOLE ACT0 window (seeded random,
        its complement, walking word index): fast vectorized write, then every word read back
        per word through the safe backend read() AND through a fast block read; any difference
        raises GosError. Then the CSR block reads (LOGIT[0..15], LAYER_CYC[0..7]) are compared
        with per-word safe reads of the same registers, and a scalar memoryview STATUS/VERSION
        read with the safe read. Leaves ACT0 holding the last pattern (the next infer rewrites
        the input). Returns counts for the log."""
        if not self._fast:
            raise GosError("check_fast_path needs host_path='fast'")
        self._require_idle()
        mm = self.be.mems["ACT0"]
        nw = MEMS["ACT0"][1] // 4
        rng = np.random.default_rng(seed)
        pats = [rng.integers(0, 2**32, nw, dtype=np.uint64).astype(np.uint32)]
        pats.append(~pats[0])
        pats.append((np.arange(nw, dtype=np.uint32) * np.uint32(0x9E3779B1)) ^ np.uint32(0xA5A5A5A5))
        words = 0
        for k, p in enumerate(pats[:max(1, patterns)]):
            self._fast_store_act0(p, 0)
            safe = np.fromiter((mm.read(4 * i) for i in range(nw)), dtype=np.uint32, count=nw)
            bad = np.flatnonzero(safe != p)
            if bad.size:
                i = int(bad[0])
                raise GosError(f"fast path ACT0 check: pattern {k}: {bad.size}/{nw} words differ "
                               f"(per-word readback); first word32 {i}: got 0x{int(safe[i]):08X} "
                               f"expected 0x{int(p[i]):08X} (fast_store={self.fast_store})")
            blk = self._act0_block_read(nw)
            bad = np.flatnonzero(blk != p)
            if bad.size:
                raise GosError(f"fast path ACT0 check: pattern {k}: fast block read differs in "
                               f"{bad.size}/{nw} words")
            words += nw
        for name, i0, n in (("LOGIT", _I_LOGIT, N_LOGITS), ("LAYER_CYC", _I_LAYER_CYC, MAX_LAYERS)):
            blk = self._csr_block(i0, n)
            ref = np.array([self.csr.read(4 * (i0 + j)) for j in range(n)], dtype=np.uint32)
            if not np.array_equal(blk, ref):
                raise GosError(f"fast path CSR check: {name} block read {blk.tolist()} != per-word "
                               f"{ref.tolist()}")
        for off in (OFF_VERSION, OFF_BUILD_ID, OFF_STATUS):
            a, b = self._fw.csr_rd(off >> 2), self.csr.read(off)
            if a != b:
                raise GosError(f"fast path CSR check: scalar read 0x{off:03X}: {a:#x} != {b:#x}")
        return {"act0_words_checked": words, "patterns": min(len(pats), max(1, patterns)),
                "csr_block_words_checked": N_LOGITS + MAX_LAYERS, "fast_store": self.fast_store,
                "windows": self._fw.note}

    def _act0_block_read(self, n: int) -> np.ndarray:
        act0 = self._fw.act0
        if self.fast_store == "block":
            return np.array(act0[:n], dtype=np.uint32)
        out = np.empty(n, dtype=np.uint32)
        out[0::2] = act0[0:n:2]
        out[1::2] = act0[1:n:2]
        return out

    # -- B1 control condition ------------------------------------------------------------
    def control_step(self, x, spin_ns: int, read_total: bool = True):
        """One host-loop iteration WITHOUT starting the accelerator (B1 control phase): input
        write into ACT0 (same host path as infer), the soft_reset write + STATUS clear poll that
        infer issues before every start, a STATUS poll spin of spin_ns (the median start->done
        time of the accelerator phase), the LOGIT read and (read_total) the TOTAL_CYC lo/hi
        read. CTRL.start is never written; STATUS must read 0 on every poll and TOTAL_CYC must
        be 0, otherwise GosError (the PL did work); a busy core is refused before anything is
        written (GosBusyError). Returns (logits, total_cyc, polls)."""
        self._require_idle()
        self.write_input(x)
        if self._fast:
            self._fw.csr_wr(_I_CTRL, CTRL_SOFT_RESET)
        else:
            self.csr.write(OFF_CTRL, CTRL_SOFT_RESET)
        self._wait_status_clear()
        status = self.status
        pc = time.perf_counter_ns
        t_end = pc() + int(spin_ns)
        polls = 0
        while True:
            st = status()
            polls += 1
            if st & 0x7:
                raise GosError(f"control phase: STATUS 0x{st:X} != 0 (the accelerator must not run)")
            if pc() >= t_end:
                break
        logits = self.read_logits()
        tc = None
        if read_total:
            tc = self.read64(OFF_TOTAL_CYC)
            if tc != 0:
                raise GosError(f"control phase: TOTAL_CYC {tc} != 0 (a job ran)")
        return logits, tc, polls

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
                        t_write_ns=tc - t0, t_clear_ns=t1 - tc, t_run_ns=t2 - t1, t_logit_ns=t3 - t2,
                        t_start_ns=self._last_t_start_ns)
        if read_counters:
            c = self.read_counters()
            r.total_cyc, r.mac_active, r.stall = c["total_cyc"], c["mac_active"], c["stall"]
            r.layer_cyc, r.violation = c["layer_cyc"], c["violation"]
            r.t_counter_ns = time.perf_counter_ns() - t3
        return r

    def set_host_path(self, host_path: str, fast_store: str | None = None):
        """Switch the host path at runtime (interleaved safe/fast measurements, exp_b3_breakdown
        --conditions). The fast windows are mapped on first use and kept."""
        if host_path not in HOST_PATHS:
            raise ValueError(f"host_path must be one of {HOST_PATHS}")
        if fast_store is not None:
            if fast_store not in FAST_STORES:
                raise ValueError(f"fast_store must be one of {FAST_STORES}")
            self.fast_store = fast_store
        if host_path == "fast" and self._fw is None:
            self._fw = self.be.fast_windows()
        self.host_path = host_path
        self._fast = host_path == "fast"

    def timed_job(self, x, clock_ns=None, timeout_s: float | None = None) -> tuple[int, int, int]:
        """PL clock calibration primitive (exp_fclk_cal.py). Untimed: input -> ACT0 and the
        pre-start clear. Timed with clock_ns (default CLOCK_MONOTONIC_RAW: not slewed by NTP):
        CTRL.start + STATUS poll until done. Then TOTAL_CYC (lo, hi). Returns
        (t_done - t_start in ns, TOTAL_CYC, polls)."""
        if self._desc is None:
            raise GosError("load_net first")
        if not self._desc_valid:
            self.write_descriptors(self._desc, self.n_layers)
        clk = clock_ns or (lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))
        self.write_input(x)
        self._clear_for_start()
        t0 = clk()
        polls, _ = self.start_and_wait(timeout_s, clear=False)
        t1 = clk()
        return t1 - t0, self.read64(OFF_TOTAL_CYC), polls

    def recover(self):
        """After a timeout / refused job: soft_reset and mark DESC for rewrite."""
        self.soft_reset()
        self._desc_valid = False
