"""Simulated gos_ register map + memories (pure numpy), for dry runs and unit tests.

Models the PS-visible semantics of FORMATS.md "CSR register map" and DECISIONS D11/D12 that the
driver depends on; it is NOT a model of the core timing. The job itself is delegated to a
``job_fn(sim) -> JobOutcome`` callback (``gos_model_backend`` supplies one built on
gos_golden + gos_cycle_model; the tests supply stubs).

Semantics modelled:
  * 32-bit word-aligned accesses; unmapped read/write, write to an RO register, and N_LAYERS/DESC
    writes while busy raise SimSlvErr (SLVERR on the hardware).
  * CTRL bit0 start (ignored while busy), bit1 soft_reset (clears status, ERR_CODE, counters,
    LOGIT and violation flags; keeps memories and DESC/N_LAYERS). CTRL reads 0.
  * STATUS {error, done, busy}; done/error sticky until the next start or soft_reset. At start:
    done/error/ERR_CODE/counters/LOGIT cleared. After an accepted start the job reports busy for
    ``busy_polls`` STATUS reads (``hang=True``: forever), then done.
  * 64-bit counters: a lo read latches the hi word into a per-counter shadow; a hi read returns
    the shadow (0 after reset). ``live_counter_hook`` lets a test change a counter between reads.
  * Memories: 64-bit words as two 32-bit MMIO words (little-endian); QPARAM odd combined words
    read back {58'b0, s}; ACT PS access while busy: writes ignored / reads 0 + sticky violation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import gos_driver as D


class SimSlvErr(RuntimeError):
    """An access the real CSR answers with SLVERR (on the PS this can abort the process)."""


@dataclass
class JobOutcome:
    err_code: int = 0                       # nonzero: refused (STATUS.error)
    logits: list = field(default_factory=list)   # int32 values for LOGIT[0..]
    total_cyc: int = 0
    mac_active: int = 0
    stall: int = 0
    layer_cyc: list = field(default_factory=list)


class SimMem:
    def __init__(self, dev: "SimDevice", name: str):
        self.dev, self.name = dev, name
        self.array = np.zeros(D.MEMS[name][1] // 4, dtype=np.uint32)   # raw store (PS view)

    def _check(self, off):
        if off % 4 or not 0 <= off < self.array.size * 4:
            raise SimSlvErr(f"{self.name}: bad offset 0x{off:X}")

    def read(self, off: int) -> int:
        self._check(off)
        if self.name.startswith("ACT") and self.dev.busy:
            self.dev.violation |= 1 << int(self.name[-1])
            return 0
        v = int(self.array[off // 4])
        if self.name == "QPARAM" and (off // 8) % 2 == 1:     # odd combined word: {58'b0, s}
            v = v & 0x3F if off % 8 == 0 else 0
        return v

    def write(self, off: int, val: int):
        self._check(off)
        if self.name.startswith("ACT") and self.dev.busy:
            self.dev.violation |= 1 << int(self.name[-1])
            return
        self.array[off // 4] = int(val) & 0xFFFF_FFFF
        self.dev.mem_gen += 1

    def words(self) -> np.ndarray:
        """64-bit PS-view words as the core sees them (QPARAM odd words masked)."""
        w = self.array.view(np.uint64).copy()
        if self.name == "QPARAM":
            w[1::2] &= np.uint64(0x3F)
        return w


class SimCsr:
    RO = {D.OFF_STATUS, D.OFF_ERR_CODE, D.OFF_VIOLATION, D.OFF_VERSION, D.OFF_BUILD_ID}

    def __init__(self, dev: "SimDevice"):
        self.dev = dev

    def read(self, off: int) -> int:
        d = self.dev
        d.reads += 1
        if off % 4:
            raise SimSlvErr(f"CSR unaligned read 0x{off:X}")
        if off == D.OFF_CTRL:
            return 0
        if off == D.OFF_STATUS:
            return d._status_read()
        if off == D.OFF_N_LAYERS:
            return d.n_layers
        if off == D.OFF_ERR_CODE:
            return d.err_code
        for name, lo in (("total_cyc", D.OFF_TOTAL_CYC), ("mac_active", D.OFF_MAC_ACTIVE),
                         ("stall", D.OFF_STALL)):
            if off in (lo, lo + 4):
                if d.live_counter_hook:
                    d.live_counter_hook(d, name, off == lo)
                v = getattr(d, name)
                if off == lo:
                    d.shadow[name] = (v >> 32) & 0xFFFF_FFFF
                    return v & 0xFFFF_FFFF
                return d.shadow[name]
        if off == D.OFF_VIOLATION:
            return d.violation
        if D.OFF_LAYER_CYC <= off < D.OFF_LAYER_CYC + 32:
            return d.layer_cyc[(off - D.OFF_LAYER_CYC) // 4]
        if D.OFF_LOGIT <= off < D.OFF_LOGIT + 64:
            return d.logit[(off - D.OFF_LOGIT) // 4]
        if off == D.OFF_VERSION:
            return d.version
        if off == D.OFF_BUILD_ID:
            return d.build_id
        if D.OFF_DESC <= off < D.OFF_DESC + 0x40 * D.MAX_LAYERS:
            r = off - D.OFF_DESC
            return int(d.desc[r // 0x40, (r % 0x40) // 4])
        raise SimSlvErr(f"CSR unmapped read 0x{off:03X}")

    def write(self, off: int, val: int):
        d = self.dev
        d.writes += 1
        val = int(val) & 0xFFFF_FFFF
        if off % 4:
            raise SimSlvErr(f"CSR unaligned write 0x{off:X}")
        if off == D.OFF_CTRL:
            if val & D.CTRL_SOFT_RESET:
                d._soft_reset()
            if val & D.CTRL_START:
                d._start()
            return
        if off == D.OFF_N_LAYERS:
            if d.busy:
                raise SimSlvErr("N_LAYERS write while busy")
            d.n_layers = val & 0xF
            return
        if D.OFF_DESC <= off < D.OFF_DESC + 0x40 * D.MAX_LAYERS:
            if d.busy:
                raise SimSlvErr("DESC write while busy")
            r = off - D.OFF_DESC
            d.desc[r // 0x40, (r % 0x40) // 4] = val
            return
        kind = "RO" if (off in self.RO or D.OFF_TOTAL_CYC <= off < D.OFF_STALL + 8
                        or D.OFF_LAYER_CYC <= off < D.OFF_LOGIT + 64) else "unmapped"
        raise SimSlvErr(f"CSR write to {kind} offset 0x{off:03X}")


class SimCsrArray:
    """Fast-host-path view of SimCsr: integer word indices and slices, every element access
    routed through SimCsr.read/write (so STATUS/counter/SLVERR semantics are kept). Used by
    ModelBackend.fast_windows(); the laptop cost of this adapter is NOT representative of a
    mapped window (see mock_mmio.py for the host-overhead measurement)."""

    def __init__(self, csr: SimCsr, words: int = 1024):
        self.csr, self.size = csr, words

    def rd(self, i: int) -> int:
        return self.csr.read(4 * i)

    def wr(self, i: int, v: int):
        self.csr.write(4 * i, v)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return np.array([self.csr.read(4 * i) for i in range(*k.indices(self.size))],
                            dtype=np.uint32)
        return self.csr.read(4 * int(k))

    def __setitem__(self, k, v):
        if isinstance(k, slice):
            for i, x in zip(range(*k.indices(self.size)), np.asarray(v).reshape(-1).tolist()):
                self.csr.write(4 * i, int(x))
        else:
            self.csr.write(4 * int(k), int(v))


class SimDevice:
    def __init__(self, job_fn, version: int = D.VERSION_CORE, build_id: int = 0,
                 busy_polls: int = 1):
        self.job_fn = job_fn
        self.version, self.build_id = version, build_id
        self.busy_polls = busy_polls
        self.hang = False
        self.live_counter_hook = None
        self.csr = SimCsr(self)
        self.mems = {n: SimMem(self, n) for n in D.MEMS}
        self.desc = np.zeros((D.MAX_LAYERS, D.DESC_WORDS), dtype=np.uint32)
        self.n_layers = 0
        self.mem_gen = 0
        self.reads = self.writes = 0
        self.jobs = 0
        self._reset_state()

    def _reset_state(self):
        self.busy = False
        self.done = self.error = False
        self.err_code = 0
        self.total_cyc = self.mac_active = self.stall = 0
        self.layer_cyc = [0] * D.MAX_LAYERS
        self.logit = [0] * D.N_LOGITS
        self.violation = 0
        self.shadow = {"total_cyc": 0, "mac_active": 0, "stall": 0}
        self._busy_left = 0
        self._pending = None

    def _soft_reset(self):
        self._reset_state()

    def _start(self):
        if self.busy:
            return
        self.done = self.error = False
        self.err_code = 0
        self.total_cyc = self.mac_active = self.stall = 0
        self.layer_cyc = [0] * D.MAX_LAYERS
        self.logit = [0] * D.N_LOGITS
        self.jobs += 1
        out = self.job_fn(self)
        if out.err_code:
            self.error, self.err_code = True, out.err_code
            self.total_cyc = out.total_cyc
            return
        self.busy = True
        self._busy_left = self.busy_polls
        self._pending = out

    def _status_read(self) -> int:
        if self.busy and not self.hang:
            if self._busy_left <= 0:
                self._finish()
            else:
                self._busy_left -= 1
        return (int(self.error) << 2) | (int(self.done) << 1) | int(self.busy)

    def _finish(self):
        out = self._pending
        self.busy, self.done = False, True
        self.total_cyc, self.mac_active, self.stall = out.total_cyc, out.mac_active, out.stall
        for l, c in enumerate(out.layer_cyc):
            self.layer_cyc[l] = int(c) & 0xFFFF_FFFF
        for i, v in enumerate(out.logits):
            self.logit[i] = int(v) & 0xFFFF_FFFF
        self._pending = None
