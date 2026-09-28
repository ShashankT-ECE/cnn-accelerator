"""MockMMIO: laptop stand-in for the mapped PL windows, for HOST-OVERHEAD measurements only.

LABEL of every number produced with it: "host overhead, laptop mock (MockMMIO), not KV260"
(source=dryrun_model, written only under results/dryrun/). It measures the Python/numpy cost of
the driver's host code paths on the laptop CPU with ZERO bus latency; it says nothing about
AXI/PS latency on the KV260 and nothing about the accelerator.

  * MockMMIO(base, length): a pynq.MMIO look-alike: .array = numpy uint32 window, read(off) /
    write(off, v) modelled on pynq 3.x MMIO.read / MMIO.write_mm (argument checks, offset >> 2,
    one numpy element access). The SAFE host path uses read()/write() and .array element stores
    exactly as it does with pynq on the board.
  * The FAST host path gets FastWindows(csr_arr=.array, csr_rd/csr_wr, act0=.array): ACT0 is the
    plain numpy array (the same view type pynq MMIO.array is); CSR scalar accesses go through
    MockCsr.fast_rd / fast_wr, which apply the same register side effects as read()/write()
    (on the board they are memoryview accesses: C-level, cheaper than these Python methods, so
    the mock's fast-path CSR cost is pessimistic).
  * Register semantics (the subset the driver relies on, FORMATS.md CSR map): VERSION/BUILD_ID;
    N_LAYERS/DESC stored; CTRL.soft_reset clears STATUS/ERR_CODE/counters/LOGIT/violation;
    CTRL.start (ignored while busy): the job is looked up, STATUS reads busy for `busy_polls`
    STATUS reads, then done with LOGIT = the golden logits of the image found in ACT0 and the
    model counters (model_cycles.json). The image is identified by the exact ACT0 bytes of the
    loaded net (so a wrong input write is detected: unknown image -> STATUS.error, ERR_CODE
    0xFF00 "mock: input not in package"). QPARAM odd combined words read back {58'b0, s}.
    Not modelled: SLVERR, busy-lock, ACT busy violations, timing.
"""
from __future__ import annotations

import time

import numpy as np

import board_common as bc
import gos_driver as D

LABEL = "host overhead, laptop mock (MockMMIO), not KV260"
MOCK_ERR_UNKNOWN_INPUT = 0xFF << 8


class MockMMIO:
    """pynq.MMIO look-alike over a numpy uint32 array (pynq 3.x MMIO.read / write_mm logic)."""

    def __init__(self, base_addr: int, length: int, name: str = ""):
        self.base_addr, self.length, self.name = base_addr, length, name
        self.array = np.zeros(length >> 2, dtype=np.uint32)

    def read(self, offset: int = 0, length: int = 4, word_order: str = "little") -> int:
        if length not in (1, 2, 4, 8):
            raise ValueError("MMIO currently only supports 1, 2, 4 and 8-byte reads.")
        if offset < 0:
            raise ValueError("Offset cannot be negative.")
        if length == 8 and word_order not in ("big", "little"):
            raise ValueError("MMIO only supports big and little endian.")
        if offset % 4:
            raise MemoryError("Unaligned read: offset must be multiple of 4.")
        lsb = int(self.array[offset >> 2])
        return lsb & ((2 ** (8 * length)) - 1)

    def write(self, offset: int, data) -> None:
        if offset < 0:
            raise ValueError("Offset cannot be negative.")
        idx = offset >> 2
        if offset % 4:
            raise MemoryError("Unaligned write: offset must be multiple of 4.")
        if type(data) is int:
            self.array[idx] = np.uint32(data)
        else:
            raise ValueError("Data type must be int.")


class MockQparam(MockMMIO):
    def read(self, offset: int = 0, length: int = 4, word_order: str = "little") -> int:
        v = super().read(offset, length, word_order)
        if (offset >> 3) & 1:                 # odd combined word: {58'b0, s}
            v = v & 0x3F if offset % 8 == 0 else 0
        return v


class MockCsr(MockMMIO):
    def __init__(self, dev: "MockMmioBackend"):
        super().__init__(D.CSR_BASE, D.CSR_SIZE, "csr")
        self.dev = dev

    def read(self, offset: int = 0, length: int = 4, word_order: str = "little") -> int:
        if offset == D.OFF_STATUS:
            self.dev._tick()
        return super().read(offset, length, word_order)

    def write(self, offset: int, data) -> None:
        super().write(offset, data)
        if offset == D.OFF_CTRL:
            self.dev._ctrl(int(data))

    # fast-path scalar accessors (word index); same side effects as read()/write()
    def fast_rd(self, i: int) -> int:
        if i == 1:                            # STATUS
            self.dev._tick()
        return int(self.array[i])

    def fast_wr(self, i: int, v: int) -> None:
        self.array[i] = v
        if i == 0:                            # CTRL
            self.dev._ctrl(v)


class MockMmioBackend(D.MmioBackend):
    kind = "mock_mmio"
    source = bc.SOURCE_DRYRUN

    def __init__(self, data_dir=bc.DEFAULT_DATA_DIR, clock_mhz: float = 200.0, busy_polls: int = 1,
                 nets=bc.NETS, build_id: int = 0):
        csr = MockCsr(self)
        mems = {n: (MockQparam if n == "QPARAM" else MockMMIO)(b, s, n)
                for n, (b, s) in D.MEMS.items()}
        super().__init__(csr, mems)
        self._clock = float(clock_mhz)
        self.busy_polls = int(busy_polls)
        a = csr.array
        a[D.OFF_VERSION >> 2] = D.VERSION_CORE
        a[D.OFF_BUILD_ID >> 2] = build_id
        self._nets = []
        for net in nets:
            pkg = bc.load_package(data_dir, net)
            x = np.ascontiguousarray(pkg.x_act)
            lut = {x[i].tobytes(): i for i in range(x.shape[0])}
            mc = pkg.model_cycles
            self._nets.append({"net": net, "desc": np.asarray(pkg.desc, dtype=np.uint32),
                               "desc_bytes": np.asarray(pkg.desc, dtype=np.uint32).tobytes(),
                               "nbytes": x.shape[1], "lut": lut,
                               "golden": np.asarray(pkg.golden_logits, dtype=np.int32),
                               "layer_cyc": [int(L["cycles"]) for L in mc["layers"]],
                               "total": int(mc["total"]["cycles"]),
                               "mac": int(mc["total"]["mac_active"])})
        self._busy_left = 0
        self._pending = None
        self.jobs = 0

    # -- register side effects -------------------------------------------------------------
    _CLR = slice(D.OFF_ERR_CODE >> 2, (D.OFF_LOGIT >> 2) + D.N_LOGITS)   # ERR_CODE..LOGIT[15]

    def _clear_results(self):
        self.csr.array[self._CLR] = 0          # (0x030-0x03C are unmapped words: read 0 anyway)

    def _ctrl(self, v: int):
        a = self.csr.array
        a[0] = 0                                        # CTRL reads 0
        if v & D.CTRL_SOFT_RESET:
            a[1] = 0
            self._clear_results()
            self._pending, self._busy_left = None, 0
        if v & D.CTRL_START and not a[1] & D.ST_BUSY:
            a[1] = 0
            self._clear_results()
            self.jobs += 1
            self._pending = self._job()
            if self._pending is None:
                a[D.OFF_ERR_CODE >> 2] = MOCK_ERR_UNKNOWN_INPUT
                a[1] = D.ST_ERROR
            else:
                a[1] = D.ST_BUSY
                self._busy_left = self.busy_polls

    def _job(self):
        a = self.csr.array
        n = int(a[D.OFF_N_LAYERS >> 2])
        d0 = D.OFF_DESC >> 2
        for e in self._nets:
            L = e["desc"].shape[0]
            if n != L:
                continue
            if a[d0:d0 + 16 * L].tobytes() != e["desc_bytes"]:
                continue
            idx = e["lut"].get(self.mems["ACT0"].array.view(np.uint8)[:e["nbytes"]].tobytes())
            return None if idx is None else (e, idx)
        return None

    def _tick(self):
        a = self.csr.array
        if not a[1] & D.ST_BUSY:
            return
        if self._busy_left > 0:
            self._busy_left -= 1
            return
        e, idx = self._pending
        a[D.OFF_LOGIT >> 2:(D.OFF_LOGIT >> 2) + e["golden"].shape[1]] = e["golden"][idx].view(np.uint32)
        lc = D.OFF_LAYER_CYC >> 2
        a[lc:lc + len(e["layer_cyc"])] = e["layer_cyc"]
        a[D.OFF_TOTAL_CYC >> 2] = e["total"] & 0xFFFF_FFFF
        a[(D.OFF_TOTAL_CYC >> 2) + 1] = e["total"] >> 32
        a[D.OFF_MAC_ACTIVE >> 2] = e["mac"] & 0xFFFF_FFFF
        a[(D.OFF_MAC_ACTIVE >> 2) + 1] = e["mac"] >> 32
        a[1] = D.ST_DONE
        self._pending = None

    def emulation_cost_ns(self, reps: int = 2000) -> dict:
        """Laptop cost of the mock's own register side effects for one job (start: job lookup;
        busy STATUS ticks; done: LOGIT/counters written; soft_reset clear), timed WITHOUT the
        driver. This cost is inside start_write / poll / status_clear on BOTH host paths."""
        pc = time.perf_counter_ns
        out = np.zeros(reps, np.int64)
        for k in range(reps):
            t0 = pc()
            self._ctrl(D.CTRL_SOFT_RESET)
            self._ctrl(D.CTRL_START)
            for _ in range(self.busy_polls + 1):
                self._tick()
            out[k] = pc() - t0
        self._ctrl(D.CTRL_SOFT_RESET)
        return {"median_ns": float(np.median(out)), "p95_ns": float(np.percentile(out, 95))}

    # -- backend API -----------------------------------------------------------------------
    def fclk0_mhz(self) -> float:
        return self._clock

    def set_fclk0(self, mhz: float) -> float:
        self._clock = float(mhz)
        return self._clock

    def fast_windows(self) -> D.FastWindows:
        return D.FastWindows(self.csr.array, self.csr.fast_rd, self.csr.fast_wr,
                             self.mems["ACT0"].array, note="MockMMIO numpy arrays (laptop)")

    def info(self) -> dict:
        return {"backend": self.kind, "label": LABEL, "busy_polls": self.busy_polls}
