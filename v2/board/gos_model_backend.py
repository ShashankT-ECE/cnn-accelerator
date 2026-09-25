"""ModelBackend: laptop dry run of the board flow (label: dryrun_model — NEVER hardware data).

The driver (gos_driver.GosDevice) talks to a simulated register map (gos_sim.SimDevice) exactly
as it talks to the KV260. On CTRL.start the job is computed from what the driver actually wrote:

  * config check: gos_pack.job_err_code(N_LAYERS, DESC)                  (RTL checker model)
  * refused job: STATUS.error, ERR_CODE, TOTAL_CYC = C_START (busy during S_CHK1..3)
  * accepted job: every layer from the memory images — input map read from ACT[in_sel] with
    gos_pack.unpack_act, weights/QPARAM with gos_pack.unpack_wgt / unpack_qparam at the
    descriptor's WGT_BASE / QP_BASE, layer computed by gos_golden.gos_layer (bit-exact golden),
    output written to ACT[!in_sel] (only the bytes the map covers) or LOGIT (out_raw)
  * counters: gos_cycle_model (LAYER_CYC = T*K + C_PIPE, TOTAL_CYC = sum + C_START + C_DONE,
    MAC_ACTIVE = sum T*K, STALL = 0)

So a dry run checks the data package, the packing, the driver's register sequence and the
scripts end to end. Needs v2/model (laptop, repo .venv); never used on the board.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import gos_driver as D
from gos_sim import JobOutcome, SimDevice

_BOARD = Path(__file__).resolve().parent
for _p in (_BOARD.parent / "model", _BOARD / "model"):
    if (_p / "gos_golden.py").is_file() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        break

import gos_cycle_model as cm  # noqa: E402
import gos_golden as gg  # noqa: E402
import gos_pack as gp  # noqa: E402


def _layer_cfg(f: dict, i: int) -> dict:
    return {"name": f"L{i}", "IC": f["IC"], "OC": f["OC"], "KH": f["KH"], "KW": f["KW"],
            "IH": f["IH"], "IW": f["IW"], "OH": f["OH"], "OW": f["OW"], "K": f["K"],
            "relu": bool(f["relu_en"]), "pool": bool(f["pool_en"]), "final": bool(f["out_raw"])}


class _ModelJob:
    def __init__(self):
        self._key = None
        self._params = None

    def _layer_params(self, sim: SimDevice, fields: list[dict]):
        wgt = sim.mems["WGT"].words()
        qp = sim.mems["QPARAM"].words()
        key = (wgt.tobytes(), qp.tobytes(), tuple(tuple(sorted(f.items())) for f in fields))
        if self._key == key:
            return self._params
        params = []
        for i, f in enumerate(fields):
            L = _layer_cfg(f, i)
            q_w = gp.unpack_wgt(wgt, L, f["WGT_BASE"])
            q_b, m, s = gp.unpack_qparam(qp, f["QP_BASE"], f["OC"])
            params.append(gg.LayerParams(L, q_w, q_b, m=tuple(int(x) for x in m),
                                         s=tuple(int(x) for x in s)))
        self._key, self._params = key, params
        return params

    def __call__(self, sim: SimDevice) -> JobOutcome:
        n = sim.n_layers
        code = gp.job_err_code(n, sim.desc)
        if code:
            return JobOutcome(err_code=code, total_cyc=cm.C_START)
        fields = [gp.decode_descriptor(sim.desc[l]) for l in range(n)]
        params = self._layer_params(sim, fields)
        layer_cyc, mac = [], 0
        logits = []
        for i, (f, P) in enumerate(zip(fields, params)):
            src = sim.mems[f"ACT{f['in_sel']}"]
            x = gp.unpack_act(src.words(), f["IC"], f["IH"], f["IW"])
            y = gg.gos_layer(x, P.cfg, P)
            if f["out_raw"]:
                logits = [int(v) for v in y.reshape(-1)]
            else:
                dst = sim.mems[f"ACT{1 - f['in_sel']}"]
                img = gp.pack_act(y)                                   # uint8 [8, 4096]
                cover = gp.pack_act(np.ones(y.shape, dtype=np.int8)).astype(bool)
                banks = gp.act_words_to_banks(dst.array.view(np.uint64))
                banks[cover] = img[cover]
                dst.array[:] = gp.act_banks_to_words(banks).view(np.uint32)
            c = cm.layer_cycles(P.cfg)
            layer_cyc.append(c["cycles"])
            mac += c["mac_active"]
        total = sum(layer_cyc) + cm.C_START + cm.C_DONE
        return JobOutcome(logits=logits, total_cyc=total, mac_active=mac, stall=0,
                          layer_cyc=layer_cyc)


class ModelBackend(D.MmioBackend):
    kind = "model"
    source = "dryrun_model"

    def __init__(self, clock_mhz: float = 200.0, busy_polls: int = 1, build_id: int = 0):
        self.sim = SimDevice(_ModelJob(), build_id=build_id, busy_polls=busy_polls)
        self._clock = float(clock_mhz)
        super().__init__(self.sim.csr, self.sim.mems)

    def fclk0_mhz(self) -> float:
        return self._clock          # nominal value given by the caller (no clock to read back)

    def set_fclk0(self, mhz: float) -> float:
        self._clock = float(mhz)
        return self._clock

    def info(self) -> dict:
        return {"backend": self.kind, "note": "gos_golden + gos_cycle_model behind gos_sim"}
