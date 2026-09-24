#!/usr/bin/env python3
"""Cycle-accurate simulation of the frozen 8x8 array for FC (1-pixel) layers.

Models the exact pe_v2 + systolic_array_v2 register semantics (nonblocking:
product samples the OLD weight; accumulator samples the OLD product; weight
updates via BREG) and drives the OS and WS schedules for a 1x1-conv (FC)
workload. Verifies the integer accumulator against the frozen L2 golden
(fc1/fc2 weights + bias + input) and reports the exact cycle schedule, resolving
Step 3.5 discrepancy A.

Run:  .venv/bin/python python/fc_schedule_sim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))


class Array:
    """Register-exact model of systolic_array_v2 (8x8 of pe_v2)."""

    def __init__(self, mode: int):
        self.mode = mode
        self.weight = np.zeros((8, 8), dtype=np.int64)   # BREG
        self.product = np.zeros((8, 8), dtype=np.int64)  # MREG
        self.accum = np.zeros((8, 8), dtype=np.int64)    # PREG
        self.shift = np.zeros((8, 7), dtype=np.int64)    # inter-column shift regs

    def step(self, act_in, w_in, weight_load, accum_clear):
        act_in = np.asarray(act_in, dtype=np.int64)
        w_in = np.asarray(w_in, dtype=np.int64)
        # act_delayed (combinational): col 0 = act_in, col c = shift_reg[c-1]
        act_delayed = np.zeros((8, 8), dtype=np.int64)
        act_delayed[:, 0] = act_in
        act_delayed[:, 1:] = self.shift
        # psum_in (combinational): row 0 <- row 7, row r <- row r-1
        psum_in = np.empty_like(self.accum)
        psum_in[0] = self.accum[7]
        psum_in[1:] = self.accum[:-1]
        # nonblocking updates: product uses OLD weight; accum uses OLD product/accum
        new_product = np.where(accum_clear, 0, act_delayed * self.weight)
        if accum_clear:
            new_accum = np.zeros((8, 8), dtype=np.int64)
        elif self.mode:
            new_accum = psum_in + self.product
        else:
            new_accum = self.accum + self.product
        self.product = new_product
        self.accum = new_accum
        if weight_load:
            self.weight = np.broadcast_to(w_in[:, None], (8, 8)).copy()
        # shift chain
        self.shift[:, 0] = act_in
        self.shift[:, 1:] = self.shift[:, :-1]


def run_os_fc(IC: int, OC: int, W: np.ndarray, x: np.ndarray) -> tuple[int, np.ndarray]:
    """OS FC: rows = 8 output neurons (col 0), IC features serialized in time."""
    arr = Array(mode=0)
    out = np.zeros(OC, dtype=np.int64)
    cycles = 0
    for g in range((OC + 7) // 8):
        oc0 = g * 8
        n_rows = min(8, OC - oc0)
        # cycle 0: accum_clear + load W[oc][0]
        arr.step(np.zeros(8), [W[oc0 + r, 0] if r < n_rows else 0 for r in range(8)],
                 weight_load=True, accum_clear=True)
        cycles += 1
        # cycles 1..IC: feed x[ic], load W[oc][ic+1] (pipelined)
        for ic in range(IC):
            w = [W[oc0 + r, ic + 1] if (ic + 1 < IC and r < n_rows) else 0 for r in range(8)]
            arr.step(np.full(8, x[ic]), w, weight_load=True, accum_clear=False)
            cycles += 1
        # tail: last product -> accumulator (1 cycle)
        arr.step(np.zeros(8), np.zeros(8), weight_load=False, accum_clear=False)
        cycles += 1
        # drain: 2 cycles per row (result_req held; RTL emits 8 rows x 2)
        for r in range(n_rows):
            out[oc0 + r] = arr.accum[r, 0]
            cycles += 2
    return cycles, out


def run_ws_fc(IC: int, OC: int, W: np.ndarray, x: np.ndarray) -> tuple[int, np.ndarray]:
    """WS FC: rows = 8 input-feature taps (tiled), 1 pixel, OC serialized (1/sweep)."""
    ntiles = (IC + 7) // 8
    out = np.zeros(OC, dtype=np.int64)
    cycles = 0
    for oc in range(OC):
        arr = Array(mode=1)
        # cycle 0: accum_clear + load tile 0 weights
        arr.step(np.zeros(8), [W[oc, r] if r < IC else 0 for r in range(8)],
                 weight_load=True, accum_clear=True)
        cycles += 1
        for t in range(ntiles):
            # diagonal skew: row r drives feature (8t+r) at its skewed cycle
            for r in range(8):
                act = np.zeros(8, dtype=np.int64)
                k = t * 8 + r
                if k < IC:
                    act[r] = x[k]
                wl = (t + 1 < ntiles and r == 7)   # next tile weights, concurrent
                w = np.zeros(8, dtype=np.int64)
                if wl:
                    for rr in range(8):
                        kk = (t + 1) * 8 + rr
                        if kk < IC:
                            w[rr] = W[oc, kk]
                arr.step(act, w, weight_load=wl, accum_clear=False)
                cycles += 1
        # drain the 7-row psum cascade tail after the last tap
        for _ in range(7):
            arr.step(np.zeros(8), np.zeros(8), weight_load=False, accum_clear=False)
            cycles += 1
        out[oc] = arr.accum[7, 0]
        cycles += 1  # result latch
    return cycles, out


def main() -> int:
    qp = np.load(REPO_ROOT / "data" / "lenet5_int8" / "quant_params.npz")
    lv = np.load(REPO_ROOT / "data" / "lenet5_int8" / "layer_vectors.npz")

    for name, wkey, bkey, inkey, acckey in (
        ("fc1", "fc1_q_w", "fc1_q_b", "fc1_in", "fc1_acc"),
        ("fc2", "fc2_q_w", "fc2_q_b", "fc2_in", "fc2_acc"),
    ):
        W = qp[wkey].astype(np.int64)
        qb = qp[bkey].astype(np.int64)
        x = lv[inkey][0].astype(np.int64)
        golden = lv[acckey][0].astype(np.int64)
        IC, OC = W.shape[1], W.shape[0]

        os_cycles, os_out = run_os_fc(IC, OC, W, x)
        ws_cycles, ws_out = run_ws_fc(IC, OC, W, x)
        target = golden                             # fc1_acc/fc2_acc IS the pure MAC (bias added in requantize)

        os_ok = np.array_equal(os_out, target)
        ws_ok = np.array_equal(ws_out, target)
        print(f"{name} (IC={IC}, OC={OC}):")
        print(f"  OS: {os_cycles:5d} cycles  bit-exact={os_ok}")
        print(f"  WS: {ws_cycles:5d} cycles  bit-exact={ws_ok}")
        if not (os_ok and ws_ok):
            print(f"  OS   ={os_out[:6]}")
            print(f"  WS   ={ws_out[:6]}")
            print(f"  GOLD ={target[:6]}")
            sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
