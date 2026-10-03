#!/usr/bin/env python3
"""Exhaustive model of INT8 DSP48E2 packing (2 MACs per DSP sharing the activation), V3 Phase 0 WS5.

Scheme (AMD WP486 style, as in v3/feas/rtl/feas_row.sv PACK=1): ad = (wh << S) + wl in the 27-bit
pre-adder, p = ad * a (27 x 18 multiplier, a sign-extended INT8), then
    lo = signed(p[S-1:0])            -> a*wl
    hi = (p >>> S) + p[S-1]          -> a*wh    (borrow correction)
Checks, for S in {18, 19}:
  * pre-adder range: (wh << S) + wl fits 27-bit signed for all INT8 wh, wl;
  * product range: p fits the 45-bit signed multiplier output;
  * per-product extraction exact for ALL 2^24 (a, wl, wh) in INT8^3 (exhaustive);
  * in-DSP accumulation alternative: largest K for which accumulating K packed products in the 48-bit P
    register and extracting once is exact for every INT8 sequence (worst case |sum of a*wl| <= K*2^14 must fit
    the S-bit signed low field), confirmed by worst-case sequences at K_max (pass) and K_max+1 (fail).
Writes v3/results/feas_packing.csv (source = model).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from common import RESULTS_DIR, base_meta, write_results_csv  # noqa: E402

I8 = np.arange(-128, 128, dtype=np.int64)


def extract(p: np.ndarray, S: int) -> tuple[np.ndarray, np.ndarray]:
    low = p & ((1 << S) - 1)
    lo = np.where(low >= (1 << (S - 1)), low - (1 << S), low)
    hi = (p >> S) + ((low >> (S - 1)) & 1)
    return lo, hi


def exhaustive(S: int) -> dict:
    wl, wh = np.meshgrid(I8, I8, indexing="ij")
    wl, wh = wl.ravel(), wh.ravel()
    ad = (wh << S) + wl
    ad_ok = bool(((ad >= -(1 << 26)) & (ad < (1 << 26))).all())
    bad = 0
    pmax = 0
    for a in I8:
        p = ad * a
        pmax = max(pmax, int(np.abs(p).max()))
        lo, hi = extract(p, S)
        bad += int(((lo != a * wl) | (hi != a * wh)).sum())
    return dict(ad_fits_27b=ad_ok, p_fits_45b=pmax < (1 << 44), combos=256 ** 3, mismatches=bad)


def accumulate_exact(S: int, a: np.ndarray, wl: np.ndarray, wh: np.ndarray) -> bool:
    p = ((wh << S) + wl) * a                     # per-step packed products
    P = int(p.sum())                             # 48-bit P register (|P| << 2^47 here)
    assert abs(P) < (1 << 47)
    lo, hi = extract(np.array([P]), S)
    return int(lo[0]) == int((a * wl).sum()) and int(hi[0]) == int((a * wh).sum())


def kmax_in_dsp(S: int) -> tuple[int, bool, bool]:
    k = ((1 << (S - 1)) - 1) // (1 << 14)        # worst case |a*wl| = 2^14 ((-128)*(-128))
    worst = lambda K: (np.full(K, -128), np.full(K, -128), np.full(K, 127))
    ok_at = accumulate_exact(S, *worst(k))
    fail_next = not accumulate_exact(S, *worst(k + 1))
    rng = np.random.default_rng(0)
    for _ in range(2000):                        # random sequences at K_max must be exact too
        a, wl, wh = (rng.integers(-128, 128, k) for _ in range(3))
        ok_at &= accumulate_exact(S, a, wl, wh)
    return k, ok_at, fail_next


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_DIR / "feas_packing.csv"))
    a = ap.parse_args()
    rows = []
    for S in (18, 19):
        t0 = time.time()
        ex = exhaustive(S)
        k, ok_at, fail_next = kmax_in_dsp(S)
        r = base_meta(source="model", duration_s=round(time.time() - t0, 2))
        r.update(scheme=f"shared_activation_preadder_S{S}", shift_s=S, **ex,
                 per_product_exact=ex["mismatches"] == 0 and ex["ad_fits_27b"] and ex["p_fits_45b"],
                 kmax_in_dsp_accumulation=k, exact_at_kmax=ok_at, fails_at_kmax_plus_1=fail_next)
        rows.append(r)
        print(f"S={S}: mismatches {ex['mismatches']}/{ex['combos']}, ad27 {ex['ad_fits_27b']}, "
              f"kmax_in_dsp {k} (exact {ok_at}, fails at k+1 {fail_next})")
    write_results_csv(a.out, rows, list(rows[0].keys()))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
