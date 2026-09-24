"""V2 step 2.1 part B: exhaustive equivalence of the hardware integer requant
against the frozen float64 reference (ARCH_SPEC "Numeric contract", DECISIONS D1).

Hardware contract (per output channel, unsigned B-bit multiplier m, shift s):

    v = acc + q_bias
    p = v * m                                           (exact)
    q = (p + 2^(s-1) - 1 + ((p >> s) & 1)) >> s          (arithmetic shift, RNE)
    clip [-128, 127]   (ReLU after clip commutes, so the check is pre-ReLU)

Reference: legacy ``lenet5.int8_quant.requantize(y, M)`` =
clip(np.rint(float64(y) * float64(M)), -128, 127).astype(int8), called unchanged.
The legacy model calls ``requantize(acc + q_b, M broadcast)``; the operation is
elementwise float64, so calling it per channel with the scalar M_c on an int64
array of v values gives bit-identical results (every |v| < 2^53 converts to
float64 exactly, same as the int32 v of the model).

For each B in {32, 40, 48} and each channel, s is chosen so that
m_rne = RNE(Fraction(M) * 2^s) (exact rational arithmetic on the float64 M) lies
in [2^(B-1), 2^B). Reachable v: |acc| <= K*16384 -> v in [-K*16384+qb, K*16384+qb].

m selection (DECISIONS.md OC-1): m_rne alone is not bit-exact on a few near-tie
channels, so candidates m_rne + d are tried in the order d = 0, +1, -1, +2, -2,
... up to |d| <= MAX_DELTA (staying in [2^(B-1), 2^B), s fixed). The first
candidate with 0 mismatches over the full check below is selected. The m_rne
result is always recorded (mismatches_rne) as evidence.

* Exact region: every integer v with |v*M| <= 130 (exact rational bound)
  within the reachable range, compared element-wise.
* Saturated region (|v*M| > 130): by monotonicity of hw q in v (m > 0) it is
  sufficient that hw q >= 127 at the first positive saturated v and <= -128 at
  the first negative one (the float64 reference is 127/-128 there because
  float rounding is monotone and 130 is representable). Additionally
  1,000,000 seeded random saturated samples per channel are compared.

Arithmetic: plain int64 only where max|v| * m < 2^62 (checked with Python
ints); otherwise an exact 2-limb int64 method (m = mh*2^k + ml, k = 32) that
evaluates the same formula without overflow (validated against Python ints).

Usage:
    python requant_check.py --nets lenet5 cifar10 [--lenet5-npz P] [--cifar10-npz P]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

import common  # noqa: F401  (puts legacy python/ on sys.path)
from common import FROZEN_DIR, REPO_ROOT, RESULTS_DIR, base_meta, write_results_csv
from lenet5.int8_quant import requantize  # legacy float64 reference (read-only)

BITS = (32, 40, 48)
ACT_PROD_MAX = 128 * 128          # |a*w| <= 16384 for INT8 x INT8
EXACT_BOUND = 130                 # |v*M| <= 130 -> exact element-wise region
N_SAT_SAMPLES = 1_000_000
SAT_SEED = 20260924
LIMB_K = 32
MAX_DELTA = 16                    # feasible-m search window around m_rne (LSB)
TWO62 = 1 << 62

# (layer, expected K) for the requantized layers; the final layer is excluded.
NET_LAYERS = {
    "lenet5": (("conv1", 25), ("conv3", 150), ("conv5", 400), ("fc1", 120)),
    "cifar10": (("conv1", 75), ("conv2", 800), ("conv3", 800)),
}
DEFAULT_NPZ = {
    "lenet5": REPO_ROOT / "data" / "lenet5_int8" / "quant_params.npz",
    "cifar10": FROZEN_DIR / "cifar10_int8" / "quant_params.npz",
}
HW_NPZ = {
    "lenet5": FROZEN_DIR / "lenet5_int8" / "hw_requant.npz",
    "cifar10": FROZEN_DIR / "cifar10_int8" / "hw_requant.npz",
}
CSV_PATH = RESULTS_DIR / "requant_equivalence.csv"
CSV_FIELDS = ["net", "layer", "channel", "B", "s", "m", "values_checked_exact",
              "values_checked_saturated", "mismatches", "first_mismatch_v",
              "m_rne", "delta_from_rne", "mismatches_rne", "candidates_tried",
              "sat_seed", "ref_float_vs_exact_diffs"]


# --------------------------------------------------------------------------- #
# Rounding shift primitives
# --------------------------------------------------------------------------- #
def rne_shift(p, s):
    """Hardware RNE shift on int64: (p + 2^(s-1) - 1 + ((p >> s) & 1)) >> s.

    Valid (no int64 overflow) for |p| < 2^62 and 1 <= s <= 62.
    """
    p = np.asarray(p, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    one = np.int64(1)
    return (p + ((one << (s - one)) - one) + ((p >> s) & one)) >> s


def rne_shift_int(p: int, s: int) -> int:
    """Same formula on unbounded Python ints (>> is floor / arithmetic)."""
    return (p + (1 << (s - 1)) - 1 + ((p >> s) & 1)) >> s


def hw_q_narrow(v, m: int, s: int):
    """Hardware q (pre-clip) with p = v*m in int64. Caller guarantees |v*m| < 2^62."""
    v = np.asarray(v, dtype=np.int64)
    return rne_shift(v * np.int64(m), s)


def hw_q_wide(v, m: int, s: int, k: int = LIMB_K):
    """Hardware q (pre-clip), exact for wide p = v*m, using int64 limbs.

    m = mh*2^k + ml; p = v*mh*2^k + v*ml = c*2^k + a_lo with
    c = v*mh + floor(v*ml / 2^k), a_lo = (v*ml) mod 2^k.  Then with
    t = 2^(s-1) - 1 + f0, f0 = floor(p/2^s) & 1 = (c >> (s-k)) & 1 (k <= s):
    (p + t) >> s = (c + ((a_lo + t) >> k)) >> (s - k).
    """
    v = np.asarray(v, dtype=np.int64)
    k = min(k, s)
    sh = s - k
    mh, ml = m >> k, m & ((1 << k) - 1)
    vmax = int(np.abs(v).max()) if v.size else 0
    assert vmax * mh < TWO62 and vmax * ml < TWO62, "limb product overflow"
    a = v * np.int64(ml)
    a_hi = a >> np.int64(k)
    a_lo = a & np.int64((1 << k) - 1)
    c = v * np.int64(mh) + a_hi
    assert vmax * mh + (vmax * ml >> k) + 1 < TWO62
    f0 = (c >> np.int64(sh)) & np.int64(1)
    t = np.int64((1 << (s - 1)) - 1) + f0            # <= 2^62, fits
    carry = (a_lo + t) >> np.int64(k)                # a_lo + t < 2^62 + 2^32
    return (c + carry) >> np.int64(sh)


def hw_q(v, m: int, s: int):
    """Dispatch: int64 only where provably |v*m| < 2^62, else the limb path."""
    v = np.asarray(v, dtype=np.int64)
    vmax = int(np.abs(v).max()) if v.size else 0
    if vmax * m < TWO62 and s <= 62:
        return hw_q_narrow(v, m, s)
    return hw_q_wide(v, m, s)


def hw_requant(v, m: int, s: int):
    """Hardware requant output (post-clip, pre-ReLU) as int8."""
    return np.clip(hw_q(v, m, s), -128, 127).astype(np.int8)


# --------------------------------------------------------------------------- #
# (m, s) selection
# --------------------------------------------------------------------------- #
def select_m_s(M: float, B: int) -> tuple[int, int]:
    """Smallest-error s with m = RNE(Fraction(M)*2^s) in [2^(B-1), 2^B)."""
    Mf = Fraction(float(M))
    assert Mf > 0, "M must be positive"
    _, E = math.frexp(float(M))                      # M = f*2^E, f in [0.5, 1)
    s = B - E                                        # M*2^s = f*2^B in [2^(B-1), 2^B)
    m = round(Mf * (1 << s)) if s >= 0 else round(Mf / (1 << -s))
    if m == 1 << B:                                  # rounding pushed m to 2^B
        s -= 1
        m = round(Mf * (1 << s))
    assert (1 << (B - 1)) <= m < (1 << B), (M, B, s, m)
    if not 1 <= s <= 63:
        raise SystemExit(f"STOP: s={s} outside [1, 63] for M={M!r}, B={B} (spec contradiction)")
    return int(m), int(s)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_layers(net: str, npz_path) -> list[dict]:
    d = np.load(npz_path)
    out = []
    for layer, k_expected in NET_LAYERS[net]:
        qw = d[f"{layer}_q_w"]
        K = int(np.prod(qw.shape[1:]))
        assert K == k_expected, f"{net}/{layer}: K={K} from q_w {qw.shape}, expected {k_expected}"
        M = np.asarray(d[f"{layer}_M"], dtype=np.float64)
        qb = np.asarray(d[f"{layer}_q_b"]).astype(np.int64)
        assert M.shape == qb.shape == (qw.shape[0],)
        assert np.all(M > 0) and np.all(np.isfinite(M))
        out.append({"layer": layer, "K": K, "M": M, "q_b": qb})
    return out


# --------------------------------------------------------------------------- #
# Per-channel check
# --------------------------------------------------------------------------- #
def regions(K: int, qb: int, M: float):
    """(vmin, vmax, lo, hi): reachable range and exact-region [lo, hi]."""
    vmin, vmax = -K * ACT_PROD_MAX + qb, K * ACT_PROD_MAX + qb
    vb = math.floor(Fraction(EXACT_BOUND) / Fraction(float(M)))   # |v| <= vb <=> |v*M| <= 130
    return vmin, vmax, max(-vb, vmin), min(vb, vmax)


def float_vs_exact_diffs(v: np.ndarray, M: float) -> int:
    """Count v where the float64 reference RNE differs from exact rational RNE(v*M).

    float64 error of v*M (|v*M| <= 130) is < 1e-13, so only v whose float product
    is within 1e-9 of a half-integer can differ; those are checked with Fraction.
    """
    y = v.astype(np.float64) * float(M)
    cand = v[np.abs(y - np.floor(y) - 0.5) < 1e-9]
    Mf = Fraction(float(M))
    n = 0
    for x in cand.tolist():
        if round(Fraction(x) * Mf) != int(np.rint(np.float64(x) * float(M))):
            n += 1
    return n


def candidate_deltas(max_delta: int) -> list[int]:
    """Search order around m_rne: 0, +1, -1, +2, -2, ..., +max_delta, -max_delta."""
    out = [0]
    for d in range(1, max_delta + 1):
        out += [d, -d]
    return out


def check_channel(K: int, qb: int, M: float, bits=BITS, seed=None, n_sat=N_SAT_SAMPLES,
                  exact_diag=True, spot_check=200):
    """Exhaustive + saturated check of one channel for each B. Returns list of dicts."""
    vmin, vmax, lo, hi = regions(K, qb, M)
    assert abs(vmin) < 1 << 30 and abs(vmax) < 1 << 30
    v_ex = np.arange(lo, hi + 1, dtype=np.int64) if hi >= lo else np.zeros(0, np.int64)
    ref_ex = requantize(v_ex, M)
    diag = float_vs_exact_diffs(v_ex, M) if exact_diag else ""

    # Saturated region: [vmin, lo-1] and [hi+1, vmax]
    n_neg, n_pos = max(0, lo - vmin), max(0, vmax - hi)
    rng = np.random.default_rng(seed)
    if n_neg + n_pos:
        idx = rng.integers(0, n_neg + n_pos, size=n_sat, dtype=np.int64)
        v_sat = np.where(idx < n_neg, vmin + idx, hi + 1 + (idx - n_neg))
    else:
        v_sat = np.zeros(0, np.int64)
    bounds = []                                      # (v, required sign)
    if n_pos:
        bounds += [(hi + 1, +1), (vmax, +1)]
    if n_neg:
        bounds += [(lo - 1, -1), (vmin, -1)]
    v_bd = np.array([b for b, _ in bounds], dtype=np.int64)
    v_sat_all = np.concatenate([v_bd, v_sat])
    ref_sat = requantize(v_sat_all, M)

    def evaluate(m: int, s: int) -> tuple[int, list[int]]:
        """Mismatch count and offending v for multiplier m, shift s."""
        mism_v = []
        # exact region
        hq = hw_requant(v_ex, m, s)
        bad = np.nonzero(hq != ref_ex)[0]
        mism = int(bad.size)
        if bad.size:
            mism_v.append(int(v_ex[bad[0]]))
        # saturated boundary (monotonicity argument) on the pre-clip value
        q_bd = hw_q(v_bd, m, s) if v_bd.size else v_bd
        for (vb_, sign), qv in zip(bounds, q_bd.tolist()):
            ok = qv >= 127 if sign > 0 else qv <= -128
            if not ok:
                mism += 1
                mism_v.append(int(vb_))
        # saturated boundary + random samples vs reference
        hs = hw_requant(v_sat_all, m, s)
        bad = np.nonzero(hs != ref_sat)[0]
        mism += int(bad.size)
        if bad.size:
            mism_v.append(int(v_sat_all[bad[0]]))
        return mism, mism_v

    results = []
    for B in bits:
        m_rne, s = select_m_s(M, B)
        mism_rne, v_rne = evaluate(m_rne, s)
        m, mism, mism_v, delta, tried = m_rne, mism_rne, v_rne, 0, 1
        if mism_rne:
            delta = ""
            for d in candidate_deltas(MAX_DELTA)[1:]:
                mc = m_rne + d
                if not (1 << (B - 1)) <= mc < (1 << B):
                    continue
                tried += 1
                mc_mism, mc_v = evaluate(mc, s)
                if mc_mism == 0:
                    m, mism, mism_v, delta = mc, 0, [], d
                    break
        # spot-check the arithmetic path against Python ints
        if spot_check and v_ex.size:
            pick = np.concatenate([v_ex[rng.integers(0, v_ex.size, spot_check)],
                                   v_sat_all[:min(spot_check, v_sat_all.size)]])
            got = hw_q(pick, m, s).tolist()
            want = [rne_shift_int(int(x) * m, s) for x in pick.tolist()]
            assert got == want, f"arithmetic path disagrees with Python ints (M={M}, B={B})"
        results.append({
            "B": B, "s": s, "m": m,
            "values_checked_exact": int(v_ex.size),
            "values_checked_saturated": int(v_sat_all.size),
            "mismatches": mism,
            "first_mismatch_v": mism_v[0] if mism_v else "",
            "m_rne": m_rne, "delta_from_rne": delta, "mismatches_rne": mism_rne,
            "candidates_tried": tried,
            "ref_float_vs_exact_diffs": diag,
        })
    return results


def mismatch_details(K, qb, M, B, v):
    m, s = select_m_s(M, B)
    q_hw = int(np.clip(rne_shift_int(v * m, s), -128, 127))
    q_ref = int(requantize(np.array([v], dtype=np.int64), M)[0])
    return {"v": v, "hw_q": q_hw, "ref_q": q_ref, "vM_exact": Fraction(v) * Fraction(float(M)),
            "vM_float": float(np.float64(v) * float(M)), "m": m, "s": s}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(nets, npz_paths, bits=BITS, n_sat=N_SAT_SAMPLES, out_csv=CSV_PATH, save=True,
        verbose=True):
    t0 = time.time()
    rows, summary = [], {}
    all_layers = {}
    for ni, net in enumerate(nets):
        layers = load_layers(net, npz_paths[net])
        all_layers[net] = layers
        for li, L in enumerate(layers):
            meta = base_meta(net=net, layer=L["layer"], source="model")
            tl = time.time()
            for c in range(L["M"].size):
                seed = [SAT_SEED, ni, li, c]
                res = check_channel(L["K"], int(L["q_b"][c]), float(L["M"][c]), bits, seed, n_sat)
                for r in res:
                    row = dict(meta)
                    row.update(net=net, layer=L["layer"], channel=c,
                               sat_seed="default_rng(%s)" % seed, **r)
                    rows.append(row)
                    agg = summary.setdefault((net, r["B"]), {"exact": 0, "sat": 0, "mism": 0,
                                                             "smin": 99, "smax": -1,
                                                             "mism_rne": 0, "adjusted": 0,
                                                             "max_abs_delta": 0})
                    agg["exact"] += r["values_checked_exact"]
                    agg["sat"] += r["values_checked_saturated"]
                    agg["mism"] += r["mismatches"]
                    agg["smin"] = min(agg["smin"], r["s"])
                    agg["smax"] = max(agg["smax"], r["s"])
                    agg["mism_rne"] += r["mismatches_rne"]
                    if r["delta_from_rne"] not in ("", 0):
                        agg["adjusted"] += 1
                        agg["max_abs_delta"] = max(agg["max_abs_delta"], abs(r["delta_from_rne"]))
            dt = time.time() - tl
            for row in rows:
                if row["net"] == net and row["layer"] == L["layer"]:
                    row["duration_s"] = round(dt, 3)
            if verbose:
                print(f"  {net}/{L['layer']}: {L['M'].size} ch, K={L['K']}, {dt:.1f}s", flush=True)

    write_results_csv(out_csv, rows, CSV_FIELDS)

    if verbose:
        print(f"\n{'net':8s} {'B':>3s} {'s_min':>5s} {'s_max':>5s} {'exact':>13s} "
              f"{'saturated':>13s} {'total':>13s} {'mism_rne':>8s} {'adj_ch':>6s} "
              f"{'max|d|':>6s} {'mismatches':>10s}")
        for (net, B), a in sorted(summary.items()):
            print(f"{net:8s} {B:3d} {a['smin']:5d} {a['smax']:5d} {a['exact']:13,d} "
                  f"{a['sat']:13,d} {a['exact'] + a['sat']:13,d} {a['mism_rne']:8d} "
                  f"{a['adjusted']:6d} {a['max_abs_delta']:6d} {a['mism']:10d}")
        diffs = sum(int(r["ref_float_vs_exact_diffs"] or 0) for r in rows if r["B"] == bits[0])
        print(f"\nfloat64 reference vs exact-rational RNE differences in exact regions: {diffs}")

    passing = [B for B in bits if all(summary[(n, B)]["mism"] == 0 for n in nets)]
    selected = min(passing) if passing else None
    if verbose:
        print(f"passing B (all channels, nets {list(nets)}): {passing}; selected B = {selected}")
    if selected is None:
        details = []
        for r in rows:
            if r["mismatches"] and r["B"] == max(bits):
                L = next(x for x in all_layers[r["net"]] if x["layer"] == r["layer"])
                c = r["channel"]
                details.append((r["net"], r["layer"], c,
                                mismatch_details(L["K"], int(L["q_b"][c]), float(L["M"][c]),
                                                 r["B"], int(r["first_mismatch_v"]))))
        print("STOP: no B passes. Mismatch details at largest B:")
        for d in details[:50]:
            print("  ", d)
    elif save:
        for net in nets:
            sel = {}
            for L in all_layers[net]:
                by = {r["channel"]: r for r in rows
                      if r["net"] == net and r["layer"] == L["layer"] and r["B"] == selected}
                ms = [(by[c]["m"], by[c]["s"]) for c in range(L["M"].size)]
                assert all(by[c]["mismatches"] == 0 for c in by)
                sel[f"{L['layer']}_m"] = np.array([m for m, _ in ms], dtype=np.uint64)
                sel[f"{L['layer']}_s"] = np.array([s for _, s in ms], dtype=np.uint8)
                assert all(int(x) == m for x, (m, _) in zip(sel[f"{L['layer']}_m"], ms))
            sel["B"] = np.array(selected, dtype=np.int64)
            HW_NPZ[net].parent.mkdir(parents=True, exist_ok=True)
            np.savez(HW_NPZ[net], **sel)
            if verbose:
                print(f"wrote {HW_NPZ[net].relative_to(REPO_ROOT)}")
    if verbose:
        print(f"wrote {out_csv} ({len(rows)} rows); "
              f"runtime {time.time() - t0:.1f}s")
    return {"rows": rows, "summary": summary, "passing": passing, "selected": selected,
            "runtime_s": time.time() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nets", nargs="+", default=["lenet5", "cifar10"], choices=list(NET_LAYERS))
    ap.add_argument("--lenet5-npz", default=str(DEFAULT_NPZ["lenet5"]))
    ap.add_argument("--cifar10-npz", default=str(DEFAULT_NPZ["cifar10"]))
    ap.add_argument("--bits", nargs="+", type=int, default=list(BITS))
    ap.add_argument("--n-sat", type=int, default=N_SAT_SAMPLES)
    ap.add_argument("--out-csv", default=str(CSV_PATH))
    ap.add_argument("--no-save", action="store_true",
                    help="do not write hw_requant.npz (always implied unless both nets run)")
    a = ap.parse_args(argv)
    npz = {"lenet5": a.lenet5_npz, "cifar10": a.cifar10_npz}
    for n in a.nets:
        if not Path(npz[n]).exists():
            raise SystemExit(f"missing {npz[n]}")
    save = not a.no_save and set(a.nets) == set(NET_LAYERS)
    if not save:
        print("note: hw_requant.npz not written (decision rule needs both nets)")
    res = run(a.nets, npz, tuple(a.bits), a.n_sat, a.out_csv, save)
    return 0 if res["selected"] is not None else 1


if __name__ == "__main__":
    sys.exit(main())
