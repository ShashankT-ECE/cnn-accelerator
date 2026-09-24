"""Tests for v2/model/requant_check.py (V2 step 2.1 part B)."""
from __future__ import annotations

import csv
import math
from fractions import Fraction

import numpy as np
import pytest

import requant_check as rc
from net_config import NET_CONFIGS

LENET_NPZ = rc.DEFAULT_NPZ["lenet5"]
CIFAR_NPZ = rc.DEFAULT_NPZ["cifar10"]


def test_fraction_round_is_half_even():
    assert [round(Fraction(n, 2)) for n in (-5, -3, -1, 1, 3, 5)] == [-2, -2, 0, 0, 2, 2]
    assert round(Fraction(7, 4)) == 2 and round(Fraction(-7, 4)) == -2


def _random_p_s(rng, n):
    s = rng.integers(1, 63, size=n)                           # s in [1, 62]
    mag = rng.integers(0, 62, size=n)                         # |p| < 2^mag <= 2^61
    p = np.array([int(rng.integers(0, 1 << int(b))) if b else 0 for b in mag], dtype=np.int64)
    p = np.where(rng.random(n) < 0.5, -p, p)
    return p, s


def _ties(rng, n):
    s = rng.integers(1, 63, size=n)
    out = []
    for si in s.tolist():
        kmax = max(1, (1 << (61 - (si - 1))) // 2)            # |(2k+1)*2^(s-1)| < 2^62
        k = int(rng.integers(0, min(kmax, 1 << 40)))
        p = (2 * k + 1) << (si - 1)
        out.append(-p if rng.random() < 0.5 else p)
    return np.array(out, dtype=np.int64), s


def test_rne_shift_vs_fraction():
    rng = np.random.default_rng(12345)
    p1, s1 = _random_p_s(rng, 150_000)
    p2, s2 = _ties(rng, 60_000)
    # small-magnitude cases where the rounding is non-trivial for every s
    s3 = rng.integers(1, 63, size=20_000)
    p3 = np.array([int(rng.integers(-(1 << min(int(x) + 2, 61)), 1 << min(int(x) + 2, 61))) for x in s3],
                  dtype=np.int64)
    p = np.concatenate([p1, p2, p3])
    s = np.concatenate([s1, s2, s3])
    assert p.size >= 200_000 and int((p < 0).sum()) > 50_000
    got = rc.rne_shift(p, s).tolist()
    n_ties = 0
    for pi, si, gi in zip(p.tolist(), s.tolist(), got):
        f = Fraction(pi, 1 << si)
        want = round(f)
        n_ties += (f.denominator == 2)
        assert gi == want, (pi, si, gi, want)
        assert rc.rne_shift_int(pi, si) == want
    assert n_ties >= 60_000


def test_rne_shift_int_wide():
    rng = np.random.default_rng(7)
    for _ in range(20_000):
        s = int(rng.integers(1, 120))
        p = int(rng.integers(-(1 << 62), 1 << 62)) * int(rng.integers(1, 1 << 40))
        if rng.random() < 0.3:
            p = ((2 * (p >> s) + 1) << (s - 1))               # exact tie
        assert rc.rne_shift_int(p, s) == round(Fraction(p, 1 << s))


def _check_ms(M, B):
    m, s = rc.select_m_s(M, B)
    assert (1 << (B - 1)) <= m < (1 << B)
    assert 1 <= s <= 63
    assert abs(Fraction(m, 1 << s) - Fraction(M)) <= Fraction(1, 1 << (s + 1))
    assert m == round(Fraction(M) * (1 << s))
    return m, s


@pytest.mark.parametrize("B", rc.BITS)
def test_select_m_s_invariants(B):
    Ms = [float(M) for L in rc.load_layers("lenet5", LENET_NPZ) for M in L["M"]]
    for key in ("cifar10", "cifar10_r1"):
        if NET_CONFIGS[key]["quant_params"].exists():
            Ms += [float(M) for L in rc.load_layers("cifar10", NET_CONFIGS[key]["quant_params"])
                   for M in L["M"]]
    rng = np.random.default_rng(3)
    Ms += list(np.exp(rng.uniform(np.log(4e-5), np.log(0.9), 2000)))
    Ms += [0.5, 0.25, 2.0 ** -14, np.nextafter(0.5, 0), np.nextafter(0.5, 1)]
    oor = []
    for M in Ms:
        try:
            _check_ms(M, B)
        except rc.ShiftOutOfRange as e:             # OC-2: must be a real range violation
            assert e.B == B and not rc.S_MIN <= e.s <= rc.S_MAX
            assert e.s == B - math.frexp(M)[1] or e.s == B - math.frexp(M)[1] - 1
            oor.append(M)
    # Only CIFAR r2 at B = 48 needs s > 63 (DECISIONS OC-2); B = 32 / 40 are always in range.
    assert (len(oor) > 0) == (B == 48 and NET_CONFIGS["cifar10"]["quant_params"].exists())


def test_out_of_range_shift_recorded_not_fatal():
    """OC-2: an s outside [1, 63] marks that (channel, B) infeasible; other B are still checked."""
    M = 8.2e-6                                      # ~ CIFAR r2 conv1 min M: s = 64 at B = 48
    with pytest.raises(rc.ShiftOutOfRange):
        rc.select_m_s(M, 48)
    res = {r["B"]: r for r in rc.check_channel(75, 0, M, bits=(32, 48), seed=0, n_sat=2_000)}
    assert res[32]["s_in_range"] is True and res[32]["mismatches"] == 0 and res[32]["s"] == 48
    r = res[48]
    assert r["s_in_range"] is False and r["s"] == 64 and r["m"] == "" and r["mismatches"] == ""
    assert r["values_checked_exact"] == r["values_checked_saturated"] == 0


def test_adopted_shifts_in_range():
    """Every adopted (selected-B) shift s is in [1, 63] for every requant channel of every
    adopted net, in hw_requant.npz and in the packed QPARAM image (OC-2)."""
    import gos_pack as gp
    from net_config import NETS
    for net in NETS:
        hw = np.load(NET_CONFIGS[net]["hw_requant"])
        B = int(hw["B"])
        assert B == gp.M_BITS
        words, bases = gp.pack_qparam(net)
        for L, base in zip(NET_CONFIGS[net]["layers"], bases):
            _, m_pk, s_pk = gp.unpack_qparam(words, base, L["OC"])
            if L["final"]:
                assert not np.any(s_pk) and not np.any(m_pk)          # unused (out_raw)
                continue
            s = hw[f"{L['name']}_s"]
            assert s.shape == (L["OC"],)
            assert np.all((s >= rc.S_MIN) & (s <= rc.S_MAX)), (net, L["name"], s.min(), s.max())
            assert np.array_equal(s_pk.astype(np.int64), s.astype(np.int64)), (net, L["name"])
            assert np.array_equal(m_pk, hw[f"{L['name']}_m"].astype(np.uint64))
            m = hw[f"{L['name']}_m"].astype(object)
            assert all((1 << (B - 1)) <= int(x) < (1 << B) for x in m)


@pytest.mark.slow
def test_full_exhaustive_cifar10_decision(tmp_path):
    """CIFAR r2: B = 48 infeasible (s = 64), B = 32 and 40 pass; selected B = 32 (OC-2)."""
    res = rc.run(["cifar10"], rc.DEFAULT_NPZ, out_csv=tmp_path / "r.csv", save=False,
                 verbose=False)
    assert res["passing"] == [32, 40] and res["selected"] == 32
    assert res["summary"][("cifar10", 48)]["s_oor"] > 0
    assert res["summary"][("cifar10", 32)]["s_oor"] == 0
    assert res["summary"][("cifar10", 32)]["mism"] == 0


@pytest.mark.parametrize("B", rc.BITS)
def test_select_m_s_round_up_edge(B):
    # M just below a power of two: M*2^(B-E) rounds to 2^B -> s is decremented.
    M = float(np.nextafter(2.0 ** -10, 0))
    m, s = _check_ms(M, B)
    assert m == 1 << (B - 1)


def test_wide_path_vs_python_ints():
    rng = np.random.default_rng(99)
    for B in rc.BITS:
        for _ in range(300):
            m = int(rng.integers(1 << (B - 1), (1 << B) - 1, dtype=np.uint64))
            s = int(rng.integers(max(1, B - 8), 64))
            v = rng.integers(-(1 << 24), 1 << 24, size=500, dtype=np.int64)
            # force ties / near ties around representable half-points
            v[:5] = [0, 1, -1, (1 << 24) - 1, -(1 << 24)]
            got = rc.hw_q_wide(v, m, s).tolist()
            want = [rc.rne_shift_int(x * m, s) for x in v.tolist()]
            assert got == want
            # dispatcher agrees too
            assert rc.hw_q(v, m, s).tolist() == want


def test_wide_path_exact_ties():
    # Choose v, m so that v*m is exactly (2j+1)*2^(s-1): p/2^s is a half-integer.
    rng = np.random.default_rng(5)
    for _ in range(2000):
        s = int(rng.integers(33, 64))
        m = 1 << (s - 1 - int(rng.integers(0, 20)))            # power of two
        shift = s - 1 - (m.bit_length() - 1)
        v = np.array([(2 * int(rng.integers(-1000, 1000)) + 1) << shift], dtype=np.int64)
        if abs(int(v[0])) >= 1 << 30:
            continue
        want = round(Fraction(int(v[0]) * m, 1 << s))
        assert rc.hw_q_wide(v, m, s)[0] == want == rc.rne_shift_int(int(v[0]) * m, s)


def test_narrow_equals_wide_where_valid():
    rng = np.random.default_rng(11)
    for _ in range(200):
        m = int(rng.integers(1 << 31, 1 << 32))
        s = int(rng.integers(32, 62))
        v = rng.integers(-(1 << 29), 1 << 29, size=1000, dtype=np.int64)
        assert rc.hw_q_narrow(v, m, s).tolist() == rc.hw_q_wide(v, m, s).tolist()


def test_end_to_end_one_lenet_channel():
    L = rc.load_layers("lenet5", LENET_NPZ)[0]                 # conv1, K=25
    assert L["K"] == 25
    M, qb = float(L["M"][0]), int(L["q_b"][0])
    res = rc.check_channel(L["K"], qb, M, seed=0, n_sat=20_000)
    vmin, vmax, lo, hi = rc.regions(L["K"], qb, M)
    for r in res:
        assert r["values_checked_exact"] == hi - lo + 1 > 100_000
        assert r["mismatches"] == 0 and r["first_mismatch_v"] == ""
    # independent recomputation on the full exact region
    v = np.arange(lo, hi + 1, dtype=np.int64)
    ref = rc.requantize(v, M)
    for B in rc.BITS:
        m, s = rc.select_m_s(M, B)
        assert np.array_equal(rc.hw_requant(v, m, s), ref)


def test_known_near_tie_channels_detected():
    """Channels where the RNE-of-M multiplier cannot reproduce the float64
    reference at a near-tie (found by the exhaustive run): the check must flag them."""
    Ls = {L["layer"]: L for L in rc.load_layers("lenet5", LENET_NPZ)}
    for layer, c, v in (("conv5", 82, 55930), ("fc1", 54, 31044)):
        L = Ls[layer]
        M = float(L["M"][c])
        ref = int(rc.requantize(np.array([v]), M)[0])
        exact = round(Fraction(v) * Fraction(M))
        m, s = rc.select_m_s(M, 48)
        hw = int(rc.hw_requant(np.array([v]), m, s)[0])
        assert hw != ref
        assert abs(Fraction(v) * Fraction(M) - Fraction(127, 2)) < Fraction(1, 10 ** 13)
        if layer == "fc1":
            assert exact != ref       # float64 reference differs from exact rational RNE


def _hw_npz_cases():
    return [(n, rc.HW_NPZ[n]) for n in rc.NET_LAYERS if rc.HW_NPZ[n].exists()]


@pytest.mark.skipif(not _hw_npz_cases(), reason="hw_requant.npz not produced (no B selected)")
def test_hw_requant_npz_matches_csv_and_reference():
    rows = list(csv.DictReader(rc.CSV_PATH.open()))
    for net, path in _hw_npz_cases():
        d = np.load(path)
        B = int(d["B"])
        by = {(r["layer"], int(r["channel"])): r for r in rows
              if r["net"] == net and int(r["B"]) == B}
        npz_in = rc.DEFAULT_NPZ[net]
        for L in rc.load_layers(net, npz_in):
            ms, ss = d[f"{L['layer']}_m"], d[f"{L['layer']}_s"]
            assert ms.dtype == np.uint64 and ss.dtype == np.uint8
            for c in range(L["M"].size):
                r = by[(L["layer"], c)]
                assert int(r["m"]) == int(ms[c]) and int(r["s"]) == int(ss[c])
                assert int(r["mismatches"]) == 0
            for c in sorted({0, L["M"].size // 2, L["M"].size - 1}):
                M, qb = float(L["M"][c]), int(L["q_b"][c])
                _, _, lo, hi = rc.regions(L["K"], qb, M)
                v = np.arange(lo, hi + 1, dtype=np.int64)
                assert np.array_equal(rc.hw_requant(v, int(ms[c]), int(ss[c])),
                                      rc.requantize(v, M))


@pytest.mark.slow
def test_full_exhaustive_lenet(tmp_path):
    res = rc.run(["lenet5"], rc.DEFAULT_NPZ, out_csv=tmp_path / "r.csv", save=False,
                 verbose=False)
    for B in rc.BITS:
        a = res["summary"][("lenet5", B)]
        assert a["exact"] > 0 and a["sat"] > 0
        assert 1 <= a["smin"] <= a["smax"] <= 63
    # consistency of decision rule with the per-row results
    for B in rc.BITS:
        ok = all(r["s_in_range"] and int(r["mismatches"]) == 0 for r in res["rows"] if r["B"] == B)
        assert ok == (B in res["passing"])


def test_candidate_delta_order():
    assert rc.candidate_deltas(3) == [0, 1, -1, 2, -2, 3, -3]


def _full_region_equal(net, layer, c, m, s, npz=None):
    npz = rc.DEFAULT_NPZ[net] if npz is None else npz
    L = next(x for x in rc.load_layers(net, npz) if x["layer"] == layer)
    M, qb = float(L["M"][c]), int(L["q_b"][c])
    _, _, lo, hi = rc.regions(L["K"], qb, M)
    v = np.arange(lo, hi + 1, dtype=np.int64)
    return np.array_equal(rc.hw_requant(v, m, s), rc.requantize(v, M))


# (NET_CONFIGS key, layer, channel): the only adjusted channel per parameter set at B=32
# (DECISIONS D1 outcome; cifar10 r2 per requant_equivalence.csv, Step 2.1c).
@pytest.mark.parametrize("key,layer,c", [("lenet5", "conv5", 82), ("cifar10", "conv3", 57),
                                         ("cifar10_r1", "conv2", 0)])
def test_adjusted_channels_at_B32(key, layer, c):
    """OC-1: at B=32 the RNE multiplier fails these channels; the searched m (±1 LSB) is exact."""
    net = "lenet5" if key == "lenet5" else "cifar10"
    npz = NET_CONFIGS[key]["quant_params"]
    if not npz.exists():
        pytest.skip(f"{key} frozen params missing")
    L = next(x for x in rc.load_layers(net, npz) if x["layer"] == layer)
    m_rne, s = rc.select_m_s(float(L["M"][c]), 32)
    assert not _full_region_equal(net, layer, c, m_rne, s, npz)
    res = rc.check_channel(L["K"], int(L["q_b"][c]), float(L["M"][c]), bits=(32,),
                           seed=0, n_sat=20_000)[0]
    assert res["mismatches_rne"] > 0 and res["mismatches"] == 0
    assert abs(res["delta_from_rne"]) == 1 and res["m"] == m_rne + res["delta_from_rne"]
    assert _full_region_equal(net, layer, c, res["m"], s, npz)
    hw_npz = NET_CONFIGS[key]["hw_requant"]           # the parameter set's own (m, s)
    if hw_npz.exists():
        d = np.load(hw_npz)
        assert int(d["B"]) == 32
        assert int(d[f"{layer}_m"][c]) == res["m"] and int(d[f"{layer}_s"][c]) == s
