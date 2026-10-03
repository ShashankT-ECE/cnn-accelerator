# copied from v2/board/stats.py @ 28dd2ad (unchanged; V3 baseline package)
"""Measurement statistics for the V2 board runs (numpy only; runs on the board and the laptop).

Conventions used by every board script (EXPERIMENTS.md "Measurement rigor"):

  warm-up     the first `warmup` measurements of a run (and of every interleaved block) are
              discarded before any statistic (caches, lazy init, branch predictors, DVFS);
  repeats     >= MIN_REPEATS (100) kept measurements per condition for a latency statistic
              (fewer -> the statistic is still computed but flagged `repeats_ok=False`);
  centre      median (robust to the long right tail of OS-jitter latencies); p95 as the tail;
  95 % CI     for the median (and any quantile q) the DISTRIBUTION-FREE ORDER-STATISTIC interval:
              with B ~ Binomial(n, q), take the order statistics x(l), x(u) (1-based) with
              l = the largest index such that P(B <= l-1) <= alpha/2 and u = the smallest index
              such that P(B >= u) <= alpha/2; then P(x(l) <= Q_q < x(u)) >= 1 - alpha for any
              continuous distribution (e.g. Conover, "Practical Nonparametric Statistics",
              3rd ed., sec. 3.2). The achieved coverage (>= 0.95, discrete) is reported. It needs
              no resampling, no seed and no distributional assumption, and is exact for i.i.d.
              samples; n < 6 gives no 95 % interval for the median (ci = None).
              Ratios / differences of medians (e.g. the PL clock calibration) use a percentile
              BOOTSTRAP (bootstrap_ci, seeded, 2000 resamples by default; seed recorded).
  interleave  conditions compared in one run (safe vs fast host path, accelerator vs CPU,
              accel vs control power phases) run in BLOCKS whose order is a random permutation
              of the conditions in every round (a randomized ABAB design; block_order(), seed
              recorded), so a slow drift (temperature, background load) hits all conditions
              alike instead of biasing the one measured last.
  sessions    3-session repeatability: each full campaign is tagged session_index 1..3; the
              per-session statistic (median for latencies, repeat mean for power) is combined
              across sessions as mean of the per-session values +- between-session SD (ddof=1),
              range and CV (between_sessions(); aggregate_sessions.py). Rows from different
              bitstreams or clocks are never combined.
"""
from __future__ import annotations

import math
import random

import numpy as np

MIN_REPEATS = 100
CONF = 0.95
CI_METHOD_ORDER = "order-statistic (binomial, distribution-free)"
CI_METHOD_BOOT = "percentile bootstrap"


def discard_warmup(x, warmup: int):
    """x without its first `warmup` entries (numpy array)."""
    a = np.asarray(x)
    return a[max(0, int(warmup)):]


def _log_binom_pmf(n: int, q: float):
    k = np.arange(n + 1, dtype=np.float64)
    lg = np.vectorize(math.lgamma)
    if q <= 0.0 or q >= 1.0:
        raise ValueError("q must be in (0, 1)")
    return (math.lgamma(n + 1) - lg(k + 1) - lg(n - k + 1)
            + k * math.log(q) + (n - k) * math.log1p(-q))


def order_stat_ci_indices(n: int, q: float = 0.5, conf: float = CONF):
    """(l, u, coverage) 1-based order-statistic indices of the distribution-free CI for the
    q-quantile, or None if n is too small for the requested confidence."""
    if n < 1:
        return None
    alpha = 1.0 - conf
    pmf = np.exp(_log_binom_pmf(n, q))
    cdf = np.cumsum(pmf)                       # cdf[k] = P(B <= k)
    # lower: largest l (1..n) with P(B <= l-1) <= alpha/2
    ok_l = np.flatnonzero(cdf[:n] <= alpha / 2)          # index k = l-1
    # upper: smallest u (1..n) with P(B >= u) = 1 - P(B <= u-1) <= alpha/2
    sf = 1.0 - np.concatenate(([0.0], cdf[:n]))          # sf[u] = P(B >= u), u = 0..n
    ok_u = np.flatnonzero(sf[1:] <= alpha / 2) + 1       # u = 1..n
    if ok_l.size == 0 or ok_u.size == 0:
        return None
    lo = int(ok_l[-1]) + 1
    hi = int(ok_u[0])
    if lo > hi:
        return None
    cov = float(pmf[lo:hi].sum())          # P(lo <= B <= hi-1) = P(x(lo) <= Q_q < x(hi))
    return lo, hi, cov


def quantile_ci(x, q: float = 0.5, conf: float = CONF):
    """(lo, hi, coverage) distribution-free CI of the q-quantile, or (None, None, None)."""
    a = np.sort(np.asarray(x, dtype=np.float64))
    r = order_stat_ci_indices(a.size, q, conf)
    if r is None:
        return None, None, None
    lo, hi, cov = r
    return float(a[lo - 1]), float(a[hi - 1]), cov


def median_ci(x, conf: float = CONF):
    return quantile_ci(x, 0.5, conf)


def bootstrap_ci(x, stat=np.median, n_boot: int = 2000, conf: float = CONF, seed: int = 0):
    """Percentile bootstrap CI of stat(x). Returns (lo, hi)."""
    a = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    bs = np.array([stat(a[i]) for i in idx])
    al = (1.0 - conf) / 2
    return float(np.quantile(bs, al)), float(np.quantile(bs, 1 - al))


def summarize(x, conf: float = CONF, min_repeats: int = MIN_REPEATS) -> dict:
    """Median + order-statistic CI, p5/p95/p99, mean, sd, min, max, n (floats; NaN if empty)."""
    a = np.asarray(x, dtype=np.float64)
    n = int(a.size)
    if n == 0:
        nan = math.nan
        return {"n": 0, "median": nan, "ci_lo": None, "ci_hi": None, "ci_coverage": None,
                "ci_method": CI_METHOD_ORDER, "p5": nan, "p95": nan, "p99": nan, "mean": nan,
                "sd": nan, "min": nan, "max": nan, "repeats_ok": False}
    lo, hi, cov = median_ci(a, conf)
    return {"n": n, "median": float(np.median(a)), "ci_lo": lo, "ci_hi": hi, "ci_coverage": cov,
            "ci_method": CI_METHOD_ORDER,
            "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if n > 1 else math.nan,
            "min": float(a.min()), "max": float(a.max()), "repeats_ok": n >= min_repeats}


def fmt(v, nd: int = 3) -> str:
    """CSV text of a float ('' for None / NaN)."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "" if math.isnan(f) else f"{f:.{nd}f}"


def ci_fields(s: dict, unit: str = "us", nd: int = 3, prefix: str = "median") -> dict:
    """Standard CSV columns of a summarize() result: <prefix>_ci_lo_<unit>, ..."""
    return {f"{prefix}_ci_lo_{unit}": fmt(s["ci_lo"], nd), f"{prefix}_ci_hi_{unit}": fmt(s["ci_hi"], nd),
            "ci_coverage": fmt(s["ci_coverage"], 4), "ci_method": s["ci_method"],
            "repeats_ok": s["repeats_ok"]}


# ---- interleaving -------------------------------------------------------------------------------
def block_order(conditions, n_rounds: int, seed: int, mode: str = "random") -> list:
    """Flat list of condition labels, one entry per block. Every round contains each condition
    once; mode 'random' = an independent random permutation per round (seeded), 'abab' = the
    same order every round with a seeded random rotation of the first round."""
    conds = list(conditions)
    rng = random.Random(seed)
    out = []
    if mode == "abab":
        k = rng.randrange(len(conds)) if conds else 0
        base = conds[k:] + conds[:k]
        for _ in range(n_rounds):
            out += base
        return out
    if mode != "random":
        raise ValueError(mode)
    for _ in range(n_rounds):
        p = conds[:]
        rng.shuffle(p)
        out += p
    return out


def new_seed() -> int:
    """A fresh seed for an interleaved run (recorded with the results)."""
    return random.SystemRandom().randrange(1, 2**31 - 1)


# ---- sessions -----------------------------------------------------------------------------------
def between_sessions(values) -> dict:
    """Combine per-session statistics: mean, SD (ddof=1), min, max, range, CV %, n."""
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    n = int(v.size)
    if n == 0:
        return {"n": 0}
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else math.nan
    return {"n": n, "mean": m, "sd": sd, "min": float(v.min()), "max": float(v.max()),
            "range": float(v.max() - v.min()),
            "cv_pct": (100.0 * sd / abs(m)) if n > 1 and m != 0 else math.nan}
