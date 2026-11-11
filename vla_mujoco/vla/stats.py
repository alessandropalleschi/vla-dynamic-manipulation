"""Small statistics helpers for the language probe (numpy only)."""

from __future__ import annotations

from math import comb

import numpy as np


def bootstrap_ci(x, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile bootstrap CI of the mean, resampling *episodes* (the independent units)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def sign_test_p(x) -> float:
    """Two-sided exact sign test p-value for 'the median difference is zero' (ties are dropped)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n))


def summarize(diffs, n_boot: int = 4000, seed: int = 0) -> dict:
    """Per-episode paired differences (positive = the language helps) -> mean, 95% CI, share positive, sign test."""
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    mean, lo, hi = bootstrap_ci(d, n_boot, seed=seed)
    return {
        "n": int(len(d)),
        "mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "frac_positive": float((d > 0).mean()) if len(d) else float("nan"),
        "sign_p": sign_test_p(d),
        "significant": bool(len(d) and lo > 0),  # the whole 95% CI above zero
    }
