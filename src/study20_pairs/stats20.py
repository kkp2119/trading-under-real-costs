#!/usr/bin/env python3
"""Research 20 statistics shared by dev20.py and run_test20.py (SPEC20 4-5, SPEC20_ANSWERS A4, DEVIATIONS20 1.16-1.19).

Daily series cover every trading day of the window (a day without a trade is 0). Day-block bootstrap: engine17.boot_index
with 10,000 resamples, circular blocks of 21 days, seed 20. One-sided p for "mean > 0" = share of bootstrap means <= 0."""
import math

import numpy as np

import pair20 as PE

G = PE.G
BOOT_B, BOOT_BLOCK, BOOT_SEED = 10_000, 21, 20
EXECS = (("M", PE.M), ("P", PE.P))
TIE = 0.05
DEFAULT_SIGNAL = (30, 2.0, 0.0)                     # SPEC20 5: the H1 setting when there is no H2


def configs():
    """Signal settings in grid order: (W, z_in, kappa)."""
    S = PE.S
    return [(W, zi, ka) for W in S.WS for zi in S.ZINS for ka in S.KAPPAS]


def sig_name(c):
    return f"W{c[0]}_z{c[1]:g}_k{c[2]:g}"


def cand_name(c, ex):
    return f"{sig_name(c)}_{ex}"


def candidates():
    """The 16 candidates in order: W, z_in, kappa, execution (M, P)."""
    return [(c, ex) for c in configs() for ex, _ in EXECS]


def boot_index(T):
    return G.boot_index(T, B=BOOT_B, block=BOOT_BLOCK, seed=BOOT_SEED)


def daily(day_codes, values, n_days):
    """Sum of `values` per day code (0..n_days-1); days without a value are 0."""
    return np.bincount(np.asarray(day_codes, dtype=np.int64), weights=np.asarray(values, dtype=np.float64), minlength=n_days)


def mean_ci(x, idx):
    """(mean, 2.5%, 97.5%) of the day-block bootstrap means of a daily series."""
    x = np.asarray(x, dtype=np.float64)
    bm = x[idx].mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def p_positive(x, idx):
    """(mean, one-sided p for mean > 0 = share of bootstrap means <= 0, one-sided 2.5% lower bound)."""
    x = np.asarray(x, dtype=np.float64)
    bm = x[idx].mean(axis=1)
    return float(x.mean()), float(np.mean(bm <= 0)), float(np.percentile(bm, 2.5))


def ratio_ci(num, den, idx):
    """Per-trade mean = sum of daily sums / sum of daily counts, with the 95% interval over resampled days."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    if den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    d = den[idx].sum(axis=1)
    r = np.where(d > 0, num[idx].sum(axis=1) / np.where(d > 0, d, 1.0), np.nan)
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return float(num.sum() / den.sum()), float(lo), float(hi)


def sharpe(x):
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(252)) if sd > 0 else float("nan")


def select_astar(rows):
    """SPEC20 4: the highest L3D1 mean daily P&L; among the candidates within 5% of it (difference <= 0.05 x |best|) the
    one with the fewest trades, then the earlier candidate. rows: [{"name", "mean", "trades"}] in candidate order."""
    best = max(r["mean"] for r in rows)
    tied = [r for r in rows if best - r["mean"] <= TIE * abs(best)]
    pick = min(tied, key=lambda r: (r["trades"], [x["name"] for x in rows].index(r["name"])))
    return pick["name"], {"highest_mean": max(rows, key=lambda r: r["mean"])["name"], "highest_mean_usd": best,
                          "tied_within_5pct": [r["name"] for r in tied], "trades_of_tied": {r["name"]: r["trades"] for r in tied},
                          "astar": pick["name"], "astar_mean_usd": pick["mean"], "profitable_in_development": bool(pick["mean"] > 0)}


def mde(x, idx, t_new, alphas=(0.025, 0.05), power=0.80):
    """Minimum detectable mean of a daily series at t_new days (one-sided): SE = sqrt(V x VIF / t_new), VIF = bootstrap
    variance of the mean / (V / T); MDE = (z_{1-a} + z_{power}) x SE."""
    from scipy.stats import norm
    x = np.asarray(x, dtype=np.float64)
    T, V = len(x), float(x.var(ddof=1))
    vb = float(x[idx].mean(axis=1).var(ddof=1))
    vif = vb / (V / T) if V > 0 else float("nan")
    se = math.sqrt(V * vif / t_new) if V > 0 else float("nan")
    return {"days_dev": T, "days_new": int(t_new), "var_daily": V, "VIF": vif, "se_new": se,
            "mde": {str(a): float((norm.ppf(1 - a) + norm.ppf(power)) * se) for a in alphas}}


def holm(p):
    return G.holm(p, alpha=0.05)
