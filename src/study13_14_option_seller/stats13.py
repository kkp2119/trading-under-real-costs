#!/usr/bin/env python3
"""Research 13 statistics (SPEC 5.3, 6, 9, 10, 11; A13-2; A13-3 E13-09..E13-12). Daily returns r = P/N0.
Bootstrap: copied from research12/stats.py (sha256 98200ddd...; boot_starts, percentile ci, deflated_sharpe) with
SEED 13 and BLOCK 21. research12 summed resamples with a numba kernel; numba is not installed in this environment,
so boot_sharpes sums the same resamples with numpy block sums (same block starts, same circular blocks)."""
import math

import numpy as np
from scipy.stats import kurtosis, norm, skew

import config13 as C


def sharpe(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(C.ANN)) if sd > 0 else float("nan")


def boot_starts(T, B=C.B_BOOT, block=C.BLOCK, seed=C.SEED):
    """Block start indices (B x ceil(T / block)) for a series of length T (research12 boot_starts)."""
    return np.random.default_rng(seed).integers(0, T, size=(B, -(-T // block)))


def boot_sharpes(x, starts, block=C.BLOCK):
    """Annualised Sharpe of each circular-block resample (the first T values of the concatenated blocks)."""
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    nb = starts.shape[1]
    xx = np.concatenate([x, x[:block]])
    c1 = np.concatenate([[0.0], np.cumsum(xx)])
    c2 = np.concatenate([[0.0], np.cumsum(xx * xx)])
    L = np.full(nb, block)
    L[-1] = T - block * (nb - 1)
    st = starts % T
    s1 = (c1[st + L] - c1[st]).sum(axis=1)
    s2 = (c2[st + L] - c2[st]).sum(axis=1)
    m = s1 / T
    var = (s2 - T * m * m) / (T - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(var > 0, m / np.sqrt(var) * math.sqrt(C.ANN), np.nan)


def ci(x, level=0.95, block=C.BLOCK, B=C.B_BOOT, seed=C.SEED):
    x = np.asarray(x, dtype=np.float64)
    sh = boot_sharpes(x, boot_starts(len(x), B, block, seed), block)
    a = (1 - level) / 2
    return float(np.nanpercentile(sh, 100 * a)), float(np.nanpercentile(sh, 100 * (1 - a)))


def judge(point, lo, hi):
    """SPEC 11 with the simultaneity rule: ('a'|'b'|'c', reason)."""
    a = lo > 0 and point >= C.SR_STAR
    b = hi < C.SR_STAR
    if a and b:
        return "c", "CI와 점추정 불일치"
    if a:
        return "a", ""
    if b:
        return "b", ""
    return "c", ""


def downgrade(verdict, reason, defect_ratio, d_dev):
    """SPEC 9: holdout defect ratio above max(5%, d_dev + 5%p) -> (c) with the reason."""
    thr = max(C.DEFECT_FLOOR, d_dev + C.DEFECT_MARGIN)
    if defect_ratio > thr:
        return "c", (reason + "; " if reason else "") + f"결함일 비율 {defect_ratio:.4f} > 기준 {thr:.4f}"
    return verdict, reason


def tail_label(verdict, mdd_hold, mdd_dev):
    if verdict != "a":
        return ""
    return "a-안정" if mdd_hold <= C.TAIL_RATIO * mdd_dev else "a-꼬리주의"


def lo_sharpe(x, lags=C.LO_LAGS, q=C.ANN):
    """E13-09: SR_daily * q / sqrt(q + 2 sum_{k=1}^{lags} (q - k) rho_k)."""
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(ddof=1)
    if len(x) < lags + 2 or sd <= 0:
        return float("nan")
    xc = x - x.mean()
    den = float(np.dot(xc, xc))
    rho = [float(np.dot(xc[k:], xc[:-k]) / den) for k in range(1, lags + 1)]
    d = q + 2 * sum((q - k) * rk for k, rk in enumerate(rho, start=1))
    return float(x.mean() / sd * q / math.sqrt(d)) if d > 0 else float("nan")


def psr(x, sr_star_daily):
    """E13-10: probabilistic Sharpe ratio against a daily benchmark."""
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    sr = x.mean() / x.std(ddof=1)
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    return float(norm.cdf((sr - sr_star_daily) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


def deflated_sharpe(x, n_trials):
    """E13-11: research12 deflated_sharpe (null variance of the daily Sharpe 1 / (T - 1))."""
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    sr = x.mean() / x.std(ddof=1)
    g = 0.5772156649
    sr0 = math.sqrt(1 / (T - 1)) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * math.e)))
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    return float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


def max_drawdown(x):
    """E13-12: largest fall of the arithmetic cumulative sum from its running peak (starting at 0)."""
    c = np.cumsum(np.asarray(x, dtype=np.float64))
    peak = np.maximum.accumulate(np.concatenate(([0.0], c)))[1:]
    return float(np.max(peak - c)) if len(c) else 0.0


def worst_window(x, n):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < n:
        return float("nan")
    c = np.concatenate([[0.0], np.cumsum(x)])
    return float(np.min(c[n:] - c[:-n]))


def tails(x):
    """6: tail metrics of daily returns x (fractions of N0)."""
    x = np.asarray(x, dtype=np.float64)
    mdd = max_drawdown(x)
    k = max(1, int(math.ceil(0.01 * len(x))))
    return {"worst_day": float(x.min()), "worst_5d": worst_window(x, 5), "worst_21d": worst_window(x, 21),
            "max_drawdown": mdd, "calmar": float(x.mean() * C.ANN / mdd) if mdd > 0 else float("nan"),
            "skew": float(skew(x)), "kurtosis": float(kurtosis(x, fisher=False)),
            "bottom1pct_mean": float(np.sort(x)[:k].mean())}


def betas(x, mkt):
    """6: OLS of x on the SPX daily return with intercept; downside beta on days with mkt < 0."""
    x, m = np.asarray(x, float), np.asarray(mkt, float)
    ok = np.isfinite(x) & np.isfinite(m)
    x, m = x[ok], m[ok]

    def ols(a, b):
        b1 = np.vstack([np.ones(len(b)), b]).T
        coef = np.linalg.lstsq(b1, a, rcond=None)[0]
        return float(coef[1])
    dn = m < 0
    return {"beta": ols(x, m), "corr": float(np.corrcoef(x, m)[0, 1]),
            "downside_beta": ols(x[dn], m[dn]) if dn.sum() > 2 else float("nan")}


def events(dates, x):
    dates, x = np.asarray(dates), np.asarray(x, float)
    return {k: (float(x[(dates >= a) & (dates <= b)].sum()) if ((dates >= a) & (dates <= b)).any() else None)
            for k, (a, b) in C.EVENTS.items()}


def defect_ratio(rows, setting_rows=None):
    """9 + A13-2: share of days with defect (i), (ii) or (iii) (u > 0.20) in the given ledger rows."""
    import pandas as pd
    df = pd.DataFrame(rows) if not hasattr(rows, "columns") else rows
    d = (df.defect_i.astype(bool) | df.defect_ii.astype(bool) | (df.u > C.DEFECT_U))
    return float(d.mean()) if len(df) else float("nan")
