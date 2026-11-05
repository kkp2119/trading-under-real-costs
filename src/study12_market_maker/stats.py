"""Research 12 statistics (SPEC 1, 10; E1-45 decision (b): daily-frequency Sharpe SE). Daily series = dollars per day.
- Sharpe = mean / sd (ddof 1) x sqrt(252), no risk-free rate; NaN when sd is 0 or fewer than 2 days (E1-35).
- CI: circular block bootstrap, block 10 trading days, 10,000 resamples, percentile, seed 12 (block starts drawn once per
  series length with numpy default_rng(seed), as in research 8-11).
- MDE (80% power, two-sided 95%): SR = z sqrt((1 + SR^2 / (2 x 252)) / years), z = z_0.975 + z_0.80 (E1-45 (b)).
- Deflated Sharpe: Bailey & Lopez de Prado, null variance of the daily Sharpe 1 / (T - 1), skew/kurtosis corrected
  (research 11 formula), N = actual number of tested settings.
- Newey-West t with 5 lags; drawdown on the cumulative dollar sum (no compounding)."""
import math

import numpy as np
from scipy.stats import kurtosis, norm, skew

from config import ANN, B_BOOT, BLOCK, SEED, SR_STAR
from fillmodel import jit

Z_MAIN = norm.ppf(0.975)
Z_POWER = norm.ppf(0.80)


def sharpe(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(ANN)) if sd > 0 else float("nan")


def boot_starts(T, B=B_BOOT, block=BLOCK, seed=SEED):
    """Block start indices (B x ceil(T / block)) for a series of length T."""
    return np.random.default_rng(seed).integers(0, T, size=(B, -(-T // block)))


@jit
def _boot_sharpes(x, starts, block, ann):
    T = x.shape[0]
    B = starts.shape[0]
    out = np.empty(B)
    for b in range(B):
        s = 0.0
        ss = 0.0
        for j in range(T):
            v = x[(starts[b, j // block] + j % block) % T]
            s += v
            ss += v * v
        m = s / T
        var = (ss - T * m * m) / (T - 1)
        out[b] = m / math.sqrt(var) * math.sqrt(ann) if var > 0 else np.nan
    return out


def boot_sharpes(x, starts, block=BLOCK):
    return _boot_sharpes(np.asarray(x, dtype=np.float64), starts, block, float(ANN))


def ci(x, starts=None, level=0.95, block=BLOCK):
    x = np.asarray(x, dtype=np.float64)
    st = boot_starts(len(x), block=block) if starts is None else starts
    sh = boot_sharpes(x, st, block)
    a = (1 - level) / 2
    return float(np.nanpercentile(sh, 100 * a)), float(np.nanpercentile(sh, 100 * (1 - a)))


def mde(n_days, level=0.95):
    z = norm.ppf(1 - (1 - level) / 2) + Z_POWER
    yrs = n_days / ANN
    d = 1 - z * z / (2 * ANN * yrs)
    return float(math.sqrt(z * z / yrs / d)) if d > 0 else float("nan")


def se_daily(sr, n_days):
    return np.sqrt((1 + np.asarray(sr, dtype=np.float64) ** 2 / (2 * ANN)) / (n_days / ANN))


def classify(x, a, b):
    """SPEC 11: 'a' when CI low > 0 and x >= SR*, 'b' when CI high < SR*, else 'c'."""
    if a > 0 and x >= SR_STAR:
        return "a"
    if b < SR_STAR:
        return "b"
    return "c"


def power_table(n_days, srs=(-2, -1, -0.5, 0, 0.5, 1, 2, 3), n=400_000, seed=SEED):
    """P(a), P(b), P(c) under the normal approximation with the daily SE (E1-45 (b))."""
    r = np.random.default_rng(seed)
    out = []
    for sr in srs:
        x = sr + se_daily(sr, n_days) * r.standard_normal(n)
        lo, hi = x - Z_MAIN * se_daily(x, n_days), x + Z_MAIN * se_daily(x, n_days)
        pa = float(np.mean((lo > 0) & (x >= SR_STAR)))
        pb = float(np.mean(hi < SR_STAR))
        out.append((sr, pa, pb, 1 - pa - pb))
    return out


def deflated_sharpe(x, n_trials):
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    sr = x.mean() / x.std(ddof=1)
    g = 0.5772156649
    sr0 = math.sqrt(1 / (T - 1)) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * math.e)))
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    return float(norm.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)))


def newey_west_t(x, lags=5):
    import statsmodels.api as sm
    x = np.asarray(x, dtype=np.float64)
    r = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(r.tvalues[0])


def regress(y, mkt, lags=5):
    """Daily net P&L on the market proxy return: alpha (dollars/day, x 252 annual), beta, Newey-West t."""
    import statsmodels.api as sm
    y, m = np.asarray(y, dtype=np.float64), np.asarray(mkt, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(m)
    r = sm.OLS(y[ok], sm.add_constant(m[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"alpha_day": float(r.params[0]), "alpha_ann": float(r.params[0] * ANN), "beta": float(r.params[1]),
            "t_alpha": float(r.tvalues[0]), "t_beta": float(r.tvalues[1]), "n": int(ok.sum())}


def max_drawdown(x):
    c = np.cumsum(np.asarray(x, dtype=np.float64))
    peak = np.maximum.accumulate(np.concatenate(([0.0], c)))[1:]
    return float(np.max(peak - c)) if len(c) else 0.0


def summary(x):
    """Reported side metrics (SPEC 10) of a daily dollar series."""
    x = np.asarray(x, dtype=np.float64)
    mdd = max_drawdown(x)
    down = x[x < 0]
    gains, losses = x[x > 0].sum(), -x[x < 0].sum()
    dsd = math.sqrt(np.mean(np.minimum(x, 0) ** 2)) if len(x) else float("nan")
    return {"n_days": int(len(x)), "mean_day": float(x.mean()) if len(x) else float("nan"), "sharpe": sharpe(x),
            "nw_t": newey_west_t(x) if len(x) > 6 and x.std() > 0 else float("nan"),
            "max_drawdown": mdd, "calmar": float(x.mean() * ANN / mdd) if mdd > 0 else float("nan"),
            "sortino": float(x.mean() / dsd * math.sqrt(ANN)) if dsd > 0 else float("nan"),
            "skew": float(skew(x)) if len(x) > 2 else float("nan"),
            "kurtosis": float(kurtosis(x, fisher=False)) if len(x) > 3 else float("nan"),
            "worst_day": float(x.min()) if len(x) else float("nan"), "best_day": float(x.max()) if len(x) else float("nan"),
            "win_rate": float(np.mean(x > 0)) if len(x) else float("nan"),
            "avg_win_loss": float(x[x > 0].mean() / -down.mean()) if len(down) and (x > 0).any() else float("nan"),
            "profit_factor": float(gains / losses) if losses > 0 else float("nan"),
            "omega": float(gains / losses) if losses > 0 else float("nan")}
