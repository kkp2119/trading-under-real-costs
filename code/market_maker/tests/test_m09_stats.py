"""M9 statistics (SPEC 10; E1-45 (b); Kyle 2026-10-27: coverage with 1,000 series x 2,000 resamples, the (b) power
table with 2,000 series x 1,000 resamples, seeds fixed and never changed after a failure)."""
import importlib.util
import math
import os

import numpy as np
import pytest
from scipy.stats import kurtosis, norm, skew

import stats

T_HOLD = 1441
TABLE_B = {-2.0: (0.000, 1.000), -1.0: (0.000, 0.948), -0.5: (0.001, 0.666), 0.0: (0.025, 0.223),
           0.5: (0.222, 0.025), 1.0: (0.667, 0.001), 2.0: (0.998, 0.000), 3.0: (1.000, 0.000)}    # E1-45 (b): P(a), P(b)


def series(sr, n, seed):
    """n iid normal daily series of length T_HOLD with annualized Sharpe sr."""
    r = np.random.default_rng(seed)
    return r.standard_normal((n, T_HOLD)) + sr / math.sqrt(252)


def test_sharpe_by_hand_and_nan_rules():
    assert abs(stats.sharpe([1, 2, 3, 4]) - 2.5 / np.std([1, 2, 3, 4], ddof=1) * math.sqrt(252)) < 1e-12
    assert abs(stats.sharpe([1, 2, 3, 4]) - 30.74085) < 1e-4
    assert math.isnan(stats.sharpe([0, 0, 0])) and math.isnan(stats.sharpe([1.0]))


def test_block_bootstrap_layout():
    st = stats.boot_starts(25, B=3, block=10)
    assert st.shape == (3, 3) and st.min() >= 0 and st.max() < 25
    x = np.arange(25, dtype=float)
    sh = stats.boot_sharpes(x, st, 10)
    for b in range(3):                                               # circular blocks of 10, truncated to T
        idx = np.concatenate([(s + np.arange(10)) % 25 for s in st[b]])[:25]
        assert abs(sh[b] - stats.sharpe(x[idx])) < 1e-9


def test_ci_coverage_1000_series_2000_resamples():
    for true in (1.0, 0.0):
        xs = series(true, 1000, seed=12 if true else 13)
        st = stats.boot_starts(T_HOLD, B=2000)
        cover = 0
        for x in xs:
            sh = stats.boot_sharpes(x, st)
            lo, hi = np.percentile(sh, [2.5, 97.5])
            cover += lo <= true <= hi
        assert 0.92 <= cover / 1000 <= 0.98, (true, cover / 1000)


def test_mde_and_power_formulas():
    assert abs(stats.mde(T_HOLD) - 1.173) < 0.0005
    z = norm.ppf(0.975) + norm.ppf(0.8)
    yrs = T_HOLD / 252
    m = stats.mde(T_HOLD)
    assert abs(m - z * math.sqrt((1 + m * m / 504) / yrs)) < 1e-12               # fixed point of the daily SE
    for sr, pa, pb, _ in stats.power_table(T_HOLD):
        assert abs(pa - TABLE_B[sr][0]) < 0.005 and abs(pb - TABLE_B[sr][1]) < 0.005


def test_power_table_reproduced_by_block_bootstrap():
    """2,000 iid normal series x 1,000 block-bootstrap resamples per true Sharpe: P(a), P(b) within +-0.05 of (b)."""
    st = stats.boot_starts(T_HOLD, B=1000)
    for k, (sr, (pa, pb)) in enumerate(TABLE_B.items()):
        xs = series(sr, 2000, seed=100 + k)
        codes = []
        for x in xs:
            sh = stats.boot_sharpes(x, st)
            lo, hi = np.percentile(sh, [2.5, 97.5])
            codes.append(stats.classify(stats.sharpe(x), lo, hi))
        codes = np.array(codes)
        assert abs(np.mean(codes == "a") - pa) <= 0.05 and abs(np.mean(codes == "b") - pb) <= 0.05, (sr, np.mean(codes == "a"), np.mean(codes == "b"))


def test_deflated_sharpe_by_hand_and_matches_research11_formula():
    r = np.random.default_rng(5)
    x = 0.08 + r.standard_t(5, 1441) * 0.9
    T, sr = len(x), x.mean() / x.std(ddof=1)
    g, n = 0.5772156649, 69
    sr0 = math.sqrt(1 / (T - 1)) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * math.e)))
    hand = norm.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(1 - skew(x) * sr + (kurtosis(x, fisher=False) - 1) / 4 * sr ** 2))
    assert abs(stats.deflated_sharpe(x, 69) - hand) < 1e-12
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "research11", "stats.py")
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("r11stats", path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert abs(m.deflated_sharpe(x, 69) - stats.deflated_sharpe(x, 69)) < 1e-12
    assert stats.deflated_sharpe(x, 2) > stats.deflated_sharpe(x, 69) > stats.deflated_sharpe(x, 1000)
