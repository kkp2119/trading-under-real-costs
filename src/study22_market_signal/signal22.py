#!/usr/bin/env python3
"""Research 22 signal (SPEC22 3; SPEC22_QUESTIONS 10-14 as approved by SPEC22_ANSWERS A8; readings in DEVIATIONS22.md 1).
Research 20's signal code is used as it is (signal20.grid, base, zscore: 5-second grid of bar ends g = 4, 9, 14, ...,
valid = bid > 0, ask > 0, ask > bid, m and sigma over the 360 most recent valid samples strictly before t, sigma floored at
1 bp):
  spy   s = ln(mid stock) - ln(mid SPY)   signal20.base(stock, SPY) -> zscore; both legs must be valid; no SPY file -> no signal
  own   s = ln(bid + ask) of the stock    zscore on the one-leg series (ln 2 cancels when m is subtracted)
Decision bars of a parent order are u = s + 5k, k = 0..59 (engine18); every start bar is on the grid (299, 304, ...), so
z(u) is the grid value of bar u: bar-end values up to u only.
  a      the adverse score: -z for a buy order (dir +1), +z for a sell order (dir -1); nan when there is no signal
  flags  what goes into engine18's P: 1.0 when a >= z_th, 0.0 when a <= -z_th, 0.5 in between, nan without a signal."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "study20_pairs"))
import signal20 as S20  # noqa: E402

W = 30                                   # minutes: 360 valid samples
BASES = ("spy", "own")
ZTH = (1.0, 2.0)
NG, STEP = 60, 5                         # engine18's decision grid of a parent order
FIRE, HOLD, NONE = 1.0, 0.0, 0.5


def series(basis, stock, ref, close_bar):
    """(g, s, valid) on signal20's grid; basis 'spy' without a reference file gives no valid point."""
    g = S20.grid(close_bar)
    if basis == "spy":
        if ref is None:
            return g, np.full(len(g), np.nan), np.zeros(len(g), bool)
        B = S20.base(stock, ref, close_bar)
        return g, B["s"], B["valid"]
    if basis != "own":
        raise KeyError(basis)
    bid, ask = stock["bid"][g], stock["ask"][g]
    valid = (bid > 0) & (ask > 0) & (ask > bid)
    s = np.full(len(g), np.nan)
    s[valid] = np.log((bid + ask)[valid].astype(np.float64))
    return g, s, valid


def floored(s, valid):
    """Diagnostic only: grid points whose sample standard deviation (before the floor) is below 1 bp; signal20.zscore's
    window, recomputed."""
    n = W * 12
    out = np.zeros(len(s), bool)
    iv = np.flatnonzero(valid)
    if len(iv) > n:
        win = np.lib.stride_tricks.sliding_window_view(s[iv], n)[:-1]
        out[iv[n:]] = win.std(axis=1, ddof=1) < S20.SIGMA_FLOOR
    return out


def z_grid(basis, stock, ref, close_bar):
    """(g, z, has, floor flag) on the grid."""
    g, s, valid = series(basis, stock, ref, close_bar)
    z, _, has = S20.zscore(s, valid, W)
    return g, z, has, floored(s, valid)


def at_decisions(g, x, has, starts):
    """(U, x at the decision bars U = start + 5k as float, nan where there is no signal)."""
    U = np.asarray(starts, dtype=np.int64)[:, None] + STEP * np.arange(NG, dtype=np.int64)[None, :]
    if np.any((U - S20.G0) % S20.STEP != 0):
        raise ValueError("a decision bar is not on the signal grid")
    k = (U - S20.G0) // S20.STEP
    inside = k < len(g)
    kk = np.minimum(k, len(g) - 1)
    return U, np.where(inside & has[kk], x[kk].astype(np.float64), np.nan)


def adverse(z_dec, dirs):
    """a = -z for a buy (dir +1), +z for a sell (dir -1)."""
    return -np.asarray(dirs, dtype=np.float64)[:, None] * z_dec


def flags(a, zth):
    """engine18's P from the adverse score: 1.0 (a >= z_th), 0.0 (a <= -z_th), 0.5 otherwise, nan without a signal."""
    p = np.full(a.shape, np.nan)
    ok = ~np.isnan(a)
    p[ok] = NONE
    p[ok & (a >= zth)] = FIRE
    p[ok & (a <= -zth)] = HOLD
    return p
