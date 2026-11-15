#!/usr/bin/env python3
"""Research 23 model inputs (SPEC23 3 with SPEC23_ANSWERS A3; readings in DEVIATIONS23.md 1). Nine inputs at the decision
bars u = start + 5k (k = 0..59) of every parent order, from bar-end values up to u only:
  1-4  research 18's inputs, taken from engine18.features / design unchanged: signed size imbalance, spread in ticks
       (capped at 10), vol_pre at arrival, signed 2 x mid change over the last 5 bars in ticks
  5-6  research 22's adverse score a_t on the SPY basis and on the own-price basis (signal22: -z for a buy, +z for a sell),
       cut to [-4, 4]; 0 where there is no signal (window not full, quote not valid for the signal, no SPY file)
  7    trade imbalance over the bars u-29..u: (vol_ask - vol_bid) / (vol_ask + vol_bid) x direction (buy +1, sell -1), so
       that more buyer-initiated volume is positive for a buy order; 0 when the denominator is 0
  8    trade intensity over the bars u-59..u: traded shares / (mean daily volume of the previous 20 trading days of the
       year / 390), cut to [0, 10]; undefined (nan) when that mean is undefined or 0, filled later with the training median
  9    input 7 x input 8 (after the fill of input 8)
Bars before 09:30:00 do not exist, so a window that starts before bar 0 uses the bars there are. The target y and the
"valid" flag (decision quote valid) are engine18's. No file is written here."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "study22_market_signal"))
import core22 as C22  # noqa: E402  (research 22, frozen: brings engine18, loader22, signal22 with their checks)

E, SG = C22.E, C22.SG
FEATS = tuple(E.FEATS) + ("a_spy", "a_own", "trade_imbalance_30s", "trade_intensity_60s", "imbalance_x_intensity")
I_VP, I_IMB, I_INT, I_PROD = 2, 6, 7, 8
A_CLIP, W_IMB, W_INT, INT_CLIP, MINUTES = 4.0, 30, 60, 10.0, 390


def adverse_inputs(b, ref, close_bar, dirs, starts):
    """([a_spy, a_own] cut to +-4 with 0 for no signal, [signal present])."""
    vals, has = [], []
    for basis in SG.BASES:
        g, z, h, _ = SG.z_grid(basis, b, ref, close_bar)
        _, zd = SG.at_decisions(g, z, h, starts)
        a = SG.adverse(zd, dirs)
        has.append(~np.isnan(a))
        vals.append(np.where(np.isnan(a), 0.0, np.clip(a, -A_CLIP, A_CLIP)))
    return vals, has


def trade_inputs(b, dirs, starts, adv):
    """(trade imbalance [n, NG], trade intensity [n, NG] or nan). adv: mean daily volume of the previous days, or nan."""
    U = np.asarray(starts, dtype=np.int64)[:, None] + E.STEP * np.arange(E.NG, dtype=np.int64)[None, :]

    def csum(x):
        return np.concatenate([[0], np.cumsum(np.asarray(x, dtype=np.int64))])
    cb, ca, cv = csum(b["vol_bid"]), csum(b["vol_ask"]), csum(b["vol"])
    lo = np.maximum(U - (W_IMB - 1), 0)
    buy, sell = ca[U + 1] - ca[lo], cb[U + 1] - cb[lo]
    den = buy + sell
    imb = np.where(den > 0, (buy - sell) / np.where(den > 0, den, 1), 0.0) * np.asarray(dirs, dtype=np.float64)[:, None]
    lo = np.maximum(U - (W_INT - 1), 0)
    v = (cv[U + 1] - cv[lo]).astype(np.float64)
    if adv is None or not np.isfinite(adv) or adv <= 0:
        return imb, np.full(U.shape, np.nan)
    return imb, np.clip(v / (adv / MINUTES), 0.0, INT_CLIP)


def raw(b, ref, close_bar, dirs, starts, adv):
    """Inputs before the fills: X [n, NG, 9] (nan: vol_pre and intensity where undefined, and the product), engine18's
    feature dict f (y, valid, vp), the signal flags and counts."""
    f = E.features(b, dirs, starts)
    (a_spy, a_own), (h_spy, h_own) = adverse_inputs(b, ref, close_bar, dirs, starts)
    imb, inten = trade_inputs(b, dirs, starts, adv)
    X = np.concatenate([E.design(f, np.nan), a_spy[..., None], a_own[..., None], imb[..., None], inten[..., None],
                        np.full(imb.shape + (1,), np.nan)], axis=-1)
    counts = {**f["counts"], "signal_spy": int(h_spy.sum()), "signal_own": int(h_own.sum()), "intensity_undefined": int(np.isnan(inten).sum())}
    return {"f": f, "X": X, "y": f["y"], "valid": f["valid"], "vp": f["vp"], "counts": counts}


def design(X, fills):
    """The nine inputs with the two fills applied (vol_pre, intensity) and the product formed."""
    Z = np.array(X, dtype=np.float64, copy=True)
    Z[..., I_VP] = np.where(np.isnan(Z[..., I_VP]), fills["vol_pre"], Z[..., I_VP])
    Z[..., I_INT] = np.where(np.isnan(Z[..., I_INT]), fills["intensity"], Z[..., I_INT])
    Z[..., I_PROD] = Z[..., I_IMB] * Z[..., I_INT]
    return Z
