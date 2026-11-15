#!/usr/bin/env python3
"""Research 25 signals and the values handed to the engine (SPEC25 3-4, SPEC25_ANSWERS, SPEC25_QUESTIONS 10-21; readings
in DEVIATIONS25.md 1). Everything is on the decision grid u = start + 5k (k = 0..59) of every parent order and uses
bar-end values up to u only.
  window   the 13 bars u-60, u-55, ..., u (12 intervals of 5 seconds); for k < 12 the first bars are before the order start
  mid      valid as in engine18.mids (bid > 0, ask > 0, ask >= bid); one invalid mid among the 13 -> no signal (nan)
  RV60     sqrt(sum of the 12 squared mid changes), cents
  q        RV60 / (ask - bid at u, cents, not capped); no signal when the spread is 0
  g        sqrt(sum of the 12 squared log mid returns) x 10,000 (bp) / vol_pre (engine18.vol_pre of the order, bp per
           minute); no signal for the whole order when vol_pre is undefined or 0
  v        research 23's trade intensity (feat23.trade_inputs, frozen file): shares traded in the bars u-59..u over the
           usual 60-second volume, cut to [0, 10]; no signal on a day without a usual volume
The 14 candidates (SPEC25 4, grid order = QUESTIONS 18) and what the engine gets (QUESTIONS 16; engine18.run is not changed):
  family A  1.0 (early fill, engine run R1_0.75) or 0.0 (hold, R2_0.25) where g >= theta, 0.5 elsewhere
  family B  p (research 23's frozen model) where q >= kappa, 0.5 elsewhere; engine runs R1_0.60 / R1_0.70 (B1), R2_0.40
            (B2), R3_0.60_0.40 (B3)
  family C  p where v >= 2, 0.5 elsewhere; engine runs R1_0.60 / R1_0.70
0.5 meets no threshold of these engine runs, and neither does a missing p (replaced by 0.5), so a point without a signal
or without p behaves as E*. The control (SPEC25 5, QUESTIONS 21) turns the volatility / volume condition around (x < the
threshold, on the points where the signal exists) and keeps the rest. "off" drops the condition altogether: used only for
the regression of ANSWERS A3 (kappa = 0 read as "no volatility condition"). No file is written here."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "study23_combined_forecast"))
import feat23 as F  # noqa: E402  (research 23, frozen: brings engine18 with research 22's checks)

E = F.E
WIN, GSTEP, NPTS = 60, 5, 13                      # bars u-60..u at 5-second spacing: 13 mids, 12 changes
THETAS, KAPPAS, P_HIS, P_LO, V_MIN = (2.0, 3.0), (0.5, 1.0), (0.60, 0.70), 0.40, 2.0
NEUTRAL, EARLY, HOLD = 0.5, 1.0, 0.0


def _g(x):
    return f"{x:g}"


def _cand(family, rule, sig, thr, engine, flag=None):
    return {"family": family, "rule": rule, "signal": sig, "threshold": thr, "engine": engine, "flag": flag}


SPECS = {}
for _t in THETAS:
    SPECS[f"A1_g{_g(_t)}"] = _cand("A", "A1", "g", _t, "R1_0.75", EARLY)
for _t in THETAS:
    SPECS[f"A2_g{_g(_t)}"] = _cand("A", "A2", "g", _t, "R2_0.25", HOLD)
for _h in P_HIS:
    for _k in KAPPAS:
        SPECS[f"B1_p{round(100 * _h)}_k{_g(_k)}"] = _cand("B", "B1", "q", _k, f"R1_{_h:.2f}")
for _k in KAPPAS:
    SPECS[f"B2_k{_g(_k)}"] = _cand("B", "B2", "q", _k, f"R2_{P_LO:.2f}")
for _k in KAPPAS:
    SPECS[f"B3_k{_g(_k)}"] = _cand("B", "B3", "q", _k, f"R3_{P_HIS[0]:.2f}_{P_LO:.2f}")
for _h in P_HIS:
    SPECS[f"C1_p{round(100 * _h)}"] = _cand("C", "C1", "v", V_MIN, f"R1_{_h:.2f}")
CANDS = tuple(SPECS)                              # SPEC25 4, table order: A 4, B 8, C 2
EARLY_FILL = tuple(c for c in CANDS if SPECS[c]["rule"] in ("A1", "B1", "C1"))        # the 8 early-fill candidates of SPEC25 8
# ANSWERS A3: the regression runs with kappa = 0 read as "no volatility condition", and research 23's run each must equal
K0 = {"k0:B1_p60": ("B1_p60_k0.5", "R1_0.60"), "k0:B2": ("B2_k0.5", "R2_0.40"), "k0:B3": ("B3_k0.5", "R3_0.60_0.40")}


def grid(starts):
    return np.asarray(starts, dtype=np.int64)[:, None] + E.STEP * np.arange(E.NG, dtype=np.int64)[None, :]


def recent(b, starts):
    """RV60 (cents), the same in log returns (bp), the spread at u (cents) and q at every decision bar; nan = no signal.
    full: the 13 mids of the window are valid (and the window does not start before the first bar)."""
    U = grid(starts)
    W = U[:, :, None] + (-GSTEP * np.arange(NPTS - 1, -1, -1, dtype=np.int64))[None, None, :]      # u-60, u-55, ..., u
    inside = W[:, :, 0] >= 0
    Wc = np.clip(W, 0, None)
    bid, ask = np.asarray(b["bid"], dtype=np.int64), np.asarray(b["ask"], dtype=np.int64)
    ok = (bid > 0) & (ask > 0) & (ask >= bid)                                # engine18.mids
    full = inside & ok[Wc].all(axis=2)
    m2 = (bid + ask)[Wc].astype(np.float64)                                   # 2 x mid, 1e-4 dollars
    rv = np.sqrt((np.diff(m2, axis=2) ** 2).sum(axis=2)) / (2 * E.TICK)       # cents (1 tick = 1 cent = 100 x 1e-4 dollars)
    pos = np.where(m2 > 0, m2, 1.0)
    rv_bp = 1e4 * np.sqrt((np.log(pos[:, :, 1:] / pos[:, :, :-1]) ** 2).sum(axis=2))
    spread_c = (ask[U] - bid[U]) / float(E.TICK)
    rv, rv_bp = np.where(full, rv, np.nan), np.where(full, rv_bp, np.nan)
    q = np.where(full & (spread_c > 0), rv / np.where(spread_c > 0, spread_c, 1.0), np.nan)
    return {"rv": rv, "rv_bp": rv_bp, "spread_c": spread_c, "q": q, "full": full}


def signals(b, dirs, starts, vp, adv):
    """{rv, q, g, v [n, NG] (nan = no signal), full, spread_c, vp_ok [n]}. vp: engine18's vol_pre of every parent (nan =
    undefined); adv: the usual daily volume of the symbol-day (nan / None / 0 = none)."""
    r = recent(b, starts)
    vp = np.asarray(vp, dtype=np.float64)
    vp_ok = np.isfinite(vp) & (vp > 0)
    g = np.where(r["full"] & vp_ok[:, None], r["rv_bp"] / np.where(vp_ok, vp, 1.0)[:, None], np.nan)
    v = F.trade_inputs(b, dirs, starts, adv)[1]
    return {"rv": r["rv"], "q": r["q"], "g": g, "v": v, "full": r["full"], "spread_c": r["spread_c"], "vp_ok": vp_ok}


def condition(x, thr, flip=False):
    """The volatility / volume condition where the signal exists: x >= thr, or x < thr for the control. No signal: False."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    safe = np.where(ok, x, thr)
    return ok & ((safe < thr) if flip else (safe >= thr))


def engine_p(cand, sig, p=None, flip=False, off=False):
    """What engine18.run gets for a candidate: family A 1.0 / 0.0 / 0.5; families B and C p or 0.5 (never nan)."""
    s = SPECS[cand]
    x = sig[s["signal"]]
    cond = np.ones(np.shape(x), dtype=bool) if off else condition(x, s["threshold"], flip)
    if s["family"] == "A":
        if off:
            raise ValueError("family A has no run without its condition")
        return np.where(cond, s["flag"], NEUTRAL)
    if p is None:
        raise ValueError(f"{cand}: no model for its probability")
    return np.where(cond & np.isfinite(p), p, NEUTRAL)


def abs_move60(b, starts):
    """Diagnostic only (SPEC25 6): |mid(u + 60) - mid(u)| in cents; nan when one of the two mids is not valid."""
    U = grid(starts)
    bid, ask = np.asarray(b["bid"], dtype=np.int64), np.asarray(b["ask"], dtype=np.int64)
    ok = (bid > 0) & (ask > 0) & (ask >= bid)
    m2 = bid + ask
    V = np.minimum(U + WIN, len(bid) - 1)
    both = ok[U] & ok[V] & (U + WIN < len(bid))
    return np.where(both, np.abs(m2[V] - m2[U]) / (2.0 * E.TICK), np.nan)
