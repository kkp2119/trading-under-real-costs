#!/usr/bin/env python3
"""Research 30B forecast and costs (SPEC30B 3.2-3.3; DEVIATIONS30B 1).
  inputs (decision bar u, information to the end of u only; per bar arrays, then taken at the event)
    x1  |d| / sigma_d
    x2  leader volume of the last s seconds / (s x the leader's mean volume a second over the 1,800 seconds before them,
        all since 09:30 when fewer), the denominator at least 1 share
    x3  leader spread, bp of the mid        x4  follower spread, bp
    x5  follower size imbalance (bid size - ask size) / (sum) x direction (0 when both are 0)
    x6  follower s-second log return / leader s-second log return, clipped to [-2, 2]; -2 when the leader did not move
    x7  SPY s-second log return x direction, bp (research 21's SPY bars); 0 when SPY has no bars or no valid quote
    x8  follower 1-minute volatility (bp, as research 30); missing -> the fit sample's median
    x9  09:35-10:00 flag, 15:00-on flag     x10 same-company family (D2) flag
  y   direction x (ln m_B,u+120 - ln m_B,u), bp; undefined when bar u + 120 has no valid quote or is past the close
  fit OLS on the 2015-2016 events of each (s, k), once; inputs x1-x4, x6-x8 and y clipped at their 2015-2016 1% / 99%
      points (the bounds are frozen with the coefficients); g_J, g_I on the filled J / I events of each (s, k, E)
  costs  c_m = mean over 2015-2016 events (filled ones for J and I) of (clipped y - pnl), bp of the filled notional, by
         the follower spread bin (1, 2, 3-4, 5-9, 10+ ticks); p_m = share of events with any fill within E, same bins
  rule   V_M = g - c_M, V_J = p_J (g_J - c_J), V_I = p_I (g_I - c_I); enter with the largest when it is > 0"""
import hashlib
import json

import numpy as np

NX = 11
XNAMES = ("x1_absd_over_sigma", "x2_leader_volume_ratio", "x3_leader_spread_bp", "x4_follower_spread_bp", "x5_imbalance_dir",
          "x6_follower_over_leader", "x7_spy_dir_bp", "x8_vol1m_bp", "x9a_0935_1000", "x9b_from_1500", "x10_same_company")
CLIP_X = (0, 1, 2, 3, 5, 6, 7)
SPREAD_BINS = ((1, 1), (2, 2), (3, 4), (5, 9), (10, 10 ** 9))
BAR_1500 = 19_799                                         # the bar ending 15:00:00
Y_H = 120


def rolling_std(x, w):
    """SD (ddof 1) of the finite values of x[u - w + 1 .. u] per bar (NaN with fewer than 2)."""
    ok = np.isfinite(x)
    z = np.where(ok, x, 0.0)
    c0 = np.r_[0, np.cumsum(ok)]
    c1 = np.r_[0.0, np.cumsum(z)]
    c2 = np.r_[0.0, np.cumsum(z * z)]
    N = len(x)
    hi = np.arange(1, N + 1)
    lo = np.maximum(hi - w, 0)
    n = c0[hi] - c0[lo]
    s1, s2 = c1[hi] - c1[lo], c2[hi] - c2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / np.where(n > 0, n, 1)) / np.where(n > 1, n - 1, 1)
    return np.where(n >= 2, np.sqrt(np.maximum(var, 0.0)), np.nan)


def bar_inputs(a, b, spy, s, d, sd, family_d2):
    """Per bar arrays of the inputs that do not depend on the direction, and the two that are multiplied by it (x5, x7)."""
    N = len(d)
    va, la, _ = _q(a)
    vb, lb, m2b = _q(b)
    vol = np.asarray(a["vol"], dtype=np.float64)
    cs = np.r_[0.0, np.cumsum(vol)]
    u = np.arange(N)
    num = cs[u + 1] - cs[np.maximum(u + 1 - s, 0)]
    e = np.maximum(u + 1 - s, 0)                                        # bars before the s-second window: [e - 1800, e)
    st = np.maximum(e - 1800, 0)
    cnt = e - st
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, (cs[e] - cs[st]) / np.where(cnt > 0, cnt, 1), 0.0)
        x2 = num / np.maximum(s * mean, 1.0)
    bid_a, ask_a = np.asarray(a["bid"], dtype=np.float64), np.asarray(a["ask"], dtype=np.float64)
    bid_b, ask_b = np.asarray(b["bid"], dtype=np.float64), np.asarray(b["ask"], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        x3 = np.where(va, 2e4 * (ask_a - bid_a) / (ask_a + bid_a), np.nan)
        x4 = np.where(vb, 2e4 * (ask_b - bid_b) / (ask_b + bid_b), np.nan)
        bs, az = np.asarray(b["bsz"], dtype=np.float64), np.asarray(b["asz"], dtype=np.float64)
        imb = np.where(bs + az > 0, (bs - az) / np.where(bs + az > 0, bs + az, 1), 0.0)
        rl = np.full(N, np.nan)
        rf = np.full(N, np.nan)
        rl[s:] = la[s:] - la[:-s]
        rf[s:] = lb[s:] - lb[:-s]
        x6 = np.where(rl != 0, np.clip(rf / np.where(rl != 0, rl, 1), -2.0, 2.0), -2.0)
        x6 = np.where(np.isfinite(rl) & np.isfinite(rf), x6, np.nan)
        x1 = np.abs(d) / sd
    if spy is not None:
        vs, ls, _ = _q(spy)
        rs = np.full(N, np.nan)
        rs[s:] = (ls[s:] - ls[:-s]) * 1e4
        spy_r = np.where(np.isfinite(rs), rs, 0.0)
    else:
        spy_r = np.zeros(N)
    r1 = np.r_[np.nan, np.diff(lb)]
    x8 = rolling_std(r1, 60) * 1e4
    x9a = (u < 1799).astype(np.float64)
    x9b = (u >= BAR_1500).astype(np.float64)
    x10 = np.full(N, 1.0 if family_d2 else 0.0)
    y = np.full(N, np.nan)
    y[: N - Y_H] = (lb[Y_H:] - lb[: N - Y_H]) * 1e4
    return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "imb": imb, "x6": x6, "spy": spy_r, "x8": x8, "x9a": x9a, "x9b": x9b,
            "x10": x10, "y_up": y, "m2b": m2b}


def _q(x):
    bid, ask = np.asarray(x["bid"], dtype=np.int64), np.asarray(x["ask"], dtype=np.int64)
    valid = (bid > 0) & (ask > bid)
    lm = np.full(len(bid), np.nan)
    lm[valid] = np.log((bid + ask)[valid].astype(np.float64) / 2e4)
    return valid, lm, bid + ask


def event_inputs(BI, u, dirn, close_bar):
    """(x vector of 11, y in bp or NaN) of an event at bar u with its direction."""
    x = np.array([BI["x1"][u], BI["x2"][u], BI["x3"][u], BI["x4"][u], dirn * BI["imb"][u], BI["x6"][u], dirn * BI["spy"][u],
                  BI["x8"][u], BI["x9a"][u], BI["x9b"][u], BI["x10"][u]])
    y = dirn * BI["y_up"][u] if u + Y_H <= close_bar - 1 else np.nan
    return x, y


def spread_bin(ticks):
    t = np.asarray(ticks)
    out = np.full(t.shape, -1, dtype=np.int64)
    for i, (a, b) in enumerate(SPREAD_BINS):
        out[(t >= a) & (t <= b) & (out < 0)] = i
    return out


# ---------------------------------------------------------------- fit and use
def clip_bounds(X, y):
    xb = {}
    for j in CLIP_X:
        v = X[:, j][np.isfinite(X[:, j])]
        xb[j] = [float(np.percentile(v, 1)), float(np.percentile(v, 99))] if len(v) else [-np.inf, np.inf]
    yy = y[np.isfinite(y)]
    yb = [float(np.percentile(yy, 1)), float(np.percentile(yy, 99))] if len(yy) else [-np.inf, np.inf]
    x8med = float(np.nanmedian(X[:, 7])) if np.isfinite(X[:, 7]).any() else 0.0
    return {"x": {str(j): b for j, b in xb.items()}, "y": yb, "x8_median": x8med}


def prep(X, bounds):
    """Inputs as the model sees them: missing x8 -> median, other missing -> 0 (never expected), clipped."""
    Z = np.array(X, dtype=np.float64, copy=True)
    Z[:, 7] = np.where(np.isfinite(Z[:, 7]), Z[:, 7], bounds["x8_median"])
    Z = np.where(np.isfinite(Z), Z, 0.0)
    for j, (lo, hi) in bounds["x"].items():
        Z[:, int(j)] = np.clip(Z[:, int(j)], lo, hi)
    return Z


def ols(X, y, bounds):
    """Coefficients (intercept first) of y clipped on the prepared inputs; rows with y undefined are left out."""
    ok = np.isfinite(y)
    if ok.sum() <= NX + 1:
        return [0.0] * (NX + 1), int(ok.sum())
    Z = np.column_stack([np.ones(int(ok.sum())), prep(X[ok], bounds)])
    yc = np.clip(y[ok], *bounds["y"])
    beta, *_ = np.linalg.lstsq(Z, yc, rcond=None)
    return [float(b) for b in beta], int(ok.sum())


def predict(beta, X, bounds):
    Z = np.column_stack([np.ones(len(X)), prep(X, bounds)])
    return Z @ np.asarray(beta)


def cost_table(yc, pnl_bp, sbin):
    """Mean (clipped y - pnl) by spread bin; NaN for a bin without events."""
    out = []
    for i in range(len(SPREAD_BINS)):
        m = (sbin == i) & np.isfinite(yc) & np.isfinite(pnl_bp)
        out.append(float((yc[m] - pnl_bp[m]).mean()) if m.any() else float("nan"))
    return out


def fill_table(filled, sbin):
    out = []
    for i in range(len(SPREAD_BINS)):
        m = sbin == i
        out.append(float(filled[m].mean()) if m.any() else 0.0)
    return out


def values(model, X, sbin, E):
    """(V_M, V_J, V_I) per event row (bp of notional); a bin without a cost estimate gives -inf for that method."""
    g = predict(model["g"]["beta"], X, model["bounds"])
    out = []
    for m in ("M", "J", "I"):
        if m == "M":
            gm, p = g, np.ones(len(X))
            c = np.asarray(model["cost"]["M"])[np.maximum(sbin, 0)]
        else:
            key = f"{m}{E}"
            gm = predict(model["g_fill"][key]["beta"], X, model["bounds"])
            p = np.asarray(model["p_fill"][key])[np.maximum(sbin, 0)]
            c = np.asarray(model["cost"][key])[np.maximum(sbin, 0)]
        v = p * (gm - c)
        v = np.where(np.isfinite(c) & (sbin >= 0), v, -np.inf)
        out.append(v)
    return g, out[0], out[1], out[2]


def choose(vm, vj, vi):
    """Method index per event (0 M, 1 J, 2 I; -1 no entry): the largest value when it is > 0; ties: M, then J."""
    V = np.column_stack([vm, vj, vi])
    k = np.argmax(V, axis=1)
    best = V[np.arange(len(V)), k]
    return np.where(best > 0, k, -1), best


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
