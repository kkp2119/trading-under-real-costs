"""Causal per-bar signals and day statistics for one symbol-day (SPEC 3, 4; E1-04, E1-12/13, E1-16/17, E1-47, E1-53).
Every array value at bar u uses bars <= u only (M1). Thresholds are integer cross-multiplications, mids are 2x (mid2 =
bid + ask), so results are exact and identical under numba and plain Python."""
import numpy as np

from config import (FI_DEN, FI_MIN, FI_NUM, FI_WIN, GCNT_WIN, HALT_K2, HALT_LAG, HALT_MIN2, HALT_WIN, MAX_SPREAD, OBI_DEN,
                    OBI_NUM, STALE_QUOTE_WIN)

FEATURES = ("mid2", "mid_ok", "qbase", "obi_bb", "obi_bs", "fi_bb", "fi_bs", "gcnt", "halt_trig", "stale_q")


def rolling_sum(x, w):
    """s[u] = sum of x[max(0, u - w + 1) .. u]."""
    c = np.concatenate(([0], np.cumsum(x, dtype=np.int64)))
    u = np.arange(len(x))
    return c[u + 1] - c[np.maximum(0, u - w + 1)]


def halt_trigger(mid2, mid_ok):
    """E1-13: at u >= HALT_WIN, valid 1-second mid changes over [u - 1799, u] give n and S2 = sum of squares (2x units);
    trigger when mid2_u and mid2_{u-15} are valid, |D15| >= 1 tick (200 in 2x units) and D15^2 * n >= 135 * S2."""
    n = len(mid2)
    dm = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=bool)
    dm[1:] = mid2[1:] - mid2[:-1]
    ok[1:] = mid_ok[1:] & mid_ok[:-1]
    cn = np.concatenate(([0], np.cumsum(ok, dtype=np.int64)))
    cs = np.concatenate(([0], np.cumsum(np.where(ok, dm * dm, 0), dtype=np.int64)))
    trig = np.zeros(n, dtype=bool)
    u = np.arange(HALT_WIN, n)
    if len(u) == 0:
        return trig
    cnt = cn[u + 1] - cn[u + 1 - HALT_WIN]
    s2 = cs[u + 1] - cs[u + 1 - HALT_WIN]
    d15 = mid2[u] - mid2[u - HALT_LAG]
    trig[u] = mid_ok[u] & mid_ok[u - HALT_LAG] & (np.abs(d15) >= HALT_MIN2) & (cnt > 0) & (d15 * d15 * cnt >= HALT_K2 * s2)
    return trig


def compute(b):
    """b: dict of the 21 bar columns (int64). Returns the FEATURES arrays."""
    bid, ask, bsz, asz = b["bid"], b["ask"], b["bsz"], b["asz"]
    mid_ok = (bid > 0) & (ask > 0) & (ask >= bid)
    mid2 = (bid + ask).astype(np.int64)
    qbase = (bid > 0) & (ask > bid) & (ask - bid <= MAX_SPREAD) & (bsz > 0) & (asz > 0)
    s, d = bsz + asz, bsz - asz
    obi_bb = (s > 0) & (OBI_DEN * d <= -OBI_NUM * s)                 # OBI <= -0.6: no buy quote
    obi_bs = (s > 0) & (OBI_DEN * d >= OBI_NUM * s)                  # OBI >= +0.6: no sell quote
    fa = rolling_sum(b["vol_ask"] - b["vol_ask_odd"], FI_WIN)
    fb = rolling_sum(b["vol_bid"] - b["vol_bid_odd"], FI_WIN)
    fdef = fa + fb >= FI_MIN
    fi_bs = fdef & (FI_DEN * (fa - fb) >= FI_NUM * (fa + fb))       # FI >= +0.6 (buying pressure): no sell quote
    fi_bb = fdef & (FI_DEN * (fa - fb) <= -FI_NUM * (fa + fb))      # FI <= -0.6: no buy quote
    return {"mid2": mid2, "mid_ok": mid_ok, "qbase": qbase, "obi_bb": obi_bb, "obi_bs": obi_bs, "fi_bb": fi_bb, "fi_bs": fi_bs,
            "gcnt": rolling_sum(b["ntr"], GCNT_WIN), "halt_trig": halt_trigger(mid2, mid_ok),
            "stale_q": rolling_sum(b["n_upd"], STALE_QUOTE_WIN) == 0}


def day_stats(b, close_bar):
    """Day-level numbers over the regular session [0, close) only (E1-01, E1-21, E1-41, E1-42): dollar volume (1e-4 $),
    first and last valid mid2, market volume, last trade bar, 1-minute realized volatility, day range ratio."""
    c = close_bar
    vol, px = b["vol"][:c], b["px_last"][:c]
    bid, ask = b["bid"][:c], b["ask"][:c]
    ok = (bid > 0) & (ask > 0) & (ask >= bid)
    mid2 = bid + ask
    idx = np.flatnonzero(ok)
    trades = np.flatnonzero(b["ntr"][:c] > 0)
    out = {"dv": int(np.sum(vol * px)), "mkt_vol": int(np.sum(vol)), "last_trade_bar": int(trades[-1]) if len(trades) else -1,
           "first_mid2": int(mid2[idx[0]]) if len(idx) else 0, "last_mid2": int(mid2[idx[-1]]) if len(idx) else 0,
           "rv_1m": float("nan"), "range_ratio": float("nan"), "no_data": bool(len(idx) == 0 and len(trades) == 0)}
    if len(idx):
        m = np.arange(59, c, 60)
        mm = m[ok[m]]
        if len(mm) > 2:
            r = np.diff(np.log(mid2[mm].astype(float)))
            out["rv_1m"] = float(np.std(r, ddof=1))
        out["range_ratio"] = float((mid2[idx].max() - mid2[idx].min()) / mid2[idx[0]])
    return out
