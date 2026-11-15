#!/usr/bin/env python3
"""Research 20 signal (SPEC20 2, SPEC20_ANSWERS A2, A4; readings fixed before any result in DEVIATIONS20.md 1).

Grid       decision times 09:30:05, :10, ... = bar ends g = 4, 9, 14, ... (bar u ends at 09:30:00 + u + 1 s; engine17's
           start bar 299 = 09:35:00 uses the same convention), up to the close.
Valid      both legs: bid > 0, ask > 0, ask > bid at the bar end (a locked quote is not valid, A2).
s          ln(mid A) - ln(mid B) at valid grid points.
m, sigma   mean and sample standard deviation (ddof 1) of the W x 12 most recent valid samples strictly before t (t itself
           excluded); with no invalid sample these are exactly the previous W minutes. No signal until W x 12 valid samples
           exist. sigma is floored at 1 bp.
z          (s - m) / sigma.
Entry      |z| >= z_in and |s - m| >= kappa x (relative spread A + relative spread B), inside the entry hours
           max(09:35, 09:30 + W min) .. close - 30 min (15:30 on a full day), ends included. z > 0: sell A, buy B.
Exit       |z| <= 0.5 (needs a signal), 30 minutes after the entry decision, or close - 5 min (15:55), whichever is first;
           no decision at a grid point without valid quotes.
Size       per leg 50,000 dollars / mid at the entry decision, rounded to 10 shares (half up), at least 10.
mid_trades the signal-only timetable for H1 (A4): both legs trade at the mid at the decision times, one position at a time,
           no fill engine.
Quote gaps (SPEC20_ADDENDUM B1): no entry and no exit decision at a grid point without valid quotes; the 30-minute and the
           15:55 exits are taken at the first valid grid point after them; a position that meets no valid grid point
           from 15:55 to the close is valued at the mids of the last valid grid point (reason R_LAST, counted).
Delayed    mid_pnl_delayed (SPEC20_ADDENDUM B2, no verdict): the same trades and decision times, prices taken d seconds
           after each decision (the first bar from there on with valid quotes on both legs)."""
import numpy as np

STEP, G0 = 5, 4                       # grid bars 4, 9, 14, ...
SIGMA_FLOOR = 1e-4                    # 1 bp
EXIT_Z = 0.5
HOLD_MAX = 1800                       # seconds after the entry decision
ENTRY_FIRST = 299                     # 09:35:00
ENTRY_BEFORE_CLOSE = 1800             # entries up to close - 30 min
FORCE_BEFORE_CLOSE = 300              # forced exit at close - 5 min
NOTIONAL, LOT = 50_000, 10
WS, ZINS, KAPPAS = (10, 30), (2.0, 3.0), (0.0, 1.0)
R_Z, R_TIME, R_CLOSE, R_LAST = 1, 2, 3, 4          # exit reasons


def grid(close_bar):
    return np.arange(G0, close_bar, STEP, dtype=np.int64)


def entry_bars(W, close_bar):
    """(first, last) decision bar of the entry hours, both included."""
    return max(ENTRY_FIRST, W * 60 - 1), close_bar - ENTRY_BEFORE_CLOSE - 1


def force_bar(close_bar):
    return close_bar - FORCE_BEFORE_CLOSE - 1


def shares(mid2):
    """50,000 dollars / mid in lots of 10 shares, half up, at least one lot; mid2 = bid + ask in 1e-4 dollars (integers)."""
    mid2 = np.asarray(mid2, dtype=np.int64)
    lots = (2 * 100_000_000 + mid2) // (2 * mid2)                 # floor(50,000 / (mid2 / 20,000) / 10 + 1/2)
    return np.maximum(lots, 1) * LOT


def base(a, b, close_bar):
    """Grid arrays of one pair-day from the two bar dicts: bars g, valid, s (nan when not valid), mid2 and relative spreads."""
    g = grid(close_bar)
    ba, aa, bb, ab = a["bid"][g], a["ask"][g], b["bid"][g], b["ask"][g]
    valid = (ba > 0) & (aa > 0) & (aa > ba) & (bb > 0) & (ab > 0) & (ab > bb)
    m2a, m2b = ba + aa, bb + ab
    s = np.full(len(g), np.nan)
    s[valid] = np.log(m2a[valid].astype(np.float64)) - np.log(m2b[valid].astype(np.float64))
    rs = np.zeros(len(g))
    rs[valid] = 2.0 * (aa - ba)[valid] / m2a[valid] + 2.0 * (ab - bb)[valid] / m2b[valid]
    return {"g": g, "valid": valid, "s": s, "mid2_a": m2a, "mid2_b": m2b, "rs": rs}


def zscore(s, valid, W):
    """(z, dev = s - m, has): rolling statistics over the n = W x 12 most recent valid samples before each valid point."""
    n = W * 12
    K = len(s)
    z, dev, has = np.full(K, np.nan), np.full(K, np.nan), np.zeros(K, bool)
    iv = np.flatnonzero(valid)
    if len(iv) > n:
        x = s[iv]
        win = np.lib.stride_tricks.sliding_window_view(x, n)[:-1]          # row j = x[j : j + n], used by dense index j + n
        m = win.mean(axis=1)
        sd = np.maximum(win.std(axis=1, ddof=1), SIGMA_FLOOR)
        k = iv[n:]
        dev[k] = x[n:] - m
        z[k] = dev[k] / sd
        has[k] = True
    return z, dev, has


def decide(z, dev, rs, has, z_in, kappa):
    """(entry side: +1 when z >= z_in (sell A, buy B), -1 when z <= -z_in, 0 otherwise; exit-by-z flag), no hours applied."""
    az = np.where(has, np.abs(z), 0.0)
    ent = has & (az >= z_in) & (np.where(has, np.abs(dev), 0.0) >= kappa * rs)
    side = np.where(ent, np.where(np.where(has, z, 0.0) > 0, 1, -1), 0).astype(np.int8)
    return side, has & (az <= EXIT_Z)


def flags(B, W, z_in, kappa, close_bar):
    """Entry side per grid point inside the entry hours and the exit-by-z flag for one configuration."""
    z, dev, has = zscore(B["s"], B["valid"], W)
    side, exz = decide(z, dev, B["rs"], has, z_in, kappa)
    lo, hi = entry_bars(W, close_bar)
    side = np.where((B["g"] >= lo) & (B["g"] <= hi), side, 0).astype(np.int8)
    return side, exz


def mid_trades(B, side, exz, close_bar):
    """Signal-only trades: rows (entry bar, exit bar, side, shares A, shares B, reason, mid2 A entry, mid2 B entry,
    mid2 A exit, mid2 B exit), int64. Entry at a flagged grid point when flat; the exit is looked for from the next grid
    point on, at valid points only. What is still open at the last grid point of the day without a valid quote there is
    valued at the mids of the last valid grid point (ADDENDUM B1)."""
    g, valid = B["g"], B["valid"]
    F, rows, K = force_bar(close_bar), [], len(g)
    k = 0
    while k < K:
        if side[k] == 0:
            k += 1
            continue
        ke, sd = k, int(side[k])
        j = ke + 1
        while j < K:
            last = j == K - 1
            if valid[j] or last:
                forced, timed, zx = g[j] >= F, g[j] - g[ke] >= HOLD_MAX, bool(exz[j])
                if last and not valid[j]:
                    reason = R_LAST
                elif forced:
                    reason = R_CLOSE
                elif zx:
                    reason = R_Z
                elif timed:
                    reason = R_TIME
                else:
                    reason = 0
                if reason:
                    break
            j += 1
        if j >= K:                                                    # no grid point after the entry (cannot happen in the entry hours)
            raise ValueError("an entry without a later grid point")
        qa, qb = int(shares(B["mid2_a"][ke])), int(shares(B["mid2_b"][ke]))
        jx = j if valid[j] else int(np.flatnonzero(valid[:j + 1])[-1])   # ADDENDUM B1: the last valid grid point (>= the entry)
        rows.append((g[ke], g[j], sd, qa, qb, reason, B["mid2_a"][ke], B["mid2_b"][ke], B["mid2_a"][jx], B["mid2_b"][jx]))
        k = j + 1
    return np.array(rows, dtype=np.int64).reshape(-1, 10)


def mid_pnl(tr):
    """Dollars per signal-only trade: side x (shares A x (mid A entry - mid A exit) + shares B x (mid B exit - mid B entry))."""
    tr = np.asarray(tr, dtype=np.float64).reshape(-1, 10)
    return tr[:, 2] * (tr[:, 3] * (tr[:, 6] - tr[:, 8]) + tr[:, 4] * (tr[:, 9] - tr[:, 7])) / 20_000.0


# ---------------------------------------------------------------- delayed mid prices (SPEC20_ADDENDUM B2)
MID_DELAYS = (1, 5, 30)                # seconds


def bar_base(a, b, close_bar):
    """Per-second arrays of the session: valid quotes on both legs, 2 x mid of leg A and of leg B (bar ends)."""
    n = close_bar
    ba, aa, bb, ab = a["bid"][:n], a["ask"][:n], b["bid"][:n], b["ask"][:n]
    return (ba > 0) & (aa > 0) & (aa > ba) & (bb > 0) & (ab > 0) & (ab > bb), ba + aa, bb + ab


def delayed_bars(v, bars, d):
    """The bar that prices a decision taken at `bars` when trading d seconds later: the first bar >= bars + d of the
    session with valid quotes on both legs; when none is left, the last valid bar of the session."""
    n = len(v)
    idx = np.where(v, np.arange(n), n)
    nxt = np.minimum.accumulate(idx[::-1])[::-1]                    # first valid bar >= u, n when there is none
    t = np.asarray(bars, dtype=np.int64) + d
    r = np.where(t < n, nxt[np.minimum(t, n - 1)], n)
    return np.where(r >= n, int(np.flatnonzero(v)[-1]), r)


def mid_pnl_delayed(mt, v, m2a, m2b, d):
    """Dollars per signal-only trade with the entry and the exit priced d seconds after their decisions (ADDENDUM B2);
    the decisions, sides and sizes are those of mid_trades. A trade valued at the last valid mids (R_LAST) keeps that exit
    value for every d. d = 0 gives mid_pnl."""
    mt = np.asarray(mt, dtype=np.int64).reshape(-1, 10)
    if not len(mt):
        return np.zeros(0)
    eb, xb = delayed_bars(v, mt[:, 0], d), delayed_bars(v, mt[:, 1], d)
    last = mt[:, 5] == R_LAST
    xa = np.where(last, mt[:, 8], m2a[xb]).astype(np.float64)
    xbv = np.where(last, mt[:, 9], m2b[xb]).astype(np.float64)
    return mt[:, 2] * (mt[:, 3] * (m2a[eb].astype(np.float64) - xa) + mt[:, 4] * (xbv - m2b[eb].astype(np.float64))) / 20_000.0


def delay_sentence(ci_lo, ratio):
    """ADDENDUM B2, fixed in advance (5-second delay): lower end of the 95% interval of the mean daily sum, ratio to delay 0."""
    if ci_lo > 0:
        return f"5초 늦게 거래해도 되돌림의 {100 * ratio:.0f}%가 남았다"
    return "5초 늦게 거래하면 되돌림은 0과 구분되지 않았다"
