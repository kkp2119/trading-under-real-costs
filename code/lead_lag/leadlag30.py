#!/usr/bin/env python3
"""Research 30: lead-lag events and the follower-only trade (SPEC30 2-3 with SPEC30_ADDENDUM A, which takes precedence;
DEVIATIONS30 1). Research 17, 20, 21 and 24 code is imported unchanged (core24 checks the frozen files at import).
  pair        research 24's pairs (universe24_<Y>.tsv); leg A = the larger median daily dollar volume of the year before
              (universe24.py) = the leader, leg B = the follower
  d           d_u = (ln m_A,u - ln m_A,u-s) - (ln m_B,u - ln m_B,u-s), mids at the end of second u; defined when both legs
              have a valid quote (bid > 0, ask > bid) at u and u - s
  sigma_d     SD (ddof 1) of the last 1,800 defined d before u (count-based); with fewer since 09:30, all of them; none
              (no event) under 300 (ADDENDUM A3); bars from 09:30 count, the day before never
  event       at decision bar u from 09:35:00 to 15:50:00 (bars 299 .. close - 601): |d| >= k sigma_d, |d| > 0, sigma_d > 0,
              the follower flat; direction = sign(d) (d > 0: buy the follower)
  entry       research 24's single leg: market order at bar u + delay through pair20.order (engine17, fill rule 3), shares
              = pair24.shares(m_B,u, 10,000 dollars); L3D1 / L0D0 = depth penalty 1 / 0 on the same fills; fee 0.3 cents
  exits       checked from the first bar after the entry is complete, at bars with a valid follower quote, in this order:
              15:55:00 (forced, any quote), stop (the follower moved against by |d| from m_B,u), target (moved with it by
              phi |d|), time (T seconds after the entry start); the exit order is a market order at the decision bar +
              delay. Limit exit (SPEC30 2A-1): a limit at the target price (tick-rounded away from the entry) posted at the
              entry bar, filled by the Q model (engine17.fill_buy / fill_sell = research 12's, theta 1); the queue ahead is
              the displayed size at that price when it is the best quote at posting, else the displayed size the first time
              it is (DEVIATIONS30 1); what is left at T, at a stop or at 15:55 goes to market
  money       per trade, dollars: mid = the orders at the mids of their decision bars (a limit fill at the mid of its fill
              bar), cash, cost = mid - cash, fee; pnl = cash - fee (pair20.money's definitions for one leg)
  limits      (validation and sealed only, SPEC30 2A-3): one position per follower, at most 5 open, the dollars of the open
              positions and the new one <= 50,000; no entry after 15:50, everything out at 15:55
Bars of 2019 on are refused before the tag cand30, and the sealed bars are never named here."""
import heapq
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
R24 = os.path.join(PROJ, "research24")
if R24 not in sys.path:
    sys.path.insert(1, R24)
import core24 as C24  # noqa: E402  (frozen-file check of research 17, 20, 21 at import)
import loader24 as L24  # noqa: E402
import pair24 as P24  # noqa: E402
import stats24 as ST24  # noqa: E402

PE = C24.PE
G = PE.G
jit = PE.jit
TICK, FEE_C = G.TICK, G.FEE_MKT                          # 100 (1 cent in 1e-4 dollars); 0.3 cents a market share
REBATE_C = 0.2                                           # R+ cents a limit share (secondary only, SPEC30 2A-5)
AMOUNT = ST24.AMOUNT                                     # 10,000 dollars
DELAYS, DELAY_MAIN = (0, 1, 5), 1
S_GRID, K_GRID = (5, 15, 30), (2.0, 3.0)
SIG_N, SIG_MIN = 1800, 300
FIRST_ENTRY = 299                                        # bar ending 09:35:00
BASE_EXIT = {"phi": 1.0, "T": 120}                       # the exploration map's exit (DEVIATIONS30 1)
MAX_OPEN, MAX_USD = 5, 50_000
EXPLORE = ("20150102", "20181231")
VALID = ("20190102", "20201231")
CAND_TAG = "cand30"
R_TARGET, R_STOP, R_TIME, R_CLOSE, R_LIMIT = 1, 2, 3, 4, 5
VOL_W = 60


class Refused(Exception):
    pass


def tag_ok(tag=CAND_TAG):
    r = subprocess.run(["git", "-C", PROJ, "merge-base", "--is-ancestor", tag, "HEAD"], capture_output=True)
    return r.returncode == 0


def tasks(window):
    """Research 24's development pair-days inside the window; 2019-2020 only after the tag cand30; nothing after 2020."""
    if window[1] > VALID[1]:
        raise Refused("research 30 reads development bars only here (2015-2020)")
    if window[1] > EXPLORE[1] and not tag_ok():
        raise Refused("2019-2020 bars need the tag cand30 (SPEC30 3.3, 9)")
    t, days, skipped = L24.dev_plan()
    t = [x for x in t if window[0] <= x["date"] <= window[1]]
    days = [d for d in days if window[0] <= d <= window[1]]
    return t, days, skipped


def load(task):
    """(leader bars, follower bars) of one pair-day through loader24 (development root, its guard)."""
    if task["date"] > EXPLORE[1] and not tag_ok():
        raise Refused(f"{task['date']}: 2019-2020 bars need the tag cand30")
    return L24.load(L24.DEV_ROOT, task["row_a"]), L24.load(L24.DEV_ROOT, task["row_b"])


def last_entry_bar(close_bar):
    return close_bar - 601                               # the bar ending 15:50:00 (12:50:00 on a 13:00 close)


def forced_bar(close_bar):
    return close_bar - 301                               # the bar ending 15:55:00 (pair20's F)


# ---------------------------------------------------------------- signal
def quotes(x):
    """(valid, ln mid, mid2) of one leg's bars."""
    bid, ask = np.asarray(x["bid"], dtype=np.int64), np.asarray(x["ask"], dtype=np.int64)
    valid = (bid > 0) & (ask > bid)
    m2 = bid + ask
    lm = np.full(len(bid), np.nan)
    lm[valid] = np.log(m2[valid].astype(np.float64) / 2e4)
    return valid, lm, m2


def dseries(a, b, s):
    """d_u for every bar (NaN where undefined)."""
    va, la, _ = quotes(a)
    vb, lb, _ = quotes(b)
    N = len(la)
    d = np.full(N, np.nan)
    ok = np.zeros(N, dtype=bool)
    ok[s:] = va[s:] & va[:-s] & vb[s:] & vb[:-s]
    d[s:] = np.where(ok[s:], (la[s:] - la[:-s]) - (lb[s:] - lb[:-s]), np.nan)
    return d


def sigma(d):
    """(sigma_d, count) per bar from the defined d strictly before it: the last SIG_N of them, all of them when fewer."""
    N = len(d)
    iv = np.flatnonzero(np.isfinite(d))
    x = d[iv]
    c1 = np.r_[0.0, np.cumsum(x)]
    c2 = np.r_[0.0, np.cumsum(x * x)]
    pos = np.searchsorted(iv, np.arange(N), side="left")             # defined d strictly before bar u
    lo = np.maximum(pos - SIG_N, 0)
    n = pos - lo
    s1, s2 = c1[pos] - c1[lo], c2[pos] - c2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / np.where(n > 0, n, 1)) / np.where(n > 1, n - 1, 1)
    sd = np.where(n >= 2, np.sqrt(np.maximum(var, 0.0)), np.nan)
    return sd, n


def candidates(d, sd, n, k, close_bar):
    """Bars where an event may start (before the flat / limits conditions)."""
    N = len(d)
    u = np.arange(N)
    with np.errstate(invalid="ignore"):
        c = (np.isfinite(d) & (n >= SIG_MIN) & (sd > 0) & (np.abs(d) > 0) & (np.abs(d) >= k * sd)
             & (u >= FIRST_ENTRY) & (u <= last_entry_bar(close_bar)))
    return np.flatnonzero(c)


# ---------------------------------------------------------------- one follower trade
NOUT = 24
# out columns: 0 entry start, 1 exit decision bar, 2 reason, 3 shares, 4 entry mkt notional, 5 entry over, 6 exit mkt shares,
# 7 exit mkt notional, 8 exit over, 9 lim shares, 10 lim notional, 11 mid sum of the exit shares (sum shares x mid2 at
# their decision / fill bar), 12 exit complete bar, 13 limit price, 14 entry mid2 (decision), 15 first limit fill bar,
# 16 entry done, 17..23 spare


@jit
def follow(X, valid, lm, m2, u, dirn, ad, qty, delay, phi, T, limit, F, last_bar, out):
    """One follower trade decided at bar u (see the module text); writes `out` (NOUT columns)."""
    for c in range(NOUT):
        out[c] = 0
    dirs1, starts1, o1 = np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros((1, 9), np.int64)
    s0 = u + delay
    PE.order(X, 0, 3, dirn, s0, 30, qty, dirs1, starts1, o1)
    out[0], out[3], out[4], out[5], out[16], out[14] = s0, qty, o1[0, 3], o1[0, 4], o1[0, 5], m2[u]
    first = max(u + 1, s0 + o1[0, 5])
    rem = qty
    px = 0
    a4 = -2
    if limit:
        tgt = m2[u] * np.exp(dirn * phi * ad) / 2.0                     # 1e-4 dollars
        if dirn > 0:
            px = int(np.ceil(tgt / TICK - 1e-9)) * TICK
            if px == X[1, s0]:
                a4 = 4 * X[3, s0]
        else:
            px = int(np.floor(tgt / TICK + 1e-9)) * TICK
            if px == X[0, s0]:
                a4 = 4 * X[2, s0]
        out[13] = px
    reason = 0
    v_dec = 0
    for v in range(s0 + 1 if limit else first, last_bar + 1):
        if limit and rem > 0 and v > s0:
            if dirn > 0:
                f, a4n = G.fill_sell(2, px, rem, a4, v, X[4], X[7], X[6], X[13], X[9], X[11])
            else:
                f, a4n = G.fill_buy(2, px, rem, a4, v, X[7], X[4], X[5], X[12], X[8], X[10])
            if f > 0:
                if out[9] == 0:
                    out[15] = v
                rem -= f
                out[9] += f
                out[10] += f * px
                out[11] += f * m2[v]
            if a4n == -2:
                if dirn > 0 and X[1, v] == px:
                    a4n = 4 * X[3, v]
                elif dirn < 0 and X[0, v] == px:
                    a4n = 4 * X[2, v]
            a4 = a4n
            if rem == 0:
                reason = R_LIMIT
                v_dec = v
                out[12] = v
                break
        if v < first:
            continue
        forced = v >= F
        if not forced and not valid[v]:
            continue
        if forced:
            reason = R_CLOSE
        else:
            mv = dirn * (lm[v] - lm[u])
            if mv <= -ad:
                reason = R_STOP
            elif (not limit) and mv >= phi * ad:
                reason = R_TARGET
            elif v - s0 >= T:
                reason = R_TIME
            else:
                continue
        v_dec = v
        s1 = min(v + delay, last_bar)
        PE.order(X, 0, 3, -dirn, s1, 30, rem, dirs1, starts1, o1)
        out[6], out[7], out[8] = o1[0, 2], o1[0, 3], o1[0, 4]
        out[11] += rem * m2[v]
        out[12] = s1 + o1[0, 5]
        break
    out[1], out[2] = v_dec, reason


def money(out, dirn, depth):
    """(pnl, mid, cost, fee, rebate) in dollars of one follower trade (pair20.money's definitions for one leg)."""
    q = out[3]
    ent = out[4] + depth * dirn * TICK * out[5]                          # entry notional, 1e-4 dollars
    ext = out[10] + out[7] - depth * dirn * TICK * out[8]                # exit notional (direction -dirn)
    cash = (-dirn * ent + dirn * ext) / 1e4
    mid = dirn * (out[11] - q * out[14]) / 2e4
    fee = FEE_C / 100.0 * (q + out[6])
    reb = REBATE_C / 100.0 * out[9]
    return cash - fee, mid, mid - cash, fee, reb


# ---------------------------------------------------------------- per pair-day
def prepare(a, b, close_bar):
    """Arrays of one pair-day that do not depend on s or k; None when a leg lacks a quote state after its first one."""
    if not (C24.C.state_ok(a) and C24.C.state_ok(b)):
        return None
    vb, lb, m2b = quotes(b)
    X = PE.stack(b)
    with np.errstate(invalid="ignore"):
        r1 = np.r_[np.nan, np.diff(lb)]
    return {"a": a, "b": b, "X": X, "valid": vb, "lm": lb, "m2": m2b, "r1": r1, "close_bar": close_bar}


def features(Q, u, dirn, ad):
    """Map variables of an event at bar u (DEVIATIONS30 1): spread ticks, displayed size on the side taken, 1-minute
    volatility (bp), time bucket, |d| / round-trip cost."""
    X, m2 = Q["X"], Q["m2"][u]
    bid, ask = X[0, u], X[1, u]
    size = X[3, u] if dirn > 0 else X[2, u]
    w = Q["r1"][max(u - VOL_W + 1, 0):u + 1]
    w = w[np.isfinite(w)]
    vol = float(w.std(ddof=1) * 1e4) if len(w) >= 2 else np.nan
    bucket = 0 if u < 1799 else 1 + (u - 1799) // 1800
    rt = (ask - bid) * 2.0 / m2 + 2.0 * (FEE_C * 100.0) * 2.0 / m2          # spread / mid + 2 x fee / mid (fee 30 in 1e-4)
    return (ask - bid) / TICK, size, vol, bucket, ad / rt


def events(Q, s, k, delays=DELAYS, base=BASE_EXIT):
    """Events of one pair-day for (s, k) without portfolio limits (the exploration map, DEVIATIONS30 1): the sequence is
    set by the market exit at the main delay (the next event only after that exit is complete); each event is valued
    under both exit methods at the main delay and under the market exit at every delay. Rows: see EV_COLS."""
    cb = Q["close_bar"]
    d = dseries(Q["a"], Q["b"], s)
    sd, n = sigma(d)
    cand = candidates(d, sd, n, k, cb)
    F, last = forced_bar(cb), cb - 1
    rows = []
    free = 0
    o = np.zeros(NOUT, dtype=np.int64)
    ol = np.zeros(NOUT, dtype=np.int64)
    od = np.zeros(NOUT, dtype=np.int64)
    i = 0
    while True:
        i = int(np.searchsorted(cand, free, side="left")) if i < len(cand) else i
        if i >= len(cand):
            break
        u = int(cand[i])
        dirn = 1 if d[u] > 0 else -1
        ad = abs(float(d[u]))
        q = int(P24.shares(Q["m2"][u], AMOUNT))
        follow(Q["X"], Q["valid"], Q["lm"], Q["m2"], u, dirn, ad, q, DELAY_MAIN, base["phi"], base["T"], False, F, last, o)
        follow(Q["X"], Q["valid"], Q["lm"], Q["m2"], u, dirn, ad, q, DELAY_MAIN, base["phi"], base["T"], True, F, last, ol)
        r = [u, dirn, ad, float(sd[u]), q, *features(Q, u, dirn, ad)]
        p1, m1, _, f1 = money(o, dirn, 1)[:4]
        r += [p1, m1, f1, money(o, dirn, 0)[0], o[2], o[1] - o[0]]
        p1, m1, _, f1, rb = money(ol, dirn, 1)
        r += [p1, m1, f1, rb, money(ol, dirn, 0)[0], ol[2], ol[9], (ol[15] - ol[0]) if ol[9] else -1]
        for dl in delays:
            if dl == DELAY_MAIN:
                continue
            follow(Q["X"], Q["valid"], Q["lm"], Q["m2"], u, dirn, ad, q, dl, base["phi"], base["T"], False, F, last, od)
            p1, m1, _, f1 = money(od, dirn, 1)[:4]
            r += [p1, m1, f1]
        # markouts: follower mid h seconds after the entry fill bar against the entry fill price, cents a share, d direction
        s0 = u + DELAY_MAIN
        fill = o[4] / q
        for h in MARKOUTS:
            v = s0 + h
            r.append(dirn * (Q["m2"][v] / 2.0 - fill) / 100.0 if v <= last and Q["valid"][v] else np.nan)
        rows.append(r)
        free = max(u + 1, int(o[12]) + 1)
    return np.asarray(rows, dtype=np.float64).reshape(-1, len(EV_COLS))


MARKOUTS = (5, 15, 30, 60, 120, 300)
EV_COLS = (("u", "dir", "ad", "sigma", "qty", "spread_ticks", "size", "vol1m_bp", "bucket", "ratio",
            "mkt_pnl", "mkt_mid", "mkt_fee", "mkt_pnl_L0D0", "mkt_reason", "mkt_hold",
            "lim_pnl", "lim_mid", "lim_fee", "lim_rebate", "lim_pnl_L0D0", "lim_reason", "lim_shares", "lim_first_fill_s")
           + tuple(f"d{dl}_{x}" for dl in DELAYS if dl != DELAY_MAIN for x in ("pnl", "mid", "fee"))
           + tuple(f"mo{h}_c" for h in MARKOUTS))                  # mkt_* and lim_* = L3D1 at the main delay; cost = mid - pnl - fee
EVI = {c: i for i, c in enumerate(EV_COLS)}


# ---------------------------------------------------------------- the day with portfolio limits (validation, sealed)
def portfolio_day(Qs, cfg, delay=DELAY_MAIN, depth=1, max_open=MAX_OPEN, max_usd=MAX_USD):
    """Trades of one day over all its pairs under SPEC30 2A-3. Qs: [prepared pair-day or None]; cfg: s, k, phi, T,
    limit (bool), gate (callable(Q, u, dirn, ad, feats) -> bool or None). Candidates are taken in time order (pair order
    for ties); an entry needs the follower flat, fewer than max_open positions open and the dollars of the open positions
    (shares x decision mid) plus the new one <= max_usd. A position counts as open from its decision bar to the bar its
    exit order is complete. Returns [(pair index, u, dirn, pnl, mid, cost, fee, rebate, reason)]."""
    heap, state = [], []
    for pi, Q in enumerate(Qs):
        if Q is None:
            state.append(None)
            continue
        d = dseries(Q["a"], Q["b"], cfg["s"])
        sd, n = sigma(d)
        cand = candidates(d, sd, n, cfg["k"], Q["close_bar"])
        state.append({"d": d, "cand": cand, "free": 0})
        if len(cand):
            heapq.heappush(heap, (int(cand[0]), pi))
    open_pos = []                                                        # (end bar, usd)
    out = []
    o = np.zeros(NOUT, dtype=np.int64)
    while heap:
        u, pi = heapq.heappop(heap)
        Q, st = Qs[pi], state[pi]
        open_pos = [(e, usd) for e, usd in open_pos if e >= u]
        nxt = None
        if u >= st["free"]:
            dirn = 1 if st["d"][u] > 0 else -1
            ad = abs(float(st["d"][u]))
            q = int(P24.shares(Q["m2"][u], AMOUNT))
            usd = q * Q["m2"][u] / 2e4
            ok = len(open_pos) < max_open and sum(x for _, x in open_pos) + usd <= max_usd
            if ok and cfg.get("gate") is not None:
                ok = bool(cfg["gate"](Q, u, dirn, ad, features(Q, u, dirn, ad)))
            if ok:
                cb = Q["close_bar"]
                follow(Q["X"], Q["valid"], Q["lm"], Q["m2"], u, dirn, ad, q, delay, cfg["phi"], cfg["T"], bool(cfg["limit"]),
                       forced_bar(cb), cb - 1, o)
                pnl, mid, cost, fee, reb = money(o, dirn, depth)
                out.append((pi, u, dirn, pnl, mid, cost, fee, reb, int(o[2])))
                end = int(o[12])
                open_pos.append((end, usd))
                st["free"] = max(u + 1, end + 1)
        j = int(np.searchsorted(st["cand"], max(u + 1, st["free"]), side="left"))
        if j < len(st["cand"]):
            nxt = int(st["cand"][j])
            heapq.heappush(heap, (nxt, pi))
    return out


# ---------------------------------------------------------------- 3.1: leg shares of research 24 trades
def leg_mid(tr):
    """(leader leg mid, follower leg mid) per research 24 trade row, dollars: pair20.money's mid term order by order
    (orders 0 and 2 are leg A, 1 and 3 leg B); their sum is pair20.money's mid."""
    tr = np.asarray(tr, dtype=np.int64).reshape(-1, PE.NTR)
    legs = np.zeros((len(tr), 2))
    for j in range(PE.NORD):
        dd = PE.ORDER_DIR[j] * tr[:, 2]
        legs[:, j % 2] += -dd * tr[:, PE.ORDER_QCOL[j]] * tr[:, PE.MID0 + j] / 2e4
    return legs[:, 0], legs[:, 1]


def leg_cash(tr, depth=1):
    """(leader leg cash, follower leg cash) per trade, dollars, with the depth penalty (the fill-based split)."""
    tr = np.asarray(tr, dtype=np.int64).reshape(-1, PE.NTR)
    legs = np.zeros((len(tr), 2))
    for j in range(PE.NORD):
        dd = PE.ORDER_DIR[j] * tr[:, 2]
        oo = tr[:, PE.O0 + j * PE.OCOL:PE.O0 + (j + 1) * PE.OCOL]
        legs[:, j % 2] += -dd * (oo[:, 1] + oo[:, 3] + depth * dd * TICK * oo[:, 4]) / 1e4
    return legs[:, 0], legs[:, 1]
