#!/usr/bin/env python3
"""Research 20 pair engine (SPEC20 2-4, SPEC20_ANSWERS A2-A4; readings fixed before any result in DEVIATIONS20.md 1).

The fill kernel is study17_new_stocks/engine17.py unchanged (sha256 checked at import): every parent order of a pair trade is one
call of engine17.exec17 with a single parent.
  M  method 0 (S0): the whole order at the end quote of its start bar.
  P  method 2 with improve = 1, no E1 rule, no stop (S2i) and a 30-second deadline: a limit one tick inside when the spread
     is >= 3 ticks, else at the best quote, followed at the kernel's 5-second decision bars; what is left at the deadline
     goes to market at that bar's end quote.
One position per pair. At a grid point (signal20) the engine takes at most one decision:
  flat      entry when the entry flag is set and the last exit orders are complete: leg A direction = -side, leg B = +side
            (z > 0: sell A, buy B), both orders start at the decision bar (+ delay).
  open      from the first grid point at or after the completion of both entry orders (and after the entry decision), at
            points with valid quotes only: exit when the forced bar (close - 5 min) is reached, |z| <= 0.5, or 30 minutes
            have passed since the entry decision. Exit orders use the entry's method, except the forced exit (always M);
            a P exit that starts before the forced bar has its deadline cut to the forced bar, so that everything still
            open at 15:55:00 goes to market. A position that meets no valid grid point from the forced bar to the close is
            valued at the mids of the last valid grid point, without an order (SPEC20_ADDENDUM B1; reason 4, counted).
Fill assumptions: L3D1 = kernel fill rule 3 + depth 1 (market shares above the displayed size one tick worse), L0D0 = fill
rule 0 + depth 0. Market shares pay 0.3 cents; limit shares pay nothing.
Trade row (int64, NTR columns): entry bar, exit bar, side, shares A, shares B, reason, then for the four orders (entry A,
entry B, exit A, exit B) the kernel's lim_sh, lim_not, mkt_sh, mkt_not, over, done, deadline shares, then the four
decision-time 2 x mids, then the exit method (0 M, 2 P, 9 valued at the last valid mids: booked as limit shares at the
mid, so no fee and no depth penalty)."""
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R17 = os.path.join(os.path.dirname(HERE), "study17_new_stocks")
ENGINE17_SHA = "2a9ddcb7db7b96201cbd63b62e44c7c00140447a947e883aeeebf6a542d8d42a"
if hashlib.sha256(open(os.path.join(R17, "engine17.py"), "rb").read()).hexdigest() != ENGINE17_SHA:
    raise ImportError("study17_new_stocks/engine17.py differs from the frozen kernel")
sys.path.insert(0, HERE)
sys.path.insert(1, R17)
import engine17 as G  # noqa: E402
import signal20 as S  # noqa: E402

TICK, FEE_MKT = G.TICK, G.FEE_MKT                     # 100 (1 cent in 1e-4 dollars), 0.3 cents per market share
HOR_P = 30                                            # P: seconds before the rest goes to market
M, P = 0, 2                                           # kernel methods
VALUED = 9                                            # exit "method" of a position valued at the last valid mids (no order)
VARIANTS = (("L3D1", 3, 1), ("L0D0", 0, 0))           # label, fill rule, depth
FIELDS = ("bid", "ask", "bsz", "asz", "bid_max", "bid_min", "ask_max", "ask_min", "vol_bid", "vol_ask", "vol_bid_odd",
          "vol_ask_odd", "px_min", "px_max")          # the kernel's bar arguments, in order
NORD, OCOL = 4, 7                                     # orders per trade, kernel columns kept per order
NTR = 6 + NORD * OCOL + NORD + 1
O0, MID0, XM = 6, 6 + NORD * OCOL, 6 + NORD * OCOL + NORD
_exec17 = G.exec17
jit = G.jit


def stack(b):
    """The 14 kernel arrays of one bar dict as a C-contiguous int64 matrix (row order FIELDS)."""
    return np.ascontiguousarray(np.stack([np.asarray(b[k], dtype=np.int64) for k in FIELDS]))


@jit
def order(X, method, lrule, d, s, horizon, qty, dirs1, starts1, out1):
    """One parent order through the research 17 kernel; the result is out1[0] (NCOL columns)."""
    dirs1[0], starts1[0] = d, s
    improve = 1 if method == 2 else 0
    _exec17(method, lrule, improve, -1, 0, dirs1, starts1, horizon, qty, X[0], X[1], X[2], X[3], X[4], X[5], X[6], X[7],
            X[8], X[9], X[10], X[11], X[12], X[13], out1)


@jit
def run_parents(X, method, lrule, dirs, starts, horizon, qty, out):
    """Regression path: the parents of engine17 one by one through order()."""
    dirs1, starts1, out1 = np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros((1, 9), np.int64)
    for i in range(len(dirs)):
        order(X, method, lrule, dirs[i], starts[i], horizon, qty, dirs1, starts1, out1)
        for c in range(9):
            out[i, c] = out1[0, c]


@jit
def pairday(method, lrule, delay, g, valid, side, exz, mid2a, mid2b, XA, XB, close_bar, tr):
    """Trades of one pair-day for one signal configuration and execution method; returns the number of rows of tr used."""
    K = len(g)
    F = close_bar - 300 - 1
    last_bar = close_bar - 1
    dirs1, starts1, out1 = np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros((1, 9), np.int64)
    n, pos, next_free, first_exit, ge, sd, qa, qb, lastv = 0, 0, 0, 0, 0, 0, 0, 0, -1
    for k in range(K):
        gk = g[k]
        if valid[k]:
            lastv = k
        if pos == 0:
            if side[k] == 0 or gk < next_free:
                continue
            sd, ge = side[k], gk
            qa = max((2 * 100000000 + mid2a[k]) // (2 * mid2a[k]), 1) * 10
            qb = max((2 * 100000000 + mid2b[k]) // (2 * mid2b[k]), 1) * 10
            s0 = gk + delay
            tr[n, 0], tr[n, 2], tr[n, 3], tr[n, 4] = gk, sd, qa, qb
            done = 0
            for leg in range(2):
                d = -sd if leg == 0 else sd
                if leg == 0:
                    order(XA, method, lrule, d, s0, HOR_P, qa, dirs1, starts1, out1)
                else:
                    order(XB, method, lrule, d, s0, HOR_P, qb, dirs1, starts1, out1)
                for c in range(OCOL):
                    tr[n, O0 + leg * OCOL + c] = out1[0, c]
                done = max(done, out1[0, 5])
            tr[n, MID0], tr[n, MID0 + 1] = mid2a[k], mid2b[k]
            first_exit = max(gk + 1, s0 + done)
            pos = 1
        else:
            if gk < first_exit:
                continue
            last = k == K - 1
            if not valid[k] and not last:
                continue
            forced = gk >= F
            if last and not valid[k]:
                reason = 4
            elif forced:
                reason = 3
            elif exz[k]:
                reason = 1
            elif gk - ge >= 1800:
                reason = 2
            else:
                continue
            if reason == 4:                                         # ADDENDUM B1: no valid quote left, no order is sent
                tr[n, O0 + 2 * OCOL], tr[n, O0 + 2 * OCOL + 1] = qa, qa * mid2a[lastv] // 2
                tr[n, O0 + 3 * OCOL], tr[n, O0 + 3 * OCOL + 1] = qb, qb * mid2b[lastv] // 2
                tr[n, 1], tr[n, 5] = gk, reason
                tr[n, MID0 + 2], tr[n, MID0 + 3] = mid2a[lastv], mid2b[lastv]
                tr[n, XM] = 9
                next_free = gk + 1
                pos = 0
                n += 1
                continue
            xm = 0 if forced else method
            hor = min(HOR_P, F - gk) if xm == 2 else HOR_P
            s1 = min(gk + delay, last_bar)
            done = 0
            for leg in range(2):
                d = sd if leg == 0 else -sd
                if leg == 0:
                    order(XA, xm, lrule, d, s1, hor, qa, dirs1, starts1, out1)
                else:
                    order(XB, xm, lrule, d, s1, hor, qb, dirs1, starts1, out1)
                for c in range(OCOL):
                    tr[n, O0 + (2 + leg) * OCOL + c] = out1[0, c]
                done = max(done, out1[0, 5])
            tr[n, 1], tr[n, 5] = gk, reason
            tr[n, MID0 + 2], tr[n, MID0 + 3] = mid2a[k], mid2b[k]
            tr[n, XM] = xm
            next_free = max(gk + 1, s1 + done)
            pos = 0
            n += 1
    if pos != 0:
        return -1                                                   # a position left open: never expected (the caller raises)
    return n


def trades(method, lrule, delay, B, side, exz, XA, XB, close_bar):
    """Trade rows (n x NTR) of one pair-day; B = signal20.base(...), side / exz = signal20.flags(...)."""
    tr = np.zeros((int((side != 0).sum()) + 1, NTR), dtype=np.int64)
    n = pairday(method, lrule, delay, B["g"], B["valid"], side, exz, B["mid2_a"], B["mid2_b"], XA, XB, close_bar, tr)
    if n < 0:
        raise RuntimeError("a position was left open at the end of the day")
    return tr[:n]


# ---------------------------------------------------------------- money
ORDER_DIR = np.array([-1, 1, 1, -1], dtype=np.int64)               # x side: entry A, entry B, exit A, exit B
ORDER_QCOL = (3, 4, 3, 4)


def money(tr, depth):
    """Per trade, dollars: pnl (cash flows of the four orders minus the market fee), mid (the four orders at the mid of
    their decision time), cost (execution prices against those mids, the depth penalty included), fee. pnl = mid - cost - fee."""
    tr = np.asarray(tr, dtype=np.int64).reshape(-1, NTR)
    n = len(tr)
    cash, fee, mid = np.zeros(n), np.zeros(n), np.zeros(n)
    for j in range(NORD):
        d = ORDER_DIR[j] * tr[:, 2]
        o = tr[:, O0 + j * OCOL:O0 + (j + 1) * OCOL]
        notional = o[:, 1] + o[:, 3] + depth * d * TICK * o[:, 4]             # 1e-4 dollars
        cash += -d * notional / 1e4
        fee += FEE_MKT / 100.0 * o[:, 2]
        mid += -d * tr[:, ORDER_QCOL[j]] * tr[:, MID0 + j] / 2e4
    return {"pnl": cash - fee, "mid": mid, "cost": mid - cash, "fee": fee}


def fills(tr, method):
    """Counts for the P statistics of a P run, over its two-leg order pairs sent as P (every entry, and the exits whose
    method is P; forced exits are M): (pairs, pairs with exactly one leg completed by limit fills only, P orders, P orders
    with deadline shares, P shares, P deadline shares). Zeros for an M run."""
    tr = np.asarray(tr, dtype=np.int64).reshape(-1, NTR)
    res = np.zeros(6, dtype=np.int64)
    if method != P:
        return res
    for first, is_p in ((0, np.ones(len(tr), bool)), (2, tr[:, XM] == P)):
        a = tr[is_p][:, O0 + first * OCOL:O0 + (first + 1) * OCOL]
        b = tr[is_p][:, O0 + (first + 1) * OCOL:O0 + (first + 2) * OCOL]
        la, lb = a[:, 6] == 0, b[:, 6] == 0                                 # no deadline shares
        res += np.array([len(a), int((la ^ lb).sum()), 2 * len(a), int((~la).sum() + (~lb).sum()),
                         int((a[:, 0] + a[:, 2] + b[:, 0] + b[:, 2]).sum()), int((a[:, 6] + b[:, 6]).sum())])
    return res
