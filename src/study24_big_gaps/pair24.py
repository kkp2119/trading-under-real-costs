#!/usr/bin/env python3
"""Research 24 pair engine with the dollar amount per leg as an argument (SPEC24 2, SPEC24_ANSWERS A3; SPEC24_QUESTIONS 9).

study20_pairs/pair20.py fixes the order size inside pairday ("50,000 dollars / mid") and is not edited. pairday below is a
copy of pair20.pairday, line for line, in which the two lines that set the shares take the amount from an argument; the
fill kernel (engine17.exec17 through pair20.order), the trade rows, pair20.money and everything else are research 20's,
imported unchanged. With 50,000 dollars the trade rows are identical to pair20.pairday's (synthetic and regression tests).
  size   per leg: amount / mid at the entry decision in lots of 10 shares, half up, at least 10 shares (research 20's rule).
         In integers: mid2 = bid + ask in 1e-4 dollars, n = amount x 2,000, lots = (2 n + mid2) // (2 mid2).
  delay  pair20's argument, unchanged: the decision is taken at grid bar t, the two entry orders start at bar t + delay and
         the two exit orders at bar t + delay (not after the last bar of the day); a market order fills whole at the end
         quote of its start bar."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R20 = os.path.join(os.path.dirname(HERE), "study20_pairs")
if R20 not in sys.path:
    sys.path.insert(1, R20)
import pair20 as PE  # noqa: E402  (checks study17_new_stocks/engine17.py at import)

S = PE.S
LOT = S.LOT
HOR_P, OCOL, O0, MID0, XM, NTR = PE.HOR_P, PE.OCOL, PE.O0, PE.MID0, PE.XM, PE.NTR
order = PE.order
jit = PE.jit


def amount_n(amount):
    """amount in whole dollars -> n = amount x 2,000 (so that lots = floor(amount / mid / 10 + 1/2) in integers)."""
    if int(amount) != amount or amount <= 0:
        raise ValueError("the amount per leg must be a positive whole number of dollars")
    return int(amount) * 2000


def shares(mid2, amount):
    """amount dollars / mid in lots of 10 shares, half up, at least one lot; mid2 = bid + ask in 1e-4 dollars."""
    mid2 = np.asarray(mid2, dtype=np.int64)
    return np.maximum((2 * amount_n(amount) + mid2) // (2 * mid2), 1) * LOT


@jit
def pairday(an, method, lrule, delay, g, valid, side, exz, mid2a, mid2b, XA, XB, close_bar, tr):
    """pair20.pairday with the shares from an = amount x 2,000 (the only two changed lines are marked R24)."""
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
            qa = max((2 * an + mid2a[k]) // (2 * mid2a[k]), 1) * 10            # R24
            qb = max((2 * an + mid2b[k]) // (2 * mid2b[k]), 1) * 10            # R24
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


def trades(amount, method, lrule, delay, B, side, exz, XA, XB, close_bar):
    """Trade rows (n x NTR) of one pair-day with `amount` dollars per leg; B = signal20.base(...), side / exz = signal20.flags(...)."""
    tr = np.zeros((int((side != 0).sum()) + 1, NTR), dtype=np.int64)
    n = pairday(amount_n(amount), method, lrule, delay, B["g"], B["valid"], side, exz, B["mid2_a"], B["mid2_b"], XA, XB, close_bar, tr)
    if n < 0:
        raise RuntimeError("a position was left open at the end of the day")
    return tr[:n]
