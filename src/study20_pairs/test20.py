#!/usr/bin/env python3
"""Research 20 synthetic tests of signal20, pair20, core20 and stats20 (step 2; they must pass before any development run).
Synthetic bars only: two legs quoted $100.00 / $100.02 with 5,000 shares displayed and no trades (a posted order fills only
when the opposite quote reaches it), moved on chosen bars. Step 2b added gap_tests (SPEC20_ADDENDUM B1: quote gaps) and
delay_tests (B2: mid prices taken d seconds late). Writes results/engine20_tests.json; exits 1 when a test fails.
  python test20.py"""
import json
import os
import sys
from decimal import Decimal, ROUND_FLOOR

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core20 as C  # noqa: E402
import pair20 as PE  # noqa: E402
import signal20 as S  # noqa: E402
import stats20 as ST  # noqa: E402

G = PE.G
N, BID, SPR, SZ = 23_400, 1_000_000, 200, 5000
GE = 7199                                                    # 11:30:00, a grid bar
RESULTS = []


def check(name, ok, **info):
    RESULTS.append({"test": name, "pass": bool(ok), **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}})


def leg(bid=BID, spread=SPR, size=SZ):
    b = {k: np.zeros(N, np.int64) for k in PE.FIELDS}
    for k in ("bid", "bid_max", "bid_min"):
        b[k][:] = bid
    for k in ("ask", "ask_max", "ask_min"):
        b[k][:] = bid + spread
    b["bsz"][:], b["asz"][:] = size, size
    return b


def setq(b, u0, u1, bid, ask=None):
    ask = bid + SPR if ask is None else ask
    for k in ("bid", "bid_max", "bid_min"):
        b[k][u0:u1] = bid
    for k in ("ask", "ask_max", "ask_min"):
        b[k][u0:u1] = ask
    return b


def ramp(b, g0, jump, until):
    """From grid bar g0: bid = BID + jump ticks, then one tick more at every grid step, up to bar `until`."""
    j = 0
    for u in range(g0, until, 5):
        setq(b, u, min(u + 5, until), BID + (jump + j) * 100)
        j += 1
    return b


def run(a, b, W=10, zi=2.0, ka=0.0, method=PE.M, lrule=3, delay=0, close_bar=N):
    B = S.base(a, b, close_bar)
    side, exz = S.flags(B, W, zi, ka, close_bar)
    return PE.trades(method, lrule, delay, B, side, exz, PE.stack(a), PE.stack(b), close_bar), B, side, exz


def od(tr, i, j):
    return tr[i, PE.O0 + j * PE.OCOL:PE.O0 + (j + 1) * PE.OCOL]


def one(b, method, lrule, d, s, horizon, qty):
    dirs1, starts1, out1 = np.zeros(1, np.int64), np.zeros(1, np.int64), np.zeros((1, 9), np.int64)
    PE.order(PE.stack(b), method, lrule, d, s, horizon, qty, dirs1, starts1, out1)
    return out1[0].copy()


def spike_leg(g0=GE, g1=GE + 30, ticks=10):
    return setq(leg(), g0, g1, BID + ticks * 100)


# ---------------------------------------------------------------- signal
def signal_tests():
    g = S.grid(N)
    check("grid 09:30:05 every 5 s to the close", g[0] == 4 and g[-1] == 23_399 and len(g) == 4680 and np.all(np.diff(g) == 5))
    check("entry hours and forced bar", S.entry_bars(10, N) == (599, 21_599) and S.entry_bars(30, N) == (1799, 21_599)
          and S.entry_bars(3, N) == (299, 21_599) and S.force_bar(N) == 23_099 and S.entry_bars(10, 12_600) == (599, 10_799)
          and S.force_bar(12_600) == 12_299)
    cases = {2_000_200: 500, 20_000_000: 50, 40_000_000: 30, 60_000_000: 20, 140_000_000: 10, 1_200_000_000: 10, 246_900: 4050,
             10_688_500: 90}
    ref = {m: max(int((Decimal(50_000) / (Decimal(m) / 20_000) / 10 + Decimal("0.5")).to_integral_value(ROUND_FLOOR)), 1) * 10 for m in cases}
    check("shares: 50,000 dollars / mid in lots of 10, half up, at least 10",
          all(int(S.shares(m)) == q == ref[m] for m, q in cases.items()), got={str(m): int(S.shares(m)) for m in cases})
    rng = np.random.default_rng(20)
    K = 4680
    valid = rng.random(K) > 0.02
    s = np.cumsum(rng.normal(0, 1e-4, K))
    s[~valid] = np.nan
    for W in S.WS:
        n = W * 12
        z, dev, has = S.zscore(s, valid, W)
        zr, hr = np.full(K, np.nan), np.zeros(K, bool)
        iv = np.flatnonzero(valid)
        for k in iv:
            prev = iv[iv < k]
            if len(prev) >= n:
                w = s[prev[-n:]]
                zr[k], hr[k] = (s[k] - w.mean()) / max(w.std(ddof=1), S.SIGMA_FLOOR), True
        check(f"z-score = brute force over the {n} most recent valid samples (W {W})",
              np.array_equal(has, hr) and np.allclose(z[has], zr[hr], rtol=1e-9, atol=1e-9) and not has[~valid].any()
              and int(np.flatnonzero(has)[0]) == int(iv[n]), n_signals=int(has.sum()))
        k0 = int(np.flatnonzero(has)[len(np.flatnonzero(has)) // 2])
        s2 = s.copy()
        s2[k0] += 1e-3
        z2, dev2, _ = S.zscore(s2, valid, W)
        check(f"the window leaves t out (W {W})", abs((s2[k0] - dev2[k0]) - (s[k0] - dev[k0])) < 1e-15
              and abs(dev2[k0] / z2[k0] - dev[k0] / z[k0]) < 1e-15 and abs(dev2[k0] - dev[k0] - 1e-3) < 1e-12)
        s3 = s.copy()
        s3[k0 + 1:] += rng.normal(0, 1e-3, K - k0 - 1)
        z3, dev3, has3 = S.zscore(s3, valid, W)
        check(f"later bars do not change the signal at t (W {W})",
              np.array_equal(z3[:k0 + 1], z[:k0 + 1], equal_nan=True) and np.array_equal(has3, has) and not np.array_equal(z3, z, equal_nan=True))
    lo2, lo05 = np.nextafter(2.0, 0.0), np.nextafter(0.5, 1.0)
    z = np.array([2.0, lo2, -2.0, -lo2, 3.0, 0.5, lo05, -0.5, 0.0, 5.0])
    has = np.array([True] * 9 + [False])
    side, exz = S.decide(z, z * 1e-4, np.zeros(10), has, 2.0, 0.0)
    check("boundary: |z| = z_in enters, just below does not; |z| = 0.5 exits, just above does not; no signal, no flag",
          side.tolist() == [1, 0, -1, 0, 1, 0, 0, 0, 0, 0] and exz.tolist() == [False] * 5 + [True, False, True, True, False])
    side3, _ = S.decide(np.array([3.0, np.nextafter(3.0, 0.0), -3.0]), np.zeros(3), np.zeros(3), np.ones(3, bool), 3.0, 0.0)
    check("boundary at z_in = 3", side3.tolist() == [1, 0, -1])
    dev = np.array([2e-4, 2e-4, 2e-4, -2e-4, -2e-4])
    rs = np.array([2e-4, np.nextafter(2e-4, 1.0), 1e-4, 2e-4, np.nextafter(2e-4, 1.0)])
    zz = np.where(dev > 0, 5.0, -5.0)
    k1, _ = S.decide(zz, dev, rs, np.ones(5, bool), 2.0, 1.0)
    k0_, _ = S.decide(zz, dev, rs, np.ones(5, bool), 2.0, 0.0)
    check("kappa: |s - m| >= kappa x (relative spreads) is required, kappa 0 always passes",
          k1.tolist() == [1, 0, 1, -1, 0] and k0_.tolist() == [1, 1, 1, -1, -1])
    sf = np.zeros(K)
    sf[2000] = 3e-4
    zf, _, hf = S.zscore(sf, np.ones(K, bool), 10)
    alt = np.where(np.arange(K) % 2 == 0, 2e-4, -2e-4)
    za, da, _ = S.zscore(alt, np.ones(K, bool), 10)
    check("sigma floor: a flat window gives sigma = 1 bp; a wider window is not floored",
          abs(zf[2000] - 3.0) < 1e-9 and np.isfinite(zf[hf]).all() and abs(da[2000] / za[2000] - np.std(alt[1880:2000], ddof=1)) < 1e-15
          and np.std(alt[1880:2000], ddof=1) > S.SIGMA_FLOOR)
    for W, close in ((10, N), (30, N), (10, 12_600)):
        gg = S.grid(close)
        lo, hi = S.entry_bars(W, close)
        klo, khi = int(np.flatnonzero(gg == lo)[0]), int(np.flatnonzero(gg == hi)[0])
        sv = np.zeros(len(gg))
        for k in (klo, klo + 1, khi, khi + 1):
            sv[k] = 6e-4
        Bm = {"g": gg, "valid": np.ones(len(gg), bool), "s": sv, "rs": np.zeros(len(gg))}
        sd, _ = S.flags(Bm, W, 2.0, 0.0, close)
        check(f"entry hours (W {W}, close bar {close}): no signal before the window is full, first entry 09:30:05 + W min, "
              f"last entry at close - 30 min", sd[klo] == 0 and sd[klo + 1] == 1 and sd[khi] == 1 and sd[khi + 1] == 0
              and int(np.abs(sd).sum()) == 2 and klo == W * 12 - 1)


# ---------------------------------------------------------------- pair engine
def pair_tests():
    b = leg()
    for name, ticks, sd in (("up", 10, 1), ("down", -10, -1)):
        a = spike_leg(ticks=ticks)
        tr, B, side, exz = run(a, b)
        q = int(S.shares(2 * (BID + ticks * 100) + SPR))
        ea = [0, 0, q, q * (BID + ticks * 100 + (0 if sd > 0 else SPR)), 0, 0, 0]       # z > 0: sell A at the bid
        eb = [0, 0, 500, 500 * (BID + (SPR if sd > 0 else 0)), 0, 0, 0]                 # z > 0: buy B at the ask
        xa = [0, 0, q, q * (BID + (SPR if sd > 0 else 0)), 0, 0, 0]
        xb = [0, 0, 500, 500 * (BID + (0 if sd > 0 else SPR)), 0, 0, 0]
        mny = PE.money(tr, 1)
        ok = (len(tr) == 1 and tr[0, :6].tolist() == [GE, GE + 30, sd, q, 500, S.R_Z] and od(tr, 0, 0).tolist() == ea
              and od(tr, 0, 1).tolist() == eb and od(tr, 0, 2).tolist() == xa and od(tr, 0, 3).tolist() == xb and tr[0, PE.XM] == PE.M)
        check(f"direction ({name}): z > 0 sells A and buys B, z < 0 the reverse; exit by |z| <= 0.5 when the gap closes", ok,
              row=tr[0][:6], z_sign=int(side[np.flatnonzero(B['g'] == GE)[0]]))
        check(f"money ({name}): pnl 24 = mid 50 - cost 20 - fee 6 (2,000 market shares x 0.3 cents)",
              q == 500 and np.allclose([mny["pnl"][0], mny["mid"][0], mny["cost"][0], mny["fee"][0]], [24.0, 50.0, 20.0, 6.0], atol=1e-9),
              got=[float(mny[k][0]) for k in ("pnl", "mid", "cost", "fee")])
        check(f"one position at a time ({name}): {int((side != 0).sum())} entry flags, one trade", (side != 0).sum() >= 2 and len(tr) == 1)
    a = spike_leg()
    trp, _, _, _ = run(a, b, method=PE.P)
    mny = PE.money(trp, 1)
    dl = [0, 0, 500, None, 0, 30, 500]
    ok = (len(trp) == 1 and trp[0, :6].tolist() == [GE, GE + 30, 1, 500, 500, S.R_Z] and trp[0, PE.XM] == PE.P
          and all(od(trp, 0, j)[[0, 2, 4, 5, 6]].tolist() == [0, 500, 0, 30, 500] for j in range(4))
          and od(trp, 0, 0)[3] == 500 * BID and od(trp, 0, 1)[3] == 500 * (BID + SPR)       # deadline: A back at 100.00 bid
          and od(trp, 0, 2)[3] == 500 * (BID + SPR) and od(trp, 0, 3)[3] == 500 * BID)
    check("P without fills: both entry orders go to market at 30 s, the exit is decided when they are done, again 30 s", ok,
          row=trp[0][:6], pnl=float(mny["pnl"][0]))
    check("P without fills: pnl -26 = mid 50 (decision-time mids) - cost 70 (the waiting included) - fee 6",
          np.allclose([mny["pnl"][0], mny["mid"][0], mny["cost"][0], mny["fee"][0]], [-26.0, 50.0, 70.0, 6.0], atol=1e-9),
          got=[float(mny[k][0]) for k in ("pnl", "mid", "cost", "fee")])
    check("P statistics without fills", PE.fills(trp, PE.P).tolist() == [2, 0, 4, 4, 2000, 2000] and PE.fills(trp, PE.M).tolist() == [0] * 6)
    # exit decisions wait for both entry orders
    a2 = spike_leg(g1=GE + 10)
    trm, _, _, _ = run(a2, b)
    trp2, _, _, _ = run(a2, b, method=PE.P)
    check("exit checks start when both entry orders are done: M exits at the next grid point, P not before 30 s",
          len(trm) == 1 and trm[0, 1] == GE + 10 and len(trp2) == 1 and trp2[0, 1] == GE + 30, m_exit=int(trm[0, 1]), p_exit=int(trp2[0, 1]))
    # invalid quotes: no decision at that grid point
    bl = setq(leg(), GE + 30, GE + 31, BID, BID)                                          # leg B locked at the exit bar only
    tri, _, _, _ = run(spike_leg(), bl)
    bl2 = setq(leg(), GE, GE + 1, BID, BID)                                               # locked at the entry bar only
    tre, _, _, _ = run(spike_leg(), bl2)
    check("no decision at a grid point with a locked quote: the exit and the entry move to the next valid point",
          len(tri) == 1 and tri[0, 1] == GE + 35 and len(tre) == 1 and tre[0, 0] == GE + 5, exit_bar=int(tri[0, 1]), entry_bar=int(tre[0, 0]))
    # 30-minute exit
    ar = ramp(leg(), GE, 20, GE + 1800 + 60)
    setq(ar, GE + 1800 + 60, N, BID + (20 + 372) * 100)
    for nm, meth in (("M", PE.M), ("P", PE.P)):
        trt, B, side, exz = run(ar, b, method=meth)
        check(f"time exit ({nm}): 30 minutes after the entry decision, same method as the entry",
              len(trt) >= 1 and trt[0, :3].tolist() == [GE, GE + 1800, 1] and trt[0, 5] == S.R_TIME and trt[0, PE.XM] == meth, row=trt[0][:6])
    # forced exit at 15:55:00, always market
    g_late = 21_594                                                                       # 15:29:55
    af = ramp(leg(), g_late, 20, N)
    for nm, meth in (("M", PE.M), ("P", PE.P)):
        trf, _, _, _ = run(af, b, method=meth)
        F = S.force_bar(N)
        qa = int(trf[0, 3])
        ok = (len(trf) == 1 and trf[0, :3].tolist() == [g_late, F, 1] and trf[0, 5] == S.R_CLOSE and trf[0, PE.XM] == PE.M
              and od(trf, 0, 2).tolist() == [0, 0, qa, qa * af["ask"][F], 0, 0, 0] and od(trf, 0, 3).tolist() == [0, 0, 500, 500 * BID, 0, 0, 0])
        check(f"forced exit ({nm} entries): at 15:55:00, both legs by market order at that bar's quote", ok, row=trf[0][:6])
    # P exit decided shortly before 15:55: its deadline is cut to 15:55:00
    F = S.force_bar(N)
    at = ramp(leg(), g_late, 20, F - 10)
    setq(at, F - 10, N, int(at["bid"][F - 11]) - 60 * 100)
    trc, B, side, exz = run(at, b, method=PE.P)
    qa = int(trc[0, 3])
    ok = (len(trc) == 1 and trc[0, 1] == F - 10 and trc[0, 5] == S.R_Z and trc[0, PE.XM] == PE.P
          and od(trc, 0, 2).tolist() == [0, 0, qa, qa * at["ask"][F], 0, 10, qa] and od(trc, 0, 3).tolist() == [0, 0, 500, 500 * BID, 0, 10, 500])
    check("a P exit decided 10 s before 15:55 sends the rest to market at 15:55:00 (deadline cut from 30 s to 10 s)", ok, row=trc[0][:6])
    # early close
    ae = ramp(leg(), 10_794, 20, N)
    tre_, _, _, _ = run(ae, b, close_bar=12_600)
    al = ramp(leg(), 10_804, 20, N)
    trl, _, _, _ = run(al, b, close_bar=12_600)
    check("early close (13:00): last entry 12:30:00, forced exit 12:55:00",
          len(tre_) == 1 and tre_[0, :3].tolist() == [10_794, 12_299, 1] and tre_[0, 5] == S.R_CLOSE and len(trl) == 0)
    # no valid grid point from 15:55 to the close (ADDENDUM B1): valued at the mids of the last valid grid point, no order
    bz = setq(leg(), F - 2, N, BID, BID)
    for delay in (0, 1):
        for nm, meth in (("M", PE.M), ("P", PE.P)):
            trz, _, _, _ = run(af, bz, method=meth, delay=delay)
            qa, m2a = int(trz[0, 3]), int(af["bid"][F - 5] + af["ask"][F - 5])
            mz = PE.money(trz, 1)
            check(f"no valid point from 15:55 to the close ({nm}, delay {delay}): valued at the mids of the last valid grid point "
                  f"(15:54:55), no order, no fee",
                  len(trz) == 1 and trz[0, 1] == N - 1 and trz[0, 5] == S.R_LAST and trz[0, PE.XM] == PE.VALUED
                  and od(trz, 0, 2).tolist() == [qa, qa * m2a // 2, 0, 0, 0, 0, 0] and od(trz, 0, 3).tolist() == [500, 500 * (BID + SPR // 2), 0, 0, 0, 0, 0]
                  and trz[0, PE.MID0 + 2:PE.MID0 + 4].tolist() == [m2a, 2 * BID + SPR]
                  and abs(mz["fee"][0] - 0.003 * (od(trz, 0, 0)[2] + od(trz, 0, 1)[2])) < 1e-9, row=trz[0][:6])
    # delay check
    ad = spike_leg()
    setq(ad, GE + 1, GE + 2, BID + 900)                                                   # the quote one second after the decision
    trd, _, _, _ = run(ad, b, delay=1)
    trd0, _, _, _ = run(ad, b, delay=0)
    check("delay 1: orders start one bar after the decision and use that bar's quote; sizes and decision mids stay at t",
          len(trd) == 1 and od(trd, 0, 0)[3] == 500 * (BID + 900) and od(trd0, 0, 0)[3] == 500 * (BID + 1000)
          and trd[0, :6].tolist() == trd0[0, :6].tolist() and trd[0, PE.MID0:PE.MID0 + 4].tolist() == trd0[0, PE.MID0:PE.MID0 + 4].tolist())
    # depth penalty and fees
    asm, bsm = spike_leg(), leg(size=200)
    asm["bsz"][:], asm["asz"][:] = 200, 200
    trs, _, _, _ = run(asm, bsm)
    m1, m0 = PE.money(trs, 1), PE.money(trs, 0)
    check("D1: market shares above the displayed size cost one tick more (4 orders x 300 shares x 1 cent = 12 dollars)",
          all(od(trs, 0, j)[4] == 300 for j in range(4)) and abs(m0["pnl"][0] - 24.0) < 1e-9 and abs(m1["pnl"][0] - 12.0) < 1e-9
          and abs(m1["cost"][0] - 32.0) < 1e-9 and abs(m1["fee"][0] - 6.0) < 1e-9)
    for nm, both in (("both legs", True), ("leg A only", False)):
        ap, bp = spike_leg(), leg()
        ap["bid_max"][GE + 3] = BID + 1000 + SPR                                          # the bid reaches my sell limit at the ask
        if both:
            bp["ask_min"][GE + 3] = BID                                                    # the ask reaches my buy limit at the bid
        for lr in (3, 0):
            trq, _, _, _ = run(ap, bp, method=PE.P, lrule=lr)
            mq = PE.money(trq, 1)
            fa = od(trq, 0, 0).tolist() == [500, 500 * (BID + 1000 + SPR), 0, 0, 0, 3, 0]
            fb = od(trq, 0, 1).tolist() == ([500, 500 * BID, 0, 0, 0, 3, 0] if both else [0, 0, 500, 500 * (BID + SPR), 0, 30, 500])
            fee = 0.003 * (1000 if both else 1500)
            st = PE.fills(trq, PE.P).tolist()
            check(f"P limit fills ({nm}, L{lr}): limit shares pay no fee, market shares 0.3 cents; one-sided count",
                  len(trq) == 1 and fa and fb and abs(mq["fee"][0] - fee) < 1e-9 and trq[0, 1] == GE + 30
                  and st == ([2, 0, 4, 2, 2000, 1000] if both else [2, 1, 4, 3, 2000, 1500])
                  and abs(mq["pnl"][0] - (mq["mid"][0] - mq["cost"][0] - mq["fee"][0])) < 1e-9, fee=float(mq["fee"][0]), stats=st)


def kernel_tests():
    s, q = 1000, 500
    for lr in (0, 3):
        for d, nm in ((1, "buy"), (-1, "sell")):
            near, far = ("bid", "ask") if d > 0 else ("ask", "bid")
            # (i) nothing fills: market at the 30 s bar with that bar's quote, not the 29 s one
            b = leg()
            setq(b, s + 29, s + 30, BID - 300 * (d < 0), BID + SPR + 300 * (d > 0))
            setq(b, s + 30, s + 31, BID - 500 * (d < 0), BID + SPR + 500 * (d > 0))
            o = one(b, PE.P, lr, d, s, PE.HOR_P, q)
            px30 = (BID + SPR + 500) if d > 0 else (BID - 500)
            check(f"30 s deadline ({nm}, L{lr}): the rest goes to market at 30 s at that bar's quote",
                  o[:7].tolist() == [0, 0, q, q * px30, 0, 30, q], out=o[:7])
            # (ii) the limit is still working at 29 s: the opposite quote reaches it in bar s + 29
            b = leg()
            if d > 0:
                b["ask_min"][s + 29] = BID
            else:
                b["bid_max"][s + 29] = BID + SPR
            o = one(b, PE.P, lr, d, s, PE.HOR_P, q)
            lim = BID if d > 0 else BID + SPR
            check(f"29 s ({nm}, L{lr}): the limit order is still working and fills in the 29th second",
                  o[:7].tolist() == [q, q * lim, 0, 0, 0, 29, 0], out=o[:7])
            # (iii) the same touch one second after the deadline changes nothing
            b = leg()
            if d > 0:
                b["ask_min"][s + 31] = BID
            else:
                b["bid_max"][s + 31] = BID + SPR
            o = one(b, PE.P, lr, d, s, PE.HOR_P, q)
            check(f"after the deadline ({nm}, L{lr}): a later touch does not matter",
                  o[:7].tolist() == [0, 0, q, q * ((BID + SPR) if d > 0 else BID), 0, 30, q], out=o[:7])
            # limit price: one tick inside when the spread is >= 3 ticks, else at the best quote
            for spread, inside in ((400, True), (300, True), (200, False)):
                lim = (BID + 100 if d > 0 else BID + spread - 100) if inside else (BID if d > 0 else BID + spread)
                res = []
                for touch in (lim, lim + 100 * d):                                # at my price / one tick short of it
                    b = leg(spread=spread)
                    if d > 0:
                        b["ask_min"][s + 3] = touch
                    else:
                        b["bid_max"][s + 3] = touch
                    res.append(one(b, PE.P, lr, d, s, PE.HOR_P, q)[:7].tolist())
                mk = (BID + spread) if d > 0 else BID
                check(f"limit price ({nm}, L{lr}, spread {spread // 100} ticks): {'one tick inside' if inside else 'at the best quote'}",
                      res[0] == [q, q * lim, 0, 0, 0, 3, 0] and res[1] == [0, 0, q, q * mk, 0, 30, q], out=res)
            o = one(leg(), PE.M, lr, d, s, PE.HOR_P, q)
            check(f"M ({nm}, L{lr}): the whole order at the start bar's quote", o[:7].tolist() == [0, 0, q, q * ((BID + SPR) if d > 0 else BID), 0, 0, 0])
    # the wrapper against engine17.run on random bars with trades
    rng = np.random.default_rng(2020)
    b = leg()
    path = BID + 100 * np.cumsum(rng.integers(-1, 2, N))
    spr = 100 * rng.integers(1, 5, N)
    b["bid"], b["ask"] = path.copy(), path + spr
    b["bid_max"], b["bid_min"] = b["bid"] + 100 * rng.integers(0, 2, N), b["bid"] - 100 * rng.integers(0, 2, N)
    b["ask_min"], b["ask_max"] = b["ask"] - 100 * rng.integers(0, 3, N), b["ask"] + 100 * rng.integers(0, 2, N)
    b["bsz"], b["asz"] = rng.integers(100, 2000, N), rng.integers(100, 2000, N)
    b["vol_bid"], b["vol_ask"] = rng.integers(0, 800, N), rng.integers(0, 800, N)
    b["vol_bid_odd"], b["vol_ask_odd"] = b["vol_bid"] // 7, b["vol_ask"] // 7
    b["px_min"], b["px_max"] = b["bid_min"], b["ask_max"]
    dirs, starts = G.parents("20200102", N)
    same = {}
    for name, meth in (("S0", PE.M), ("S2i", PE.P)):
        for lr in (0, 3):
            ref = G.run(b, name, lr, dirs, starts)
            out = np.zeros_like(ref)
            PE.run_parents(PE.stack(b), meth, lr, dirs, starts, G.HOR, G.QTY, out)
            same[f"{name} L{lr}"] = bool(np.array_equal(ref, out)) and bool((ref[:, 0] > 0).any() or name == "S0")
    check("the single-order path equals engine17.run on random bars (S0, S2i; L0, L3)", all(same.values()), same=same)


def timetable_tests():
    """A random-walk day: the M engine follows the signal-only timetable; P respects the order of events."""
    rng = np.random.default_rng(7)
    a, b = leg(), leg()
    path = BID + 100 * np.cumsum(rng.integers(-1, 2, N))
    for k in ("bid", "bid_max", "bid_min"):
        a[k] = path.copy()
    for k in ("ask", "ask_max", "ask_min"):
        a[k] = path + SPR
    lock = rng.random(N) < 0.003
    a["ask"][lock] = a["bid"][lock]                                                       # some locked bar ends
    a["ask_min"] = np.minimum(a["ask_min"], a["ask"])
    a["bid_max"] += 100 * (rng.random(N) < 0.05)                                         # touches that fill resting sell limits
    a["ask_min"] -= 100 * (rng.random(N) < 0.05)
    a["ask_min"] = np.maximum(a["ask_min"], a["bid"] - 100)
    B = S.base(a, b, N)
    XA, XB = PE.stack(a), PE.stack(b)
    ok_mid, ok_seq, ok_sh, ok_id, n_tr = True, True, True, True, 0
    for W, zi, ka in ST.configs():
        side, exz = S.flags(B, W, zi, ka, N)
        mt = S.mid_trades(B, side, exz, N)
        trm = PE.trades(PE.M, 3, 0, B, side, exz, XA, XB, N)
        ok_mid &= np.array_equal(mt[:, :6], trm[:, :6]) and np.allclose(S.mid_pnl(mt), PE.money(trm, 1)["mid"], atol=1e-9)
        ok_mid &= np.array_equal(mt[:, 6:8], trm[:, PE.MID0:PE.MID0 + 2]) and np.array_equal(mt[:, 8:10], trm[:, PE.MID0 + 2:PE.MID0 + 4])
        n_tr += len(trm)
        for meth in (PE.M, PE.P):
            for lr in (3, 0):
                for delay in (0, 1):
                    tr = PE.trades(meth, lr, delay, B, side, exz, XA, XB, N)
                    if not len(tr):
                        continue
                    d_e = np.maximum(tr[:, PE.O0 + 5], tr[:, PE.O0 + PE.OCOL + 5])
                    d_x = np.maximum(tr[:, PE.O0 + 2 * PE.OCOL + 5], tr[:, PE.O0 + 3 * PE.OCOL + 5])
                    ok_seq &= bool(np.all(tr[:, 1] > tr[:, 0]) and np.all(tr[:, 1] >= tr[:, 0] + delay + d_e)
                                   and np.all(tr[1:, 0] > tr[:-1, 1]) and np.all(tr[1:, 0] >= tr[:-1, 1] + delay + d_x[:-1])
                                   and np.all(B["valid"][(tr[:, 0] - 4) // 5]) and np.all(tr[:, 1] - tr[:, 0] <= 1800 + 30)
                                   and np.all(d_e <= 30) and np.all(d_x <= 30))
                    for j in range(4):
                        o = tr[:, PE.O0 + j * PE.OCOL:PE.O0 + (j + 1) * PE.OCOL]
                        ok_sh &= bool(np.all(o[:, 0] + o[:, 2] == tr[:, PE.ORDER_QCOL[j]]) and np.all(o[:, 6] <= o[:, 2]))
                    for dep in (0, 1):
                        m = PE.money(tr, dep)
                        ok_id &= bool(np.allclose(m["pnl"], m["mid"] - m["cost"] - m["fee"], atol=1e-9))
    check("random day: the M engine's trades equal the signal-only timetable (bars, side, shares, reasons, mids)", ok_mid, m_trades=n_tr)
    U = 12_004                                                                            # change every bar after 12:50:05
    a2 = {k: v.copy() for k, v in a.items()}
    for k in ("bid", "bid_max", "bid_min", "ask", "ask_max", "ask_min"):
        a2[k][U + 1:] += 700
    B2, XA2 = S.base(a2, b, N), PE.stack(a2)
    kU = int(np.flatnonzero(B["g"] == U)[0])
    ok_fut, n_cmp = True, 0
    for W, zi, ka in ST.configs():
        side, exz = S.flags(B, W, zi, ka, N)
        side2, exz2 = S.flags(B2, W, zi, ka, N)
        ok_fut &= np.array_equal(side[:kU + 1], side2[:kU + 1]) and np.array_equal(exz[:kU + 1], exz2[:kU + 1])
        for meth in (PE.M, PE.P):
            t1 = PE.trades(meth, 3, 0, B, side, exz, XA, XB, N)
            t2 = PE.trades(meth, 3, 0, B2, side2, exz2, XA2, XB, N)
            keep1, keep2 = t1[t1[:, 1] + 30 <= U], t2[t2[:, 1] + 30 <= U]
            ok_fut &= np.array_equal(keep1, keep2)
            n_cmp += len(keep1)
    check("random day: changing every bar after t leaves the flags up to t and the trades finished by t unchanged", ok_fut and n_cmp > 0,
          trades_compared=n_cmp)
    check("random day: exits after both entry orders, entries after both exit orders, valid entry points, at most 30 min + 30 s", ok_seq)
    check("random day: limit + market shares = order size for every order", ok_sh)
    check("random day: pnl = mid - cost - fee", ok_id)


def stats_tests():
    idx = ST.boot_index(250)
    check("bootstrap index: 10,000 x T, blocks of 21 days, seed 20, reproducible",
          idx.shape == (10_000, 250) and idx.min() >= 0 and idx.max() < 250 and np.array_equal(idx, ST.boot_index(250))
          and np.all((idx[:, 1:21] - idx[:, :20]) % 250 == 1) and np.array_equal(idx, G.boot_index(250, B=10_000, block=21, seed=20)))
    x = np.full(250, 3.0)
    check("constant daily series: mean = CI ends; p = 0 when positive, 1 when zero or negative",
          ST.mean_ci(x, idx) == (3.0, 3.0, 3.0) and ST.p_positive(x, idx)[1] == 0.0 and ST.p_positive(-x, idx)[1] == 1.0
          and ST.p_positive(0 * x, idx)[1] == 1.0)
    rng = np.random.default_rng(20)
    y = rng.normal(0.0, 1.0, 250)
    y += 0.1 - y.mean()                                                                  # sample mean exactly +0.1
    bm = y[idx].mean(axis=1)
    m, p, lb = ST.p_positive(y, idx)
    check("one-sided p = share of bootstrap means <= 0; lower bound = 2.5th percentile",
          abs(p - np.mean(bm <= 0)) < 1e-15 and abs(lb - np.percentile(bm, 2.5)) < 1e-15 and abs(m - y.mean()) < 1e-15 and 0 < p < 0.5, p=p, mean=m)
    check("daily sums: days without a trade are 0", ST.daily([0, 0, 3], [1.0, 2.0, 5.0], 5).tolist() == [3.0, 0.0, 0.0, 5.0, 0.0])
    num, den = ST.daily([0, 1, 1, 3], [2.0, 4.0, 6.0, -1.0], 250), ST.daily([0, 1, 1, 3], [1, 1, 1, 1], 250)
    r, lo, hi = ST.ratio_ci(num, den, idx)
    check("per-trade mean = sum of daily sums / number of trades", abs(r - 11.0 / 4) < 1e-12 and lo <= r <= hi)
    check("Sharpe = mean / sd(ddof 1) x sqrt(252)", abs(ST.sharpe(y) - y.mean() / y.std(ddof=1) * np.sqrt(252)) < 1e-12)
    rows = [{"name": "a", "mean": 100.0, "trades": 1000}, {"name": "b", "mean": 96.0, "trades": 500}, {"name": "c", "mean": 94.0, "trades": 10}]
    neg = [{"name": "a", "mean": -10.0, "trades": 100}, {"name": "b", "mean": -10.4, "trades": 50}, {"name": "c", "mean": -10.6, "trades": 1}]
    same = [{"name": "a", "mean": 5.0, "trades": 7}, {"name": "b", "mean": 5.0, "trades": 7}]
    sa, sn, ss = ST.select_astar(rows), ST.select_astar(neg), ST.select_astar(same)
    check("A*: highest mean; within 5% of it the fewest trades, then the earlier candidate; a mean <= 0 means no profit in development",
          sa[0] == "b" and sa[1]["tied_within_5pct"] == ["a", "b"] and sa[1]["profitable_in_development"] and sn[0] == "b"
          and sn[1]["tied_within_5pct"] == ["a", "b"] and not sn[1]["profitable_in_development"] and ss[0] == "a")
    h2, h1, hs = ST.holm({"H1": 0.01, "H2": 0.04}), ST.holm({"H1": 0.03, "H2": 0.04}), ST.holm({"H1": 0.04})
    check("Holm: 0.025 then 0.05 with two hypotheses, 0.05 with one",
          h2["H1"]["rejected"] and h2["H2"]["rejected"] and h2["H1"]["level"] == 0.025 and h2["H2"]["level"] == 0.05
          and not h1["H1"]["rejected"] and not h1["H2"]["rejected"] and hs["H1"]["rejected"] and hs["H1"]["level"] == 0.05)
    w = rng.normal(0.0, 2.0, 1500)
    idw = ST.boot_index(1500)
    pw = ST.mde(w, idw, 1470)
    from scipy.stats import norm
    want = (norm.ppf(0.975) + norm.ppf(0.8)) * np.sqrt(w.var(ddof=1) * pw["VIF"] / 1470)
    check("MDE = (z_{1-a} + z_{0.8}) x sqrt(V x VIF / days); VIF near 1 for independent days",
          abs(pw["mde"]["0.025"] - want) < 1e-12 and 0.7 < pw["VIF"] < 1.3 and pw["mde"]["0.05"] < pw["mde"]["0.025"], VIF=pw["VIF"])
    names = [ST.cand_name(c, ex) for c, ex in ST.candidates()]
    check("16 candidates: W x z_in x kappa x execution", len(names) == 16 and len(set(names)) == 16 and names[0] == "W10_z2_k0_M"
          and names[-1] == "W30_z3_k1_P" and ST.DEFAULT_SIGNAL in ST.configs(), names=names)


# ---------------------------------------------------------------- step 2b: quote gaps (SPEC20_ADDENDUM B1)
def lock(b, u0, u1, px=None):
    """Bars u0..u1-1 with a locked quote (ask = bid) at px (default: the bar's own bid). In these bars a quote state is
    carried forward once it exists, so a quote that disappears during the day can only show up as a locked or a stale quote."""
    v = b["bid"][u0:u1].copy() if px is None else px
    for k in ("bid", "bid_max", "bid_min", "ask", "ask_max", "ask_min"):
        b[k][u0:u1] = v
    return b


def nostate(b, u0, u1):
    for k in PE.FIELDS:
        b[k][u0:u1] = 0
    return b


def both(a, b, method=PE.M, W=10, close_bar=N):
    """(trades of the method, signal-only trades, True when the M engine and the signal-only timetable agree, grid arrays)."""
    B = S.base(a, b, close_bar)
    side, exz = S.flags(B, W, 2.0, 0.0, close_bar)
    XA, XB = PE.stack(a), PE.stack(b)
    trm = PE.trades(PE.M, 3, 0, B, side, exz, XA, XB, close_bar)
    tr = trm if method == PE.M else PE.trades(method, 3, 0, B, side, exz, XA, XB, close_bar)
    mt = S.mid_trades(B, side, exz, close_bar)
    same = (np.array_equal(mt[:, :6], trm[:, :6]) and np.array_equal(mt[:, 6:8], trm[:, PE.MID0:PE.MID0 + 2])
            and np.array_equal(mt[:, 8:10], trm[:, PE.MID0 + 2:PE.MID0 + 4]))
    return tr, mt, bool(same), B


def gap_tests():
    F = S.force_bar(N)
    # 1. one leg without a quote for the first 605 seconds (the sealed case; never in the development bars)
    for nm, which in (("leg A", 0), ("leg B", 1)):
        a, b = setq(leg(), 1209, 1239, BID + 1000), leg()
        nostate((a, b)[which], 0, 605)
        tr, mt, same, B = both(a, b)
        _, _, has = S.zscore(B["s"], B["valid"], 10)
        a0, b0 = setq(leg(), 1204, 1209, BID + 1000), leg()                         # the same gap, a spike one grid point earlier
        nostate((a0, b0)[which], 0, 605)
        tr0, _, same0, _ = both(a0, b0)
        res = C.pairday_all(a, b, N, [(10, 2.0, 0.0)])
        check(f"gap at the open ({nm} without a quote for 605 s): no decision before the first valid point (09:40:10), first signal "
              f"after W x 12 valid samples (09:50:10), H1 on the same timetable",
              C.state_ok(a) and C.state_ok(b) and int(B["g"][np.argmax(B["valid"])]) == 609 and int(B["g"][np.argmax(has)]) == 1209
              and len(tr) == 1 and tr[0, :2].tolist() == [1209, 1239] and same and len(tr0) == 0 and same0 and res[5] == 0
              and res[0][0, 0, 0] == 1 and res[0][0, 1, 0] == 1, first_valid_bar=int(B["g"][np.argmax(B["valid"])]),
              first_signal_bar=int(B["g"][np.argmax(has)]), trade=tr[0][:6])
    # 2. a gap during the day on one leg: 30 seconds and 10 minutes (as a locked quote)
    for secs in (30, 600):
        a, b = setq(leg(), GE, GE + secs + 100, BID + 1000), lock(leg(), GE - 2, GE - 2 + secs)
        tr, mt, same, B = both(a, b)
        k0, k1 = int(np.flatnonzero(B["g"] == GE)[0]), int(np.flatnonzero(B["g"] == GE + secs)[0])
        check(f"gap of {secs} s while flat: no entry at the grid points inside the gap, entry at the first valid grid point after it "
              f"(z from the W x 12 valid samples before the gap), H1 on the same timetable",
              not B["valid"][k0:k1].any() and B["valid"][k1] and len(tr) >= 1 and tr[0, 0] == GE + secs and same, entry_bar=int(tr[0, 0]))
        a, b = spike_leg(), lock(leg(), GE + 28, GE + 28 + secs)
        for nm, meth in (("M", PE.M), ("P", PE.P)):
            tr, mt, same, B = both(a, b, method=meth)
            check(f"gap of {secs} s while a position is open ({nm}): no exit decision inside the gap, exit at the first valid grid point "
                  f"after it by the usual rule (here |z| <= 0.5), H1 on the same timetable",
                  len(tr) == 1 and tr[0, :2].tolist() == [GE, GE + 30 + secs] and tr[0, 5] == S.R_Z and tr[0, PE.XM] == meth and same
                  and mt[0, 1] == GE + 30 + secs, exit_bar=int(tr[0, 1]))
    # 3. no valid quote at the 30-minute point
    ar = ramp(leg(), GE, 20, GE + 1800 + 60)
    setq(ar, GE + 1800 + 60, N, BID + (20 + 372) * 100)
    bl = lock(leg(), GE + 1798, GE + 1801)
    for nm, meth in (("M", PE.M), ("P", PE.P)):
        tr, mt, same, B = both(ar, bl, method=meth)
        note = "by market order" if meth == PE.M else "with a P order, the entry's method (SPEC20_ANSWERS A3; SPEC20_ADDENDUM B1 as revised)"
        check(f"no valid quote at the 30-minute point ({nm}): exit at the next valid grid point (5 s later) {note}; H1 on the same timetable",
              tr[0, :2].tolist() == [GE, GE + 1805] and tr[0, 5] == S.R_TIME and tr[0, PE.XM] == meth and same and mt[0, 1] == GE + 1805,
              exit_bar=int(tr[0, 1]), exit_method=int(tr[0, PE.XM]))
    # 4. no valid quote at 15:55:00, and none up to the close
    g_late = 21_594
    af = ramp(leg(), g_late, 20, N)
    for nm, meth in (("M", PE.M), ("P", PE.P)):
        tr, mt, same, B = both(af, lock(leg(), F - 2, F + 3), method=meth)
        check(f"no valid quote at 15:55:00 ({nm} entries): forced exit by market order at the first valid grid point after it (15:55:05), "
              f"H1 on the same timetable", len(tr) == 1 and tr[0, :2].tolist() == [g_late, F + 5] and tr[0, 5] == S.R_CLOSE
              and tr[0, PE.XM] == PE.M and od(tr, 0, 3).tolist() == [0, 0, 500, 500 * BID, 0, 0, 0] and same, exit_bar=int(tr[0, 1]))
        tr, mt, same, B = both(af, lock(leg(), F - 2, N), method=meth)
        m2a = int(af["bid"][F - 5] + af["ask"][F - 5])
        res = C.pairday_all(af, lock(leg(), F - 2, N), N, [(10, 2.0, 0.0)])
        check(f"no valid quote from 15:55 to the close ({nm} entries): valued at the mids of the last valid grid point, counted, "
              f"H1 with the same value", len(tr) == 1 and tr[0, 1] == N - 1 and tr[0, 5] == S.R_LAST and tr[0, PE.XM] == PE.VALUED and same
              and mt[0, 5] == S.R_LAST and mt[0, 8:10].tolist() == [m2a, 2 * BID + SPR] and all(res[0][0, k, C.COL["r_last"]] == 1 for k in range(5))
              and res[1] == 0, exit_mid2=mt[0][8:10])
    # 5. a P order is working when the quote goes (kernel of research 17, unchanged)
    s0, q = 1000, 500
    for lr in (0, 3):
        o = one(lock(leg(), s0 + 2, s0 + 14, BID), PE.P, lr, 1, s0, PE.HOR_P, q)
        check(f"P buy order, the market locks at its limit price (the ask comes down to the bid, L{lr}): filled at the limit in that second",
              o[:7].tolist() == [q, q * BID, 0, 0, 0, 2, 0], out=o[:7])
        o = one(lock(leg(), s0 + 2, s0 + 14, BID + SPR), PE.P, lr, 1, s0, PE.HOR_P, q)
        check(f"P buy order, the market locks away from its limit for 12 s (L{lr}): no fill, the order is withdrawn at the next 5 s "
              f"decision bar, posted again when the spread is back, the rest to market at 30 s", o[:7].tolist() == [0, 0, q, q * (BID + SPR), 0, 30, q], out=o[:7])
        o = one(lock(leg(), s0 + 2, N, BID + 300), PE.P, lr, 1, s0, PE.HOR_P, q)
        check(f"P buy order, the market stays locked through the deadline (L{lr}): market order at 30 s at the locked price",
              o[:7].tolist() == [0, 0, q, q * (BID + 300), 0, 30, q], out=o[:7])
    a, b = spike_leg(), lock(leg(), GE + 3, GE + 13)
    tr, mt, same, B = both(a, b, method=PE.P)
    check("P entry, leg B locked from second 3 to 13 at its bid: the buy order of leg B is filled at its limit when the market locks "
          "there; decisions at the locked grid points are skipped; H1 on the same timetable",
          len(tr) == 1 and od(tr, 0, 1).tolist() == [500, 500 * BID, 0, 0, 0, 3, 0] and od(tr, 0, 0)[5] == 30 and tr[0, 1] == GE + 30 and same,
          entry_b=od(tr, 0, 1))
    # 6. a bar without a quote state after the first state cannot come out of bars1s/bars.py: the pair-day is left out
    a, b = spike_leg(), nostate(leg(), GE + 10, GE + 40)
    res = C.pairday_all(a, b, N, [(10, 2.0, 0.0)], keep=(0,), mid_delays=S.MID_DELAYS)
    check("a bar without a quote state after a leg's first state: the pair-day is left out whole and counted",
          not C.state_ok(b) and C.state_ok(a) and res[5] == 1 and not res[0].any() and not res[6].any() and len(res[2][0]["M"]) == 0
          and C.state_ok(nostate(leg(), 0, 605)) and C.state_ok(nostate(leg(), 0, N)))


# ---------------------------------------------------------------- step 2b: delayed mid prices (SPEC20_ADDENDUM B2)
def delay_tests():
    F = S.force_bar(N)
    b = leg()
    a = setq(leg(), GE, GE + 5, BID + 1000)                                              # a 5-second spike: one grid point
    tr, mt, same, B = both(a, b)
    v, m2a, m2b = S.bar_base(a, b, N)
    p = {d: float(S.mid_pnl_delayed(mt, v, m2a, m2b, d)[0]) for d in (0, 1, 5, 30)}
    check("delayed mid, 5-second spike: 50 dollars at delay 0 and 1 s, 0 at 5 s and 30 s (the entry price is taken after the reversal)",
          len(mt) == 1 and mt[0, :2].tolist() == [GE, GE + 5] and np.allclose([p[0], p[1], p[5], p[30]], [50.0, 50.0, 0.0, 0.0], atol=1e-9)
          and abs(p[0] - S.mid_pnl(mt)[0]) < 1e-12, pnl=p)
    a = setq(setq(leg(), GE, GE + 2, BID + 1000), GE + 2, GE + 5, BID + 800)
    bl = lock(leg(), GE + 1, GE + 2)
    tr, mt, same, B = both(a, bl)
    v, m2a, m2b = S.bar_base(a, bl, N)
    check("delayed mid, no valid quote 1 s after the decision: the first valid bar after it is used (2 s, 40 dollars instead of 50)",
          len(mt) == 1 and S.delayed_bars(v, mt[:, 0], 1).tolist() == [GE + 2] and S.delayed_bars(v, mt[:, 0], 0).tolist() == [GE]
          and abs(S.mid_pnl_delayed(mt, v, m2a, m2b, 1)[0] - 40.0) < 1e-9 and abs(S.mid_pnl_delayed(mt, v, m2a, m2b, 0)[0] - 50.0) < 1e-9)
    g_late = 21_594
    af = ramp(leg(), g_late, 20, N)
    bq = lock(leg(), F + 10, N)                                                           # nothing valid after 15:55:10
    tr, mt, same, B = both(af, bq)
    v, m2a, m2b = S.bar_base(af, bq, N)
    xb = S.delayed_bars(v, mt[:, 1], 30)
    want = float(mt[0, 3] * (m2a[g_late + 30] - m2a[F + 9]) + mt[0, 4] * (m2b[F + 9] - m2b[g_late + 30])) / 20_000.0
    check("delayed mid, no valid bar left in the session after the delay: the last valid bar of the session is used",
          mt[0, 1] == F and xb.tolist() == [F + 9] and abs(S.mid_pnl_delayed(mt, v, m2a, m2b, 30)[0] - want) < 1e-9)
    bz = lock(leg(), F - 2, N)
    tr, mt, same, B = both(af, bz)
    v, m2a, m2b = S.bar_base(af, bz, N)
    want = {d: float(mt[0, 3] * (m2a[g_late + d] - mt[0, 8]) + mt[0, 4] * (mt[0, 9] - m2b[g_late + d])) / 20_000.0 for d in (0, 1, 5, 30)}
    check("delayed mid, a trade valued at the last valid mids keeps that exit value for every delay",
          mt[0, 5] == S.R_LAST and all(abs(S.mid_pnl_delayed(mt, v, m2a, m2b, d)[0] - want[d]) < 1e-9 for d in want)
          and abs(want[0] - S.mid_pnl(mt)[0]) < 1e-9)
    # a random day with locked bar ends: delay 0 equals the H1 values; pairday_all sums the delays
    rng = np.random.default_rng(11)
    a = leg()
    path = BID + 100 * np.cumsum(rng.integers(-1, 2, N))
    for k in ("bid", "bid_max", "bid_min"):
        a[k] = path.copy()
    for k in ("ask", "ask_max", "ask_min"):
        a[k] = path + SPR
    lk = rng.random(N) < 0.004
    for k in ("ask", "ask_max", "ask_min"):
        a[k][lk] = a["bid"][lk]
    B = S.base(a, b, N)
    v, m2a, m2b = S.bar_base(a, b, N)
    ok0, okd, n = True, True, 0
    cfgs = ST.configs()
    res = C.pairday_all(a, b, N, cfgs, mid_delays=S.MID_DELAYS)
    for ci, (W, zi, ka) in enumerate(cfgs):
        side, exz = S.flags(B, W, zi, ka, N)
        mt = S.mid_trades(B, side, exz, N)
        n += len(mt)
        ok0 &= bool(np.allclose(S.mid_pnl_delayed(mt, v, m2a, m2b, 0), S.mid_pnl(mt), atol=1e-12))
        for di, d in enumerate(S.MID_DELAYS):
            pd_ = S.mid_pnl_delayed(mt, v, m2a, m2b, d)
            okd &= bool(res[6][ci, di, 0] == len(mt) and abs(res[6][ci, di, 1] - pd_.sum()) < 1e-9 and res[6][ci, di, 2] == (pd_ > 0).sum())
            eb = S.delayed_bars(v, mt[:, 0], d)
            okd &= bool(np.all(v[eb]) and np.all(eb >= mt[:, 0] + d))
    check("random day: delayed mid at delay 0 equals the H1 mid P&L for all 8 settings", ok0 and n > 0, trades=n)
    check("random day: pairday_all sums the delayed mid P&L (trades, dollars, wins) for delays 1, 5 and 30 s; delayed bars are valid "
          "and not earlier than the decision + delay", okd and res[1] == 0 and res[5] == 0)
    # the table and the sentence
    A = np.zeros((4, 1, len(C.KINDS), C.NAGG))
    D = np.zeros((4, 1, 3, 3))
    A[:, 0, 0, C.COL["n"]], A[:, 0, 0, C.COL["pnl"]], A[:, 0, 0, C.COL["wins"]] = [2, 1, 3, 2], [20.0, 10.0, 30.0, 20.0], [2, 1, 3, 1]
    D[:, 0, :, 0] = np.array([2, 1, 3, 2])[:, None]
    D[:, 0, 0, 1], D[:, 0, 1, 1], D[:, 0, 2, 1] = [10.0, 5.0, 15.0, 10.0], [4.0, 2.0, 6.0, 4.0], [-2.0, 1.0, -3.0, 0.0]
    D[:, 0, :, 2] = np.array([[1, 1, 0], [1, 0, 1], [2, 1, 0], [1, 1, 0]])
    M_ = {"day": np.array([0, 1, 1, 3]), "day_year": np.array([2020] * 5), "pair": np.zeros(4, np.int64), "names": ["x"]}
    rows = C.delayed_rows(A, D, M_, 0, (1, 5, 30), 5, ST.boot_index(5))
    check("delayed table: trades, dollars per trade, mean daily sum, ratio to delay 0, win rate",
          rows["0"]["trades"] == 8 and abs(rows["0"]["per_trade_usd"] - 10.0) < 1e-12 and abs(rows["0"]["mean_daily_usd"] - 16.0) < 1e-12
          and abs(rows["1"]["ratio_to_delay0"] - 0.5) < 1e-12 and abs(rows["5"]["ratio_to_delay0"] - 0.2) < 1e-12
          and abs(rows["30"]["mean_daily_usd"] + 0.8) < 1e-12 and abs(rows["1"]["win_rate"] - 5 / 8) < 1e-12
          and abs(rows["5"]["per_trade_usd"] - 2.0) < 1e-12 and rows["0"]["win_rate"] == 7 / 8, rows={k: v["mean_daily_usd"] for k, v in rows.items()})
    check("5-second sentence (ADDENDUM B2): lower end of the 95% interval above 0 -> the share that is left; otherwise not distinguishable from 0",
          S.delay_sentence(0.01, 0.4321) == "5초 늦게 거래해도 되돌림의 43%가 남았다" and S.delay_sentence(0.0, 0.9) == "5초 늦게 거래하면 되돌림은 0과 구분되지 않았다"
          and S.delay_sentence(-3.0, 0.2) == "5초 늦게 거래하면 되돌림은 0과 구분되지 않았다" and S.MID_DELAYS == (1, 5, 30))


def main():
    signal_tests()
    kernel_tests()
    pair_tests()
    timetable_tests()
    stats_tests()
    gap_tests()
    delay_tests()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    n_ok = sum(r["pass"] for r in RESULTS)
    with open(os.path.join(HERE, "results", "engine20_tests.json"), "w") as fh:
        json.dump({"passed": n_ok, "total": len(RESULTS), "numba": bool(G.HAVE_NUMBA), "tests": RESULTS}, fh, indent=1, default=str,
                  ensure_ascii=False)
    for r in RESULTS:
        if not r["pass"]:
            print("FAIL", json.dumps(r, default=str, ensure_ascii=False)[:600])
    print(f"{n_ok}/{len(RESULTS)} passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
