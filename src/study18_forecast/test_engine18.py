#!/usr/bin/env python3
"""Research 18 synthetic tests of engine18 (step 2; must pass before any development run). Synthetic bars only: a flat
$100.00 / $100.04 quote with no trades (a posted order never fills unless the quote crosses it), shifted from a chosen bar
on, and random bars for the equivalence checks (exec18 = study17_new_stocks exec17 with the rules off; R2 skipping every E1 =
S2i). Writes results/engine18_tests.json; exits 1 when a test fails.
  python test_engine18.py"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "study17_new_stocks"))
import engine18 as G  # noqa: E402
import engine17 as G17  # noqa: E402

BID, ASK, SZ, N, S = 1_000_000, 1_000_400, 5000, 1200, 50
FIELDS = ("bid", "ask", "bsz", "asz", "bid_max", "bid_min", "ask_max", "ask_min", "vol_bid", "vol_ask", "vol_bid_odd",
          "vol_ask_odd", "px_min", "px_max", "vol")
RESULTS = []


def bar(spread=400):
    b = {k: np.zeros(N, np.int64) for k in FIELDS}
    for k in ("bid", "bid_max", "bid_min"):
        b[k][:] = BID
    for k in ("ask", "ask_max", "ask_min"):
        b[k][:] = BID + spread
    b["bsz"][:], b["asz"][:] = SZ, SZ
    return b


def shift(b, u0, u1, dbid, dask=None):
    """Move the bid side by dbid and the ask side by dask (default dbid) on bars u0..u1-1."""
    dask = dbid if dask is None else dask
    for k in ("bid", "bid_max", "bid_min"):
        b[k][u0:u1] += dbid
    for k in ("ask", "ask_max", "ask_min"):
        b[k][u0:u1] += dask
    return b


def rand_bars(seed, spreads=(0, 1, 2, 3, 4, 6)):
    """Random but internally ordered bars (bid_min <= bid <= bid_max, same for the ask) with trades."""
    rng = np.random.default_rng(seed)
    mid = 1_000_000 + 100 * np.cumsum(rng.integers(-1, 2, N) * (rng.random(N) < 0.3))
    spr = 100 * rng.choice(spreads, N)
    b = {k: np.zeros(N, np.int64) for k in FIELDS}
    b["bid"] = (mid - spr // 2) // 100 * 100
    b["ask"] = b["bid"] + spr
    b["bid_max"] = b["bid"] + 100 * (rng.random(N) < 0.2)
    b["bid_min"] = b["bid"] - 100 * (rng.random(N) < 0.2)
    b["ask_max"] = b["ask"] + 100 * (rng.random(N) < 0.2)
    b["ask_min"] = b["ask"] - 100 * (rng.random(N) < 0.2)
    b["bsz"], b["asz"] = rng.integers(1, 30, N) * 100, rng.integers(1, 30, N) * 100
    b["vol_bid"], b["vol_ask"] = rng.integers(0, 8, N) * 100, rng.integers(0, 8, N) * 100
    b["vol_bid_odd"], b["vol_ask_odd"] = (rng.random(N) < 0.2) * 37, (rng.random(N) < 0.2) * 41
    b["px_min"], b["px_max"] = b["bid"] - 100, b["ask"] + 100
    b["vol"] = b["vol_bid"] + b["vol_ask"]
    return b


def one(b, name, lrule, d, P=None, delay=0, s=S):
    P = None if P is None else np.asarray(P, np.float64).reshape(1, G.NG)
    return G.run(b, name, lrule, np.array([d], np.int64), np.array([s], np.int64), P, delay)[0]


def pfull(v):
    return np.full(G.NG, v, np.float64)


def check(name, ok, **info):
    RESULTS.append({"test": name, "pass": bool(ok), **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}})


# ---------------------------------------------------------------- equivalence with research 17
def equivalence_tests():
    """Rules off: exec18 = exec17 on random bars (S0, S1, S2i, E*; L0 and L3; all nine engine17 columns); with p given but
    thresholds that never bind, R1 / R2 / R3 = R0; R2 skipping every E1 = S2i."""
    dirs = np.array([1, -1] * 10, np.int64)
    starts = np.arange(S, S + 20 * 2, 2, dtype=np.int64)
    rng = np.random.default_rng(7)
    ok17, okoff, oks2 = True, True, True
    for seed in range(12):
        b = rand_bars(seed)
        for lr in G.LRULES:
            for m17, m18 in (("S0", "S0"), ("S1", "S1"), ("S2i", "S2i"), ("Estar", "R0")):
                a = G17.run(b, m17, lr, dirs, starts)
                c = G.run(b, m18, lr, dirs, starts)
                ok17 &= np.array_equal(a, c[:, :9]) and not c[:, 9:11].any()
            P = rng.uniform(0.41, 0.59, (len(dirs), G.NG))                        # never <= 0.40 or >= 0.60
            r0 = G.run(b, "R0", lr, dirs, starts, P)
            for name in ("R1_0.60", "R2_0.40", "R3_0.60_0.40"):
                okoff &= np.array_equal(G.run(b, name, lr, dirs, starts, P), r0)
            s2 = G.run(b, "S2i", lr, dirs, starts)
            r2 = G.run(b, "R2_0.40", lr, dirs, starts, np.full((len(dirs), G.NG), 0.40))
            oks2 &= np.array_equal(r2[:, :9], s2[:, :9])
    check("rules off: exec18 = engine17 exec17 for S0, S1, S2i, E* (L0, L3, 12 random days x 20 parents)", ok17)
    check("p in (0.41, 0.59): R1_0.60, R2_0.40, R3_0.60_0.40 = R0 exactly", okoff)
    check("R2 with p = p_lo at every decision bar = S2i (E*'s limit rule at every E1 bar)", oks2)


# ---------------------------------------------------------------- features and y
def feature_tests():
    dirs, starts = np.array([1, -1], np.int64), np.array([S, S], np.int64)
    b = bar()
    b["bsz"][:], b["asz"][:] = 8000, 2000
    f = G.features(b, dirs, starts)
    check("imbalance (8000 - 2000) / 10000 = +0.6 for a buy, -0.6 for a sell", np.allclose(f["imb"][0], 0.6) and
          np.allclose(f["imb"][1], -0.6), buy=f["imb"][0, :3], sell=f["imb"][1, :3])
    check("no mid change in 60 bars -> y = -1 (left out)", (f["y"] == -1).all())
    b = shift(bar(), S + 10, N, 100)                                       # mid up 1 cent from s+10
    f = G.features(b, dirs, starts)
    check("mid up after the decision bar -> y = 1 (buy), 0 (sell) at t = 0", f["y"][0, 0] == 1 and f["y"][1, 0] == 0,
          y=f["y"][:, :3])
    b = shift(bar(), S + 10, N, -100)
    f = G.features(b, dirs, starts)
    check("mid down -> y = 0 (buy), 1 (sell)", f["y"][0, 0] == 0 and f["y"][1, 0] == 1)
    for lag, want in ((60, 1), (61, -1)):
        f = G.features(shift(bar(), S + lag, N, 100), dirs, starts)
        check(f"first change at u + {lag} -> y = {want}", f["y"][0, 0] == want, y=int(f["y"][0, 0]))
    b = shift(bar(), S + 20, N, 100)
    b["bid"][S + 5] = 0                                                    # invalid quote: skipped, not a change
    b["ask"][S + 7], b["bid"][S + 7] = BID - 100, BID                     # crossed (ask < bid): skipped
    f = G.features(b, dirs, starts)
    check("invalid bars (bid 0, ask < bid) are skipped in y", f["y"][0, 0] == 1 and f["y"][1, 0] == 0, y=f["y"][:, 0])
    b = bar()
    b["bid"][S + 5] = 0
    f = G.features(b, dirs, starts)
    m = {"vol_pre_fill": 1.0, "mean": [0, 0, 0, 0], "sd": [1, 1, 1, 1], "coef": [0.1, 0.1, 0.1, 0.1], "intercept": 0.0}
    P = G.predict(m, f)
    check("invalid decision quote -> not valid, p = nan (E* as is)", not f["valid"][0, 1] and np.isnan(P[0, 1]) and
          f["valid"][0, 0] and not np.isnan(P[0, 0]))
    b = bar(400)
    for k, v in ((S, 0), (S + 5, 1500), (S + 10, 100)):
        b["ask"][k] = BID + v
    f = G.features(b, dirs[:1], starts[:1])
    check("spread ticks 4 / locked 0 / 15c capped at 10 / 1", list(f["spr"][0, :3]) == [0.0, 10.0, 1.0] and f["spr"][0, 3] == 4.0,
          spr=f["spr"][0, :4])
    b = shift(bar(), S + 3, N, 100)                                       # mid +1c between s-5 and s+5
    b2 = shift(bar(), S + 3, N, 100, 0)                                  # bid only: +0.5c
    f, f2 = G.features(b, dirs, starts), G.features(b2, dirs, starts)
    check("5-second mid change: +1 tick buy / -1 sell, half tick 0.5; t = 0 uses s-5..s",
          f["d5"][0, 1] == 1.0 and f["d5"][1, 1] == -1.0 and f["d5"][0, 0] == 0.0 and f2["d5"][0, 1] == 0.5,
          d5=f["d5"][:, :3], half=f2["d5"][0, :3])


def future_tests():
    """Inputs and p at grid point k use bars <= u_k only: replacing every bar after u_k leaves points 0..k unchanged."""
    dirs = np.array([1, -1, 1, -1], np.int64)
    starts = np.array([700, 703, 707, 711], np.int64)                     # vol_pre defined (>= 10 one-minute returns)
    m = {"vol_pre_fill": 5.0, "mean": [0.1, 2, 5, 0], "sd": [0.5, 1.5, 2, 1], "coef": [0.3, -0.2, 0.1, 0.4], "intercept": 0.05}
    ok, okk = True, True
    for seed in range(6):
        b = rand_bars(seed)
        f = G.features(b, dirs, starts)
        P = G.predict(m, f)
        for k in (0, 7, 30, 59):
            c = rand_bars(100 + seed)
            for i in range(len(dirs)):
                u = starts[i] + 5 * k
                bb = {x: b[x].copy() for x in FIELDS}
                for x in FIELDS:
                    bb[x][u + 1:] = c[x][u + 1:]
                g = G.features(bb, dirs, starts)
                Q = G.predict(m, g)
                for x in ("imb", "spr", "d5"):
                    ok &= np.array_equal(g[x][i, :k + 1], f[x][i, :k + 1])
                ok &= np.array_equal(g["vp"][i], f["vp"][i]) and not np.isnan(f["vp"][i])
                ok &= np.array_equal(Q[i, :k + 1], P[i, :k + 1], equal_nan=True)
                ok &= np.array_equal(g["valid"][i, :k + 1], f["valid"][i, :k + 1])
        b = rand_bars(seed, spreads=(4, 5, 6))                            # R1 fires at k = 6 at the latest
        Pk = np.full((1, G.NG), 0.5)
        Pk[0, 6] = 0.7
        o = G.run(b, "R1_0.70", 3, dirs[:1], starts[:1], Pk)[0]
        u = int(starts[0] + o[5])
        c = rand_bars(200 + seed)
        bb = {x: b[x].copy() for x in FIELDS}
        for x in FIELDS:
            bb[x][u + 1:] = c[x][u + 1:]
        okk &= np.array_equal(G.run(bb, "R1_0.70", 3, dirs[:1], starts[:1], Pk)[0], o) and o[5] <= 30
    check("future bars changed after u_k: imbalance, spread, 5 s change, vol_pre, valid and p at points 0..k unchanged", ok)
    check("an R1 order filled by bar u: replacing bars after u changes nothing", okk)


# ---------------------------------------------------------------- rules
def threshold_tests():
    for lr in G.LRULES:
        for d, side in ((1, "buy"), (-1, "sell")):
            px = ASK if d > 0 else BID
            for h in G.P_HI:
                o = one(bar(), f"R1_{h:.2f}", lr, d, pfull(h))
                o2 = one(bar(), f"R1_{h:.2f}", lr, d, pfull(h - 1e-9))
                check(f"R1 L{lr} {side} p = p_hi {h:.2f} fires at arrival; p_hi - 1e-9 does not",
                      o[9] == G.QTY and o[2] == G.QTY and o[3] == G.QTY * px and o[5] == 0 and o2[9] == 0 and o2[6] == G.QTY,
                      fire=o, nofire=o2)
            Pk = np.full(G.NG, np.nan)
            Pk[3] = 0.65
            o = one(bar(), "R3_0.65_0.30", lr, d, Pk)
            check(f"R1 L{lr} {side} fires at the first grid point with p >= p_hi (t = 15 s); nan = no p", o[9] == G.QTY and o[5] == 15,
                  out=o)
            for lo in G.P_LO:
                o = one(bar(200), f"R2_{lo:.2f}", lr, d, pfull(lo))
                o2 = one(bar(200), f"R2_{lo:.2f}", lr, d, pfull(lo + 1e-9))
                check(f"R2 L{lr} {side} spread 2c, p = p_lo {lo:.2f} skips E1 at all 60 grid bars (deadline); p_lo + 1e-9 -> E1 at arrival",
                      o[10] == G.NG and o[6] == G.QTY and o[5] == G.HOR and o2[10] == 0 and o2[2] == G.QTY and o2[5] == 0 and o2[6] == 0,
                      skip=o, e1=o2)
        o = one(bar(400), "R2_0.40", lr, 1, pfull(0.1))
        check(f"R2 L{lr}: no E1 condition (spread 4c) -> nothing to skip, R2 = R0", np.array_equal(o, one(bar(400), "R0", lr, 1)))
        Pd = pfull(0.9)
        o = one(bar(200), "R1_0.60", lr, 1, Pd)
        check(f"R1 L{lr}: E1 comes before R1 (spread 2c and p 0.9 -> E1 market, R1 0)", o[9] == 0 and o[2] == G.QTY and o[5] == 0, out=o)


def d1_test():
    """R1 market above the displayed size: D1 adds 1 cent per share above it (L3D1), D0 does not."""
    for d, side in ((1, "buy"), (-1, "sell")):
        b = bar()
        b["asz"][S + 10], b["bsz"][S + 10] = 300, 300
        Pk = np.full(G.NG, 0.5)
        Pk[2] = 0.75
        mid2 = np.array([BID + ASK])
        for lr, dep in ((3, 1), (3, 0), (0, 1)):
            o = one(b, "R1_0.75", lr, d, Pk)
            is_ = float(G.costs(o[None, :], np.array([d]), mid2, dep)[0])
            want = 2.0 + dep * 700 / G.QTY + G.FEE_MKT                     # half spread 2c + 0.7 x dep + fee 0.3
            check(f"D1 R1 market L{lr}D{dep} {side}: at bar s+10 end quote, over 700 shares, IS {want:.2f}c",
                  o[9] == G.QTY and o[4] == 700 and o[5] == 10 and abs(is_ - want) < 1e-12, is_c=is_, want_c=want, out=o)


def r2_keeps_limit():
    for d, side in ((1, "buy"), (-1, "sell")):
        b = bar(200)
        px = BID if d > 0 else BID + 200
        if d > 0:
            b["ask_min"][S + 40] = BID                                    # the ask touches my bid at s+40
        else:
            b["bid_max"][S + 40] = BID + 200
        for lr in G.LRULES:
            o = one(b, "R2_0.30", lr, d, pfull(0.2))
            check(f"R2 L{lr} {side}: the limit at the touch stays posted after the skips and fills at {px} on s+40",
                  o[0] == G.QTY and o[1] == G.QTY * px and o[5] == 40 and o[10] == 8 and o[2] == 0, out=o)
    b = bar(200)                                                          # L0 queue: 5,000 ahead, 700 traded per bar
    b["vol_bid"][S + 1:] = 700
    o = one(b, "R2_0.30", 0, 1, pfull(0.2))
    check("R2 L0 buy: queue position kept across the skip at s+5 (5,000 ahead, 700 a bar -> filled at s+8..s+9; a re-post would wait)",
          o[0] == G.QTY and o[5] == 9 and o[10] == 2, out=o)


def delay_tests():
    for lr in G.LRULES:
        b = bar(200)
        b["ask"][S + 1] += 300
        b["asz"][S + 1] = 400
        o = one(b, "R0", lr, 1, delay=1)
        check(f"delay L{lr}: E1 decided at s -> market at bar s+1 ask (+3c), size of s+1", o[2] == G.QTY and o[3] == G.QTY * (ASK - 200 + 300)
              and o[4] == 600 and o[5] == 1, out=o)
        b = bar()
        b["ask"][S + G.HOR + 1] += 500
        o = one(b, "R0", lr, 1, delay=1)
        check(f"delay L{lr}: deadline decided at D -> market at D + 1", o[6] == G.QTY and o[5] == G.HOR + 1 and o[3] == G.QTY * (ASK + 500),
              out=o)
        b = bar()
        b["bid"][S + 1] += 100                                            # improve decided at s, priced at s+1: bid + 1c + 1c
        b["ask_min"][S + 2] = BID + 200
        o = one(b, "R0", lr, 1, delay=1)
        o0 = one(b, "R0", lr, 1)
        check(f"delay L{lr}: the limit decided at s is priced on bar s+1 and fills from s+2", o[0] == G.QTY and o[1] == G.QTY * (BID + 200)
              and o[5] == 2 and o0[0] == 0, delay=o, nodelay=o0)
        b = bar()
        b["ask"][S + 11] += 100
        Pk = np.full(G.NG, 0.5)
        Pk[2] = 0.6
        o = one(b, "R1_0.60", lr, 1, Pk, delay=1)
        check(f"delay L{lr}: R1 decided at s+10 -> market at s+11 ask", o[9] == G.QTY and o[5] == 11 and o[3] == G.QTY * (ASK + 100), out=o)
    b = rand_bars(3)
    dirs, starts = np.array([1, -1] * 5, np.int64), np.arange(S, S + 20, 2, dtype=np.int64)
    a = G.run(b, "R0", 3, dirs, starts, None, 1)
    check("delay on random bars: every parent ends with 1,000 shares, at most at D + 1", bool(np.all(a[:, 0] + a[:, 2] == G.QTY))
          and bool(np.all(a[:, 5] <= G.HOR + 1)))


# ---------------------------------------------------------------- statistics
def stats_tests():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    x = np.round(rng.normal(0, 3, 400), 1)
    codes = rng.integers(0, 15, 400)
    w = rng.integers(0, 4, 15).astype(np.float64)
    xs, cs = G.sorted_by(x, codes)
    ok = True
    for q in (95.0, 50.0, 0.0, 100.0, 97.3):
        ok &= abs(G.wpct(xs, cs, w, q) - np.percentile(np.repeat(x, w[codes].astype(int)), q)) < 1e-12
    check("weighted percentile = np.percentile (linear) on the repeated days", ok)
    a, b_ = rng.normal(0, 1, 400), rng.normal(0.2, 1, 400)
    idx = G.boot_index(15, B=50)
    W = G.day_counts(idx, 15)
    est, dist = G.p95_diff_boot(a, b_, codes, W)
    want = [np.percentile(np.repeat(a, W[k][codes].astype(int)), 95) - np.percentile(np.repeat(b_, W[k][codes].astype(int)), 95)
            for k in range(50)]
    check("bootstrap p95 difference = np.percentile on each resample; day counts = bincount of the block index",
          np.allclose(dist, want, atol=1e-12) and abs(est - (np.percentile(a, 95) - np.percentile(b_, 95))) < 1e-15
          and np.array_equal(W[3], np.bincount(idx[3], minlength=15)))
    p = np.round(rng.random(400), 2)                                     # ties
    y = (rng.random(400) < p).astype(np.int8)
    e, dist = G.auc_boot(p, y, codes, W)
    want = [roc_auc_score(y, p, sample_weight=W[k][codes]) for k in range(50)]
    check("AUC exact (ties 1/2) = sklearn roc_auc_score; bootstrap AUC = weighted roc_auc_score",
          abs(e - roc_auc_score(y, p)) < 1e-12 and np.allclose(dist, want, atol=1e-12), auc=e)
    h = G.holm({"H1": 0.01, "H2": 0.02, "H3": 0.04})
    h2 = G.holm({"H1": 0.001, "H2": 0.06, "H3": 0.01})
    h3 = G.holm({"H1": 0.03, "H2": 0.035, "H3": 0.001})               # H1 fails 0.025 -> H2 not tested
    check("Holm with three: 0.0167, 0.025, 0.05; stops at the first failure",
          [round(h[k]["level"], 4) for k in ("H1", "H2", "H3")] == [0.0167, 0.025, 0.05] and all(h[k]["rejected"] for k in h)
          and h2["H1"]["rejected"] and h2["H3"]["rejected"] and not h2["H2"]["rejected"]
          and h3["H3"]["rejected"] and not h3["H1"]["rejected"] and not h3["H2"]["rejected"], cases=[h, h2, h3])
    rows = {c: {"mean": 3.0, "p95": 18.0, "fired": 0.5} for c in G.CANDS}
    rows["R1_0.60"] = {"mean": 2.900, "p95": 18.5, "fired": 0.2}
    rows["R3_0.70_0.35"] = {"mean": 2.915, "p95": 18.2, "fired": 0.3}
    rows["R2_0.30"] = {"mean": 2.925, "p95": 17.0, "fired": 0.1}
    a1, rec = G.select_astar(rows)
    rows["R3_0.70_0.35"]["p95"] = 18.5
    a2, _ = G.select_astar(rows)
    rows["R1_0.60"]["fired"] = 0.4
    a3, _ = G.select_astar(rows)
    check("A*: tie within 0.02c -> lower p95, then the lower firing share; 0.025c away is not a tie",
          a1 == "R3_0.70_0.35" and a2 == "R1_0.60" and a3 == "R3_0.70_0.35" and "R2_0.30" not in rec["tied_within_0.02"], first=rec)
    check("candidates: R0 + R1 x 4 + R2 x 3 + R3 x 12 = 20; R3_0.60_0.40 = (p_hi 0.60, p_lo 0.40)",
          len(G.CANDS) == 20 and G.params("R3_0.60_0.40")[4:] == (0.6, 0.4) and G.params("R1_0.75")[4:] == (0.75, -1.0)
          and G.params("R0") == (2, 1, 2, 0, -1.0, -1.0))
    b3 = G.branch_c3(np.array([400, 500, 300, 401]))
    check("C3 spread 4c -> S2i, 5c -> E*", list(b3) == [True, False, True, False], flags=b3)


def main():
    equivalence_tests()
    feature_tests()
    future_tests()
    threshold_tests()
    d1_test()
    r2_keeps_limit()
    delay_tests()
    stats_tests()
    ok = all(r["pass"] for r in RESULTS)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "engine18_tests.json"), "w") as fh:
        json.dump({"pass": ok, "n": len(RESULTS), "failed": [r["test"] for r in RESULTS if not r["pass"]], "tests": RESULTS},
                  fh, indent=1, default=str)
    for r in RESULTS:
        print(("PASS " if r["pass"] else "FAIL ") + r["test"])
    print(f"{sum(r['pass'] for r in RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
