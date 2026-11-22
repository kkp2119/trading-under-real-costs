#!/usr/bin/env python3
"""Research 17 synthetic tests of engine17 (step 2; must pass before any development run). Synthetic bars only: a flat
$100.00 / $100.04 quote with no trades (so a posted order never fills unless the quote crosses it), shifted from a chosen
bar on. Writes results/engine17_tests.json; exits 1 when a test fails.
  python test_engine17.py"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine17 as G  # noqa: E402

BID, ASK, SZ, N, S = 1_000_000, 1_000_400, 5000, 400, 50
FIELDS = ("bid", "ask", "bsz", "asz", "bid_max", "bid_min", "ask_max", "ask_min", "vol_bid", "vol_ask", "vol_bid_odd",
          "vol_ask_odd", "px_min", "px_max")
RESULTS = []


def bar():
    b = {k: np.zeros(N, np.int64) for k in FIELDS}
    for k in ("bid", "bid_max", "bid_min"):
        b[k][:] = BID
    for k in ("ask", "ask_max", "ask_min"):
        b[k][:] = ASK
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


def one(b, name, lrule, d):
    return G.run(b, name, lrule, np.array([d], np.int64), np.array([S], np.int64))[0]


def check(name, ok, **info):
    RESULTS.append({"test": name, "pass": bool(ok), **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}})


def stop_tests():
    for run in ("Estar_stop", "S2i_stop"):
        for lr in G.LRULES:
            for d, side in ((1, "buy"), (-1, "sell")):
                for cents, fire in ((2.99, False), (3.00, True)):
                    move = int(round(cents * G.TICK)) * d                     # against the order: buy up, sell down
                    o = one(shift(bar(), S + 10, N, move), run, lr, d)
                    px = (ASK if d > 0 else BID) + move
                    if fire:
                        ok = o[7] == G.QTY and o[2] == G.QTY and o[5] == 10 and o[3] == G.QTY * px and o[6] == 0
                    else:
                        ok = o[7] == 0 and o[6] == G.QTY and o[5] == G.HOR
                    check(f"stop {run} L{lr} {side} {cents:.2f}c -> {'fires' if fire else 'no stop'}", ok, out=o[:9])
    for lr in G.LRULES:                                                # favourable 3c move: never a stop
        for d, side in ((1, "buy"), (-1, "sell")):
            o = one(shift(bar(), S + 10, N, -300 * d), "Estar_stop", lr, d)
            check(f"stop Estar_stop L{lr} {side} favourable 3c -> no stop", o[7] == 0, out=o)
        o = one(shift(bar(), S + 7, S + 10, 300), "Estar_stop", lr, 1)  # adverse only between decision bars
        check(f"stop L{lr} buy 3c on bars s+7..s+9 only (not a decision bar) -> no stop", o[7] == 0 and o[6] == G.QTY, out=o)
        o = one(shift(bar(), S + 10, N, 300), "Estar", lr, 1)          # stop 0 = E* as before
        check(f"stop off (Estar) L{lr} buy 3c -> deadline as E*", o[7] == 0 and o[6] == G.QTY, out=o)
        b = shift(bar(), S + 10, N, 500, 300)                         # spread 2c and +4c mid at s+10: E1 and stop both
        oe, os_ = one(b, "Estar_stop", lr, 1), one(b, "S2i_stop", lr, 1)
        check(f"stop L{lr} E1 and stop on the same bar -> E1 label for E*, stop label for S2i (same price)",
              oe[7] == 0 and oe[2] == G.QTY and oe[5] == 10 and os_[7] == G.QTY and os_[5] == 10 and oe[3] == os_[3],
              estar=oe, s2i=os_)
        b = bar()
        b["bid"][S + 10] = 0
        o = one(b, "Estar_stop", lr, 1)
        check(f"stop L{lr} bid 0 on a decision bar -> not tested, counted", o[8] == 1 and o[7] == 0, out=o)


def d1_test():
    """Stop market above the displayed size: D1 adds 1 cent per share above it (L3D1), D0 does not."""
    b = shift(bar(), S + 10, N, 300)
    b["asz"][S + 10] = 300
    dirs, mid2 = np.array([1]), np.array([BID + ASK])
    for lr, dep in ((3, 1), (3, 0), (0, 1)):
        o = one(b, "Estar_stop", lr, 1)
        is_ = float(G.costs(o[None, :], dirs, mid2, dep)[0])
        want = (ASK + 300 - (BID + ASK) / 2) / 100 + dep * 700 * 1 / G.QTY + G.FEE_MKT    # 5.00 + 0.70 x dep + 0.30
        check(f"D1 stop market L{lr}D{dep}: over 700 shares, IS {want:.2f}c", o[7] == G.QTY and o[4] == 700 and abs(is_ - want) < 1e-12,
              is_c=is_, want_c=want, out=o)


def branch_tests():
    b2 = G.branch_c2([4.68, 4.6799, np.nan, 4.67696, 4.6800001])
    check("C2 vol_pre 4.68 -> E*, 4.6799 -> S2i, undefined -> E*", list(b2) == [False, True, False, True, False], flags=b2)
    b3 = G.branch_c3(np.array([400, 500, 300, 401]))
    check("C3 spread 4c -> S2i, 5c -> E*", list(b3) == [True, False, True, False], flags=b3)
    o = np.zeros((len(G.RUN_NAMES), 4, G.NCOL), np.int64)
    for i in range(len(G.RUN_NAMES)):
        o[i, :, 0] = i + 1                                             # marker: run number
    vp, spr = np.array([4.0, 5.0, np.nan, 4.0]), np.array([400, 400, 500, 500])
    c, f = G.compose(o, vp, spr)
    r = {k: i + 1 for i, k in enumerate(G.RUN_NAMES)}
    want = {"C0": [r["Estar"]] * 4, "C1": [r["S2i"]] * 4, "C2": [r["S2i"], r["Estar"], r["Estar"], r["S2i"]],
            "C3": [r["S2i"], r["S2i"], r["Estar"], r["Estar"]], "C4": [r["Estar_stop"]] * 4,
            "C5": [r["S2i_stop"], r["Estar_stop"], r["Estar_stop"], r["S2i_stop"]]}
    check("routing C0-C5 picks the right run per parent", all(list(c[k][:, 0]) == want[k] for k in G.CANDS),
          got={k: c[k][:, 0].tolist() for k in G.CANDS})
    check("branch flags C2/C5 = vol rule, C3 = spread rule", list(f["C2"]) == [True, False, False, True] and
          list(f["C5"]) == list(f["C2"]) and list(f["C3"]) == [True, True, False, False])


def vol_pre_tests():
    rng = np.random.default_rng(0)
    m = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 2000)))
    v = np.ones(2000, bool)
    a, _ = G.vol_pre(m, v, 600)                                       # points 599, 539, ..., 59: 10 points, 9 returns
    pts = np.array([659 - 60 * k for k in range(11)])[::-1]
    want = float(np.std(np.diff(np.log(m[pts])), ddof=1) * 1e4)
    b_, _ = G.vol_pre(m, v, 660)                                      # 11 points, 10 returns
    v2 = v.copy()
    v2[359] = False
    c, _ = G.vol_pre(m, v2, 660)                                      # one invalid point -> 8 returns
    check("vol_pre 9 returns -> undefined, 10 -> defined (= manual), an invalid point drops returns",
          np.isnan(a) and abs(b_ - want) < 1e-12 and np.isnan(c), nine=a, ten=b_, manual=want, invalid=c)
    check("vol_pre undefined routes to E* in C2", not G.branch_c2([a])[0])


def stats_tests():
    h = G.holm({"H1": 0.01, "H2": 0.04})
    h2 = G.holm({"H1": 0.03, "H2": 0.04})
    h3 = G.holm({"H1": 0.001, "H2": 0.06})
    h4 = G.holm({"H1": 0.04})
    check("Holm levels 0.025 then 0.05 (and 0.05 alone)",
          h["H1"]["rejected"] and h["H2"]["rejected"] and h["H1"]["level"] == 0.025 and not h2["H1"]["rejected"]
          and not h2["H2"]["rejected"] and h3["H1"]["rejected"] and not h3["H2"]["rejected"] and h4["H1"]["rejected"],
          cases=[h, h2, h3, h4])
    rows = {c: {"mean": 3.0, "p95": 18.0} for c in G.CANDS}
    rows["C2"] = {"mean": 2.900, "p95": 18.5}
    rows["C5"] = {"mean": 2.915, "p95": 18.2}
    rows["C3"] = {"mean": 2.925, "p95": 17.0}
    a, rec = G.select_astar(rows)
    rows["C5"]["p95"] = 18.5
    b_, _ = G.select_astar(rows)
    check("A*: tie within 0.02c -> lower p95 (C5), then fewer rules (C2); 0.025c away is not a tie",
          a == "C5" and b_ == "C2" and "C3" not in rec["tied_within_0.02"], first=rec)
    d = np.array([-1.0, -2.0, -3.0, -1.0])
    codes = np.array([0, 0, 1, 1])
    est, p, ub = G.one_sided(d, codes, 2, G.boot_index(2, B=200))
    check("one-sided bootstrap: mean, p, upper bound", abs(est + 1.75) < 1e-12 and p == 0.0 and ub < 0, est=est, p=p, upper=ub)


def main():
    stop_tests()
    d1_test()
    branch_tests()
    vol_pre_tests()
    stats_tests()
    ok = all(r["pass"] for r in RESULTS)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "engine17_tests.json"), "w") as fh:
        json.dump({"pass": ok, "n": len(RESULTS), "failed": [r["test"] for r in RESULTS if not r["pass"]], "tests": RESULTS},
                  fh, indent=1, default=str)
    for r in RESULTS:
        print(("PASS " if r["pass"] else "FAIL ") + r["test"])
    print(f"{sum(r['pass'] for r in RESULTS)}/{len(RESULTS)} passed")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
