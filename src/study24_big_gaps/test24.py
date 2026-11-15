#!/usr/bin/env python3
"""Research 24 synthetic tests (step-2 prompt 1; they must pass before the development run). Synthetic bars only; no bar
of research 24 is read and the sealed folder is not named outside temporary stand-ins.
  earlier   research 20's 117 tests and research 21's synthetic tests (all but its test of research 21's own live folder),
            rerun here through their test functions
  frozen    the files of research 17, 20 and 21 used here equal the files of their tags
  amount    the share rule at 10,000 dollars (lots of 10, half up, at least 10); with 50,000 dollars the trade rows are
            research 20's; with 10,000 the P&L of a known trade
  delay     with every order 1 or 5 seconds late the decisions stay where they are and the four orders fill at the end
            quotes of the later bars (entry and exit, the forced exit included)
  bounds    kappa 2 and 3 and z_in 3: equal to the threshold enters, just below does not
  over      trades with an order above the displayed size and the shares above it
  top 5     the mean without the 5 largest days, without the 5 largest pairs and without both
  rules     A*: the 5% tie and "no profit in development"; pair types; the loader (sizes / 100 from 2025-11-03, a value
            that is not a multiple of 100 stops the load, development-only guard)
  world     four synthetic pairs through the development run, the rehearsal, the guard and the sealed run with
            hand-computed P&L per delay; a changed development summary stops the guard; a second world without profit
            at a 1-second delay: the sealed run is refused
Writes results/engine24_tests.json (an earlier file is moved to results/superseded/); exits 1 when a test fails.
  python test24.py"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from decimal import Decimal, ROUND_FLOOR

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core24 as C24  # noqa: E402
import dev24 as DV  # noqa: E402
import loader24 as L24  # noqa: E402
import pair24 as P24  # noqa: E402
import run_test24 as RT  # noqa: E402
import stats24 as ST24  # noqa: E402
import test21 as T21  # noqa: E402  (research 21's test functions; its main is never called)

T20, C21, L21 = T21.T20, C24.C21, L24.L21
C, PE, S, ST20 = C24.C, C24.PE, C24.S, C24.ST20
N, BID, SPR, GE = T20.N, T20.BID, T20.SPR, T20.GE
COL = C24.COL
RESULTS = []
EXTRA_FROZEN = {"study21_more_pairs/test21.py": ("test-run-21", "ba1969b24cabe316b3b7b4b5fdd6d4244393bd0000be782f9671022d045241bc"),
                "study21_more_pairs/dev21.py": ("test-run-21", "b1545333962b9819f7186a6444b4ae267d6bd5d9342c4c6377e3755afde7b115"),
                "study21_more_pairs/run_test21.py": ("test-run-21", "574f063a893d616d662d577f57cd7389f93d9cce04821062876ae967c2ab12a5"),
                "study20_pairs/test20.py": ("code-freeze-20", "388cf5bad6dbedb812c0323bfe0ccee06121262b2436141ff127bc5a9912f1fa")}
OWN_FOLDER = "the real sealed run is refused now"            # research 21's test of its own live folder: not rerun here


def check(name, ok, **info):
    RESULTS.append({"test": name, "pass": bool(ok), **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}})


def close(a, b, tol=1e-9):
    return bool(np.allclose(a, b, rtol=0, atol=tol))


def quiet(fn, *a, **k):
    """(result, error text or None) of a call whose printed output is dropped."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            return fn(*a, **k), None
        except SystemExit as e:
            return None, str(e)
        except Exception as e:                                # a crash is a failed test, not a crashed test run
            return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- earlier studies
def earlier_tests(tmp):
    T21.RESULTS.clear()
    r20 = T21.research20_tests()
    T21.frozen_tests()
    sub = os.path.join(tmp, "r21")
    os.makedirs(sub)
    T21.size_tests(sub)
    T21.kappa_tests()
    T21.control_tests()
    T21.stats_tests()
    T21.core_tests()
    T21.world_tests(sub)
    got = list(T21.RESULTS)
    stored = [r["test"] for r in json.load(open(os.path.join(C24.PROJ, "study21_more_pairs", "results", "engine21_tests.json")))["tests"]]
    want = [t for t in stored if not t.startswith(OWN_FOLDER)]
    n_ok = sum(r["pass"] for r in got)
    check(got[0]["test"], got[0]["pass"], passed=got[0].get("passed"), total=got[0].get("total"))
    check(f"research 21's synthetic tests pass here ({len(want)}: the names of its stored result without the test of its own live folder)",
          len(got) == len(want) == n_ok and [r["test"] for r in got] == want and len(stored) - len(want) == 1, passed=n_ok, total=len(got),
          failed=[r["test"] for r in got if not r["pass"]])
    return r20, f"{n_ok}/{len(got)}"


def frozen_tests():
    tags = {**{p: ("code-freeze-20", h) for p, h in C21.FROZEN20.items()}, **{p: ("test-run-21", h) for p, h in C24.FROZEN21.items()}, **EXTRA_FROZEN}
    got = {}
    for p, (tag, h) in tags.items():
        blob = subprocess.run(["git", "-C", C24.PROJ, "show", f"{tag}:{p}"], capture_output=True).stdout
        got[p] = (hashlib.sha256(blob).hexdigest() == h, hashlib.sha256(open(os.path.join(C24.PROJ, p), "rb").read()).hexdigest() == h)
    check("the engine files of research 17, 20 and 21 and their test files equal the files of their tags (git) and the working files",
          all(a and b for a, b in got.values()) and C24.frozen_differences() == [], files={p: list(v) for p, v in got.items()})


# ---------------------------------------------------------------- amount
def amount_tests():
    cases = {2_000_200: 100, 20_000_000: 10, 40_000_000: 10, 40_000_400: 10, 8_000_000: 30, 7_999_800: 30, 8_000_200: 20, 1_333_400: 150,
             246_900: 810, 10_688_500: 20, 100_000: 2000, 39_999_800: 10}
    ref = {m: max(int((Decimal(10_000) / (Decimal(m) / 20_000) / 10 + Decimal("0.5")).to_integral_value(ROUND_FLOOR)), 1) * 10 for m in cases}
    check("shares at 10,000 dollars: amount / mid in lots of 10, half up (400.00 dollars -> 2.5 lots -> 30 shares, one tick above -> 20), at "
          "least 10 shares (1,000 dollars -> 10; 2,000 dollars -> 10, more than the amount)",
          all(int(P24.shares(m, 10_000)) == q == ref[m] for m, q in cases.items()), got={str(m): int(P24.shares(m, 10_000)) for m in cases})
    m = np.array(list(cases))
    try:
        P24.amount_n(10_000.5)
        bad = False
    except ValueError:
        bad = True
    check("shares at 50,000 dollars equal research 20's rule (signal20.shares); the amount must be a positive whole number of dollars",
          np.array_equal(P24.shares(m, 50_000), S.shares(m)) and P24.amount_n(50_000) == 100_000_000 and P24.amount_n(10_000) == 20_000_000 and bad)
    same, n_tr = True, 0
    for seed in (31, 32, 33):
        a, b = T21.random_day(seed)
        B = S.base(a, b, N)
        XA, XB = PE.stack(a), PE.stack(b)
        for cfg in ((10, 2.0, 0.0), (30, 3.0, 1.0)):
            side, exz = S.flags(B, *cfg, N)
            for method in (PE.M, PE.P):
                for lr in (3, 0):
                    for d in (0, 1, 5):
                        t20, t24 = PE.trades(method, lr, d, B, side, exz, XA, XB, N), P24.trades(50_000, method, lr, d, B, side, exz, XA, XB, N)
                        same &= bool(np.array_equal(t20, t24))
                        n_tr += len(t24)
    check("with 50,000 dollars pair24's trade rows are pair20's exactly: three random days, two settings, market and passive orders, both "
          "fill rules, delays 0, 1 and 5", same and n_tr > 500, trades_compared=n_tr)
    a, b = T20.spike_leg(), T20.leg()
    B = S.base(a, b, N)
    side, exz = S.flags(B, 10, 3.0, 2.0, N)
    t5 = P24.trades(50_000, PE.M, 3, 0, B, side, exz, PE.stack(a), PE.stack(b), N)
    t1 = P24.trades(10_000, PE.M, 3, 0, B, side, exz, PE.stack(a), PE.stack(b), N)
    m5, m1 = PE.money(t5, 0), PE.money(t1, 0)
    check("a 10-tick spike of 30 s at 100 dollars with 10,000 dollars per leg: one trade of 100 shares a leg (500 with 50,000), P&L 4.8 = "
          "mid 10 - cost 4 - fee 1.2 dollars, a fifth of the 50,000-dollar trade; the decisions are the same",
          len(t1) == 1 == len(t5) and t1[0, 3:5].tolist() == [100, 100] and t5[0, 3:5].tolist() == [500, 500]
          and close([m1["pnl"][0], m1["mid"][0], m1["cost"][0], m1["fee"][0]], [4.8, 10.0, 4.0, 1.2]) and close(m5["pnl"][0], 24.0)
          and np.array_equal(t1[:, [0, 1, 2, 5]], t5[:, [0, 1, 2, 5]]) and np.array_equal(t1[0, 3:5], P24.shares(t1[0, PE.MID0:PE.MID0 + 2], 10_000)),
          pnl=[float(m1["pnl"][0]), float(m5["pnl"][0])])


# ---------------------------------------------------------------- delay
def delay_tests():
    ok_time, ok_fill, n_tr, reasons = True, True, 0, np.zeros(5, int)
    ramps = {"time": T20.ramp(T20.leg(), 5399, 10, 5399 + 2700),            # a gap that keeps widening from 11:00:00 for 45 minutes: exits by time
             "close": T20.ramp(T20.leg(), 21_479, 10, N)}                    # a gap that opens at 15:28:00 and widens to the close: forced exit at 15:55
    for seed in (41, 42, 43, 44, "time", "close"):
        a, b = T21.random_day(seed) if isinstance(seed, int) else (ramps[seed], T20.leg())
        b["bsz"][:], b["asz"][:] = 40, 40                                    # leg B shows less than the order: every B order is above the size
        B = S.base(a, b, N)
        XA, XB = PE.stack(a), PE.stack(b)
        side, exz = S.flags(B, 10, 2.0, 0.0, N)
        base = P24.trades(10_000, PE.M, 3, 0, B, side, exz, XA, XB, N)
        for d in (0, 1, 5):
            tr = P24.trades(10_000, PE.M, 3, d, B, side, exz, XA, XB, N)
            ok_time &= bool(np.array_equal(tr[:, :6], base[:, :6]) and np.array_equal(tr[:, PE.MID0:PE.MID0 + 4], base[:, PE.MID0:PE.MID0 + 4]))
            for i in range(len(tr)):
                if tr[i, 5] == S.R_LAST:
                    continue
                for j, (leg, bar, sgn) in enumerate(((a, tr[i, 0] + d, -tr[i, 2]), (b, tr[i, 0] + d, tr[i, 2]),
                                                     (a, min(tr[i, 1] + d, N - 1), tr[i, 2]), (b, min(tr[i, 1] + d, N - 1), -tr[i, 2]))):
                    q = tr[i, 3 + j % 2]
                    px, shown = (leg["ask"][bar], leg["asz"][bar]) if sgn > 0 else (leg["bid"][bar], leg["bsz"][bar])
                    o = T20.od(tr, i, j)
                    ok_fill &= bool(o[2] == q and o[3] == q * px and o[0] == 0 and o[4] == max(0, q - shown) and o[5] == 0)
            if d == 1:
                n_tr += len(tr)
                reasons += np.bincount(tr[:, 5], minlength=5)
    check("delay 1 and 5 on four random days and two days with a widening gap: the decision bars, sides, shares, exit reasons and decision mids are those of delay 0; every "
          "entry order and every exit order (by z, by time and the forced exit at 15:55) fills whole at the end quote of the bar d seconds "
          "after its decision (buy at the ask, sell at the bid), and the shares above the displayed size are counted at that bar",
          ok_time and ok_fill and n_tr > 100 and reasons[S.R_Z] > 0 and reasons[S.R_TIME] > 0 and reasons[S.R_CLOSE] > 0, trades=n_tr,
          exit_reasons=reasons[1:].tolist(), same_decisions=ok_time, fills_at_the_later_bar=ok_fill)
    # a spike that lasts one bar: the profit is there at delay 0 only; one that lasts three bars: at delays 0 and 1
    b = T20.leg()
    got = {}
    for name, bars in (("one bar", 1), ("three bars", 3), ("thirty bars", 30)):
        a = T20.setq(T20.leg(), GE, GE + bars, BID + 1400)
        B = S.base(a, b, N)
        side, exz = S.flags(B, 10, 3.0, 3.0, N)
        got[name] = [round(float(PE.money(P24.trades(10_000, PE.M, 3, d, B, side, exz, PE.stack(a), PE.stack(b), N), 0)["pnl"].sum()), 6) for d in (0, 1, 5)]
    check("a 14-tick gap at 100 dollars, 10,000 dollars a leg: lasting 30 s it pays 8.8 dollars at delays 0, 1 and 5; lasting 3 s it pays 8.8 at "
          "delays 0 and 1 and loses 5.2 at delay 5; lasting 1 s it pays at delay 0 only (the order arrives after the gap has closed)",
          got == {"one bar": [8.8, -5.2, -5.2], "three bars": [8.8, 8.8, -5.2], "thirty bars": [8.8, 8.8, 8.8]}, pnl_by_delay=got)


# ---------------------------------------------------------------- thresholds
def bound_tests():
    r0 = 2.0 ** -12                                                          # about 2.4 bp, a power of two: kappa x r0 has no rounding
    for ka in ST24.KAPPAS:
        w = ka * r0
        dev = np.array([w, w, np.nextafter(w, 0.0), -w, -w, -np.nextafter(w, 0.0), w])
        rs = np.array([r0, np.nextafter(r0, 1.0), r0, r0, np.nextafter(r0, 1.0), r0, r0 / 2])
        side, _ = S.decide(np.where(dev > 0, 5.0, -5.0), dev, rs, np.ones(len(dev), bool), ST24.Z_IN, ka)
        check(f"kappa {ka:g} boundary: |s - m| equal to kappa x (relative spreads) enters, just below does not (both signs)",
              side.tolist() == [1, 0, 0, -1, 0, 0, 1], side=side)
    z = np.array([3.0, np.nextafter(3.0, 0.0), -3.0, -np.nextafter(3.0, 0.0), 2.99, 3.01])
    side, _ = S.decide(z, np.sign(z) * 1.0, np.zeros(6), np.ones(6, bool), ST24.Z_IN, 2.0)
    check("z_in = 3 boundary: |z| equal to 3 enters, just below does not (both signs)", side.tolist() == [1, 0, -1, 0, 0, 1], side=side)
    b = T20.leg()
    n = {t: {ka: len(T20.run(T20.spike_leg(ticks=t), b, zi=3.0, ka=ka)[0]) for ka in (2.0, 3.0)} for t in (7, 10, 14)}
    check("through the pair engine (two legs with 2-tick spreads at 100 dollars, relative spreads 4 bp): a 7 bp gap trades with neither "
          "kappa, a 10 bp gap with kappa 2 only, a 14 bp gap with both; the grid is W {10, 30} x kappa {2, 3} at z_in 3 in that order",
          n == {7: {2.0: 0, 3.0: 0}, 10: {2.0: 1, 3.0: 0}, 14: {2.0: 1, 3.0: 1}} and ST24.candidates() == [(10, 3.0, 2.0), (10, 3.0, 3.0), (30, 3.0, 2.0), (30, 3.0, 3.0)]
          and [ST24.sig_name(c) for c in ST24.candidates()] == ["W10_z3_k2", "W10_z3_k3", "W30_z3_k2", "W30_z3_k3"], trades=n)


# ---------------------------------------------------------------- shares above the displayed size
def over_tests():
    cfgs = [(10, 3.0, 2.0)]
    res = {}
    for name, (sa, sb, sx) in {"enough": (200, 200, None), "short on all four": (60, 60, None), "short at one exit only": (200, 200, 30)}.items():
        a, b = T20.spike_leg(ticks=14), T20.leg()
        a["bsz"][:], a["asz"][:], b["bsz"][:], b["asz"][:] = sa, sa, sb, sb
        if sx is not None:
            b["bsz"][GE + 30:GE + 31] = sx                                   # leg B is sold back at bar GE + 30 (delay 0)
        out, X, mism, kept, _, _, left, _ = C24.pairday(a, b, N, cfgs, 10_000, (0, 1), keep=((0, 0),))
        res[name] = (int(out[0, 0, 1, COL["n"]]), int(X[0, 0]), int(out[0, 0, 1, COL["over_shares"]]), round(float(out[0, 0, 2, COL["pnl"]] - out[0, 0, 1, COL["pnl"]]), 6),
                     int(X[0, 1]), C24.over_trades(kept[(0, 0)]), mism, left)
    check("orders above the displayed size (100-share orders): none with 200 shown; with 60 shown all four orders are above it (one trade "
          "counted once, 160 shares, L3D1 1.6 dollars below L0D0); with 30 shown at one exit bar only the trade still counts once (70 "
          "shares, 0.7 dollars) at delay 0 and not at delay 1, when that bar is not used",
          res == {"enough": (1, 0, 0, 0.0, 0, 0, 0, 0), "short on all four": (1, 1, 160, 1.6, 1, 1, 0, 0), "short at one exit only": (1, 1, 70, 0.7, 0, 1, 0, 0)}, got=res)
    days = ["20200102", "20200103"]
    tasks = [{"date": d, "year": 2020, "name": "AA/BB"} for d in days]
    A = np.zeros((2, 1, 1, 3, C24.NAGG))
    A[:, 0, 0, 1, COL["n"]], A[:, 0, 0, 1, COL["pnl"]], A[:, 0, 0, 1, COL["mkt_shares"]], A[:, 0, 0, 1, COL["over_shares"]] = [3, 1], [30.0, -10.0], [1200, 400], [100, 60]
    r = C24.run_row(A, np.array([[[2.0]], [[1.0]]]), C.meta(tasks, days), 0, 0, "L3D1", 2, ST24.boot_index(2), "x")
    check("table row: trades with an order above the displayed size / trades, and shares above it / market shares",
          r["over_trades"] == 3 and close(r["over_trade_frac"], 0.75) and close(r["over_share_frac"], 0.1) and r["trades"] == 4 and close(r["mean_daily_usd"], 10.0))


# ---------------------------------------------------------------- the 5 largest days and pairs
def top_tests():
    days = [f"202001{d:02d}" for d in range(2, 14)]                           # 12 days
    pairs = [("AA", "AB"), ("BA", "BB"), ("CA", "CB"), ("DA", "DB"), ("EA", "EB"), ("FA", "FB"), ("GA", "GB")]
    rng = np.random.default_rng(24)
    tasks, vals = [], []
    for di, d in enumerate(days):
        for pi, (a, b) in enumerate(pairs):
            if (di + pi) % 5 == 0:
                continue                                                     # not every pair trades every day
            x, y = (a, b) if di < 6 else (b, a)                              # the same pair listed the other way round later
            tasks.append({"date": d, "year": 2020, "family": "E2", "type": "index_etf", "pair": f"P{pi}", "name": f"{x}/{y}", "key": ST24.pair_key(x, y)})
            vals.append(float(rng.integers(-40, 60)) + 25.0 * (pi == 2) + 60.0 * (di == 7))
    vals = np.array(vals)
    A = np.zeros((len(tasks), 1, 1, 3, C24.NAGG))
    A[:, 0, 0, 1, COL["pnl"]], A[:, 0, 0, 1, COL["n"]] = vals, 1
    idx = ST24.boot_index(len(days))
    w = C24.without_top(A, tasks, days, 0, 0, "L3D1", idx)
    pos = {d: i for i, d in enumerate(days)}
    d = np.zeros(len(days))
    tot = {}
    for t, v in zip(tasks, vals):
        d[pos[t["date"]]] += v
        tot[t["key"]] = tot.get(t["key"], 0.0) + v
    order_d = sorted(range(len(days)), key=lambda i: (-d[i], i))[:5]
    top_p = sorted(tot, key=lambda q: (-tot[q], q))[:5]
    d2 = np.zeros(len(days))
    for t, v in zip(tasks, vals):
        if t["key"] not in top_p:
            d2[pos[t["date"]]] += v
    order_d2 = sorted(range(len(days)), key=lambda i: (-d2[i], i))[:5]
    rest = lambda x, drop: float(np.delete(x, drop).mean())                  # noqa: E731
    check("without the 5 largest days: those days leave the sample and the mean is over the 7 days left; without the 5 largest pairs "
          "(a pair = its two symbols whatever the order they are listed in): every day stays; without both: the pairs first, then the 5 "
          "largest days of what is left. All equal a direct computation",
          close(w["all"]["mean_daily_usd"], d.mean()) and close(w["without_top_days"]["mean_daily_usd"], rest(d, order_d)) and w["without_top_days"]["days"] == 7
          and [x[0] for x in w["without_top_days"]["days_removed"]] == [days[i] for i in order_d] and close(w["without_top_pairs"]["mean_daily_usd"], d2.mean())
          and w["without_top_pairs"]["days"] == 12 and [x[0] for x in w["without_top_pairs"]["pairs_removed"]] == top_p and len(tot) == 7
          and close(w["without_both"]["mean_daily_usd"], rest(d2, order_d2)) and [x[0] for x in w["without_both"]["days_removed"]] == [days[i] for i in order_d2]
          and close(w["without_top_days"]["share_of_total"], d[order_d].sum() / d.sum()) and close(w["without_top_pairs"]["share_of_total"], sum(tot[q] for q in top_p) / d.sum())
          and "CA/CB" in top_p and days[7] in [x[0] for x in w["without_top_days"]["days_removed"]], removed_pairs=top_p, removed_days=[days[i] for i in order_d])
    bk = C24.by_key(A, tasks, 0, 0, "L3D1")
    check("per-pair totals use one name for the two symbols whatever the order (AA/AB and AB/AA are one pair)",
          set(bk) == set(tot) and all(close(bk[q]["total_usd"], tot[q]) for q in tot) and sum(v["pair_days"] for v in bk.values()) == len(tasks))
    idx8 = ST24.boot_index(250)
    check("bootstrap: 10,000 draws of circular 21-day blocks with seed 24 (differs from seeds 20 and 21, repeats itself)",
          idx8.shape == (10_000, 250) and not np.array_equal(idx8, ST20.boot_index(250)) and not np.array_equal(idx8, C24.C21.ST21.boot_index(250))
          and np.array_equal(idx8, ST24.boot_index(250)) and ST24.BOOT_SEED == 24)


# ---------------------------------------------------------------- rules, types, loader
def rule_tests(tmp):
    rows = [{"name": "a", "mean": 100.0, "trades": 50}, {"name": "b", "mean": 96.0, "trades": 30}, {"name": "c", "mean": 94.9, "trades": 5}, {"name": "d", "mean": 95.0, "trades": 30}]
    a1, r1 = ST24.select_astar(rows)
    a2, r2 = ST24.select_astar([{"name": "a", "mean": -3.0, "trades": 9}, {"name": "b", "mean": -3.0, "trades": 9}, {"name": "c", "mean": -8.0, "trades": 1}])
    a3, r3 = ST24.select_astar([{"name": "a", "mean": 0.0, "trades": 0}, {"name": "b", "mean": 0.0, "trades": 0}])
    check("A*: the highest mean; within 5% of it the fewest trades, then the earlier candidate (exactly 5% is inside, 5.1% is not); a mean of "
          "0 or below is 'no profit in development'", a1 == "b" and r1["tied_within_5pct"] == ["a", "b", "d"] and r1["profitable_in_development"]
          and a2 == "a" and not r2["profitable_in_development"] and a3 == "a" and not r3["profitable_in_development"], picks=[a1, a2, a3])
    try:
        ST24.pair_type("E2", "SPXU", "VOO")
        mixed = False
    except ValueError:
        mixed = True
    check("pair types (ANSWERS A5): family D2 = same company; both legs in the leveraged / inverse index list; both in the single-stock or "
          "crypto list; every other ETF pair = plain index ETF; one leg in a list and one outside is an error; the key ignores the order",
          ST24.pair_type("D2", "VIAB", "VIA") == "same_company" and ST24.pair_type("E2", "SPXU", "SPXS") == "leveraged_index_etf"
          and ST24.pair_type("E2", "SH", "SPDN") == "leveraged_index_etf" and ST24.pair_type("E2", "IBIT", "FBTC") == "single_stock_or_crypto_etf"
          and ST24.pair_type("E2", "NVDL", "NVDX") == "single_stock_or_crypto_etf" and ST24.pair_type("E2", "IWF", "VUG") == "index_etf"
          and ST24.pair_type("E2", "GLDM", "IAU") == "index_etf" and mixed and ST24.pair_key("SPXS", "SPXU") == ST24.pair_key("SPXU", "SPXS") == "SPXS/SPXU"
          and ST24.TYPES == ("index_etf", "leveraged_index_etf", "single_stock_or_crypto_etf", "same_company"))
    check("constants: 10,000 dollars per leg, delays 0, 1, 5 with 1 for the verdict, L3D1 primary, seed 24, alpha 0.05, 5 largest, trials "
          "847 + 4, the unlock phrase, the sealed window and the fixed sentences",
          ST24.AMOUNT == 10_000 and ST24.DELAYS == (0, 1, 5) and ST24.DELAY_MAIN == 1 and ST24.VAR_MAIN == "L3D1" and ST24.BOOT_SEED == 24 and ST24.ALPHA == 0.05
          and ST24.TOP == 5 and (ST24.TRIALS_BEFORE, ST24.TRIALS_ADDED) == (847, 4) and RT.PHRASE == "연구24 테스트를 한 번 연다" and RT.WINDOW == ("20210104", "20261110")
          and DV.SEALED_WINDOW == RT.WINDOW and L24.DEV_WINDOW == ("20150102", "20201231") and RT.MAIN == DV.MAIN == "d1_L3D1"
          and ST24.NO_PROFIT.startswith("1초 늦은 주문은 개발 구간에서") and len(ST24.SENTENCES) == 2 and S.LOT == 10 and PE.FEE_MKT == 0.3)
    # loader: the size unit and the development-only guard
    root = os.path.join(tmp, "loader", "dev", "bars_1s")
    b = T21.full(T20.spike_leg())
    b["bsz"][:], b["asz"][:] = 30_000, 20_000
    rows = {d: T21.save(root, "XX", d, b) for d in ("20201231", "20251031", "20251103")}
    bad = {c: v.copy() for c, v in b.items()}
    bad["asz"][777] = 20_050
    rb = T21.save(root, "YY", "20251103", bad)
    with_root = lambda fn: _with(L24, {"DEV_ROOT": root}, fn)                # noqa: E731
    got = {d: L24.read(os.path.join(root, r["file"]), r) for d, r in rows.items()}
    g = {}
    for name, fn in {"not a multiple of 100 from 2025-11-03": lambda: L24.read(os.path.join(root, rb["file"]), rb),
                     "another folder": lambda: L24.load(root, rows["20201231"]),
                     "date after 2020": lambda: with_root(lambda: L24.load(root, rows["20251103"])),
                     "sealed path": lambda: with_root(lambda: L24.load(os.path.join(tmp, "loader", "sealed", "bars_1s"), rows["20201231"]))}.items():
        try:
            fn()
            g[name] = False
        except L24.Refused:
            g[name] = True
        except OSError:
            g[name] = False
    ok_dev = with_root(lambda: np.array_equal(L24.load(root, rows["20201231"])["asz"], b["asz"]))
    ok_allow = np.array_equal(L24.load(root, rows["20251103"], True)["asz"], b["asz"] // 100)
    check("loader: a file dated up to 2025-10-31 is read as stored, from 2025-11-03 bsz and asz are divided by 100 (prices and volumes as "
          "stored), a size that is not a multiple of 100 stops the load; without allow_sealed it refuses another folder, a date after 2020 "
          "and a path under 'sealed'; the development bars of research 20 and 21 are the only other roots (regression)",
          all(np.array_equal(got[d][c], b[c]) for d in ("20201231", "20251031") for c in L24.COLS) and np.array_equal(got["20251103"]["bsz"], b["bsz"] // 100)
          and np.array_equal(got["20251103"]["ask"], b["ask"]) and all(g.values()) and ok_dev and ok_allow
          and L24.REGRESSION_ROOTS == (L21.R20_DEV_ROOT, L21.DEV_ROOT) and L24.read is L21.read, refused=g)


def _with(mod, values, fn):
    old = {k: getattr(mod, k) for k in values}
    for k, v in values.items():
        setattr(mod, k, v)
    try:
        return fn()
    finally:
        for k, v in old.items():
            setattr(mod, k, v)


# ---------------------------------------------------------------- the synthetic world
DEV_DAYS = ("2020-12-21", "2020-12-22", "2020-12-23", "2020-12-24", "2020-12-28", "2020-12-29", "2020-12-30", "2020-12-31")
SEALED_DAYS = ("2025-10-28", "2025-10-29", "2025-10-30", "2025-10-31", "2025-11-03", "2025-11-04", "2025-11-05", "2025-11-06")
UNI = (("E2", "P01", "EA", "EB"), ("E2", "P02", "SPXU", "SPXS"), ("E2", "P03", "IBIT", "FBTC"), ("D2", "P04", "DA", "DB"))
# dollars per trade (L0D0) at delays 0, 1, 5: a 14-tick gap of 30 s, of 1 s, a 10-tick gap of 3 s, a 10-tick level shift
LONG14, ONE14, THREE10, SHIFT10 = (8.8, 8.8, 8.8), (8.8, -5.2, -5.2), (4.8, 4.8, -5.2), (-5.2, -5.2, -5.2)
DA_PENALTY = 1.6                                             # DA/DB shows 60 shares: 40 above the size on each of four orders


def world_leg(sym, i, size):
    """Bars of one symbol on day number i (100 dollars, 2-tick spread). EA: a level shift of 10 ticks at 09:45:00 that stays
    (a losing trade for W 10 and kappa 2 only) and a 14-tick gap of 30 s at 11:30:00 (not on day 5); SPXU: a 14-tick gap
    of one second at 11:40:00 every day; IBIT: a 10-tick gap of 3 s at 11:50:00 on even days; DA: a 14-tick gap of 30 s at
    12:00:00 on odd days, 60 shares displayed on both legs of its pair. The B legs are flat."""
    if sym == "EA":
        b = T20.setq(T20.leg(size=size), 899, N, BID + 1000)
        return T20.setq(b, GE, GE + 30, BID + 2400) if i != 5 else b
    if sym == "SPXU":
        return T20.setq(T20.leg(size=size), GE + 600, GE + 601, BID + 1400)
    if sym == "IBIT":
        return T20.setq(T20.leg(size=size), GE + 1200, GE + 1203, BID + 1000) if i % 2 == 0 else T20.leg(size=size)
    small = size * 3 // 10
    if sym == "DA":
        return T20.setq(T20.leg(size=small), GE + 1800, GE + 1830, BID + 1400) if i % 2 == 1 else T20.leg(size=small)
    return T20.leg(size=small if sym == "DB" else size)


def expected(uni, n_days=8):
    """{candidate: {delay: (trades, total L3D1, total L0D0)}} of the world, from the per-trade values above."""
    cnt = {"EA": sum(i != 5 for i in range(n_days)), "SPXU": n_days, "IBIT": sum(i % 2 == 0 for i in range(n_days)), "DA": sum(i % 2 == 1 for i in range(n_days))}
    have = {a for _, _, a, _ in uni}
    out = {}
    for W, _, ka in ST24.candidates():
        r = {}
        for di, d in enumerate(ST24.DELAYS):
            n, l0, pen = 0, 0.0, 0.0
            for sym, per, k2_only in (("EA", LONG14, False), ("SPXU", ONE14, False), ("IBIT", THREE10, True), ("DA", LONG14, False)):
                if sym in have and not (k2_only and ka > 2):
                    n, l0 = n + cnt[sym], l0 + cnt[sym] * per[di]
                    pen += DA_PENALTY * cnt[sym] * (sym == "DA")
            if "EA" in have and W == 10 and ka == 2:
                n, l0 = n + n_days, l0 + n_days * SHIFT10[di]
            r[d] = (n, round(l0 - pen, 6), round(l0, 6))
        out[ST24.sig_name((W, ST24.Z_IN, ka))] = r
    return out


def build_world(tmp):
    repo, dev_root, sealed_root = os.path.join(tmp, "repo"), os.path.join(tmp, "dev", "bars_1s"), os.path.join(tmp, "sealed", "bars_1s")
    os.makedirs(os.path.join(repo, "results"))
    pd.DataFrame({"date": DEV_DAYS + SEALED_DAYS, "close_time": ["13:00" if d == "2020-12-24" else "16:00" for d in DEV_DAYS + SEALED_DAYS]}).to_csv(
        os.path.join(repo, "calendar24.csv"), index=False)
    write_universe(repo, UNI)
    for root, days, name in ((dev_root, DEV_DAYS, "DATA_MANIFEST24.tsv"), (sealed_root, SEALED_DAYS, "SEALED_MANIFEST24.tsv")):
        rows = []
        for i, d in enumerate(days):
            dd = d.replace("-", "")
            for s in sorted({x for _, _, a, b in UNI for x in (a, b)}):
                if (dd, s) == ("20201223", "FBTC"):
                    continue                                              # one leg missing: no pair-day for IBIT/FBTC on that day (an even day)
                rows.append(T21.save(root, s, dd, T21.full(world_leg(s, i, 20_000 if dd >= "20251103" else 200))))
        pd.DataFrame(rows)[["sym", "date", "file", "rows", "bytes", "sha256"]].to_csv(os.path.join(repo, name), sep="\t", index=False)
    for doc in RT.DOCS:
        open(os.path.join(repo, f"{doc}.md"), "w").write(f"{doc} stand-in\n")
        open(os.path.join(repo, f"{doc}.sha256"), "w").write(f"{RT.sha(os.path.join(repo, doc + '.md'))}  {doc}.md\n")
    return repo, dev_root, sealed_root


def write_universe(repo, uni):
    for y in (2020, 2025):
        pd.DataFrame([(y, f, p, leg, s, s) for f, p, a, b in uni for leg, s in (("A", a), ("B", b))],
                     columns=["year", "family", "pair", "leg", "sym", "symbol_15"]).to_csv(os.path.join(repo, f"universe24_{y}.tsv"), sep="\t", index=False)


def write_freeze(repo, drop=None):
    lines = []
    for p in RT.CORE:
        if p == drop:
            continue
        q = os.path.normpath(os.path.join(repo, p))
        if not os.path.exists(q):
            os.makedirs(os.path.dirname(q), exist_ok=True)
            open(q, "w").write(f"stand-in for {p}\n")
        lines.append(f"| `{p}` | `{RT.sha(q)}` | test |")
    open(RT.FREEZE, "w", encoding="utf-8").write("# FREEZE (synthetic)\n\n| file | sha256 | role |\n|---|---|---|\n" + "\n".join(lines) + "\n")


def totals(T):
    return {d: (T["runs"][f"d{d}_L3D1"]["trades"], round(T["runs"][f"d{d}_L3D1"]["total_usd"], 6), round(T["runs"][f"d{d}_L0D0"]["total_usd"], 6)) for d in ST24.DELAYS}


def world_tests(tmp):
    repo, dev_root, sealed_root = build_world(tmp)
    res = os.path.join(repo, "results")
    saved = {(m, k): getattr(m, k) for m, ks in ((L24, ("CALENDAR", "UNIVERSE", "DEV_MANIFEST", "DEV_ROOT")), (DV, ("RES", "LOCAL", "SEALED_WINDOW")),
                                                 (RT, ("RES", "DEV_SUMMARY", "SEALED_ROOT", "SEALED_BARS", "SEALED_MANIFEST", "WINDOW", "UNLOCK", "FREEZE", "HERE"))) for k in ks}
    try:
        L24.CALENDAR, L24.UNIVERSE = os.path.join(repo, "calendar24.csv"), os.path.join(repo, "universe24_{y}.tsv")
        L24.DEV_MANIFEST, L24.DEV_ROOT = os.path.join(repo, "DATA_MANIFEST24.tsv"), dev_root
        DV.RES, DV.LOCAL, DV.SEALED_WINDOW = res, os.path.join(tmp, "dev_results"), ("20251028", "20251106")
        RT.RES, RT.DEV_SUMMARY, RT.SEALED_ROOT, RT.SEALED_BARS = res, os.path.join(res, "dev24_summary.json"), os.path.dirname(sealed_root), sealed_root
        RT.SEALED_MANIFEST, RT.WINDOW = os.path.join(repo, "SEALED_MANIFEST24.tsv"), ("20251028", "20251106")
        RT.UNLOCK, RT.FREEZE, RT.HERE = os.path.join(repo, "UNLOCK24.txt"), os.path.join(repo, "FREEZE24.md"), repo
        _world(tmp, repo, res)
        # a second world without profit at a 1-second delay: only the pair whose gap lasts one second
        os.rename(res, os.path.join(repo, "results_world1"))
        os.makedirs(res)
        for f in ("FREEZE24.md", "UNLOCK24.txt"):
            os.rename(os.path.join(repo, f), os.path.join(repo, f + ".world1"))
        os.rename(DV.LOCAL, DV.LOCAL + "_world1")
        write_universe(repo, UNI[1:2])
        _world_no_profit(repo, res)
    finally:
        for (m, k), v in saved.items():
            setattr(m, k, v)


def _world(tmp, repo, res):
    tasks, days, skipped = L24.dev_plan()
    check("small world: the development plan has 31 pair-days (4 pairs x 8 days, the day with one leg missing is no pair-day), each with its "
          "family, pair type and key; the early close is read from the calendar",
          len(tasks) == 31 and len(days) == 8 and skipped == {"E2": {"one_leg": 1, "neither": 0}, "D2": {"one_leg": 0, "neither": 0}}
          and {t["close_bar"] for t in tasks if t["date"] == "20201224"} == {12_600}
          and {t["name"]: (t["family"], t["type"], t["key"]) for t in tasks} == {"EA/EB": ("E2", "index_etf", "EA/EB"), "SPXU/SPXS": ("E2", "leveraged_index_etf", "SPXS/SPXU"),
                                                                                 "IBIT/FBTC": ("E2", "single_stock_or_crypto_etf", "FBTC/IBIT"), "DA/DB": ("D2", "same_company", "DA/DB")})
    dev, err = quiet(DV.analyse, 0, check_prerequisites=False)
    dev = dev or {}
    want = expected(UNI)
    for name in want:                                                        # day 2 (an even day) has no IBIT/FBTC pair-day
        if name.endswith("k2"):
            want[name] = {d: (n - 1, round(a - THREE10[ST24.DELAYS.index(d)], 6), round(b - THREE10[ST24.DELAYS.index(d)], 6)) for d, (n, a, b) in want[name].items()}
    got = {name: totals(T) for name, T in dev.get("candidates", {}).items()}
    check("small world, development run: trades and total P&L of the 4 candidates at delays 0, 1 and 5 under L3D1 and L0D0 equal the "
          "hand-computed values (a gap of 30 s pays at every delay, of 3 s at delays 0 and 1, of 1 s at delay 0 only; the level shift loses "
          "for W10 kappa 2 only; 60 displayed shares cost 1.6 dollars a trade under L3D1)", err is None and got == want, error=err, got=got, want=want)
    A = dev.get("candidates", {}).get("W30_z3_k2", {})
    sel = dev.get("selection", {})
    check("small world, A*: W30_z3_k2 has the highest mean daily P&L at delay 1 under L3D1 ((7 x 8.8 - 8 x 5.2 + 3 x 4.8 + 4 x 7.2) / 8 = 7.9 "
          "dollars) and is profitable; W10_z3_k2 carries the losing level-shift trades; the power table and the trial count are written",
          dev.get("astar") == "W30_z3_k2" and dev["profitable_in_development"] and dev["sealed_run"] and sel["tied_within_5pct"] == ["W30_z3_k2"]
          and close(sel["astar_mean_usd"], (7 * 8.8 - 8 * 5.2 + 3 * 4.8 + 4 * 7.2) / 8) and dev["hypotheses"] == ["H1"] and dev["power"]["days_new"] == 8
          and dev["trials_cumulative"] == 851 and dev["checks"]["second_pass_equal"] is True and dev["checks"]["settings_with_m_trades_unlike_the_signal_timetable"] == 0
          and close(dev["prediction_third_line_delay0_L3D1_mean_usd"], got["W30_z3_k2"][0][1] / 8) and os.path.exists(os.path.join(tmp, "dev_results", "dev24_trades_W30_z3_k2_delay1.npz")),
          astar=dev.get("astar"), selection=sel)
    bt, bf = A.get("by_type", {}), A.get("by_family", {})
    r = lambda x, k: (x["runs"][k]["trades"], round(x["runs"][k]["total_usd"], 6))          # noqa: E731
    check("small world, splits of A* (delay 1, L3D1): plain index ETF 7 trades 61.6, leveraged 8 trades -41.6, single-stock or crypto 3 "
          "trades 14.4, same company 4 trades 28.8; family E2 = the three ETF types, D2 = same company; the splits add up to the whole",
          {k: r(v, "d1_L3D1") for k, v in bt.items()} == {"index_etf": (7, 61.6), "leveraged_index_etf": (8, -41.6), "single_stock_or_crypto_etf": (3, 14.4), "same_company": (4, 28.8)}
          and {k: r(v, "d1_L3D1") for k, v in bf.items()} == {"E2": (18, 34.4), "D2": (4, 28.8)} and close(sum(v["runs"]["d1_L3D1"]["mean_daily_usd"] for v in bt.values()), A["runs"]["d1_L3D1"]["mean_daily_usd"])
          and bt["same_company"]["runs"]["d1_L3D1"]["over_trades"] == 4 and close(A["runs"]["d1_L3D1"]["over_trade_frac"], 4 / 22) and A["runs"]["d1_L0D0"]["over_trades"] == 4
          and close(A["runs"]["d1_L3D1"]["over_share_frac"], 640 / 8800) and bt["index_etf"]["pair_days"] == 8 and bt["single_stock_or_crypto_etf"]["pair_days"] == 7)
    w = A.get("without_top", {}).get("d1_L3D1", {})
    daily = [8.8 * (i != 5) - 5.2 + 4.8 * (i % 2 == 0 and i != 2) + 7.2 * (i % 2 == 1) for i in range(8)]
    top = sorted(range(8), key=lambda i: (-daily[i], i))[:5]
    check("small world, the 5 largest days of A*: the three days with 10.8 dollars and the first two with 8.4 leave the sample and the mean "
          "of the three days left is (8.4 + 3.6 + 2.0) / 3; with only four pairs the 5 largest pairs are all of them and nothing is left",
          close(w["all"]["mean_daily_usd"], sum(daily) / 8) and close(w["without_top_days"]["mean_daily_usd"], (sum(daily) - sum(daily[i] for i in top)) / 3)
          and w["without_top_days"]["days"] == 3 and len(w["without_top_pairs"]["pairs_removed"]) == 4 and w["without_top_pairs"]["mean_daily_usd"] == 0.0
          and close(w["without_top_pairs"]["share_of_total"], 1.0) and w["without_both"]["mean_daily_usd"] == 0.0
          and A["date_split"]["d1_L3D1"]["from"] is None and A["date_split"]["d1_L3D1"]["before"]["days"] == 8
          and set(A["by_pair_key"]) == {"EA/EB", "SPXS/SPXU", "FBTC/IBIT", "DA/DB"} and close(A["by_pair_key"]["SPXS/SPXU"]["total_usd"], -41.6), daily=daily, got=w)
    dg = dev.get("diag", {}).get("family", {})
    check("small world, diagnostics: kappa 3 keeps out the 10-tick gaps that kappa 2 lets in (more |z| >= 3 points are kept out), and the "
          "sigma floor applies to most signal points", dg["kappa_blocked"]["W30_z3_k3"]["kappa_blocked"] > dg["kappa_blocked"]["W30_z3_k2"]["kappa_blocked"]
          and dg["kappa_blocked"]["W30_z3_k2"]["z_points"] == dg["kappa_blocked"]["W30_z3_k3"]["z_points"] > 0
          and all(0.5 < dg["sigma_floor"][w_]["share"] <= 1 for w_ in ("W10", "W30")), diag=dg)
    dd = pd.read_csv(os.path.join(res, "dev24_daily.csv"), dtype={"date": str}) if os.path.exists(os.path.join(res, "dev24_daily.csv")) else pd.DataFrame()
    gs = dd.groupby("setting").sum(numeric_only=True) if len(dd) else None
    _, again = quiet(DV.analyse, 0, check_prerequisites=False)
    check("small world: the development run writes its summary, report and daily file once and refuses a second run; the daily file has one "
          "row per pair-day and candidate (31 x 4) with the columns of the sealed run's daily file, and its sums are the tables",
          all(os.path.exists(os.path.join(res, f)) for f in DV.FILES) and again is not None and "once" in again and list(dd.columns) == list(C24.daily_cols(ST24.DELAYS))
          and len(dd) == 31 * 4 and all(int(gs.loc[n, f"trades_d{d}"]) == want[n][d][0] and close(gs.loc[n, f"d{d}_L3D1"], want[n][d][1], 1e-6)
                                        and close(gs.loc[n, f"d{d}_L0D0"], want[n][d][2], 1e-6) for n in want for d in ST24.DELAYS)
          and dev["checks"]["daily_file_gap"]["trades"] == 0 and dev["checks"]["daily_file_gap"]["usd"] < 1e-9 and dev["daily"]["rows"] == 124, second_run=again)
    # ---- rehearsal
    reh, err = quiet(RT.main, "rehearsal", 0)
    reh = reh or {}
    k = reh.get("checks", {}).get("rehearsal", {})
    rd = pd.read_csv(os.path.join(res, "rehearsal24_daily.csv"), dtype={"date": str}) if reh else pd.DataFrame()
    check("small world, rehearsal: run_test24's code path on the development part reproduces A*'s tables of the development run exactly "
          "(largest difference 0), reads no sealed path, writes one daily row per pair-day (31) and adds no trial",
          err is None and k.get("pass") is True and k["max_abs_diff"] == 0.0 and k["numbers_compared"] > 300 and k["no_sealed_path_read"]
          and reh["astar"] == "W30_z3_k2" and reh["trials_added"] == 0 and len(rd) == 31 and reh["read"].endswith(os.path.join("dev", "bars_1s")), error=err, checks=k)
    # ---- guard
    g = {"no FREEZE, no unlock": RT.guard()}
    write_freeze(repo)
    g["no unlock file"] = RT.guard()
    open(RT.UNLOCK, "w", encoding="utf-8").write("연구24 테스트를 연다\n")
    g["wrong phrase"] = RT.guard()
    open(RT.UNLOCK, "w", encoding="utf-8").write(RT.PHRASE + "\n")
    g["all in place"] = RT.guard()
    open(os.path.join(repo, "SPEC24_ANSWERS.md"), "a").write("x")
    g["document changed"] = RT.guard()
    open(os.path.join(repo, "SPEC24_ANSWERS.md"), "w").write("SPEC24_ANSWERS stand-in\n")
    core = os.path.join(repo, "pair24.py")
    keep = open(core).read()
    open(core, "w").write(keep + "#\n")
    g["frozen file changed"] = RT.guard()
    open(core, "w").write(keep)
    write_freeze(repo, drop="stats24.py")
    g["core file not listed"] = RT.guard()
    write_freeze(repo)
    g["all in place again"] = RT.guard()
    check("guard of the sealed run: refuses without the unlock file, with another phrase, with a changed document, a changed frozen file "
          "or a core file missing from FREEZE24.md; passes only when everything is in place",
          g["no unlock file"] == ["no unlock phrase in UNLOCK24.txt"] == g["wrong phrase"] and g["all in place"] == [] == g["all in place again"]
          and "FREEZE24.md missing or without the core files" in g["no FREEZE, no unlock"] and g["document changed"] == ["SPEC24_ANSWERS.md differs from SPEC24_ANSWERS.sha256"]
          and g["frozen file changed"] == ["frozen file differs: pair24.py"] and g["core file not listed"] == ["FREEZE24.md missing or without the core files"], guard=g)
    txt = open(RT.DEV_SUMMARY, encoding="utf-8").read()
    d0 = json.loads(txt)
    cases = {"another A*": {**d0, "astar": "W30_z3_k3", "astar_signal": [30, 3.0, 3.0], "selection": {**d0["selection"], "astar": "W30_z3_k3"}},
             "profit flag flipped": {**d0, "profitable_in_development": False},
             "one number changed": {**d0, "power": {**d0["power"], "dev_effect_usd": d0["power"]["dev_effect_usd"] + 1e-6}}}
    gg, usable = {}, {}
    for name, v in cases.items():
        with open(RT.DEV_SUMMARY, "w", encoding="utf-8") as fh:
            json.dump(v, fh, indent=1, ensure_ascii=False)
        gg[name] = RT.guard()
        try:
            RT.selection()
            usable[name] = True
        except Exception:
            usable[name] = False
    open(RT.DEV_SUMMARY, "w", encoding="utf-8").write(txt + "\n")
    gg["one newline appended"] = RT.guard()
    with open(RT.DEV_SUMMARY, "w", encoding="utf-8") as fh:
        json.dump(cases["another A*"], fh, indent=1, ensure_ascii=False)
    _, refused = quiet(RT.main, "sealed", 0)
    nothing = not any(os.path.exists(os.path.join(res, f)) for f in RT.MODES["sealed"]["files"])
    open(RT.DEV_SUMMARY, "w", encoding="utf-8").write(txt)
    back = RT.guard()
    check("a development summary whose content alone changes (another A*, the profit flag flipped, one number, one newline) stops the guard "
          "with 'frozen file differs', the sealed run writes nothing, and the original content passes again",
          all(v[:1] == ["frozen file differs: results/dev24_summary.json"] for v in gg.values()) and usable["another A*"] and not usable["profit flag flipped"]
          and any("no usable development selection" in x for x in gg["profit flag flipped"]) and refused is not None and refused.startswith("refused") and nothing and back == [],
          guard=gg, readable_by_selection=usable, sealed_run=refused)
    # ---- the sealed run
    t, err = quiet(RT.main, "sealed", 0)
    t = t or {}
    ws = expected(UNI)["W30_z3_k2"]
    td = pd.read_csv(os.path.join(res, "test24_daily.csv"), dtype={"date": str}) if t else pd.DataFrame()
    v = t.get("verdict", {}).get("H1", {})
    sp = t.get("date_split", {}).get("d1_L3D1", {})
    check("small world, sealed run with the unlock phrase: 32 pair-days of 2025, A* alone at delays 0, 1 and 5 with the hand-computed totals; "
          "H1 = mean daily P&L at delay 1 under L3D1 (8.5 dollars) with its one-sided p at 0.05 and the fixed sentence; sizes stored 100 x "
          "from 2025-11-03 give the same 1.6-dollar penalty in both parts; the split at 2025-11-03, the types, the daily file; one trial",
          err is None and t.get("mode") == "sealed" and totals(t) == ws and t["counts"]["pair_days"] == 32 and t["hypotheses"] == ["H1"]
          and close(v["mean_daily_usd"], ws[1][1] / 8) and v["p"] == t["runs"]["d1_L3D1"]["p_one_sided"] and v["level"] == 0.05 and v["confirmed"] == (v["p"] < 0.05)
          and v["sentence"] == ST24.SENTENCES[v["confirmed"]] and sp["before"]["days"] == 4 and sp["from"]["days"] == 4
          and close(sp["before"]["mean_daily_usd"] * 4 + sp["from"]["mean_daily_usd"] * 4, ws[1][1]) and close(t["by_type"]["same_company"]["runs"]["d1_L0D0"]["total_usd"] - t["by_type"]["same_company"]["runs"]["d1_L3D1"]["total_usd"], 4 * DA_PENALTY)
          and t["by_type"]["same_company"]["runs"]["d1_L3D1"]["over_trades"] == 4 and len(td) == 32 and list(td.columns) == list(C24.daily_cols(ST24.DELAYS))
          and t["trials_added"] == 1 and t["trials_before"] == 851 and t["read"].endswith(os.path.join("sealed", "bars_1s")) and "rehearsal" not in t["checks"],
          error=err, verdict=v, totals=totals(t) if t else None)
    _, again = quiet(RT.main, "sealed", 0)
    check("small world: a second sealed run is refused (test results already exist)", again is not None and "test results already exist" in again, message=again)


def _world_no_profit(repo, res):
    dev, err = quiet(DV.analyse, 0, check_prerequisites=False)
    dev = dev or {}
    got = {name: totals(T) for name, T in dev.get("candidates", {}).items()}
    check("second world (only the pair whose gap lasts one second): every candidate makes 8.8 dollars a trade at delay 0 and loses 5.2 at "
          "delays 1 and 5; A* is the first candidate (all equal), its mean is below 0: 'no profit in development', no hypothesis, the fixed sentence",
          err is None and got == expected(UNI[1:2]) and dev["astar"] == "W10_z3_k2" and not dev["profitable_in_development"] and not dev["sealed_run"]
          and dev["hypotheses"] == [] and dev["sentence_no_profit"] == ST24.NO_PROFIT and close(dev["selection"]["astar_mean_usd"], -5.2)
          and dev["prediction_third_line_delay0_L3D1_mean_usd"] > 0, error=err, got=got)
    reh, err = quiet(RT.main, "rehearsal", 0)
    write_freeze(repo)
    open(RT.UNLOCK, "w", encoding="utf-8").write(RT.PHRASE + "\n")
    g = RT.guard()
    _, refused = quiet(RT.main, "sealed", 0)
    check("second world: with everything else in place the guard refuses the sealed run for one reason, no profit in development, and "
          "nothing is written (the rehearsal still runs and equals the development run)",
          g == [RT.NOT_PROFITABLE] and refused is not None and RT.NOT_PROFITABLE in refused and not any(os.path.exists(os.path.join(res, f)) for f in RT.MODES["sealed"]["files"])
          and err is None and reh["checks"]["rehearsal"]["pass"] and reh["checks"]["rehearsal"]["max_abs_diff"] == 0.0, guard=g, sealed_run=refused)


def repo_guard_test():
    why = RT.guard()
    check("the real sealed run is refused now (no UNLOCK24.txt; Claude never writes it)", "no unlock phrase in UNLOCK24.txt" in why
          and not os.path.exists(os.path.join(HERE, "UNLOCK24.txt")), reasons=why)


def main():
    with tempfile.TemporaryDirectory(prefix="r24_tests_") as tmp:
        r20, r21 = earlier_tests(tmp)
        frozen_tests()
        amount_tests()
        delay_tests()
        bound_tests()
        over_tests()
        top_tests()
        rule_tests(tmp)
        world_tests(os.path.join(tmp, "world"))
    repo_guard_test()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    n_ok = sum(r["pass"] for r in RESULTS)
    path = os.path.join(HERE, "results", "engine24_tests.json")
    C24.keep_old(path)
    with open(path, "w") as fh:
        json.dump({"passed": n_ok, "total": len(RESULTS), "research20_tests": r20, "research21_tests": r21, "numba": bool(PE.G.HAVE_NUMBA), "tests": RESULTS},
                  fh, indent=1, default=str, ensure_ascii=False)
    for r in RESULTS:
        print(("PASS " if r["pass"] else "FAIL ") + r["test"][:150])
        if not r["pass"]:
            print("     ", json.dumps(r, default=str, ensure_ascii=False)[:3000])
    print(f"{n_ok}/{len(RESULTS)} passed (research 20's tests: {r20}, research 21's tests: {r21})")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
