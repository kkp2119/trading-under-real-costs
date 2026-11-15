#!/usr/bin/env python3
"""Research 22 synthetic tests (step-2 prompt 1; they must pass before any development run). Synthetic bars only; no bar
of research 22, of the development universe or of SPY is read, and the sealed folders are not named outside temporary
stand-ins.
  earlier   research 18's 78 engine tests and research 20's 117 signal tests, rerun here (same names as their stored results)
  frozen    engine17, engine18, signal20 and loader15 equal the files of their tags
  flags     sign (buy: z <= -z_th fires the early fill, z >= z_th the hold; sell the other way round) and the boundary
  engine    early fill at the bar-end quote with the D1 penalty, hold skips E*'s immediate fill, thresholds 1 and 2
  no signal window not full, no SPY file, an invalid quote: fills equal E* exactly
  future    values after a decision bar do not change that decision
  bases     the SPY bars do not change the own-price results
  sizes     2025-10-31 as stored, 2025-11-03 divided by 100, a value that is not a multiple of 100 stops the load
  rules     A*: the 0.02 cent threshold against G0 and the tie rule among candidates; Holm with one and two hypotheses
  control   SPEC22_ADDENDUM D1: the run with the reversed sign fires exactly where the candidate holds and the other way round
  stats     bootstrap seed, visited decision bars, the prediction sums against a direct loop, the daily file
  world     a small synthetic world through the development run, the rehearsal, the guard and the sealed run, once with the
            selection as it falls and once with a forced A* (the H2 path); a changed development summary stops the guard
Writes results/engine22_tests.json (an earlier file is moved to results/superseded/); exits 1 when a test fails.
  python test22.py"""
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import core22 as C  # noqa: E402
import dev22 as DV  # noqa: E402
import loader22 as L22  # noqa: E402
import run_test22 as RT  # noqa: E402
import signal22 as SG  # noqa: E402
import test_engine18 as T18  # noqa: E402  (research 18's test functions; its main is never called)
import test20 as T20  # noqa: E402  (research 20's test functions; its main is never called)

E, S20, L15, NSEC = C.E, SG.S20, L22.L15, L22.NSEC
BID, SPR, SZ = 1_000_000, 800, 5000        # flat quote 100.00 / 100.08: E* posts at 100.01 and nothing fills
U0 = 6004                                  # a decision bar well after the window is full (11:10:05)
RESULTS = []


def check(name, ok, **info):
    RESULTS.append({"test": name, "pass": bool(ok), **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}})


def keep_old(path):
    if os.path.exists(path):
        d = os.path.join(os.path.dirname(path), "superseded")
        os.makedirs(d, exist_ok=True)
        stem, ext = os.path.splitext(os.path.basename(path))
        q = os.path.join(d, f"{stem}_{dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y%m%d_%H%M%S')}{ext}")
        if os.path.exists(q):
            raise RuntimeError(f"{q} exists")
        os.rename(path, q)


# ---------------------------------------------------------------- synthetic bars
def flat(bid=BID, spread=SPR, size=SZ):
    b = {c: np.zeros(NSEC, np.int64) for c in L22.COLS}
    for k in ("bid", "bid_max", "bid_min"):
        b[k][:] = bid
    for k in ("ask", "ask_max", "ask_min"):
        b[k][:] = bid + spread
    b["bsz"][:], b["asz"][:] = size, size
    return b


def setq(b, u0, u1, bid, ask):
    for k in ("bid", "bid_max", "bid_min"):
        b[k][u0:u1] = bid
    for k in ("ask", "ask_max", "ask_min"):
        b[k][u0:u1] = ask
    return b


def rand_day(seed, bid0=BID, tick_p=0.3):
    """A random day with trades: ordered quotes, sizes in round lots, the volume identity holds."""
    rng = np.random.default_rng(seed)
    mid = bid0 + 100 * np.cumsum(rng.integers(-1, 2, NSEC) * (rng.random(NSEC) < tick_p))
    spr = 100 * rng.choice((1, 2, 2, 3, 4, 6), NSEC)
    b = {c: np.zeros(NSEC, np.int64) for c in L22.COLS}
    b["bid"] = (mid - spr // 2) // 100 * 100
    b["ask"] = b["bid"] + spr
    b["bid_max"] = b["bid"] + 100 * (rng.random(NSEC) < 0.2)
    b["bid_min"] = b["bid"] - 100 * (rng.random(NSEC) < 0.2)
    b["ask_max"] = b["ask"] + 100 * (rng.random(NSEC) < 0.2)
    b["ask_min"] = b["ask"] - 100 * (rng.random(NSEC) < 0.2)
    b["bsz"], b["asz"] = rng.integers(1, 30, NSEC) * 100, rng.integers(1, 30, NSEC) * 100
    b["vol_bid"], b["vol_ask"] = rng.integers(0, 8, NSEC) * 100, rng.integers(0, 8, NSEC) * 100
    b["vol_bid_odd"], b["vol_ask_odd"] = (rng.random(NSEC) < 0.2) * 37, (rng.random(NSEC) < 0.2) * 41
    b["px_min"], b["px_max"] = b["bid"] - 100, b["ask"] + 100
    b["vol"] = b["vol_bid"] + b["vol_ask"]
    return b


def run1(b, ref, runs, dirs, starts, delay_runs=(), bases=None, close=NSEC):
    return C.run_parents(b, ref, close, np.array(dirs, np.int64), np.array(starts, np.int64), runs, delay_runs, bases)


def col(r, names, run, lr=3):
    return r["outs"][E.LRULES.index(lr), list(names).index(run)]


# ---------------------------------------------------------------- earlier studies
def earlier_tests():
    T18.RESULTS.clear()
    for fn in (T18.equivalence_tests, T18.feature_tests, T18.future_tests, T18.threshold_tests, T18.d1_test, T18.r2_keeps_limit,
               T18.delay_tests, T18.stats_tests):
        fn()
    s18 = json.load(open(os.path.join(PROJ, "study18_forecast", "results", "engine18_tests.json")))
    n18 = sum(r["pass"] for r in T18.RESULTS)
    check("research 18's engine tests pass here (78, the names of its stored result)",
          len(T18.RESULTS) == 78 == n18 and [r["test"] for r in T18.RESULTS] == [r["test"] for r in s18["tests"]],
          passed=n18, total=len(T18.RESULTS), failed=[r["test"] for r in T18.RESULTS if not r["pass"]])
    T20.RESULTS.clear()
    for fn in (T20.signal_tests, T20.kernel_tests, T20.pair_tests, T20.timetable_tests, T20.stats_tests, T20.gap_tests, T20.delay_tests):
        fn()
    s20 = json.load(open(os.path.join(PROJ, "study20_pairs", "results", "engine20_tests.json")))
    n20 = sum(r["pass"] for r in T20.RESULTS)
    check("research 20's signal and engine tests pass here (117, the names of its stored result)",
          len(T20.RESULTS) == 117 == n20 and [r["test"] for r in T20.RESULTS] == [r["test"] for r in s20["tests"]],
          passed=n20, total=len(T20.RESULTS), failed=[r["test"] for r in T20.RESULTS if not r["pass"]])
    return f"{n18}/{len(T18.RESULTS)}", f"{n20}/{len(T20.RESULTS)}"


def frozen_tests():
    got = {}
    for p, (tag, h) in C.FROZEN.items():
        blob = subprocess.run(["git", "show", f"{tag}:{p}"], capture_output=True, cwd=PROJ).stdout
        got[p] = (hashlib.sha256(blob).hexdigest() == h, hashlib.sha256(open(os.path.join(PROJ, p), "rb").read()).hexdigest() == h)
    check("engine17, engine18, signal20 and loader15 equal the files of their tags (git) and the working files", all(a and b for a, b in got.values()),
          files={p: list(v) for p, v in got.items()}, tags={p: t for p, (t, _) in C.FROZEN.items()})


# ---------------------------------------------------------------- flags
def flag_tests():
    z = np.array([[2.5, -2.5, 0.3, np.nan]])
    a_buy, a_sell = SG.adverse(z, np.array([1])), SG.adverse(z, np.array([-1]))
    check("adverse score: -z for a buy order, +z for a sell order, nan stays nan",
          np.array_equal(a_buy, -z, equal_nan=True) and np.array_equal(a_sell, z, equal_nan=True), buy=a_buy, sell=a_sell)
    fb, fs = SG.flags(a_buy, 2.0), SG.flags(a_sell, 2.0)
    check("flags: a buy fires the early fill at z <= -z_th and the hold at z >= z_th; a sell the other way round; nan without a signal",
          np.array_equal(fb, [[SG.HOLD, SG.FIRE, SG.NONE, np.nan]], equal_nan=True) and np.array_equal(fs, [[SG.FIRE, SG.HOLD, SG.NONE, np.nan]], equal_nan=True),
          buy=fb, sell=fs)
    lo, hi = np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0)
    a = np.array([[1.0, lo, hi, -1.0, -lo, -hi, 2.0, np.nextafter(2.0, 0.0)]])
    f1, f2 = SG.flags(a, 1.0), SG.flags(a, 2.0)
    check("boundary: a equal to z_th fires, a just below does not (both conditions, thresholds 1 and 2)",
          np.array_equal(f1, [[1.0, 0.5, 1.0, 0.0, 0.5, 0.0, 1.0, 1.0]]) and np.array_equal(f2, [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5]]), z1=f1, z2=f2)
    swap = {SG.FIRE: SG.HOLD, SG.HOLD: SG.FIRE, SG.NONE: SG.NONE}
    az = np.array([[2.5, -2.5, 1.0, -1.0, 0.3, np.nan]])
    fa, fr = SG.flags(az, 1.0), SG.flags(-az, 1.0)
    check("control (ADDENDUM D1): with the sign of a reversed the early condition and the hold condition change places, the rest stays",
          np.array_equal(fr[~np.isnan(fr)], [swap[v] for v in fa[~np.isnan(fa)]]) and np.isnan(fr[0, -1]) and C.flip_name("own_z2_G3") == "flip_own_z2_G3"
          and C.parse("flip_own_z2_G3") == ("own", 2.0, "G3") and C.engine_name("flip_spy_z1_G2") == "R2_0.25" and C.is_flip("flip_spy_z1_G1")
          and not C.is_flip("spy_z1_G1") and C.signalled("half_G1") is None and C.bases_of(("G0", "flip_spy_z1_G1")) == ("spy",), flags=fa, reversed=fr)
    b = flat()
    P = np.full((1, E.NG), SG.NONE)
    base = E.run(b, "R0", 3, np.array([1]), np.array([U0 - 50]), None, 0)
    same = all(np.array_equal(E.run(b, n, 3, np.array([1]), np.array([U0 - 50]), P, 0)[:, :9], base[:, :9]) for n in C.ENGINE.values())
    P1 = P.copy()
    P1[0, 10] = SG.FIRE
    o1 = E.run(b, C.ENGINE["G1"], 3, np.array([1]), np.array([U0 - 50]), P1, 0)[0]
    check("engine: flags 0.5 never act (R1_0.75, R2_0.25, R3_0.75_0.25 equal E*); one flag 1.0 sends the rest to market at that bar",
          same and o1[9] == E.QTY and o1[5] == 50 and o1[3] == E.QTY * b["ask"][U0], r1=o1)


# ---------------------------------------------------------------- engine scenarios
def dip(ticks=3.0, back=None, spread=SPR):
    """Stock flat at 100.00 / 100.04; from U0 on both sides lower by `ticks` cents (back at bar `back` when given)."""
    b = flat(spread=spread)
    d = int(round(ticks * 100))
    setq(b, U0, NSEC if back is None else back, BID - d, BID + spread - d)
    return b


def engine_tests():
    names = C.BASE + C.CANDS
    s = U0 - 50
    r = run1(dip(3.0), flat(bid=3_000_000, spread=100), names, [1, -1], [s, s])
    g0 = col(r, names, "G0")
    ok, info = True, {}
    for c in C.CANDS:
        _, _, rule = C.parse(c)
        o = col(r, names, c)
        buy_early = o[0, 9] == E.QTY and o[0, 5] == 50 and o[0, 3] == E.QTY * (BID + SPR - 300)
        want = buy_early if rule in ("G1", "G3") else np.array_equal(o[0, :9], g0[0, :9])
        ok &= bool(want and np.array_equal(o[1, :9], g0[1, :9]) and o[1, 9] == 0)
        info[c] = [int(o[0, 9]), int(o[0, 5]), int(o[1, 9]), int(o[1, 10])]
    check("a 3-cent dip (z = -3): the buy order is filled early at that bar's ask by G1 and G3 under both bases and thresholds; the sell "
          "order and G2 stay with E*", ok, a_buy=r["A"]["spy"][0, 8:12], a_sell=r["A"]["spy"][1, 8:12], runs=info)
    fl = tuple(C.flip_name(c) for c in ("own_z1_G1", "spy_z2_G3", "own_z1_G2"))
    rf = run1(dip(3.0), flat(bid=3_000_000, spread=100), ("G0",) + fl, [1, -1], [s, s])
    gf = col(rf, ("G0",) + fl, "G0")
    o1, o3, o2 = (col(rf, ("G0",) + fl, f) for f in fl)
    sell_early = lambda o: o[1, 9] == E.QTY and o[1, 5] == 50 and o[1, 3] == E.QTY * (BID - 300)      # noqa: E731  (the bid of bar U0)
    check("control in the 3-cent dip: the reversed sign leaves the buy order with E* and fills the sell order early at that bar's bid "
          "(G1 and G3); the G2 control does nothing here", bool(sell_early(o1) and sell_early(o3) and np.array_equal(o1[0, :9], gf[0, :9])
          and np.array_equal(o3[0, :9], gf[0, :9]) and o1[0, 9] == 0 and np.array_equal(o2[:, :9], gf[:, :9])), control=[o1[:, 9], o3[:, 9], o2[:, 10]])
    r15 = run1(dip(1.5), flat(bid=3_000_000, spread=100), names, [1], [s])
    f = {c: int(col(r15, names, c)[0, 9]) for c in C.CANDS if C.parse(c)[2] == "G1"}
    check("a 1.5-cent dip (z = -1.5): threshold 1 fires, threshold 2 does not", all((v == E.QTY) == (C.parse(c)[1] == 1.0) for c, v in f.items()),
          a=r15["A"]["own"][0, 10], early_shares=f)
    # the rise with a 2-cent spread from U0: E* takes the spread at once; a buy order sees a <= -z_th (hold), a sell a >= z_th
    b = flat()
    setq(b, U0, NSEC, BID + 600, BID + 800)                                   # bid up 6 cents, ask unchanged: mid +3 cents, spread 2 cents
    r2 = run1(b, flat(bid=3_000_000, spread=100), names, [1, -1], [s, s])
    g0 = col(r2, names, "G0")
    ok, info = bool(g0[0, 5] == 50 and g0[0, 2] == E.QTY and g0[1, 5] == 50), {}
    for c in C.CANDS:
        rule = C.parse(c)[2]
        o = col(r2, names, c)
        held = o[0, 10] >= 1 and o[0, 5] > 50 and o[0, 9] == 0
        ok &= bool((held if rule in ("G2", "G3") else np.array_equal(o[0, :9], g0[0, :9])) and np.array_equal(o[1, :9], g0[1, :9]) and o[1, 9] == 0)
        info[c] = [int(o[0, 10]), int(o[0, 5]), int(o[1, 9])]
    check("a 3-cent rise of the mid with a 2-cent spread: G2 and G3 skip E*'s immediate fill for the buy order (hold); the sell order's early "
          "condition coincides with E*'s market order and is not counted as a G1 action", ok, e_star=g0[:, [2, 5]], runs=info)
    # D1: an early fill larger than the displayed size pays one tick more on the excess under L3D1, nothing more under L0D0
    b = dip(3.0)
    b["asz"][U0] = 300
    rd = run1(b, None, ("G0", "own_z1_G1"), [1], [s])
    o2 = col(rd, ("G0", "own_z1_G1"), "own_z1_G1")
    o = o2[0]
    mid2 = np.array([b["bid"][s] + b["ask"][s]])
    c1 = float(E.costs(o2, np.array([1]), mid2, 1)[0])
    c0 = float(E.costs(col(rd, ("G0", "own_z1_G1"), "own_z1_G1", lr=0), np.array([1]), mid2, 0)[0])
    px = (BID + SPR - 300 - mid2[0] / 2) / 100
    check("early fill above the displayed size: 700 shares one tick worse under L3D1, the bar-end ask and the fee otherwise",
          o[4] == 700 and abs(c1 - (px + 0.7 * 1.0 + E.FEE_MKT)) < 1e-9 and abs(c0 - (px + E.FEE_MKT)) < 1e-9, over=int(o[4]), l3d1=c1, l0d0=c0)
    # delay 1: the early fill decided at U0 is done at U0 + 1 with that bar's quote
    b = dip(3.0)
    setq(b, U0 + 1, U0 + 2, BID - 100, BID + 300)
    rdl = run1(b, None, ("G0", "own_z1_G1"), [1], [s], delay_runs=("own_z1_G1",))
    od = rdl["outs"][E.LRULES.index(3), 2][0]
    check("delay check: the early fill decided at bar t is done at bar t + 1 with that bar's ask", od[9] == E.QTY and od[5] == 51 and od[3] == E.QTY * (BID + 300),
          out=od)


def no_signal_tests():
    names = C.BASE + C.CANDS
    early = flat()
    setq(early, 1004, NSEC, BID - 300, BID + SPR - 300)                      # a dip at 09:46:45: the window is not full yet
    r = run1(early, flat(bid=3_000_000, spread=100), names, [1, -1], [954, 954])
    g0 = col(r, names, "G0")
    check("window not full (before 10:00:05): no signal, every candidate fills exactly as E*",
          all(np.array_equal(col(r, names, c), g0) for c in C.CANDS) and np.isnan(r["A"]["spy"]).all() and np.isnan(r["A"]["own"]).all()
          and r["sig"]["own"][1] == 0, with_signal=[r["sig"]["spy"][1], r["sig"]["own"][1]])
    s = U0 - 50
    r = run1(dip(3.0), None, names, [1, -1], [s, s])
    g0 = col(r, names, "G0")
    spy = all(np.array_equal(col(r, names, c), g0) for c in C.CANDS if C.parse(c)[0] == "spy")
    own = col(r, names, "own_z1_G1")[0, 9] == E.QTY
    check("no SPY file: the SPY-basis candidates fill exactly as E* (no signal all day); the own-price basis still works",
          spy and own and r["sig"]["spy"][1] == 0 and np.isnan(r["A"]["spy"]).all())
    locked = flat(bid=3_000_000, spread=0)
    r = run1(dip(3.0), locked, names, [1, -1], [s, s])
    g0 = col(r, names, "G0")
    check("SPY quote locked all day (not valid): the SPY-basis candidates fill exactly as E*",
          all(np.array_equal(col(r, names, c), g0) for c in C.CANDS if C.parse(c)[0] == "spy") and r["sig"]["spy"][1] == 0)
    b = dip(3.0)
    setq(b, U0, U0 + 1, BID - 300, BID - 300)                                 # the stock quote is locked at the decision bar itself
    r = run1(b, flat(bid=3_000_000, spread=100), names, [1, -1], [s, s])
    g0 = col(r, names, "G0")
    check("stock quote not valid at the decision bar: no signal there, every candidate does what E* does at that bar",
          bool(np.isnan(r["A"]["own"][:, 10]).all() and np.isnan(r["A"]["spy"][:, 10]).all()
               and all(np.array_equal(col(r, names, c)[:, :9], g0[:, :9]) for c in C.CANDS)), e_star=g0[:, [2, 5]])


def future_tests():
    names = C.BASE + C.CANDS
    b, ref = rand_day(1), rand_day(2, bid0=3_000_000, tick_p=0.5)
    dirs, starts = E.parents("20190301", NSEC)
    cut = int(np.sort(starts)[10]) + 150                                       # some parents end before the cut, some straddle it
    b2, ref2 = {k: v.copy() for k, v in b.items()}, {k: v.copy() for k, v in ref.items()}
    for x in (b2, ref2):
        for k in ("bid", "ask", "bid_max", "bid_min", "ask_max", "ask_min"):
            x[k][cut + 1:] += 5000
        x["vol_bid"][cut + 1:] = 0
    r1, r2 = run1(b, ref, names, dirs, starts), run1(b2, ref2, names, dirs, starts)
    U = starts[:, None] + SG.STEP * np.arange(SG.NG)[None, :]
    before = U <= cut
    same_a = all(np.array_equal(r1["A"][k][before], r2["A"][k][before], equal_nan=True) for k in C.BASES)
    changed = any(not np.array_equal(r1["A"][k][~before], r2["A"][k][~before], equal_nan=True) for k in C.BASES)
    done = starts + E.HOR <= cut
    same_o = bool(np.array_equal(r1["outs"][:, :, done], r2["outs"][:, :, done]))
    check("no look-ahead: changing every value after a bar leaves the scores of the decision bars up to it, and the orders that end by "
          "it, unchanged (the later scores do change)", same_a and changed and same_o and done.sum() >= 5 and (~done).sum() >= 5,
          parents_done_by_cut=int(done.sum()), decision_bars_before=int(before.sum()))


def basis_tests():
    names = C.BASE + C.CANDS
    b, ref, ref2 = rand_day(3), rand_day(4, bid0=3_000_000, tick_p=0.5), rand_day(5, bid0=2_500_000, tick_p=0.5)
    dirs, starts = E.parents("20190304", NSEC)
    r1, r2 = run1(b, ref, names, dirs, starts), run1(b, ref2, names, dirs, starts)
    own = [names.index(c) for c in C.CANDS if C.parse(c)[0] == "own"]
    spy = [names.index(c) for c in C.CANDS if C.parse(c)[0] == "spy"]
    check("the bases do not mix: other SPY bars leave the own-price scores and fills unchanged and change the SPY-basis ones",
          np.array_equal(r1["A"]["own"], r2["A"]["own"], equal_nan=True) and np.array_equal(r1["outs"][:, own], r2["outs"][:, own])
          and np.array_equal(r1["pred"]["own"], r2["pred"]["own"]) and not np.array_equal(r1["A"]["spy"], r2["A"]["spy"], equal_nan=True)
          and not np.array_equal(r1["outs"][:, spy], r2["outs"][:, spy]))
    fired, base_acts = {c: 0 for c in C.CANDS}, False
    for k, day in enumerate(("20190306", "20190307", "20190308", "20190311", "20190312", "20190313")):
        dd, ss = E.parents(day, NSEC)
        rr = run1(rand_day(20 + k), rand_day(40 + k, bid0=3_000_000, tick_p=0.5), names, dd, ss)
        for c in C.CANDS:
            o = rr["outs"][1, names.index(c)]
            fired[c] += int((o[:, 9] > 0).sum() + o[:, 10].sum())
        base_acts |= bool(rr["outs"][:, :len(C.BASE), :, 9:11].any())
    check("random days: every candidate acts at least once, and the base runs never do", all(v > 0 for v in fired.values()) and not base_acts, actions=fired)


# ---------------------------------------------------------------- loader
def save(root, sym, date, b, mult=1):
    os.makedirs(os.path.join(root, date[:4]), exist_ok=True)
    rel = f"{date[:4]}/{sym}_{date}.npz"
    x = {c: b[c].astype(np.int32) for c in L22.COLS}
    for c in ("bsz", "asz"):
        x[c] = (b[c] * mult).astype(np.int32)
    np.savez_compressed(os.path.join(root, rel), **x)
    p = os.path.join(root, rel)
    return {"sym": sym, "date": date, "file": rel, "rows": str(NSEC), "bytes": str(os.path.getsize(p)), "sha256": L22.sha256(p)}


def task_of(root, row, ref_root=None, ref_row=None, close=NSEC):
    return {"kind": "file", "year": int(row["date"][:4]), "sym": row["sym"], "date": row["date"], "close_bar": close,
            "path": os.path.join(root, row["file"]), "sha256": row["sha256"],
            "ref_path": os.path.join(ref_root, ref_row["file"]) if ref_row else None, "ref_sha256": ref_row["sha256"] if ref_row else None}


def size_tests(tmp):
    root = os.path.join(tmp, "sizes")
    b, ref = rand_day(6), rand_day(7, bid0=3_000_000)
    rows = {d: (save(root, "AA", d, b, m), save(root, "SPY", d, ref, m)) for d, m in (("20251031", 1), ("20251103", 100), ("20251104", 100))}
    got = {d: L22.load(task_of(root, r[0], root, r[1])) for d, r in rows.items()}
    same = lambda x, y: all(np.array_equal(x[c], y[c]) for c in L22.COLS)   # noqa: E731
    check("sizes: a file of 2025-10-31 is read as stored; files of 2025-11-03 and later have bsz and asz divided by 100 (stock and SPY); "
          "prices and volumes unchanged", all(same(got[d][0], b) and same(got[d][1], ref) for d in rows),
          stored_max_bsz={d: int(np.load(os.path.join(root, rows[d][0]["file"]))["bsz"].max()) for d in rows}, read_max_bsz={d: int(got[d][0]["bsz"].max()) for d in rows})
    bad = {k: v.copy() for k, v in b.items()}
    bad["asz"][777] = 250
    rb = save(root, "BAD", "20251103", bad, 1)
    try:
        L22.load(task_of(root, rb))
        ok, msg = False, "loaded"
    except L22.Refused as e:
        ok, msg = "multiple of 100" in str(e), str(e)
    L22.load(task_of(root, save(root, "BAD", "20251031", bad, 1)))               # the same sizes before the change day are fine
    check("sizes: a value that is not a multiple of 100 on 2025-11-03 stops the load", ok, message=msg)
    r1 = dict(rows["20251031"][0])
    r1["sha256"] = "0" * 64
    try:
        L22.load(task_of(root, r1))
        ok = False
    except L22.Refused as e:
        ok = "sha256" in str(e)
    check("loader: a file whose sha256 differs from its manifest row is refused", ok)
    names = C.BASE + C.CANDS
    dd, ss = E.parents("20251103", NSEC)
    r3, r1_ = run1(*got["20251103"], names, dd, ss), run1(*got["20251031"], names, dd, ss)
    with np.load(os.path.join(root, rows["20251103"][0]["file"])) as z:
        raw = {c: z[c].astype(np.int64) for c in L22.COLS}
    rr = run1(raw, got["20251103"][1], names, dd, ss)
    sd3 = C.symday(task_of(root, rows["20251103"][0], root, rows["20251103"][1]), names)
    check("sizes: the same bars stored in shares (10-31) and in 100 x shares (11-03) give identical fills once read; unread (100 x) sizes "
          "would change the fills; a symbol-day of the plan goes through the same loader",
          np.array_equal(r3["outs"], r1_["outs"]) and not np.array_equal(rr["outs"], r3["outs"]) and rr["outs"][:, :, :, 4].sum() < r3["outs"][:, :, :, 4].sum()
          and np.array_equal(sd3["outs"], r3["outs"][:, :, E.quote_ok(got["20251103"][0])[ss]]),
          over_shares_read=int(r3["outs"][:, :, :, 4].sum()), over_shares_unread=int(rr["outs"][:, :, :, 4].sum()))


# ---------------------------------------------------------------- rules
def rule_tests():
    def rows(mean=3.10, p95=20.0, fired=0.3, **over):
        r = {c: {"mean": mean, "p95": p95, "fired": fired} for c in C.CANDS}
        for c, v in over.items():
            r[c] = {**r[c], **v}
        return r
    a1, _ = C.select_astar(rows(mean=3.10), 3.11)
    a2, _ = C.select_astar(rows(own_z1_G1={"mean": 3.08}), 3.10)
    a3, rec3 = C.select_astar(rows(own_z1_G1={"mean": 3.0799}), 3.10)
    check("A*: the best candidate must be lower than G0 by more than 0.02 cents (0.01 and exactly 0.02 give G0, 0.0201 gives the candidate)",
          (a1, a2, a3) == ("G0", "G0", "own_z1_G1") and rec3["improves_in_development"], picks=[a1, a2, a3])
    r = rows(mean=3.10, spy_z2_G1={"mean": 3.000, "p95": 20.0, "fired": 0.3}, own_z1_G2={"mean": 3.015, "p95": 18.0, "fired": 0.5},
             own_z2_G3={"mean": 3.019, "p95": 18.0, "fired": 0.2}, spy_z1_G1={"mean": 3.03, "p95": 10.0, "fired": 0.0})
    a4, rec4 = C.select_astar(r, 3.20)
    check("A*: among the candidates within 0.02 cents of the lowest mean the lower p95 wins, then the lower share of fired parents; a "
          "candidate outside 0.02 cents does not count", a4 == "own_z2_G3" and rec4["tied_within_0.02"] == ["spy_z2_G1", "own_z1_G2", "own_z2_G3"]
          and rec4["lowest_p95_among_tied"] == ["own_z1_G2", "own_z2_G3"], record=rec4)
    a5, _ = C.select_astar(rows(mean=3.00), 3.20)
    a6, rec6 = C.select_astar(rows(mean=3.10, own_z1_G1={"mean": 3.00, "p95": 30.0}), 3.015)
    check("A*: all equal -> the first of the grid order; G0 inside 0.02 cents of the best candidate is A* whatever the percentiles (G0 "
          "takes no part in the tie rule)", a5 == C.CANDS[0] == "spy_z1_G1" and a6 == "G0" and "tied_within_0.02" not in rec6, picks=[a5, a6])
    h1 = C.holm({"H1": 0.03})
    h2a, h2b, h2c = C.holm({"H1": 0.03, "H2": 0.04}), C.holm({"H1": 0.001, "H2": 0.04}), C.holm({"H1": 0.001, "H2": 0.06})
    check("Holm: one hypothesis at 0.05; two at 0.025 then 0.05, stopping at the first failure",
          h1["H1"]["rejected"] and h1["H1"]["level"] == 0.05 and not h2a["H1"]["rejected"] and not h2a["H2"]["rejected"]
          and h2b["H1"]["rejected"] and h2b["H2"]["rejected"] and h2b["H1"]["level"] == 0.025 and h2b["H2"]["level"] == 0.05
          and h2c["H1"]["rejected"] and not h2c["H2"]["rejected"], one=h1, two=[h2a, h2b, h2c])
    check("candidate grid: 12 names in the order of SPEC22 4 (basis, threshold, rule) and their engine runs",
          C.CANDS == ("spy_z1_G1", "spy_z1_G2", "spy_z1_G3", "spy_z2_G1", "spy_z2_G2", "spy_z2_G3", "own_z1_G1", "own_z1_G2", "own_z1_G3",
                      "own_z2_G1", "own_z2_G2", "own_z2_G3") and [C.engine_name(c) for c in C.CANDS[:3]] == ["R1_0.75", "R2_0.25", "R3_0.75_0.25"]
          and C.engine_name("G0") == "R0" and E.params("R3_0.75_0.25")[4:] == (0.75, 0.25) and C.parse("own_z2_G3") == ("own", 2.0, "G3"))


# ---------------------------------------------------------------- statistics
def stats_tests():
    idx = E.boot_index(250, seed=C.SEED)
    check("bootstrap: 10,000 draws of circular 21-day blocks with seed 22 (differs from seed 18, repeats itself)",
          idx.shape == (10_000, 250) and not np.array_equal(idx, E.boot_index(250)) and np.array_equal(idx, E.boot_index(250, seed=22)) and C.SEED == 22)
    # visited decision bars against a direct count on random days
    names = ("G0", "own_z1_G3")
    b = rand_day(8)
    dirs, starts = E.parents("20190305", NSEC)
    r = run1(b, None, names, dirs, starts)
    ok, tot = True, 0
    for run in names:
        o = col(r, names, run)
        for i in range(len(dirs)):
            ended_mkt = o[i, 2] > 0
            want = sum(1 for k in range(E.NG) if (5 * k <= o[i, 5] if ended_mkt else 5 * k < o[i, 5]))
            ok &= bool(C.visited(o)[i] == want)
            tot += want
    f_ = flat()
    s_ = U0 - 50
    v1 = int(C.visited(run1(f_, None, ("G0",), [1], [s_])["outs"][1, 0])[0])                       # nothing fills: deadline
    v2 = int(C.visited(run1(dip(3.0), None, ("own_z1_G1",), [1], [s_])["outs"][1, 0])[0])          # early fill at the 11th decision bar
    f2 = flat()
    f2["ask_min"][s_ + 12] = BID + 100                                                              # the posted bid + 1 cent is crossed at bar s + 12
    o3 = run1(f2, None, ("G0",), [1], [s_])["outs"][1, 0]
    ok &= bool(v1 == 60 and v2 == 11 and o3[0, 0] == E.QTY and o3[0, 5] == 12 and C.visited(o3)[0] == 3)
    fr = C.firing(col(r, names, "own_z1_G3"), "own_z1_G3")
    o = col(r, names, "own_z1_G3")
    check("decision bars visited (market end: those up to the bar; limit end: those before it) and the firing shares",
          ok and abs(fr["decisions"] - ((o[:, 9] > 0).sum() + o[:, 10].sum()) / C.visited(o).sum()) < 1e-12
          and abs(fr["parents"] - ((o[:, 9] > 0) | (o[:, 10] > 0)).mean()) < 1e-12 and C.firing(col(r, names, "G0"), "G0")["parents"] == 0.0,
          visited_total=tot, firing=fr, scenarios=[v1, v2, int(C.visited(o3)[0])])
    # prediction sums against a direct loop
    ref = rand_day(9, bid0=3_000_000, tick_p=0.5)
    close = 12_600
    dirs, starts = np.array([1, -1, 1, -1]), np.array([2004, 5004, 8004, close - 366])     # the last one is the last start bar of a half day
    r = run1(b, ref, C.BASE + C.CANDS, dirs, starts, close=close)
    m2 = b["bid"] + b["ask"]
    okm = (b["bid"] > 0) & (b["ask"] > 0) & (b["ask"] >= b["bid"])
    worst, left = 0.0, 0
    for basis in C.BASES:
        A = r["A"][basis]
        want = np.zeros((len(C.GROUPS), len(C.HORIZONS), 3))
        for i in range(len(dirs)):
            for k in range(E.NG):
                a = A[i, k]
                if np.isnan(a):
                    continue
                u = int(starts[i]) + 5 * k
                member = [True] + [c for z in C.ZTH for c in (a >= z, a <= -z)]
                for hi, h in enumerate(C.HORIZONS):
                    fut = u + h <= close - 1 and okm[u + h]
                    for gi, mem in enumerate(member):
                        if mem and fut:
                            want[gi, hi, 0] += (m2[u + h] / m2[u] - 1.0) * dirs[i] * 1e4
                            want[gi, hi, 1] += 1
                        elif mem:
                            want[gi, hi, 2] += 1
        worst = max(worst, float(np.abs(want - r["pred"][basis]).max()))
        left += int(want[0, 1, 2])
    check("prediction sums (adverse mid change 60 / 180 s after each decision grid point with a signal, per condition; points past the "
          "session's last bar left out and counted) equal a direct loop", worst < 1e-6 and left > 0, max_abs_diff=worst, left_out_180s=left)
    check("groups of the prediction table: all points, then early (a >= z_th) and hold (a <= -z_th) for thresholds 1 and 2; horizons 60 and 180 s",
          C.GROUPS == ("all", "z1_early", "z1_hold", "z2_early", "z2_hold") and C.HORIZONS == (60, 180))


# ---------------------------------------------------------------- the synthetic world
DEV_DAYS = {2019: ["20190301", "20190304", "20190305", "20190306", "20190307", "20190308", "20190311", "20190312"],
            2020: ["20201118", "20201119", "20201120", "20201123", "20201124", "20201125", "20201127", "20201130"]}
SEALED_DAYS = ["20210104", "20210105", "20210106", "20210107", "20210108", "20210111", "20251029", "20251030", "20251031", "20251103",
               "20251104", "20251105"]
HALF_DAYS = ("20201127",)


@contextlib.contextmanager
def patched(pairs):
    old = [(o, k, getattr(o, k)) for o, k, _ in pairs]
    for o, k, v in pairs:
        setattr(o, k, v)
    try:
        yield
    finally:
        for o, k, v in old:
            setattr(o, k, v)


def quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            fn(*a)
            return None
        except SystemExit as e:
            return str(e)
        except Exception as e:                                # a crash is a failed test, not a crashed test run
            return f"{type(e).__name__}: {e}"


def tsv(path, rows, cols=("sym", "date", "file", "rows", "bytes", "sha256")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows, columns=list(cols)).to_csv(path, sep="\t", index=False)


def build_world(tmp):
    """Folders: tmp/study22_market_signal (the stand-in repository), tmp/study21_more_pairs (reference manifests), tmp/data/... (bars)."""
    repo = os.path.join(tmp, "study22_market_signal")
    data = os.path.join(tmp, "data")
    os.makedirs(os.path.join(repo, "results"))
    seed = 100
    src, dev_ref = {}, []
    for y, days in DEV_DAYS.items():
        root = os.path.join(data, f"dev_{y}")
        rows = []
        for d in days:
            for s in ("DA", "DB"):
                seed += 1
                rows.append(save(root, s, d, rand_day(seed, bid0=BID + 30_000 * (s == "DB"))))
            if d != "20190306":                                              # one development day without an SPY file
                seed += 1
                dev_ref.append(save(os.path.join(data, "ref_dev"), "SPY", d, rand_day(seed, bid0=3_000_000, tick_p=0.5)))
        tsv(os.path.join(root, "MANIFEST.tsv"), rows)
        tsv(os.path.join(repo, f"universe_dev_{y}.tsv"), [(y, 1, "DA"), (y, 2, "DB")], cols=("year", "slot", "sym"))
        src[f"s{y}"] = (root, range(y, y + 1), os.path.join(repo, "universe_dev_{y}.tsv"), os.path.join(repo, "calendar_dev.csv"))
    pd.DataFrame({"date": [f"{d[:4]}-{d[4:6]}-{d[6:]}" for v in DEV_DAYS.values() for d in v],
                  "close_time": ["13:00" if d in HALF_DAYS else "16:00" for v in DEV_DAYS.values() for d in v]}).to_csv(os.path.join(repo, "calendar_dev.csv"), index=False)
    tsv(os.path.join(tmp, "study21_more_pairs", "DATA_MANIFEST21.tsv"), dev_ref)
    sroot, rroot = os.path.join(data, "sealed", "bars_1s_new_stocks"), os.path.join(data, "ref_sealed", "bars_1s")
    rows, ref = [], []
    for d in SEALED_DAYS:
        mult = 100 if d >= L22.SIZE_FROM else 1
        for s in ("SA", "SB", "XX"):                                         # XX is not in the universe lists
            seed += 1
            rows.append(save(sroot, s, d, rand_day(seed, bid0=BID + 20_000 * (s == "SB")), mult))
        seed += 1
        ref.append(save(rroot, "SPY", d, rand_day(seed, bid0=3_000_000, tick_p=0.5), mult))
    seed += 1
    rows.append(save(sroot, "SA", "20210112", rand_day(seed)))                 # a day that is not in the calendar
    tsv(os.path.join(repo, "SEALED_MANIFEST22.tsv"), rows)
    tsv(os.path.join(tmp, "study21_more_pairs", "SEALED_MANIFEST21.tsv"), ref)
    tsv(os.path.join(repo, "EXCLUDE22.tsv"), [("SB", "20210105", "2021/SB_20210105.npz", "test")], cols=("sym", "date", "file", "reason"))
    pd.DataFrame({"date": [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in SEALED_DAYS], "close_time": "16:00"}).to_csv(os.path.join(repo, "calendar22.csv"), index=False)
    for y in range(2021, 2027):
        tsv(os.path.join(repo, f"universe22_{y}.tsv"), [(y, 1, "SA"), (y, 2, "SB")], cols=("year", "slot", "sym"))
    for doc in RT.DOCS:
        open(os.path.join(repo, f"{doc}.md"), "w").write(f"{doc} stand-in\n")
        open(os.path.join(repo, f"{doc}.sha256"), "w").write(f"{RT.sha(os.path.join(repo, doc + '.md'))}  {doc}.md\n")
    res = os.path.join(repo, "results")
    pairs = [(L15, "SOURCES", src), (L15, "YEARS", range(2019, 2021)), (L22, "DEV_SPY_ROOT", os.path.join(data, "ref_dev")),
             (L22, "DEV_SPY_MANIFEST", os.path.join(tmp, "study21_more_pairs", "DATA_MANIFEST21.tsv")),
             (DV, "RES", res), (DV, "LOCAL", os.path.join(data, "dev_results")), (DV, "CALENDAR", os.path.join(repo, "calendar22.csv")),
             (DV, "TARGETS", None), (RT, "HERE", repo), (RT, "RES", res), (RT, "UNLOCK", os.path.join(repo, "UNLOCK22.txt")),
             (RT, "FREEZE", os.path.join(repo, "FREEZE22.md")), (RT, "DEV_SUMMARY", os.path.join(res, "dev22_summary.json")),
             (RT, "SEALED_ROOT", os.path.join(data, "sealed")), (RT, "SEALED_BARS", sroot), (RT, "SEALED_MANIFEST", os.path.join(repo, "SEALED_MANIFEST22.tsv")),
             (RT, "EXCLUDE", os.path.join(repo, "EXCLUDE22.tsv")), (RT, "CALENDAR", os.path.join(repo, "calendar22.csv")),
             (RT, "UNIVERSE", os.path.join(repo, "universe22_{y}.tsv")), (RT, "REF_SEALED_ROOT", rroot),
             (RT, "REF_SEALED_MANIFEST", os.path.join(tmp, "study21_more_pairs", "SEALED_MANIFEST21.tsv")),
             (RT, "NONCOMMON", {"sealed": {(2021, "SB")}, "rehearsal": {(2019, "DB")}})]
    return repo, res, pairs


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


def world_round(repo, res, label, forced):
    """Development run -> rehearsal -> guard -> sealed run in the synthetic world; forced: an A* to impose (None: as it falls)."""
    pre = f"world ({label}): "
    err = quiet(DV.main, 1)
    dev = json.load(open(RT.DEV_SUMMARY)) if os.path.exists(RT.DEV_SUMMARY) else {}
    daily = pd.read_csv(os.path.join(res, "dev22_daily.csv"), dtype={"date": str}) if dev else pd.DataFrame()
    n_days = sum(len(v) for v in DEV_DAYS.values())
    ok = (err is None and all(os.path.exists(os.path.join(res, f)) for f in DV.FILES) and dev["sim"]["symbol_days"] == 2 * n_days
          and dev["sim"]["symbol_days_without_ref"] == 2 and dev["sim"]["days_without_ref"] == ["20190306"]
          and set(dev["tables"]["L3D1"]) == set(DV.REPORT) and len(daily) == n_days * 2 * (len(DV.REPORT) + (1 if dev["h2_exists"] else 0))
          and (dev["control_flip"] is not None) == dev["h2_exists"]
          and (not dev["h2_exists"] or dev["control_flip"]["L3D1"]["control"] == C.flip_name(dev["astar"]))
          and dev["checks"]["daily_file_max_gap_c"] < 1e-6 and dev["h2_exists"] == (dev["astar"] != "G0") and dev["trials_cumulative"] == 825
          and (forced is None or dev["astar"] == forced) and (dev["delay_check"] is not None) == dev["h2_exists"]
          and (("H2" in dev["power"]) == dev["h2_exists"]) and dev["diagnostics"]["spy"]["no_signal_share"] > dev["diagnostics"]["own"]["no_signal_share"])
    check(pre + "the development run writes its four files once: 15 runs under two fill rules, the selection, power, the control with the "
          "reversed sign when A* is a candidate, and the daily file whose sums give the table means; a day without SPY has no SPY-basis signal", ok, error=err, astar=dev.get("astar"),
          h2=dev.get("h2_exists"), selection=dev.get("selection"))
    err2 = quiet(DV.main, 1)
    check(pre + "a second development run is refused (results exist)", err2 is not None and "once" in err2, message=err2)
    err = quiet(RT.main, "rehearsal", 1)
    reh = json.load(open(os.path.join(res, "rehearsal22_summary.json"))) if os.path.exists(os.path.join(res, "rehearsal22_summary.json")) else {}
    check(pre + "the rehearsal (the sealed code path on the development data) equals the development numbers: max difference 0, no sealed "
          "path read", err is None and reh.get("checks", {}).get("pass") is True and reh["checks"]["max_abs_diff"] == 0.0
          and reh["checks"]["numbers_compared"] > 100 and reh["astar"] == dev.get("astar") and reh["trials_added"] == 0,
          error=err, checks=reh.get("checks"))
    # ---- guard
    g = {"no FREEZE, no unlock": RT.guard()}
    write_freeze(repo)
    g["no unlock file"] = RT.guard()
    open(RT.UNLOCK, "w", encoding="utf-8").write("연구22 테스트를 연다\n")
    g["wrong phrase"] = RT.guard()
    open(RT.UNLOCK, "w", encoding="utf-8").write(RT.PHRASE + "\n")
    g["all in place"] = RT.guard()
    open(os.path.join(repo, "SPEC22_ANSWERS.md"), "a").write("x")
    g["document changed"] = RT.guard()
    open(os.path.join(repo, "SPEC22_ANSWERS.md"), "w").write("SPEC22_ANSWERS stand-in\n")
    core = os.path.join(repo, "signal22.py")
    keep = open(core).read()
    open(core, "w").write(keep + "#\n")
    g["frozen file changed"] = RT.guard()
    open(core, "w").write(keep)
    write_freeze(repo, drop="loader22.py")
    g["core file not listed"] = RT.guard()
    write_freeze(repo)
    g["all in place again"] = RT.guard()
    ok = (g["no unlock file"] == ["no unlock phrase in UNLOCK22.txt"] == g["wrong phrase"] and g["all in place"] == [] == g["all in place again"]
          and "FREEZE22.md missing or without the core files" in g["no FREEZE, no unlock"]
          and g["document changed"] == ["SPEC22_ANSWERS.md differs from SPEC22_ANSWERS.sha256"]
          and g["frozen file changed"] == ["frozen file differs: signal22.py"]
          and g["core file not listed"] == ["FREEZE22.md missing or without the core files"])
    check(pre + "guard of the sealed run: refuses without the unlock file, with another phrase, with a changed document, a changed frozen "
          "file or a core file missing from FREEZE22.md; passes only when everything is in place", ok, guard=g)
    # ---- the development summary is pinned: any change of its content stops the guard
    txt = open(RT.DEV_SUMMARY, encoding="utf-8").read()
    d0 = json.loads(txt)
    other = next(c for c in C.CANDS if c != d0["astar"])
    flip = not d0["h2_exists"]
    cases = {"another A*": {**d0, "astar": other, "h2_exists": True, "prediction_setting": {**d0["prediction_setting"], "basis": C.parse(other)[0], "z_th": C.parse(other)[1]}},
             "H2 flipped": {**d0, "h2_exists": flip},
             "prediction setting changed": {**d0, "prediction_setting": {**d0["prediction_setting"], "z_th": 3.0 - d0["prediction_setting"]["z_th"]}}}
    res_g, usable = {}, {}
    for k, v in cases.items():
        with open(RT.DEV_SUMMARY, "w") as fh:
            json.dump(v, fh, indent=1, default=str, ensure_ascii=False)
        res_g[k] = RT.guard()
        try:
            RT.selection()
            usable[k] = True
        except Exception:
            usable[k] = False
    open(RT.DEV_SUMMARY, "w", encoding="utf-8").write(txt + "\n")
    res_g["one newline appended"] = RT.guard()
    with open(RT.DEV_SUMMARY, "w") as fh:
        json.dump(cases["H2 flipped"], fh, indent=1, default=str, ensure_ascii=False)
    refused = quiet(RT.main, "sealed", 1)
    nothing = not any(os.path.exists(os.path.join(res, f)) for f in RT.MODES["sealed"]["files"])
    open(RT.DEV_SUMMARY, "w", encoding="utf-8").write(txt)
    back = RT.guard()
    check(pre + "a development summary whose content alone changes (another A*, H2 flipped, another prediction setting, one newline) stops "
          "the guard with 'frozen file differs', the sealed run writes nothing, and the original content passes again",
          all(v[:1] == ["frozen file differs: results/dev22_summary.json"] for v in res_g.values()) and usable["another A*"] and not usable["H2 flipped"]
          and any("no usable development selection" in x for x in res_g["H2 flipped"]) and refused is not None and refused.startswith("refused")
          and nothing and back == [], guard=res_g, readable_by_selection=usable, sealed_run=refused)
    # ---- the sealed run
    err = quiet(RT.main, "sealed", 1)
    t = json.load(open(os.path.join(res, "test22_summary.json"))) if os.path.exists(os.path.join(res, "test22_summary.json")) else {}
    td = pd.read_csv(os.path.join(res, "test22_daily.csv"), dtype={"date": str}) if t else pd.DataFrame(columns=["date"])
    h2 = dev.get("h2_exists")
    want_sd = 2 * len(SEALED_DAYS) - 1                                         # SA and SB on the calendar days, minus the excluded SB day
    hyp = ["H1", "H2"] if h2 else ["H1"]
    ok = (err is None and t.get("mode") == "sealed" and t["counts"]["symbol_days"] == want_sd and t["sim"]["excluded_rows"] == 1
          and t["hypotheses"] == hyp and sorted(t["verdict"][h]["level"] for h in hyp) == ([0.025, 0.05] if h2 else [0.05])
          and (t["verdict"]["H2"]["result"] == "없음") == (not h2) and t["trials_added"] == len(hyp)
          and t["verdict"]["H1"]["sentence"] in RT.SENTENCES["H1"].values()
          and t["date_split"]["L3D1"]["Estar_minus_S0"]["before"]["days"] == 9 and t["date_split"]["L3D1"]["Estar_minus_S0"]["from"]["days"] == 3
          and t["counts"]["parents_before_cut"] + t["counts"]["parents_from_cut"] == t["counts"]["parents"]
          and t["common_only"]["excluded_symbol_years"] == ["2021 SB"] and 0 < t["common_only"]["parents"] < t["counts"]["parents"]
          and set(td.date) == set(SEALED_DAYS) and set(td.run) == {"S0", "S2i", "G0", "C3"} | ({t["astar"], C.flip_name(t["astar"])} if h2 else set())
          and set(t["prediction"]["groups"]) == {"all", f"z{t['prediction']['z_th']:g}_early", f"z{t['prediction']['z_th']:g}_hold"}
          and set(t["prediction"]["tables"]) == set(C.BASES) and all(set(v) == set(C.GROUPS) for v in t["prediction"]["tables"].values())
          and set(t["diagnostics"]) == set(C.BASES) and "C3_minus_Estar" in t["differences"]["L3D1"] and "C3_minus_Estar" in t["differences"]["L0D0"]
          and (t["control_flip"] is not None) == bool(h2)
          and (not h2 or {"control_minus_Estar", "Astar_minus_control"} <= set(t["control_flip"]["L3D1"]))
          and ("G0_delay_minus_G0" in t["delay_check"]["L3D1"]) and ((f"{t['astar']}_delay_minus_G0_delay" in t["delay_check"]["L3D1"]) == bool(h2))
          and t["read"]["bars_root"].endswith("bars_1s_new_stocks"))
    check(pre + "the sealed run with the unlock phrase: universe symbols on calendar days minus the exclude list, H1" + (" and H2" if h2 else " alone")
          + " with Holm, C3 - E*, the control when A* is a candidate, the split at 2025-11-03, common stocks only, the whole prediction "
          "table, the delay check, three files",
          ok, error=err, verdict=t.get("verdict"), counts=t.get("counts"))
    again = quiet(RT.main, "sealed", 1)
    check(pre + "a second sealed run is refused (test results already exist)", again is not None and "test results already exist" in again, message=again)
    return dev


def world_tests(tmp):
    repo, res, pairs = build_world(tmp)
    with patched(pairs):
        dev = world_round(repo, res, "selection as it falls", None)
        # second round: the same world with a forced A* so that the H2 path runs end to end
        os.rename(res, os.path.join(repo, "results_round1"))
        os.makedirs(res)
        for f in ("FREEZE22.md", "UNLOCK22.txt"):
            os.rename(os.path.join(repo, f), os.path.join(repo, f + ".round1"))
        os.rename(DV.LOCAL, DV.LOCAL + "_round1")
        force = "G0" if dev.get("h2_exists") else "own_z1_G1"
        if force == "G0":
            fake = lambda rows, g0: ("G0", {"lowest_mean": "own_z1_G1", "lowest_mean_c": 0.0, "g0_mean_c": g0, "g0_minus_lowest_c": 0.0, "threshold_c": C.TIE,   # noqa: E731
                                            "improves_in_development": False, "astar": "G0"})
        else:
            fake = lambda rows, g0: (force, {"lowest_mean": force, "lowest_mean_c": rows[force]["mean"], "g0_mean_c": g0,   # noqa: E731
                                             "g0_minus_lowest_c": g0 - rows[force]["mean"], "threshold_c": C.TIE, "improves_in_development": True,
                                             "tied_within_0.02": [force], "lowest_p95_among_tied": [force], "fired": {force: rows[force]["fired"]}, "astar": force})
        with patched([(C, "select_astar", fake)]):
            dev2 = world_round(repo, res, "forced A* " + force, force)
    check("world: the two rounds cover both outcomes of the selection (A* = G0 without H2, and a candidate with H2)",
          {dev.get("h2_exists"), dev2.get("h2_exists")} == {True, False}, round1=dev.get("astar"), round2=dev2.get("astar"))


def repo_guard_test():
    why = RT.guard()
    check("the real sealed run is refused now (no UNLOCK22.txt; Claude never writes it)", "no unlock phrase in UNLOCK22.txt" in why
          and not os.path.exists(os.path.join(HERE, "UNLOCK22.txt")), reasons=why)
    check("constants: seed 22, 21-day blocks, 10,000 draws, tie 0.02 cents, 20 parents of 1,000 shares over 300 s, market fee 0.3 cents, "
          "the unlock phrase and the sealed window", C.SEED == 22 and E.BOOT_BLOCK == 21 and E.BOOT_B == 10_000 and C.TIE == 0.02 and E.N_PARENT == 20
          and E.QTY == 1000 and E.HOR == 300 and E.FEE_MKT == 0.3 and RT.PHRASE == "연구22 테스트를 한 번 연다" and RT.WINDOW == ("20210104", "20261110")
          and L22.SIZE_FROM == "20251103" and C.CUT == "20251103" and DV.N_NEW == 20 and E.VARIANTS[0] == ("L3D1", 3, 1))


def main():
    r18, r20 = earlier_tests()
    frozen_tests()
    flag_tests()
    engine_tests()
    no_signal_tests()
    future_tests()
    basis_tests()
    with tempfile.TemporaryDirectory(prefix="r22_tests_") as tmp:
        size_tests(tmp)
        rule_tests()
        stats_tests()
        world_tests(tmp)
    repo_guard_test()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    n_ok = sum(r["pass"] for r in RESULTS)
    path = os.path.join(HERE, "results", "engine22_tests.json")
    keep_old(path)
    with open(path, "w") as fh:
        json.dump({"passed": n_ok, "total": len(RESULTS), "research18_tests": r18, "research20_tests": r20, "numba": bool(E.HAVE_NUMBA), "tests": RESULTS},
                  fh, indent=1, default=str, ensure_ascii=False)
    for r in RESULTS:
        print(("PASS " if r["pass"] else "FAIL ") + r["test"][:150])
        if not r["pass"]:
            print("     ", json.dumps(r, default=str, ensure_ascii=False)[:1800])
    print(f"{n_ok}/{len(RESULTS)} passed (research 18's tests: {r18}, research 20's tests: {r20})")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
