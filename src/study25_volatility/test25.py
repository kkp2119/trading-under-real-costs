#!/usr/bin/env python3
"""Research 25 synthetic tests (step-2 prompt 1; they must pass before the validation run). Synthetic bars only; no bar of
the development universe, of SPY or of any sealed folder is read, and the sealed folders are not named outside temporary
stand-ins.
  earlier   research 23's synthetic tests (which rerun research 18's 78, research 20's 117 and research 22's), all but its
            two tests of research 23's own live folder, rerun here through research 23's test functions
  frozen    the files of research 23 used here equal the files of their tags; the frozen model is read only with its sha256
  signals   RV60, q, g and v against direct loops and hand values; one invalid mid among the 13 -> no signal; the window is
            the bars u-60..u; pre-order bars right after the order start; spread 0; vol_pre undefined or 0; no usual volume
  future    values after a decision bar change neither the signals nor what the engine gets at that bar
  engine    what the engine gets (A: 1 / 0 / 0.5, B and C: p or 0.5), the thresholds as ">=", early fill and hold through
            engine18, a day on which no condition holds (all 14 candidates equal R0 parent by parent), the control,
            kappa = 0 as "no condition" against research 23's own code path
  rules     A*: the 0.02 cent threshold against R0 and the tie rule; the candidate grid; the prediction table check
  stats     bootstrap seed 25, the dollar saving, the diagnostics against direct formulas, the regression function
  world     a small synthetic world through validation run -> rehearsal -> guard -> sealed run, once with the selection as
            it falls and once with the other outcome forced; a changed model file or validation summary stops the guard;
            A* = R0 refuses the rehearsal and the sealed run
Writes results/engine25_tests.json (an earlier file is moved to results/superseded/); exits 1 when a test fails.
  python test25.py"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import core25 as C  # noqa: E402
import dev25 as DV  # noqa: E402
import run_test25 as RT  # noqa: E402
import test23 as T23  # noqa: E402  (research 23's test functions; its main is never called)

C23, E, F, M, L22, S = C.C23, C.E, C.F, C.M, C.L22, C.S
L15, NSEC = L22.L15, L22.NSEC
BID, SPR, U0 = T23.BID, T23.SPR, T23.U0
flat, setq, rand_day, save, tsv, task_of, patched, quiet, hand, arr, trend_day, load_json = (
    T23.flat, T23.setq, T23.rand_day, T23.save, T23.tsv, T23.task_of, T23.patched, T23.quiet, T23.hand, T23.arr, T23.trend_day, T23.load_json)
RESULTS = []
TESTS23_OWN_FOLDER = 2                     # research 23's repo_guard_test: two tests of its own live folder, not rerun here
EXTRA_FROZEN = {"study23_combined_forecast/test23.py": ("code-freeze-23", "90791a1b5c668a772fdaddfdfab3293a1edfaf9d4c42ebc5c0fbca1dd4edeb6c"),
                "study23_combined_forecast/dev23.py": ("code-freeze-23", "f9c8912e1061125301519dda29cd99344e5a97d6dc07807bcef95009258359dd"),
                "study23_combined_forecast/run_test23.py": ("code-freeze-23", "fe29e06510b169c3c43b8a6555b155204ecf8418650048802f488aa9cdda8a30")}
QCOLS = ("bid", "bid_max", "bid_min", "ask", "ask_max", "ask_min")
S0_ = U0 - 50                               # the order start used by the hand-made days (11:09:15)
ADV = 390 * 10_000.0                        # the usual 60-second volume is 10,000 shares


def check(name, ok, **info):
    RESULTS.append({"test": name, "pass": bool(ok), **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}})


def jump(b, u, cents):
    """Bid and ask move by `cents` from bar u on."""
    for k in QCOLS:
        b[k][u:] += 100 * cents
    return b


def wiggle(b, s, cents=1):
    """The mid at the vol_pre points s-1-60k is `cents` higher for odd k: 30 one-minute returns of alternating sign, so that
    vol_pre exists; the bars from s-60 on are untouched."""
    for k in range(1, 31, 2):
        p = s - 1 - 60 * k
        for c in QCOLS:
            b[c][p - 59:p + 1] += 100 * cents
    return b


def trades(b, u0, u1, side="ask", shares=500):
    b[f"vol_{side}"][u0:u1] = shares
    b["vol"][u0:u1] = shares
    return b


def col(r, names, run, lr=3):
    return r["outs"][E.LRULES.index(lr), list(names).index(run)]


# ---------------------------------------------------------------- earlier studies
def earlier_tests(tmp):
    T23.RESULTS.clear()
    sub = os.path.join(tmp, "r23")
    os.makedirs(sub)
    r18, r20, r22 = T23.earlier_tests(sub)
    T23.frozen_tests()
    T23.trade_tests()
    T23.adv_tests(sub)
    T23.signal_tests()
    T23.future_tests()
    T23.model_tests(sub)
    T23.engine_tests(sub)
    T23.rule_tests()
    T23.stats_tests()
    T23.world_tests(os.path.join(sub, "world"))
    got = list(T23.RESULTS)
    stored = json.load(open(os.path.join(PROJ, "study23_combined_forecast", "results", "engine23_tests.json")))["tests"]
    want = [r["test"] for r in stored][:-TESTS23_OWN_FOLDER]
    n_ok = sum(r["pass"] for r in got)
    for r in got[:3]:
        check(r["test"], r["pass"], passed=r.get("passed"), total=r.get("total"), failed=r.get("failed"))
    check(f"research 23's synthetic tests pass here ({len(want)}: the names of its stored result without the two tests of its own live folder)",
          len(got) == len(want) == n_ok and [r["test"] for r in got] == want, passed=n_ok, total=len(got), failed=[r["test"] for r in got if not r["pass"]])
    return r18, r20, r22, f"{n_ok}/{len(got)}"


def frozen_tests():
    got = {}
    for p, (tag, h) in {**C.FROZEN, **EXTRA_FROZEN}.items():
        blob = subprocess.run(["git", "show", f"{tag}:{p}"], capture_output=True, cwd=PROJ).stdout
        got[p] = (hashlib.sha256(blob).hexdigest() == h, hashlib.sha256(open(os.path.join(PROJ, p), "rb").read()).hexdigest() == h)
    check("core23, feat23, model23, research 23's model file and its test, development and sealed-run files equal the files of their tag (git) "
          "and the working files", all(a and b for a, b in got.values()) and C.frozen_differences() == [], files={p: list(v) for p, v in got.items()})
    m = C.model23()
    check("research 23's frozen model is read only when it equals its sha256: nine inputs, the two fills, never refitted here",
          m["features"] == list(F.FEATS) and len(m["coef"]) == 9 and set(m["fills"]) == {"vol_pre", "intensity"}
          and C.sha(C.MODEL23) == C.FROZEN["study23_combined_forecast/model23.json"][1] == open(os.path.join(PROJ, "study23_combined_forecast", "model23.sha256")).read().split()[0]
          and not hasattr(DV, "fit_stage"))


# ---------------------------------------------------------------- signals
def direct_signals(b, dirs, starts, vp, adv):
    """RV60, RV in bp, q, g, v at every decision bar by a plain loop over the bars."""
    n = len(dirs)
    out = {k: np.full((n, E.NG), np.nan) for k in ("rv", "rv_bp", "q", "g", "v")}
    for i in range(n):
        for k in range(E.NG):
            u = int(starts[i]) + 5 * k
            bars = [u - 60 + 5 * j for j in range(13)]
            mids = [(int(b["bid"][x]) + int(b["ask"][x])) / 2.0 if x >= 0 and b["bid"][x] > 0 and b["ask"][x] > 0 and b["ask"][x] >= b["bid"][x] else None
                    for x in bars]
            if adv is not None and np.isfinite(adv) and adv > 0:
                out["v"][i, k] = min(max(b["vol"][max(u - 59, 0):u + 1].sum() / (adv / 390.0), 0.0), 10.0)
            if any(m is None for m in mids):
                continue
            rv = np.sqrt(sum((mids[j + 1] - mids[j]) ** 2 for j in range(12))) / 100.0
            rb = 1e4 * np.sqrt(sum(np.log(mids[j + 1] / mids[j]) ** 2 for j in range(12)))
            out["rv"][i, k], out["rv_bp"][i, k] = rv, rb
            spr = (int(b["ask"][u]) - int(b["bid"][u])) / 100.0
            if spr > 0:
                out["q"][i, k] = rv / spr
            if np.isfinite(vp[i]) and vp[i] > 0:
                out["g"][i, k] = rb / vp[i]
    return out


def signal_tests():
    s = S0_
    # ---- a hand example: three mid changes inside the window of the first decision bar
    b = flat()
    jump(b, s - 58, 1), jump(b, s - 33, -2), jump(b, s - 12, 2)
    r = S.recent(b, arr([s]))
    check("RV60 by hand: mid changes of +1, -2 and +2 cents inside the bars u-60..u give sqrt(1 + 4 + 4) = 3 cents; q = 3 / 8 with an "
          "8-cent spread", r["rv"][0, 0] == 3.0 and r["q"][0, 0] == 3.0 / 8.0 and r["spread_c"][0, 0] == 8.0 and bool(r["full"][0, 0]),
          rv=r["rv"][0, :14], q=r["q"][0, :3])
    m0, m1 = (BID + BID + SPR) / 2.0, (BID + BID + SPR) / 2.0 + 300.0
    b = jump(flat(), s - 12, 3)
    sg = S.signals(b, arr([1]), arr([s]), np.array([2.0]), ADV)
    want_g = 1e4 * abs(np.log(m1 / m0)) / 2.0
    check("g by hand: one 3-cent move in the window is 10,000 x |log return| (bp) over vol_pre (2 bp per minute)",
          abs(sg["g"][0, 0] - want_g) < 1e-12 and abs(sg["g"][0, 0] - 1.4993) < 1e-3 and sg["rv"][0, 0] == 3.0, g=float(sg["g"][0, 0]), want=want_g)
    # ---- every decision bar of the 20 parents of a random day against a direct loop
    b = rand_day(71)
    rng = np.random.default_rng(72)
    for u in rng.choice(np.arange(300, NSEC - 400), 400, replace=False):      # invalid, locked and crossed quotes here and there
        kind = rng.integers(0, 3)
        if kind == 0:
            b["bid"][u] = 0
        elif kind == 1:
            b["ask"][u] = b["bid"][u]
        else:
            b["ask"][u] = b["bid"][u] - 100
    dirs, starts = E.parents("20190305", NSEC)
    f = E.features(b, dirs, starts)
    vp = f["vp"].copy()
    vp[3], vp[7], vp[11] = np.nan, 0.0, -1.0
    sg = S.signals(b, dirs, starts, vp, 2.0e6)
    rc = S.recent(b, starts)
    d = direct_signals(b, dirs, starts, vp, 2.0e6)
    gap = {k: float(np.nanmax(np.abs(np.where(np.isnan(d[k]), 0.0, d[k]) - np.where(np.isnan(x), 0.0, x)))) for k, x in
           (("rv", sg["rv"]), ("rv_bp", rc["rv_bp"]), ("q", sg["q"]), ("g", sg["g"]), ("v", sg["v"]))}
    same_nan = all(np.array_equal(np.isnan(d[k]), np.isnan(x)) for k, x in (("rv", sg["rv"]), ("q", sg["q"]), ("g", sg["g"]), ("v", sg["v"])))
    check("RV60, q, g and v at every decision bar of 20 random parents (with invalid, locked and crossed quotes on the day) equal a direct "
          "loop over the bars, with the same points without a signal", same_nan and max(gap.values()) < 1e-9 and 0 < np.isnan(sg["rv"]).sum() < sg["rv"].size
          and 0 < (np.isnan(sg["q"]) & ~np.isnan(sg["rv"])).sum(), max_abs_diff=gap, no_signal={k: int(np.isnan(sg[k]).sum()) for k in ("rv", "q", "g", "v")})
    raw = F.raw(b, None, NSEC, dirs, starts, 2.0e6)
    check("v is research 23's input 8 (feat23.trade_inputs, called as it is): the same numbers as the intensity column of its inputs",
          np.array_equal(sg["v"], raw["X"][..., F.I_INT]) and np.array_equal(sg["v"], F.trade_inputs(b, dirs, starts, 2.0e6)[1]) and float(sg["v"].max()) <= 10.0)
    check("g: an order whose vol_pre is undefined, 0 or negative has no g at any of its 60 decision bars; the other orders have g wherever "
          "the window is full", bool(np.isnan(sg["g"][[3, 7, 11]]).all()) and not sg["vp_ok"][[3, 7, 11]].any()
          and bool(np.isnan(sg["g"][~sg["vp_ok"]]).all()) and np.array_equal(np.isnan(sg["g"][sg["vp_ok"]]), ~sg["full"][sg["vp_ok"]])
          and int(sg["vp_ok"].sum()) >= 10 and np.array_equal(sg["vp_ok"], np.isfinite(vp) & (vp > 0)), orders_with_vol_pre=int(sg["vp_ok"].sum()))
    # ---- one invalid mid among the 13
    base = jump(flat(), s - 12, 2)
    ok_all = S.recent(base, arr([s]))
    res = {}
    for off in list(range(-60, 1, 5)) + [-57, -61, 1]:
        for kind in ("no bid", "crossed"):
            x = {k: v.copy() for k, v in base.items()}
            if kind == "no bid":
                x["bid"][s + off] = 0
            else:
                x["ask"][s + off] = x["bid"][s + off] - 100
            res[(off, kind)] = bool(np.isnan(S.recent(x, arr([s]))["rv"][0, 0]))
    on_grid = [res[(o, k)] for o in range(-60, 1, 5) for k in ("no bid", "crossed")]
    off_grid = [res[(o, k)] for o in (-57, -61, 1) for k in ("no bid", "crossed")]
    check("one invalid mid (no bid, or a crossed quote) at any of the 13 bars u-60, u-55, ..., u -> no signal at u; an invalid quote at "
          "a bar that is not one of the 13 (u-57, u-61, u+1) changes nothing", ok_all["rv"][0, 0] == 2.0 and all(on_grid) and not any(off_grid),
          on_grid=on_grid, off_grid=off_grid)
    x = {k: v.copy() for k, v in base.items()}
    x["ask"][s - 30] = x["bid"][s - 30]                                       # locked inside the window: a valid mid
    y = {k: v.copy() for k, v in base.items()}
    y["ask"][s] = y["bid"][s]                                                 # locked at the decision bar: spread 0
    rx, ry = S.recent(x, arr([s])), S.signals(y, arr([1]), arr([s]), np.array([2.0]), ADV)
    check("a locked quote (ask = bid) is a valid mid inside the window; at the decision bar itself the spread is 0, so q has no signal there "
          "while RV60 and g exist", bool(rx["full"][0, 0]) and not np.isnan(rx["q"][0, 0]) and np.isnan(ry["q"][0, 0]) and not np.isnan(ry["rv"][0, 0])
          and not np.isnan(ry["g"][0, 0]) and bool(np.isnan(S.engine_p("B1_p60_k0.5", ry, np.full((1, E.NG), 0.9))[0, 0]) is np.False_
                                                   and S.engine_p("B1_p60_k0.5", ry, np.full((1, E.NG), 0.9))[0, 0] == 0.5))
    # ---- the window is u-60..u
    got = {}
    for off in (-61, -60, -58, -5, -1, 0, 1, 5):
        z = jump(flat(), s + off, 2)
        got[off] = float(S.recent(z, arr([s]))["rv"][0, 0])
    check("window at decision bar u: a mid change at u-60 or earlier is outside (the first of the 13 mids already holds it), changes at "
          "u-58 .. u are inside, a change after u is outside",
          got == {-61: 0.0, -60: 0.0, -58: 2.0, -5: 2.0, -1: 2.0, 0: 2.0, 1: 0.0, 5: 0.0}, rv=got)
    z = jump(flat(), s - 32, 2)                                               # between the grid bars s-35 and s-30, before the order start
    rz = S.recent(z, arr([s]))["rv"][0]
    check("right after the order start (k < 12) the window reaches into the bars before the start: a move at s-32 is in the windows of the "
          "decision bars k = 0..5 and out of them from k = 6 on", bool((rz[:6] == 2.0).all() and (rz[6:] == 0.0).all()), rv=rz[:9])
    e = S.recent(flat(), arr([40]))
    check("a window that would start before the first bar of the day gives no signal (not a real start bar: the first is 09:35:00)",
          bool(np.isnan(e["rv"][0, :4]).all() and e["rv"][0, 4] == 0.0) and E.FIRST_START == 299 and E.FIRST_START - S.WIN >= 0, rv=e["rv"][0, :6])


def future_tests():
    b = rand_day(81)
    dirs, starts = E.parents("20190306", NSEC)
    hm = hand({"trade_imbalance_30s": 3.0, "mid_change_5s": 0.4})
    adv = 2.0e6
    r0 = C.run_parents(b, None, NSEC, dirs, starts, adv, C.CANDS, model=hm)
    i, k = 4, 23
    u = int(starts[i]) + 5 * k
    rng = np.random.default_rng(82)
    x = {c: v.copy() for c, v in b.items()}
    n = NSEC - (u + 1)
    x["bid"][u + 1:] = b["bid"][u + 1:] + 100 * rng.integers(-30, 31, n)
    x["ask"][u + 1:] = x["bid"][u + 1:] + 100 * rng.integers(1, 9, n)
    for c in ("bsz", "asz", "vol_bid", "vol_ask"):
        x[c][u + 1:] = rng.integers(0, 50, n) * 100
    x["vol"][u + 1:] = x["vol_bid"][u + 1:] + x["vol_ask"][u + 1:]
    r1 = C.run_parents(x, None, NSEC, dirs, starts, adv, C.CANDS, model=hm)
    same_sig = all(np.array_equal(r0["sig"][c][i, :k + 1], r1["sig"][c][i, :k + 1], equal_nan=True) for c in ("rv", "q", "g", "v"))
    same_p = np.array_equal(r0["P23"][i, :k + 1], r1["P23"][i, :k + 1], equal_nan=True)
    same_pe = all(np.array_equal(r0["Pe"][c][i, :k + 1], r1["Pe"][c][i, :k + 1]) for c in C.CANDS)
    later = any(not np.array_equal(r0["sig"][c][i, k + 1:], r1["sig"][c][i, k + 1:], equal_nan=True) for c in ("rv", "q", "v"))
    check("no look-ahead: with every bar after a decision bar u replaced (quotes, sizes, volumes), RV60, q, g, v, p and what the engine gets "
          "for each of the 14 candidates are unchanged at u and at every earlier decision bar of the order; later decision bars do change",
          same_sig and same_p and same_pe and later and len(r0["Pe"]) == 14)
    g0 = S.signals(b, dirs, starts, r0["raw"]["vp"], adv)["g"]
    x2 = {c: v.copy() for c, v in b.items()}
    s_i = int(starts[i])
    x2["bid"][s_i:] += 500
    x2["ask"][s_i:] += 500
    g2 = S.signals(x2, dirs, starts, E.features(x2, dirs, starts)["vp"], adv)["g"]
    check("vol_pre is fixed at the order start: bars from the start bar on do not change it (engine18.vol_pre uses the bars up to start - 1)",
          E.features(x2, dirs, starts)["vp"][i] == r0["raw"]["vp"][i] and not np.array_equal(g0[i], g2[i], equal_nan=True))


# ---------------------------------------------------------------- what the engine gets and what it does with it
def flag_tests():
    g = np.array([[np.nextafter(2.0, 0.0), 2.0, 2.5, np.nextafter(3.0, 0.0), 3.0, 7.0, np.nan, 0.0]])
    sig = {"g": g, "q": g, "v": g}
    a = {c: S.engine_p(c, sig)[0].tolist() for c in ("A1_g2", "A1_g3", "A2_g2", "A2_g3")}
    check("family A: 1 (early fill) or 0 (hold) where g >= theta, 0.5 elsewhere; theta = 2 and 3 count as reached at exactly 2 and 3 and "
          "not just below; no g -> 0.5", a == {"A1_g2": [0.5, 1, 1, 1, 1, 1, 0.5, 0.5], "A1_g3": [0.5, 0.5, 0.5, 0.5, 1, 1, 0.5, 0.5],
                                             "A2_g2": [0.5, 0, 0, 0, 0, 0, 0.5, 0.5], "A2_g3": [0.5, 0.5, 0.5, 0.5, 0, 0, 0.5, 0.5]}, values=a)
    q = np.array([[np.nextafter(0.5, 0.0), 0.5, 0.75, np.nextafter(1.0, 0.0), 1.0, 4.0, np.nan, 0.0]])
    p = np.array([[0.9, 0.8, 0.7, 0.6, 0.3, np.nan, 0.9, 0.2]])
    sig = {"q": q, "g": q, "v": np.array([[np.nextafter(2.0, 0.0), 2.0, 2.0, 5.0, 10.0, 3.0, np.nan, 0.0]])}
    bq = {c: S.engine_p(c, sig, p)[0].tolist() for c in ("B1_p60_k0.5", "B1_p70_k1", "B2_k0.5", "B3_k1", "C1_p60", "C1_p70")}
    check("families B and C: p where the second condition holds (q >= kappa at exactly 0.5 and 1.0, v >= 2 at exactly 2), else 0.5; a point "
          "without the signal or without p gets 0.5, never nan",
          bq == {"B1_p60_k0.5": [0.5, 0.8, 0.7, 0.6, 0.3, 0.5, 0.5, 0.5], "B1_p70_k1": [0.5, 0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5],
                 "B2_k0.5": [0.5, 0.8, 0.7, 0.6, 0.3, 0.5, 0.5, 0.5], "B3_k1": [0.5, 0.5, 0.5, 0.5, 0.3, 0.5, 0.5, 0.5],
                 "C1_p60": [0.5, 0.8, 0.7, 0.6, 0.3, 0.5, 0.5, 0.5], "C1_p70": [0.5, 0.8, 0.7, 0.6, 0.3, 0.5, 0.5, 0.5]}, values=bq)
    fl = {c: S.engine_p(c, sig, p, flip=True)[0].tolist() for c in ("B1_p60_k0.5", "B3_k1", "C1_p60")}
    fa = S.engine_p("A1_g2", {"g": g}, flip=True)[0].tolist()
    off = S.engine_p("B1_p60_k0.5", sig, p, off=True)[0].tolist()
    check("control (QUESTIONS 21): only the volatility / volume condition is turned around (x < threshold where the signal exists), the p "
          "rule stays; a point without the signal stays at 0.5 in both; 'off' (the regression of ANSWERS A3) hands p over wherever p exists",
          fl == {"B1_p60_k0.5": [0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2], "B3_k1": [0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5, 0.2],
                 "C1_p60": [0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2]} and fa == [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
          and off == [0.9, 0.8, 0.7, 0.6, 0.3, 0.5, 0.9, 0.2], control=fl, control_a=fa, off=off)
    try:
        S.engine_p("B2_k1", sig)
        refused = False
    except ValueError:
        refused = True
    # value sets on a random day
    b = rand_day(91)
    dirs, starts = E.parents("20190307", NSEC)
    hm = hand({"trade_imbalance_30s": 3.0, "mid_change_5s": 0.4})
    b["bid"][int(starts[2]) + 25] = 0                                           # a decision bar without a valid quote
    names = C.CANDS + tuple(C.CTRL + c for c in C.CANDS) + C.K0
    r = C.run_parents(b, None, NSEC, dirs, starts, 2.0e6, names, model=hm)
    p23, Pe, sg = r["P23"], r["Pe"], r["sig"]
    a_ok = all(set(np.unique(Pe[pre + c]).tolist()) <= {0.0, 0.5, 1.0} for c in C.CANDS if C.SPECS[c]["family"] == "A" for pre in ("", C.CTRL))
    bc_ok = all(bool(((Pe[pre + c] == 0.5) | (Pe[pre + c] == p23)).all()) and not np.isnan(Pe[pre + c]).any()
                for c in C.CANDS if C.SPECS[c]["family"] != "A" for pre in ("", C.CTRL))
    comp = True
    for c in C.CANDS:
        x = sg[C.SPECS[c]["signal"]]
        on, ctl = S.condition(x, C.SPECS[c]["threshold"]), S.condition(x, C.SPECS[c]["threshold"], flip=True)
        comp = comp and not (on & ctl).any() and np.array_equal(on | ctl, ~np.isnan(x))
    k0_ok = all(np.array_equal(Pe[k], np.where(np.isfinite(p23), p23, 0.5)) for k in C.K0)
    check("on a random day: family A hands over only 1, 0 or 0.5; families B and C only p or 0.5 (also at a decision bar without a valid "
          "quote, where p does not exist); a candidate and its control never hold at the same point and together cover exactly the points "
          "where the signal exists; the kappa = 0 runs hand over p wherever it exists; a B or C candidate without a model is an error",
          a_ok and bc_ok and comp and k0_ok and refused and np.isnan(p23[2, 5]) and all(Pe[c][2, 5] == 0.5 for c in C.CANDS)
          and sorted(np.unique(Pe["A1_g2"]).tolist()) == [0.5, 1.0] and sorted(np.unique(Pe["A2_g2"]).tolist()) == [0.0, 0.5])
    check("run names: a candidate, its control and a regression run use the engine run of the candidate (R1_0.75 / R2_0.25 for family A, "
          "research 23's R1_0.60, R1_0.70, R2_0.40, R3_0.60_0.40 for B and C); family A needs no model",
          [C.engine_name(x) for x in ("A1_g3", "A2_g2", "B1_p70_k1", "B2_k0.5", "B3_k1", "C1_p70", "ctrl:B3_k0.5", "ctrl:A2_g3", "k0:B1_p60", "k0:B2", "k0:B3", "S2i")]
          == ["R1_0.75", "R2_0.25", "R1_0.70", "R2_0.40", "R3_0.60_0.40", "R1_0.70", "R3_0.60_0.40", "R2_0.25", "R1_0.60", "R2_0.40", "R3_0.60_0.40", "S2i"]
          and not C.needs_model("A1_g2") and C.needs_model("ctrl:C1_p60") and not C.needs_model("R0") and C.cand_of("k0:B3") == "B3_k0.5"
          and E.params("R1_0.75")[4:] == (0.75, -1.0) and E.params("R2_0.25")[4:] == (-1.0, 0.25))


def engine_tests(tmp):
    s = S0_
    hm = hand({"trade_imbalance_30s": 5.0})                                   # p = sigmoid(5 x signed trade imbalance)
    two, d2 = arr([s, s]), arr([1, -1])
    # ---- family B, early fill: wide spread (E* rests a limit order and nothing fills), buyer-initiated trades before the start
    names = ("R0", "B1_p60_k0.5", "B1_p60_k1", "B1_p70_k0.5", "B2_k0.5", "B3_k0.5", "ctrl:B1_p60_k0.5", "ctrl:B1_p60_k1")
    out = {}
    for cents in (3, 4, 8):
        b = jump(trades(flat(), s - 20, s), s - 12, cents)
        out[cents] = C.run_parents(b, None, NSEC, d2, two, None, names, model=hm)
    r = out[4]
    p, q = r["P23"], r["sig"]["q"]
    check("B setup: p is high for the buy order and low for the sell order for the six decision bars with the trades in the 30-second "
          "window; a 4-cent move before the start with an 8-cent spread gives q = 0.5 exactly at the decision bars k = 0..9",
          bool((p[0, :6] > 0.99).all() and (p[1, :6] < 0.01).all() and (p[:, 6:] == 0.5).all() and (q[:, :10] == 0.5).all() and (q[:, 10:] == 0.0).all())
          and bool((out[3]["sig"]["q"][:, :10] == 0.375).all()) and bool((out[8]["sig"]["q"][:, :10] == 1.0).all()), q=q[0, :12])
    r0, b05, b1, b70, b2 = (col(r, names, k) for k in ("R0", "B1_p60_k0.5", "B1_p60_k1", "B1_p70_k0.5", "B2_k0.5"))
    ask0 = BID + SPR + 400
    check("B1 (p >= p_hi and q >= kappa): with q = 0.5 the buy order goes to market at the first decision bar at that bar's ask for kappa "
          "0.5 (p_hi 0.60 and 0.70) and stays with E* for kappa 1.0; the sell order (low p) and B2 (wide spread: nothing to hold) equal E*",
          b05[0, 9] == E.QTY and b05[0, 5] == 0 and b05[0, 3] == E.QTY * ask0 and np.array_equal(b70[0], b05[0]) and np.array_equal(b1, r0)
          and np.array_equal(b05[1], r0[1]) and np.array_equal(b2, r0) and r0[0, 5] == E.HOR and not r0[:, 9:11].any()
          and np.array_equal(col(r, names, "B3_k0.5"), b05), b1=b05[:, [2, 5, 9]], r0=r0[:, [2, 5]])
    check("B1 thresholds: q = 0.375 fires neither kappa, q = 1.0 fires both",
          np.array_equal(col(out[3], names, "B1_p60_k0.5"), col(out[3], names, "R0")) and np.array_equal(col(out[3], names, "B1_p60_k1"), col(out[3], names, "R0"))
          and col(out[8], names, "B1_p60_k1")[0, 9] == E.QTY and col(out[8], names, "B1_p60_k0.5")[0, 9] == E.QTY)
    c05, c1 = col(r, names, "ctrl:B1_p60_k0.5"), col(r, names, "ctrl:B1_p60_k1")
    check("control of B1: with q = 0.5 the control of kappa 0.5 (q < 0.5) stays with E* and the control of kappa 1.0 (q < 1.0) fills the buy "
          "order early: the p rule is kept, only the volatility condition is turned around",
          np.array_equal(c05, r0) and c1[0, 9] == E.QTY and c1[0, 5] == 0 and np.array_equal(c1[1], r0[1])
          and C.firing(c1, "ctrl:B1_p60_k1")["parents"] == 0.5 and C.firing(c05, "ctrl:B1_p60_k0.5")["parents"] == 0.0)
    # ---- family B, hold: 2-cent spread (E* takes the spread at once)
    namesh = ("R0", "B2_k0.5", "B2_k1", "B3_k0.5", "B3_k1", "B1_p60_k0.5", "ctrl:B2_k1")
    h = {}
    for cents in (1, 2):
        b = jump(trades(flat(spread=200), s - 20, s), s - 12, cents)
        h[cents] = C.run_parents(b, None, NSEC, d2, two, None, namesh, model=hm)
    r = h[1]
    r0, h05, h1, h3, hc = (col(r, namesh, k) for k in ("R0", "B2_k0.5", "B2_k1", "B3_k0.5", "ctrl:B2_k1"))
    check("B2 (p <= 0.40 and q >= kappa) with a 2-cent spread: E* takes the spread at once; with q = 0.5 the sell order (low p) is held for "
          "the six decision bars with the trades in the window and goes to market at the seventh for kappa 0.5, not for kappa 1.0; the buy "
          "order is not held; B3 holds the same way; the control of kappa 1.0 holds",
          bool(r0[:, 5].max() == 0 and (r0[:, 2] == E.QTY).all() and (r["sig"]["q"][:, :10] == 0.5).all() and h05[1, 10] == 6 and h05[1, 5] == 30
               and h05[1, 2] == E.QTY and np.array_equal(h05[0], r0[0]) and np.array_equal(h1, r0) and np.array_equal(h3[1], h05[1])
               and hc[1, 10] == 6 and np.array_equal(col(h[2], namesh, "B2_k1")[1, [5, 10]], arr([30, 6]))
               and np.array_equal(col(r, namesh, "B1_p60_k0.5")[:, :9], r0[:, :9])), hold=h05[:, [5, 10]], r0=r0[:, [2, 5]])
    l0 = r["outs"][E.LRULES.index(0), namesh.index("B2_k0.5")]
    check("the same hold under the L0 fill rule (both fill rules are run)", bool(l0[1, 10] == 6 and l0[1, 5] == 30 and r["outs"].shape == (2, len(namesh), 2, E.NCOL)))
    # ---- family C: trade intensity
    namesc = ("R0", "C1_p60", "C1_p70", "ctrl:C1_p60")
    c = {}
    for label, adv in (("v = 2", 390 * 5_000.0), ("v below 2", 390 * 5_001.0), ("no usual volume", np.nan)):
        c[label] = C.run_parents(trades(flat(), s - 20, s), None, NSEC, d2, two, adv, namesc, model=hm)
    e2, eb, en = c["v = 2"], c["v below 2"], c["no usual volume"]
    check("C1 (p >= p_hi and v >= 2): 10,000 shares in the last 60 seconds over a usual 5,000 give v = 2 exactly and the buy order is "
          "filled early at the first decision bar; with v just below 2 it stays with E* and the control fills instead; on a day without a "
          "usual volume there is no v and both the candidate and its control equal E*",
          e2["sig"]["v"][0, 0] == 2.0 and col(e2, namesc, "C1_p60")[0, 9] == E.QTY and col(e2, namesc, "C1_p70")[0, 9] == E.QTY
          and np.array_equal(col(e2, namesc, "ctrl:C1_p60"), col(e2, namesc, "R0")) and eb["sig"]["v"][0, 0] < 2.0
          and np.array_equal(col(eb, namesc, "C1_p60"), col(eb, namesc, "R0")) and col(eb, namesc, "ctrl:C1_p60")[0, 9] == E.QTY
          and bool(np.isnan(en["sig"]["v"]).all()) and all(np.array_equal(col(en, namesc, k), col(en, namesc, "R0")) for k in namesc[1:])
          and bool((en["P23"][0, :6] > 0.99).all()), v=[float(e2["sig"]["v"][0, 0]), float(eb["sig"]["v"][0, 0])])
    # ---- family A: vol_pre from one-minute moves before the start, a move after the start
    namesa = ("R0", "A1_g2", "A1_g3", "A2_g2", "A2_g3", "ctrl:A1_g2", "ctrl:A1_g3")
    a = {}
    for cents in (2, 3, 4):
        a[cents] = C.run_parents(jump(wiggle(flat(), s), s + 3, cents), None, NSEC, d2, two, None, namesa)
    g = {k: float(v["sig"]["g"][0, 1]) for k, v in a.items()}
    vp = float(a[3]["raw"]["vp"][0])
    r0 = col(a[3], namesa, "R0")
    a2, a3 = col(a[3], namesa, "A1_g2"), col(a[3], namesa, "A1_g3")
    check("A1 (g >= theta): with a vol_pre of about 1 bp per minute, a 3-cent move after the start gives 2 <= g < 3: A1 with theta 2 "
          "sends both orders to market at the next decision bar (g has no direction) and A1 with theta 3 stays with E*; a 2-cent move fires "
          "neither and a 4-cent move both; before the move g is 0",
          abs(vp - 1.0166) < 1e-3 and g[2] < 2.0 <= g[3] < 3.0 <= g[4] and bool((a2[:, 9] == E.QTY).all() and (a2[:, 5] == 5).all())
          and np.array_equal(a3, r0) and np.array_equal(col(a[2], namesa, "A1_g2"), col(a[2], namesa, "R0"))
          and bool((col(a[4], namesa, "A1_g3")[:, 9] == E.QTY).all()) and a[3]["sig"]["g"][0, 0] == 0.0 and r0[0, 5] == E.HOR
          and np.array_equal(col(a[3], namesa, "A2_g2"), r0), g=g, vol_pre=vp)
    cg2, cg3 = col(a[3], namesa, "ctrl:A1_g2"), col(a[3], namesa, "ctrl:A1_g3")
    check("control of A1: g < theta fires at the first decision bar (g = 0 there), for both thetas", bool((cg2[:, 9] == E.QTY).all() and (cg2[:, 5] == 0).all()
                                                                                                   and (cg3[:, 5] == 0).all()))
    ah = C.run_parents(jump(wiggle(flat(spread=200), s), s - 12, 6), None, NSEC, d2, two, None, namesa)
    r0, hold2, hold3 = col(ah, namesa, "R0"), col(ah, namesa, "A2_g2"), col(ah, namesa, "A2_g3")
    gh = ah["sig"]["g"][0]
    check("A2 (g >= theta) with a 2-cent spread: E* takes the spread at once; a 6-cent move before the start keeps g >= 3 at the decision "
          "bars k = 0..9, both orders are held there and go to market at k = 10; A1 on top of E*'s own market order is not counted",
          bool(r0[:, 5].max() == 0 and (gh[:10] >= 3.0).all() and (gh[10:] == 0.0).all() and (hold2[:, 10] == 10).all() and (hold2[:, 5] == 50).all()
               and np.array_equal(hold3, hold2) and np.array_equal(col(ah, namesa, "A1_g2")[:, :9], r0[:, :9]) and (col(ah, namesa, "A1_g2")[:, 9] == 0).all()),
          g=gh[:12], hold=hold2[:, [5, 10]])
    und = {}
    for label, day, st in (("flat before the start (vol_pre 0)", jump(flat(), s + 3, 6), two),
                           ("start at 09:36 (fewer than 10 one-minute returns)", jump(wiggle(flat(), s), 364 + 3, 6), arr([364, 364]))):
        x = C.run_parents(day, None, NSEC, d2, st, None, namesa)
        und[label] = (bool(np.isnan(x["sig"]["g"]).all()) and not x["sig"]["vp_ok"].any() and float(np.nanmax(x["sig"]["rv"])) == 6.0
                      and all(np.array_equal(col(x, namesa, k), col(x, namesa, "R0")) for k in namesa[1:]))
    check("vol_pre undefined or 0: g has no signal at any decision bar of the order and family A (and its control) equals E* for the whole "
          "order, however large the move", all(und.values()), cases=und)
    # ---- a day on which no condition holds: all 14 candidates equal R0 parent by parent, while p alone would have acted
    b = wiggle(flat(spread=200), s)
    b["bsz"][:], b["asz"][:] = 9000, 1000
    hi = hand({"imbalance": 5.0})                                             # p from the displayed sizes: high for a buy, low for a sell
    alln = ("R0",) + C.CANDS + C.K0
    n = C.run_parents(b, None, NSEC, d2, two, ADV, alln, model=hi)
    base = col(n, alln, "R0")
    check("a day without a move in any window and without a trade (q = 0, g = 0, v = 0): the 14 candidates equal R0 parent by parent under "
          "both fill rules although p is far from 0.5 (research 23's rule on p alone, the kappa = 0 run, does act)",
          all(np.array_equal(n["outs"][:, alln.index(c)], n["outs"][:, 0]) for c in C.CANDS) and bool((n["P23"][0] > 0.9).all() and (n["P23"][1] < 0.1).all())
          and bool((n["sig"]["q"] == 0).all() and (n["sig"]["g"] == 0).all() and (n["sig"]["v"] == 0).all()) and not np.array_equal(col(n, alln, "k0:B2"), base)
          and col(n, alln, "k0:B2")[1, 10] > 0, hold_k0=col(n, alln, "k0:B2")[:, [5, 10]])
    # ---- kappa = 0 as "no condition" against research 23's own code path, on a random day with invalid quotes
    b = trend_day(95)
    rng = np.random.default_rng(96)
    for u in rng.choice(np.arange(300, NSEC - 400), 300, replace=False):
        b["bid"][u] = 0
    dirs, starts = E.parents("20190308", NSEC)
    okq = E.quote_ok(b)[starts]
    dirs, starts = dirs[okq], starts[okq]
    hm2 = hand({"trade_imbalance_30s": 3.0, "mid_change_5s": 0.4})
    old = C23.run_parents(b, None, NSEC, dirs, starts, 2.0e6, ("R0", "R1_0.60", "R2_0.40", "R3_0.60_0.40"), ("R1_0.60",), model=hm2)
    new = C.run_parents(b, None, NSEC, dirs, starts, 2.0e6, ("R0",) + C.K0 + ("B1_p60_k1", "B2_k1", "B3_k1"), ("k0:B1_p60",), model=hm2)
    miss = int((new["raw"]["valid"] & np.isnan(new["sig"]["q"])).sum())
    check("ANSWERS A3: with kappa = 0 read as 'no volatility condition' B1 (p_hi 0.60), B2 and B3 equal research 23's R1_0.60, R2_0.40 and "
          "R3_0.60_0.40 parent by parent under both fill rules and with the delay (research 23's own code path on the same bars), the "
          "probability is the same at every decision point (there are points with p and without q on the day), and a real candidate "
          "(kappa 1.0) does not equal research 23's run",
          np.array_equal(new["outs"][:, :4], old["outs"][:, :4]) and np.array_equal(new["outs"][:, 7], old["outs"][:, 4])
          and np.array_equal(new["P23"], old["P23"], equal_nan=True) and miss > 0 and old["outs"][:, 1:4, :, 9:11].any()
          and any(not np.array_equal(new["outs"][:, 4 + j], old["outs"][:, 1 + j]) for j in range(3)),
          parents=int(len(dirs)), quote_valid_without_q=miss, kappa1_differs=[not np.array_equal(new["outs"][:, 4 + j], old["outs"][:, 1 + j]) for j in range(3)])
    # ---- through the plan's loader: an empty file, the size correction, the counts
    root = os.path.join(tmp, "engine")
    empty = {c: np.zeros(NSEC, np.int64) for c in L22.COLS}
    te = {**task_of(root, save(root, "EE", "20260514", empty)), "adv": np.nan}
    rd = trend_day(97)
    rows = {d: save(root, "AA", d, rd, mult) for d, mult in (("20251031", 1), ("20251103", 100))}
    ta = [{**task_of(root, rows[d]), "adv": 1.0e6} for d in rows]
    runs = ("R0", "A1_g2", "B3_k0.5", "C1_p60", "ctrl:B3_k0.5")
    par, OUT, G, info = C.simulate(ta + [te], runs, ("R0", "B3_k0.5"), 1, model=hm2, grid=True)
    dd, ss = E.parents("20251103", NSEC)
    okq = E.quote_ok(rd)[ss]
    direct = C.run_parents(rd, None, NSEC, dd[okq], ss[okq], 1.0e6, runs, ("R0", "B3_k0.5"), model=hm2)
    sel = (par.date == "20251103").to_numpy()
    gc = info["grid_counts"]
    check("a file without any quote or trade gives no parent order and no error; a symbol-day of 2025-11-03 stored in 100 x shares is read "
          "through loader22 (sizes / 100) and equals the run on the bars in shares; the grid arrays (g, q, v, the 60-second move, p) and the "
          "counts of points without a signal come back per decision point",
          info["symbol_days_without_parents"] == 1 and int(sel.sum()) == int(okq.sum()) and np.array_equal(OUT[:, :, sel], direct["outs"])
          and all(np.array_equal(G[k][sel], direct["sig"][k], equal_nan=True) for k in ("g", "q", "v")) and np.array_equal(G["P23"][sel], direct["P23"], equal_nan=True)
          and OUT.shape[1] == 7 and G["move"].shape == G["q"].shape == G["y"].shape and gc["points"] == G["q"].size
          and gc["q_defined"] == int((~np.isnan(G["q"])).sum()) and gc["g_defined"] == int((~np.isnan(G["g"])).sum())
          and gc["quote_valid_q_undefined"] == int((G["valid"] & np.isnan(G["q"])).sum())
          and gc["quote_valid_q_undefined"] == gc["quote_valid_window_not_full"] + gc["quote_valid_spread_zero"]
          and gc["parents_vol_pre_undefined_or_zero"] == int((~(np.isfinite(par.vol_pre) & (par.vol_pre > 0))).sum()), counts=gc)
    mv = S.abs_move60(rd, ss[okq])
    u = int(ss[okq][0]) + 5 * 7
    check("diagnostic move: |mid(u + 60) - mid(u)| in cents (uses the future, never handed to a rule)",
          abs(mv[0, 7] - abs((rd["bid"][u + 60] + rd["ask"][u + 60]) - (rd["bid"][u] + rd["ask"][u])) / 200.0) < 1e-12 and np.array_equal(G["move"][sel], mv, equal_nan=True))


# ---------------------------------------------------------------- rules
def rule_tests():
    def rows(mean=3.10, p95=20.0, fired=0.3, over=None):
        r = {c: {"mean": mean, "p95": p95, "fired": fired} for c in C.CANDS}
        for c, v in (over or {}).items():
            r[c] = {**r[c], **v}
        return r
    a1, _ = C.select_astar(rows(mean=3.10), 3.11)
    a2, _ = C.select_astar(rows(over={"B2_k0.5": {"mean": 3.08}}), 3.10)
    a3, rec3 = C.select_astar(rows(over={"B2_k0.5": {"mean": 3.0799}}), 3.10)
    check("A*: the best candidate must be lower than R0 by more than 0.02 cents (0.01 and exactly 0.02 give R0, 0.0201 gives the candidate)",
          (a1, a2, a3) == ("R0", "R0", "B2_k0.5") and rec3["improves_in_development"], picks=[a1, a2, a3])
    r = rows(mean=3.10, over={"A2_g2": {"mean": 3.000, "p95": 20.0, "fired": 0.3}, "B1_p70_k1": {"mean": 3.015, "p95": 18.0, "fired": 0.5},
                              "C1_p60": {"mean": 3.019, "p95": 18.0, "fired": 0.2}, "A1_g2": {"mean": 3.03, "p95": 10.0, "fired": 0.0}})
    a4, rec4 = C.select_astar(r, 3.20)
    check("A*: among the candidates within 0.02 cents of the lowest mean the lower p95 wins, then the lower share of fired parents; a "
          "candidate outside 0.02 cents does not count", a4 == "C1_p60" and rec4["tied_within_0.02"] == ["A2_g2", "B1_p70_k1", "C1_p60"]
          and rec4["lowest_p95_among_tied"] == ["B1_p70_k1", "C1_p60"], record=rec4)
    a5, _ = C.select_astar(rows(mean=3.00), 3.20)
    a6, rec6 = C.select_astar(rows(mean=3.10, over={"B3_k1": {"mean": 3.00, "p95": 30.0}}), 3.015)
    check("A*: all equal -> the first of the grid order; R0 inside 0.02 cents of the best candidate is A* whatever the percentiles (R0 takes "
          "no part in the tie rule)", a5 == C.CANDS[0] == "A1_g2" and a6 == "R0" and "tied_within_0.02" not in rec6, picks=[a5, a6])
    fam = [C.SPECS[c]["family"] for c in C.CANDS]
    check("candidate grid (SPEC25 4): 14 candidates in the table order, A 4 (theta 2, 3 for early fill and for hold), B 8 (B1 p_hi 0.60, 0.70 "
          "x kappa 0.5, 1.0; B2 and B3 kappa 0.5, 1.0), C 2 (p_hi 0.60, 0.70 with v >= 2); 8 of them fill early; R0 is not a candidate",
          C.CANDS == ("A1_g2", "A1_g3", "A2_g2", "A2_g3", "B1_p60_k0.5", "B1_p60_k1", "B1_p70_k0.5", "B1_p70_k1", "B2_k0.5", "B2_k1", "B3_k0.5", "B3_k1",
                      "C1_p60", "C1_p70") and fam.count("A") == 4 and fam.count("B") == 8 and fam.count("C") == 2 and "R0" not in C.CANDS
          and S.EARLY_FILL == ("A1_g2", "A1_g3", "B1_p60_k0.5", "B1_p60_k1", "B1_p70_k0.5", "B1_p70_k1", "C1_p60", "C1_p70")
          and (S.THETAS, S.KAPPAS, S.P_HIS, S.P_LO, S.V_MIN) == ((2.0, 3.0), (0.5, 1.0), (0.60, 0.70), 0.40, 2.0)
          and DV.REPORT == ("S0", "S2i", "C3", "R0") + C.CANDS and (S.WIN, S.GSTEP, S.NPTS) == (60, 5, 13))
    t = {"L3D1": {c: {"minus_R0_mean": 0.01, "is_mean": 3.2} for c in C.CANDS}}
    t["L3D1"]["A2_g2"]["is_mean"], t["L3D1"]["B2_k1"]["is_mean"] = 3.15, 3.12
    pc1 = DV.predictions_check(t, "R0")
    t["L3D1"]["C1_p70"]["minus_R0_mean"] = -0.001
    t["L3D1"]["B2_k1"]["is_mean"] = 3.16
    pc2 = DV.predictions_check(t, "B2_k1")
    check("prediction table check (SPEC25 8): A* not R0; H left to the sealed run; all 8 early-fill candidates above R0; the best of family B "
          "below the best of family A", [x["outcome"] for x in pc1] == ["일어나지 않음", "해당 없음 (A* = R0)", "맞음", "맞음"]
          and [x["outcome"] for x in pc2] == ["일어남", "봉인 실행에서 판정", "틀림", "틀림"] and [x["probability"] for x in pc1] == [0.25, 0.35, 0.50, 0.60])


# ---------------------------------------------------------------- statistics
def stats_tests(tmp):
    idx = E.boot_index(250, seed=C.SEED)
    check("bootstrap: 10,000 draws of circular 21-day blocks with seed 25 (differs from seeds 18 and 23, repeats itself)",
          idx.shape == (10_000, 250) and not np.array_equal(idx, E.boot_index(250)) and not np.array_equal(idx, E.boot_index(250, seed=23))
          and np.array_equal(idx, C.boot(250)[0]) and C.SEED == 25)
    # a small plan through simulate and the tables
    root = os.path.join(tmp, "stats")
    days = ["20190301", "20190304", "20190305", "20190306", "20200302", "20200303"]
    tasks = [{**task_of(root, save(root, sym, d, trend_day(200 + 10 * i + j, bid0=BID + 30_000 * j))), "adv": 1.5e6}
             for i, d in enumerate(days) for j, sym in enumerate(("AA", "BB"))]
    hm = hand({"trade_imbalance_30s": 3.0, "mid_change_5s": 0.4})
    names = list(C.BASE + C.CANDS)
    par, OUT, G, info = C.simulate(tasks, names, (), 1, model=hm, grid=True)
    tabs, st = C.tables(par, OUT, names, DV.REPORT)
    IS, _ = C.is_table(par, OUT, names, 3, 1)
    c = max(C.CANDS, key=lambda k: tabs["L3D1"][k]["firing"]["parents"])
    row = tabs["L3D1"][c]
    yrs = par.date.str[:4].to_numpy()
    usd = float(((IS["R0"] - IS[c]) * 1000 / 100).sum())
    check("dollar saving against R0 (ANSWERS A7): the sum over the parents of (IS_R0 - IS_rule) x 1,000 shares / 100, in total and by year; "
          "the paired difference in cents per share times the number of parents times 10 is the same number with the other sign",
          abs(row["usd_saved_vs_R0"] - usd) < 1e-6 and abs(row["usd_saved_vs_R0"] + row["minus_R0_mean"] * len(par) * 10) < 1e-6
          and abs(sum(v["usd_saved_vs_R0"] for v in row["by_year"].values()) - usd) < 1e-6
          and abs(row["by_year"]["2020"]["usd_saved_vs_R0"] - float(((IS["R0"] - IS[c])[yrs == "2020"] * 10).sum())) < 1e-6
          and tabs["L3D1"]["R0"]["usd_saved_vs_R0"] == 0.0 and row["firing"]["parents"] > 0 and C.USD_PER_CENT == 10.0, run=c, usd=usd)
    fr = DV.flat_rows(tabs)
    daily = C.daily_rows(par, OUT, names)
    g = daily[(daily.variant == "L3D1") & (daily.run == c)]
    check("tables and files: S0, S2i, C3, R0 and the 14 candidates under both fill rules; the daily file (date, variant, run, parents, sum of "
          "IS) gives the table mean; the flat file carries the dollar saving",
          set(tabs) == {"L3D1", "L0D0"} and list(tabs["L3D1"]) == list(DV.REPORT) and len(fr) == 2 * len(DV.REPORT) and "usd_saved_vs_R0" in fr.columns
          and "usd_saved_vs_R0_2019" in fr.columns and abs(g.is_sum_c.sum() / g.n_parents.sum() - row["is_mean"]) < 1e-9
          and set(daily.run) == set(DV.REPORT) and tabs["L3D1"]["R0"]["firing"]["parents"] == 0.0)
    # diagnostics against direct formulas
    dg = C.diagnostics(G)
    q, mv, p, y, valid = G["q"], G["move"], G["P23"], G["y"], G["valid"]
    ok = dg["grid_points"] == q.size and sum(b["points"] for b in dg["q_bins"]) == q.size
    for k in ("g", "q", "v"):
        x = G[k][~np.isnan(G[k])]
        ok = ok and dg["percentiles"][k]["n"] == len(x) and abs(dg["percentiles"][k]["p95"] - np.percentile(x, 95)) < 1e-12
    b2 = dg["q_bins"][2]
    m = (q >= 0.5) & (q < 1.0)
    keep = m & valid & (y >= 0) & np.isfinite(p)
    ok = (ok and b2["bin"] == "[0.5, 1)" and b2["points"] == int(m.sum()) and abs(b2["mean_abs_move_60s_c"] - mv[m & ~np.isnan(mv)].mean()) < 1e-12
          and b2["scored_points"] == int(keep.sum()) and abs(b2["auc"] - T23.auc_direct(p[keep], y[keep])) < 1e-12
          and dg["q_bins"][-1]["bin"] == "no signal" and dg["q_bins"][-1]["points"] == int(np.isnan(q).sum())
          and [b["bin"] for b in dg["q_bins"][:5]] == ["[0, 0.25)", "[0.25, 0.5)", "[0.5, 1)", "[1, 2)", "[2, inf)"])
    pe = S.engine_p("B1_p60_k0.5", G, p)
    ok = ok and abs(dg["condition_share_of_grid"]["B1_p60_k0.5"] - float((pe >= 0.6).mean())) < 1e-15 and set(dg["condition_share_of_grid"]) == set(C.CANDS)
    ok = ok and abs(dg["condition_share_of_grid"]["A2_g3"] - float((G["g"] >= 3.0).mean())) < 1e-15
    check("diagnostics (SPEC25 6): percentiles of g, q, v over the points with a signal; by q bin [0, 0.25), [0.25, 0.5), [0.5, 1), [1, 2), "
          "[2, inf) and 'no signal' the points, the mean absolute 60-second move and the AUC of the model on the scored points of the bin "
          "(rank formula); the share of decision points where each candidate's condition holds", ok, bins=[(b["bin"], b["points"]) for b in dg["q_bins"]])
    # the regression function against a stand-in record of research 23's validation run made with research 23's own code
    n23 = list(C23.BASE + ("R1_0.60", "R2_0.40", "R3_0.60_0.40"))
    par23, OUT23, G23, _ = C23.simulate(tasks, n23, (), 1, model=hm, grid=True)
    tabs23, _ = C23.tables(par23, OUT23, n23, ("S0", "S2i", "C3", "R0", "R1_0.60", "R2_0.40", "R3_0.60_0.40"))
    f_par, f_sum = os.path.join(root, "dev23_valid_parents.npz"), os.path.join(root, "dev23_summary.json")
    np.savez_compressed(f_par, date=par23.date.to_numpy().astype("U8"), sym=par23.sym.to_numpy().astype("U8"), dir=par23.dir.to_numpy(),
                        start=par23.start.to_numpy(), mid2=par23.mid2.to_numpy(), spread=par23.spread.to_numpy(), out=OUT23, runs=np.array(n23),
                        lrules=np.array(E.LRULES), P23=G23["P23"], y=G23["y"], valid=G23["valid"])
    json.dump({"tables": tabs23}, open(f_sum, "w"), default=C.jdefault)
    nk = list(C.BASE + C.K0)
    park, OUTk, Gk, _ = C.simulate(tasks, nk, (), 1, model=hm, grid=True)
    pairs = [(k, S.K0[k][1]) for k in C.K0]
    with patched([(DV, "R23_PARENTS", f_par), (DV, "R23_SUMMARY", f_sum), (DV, "CHECK23", True)]):
        ok1, chk1 = DV.regression(park, OUTk, nk, Gk, pairs)
        bad = OUTk.copy()
        bad[1, nk.index("k0:B2"), 7, 5] += 5
        ok2, chk2 = DV.regression(park, bad, nk, Gk, pairs)
        ok3, chk3 = DV.regression(par, OUT, names, G)                           # the validation run's own gate: base runs only
        Gb = {**Gk, "P23": np.where(np.isnan(Gk["P23"]), np.nan, Gk["P23"] + 1e-12)}
        ok4, chk4 = DV.regression(park, OUTk, nk, Gb, pairs)
    with patched([(DV, "CHECK23", False)]):
        ok5, chk5 = DV.regression(park, OUTk, nk, Gk, pairs)
    check("regression function: against a record made with research 23's own code on the same synthetic plan, the parents, the outputs of "
          "S0, S2i, R0 and of the three kappa = 0 runs, the probability and the mean IS are equal (differences exactly 0); one changed "
          "output of one parent, or a probability changed by 1e-12, fails it; without a record (synthetic world) only the engine checks run",
          ok1 and chk1["mean_is_max_abs_diff"] == 0.0 and all(chk1["outputs_equal"].values()) and len(chk1["outputs_equal"]) == 6 and chk1["probability_equal"]
          and not ok2 and chk2["outputs_differing_parents"]["k0:B2 = research 23's R2_0.40"] == 1 and ok3 and len(chk3["outputs_equal"]) == 3
          and not ok4 and not chk4["probability_equal"] and ok5 and "outputs_equal" not in chk5, first=chk1["outputs_equal"], n=chk1["n_parents"])


# ---------------------------------------------------------------- the synthetic world
DEV_DAYS = {2019: ["20190301", "20190304", "20190305", "20190306", "20190307", "20190308", "20190311", "20190312"],
            2020: ["20201118", "20201119", "20201120", "20201123", "20201124", "20201125", "20201127", "20201130"]}
SEALED_DAYS = ["20210104", "20210105", "20210106", "20210107", "20210108", "20210111", "20251029", "20251030", "20251031", "20251103",
               "20251104", "20251105"]
HALF_DAYS = ("20201127",)
WORLD_MODEL = {"trade_imbalance_30s": 3.0, "mid_change_5s": 0.4}


def burst(b, seed):
    """Every 30 minutes two minutes with six times the volume and a 15-cent move every 10 seconds, so that g and v reach
    their thresholds somewhere in the world (the volume identity and the order of the quotes are kept)."""
    rng = np.random.default_rng(seed)
    off = np.zeros(NSEC, np.int64)
    for t0 in range(900, NSEC - 300, 1800):
        for c in ("vol_bid", "vol_ask"):
            b[c][t0:t0 + 120] *= 6
        for j in range(12):
            off[t0 + 10 * j:] += 1500 * (1 if rng.random() < 0.5 else -1)
    for c in QCOLS + ("px_min", "px_max"):
        b[c] = b[c] + off
    b["vol"] = b["vol_bid"] + b["vol_ask"]
    return b


def build_world(tmp):
    """Folders: tmp/study25_volatility (the stand-in repository), tmp/study23_combined_forecast (the stand-in model), tmp/study21_more_pairs (reference
    manifests), tmp/data/... (bars)."""
    repo, data, r23 = os.path.join(tmp, "study25_volatility"), os.path.join(tmp, "data"), os.path.join(tmp, "study23_combined_forecast")
    os.makedirs(os.path.join(repo, "results"))
    os.makedirs(r23)
    seed, src, dev_ref = 700, {}, []
    for y, days in DEV_DAYS.items():
        root = os.path.join(data, f"dev_{y}")
        rows = []
        for d in days:
            for s in ("DA", "DB"):
                seed += 1
                day = trend_day(seed, bid0=BID + 30_000 * (s == "DB"))
                rows.append(save(root, s, d, burst(day, seed) if s == "DB" else day))
            if d != "20190306":                                              # one validation day without an SPY file
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
            day = trend_day(seed, bid0=BID + 20_000 * (s == "SB"))
            rows.append(save(sroot, s, d, burst(day, seed) if s == "SB" else day, mult))
        seed += 1
        ref.append(save(rroot, "SPY", d, rand_day(seed, bid0=3_000_000, tick_p=0.5), mult))
    seed += 1
    rows.append(save(sroot, "SA", "20210112", trend_day(seed)))                # a day that is not in the calendar
    tsv(os.path.join(repo, "SEALED_MANIFEST25.tsv"), rows)
    tsv(os.path.join(tmp, "study21_more_pairs", "SEALED_MANIFEST21.tsv"), ref)
    tsv(os.path.join(repo, "EXCLUDE25.tsv"), [("SB", "20210105", "2021/SB_20210105.npz", "test")], cols=("sym", "date", "file", "reason"))
    pd.DataFrame({"date": [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in SEALED_DAYS], "close_time": "16:00"}).to_csv(os.path.join(repo, "calendar25.csv"), index=False)
    for y in range(2021, 2027):
        tsv(os.path.join(repo, f"universe25_{y}.tsv"), [(y, 1, "SA"), (y, 2, "SB")], cols=("year", "slot", "sym"))
    for doc in RT.DOCS:
        open(os.path.join(repo, f"{doc}.md"), "w").write(f"{doc} stand-in\n")
        open(os.path.join(repo, f"{doc}.sha256"), "w").write(f"{C.sha(os.path.join(repo, doc + '.md'))}  {doc}.md\n")
    model = os.path.join(r23, "model23.json")
    json.dump(hand(WORLD_MODEL), open(model, "w"), indent=1)
    open(os.path.join(r23, "model23.sha256"), "w").write(f"{C.sha(model)}  model23.json\n")
    res = os.path.join(repo, "results")
    pairs = [(L15, "SOURCES", src), (L15, "YEARS", range(2019, 2021)), (L22, "DEV_SPY_ROOT", os.path.join(data, "ref_dev")),
             (L22, "DEV_SPY_MANIFEST", os.path.join(tmp, "study21_more_pairs", "DATA_MANIFEST21.tsv")), (C, "MODEL23", model), (C, "model23", lambda: M.load(model)),
             (DV, "RES", res), (DV, "LOCAL", os.path.join(data, "dev_results")), (DV, "CALENDAR", os.path.join(repo, "calendar25.csv")), (DV, "CHECK23", False),
             (RT, "HERE", repo), (RT, "RES", res), (RT, "UNLOCK", os.path.join(repo, "UNLOCK25.txt")), (RT, "FREEZE", os.path.join(repo, "FREEZE25.md")),
             (RT, "DEV_SUMMARY", os.path.join(res, "dev25_summary.json")), (RT, "MODEL", model), (RT, "SEALED_ROOT", os.path.join(data, "sealed")),
             (RT, "SEALED_BARS", sroot), (RT, "SEALED_MANIFEST", os.path.join(repo, "SEALED_MANIFEST25.tsv")), (RT, "EXCLUDE", os.path.join(repo, "EXCLUDE25.tsv")),
             (RT, "CALENDAR", os.path.join(repo, "calendar25.csv")), (RT, "UNIVERSE", os.path.join(repo, "universe25_{y}.tsv")), (RT, "REF_SEALED_ROOT", rroot),
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
        lines.append(f"| `{p}` | `{C.sha(q)}` | test |")
    open(RT.FREEZE, "w", encoding="utf-8").write("# FREEZE (synthetic)\n\n| file | sha256 | role |\n|---|---|---|\n" + "\n".join(lines) + "\n")


def world_round(repo, res, label, forced):
    """Validation run -> rehearsal -> guard -> sealed run in the synthetic world; forced: an A* to impose (None: as it falls)."""
    pre = f"world ({label}): "
    side = RT.MODEL.replace(".json", ".sha256")
    n_va, n_days = 2 * sum(len(v) for v in DEV_DAYS.values()), sum(len(v) for v in DEV_DAYS.values())
    # ---- validation run
    err = quiet(DV.valid_stage, 1)
    dev = load_json(RT.DEV_SUMMARY)
    daily = pd.read_csv(os.path.join(res, "dev25_daily.csv"), dtype={"date": str}) if dev else pd.DataFrame()
    imp = dev.get("improves_in_development")
    t3 = dev.get("tables", {}).get("L3D1", {})
    ok = (err is None and all(os.path.exists(os.path.join(res, f)) for f in DV.FILES) and dev["sim"]["symbol_days"] == n_va
          and dev["sim"]["symbol_days_without_ref"] == 2 and dev["sim"]["days_without_ref"] == ["20190306"] and dev["sim"]["first_day"] == "20190301"
          and dev["years"] == [2019, 2020] and set(t3) == set(DV.REPORT) == set(dev["tables"]["L0D0"])
          and len(daily) == n_days * 2 * (len(DV.REPORT) + (1 if imp else 0)) and dev["model_sha256"] == C.sha(RT.MODEL)
          and t3["R0"]["firing"]["parents"] == 0.0 and t3["R0"]["minus_R0_mean"] == 0.0 and t3["R0"]["usd_saved_vs_R0"] == 0.0
          and all(any(t3[c]["firing"]["parents"] > 0 for c in C.CANDS if C.SPECS[c]["family"] == f) for f in "ABC")
          and set(dev["usd_saved_vs_R0_L3D1"]) == set(C.CANDS)
          and imp == (dev["astar"] != "R0") == dev["sealed_run"] and (dev["astar"] == "R0" or dev["astar"] in C.CANDS) and dev["trials_cumulative"] == 865
          and (forced is None or dev["astar"] == forced) and (dev["delay_check"] is not None) == imp == (dev["control"] is not None) == (dev["power"] is not None)
          and (not imp or (dev["control"]["L3D1"]["control"] == C.CTRL + dev["astar"] and dev["checks"]["second_pass_equal"] is True
                           and set(dev["power"]["by_n"]) == {"20", "12", "8"} and dev["power"]["days_new"] == len(SEALED_DAYS)))
          and (imp or dev["sentence_no_improvement"] == DV.SENT_NONE) and dev["hypotheses"] == (["H"] if imp else [])
          and dev["checks"]["daily_file_max_gap_c"] < 1e-6 and dev["sim"]["adv"]["adv_undefined"] == 2 * 2 * 5
          and len(dev["diagnostics"]["q_bins"]) == 6 and set(dev["diagnostics"]["condition_share_of_grid"]) == set(C.CANDS)
          and [p["probability"] for p in dev["predictions_check"]] == [0.25, 0.35, 0.50, 0.60] and dev["checks"]["regression"]["targets"] == "none (synthetic data)"
          and dev["sim"]["grid_counts"]["g_defined"] > 0 and dev["sim"]["grid_counts"]["parents_vol_pre_undefined_or_zero"] > 0)
    check(pre + "the validation run reads the validation years with the frozen model: S0, S2i, R0, C3 and the 14 candidates under two fill "
          "rules with the dollar savings; the selection; the diagnostics; the prediction check; the control, the delay check and the power for "
          "N = 20, 12, 8 only when A* is a candidate; the daily file", ok, error=err, astar=dev.get("astar"), selection=dev.get("selection"))
    err2 = quiet(DV.valid_stage, 1)
    check(pre + "a second validation run is refused (results exist)", err2 is not None and "once" in err2, message=err2)
    if not imp:
        # ---- A* = R0: no rehearsal, no sealed run, whatever else is in place
        reh = quiet(RT.main, "rehearsal", 1)
        write_freeze(repo)
        open(RT.UNLOCK, "w", encoding="utf-8").write(RT.PHRASE + "\n")
        g = RT.guard()
        refused = quiet(RT.main, "sealed", 1)
        nothing = not any(os.path.exists(os.path.join(res, f)) for m in RT.MODES.values() for f in m["files"])
        check(pre + "A* = R0 (개발에서 개선 없음): the rehearsal is refused, and with the freeze file, the unlock phrase and every hash in place "
              "the guard gives that single reason and the sealed run writes nothing", reh is not None and RT.NOT_IMPROVED in reh and g == [RT.NOT_IMPROVED]
              and refused is not None and refused == "refused: " + RT.NOT_IMPROVED and nothing, guard=g, rehearsal=reh)
        return dev
    # ---- rehearsal
    err = quiet(RT.main, "rehearsal", 1)
    reh = load_json(os.path.join(res, "rehearsal25_summary.json"))
    check(pre + "the rehearsal (the sealed code path on the validation data) equals the validation numbers: tables, control and delay check "
          "with max difference 0, no sealed path read", err is None and reh.get("checks", {}).get("pass") is True and reh["checks"]["max_abs_diff"] == 0.0
          and reh["checks"]["table_numbers_compared"] > 100 and reh["checks"]["numbers_compared"] > reh["checks"]["table_numbers_compared"]
          and reh["astar"] == dev.get("astar") and reh["trials_added"] == 0 and reh["sim"]["symbol_days"] == n_va
          and reh["sim"]["days_without_ref"] == ["20190306"], error=err, checks=reh.get("checks"))
    # ---- guard
    g = {"no FREEZE, no unlock": RT.guard()}
    write_freeze(repo)
    g["no unlock file"] = RT.guard()
    open(RT.UNLOCK, "w", encoding="utf-8").write("연구25 테스트를 연다\n")
    g["wrong phrase"] = RT.guard()
    open(RT.UNLOCK, "w", encoding="utf-8").write(RT.PHRASE + "\n")
    g["all in place"] = RT.guard()
    open(os.path.join(repo, "SPEC25_ANSWERS.md"), "a").write("x")
    g["document changed"] = RT.guard()
    open(os.path.join(repo, "SPEC25_ANSWERS.md"), "w").write("SPEC25_ANSWERS stand-in\n")
    core = os.path.join(repo, "signal25.py")
    keep = open(core).read()
    open(core, "w").write(keep + "#\n")
    g["frozen file changed"] = RT.guard()
    open(core, "w").write(keep)
    write_freeze(repo, drop="core25.py")
    g["core file not listed"] = RT.guard()
    write_freeze(repo)
    g["all in place again"] = RT.guard()
    ok = (g["no unlock file"] == ["no unlock phrase in UNLOCK25.txt"] == g["wrong phrase"] and g["all in place"] == [] == g["all in place again"]
          and "FREEZE25.md missing or without the core files" in g["no FREEZE, no unlock"]
          and g["document changed"] == ["SPEC25_ANSWERS.md differs from SPEC25_ANSWERS.sha256"]
          and g["frozen file changed"] == ["frozen file differs: signal25.py"]
          and g["core file not listed"] == ["FREEZE25.md missing or without the core files"])
    check(pre + "guard of the sealed run: refuses without the unlock file, with another phrase, with a changed document, a changed frozen "
          "file or a core file missing from FREEZE25.md; passes only when everything is in place", ok, guard=g)
    # ---- the model file is pinned: any change of its content stops the guard, also when its own sha256 file is rewritten to match
    mtxt, stxt = open(RT.MODEL).read(), open(side).read()
    m0 = json.loads(mtxt)
    gm = {}
    open(RT.MODEL, "w").write(mtxt + "\n")
    gm["one newline appended"] = RT.guard()
    m1 = {**m0, "coef": [m0["coef"][0] + 1e-9] + m0["coef"][1:]}
    open(RT.MODEL, "w").write(json.dumps(m1, indent=1))
    gm["one coefficient changed"] = RT.guard()
    open(side, "w").write(f"{C.sha(RT.MODEL)}  model23.json\n")
    gm["coefficient changed and model23.sha256 rewritten to match"] = RT.guard()
    refused = quiet(RT.main, "sealed", 1)
    nothing = not any(os.path.exists(os.path.join(res, f)) for f in RT.MODES["sealed"]["files"])
    open(RT.MODEL, "w").write(mtxt)
    open(side, "w").write(stxt)
    back = RT.guard()
    check(pre + "a model file whose content alone changes (one newline, one coefficient, also with its own sha256 file rewritten to match) stops "
          "the guard with 'frozen file differs: ../study23_combined_forecast/model23.json', the sealed run writes nothing, and the original content passes again",
          all(v[:1] == ["frozen file differs: ../study23_combined_forecast/model23.json"] for v in gm.values())
          and "model23.json missing or differs from model23.sha256" in gm["one coefficient changed"]
          and "model23.json missing or differs from model23.sha256" not in gm["coefficient changed and model23.sha256 rewritten to match"]
          and any("another model file" in x for x in gm["coefficient changed and model23.sha256 rewritten to match"])
          and refused is not None and refused.startswith("refused") and nothing and back == [], guard=gm, sealed_run=refused)
    # ---- the validation summary is pinned
    txt = open(RT.DEV_SUMMARY, encoding="utf-8").read()
    d0 = json.loads(txt)
    other = next(c for c in C.CANDS if c != d0["astar"])
    cases = {"another A*": {**d0, "astar": other, "selection": {**d0["selection"], "astar": other}},
             "A* set to R0": {**d0, "astar": "R0", "improves_in_development": False, "selection": {**d0["selection"], "astar": "R0"}},
             "improvement flag flipped": {**d0, "improves_in_development": False},
             "one digit changed": {**d0, "tables": {**d0["tables"], "L3D1": {**d0["tables"]["L3D1"], "R0": {**d0["tables"]["L3D1"]["R0"],
                                                                                                         "is_mean": d0["tables"]["L3D1"]["R0"]["is_mean"] + 1e-6}}}}}
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
        json.dump(cases["another A*"], fh, indent=1, default=str, ensure_ascii=False)
    refused = quiet(RT.main, "sealed", 1)
    nothing = not any(os.path.exists(os.path.join(res, f)) for f in RT.MODES["sealed"]["files"])
    open(RT.DEV_SUMMARY, "w", encoding="utf-8").write(txt)
    back = RT.guard()
    check(pre + "a validation summary whose content alone changes (another A*, A* set to R0, the improvement flag, one digit, one newline) "
          "stops the guard with 'frozen file differs', the sealed run writes nothing, and the original content passes again",
          all(v[:1] == ["frozen file differs: results/dev25_summary.json"] for v in res_g.values()) and usable["another A*"] and usable["A* set to R0"]
          and RT.NOT_IMPROVED in res_g["A* set to R0"] and not usable["improvement flag flipped"]
          and any("no usable development selection" in x for x in res_g["improvement flag flipped"]) and refused is not None and refused.startswith("refused")
          and nothing and back == [], guard=res_g, readable_by_selection=usable, sealed_run=refused)
    # ---- the sealed run
    err = quiet(RT.main, "sealed", 1)
    t = load_json(os.path.join(res, "test25_summary.json"))
    td = pd.read_csv(os.path.join(res, "test25_daily.csv"), dtype={"date": str}) if t else pd.DataFrame(columns=["date", "run"])
    a = dev["astar"]
    want_sd = 2 * len(SEALED_DAYS) - 1                                         # SA and SB on the calendar days, minus the excluded SB day
    v = t.get("verdict", {}).get("H", {})
    ok = (err is None and t.get("mode") == "sealed" and t["sim"]["symbol_days"] == want_sd and t["sim"]["excluded_rows"] == 1 and t["astar"] == a
          and t["hypotheses"] == ["H"] and v["level"] == 0.05 and v["p"] == t["differences"]["L3D1"]["Astar_minus_Estar"]["p_one_sided"]
          and v["confirmed"] == (v["p"] < 0.05) and v["sentence"] == RT.SENTENCES[v["confirmed"]] and v["mean_c"] == t["differences"]["L3D1"]["Astar_minus_Estar"]["mean"]
          and t["trials_added"] == 1 and t["trials_before"] == 865 and list(t["tables"]["L3D1"]) == ["S0", "S2i", "C3", "R0", a]
          and set(t["differences"]["L3D1"]) == {"Estar_minus_S0", "C3_minus_Estar", "S2i_minus_Estar", "Astar_minus_Estar", "Astar_minus_S0",
                                                "Astar_minus_C3", "Astar_minus_S2i"} == set(t["differences"]["L0D0"])
          and t["date_split"]["L3D1"]["Estar_minus_S0"]["before"]["days"] == 9 and t["date_split"]["L3D1"]["Estar_minus_S0"]["from"]["days"] == 3
          and t["counts"]["parents_before_cut"] + t["counts"]["parents_from_cut"] == t["counts"]["parents"]
          and t["common_only"]["excluded_symbol_years"] == ["2021 SB"] and 0 < t["common_only"]["parents"] < t["counts"]["parents"]
          and set(td.date) == set(SEALED_DAYS) and set(td.run) == {"S0", "S2i", "R0", "C3", a, C.CTRL + a}
          and {"control_minus_Estar", "Astar_minus_control"} <= set(t["control"]["L3D1"]) and "R0_delay_minus_R0" in t["delay_check"]["L3D1"]
          and f"{a}_delay_minus_R0_delay" in t["delay_check"]["L3D1"]
          and abs(t["dollars"]["L3D1"]["Astar_saving_vs_Estar_total"] + t["differences"]["L3D1"]["Astar_minus_Estar"]["mean"] * t["counts"]["parents"] * 10) < 1e-6
          and abs(sum(t["dollars"]["L3D1"]["by_year"].values()) - t["dollars"]["L3D1"]["Astar_saving_vs_Estar_total"]) < 1e-6
          and t["read"]["bars_root"].endswith("bars_1s_new_stocks") and t["model_sha256"] == C.sha(RT.MODEL) and t["sim"]["adv"]["adv_undefined"] > 0)
    check(pre + "the sealed run with the unlock phrase: universe symbols on calendar days minus the exclude list, H (A* - E* < 0, L3D1, "
          "one-sided, 0.05) with its fixed sentence, E* - S0, C3 - E*, S2i - E*, A* against S0 / C3 / S2i, the dollar saving, the control, the "
          "delay check, the split at 2025-11-03, common stocks only, three files", ok, error=err, verdict=t.get("verdict"), counts=t.get("counts"))
    again = quiet(RT.main, "sealed", 1)
    check(pre + "a second sealed run is refused (test results already exist)", again is not None and "test results already exist" in again, message=again)
    return dev


def world_tests(tmp):
    repo, res, pairs = build_world(tmp)
    with patched(pairs):
        dev = world_round(repo, res, "selection as it falls", None)
        # second round: the same world with the other outcome forced so that both outcomes of the selection run end to end
        os.rename(res, os.path.join(repo, "results_round1"))
        os.makedirs(res)
        for f in ("FREEZE25.md", "UNLOCK25.txt"):
            if os.path.exists(os.path.join(repo, f)):
                os.rename(os.path.join(repo, f), os.path.join(repo, f + ".round1"))
        os.rename(DV.LOCAL, DV.LOCAL + "_round1")
        force = "R0" if dev.get("improves_in_development") else "B3_k0.5"
        if force == "R0":
            fake = lambda rows, r0: ("R0", {"lowest_mean": "A1_g2", "lowest_mean_c": rows["A1_g2"]["mean"], "r0_mean_c": r0,              # noqa: E731
                                            "r0_minus_lowest_c": r0 - rows["A1_g2"]["mean"], "threshold_c": C.TIE, "improves_in_development": False, "astar": "R0"})
        else:
            fake = lambda rows, r0: (force, {"lowest_mean": force, "lowest_mean_c": rows[force]["mean"], "r0_mean_c": r0,   # noqa: E731
                                             "r0_minus_lowest_c": r0 - rows[force]["mean"], "threshold_c": C.TIE, "improves_in_development": True,
                                             "tied_within_0.02": [force], "lowest_p95_among_tied": [force], "fired": {force: rows[force]["fired"]}, "astar": force})
        with patched([(C, "select_astar", fake)]):
            dev2 = world_round(repo, res, "forced A* " + force, force)
    check("world: the two rounds cover both outcomes of the selection (A* = R0: no rehearsal and no sealed run; a candidate: rehearsal, guard "
          "and sealed run)", {dev.get("improves_in_development"), dev2.get("improves_in_development")} == {True, False},
          round1=dev.get("astar"), round2=dev2.get("astar"))


def repo_guard_test():
    why = RT.guard()
    check("the real sealed run is refused now (no UNLOCK25.txt: Claude never writes it; no FREEZE25.md)", "no unlock phrase in UNLOCK25.txt" in why
          and "FREEZE25.md missing or without the core files" in why and not os.path.exists(os.path.join(HERE, "UNLOCK25.txt")), reasons=why)
    summ = json.load(open(os.path.join(HERE, "results", "universe25_summary.json")))
    nonc = {(int(y), s) for y, v in summ["years"].items() for s in v["share_code_not_10_11"]}
    check("constants: seed 25, 21-day blocks, 10,000 draws, tie 0.02 cents, 20 parents of 1,000 shares over 300 s, market fee 0.3 cents, L3D1 "
          "primary, validation 2019-2020, N of 20 / 12 / 8, trials 851 -> 865 -> 866, the unlock phrase, the sealed window, the fixed sentences "
          "of SPEC25 7 and the 24 symbol-years of universe E that are not common stock",
          C.SEED == 25 and E.BOOT_BLOCK == 21 and E.BOOT_B == 10_000 and C.TIE == 0.02 and E.N_PARENT == 20 and E.QTY == 1000 and E.HOR == 300
          and E.FEE_MKT == 0.3 and RT.PHRASE == "연구25 테스트를 한 번 연다" and RT.WINDOW == ("20210104", "20261110") and L22.SIZE_FROM == "20251103" == C.CUT
          and DV.NS == (20, 12, 8) and E.VARIANTS[0] == ("L3D1", 3, 1) and DV.VALID == (2019, 2020) == RT.VALID
          and (DV.TRIALS_BEFORE, DV.TRIALS_ADDED, RT.TRIALS_BEFORE) == (851, 14, 865) and DV.SENT_NONE == RT.SENTENCES["none"]
          and RT.SENTENCES[True] == "직전 변동성과 거래량을 조건에 넣으면 단순한 인내보다 더 쌌다"
          and RT.SENTENCES[False] == "직전 변동성과 거래량을 조건에 넣어도 평균 비용은 확실히 줄지 않았다"
          and RT.NONCOMMON["sealed"] == nonc and len(nonc) == 24 and {"../study23_combined_forecast/model23.json", "results/dev25_summary.json"} <= set(RT.CORE)
          and C.ALPHA == 0.05 and C.QBINS == (0.0, 0.25, 0.5, 1.0, 2.0, float("inf")))


def main():
    with tempfile.TemporaryDirectory(prefix="r25_tests_") as tmp:
        r18, r20, r22, r23 = earlier_tests(tmp)
        frozen_tests()
        signal_tests()
        future_tests()
        flag_tests()
        engine_tests(tmp)
        rule_tests()
        stats_tests(tmp)
        world_tests(os.path.join(tmp, "world"))
    repo_guard_test()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    n_ok = sum(r["pass"] for r in RESULTS)
    path = os.path.join(HERE, "results", "engine25_tests.json")
    T23.keep_old(path)
    with open(path, "w") as fh:
        json.dump({"passed": n_ok, "total": len(RESULTS), "research18_tests": r18, "research20_tests": r20, "research22_tests": r22, "research23_tests": r23,
                   "numba": bool(E.HAVE_NUMBA), "tests": RESULTS}, fh, indent=1, default=str, ensure_ascii=False)
    for r in RESULTS:
        print(("PASS " if r["pass"] else "FAIL ") + r["test"][:150])
        if not r["pass"]:
            print("     ", json.dumps(r, default=str, ensure_ascii=False)[:2500])
    print(f"{n_ok}/{len(RESULTS)} passed (research 18's tests: {r18}, research 20's tests: {r20}, research 22's tests: {r22}, research 23's tests: {r23})")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
