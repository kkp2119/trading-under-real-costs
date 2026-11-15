#!/usr/bin/env python3
"""Research 23 synthetic tests (step-2 prompt 1; they must pass before the model is fitted). Synthetic bars only; no bar
of research 23, of the development universe or of SPY is read, and the sealed folders are not named outside temporary
stand-ins.
  earlier   research 18's 78 engine tests, research 20's 117 signal tests and research 22's synthetic tests (all but its
            two tests of research 22's own live folder), rerun here through research 22's test functions
  frozen    the files of research 18, 20 and 22 used here equal the files of their tags
  inputs    trade imbalance: sign, the window u-29..u; trade intensity: the window u-59..u, the cut at 0 and 10, the
            usual volume (previous 20 trading days of the year, at least 5) and the fill with the training median;
            zero denominators; inputs 5 and 6 are 0 without a signal and cut at +-4; inputs 1-4 equal engine18's
  future    values after a decision bar change none of the nine inputs at that bar
  model     research 18's training order line by line gives the same numbers as the new code with four inputs; fills,
            standardisation, the probability formula, a left-out input, the model file and its sha256
  engine    the probabilities reach engine18 (early fill, hold), the control with 1 - p, research 18's model as a second
            source, an empty symbol-day, the size correction through loader22
  rules     A*: the 0.02 cent threshold against R0 and the tie rule among candidates; Holm with one and two hypotheses
  stats     bootstrap seed 23, the AUC and the other prediction metrics against direct formulas, the daily sums
  world     a small synthetic world through fit -> validation run -> rehearsal -> guard -> sealed run, once with the
            selection as it falls and once with a forced A*; a changed model file or validation summary stops the guard
Writes results/engine23_tests.json (an earlier file is moved to results/superseded/); exits 1 when a test fails.
  python test23.py"""
import datetime as dt
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
import core23 as C  # noqa: E402
import dev23 as DV  # noqa: E402
import run_test23 as RT  # noqa: E402
import test22 as T22  # noqa: E402  (research 22's test functions; its main is never called)

E, F, M, L22, SG = C.E, C.F, C.M, C.L22, C.SG
L15, NSEC = L22.L15, L22.NSEC
BID, SPR, U0 = T22.BID, T22.SPR, T22.U0
flat, setq, rand_day, save, tsv, task_of, patched, quiet = (T22.flat, T22.setq, T22.rand_day, T22.save, T22.tsv, T22.task_of,
                                                            T22.patched, T22.quiet)
RESULTS = []
TESTS22_OWN_FOLDER = 2                     # research 22's repo_guard_test: two tests of its own live folder, not rerun here
EXTRA_FROZEN = {"study22_market_signal/test22.py": ("code-freeze-22", "06a6a2e01d5492a4c7044466b7f74be9b406bd672936ab6417bc86970aa4b058"),
                "study22_market_signal/dev22.py": ("code-freeze-22", "ce44b1cc34c51e5060bc338828bee6681d4c6b931b1b30c62e8c5f29d0dc898d"),
                "study22_market_signal/run_test22.py": ("code-freeze-22", "95feef44b31cef8b553706dc0a910a1c8a86a5ed1a742399dcca7f8240236b77"),
                "study18_forecast/test_engine18.py": ("test-run-18", "23336a2cb63babde6075b4eb14e086d7c8d72c13894f864326bac1a102b6516a"),
                "study20_pairs/test20.py": ("code-freeze-20", "388cf5bad6dbedb812c0323bfe0ccee06121262b2436141ff127bc5a9912f1fa")}
ALL = list(range(len(F.FEATS)))


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


def hand(coef, mean=None, sd=None, intercept=0.0):
    """A hand-made nine-input model: p = sigmoid(intercept + sum coef_j (x_j - mean_j) / sd_j)."""
    c = np.zeros(len(F.FEATS))
    for k, v in coef.items():
        c[F.FEATS.index(k)] = v
    return {"features": list(F.FEATS), "cols": ALL, "mean": list(mean or np.zeros(len(F.FEATS))), "sd": list(sd or np.ones(len(F.FEATS))),
            "coef": c.tolist(), "intercept": intercept, "fills": {"vol_pre": 5.0, "intensity": 1.0}, "base_rate_train": 0.5}


def hand18(coef, intercept=0.0):
    """A hand-made four-input model in research 18's form."""
    return {"features": list(E.FEATS), "mean": [0.0, 0.0, 0.0, 0.0], "sd": [1.0, 1.0, 1.0, 1.0], "coef": list(coef), "intercept": intercept,
            "vol_pre_fill": 5.0, "base_rate_train": 0.5}


def arr(x):
    return np.array(x, dtype=np.int64)


# ---------------------------------------------------------------- earlier studies
def earlier_tests(tmp):
    T22.RESULTS.clear()
    r18, r20 = T22.earlier_tests()
    for fn in (T22.frozen_tests, T22.flag_tests, T22.engine_tests, T22.no_signal_tests, T22.future_tests, T22.basis_tests):
        fn()
    sub = os.path.join(tmp, "r22")
    os.makedirs(sub)
    T22.size_tests(sub)
    T22.rule_tests()
    T22.stats_tests()
    T22.world_tests(sub)
    got = list(T22.RESULTS)
    stored = json.load(open(os.path.join(PROJ, "study22_market_signal", "results", "engine22_tests.json")))["tests"]
    want = [r["test"] for r in stored][:-TESTS22_OWN_FOLDER]
    n_ok = sum(r["pass"] for r in got)
    for r in got[:2]:
        check(r["test"], r["pass"], passed=r.get("passed"), total=r.get("total"), failed=r.get("failed"))
    check(f"research 22's synthetic tests pass here ({len(want)}: the names of its stored result without the two tests of its own live folder)",
          len(got) == len(want) == n_ok and [r["test"] for r in got] == want, passed=n_ok, total=len(got), failed=[r["test"] for r in got if not r["pass"]])
    return r18, r20, f"{n_ok}/{len(got)}"


def frozen_tests():
    got = {}
    for p, (tag, h) in {**C.FROZEN, **EXTRA_FROZEN}.items():
        blob = subprocess.run(["git", "show", f"{tag}:{p}"], capture_output=True, cwd=PROJ).stdout
        got[p] = (hashlib.sha256(blob).hexdigest() == h, hashlib.sha256(open(os.path.join(PROJ, p), "rb").read()).hexdigest() == h)
    check("core22, loader22, signal22, research 18's model file and the test files of research 18, 20 and 22 equal the files of their tags (git) "
          "and the working files", all(a and b for a, b in got.values()) and C.frozen_differences() == [], files={p: list(v) for p, v in got.items()})
    m = C.model18()
    check("research 18's frozen model is read only when it equals its sha256: four inputs in research 18's order",
          m["features"] == list(E.FEATS) == list(F.FEATS[:4]) and len(m["coef"]) == 4 and C.sha(C.MODEL18) == C.FROZEN["study18_forecast/model18.json"][1])


# ---------------------------------------------------------------- inputs 7-9
def trade_tests():
    s = U0 - 50
    adv = 390 * 10_000.0                                                    # the usual 60-second volume is 10,000 shares
    b = flat()
    b["vol_ask"][s - 20:s] = 500
    b["vol_bid"][s - 20:s] = 100
    b["vol"][s - 20:s] = 600
    imb, inten = F.trade_inputs(b, arr([1, -1]), arr([s, s]), adv)
    check("trade imbalance sign: more buyer-initiated volume is positive for a buy order and negative for a sell order",
          abs(imb[0, 0] - 400 / 600) < 1e-12 and abs(imb[1, 0] + 400 / 600) < 1e-12 and np.array_equal(imb[0], -imb[1]), buy=imb[0, :8], sell=imb[1, :8])
    check("trade intensity: the 60-second volume over the usual 60-second volume (previous-days mean / 390), the same for a buy and a sell",
          abs(inten[0, 0] - 20 * 600 / 10_000) < 1e-12 and np.array_equal(inten[0], inten[1]), intensity=inten[0, :14])
    got_i, got_t = {}, {}
    for w in (s - 60, s - 59, s - 30, s - 29, s, s + 1):
        b = flat()
        b["vol_ask"][w], b["vol"][w] = 300, 300
        i_, t_ = F.trade_inputs(b, arr([1]), arr([s]), adv)
        got_i[w - s], got_t[w - s] = float(i_[0, 0]), float(t_[0, 0])
    check("windows at decision bar u: the imbalance uses the bars u-29..u (a trade at u-30 or u+1 is out, at u-29 and u is in); the intensity "
          "uses the bars u-59..u (u-60 and u+1 out, u-59 and u in)",
          got_i == {-60: 0.0, -59: 0.0, -30: 0.0, -29: 1.0, 0: 1.0, 1: 0.0} and [got_t[k] > 0 for k in (-60, -59, -30, -29, 0, 1)] == [False, True, True, True, True, False]
          and abs(got_t[0] - 300 / 10_000) < 1e-15, imbalance=got_i, intensity=got_t)
    # every decision bar of random parents against a direct loop (vol_in counts for the intensity, not for the imbalance)
    b = rand_day(31)
    rng = np.random.default_rng(32)
    b["vol_in"] = rng.integers(0, 4, NSEC) * 100
    b["vol"] = b["vol_bid"] + b["vol_ask"] + b["vol_in"]
    dirs, starts = E.parents("20190305", NSEC)
    imb, inten = F.trade_inputs(b, dirs, starts, 2.0e6)
    wi, wt = 0.0, 0.0
    for i in range(len(dirs)):
        for k in range(E.NG):
            u = int(starts[i]) + 5 * k
            buy, sell = b["vol_ask"][u - 29:u + 1].sum(), b["vol_bid"][u - 29:u + 1].sum()
            wi = max(wi, abs(imb[i, k] - (dirs[i] * (buy - sell) / (buy + sell) if buy + sell else 0.0)))
            wt = max(wt, abs(inten[i, k] - min(max(b["vol"][u - 59:u + 1].sum() / (2.0e6 / 390), 0.0), 10.0)))
    check("trade inputs at every decision bar of 20 random parents equal a direct loop over the bars (vol_in and vol_unc enter the "
          "intensity only)", wi < 1e-12 and wt < 1e-12 and (inten > 0).all(), max_abs_diff=[wi, wt])
    i0, t0 = F.trade_inputs(b, arr([1]), arr([20]), 2.0e6)                   # not a real start bar: the window starts before 09:30:00
    check("a window that would start before the first bar uses the bars there are",
          abs(t0[0, 0] - min(b["vol"][0:21].sum() / (2.0e6 / 390), 10.0)) < 1e-12
          and abs(i0[0, 0] - (b["vol_ask"][0:21].sum() - b["vol_bid"][0:21].sum()) / (b["vol_ask"][0:21].sum() + b["vol_bid"][0:21].sum())) < 1e-12)
    # zero denominators, the cut and the undefined usual volume
    b = flat()
    i_, t_ = F.trade_inputs(b, arr([1, -1]), arr([s, s]), adv)
    b2 = flat()
    b2["vol_in"][s - 10:s], b2["vol"][s - 10:s] = 400, 400
    i2, t2 = F.trade_inputs(b2, arr([1]), arr([s]), adv)
    b3 = flat()
    b3["vol_ask"][s - 5:s], b3["vol"][s - 5:s] = 100_000, 100_000
    i3, t3 = F.trade_inputs(b3, arr([-1]), arr([s]), adv)
    check("no trade in the window: imbalance 0 and intensity 0; trades inside the spread only: imbalance 0 (denominator 0) with a positive "
          "intensity; a very large volume: intensity cut at 10",
          not i_.any() and not t_.any() and i2[0, 0] == 0.0 and abs(t2[0, 0] - 0.4) < 1e-12 and t3[0, 0] == 10.0 and i3[0, 0] == -1.0
          and t3.min() >= 0.0 and t3.max() <= 10.0)
    und = [F.trade_inputs(b3, arr([-1]), arr([s]), a)[1] for a in (np.nan, None, 0.0)]
    raw = F.raw(b3, None, NSEC, arr([-1]), arr([s]), np.nan)
    X = F.design(raw["X"], {"vol_pre": 7.0, "intensity": 0.8})
    check("no usual volume (fewer than 5 earlier days, or a mean of 0): the intensity is undefined, takes the training median when the "
          "inputs are built, and the product is formed after that",
          all(np.isnan(u).all() for u in und) and np.isnan(raw["X"][..., F.I_INT]).all() and np.isnan(raw["X"][..., F.I_PROD]).all()
          and (X[..., F.I_INT] == 0.8).all() and np.array_equal(X[..., F.I_PROD], X[..., F.I_IMB] * 0.8) and X[0, 0, F.I_PROD] == -0.8
          and raw["counts"]["intensity_undefined"] == E.NG)
    rawd = F.raw(b3, None, NSEC, arr([-1]), arr([s]), adv)
    Xd = F.design(rawd["X"], {"vol_pre": 7.0, "intensity": 0.8})
    check("with a usual volume the product is imbalance x intensity at every decision bar and the fill value is not used",
          np.array_equal(Xd[..., F.I_PROD], Xd[..., F.I_IMB] * Xd[..., F.I_INT]) and Xd[0, 0, F.I_PROD] == -10.0 and rawd["counts"]["intensity_undefined"] == 0)


def adv_tests(tmp):
    root = os.path.join(tmp, "adv")
    days = {2019: [f"201903{d:02d}" for d in (1, 4, 5, 6, 7, 8, 11, 12)], 2020: ["20200102", "20200103"]}
    tasks, vols = [], {}
    for y, dd in days.items():
        for j, d in enumerate(dd):
            for sym in ("AA", "BB"):
                if sym == "BB" and d in ("20190305", "20190306"):
                    continue                                                 # BB has no file on two days
                b = flat()
                v = 1000 * (j + 1) * (2 if sym == "BB" else 1)
                b["vol_in"][100], b["vol"][100] = v, v
                vols[(sym, d)] = v
                tasks.append(task_of(root, save(root, sym, d, b)))
    out, info = C.with_adv(tasks, {y: list(v) for y, v in days.items()}, 1)
    adv = {(t["sym"], t["date"]): t["adv"] for t in out}
    ok = (all(np.isnan(adv[("AA", d)]) for d in days[2019][:5]) and adv[("AA", "20190308")] == np.mean([1000 * k for k in range(1, 6)])
          and adv[("AA", "20190312")] == np.mean([1000 * k for k in range(1, 8)])
          and np.isnan(adv[("BB", "20190311")]) and adv[("BB", "20190312")] == np.mean([2000 * k for k in (1, 2, 5, 6, 7)])
          and all(np.isnan(adv[(s, d)]) for s in ("AA", "BB") for d in days[2020]) and info["adv_undefined"] == 5 + 5 + 4 and len(out) == len(tasks))
    check("usual volume: the mean day volume of the symbol over the earlier trading days of the same year in the plan (up to 20), undefined "
          "with fewer than 5 of them; a missing file is not a day; a new year starts again", ok, info=info,
          aa={d: adv[("AA", d)] for d in days[2019]}, bb_last=adv[("BB", "20190312")])
    many = {2019: [f"2019{m:02d}{d:02d}" for m in (3, 4) for d in range(1, 16)]}          # 30 days: the window is the previous 20
    dv = {("ZZ", d): float(i + 1) for i, d in enumerate(many[2019])}
    a = E.adv_prior(dv, many, list(dv))
    check("usual volume: with more than 20 earlier days only the previous 20 are used", a[("ZZ", many[2019][25])][0] == np.mean(np.arange(6, 26))
          and a[("ZZ", many[2019][25])][1] == 20 and E.ADV_WIN == 20 and E.ADV_MIN == 5 and F.MINUTES == 390)
    t = next(t for t in out if (t["sym"], t["date"]) == ("AA", "20190312"))
    r = C.symday(t, {"runs": (), "delay_runs": (), "model": None, "m18": None, "grid": True})
    check("a symbol-day of the plan carries its usual volume into input 8", len(r["dir"]) == 20 and not np.isnan(r["X"][..., F.I_INT]).any()
          and r["X"][..., F.I_INT].max() <= 10.0 and r["outs"].shape == (2, 0, 20, E.NCOL))


# ---------------------------------------------------------------- inputs 1-6
def signal_tests():
    s = U0 - 50
    ref = flat(bid=3_000_000, spread=100)
    r3 = F.raw(T22.dip(3.0), ref, NSEC, arr([1, -1]), arr([s, s]), np.nan)
    r6 = F.raw(T22.dip(6.0), ref, NSEC, arr([1, -1]), arr([s, s]), np.nan)
    a3, a6 = r3["X"][:, :, 4:6], r6["X"][:, :, 4:6]
    check("inputs 5 and 6: after a 3-cent dip the adverse score is about +3 for a buy and -3 for a sell on both bases (research 22's "
          "score), 0 before the dip; after a 6-cent dip it is cut to +4 and -4",
          np.allclose(a3[0, 10], 3.0, atol=0.01) and np.allclose(a3[1, 10], -3.0, atol=0.01) and not a3[:, :10].any()
          and np.array_equal(a6[0, 10], [4.0, 4.0]) and np.array_equal(a6[1, 10], [-4.0, -4.0]) and np.abs(r6["X"][:, :, 4:6]).max() == 4.0
          and F.A_CLIP == 4.0, dip3=a3[:, 10], dip6=a6[:, 10])
    g, z, has, _ = SG.z_grid("own", T22.dip(6.0), None, NSEC)
    _, zd = SG.at_decisions(g, z, has, arr([s]))
    check("inputs 5 and 6 are research 22's adverse score -dir x z, cut to [-4, 4]",
          np.array_equal(r6["X"][0, :, 5], np.clip(SG.adverse(zd, arr([1]))[0], -4.0, 4.0)) and float(-zd[0, 10]) > 4.0, z=float(zd[0, 10]))
    early = flat()
    setq(early, 1004, NSEC, BID - 300, BID + SPR - 300)                      # a dip at 09:46:45: the window is not full yet
    re_ = F.raw(early, ref, NSEC, arr([1, -1]), arr([954, 954]), np.nan)
    rn = F.raw(T22.dip(3.0), None, NSEC, arr([1, -1]), arr([s, s]), np.nan)
    lk = T22.dip(3.0)
    setq(lk, U0, U0 + 1, BID - 300, BID - 300)                                # the stock quote is locked at the decision bar itself
    rl = F.raw(lk, ref, NSEC, arr([1, -1]), arr([s, s]), np.nan)
    rs = F.raw(T22.dip(3.0), flat(bid=3_000_000, spread=0), NSEC, arr([1, -1]), arr([s, s]), np.nan)
    check("no signal -> inputs 5 and 6 are 0: the window is not full (both), no SPY file or an SPY quote that is not valid (input 5 only, "
          "input 6 still works), a stock quote that is not valid for the signal at the decision bar (both, at that bar)",
          not re_["X"][:, :, 4:6].any() and re_["counts"]["signal_spy"] == 0 == re_["counts"]["signal_own"]
          and not rn["X"][:, :, 4].any() and rn["counts"]["signal_spy"] == 0 and abs(rn["X"][0, 10, 5] - 3.0) < 0.01
          and not rs["X"][:, :, 4].any() and abs(rs["X"][0, 10, 5] - 3.0) < 0.01
          and not rl["X"][:, 10, 4:6].any() and rl["X"][0, 11, 5] != 0.0 and bool(rl["valid"][:, 10].all()))
    b, ref2 = rand_day(41), rand_day(42, bid0=3_000_000, tick_p=0.5)
    b["bid"][7000:7003] = 0                                                   # three bars without a valid quote
    dirs, starts = E.parents("20190307", NSEC)
    r = F.raw(b, ref2, NSEC, dirs, starts, 1.0e6)
    f = E.features(b, dirs, starts)
    m18 = hand18([0.7, -0.1, 0.05, 0.4], 0.1)
    p_new, p_old = M.predict(M.from18(m18), r["X"], r["valid"]), E.predict(m18, f)
    check("inputs 1-4, y and the valid flag are engine18's values unchanged, and a four-input model scored by the new code gives engine18's "
          "probabilities exactly", np.array_equal(r["X"][..., :4], E.design(f, np.nan), equal_nan=True) and np.array_equal(r["y"], f["y"])
          and np.array_equal(r["valid"], f["valid"]) and np.array_equal(r["vp"], f["vp"], equal_nan=True)
          and np.array_equal(p_new, p_old, equal_nan=True) and r["X"].shape == (20, E.NG, 9), max_abs_diff=float(np.nanmax(np.abs(p_new - p_old))))
    check("the nine inputs in order: research 18's four, the two adverse scores, trade imbalance (30 s), trade intensity (60 s), their product",
          F.FEATS == ("imbalance", "spread_ticks", "vol_pre", "mid_change_5s", "a_spy", "a_own", "trade_imbalance_30s", "trade_intensity_60s",
                      "imbalance_x_intensity") and (F.W_IMB, F.W_INT, F.INT_CLIP) == (30, 60, 10.0) and (F.I_VP, F.I_IMB, F.I_INT, F.I_PROD) == (2, 6, 7, 8))


def future_tests():
    b, ref = rand_day(51), rand_day(52, bid0=3_000_000, tick_p=0.5)
    dirs, starts = E.parents("20190301", NSEC)
    cut = int(np.sort(starts)[10]) + 150
    b2, ref2 = {k: v.copy() for k, v in b.items()}, {k: v.copy() for k, v in ref.items()}
    for x in (b2, ref2):
        for k in ("bid", "bid_max", "bid_min"):
            x[k][cut + 1:] += 5000
        for k in ("ask", "ask_max", "ask_min"):
            x[k][cut + 1:] += 5200
        x["bsz"][cut + 1:], x["asz"][cut + 1:] = 100, 300
        x["vol_bid"][cut + 1:] = 0
        x["vol_ask"][cut + 1:] *= 3
        x["vol"][cut + 1:] = x["vol_bid"][cut + 1:] + x["vol_ask"][cut + 1:]
    fills = {"vol_pre": 5.0, "intensity": 1.0}
    X1 = F.design(F.raw(b, ref, NSEC, dirs, starts, 2.0e7)["X"], fills)
    X2 = F.design(F.raw(b2, ref2, NSEC, dirs, starts, 2.0e7)["X"], fills)
    U = starts[:, None] + E.STEP * np.arange(E.NG)[None, :]
    before = U <= cut
    same = [bool(np.array_equal(X1[..., j][before], X2[..., j][before])) for j in ALL]
    later = [bool(not np.array_equal(X1[..., j][~before], X2[..., j][~before])) for j in ALL]
    m = hand({"imbalance": 0.5, "mid_change_5s": 0.4, "a_spy": 0.3, "a_own": 0.3, "trade_imbalance_30s": 0.6, "trade_intensity_60s": 0.2, "imbalance_x_intensity": 0.3})
    ok = np.ones_like(before)
    p1, p2 = M.predict(m, F.raw(b, ref, NSEC, dirs, starts, 2.0e7)["X"], ok), M.predict(m, F.raw(b2, ref2, NSEC, dirs, starts, 2.0e7)["X"], ok)
    check("no look-ahead: changing every value after a bar leaves all nine inputs, and the probability, of the decision bars up to it "
          "unchanged (the inputs of later bars do change, except vol_pre of orders that started before it)",
          all(same) and all(later[j] for j in ALL if j != F.I_VP) and np.array_equal(p1[before], p2[before]) and not np.array_equal(p1[~before], p2[~before])
          and before.sum() > 200 and (~before).sum() > 200, same=same, later_changed=later, decision_bars_before=int(before.sum()))


# ---------------------------------------------------------------- model
def toy(seed=7, n=400):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, E.NG, 9))
    X[..., 1] = rng.integers(1, 11, (n, E.NG))
    vp = np.abs(rng.normal(6, 3, n))
    vp[rng.random(n) < 0.1] = np.nan
    X[..., F.I_VP] = vp[:, None]
    X[..., F.I_IMB] = np.clip(X[..., F.I_IMB], -1, 1)
    X[..., F.I_INT] = np.clip(np.abs(X[..., F.I_INT]) * 2, 0, 10)
    X[rng.random(n) < 0.15, :, F.I_INT] = np.nan
    X[..., F.I_PROD] = np.nan
    lin = 0.6 * X[..., 0] + 0.5 * X[..., 3] + 0.4 * X[..., 5] + 0.8 * X[..., F.I_IMB] - 0.3 * X[..., 4]
    y = (rng.random((n, E.NG)) < 1 / (1 + np.exp(-lin))).astype(np.int8)
    y[rng.random((n, E.NG)) < 0.1] = -1
    valid = rng.random((n, E.NG)) < 0.97
    return X, y, valid, vp


def model_tests(tmp):
    from sklearn.linear_model import LogisticRegression
    X, y, valid, vp = toy()
    m4 = M.fit(X, y, valid, vp, cols=[0, 1, 2, 3])
    # study18_forecast/dev18.py fit, line by line, on the four columns
    fill = float(np.median(vp[~np.isnan(vp)]))
    X4 = X[..., :4]
    keep = valid & (y >= 0)
    Xk = X4[keep].copy()
    Xk[:, 2] = np.where(np.isnan(Xk[:, 2]), fill, Xk[:, 2])
    mean, sd = Xk.mean(axis=0), Xk.std(axis=0)
    Z = (Xk - mean) / sd
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000).fit(Z, y[keep])
    check("training order: research 18's fit lines on four inputs and the new code with four inputs give the same fill, means, standard "
          "deviations, coefficients and intercept exactly", m4["vol_pre_fill"] == fill and m4["mean"] == mean.tolist() and m4["sd"] == sd.tolist()
          and m4["coef"] == clf.coef_[0].tolist() and m4["intercept"] == float(clf.intercept_[0]) and m4["n_train_points"] == int(keep.sum())
          and m4["n_train_parents"] == len(vp) and m4["features"] == list(E.FEATS) and m4["estimator"] == M.ESTIMATOR, coef=m4["coef"])
    m9 = M.fit(X, y, valid, vp)
    it = X[..., F.I_INT][keep]
    Xd = F.design(X, m9["fills"])
    check("nine-input fit: vol_pre fill = median over the training parents, intensity fill = median over the training points with a value, "
          "product after the fill, mean and sd of the training points, probability formula equal to scikit-learn's",
          m9["fills"]["vol_pre"] == fill and m9["fills"]["intensity"] == float(np.median(it[~np.isnan(it)])) and np.allclose(m9["mean"], Xd[keep].mean(axis=0), rtol=0, atol=1e-12)
          and np.allclose(m9["sd"], Xd[keep].std(axis=0), rtol=0, atol=1e-12) and m9["check_vs_predict_proba_max_abs"] < 1e-12 and len(m9["coef"]) == 9
          and m9["cols"] == ALL and m9["columns_without_variation"] == [] and m9["base_rate_train"] == float(y[keep].mean())
          and m9["coef"][0] > 0.3 and m9["coef"][3] > 0.3 and m9["coef"][4] < -0.1 and m9["coef"][F.I_IMB] > 0.2, coef=m9["coef"], fills=m9["fills"])
    p = M.predict(m9, X, valid)
    z = (Xd - np.asarray(m9["mean"])) / np.asarray(m9["sd"])
    want = 1 / (1 + np.exp(-(m9["intercept"] + z @ np.asarray(m9["coef"]))))
    check("probability: 1 / (1 + exp(-(intercept + sum coef x standardised input))) with the two fills; no value where the decision quote is "
          "not valid", np.array_equal(p[valid], want[valid]) and np.isnan(p[~valid]).all() and (~valid).sum() > 0 and 0 < np.nanmin(p) < np.nanmax(p) < 1)
    j = F.FEATS.index("a_own")
    mj = M.fit(X, y, valid, vp, cols=[c for c in ALL if c != j])
    X2 = X.copy()
    X2[..., j] += 100.0
    X3 = X.copy()
    X3[..., 0] += 1.0
    check("a model with one input left out has eight coefficients, does not read the left-out input and still reads the others",
          len(mj["coef"]) == 8 and F.FEATS[j] not in mj["features"] and np.array_equal(M.predict(mj, X, valid), M.predict(mj, X2, valid), equal_nan=True)
          and not np.array_equal(M.predict(mj, X, valid), M.predict(mj, X3, valid), equal_nan=True)
          and not np.array_equal(M.predict(m9, X, valid), M.predict(m9, X2, valid), equal_nan=True))
    Xc = X.copy()
    Xc[..., 4] = 0.0
    mc = M.fit(Xc, y, valid, vp)
    check("an input without variation in the training points gets sd 1 and is recorded (it then adds nothing)",
          mc["columns_without_variation"] == ["a_spy"] and mc["sd"][4] == 1.0 and abs(mc["coef"][4]) < 1e-9 and np.isfinite(mc["coef"]).all())
    path = os.path.join(tmp, "model", "model23.json")
    os.makedirs(os.path.dirname(path))
    h = M.save(m9, path)
    written = h == M.sha(path) == open(path.replace(".json", ".sha256")).read().split()[0] and open(path.replace(".json", ".sha256")).read().split()[1] == "model23.json"
    again = quiet(M.save, m9, path)
    back = M.load(path)
    open(path, "a").write("\n")
    changed = quiet(M.load, path)
    check("model file: written once with its sha256 next to it, a second write is refused, the stored numbers come back exactly, and a file "
          "whose content changed is not read", written and h == hashlib.sha256(open(path, "rb").read()[:-1]).hexdigest()
          and again is not None and "once" in again and back["coef"] == m9["coef"] and back["mean"] == m9["mean"] and back["fills"] == m9["fills"]
          and changed is not None and "differs" in changed, second_write=again, changed=changed)


# ---------------------------------------------------------------- engine with the probabilities
def engine_tests(tmp):
    s = U0 - 50
    adv = 390 * 10_000.0
    hm = hand({"trade_imbalance_30s": 5.0})                                   # p = sigmoid(5 x signed trade imbalance)
    names = ("R0", "R1_0.60", "R2_0.40", "R3_0.60_0.40", "ctrl:R1_0.60", "ctrl:R2_0.40")

    def day(spread):
        b = flat(spread=spread)
        b["vol_ask"][s - 20:s], b["vol"][s - 20:s] = 500, 500                 # buyer-initiated trades before the orders start
        return b
    col = lambda r, run, lr=3: r["outs"][E.LRULES.index(lr), names.index(run)]       # noqa: E731
    w = C.run_parents(day(SPR), None, NSEC, arr([1, -1]), arr([s, s]), adv, names, model=hm)
    p = w["P23"]
    r0, r1, r2, c1 = col(w, "R0"), col(w, "R1_0.60"), col(w, "R2_0.40"), col(w, "ctrl:R1_0.60")
    check("probability from the trade inputs: buyer-initiated volume gives a high p to the buy order and a low p to the sell order while it is "
          "inside the 30-second window, 0.5 afterwards", bool((p[0, :6] > 0.99).all() and (p[1, :6] < 0.01).all() and (p[:, 6:] == 0.5).all()), p_buy=p[0, :8], p_sell=p[1, :8])
    check("R1 (p >= p_hi): the buy order goes to market at the first decision bar at that bar's ask; the sell order and R2 stay with E* (wide "
          "spread: E* rests a limit order)", r1[0, 9] == E.QTY and r1[0, 5] == 0 and r1[0, 3] == E.QTY * (BID + SPR) and np.array_equal(r1[1, :9], r0[1, :9])
          and r1[1, 9] == 0 and np.array_equal(r2, r0) and r0[0, 5] == E.HOR and not r0[:, 9:11].any(), r1=r1[:, [2, 5, 9]], r0=r0[:, [2, 5]])
    check("control with 1 - p: the sell order is filled early at that bar's bid instead, the buy order stays with E*",
          c1[1, 9] == E.QTY and c1[1, 5] == 0 and c1[1, 3] == E.QTY * BID and np.array_equal(c1[0, :9], r0[0, :9]) and c1[0, 9] == 0
          and C.source("ctrl:R1_0.60") == "ctrl" and C.engine_name("ctrl:R1_0.60") == "R1_0.60", control=c1[:, [2, 5, 9]])
    n = C.run_parents(day(200), None, NSEC, arr([1, -1]), arr([s, s]), adv, names, model=hm)
    r0, r1, r2, r3, c2 = col(n, "R0"), col(n, "R1_0.60"), col(n, "R2_0.40"), col(n, "R3_0.60_0.40"), col(n, "ctrl:R2_0.40")
    check("R2 (p <= p_lo) with a 2-cent spread: E* takes the spread at once; the sell order (low p) is held for the six decision bars with the "
          "trades in the window and goes to market at the seventh; the buy order (high p) is not held; R1 on top of E*'s own market order is "
          "not counted", bool(r0[:, 5].max() == 0 and (r0[:, 2] == E.QTY).all() and r2[1, 10] == 6 and r2[1, 5] == 30 and r2[1, 2] == E.QTY
                              and r2[1, 3] == E.QTY * BID and np.array_equal(r2[0, :9], r0[0, :9]) and r2[0, 10] == 0 and r1[0, 9] == 0
                              and np.array_equal(r1[:, :9], r0[:, :9]) and np.array_equal(r3[1], r2[1])), r2=r2[:, [5, 10]], r0=r0[:, [2, 5]])
    check("control of R2 with 1 - p: the buy order is held instead and the sell order is not",
          bool(c2[0, 10] == 6 and c2[0, 5] == 30 and np.array_equal(c2[1, :9], r0[1, :9]) and c2[1, 10] == 0), control=c2[:, [5, 10]])
    l0 = n["outs"][E.LRULES.index(0), names.index("R2_0.40")]
    check("the same hold under the L0 fill rule (both fill rules are run)", bool(l0[1, 10] == 6 and l0[1, 5] == 30 and n["outs"].shape == (2, len(names), 2, E.NCOL)))
    # two sources of p side by side: research 18's model reads the displayed sizes, the hand model the trades
    b = flat()
    b["bsz"][:], b["asz"][:] = 9000, 1000
    m18 = hand18([5.0, 0.0, 0.0, 0.0])
    both = ("R0", "R1_0.60", "m18:R1_0.60")
    t = C.run_parents(b, None, NSEC, arr([1, -1]), arr([s, s]), adv, both, model=hm, m18=m18)
    o23, o18, o0 = (t["outs"][1, both.index(k)] for k in ("R1_0.60", "m18:R1_0.60", "R0"))
    check("two models side by side: the run named m18:<candidate> uses research 18's model (here: more bid size -> high p for the buy order -> "
          "early fill) and the candidate itself uses the nine-input model (no trades -> p 0.5 -> E*)",
          o18[0, 9] == E.QTY and o18[0, 5] == 0 and o18[1, 9] == 0 and np.array_equal(o23, o0) and abs(t["P18"][0, 0] - 1 / (1 + np.exp(-4.0))) < 1e-12
          and bool((t["P23"] == 0.5).all()) and C.source("m18:R1_0.60") == "m18" and C.source("R1_0.60") == "m23" and C.source("S2i") is None)
    try:
        C.run_parents(b, None, NSEC, arr([1]), arr([s]), adv, ("R1_0.60",))
        refused = False
    except ValueError:
        refused = True
    base = C.run_parents(b, None, NSEC, arr([1]), arr([s]), adv, C.BASE)
    bad = b.copy()
    bad = {k: v.copy() for k, v in b.items()}
    bad["bid"][s + 25] = 0                                                    # no valid quote at the sixth decision bar
    q = C.run_parents(bad, None, NSEC, arr([1]), arr([s]), adv, ("R0",), model=hm, m18=m18)
    check("a candidate without a model is an error, the base runs need none, and a decision bar without a valid quote gets no probability "
          "from either model", refused and base["outs"].shape == (2, 3, 1, E.NCOL) and base["P23"] is None and base["P18"] is None
          and np.isnan(q["P23"][0, 5]) and np.isnan(q["P18"][0, 5]) and not np.isnan(q["P23"][0, 4]) and not q["raw"]["valid"][0, 5])
    fr = C.firing(col(w, "R1_0.60"), "R1_0.60")
    check("firing shares use the engine name of a run (candidate, research 18 twin or control); base runs and C3 have none",
          fr["parents"] == 0.5 and fr["early_parents"] == 0.5 and C.firing(col(w, "ctrl:R1_0.60"), "ctrl:R1_0.60")["parents"] == 0.5
          and C.firing(col(n, "R2_0.40"), "R2_0.40")["holds_per_parent"] == 3.0 and C.firing(col(w, "R0"), "R0")["parents"] == 0.0
          and not C.acts("C3") and not C.acts("S0") and C.acts("m18:R3_0.75_0.40"), firing=fr)
    # an empty file (no quote and no trade all day) and the size correction, through the plan's loader
    root = os.path.join(tmp, "engine")
    empty = {c: np.zeros(NSEC, np.int64) for c in L22.COLS}
    te = {**task_of(root, save(root, "EE", "20260514", empty)), "adv": np.nan}
    spec = {"runs": ("R0", "R1_0.60", "m18:R1_0.60"), "delay_runs": ("R0",), "model": hm, "m18": m18, "grid": True}
    re_ = C.symday(te, spec)
    rd = rand_day(61)
    rows = {d: save(root, "AA", d, rd, mult) for d, mult in (("20251031", 1), ("20251103", 100))}
    ta = [{**task_of(root, rows[d]), "adv": 1.0e6} for d in rows]
    par, OUT, G, info = C.simulate(ta + [te], spec["runs"], spec["delay_runs"], 1, model=hm, m18=m18, grid=True)
    dd, ss = E.parents("20251103", NSEC)
    okq = E.quote_ok(rd)[ss]
    direct = C.run_parents(rd, None, NSEC, dd[okq], ss[okq], 1.0e6, spec["runs"], spec["delay_runs"], model=hm, m18=m18)
    with np.load(os.path.join(root, rows["20251103"]["file"])) as zf:
        stored = {c: zf[c].astype(np.int64) for c in L22.COLS}
    unread = C.run_parents(stored, None, NSEC, dd[okq], ss[okq], 1.0e6, spec["runs"], spec["delay_runs"], model=hm, m18=m18)
    sel = (par.date == "20251103").to_numpy()
    check("a file without any quote or trade gives no parent order and no error; a symbol-day of 2025-11-03 stored in 100 x shares is read "
          "through loader22 (sizes / 100) and equals the run on the bars in shares, which the unread sizes would not",
          len(re_["dir"]) == 0 and re_["outs"].shape == (2, 4, 0, E.NCOL) and re_["n_dropped"] == 20 and info["symbol_days_without_parents"] == 1
          and int(sel.sum()) == int(okq.sum()) and np.array_equal(OUT[:, :, sel], direct["outs"]) and np.array_equal(G["X"][sel], direct["raw"]["X"], equal_nan=True)
          and not np.array_equal(unread["outs"], direct["outs"]) and stored["bsz"].max() == 100 * rd["bsz"].max()
          and OUT.shape[1] == 4 and G["P23"].shape == G["P18"].shape == G["y"].shape, parents=int(len(par)), dropped=info["dropped_parents"])


# ---------------------------------------------------------------- rules
def rule_tests():
    def rows(mean=3.10, p95=20.0, fired=0.3, over=None):
        r = {c: {"mean": mean, "p95": p95, "fired": fired} for c in C.CANDS}
        for c, v in (over or {}).items():
            r[c] = {**r[c], **v}
        return r
    a1, _ = C.select_astar(rows(mean=3.10), 3.11)
    a2, _ = C.select_astar(rows(over={"R1_0.60": {"mean": 3.08}}), 3.10)
    a3, rec3 = C.select_astar(rows(over={"R1_0.60": {"mean": 3.0799}}), 3.10)
    check("A*: the best candidate must be lower than R0 by more than 0.02 cents (0.01 and exactly 0.02 give R0, 0.0201 gives the candidate)",
          (a1, a2, a3) == ("R0", "R0", "R1_0.60") and rec3["improves_in_development"], picks=[a1, a2, a3])
    r = rows(mean=3.10, over={"R2_0.30": {"mean": 3.000, "p95": 20.0, "fired": 0.3}, "R1_0.70": {"mean": 3.015, "p95": 18.0, "fired": 0.5},
                              "R3_0.60_0.40": {"mean": 3.019, "p95": 18.0, "fired": 0.2}, "R1_0.60": {"mean": 3.03, "p95": 10.0, "fired": 0.0}})
    a4, rec4 = C.select_astar(r, 3.20)
    check("A*: among the candidates within 0.02 cents of the lowest mean the lower p95 wins, then the lower share of fired parents; a "
          "candidate outside 0.02 cents does not count", a4 == "R3_0.60_0.40" and rec4["tied_within_0.02"] == ["R1_0.70", "R2_0.30", "R3_0.60_0.40"]
          and rec4["lowest_p95_among_tied"] == ["R1_0.70", "R3_0.60_0.40"], record=rec4)
    a5, _ = C.select_astar(rows(mean=3.00), 3.20)
    a6, rec6 = C.select_astar(rows(mean=3.10, over={"R1_0.60": {"mean": 3.00, "p95": 30.0}}), 3.015)
    check("A*: all equal -> the first of the grid order; R0 inside 0.02 cents of the best candidate is A* whatever the percentiles (R0 takes "
          "no part in the tie rule)", a5 == C.CANDS[0] == "R1_0.60" and a6 == "R0" and "tied_within_0.02" not in rec6, picks=[a5, a6])
    h1 = C.holm({"H1": 0.03})
    h2a, h2b, h2c = C.holm({"H1": 0.03, "H2": 0.04}), C.holm({"H1": 0.001, "H2": 0.04}), C.holm({"H1": 0.001, "H2": 0.06})
    check("Holm: one hypothesis at 0.05; two at 0.025 then 0.05, stopping at the first failure",
          h1["H1"]["rejected"] and h1["H1"]["level"] == 0.05 and not h2a["H1"]["rejected"] and not h2a["H2"]["rejected"]
          and h2b["H1"]["rejected"] and h2b["H2"]["rejected"] and h2b["H1"]["level"] == 0.025 and h2b["H2"]["level"] == 0.05
          and h2c["H1"]["rejected"] and not h2c["H2"]["rejected"], one=h1, two=[h2a, h2b, h2c])
    check("candidate grid: research 18's 19 candidates in its order (R1 x 4, R2 x 3, R3 x 12) with p_hi 0.60-0.75 and p_lo 0.30-0.40; R0 is "
          "not a candidate", len(C.CANDS) == 19 and C.CANDS[:8] == ("R1_0.60", "R1_0.65", "R1_0.70", "R1_0.75", "R2_0.30", "R2_0.35", "R2_0.40", "R3_0.60_0.30")
          and C.CANDS[-1] == "R3_0.75_0.40" and "R0" not in C.CANDS and E.P_HI == (0.60, 0.65, 0.70, 0.75) and E.P_LO == (0.30, 0.35, 0.40)
          and E.params("R3_0.65_0.35")[4:] == (0.65, 0.35) and DV.REPORT == ("S0", "S2i", "C3", "R0") + C.CANDS and len(DV.REF18) == 19)


# ---------------------------------------------------------------- statistics
def auc_direct(p, y):
    from scipy.stats import rankdata
    r = rankdata(p)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stats_tests():
    idx = E.boot_index(250, seed=C.SEED)
    check("bootstrap: 10,000 draws of circular 21-day blocks with seed 23 (differs from seeds 18 and 22, repeats itself)",
          idx.shape == (10_000, 250) and not np.array_equal(idx, E.boot_index(250)) and not np.array_equal(idx, E.boot_index(250, seed=22))
          and np.array_equal(idx, E.boot_index(250, seed=23)) and C.SEED == 23)
    rng = np.random.default_rng(5)
    days = [f"2019{m:02d}{d:02d}" for m in (3, 4, 5) for d in range(1, 21)]
    n = 600
    par = pd.DataFrame({"date": np.repeat(days, n // len(days)), "sym": np.tile(["AA", "BB"], n // 2)})
    lin = rng.normal(size=(n, E.NG))
    y = (rng.random((n, E.NG)) < 1 / (1 + np.exp(-1.2 * lin))).astype(np.int8)
    y[rng.random((n, E.NG)) < 0.12] = -1
    valid = rng.random((n, E.NG)) < 0.95
    p23 = np.round(1 / (1 + np.exp(-lin)), 2)                                 # rounded: ties
    p18 = np.round(1 / (1 + np.exp(-(0.3 * lin + rng.normal(size=lin.shape)))), 2)
    G = {"y": y, "valid": valid, "P23": np.where(valid, p23, np.nan), "P18": np.where(valid, p18, np.nan)}
    mask = (par.sym == "AA").to_numpy()
    pr, daily = C.prediction(par, G, 0.48, 0.52, {"aa": mask, "none": np.zeros(n, bool)})
    keep = valid & (y >= 0)
    a23, a18 = auc_direct(p23[keep], y[keep]), auc_direct(p18[keep], y[keep])
    c = pr["combined"]
    br = float(np.mean((p23[keep] - y[keep]) ** 2))
    check("prediction metrics on the scored points (decision quote valid, y defined) equal direct formulas: AUC with ties 1/2 (rank formula), "
          "hit rate at 0.5, Brier score, Brier skill score against the constant training base rate; both models on the same points",
          abs(c["auc"] - a23) < 1e-12 and abs(pr["study18_forecast"]["auc"] - a18) < 1e-12 and abs(pr["auc_diff"]["mean"] - (a23 - a18)) < 1e-12
          and abs(c["accuracy_0.5"] - np.mean((p23[keep] >= 0.5) == (y[keep] == 1))) < 1e-15 and abs(c["brier"] - br) < 1e-15
          and abs(c["bss"] - (1 - br / np.mean((0.48 - y[keep]) ** 2))) < 1e-12 and abs(pr["study18_forecast"]["brier_ref_train_base_rate"] - np.mean((0.52 - y[keep]) ** 2)) < 1e-15
          and pr["counts"] == {"grid_points": n * E.NG, "valid_points": int(valid.sum()), "y_missing_points": int((valid & (y < 0)).sum()),
                               "y_missing_share_of_valid": float((valid & (y < 0)).sum() / valid.sum()), "scored_points": int(keep.sum()), "days": len(days)}
          and c["n"] == int(keep.sum()), auc=[c["auc"], pr["study18_forecast"]["auc"]])
    km = keep & mask[:, None]
    check("bootstrap of the AUC on resampled days: the interval holds the estimate, the share of draws at or below 0.5 is 0 for an informative "
          "model and 1 for 1 - p; the AUC difference has its own interval from the same draws; a subset of parents has the direct AUC of its points",
          c["auc_boot"]["ci_lo"] < c["auc"] < c["auc_boot"]["ci_hi"] and c["auc_boot"]["share_le_0.5"] == 0.0 and c["auc_boot"]["se"] > 0
          and pr["auc_diff"]["ci_lo"] < pr["auc_diff"]["mean"] < pr["auc_diff"]["ci_hi"] and pr["auc_diff"]["ci_lo"] > 0 and pr["auc_diff"]["share_le_0"] == 0.0
          and C.prediction(par, {**G, "P23": 1 - G["P23"]}, 0.5, 0.5)[0]["combined"]["auc_boot"]["share_le_0.5"] == 1.0
          and abs(pr["subsets"]["aa"]["auc_combined"] - auc_direct(p23[km], y[km])) < 1e-12 and pr["subsets"]["aa"]["n"] == int(km.sum())
          and pr["subsets"]["none"] == {"n": 0, "auc_combined": None, "auc_research18": None} and pr["boot"]["seed"] == 23, boot=c["auc_boot"], diff=pr["auc_diff"])
    d0 = days[7]
    k0 = keep & (par.date == d0).to_numpy()[:, None]
    row = daily[daily.date == d0].iloc[0]
    yr = pr["by_year"]["2019"]
    check("daily prediction file: per day the scored points, the sum of y and, per model, the sums of p, of the squared error and of the hits; "
          "the AUC by year; the calibration deciles hold every point once",
          len(daily) == len(days) and row.n_scored == k0.sum() and row.y_sum == y[k0].sum() and abs(row.p_combined_sum - p23[k0].sum()) < 1e-9
          and abs(row.sq_err_research18_sum - ((p18[k0] - y[k0]) ** 2).sum()) < 1e-9 and row.hit_combined == ((p23[k0] >= 0.5) == (y[k0] == 1)).sum()
          and daily.n_scored.sum() == keep.sum() and abs(yr["auc_combined"] - a23) < 1e-12 and yr["n"] == int(keep.sum())
          and sum(d["n"] for d in pr["deciles_combined"]) == int(keep.sum()))
    check("the AUC of a set of points without both classes is not defined", C.auc_point(np.array([0.2, 0.7]), np.array([1, 1], dtype=np.int8)) is None
          and C.auc_point(np.array([]), np.array([], dtype=np.int8)) is None and C.auc_point(np.array([0.2, 0.7, 0.7]), np.array([0, 1, 0], dtype=np.int8)) == 0.75)


# ---------------------------------------------------------------- the synthetic world
DEV_DAYS = {2017: ["20170301", "20170302", "20170303", "20170306", "20170307", "20170308", "20170309", "20170310"],
            2018: ["20181112", "20181113", "20181114", "20181115", "20181116", "20181119", "20181120", "20181121"],
            2019: ["20190301", "20190304", "20190305", "20190306", "20190307", "20190308", "20190311", "20190312"],
            2020: ["20201118", "20201119", "20201120", "20201123", "20201124", "20201125", "20201127", "20201130"]}
SEALED_DAYS = ["20210104", "20210105", "20210106", "20210107", "20210108", "20210111", "20251029", "20251030", "20251031", "20251103",
               "20251104", "20251105"]
HALF_DAYS = ("20201127",)
STANDIN18 = {"features": list(E.FEATS), "mean": [0.0, 3.0, 5.0, 0.0], "sd": [0.5, 2.0, 3.0, 1.5], "coef": [0.2, 0.0, 0.0, 0.8], "intercept": 0.0,
             "vol_pre_fill": 5.0, "base_rate_train": 0.5}


def trend_day(seed, bid0=BID):
    """A day whose mid moves in runs of one direction, with more buyer-initiated volume in up runs: the next mid change can be
    predicted from the last one and from the trades. Spread constant inside 10-minute blocks."""
    rng = np.random.default_rng(seed)
    d = np.where(np.cumsum(rng.random(NSEC) < 0.01) % 2 == 0, 1, -1) * (1 if rng.random() < 0.5 else -1)
    mid = bid0 + 100 * np.cumsum(d * (rng.random(NSEC) < 0.25))
    spr = 100 * np.repeat(rng.choice((1, 2, 3, 4, 6), NSEC // 600), 600)
    b = {c: np.zeros(NSEC, np.int64) for c in L22.COLS}
    b["bid"] = (mid - spr // 2) // 100 * 100
    b["ask"] = b["bid"] + spr
    b["bid_max"] = b["bid"] + 100 * (rng.random(NSEC) < 0.2)
    b["bid_min"] = b["bid"] - 100 * (rng.random(NSEC) < 0.2)
    b["ask_max"] = b["ask"] + 100 * (rng.random(NSEC) < 0.2)
    b["ask_min"] = b["ask"] - 100 * (rng.random(NSEC) < 0.2)
    b["bsz"], b["asz"] = rng.integers(1, 30, NSEC) * 100, rng.integers(1, 30, NSEC) * 100
    b["vol_ask"] = rng.integers(0, 8, NSEC) * 100 * (d > 0) + rng.integers(0, 3, NSEC) * 100
    b["vol_bid"] = rng.integers(0, 8, NSEC) * 100 * (d < 0) + rng.integers(0, 3, NSEC) * 100
    b["vol_bid_odd"], b["vol_ask_odd"] = (rng.random(NSEC) < 0.2) * 37 * (b["vol_bid"] > 0), (rng.random(NSEC) < 0.2) * 41 * (b["vol_ask"] > 0)
    b["px_min"], b["px_max"] = b["bid"] - 100, b["ask"] + 100
    b["vol"] = b["vol_bid"] + b["vol_ask"]
    return b


def build_world(tmp):
    """Folders: tmp/study23_combined_forecast (the stand-in repository), tmp/study21_more_pairs (reference manifests), tmp/data/... (bars)."""
    repo, data = os.path.join(tmp, "study23_combined_forecast"), os.path.join(tmp, "data")
    os.makedirs(os.path.join(repo, "results"))
    seed, src, dev_ref = 500, {}, []
    for y, days in DEV_DAYS.items():
        root = os.path.join(data, f"dev_{y}")
        rows = []
        for d in days:
            for s in ("DA", "DB"):
                seed += 1
                rows.append(save(root, s, d, trend_day(seed, bid0=BID + 30_000 * (s == "DB"))))
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
            rows.append(save(sroot, s, d, trend_day(seed, bid0=BID + 20_000 * (s == "SB")), mult))
        seed += 1
        ref.append(save(rroot, "SPY", d, rand_day(seed, bid0=3_000_000, tick_p=0.5), mult))
    seed += 1
    rows.append(save(sroot, "SA", "20210112", trend_day(seed)))                # a day that is not in the calendar
    tsv(os.path.join(repo, "SEALED_MANIFEST23.tsv"), rows)
    tsv(os.path.join(tmp, "study21_more_pairs", "SEALED_MANIFEST21.tsv"), ref)
    tsv(os.path.join(repo, "EXCLUDE23.tsv"), [("SB", "20210105", "2021/SB_20210105.npz", "test")], cols=("sym", "date", "file", "reason"))
    pd.DataFrame({"date": [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in SEALED_DAYS], "close_time": "16:00"}).to_csv(os.path.join(repo, "calendar23.csv"), index=False)
    for y in range(2021, 2027):
        tsv(os.path.join(repo, f"universe23_{y}.tsv"), [(y, 1, "SA"), (y, 2, "SB")], cols=("year", "slot", "sym"))
    for doc in RT.DOCS:
        open(os.path.join(repo, f"{doc}.md"), "w").write(f"{doc} stand-in\n")
        open(os.path.join(repo, f"{doc}.sha256"), "w").write(f"{C.sha(os.path.join(repo, doc + '.md'))}  {doc}.md\n")
    res, model = os.path.join(repo, "results"), os.path.join(repo, "model23.json")
    pairs = [(L15, "SOURCES", src), (L15, "YEARS", range(2017, 2021)), (L22, "DEV_SPY_ROOT", os.path.join(data, "ref_dev")),
             (L22, "DEV_SPY_MANIFEST", os.path.join(tmp, "study21_more_pairs", "DATA_MANIFEST21.tsv")), (C, "model18", lambda: dict(STANDIN18)),
             (DV, "RES", res), (DV, "LOCAL", os.path.join(data, "dev_results")), (DV, "CALENDAR", os.path.join(repo, "calendar23.csv")),
             (DV, "MODEL", model), (DV, "CHECK18", False), (RT, "HERE", repo), (RT, "RES", res), (RT, "UNLOCK", os.path.join(repo, "UNLOCK23.txt")),
             (RT, "FREEZE", os.path.join(repo, "FREEZE23.md")), (RT, "DEV_SUMMARY", os.path.join(res, "dev23_summary.json")), (RT, "MODEL", model),
             (RT, "SEALED_ROOT", os.path.join(data, "sealed")), (RT, "SEALED_BARS", sroot), (RT, "SEALED_MANIFEST", os.path.join(repo, "SEALED_MANIFEST23.tsv")),
             (RT, "EXCLUDE", os.path.join(repo, "EXCLUDE23.tsv")), (RT, "CALENDAR", os.path.join(repo, "calendar23.csv")),
             (RT, "UNIVERSE", os.path.join(repo, "universe23_{y}.tsv")), (RT, "REF_SEALED_ROOT", rroot),
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


def load_json(path):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def world_round(repo, res, label, forced):
    """Fit -> validation run -> rehearsal -> guard -> sealed run in the synthetic world; forced: an A* to impose (None: as it falls)."""
    pre = f"world ({label}): "
    side = RT.MODEL.replace(".json", ".sha256")
    n_tr, n_va = 2 * sum(len(DEV_DAYS[y]) for y in (2017, 2018)), 2 * sum(len(DEV_DAYS[y]) for y in (2019, 2020))
    # ---- fit
    err = quiet(DV.fit_stage, 1)
    fit = load_json(os.path.join(res, DV.FIT_FILE))
    md = fit.get("model", {})
    ok = (err is None and os.path.exists(RT.MODEL) and os.path.exists(side) and fit["model_sha256"] == M.sha(RT.MODEL) == open(side).read().split()[0]
          and md["features"] == list(F.FEATS) and len(md["coef"]) == 9 and md["train_years"] == [2017, 2018] and fit["sim"]["symbol_days"] == n_tr
          and fit["sim"]["symbol_days_without_ref"] == 0 and fit["sim"]["days_without_ref"] == [] and fit["sim"]["years"] == list(DV.TRAIN)
          and md["n_train_parents"] == fit["sim"]["parents"] and 0 < md["n_train_points"] <= md["n_train_parents"] * E.NG
          and set(fit["leave_one_out_models"]) == set(F.FEATS) and all(len(v["coef"]) == 8 and k not in v["features"] for k, v in fit["leave_one_out_models"].items())
          and fit["train"]["auc_combined"] > 0.6 and fit["sim"]["runs"] == [] and fit["sim"]["adv"]["adv_undefined"] == 2 * 2 * 5
          and 0 < fit["train"]["intensity_undefined_share_of_scored"] < 1 and M.load(RT.MODEL)["coef"] == md["coef"]
          and fit["regression_four_inputs"]["targets"] == "none (synthetic data)" and md["coef"][3] > 0 and md["coef"][F.I_IMB] > 0)
    check(pre + "the fit stage reads the training years only, simulates no order, fits the nine-input model once, writes model23.json with its "
          "sha256 and the summary with the nine left-one-out models; the model predicts the synthetic moves", ok, error=err, coef=md.get("coef"),
          train=fit.get("train"))
    err2 = quiet(DV.fit_stage, 1)
    check(pre + "a second fit is refused (the model exists)", err2 is not None and "once" in err2, message=err2)
    # ---- validation run
    err = quiet(DV.valid_stage, 1)
    dev = load_json(RT.DEV_SUMMARY)
    daily = pd.read_csv(os.path.join(res, "dev23_daily.csv"), dtype={"date": str}) if dev else pd.DataFrame()
    dpr = pd.read_csv(os.path.join(res, "dev23_daily_pred.csv"), dtype={"date": str}) if dev else pd.DataFrame()
    n_days = sum(len(DEV_DAYS[y]) for y in (2019, 2020))
    runs = DV.REPORT + DV.REF18
    ok = (err is None and all(os.path.exists(os.path.join(res, f)) for f in DV.FILES) and dev["sim"]["symbol_days"] == n_va
          and dev["sim"]["symbol_days_without_ref"] == 2 and dev["sim"]["days_without_ref"] == ["20190306"] and dev["sim"]["first_day"] == "20190301"
          and dev["years"] == [2019, 2020] and dev["sim"]["plan_symbol_days_all_years"] == n_tr + n_va
          and set(dev["tables"]["L3D1"]) == set(runs) == set(dev["tables"]["L0D0"]) and len(daily) == n_days * 2 * (len(runs) + (1 if dev["h2_exists"] else 0))
          and len(dpr) == n_days and dev["model_sha256"] == M.sha(RT.MODEL) and dev["model"]["coef"] == md["coef"]
          and dev["prediction"]["combined"]["auc"] > 0.6 and dev["prediction"]["counts"]["scored_points"] == int(dpr.n_scored.sum())
          and set(dev["leave_one_out"]) == set(F.FEATS) and all(abs(v["auc_drop"] - (dev["prediction"]["combined"]["auc"] - v["auc"])) < 1e-15 for v in dev["leave_one_out"].values())
          and dev["tables"]["L3D1"]["R3_0.60_0.40"]["firing"]["parents"] > 0 and dev["tables"]["L3D1"]["m18:R3_0.60_0.40"]["firing"]["parents"] > 0
          and dev["tables"]["L3D1"]["R0"]["firing"]["parents"] == 0.0 and dev["tables"]["L3D1"]["R0"]["minus_R0_mean"] == 0.0
          and dev["h2_exists"] == (dev["astar"] != "R0") and (dev["astar"] == "R0" or dev["astar"] in C.CANDS) and dev["trials_cumulative"] == 846
          and (forced is None or dev["astar"] == forced) and (dev["delay_check"] is not None) == dev["h2_exists"] == (dev["control"] is not None)
          and (not dev["h2_exists"] or dev["control"]["L3D1"]["control"] == C.CTRL + dev["astar"]) and (not dev["h2_exists"] or dev["checks"]["second_pass_equal"] is True)
          and ("H2" in dev["power"]) == dev["h2_exists"] == ("reference_lowest_candidate" not in dev["power"]) and dev["power"]["days_new"] == len(SEALED_DAYS)
          and dev["checks"]["daily_file_max_gap_c"] < 1e-6 and dev["research18_candidates"]["astar_research18"] == "R0"
          and dev["hypotheses"] == (["H1", "H2"] if dev["h2_exists"] else ["H1"]) and dev["sim"]["adv"]["adv_undefined"] == 2 * 2 * 5)
    check(pre + "the validation run reads the validation years with the frozen model: S0, S2i, R0, C3, the 19 candidates and the same 19 rules "
          "with research 18's model under two fill rules; both models' AUC on the same points and the nine AUCs with one input left out; the "
          "selection, the power, the control and the delay check when A* is a candidate; the two daily files", ok, error=err, astar=dev.get("astar"),
          h2=dev.get("h2_exists"), selection=dev.get("selection"), auc=[dev.get("prediction", {}).get(k, {}).get("auc") for k in ("combined", "study18_forecast")])
    err2 = quiet(DV.valid_stage, 1)
    check(pre + "a second validation run is refused (results exist)", err2 is not None and "once" in err2, message=err2)
    # ---- rehearsal
    err = quiet(RT.main, "rehearsal", 1)
    reh = load_json(os.path.join(res, "rehearsal23_summary.json"))
    check(pre + "the rehearsal (the sealed code path on the validation data) equals the validation numbers: tables and prediction metrics with "
          "max difference 0, no sealed path read", err is None and reh.get("checks", {}).get("pass") is True and reh["checks"]["max_abs_diff"] == 0.0
          and reh["checks"]["table_numbers_compared"] > 100 and reh["checks"]["prediction_numbers_compared"] > 40 and reh["astar"] == dev.get("astar")
          and reh["trials_added"] == 0 and reh["sim"]["symbol_days"] == n_va and reh["sim"]["symbol_days_without_ref"] == 2
          and reh["sim"]["days_without_ref"] == ["20190306"], error=err, checks=reh.get("checks"))
    # ---- guard
    g = {"no FREEZE, no unlock": RT.guard()}
    write_freeze(repo)
    g["no unlock file"] = RT.guard()
    open(RT.UNLOCK, "w", encoding="utf-8").write("연구23 테스트를 연다\n")
    g["wrong phrase"] = RT.guard()
    open(RT.UNLOCK, "w", encoding="utf-8").write(RT.PHRASE + "\n")
    g["all in place"] = RT.guard()
    open(os.path.join(repo, "SPEC23_ANSWERS.md"), "a").write("x")
    g["document changed"] = RT.guard()
    open(os.path.join(repo, "SPEC23_ANSWERS.md"), "w").write("SPEC23_ANSWERS stand-in\n")
    core = os.path.join(repo, "feat23.py")
    keep = open(core).read()
    open(core, "w").write(keep + "#\n")
    g["frozen file changed"] = RT.guard()
    open(core, "w").write(keep)
    write_freeze(repo, drop="model23.py")
    g["core file not listed"] = RT.guard()
    write_freeze(repo)
    g["all in place again"] = RT.guard()
    ok = (g["no unlock file"] == ["no unlock phrase in UNLOCK23.txt"] == g["wrong phrase"] and g["all in place"] == [] == g["all in place again"]
          and "FREEZE23.md missing or without the core files" in g["no FREEZE, no unlock"]
          and g["document changed"] == ["SPEC23_ANSWERS.md differs from SPEC23_ANSWERS.sha256"]
          and g["frozen file changed"] == ["frozen file differs: feat23.py"]
          and g["core file not listed"] == ["FREEZE23.md missing or without the core files"])
    check(pre + "guard of the sealed run: refuses without the unlock file, with another phrase, with a changed document, a changed frozen "
          "file or a core file missing from FREEZE23.md; passes only when everything is in place", ok, guard=g)
    # ---- the model file is pinned: any change of its content stops the guard, also when its own sha256 file is rewritten to match
    mtxt, stxt = open(RT.MODEL).read(), open(side).read()
    m0 = json.loads(mtxt)
    gm = {}
    open(RT.MODEL, "w").write(mtxt + "\n")
    gm["one newline appended"] = RT.guard()
    m1 = {**m0, "coef": [m0["coef"][0] + 1e-9] + m0["coef"][1:]}
    open(RT.MODEL, "w").write(json.dumps(m1, indent=1))
    gm["one coefficient changed"] = RT.guard()
    open(side, "w").write(f"{M.sha(RT.MODEL)}  model23.json\n")
    gm["coefficient changed and model23.sha256 rewritten to match"] = RT.guard()
    refused = quiet(RT.main, "sealed", 1)
    nothing = not any(os.path.exists(os.path.join(res, f)) for f in RT.MODES["sealed"]["files"])
    open(RT.MODEL, "w").write(mtxt)
    open(side, "w").write(stxt)
    back = RT.guard()
    check(pre + "a model file whose content alone changes (one newline, one coefficient, also with its own sha256 file rewritten to match) stops "
          "the guard with 'frozen file differs: model23.json', the sealed run writes nothing, and the original content passes again",
          all(v[:1] == ["frozen file differs: model23.json"] for v in gm.values())
          and "model23.json missing or differs from model23.sha256" in gm["one coefficient changed"]
          and "model23.json missing or differs from model23.sha256" not in gm["coefficient changed and model23.sha256 rewritten to match"]
          and any("another model file" in x for x in gm["coefficient changed and model23.sha256 rewritten to match"])
          and refused is not None and refused.startswith("refused") and nothing and back == [], guard=gm, sealed_run=refused)
    # ---- the validation summary is pinned
    txt = open(RT.DEV_SUMMARY, encoding="utf-8").read()
    d0 = json.loads(txt)
    other = next(c for c in C.CANDS if c != d0["astar"])
    cases = {"another A*": {**d0, "astar": other, "h2_exists": True}, "H2 flipped": {**d0, "h2_exists": not d0["h2_exists"]},
             "one AUC digit changed": {**d0, "prediction": {**d0["prediction"], "combined": {**d0["prediction"]["combined"], "auc": d0["prediction"]["combined"]["auc"] + 1e-6}}}}
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
    check(pre + "a validation summary whose content alone changes (another A*, H2 flipped, one digit of an AUC, one newline) stops the guard "
          "with 'frozen file differs', the sealed run writes nothing, and the original content passes again",
          all(v[:1] == ["frozen file differs: results/dev23_summary.json"] for v in res_g.values()) and usable["another A*"] and not usable["H2 flipped"]
          and any("no usable development selection" in x for x in res_g["H2 flipped"]) and refused is not None and refused.startswith("refused")
          and nothing and back == [], guard=res_g, readable_by_selection=usable, sealed_run=refused)
    # ---- the sealed run
    err = quiet(RT.main, "sealed", 1)
    t = load_json(os.path.join(res, "test23_summary.json"))
    td = pd.read_csv(os.path.join(res, "test23_daily.csv"), dtype={"date": str}) if t else pd.DataFrame(columns=["date", "run"])
    tp = pd.read_csv(os.path.join(res, "test23_daily_pred.csv"), dtype={"date": str}) if t else pd.DataFrame(columns=["date"])
    h2 = dev.get("h2_exists")
    want_sd = 2 * len(SEALED_DAYS) - 1                                         # SA and SB on the calendar days, minus the excluded SB day
    hyp = ["H1", "H2"] if h2 else ["H1"]
    ok = (err is None and t.get("mode") == "sealed" and t["sim"]["symbol_days"] == want_sd and t["sim"]["excluded_rows"] == 1
          and t["hypotheses"] == hyp and sorted(t["verdict"][h]["level"] for h in hyp) == ([0.025, 0.05] if h2 else [0.05])
          and (t["verdict"]["H2"]["result"] == "없음") == (not h2) and t["trials_added"] == len(hyp) and t["trials_before"] == 846
          and t["verdict"]["H1"]["sentence"] in RT.SENTENCES["H1"].values() and t["verdict"]["H1"]["auc"] == t["prediction"]["combined"]["auc"]
          and t["verdict"]["H1"]["p"] == t["prediction"]["combined"]["auc_boot"]["share_le_0.5"] and t["verdict"]["H1"]["confirmed"] is True
          and (not h2 or (t["verdict"]["H2"]["p"] == t["differences"]["L3D1"]["Astar_minus_Estar"]["p_one_sided"] and t["verdict"]["H2"]["sentence"] in RT.SENTENCES["H2"].values()))
          and {"auc_diff", "study18_forecast", "by_year", "subsets"} <= set(t["prediction"]) and set(t["prediction"]["subsets"]) == {"common_only", "before_cut", "from_cut"}
          and t["prediction"]["subsets"]["before_cut"]["n"] + t["prediction"]["subsets"]["from_cut"]["n"] == t["prediction"]["counts"]["scored_points"]
          and 0 < t["prediction"]["subsets"]["common_only"]["n"] < t["prediction"]["counts"]["scored_points"]
          and t["date_split"]["L3D1"]["Estar_minus_S0"]["before"]["days"] == 9 and t["date_split"]["L3D1"]["Estar_minus_S0"]["from"]["days"] == 3
          and t["counts"]["parents_before_cut"] + t["counts"]["parents_from_cut"] == t["counts"]["parents"]
          and t["common_only"]["excluded_symbol_years"] == ["2021 SB"] and 0 < t["common_only"]["parents"] < t["counts"]["parents"]
          and set(td.date) == set(SEALED_DAYS) == set(tp.date) and set(td.run) == {"S0", "S2i", "R0", "C3"} | ({t["astar"], C.CTRL + t["astar"]} if h2 else set())
          and {"Estar_minus_S0", "C3_minus_Estar", "S2i_minus_Estar"} <= set(t["differences"]["L3D1"]) and set(t["differences"]["L0D0"]) == set(t["differences"]["L3D1"])
          and ({"Astar_minus_Estar", "Astar_minus_C3", "Astar_minus_S2i"} <= set(t["differences"]["L3D1"])) == bool(h2)
          and (t["control"] is not None) == bool(h2) and (not h2 or {"control_minus_Estar", "Astar_minus_control"} <= set(t["control"]["L3D1"]))
          and ("R0_delay_minus_R0" in t["delay_check"]["L3D1"]) and ((f"{t['astar']}_delay_minus_R0_delay" in t["delay_check"]["L3D1"]) == bool(h2))
          and t["read"]["bars_root"].endswith("bars_1s_new_stocks") and t["model_sha256"] == M.sha(RT.MODEL) and t["sim"]["adv"]["adv_undefined"] > 0)
    check(pre + "the sealed run with the unlock phrase: universe symbols on calendar days minus the exclude list, H1 (AUC > 0.5)" + (" and H2" if h2 else " alone")
          + " with Holm, the AUC difference to research 18's model, E* - S0, C3 - E*, S2i - E*, the control when A* is a candidate, the split "
          "at 2025-11-03, common stocks only, the delay check, four files", ok, error=err, verdict=t.get("verdict"), counts=t.get("counts"))
    again = quiet(RT.main, "sealed", 1)
    check(pre + "a second sealed run is refused (test results already exist)", again is not None and "test results already exist" in again, message=again)
    return dev


def world_tests(tmp):
    repo, res, pairs = build_world(tmp)
    with patched(pairs):
        dev = world_round(repo, res, "selection as it falls", None)
        # second round: the same world with a forced A* so that both outcomes of the selection run end to end
        os.rename(res, os.path.join(repo, "results_round1"))
        os.makedirs(res)
        for f in ("FREEZE23.md", "UNLOCK23.txt", "model23.json", "model23.sha256"):
            os.rename(os.path.join(repo, f), os.path.join(repo, f + ".round1"))
        os.rename(DV.LOCAL, DV.LOCAL + "_round1")
        force = "R0" if dev.get("h2_exists") else "R3_0.60_0.40"
        if force == "R0":
            fake = lambda rows, r0: ("R0", {"lowest_mean": "R1_0.60", "lowest_mean_c": rows["R1_0.60"]["mean"], "r0_mean_c": r0,          # noqa: E731
                                            "r0_minus_lowest_c": r0 - rows["R1_0.60"]["mean"], "threshold_c": C.TIE, "improves_in_development": False, "astar": "R0"})
        else:
            fake = lambda rows, r0: (force, {"lowest_mean": force, "lowest_mean_c": rows[force]["mean"], "r0_mean_c": r0,   # noqa: E731
                                             "r0_minus_lowest_c": r0 - rows[force]["mean"], "threshold_c": C.TIE, "improves_in_development": True,
                                             "tied_within_0.02": [force], "lowest_p95_among_tied": [force], "fired": {force: rows[force]["fired"]}, "astar": force})
        with patched([(C, "select_astar", fake)]):
            dev2 = world_round(repo, res, "forced A* " + force, force)
        same_model = open(os.path.join(repo, "model23.json")).read().split('"written"')[0] == open(os.path.join(repo, "model23.json.round1")).read().split('"written"')[0]
    check("world: the two rounds cover both outcomes of the selection (A* = R0 without H2, and a candidate with H2), and the second fit on the "
          "same data gives the same coefficients", {dev.get("h2_exists"), dev2.get("h2_exists")} == {True, False} and same_model,
          round1=dev.get("astar"), round2=dev2.get("astar"))


def repo_guard_test():
    why = RT.guard()
    check("the real sealed run is refused now (no UNLOCK23.txt; Claude never writes it)", "no unlock phrase in UNLOCK23.txt" in why
          and not os.path.exists(os.path.join(HERE, "UNLOCK23.txt")), reasons=why)
    check("constants: seed 23, 21-day blocks, 10,000 draws, tie 0.02 cents, 20 parents of 1,000 shares over 300 s, market fee 0.3 cents, L3D1 "
          "primary, training 2015-2018, validation 2019-2020, N = 20, trials 826 -> 846, the unlock phrase, the sealed window and the fixed sentences",
          C.SEED == 23 and E.BOOT_BLOCK == 21 and E.BOOT_B == 10_000 and C.TIE == 0.02 and E.N_PARENT == 20 and E.QTY == 1000 and E.HOR == 300
          and E.FEE_MKT == 0.3 and RT.PHRASE == "연구23 테스트를 한 번 연다" and RT.WINDOW == ("20210104", "20261110") and L22.SIZE_FROM == "20251103" == C.CUT
          and DV.N_NEW == 20 and E.VARIANTS[0] == ("L3D1", 3, 1) and DV.TRAIN == (2015, 2016, 2017, 2018) and DV.VALID == (2019, 2020) == RT.VALID
          and (DV.TRIALS_BEFORE, DV.TRIALS_ADDED, RT.TRIALS_BEFORE) == (826, 20, 846) and DV.SENT_NONE == RT.SENTENCES["none"]
          and M.ESTIMATOR == "LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)" and len(RT.NONCOMMON["sealed"]) == 18
          and {"model23.json", "results/dev23_summary.json"} <= set(RT.CORE))


def main():
    with tempfile.TemporaryDirectory(prefix="r23_tests_") as tmp:
        r18, r20, r22 = earlier_tests(tmp)
        frozen_tests()
        trade_tests()
        adv_tests(tmp)
        signal_tests()
        future_tests()
        model_tests(tmp)
        engine_tests(tmp)
        rule_tests()
        stats_tests()
        world_tests(os.path.join(tmp, "world"))
    repo_guard_test()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    n_ok = sum(r["pass"] for r in RESULTS)
    path = os.path.join(HERE, "results", "engine23_tests.json")
    keep_old(path)
    with open(path, "w") as fh:
        json.dump({"passed": n_ok, "total": len(RESULTS), "research18_tests": r18, "research20_tests": r20, "research22_tests": r22,
                   "numba": bool(E.HAVE_NUMBA), "tests": RESULTS}, fh, indent=1, default=str, ensure_ascii=False)
    for r in RESULTS:
        print(("PASS " if r["pass"] else "FAIL ") + r["test"][:150])
        if not r["pass"]:
            print("     ", json.dumps(r, default=str, ensure_ascii=False)[:2500])
    print(f"{n_ok}/{len(RESULTS)} passed (research 18's tests: {r18}, research 20's tests: {r20}, research 22's tests: {r22})")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
