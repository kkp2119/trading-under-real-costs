#!/usr/bin/env python3
"""Research 21 synthetic tests (step-2 prompt 1; they must pass before any development run). Synthetic bars only.
  research 20   its 117 synthetic tests, run from test20.py's own functions on the unchanged engine files (nothing is
                written in study20_pairs/); the engine files equal tag code-freeze-20
  sizes         loader21: a file dated 2025-10-31 is read as stored, one dated 2025-11-03 has bsz and asz divided by 100,
                prices and trade volumes untouched; the one-tick depth penalty of L3D1 is computed from the divided sizes
  kappa         boundaries for kappa 1 and 2 (equal enters, just below does not), and through the pair engine
  control       SPEC21_ADDENDUM C1: the worked examples and the yearly numbers of control pairs
  statistics    seed 21, the 8 candidates, the A* rule (5% tie, mean <= 0 drops H2), Holm with 2, 3 and 4 hypotheses,
                the sentences of SPEC21 6, the date split
  core21        the same aggregates as core20.pairday_all for mid, M_L3D1, M_L0D0; diagnostics; family tables
  small world   synthetic bars for 12 symbols on 8 "development" days of 2020 and 8 "sealed" days around 2025-11-03 in a
                temporary folder: dev21.analyse, the rehearsal and the sealed path of run_test21 (guard included) are run
                end to end and checked against hand-computed values (families kept apart, A* per family, a family
                without H2, the size unit across 2025-11-03, the daily files of the development run and of the sealed
                run, a development summary whose content changed after the freeze)
Writes results/engine21_tests.json; exits 1 when a test fails.
  python test21.py"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core21 as C21  # noqa: E402
import loader21 as L21  # noqa: E402
import stats21 as ST21  # noqa: E402
import dev21 as DV  # noqa: E402
import run_test21 as RT  # noqa: E402
import test20 as T20  # noqa: E402  (research 20's test functions and synthetic legs; its main is never called)

C, S, PE, ST20 = C21.C, C21.S, C21.PE, C21.ST20
G = PE.G
N, BID, SPR, GE = T20.N, T20.BID, T20.SPR, T20.GE
T20_SHA = "388cf5bad6dbedb812c0323bfe0ccee06121262b2436141ff127bc5a9912f1fa"
RESULTS = []
H1 = ST21.sig_name(ST21.H1_SIGNAL)


def check(name, ok, **info):
    RESULTS.append({"test": name, "pass": bool(ok), **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}})


def close(a, b, tol=1e-9):
    return bool(np.allclose(a, b, rtol=0, atol=tol))


# ---------------------------------------------------------------- research 20's tests and the frozen engine
def research20_tests():
    T20.RESULTS.clear()
    for fn in (T20.signal_tests, T20.kernel_tests, T20.pair_tests, T20.timetable_tests, T20.stats_tests, T20.gap_tests, T20.delay_tests):
        fn()
    n_ok = sum(r["pass"] for r in T20.RESULTS)
    stored = json.load(open(os.path.join(C21.R20, "results", "engine20_tests.json")))
    check("research 20's 117 synthetic tests pass on the unchanged engine files (test20.py's functions; nothing written in study20_pairs/)",
          len(T20.RESULTS) == 117 and n_ok == 117 and [r["test"] for r in T20.RESULTS] == [r["test"] for r in stored["tests"]],
          passed=n_ok, total=len(T20.RESULTS), failed=[r["test"] for r in T20.RESULTS if not r["pass"]])
    return f"{n_ok}/{len(T20.RESULTS)}"


def frozen_tests():
    want = dict(C21.FROZEN20, **{"study20_pairs/test20.py": T20_SHA})
    disk = {p: hashlib.sha256(open(os.path.join(C21.PROJ, p), "rb").read()).hexdigest() for p in want}
    tag = {p: hashlib.sha256(subprocess.run(["git", "-C", C21.PROJ, "show", f"code-freeze-20:{p}"], capture_output=True, check=True).stdout).hexdigest()
           for p in want}
    check("the engine files of research 17 and 20 on disk = tag code-freeze-20 = the sha256 list of core21 (and test20.py)",
          disk == want and tag == want and not C21.frozen_differences() and PE.ENGINE17_SHA == want["study17_new_stocks/engine17.py"], files=sorted(want))


# ---------------------------------------------------------------- loader: sizes and guards
def full(b):
    """A 21-column bar dict from a synthetic leg of test20 (the 14 kernel columns); the other columns are 0."""
    out = {c: np.zeros(N, np.int64) for c in L21.COLS}
    for k, v in b.items():
        out[k] = np.asarray(v, dtype=np.int64).copy()
    out["vol"] = out["vol_bid"] + out["vol_ask"] + out["vol_in"] + out["vol_unc"]
    return out


def save(root, sym, date, b, dtype=np.int32):
    """Write one synthetic bar file and return its manifest row."""
    os.makedirs(os.path.join(root, date[:4]), exist_ok=True)
    p = os.path.join(root, date[:4], f"{sym}_{date}.npz")
    np.savez_compressed(p, **{c: b[c].astype(dtype) for c in L21.COLS})
    return {"sym": sym, "date": date, "file": os.path.relpath(p, root), "rows": str(N), "bytes": str(os.path.getsize(p)), "sha256": L21.sha256(p)}


def size_tests(tmp):
    root = os.path.join(tmp, "sizes")
    b = full(T20.spike_leg())
    b["bsz"][:], b["asz"][:] = 30_000, 20_000
    b["bsz"][100:200] = 0                                               # no size: stays 0
    b["vol_bid"][5000:5100], b["vol_ask"][6000:6050], b["vol_bid_odd"][5000:5010] = 700, 300, 50
    b["ntr"][5000:5100], b["px_last"][5000:5100], b["px_min"][5000:5100], b["px_max"][5000:5100] = 3, BID, BID, BID + SPR
    b["vol"] = b["vol_bid"] + b["vol_ask"] + b["vol_in"] + b["vol_unc"]
    rows = {d: save(root, "XX", d, b) for d in ("20201231", "20251031", "20251103", "20261110")}
    got = {d: L21.read(os.path.join(root, r["file"]), r) for d, r in rows.items()}
    other = [c for c in L21.COLS if c not in L21.SIZE_COLS]
    check("sizes: a file dated 2020-12-31 or 2025-10-31 is read exactly as stored (all 21 columns)",
          all(np.array_equal(got[d][c], b[c]) for d in ("20201231", "20251031") for c in L21.COLS))
    check("sizes: a file dated 2025-11-03 or later has bsz and asz divided by 100; prices and trade volumes are as stored",
          all(np.array_equal(got[d]["bsz"], b["bsz"] // 100) and np.array_equal(got[d]["asz"], b["asz"] // 100)
              and all(np.array_equal(got[d][c], b[c]) for c in other) for d in ("20251103", "20261110"))
          and int(got["20251103"]["bsz"].max()) == 300 and int(got["20251103"]["asz"].max()) == 200 and int(got["20251103"]["bsz"][150]) == 0
          and int(got["20251103"]["vol_bid"].sum()) == 70_000, untouched_columns=len(other))
    check("sizes: the stored file is not changed by reading (sha256 as in its manifest row)",
          all(L21.sha256(os.path.join(root, r["file"])) == r["sha256"] for r in rows.values()))
    d0 = {c: b[c].copy() for c in L21.COLS}
    check("fix_sizes: the cut is 2025-11-03 (2025-11-02 unchanged, 2025-11-03 divided)",
          int(L21.fix_sizes({c: v.copy() for c, v in d0.items()}, "20251102")["asz"][0]) == 20_000
          and int(L21.fix_sizes({c: v.copy() for c, v in d0.items()}, "20251103")["asz"][0]) == 200
          and L21.SIZE_COLS == ("bsz", "asz") and L21.SIZE_FROM == "20251103" and L21.SIZE_DIV == 100)
    bad = {c: v.copy() for c, v in b.items()}
    bad["asz"][777] = 20_050
    rb = save(root, "YY", "20251103", bad)
    rb0 = save(root, "YY", "20251031", bad)
    refused = {}
    for name, fn in (("not a multiple of 100 from 2025-11-03", lambda: L21.read(os.path.join(root, rb["file"]), rb)),
                     ("sha256", lambda: L21.read(os.path.join(root, rows["20251031"]["file"]), dict(rows["20251031"], sha256="0" * 64))),
                     ("dtype", lambda: L21.read(os.path.join(root, save(root, "ZZ", "20200102", b, np.int64)["file"]), save(root, "ZZ", "20200102", b, np.int64)))):
        try:
            fn()
            refused[name] = False
        except L21.Refused:
            refused[name] = True
    ok_before = int(L21.read(os.path.join(root, rb0["file"]), rb0)["asz"][777]) == 20_050
    check("read() refuses: a quote size that is not a multiple of 100 from 2025-11-03 (the same value before that day is read as stored), "
          "a sha256 that differs from the manifest, a column that is not int32", all(refused.values()) and ok_before, refused=refused)
    # the development-only guard
    g = {}
    for name, (rt, row, allow) in {"another folder": (root, rows["20201231"], False), "date after 2020": (L21.DEV_ROOT, rows["20251103"], False),
                                    "sealed path": (os.path.join(tmp, "sealed", "bars_1s"), rows["20201231"], False)}.items():
        try:
            L21.load(rt, row, allow)
            g[name] = False
        except L21.Refused:
            g[name] = True
        except OSError:
            g[name] = False
    ok_allow = np.array_equal(L21.load(root, rows["20251103"], True)["asz"], b["asz"] // 100)
    check("load() without allow_sealed refuses a folder that is not a development folder, a date after 2020 and a path under 'sealed'; "
          "with allow_sealed (run_test21 after its guard) it reads", all(g.values()) and ok_allow, refused=g)
    # the depth penalty from the divided sizes
    a, bb = full(T20.spike_leg()), full(T20.leg())
    for x in (a, bb):
        x["bsz"][:], x["asz"][:] = 20_000, 20_000                       # 200 shares in the unit used from 2025-11-03
    res = {}
    for d in ("20251031", "20251103"):
        ra, rbb = save(root, "AA", d, a), save(root, "BB", d, bb)
        la, lb = L21.read(os.path.join(root, ra["file"]), ra), L21.read(os.path.join(root, rbb["file"]), rbb)
        tr, _, _, _ = T20.run(la, lb)
        res[d] = (len(tr), [int(T20.od(tr, 0, j)[4]) for j in range(4)], float(PE.money(tr, 1)["pnl"][0]), float(PE.money(tr, 0)["pnl"][0]))
    check("L3D1 with the divided sizes: 500-share market orders against 200 displayed shares pay one tick on 300 shares per order "
          "(pnl 12 instead of 24) on 2025-11-03; the same stored value read on 2025-10-31 (20,000 shares) gives no penalty",
          res["20251103"][:2] == (1, [300] * 4) and close(res["20251103"][2:], [12.0, 24.0]) and res["20251031"][:2] == (1, [0] * 4)
          and close(res["20251031"][2:], [24.0, 24.0]), got=res)


# ---------------------------------------------------------------- kappa
def kappa_tests():
    for ka in (1.0, 2.0):
        w = ka * 2e-4
        dev = np.array([w, w, np.nextafter(w, 0.0), -w, -w, -np.nextafter(w, 0.0), w])
        rs = np.array([2e-4, np.nextafter(2e-4, 1.0), 2e-4, 2e-4, np.nextafter(2e-4, 1.0), 2e-4, 1e-4])
        side, _ = S.decide(np.where(dev > 0, 5.0, -5.0), dev, rs, np.ones(len(dev), bool), 2.0, ka)
        check(f"kappa {ka:g} boundary: |s - m| equal to kappa x (relative spreads) enters, just below does not (both signs)",
              side.tolist() == [1, 0, 0, -1, 0, 0, 1], side=side)
    b = T20.leg()
    n10 = {ka: len(T20.run(T20.spike_leg(ticks=10), b, ka=ka)[0]) for ka in (0.0, 1.0, 2.0, 3.0)}
    n3 = {ka: len(T20.run(T20.spike_leg(ticks=3), b, ka=ka)[0]) for ka in (0.0, 1.0, 2.0)}
    check("kappa through the pair engine (two legs with 2-tick spreads at 100 dollars, relative spreads 4 bp): a 10 bp gap trades with "
          "kappa 0, 1, 2 and not with 3; a 3 bp gap trades with kappa 0 only", n10 == {0.0: 1, 1.0: 1, 2.0: 1, 3.0: 0} and n3 == {0.0: 1, 1.0: 0, 2.0: 0},
          trades_10bp=n10, trades_3bp=n3)
    B = S.base(T20.spike_leg(ticks=3), b, N)
    ok, got = True, {}
    for c in ST21.settings():
        side, _ = S.flags(B, *c, N)
        d = C21.diagnose(B, *c, N)
        ok &= d[2] - d[3] == (side != 0).sum() and d[1] <= d[0] and d[3] <= d[2]
        got[ST21.sig_name(c)] = d.tolist()
    d1, d0 = C21.diagnose(B, 10, 2.0, 1.0, N), C21.diagnose(B, 10, 2.0, 0.0, N)
    check("diagnostics: |z| >= z_in points minus the points kept out by kappa = entry flags, for the 9 settings; a 3 bp gap is kept "
          "out by kappa 1 at all of its 6 grid points and by kappa 0 at none", ok and d1[2] == 6 and d1[3] == 6 and d0[3] == 0 and d0[2] == 6, counts=got)


# ---------------------------------------------------------------- control pairs
def control_tests():
    ex = {2015: ["MET/SO", "DUK/MPC", "VLO/USB"], 2016: [], 2022: ["JPM/HLT", "MAR/NSC", "V/BAC", "CSX/MA"]}
    got = {y: [f"{a}/{b}" for _, a, b in L21.controls(y)] for y in range(2015, 2027)}
    check("control pairs: the worked examples of SPEC21_ADDENDUM C1 (2015, 2016, 2022) from the selected lists",
          all(got[y] == v for y, v in ex.items()) and [p for p, _, _ in L21.controls(2015)] == ["C1", "C2", "C3"], got={y: got[y] for y in ex})
    check("control pairs: numbers per year, development 3, 0, 2, 2, 3, 3 and sealed 2, 4, 4, 3, 3, 4",
          [len(got[y]) for y in range(2015, 2021)] == [3, 0, 2, 2, 3, 3] and [len(got[y]) for y in range(2021, 2027)] == [2, 4, 4, 3, 3, 4],
          counts={y: len(v) for y, v in got.items()})
    sp = [("S1", "A1", "B1"), ("S2", "A2", "B2"), ("S3", "A3", "B3"), ("S4", "A4", "B4")]
    f = L21.control_pairs
    check("control rule: next B in order with wrap-around, never the own B, a B is used once, same first SIC digit is skipped, "
          "no candidate left -> no control pair for that A",
          f(sp, ["1000", "2000", "3000", "4000"]) == [("C1", "A1", "B2"), ("C2", "A2", "B3"), ("C3", "A3", "B4"), ("C4", "A4", "B1")]
          and f(sp, ["6021", "6311", "4911", "6021"]) == [("C1", "A1", "B3"), ("C3", "A3", "B4")]
          and f(sp, ["6021", "6022", "6199", "6311"]) == [] and f(sp[:1], ["6021"]) == []
          and f(sp[:2], ["6021", "4911"]) == [("C1", "A1", "B2"), ("C2", "A2", "B1")]
          and f(sp, ["7011", "7389", "4011", "6021"]) == [("C1", "A1", "B3"), ("C2", "A2", "B4"), ("C3", "A3", "B1"), ("C4", "A4", "B2")])
    codes = L21.sic_codes()
    check("SIC codes: one 4-digit code per family S pair and year (48 pair-years), the first digit decides",
          len(codes) == 48 and all(len(v) == 4 and v.isdigit() for v in codes.values()) and codes[(2015, "S1")] == "6311")


# ---------------------------------------------------------------- statistics, candidates, hypotheses
def stats_tests():
    idx = ST21.boot_index(250)
    check("bootstrap index: 10,000 x T, circular blocks of 21 days, seed 21 (not research 20's seed 20), reproducible",
          idx.shape == (10_000, 250) and np.array_equal(idx, G.boot_index(250, B=10_000, block=21, seed=21)) and np.array_equal(idx, ST21.boot_index(250))
          and not np.array_equal(idx, ST20.boot_index(250)) and np.array_equal(ST21.boot_index(250, seed=20), ST20.boot_index(250))
          and np.all((idx[:, 1:21] - idx[:, :20]) % 250 == 1) and (ST21.BOOT_B, ST21.BOOT_BLOCK, ST21.BOOT_SEED) == (10_000, 21, 21))
    names = [ST21.cand_name(c) for c in ST21.candidates()]
    check("candidates: 8 per family, W {10, 30} x z_in {2, 3} x kappa {1, 2}, market orders; the 9th setting of the tables is W30_z2_k0; "
          "trials 794 + 16 = 810",
          names == ["W10_z2_k1_M", "W10_z2_k2_M", "W10_z3_k1_M", "W10_z3_k2_M", "W30_z2_k1_M", "W30_z2_k2_M", "W30_z3_k1_M", "W30_z3_k2_M"]
          and ST21.settings()[:8] == ST21.candidates() and ST21.settings()[8] == (30, 2.0, 0.0) and ST21.H1_SIGNAL not in ST21.candidates()
          and ST21.TRIALS_BEFORE + ST21.TRIALS_ADDED == 810 and C21.KINDS == ("mid", "M_L3D1", "M_L0D0"), names=names)
    rows = [{"name": "a", "mean": 100.0, "trades": 1000}, {"name": "b", "mean": 96.0, "trades": 500}, {"name": "c", "mean": 94.9, "trades": 10}]
    edge = [{"name": "a", "mean": 100.0, "trades": 9}, {"name": "b", "mean": 95.0, "trades": 5}]
    neg = [{"name": "a", "mean": -10.0, "trades": 100}, {"name": "b", "mean": -10.4, "trades": 50}, {"name": "c", "mean": -10.6, "trades": 1}]
    zero = [{"name": "a", "mean": -5.0, "trades": 40}, {"name": "b", "mean": 0.0, "trades": 0}, {"name": "c", "mean": 0.0, "trades": 0}]
    sa, se, sn, sz = (ST21.select_astar(r) for r in (rows, edge, neg, zero))
    check("A*: the highest L3D1 mean; inside 5% of it (the 5% line included) the fewest trades, then the earlier candidate",
          sa[0] == "b" and sa[1]["tied_within_5pct"] == ["a", "b"] and se[0] == "b" and sn[0] == "b" and sn[1]["tied_within_5pct"] == ["a", "b"])
    h = ST21.hypotheses
    check("A* with a mean <= 0 (negative, or 0 without a trade) means no profit in development: that family's H2 is dropped",
          sa[1]["profitable_in_development"] and not sn[1]["profitable_in_development"] and sz[0] == "b" and not sz[1]["profitable_in_development"]
          and h({"E": True, "S": True}) == ["H1-E", "H1-S", "H2-E", "H2-S"] and h({"E": True, "S": False}) == ["H1-E", "H1-S", "H2-E"]
          and h({"E": False, "S": True}) == ["H1-E", "H1-S", "H2-S"] and h({"E": False, "S": False}) == ["H1-E", "H1-S"])
    lv = {}
    for m in (2, 3, 4):
        r = ST21.holm({f"h{i}": 0.001 * (i + 1) for i in range(m)})
        lv[m] = [r[f"h{i}"]["level"] for i in range(m)]
    h3 = ST21.holm({"H1-E": 0.0, "H1-S": 0.02, "H2-E": 0.03})
    h4 = ST21.holm({"H1-E": 0.012, "H1-S": 0.013, "H2-E": 0.02, "H2-S": 0.04})
    h4b = ST21.holm({"H1-E": 0.0125, "H1-S": 0.0, "H2-E": 0.0, "H2-S": 0.0})
    check("Holm (overall 0.05): levels 0.025, 0.05 with two hypotheses; 0.05/3, 0.025, 0.05 with three; 0.0125, 0.05/3, 0.025, 0.05 "
          "with four; p must be below its level; everything after the first failure is not rejected",
          lv[2] == [0.025, 0.05] and lv[3] == [0.05 / 3, 0.025, 0.05] and lv[4] == [0.0125, 0.05 / 3, 0.025, 0.05]
          and [h3[k]["rejected"] for k in ("H1-E", "H1-S", "H2-E")] == [True, True, True]
          and [h4[k]["rejected"] for k in ("H1-E", "H1-S", "H2-E", "H2-S")] == [True, True, True, True]
          and [h4b[k]["rejected"] for k in ("H1-S", "H2-E", "H2-S", "H1-E")] == [True, True, True, True] and h4b["H1-E"]["level"] == 0.05
          and [ST21.holm({"a": 0.025, "b": 0.03})[k]["rejected"] for k in "ab"] == [False, False]
          and [ST21.holm({"a": 0.02, "b": 0.03, "c": 0.001})[k]["rejected"] for k in "cab"] == [True, True, True]
          and [ST21.holm({"a": 0.03, "b": 0.03, "c": 0.001})[k]["rejected"] for k in "cab"] == [True, False, False]
          and ST21.strictest_level(2) == 0.025 and ST21.strictest_level(3) == 0.05 / 3 and ST21.strictest_level(4) == 0.0125, levels=lv)
    check("sentences of SPEC21 6, as written, per family",
          ST21.sentence("H1", "E", True) == "거의 같은 ETF 쌍의 가격이 벌어지면 처음 보는 기간에도 다시 붙었다"
          and ST21.sentence("H1", "S", False) == "같은 업종 주식 쌍의 되돌림은 처음 보는 기간에서 확인되지 않았다"
          and ST21.sentence("H2", "E", True) == "거의 같은 ETF 쌍의 차익은 실제 비용을 넣고도 수익이 났다"
          and ST21.sentence("H2", "S", False) == "같은 업종 주식 쌍의 되돌림은 있었지만 비용을 넘는 수익은 확인되지 않았다"
          and ST21.no_h2_sentence("S") == "같은 업종 주식 쌍은 개발 구간에서 이미 비용을 넘지 못했다. 그래서 수익 가설은 봉인 데이터로 시험하지 않았다"
          and ST21.no_h2_sentence("E") == "거의 같은 ETF 쌍은 개발 구간에서 이미 비용을 넘지 못했다. 그래서 수익 가설은 봉인 데이터로 시험하지 않았다"
          and RT.PHRASE == "연구21 테스트를 한 번 연다" and RT.WINDOW == ("20210104", "20261110"))
    days = ["20251029", "20251030", "20251031", "20251103", "20251104"]
    d, n = np.array([1.0, 2.0, 3.0, 10.0, 20.0]), np.array([1, 0, 2, 3, 4])
    sp = C21.date_split(d, n, days)
    sp0 = C21.date_split(d, n, ["20201224", "20201228", "20201229", "20201230", "20201231"])
    check("date split at 2025-11-03: days, mean and trades of each part, its own bootstrap index; a part without days is empty",
          sp["before"]["days"] == 3 and sp["from"]["days"] == 2 and close(sp["before"]["mean_daily_usd"], 2.0) and close(sp["from"]["mean_daily_usd"], 15.0)
          and sp["before"]["trades"] == 3 and sp["from"]["trades"] == 7 and sp["before"]["last_day"] == "20251031" and sp["from"]["first_day"] == "20251103"
          and sp["before"]["ci_lo"] <= 2.0 <= sp["before"]["ci_hi"] and sp0["from"] is None and sp0["before"]["days"] == 5)


# ---------------------------------------------------------------- core21 against core20, family tables
def random_day(seed):
    rng = np.random.default_rng(seed)
    a, b = T20.leg(), T20.leg(spread=300)
    path = BID + 100 * np.cumsum(rng.integers(-1, 2, N))
    for k in ("bid", "bid_max", "bid_min"):
        a[k] = path.copy()
    for k in ("ask", "ask_max", "ask_min"):
        a[k] = path + SPR
    lk = rng.random(N) < 0.004
    for k in ("ask", "ask_max", "ask_min"):
        a[k][lk] = a["bid"][lk]
    a["bsz"], a["asz"] = rng.integers(1, 20, N) * 100, rng.integers(1, 20, N) * 100
    return a, b


def core_tests():
    a, b = random_day(21)
    cfgs = ST20.configs() + ST21.settings()
    r20 = C.pairday_all(a, b, N, cfgs, keep=(0, 9), mid_delays=S.MID_DELAYS)
    r21 = C21.pairday(a, b, N, cfgs, keep=(0, 9), mid_delays=S.MID_DELAYS, diag=True)
    rm = C21.pairday(a, b, N, cfgs, mid_delays=S.MID_DELAYS, mid_only=True)
    check("core21.pairday = core20.pairday_all for mid, M_L3D1 and M_L0D0 on a random day with locked bar ends (17 settings): aggregates, "
          "delayed-mid sums, kept M trades, grid counts; trades exist and the depth penalty is at work",
          np.array_equal(r21[0], r20[0][:, :3]) and np.array_equal(r21[6], r20[6]) and r21[1] == r20[1] == 0 and r21[3:6] == r20[3:6]
          and all(np.array_equal(r21[2][ci], r20[2][ci]["M"]) for ci in (0, 9)) and r21[0][:, 1, 0].sum() > 0
          and (r21[0][:, 1, C21.COL["pnl"]] < r21[0][:, 2, C21.COL["pnl"]]).any(), m_trades=int(r21[0][:, 1, 0].sum()))
    check("mid_only (control pairs): the same mid aggregates and delayed sums, the M columns stay 0",
          np.array_equal(rm[0][:, 0], r21[0][:, 0]) and not rm[0][:, 1:].any() and np.array_equal(rm[6], r21[6]))
    r1 = C21.pairday(a, b, N, cfgs, delay=1)
    r1_20 = C.pairday_all(a, b, N, cfgs, delay=1)
    check("order delay 1 s: the M aggregates equal core20's; no mid run", np.array_equal(r1[0][:, 1:], r1_20[0][:, 1:3]) and not r1[0][:, 0].any())
    B = S.base(a, b, N)
    ok = True
    for W in ST21.WS:
        z, dev, has = S.zscore(B["s"], B["valid"], W)
        sd = C21.sigma_raw(B["s"], B["valid"], W)
        ok &= np.array_equal(np.isfinite(sd), has) and np.array_equal(dev[has] / np.maximum(sd[has], S.SIGMA_FLOOR), z[has])
    sf = np.zeros(4680)
    sf[2000] = 3e-4
    flat = C21.sigma_raw(sf, np.ones(4680, bool), 10)
    check("sigma before the floor: z of signal20 = (s - m) / max(sigma, 1 bp) exactly; a flat window has sigma 0 and a window with one "
          "3 bp point has 3 bp / sqrt(120), both below the 1 bp floor", ok and flat[1999] == 0.0 and abs(flat[2001] - 3e-4 / np.sqrt(120)) < 1e-15
          and flat[2001] < S.SIGMA_FLOOR
          and r21[7][:, 1].sum() <= r21[7][:, 0].sum() and (r21[7][:, 0] > 0).all())
    # family tables from hand-made aggregates
    tasks = [{"date": d, "year": 2020, "family": f, "pair": p, "name": nm} for d in ("20201230", "20201231")
             for f, p, nm in (("E", "E1", "EA/EB"), ("S", "S1", "SA/SB"), ("C", "C1", "SA/SD"))]
    A = np.zeros((6, 1, 3, C21.NAGG))
    A[:, 0, :, C21.COL["n"]] = 1
    A[:, 0, 0, C21.COL["pnl"]] = [10, 200, 3000, 20, 400, 5000]
    A[:, 0, 1, C21.COL["pnl"]] = [1, 20, 300, 2, 40, 500]
    D = np.zeros((6, 1, 3, 3))
    D[:, 0, :, 0], D[:, 0, :, 1] = 1, np.array([10, 200, 3000, 20, 400, 5000])[:, None] / 2
    days = ["20201230", "20201231"]
    idx = ST21.boot_index(2)
    got = {}
    for f in ("E", "S", "C"):
        M, blocks, Af, Df, tf = C21.family_tables(A, D, tasks, days, [ST21.H1_SIGNAL], idx, f, mid_only=(f == "C"))
        b_ = blocks[H1]
        got[f] = (b_["rows"]["mid"]["total_usd"], b_["rows"].get("M_L3D1", {}).get("total_usd"), M["names"], len(tf),
                  C21.run_series(Af, Df, M, 0, "mid", 2)[0].tolist(), b_["delayed"]["5"]["ratio_to_delay0"], sorted(b_["rows"]))
    check("family tables: a family's daily P&L is the sum of that family's pair-days only (E, S and the control pairs kept apart)",
          got["E"] == (30.0, 3.0, ["EA/EB"], 2, [10.0, 20.0], 0.5, ["M_L0D0", "M_L3D1", "mid"])
          and got["S"] == (600.0, 60.0, ["SA/SB"], 2, [200.0, 400.0], 0.5, ["M_L0D0", "M_L3D1", "mid"])
          and got["C"] == (8000.0, None, ["SA/SD"], 2, [3000.0, 5000.0], 0.5, ["mid"]), got=got)
    pdr = C21.pair_daily(A, D, tasks, 0, H1)
    check("pair-by-pair daily rows: one row per pair-day with the mid, delayed-mid and M columns; a control pair-day has no M values",
          len(pdr) == 6 and len(pdr[0]) == len(C21.DAILY_COLS) and pdr[1][:7] == ("20201230", "S", "S1", "SA/SB", H1, 1, 200.0) and pdr[1][-2:] == (20.0, 0.0)
          and pdr[2][-3:] == (None, None, None) and pdr[2][7:10] == (1500.0, 1500.0, 1500.0))
    sr = C21.side_row("E", C21.family_tables(A, D, tasks, days, [ST21.H1_SIGNAL], idx, "E")[1][H1])
    sc = C21.side_row("C", C21.family_tables(A, D, tasks, days, [ST21.H1_SIGNAL], idx, "C", mid_only=True)[1][H1])
    r20d, r20s = C21.r20_side("dev"), C21.r20_side("sealed")
    check("side-by-side line: mid per trade, share left after 5 s, cost per trade (execution cost + fee) and P&L per trade per fill "
          "assumption; the control line has no cost; research 20's lines come from its stored results",
          close(sr["mid_per_trade"], 15.0) and sr["ratio_5s"] == 0.5 and close(sr["L3D1"]["pnl_per_trade"], 1.5) and sc["L3D1"] is None and sc["L0D0"] is None
          and r20d["trades"] == 207_420 and r20s["trades"] > 0 and all(close(r["mid_per_trade"] - r[v]["cost_per_trade"], r[v]["pnl_per_trade"], 1e-6)
                                                                         for r in (r20d, r20s) for v in C21.VARS))


# ---------------------------------------------------------------- a small world: development run, rehearsal, sealed path
DEV_DAYS = ("2020-12-21", "2020-12-22", "2020-12-23", "2020-12-24", "2020-12-28", "2020-12-29", "2020-12-30", "2020-12-31")
SEALED_DAYS = ("2025-10-28", "2025-10-29", "2025-10-30", "2025-10-31", "2025-11-03", "2025-11-04", "2025-11-05", "2025-11-06")
UNI = {2020: (("E", "E1", "EA", "EB", ""), ("E", "E2", "EC", "ED", ""), ("S", "S1", "SA", "SB", "6021"), ("S", "S2", "SC", "SD", "4911"),
              ("S", "S3", "SE", "SF", "7374"), ("S", "S4", "SG", "SH", "6022")),
       2025: (("E", "E1", "EA", "EB", ""), ("E", "E2", "EC", "ED", ""), ("S", "S1", "SA", "SB", "6021"), ("S", "S2", "SC", "SD", "4911"))}


def world_leg(sym, i, size):
    """Bars of one symbol on day number i. EA: a level shift of 10 ticks at 09:45:00 that stays (a losing trade for W 10
    only) and a 10-tick spike of 30 s at 11:30:00 (not on day 5); EC: a 5-tick spike on even days; SA, SE: a 9-tick spike
    every day; SC: on odd days; the B legs are flat, those of family S with a 6-tick spread."""
    if sym == "EA":
        b = T20.setq(T20.leg(size=size), 899, N, BID + 1000)
        return T20.setq(b, GE, GE + 30, BID + 2000) if i != 5 else b
    if sym in ("EC", "SA", "SC", "SE"):
        g0, ticks, on = {"EC": (GE + 600, 5, i % 2 == 0), "SA": (GE + 1200, 9, True), "SC": (GE + 1800, 9, i % 2 == 1), "SE": (GE + 300, 9, True)}[sym]
        b = T20.leg(size=size)
        return T20.setq(b, g0, g0 + 30, BID + ticks * 100) if on else b
    return T20.leg(spread=600 if sym in ("SB", "SD", "SF", "SH") else SPR, size=size)


def build_world(tmp):
    repo, dev_root, sealed_root = os.path.join(tmp, "repo"), os.path.join(tmp, "dev", "bars_1s"), os.path.join(tmp, "sealed", "bars_1s")
    os.makedirs(os.path.join(repo, "results"))
    pd.DataFrame({"date": DEV_DAYS + SEALED_DAYS, "close_time": ["13:00" if d == "2020-12-24" else "16:00" for d in DEV_DAYS + SEALED_DAYS]}).to_csv(
        os.path.join(repo, "calendar21.csv"), index=False)
    stats = []
    for y, ps in UNI.items():
        pd.DataFrame([(y, f, p, leg, s, s) for f, p, a, b, _ in ps for leg, s in (("A", a), ("B", b))],
                     columns=["year", "family", "pair", "leg", "sym", "symbol_15"]).to_csv(os.path.join(repo, f"universe21_{y}.tsv"), sep="\t", index=False)
        stats += [(y, f, p, a, b, sic) for f, p, a, b, sic in ps]
    pd.DataFrame(stats, columns=["year", "family", "pair", "sym_a", "sym_b", "sic"]).to_csv(os.path.join(repo, "results", "universe21_pairs.tsv"),
                                                                                         sep="\t", index=False)
    for root, days, y, name in ((dev_root, DEV_DAYS, 2020, "DATA_MANIFEST21.tsv"), (sealed_root, SEALED_DAYS, 2025, "SEALED_MANIFEST21.tsv")):
        rows = []
        for i, d in enumerate(days):
            dd = d.replace("-", "")
            for s in sorted({x for _, _, a, b, _ in UNI[y] for x in (a, b)}):
                if (y, i, s) == (2020, 2, "SD"):
                    continue                                              # one leg missing: no pair-day for SC/SD and SA/SD
                rows.append(save(root, s, dd, full(world_leg(s, i, 20_000 if dd >= "20251103" else 200))))
        pd.DataFrame(rows)[["sym", "date", "file", "rows", "bytes", "sha256"]].to_csv(os.path.join(repo, name), sep="\t", index=False)
    return repo, dev_root, sealed_root


def world_tests(tmp):
    repo, dev_root, sealed_root = build_world(tmp)
    res = os.path.join(repo, "results")
    saved = {(m, k): getattr(m, k) for m, ks in ((L21, ("CALENDAR", "UNIVERSE", "PAIR_STATS", "DEV_MANIFEST", "DEV_ROOT")),
                                                 (DV, ("RES", "LOCAL", "SEALED_WINDOW")),
                                                 (RT, ("RES", "DEV_SUMMARY", "SEALED_ROOT", "SEALED_MANIFEST", "WINDOW", "UNLOCK", "FREEZE", "HERE"))) for k in ks}
    try:
        L21.CALENDAR, L21.UNIVERSE = os.path.join(repo, "calendar21.csv"), os.path.join(repo, "universe21_{y}.tsv")
        L21.PAIR_STATS, L21.DEV_MANIFEST, L21.DEV_ROOT = os.path.join(res, "universe21_pairs.tsv"), os.path.join(repo, "DATA_MANIFEST21.tsv"), dev_root
        DV.RES, DV.LOCAL, DV.SEALED_WINDOW = res, os.path.join(tmp, "dev_results"), ("20251028", "20251106")
        RT.RES, RT.DEV_SUMMARY, RT.SEALED_ROOT = res, os.path.join(res, "dev21_summary.json"), os.path.dirname(sealed_root)
        RT.SEALED_MANIFEST, RT.WINDOW = os.path.join(repo, "SEALED_MANIFEST21.tsv"), ("20251028", "20251106")
        RT.UNLOCK, RT.FREEZE, RT.HERE = os.path.join(repo, "UNLOCK21.txt"), os.path.join(repo, "FREEZE21.md"), repo
        _world(tmp, repo, res)
    finally:
        for (m, k), v in saved.items():
            setattr(m, k, v)


def _world(tmp, repo, res):
    tasks, days, skipped = L21.dev_plan()
    check("small world: the development plan has 16 pair-days of family E, 31 of S and 23 of the control pairs (3 a year); the day with "
          "one leg missing is no pair-day; the early close is read from the calendar",
          [sum(t["family"] == f for t in tasks) for f in "ESC"] == [16, 31, 23] and len(days) == 8
          and skipped == {"E": {"one_leg": 0, "neither": 0}, "S": {"one_leg": 1, "neither": 0}, "C": {"one_leg": 1, "neither": 0}}
          and {t["close_bar"] for t in tasks if t["date"] == "20201224"} == {12_600} and [f"{a}/{b}" for _, a, b in L21.controls(2020)] == ["SA/SD", "SC/SF", "SE/SH"])
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dev = DV.analyse(0, check_prerequisites=False)
    E, Sf, ctl = dev["families"]["E"], dev["families"]["S"], dev["control"]
    tot = lambda F, name, k: F["settings"][name]["rows"][k]["total_usd"]            # noqa: E731
    trd = lambda F, name, k: F["settings"][name]["rows"][k]["trades"]               # noqa: E731
    got = {"E": {n: (trd(E, n, "M_L3D1"), tot(E, n, "M_L3D1"), tot(E, n, "M_L0D0")) for n in E["settings"]},
           "S": {n: (trd(Sf, n, "M_L3D1"), tot(Sf, n, "M_L3D1"), tot(Sf, n, "M_L0D0")) for n in Sf["settings"]}}
    want_e = {"W10_z2_k1": (19, -272.0, -44.0), "W10_z2_k2": (15, -220.0, -40.0), "W10_z3_k1": (19, -272.0, -44.0), "W10_z3_k2": (15, -220.0, -40.0),
              "W30_z2_k1": (11, 32.0, 164.0), "W30_z2_k2": (7, 84.0, 168.0), "W30_z3_k1": (11, 32.0, 164.0), "W30_z3_k2": (7, 84.0, 168.0),
              "W30_z2_k0": (11, 32.0, 164.0)}
    want_s = {n: ((20, -260.0, -20.0) if n.endswith(("k1", "k0")) else (0, 0.0, 0.0)) for n in want_e}
    check("small world, development run: trades and total P&L of the 9 settings of each family equal the hand-computed values "
          "(L3D1 = L0D0 minus 12 dollars a trade: 300 shares above the displayed 200 on four orders)",
          all(got["E"][n][0] == want_e[n][0] and close(got["E"][n][1:], want_e[n][1:], 1e-6) for n in want_e)
          and all(got["S"][n][0] == want_s[n][0] and close(got["S"][n][1:], want_s[n][1:], 1e-6) for n in want_s), got=got)
    check("small world: families are kept apart (H1 mid totals E 450 = 7 x 50 + 4 x 25, S 900 = 20 x 45, control 855 = 19 x 45; pair lists)",
          close([tot(E, H1, "mid"), tot(Sf, H1, "mid"), ctl["rows"]["mid"]["total_usd"]], [450.0, 900.0, 855.0], 1e-6)
          and E["pairs"] == ["EA/EB", "EC/ED"] and Sf["pairs"] == ["SA/SB", "SC/SD", "SE/SF", "SG/SH"] and ctl["pairs"] == ["SA/SD", "SC/SF", "SE/SH"]
          and sorted(ctl["rows"]) == ["mid"] and close(E["settings"][H1]["rows"]["mid"]["mean_daily_usd"], 56.25, 1e-9)
          and [E["pair_days"], Sf["pair_days"], ctl["pair_days"]] == [16, 31, 23])
    check("small world, A* per family: E = W30_z2_k2_M (the highest mean, the earlier of two equal candidates) with H2; S = W10_z2_k2_M "
          "with a mean of 0 and no trade -> no profit in development, H2-S dropped; three hypotheses, first Holm level 0.05 / 3",
          E["astar"] == "W30_z2_k2_M" and E["h2_exists"] and E["selection"]["tied_within_5pct"] == ["W30_z2_k2_M", "W30_z3_k2_M"]
          and Sf["astar"] == "W10_z2_k2_M" and not Sf["h2_exists"] and Sf["selection"]["astar_mean_usd"] == 0.0
          and dev["hypotheses"] == ["H1-E", "H1-S", "H2-E"] and dev["holm_levels"]["first"] == 0.05 / 3 and list(dev["power"]) == dev["hypotheses"]
          and dev["trials_cumulative"] == 810 and all(dev["checks"]["second_pass_equal"].values()), astar={f: dev["families"][f]["astar"] for f in "ES"})
    dgE, dgS = E["diag"]["family"], Sf["diag"]["family"]
    check("small world, diagnostics: in S every |z| >= z_in point is kept out by kappa 2 and none by kappa 1; in E kappa 2 keeps out "
          "the 5-tick gaps only; the sigma floor applies to most signal points but not to all",
          dgS["kappa_blocked"]["W30_z2_k2"]["share"] == 1.0 and dgS["kappa_blocked"]["W30_z2_k1"]["share"] == 0.0
          and 0 < dgE["kappa_blocked"]["W30_z2_k2"]["share"] < 1 and dgE["kappa_blocked"]["W30_z2_k1"]["share"] == 0.0
          and all(0.5 < g["sigma_floor"][w]["share"] < 1 for g in (dgE, dgS) for w in ("W10", "W30"))
          and set(E["diag"]["by_pair"]) == {"EA/EB", "EC/ED"}, e=dgE["kappa_blocked"]["W30_z2_k2"], s_floor=dgS["sigma_floor"])
    side = {r["key"]: r for r in dev["side_by_side"]}
    check("small world, side-by-side table and delay check: E 450 / 11 per trade at the mid, cost 26 (L0D0) and 38 (L3D1) a trade, "
          "the control line without costs; A* with orders one second late keeps its trades",
          list(side) == ["R20", "E", "S", "C"] and close(side["E"]["mid_per_trade"], 450 / 11, 1e-9) and close(side["E"]["L0D0"]["cost_per_trade"], 26.0, 1e-9)
          and close(side["E"]["L3D1"]["cost_per_trade"], 38.0, 1e-9) and close(side["S"]["L3D1"]["pnl_per_trade"], -13.0, 1e-9) and side["C"]["L3D1"] is None
          and side["R20"]["trades"] == 207_420 and E["delay_check"]["L3D1"]["trades"] == 7 and close(E["delay_check"]["L3D1"]["total_usd"], 84.0, 1e-6)
          and os.path.exists(os.path.join(tmp, "dev_results", "dev21_trades_E_W30_z2_k2.npz")))
    ok_files = all(os.path.exists(os.path.join(res, f)) for f in ("dev21_summary.json", "dev21_report.md", "dev21_rows.csv"))
    try:
        with contextlib.redirect_stdout(buf):
            DV.analyse(0, check_prerequisites=False)
        again = False
    except SystemExit:
        again = True
    check("small world: the development run writes its summary, report and rows once and refuses a second run", ok_files and again)
    # rehearsal
    with contextlib.redirect_stdout(buf):
        reh = RT.main("rehearsal", 0)
    k = reh["checks"]
    check("small world, rehearsal: run_test21's code path on the development part reproduces the development run exactly "
          "(largest difference 0 for the rows, the delayed table, the control pairs and the side-by-side table) and reads no sealed path",
          k["pass"] and k["dev21_max_abs_diff"] == 0.0 and k["delayed_max_abs_diff"] == 0.0 and k["control_max_abs_diff"] == 0.0
          and k["side_by_side_max_abs_diff"] == 0.0 and k["no_sealed_path_read"] and k["rows_compared"] == 12
          and reh["read"]["bars_root"].endswith(os.path.join("dev", "bars_1s")) and list(reh["families"]["S"]["settings"]) == [H1, "W10_z2_k2"], checks=k)
    # the daily file of the development run (an output added before the development run; the numbers of the tables)
    dd, rd = pd.read_csv(os.path.join(res, "dev21_daily.csv")), pd.read_csv(os.path.join(res, "rehearsal21_daily.csv"))
    gs = dd.groupby(["family", "setting"]).sum(numeric_only=True)
    fams = {"E": E, "S": Sf}
    sums = all(close(gs.loc[(f, n), kk], F["settings"][n]["rows"][kk]["total_usd"], 1e-3)
               and int(gs.loc[(f, n), "trades_mid" if kk == "mid" else "trades_M"]) == F["settings"][n]["rows"][kk]["trades"]
               for f, F in fams.items() for n in F["settings"] for kk in C21.KINDS)
    late = all(close(gs.loc[(f, n), f"mid_delay{d}"], F["settings"][n]["delayed"][str(d)]["mean_daily_usd"] * 8, 1e-3)
               for f, F in fams.items() for n in F["settings"] for d in S.MID_DELAYS)
    key = ["date", "family", "pair", "setting"]
    both = rd.merge(dd, on=key, how="left", suffixes=("", "_dev"), indicator=True)
    same = bool((both["_merge"] == "both").all()) and all(
        np.array_equal(both[c].to_numpy(float), both[c + "_dev"].to_numpy(float), equal_nan=True) for c in C21.DAILY_COLS if c not in key + ["name"])
    check("small world, daily file of the development run: results/dev21_daily.csv has the columns of the sealed run's daily file and one row "
          "per pair-day and setting (E 16 x 9, S 31 x 9, control 23 x 1 = 446); its sums equal the tables for every family, setting and run "
          "(E 450 at the mid and 84 / 168 for W30_z2_k2, S 900, control 855) and the delayed table; control rows have no M values; the "
          "rehearsal's daily rows (117) are the same rows; the summary records the file and a largest gap of 0",
          list(dd.columns) == list(C21.DAILY_COLS) and len(dd) == 16 * 9 + 31 * 9 + 23 and len(rd) == 16 * 2 + 31 * 2 + 23 and sums and late and same
          and [len(dd[(dd.family == f) & (dd.setting == H1)]) for f in "ESC"] == [16, 31, 23] and sorted(dd[dd.family == "C"].setting.unique()) == [H1]
          and close([gs.loc[("E", H1), "mid"], gs.loc[("S", H1), "mid"], gs.loc[("C", H1), "mid"]], [450.0, 900.0, 855.0], 1e-3)
          and close([gs.loc[("E", "W30_z2_k2"), "M_L3D1"], gs.loc[("E", "W30_z2_k2"), "M_L0D0"], gs.loc[("S", "W10_z2_k1"), "M_L3D1"]], [84.0, 168.0, -260.0], 1e-3)
          and close(gs.loc[("C", H1), "mid"], ctl["rows"]["mid"]["total_usd"], 1e-3)
          and dd[dd.family == "C"][["trades_M", "M_L3D1", "M_L0D0"]].isna().all().all() and dd[dd.family != "C"][["trades_M", "M_L3D1", "M_L0D0"]].notna().all().all()
          and dev["daily"] == {"file": "results/dev21_daily.csv", "rows": 446, "columns": list(C21.DAILY_COLS)}
          and dev["checks"]["daily_file_max_gap_to_tables"]["trades"] == 0 and dev["checks"]["daily_file_max_gap_to_tables"]["usd"] < 1e-9,
          rows=len(dd), rehearsal_rows=len(rd), gap=dev["checks"]["daily_file_max_gap_to_tables"])
    # guard
    core_dir = lambda p: os.path.normpath(os.path.join(repo, p))                    # noqa: E731
    lines = []
    for p in RT.CORE:
        q = core_dir(p)
        if not os.path.exists(q):
            os.makedirs(os.path.dirname(q), exist_ok=True)
            open(q, "w").write(f"stand-in for {p}\n")
        lines.append(f"| `{p}` | `{RT.sha(q)}` | x |")
    open(RT.FREEZE, "w", encoding="utf-8").write("# FREEZE (synthetic)\n\n| file | sha256 | role |\n|---|---|---|\n" + "\n".join(lines) + "\n")
    for doc in RT.DOCS:
        open(os.path.join(repo, f"{doc}.md"), "w").write(f"{doc} stand-in\n")
        open(os.path.join(repo, f"{doc}.sha256"), "w").write(f"{RT.sha(os.path.join(repo, doc + '.md'))}  {doc}.md\n")
    g = {"no unlock file": RT.guard()}
    open(RT.UNLOCK, "w", encoding="utf-8").write("연구21 테스트를 연다\n")
    g["wrong phrase"] = RT.guard()
    open(RT.UNLOCK, "w", encoding="utf-8").write(RT.PHRASE + "\n")
    g["all in place"] = RT.guard()
    open(os.path.join(repo, "SPEC21_ADDENDUM.md"), "a").write("changed\n")
    g["document changed"] = RT.guard()
    open(os.path.join(repo, "SPEC21_ADDENDUM.md"), "w").write("SPEC21_ADDENDUM stand-in\n")
    open(core_dir("stats21.py"), "a").write("changed\n")
    g["frozen file changed"] = RT.guard()
    open(core_dir("stats21.py"), "w").write("stand-in for stats21.py\n")
    txt = open(RT.FREEZE, encoding="utf-8").read()
    open(RT.FREEZE, "w", encoding="utf-8").write("\n".join(ln for ln in txt.split("\n") if "`loader21.py`" not in ln))
    g["core file not listed"] = RT.guard()
    open(RT.FREEZE, "w", encoding="utf-8").write(txt)
    os.rename(RT.DEV_SUMMARY, RT.DEV_SUMMARY + ".away")
    g["no development summary"] = RT.guard()
    os.rename(RT.DEV_SUMMARY + ".away", RT.DEV_SUMMARY)
    g["all in place again"] = RT.guard()
    check("guard of the sealed run: refuses without the unlock file, with another phrase, with a changed document, a changed frozen file, "
          "a core file missing from FREEZE21.md or no development summary; passes only when everything is in place",
          g["all in place"] == [] and g["all in place again"] == [] and g["no unlock file"] == ["no unlock phrase in UNLOCK21.txt"]
          and g["wrong phrase"] == ["no unlock phrase in UNLOCK21.txt"] and g["document changed"] == ["SPEC21_ADDENDUM.md differs from SPEC21_ADDENDUM.sha256"]
          and g["frozen file changed"] == ["frozen file differs: stats21.py"] and g["core file not listed"] == ["FREEZE21.md missing or without the core files"]
          and len(g["no development summary"]) == 2 and g["no development summary"][0] == "frozen file differs: results/dev21_summary.json", guard=g)
    # the development summary is there and readable, but its content changed after FREEZE21.md pinned its sha256
    original = open(RT.DEV_SUMMARY, "rb").read()

    def changed(change):
        d = json.loads(original)
        change(d)
        with open(RT.DEV_SUMMARY, "w") as fh:
            json.dump(d, fh, indent=1, default=str, ensure_ascii=False)
        try:
            s = {f: (x["astar"], x["h2"]) for f, x in RT.selection().items()}       # what the run would read without the sha256 check
        except Exception as e:
            s = f"selection() refuses: {e}"
        return RT.guard(), s

    def other_astar(d):                                                             # E: the other candidate of the tie, H2 as it was
        d["families"]["E"].update(astar="W30_z3_k2_M", astar_signal=[30, 3.0, 2.0])

    def add_h2(d):                                                                  # S: an H2 that the development run did not give
        d["families"]["S"]["h2_exists"] = True
        d["families"]["S"]["selection"]["astar_mean_usd"] = 1.0

    def drop_h2(d):                                                                 # E: its H2 taken away
        d["families"]["E"]["h2_exists"] = False
        d["families"]["E"]["selection"]["astar_mean_usd"] = -1.0

    gc = {"another grid candidate as A* of E": changed(other_astar), "an H2 added for S": changed(add_h2), "the H2 of E removed": changed(drop_h2)}
    with open(RT.DEV_SUMMARY, "wb") as fh:
        fh.write(original + b"\n")
    gc["one line break added"] = (RT.guard(), None)
    changed(other_astar)                                                            # the changed file stays for the sealed run below
    try:
        with contextlib.redirect_stdout(buf):
            RT.main("sealed", 0)
        stopped = "the sealed run started"
    except SystemExit as e:
        stopped = str(e)
    nothing_written = not any(os.path.exists(os.path.join(res, f)) for f in RT.MODES["sealed"]["files"])
    with open(RT.DEV_SUMMARY, "wb") as fh:
        fh.write(original)
    differs = ["frozen file differs: results/dev21_summary.json"]
    check("guard of the sealed run, development summary changed after the freeze (content only; the file is there and readable): refused on "
          "its sha256 alone with another grid candidate as A*, an H2 added, an H2 removed (three selections that selection() itself accepts) "
          "or one line break added; the sealed run stops before it writes anything; with the original bytes back the guard passes",
          all(why == differs for why, _ in gc.values())
          and gc["another grid candidate as A* of E"][1] == {"E": ("W30_z3_k2_M", True), "S": ("W10_z2_k2_M", False)}
          and gc["an H2 added for S"][1] == {"E": ("W30_z2_k2_M", True), "S": ("W10_z2_k2_M", True)}
          and gc["the H2 of E removed"][1] == {"E": ("W30_z2_k2_M", False), "S": ("W10_z2_k2_M", False)}
          and stopped == "refused: " + differs[0] and nothing_written and RT.guard() == [] and open(RT.DEV_SUMMARY, "rb").read() == original,
          guard=gc, sealed_run=stopped)
    # sealed path (synthetic "sealed" days around 2025-11-03; the selection comes from the development summary)
    with contextlib.redirect_stdout(buf):
        out = RT.main("sealed", 0)
    Fe, Fs = out["families"]["E"], out["families"]["S"]
    v = out["verdict"]
    check("small world, sealed path: three hypotheses under Holm (0.05 / 3, 0.025, 0.05), H1-E 56.25 and H1-S 67.5 dollars a day, "
          "H2-E = E's A* under L3D1 (10.5 a day), H2-S absent with the fixed sentence; sentences from SPEC21 6",
          list(out["hypotheses"]) == ["H1-E", "H1-S", "H2-E"] and [v[h]["level"] for h in out["hypotheses"]] == [0.05 / 3, 0.025, 0.05]
          and all(v[h]["rejected"] and v[h]["result"] == "확인" for h in out["hypotheses"])
          and close([v["H1-E"]["mean_daily_usd"], v["H1-S"]["mean_daily_usd"], v["H2-E"]["mean_daily_usd"]], [56.25, 67.5, 10.5], 1e-9)
          and v["H2-S"] == {"result": "없음", "sentence": ST21.no_h2_sentence("S")} and v["H1-E"]["sentence"] == ST21.sentence("H1", "E", True)
          and v["H2-E"]["sentence"] == ST21.sentence("H2", "E", True) and list(Fe["settings"]) == [H1, "W30_z2_k2"] and list(Fs["settings"]) == [H1, "W10_z2_k2"],
          verdict={h: (x.get("p"), x.get("level"), x["result"]) for h, x in v.items()})
    sp = Fe["date_split"][H1]
    per = {k: {part: (sp[k][part]["mean_daily_usd"], sp[k][part]["days"], sp[k][part]["trades"]) for part in ("before", "from")} for k in ("M_L3D1", "M_L0D0", "mid")}
    check("small world, the size unit across 2025-11-03: with sizes stored 100 x from that day, the L3D1 penalty is the same 12 dollars a "
          "trade in both parts (L0D0 23.5 and 17.5, L3D1 5.5 and 2.5 dollars a day; 6 and 5 trades in 4 days each)",
          close([per["M_L0D0"]["before"][0], per["M_L0D0"]["from"][0], per["M_L3D1"]["before"][0], per["M_L3D1"]["from"][0]], [23.5, 17.5, 5.5, 2.5], 1e-9)
          and per["M_L3D1"]["before"][1:] == (4, 6) and per["M_L3D1"]["from"][1:] == (4, 5) and sp["mid_delay5"]["from"]["days"] == 4
          and set(out["control"]["date_split"]) == {"mid", "mid_delay1", "mid_delay5", "mid_delay30"} and out["control"]["pairs"] == ["SA/SD", "SC/SB"], split=per)
    dly = pd.read_csv(os.path.join(res, "test21_daily.csv"))
    e_h1 = dly[(dly.family == "E") & (dly.setting == H1)]
    check("small world, daily file: one row per pair-day and setting (E and S with their two settings, control with one), the columns of "
          "every run; its sums equal the summary; control rows have no M values",
          list(dly.columns) == list(C21.DAILY_COLS) and len(dly) == 16 * 2 + 16 * 2 + 16 and close(e_h1.M_L3D1.sum(), Fe["settings"][H1]["rows"]["M_L3D1"]["total_usd"], 1e-3)
          and close(e_h1.mid.sum(), 450.0, 1e-3) and close(e_h1[e_h1.date >= 20251103].M_L0D0.sum(), 70.0, 1e-3) and int(e_h1.trades_M.sum()) == 11
          and dly[dly.family == "C"].M_L3D1.isna().all() and dly[dly.family != "C"].M_L3D1.notna().all()
          and close(dly[(dly.family == "C")].mid.sum(), out["control"]["rows"]["mid"]["total_usd"], 1e-3), rows=len(dly))
    side = {r["key"]: r for r in out["side_by_side"]}
    try:
        with contextlib.redirect_stdout(buf):
            RT.main("sealed", 0)
        again = False
    except SystemExit as e:
        again = "test results already exist" in str(e)
    check("small world, sealed path: results are written once (summary, report, daily file) and a second run is refused; the side-by-side "
          "table takes research 20's sealed values; the share left after 5 s is reported for E and S",
          again and all(os.path.exists(os.path.join(res, f)) for f in RT.MODES["sealed"]["files"]) and "봉인" in side["R20"]["label"]
          and set(out["ratio_5s"]) == {"E", "S", "E_greater_than_S"} and out["read"]["bars_root"].endswith(os.path.join("sealed", "bars_1s"))
          and out["counts"]["days"] == 8 and "checks" not in out)


def repo_guard_test():
    why = RT.guard()
    check("the real sealed run is refused now (no UNLOCK21.txt; Claude never writes it)", "no unlock phrase in UNLOCK21.txt" in why
          and not os.path.exists(os.path.join(HERE, "UNLOCK21.txt")), reasons=why)


def main():
    r20 = research20_tests()
    frozen_tests()
    with tempfile.TemporaryDirectory(prefix="r21_tests_") as tmp:
        size_tests(tmp)
        kappa_tests()
        control_tests()
        stats_tests()
        core_tests()
        world_tests(tmp)
    repo_guard_test()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    n_ok = sum(r["pass"] for r in RESULTS)
    path = os.path.join(HERE, "results", "engine21_tests.json")
    C21.keep_old(path)
    with open(path, "w") as fh:
        json.dump({"passed": n_ok, "total": len(RESULTS), "research20_tests": r20, "numba": bool(G.HAVE_NUMBA), "tests": RESULTS}, fh, indent=1,
                  default=str, ensure_ascii=False)
    for r in RESULTS:
        if not r["pass"]:
            print("FAIL", json.dumps(r, default=str, ensure_ascii=False)[:1500])
    print(f"{n_ok}/{len(RESULTS)} passed (research 20's tests inside the first: {r20})")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
