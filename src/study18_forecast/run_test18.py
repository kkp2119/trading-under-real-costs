#!/usr/bin/env python3
"""Research 18 sealed test (SPEC18 5, SPEC18_ANSWERS 4, A-21, A-22; step-2 prompt 4), to be run once at step 4:
  python run_test18.py --go           sealed universe B 2021-01-04..2026-11-05 (minus EXCLUDE18)
  python run_test18.py --rehearsal    the same code on the validation data (existing universe 2019-2020)
The sealed run refuses unless study18_forecast/UNLOCK18.txt holds exactly PHRASE (written by Kyle; Claude never creates the file),
no test result exists, every file listed in FREEZE18.md has its sha256, SPEC18.md, SPEC18_ANSWERS.md and model18.json match
their sha256 files and ASTAR equals the development selection (results/dev18_summary.json). The guard runs before any sealed
path is touched. The rehearsal never builds a sealed path and checks that no path it read is under the sealed folder.
Methods S0, S1, S2i, E* (R0), C3 and A* through engine18.symday18; the model is read from model18.json and never refitted.
Verdict (L3D1; day blocks 21, 10,000, seed 18; Holm 0.0167, 0.025, 0.05 on the one-sided bootstrap p, stopping at the first
failure; A* = R0: H1 alone at 0.05):
  H1  AUC > 0.5 on every scored grid point (valid quote, y defined); p = share of bootstrap AUC <= 0.5
  H2  mean IS A* - E* < 0;                 p = share of bootstrap means >= 0
  H3  p95(A*) - p95(E*) < 0 (np.percentile, linear; recomputed on the resampled days); p = share of bootstrap values >= 0
  Bounds: one-sided at the Holm level of each hypothesis (e.g. 98.33% at 0.0167) and the unadjusted one-sided 97.5% bound.
Secondary, no verdict: L0D0, A* - S0, A* - C3, A* - S2i, years and yearly AUC, calibration deciles, the 16-E 3 x 3 cells
(development cut-offs), common stocks only (ANSWERS 1), the delay check (ANSWERS 4: A*_delay - E*_delay, mean and p95, L3D1
and L0D0). Results are written once at the end."""
import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "study15_ml_execution"))
import engine18 as G  # noqa: E402
import loader15 as L  # noqa: E402  (development bars, rehearsal only)

RES = os.path.join(HERE, "results")
ASTAR = "R0"                                                          # development selection (FREEZE18.md): E* itself
UNLOCK = os.path.join(HERE, "UNLOCK18.txt")
PHRASE = "연구18 테스트를 한 번 연다"
FREEZE = os.path.join(HERE, "FREEZE18.md")
MODEL = os.path.join(HERE, "model18.json")
DEV_SUMMARY = os.path.join(RES, "dev18_summary.json")
SEALED_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research18_data/sealed")
SEALED_BARS = os.path.join(SEALED_ROOT, "bars_1s_new_stocks")
SEALED_MANIFEST = os.path.join(HERE, "SEALED_MANIFEST18.tsv")
EXCLUDE = os.path.join(HERE, "EXCLUDE18.tsv")
CALENDAR = os.path.join(HERE, "calendar18.csv")
UNIVERSE = os.path.join(HERE, "universe18_{y}.tsv")
WINDOW = ("20210104", "20261105")
VALID_YEARS = (2019, 2020)
MODES = {"sealed": {"files": ("test18_summary.json", "test18_report.md"), "title": "봉인 테스트 (새 유니버스 B 2021–2026)"},
         "rehearsal": {"files": ("rehearsal18_summary.json", "rehearsal18_report.md"),
                       "title": "리허설 (검증 데이터: 기존 유니버스 2019–2020, 판정 아님)"}}
NONCOMMON = {"sealed": {(2021, "AEM"), (2021, "PRGO"), (2021, "XP"), (2022, "AER"), (2023, "LOGI"), (2023, "TGH"), (2023, "AER"),
                        (2023, "TNK"), (2023, "AGO"), (2023, "FLNG"), (2024, "ADNT"), (2024, "BLCO"), (2024, "NTB"), (2024, "KRC"),
                        (2024, "REG"), (2025, "LOGI"), (2025, "FDP"), (2025, "ADNT"), (2025, "FROG"), (2026, "FLEX"), (2026, "TIGO"),
                        (2026, "HIW")},                                                    # ANSWERS 1: 19 foreign, 3 REIT
             "rehearsal": {(2019, "CATM"), (2019, "VNOM"), (2020, "DOX")}}               # same rule on the validation lists
SENTENCES = {"H1": {True: "호가 잔량 불균형은 처음 보는 종목에서도 다음 가격 움직임을 예측했다",
                    False: "개발 데이터의 예측력은 새 종목에서 재현되지 않았다"},
             "H2": {True: "예측으로 기다릴지 정하면 단순한 인내보다 더 싸다", False: "예측만으로는 평균 비용이 확실히 줄지 않았다"},
             "H3": {True: "예측이 최악의 주문 비용을 줄였다", False: "최악 5%의 비용은 줄지 않았다"}}
CORE = ("engine18.py", "run_test18.py", "model18.json", "SEALED_MANIFEST18.tsv", "EXCLUDE18.tsv", "calendar18.csv")


class Refused(Exception):
    pass


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- guard (sealed run only, before any sealed path)
def frozen_files():
    """[(path relative to study18_forecast, sha256)] from the FREEZE18.md table rows | `path` | `sha256` |."""
    if not os.path.isfile(FREEZE):
        return []
    pat = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
    return [m.groups() for m in map(pat.match, open(FREEZE, encoding="utf-8")) if m]


def guard():
    why = []
    if any(os.path.exists(os.path.join(RES, f)) for f in MODES["sealed"]["files"]):
        why.append("test results already exist")
    if not os.path.isfile(UNLOCK) or open(UNLOCK, encoding="utf-8").read().strip() != PHRASE:
        why.append("no unlock phrase in UNLOCK18.txt")
    ff = frozen_files()
    if not ff or not all(any(p == c for p, _ in ff) for c in CORE):
        why.append("FREEZE18.md missing or without the core files")
    for p, h in ff:
        q = os.path.normpath(os.path.join(HERE, p))
        if not os.path.isfile(q) or sha(q) != h:
            why.append(f"frozen file differs: {p}")
    for doc, path in (("SPEC18", "SPEC18.md"), ("SPEC18_ANSWERS", "SPEC18_ANSWERS.md"), ("model18", "model18.json")):
        if open(os.path.join(HERE, f"{doc}.sha256")).read().split()[0] != sha(os.path.join(HERE, path)):
            why.append(f"{path} differs from {doc}.sha256")
    if ASTAR not in G.CANDS or not os.path.isfile(DEV_SUMMARY) or json.load(open(DEV_SUMMARY))["astar"] != ASTAR:
        why.append("ASTAR differs from the development selection")
    return why


# ---------------------------------------------------------------- sources
def sealed_plan(read):
    """[('sealed', year, SEALED_MANIFEST18 row, close bar)] in the window, universe symbols only, minus EXCLUDE18; calendars."""
    for p in (SEALED_MANIFEST, EXCLUDE, CALENDAR):
        read.append(p)
    m = pd.read_csv(SEALED_MANIFEST, sep="\t", dtype=str, keep_default_na=False)
    ex = pd.read_csv(EXCLUDE, sep="\t", dtype=str, keep_default_na=False)
    exk = set(zip(ex.sym, ex.date))
    c = pd.read_csv(CALENDAR, dtype=str)
    cb = {d.replace("-", ""): L.NSEC - ((16 - int(t[:2])) * 3600 - int(t[3:5]) * 60) for d, t in zip(c.date, c.close_time)}
    uni = {}
    for y in sorted({d[:4] for d in m.date}):
        read.append(UNIVERSE.format(y=y))
        uni[y] = set(pd.read_csv(UNIVERSE.format(y=y), sep="\t", dtype=str).sym)
    m = m[(m.date >= WINDOW[0]) & (m.date <= WINDOW[1])]
    keep = [s in uni[d[:4]] and (s, d) not in exk for s, d in zip(m.sym, m.date)]
    tasks = [("sealed", int(r["date"][:4]), r, cb[r["date"]]) for r in m[keep].to_dict("records")]
    tasks.sort(key=lambda t: (t[1], t[2]["sym"], t[2]["date"]))
    cal = {}
    for d in sorted(cb):
        if WINDOW[0] <= d <= WINDOW[1]:
            cal.setdefault(int(d[:4]), []).append(d)
    info = {"manifest_rows": int(len(pd.read_csv(SEALED_MANIFEST, sep="\t", dtype=str))), "excluded_rows": int(len(exk)),
            "excluded_in_window": int(sum(1 for s, d in exk if WINDOW[0] <= d <= WINDOW[1])), "tasks": len(tasks)}
    return tasks, cal, info


def dev_plan(read):
    """[('dev', year, loader15 MANIFEST row, close bar)] for the validation years 2019-2020 (the dev18 order); calendars."""
    tasks, cal = [], {}
    for y in VALID_YEARS:
        src = L.SOURCES[L.source_of(y)]
        read += [os.path.join(src[0], "MANIFEST.tsv"), src[2].format(y=y), src[3]]
        man, cb = L.manifest(y), L.close_bars(y)
        tasks += [("dev", y, r, cb[d]) for (s, d), r in sorted(man.items())]
        cal[y] = sorted(cb)
    return tasks, cal, {"tasks": len(tasks)}


def load_sealed(row):
    """run_test15.load_bar checks: sha256 against SEALED_MANIFEST18, 21 int32 columns x 23,400 rows, volume identity."""
    path = os.path.join(SEALED_BARS, row["file"])
    if sha(path) != row["sha256"]:
        raise Refused(f"{row['sym']} {row['date']}: sha256 differs from SEALED_MANIFEST18")
    with np.load(path, allow_pickle=False) as z:
        if sorted(z.files) != sorted(L.COLS):
            raise Refused(f"{row['sym']} {row['date']}: columns")
        b = {}
        for c in L.COLS:
            if z[c].dtype != np.int32 or z[c].shape != (L.NSEC,):
                raise Refused(f"{row['sym']} {row['date']}: {c} dtype/shape")
            b[c] = z[c].astype(np.int64)
    if not np.array_equal(b["vol"], b["vol_bid"] + b["vol_ask"] + b["vol_in"] + b["vol_unc"]):
        raise Refused(f"{row['sym']} {row['date']}: vol identity")
    return b, path


def runs_for(astar):
    extra = () if astar == "R0" else (astar,)
    return ("S0", "S1", "S2i", "R0") + extra, ("R0",) + extra


def worker(t):
    kind, year, row, close_bar, model, astar = t
    if kind == "sealed":
        b, path = load_sealed(row)
    else:
        b, path = L.load(year, row), os.path.join(L.SOURCES[L.source_of(year)][0], row["file"])
    runs, delay = runs_for(astar)
    meta, outs = G.symday18(b, row["date"], close_bar, model, runs, delay)
    meta.pop("X")
    return row["sym"], row["date"], meta, outs, path


# ---------------------------------------------------------------- analysis
def bounds(dist, lvl, lower):
    """One-sided bound at level lvl (lower for H1, upper for H2 / H3) and the unadjusted one-sided 97.5% bound."""
    q = (100 * lvl, 2.5) if lower else (100 * (1 - lvl), 97.5)
    return float(np.percentile(dist, q[0])), float(np.percentile(dist, q[1]))


def test_stats(IS_a, IS_e, P, y, keep, pcodes, gcodes, W):
    """{'H1': (AUC, dist), 'H2': (mean, dist), 'H3': (p95 diff, dist)} on one sample (parents with day codes pcodes, scored
    grid points keep with day codes gcodes)."""
    auc, adist = G.auc_boot(P[keep], y[keep].astype(np.int8), gcodes[keep], W)
    s = np.bincount(pcodes, weights=IS_a - IS_e, minlength=W.shape[1])
    n = np.bincount(pcodes, minlength=W.shape[1]).astype(np.float64)
    mdist = (W @ s) / (W @ n)
    pe, pdist = G.p95_diff_boot(IS_a, IS_e, pcodes, W)
    return {"H1": (auc, adist), "H2": (float((IS_a - IS_e).mean()), mdist), "H3": (pe, pdist)}


def pvalue(h, est, dist):
    return float(np.mean(dist <= 0.5)) if h == "H1" else float(np.mean(dist >= 0))


def cell_stats(d, mask, codes, n_days):
    if mask.sum() == 0:
        return {"n": 0}
    c = codes[mask]
    T = int((np.bincount(c, minlength=n_days) > 0).sum())
    est, lo, hi = G.diff_ci(d[mask], c, n_days, G.boot_index(T))
    return {"n": int(mask.sum()), "days": T, "mean_c": est, "ci_lo": lo, "ci_hi": hi}


def auc_point(P, y, codes):
    if len(y) == 0 or y.min() == y.max():
        return None
    c = pd.Categorical(codes).codes.astype(np.int64)
    return G.auc_boot(P, y.astype(np.int8), c, np.ones((1, int(c.max()) + 1)))[0]


def deciles(p, y):
    edges = np.quantile(p, np.linspace(0, 1, 11))
    k = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, 9)
    return [{"decile": int(i + 1), "n": int((k == i).sum()), "mean_p": float(p[k == i].mean()), "mean_y": float(y[k == i].mean())}
            for i in range(10)]


def analyse(par, OUT, grid, dayvol, cal, mode, astar):
    days = sorted(par.date.unique())
    codes = pd.Categorical(par.date, categories=days).codes.astype(np.int64)
    T = len(days)
    idx = G.boot_index(T)
    W = G.day_counts(idx, T)
    yr = par.date.str[:4].astype(int).to_numpy()
    P, y, valid = grid["P"], grid["y"], grid["valid"]
    keep = valid & (y >= 0)
    gcodes = np.repeat(codes, G.NG).reshape(P.shape)
    gyr = np.repeat(yr, G.NG).reshape(P.shape)
    runs, delay = runs_for(astar)
    names = list(runs) + [f"{m}@d1" for m in delay]
    hyp = ("H1",) if astar == "R0" else ("H1", "H2", "H3")
    common = np.array([(a, s) not in NONCOMMON[mode] for a, s in zip(yr, par.sym)])
    adv = G.adv_prior(dayvol, cal, sorted(set(zip(par.sym, par.date))))
    tv = G.tercile(par.vol_pre.to_numpy(), G.CUT_VOL)
    ta = G.tercile(np.array([adv[(s, d)][0] for s, d in zip(par.sym, par.date)]), G.CUT_ADV)
    out = {"astar": astar, "hypotheses": {"H1": "AUC > 0.5", "H2": f"{astar} - R0 < 0", "H3": f"p95({astar}) - p95(R0) < 0"},
           "variants": {}}
    for lab, lr, dep in G.VARIANTS:
        o = OUT[G.LRULES.index(lr)]
        outs = {m: o[names.index(m)] for m in runs}
        outs["C3"] = np.where(G.branch_c3(par.spread.to_numpy())[:, None], outs["S2i"], outs["R0"])
        dirs, mid2 = par.dir.to_numpy(), par.mid2.to_numpy()
        IS = {m: G.costs(v, dirs, mid2, dep) for m, v in outs.items()}
        ISd = {m: G.costs(o[names.index(f"{m}@d1")], dirs, mid2, dep) for m in delay}
        ms = ("S0", "S1", "S2i", "C3", "R0") + (() if astar == "R0" else (astar,))
        v = {"methods": {m: {"mean_is_c": float(IS[m].mean()), "p95_is_c": float(np.percentile(IS[m], 95)),
                             "limit_frac": float((outs[m][:, 0] / G.QTY).mean()),
                             "fired_frac": float(G.fired(outs[m], m).mean()) if m in G.CANDS else 0.0,
                             "r1_parent_frac": float((outs[m][:, 9] > 0).mean()), "r2_parent_frac": float((outs[m][:, 10] > 0).mean()),
                             "market_nonpositive_price": int(outs[m][:, 11].sum())} for m in ms},
             "tests": {}, "secondary": {}, "yearly": {}, "cells": [], "common_stocks_only": {}, "delay": {}}
        st = test_stats(IS[astar], IS["R0"], P, y, keep, codes, gcodes, W)
        for h in hyp:
            est, dist = st[h]
            v["tests"][h] = {"estimate": est, "p_one_sided": pvalue(h, est, dist), "boot_se": float(dist.std(ddof=1)),
                             "ci95": [float(x) for x in np.percentile(dist, [2.5, 97.5])], "dist": dist}
        if common.any():
            cd = sorted(set(par.date[common]))
            cc = pd.Categorical(par.date[common], categories=cd).codes.astype(np.int64)
            Wc = G.day_counts(G.boot_index(len(cd)), len(cd))
            gc = np.repeat(cc, G.NG).reshape(int(common.sum()), G.NG)
            sc = test_stats(IS[astar][common], IS["R0"][common], P[common], y[common], keep[common], cc, gc, Wc)
            v["common_stocks_only"] = {"n_parents": int(common.sum()), "days": len(cd),
                                       "excluded_symbol_years": sorted(f"{a} {s}" for a, s in NONCOMMON[mode]),
                                       **{h: {"estimate": sc[h][0], "p_one_sided": pvalue(h, *sc[h]),
                                              "bound_97_5": bounds(sc[h][1], 0.025, h == "H1")[1]} for h in hyp}}
        for name, (a, b) in (("A* - S0", (astar, "S0")), ("A* - C3", (astar, "C3")), ("A* - S2i", (astar, "S2i")), ("R0 - S0", ("R0", "S0"))):
            est, lo, hi = G.diff_ci(IS[a] - IS[b], codes, T, idx)
            v["secondary"][name] = {"mean_c": est, "ci95": [lo, hi], "p95_diff_c": float(np.percentile(IS[a], 95) - np.percentile(IS[b], 95))}
        for x in sorted(set(yr)):
            m_ = yr == x
            gm = keep & (gyr == x)
            v["yearly"][str(x)] = {"n_parents": int(m_.sum()), "H2_mean_c": float((IS[astar] - IS["R0"])[m_].mean()),
                                   "H3_p95_diff_c": float(np.percentile(IS[astar][m_], 95) - np.percentile(IS["R0"][m_], 95)),
                                   "auc": auc_point(P[gm], y[gm], gcodes[gm])}
        incl = (tv >= 0) & (ta >= 0)
        cells = [(f"vol {G.TERC[i]} / adv {G.TERC[j]}", incl & (tv == i) & (ta == j)) for i in range(3) for j in range(3)]
        for cname, m_ in cells + [("excluded (no vol_pre or adv_prior)", ~incl)]:
            gm = keep & m_[:, None]
            v["cells"].append({"cell": cname, "auc": auc_point(P[gm], y[gm], gcodes[gm]),
                               **({"H2": cell_stats(IS[astar] - IS["R0"], m_, codes, T)} if astar != "R0" else {})})
        for name, (a, b) in (("A*_delay - E*_delay", (ISd[astar], ISd["R0"])), ("E*_delay - E*", (ISd["R0"], IS["R0"])),
                             ("A*_delay - A*", (ISd[astar], IS[astar]))):
            est, lo, hi = G.diff_ci(a - b, codes, T, idx)
            pe, pdist = G.p95_diff_boot(a, b, codes, W)
            v["delay"][name] = {"mean_c": est, "ci95": [lo, hi], "p95_diff_c": pe,
                                "p95_ci95": [float(x) for x in np.percentile(pdist, [2.5, 97.5])]}
        out["variants"][lab] = v
    prim = out["variants"]["L3D1"]["tests"]
    holm = G.holm({h: prim[h]["p_one_sided"] for h in hyp})
    out["verdict"] = {}
    for h in hyp:
        x = holm[h]
        b_lvl, b975 = bounds(prim[h]["dist"], x["level"], h == "H1")
        out["verdict"][h] = {**x, "result": "확인" if x["rejected"] else "확인 안 됨", "estimate": prim[h]["estimate"],
                             "bound_at_level": b_lvl, "bound_97_5": b975, "sentence": SENTENCES[h][x["rejected"]]}
    if astar == "R0":
        for h in ("H2", "H3"):
            out["verdict"][h] = {"result": "없음", "sentence": "Q2·Q3는 개발에서 개선 없음 (A* = R0)"}
    out["calibration"] = deciles(P[keep], y[keep])
    for lab in out["variants"]:
        for h in out["variants"][lab]["tests"]:
            out["variants"][lab]["tests"][h].pop("dist")
    out["counts"] = {"n_parents": int(len(par)), "n_days": T, "n_symbol_days": int(par[["date", "sym"]].drop_duplicates().shape[0]),
                     "grid_points": int(P.size), "valid_points": int(valid.sum()), "scored_points": int(keep.sum()),
                     "y_missing_share_of_valid": float((valid & (y < 0)).sum() / valid.sum()), "vol_pre_undefined": int(par.vol_pre.isna().sum())}
    return out


# ---------------------------------------------------------------- rehearsal checks (development numbers of step 2)
def rehearsal_checks(out, read):
    dev = json.load(open(DEV_SUMMARY))
    rows = {(r["variant"], r["method"]): r for r in dev["rows"]}
    a = out["astar"]
    diffs = []
    for lab in ("L3D1", "L0D0"):
        v = out["variants"][lab]
        for m, r in v["methods"].items():
            d = rows[(lab, m)]
            diffs += [abs(r["mean_is_c"] - d["mean_is_c"]), abs(r["p95_is_c"] - d["p95_is_c"]), abs(r["limit_frac"] - d["limit_frac"]),
                      abs(r["fired_frac"] - d["fired_frac"])]
        t = v["tests"]
        diffs.append(abs(t["H1"]["estimate"] - dev["prediction"]["valid"]["auc"]))
        diffs.append(abs(t["H1"]["boot_se"] - dev["prediction"]["valid"]["auc_boot"]["se"]))
        if "H2" in t:
            diffs += [abs(t["H2"]["estimate"] - rows[(lab, a)]["minus_R0_c"]), abs(t["H2"]["ci95"][0] - rows[(lab, a)]["minus_R0_lo"]),
                      abs(t["H2"]["ci95"][1] - rows[(lab, a)]["minus_R0_hi"]), abs(t["H3"]["estimate"] - rows[(lab, a)]["p95_minus_R0_c"]),
                      abs(t["H3"]["ci95"][0] - rows[(lab, a)]["p95_minus_R0_lo"]), abs(t["H3"]["ci95"][1] - rows[(lab, a)]["p95_minus_R0_hi"])]
            diffs += [abs(x["H2_mean_c"] - rows[(lab, a)][f"minus_R0_{yy}"]) for yy, x in v["yearly"].items()]
        dd = dev["delay"][lab]["astar_delay_minus_R0_delay"]
        x = v["delay"]["A*_delay - E*_delay"]
        diffs += [abs(x["mean_c"] - dd["mean_c"]), abs(x["p95_diff_c"] - dd["p95_diff_c"]), abs(x["ci95"][0] - dd["ci95"][0]),
                  abs(x["p95_ci95"][1] - dd["p95_ci95"][1])]
    for yy, x in out["variants"]["L3D1"]["yearly"].items():
        diffs.append(abs(x["auc"] - dev["prediction"]["valid"]["yearly_auc"][yy]["auc"]))
    for k, x in enumerate(out["calibration"]):
        d = dev["prediction"]["valid"]["deciles"][k]
        diffs += [abs(x["mean_p"] - d["mean_p"]), abs(x["mean_y"] - d["mean_y"]), abs(x["n"] - d["n"])]
    chk = {"dev18_max_abs_diff": float(max(diffs)), "n_compared": len(diffs)}
    chk["dev18_numbers_equal"] = chk["dev18_max_abs_diff"] < 1e-9
    chk["no_sealed_path_read"] = not any(os.path.abspath(p).startswith(SEALED_ROOT) for p in read)
    chk["pass"] = bool(chk["dev18_numbers_equal"] and chk["no_sealed_path_read"])
    return chk


# ---------------------------------------------------------------- report
def report(out, mode, path):
    a = out["astar"]
    L_ = [f"# 연구 18 · {MODES[mode]['title']}", "",
          f"- 주문 {out['counts']['n_parents']:,}건, {out['counts']['n_days']}일, 종목·일 {out['counts']['n_symbol_days']:,}개, "
          f"채점 격자 지점 {out['counts']['scored_points']:,}개. A* = {a}.",
          "- 판정(L3D1): 일 블록 21, 10,000회, seed 18의 단측 부트스트랩 p에 Holm(0.0167, 0.025, 0.05). 한계는 각 가설의 Holm 수준 단측 한계와 보정 전 단측 97.5% 한계.", ""]
    if mode == "rehearsal":
        L_ += ["**리허설: 검증 데이터라서 아래 \"확인\" 여부와 문장은 코드 경로 점검용이며 판정이 아니다.**", ""]
    L_ += ["## 판정", "", "| 가설 | 내용 | 추정값 | p | Holm 수준 | 수준 한계 | 97.5% 한계 | 결과 | 문장 |", "|---|---|---|---|---|---|---|---|---|"]
    for h, x in out["verdict"].items():
        if "p" in x:
            f = (lambda z: f"{z:.4f}") if h == "H1" else (lambda z: f"{z:+.4f}")
            L_.append(f"| {h} | {out['hypotheses'][h]} | {f(x['estimate'])} | {x['p']:.4f} | {x['level']:.4f} | {f(x['bound_at_level'])} | "
                      f"{f(x['bound_97_5'])} | {x['result']} | \"{x['sentence']}\" |")
        else:
            L_.append(f"| {h} | — | — | — | — | — | — | {x['result']} | {x['sentence']} |")
    for lab in ("L3D1", "L0D0"):
        v = out["variants"][lab]
        L_ += ["", f"## {lab}: 방법별 평균 IS (p95) · 지정가 · 발동", "", "| 방법 | 평균 IS | p95 | 지정가 | 발동 | R1 | R2 |", "|---|---|---|---|---|---|---|"]
        for m, r in v["methods"].items():
            L_.append(f"| {m} | {r['mean_is_c']:.3f} | {r['p95_is_c']:.2f} | {r['limit_frac']:.3f} | {r['fired_frac']:.3f} | {r['r1_parent_frac']:.3f} | "
                      f"{r['r2_parent_frac']:.3f} |")
        L_ += ["", "| 비교 | 추정값 | 95% CI | 단측 p |", "|---|---|---|---|"]
        for h, x in v["tests"].items():
            L_.append(f"| {h}{'' if lab == 'L3D1' else ' (보조)'} | {x['estimate']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | {x['p_one_sided']:.4f} |")
        for n, x in v["secondary"].items():
            L_.append(f"| {n} 평균 (보조; p95 차이 {x['p95_diff_c']:+.2f}) | {x['mean_c']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | |")
        c = v["common_stocks_only"]
        if c:
            for h in [k for k in c if k.startswith("H")]:
                L_.append(f"| {h} 보통주만 (보조, {c['n_parents']:,}건) | {c[h]['estimate']:+.4f} | 97.5% 한계 {c[h]['bound_97_5']:+.4f} | {c[h]['p_one_sided']:.4f} |")
        for n, x in v["delay"].items():
            L_.append(f"| 지연: {n} (보조) | 평균 {x['mean_c']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | p95 차이 {x['p95_diff_c']:+.2f} "
                      f"[{x['p95_ci95'][0]:+.2f}, {x['p95_ci95'][1]:+.2f}] |")
        L_ += ["", f"### {lab} 연도별", "", "| 연도 | 주문 | A* − R0 평균 | p95 차이 | AUC |", "|---|---|---|---|---|"]
        for yy, x in v["yearly"].items():
            au = "—" if x["auc"] is None else f"{x['auc']:.4f}"
            L_.append(f"| {yy} | {x['n_parents']:,} | {x['H2_mean_c']:+.3f} | {x['H3_p95_diff_c']:+.2f} | {au} |")
        L_ += ["", f"### {lab} 3 × 3 칸 (16-E 개발 기준값)", "", "| 칸 | AUC | A* − R0 [95% CI] (n) |", "|---|---|---|"]
        for c in v["cells"]:
            h2 = c.get("H2")
            s2 = (f"{h2['mean_c']:+.3f} [{h2['ci_lo']:+.3f}, {h2['ci_hi']:+.3f}] ({h2['n']:,})" if h2 and h2["n"] else "—")
            au = "—" if c["auc"] is None else f"{c['auc']:.4f}"
            L_.append(f"| {c['cell']} | {au} | {s2} |")
    L_ += ["", "## 보정표 (10분위)", "", "| 분위 | n | 평균 p | 평균 y |", "|---|---|---|---|"]
    for x in out["calibration"]:
        L_.append(f"| {x['decile']} | {x['n']:,} | {x['mean_p']:.4f} | {x['mean_y']:.4f} |")
    if "checks" in out:
        c = out["checks"]
        L_ += ["", "## 리허설 점검", "", f"- 2단계 개발 숫자 {c['n_compared']}개와 최대 차이 {c['dev18_max_abs_diff']:.1e} → "
               f"{'일치' if c['dev18_numbers_equal'] else '불일치'}", f"- 봉인 경로를 읽지 않음: {c['no_sealed_path_read']}"]
    L_ += ["", f"- 읽은 파일: 막대 {out['read']['bar_files']:,}개, 그 밖 {len(out['read']['other'])}개. 실행 {out['runtime_s']} s"]
    with open(path, "w") as fh:
        fh.write("\n".join(L_) + "\n")


# ---------------------------------------------------------------- run
def main(mode, workers):
    t0 = time.time()
    if mode == "sealed":
        why = guard()
        if why:
            raise SystemExit("refused: " + "; ".join(why))
    elif ASTAR not in G.CANDS:
        raise SystemExit("ASTAR is not set")
    model = json.load(open(MODEL))
    read = [MODEL]
    tasks, cal, info = sealed_plan(read) if mode == "sealed" else dev_plan(read)
    with mp.get_context("spawn").Pool(workers) as pool:
        res = list(pool.imap(worker, [(k, y, r, cb, model, ASTAR) for k, y, r, cb in tasks], chunksize=8))
    bars = [r[4] for r in res]
    par = pd.DataFrame({"date": np.concatenate([[d] * len(m["dir"]) for _, d, m, _, _ in res]),
                        "sym": np.concatenate([[s] * len(m["dir"]) for s, _, m, _, _ in res]),
                        **{k: np.concatenate([m[k] for _, _, m, _, _ in res]) for k in ("dir", "start", "mid2", "spread", "vol_pre")}})
    OUT = np.concatenate([r[3] for r in res], axis=2)
    grid = {k: np.concatenate([m[k] for _, _, m, _, _ in res]) for k in ("P", "y", "valid")}
    dayvol = {(s, d): m["day_vol"] for s, d, m, _, _ in res}
    out = analyse(par, OUT, grid, dayvol, cal, mode, ASTAR)
    out.update({"mode": mode, "plan": info, "dropped_parents": int(sum(m["n_dropped"] for _, _, m, _, _ in res)),
                "grid_counts": {k: int(sum(m["counts"][k] for _, _, m, _, _ in res)) for k in res[0][2]["counts"]},
                "model_sha256": sha(MODEL), "read": {"bar_files": len(bars), "other": read}, "runtime_s": round(time.time() - t0, 1),
                "written": dt.datetime.now().isoformat(timespec="seconds")})
    if mode == "rehearsal":
        out["checks"] = rehearsal_checks(out, read + bars)
    files = MODES[mode]["files"]
    if mode == "sealed" and any(os.path.exists(os.path.join(RES, f)) for f in files):
        raise SystemExit("refused: test results appeared during the run")
    with open(os.path.join(RES, files[0]), "w") as fh:
        json.dump(out, fh, indent=1, default=str, ensure_ascii=False)
    report(out, mode, os.path.join(RES, files[1]))
    print(open(os.path.join(RES, files[1])).read())
    if mode == "rehearsal" and not out["checks"]["pass"]:
        raise SystemExit("rehearsal check failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--go", action="store_true", help="the sealed test, once")
    g.add_argument("--rehearsal", action="store_true", help="the same code on the validation data")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main("sealed" if a.go else "rehearsal", a.workers)
