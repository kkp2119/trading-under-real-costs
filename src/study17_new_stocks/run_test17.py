#!/usr/bin/env python3
"""Research 17 sealed test (SPEC17 4, SPEC17_ANSWERS 8; step-2 prompt 3), to be run once at step 4:
  python run_test17.py --go           sealed new universe 2021-01-04..2026-11-03 (minus EXCLUDE17)
  python run_test17.py --rehearsal    the same code on the development data (existing universe 2015-2020)
The sealed run refuses unless research17/UNLOCK17.txt holds exactly PHRASE (written by Kyle; Claude never creates the file),
no test result exists, every file listed in FREEZE17.md has its sha256, SPEC17.md and SPEC17_ANSWERS.md match their sha256
files and ASTAR equals the development selection (results/dev17_summary.json). The guard runs before any sealed path is
touched. The rehearsal never builds a sealed path and checks that no path it read is under the sealed folder.
Methods S0, S1, S2i, C0 (E*) and A* through engine17. Verdict (L3D1): H1 C0 - S0 < 0, H2 A* - C0 < 0; one-sided bootstrap
p = share of bootstrap means >= 0 (day blocks 21, 10,000, seed 17), Holm at 0.05 (0.025 then 0.05), one-sided 97.5% upper
bounds. Secondary, no verdict: L0D0, A* - S2i, C0 - S1, p95, years, the 16-E 3 x 3 cells (development cut-offs), common
stocks only (ANSWERS 2). Results are written once at the end."""
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
import engine17 as G  # noqa: E402
import loader15 as L  # noqa: E402  (development bars, rehearsal only)

RES = os.path.join(HERE, "results")
ASTAR = "C3"                                                          # development selection (FREEZE17.md)
UNLOCK = os.path.join(HERE, "UNLOCK17.txt")
PHRASE = "연구17 테스트를 한 번 연다"
FREEZE = os.path.join(HERE, "FREEZE17.md")
DEV_SUMMARY = os.path.join(RES, "dev17_summary.json")
SEALED_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research17_data/sealed")
SEALED_BARS = os.path.join(SEALED_ROOT, "bars_1s_new_stocks")
SEALED_MANIFEST = os.path.join(HERE, "SEALED_MANIFEST.tsv")
EXCLUDE = os.path.join(HERE, "EXCLUDE17.tsv")
CALENDAR = os.path.join(HERE, "calendar17.csv")
UNIVERSE = os.path.join(HERE, "universe17_{y}.tsv")
WINDOW = ("20210104", "20261103")
DEV_YEARS = tuple(range(2015, 2021))
MODES = {"sealed": {"files": ("test17_summary.json", "test17_report.md"), "title": "봉인 테스트 (새 유니버스 2021–2026)"},
         "rehearsal": {"files": ("rehearsal17_summary.json", "rehearsal17_report.md"),
                       "title": "리허설 (개발 데이터: 기존 유니버스 2015–2020, 판정 아님)"}}
NONCOMMON = {"sealed": {(2021, "TRTN"), (2021, "CCC"), (2022, "NNN"), (2022, "CLB"), (2022, "CONE"), (2022, "FCPT"),
                        (2023, "FTAI"), (2023, "MAXN"), (2023, "SJT"), (2023, "CEQP"), (2023, "LTC"), (2023, "WES"),
                        (2025, "BXP"), (2025, "CNI"), (2026, "SPNT"), (2026, "ARE"), (2026, "GIL"), (2026, "BEP"),
                        (2026, "RY")},                                                                  # ANSWERS 2
             "rehearsal": {(2015, "BG"), (2017, "ALLE"), (2017, "CHKP"), (2018, "WCN"), (2019, "CATM"), (2019, "VNOM"),
                           (2020, "DOX")}}                                       # same rule on the development lists
SENTENCES = {"H1": {True: "E*의 절감은 처음 보는 종목에서도 재현됐다", False: "E*의 절감은 원래 고른 종목에 한정됐을 수 있다"},
             "H2": {True: "시장 상황에 맞춰 인내 수준을 바꾸면 E*보다 더 싸다", False: "상황 맞춤형 규칙은 단순한 인내보다 확실히 낫지 않았다"}}
CORE = ("engine17.py", "run_test17.py", "SEALED_MANIFEST.tsv", "EXCLUDE17.tsv", "calendar17.csv")


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
    """[(path relative to research17, sha256)] from the FREEZE17.md table rows | `path` | `sha256` |."""
    if not os.path.isfile(FREEZE):
        return []
    pat = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
    return [m.groups() for m in map(pat.match, open(FREEZE, encoding="utf-8")) if m]


def guard():
    why = []
    if any(os.path.exists(os.path.join(RES, f)) for f in MODES["sealed"]["files"]):
        why.append("test results already exist")
    if not os.path.isfile(UNLOCK) or open(UNLOCK, encoding="utf-8").read().strip() != PHRASE:
        why.append("no unlock phrase in UNLOCK17.txt")
    ff = frozen_files()
    if not ff or not all(any(p == c for p, _ in ff) for c in CORE):
        why.append("FREEZE17.md missing or without the core files")
    for p, h in ff:
        q = os.path.normpath(os.path.join(HERE, p))
        if not os.path.isfile(q) or sha(q) != h:
            why.append(f"frozen file differs: {p}")
    for doc in ("SPEC17", "SPEC17_ANSWERS"):
        if open(os.path.join(HERE, f"{doc}.sha256")).read().split()[0] != sha(os.path.join(HERE, f"{doc}.md")):
            why.append(f"{doc}.md differs from {doc}.sha256")
    if not os.path.isfile(DEV_SUMMARY) or json.load(open(DEV_SUMMARY))["astar"] != ASTAR:
        why.append("ASTAR differs from the development selection")
    return why


# ---------------------------------------------------------------- sources
def sealed_plan(read):
    """[('sealed', year, SEALED_MANIFEST row, close bar)] in the window, universe symbols only, minus EXCLUDE17; calendars."""
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
    """[('dev', year, loader15 MANIFEST row, close bar)] for 2015-2020 (the dev17 order); calendars."""
    tasks, cal = [], {}
    for y in DEV_YEARS:
        src = L.SOURCES[L.source_of(y)]
        read += [os.path.join(src[0], "MANIFEST.tsv"), src[2].format(y=y), src[3]]
        man, cb = L.manifest(y), L.close_bars(y)
        tasks += [("dev", y, r, cb[d]) for (s, d), r in sorted(man.items())]
        cal[y] = sorted(cb)
    return tasks, cal, {"tasks": len(tasks)}


def load_sealed(row):
    """run_test15.load_bar checks: sha256 against SEALED_MANIFEST, 21 int32 columns x 23,400 rows, volume identity."""
    path = os.path.join(SEALED_BARS, row["file"])
    if sha(path) != row["sha256"]:
        raise Refused(f"{row['sym']} {row['date']}: sha256 differs from SEALED_MANIFEST")
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


def worker(t):
    kind, year, row, close_bar = t
    if kind == "sealed":
        b, path = load_sealed(row)
    else:
        b, path = L.load(year, row), os.path.join(L.SOURCES[L.source_of(year)][0], row["file"])
    meta, outs = G.symday(b, row["date"], close_bar)
    return row["sym"], row["date"], meta, outs, path


# ---------------------------------------------------------------- analysis
def cell_stats(d, mask, codes, n_days):
    if mask.sum() == 0:
        return {"n": 0}
    c = codes[mask]
    T = int((np.bincount(c, minlength=n_days) > 0).sum())
    est, lo, hi = G.diff_ci(d[mask], c, n_days, G.boot_index(T))
    return {"n": int(mask.sum()), "days": T, "mean_c": est, "ci_lo": lo, "ci_hi": hi}


def analyse(par, OUT, dayvol, cal, mode, astar):
    days = sorted(par.date.unique())
    codes = pd.Categorical(par.date, categories=days).codes
    idx = G.boot_index(len(days))
    yr = par.date.str[:4].astype(int).to_numpy()
    methods = ("S0", "S1", "S2i", "C0") + (() if astar == "C0" else (astar,))
    hyp = {"H1": ("C0", "S0")} if astar == "C0" else {"H1": ("C0", "S0"), "H2": (astar, "C0")}
    adv = G.adv_prior(dayvol, cal, sorted(set(zip(par.sym, par.date))))
    tv = G.tercile(par.vol_pre.to_numpy(), G.CUT_VOL)
    ta = G.tercile(np.array([adv[(s, d)][0] for s, d in zip(par.sym, par.date)]), G.CUT_ADV)
    common = np.array([(y, s) not in NONCOMMON[mode] for y, s in zip(yr, par.sym)])
    out = {"astar": astar, "methods": methods, "hypotheses": {k: f"{a} - {b}" for k, (a, b) in hyp.items()}, "variants": {}}
    for lab, lr, dep in G.VARIANTS:
        o = OUT[G.LRULES.index(lr)]
        cand, flags = G.compose(o, par.vol_pre.to_numpy(), par.spread.to_numpy())
        outs = {"S0": o[G.RUN_NAMES.index("S0")], "S1": o[G.RUN_NAMES.index("S1")], "S2i": o[G.RUN_NAMES.index("S2i")],
                "C0": cand["C0"], astar: cand[astar]}
        IS = {m: G.costs(outs[m], par.dir.to_numpy(), par.mid2.to_numpy(), dep) for m in methods}
        v = {"methods": {m: {"mean_is_c": float(IS[m].mean()), "p95_is_c": float(np.percentile(IS[m], 95)),
                             "limit_frac": float((outs[m][:, 0] / G.QTY).mean()),
                             "stop_parent_frac": float((outs[m][:, 7] > 0).mean()),
                             "s2i_branch_frac": float(flags[m].mean()) if m in flags else float(m == "S2i")} for m in methods},
             "hypotheses": {}, "secondary": {}, "yearly": {}, "cells": [], "common_stocks_only": {}}
        for h, (a, b) in hyp.items():
            est, p, ub = G.one_sided(IS[a] - IS[b], codes, len(days), idx)
            _, lo, hi = G.diff_ci(IS[a] - IS[b], codes, len(days), idx)
            v["hypotheses"][h] = {"diff": f"{a} - {b}", "mean_c": est, "p_one_sided": p, "upper_97_5_c": ub, "ci95": [lo, hi]}
            sub = codes[common]
            e2, p2, u2 = G.one_sided((IS[a] - IS[b])[common], pd.Categorical(par.date[common], categories=sorted(set(par.date[common]))).codes,
                                     len(set(par.date[common])), G.boot_index(len(set(par.date[common])))) if common.any() else (None, None, None)
            v["common_stocks_only"][h] = {"n_parents": int(common.sum()), "mean_c": e2, "p_one_sided": p2, "upper_97_5_c": u2,
                                          "excluded_symbol_years": sorted(f"{y} {s}" for y, s in NONCOMMON[mode]), "days": int(len(set(sub)))}
            v["yearly"][h] = {str(y): {"n": int((yr == y).sum()), "mean_c": float((IS[a] - IS[b])[yr == y].mean())} for y in sorted(set(yr))}
        for name, (a, b) in (("A* - S2i", (astar, "S2i")), ("C0 - S1", ("C0", "S1"))):
            est, lo, hi = G.diff_ci(IS[a] - IS[b], codes, len(days), idx)
            v["secondary"][name] = {"mean_c": est, "ci95": [lo, hi]}
        incl = (tv >= 0) & (ta >= 0)
        for i in range(3):
            for j in range(3):
                m_ = incl & (tv == i) & (ta == j)
                v["cells"].append({"cell": f"vol {G.TERC[i]} / adv {G.TERC[j]}",
                                   **{h: cell_stats(IS[a] - IS[b], m_, codes, len(days)) for h, (a, b) in hyp.items()}})
        v["cells"].append({"cell": "excluded (no vol_pre or adv_prior)",
                           **{h: cell_stats(IS[a] - IS[b], ~incl, codes, len(days)) for h, (a, b) in hyp.items()}})
        out["variants"][lab] = v
    prim = out["variants"]["L3D1"]["hypotheses"]
    holm = G.holm({h: prim[h]["p_one_sided"] for h in hyp})
    out["verdict"] = {h: {**holm[h], "result": "확인" if holm[h]["rejected"] else "확인 안 됨", "mean_c": prim[h]["mean_c"],
                          "upper_97_5_c": prim[h]["upper_97_5_c"], "sentence": SENTENCES[h][holm[h]["rejected"]]} for h in hyp}
    if astar == "C0":
        out["verdict"]["H2"] = {"result": "없음", "sentence": "Q2는 개발에서 개선 없음 (A* = C0)"}
    out["counts"] = {"n_parents": int(len(par)), "n_days": len(days), "n_symbol_days": int(par[["date", "sym"]].drop_duplicates().shape[0]),
                     "symbol_days_loaded": len(dayvol), "vol_pre_undefined": int(par.vol_pre.isna().sum()),
                     "stop_tests_skipped_invalid_quote": {f"L{lr}": int(OUT[G.LRULES.index(lr), :, :, 8].sum()) for lr in G.LRULES}}
    return out


# ---------------------------------------------------------------- rehearsal checks (development numbers of step 2)
def rehearsal_checks(out, read):
    dev = json.load(open(DEV_SUMMARY))
    rows = {(r["variant"], r["method"]): r for r in dev["rows"]}
    a = out["astar"]
    chk, diffs = {}, []
    for lab in ("L3D1", "L0D0"):
        v = out["variants"][lab]
        for m, r in v["methods"].items():
            d = rows[(lab, m)]
            diffs += [abs(r["mean_is_c"] - d["mean_is_c"]), abs(r["p95_is_c"] - d["p95_is_c"]), abs(r["limit_frac"] - d["limit_frac"]),
                      abs(r["stop_parent_frac"] - d["stop_parent_frac"]), abs(r["s2i_branch_frac"] - d["s2i_branch_frac"])]
        h1 = v["hypotheses"]["H1"]
        diffs += [abs(h1["mean_c"] - rows[(lab, "C0")]["minus_S0_c"]), abs(h1["upper_97_5_c"] - rows[(lab, "C0")]["minus_S0_hi"]),
                  abs(h1["ci95"][0] - rows[(lab, "C0")]["minus_S0_lo"])]
        if "H2" in v["hypotheses"]:
            h2 = v["hypotheses"]["H2"]
            diffs += [abs(h2["mean_c"] - rows[(lab, a)]["minus_C0_c"]), abs(h2["upper_97_5_c"] - rows[(lab, a)]["minus_C0_hi"])]
        for y, x in v["yearly"].get("H2", {}).items():
            diffs.append(abs(x["mean_c"] - rows[(lab, a)][f"minus_C0_{y}"]))
    chk["dev17_max_abs_diff"] = float(max(diffs))
    chk["dev17_numbers_equal"] = chk["dev17_max_abs_diff"] < 1e-6
    c16 = pd.read_csv(os.path.join(PROJ, "study16_followups", "results", "decomp16_cells.csv"))
    worst, nmatch = 0.0, True
    for lab in ("L3D1", "L0D0"):
        ref = c16[(c16.set == f"dev {lab}") & (c16.split == "vol_pre x adv_prior")].set_index("cell")
        for c in out["variants"][lab]["cells"]:
            r = ref.loc[c["cell"]]
            nmatch &= int(r.n) == c["H1"]["n"]
            worst = max(worst, abs(float(r.estar_minus_s0_c) - c["H1"]["mean_c"]))
    chk["cells_equal_16e"] = {"n_equal": bool(nmatch), "max_abs_diff_c": worst, "pass": bool(nmatch and worst < 5e-6)}
    chk["no_sealed_path_read"] = not any(os.path.abspath(p).startswith(SEALED_ROOT) for p in read)
    chk["pass"] = bool(chk["dev17_numbers_equal"] and chk["cells_equal_16e"]["pass"] and chk["no_sealed_path_read"])
    return chk


# ---------------------------------------------------------------- report
def report(out, mode, path):
    L_ = [f"# 연구 17 · {MODES[mode]['title']}", "",
          f"- 주문 {out['counts']['n_parents']:,}건, {out['counts']['n_days']}일, 종목·일 {out['counts']['n_symbol_days']:,}개. A* = {out['astar']}.",
          "- 판정(L3D1): 부트스트랩 단측 p(평균 ≥ 0 비율, 일 블록 21, 10,000회, seed 17)에 Holm(0.025, 0.05), 상한은 단측 97.5%.", ""]
    if mode == "rehearsal":
        L_ += ["**리허설: 개발 데이터라서 아래 \"확인\" 여부와 문장은 코드 경로 점검용이며 판정이 아니다.**", ""]
    L_ += ["## 판정", "", "| 가설 | 차이 | 평균 (¢/주) | 단측 97.5% 상한 | p | Holm 수준 | 결과 | 문장 |", "|---|---|---|---|---|---|---|---|"]
    for h, x in out["verdict"].items():
        if "p" in x:
            L_.append(f"| {h} | {out['hypotheses'][h]} | {x['mean_c']:+.4f} | {x['upper_97_5_c']:+.4f} | {x['p']:.4f} | {x['level']:.3f} | "
                      f"{x['result']} | \"{x['sentence']}\" |")
        else:
            L_.append(f"| {h} | — | — | — | — | — | {x['result']} | {x['sentence']} |")
    for lab in ("L3D1", "L0D0"):
        v = out["variants"][lab]
        L_ += ["", f"## {lab}: 방법별 평균 IS (p95) · 지정가 · 손절 · S2i 분기", "", "| 방법 | 평균 IS | p95 | 지정가 | 손절 | S2i 분기 |", "|---|---|---|---|---|---|"]
        for m, r in v["methods"].items():
            L_.append(f"| {m} | {r['mean_is_c']:.3f} | {r['p95_is_c']:.2f} | {r['limit_frac']:.3f} | {r['stop_parent_frac']:.3f} | {r['s2i_branch_frac']:.3f} |")
        L_ += ["", "| 비교 | 평균 | 95% CI | 단측 97.5% 상한 | p |", "|---|---|---|---|---|"]
        for h, x in v["hypotheses"].items():
            L_.append(f"| {h}: {x['diff']} | {x['mean_c']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | {x['upper_97_5_c']:+.4f} | {x['p_one_sided']:.4f} |")
        for n, x in v["secondary"].items():
            L_.append(f"| {n} (보조) | {x['mean_c']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | | |")
        for h, x in v["common_stocks_only"].items():
            L_.append(f"| {h} 보통주만 (보조, {x['n_parents']:,}건) | {x['mean_c']:+.4f} | | {x['upper_97_5_c']:+.4f} | {x['p_one_sided']:.4f} |")
        ys = sorted(next(iter(v["yearly"].values())))
        L_ += ["", f"### {lab} 연도별", "", "| 가설 | " + " | ".join(ys) + " |", "|---" * (len(ys) + 1) + "|"]
        for h, x in v["yearly"].items():
            L_.append(f"| {h} | " + " | ".join(f"{x[y]['mean_c']:+.3f}" for y in ys) + " |")
        L_ += ["", f"### {lab} 3 × 3 칸 (16-E 개발 기준값: vol_pre {G.CUT_VOL[0]:.3f} / {G.CUT_VOL[1]:.3f} bp, adv {G.CUT_ADV[0]:,.0f} / {G.CUT_ADV[1]:,.0f}주)", "",
               "| 칸 | " + " | ".join(f"{h} 평균 [95% CI] (n)" for h in v["hypotheses"]) + " |", "|---" * (len(v["hypotheses"]) + 1) + "|"]
        for c in v["cells"]:
            L_.append(f"| {c['cell']} | " + " | ".join(
                (f"{c[h]['mean_c']:+.3f} [{c[h]['ci_lo']:+.3f}, {c[h]['ci_hi']:+.3f}] ({c[h]['n']:,})" if c[h]["n"] else "—") for h in v["hypotheses"]) + " |")
    if "checks" in out:
        c = out["checks"]
        L_ += ["", "## 리허설 점검", "", f"- 2단계 개발 숫자와 최대 차이 {c['dev17_max_abs_diff']:.1e} → {'일치' if c['dev17_numbers_equal'] else '불일치'}",
               f"- 3 × 3 칸 E* − S0가 16-E 개발 칸과 같음: {c['cells_equal_16e']}", f"- 봉인 경로를 읽지 않음: {c['no_sealed_path_read']}"]
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
    read = []
    tasks, cal, info = sealed_plan(read) if mode == "sealed" else dev_plan(read)
    with mp.get_context("spawn").Pool(workers) as pool:
        res = list(pool.imap(worker, tasks, chunksize=8))
    bars = [r[4] for r in res]
    par = pd.DataFrame({"date": np.concatenate([[d] * len(m["dir"]) for _, d, m, _, _ in res]),
                        "sym": np.concatenate([[s] * len(m["dir"]) for s, _, m, _, _ in res]),
                        **{k: np.concatenate([m[k] for _, _, m, _, _ in res]) for k in ("dir", "start", "mid2", "spread", "vol_pre")}})
    OUT = np.concatenate([r[3] for r in res], axis=2)
    dayvol = {(s, d): m["day_vol"] for s, d, m, _, _ in res}
    out = analyse(par, OUT, dayvol, cal, mode, ASTAR)
    out.update({"mode": mode, "plan": info, "dropped_parents": int(sum(m["n_dropped"] for _, _, m, _, _ in res)),
                "read": {"bar_files": len(bars), "other": read}, "runtime_s": round(time.time() - t0, 1),
                "written": dt.datetime.now().isoformat(timespec="seconds")})
    if mode == "rehearsal":
        out["checks"] = rehearsal_checks(out, read + bars)
    files = MODES[mode]["files"]
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
    g.add_argument("--rehearsal", action="store_true", help="the same code on the development data")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main("sealed" if a.go else "rehearsal", a.workers)
