#!/usr/bin/env python3
"""Research 22 sealed test (SPEC22 5-6, SPEC22_ANSWERS; step-2 prompt 3), to be run once:
  python run_test22.py --go           sealed universe C 2021-01-04..2026-11-10 (minus EXCLUDE22), SPY of
                                                     research 21's sealed bars as the reference
  python run_test22.py --rehearsal    the same code path on the development data (existing universe 2015-2020)
The sealed run refuses unless study22_market_signal/UNLOCK22.txt holds exactly PHRASE (written by Kyle; Claude never creates the
file), no test result exists, every file of CORE is listed in FREEZE22.md and every listed file has its sha256, SPEC22.md,
SPEC22_ANSWERS.md and SPEC22_ADDENDUM.md match their sha256 files, the engine files of research 17, 18, 20 and loader15 equal their tags and
the development selection is usable. A* and whether H2 exists are read from results/dev22_summary.json, which FREEZE22.md
pins. The guard runs before any sealed path is touched. The rehearsal never builds a sealed path.
Runs: S0, S2i, G0 (E*), C3 (ADDENDUM D2) and, when A* is not G0, A* and its control with the reversed sign (ADDENDUM D1);
G0 and A* once more with delay 1.
Verdict (L3D1; day blocks 21, 10,000 draws, seed 22; one-sided p = share of bootstrap means >= 0; Holm over the hypotheses
that exist, overall alpha 0.05):
  H1  mean IS E* - S0 < 0            H2  mean IS A* - E* < 0 (only when the development run chose an A* other than G0)
Secondary, no verdict: L0D0, years, 95th-percentile differences, C3 - E*, A* - S0, A* - C3, A* - S2i, firing shares, the
control (control - E*, A* - control), the adverse mid change 60 / 180 s after the signal conditions for both bases and
thresholds (ANSWERS A5, ADDENDUM D3), the delay check, the split at 2025-11-03, common stocks only (ANSWERS A6). Results are written once at the end."""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import core22 as C  # noqa: E402
import loader22 as L22  # noqa: E402

E = C.E
RES = os.path.join(HERE, "results")
UNLOCK = os.path.join(HERE, "UNLOCK22.txt")
PHRASE = "연구22 테스트를 한 번 연다"
FREEZE = os.path.join(HERE, "FREEZE22.md")
DEV_SUMMARY = os.path.join(RES, "dev22_summary.json")
SEALED_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research22_data/sealed")
SEALED_BARS = os.path.join(SEALED_ROOT, "bars_1s_new_stocks")
SEALED_MANIFEST = os.path.join(HERE, "SEALED_MANIFEST22.tsv")
EXCLUDE = os.path.join(HERE, "EXCLUDE22.tsv")
CALENDAR = os.path.join(HERE, "calendar22.csv")
UNIVERSE = os.path.join(HERE, "universe22_{y}.tsv")
REF_SEALED_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research21_data/sealed/bars_1s")       # SPY 2021-: read by the sealed run only
REF_SEALED_MANIFEST = os.path.join(PROJ, "study21_more_pairs", "SEALED_MANIFEST21.tsv")
WINDOW = ("20210104", "20261110")
DOCS = ("SPEC22", "SPEC22_ANSWERS", "SPEC22_ADDENDUM")
TRIALS_BEFORE = 825
MODES = {"sealed": {"files": ("test22_summary.json", "test22_report.md", "test22_daily.csv"), "title": "봉인 테스트 (새 유니버스 C 2021–2026)"},
         "rehearsal": {"files": ("rehearsal22_summary.json", "rehearsal22_report.md", "rehearsal22_daily.csv"),
                       "title": "리허설 (개발 데이터: 기존 유니버스 2015–2020, 판정 아님)"}}
# ANSWERS A6: symbol-years whose CRSP share code is not 10 / 11 (SPEC22_QUESTIONS 5; development lists: research 17's rule)
NONCOMMON = {"sealed": {(2021, "TSE"), (2021, "GRMN"), (2021, "INSW"), (2022, "NVT"), (2022, "OCDX"), (2023, "INMD"), (2023, "NE"),
                        (2024, "CRSP"), (2025, "RBA"), (2025, "ESNT"), (2025, "AXS"), (2025, "GCT"), (2026, "NAMS"), (2021, "MMP"),
                        (2022, "MMP"), (2023, "BIP"), (2024, "AB"), (2025, "BIP"), (2024, "IRM"), (2025, "TRNO"), (2026, "SLG"), (2026, "CDP")},
             "rehearsal": {(2015, "BG"), (2017, "ALLE"), (2017, "CHKP"), (2018, "WCN"), (2019, "CATM"), (2019, "VNOM"), (2020, "DOX")}}
SENTENCES = {"H1": {True: "E*의 절감은 세 번째 새 종목 묶음에서도 재현됐다", False: "E*의 절감은 세 번째 종목 묶음에서는 확인되지 않았다"},
             "H2": {True: "기준 가격에서 벗어난 정도로 기다릴지 정하면 단순한 인내보다 더 쌌다", False: "기준 가격 신호는 평균 비용을 확실히 줄이지 못했다"},
             "none": "기준 가격 신호는 개발 구간에서 E*를 이기지 못했다. 그래서 개선 가설은 봉인 데이터로 시험하지 않았다"}
CORE = (("run_test22.py", "core22.py", "loader22.py", "signal22.py", "../study18_forecast/engine18.py", "../study20_pairs/signal20.py",
         "../study15_ml_execution/loader15.py", "SEALED_MANIFEST22.tsv", "EXCLUDE22.tsv", "calendar22.csv", "results/dev22_summary.json",
         "../study21_more_pairs/SEALED_MANIFEST21.tsv") + tuple(f"universe22_{y}.tsv" for y in range(2021, 2027)))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- guard (sealed run only, before any sealed path)
def frozen_files():
    """[(path relative to study22_market_signal, sha256)] from the FREEZE22.md table rows | `path` | `sha256` |."""
    if not os.path.isfile(FREEZE):
        return []
    pat = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
    return [m.groups() for m in map(pat.match, open(FREEZE, encoding="utf-8")) if m]


def selection():
    """A*, whether H2 exists and the signal setting of the prediction table, from the development summary."""
    d = json.load(open(DEV_SUMMARY, encoding="utf-8"))
    a, h2, ps = d["astar"], d["h2_exists"], d["prediction_setting"]
    if not (a == "G0" or a in C.CANDS) or bool(h2) != (a != "G0"):
        raise ValueError(f"A* {a!r} with h2_exists {h2!r}")
    if ps["basis"] not in C.BASES or float(ps["z_th"]) not in C.ZTH or (h2 and (ps["basis"], float(ps["z_th"])) != C.parse(a)[:2]):
        raise ValueError(f"prediction setting {ps!r}")
    return {"astar": a, "h2": bool(h2), "basis": ps["basis"], "z_th": float(ps["z_th"])}


def guard():
    why = []
    if any(os.path.exists(os.path.join(RES, f)) for f in MODES["sealed"]["files"]):
        why.append("test results already exist")
    if not os.path.isfile(UNLOCK) or open(UNLOCK, encoding="utf-8").read().strip() != PHRASE:
        why.append("no unlock phrase in UNLOCK22.txt")
    ff = frozen_files()
    if not ff or not all(any(p == c for p, _ in ff) for c in CORE):
        why.append("FREEZE22.md missing or without the core files")
    for p, h in ff:
        q = os.path.normpath(os.path.join(HERE, p))
        if not os.path.isfile(q) or sha(q) != h:
            why.append(f"frozen file differs: {p}")
    for doc in DOCS:
        f = os.path.join(HERE, f"{doc}.sha256")
        if not os.path.isfile(f) or open(f).read().split()[0] != sha(os.path.join(HERE, f"{doc}.md")):
            why.append(f"{doc}.md differs from {doc}.sha256")
    if C.frozen_differences():
        why.append("engine files differ from their tags")
    try:
        selection()
    except Exception as e:                                   # a missing or unreadable development summary included
        why.append(f"no usable development selection: {e}")
    return why


# ---------------------------------------------------------------- analysis
def analyse(par, OUT, sd, names, delay_runs, sel, mode):
    a, h2 = sel["astar"], sel["h2"]
    alln = list(names) + [f"{m}@d1" for m in delay_runs]
    rep = ("S0", "S2i", "C3", "G0") + ((a,) if h2 else ())
    tabs, st = C.tables(par, OUT, alln, rep)
    codes, days, idx, W = st
    dates = par.date.to_numpy()
    years = par.date.str[:4].astype(int).to_numpy()
    common = np.array([(y, s) not in NONCOMMON[mode] for y, s in zip(years, par.sym)])
    pairs = [("Estar_minus_S0", "G0", "S0"), ("C3_minus_Estar", "C3", "G0")] + (            # C3 - E*: ADDENDUM D2
        [("Astar_minus_Estar", a, "G0"), ("Astar_minus_S0", a, "S0"), ("Astar_minus_C3", a, "C3"), ("Astar_minus_S2i", a, "S2i")] if h2 else [])
    diffs, by_year, split, com = {}, {}, {}, {}
    for lab, lr, dep in E.VARIANTS:
        IS, _ = C.is_table(par, OUT, alln, lr, dep)
        diffs[lab] = {k: C.pair(IS, x, y, codes, len(days), idx, W) for k, x, y in pairs}
        by_year[lab] = {k: {str(yr): C.subset_stat(IS[x] - IS[y], dates, years == yr) for yr in sorted(set(years))} for k, x, y in pairs}
        split[lab] = {k: {"before": C.subset_stat(IS[x] - IS[y], dates, dates < C.CUT), "from": C.subset_stat(IS[x] - IS[y], dates, dates >= C.CUT)}
                      for k, x, y in pairs}
        com[lab] = {k: C.subset_stat(IS[x] - IS[y], dates, common) for k, x, y in pairs}
    p = {"H1": diffs["L3D1"]["Estar_minus_S0"]["p_one_sided"]}
    if h2:
        p["H2"] = diffs["L3D1"]["Astar_minus_Estar"]["p_one_sided"]
    hol = C.holm(p)
    key = {"H1": "Estar_minus_S0", "H2": "Astar_minus_Estar"}
    verdict = {h: {**v, "confirmed": bool(v["rejected"]), "result": "확인" if v["rejected"] else "확인 안 됨",
                   "mean_c": diffs["L3D1"][key[h]]["mean"], "ci_lo": diffs["L3D1"][key[h]]["ci_lo"], "ci_hi": diffs["L3D1"][key[h]]["ci_hi"],
                   "upper_97_5": diffs["L3D1"][key[h]]["upper_97_5"], "sentence": SENTENCES[h][bool(v["rejected"])]} for h, v in hol.items()}
    if not h2:
        verdict["H2"] = {"result": "없음", "sentence": SENTENCES["none"]}
    grp = ("all", f"z{sel['z_th']:g}_early", f"z{sel['z_th']:g}_hold")
    ptabs = {b: C.prediction_table(sd, b) for b in C.BASES}                              # ADDENDUM D3: the whole table
    ptab = ptabs[sel["basis"]]
    out = {"astar": a, "h2_exists": h2, "hypotheses": list(p), "verdict": verdict, "differences": diffs, "tables": tabs,
           "by_year": by_year, "date_split": split,
           "common_only": {"excluded_symbol_years": sorted(f"{y} {s}" for y, s in NONCOMMON[mode]), "parents": int(common.sum()), "differences": com},
           "prediction": {"basis": sel["basis"], "z_th": sel["z_th"], "groups": {g: ptab[g] for g in grp}, "tables": ptabs},
           "diagnostics": {b: C.signal_counts(sd, b) for b in C.BASES},
           "delay_check": C.delay_table(par, OUT, alln, delay_runs, st),
           "control_flip": C.control_table(par, OUT, alln, a, st) if h2 else None,
           "counts": {"parents": int(len(par)), "days": len(days), "first_day": days[0], "last_day": days[-1],
                      "symbol_days": int(len(sd["date"])), "parents_before_cut": int((dates < C.CUT).sum()), "parents_from_cut": int((dates >= C.CUT).sum()),
                      "market_at_nonpositive_price": int(OUT[:, :, :, 11].sum())}}
    return out


def rehearsal_checks(out, tasks):
    """The rehearsal against the development run: every number both runs hold must be equal, and no sealed path was read."""
    dev = json.load(open(DEV_SUMMARY, encoding="utf-8"))
    worst, n = 0.0, 0

    def cmp(x, y):
        nonlocal worst, n
        if isinstance(x, dict) and isinstance(y, dict):
            for k in x:
                if k in y:
                    cmp(x[k], y[k])
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            worst, n = max(worst, abs(float(x) - float(y))), n + 1
    for lab in out["tables"]:
        for m, r in out["tables"][lab].items():
            cmp(r, dev["tables"][lab][m])
    n_tab = n
    cmp(out["prediction"]["tables"], dev["prediction"])
    cmp(out["diagnostics"], dev["diagnostics"])
    if out["h2_exists"]:
        for lab, v in out["delay_check"].items():
            cmp(v, dev["delay_check"][lab])
        cmp(out["control_flip"], dev["control_flip"])
    paths = [t["path"] for t in tasks] + [t["ref_path"] for t in tasks if t["ref_path"]]
    no_sealed = not any(os.path.abspath(q).startswith((SEALED_ROOT, os.path.dirname(REF_SEALED_ROOT))) for q in paths)
    ok = worst == 0.0 and n_tab > 0 and no_sealed and out["astar"] == dev["astar"] and out["h2_exists"] == dev["h2_exists"]
    return {"numbers_compared": n, "table_numbers_compared": n_tab, "max_abs_diff": worst, "selection_equal": out["astar"] == dev["astar"],
            "no_sealed_path_read": bool(no_sealed), "pass": bool(ok)}


def f3(x):
    return "—" if x is None else f"{x:+.3f}"


def report(out, mode):
    """The report text (built before any file is written)."""
    a, h2 = out["astar"], out["h2_exists"]
    c = out["counts"]
    L_ = [f"# 연구 22 · {MODES[mode]['title']}", "",
          f"- 종목·일 {c['symbol_days']:,}개, 주문 {c['parents']:,}건, {c['days']:,}일({c['first_day']} ~ {c['last_day']}). A* = {a}"
          + ("" if h2 else " (개발에서 개선 없음: H2 없음)") + ".",
          "- IS 단위는 ¢/주다. 판정: 부트스트랩 단측 p(평균이 0 이상인 비율, 일 블록 21, 10,000회, seed 22)에 Holm(전체 α 0.05).",
          "- 2025-11-03부터의 호가 잔량은 100으로 나눠 읽었다(SPEC22 1절).", "", "## 판정 (L3D1)", "",
          "| 가설 | 차이 | 평균 (¢/주) | 95% CI | p | Holm 수준 | 결과 | 문장 |", "|---|---|---|---|---|---|---|---|"]
    lab_h = {"H1": "E* − S0", "H2": "A* − E*"}
    for h in ("H1", "H2"):
        v = out["verdict"][h]
        if "p" in v:
            L_.append(f"| {h} | {lab_h[h]} | {f3(v['mean_c'])} | [{f3(v['ci_lo'])}, {f3(v['ci_hi'])}] | {v['p']:.4f} | {v['level']:.4f} | {v['result']} | \"{v['sentence']}\" |")
        else:
            L_.append(f"| {h} | {lab_h[h]} | — | — | — | — | {v['result']} | \"{v['sentence']}\" |")
    names_k = {"Estar_minus_S0": "E* − S0", "C3_minus_Estar": "C3 − E*", "Astar_minus_Estar": "A* − E*", "Astar_minus_S0": "A* − S0", "Astar_minus_C3": "A* − C3", "Astar_minus_S2i": "A* − S2i"}
    L_ += ["", "## 보조 결과 (판정 아님)", ""]
    for lab in ("L3D1", "L0D0"):
        L_ += [f"### 규칙별 · {lab}", "", "| 규칙 | 평균 IS [95% CI] | p95 | − G0 [95% CI] | 발동: 주문 | 발동: 결정 시점 |", "|---|---|---|---|---|---|"]
        for m, r in out["tables"][lab].items():
            L_.append(f"| {m} | {r['is_mean']:.3f} [{r['is_ci_lo']:.3f}, {r['is_ci_hi']:.3f}] | {r['p95']:.2f} | {f3(r['minus_G0_mean'])} "
                      f"[{f3(r['minus_G0_ci_lo'])}, {f3(r['minus_G0_ci_hi'])}] | {100 * r['firing']['parents']:.1f}% | {100 * r['firing']['decisions']:.2f}% |")
        L_ += ["", "| 차이 | 평균 [95% CI] | p | p95 차이 [95% CI] |", "|---|---|---|---|"]
        for k, d in out["differences"][lab].items():
            L_.append(f"| {names_k[k]} | {f3(d['mean'])} [{f3(d['ci_lo'])}, {f3(d['ci_hi'])}] | {d['p_one_sided']:.4f} | {f3(d['p95_diff'])} "
                      f"[{f3(d['p95_ci_lo'])}, {f3(d['p95_ci_hi'])}] |")
        yrs = sorted(next(iter(out["by_year"][lab].values())))
        L_ += ["", "| 연도별 평균 | " + " | ".join(yrs) + " |", "|---|" + "---|" * len(yrs)]
        for k, v in out["by_year"][lab].items():
            L_.append(f"| {names_k[k]} | " + " | ".join(f3(v[y].get("mean")) for y in yrs) + " |")
        L_ += ["", "| 차이 | 2025-10-31까지 | 2025-11-03부터 | 보통주만 |", "|---|---|---|---|"]
        for k in out["differences"][lab]:
            cells = []
            for s in (out["date_split"][lab][k]["before"], out["date_split"][lab][k]["from"], out["common_only"]["differences"][lab][k]):
                cells.append("—" if not s.get("n") else f"{f3(s.get('mean'))} [{f3(s.get('ci_lo'))}, {f3(s.get('ci_hi'))}] (주문 {s['n']:,})")
            L_.append(f"| {names_k[k]} | " + " | ".join(cells) + " |")
        L_.append("")
    if out.get("control_flip"):
        L_ += ["### 대조 실행: 부호를 뒤집은 신호 (추가 문서 D1, 판정 아님)", "",
               f"A*와 같은 기준·기준값·규칙에서 a_t의 부호만 뒤집었다({out['control_flip']['L3D1']['control']}).", "",
               "| 가정 | 대조 평균 IS [95% CI] | 대조 − E* [95% CI] | A* − 대조 [95% CI] | 발동: 주문 (대조 / A*) |", "|---|---|---|---|---|"]
        for lab, v in out["control_flip"].items():
            x, y, z = v["control_is"], v["control_minus_Estar"], v["Astar_minus_control"]
            L_.append(f"| {lab} | {x['mean']:.3f} [{x['ci_lo']:.3f}, {x['ci_hi']:.3f}] | {f3(y['mean'])} [{f3(y['ci_lo'])}, {f3(y['ci_hi'])}] | "
                      f"{f3(z['mean'])} [{f3(z['ci_lo'])}, {f3(z['ci_hi'])}] | {100 * v['control_firing']['parents']:.1f}% / {100 * v['astar_firing']['parents']:.1f}% |")
        L_.append("")
    pr = out["prediction"]
    L_ += ["### 신호의 예측력 (경로와 무관한 결정 격자 시점, 추가 문서 D3)", "",
           f"값은 그 시점 뒤 중간값 변화(bp)이고, 주문에 불리한 방향이 양수다. 예측표 다섯째 줄은 기준 {pr['basis']}, 기준값 {pr['z_th']:g}의 조기 체결 조건 60초 값으로 본다.", "",
           "| 기준 | 조건 | 60초 뒤 [95% CI] | 시점 수 | 180초 뒤 [95% CI] | 시점 수 (빠진 시점) |", "|---|---|---|---|---|---|"]
    lab_g = {"all": "신호가 있는 모든 시점"}
    lab_g.update({f"z{z:g}_{k}": f"a {'≥' if k == 'early' else '≤ −'}{z:g} ({'조기 체결' if k == 'early' else '대기 유지'} 조건)" for z in C.ZTH for k in ("early", "hold")})
    for b in C.BASES:
        for g in C.GROUPS:
            r6, r18 = pr["tables"][b][g]["60"], pr["tables"][b][g]["180"]
            L_.append(f"| {b} | {lab_g[g]} | {f3(r6.get('mean_bp'))} [{f3(r6.get('ci_lo'))}, {f3(r6.get('ci_hi'))}] | {r6['points']:,} | {f3(r18.get('mean_bp'))} "
                      f"[{f3(r18.get('ci_lo'))}, {f3(r18.get('ci_hi'))}] | {r18['points']:,} ({r18['left_out']:,}) |")
    L_.append("")
    for b in C.BASES:
        d = out["diagnostics"][b]
        L_.append(f"- {b}: 결정 격자 시점 {d['grid_points']:,}개 중 신호 없음 {100 * d['no_signal_share']:.1f}%, 신호가 있는 시점 중 σ 하한(1bp) 적용 "
                  f"{100 * d['floor_share_of_signal']:.1f}%.")
    L_ += ["", "### 지연 점검 (결정은 막대 t, 실행은 막대 t + 1)", "", "| 가정 | 항목 | 평균 [95% CI] |", "|---|---|---|"]
    for lab, v in out["delay_check"].items():
        for k, x in v.items():
            if isinstance(x, dict):
                L_.append(f"| {lab} | {k.replace('_delay_minus_', ' 지연 − ').replace('_delay', ' 지연')} | {f3(x['mean'])} [{f3(x['ci_lo'])}, {f3(x['ci_hi'])}] |")
    s = out["sim"]
    L_ += ["", "## 자료 관련 건수", "",
           f"- 시작 호가가 유효하지 않아 뺀 주문 {s['dropped_parents']:,}건. SPY 파일이 없는 종목·일 {s['symbol_days_without_ref']:,}개. 0 이하 가격의 시장가 주문 {c['market_at_nonpositive_price']}건.",
           f"- 2025-10-31까지의 주문 {c['parents_before_cut']:,}건, 2025-11-03부터의 주문 {c['parents_from_cut']:,}건.",
           f"- 보통주가 아닌 종목·연도 {len(out['common_only']['excluded_symbol_years'])}개를 뺀 주문 {out['common_only']['parents']:,}건."]
    if mode == "rehearsal":
        k = out["checks"]
        L_ += ["", "## 리허설 점검", "",
               f"- 개발 실행(dev22)의 숫자와 최대 차이 {k['max_abs_diff']:.1e}(숫자 {k['numbers_compared']:,}개) → {'일치' if k['pass'] else '불일치'}. "
               f"A*와 H2 유무 같음 {k['selection_equal']}, 봉인 경로를 읽지 않음 {k['no_sealed_path_read']}.",
               "- 이 보고서의 \"확인\" 여부는 개발 데이터로 낸 것이라 판정이 아니다."]
    L_ += ["", f"- 일 단위 결과: results/{MODES[mode]['files'][2]}. 읽은 막대 폴더: {out['read']['bars_root']}, 기준 가격 {out['read']['ref_root']}. 실행 {out['runtime_s']} s", ""]
    return "\n".join(L_)


# ---------------------------------------------------------------- run
def main(mode, workers):
    t0 = time.time()
    if mode == "sealed":
        why = guard()
        if why:
            raise SystemExit("refused: " + "; ".join(why))
        tasks, cal, pinfo = L22.plan_files(SEALED_MANIFEST, EXCLUDE, CALENDAR, UNIVERSE, WINDOW, SEALED_BARS, REF_SEALED_MANIFEST, REF_SEALED_ROOT)
        roots = (SEALED_BARS, REF_SEALED_ROOT)
    else:
        if any(os.path.exists(os.path.join(RES, f)) for f in MODES[mode]["files"]):
            raise SystemExit("rehearsal results already exist: rename them first (nothing is overwritten)")
        tasks, cal, pinfo = L22.dev_plan()
        roots = ("study15_ml_execution/loader15 sources (2015-2020)", L22.DEV_SPY_ROOT)
    sel = selection()
    names = list(C.BASE) + ([sel["astar"], C.flip_name(sel["astar"])] if sel["h2"] else [])     # the control: ADDENDUM D1
    delay_runs = ["G0"] + ([sel["astar"]] if sel["h2"] else [])
    par, OUT, sd, info = C.simulate(tasks, names, delay_runs, workers, bases=C.BASES)             # both bases: ADDENDUM D3
    out = analyse(par, OUT, sd, names, delay_runs, sel, mode)
    out["sim"] = {**info, **pinfo}
    out["read"] = {"bars_root": roots[0].replace(os.path.expanduser("~"), "~"), "ref_root": roots[1].replace(os.path.expanduser("~"), "~")}
    if mode == "rehearsal":
        out["checks"] = rehearsal_checks(out, tasks)
    out.update({"mode": mode, "boot": {"B": E.BOOT_B, "block": E.BOOT_BLOCK, "seed": C.SEED}, "trials_before": TRIALS_BEFORE,
                "trials_added": len(out["hypotheses"]) if mode == "sealed" else 0, "runtime_s": round(time.time() - t0, 1),
                "written": dt.datetime.now().isoformat(timespec="seconds")})
    f_sum, f_rep, f_day = (os.path.join(RES, f) for f in MODES[mode]["files"])
    daily = C.daily_rows(par, OUT[:, :len(names)], names)
    text = report(out, mode)
    summary = json.dumps(out, indent=1, default=C.jdefault, ensure_ascii=False)
    os.makedirs(RES, exist_ok=True)
    daily.to_csv(f_day, index=False, float_format="%.6f")
    with open(f_rep, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(f_sum, "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(text)
    if mode == "rehearsal" and not out["checks"]["pass"]:
        raise SystemExit("the rehearsal checks failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--go", action="store_true")
    g.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    main("sealed" if a.go else "rehearsal", a.workers)
