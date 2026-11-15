#!/usr/bin/env python3
"""Research 25 sealed test (SPEC25 5 and 7, SPEC25_ANSWERS; step-2 prompt 4), to be run once and only when A* is not R0:
  python run_test25.py --go           sealed universe E 2021-01-04..2026-11-10 (minus EXCLUDE25), SPY of
                                                     research 21's sealed bars as the reference of the model's input 5
  python run_test25.py --rehearsal    the same code path on the validation data (existing universe 2019-2020)
The sealed run refuses unless study25_volatility/UNLOCK25.txt holds exactly PHRASE (written by Kyle; Claude never creates the
file), no test result exists, every file of CORE is listed in FREEZE25.md and every listed file has its sha256 (research
23's frozen model model23.json and the validation result results/dev25_summary.json among them), model23.json equals its
own sha256 file, SPEC25.md and SPEC25_ANSWERS.md match their sha256 files, the files of research 15-23 equal their tags,
the validation selection is usable and A* is a candidate. When A* = R0 ("개발에서 개선 없음") there is no sealed run and no
rehearsal: both refuse. A* is read from results/dev25_summary.json. The guard runs before any sealed path is touched. The
rehearsal never builds a sealed path.
Runs: S0, S2i, R0 (E*), C3, A* and its control (the volatility / volume condition turned around); R0 and A* once more with
delay 1.
Verdict (day blocks 21, 10,000 draws, seed 25; one hypothesis, alpha 0.05):
  H   mean IS A* - E* < 0 under L3D1        one-sided p = share of bootstrap means >= 0
Secondary, no verdict: E* - S0, C3 - E*, S2i - E*, A* - S0 / C3 / S2i, L0D0, years, 95th-percentile differences, the dollar
saving of A* against E*, firing shares, the control, the delay check, the split at 2025-11-03, common stocks only
(ANSWERS A5). Results are written once at the end."""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import core25 as C  # noqa: E402

E, M, L22 = C.E, C.M, C.L22
RES = os.path.join(HERE, "results")
UNLOCK = os.path.join(HERE, "UNLOCK25.txt")
PHRASE = "연구25 테스트를 한 번 연다"
FREEZE = os.path.join(HERE, "FREEZE25.md")
DEV_SUMMARY = os.path.join(RES, "dev25_summary.json")
MODEL = C.MODEL23
SEALED_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research25_data/sealed")
SEALED_BARS = os.path.join(SEALED_ROOT, "bars_1s_new_stocks")
SEALED_MANIFEST = os.path.join(HERE, "SEALED_MANIFEST25.tsv")
EXCLUDE = os.path.join(HERE, "EXCLUDE25.tsv")
CALENDAR = os.path.join(HERE, "calendar25.csv")
UNIVERSE = os.path.join(HERE, "universe25_{y}.tsv")              # the first N slots of universe_E.csv, written when N is fixed
REF_SEALED_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research21_data/sealed/bars_1s")       # SPY 2021-: read by the sealed run only
REF_SEALED_MANIFEST = os.path.join(PROJ, "study21_more_pairs", "SEALED_MANIFEST21.tsv")
WINDOW = ("20210104", "20261110")
VALID = (2019, 2020)                                             # the rehearsal reads the validation years
DOCS = ("SPEC25", "SPEC25_ANSWERS")
TRIALS_BEFORE = 865
NOT_IMPROVED = "no improvement in development (A* = R0): the sealed run is not made (SPEC25 4)"
MODES = {"sealed": {"files": ("test25_summary.json", "test25_report.md", "test25_daily.csv"), "title": "봉인 테스트 (새 유니버스 E 2021–2026)"},
         "rehearsal": {"files": ("rehearsal25_summary.json", "rehearsal25_report.md", "rehearsal25_daily.csv"),
                       "title": "리허설 (검증 데이터: 기존 유니버스 2019–2020, 판정 아님)"}}
# ANSWERS A5: symbol-years whose CRSP share code is not 10 / 11 (SPEC25_QUESTIONS 5; validation lists: research 17's rule)
NONCOMMON = {"sealed": {(2021, "AUDC"), (2022, "PAGS"), (2022, "TPGY"), (2023, "MEOH"), (2023, "GLBE"), (2023, "CSIQ"), (2024, "STNG"), (2024, "MGA"),
                        (2024, "TGLS"), (2024, "KRNT"), (2025, "TBBB"), (2025, "PRTA"), (2025, "GLBE"), (2025, "FNV"), (2026, "NTR"), (2026, "BAM"),
                        (2026, "KNSA"), (2026, "TBBB"), (2026, "GSL"), (2022, "ACC"), (2025, "NSA"), (2025, "PCH"), (2023, "NS"), (2024, "IEP")},
             "rehearsal": {(2019, "CATM"), (2019, "VNOM"), (2020, "DOX")}}
SENTENCES = {True: "직전 변동성과 거래량을 조건에 넣으면 단순한 인내보다 더 쌌다", False: "직전 변동성과 거래량을 조건에 넣어도 평균 비용은 확실히 줄지 않았다",
             "none": "직전 변동성과 거래량을 넣은 규칙은 검증 구간에서 E*를 이기지 못했다. 그래서 봉인 데이터로 시험하지 않았다"}
CORE = (("run_test25.py", "core25.py", "signal25.py", "results/dev25_summary.json", "../study23_combined_forecast/core23.py", "../study23_combined_forecast/feat23.py",
         "../study23_combined_forecast/model23.py", "../study23_combined_forecast/model23.json", "../study22_market_signal/core22.py", "../study22_market_signal/loader22.py", "../study22_market_signal/signal22.py",
         "../study18_forecast/engine18.py", "../study20_pairs/signal20.py", "../study15_ml_execution/loader15.py", "SEALED_MANIFEST25.tsv", "EXCLUDE25.tsv", "calendar25.csv",
         "universe_E.csv", "../study21_more_pairs/SEALED_MANIFEST21.tsv") + tuple(f"universe25_{y}.tsv" for y in range(2021, 2027)))


# ---------------------------------------------------------------- guard (sealed run only, before any sealed path)
def frozen_files():
    """[(path relative to study25_volatility, sha256)] from the FREEZE25.md table rows | `path` | `sha256` |."""
    if not os.path.isfile(FREEZE):
        return []
    pat = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
    return [m.groups() for m in map(pat.match, open(FREEZE, encoding="utf-8")) if m]


def model_ok():
    side = os.path.splitext(MODEL)[0] + ".sha256"
    return bool(os.path.isfile(MODEL) and os.path.isfile(side) and open(side).read().split()[:1] == [C.sha(MODEL)])


def selection():
    """A* from the validation summary; the summary must belong to the model file and be consistent with itself."""
    d = json.load(open(DEV_SUMMARY, encoding="utf-8"))
    a, imp = d["astar"], d["improves_in_development"]
    if not (a == "R0" or a in C.CANDS) or not isinstance(imp, bool) or imp != (a != "R0") or d["selection"]["astar"] != a:
        raise ValueError(f"A* {a!r} with improves_in_development {imp!r}")
    if d["model_sha256"] != C.sha(MODEL):
        raise ValueError("the validation summary was made with another model file")
    return {"astar": a, "improves": imp}


def guard():
    why = []
    if any(os.path.exists(os.path.join(RES, f)) for f in MODES["sealed"]["files"]):
        why.append("test results already exist")
    if not os.path.isfile(UNLOCK) or open(UNLOCK, encoding="utf-8").read().strip() != PHRASE:
        why.append("no unlock phrase in UNLOCK25.txt")
    ff = frozen_files()
    if not ff or not all(any(p == c for p, _ in ff) for c in CORE):
        why.append("FREEZE25.md missing or without the core files")
    for p, h in ff:
        q = os.path.normpath(os.path.join(HERE, p))
        if not os.path.isfile(q) or C.sha(q) != h:
            why.append(f"frozen file differs: {p}")
    if not model_ok():
        why.append("model23.json missing or differs from model23.sha256")
    for doc in DOCS:
        f = os.path.join(HERE, f"{doc}.sha256")
        if not os.path.isfile(f) or open(f).read().split()[0] != C.sha(os.path.join(HERE, f"{doc}.md")):
            why.append(f"{doc}.md differs from {doc}.sha256")
    if C.frozen_differences():
        why.append("files of earlier studies differ from their tags")
    try:
        if not selection()["improves"]:
            why.append(NOT_IMPROVED)
    except Exception as e:                                   # a missing or unreadable summary or model included
        why.append(f"no usable development selection: {e}")
    return why


# ---------------------------------------------------------------- analysis
def analyse(par, OUT, names, delay_runs, sel, mode):
    a = sel["astar"]
    alln = list(names) + [f"{m}@d1" for m in delay_runs]
    rep = ("S0", "S2i", "C3", "R0", a)
    tabs, st = C.tables(par, OUT, alln, rep)
    codes, days, idx, W = st
    dates = par.date.to_numpy()
    years = par.date.str[:4].astype(int).to_numpy()
    common = np.array([(y, s) not in NONCOMMON[mode] for y, s in zip(years, par.sym)])
    pairs = [("Estar_minus_S0", "R0", "S0"), ("C3_minus_Estar", "C3", "R0"), ("S2i_minus_Estar", "S2i", "R0"), ("Astar_minus_Estar", a, "R0"),
             ("Astar_minus_S0", a, "S0"), ("Astar_minus_C3", a, "C3"), ("Astar_minus_S2i", a, "S2i")]
    diffs, by_year, split, com, usd = {}, {}, {}, {}, {}
    for lab, lr, dep in E.VARIANTS:
        IS, _ = C.is_table(par, OUT, alln, lr, dep)
        diffs[lab] = {k: C.pair(IS, x, y, codes, len(days), idx, W) for k, x, y in pairs}
        by_year[lab] = {k: {str(yr): C.subset_stat(IS[x] - IS[y], dates, years == yr) for yr in sorted(set(years))} for k, x, y in pairs}
        split[lab] = {k: {"before": C.subset_stat(IS[x] - IS[y], dates, dates < C.CUT), "from": C.subset_stat(IS[x] - IS[y], dates, dates >= C.CUT)}
                      for k, x, y in pairs}
        com[lab] = {k: C.subset_stat(IS[x] - IS[y], dates, common) for k, x, y in pairs}
        day_usd = np.bincount(codes, weights=(IS["R0"] - IS[a]) * C.USD_PER_CENT, minlength=len(days))
        cum = np.cumsum(day_usd)
        usd[lab] = {"Astar_saving_vs_Estar_total": float(day_usd.sum()), "Estar_saving_vs_S0_total": float(((IS["S0"] - IS["R0"]) * C.USD_PER_CENT).sum()),
                    "by_year": {str(yr): float(((IS["R0"] - IS[a])[years == yr] * C.USD_PER_CENT).sum()) for yr in sorted(set(years))},
                    "cumulative_min": float(cum.min()), "cumulative_max": float(cum.max()), "days_positive": int((day_usd > 0).sum()),
                    "days_negative": int((day_usd < 0).sum())}
    d = diffs["L3D1"]["Astar_minus_Estar"]
    ok = bool(d["p_one_sided"] < C.ALPHA)
    verdict = {"H": {"p": d["p_one_sided"], "level": C.ALPHA, "confirmed": ok, "result": "확인" if ok else "확인 안 됨", "mean_c": d["mean"],
                     "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "upper_97_5": d["upper_97_5"], "sentence": SENTENCES[ok]}}
    out = {"astar": a, "hypotheses": ["H"], "verdict": verdict, "differences": diffs, "tables": tabs, "by_year": by_year, "date_split": split,
           "dollars": usd, "common_only": {"excluded_symbol_years": sorted(f"{y} {s}" for y, s in NONCOMMON[mode]), "parents": int(common.sum()), "differences": com},
           "delay_check": C.delay_table(par, OUT, alln, delay_runs, st), "control": C.control_table(par, OUT, alln, a, st),
           "counts": {"parents": int(len(par)), "days": len(days), "first_day": days[0], "last_day": days[-1],
                      "parents_before_cut": int((dates < C.CUT).sum()), "parents_from_cut": int((dates >= C.CUT).sum()),
                      "market_at_nonpositive_price": int(OUT[:, :, :, 11].sum())}}
    return out


def rehearsal_checks(out, tasks):
    """The rehearsal against the validation run: every number both runs hold must be equal, and no sealed path was read."""
    dev = json.load(open(DEV_SUMMARY, encoding="utf-8"))
    worst, n = 0.0, 0

    def cmp(x, y):
        nonlocal worst, n
        if isinstance(x, dict) and isinstance(y, dict):
            for k in x:
                if k in y:
                    cmp(x[k], y[k])
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for u, v in zip(x, y):
                cmp(u, v)
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            worst, n = max(worst, abs(float(x) - float(y))), n + 1
    for lab in out["tables"]:
        for m, r in out["tables"][lab].items():
            cmp(r, dev["tables"][lab][m])
    n_tab = n
    for lab, v in out["delay_check"].items():
        cmp(v, dev["delay_check"][lab])
    cmp(out["control"], dev["control"])
    paths = [t["path"] for t in tasks] + [t["ref_path"] for t in tasks if t["ref_path"]]
    no_sealed = not any(os.path.abspath(q).startswith((SEALED_ROOT, os.path.dirname(REF_SEALED_ROOT))) for q in paths)
    ok = worst == 0.0 and n_tab > 0 and n > n_tab and no_sealed and out["astar"] == dev["astar"]
    return {"numbers_compared": n, "table_numbers_compared": n_tab, "max_abs_diff": worst, "selection_equal": out["astar"] == dev["astar"],
            "no_sealed_path_read": bool(no_sealed), "pass": bool(ok)}


def f3(x):
    return "—" if x is None else f"{x:+.3f}"


def usd(x):
    return "—" if x is None else f"{round(x) + 0:+,.0f}"


def report(out, mode):
    """The report text (built before any file is written)."""
    a, c, s = out["astar"], out["counts"], out["sim"]
    sp = C.SPECS[a]
    L_ = [f"# 연구 25 · {MODES[mode]['title']}", "",
          f"- 종목·일 {s['symbol_days']:,}개, 주문 {c['parents']:,}건, {c['days']:,}일({c['first_day']} ~ {c['last_day']}). A* = {a} "
          f"(신호 {sp['signal']}, 문턱 {sp['threshold']:g}, 엔진 경로 {sp['engine']}). 예측 확률은 연구 23의 동결 모델이다.",
          "- 판정: 일 블록 부트스트랩(21일, 10,000회, seed 25)의 단측 p(평균이 0 이상인 비율), 수준 0.05.",
          "- IS 단위는 ¢/주다. 2025-11-03부터의 호가 잔량은 100으로 나눠 읽었다(SPEC25 1절).", "", "## 판정", "",
          "| 가설 | 값 | 95% CI | p | 수준 | 결과 | 문장 |", "|---|---|---|---|---|---|---|"]
    v = out["verdict"]["H"]
    L_.append(f"| H | A* − E* {f3(v['mean_c'])}¢ (L3D1) | [{f3(v['ci_lo'])}, {f3(v['ci_hi'])}] | {v['p']:.4f} | {v['level']:.2f} | {v['result']} | \"{v['sentence']}\" |")
    L_ += ["", "## 보조 결과 (판정 아님)", ""]
    names_k = {"Estar_minus_S0": "E* − S0", "C3_minus_Estar": "C3 − E*", "S2i_minus_Estar": "S2i − E*", "Astar_minus_Estar": "A* − E*",
               "Astar_minus_S0": "A* − S0", "Astar_minus_C3": "A* − C3", "Astar_minus_S2i": "A* − S2i"}
    for lab in ("L3D1", "L0D0"):
        L_ += [f"### 규칙별 · {lab}", "", "| 규칙 | 평균 IS [95% CI] | p95 | − R0 [95% CI] | 발동: 주문 | 발동: 결정 시점 |", "|---|---|---|---|---|---|"]
        for m, r in out["tables"][lab].items():
            L_.append(f"| {m} | {r['is_mean']:.3f} [{r['is_ci_lo']:.3f}, {r['is_ci_hi']:.3f}] | {r['p95']:.2f} | {f3(r['minus_R0_mean'])} "
                      f"[{f3(r['minus_R0_ci_lo'])}, {f3(r['minus_R0_ci_hi'])}] | {100 * r['firing']['parents']:.1f}% | {100 * r['firing']['decisions']:.2f}% |")
        L_ += ["", "| 차이 | 평균 [95% CI] | p | p95 차이 [95% CI] |", "|---|---|---|---|"]
        for k, x in out["differences"][lab].items():
            L_.append(f"| {names_k[k]} | {f3(x['mean'])} [{f3(x['ci_lo'])}, {f3(x['ci_hi'])}] | {x['p_one_sided']:.4f} | {f3(x['p95_diff'])} "
                      f"[{f3(x['p95_ci_lo'])}, {f3(x['p95_ci_hi'])}] |")
        yrs = sorted(next(iter(out["by_year"][lab].values())))
        L_ += ["", "| 연도별 평균 | " + " | ".join(yrs) + " |", "|---|" + "---|" * len(yrs)]
        for k, x in out["by_year"][lab].items():
            L_.append(f"| {names_k[k]} | " + " | ".join(f3(x[y].get("mean")) for y in yrs) + " |")
        L_ += ["", "| 차이 | 2025-10-31까지 | 2025-11-03부터 | 보통주만 |", "|---|---|---|---|"]
        for k in out["differences"][lab]:
            cells = []
            for x in (out["date_split"][lab][k]["before"], out["date_split"][lab][k]["from"], out["common_only"]["differences"][lab][k]):
                cells.append("—" if not x.get("n") else f"{f3(x.get('mean'))} [{f3(x.get('ci_lo'))}, {f3(x.get('ci_hi'))}] (주문 {x['n']:,})")
            L_.append(f"| {names_k[k]} | " + " | ".join(cells) + " |")
        u = out["dollars"][lab]
        L_ += ["", f"- 달러 기준 누적 절감(A* 대 E*, 주문마다 (IS_E* − IS_A*) × 1,000주): 합계 {usd(u['Astar_saving_vs_Estar_total'])}달러, 연도별 "
               + ", ".join(f"{y} {usd(x)}" for y, x in u["by_year"].items()) + f". 누적값의 범위 {usd(u['cumulative_min'])} ~ {usd(u['cumulative_max'])}달러. "
               f"참고로 E* 대 S0는 {usd(u['Estar_saving_vs_S0_total'])}달러.", ""]
    L_ += ["### 대조 실행: 변동성·거래량 조건만 뒤집음 (판정 아님)", "",
           "| 가정 | 대조 평균 IS [95% CI] | 대조 − E* [95% CI] | A* − 대조 [95% CI] | 발동: 주문 (대조 / A*) |", "|---|---|---|---|---|"]
    for lab, x in out["control"].items():
        p_, q_, r_ = x["control_is"], x["control_minus_Estar"], x["Astar_minus_control"]
        L_.append(f"| {lab} | {p_['mean']:.3f} [{p_['ci_lo']:.3f}, {p_['ci_hi']:.3f}] | {f3(q_['mean'])} [{f3(q_['ci_lo'])}, {f3(q_['ci_hi'])}] | "
                  f"{f3(r_['mean'])} [{f3(r_['ci_lo'])}, {f3(r_['ci_hi'])}] | {100 * x['control_firing']['parents']:.1f}% / {100 * x['astar_firing']['parents']:.1f}% |")
    L_ += ["", "### 지연 점검 (결정은 막대 t, 실행은 막대 t + 1)", "", "| 가정 | 항목 | 평균 [95% CI] |", "|---|---|---|"]
    for lab, x in out["delay_check"].items():
        for k, y in x.items():
            if isinstance(y, dict):
                L_.append(f"| {lab} | {k.replace('_delay_minus_', ' 지연 − ').replace('_delay', ' 지연')} | {f3(y['mean'])} [{f3(y['ci_lo'])}, {f3(y['ci_hi'])}] |")
    gc = s["grid_counts"]
    L_ += ["", "## 자료 관련 건수", "",
           f"- 시작 호가가 유효하지 않아 뺀 주문 {s['dropped_parents']:,}건. 주문이 하나도 없는 종목·일 {s['symbol_days_without_parents']:,}개. SPY 파일이 없는 종목·일 "
           f"{s['symbol_days_without_ref']:,}개. 0 이하 가격의 시장가 주문 {c['market_at_nonpositive_price']}건.",
           f"- 결정 시점 {gc['points']:,}개 가운데 q가 정의된 시점 {gc['q_defined']:,}개, g {gc['g_defined']:,}개, v {gc['v_defined']:,}개. vol_pre가 없거나 0인 주문 "
           f"{gc['parents_vol_pre_undefined_or_zero']:,}건. 평소 체결량이 없는 종목·일 {s['adv']['adv_undefined'] + s['adv']['adv_zero']:,}개.",
           f"- 2025-10-31까지의 주문 {c['parents_before_cut']:,}건, 2025-11-03부터의 주문 {c['parents_from_cut']:,}건.",
           f"- 보통주가 아닌 종목·연도 {len(out['common_only']['excluded_symbol_years'])}개를 뺀 주문 {out['common_only']['parents']:,}건."]
    if mode == "rehearsal":
        k = out["checks"]
        L_ += ["", "## 리허설 점검", "",
               f"- 검증 실행(dev25)의 숫자와 최대 차이 {k['max_abs_diff']:.1e}(숫자 {k['numbers_compared']:,}개) → {'일치' if k['pass'] else '불일치'}. "
               f"A* 같음 {k['selection_equal']}, 봉인 경로를 읽지 않음 {k['no_sealed_path_read']}.",
               "- 이 보고서의 \"확인\" 여부는 검증 데이터로 낸 것이라 판정이 아니다."]
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
        if not selection()["improves"]:
            raise SystemExit("refused: " + NOT_IMPROVED)
        tasks, cal, pinfo = C.dev_tasks(VALID)
        roots = ("study15_ml_execution/loader15 sources (2019-2020)", L22.DEV_SPY_ROOT)
    sel = selection()
    if not model_ok():
        raise SystemExit("model23.json missing or differs from model23.sha256")
    model = C.model23()
    tasks, ainfo = C.with_adv(tasks, cal, workers)
    a = sel["astar"]
    names = list(C.BASE) + [a, C.CTRL + a]
    delay_runs = ["R0", a]
    par, OUT, _, info = C.simulate(tasks, names, delay_runs, workers, model=model)
    out = analyse(par, OUT, names, delay_runs, sel, mode)
    out["sim"] = {**info, **pinfo, "adv": ainfo}
    out["model_sha256"] = C.sha(MODEL)
    out["read"] = {"bars_root": roots[0].replace(os.path.expanduser("~"), "~"), "ref_root": roots[1].replace(os.path.expanduser("~"), "~")}
    if mode == "rehearsal":
        out["checks"] = rehearsal_checks(out, tasks)
    out.update({"mode": mode, "boot": {"B": E.BOOT_B, "block": E.BOOT_BLOCK, "seed": C.SEED}, "trials_before": TRIALS_BEFORE,
                "trials_added": 1 if mode == "sealed" else 0, "runtime_s": round(time.time() - t0, 1),
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
