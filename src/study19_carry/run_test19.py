"""Research 19 step 4: the single sealed test (holding months 2013-01..2026-02) and its rehearsal on development data.

  python run_test19.py --go          # sealed files, once; needs study19_carry/UNLOCK19.txt written by Kyle
  python run_test19.py --rehearsal   # the same code path on the build files, development window, no verdict

A* = CS40-M (development, FREEZE19.md). Verdict on monthly returns net of 4 bp: H1 Sharpe(C0) > 0 and H2 Sharpe(A*) -
Sharpe(C0) > 0; one-sided bootstrap p = share of resamples <= 0 (circular 12-month blocks, 10,000, seed 19, the same
months for both series); Holm at 0.05 (the smaller p at 0.025, then 0.05). With A* = C0 only H1 is tested, at 0.05.
Order in --go: guard (nothing is read) -> load build + sealed -> sha256 of every file against the manifests -> plan from
dates and missing values only (every month with an exit day and a portfolio) -> development months 1990-01..2012-11
recomputed from the full data and compared with results/dev19_monthly.csv -> only then the sealed statistics.
A failure before the last step stops without writing results.
Auxiliary (no verdict): the 22 candidates, cost 8 bp, drawdown, worst 12 months, crisis months, yearly sums, correlation
with the monthly VIX change, the Sharpe 0.5 line.
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine19 as G  # noqa: E402

RES = os.path.join(HERE, "results")
ASTAR = "CS40-M"                                                          # development selection (FREEZE19.md)
PHRASE = "연구19 테스트를 한 번 연다"
UNLOCK = os.path.join(HERE, "UNLOCK19.txt")
FREEZE = os.path.join(HERE, "FREEZE19.md")
DEV_SUMMARY = os.path.join(RES, "dev19_summary.json")
DEV_MONTHLY = os.path.join(RES, "dev19_monthly.csv")
A1 = os.path.join(RES, "a1_check19.json")
DOCS = (("SPEC19", "SPEC19.md"), ("SPEC19_ANSWERS", "SPEC19_ANSWERS.md"), ("SPEC19_ADDENDUM", "SPEC19_ADDENDUM.md"))
CORE = ("engine19.py", "run_test19.py", "SEALED_MANIFEST19.tsv", "SEALED_MANIFEST19_h10.tsv",
        "SEALED_MANIFEST19_candidates_20261110_104129.tsv",              # holds the sha256 of the Japan CD rate file
        "results/dev19_summary.json", "results/dev19_monthly.csv", "results/a1_check19.json")
MODES = {"sealed": {"files": ("test19_summary.json", "test19_report.md"), "parts": ("build", "sealed"), "window": G.SEALED,
                    "same_as_dev": ((1990, 1), (2012, 11)), "allow_incomplete": (),
                    "crisis": ((2015, 1), (2020, 3), (2024, 8)), "title": "봉인 테스트 (보유 월 2013-01 ~ 2026-02)"},
         "rehearsal": {"files": ("rehearsal19_summary.json", "rehearsal19_report.md"), "parts": ("build",), "window": G.DEV,
                       "same_as_dev": G.DEV, "allow_incomplete": ((2012, 12),),
                       "crisis": ((1992, 9), (1998, 8), (1998, 9), (1998, 10), (2008, 9), (2008, 10), (2008, 11), (2008, 12)),
                       "title": "리허설 (개발 데이터: 보유 월 1990-01 ~ 2012-12, 판정 아님)"}}
CRISIS_COLS = ("C0", "V80-M", "V80-D", "DV20-M", "DV20-D", "CS20-M", "CS20-D")   # fixed before the development run
SENTENCES = {"H1": {True: "캐리 트레이드는 처음 보는 13년에도 비용을 넘는 수익을 냈다",
                    False: "처음 보는 13년에서 캐리 트레이드의 수익은 0과 구분되지 않았다"},
             "H2": {True: "매크로 위험 지표로 포지션을 비우면 캐리 트레이드가 나아졌다",
                    False: "매크로 필터의 개선은 처음 보는 기간에서 확인되지 않았다"}}      # SPEC19 6, verbatim
BASELINE = 0.5
TRIALS_BEFORE = 773                                                       # 751 + the 22 development candidates (ADDENDUM 7)
PRED = (("H1 확인", 0.2), ("봉인 C0 샤프가 −0.2 ~ 0.5 사이", 0.6), ("A*가 C0이 아님", 0.6), ("H2 확인 (A*가 있을 때)", 0.2),
        ("A*의 봉인 최대 낙폭이 C0보다 작음 (A*가 있을 때)", 0.6), ("A*의 빈도가 M", 0.5),
        ("A* 필터의 D 버전이 봉인에서 M 버전보다 샤프가 높음", 0.4))            # SPEC19 7 and ADDENDUM 6, verbatim
DEV_KEYS = ("sharpe", "ann_return", "ann_vol", "max_drawdown", "worst_12m")
KEYS = ("sharpe", "ann_return", "ann_vol", "max_drawdown", "max_drawdown_daily", "worst_12m", "gross_sum", "cost_sum", "net_sum",
        "cost_share", "exits_per_year", "mid_exits_per_year", "mid_entries_per_year", "entries_per_year", "flat_share", "flat_months",
        "half1_sum", "half2_sum", "sharpe_cost2", "months")


# ---------------------------------------------------------------- guard
def frozen_files():
    """[(path, sha256)] from the table rows of FREEZE19.md."""
    if not os.path.isfile(FREEZE):
        return []
    pat = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
    with open(FREEZE, encoding="utf-8") as f:
        return [m.groups() for m in map(pat.match, f) if m]


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def guard():
    """Reasons not to run the sealed test; reads no data file."""
    why = []
    if any(os.path.exists(os.path.join(RES, f)) for f in MODES["sealed"]["files"]):
        why.append("test results already exist")
    if not os.path.isfile(UNLOCK) or open(UNLOCK, encoding="utf-8").read().strip() != PHRASE:
        why.append("no unlock phrase in UNLOCK19.txt")
    ff = frozen_files()
    if not ff or not all(any(p == c for p, _ in ff) for c in CORE):
        why.append("FREEZE19.md missing or without the core files")
    for p, h in ff:
        q = os.path.normpath(os.path.join(HERE, p))
        if not os.path.isfile(q) or G.sha(q) != h:
            why.append(f"frozen file differs: {p}")
    for doc, path in DOCS:
        s = os.path.join(HERE, f"{doc}.sha256")
        if not os.path.isfile(s) or not os.path.isfile(os.path.join(HERE, path)) or open(s).read().split()[0] != G.sha(os.path.join(HERE, path)):
            why.append(f"{path} differs from {doc}.sha256")
    dev = read_json(DEV_SUMMARY) if os.path.isfile(DEV_SUMMARY) else {}
    if ASTAR not in G.CANDS or dev.get("astar") != ASTAR:
        why.append("ASTAR differs from the development selection")
    if dev.get("engine19_sha256") != G.sha(os.path.join(HERE, "engine19.py")):
        why.append("engine19.py differs from the engine of the development run")
    if not os.path.isfile(A1) or read_json(A1).get("dem_path") != G.DEM_PATH:
        why.append("engine19.DEM_PATH differs from results/a1_check19.json")
    return why


# ---------------------------------------------------------------- plan (dates and missing values only)
def available(D, m):
    """(currencies usable in holding month m, rate months that were filled): the conditions of engine19.portfolio."""
    t = G.prv(m)
    i0, i1 = D.first[m], D.first.get(G.nxt(m), len(D.days) - 1)
    ok = [c for j, c in enumerate(G.CCY) if G.rate_at(D.rate.get(c, {}), t) is not None and D.fresh[i0, j] and D.fresh[i1, j]]
    fills = [f"{c} {G.lab(t)}" for c in ok + ["USD"] if t not in D.rate.get(c, {}) and G.rate_at(D.rate.get(c, {}), t) is not None]
    return ok, fills


def plan(D, window, allow_incomplete=()):
    """(problems, description). A problem stops the run before any return is computed."""
    ms = G.months(*window)
    prob = []
    gone = [G.lab(m) for m in ms if m not in D.first]
    if gone:
        prob.append(f"months without a business day: {gone[:3]}{'...' if len(gone) > 3 else ''}")
    inc = [G.lab(m) for m in G.incomplete(D, window) if m in D.first and m not in allow_incomplete]
    if inc:
        prob.append(f"months without an exit day: {inc}")
    here = [m for m in ms if m in D.first]
    none = [G.lab(m) for m in here if G.portfolio(D, m) is None]
    if none:
        prob.append(f"months without a portfolio: {none}")
    by, fills = {}, []
    for m in here:
        ok, f = available(D, m)
        by.setdefault(len(ok), []).append(m)
        fills += f
    hs = set(here)
    start = np.array([m in hs for m in D.ym])
    stale = int((start & ~np.where(np.isnan(D.S), True, D.fresh).all(axis=1)).sum())
    return prob, {"months": len(here), "first_day": D.days[0], "last_day": D.days[-1], "business_days_in_window": int(start.sum()),
                  "currencies_per_holding_month": {str(k): f"{len(v)} ({G.lab(v[0])}..{G.lab(v[-1])})" for k, v in sorted(by.items())},
                  "rate_fills_used": fills, "days_with_a_stale_quote": stale,
                  "incomplete_months": [G.lab(m) for m in G.incomplete(D, window)]}


def same_as_dev(D, sims, window, path):
    """Monthly net of the 22 candidates over the window against results/dev19_monthly.csv (written by dev19.py)."""
    with open(path, encoding="utf-8") as f:
        rows = [ln.rstrip("\n").split(",") for ln in f]
    if rows[0] != ["month"] + list(G.CANDS):
        return {"months": 0, "numbers": 0, "max_abs_diff": None, "equal": False, "note": "unexpected header"}
    dev = {r[0]: [float(x) for x in r[1:]] for r in rows[1:]}
    worst, n, ms = 0.0, 0, None
    for k, c in enumerate(G.CANDS):
        ms, g, cst = G.monthly(D, sims[c], window)
        if any(G.lab(m) not in dev for m in ms):
            return {"months": len(ms), "numbers": n, "max_abs_diff": None, "equal": False, "note": "a month is not in the development file"}
        d = np.abs((g - cst) - np.array([dev[G.lab(m)][k] for m in ms]))
        worst, n = max(worst, float(d.max()) if len(d) else 0.0), n + len(d)
    return {"months": len(ms), "numbers": n, "max_abs_diff": worst, "equal": bool(worst <= 1e-12 and n > 0)}


# ---------------------------------------------------------------- statistics
def evaluate(D, sims, sims2, window, astar, crisis):
    tab = G.table(D, sims, window, sims2)
    ms = [m for m in G.months(*window) if m in D.first]
    idx = G.boot_index(len(ms))
    dist = {"C0": G.boot_sharpe(tab["C0"]["net"], idx)}
    hyp = {"H1": ("샤프(C0) > 0", tab["C0"]["sharpe"], dist["C0"])}
    if astar != "C0":
        dist[astar] = G.boot_sharpe(tab[astar]["net"], idx)
        hyp["H2"] = (f"샤프({astar}) − 샤프(C0) > 0", tab[astar]["sharpe"] - tab["C0"]["sharpe"], dist[astar] - dist["C0"])
    holm = G.holm({h: float(np.mean(x[2] <= 0)) for h, x in hyp.items()})
    verdict = {}
    for h, (what, est, d) in hyp.items():
        r = G.one_sided(est, d, holm[h]["level"])
        r.update(what=what, level=holm[h]["level"], rejected=holm[h]["rejected"], result="확인" if holm[h]["rejected"] else "확인 안 됨",
                 sentence=SENTENCES[h][holm[h]["rejected"]])
        verdict[h] = r
    if astar == "C0":
        verdict["H2"] = {"result": "없음", "sentence": "개발에서 개선 없음 (A* = C0)"}
    dv = G.vix_change(D, window)
    line = {}
    for c, d in dist.items():
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        line[c] = {"sharpe": tab[c]["sharpe"], "ci95": [lo, hi], "above": bool(tab[c]["sharpe"] > BASELINE),
                   "ci_contains": bool(lo <= BASELINE <= hi), "share_of_resamples_above": float(np.mean(d > BASELINE))}
    years = sorted({m[0] for m in ms})
    yearly = {str(y): {c: float(sum(x for x, m in zip(tab[c]["net"], ms) if m[0] == y)) for c in G.CANDS} for y in years}
    cols = list(CRISIS_COLS) + ([astar] if astar not in CRISIS_COLS else [])
    return {"hypotheses": len(hyp), "holm_levels": sorted(v["level"] for v in verdict.values() if "level" in v), "verdict": verdict,
            "candidates": {c: {k: tab[c][k] for k in KEYS} for c in G.CANDS}, "monthly_net": {c: tab[c]["net"] for c in G.CANDS},
            "months": [G.lab(m) for m in ms],
            "crisis": {"columns": cols, "rows": {G.lab(m): [tab[c]["net"][ms.index(m)] for c in cols] for m in crisis if m in ms}},
            "yearly": yearly, "vix_change_corr": {c: G.corr(tab[c]["net"], dv) for c in G.CANDS}, "baseline_0.5": line}


def missing(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def versus_dev(S, astar_from_rule):
    """Rehearsal: every number of the development summary against the same number here."""
    dev = read_json(DEV_SUMMARY)
    n, worst, other = 0, 0.0, []
    for c in G.CANDS:
        for k in KEYS:
            a, b = dev["candidates"][c][k], S["candidates"][c][k]
            if missing(a) or missing(b):
                if not (missing(a) and missing(b)):
                    other.append(f"{c}.{k}")
                continue
            n, worst = n + 1, max(worst, abs(a - b))
    for m, row in dev["crisis"]["rows"].items():
        if dev["crisis"]["columns"] != S["crisis"]["columns"] or m not in S["crisis"]["rows"]:
            other.append(f"crisis {m}")
            continue
        n, worst = n + len(row), max(worst, max(abs(a - b) for a, b in zip(row, S["crisis"]["rows"][m])))
    return {"numbers": n, "max_abs_diff": worst, "not_comparable": other, "astar_from_rule": astar_from_rule,
            "astar_matches": bool(astar_from_rule == dev["astar"] == S["astar"])}


def predictions(S):
    """PREDICTIONS19.md against the outcome, item by item (None where the item does not apply)."""
    t, v, a = S["candidates"], S["verdict"], S["astar"]
    f, q = G.kind(a)
    has = a != "C0"
    out = [v["H1"]["rejected"], -0.2 <= t["C0"]["sharpe"] <= 0.5, has, v["H2"]["rejected"] if has else None,
           t[a]["max_drawdown"] < t["C0"]["max_drawdown"] if has else None, q == "M" if has else None,
           t[f"{f}-D"]["sharpe"] > t[f"{f}-M"]["sharpe"] if has else None]
    return [{"item": x, "probability": p, "outcome": None if o is None else bool(o)} for (x, p), o in zip(PRED, out)]


# ---------------------------------------------------------------- report
def pc(x, d=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.{d}f}"


def share(x, d=0):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def render(S):
    t, v, p, a = S["candidates"], S["verdict"], S["plan"], S["astar"]
    sealed = S["mode"] == "sealed"
    lv = ", ".join(f"{x:g}" for x in S["holm_levels"])
    if S["hypotheses"] == 2:
        holm = f"가설 2개에 Holm을 썼다(수준 {lv}): p가 작은 가설을 0.025로, 그다음을 0.05로 본다(첫 가설이 통과하지 못하면 둘째도 통과하지 못한다)."
    else:
        holm = f"가설은 H1 하나이고 수준은 {lv}다(A* = C0라서 H2가 없다)."
    L = [f"# 연구 19 · {S['title']}", "",
         f"- 보유 월 {S['months'][0]} ~ {S['months'][-1]}, {len(S['months'])}개월. A* = {a}. 수익은 자본 1달러 기준, 비용 4bp 차감 후, 복리 없음.",
         f"- 판정: 월 수익의 순환 블록 부트스트랩(블록 12개월, 10,000회, seed 19) 단측 p = 다시 뽑은 통계량이 0 이하인 비율. {holm}",
         "- 하한은 부트스트랩 분포의 백분위다: 각 가설의 수준 하한과 2.5% 하한."]
    if not sealed:
        L.append("- **리허설이다.** 개발 데이터로 같은 코드 경로를 돌린 것이고, 아래 판정 표는 판정이 아니다.")
    L += ["", "## 판정" if sealed else "## 판정 표 (리허설, 판정 아님)", "",
          "| 가설 | 내용 | 추정값 | p | Holm 수준 | 수준 하한 | 2.5% 하한 | 결과 | 문장 |", "|---|---|---|---|---|---|---|---|---|"]
    for h in ("H1", "H2"):
        x = v[h]
        if "p" in x:
            L.append(f"| {h} | {x['what']} | {x['estimate']:+.4f} | {x['p']:.4f} | {x['level']:.4f} | {x['bound_at_level']:+.4f} | {x['bound_97_5']:+.4f} | "
                     f"{x['result']} | \"{x['sentence']}\" |")
        else:
            L.append(f"| {h} | — | — | — | — | — | — | {x['result']} | {x['sentence']} |")
    L += ["", "## 보조 (판정 아님)", "", "### 기준선 0.5와 비교", "", "| 후보 | 샤프 | 95% 구간 | 0.5 위 | 구간이 0.5 포함 |", "|---|---|---|---|---|"]
    for c, x in S["baseline_0.5"].items():
        L.append(f"| {c} | {x['sharpe']:.3f} | [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] | {'예' if x['above'] else '아니오'} | {'예' if x['ci_contains'] else '아니오'} |")
    L += ["", "### 후보 22개", "",
          "| 후보 | 샤프 | 연수익 | 연변동성 | 최대 낙폭 | 최악 12개월 | 비용 비율 | 정리/년 | 비운 날 | 8bp 샤프 | VIX 변화 상관 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in G.CANDS:
        r, k = t[c], S["vix_change_corr"][c]
        L.append(f"| {c} | {r['sharpe']:.3f} | {pc(r['ann_return'])} | {100 * r['ann_vol']:.2f} | {100 * r['max_drawdown']:.1f} | {pc(r['worst_12m'], 1)} | "
                 f"{share(r['cost_share'])} | {r['exits_per_year']:.2f} | {share(r['flat_share'], 1)} | {r['sharpe_cost2']:.3f} | "
                 f"{'—' if k is None else f'{k:+.2f}'} |")
    L += ["", "### 빈도 비교 (필터 7개 × M·W·D)", "",
          "| 필터 | 샤프 M | 샤프 W | 샤프 D | 정리/년 M | W | D | 비용 비율 M | W | D | 비운 날 M | W | D |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in G.FILTERS:
        r = [t[f"{f}-{q}"] for q in G.FREQS]
        L.append(f"| {f} | " + " | ".join(f"{x['sharpe']:.3f}" for x in r) + " | " + " | ".join(f"{x['exits_per_year']:.2f}" for x in r) + " | "
                 + " | ".join(share(x["cost_share"]) for x in r) + " | " + " | ".join(share(x["flat_share"], 1) for x in r) + " |")
    L += ["", "### 위기 달의 월 수익 (%)", "", "| 달 | " + " | ".join(S["crisis"]["columns"]) + " |", "|---|" + "---|" * len(S["crisis"]["columns"])]
    for m, row in S["crisis"]["rows"].items():
        L.append(f"| {m} | " + " | ".join(pc(x) for x in row) + " |")
    names = ["C0"] + ([a] if a != "C0" else [])
    L += ["", "### 연도별 수익 합 (%)", "", "| 연도 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for y, row in S["yearly"].items():
        L.append(f"| {y} | " + " | ".join(pc(row[c], 1) for c in names) + " |")
    L += ["", "### 개발 구간과 비교", "", "| 후보 | 구간 | 샤프 | 연수익 | 연변동성 | 최대 낙폭 | 최악 12개월 |", "|---|---|---|---|---|---|---|"]
    for c in names:
        for label, r in (("개발 1990–2012 (dev19)", S["development"][c]), ("이번 실행", t[c])):
            L.append(f"| {c} | {label} | {r['sharpe']:.3f} | {pc(r['ann_return'])} | {100 * r['ann_vol']:.2f} | {100 * r['max_drawdown']:.1f} | {pc(r['worst_12m'], 1)} |")
    word = {True: "맞음", False: "틀림", None: "해당 없음"}
    L += ["", "### 예측표 대조" + ("" if sealed else " (리허설: 개발 데이터라 의미 없음)"), "", "| 항목 | 예측 확률 | 결과 |", "|---|---|---|"]
    L += [f"| {x['item']} | {x['probability']} | {word[x['outcome']]} |" for x in S["predictions"]]
    s = S["same_as_dev"]
    exits = (f"청산일이 자료 밖인 달: {', '.join(p['incomplete_months'])} (리허설에서만 허용, 마지막 자료일로 평가)." if p["incomplete_months"]
             else "청산일도 모두 있다.")
    L += ["", "## 실행 점검", "",
          f"- 계획(날짜와 결측만): {p['months']}개월 모두 포지션이 있다. {exits} 달력 {p['first_day']} ~ {p['last_day']}.",
          "- 보유 월별 통화 수: " + "; ".join(f"{k}개 {x}" for k, x in p["currencies_per_holding_month"].items()),
          f"- 금리 채움(A4) 사용: {', '.join(p['rate_fills_used']) or '없음'}. 직전 환율을 쓴 날: {p['days_with_a_stale_quote']}일.",
          f"- 개발 구간 재현: {s['months']}개월 × 22개 후보 = {s['numbers']:,}개 월 수익이 results/dev19_monthly.csv와 최대 차이 {s['max_abs_diff']:.1e}."]
    if "vs_dev" in S:
        z = S["vs_dev"]
        L.append(f"- 개발 요약과 비교: 숫자 {z['numbers']:,}개, 최대 차이 {z['max_abs_diff']:.1e}, 비교 못 한 항목 {len(z['not_comparable'])}개. "
                 f"선택 규칙을 다시 적용한 A* = {z['astar_from_rule']} ({'일치' if z['astar_matches'] else '불일치'}).")
    L.append(f"- 읽은 파일 {S['read']['files']}개 (sealed 폴더 {S['read']['sealed']}개, raw 폴더 {S['read']['raw']}개), sha256 모두 일치. 실행 {S['runtime_s']:.1f} s")
    if S["trials"]:
        L.append(f"- 시행 수: {S['trials']['before']} → {S['trials']['after']} (봉인 판정 {S['trials']['added']}개).")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- run
def execute(mode, astar=None, cfg=None, res=None, dev_monthly=None):
    """Runs one mode and writes its two files; returns 0, or 1 after printing why nothing was written."""
    t0 = time.time()
    cfg = dict(MODES[mode], **(cfg or {}))
    astar, res, dev_monthly = ASTAR if astar is None else astar, RES if res is None else res, DEV_MONTHLY if dev_monthly is None else dev_monthly
    out = [os.path.join(res, f) for f in cfg["files"]]
    if any(os.path.exists(p) for p in out):
        print(f"stop: {cfg['files']} already exist (nothing is overwritten)")
        return 1
    D, read = G.load(cfg["parts"])
    bad = G.verify(read)
    if bad:
        print("stop: sha256 differs from the manifests:", bad)
        return 1
    prob, desc = plan(D, cfg["window"], cfg["allow_incomplete"])
    if mode == "sealed" and desc["months"] != len(G.months(*cfg["window"])):
        prob.append("the window is not fully covered")
    if prob:
        print("stop before any return is computed:", prob)
        return 1
    sims, sims2 = G.run_all(D), G.run_all(D, 2 * G.COST)
    same = same_as_dev(D, sims, cfg["same_as_dev"], dev_monthly)
    if not same["equal"]:
        print("stop: the development months recomputed here differ from results/dev19_monthly.csv:", same)
        return 1
    S = {"written": None, "mode": mode, "title": cfg["title"], "astar": astar, "window": [G.lab(cfg["window"][0]), G.lab(cfg["window"][1])],
         "cost": G.COST, "dem_path": G.DEM_PATH, "boot": {"draws": G.BOOT_B, "block_months": G.BOOT_BLOCK, "seed": G.BOOT_SEED},
         "plan": desc, "same_as_dev": same}
    S.update(evaluate(D, sims, sims2, cfg["window"], astar, cfg["crisis"]))
    dev = read_json(DEV_SUMMARY)["candidates"]
    S["development"] = {c: {k: dev[c][k] for k in DEV_KEYS} for c in S["baseline_0.5"]}
    S["predictions"] = predictions(S)
    S["trials"] = {"before": TRIALS_BEFORE, "added": S["hypotheses"], "after": TRIALS_BEFORE + S["hypotheses"]} if mode == "sealed" else None
    if mode == "rehearsal":
        S["vs_dev"] = versus_dev(S, G.select_astar(G.table(D, sims, cfg["window"], sims2))[0])
    S["read"] = {"files": len(read), "sealed": sum(f"{os.sep}sealed{os.sep}" in p for p in read), "raw": sum(f"{os.sep}raw{os.sep}" in p for p in read),
                 "paths": [os.path.relpath(p, G.DATA) for p in read]}
    S["engine19_sha256"] = G.sha(os.path.join(os.path.dirname(os.path.abspath(G.__file__)), "engine19.py"))
    S["written"], S["runtime_s"] = dt.datetime.now().isoformat(timespec="seconds"), round(time.time() - t0, 1)
    text, js = render(S), json.dumps(S, indent=1, ensure_ascii=False)       # both built before anything is written
    for p, body in zip(out, (js, text)):
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)
    print(text)
    return 0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--go", action="store_true")
    g.add_argument("--rehearsal", action="store_true")
    a = ap.parse_args()
    if a.rehearsal:
        return execute("rehearsal")
    why = guard()
    if why:
        print("guard: the sealed test does not run:", why)
        return 1
    return execute("sealed")


if __name__ == "__main__":
    sys.exit(main())
