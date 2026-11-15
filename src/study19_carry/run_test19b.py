"""Research 19B: the carry trade without a filter (C0) over the longer history (SPEC19B.md). One sealed run and a rehearsal.

  python run_test19b.py --go          # holding months 1979-02..2026-02, once; needs UNLOCK19B.txt written by Kyle
  python run_test19b.py --rehearsal   # 1990-01..2026-02 only: must reproduce the stored values of research 19

engine19 is used unchanged (rules of SPEC19, ADDENDUM, ANSWERS). What is new here: the calendar starts on 1979-01-01
(first formation month 1979-01, first holding month 1979-02), the Fed's 1971-1989 DM/US$ file stands in front of the
1990-1998 one, and the cost is 8 bp of the gross notional for holding months before 1990 and 4 bp from 1990-01.
Verdict on monthly returns net of cost: H1b Sharpe > 0 over 1979-02..1989-12 and H1c Sharpe > 0 over 1979-02..2026-02;
one-sided bootstrap p = share of resamples <= 0 (circular 12-month blocks, 10,000, seed 192); Holm at 0.025 (the
smaller p at 0.0125, then 0.025), because the question is looked at a second time.
Order in --go: guard (nothing is read) -> load -> sha256 of every file against the manifests -> plan from dates and
missing values only -> months 1990-01..2012-11 and 2013-01..2026-02 compared with the stored values of research 19 ->
only then the statistics. A failure before the last step stops without writing results.
The rehearsal starts its calendar on 1989-12-01 like research 19, so it computes no return before 1990.
Auxiliary (no verdict): decades, cost before 1990 at 4 and 16 bp, AUD and NZD only from their floats, drawdown, worst 12
months, yearly sums, the monthly series, and the 1979-1989 mark rate against the OECD month-end series.
"""
import argparse
import copy
import csv
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
PHRASE = "연구19B 테스트를 한 번 연다"
UNLOCK = os.path.join(HERE, "UNLOCK19B.txt")
FREEZE = os.path.join(HERE, "FREEZE19B.md")
CHECK = os.path.join(RES, "check19b_quality.json")
DEV_MONTHLY = os.path.join(RES, "dev19_monthly.csv")                       # research 19, development (C0 column)
TEST_SUMMARY = os.path.join(RES, "test19_summary.json")                   # research 19, sealed run (monthly_net of C0)
DEM_OLD = "H10_DEM_7189"                                                  # Fed dat89_ge.txt, candidates/build (fetch19b_h10.py)
OECD_DEM = "CCUSSP01DEM650N"                                              # OECD month-end USD per mark in euro units (ANSWERS A1)
DEM_WINDOW = ((1979, 1), (1989, 12))
START_FORMATION = (1979, 1)                                               # first formation month with six currencies (check19b.py)
CAL_START, REH_START = "1979-01-01", "1989-12-01"
W_B, W_C = ((1979, 2), (1989, 12)), ((1979, 2), (2026, 2))
SWITCH = (1990, 1)                                                        # holding months from here cost 4 bp, earlier ones PRE_COST
PRE_COST, PRE_ROBUST = 0.0008, (0.0004, 0.0016)
BOOT_SEED, ALPHA = 192, 0.025                                             # Holm levels 0.0125 and 0.025
DECADES = (("1979–89", (1979, 2), (1989, 12)), ("1990–99", (1990, 1), (1999, 12)), ("2000–09", (2000, 1), (2009, 12)),
           ("2010–19", (2010, 1), (2019, 12)), ("2020–26", (2020, 1), (2026, 2)))
FLOAT_FROM = {"AUD": (1984, 1), "NZD": (1985, 4)}                         # sensitivity: first holding month of each
REPRODUCE = {"dev": ((1990, 1), (2012, 11)), "sealed": ((2013, 1), (2026, 2))}
SENTENCES = {"H1c": {True: "47년 전체로 보면 캐리 트레이드는 비용을 넘는 수익을 냈다. 다만 최근 13년에는 거의 0이었다",
                     False: "47년 전체로 봐도 캐리 트레이드의 수익은 0과 확실히 구분되지 않았다"},
             "H1b": {True: "처음 보는 1980년대에서도 캐리 트레이드는 수익을 냈다",
                     False: "1980년대만으로는 판정할 수 없었다"}}             # SPEC19B 5, verbatim
PRED = (("1979–89 샤프가 0보다 큼", 0.75), ("H1b 확인", 0.2), ("전체 기간 샤프가 0.15 ~ 0.40 사이", 0.6), ("H1c 확인", 0.3))   # SPEC19B 6
TRIALS_BEFORE = 775
FILES = {"sealed": ("test19b_summary.json", "test19b_report.md"), "rehearsal": ("rehearsal19b_summary.json", "rehearsal19b_report.md")}
CORE = ("engine19.py", "run_test19b.py", "SPEC19B.md", "SEALED_MANIFEST19.tsv", "SEALED_MANIFEST19_h10.tsv", "SEALED_MANIFEST19B_h10.tsv",
        "SEALED_MANIFEST19_candidates_20261110_104129.tsv", "SEALED_MANIFEST19_candidates_20261110_104201.tsv",
        "results/dev19_monthly.csv", "results/test19_summary.json", "results/check19b_quality.json")


# ---------------------------------------------------------------- data
def load_b(parts, start, root=None, cand=None):
    """(Data, files read): engine19.load with a chosen calendar start and both Fed DM/US$ files (1971-1989, 1990-1998).
    VIX and the Baa spread are not loaded: C0 has no filter."""
    root, cand = G.DATA if root is None else root, G.CAND if cand is None else cand
    read, fx = [], {}
    for c, (s, usd) in G.FX.items():
        rows = G.series_rows(s, parts, root, read)
        if c == "EUR":
            old, new = G.series_rows(DEM_OLD, ("build",), cand, read), G.series_rows(G.DEM_SERIES, ("build",), cand, read)
            if old and new and old[-1][0] >= new[0][0]:
                raise ValueError("the two Fed DM/US$ files overlap")
            fx[c] = G.eur_slot(rows, old + new)
        else:
            fx[c] = G.fx_usd_per_unit(rows, usd)
    r = {k: G.monthly_rate(G.series_rows(f"IR3TIB01{v}M156N", parts, root, read)) for k, v in G.RATE.items()}
    de = G.monthly_rate(G.series_rows("IR3TIB01DEM156N", ("build",), root, read))
    r["EUR"] = G.splice(de, r["EUR"], G.EUR_RATE_FROM)
    r["JPY"] = G.splice(G.monthly_rate(G.series_rows(G.JP_CD_SERIES, ("build",), cand, read)), r["JPY"], G.JP_IB_FROM)
    return G.make_data(fx, r, {}, {}, start, None), read


def available(D, m):
    """Currencies usable in holding month m by the conditions of engine19.portfolio (presence only, no ranking)."""
    t = G.prv(m)
    if m not in D.first or t not in D.first or G.rate_at(D.rate.get("USD", {}), t) is None:
        return []
    i0, i1 = D.first[m], D.first.get(G.nxt(m), len(D.days) - 1)
    return [c for j, c in enumerate(G.CCY) if G.rate_at(D.rate.get(c, {}), t) is not None and D.fresh[i0, j] and D.fresh[i1, j]]


def filled(D, m):
    """Rates of formation month m - 1 that the one-month fill of ANSWERS A4 supplies (for the usable currencies and USD)."""
    t = G.prv(m)
    return [f"{c} {G.lab(t)}" for c in available(D, m) + ["USD"] if t not in D.rate.get(c, {}) and G.rate_at(D.rate.get(c, {}), t) is not None]


def plan(D, window, allow_incomplete=()):
    """(problems, description) from dates and missing values only; a problem stops the run before any return."""
    ms = G.months(*window)
    prob = []
    gone = [G.lab(m) for m in ms if m not in D.first]
    if gone:
        prob.append(f"months without a business day: {gone[:3]}{'...' if len(gone) > 3 else ''}")
    inc = [G.lab(m) for m in G.incomplete(D, window) if m in D.first and m not in allow_incomplete]
    if inc:
        prob.append(f"months without an exit day: {inc}")
    here = [m for m in ms if m in D.first]
    n = {m: len(available(D, m)) for m in here}
    none = [G.lab(m) for m in here if n[m] < 4]
    if none:
        prob.append(f"months with fewer than four currencies (no portfolio): {none[:6]}")
    years = {}
    for m in here:
        years.setdefault(m[0], []).append(n[m])
    hs = set(here)
    start = np.array([m in hs for m in D.ym])
    stale = int((start & ~np.where(np.isnan(D.S), True, D.fresh).all(axis=1)).sum())
    fills = [x for m in here for x in filled(D, m)]
    return prob, {"months": len(here), "first_day": D.days[0], "last_day": D.days[-1], "business_days_in_window": int(start.sum()),
                  "currencies_by_year": {str(y): (f"{min(v)}" if min(v) == max(v) else f"{min(v)}–{max(v)}") for y, v in sorted(years.items())},
                  "months_with_fewer_than_six": [G.lab(m) for m in here if n[m] < 6], "rate_fills_used": fills,
                  "days_with_a_stale_quote": stale, "incomplete_months": [G.lab(m) for m in G.incomplete(D, window)]}


def dem_quality(D, oecd_rows, window=DEM_WINDOW):
    """Sealed run only (it uses values before 1990): the month-end value of the mark slot against the OECD month-end
    series, the comparison of ANSWERS A1, here for the older Fed file. Not a verdict and it changes nothing."""
    oecd = {G.ym(d): float(v) for d, v in oecd_rows if v}
    j = G.CCY.index("EUR")
    last = {}
    for i, m in enumerate(D.ym):
        if D.fresh[i, j]:
            last[m] = float(D.S[i, j])
    d = sorted(abs(last[m] / oecd[m] - 1.0) for m in G.months(*window) if m in last and m in oecd)
    if not d:
        return None
    return {"months": len(d), "median_abs_rel_diff": float(np.median(d)), "max_abs_rel_diff": float(d[-1]), "limit_of_answers_a1": 0.005}


def without_pegged(D):
    """Sensitivity: AUD and NZD enter the universe at the holding months of FLOAT_FROM (their rates before are dropped)."""
    D2 = copy.copy(D)
    D2.rate = dict(D.rate)
    for c, first in FLOAT_FROM.items():
        D2.rate[c] = {t: v for t, v in D.rate[c].items() if t >= G.prv(first)}
    return D2


# ---------------------------------------------------------------- returns
def simulate(D, costs):
    """{cost: engine19.simulate of C0}; the P&L is the same for every cost, only the cost column differs."""
    ports = G.portfolios(D)
    return {c: G.simulate(D, None, c, ports) for c in costs}


def series(D, sims, window, pre=PRE_COST):
    """(months, net, gross, cost) of C0: the cost of a holding month is `pre` before SWITCH and engine19.COST from it."""
    ms, g, post = G.monthly(D, sims[G.COST], window)
    before = G.monthly(D, sims[pre], window)[2]
    cost = np.where(np.array([m < SWITCH for m in ms], dtype=bool), before, post)
    return ms, g - cost, g, cost


def stored(dev_path=None, test_path=None):
    """C0 monthly net of research 19: ({month: value} development, {month: value} sealed run)."""
    with open(DEV_MONTHLY if dev_path is None else dev_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    k = rows[0].index("C0")
    with open(TEST_SUMMARY if test_path is None else test_path, encoding="utf-8") as f:
        t = json.load(f)
    return {r[0]: float(r[k]) for r in rows[1:]}, dict(zip(t["months"], t["monthly_net"]["C0"]))


def compare(ms, net, ref, window):
    """Months of the window against the stored values: count, largest difference, equal (exactly)."""
    k = [i for i, m in enumerate(ms) if window[0] <= m <= window[1]]
    if not k or any(G.lab(ms[i]) not in ref for i in k):
        return {"months": len(k), "max_abs_diff": None, "equal": False}
    d = max(abs(float(net[i]) - ref[G.lab(ms[i])]) for i in k)
    return {"months": len(k), "max_abs_diff": d, "equal": bool(d == 0.0 and len(k) == len(G.months(*window)))}


def window_stats(ms, net, gross, cost, a, b):
    k = np.array([a <= m <= b for m in ms], dtype=bool)
    x = net[k]
    gs, cs = float(gross[k].sum()), float(cost[k].sum())
    return {"months": int(k.sum()), "sharpe": G.sharpe(x), "ann_return": float(x.mean() * 12), "ann_vol": float(x.std(ddof=1) * math.sqrt(12)),
            "max_drawdown": G.max_drawdown(x), "worst_12m": G.worst12(x), "net_sum": float(x.sum()), "gross_sum": gs, "cost_sum": cs,
            "cost_share": cs / gs if gs > 0 else None}


def evaluate(D, sims, cfg):
    wb, wc = cfg["w_b"], cfg["w_c"]
    ms, net, g, c = series(D, sims, wc)
    kb = np.array([wb[0] <= m <= wb[1] for m in ms], dtype=bool)
    hyp = {"H1b": ("샤프 > 0, " + f"{G.lab(wb[0])} ~ {G.lab(wb[1])}", net[kb]), "H1c": ("샤프 > 0, " + f"{G.lab(wc[0])} ~ {G.lab(wc[1])}", net)}
    dist = {h: G.boot_sharpe(x, G.boot_index(len(x), seed=BOOT_SEED)) for h, (_, x) in hyp.items()}
    holm = G.holm({h: float(np.mean(dist[h] <= 0)) for h in hyp}, ALPHA)
    verdict = {}
    for h, (what, x) in hyp.items():
        r = G.one_sided(G.sharpe(x), dist[h], holm[h]["level"])
        r.update(what=what, months=int(len(x)), level=holm[h]["level"], rejected=holm[h]["rejected"],
                 result="확인" if holm[h]["rejected"] else "확인 안 됨", sentence=SENTENCES[h][holm[h]["rejected"]])
        verdict[h] = r
    robust = {}
    for pre in cfg["pre_robust"]:
        _, n2, g2, c2 = series(D, sims, wc, pre)
        robust[f"{pre * 1e4:g}bp"] = {"H1b": window_stats(ms, n2, g2, c2, *wb), "H1c": window_stats(ms, n2, g2, c2, *wc)}
    D2 = without_pegged(D)
    s2 = simulate(D2, (G.COST, PRE_COST))
    ms2, n2, g2, c2 = series(D2, s2, wc)
    count2 = [len(available(D2, m)) for m in ms2]
    floats = {"first_holding_month": {k: G.lab(v) for k, v in FLOAT_FROM.items()}, "H1b": window_stats(ms2, n2, g2, c2, *wb),
              "H1c": window_stats(ms2, n2, g2, c2, *wc), "months_with_two_and_two": int(sum(4 <= x < 6 for x in count2)),
              "months_without_a_portfolio": int(sum(x < 4 for x in count2))}
    years = sorted({m[0] for m in ms})
    return {"hypotheses": len(hyp), "holm_levels": sorted(v["level"] for v in verdict.values()), "verdict": verdict,
            "windows": {"H1b": window_stats(ms, net, g, c, *wb), "H1c": window_stats(ms, net, g, c, *wc)},
            "decades": {name: window_stats(ms, net, g, c, a, b) for name, a, b in cfg["decades"]},
            "cost_robustness_before_1990": robust, "floating_only": floats,
            "yearly": {str(y): float(sum(x for x, m in zip(net, ms) if m[0] == y)) for y in years},
            "months": [G.lab(m) for m in ms], "monthly_net": [float(x) for x in net], "monthly_gross": [float(x) for x in g],
            "monthly_cost": [float(x) for x in c]}


def predictions(S):
    v, w = S["verdict"], S["windows"]
    out = [w["H1b"]["sharpe"] > 0, v["H1b"]["rejected"], 0.15 <= w["H1c"]["sharpe"] <= 0.40, v["H1c"]["rejected"]]
    return [{"item": x, "probability": p, "outcome": bool(o)} for (x, p), o in zip(PRED, out)]


# ---------------------------------------------------------------- guard
def frozen_files():
    """[(path, sha256)] from the table rows of FREEZE19B.md."""
    if not os.path.isfile(FREEZE):
        return []
    pat = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
    with open(FREEZE, encoding="utf-8") as f:
        return [m.groups() for m in map(pat.match, f) if m]


def guard():
    """Reasons not to run the sealed test; reads no data file."""
    why = []
    if any(os.path.exists(os.path.join(RES, f)) for f in FILES["sealed"]):
        why.append("test results already exist")
    if not os.path.isfile(UNLOCK) or open(UNLOCK, encoding="utf-8").read().strip() != PHRASE:
        why.append("no unlock phrase in UNLOCK19B.txt")
    ff = frozen_files()
    if not ff or not all(any(p == c for p, _ in ff) for c in CORE):
        why.append("FREEZE19B.md missing or without the core files")
    for p, h in ff:
        q = os.path.normpath(os.path.join(HERE, p))
        if not os.path.isfile(q) or G.sha(q) != h:
            why.append(f"frozen file differs: {p}")
    s, doc = os.path.join(HERE, "SPEC19B.sha256"), os.path.join(HERE, "SPEC19B.md")
    if not os.path.isfile(s) or not os.path.isfile(doc) or open(s).read().split()[0] != G.sha(doc):
        why.append("SPEC19B.md differs from SPEC19B.sha256")
    chk = json.load(open(CHECK, encoding="utf-8")) if os.path.isfile(CHECK) else {}
    if chk.get("start_formation_month") != G.lab(START_FORMATION) or W_B[0] != G.nxt(START_FORMATION) or W_C[0] != W_B[0]:
        why.append("the start month differs from results/check19b_quality.json")
    return why


# ---------------------------------------------------------------- report
def pc(x, d=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.{d}f}"


def share(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def row(name, r):
    return (f"| {name} | {r['months']} | {r['sharpe']:.3f} | {pc(r['ann_return'])} | {100 * r['ann_vol']:.2f} | {100 * r['max_drawdown']:.1f} | "
            f"{pc(r['worst_12m'], 1)} | {share(r['cost_share'])} |")


def render(S):
    v, p = S["verdict"], S["plan"]
    lv = ", ".join(f"{x:g}" for x in S["holm_levels"])
    L = ["# 연구 19B · 봉인 테스트 (C0, 보유 월 " + f"{S['months'][0]} ~ {S['months'][-1]})", "",
         f"- 필터 없는 캐리(C0), {len(S['months'])}개월. 수익은 자본 1달러 기준, 비용 차감 후, 복리 없음.",
         "- 비용: 1990년 전 보유 월은 총 노셔널의 8bp(자본의 16bp), 1990-01부터는 4bp(자본의 8bp).",
         "- 판정: 월 수익의 순환 블록 부트스트랩(블록 12개월, 10,000회, seed 192) 단측 p = 다시 뽑은 샤프가 0 이하인 비율. "
         f"가설 2개에 Holm을 썼다(수준 {lv}): 같은 질문을 두 번째로 보는 것이라 수준을 절반으로 낮췄다.",
         "- 하한은 부트스트랩 분포의 백분위다: 각 가설의 수준 하한과 2.5% 하한.", "",
         "## 판정", "", "| 가설 | 내용 | 개월 | 샤프 | p | Holm 수준 | 수준 하한 | 2.5% 하한 | 결과 | 문장 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for h in ("H1b", "H1c"):
        x = v[h]
        L.append(f"| {h} | {x['what']} | {x['months']} | {x['estimate']:+.4f} | {x['p']:.4f} | {x['level']:.4f} | {x['bound_at_level']:+.4f} | "
                 f"{x['bound_97_5']:+.4f} | {x['result']} | \"{x['sentence']}\" |")
    head = ["| 구간 | 개월 | 샤프 | 연수익 % | 변동성 % | 최대 낙폭 % | 최악 12개월 % | 비용 비율 |", "|---|---|---|---|---|---|---|---|"]
    L += ["", "## 보조 (판정 아님)", "", "### 구간별"] + [""] + head
    L += [row(f"H1b 구간 ({v['H1b']['what'].split(', ')[1]})", S["windows"]["H1b"]), row("전체", S["windows"]["H1c"])]
    L += [row(name, r) for name, r in S["decades"].items()]
    L += ["", "### 1990년 전 비용을 바꾸면", ""] + head
    for k, r in S["cost_robustness_before_1990"].items():
        L += [row(f"{k}, H1b 구간", r["H1b"]), row(f"{k}, 전체", r["H1c"])]
    f = S["floating_only"]
    L += ["", f"### 변동 환율 이전을 빼면 (AUD는 {f['first_holding_month']['AUD']}부터, NZD는 {f['first_holding_month']['NZD']}부터)", ""] + head
    L += [row("H1b 구간", f["H1b"]), row("전체", f["H1c"]), "",
          f"- 통화가 4 ~ 5개라 2개씩 잡은 달: {f['months_with_two_and_two']}개월. 포지션이 없는 달: {f['months_without_a_portfolio']}개월."]
    L += ["", "### 연도별 수익 합 (%)", "", "| 연도 | 수익 | 연도 | 수익 | 연도 | 수익 | 연도 | 수익 |", "|---|---|---|---|---|---|---|---|"]
    ys = list(S["yearly"].items())
    for i in range(0, len(ys), 4):
        cols = [s for y, x in ys[i:i + 4] for s in (y, pc(x, 1))]
        L.append("| " + " | ".join(cols + [""] * (8 - len(cols))) + " |")
    word = {True: "일어남", False: "일어나지 않음"}
    L += ["", "### 예측 대조 (SPEC19B 6절)", "", "| 항목 | 예측 확률 | 결과 |", "|---|---|---|"]
    L += [f"| {x['item']} | {x['probability']} | {word[x['outcome']]} |" for x in S["predictions"]]
    r = S["reproduced"]
    L += ["", "## 실행 점검", "",
          f"- 계획(날짜와 결측만): {p['months']}개월 모두 포지션과 청산일이 있다. 달력 {p['first_day']} ~ {p['last_day']}.",
          "- 연도별 통화 수: " + ", ".join(f"{y} {n}" for y, n in p["currencies_by_year"].items()),
          f"- 통화가 6개 미만인 달: {', '.join(p['months_with_fewer_than_six']) or '없음'}. 금리 채움(A4) 사용: {', '.join(p['rate_fills_used']) or '없음'}. "
          f"직전 환율을 쓴 날: {p['days_with_a_stale_quote']}일.",
          f"- 연구 19 재현: 개발 {r['dev']['months']}개월(1990-01 ~ 2012-11)과 봉인 {r['sealed']['months']}개월(2013-01 ~ 2026-02)의 C0 월 수익이 "
          f"저장된 값과 최대 차이 {max(r['dev']['max_abs_diff'], r['sealed']['max_abs_diff']):.1e}.",
          f"- 읽은 파일 {S['read']['files']}개, sha256 모두 일치. 실행 {S['runtime_s']:.1f} s",
          f"- 시행 수: {S['trials']['before']} → {S['trials']['after']} (가설 {S['trials']['added']}개)."]
    q = S.get("dem_quality")
    if q:
        L.insert(-2, f"- 마르크 환율(연준 1971–1989 파일)을 OECD 월말 값과 비교({q['months']}개월, 1979-01 ~ 1989-12): 차이 절댓값의 중앙값 "
                     f"{100 * q['median_abs_rel_diff']:.2f}%, 최대 {100 * q['max_abs_rel_diff']:.2f}% (ANSWERS A1의 기준은 0.5%).")
    return "\n".join(L) + "\n"


def render_rehearsal(S):
    a, b = S["pass_build"], S["pass_full"]
    L = ["# 연구 19B · 리허설 (1990-01 ~ 2026-02, 판정 아님)", "",
         "- 봉인 실행과 같은 코드 경로(load_b → simulate → series)로 C0 월 수익을 다시 계산해, 연구 19의 저장된 값과 비교했다.",
         f"- 달력은 연구 19처럼 {S['calendar_start']}에서 시작한다. 1990년 전 보유 월의 수익은 계산하지 않았다(1989-12 포트폴리오 없음: "
         f"{'확인' if S['no_portfolio_before_1990'] else '실패'}).", "",
         "| 실행 | 자료 | 비교 구간 | 개월 | 최대 차이 | 같음 |", "|---|---|---|---|---|---|",
         f"| A | build만 | 개발 1990-01 ~ 2012-12 (dev19_monthly.csv) | {a['dev']['months']} | {a['dev']['max_abs_diff']:.1e} | {'예' if a['dev']['equal'] else '아니오'} |",
         f"| B | build + sealed | 봉인 2013-01 ~ 2026-02 (test19_summary.json) | {b['sealed']['months']} | {b['sealed']['max_abs_diff']:.1e} | {'예' if b['sealed']['equal'] else '아니오'} |",
         f"| B | build + sealed | 개발 1990-01 ~ 2012-11 | {b['dev']['months']} | {b['dev']['max_abs_diff']:.1e} | {'예' if b['dev']['equal'] else '아니오'} |", "",
         "- 2012-12는 A에서만 비교한다. 연구 19의 저장 값이 2012-12-31 환율로 평가한 값이라서(DEVIATIONS19 1-11), 전체 자료로 계산한 B와는 다르다.",
         f"- 저장된 값 {a['dev']['months']} + {b['sealed']['months']}개가 모두 차이 0으로 재현됐다: {'예' if S['equal'] else '아니오'}.",
         f"- 읽은 파일: A {S['read']['build']}개, B {S['read']['full']}개. sha256 모두 일치. 실행 {S['runtime_s']:.1f} s"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- run
CFG = {"cal_start": CAL_START, "w_b": W_B, "w_c": W_C, "decades": DECADES, "pre_robust": PRE_ROBUST, "reproduce": REPRODUCE,
       "reh_start": REH_START, "reh_dev": ((1990, 1), (2012, 12)), "reh_full": ((1990, 1), (2026, 2))}


def write(res, mode, S, text):
    out = [os.path.join(res, f) for f in FILES[mode]]
    js = json.dumps(S, indent=1, ensure_ascii=False)                       # built before anything is written
    for p, body in zip(out, (js, text)):
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)
    print(text)


def run_sealed(cfg=None, res=None, dev_path=None, test_path=None):
    """The sealed run; returns 0, or 1 after printing why nothing was written."""
    t0 = time.time()
    cfg, res = dict(CFG, **(cfg or {})), RES if res is None else res
    if any(os.path.exists(os.path.join(res, f)) for f in FILES["sealed"]):
        print(f"stop: {FILES['sealed']} already exist (nothing is overwritten)")
        return 1
    D, read = load_b(("build", "sealed"), cfg["cal_start"])
    oecd_rows = G.series_rows(OECD_DEM, ("build",), G.CAND, read)         # rows only; used after the checks below
    bad = G.verify(read)
    if bad:
        print("stop: sha256 differs from the manifests:", bad)
        return 1
    prob, desc = plan(D, cfg["w_c"])
    if desc["months"] != len(G.months(*cfg["w_c"])):
        prob.append("the window is not fully covered")
    if prob:
        print("stop before any return is computed:", prob)
        return 1
    sims = simulate(D, (G.COST, PRE_COST) + tuple(cfg["pre_robust"]))
    ms, net, _, _ = series(D, sims, cfg["w_c"])
    dev, sealed = stored(dev_path, test_path)
    rep = {"dev": compare(ms, net, dev, cfg["reproduce"]["dev"]), "sealed": compare(ms, net, sealed, cfg["reproduce"]["sealed"])}
    if not (rep["dev"]["equal"] and rep["sealed"]["equal"]):
        print("stop: the months from 1990 on differ from the stored values of research 19:", rep)
        return 1
    S = {"written": None, "mode": "sealed", "window": [G.lab(cfg["w_c"][0]), G.lab(cfg["w_c"][1])], "cost_before_1990": PRE_COST,
         "cost_from_1990": G.COST, "boot": {"draws": G.BOOT_B, "block_months": G.BOOT_BLOCK, "seed": BOOT_SEED, "holm_alpha": ALPHA},
         "plan": desc, "reproduced": rep}
    S.update(evaluate(D, sims, cfg))
    S["dem_quality"] = dem_quality(D, oecd_rows)
    S["predictions"] = predictions(S)
    S["trials"] = {"before": TRIALS_BEFORE, "added": S["hypotheses"], "after": TRIALS_BEFORE + S["hypotheses"]}
    S["read"] = {"files": len(read), "paths": [os.path.relpath(p, G.DATA) for p in read]}
    S["engine19_sha256"] = G.sha(os.path.join(os.path.dirname(os.path.abspath(G.__file__)), "engine19.py"))
    S["written"], S["runtime_s"] = dt.datetime.now().isoformat(timespec="seconds"), round(time.time() - t0, 1)
    write(res, "sealed", S, render(S))
    return 0


def run_rehearsal(cfg=None, res=None, dev_path=None, test_path=None):
    """1990 on only: pass A on the build files (development months), pass B on build + sealed (sealed months)."""
    t0 = time.time()
    cfg, res = dict(CFG, **(cfg or {})), RES if res is None else res
    if any(os.path.exists(os.path.join(res, f)) for f in FILES["rehearsal"]):
        print(f"stop: {FILES['rehearsal']} already exist (nothing is overwritten)")
        return 1
    dev, sealed = stored(dev_path, test_path)
    out, n_read, none_before = {}, {}, True
    for name, parts, window, last in (("build", ("build",), cfg["reh_dev"], (cfg["reh_dev"][1],)), ("full", ("build", "sealed"), cfg["reh_full"], ())):
        D, read = load_b(parts, cfg["reh_start"])
        bad = G.verify(read)
        if bad:
            print("stop: sha256 differs from the manifests:", bad)
            return 1
        prob, _ = plan(D, window, last)
        if prob or D.days[0] < cfg["reh_start"]:
            print("stop before any return is computed:", prob)
            return 1
        none_before = none_before and G.portfolio(D, G.prv(window[0])) is None
        sims = simulate(D, (G.COST, PRE_COST))
        ms, net, _, _ = series(D, sims, window)
        out[name] = {"dev": compare(ms, net, dev, cfg["reh_dev"] if name == "build" else cfg["reproduce"]["dev"])}
        if name == "full":
            out[name]["sealed"] = compare(ms, net, sealed, cfg["reproduce"]["sealed"])
        n_read[name] = len(read)
    S = {"written": dt.datetime.now().isoformat(timespec="seconds"), "mode": "rehearsal", "calendar_start": cfg["reh_start"],
         "no_portfolio_before_1990": bool(none_before), "pass_build": out["build"], "pass_full": out["full"],
         "equal": bool(out["build"]["dev"]["equal"] and out["full"]["sealed"]["equal"] and out["full"]["dev"]["equal"] and none_before),
         "read": n_read, "engine19_sha256": G.sha(os.path.join(os.path.dirname(os.path.abspath(G.__file__)), "engine19.py")),
         "runtime_s": round(time.time() - t0, 1)}
    write(res, "rehearsal", S, render_rehearsal(S))
    return 0 if S["equal"] else 1


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--go", action="store_true")
    g.add_argument("--rehearsal", action="store_true")
    a = ap.parse_args()
    if a.rehearsal:
        return run_rehearsal()
    why = guard()
    if why:
        print("guard: the sealed test does not run:", why)
        return 1
    return run_sealed()


if __name__ == "__main__":
    sys.exit(main())
