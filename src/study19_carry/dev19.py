"""Research 19 step 2: development run (holding months 1990-01..2012-12, build files only).

Reads $DATA_DIR/research19_data/build and candidates/build (rows before 2013); never the sealed or raw folders.
Order: A1 path check -> sha256 of every file read against the manifests -> data checks -> 22 candidates at 4 bp and 8 bp
-> checks on the output -> A* by engine19.select_astar (ADDENDUM 4, applied as coded) -> H2 power.
Writes results/dev19_summary.json, dev19_report.md, dev19_monthly.csv. A failed check writes
results/dev19_checks_FAILED.json and stops before any summary.
"""
import datetime as dt
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine19 as G  # noqa: E402

RES = os.path.join(HERE, "results")
OUT = {k: os.path.join(RES, f) for k, f in (("summary", "dev19_summary.json"), ("report", "dev19_report.md"), ("monthly", "dev19_monthly.csv"))}
TRIALS_BEFORE, TRIALS_ADDED = 751, 22
SEALED_N = len(G.months(*G.SEALED))                                        # 158
CRISIS = [(1992, 9), (1998, 8), (1998, 9), (1998, 10), (2008, 9), (2008, 10), (2008, 11), (2008, 12)]
CRISIS_COLS = ["C0", "V80-M", "V80-D", "DV20-M", "DV20-D", "CS20-M", "CS20-D"]   # fixed before the run (DEVIATIONS19 1-17)
KEYS = ("sharpe", "ann_return", "ann_vol", "max_drawdown", "max_drawdown_daily", "worst_12m", "gross_sum", "cost_sum", "net_sum",
        "cost_share", "exits_per_year", "mid_exits_per_year", "mid_entries_per_year", "entries_per_year", "flat_share", "flat_months",
        "half1_sum", "half2_sum", "sharpe_cost2", "months")


def direct_monthly(D, m):
    """Monthly formula of the equivalence test, straight from the data: spot return of each leg + differential x days / 365."""
    p = G.portfolio(D, m)
    i0, i1 = D.first[m], D.first.get(G.nxt(m), len(D.days) - 1)
    idx, w, dr = p
    return float(w @ (D.S[i1, idx] / D.S[i0, idx] - 1.0)) + float(w @ dr) / 100.0 * (D.date[i1] - D.date[i0]).days / 365.0


def data_checks(D, read):
    ports = G.portfolios(D)
    dev = G.months(*G.DEV)
    n, avail, fills = {}, {}, []
    for m in dev:
        t = G.prv(m)
        i0, i1 = D.first[m], D.first.get(G.nxt(m), len(D.days) - 1)
        us_fill = t not in D.rate["USD"]
        ok = []
        for j, c in enumerate(G.CCY):
            r = G.rate_at(D.rate.get(c, {}), t)
            if r is not None and D.fresh[i0, j] and D.fresh[i1, j]:
                ok.append(c)
                if t not in D.rate[c]:
                    fills.append(f"{c} {G.lab(t)}")
        if us_fill:
            fills.append(f"USD {G.lab(t)}")
        avail[m] = ok
        n[m] = len(ok)
    by = {}
    for m in dev:
        by.setdefault(n[m], []).append(m)
    first = {c: G.lab(min(m for m in dev if c in avail[m])) for c in G.CCY if any(c in avail[m] for m in dev)}
    legs = {c: {"long": 0, "short": 0} for c in G.CCY}
    held = [m for m in dev if ports[m] is not None]
    for m in held:
        idx, w, _ = ports[m]
        for j, x in zip(idx, w):
            legs[G.CCY[j]]["long" if x > 0 else "short"] += 1
    partial = int((D.fresh.any(axis=1) & ~np.where(np.isnan(D.S), True, D.fresh).all(axis=1)).sum())
    return {"files_read": len(read), "calendar": {"first_day": D.days[0], "last_day": D.days[-1], "business_days": len(D.days),
                                                  "days_with_a_stale_quote": partial,
                                                  "filters_active_from": D.days[D.last[G.ACTIVE_FROM]] if G.ACTIVE_FROM in D.last else None},
            "currencies_per_holding_month": {str(k): f"{len(v)} ({G.lab(v[0])}..{G.lab(v[-1])})" for k, v in sorted(by.items())},
            "first_holding_month": first, "legs": legs, "rate_fills_used": fills,
            "incomplete_months": [G.lab(m) for m in G.incomplete(D, G.DEV)], "months_without_a_portfolio": [G.lab(m) for m in dev if ports[m] is None],
            "portfolios": {G.lab(m): {"long": [G.CCY[j] for j, x in zip(ports[m][0], ports[m][1]) if x > 0],
                                      "short": [G.CCY[j] for j, x in zip(ports[m][0], ports[m][1]) if x < 0]} for m in held}}


def output_checks(D, sims, tab, read):
    """[(name, ok, detail)] on the output; any False stops the run."""
    dev = G.months(*G.DEV)
    out = []
    ms, g0, c0 = G.monthly(D, sims["C0"], G.DEV)
    out.append(("no sealed or raw path read", not any(f"{os.sep}{x}{os.sep}" in p for p in read for x in ("sealed", "raw")), len(read)))
    out.append(("calendar ends 2012-12-31", D.days[-1] == "2012-12-31", D.days[-1]))
    out.append(("incomplete months = [2012-12]", G.incomplete(D, G.DEV) == [(2012, 12)], [G.lab(m) for m in G.incomplete(D, G.DEV)]))
    out.append(("C0 has 276 months and a portfolio in each", ms == dev and tab["C0"]["months"] == 276
                and all(G.portfolio(D, m) is not None for m in dev), len(ms)))
    out.append(("C0 pays 8 bp of capital every month", bool(np.all(np.abs(c0 - 0.0008) < 1e-15)), float(np.abs(c0 - 0.0008).max())))
    out.append(("C0 is never flat", tab["C0"]["flat_share"] == 0 and tab["C0"]["exits_per_year"] == 0, tab["C0"]["flat_share"]))
    worst = max(abs(g0[k] - direct_monthly(D, m)) for k, m in enumerate(ms))
    out.append(("C0 monthly gross = monthly formula (equivalence on the real data)", worst < 1e-12, worst))
    pre = [k for k, m in enumerate(ms) if m <= G.ACTIVE_FROM]
    same = all(np.array_equal(np.array(tab[c]["net"])[pre], np.array(tab["C0"]["net"])[pre]) for c in G.CANDS)
    out.append(("every candidate = C0 through 1993-01", same, len(pre)))
    bad = []
    for c in G.CANDS[1:]:
        f, q = G.kind(c)
        _, g, cc = G.monthly(D, sims[c], G.DEV)
        if q == "M" and not all((g[k] == g0[k] and cc[k] == c0[k]) or (g[k] == 0 and cc[k] == 0) for k in range(len(ms))):
            bad.append(c)
        if q == "D" and not all(sims[c]["held"][D.first[m]] == sims[f"{f}-M"]["held"][D.first[m]] for m in ms):
            bad.append(c)
        if not (np.isfinite(sims[c]["pnl"]).all() and np.isfinite(sims[c]["cst"]).all()):
            bad.append(c + " not finite")
    out.append(("M months are C0 months or empty; D agrees with M on every month's first day", not bad, bad))
    out.append(("net sum = gross sum - cost sum", all(abs(tab[c]["net_sum"] - (tab[c]["gross_sum"] - tab[c]["cost_sum"])) < 1e-12 for c in G.CANDS), ""))
    return out


def power(tab, astar):
    z = {"h1_formula": {"months": SEALED_N, "sharpe_se": math.sqrt(12 / SEALED_N),
                        "mde_alpha_0.05": G.mde(math.sqrt(12 / SEALED_N), 0.05), "mde_alpha_0.025": G.mde(math.sqrt(12 / SEALED_N), 0.025)}}
    idx = G.boot_index(len(tab["C0"]["net"]))
    c0 = G.boot_sharpe(tab["C0"]["net"], idx)
    z["h1_dev_boot_se"] = float(c0.std(ddof=1))
    z["h1_dev_boot_se_scaled"] = float(c0.std(ddof=1) * math.sqrt(len(tab["C0"]["net"]) / SEALED_N))
    if astar != "C0":
        d = G.boot_sharpe(tab[astar]["net"], idx) - c0
        se = float(d.std(ddof=1))
        s158 = se * math.sqrt(len(tab["C0"]["net"]) / SEALED_N)
        dev = tab[astar]["sharpe"] - tab["C0"]["sharpe"]
        z["h2"] = {"dev_difference": dev, "boot_se_dev": se, "se_scaled": s158, "mde_alpha_0.025": G.mde(s158, 0.025),
                   "mde_alpha_0.05": G.mde(s158, 0.05), "underpowered_0.025": bool(dev < G.mde(s158, 0.025)),
                   "underpowered_0.05": bool(dev < G.mde(s158, 0.05)),
                   "corr_monthly_net": G.corr(tab[astar]["net"], tab["C0"]["net"])}
    return z


def pc(x, d=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.{d}f}"


def share(x, d=0):
    return "—" if x is None else f"{100 * x:.{d}f}%"


def report(S):
    t, sel, dc = S["candidates"], S["selection"], S["data"]
    L = ["# 연구 19 · 개발 실행 (보유 월 1990-01 ~ 2012-12, 276개월)", "",
         f"- 작성 {S['written']}. build 파일만 읽었다(파일 {dc['files_read']}개, sha256 모두 일치). 봉인·원본 폴더는 읽지 않았다.",
         f"- A1 경로: {S['dem_path']} (연준 H.10 과거 자료의 일별 DM/US$). 필터는 {dc['calendar']['filters_active_from']} 확인부터 작동한다.",
         "- 수익은 자본 1달러 기준 %, 복리 없음. 비용은 진입·중간 정리마다 총 노셔널의 4bp(자본의 8bp)다.",
         f"- 마지막 달 2012-12는 2012-12-31 환율로 평가했다(청산일 2013-01-02는 봉인 파일, DEVIATIONS19 1-11).",
         "", f"## A* = {S['astar']}", ""]
    if S["astar"] == "C0":
        L.append(f"- 이유: {sel['reason_ko']}")
    else:
        L += [f"- 최고 후보 {sel['best']} 샤프 {sel['best_sharpe']:.3f}, C0 {sel['c0_sharpe']:.3f} (차이 {sel['best_minus_c0']:+.3f})",
              f"- 동률 묶음(0.05 안): {', '.join(sel['tie'])}", f"- 고른 이유: {sel['reason_ko']}"]
    L.append(f"- 절반 조건을 통과하지 못한 후보: {', '.join(sel['failed_halves']) or '없음'}")
    L += ["", "## 후보 22개 (비용 4bp 차감 후)", "",
          "| 후보 | 샤프 | 연수익 | 연변동성 | 최대 낙폭 | 최악 12개월 | 비용 비율 | 정리/년 | 비운 날 | 앞 절반 합 | 뒤 절반 합 | 8bp 샤프 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in G.CANDS:
        r = t[c]
        L.append(f"| {c} | {r['sharpe']:.3f} | {pc(r['ann_return'])} | {100 * r['ann_vol']:.2f} | {100 * r['max_drawdown']:.1f} | {pc(r['worst_12m'], 1)} | "
                 f"{share(r['cost_share'])} | {r['exits_per_year']:.2f} | {share(r['flat_share'], 1)} | "
                 f"{pc(r['half1_sum'], 1)} | {pc(r['half2_sum'], 1)} | {r['sharpe_cost2']:.3f} |")
    L += ["", "## 빈도 비교 (필터 7개 × M·W·D)", "",
          "| 필터 | 샤프 M | 샤프 W | 샤프 D | 정리/년 M | W | D | 비용 비율 M | W | D | 비운 날 M | W | D |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in G.FILTERS:
        r = [t[f"{f}-{q}"] for q in G.FREQS]
        L.append(f"| {f} | " + " | ".join(f"{x['sharpe']:.3f}" for x in r) + " | " + " | ".join(f"{x['exits_per_year']:.2f}" for x in r) + " | "
                 + " | ".join(share(x["cost_share"]) for x in r) + " | " + " | ".join(share(x["flat_share"], 1) for x in r) + " |")
    L += ["", f"- C0: 샤프 {t['C0']['sharpe']:.3f}, 비용 비율 {share(t['C0']['cost_share'])}.",
          "", "## 위기 달의 월 수익 (%)", "", "| 달 | " + " | ".join(S["crisis"]["columns"]) + " |", "|---|" + "---|" * len(S["crisis"]["columns"])]
    for m, row in S["crisis"]["rows"].items():
        L.append(f"| {m} | " + " | ".join(pc(x) for x in row) + " |")
    p = S["power"]
    L += ["", "## 검정력", "", "| 가설 | 개발 구간 값 | SE (158개월) | MDE α 0.025 | MDE α 0.05 | 표시 |", "|---|---|---|---|---|---|"]
    h1 = p["h1_formula"]
    flag = "검정력 부족" if t["C0"]["sharpe"] < h1["mde_alpha_0.05"] else "개발 샤프는 MDE 이상 (SPEC19 5절은 미리 '검정력 부족'으로 표시)"
    L.append(f"| H1 샤프(C0) > 0 | {t['C0']['sharpe']:.3f} | {h1['sharpe_se']:.3f} (공식) | {h1['mde_alpha_0.025']:.3f} | {h1['mde_alpha_0.05']:.3f} | {flag} |")
    if "h2" in p:
        h2 = p["h2"]
        L.append(f"| H2 샤프(A*) − 샤프(C0) > 0 | {h2['dev_difference']:+.3f} | {h2['se_scaled']:.3f} (부트스트랩) | {h2['mde_alpha_0.025']:.3f} | "
                 f"{h2['mde_alpha_0.05']:.3f} | {'검정력 부족' if h2['underpowered_0.05'] else '충분'} |")
    else:
        L.append("| H2 | 없음 (A* = C0) | | | | |")
    L += ["", f"- H1의 부트스트랩 SE(개발 276개월을 158개월로 환산): {p['h1_dev_boot_se_scaled']:.3f}. 공식 값 {h1['sharpe_se']:.3f}과 비교용이다.",
          "", "## 자료 점검", "",
          f"- 달력: {dc['calendar']['first_day']} ~ {dc['calendar']['last_day']}, 영업일 {dc['calendar']['business_days']:,}일, 직전 값을 쓴 날 {dc['calendar']['days_with_a_stale_quote']}일",
          "- 보유 월별 통화 수: " + "; ".join(f"{k}개 {v}" for k, v in dc["currencies_per_holding_month"].items()),
          "- 통화별 첫 보유 월: " + ", ".join(f"{c} {v}" for c, v in dc["first_holding_month"].items()),
          "- 통화별 롱/숏 달 수: " + ", ".join(f"{c} {v['long']}/{v['short']}" for c, v in dc["legs"].items()),
          f"- 금리 채움(A4) 사용: {', '.join(dc['rate_fills_used']) or '없음'}",
          f"- 청산일이 자료 밖인 달: {', '.join(dc['incomplete_months'])}",
          "", "## 출력 점검", ""]
    L += [f"- {'통과' if ok else '실패'}: {name}" for name, ok, _ in S["checks"]]
    L += ["", f"- 시행 수: {S['trials']['before']} → {S['trials']['after']} (후보 22개)."]
    return "\n".join(L) + "\n"


REASON_KO = {"no filter candidate has a positive net sum in both halves": "두 절반 모두 수익이 양수인 필터 후보가 없다",
             "the best filter candidate is not 0.05 above C0": "가장 좋은 필터 후보가 C0보다 0.05 이상 높지 않다 (개발에서 개선 없음)",
             "lowest frequency, then smallest flat share, among the candidates within 0.05 of the best": "최고 샤프의 0.05 안에서 빈도가 가장 낮고, 그다음 비운 날이 가장 적은 후보"}


def main():
    for p in OUT.values():
        if os.path.exists(p):
            print(f"stop: {p} exists (nothing is overwritten)")
            return 1
    with open(os.path.join(RES, "a1_check19.json"), encoding="utf-8") as f:
        a1 = json.load(f)
    if a1["dem_path"] != G.DEM_PATH:
        print("stop: engine19.DEM_PATH differs from results/a1_check19.json")
        return 1
    D, read = G.load(("build",))
    bad = G.verify(read)
    if bad:
        print("stop: sha256 differs from the manifests:", bad)
        return 1
    dc = data_checks(D, read)
    sims, sims2 = G.run_all(D), G.run_all(D, 2 * G.COST)
    tab = G.table(D, sims, G.DEV, sims2)
    checks = output_checks(D, sims, tab, read)
    if not all(ok for _, ok, _ in checks):
        with open(os.path.join(RES, "dev19_checks_FAILED.json"), "w", encoding="utf-8") as f:
            json.dump([[n, bool(ok), str(d)] for n, ok, d in checks], f, indent=1, ensure_ascii=False)
        print("stop: output check failed:", [n for n, ok, _ in checks if not ok])
        return 1
    astar, rec = G.select_astar(tab)
    rec["reason_ko"] = REASON_KO[rec["reason"]]
    ms = G.months(*G.DEV)
    cols = CRISIS_COLS + ([astar] if astar not in CRISIS_COLS else [])
    dev_sha = G.sha(os.path.join(HERE, "DEVIATIONS19.md"))
    S = {"written": dt.datetime.now().isoformat(timespec="seconds"), "window": [G.lab(G.DEV[0]), G.lab(G.DEV[1])],
         "dem_path": G.DEM_PATH, "cost": G.COST, "astar": astar, "selection": rec,
         "candidates": {c: {k: tab[c][k] for k in KEYS} for c in G.CANDS},
         "crisis": {"columns": cols, "rows": {G.lab(m): [tab[c]["net"][ms.index(m)] for c in cols] for m in CRISIS}},
         "power": power(tab, astar), "data": dc, "checks": [[n, bool(ok), str(d)] for n, ok, d in checks],
         "trials": {"before": TRIALS_BEFORE, "added": TRIALS_ADDED, "after": TRIALS_BEFORE + TRIALS_ADDED},
         "files_read": [os.path.relpath(p, G.DATA) for p in read], "deviations19_sha256_at_run": dev_sha,
         "engine19_sha256": G.sha(os.path.join(HERE, "engine19.py"))}
    text = report(S)                                                       # everything is built before anything is written
    csv = "month," + ",".join(G.CANDS) + "\n" + "".join(
        G.lab(m) + "," + ",".join(repr(tab[c]["net"][k]) for c in G.CANDS) + "\n" for k, m in enumerate(ms))
    js = json.dumps(S, indent=1, ensure_ascii=False)
    for key, body in (("monthly", csv), ("summary", js), ("report", text)):
        with open(OUT[key], "w", encoding="utf-8") as f:
            f.write(body)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
