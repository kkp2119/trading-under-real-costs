"""Research 19B: data coverage before 1990 (presence only). No return, no Sharpe, no rate ranking, no value is printed.

Reads the build files (rows before 2013) through run_test19b.load_b and looks only at which dates and months hold a
value: coverage of each currency 1979-01..1989-12, the first formation month with six usable currencies, the number
of usable currencies in every holding month, the uses of the one-month rate fill, the Fed DM/US$ file.
engine19.portfolio and engine19.simulate are not called. Writes results/check19b_quality.json and check19b_report.md.
"""
import datetime as dt
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine19 as G  # noqa: E402
import run_test19b as R  # noqa: E402

RES = os.path.join(HERE, "results")
SCAN = ((1971, 3), (1989, 12))                                             # holding months scanned for the start
OLD = ((1979, 1), (1989, 12))
END = (2026, 2)


def mde(n_months, alpha, power=0.80):
    z = statistics.NormalDist().inv_cdf
    return (z(1 - alpha) + z(power)) * math.sqrt(12 / n_months)


def runs(ms):
    """Month lists as ranges: ['1979-01..1979-04', '1981-07']."""
    out, start, prev = [], None, None
    for m in ms:
        if start is None or G.nxt(prev) != m:
            if start is not None:
                out.append(G.lab(start) if start == prev else f"{G.lab(start)}..{G.lab(prev)}")
            start = m
        prev = m
    if start is not None:
        out.append(G.lab(start) if start == prev else f"{G.lab(start)}..{G.lab(prev)}")
    return out


def main():
    D, read = R.load_b(("build",), "1971-01-01")
    bad = G.verify(read)
    if bad:
        print("stop: sha256 differs from the manifests:", bad)
        return 1
    n = {m: len(R.available(D, m)) for m in G.months(*SCAN)}
    six = [m for m in G.months(*SCAN) if n[m] >= 6]
    start_hold = six[0] if six else None
    start_form = G.prv(start_hold) if start_hold else None
    old = G.months(*OLD)
    per = {}
    for j, c in enumerate(G.CCY):
        fx_months = {m for i, m in enumerate(D.ym) if D.fresh[i, j]}
        rate = D.rate.get(c, {})
        usable = [m for m in G.months(*SCAN) if c in R.available(D, m)]
        k = [i for i, m in enumerate(D.ym) if OLD[0] <= m <= OLD[1]]
        per[c] = {"fx_months_missing": len([m for m in old if m not in fx_months]),
                  "rate_months_missing": len([m for m in old if m not in rate]), "rate_missing_runs": runs([m for m in old if m not in rate]),
                  "first_rate_month": G.lab(min(rate)) if rate else None,
                  "first_usable_holding_month": G.lab(usable[0]) if usable else None,
                  "usable_holding_months_1979_1989": len([m for m in usable if OLD[0] <= m <= OLD[1]]),
                  "days_without_a_fresh_quote_1979_1989": int(sum(not D.fresh[i, j] for i in k))}
    us = D.rate["USD"]
    window = (start_hold, OLD[1]) if start_hold else None
    hold = G.months(*window) if window else []
    by_year = {}
    for m in hold:
        by_year.setdefault(m[0], []).append(n[m])
    fills = [x for m in hold for x in R.filled(D, m)]
    days = [i for i, m in enumerate(D.ym) if OLD[0] <= m <= OLD[1]]
    dem = G.series_rows(R.DEM_OLD, ("build",), G.CAND)
    dem_days = {d for d, v in dem if v and "1979-01-01" <= d <= "1989-12-31"}
    cal = {D.days[i] for i in days}
    j_eur = G.CCY.index("EUR")
    others = {D.days[i] for i in days if any(D.fresh[i, j] for j in range(len(G.CCY)) if j != j_eur)}
    n_b = len(hold)
    n_c = len(G.months(start_hold, END)) if start_hold else 0
    q = {"written": dt.datetime.now().isoformat(timespec="seconds"), "files_read": len(read),
         "sealed_or_raw_files_read": sum(f"{os.sep}{x}{os.sep}" in p for p in read for x in ("sealed", "raw")),
         "start_formation_month": G.lab(start_form) if start_form else None, "start_holding_month": G.lab(start_hold) if start_hold else None,
         "start_in_1979_01_to_1980_12": bool(start_form and (1979, 1) <= start_form <= (1980, 12)),
         "currencies_before_the_start": {"max": max(n[m] for m in G.months(*SCAN) if start_hold is None or m < start_hold)},
         "holding_months_old_segment": n_b, "holding_months_full": n_c,
         "currencies_by_year": {str(y): (f"{min(v)}" if min(v) == max(v) else f"{min(v)}–{max(v)}") for y, v in sorted(by_year.items())},
         "months_by_count": {str(k): len([m for m in hold if n[m] == k]) for k in sorted(set(n[m] for m in hold))},
         "months_with_fewer_than_six": [G.lab(m) for m in hold if n[m] < 6],
         "universe": [c for c in G.CCY if per[c]["usable_holding_months_1979_1989"] > 0],
         "never_usable_before_1990": [c for c in G.CCY if per[c]["usable_holding_months_1979_1989"] == 0],
         "per_currency": per, "us_rate_months_missing_1979_1989": len([m for m in old if m not in us]),
         "rate_fills_used_old_segment": fills, "business_days_1979_1989": len(days),
         "days_with_a_stale_quote_1979_1989": int(sum(1 for i in days if any((not D.fresh[i, j]) and D.S[i, j] == D.S[i, j] for j in range(len(G.CCY))))),
         "incomplete_holding_months": [G.lab(m) for m in G.incomplete(D, window)] if window else None,
         "dem_file": {"first_day": dem[0][0], "last_day": dem[-1][0], "rows": len(dem), "days_with_value_1979_1989": len(dem_days),
                      "h10_days_other_currencies_1979_1989": len(others), "days_only_in_dem": len(dem_days - others),
                      "days_missing_in_dem": len(others - dem_days), "calendar_days_1979_1989": len(cal)},
         "power_formula": {"H1b": {"months": n_b, "sharpe_se": math.sqrt(12 / n_b) if n_b else None,
                                   "mde_alpha_0.0125": mde(n_b, 0.0125) if n_b else None, "mde_alpha_0.025": mde(n_b, 0.025) if n_b else None},
                           "H1c": {"months": n_c, "sharpe_se": math.sqrt(12 / n_c) if n_c else None,
                                   "mde_alpha_0.0125": mde(n_c, 0.0125) if n_c else None, "mde_alpha_0.025": mde(n_c, 0.025) if n_c else None}}}
    os.makedirs(RES, exist_ok=True)
    for name in ("check19b_quality.json", "check19b_report.md"):
        if os.path.exists(os.path.join(RES, name)):
            print(f"stop: results/{name} exists (nothing is overwritten)")
            return 1
    L = ["# 연구 19B · 1990년 전 자료 확인 (값 없음: 날짜와 결측만)", "",
         f"- 작성 {q['written']}. build 파일 {q['files_read']}개만 읽었다(sealed·raw {q['sealed_or_raw_files_read']}개). 수익, 샤프, 금리차 순위는 계산하지 않았다.",
         f"- **시작:** 통화가 6개 이상이 되는 첫 형성 월은 {q['start_formation_month']}, 첫 보유 월은 {q['start_holding_month']}이다. 그 전에는 최대 {q['currencies_before_the_start']['max']}개였다.",
         f"- 보유 월 수: 1990년 전 {n_b}개월, 전체({q['start_holding_month']} ~ {G.lab(END)}) {n_c}개월.", "",
         "## 통화별 (1979-01 ~ 1989-12, 132개월)", "",
         "| 통화 | 환율 없는 달 | 금리 없는 달 | 금리 첫 달 | 쓸 수 있는 첫 보유 월 | 쓸 수 있는 보유 월 수 | 환율 없는 영업일 |", "|---|---|---|---|---|---|---|"]
    for c in G.CCY:
        p = per[c]
        miss = f"{p['rate_months_missing']}" + (f" ({', '.join(p['rate_missing_runs'])})" if p["rate_missing_runs"] else "")
        L.append(f"| {c}{' (DEM)' if c == 'EUR' else ''} | {p['fx_months_missing']} | {miss} | {p['first_rate_month'] or '없음'} | "
                 f"{p['first_usable_holding_month'] or '없음'} | {p['usable_holding_months_1979_1989']} | {p['days_without_a_fresh_quote_1979_1989']} |")
    d = q["dem_file"]
    L += ["", f"- 미국 금리 없는 달: {q['us_rate_months_missing_1979_1989']}개. 금리 채움(A4) 사용: {', '.join(fills) or '없음'} ({len(fills)}번).",
          f"- H.10 영업일 {q['business_days_1979_1989']:,}일 중 직전 환율을 쓴 날: {q['days_with_a_stale_quote_1979_1989']}일.",
          f"- 청산일이 자료 밖인 보유 월: {', '.join(q['incomplete_holding_months']) or '없음'}.",
          "", "## 보유 월별 통화 수", "", "- 연도별: " + ", ".join(f"{y} {v}" for y, v in q["currencies_by_year"].items()),
          "- 개수별 달 수: " + ", ".join(f"{k}개 {v}개월" for k, v in q["months_by_count"].items()),
          f"- 6개 미만인 달: {', '.join(q['months_with_fewer_than_six']) or '없음'}",
          f"- 유니버스: {', '.join(q['universe'])}. 1990년 전에 한 번도 못 쓰는 통화: {', '.join(q['never_usable_before_1990']) or '없음'}.",
          "", "## 연준 DM/US$ 파일 (dat89_ge.txt)", "",
          f"- {d['first_day']} ~ {d['last_day']}, {d['rows']:,}행. 1979–1989에 값이 있는 날 {d['days_with_value_1979_1989']:,}일.",
          f"- 같은 기간 다른 통화의 H.10 날짜 {d['h10_days_other_currencies_1979_1989']:,}일과 비교: DEM에만 있는 날 {d['days_only_in_dem']}일, DEM에 없는 날 {d['days_missing_in_dem']}일.",
          "", "## 검정력 (공식만, 자료 값 없음)", "", "| 가설 | 개월 | 샤프 SE | MDE α 0.0125 | MDE α 0.025 |", "|---|---|---|---|---|"]
    for h, p in q["power_formula"].items():
        L.append(f"| {h} | {p['months']} | {p['sharpe_se']:.3f} | {p['mde_alpha_0.0125']:.3f} | {p['mde_alpha_0.025']:.3f} |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(RES, "check19b_quality.json"), "w", encoding="utf-8") as f:
        json.dump(q, f, indent=1, ensure_ascii=False)
    with open(os.path.join(RES, "check19b_report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
