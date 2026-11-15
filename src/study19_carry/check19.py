"""Research 19 step 1: quality check of the split FRED files (no strategy number is computed).

build/ (rows before 2013-01-01) is read with values: first date, missing months, month-end fallback, currencies per month,
the DEM -> EUR join and the quote direction. sealed/ is read as (date, value present?) only, never as numbers: rows,
first and last date, missing months, and the last complete month with every series. raw/ is not read.
Writes results/check19_quality.json and results/check19_report.md.
"""
import calendar
import datetime as dt
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch19 import DATA, FX, MACRO, RATES  # noqa: E402

CAND = os.path.join(DATA, "candidates")
RES = os.path.join(HERE, "results")
TODAY = dt.date.today()
LAST_COMPLETE = (TODAY.year, TODAY.month - 1) if TODAY.month > 1 else (TODAY.year - 1, 12)
DEV = ((1990, 1), (2012, 12))
SEAL0 = (2013, 1)
JOIN = ((1998, 12), (1999, 1))
DAILY = set(FX) | set(MACRO)
# currency -> (FX series, quote: True if USD per unit of foreign currency, rate country)
CCY = {"AUD": ("DEXUSAL", True, "AU"), "CAD": ("DEXCAUS", False, "CA"), "CHF": ("DEXSZUS", False, "CH"),
       "EUR": ("DEXUSEU", True, "EZ"), "GBP": ("DEXUSUK", True, "GB"), "JPY": ("DEXJPUS", False, "JP"),
       "NZD": ("DEXUSNZ", True, "NZ"), "NOK": ("DEXNOUS", False, "NO"), "SEK": ("DEXSDUS", False, "SE")}
DEM = ("DEXGEUS", False, "DE")
RATE = {c: f"IR3TIB01{c}M156N" for c in ("AU", "CA", "CH", "DE", "EZ", "GB", "JP", "NZ", "NO", "SE", "US")}
# candidates found on FRED (SPEC19_QUESTIONS.md); used here only to show what each would cover
# DEM quotes are read off the 1990-1998 levels and the 1999-2012 match with DEXUSEU (check19_quality.json dem_check);
# "EUR units" = DEM / 1.95583, so returns equal DEM returns
CANDS = {"DEXGEUS": [("CCUSSP01DEM650N", "OECD spot, end of month, USD per DEM in EUR units"),
                     ("EXGEUS", "H.10 monthly average, DEM per USD"),
                     ("CCUSMA02DEM618N", "OECD monthly average, DEM in EUR units per USD")],
         "IR3TIB01JPM156N": [("IR3TCD01JPM156N", "OECD 3-month CD rate, Japan"),
                             ("IRSTCI01JPM156N", "OECD call money / interbank immediate, Japan")],
         "IR3TIB01CHM156N": [("IRSTCI01CHM156N", "OECD call money / interbank immediate, Switzerland")]}


def months(a, b):
    out, (y, m) = [], a
    while (y, m) <= b:
        out.append((y, m))
        y, m = (y, m + 1) if m < 12 else (y + 1, 1)
    return out


def ym(s):
    return int(s[:4]), int(s[5:7])


def fmt(k):
    return f"{k[0]}-{k[1]:02d}" if k else ""


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\n").split(",", 1) for ln in f.readlines()[1:] if ln.strip()]


def present(v):
    return v.strip() not in ("", ".")


def build(series, root=DATA):
    """[(date, float or None)] before 2013-01-01."""
    out = [(d, float(v) if present(v) else None) for d, v in rows(os.path.join(root, "build", f"{series}.csv"))]
    assert all(d < "2013-01-01" for d, _ in out)
    return out


def sealed(series, root=DATA):
    """[(date, value present?)] from 2013-01-01; the values themselves are never converted or kept."""
    out = [(d, present(v)) for d, v in rows(os.path.join(root, "sealed", f"{series}.csv"))]
    assert all(d >= "2013-01-01" for d, _ in out)
    return out


def last_weekday(y, m):
    d = dt.date(y, m, calendar.monthrange(y, m)[1])
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d.isoformat()


def month_end(obs):
    """{(y, m): (date, value)} with the last present observation of each month (the month-end rule of SPEC19 2)."""
    out = {}
    for d, v in obs:
        if v is not None:
            out[ym(d)] = (d, v)
    return out


def gaps(keys, have):
    miss = [k for k in keys if k not in have]
    runs, start, prev = [], None, None
    for k in miss:
        if start is None or months(prev, k)[1:] != [k]:
            if start is not None:
                runs.append(f"{fmt(start)}..{fmt(prev)}" if start != prev else fmt(start))
            start = k
        prev = k
    if start is not None:
        runs.append(f"{fmt(start)}..{fmt(prev)}" if start != prev else fmt(start))
    return len(miss), runs


def need(series):
    """Months of build data the plan needs: formation from 1989-12 (first return month 1990-01) through 2012-12."""
    lo, hi = (1989, 12), DEV[1]
    if series in ("DEXUSEU", RATE["EZ"]):
        lo = (1999, 1)
    if series == RATE["DE"]:
        hi = (1998, 12)
    if series == "DEM":
        hi = (1999, 1)                                     # the 1999-01 return of a 1998-12 position is in DEM
    if series == "VIXCLS":
        lo = DEV[0]                                        # VIXCLS starts 1990-01-02
    return months(lo, hi)


def dev_row(series, root=DATA, alias=None):
    obs = build(series, root)
    me = month_end(obs)
    keys = need(alias or series)
    n, runs = gaps(keys, me)
    r = {"first_date": obs[0][0] if obs else "", "first_value_date": next((d for d, v in obs if v is not None), ""),
         "last_date": obs[-1][0] if obs else "", "need": f"{fmt(keys[0])}..{fmt(keys[-1])}", "missing_months": n,
         "missing_runs": runs, "empty_rows": sum(v is None for _, v in obs)}
    if series in DAILY:
        r["month_end_fallback"] = sum(1 for k in keys if k in me and me[k][0] < last_weekday(*k))
    return r, me


def seal_row(series, root=DATA):
    obs = sealed(series, root)
    have = {ym(d) for d, p in obs if p}
    last = max(have) if have else None
    keys = months(SEAL0, LAST_COMPLETE)
    inner = [k for k in keys if last and k <= last]
    n_in, runs = gaps(inner, have)
    return {"rows": len(obs), "first_date": obs[0][0] if obs else "", "last_date": obs[-1][0] if obs else "",
            "last_value_month": fmt(last), "empty_rows": sum(not p for _, p in obs), "missing_months_inside": n_in,
            "missing_runs_inside": runs, "months_after_last_value": len([k for k in keys if not last or k > last])}, have


def counts(fx_me, rate_me, dem, extra_rate=None):
    """Formation months t = 1989-12..2012-11 (returns 1990-01..2012-12): currencies with FX at t and t + 1, own rate
    at t and the US rate at t."""
    extra_rate = extra_rate or {}
    out = {}
    for t in months((1989, 12), (2012, 11)):
        t1 = months(t, (2013, 1))[1]
        n = []
        for c, (s, _, cc) in CCY.items():
            fxs, rc = (s, cc)
            if c == "EUR" and t <= JOIN[0]:
                fxs, rc = "DEM", "DE"
            fx = fx_me.get(fxs) if fxs != "DEM" else dem
            r = rate_me[RATE[rc]]
            r_ok = t in r or t in extra_rate.get(rc, {})
            if fx is not None and t in fx and t1 in fx and r_ok and t in rate_me[RATE["US"]]:
                n.append(c)
        out[t] = n
    return out


def summarize(cnt):
    lt6 = [t for t, n in cnt.items() if len(n) < 6]
    by = {}
    for t, n in cnt.items():
        by.setdefault(len(n), []).append(t)
    first = {c: fmt(min((t for t, n in cnt.items() if c in n), default=None)) for c in CCY}
    return {"min": min(len(n) for n in cnt.values()), "months_below_6": len(lt6), "below_6_runs": gaps(lt6, set())[1],
            "months_by_count": {k: f"{len(v)} ({fmt(v[0])}..{fmt(v[-1])})" for k, v in sorted(by.items())},
            "first_formation_month": first}


def mde(n_months):
    z = statistics.NormalDist().inv_cdf
    se = (12 / n_months) ** 0.5
    return {"months": n_months, "years": round(n_months / 12, 2), "sharpe_se": round(se, 4),
            "mde_alpha_0.025": round((z(0.975) + z(0.80)) * se, 3), "mde_alpha_0.05": round((z(0.95) + z(0.80)) * se, 3)}


def main():
    q = {"written": dt.datetime.now().isoformat(timespec="seconds"), "today": TODAY.isoformat(),
         "last_complete_month": fmt(LAST_COMPLETE), "dev": {}, "sealed": {}, "candidates": {}}
    main_series = [s for s in FX + RATES + MACRO if os.path.isfile(os.path.join(DATA, "build", f"{s}.csv"))]
    q["not_downloaded"] = [s for s in FX + RATES + MACRO if s not in main_series]
    me_all, seal_have = {}, {}
    for s in main_series:
        q["dev"][s], me_all[s] = dev_row(s)
        q["sealed"][s], seal_have[s] = seal_row(s)
    # quote direction from development levels (1990-2012 medians of month-end values)
    q["direction"] = {}
    for c, (s, usd_per, _) in CCY.items():
        v = [x for k, (_, x) in me_all[s].items() if DEV[0] <= k <= DEV[1]]
        q["direction"][s] = {"ccy": c, "quote": "USD per " + c if usd_per else c + " per USD",
                             "dev_median_level": round(statistics.median(v), 4)}
    # candidates: development coverage for the gap they would fill, sealed dates only
    cand_me = {}
    for orig, lst in CANDS.items():
        for s, what in lst:
            if not os.path.isfile(os.path.join(CAND, "build", f"{s}.csv")):
                continue
            d, cand_me[s] = dev_row(s, CAND, alias="DEM" if orig == "DEXGEUS" else orig)
            sr, _ = seal_row(s, CAND)
            q["candidates"][s] = {"for": orig, "what": what, "dev": d, "sealed": sr}
            if orig == "DEXGEUS":
                v = [x for k, (_, x) in cand_me[s].items() if DEV[0] <= k <= (1998, 12)]
                q["candidates"][s]["dev_median_level_1990_1998"] = round(statistics.median(v), 4)
                q["candidates"][s]["quote"] = what.split(", ")[-1]
    eu, sp = me_all["DEXUSEU"], cand_me.get("CCUSSP01DEM650N", {})
    ks = [k for k in eu if k in sp]
    rel = sorted(abs(sp[k][1] - eu[k][1]) / eu[k][1] for k in ks)
    q["dem_check"] = {"CCUSSP01DEM650N_vs_DEXUSEU_month_end_1999_2012": {"months": len(ks), "median_rel_diff": round(rel[len(rel) // 2], 5),
                                                                         "max_rel_diff": round(rel[-1], 5)} if ks else None}
    # DEM -> EUR join months
    q["join"] = {fmt(k): {s: (k in me_all.get(s, {}) if s in me_all else k in cand_me.get(s, {}))
                          for s in ("CCUSSP01DEM650N", "EXGEUS", "DEXUSEU", RATE["DE"], RATE["EZ"])} for k in JOIN}
    fx_me = {s: me_all.get(s) for s, _, _ in CCY.values()}
    rate_me = {s: me_all[s] for s in RATE.values()}
    q["currencies"] = {
        "A_main_only": summarize(counts(fx_me, rate_me, None)),
        "B_main_plus_DEM_CCUSSP01": summarize(counts(fx_me, rate_me, cand_me.get("CCUSSP01DEM650N"))),
        "C_B_plus_JP_CD_and_CH_call_before_start": summarize(counts(
            fx_me, rate_me, cand_me.get("CCUSSP01DEM650N"),
            {"JP": cand_me.get("IR3TCD01JPM156N", {}), "CH": cand_me.get("IRSTCI01CHM156N", {})}))}
    # sealed end: last complete month in which every series the sealed test needs has a value
    req = [CCY[c][0] for c in CCY] + [RATE[CCY[c][2]] for c in CCY] + [RATE["US"]] + MACRO
    ok = [k for k in months(SEAL0, LAST_COMPLETE) if all(k in seal_have[s] for s in req)]
    end = max(ok) if ok else None
    contiguous = all(k in ok for k in months(SEAL0, end)) if end else False
    nxt = months(end, LAST_COMPLETE)[1:2] if end else []
    binding = sorted(s for s in req if nxt and nxt[0] not in seal_have[s])
    ok2 = [k for k in months(SEAL0, LAST_COMPLETE) if all(k in seal_have[s] for s in req if s not in binding)]
    later = {"without": binding, "end": fmt(max(ok2)) if ok2 else ""}
    n_seal = len(months(SEAL0, end)) if end else 0
    q["sealed_end"] = {"required_series": len(req), "end_month": fmt(end), "months": n_seal, "contiguous_from_2013_01": contiguous,
                       "binding_series": binding, "end_if_binding_series_were_missing_currencies": later}
    q["power_h1"] = {"sealed_end": mde(n_seal) if n_seal else None, "plan_166": mde(166)}
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "check19_quality.json"), "w", encoding="utf-8") as f:
        json.dump(q, f, indent=1, ensure_ascii=False)
    report(q)


def report(q):
    L = ["# 연구 19 · 1단계 품질 확인 (check19)", "",
         f"- 작성 {q['written']}, 가장 최근 완결된 달 {q['last_complete_month']}. 봉인 파일은 날짜와 값 유무만 읽었다.",
         f"- 받지 못한 시리즈: {', '.join(q['not_downloaded']) or '없음'}", "",
         "## 시리즈", "", "| 시리즈 | 표기 | 개발 첫 날짜 | 필요 구간 | 개발 결측 월 | 월말 대체 | 봉인 마지막 날짜 | 마지막 값 월 | 봉인 결측(안/뒤) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s, d in q["dev"].items():
        z = q["sealed"][s]
        quote = q["direction"].get(s, {}).get("quote", "% p.a., 월평균" if s.startswith("IR3") else ("지수" if s == "VIXCLS" else "%p"))
        miss = f"{d['missing_months']}" + (f" ({', '.join(d['missing_runs'])})" if d["missing_runs"] else "")
        L.append(f"| {s} | {quote} | {d['first_value_date']} | {d['need']} | {miss} | {d.get('month_end_fallback', '')} | "
                 f"{z['last_date']} | {z['last_value_month']} | {z['missing_months_inside']} / {z['months_after_last_value']} |")
    L += ["", "## 대체 후보 (적용하지 않음)", "", "| 후보 | 대상 | 내용 | 개발 첫 날짜 | 필요 구간 결측 | 봉인 마지막 값 월 |", "|---|---|---|---|---|---|"]
    for s, c in q["candidates"].items():
        L.append(f"| {s} | {c['for']} | {c['what']} | {c['dev']['first_value_date']} | {c['dev']['need']}: {c['dev']['missing_months']} | "
                 f"{c['sealed']['last_value_month'] or '없음'} |")
    L += ["", "## 표기 방향 (개발 1990–2012 월말 중앙값)", "", "| 시리즈 | 통화 | 표기 | 중앙값 |", "|---|---|---|---|"]
    for s, d in q["direction"].items():
        L.append(f"| {s} | {d['ccy']} | {d['quote']} | {d['dev_median_level']} |")
    for s, c in q["candidates"].items():
        if "dev_median_level_1990_1998" in c:
            L.append(f"| {s} (후보) | DEM | {c['quote']} | {c['dev_median_level_1990_1998']} (1990–1998) |")
    dc = q["dem_check"]["CCUSSP01DEM650N_vs_DEXUSEU_month_end_1999_2012"]
    if dc:
        L.append(f"\n- CCUSSP01DEM650N과 DEXUSEU 월말 값(1999–2012, {dc['months']}개월): 상대 차이 중앙값 {dc['median_rel_diff']}, 최대 {dc['max_rel_diff']}")
    L += ["", "## DEM → EUR 이음 (월말 값 유무)", "", "| 달 | " + " | ".join(next(iter(q["join"].values()))) + " |",
          "|---|" + "---|" * len(next(iter(q["join"].values())))]
    for k, v in q["join"].items():
        L.append(f"| {k} | " + " | ".join("있음" if x else "없음" for x in v.values()) + " |")
    L += ["", "## 형성 월별 통화 수 (1989-12..2012-11, FX t·t+1, 금리 t, 미국 금리 t)", "",
          "| 경우 | 최소 | 6개 미만 달 | 개수별 달 |", "|---|---|---|---|"]
    for k, v in q["currencies"].items():
        L.append(f"| {k} | {v['min']} | {v['months_below_6']} | {'; '.join(f'{n}개: {s}' for n, s in v['months_by_count'].items())} |")
    L += ["", "| 경우 | " + " | ".join(CCY) + " |", "|---|" + "---|" * len(CCY)]
    for k, v in q["currencies"].items():
        L.append(f"| {k} | " + " | ".join(v["first_formation_month"][c] for c in CCY) + " |")
    inside = {s: z["missing_runs_inside"] for s, z in q["sealed"].items() if z["missing_runs_inside"]}
    L += ["", "## 봉인 구간 안의 빈 달 (날짜만)", ""] + [f"- {s}: {', '.join(r)}" for s, r in inside.items()] + (["- 없음"] if not inside else [])
    L.append("- 일별 시리즈의 빈 행(휴일)은 빈 달을 만들지 않는다: " + ", ".join(f"{s} {z['empty_rows']}" for s, z in q["sealed"].items() if z["empty_rows"] and s in DAILY))
    e = q["sealed_end"]
    L += ["", "## 봉인 구간 끝", "",
          f"- 필요한 시리즈 {e['required_series']}개가 모두 있는 마지막 달: **{e['end_month']}** (2013-01부터 {e['months']}개월, 빈 달 없음: {e['contiguous_from_2013_01']})",
          f"- 끝을 정하는 시리즈: {', '.join(e['binding_series']) or '없음'}; 이 시리즈를 빠진 통화로 보면 끝은 {e['end_if_binding_series_were_missing_currencies']['end']}",
          "", "## H1 검정력 (공식만, 데이터 값 없음)", "", "| 개월 | 년 | 샤프 SE | MDE α 0.025 | MDE α 0.05 |", "|---|---|---|---|---|"]
    for k in ("sealed_end", "plan_166"):
        p = q["power_h1"][k]
        if p:
            L.append(f"| {p['months']} | {p['years']} | {p['sharpe_se']} | {p['mde_alpha_0.025']} | {p['mde_alpha_0.05']} |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(RES, "check19_report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
