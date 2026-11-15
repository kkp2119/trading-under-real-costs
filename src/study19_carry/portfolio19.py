"""Research 19: carry (A*) and the option seller of studies 13-14 as one risk-parity bundle (PORTFOLIO19.md).

Descriptive only: no verdict and no test count.

  python portfolio19.py --dev      # 1996-03..2012-12: fixes the weights; run once before the freeze
  python portfolio19.py --sealed   # 2013-01..2025-08: runs only after run_test19.py --go has written its results

Carry: monthly net return of A*, read from results/dev19_monthly.csv (development) or results/test19_summary.json
(sealed); the carry engine is not run here.
Option: setting O2 of studies 13-14 (16-delta strangle held 21 trading days, judged cost scenario 'base'): the saved
daily P&L on $1M notional (study13_14_option_seller/results/dev_daily.csv, study13_14_option_seller/study14/results14/holdout_daily14.csv) summed by
calendar month and divided by 1,000,000. No option engine and no holdout script is run.
Weights: inverse of the standard deviation of the 1996-03..2012-12 monthly returns, sum 1, fixed by --dev.
The sealed VIX (for the months with the largest VIX rise) is read in --sealed only, after the test results exist.
"""
import argparse
import csv
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
R13 = os.path.normpath(os.path.join(HERE, "..", "study13_14_option_seller"))
OPT = {"dev": os.path.join(R13, "results", "dev_daily.csv"), "sealed": os.path.join(R13, "study14", "results14", "holdout_daily14.csv")}
OPT_SHA256 = {"dev": "b6822a0a00428959c046f21726f29957260f49b7c885de3a7305ae9728b5c7f8",       # as recorded by study 14
              "sealed": "93843c7416d972b0c84daca5ed39501288041c29b82927f540863b4d25509de3"}    # hashed 2026-11-10, before the freeze
OPT_SETTING, OPT_COL, OPT_N0, OPT_START = "O2", "O2_base", 1_000_000.0, "19960306"   # statistics start of study 13
WINDOW = {"dev": ((1996, 3), (2012, 12)), "sealed": ((2013, 1), (2025, 8))}
CRISIS = {"dev": ((1998, 8), (1998, 9), (1998, 10), (2008, 9), (2008, 10), (2008, 11), (2008, 12)),
          "sealed": ((2015, 8), (2018, 2), (2020, 3), (2024, 8))}
JUMP_PCT = 90                                                              # months at or above this percentile of the VIX change
FILES = {"dev": ("portfolio19_dev.json", "portfolio19_dev_report.md"), "sealed": ("portfolio19_sealed.json", "portfolio19_sealed_report.md")}
TEST_FILES = ("test19_summary.json", "test19_report.md")                   # written by run_test19.py --go
SENTENCES = {"sharpe": {True: "두 전략을 묶으니 각각보다 나았다", False: "묶어도 더 나아지지 않았다"},
             "jump": {True: "위기 때는 두 전략이 함께 무너졌다. 둘 다 위기 보험을 파는 전략이기 때문이다",
                      False: "위기 때도 두 전략은 따로 움직였다"}}          # PORTFOLIO19 4, verbatim
PRED = (("봉인 구간에서 묶음 샤프가 두 칸 모두보다 높음", 0.35), ("VIX 급등 달 상관이 0.5 이상", 0.5))   # PORTFOLIO19 5, verbatim
TITLE = {"dev": "개발 구간 (1996-03 ~ 2012-12)", "sealed": "봉인 구간 (2013-01 ~ 2025-08)"}


# ---------------------------------------------------------------- the two monthly series
def option_monthly(path, window, col=OPT_COL, start=OPT_START, n0=OPT_N0):
    """(monthly return per dollar of capital, trading days in each month): calendar-month sums of the daily P&L / n0."""
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        head = next(r)
        if "date" not in head or col not in head:
            raise ValueError(f"{path}: no 'date' or '{col}' column")
        i, j = head.index("date"), head.index(col)
        s, n = {}, {}
        for row in r:
            d = row[i]
            if len(d) != 8 or not d.isdigit():
                raise ValueError(f"{path}: unexpected date {d!r}")
            if d < start:
                continue
            m = (int(d[:4]), int(d[4:6]))
            s[m] = s.get(m, 0.0) + float(row[j])
            n[m] = n.get(m, 0) + 1
    ms = G.months(*window)
    miss = [G.lab(m) for m in ms if m not in s]
    if miss:
        raise ValueError(f"{path}: no daily result in {miss[:4]}{'...' if len(miss) > 4 else ''}")
    return [s[m] / n0 for m in ms], [n[m] for m in ms]


def carry_from_csv(path, astar, window):
    """A* monthly net from results/dev19_monthly.csv."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    k = rows[0].index(astar)
    by = {r[0]: float(r[k]) for r in rows[1:]}
    return [by[G.lab(m)] for m in G.months(*window)]


def carry_from_test(path, astar, window):
    """A* monthly net from results/test19_summary.json (the sealed run stores the months and the 22 monthly series)."""
    with open(path, encoding="utf-8") as f:
        t = json.load(f)
    by = dict(zip(t["months"], t["monthly_net"][astar]))
    return [by[G.lab(m)] for m in G.months(*window)]


def inverse_vol(carry, option):
    sc, so = float(np.std(carry, ddof=1)), float(np.std(option, ddof=1))
    wc = (1.0 / sc) / (1.0 / sc + 1.0 / so)
    return {"carry": wc, "option": 1.0 - wc, "sigma_carry_monthly": sc, "sigma_option_monthly": so}


def metrics(x):
    x = np.asarray(x, dtype=float)
    return {"sharpe": G.sharpe(x), "ann_return": float(x.mean() * 12), "ann_vol": float(x.std(ddof=1) * math.sqrt(12)),
            "max_drawdown": G.max_drawdown(x), "worst_12m": G.worst12(x), "sum": float(x.sum()), "months": int(len(x))}


def evaluate(window, carry, option, w, dvix, crisis):
    """Sleeves, bundle, correlations (all months and the months with the largest VIX rise) and crisis months."""
    ms = G.months(*window)
    c, o = np.asarray(carry, dtype=float), np.asarray(option, dtype=float)
    p = w["carry"] * c + w["option"] * o
    dv = np.array([np.nan if v is None else v for v in dvix], dtype=float)
    thr = float(np.nanpercentile(dv, JUMP_PCT))
    top = [k for k in range(len(ms)) if dv[k] >= thr]
    m = {"carry": metrics(c), "option": metrics(o), "bundle": metrics(p)}
    return {"months": [G.lab(x) for x in ms], "carry": [float(x) for x in c], "option": [float(x) for x in o], "bundle": [float(x) for x in p],
            "metrics": m, "bundle_sharpe_above_both": bool(m["bundle"]["sharpe"] > max(m["carry"]["sharpe"], m["option"]["sharpe"])),
            "corr_all": G.corr(list(c), list(o)),
            "vix_jump": {"percentile": JUMP_PCT, "threshold_vix_points": thr, "months": [G.lab(ms[k]) for k in top], "n": len(top),
                         "months_without_vix_change": int(np.isnan(dv).sum()),
                         "corr": G.corr([float(c[k]) for k in top], [float(o[k]) for k in top]),
                         "carry_sum": float(c[top].sum()), "option_sum": float(o[top].sum()), "bundle_sum": float(p[top].sum())},
            "crisis": {G.lab(x): {"carry": float(c[ms.index(x)]), "option": float(o[ms.index(x)]), "bundle": float(p[ms.index(x)])}
                       for x in crisis if x in ms}}


# ---------------------------------------------------------------- report
def pc(x, d=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.{d}f}"


def num(x, d=3):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{d}f}"


def render(S):
    E, w = S["result"], S["weights"]
    L = [f"# 연구 19 · 전략 묶음: {TITLE[S['mode']]}", "",
         "- 서술용 분석이다. 판정이 아니고 검정 수에 넣지 않는다(PORTFOLIO19.md).",
         f"- 캐리 칸은 연구 19의 A* = {S['astar']}, 옵션 칸은 연구 13·14의 설정 {OPT_SETTING}(1개월 스트랭글, 판정 비용)이다. 둘 다 비용 차감 후 월 수익, 자본 1달러 기준, 복리 없음.",
         f"- 비중(개발 1996-03 ~ 2012-12의 변동성 역수, 고정): 캐리 {w['carry']:.4f}, 옵션 {w['option']:.4f}. "
         f"월 변동성은 캐리 {100 * w['sigma_carry_monthly']:.3f}%, 옵션 {100 * w['sigma_option_monthly']:.3f}%.",
         f"- {len(E['months'])}개월 ({E['months'][0]} ~ {E['months'][-1]}).", "",
         "## 칸별·묶음 결과", "", "| | 샤프 | 연수익 % | 변동성 % | 최대 낙폭 % | 최악 12개월 % |", "|---|---|---|---|---|---|"]
    for k, name in (("carry", f"캐리 ({S['astar']})"), ("option", f"옵션 ({OPT_SETTING})"), ("bundle", "묶음")):
        r = E["metrics"][k]
        L.append(f"| {name} | {r['sharpe']:.3f} | {pc(r['ann_return'])} | {100 * r['ann_vol']:.2f} | {100 * r['max_drawdown']:.1f} | {pc(r['worst_12m'], 1)} |")
    j = E["vix_jump"]
    L += ["", f"- 묶음 샤프가 두 칸 모두보다 높은가: {'예' if E['bundle_sharpe_above_both'] else '아니오'}",
          "", "## 두 칸의 월 수익 상관", "", "| 구간 | 달 수 | 상관 |", "|---|---|---|",
          f"| 전체 | {len(E['months'])} | {num(E['corr_all'], 2)} |",
          f"| VIX 급등 달 (월 변화가 {j['threshold_vix_points']:+.2f}포인트 이상, 상위 10%) | {j['n']} | {num(j['corr'], 2)} |", "",
          f"- VIX 급등 달의 수익 합: 캐리 {pc(j['carry_sum'], 1)}%, 옵션 {pc(j['option_sum'], 1)}%, 묶음 {pc(j['bundle_sum'], 1)}%",
          "", "## 위기 달의 월 수익 (%)" + (" (개발 구간, 참고)" if S["mode"] == "dev" else ""), "", "| 달 | 캐리 | 옵션 | 묶음 |", "|---|---|---|---|"]
    L += [f"| {m} | {pc(r['carry'])} | {pc(r['option'])} | {pc(r['bundle'])} |" for m, r in E["crisis"].items()]
    if S["mode"] == "sealed":
        L += ["", "## 미리 정한 문장 (PORTFOLIO19 4절)", ""] + [f"- \"{s}\"" for s in S["sentences"]]
        word = {True: "맞음", False: "틀림", None: "계산 불가"}
        L += ["", "## 예측 대조 (PORTFOLIO19 5절)", "", "| 항목 | 예측 확률 | 결과 |", "|---|---|---|"]
        L += [f"| {x['item']} | {x['probability']} | {word[x['outcome']]} |" for x in S["predictions"]]
        L += ["", "- Option Seller의 2013–2025 결과는 연구 14에서 이미 봤다. 이 묶음에서 처음 보는 부분은 캐리 쪽뿐이다."]
    d = S["inputs"]
    L += ["", "## 입력과 점검", "",
          f"- 옵션 일별 결과: {d['option_file']} (sha256 `{d['option_sha256'][:8]}…`), 열 {OPT_COL}, 100만 달러로 나눔. 달별 거래일 수 {d['option_days_min']} ~ {d['option_days_max']}일.",
          f"- 캐리 월 수익: {d['carry_file']}",
          f"- VIX: 읽은 파일 {d['data_files']}개 (sealed 폴더 {d['sealed_files']}개, raw 폴더 {d['raw_files']}개), sha256 모두 일치."]
    L += [f"- {'통과' if ok else '실패'}: {name}" for name, ok in S["checks"]]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- runs
def write(S, res, mode):
    out = [os.path.join(res, f) for f in FILES[mode]]
    text, js = render(S), json.dumps(S, indent=1, ensure_ascii=False)       # both built before anything is written
    for p, body in zip(out, (js, text)):
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)
    print(text)


def inputs(mode, opt_path, carry_path, read, days):
    return {"option_file": os.path.relpath(opt_path, os.path.dirname(HERE)), "option_sha256": G.sha(opt_path),
            "carry_file": os.path.relpath(carry_path, HERE), "option_days_min": int(min(days)), "option_days_max": int(max(days)),
            "data_files": len(read), "sealed_files": sum(f"{os.sep}sealed{os.sep}" in p for p in read),
            "raw_files": sum(f"{os.sep}raw{os.sep}" in p for p in read)}


def same_files(opt, opt_sha, which):
    """Names of the option result files whose sha256 is not the expected one."""
    return [k for k in which if G.sha(opt[k]) != opt_sha[k]]


def run_dev(res=None, opt=None, window=None, crisis=None, opt_sha=None):
    res, opt, opt_sha = RES if res is None else res, OPT if opt is None else opt, OPT_SHA256 if opt_sha is None else opt_sha
    window, crisis = WINDOW["dev"] if window is None else window, CRISIS["dev"] if crisis is None else crisis
    if any(os.path.exists(os.path.join(res, f)) for f in FILES["dev"]):
        print(f"stop: {FILES['dev']} already exist (nothing is overwritten)")
        return 1
    if same_files(opt, opt_sha, ("dev", "sealed")):
        print("stop: an option result file is not the saved file of studies 13-14:", same_files(opt, opt_sha, ("dev", "sealed")))
        return 1
    with open(os.path.join(res, "dev19_summary.json"), encoding="utf-8") as f:
        astar = json.load(f)["astar"]
    carry_path = os.path.join(res, "dev19_monthly.csv")
    try:
        carry = carry_from_csv(carry_path, astar, window)
        option, days = option_monthly(opt["dev"], window)
    except (ValueError, KeyError, OSError) as e:
        print("stop: the monthly returns cannot be made:", e)
        return 1
    D, read = G.load(("build",))
    bad = G.verify(read)
    if bad:
        print("stop: sha256 differs from the manifests:", bad)
        return 1
    w = inverse_vol(carry, option)
    E = evaluate(window, carry, option, w, G.vix_change(D, window), crisis)
    S = {"written": dt.datetime.now().isoformat(timespec="seconds"), "mode": "dev", "window": [G.lab(window[0]), G.lab(window[1])],
         "astar": astar, "option": {"setting": OPT_SETTING, "column": OPT_COL, "notional": OPT_N0, "start_day": OPT_START},
         "weights": w, "result": E, "inputs": inputs("dev", opt["dev"], carry_path, read, days),
         "option_sha256": {k: G.sha(opt[k]) for k in ("dev", "sealed")}}
    S["checks"] = [["두 칸의 달 수가 같고 빠진 달이 없다", len(carry) == len(option) == len(G.months(*window))],
                   ["비중의 합이 1이고 둘 다 0과 1 사이다", abs(w["carry"] + w["option"] - 1) < 1e-15 and 0 < w["carry"] < 1],
                   ["sealed·raw 폴더를 읽지 않았다", S["inputs"]["sealed_files"] == 0 and S["inputs"]["raw_files"] == 0],
                   ["모든 달에 VIX 변화가 있다", E["vix_jump"]["months_without_vix_change"] == 0]]
    if not all(ok for _, ok in S["checks"]):
        print("stop: a check failed:", [n for n, ok in S["checks"] if not ok])
        return 1
    write(S, res, "dev")
    return 0


def run_sealed(res=None, opt=None, window=None, crisis=None, opt_sha=None):
    res, opt, opt_sha = RES if res is None else res, OPT if opt is None else opt, OPT_SHA256 if opt_sha is None else opt_sha
    window, crisis = WINDOW["sealed"] if window is None else window, CRISIS["sealed"] if crisis is None else crisis
    if not all(os.path.isfile(os.path.join(res, f)) for f in TEST_FILES):
        print(f"stop: the sealed test has not been run ({', '.join(TEST_FILES)} must exist); nothing is read")
        return 1
    if any(os.path.exists(os.path.join(res, f)) for f in FILES["sealed"]):
        print(f"stop: {FILES['sealed']} already exist (nothing is overwritten)")
        return 1
    dev_path, test_path = os.path.join(res, FILES["dev"][0]), os.path.join(res, TEST_FILES[0])
    if not os.path.isfile(dev_path):
        print("stop: results/portfolio19_dev.json (the fixed weights) is missing")
        return 1
    with open(dev_path, encoding="utf-8") as f:
        dev = json.load(f)
    with open(test_path, encoding="utf-8") as f:
        test = json.load(f)
    if test.get("mode") != "sealed" or test.get("astar") != dev["astar"]:
        print("stop: results/test19_summary.json is not the sealed run of the same A*")
        return 1
    if same_files(opt, opt_sha, ("dev", "sealed")) or dev["option_sha256"] != {k: opt_sha[k] for k in ("dev", "sealed")}:
        print("stop: an option result file differs from the one hashed at the freeze")
        return 1
    dw = tuple(G.ym(x + "-01") for x in dev["window"])
    w0 = inverse_vol(carry_from_csv(os.path.join(res, "dev19_monthly.csv"), dev["astar"], dw), option_monthly(opt["dev"], dw)[0])
    w = dev["weights"]
    if abs(w0["carry"] - w["carry"]) > 1e-12:
        print("stop: the weights recomputed from the development files differ from the fixed weights")
        return 1
    try:
        carry = carry_from_test(test_path, dev["astar"], window)
        option, days = option_monthly(opt["sealed"], window)
    except (ValueError, KeyError, OSError) as e:
        print("stop: the monthly returns cannot be made:", e)
        return 1
    D, read = G.load(("build", "sealed"))                                  # the sealed VIX, only after the test results exist
    bad = G.verify(read)
    if bad:
        print("stop: sha256 differs from the manifests:", bad)
        return 1
    E = evaluate(window, carry, option, w, G.vix_change(D, window), crisis)
    jump = E["vix_jump"]["corr"]
    S = {"written": dt.datetime.now().isoformat(timespec="seconds"), "mode": "sealed", "window": [G.lab(window[0]), G.lab(window[1])],
         "astar": dev["astar"], "option": dev["option"], "weights": w, "result": E,
         "inputs": inputs("sealed", opt["sealed"], test_path, read, days),
         "sentences": [SENTENCES["sharpe"][E["bundle_sharpe_above_both"]]] + ([] if jump is None else [SENTENCES["jump"][bool(jump >= 0.5)]]),
         "predictions": [{"item": PRED[0][0], "probability": PRED[0][1], "outcome": E["bundle_sharpe_above_both"]},
                         {"item": PRED[1][0], "probability": PRED[1][1], "outcome": None if jump is None else bool(jump >= 0.5)}],
         "development": {k: dev["result"]["metrics"][k] for k in ("carry", "option", "bundle")}}
    S["checks"] = [["두 칸의 달 수가 같고 빠진 달이 없다", len(carry) == len(option) == len(G.months(*window))],
                   ["비중은 동결 때의 숫자이고, 개발 파일로 다시 계산한 값과 같다", abs(w0["carry"] - w["carry"]) <= 1e-12 and abs(w0["option"] - w["option"]) <= 1e-12],
                   ["모든 달에 VIX 변화가 있다", E["vix_jump"]["months_without_vix_change"] == 0]]
    write(S, res, "sealed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", action="store_true")
    g.add_argument("--sealed", action="store_true")
    a = ap.parse_args()
    return run_dev() if a.dev else run_sealed()


if __name__ == "__main__":
    sys.exit(main())
