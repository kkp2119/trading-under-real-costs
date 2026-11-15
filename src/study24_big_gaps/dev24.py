#!/usr/bin/env python3
"""Research 24 step 2 development run (SPEC24 4; step-2 prompt 3): the pairs of 2015-2020 (both families together), market
orders with 10,000 dollars per leg, every order 0, 1 and 5 seconds late, fill assumptions L3D1 (primary) and L0D0.
Development bars only (loader24); no sealed file is read. Run once on Kyle's word: it refuses unless the synthetic tests
(results/engine24_tests.json) and the regression test (results/regress24.json) passed, and refuses to overwrite a result.
  per candidate (4: W {10, 30} x kappa {2, 3}, z_in 3) and run (delay x fill assumption): mean daily P&L with the day-block
              95% interval (21 days, 10,000, seed 24), trades, trades per day, win rate, holding time, pnl = mid - cost -
              fee per trade, the share of trades with an order above the displayed size; by family, by pair type, by
              year, by pair; without the 5 largest days, without the 5 largest pairs, without both
  A*          SPEC24 4, mechanically (stats20.select_astar on the delay-1 L3D1 rows); a mean <= 0 -> "개발에서 수익 없음":
              no freeze and no sealed run
  diagnosis   share of signal points with sigma at the 1 bp floor; share of |z| >= z_in points kept out by kappa
  power       MDE of H1 at the sealed length (one-sided 0.05, power 80%)
  daily file  results/dev24_daily.csv: one row per pair-day and candidate with the columns of run_test24's daily file
  python dev24.py run [--workers 4]
Trade rows of A* (delay 1) -> $DATA_DIR/research24_data/dev_results/, aggregates -> results/dev24_*."""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core24 as C24  # noqa: E402
import loader24 as L24  # noqa: E402
import stats24 as ST24  # noqa: E402

C, ST20 = C24.C, C24.ST20
RES = os.path.join(HERE, "results")
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research24_data/dev_results")
FILES = ("dev24_summary.json", "dev24_report.md", "dev24_daily.csv")
SEALED_WINDOW = ("20210104", "20261110")               # calendar days only (calendar24.csv); no sealed bar is read here
MAIN = C24.run_name(ST24.DELAY_MAIN, ST24.VAR_MAIN)


def sealed_days():
    return int(sum(SEALED_WINDOW[0] <= d <= SEALED_WINDOW[1] for d in L24.close_bars()))


def prerequisites():
    t = json.load(open(os.path.join(RES, "engine24_tests.json")))
    r = json.load(open(os.path.join(RES, "regress24.json")))
    if t["passed"] != t["total"] or not r["pass"]:
        raise SystemExit("the synthetic tests or the regression test did not pass")
    return {"synthetic_tests": f"{t['passed']}/{t['total']}", "regression_pass": r["pass"],
            "regression_max_abs_diff": {"study20_pairs": r["study20_pairs"]["max_abs_diff"], "study21_more_pairs": r["study21_more_pairs"]["max_abs_diff"],
                                        "research21_delay_1": r["study21_more_pairs"]["delay_1_max_abs_diff"]}}


def daily_gap(daily, cands):
    """Largest difference between the daily file and the tables: dollars and trades of every candidate and run."""
    g = daily.groupby("setting").sum(numeric_only=True)
    usd, trd = 0.0, 0
    for name, T in cands.items():
        for d in ST24.DELAYS:
            for v in ST24.VARS:
                r = T["runs"][C24.run_name(d, v)]
                usd = max(usd, abs(float(g.loc[name, f"d{d}_{v}"]) - r["total_usd"]))
                trd = max(trd, abs(int(g.loc[name, f"trades_d{d}"]) - r["trades"]))
    return {"usd": float(usd), "trades": int(trd)}


def usd(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+,.1f}"


def run_line(label, r):
    return (f"| {label} | {usd(r['mean_daily_usd'])} [{usd(r['ci_lo'])}, {usd(r['ci_hi'])}] | {r['trades']:,} | {r['trades_per_day']:.2f} | "
            f"{100 * r['win_rate']:.1f}% | {r['mean_hold_s']:.0f} | {usd(r['pnl_per_trade'])} = {usd(r['mid_per_trade'])} − {r['cost_per_trade']:,.1f} − {r['fee_per_trade']:,.1f} | "
            f"{100 * r['over_trade_frac']:.1f}% | {100 * r['over_share_frac']:.1f}% |") if r["trades"] else f"| {label} | {usd(r['mean_daily_usd'])} | 0 | 0.00 | — | — | — | — | — |"


def report(out):
    """The report text (built before any file is written)."""
    s = out["sim"]
    L_ = ["# 연구 24 · 개발 실행 (쌍 2015–2020, 한 번)", "",
          f"- 쌍·일 {s['pair_days']:,}개, {s['days']:,}일({s['first_day']} ~ {s['last_day']}), 서로 다른 쌍 {s['pairs']}개. 다리당 1만 달러, 시장가, z_in = 3.",
          "- 일 손익은 그날 모든 쌍의 합계(달러)이고 거래가 없는 날은 0이다. CI는 일 블록 부트스트랩(21일, 10,000회, seed 24)의 95% 구간이다.",
          "- 선택과 판정의 기준은 지연 1초, L3D1이다. 거래당 손익은 중간값 손익 − 체결 비용 − 수수료다.", ""]
    head = ["| 실행 | 평균 일 손익 [95% CI] | 거래 수 | 하루 거래 | 이긴 거래 | 보유(초) | 거래당 손익 = 중간값 − 비용 − 수수료 | 잔량 초과 거래 | 초과 주식 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, T in out["candidates"].items():
        L_ += [f"## 후보 {name}" + (" (A*)" if name == out["astar"] else ""), ""] + head
        for d in ST24.DELAYS:
            for v in ST24.VARS:
                L_.append(run_line(f"지연 {d}초, {v}", T["runs"][C24.run_name(d, v)]))
        L_ += ["", "| 나눈 부분 (지연 1초, L3D1) | 쌍·일 | 평균 일 손익 [95% CI] | 거래 수 | 거래당 손익 |", "|---|---|---|---|---|"]
        for key, lab in (("by_family", {f: f"가족 {f}" for f in ST24.FAMILIES}), ("by_type", ST24.TYPE_KO)):
            for k, x in T[key].items():
                r = x["runs"][MAIN]
                L_.append(f"| {lab[k]} | {x['pair_days']:,} | {usd(r['mean_daily_usd'])} [{usd(r['ci_lo'])}, {usd(r['ci_hi'])}] | {r['trades']:,} | {usd(r['pnl_per_trade']) if r['trades'] else '—'} |")
        yrs = T["runs"][MAIN]["by_year"]
        L_ += ["", "| 연도별 평균 일 손익 (지연 1초, L3D1) | " + " | ".join(yrs) + " |", "|---|" + "---|" * len(yrs), "| | " + " | ".join(usd(v) for v in yrs.values()) + " |"]
        w = T["without_top"][MAIN]
        L_ += ["", "| 뺀 것 (지연 1초, L3D1) | 평균 일 손익 [95% CI] | 전체 합계에서 차지한 비중 |", "|---|---|---|",
               f"| 없음 | {usd(w['all']['mean_daily_usd'])} [{usd(w['all']['ci_lo'])}, {usd(w['all']['ci_hi'])}] | |",
               f"| 손익이 가장 큰 5일 | {usd(w['without_top_days']['mean_daily_usd'])} [{usd(w['without_top_days']['ci_lo'])}, {usd(w['without_top_days']['ci_hi'])}] | {100 * w['without_top_days']['share_of_total']:.0f}% |",
               f"| 손익이 가장 큰 5쌍 | {usd(w['without_top_pairs']['mean_daily_usd'])} [{usd(w['without_top_pairs']['ci_lo'])}, {usd(w['without_top_pairs']['ci_hi'])}] | {100 * w['without_top_pairs']['share_of_total']:.0f}% |",
               f"| 둘 다 | {usd(w['without_both']['mean_daily_usd'])} [{usd(w['without_both']['ci_lo'])}, {usd(w['without_both']['ci_hi'])}] | |",
               "", "- 뺀 5쌍: " + ", ".join(f"{q} ({usd(v)})" for q, v in w["without_top_pairs"]["pairs_removed"]) + ".", ""]
    sel = out["selection"]
    L_ += ["## A* 선택 (SPEC24 4절)", "", f"- 지연 1초, L3D1의 평균 일 손익이 가장 높은 후보: {sel['highest_mean']} ({usd(sel['highest_mean_usd'])}달러). "
           f"5% 안의 후보: {', '.join(sel['tied_within_5pct'])}. 거래 수: {sel['trades_of_tied']}.",
           f"- **A* = {out['astar']}**, 평균 일 손익 {usd(sel['astar_mean_usd'])}달러."]
    L_ += ([f"- 0보다 크므로 봉인 구간에서 H1을 시험한다."] if out["profitable_in_development"] else
           [f"- 0 이하이므로 **개발에서 수익 없음**이다. 봉인 실행을 하지 않는다.", f"- \"{ST24.NO_PROFIT}\""])
    p = out["power"]
    L_ += ["", f"## 검정력 (봉인 구간 {p['days_new']:,}일, 단측 α 0.05, 검정력 80%)", "",
           f"- A*의 개발 효과 {usd(p['dev_effect_usd'])}달러/일, MDE {p['mde']['0.05']:,.1f}달러/일 → {'검정력 부족' if p['underpowered'] else '검정력 충분'}.", "",
           "## 진단 (규칙을 바꾸는 데 쓰지 않는다)", ""]
    dg = out["diag"]["family"]
    L_.append("- σ 하한(1bp)이 적용된 시점의 비율: " + ", ".join(f"{k} {100 * v['share']:.1f}%" for k, v in dg["sigma_floor"].items()) + ".")
    L_.append("- |z| ≥ 3인 시점 가운데 κ 조건에 막힌 비율: " + ", ".join(f"{k} {100 * v['share']:.1f}%" for k, v in dg["kappa_blocked"].items()) + ".")
    c = out["checks"]
    L_ += ["", "## 점검", "", f"- 지연 0초의 거래 시간표가 신호만의 시간표와 다른 설정: {c['settings_with_m_trades_unlike_the_signal_timetable']}개.",
           f"- 호가 문제로 뺀 쌍·일 {c['pair_days_left_out_no_state_after_first_state']:,}개. 한쪽 다리만 있거나 둘 다 없는 쌍·일: {out['sim']['skipped_pair_days']}.",
           f"- 일 단위 파일과 표의 차이: 손익 {c['daily_file_gap']['usd']:.2e}달러, 거래 수 {c['daily_file_gap']['trades']}.",
           f"- 시행 수: {out['trials_before']} → {out['trials_cumulative']}(후보 4개). 지연 0초와 5초, 진단은 세지 않는다. 실행 {out['runtime_s']} s.", ""]
    return "\n".join(L_)


def analyse(workers, check_prerequisites=True):
    t0 = time.time()
    if any(os.path.exists(os.path.join(RES, f)) for f in FILES):
        raise SystemExit("development results exist: the development run is made once (nothing is overwritten)")
    pre = prerequisites() if check_prerequisites else {}
    tasks, days, skipped = L24.dev_plan()
    cfgs = ST24.candidates()
    names = [ST24.sig_name(c) for c in cfgs]
    A, X, info, ex = C24.simulate(tasks, L24.DEV_ROOT, False, cfgs, workers=workers, diag=True)
    if info["settings_with_m_trades_unlike_the_signal_timetable"]:
        print(json.dumps(info, indent=1))
        raise SystemExit("the delay-0 trades differ from the signal-only timetable: nothing written")
    n_days = len(days)
    idx = ST24.boot_index(n_days)
    cands = {name: {"signal": list(c), **C24.setting_tables(A, X, tasks, days, ci, ST24.DELAYS, idx, name)} for ci, (name, c) in enumerate(zip(names, cfgs))}
    prim = [{"name": name, "mean": cands[name]["runs"][MAIN]["mean_daily_usd"], "trades": cands[name]["runs"][MAIN]["trades"]} for name in names]
    astar, sel = ST24.select_astar(prim)
    ci_a = names.index(astar)
    profitable = bool(sel["profitable_in_development"])
    A1, _, _, ex1 = C24.simulate(tasks, L24.DEV_ROOT, False, [cfgs[ci_a]], delays=(ST24.DELAY_MAIN,), keep=((0, ST24.DELAY_MAIN),), workers=workers)
    second = bool(np.array_equal(A1[:, 0, 0], A[:, ci_a, ST24.DELAYS.index(ST24.DELAY_MAIN)]))
    M = C.meta(tasks, days)
    d = C24.series(A, M, ci_a, ST24.DELAYS.index(ST24.DELAY_MAIN), ST24.VAR_MAIN, n_days)
    t_new = sealed_days()
    pw = ST20.mde(d, idx, t_new, alphas=(ST24.ALPHA,))
    pw = {**pw, "mde": {"0.05": pw["mde"][str(ST24.ALPHA)]}, "dev_effect_usd": float(d.mean()), "series": f"daily P&L of {astar}, delay 1, L3D1"}
    pw["underpowered"] = bool(not pw["mde"]["0.05"] <= pw["dev_effect_usd"])
    daily = pd.DataFrame([r for ci, name in enumerate(names) for r in C24.pair_daily(A, tasks, ci, name, ST24.DELAYS)], columns=C24.daily_cols(ST24.DELAYS))
    years = sorted({t["year"] for t in tasks})
    out = {"astar": astar, "astar_signal": list(cfgs[ci_a]), "selection": sel, "profitable_in_development": profitable, "sealed_run": profitable,
           "sentence_no_profit": None if profitable else ST24.NO_PROFIT, "hypotheses": ["H1"] if profitable else [],
           "amount_usd": ST24.AMOUNT, "delays": list(ST24.DELAYS), "delay_main": ST24.DELAY_MAIN, "candidates": cands,
           "prediction_third_line_delay0_L3D1_mean_usd": cands[astar]["runs"][C24.run_name(0, ST24.VAR_MAIN)]["mean_daily_usd"],
           "diag": C24.diag_shares(ex["diag"], tasks, days, cfgs), "power": pw,
           "checks": {**pre, "settings_with_m_trades_unlike_the_signal_timetable": info["settings_with_m_trades_unlike_the_signal_timetable"],
                      "pair_days_left_out_no_state_after_first_state": info["pair_days_left_out_no_state_after_first_state"],
                      "second_pass_equal": second, "daily_file_gap": daily_gap(daily, cands)},
           "sim": {**info, "days": n_days, "first_day": days[0], "last_day": days[-1], "pairs": len({t["key"] for t in tasks}),
                   "pair_days_by_year": {str(y): int(sum(t["year"] == y for t in tasks)) for y in years},
                   "pairs_by_year": {str(y): len({t["key"] for t in tasks if t["year"] == y}) for y in years},
                   "pair_days_by_family": {f: int(sum(t["family"] == f for t in tasks)) for f in ST24.FAMILIES},
                   "pair_days_by_type": {k: int(sum(t["type"] == k for t in tasks)) for k in ST24.TYPES},
                   "skipped_pair_days": skipped, "sealed_calendar_days": t_new, "workers": workers},
           "daily": {"file": "results/dev24_daily.csv", "rows": int(len(daily)), "columns": list(daily.columns)},
           "trials_before": ST24.TRIALS_BEFORE, "trials_added": ST24.TRIALS_ADDED, "trials_cumulative": ST24.TRIALS_BEFORE + ST24.TRIALS_ADDED,
           "boot": {"B": ST24.BOOT_B, "block": ST24.BOOT_BLOCK, "seed": ST24.BOOT_SEED}, "runtime_s": round(time.time() - t0, 1),
           "written": dt.datetime.now().isoformat(timespec="seconds")}
    text = report(out)
    summary = json.dumps(out, indent=1, default=C24.jdefault, ensure_ascii=False)
    os.makedirs(RES, exist_ok=True)
    os.makedirs(LOCAL, exist_ok=True)
    tp = os.path.join(LOCAL, f"dev24_trades_{astar}_delay{ST24.DELAY_MAIN}.npz")
    if not os.path.exists(tp):
        np.savez_compressed(tp, M=ex1["kept"][(0, ST24.DELAY_MAIN)], dates=np.array([t["date"] for t in tasks]), pairs=np.array([t["name"] for t in tasks]))
    daily.to_csv(os.path.join(RES, FILES[2]), index=False, float_format="%.6f")
    with open(os.path.join(RES, FILES[1]), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(RES, FILES[0]), "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(text)
    return json.loads(summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    analyse(a.workers)
