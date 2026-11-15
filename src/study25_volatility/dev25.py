#!/usr/bin/env python3
"""Research 25 validation run (SPEC25 4-6, SPEC25_ANSWERS; step-2 prompt 2): the existing universe 2019-2020 with SPY of
research 21's development bars, the same parent orders as research 23's validation period. Made once on Kyle's word;
refuses when its results exist. Research 23's frozen model (model23.json) is read after its sha256 check and never refitted.
  gate        (stop, nothing written, when a check fails) the parents, the per-parent outputs of S0, S2i, R0, the
              probability at every decision point and the mean IS of S0, S2i, C3, R0 equal research 23's validation run
  tables      S0, S2i, C3, R0 and the 14 candidates under L3D1 and L0D0: mean IS with its interval, the 95th percentile,
              the paired difference to R0 with its interval, the dollar saving against R0 (ANSWERS A7), firing, by year
  A*          SPEC25 4 and QUESTIONS 19, applied mechanically; A* = R0 -> "개발에서 개선 없음", no collection, no sealed run
  only when A* is not R0: the control (condition turned around), the delay check, the MDE of H for N = 20, 12, 8
  diagnostics SPEC25 6 (never used for a rule)
  -> results/dev25_summary.json, dev25_rows.csv, dev25_report.md, dev25_daily.csv
No sealed path is read. Nothing after 2020-12-31 is reachable (loader22.dev_plan).
  python dev25.py run [--workers 4]"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import core25 as C  # noqa: E402

E, S, L22 = C.E, C.S, C.L22
RES = os.path.join(HERE, "results")
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research25_data/dev_results")
CALENDAR = os.path.join(HERE, "calendar25.csv")
VALID = (2019, 2020)
FILES = ("dev25_summary.json", "dev25_rows.csv", "dev25_report.md", "dev25_daily.csv")
NS = (20, 12, 8)                                                 # SPEC25 1: N is fixed when the collection starts
TRIALS_BEFORE, TRIALS_ADDED = 851, 14                            # SPEC25 9: the 14 candidates
REPORT = ("S0", "S2i", "C3", "R0") + C.CANDS
# research 23's validation run (2019-2020): its per-parent arrays (outside the repository) and its summary
R23_PARENTS = os.path.join(os.environ.get("DATA_DIR", "data"), "research23_data/dev_results/dev23_valid_parents.npz")
R23_SUMMARY = os.path.join(PROJ, "study23_combined_forecast", "results", "dev23_summary.json")
CHECK23 = True                                                   # False in the synthetic tests only (no research 23 values there)
SENT_NONE = "직전 변동성과 거래량을 넣은 규칙은 검증 구간에서 E*를 이기지 못했다. 그래서 봉인 데이터로 시험하지 않았다"
PREDICTIONS = (("A*가 R0가 아님", 0.25), ("H 확인 (A*가 있을 때)", 0.35), ("조기 체결 후보 8개(A1, B1, C1)가 모두 R0보다 평균 IS가 높음", 0.50),
               ("검증 구간에서 가족 B의 최선이 가족 A의 최선보다 평균 IS가 낮음", 0.60))
RULE_KO = {"A1": "g ≥ θ → 조기 체결", "A2": "g ≥ θ → 대기 유지", "B1": "p ≥ p_hi 이고 q ≥ κ → 조기 체결", "B2": "p ≤ 0.40 이고 q ≥ κ → 대기 유지",
           "B3": "B1(p_hi 0.60) + B2", "C1": "p ≥ p_hi 이고 v ≥ 2 → 조기 체결"}


def sealed_days():
    return int(len(pd.read_csv(CALENDAR)))


def plan(years, workers):
    """(tasks of the years with adv, plan info, adv info): development sources only."""
    tasks, cal, pinfo = C.dev_tasks(years)
    tasks, ainfo = C.with_adv(tasks, cal, workers)
    return tasks, pinfo, ainfo


# ---------------------------------------------------------------- regression against research 23's validation run
def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))


def regression(par, OUT, names, G, pairs=()):
    """The validation period against research 23's record: parents, per-parent outputs of S0, S2i, R0 and of every (run
    here, run of research 23) in `pairs`, the probability and the target at every decision point, the mean IS of S0, S2i,
    C3, R0 (differences must be exactly 0). CHECK23 False (the synthetic tests only): the comparisons are skipped."""
    base = [names.index(m) for m in C.BASE]
    chk = {"n_parents": int(len(par)), "targets": "research 23's validation run (dev23_valid_parents.npz, dev23_summary.json)" if CHECK23 else "none (synthetic data)",
           "shares_sum_to_qty": bool(np.all(OUT[:, :, :, 0] + OUT[:, :, :, 2] == E.QTY)),
           "no_rule_action_in_base_runs": bool(not OUT[:, base, :, 9:11].any()),
           "market_at_nonpositive_price": int(OUT[:, :, :, 11].sum()), "no_date_after_2020": bool(par.date.max() <= L22.DEV_LAST)}
    ok = chk["shares_sum_to_qty"] and chk["no_rule_action_in_base_runs"] and chk["no_date_after_2020"]
    if not CHECK23:
        return bool(ok), chk
    with np.load(R23_PARENTS) as z:
        old = {k: z[k] for k in z.files}
    runs23 = [str(x) for x in old["runs"]]
    chk["parents_equal"] = bool(len(par) == len(old["date"]) and same(par.date.to_numpy().astype("U8"), old["date"]) and same(par.sym.to_numpy().astype("U8"), old["sym"])
                                and all(same(par[k].to_numpy(), old[k]) for k in ("dir", "start", "mid2", "spread")))
    chk["lrules_equal"] = bool(list(old["lrules"]) == list(E.LRULES))
    cmp = [(m, m) for m in C.BASE] + list(pairs)
    chk["outputs_equal"] = {f"{a} = research 23's {b}": bool(chk["parents_equal"] and same(OUT[:, names.index(a)], old["out"][:, runs23.index(b)])) for a, b in cmp}
    chk["outputs_differing_parents"] = {f"{a} = research 23's {b}": (int((OUT[:, names.index(a)] != old["out"][:, runs23.index(b)]).any(axis=(0, 2)).sum())
                                                                    if chk["parents_equal"] else None) for a, b in cmp}
    if G is not None and "P23" in G:
        chk["probability_equal"] = bool(same(G["P23"], old["P23"]))
        chk["probability_max_abs_diff"] = float(np.nanmax(np.abs(G["P23"] - old["P23"]))) if G["P23"].shape == old["P23"].shape else None
        chk["target_and_valid_equal"] = bool(same(G["y"], old["y"]) and same(G["valid"], old["valid"]))
        ok = ok and chk["probability_equal"] and chk["target_and_valid_equal"]
    tabs23 = json.load(open(R23_SUMMARY, encoding="utf-8"))["tables"]
    chk["mean_is"], worst = {}, 0.0
    codes, days = C.day_codes(par.date.to_numpy())
    n_day = np.bincount(codes, minlength=len(days)).astype(np.float64)
    for lab, lr, dep in E.VARIANTS:
        IS, _ = C.is_table(par, OUT, names, lr, dep)
        for a, b in [("S0", "S0"), ("S2i", "S2i"), ("C3", "C3"), ("R0", "R0")] + list(pairs):
            got = float(np.bincount(codes, weights=IS[a], minlength=len(days)).sum() / n_day.sum())     # the table mean of engine18.boot_stat
            want = float(tabs23[lab][b]["is_mean"])
            worst = max(worst, abs(got - want))
            chk["mean_is"][f"{lab} {a}"] = {"got": got, "study23_combined_forecast": want, "research23_run": b, "abs_diff": abs(got - want),
                                            "equal_3dp": round(got, 3) == round(want, 3)}
    chk["mean_is_max_abs_diff"] = worst
    ok = (ok and chk["parents_equal"] and chk["lrules_equal"] and all(chk["outputs_equal"].values()) and worst == 0.0)
    return bool(ok), chk


# ---------------------------------------------------------------- report
def flat_rows(tabs):
    rows = []
    for lab, t in tabs.items():
        for m, r in t.items():
            row = {"variant": lab, "run": m, "n_parents": r["n_parents"], "mean_is_c": r["is_mean"], "is_ci_lo": r["is_ci_lo"],
                   "is_ci_hi": r["is_ci_hi"], "p95_is_c": r["p95"], "minus_R0_c": r["minus_R0_mean"], "minus_R0_lo": r["minus_R0_ci_lo"],
                   "minus_R0_hi": r["minus_R0_ci_hi"], "minus_R0_p": r["minus_R0_p_one_sided"], "p95_minus_R0_c": r["p95_minus_R0"],
                   "p95_minus_R0_lo": r["p95_minus_R0_ci_lo"], "p95_minus_R0_hi": r["p95_minus_R0_ci_hi"], "usd_saved_vs_R0": r["usd_saved_vs_R0"],
                   "fired_parent_frac": r["firing"]["parents"], "fired_decision_frac": r["firing"]["decisions"],
                   "early_parent_frac": r["firing"]["early_parents"], "hold_parent_frac": r["firing"]["hold_parents"],
                   "limit_share_frac": r["limit_share_frac"], "deadline_parent_frac": r["deadline_parent_frac"], "mean_done_s": r["mean_done_s"]}
            row.update({f"minus_R0_{y}": v["minus_R0"] for y, v in r["by_year"].items()})
            row.update({f"usd_saved_vs_R0_{y}": v["usd_saved_vs_R0"] for y, v in r["by_year"].items()})
            rows.append(row)
    return pd.DataFrame(rows)


def f3(x):
    return "—" if x is None else f"{x:+.3f}"


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def usd(x):
    return "—" if x is None else f"{round(x) + 0:+,.0f}"


def report(out):
    """The report text (built before any file is written)."""
    s, sel, dg = out["sim"], out["selection"], out["diagnostics"]
    gc = s["grid_counts"]
    L_ = ["# 연구 25 · 검증 실행 (기존 유니버스 2019–2020, 한 번)", "",
          f"- 종목·일 {s['symbol_days']:,}개, 주문 {s['parents']:,}건, {s['days']:,}일. 예측 확률은 연구 23의 동결 모델(model23.json `{out['model_sha256'][:12]}…`)이다.",
          "- IS 단위는 ¢/주다. CI는 일 블록 부트스트랩(21일, 10,000회, seed 25)의 95% 구간이다. \"− R0\"는 같은 주문끼리의 차이다(R0 = E*).",
          "- 달러 절감은 주문마다 (IS_R0 − IS_규칙) × 1,000주를 더한 값이다(양수면 R0보다 쌈).",
          "- 신호: g = 직전 1분 변동폭(bp) ÷ vol_pre, q = 직전 1분 변동폭(¢) ÷ 스프레드, v = 직전 60초 체결량 ÷ 평소 체결량, p = 예측 확률.", "",
          "| 규칙 | 조건 | 엔진 경로 |", "|---|---|---|"]
    for c in C.CANDS:
        sp = C.SPECS[c]
        L_.append(f"| {c} | {RULE_KO[sp['rule']]} ({'θ' if sp['family'] == 'A' else 'κ' if sp['family'] == 'B' else 'v'} = {sp['threshold']:g}) | {sp['engine']} |")
    L_.append("")
    for lab in ("L3D1", "L0D0"):
        L_ += [f"## 후보 표 · {lab}" + (" (주 가정)" if lab == "L3D1" else " (보조)"), "",
               "| 규칙 | 평균 IS [95% CI] | p95 | − R0 [95% CI] | p95 − R0 | 달러 절감 (R0 대비) | 발동: 주문 | 발동: 결정 시점 |", "|---|---|---|---|---|---|---|---|"]
        for m in REPORT:
            r = out["tables"][lab][m]
            L_.append(f"| {m} | {r['is_mean']:.3f} [{r['is_ci_lo']:.3f}, {r['is_ci_hi']:.3f}] | {r['p95']:.2f} | {f3(r['minus_R0_mean'])} "
                      f"[{f3(r['minus_R0_ci_lo'])}, {f3(r['minus_R0_ci_hi'])}] | {f3(r['p95_minus_R0'])} | {usd(r['usd_saved_vs_R0'])} | "
                      f"{100 * r['firing']['parents']:.1f}% | {100 * r['firing']['decisions']:.2f}% |")
        L_ += ["", f"| 연도별 − R0 ({lab}) | " + " | ".join(str(y) for y in out["years"]) + " | 달러 절감 " + " / ".join(str(y) for y in out["years"]) + " |",
               "|---|" + "---|" * (len(out["years"]) + 1)]
        for m in ("S0", "S2i", "C3") + C.CANDS:
            by = out["tables"][lab][m]["by_year"]
            L_.append(f"| {m} | " + " | ".join(f3(by[str(y)]["minus_R0"]) for y in out["years"]) + " | "
                      + " / ".join(usd(by[str(y)]["usd_saved_vs_R0"]) for y in out["years"]) + " |")
        L_.append("")
    L_ += ["## A* 선택 (SPEC25 4절, 질문 19번)", "",
           f"- L3D1 평균 IS가 가장 낮은 후보: {sel['lowest_mean']} ({sel['lowest_mean_c']:.3f}¢). R0는 {sel['r0_mean_c']:.3f}¢이고 차이는 {sel['r0_minus_lowest_c']:+.3f}¢다(문턱 0.02¢)."]
    if out["astar"] == "R0":
        L_ += ["- 0.02¢ 넘게 낮지 않으므로 **A* = R0**이다. **개발에서 개선 없음**: 새 종목을 수집하지 않고 봉인 실행을 하지 않는다.", f"- \"{SENT_NONE}\""]
    else:
        L_ += [f"- 0.02¢ 안의 후보: {', '.join(sel['tied_within_0.02'])}. p95가 가장 낮은 후보: {', '.join(sel['lowest_p95_among_tied'])}.",
               f"- **A* = {out['astar']}**. H(A* − E* < 0)를 새 유니버스 E의 봉인 데이터로 시험한다(수집은 Kyle이 허락한 뒤)."]
    if out.get("control"):
        L_ += ["", "## 대조 실행: 변동성·거래량 조건만 뒤집음 (판정 아님, 후보 아님)", "",
               "| 가정 | 대조 평균 IS [95% CI] | 대조 − E* [95% CI] | A* − 대조 [95% CI] | 발동: 주문 (대조 / A*) |", "|---|---|---|---|---|"]
        for lab, v in out["control"].items():
            a, b, c = v["control_is"], v["control_minus_Estar"], v["Astar_minus_control"]
            L_.append(f"| {lab} | {a['mean']:.3f} [{a['ci_lo']:.3f}, {a['ci_hi']:.3f}] | {f3(b['mean'])} [{f3(b['ci_lo'])}, {f3(b['ci_hi'])}] | "
                      f"{f3(c['mean'])} [{f3(c['ci_lo'])}, {f3(c['ci_hi'])}] | {100 * v['control_firing']['parents']:.1f}% / {100 * v['astar_firing']['parents']:.1f}% |")
    if out.get("delay_check"):
        L_ += ["", "## 지연 점검 (참고: 결정은 막대 t, 실행은 막대 t + 1)", "", "| 가정 | A* 지연 − R0 지연 [95% CI] | R0 지연 − R0 | A* 지연 − A* |", "|---|---|---|---|"]
        a_ = out["astar"]
        for lab, v in out["delay_check"].items():
            a, b, c = v[f"{a_}_delay_minus_R0_delay"], v["R0_delay_minus_R0"], v[f"{a_}_delay_minus_{a_}"]
            L_.append(f"| {lab} | {f3(a['mean'])} [{f3(a['ci_lo'])}, {f3(a['ci_hi'])}] | {f3(b['mean'])} | {f3(c['mean'])} |")
    if out.get("power"):
        pw = out["power"]
        L_ += ["", f"## 검정력 (H: A* − E*, 유니버스 E {pw['days_new']:,}일, 단측 α 0.05, 검정력 80%)", "", "| N (연당) | 검증 구간의 효과 | MDE | 판단 |", "|---|---|---|---|"]
        for n in NS:
            p_ = pw["by_n"][str(n)]
            L_.append(f"| {n} | {f3(p_['effect_dev_c'])}¢ | {p_['mde_c']['0.0500']:.3f}¢ | {'검정력 부족' if p_['underpowered']['0.0500'] else '검정력 충분'} |")
    L_ += ["", "## 진단 (SPEC25 6절, 규칙을 바꾸는 데 쓰지 않는다)", "",
           "| 신호 | 정의된 시점 | 격자 대비 | " + " | ".join(f"p{p}" for p in C.PCTS) + " |", "|---|---|---|" + "---|" * len(C.PCTS)]
    for k in ("g", "q", "v"):
        x = dg["percentiles"][k]
        L_.append(f"| {k} | {x['n']:,} | {100 * x['share_of_grid']:.1f}% | " + " | ".join(f"{x[f'p{p}']:.3f}" if f"p{p}" in x else "—" for p in C.PCTS) + " |")
    L_ += ["", "| 규칙 | 조건이 참인 결정 시점 (격자 대비) | 발동: 결정 시점 (L3D1) | 발동: 주문 (L3D1) |", "|---|---|---|---|"]
    for c in C.CANDS:
        f = out["tables"]["L3D1"][c]["firing"]
        L_.append(f"| {c} | {100 * dg['condition_share_of_grid'][c]:.2f}% | {100 * f['decisions']:.2f}% | {100 * f['parents']:.1f}% |")
    L_ += ["", "| q 구간 | 시점 | 격자 대비 | 다음 60초 중간값 변화의 절댓값 평균 (¢) | 채점 시점 | 예측 AUC |", "|---|---|---|---|---|---|"]
    for b in dg["q_bins"]:
        L_.append(f"| {'신호 없음' if b['bin'] == 'no signal' else b['bin']} | {b['points']:,} | {100 * b['share_of_grid']:.1f}% | "
                  f"{'—' if b['mean_abs_move_60s_c'] is None else format(b['mean_abs_move_60s_c'], '.3f')} | {b['scored_points']:,} | {f4(b['auc'])} |")
    L_ += ["", "## 예측표 대조 (검증 구간에서 판정할 수 있는 항목)", "", "| 항목 | 예측 | 결과 |", "|---|---|---|"]
    for p_ in out["predictions_check"]:
        L_.append(f"| {p_['item']} | {p_['probability']:.2f} | {p_['outcome']} |")
    c = out["checks"]["regression"]
    L_ += ["", "## 점검", "",
           f"- 회귀: 주문 {c['n_parents']:,}건. " + ("연구 23의 검증 실행과 주문, S0·S2i·R0의 주문별 결과, 결정 시점의 예측 확률이 같고 평균 IS의 최대 차이는 "
                                              f"{c['mean_is_max_abs_diff']:.1e}¢다." if "mean_is_max_abs_diff" in c else "합성 자료라 대조하지 않았다."),
           f"- 결정 시점 {gc['points']:,}개 가운데 q가 정의된 시점 {gc['q_defined']:,}개, g {gc['g_defined']:,}개, v {gc['v_defined']:,}개. "
           f"호가는 유효한데 q가 없는 시점 {gc['quote_valid_q_undefined']:,}개(창 미충족 {gc['quote_valid_window_not_full']:,}, 스프레드 0 {gc['quote_valid_spread_zero']:,}).",
           f"- vol_pre가 없거나 0인 주문 {gc['parents_vol_pre_undefined_or_zero']:,}건(가족 A는 그 주문에서 E*와 같다). 평소 체결량이 없는 종목·일 "
           f"{s['adv']['adv_undefined'] + s['adv']['adv_zero']:,}개(C1은 그날 E*와 같다).",
           f"- 수량 합계 확인 {c['shares_sum_to_qty']}. 0 이하 가격의 시장가 주문 {c['market_at_nonpositive_price']}건. 일 단위 파일의 합계와 표의 평균의 가장 큰 차이: "
           f"{out['checks']['daily_file_max_gap_c']:.2e}¢.",
           f"- 시행 수: {out['trials_before']} → {out['trials_cumulative']}(후보 14개). 대조 실행, 지연 점검, 진단은 세지 않는다. 실행 {out['runtime_s']} s.", ""]
    return "\n".join(L_)


def predictions_check(tabs, astar):
    t = tabs["L3D1"]
    early = all(t[c]["minus_R0_mean"] > 0 for c in S.EARLY_FILL)
    best = {f: min(t[c]["is_mean"] for c in C.CANDS if C.SPECS[c]["family"] == f) for f in ("A", "B")}
    return [{"item": PREDICTIONS[0][0], "probability": PREDICTIONS[0][1], "outcome": "일어남" if astar != "R0" else "일어나지 않음", "astar": astar},
            {"item": PREDICTIONS[1][0], "probability": PREDICTIONS[1][1], "outcome": "봉인 실행에서 판정" if astar != "R0" else "해당 없음 (A* = R0)"},
            {"item": PREDICTIONS[2][0], "probability": PREDICTIONS[2][1], "outcome": "맞음" if early else "틀림",
             "minus_R0_c": {c: t[c]["minus_R0_mean"] for c in S.EARLY_FILL}},
            {"item": PREDICTIONS[3][0], "probability": PREDICTIONS[3][1], "outcome": "맞음" if best["B"] < best["A"] else "틀림", "best_mean_is_c": best}]


# ---------------------------------------------------------------- the run
def valid_stage(workers):
    t0 = time.time()
    if any(os.path.exists(os.path.join(RES, f)) for f in FILES):
        raise SystemExit("validation results exist: the validation run is made once (nothing is overwritten)")
    model = C.model23()
    msha = C.sha(C.MODEL23)
    tasks, pinfo, ainfo = plan(VALID, workers)
    names = list(C.BASE + C.CANDS)
    par, OUT, G, info = C.simulate(tasks, names, (), workers, model=model, grid=True)
    ok, reg = regression(par, OUT, names, G)
    if not ok:
        print(json.dumps(reg, indent=1))
        raise SystemExit("regression failed: nothing written")
    tabs, st = C.tables(par, OUT, names, REPORT)
    codes, days, idx, W = st
    rows = {c: {"mean": tabs["L3D1"][c]["is_mean"], "p95": tabs["L3D1"][c]["p95"], "fired": tabs["L3D1"][c]["firing"]["parents"]} for c in C.CANDS}
    astar, sel = C.select_astar(rows, tabs["L3D1"]["R0"]["is_mean"])
    improves = astar != "R0"
    dly, second, ctrl, daily2, pw = None, None, None, None, None
    T_new = sealed_days()
    if improves:
        cn = C.CTRL + astar
        par2, OUT2, _, _ = C.simulate(tasks, ("R0", astar, cn), ("R0", astar), workers, model=model)
        n2 = ["R0", astar, cn, "R0@d1", f"{astar}@d1"]
        second = bool(np.array_equal(OUT2[:, 0], OUT[:, names.index("R0")]) and np.array_equal(OUT2[:, 1], OUT[:, names.index(astar)])
                      and par2[["date", "sym", "dir", "start"]].equals(par[["date", "sym", "dir", "start"]]))
        dly = C.delay_table(par2, OUT2, n2, ("R0", astar), st)
        ctrl = C.control_table(par2, OUT2, n2, astar, st)
        daily2 = C.daily_rows(par2, OUT2[:, [2]], [cn])
        IS, _ = C.is_table(par, OUT, names, 3, 1)
        pw = {"days_new": T_new, "series": f"{astar} - R0, L3D1",
              "by_n": {str(n): C.power(IS[astar] - IS["R0"], par, n, T_new, idx, (C.ALPHA,)) for n in NS}}
    daily = C.daily_rows(par, OUT, names)
    if daily2 is not None:
        daily = pd.concat([daily, daily2], ignore_index=True)
    gap = 0.0
    for lab in ("L3D1", "L0D0"):
        for m in REPORT:
            g = daily[(daily.variant == lab) & (daily.run == m)]
            gap = max(gap, abs(g.is_sum_c.sum() / g.n_parents.sum() - tabs[lab][m]["is_mean"]))
    out = {"astar": astar, "improves_in_development": improves, "sealed_run": improves, "selection": sel, "sentence_no_improvement": None if improves else SENT_NONE,
           "hypotheses": ["H"] if improves else [], "model_sha256": msha, "tables": tabs,
           "usd_saved_vs_R0_L3D1": {c: tabs["L3D1"][c]["usd_saved_vs_R0"] for c in C.CANDS},
           "delay_check": dly, "control": ctrl, "power": pw, "diagnostics": C.diagnostics(G), "predictions_check": predictions_check(tabs, astar),
           "years": sorted({int(d[:4]) for d in days}),
           "sim": {**info, **pinfo, "adv": ainfo, "days": len(days), "first_day": days[0], "last_day": days[-1], "sealed_days_for_power": T_new},
           "checks": {"regression": reg, "second_pass_equal": second, "daily_file_max_gap_c": float(gap)},
           "daily": {"file": "results/dev25_daily.csv", "rows": int(len(daily)), "columns": list(daily.columns)},
           "candidates": {c: C.SPECS[c] for c in C.CANDS}, "trials_before": TRIALS_BEFORE, "trials_added": TRIALS_ADDED,
           "trials_cumulative": TRIALS_BEFORE + TRIALS_ADDED, "boot": {"B": E.BOOT_B, "block": E.BOOT_BLOCK, "seed": C.SEED},
           "runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")}
    text = report(out)
    summary = json.dumps(out, indent=1, default=C.jdefault, ensure_ascii=False)
    os.makedirs(RES, exist_ok=True)
    os.makedirs(LOCAL, exist_ok=True)
    f_par = os.path.join(LOCAL, "dev25_valid_parents.npz")
    if not os.path.exists(f_par):
        np.savez_compressed(f_par, date=par.date.to_numpy().astype("U8"), sym=par.sym.to_numpy().astype("U8"), dir=par.dir.to_numpy(),
                            start=par.start.to_numpy(), mid2=par.mid2.to_numpy(), spread=par.spread.to_numpy(), out=OUT, runs=np.array(names),
                            lrules=np.array(E.LRULES))
    daily.to_csv(os.path.join(RES, FILES[3]), index=False, float_format="%.6f")
    flat_rows(tabs).to_csv(os.path.join(RES, FILES[1]), index=False)
    with open(os.path.join(RES, FILES[2]), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(RES, FILES[0]), "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    valid_stage(a.workers)
