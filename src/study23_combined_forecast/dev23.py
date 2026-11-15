#!/usr/bin/env python3
"""Research 23 development run (SPEC23 3-5, SPEC23_ANSWERS; step-2 prompt 2): the existing universe with SPY of research
21's development bars. Two stages, each made once on Kyle's word; each refuses when its results exist.
  fit     training years 2015-2018: the nine inputs and y at the decision grid of every parent, no order is simulated.
          Gate first (stop, nothing written): research 18's training order with the four inputs of research 18 reproduces
          the frozen coefficients of study18_forecast/model18.json. Then the nine-input model is fitted once -> model23.json,
          model23.sha256. Reference only: nine fits with one input left out, the training AUC of both models.
          -> results/fit23_summary.json
  valid   validation years 2019-2020: S0, S2i, R0 (E*), the 19 candidates with the nine-input model's p and the same 19
          rules with research 18's frozen model (reference) under L0 and L3. Regression first (stop, nothing written, when
          a check fails): parents, the mean IS of S0, S2i, C3, R0 at 3 decimals, the AUC of research 18's model and the
          19 research 18 candidates equal research 18's recorded validation values.
          prediction  AUC, hit rate, Brier skill score of both models on the same points; AUC difference with its 95%
                      interval; the AUC without each input (reference, the model is not changed)
          tables      per run and variant: mean IS with its interval, 95th percentile, the paired difference to R0, firing
          A*          SPEC23 4 and ANSWERS A6, applied mechanically; A* = R0 -> "개발에서 개선 없음", no H2
          only when A* is not R0: the control with 1 - p and the delay check (reference, not counted as trials)
          power       MDE of H2 for universe D (N = 20, the days of calendar23.csv)
          -> results/dev23_summary.json, dev23_rows.csv, dev23_report.md, dev23_daily.csv, dev23_daily_pred.csv
No sealed path is read. Nothing after 2020-12-31 is reachable (loader22.dev_plan).
  python dev23.py fit [--workers 4]      python dev23.py valid      python dev23.py run (both)"""
import argparse
import datetime as dt
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import core23 as C  # noqa: E402

E, F, M, L22 = C.E, C.F, C.M, C.L22
RES = os.path.join(HERE, "results")
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research23_data/dev_results")
CALENDAR = os.path.join(HERE, "calendar23.csv")
MODEL = os.path.join(HERE, "model23.json")
TRAIN, VALID = (2015, 2016, 2017, 2018), (2019, 2020)
FIT_FILE = "fit23_summary.json"
FILES = ("dev23_summary.json", "dev23_rows.csv", "dev23_report.md", "dev23_daily.csv", "dev23_daily_pred.csv")
N_NEW = 20                                                       # SPEC23_ANSWERS A7
# research 18's recorded validation values (2019-2020): study18_forecast/results/dev18_rows.csv and dev18_summary.json
TARGETS = {"L3D1": {"S0": 4.537231, "S2i": 3.216910, "C3": 3.179135, "R0": 3.217108},
           "L0D0": {"S0": 3.745448, "S2i": 2.742201, "C3": 2.702849, "R0": 2.668957}}
N_PARENTS, SCORED18, AUC18 = 99_080, 5_155_574, 0.568675224143437
ROWS18 = os.path.join(PROJ, "study18_forecast", "results", "dev18_rows.csv")
CHECK18 = True                                                   # False in the synthetic tests only (no research 18 values there)
TOL_COEF, TOL_AUC, TOL_ROW = 1e-9, 1e-12, 5.1e-7
TRIALS_BEFORE, TRIALS_ADDED = 826, 20                            # step-2 prompt 2: one model and 19 candidates
REPORT = ("S0", "S2i", "C3", "R0") + C.CANDS
REF18 = tuple(C.M18 + c for c in C.CANDS)
ASTAR18 = "R0"                                                   # research 18's A* (study18_forecast/results/dev18_summary.json)
SENT_NONE = "신호를 합친 예측은 검증 구간에서 E*를 이기지 못했다. 그래서 개선 가설은 봉인 데이터로 시험하지 않았다"
LABEL = {"imbalance": "1 잔량 불균형", "spread_ticks": "2 스프레드", "vol_pre": "3 vol_pre", "mid_change_5s": "4 직전 5초 중간값 변화",
         "a_spy": "5 a_t (SPY 기준)", "a_own": "6 a_t (자기 가격 기준)", "trade_imbalance_30s": "7 체결 불균형 30초",
         "trade_intensity_60s": "8 체결 강도 60초", "imbalance_x_intensity": "9 7번 × 8번"}


def sealed_days():
    return int(len(pd.read_csv(CALENDAR)))


def plan(years, workers):
    """(tasks of the years with adv, plan info, adv info): development sources only."""
    tasks, cal, pinfo = C.dev_tasks(years)
    tasks, ainfo = C.with_adv(tasks, cal, workers)
    return tasks, pinfo, ainfo


# ---------------------------------------------------------------- fit stage
def regression_fit(X, y, valid, vp):
    """SPEC23_ANSWERS A5: the new training code with research 18's four inputs against study18_forecast/model18.json."""
    if not CHECK18:
        return {"pass": True, "targets": "none (synthetic data)"}
    m18 = C.model18()
    m4 = M.fit(X, y, valid, vp, cols=[0, 1, 2, 3])
    d = {k: float(np.max(np.abs(np.asarray(m4[k]) - np.asarray(m18[k])))) for k in ("mean", "sd", "coef")}
    d.update({k: abs(float(m4[k]) - float(m18[k])) for k in ("intercept", "vol_pre_fill", "base_rate_train")})
    same_n = bool(m4["n_train_points"] == m18["n_train_points"] and m4["n_train_parents"] == m18["n_train_parents"]
                  and m4["features"] == m18["features"])
    return {"pass": bool(same_n and max(d.values()) <= TOL_COEF), "exactly_equal": bool(same_n and max(d.values()) == 0.0), "max_abs_diff": d,
            "tolerance": TOL_COEF, "n_train_points": [m4["n_train_points"], m18["n_train_points"]],
            "n_train_parents": [m4["n_train_parents"], m18["n_train_parents"]], "coef_new_code": m4["coef"], "coef_model18": m18["coef"],
            "sklearn": [m4["sklearn"], m18["sklearn"]], "n_iter": [m4["n_iter"], m18["n_iter"]], "targets": "study18_forecast/model18.json"}


def fit_stage(workers):
    t0 = time.time()
    side = os.path.splitext(MODEL)[0] + ".sha256"
    f_fit = os.path.join(RES, FIT_FILE)
    if any(os.path.exists(p) for p in (MODEL, side, f_fit)):
        raise SystemExit("the model or its summary exists: the model is fitted once (nothing is overwritten)")
    tasks, pinfo, ainfo = plan(TRAIN, workers)
    par, _, G, info = C.simulate(tasks, workers=workers, grid=True)
    X, y, valid, vp = G["X"], G["y"], G["valid"], par.vol_pre.to_numpy()
    reg = regression_fit(X, y, valid, vp)
    if not reg["pass"]:
        print(json.dumps(reg, indent=1))
        raise SystemExit("the four-input fit does not reproduce research 18's model: nothing written")
    model = M.fit(X, y, valid, vp)
    model.update({"train_years": sorted({int(d[:4]) for d in par.date.unique()}), "written": dt.datetime.now().isoformat(timespec="seconds")})
    allc = list(range(len(F.FEATS)))
    loo = {F.FEATS[j]: M.fit(X, y, valid, vp, cols=[c for c in allc if c != j]) for j in allc}       # reference, no file of their own
    keep = valid & (y >= 0)
    m18 = C.model18()
    it = X[..., F.I_INT][keep]
    train = {"scored_points": int(keep.sum()), "grid_points": int(y.size), "base_rate": float(y[keep].mean()),
             "auc_combined": C.auc_point(M.predict(model, X, valid)[keep], y[keep].astype(np.int8)),
             "auc_research18": C.auc_point(M.predict(M.from18(m18), X, valid)[keep], y[keep].astype(np.int8)),
             "intensity_undefined_share_of_scored": float(np.isnan(it).mean()), "vol_pre_undefined_parents": int(np.isnan(vp).sum()),
             "no_signal_share": {"a_spy": float(1 - info["grid_counts"]["signal_spy"] / y.size), "a_own": float(1 - info["grid_counts"]["signal_own"] / y.size)}}
    out = {"model": model, "leave_one_out_models": loo, "train": train, "regression_four_inputs": reg,
           "sim": {**info, **pinfo, "adv": ainfo, "days": int(par.date.nunique()), "first_day": par.date.min(), "last_day": par.date.max()},
           "runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")}
    text = json.dumps(out, indent=1, default=C.jdefault, ensure_ascii=False)          # built before any file is written
    os.makedirs(RES, exist_ok=True)
    out["model_sha256"] = M.save(model, MODEL)
    with open(f_fit, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"model_sha256": out["model_sha256"], **json.loads(text)}, indent=1, ensure_ascii=False))
    print(f"fit: {model['n_train_points']:,} points of {model['n_train_parents']:,} parents, n_iter {model['n_iter']}, model23.json {out['model_sha256'][:12]}…, "
          f"{out['runtime_s']} s")
    for f, c in zip(model["features"], model["coef"]):
        print(f"  {f:24s} {c:+.5f}")


# ---------------------------------------------------------------- validation stage
def regression(par, OUT, names, auc18=None, scored18=None):
    """The validation period against research 18's record (CHECK18 False, the synthetic tests only: comparisons skipped)."""
    chk = {"n_parents": int(len(par)), "n_parents_equal_research18": bool(not CHECK18 or len(par) == N_PARENTS), "mean_is_3dp": {},
           "targets": "research 18's validation values (dev18_rows.csv, dev18_summary.json)" if CHECK18 else "none (synthetic data)"}
    worst = 0.0
    r18 = pd.read_csv(ROWS18).set_index(["variant", "method"]).mean_is_c if CHECK18 else None
    for lab, lr, dep in E.VARIANTS:
        IS, _ = C.is_table(par, OUT, names, lr, dep)
        for m, want in (TARGETS[lab].items() if CHECK18 else ()):
            got = float(IS[m].mean())
            chk["mean_is_3dp"][f"{lab} {m}"] = {"got": got, "want": want, "pass": round(got, 3) == round(want, 3)}
        for m in (REF18 if CHECK18 else ()):
            if m in IS:
                worst = max(worst, abs(float(IS[m].mean()) - float(r18[(lab, C.engine_name(m))])))
    n18 = sum(m in names for m in REF18)
    chk["research18_candidates"] = {"compared": n18 * len(E.VARIANTS) if CHECK18 else 0, "max_abs_diff_mean_is": worst,
                                    "pass": bool(not CHECK18 or (n18 == len(REF18) and worst <= TOL_ROW))}
    chk["auc_research18"] = {"got": auc18, "want": AUC18 if CHECK18 else None, "scored_points": scored18,
                             "pass": bool(not CHECK18 or (auc18 is not None and abs(auc18 - AUC18) <= TOL_AUC and scored18 == SCORED18))}
    base = [names.index(m) for m in C.BASE]
    chk["shares_sum_to_qty"] = bool(np.all(OUT[:, :, :, 0] + OUT[:, :, :, 2] == E.QTY))
    chk["no_rule_action_in_base_runs"] = bool(not OUT[:, base, :, 9:11].any())
    chk["market_at_nonpositive_price"] = int(OUT[:, :, :, 11].sum())
    chk["no_date_after_2020"] = bool(par.date.max() <= L22.DEV_LAST)
    ok = (chk["n_parents_equal_research18"] and all(v["pass"] for v in chk["mean_is_3dp"].values()) and chk["research18_candidates"]["pass"]
          and chk["auc_research18"]["pass"] and chk["shares_sum_to_qty"] and chk["no_rule_action_in_base_runs"] and chk["no_date_after_2020"])
    return bool(ok), chk


def h1_power(se_valid, par, t_new):
    """Reference (research 18's scaling): the bootstrap SE of the validation AUC scaled to universe D by the number of
    orders and by the number of days alone; MDE of AUC - 0.5, one-sided, power 80%."""
    from scipy.stats import norm
    n_dev, t_dev = float(par.groupby("date").sym.nunique().mean()), int(par.date.nunique())
    se_o, se_d = se_valid * math.sqrt(n_dev * t_dev / (N_NEW * t_new)), se_valid * math.sqrt(t_dev / t_new)
    z = {a: norm.ppf(1 - a) + norm.ppf(C.POWER) for a in (C.ALPHA / 2, C.ALPHA)}
    return {"se_valid": se_valid, "stocks_per_day_valid": n_dev, "days_valid": t_dev, "n_new": N_NEW, "days_new": t_new,
            "se_new_orders": se_o, "se_new_days": se_d, "mde_auc_minus_0.5_orders": {f"{a:.4f}": z[a] * se_o for a in z},
            "mde_auc_minus_0.5_days": {f"{a:.4f}": z[a] * se_d for a in z}}


def flat_rows(tabs):
    rows = []
    for lab, t in tabs.items():
        for m, r in t.items():
            row = {"variant": lab, "run": m, "n_parents": r["n_parents"], "mean_is_c": r["is_mean"], "is_ci_lo": r["is_ci_lo"],
                   "is_ci_hi": r["is_ci_hi"], "p95_is_c": r["p95"], "minus_R0_c": r["minus_R0_mean"], "minus_R0_lo": r["minus_R0_ci_lo"],
                   "minus_R0_hi": r["minus_R0_ci_hi"], "minus_R0_p": r["minus_R0_p_one_sided"], "p95_minus_R0_c": r["p95_minus_R0"],
                   "p95_minus_R0_lo": r["p95_minus_R0_ci_lo"], "p95_minus_R0_hi": r["p95_minus_R0_ci_hi"],
                   "fired_parent_frac": r["firing"]["parents"], "fired_decision_frac": r["firing"]["decisions"],
                   "early_parent_frac": r["firing"]["early_parents"], "hold_parent_frac": r["firing"]["hold_parents"],
                   "limit_share_frac": r["limit_share_frac"], "deadline_parent_frac": r["deadline_parent_frac"], "mean_done_s": r["mean_done_s"]}
            row.update({f"minus_R0_{y}": v["minus_R0"] for y, v in r["by_year"].items()})
            rows.append(row)
    return pd.DataFrame(rows)


def f3(x):
    return "—" if x is None else f"{x:+.3f}"


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def report(out):
    """The report text (built before any file is written)."""
    md, pr, s = out["model"], out["prediction"], out["sim"]
    L_ = ["# 연구 23 · 개발 실행 (기존 유니버스: 학습 2015–2018, 검증 2019–2020, 각각 한 번)", "",
          f"- 검증 구간: 종목·일 {s['symbol_days']:,}개, 주문 {s['parents']:,}건, {s['days']:,}일. SPY 파일이 없는 종목·일 {s['symbol_days_without_ref']}개.",
          "- IS 단위는 ¢/주다. CI는 일 블록 부트스트랩(21일, 10,000회, seed 23)의 95% 구간이다. \"− R0\"는 같은 주문끼리의 차이다(R0 = E*).",
          "- 후보 이름은 연구 18과 같다: R1_h(p ≥ h면 조기 체결), R2_l(p ≤ l이면 대기 유지), R3_h_l(둘 다). p는 입력 9개 모델의 예측 확률이다.", "",
          "## 모델 (학습 2015–2018, 한 번 적합)", "", "| 입력 | 표준화 계수 | 학습 평균 | 학습 SD |", "|---|---|---|---|"]
    for f, c, m, sd in zip(md["features"], md["coef"], md["mean"], md["sd"]):
        L_.append(f"| {LABEL.get(f, f)} | {c:+.4f} | {m:.4f} | {sd:.4f} |")
    L_ += [f"| 절편 | {md['intercept']:+.4f} | | |", "",
           f"- 학습 시점 {md['n_train_points']:,}개(주문 {md['n_train_parents']:,}건), 반복 {md['n_iter']}회. vol_pre가 없는 주문은 {md['fills']['vol_pre']:.4f} bp, "
           f"체결 강도를 낼 수 없는 날은 {md['fills']['intensity']:.4f}(학습 구간 중앙값)로 채웠다. model23.json sha256 `{out['model_sha256']}`.", "",
           "## 예측 평가 (검증 2019–2020, 두 모델을 같은 시점에서 채점)", "",
           "| 모델 | AUC [95% CI] | 적중률 (0.5) | Brier | Brier skill score |", "|---|---|---|---|---|"]
    for k, name in (("combined", "합친 모델 (입력 9개)"), ("study18_forecast", "연구 18 모델 (입력 4개)")):
        x = pr[k]
        L_.append(f"| {name} | {x['auc']:.4f} [{x['auc_boot']['ci_lo']:.4f}, {x['auc_boot']['ci_hi']:.4f}] | {x['accuracy_0.5']:.4f} | {x['brier']:.5f} | {x['bss']:+.4f} |")
    d = pr["auc_diff"]
    L_ += ["", f"- AUC 차이(합친 모델 − 연구 18 모델): {d['mean']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]. 채점 시점 {pr['counts']['scored_points']:,}개, {pr['counts']['days']}일.",
           "- 연도별 AUC(합친 모델 / 연구 18 모델): " + ", ".join(f"{y} {f4(v['auc_combined'])} / {f4(v['auc_research18'])}" for y, v in pr["by_year"].items()) + ".",
           f"- 참고로 학습 구간의 AUC는 합친 모델 {f4(out['train']['auc_combined'])}, 연구 18 모델 {f4(out['train']['auc_research18'])}다.", "",
           "### 입력을 하나씩 뺀 모델 (참고: 학습 구간에서 다시 적합, 동결 모델은 바꾸지 않는다)", "", "| 뺀 입력 | 검증 AUC | 줄어든 AUC |", "|---|---|---|"]
    for f, v in out["leave_one_out"].items():
        L_.append(f"| {LABEL.get(f, f)} | {f4(v['auc'])} | {v['auc_drop']:+.5f} |")
    L_.append("")
    for lab in ("L3D1", "L0D0"):
        L_ += [f"## 후보 표 · {lab}" + (" (주 가정)" if lab == "L3D1" else " (보조)"), "",
               "| 규칙 | 평균 IS [95% CI] | p95 | − R0 [95% CI] | p95 − R0 | 발동: 주문 | 발동: 결정 시점 |", "|---|---|---|---|---|---|---|"]
        for m in REPORT:
            r = out["tables"][lab][m]
            L_.append(f"| {m} | {r['is_mean']:.3f} [{r['is_ci_lo']:.3f}, {r['is_ci_hi']:.3f}] | {r['p95']:.2f} | {f3(r['minus_R0_mean'])} "
                      f"[{f3(r['minus_R0_ci_lo'])}, {f3(r['minus_R0_ci_hi'])}] | {f3(r['p95_minus_R0'])} | {100 * r['firing']['parents']:.1f}% | "
                      f"{100 * r['firing']['decisions']:.2f}% |")
        L_ += ["", f"| 연도별 − R0 ({lab}) | " + " | ".join(str(y) for y in out["years"]) + " |", "|---|" + "---|" * len(out["years"])]
        for m in ("S0", "S2i", "C3") + C.CANDS:
            L_.append(f"| {m} | " + " | ".join(f3(out["tables"][lab][m]["by_year"][str(y)]["minus_R0"]) for y in out["years"]) + " |")
        L_.append("")
    r18 = out["research18_candidates"]
    L_ += ["## 비교: 연구 18의 모델로 돌린 같은 19개 후보 (참고, 선택에 쓰지 않는다)", "",
           f"- 연구 18의 A*는 {r18['astar_research18']}였다(연구 18의 선택 규칙). 그래서 \"연구 18의 A*\"와의 비교는 R0(E*)와의 비교와 같다.",
           f"- 연구 18 모델의 후보 가운데 L3D1 평균 IS가 가장 낮은 것은 {r18['lowest_mean']}({r18['lowest_mean_c']:.3f}¢, R0와의 차이 {f3(r18['lowest_minus_R0_c'])}¢)다.", "",
           "| 규칙 (연구 18 모델) | L3D1 평균 IS | − R0 [95% CI] | 발동: 주문 | L0D0 평균 IS | − R0 [95% CI] |", "|---|---|---|---|---|---|"]
    for m in REF18:
        a, b = out["tables"]["L3D1"][m], out["tables"]["L0D0"][m]
        L_.append(f"| {C.engine_name(m)} | {a['is_mean']:.3f} | {f3(a['minus_R0_mean'])} [{f3(a['minus_R0_ci_lo'])}, {f3(a['minus_R0_ci_hi'])}] | "
                  f"{100 * a['firing']['parents']:.1f}% | {b['is_mean']:.3f} | {f3(b['minus_R0_mean'])} [{f3(b['minus_R0_ci_lo'])}, {f3(b['minus_R0_ci_hi'])}] |")
    sel = out["selection"]
    L_ += ["", "## A* 선택 (SPEC23 4절, 답변 A6)", "",
           f"- L3D1 평균 IS가 가장 낮은 후보: {sel['lowest_mean']} ({sel['lowest_mean_c']:.3f}¢). R0는 {sel['r0_mean_c']:.3f}¢이고 차이는 {sel['r0_minus_lowest_c']:+.3f}¢다(문턱 0.02¢)."]
    if out["astar"] == "R0":
        L_ += ["- 0.02¢ 넘게 낮지 않으므로 **A* = R0**이다. H2는 없다.", f"- \"{SENT_NONE}\""]
    else:
        L_ += [f"- 0.02¢ 안의 후보: {', '.join(sel['tied_within_0.02'])}. p95가 가장 낮은 후보: {', '.join(sel['lowest_p95_among_tied'])}.",
               f"- **A* = {out['astar']}**. H2(A* − E* < 0)를 봉인 데이터로 시험한다."]
    if out.get("control"):
        L_ += ["", "## 대조 실행: 예측 확률을 1 − p로 뒤집음 (판정 아님, 후보 아님)", "",
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
    pw = out["power"]
    L_ += ["", f"## 검정력 (유니버스 D: 연 {N_NEW}개, {pw['days_new']:,}일, 단측, 검정력 80%)", ""]
    p2 = pw["H2"] if pw.get("H2") else pw["reference_lowest_candidate"]
    a1 = sorted(p2["mde_c"])[0]
    L_ += [("- H2 (A* − E*)" if pw.get("H2") else f"- 참고 (H2 없음. 평균 IS가 가장 낮은 후보 {sel['lowest_mean']} − R0)")
           + f": 검증 구간의 효과 {f3(p2['effect_dev_c'])}¢, MDE {p2['mde_c'][a1]:.3f}¢ (α {float(a1):.4f}), {p2['mde_c']['0.0500']:.3f}¢ (α 0.05) → "
           + ("검정력 부족" if p2["underpowered"][a1] else "검정력 충분") + ".",
           f"- 참고 (H1): 검증 AUC의 부트스트랩 SE {pw['H1_auc']['se_valid']:.5f}. 유니버스 D로 환산한 MDE(AUC − 0.5)는 주문 수 비례 "
           f"{pw['H1_auc']['mde_auc_minus_0.5_orders']['0.0500']:.4f}, 일수만 비례 {pw['H1_auc']['mde_auc_minus_0.5_days']['0.0500']:.4f}다(α 0.05)."]
    c = out["checks"]
    L_ += ["", "## 점검", "",
           "- 회귀: " + ", ".join(f"{k} {v['got']:.3f}" for k, v in c["regression"]["mean_is_3dp"].items()) + " → 연구 18의 검증 구간 값과 소수 3자리까지 같다.",
           f"- 연구 18 모델의 검증 AUC {f4(c['regression']['auc_research18']['got'])}(기록 {f4(c['regression']['auc_research18']['want'])}). 연구 18 모델 후보의 "
           f"평균 IS와 연구 18 기록의 가장 큰 차이 {c['regression']['research18_candidates']['max_abs_diff_mean_is']:.1e}¢.",
           f"- 주문 수 {c['regression']['n_parents']:,}건. 수량 합계 확인 {c['regression']['shares_sum_to_qty']}. 0 이하 가격의 시장가 주문 {c['regression']['market_at_nonpositive_price']}건.",
           f"- 일 단위 파일의 합계와 표의 평균의 가장 큰 차이: {c['daily_file_max_gap_c']:.2e}¢.",
           f"- 시행 수: {out['trials_before']} → {out['trials_cumulative']}(모델 1개와 후보 19개). 대조 실행, 지연 점검, 입력을 뺀 평가, 연구 18 모델의 후보는 세지 않는다. 실행 {out['runtime_s']} s.", ""]
    return "\n".join(L_)


def valid_stage(workers):
    t0 = time.time()
    if any(os.path.exists(os.path.join(RES, f)) for f in FILES):
        raise SystemExit("validation results exist: the validation run is made once (nothing is overwritten)")
    f_fit = os.path.join(RES, FIT_FILE)
    if not (os.path.exists(MODEL) and os.path.exists(f_fit)):
        raise SystemExit("no fitted model: run the fit stage first")
    model = M.load(MODEL)
    fit = json.load(open(f_fit, encoding="utf-8"))
    msha = M.sha(MODEL)
    if fit["model_sha256"] != msha:
        raise SystemExit("model23.json is not the model of results/fit23_summary.json")
    m18 = C.model18()
    tasks, pinfo, ainfo = plan(VALID, workers)
    names = list(C.BASE + C.CANDS + REF18)
    par, OUT, G, info = C.simulate(tasks, names, (), workers, model=model, m18=m18, grid=True)
    pred, dpred = C.prediction(par, G, model["base_rate_train"], m18["base_rate_train"])
    ok, reg = regression(par, OUT, names, pred["study18_forecast"]["auc"], pred["counts"]["scored_points"])
    if not ok:
        print(json.dumps(reg, indent=1))
        raise SystemExit("regression failed: nothing written")
    keep, ys, _, _, _, _ = C.scored(par, G)
    loo = {}
    for f, m in fit["leave_one_out_models"].items():                               # reference: the model is not changed
        a = C.auc_point(M.predict(m, G["X"], G["valid"])[keep], ys)
        loo[f] = {"auc": a, "auc_drop": pred["combined"]["auc"] - a, "inputs": m["features"]}
    tabs, st = C.tables(par, OUT, names, REPORT + REF18)
    codes, days, idx, W = st
    rows = {c: {"mean": tabs["L3D1"][c]["is_mean"], "p95": tabs["L3D1"][c]["p95"], "fired": tabs["L3D1"][c]["firing"]["parents"]} for c in C.CANDS}
    astar, sel = C.select_astar(rows, tabs["L3D1"]["R0"]["is_mean"])
    h2 = astar != "R0"
    low18 = min(REF18, key=lambda m: (tabs["L3D1"][m]["is_mean"], REF18.index(m)))
    r18 = {"astar_research18": ASTAR18, "note": "research 18's A* was R0, so the comparison with it is the comparison with R0",
           "lowest_mean": C.engine_name(low18), "lowest_mean_c": tabs["L3D1"][low18]["is_mean"], "lowest_minus_R0_c": tabs["L3D1"][low18]["minus_R0_mean"]}
    dly, second, ctrl, daily2 = None, None, None, None
    if h2:
        cn = C.CTRL + astar
        par2, OUT2, _, _ = C.simulate(tasks, ("R0", astar, cn), ("R0", astar), workers, model=model)
        n2 = ["R0", astar, cn, "R0@d1", f"{astar}@d1"]
        second = bool(np.array_equal(OUT2[:, 0], OUT[:, names.index("R0")]) and np.array_equal(OUT2[:, 1], OUT[:, names.index(astar)])
                      and par2[["date", "sym", "dir", "start"]].equals(par[["date", "sym", "dir", "start"]]))
        dly = C.delay_table(par2, OUT2, n2, ("R0", astar), st)
        ctrl = C.control_table(par2, OUT2, n2, astar, st)
        daily2 = C.daily_rows(par2, OUT2[:, [2]], [cn])
    IS, _ = C.is_table(par, OUT, names, 3, 1)
    T_new = sealed_days()
    pw = {"days_new": T_new, "n_new": N_NEW, "H1_auc": h1_power(pred["combined"]["auc_boot"]["se"], par, T_new)}
    if h2:
        pw["H2"] = C.power(IS[astar] - IS["R0"], par, N_NEW, T_new, idx, (C.ALPHA / 2, C.ALPHA))
    else:
        pw["reference_lowest_candidate"] = C.power(IS[sel["lowest_mean"]] - IS["R0"], par, N_NEW, T_new, idx, (C.ALPHA / 2, C.ALPHA))
    daily = C.daily_rows(par, OUT, names)
    if daily2 is not None:
        daily = pd.concat([daily, daily2], ignore_index=True)
    gap = 0.0
    for lab in ("L3D1", "L0D0"):
        for m in REPORT + REF18:
            g = daily[(daily.variant == lab) & (daily.run == m)]
            gap = max(gap, abs(g.is_sum_c.sum() / g.n_parents.sum() - tabs[lab][m]["is_mean"]))
    out = {"astar": astar, "h2_exists": h2, "selection": sel, "sentence_no_h2": None if h2 else SENT_NONE,
           "hypotheses": ["H1"] + (["H2"] if h2 else []), "holm_levels": [C.ALPHA / 2, C.ALPHA] if h2 else [C.ALPHA],
           "model": {k: model[k] for k in ("features", "cols", "mean", "sd", "coef", "intercept", "fills", "n_train_points", "n_train_parents",
                                           "base_rate_train", "n_iter", "sklearn", "estimator", "check_vs_predict_proba_max_abs")},
           "model_sha256": msha, "train": fit["train"], "prediction": pred, "leave_one_out": loo, "tables": tabs, "research18_candidates": r18,
           "delay_check": dly, "control": ctrl, "power": pw, "years": sorted({int(d[:4]) for d in days}),
           "sim": {**info, **pinfo, "adv": ainfo, "days": len(days), "first_day": days[0], "last_day": days[-1], "sealed_days_for_power": T_new, "n_new": N_NEW},
           "checks": {"regression": reg, "second_pass_equal": second, "daily_file_max_gap_c": float(gap)},
           "daily": {"file": "results/dev23_daily.csv", "rows": int(len(daily)), "columns": list(daily.columns),
                     "prediction_file": "results/dev23_daily_pred.csv", "prediction_rows": int(len(dpred))},
           "candidates": list(C.CANDS), "trials_before": TRIALS_BEFORE, "trials_added": TRIALS_ADDED, "trials_cumulative": TRIALS_BEFORE + TRIALS_ADDED,
           "boot": {"B": E.BOOT_B, "block": E.BOOT_BLOCK, "seed": C.SEED}, "runtime_s": round(time.time() - t0, 1),
           "written": dt.datetime.now().isoformat(timespec="seconds")}
    text = report(out)
    summary = json.dumps(out, indent=1, default=C.jdefault, ensure_ascii=False)
    os.makedirs(RES, exist_ok=True)
    os.makedirs(LOCAL, exist_ok=True)
    f_par = os.path.join(LOCAL, "dev23_valid_parents.npz")
    if not os.path.exists(f_par):
        np.savez_compressed(f_par, date=par.date.to_numpy().astype("U8"), sym=par.sym.to_numpy().astype("U8"), dir=par.dir.to_numpy(),
                            start=par.start.to_numpy(), mid2=par.mid2.to_numpy(), spread=par.spread.to_numpy(), out=OUT, runs=np.array(names),
                            lrules=np.array(E.LRULES), P23=G["P23"], P18=G["P18"], y=G["y"], valid=G["valid"])
    daily.to_csv(os.path.join(RES, FILES[3]), index=False, float_format="%.6f")
    dpred.to_csv(os.path.join(RES, FILES[4]), index=False, float_format="%.6f")
    flat_rows(tabs).to_csv(os.path.join(RES, FILES[1]), index=False)
    with open(os.path.join(RES, FILES[2]), "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(os.path.join(RES, FILES[0]), "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "valid", "run"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd in ("fit", "run"):
        fit_stage(a.workers)
    if a.cmd in ("valid", "run"):
        valid_stage(a.workers)
