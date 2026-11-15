#!/usr/bin/env python3
"""Research 22 development run (SPEC22 4-5, SPEC22_ANSWERS; step-2 prompt 2): the existing universe 2015-2020 with SPY of
research 21's development bars as the reference. Run once, on Kyle's word; it refuses when its results exist.
  pass 1   S0, S2i, G0 (E*) and the 12 candidates under L0 and L3. Regression first (stop, nothing written, when a check
           fails): the parents and the mean IS of S0, S2i, G0 and C3 at 3 decimals equal research 17 / 18's development values.
  tables   per run and variant (L3D1, L0D0): mean IS with its 95% interval, 95th percentile, the paired difference to G0
           (same parents) with its interval, firing shares (parents, decision bars), years.
  signal   prediction (ANSWERS A5): adverse mid change 60 / 180 s after the decision grid points, per basis, threshold and
           condition; diagnostics: decision grid points without a signal and with the 1 bp sigma floor.
  A*       SPEC22 4 and ANSWERS A4, applied mechanically; A* = G0 -> "개발에서 개선 없음", no H2.
  delay    when A* is not G0: G0 and A* once more with every action one bar late (reference only).
  control  when A* is not G0 (SPEC22_ADDENDUM D1, no verdict, not a candidate, not counted as a trial): A*'s basis, threshold
           and rule with the sign of the adverse score reversed; control - E* and A* - control over the same parents.
  power    MDE of H1 (E* - S0) and, when it exists, H2 (A* - E*) for universe C (N = 20, the days of calendar22.csv).
Writes results/dev22_summary.json, dev22_rows.csv, dev22_report.md, dev22_daily.csv and, outside the repository,
$DATA_DIR/research22_data/dev_results/dev22_parents.npz. No sealed path is read.
  python dev22.py run [--workers 4]"""
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
import core22 as C  # noqa: E402
import loader22 as L22  # noqa: E402

E = C.E
RES = os.path.join(HERE, "results")
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research22_data/dev_results")
CALENDAR = os.path.join(HERE, "calendar22.csv")
FILES = ("dev22_summary.json", "dev22_rows.csv", "dev22_report.md", "dev22_daily.csv")
N_NEW = 20                                                       # SPEC22_ANSWERS A7
TARGETS = {"L3D1": {"S0": 4.579, "S2i": 3.112, "G0": 3.148, "C3": 3.071},      # study17_new_stocks/results/dev17_rows.csv (C0 = E*),
           "L0D0": {"S0": 3.785, "S2i": 2.674, "G0": 2.605, "C3": 2.627}}      # study18_forecast/dev18.py TARGETS; 2015-2020
N_PARENTS = 295_960
TRIALS_BEFORE, TRIALS_ADDED = 813, 12
REPORT = ("S0", "S2i", "C3", "G0") + C.CANDS
SENT_NONE = "기준 가격 신호는 개발 구간에서 E*를 이기지 못했다. 그래서 개선 가설은 봉인 데이터로 시험하지 않았다"


def sealed_days():
    return int(len(pd.read_csv(CALENDAR)))


def regression(par, OUT, names):
    """TARGETS None (the synthetic tests only): the comparison with research 17's values is skipped and recorded so."""
    chk = {"n_parents": int(len(par)), "n_parents_equal_research17": bool(TARGETS is None or len(par) == N_PARENTS), "mean_is_3dp": {},
           "targets": "research 17 / 18 development values" if TARGETS is not None else "none (synthetic data)"}
    for lab, lr, dep in E.VARIANTS:
        IS, _ = C.is_table(par, OUT, names, lr, dep)
        for m, want in (TARGETS[lab].items() if TARGETS is not None else ()):
            got = float(IS[m].mean())
            chk["mean_is_3dp"][f"{lab} {m}"] = {"got": got, "want": want, "pass": round(got, 3) == want}
    chk["shares_sum_to_qty"] = bool(np.all(OUT[:, :, :, 0] + OUT[:, :, :, 2] == E.QTY))
    chk["no_rule_action_in_base_runs"] = bool(not OUT[:, [names.index(m) for m in C.BASE], :, 9:11].any())
    chk["market_at_nonpositive_price"] = int(OUT[:, :, :, 11].sum())
    chk["no_date_after_2020"] = bool(par.date.max() <= L22.DEV_LAST)
    ok = (chk["n_parents_equal_research17"] and all(v["pass"] for v in chk["mean_is_3dp"].values()) and chk["shares_sum_to_qty"]
          and chk["no_rule_action_in_base_runs"] and chk["no_date_after_2020"])
    return ok, chk


def flat_rows(tabs):
    rows = []
    for lab, t in tabs.items():
        for m, r in t.items():
            row = {"variant": lab, "run": m, "n_parents": r["n_parents"], "mean_is_c": r["is_mean"], "is_ci_lo": r["is_ci_lo"],
                   "is_ci_hi": r["is_ci_hi"], "p95_is_c": r["p95"], "minus_G0_c": r["minus_G0_mean"], "minus_G0_lo": r["minus_G0_ci_lo"],
                   "minus_G0_hi": r["minus_G0_ci_hi"], "minus_G0_p": r["minus_G0_p_one_sided"], "p95_minus_G0_c": r["p95_minus_G0"],
                   "p95_minus_G0_lo": r["p95_minus_G0_ci_lo"], "p95_minus_G0_hi": r["p95_minus_G0_ci_hi"],
                   "fired_parent_frac": r["firing"]["parents"], "fired_decision_frac": r["firing"]["decisions"],
                   "early_parent_frac": r["firing"]["early_parents"], "hold_parent_frac": r["firing"]["hold_parents"],
                   "limit_share_frac": r["limit_share_frac"], "deadline_parent_frac": r["deadline_parent_frac"], "mean_done_s": r["mean_done_s"]}
            row.update({f"minus_G0_{y}": v["minus_G0"] for y, v in r["by_year"].items()})
            rows.append(row)
    return pd.DataFrame(rows)


def f3(x):
    return "—" if x is None else f"{x:+.3f}"


def report(out):
    """The report text (built before any file is written)."""
    L_ = ["# 연구 22 · 개발 실행 (기존 유니버스 2015–2020, 한 번)", "",
          f"- 종목·일 {out['sim']['symbol_days']:,}개, 주문 {out['sim']['parents']:,}건, {out['sim']['days']:,}일. SPY 파일이 없는 종목·일 {out['sim']['symbol_days_without_ref']}개.",
          "- IS 단위는 ¢/주다. CI는 일 블록 부트스트랩(21일, 10,000회, seed 22)의 95% 구간이다. \"− G0\"는 같은 주문끼리의 차이다.",
          "- 후보 이름은 기준(spy = SPY 기준, own = 자기 가격 기준)_z기준값_규칙이다(G1 조기 체결, G2 대기 유지, G3 둘 다).", ""]
    for lab in ("L3D1", "L0D0"):
        L_ += [f"## 후보 표 · {lab}" + (" (주 가정)" if lab == "L3D1" else " (보조)"), "",
               "| 규칙 | 평균 IS [95% CI] | p95 | − G0 [95% CI] | p95 − G0 | 발동: 주문 | 발동: 결정 시점 |", "|---|---|---|---|---|---|---|"]
        for m in REPORT:
            r = out["tables"][lab][m]
            L_.append(f"| {m} | {r['is_mean']:.3f} [{r['is_ci_lo']:.3f}, {r['is_ci_hi']:.3f}] | {r['p95']:.2f} | {f3(r['minus_G0_mean'])} "
                      f"[{f3(r['minus_G0_ci_lo'])}, {f3(r['minus_G0_ci_hi'])}] | {f3(r['p95_minus_G0'])} | {100 * r['firing']['parents']:.1f}% | "
                      f"{100 * r['firing']['decisions']:.2f}% |")
        L_ += ["", f"| 연도별 − G0 ({lab}) | " + " | ".join(str(y) for y in out["years"]) + " |", "|---|" + "---|" * len(out["years"])]
        for m in ("S0",) + C.CANDS:
            L_.append(f"| {m} | " + " | ".join(f3(out["tables"][lab][m]["by_year"][str(y)]["minus_G0"]) for y in out["years"]) + " |")
        L_.append("")
    L_ += ["## 신호의 예측력 (판정 아님, 경로와 무관한 결정 격자 시점)", "",
           "값은 그 시점 뒤 중간값 변화(bp)이고, 주문에 불리한 방향이 양수다.", "",
           "| 기준 | 조건 | 60초 뒤 [95% CI] | 시점 수 | 180초 뒤 [95% CI] | 시점 수 |", "|---|---|---|---|---|---|"]
    lab_g = {"all": "신호가 있는 모든 시점"}
    lab_g.update({f"z{z:g}_{c}": f"a {'≥' if c == 'early' else '≤ −'}{z:g} ({'조기 체결' if c == 'early' else '대기 유지'} 조건)" for z in C.ZTH for c in ("early", "hold")})
    for b in C.BASES:
        for g in C.GROUPS:
            cells = []
            for h in C.HORIZONS:
                r = out["prediction"][b][g][str(h)]
                cells += [f"{f3(r.get('mean_bp'))} [{f3(r.get('ci_lo'))}, {f3(r.get('ci_hi'))}]", f"{r['points']:,}"]
            L_.append(f"| {b} | {lab_g[g]} | " + " | ".join(cells) + " |")
    L_ += ["", "## 진단 (규칙을 바꾸는 데 쓰지 않는다)", ""]
    for b in C.BASES:
        d = out["diagnostics"][b]
        L_.append(f"- {b}: 결정 격자 시점 {d['grid_points']:,}개 중 신호 없음 {100 * d['no_signal_share']:.1f}%, 신호가 있는 시점 중 σ 하한(1bp) 적용 "
                  f"{100 * d['floor_share_of_signal']:.1f}%.")
    s = out["selection"]
    L_ += ["", "## A* 선택 (SPEC22 4절, 답변 A4)", "",
           f"- L3D1 평균 IS가 가장 낮은 후보: {s['lowest_mean']} ({s['lowest_mean_c']:.3f}¢). G0는 {s['g0_mean_c']:.3f}¢이고 차이는 {s['g0_minus_lowest_c']:+.3f}¢다(문턱 0.02¢)."]
    if out["astar"] == "G0":
        L_ += ["- 0.02¢ 넘게 낮지 않으므로 **A* = G0**이다. H2는 없다.", f"- \"{SENT_NONE}\""]
    else:
        L_ += [f"- 0.02¢ 안의 후보: {', '.join(s['tied_within_0.02'])}. p95가 가장 낮은 후보: {', '.join(s['lowest_p95_among_tied'])}.",
               f"- **A* = {out['astar']}**. H2(A* − E* < 0)를 봉인 데이터로 시험한다."]
    if out.get("delay_check"):
        L_ += ["", "## 지연 점검 (참고: 결정은 막대 t, 실행은 막대 t + 1)", "", "| 가정 | A* 지연 − G0 지연 [95% CI] | G0 지연 − G0 | A* 지연 − A* |", "|---|---|---|---|"]
        a_ = out["astar"]
        for lab, v in out["delay_check"].items():
            a, b, c = v[f"{a_}_delay_minus_G0_delay"], v["G0_delay_minus_G0"], v[f"{a_}_delay_minus_{a_}"]
            L_.append(f"| {lab} | {f3(a['mean'])} [{f3(a['ci_lo'])}, {f3(a['ci_hi'])}] | {f3(b['mean'])} | {f3(c['mean'])} |")
    if out.get("control_flip"):
        L_ += ["", "## 대조 실행: 부호를 뒤집은 신호 (추가 문서 D1, 판정 아님, 후보 아님)", "",
               f"A*와 같은 기준·기준값·규칙에서 a_t의 부호만 뒤집었다({out['control_flip']['L3D1']['control']}).", "",
               "| 가정 | 대조 평균 IS [95% CI] | 대조 − E* [95% CI] | A* − 대조 [95% CI] | 발동: 주문 (대조 / A*) |", "|---|---|---|---|---|"]
        for lab, v in out["control_flip"].items():
            a, b, c = v["control_is"], v["control_minus_Estar"], v["Astar_minus_control"]
            L_.append(f"| {lab} | {a['mean']:.3f} [{a['ci_lo']:.3f}, {a['ci_hi']:.3f}] | {f3(b['mean'])} [{f3(b['ci_lo'])}, {f3(b['ci_hi'])}] | "
                      f"{f3(c['mean'])} [{f3(c['ci_lo'])}, {f3(c['ci_hi'])}] | {100 * v['control_firing']['parents']:.1f}% / {100 * v['astar_firing']['parents']:.1f}% |")
    L_ += ["", f"## 검정력 (유니버스 C: 연 {N_NEW}개, {out['power']['H1']['days_new']:,}일, 단측, 검정력 80%)", "",
           "| 가설 | 개발 효과 (¢) | MDE (Holm 첫 수준) | MDE (0.05) | 표시 |", "|---|---|---|---|---|"]
    for h, p in out["power"].items():
        a1 = sorted(p["mde_c"])[0]
        L_.append(f"| {h} | {f3(p['effect_dev_c'])} | {p['mde_c'][a1]:.3f} (α {float(a1):.4f}) | {p['mde_c']['0.0500']:.3f} | "
                  f"{'검정력 부족' if p['underpowered'][a1] else '검정력 충분'} |")
    c = out["checks"]
    L_ += ["", "## 점검", "",
           "- 회귀: " + ", ".join(f"{k} {v['got']:.3f}" for k, v in c["regression"]["mean_is_3dp"].items()) + " → 연구 17·18의 개발 값과 소수 3자리까지 같다.",
           f"- 주문 수 {c['regression']['n_parents']:,}건(연구 17과 같음). 수량 합계 확인 {c['regression']['shares_sum_to_qty']}. 0 이하 가격의 시장가 주문 {c['regression']['market_at_nonpositive_price']}건.",
           f"- 일 단위 파일의 합계와 표의 평균의 가장 큰 차이: {c['daily_file_max_gap_c']:.2e}¢.",
           f"- 시행 수: {out['trials_before']} → {out['trials_cumulative']}. 실행 {out['runtime_s']} s.", ""]
    return "\n".join(L_)


def main(workers):
    t0 = time.time()
    if any(os.path.exists(os.path.join(RES, f)) for f in FILES):
        raise SystemExit("development results exist: the development run is made once (nothing is overwritten)")
    tasks, cal, pinfo = L22.dev_plan()
    if any("sealed" in t["path"] or (t["ref_path"] and "sealed" in t["ref_path"]) for t in tasks):
        raise SystemExit("a sealed path in the development plan")
    names = list(C.BASE + C.CANDS)
    par, OUT, sd, info = C.simulate(tasks, names, (), workers, bases=C.BASES)
    ok, reg = regression(par, OUT, names)
    if not ok:
        print(json.dumps(reg, indent=1))
        raise SystemExit("regression failed: nothing written")
    tabs, st = C.tables(par, OUT, names, REPORT)
    codes, days, idx, W = st
    rows = {c: {"mean": tabs["L3D1"][c]["is_mean"], "p95": tabs["L3D1"][c]["p95"], "fired": tabs["L3D1"][c]["firing"]["parents"]} for c in C.CANDS}
    astar, sel = C.select_astar(rows, tabs["L3D1"]["G0"]["is_mean"])
    h2 = astar != "G0"
    dly, second, ctrl, daily2 = None, None, None, None
    if h2:
        flip = C.flip_name(astar)
        par2, OUT2, _, _ = C.simulate(tasks, ("G0", astar, flip), ("G0", astar), workers, bases=())
        n2 = ["G0", astar, flip, "G0@d1", f"{astar}@d1"]
        second = bool(np.array_equal(OUT2[:, 0], OUT[:, names.index("G0")]) and np.array_equal(OUT2[:, 1], OUT[:, names.index(astar)])
                      and par2[["date", "sym", "dir", "start"]].equals(par[["date", "sym", "dir", "start"]]))
        dly = C.delay_table(par2, OUT2, n2, ("G0", astar), st)
        ctrl = C.control_table(par2, OUT2, n2, astar, st)                       # SPEC22_ADDENDUM D1
        daily2 = C.daily_rows(par2, OUT2[:, [2]], [flip])
    IS, _ = C.is_table(par, OUT, names, 3, 1)
    alphas = (C.ALPHA / (2 if h2 else 1), C.ALPHA)
    T_new = sealed_days()
    pw = {"H1": C.power(IS["G0"] - IS["S0"], par, N_NEW, T_new, idx, alphas)}
    if h2:
        pw["H2"] = C.power(IS[astar] - IS["G0"], par, N_NEW, T_new, idx, alphas)
    daily = C.daily_rows(par, OUT, names)
    if daily2 is not None:
        daily = pd.concat([daily, daily2], ignore_index=True)
    gap = 0.0
    for lab in ("L3D1", "L0D0"):
        for m in REPORT:
            g = daily[(daily.variant == lab) & (daily.run == m)]
            gap = max(gap, abs(g.is_sum_c.sum() / g.n_parents.sum() - tabs[lab][m]["is_mean"]))
    out = {"astar": astar, "h2_exists": h2, "selection": sel, "sentence_no_h2": None if h2 else SENT_NONE,
           "hypotheses": ["H1"] + (["H2"] if h2 else []), "holm_levels": [C.ALPHA / 2, C.ALPHA] if h2 else [C.ALPHA],
           "prediction_setting": {"basis": C.parse(astar if h2 else sel["lowest_mean"])[0], "z_th": C.parse(astar if h2 else sel["lowest_mean"])[1],
                                  "from": "A*" if h2 else "the candidate with the lowest development mean (A* = G0)"},
           "tables": tabs, "prediction": {b: C.prediction_table(sd, b) for b in C.BASES},
           "diagnostics": {b: C.signal_counts(sd, b) for b in C.BASES}, "delay_check": dly, "control_flip": ctrl, "power": pw,
           "years": sorted({int(d[:4]) for d in days}),
           "sim": {**info, **pinfo, "days": len(days), "first_day": days[0], "last_day": days[-1], "sealed_days_for_power": T_new, "n_new": N_NEW},
           "checks": {"regression": reg, "second_pass_equal": second, "daily_file_max_gap_c": float(gap)},
           "daily": {"file": "results/dev22_daily.csv", "rows": int(len(daily)), "columns": list(daily.columns)},
           "candidates": list(C.CANDS), "trials_before": TRIALS_BEFORE, "trials_added": TRIALS_ADDED, "trials_cumulative": TRIALS_BEFORE + TRIALS_ADDED,
           "boot": {"B": E.BOOT_B, "block": E.BOOT_BLOCK, "seed": C.SEED}, "runtime_s": round(time.time() - t0, 1),
           "written": dt.datetime.now().isoformat(timespec="seconds")}
    text = report(out)
    summary = json.dumps(out, indent=1, default=C.jdefault, ensure_ascii=False)
    os.makedirs(RES, exist_ok=True)
    os.makedirs(LOCAL, exist_ok=True)
    f_par = os.path.join(LOCAL, "dev22_parents.npz")
    if not os.path.exists(f_par):
        np.savez_compressed(f_par, date=par.date.to_numpy().astype("U8"), sym=par.sym.to_numpy().astype("U8"), dir=par.dir.to_numpy(),
                            start=par.start.to_numpy(), mid2=par.mid2.to_numpy(), spread=par.spread.to_numpy(), out=OUT,
                            runs=np.array(names), lrules=np.array(E.LRULES))
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
    main(a.workers)
