#!/usr/bin/env python3
"""Research 21 step 2 development run (SPEC21 4, SPEC21_ADDENDUM C1-C2; step-2 prompt 3): families E and S separately on
the pairs of 2015-2020, market orders, fill assumptions L3D1 (primary) and L0D0. Development bars only (loader21); no
sealed file and no sealed or test folder of studies 12-20 is read. Run once: it refuses unless the synthetic tests
(results/engine21_tests.json) and the regression test (results/regress21.json) passed, and refuses to overwrite a summary.
  per family  9 signal settings (the 8 candidates W {10, 30} x z_in {2, 3} x kappa {1, 2} and W30_z2_k0): the M runs with
              mean daily P&L and day-block 95% interval (21 days, 10,000, seed 21), Sharpe, trades, win rate, holding
              time, pnl = mid - cost - fee per trade, by year and by pair; the mid table (a preview of H1); the delayed-mid
              table (1, 5, 30 s); W30_z2_k0 with market orders is the comparison run C2 (no candidate)
  A*          SPEC21 4 per family, mechanically (stats20.select_astar on the 8 candidates, L3D1); a mean <= 0 -> "no
              profit in development", that family has no H2
  control     SPEC21_ADDENDUM C1: mid P&L and delayed-mid table of the control pairs at W30_z2_k0
  diagnosis   share of signal points with sigma at the 1 bp floor; share of |z| >= z_in points kept out by kappa
  side by side research 20 (stored development results), E, S and the control pairs at W30_z2_k0, market orders
  power       MDE of every hypothesis left at the sealed length (one-sided, the first Holm level and 0.05, power 80%)
  delay       each family's A* once more with every order one second late
  daily file  results/dev21_daily.csv: one row per pair-day and setting (the 9 settings of each family, W30_z2_k0 for the
              control pairs) with the columns of run_test21's daily file (core21.DAILY_COLS; DEVIATIONS21 1.18). An output
              only: it is written from the arrays the tables are made of, and its sums are compared with the tables
  python dev21.py run [--workers 4]
Trade rows of each A* -> $DATA_DIR/research21_data/dev_results/, aggregates -> results/dev21_*."""
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
import core21 as C21  # noqa: E402
import loader21 as L21  # noqa: E402
import stats21 as ST21  # noqa: E402

C, S, ST20 = C21.C, C21.S, C21.ST20
RES = os.path.join(HERE, "results")
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research21_data/dev_results")
SEALED_WINDOW = ("20210104", "20261110")               # calendar days only (calendar21.csv); no sealed bar is read here
H1 = ST21.sig_name(ST21.H1_SIGNAL)
SIDE_LABELS = {"R20": "같은 회사 (연구 20)", "E": "ETF (가족 E)", "S": "같은 업종 (가족 S)", "C": "대조 쌍 (C1)"}


def sealed_days():
    return int(sum(SEALED_WINDOW[0] <= d <= SEALED_WINDOW[1] for d in L21.close_bars()))


def prerequisites():
    t = json.load(open(os.path.join(RES, "engine21_tests.json")))
    r = json.load(open(os.path.join(RES, "regress21.json")))
    if t["passed"] != t["total"] or not r["pass"]:
        raise SystemExit("the synthetic tests or the regression test did not pass")
    return {"synthetic_tests": f"{t['passed']}/{t['total']}", "research20_tests": t["research20_tests"], "regression_max_abs_diff": r["max_abs_diff"],
            "regression_numbers_compared": r["numbers_compared"]}


def daily_gap(daily, fam, control, n_days):
    """Largest difference between the daily file and the tables: |sum of a dollar column - total of the same run| and
    |sum of a trade column - trades of the same run|, over every family, setting and run (0 up to the rounding of sums:
    the file holds the numbers the tables are made of). Nothing is decided on it; it is recorded in the checks."""
    g = daily.groupby(["family", "setting"]).sum(numeric_only=True)
    usd, trd = 0.0, 0
    for f, name, blk in [(f, name, blk) for f, F in fam.items() for name, blk in F["settings"].items()] + [(L21.CONTROL, H1, control)]:
        x = g.loc[(f, name)]
        for k, r in blk["rows"].items():
            usd = max(usd, abs(float(x[k]) - r["total_usd"]))
            trd = max(trd, abs(int(x["trades_mid" if k == "mid" else "trades_M"]) - r["trades"]))
        for d in S.MID_DELAYS:
            usd = max(usd, abs(float(x[f"mid_delay{d}"]) - blk["delayed"][str(d)]["mean_daily_usd"] * n_days))
    return {"usd": float(usd), "trades": int(trd)}


def analyse(workers, check_prerequisites=True):
    t0 = time.time()
    if os.path.exists(os.path.join(RES, "dev21_summary.json")):
        raise SystemExit("results/dev21_summary.json exists: the development run is done once (nothing is overwritten)")
    pre = prerequisites() if check_prerequisites else {}
    tasks, days, skipped = L21.dev_plan()
    cfgs, cands = ST21.settings(), ST21.candidates()
    es = [t for t in tasks if t["family"] in ST21.FAMILIES]
    ct = [t for t in tasks if t["family"] == L21.CONTROL]
    A, info, ex = C21.simulate(es, L21.DEV_ROOT, False, cfgs, workers=workers, mid_delays=S.MID_DELAYS, diag=True)
    Ac, infoc, exc = C21.simulate(ct, L21.DEV_ROOT, False, [ST21.H1_SIGNAL], workers=workers, mid_delays=S.MID_DELAYS, mid_only=True)
    print(f"simulated {len(es):,} pair-days x {len(cfgs)} settings and {len(ct):,} control pair-days, {time.time() - t0:.0f} s", flush=True)
    if info["settings_with_m_trades_unlike_the_signal_timetable"]:
        with open(os.path.join(RES, "dev21_checks_FAILED.json"), "w") as fh:
            json.dump(info, fh, indent=1)
        raise SystemExit("the M trades differ from the signal-only timetable: results/dev21_checks_FAILED.json (no summary written)")
    n_days = len(days)
    idx = ST21.boot_index(n_days)
    fam, series, same, drows = {}, {}, {}, []
    os.makedirs(LOCAL, exist_ok=True)
    for f in ST21.FAMILIES:
        M, blocks, Af, Df, tf = C21.family_tables(A, ex["mid_delayed"], es, days, cfgs, idx, f)
        for ci, c in enumerate(cfgs):                      # rows of the daily file: every setting of the family (output only)
            drows += C21.pair_daily(Af, Df, tf, ci, ST21.sig_name(c))
        prim = [{"name": ST21.cand_name(c), "mean": blocks[ST21.sig_name(c)]["rows"]["M_L3D1"]["mean_daily_usd"],
                 "trades": blocks[ST21.sig_name(c)]["rows"]["M_L3D1"]["trades"]} for c in cands]
        astar, sel = ST21.select_astar(prim)
        a_cfg = cands[[p["name"] for p in prim].index(astar)]
        ci_a = cfgs.index(a_cfg)
        # delay check and the kept trades of A* (a second and a third pass over the family's pair-days)
        A1, _, _ = C21.simulate(tf, L21.DEV_ROOT, False, [a_cfg], delay=1, workers=workers)
        delay = {v: C.row(A1, M, 0, C21.KINDS.index(f"M_{v}"), n_days, idx, astar + "_delay1") for v in C21.VARS}
        A0, _, ex0 = C21.simulate(tf, L21.DEV_ROOT, False, [a_cfg], delay=0, keep=(0,), workers=workers)
        same[f] = bool(np.array_equal(A0[:, 0], Af[:, ci_a]))
        tp = os.path.join(LOCAL, f"dev21_trades_{f}_{ST21.sig_name(a_cfg)}.npz")
        if not os.path.exists(tp):
            np.savez_compressed(tp, M=ex0["kept"][0], dates=np.array([t["date"] for t in tf]), pairs=np.array([t["name"] for t in tf]))
        series[f"H1-{f}"] = (C.daily(Af, M, cfgs.index(ST21.H1_SIGNAL), 0, "pnl", n_days), f"daily sum of mid P&L, family {f}, setting {H1}")
        series[f"H2-{f}"] = (C.daily(Af, M, ci_a, 1, "pnl", n_days), f"daily P&L of {astar}, family {f}, L3D1")
        h2 = bool(sel["profitable_in_development"])
        fam[f] = {"x": ST21.FAMILY_X[f], "astar": astar, "astar_signal": list(a_cfg), "selection": sel, "h2_exists": h2,
                  "q2": f"A* = {astar}" if h2 else f"개발에서 수익 없음 (H2-{f} 없음)", "settings": blocks, "delay_check": delay,
                  "diag": C21.diag_shares(ex["diag"][C21.family_index(es, f)], M, cfgs), "pair_days": len(tf), "pairs": M["names"],
                  "skipped_pair_days": skipped[f], "trades_file": tp.replace(os.path.expanduser("~"), "~")}
    Mc, cb, Acf, Dcf, tcf = C21.family_tables(Ac, exc["mid_delayed"], ct, days, [ST21.H1_SIGNAL], idx, L21.CONTROL, mid_only=True)
    years = sorted({t["year"] for t in tasks})
    control = {"pairs_by_year": {str(y): [f"{a}/{b}" for _, a, b in L21.controls(y)] for y in years}, "pair_days": len(ct), "pairs": Mc["names"],
               "skipped_pair_days": skipped[L21.CONTROL], **cb[H1]}
    drows += C21.pair_daily(Acf, Dcf, tcf, 0, H1)
    daily = pd.DataFrame(drows, columns=C21.DAILY_COLS)
    daily["trades_M"] = daily["trades_M"].astype("Int64")               # empty for the control pairs
    h2 = {f: fam[f]["h2_exists"] for f in ST21.FAMILIES}
    hyps = ST21.hypotheses(h2)
    alphas = (ST21.strictest_level(len(hyps)), ST21.ALPHA)
    t_new = sealed_days()
    power = {}
    for h in hyps:
        d, label = series[h]
        p = ST20.mde(d, idx, t_new, alphas=alphas)
        p["mde"] = {"first_holm_level": p["mde"][str(alphas[0])], "0.05": p["mde"][str(alphas[1])]}
        p["dev_effect_usd"] = float(d.mean())
        p["underpowered"] = {k: bool(not m <= p["dev_effect_usd"]) for k, m in p["mde"].items()}
        p["series"] = label
        power[h] = p
    side = [dict(C21.r20_side("dev"), key="R20")] + [dict(C21.side_row(f, fam[f]["settings"][H1]), key=f) for f in ST21.FAMILIES]
    side.append(dict(C21.side_row(L21.CONTROL, cb[H1]), key=L21.CONTROL))
    for r in side:
        r["label"] = SIDE_LABELS[r["key"]] + (", 개발 2015–2020" if r["key"] == "R20" else "")
    last = sum(r["exit_last"] for F in fam.values() for b in F["settings"].values() for r in b["rows"].values()) + control["rows"]["mid"]["exit_last"]
    out = {"families": fam, "control": control, "hypotheses": hyps, "holm_levels": {"first": alphas[0], "last": alphas[1]}, "power": power,
           "side_by_side": side, "h1_signal": list(ST21.H1_SIGNAL), "candidates": [ST21.cand_name(c) for c in cands],
           "checks": {**pre, "m_trades_equal_signal_timetable": True, "second_pass_equal": same,
                      "trades_valued_at_last_valid_mids": int(last),
                      "pair_days_left_out_no_state_after_first_state": info["pair_days_left_out_no_state_after_first_state"]
                      + infoc["pair_days_left_out_no_state_after_first_state"],
                      "valid_grid_share": info["valid_grid_points"] / max(info["grid_points"], 1),
                      "valid_grid_share_control": infoc["valid_grid_points"] / max(infoc["grid_points"], 1),
                      "daily_file_max_gap_to_tables": daily_gap(daily, fam, control, n_days)},
           "daily": {"file": "results/dev21_daily.csv", "rows": len(daily), "columns": list(C21.DAILY_COLS)},
           "sim": {"pair_days": len(es), "control_pair_days": len(ct), "days": n_days, "first_day": days[0], "last_day": days[-1],
                   "sealed_calendar_days": t_new, "skipped_pair_days": skipped, "workers": workers},
           "trials_before": ST21.TRIALS_BEFORE, "trials_added": ST21.TRIALS_ADDED, "trials_cumulative": ST21.TRIALS_BEFORE + ST21.TRIALS_ADDED,
           "boot": {"B": ST21.BOOT_B, "block": ST21.BOOT_BLOCK, "seed": ST21.BOOT_SEED},
           "runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")}
    if not all(same.values()):
        with open(os.path.join(RES, "dev21_checks_FAILED.json"), "w") as fh:
            json.dump(out["checks"], fh, indent=1, default=str)
        raise SystemExit("a second pass differs from the first: results/dev21_checks_FAILED.json (no summary written)")
    f_day = os.path.join(RES, "dev21_daily.csv")
    C21.keep_old(f_day)                                                 # a file left by a run that stopped before its summary is kept
    daily.to_csv(f_day, index=False, float_format="%.4f")
    with open(os.path.join(RES, "dev21_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str, ensure_ascii=False)
    flat = [{"family": f, "setting": name, "run": k} | {a: b for a, b in r.items() if a not in ("by_year", "by_pair", "per_trade", "name", "kind")}
            | {f"y{y}": v for y, v in r["by_year"].items()} for f, F in fam.items() for name, blk in F["settings"].items() for k, r in blk["rows"].items()]
    pd.DataFrame(flat).to_csv(os.path.join(RES, "dev21_rows.csv"), index=False, float_format="%.6f")
    report(out)
    print(open(os.path.join(RES, "dev21_report.md")).read())
    return out


# ---------------------------------------------------------------- report
def fmt_ci(r):
    return f"{r['mean_daily_usd']:+,.1f} [{r['ci_lo']:+,.1f}, {r['ci_hi']:+,.1f}]"


def pct(x):
    return "—" if x is None or x != x else f"{100 * x:.1f}%"


def run_line(label, r):
    return (f"| {label} | {fmt_ci(r)} | {r['sharpe']:+.2f} | {r['trades']:,} | {r['trades_per_day']:.1f} | {r['win_rate']:.3f} | "
            f"{r['mean_hold_s']:.0f} | {r['pnl_per_trade']:+.2f} | {r['mid_per_trade']:+.2f} | {r['cost_per_trade']:.2f} | {r['fee_per_trade']:.2f} |")


def mid_lines(blocks):
    L_ = ["| 설정 | 거래 수 | 거래당 중간값 손익 [95% CI] | 일 합 평균 [95% CI] | 단측 p | 승률 | 보유(초) |", "|---|---|---|---|---|---|---|"]
    for name, b in blocks.items():
        m, pt = b["rows"]["mid"], b["rows"]["mid"]["per_trade"]
        L_.append(f"| {name} | {m['trades']:,} | {pt['mean']:+.2f} [{pt['lo']:+.2f}, {pt['hi']:+.2f}] | {fmt_ci(m)} | {m['p_one_sided']:.4f} | "
                  f"{m['win_rate']:.3f} | {m['mean_hold_s']:.0f} |")
    return L_


def delay_lines(blocks):
    L_ = ["| 설정 | 0초 거래당 | 1초 거래당 (0초 대비) | 5초 거래당 (0초 대비) | 5초 일 합 평균 [95% CI] | 30초 거래당 (0초 대비) |", "|---|---|---|---|---|---|"]
    for name, b in blocks.items():
        d = b["delayed"]
        L_.append(f"| {name} | {d['0']['per_trade_usd']:+.2f} | {d['1']['per_trade_usd']:+.2f} ({pct(d['1']['ratio_to_delay0'])}) | "
                  f"{d['5']['per_trade_usd']:+.2f} ({pct(d['5']['ratio_to_delay0'])}) | {fmt_ci(d['5'])} | "
                  f"{d['30']['per_trade_usd']:+.2f} ({pct(d['30']['ratio_to_delay0'])}) |")
    return L_


def side_lines(side):
    def cell(x, k, f):
        return "—" if x is None else f.format(x[k])
    L_ = ["| 쌍의 종류 | 거래당 중간값 손익 | 5초 지연 뒤 남는 비율 | 거래당 비용 L3D1 | 비용 차감 후 L3D1 | 거래당 비용 L0D0 | 비용 차감 후 L0D0 | 하루 평균 거래 수 |",
          "|---|---|---|---|---|---|---|---|"]
    for r in side:
        L_.append(f"| {r['label']} | {r['mid_per_trade']:+.2f} | {pct(r['ratio_5s'])} | {cell(r['L3D1'], 'cost_per_trade', '{:.2f}')} | "
                  f"{cell(r['L3D1'], 'pnl_per_trade', '{:+.2f}')} | {cell(r['L0D0'], 'cost_per_trade', '{:.2f}')} | "
                  f"{cell(r['L0D0'], 'pnl_per_trade', '{:+.2f}')} | {r['trades_per_day']:.1f} |")
    return L_


def family_lines(f, F, cand_names):
    B = F["settings"]
    names = [n[:-2] for n in cand_names]
    yrs = sorted(B[H1]["rows"]["mid"]["by_year"])
    L_ = [f"# 가족 {f}: {F['x']}", "", f"- 쌍·일 {F['pair_days']:,}개. 쌍: {', '.join(F['pairs'])}",
          f"- 한쪽 다리만 있는 날 {F['skipped_pair_days']['one_leg']}일과 두 다리 모두 없는 날 {F['skipped_pair_days']['neither']}일은 쌍의 거래일에서 뺐다.",
          "", f"**{F['q2']}**", ""]
    for v in C21.VARS:
        L_ += [f"## 가족 {f} · {v}: 후보 8개와 비교용 실행", "",
               "| 후보 | 평균 일 손익 [95% CI] | 샤프 | 거래 수 | 하루 | 승률 | 보유(초) | 거래당 손익 | = 중간값 | − 비용 | − 수수료 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        L_ += [run_line(n + "_M", B[n]["rows"][f"M_{v}"]) for n in names]
        L_ += [run_line(H1 + "_M (비교용 C2, 후보 아님)", B[H1]["rows"][f"M_{v}"]), ""]
    L_ += [f"## 가족 {f} · 중간값 손익 표 (신호만의 시간표, 비용 0; H1 미리 보기는 {H1})", ""] + mid_lines(B)
    L_ += ["", f"## 가족 {f} · 지연 중간값 손익 (SPEC20_ADDENDUM B2의 정의, 판정 아님)", ""] + delay_lines(B)
    L_ += ["", f"- {H1}의 5초 문장(개발 구간, 참고): \"{B[H1]['sentence_5s']}\""]
    for v in C21.VARS:
        L_ += ["", f"## 가족 {f} · {v}: 연도별 평균 일 손익", "", "| 후보 | " + " | ".join(yrs) + " |", "|---" * (len(yrs) + 1) + "|"]
        L_ += [f"| {n}_M | " + " | ".join(f"{B[n]['rows'][f'M_{v}']['by_year'][y]:+,.1f}" for y in yrs) + " |" for n in names + [H1]]
    pn = F["pairs"]
    L_ += ["", f"## 가족 {f} · L3D1: 쌍별 손익 (쌍·일당 달러)", "", "| 후보 | " + " | ".join(pn) + " |", "|---" * (len(pn) + 1) + "|"]
    L_ += [f"| {n}_M | " + " | ".join(f"{B[n]['rows']['M_L3D1']['by_pair'][p]['per_pair_day_usd']:+,.1f}" for p in pn) + " |" for n in names + [H1]]
    L_.append("| 쌍·일 수 | " + " | ".join(f"{B[H1]['rows']['mid']['by_pair'][p]['pair_days']:,}" for p in pn) + " |")
    dg = F["diag"]
    L_ += ["", f"## 가족 {f} · 진단 (진입 시간대의 신호가 있는 격자 시점 기준, 규칙은 바꾸지 않는다)", "",
           "| 범위 | σ 하한 적용 W10 | σ 하한 적용 W30 | " + " | ".join(f"κ에 막힘 {n}" for n in names) + " |", "|---" * (3 + len(names)) + "|"]
    for label, g in [("가족 전체", dg["family"])] + list(dg["by_pair"].items()):
        L_.append(f"| {label} | {pct(g['sigma_floor']['W10']['share'])} | {pct(g['sigma_floor']['W30']['share'])} | "
                  + " | ".join(pct(g["kappa_blocked"][n]["share"]) for n in names) + " |")
    L_ += ["- σ 하한 적용: 직전 W분의 표준편차가 1bp보다 작아 1bp로 둔 시점의 비율이다.",
           "- κ에 막힘: |z| ≥ z_in인 시점 가운데 |s − m| < κ × (두 다리의 상대 스프레드 합)이라 진입 표시가 서지 않은 비율이다."]
    sel = F["selection"]
    L_ += ["", f"## 가족 {f} · A* 선택 (SPEC21 4절, L3D1)", "",
           f"- 평균 일 손익 최고: {sel['highest_mean']} ({sel['highest_mean_usd']:+,.2f}달러)",
           f"- 5% 안 동률: {', '.join(sel['tied_within_5pct'])} (거래 수 {sel['trades_of_tied']})",
           f"- **A* = {sel['astar']}** ({sel['astar_mean_usd']:+,.2f}달러/일) → "
           + (f"개발에서 수익이 있다. H2-{f}를 둔다." if F["h2_exists"] else f"0 이하이므로 **\"개발에서 수익 없음\"**이다. H2-{f}를 없앤다."),
           "", "| 지연 점검 (참고: 신호 t, 주문 t + 1초) | A* | 지연 1초 | 거래 수 |", "|---|---|---|---|"]
    a_sig = ST21.sig_name(tuple(F["astar_signal"]))
    for v in C21.VARS:
        d = F["delay_check"][v]
        L_.append(f"| {v} | {fmt_ci(B[a_sig]['rows'][f'M_{v}'])} | {fmt_ci(d)} | {d['trades']:,} |")
    return L_ + [""]


def report(out):
    s, ctl = out["sim"], out["control"]
    L_ = ["# 연구 21 · 2단계 개발 실행 (쌍 2015–2020, 가족마다 후보 8개)", "",
          f"- 쌍·일 {s['pair_days']:,}개(가족 E·S)와 대조 쌍·일 {s['control_pair_days']:,}개, {s['days']:,}일({s['first_day']} ~ {s['last_day']}). 봉인 데이터는 읽지 않았다.",
          f"- CI: 일 블록 {out['boot']['block']}, {out['boot']['B']:,}회, seed {out['boot']['seed']}. 손익 단위는 달러(다리마다 5만 달러). "
          f"시행 수 +{out['trials_added']}, 누적 {out['trials_cumulative']}.",
          "- 후보 이름: W(분)_z(z_in)_k(κ)_M(시장가). 가족의 일 손익은 그 가족 쌍들의 손익 합이다.", ""]
    L_ += ["| 가족 | 결과 |", "|---|---|"] + [f"| {f} ({F['x']}) | {F['q2']} |" for f, F in out["families"].items()]
    L_ += ["", f"- 남은 가설: {', '.join(out['hypotheses'])} (Holm 첫 수준 {out['holm_levels']['first']:.4f})", ""]
    L_ += [f"## 나란히 놓은 표 ({H1}, 시장가)", ""] + side_lines(out["side_by_side"])
    L_ += ["- 거래당 비용은 스프레드·체결 비용과 수수료의 합이다. 비용 차감 후 = 중간값 손익 − 비용.",
           "- 대조 쌍은 추가 문서 C1대로 중간값 손익과 지연 표만 계산했다. 그래서 비용 칸이 비어 있다.", ""]
    for f, F in out["families"].items():
        L_ += family_lines(f, F, out["candidates"])
    L_ += ["# 대조 쌍 (SPEC21_ADDENDUM C1, 판정 아님)", "", "| 연도 | 대조 쌍 |", "|---|---|"]
    L_ += [f"| {y} | {', '.join(p) if p else '없음'} |" for y, p in ctl["pairs_by_year"].items()]
    L_ += ["", f"- 쌍·일 {ctl['pair_days']:,}개. 한쪽 다리만 있는 날 {ctl['skipped_pair_days']['one_leg']}일, 둘 다 없는 날 {ctl['skipped_pair_days']['neither']}일.", ""]
    L_ += mid_lines({H1: ctl}) + [""] + delay_lines({H1: ctl}) + ["", f"- 5초 문장(참고): \"{ctl['sentence_5s']}\"", ""]
    L_ += [f"# 검정력 (봉인 구간 {s['sealed_calendar_days']:,}일 기준, 단측, 검정력 80%)", "",
           f"| 가설 | 계열 | 개발 효과 (달러/일) | MDE α {out['holm_levels']['first']:.4f} | MDE α 0.05 | VIF | 판정 |", "|---|---|---|---|---|---|---|"]
    for h, p in out["power"].items():
        u = p["underpowered"]
        flag = "검정력 부족" if u["first_holm_level"] and u["0.05"] else ("검정력 충분" if not u["first_holm_level"] else "Holm 첫 수준에서만 부족")
        L_.append(f"| {h} | {p['series']} | {p['dev_effect_usd']:+,.1f} | {p['mde']['first_holm_level']:,.1f} | {p['mde']['0.05']:,.1f} | {p['VIF']:.2f} | {flag} |")
    c = out["checks"]
    L_ += ["", "# 점검", ""]
    if "synthetic_tests" in c:
        L_.append(f"- 합성 테스트 {c['synthetic_tests']} 통과(연구 20의 테스트 {c['research20_tests']} 포함). 회귀 테스트: 숫자 {c['regression_numbers_compared']:,}개, "
                  f"최대 차이 {max(c['regression_max_abs_diff'].values()):.1e}.")
    L_ += [f"- M의 거래가 신호만의 시간표와 같음: {c['m_trades_equal_signal_timetable']}. 두 번째 실행이 첫 실행과 같음: {c['second_pass_equal']}.",
           f"- 유효 격자 시점 비율 {c['valid_grid_share']:.4f}(대조 {c['valid_grid_share_control']:.4f}). 마지막 유효 중간값으로 평가한 거래 "
           f"{c['trades_valued_at_last_valid_mids']}건. 자료 이상으로 뺀 쌍·일 {c['pair_days_left_out_no_state_after_first_state']}개.",
           f"- 쌍별 일 손익 파일: {out['daily']['file']} ({out['daily']['rows']:,}줄, 쌍·일과 설정마다 한 줄). 표의 합계와 가장 큰 차이 "
           f"{c['daily_file_max_gap_to_tables']['usd']:.1e}달러, 거래 수 차이 {c['daily_file_max_gap_to_tables']['trades']}건.",
           f"- 실행 {out['runtime_s']} s (작업자 {s['workers']}개)"]
    with open(os.path.join(RES, "dev21_report.md"), "w") as fh:
        fh.write("\n".join(L_) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--workers", type=int, default=4)
    analyse(ap.parse_args().workers)
