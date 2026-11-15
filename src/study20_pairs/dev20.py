#!/usr/bin/env python3
"""Research 20 step 2 development run (SPEC20 4, SPEC20_ANSWERS A2-A4; step-2 prompt 2): the 16 candidates
W {10, 30} x z_in {2, 3} x kappa {0, 1} x execution {M, P} on the pairs of 2015-2020, fill assumptions L3D1 (primary) and
L0D0. Development bars only (loader20); no sealed file and no sealed or test folder of studies 12-19 is read.
Refuses to run unless the synthetic tests (results/engine20_tests.json) and the regression test (results/regress20.json)
passed, and refuses to overwrite an existing summary.
  per run     mean daily P&L with the day-block 95% interval (21 days, 10,000, seed 20), Sharpe, trades, win rate, holding
              time, pnl = mid - cost - fee per trade, P fill statistics, by year and by pair
  mid table   the 8 signal settings on the signal-only timetable (a preview of H1)
  A*          SPEC20 4, mechanically (stats20.select_astar); mean <= 0 -> no profit in development, no H2
  power       MDE at the sealed length from the development variance (one-sided 0.025 and 0.05, power 80%)
  delay       A* once more with every order one second late
  python dev20.py run [--workers 8]
Trade rows of the A* setting -> $DATA_DIR/research20_data/dev_results/, aggregates -> results/dev20_*."""
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
import core20 as C  # noqa: E402
import loader20 as L  # noqa: E402
import stats20 as ST  # noqa: E402

RES = os.path.join(HERE, "results")
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research20_data/dev_results")
SEALED_WINDOW = ("20210104", "20261231")               # calendar days only (calendar20.csv); no sealed bar is read here
TRIALS_BEFORE, TRIALS_ADDED = 777, 16
VARS = ("L3D1", "L0D0")


def sealed_days():
    return int(sum(SEALED_WINDOW[0] <= d <= SEALED_WINDOW[1] for d in L.close_bars()))


def prerequisites():
    t = json.load(open(os.path.join(RES, "engine20_tests.json")))
    r = json.load(open(os.path.join(RES, "regress20.json")))
    if t["passed"] != t["total"] or not r["pass"]:
        raise SystemExit("the synthetic tests or the regression test did not pass")
    return {"synthetic_tests": f"{t['passed']}/{t['total']}", "regression": r["mean_is_3dp"], "regression_parents": r["parents"]}


def table(A, M, cfgs, n_days, idx):
    rows, mids = [], []
    for ci, c in enumerate(cfgs):
        m = C.row(A, M, ci, 0, n_days, idx, ST.sig_name(c) + "_mid")
        m["per_trade"] = dict(zip(("mean", "lo", "hi"), C.per_trade(A, M, ci, 0, "pnl", n_days, idx)))
        m.update({"W": c[0], "z_in": c[1], "kappa": c[2]})
        mids.append(m)
        for ex, _ in ST.EXECS:
            for v in VARS:
                r = C.row(A, M, ci, C.KINDS.index(f"{ex}_{v}"), n_days, idx, ST.cand_name(c, ex))
                r.update({"variant": v, "W": c[0], "z_in": c[1], "kappa": c[2], "exec": ex})
                rows.append(r)
    return rows, mids


def analyse(workers):
    t0 = time.time()
    if os.path.exists(os.path.join(RES, "dev20_summary.json")):
        raise SystemExit("results/dev20_summary.json exists: rename it before another run (nothing is overwritten)")
    pre = prerequisites()
    tasks, days, skipped = L.dev_plan()
    cfgs = ST.configs()
    A, info, _ = C.simulate(tasks, L.DEV_ROOT, False, cfgs, delay=0, workers=workers)
    print(f"simulated {len(tasks):,} pair-days x {len(cfgs)} settings, {time.time() - t0:.0f} s", flush=True)
    if info["settings_with_m_trades_unlike_the_signal_timetable"]:
        with open(os.path.join(RES, "dev20_checks_FAILED.json"), "w") as fh:
            json.dump(info, fh, indent=1)
        raise SystemExit("the M trades differ from the signal-only timetable: results/dev20_checks_FAILED.json (no summary written)")
    M = C.meta(tasks, days)
    n_days = len(days)
    idx = ST.boot_index(n_days)
    rows, mids = table(A, M, cfgs, n_days, idx)
    prim = [{"name": r["name"], "mean": r["mean_daily_usd"], "trades": r["trades"]} for r in rows if r["variant"] == "L3D1"]
    astar, sel = ST.select_astar(prim)
    ar = next(r for r in rows if r["name"] == astar and r["variant"] == "L3D1")
    a_cfg, a_ex = (ar["W"], ar["z_in"], ar["kappa"]), ar["exec"]
    h2 = bool(sel["profitable_in_development"])
    h1_cfg = a_cfg if h2 else ST.DEFAULT_SIGNAL
    ci_a, ci_h1 = cfgs.index(a_cfg), cfgs.index(h1_cfg)
    t_new = sealed_days()
    d_h2 = C.daily(A, M, ci_a, C.KINDS.index(f"{a_ex}_L3D1"), "pnl", n_days)
    d_h1 = C.daily(A, M, ci_h1, 0, "pnl", n_days)
    power = {}
    for h, d in (("H1", d_h1), ("H2", d_h2)):
        p = ST.mde(d, idx, t_new)
        p["dev_effect_usd"] = float(d.mean())
        p["underpowered"] = {a: bool(m > p["dev_effect_usd"]) for a, m in p["mde"].items()}
        power[h] = p
    power["H1"]["series"] = f"daily sum of mid P&L, setting {ST.sig_name(h1_cfg)}"
    power["H2"]["series"] = f"daily P&L of {astar}, L3D1" + ("" if h2 else " (no H2: shown for reference only)")
    # delay check and the kept trades of the A* setting (a second and third pass over the same pair-days)
    A1, info1, _ = C.simulate(tasks, L.DEV_ROOT, False, [a_cfg], delay=1, workers=workers)
    delay = {v: C.row(A1, M, 0, C.KINDS.index(f"{a_ex}_{v}"), n_days, idx, astar + "_delay1") for v in VARS}
    A0, info0, kept = C.simulate(tasks, L.DEV_ROOT, False, [a_cfg], delay=0, keep=(0,), workers=workers)
    same = bool(np.array_equal(A0[:, 0], A[:, ci_a]))
    os.makedirs(LOCAL, exist_ok=True)
    tp = os.path.join(LOCAL, f"dev20_trades_{ST.sig_name(a_cfg)}.npz")
    if not os.path.exists(tp):
        np.savez_compressed(tp, M=kept[0]["M"], P_L3=kept[0]["P_L3"], P_L0=kept[0]["P_L0"], dates=np.array([t["date"] for t in tasks]),
                            pairs=np.array([t["name"] for t in tasks]))
    out = {"astar": astar, "astar_signal": list(a_cfg), "astar_exec": a_ex, "selection": sel, "h2_exists": h2,
           "h1_signal": list(h1_cfg), "q2": f"A* = {astar}" if h2 else "개발에서 수익 없음 (H2 없음, H1만)",
           "power": power, "delay_check": delay, "rows": rows, "mid_rows": mids,
           "checks": {**pre, "m_trades_equal_signal_timetable": True, "second_pass_equal": same,
                      "safeguard_exits_last_grid_point": int(sum(r["exit_last"] for r in rows)),
                      "valid_grid_share": info["valid_grid_points"] / info["grid_points"]},
           "sim": {**info, "days": n_days, "first_day": days[0], "last_day": days[-1], "skipped_pair_days": skipped,
                   "sealed_calendar_days": t_new, "pairs": M["names"]},
           "trials_before": TRIALS_BEFORE, "trials_added": TRIALS_ADDED, "trials_cumulative": TRIALS_BEFORE + TRIALS_ADDED,
           "boot": {"B": ST.BOOT_B, "block": ST.BOOT_BLOCK, "seed": ST.BOOT_SEED}, "trades_file": tp.replace(os.path.expanduser("~"), "~"),
           "runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")}
    if not same:
        with open(os.path.join(RES, "dev20_checks_FAILED.json"), "w") as fh:
            json.dump(out["checks"], fh, indent=1, default=str)
        raise SystemExit("the second pass differs from the first: results/dev20_checks_FAILED.json (no summary written)")
    with open(os.path.join(RES, "dev20_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str, ensure_ascii=False)
    flat = [{k: v for k, v in r.items() if k not in ("by_year", "by_pair")} | {f"y{y}": v for y, v in r["by_year"].items()}
            | {f"pair_{p}": v["per_pair_day_usd"] for p, v in r["by_pair"].items()} for r in rows]
    pd.DataFrame(flat).to_csv(os.path.join(RES, "dev20_rows.csv"), index=False, float_format="%.6f")
    report(out)
    print(open(os.path.join(RES, "dev20_report.md")).read())


def fmt_ci(r, k="mean_daily_usd"):
    return f"{r[k]:+,.1f} [{r['ci_lo']:+,.1f}, {r['ci_hi']:+,.1f}]"


def report(out):
    R, s, n_days = out["rows"], out["sim"], out["sim"]["days"]
    yrs = sorted(R[0]["by_year"])
    L_ = ["# 연구 20 · 2단계 개발 실행 (쌍 2015–2020, 후보 16개)", "",
          f"- 쌍·일 {s['pair_days']:,}개, {n_days:,}일({s['first_day']} ~ {s['last_day']}). 봉인 데이터는 읽지 않았다.",
          f"- CI: 일 블록 21, 10,000회, seed 20. 손익 단위는 달러(다리마다 5만 달러). 시행 수 +{out['trials_added']}, 누적 {out['trials_cumulative']}.",
          "- 후보 이름: W(분)_z(z_in)_k(κ)_실행(M 시장가, P 지정가 후 30초 뒤 시장가).", "", f"**{out['q2']}**", ""]
    for v in VARS:
        L_ += [f"## {v}: 후보 16개", "",
               "| 후보 | 평균 일 손익 [95% CI] | 샤프 | 거래 수 | 하루 | 승률 | 보유(초) | 거래당 손익 | = 중간값 | − 비용 | − 수수료 | 한쪽만 지정가 | 기한 시장가 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in (x for x in R if x["variant"] == v):
            pf = (f"{r['p_one_leg_limit_frac']:.3f} | {r['p_deadline_order_frac']:.3f}" if r["exec"] == "P" else "— | —")
            L_.append(f"| {r['name']} | {fmt_ci(r)} | {r['sharpe']:+.2f} | {r['trades']:,} | {r['trades_per_day']:.1f} | {r['win_rate']:.3f} | "
                      f"{r['mean_hold_s']:.0f} | {r['pnl_per_trade']:+.2f} | {r['mid_per_trade']:+.2f} | {r['cost_per_trade']:.2f} | "
                      f"{r['fee_per_trade']:.2f} | {pf} |")
        L_.append("")
    L_ += ["## 중간값 손익 표 (신호 설정 8개, 신호만의 시간표, 비용 0; H1 미리 보기)", "",
           "| 설정 | 거래 수 | 거래당 중간값 손익 [95% CI] | 평균 일 합 [95% CI] | 단측 p | 승률 | 보유(초) | z 청산 | 시간 청산 | 15:55 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for m in out["mid_rows"]:
        pt = m["per_trade"]
        L_.append(f"| {m['name'][:-4]} | {m['trades']:,} | {pt['mean']:+.2f} [{pt['lo']:+.2f}, {pt['hi']:+.2f}] | {fmt_ci(m)} | {m['p_one_sided']:.4f} | "
                  f"{m['win_rate']:.3f} | {m['mean_hold_s']:.0f} | {m['exit_z']:.3f} | {m['exit_time']:.3f} | {m['exit_close']:.3f} |")
    for v in VARS:
        L_ += ["", f"## {v}: 연도별 평균 일 손익", "", "| 후보 | " + " | ".join(yrs) + " |", "|---" * (len(yrs) + 1) + "|"]
        for r in (x for x in R if x["variant"] == v):
            L_.append(f"| {r['name']} | " + " | ".join(f"{r['by_year'][y]:+,.1f}" for y in yrs) + " |")
    pn = s["pairs"]
    L_ += ["", "## L3D1: 쌍별 손익 (쌍·일당 달러)", "", "| 후보 | " + " | ".join(pn) + " |", "|---" * (len(pn) + 1) + "|"]
    for r in (x for x in R if x["variant"] == "L3D1"):
        L_.append(f"| {r['name']} | " + " | ".join(f"{r['by_pair'][p]['per_pair_day_usd']:+,.1f}" for p in pn) + " |")
    L_.append("| 쌍·일 수 | " + " | ".join(f"{R[0]['by_pair'][p]['pair_days']:,}" for p in pn) + " |")
    L_.append("- FOXA/FOX는 2015–2019-03의 21세기 폭스와 2020년의 새 Fox Corp를 같은 이름으로 묶은 값이다.")
    sel = out["selection"]
    L_ += ["", "## A* 선택 (SPEC20 4절, L3D1)", "",
           f"- 평균 일 손익 최고: {sel['highest_mean']} ({sel['highest_mean_usd']:+,.2f}달러)",
           f"- 5% 안 동률: {', '.join(sel['tied_within_5pct'])} (거래 수 {sel['trades_of_tied']})",
           f"- **A* = {sel['astar']}** ({sel['astar_mean_usd']:+,.2f}달러/일) → "
           + ("개발에서 수익이 있다. H2를 둔다." if out["h2_exists"] else "0 이하이므로 **\"개발에서 수익 없음\"**이다. H2를 없애고 봉인 테스트는 H1만 한다."),
           f"- H1의 신호 설정: W = {out['h1_signal'][0]}, z_in = {out['h1_signal'][1]:g}, κ = {out['h1_signal'][2]:g}"
           + ("" if out["h2_exists"] else " (기본값)"), "", "## 검정력 (봉인 구간 " + f"{s['sealed_calendar_days']:,}일 기준, 단측, 검정력 80%)", "",
           "| 가설 | 계열 | 개발 효과 (달러/일) | MDE α 0.025 | MDE α 0.05 | VIF | 판정 |", "|---|---|---|---|---|---|---|"]
    for h in ("H1", "H2"):
        p = out["power"][h]
        flag = "검정력 부족" if p["underpowered"]["0.025"] else ("검정력 충분" if not p["underpowered"]["0.05"] else "α 0.05에서만 부족")
        L_.append(f"| {h} | {p['series']} | {p['dev_effect_usd']:+,.1f} | {p['mde']['0.025']:,.1f} | {p['mde']['0.05']:,.1f} | {p['VIF']:.2f} | {flag} |")
    L_ += ["", "## 지연 점검 (참고: 신호 t, 주문 t + 1초)", "", "| 가정 | A* | 평균 일 손익 [95% CI] | 지연 1초 | 거래 수 |", "|---|---|---|---|---|"]
    for v in VARS:
        base = next(r for r in R if r["name"] == out["astar"] and r["variant"] == v)
        d = out["delay_check"][v]
        L_.append(f"| {v} | {out['astar']} | {fmt_ci(base)} | {fmt_ci(d)} | {d['trades']:,} |")
    c = out["checks"]
    L_ += ["", "## 점검", "", f"- 합성 테스트 {c['synthetic_tests']} 통과. 회귀 테스트(주문 {c['regression_parents']:,}건): "
           + ", ".join(f"{k} {v['got']:.3f}" for k, v in c["regression"].items()),
           f"- M의 거래가 신호만의 시간표와 같음: {c['m_trades_equal_signal_timetable']}. 두 번째 실행이 첫 실행과 같음: {c['second_pass_equal']}.",
           f"- 유효 격자 시점 비율 {c['valid_grid_share']:.4f}. 마지막 격자 시점의 안전장치 청산 {c['safeguard_exits_last_grid_point']}건.",
           f"- 한쪽 다리만 있는 날 {s['skipped_pair_days']['one_leg']}일과 두 다리 모두 없는 날 {s['skipped_pair_days']['neither']}일은 쌍의 거래일에서 뺐다.",
           f"- 실행 {out['runtime_s']} s"]
    with open(os.path.join(RES, "dev20_report.md"), "w") as fh:
        fh.write("\n".join(L_) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--workers", type=int, default=8)
    analyse(ap.parse_args().workers)
