#!/usr/bin/env python3
"""Research 20 step 2b (SPEC20_ADDENDUM B2): the mid P&L of the signal-only trades when both prices are taken d seconds
after the decisions, d in {1, 5, 30}, for the 8 signal settings on the development data (pairs of 2015-2020). No verdict
and no candidate choice: the trial count stays at 793. Development bars only (loader20); no sealed file is read.
The same pass rebuilds every table row of the development run with the step-2b code and compares it with
results/dev20_summary.json (the step-2 numbers must not move).
Writes results/delay20_summary.json and results/delay20_report.md; refuses to overwrite.
  python delay20.py run [--workers 8]"""
import argparse
import datetime as dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core20 as C  # noqa: E402
import dev20 as DV  # noqa: E402
import loader20 as L  # noqa: E402
import signal20 as S  # noqa: E402
import stats20 as ST  # noqa: E402

RES = os.path.join(HERE, "results")
NUMERIC = ("mean_daily_usd", "ci_lo", "ci_hi", "p_one_sided", "sharpe", "trades", "win_rate", "mean_hold_s", "pnl_per_trade",
           "mid_per_trade", "cost_per_trade", "fee_per_trade", "total_usd", "exit_last")


def compare(rows, mids, dev):
    """Largest absolute difference between the rebuilt rows and the stored development rows, and the number of rows."""
    diffs, n = [], 0
    for new, old in ((rows, dev["rows"]), (mids, dev["mid_rows"])):
        ref = {(r["name"], r.get("variant")): r for r in old}
        for r in new:
            d = ref[(r["name"], r.get("variant"))]
            diffs += [abs(r[f] - d[f]) for f in NUMERIC] + [abs(r["by_year"][y] - d["by_year"][y]) for y in d["by_year"]]
            diffs += [abs(r["by_pair"][p]["per_pair_day_usd"] - d["by_pair"][p]["per_pair_day_usd"]) for p in d["by_pair"]]
            n += 1
    return float(max(diffs)), n


def main(workers):
    t0 = time.time()
    if os.path.exists(os.path.join(RES, "delay20_summary.json")):
        raise SystemExit("results/delay20_summary.json exists: rename it before another run (nothing is overwritten)")
    dev = json.load(open(os.path.join(RES, "dev20_summary.json")))
    tasks, days, skipped = L.dev_plan()
    cfgs = ST.configs()
    A, info, kept = C.simulate(tasks, L.DEV_ROOT, False, cfgs, delay=0, workers=workers, mid_delays=S.MID_DELAYS)
    D = kept["mid_delayed"]
    M = C.meta(tasks, days)
    n_days = len(days)
    idx = ST.boot_index(n_days)
    rows, mids = DV.table(A, M, cfgs, n_days, idx)
    worst, n_rows = compare(rows, mids, dev)
    out = {"delays_s": list(S.MID_DELAYS), "settings": {},
           "checks": {"dev20_rows_compared": n_rows, "dev20_max_abs_diff": worst, "dev20_numbers_equal": worst == 0.0,
                      "m_trades_equal_signal_timetable": info["settings_with_m_trades_unlike_the_signal_timetable"] == 0,
                      "pair_days_left_out_no_state_after_first_state": info["pair_days_left_out_no_state_after_first_state"],
                      "trades_valued_at_last_valid_mids": int(sum(m["exit_last"] for m in mids))},
           "sim": {**info, "days": n_days, "first_day": days[0], "last_day": days[-1]},
           "trials_cumulative": dev["trials_cumulative"], "boot": {"B": ST.BOOT_B, "block": ST.BOOT_BLOCK, "seed": ST.BOOT_SEED}}
    for ci, c in enumerate(cfgs):
        r = C.delayed_rows(A, D, M, ci, S.MID_DELAYS, n_days, idx)
        out["settings"][ST.sig_name(c)] = {"signal": list(c), "rows": r, "sentence_5s": S.delay_sentence(r["5"]["ci_lo"], r["5"]["ratio_to_delay0"])}
    out.update({"runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")})
    if not (out["checks"]["dev20_numbers_equal"] and out["checks"]["m_trades_equal_signal_timetable"]):
        with open(os.path.join(RES, "delay20_checks_FAILED.json"), "w") as fh:
            json.dump(out["checks"], fh, indent=1)
        raise SystemExit("the step-2 development numbers are not reproduced: results/delay20_checks_FAILED.json (no summary written)")
    with open(os.path.join(RES, "delay20_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str, ensure_ascii=False)
    report(out)
    print(open(os.path.join(RES, "delay20_report.md")).read())


def report(out):
    s, c = out["sim"], out["checks"]
    L_ = ["# 연구 20 · 2단계 보완: 지연 중간값 손익 (SPEC20_ADDENDUM B2, 개발 구간, 판정 아님)", "",
          f"- 쌍·일 {s['pair_days']:,}개, {s['days']:,}일({s['first_day']} ~ {s['last_day']}). 봉인 데이터는 읽지 않았다.",
          "- H1과 같은 거래(신호만의 시간표)를 쓴다. 판단 시점은 그대로 두고, 진입·청산 가격만 판단 d초 뒤의 중간값으로 바꾼다.",
          "- 그 시점에 두 다리 호가가 유효하지 않으면 그 뒤 첫 유효 막대의 중간값을 쓴다. CI는 일 블록 21, 10,000회, seed 20이다.",
          f"- 후보 선택에 쓰지 않는 점검이라 시행 수는 {out['trials_cumulative']} 그대로다.", "",
          "| 신호 설정 | 지연 | 거래당 평균 (달러) | 일 합 평균 [95% CI] | 지연 0초 대비 | 승률 |", "|---|---|---|---|---|---|"]
    for name, st in out["settings"].items():
        for d, r in st["rows"].items():
            L_.append(f"| {name} | {d}초 | {r['per_trade_usd']:+.2f} | {r['mean_daily_usd']:+,.1f} [{r['ci_lo']:+,.1f}, {r['ci_hi']:+,.1f}] | "
                      f"{100 * r['ratio_to_delay0']:.1f}% | {100 * r['win_rate']:.1f}% |")
    L_ += ["", "## 5초 문장 (ADDENDUM B2의 표를 개발 구간에 적용한 것, 참고)", ""]
    L_ += [f"- {name}: \"{st['sentence_5s']}\"" for name, st in out["settings"].items()]
    L_ += ["", "## 점검", "",
           f"- 2단계 개발 표({c['dev20_rows_compared']}줄: 후보 16개 × 가정 2개, 중간값 8줄)를 지금 코드로 다시 계산한 값과 저장된 값의 최대 차이: "
           f"{c['dev20_max_abs_diff']:.1e} → {'일치' if c['dev20_numbers_equal'] else '불일치'}",
           f"- M의 거래가 신호만의 시간표와 같음: {c['m_trades_equal_signal_timetable']}. 막대 정의에 어긋나 뺀 쌍·일: "
           f"{c['pair_days_left_out_no_state_after_first_state']}개. 마지막 유효 중간값으로 평가한 거래: {c['trades_valued_at_last_valid_mids']}건.",
           f"- 실행 {out['runtime_s']} s"]
    with open(os.path.join(RES, "delay20_report.md"), "w") as fh:
        fh.write("\n".join(L_) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args().workers)
