#!/usr/bin/env python3
"""Research 20 sealed test (SPEC20 5-6, SPEC20_ANSWERS A4; step-2 prompt 3), to be run once at step 4:
  python run_test20.py --go           sealed pairs 2021-01-04..2026-11-09
  python run_test20.py --rehearsal    the same code path on the development data (pairs of 2015-2020)
The sealed run refuses unless study20_pairs/UNLOCK20.txt holds exactly PHRASE (written by Kyle; Claude never creates the file),
no test result exists, every file listed in FREEZE20.md has its sha256, SPEC20.md, SPEC20_ANSWERS.md and SPEC20_ADDENDUM.md
match their sha256 files and the constants below equal the development selection (results/dev20_summary.json). The guard runs before any
sealed path is built. The rehearsal never builds a sealed path and checks that nothing it read lies under the sealed folder.
Verdict
  H1  the mean of the daily sums of the mid P&L of the signal-only trades (setting H1_SIGNAL) is > 0
  H2  the mean daily P&L of A* (L3D1) is > 0; only when the development selection was profitable (H2_EXISTS)
  one-sided p = share of day-block bootstrap means <= 0 (blocks of 21 days, 10,000 resamples, seed 20);
  Holm at 0.05 (0.025 then 0.05) with H1 and H2, H1 alone at 0.05 without H2. Sentences: SPEC20 6, as written.
Secondary, no verdict (SPEC20 5): L0D0, pairs and years, the cost per trade of M and P, trades, holding time, the share of
P order pairs with one leg filled by limit only, the one-second delay check, the daily P&L for the cumulative curve.
Secondary, no verdict (SPEC20_ADDENDUM B2): the H1 trades priced 1, 5 and 30 seconds after the decisions; the 5-second
sentence is chosen from the addendum's table. Quote gaps follow SPEC20_ADDENDUM B1 (signal20, pair20); trades valued at the
last valid mids and pair-days left out for a bar without a quote state after the first one are counted in the report.
Results are written once at the end."""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core20 as C  # noqa: E402
import loader20 as L  # noqa: E402
import signal20 as S  # noqa: E402
import stats20 as ST  # noqa: E402

RES = os.path.join(HERE, "results")
# ---- development selection (FREEZE20.md); the guard compares these with results/dev20_summary.json
ASTAR = "W30_z3_k1_P"
ASTAR_SIGNAL = (30, 3.0, 1.0)                      # (W, z_in, kappa)
ASTAR_EXEC = "P"                                     # "M" or "P"
H2_EXISTS = False                                    # development: no candidate with a mean daily P&L above 0
H1_SIGNAL = (30, 2.0, 0.0)                         # A*'s setting when H2 exists, else the default (30, 2, 0)
# ----
UNLOCK = os.path.join(HERE, "UNLOCK20.txt")
PHRASE = "연구20 테스트를 한 번 연다"
FREEZE = os.path.join(HERE, "FREEZE20.md")
DEV_SUMMARY = os.path.join(RES, "dev20_summary.json")
DELAY_SUMMARY = os.path.join(RES, "delay20_summary.json")           # development table of the delayed mid P&L (rehearsal check)
SEALED_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research20_data/sealed")
SEALED_MANIFEST = os.path.join(HERE, "SEALED_MANIFEST20.tsv")
WINDOW = ("20210104", "20261109")
VARS = ("L3D1", "L0D0")
MODES = {"sealed": {"files": ("test20_summary.json", "test20_report.md", "test20_daily.csv"), "title": "봉인 테스트 (쌍 2021–2026)"},
         "rehearsal": {"files": ("rehearsal20_summary.json", "rehearsal20_report.md", "rehearsal20_daily.csv"),
                       "title": "리허설 (개발 데이터: 쌍 2015–2020, 판정 아님)"}}
SENTENCES = {"H1": {True: "두 종류 주식의 가격이 벌어지면 처음 보는 기간에도 다시 붙었다",
                    False: "개발 구간의 되돌림은 처음 보는 기간에서 확인되지 않았다"},
             "H2": {True: "그 차익은 실제 비용을 넣고도 수익이 났다", False: "되돌림은 있었지만 비용을 넘는 수익은 확인되지 않았다"}}
NO_H2 = "개발 구간에서 이미 비용을 넘지 못했다. 그래서 수익 가설은 봉인 데이터로 시험하지 않았다"
CORE = ("run_test20.py", "core20.py", "pair20.py", "signal20.py", "stats20.py", "loader20.py", "../study17_new_stocks/engine17.py",
        "SEALED_MANIFEST20.tsv", "calendar20.csv", "results/dev20_summary.json")


class Refused(Exception):
    pass


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- guard (sealed run only, before any sealed path)
def frozen_files():
    """[(path relative to study20_pairs, sha256)] from the FREEZE20.md table rows | `path` | `sha256` |."""
    if not os.path.isfile(FREEZE):
        return []
    pat = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
    return [m.groups() for m in map(pat.match, open(FREEZE, encoding="utf-8")) if m]


def selection_ok():
    if not os.path.isfile(DEV_SUMMARY):
        return False
    d = json.load(open(DEV_SUMMARY))
    return (d["astar"] == ASTAR and tuple(d["astar_signal"]) == tuple(ASTAR_SIGNAL) and d["astar_exec"] == ASTAR_EXEC
            and d["h2_exists"] is H2_EXISTS and tuple(d["h1_signal"]) == tuple(H1_SIGNAL))


def guard():
    why = []
    if any(os.path.exists(os.path.join(RES, f)) for f in MODES["sealed"]["files"]):
        why.append("test results already exist")
    if not os.path.isfile(UNLOCK) or open(UNLOCK, encoding="utf-8").read().strip() != PHRASE:
        why.append("no unlock phrase in UNLOCK20.txt")
    ff = frozen_files()
    if not ff or not all(any(p == c for p, _ in ff) for c in CORE):
        why.append("FREEZE20.md missing or without the core files")
    for p, h in ff:
        q = os.path.normpath(os.path.join(HERE, p))
        if not os.path.isfile(q) or sha(q) != h:
            why.append(f"frozen file differs: {p}")
    for doc in ("SPEC20", "SPEC20_ANSWERS", "SPEC20_ADDENDUM"):
        f = os.path.join(HERE, f"{doc}.sha256")
        if not os.path.isfile(f) or open(f).read().split()[0] != sha(os.path.join(HERE, f"{doc}.md")):
            why.append(f"{doc}.md differs from {doc}.sha256")
    if not selection_ok():
        why.append("the constants differ from the development selection")
    return why


# ---------------------------------------------------------------- analysis
def cost_split(A, M, ci, v, n_days, idx):
    """Cost per trade (execution cost + fee) of M and P for one setting and their difference P - M with a day-block 95%
    interval (difference of the two ratios of daily sums to daily trade counts)."""
    res = {}
    num, den = {}, {}
    for ex in ("M", "P"):
        ki = C.KINDS.index(f"{ex}_{v}")
        num[ex] = C.daily(A, M, ci, ki, "cost", n_days) + C.daily(A, M, ci, ki, "fee", n_days)
        den[ex] = C.daily(A, M, ci, ki, "n", n_days)
        res[ex] = dict(zip(("mean", "lo", "hi"), ST.ratio_ci(num[ex], den[ex], idx)))
    if den["M"].sum() and den["P"].sum():
        dm, dp = den["M"][idx].sum(axis=1), den["P"][idx].sum(axis=1)
        ok = (dm > 0) & (dp > 0)
        diff = num["P"][idx].sum(axis=1)[ok] / dp[ok] - num["M"][idx].sum(axis=1)[ok] / dm[ok]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        res["P_minus_M"] = {"mean": float(num["P"].sum() / den["P"].sum() - num["M"].sum() / den["M"].sum()), "lo": float(lo), "hi": float(hi)}
    else:
        res["P_minus_M"] = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return res


def analyse(A, A1, D, tasks, days, cfgs):
    M = C.meta(tasks, days)
    n_days = len(days)
    idx = ST.boot_index(n_days)
    ci_h1, ci_a = cfgs.index(tuple(H1_SIGNAL)), cfgs.index(tuple(ASTAR_SIGNAL))
    k_a = C.KINDS.index(f"{ASTAR_EXEC}_L3D1")
    out = {"astar": ASTAR, "astar_signal": list(ASTAR_SIGNAL), "astar_exec": ASTAR_EXEC, "h2_exists": H2_EXISTS, "h1_signal": list(H1_SIGNAL),
           "hypotheses": {}, "settings": {}}
    series = {"H1": C.daily(A, M, ci_h1, 0, "pnl", n_days)}
    if H2_EXISTS:
        series["H2"] = C.daily(A, M, ci_a, k_a, "pnl", n_days)
    for h, d in series.items():
        mean, p, lb = ST.p_positive(d, idx)
        _, lo, hi = ST.mean_ci(d, idx)
        out["hypotheses"][h] = {"series": (f"daily sum of mid P&L, setting {ST.sig_name(H1_SIGNAL)}" if h == "H1" else f"daily P&L of {ASTAR}, L3D1"),
                                "mean_daily_usd": mean, "p_one_sided": p, "lower_2_5": lb, "ci95": [lo, hi]}
    out["hypotheses"]["H1"]["per_trade_usd"] = dict(zip(("mean", "lo", "hi"), C.per_trade(A, M, ci_h1, 0, "pnl", n_days, idx)))
    holm = ST.holm({h: x["p_one_sided"] for h, x in out["hypotheses"].items()})
    out["verdict"] = {h: {**holm[h], "result": "확인" if holm[h]["rejected"] else "확인 안 됨", "mean_daily_usd": out["hypotheses"][h]["mean_daily_usd"],
                          "lower_2_5": out["hypotheses"][h]["lower_2_5"], "sentence": SENTENCES[h][holm[h]["rejected"]]} for h in series}
    if not H2_EXISTS:
        out["verdict"]["H2"] = {"result": "없음", "sentence": NO_H2}
    for ci, c in enumerate(cfgs):
        rows = {k: C.row(A, M, ci, ki, n_days, idx, ST.sig_name(c) + "_" + k) for ki, k in enumerate(C.KINDS)}
        rows["mid"]["per_trade"] = dict(zip(("mean", "lo", "hi"), C.per_trade(A, M, ci, 0, "pnl", n_days, idx)))
        out["settings"][ST.sig_name(c)] = {"signal": list(c), "rows": rows, "cost_per_trade": {v: cost_split(A, M, ci, v, n_days, idx) for v in VARS}}
    out["delay_check"] = {v: C.row(A1, M, 0, C.KINDS.index(f"{ASTAR_EXEC}_{v}"), n_days, idx, ASTAR + "_delay1") for v in VARS}
    dr = C.delayed_rows(A, D, M, ci_h1, S.MID_DELAYS, n_days, idx)                       # SPEC20_ADDENDUM B2, no verdict
    out["mid_delayed"] = {"signal": ST.sig_name(H1_SIGNAL), "delays_s": list(S.MID_DELAYS), "rows": dr,
                          "sentence_5s": S.delay_sentence(dr["5"]["ci_lo"], dr["5"]["ratio_to_delay0"])}
    daily = pd.DataFrame({"date": days, "h1_mid_sum": series["H1"],
                          **{f"h1_mid_delay{d}": ST.daily(M["day"], D[:, ci_h1, di, 1], n_days) for di, d in enumerate(S.MID_DELAYS)},
                          **{f"astar_{v}": C.daily(A, M, ci_a, C.KINDS.index(f"{ASTAR_EXEC}_{v}"), "pnl", n_days) for v in VARS},
                          **{f"astar_signal_{k}": C.daily(A, M, ci_a, ki, "pnl", n_days) for ki, k in enumerate(C.KINDS)},
                          "astar_trades": C.daily(A, M, ci_a, k_a, "n", n_days)})
    out["counts"] = {"pair_days": len(tasks), "days": n_days, "first_day": days[0], "last_day": days[-1], "pairs": M["names"],
                     "trades_valued_at_last_valid_mids": {name: {k: r["exit_last"] for k, r in st["rows"].items()}
                                                          for name, st in out["settings"].items()}}
    return out, daily


# ---------------------------------------------------------------- rehearsal checks (the development numbers of step 2)
NUMERIC = ("mean_daily_usd", "ci_lo", "ci_hi", "p_one_sided", "sharpe", "trades", "win_rate", "mean_hold_s", "pnl_per_trade",
           "mid_per_trade", "cost_per_trade", "fee_per_trade", "total_usd")


def rehearsal_checks(out, read):
    dev = json.load(open(DEV_SUMMARY))
    diffs, n = [], 0
    for name, st in out["settings"].items():
        for k, r in st["rows"].items():
            if k == "mid":
                d = next(x for x in dev["mid_rows"] if x["name"] == name + "_mid")
            else:
                ex, v = k.split("_")
                d = next(x for x in dev["rows"] if x["name"] == f"{name}_{ex}" and x["variant"] == v)
            diffs += [abs(r[f] - d[f]) for f in NUMERIC] + [abs(r["by_year"][y] - d["by_year"][y]) for y in d["by_year"]]
            diffs += [abs(r["by_pair"][p]["per_pair_day_usd"] - d["by_pair"][p]["per_pair_day_usd"]) for p in d["by_pair"]]
            n += 1
    for v in VARS:
        diffs += [abs(out["delay_check"][v][f] - dev["delay_check"][v][f]) for f in NUMERIC]
    h1 = next(x for x in dev["mid_rows"] if x["name"] == ST.sig_name(H1_SIGNAL) + "_mid")
    diffs += [abs(out["hypotheses"]["H1"]["mean_daily_usd"] - h1["mean_daily_usd"]), abs(out["hypotheses"]["H1"]["p_one_sided"] - h1["p_one_sided"])]
    if H2_EXISTS:
        a = next(x for x in dev["rows"] if x["name"] == ASTAR and x["variant"] == "L3D1")
        diffs += [abs(out["hypotheses"]["H2"]["mean_daily_usd"] - a["mean_daily_usd"]), abs(out["hypotheses"]["H2"]["p_one_sided"] - a["p_one_sided"])]
    dl = json.load(open(DELAY_SUMMARY))["settings"][ST.sig_name(H1_SIGNAL)]
    dd = [abs(out["mid_delayed"]["rows"][d][f] - dl["rows"][d][f]) for d in dl["rows"] for f in dl["rows"][d]]
    chk = {"rows_compared": n, "dev20_max_abs_diff": float(max(diffs)), "delay20_max_abs_diff": float(max(dd)),
           "delay20_sentence_equal": out["mid_delayed"]["sentence_5s"] == dl["sentence_5s"], "selection_equal": selection_ok(),
           "no_sealed_path_read": not any(os.path.abspath(p).startswith(SEALED_ROOT) for p in read)}
    chk["dev20_numbers_equal"] = chk["dev20_max_abs_diff"] == 0.0 and chk["delay20_max_abs_diff"] == 0.0
    chk["pass"] = bool(chk["dev20_numbers_equal"] and chk["delay20_sentence_equal"] and chk["selection_equal"] and chk["no_sealed_path_read"])
    return chk


# ---------------------------------------------------------------- report
def ci(r):
    return f"{r['mean_daily_usd']:+,.1f} [{r['ci_lo']:+,.1f}, {r['ci_hi']:+,.1f}]"


def report(out, mode, path):
    c = out["counts"]
    L_ = [f"# 연구 20 · {MODES[mode]['title']}", "",
          f"- 쌍·일 {c['pair_days']:,}개, {c['days']:,}일({c['first_day']} ~ {c['last_day']}). A* = {out['astar']}"
          + ("" if out["h2_exists"] else " (개발에서 수익 없음: H2 없음)") + f". H1의 신호 설정은 {ST.sig_name(out['h1_signal'])}.",
          "- 판정: 부트스트랩 단측 p(평균이 0 이하인 비율, 일 블록 21, 10,000회, seed 20)에 "
          + ("Holm(0.025, 0.05)" if out["h2_exists"] else "H1 단독 0.05") + ". 하한은 단측 2.5%. 손익 단위는 달러(다리마다 5만 달러).", ""]
    if mode == "rehearsal":
        L_ += ["**리허설: 개발 데이터라서 아래 \"확인\" 여부와 문장은 코드 경로 점검용이며 판정이 아니다.**", ""]
    L_ += ["## 판정", "", "| 가설 | 계열 | 평균 (달러/일) | 단측 2.5% 하한 | p | Holm 수준 | 결과 | 문장 |", "|---|---|---|---|---|---|---|---|"]
    for h, x in out["verdict"].items():
        if "p" in x:
            L_.append(f"| {h} | {out['hypotheses'][h]['series']} | {x['mean_daily_usd']:+,.1f} | {x['lower_2_5']:+,.1f} | {x['p']:.4f} | {x['level']:.3f} | "
                      f"{x['result']} | \"{x['sentence']}\" |")
        else:
            L_.append(f"| {h} | — | — | — | — | — | {x['result']} | \"{x['sentence']}\" |")
    pt = out["hypotheses"]["H1"]["per_trade_usd"]
    L_ += ["", f"- H1 거래당 중간값 손익: {pt['mean']:+.2f}달러 [{pt['lo']:+.2f}, {pt['hi']:+.2f}]"]
    for name, st in out["settings"].items():
        R = st["rows"]
        yrs = sorted(R["mid"]["by_year"])
        L_ += ["", f"## 보조 (판정 아님): 신호 설정 {name}", "",
               "| 실행 | 평균 일 손익 [95% CI] | 샤프 | 거래 수 | 승률 | 보유(초) | 거래당 손익 | = 중간값 | − 비용 | − 수수료 | 한쪽만 지정가 | 기한 시장가 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, r in R.items():
            pf = (f"{r['p_one_leg_limit_frac']:.3f} | {r['p_deadline_order_frac']:.3f}" if k.startswith("P") else "— | —")
            L_.append(f"| {k} | {ci(r)} | {r['sharpe']:+.2f} | {r['trades']:,} | {r['win_rate']:.3f} | {r['mean_hold_s']:.0f} | {r['pnl_per_trade']:+.2f} | "
                      f"{r['mid_per_trade']:+.2f} | {r['cost_per_trade']:.2f} | {r['fee_per_trade']:.2f} | {pf} |")
        L_ += ["", "| 가정 | M의 거래당 비용 | P의 거래당 비용 | P − M [95% CI] |", "|---|---|---|---|"]
        for v in VARS:
            x = st["cost_per_trade"][v]
            L_.append(f"| {v} | {x['M']['mean']:.2f} | {x['P']['mean']:.2f} | {x['P_minus_M']['mean']:+.2f} [{x['P_minus_M']['lo']:+.2f}, {x['P_minus_M']['hi']:+.2f}] |")
        L_ += ["", "| 연도별 평균 일 손익 | " + " | ".join(yrs) + " |", "|---" * (len(yrs) + 1) + "|"]
        for k, r in R.items():
            L_.append(f"| {k} | " + " | ".join(f"{r['by_year'][y]:+,.1f}" for y in yrs) + " |")
        pn = c["pairs"]
        L_ += ["", "| 쌍별 (쌍·일당 달러) | " + " | ".join(pn) + " |", "|---" * (len(pn) + 1) + "|"]
        for k, r in R.items():
            L_.append(f"| {k} | " + " | ".join(f"{r['by_pair'][p]['per_pair_day_usd']:+,.1f}" for p in pn) + " |")
        L_.append("| 쌍·일 수 | " + " | ".join(f"{R['mid']['by_pair'][p]['pair_days']:,}" for p in pn) + " |")
    L_ += ["", "## 보조: 지연 점검 (신호 t, 주문 t + 1초)", "", "| 가정 | A* | 지연 1초 | 거래 수 |", "|---|---|---|---|"]
    for v in VARS:
        base = out["settings"][ST.sig_name(out["astar_signal"])]["rows"][f"{out['astar_exec']}_{v}"]
        d = out["delay_check"][v]
        L_.append(f"| {v} | {ci(base)} | {ci(d)} | {d['trades']:,} |")
    md = out["mid_delayed"]
    L_ += ["", f"## 보조: 지연 중간값 손익 (SPEC20_ADDENDUM B2, 신호 설정 {md['signal']}, 판정 아님)", "",
           "- H1과 같은 거래다. 판단 시점은 그대로 두고 진입·청산 가격만 판단 d초 뒤의 중간값으로 바꿨다.", "",
           "| 지연 | 거래당 평균 (달러) | 일 합 평균 [95% CI] | 지연 0초 대비 | 승률 |", "|---|---|---|---|---|"]
    for d, r in md["rows"].items():
        L_.append(f"| {d}초 | {r['per_trade_usd']:+.2f} | {r['mean_daily_usd']:+,.1f} [{r['ci_lo']:+,.1f}, {r['ci_hi']:+,.1f}] | "
                  f"{100 * r['ratio_to_delay0']:.1f}% | {100 * r['win_rate']:.1f}% |")
    L_ += ["", f"- 5초 문장: \"{md['sentence_5s']}\""]
    sim = out["sim"]
    lastv = sum(sum(x.values()) for x in c["trades_valued_at_last_valid_mids"].values())
    L_ += ["", "## 호가가 비는 구간 (SPEC20_ADDENDUM B1)", "",
           f"- 유효 격자 시점 비율 {sim['valid_grid_points'] / max(sim['grid_points'], 1):.4f}.",
           f"- 15:55부터 마감까지 유효 호가가 없어 마지막 유효 중간값으로 평가한 거래: {lastv}건 (실행별: {c['trades_valued_at_last_valid_mids']}).",
           f"- 첫 호가 뒤에 호가 상태가 없는 막대가 있어 통째로 뺀 쌍·일: {sim['pair_days_left_out_no_state_after_first_state']}개."]
    if "checks" in out:
        k = out["checks"]
        L_ += ["", "## 리허설 점검", "", f"- 2단계 개발 숫자와 최대 차이 {k['dev20_max_abs_diff']:.1e} (표 {k['rows_compared']}개), "
               f"지연 중간값 표와 최대 차이 {k['delay20_max_abs_diff']:.1e}, 5초 문장 같음 {k['delay20_sentence_equal']} → {'일치' if k['dev20_numbers_equal'] else '불일치'}",
               f"- 상수가 개발 선택과 같음: {k['selection_equal']}. 봉인 경로를 읽지 않음: {k['no_sealed_path_read']}."]
    L_ += ["", f"- 일 손익(누적 곡선용): results/{MODES[mode]['files'][2]}. 읽은 막대 폴더: {out['read']['bars_root']}. 실행 {out['runtime_s']} s"]
    with open(path, "w") as fh:
        fh.write("\n".join(L_) + "\n")


# ---------------------------------------------------------------- run
def main(mode, workers):
    t0 = time.time()
    if not isinstance(ASTAR_SIGNAL, tuple) or not isinstance(H2_EXISTS, bool):
        raise SystemExit("the development selection is not set in run_test20.py")
    if mode == "sealed":
        why = guard()
        if why:
            raise SystemExit("refused: " + "; ".join(why))
        root, allow = os.path.join(SEALED_ROOT, "bars_1s"), True
        tasks, days, skipped = L.plan(L.manifest(SEALED_MANIFEST), WINDOW)
        read = [SEALED_MANIFEST, L.CALENDAR, root]
    else:
        if any(os.path.exists(os.path.join(RES, f)) for f in MODES[mode]["files"]):
            raise SystemExit("rehearsal results already exist: rename them first (nothing is overwritten)")
        root, allow = L.DEV_ROOT, False
        tasks, days, skipped = L.dev_plan()
        read = [L.DEV_MANIFEST, L.CALENDAR, root]
    cfgs = [tuple(H1_SIGNAL)] + ([] if tuple(ASTAR_SIGNAL) == tuple(H1_SIGNAL) else [tuple(ASTAR_SIGNAL)])
    A, info, kept = C.simulate(tasks, root, allow, cfgs, delay=0, workers=workers, mid_delays=S.MID_DELAYS)
    if info["settings_with_m_trades_unlike_the_signal_timetable"]:
        raise SystemExit("the M trades differ from the signal-only timetable (nothing written)")
    A1, _, _ = C.simulate(tasks, root, allow, [tuple(ASTAR_SIGNAL)], delay=1, workers=workers)
    out, daily = analyse(A, A1, kept["mid_delayed"], tasks, days, cfgs)
    out["sim"] = {**info, "skipped_pair_days": skipped}
    out["read"] = {"bars_root": root.replace(os.path.expanduser("~"), "~"), "other": [os.path.relpath(p, HERE) for p in read[:2]]}
    if mode == "rehearsal":
        out["checks"] = rehearsal_checks(out, read)
    out.update({"mode": mode, "boot": {"B": ST.BOOT_B, "block": ST.BOOT_BLOCK, "seed": ST.BOOT_SEED},
                "runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")})
    f_sum, f_rep, f_day = (os.path.join(RES, f) for f in MODES[mode]["files"])
    daily.to_csv(f_day, index=False, float_format="%.4f")
    with open(f_sum, "w") as fh:
        json.dump(out, fh, indent=1, default=str, ensure_ascii=False)
    report(out, mode, f_rep)
    print(open(f_rep).read())
    if mode == "rehearsal" and not out["checks"]["pass"]:
        raise SystemExit("the rehearsal checks failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--go", action="store_true")
    g.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main("sealed" if a.go else "rehearsal", a.workers)
