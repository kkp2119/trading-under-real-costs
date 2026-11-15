#!/usr/bin/env python3
"""Research 24 sealed test (SPEC24 5-6, SPEC24_ANSWERS; step-2 prompt 4), to be run once:
  python run_test24.py --go           sealed pairs 2021-01-04..2026-11-10
  python run_test24.py --rehearsal    the same code path on the development data (pairs of 2015-2020)
The sealed run refuses unless study24_big_gaps/UNLOCK24.txt holds exactly PHRASE (written by Kyle; Claude never creates the
file), no test result exists, every file of CORE is listed in FREEZE24.md and every listed file has its sha256, SPEC24.md
and SPEC24_ANSWERS.md match their sha256 files, the engine files of research 17, 20 and 21 equal their tags, the
development selection is usable and A* was profitable in development (SPEC24 4: a mean <= 0 means no sealed run). The guard
runs before any sealed path is built. A* is read from results/dev24_summary.json, which FREEZE24.md pins. The rehearsal
never builds a sealed path.
Verdict (one hypothesis, alpha 0.05; one-sided p = share of day-block bootstrap means <= 0, blocks of 21 days, 10,000
resamples, seed 24):
  H1  the mean daily P&L of A* (market orders, 10,000 dollars per leg, every order 1 second late, L3D1) is > 0
Secondary, no verdict (SPEC24 5, ANSWERS A5): delays 0 and 5 seconds, L0D0, families, pair types, pairs and years,
pnl = mid - cost - fee per trade, the share of trades with an order above the displayed size, the result without the 5
largest days and the 5 largest pairs, the split at 2025-11-03, the diagnostics. Results are written once at the end."""
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
import core24 as C24  # noqa: E402
import loader24 as L24  # noqa: E402
import stats24 as ST24  # noqa: E402

C, ST20 = C24.C, C24.ST20
RES = os.path.join(HERE, "results")
UNLOCK = os.path.join(HERE, "UNLOCK24.txt")
PHRASE = "연구24 테스트를 한 번 연다"
FREEZE = os.path.join(HERE, "FREEZE24.md")
DEV_SUMMARY = os.path.join(RES, "dev24_summary.json")
SEALED_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research24_data/sealed")
SEALED_BARS = os.path.join(SEALED_ROOT, "bars_1s")
SEALED_MANIFEST = os.path.join(HERE, "SEALED_MANIFEST24.tsv")
WINDOW = ("20210104", "20261110")
DOCS = ("SPEC24", "SPEC24_ANSWERS")
MAIN = C24.run_name(ST24.DELAY_MAIN, ST24.VAR_MAIN)
NOT_PROFITABLE = "no profit in development: the sealed run is not made (SPEC24 4)"
MODES = {"sealed": {"files": ("test24_summary.json", "test24_report.md", "test24_daily.csv"), "title": "봉인 테스트 (쌍 2021–2026)"},
         "rehearsal": {"files": ("rehearsal24_summary.json", "rehearsal24_report.md", "rehearsal24_daily.csv"),
                       "title": "리허설 (개발 데이터: 쌍 2015–2020, 판정 아님)"}}
CORE = ("run_test24.py", "core24.py", "pair24.py", "loader24.py", "stats24.py", "../study21_more_pairs/core21.py", "../study21_more_pairs/loader21.py",
        "../study21_more_pairs/stats21.py", "../study20_pairs/core20.py", "../study20_pairs/pair20.py", "../study20_pairs/signal20.py", "../study20_pairs/stats20.py",
        "../study17_new_stocks/engine17.py", "SEALED_MANIFEST24.tsv", "calendar24.csv", "results/dev24_summary.json") + tuple(
            f"universe24_{y}.tsv" for y in range(2021, 2027))


class Refused(Exception):
    pass


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- development selection and guard
def selection():
    """{"astar", "signal" (W, z_in, kappa), "profitable"} from the development summary; refuses anything outside the grid."""
    d = json.load(open(DEV_SUMMARY, encoding="utf-8"))
    sig = (int(d["astar_signal"][0]), float(d["astar_signal"][1]), float(d["astar_signal"][2]))
    if sig not in ST24.candidates() or d["astar"] != ST24.sig_name(sig) or not isinstance(d["profitable_in_development"], bool):
        raise Refused("the development selection is not a candidate of the grid")
    if d["profitable_in_development"] != (d["selection"]["astar_mean_usd"] > 0) or d["selection"]["astar"] != d["astar"]:
        raise Refused("the profit flag does not follow from the A* mean")
    return {"astar": d["astar"], "signal": sig, "profitable": d["profitable_in_development"]}


def frozen_files():
    """[(path relative to study24_big_gaps, sha256)] from the FREEZE24.md table rows | `path` | `sha256` |."""
    if not os.path.isfile(FREEZE):
        return []
    pat = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
    return [m.groups() for m in map(pat.match, open(FREEZE, encoding="utf-8")) if m]


def guard():
    why = []
    if any(os.path.exists(os.path.join(RES, f)) for f in MODES["sealed"]["files"]):
        why.append("test results already exist")
    if not os.path.isfile(UNLOCK) or open(UNLOCK, encoding="utf-8").read().strip() != PHRASE:
        why.append("no unlock phrase in UNLOCK24.txt")
    ff = frozen_files()
    if not ff or not all(any(p == c for p, _ in ff) for c in CORE):
        why.append("FREEZE24.md missing or without the core files")
    for p, h in ff:
        q = os.path.normpath(os.path.join(HERE, p))
        if not os.path.isfile(q) or sha(q) != h:
            why.append(f"frozen file differs: {p}")
    for doc in DOCS:
        f = os.path.join(HERE, f"{doc}.sha256")
        if not os.path.isfile(f) or open(f).read().split()[0] != sha(os.path.join(HERE, f"{doc}.md")):
            why.append(f"{doc}.md differs from {doc}.sha256")
    if C24.frozen_differences():
        why.append("engine files differ from their tags")
    try:
        if not selection()["profitable"]:
            why.append(NOT_PROFITABLE)
    except Exception as e:                                   # a missing or unreadable development summary included
        why.append(f"no usable development selection: {e}")
    return why


# ---------------------------------------------------------------- analysis
def analyse(tasks, days, skipped, sel, root, allow_sealed, workers):
    cfg, name = sel["signal"], sel["astar"]
    A, X, info, ex = C24.simulate(tasks, root, allow_sealed, [cfg], workers=workers, diag=True)
    n_days = len(days)
    idx = ST24.boot_index(n_days)
    T = C24.setting_tables(A, X, tasks, days, 0, ST24.DELAYS, idx, name)
    r = T["runs"][MAIN]
    ok = bool(r["p_one_sided"] < ST24.ALPHA)
    verdict = {"H1": {"p": r["p_one_sided"], "level": ST24.ALPHA, "confirmed": ok, "result": "확인" if ok else "확인 안 됨",
                      "mean_daily_usd": r["mean_daily_usd"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "lower_2_5": r["lower_2_5"],
                      "sentence": ST24.SENTENCES[ok]}}
    years = sorted({t["year"] for t in tasks})
    daily = pd.DataFrame(C24.pair_daily(A, tasks, 0, name, ST24.DELAYS), columns=C24.daily_cols(ST24.DELAYS))
    g = daily.sum(numeric_only=True)
    gap = max(abs(float(g[f"d{d}_{v}"]) - T["runs"][C24.run_name(d, v)]["total_usd"]) for d in ST24.DELAYS for v in ST24.VARS)
    out = {"astar": name, "astar_signal": list(cfg), "profitable_in_development": sel["profitable"], "hypotheses": ["H1"], "verdict": verdict,
           "amount_usd": ST24.AMOUNT, "delays": list(ST24.DELAYS), "delay_main": ST24.DELAY_MAIN, **T,
           "diag": C24.diag_shares(ex["diag"], tasks, days, [cfg]),
           "counts": {"pair_days": len(tasks), "days": n_days, "first_day": days[0], "last_day": days[-1], "pairs": len({t["key"] for t in tasks}),
                      "pair_days_by_year": {str(y): int(sum(t["year"] == y for t in tasks)) for y in years},
                      "pair_days_by_family": {f: int(sum(t["family"] == f for t in tasks)) for f in ST24.FAMILIES},
                      "pair_days_by_type": {k: int(sum(t["type"] == k for t in tasks)) for k in ST24.TYPES},
                      "trades_valued_at_last_valid_mids": int(r["exit_last"])},
           "sim": {**info, "skipped_pair_days": skipped, "workers": workers}, "checks": {"daily_file_gap_usd": float(gap)}}
    return out, daily


def rehearsal_checks(out, tasks, root):
    """The rehearsal against the development run: every number of A*'s tables must be equal, and no sealed path was read."""
    dev = json.load(open(DEV_SUMMARY, encoding="utf-8"))
    old = dev["candidates"][out["astar"]]
    worst, n = 0.0, 0
    for key in ("runs", "by_family", "by_type", "without_top", "date_split", "by_pair_key"):
        new = json.loads(json.dumps(out[key], default=C24.jdefault))
        w, m = C24.deep_diff(new, old[key])
        worst, n = max(worst, w), n + m
    paths = [os.path.abspath(os.path.join(root, t[k]["file"])) for t in tasks for k in ("row_a", "row_b")]
    no_sealed = not any("sealed" in p.split(os.sep) for p in paths) and os.path.abspath(root) == os.path.abspath(L24.DEV_ROOT)
    ok = worst == 0.0 and n > 0 and no_sealed and out["astar"] == dev["astar"]
    return {"numbers_compared": n, "max_abs_diff": worst, "selection_equal": out["astar"] == dev["astar"], "no_sealed_path_read": bool(no_sealed), "pass": bool(ok)}


def usd(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+,.1f}"


def report(out, mode):
    """The report text (built before any file is written)."""
    c, v = out["counts"], out["verdict"]["H1"]
    L_ = [f"# 연구 24 · {MODES[mode]['title']}", "",
          f"- 쌍·일 {c['pair_days']:,}개, {c['days']:,}일({c['first_day']} ~ {c['last_day']}), 서로 다른 쌍 {c['pairs']}개. A* = {out['astar']}, 다리당 1만 달러, 시장가.",
          "- 판정: 일 블록 부트스트랩(21일, 10,000회, seed 24)의 단측 p(평균이 0 이하인 비율), 수준 0.05. 일 손익은 그날 모든 쌍의 합계(달러)다.",
          "- 2025-11-03부터의 호가 잔량은 100으로 나눠 읽었다(SPEC24 3절).", "", "## 판정 (지연 1초, L3D1)", "",
          "| 가설 | 평균 일 손익 | 95% CI | p | 수준 | 결과 | 문장 |", "|---|---|---|---|---|---|---|",
          f"| H1 | {usd(v['mean_daily_usd'])} | [{usd(v['ci_lo'])}, {usd(v['ci_hi'])}] | {v['p']:.4f} | {v['level']:.2f} | {v['result']} | \"{v['sentence']}\" |",
          "", "## 보조 결과 (판정 아님)", "",
          "| 실행 | 평균 일 손익 [95% CI] | p | 거래 수 | 하루 거래 | 이긴 거래 | 거래당 손익 = 중간값 − 비용 − 수수료 | 잔량 초과 거래 | 초과 주식 |", "|---|---|---|---|---|---|---|---|---|"]
    for d in ST24.DELAYS:
        for var in ST24.VARS:
            r = out["runs"][C24.run_name(d, var)]
            if r["trades"]:
                L_.append(f"| 지연 {d}초, {var} | {usd(r['mean_daily_usd'])} [{usd(r['ci_lo'])}, {usd(r['ci_hi'])}] | {r['p_one_sided']:.4f} | {r['trades']:,} | {r['trades_per_day']:.2f} | "
                          f"{100 * r['win_rate']:.1f}% | {usd(r['pnl_per_trade'])} = {usd(r['mid_per_trade'])} − {r['cost_per_trade']:,.1f} − {r['fee_per_trade']:,.1f} | "
                          f"{100 * r['over_trade_frac']:.1f}% | {100 * r['over_share_frac']:.1f}% |")
            else:
                L_.append(f"| 지연 {d}초, {var} | {usd(r['mean_daily_usd'])} | — | 0 | 0.00 | — | — | — | — |")
    L_ += ["", "| 나눈 부분 (지연 1초, L3D1) | 쌍·일 | 평균 일 손익 [95% CI] | 거래 수 | 거래당 손익 |", "|---|---|---|---|---|"]
    for key, lab in (("by_family", {f: f"가족 {f}" for f in ST24.FAMILIES}), ("by_type", ST24.TYPE_KO)):
        for k, x in out[key].items():
            r = x["runs"][MAIN]
            L_.append(f"| {lab[k]} | {x['pair_days']:,} | {usd(r['mean_daily_usd'])} [{usd(r['ci_lo'])}, {usd(r['ci_hi'])}] | {r['trades']:,} | {usd(r['pnl_per_trade']) if r['trades'] else '—'} |")
    yrs = out["runs"][MAIN]["by_year"]
    L_ += ["", "| 연도별 평균 일 손익 (지연 1초, L3D1) | " + " | ".join(yrs) + " |", "|---|" + "---|" * len(yrs), "| | " + " | ".join(usd(x) for x in yrs.values()) + " |"]
    w = out["without_top"][MAIN]
    L_ += ["", "| 뺀 것 (지연 1초, L3D1) | 평균 일 손익 [95% CI] | 전체 합계에서 차지한 비중 |", "|---|---|---|",
           f"| 없음 | {usd(w['all']['mean_daily_usd'])} [{usd(w['all']['ci_lo'])}, {usd(w['all']['ci_hi'])}] | |",
           f"| 손익이 가장 큰 5일 | {usd(w['without_top_days']['mean_daily_usd'])} [{usd(w['without_top_days']['ci_lo'])}, {usd(w['without_top_days']['ci_hi'])}] | {100 * w['without_top_days']['share_of_total']:.0f}% |",
           f"| 손익이 가장 큰 5쌍 | {usd(w['without_top_pairs']['mean_daily_usd'])} [{usd(w['without_top_pairs']['ci_lo'])}, {usd(w['without_top_pairs']['ci_hi'])}] | {100 * w['without_top_pairs']['share_of_total']:.0f}% |",
           f"| 둘 다 | {usd(w['without_both']['mean_daily_usd'])} [{usd(w['without_both']['ci_lo'])}, {usd(w['without_both']['ci_hi'])}] | |",
           "", "- 뺀 5쌍: " + ", ".join(f"{q} ({usd(x)})" for q, x in w["without_top_pairs"]["pairs_removed"]) + ".",
           "- 뺀 5일: " + ", ".join(f"{q} ({usd(x)})" for q, x in w["without_top_days"]["days_removed"]) + ".", "",
           "| 날짜로 나눈 결과 (지연 1초, L3D1) | 일수 | 평균 일 손익 [95% CI] | 거래 수 |", "|---|---|---|---|"]
    for key, lab in (("before", "2025-10-31까지"), ("from", "2025-11-03부터")):
        x = out["date_split"][MAIN][key]
        L_.append(f"| {lab} | — | — | — |" if x is None else f"| {lab} | {x['days']:,} | {usd(x['mean_daily_usd'])} [{usd(x['ci_lo'])}, {usd(x['ci_hi'])}] | {x['trades']:,} |")
    dg = out["diag"]["family"]
    s = out["sim"]
    L_ += ["", "## 진단과 자료 관련 건수", "",
           "- σ 하한(1bp)이 적용된 시점의 비율: " + ", ".join(f"{k} {100 * x['share']:.1f}%" for k, x in dg["sigma_floor"].items()) + ". |z| ≥ 3인 시점 가운데 κ 조건에 막힌 비율: "
           + ", ".join(f"{k} {100 * x['share']:.1f}%" for k, x in dg["kappa_blocked"].items()) + ".",
           f"- 호가 문제로 뺀 쌍·일 {s['pair_days_left_out_no_state_after_first_state']:,}개. 한쪽 다리만 있거나 둘 다 없는 쌍·일: {s['skipped_pair_days']}. "
           f"마지막 유효 중간값으로 평가한 거래 {c['trades_valued_at_last_valid_mids']}건.",
           f"- 지연 0초의 거래 시간표가 신호만의 시간표와 다른 설정: {s['settings_with_m_trades_unlike_the_signal_timetable']}개."]
    if mode == "rehearsal":
        k = out["checks"]["rehearsal"]
        L_ += ["", "## 리허설 점검", "",
               f"- 개발 실행(dev24)의 숫자와 최대 차이 {k['max_abs_diff']:.1e}(숫자 {k['numbers_compared']:,}개) → {'일치' if k['pass'] else '불일치'}. "
               f"A* 같음 {k['selection_equal']}, 봉인 경로를 읽지 않음 {k['no_sealed_path_read']}.",
               "- 이 보고서의 \"확인\" 여부는 개발 데이터로 낸 것이라 판정이 아니다."]
    L_ += ["", f"- 쌍별 일 손익: results/{MODES[mode]['files'][2]}. 읽은 막대 폴더: {out['read']}. 실행 {out['runtime_s']} s", ""]
    return "\n".join(L_)


# ---------------------------------------------------------------- run
def main(mode, workers):
    t0 = time.time()
    if mode == "sealed":
        why = guard()
        if why:
            raise SystemExit("refused: " + "; ".join(why))
        tasks, days, skipped = L24.plan(L24.manifest(SEALED_MANIFEST), WINDOW)
        root, allow = SEALED_BARS, True
    else:
        if any(os.path.exists(os.path.join(RES, f)) for f in MODES[mode]["files"]):
            raise SystemExit("rehearsal results already exist: rename them first (nothing is overwritten)")
        tasks, days, skipped = L24.dev_plan()
        root, allow = L24.DEV_ROOT, False
    sel = selection()
    out, daily = analyse(tasks, days, skipped, sel, root, allow, workers)
    out["read"] = root.replace(os.path.expanduser("~"), "~")
    if mode == "rehearsal":
        out["checks"]["rehearsal"] = rehearsal_checks(out, tasks, root)
    out.update({"mode": mode, "boot": {"B": ST24.BOOT_B, "block": ST24.BOOT_BLOCK, "seed": ST24.BOOT_SEED},
                "trials_before": ST24.TRIALS_BEFORE + ST24.TRIALS_ADDED, "trials_added": 1 if mode == "sealed" else 0,
                "runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")})
    f_sum, f_rep, f_day = (os.path.join(RES, f) for f in MODES[mode]["files"])
    text = report(out, mode)
    summary = json.dumps(out, indent=1, default=C24.jdefault, ensure_ascii=False)
    os.makedirs(RES, exist_ok=True)
    daily.to_csv(f_day, index=False, float_format="%.6f")
    with open(f_rep, "w", encoding="utf-8") as fh:
        fh.write(text)
    with open(f_sum, "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(text)
    if mode == "rehearsal" and not out["checks"]["rehearsal"]["pass"]:
        raise SystemExit("the rehearsal checks failed")
    return json.loads(summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--go", action="store_true")
    g.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    main("sealed" if a.go else "rehearsal", a.workers)
