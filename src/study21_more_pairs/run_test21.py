#!/usr/bin/env python3
"""Research 21 sealed test (SPEC21 5-6, SPEC21_ADDENDUM C1-C2; step-2 prompt 4), to be run once:
  python run_test21.py --go           sealed pairs 2021-01-04..2026-11-10
  python run_test21.py --rehearsal    the same code path on the development data (pairs of 2015-2020)
The sealed run refuses unless study21_more_pairs/UNLOCK21.txt holds exactly PHRASE (written by Kyle; Claude never creates the
file), no test result exists, every file listed in FREEZE21.md has its sha256, SPEC21.md, SPEC21_ANSWERS.md and
SPEC21_ADDENDUM.md match their sha256 files and the engine files of research 20 equal tag code-freeze-20. The guard runs
before any sealed path is built. The development selection (each family's A* and whether it has an H2) is read from
results/dev21_summary.json, which FREEZE21.md pins. The rehearsal never builds a sealed path.
Verdict (Holm over the hypotheses left, overall alpha 0.05; one-sided p = share of day-block bootstrap means <= 0, blocks
of 21 days, 10,000 resamples, seed 21):
  H1-E, H1-S  the mean of the daily sums of the mid P&L of the signal-only trades, setting W30_z2_k0, is > 0
  H2-E, H2-S  the mean daily P&L of the family's A* (market orders, L3D1) is > 0; only for a family whose A* was
              profitable in development
Sentences: SPEC21 6, as written. Secondary, no verdict (SPEC21 5, SPEC21_ADDENDUM): the delayed mid P&L (1, 5, 30 s) and
its 5-second sentence per family, L0D0, pairs and years, pnl = mid - cost - fee per trade, the comparison run C2
(W30_z2_k0 with market orders), each family's A* setting also when it has no H2, the control pairs C1 (mid P&L and delayed
table), every run split at 2025-11-03 (quote-size unit), the side-by-side table with research 20, and the pair-by-pair
daily P&L of every run. Results are written once at the end."""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core21 as C21  # noqa: E402
import loader21 as L21  # noqa: E402
import stats21 as ST21  # noqa: E402

C, S, ST20 = C21.C, C21.S, C21.ST20
RES = os.path.join(HERE, "results")
UNLOCK = os.path.join(HERE, "UNLOCK21.txt")
PHRASE = "연구21 테스트를 한 번 연다"
FREEZE = os.path.join(HERE, "FREEZE21.md")
DEV_SUMMARY = os.path.join(RES, "dev21_summary.json")
SEALED_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research21_data/sealed")
SEALED_MANIFEST = os.path.join(HERE, "SEALED_MANIFEST21.tsv")
WINDOW = ("20210104", "20261110")                    # the sealed part ends on the last collected day
DOCS = ("SPEC21", "SPEC21_ANSWERS", "SPEC21_ADDENDUM")
H1 = ST21.sig_name(ST21.H1_SIGNAL)
MODES = {"sealed": {"files": ("test21_summary.json", "test21_report.md", "test21_daily.csv"), "title": "봉인 테스트 (쌍 2021–2026)", "r20": "sealed"},
         "rehearsal": {"files": ("rehearsal21_summary.json", "rehearsal21_report.md", "rehearsal21_daily.csv"),
                       "title": "리허설 (개발 데이터: 쌍 2015–2020, 판정 아님)", "r20": "dev"}}
CORE = ("run_test21.py", "core21.py", "loader21.py", "stats21.py", "../study20_pairs/core20.py", "../study20_pairs/pair20.py",
        "../study20_pairs/signal20.py", "../study20_pairs/stats20.py", "../study17_new_stocks/engine17.py", "SEALED_MANIFEST21.tsv", "calendar21.csv",
        "results/universe21_pairs.tsv", "results/dev21_summary.json", "../study20_pairs/results/test20_summary.json") + tuple(
            f"universe21_{y}.tsv" for y in range(2021, 2027))
SIDE_LABELS = {"R20": "같은 회사 (연구 20)", "E": "ETF (가족 E)", "S": "같은 업종 (가족 S)", "C": "대조 쌍 (C1)"}
RUN_LABELS = {"mid": "중간값 (지연 0초)", "mid_delay1": "중간값, 지연 1초", "mid_delay5": "중간값, 지연 5초", "mid_delay30": "중간값, 지연 30초",
              "M_L3D1": "시장가, L3D1", "M_L0D0": "시장가, L0D0"}


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
    """{family: {"astar", "signal" (W, z_in, kappa), "h2"}} from the development summary; refuses anything outside the grid."""
    d = json.load(open(DEV_SUMMARY))
    sel = {}
    for f in ST21.FAMILIES:
        F = d["families"][f]
        sig = (int(F["astar_signal"][0]), float(F["astar_signal"][1]), float(F["astar_signal"][2]))
        if sig not in ST21.candidates() or F["astar"] != ST21.cand_name(sig) or not isinstance(F["h2_exists"], bool):
            raise Refused(f"family {f}: the development selection is not a candidate of the grid")
        if F["h2_exists"] != (F["selection"]["astar_mean_usd"] > 0):
            raise Refused(f"family {f}: H2 does not follow from the A* mean")
        sel[f] = {"astar": F["astar"], "signal": sig, "h2": F["h2_exists"]}
    return sel


def frozen_files():
    """[(path relative to study21_more_pairs, sha256)] from the FREEZE21.md table rows | `path` | `sha256` |."""
    if not os.path.isfile(FREEZE):
        return []
    pat = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
    return [m.groups() for m in map(pat.match, open(FREEZE, encoding="utf-8")) if m]


def guard():
    why = []
    if any(os.path.exists(os.path.join(RES, f)) for f in MODES["sealed"]["files"]):
        why.append("test results already exist")
    if not os.path.isfile(UNLOCK) or open(UNLOCK, encoding="utf-8").read().strip() != PHRASE:
        why.append("no unlock phrase in UNLOCK21.txt")
    ff = frozen_files()
    if not ff or not all(any(p == c for p, _ in ff) for c in CORE):
        why.append("FREEZE21.md missing or without the core files")
    for p, h in ff:
        q = os.path.normpath(os.path.join(HERE, p))
        if not os.path.isfile(q) or sha(q) != h:
            why.append(f"frozen file differs: {p}")
    for doc in DOCS:
        f = os.path.join(HERE, f"{doc}.sha256")
        if not os.path.isfile(f) or open(f).read().split()[0] != sha(os.path.join(HERE, f"{doc}.md")):
            why.append(f"{doc}.md differs from {doc}.sha256")
    if C21.frozen_differences():
        why.append("engine files differ from tag code-freeze-20")
    try:
        selection()
    except Exception as e:                                   # a missing or unreadable development summary included
        why.append(f"no usable development selection: {e}")
    return why


# ---------------------------------------------------------------- analysis
def analyse(A, D, Ac, Dc, es, ct, days, cfgs, sel, r20_part):
    n_days = len(days)
    idx = ST21.boot_index(n_days)
    out = {"h1_signal": list(ST21.H1_SIGNAL), "families": {}, "hypotheses": {}, "verdict": {}}
    hyp, rows = {}, []
    for f in ST21.FAMILIES:
        M, blocks, Af, Df, tf = C21.family_tables(A, D, es, days, cfgs, idx, f)
        a_sig = ST21.sig_name(sel[f]["signal"])
        keep = {name: blocks[name] for name in (H1, a_sig)}                  # the family's own two settings
        split = {name: {run: C21.date_split(*C21.run_series(Af, Df, M, cfgs.index(tuple(keep[name]["signal"])), run, n_days), days)
                        for run in C21.RUNS} for name in keep}
        out["families"][f] = {"x": ST21.FAMILY_X[f], "astar": sel[f]["astar"], "astar_signal": list(sel[f]["signal"]), "h2_exists": sel[f]["h2"],
                              "settings": keep, "date_split": split, "pair_days": len(tf), "pairs": M["names"]}
        m = blocks[H1]["rows"]["mid"]
        hyp[f"H1-{f}"] = {"series": f"daily sum of mid P&L, family {f}, setting {H1}", "mean_daily_usd": m["mean_daily_usd"],
                          "p_one_sided": m["p_one_sided"], "lower_2_5": m["lower_2_5"], "ci95": [m["ci_lo"], m["ci_hi"]], "per_trade_usd": m["per_trade"]}
        a = blocks[a_sig]["rows"]["M_L3D1"]
        hyp[f"H2-{f}"] = {"series": f"daily P&L of {sel[f]['astar']}, family {f}, L3D1", "mean_daily_usd": a["mean_daily_usd"],
                          "p_one_sided": a["p_one_sided"], "lower_2_5": a["lower_2_5"], "ci95": [a["ci_lo"], a["ci_hi"]]}
        for name in keep:
            rows += C21.pair_daily(Af, Df, tf, cfgs.index(tuple(keep[name]["signal"])), name)
    names = ST21.hypotheses({f: sel[f]["h2"] for f in ST21.FAMILIES})
    out["hypotheses"] = {h: hyp[h] for h in names}
    holm = ST21.holm({h: hyp[h]["p_one_sided"] for h in names})
    for h in names:
        ok = holm[h]["rejected"]
        out["verdict"][h] = {**holm[h], "result": "확인" if ok else "확인 안 됨", "mean_daily_usd": hyp[h]["mean_daily_usd"],
                             "lower_2_5": hyp[h]["lower_2_5"], "sentence": ST21.sentence(h[:2], h[-1], ok)}
    for f in ST21.FAMILIES:
        if not sel[f]["h2"]:
            out["verdict"][f"H2-{f}"] = {"result": "없음", "sentence": ST21.no_h2_sentence(f)}
    Mc, cb, Acf, Dcf, tcf = C21.family_tables(Ac, Dc, ct, days, [ST21.H1_SIGNAL], idx, L21.CONTROL, mid_only=True)
    years = sorted({t["year"] for t in es + ct})
    out["control"] = {"pairs_by_year": {str(y): [f"{a}/{b}" for _, a, b in L21.controls(y)] for y in years}, "pair_days": len(tcf),
                      "pairs": Mc["names"], **cb[H1],
                      "date_split": {run: C21.date_split(*C21.run_series(Acf, Dcf, Mc, 0, run, n_days), days) for run in C21.RUNS if run.startswith("mid")}}
    rows += C21.pair_daily(Acf, Dcf, tcf, 0, H1)
    side = [dict(C21.r20_side(r20_part), key="R20")] + [dict(C21.side_row(f, out["families"][f]["settings"][H1]), key=f) for f in ST21.FAMILIES]
    side.append(dict(C21.side_row(L21.CONTROL, cb[H1]), key=L21.CONTROL))
    for r in side:
        r["label"] = SIDE_LABELS[r["key"]] + ((", 봉인 2021–2026" if r20_part == "sealed" else ", 개발 2015–2020") if r["key"] == "R20" else "")
    out["side_by_side"] = side
    r5 = {f: out["families"][f]["settings"][H1]["delayed"]["5"]["ratio_to_delay0"] for f in ST21.FAMILIES}
    out["ratio_5s"] = {**r5, "E_greater_than_S": bool(r5["E"] > r5["S"])}
    last = {f: {name: {k: r["exit_last"] for k, r in b["rows"].items()} for name, b in F["settings"].items()} for f, F in out["families"].items()}
    last[L21.CONTROL] = {H1: {"mid": out["control"]["rows"]["mid"]["exit_last"]}}
    out["counts"] = {"pair_days": len(es), "control_pair_days": len(ct), "days": n_days, "first_day": days[0], "last_day": days[-1],
                     "trades_valued_at_last_valid_mids": last}
    daily = pd.DataFrame(rows, columns=C21.DAILY_COLS)
    daily["trades_M"] = daily["trades_M"].astype("Int64")               # empty for the control pairs
    return out, daily


# ---------------------------------------------------------------- rehearsal checks (the development numbers of dev21.py)
def rehearsal_checks(out, read):
    dev = json.load(open(DEV_SUMMARY))
    d_rows, d_del, n, sent = [], [], 0, True
    for f, F in out["families"].items():
        for name, b in F["settings"].items():
            old = dev["families"][f]["settings"][name]
            for k, r in b["rows"].items():
                d_rows += C21.row_diffs(r, old["rows"][k])
                n += 1
            d_del += C21.delayed_diffs(b["delayed"], old["delayed"])
            sent &= b["sentence_5s"] == old["sentence_5s"]
    d_ctl = C21.row_diffs(out["control"]["rows"]["mid"], dev["control"]["rows"]["mid"]) + C21.delayed_diffs(out["control"]["delayed"], dev["control"]["delayed"])
    d_side = [C21.diff(a[k], b[k]) for a, b in zip(out["side_by_side"], dev["side_by_side"]) for k in ("mid_per_trade", "ratio_5s", "trades_per_day")]
    d_side += [C21.diff(a[v][k], b[v][k]) for a, b in zip(out["side_by_side"], dev["side_by_side"]) for v in C21.VARS if a[v] is not None
               for k in ("cost_per_trade", "pnl_per_trade")]
    sel = selection()
    chk = {"rows_compared": n, "dev21_max_abs_diff": float(max(d_rows)), "delayed_max_abs_diff": float(max(d_del)),
           "control_max_abs_diff": float(max(d_ctl)), "side_by_side_max_abs_diff": float(max(d_side)), "sentences_5s_equal": bool(sent),
           "hypotheses_equal": list(out["hypotheses"]) == dev["hypotheses"],
           "selection_equal": all(out["families"][f]["astar"] == dev["families"][f]["astar"] and sel[f]["h2"] == dev["families"][f]["h2_exists"]
                                  for f in ST21.FAMILIES),
           "control_pairs_equal": out["control"]["pairs_by_year"] == dev["control"]["pairs_by_year"],
           "no_sealed_path_read": not any("sealed" in os.path.abspath(p).split(os.sep) for p in read)}
    chk["dev21_numbers_equal"] = all(chk[k] == 0.0 for k in ("dev21_max_abs_diff", "delayed_max_abs_diff", "control_max_abs_diff",
                                                              "side_by_side_max_abs_diff"))
    chk["pass"] = bool(chk["dev21_numbers_equal"] and chk["sentences_5s_equal"] and chk["hypotheses_equal"] and chk["selection_equal"]
                       and chk["control_pairs_equal"] and chk["no_sealed_path_read"])
    return chk


# ---------------------------------------------------------------- report
def ci(r):
    return f"{r['mean_daily_usd']:+,.1f} [{r['ci_lo']:+,.1f}, {r['ci_hi']:+,.1f}]"


def pct(x):
    return "—" if x is None or x != x else f"{100 * x:.1f}%"


def split_cell(p):
    return "—" if p is None else f"{ci(p)} ({p['days']:,}일, 거래 {p['trades']:,})"


def split_lines(split, rows, delayed):
    """One line per run: the whole window, the days to 2025-10-31 and the days from 2025-11-03."""
    L_ = ["| 실행 | 전체 | 2025-10-31까지 | 2025-11-03부터 |", "|---|---|---|---|"]
    for run, sp in split.items():
        whole = delayed[run[9:]] if run.startswith("mid_delay") else rows[run]
        L_.append(f"| {RUN_LABELS[run]} | {ci(whole)} | {split_cell(sp['before'])} | {split_cell(sp['from'])} |")
    return L_


def report(out, mode, path):
    c = out["counts"]
    L_ = [f"# 연구 21 · {MODES[mode]['title']}", "",
          f"- 쌍·일 {c['pair_days']:,}개(가족 E·S)와 대조 쌍·일 {c['control_pair_days']:,}개, {c['days']:,}일({c['first_day']} ~ {c['last_day']}).",
          f"- 판정: 부트스트랩 단측 p(평균이 0 이하인 비율, 일 블록 {out['boot']['block']}, {out['boot']['B']:,}회, seed {out['boot']['seed']})에 "
          f"Holm(가설 {len(out['hypotheses'])}개, 전체 α 0.05). 하한은 단측 2.5%. 손익 단위는 달러(다리마다 5만 달러).",
          "- 2025-11-03부터의 호가 잔량은 100으로 나눠 읽었다(SPEC21 3절).", ""]
    if mode == "rehearsal":
        L_ += ["**리허설: 개발 데이터라서 아래 \"확인\" 여부와 문장은 코드 경로 점검용이며 판정이 아니다.**", ""]
    L_ += ["## 판정", "", "| 가설 | 계열 | 평균 (달러/일) | 단측 2.5% 하한 | p | Holm 수준 | 결과 | 문장 |", "|---|---|---|---|---|---|---|---|"]
    for h, x in out["verdict"].items():
        if "p" in x:
            L_.append(f"| {h} | {out['hypotheses'][h]['series']} | {x['mean_daily_usd']:+,.1f} | {x['lower_2_5']:+,.1f} | {x['p']:.4f} | {x['level']:.4f} | "
                      f"{x['result']} | \"{x['sentence']}\" |")
        else:
            L_.append(f"| {h} | — | — | — | — | — | {x['result']} | \"{x['sentence']}\" |")
    L_ += [""]
    for f, F in out["families"].items():
        pt = out["hypotheses"][f"H1-{f}"]["per_trade_usd"]
        L_ += [f"## 가족 {f}: {F['x']} (보조, 판정 아님)", "",
               f"- 쌍·일 {F['pair_days']:,}개. A* = {F['astar']}" + ("" if F["h2_exists"] else " (개발에서 수익 없음: H2 없음, 아래 A* 값은 참고)") + ".",
               f"- H1 거래당 중간값 손익: {pt['mean']:+.2f}달러 [{pt['lo']:+.2f}, {pt['hi']:+.2f}]",
               f"- 5초 문장: \"{F['settings'][H1]['sentence_5s']}\""]
        for name, b in F["settings"].items():
            R = b["rows"]
            yrs, pn = sorted(R["mid"]["by_year"]), F["pairs"]
            tag = "H1 설정, 시장가 실행은 비교용 C2" if name == H1 else "A*의 신호 설정"
            L_ += ["", f"### 가족 {f} · 신호 설정 {name} ({tag})", "",
                   "| 실행 | 평균 일 손익 [95% CI] | 샤프 | 거래 수 | 하루 | 승률 | 보유(초) | 거래당 손익 | = 중간값 | − 비용 | − 수수료 |",
                   "|---|---|---|---|---|---|---|---|---|---|---|"]
            for k, r in R.items():
                L_.append(f"| {RUN_LABELS[k]} | {ci(r)} | {r['sharpe']:+.2f} | {r['trades']:,} | {r['trades_per_day']:.1f} | {r['win_rate']:.3f} | "
                          f"{r['mean_hold_s']:.0f} | {r['pnl_per_trade']:+.2f} | {r['mid_per_trade']:+.2f} | {r['cost_per_trade']:.2f} | {r['fee_per_trade']:.2f} |")
            L_ += ["", "| 지연 | 거래당 평균 (달러) | 일 합 평균 [95% CI] | 지연 0초 대비 | 승률 |", "|---|---|---|---|---|"]
            for d, r in b["delayed"].items():
                L_.append(f"| {d}초 | {r['per_trade_usd']:+.2f} | {ci(r)} | {pct(r['ratio_to_delay0'])} | {pct(r['win_rate'])} |")
            L_ += ["", "| 연도별 평균 일 손익 | " + " | ".join(yrs) + " |", "|---" * (len(yrs) + 1) + "|"]
            L_ += [f"| {RUN_LABELS[k]} | " + " | ".join(f"{r['by_year'][y]:+,.1f}" for y in yrs) + " |" for k, r in R.items()]
            L_ += ["", "| 쌍별 (쌍·일당 달러) | " + " | ".join(pn) + " |", "|---" * (len(pn) + 1) + "|"]
            L_ += [f"| {RUN_LABELS[k]} | " + " | ".join(f"{r['by_pair'][p]['per_pair_day_usd']:+,.1f}" for p in pn) + " |" for k, r in R.items()]
            L_.append("| 쌍·일 수 | " + " | ".join(f"{R['mid']['by_pair'][p]['pair_days']:,}" for p in pn) + " |")
            L_ += ["", f"잔량 단위가 바뀐 날로 나눈 값 (평균 일 손익 [95% CI], 구간마다 따로 부트스트랩):", ""]
            L_ += split_lines(F["date_split"][name], R, b["delayed"])
        L_.append("")
    ctl = out["control"]
    m, pt = ctl["rows"]["mid"], ctl["rows"]["mid"]["per_trade"]
    L_ += ["## 대조 쌍 (SPEC21_ADDENDUM C1, 판정 아님)", "", "| 연도 | 대조 쌍 |", "|---|---|"]
    L_ += [f"| {y} | {', '.join(p) if p else '없음'} |" for y, p in ctl["pairs_by_year"].items()]
    L_ += ["", f"- 쌍·일 {ctl['pair_days']:,}개, 거래 {m['trades']:,}건. 거래당 중간값 손익 {pt['mean']:+.2f}달러 [{pt['lo']:+.2f}, {pt['hi']:+.2f}], "
           f"일 합 평균 {ci(m)}, 단측 p {m['p_one_sided']:.4f}.", f"- 5초 문장(참고): \"{ctl['sentence_5s']}\"", "",
           "| 지연 | 거래당 평균 (달러) | 일 합 평균 [95% CI] | 지연 0초 대비 | 승률 |", "|---|---|---|---|---|"]
    for d, r in ctl["delayed"].items():
        L_.append(f"| {d}초 | {r['per_trade_usd']:+.2f} | {ci(r)} | {pct(r['ratio_to_delay0'])} | {pct(r['win_rate'])} |")
    L_ += [""] + split_lines(ctl["date_split"], ctl["rows"], ctl["delayed"])

    def cell(x, k, f):
        return "—" if x is None else f.format(x[k])
    L_ += ["", f"## 나란히 놓은 표 ({H1}, 시장가)", "",
           "| 쌍의 종류 | 거래당 중간값 손익 | 5초 지연 뒤 남는 비율 | 거래당 비용 L3D1 | 비용 차감 후 L3D1 | 거래당 비용 L0D0 | 비용 차감 후 L0D0 | 하루 평균 거래 수 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in out["side_by_side"]:
        L_.append(f"| {r['label']} | {r['mid_per_trade']:+.2f} | {pct(r['ratio_5s'])} | {cell(r['L3D1'], 'cost_per_trade', '{:.2f}')} | "
                  f"{cell(r['L3D1'], 'pnl_per_trade', '{:+.2f}')} | {cell(r['L0D0'], 'cost_per_trade', '{:.2f}')} | "
                  f"{cell(r['L0D0'], 'pnl_per_trade', '{:+.2f}')} | {r['trades_per_day']:.1f} |")
    L_ += ["- 거래당 비용은 스프레드·체결 비용과 수수료의 합이다. 대조 쌍은 중간값 손익과 지연 표만 계산한다(추가 문서 C1).",
           "- 연구 20의 값은 저장된 결과에서 옮겼다." + (" 연구 20의 봉인 실행은 2025-11-03 이후 잔량을 보정하지 않아, 그 L3D1 비용은 그만큼 낮게 잡혀 있다."
                                           if mode == "sealed" else ""),
           f"- 5초 지연 뒤 남는 비율: E {pct(out['ratio_5s']['E'])}, S {pct(out['ratio_5s']['S'])} → E가 S보다 "
           + ("크다." if out["ratio_5s"]["E_greater_than_S"] else "크지 않다.")]
    sim = out["sim"]
    lastv = sum(v for fam in c["trades_valued_at_last_valid_mids"].values() for st in fam.values() for v in st.values())
    L_ += ["", "## 자료 관련 건수", "",
           f"- 유효 격자 시점 비율 {sim['valid_grid_points'] / max(sim['grid_points'], 1):.4f}(대조 {sim['control']['valid_grid_points'] / max(sim['control']['grid_points'], 1):.4f}).",
           f"- 마지막 유효 중간값으로 평가한 거래: {lastv}건. 첫 호가 뒤에 호가 상태가 없는 막대가 있어 뺀 쌍·일: "
           f"{sim['pair_days_left_out_no_state_after_first_state'] + sim['control']['pair_days_left_out_no_state_after_first_state']}개.",
           f"- 파일이 없어 쌍의 거래일에서 뺀 날(가족별): {sim['skipped_pair_days']}"]
    if "checks" in out:
        k = out["checks"]
        L_ += ["", "## 리허설 점검", "", f"- 개발 실행(dev21)의 숫자와 최대 차이: 표 {k['dev21_max_abs_diff']:.1e}(줄 {k['rows_compared']}개), 지연 표 "
               f"{k['delayed_max_abs_diff']:.1e}, 대조 쌍 {k['control_max_abs_diff']:.1e}, 나란히 놓은 표 {k['side_by_side_max_abs_diff']:.1e} → "
               f"{'일치' if k['dev21_numbers_equal'] else '불일치'}",
               f"- 5초 문장 같음 {k['sentences_5s_equal']}, 가설 목록 같음 {k['hypotheses_equal']}, A*와 H2 유무 같음 {k['selection_equal']}, "
               f"대조 쌍 명단 같음 {k['control_pairs_equal']}, 봉인 경로를 읽지 않음 {k['no_sealed_path_read']}."]
    L_ += ["", f"- 쌍별 일 손익: results/{MODES[mode]['files'][2]}. 읽은 막대 폴더: {out['read']['bars_root']}. 실행 {out['runtime_s']} s"]
    with open(path, "w") as fh:
        fh.write("\n".join(L_) + "\n")


# ---------------------------------------------------------------- run
def main(mode, workers):
    t0 = time.time()
    if mode == "sealed":
        why = guard()
        if why:
            raise SystemExit("refused: " + "; ".join(why))
        root, allow = os.path.join(SEALED_ROOT, "bars_1s"), True
        tasks, days, skipped = L21.plan(L21.manifest(SEALED_MANIFEST), WINDOW)
        read = [SEALED_MANIFEST, L21.CALENDAR, root]
    else:
        if any(os.path.exists(os.path.join(RES, f)) for f in MODES[mode]["files"]):
            raise SystemExit("rehearsal results already exist: rename them first (nothing is overwritten)")
        root, allow = L21.DEV_ROOT, False
        tasks, days, skipped = L21.dev_plan()
        read = [L21.DEV_MANIFEST, L21.CALENDAR, root]
    sel = selection()
    cfgs = [ST21.H1_SIGNAL]
    for f in ST21.FAMILIES:
        if sel[f]["signal"] not in cfgs:
            cfgs.append(sel[f]["signal"])
    es = [t for t in tasks if t["family"] in ST21.FAMILIES]
    ct = [t for t in tasks if t["family"] == L21.CONTROL]
    A, info, ex = C21.simulate(es, root, allow, cfgs, delay=0, workers=workers, mid_delays=S.MID_DELAYS)
    if info["settings_with_m_trades_unlike_the_signal_timetable"]:
        raise SystemExit("the M trades differ from the signal-only timetable (nothing written)")
    Ac, infoc, exc = C21.simulate(ct, root, allow, [ST21.H1_SIGNAL], delay=0, workers=workers, mid_delays=S.MID_DELAYS, mid_only=True)
    out, daily = analyse(A, ex["mid_delayed"], Ac, exc["mid_delayed"], es, ct, days, cfgs, sel, MODES[mode]["r20"])
    out["sim"] = {**info, "control": infoc, "skipped_pair_days": skipped, "workers": workers}
    out["read"] = {"bars_root": root.replace(os.path.expanduser("~"), "~"), "other": [os.path.relpath(p, HERE) for p in read[:2]]}
    if mode == "rehearsal":
        out["checks"] = rehearsal_checks(out, read)
    out.update({"mode": mode, "boot": {"B": ST21.BOOT_B, "block": ST21.BOOT_BLOCK, "seed": ST21.BOOT_SEED},
                "runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")})
    f_sum, f_rep, f_day = (os.path.join(RES, f) for f in MODES[mode]["files"])
    daily.to_csv(f_day, index=False, float_format="%.4f")
    with open(f_sum, "w") as fh:
        json.dump(out, fh, indent=1, default=str, ensure_ascii=False)
    report(out, mode, f_rep)
    print(open(f_rep).read())
    if mode == "rehearsal" and not out["checks"]["pass"]:
        raise SystemExit("the rehearsal checks failed")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--go", action="store_true")
    g.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    main("sealed" if a.go else "rehearsal", a.workers)
