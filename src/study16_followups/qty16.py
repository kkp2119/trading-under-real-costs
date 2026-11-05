#!/usr/bin/env python3
"""Research 16-D quantity-limited fill check of E* (QTY16.md; implementation details DEVIATIONS16 6) on the research 15
development years 2015-2020 only: fill rule L3 = the L2 trade rule (only trades through our price) and a cross filled up to
the opposite displayed size of the bar, x market depth D (D1 as in 16-C).
  python qty16.py run [--workers 8]   per-parent tables -> $DATA_DIR/research16_data/, aggregates -> results/
Engine: robust16.exec16 with a third fill rule (3 = L3). L0 and L2 are recomputed with the same kernel and must equal the
stored 16-C per-parent output. Costs, D1, the bootstrap and the summary rows come from robust16 unchanged."""
import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robust16 as R16  # noqa: E402
from robust16 import E, L, P, QTY, HOR, TICK, STEP, YEARS, METHODS, COMPARE, NCOL, METHOD_ARGS  # noqa: E402
from robust16 import fill_buy, fill_sell, fill_buy_l2, fill_sell_l2, jit  # noqa: E402

RES, LOCAL = R16.RES, R16.LOCAL
LRULES = (0, 2, 3)
VARIANTS = (("L0D0", 0, 0), ("L2D1", 2, 1), ("L3D0", 3, 0), ("L3D1", 3, 1))
REPRO = ("L0D0", "L2D1")
VERDICT_VARIANT = "L3D1"
HAND_CASE = ("20190528", "AKAM", -1, "15:44:09", "15:44:33")   # the 16-C L2 hand example (QTY16 4)
SOURCES = (("lim", "(가) 지정가"), ("e1", "(나) E1 시장가"), ("dl", "(다) 기한 시장가"))


# ---------------------------------------------------------------- engine
@jit
def fill_buy_l3(price, rem, u, ask_min, bid_max, asz, vol_bid, vol_bid_odd):
    """L3 buy (QTY16 4): the Q-2 volume (every bid-side trade below my price) and, when the ask reaches my price, at most
    the displayed ask size of the bar; the larger of the two, never both."""
    t = 0
    if bid_max[u] > 0 and bid_max[u] < price:                        # 1) trades through my price
        t = vol_bid[u] - vol_bid_odd[u]
        if t < 0:
            t = 0
    if ask_min[u] > 0 and ask_min[u] <= price:                       # 2) cross, with or without a print through
        return min(rem, max(asz[u], t))
    return min(rem, t)


@jit
def fill_sell_l3(price, rem, u, bid_max, ask_min, bsz, vol_ask, vol_ask_odd):
    """Mirror of fill_buy_l3."""
    t = 0
    if ask_min[u] > price:
        t = vol_ask[u] - vol_ask_odd[u]
        if t < 0:
            t = 0
    if bid_max[u] > 0 and bid_max[u] >= price:
        return min(rem, max(bsz[u], t))
    return min(rem, t)


@jit
def exec_q(method, lrule, improve, e1_k, dirs, starts, horizon, qty, bid, ask, bsz, asz, bid_max, bid_min, ask_max, ask_min,
           vol_bid, vol_ask, vol_bid_odd, vol_ask_odd, px_min, px_max, out):
    """robust16.exec16 line for line with the fill rule lrule 3 (L3) added (0 L0, 2 L2 as before)."""
    for i in range(len(dirs)):
        d, s = dirs[i], starts[i]
        D = s + horizon
        lim_sh, lim_not, mkt_sh, mkt_not, over, done, dl = 0, 0, 0, 0, 0, 0, 0
        if method == 0 or method == 1:
            n_sl = 1 if method == 0 else 10
            step = 0 if method == 0 else max(5, (horizon // 10) // 5 * 5)
            for k in range(n_sl):
                u = s + k * step
                q = qty // n_sl
                px = ask[u] if d > 0 else bid[u]
                disp = asz[u] if d > 0 else bsz[u]
                mkt_sh += q
                mkt_not += q * px
                over += max(0, q - disp)
                done = u - s
        else:
            rem, has, px, a4 = qty, False, 0, 0
            for u in range(s, D + 1):
                if u > s and has and rem > 0:
                    if d > 0:
                        if lrule == 0:
                            f, a4 = fill_buy(2, px, rem, a4, u, ask_min, bid_max, bid_min, px_min, vol_bid, vol_bid_odd)
                        elif lrule == 2:
                            f, a4 = fill_buy_l2(px, rem, a4, u, ask_min, bid_max, px_min, vol_bid, vol_bid_odd)
                        else:
                            f = fill_buy_l3(px, rem, u, ask_min, bid_max, asz, vol_bid, vol_bid_odd)
                    else:
                        if lrule == 0:
                            f, a4 = fill_sell(2, px, rem, a4, u, bid_max, ask_min, ask_max, px_max, vol_ask, vol_ask_odd)
                        elif lrule == 2:
                            f, a4 = fill_sell_l2(px, rem, a4, u, bid_max, ask_min, px_max, vol_ask, vol_ask_odd)
                        else:
                            f = fill_sell_l3(px, rem, u, bid_max, ask_min, bsz, vol_ask, vol_ask_odd)
                    if f > 0:
                        lim_sh += f
                        lim_not += f * px
                        rem -= f
                if rem == 0:
                    done = u - s
                    break
                if (u - s) % STEP != 0:
                    continue
                spr = ask[u] - bid[u]
                if u == D or (e1_k >= 0 and spr <= e1_k * TICK):
                    mp_ = ask[u] if d > 0 else bid[u]
                    disp = asz[u] if d > 0 else bsz[u]
                    mkt_sh += rem
                    mkt_not += rem * mp_
                    over += max(0, rem - disp)
                    if u == D:
                        dl = rem
                    rem = 0
                    done = u - s
                    break
                if spr >= TICK:
                    if improve == 1 and spr >= 3 * TICK:
                        npx = bid[u] + TICK if d > 0 else ask[u] - TICK
                        na4 = 0
                    else:
                        npx = bid[u] if d > 0 else ask[u]
                        na4 = 4 * (bsz[u] if d > 0 else asz[u])
                    if not (has and npx == px):
                        has, px, a4 = True, npx, na4
                else:
                    has = False
        out[i, 0], out[i, 1], out[i, 2], out[i, 3], out[i, 4], out[i, 5], out[i, 6] = lim_sh, lim_not, mkt_sh, mkt_not, over, \
            done, dl


def run(b, name, lrule, dirs, starts):
    method, improve, e1_k = METHOD_ARGS[name]
    out = np.zeros((len(dirs), NCOL), dtype=np.int64)
    exec_q(method, lrule, improve, e1_k, dirs, starts, HOR, QTY, b["bid"], b["ask"], b["bsz"], b["asz"], b["bid_max"],
           b["bid_min"], b["ask_max"], b["ask_min"], b["vol_bid"], b["vol_ask"], b["vol_bid_odd"], b["vol_ask_odd"], b["px_min"],
           b["px_max"], out)
    return out


def task(t):
    year, row, close_bar = t
    b = L.load(year, row)
    dirs, starts = E.parents(row["date"], close_bar)
    ok = P.X.quote_ok(b)[starts]
    dirs, starts = dirs[ok], starts[ok]
    outs = np.zeros((len(LRULES), len(METHODS), len(dirs), NCOL), dtype=np.int64)
    for li, lr in enumerate(LRULES):
        for mi, m in enumerate(METHODS):
            outs[li, mi] = run(b, m, lr, dirs, starts)
    meta = {"date": row["date"], "sym": row["sym"], "dir": dirs, "start": starts, "mid2": (b["bid"] + b["ask"])[starts],
            "spread": (b["ask"] - b["bid"])[starts]}
    return year, meta, outs


def simulate(workers):
    tasks = []
    for y in YEARS:
        man, cb = L.manifest(y), L.close_bars(y)
        tasks += [(y, r, cb[d]) for (s, d), r in sorted(man.items())]
    assert all(r["date"] <= "20201231" for _, r, _ in tasks), "date after 2020"
    metas, outs = [], []
    with mp.get_context("spawn").Pool(workers) as pool:
        for y, meta, o in pool.imap(task, tasks, chunksize=8):
            metas.append(meta)
            outs.append(o)
    par = pd.DataFrame({"date": np.concatenate([[m["date"]] * len(m["dir"]) for m in metas]),
                        "sym": np.concatenate([[m["sym"]] * len(m["dir"]) for m in metas]),
                        **{k: np.concatenate([m[k] for m in metas]) for k in ("dir", "start", "mid2", "spread")}})
    OUT = np.concatenate(outs, axis=2)
    os.makedirs(LOCAL, exist_ok=True)
    np.savez_compressed(os.path.join(LOCAL, "qty16_parents.npz"), out=OUT, methods=np.array(METHODS), lrules=np.array(LRULES))
    par.to_csv(os.path.join(LOCAL, "qty16_parents.csv.gz"), index=False)
    return par, OUT, {"symbol_days": len(tasks)}


# ---------------------------------------------------------------- sources of the E* saving (QTY16 4)
def sources(o, is_s0, dirs, mid2, depth):
    """Per-parent contributions (limit, E1 market, deadline market) to IS(S0) - IS(E*), cents per share; they sum to it."""
    lim_sh, lim_not, mkt_sh, mkt_not, over, dl = o[:, 0], o[:, 1], o[:, 2], o[:, 3], o[:, 4], o[:, 6]
    mid0 = mid2 / 2
    lim_cost = dirs * (lim_not - lim_sh * mid0) / 100 / QTY
    mkt_cost = (dirs * (mkt_not + depth * dirs * TICK * over - mkt_sh * mid0) / 100 + E.FEE_MKT * mkt_sh) / QTY
    m = mkt_sh / QTY * is_s0 - mkt_cost
    is_dl = dl > 0
    return {"lim": lim_sh / QTY * is_s0 - lim_cost, "e1": np.where(is_dl, 0.0, m), "dl": np.where(is_dl, m, 0.0)}


# ---------------------------------------------------------------- hand check (written from the QTY16 text, not the kernel)
def fill_case_l3(b, d, px, rem, u):
    """(shares filled, through volume t, opposite displayed size, case) for one resting limit order in bar u under L3."""
    if d > 0:
        cross = 0 < b["ask_min"][u] <= px
        behind = 0 < b["bid_max"][u] < px
        t = max(int(b["vol_bid"][u] - b["vol_bid_odd"][u]), 0) if behind else 0
        opp = int(b["asz"][u])
    else:
        cross = b["bid_max"][u] > 0 and b["bid_max"][u] >= px
        behind = b["ask_min"][u] > px
        t = max(int(b["vol_ask"][u] - b["vol_ask_odd"][u]), 0) if behind else 0
        opp = int(b["bsz"][u])
    if cross:
        return min(rem, max(opp, t)), t, opp, "cross: min(rem, max(opposite size, through volume))"
    if t > 0:
        return min(rem, t), t, opp, "trades through my price (Q-2 volume)"
    return 0, t, opp, "no fill"


def trace_l3(b, d, s, improve, e1_k):
    """One parent through the posting logic (robust16.trace_posting) with the L3 fill."""
    rem, has, px, rows, lim, mkt = QTY, False, 0, [], 0, 0
    for u in range(s, s + HOR + 1):
        if u > s and has and rem > 0:
            f, t, opp, case = fill_case_l3(b, d, px, rem, u)
            rows.append({"u": u, "px": px, "rem_before": rem, "fill": f, "through_vol": t, "opposite_size": opp, "case": case})
            lim += f
            rem -= f
        if rem == 0:
            break
        if (u - s) % STEP:
            continue
        spr = int(b["ask"][u] - b["bid"][u])
        if u == s + HOR or (e1_k >= 0 and spr <= e1_k * TICK):
            rows.append({"u": u, "market": rem, "deadline": u == s + HOR, "spread_ticks": spr / TICK})
            mkt += rem
            rem = 0
            break
        if spr >= TICK:
            npx = (int(b["bid"][u]) + TICK if d > 0 else int(b["ask"][u]) - TICK) if improve and spr >= 3 * TICK else \
                (int(b["bid"][u]) if d > 0 else int(b["ask"][u]))
            if not (has and npx == px):
                has, px = True, npx
        else:
            has = False
    return lim, mkt, rows


def hand_l3(par, OUT):
    date, sym, d, start, at = HAND_CASE
    k = np.flatnonzero((par.date == date).to_numpy() & (par.sym == sym).to_numpy() & (par.dir == d).to_numpy() &
                       np.array([R16.hhmmss(u) == start for u in par.start]))
    if len(k) != 1:
        return {"pass": False, "error": f"{len(k)} parents match {HAND_CASE[:4]}"}
    i = int(k[0])
    b = R16.load_bar(date, sym)
    s = int(par.start.iloc[i])
    _, improve, e1_k = METHOD_ARGS["Estar"]
    lim3, mkt3, rows3 = trace_l3(b, d, s, improve, e1_k)
    lim2, mkt2, rows2 = R16.trace_posting(b, d, s, 2, improve, e1_k)
    je = METHODS.index("Estar")
    eng3 = (int(OUT[LRULES.index(3), je, i, 0]), int(OUT[LRULES.index(3), je, i, 2]))
    eng2 = (int(OUT[LRULES.index(2), je, i, 0]), int(OUT[LRULES.index(2), je, i, 2]))
    r = next((x for x in rows3 if R16.hhmmss(x["u"]) == at and "fill" in x), None)
    shown, rule_ok = None, False
    if r is not None:
        u = r["u"]
        before3 = [(x["u"], x["fill"]) for x in rows3 if x["u"] < u and "fill" in x]
        before2 = [(x["u"], x["fill"]) for x in rows2 if x["u"] < u]
        shown = {"time": at, "our_price": r["px"] / 1e4, "rem_before": r["rem_before"],
                 "bid_min": int(b["bid_min"][u]) / 1e4, "bid_max": int(b["bid_max"][u]) / 1e4,
                 "ask_min": int(b["ask_min"][u]) / 1e4, "ask_max": int(b["ask_max"][u]) / 1e4,
                 "bid_end": int(b["bid"][u]) / 1e4, "bsz_end": int(b["bsz"][u]), "ask_end": int(b["ask"][u]) / 1e4,
                 "asz_end": int(b["asz"][u]), "trade_px_min": int(b["px_min"][u]) / 1e4, "trade_px_max": int(b["px_max"][u]) / 1e4,
                 "round_lot_vol_ask_side": int(b["vol_ask"][u] - b["vol_ask_odd"][u]),
                 "round_lot_vol_bid_side": int(b["vol_bid"][u] - b["vol_bid_odd"][u]),
                 "through_vol_t": r["through_vol"], "opposite_size": r["opposite_size"], "L3_fill": r["fill"], "case": r["case"],
                 "L2_fill": next((x["fill"] for x in rows2 if x["u"] == u), None),
                 "fill_equals_opposite_size": r["fill"] == r["opposite_size"],
                 "paths_identical_before": before3 == before2}
        rule_ok = r["fill"] == min(r["rem_before"], max(r["opposite_size"], r["through_vol"])) and shown["paths_identical_before"]
    after = [{"time": R16.hhmmss(x["u"]), **{k_: v for k_, v in x.items() if k_ not in ("u",)}}
             for x in rows3 if (r is not None and x["u"] >= r["u"]) and (x.get("fill", 1) > 0)]
    match = (lim3, mkt3) == eng3 and (lim2, mkt2) == eng2
    return {"date": date, "sym": sym, "side": "buy" if d > 0 else "sell", "start": start, "parent_index": i,
            "L2": {"limit_sh": lim2, "market_sh": mkt2}, "L3": {"limit_sh": lim3, "market_sh": mkt3},
            "trace_equals_engine": bool(match), "bar": shown, "L3_events_from_bar": after,
            "pass": bool(match and rule_ok)}


# ---------------------------------------------------------------- analysis
def analyse(par, OUT, sim_info, t0):
    rule_sha = R16.sha(os.path.join(HERE, "QTY16.md"))
    days = sorted(par.date.unique())
    codes = pd.Categorical(par.date, categories=days).codes
    idx = R16.boot_index(len(days))
    dirs, mid2 = par.dir.to_numpy(), par.mid2.to_numpy()
    IS = {lab: {m: R16.costs(OUT[LRULES.index(lr), j], dirs, mid2, dep) for j, m in enumerate(METHODS)}
          for lab, lr, dep in VARIANTS}
    yr = par.date.str[:4].to_numpy()

    # checks (QTY16 4)
    z = np.load(os.path.join(LOCAL, "robust16_parents.npz"))
    p16 = pd.read_csv(os.path.join(LOCAL, "robust16_parents.csv.gz"), dtype={"date": str, "sym": str})
    same_par = all(np.array_equal(par[c].to_numpy(), p16[c].to_numpy()) for c in p16.columns)
    same_out = {f"L{lr}": bool(np.array_equal(OUT[LRULES.index(lr)], z["out"][list(z["lrules"]).index(lr)])) for lr in (0, 2)}
    rows = []
    for lab, lr, dep in VARIANTS:
        li = LRULES.index(lr)
        lim = {m: OUT[li, j, :, 0] / QTY for j, m in enumerate(METHODS)}
        dl = {m: OUT[li, j, :, 6] / QTY for j, m in enumerate(METHODS)}
        done = {m: OUT[li, j, :, 5].astype(np.float64) for j, m in enumerate(METHODS)}
        rows += R16.method_rows(IS[lab], lim, dl, done, codes, len(days), idx, lab, "2015-2020")
    old = pd.read_csv(os.path.join(RES, "robust16_summary.csv"))
    new = pd.DataFrame(rows)
    num = [c for c in new.columns if c not in ("period", "variant", "method")]
    repro = {}
    for v in REPRO:
        a = new[new.variant == v].set_index("method")[num]
        b_ = old[(old.period == "2015-2020") & (old.variant == v)].set_index("method").loc[a.index, num]
        mx = float(np.nanmax(np.abs(a.to_numpy(float) - b_.to_numpy(float))))
        repro[v] = {"max_abs_diff": mx, "pass": mx < 5e-4}
    shares_ok, single_mkt, src_ok = True, True, True
    for li in range(len(LRULES)):
        for j, m in enumerate(METHODS):
            o = OUT[li, j]
            shares_ok &= bool(np.all(o[:, 0] + o[:, 2] == QTY))
            if m in ("S2", "S2i", "Estar"):
                single_mkt &= bool(np.all((o[:, 6] == 0) | (o[:, 6] == o[:, 2])))
    src = {}
    je = METHODS.index("Estar")
    for lab, lr, dep in VARIANTS:
        sv = sources(OUT[LRULES.index(lr), je], IS[lab]["S0"], dirs, mid2, dep)
        tot = IS[lab]["S0"] - IS[lab]["Estar"]
        src_ok &= bool(np.max(np.abs(sv["lim"] + sv["e1"] + sv["dl"] - tot)) < 1e-9)
        src[lab] = {"total": R16.diff_ci(tot, codes, len(days), idx)}
        for k_, _ in SOURCES:
            src[lab][k_] = R16.diff_ci(sv[k_], codes, len(days), idx)
    chk = {"rule_sha256": rule_sha, "parents_equal_16c": bool(same_par), "per_parent_out_equal_16c": same_out,
           "summary_reproduction": repro, "shares_sum_to_qty": shares_ok, "one_market_order_per_posting_parent": single_mkt,
           "sources_sum_to_saving": src_ok, "hand_l3": hand_l3(par, OUT), "no_date_after_2020": bool(par.date.max() <= "20201231")}
    ok = (same_par and all(same_out.values()) and all(v["pass"] for v in repro.values()) and shares_ok and single_mkt
          and src_ok and chk["hand_l3"]["pass"] and chk["no_date_after_2020"])
    os.makedirs(RES, exist_ok=True)
    if not ok:
        with open(os.path.join(RES, "qty16_checks_FAILED.json"), "w") as fh:
            json.dump(chk, fh, indent=1, default=str, ensure_ascii=False)
        raise SystemExit("a QTY16 check failed: results/qty16_checks_FAILED.json (no summary written)")

    # summary
    base = {c: next(r for r in rows if r["variant"] == "L0D0" and r["method"] == "Estar")[f"minus_{c}_c"] for c in COMPARE}
    for r in rows:
        if r["method"] == "Estar":
            for c in COMPARE:
                r[f"pct_of_L0D0_saving_vs_{c}"] = 100 * r[f"minus_{c}_c"] / base[c]
            for k_, _ in SOURCES + (("total", ""),):
                r[f"src_{k_}_c"], r[f"src_{k_}_lo"], r[f"src_{k_}_hi"] = src[r["variant"]][k_]
    pd.DataFrame(rows).to_csv(os.path.join(RES, "qty16_summary.csv"), index=False, float_format="%.6f")
    vr = next(r for r in rows if r["variant"] == VERDICT_VARIANT and r["method"] == "Estar")
    verdict = "견고" if vr["minus_S0_hi"] < 0 else "가정 의존"
    yearly = {lab: {y: float((IS[lab]["Estar"] - IS[lab]["S0"])[yr == str(y)].mean()) for y in YEARS} for lab, _, _ in VARIANTS}
    out = {"rule": "QTY16.md", "rule_sha256": rule_sha, "trials_added": 2, "trials_cumulative": 722, "verdict": verdict,
           "verdict_variant": VERDICT_VARIANT,
           "estar_minus": {c: [vr[f"minus_{c}_c"], vr[f"minus_{c}_lo"], vr[f"minus_{c}_hi"]] for c in COMPARE},
           "rows": rows, "sources": src, "yearly_estar_minus_s0": yearly, "checks": chk, "sim": sim_info,
           "n_parents": int(len(par)), "n_days": len(days), "runtime_s": round(time.time() - t0, 1), "python": sys.executable,
           "written": dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(RES, "qty16_result.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str, ensure_ascii=False)
    report(out)
    print(open(os.path.join(RES, "qty16_report.md")).read())


# ---------------------------------------------------------------- report
def report(out):
    R = pd.DataFrame(out["rows"])
    vm = out["estar_minus"]["S0"]
    labs = [v for v, _, _ in VARIANTS]
    L_ = [f"**판정: {out['verdict']}** — {out['verdict_variant']}에서 E* − S0 = {vm[0]:+.3f}¢/주, 95% CI [{vm[1]:+.3f}, {vm[2]:+.3f}] "
          f"(상한 {'< 0' if vm[2] < 0 else '≥ 0'})", "",
          "# 연구 16-D · E* 체결 수량 제한 점검 (개발 구간 2015–2020)", "",
          f"- 규칙 QTY16.md sha256 `{out['rule_sha256']}` (2026-10-30 18:00 ET 작성, 새 시뮬레이션 전)",
          f"- 시행 수 +2, 누적 {out['trials_cumulative']} (견고성 점검, 방법을 새로 고르지 않음)",
          f"- 주문 {out['n_parents']:,}건, {out['n_days']}일, 종목·날 {out['sim']['symbol_days']:,}개, 1,000주 / 300초, 비용 R",
          "- **L3** = L2의 체결 기록 규칙(우리 가격을 뚫고 지나간 정규 단위 체결량까지) + 교차 시 그 초 반대쪽 최우선 물량까지만 체결"
          "(둘 중 큰 값). 교차는 뚫고 지나간 체결이 없어도 인정한다(QTY16 4절). 그래서 L3는 L2 안에 포함되지 않는다.", "",
          "## 표 1. 평균 IS (¢/주, R) · 괄호는 95번째 백분위", "",
          "| 변형 | " + " | ".join(METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
    for v in labs:
        r = R[R.variant == v].set_index("method")
        L_.append(f"| {v} | " + " | ".join(f"{r.loc[m, 'mean_is_c']:.3f} ({r.loc[m, 'p95_is_c']:.2f})" for m in METHODS) + " |")
    L_ += ["", "## 표 2. E* − 비교 대상 (¢/주, 음수 = E*가 쌈) · 95% CI (일 블록 21, 10,000회, seed 16) · L0D0 대비 남은 절감액 %", "",
           "| 변형 | E* − S0 | E* − S1 | E* − S2i | 남은 % (S0 / S1 / S2i) |", "|---|---|---|---|---|"]
    for v in labs:
        e = R[(R.variant == v) & (R.method == "Estar")].iloc[0]
        L_.append(f"| {v} | " + " | ".join(f"{e[f'minus_{c}_c']:+.3f} [{e[f'minus_{c}_lo']:+.3f}, {e[f'minus_{c}_hi']:+.3f}]"
                                          for c in COMPARE) +
                  " | " + " / ".join(f"{e[f'pct_of_L0D0_saving_vs_{c}']:.0f}%" for c in COMPARE) + " |")
    L_ += ["", "## 표 3. E* 절감액(S0 − E*)의 출처 (¢/주, 합계 대비 %, 95% CI)", "",
           "| 변형 | 합계 | " + " | ".join(n for _, n in SOURCES) + " |", "|---" * (len(SOURCES) + 2) + "|"]
    for v in labs:
        s = out["sources"][v]
        tot = s["total"][0]
        L_.append(f"| {v} | {tot:.3f} [{s['total'][1]:.3f}, {s['total'][2]:.3f}] | " +
                  " | ".join(f"{s[k][0]:+.3f} ({100 * s[k][0] / tot:.0f}%) [{s[k][1]:+.3f}, {s[k][2]:+.3f}]" for k, _ in SOURCES) + " |")
    L_ += ["", "- 주식 하나하나를 S0의 주당 비용과 비교해 나눴다(QTY16 4절). 세 기여의 합은 절감액과 같다.",
           "", "## 표 4. 처리 방식 (지정가 비율 / 기한 시장가 비율 / 평균 완료 초)", "",
           "| 변형 | " + " | ".join(METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
    for v in labs:
        r = R[R.variant == v].set_index("method")
        L_.append(f"| {v} | " + " | ".join(f"{r.loc[m, 'limit_frac']:.2f} / {r.loc[m, 'deadline_mkt_frac']:.2f} / "
                                          f"{r.loc[m, 'mean_done_s']:.0f}" for m in METHODS) + " |")
    L_ += ["", "## 연도별 E* − S0 (¢/주)", "", "| 변형 | " + " | ".join(str(y) for y in YEARS) + " |",
           "|---" * (len(YEARS) + 1) + "|"]
    for v in labs:
        L_.append(f"| {v} | " + " | ".join(f"{out['yearly_estar_minus_s0'][v][y]:+.3f}" for y in YEARS) + " |")
    c = out["checks"]
    h = c["hand_l3"]
    f = h["bar"]
    L_ += ["", "## 대조 점검", "",
           f"- **재현 (주문별):** 주문 표가 16-C와 같음 {c['parents_equal_16c']}, L0·L2 출력이 16-C 저장 출력과 정수 단위로 같음 "
           f"{c['per_parent_out_equal_16c']}",
           "- **재현 (요약):** " + ", ".join(f"{v} 최대 차이 {x['max_abs_diff']:.1e} ({'통과' if x['pass'] else '실패'})"
                                          for v, x in c["summary_reproduction"].items()),
           f"- 지정가 + 시장가 = 1,000주 {c['shares_sum_to_qty']}, 대기형 주문의 시장가는 한 번 {c['one_market_order_per_posting_parent']}, "
           f"출처 합 = 절감액 {c['sources_sum_to_saving']}",
           f"- **L3 손 계산** ({h['sym']} {h['date']} {h['start']} 시작, E* {'매수' if h['side'] == 'buy' else '매도'}):",
           f"  - 엔진과 별도로 Python으로 다시 계산했다. 지정가·시장가 주식은 L2 {h['L2']['limit_sh']}/{h['L2']['market_sh']}, "
           f"L3 {h['L3']['limit_sh']}/{h['L3']['market_sh']}이고, 엔진과 같음: {h['trace_equals_engine']}"]
    if f:
        L_ += [f"  - {f['time']}: 우리 매도가 ${f['our_price']:.2f}, 남은 수량 {f['rem_before']:,}주. 시장 매수 호가 범위 ${f['bid_min']:.2f}–"
               f"${f['bid_max']:.2f}(교차), 매도 호가 최저 ${f['ask_min']:.2f}, 최고 체결가 ${f['trade_px_max']:.2f}(뚫고 지나간 체결 없음).",
               f"  - 그 초 끝 최우선 매수 호가 ${f['bid_end']:.2f}에 보이는 물량 {f['bsz_end']:,}주, 뚫고 지나간 체결량 t = {f['through_vol_t']:,}주.",
               f"  - L3 체결 = min({f['rem_before']:,}, max({f['opposite_size']:,}, {f['through_vol_t']:,})) = **{f['L3_fill']:,}주**. "
               f"반대쪽 물량과 같음: {f['fill_equals_opposite_size']}. (L0 1,000주, L2 {f['L2_fill']}주)",
               f"  - 그 전까지 L2와 경로가 같다: {f['paths_identical_before']}",
               "  - 그 뒤 L3 사건: " + "; ".join(
                   f"{e['time']} " + (f"지정가 {e['fill']}주 ({e['case']})" if "fill" in e else
                                      f"시장가 {e['market']}주 ({'기한' if e['deadline'] else 'E1, 스프레드 %.0f틱' % e['spread_ticks']})")
                   for e in h["L3_events_from_bar"])]
    L_ += [f"- 2021년 이후 날짜 없음: {c['no_date_after_2020']}", "",
           "## 비고", "",
           "- IS (R) = 방향 × (평균 체결가 − 도착 중간값) + 0.3¢ × 시장가 주식 / 1,000 (exec15.costs). D1은 넘는 주식 × 1¢ (16-C와 같음).",
           "- 반대쪽 물량은 그 초 끝 NBBO 크기다(교차한 순간의 크기는 막대에 없음).",
           f"- 실행 {out['runtime_s']} s, {out['python']}"]
    with open(os.path.join(RES, "qty16_report.md"), "w") as fh:
        fh.write("\n".join(L_) + "\n")


def main(workers):
    t0 = time.time()
    with open(os.path.join(HERE, "QTY16.sha256")) as fh:
        if fh.read().split()[0] != R16.sha(os.path.join(HERE, "QTY16.md")):
            raise SystemExit("QTY16.md differs from QTY16.sha256")
    par, OUT, info = simulate(workers)
    print(f"simulated {len(par):,} parents, {info['symbol_days']:,} symbol-days, {time.time() - t0:.0f} s", flush=True)
    analyse(par, OUT, info, t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.workers)
