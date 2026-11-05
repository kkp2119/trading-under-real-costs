#!/usr/bin/env python3
"""Research 16-C robustness check of E* (ROBUST16.md; implementation details DEVIATIONS16.md 4): limit fill rule L x market
depth D on the research 15 development years 2015-2020 only (loader15 cannot reach 2021+). The sealed 2021-2026 period is
only summarised from the stored test15 per-parent file (L0D0, no new simulation).
  python robust16.py run [--workers 8]   per-parent tables -> $DATA_DIR/research16_data/, aggregates -> results/
Engine: the exec15 / r2_15 parent-order logic in one kernel with a fill-rule argument (0 = L0, the research 12 Q model of
fillmodel15 as is; 2 = L2, only trades through our price fill us). L1 needs exchange codes, which the 1-second bars do not
have, so L1D0 and L1D1 are skipped. D1 is applied afterwards from the market shares above the displayed size."""
import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R15 = os.path.join(os.path.dirname(HERE), "study15_ml_execution")
sys.path.insert(0, R15)
import exec15 as E  # noqa: E402
import loader15 as L  # noqa: E402
import policy15 as P  # noqa: E402
import r2_15 as R2  # noqa: E402
from fillmodel15 import fill_buy, fill_sell, jit  # noqa: E402

RES = os.path.join(HERE, "results")
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research16_data")
TEST15 = os.path.join(os.environ.get("DATA_DIR", "data"), "research15_data/test15")
TEST15_SUMMARY = os.path.join(R15, "results15", "test15", "test15_summary.json")
S15_SUMMARY = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "study15", "s15_exec_summary.json")
QTY, HOR = 1000, 300
TICK, STEP = E.TICK, E.STEP
YEARS = tuple(range(2015, 2021))
METHODS = ("S0", "S1", "S2", "S2i", "Estar")
LRULES = (0, 2)
VARIANTS = (("L0D0", 0, 0), ("L0D1", 0, 1), ("L2D0", 2, 0), ("L2D1", 2, 1))
SKIPPED = ("L1D0", "L1D1")
MOST_CONSERVATIVE = "L2D1"
COMPARE = ("S0", "S1", "S2i")
ESTAR_NAME, ESTAR = "E12|E2off|E3off|E4off", (2, 0, 0, -1)    # r2_15 e1_k, e2, e3, e4_x
BOOT_B, BOOT_BLOCK, BOOT_SEED = 10_000, 21, 16
NCOL = 7                                               # lim_sh, lim_not, mkt_sh, mkt_not, over, done (s), deadline sh


# ---------------------------------------------------------------- engine
@jit
def fill_buy_l2(price, rem, a4, u, ask_min, bid_max, px_min, vol_bid, vol_bid_odd):
    """L2 buy (DEVIATIONS16 4): only trades strictly below my price fill me."""
    if ask_min[u] > 0 and ask_min[u] <= price and px_min[u] > 0 and px_min[u] < price:   # Q-1 with a print through
        return rem, a4
    credit = vol_bid[u] - vol_bid_odd[u]
    if credit < 0:
        credit = 0
    if bid_max[u] > 0 and bid_max[u] < price:                        # Q-2: every bid-side trade is below my price
        return min(rem, credit), 0
    return 0, a4                                                     # Q-3 (trades at my price) and Q-4: no fill


@jit
def fill_sell_l2(price, rem, a4, u, bid_max, ask_min, px_max, vol_ask, vol_ask_odd):
    """Mirror of fill_buy_l2."""
    if bid_max[u] > 0 and bid_max[u] >= price and px_max[u] > price:
        return rem, a4
    credit = vol_ask[u] - vol_ask_odd[u]
    if credit < 0:
        credit = 0
    if ask_min[u] > price:
        return min(rem, credit), 0
    return 0, a4


@jit
def exec16(method, lrule, improve, e1_k, dirs, starts, horizon, qty, bid, ask, bsz, asz, bid_max, bid_min, ask_max, ask_min,
           vol_bid, vol_ask, vol_bid_odd, vol_ask_odd, px_min, px_max, out):
    """exec15.exec_day (method 0 S0, 1 S1, 2 posting) and r2_15.exec_rules (posting with E1 at e1_k ticks) with the fill
    rule lrule (0 L0, 2 L2). out[i] = (limit shares, limit notional, market shares, market notional, market shares above
    the displayed size, seconds from the start to the last share, shares sent at market at the deadline)."""
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
                        else:
                            f, a4 = fill_buy_l2(px, rem, a4, u, ask_min, bid_max, px_min, vol_bid, vol_bid_odd)
                    else:
                        if lrule == 0:
                            f, a4 = fill_sell(2, px, rem, a4, u, bid_max, ask_min, ask_max, px_max, vol_ask, vol_ask_odd)
                        else:
                            f, a4 = fill_sell_l2(px, rem, a4, u, bid_max, ask_min, px_max, vol_ask, vol_ask_odd)
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


METHOD_ARGS = {"S0": (0, 0, -1), "S1": (1, 0, -1), "S2": (2, 0, -1), "S2i": (2, 1, -1), "Estar": (2, 1, ESTAR[0])}


def run(b, name, lrule, dirs, starts):
    method, improve, e1_k = METHOD_ARGS[name]
    out = np.zeros((len(dirs), NCOL), dtype=np.int64)
    exec16(method, lrule, improve, e1_k, dirs, starts, HOR, QTY, b["bid"], b["ask"], b["bsz"], b["asz"], b["bid_max"],
           b["bid_min"], b["ask_max"], b["ask_min"], b["vol_bid"], b["vol_ask"], b["vol_bid_odd"], b["vol_ask_odd"], b["px_min"],
           b["px_max"], out)
    return out


def costs(out, dirs, mid2, depth):
    """IS (R), cents per share: exec15.costs; depth 1 (D1) prices market shares above the displayed size one tick worse."""
    mkt_not = out[..., 3] + depth * dirs * TICK * out[..., 4]
    avg = (out[..., 1] + mkt_not) / QTY
    return dirs * (avg - mid2 / 2) / 100 + E.FEE_MKT * out[..., 2] / QTY


def reference(b, dirs, starts):
    """The five L0 methods through the research 15 code as is (exec15.run_method, r2_15.run_rules)."""
    n = len(b["bid"])
    T_, F_ = np.ones(n, np.bool_), np.zeros(n, np.bool_)
    return [E.run_method(b, 0, dirs, starts, HOR, QTY, F_, F_, F_, F_, 0),
            E.run_method(b, 1, dirs, starts, HOR, QTY, F_, F_, F_, F_, 0),
            E.run_method(b, 2, dirs, starts, HOR, QTY, T_, T_, F_, F_, 0),
            E.run_method(b, 2, dirs, starts, HOR, QTY, T_, T_, F_, F_, 1),
            R2.run_rules(b, dirs, starts, HOR, QTY, *ESTAR)]


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
    same = all(np.array_equal(outs[0, mi, :, :5], r) for mi, r in enumerate(reference(b, dirs, starts)))
    meta = {"date": row["date"], "sym": row["sym"], "dir": dirs, "start": starts, "mid2": (b["bid"] + b["ask"])[starts],
            "spread": (b["ask"] - b["bid"])[starts], "n_dropped": int((~ok).sum())}
    return year, meta, outs, same


# ---------------------------------------------------------------- statistics
def boot_index(T, B=BOOT_B, block=BOOT_BLOCK, seed=BOOT_SEED):
    """run_test15.boot_ci resampling (circular blocks of days) with seed 16, as an index matrix B x T."""
    rng = np.random.default_rng(seed)
    k = math.ceil(T / block)
    return (rng.integers(0, T, size=(B, k))[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :T] % T


def diff_ci(d, date_codes, n_days, idx):
    """Per-parent mean of d with the 95% interval of the day-block bootstrap (ratio of day sums to day counts)."""
    s = np.bincount(date_codes, weights=d, minlength=n_days)
    n = np.bincount(date_codes, minlength=n_days).astype(np.float64)
    keep = n > 0
    s, n = s[keep], n[keep]
    stat = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(s.sum() / n.sum()), float(lo), float(hi)


def method_rows(IS, lim, dl, done, codes, n_days, idx, label, period):
    """Summary rows (one per method) for one variant: IS (R) mean and p95, differences with CIs, execution shares."""
    rows = []
    for m in METHODS:
        r = {"period": period, "variant": label, "method": m, "n_parents": int(len(IS[m])), "mean_is_c": float(IS[m].mean()),
             "p95_is_c": float(np.percentile(IS[m], 95))}
        for c in COMPARE:
            est, lo, hi = diff_ci(IS[m] - IS[c], codes, n_days, idx)
            r.update({f"minus_{c}_c": est, f"minus_{c}_lo": lo, f"minus_{c}_hi": hi})
        r.update({"limit_frac": float(lim[m].mean()),
                  "deadline_mkt_frac": float(dl[m].mean()) if dl is not None else float("nan"),
                  "mean_done_s": float(done[m].mean()) if done is not None else float("nan")})
        rows.append(r)
    return rows


# ---------------------------------------------------------------- hand checks (written from the rule text, not the kernel)
def hhmmss(u):
    s = 9 * 3600 + 30 * 60 + int(u)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def fill_case(b, d, px, rem, q, u, lrule):
    """(shares filled, queue ahead after, case) for one resting limit order in bar u, in shares."""
    if d > 0:
        cross = 0 < b["ask_min"][u] <= px
        through = 0 < b["px_min"][u] < px
        credit = max(int(b["vol_bid"][u] - b["vol_bid_odd"][u]), 0)
        away = 0 < b["bid_max"][u] < px
        at = b["bid_min"][u] == px and b["bid_max"][u] == px
    else:
        cross = b["bid_max"][u] > 0 and b["bid_max"][u] >= px
        through = b["px_max"][u] > px
        credit = max(int(b["vol_ask"][u] - b["vol_ask_odd"][u]), 0)
        away = b["ask_min"][u] > px
        at = b["ask_min"][u] == px and b["ask_max"][u] == px
    if cross and (lrule == 0 or through):
        return rem, q, "Q-1 (cross)"
    if away:
        return min(rem, credit), 0, "Q-2 (market quote behind us)"
    if at:
        if lrule == 2:
            return 0, q, "Q-3, L2: same-price trades do not fill"
        if credit > q:
            return min(rem, credit - q), 0, "Q-3 (queue used up)"
        return 0, q - credit, "Q-3 (queue shrinking)"
    return 0, q, "Q-4" + (" / cross without a print through" if cross else "")


def trace_posting(b, d, s, lrule, improve, e1_k):
    """One parent through the posting logic (exec15 / r2_15 text): (limit shares, market shares, rows per resting bar)."""
    rem, has, px, q, rows, lim, mkt = QTY, False, 0, 0, [], 0, 0
    for u in range(s, s + HOR + 1):
        if u > s and has and rem > 0:
            f, q2, case = fill_case(b, d, px, rem, q, u, lrule)
            rows.append({"u": u, "px": px, "queue_before": q, "fill": f, "case": case})
            q = q2
            lim += f
            rem -= f
        if rem == 0:
            break
        if (u - s) % STEP:
            continue
        spr = int(b["ask"][u] - b["bid"][u])
        if u == s + HOR or (e1_k >= 0 and spr <= e1_k * TICK):
            mkt += rem
            rem = 0
            break
        if spr >= TICK:
            if improve and spr >= 3 * TICK:
                npx, nq = (int(b["bid"][u]) + TICK, 0) if d > 0 else (int(b["ask"][u]) - TICK, 0)
            else:
                npx, nq = (int(b["bid"][u]), int(b["bsz"][u])) if d > 0 else (int(b["ask"][u]), int(b["asz"][u]))
            if not (has and npx == px):
                has, px, q = True, npx, nq
        else:
            has = False
    return lim, mkt, rows


def load_bar(date, sym):
    y = int(date[:4])
    return L.load(y, L.manifest(y)[(sym, date)])


def hand_d1(par, OUT, IS):
    rng = np.random.default_rng(BOOT_SEED)
    j0 = METHODS.index("S0")
    cand = np.flatnonzero(OUT[0, j0, :, 4] > 0)
    i = int(rng.choice(cand))
    p = par.iloc[i]
    b = load_bar(p.date, p.sym)
    d, s = int(p.dir), int(p.start)
    disp = int(b["asz"][s] if d > 0 else b["bsz"][s])
    px = int(b["ask"][s] if d > 0 else b["bid"][s])
    over = max(0, QTY - disp)
    eng = float(IS["L0D1"]["S0"][i] - IS["L0D0"]["S0"][i])
    return {"date": p.date, "sym": p.sym, "side": "buy" if d > 0 else "sell", "time": hhmmss(s), "best_price": px / 1e4,
            "displayed": disp, "shares_over": over, "penalty_usd": over * 0.01, "penalty_c_per_share": over / QTY,
            "engine_is_d1_minus_d0_c": eng, "candidates_with_over": int(len(cand)),
            "pass": bool(abs(eng - over / QTY) < 1e-9 and over == int(OUT[0, j0, i, 4]))}


def hand_l2(par, OUT):
    """rng 16 order over the E* parents whose L0 and L2 limit shares differ; the first whose first differing bar is an L0
    fill with no L2 fill is shown (a first difference can also be a smaller L2 fill, e.g. a cross without a print through
    followed by Q-2)."""
    rng = np.random.default_rng(BOOT_SEED)
    je = METHODS.index("Estar")
    cand = np.flatnonzero(OUT[0, je, :, 0] != OUT[1, je, :, 0])
    _, improve, e1_k = METHOD_ARGS["Estar"]
    tried = 0
    for i in rng.permutation(cand)[:50]:
        tried += 1
        p = par.iloc[int(i)]
        b = load_bar(p.date, p.sym)
        d, s = int(p.dir), int(p.start)
        lim0, mkt0, rows0 = trace_posting(b, d, s, 0, improve, e1_k)
        lim2, mkt2, rows2 = trace_posting(b, d, s, 2, improve, e1_k)
        key = lambda r: (r["u"], r["px"], r["fill"])                                     # noqa: E731
        k = next((n for n, (r0, r2) in enumerate(zip(rows0, rows2)) if key(r0) != key(r2)), None)
        if k is not None and rows0[k]["fill"] > 0 and rows2[k]["fill"] == 0:
            break
    i = int(i)
    match = (lim0, mkt0) == (int(OUT[0, je, i, 0]), int(OUT[0, je, i, 2])) and \
            (lim2, mkt2) == (int(OUT[1, je, i, 0]), int(OUT[1, je, i, 2]))
    shown = None
    if k is not None and rows0[k]["fill"] > 0 and rows2[k]["fill"] == 0:
        u, r0, r2 = rows0[k]["u"], rows0[k], rows2[k]
        same_before = all(key(a) == key(c) for a, c in zip(rows0[:k], rows2[:k]))
        side = ("bid", "ask") if d > 0 else ("ask", "bid")
        shown = {"time": hhmmss(u), "our_price": r0["px"] / 1e4, "queue_before": r0["queue_before"],
                 f"market_{side[0]}_min": int(b[f"{side[0]}_min"][u]) / 1e4, f"market_{side[0]}_max": int(b[f"{side[0]}_max"][u]) / 1e4,
                 f"opposite_{side[1]}_extreme": int(b["ask_min"][u] if d > 0 else b["bid_max"][u]) / 1e4,
                 "trade_px_min": int(b["px_min"][u]) / 1e4, "trade_px_max": int(b["px_max"][u]) / 1e4,
                 "round_lot_volume_our_side": int(b["vol_bid"][u] - b["vol_bid_odd"][u]) if d > 0 else
                 int(b["vol_ask"][u] - b["vol_ask_odd"][u]),
                 "L0_fill": r0["fill"], "L0_case": r0["case"], "L2_fill": r2["fill"], "L2_case": r2["case"],
                 "paths_identical_before": bool(same_before)}
    return {"date": p.date, "sym": p.sym, "side": "buy" if d > 0 else "sell", "start": hhmmss(s), "candidates": int(len(cand)),
            "parents_tried": tried,
            "L0": {"limit_sh": lim0, "market_sh": mkt0}, "L2": {"limit_sh": lim2, "market_sh": mkt2},
            "trace_equals_engine": bool(match), "first_L0_only_fill": shown,
            "pass": bool(match and shown is not None and shown["paths_identical_before"])}


# ---------------------------------------------------------------- run
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def simulate(workers):
    tasks = []
    for y in YEARS:
        man, cb = L.manifest(y), L.close_bars(y)
        tasks += [(y, r, cb[d]) for (s, d), r in sorted(man.items())]
    assert all(r["date"] <= "20201231" for _, r, _ in tasks), "date after 2020"
    metas, outs, same = [], [], []
    with mp.get_context("spawn").Pool(workers) as pool:
        for y, meta, o, sm in pool.imap(task, tasks, chunksize=8):
            metas.append(meta)
            outs.append(o)
            same.append(sm)
    par = pd.DataFrame({"date": np.concatenate([[m["date"]] * len(m["dir"]) for m in metas]),
                        "sym": np.concatenate([[m["sym"]] * len(m["dir"]) for m in metas]),
                        **{k: np.concatenate([m[k] for m in metas]) for k in ("dir", "start", "mid2", "spread")}})
    OUT = np.concatenate(outs, axis=2)
    os.makedirs(LOCAL, exist_ok=True)
    np.savez_compressed(os.path.join(LOCAL, "robust16_parents.npz"), out=OUT, methods=np.array(METHODS),
                        lrules=np.array(LRULES))
    par.to_csv(os.path.join(LOCAL, "robust16_parents.csv.gz"), index=False)
    return par, OUT, {"symbol_days": len(tasks), "symbol_days_differ_from_r15_code": int(sum(not x for x in same)),
                      "dropped_parents": int(sum(m["n_dropped"] for m in metas))}


def sealed_summary():
    """ROBUST16 4: test15 per-parent results (stored), L0D0, the 3절 methods only."""
    z = np.load(os.path.join(TEST15, "test15_parents.npz"))
    par = pd.read_csv(os.path.join(TEST15, "test15_parents.csv.gz"), dtype={"date": str, "sym": str})
    ms = list(z["methods"])
    tag = f"{QTY}_{HOR}"
    IS = {m: z[f"is_r|{tag}"][:, ms.index(m)].astype(np.float64) for m in METHODS}
    lim = {m: z[f"lim|{tag}"][:, ms.index(m)].astype(np.float64) for m in METHODS}
    days = sorted(par.date.unique())
    codes = pd.Categorical(par.date, categories=days).codes
    idx = boot_index(len(days))
    rows = method_rows(IS, lim, None, None, codes, len(days), idx, "L0D0", "2021-2026 (test15 stored)")
    yr = par.date.str[:4].to_numpy()
    yearly = {y: float((IS["Estar"] - IS["S0"])[yr == y].mean()) for y in sorted(set(yr))}
    orig = json.load(open(TEST15_SUMMARY))["primary"]
    return rows, yearly, {"methods_in_file": ms, "n_parents": int(len(par)), "n_days": len(days),
                          "test15_primary_seed15": orig, "S1_in_test15": "S1" in ms}


def analyse(par, OUT, sim_info, t0):
    rule_sha = sha(os.path.join(HERE, "ROBUST16.md"))
    days = sorted(par.date.unique())
    codes = pd.Categorical(par.date, categories=days).codes
    idx = boot_index(len(days))
    dirs, mid2 = par.dir.to_numpy(), par.mid2.to_numpy()
    IS = {lab: {m: costs(OUT[LRULES.index(lr), j], dirs, mid2, dep) for j, m in enumerate(METHODS)}
          for lab, lr, dep in VARIANTS}
    yr = par.date.str[:4].to_numpy()

    # checks (ROBUST16 5)
    s15 = json.load(open(S15_SUMMARY))["by_year"]
    rep = {}
    for y in YEARS:
        k = yr == str(y)
        a, e = float(IS["L0D0"]["S0"][k].mean()), float(IS["L0D0"]["Estar"][k].mean())
        ref = s15[str(y)]
        rep[y] = {"n": int(k.sum()), "n_s15": ref["n_orders"], "S0": a, "S0_s15": ref["IS_S0_mean"], "Estar": e,
                  "Estar_s15": ref["IS_Estar_mean"],
                  "pass": int(k.sum()) == ref["n_orders"] and round(a, 3) == round(ref["IS_S0_mean"], 3)
                  and round(e, 3) == round(ref["IS_Estar_mean"], 3)}
    r2diff = 0.0
    for y in YEARS:                                                  # per parent against the stored R2 run (float32)
        _, a, ms = R2.load(y)
        k = yr == str(y)
        for m in METHODS:
            col = ms.index(ESTAR_NAME if m == "Estar" else m)
            r2diff = max(r2diff, float(np.abs(a["is_r"][:, col].astype(np.float64) - IS["L0D0"][m][k]).max()))
    chk = {"rule_sha256": rule_sha, "reproduction_by_year": rep, "max_abs_diff_vs_stored_r2_per_parent_c": r2diff,
           "symbol_days_differ_from_r15_code": sim_info["symbol_days_differ_from_r15_code"],
           "hand_d1": hand_d1(par, OUT, IS), "hand_l2": hand_l2(par, OUT),
           "no_date_after_2020": bool(par.date.max() <= "20201231")}
    ok = (all(v["pass"] for v in rep.values()) and chk["symbol_days_differ_from_r15_code"] == 0 and chk["hand_d1"]["pass"]
          and chk["hand_l2"]["pass"] and chk["no_date_after_2020"])
    os.makedirs(RES, exist_ok=True)
    if not ok:
        with open(os.path.join(RES, "robust16_checks_FAILED.json"), "w") as fh:
            json.dump(chk, fh, indent=1, default=str, ensure_ascii=False)
        raise SystemExit("a ROBUST16 5 check failed: results/robust16_checks_FAILED.json (no summary written)")

    # summary
    rows = []
    for lab, lr, dep in VARIANTS:
        li = LRULES.index(lr)
        lim = {m: OUT[li, j, :, 0] / QTY for j, m in enumerate(METHODS)}
        dl = {m: OUT[li, j, :, 6] / QTY for j, m in enumerate(METHODS)}
        done = {m: OUT[li, j, :, 5].astype(np.float64) for j, m in enumerate(METHODS)}
        rows += method_rows(IS[lab], lim, dl, done, codes, len(days), idx, lab, "2015-2020")
    yearly = {lab: {y: float((IS[lab]["Estar"] - IS[lab]["S0"])[yr == str(y)].mean()) for y in YEARS} for lab, _, _ in VARIANTS}
    s_rows, s_yearly, s_info = sealed_summary()
    summ = pd.DataFrame(rows + s_rows + [{"period": "2015-2020", "variant": v, "method": "all",
                                          "status": "skipped: no exchange code in the 1-second bars"} for v in SKIPPED])
    summ["status"] = summ.get("status").fillna("computed")
    summ.to_csv(os.path.join(RES, "robust16_summary.csv"), index=False, float_format="%.6f")
    mc = next(r for r in rows if r["variant"] == MOST_CONSERVATIVE and r["method"] == "Estar")
    verdict = "견고" if mc["minus_S0_hi"] < 0 else "가정 의존"
    out = {"rule": "ROBUST16.md", "rule_sha256": rule_sha, "trials_added": 6, "trials_cumulative": 720,
           "verdict": verdict, "most_conservative": MOST_CONSERVATIVE, "skipped": list(SKIPPED),
           "estar_minus": {c: [mc[f"minus_{c}_c"], mc[f"minus_{c}_lo"], mc[f"minus_{c}_hi"]] for c in COMPARE},
           "rows": rows, "yearly_estar_minus_s0": yearly, "sealed": {"rows": s_rows, "yearly_estar_minus_s0": s_yearly, **s_info},
           "checks": chk, "sim": sim_info, "n_parents": int(len(par)), "n_days": len(days),
           "runtime_s": round(time.time() - t0, 1), "python": sys.executable,
           "written": dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(RES, "robust16_result.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str, ensure_ascii=False)
    report(out)
    print(open(os.path.join(RES, "robust16_report.md")).read())


# ---------------------------------------------------------------- report
ENGINE_TEXT = [
    "- **(a) 지정가 체결 정보:** fillmodel15 `fill_buy`(Q 모델)는 1초 막대의 `ask_min`(그 초 최저 매도 호가), `bid_min`·`bid_max`"
    "(그 초 시장 매수 호가 범위), `vol_bid − vol_bid_odd`(시장 매수 호가 이하 가격의 정규 단위 체결량)를 쓴다(매도는 대칭).",
    "  - **같은 가격:** 그 초 내내 시장 매수 호가 = 우리 가격(Q-3)이면, 그 가격 이하 체결량이 줄 앞 물량을 먼저 줄이고 넘는 만큼만 우리에게 체결된다. 1틱 안쪽 주문은 줄 앞이 0이다.",
    "  - **뚫고 지나감:** 시장 매수 호가가 그 초 내내 우리 가격 아래면(Q-2) 그 초의 매수 쪽 체결량만큼 체결된다(줄 무시). 최저 매도 호가가 우리 가격 이하로 내려오면(Q-1) 체결 기록 없이도 남은 수량 전부가 체결된다.",
    "  - **호가가 움직인 초:** 그 초 안에 시장 호가가 움직이면(Q-4) 체결이 없다.",
    "- **(b) 장외·호가 안쪽 체결:**",
    "  - 막대의 체결 쿼리는 거래소 코드(ex)를 받지 않고 거래소 필터도 없다. 그래서 TRF(D) 체결이 `vol_bid`·`vol_ask`에 섞여 판정에 들어가고, 구분할 수 없다 → L1은 건너뛴다.",
    "  - 호가 안쪽 가격 체결(시장 매수 호가 < 가격 < 시장 매도 호가)은 `vol_in`으로 따로 집계되어 판정에 들어가지 않는다.",
    "  - 단주('I')는 줄을 움직이지 않는다.",
    "  - 각 체결의 매수·매도 쪽은 체결 시각 직전 NBBO로 정한다.",
    "- **(c) 시장가가 보이는 물량보다 클 때 (D0):** 결정 초의 최우선 호가(매수는 ask)에 전량 체결된다. 넘는 주식 수는 `over`로 기록만 하고 비용에 넣지 않는다.",
]


def report(out):
    R = pd.DataFrame(out["rows"])
    mc = out["estar_minus"]["S0"]
    L_ = [f"**판정: {out['verdict']}** — 가장 보수적인 변형 {out['most_conservative']}에서 E* − S0 = {mc[0]:+.3f}¢/주, "
          f"95% CI [{mc[1]:+.3f}, {mc[2]:+.3f}] (상한 {'< 0' if mc[2] < 0 else '≥ 0'})", "",
          "# 연구 16-C · E* 견고성 점검 (개발 구간 2015–2020)", "",
          f"- 규칙 ROBUST16.md sha256 `{out['rule_sha256']}` (2026-10-30 15:46 ET 작성, 새 시뮬레이션 전)",
          f"- 시행 수 +6, 누적 {out['trials_cumulative']} (견고성 점검, 방법을 새로 고르지 않음)",
          f"- 주문 {out['n_parents']:,}건, {out['n_days']}일, 종목·날 {out['sim']['symbol_days']:,}개, 1,000주 / 300초, 비용 R",
          "- **L1D0·L1D1은 건너뜀:** 1초 막대에 거래소 코드가 없음(정보 없음). 그래서 가장 보수적인 변형은 L2D1이다.", "",
          "## 엔진 확인 (코드)", "", *ENGINE_TEXT, "",
          "## 표 1. 평균 IS (¢/주, R) · 괄호는 95번째 백분위", "",
          "| 변형 | " + " | ".join(METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
    for v, _, _ in VARIANTS:
        r = R[R.variant == v].set_index("method")
        L_.append(f"| {v} | " + " | ".join(f"{r.loc[m, 'mean_is_c']:.3f} ({r.loc[m, 'p95_is_c']:.2f})" for m in METHODS) + " |")
    L_ += ["", "## 표 2. E* − 비교 대상 (¢/주, 음수 = E*가 쌈) · 95% CI (일 블록 21, 10,000회, seed 16)", "",
           "| 변형 | E* − S0 | E* − S1 | E* − S2i |", "|---|---|---|---|"]
    for v, _, _ in VARIANTS:
        e = R[(R.variant == v) & (R.method == "Estar")].iloc[0]
        L_.append(f"| {v} | " + " | ".join(f"{e[f'minus_{c}_c']:+.3f} [{e[f'minus_{c}_lo']:+.3f}, {e[f'minus_{c}_hi']:+.3f}]"
                                          for c in COMPARE) + " |")
    L_ += ["", "## 표 3. 처리 방식 (지정가 비율 / 기한 시장가 비율 / 평균 완료 초)", "",
           "| 변형 | " + " | ".join(METHODS) + " |", "|---" * (len(METHODS) + 1) + "|"]
    for v, _, _ in VARIANTS:
        r = R[R.variant == v].set_index("method")
        L_.append(f"| {v} | " + " | ".join(f"{r.loc[m, 'limit_frac']:.2f} / {r.loc[m, 'deadline_mkt_frac']:.2f} / "
                                          f"{r.loc[m, 'mean_done_s']:.0f}" for m in METHODS) + " |")
    L_ += ["", "## 연도별 E* − S0 (¢/주)", "", "| 변형 | " + " | ".join(str(y) for y in YEARS) + " |",
           "|---" * (len(YEARS) + 1) + "|"]
    for v, _, _ in VARIANTS:
        L_.append(f"| {v} | " + " | ".join(f"{out['yearly_estar_minus_s0'][v][y]:+.3f}" for y in YEARS) + " |")
    s = out["sealed"]
    S = pd.DataFrame(s["rows"]).set_index("method")
    o = s["test15_primary_seed15"]
    L_ += ["", "## 봉인 구간 2021–2026: 저장된 test15 결과 요약 (L0D0, 새 시뮬레이션 없음)", "",
           f"- 파일의 방법: {', '.join(s['methods_in_file'])} · 주문 {s['n_parents']:,}건, {s['n_days']}일. "
           f"S1 있음: {s['S1_in_test15']}. S3는 3절 방법이 아니라 제외했다.",
           "- 기한 시장가 비율과 완료 시간은 저장돼 있지 않다(저장 안 됨).", "",
           "| 방법 | 평균 IS | p95 | − S0 [CI] | − S1 [CI] | − S2i [CI] | 지정가 비율 |", "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        r = S.loc[m]
        L_.append(f"| {m} | {r.mean_is_c:.3f} | {r.p95_is_c:.2f} | " +
                  " | ".join(f"{r[f'minus_{c}_c']:+.3f} [{r[f'minus_{c}_lo']:+.3f}, {r[f'minus_{c}_hi']:+.3f}]" for c in COMPARE) +
                  f" | {r.limit_frac:.2f} |")
    L_ += ["", "연도별 E* − S0: " + ", ".join(f"{y} {v:+.3f}" for y, v in s["yearly_estar_minus_s0"].items()),
           f"(test15 원래 주 결과, seed 15: {o['estimate']:+.4f} [{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}])", "",
           "## 대조 점검", ""]
    c = out["checks"]
    L_.append("| 연도 | 주문 (s15) | S0 (s15) | E* (s15) | 결과 |")
    L_.append("|---|---|---|---|---|")
    for y, v in c["reproduction_by_year"].items():
        L_.append(f"| {y} | {v['n']:,} ({v['n_s15']:,}) | {v['S0']:.3f} ({v['S0_s15']:.3f}) | {v['Estar']:.3f} ({v['Estar_s15']:.3f}) | "
                  f"{'통과' if v['pass'] else '실패'} |")
    d1, l2 = c["hand_d1"], c["hand_l2"]
    L_ += ["", f"- 주문별로 저장된 R2 결과와 비교한 최대 차이: {c['max_abs_diff_vs_stored_r2_per_parent_c']:.2e}¢. "
               f"연구 15 원본 코드와 출력이 다른 종목·날: {c['symbol_days_differ_from_r15_code']}개",
           f"- **D1 손 계산** ({d1['sym']} {d1['date']} {d1['time']}, S0 {'매수' if d1['side'] == 'buy' else '매도'}):",
           f"  - 최우선 {'매도' if d1['side'] == 'buy' else '매수'} 호가 ${d1['best_price']:.2f}에 보이는 물량은 {d1['displayed']:,}주다.",
           f"  - 넘는 주식은 1,000 − {d1['displayed']:,} = {d1['shares_over']:,}주이고, 벌점은 ${d1['penalty_usd']:.2f} = {d1['penalty_c_per_share']:.3f}¢/주다.",
           f"  - 엔진의 IS 차이는 {d1['engine_is_d1_minus_d0_c']:.3f}¢로 {'일치' if d1['pass'] else '불일치'}한다.",
           f"- **L2 손 계산** ({l2['sym']} {l2['date']} {l2['start']} 시작, E* {'매수' if l2['side'] == 'buy' else '매도'}):",
           f"  - 엔진과 별도로 Python으로 다시 계산했다. 지정가·시장가 주식은 L0 {l2['L0']['limit_sh']}/{l2['L0']['market_sh']}, "
           f"L2 {l2['L2']['limit_sh']}/{l2['L2']['market_sh']}이고, 엔진과 같음: {l2['trace_equals_engine']}"]
    f = l2["first_L0_only_fill"]
    if f:
        L_ += [f"  - L0에서만 체결되는 첫 초는 {f['time']}다. 우리 가격 ${f['our_price']:.2f}, 줄 앞 {f['queue_before']:,}주에서 시작했다.",
               "  - 그 초의 막대 값: " + ", ".join(f"{k} {v}" for k, v in f.items() if k not in ("time", "our_price", "queue_before", "L0_fill",
                                                                                     "L0_case", "L2_fill", "L2_case", "paths_identical_before")),
               f"  - L0: {f['L0_fill']}주 체결({f['L0_case']}). L2: {f['L2_fill']}주({f['L2_case']}).",
               f"  - 그 전까지 두 경로는 같다: {f['paths_identical_before']}"]
    L_ += [f"- 2021년 이후 날짜 없음: {c['no_date_after_2020']}", "",
           "## 비고", "",
           "- IS (R) = 방향 × (평균 체결가 − 도착 중간값) + 0.3¢ × 시장가 주식 / 1,000 (exec15.costs).",
           "- D1은 같은 시뮬레이션에 넘는 주식 × 1¢를 더해 구했다. 결정이 체결 가격에 의존하지 않으므로 정확하다.",
           "- L2 구현은 DEVIATIONS16.md 4절에 있다. 막대에서 같은 가격과 그 아래 체결을 나눌 수 없어, Q-3 초의 체결은 모두 같은 가격으로 보았다(보수적).",
           f"- 실행 {out['runtime_s']} s, {out['python']}"]
    with open(os.path.join(RES, "robust16_report.md"), "w") as fh:
        fh.write("\n".join(L_) + "\n")


def main(workers):
    t0 = time.time()
    with open(os.path.join(HERE, "ROBUST16.sha256")) as fh:
        if fh.read().split()[0] != sha(os.path.join(HERE, "ROBUST16.md")):
            raise SystemExit("ROBUST16.md differs from ROBUST16.sha256")
    par, OUT, info = simulate(workers)
    print(f"simulated {len(par):,} parents, {info['symbol_days']:,} symbol-days, {time.time() - t0:.0f} s", flush=True)
    analyse(par, OUT, info, t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.workers)
