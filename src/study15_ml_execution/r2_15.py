#!/usr/bin/env python3
"""Research 15 experiment R2 (Kyle 2026-10-29; DEVIATIONS15 23): refine the S2i execution rule without machine learning,
2015-2020 (fixed rules, no training, so the training years carry no leak). 2021+ data is never read.
Base S2i (exec15): post the remaining quantity at the best quote, one tick inside when the spread >= 3 ticks (alone at that
price), keep the queue while the price is unchanged, market at the deadline. Rules (36 combinations), checked in this order
at each 5 s decision u < D: E4 stop (mid moved >= X ticks against us since arrival -> market), E1 (spread <= k ticks ->
market), E3 (last 60 s and spread <= 2 ticks -> market); otherwise post (E2: 2 ticks inside at >= 6 ticks, 3 at >= 9).
Baselines S0, S1, S2, S2i through exec15.run_method. Parent orders, costs and fills as in exec15.
  python r2_15.py run      -> $DATA_DIR/research15_data/val15/r2_<Y>.npz, results15/r2_15_*.csv, r2_15_report.txt"""
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
import exec15 as E  # noqa: E402
import loader15 as L  # noqa: E402
import policy15 as P  # noqa: E402
from fillmodel15 import fill_buy, fill_sell, jit  # noqa: E402

RES = E.RES
VAL = E.VAL
TICK, STEP = 100, 5
YEARS = tuple(range(2015, 2021))
E1S, E2S, E3S, E4S = (-1, 1, 2), (0, 1), (0, 1), (-1, 3, 6)
TIE_GAP = 0.05


def combos():
    out = []
    for k in E1S:
        for e2 in E2S:
            for e3 in E3S:
                for x in E4S:
                    name = f"E1{'off' if k < 0 else k}|E2{'on' if e2 else 'off'}|E3{'on' if e3 else 'off'}|E4{'off' if x < 0 else x}"
                    out.append((name, k, e2, e3, x))
    return out


COMBOS = combos()
METHODS = list(E.BASELINES) + [c[0] for c in COMBOS]
OFF = "E1off|E2off|E3off|E4off"


@jit
def exec_rules(dirs, starts, horizon, qty, mid2_0, bid, ask, bsz, asz, bid_max, bid_min, ask_max, ask_min, vol_bid, vol_ask,
               vol_bid_odd, vol_ask_odd, px_min, px_max, e1_k, e2, e3, e4_x, out):
    """S2i with the R2 rules; out[i] = (limit shares, limit notional, market shares, market notional, market shares above
    the displayed size)."""
    for i in range(len(dirs)):
        d, s = dirs[i], starts[i]
        D = s + horizon
        lim_sh, lim_not, mkt_sh, mkt_not, over = 0, 0, 0, 0, 0
        rem, has, px, a4 = qty, False, 0, 0
        for u in range(s, D + 1):
            if u > s and has and rem > 0:
                if d > 0:
                    f, a4 = fill_buy(2, px, rem, a4, u, ask_min, bid_max, bid_min, px_min, vol_bid, vol_bid_odd)
                else:
                    f, a4 = fill_sell(2, px, rem, a4, u, bid_max, ask_min, ask_max, px_max, vol_ask, vol_ask_odd)
                if f > 0:
                    lim_sh += f
                    lim_not += f * px
                    rem -= f
            if rem == 0:
                break
            if (u - s) % STEP != 0:
                continue
            spr = ask[u] - bid[u]
            go = u == D
            if not go and e4_x > 0 and d * ((bid[u] + ask[u]) - mid2_0[i]) >= 2 * e4_x * TICK:
                go = True
            if not go and e1_k >= 0 and spr <= e1_k * TICK:
                go = True
            if not go and e3 == 1 and u >= D - 60 and spr <= 2 * TICK:
                go = True
            if go:
                mp_ = ask[u] if d > 0 else bid[u]
                disp = asz[u] if d > 0 else bsz[u]
                mkt_sh += rem
                mkt_not += rem * mp_
                over += max(0, rem - disp)
                rem = 0
                break
            if spr >= TICK:
                if spr >= 3 * TICK:
                    k = 1
                    if e2 == 1:
                        k = 3 if spr >= 9 * TICK else (2 if spr >= 6 * TICK else 1)
                    npx = bid[u] + k * TICK if d > 0 else ask[u] - k * TICK
                    na4 = 0
                else:
                    npx = bid[u] if d > 0 else ask[u]
                    na4 = 4 * (bsz[u] if d > 0 else asz[u])
                if not (has and npx == px):
                    has, px, a4 = True, npx, na4
            else:
                has = False
        out[i, 0], out[i, 1], out[i, 2], out[i, 3], out[i, 4] = lim_sh, lim_not, mkt_sh, mkt_not, over


def run_rules(b, dirs, starts, horizon, qty, e1_k, e2, e3, e4_x):
    out = np.zeros((len(dirs), 5), dtype=np.int64)
    mid2_0 = (b["bid"] + b["ask"])[starts]
    exec_rules(dirs, starts, horizon, qty, mid2_0, b["bid"], b["ask"], b["bsz"], b["asz"], b["bid_max"], b["bid_min"],
               b["ask_max"], b["ask_min"], b["vol_bid"], b["vol_ask"], b["vol_bid_odd"], b["vol_ask_odd"], b["px_min"],
               b["px_max"], e1_k, e2, e3, e4_x, out)
    return out


def day_methods(b, day, close_bar):
    """(parent meta, {(qty, horizon): {is_r, is_rp, lim, over, mkt}}) for one symbol-day: S0, S1, S2, S2i and the 36 rules."""
    n = len(b["bid"])
    dirs, starts = E.parents(day, close_bar)
    ok = P.X.quote_ok(b)[starts]
    dirs, starts = dirs[ok], starts[ok]
    mid2 = (b["bid"] + b["ask"])[starts]
    meta = {"dir": dirs, "start": starts, "mid2": mid2, "spread": (b["ask"] - b["bid"])[starts], "n_dropped": int((~ok).sum())}
    T_, F_ = np.ones(n, np.bool_), np.zeros(n, np.bool_)
    res = {}
    for qty, horizon in E.SIZES:
        cols = {"is_r": [], "is_rp": [], "lim": [], "over": [], "mkt": []}
        for name in METHODS:
            if name in ("S0", "S1"):
                out = E.run_method(b, 0 if name == "S0" else 1, dirs, starts, horizon, qty, F_, F_, F_, F_, 0)
            elif name in ("S2", "S2i"):
                out = E.run_method(b, 2, dirs, starts, horizon, qty, T_, T_, F_, F_, int(name == "S2i"))
            else:
                _, k, e2, e3, x = COMBOS[METHODS.index(name) - len(E.BASELINES)]
                out = run_rules(b, dirs, starts, horizon, qty, k, e2, e3, x)
            ir, irp, lim, over = E.costs(out, dirs, mid2, qty)
            for key, v in (("is_r", ir), ("is_rp", irp), ("lim", lim), ("over", over), ("mkt", out[:, 2])):
                cols[key].append(v)
        res[(qty, horizon)] = {k: np.column_stack(v).astype(np.float32) for k, v in cols.items()}
    return meta, res


def task(t):
    year, brow, close_bar = t
    b = L.load(year, brow)
    meta, res = day_methods(b, brow["date"], close_bar)
    meta.update({"date": brow["date"], "sym": brow["sym"]})
    return year, meta, res


def run_all(log):
    tasks = []
    for y in YEARS:
        man, cb = L.manifest(y), L.close_bars(y)
        tasks += [(y, r, cb[d]) for (s, d), r in sorted(man.items())]
    log(f"R2: {len(tasks)} symbol-days 2015-2020, {len(METHODS)} methods x {len(E.SIZES)} sizes")
    t0 = time.time()
    acc = {y: {"meta": [], "res": []} for y in YEARS}
    with mp.get_context("spawn").Pool(8) as pool:
        for i, (y, meta, res) in enumerate(pool.imap(task, tasks, chunksize=8), 1):
            acc[y]["meta"].append(meta)
            acc[y]["res"].append(res)
            if i % 2000 == 0 or i == len(tasks):
                log(f"   {i}/{len(tasks)} symbol-days, {time.time() - t0:.0f}s")
    os.makedirs(VAL, exist_ok=True)
    for y in YEARS:
        ms = acc[y]["meta"]
        par = pd.DataFrame({"date": np.concatenate([[m["date"]] * len(m["dir"]) for m in ms]),
                            "sym": np.concatenate([[m["sym"]] * len(m["dir"]) for m in ms]),
                            **{k: np.concatenate([m[k] for m in ms]) for k in ("dir", "start", "mid2", "spread")}})
        arr = {f"{k}|{q}_{h}": np.concatenate([r[(q, h)][k] for r in acc[y]["res"]])
               for (q, h) in E.SIZES for k in ("is_r", "is_rp", "lim", "over", "mkt")}
        np.savez_compressed(os.path.join(VAL, f"r2_{y}.npz"), methods=np.array(METHODS), **arr)
        par.to_csv(os.path.join(VAL, f"r2_{y}_parents.csv.gz"), index=False)
        log(f"R2 {y}: {len(par)} parents, dropped (no quote at start) {sum(m['n_dropped'] for m in ms)}")
    log(f"R2 run: {time.time() - t0:.0f}s")


def load(y, size=E.SIZES[0]):
    z = np.load(os.path.join(VAL, f"r2_{y}.npz"))
    par = pd.read_csv(os.path.join(VAL, f"r2_{y}_parents.csv.gz"), dtype={"date": str, "sym": str})
    tag = f"{size[0]}_{size[1]}"
    return par, {k: z[f"{k}|{tag}"] for k in ("is_r", "is_rp", "lim", "over", "mkt")}, list(z["methods"])


def n_rules(name):
    return sum(not p.endswith("off") for p in name.split("|"))


def e4_x(name):
    v = name.split("|")[3][2:]
    return 10**9 if v == "off" else int(v)


def stats(a, j):
    ir, irp = a["is_r"][:, j].astype(np.float64), a["is_rp"][:, j].astype(np.float64)
    return {"mean_R": ir.mean(), "median_R": np.median(ir), "p95_R": np.percentile(ir, 95), "mean_Rp": irp.mean(),
            "median_Rp": np.median(irp), "p95_Rp": np.percentile(irp, 95), "limit_frac": float(a["lim"][:, j].mean())}


def analyse(log):
    rows, yearly = [], {}
    for y in YEARS:
        par, a, ms = load(y)
        for j, m in enumerate(ms):
            rows.append({"year": y, "method": m, **stats(a, j)})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(RES, "r2_15_by_year.csv"), index=False)
    piv = T.pivot(index="method", columns="year", values="mean_R")
    piv["mean_of_years"] = piv[list(YEARS)].mean(axis=1)
    piv = piv.sort_values("mean_of_years")
    piv.to_csv(os.path.join(RES, "r2_15_mean_by_year.csv"))
    rules = piv[[m.startswith("E1") for m in piv.index]]
    best = rules.mean_of_years.min()
    near = rules[rules.mean_of_years <= best + TIE_GAP]
    order = sorted(near.index, key=lambda m: (n_rules(m), -e4_x(m), near.loc[m, "mean_of_years"], m))
    sel = {"rule": order[0], "mean_of_years_R": float(piv.loc[order[0], "mean_of_years"]), "best_rule": str(rules.mean_of_years.idxmin()),
           "best_mean_of_years_R": float(best), "candidates_within_0.05": order, "n_combinations": int(len(rules)),
           "baselines_mean_of_years_R": {b_: float(piv.loc[b_, "mean_of_years"]) for b_ in E.BASELINES},
           "rule_text": "min mean over 2015-2020 of the yearly mean IS (R); within 0.05 c: fewer rules on, then larger E4 X "
                        "(off largest), then mean, then name (DEVIATIONS15 23)",
           "written": f"{dt.datetime.now():%Y-%m-%d %H:%M} ET"}
    open(os.path.join(RES, "r2_15_selection.json"), "w").write(json.dumps(sel, indent=1))
    log(f"R2 select: best {sel['best_rule']} {best:.4f}, {len(order)} within {TIE_GAP}, chosen {sel['rule']}")
    return T, piv, sel


def report(log):
    T, piv, sel = analyse(log)
    E_ = sel["rule"]
    names = list(E.BASELINES) + [E_]
    ff = lambda v: f"{v:.3f}"                                                  # noqa: E731
    out = [f"Research 15 experiment R2 (no ML; 2015-2020; 1,000 sh / 300 s; IS cents per share) written {dt.datetime.now():%Y-%m-%d %H:%M} ET",
           f"selected E*: {E_} (mean of yearly means {sel['mean_of_years_R']:.4f}); best {sel['best_rule']} "
           f"{sel['best_mean_of_years_R']:.4f}; within 0.05: {len(sel['candidates_within_0.05'])} of {sel['n_combinations']}",
           "baselines (mean of yearly means): " + ", ".join(f"{k} {v:.4f}" for k, v in sel["baselines_mean_of_years_R"].items()), "",
           "mean IS (R) by year: baselines, E* and the top 10 combinations"]
    top = [m for m in piv.index if m.startswith("E1")][:10]
    out.append(piv.loc[names + [m for m in top if m != E_]].to_string(float_format=ff))
    out += ["", "E* and baselines by year (mean, median, p95 of R and R+, limit fraction):"]
    out.append(T[T.method.isin(names)].set_index(["year", "method"]).to_string(float_format=ff))
    single = {"E1 k=1": "E11|E2off|E3off|E4off", "E1 k=2": "E12|E2off|E3off|E4off", "E2 on": "E1off|E2on|E3off|E4off",
              "E3 on": "E1off|E2off|E3on|E4off", "E4 X=3": "E1off|E2off|E3off|E43", "E4 X=6": "E1off|E2off|E3off|E46"}
    eff = pd.DataFrame({lab: piv.loc[m, list(YEARS) + ["mean_of_years"]] - piv.loc["S2i", list(YEARS) + ["mean_of_years"]]
                        for lab, m in single.items()}).T
    out += ["", f"rule effects: one rule on minus S2i (all off = S2i check: {OFF} - S2i = "
                f"{float(piv.loc[OFF, 'mean_of_years'] - piv.loc['S2i', 'mean_of_years']):+.6f})", eff.to_string(float_format=ff)]
    pars, arrs = [], []
    for y in YEARS:
        par, a, ms = load(y)
        pars.append(par)
        arrs.append(np.column_stack([a["is_r"][:, ms.index(n)] for n in names]))
    par, A = pd.concat(pars, ignore_index=True), np.vstack(arrs).astype(np.float64)
    for title, col, buckets in (("spread at arrival (ticks 1, 2, 3-4, 5-8, 9+)", "spread", __import__("exec15_report").SPREAD_B),
                                ("start time", "start", __import__("exec15_report").TOD_B)):
        v = par[col].to_numpy()
        tb = {bn: {"parents": int(((v >= lo) & (v <= hi)).sum()),
                   **{n: A[(v >= lo) & (v <= hi), i].mean() for i, n in enumerate(names)}} for bn, lo, hi in buckets}
        out += ["", f"mean IS (R) by the {title}, 2015-2020 pooled:", pd.DataFrame(tb).T.to_string(float_format=ff)]
    sens = {}
    for y in YEARS:
        par, a, ms = load(y, E.SIZES[1])
        sens[y] = {n: a["is_r"][:, ms.index(n)].mean() for n in names}
    s = pd.DataFrame(sens)
    s["mean_of_years"] = s.mean(axis=1)
    out += ["", "sensitivity 300 sh / 60 s, mean IS (R) by year:", s.to_string(float_format=ff)]
    txt = "\n".join(out)
    open(os.path.join(RES, "r2_15_report.txt"), "w").write(txt + "\n")
    log("R2 report written")
    print(txt)


def main():
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    f = open(os.path.join(HERE, "logs", f"r2_15_{sys.argv[1]}_{dt.datetime.now():%Y%m%d_%H%M%S}.log"), "a")

    def log(m):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {m}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    if sys.argv[1] == "run":
        run_all(log)
        report(log)
    elif sys.argv[1] == "report":
        report(log)
    else:
        sys.exit("usage: r2_15.py run | report")


if __name__ == "__main__":
    main()
