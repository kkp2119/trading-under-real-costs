#!/usr/bin/env python3
"""Research 15 reference experiment R1 (Kyle 2026-10-29; after the step 3 results; DEVIATIONS15 19): simple rules that stand
nearer the front of the queue, no machine learning, validation years 2019 and 2020 only (2021+ never read).
Variants on simulator15.kernel (09:30-16:00, Q model, vopt = [keep, improve, exit improve]): B0, V1 keep, V2 one tick inside
when the spread >= 3 ticks, V3 = V1 + V2, V4 = V3 + exit side one tick inside; reference B0 under the research 12 T model.
Per year and variant: P&L per share (R, R+), limit shares, markouts 5/30/60 s, daily net Sharpe, market shares and their cost
(fee under R and the price versus the decision mid), and markouts of limit fills by the spread at the order's decision bar.
Run: python r1_15.py. Writes results15/r1_15.txt, results15/r1_15.csv; runs to $DATA_DIR/research15_data/val15/."""
import datetime as dt
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import accounting15 as A  # noqa: E402
import features15  # noqa: E402
import loader15 as L  # noqa: E402
import model15 as M15  # noqa: E402
import policy15 as P  # noqa: E402
import simulator15 as S  # noqa: E402

RES = os.path.join(HERE, "results15")
VAL = os.path.join(M15.DATA, "val15")
VARIANTS = (("B0", "Q", (0, 0, 0)), ("V1", "Q", (1, 0, 0)), ("V2", "Q", (0, 1, 0)), ("V3", "Q", (1, 1, 0)),
            ("V4", "Q", (1, 1, 1)), ("B0_T", "T", (0, 0, 0)))
BUCKETS = (("2", 200, 299), ("3-4", 300, 499), ("5-8", 500, 899), ("9-12", 900, 1200))
KS = (5, 30, 60)
YEARS = (2019, 2020)


def one(task):
    year, brow, close_bar, gate = task
    b = L.load(year, brow)
    f = features15.compute(b)
    d0, df = P.B0W2.d_bounds(close_bar)
    rows = []
    for name, model, vopt in VARIANTS:
        fills, _, st = S.simulate(b, f, P.B0W2.kernel_params(model, close_bar, gate), vopt=np.array(vopt))
        r = {"variant": name, "sym": brow["sym"], "date": brow["date"], **P.summarize(fills, st, f, close_bar, d0, df)}
        a = A.account(fills, f["mid2"], f["mid_ok"], close_bar, d0, df)
        r["edge2_mkt"] = a["edge2_mkt"]
        lim = fills[fills[:, 2] == 0] if len(fills) else fills
        spr = b["ask"][lim[:, 6]] - b["bid"][lim[:, 6]] if len(lim) else np.zeros(0, np.int64)
        for bn, lo, hi in BUCKETS:
            sel = lim[(spr >= lo) & (spr <= hi)]
            r[f"b{bn}_sh"] = int(sel[:, 3].sum()) if len(sel) else 0
            for k in KS:
                if len(sel):
                    u, s, q, p = sel[:, 0], sel[:, 1], sel[:, 3], sel[:, 4]
                    t = u + k
                    ok = t < close_bar
                    ok[ok] &= f["mid_ok"][t[ok]]
                    r[f"b{bn}_mk{k}_sum2"] = int(np.sum((s * q * (f["mid2"][np.minimum(t, len(b["bid"]) - 1)] - 2 * p))[ok]))
                    r[f"b{bn}_mk{k}_sh"] = int(q[ok].sum())
                else:
                    r[f"b{bn}_mk{k}_sum2"] = r[f"b{bn}_mk{k}_sh"] = 0
        rows.append(r)
    return rows


def main():
    g = M15.read_gates().set_index(["sym", "date"])
    tasks = []
    for y in YEARS:
        man, cb = L.manifest(y), L.close_bars(y)
        tasks += [(y, r, cb[d], bool(g.loc[(s, d)].gate)) for (s, d), r in sorted(man.items())]
    t0 = dt.datetime.now()
    with mp.get_context("spawn").Pool(8) as pool:
        runs = pd.DataFrame([r for rows in pool.map(one, tasks, chunksize=8) for r in rows])
    p = os.path.join(VAL, "r1_runs.csv.gz")
    runs.to_csv(p, index=False)
    out = []
    for y in YEARS:
        obs = P.obs_days(y)
        ry = runs[runs.date.str[:4] == str(y)]
        for v, _, _ in VARIANTS:
            x = ry[ry.variant == v]
            dr = x.groupby("date").net_R.sum()
            dp = x.groupby("date").net_Rp.sum()
            dR = np.array([float(dr.get(d, 0)) / 1e4 for d in obs])
            dP = np.array([float(dp.get(d, 0)) / 1e4 for d in obs])
            sh = x.sh_lim.sum() + x.sh_mkt.sum()
            row = {"year": y, "variant": v, "pps_R_cents": 100 * dR.sum() / sh, "pps_Rp_cents": 100 * dP.sum() / sh,
                   "sharpe_R": P.sharpe(dR), "sharpe_Rp": P.sharpe(dP), "net_R_usd": dR.sum(), "sh_lim": int(x.sh_lim.sum()),
                   "n_fill_ev": int(x.n_fill_ev.sum()),
                   **{f"mk{k}_cents": x[f"mk{k}_sum2"].sum() / 2 / x[f"mk{k}_sh"].sum() / 100 for k in KS},
                   "sh_mkt": int(x.sh_mkt.sum()), "mkt_fee_R_usd": 30 * x.sh_mkt.sum() / 1e4,
                   "mkt_cost_vs_mid_usd": -x.edge2_mkt.sum() / 2 / 1e4, "n_stale": int(x.n_stale.sum())}
            for bn, _, _ in BUCKETS:
                row[f"b{bn}_sh"] = int(x[f"b{bn}_sh"].sum())
                for k in KS:
                    s_ = x[f"b{bn}_mk{k}_sh"].sum()
                    row[f"b{bn}_mk{k}_cents"] = x[f"b{bn}_mk{k}_sum2"].sum() / 2 / s_ / 100 if s_ else float("nan")
            out.append(row)
    T = pd.DataFrame(out)
    T.to_csv(os.path.join(RES, "r1_15.csv"), index=False)
    lines = [f"Research 15 reference experiment R1 (validation 2019, 2020; 09:30-16:00; costs R, R+); {len(tasks)} symbol-days, "
             f"{len(runs)} runs, {(dt.datetime.now() - t0).seconds}s; B0_T = B0 under the research 12 T model (front of queue bound)"]
    main_cols = ["pps_R_cents", "pps_Rp_cents", "sharpe_R", "sharpe_Rp", "net_R_usd", "sh_lim", "mk5_cents", "mk30_cents",
                 "mk60_cents", "sh_mkt", "mkt_fee_R_usd", "mkt_cost_vs_mid_usd", "n_stale"]
    for y in YEARS:
        lines += ["", f"{y}:", T[T.year == y].set_index("variant")[main_cols].T.to_string(float_format=lambda v: f"{v:.3f}")]
        bc = [c for bn, _, _ in BUCKETS for c in (f"b{bn}_sh", f"b{bn}_mk5_cents", f"b{bn}_mk30_cents", f"b{bn}_mk60_cents")]
        lines += [f"{y} limit fills by the spread (ticks) at the order's decision bar: shares and markouts (cents)",
                  T[T.year == y].set_index("variant")[bc].T.to_string(float_format=lambda v: f"{v:.3f}")]
    txt = "\n".join(lines)
    open(os.path.join(RES, "r1_15.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
