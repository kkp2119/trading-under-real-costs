"""Research 15-E report (declaration 8; validation 2019 = fold A, 2020 = fold B only). Called by exec15.py report."""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

import exec15 as E
import loader15 as L
import policy15 as P

SPREAD_B = (("1", 0, 149), ("2", 150, 249), ("3-4", 250, 449), ("5-8", 450, 849), ("9+", 850, 10**9))
TOD_B = (("09:35-10:00", 0, 1799), ("10:00-15:30", 1800, 21599), ("15:30-15:54", 21600, 10**9))
YEAR = {"A": 2019, "B": 2020}


def summary(a, st, names, par, qty):
    rows = {}
    mid_c = par.mid2.to_numpy() / 200                                          # cents
    for n in names:
        j = st.index(n)
        ir, irp = a["is_r"][:, j].astype(np.float64), a["is_rp"][:, j].astype(np.float64)
        mk = a["mkt"][:, j].astype(np.float64)
        rows[n] = {"mean_R": ir.mean(), "median_R": np.median(ir), "p95_R": np.percentile(ir, 95), "mean_Rp": irp.mean(),
                   "median_Rp": np.median(irp), "p95_Rp": np.percentile(irp, 95), "mean_R_bp": (ir / mid_c * 1e4).mean(),
                   "limit_frac": a["lim"][:, j].mean(), "over_frac": a["over"][:, j].sum() / mk.sum() if mk.sum() else 0.0}
    return pd.DataFrame(rows).T


def savings(a, st, par, base, sel, year, qty):
    """Daily savings in dollars (sum over the day's parents of (IS base - IS selected) x qty / 100), Sharpe, totals."""
    d = (a["is_r"][:, st.index(base)].astype(np.float64) - a["is_r"][:, st.index(sel)]) * qty / 100
    s = pd.Series(d).groupby(par.date.to_numpy()).sum()
    obs = P.obs_days(year)
    x = np.array([float(s.get(day, 0.0)) for day in obs])
    notional = float((qty * par.mid2.to_numpy() / 2e4).sum())                  # dollars
    return {"total_usd": x.sum(), "mean_day_usd": x.mean(), "sharpe": P.sharpe(x), "days": len(x),
            "notional_usd": notional, "savings_bp": x.sum() / notional * 1e4,
            "per_1bn_usd": x.sum() / notional * 1e9, "positive_days": int((x > 0).sum())}


def by_bucket(a, st, names, par, col, buckets):
    out = {}
    v = par[col].to_numpy()
    for bn, lo, hi in buckets:
        m = (v >= lo) & (v <= hi)
        out[bn] = {"parents": int(m.sum()), **{n: a["is_r"][m, st.index(n)].mean() if m.any() else np.nan for n in names}}
    return pd.DataFrame(out).T


def report(log):
    res = E.RES
    sel = json.load(open(os.path.join(res, "exec15_selection.json")))
    S = sel["setting"]
    names = list(E.BASELINES) + [S]
    st_tab = pd.read_csv(os.path.join(res, "exec15_settings.csv"), index_col=0)
    ff = lambda v: f"{v:.3f}"                                                  # noqa: E731
    out = [f"Research 15-E report (validation only; test not run) written {dt.datetime.now():%Y-%m-%d %H:%M} ET",
           f"selected S3: {S}; mean of fold mean IS (R) {sel['mean_is_R']:.4f} c; best {sel['best_setting']} "
           f"{sel['best_mean_is_R']:.4f}; within 0.05: {len(sel['candidates_within_0.05'])}; S3 settings {sel['n_s3']}",
           "baselines mean of fold means IS (R): " + ", ".join(f"{k} {v:.4f}" for k, v in sel["baselines_mean_is_R"].items()), ""]
    for f in ("A", "B"):
        par, a, st = E.load(f)
        q = E.SIZES[0][0]
        out += [f"fold {f} ({YEAR[f]}): {len(par)} parents (1,000 sh / 300 s), IS in cents per share",
                summary(a, st, names, par, q).to_string(float_format=ff)]
        dif = {b_: a["is_r"][:, st.index(S)].mean() - a["is_r"][:, st.index(b_)].mean() for b_ in E.BASELINES}
        out.append("selected minus baseline (mean IS R): " + ", ".join(f"{k} {v:+.4f}" for k, v in dif.items()))
        for b_ in ("S2", "S0"):
            s_ = savings(a, st, par, b_, S, YEAR[f], q)
            out.append(f"savings vs {b_}: total ${s_['total_usd']:,.0f} over {s_['days']} days (positive days {s_['positive_days']}), "
                       f"mean/day ${s_['mean_day_usd']:,.1f}, Sharpe {s_['sharpe']:.3f}, {s_['savings_bp']:.3f} bp of "
                       f"${s_['notional_usd']:,.0f} traded = ${s_['per_1bn_usd']:,.0f} per $1bn")
        par["spread_ticks"] = par.spread
        out += ["mean IS (R) by the spread at arrival (1e-4 $ ranges = ticks 1, 2, 3-4, 5-8, 9+):",
                by_bucket(a, st, names, par, "spread", SPREAD_B).to_string(float_format=ff),
                "mean IS (R) by the start time:", by_bucket(a, st, names, par, "start", TOD_B).to_string(float_format=ff)]
        par2, a2, st2 = E.load(f, E.SIZES[1])
        out += ["sensitivity 300 sh / 60 s (reference):", summary(a2, st2, names, par2, E.SIZES[1][0]).to_string(float_format=ff), ""]
    out += ["top 10 S3 settings by the mean of the fold mean IS (R):",
            st_tab[[i.startswith("S3|") for i in st_tab.index]].head(10).to_string(float_format=lambda v: f"{v:.4f}")]
    txt = "\n".join(out)
    open(os.path.join(res, "exec15_report.txt"), "w").write(txt + "\n")
    log("exec report written")
    print(txt)
