#!/usr/bin/env python3
"""Research 18 step 1 power calculation (SPEC18 5; step-1 prompt 4), development data only: research 17's per-parent outputs
of the existing universe ($DATA_DIR/research17_data/dev/dev17_parents.*), validation years 2019-2020, L3D1. No sealed or
new-universe data is read. Proxy for A* - E*: C4 (E* + stop-loss) - C0 (E*), the research 17 candidate closest to R1.
  H2 (mean)  d = IS(C4) - IS(C0) per parent. study17_new_stocks/power17's model: x_st = symbol-day mean, D_t = day mean,
             W = mean within-day variance of x_st, V_dev = Var(D_t), c = V_dev - W/n_dev, V_N = W/N + c,
             VIF = (day-block bootstrap variance of mean D_t) / (Var(D_t)/T); SE_new = sqrt(V_N x VIF / T_new).
  H3 (p95)   p95(C4) - p95(C0) over the pooled parents. SE_dev = sd of the day-block bootstrap (days resampled with their
             parents; weighted inverted-CDF 95th percentile in each draw). Scaled to the new universe two ways:
               orders  SE_dev x sqrt(n_dev x T_dev / (N x T_new))   (variance proportional to stock-days; the prompt's scaling)
               days    SE_dev x sqrt(T_dev / T_new)                 (all variance common to the day; N does not help)
  MDE (one-sided, power 80%) = (z_(1-a) + z_0.8) x SE_new at a = 0.0167, 0.025, 0.05 (Holm levels for three hypotheses).
Bootstrap: circular day blocks 21, 10,000 draws, seed 18. Mean effects are quoted from research 17 only (KNOWN).
  python power18.py"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "study17_new_stocks"))
import engine17 as G  # noqa: E402

DEV = os.path.join(os.environ.get("DATA_DIR", "data"), "research17_data/dev")
VALID = ("20190101", "20201231")
N_NEW = int(json.load(open(os.path.join(HERE, "results", "n_estimate18.json")))["N"])
CAL = os.path.join(HERE, "calendar18.csv")
SEALED = ("20210104", "20261105")
SEED = 18
ALPHAS = (0.05 / 3, 0.025, 0.05)
KNOWN = {"source": "study17_new_stocks/results/dev17_rows.csv (L3D1, 2015-2020)", "C4_minus_C0_mean_c": 0.384279,
         "C4_minus_C0_mean_2019_c": 0.338513, "C4_minus_C0_mean_2020_c": 0.443780,
         "p95_C4_c": 15.7, "p95_C0_c": 18.6, "p95_C4_minus_C0_c": -2.9}


def h2(d, par, idx):
    sd = pd.DataFrame({"date": par.date, "sym": par.sym, "d": d}).groupby(["date", "sym"]).d.mean()
    day = sd.groupby(level=0).mean()
    W = float(sd.groupby(level=0).var(ddof=1).mean())
    n_dev = float(sd.groupby(level=0).size().mean())
    V = float(day.var(ddof=1))
    c = V - W / n_dev
    vN = W / N_NEW + c
    v = day.to_numpy()
    f = float(v[idx].mean(axis=1).var(ddof=1) / (v.var(ddof=1) / len(v)))
    return {"dev_stocks_per_day": n_dev, "W_within_day_var": W, "V_dev_daymean_var": V, "c_common_var": c,
            "V_new_daymean_var": vN, "VIF": f, "se_dev_c": math.sqrt(V * f / len(v))}, vN * f


def boot_p95_diff(a, b, codes, T, idx):
    """sd over the draws of p95(a) - p95(b), days resampled with all their parents (weighted inverted CDF)."""
    out = []
    for x in (a, b):
        o = np.argsort(x, kind="stable")
        out.append((x[o], codes[o]))
    res = np.empty(len(idx))
    for k, row in enumerate(idx):
        w = np.bincount(row, minlength=T)
        q = []
        for xs, cs in out:
            cw = np.cumsum(w[cs])
            q.append(xs[np.searchsorted(cw, 0.95 * cw[-1])])
        res[k] = q[0] - q[1]
    return float(res.std(ddof=1))


def mde(se):
    return {f"{a:.4f}": (norm.ppf(1 - a) + norm.ppf(0.8)) * se for a in ALPHAS}


def main():
    z = np.load(os.path.join(DEV, "dev17_parents.npz"))
    par = pd.read_csv(os.path.join(DEV, "dev17_parents.csv.gz"), dtype={"date": str, "sym": str})
    assert par.date.max() <= "20201231", "development data only"
    keep = ((par.date >= VALID[0]) & (par.date <= VALID[1])).to_numpy()
    o = z["out"][list(z["lrules"]).index(3)][:, keep]
    par = par[keep].reset_index(drop=True)
    runs = list(z["runs"])
    dirs, mid2 = par.dir.to_numpy(), par.mid2.to_numpy()
    IS = {m: G.costs(o[runs.index(m)], dirs, mid2, 1) for m in ("Estar", "Estar_stop")}
    days = sorted(par.date.unique())
    T = len(days)
    codes = pd.Categorical(par.date, categories=days).codes.astype(np.int64)
    idx = G.boot_index(T, seed=SEED)
    cal = pd.read_csv(CAL, dtype=str).date.str.replace("-", "")
    t_new = int(((cal >= SEALED[0]) & (cal <= SEALED[1])).sum())

    h2v, var_new = h2(IS["Estar_stop"] - IS["Estar"], par, idx)
    h2v["se_new_c"] = math.sqrt(var_new / t_new)
    h2v["mde_new_c"] = mde(h2v["se_new_c"])

    n_dev = float(par.groupby("date").sym.nunique().mean())
    se_dev = boot_p95_diff(IS["Estar_stop"], IS["Estar"], codes, T, idx)
    h3v = {"dev_stocks_per_day": n_dev, "se_dev_c": se_dev,
           "se_new_orders_c": se_dev * math.sqrt(n_dev * T / (N_NEW * t_new)), "se_new_days_c": se_dev * math.sqrt(T / t_new)}
    h3v["mde_new_orders_c"] = mde(h3v["se_new_orders_c"])
    h3v["mde_new_days_c"] = mde(h3v["se_new_days_c"])

    out = {"proxy": "C4 (E* + stop) - C0 (E*), L3D1", "valid_window": VALID, "valid_days": T, "valid_parents": int(len(par)),
           "n_new_per_year": N_NEW, "t_new_days": t_new, "sealed_window": SEALED, "alphas": [round(a, 4) for a in ALPHAS],
           "power": 0.8, "boot": {"B": int(idx.shape[0]), "block": G.BOOT_BLOCK, "seed": SEED}, "known_effects": KNOWN,
           "H2": h2v, "H3": h3v, "doc": __doc__.split("\n  ../")[0]}
    with open(os.path.join(HERE, "results", "power18.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "doc"}, indent=1))


if __name__ == "__main__":
    main()
