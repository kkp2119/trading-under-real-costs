#!/usr/bin/env python3
"""Research 17 step 1 power calculation (SPEC17 4), development data only: the 16-D per-parent L3D1 outputs of the existing
universe 2015-2020 ($DATA_DIR/research16_data/qty16_parents.npz). No sealed or new-universe data is read.
Variance model for a daily mean over N stocks (20 parents each): x_st = symbol-day mean of the difference, D_t = day mean.
  W = mean within-day variance of x_st across stocks, V10 = Var(D_t) with the dev 10 stocks a day
  c = V10 - W/10 (common day component), V_N = W/N + c
Serial dependence: VIF = (day-block bootstrap variance of mean D_t, blocks 21, 10,000, seed 17) / (Var(D_t)/T), assumed
unchanged at N. SE_new = sqrt(V_N x VIF / T_new); MDE (one-sided, power 80%) = (z_{1-a} + z_{0.8}) x SE_new.
  H1: E* - S0 ; H2 proxy: S2i - E* (A* - E* is unknown until step 2).
  python power17.py"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "study16_followups"))
import robust16 as R16  # noqa: E402

N_NEW = int(json.load(open(os.path.join(HERE, "results", "n_estimate.json")))["N"])
CAL = os.path.join(HERE, "calendar17.csv")
SEALED = ("20210104", "20261103")      # latest TAQ day on 2026-11-04 (calendar17.csv); see SPEC17_QUESTIONS
KNOWN = {"H1: E* - S0": -1.4313, "H2 proxy: S2i - E*": -0.0357}   # 16-D L3D1 (E* - S2i = +0.0357), quoted only
BOOT_SEED = 17


def series(d, par):
    sd = pd.DataFrame({"date": par.date, "sym": par.sym, "d": d}).groupby(["date", "sym"]).d.mean()
    day = sd.groupby(level=0).mean()
    within = sd.groupby(level=0).var(ddof=1).mean()
    return sd, day, within


def vif(day):
    T = len(day)
    idx = R16.boot_index(T, seed=BOOT_SEED)
    v = day.to_numpy()
    boot = v[idx].mean(axis=1)
    return float(boot.var(ddof=1) / (v.var(ddof=1) / T))


def main():
    q = np.load(os.path.join(R16.LOCAL, "qty16_parents.npz"))
    par = pd.read_csv(os.path.join(R16.LOCAL, "qty16_parents.csv.gz"), dtype={"date": str, "sym": str})
    assert par.date.max() <= "20201231", "development data only"
    o = q["out"][list(q["lrules"]).index(3)]
    dirs, mid2 = par.dir.to_numpy(), par.mid2.to_numpy()
    IS = {m: R16.costs(o[R16.METHODS.index(m)], dirs, mid2, 1) for m in ("S0", "S2i", "Estar")}
    cal = pd.read_csv(CAL, dtype=str).date.str.replace("-", "")
    t_new = int(((cal >= SEALED[0]) & (cal <= SEALED[1])).sum())
    zb = norm.ppf(0.80)
    out = {"n_new_per_year": N_NEW, "t_new_days": t_new, "sealed_window": SEALED, "dev_parents": int(len(par)),
           "dev_days": int(par.date.nunique()), "method": "V_N = W/N + c with c = V_dev - W/n_dev; SE = sqrt(V_N x VIF / T); MDE = (z_(1-a) + z_0.8) x SE", "h": {}}
    for name, d in (("H1: E* - S0", IS["Estar"] - IS["S0"]), ("H2 proxy: S2i - E*", IS["S2i"] - IS["Estar"])):
        sd, day, W = series(d, par)
        n_dev = float(sd.groupby(level=0).size().mean())
        V10 = float(day.var(ddof=1))
        c = V10 - W / n_dev
        vN = W / N_NEW + c
        f = vif(day)
        se_dev = math.sqrt(V10 * f / len(day))
        se_new = math.sqrt(vN * f / t_new)
        h = {"dev_mean_c": float(d.mean()), "known_effect_16D_c": KNOWN[name], "dev_stocks_per_day": n_dev,
             "W_within_day_var": float(W), "V_dev_daymean_var": V10, "c_common_var": float(c), "V_new_daymean_var": float(vN),
             "VIF": f, "se_dev_c": se_dev, "se_new_c": se_new, "mde": {}}
        for a in (0.05, 0.025):
            h["mde"][str(a)] = {"dev_scale_c": (norm.ppf(1 - a) + zb) * se_dev, "new_universe_c": (norm.ppf(1 - a) + zb) * se_new}
        mde = h["mde"]["0.025"]["new_universe_c"]
        h["underpowered_at_holm_0.025"] = bool(mde > abs(KNOWN[name]))
        out["h"][name] = h
    with open(os.path.join(HERE, "results", "power17.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
