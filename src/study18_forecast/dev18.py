#!/usr/bin/env python3
"""Research 18 step 2 development run (SPEC18 3-5, SPEC18_ANSWERS; step-2 prompt 1-3) on the existing universe 2015-2020
through engine18, read through study15_ml_execution/loader15 (development bars only; no research 18 sealed file and no research
12/15/16/17 sealed or test folder is read).
  pass 1 (2015-2020)  parents, grid inputs and y, S0 / S1 / S2i / R0 under L0 and L3. Regression (stop when a check fails,
                      no model and no summary written): parents and vol_pre equal to research 17 dev17, per-parent outputs of
                      S0 / S1 / S2i / R0 equal to dev17's S0 / S1 / S2i / E*, mean IS at 3 decimals equal to 16-D / 17.
  fit (2015-2018)     vol_pre fill = median of the training parents; mean / sd of the training points (valid quote, y defined);
                      LogisticRegression(penalty l2, C 1.0, lbfgs) once -> model18.json (+ model18.sha256).
  pass 2 (2019-2020)  p from model18.json inside the worker (engine18.symday18, the run_test18 path): S0, S1, S2i, R0 and the
                      19 candidates under L0 and L3, then R0 and the 19 candidates again with delay 1.
  report              prediction metrics (train for reference, valid), coefficients, H1 MDE; per candidate and variant mean /
                      p95 IS, candidate - R0 with 95% CIs (day blocks 21, 10,000, seed 18), firing shares, years; A* (SPEC18 4);
                      the H2 / H3 power flags against the step-1 MDE (alpha 0.0167); the delay check for A* and R0.
  python dev18.py run [--workers 8]
Per-parent and grid outputs -> $DATA_DIR/research18_data/dev/, aggregates -> results/dev18_*."""
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
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "study15_ml_execution"))
import engine18 as G  # noqa: E402
import loader15 as L  # noqa: E402

RES = os.path.join(HERE, "results")
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research18_data/dev")
DEV17 = os.path.join(os.environ.get("DATA_DIR", "data"), "research17_data/dev")
MODEL = os.path.join(HERE, "model18.json")
YEARS, TRAIN, VALID = tuple(range(2015, 2021)), (2015, 2016, 2017, 2018), (2019, 2020)
BASE = ("S0", "S1", "S2i", "R0")
TARGETS = {"L0D0": {"S0": 3.785, "S1": 3.737, "S2i": 2.674, "R0": 2.605},
           "L3D1": {"S0": 4.579, "S1": 3.737, "S2i": 3.112, "R0": 3.148}}
R17_NAME = {"S0": "S0", "S1": "S1", "S2i": "S2i", "R0": "Estar"}
TRIALS_BEFORE, TRIALS_ADDED = 730, 20
N_SEALED, T_SEALED = 20, 1468                                    # results/n_estimate18.json, calendar18 window
ALPHA1, POWER = 0.05 / 3, 0.8
SEALED_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research18_data/sealed")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- simulation
def task(t):
    year, row, close_bar, model, runs, delay_runs = t
    b = L.load(year, row)
    meta, outs = G.symday18(b, row["date"], close_bar, model, runs, delay_runs)
    if model is None:
        meta.pop("P")
    return row["sym"], row["date"], meta, outs


def simulate(years, model, runs, delay_runs, workers):
    tasks = []
    for y in years:
        man, cb = L.manifest(y), L.close_bars(y)
        tasks += [(y, r, cb[d], model, runs, delay_runs) for (s, d), r in sorted(man.items())]
    assert all(r["date"] <= "20201231" for _, r, _, _, _, _ in tasks), "date after 2020"
    paths = [os.path.join(L.SOURCES[L.source_of(y)][0], r["file"]) for y, r, _, _, _, _ in tasks]
    assert not any(os.path.abspath(p).startswith(SEALED_ROOT) for p in paths), "sealed path"
    with mp.get_context("spawn").Pool(workers) as pool:
        res = list(pool.imap(task, tasks, chunksize=8))
    par = pd.DataFrame({"date": np.concatenate([[d] * len(m["dir"]) for _, d, m, _ in res]),
                        "sym": np.concatenate([[s] * len(m["dir"]) for s, _, m, _ in res]),
                        **{k: np.concatenate([m[k] for _, _, m, _ in res]) for k in ("dir", "start", "mid2", "spread", "vol_pre")}})
    OUT = np.concatenate([o for _, _, _, o in res], axis=2)
    grid = {k: np.concatenate([m[k] for _, _, m, _ in res]) for k in ("X", "y", "valid") + (("P",) if model is not None else ())}
    counts = {k: int(sum(m["counts"][k] for _, _, m, _ in res)) for k in res[0][2]["counts"]}
    info = {"symbol_days": len(tasks), "dropped_parents": int(sum(m["n_dropped"] for _, _, m, _ in res)), "grid_counts": counts}
    return par, OUT, grid, info


# ---------------------------------------------------------------- regression (research 17 dev17, 16-D / 17 means)
def regression(par, OUT):
    z = np.load(os.path.join(DEV17, "dev17_parents.npz"))
    p17 = pd.read_csv(os.path.join(DEV17, "dev17_parents.csv.gz"), dtype={"date": str, "sym": str}, float_precision="round_trip")
    cols = ("date", "sym", "dir", "start", "mid2", "spread")
    chk = {"parents_equal_dev17": bool(len(par) == len(p17) and all(np.array_equal(par[c].to_numpy(), p17[c].to_numpy()) for c in cols))}
    a, b = par.vol_pre.to_numpy(), p17.vol_pre.to_numpy()
    chk["vol_pre_equal_dev17"] = bool(len(a) == len(b) and np.array_equal(np.isnan(a), np.isnan(b)) and
                                      np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]))
    r17, l17 = list(z["runs"]), list(z["lrules"])
    chk["per_parent_equal_dev17"] = {f"L{lr}": {m: bool(np.array_equal(OUT[G.LRULES.index(lr), BASE.index(m), :, :9],
                                                                       z["out"][l17.index(lr), r17.index(R17_NAME[m])]))
                                                for m in BASE} for lr in G.LRULES}
    dirs, mid2 = par.dir.to_numpy(), par.mid2.to_numpy()
    chk["mean_is_3dp"] = {}
    for lab, lr, dep in G.VARIANTS:
        for m, want in TARGETS[lab].items():
            got = float(G.costs(OUT[G.LRULES.index(lr), BASE.index(m)], dirs, mid2, dep).mean())
            chk["mean_is_3dp"][f"{lab} {m}"] = {"got": got, "want": want, "pass": round(got, 3) == want}
    chk["shares_sum_to_qty"] = bool(np.all(OUT[:, :, :, 0] + OUT[:, :, :, 2] == G.QTY))
    chk["no_rule_action"] = bool(not OUT[:, :, :, 9:11].any())
    chk["market_at_nonpositive_price"] = int(OUT[:, :, :, 11].sum())
    chk["no_date_after_2020"] = bool(par.date.max() <= "20201231")
    ok = (chk["parents_equal_dev17"] and chk["vol_pre_equal_dev17"] and all(all(v.values()) for v in chk["per_parent_equal_dev17"].values())
          and all(v["pass"] for v in chk["mean_is_3dp"].values()) and chk["shares_sum_to_qty"] and chk["no_rule_action"]
          and chk["no_date_after_2020"])
    return ok, chk


# ---------------------------------------------------------------- model (SPEC18 3, ANSWERS 3, A-13, A-15)
def fit(par, grid):
    from sklearn import __version__ as skv
    from sklearn.linear_model import LogisticRegression
    yr = par.date.str[:4].astype(int).to_numpy()
    tr = np.isin(yr, TRAIN)
    vp = par.vol_pre.to_numpy()[tr]
    fill = float(np.median(vp[~np.isnan(vp)]))
    X, y, v = grid["X"][tr], grid["y"][tr], grid["valid"][tr]
    keep = v & (y >= 0)
    Xk = X[keep].copy()
    Xk[:, 2] = np.where(np.isnan(Xk[:, 2]), fill, Xk[:, 2])
    mean, sd = Xk.mean(axis=0), Xk.std(axis=0)
    Z = (Xk - mean) / sd
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000).fit(Z, y[keep])
    model = {"features": list(G.FEATS), "mean": mean.tolist(), "sd": sd.tolist(), "coef": clf.coef_[0].tolist(),
             "intercept": float(clf.intercept_[0]), "vol_pre_fill": fill, "train_years": list(TRAIN),
             "n_train_points": int(keep.sum()), "n_train_parents": int(tr.sum()), "base_rate_train": float(y[keep].mean()),
             "sklearn": skv, "estimator": "LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)",
             "n_iter": int(clf.n_iter_[0]), "p_formula": "1 / (1 + exp(-(intercept + sum coef_j (x_j - mean_j) / sd_j)))",
             "written": dt.datetime.now().isoformat(timespec="seconds")}
    rng = np.random.default_rng(18)
    pick = rng.choice(len(Z), min(200_000, len(Z)), replace=False)
    pz = 1.0 / (1.0 + np.exp(-(model["intercept"] + ((Xk[pick] - np.asarray(model["mean"])) / np.asarray(model["sd"])) @ np.asarray(model["coef"]))))
    model["check_vs_predict_proba_max_abs"] = float(np.abs(pz - clf.predict_proba(Z[pick])[:, 1]).max())
    with open(MODEL, "w") as fh:
        json.dump(model, fh, indent=1)
    h = sha(MODEL)
    with open(MODEL.replace(".json", ".sha256"), "w") as fh:
        fh.write(f"{h}  model18.json\n")
    return model, h


def p_from_X(model, X, valid):
    Xf = X.copy()
    Xf[..., 2] = np.where(np.isnan(Xf[..., 2]), model["vol_pre_fill"], Xf[..., 2])
    z = (Xf - np.asarray(model["mean"])) / np.asarray(model["sd"])
    return np.where(valid, 1.0 / (1.0 + np.exp(-(model["intercept"] + z @ np.asarray(model["coef"])))), np.nan)


# ---------------------------------------------------------------- prediction metrics (step-2 prompt 2)
def scored(par, P, y, valid):
    """Flat arrays of the scored points (valid quote, y defined): p, y, day code, year; and the counts."""
    keep = valid & (y >= 0)
    dates = np.repeat(par.date.to_numpy(), G.NG).reshape(len(par), G.NG)
    days = sorted(set(dates[keep]))
    codes = pd.Categorical(dates[keep], categories=days).codes.astype(np.int64)
    n = {"grid_points": int(y.size), "valid_points": int(valid.sum()), "y_missing_points": int((valid & (y < 0)).sum()),
         "y_missing_share_of_valid": float((valid & (y < 0)).sum() / valid.sum()), "scored_points": int(keep.sum()), "days": len(days)}
    return P[keep], y[keep].astype(np.int8), codes, np.array([int(d[:4]) for d in dates[keep]]), n


def pred_metrics(p, y, codes, yrs, base_train, boot):
    T = int(codes.max()) + 1
    out = {"base_rate": float(y.mean())}
    if boot:
        W = G.day_counts(G.boot_index(T), T)
        auc, dist = G.auc_boot(p, y, codes, W)
        out["auc_boot"] = {"se": float(dist.std(ddof=1)), "ci95": [float(x) for x in np.percentile(dist, [2.5, 97.5])],
                           "share_le_0.5": float(np.mean(dist <= 0.5)), "B": G.BOOT_B, "block": G.BOOT_BLOCK, "seed": G.BOOT_SEED}
    else:
        auc = G.auc_boot(p, y, codes, np.ones((1, T)))[0]
    brier = float(np.mean((p - y) ** 2))
    ref = float(np.mean((base_train - y) ** 2))
    out.update({"auc": auc, "accuracy_0.5": float(np.mean((p >= 0.5) == (y == 1))), "brier": brier,
                "brier_ref_train_base_rate": ref, "bss": 1 - brier / ref})
    edges = np.quantile(p, np.linspace(0, 1, 11))
    k = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, 9)
    out["deciles"] = [{"decile": int(i + 1), "n": int((k == i).sum()), "p_lo": float(edges[i]), "p_hi": float(edges[i + 1]),
                       "mean_p": float(p[k == i].mean()), "mean_y": float(y[k == i].mean())} for i in range(10)]
    out["yearly_auc"] = {}
    for yr in sorted(set(yrs)):
        m = yrs == yr
        c = pd.Categorical(codes[m]).codes.astype(np.int64)
        out["yearly_auc"][str(yr)] = {"n": int(m.sum()), "auc": G.auc_boot(p[m], y[m], c, np.ones((1, int(c.max()) + 1)))[0],
                                      "base_rate": float(y[m].mean())}
    return out, (dist if boot else None)


def h1_power(se_dev, par_valid):
    from scipy.stats import norm
    n_dev = float(par_valid.groupby("date").sym.nunique().mean())
    T_dev = par_valid.date.nunique()
    se_o = se_dev * math.sqrt(n_dev * T_dev / (N_SEALED * T_SEALED))
    se_d = se_dev * math.sqrt(T_dev / T_SEALED)
    z = {a: norm.ppf(1 - a) + norm.ppf(POWER) for a in (ALPHA1, 0.025, 0.05)}
    return {"se_valid": se_dev, "valid_stocks_per_day": n_dev, "valid_days": int(T_dev), "sealed_N": N_SEALED, "sealed_days": T_SEALED,
            "se_sealed_orders": se_o, "se_sealed_days": se_d,
            "mde_auc_minus_0.5_orders": {f"{a:.4f}": z[a] * se_o for a in z}, "mde_auc_minus_0.5_days": {f"{a:.4f}": z[a] * se_d for a in z}}


# ---------------------------------------------------------------- rules (step-2 prompt 3)
def outs_of(par, OUT, names, lr):
    o = OUT[G.LRULES.index(lr)]
    outs = {m: o[i] for i, m in enumerate(names)}
    outs["C3"] = np.where(G.branch_c3(par.spread.to_numpy())[:, None], outs["S2i"], outs["R0"])
    return outs


def rule_rows(par, outs, lab, dep, codes, n_days, idx, W):
    dirs, mid2 = par.dir.to_numpy(), par.mid2.to_numpy()
    yr = par.date.str[:4].to_numpy()
    IS = {m: G.costs(outs[m], dirs, mid2, dep) for m in ("S0", "S1", "S2i", "C3") + G.CANDS}
    rows = []
    for m in ("S0", "S1", "S2i", "C3") + G.CANDS:
        d = IS[m] - IS["R0"]
        est, lo, hi = G.diff_ci(d, codes, n_days, idx)
        pe, pdist = G.p95_diff_boot(IS[m], IS["R0"], codes, W)
        o = outs[m]
        f = G.fired(o, m) if m in G.CANDS else np.zeros(len(o), bool)
        r = {"variant": lab, "method": m, "n_parents": int(len(o)), "mean_is_c": float(IS[m].mean()),
             "p95_is_c": float(np.percentile(IS[m], 95)), "minus_R0_c": est, "minus_R0_lo": lo, "minus_R0_hi": hi,
             "p95_minus_R0_c": pe, "p95_minus_R0_lo": float(np.percentile(pdist, 2.5)), "p95_minus_R0_hi": float(np.percentile(pdist, 97.5)),
             "fired_frac": float(f.mean()), "r1_parent_frac": float((o[:, 9] > 0).mean()), "r1_share_frac": float((o[:, 9] / G.QTY).mean()),
             "r2_parent_frac": float((o[:, 10] > 0).mean()), "r2_skips_per_parent": float(o[:, 10].mean()),
             "limit_frac": float((o[:, 0] / G.QTY).mean()), "deadline_parent_frac": float((o[:, 6] > 0).mean()),
             "mean_done_s": float(o[:, 5].mean()), "market_nonpositive_price": int(o[:, 11].sum())}
        r.update({f"minus_R0_{y}": float(d[yr == str(y)].mean()) for y in VALID})
        rows.append(r)
    return rows, IS


def delay_check(par, OUT, names, astar, codes, n_days, idx, W):
    dirs, mid2 = par.dir.to_numpy(), par.mid2.to_numpy()
    res = {}
    for lab, lr, dep in G.VARIANTS:
        o = OUT[G.LRULES.index(lr)]
        g = lambda m: G.costs(o[names.index(m)], dirs, mid2, dep)              # noqa: E731
        r0, a, r0d, ad = g("R0"), g(astar), g("R0@d1"), g(f"{astar}@d1")
        v = {"R0_delay_mean_c": float(r0d.mean()), "R0_delay_p95_c": float(np.percentile(r0d, 95)),
             "astar_delay_mean_c": float(ad.mean()), "astar_delay_p95_c": float(np.percentile(ad, 95)),
             "market_nonpositive_price": int(o[names.index("R0@d1"), :, 11].sum() + o[names.index(f"{astar}@d1"), :, 11].sum())}
        for name, (x, z) in (("astar_delay_minus_R0_delay", (ad, r0d)), ("R0_delay_minus_R0", (r0d, r0)), ("astar_delay_minus_astar", (ad, a))):
            est, lo, hi = G.diff_ci(x - z, codes, n_days, idx)
            pe, pdist = G.p95_diff_boot(x, z, codes, W)
            v[name] = {"mean_c": est, "ci95": [lo, hi], "p95_diff_c": pe,
                       "p95_ci95": [float(np.percentile(pdist, 2.5)), float(np.percentile(pdist, 97.5))]}
        res[lab] = v
    return res


def power_flags(IS, astar):
    pw = json.load(open(os.path.join(RES, "power18.json")))
    key = f"{ALPHA1:.4f}"
    e2 = float((IS[astar] - IS["R0"]).mean())
    e3 = float(np.percentile(IS[astar], 95) - np.percentile(IS["R0"], 95))
    m2, m3 = pw["H2"]["mde_new_c"][key], pw["H3"]["mde_new_days_c"][key]
    return {"alpha": key, "H2": {"effect_c": e2, "mde_c": m2, "underpowered": bool(abs(e2) < m2), "direction_ok": bool(e2 < 0)},
            "H3": {"effect_c": e3, "mde_c": m3, "mde_basis": "days-only scaling (ANSWERS A-26)", "underpowered": bool(abs(e3) < m3),
                   "direction_ok": bool(e3 < 0), "mde_orders_c": pw["H3"]["mde_new_orders_c"][key]}}


# ---------------------------------------------------------------- run
def main(workers):
    t0 = time.time()
    os.makedirs(LOCAL, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    par, OUT, grid, info1 = simulate(YEARS, None, BASE, (), workers)
    print(f"pass 1: {len(par):,} parents, {info1['symbol_days']:,} symbol-days, {time.time() - t0:.0f} s", flush=True)
    ok, chk = regression(par, OUT)
    if not ok:
        with open(os.path.join(RES, "dev18_checks_FAILED.json"), "w") as fh:
            json.dump(chk, fh, indent=1, default=str)
        raise SystemExit("a regression check failed: results/dev18_checks_FAILED.json (no model, no summary written)")
    np.savez_compressed(os.path.join(LOCAL, "dev18_pass1.npz"), out=OUT, names=np.array(BASE), lrules=np.array(G.LRULES),
                        X=grid["X"].astype(np.float32), y=grid["y"], valid=grid["valid"])
    par.to_csv(os.path.join(LOCAL, "dev18_parents.csv.gz"), index=False)
    model, msha = fit(par, grid)
    print(f"fit: {model['n_train_points']:,} points, coef {model['coef']}, intercept {model['intercept']:.4f}, "
          f"n_iter {model['n_iter']}, {time.time() - t0:.0f} s", flush=True)

    names = list(BASE) + [c for c in G.CANDS if c != "R0"]
    delay = list(G.CANDS)
    pv, OUT2, grid2, info2 = simulate(VALID, model, names, delay, workers)
    names_all = names + [f"{m}@d1" for m in delay]
    print(f"pass 2: {len(pv):,} parents, {time.time() - t0:.0f} s", flush=True)
    yr1 = par.date.str[:4].astype(int).to_numpy()
    vmask = np.isin(yr1, VALID)
    cons = {"parents_equal_pass1": bool(all(np.array_equal(pv[c].to_numpy(), par[vmask][c].to_numpy()) for c in ("date", "sym", "dir", "start", "mid2"))),
            "base_runs_equal_pass1": bool(all(np.array_equal(OUT2[:, names.index(m)], OUT[:, BASE.index(m)][:, vmask]) for m in BASE)),
            "y_valid_equal_pass1": bool(np.array_equal(grid2["y"], grid["y"][vmask]) and np.array_equal(grid2["valid"], grid["valid"][vmask])),
            "p_worker_vs_stored_X_max_abs": float(np.nanmax(np.abs(grid2["P"] - p_from_X(model, grid["X"][vmask], grid["valid"][vmask]))))}
    np.savez_compressed(os.path.join(LOCAL, "dev18_valid.npz"), out=OUT2, names=np.array(names_all), lrules=np.array(G.LRULES),
                        P=grid2["P"], y=grid2["y"], valid=grid2["valid"])

    # prediction metrics
    tr = np.isin(yr1, TRAIN)
    ptr = p_from_X(model, grid["X"][tr], grid["valid"][tr])
    a, b, c, d, ntr = scored(par[tr].reset_index(drop=True), ptr, grid["y"][tr], grid["valid"][tr])
    m_tr, _ = pred_metrics(a, b, c, d, model["base_rate_train"], boot=False)
    a, b, c, d, nva = scored(pv, grid2["P"], grid2["y"], grid2["valid"])
    m_va, _ = pred_metrics(a, b, c, d, model["base_rate_train"], boot=True)
    pred = {"train": {**ntr, **m_tr}, "valid": {**nva, **m_va}, "grid_counts_pass1": info1["grid_counts"]}
    pred["h1_power"] = h1_power(m_va["auc_boot"]["se"], pv)
    print(f"AUC train {m_tr['auc']:.4f}, valid {m_va['auc']:.4f}, {time.time() - t0:.0f} s", flush=True)

    # rules
    days = sorted(pv.date.unique())
    codes = pd.Categorical(pv.date, categories=days).codes.astype(np.int64)
    idx = G.boot_index(len(days))
    W = G.day_counts(idx, len(days))
    rows, IS_all = [], {}
    for lab, lr, dep in G.VARIANTS:
        r, IS = rule_rows(pv, outs_of(pv, OUT2, names_all, lr), lab, dep, codes, len(days), idx, W)
        rows += r
        IS_all[lab] = IS
    pd.DataFrame(rows).to_csv(os.path.join(RES, "dev18_rows.csv"), index=False, float_format="%.6f")
    prim = {r["method"]: {"mean": r["mean_is_c"], "p95": r["p95_is_c"], "fired": r["fired_frac"]}
            for r in rows if r["variant"] == "L3D1" and r["method"] in G.CANDS}
    astar, sel = G.select_astar(prim)
    flags = None if astar == "R0" else power_flags(IS_all["L3D1"], astar)
    dly = delay_check(pv, OUT2, names_all, astar, codes, len(days), idx, W)
    out = {"astar": astar, "selection": sel, "q2": "개발에서 개선 없음 (H2·H3 없음, H1만)" if astar == "R0" else f"A* = {astar}",
           "power_flags": flags, "delay": dly, "prediction": pred, "model": {k: model[k] for k in ("features", "mean", "sd", "coef", "intercept",
           "vol_pre_fill", "n_train_points", "base_rate_train", "n_iter", "check_vs_predict_proba_max_abs")}, "model_sha256": msha,
           "trials_before": TRIALS_BEFORE, "trials_added": TRIALS_ADDED, "trials_cumulative": TRIALS_BEFORE + TRIALS_ADDED,
           "rows": rows, "checks": chk, "consistency_pass2": cons, "sim": {"pass1": info1, "pass2": info2},
           "n_parents_valid": int(len(pv)), "n_days_valid": len(days), "boot": {"B": G.BOOT_B, "block": G.BOOT_BLOCK, "seed": G.BOOT_SEED},
           "runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(RES, "dev18_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str, ensure_ascii=False)
    report(out)
    print(open(os.path.join(RES, "dev18_report.md")).read())


# ---------------------------------------------------------------- report
def report(out):
    R = pd.DataFrame(out["rows"])
    pr, md = out["prediction"], out["model"]
    L_ = ["# 연구 18 · 2단계 개발 실행 (기존 유니버스: 학습 2015–2018, 검증 2019–2020)", "",
          f"- 검증 주문 {out['n_parents_valid']:,}건, {out['n_days_valid']}일. 봉인 데이터는 읽지 않았다.",
          f"- CI: 일 블록 21, 10,000회, seed 18. 시행 수 +{out['trials_added']}, 누적 {out['trials_cumulative']}.", "",
          f"**A* = {out['astar']}**", "", "## 예측 평가", "",
          "| 구간 | 격자 지점 | 채점 지점 | y 없음 (유효 지점 중) | 기준률 | AUC | 적중률 (0.5) | Brier | BSS |", "|---|---|---|---|---|---|---|---|---|"]
    for k, name in (("train", "학습 2015–2018 (참고)"), ("valid", "검증 2019–2020")):
        x = pr[k]
        L_.append(f"| {name} | {x['grid_points']:,} | {x['scored_points']:,} | {x['y_missing_share_of_valid']:.3f} | {x['base_rate']:.4f} | "
                  f"{x['auc']:.4f} | {x['accuracy_0.5']:.4f} | {x['brier']:.4f} | {x['bss']:+.4f} |")
    v = pr["valid"]
    L_ += ["", f"- 검증 AUC 95% CI {v['auc_boot']['ci95'][0]:.4f}–{v['auc_boot']['ci95'][1]:.4f}, 부트스트랩 SE {v['auc_boot']['se']:.4f}. "
           "BSS의 기준은 학습 기준률(상수 예측)이다.",
           "- 연도별 AUC: " + ", ".join(f"{y} {x['auc']:.4f}" for k in ("train", "valid") for y, x in pr[k]["yearly_auc"].items()), "",
           "### 계수 (표준화)", "", "| 입력 | 계수 | 학습 평균 | 학습 SD |", "|---|---|---|---|"]
    for f, c, m, s in zip(md["features"], md["coef"], md["mean"], md["sd"]):
        L_.append(f"| {f} | {c:+.4f} | {m:.4f} | {s:.4f} |")
    L_ += [f"| 절편 | {md['intercept']:+.4f} | | |", "", f"- vol_pre 결측 대체 {md['vol_pre_fill']:.4f} bp, 학습 지점 {md['n_train_points']:,}, "
           f"반복 {md['n_iter']}, predict_proba와 최대 차이 {md['check_vs_predict_proba_max_abs']:.1e}", "",
           "### 검증 10분위 보정표", "", "| 분위 | n | p 범위 | 평균 p | 평균 y |", "|---|---|---|---|---|"]
    for x in v["deciles"]:
        L_.append(f"| {x['decile']} | {x['n']:,} | {x['p_lo']:.3f}–{x['p_hi']:.3f} | {x['mean_p']:.4f} | {x['mean_y']:.4f} |")
    h = pr["h1_power"]
    L_ += ["", f"### H1 검정력 (단측 α 0.0167, 검정력 80%)", "",
           f"- 봉인 규모 환산(20종목 × 1,468일, 주문 수 비례) SE {h['se_sealed_orders']:.5f} → MDE(AUC − 0.5) {h['mde_auc_minus_0.5_orders']['0.0167']:.4f}",
           f"- 참고: 일수만 비례 SE {h['se_sealed_days']:.5f} → MDE {h['mde_auc_minus_0.5_days']['0.0167']:.4f}", ""]
    for lab in ("L3D1", "L0D0"):
        L_ += [f"## {lab}: 후보별 평균·p95 IS와 R0 대비 (¢/주)", "",
               "| 방법 | 평균 IS | p95 | − R0 [95% CI] | p95 − R0 [95% CI] | 발동 | R1 | R2 | 지정가 | 2019 | 2020 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in R[R.variant == lab].iterrows():
            L_.append(f"| {r.method} | {r.mean_is_c:.3f} | {r.p95_is_c:.2f} | {r.minus_R0_c:+.3f} [{r.minus_R0_lo:+.3f}, {r.minus_R0_hi:+.3f}] | "
                      f"{r.p95_minus_R0_c:+.2f} [{r.p95_minus_R0_lo:+.2f}, {r.p95_minus_R0_hi:+.2f}] | {r.fired_frac:.3f} | {r.r1_parent_frac:.3f} | "
                      f"{r.r2_parent_frac:.3f} | {r.limit_frac:.3f} | {r.minus_R0_2019:+.3f} | {r.minus_R0_2020:+.3f} |")
        L_.append("")
    s = out["selection"]
    L_ += ["## A* 선택 (SPEC18 4절, L3D1)", "", f"- 평균 IS 최저 {s['lowest_mean']} ({s['lowest_mean_c']:.3f}¢), 0.02¢ 안 동률 {s['tied_within_0.02']}, "
           f"그중 p95 최저 {s['lowest_p95_among_tied']}, 발동 비율 {s['fired']} → **{s['astar']}**", ""]
    f = out["power_flags"]
    if f:
        L_ += ["## H2·H3 검정력 (1단계 MDE, α 0.0167)", "",
               f"- H2: A* − R0 = {f['H2']['effect_c']:+.4f}¢, MDE {f['H2']['mde_c']:.4f}¢ → **{'검정력 부족' if f['H2']['underpowered'] else '검정력 충분'}**",
               f"- H3: p95(A*) − p95(R0) = {f['H3']['effect_c']:+.3f}¢, MDE {f['H3']['mde_c']:.3f}¢ (일수만 비례; 주문 수 비례 {f['H3']['mde_orders_c']:.3f}¢) → "
               f"**{'검정력 부족' if f['H3']['underpowered'] else '검정력 충분'}**{'' if f['H3']['direction_ok'] else ' (방향도 반대)'}", ""]
    else:
        L_ += ["## H2·H3", "", "- A* = R0: Q2·Q3는 개발에서 개선 없음, 봉인 검정은 H1만 (α 0.05)", ""]
    L_ += ["## 지연 점검 (참고, 결정은 막대 t, 모든 행동은 t+1 호가)", "", "| 체결 가정 | 비교 | 평균 [95% CI] | p95 차이 [95% CI] |", "|---|---|---|---|"]
    for lab, x in out["delay"].items():
        for k, name in (("astar_delay_minus_R0_delay", "A*_지연 − E*_지연"), ("R0_delay_minus_R0", "E*_지연 − E*"), ("astar_delay_minus_astar", "A*_지연 − A*")):
            y_ = x[k]
            L_.append(f"| {lab} | {name} | {y_['mean_c']:+.3f} [{y_['ci95'][0]:+.3f}, {y_['ci95'][1]:+.3f}] | {y_['p95_diff_c']:+.2f} "
                      f"[{y_['p95_ci95'][0]:+.2f}, {y_['p95_ci95'][1]:+.2f}] |")
    c = out["checks"]
    L_ += ["", "## 회귀 점검 (2015–2020)", "", f"- 주문표·vol_pre = 연구 17 dev17: {c['parents_equal_dev17']}, {c['vol_pre_equal_dev17']}",
           f"- 주문별 출력 = dev17 (S0, S1, S2i, E*): {c['per_parent_equal_dev17']}",
           "- 평균 IS 3자리: " + ", ".join(f"{k} {v['got']:.3f}" for k, v in c["mean_is_3dp"].items()),
           f"- 1,000주 합 {c['shares_sum_to_qty']}, 규칙 행동 0 {c['no_rule_action']}, 0 이하 가격 시장가 {c['market_at_nonpositive_price']}",
           f"- 2차 실행 일관성: {out['consistency_pass2']}", f"- 실행 {out['runtime_s']} s"]
    with open(os.path.join(RES, "dev18_report.md"), "w") as fh:
        fh.write("\n".join(L_) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.workers)
