#!/usr/bin/env python3
"""Research 23 simulation and statistics shared by dev23.py, run_test23.py, regress23.py and test23.py (SPEC23 2-5,
SPEC23_ANSWERS, DEVIATIONS23.md 1). The engines of research 18 and 22 are imported and not changed: engine18.run with
research 18's run names and a P array, loader22 for the bars (sizes / 100 from 2025-11-03), signal22 for the adverse scores.
  runs        S0, S2i, R0 (= E*); the 19 candidates of SPEC23 4 (research 18's names R1_h, R2_l, R3_h_l) with the
              probability of the nine-input model; "m18:<candidate>" the same rule with the probability of research 18's
              frozen model (reference); "ctrl:<candidate>" the control with 1 - p (SPEC23 5; not a candidate)
  C3          per parent: S2i when the arrival spread is at most 4 cents, else E* (engine18.branch_c3)
  adv         the mean daily volume of the previous 20 trading days of the year (engine18.adv_prior on the day volumes of
              the plan's own files), needed by input 8; a pass over the files before the simulation
  IS          engine18.costs, cents per share; L3D1 (primary) and L0D0
  statistics  circular day-block bootstrap (21 days, 10,000 draws, seed 23): mean and two-sided 95% interval, one-sided
              p = share of bootstrap means >= 0, 95th percentile differences; the exact AUC of both models on the same
              scored points (decision quote valid, y defined) and on the same bootstrap draws
  A*          SPEC23 4 with ANSWERS A6 (select_astar)
No file is written here."""
import hashlib
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import feat23 as F  # noqa: E402
import model23 as M  # noqa: E402

C22, E, SG = F.C22, F.E, F.SG
L22 = C22.L22
# files of earlier studies used on the run path, never edited: sha256 of the file at its tag (core22 checks engine17,
# engine18, signal20 and loader15 itself)
FROZEN = {"study22_market_signal/core22.py": ("code-freeze-22", "954d6a22cc69df72f30ba109380e3434efbed4dabb6ea4a9f37ac361a3cac9b0"),
          "study22_market_signal/loader22.py": ("code-freeze-22", "370a2e138a1715b8f868f2e715179b80f6f14eb7f729c106768cb9f8c97117f7"),
          "study22_market_signal/signal22.py": ("code-freeze-22", "22f17a2c5deca70214050a597b03a6c8c79421910993ced1965c622923cb7845"),
          "study18_forecast/model18.json": ("test-run-18", "8dff572ef84ce73d9e2d7fbc6cb0fa7eb5bd2baf856ba90039da56bbbd46e439")}
MODEL18 = os.path.join(PROJ, "study18_forecast", "model18.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def frozen_differences():
    """Paths whose file differs from the sha256 of its tag (research 22's list and this one)."""
    return C22.frozen_differences() + [p for p, (_, h) in FROZEN.items() if sha(os.path.join(PROJ, p)) != h]


if frozen_differences():
    raise ImportError(f"files of earlier studies differ from their tags: {frozen_differences()}")

BASE = ("S0", "S2i", "R0")
CANDS = tuple(c for c in E.CANDS if c != "R0")                       # SPEC23 4: the 19 candidates in research 18's grid order
M18, CTRL = "m18:", "ctrl:"
SEED, TIE, EPS = 23, 0.02, 1e-12
CUT = "20251103"
ALPHA, POWER = 0.05, 0.8
jdefault, day_codes, mean_ci, one_sided, pair, visited, power = (C22.jdefault, C22.day_codes, C22.mean_ci, C22.one_sided, C22.pair,
                                                                 C22.visited, C22.power)     # these take the bootstrap index as an argument


def engine_name(run):
    """engine18's run name of a run."""
    for pre in (M18, CTRL):
        if run.startswith(pre):
            run = run[len(pre):]
            if run not in CANDS:
                raise KeyError(run)
            return run
    if run not in BASE + CANDS:
        raise KeyError(run)
    return run


def source(run):
    """Where a run's P comes from: None (no P), "m23", "m18" or "ctrl" (1 - p of the nine-input model)."""
    if run.startswith(M18):
        return "m18"
    if run.startswith(CTRL):
        return "ctrl"
    return "m23" if run in CANDS else None


def model18():
    """Research 18's frozen model, read only when the file equals its tag and its own sha256 file."""
    h = sha(MODEL18)
    if h != FROZEN["study18_forecast/model18.json"][1] or open(os.path.splitext(MODEL18)[0] + ".sha256").read().split()[0] != h:
        raise SystemExit("study18_forecast/model18.json differs from its frozen sha256")
    return json.load(open(MODEL18))


# ---------------------------------------------------------------- development plan
def dev_tasks(years):
    """(symbol-days of the years, calendar, plan info) from loader22.dev_plan: development sources only, nothing after 2020.
    The plan info is about these years (loader22's own counts are for 2015-2020 as a whole and are kept under other names)."""
    tasks, cal, p = L22.dev_plan()
    if any("sealed" in t["path"] or (t["ref_path"] and "sealed" in t["ref_path"]) for t in tasks):
        raise SystemExit("a sealed path in the development plan")
    tasks = [t for t in tasks if t["year"] in years]
    if not tasks or any(t["date"] > L22.DEV_LAST for t in tasks):
        raise SystemExit("no development symbol-day of the years, or a date after 2020")
    info = {"years": list(years), "plan_symbol_days_all_years": p["tasks"], "ref_files_all_years": p["ref_files"],
            "days_without_ref": [d for d in p["days_without_ref"] if int(d[:4]) in years]}
    return tasks, cal, info


# ---------------------------------------------------------------- the usual volume of a symbol (input 8)
def _dayvol(t):
    return t["sym"], t["date"], int(L22.read_npz(t["path"], t["sha256"], f"{t['sym']} {t['date']}")["vol"].sum())


def with_adv(tasks, cal, workers=4):
    """(tasks with "adv", info). adv: the mean of the day volumes (sum of the bar column vol) of the symbol over the previous
    20 trading days of the same year that are in the plan, when at least 5 of them are (engine18.adv_prior); else nan."""
    if workers <= 1:
        res = [_dayvol(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            res = list(pool.imap(_dayvol, tasks, chunksize=32))
    dayvol = {(s, d): v for s, d, v in res}
    adv = E.adv_prior(dayvol, cal, list(dayvol))
    out = [{**t, "adv": adv[(t["sym"], t["date"])][0]} for t in tasks]
    a = np.array([t["adv"] for t in out], dtype=np.float64)
    info = {"symbol_days": len(out), "adv_undefined": int(np.isnan(a).sum()), "adv_zero": int((a == 0).sum()),
            "day_volume_zero": int(sum(v == 0 for v in dayvol.values()))}
    return out, info


# ---------------------------------------------------------------- one symbol-day
def run_parents(b, ref, close_bar, dirs, starts, adv, runs, delay_runs=(), model=None, m18=None):
    """The nine inputs and y at the decision grid of given parents, the probabilities of the models handed in, and every run
    under L0 and L3 (delay_runs with delay 1 after them)."""
    raw = F.raw(b, ref, close_bar, dirs, starts, adv)
    P = {"m23": M.predict(model, raw["X"], raw["valid"]) if model is not None else None,
         "m18": E.predict(m18, raw["f"]) if m18 is not None else None}
    P["ctrl"] = None if P["m23"] is None else 1.0 - P["m23"]

    def p_of(name):
        s = source(name)
        if s is not None and P[s] is None:
            raise ValueError(f"{name}: no model for its probability")
        return None if s is None else P[s]
    names = list(runs) + list(delay_runs)
    outs = np.zeros((len(E.LRULES), len(names), len(dirs), E.NCOL), dtype=np.int64)
    for li, lr in enumerate(E.LRULES):
        for ri, name in enumerate(names):
            outs[li, ri] = E.run(b, engine_name(name), lr, dirs, starts, p_of(name), 1 if ri >= len(runs) else 0)
    return {"outs": outs, "raw": raw, "P23": P["m23"], "P18": P["m18"]}


def symday(t, spec):
    """One symbol-day of a plan: the 20 parents of the date (engine18.parents) with a valid start quote, then run_parents.
    spec: runs, delay_runs, model, m18, grid (True: the raw inputs X are returned)."""
    b, ref = L22.load(t)
    dirs, starts = E.parents(t["date"], t["close_bar"])
    ok = E.quote_ok(b)[starts]
    dirs, starts = dirs[ok], starts[ok]
    r = run_parents(b, ref, t["close_bar"], dirs, starts, t["adv"], spec["runs"], spec["delay_runs"], spec["model"], spec["m18"])
    raw = r["raw"]
    out = {"sym": t["sym"], "date": t["date"], "dir": dirs, "start": starts, "mid2": (b["bid"] + b["ask"])[starts],
           "spread": (b["ask"] - b["bid"])[starts], "vol_pre": raw["vp"], "n_dropped": int((~ok).sum()), "outs": r["outs"],
           "y": raw["y"], "valid": raw["valid"], "counts": raw["counts"], "has_ref": t["ref_path"] is not None}
    if spec["grid"]:
        out["X"] = raw["X"]
    for k in ("P23", "P18"):
        if r[k] is not None:
            out[k] = r[k]
    return out


def _work(a):
    return symday(*a)


def simulate(tasks, runs=(), delay_runs=(), workers=4, model=None, m18=None, grid=False):
    """(par, OUT, G, info). par: one row per parent (date, sym, dir, start, mid2, spread, vol_pre). OUT [lrule, run, parent,
    col]. G: y and valid [parent, NG], P23 / P18 for the models handed in, X [parent, NG, 9] when grid is True."""
    if not tasks:
        raise ValueError("no symbol-day in the plan")
    spec = {"runs": tuple(runs), "delay_runs": tuple(delay_runs), "model": model, "m18": m18, "grid": bool(grid)}
    args = [(t, spec) for t in tasks]
    if workers <= 1:
        res = [_work(a) for a in args]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            res = list(pool.imap(_work, args, chunksize=8))
    n = [len(r["dir"]) for r in res]
    par = pd.DataFrame({"date": np.repeat([r["date"] for r in res], n), "sym": np.repeat([r["sym"] for r in res], n),
                        **{k: np.concatenate([r[k] for r in res]) for k in ("dir", "start", "mid2", "spread", "vol_pre")}})
    OUT = np.concatenate([r["outs"] for r in res], axis=2)
    G = {k: np.concatenate([r[k] for r in res]) for k in ("y", "valid", "X", "P23", "P18") if k in res[0]}
    info = {"symbol_days": len(tasks), "parents": int(len(par)), "dropped_parents": int(sum(r["n_dropped"] for r in res)),
            "symbol_days_without_parents": int(sum(k == 0 for k in n)), "symbol_days_without_ref": int(sum(not r["has_ref"] for r in res)),
            "grid_counts": {k: int(sum(r["counts"][k] for r in res)) for k in res[0]["counts"]},
            "runs": list(runs), "delay_runs": list(delay_runs), "workers": workers}
    return par, OUT, G, info


# ---------------------------------------------------------------- IS and statistics
def boot(n_days):
    idx = E.boot_index(n_days, seed=SEED)
    return idx, E.day_counts(idx, n_days)


def is_table(par, OUT, names, lr, dep):
    """{run: per-parent IS in cents per share}, C3 added when S2i and R0 are among the names; and the outs per run."""
    o = OUT[E.LRULES.index(lr)]
    outs = {m: o[i] for i, m in enumerate(names)}
    if "S2i" in outs and "R0" in outs:
        outs["C3"] = np.where(E.branch_c3(par.spread.to_numpy())[:, None], outs["S2i"], outs["R0"])
    dirs, mid2 = par.dir.to_numpy(), par.mid2.to_numpy()
    return {m: E.costs(v, dirs, mid2, dep) for m, v in outs.items()}, outs


def acts(run):
    """True for a run whose rule can act differently from E* (a candidate, its research 18 twin or its control)."""
    return run not in ("C3",) + BASE and "@" not in run


def firing(out, run):
    """Shares of parents and of visited decision bars where the rule acted differently from E*."""
    if not acts(run):
        return {"parents": 0.0, "decisions": 0.0, "early_parents": 0.0, "hold_parents": 0.0, "holds_per_parent": 0.0}
    f = E.fired(out, engine_name(run))
    n = (out[:, 9] > 0).sum() + out[:, 10].sum()
    return {"parents": float(f.mean()), "decisions": float(n / visited(out).sum()), "early_parents": float((out[:, 9] > 0).mean()),
            "hold_parents": float((out[:, 10] > 0).mean()), "holds_per_parent": float(out[:, 10].mean())}


def run_row(run, IS, outs, ref, codes, n_days, idx, W, years):
    """One table row: mean IS and interval, p95, the paired difference to `ref` (same parents), firing, by year."""
    x, o = IS[run], outs[run]
    d = x - IS[ref]
    pe, pdist = E.p95_diff_boot(x, IS[ref], codes, W)
    return {"run": run, "n_parents": int(len(x)), **{f"is_{k}": v for k, v in mean_ci(x, codes, n_days, idx).items()},
            "p95": float(np.percentile(x, 95)), **{f"minus_{ref}_{k}": v for k, v in one_sided(d, codes, n_days, idx).items()},
            f"p95_minus_{ref}": pe, f"p95_minus_{ref}_ci_lo": float(np.percentile(pdist, 2.5)), f"p95_minus_{ref}_ci_hi": float(np.percentile(pdist, 97.5)),
            "firing": firing(o, run), "limit_share_frac": float((o[:, 0] / E.QTY).mean()), "deadline_parent_frac": float((o[:, 6] > 0).mean()),
            "mean_done_s": float(o[:, 5].mean()), "market_nonpositive_price": int(o[:, 11].sum()),
            "by_year": {str(y): {"is_mean": float(x[years == y].mean()), f"minus_{ref}": float(d[years == y].mean()), "n_parents": int((years == y).sum())}
                        for y in sorted(set(years))}}


def subset_stat(d, dates, mask):
    """one_sided on the parents of `mask`, with their own day codes and bootstrap index (seed 23)."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    codes, days = day_codes(dates[mask])
    if len(days) < 2:
        return {"n": n, "days": len(days), "mean": float(d[mask].mean())}
    return {"n": n, "days": len(days), **one_sided(d[mask], codes, len(days), E.boot_index(len(days), seed=SEED))}


def tables(par, OUT, names, report):
    """({variant: {run: row}}, (day codes, days, bootstrap index, day weights)) for the runs in `report` (differences to R0)."""
    codes, days = day_codes(par.date.to_numpy())
    idx, W = boot(len(days))
    years = par.date.str[:4].astype(int).to_numpy()
    out = {}
    for lab, lr, dep in E.VARIANTS:
        IS, outs = is_table(par, OUT, names, lr, dep)
        out[lab] = {m: run_row(m, IS, outs, "R0", codes, len(days), idx, W, years) for m in report}
    return out, (codes, days, idx, W)


def delay_table(par, OUT, names, runs, st):
    """Reference only (delay 1: decided at bar t, done at bar t + 1): per variant the mean IS of each delayed run and its
    difference to the undelayed run, and for a candidate the delayed difference to delayed R0."""
    codes, days, idx, _ = st
    res = {}
    for lab, lr, dep in E.VARIANTS:
        IS, _ = is_table(par, OUT, names, lr, dep)
        v = {}
        for m in runs:
            v[f"{m}_delay_mean_c"] = float(IS[f"{m}@d1"].mean())
            v[f"{m}_delay_minus_{m}"] = one_sided(IS[f"{m}@d1"] - IS[m], codes, len(days), idx)
            if m != "R0":
                v[f"{m}_delay_minus_R0_delay"] = one_sided(IS[f"{m}@d1"] - IS["R0@d1"], codes, len(days), idx)
        res[lab] = v
    return res


def control_table(par, OUT, names, astar, st):
    """SPEC23 5 (no verdict, not a candidate): the control of A* with 1 - p under both variants: its mean IS and firing,
    control - E* and A* - control over the same parents (mean, one-sided p, interval, p95 difference)."""
    codes, days, idx, W = st
    f = CTRL + astar
    res = {}
    for lab, lr, dep in E.VARIANTS:
        IS, outs = is_table(par, OUT, names, lr, dep)
        res[lab] = {"control": f, "control_is": mean_ci(IS[f], codes, len(days), idx), "control_p95": float(np.percentile(IS[f], 95)),
                    "control_firing": firing(outs[f], f), "astar_firing": firing(outs[astar], astar),
                    "control_minus_Estar": pair(IS, f, "R0", codes, len(days), idx, W),
                    "Astar_minus_control": pair(IS, astar, f, codes, len(days), idx, W)}
    return res


def select_astar(rows, r0_mean):
    """SPEC23 4 with ANSWERS A6. rows: {candidate: {"mean", "p95", "fired"}} under L3D1 in the validation period.
    The best candidate must be lower than R0 by more than 0.02 cents, else A* = R0 (R0 takes no part in the tie rule).
    Among the candidates within 0.02 cents of the lowest mean: the lower p95, then the lower share of fired parents, then
    the grid order. Comparisons use a tolerance of 1e-12 cents."""
    best = min(rows[c]["mean"] for c in CANDS)
    lowest = min(CANDS, key=lambda c: (rows[c]["mean"], CANDS.index(c)))
    rec = {"lowest_mean": lowest, "lowest_mean_c": best, "r0_mean_c": r0_mean, "r0_minus_lowest_c": r0_mean - best, "threshold_c": TIE}
    if not r0_mean - best > TIE + EPS:
        return "R0", {**rec, "improves_in_development": False, "astar": "R0"}
    tied = [c for c in CANDS if rows[c]["mean"] - best <= TIE + EPS]
    p95 = min(rows[c]["p95"] for c in tied)
    low = [c for c in tied if rows[c]["p95"] == p95]
    pick = min(low, key=lambda c: (rows[c]["fired"], CANDS.index(c)))
    return pick, {**rec, "improves_in_development": True, "tied_within_0.02": tied, "lowest_p95_among_tied": low,
                  "fired": {c: rows[c]["fired"] for c in low}, "astar": pick}


def holm(p):
    """engine18.holm on the hypotheses that exist (1 or 2): levels 0.05, or 0.025 and 0.05."""
    return E.holm(p, alpha=ALPHA)


def daily_rows(par, OUT, names):
    """Daily file: one row per date, variant and run with the number of parents and the sum of their IS (cents per share)."""
    rows = []
    codes, days = day_codes(par.date.to_numpy())
    n = np.bincount(codes, minlength=len(days))
    for lab, lr, dep in E.VARIANTS:
        IS, _ = is_table(par, OUT, names, lr, dep)
        for m, x in IS.items():
            s = np.bincount(codes, weights=x, minlength=len(days))
            rows += [(d, lab, m, int(k), float(v)) for d, k, v in zip(days, n, s)]
    return pd.DataFrame(rows, columns=["date", "variant", "run", "n_parents", "is_sum_c"])


# ---------------------------------------------------------------- prediction (SPEC23 3, 5)
def scored(par, G):
    """The scored points (decision quote valid, y defined) as flat arrays: keep mask [parent, NG], y, day codes over the days
    with scored points, those days, the parent index of every point, and the counts."""
    y, valid = G["y"], G["valid"]
    keep = valid & (y >= 0)
    pc, days_all = day_codes(par.date.to_numpy())
    owner = np.repeat(np.arange(len(par), dtype=np.int64), E.NG).reshape(len(par), E.NG)[keep]
    u, codes = np.unique(pc[owner], return_inverse=True)
    n = {"grid_points": int(y.size), "valid_points": int(valid.sum()), "y_missing_points": int((valid & (y < 0)).sum()),
         "y_missing_share_of_valid": float((valid & (y < 0)).sum() / max(int(valid.sum()), 1)), "scored_points": int(keep.sum()), "days": int(len(u))}
    return keep, y[keep].astype(np.int8), codes.astype(np.int64), [days_all[i] for i in u], owner, n


def auc_point(p, y):
    """The exact AUC (Mann-Whitney, ties 1/2) of the points, or None when one class is missing."""
    if len(y) == 0 or y.min() == y.max():
        return None
    return float(E.auc_boot(p, y, np.zeros(len(y), dtype=np.int64), np.ones((1, 1)))[0])


def pred_metrics(p, y, codes, base_train, W=None):
    """AUC, hit rate at 0.5, Brier score and the Brier skill score against the constant training base rate; with day
    weights W also the bootstrap distribution of the AUC (se, 95% interval, share of draws <= 0.5)."""
    T = int(codes.max()) + 1
    auc, dist = E.auc_boot(p, y, codes, W if W is not None else np.ones((1, T)))
    brier, ref = float(np.mean((p - y) ** 2)), float(np.mean((base_train - y) ** 2))
    out = {"n": int(len(y)), "base_rate": float(y.mean()), "auc": auc, "accuracy_0.5": float(np.mean((p >= 0.5) == (y == 1))),
           "brier": brier, "brier_ref_train_base_rate": ref, "bss": 1.0 - brier / ref}
    if W is not None:
        lo, hi = np.percentile(dist, [2.5, 97.5])
        out["auc_boot"] = {"se": float(dist.std(ddof=1)), "ci_lo": float(lo), "ci_hi": float(hi), "share_le_0.5": float(np.mean(dist <= 0.5))}
    return out, (dist if W is not None else None)


def prediction(par, G, base23, base18, subsets=None):
    """Both models on the same scored points and the same bootstrap draws (day blocks 21, 10,000, seed 23): their metrics,
    AUC(combined) - AUC(research 18) with its 95% interval, the AUC by year, the calibration deciles of the combined model
    and, for parent masks in `subsets`, the two AUCs of those parents' points. Also the per-day sums for the daily file."""
    keep, y, codes, days, owner, n = scored(par, G)
    p23, p18 = G["P23"][keep], G["P18"][keep]
    T = len(days)
    W = E.day_counts(E.boot_index(T, seed=SEED), T)
    m23, d23 = pred_metrics(p23, y, codes, base23, W)
    m18, d18 = pred_metrics(p18, y, codes, base18, W)
    diff = d23 - d18
    lo, hi = np.percentile(diff, [2.5, 97.5])
    yrs = np.array([int(d[:4]) for d in days])[codes]
    edges = np.quantile(p23, np.linspace(0, 1, 11))
    k = np.clip(np.searchsorted(edges[1:-1], p23, side="right"), 0, 9)
    out = {"counts": n, "combined": m23, "study18_forecast": m18,
           "auc_diff": {"mean": m23["auc"] - m18["auc"], "ci_lo": float(lo), "ci_hi": float(hi), "share_le_0": float(np.mean(diff <= 0))},
           "by_year": {str(v): {"n": int((yrs == v).sum()), "auc_combined": auc_point(p23[yrs == v], y[yrs == v]),
                                "auc_research18": auc_point(p18[yrs == v], y[yrs == v]), "base_rate": float(y[yrs == v].mean())} for v in sorted(set(yrs))},
           "deciles_combined": [{"decile": int(i + 1), "n": int((k == i).sum()), "p_lo": float(edges[i]), "p_hi": float(edges[i + 1]),
                                 "mean_p": float(p23[k == i].mean()) if (k == i).any() else None,
                                 "mean_y": float(y[k == i].mean()) if (k == i).any() else None} for i in range(10)],
           "p_range_combined": [float(p23.min()), float(p23.max())],
           "boot": {"B": E.BOOT_B, "block": E.BOOT_BLOCK, "seed": SEED}}
    if subsets:
        out["subsets"] = {}
        for name, mask in subsets.items():
            m = np.asarray(mask)[owner]
            out["subsets"][name] = {"n": int(m.sum()), "auc_combined": auc_point(p23[m], y[m]), "auc_research18": auc_point(p18[m], y[m])}
    bc = lambda w: np.bincount(codes, weights=w, minlength=T)                                    # noqa: E731
    daily = pd.DataFrame({"date": days, "n_scored": np.bincount(codes, minlength=T), "y_sum": bc(y), "p_combined_sum": bc(p23), "p_research18_sum": bc(p18),
                          "sq_err_combined_sum": bc((p23 - y) ** 2), "sq_err_research18_sum": bc((p18 - y) ** 2),
                          "hit_combined": bc((p23 >= 0.5) == (y == 1)), "hit_research18": bc((p18 >= 0.5) == (y == 1))})
    return out, daily
