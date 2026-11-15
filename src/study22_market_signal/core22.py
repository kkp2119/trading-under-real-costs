#!/usr/bin/env python3
"""Research 22 simulation and statistics shared by dev22.py, run_test22.py, regress22.py and test22.py (SPEC22 2-5,
SPEC22_ANSWERS, DEVIATIONS22.md 1). Research 18's engine is imported and not changed: engine18.run with the run names
S0, S2i, R0 and, for the candidates, R1_0.75 / R2_0.25 / R3_0.75_0.25 on a P array of flags (signal22.flags).
  runs        S0, S2i, G0 (= engine R0 = E*), the 12 candidates <basis>_z<z_th>_<G1|G2|G3> in the order of SPEC22 4,
              half_G1 / half_G2 / half_G3 (regression only: every flag 0.5, no signal is computed for them) and
              flip_<candidate> (SPEC22_ADDENDUM D1, a control and not a candidate: the same basis, threshold and rule
              with the sign of the adverse score reversed)
  C3          per parent: S2i when the arrival spread is at most 4 cents, else E* (engine18.branch_c3)
  IS          engine18.costs, cents per share; L3D1 (primary) and L0D0
  statistics  circular day-block bootstrap (21 days, 10,000 draws, seed 22) on the days with parent orders: mean and
              two-sided 95% interval of a per-parent series (ratio of day sums to day counts), one-sided p = share of
              bootstrap means >= 0, the 95th percentile and its paired difference (engine18.p95_diff_boot)
  firing      parents where the rule acted differently from E* (engine18.fired), and actions per decision bar visited
  A*          SPEC22 4 with ANSWERS A4 (select_astar)
  prediction  SPEC22 5 / ANSWERS A5: the adverse mid change 60 and 180 seconds after every decision grid point with a
              signal, path independent, for all points and for the points of each condition
No file is written here."""
import hashlib
import math
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
# engine files of earlier studies, imported and never edited: sha256 of the file at its tag (test22.py reads the tags)
FROZEN = {"study17_new_stocks/engine17.py": ("test-run-17", "2a9ddcb7db7b96201cbd63b62e44c7c00140447a947e883aeeebf6a542d8d42a"),
          "study18_forecast/engine18.py": ("test-run-18", "21d9459edbc7cfa9c5c8077225b6db9ab0c2bc5a3535ab81eaaa57292a707065"),
          "study20_pairs/signal20.py": ("code-freeze-20", "e5d1127ccad05c9980cfc6c79dbffbfc2fb19c5fef40d60674af05cf33bad224"),
          "study15_ml_execution/loader15.py": ("test-run-18", "d5d2beb11cea016265ab2af7828e0e8db8a4c2d9faa19e71cb589dff99b07fd6")}


def frozen_differences():
    """Paths of FROZEN whose file differs from the sha256 of its tag."""
    return [p for p, (_, h) in FROZEN.items() if hashlib.sha256(open(os.path.join(PROJ, p), "rb").read()).hexdigest() != h]


if frozen_differences():
    raise ImportError(f"engine files differ from their tags: {frozen_differences()}")
sys.path.insert(0, HERE)
for _r in ("study17_new_stocks", "study18_forecast", "study20_pairs"):
    sys.path.insert(1, os.path.join(PROJ, _r))
import engine18 as E  # noqa: E402
import loader22 as L22  # noqa: E402
import signal22 as SG  # noqa: E402

BASES, ZTH = SG.BASES, SG.ZTH
RULES = ("G1", "G2", "G3")
CANDS = tuple(f"{b}_z{z:g}_{r}" for b in BASES for z in ZTH for r in RULES)            # SPEC22 4, grid order
HALF = tuple(f"half_{r}" for r in RULES)                                               # regression: all flags 0.5
FLIP = "flip_"                                                                         # ADDENDUM D1: control with -a
ENGINE = {"G1": "R1_0.75", "G2": "R2_0.25", "G3": "R3_0.75_0.25"}                      # 0.75 / 0.25 only separate 1.0, 0.5, 0.0
BASE = ("S0", "S2i", "G0")
SEED, TIE, EPS = 22, 0.02, 1e-12
HORIZONS = (60, 180)
GROUPS = ("all",) + tuple(f"z{z:g}_{c}" for z in ZTH for c in ("early", "hold"))       # early: a >= z_th, hold: a <= -z_th
CUT = "20251103"
ALPHA, POWER = 0.05, 0.8
OUT_DTYPE = np.int64


def flip_name(c):
    """The control of candidate c with the sign of the adverse score reversed (SPEC22_ADDENDUM D1)."""
    if c not in CANDS:
        raise KeyError(c)
    return FLIP + c


def is_flip(run):
    return run.startswith(FLIP) and run[len(FLIP):] in CANDS


def signalled(run):
    """The candidate whose basis and threshold a run uses (the candidate itself or the one behind a control), else None."""
    c = run[len(FLIP):] if is_flip(run) else run
    return c if c in CANDS else None


def parse(c):
    """(basis, z_th, rule) of a candidate name (or of the candidate behind a control)."""
    b, z, r = (c[len(FLIP):] if is_flip(c) else c).split("_")
    return b, float(z[1:]), r


def engine_name(run):
    if run == "G0":
        return "R0"
    if run in ("S0", "S1", "S2i"):
        return run
    return ENGINE[run.split("_")[-1]]


def bases_of(runs):
    return tuple(b for b in BASES if any(signalled(r) and parse(r)[0] == b for r in runs))


def jdefault(o):
    """json default: numpy scalars and arrays as plain numbers and lists; anything else is an error."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o).__name__}")


# ---------------------------------------------------------------- one symbol-day
def prediction(b, close_bar, U, a, dirs):
    """[group, horizon, 3]: sum of the adverse mid change in bp, points counted, points left out (the bar u + h is after the
    session's last bar or has no valid quote)."""
    m2 = b["bid"] + b["ask"]
    okm = (b["bid"] > 0) & (b["ask"] > 0) & (b["ask"] >= b["bid"])
    has = ~np.isnan(a)
    masks = [has] + [m for z in ZTH for m in (has & (a >= z), has & (a <= -z))]
    out = np.zeros((len(GROUPS), len(HORIZONS), 3))
    base = np.where(has, m2[U], 1).astype(np.float64)
    for hi, h in enumerate(HORIZONS):
        F = U + h
        inside = F <= close_bar - 1
        Fc = np.minimum(F, close_bar - 1)
        fut = inside & okm[Fc]
        ret = (m2[Fc] / base - 1.0) * dirs[:, None] * 1e4
        for gi, m in enumerate(masks):
            use = m & fut
            out[gi, hi] = (ret[use].sum(), use.sum(), (m & ~fut).sum())
    return out


def run_parents(b, ref, close_bar, dirs, starts, runs, delay_runs=(), bases=None):
    """The runs of given parent orders on one symbol-day's bars (ref: the SPY bars or None): the flags of the bases in use,
    every run under L0 and L3 (delay_runs with delay 1 after them). The signal counts and prediction sums are returned
    for the bases of the candidates among the runs and for the bases named in `bases`."""
    names = list(runs) + list(delay_runs)
    need = set(bases_of(names)) | set(bases or ())                     # the bases of the candidates, plus those asked for
    use = tuple(b for b in BASES if b in need)
    A, sig, pred = {}, {}, {}
    for basis in use:
        g, z, has, fl = SG.z_grid(basis, b, ref, close_bar)
        U, zd = SG.at_decisions(g, z, has, starts)
        A[basis] = SG.adverse(zd, dirs)
        _, fd = SG.at_decisions(g, fl, has, starts)
        sig[basis] = (int(zd.size), int((~np.isnan(zd)).sum()), int(np.nansum(fd)))
        pred[basis] = prediction(b, close_bar, U, A[basis], dirs)
    half = np.full((len(dirs), E.NG), SG.NONE)

    def p_of(name):
        if signalled(name):
            basis, zth, _ = parse(name)
            return SG.flags(-A[basis] if is_flip(name) else A[basis], zth)        # the control reverses the sign of a
        return half if name in HALF else None
    outs = np.zeros((len(E.LRULES), len(names), len(dirs), E.NCOL), dtype=OUT_DTYPE)
    for li, lr in enumerate(E.LRULES):
        for ri, name in enumerate(names):
            outs[li, ri] = E.run(b, engine_name(name), lr, dirs, starts, p_of(name), 1 if ri >= len(runs) else 0)
    return {"outs": outs, "sig": sig, "pred": pred, "A": A}


def symday(t, runs, delay_runs=(), bases=None):
    """One symbol-day of a plan: the 20 parents of the date (engine18.parents) with a valid start quote, then run_parents."""
    b, ref = L22.load(t)
    dirs, starts = E.parents(t["date"], t["close_bar"])
    ok = E.quote_ok(b)[starts]
    dirs, starts = dirs[ok], starts[ok]
    r = run_parents(b, ref, t["close_bar"], dirs, starts, runs, delay_runs, bases)
    return {"sym": t["sym"], "date": t["date"], "dir": dirs, "start": starts, "mid2": (b["bid"] + b["ask"])[starts],
            "spread": (b["ask"] - b["bid"])[starts], "n_dropped": int((~ok).sum()), "outs": r["outs"], "sig": r["sig"], "pred": r["pred"],
            "has_ref": t["ref_path"] is not None}


def _work(a):
    return symday(*a)


def simulate(tasks, runs, delay_runs=(), workers=4, bases=None):
    """(par, OUT, sd, info). par: one row per parent (date, sym, dir, start, mid2, spread). OUT [lrule, run, parent, col].
    sd: one row per symbol-day with the signal counts and prediction sums of each basis."""
    args = [(t, tuple(runs), tuple(delay_runs), bases) for t in tasks]
    if workers <= 1:
        res = [_work(a) for a in args]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            res = list(pool.imap(_work, args, chunksize=8))
    n = [len(r["dir"]) for r in res]
    par = pd.DataFrame({"date": np.repeat([r["date"] for r in res], n), "sym": np.repeat([r["sym"] for r in res], n),
                        **{k: np.concatenate([r[k] for r in res]) for k in ("dir", "start", "mid2", "spread")}})
    OUT = np.concatenate([r["outs"] for r in res], axis=2)
    use = tuple(res[0]["sig"]) if res else ()
    sd = {"date": np.array([r["date"] for r in res]), "sym": np.array([r["sym"] for r in res]), "bases": use,
          "sig": {b: np.array([r["sig"][b] for r in res], dtype=np.int64) for b in use},
          "pred": {b: np.stack([r["pred"][b] for r in res]) for b in use}}
    info = {"symbol_days": len(tasks), "parents": int(len(par)), "dropped_parents": int(sum(r["n_dropped"] for r in res)),
            "symbol_days_without_ref": int(sum(not r["has_ref"] for r in res)), "runs": list(runs), "delay_runs": list(delay_runs),
            "workers": workers}
    return par, OUT, sd, info


# ---------------------------------------------------------------- IS and statistics
def day_codes(dates):
    days = sorted(set(dates))
    return pd.Categorical(dates, categories=days).codes.astype(np.int64), days


def boot(n_days):
    idx = E.boot_index(n_days, seed=SEED)
    return idx, E.day_counts(idx, n_days)


def is_table(par, OUT, names, lr, dep):
    """{run: per-parent IS in cents per share}, C3 added when S2i and G0 are among the names; and the outs per run."""
    o = OUT[E.LRULES.index(lr)]
    outs = {m: o[i] for i, m in enumerate(names)}
    if "S2i" in outs and "G0" in outs:
        outs["C3"] = np.where(E.branch_c3(par.spread.to_numpy())[:, None], outs["S2i"], outs["G0"])
    dirs, mid2 = par.dir.to_numpy(), par.mid2.to_numpy()
    return {m: E.costs(v, dirs, mid2, dep) for m, v in outs.items()}, outs


def mean_ci(x, codes, n_days, idx):
    est, lo, hi = E.diff_ci(x, codes, n_days, idx)
    return {"mean": est, "ci_lo": lo, "ci_hi": hi}


def one_sided(d, codes, n_days, idx):
    """Mean of d, one-sided p (share of bootstrap means >= 0), two-sided 95% interval and the one-sided 97.5% upper bound."""
    est, dist = E.boot_stat(d, codes, n_days, idx)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return {"mean": est, "p_one_sided": float(np.mean(dist >= 0)), "ci_lo": float(lo), "ci_hi": float(hi), "upper_97_5": float(hi)}


def visited(out):
    """Decision bars (t = 0..295 s) a method-2 parent reached with shares left: a parent that ends with a market order at
    t = done has seen done / 5 + 1 of them (60 at the deadline), one that ends with a limit fill at done has seen those
    before the fill."""
    done = out[:, 5]
    return np.where(out[:, 2] > 0, np.minimum(done // E.STEP + 1, E.NG), np.minimum(-(-done // E.STEP), E.NG))


def firing(out, run):
    """Shares of parents and of visited decision bars where the rule acted differently from E*."""
    if not signalled(run) and run not in HALF:
        return {"parents": 0.0, "decisions": 0.0, "early_parents": 0.0, "hold_parents": 0.0, "holds_per_parent": 0.0}
    f = E.fired(out, engine_name(run))
    acts = (out[:, 9] > 0).sum() + out[:, 10].sum()
    return {"parents": float(f.mean()), "decisions": float(acts / visited(out).sum()), "early_parents": float((out[:, 9] > 0).mean()),
            "hold_parents": float((out[:, 10] > 0).mean()), "holds_per_parent": float(out[:, 10].mean())}


def run_row(run, IS, outs, ref, codes, n_days, idx, W, years):
    """One table row: mean IS and interval, p95, the paired difference to `ref` (same parents), firing, by year."""
    x, o = IS[run], outs[run]
    d = x - IS[ref]
    pe, pdist = E.p95_diff_boot(x, IS[ref], codes, W)
    r = {"run": run, "n_parents": int(len(x)), **{f"is_{k}": v for k, v in mean_ci(x, codes, n_days, idx).items()},
         "p95": float(np.percentile(x, 95)), **{f"minus_{ref}_{k}": v for k, v in one_sided(d, codes, n_days, idx).items()},
         f"p95_minus_{ref}": pe, f"p95_minus_{ref}_ci_lo": float(np.percentile(pdist, 2.5)), f"p95_minus_{ref}_ci_hi": float(np.percentile(pdist, 97.5)),
         "firing": firing(o, run), "limit_share_frac": float((o[:, 0] / E.QTY).mean()), "deadline_parent_frac": float((o[:, 6] > 0).mean()),
         "mean_done_s": float(o[:, 5].mean()), "market_nonpositive_price": int(o[:, 11].sum()),
         "by_year": {str(y): {"is_mean": float(x[years == y].mean()), f"minus_{ref}": float(d[years == y].mean()), "n_parents": int((years == y).sum())}
                     for y in sorted(set(years))}}
    return r


def subset_stat(d, dates, mask):
    """one_sided on the parents of `mask`, with their own day codes and bootstrap index (seed 22)."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    codes, days = day_codes(dates[mask])
    if len(days) < 2:
        return {"n": n, "days": len(days), "mean": float(d[mask].mean())}
    return {"n": n, "days": len(days), **one_sided(d[mask], codes, len(days), E.boot_index(len(days), seed=SEED))}


def tables(par, OUT, names, report):
    """({variant: {run: row}}, (day codes, days, bootstrap index, day weights)) for the runs in `report` (differences to G0)."""
    codes, days = day_codes(par.date.to_numpy())
    idx, W = boot(len(days))
    years = par.date.str[:4].astype(int).to_numpy()
    out = {}
    for lab, lr, dep in E.VARIANTS:
        IS, outs = is_table(par, OUT, names, lr, dep)
        out[lab] = {m: run_row(m, IS, outs, "G0", codes, len(days), idx, W, years) for m in report}
    return out, (codes, days, idx, W)


def pair(IS, a, b, codes, n_days, idx, W):
    """IS(a) - IS(b) over the same parents: mean, one-sided p, interval, and the difference of the 95th percentiles."""
    r = one_sided(IS[a] - IS[b], codes, n_days, idx)
    pe, pdist = E.p95_diff_boot(IS[a], IS[b], codes, W)
    return {**r, "p95_diff": pe, "p95_ci_lo": float(np.percentile(pdist, 2.5)), "p95_ci_hi": float(np.percentile(pdist, 97.5))}


def delay_table(par, OUT, names, runs, st):
    """Reference only (delay 1: decided at bar t, done at bar t + 1): per variant the mean IS of each delayed run and its
    difference to the undelayed run, and for a candidate the delayed difference to delayed G0."""
    codes, days, idx, _ = st
    res = {}
    for lab, lr, dep in E.VARIANTS:
        IS, _ = is_table(par, OUT, names, lr, dep)
        v = {}
        for m in runs:
            v[f"{m}_delay_mean_c"] = float(IS[f"{m}@d1"].mean())
            v[f"{m}_delay_minus_{m}"] = one_sided(IS[f"{m}@d1"] - IS[m], codes, len(days), idx)
            if m != "G0":
                v[f"{m}_delay_minus_G0_delay"] = one_sided(IS[f"{m}@d1"] - IS["G0@d1"], codes, len(days), idx)
        res[lab] = v
    return res


def control_table(par, OUT, names, astar, st):
    """SPEC22_ADDENDUM D1 (no verdict, not a candidate): the control of A* with the reversed sign under both variants: its
    mean IS and firing, control - E* and A* - control over the same parents (mean, one-sided p, interval, p95 difference)."""
    codes, days, idx, W = st
    f = flip_name(astar)
    res = {}
    for lab, lr, dep in E.VARIANTS:
        IS, outs = is_table(par, OUT, names, lr, dep)
        res[lab] = {"control": f, "control_is": mean_ci(IS[f], codes, len(days), idx), "control_p95": float(np.percentile(IS[f], 95)),
                    "control_firing": firing(outs[f], f), "astar_firing": firing(outs[astar], astar),
                    "control_minus_Estar": pair(IS, f, "G0", codes, len(days), idx, W),
                    "Astar_minus_control": pair(IS, astar, f, codes, len(days), idx, W)}
    return res


def select_astar(rows, g0_mean):
    """SPEC22 4 with ANSWERS A4. rows: {candidate: {"mean", "p95", "fired"}} under L3D1 in the development period.
    The best candidate must be lower than G0 by more than 0.02 cents, else A* = G0 (G0 takes no part in the tie rule).
    Among the candidates within 0.02 cents of the lowest mean: the lower p95, then the lower share of fired parents, then
    the grid order. Comparisons use a tolerance of 1e-12 cents."""
    best = min(rows[c]["mean"] for c in CANDS)
    lowest = min(CANDS, key=lambda c: (rows[c]["mean"], CANDS.index(c)))
    rec = {"lowest_mean": lowest, "lowest_mean_c": best, "g0_mean_c": g0_mean, "g0_minus_lowest_c": g0_mean - best, "threshold_c": TIE}
    if not g0_mean - best > TIE + EPS:
        return "G0", {**rec, "improves_in_development": False, "astar": "G0"}
    tied = [c for c in CANDS if rows[c]["mean"] - best <= TIE + EPS]
    p95 = min(rows[c]["p95"] for c in tied)
    low = [c for c in tied if rows[c]["p95"] == p95]
    pick = min(low, key=lambda c: (rows[c]["fired"], CANDS.index(c)))
    return pick, {**rec, "improves_in_development": True, "tied_within_0.02": tied, "lowest_p95_among_tied": low,
                  "fired": {c: rows[c]["fired"] for c in low}, "astar": pick}


def holm(p):
    """engine18.holm on the hypotheses that exist (1 or 2): levels 0.05, or 0.025 and 0.05."""
    return E.holm(p, alpha=ALPHA)


def power(d, par, n_new, t_new, idx, alphas):
    """MDE of a per-parent series for a new universe of n_new stocks over t_new days (research 17 / 18 model): x = symbol-day
    mean, D = day mean, W = mean within-day variance of x, V = Var(D), c = V - W / n_dev, V_new = W / n_new + c,
    VIF = bootstrap variance of mean D over Var(D) / T, SE_new = sqrt(V_new x VIF / t_new); one-sided, power 80%."""
    from scipy.stats import norm
    sdm = pd.DataFrame({"date": par.date.to_numpy(), "sym": par.sym.to_numpy(), "d": d}).groupby(["date", "sym"]).d.mean()
    day = sdm.groupby(level=0).mean()
    Wv = float(sdm.groupby(level=0).var(ddof=1).mean())
    n_dev = float(sdm.groupby(level=0).size().mean())
    V = float(day.var(ddof=1))
    c = V - Wv / n_dev
    v_new = Wv / n_new + c
    v = day.to_numpy()
    vif = float(v[idx].mean(axis=1).var(ddof=1) / (v.var(ddof=1) / len(v)))
    se = math.sqrt(max(v_new, 0.0) * vif / t_new)
    return {"days_dev": int(len(v)), "stocks_per_day_dev": n_dev, "W_within_day_var": Wv, "V_dev_daymean_var": V, "c_common_var": c,
            "n_new": n_new, "days_new": t_new, "V_new_daymean_var": v_new, "VIF": vif, "se_new_c": se, "effect_dev_c": float(d.mean()),
            "mde_c": {f"{a:.4f}": float((norm.ppf(1 - a) + norm.ppf(POWER)) * se) for a in alphas},
            "underpowered": {f"{a:.4f}": bool((norm.ppf(1 - a) + norm.ppf(POWER)) * se > abs(float(d.mean()))) for a in alphas}}


def prediction_table(sd, basis):
    """{group: {horizon: mean bp, 95% interval, points, points left out, days}} of one basis (day-block bootstrap, seed 22)."""
    P = sd["pred"][basis]
    codes, days = day_codes(sd["date"])
    out = {}
    for gi, g in enumerate(GROUPS):
        out[g] = {}
        for hi, h in enumerate(HORIZONS):
            s = np.bincount(codes, weights=P[:, gi, hi, 0], minlength=len(days))
            n = np.bincount(codes, weights=P[:, gi, hi, 1], minlength=len(days))
            keep = n > 0
            r = {"points": int(n.sum()), "left_out": int(P[:, gi, hi, 2].sum()), "days": int(keep.sum())}
            if keep.sum() >= 2:
                idx = E.boot_index(int(keep.sum()), seed=SEED)
                dist = s[keep][idx].sum(axis=1) / n[keep][idx].sum(axis=1)
                r.update({"mean_bp": float(s.sum() / n.sum()), "ci_lo": float(np.percentile(dist, 2.5)), "ci_hi": float(np.percentile(dist, 97.5))})
            elif keep.sum() == 1:
                r.update({"mean_bp": float(s.sum() / n.sum()), "ci_lo": None, "ci_hi": None})
            out[g][str(h)] = r
    return out


def signal_counts(sd, basis):
    """Decision grid points of the basis: all, with a signal, with the sigma floor applied (among those with a signal)."""
    S = sd["sig"][basis].sum(axis=0)
    return {"grid_points": int(S[0]), "with_signal": int(S[1]), "no_signal_share": float(1 - S[1] / S[0]) if S[0] else None,
            "floor_applied": int(S[2]), "floor_share_of_signal": float(S[2] / S[1]) if S[1] else None}


def daily_rows(par, OUT, names, variants=None):
    """Daily file: one row per date, variant and run with the number of parents and the sum of their IS (cents per share)."""
    rows = []
    codes, days = day_codes(par.date.to_numpy())
    for lab, lr, dep in (variants or E.VARIANTS):
        IS, _ = is_table(par, OUT, names, lr, dep)
        n = np.bincount(codes, minlength=len(days))
        for m, x in IS.items():
            s = np.bincount(codes, weights=x, minlength=len(days))
            rows += [(d, lab, m, int(k), float(v)) for d, k, v in zip(days, n, s)]
    return pd.DataFrame(rows, columns=["date", "variant", "run", "n_parents", "is_sum_c"])
