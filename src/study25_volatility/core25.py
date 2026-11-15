#!/usr/bin/env python3
"""Research 25 simulation and statistics shared by dev25.py, run_test25.py, regress25.py and test25.py (SPEC25 2-6,
SPEC25_ANSWERS, DEVIATIONS25.md 1). The engines and the model of research 18, 22 and 23 are imported and not changed:
engine18.run with research 18's run names and an array P, loader22 for the bars (sizes / 100 from 2025-11-03), feat23 and
model23 for the nine inputs and the probability of research 23's frozen model (model23.json, never refitted).
  runs        S0, S2i, R0 (= E*); the 14 candidates of SPEC25 4 (signal25.CANDS); "ctrl:<candidate>" the control with the
              volatility / volume condition turned around (SPEC25 5; not a candidate); "k0:..." the regression runs of
              ANSWERS A3 (no volatility condition; not candidates)
  C3          per parent: S2i when the arrival spread is at most 4 cents, else E* (engine18.branch_c3)
  IS          engine18.costs, cents per share; L3D1 (primary) and L0D0
  dollars     saving against R0 of a run: sum over the parents of (IS_R0 - IS_run) x 1,000 shares / 100 (ANSWERS A7)
  statistics  circular day-block bootstrap (21 days, 10,000 draws, seed 25): mean and two-sided 95% interval, one-sided
              p = share of bootstrap means >= 0, 95th percentile differences
  A*          SPEC25 4 with QUESTIONS 19 (select_astar)
  diagnostics SPEC25 6 with QUESTIONS 26 (never used for a rule)
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
sys.path.insert(0, os.path.join(PROJ, "study23_combined_forecast"))
import core23 as C23  # noqa: E402  (research 23, frozen; checks the files of research 15-22 at import)
import signal25 as S  # noqa: E402

E, F, M, L22 = C23.E, C23.F, C23.M, C23.L22
# files of research 23 used on the run path, never edited: sha256 of the file at its tag
FROZEN = {"study23_combined_forecast/core23.py": ("code-freeze-23", "09ce9662bb1ed2222151393974b25fe6a76d2cb47b3a82526675d4c6cc073669"),
          "study23_combined_forecast/feat23.py": ("code-freeze-23", "e2667f7356225d5c89c80e3085a8bdaf1a3e87a3ef5672e6ffe36ea462dd055d"),
          "study23_combined_forecast/model23.py": ("code-freeze-23", "9bab7a3d95e293036018246844e9dd91191d014ba15995f95a53247c35eb3fb4"),
          "study23_combined_forecast/model23.json": ("code-freeze-23", "61b6610038345f60e088df7260a84a380c2aa5665bdd43d03a17b0be86c5bc04")}
MODEL23 = os.path.join(PROJ, "study23_combined_forecast", "model23.json")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def frozen_differences():
    """Paths whose file differs from the sha256 of its tag (research 23's own list and this one)."""
    return C23.frozen_differences() + [p for p, (_, h) in FROZEN.items() if sha(os.path.join(PROJ, p)) != h]


if frozen_differences():
    raise ImportError(f"files of earlier studies differ from their tags: {frozen_differences()}")

BASE = ("S0", "S2i", "R0")
CANDS, SPECS, K0 = S.CANDS, S.SPECS, tuple(S.K0)
CTRL = "ctrl:"
SEED, TIE, EPS = 25, 0.02, 1e-12
CUT = "20251103"
ALPHA, POWER = 0.05, 0.8
USD_PER_CENT = E.QTY / 100.0                     # dollars per parent order for 1 cent per share
QBINS = (0.0, 0.25, 0.5, 1.0, 2.0, float("inf"))  # QUESTIONS 26
PCTS = (1, 5, 25, 50, 75, 95, 99)
jdefault, day_codes, mean_ci, one_sided, pair, visited, power = (C23.jdefault, C23.day_codes, C23.mean_ci, C23.one_sided, C23.pair,
                                                                 C23.visited, C23.power)     # these take the bootstrap index as an argument
is_table, delay_table, daily_rows, dev_tasks, with_adv, auc_point = (C23.is_table, C23.delay_table, C23.daily_rows, C23.dev_tasks,
                                                                     C23.with_adv, C23.auc_point)


def model23():
    """Research 23's frozen model, read only when the file equals its tag and its own sha256 file."""
    h = sha(MODEL23)
    if h != FROZEN["study23_combined_forecast/model23.json"][1] or open(os.path.splitext(MODEL23)[0] + ".sha256").read().split()[0] != h:
        raise SystemExit("study23_combined_forecast/model23.json differs from its frozen sha256")
    return json.load(open(MODEL23))


def cand_of(run):
    """The candidate behind a run name (itself, its control or a regression run), or None for a base run."""
    if run.startswith(CTRL):
        run = run[len(CTRL):]
        if run not in CANDS:
            raise KeyError(run)
        return run
    if run in S.K0:
        return S.K0[run][0]
    if run in CANDS:
        return run
    if run in BASE:
        return None
    raise KeyError(run)


def engine_name(run):
    """engine18's run name of a run."""
    if run in S.K0:
        return S.K0[run][1]
    c = cand_of(run)
    return run if c is None else SPECS[c]["engine"]


def needs_model(run):
    c = cand_of(run)
    return c is not None and SPECS[c]["family"] != "A"


# ---------------------------------------------------------------- one symbol-day
def run_parents(b, ref, close_bar, dirs, starts, adv, runs, delay_runs=(), model=None):
    """The inputs, the probability of the model handed in, the signals and every run under L0 and L3 (delay_runs with
    delay 1 after them). Pe: what engine18 got for every run that has a P."""
    raw = F.raw(b, ref, close_bar, dirs, starts, adv)
    p23 = M.predict(model, raw["X"], raw["valid"]) if model is not None else None
    sig = S.signals(b, dirs, starts, raw["vp"], adv)
    names = list(runs) + list(delay_runs)
    Pe = {}
    for name in names:
        c = cand_of(name)
        if c is None or name in Pe:
            continue
        if needs_model(name) and p23 is None:
            raise ValueError(f"{name}: no model for its probability")
        Pe[name] = S.engine_p(c, sig, p23, flip=name.startswith(CTRL), off=name in S.K0)
    outs = np.zeros((len(E.LRULES), len(names), len(dirs), E.NCOL), dtype=np.int64)
    for li, lr in enumerate(E.LRULES):
        for ri, name in enumerate(names):
            outs[li, ri] = E.run(b, engine_name(name), lr, dirs, starts, Pe.get(name), 1 if ri >= len(runs) else 0)
    return {"outs": outs, "raw": raw, "P23": p23, "sig": sig, "Pe": Pe}


def signal_counts(raw, sig):
    """Counts of decision points (and of parents for vol_pre) by what exists there; no value is kept."""
    valid, full, q = raw["valid"], sig["full"], sig["q"]
    return {"window_not_full": int((~full).sum()), "spread_zero_in_full_window": int((full & (sig["spread_c"] <= 0)).sum()),
            "q_defined": int((~np.isnan(q)).sum()), "g_defined": int((~np.isnan(sig["g"])).sum()), "v_defined": int((~np.isnan(sig["v"])).sum()),
            "quote_valid": int(valid.sum()), "quote_valid_q_undefined": int((valid & np.isnan(q)).sum()),
            "quote_valid_window_not_full": int((valid & ~full).sum()), "quote_valid_spread_zero": int((valid & full & (sig["spread_c"] <= 0)).sum()),
            "parents_vol_pre_undefined_or_zero": int((~sig["vp_ok"]).sum())}


def symday(t, spec):
    """One symbol-day of a plan: the 20 parents of the date (engine18.parents) with a valid start quote, then run_parents.
    spec: runs, delay_runs, model, grid (True: the signals, p and the 60-second move of every decision point are returned)."""
    b, ref = L22.load(t)
    dirs, starts = E.parents(t["date"], t["close_bar"])
    ok = E.quote_ok(b)[starts]
    dirs, starts = dirs[ok], starts[ok]
    r = run_parents(b, ref, t["close_bar"], dirs, starts, t["adv"], spec["runs"], spec["delay_runs"], spec["model"])
    raw, sig = r["raw"], r["sig"]
    out = {"sym": t["sym"], "date": t["date"], "dir": dirs, "start": starts, "mid2": (b["bid"] + b["ask"])[starts],
           "spread": (b["ask"] - b["bid"])[starts], "vol_pre": raw["vp"], "n_dropped": int((~ok).sum()), "outs": r["outs"],
           "counts": {**raw["counts"], **signal_counts(raw, sig)}, "has_ref": t["ref_path"] is not None}
    if spec["grid"]:
        out.update({"y": raw["y"], "valid": raw["valid"], "g": sig["g"], "q": sig["q"], "v": sig["v"], "move": S.abs_move60(b, starts)})
        if r["P23"] is not None:
            out["P23"] = r["P23"]
    return out


def _work(a):
    return symday(*a)


def simulate(tasks, runs=(), delay_runs=(), workers=4, model=None, grid=False):
    """(par, OUT, G, info). par: one row per parent (date, sym, dir, start, mid2, spread, vol_pre). OUT [lrule, run, parent,
    col]. G (grid True): y, valid, g, q, v, move [parent, NG] and P23 when a model is handed in."""
    if not tasks:
        raise ValueError("no symbol-day in the plan")
    spec = {"runs": tuple(runs), "delay_runs": tuple(delay_runs), "model": model, "grid": bool(grid)}
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
    G = {k: np.concatenate([r[k] for r in res]) for k in ("y", "valid", "g", "q", "v", "move", "P23") if k in res[0]}
    info = {"symbol_days": len(tasks), "parents": int(len(par)), "dropped_parents": int(sum(r["n_dropped"] for r in res)),
            "symbol_days_without_parents": int(sum(k == 0 for k in n)), "symbol_days_without_ref": int(sum(not r["has_ref"] for r in res)),
            "grid_counts": {k: int(sum(r["counts"][k] for r in res)) for k in res[0]["counts"]},
            "runs": list(runs), "delay_runs": list(delay_runs), "workers": workers}
    return par, OUT, G, info


# ---------------------------------------------------------------- IS and statistics
def boot(n_days):
    idx = E.boot_index(n_days, seed=SEED)
    return idx, E.day_counts(idx, n_days)


def acts(run):
    """True for a run whose rule can act differently from E* (a candidate, its control or a regression run)."""
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
    """One table row: mean IS and interval, p95, the paired difference to `ref` (same parents) in cents per share and as a
    dollar saving, firing, by year."""
    x, o = IS[run], outs[run]
    d = x - IS[ref]
    pe, pdist = E.p95_diff_boot(x, IS[ref], codes, W)
    return {"run": run, "n_parents": int(len(x)), **{f"is_{k}": v for k, v in mean_ci(x, codes, n_days, idx).items()},
            "p95": float(np.percentile(x, 95)), **{f"minus_{ref}_{k}": v for k, v in one_sided(d, codes, n_days, idx).items()},
            f"p95_minus_{ref}": pe, f"p95_minus_{ref}_ci_lo": float(np.percentile(pdist, 2.5)), f"p95_minus_{ref}_ci_hi": float(np.percentile(pdist, 97.5)),
            f"usd_saved_vs_{ref}": float(-d.sum() * USD_PER_CENT),
            "firing": firing(o, run), "limit_share_frac": float((o[:, 0] / E.QTY).mean()), "deadline_parent_frac": float((o[:, 6] > 0).mean()),
            "mean_done_s": float(o[:, 5].mean()), "market_nonpositive_price": int(o[:, 11].sum()),
            "by_year": {str(y): {"is_mean": float(x[years == y].mean()), f"minus_{ref}": float(d[years == y].mean()),
                                 f"usd_saved_vs_{ref}": float(-d[years == y].sum() * USD_PER_CENT), "n_parents": int((years == y).sum())}
                        for y in sorted(set(years))}}


def subset_stat(d, dates, mask):
    """one_sided on the parents of `mask`, with their own day codes and bootstrap index (seed 25)."""
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


def control_table(par, OUT, names, astar, st):
    """SPEC25 5 (no verdict, not a candidate): the control of A* (its volatility / volume condition turned around) under
    both variants: its mean IS and firing, control - E* and A* - control over the same parents."""
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
    """SPEC25 4 with QUESTIONS 19. rows: {candidate: {"mean", "p95", "fired"}} under L3D1 in the validation period.
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


# ---------------------------------------------------------------- diagnostics (SPEC25 6; never used for a rule)
def diagnostics(G):
    """Percentiles of g, q, v over the decision points where the signal exists; the share of decision points where each
    candidate's condition holds; by q bin: the mean |mid(u + 60) - mid(u)| in cents and the AUC of research 23's model on
    the scored points (decision quote valid, y defined) of the bin."""
    n_grid = int(G["y"].size)
    out = {"grid_points": n_grid, "percentiles": {}, "condition_share_of_grid": {}, "q_bins": []}
    for k in ("g", "q", "v"):
        x = G[k][~np.isnan(G[k])]
        out["percentiles"][k] = {"n": int(len(x)), "share_of_grid": float(len(x) / n_grid),
                                 **({f"p{p}": float(v) for p, v in zip(PCTS, np.percentile(x, PCTS))} if len(x) else {})}
    sig = {k: G[k] for k in ("g", "q", "v")}
    p = G.get("P23")
    for c in CANDS:
        s = SPECS[c]
        if s["family"] != "A" and p is None:
            continue
        pe = S.engine_p(c, sig, p)
        phi, plo = E.params(s["engine"])[4:]
        hit = ((pe >= phi) if phi >= 0 else np.zeros(pe.shape, bool)) | ((pe <= plo) if plo >= 0 else np.zeros(pe.shape, bool))
        out["condition_share_of_grid"][c] = float(hit.mean())
    q, mv = G["q"], G["move"]
    keep = G["valid"] & (G["y"] >= 0) & (np.isfinite(p) if p is not None else False)
    edges = list(zip(QBINS[:-1], QBINS[1:])) + [(None, None)]
    for lo, hi in edges:
        m = np.isnan(q) if lo is None else (~np.isnan(q)) & (np.where(np.isnan(q), -1.0, q) >= lo) & (np.where(np.isnan(q), -1.0, q) < hi)
        mm, ks = m & ~np.isnan(mv), m & keep
        out["q_bins"].append({"bin": "no signal" if lo is None else f"[{lo:g}, {hi:g})", "points": int(m.sum()), "share_of_grid": float(m.mean()),
                              "points_with_move": int(mm.sum()), "mean_abs_move_60s_c": float(mv[mm].mean()) if mm.any() else None,
                              "scored_points": int(ks.sum()), "base_rate": float((G["y"][ks] == 1).mean()) if ks.any() else None,
                              "auc": auc_point(p[ks], G["y"][ks].astype(np.int8)) if (p is not None and ks.any()) else None})
    return out
