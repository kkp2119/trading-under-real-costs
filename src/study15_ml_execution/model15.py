"""Research 15 step 3: gates, training rows, models and predictions (Kyle 2026-10-29 A-B; DEVIATIONS15 15-16).
Run with python (lightgbm, numba). Only 2015-2020 rows are read (xy, xy_full, loader15).

Rows: decisions that pass the B0 gates (row: no_quote 0, 2 <= spread <= 12 ticks, which is qbase and min spread 2; day:
G-adv and a valid quote at bar 4) give one buy row and one mirrored sell row (signed inputs negated, bsz <-> asz, side
-1). Targets: v_h = frac x mk_h (0 without a fill), t_h = tk_h, h = 5, 30, 60; rows with a NaN target are dropped for that
target. Ridge: NaN -> 0 plus one missing indicator per input, standardized with the training rows, normal equations in
float64, unpenalized intercept. LightGBM: one sample of at most 6,000,000 training rows (default_rng(15)); sample rows of
the last training year are the early-stopping set (patience 50, L2), the others are fitted; the early-stopped model is
kept. Every fitting function receives training rows only; fold_check() records the year sets."""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features15  # noqa: E402
import loader15 as L  # noqa: E402
import session15  # noqa: E402
import xy15 as X  # noqa: E402

DATA = os.path.join(os.environ.get("DATA_DIR", "data"), "research15_data")
XY, XY_FULL = os.path.join(DATA, "xy"), os.path.join(DATA, "xy_full")
MODEL_DIR = os.path.join(DATA, "models15")
META = os.path.join(HERE, "model_meta")
GATES = os.path.join(META, "gates15.tsv")
SIGNED = ("imb", "dmid_1", "dmid_5", "dmid_30", "dmid_60", "dmid_300", "ofi_5", "ofi_30", "ofi_60", "ofi_300", "dmid_day")
FEATS = X.FEATURES + ("side",)
TARGETS = ("v5", "v30", "v60", "t5", "t30", "t60")
RIDGE = {"ridge_a10": 10.0, "ridge_a1000": 1000.0}
LGBM = {"lgb_l31_m1000": (31, 1000), "lgb_l31_m10000": (31, 10000), "lgb_l127_m1000": (127, 1000),
        "lgb_l127_m10000": (127, 10000)}
MODELS = tuple(RIDGE) + tuple(LGBM)
LGB_MAX_ROWS, LGB_SEED, LGB_ROUNDS, LGB_PATIENCE, LGB_THREADS = 6_000_000, 15, 1000, 50, 8
FOLDS = {"A": ((2015, 2016, 2017, 2018), 2019), "B": ((2015, 2016, 2017, 2018, 2019), 2020),
         "A16": ((2016, 2017, 2018), 2019), "B16": ((2016, 2017, 2018, 2019), 2020),
         "final": ((2015, 2016, 2017, 2018, 2019, 2020), None)}
C = {c: i for i, c in enumerate(X.COLUMNS)}
FI = [C[f] for f in X.FEATURES]
SIGN_MASK = np.array([-1.0 if f in SIGNED else 1.0 for f in X.FEATURES], dtype=np.float32)
SWAP = [X.FEATURES.index("asz") if f == "bsz" else X.FEATURES.index("bsz") if f == "asz" else i
        for i, f in enumerate(X.FEATURES)]


def lgb():
    import lightgbm
    return lightgbm


# ---------------------------------------------------------------- gates
def gate_table():
    """One row per 2015-2020 symbol-day with a bar file: dv (1e-4 $), no_data, G-adv gate, valid quote at bar 4 (window 2)
    and at bar 899 (window 0, research 12 B0). Writes model_meta/gates15.tsv."""
    cal = session15.Calendar()
    rows, avail = [], set()
    for y in L.YEARS:
        cb = L.close_bars(y)
        for (s, d), r in sorted(L.manifest(y).items()):
            b = L.load(y, r)
            ds = features15.day_stats(b, cb[d])
            ok = (b["bid"] > 0) & (b["ask"] > 0) & (b["ask"] >= b["bid"])
            rows.append({"sym": s, "date": d, "year": y, "close_bar": cb[d], "dv": ds["dv"], "no_data": ds["no_data"],
                         "d0_ok_w2": bool(ok[4]), "d0_ok_w0": bool(ok[899])})
            avail.add((s, d))
    t = pd.DataFrame(rows)
    dv = {(r.sym, r.date): int(r.dv) for r in t.itertuples()}
    zero = {(r.sym, r.date) for r in t.itertuples() if r.no_data}
    t["gate"] = [session15.gate(cal, r.sym, r.date, dv, zero, avail) for r in t.itertuples()]
    os.makedirs(META, exist_ok=True)
    t.to_csv(GATES, sep="\t", index=False)
    return t


def read_gates():
    return pd.read_csv(GATES, sep="\t", dtype={"date": str, "sym": str})


# ---------------------------------------------------------------- rows
def row_gate(M):
    sp = M[:, C["spread_tk"]]
    return (M[:, C["no_quote"]] == 0) & (sp >= 2) & (sp <= 12)


def sides(M):
    """(X buy, X sell mirrored), float32 n x len(FEATS)."""
    F = M[:, FI]
    xb = np.column_stack([F, np.ones(len(M), np.float32)])
    xs = np.column_stack([(F * SIGN_MASK)[:, SWAP], -np.ones(len(M), np.float32)])
    return xb.astype(np.float32), xs.astype(np.float32)


def targets(M, p):
    """float32 n x 6 for side prefix p ('b' / 's'): v_h = frac x mk_h (0 without a fill), t_h = tk_h."""
    out = []
    fill, frac = M[:, C[f"{p}_fill"]], M[:, C[f"{p}_frac"]]
    for h in X.HS:
        out.append(np.where(fill == 0, 0.0, frac * M[:, C[f"{p}_mk_{h}"]]))
    for h in X.HS:
        out.append(M[:, C[f"{p}_tk_{h}"]])
    return np.column_stack(out).astype(np.float32)


def sampled_2020(M, day, close_bar):
    """Rows of a full-decision file kept by the 30 s rule of xy15 (offset default_rng(15 + date)) (DEVIATIONS15 16, E)."""
    d = X.decisions(close_bar)
    k = X.offset(day)
    return np.isin(M[:, 0].astype(np.int64), d[k::X.SAMPLE_EVERY])


def day_files(year, full=False):
    root = XY_FULL if full else XY
    m = pd.read_csv(os.path.join(root, "MANIFEST.tsv"), sep="\t", dtype={"date": str, "sym": str})
    m = m[m.date.str[:4] == str(year)]
    return m, root


def assemble(years, log=print):
    """Training rows of `years` (sampled files; 2020 thinned by the 30 s rule). Returns (X float32 N x 32, Y float32 N x 6,
    year int16 N, info)."""
    g = read_gates().set_index(["sym", "date"])
    xs, ys, yr = [], [], []
    info = {}
    for y in years:
        m, root = day_files(y)
        n_dec = n_row = n_day_out = 0
        for r in m.itertuples():
            gr = g.loc[(r.sym, r.date)]
            with np.load(os.path.join(root, r.file), allow_pickle=False) as z:
                M = z["xy"]
            if y == 2020:
                M = M[sampled_2020(M, r.date, int(r.close_bar))]
            n_dec += len(M)
            if not (gr.gate and gr.d0_ok_w2):
                n_day_out += len(M)
                continue
            M = M[row_gate(M)]
            n_row += len(M)
            xb, xsl = sides(M)
            xs += [xb, xsl]
            ys += [targets(M, "b"), targets(M, "s")]
            yr.append(np.full(2 * len(M), y, np.int16))
        info[y] = {"decisions": n_dec, "out_by_day_gate": n_day_out, "out_by_row_gate": n_dec - n_day_out - n_row,
                   "kept_decisions": n_row, "rows_both_sides": 2 * n_row}
        log(f"assemble {y}: {info[y]}")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(yr), info


# ---------------------------------------------------------------- ridge
def ridge_design(Xf):
    nan = np.isnan(Xf[:, :len(X.FEATURES)])
    return np.hstack([np.where(np.isnan(Xf), 0.0, Xf), nan]).astype(np.float64)


def ridge_fit(Xf, y, alphas, chunk=1_000_000):
    """{alpha: model dict}. Standardization and normal equations use the rows given (training rows only)."""
    ok = np.flatnonzero(~np.isnan(y))
    n = len(ok)
    k = len(FEATS) + len(X.FEATURES)
    s1, s2 = np.zeros(k), np.zeros(k)
    for i in range(0, n, chunk):
        D = ridge_design(Xf[ok[i:i + chunk]])
        s1 += D.sum(0)
        s2 += (D * D).sum(0)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0))
    sd[sd == 0] = 1.0
    ybar = float(np.mean(y[ok], dtype=np.float64))
    A, bvec = np.zeros((k, k)), np.zeros(k)
    for i in range(0, n, chunk):
        Z = (ridge_design(Xf[ok[i:i + chunk]]) - mu) / sd
        A += Z.T @ Z
        bvec += Z.T @ (y[ok[i:i + chunk]].astype(np.float64) - ybar)
    out = {}
    for a in alphas:
        w = np.linalg.solve(A + a * np.eye(k), bvec)
        out[a] = {"kind": "ridge", "alpha": a, "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "b": ybar, "n": int(n)}
    return out


def ridge_predict(m, Xf):
    D = ridge_design(Xf)
    return ((D - np.asarray(m["mu"])) / np.asarray(m["sd"])) @ np.asarray(m["w"]) + m["b"]


# ---------------------------------------------------------------- LightGBM
def lgb_sample(n, seed=LGB_SEED, cap=LGB_MAX_ROWS):
    """Sorted indices of at most `cap` training rows (default_rng(seed))."""
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def lgb_fit(Xs, ys, yr, es_year, leaves, min_leaf, threads=LGB_THREADS, log=print):
    """Xs, ys, yr: the sampled training rows. Fit on yr != es_year, early stopping on yr == es_year."""
    lg = lgb()
    ok = ~np.isnan(ys)
    fit, es = ok & (yr != es_year), ok & (yr == es_year)
    dpar = {"feature_pre_filter": False, "verbose": -1}
    dfit = lg.Dataset(Xs[fit], ys[fit], feature_name=list(FEATS), params=dpar, free_raw_data=False)
    des = lg.Dataset(Xs[es], ys[es], reference=dfit, params=dpar, free_raw_data=False)
    par = {"objective": "regression", "learning_rate": 0.05, "num_leaves": leaves, "min_data_in_leaf": min_leaf,
           "feature_fraction": 0.8, "num_threads": threads, "seed": LGB_SEED, "deterministic": True,
           "force_row_wise": True, "verbose": -1, "metric": "l2"}
    t0 = time.time()
    bst = lg.train(par, dfit, num_boost_round=LGB_ROUNDS, valid_sets=[des],
                   callbacks=[lg.early_stopping(LGB_PATIENCE, verbose=False)])
    log(f"   lgb leaves {leaves} min {min_leaf}: fit rows {int(fit.sum())}, es rows {int(es.sum())} ({es_year}), "
        f"best iteration {bst.best_iteration}, {time.time() - t0:.0f}s")
    return bst


# ---------------------------------------------------------------- fold training
def fold_check(train_years, val_year, es_year):
    """Leak rules (Kyle F): validation year outside the training years, early-stopping year = last training year."""
    assert val_year is None or val_year not in train_years and val_year > max(train_years)
    assert es_year == max(train_years) and es_year in train_years
    return {"train_years": list(train_years), "val_year": val_year, "es_year": es_year}


def train_fold(fold, models=MODELS, targets_=TARGETS, log=print):
    """Fit `models` x `targets_` on the fold's training years; write MODEL_DIR/<fold>/ files and fold.json."""
    train_years, val_year = FOLDS[fold]
    chk = fold_check(train_years, val_year, max(train_years))
    out = os.path.join(MODEL_DIR, fold)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    Xf, Y, yr, info = assemble(train_years, log)
    assert set(np.unique(yr)) <= set(train_years)
    log(f"fold {fold}: {len(Xf)} training rows, assembled in {time.time() - t0:.0f}s")
    rec = {**chk, "rows": info, "n_rows": int(len(Xf)), "files": {}}
    ti = {t: i for i, t in enumerate(TARGETS)}
    ridge_names = [m for m in models if m in RIDGE]
    for t in targets_:
        if ridge_names:
            fits = ridge_fit(Xf, Y[:, ti[t]], [RIDGE[m] for m in ridge_names])
            for m in ridge_names:
                p = os.path.join(out, f"{m}_{t}.json")
                open(p, "w").write(json.dumps(fits[RIDGE[m]]))
                rec["files"][f"{m}_{t}"] = sha(p)
    lgb_names = [m for m in models if m in LGBM]
    if lgb_names:
        idx = lgb_sample(len(Xf))
        rec["lgb_sample"] = {"seed": LGB_SEED, "rows": int(len(idx)), "of": int(len(Xf)),
                             "years": {int(y): int((yr[idx] == y).sum()) for y in train_years}}
        Xs, Ys, yrs = Xf[idx], Y[idx], yr[idx]
        for t in targets_:
            for m in lgb_names:
                bst = lgb_fit(Xs, Ys[:, ti[t]], yrs, chk["es_year"], *LGBM[m], log=log)
                p = os.path.join(out, f"{m}_{t}.txt")
                bst.save_model(p, num_iteration=bst.best_iteration)
                rec["files"][f"{m}_{t}"] = sha(p)
                rec.setdefault("best_iteration", {})[f"{m}_{t}"] = int(bst.best_iteration)
    rec["seconds"] = round(time.time() - t0)
    p = os.path.join(out, "fold.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    for k in ("files", "best_iteration"):
        rec[k] = {**old.get(k, {}), **rec.get(k, {})}
    open(p, "w").write(json.dumps(rec, indent=1))
    return rec


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_models(fold, names):
    """{name: model} for names '<model>_<target>' of a fold (LightGBM Booster or ridge dict)."""
    lg = None
    out = {}
    for n in names:
        p = os.path.join(MODEL_DIR, fold, n)
        if os.path.exists(p + ".json"):
            out[n] = json.load(open(p + ".json"))
        else:
            lg = lg or lgb()
            out[n] = lg.Booster(model_file=p + ".txt")
    return out


def predict(m, Xf, threads=1):
    if isinstance(m, dict):
        return ridge_predict(m, Xf)
    return m.predict(Xf, num_threads=threads)
