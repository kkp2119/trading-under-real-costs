#!/usr/bin/env python3
"""Research 23 model: research 18's training order (study18_forecast/dev18.py fit) moved to a new file (SPEC23_ANSWERS A5); the
files of research 18 are not edited and its model file is never written.
  fit      on the training parents: vol_pre fill = median of the parents' defined vol_pre (research 18); points kept =
           decision quote valid and y defined; intensity fill = median of the kept points' defined intensity; the product
           is formed after the fill; mean and sd of the kept points; LogisticRegression(penalty l2, C 1.0, lbfgs,
           max_iter 1000) once on the standardised inputs. cols selects the inputs (all nine by default; the first four
           reproduce research 18's model; one left out for the reference fits).
  predict  p = 1 / (1 + exp(-(intercept + sum coef_j (x_j - mean_j) / sd_j))); nan where the decision quote is not valid.
A column without variation gets sd 1 (it then contributes nothing); this cannot happen with real data and is recorded."""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import feat23 as F  # noqa: E402

ESTIMATOR = "LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)"


def fit(X, y, valid, vp, cols=None):
    """X [n, NG, 9] raw inputs of the training parents, y [n, NG], valid [n, NG], vp [n]. Returns the model dict."""
    from sklearn import __version__ as skv
    from sklearn.linear_model import LogisticRegression
    cols = list(range(len(F.FEATS))) if cols is None else list(cols)
    vpd = vp[~np.isnan(vp)]
    fill_vp = float(np.median(vpd)) if len(vpd) else 0.0
    keep = valid & (y >= 0)
    Xk = X[keep]
    it = Xk[:, F.I_INT]
    itd = it[~np.isnan(it)]
    fills = {"vol_pre": fill_vp, "intensity": float(np.median(itd)) if len(itd) else 1.0}
    Xk = F.design(Xk, fills)
    if cols != list(range(len(F.FEATS))):
        Xk = np.ascontiguousarray(Xk[:, cols])
    mean, sd0 = Xk.mean(axis=0), Xk.std(axis=0)
    sd = np.where(sd0 > 0, sd0, 1.0)
    Z = (Xk - mean) / sd
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000).fit(Z, y[keep])
    model = {"features": [F.FEATS[c] for c in cols], "cols": cols, "mean": mean.tolist(), "sd": sd.tolist(), "coef": clf.coef_[0].tolist(),
             "intercept": float(clf.intercept_[0]), "fills": fills, "vol_pre_fill": fill_vp, "columns_without_variation": [F.FEATS[c] for c, s in zip(cols, sd0) if not s > 0],
             "n_train_points": int(keep.sum()), "n_train_parents": int(len(vp)), "base_rate_train": float(y[keep].mean()),
             "sklearn": skv, "estimator": ESTIMATOR, "n_iter": int(clf.n_iter_[0]),
             "p_formula": "1 / (1 + exp(-(intercept + sum coef_j (x_j - mean_j) / sd_j)))"}
    rng = np.random.default_rng(23)
    pick = rng.choice(len(Z), min(200_000, len(Z)), replace=False)
    pz = 1.0 / (1.0 + np.exp(-(model["intercept"] + Z[pick] @ np.asarray(model["coef"]))))
    model["check_vs_predict_proba_max_abs"] = float(np.abs(pz - clf.predict_proba(Z[pick])[:, 1]).max())
    return model


def predict(model, X, valid):
    """p [n, NG] of the raw inputs X [n, NG, 9]; nan where the decision quote is not valid."""
    Z = F.design(X, model["fills"])
    if list(model["cols"]) != list(range(len(F.FEATS))):
        Z = np.ascontiguousarray(Z[..., list(model["cols"])])        # the memory order of engine18.design's array
    z = (Z - np.asarray(model["mean"])) / np.asarray(model["sd"])
    p = 1.0 / (1.0 + np.exp(-(model["intercept"] + z @ np.asarray(model["coef"]))))
    return np.where(valid, p, np.nan)


def from18(m18):
    """Research 18's model (four inputs, model18.json) in this file's form, so that predict scores it on the stored inputs."""
    return {**m18, "cols": [0, 1, 2, 3], "fills": {"vol_pre": m18["vol_pre_fill"], "intensity": 1.0}}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def save(model, path):
    """Writes the model and its .sha256 next to it; refuses to overwrite."""
    side = os.path.splitext(path)[0] + ".sha256"
    if os.path.exists(path) or os.path.exists(side):
        raise SystemExit(f"{os.path.basename(path)} exists: the model is fitted once (nothing is overwritten)")
    with open(path, "w") as fh:
        json.dump(model, fh, indent=1)
    h = sha(path)
    with open(side, "w") as fh:
        fh.write(f"{h}  {os.path.basename(path)}\n")
    return h


def load(path):
    """The model of a file whose sha256 equals the .sha256 file next to it."""
    side = os.path.splitext(path)[0] + ".sha256"
    if open(side).read().split()[0] != sha(path):
        raise SystemExit(f"{os.path.basename(path)} differs from its sha256 file")
    return json.load(open(path))
