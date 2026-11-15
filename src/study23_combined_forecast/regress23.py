#!/usr/bin/env python3
"""Research 23 regression test on the development data (step-2 prompt 1; every check must hold before the model is fitted).
The nine-input model is not fitted here and no candidate is run with it.
  training years 2015-2018 (inputs and y only, no order simulated)
    four inputs   research 18's training order in the new code with research 18's four inputs reproduces the frozen
                  coefficients, means, standard deviations, intercept and fill of study18_forecast/model18.json
    stored        the parents, y, the valid flags and inputs 1-4 equal research 18's stored development output
  validation years 2019-2020 (S0, S2i, R0 and the 19 rules with research 18's frozen model)
    AUC           the frozen model scored by the new code has research 18's recorded validation AUC (and hit rate, Brier
                  score, Brier skill score, scored points)
    means         R0, S0, S2i and C3 equal research 18's validation values at 3 decimals; the 19 research 18 candidates
                  equal research 18's rows (dev18_rows.csv, 6 decimals)
    stored        parents, y, valid flags, probabilities and per-parent outputs equal research 18's stored validation output
The stored output of research 18 is its development output ($DATA_DIR/research18_data/dev), not a sealed or test folder;
when a file of it is missing the comparison is recorded as not made and the test fails. Writes results/regress23.json (an
earlier file is moved to results/superseded/); exits 1 when a check fails.
  python regress23.py [--workers 4]"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import core23 as C  # noqa: E402
import dev23 as DV  # noqa: E402

E, F, M = C.E, C.F, C.M
DEV18 = os.path.join(os.environ.get("DATA_DIR", "data"), "research18_data/dev")
SUMMARY18 = os.path.join(PROJ, "study18_forecast", "results", "dev18_summary.json")
PCOLS = ("date", "sym", "dir", "start", "mid2", "spread")


def keep_old(path):
    if os.path.exists(path):
        d = os.path.join(os.path.dirname(path), "superseded")
        os.makedirs(d, exist_ok=True)
        stem, ext = os.path.splitext(os.path.basename(path))
        q = os.path.join(d, f"{stem}_{dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y%m%d_%H%M%S')}{ext}")
        if os.path.exists(q):
            raise RuntimeError(f"{q} exists")
        os.rename(path, q)


def same_nan(a, b):
    return bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True))


def parents_equal(par, p18):
    return bool(len(par) == len(p18) and all(np.array_equal(par[c].to_numpy(), p18[c].to_numpy()) for c in PCOLS)
                and same_nan(par.vol_pre.to_numpy(), p18.vol_pre.to_numpy()))


def main(workers):
    t0 = time.time()
    m18 = C.model18()
    have = all(os.path.exists(os.path.join(DEV18, f)) for f in ("dev18_parents.csv.gz", "dev18_pass1.npz", "dev18_valid.npz"))
    p18 = pd.read_csv(os.path.join(DEV18, "dev18_parents.csv.gz"), dtype={"date": str, "sym": str}, float_precision="round_trip") if have else None
    yr18 = p18.date.str[:4].astype(int).to_numpy() if have else None
    chk = {"stored_output_of_research18_found": bool(have)}

    # ---- training years: inputs and y only
    tasks, pinfo, ainfo = DV.plan(DV.TRAIN, workers)
    par, _, G, info = C.simulate(tasks, workers=workers, grid=True)
    chk["four_input_fit"] = DV.regression_fit(G["X"], G["y"], G["valid"], par.vol_pre.to_numpy())
    if have:
        tr = np.isin(yr18, DV.TRAIN)
        with np.load(os.path.join(DEV18, "dev18_pass1.npz"), allow_pickle=False) as z:
            chk["train_stored"] = {"parents_equal": parents_equal(par, p18[tr]), "y_equal": bool(np.array_equal(G["y"], z["y"][tr])),
                                   "valid_equal": bool(np.array_equal(G["valid"], z["valid"][tr])),
                                   "inputs_1_4_equal_float32": same_nan(G["X"][..., :4].astype(np.float32), z["X"][tr])}
    sim_t = {**info, **pinfo, "adv": ainfo}
    del G, par

    # ---- validation years: research 18's frozen model only
    tasks, pinfo, ainfo = DV.plan(DV.VALID, workers)
    names = list(C.BASE + DV.REF18)
    par, OUT, G, info = C.simulate(tasks, names, (), workers, m18=m18)
    keep, y, codes, days, _, n = C.scored(par, G)
    met, _ = C.pred_metrics(G["P18"][keep], y, codes, m18["base_rate_train"])
    ok_v, chk["validation"] = DV.regression(par, OUT, names, met["auc"], n["scored_points"])
    v18 = json.load(open(SUMMARY18))["prediction"]["valid"]
    chk["metrics_research18_model"] = {k: {"got": met[k], "want": v18[k], "pass": bool(abs(met[k] - v18[k]) <= DV.TOL_AUC)} for k in ("auc", "accuracy_0.5", "brier", "bss", "base_rate")}
    chk["metrics_research18_model"]["counts"] = {"got": [n["grid_points"], n["valid_points"], n["scored_points"], n["days"]],
                                                 "want": [v18["grid_points"], v18["valid_points"], v18["scored_points"], v18["days"]]}
    chk["metrics_research18_model"]["counts"]["pass"] = chk["metrics_research18_model"]["counts"]["got"] == chk["metrics_research18_model"]["counts"]["want"]
    if have:
        va = np.isin(yr18, DV.VALID)
        with np.load(os.path.join(DEV18, "dev18_valid.npz"), allow_pickle=False) as z:
            zn, zl, zo, zP = list(z["names"]), list(z["lrules"]), z["out"], z["P"]
            li = [zl.index(lr) for lr in E.LRULES]
            chk["validation_stored"] = {
                "parents_equal": parents_equal(par, p18[va]), "y_equal": bool(np.array_equal(G["y"], z["y"])), "valid_equal": bool(np.array_equal(G["valid"], z["valid"])),
                "probabilities_equal": same_nan(G["P18"], zP), "probabilities_max_abs_diff": float(np.nanmax(np.abs(G["P18"] - zP))) if G["P18"].shape == zP.shape else None,
                "per_parent_outputs_equal": {m: bool(np.array_equal(OUT[:, names.index(m)], zo[li][:, zn.index(C.engine_name(m))])) for m in names}}
    ok = bool(have and chk["four_input_fit"]["pass"] and ok_v and all(v["pass"] for v in chk["metrics_research18_model"].values())
              and all(chk["train_stored"].values()) and chk["validation_stored"]["parents_equal"] and chk["validation_stored"]["y_equal"]
              and chk["validation_stored"]["valid_equal"] and chk["validation_stored"]["probabilities_equal"]
              and all(chk["validation_stored"]["per_parent_outputs_equal"].values()))
    out = {"pass": ok, "nine_input_model_fitted": False, "checks": chk, "sim_train": sim_t, "sim_valid": {**info, **pinfo, "adv": ainfo},
           "runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")}
    path = os.path.join(HERE, "results", "regress23.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keep_old(path)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=C.jdefault)
    brief = json.loads(json.dumps(out, default=C.jdefault))
    brief["checks"]["validation_stored"]["per_parent_outputs_equal"] = f"{sum(chk['validation_stored']['per_parent_outputs_equal'].values())} of {len(names)} runs equal" if have else None
    print(json.dumps(brief, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args().workers)
