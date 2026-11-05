#!/usr/bin/env python3
"""Research 15 step 3 driver (Kyle 2026-10-29 B-G; DEVIATIONS15 15-16). python run15.py <phase> [...]
  train <fold>      fit the 36 models of fold A or B (model15.train_fold)
  sim <fold>        B0 + 252 settings on the fold's validation year (policy15.sim_day, 8 processes)
  select            fold Sharpe table, selection rule, top 10 -> results15/
  ref16             selected model and target trained on 2016- (A16, B16) and simulated (reference, not selection)
  final             selected model and target trained on 2015-2020 (2020 thinned) -> SELECTED15.json
  report            report15.txt
Large tables go to $DATA_DIR/research15_data/val15/ (sha256 in results15/files15.tsv); 2021+ data is never read."""
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model15 as M15  # noqa: E402
import policy15 as P  # noqa: E402

VAL = os.path.join(M15.DATA, "val15")
RES = os.path.join(HERE, "results15")
WORKERS = 8
TOP_GAP = 0.1


def logger(name):
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    f = open(os.path.join(HERE, "logs", f"run15_{name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), "a")

    def log(m):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {m}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def record(path):
    """Append (file, bytes, sha256, utc) to results15/files15.tsv."""
    os.makedirs(RES, exist_ok=True)
    p = os.path.join(RES, "files15.tsv")
    new = not os.path.exists(p)
    with open(p, "a") as f:
        if new:
            f.write("file\tbytes\tsha256\twritten_utc\n")
        f.write(f"{path}\t{os.path.getsize(path)}\t{sha(path)}\t{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}\n")


def run_sim(fold, sets, mfold, tag, log):
    tasks = P.tasks(fold, sets, mfold)
    log(f"sim {tag}: {len(tasks)} symbol-days x {len(sets)} settings, model fold {mfold}")
    t0 = time.time()
    rows, stats, samp = [], {}, {}
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for i, (r, st, sp) in enumerate(pool.imap(P.sim_day, tasks, chunksize=2), 1):
            rows += r
            for k, v in st.items():
                stats[k] = stats.get(k, 0) + v
                samp.setdefault(k, []).append(sp[k])
            if i % 250 == 0 or i == len(tasks):
                log(f"   {i}/{len(tasks)} symbol-days, {time.time() - t0:.0f}s")
    os.makedirs(VAL, exist_ok=True)
    runs = pd.DataFrame(rows).sort_values(["setting", "sym", "date"], kind="stable").reset_index(drop=True)
    p = os.path.join(VAL, f"runs_{tag}.csv.gz")
    runs.to_csv(p, index=False)
    record(p)
    if stats:
        np.savez_compressed(os.path.join(VAL, f"pred_{tag}.npz"), **{f"stats|{k}": v for k, v in stats.items()},
                            **{f"samp|{k}": np.concatenate(v) for k, v in samp.items()})
        record(os.path.join(VAL, f"pred_{tag}.npz"))
    log(f"sim {tag}: done, {len(runs)} run rows, {time.time() - t0:.0f}s")
    return runs


def fold_table(runs, year):
    obs = P.obs_days(year)
    out, daily = [], {"date": obs}
    for s, g in runs.groupby("setting", sort=False):
        xr, xp = P.daily(runs, s, obs), P.daily(runs, s, obs, "net_Rp")
        daily[f"{s}|R"], daily[f"{s}|R+"] = xr, xp
        sh = g.sh_lim.sum() + g.sh_mkt.sum()
        row = {"setting": s, "sharpe_R": P.sharpe(xr), "sharpe_Rp": P.sharpe(xp), "net_R_usd": xr.sum(), "net_Rp_usd": xp.sum(),
               "mean_day_R": xr.mean(), "sh_lim": int(g.sh_lim.sum()), "sh_mkt": int(g.sh_mkt.sum()),
               "n_fill_ev": int(g.n_fill_ev.sum()), "n_mkt_fills": int(g.n_mkt_fills.sum()),
               "pnl_per_share_R_cents": 100 * xr.sum() / sh if sh else float("nan"),
               "pnl_per_share_Rp_cents": 100 * xp.sum() / sh if sh else float("nan"),
               **{f"markout{k}_cents": g[f"mk{k}_sum2"].sum() / 2 / g[f"mk{k}_sh"].sum() / 100 if g[f"mk{k}_sh"].sum() else
                  float("nan") for k in (5, 30, 60)},
               **{f"{b}_{c}_usd": g[f"{b}_{c}"].sum() / 1e4 for b in ("open", "mid", "close") for c in ("R", "R+")}}
        out.append(row)
    return pd.DataFrame(out), pd.DataFrame(daily)


def parse(s):
    if s == "B0":
        return {"model": None, "target": None, "policy": "B0", "theta": None, "h": None}
    m, t, pol, th = s.split("|")
    return {"model": m, "target": t, "policy": pol, "theta": float(th), "h": int(t[1:])}


def simplicity(s):
    """Tie rule (DEVIATIONS15 16): ridge before LightGBM, fewer leaves, larger theta."""
    q = parse(s)
    lg = q["model"] in M15.LGBM
    return (int(lg), M15.LGBM[q["model"]][0] if lg else 0, -q["theta"])


def select(log):
    ta = pd.read_csv(os.path.join(RES, "val15_fold_A.csv")).set_index("setting")
    tb = pd.read_csv(os.path.join(RES, "val15_fold_B.csv")).set_index("setting")
    t = pd.DataFrame({"sharpe_R_A": ta.sharpe_R, "sharpe_R_B": tb.sharpe_R, "sharpe_Rp_A": ta.sharpe_Rp,
                      "sharpe_Rp_B": tb.sharpe_Rp})
    t["mean_sharpe_R"] = (t.sharpe_R_A + t.sharpe_R_B) / 2
    t["mean_sharpe_Rp"] = (t.sharpe_Rp_A + t.sharpe_Rp_B) / 2
    cand = t.drop(index="B0").dropna(subset=["mean_sharpe_R"])
    best = cand.mean_sharpe_R.max()
    near = cand[cand.mean_sharpe_R >= best - TOP_GAP]
    order = sorted(near.index, key=lambda s: (simplicity(s), -near.loc[s, "mean_sharpe_R"], s))
    chosen = order[0]
    t["n_settings"] = len(cand)
    t.sort_values("mean_sharpe_R", ascending=False).to_csv(os.path.join(RES, "val15_settings.csv"))
    sel = {"setting": chosen, **parse(chosen), "mean_sharpe_R": float(t.loc[chosen, "mean_sharpe_R"]),
           "fold_A": {k: float(ta.loc[chosen, k]) for k in ("sharpe_R", "sharpe_Rp")},
           "fold_B": {k: float(tb.loc[chosen, k]) for k in ("sharpe_R", "sharpe_Rp")},
           "best_mean_sharpe_R": float(best), "best_setting": str(cand.mean_sharpe_R.idxmax()),
           "candidates_within_0.1": order, "n_settings": int(len(cand)), "n_settings_nan": int(len(t) - 1 - len(cand)),
           "rule": "max mean of fold A and B daily net Sharpe (R); within 0.1 of the max: ridge > LightGBM, fewer leaves, "
                   "larger theta, then mean Sharpe, then name (DEVIATIONS15 16)"}
    open(os.path.join(RES, "selection15.json"), "w").write(json.dumps(sel, indent=1))
    log(f"select: {len(cand)} settings with a Sharpe, best {sel['best_setting']} {best:.4f}, "
        f"{len(order)} within {TOP_GAP}, chosen {chosen}")
    return sel


def pred_metrics(tag):
    """{name: (n, R2, Spearman on the 1-in-50 sample)} from val15/pred_<tag>.npz (reference only)."""
    from scipy.stats import spearmanr
    z = np.load(os.path.join(VAL, f"pred_{tag}.npz"))
    out = {}
    for k in z.files:
        if not k.startswith("stats|"):
            continue
        nm = k.split("|", 1)[1]
        n, sy, syy, sh, shh, syh = z[k]
        sst = syy - sy * sy / n
        sse = syy - 2 * syh + shh
        s = z[f"samp|{nm}"]
        out[nm] = (int(n), float(1 - sse / sst) if sst > 0 else float("nan"),
                   float(spearmanr(s[:, 0], s[:, 1]).statistic) if len(s) > 2 else float("nan"))
    return out


def importance(fold, name):
    p = os.path.join(M15.MODEL_DIR, fold, name)
    if os.path.exists(p + ".txt"):
        b = M15.lgb().Booster(model_file=p + ".txt")
        g = b.feature_importance("gain")
        return "gain", sorted(zip(b.feature_name(), (g / g.sum()).tolist()), key=lambda x: -x[1])
    m = json.load(open(p + ".json"))
    names = list(M15.FEATS) + [f"na_{f}" for f in M15.X.FEATURES]
    w = np.abs(np.asarray(m["w"]))
    return "|standardized coefficient|", sorted(zip(names, (w / w.sum()).tolist()), key=lambda x: -x[1])


def report(log):
    sel = json.load(open(os.path.join(RES, "SELECTED15.json")))
    st = pd.read_csv(os.path.join(RES, "val15_settings.csv"), index_col=0)
    L_ = [f"Research 15 step 3 report (validation only; test not run) written {dt.datetime.now():%Y-%m-%d %H:%M} ET",
          f"settings with a Sharpe: {sel['n_settings']} (+ B0); without a Sharpe (no trades): {sel['n_settings_nan']}",
          f"selected: {sel['setting']} (model {sel['model']}, target {sel['target']}, policy {sel['policy']}, theta "
          f"{sel['theta']}, h {sel['h']}); mean Sharpe R {sel['mean_sharpe_R']:.3f}; best {sel['best_setting']} "
          f"{sel['best_mean_sharpe_R']:.3f}; within 0.1: {len(sel['candidates_within_0.1'])}",
          f"final model {sel['final_model_file']} sha256 {sel['final_model_sha256']}", ""]
    cols = ["sharpe_R", "sharpe_Rp", "net_R_usd", "net_Rp_usd", "pnl_per_share_R_cents", "pnl_per_share_Rp_cents",
            "n_fill_ev", "sh_lim", "sh_mkt", "n_mkt_fills", "markout5_cents", "markout30_cents", "markout60_cents",
            "open_R_usd", "mid_R_usd", "close_R_usd", "open_R+_usd", "mid_R+_usd", "close_R+_usd"]
    for f in ("A", "B"):
        t = pd.read_csv(os.path.join(RES, f"val15_fold_{f}.csv")).set_index("setting")
        L_.append(f"fold {f} (validation {M15.FOLDS[f][1]}): B0 vs selected")
        L_.append(t.loc[["B0", sel["setting"]], cols].T.to_string(float_format=lambda v: f"{v:.4f}"))
        L_.append("")
    L_.append("top 10 settings by mean Sharpe R (folds A, B)")
    L_.append(st.drop(index="B0").head(10)[["mean_sharpe_R", "sharpe_R_A", "sharpe_R_B", "mean_sharpe_Rp", "sharpe_Rp_A",
                                            "sharpe_Rp_B"]].to_string(float_format=lambda v: f"{v:.3f}"))
    L_.append(f"B0 (9:30-16:00): mean Sharpe R {st.loc['B0', 'mean_sharpe_R']:.3f} (A {st.loc['B0', 'sharpe_R_A']:.3f}, "
              f"B {st.loc['B0', 'sharpe_R_B']:.3f})")
    L_.append("")
    for fold in ("final", "A", "B"):
        kind, imp = importance(fold, f"{sel['model']}_{sel['target']}")
        L_.append(f"input importance ({kind}) of the selected model, fold {fold}: "
                  + ", ".join(f"{n} {v:.3f}" for n, v in imp[:15]))
    L_.append("")
    for f in ("A", "B"):
        pm = pred_metrics(f)
        L_.append(f"prediction metrics fold {f} (gated validation rows; R2 exact, Spearman on 1 in 50 rows):")
        L_.append(pd.DataFrame(pm, index=["n", "R2", "spearman"]).T.sort_index().to_string(float_format=lambda v: f"{v:.4f}"))
    for f, base in (("A16", "A"), ("B16", "B")):
        p = os.path.join(RES, f"val15_fold_{f}.csv")
        if os.path.exists(p):
            t = pd.read_csv(p).set_index("setting")
            b = pd.read_csv(os.path.join(RES, f"val15_fold_{base}.csv")).set_index("setting")
            L_.append(f"reference 2016- training, fold {f}: selected Sharpe R {t.loc[sel['setting'], 'sharpe_R']:.3f} "
                      f"(2015- {b.loc[sel['setting'], 'sharpe_R']:.3f}), R+ {t.loc[sel['setting'], 'sharpe_Rp']:.3f} "
                      f"(2015- {b.loc[sel['setting'], 'sharpe_Rp']:.3f})")
    txt = "\n".join(L_)
    open(os.path.join(RES, "report15.txt"), "w").write(txt + "\n")
    log("report written")
    print(txt)


def main():
    ph = sys.argv[1]
    log = logger(ph + ("_" + sys.argv[2] if len(sys.argv) > 2 else ""))
    os.makedirs(RES, exist_ok=True)
    if ph == "train":
        rec = M15.train_fold(sys.argv[2], log=log)
        log(f"train {sys.argv[2]}: {len(rec['files'])} model files, {rec['seconds']}s")
    elif ph == "sim":
        fold = sys.argv[2]
        runs = run_sim(fold, P.settings(), fold, fold, log)
        tab, daily = fold_table(runs, M15.FOLDS[fold][1])
        tab.to_csv(os.path.join(RES, f"val15_fold_{fold}.csv"), index=False)
        daily.to_csv(os.path.join(RES, f"val15_daily_{fold}.csv"), index=False)
        log(f"sim {fold}: table written ({len(tab)} settings)")
    elif ph == "select":
        select(log)
    elif ph == "ref16":
        sel = json.load(open(os.path.join(RES, "selection15.json")))
        name = f"{sel['model']}_{sel['target']}"
        sets = [s for s in P.settings() if s[0] in ("B0", sel["setting"])]
        for fold, base in (("A16", "A"), ("B16", "B")):
            M15.train_fold(fold, models=(sel["model"],), targets_=(sel["target"],), log=log)
            runs = run_sim(base, [s for s in sets if s[0] != "B0"], fold, fold, log)
            tab, _ = fold_table(runs, M15.FOLDS[fold][1])
            tab.to_csv(os.path.join(RES, f"val15_fold_{fold}.csv"), index=False)
        log(f"ref16 {name}: done")
    elif ph == "final":
        sel = json.load(open(os.path.join(RES, "selection15.json")))
        rec = M15.train_fold("final", models=(sel["model"],), targets_=(sel["target"],), log=log)
        name = f"{sel['model']}_{sel['target']}"
        ext = ".json" if sel["model"] in M15.RIDGE else ".txt"
        path = os.path.join(M15.MODEL_DIR, "final", name + ext)
        out = {**sel, "final_model_file": path, "final_model_sha256": sha(path), "final_train": rec,
               "frozen_utc": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
               "fold_model_sha256": {f: json.load(open(os.path.join(M15.MODEL_DIR, f, "fold.json")))["files"].get(name)
                                     for f in ("A", "B")}}
        open(os.path.join(RES, "SELECTED15.json"), "w").write(json.dumps(out, indent=1))
        log(f"final: {name} sha256 {out['final_model_sha256']}")
    elif ph == "report":
        report(log)
    else:
        sys.exit(f"unknown phase {ph}")


if __name__ == "__main__":
    main()
