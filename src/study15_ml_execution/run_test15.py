#!/usr/bin/env python3
"""Research 15 test (TEST15_DECLARATION.md): E* versus S0 on the sealed 2021+ stock bars, once.
  python run_test15.py --go
Refuses unless research15/UNLOCK15.txt holds exactly the phrase written by Kyle ("연구15 테스트를 한 번 연다"); Claude never
creates that file. Refuses when any result file exists or a frozen item differs. The guard runs before any sealed path is
touched. Results are written once at the end; every path read is recorded. Sources, output folders and the unlock path are
arguments so the synthetic tests can run the same code on 2020 development bars in temporary folders."""
import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exec15 as E  # noqa: E402
import exec15_report as ER  # noqa: E402
import loader15 as L  # noqa: E402
import model15 as M15  # noqa: E402
import r2_15 as R2  # noqa: E402
import xy15 as X  # noqa: E402

R12 = os.path.join(os.path.dirname(HERE), "study12_market_maker")
UNLOCK = os.path.join(HERE, "UNLOCK15.txt")
PHRASE = "연구15 테스트를 한 번 연다"
OUT = os.path.join(HERE, "results15", "test15")
EXT = os.path.join(M15.DATA, "test15")
SEALED_ROOTS = (os.path.join(os.environ.get("DATA_DIR", "data"), "research12_data/bars_1s_holdout_stocks"),
                os.path.join(os.environ.get("DATA_DIR", "data"), "research12_data/bars_1s_reserve_stocks"))
START = "20210104"
ESTAR_NAME, ESTAR = "E12|E2off|E3off|E4off", (2, 0, 0, -1)          # e1_k, e2, e3, e4_x
ML_PATH = os.path.join(M15.MODEL_DIR, "final", "ridge_a10_v5.json")
ML_SHA = "806efd312523e1a4d84e0df917efd7442f2b36b0f3fe54746cd000f967488468"
ML_THETA, ML_IMPROVE = -0.5, 1
SEL_PATH = os.path.join(HERE, "results15", "r2_15_selection.json")
SEL_SHA = "1cd82e3f8e628847"                                           # prefix, checked against the full hash
RESULT_FILES = ("test15_summary.json", "test15_daily.csv", "test15_read_paths.txt", "test15_report.txt")
METHODS = ("S0", "S1", "S2", "S2i", "Estar", "S3")
BOOT_B, BOOT_BLOCK, BOOT_SEED = 10_000, 21, 15


class Refused(Exception):
    pass


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def frozen_checks():
    why = []
    if not os.path.exists(ML_PATH) or sha(ML_PATH) != ML_SHA:
        why.append("frozen ML model missing or sha256 differs")
    if not os.path.exists(SEL_PATH) or not sha(SEL_PATH).startswith(SEL_SHA):
        why.append("r2_15_selection.json missing or sha256 differs")
    elif json.load(open(SEL_PATH))["rule"] != ESTAR_NAME:
        why.append("E* differs from the R2 selection")
    c = dict((n, (k, e2, e3, x)) for n, k, e2, e3, x in R2.COMBOS)
    if c.get(ESTAR_NAME) != ESTAR:
        why.append("E* parameters differ from r2_15.COMBOS")
    return why


def guard(out_dir, unlock_path, frozen_fn):
    """Reasons to refuse, checked before any sealed path is touched."""
    why = []
    if any(os.path.exists(os.path.join(out_dir, f)) for f in RESULT_FILES):
        why.append("test results already exist")
    if not os.path.isfile(unlock_path) or open(unlock_path, encoding="utf-8").read().strip() != PHRASE:
        why.append("no unlock phrase in UNLOCK15.txt")
    why += frozen_fn()
    return why


def close_bars(calendar_path):
    c = pd.read_csv(calendar_path, dtype=str)
    return {d.replace("-", ""): L.NSEC - ((16 - int(t[:2])) * 3600 - int(t[3:5]) * 60) for d, t in zip(c.date, c.close_time)}


def default_sources():
    return [(r, os.path.join(R12, "universe_{y}.tsv"), os.path.join(R12, "calendar.csv"), START) for r in SEALED_ROOTS]


def plan(sources, read):
    """[(root, MANIFEST row, close bar)] for the universe symbols with a bar file on or after each source's start date."""
    out = []
    for root, uni_pat, cal, start in sources:
        mp_ = os.path.join(root, "MANIFEST.tsv")
        read.append(mp_)
        m = pd.read_csv(mp_, sep="\t", dtype=str, keep_default_na=False)
        m = m[m.date >= start].drop_duplicates(["sym", "date"], keep="last")
        read.append(cal)
        cb = close_bars(cal)
        unis = {}
        for y in sorted({d[:4] for d in m.date}):
            p = uni_pat.format(y=y)
            read.append(p)
            unis[y] = set(pd.read_csv(p, sep="\t", dtype=str).sym)
        m = m[[s in unis[d[:4]] for s, d in zip(m.sym, m.date)]]
        out += [(root, r, cb[r["date"]]) for r in m.to_dict("records")]
    return sorted(out, key=lambda t: (t[1]["date"], t[1]["sym"]))


def load_bar(root, row):
    """loader15.load checks with an explicit root: sha256, 21 int32 columns x 23,400 rows, volume identity."""
    path = os.path.join(root, row["file"])
    if sha(path) != row["sha256"]:
        raise Refused(f"{row['sym']} {row['date']}: sha256 differs from MANIFEST")
    with np.load(path, allow_pickle=False) as z:
        if sorted(z.files) != sorted(L.COLS):
            raise Refused(f"{row['sym']} {row['date']}: columns")
        b = {}
        for c in L.COLS:
            if z[c].dtype != np.int32 or z[c].shape != (L.NSEC,):
                raise Refused(f"{row['sym']} {row['date']}: {c} dtype/shape")
            b[c] = z[c].astype(np.int64)
    if not np.array_equal(b["vol"], b["vol_bid"] + b["vol_ask"] + b["vol_in"] + b["vol_unc"]):
        raise Refused(f"{row['sym']} {row['date']}: vol identity")
    return b, path


_ML = {}


def day(task):
    root, row, close_bar = task
    b, path = load_bar(root, row)
    if "m" not in _ML:
        _ML["m"] = json.load(open(ML_PATH))
    n = len(b["bid"])
    dirs, starts = E.parents(row["date"], close_bar)
    ok = X.quote_ok(b)[starts]
    dirs, starts = dirs[ok], starts[ok]
    mid2 = (b["bid"] + b["ask"])[starts]
    us = X.decisions(close_bar)
    Mx = np.full((len(us), len(X.COLUMNS)), np.nan, dtype=np.float32)
    Mx[:, 0] = us
    Mx[:, M15.FI] = X.features(b, us, close_bar)
    xb, xs = M15.sides(Mx)
    pb, ps = np.full(n, np.nan), np.full(n, np.nan)
    pb[us], ps[us] = M15.ridge_predict(_ML["m"], xb), M15.ridge_predict(_ML["m"], xs)
    T_, F_ = np.ones(n, np.bool_), np.zeros(n, np.bool_)
    res = {}
    for qty, hor in E.SIZES:
        outs = {"S0": E.run_method(b, 0, dirs, starts, hor, qty, F_, F_, F_, F_, 0),
                "S1": E.run_method(b, 1, dirs, starts, hor, qty, F_, F_, F_, F_, 0),
                "S2": E.run_method(b, 2, dirs, starts, hor, qty, T_, T_, F_, F_, 0),
                "S2i": E.run_method(b, 2, dirs, starts, hor, qty, T_, T_, F_, F_, 1),
                "Estar": R2.run_rules(b, dirs, starts, hor, qty, *ESTAR),
                "S3": E.run_method(b, 2, dirs, starts, hor, qty, pb >= ML_THETA, ps >= ML_THETA, F_, F_, ML_IMPROVE)}
        cols = {k: [] for k in ("is_r", "is_rp", "lim", "over", "mkt")}
        for mth in METHODS:
            ir, irp, lim, over = E.costs(outs[mth], dirs, mid2, qty)
            for k, v in (("is_r", ir), ("is_rp", irp), ("lim", lim), ("over", over), ("mkt", outs[mth][:, 2])):
                cols[k].append(v)
        res[(qty, hor)] = {k: np.column_stack(v) for k, v in cols.items()}
    meta = {"date": row["date"], "sym": row["sym"], "dir": dirs, "start": starts, "mid2": mid2,
            "spread": (b["ask"] - b["bid"])[starts], "n_dropped": int((~ok).sum())}
    return meta, res, path


def boot_ci(day_sum, day_n, B=BOOT_B, block=BOOT_BLOCK, seed=BOOT_SEED):
    """Circular block bootstrap over days of the per-parent mean sum(day_sum) / sum(day_n); 2.5 and 97.5 percentiles."""
    s, n = np.asarray(day_sum, np.float64), np.asarray(day_n, np.float64)
    T = len(s)
    rng = np.random.default_rng(seed)
    k = math.ceil(T / block)
    idx = (rng.integers(0, T, size=(B, k))[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :T] % T
    stat = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(s.sum() / n.sum()), float(lo), float(hi)


def verdict(lo, hi):
    return "봉인 구간에서도 절감" if hi < 0 else "더 비쌈" if lo > 0 else "판단 보류"


def analyse(par, A, obs):
    """Primary and secondary results from the per-parent table par and matrices A[size][key] (columns METHODS)."""
    j = {m: i for i, m in enumerate(METHODS)}
    main = A[E.SIZES[0]]
    out = {"n_parents": int(len(par)), "n_days": len(obs), "n_symbol_days": int(par[["date", "sym"]].drop_duplicates().shape[0])}

    def diff(a, b_, key="is_r"):
        d = main[key][:, j[a]] - main[key][:, j[b_]]
        g = pd.DataFrame({"date": par.date, "d": d}).groupby("date").d.agg(["sum", "count"])
        g = g.reindex(obs, fill_value=0)
        g = g[g["count"] > 0]
        return boot_ci(g["sum"], g["count"])
    est, lo, hi = diff("Estar", "S0")
    out["primary"] = {"metric": "mean IS (E* - S0), cents per share, R", "estimate": est, "ci95": [lo, hi], "verdict": verdict(lo, hi)}
    out["secondary_diffs"] = {f"{a} - {b_}": dict(zip(("estimate", "lo", "hi"), diff(a, b_)))
                              for a, b_ in (("Estar", "S2i"), ("S2i", "S0"), ("Estar", "S1"), ("S3", "S2i"))}
    mid_c = par.mid2.to_numpy() / 200
    out["methods"] = {}
    for m in METHODS:
        ir, irp = main["is_r"][:, j[m]], main["is_rp"][:, j[m]]
        mk = main["mkt"][:, j[m]]
        out["methods"][m] = {"mean_R": float(ir.mean()), "median_R": float(np.median(ir)), "p95_R": float(np.percentile(ir, 95)),
                             "mean_Rp": float(irp.mean()), "median_Rp": float(np.median(irp)), "p95_Rp": float(np.percentile(irp, 95)),
                             "mean_R_bp": float((ir / mid_c * 1e4).mean()), "limit_frac": float(main["lim"][:, j[m]].mean()),
                             "over_frac": float(main["over"][:, j[m]].sum() / mk.sum()) if mk.sum() else 0.0}
    qty = E.SIZES[0][0]
    sv = pd.Series((main["is_r"][:, j["S0"]] - main["is_r"][:, j["Estar"]]) * qty / 100).groupby(par.date.to_numpy()).sum()
    x = np.array([float(sv.get(d, 0.0)) for d in obs])
    notional = float((qty * par.mid2.to_numpy() / 2e4).sum())
    out["savings_vs_S0"] = {"total_usd": float(x.sum()), "mean_day_usd": float(x.mean()), "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
                            "bp_of_notional": float(x.sum() / notional * 1e4), "per_1bn_usd": float(x.sum() / notional * 1e9),
                            "notional_usd": notional}
    yr = par.date.str[:4].to_numpy()
    out["by_year"] = {y: {m: float(main["is_r"][yr == y, j[m]].mean()) for m in METHODS} for y in sorted(set(yr))}
    for name, col, buckets in (("by_time", "start", ER.TOD_B), ("by_spread", "spread", ER.SPREAD_B)):
        v = par[col].to_numpy()
        out[name] = {bn: {"parents": int(((v >= lo_) & (v <= hi_)).sum()),
                          **{m: float(main["is_r"][(v >= lo_) & (v <= hi_), j[m]].mean()) if ((v >= lo_) & (v <= hi_)).any() else None
                             for m in METHODS}} for bn, lo_, hi_ in buckets}
    sens = A[E.SIZES[1]]
    out["sensitivity_300sh_60s"] = {m: float(sens["is_r"][:, j[m]].mean()) for m in METHODS}
    return out, x


def main(argv=None, sources=None, out_dir=OUT, ext_dir=EXT, unlock_path=UNLOCK, workers=8, frozen_fn=frozen_checks):
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        raise Refused("give --go")
    why = guard(out_dir, unlock_path, frozen_fn)
    if why:
        raise Refused("; ".join(why))
    t0 = dt.datetime.now()
    read = [unlock_path, ML_PATH, SEL_PATH]
    tasks = plan(sources if sources is not None else default_sources(), read)
    metas, res = [], {s: {k: [] for k in ("is_r", "is_rp", "lim", "over", "mkt")} for s in E.SIZES}
    if workers > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(day, tasks, chunksize=8)
    else:
        results = [day(t) for t in tasks]
    for meta, r, path in results:
        metas.append(meta)
        read.append(path)
        for s in E.SIZES:
            for k in res[s]:
                res[s][k].append(r[s][k])
    par = pd.DataFrame({"date": np.concatenate([[m["date"]] * len(m["dir"]) for m in metas]),
                        "sym": np.concatenate([[m["sym"]] * len(m["dir"]) for m in metas]),
                        **{k: np.concatenate([m[k] for m in metas]) for k in ("dir", "start", "mid2", "spread")}})
    A = {s: {k: np.vstack(v) for k, v in d.items()} for s, d in res.items()}
    obs = sorted({t[1]["date"] for t in tasks})
    summary, savings = analyse(par, A, obs)
    summary.update({"declaration_sha256": sha(os.path.join(HERE, "TEST15_DECLARATION.md")) if os.path.exists(os.path.join(HERE, "TEST15_DECLARATION.md")) else None,
                    "estar": ESTAR_NAME, "ml_model_sha256": ML_SHA, "dropped_parents": int(sum(m["n_dropped"] for m in metas)),
                    "first_day": obs[0] if obs else None, "last_day": obs[-1] if obs else None,
                    "started": f"{t0:%Y-%m-%d %H:%M:%S}", "finished": f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"})
    # everything is computed; write once
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ext_dir, exist_ok=True)
    np.savez_compressed(os.path.join(ext_dir, "test15_parents.npz"), methods=np.array(METHODS),
                        **{f"{k}|{q}_{h}": v for (q, h), d in A.items() for k, v in d.items()})
    par.to_csv(os.path.join(ext_dir, "test15_parents.csv.gz"), index=False)
    summary["parents_file_sha256"] = sha(os.path.join(ext_dir, "test15_parents.npz"))
    pd.DataFrame({"date": obs, "savings_vs_S0_usd": savings}).to_csv(os.path.join(out_dir, "test15_daily.csv"), index=False)
    open(os.path.join(out_dir, "test15_read_paths.txt"), "w").write("\n".join(read) + "\n")
    open(os.path.join(out_dir, "test15_summary.json"), "w").write(json.dumps(summary, indent=1, ensure_ascii=False))
    p = summary["primary"]
    open(os.path.join(out_dir, "test15_report.txt"), "w").write(
        f"Research 15 test ({summary['first_day']} .. {summary['last_day']}, {summary['n_days']} days, {summary['n_parents']} parents)\n"
        f"primary: {p['metric']} = {p['estimate']:.4f}, 95% CI [{p['ci95'][0]:.4f}, {p['ci95'][1]:.4f}] -> {p['verdict']}\n")
    return summary


if __name__ == "__main__":
    try:
        s = main(sys.argv[1:])
        print(json.dumps(s["primary"], ensure_ascii=False))
    except Refused as e:
        sys.exit(f"refused: {e}")
