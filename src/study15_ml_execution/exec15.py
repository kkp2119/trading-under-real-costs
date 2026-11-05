#!/usr/bin/env python3
"""Research 15-E: machine learning for cheaper execution (EXEC15_DECLARATION.md; implementation DEVIATIONS15 20).
Validation years 2019 (fold A models) and 2020 (fold B models) only; 2021+ data is never read; no model is retrained.
  python exec15.py sim A|B     parent orders x (S0, S1, S2, S2i, 216 S3 settings) x (1,000 sh / 300 s, 300 sh / 60 s)
  python exec15.py select      selection rule (declaration 5) -> results15/exec15_selection.json
  python exec15.py report      results15/exec15_report.txt
Per-parent matrices go to $DATA_DIR/research15_data/val15/exec15_<fold>.npz; tables to results15/."""
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features15  # noqa: E402
import loader15 as L  # noqa: E402
import model15 as M15  # noqa: E402
import policy15 as P  # noqa: E402
from fillmodel15 import fill_buy, fill_sell, jit  # noqa: E402  (the fill functions simulator15 uses)

RES = os.path.join(HERE, "results15")
VAL = os.path.join(M15.DATA, "val15")
TICK, N_PARENT, FIRST_START, STEP = 100, 20, 299, 5
SIZES = ((1000, 300), (300, 60))                     # main, sensitivity (reference only)
FEE_MKT, REBATE = 0.3, 0.2                           # cents per share: R market fee; R+ limit rebate
HS, THETAS = (5, 30, 60), (-0.5, 0.0, 0.5)
TIE_GAP = 0.05
BASELINES = ("S0", "S1", "S2", "S2i")


def s3_settings():
    out = []
    for m in M15.MODELS:
        for h in HS:
            for th in THETAS:
                for imp in (0, 1):
                    for rush in (0, 1):
                        out.append((f"S3|{m}|{h}|{th}|imp{imp}|rush{rush}", m, h, th, imp, rush))
    return out


SETTINGS = list(BASELINES) + [s[0] for s in s3_settings()]


def start_grid(close_bar, horizon=300):
    """Start bars u = 299 (09:35:00 end), 304, ..., close - 61 - 5 - horizon (15:53:55 end on a full day)."""
    return np.arange(FIRST_START, close_bar - 61 - STEP - horizon + 1, STEP, dtype=np.int64)


def parents(day, close_bar):
    """(direction, start bar) of the 20 parent orders of a date (same for every symbol): default_rng(15 + YYYYMMDD)."""
    rng = np.random.default_rng(15 + int(day))
    dirs = rng.permutation(np.array([1] * (N_PARENT // 2) + [-1] * (N_PARENT // 2), dtype=np.int64))
    starts = rng.choice(start_grid(close_bar), N_PARENT, replace=False).astype(np.int64)
    return dirs, starts


@jit
def exec_day(method, dirs, starts, horizon, qty, bid, ask, bsz, asz, bid_max, bid_min, ask_max, ask_min, vol_bid, vol_ask,
             vol_bid_odd, vol_ask_odd, px_min, px_max, post_b, post_s, rush_b, rush_s, improve, out):
    """method 0 S0, 1 S1, 2 posting (S2 / S2i / S3 through post_* / rush_* / improve). out[i] = (limit shares, limit
    notional, market shares, market notional, market shares above the displayed size) for parent i."""
    for i in range(len(dirs)):
        d, s = dirs[i], starts[i]
        D = s + horizon
        lim_sh, lim_not, mkt_sh, mkt_not, over = 0, 0, 0, 0, 0
        if method == 0 or method == 1:
            n_sl = 1 if method == 0 else 10
            step = 0 if method == 0 else max(5, (horizon // 10) // 5 * 5)
            for k in range(n_sl):
                u = s + k * step
                q = qty // n_sl
                px = ask[u] if d > 0 else bid[u]
                disp = asz[u] if d > 0 else bsz[u]
                mkt_sh += q
                mkt_not += q * px
                over += max(0, q - disp)
        else:
            rem, has, px, a4 = qty, False, 0, 0
            for u in range(s, D + 1):
                if u > s and has and rem > 0:
                    if d > 0:
                        f, a4 = fill_buy(2, px, rem, a4, u, ask_min, bid_max, bid_min, px_min, vol_bid, vol_bid_odd)
                    else:
                        f, a4 = fill_sell(2, px, rem, a4, u, bid_max, ask_min, ask_max, px_max, vol_ask, vol_ask_odd)
                    if f > 0:
                        lim_sh += f
                        lim_not += f * px
                        rem -= f
                if rem == 0:
                    break
                if (u - s) % STEP != 0:
                    continue
                rush = rush_b[u] if d > 0 else rush_s[u]
                if u == D or rush:
                    mp_ = ask[u] if d > 0 else bid[u]
                    disp = asz[u] if d > 0 else bsz[u]
                    mkt_sh += rem
                    mkt_not += rem * mp_
                    over += max(0, rem - disp)
                    rem = 0
                    break
                post = post_b[u] if d > 0 else post_s[u]
                spr = ask[u] - bid[u]
                if post and spr >= TICK:
                    if improve == 1 and spr >= 3 * TICK:
                        npx = bid[u] + TICK if d > 0 else ask[u] - TICK
                        na4 = 0
                    else:
                        npx = bid[u] if d > 0 else ask[u]
                        na4 = 4 * (bsz[u] if d > 0 else asz[u])
                    if not (has and npx == px):
                        has, px, a4 = True, npx, na4
                else:
                    has = False
        out[i, 0], out[i, 1], out[i, 2], out[i, 3], out[i, 4] = lim_sh, lim_not, mkt_sh, mkt_not, over


def run_method(b, method, dirs, starts, horizon, qty, post_b, post_s, rush_b, rush_s, improve):
    out = np.zeros((len(dirs), 5), dtype=np.int64)
    exec_day(method, dirs, starts, horizon, qty, b["bid"], b["ask"], b["bsz"], b["asz"], b["bid_max"], b["bid_min"],
             b["ask_max"], b["ask_min"], b["vol_bid"], b["vol_ask"], b["vol_bid_odd"], b["vol_ask_odd"], b["px_min"],
             b["px_max"], post_b, post_s, rush_b, rush_s, improve, out)
    return out


def costs(out, dirs, mid2, qty):
    """(IS R, IS R+) in cents per share, limit fraction, market shares above the displayed size."""
    avg = (out[:, 1] + out[:, 3]) / qty                                # 1e-4 dollars
    is_gross = dirs * (avg - mid2 / 2) / 100
    is_r = is_gross + FEE_MKT * out[:, 2] / qty
    return is_r, is_r - REBATE * out[:, 0] / qty, out[:, 0] / qty, out[:, 4]


_CACHE = {}


def day_task(task):
    """task: (fold, year, bar MANIFEST row, xy path, close bar, model fold) -> (parent meta, {size: arrays})."""
    fold, year, brow, xy_path, close_bar, mfold = task
    b = L.load(year, brow)
    n = len(b["bid"])
    dirs, starts = parents(brow["date"], close_bar)
    ok = P.X.quote_ok(b)[starts]
    dirs, starts = dirs[ok], starts[ok]
    mid2 = (b["bid"] + b["ask"])[starts]
    meta = {"date": brow["date"], "sym": brow["sym"], "dir": dirs, "start": starts, "mid2": mid2,
            "spread": (b["ask"] - b["bid"])[starts], "n_dropped": int((~ok).sum())}
    with np.load(xy_path, allow_pickle=False) as z:
        Mx = z["xy"]
    us = Mx[:, 0].astype(np.int64)
    xb, xs = M15.sides(Mx)
    names = [f"{m}_{t}" for m in M15.MODELS for t in M15.TARGETS]
    key = (mfold,)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = M15.load_models(mfold, names)
    pred = {}
    for nm, m in _CACHE[key].items():
        pb, ps = np.full(n, np.nan), np.full(n, np.nan)
        pb[us], ps[us] = M15.predict(m, xb), M15.predict(m, xs)
        pred[nm] = (pb, ps)
    T_, F_ = np.ones(n, np.bool_), np.zeros(n, np.bool_)
    res = {}
    for qty, horizon in SIZES:
        cols = {"is_r": [], "is_rp": [], "lim": [], "over": [], "mkt": []}
        for name in SETTINGS:
            if name == "S0":
                out = run_method(b, 0, dirs, starts, horizon, qty, F_, F_, F_, F_, 0)
            elif name == "S1":
                out = run_method(b, 1, dirs, starts, horizon, qty, F_, F_, F_, F_, 0)
            elif name in ("S2", "S2i"):
                out = run_method(b, 2, dirs, starts, horizon, qty, T_, T_, F_, F_, int(name == "S2i"))
            else:
                _, m, h, th, imp, rush = name.split("|")
                vb, vs = pred[f"{m}_v{h}"]
                tb, ts = pred[f"{m}_t{h}"]
                rb = (tb >= 0) if rush == "rush1" else F_
                rs = (ts >= 0) if rush == "rush1" else F_
                out = run_method(b, 2, dirs, starts, horizon, qty, vb >= float(th), vs >= float(th), rb, rs, int(imp == "imp1"))
            ir, irp, lim, over = costs(out, dirs, mid2, qty)
            cols["is_r"].append(ir)
            cols["is_rp"].append(irp)
            cols["lim"].append(lim)
            cols["over"].append(over)
            cols["mkt"].append(out[:, 2])
        res[(qty, horizon)] = {k: np.column_stack(v).astype(np.float32) for k, v in cols.items()}
    return meta, res


def tasks(fold):
    year = M15.FOLDS[fold][1]
    m, root = M15.day_files(year, full=year == 2019)
    man, cb = L.manifest(year), L.close_bars(year)
    return [(fold, year, man[(r.sym, r.date)], os.path.join(root, r.file), cb[r.date], fold) for r in m.itertuples()]


def sim(fold, log):
    t = tasks(fold)
    log(f"exec sim {fold}: {len(t)} symbol-days, {len(SETTINGS)} methods x {len(SIZES)} sizes, models {fold}")
    t0 = time.time()
    metas, mats = [], {s: {k: [] for k in ("is_r", "is_rp", "lim", "over", "mkt")} for s in SIZES}
    with mp.get_context("spawn").Pool(8) as pool:
        for i, (meta, res) in enumerate(pool.imap(day_task, t, chunksize=2), 1):
            metas.append(meta)
            for s in SIZES:
                for k in mats[s]:
                    mats[s][k].append(res[s][k])
            if i % 250 == 0 or i == len(t):
                log(f"   {i}/{len(t)} symbol-days, {time.time() - t0:.0f}s")
    par = pd.DataFrame({"date": np.concatenate([[m["date"]] * len(m["dir"]) for m in metas]),
                        "sym": np.concatenate([[m["sym"]] * len(m["dir"]) for m in metas]),
                        "dir": np.concatenate([m["dir"] for m in metas]), "start": np.concatenate([m["start"] for m in metas]),
                        "mid2": np.concatenate([m["mid2"] for m in metas]), "spread": np.concatenate([m["spread"] for m in metas])})
    arrays = {f"{k}|{q}_{h}": np.concatenate(v) for (q, h), d in mats.items() for k, v in d.items()}
    os.makedirs(VAL, exist_ok=True)
    p = os.path.join(VAL, f"exec15_{fold}.npz")
    np.savez_compressed(p, settings=np.array(SETTINGS), **arrays)
    par.to_csv(os.path.join(VAL, f"exec15_{fold}_parents.csv.gz"), index=False)
    log(f"exec sim {fold}: {len(par)} parents, dropped (no quote at start) {sum(m['n_dropped'] for m in metas)}, "
        f"{time.time() - t0:.0f}s -> {p}")


def load(fold, size=SIZES[0]):
    z = np.load(os.path.join(VAL, f"exec15_{fold}.npz"))
    par = pd.read_csv(os.path.join(VAL, f"exec15_{fold}_parents.csv.gz"), dtype={"date": str, "sym": str})
    tag = f"{size[0]}_{size[1]}"
    return par, {k: z[f"{k}|{tag}"] for k in ("is_r", "is_rp", "lim", "over", "mkt")}, list(z["settings"])


def tie_key(name):
    _, m, h, th, imp, rush = name.split("|")
    lg = m in M15.LGBM
    return (int(lg), M15.LGBM[m][0] if lg else 0, int(imp == "imp1"), int(rush == "rush1"), -float(th))


def select(log):
    means = {}
    for f in ("A", "B"):
        _, a, st = load(f)
        means[f] = pd.Series(a["is_r"].mean(axis=0, dtype=np.float64), index=st)
    t = pd.DataFrame({"is_R_A": means["A"], "is_R_B": means["B"]})
    t["mean_is_R"] = (t.is_R_A + t.is_R_B) / 2
    s3 = t[[s.startswith("S3|") for s in t.index]]
    best = s3.mean_is_R.min()
    near = s3[s3.mean_is_R <= best + TIE_GAP]
    order = sorted(near.index, key=lambda s: (tie_key(s), near.loc[s, "mean_is_R"], s))
    sel = {"setting": order[0], "mean_is_R": float(t.loc[order[0], "mean_is_R"]), "best_setting": str(s3.mean_is_R.idxmin()),
           "best_mean_is_R": float(best), "candidates_within_0.05": order, "n_s3": int(len(s3)),
           "baselines_mean_is_R": {b_: float(t.loc[b_, "mean_is_R"]) for b_ in BASELINES},
           "rule": "min mean of fold A and B mean IS (R); within 0.05 c: ridge > LightGBM, fewer leaves, improve off, rush off, "
                   "larger theta_post, then mean IS, then name (declaration 5, DEVIATIONS15 20)"}
    t.sort_values("mean_is_R").to_csv(os.path.join(RES, "exec15_settings.csv"))
    open(os.path.join(RES, "exec15_selection.json"), "w").write(json.dumps(sel, indent=1))
    log(f"exec select: best {sel['best_setting']} {best:.4f}, {len(order)} within {TIE_GAP}, chosen {sel['setting']}")
    return sel


def main():
    ph = sys.argv[1]
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    f = open(os.path.join(HERE, "logs", f"exec15_{ph}{'_' + sys.argv[2] if len(sys.argv) > 2 else ''}_"
                                        f"{dt.datetime.now():%Y%m%d_%H%M%S}.log"), "a")

    def log(m):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {m}"
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
    if ph == "sim":
        sim(sys.argv[2], log)
    elif ph == "select":
        select(log)
    elif ph == "report":
        import exec15_report
        exec15_report.report(log)
    else:
        sys.exit(f"unknown phase {ph}")


if __name__ == "__main__":
    main()
