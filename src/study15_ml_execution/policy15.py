"""Research 15 step 3 policy simulation (Kyle 2026-10-29 C; DEVIATIONS15 15-16). Run with python.

Every setting is the research 12 B0 kernel (simulator15, window 2: decisions 09:30:05-15:58:55, flatten 15:59:00, reset 5,
L 3, stale 60 s, min spread 2 ticks, Q model, G-adv) with a filter: P1 quotes a side when its prediction >= theta, P2 only
the side with the larger prediction (ties -> buy) when >= theta, P3 (taker_kernel) enters 100 shares at market on the
larger predicted tk_h >= theta and leaves h seconds later. B0 has no filter. Buy predictions use the buy row, sell
predictions the mirrored sell row (model15.sides). Costs: research 12 scenarios R and R+ (accounting15.net)."""
import dataclasses
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import accounting15 as A  # noqa: E402
import config15  # noqa: E402
import features15  # noqa: E402
import loader15 as L  # noqa: E402
import model15 as M15  # noqa: E402
import simulator15 as S  # noqa: E402
import xy15 as X  # noqa: E402

B0W2 = dataclasses.replace(config15.B0, window=2)
TH12 = (0.0, 0.1, 0.25, 0.5, 1.0)
TH3 = (0.0, 0.5, 1.0, 2.0)
TOD = (("open", 0, 1799), ("mid", 1800, 21599), ("close", 21600, 10**9))       # 09:30-10:00, 10:00-15:30, 15:30-close
SPEARMAN_EVERY = 50


def settings(models=M15.MODELS):
    """[(name, model, target, policy, theta, h)]; B0 first."""
    out = [("B0", None, None, "B0", None, None)]
    for m in models:
        for h in X.HS:
            for pol in ("P1", "P2"):
                out += [(f"{m}|v{h}|{pol}|{th}", m, f"v{h}", pol, th, h) for th in TH12]
            out += [(f"{m}|t{h}|P3|{th}", m, f"t{h}", "P3", th, h) for th in TH3]
    return out


def mark2(mid2, mid_ok, u):
    """mid2 at the end of bar u, or of the last earlier bar with a valid mid (0 if none)."""
    v = np.flatnonzero(mid_ok[:u + 1])
    return int(mid2[v[-1]]) if len(v) else 0


def tod_net(fills, mid2, mid_ok, scenario):
    """Net P&L (1e-4 $) per time bucket: cash and fees of the fills inside the bucket plus the change of the inventory value
    at the bucket ends (mid). The three buckets add up to accounting15.net for a flat day."""
    out = {}
    if len(fills) == 0:
        return {f"{n}_{scenario}": 0 for n, _, _ in TOD}
    mkt_fee, lim_cost = config15.COST[scenario]
    u, s, kind, qty, p = (fills[:, i] for i in range(5))
    fee = np.where(kind == 1, mkt_fee, lim_cost) * qty
    for name, lo, hi in TOD:
        inb = (u >= lo) & (u <= hi)
        cash2 = 2 * int(np.sum((-s * qty * p - fee)[inb]))
        q0, q1 = int(np.sum((s * qty)[u < lo])), int(np.sum((s * qty)[u <= hi]))
        m0 = mark2(mid2, mid_ok, lo - 1) if lo > 0 and q0 else 0
        m1 = mark2(mid2, mid_ok, min(hi, len(mid2) - 1)) if q1 else 0
        out[f"{name}_{scenario}"] = (cash2 + q1 * m1 - q0 * m0) / 2
    return out


def summarize(fills, st, f, close_bar, d0, df):
    a = A.account(fills, f["mid2"], f["mid_ok"], close_bar, d0, df)
    r = {"net_R": int(A.net(a, "R")), "net_Rp": int(A.net(a, "R+")), "sh_lim": a["sh_lim_buy"] + a["sh_lim_sell"],
         "sh_mkt": a["sh_mkt_buy"] + a["sh_mkt_sell"], "n_fill_ev": a["n_fill_ev"],
         "n_mkt_fills": int((fills[:, 2] == 1).sum()) if len(fills) else 0, "skip": int(st[config15.S_SKIP]),
         "n_int": int(st[config15.S_NINT_B] + st[config15.S_NINT_S]), "n_stale": int(st[config15.S_NSTALE])}
    for k in (5, 30, 60):
        r[f"mk{k}_sum2"], r[f"mk{k}_sh"] = a[f"mk{k}_sum2"], a[f"mk{k}_sh"]
    r.update(tod_net(fills, f["mid2"], f["mid_ok"], "R"))
    r.update(tod_net(fills, f["mid2"], f["mid_ok"], "R+"))
    return r


_CACHE = {}


def _models(fold, names):
    key = (fold, tuple(names))
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = M15.load_models(fold, names)
    return _CACHE[key]


def sim_day(task):
    """task: (fold, year, MANIFEST row of the bar file, xy file path, close bar, gate, setting list, model fold).
    Returns (run rows, prediction statistics, spearman sample)."""
    fold, year, brow, xy_path, close_bar, gate, sets, mfold = task
    b = L.load(year, brow)
    f = features15.compute(b)
    with np.load(xy_path, allow_pickle=False) as z:
        Mx = z["xy"]
    us = Mx[:, 0].astype(np.int64)
    xb, xs = M15.sides(Mx)
    names = sorted({f"{s[1]}_{s[2]}" for s in sets if s[1] is not None})
    models = _models(mfold, names)
    pred = {n: (M15.predict(m, xb), M15.predict(m, xs)) for n, m in models.items()}
    p = B0W2.kernel_params("Q", close_bar, gate)
    d0, df = B0W2.d_bounds(close_bar)
    n = len(b["bid"])
    rows = []
    for name, m, tgt, pol, th, h in sets:
        if pol == "B0":
            fills, _, st = S.simulate(b, f, p)
        elif pol in ("P1", "P2"):
            pb, ps = pred[f"{m}_{tgt}"]
            ab, as_ = np.zeros(n, np.bool_), np.zeros(n, np.bool_)
            if pol == "P1":
                ab[us], as_[us] = pb >= th, ps >= th
            else:
                ab[us], as_[us] = (pb >= ps) & (pb >= th), (ps > pb) & (ps >= th)
            fills, _, st = S.simulate(b, f, p, ab, as_)
        else:
            pb, ps = pred[f"{m}_{tgt}"]
            td = np.zeros(n, np.int64)
            td[us] = np.where((pb >= ps) & (pb >= th), 1, np.where((ps > pb) & (ps >= th), -1, 0))
            fills, st = S.simulate_taker(b, f, p, h, td)
        rows.append({"setting": name, "sym": brow["sym"], "date": brow["date"], **summarize(fills, st, f, close_bar, d0, df)})
    # prediction statistics on the gated rows (reference only)
    stats, samp = {}, {}
    if gate and f["mid_ok"][4]:
        g = M15.row_gate(Mx)
        Yb, Ys = M15.targets(Mx, "b")[g], M15.targets(Mx, "s")[g]
        ti = {t: i for i, t in enumerate(M15.TARGETS)}
        keep = np.arange(int(g.sum())) % SPEARMAN_EVERY == 0
        for nm, (pb, ps) in pred.items():
            t = nm.rsplit("_", 1)[1]
            y = np.concatenate([Yb[:, ti[t]], Ys[:, ti[t]]]).astype(np.float64)
            yh = np.concatenate([pb[g], ps[g]]).astype(np.float64)
            ok = ~np.isnan(y)
            y, yh = y[ok], yh[ok]
            stats[nm] = np.array([len(y), y.sum(), (y * y).sum(), yh.sum(), (yh * yh).sum(), (y * yh).sum()])
            kk = np.concatenate([keep, keep])[ok]
            samp[nm] = np.column_stack([y[kk], yh[kk]]).astype(np.float32)
    return rows, stats, samp


def tasks(fold, sets, mfold=None):
    """One task per validation symbol-day of the fold (bar file from loader15, full-decision xy file)."""
    _, val_year = M15.FOLDS[fold]
    g = M15.read_gates().set_index(["sym", "date"])
    m, root = M15.day_files(val_year, full=val_year == 2019)        # 2019: xy_full (every decision); 2020: xy (every decision)
    man = L.manifest(val_year)
    out = []
    for r in m.itertuples():
        out.append((fold, val_year, man[(r.sym, r.date)], os.path.join(root, r.file), int(r.close_bar),
                    bool(g.loc[(r.sym, r.date)].gate), sets, mfold or fold))
    return out


def daily(runs, setting, obs, col="net_R"):
    r = runs[runs.setting == setting]
    s = r.groupby("date")[col].sum()
    return np.array([float(s.get(d, 0)) / 1e4 for d in obs])


def sharpe(x):
    """research12 stats.sharpe: mean / sd(ddof 1) x sqrt(252)."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")


def obs_days(year):
    return sorted({d for _, d in L.manifest(year)})
