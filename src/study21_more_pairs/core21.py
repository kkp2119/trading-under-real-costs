#!/usr/bin/env python3
"""Research 21: the simulation and the tables shared by dev21.py, run_test21.py and regress21.py (one code path for the
development run, the rehearsal, the sealed run and the regression on research 20's development data).

The engine is research 20's, imported unchanged: signal20 (grid, z, flags, signal-only timetable, delayed mids), pair20
(pair engine on the research 17 kernel), core20 (aggregation and table rows), stats20. Their sha256 must equal the files
of tag code-freeze-20 (FROZEN20; checked at import). New here:
  bars        read through loader21 (quote sizes divided by 100 from 2025-11-03)
  runs        market orders only: per pair-day and signal setting three runs are aggregated, mid (signal-only timetable at
              the mid, H1), M_L3D1 and M_L0D0 (the same M trades with depth penalty 1 / 0); the first three kinds of core20
  families    every task carries its family (E, S, or C for the control pairs); tables are built per family from that
              family's pair-days only, over every trading day of the window
  diagnostics per pair-day and setting, inside the entry hours: grid points with a signal, those whose rolling standard
              deviation is below the 1 bp floor, those with |z| >= z_in, and of these the ones kept out by the kappa rule
  date split  mean daily values before 2025-11-03 and from that day on, each part with its own bootstrap index"""
import hashlib
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
R20 = os.path.join(PROJ, "study20_pairs")
FROZEN20 = {"study20_pairs/signal20.py": "e5d1127ccad05c9980cfc6c79dbffbfc2fb19c5fef40d60674af05cf33bad224",
            "study20_pairs/pair20.py": "27623a242fd03fbb843ea71853f1d2409793c0b78664466fc5d4ae1b20f5e5d4",
            "study20_pairs/core20.py": "35887641113e9b3ca581e986a3c0b06d67724438275e0f2dfff2de4469aa3b9c",
            "study20_pairs/stats20.py": "45f16c9e7b18e05082b8d4e401fefc401ab1685601b8f59626ea9ed3d16942f0",
            "study20_pairs/loader20.py": "b6fc30b560e66897fe3a502a4e1dccc3a0328e2118a2d62ae110466d5e0cce30",
            "study17_new_stocks/engine17.py": "2a9ddcb7db7b96201cbd63b62e44c7c00140447a947e883aeeebf6a542d8d42a"}


def frozen_differences():
    """Paths of FROZEN20 whose file differs from the sha256 of tag code-freeze-20."""
    return [p for p, h in FROZEN20.items() if hashlib.sha256(open(os.path.join(PROJ, p), "rb").read()).hexdigest() != h]


if frozen_differences():
    raise ImportError(f"engine files differ from tag code-freeze-20: {frozen_differences()}")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if R20 not in sys.path:
    sys.path.insert(1, R20)
import core20 as C  # noqa: E402
import pair20 as PE  # noqa: E402
import signal20 as S  # noqa: E402
import stats20 as ST20  # noqa: E402
import loader21 as L21  # noqa: E402
import stats21 as ST21  # noqa: E402

KINDS = C.KINDS[:3]                                    # ("mid", "M_L3D1", "M_L0D0")
VARS = ("L3D1", "L0D0")
COL, NAGG = C.COL, C.NAGG
DIAG = ("signal_points", "sigma_floor_points", "z_points", "kappa_blocked_points")
NDIAG = len(DIAG)
RUNS = ("mid",) + tuple(f"mid_delay{d}" for d in S.MID_DELAYS) + KINDS[1:]


def sigma_raw(s, valid, W):
    """The rolling sample standard deviation of signal20.zscore before its 1 bp floor (nan where there is no signal)."""
    n, K = W * 12, len(s)
    sd = np.full(K, np.nan)
    iv = np.flatnonzero(valid)
    if len(iv) > n:
        sd[iv[n:]] = np.lib.stride_tricks.sliding_window_view(s[iv], n)[:-1].std(axis=1, ddof=1)
    return sd


def diagnose(B, W, z_in, kappa, close_bar):
    """Counts over the grid points inside the entry hours that have a signal: all, sigma below the floor, |z| >= z_in,
    and of the last the points where |s - m| < kappa x (relative spreads). z points - blocked points = entry flags."""
    z, dev, has = S.zscore(B["s"], B["valid"], W)
    lo, hi = S.entry_bars(W, close_bar)
    inh = has & (B["g"] >= lo) & (B["g"] <= hi)
    zok = inh & (np.where(has, np.abs(z), 0.0) >= z_in)
    blocked = zok & ~(np.where(has, np.abs(dev), 0.0) >= kappa * B["rs"])
    floor = inh & (np.nan_to_num(sigma_raw(B["s"], B["valid"], W), nan=np.inf) < S.SIGMA_FLOOR)
    return np.array([inh.sum(), floor.sum(), zok.sum(), blocked.sum()], dtype=np.float64)


def pairday(a, b, cb, cfgs, delay=0, keep=(), mid_delays=(), mid_only=False, diag=False):
    """One pair-day from its two bar dicts (core20.pairday_all without the P runs): aggregates [settings, kinds, NAGG], the
    number of settings whose M trades differ from the signal-only timetable (0 expected, delay 0 only), kept M trade rows
    {setting index: rows}, valid and total grid points, the left-out flag (a bar without a quote state after a leg's first
    state, SPEC20_ADDENDUM B1), delayed-mid sums [settings, delays, (trades, dollars, wins)] (delay 0 only) and the
    diagnostics [settings, NDIAG]. mid_only skips the M engine (control pairs)."""
    out = np.zeros((len(cfgs), len(KINDS), NAGG))
    D = np.zeros((len(cfgs), len(mid_delays), 3))
    Gd = np.zeros((len(cfgs), NDIAG))
    kept = {ci: np.zeros((0, PE.NTR), dtype=np.int64) for ci in keep}
    if not (C.state_ok(a) and C.state_ok(b)):
        return out, 0, kept, 0, len(S.grid(cb)), 1, D, Gd
    B = S.base(a, b, cb)
    XA, XB = (None, None) if mid_only else (PE.stack(a), PE.stack(b))
    bars = S.bar_base(a, b, cb) if (mid_delays and delay == 0) else None
    mismatch = 0
    for ci, (W, zi, ka) in enumerate(cfgs):
        side, exz = S.flags(B, W, zi, ka, cb)
        mt = None
        if delay == 0:
            mt = S.mid_trades(B, side, exz, cb)
            out[ci, 0] = C._agg_mid(mt)
            for di, d in enumerate(mid_delays):
                pd_ = S.mid_pnl_delayed(mt, *bars, d)
                D[ci, di] = len(pd_), pd_.sum(), (pd_ > 0).sum()
        if not mid_only:
            trm = PE.trades(PE.M, 3, delay, B, side, exz, XA, XB, cb)
            if mt is not None:
                mismatch += int(not (np.array_equal(mt[:, :6], trm[:, :6]) and np.array_equal(mt[:, 6:8], trm[:, PE.MID0:PE.MID0 + 2])
                                     and np.array_equal(mt[:, 8:10], trm[:, PE.MID0 + 2:PE.MID0 + 4])))
            out[ci, 1], out[ci, 2] = C._agg(trm, 1, PE.M), C._agg(trm, 0, PE.M)
            if ci in keep:
                kept[ci] = trm
        if diag:
            Gd[ci] = diagnose(B, W, zi, ka, cb)
    return out, mismatch, kept, int(B["valid"].sum()), len(B["g"]), 0, D, Gd


def work(args):
    """One pair-day read through loader21 (see pairday)."""
    t, root, allow_sealed, cfgs, delay, keep, mid_delays, mid_only, diag = args
    a, b = L21.load(root, t["row_a"], allow_sealed), L21.load(root, t["row_b"], allow_sealed)
    return pairday(a, b, t["close_bar"], cfgs, delay, keep, mid_delays, mid_only, diag)


def simulate(tasks, root, allow_sealed, cfgs, delay=0, keep=(), workers=4, mid_delays=(), mid_only=False, diag=False):
    """(AGG array [pair-days, settings, kinds, NAGG], info, extra). extra: "kept" {setting index: M trade rows with the
    pair-day index in column 0}, "mid_delayed" [pair-days, settings, delays, 3], "diag" [pair-days, settings, NDIAG].
    workers = 0 runs in this process (synthetic tests)."""
    args = [(t, root, allow_sealed, list(cfgs), delay, tuple(keep), tuple(mid_delays), mid_only, diag) for t in tasks]
    if workers and args:
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(work, args, chunksize=8)
    else:
        res = [work(x) for x in args]
    nc, nd = len(cfgs), len(mid_delays)
    A = np.stack([r[0] for r in res]) if res else np.zeros((0, nc, len(KINDS), NAGG))
    extra = {"kept": {ci: (np.concatenate([np.column_stack([np.full(len(r[2][ci]), i, dtype=np.int64), r[2][ci]]) for i, r in enumerate(res)])
                           if res else np.zeros((0, PE.NTR + 1), dtype=np.int64)) for ci in keep},
             "mid_delayed": np.stack([r[6] for r in res]) if res else np.zeros((0, nc, nd, 3)),
             "diag": np.stack([r[7] for r in res]) if res else np.zeros((0, nc, NDIAG))}
    info = {"pair_days": len(tasks), "settings_with_m_trades_unlike_the_signal_timetable": int(sum(r[1] for r in res)),
            "valid_grid_points": int(sum(r[3] for r in res)), "grid_points": int(sum(r[4] for r in res)),
            "pair_days_left_out_no_state_after_first_state": int(sum(r[5] for r in res))}
    return A, info, extra


# ---------------------------------------------------------------- families and tables
def family_index(tasks, family):
    return np.array([i for i, t in enumerate(tasks) if t["family"] == family], dtype=np.int64)


def block(A, D, M, ci, n_days, idx, name, mid_only=False):
    """Tables of one family and signal setting: the row of every run (core20.row), the per-trade mid P&L with its interval
    on the mid row, the delayed-mid rows (SPEC20_ADDENDUM B2) and the 5-second sentence."""
    rows = {k: C.row(A, M, ci, ki, n_days, idx, f"{name}_{k}") for ki, k in enumerate(KINDS) if not (mid_only and ki)}
    rows["mid"]["per_trade"] = dict(zip(("mean", "lo", "hi"), C.per_trade(A, M, ci, 0, "pnl", n_days, idx)))
    dr = C.delayed_rows(A, D, M, ci, S.MID_DELAYS, n_days, idx)
    return {"rows": rows, "delayed": dr, "sentence_5s": S.delay_sentence(dr["5"]["ci_lo"], dr["5"]["ratio_to_delay0"])}


def run_series(A, D, M, ci, run, n_days):
    """(daily P&L, daily trade count) of one run name of RUNS for setting ci, over every day of the window."""
    if run.startswith("mid_delay"):
        di = S.MID_DELAYS.index(int(run[9:]))
        return ST20.daily(M["day"], D[:, ci, di, 1], n_days), ST20.daily(M["day"], D[:, ci, di, 0], n_days)
    ki = KINDS.index(run)
    return C.daily(A, M, ci, ki, "pnl", n_days), C.daily(A, M, ci, ki, "n", n_days)


def date_split(d, n, days, cut=L21.SIZE_FROM):
    """Mean daily value with the day-block 95% interval and the trades, for the days before `cut` and from `cut` on; each
    part has its own bootstrap index (its own length, seed 21). None for a part without days."""
    dd = np.array(days)
    out = {}
    for key, sel in (("before", dd < cut), ("from", dd >= cut)):
        T = int(sel.sum())
        if T == 0:
            out[key] = None
            continue
        mean, lo, hi = ST20.mean_ci(d[sel], ST21.boot_index(T))
        out[key] = {"days": T, "first_day": str(dd[sel][0]), "last_day": str(dd[sel][-1]), "mean_daily_usd": mean, "ci_lo": lo, "ci_hi": hi,
                    "trades": int(n[sel].sum())}
    return out


DAILY_COLS = ("date", "family", "pair", "name", "setting", "trades_mid", "mid") + tuple(f"mid_delay{d}" for d in S.MID_DELAYS) + (
    "trades_M", "M_L3D1", "M_L0D0")


def pair_daily(A, D, tasks, ci, setting):
    """One row per pair-day for one setting (DAILY_COLS): trades and dollars of the signal-only timetable (mid, and the mid
    priced 1, 5 and 30 seconds late), trades and dollars of the M runs. A control pair-day has the mid columns only."""
    rows = []
    for i, t in enumerate(tasks):
        m = (None, None, None) if t["family"] == L21.CONTROL else (int(A[i, ci, 1, COL["n"]]), A[i, ci, 1, COL["pnl"]], A[i, ci, 2, COL["pnl"]])
        rows.append((t["date"], t["family"], t["pair"], t["name"], setting, int(A[i, ci, 0, COL["n"]]), A[i, ci, 0, COL["pnl"]])
                    + tuple(D[i, ci, di, 1] for di in range(len(S.MID_DELAYS))) + m)
    return rows


def diag_shares(Gd, M, cfgs):
    """Diagnostics of one family from [pair-days, settings, NDIAG]: per W the share of signal points with sigma below the
    floor, per setting the share of |z| >= z_in points kept out by the kappa rule; for the family and per pair."""
    def shares(g):
        tot = g.sum(axis=0)
        fl, kb = {}, {}
        for ci, c in enumerate(cfgs):
            fl.setdefault(f"W{c[0]}", {"signal_points": int(tot[ci, 0]), "below_floor": int(tot[ci, 1]),
                                       "share": float(tot[ci, 1] / tot[ci, 0]) if tot[ci, 0] else float("nan")})
            kb[ST21.sig_name(c)] = {"z_points": int(tot[ci, 2]), "kappa_blocked": int(tot[ci, 3]),
                                    "share": float(tot[ci, 3] / tot[ci, 2]) if tot[ci, 2] else float("nan")}
        return {"sigma_floor": fl, "kappa_blocked": kb}
    out = {"family": shares(Gd), "by_pair": {}}
    for pi, nm in enumerate(M["names"]):
        out["by_pair"][nm] = shares(Gd[M["pair"] == pi])
    return out


def family_tables(A, D, tasks, days, cfgs, idx, family, mid_only=False):
    """One family's pair-days only: (meta, {setting name: {"signal", "rows", "delayed", "sentence_5s"}}, A, D, tasks)."""
    sel = family_index(tasks, family)
    tf = [tasks[i] for i in sel]
    Af, Df = A[sel], D[sel]
    M = C.meta(tf, days)
    blocks = {ST21.sig_name(c): {"signal": list(c), **block(Af, Df, M, ci, len(days), idx, ST21.sig_name(c), mid_only)} for ci, c in enumerate(cfgs)}
    return M, blocks, Af, Df, tf


def side_row(label, blk):
    """A line of the side-by-side table (setting W30_z2_k0, market orders) from a block: mid P&L per trade, the share left
    after a 5-second delay, trades per day, and per fill assumption the cost per trade (execution cost + fee) and the P&L
    per trade after costs (None when the M engine was not run: control pairs, SPEC21_ADDENDUM C1)."""
    mid = blk["rows"]["mid"]
    r = {"label": label, "trades": mid["trades"], "trades_per_day": mid["trades_per_day"], "mid_per_trade": mid["per_trade"]["mean"],
         "ratio_5s": blk["delayed"]["5"]["ratio_to_delay0"]}
    for v in VARS:
        m = blk["rows"].get(f"M_{v}")
        r[v] = None if m is None else {"cost_per_trade": m["cost_per_trade"] + m["fee_per_trade"], "pnl_per_trade": m["pnl_per_trade"]}
    return r


def keep_old(path):
    """An existing output is moved to <folder>/superseded/<name>_<its time> before a new one is written (nothing is
    overwritten or deleted)."""
    import datetime as dt
    if os.path.exists(path):
        d = os.path.join(os.path.dirname(path), "superseded")
        os.makedirs(d, exist_ok=True)
        stem, ext = os.path.splitext(os.path.basename(path))
        stamp = dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y%m%d_%H%M%S")
        q = os.path.join(d, f"{stem}_{stamp}{ext}")
        if os.path.exists(q):
            raise RuntimeError(f"{q} exists")
        os.replace(path, q)


R20_FILES = {"dev": ("results/dev20_summary.json", "results/delay20_summary.json"), "sealed": ("results/test20_summary.json",)}


def r20_side(part, name="W30_z2_k0"):
    """The side-by-side line of research 20 (two share classes of one issuer) from its stored results, read only:
    part "dev" = its development run (dev20_summary.json, delay20_summary.json), "sealed" = its sealed test
    (test20_summary.json; quote sizes there are not divided by 100 from 2025-11-03, so its L3D1 cost is too low from
    that day on, synthesis/size_unit_audit.md)."""
    import json
    if part == "dev":
        dev = json.load(open(os.path.join(R20, R20_FILES["dev"][0])))
        dl = json.load(open(os.path.join(R20, R20_FILES["dev"][1])))
        mid = next(r for r in dev["mid_rows"] if r["name"] == f"{name}_mid")
        rows = {v: next(r for r in dev["rows"] if r["name"] == f"{name}_M" and r["variant"] == v) for v in VARS}
        ratio = dl["settings"][name]["rows"]["5"]["ratio_to_delay0"]
    else:
        t = json.load(open(os.path.join(R20, R20_FILES["sealed"][0])))
        if t["mid_delayed"]["signal"] != name:
            raise ValueError("research 20's delayed-mid table is for another setting")
        mid = t["settings"][name]["rows"]["mid"]
        rows = {v: t["settings"][name]["rows"][f"M_{v}"] for v in VARS}
        ratio = t["mid_delayed"]["rows"]["5"]["ratio_to_delay0"]
    r = {"label": "R20", "trades": mid["trades"], "trades_per_day": mid["trades_per_day"], "mid_per_trade": mid["per_trade"]["mean"],
         "ratio_5s": ratio}
    for v in VARS:
        r[v] = {"cost_per_trade": rows[v]["cost_per_trade"] + rows[v]["fee_per_trade"], "pnl_per_trade": rows[v]["pnl_per_trade"]}
    return r


# ---------------------------------------------------------------- comparing stored and rebuilt tables (regression, rehearsal)
NUMERIC = ("mean_daily_usd", "ci_lo", "ci_hi", "p_one_sided", "lower_2_5", "sharpe", "trades", "trades_per_day", "win_rate", "mean_hold_s",
           "pnl_per_trade", "mid_per_trade", "cost_per_trade", "fee_per_trade", "exit_z", "exit_time", "exit_close", "exit_last",
           "days_with_trades", "total_usd")


def diff(a, b):
    """|a - b|; 0 when both are nan, inf when only one is."""
    import math
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return 0.0 if (math.isnan(a) and math.isnan(b)) else math.inf
    return abs(a - b)


def row_diffs(new, old):
    """Absolute differences of every number of a table row (NUMERIC, by year, by pair, the per-trade interval if stored)."""
    d = [diff(new[f], old[f]) for f in NUMERIC]
    d += [diff(new["by_year"][y], old["by_year"][y]) for y in old["by_year"]]
    d += [diff(new["by_pair"][p][k], old["by_pair"][p][k]) for p in old["by_pair"] for k in old["by_pair"][p]]
    if "per_trade" in old:
        d += [diff(new["per_trade"][k], old["per_trade"][k]) for k in ("mean", "lo", "hi")]
    if set(new["by_year"]) != set(old["by_year"]) or set(new["by_pair"]) != set(old["by_pair"]):
        d.append(float("inf"))
    return d


def delayed_diffs(new, old):
    d = [diff(new[k][f], old[k][f]) for k in old for f in old[k]] if set(new) == set(old) else [float("inf")]
    return d
