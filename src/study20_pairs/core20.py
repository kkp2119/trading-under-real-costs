#!/usr/bin/env python3
"""Research 20: the simulation and the tables shared by dev20.py and run_test20.py (one code path for the development run,
the rehearsal and the sealed run).

simulate()   every pair-day of a plan through signal20 and pair20 for a list of signal settings. Per pair-day and setting
             five runs are aggregated: mid (signal-only timetable at the mid, H1), M_L3D1, M_L0D0 (the same M trades, depth
             penalty 1 / 0), P_L3D1 (fill rule 3, depth 1), P_L0D0 (fill rule 0, depth 0). Only sums per pair-day are kept
             (NAGG columns); trade rows are returned for the settings in `keep`; with mid_delays the mid P&L of the
             signal-only trades priced d seconds late is summed too (SPEC20_ADDENDUM B2).
rows()       per run: mean daily P&L with the day-block 95% interval, Sharpe, trades, win rate, holding time, the split
             pnl = mid - cost - fee per trade, P fill statistics, by year and by pair.
Daily series cover every trading day of the window; a pair-day needs both legs (loader20.plan)."""
import multiprocessing as mp

import numpy as np

import loader20 as L
import pair20 as PE
import signal20 as S
import stats20 as ST

KINDS = ("mid", "M_L3D1", "M_L0D0", "P_L3D1", "P_L0D0")
AGG = ("n", "pnl", "mid", "cost", "fee", "wins", "hold", "r_z", "r_time", "r_close", "r_last", "f_pairs", "f_one", "f_orders",
       "f_dl_orders", "f_shares", "f_dl_shares", "mkt_shares", "lim_shares", "over_shares")
NAGG = len(AGG)
COL = {k: i for i, k in enumerate(AGG)}


def _agg(tr, depth, method):
    v = np.zeros(NAGG)
    if not len(tr):
        return v
    m = PE.money(tr, depth)
    o = tr[:, PE.O0:PE.O0 + PE.NORD * PE.OCOL].reshape(len(tr), PE.NORD, PE.OCOL)
    v[:11] = (len(tr), m["pnl"].sum(), m["mid"].sum(), m["cost"].sum(), m["fee"].sum(), (m["pnl"] > 0).sum(), (tr[:, 1] - tr[:, 0]).sum(),
              (tr[:, 5] == S.R_Z).sum(), (tr[:, 5] == S.R_TIME).sum(), (tr[:, 5] == S.R_CLOSE).sum(), (tr[:, 5] == S.R_LAST).sum())
    v[11:17] = PE.fills(tr, method)
    v[17:20] = o[:, :, 2].sum(), o[:, :, 0].sum(), o[:, :, 4].sum()
    return v


def _agg_mid(mt):
    v = np.zeros(NAGG)
    if not len(mt):
        return v
    p = S.mid_pnl(mt)
    v[:11] = (len(mt), p.sum(), p.sum(), 0.0, 0.0, (p > 0).sum(), (mt[:, 1] - mt[:, 0]).sum(), (mt[:, 5] == S.R_Z).sum(),
              (mt[:, 5] == S.R_TIME).sum(), (mt[:, 5] == S.R_CLOSE).sum(), (mt[:, 5] == S.R_LAST).sum())
    return v


def state_ok(x):
    """bars1s/bars.py carries a quote state forward once it exists, so no bar may lack a state after the first one."""
    st = (x["bid"] > 0) & (x["ask"] > 0)
    return bool(st[int(np.argmax(st)):].all()) if st.any() else True


def pairday_all(a, b, cb, cfgs, delay=0, keep=(), mid_delays=()):
    """One pair-day from its two bar dicts: aggregates [settings, kinds, NAGG], the number of settings whose M trades differ
    from the signal-only timetable (0 expected, delay 0 only), the kept trade rows {setting index: {run: rows}}, valid and
    total grid points, excluded flag, delayed-mid sums [settings, delays, (trades, dollars, wins)] (ADDENDUM B2, delay 0 only).
    A pair-day with a bar without a quote state after a leg's first state contradicts the bar definition: it is left out
    whole and counted (never met in the development bars)."""
    out = np.zeros((len(cfgs), len(KINDS), NAGG))
    D = np.zeros((len(cfgs), len(mid_delays), 3))
    kept = {ci: {k: np.zeros((0, PE.NTR), dtype=np.int64) for k in ("M", "P_L3", "P_L0")} for ci in keep}
    if not (state_ok(a) and state_ok(b)):
        return out, 0, kept, 0, len(S.grid(cb)), 1, D
    B = S.base(a, b, cb)
    XA, XB = PE.stack(a), PE.stack(b)
    bars = S.bar_base(a, b, cb) if (mid_delays and delay == 0) else None
    mismatch = 0
    for ci, (W, zi, ka) in enumerate(cfgs):
        side, exz = S.flags(B, W, zi, ka, cb)
        trm = PE.trades(PE.M, 3, delay, B, side, exz, XA, XB, cb)
        tp3 = PE.trades(PE.P, 3, delay, B, side, exz, XA, XB, cb)
        tp0 = PE.trades(PE.P, 0, delay, B, side, exz, XA, XB, cb)
        if delay == 0:
            mt = S.mid_trades(B, side, exz, cb)
            mismatch += int(not (np.array_equal(mt[:, :6], trm[:, :6]) and np.array_equal(mt[:, 6:8], trm[:, PE.MID0:PE.MID0 + 2])
                                 and np.array_equal(mt[:, 8:10], trm[:, PE.MID0 + 2:PE.MID0 + 4])))
            out[ci, 0] = _agg_mid(mt)
            for di, d in enumerate(mid_delays):
                pd_ = S.mid_pnl_delayed(mt, *bars, d)
                D[ci, di] = len(pd_), pd_.sum(), (pd_ > 0).sum()
        out[ci, 1], out[ci, 2] = _agg(trm, 1, PE.M), _agg(trm, 0, PE.M)
        out[ci, 3], out[ci, 4] = _agg(tp3, 1, PE.P), _agg(tp0, 0, PE.P)
        if ci in keep:
            kept[ci] = {"M": trm, "P_L3": tp3, "P_L0": tp0}
    return out, mismatch, kept, int(B["valid"].sum()), len(B["g"]), 0, D


def work(args):
    """One pair-day read through loader20 (see pairday_all)."""
    t, root, allow_sealed, cfgs, delay, keep, mid_delays = args
    a, b = L.load(root, t["row_a"], allow_sealed), L.load(root, t["row_b"], allow_sealed)
    return pairday_all(a, b, t["close_bar"], cfgs, delay, keep, mid_delays)


def simulate(tasks, root, allow_sealed, cfgs, delay=0, keep=(), workers=8, mid_delays=()):
    """(AGG array [pair-days, settings, kinds, NAGG], info, kept trades per setting with the pair-day index in column 0).
    With mid_delays the delayed-mid sums [pair-days, settings, delays, 3] are returned as kept["mid_delayed"]."""
    args = [(t, root, allow_sealed, list(cfgs), delay, tuple(keep), tuple(mid_delays)) for t in tasks]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(work, args, chunksize=8)
    A = np.stack([r[0] for r in res]) if res else np.zeros((0, len(cfgs), len(KINDS), NAGG))
    kept = {}
    for ci in keep:
        kept[ci] = {k: np.concatenate([np.column_stack([np.full(len(r[2][ci][k]), i, dtype=np.int64), r[2][ci][k]]) for i, r in enumerate(res)])
                    for k in ("M", "P_L3", "P_L0")}
    if mid_delays:
        kept["mid_delayed"] = np.stack([r[6] for r in res])
    info = {"pair_days": len(tasks), "settings_with_m_trades_unlike_the_signal_timetable": int(sum(r[1] for r in res)),
            "valid_grid_points": int(sum(r[3] for r in res)), "grid_points": int(sum(r[4] for r in res)),
            "pair_days_left_out_no_state_after_first_state": int(sum(r[5] for r in res))}
    return A, info, kept


def meta(tasks, days):
    """Per pair-day: day code, year, pair name; and the pair names in order of first appearance."""
    pos = {d: i for i, d in enumerate(days)}
    names = []
    for t in tasks:
        if t["name"] not in names:
            names.append(t["name"])
    return {"day": np.array([pos[t["date"]] for t in tasks], dtype=np.int64), "year": np.array([t["year"] for t in tasks], dtype=np.int64),
            "pair": np.array([names.index(t["name"]) for t in tasks], dtype=np.int64), "names": names,
            "day_year": np.array([int(d[:4]) for d in days], dtype=np.int64)}


def daily(A, M, ci, ki, col, n_days):
    return ST.daily(M["day"], A[:, ci, ki, COL[col]], n_days)


def row(A, M, ci, ki, n_days, idx, label):
    """The table row of one run (setting ci, kind ki)."""
    x = A[:, ci, ki]
    tot = x.sum(axis=0)
    n = tot[COL["n"]]
    d = daily(A, M, ci, ki, "pnl", n_days)
    mean, lo, hi = ST.mean_ci(d, idx)
    _, p, lb = ST.p_positive(d, idx)
    per = (lambda c: float(tot[COL[c]] / n) if n else float("nan"))                      # noqa: E731
    r = {"name": label, "kind": KINDS[ki], "mean_daily_usd": mean, "ci_lo": lo, "ci_hi": hi, "p_one_sided": p, "lower_2_5": lb,
         "sharpe": ST.sharpe(d), "trades": int(n), "trades_per_day": float(n / n_days), "win_rate": per("wins"),
         "mean_hold_s": per("hold"), "pnl_per_trade": per("pnl"), "mid_per_trade": per("mid"), "cost_per_trade": per("cost"),
         "fee_per_trade": per("fee"), "exit_z": per("r_z"), "exit_time": per("r_time"), "exit_close": per("r_close"),
         "exit_last": int(tot[COL["r_last"]]), "days_with_trades": int((ST.daily(M["day"], x[:, COL["n"]], n_days) > 0).sum()),
         "total_usd": float(tot[COL["pnl"]])}
    sh = tot[COL["mkt_shares"]] + tot[COL["lim_shares"]]
    r["limit_share_frac"] = float(tot[COL["lim_shares"]] / sh) if sh else float("nan")
    r["over_share_frac"] = float(tot[COL["over_shares"]] / sh) if sh else float("nan")
    fp, fo, fs = tot[COL["f_pairs"]], tot[COL["f_orders"]], tot[COL["f_shares"]]
    r["p_one_leg_limit_frac"] = float(tot[COL["f_one"]] / fp) if fp else float("nan")
    r["p_deadline_order_frac"] = float(tot[COL["f_dl_orders"]] / fo) if fo else float("nan")
    r["p_deadline_share_frac"] = float(tot[COL["f_dl_shares"]] / fs) if fs else float("nan")
    r["by_year"] = {str(y): float(d[M["day_year"] == y].mean()) for y in sorted(set(M["day_year"].tolist()))}
    r["by_pair"] = {}
    for pi, nm in enumerate(M["names"]):
        sel = M["pair"] == pi
        npd = int(sel.sum())
        r["by_pair"][nm] = {"pair_days": npd, "per_pair_day_usd": float(x[sel, COL["pnl"]].sum() / npd), "trades": int(x[sel, COL["n"]].sum()),
                            "total_usd": float(x[sel, COL["pnl"]].sum())}
    return r


def per_trade(A, M, ci, ki, col, n_days, idx):
    """Per-trade mean of a column with its day-block 95% interval (ratio of daily sums to daily trade counts)."""
    return ST.ratio_ci(daily(A, M, ci, ki, col, n_days), daily(A, M, ci, ki, "n", n_days), idx)


def delayed_rows(A, D, M, ci, delays, n_days, idx):
    """SPEC20_ADDENDUM B2 for one setting: per delay (0 = the H1 series) trades, dollars per trade, the mean daily sum with
    its day-block 95% interval, the ratio to delay 0 and the share of trades above 0."""
    base = daily(A, M, ci, 0, "pnl", n_days)
    n0 = A[:, ci, 0, COL["n"]].sum()
    m0, lo0, hi0 = ST.mean_ci(base, idx)
    rows = {"0": {"trades": int(n0), "per_trade_usd": float(base.sum() / n0) if n0 else float("nan"), "mean_daily_usd": m0, "ci_lo": lo0,
                  "ci_hi": hi0, "ratio_to_delay0": 1.0, "win_rate": float(A[:, ci, 0, COL["wins"]].sum() / n0) if n0 else float("nan")}}
    for di, d in enumerate(delays):
        x = ST.daily(M["day"], D[:, ci, di, 1], n_days)
        n = D[:, ci, di, 0].sum()
        m, lo, hi = ST.mean_ci(x, idx)
        rows[str(d)] = {"trades": int(n), "per_trade_usd": float(x.sum() / n) if n else float("nan"), "mean_daily_usd": m, "ci_lo": lo,
                        "ci_hi": hi, "ratio_to_delay0": float(m / m0) if m0 else float("nan"),
                        "win_rate": float(D[:, ci, di, 2].sum() / n) if n else float("nan")}
    return rows
