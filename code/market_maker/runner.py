"""Shared executor (ENGINE_DESIGN_E1 3): one symbol-day is read once, its features computed once, then every
(setting, model) runs. Rows are sorted by key, so the result does not depend on the number of workers (M14).
Record mode (E1-66, E1-68): for the named settings, model Q, the fill rows the kernel already returns are kept (one row
per fill) together with the inventory path every 60 s; recording only copies outputs, so the run table is unchanged."""
import multiprocessing as mp

import numpy as np
import pandas as pd

import accounting
import features
import placebo as plc
import simulator
from config import (N_STATS, QPATH_BARS, S_MAXQ, S_MKT_OVER, S_MKT_STALEQ, S_NDEC, S_NDEC_QB, S_NINT_B, S_NINT_S,
                    S_NINTF_B, S_NINTF_S, S_NMKT, S_NSTALE, S_QDF, S_SKIP, S_SUMQ)

STAT_NAMES = {S_SKIP: "skip", S_NDEC: "n_dec", S_NDEC_QB: "n_dec_qb", S_NINT_B: "n_int_b", S_NINT_S: "n_int_s",
              S_NINTF_B: "n_intf_b", S_NINTF_S: "n_intf_s", S_NSTALE: "n_stale", S_QDF: "q_df", S_MAXQ: "max_q",
              S_SUMQ: "sum_q", S_MKT_OVER: "mkt_over_disp", S_MKT_STALEQ: "mkt_stale_quote", S_NMKT: "n_mkt"}
assert len(STAT_NAMES) == N_STATS
RECORD_MODEL = "Q"


def q_path(fills):
    """Inventory held at the end of each QPATH_BARS bar, from the fill rows (signed shares filled up to that bar)."""
    if len(fills) == 0:
        return np.zeros(len(QPATH_BARS), dtype=np.int64)
    o = np.argsort(fills[:, 0], kind="stable")
    u, signed = fills[o, 0], np.cumsum(fills[o, 1] * fills[o, 3])
    k = np.searchsorted(u, QPATH_BARS, side="right")
    return np.where(k > 0, signed[np.maximum(k - 1, 0)], 0).astype(np.int64)


def run_day(b, close_bar, gate, settings, models, nesting_for=(), record=()):
    """Rows for one symbol-day (+ day stats, + {setting: (fills, q path)} for the settings in `record`, model Q).
    settings: [(name, Config)]; nesting_for: setting names whose Q orders are replayed through all models (E1-60)."""
    f = features.compute(b)
    ds = features.day_stats(b, close_bar)
    rows, rec = [], {}
    for name, cfg in settings:
        d0, df = cfg.d_bounds(close_bar)
        for m in models:
            fills, orders, st = simulator.simulate(b, f, cfg.kernel_params(m, close_bar, gate))
            row = {"setting": name, "model": m, **{STAT_NAMES[i]: int(st[i]) for i in range(N_STATS)},
                   **accounting.account(fills, f["mid2"], f["mid_ok"], close_bar, d0, df),
                   "mkt_vol": ds["mkt_vol"], "ja_skip": int(ds["last_trade_bar"] < df), "nested": -1}
            if m == "Q" and name in nesting_for and len(orders):
                row["nested"] = int(simulator.nested(simulator.replay_all(b, orders)))
            if m == RECORD_MODEL and name in record:
                rec[name] = (fills.copy(), q_path(fills))
            rows.append(row)
    return rows, ds, rec


def _work(args):
    split, sym, day, close_bar, gate, settings, models, nesting_for, record = args
    rows, ds, rec = run_day(split.load(sym, day), close_bar, gate, settings, models, nesting_for, record)
    for r in rows:
        r["sym"], r["date"] = sym, day
    return rows, {"sym": sym, "date": day, **ds}, (sym, day, rec)


def run_tasks_record(split, cal, tasks, gates, settings, models, n_workers=1, nesting_for=(), record=()):
    """(runs, day stats, records). records: {setting: {"fills": DataFrame one row per fill, "qpath": (index DataFrame,
    int64 array tasks x len(QPATH_BARS))}}, all sorted by (sym, date)."""
    jobs = [(split, s, d, cal.close[d], gates[(s, d)], settings, models, tuple(nesting_for), tuple(record))
            for s, d in tasks]
    if n_workers > 1:
        with mp.get_context("spawn").Pool(n_workers) as pool:
            res = pool.map(_work, jobs, chunksize=max(1, len(jobs) // (8 * n_workers)))
    else:
        res = [_work(j) for j in jobs]
    runs = pd.DataFrame([r for rows, _, _ in res for r in rows])
    days = pd.DataFrame([d for _, d, _ in res])
    if len(runs):
        runs = runs.sort_values(["setting", "model", "sym", "date"], kind="stable").reset_index(drop=True)
    if len(days):
        days = days.sort_values(["sym", "date"], kind="stable").reset_index(drop=True)
    records = {}
    recs = sorted((r for _, _, r in res), key=lambda x: (x[0], x[1]))
    for name in record:
        fl, idx, qp = [], [], []
        for sym, day, rec in recs:
            if name not in rec:
                continue
            fills, path = rec[name]
            if len(fills):
                fl.append(pd.DataFrame(fills, columns=["u", "side", "kind", "qty", "price", "ref_mid2", "dec"]).assign(
                    sym=sym, date=day))
            idx.append((sym, day))
            qp.append(path)
        cols = ["sym", "date", "u", "side", "kind", "qty", "price", "ref_mid2", "dec"]
        fills_df = pd.concat(fl, ignore_index=True)[cols] if fl else pd.DataFrame(columns=cols)
        records[name] = {"fills": fills_df, "qpath": (pd.DataFrame(idx, columns=["sym", "date"]),
                                                      np.array(qp, dtype=np.int64).reshape(len(qp), len(QPATH_BARS)))}
    return runs, days, records


def run_tasks(split, cal, tasks, gates, settings, models, n_workers=1, nesting_for=()):
    """tasks: [(sym, day)]; gates: {(sym, day): bool}. Returns (runs DataFrame, day-stats DataFrame), sorted by key."""
    runs, days, _ = run_tasks_record(split, cal, tasks, gates, settings, models, n_workers, nesting_for)
    return runs, days


def run_placebo_day(b, close_bar, gate, cfg, dim, sym, day, rs):
    """B5 rows for one symbol-day: per permutation r, net R and limit shares (model Q)."""
    f = features.compute(b)
    d0, df = cfg.d_bounds(close_bar)
    out = []
    for r in rs:
        fp = plc.permuted(f, dim, r, sym, day, d0, df)
        fills, _, _ = simulator.simulate(b, fp, cfg.kernel_params("Q", close_bar, gate))
        acc = accounting.account(fills, f["mid2"], f["mid_ok"], close_bar, d0, df)
        out.append({"r": r, "net_R": int(accounting.net(acc, "R")), "lim_sh": acc["sh_lim_buy"] + acc["sh_lim_sell"],
                    "sym": sym, "date": day})
    return out
