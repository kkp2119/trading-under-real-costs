#!/usr/bin/env python3
"""Research 24: the simulation and the tables shared by dev24.py, run_test24.py, regress24.py and test24.py (one code path
for the development run, the rehearsal, the sealed run and the regression on the development data of research 20 and 21).

The engines of research 17, 20 and 21 are imported unchanged (sha256 checked at import, FROZEN): signal20 (grid, z, flags),
pair20 (kernel calls, money), core20 (aggregation and table rows), stats20, core21 (diagnostics), loader21 (file checks,
quote sizes / 100 from 2025-11-03). New here:
  amount      the trades come from pair24.trades with the dollar amount per leg as an argument (10,000; 50,000 in the
              regression test)
  delays      every setting is run once per delay (0, 1, 5 seconds: pair20's delay argument) with market orders; the same
              trades are valued under L3D1 (depth penalty 1) and L0D0 (depth penalty 0)
  over        trades with at least one of their four orders above the displayed size, and the shares above it
  splits      by family, by pair type (ANSWERS A5), by year, by pair, before / from 2025-11-03
  top 5       the mean daily P&L without the 5 days with the largest daily P&L, without the 5 pairs with the largest
              total P&L, and without both (SPEC24 5; SPEC24_QUESTIONS 20)
Daily series cover every trading day of the window (a day without a trade is 0)."""
import hashlib
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for _p in (os.path.join(PROJ, "study21_more_pairs"), os.path.join(PROJ, "study20_pairs")):
    if _p not in sys.path:
        sys.path.insert(1, _p)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import core21 as C21  # noqa: E402  (checks the research 17 and 20 files against tag code-freeze-20 at import)

# research 21 files used on the run path: sha256 of the files of tag test-run-21 (core21 equals code-freeze-21 too)
FROZEN21 = {"study21_more_pairs/core21.py": "73f92ca92285ed7f288a784bbc0ff0ef457517014da19291e4fe1f5541128695",
            "study21_more_pairs/loader21.py": "00729cb26b57abe6a97e028caba02cf5c894754f8be7bf789e5e833874d5a5a5",
            "study21_more_pairs/stats21.py": "0d156eea551a3c320fe8fda70636084d71a5fe520da00a0f8893199dec27d589"}


def frozen_differences():
    """Paths of the imported engine files (research 17, 20, 21) that differ from the sha256 of their tags."""
    return C21.frozen_differences() + [p for p, h in FROZEN21.items() if hashlib.sha256(open(os.path.join(PROJ, p), "rb").read()).hexdigest() != h]


if frozen_differences():
    raise ImportError(f"engine files differ from their tags: {frozen_differences()}")
import loader24 as L24  # noqa: E402
import pair24 as P24  # noqa: E402
import stats24 as ST24  # noqa: E402

C, PE, S, ST20 = C21.C, C21.PE, C21.S, C21.ST20
VARS = ST24.VARS
COL, NAGG, NDIAG = C.COL, C.NAGG, C21.NDIAG
DEPTH = {"L3D1": 1, "L0D0": 0}
KI = {"L3D1": 1, "L0D0": 2}                                 # where a variant sits in a core20-style kinds axis (0 is not used)
keep_old, row_diffs, diff = C21.keep_old, C21.row_diffs, C21.diff


def over_trades(tr):
    """Trades with at least one of their four orders above the displayed size (the kernel's `over` shares > 0)."""
    if not len(tr):
        return 0
    o = tr[:, PE.O0:PE.O0 + PE.NORD * PE.OCOL].reshape(len(tr), PE.NORD, PE.OCOL)
    return int((o[:, :, 4] > 0).any(axis=1).sum())


def same_timetable(mt, tr, same_amount):
    """The M trades at delay 0 against the signal-only timetable: decision bars, side, exit reason and decision mids (and
    the shares when the amount is research 20's 50,000 dollars)."""
    cols = [0, 1, 2, 3, 4, 5] if same_amount else [0, 1, 2, 5]
    return bool(np.array_equal(mt[:, cols], tr[:, cols]) and np.array_equal(mt[:, 6:8], tr[:, PE.MID0:PE.MID0 + 2])
                and np.array_equal(mt[:, 8:10], tr[:, PE.MID0 + 2:PE.MID0 + 4]))


def pairday(a, b, cb, cfgs, amount, delays, keep=(), diag=False):
    """One pair-day from its two bar dicts: aggregates [settings, delays, 3, NAGG] (axis 2: unused copy, L3D1, L0D0, the
    layout of core20's kinds), trades with an order above the displayed size [settings, delays], the number of settings
    whose delay-0 trades differ from the signal-only timetable, kept trade rows {(setting index, delay): rows}, valid and
    total grid points, the left-out flag (a bar without a quote state after a leg's first state) and the diagnostics."""
    out = np.zeros((len(cfgs), len(delays), 3, NAGG))
    X = np.zeros((len(cfgs), len(delays)))
    Gd = np.zeros((len(cfgs), NDIAG))
    kept = {k: np.zeros((0, PE.NTR), dtype=np.int64) for k in keep}
    if not (C.state_ok(a) and C.state_ok(b)):
        return out, X, 0, kept, 0, len(S.grid(cb)), 1, Gd
    B = S.base(a, b, cb)
    XA, XB = PE.stack(a), PE.stack(b)
    mismatch = 0
    for ci, (W, zi, ka) in enumerate(cfgs):
        side, exz = S.flags(B, W, zi, ka, cb)
        for di, d in enumerate(delays):
            tr = P24.trades(amount, PE.M, 3, d, B, side, exz, XA, XB, cb)
            if d == 0:
                mismatch += int(not same_timetable(S.mid_trades(B, side, exz, cb), tr, amount == S.NOTIONAL))
            out[ci, di, 1], out[ci, di, 2] = C._agg(tr, 1, PE.M), C._agg(tr, 0, PE.M)
            out[ci, di, 0] = out[ci, di, 1]
            X[ci, di] = over_trades(tr)
            if (ci, d) in kept:
                kept[(ci, d)] = tr
        if diag:
            Gd[ci] = C21.diagnose(B, W, zi, ka, cb)
    return out, X, mismatch, kept, int(B["valid"].sum()), len(B["g"]), 0, Gd


def work(args):
    """One pair-day read through loader24 (see pairday)."""
    t, root, allow_sealed, cfgs, amount, delays, keep, diag = args
    a, b = L24.load(root, t["row_a"], allow_sealed), L24.load(root, t["row_b"], allow_sealed)
    return pairday(a, b, t["close_bar"], cfgs, amount, delays, keep, diag)


def simulate(tasks, root, allow_sealed, cfgs, amount=ST24.AMOUNT, delays=ST24.DELAYS, keep=(), workers=4, diag=False):
    """(A [pair-days, settings, delays, 3, NAGG], X [pair-days, settings, delays], info, extra). extra: "kept" {(setting
    index, delay): M trade rows with the pair-day index in column 0}, "diag" [pair-days, settings, NDIAG]. workers = 0 runs
    in this process (synthetic tests)."""
    args = [(t, root, allow_sealed, list(cfgs), amount, tuple(delays), tuple(keep), diag) for t in tasks]
    if workers and args:
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(work, args, chunksize=8)
    else:
        res = [work(x) for x in args]
    nc, nd = len(cfgs), len(delays)
    A = np.stack([r[0] for r in res]) if res else np.zeros((0, nc, nd, 3, NAGG))
    X = np.stack([r[1] for r in res]) if res else np.zeros((0, nc, nd))
    extra = {"kept": {k: (np.concatenate([np.column_stack([np.full(len(r[3][k]), i, dtype=np.int64), r[3][k]]) for i, r in enumerate(res)])
                          if res else np.zeros((0, PE.NTR + 1), dtype=np.int64)) for k in keep},
             "diag": np.stack([r[7] for r in res]) if res else np.zeros((0, nc, NDIAG))}
    info = {"pair_days": len(tasks), "amount_usd": amount, "delays": list(delays),
            "settings_with_m_trades_unlike_the_signal_timetable": int(sum(r[2] for r in res)),
            "valid_grid_points": int(sum(r[4] for r in res)), "grid_points": int(sum(r[5] for r in res)),
            "pair_days_left_out_no_state_after_first_state": int(sum(r[6] for r in res))}
    return A, X, info, extra


# ---------------------------------------------------------------- tables
def run_name(delay, var):
    return f"d{delay}_{var}"


def series(A, M, ci, di, var, n_days, col="pnl"):
    """Daily sum of a column of one run over every day of the window."""
    return C.daily(A[:, :, di], M, ci, KI[var], col, n_days)


def run_row(A, X, M, ci, di, var, n_days, idx, label):
    """core20.row of one run (setting ci, delay index di, variant) plus the trades with an order above the displayed size."""
    r = C.row(A[:, :, di], M, ci, KI[var], n_days, idx, label)
    r["over_trades"] = int(X[:, ci, di].sum())
    r["over_trade_frac"] = float(r["over_trades"] / r["trades"]) if r["trades"] else float("nan")
    return r


def brief(r):
    """The numbers of a row that a split table shows."""
    return {k: r[k] for k in ("mean_daily_usd", "ci_lo", "ci_hi", "p_one_sided", "trades", "trades_per_day", "win_rate", "mean_hold_s",
                              "pnl_per_trade", "mid_per_trade", "cost_per_trade", "fee_per_trade", "total_usd", "over_trades", "over_trade_frac",
                              "over_share_frac")}


def subset(tasks, key, value):
    return np.array([i for i, t in enumerate(tasks) if t[key] == value], dtype=np.int64)


def split_rows(A, X, tasks, days, ci, delays, idx, key, values):
    """{value: {run: brief row}} over the pair-days whose task[key] == value (the daily series still cover every day);
    a value without a pair-day is left out."""
    out = {}
    for v in values:
        sel = subset(tasks, key, v)
        if not len(sel):
            continue
        M = C.meta([tasks[i] for i in sel], days)
        out[v] = {"pair_days": int(len(sel)), "pairs": len(M["names"]),
                  "runs": {run_name(d, var): brief(run_row(A[sel], X[sel], M, ci, di, var, len(days), idx, f"{v}_{run_name(d, var)}"))
                           for di, d in enumerate(delays) for var in VARS}}
    return out


def stat(x, idx):
    mean, lo, hi = ST20.mean_ci(x, idx)
    _, p, _ = ST20.p_positive(x, idx)
    return {"mean_daily_usd": mean, "ci_lo": lo, "ci_hi": hi, "p_one_sided": p, "days": int(len(x))}


def top_days(d, k=ST24.TOP):
    """Indices of the k days with the largest value (ties: the earlier day first)."""
    return np.argsort(-np.asarray(d, dtype=np.float64), kind="stable")[:k]


def without_top(A, tasks, days, ci, di, var, idx, k=ST24.TOP):
    """SPEC24 5 / SPEC24_QUESTIONS 20 for one run: the mean daily P&L (a) without the k days with the largest daily P&L
    (the days left are resampled with their own index, seed 24), (b) without the k pairs with the largest total P&L of the
    window (a pair = its two symbols, whatever the year and the order; every day stays), (c) without both: the pairs are
    taken out first, then the k largest days of what is left."""
    n_days = len(days)
    M = C.meta(tasks, days)
    d = series(A, M, ci, di, var, n_days)
    td = top_days(d, k)
    keep_d = np.ones(n_days, bool)
    keep_d[td] = False
    pnl = A[:, ci, di, KI[var], COL["pnl"]]
    keys = sorted({t["key"] for t in tasks})
    tot = {key: 0.0 for key in keys}
    for i, t in enumerate(tasks):
        tot[t["key"]] += float(pnl[i])
    top_p = sorted(keys, key=lambda q: (-tot[q], q))[:k]
    sel = np.array([i for i, t in enumerate(tasks) if t["key"] not in top_p], dtype=np.int64)
    d2 = ST20.daily(M["day"][sel], pnl[sel], n_days)
    td2 = top_days(d2, k)
    keep2 = np.ones(n_days, bool)
    keep2[td2] = False
    return {"all": stat(d, idx), "total_usd": float(d.sum()),
            "without_top_days": {**stat(d[keep_d], ST24.boot_index(int(keep_d.sum()))), "days_removed": [[days[i], float(d[i])] for i in td],
                                 "share_of_total": float(d[td].sum() / d.sum()) if d.sum() else float("nan")},
            "without_top_pairs": {**stat(d2, idx), "pairs_removed": [[q, tot[q]] for q in top_p],
                                  "share_of_total": float(sum(tot[q] for q in top_p) / d.sum()) if d.sum() else float("nan")},
            "without_both": {**stat(d2[keep2], ST24.boot_index(int(keep2.sum()))), "days_removed": [[days[i], float(d2[i])] for i in td2]}}


def date_split(d, n, days, cut=L24.SIZE_FROM):
    """Mean daily value with the day-block 95% interval and the trades, for the days before `cut` and from `cut` on; each
    part has its own bootstrap index (its own length, seed 24). None for a part without days."""
    dd = np.array(days)
    out = {}
    for key, sel in (("before", dd < cut), ("from", dd >= cut)):
        T = int(sel.sum())
        if T == 0:
            out[key] = None
            continue
        out[key] = {**stat(d[sel], ST24.boot_index(T)), "first_day": str(dd[sel][0]), "last_day": str(dd[sel][-1]), "trades": int(n[sel].sum())}
    return out


def by_key(A, tasks, ci, di, var):
    """Per pair (its two symbols, whatever the year and the order): pair-days, trades and total P&L of one run."""
    out = {}
    for i, t in enumerate(tasks):
        r = out.setdefault(t["key"], {"family": t["family"], "type": t["type"], "pair_days": 0, "trades": 0, "total_usd": 0.0, "years": []})
        r["pair_days"] += 1
        r["trades"] += int(A[i, ci, di, KI[var], COL["n"]])
        r["total_usd"] += float(A[i, ci, di, KI[var], COL["pnl"]])
        if t["year"] not in r["years"]:
            r["years"].append(t["year"])
    return out


def setting_tables(A, X, tasks, days, ci, delays, idx, name):
    """Every table of one signal setting: the row of each run (delay x variant), the splits by family and by pair type,
    the top-5 table of each run, the split at 2025-11-03 of each run and the per-pair totals of the main run."""
    n_days = len(days)
    M = C.meta(tasks, days)
    runs, top, split = {}, {}, {}
    for di, d in enumerate(delays):
        for var in VARS:
            rn = run_name(d, var)
            runs[rn] = run_row(A, X, M, ci, di, var, n_days, idx, f"{name}_{rn}")
            top[rn] = without_top(A, tasks, days, ci, di, var, idx)
            split[rn] = date_split(series(A, M, ci, di, var, n_days), series(A, M, ci, di, var, n_days, "n"), days)
    main = list(delays).index(ST24.DELAY_MAIN) if ST24.DELAY_MAIN in delays else 0
    return {"runs": runs, "by_family": split_rows(A, X, tasks, days, ci, delays, idx, "family", ST24.FAMILIES),
            "by_type": split_rows(A, X, tasks, days, ci, delays, idx, "type", ST24.TYPES), "without_top": top, "date_split": split,
            "by_pair_key": by_key(A, tasks, ci, main, ST24.VAR_MAIN)}


def daily_cols(delays):
    return ("date", "family", "type", "pair", "name", "setting") + tuple(c for d in delays for c in (f"trades_d{d}", f"d{d}_L3D1", f"d{d}_L0D0"))


def pair_daily(A, tasks, ci, setting, delays):
    """One row per pair-day for one setting (daily_cols): per delay the trades and the dollars under L3D1 and L0D0."""
    rows = []
    for i, t in enumerate(tasks):
        r = [t["date"], t["family"], t["type"], t["pair"], t["name"], setting]
        for di in range(len(delays)):
            r += [int(A[i, ci, di, 1, COL["n"]]), A[i, ci, di, 1, COL["pnl"]], A[i, ci, di, 2, COL["pnl"]]]
        rows.append(tuple(r))
    return rows


def diag_shares(Gd, tasks, days, cfgs):
    """core21.diag_shares for all pair-days (sigma at the 1 bp floor per W, |z| >= z_in points kept out by kappa per setting)."""
    return C21.diag_shares(Gd, C.meta(tasks, days), cfgs)


# ---------------------------------------------------------------- comparing two sets of tables (rehearsal)
def deep_diff(x, y):
    """(largest absolute difference, numbers compared) over everything two nested tables both hold; a key or a length that
    only one of them has, or a number against a non-number, counts as an infinite difference. Two nan are equal."""
    worst, n = 0.0, 0
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y):
            return float("inf"), 0
        for k in x:
            w, m = deep_diff(x[k], y[k])
            worst, n = max(worst, w), n + m
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        if len(x) != len(y):
            return float("inf"), 0
        for u, v in zip(x, y):
            w, m = deep_diff(u, v)
            worst, n = max(worst, w), n + m
    elif isinstance(x, bool) or isinstance(y, bool) or isinstance(x, str) or isinstance(y, str) or x is None or y is None:
        worst = 0.0 if x == y else float("inf")
    else:
        worst, n = diff(x, y), 1
    return worst, n


def jdefault(o):
    """json default: numpy scalars and arrays as plain numbers and lists; anything else is an error."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o).__name__}")
