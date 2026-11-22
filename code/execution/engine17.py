#!/usr/bin/env python3
"""Research 17 execution engine (SPEC17 2-3, SPEC17_ANSWERS 5-7 and 9), shared by test_engine17.py, dev17.py and run_test17.py.
Copied code (the originals are not changed):
  research15 exec15  TICK, STEP, FEE_MKT, start_grid, parents (default_rng(15 + YYYYMMDD), ANSWERS 5)
  research15 xy15    quote_ok          research15 fillmodel15  jit, fill_buy, fill_sell (L0, kind 2 = Q)
  research16 qty16   fill_buy_l3, fill_sell_l3, exec_q (L2 dropped: not used here)
  research16 robust16 costs, boot_index, diff_ci          research16 decomp16 mids, ret_std, vol_pre, tercile, adv_prior
New code: the stop-loss inside the kernel (C4, C5), the candidate routing C0-C5 and the one-sided bootstrap / Holm helpers.
  Stop-loss (ANSWERS 7): at the 5-second decision bars of a posting order, after the deadline and E1 checks, when the mid
  has moved against the order by at least STOP = 3 cents from the arrival mid (buy: up, sell: down), the remaining shares
  go to market at the decision bar's end quote, as E*'s own market orders (same price, same D1 count). The test is in
  integers on bid + ask (2 x mid); a decision bar with bid or ask <= 0 is not tested and counted (column 8).
  Routing (SPEC17 3): parents run independently, so a routed candidate is the per-parent choice between two runs.
    C0 E*   C1 S2i   C2 vol_pre < 4.68 bp -> S2i, else (undefined included, ANSWERS 9) E*
    C3 arrival spread <= 4 cents -> S2i, else E*   C4 E* + stop   C5 C2 with the stop on either branch"""
import math

import numpy as np

try:
    from numba import njit as _njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

# ---------------------------------------------------------------- constants (exec15, robust16, SPEC17)
TICK, N_PARENT, FIRST_START, STEP = 100, 20, 299, 5     # prices in 1e-4 dollars, 1 tick = 1 cent
FEE_MKT = 0.3                                            # cents per share, cost scenario R
QTY, HOR = 1000, 300
STOP = 3 * TICK                                          # C4 / C5: 3 cents against the arrival mid
VOL_CUT = 4.68                                           # C2: vol_pre < 4.68 bp -> S2i (ANSWERS 6)
SPREAD_CUT = 4 * TICK                                    # C3: arrival spread <= 4 cents -> S2i
E1_K = 2                                                 # E* = E12|E2off|E3off|E4off (r2_15), e1_k = 2
LRULES = (0, 3)
VARIANTS = (("L3D1", 3, 1), ("L0D0", 0, 0))              # (label, fill rule, depth); L3D1 is the primary variant
RUNS = {"S0": (0, 0, -1, 0), "S1": (1, 0, -1, 0), "S2i": (2, 1, -1, 0), "Estar": (2, 1, E1_K, 0),
        "S2i_stop": (2, 1, -1, STOP), "Estar_stop": (2, 1, E1_K, STOP)}          # method, improve, e1_k, stop
RUN_NAMES = tuple(RUNS)
CANDS = ("C0", "C1", "C2", "C3", "C4", "C5")
RULE_COUNT = {"C0": 1, "C1": 0, "C2": 2, "C3": 2, "C4": 2, "C5": 3}                # tie-break (SPEC17_QUESTIONS 16)
TIE = 0.02                                                                       # cents
NCOL = 9            # lim_sh, lim_not, mkt_sh, mkt_not, over, done (s), deadline sh, stop sh, stop tests skipped
PRE_MIN, PRE_N, PRE_STEP = 10, 30, 60                    # vol_pre: >= 10 one-minute returns out of up to 30 (16-E)
ADV_WIN, ADV_MIN = 20, 5
CUT_VOL = (4.676959528125974, 8.67061909828224)          # 16-E development tercile cut-offs (decomp16_result.json)
CUT_ADV = (484312.9, 801015.7)
TERC = ("low", "mid", "high")
BOOT_B, BOOT_BLOCK, BOOT_SEED = 10_000, 21, 17


def jit(f):
    """fillmodel15.jit: numba-compiled f when numba is available (f.py_func is the plain function either way)."""
    if HAVE_NUMBA:
        return _njit(cache=False)(f)
    f.py_func = f
    return f


# ---------------------------------------------------------------- parents and quotes (exec15, xy15)
def start_grid(close_bar, horizon=HOR):
    """Start bars u = 299 (09:35:00 end), 304, ..., close - 61 - 5 - horizon (15:53:55 end on a full day)."""
    return np.arange(FIRST_START, close_bar - 61 - STEP - horizon + 1, STEP, dtype=np.int64)


def parents(day, close_bar):
    """(direction, start bar) of the 20 parent orders of a date (same for every symbol): default_rng(15 + YYYYMMDD)."""
    rng = np.random.default_rng(15 + int(day))
    dirs = rng.permutation(np.array([1] * (N_PARENT // 2) + [-1] * (N_PARENT // 2), dtype=np.int64))
    starts = rng.choice(start_grid(close_bar), N_PARENT, replace=False).astype(np.int64)
    return dirs, starts


def quote_ok(b):
    return (b["bid"] > 0) & (b["ask"] > 0) & (b["ask"] >= b["bid"]) & (b["bsz"] > 0) & (b["asz"] > 0)


# ---------------------------------------------------------------- fill rules (fillmodel15 L0, qty16 L3)
@jit
def fill_buy(kind, price, rem, a4, u, ask_min, bid_max, bid_min, px_min, vol_bid, vol_bid_odd):
    """(filled shares, new a4) for a buy at `price` with `rem` shares left, in bar u."""
    if ask_min[u] > 0 and ask_min[u] <= price:                      # crossed (S, T, Q-1)
        return rem, a4
    if kind == 0:
        return 0, a4
    if kind == 1:
        if px_min[u] > 0 and px_min[u] <= price:                     # touched (T)
            return rem, a4
        return 0, a4
    credit = vol_bid[u] - vol_bid_odd[u]                             # odd lots never move the queue
    if credit < 0:
        credit = 0
    if bid_max[u] > 0 and bid_max[u] < price:                        # Q-2: market bid below mine, nothing ahead
        return min(rem, credit), 0
    if bid_min[u] == price and bid_max[u] == price:                  # Q-3: market bid at my price all second
        if a4 < 0:
            return 0, a4
        if 4 * credit > a4:
            return min(rem, (4 * credit - a4) // 4), 0
        return 0, a4 - 4 * credit
    return 0, a4                                                     # Q-4


@jit
def fill_sell(kind, price, rem, a4, u, bid_max, ask_min, ask_max, px_max, vol_ask, vol_ask_odd):
    """Mirror of fill_buy (bid <-> ask, min <-> max, inequalities reversed)."""
    if bid_max[u] > 0 and bid_max[u] >= price:
        return rem, a4
    if kind == 0:
        return 0, a4
    if kind == 1:
        if px_max[u] > 0 and px_max[u] >= price:
            return rem, a4
        return 0, a4
    credit = vol_ask[u] - vol_ask_odd[u]
    if credit < 0:
        credit = 0
    if ask_min[u] > price:                                           # 0 (no state) is never above a positive price
        return min(rem, credit), 0
    if ask_min[u] == price and ask_max[u] == price:
        if a4 < 0:
            return 0, a4
        if 4 * credit > a4:
            return min(rem, (4 * credit - a4) // 4), 0
        return 0, a4 - 4 * credit
    return 0, a4


@jit
def fill_buy_l3(price, rem, u, ask_min, bid_max, asz, vol_bid, vol_bid_odd):
    """L3 buy (QTY16 4): the Q-2 volume (every bid-side trade below my price) and, when the ask reaches my price, at most
    the displayed ask size of the bar; the larger of the two, never both."""
    t = 0
    if bid_max[u] > 0 and bid_max[u] < price:                        # 1) trades through my price
        t = vol_bid[u] - vol_bid_odd[u]
        if t < 0:
            t = 0
    if ask_min[u] > 0 and ask_min[u] <= price:                       # 2) cross, with or without a print through
        return min(rem, max(asz[u], t))
    return min(rem, t)


@jit
def fill_sell_l3(price, rem, u, bid_max, ask_min, bsz, vol_ask, vol_ask_odd):
    """Mirror of fill_buy_l3."""
    t = 0
    if ask_min[u] > price:
        t = vol_ask[u] - vol_ask_odd[u]
        if t < 0:
            t = 0
    if bid_max[u] > 0 and bid_max[u] >= price:
        return min(rem, max(bsz[u], t))
    return min(rem, t)


# ---------------------------------------------------------------- kernel (qty16.exec_q + stop-loss)
@jit
def exec17(method, lrule, improve, e1_k, stop, dirs, starts, horizon, qty, bid, ask, bsz, asz, bid_max, bid_min, ask_max,
           ask_min, vol_bid, vol_ask, vol_bid_odd, vol_ask_odd, px_min, px_max, out):
    """qty16.exec_q line for line (lrule 0 L0, 3 L3) with the stop-loss: stop > 0 sends the rest to market at a decision
    bar where (bid + ask) has moved against the order by >= 2 x stop from the arrival bar. Deadline, then E1, then stop."""
    for i in range(len(dirs)):
        d, s = dirs[i], starts[i]
        D = s + horizon
        m2 = bid[s] + ask[s]
        lim_sh, lim_not, mkt_sh, mkt_not, over, done, dl, st_sh, skip = 0, 0, 0, 0, 0, 0, 0, 0, 0
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
                done = u - s
        else:
            rem, has, px, a4 = qty, False, 0, 0
            for u in range(s, D + 1):
                if u > s and has and rem > 0:
                    if d > 0:
                        if lrule == 0:
                            f, a4 = fill_buy(2, px, rem, a4, u, ask_min, bid_max, bid_min, px_min, vol_bid, vol_bid_odd)
                        else:
                            f = fill_buy_l3(px, rem, u, ask_min, bid_max, asz, vol_bid, vol_bid_odd)
                    else:
                        if lrule == 0:
                            f, a4 = fill_sell(2, px, rem, a4, u, bid_max, ask_min, ask_max, px_max, vol_ask, vol_ask_odd)
                        else:
                            f = fill_sell_l3(px, rem, u, bid_max, ask_min, bsz, vol_ask, vol_ask_odd)
                    if f > 0:
                        lim_sh += f
                        lim_not += f * px
                        rem -= f
                if rem == 0:
                    done = u - s
                    break
                if (u - s) % STEP != 0:
                    continue
                spr = ask[u] - bid[u]
                e1 = e1_k >= 0 and spr <= e1_k * TICK
                hit = False
                if stop > 0 and u > s:
                    if bid[u] > 0 and ask[u] > 0:
                        mv = (bid[u] + ask[u] - m2) if d > 0 else (m2 - bid[u] - ask[u])
                        hit = mv >= 2 * stop
                    else:
                        skip += 1
                if u == D or e1 or hit:
                    mp_ = ask[u] if d > 0 else bid[u]
                    disp = asz[u] if d > 0 else bsz[u]
                    mkt_sh += rem
                    mkt_not += rem * mp_
                    over += max(0, rem - disp)
                    if u == D:
                        dl = rem
                    elif not e1:
                        st_sh = rem
                    rem = 0
                    done = u - s
                    break
                if spr >= TICK:
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
        out[i, 0], out[i, 1], out[i, 2], out[i, 3], out[i, 4], out[i, 5], out[i, 6], out[i, 7], out[i, 8] = \
            lim_sh, lim_not, mkt_sh, mkt_not, over, done, dl, st_sh, skip


def run(b, name, lrule, dirs, starts):
    method, improve, e1_k, stop = RUNS[name]
    out = np.zeros((len(dirs), NCOL), dtype=np.int64)
    exec17(method, lrule, improve, e1_k, stop, dirs, starts, HOR, QTY, b["bid"], b["ask"], b["bsz"], b["asz"], b["bid_max"],
           b["bid_min"], b["ask_max"], b["ask_min"], b["vol_bid"], b["vol_ask"], b["vol_bid_odd"], b["vol_ask_odd"],
           b["px_min"], b["px_max"], out)
    return out


def costs(out, dirs, mid2, depth):
    """IS (R), cents per share: exec15.costs; depth 1 (D1) prices market shares above the displayed size one tick worse."""
    mkt_not = out[..., 3] + depth * dirs * TICK * out[..., 4]
    avg = (out[..., 1] + mkt_not) / QTY
    return dirs * (avg - mid2 / 2) / 100 + FEE_MKT * out[..., 2] / QTY


# ---------------------------------------------------------------- pre-order variables (decomp16)
def mids(b):
    bid, ask = b["bid"].astype(np.float64), b["ask"].astype(np.float64)
    valid = (b["bid"] > 0) & (b["ask"] > 0) & (b["ask"] >= b["bid"])
    return np.where(valid, (bid + ask) / 2, np.nan), valid


def ret_std(m, v, pts, scale):
    """Std (ddof 1) of log returns between consecutive points with both mids valid; (value, n returns)."""
    a, b_ = pts[:-1], pts[1:]
    ok = v[a] & v[b_]
    r = np.log(m[b_[ok]] / m[a[ok]])
    return (float(np.std(r, ddof=1) * scale) if len(r) >= 2 else np.nan), int(len(r))


def vol_pre(m, v, s):
    """DECOMP16 appendix: points s-1-60k (k = 0..30, >= 0) in time order; (bp per minute or nan, last bar used)."""
    pts = (s - 1 - PRE_STEP * np.arange(PRE_N + 1))[::-1]
    pts = pts[pts >= 0]
    val, n = ret_std(m, v, pts, 1e4) if len(pts) >= 2 else (np.nan, 0)
    return (val if n >= PRE_MIN else np.nan), int(pts.max()) if len(pts) else -1


def tercile(x, c):
    t = np.full(len(x), -1)
    ok = ~np.isnan(x)
    t[ok] = np.where(x[ok] <= c[0], 0, np.where(x[ok] <= c[1], 1, 2))
    return t


def adv_prior(dayvol, cal, keys):
    """{(sym, date): (mean daily volume or nan, n days, last day used)} over the previous 20 trading days of the same year."""
    res = {}
    pos = {y: {d: i for i, d in enumerate(c)} for y, c in cal.items()}
    for sym, d in keys:
        y = int(d[:4])
        i = pos[y][d]
        win = cal[y][max(0, i - ADV_WIN):i]
        vols = [(w, dayvol[(sym, w)]) for w in win if (sym, w) in dayvol]
        res[(sym, d)] = ((float(np.mean([x for _, x in vols])) if len(vols) >= ADV_MIN else np.nan), len(vols),
                         vols[-1][0] if vols else None)
    return res


# ---------------------------------------------------------------- one symbol-day
def symday(b, date, close_bar):
    """Parents of the date with a valid start quote, every run under L0 and L3, and the pre-order variables.
    (meta, outs[len(LRULES), len(RUN_NAMES), n, NCOL])."""
    dirs, starts = parents(date, close_bar)
    ok = quote_ok(b)[starts]
    dirs, starts = dirs[ok], starts[ok]
    outs = np.zeros((len(LRULES), len(RUN_NAMES), len(dirs), NCOL), dtype=np.int64)
    for li, lr in enumerate(LRULES):
        for ri, name in enumerate(RUN_NAMES):
            outs[li, ri] = run(b, name, lr, dirs, starts)
    m, v = mids(b)
    vp = np.array([vol_pre(m, v, int(s))[0] for s in starts], dtype=np.float64)
    meta = {"dir": dirs, "start": starts, "mid2": (b["bid"] + b["ask"])[starts], "spread": (b["ask"] - b["bid"])[starts],
            "vol_pre": vp, "day_vol": int(b["vol"].sum()), "n_dropped": int((~ok).sum())}
    return meta, outs


# ---------------------------------------------------------------- candidates
def branch_c2(vp):
    """S2i when vol_pre < 4.68 bp; an undefined vol_pre (nan) goes to E* (ANSWERS 9)."""
    vp = np.asarray(vp, dtype=np.float64)
    return np.where(np.isnan(vp), False, vp < VOL_CUT)


def branch_c3(spread):
    return np.asarray(spread) <= SPREAD_CUT


def compose(o, vp, spread):
    """o[len(RUN_NAMES), n, NCOL] for one fill rule -> ({candidate: out[n, NCOL]}, {candidate: S2i-branch flag[n]})."""
    r = {k: o[i] for i, k in enumerate(RUN_NAMES)}
    b2, b3 = branch_c2(vp), branch_c3(spread)
    n = o.shape[1]
    pick = lambda flag, a, b_: np.where(flag[:, None], r[a], r[b_])                      # noqa: E731
    out = {"C0": r["Estar"], "C1": r["S2i"], "C2": pick(b2, "S2i", "Estar"), "C3": pick(b3, "S2i", "Estar"),
           "C4": r["Estar_stop"], "C5": pick(b2, "S2i_stop", "Estar_stop")}
    flags = {"C0": np.zeros(n, bool), "C1": np.ones(n, bool), "C2": b2, "C3": b3, "C4": np.zeros(n, bool), "C5": b2}
    return out, flags


# ---------------------------------------------------------------- statistics (robust16 / run_test15 with seed 17)
def boot_index(T, B=BOOT_B, block=BOOT_BLOCK, seed=BOOT_SEED):
    """Circular blocks of days (run_test15.boot_ci resampling) as an index matrix B x T."""
    rng = np.random.default_rng(seed)
    k = math.ceil(T / block)
    return (rng.integers(0, T, size=(B, k))[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :T] % T


def boot_stat(d, date_codes, n_days, idx):
    """(per-parent mean, bootstrap distribution of the ratio of day sums to day counts)."""
    s = np.bincount(date_codes, weights=d, minlength=n_days)
    n = np.bincount(date_codes, minlength=n_days).astype(np.float64)
    keep = n > 0
    s, n = s[keep], n[keep]
    if idx.max() >= len(s):
        raise ValueError("bootstrap index is longer than the days with parents")
    return float(s.sum() / n.sum()), s[idx].sum(axis=1) / n[idx].sum(axis=1)


def diff_ci(d, date_codes, n_days, idx):
    """robust16.diff_ci: mean and the two-sided 95% interval."""
    est, stat = boot_stat(d, date_codes, n_days, idx)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return est, float(lo), float(hi)


def one_sided(d, date_codes, n_days, idx):
    """ANSWERS 8: (mean, p = share of bootstrap means >= 0, one-sided 97.5% upper bound)."""
    est, stat = boot_stat(d, date_codes, n_days, idx)
    return est, float(np.mean(stat >= 0)), float(np.percentile(stat, 97.5))


def holm(p, alpha=0.05):
    """{name: rejected} by Holm: the k-th smallest p (k = 0..m-1) against alpha / (m - k), stopping at the first failure.
    With H1 and H2 the levels are 0.025 then 0.05 (ANSWERS 8)."""
    order = sorted(p, key=lambda k: (p[k], k))
    res, go = {}, True
    for k, name in enumerate(order):
        lvl = alpha / (len(order) - k)
        go = go and p[name] < lvl
        res[name] = {"p": p[name], "level": lvl, "rejected": bool(go)}
    return res


def select_astar(rows):
    """SPEC17 3: lowest L3D1 mean IS; within 0.02 cents of it, lower p95, then fewer rules (then the lower number).
    rows: {candidate: {"mean": , "p95": }}. Returns (A*, record of the steps)."""
    best = min(rows[c]["mean"] for c in CANDS)
    tied = [c for c in CANDS if rows[c]["mean"] - best <= TIE]
    p95 = min(rows[c]["p95"] for c in tied)
    low = [c for c in tied if rows[c]["p95"] == p95]
    pick = min(low, key=lambda c: (RULE_COUNT[c], CANDS.index(c)))
    return pick, {"lowest_mean": min(CANDS, key=lambda c: rows[c]["mean"]), "lowest_mean_c": best, "tied_within_0.02": tied,
                  "lowest_p95_among_tied": low, "rule_count": {c: RULE_COUNT[c] for c in low}, "astar": pick}
