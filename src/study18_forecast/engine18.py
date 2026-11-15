#!/usr/bin/env python3
"""Research 18 execution engine (SPEC18 3-5, SPEC18_ANSWERS 3-4 and appendix A), shared by test_engine18.py, dev18.py and
run_test18.py. A copy of study17_new_stocks/engine17.py (sha256 2a9ddcb7..., not changed); changed or new parts are marked "R18".
Copied code (engine17): TICK, STEP, FEE_MKT, start_grid, parents, quote_ok, the L0 / L3 fill rules, the qty16 kernel with the
stop-loss, costs, mids, ret_std, vol_pre, tercile, adv_prior, boot_index, boot_stat, diff_ci, holm, the C3 branch.
New code (R18):
  features  the decision grid t = 0, 5, ..., 295 s of every parent (u = s + t, path independent, ANSWERS 3) with four inputs
            from bar-end values up to u only: signed imbalance (bsz - asz) / (bsz + asz) x dir, spread in ticks capped at
            10, vol_pre at arrival (16-E), signed 2 x mid change over the last 5 bars in ticks (x dir). y = 1 when the first
            bar-end mid in u+1..u+60 that differs from mid(u) has moved against the order (buy up, sell down), 0 when it
            moved with it, -1 when none differs; bars with an invalid quote (bid or ask <= 0, ask < bid) are skipped.
            A decision bar with an invalid quote is not scored and gets no p (E* as is, ANSWERS A-11).
  model     standardised logistic regression read from model18.json: p = 1 / (1 + exp(-(b0 + sum b_j (x_j - m_j) / s_j))),
            vol_pre undefined -> the 2015-2018 median (SPEC18 3).
  kernel    exec18 = exec17 with the rules R1 (p >= p_hi at a decision bar with shares left: the rest to market at that bar's
            end quote, after the deadline and E1 checks) and R2 (E1 true and p <= p_lo: no market order, E*'s limit rule at
            that bar instead), and delay = 1 for the delay check (ANSWERS 4: every action decided at bar u is taken at bar
            u + 1 with that bar's quote).
  stats     weighted linear-interpolation percentile (np.percentile on the resampled days), bootstrap p95 differences and
            the exact Mann-Whitney AUC (ties 1/2) on resampled days; bootstrap seed 18."""
import math

import numpy as np

try:
    from numba import njit as _njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    prange = range

# ---------------------------------------------------------------- constants (exec15, robust16, SPEC17, SPEC18)
TICK, N_PARENT, FIRST_START, STEP = 100, 20, 299, 5     # prices in 1e-4 dollars, 1 tick = 1 cent
FEE_MKT = 0.3                                            # cents per share, cost scenario R
QTY, HOR = 1000, 300
SPREAD_CUT = 4 * TICK                                    # C3: arrival spread <= 4 cents -> S2i
E1_K = 2                                                 # E* = E12|E2off|E3off|E4off (r2_15), e1_k = 2
LRULES = (0, 3)
VARIANTS = (("L3D1", 3, 1), ("L0D0", 0, 0))              # (label, fill rule, depth); L3D1 is the primary variant
NG, YWIN, SPR_CAP = 60, 60, 10                           # R18: grid t = 0..295 s, y window u+1..u+60, spread cap
FEATS = ("imbalance", "spread_ticks", "vol_pre", "mid_change_5s")
P_HI = (0.60, 0.65, 0.70, 0.75)                          # R18: SPEC18 4 threshold grid
P_LO = (0.30, 0.35, 0.40)
CANDS = (("R0",) + tuple(f"R1_{h:.2f}" for h in P_HI) + tuple(f"R2_{lo:.2f}" for lo in P_LO)
         + tuple(f"R3_{h:.2f}_{lo:.2f}" for h in P_HI for lo in P_LO))                 # R0 + 19 candidates
TIE = 0.02                                                                             # cents
# R18: method, improve, e1_k, stop; the rules add (p_hi, p_lo), -1 = off
BASE_RUNS = {"S0": (0, 0, -1, 0), "S1": (1, 0, -1, 0), "S2i": (2, 1, -1, 0), "R0": (2, 1, E1_K, 0)}
NCOL = 12   # lim_sh, lim_not, mkt_sh, mkt_not, over, done (s), deadline sh, stop sh, stop tests skipped,
#             R18: R1 sh, R2 skips, market orders at a non-positive price
PRE_MIN, PRE_N, PRE_STEP = 10, 30, 60                    # vol_pre: >= 10 one-minute returns out of up to 30 (16-E)
ADV_WIN, ADV_MIN = 20, 5
CUT_VOL = (4.676959528125974, 8.67061909828224)          # 16-E development tercile cut-offs (decomp16_result.json)
CUT_ADV = (484312.9, 801015.7)
TERC = ("low", "mid", "high")
BOOT_B, BOOT_BLOCK, BOOT_SEED = 10_000, 21, 18           # R18: seed 18


def jit(f):
    """fillmodel15.jit: numba-compiled f when numba is available (f.py_func is the plain function either way)."""
    if HAVE_NUMBA:
        return _njit(cache=False)(f)
    f.py_func = f
    return f


def pjit(f):
    """R18: parallel numba for the bootstrap loops."""
    if HAVE_NUMBA:
        return _njit(cache=False, parallel=True)(f)
    f.py_func = f
    return f


def params(name):
    """R18: (method, improve, e1_k, stop, p_hi, p_lo) of a run name (S0, S1, S2i, R0, R1_h, R2_l, R3_h_l)."""
    if name in BASE_RUNS:
        return BASE_RUNS[name] + (-1.0, -1.0)
    k = name.split("_")
    m, i, e, s = BASE_RUNS["R0"]
    if k[0] == "R1":
        return m, i, e, s, float(k[1]), -1.0
    if k[0] == "R2":
        return m, i, e, s, -1.0, float(k[1])
    if k[0] == "R3":
        return m, i, e, s, float(k[1]), float(k[2])
    raise KeyError(name)


def kind(name):
    return name.split("_")[0]


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


# ---------------------------------------------------------------- kernel (exec17 + R1 / R2 + delay, R18)
@jit
def exec18(method, lrule, improve, e1_k, stop, phi, plo, delay, P, dirs, starts, horizon, qty, bid, ask, bsz, asz, bid_max,
           bid_min, ask_max, ask_min, vol_bid, vol_ask, vol_bid_odd, vol_ask_odd, px_min, px_max, out):
    """exec17 line for line when phi < 0, plo < 0 and delay = 0. At a decision bar u = s + 5k (k < NG, p = P[i, k], nan = no
    p): deadline, then E1 (skipped by R2 when p <= plo), then the stop, then R1 (p >= phi) send the rest to market.
    delay = 1: the market order, the new limit price or the cancel decided at u is taken at bar u + 1 with bar u + 1's
    quote (fills of the standing order at u + 1 come first); the deadline order of bar D goes at D + 1."""
    ng = P.shape[1]
    for i in range(len(dirs)):
        d, s = dirs[i], starts[i]
        D = s + horizon
        m2 = bid[s] + ask[s]
        lim_sh, lim_not, mkt_sh, mkt_not, over, done, dl, st_sh, skip = 0, 0, 0, 0, 0, 0, 0, 0, 0
        r1_sh, r2_n, bad = 0, 0, 0
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
                if px <= 0:
                    bad += 1
                done = u - s
        else:
            rem, has, px, a4 = qty, False, 0, 0
            pend_u, pend_kind, pend_tag, pend_imp = -1, 0, 0, False
            for u in range(s, D + delay + 1):
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
                if pend_u == u:                                              # R18: delay 1, the action of bar u - 1
                    pend_u = -1
                    if pend_kind == 1:
                        mp_ = ask[u] if d > 0 else bid[u]
                        disp = asz[u] if d > 0 else bsz[u]
                        mkt_sh += rem
                        mkt_not += rem * mp_
                        over += max(0, rem - disp)
                        if mp_ <= 0:
                            bad += 1
                        if pend_tag == 0:
                            dl = rem
                        elif pend_tag == 2:
                            st_sh = rem
                        elif pend_tag == 3:
                            r1_sh = rem
                        rem = 0
                        done = u - s
                        break
                    if pend_kind == 2:
                        if pend_imp:
                            npx = bid[u] + TICK if d > 0 else ask[u] - TICK
                            na4 = 0
                        else:
                            npx = bid[u] if d > 0 else ask[u]
                            na4 = 4 * (bsz[u] if d > 0 else asz[u])
                        if not (has and npx == px):
                            has, px, a4 = True, npx, na4
                    else:
                        has = False
                if u > D or (u - s) % STEP != 0:
                    continue
                spr = ask[u] - bid[u]
                k = (u - s) // STEP
                pk = P[i, k] if k < ng else np.nan
                e1 = e1_k >= 0 and spr <= e1_k * TICK
                if e1 and plo >= 0 and u < D and not np.isnan(pk) and pk <= plo:      # R18: R2
                    e1 = False
                    r2_n += 1
                hit = False
                if stop > 0 and u > s:
                    if bid[u] > 0 and ask[u] > 0:
                        mv = (bid[u] + ask[u] - m2) if d > 0 else (m2 - bid[u] - ask[u])
                        hit = mv >= 2 * stop
                    else:
                        skip += 1
                r1 = (not e1) and (not hit) and u < D and phi >= 0 and (not np.isnan(pk)) and pk >= phi   # R18: R1
                if u == D or e1 or hit or r1:
                    tag = 0 if u == D else (1 if e1 else (2 if hit else 3))
                    if delay == 1:
                        pend_u, pend_kind, pend_tag = u + 1, 1, tag
                        continue
                    mp_ = ask[u] if d > 0 else bid[u]
                    disp = asz[u] if d > 0 else bsz[u]
                    mkt_sh += rem
                    mkt_not += rem * mp_
                    over += max(0, rem - disp)
                    if mp_ <= 0:
                        bad += 1
                    if tag == 0:
                        dl = rem
                    elif tag == 2:
                        st_sh = rem
                    elif tag == 3:
                        r1_sh = rem
                    rem = 0
                    done = u - s
                    break
                if spr >= TICK:
                    imp = improve == 1 and spr >= 3 * TICK
                    if delay == 1:
                        pend_u, pend_kind, pend_imp = u + 1, 2, imp
                        continue
                    if imp:
                        npx = bid[u] + TICK if d > 0 else ask[u] - TICK
                        na4 = 0
                    else:
                        npx = bid[u] if d > 0 else ask[u]
                        na4 = 4 * (bsz[u] if d > 0 else asz[u])
                    if not (has and npx == px):
                        has, px, a4 = True, npx, na4
                else:
                    if delay == 1:
                        pend_u, pend_kind = u + 1, 3
                        continue
                    has = False
        out[i, 0], out[i, 1], out[i, 2], out[i, 3], out[i, 4], out[i, 5], out[i, 6], out[i, 7], out[i, 8] = \
            lim_sh, lim_not, mkt_sh, mkt_not, over, done, dl, st_sh, skip
        out[i, 9], out[i, 10], out[i, 11] = r1_sh, r2_n, bad


def run(b, name, lrule, dirs, starts, P=None, delay=0):
    method, improve, e1_k, stop, phi, plo = params(name)
    if P is None:
        P = np.full((len(dirs), NG), np.nan)
    out = np.zeros((len(dirs), NCOL), dtype=np.int64)
    exec18(method, lrule, improve, e1_k, stop, phi, plo, delay, np.ascontiguousarray(P, dtype=np.float64), dirs, starts,
           HOR, QTY, b["bid"], b["ask"], b["bsz"], b["asz"], b["bid_max"], b["bid_min"], b["ask_max"], b["ask_min"],
           b["vol_bid"], b["vol_ask"], b["vol_bid_odd"], b["vol_ask_odd"], b["px_min"], b["px_max"], out)
    return out


def costs(out, dirs, mid2, depth):
    """IS (R), cents per share: exec15.costs; depth 1 (D1) prices market shares above the displayed size one tick worse."""
    mkt_not = out[..., 3] + depth * dirs * TICK * out[..., 4]
    avg = (out[..., 1] + mkt_not) / QTY
    return dirs * (avg - mid2 / 2) / 100 + FEE_MKT * out[..., 2] / QTY


def fired(out, name):
    """R18 (ANSWERS A-20): parents where the rule acted differently from E* at least once (R1 market, R2 skip)."""
    k = kind(name)
    r1, r2 = out[:, 9] > 0, out[:, 10] > 0
    return {"R0": np.zeros(len(out), bool), "R1": r1, "R2": r2, "R3": r1 | r2}[k]


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


# ---------------------------------------------------------------- R18: grid features, y and the model
def features(b, dirs, starts):
    """Grid u = s + 5k (k = 0..NG-1) of every parent: {imb, spr, d5 [n, NG] float64, vp [n] (nan = undefined), y [n, NG]
    int8 (-1 = no change in u+1..u+60), valid [n, NG] (decision quote valid), counts}. Inputs use bars <= u only."""
    U = starts[:, None] + STEP * np.arange(NG)[None, :]
    bid, ask, bsz, asz = b["bid"], b["ask"], b["bsz"], b["asz"]
    ok = (bid > 0) & (ask > 0) & (ask >= bid)                                # ANSWERS A-11
    m2 = bid + ask                                                           # 2 x mid (integer)
    d = dirs[:, None]
    tot = bsz[U] + asz[U]
    imb = np.where(tot > 0, (bsz[U] - asz[U]) / np.where(tot > 0, tot, 1), 0.0) * d
    spr = np.minimum((ask[U] - bid[U]) / TICK, SPR_CAP).astype(np.float64)
    prev = U - STEP
    d5 = np.where(ok[prev], (m2[U] - m2[prev]) / (2 * TICK), 0.0) * d
    F = U[:, :, None] + np.arange(1, YWIN + 1)[None, None, :]
    ch = ok[F] & (m2[F] != m2[U][:, :, None])
    has = ch.any(axis=2)
    first = np.take_along_axis(m2[F], ch.argmax(axis=2)[..., None], axis=2)[..., 0]
    y = np.where(has, ((first - m2[U]) * d > 0).astype(np.int8), np.int8(-1)).astype(np.int8)
    valid = ok[U]
    m, v = mids(b)
    vp = np.array([vol_pre(m, v, int(s))[0] for s in starts], dtype=np.float64)
    counts = {"points": int(U.size), "invalid_decision_quote": int((~valid).sum()),
              "zero_size_imbalance_0": int((valid & (tot <= 0)).sum()), "d5_prev_invalid_0": int((valid & ~ok[prev]).sum())}
    return {"imb": imb, "spr": spr, "d5": d5, "vp": vp, "y": y, "valid": valid, "counts": counts}


def design(f, vp_fill):
    """[n, NG, 4] raw inputs in FEATS order; an undefined vol_pre takes vp_fill."""
    vp = np.where(np.isnan(f["vp"]), vp_fill, f["vp"])
    return np.stack([f["imb"], f["spr"], np.broadcast_to(vp[:, None], f["imb"].shape), f["d5"]], axis=-1)


def predict(model, f):
    """p [n, NG] from model18.json values; nan where the decision quote is invalid."""
    X = design(f, model["vol_pre_fill"])
    z = (X - np.asarray(model["mean"])) / np.asarray(model["sd"])
    p = 1.0 / (1.0 + np.exp(-(model["intercept"] + z @ np.asarray(model["coef"]))))
    return np.where(f["valid"], p, np.nan)


# ---------------------------------------------------------------- one symbol-day
def symday18(b, date, close_bar, model, runs, delay_runs=()):
    """Parents of the date with a valid start quote, grid features and p (model None: no p), the runs under L0 and L3
    (delay_runs with delay 1 after them). (meta, outs[len(LRULES), len(runs) + len(delay_runs), n, NCOL])."""
    dirs, starts = parents(date, close_bar)
    ok = quote_ok(b)[starts]
    dirs, starts = dirs[ok], starts[ok]
    f = features(b, dirs, starts)
    P = predict(model, f) if model is not None else np.full((len(dirs), NG), np.nan)
    names = list(runs) + list(delay_runs)
    outs = np.zeros((len(LRULES), len(names), len(dirs), NCOL), dtype=np.int64)
    for li, lr in enumerate(LRULES):
        for ri, name in enumerate(names):
            outs[li, ri] = run(b, name, lr, dirs, starts, P, 1 if ri >= len(runs) else 0)
    meta = {"dir": dirs, "start": starts, "mid2": (b["bid"] + b["ask"])[starts], "spread": (b["ask"] - b["bid"])[starts],
            "vol_pre": f["vp"], "day_vol": int(b["vol"].sum()), "n_dropped": int((~ok).sum()), "P": P, "y": f["y"],
            "valid": f["valid"], "X": design(f, np.nan), "counts": f["counts"]}
    return meta, outs


def branch_c3(spread):
    return np.asarray(spread) <= SPREAD_CUT


# ---------------------------------------------------------------- statistics (robust16 / run_test15, seed 18)
def boot_index(T, B=BOOT_B, block=BOOT_BLOCK, seed=BOOT_SEED):
    """Circular blocks of days (run_test15.boot_ci resampling) as an index matrix B x T."""
    rng = np.random.default_rng(seed)
    k = math.ceil(T / block)
    return (rng.integers(0, T, size=(B, k))[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :T] % T


def day_counts(idx, T):
    """R18: B x T times each day is drawn."""
    B = idx.shape[0]
    return np.bincount((idx + T * np.arange(B)[:, None]).ravel(), minlength=B * T).reshape(B, T).astype(np.float64)


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


@jit
def wpct(xs, cs, w, q):
    """np.percentile(x, q) (linear) of the multiset where sorted value xs[i] of day cs[i] appears w[cs[i]] times."""
    N = 0.0
    for i in range(len(xs)):
        N += w[cs[i]]
    h = (N - 1.0) * q / 100.0
    lo = math.floor(h)
    fr = h - lo
    cum, a, b_ = 0.0, np.nan, np.nan
    for i in range(len(xs)):
        cum += w[cs[i]]
        if np.isnan(a) and cum > lo:
            a = xs[i]
        if cum > lo + 1.0:
            b_ = xs[i]
            break
    if np.isnan(b_):
        b_ = a
    return a + (b_ - a) * fr


@pjit
def boot_pct_diff(xa, ca, xb, cb, W, q):
    """Per draw (rows of W, B x T day weights): wpct(a) - wpct(b); xa / xb sorted with their day codes."""
    res = np.empty(W.shape[0])
    for k in prange(W.shape[0]):
        res[k] = wpct(xa, ca, W[k], q) - wpct(xb, cb, W[k], q)
    return res


def sorted_by(x, codes):
    o = np.argsort(x, kind="stable")
    return np.ascontiguousarray(x[o], dtype=np.float64), np.ascontiguousarray(codes[o], dtype=np.int64)


def p95_diff_boot(a, b_, codes, W, q=95.0):
    """(p95(a) - p95(b) with np.percentile, bootstrap distribution of the same on resampled days; ANSWERS 4)."""
    xa, ca = sorted_by(a, codes)
    xb, cb = sorted_by(b_, codes)
    return float(np.percentile(a, q) - np.percentile(b_, q)), boot_pct_diff(xa, ca, xb, cb, W, q)


@pjit
def _boot_auc(gend, dy, yy, W):
    res = np.empty(W.shape[0])
    for k in prange(W.shape[0]):
        w = W[k]
        cn, num, wp, st = 0.0, 0.0, 0.0, 0
        for g in range(len(gend)):
            pw, nw = 0.0, 0.0
            for i in range(st, gend[g]):
                if yy[i] == 1:
                    pw += w[dy[i]]
                else:
                    nw += w[dy[i]]
            num += pw * (cn + 0.5 * nw)
            cn += nw
            wp += pw
            st = gend[g]
        res[k] = num / (wp * cn)
    return res


def auc_prep(p, y, codes):
    """Points sorted by p, grouped by equal p (ties count 1/2)."""
    o = np.argsort(p, kind="stable")
    ps = p[o]
    gend = np.append(np.flatnonzero(np.diff(ps) != 0) + 1, len(ps)).astype(np.int64)
    return gend, np.ascontiguousarray(codes[o], dtype=np.int64), np.ascontiguousarray(y[o], dtype=np.int8)


def auc_boot(p, y, codes, W):
    """(exact AUC, AUC on each row of day weights W)."""
    g, dy, yy = auc_prep(p, y, codes)
    T = int(codes.max()) + 1
    return float(_boot_auc(g, dy, yy, np.ones((1, T)))[0]), _boot_auc(g, dy, yy, W)


def holm(p, alpha=0.05):
    """{name: rejected} by Holm: the k-th smallest p (k = 0..m-1) against alpha / (m - k), stopping at the first failure.
    With three hypotheses the levels are 0.0167, 0.025, 0.05 (SPEC18 5)."""
    order = sorted(p, key=lambda k: (p[k], k))
    res, go = {}, True
    for k, name in enumerate(order):
        lvl = alpha / (len(order) - k)
        go = go and p[name] < lvl
        res[name] = {"p": p[name], "level": lvl, "rejected": bool(go)}
    return res


def select_astar(rows):
    """SPEC18 4: lowest L3D1 mean IS; within 0.02 cents of it, lower p95, then the lower firing share (then the candidate
    order R0, R1, R2, R3). rows: {candidate: {"mean": , "p95": , "fired": }}. Returns (A*, record of the steps)."""
    best = min(rows[c]["mean"] for c in CANDS)
    tied = [c for c in CANDS if rows[c]["mean"] - best <= TIE]
    p95 = min(rows[c]["p95"] for c in tied)
    low = [c for c in tied if rows[c]["p95"] == p95]
    pick = min(low, key=lambda c: (rows[c]["fired"], CANDS.index(c)))
    return pick, {"lowest_mean": min(CANDS, key=lambda c: rows[c]["mean"]), "lowest_mean_c": best, "tied_within_0.02": tied,
                  "lowest_p95_among_tied": low, "fired": {c: rows[c]["fired"] for c in low}, "astar": pick}
