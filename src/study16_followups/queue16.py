#!/usr/bin/env python3
"""Research 16-B diagnostic (QUEUE16.md; implementation details DEVIATIONS16.md 2): the research 12 B0 market maker with
the queue ahead of a new limit order set to x times the displayed size at its price, x = 0, 0.1, ..., 1.0 (x = 1 is
research 12 model Q, x = 0 the front of the queue). Research 12 dev stock bars 2019-2020 only (research 12 loader; the
holdout and reserve folders are never opened). No verdict: a diagnostic.
  python queue16.py run [--workers 6]    sweep, checks, hand trace -> research16/results/
Engine: research 12 kernel and Q fill model copied below; the only change is the queue unit K (queue ahead in 1/K shares;
research 12 has K = 4 and theta4 = 4 theta). Here K = 10 and the theta slot of the parameter vector holds n = 10 x."""
import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R12 = os.path.join(os.path.dirname(HERE), "study12_market_maker")
sys.path.insert(0, R12)
import accounting  # noqa: E402
import features  # noqa: E402
import loader  # noqa: E402
import session  # noqa: E402
import simulator  # noqa: E402
from config import (B0, COST, FILL_COLS, MAX_ROWS, N_STATS, ORDER_COLS, P_THETA4, Q, S_MAXQ, S_MKT_OVER,  # noqa: E402
                    S_MKT_STALEQ, S_NDEC, S_NDEC_QB, S_NINT_B, S_NINT_S, S_NINTF_B, S_NINTF_S, S_NMKT, S_NSTALE, S_QDF,
                    S_SKIP, S_SUMQ, TICK)
from fillmodel import jit, market_price  # noqa: E402
from simulator import _close, _order, _push  # noqa: E402
from stats import sharpe  # noqa: E402

RES = os.path.join(HERE, "results")
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research16_data")
DEV_DAILY = os.path.join(R12, "results", "dev_daily.csv")
S12_SUMMARY = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "study12", "s12_summary.json")
K = 10
N_GRID = tuple(range(11))                                   # x = n / 10
SCN = ("R", "R+")
REF_RUNS = ("Q", "Q50", "T")                                # research 12 engine as is (checks and the T reference)
KEYS = ("cash", "sh_lim_buy", "sh_lim_sell", "sh_mkt_buy", "sh_mkt_sell", "n_fill_ev")
SR_LINE = 0.5
TOL_USD = 0.01
TRACE_X = 5                                                 # n for the hand trace (x = 0.5)


# ---------------------------------------------------------------- engine (research 12 fillmodel.py / simulator.py, + K)
@jit
def fill_buy(kind, price, rem, a, u, ask_min, bid_max, bid_min, px_min, vol_bid, vol_bid_odd, K):
    """research 12 fill_buy with the queue in 1/K shares (research 12: K = 4)."""
    if ask_min[u] > 0 and ask_min[u] <= price:                      # crossed (S, T, Q-1)
        return rem, a
    if kind == 0:
        return 0, a
    if kind == 1:
        if px_min[u] > 0 and px_min[u] <= price:                     # touched (T)
            return rem, a
        return 0, a
    credit = vol_bid[u] - vol_bid_odd[u]                             # odd lots never move the queue
    if credit < 0:
        credit = 0
    if bid_max[u] > 0 and bid_max[u] < price:                        # Q-2: market bid below mine, nothing ahead
        return min(rem, credit), 0
    if bid_min[u] == price and bid_max[u] == price:                  # Q-3: market bid at my price all second
        if a < 0:
            return 0, a
        if K * credit > a:
            return min(rem, (K * credit - a) // K), 0
        return 0, a - K * credit
    return 0, a                                                      # Q-4


@jit
def fill_sell(kind, price, rem, a, u, bid_max, ask_min, ask_max, px_max, vol_ask, vol_ask_odd, K):
    """Mirror of fill_buy (bid <-> ask, min <-> max, inequalities reversed)."""
    if bid_max[u] > 0 and bid_max[u] >= price:
        return rem, a
    if kind == 0:
        return 0, a
    if kind == 1:
        if px_max[u] > 0 and px_max[u] >= price:
            return rem, a
        return 0, a
    credit = vol_ask[u] - vol_ask_odd[u]
    if credit < 0:
        credit = 0
    if ask_min[u] > price:                                           # 0 (no state) is never above a positive price
        return min(rem, credit), 0
    if ask_min[u] == price and ask_max[u] == price:
        if a < 0:
            return 0, a
        if K * credit > a:
            return min(rem, (K * credit - a) // K), 0
        return 0, a - K * credit
    return 0, a


@jit
def kernel(bid, ask, bsz, asz, bid_max, bid_min, ask_max, ask_min, vol_bid, vol_ask, vol_bid_odd, vol_ask_odd, px_min, px_max,
           mid2, qbase, obi_bb, obi_bs, fi_bb, fi_bs, gcnt, halt_trig, stale_q, p, fills, orders, st, lots, K):
    """research 12 simulator.kernel; the fill calls take K and p[P_THETA4] is theta x K."""
    for i in range(st.shape[0]):
        st[i] = 0
    mode, delta, L, stale_s, long_only = p[0], p[1], p[2], p[3], p[4]
    use_obi, use_halt, use_fi, min_spr, gcnt_m = p[5], p[6], p[7], p[8] * TICK, p[9]
    d0, df, naive, kind, theta4, gate = p[10], p[11], p[12], p[13], p[14], p[15]
    if gate == 0:
        st[S_SKIP] = 1
        return 0, 0
    if not (bid[d0] > 0 and ask[d0] > 0 and ask[d0] >= bid[d0]):     # E1-26
        st[S_SKIP] = 2
        return 0, 0
    LQ = L * Q
    q = 0
    hit = -1                                                         # bar where |q| reached L*Q (E1-50 step 2)
    halt_end = -1
    b_px, b_rem, b_a4, b_dec, b_hit, b_slot = 0, 0, 0, -1, 0, -1     # resting buy (b_slot = its row in `orders`)
    a_px, a_rem, a_a4, a_dec, a_hit, a_slot = 0, 0, 0, -1, 0, -1     # resting sell
    m_qty, m_side, m_dec = 0, 0, -1                                  # market order waiting for its bar
    lh, lt, sum_pq, sum_q = 0, 0, 0, 0                               # A_naive FIFO lots (E1-55)
    nf, no = 0, 0
    for u in range(d0, df + 2):
        # 1. fills in bar u
        if b_rem > 0:
            f, b_a4 = fill_buy(kind, b_px, b_rem, b_a4, u, ask_min, bid_max, bid_min, px_min, vol_bid, vol_bid_odd, K)
            if f > 0:
                ref = mid2[b_dec] if mode == 0 else mid2[u - 1]
                nf = _push(fills, nf, u, 1, 0, f, b_px, ref, b_dec)
                q += f
                b_rem -= f
                if b_hit == 0:
                    st[S_NINTF_B] += 1
                    b_hit = 1
                if naive == 1:
                    lots[lt, 0] = b_px
                    lots[lt, 1] = f
                    lt += 1
                    sum_pq += b_px * f
                    sum_q += f
        if a_rem > 0:
            f, a_a4 = fill_sell(kind, a_px, a_rem, a_a4, u, bid_max, ask_min, ask_max, px_max, vol_ask, vol_ask_odd, K)
            if f > 0:
                ref = mid2[a_dec] if mode == 0 else mid2[u - 1]
                nf = _push(fills, nf, u, -1, 0, f, a_px, ref, a_dec)
                q -= f
                a_rem -= f
                if a_hit == 0:
                    st[S_NINTF_S] += 1
                    a_hit = 1
                if naive == 1:
                    left = f
                    while left > 0:
                        take = min(left, lots[lh, 1])
                        lots[lh, 1] -= take
                        sum_pq -= take * lots[lh, 0]
                        sum_q -= take
                        left -= take
                        if lots[lh, 1] == 0:
                            lh += 1
        if m_qty > 0:
            nf = _push(fills, nf, u, m_side, 1, m_qty, market_price(m_side, m_dec, u, bid, ask, bid_min, ask_max),
                       mid2[m_dec], m_dec)
            m_qty = 0
        if u == df + 1:
            break
        aq = q if q >= 0 else -q
        if aq > st[S_MAXQ]:
            st[S_MAXQ] = aq
        st[S_SUMQ] += aq
        # 2. stale timer
        if aq == LQ:
            if hit < 0:
                hit = u
        else:
            hit = -1
        # 3. volatility halt (E1-51)
        if use_halt == 1 and halt_trig[u]:
            if u + 60 > halt_end:
                halt_end = u + 60
            b_rem = 0
            a_rem = 0
            b_slot = _close(orders, b_slot, u)
            a_slot = _close(orders, a_slot, u)
        # 4. flatten at d_f
        if u == df:
            b_rem = 0
            a_rem = 0
            b_slot = _close(orders, b_slot, u)
            a_slot = _close(orders, a_slot, u)
            st[S_QDF] = q
            if q != 0:
                m_qty = aq
                m_side = -1 if q > 0 else 1
                m_dec = u
                st[S_NMKT] += 1
                if (m_side < 0 and m_qty > bsz[u]) or (m_side > 0 and m_qty > asz[u]):
                    st[S_MKT_OVER] += 1
                if stale_q[u]:
                    st[S_MKT_STALEQ] += 1
                q = 0
            continue
        if not (mode == 1 or (u - d0) % delta == 0):
            continue
        # 4. decision at the end of bar u
        st[S_NDEC] += 1
        if qbase[u]:
            st[S_NDEC_QB] += 1
        if stale_s >= 0 and hit >= 0 and u - hit >= stale_s and m_qty == 0:      # E1-52
            m_qty = Q
            m_side = -1 if q > 0 else 1
            m_dec = u
            q += m_side * Q
            hit = -1
            st[S_NSTALE] += 1
            st[S_NMKT] += 1
            if (m_side < 0 and m_qty > bsz[u]) or (m_side > 0 and m_qty > asz[u]):
                st[S_MKT_OVER] += 1
            if stale_q[u]:
                st[S_MKT_STALEQ] += 1
        ok = (qbase[u] and ask[u] - bid[u] >= min_spr and (gcnt_m == 0 or gcnt[u] >= gcnt_m) and u > halt_end)
        buy_ok = ok and not (use_obi == 1 and obi_bb[u]) and not (use_fi == 1 and fi_bb[u])
        sell_ok = ok and not (use_obi == 1 and obi_bs[u]) and not (use_fi == 1 and fi_bs[u])
        b_size = min(Q, LQ - q)
        if long_only == 1:
            s_size = min(Q, q)
        else:
            s_size = min(Q, LQ + q)
        pb = bid[u]
        pa = ask[u]
        sell_a4 = theta4 * asz[u]
        if naive == 1 and sum_q > 0:
            den = TICK * sum_q
            target = ((sum_pq + TICK * sum_q + den - 1) // den) * TICK
            if target > pa:
                pa = target
                sell_a4 = -1
        # buy side
        if mode == 1 and buy_ok and b_size > 0 and b_rem > 0 and b_px == pb:
            if b_rem > b_size:
                b_rem = b_size
        else:
            b_rem = 0
            b_slot = _close(orders, b_slot, u)
            if buy_ok and b_size > 0:
                b_px, b_rem, b_a4, b_dec, b_hit, b_slot = pb, b_size, theta4 * bsz[u], u, 0, no
                no = _order(orders, no, u, 1, b_px, b_rem, b_a4)
                st[S_NINT_B] += 1
        # sell side
        if mode == 1 and sell_ok and s_size > 0 and a_rem > 0 and a_px == pa:
            if a_rem > s_size:
                a_rem = s_size
        else:
            a_rem = 0
            a_slot = _close(orders, a_slot, u)
            if sell_ok and s_size > 0:
                a_px, a_rem, a_a4, a_dec, a_hit, a_slot = pa, s_size, sell_a4, u, 0, no
                no = _order(orders, no, u, -1, a_px, a_rem, a_a4)
                st[S_NINT_S] += 1
    return nf, no


def simulate(b, f, params):
    """research 12 simulator.simulate with the K kernel."""
    fills = np.zeros((MAX_ROWS, len(FILL_COLS)), dtype=np.int64)
    orders = np.zeros((MAX_ROWS, len(ORDER_COLS)), dtype=np.int64)
    st = np.zeros(N_STATS, dtype=np.int64)
    lots = np.zeros((MAX_ROWS, 2), dtype=np.int64)
    nf, no = kernel(*[b[k] for k in simulator.KERNEL_BARS], *[f[k] for k in simulator.KERNEL_FEATS], params, fills, orders,
                    st, lots, K)
    return fills[:nf].copy(), orders[:no].copy(), st


def params_x(n, close_bar, gate):
    p = B0.kernel_params("Q", close_bar, gate)
    p[P_THETA4] = n
    return p


# ---------------------------------------------------------------- workers
_SP = None


def _init():
    global _SP
    _SP = loader.Split("dev", "stocks")


def _day_stats(task):
    sym, day, close_bar = task
    return sym, day, features.day_stats(_SP.load(sym, day), close_bar)


def _row(sym, day, run, st, acc):
    return {"sym": sym, "date": day, "run": run, "skip": int(st[S_SKIP]), **{k: int(acc[k]) for k in KEYS}}


def _sweep(task):
    """All x plus research 12 Q, Q50, T for one symbol-day; flags whether x = 1 / 0.5 fill rows equal Q / Q50 bit for bit."""
    sym, day, close_bar, gate = task
    b = _SP.load(sym, day)
    f = features.compute(b)
    d0, df = B0.d_bounds(close_bar)
    rows, kept = [], {}
    for n in N_GRID:
        fills, _, st = simulate(b, f, params_x(n, close_bar, gate))
        rows.append(_row(sym, day, f"x{n}", st, accounting.account(fills, f["mid2"], f["mid_ok"], close_bar, d0, df)))
        kept[n] = fills
    same = {}
    for m in REF_RUNS:
        fills, _, st = simulator.simulate(b, f, B0.kernel_params(m, close_bar, gate))
        rows.append(_row(sym, day, f"r12_{m}", st, accounting.account(fills, f["mid2"], f["mid_ok"], close_bar, d0, df)))
        if m == "Q":
            same["x1_eq_Q"] = bool(np.array_equal(kept[10], fills))
        if m == "Q50":
            same["x05_eq_Q50"] = bool(np.array_equal(kept[5], fills))
    return rows, same


# ---------------------------------------------------------------- aggregation
def daily(runs, run, scn, obs):
    """research 12 select_candidate.daily: daily portfolio net in dollars over obs (days with no row are 0)."""
    r = runs[runs.run == run]
    s = r.assign(net=accounting.net_arrays(r, scn)).groupby("date").net.sum()
    return np.array([float(s.get(d, 0)) / 1e4 for d in obs])


def metrics(runs, run, scn, obs):
    r = runs[runs.run == run]
    x = daily(runs, run, scn, obs)
    lim = int((r.sh_lim_buy + r.sh_lim_sell).sum())
    mkt = int((r.sh_mkt_buy + r.sh_mkt_sell).sum())
    net = float(x.sum())
    return {"net_usd": net, "pnl_per_share_c": net / (lim + mkt) * 100 if lim + mkt else float("nan"),
            "sharpe": sharpe(x), "limit_fills": int(r.n_fill_ev.sum()), "limit_shares": lim, "market_shares": mkt,
            "loss_day_frac": float(np.mean(x < 0))}


def crossing(xs, ys, level):
    """DEVIATIONS16 2: first crossing from >= level to < level going up from x = 0 (linear), plus every sign change."""
    ys = np.asarray(ys, dtype=float)
    if ys[0] < level:
        return {"status": "below_at_0", "x": None, "all": []}
    allc = []
    for i in range(len(xs) - 1):
        a, b = ys[i] - level, ys[i + 1] - level
        if (a >= 0 > b) or (a < 0 <= b):
            allc.append({"x": float(xs[i] + (xs[i + 1] - xs[i]) * a / (a - b)), "dir": "down" if a >= 0 else "up"})
    if not allc:
        return {"status": "above_at_1", "x": None, "all": []}
    return {"status": "crossed", "x": allc[0]["x"], "all": allc}


# ---------------------------------------------------------------- hand trace (independent of the engine: shares, floats)
def trace_order(b, side, price, size, dec, end, x):
    """Bar-by-bar Q(x) for one resting limit order, written from the rule text in shares (not the engine's integers)."""
    q = x * float(b["bsz"][dec] if side > 0 else b["asz"][dec])
    rem, out = size, []
    for u in range(dec + 1, end + 1):
        if side > 0:
            crossed = 0 < b["ask_min"][u] <= price
            credit = max(int(b["vol_bid"][u] - b["vol_bid_odd"][u]), 0)
            away = 0 < b["bid_max"][u] < price
            at = b["bid_min"][u] == price and b["bid_max"][u] == price
            lo, hi = b["bid_min"][u], b["bid_max"][u]
        else:
            crossed = b["bid_max"][u] > 0 and b["bid_max"][u] >= price
            credit = max(int(b["vol_ask"][u] - b["vol_ask_odd"][u]), 0)
            away = b["ask_min"][u] > price
            at = b["ask_min"][u] == price and b["ask_max"][u] == price
            lo, hi = b["ask_min"][u], b["ask_max"][u]
        q0 = q
        if crossed:
            case, fill = "cross", rem
        elif away:
            case, fill, q = "Q-2", min(rem, credit), 0.0
        elif at:
            case = "Q-3"
            if credit > q:
                fill, q = min(rem, int(math.floor(credit - q))), 0.0
            else:
                fill, q = 0, q - credit
        else:
            case, fill = "Q-4", 0
        out.append({"u": u, "time": hhmmss(u), "case": case, "mkt_lo": int(lo), "mkt_hi": int(hi), "credit": credit,
                    "queue_before": q0, "queue_after": q, "fill": int(fill), "rem_after": int(rem - fill)})
        rem -= fill
        if rem == 0:
            break
    return out


def hhmmss(u):
    s = 9 * 3600 + 30 * 60 + int(u)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def hand_trace(sp, cal, gated):
    """DEVIATIONS16 2: rng 16 draws a gated symbol-day with >= 2 Q-3 limit fills at x = 0.5, then 2 of them."""
    rng = np.random.default_rng(16)
    for i in rng.permutation(len(gated)):
        sym, day = gated[i]
        b = sp.load(sym, day)
        f = features.compute(b)
        fills, orders, _ = simulate(b, f, params_x(TRACE_X, cal.close[day], True))
        lim = fills[fills[:, 2] == 0]
        cand, n_orders_checked, all_match = [], 0, True
        for o in orders:
            dec, side, price, size, a, end = (int(v) for v in o)
            fo = lim[(lim[:, 6] == dec) & (lim[:, 1] == side)]
            if not len(fo):
                continue
            n_orders_checked += 1
            tr = trace_order(b, side, price, size, dec, end, TRACE_X / 10)
            tf = [(r["u"], r["fill"]) for r in tr if r["fill"] > 0]
            ef = [(int(r[0]), int(r[3])) for r in fo]
            all_match &= tf == ef and a == TRACE_X * int(b["bsz"][dec] if side > 0 else b["asz"][dec])
            for r in fo:
                last = next((t for t in tr if t["u"] == int(r[0])), None)
                if last is not None and last["case"] == "Q-3":
                    cand.append((dec, side, price, size, end, int(r[0]), int(r[3]), tr))
        if len(cand) < 2:
            continue
        pick = sorted(rng.choice(len(cand), 2, replace=False))
        out = {"sym": sym, "date": day, "orders_with_fills_traced": n_orders_checked,
               "trace_equals_engine_all_orders": bool(all_match), "q3_fills_that_day": len(cand), "fills": []}
        for j in pick:
            dec, side, price, size, end, u_fill, qty, tr = cand[j]
            disp = int(b["bsz"][dec] if side > 0 else b["asz"][dec])
            rows = [t for t in tr if t["u"] <= u_fill]
            out["fills"].append({"side": "buy" if side > 0 else "sell", "price": price / 1e4, "size": size,
                                 "dec_time": hhmmss(dec), "displayed_at_dec": disp, "queue_start": TRACE_X / 10 * disp,
                                 "engine_fill_bar": hhmmss(u_fill), "engine_fill_qty": qty,
                                 "trace_fill_qty": rows[-1]["fill"], "match": rows[-1]["fill"] == qty, "rows": rows})
        out["pass"] = bool(all_match and all(x["match"] for x in out["fills"]))
        return out
    return {"pass": False, "error": "no symbol-day with 2 Q-3 fills"}


# ---------------------------------------------------------------- run
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def run(workers):
    t0 = time.time()
    rule_sha = sha(os.path.join(HERE, "QUEUE16.md"))
    with open(os.path.join(HERE, "QUEUE16.sha256")) as fh:
        saved = fh.read().split()[0]
    if rule_sha != saved:
        raise SystemExit("QUEUE16.md differs from QUEUE16.sha256")
    cal = session.Calendar()
    sp = loader.Split("dev", "stocks")
    avail = sp.available()
    tasks = session.tasks(cal, "dev", "stocks", avail)
    obs = session.observation_days(cal, "dev", avail)
    assert all(d <= "20201231" for _, d in tasks), "date after the dev split"
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init) as pool:
        ds = pool.map(_day_stats, [(s, d, cal.close[d]) for s, d in tasks], chunksize=64)
    dv = {(s, d): int(x["dv"]) for s, d, x in ds}
    zero = {(s, d) for s, d, x in ds if x["no_data"]}
    gates = {(s, d): session.gate(cal, s, d, dv, zero, avail) for s, d in tasks}
    print(f"pass 0: {len(tasks)} symbol-days, {sum(gates.values())} gated, {len(obs)} observation days, "
          f"{time.time() - t0:.0f} s", flush=True)
    with ctx.Pool(workers, initializer=_init) as pool:
        res = pool.map(_sweep, [(s, d, cal.close[d], gates[(s, d)]) for s, d in tasks], chunksize=16)
    runs = pd.DataFrame([r for rows, _ in res for r in rows]).sort_values(["run", "sym", "date"], kind="stable")
    same = {k: int(sum(not s[k] for _, s in res)) for k in ("x1_eq_Q", "x05_eq_Q50")}
    print(f"pass 1: {len(runs)} rows, {time.time() - t0:.0f} s", flush=True)
    os.makedirs(LOCAL, exist_ok=True)
    runs.to_csv(os.path.join(LOCAL, "queue16_runs.csv.gz"), index=False)

    # checks (QUEUE16 5 and DEVIATIONS16 2)
    ref = pd.read_csv(DEV_DAILY, dtype={"date": str})
    with open(S12_SUMMARY) as fh:
        s12 = json.load(fh)["columns"]["B0_Q"]
    chk = {"queue16_sha256": rule_sha}
    ref_dates = ref.date.tolist()
    chk["dates_equal_dev_daily"] = ref_dates == obs
    for run_, col, name in (("x10", "B0|Q|R", "x1_daily_vs_B0_Q_R"), ("x5", "B0|Q50|R", "x05_daily_vs_B0_Q50_R"),
                            ("r12_T", "B0|T|R", "r12T_daily_vs_B0_T_R")):
        d = np.abs(daily(runs, run_, "R", obs) - ref[col].to_numpy())
        chk[name] = {"max_abs_diff_usd": float(d.max()), "days_over_0.01": int((d > TOL_USD).sum()),
                     "pass": bool(d.max() <= TOL_USD)}
    m1 = metrics(runs, "x10", "R", obs)
    chk["x1_shares_vs_s12"] = {"limit": m1["limit_shares"], "market": m1["market_shares"],
                               "s12_limit": s12["limit_shares"], "s12_market": s12["market_shares"],
                               "pass": m1["limit_shares"] == s12["limit_shares"] and m1["market_shares"] == s12["market_shares"]}
    chk["symbol_days_fills_differ"] = same
    gated = sorted(k for k, g in gates.items() if g)
    chk["hand_trace"] = hand_trace(sp, cal, gated)
    chk["no_date_after_2020"] = bool(runs.date.max() <= "20201231")
    stop = not (chk["dates_equal_dev_daily"] and chk["x1_daily_vs_B0_Q_R"]["pass"] and chk["x1_shares_vs_s12"]["pass"]
                and chk["hand_trace"]["pass"] and chk["no_date_after_2020"])
    os.makedirs(RES, exist_ok=True)
    if stop:
        with open(os.path.join(RES, "queue16_checks_FAILED.json"), "w") as fh:
            json.dump(chk, fh, indent=1, default=str)
        raise SystemExit("a QUEUE16 5 check failed: results/queue16_checks_FAILED.json (no sweep written)")

    # sweep
    xs = [n / 10 for n in N_GRID]
    rows = []
    for n in N_GRID:
        for scn in SCN:
            rows.append({"x": n / 10, "scenario": scn, **metrics(runs, f"x{n}", scn, obs)})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(RES, "queue16_sweep.csv"), index=False, float_format="%.6f")
    dd = pd.DataFrame({"date": obs, **{f"x{n / 10:.1f}": daily(runs, f"x{n}", "R", obs) for n in N_GRID}})
    dd.to_csv(os.path.join(RES, "queue16_daily.csv"), index=False, float_format="%.4f")
    cross = {}
    for scn in SCN:
        s = sweep[sweep.scenario == scn]
        cross[scn] = {"x_star_breakeven": crossing(xs, s.pnl_per_share_c, 0.0),
                      "x_2star_sharpe": crossing(xs, s.sharpe, SR_LINE)}
    tref = {scn: metrics(runs, "r12_T", scn, obs) for scn in SCN}
    out = {"rule": "QUEUE16.md", "rule_sha256": rule_sha, "trials_added": 22, "trials_cumulative": 714,
           "verdict": "none (diagnostic)", "K": K, "x_grid": xs, "observation_days": len(obs),
           "symbol_days": len(tasks), "gated_symbol_days": len(gated), "sweep": rows, "crossings": cross,
           "T_reference": tref, "checks": chk, "costs_1e4_usd_per_share": {s: COST[s] for s in SCN},
           "runtime_s": round(time.time() - t0, 1), "python": sys.executable, "numpy": np.__version__,
           "pandas": pd.__version__, "written": dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(RES, "queue16_result.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    report(out, sweep)
    print(open(os.path.join(RES, "queue16_report.md")).read())


# ---------------------------------------------------------------- report
def _cross_text(c, what, thr):
    if c["status"] == "below_at_0":
        return f"줄 맨 앞에서도 {what} 아님"
    if c["status"] == "above_at_1":
        return f"x = 1에서도 {thr} 이상"
    extra = "" if len(c["all"]) == 1 else " (교차 " + ", ".join(f"{a['x']:.3f}{'↓' if a['dir'] == 'down' else '↑'}"
                                                               for a in c["all"]) + ")"
    return f"x = {c['x']:.3f}{extra}"


def report(out, sweep):
    c, chk = out["crossings"], out["checks"]
    R, Rp = sweep[sweep.scenario == "R"].reset_index(drop=True), sweep[sweep.scenario == "R+"].reset_index(drop=True)
    L = [f"**x\\*(본전, R) = {_cross_text(c['R']['x_star_breakeven'], '본전', '0¢')} · "
         f"x\\*\\*(샤프 0.5, R) = {_cross_text(c['R']['x_2star_sharpe'], '합격선', '샤프 0.5')}** "
         f"(R+: x\\* = {_cross_text(c['R+']['x_star_breakeven'], '본전', '0¢')}, "
         f"x\\*\\* = {_cross_text(c['R+']['x_2star_sharpe'], '합격선', '샤프 0.5')})", "",
         "# 연구 16-B · 줄 위치에 따른 Market Maker 손익 (진단, 판정 없음)", "",
         f"- 규칙 QUEUE16.md sha256 `{out['rule_sha256']}` (2026-10-30 12:40 ET 작성, 계산 전)",
         f"- 시행 수 +22, 누적 {out['trials_cumulative']}",
         f"- 연구 12 B0 그대로, 줄 앞 물량 = x × 보이는 물량. 개발 구간 2019–2020, {out['observation_days']}일, "
         f"종목·날 {out['symbol_days']}개(게이트 통과 {out['gated_symbol_days']}개)", "",
         "## x별 결과", "",
         "| x | 주당 R (¢) | 주당 R+ (¢) | 샤프 R | 샤프 R+ | 순손익 R ($) | 순손익 R+ ($) | 지정가 체결 | 지정가 주식 | 시장가 주식 | 손실일 R |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i in range(len(R)):
        a, b = R.iloc[i], Rp.iloc[i]
        L.append(f"| {a.x:.1f} | {a.pnl_per_share_c:+.3f} | {b.pnl_per_share_c:+.3f} | {a.sharpe:+.2f} | {b.sharpe:+.2f} | "
                 f"{a.net_usd:+,.0f} | {b.net_usd:+,.0f} | {int(a.limit_fills):,} | {int(a.limit_shares):,} | "
                 f"{int(a.market_shares):,} | {a.loss_day_frac:.3f} |")
    t, q0, q1 = out["T_reference"]["R"], R.iloc[0], R.iloc[-1]
    avg = ", ".join(f"{a.x:.1f}: {a.limit_shares / a.limit_fills:.1f}" for a in R.itertuples())
    L += ["", f"- 줄 위치의 값 (x = 1 → 0): R {q0.pnl_per_share_c - q1.pnl_per_share_c:+.3f}¢/주, "
              f"R+ {Rp.iloc[0].pnl_per_share_c - Rp.iloc[-1].pnl_per_share_c:+.3f}¢/주. "
              f"Q(1) → T 차이 {t['pnl_per_share_c'] - q1.pnl_per_share_c:+.3f}¢ 중 줄 위치가 "
              f"{q0.pnl_per_share_c - q1.pnl_per_share_c:+.3f}¢, 나머지 {t['pnl_per_share_c'] - q0.pnl_per_share_c:+.3f}¢는 "
              f"T의 체결 가정(닿으면 전량 체결)에서 온다.",
          f"- 지정가 체결 수는 주문·초마다 1행이라 부분 체결이 많으면 늘어난다. 평균 체결 크기(주): {avg} "
          f"(0 < x < 1에서는 줄 앞 물량이 100주 단위가 아니어서 작은 부분 체결이 생김, DEVIATIONS16.md 3절)."]
    L += ["", "## T 모델과 Q(0)", "",
          "| 모델 | 주당 R (¢) | 샤프 R | 순손익 R ($) | 지정가 체결 | 지정가 주식 | 시장가 주식 |", "|---|---|---|---|---|---|---|",
          f"| T (연구 12) | {t['pnl_per_share_c']:+.3f} | {t['sharpe']:+.2f} | {t['net_usd']:+,.0f} | {t['limit_fills']:,} | "
          f"{t['limit_shares']:,} | {t['market_shares']:,} |",
          f"| Q(0) | {q0.pnl_per_share_c:+.3f} | {q0.sharpe:+.2f} | {q0.net_usd:+,.0f} | {int(q0.limit_fills):,} | "
          f"{int(q0.limit_shares):,} | {int(q0.market_shares):,} |",
          f"| 차이 (T − Q(0)) | {t['pnl_per_share_c'] - q0.pnl_per_share_c:+.3f} | {t['sharpe'] - q0.sharpe:+.2f} | "
          f"{t['net_usd'] - q0.net_usd:+,.0f} | {t['limit_fills'] - int(q0.limit_fills):+,} | "
          f"{t['limit_shares'] - int(q0.limit_shares):+,} | {t['market_shares'] - int(q0.market_shares):+,} |", "",
          "- T는 거래 가격이 우리 호가에 닿기만 하면 남은 주문 전체가 체결된다.",
          "- Q(0)은 줄 맨 앞이어도 그 초의 정규 단위 체결량만큼만 채워진다. 그 초 안에 시장 호가가 움직이면(Q-4) 체결이 없고, 단주는 세지 않는다.", "",
          "## 엔진의 취소 처리 (코드 확인)", "",
          "- **취소:** 보이는 물량이 취소로 줄어도 우리 앞 물량은 줄지 않는다.",
          "- **앞 물량이 줄어드는 경우:** 그 초 내내 시장 호가가 우리 가격일 때(Q-3), 그 가격의 정규 단위 체결량만큼만 줄어든다.",
          "- **앞 물량이 0이 되는 경우:** 시장 호가가 그 초 내내 우리 가격 밖에 있을 때(Q-2)다.",
          "- **재호가:** B0은 5초마다 주문을 취소하고 다시 내므로, 그때마다 x × 그 시점 보이는 물량에서 다시 줄을 선다.", "",
          "## 대조 점검", ""]
    for name in ("x1_daily_vs_B0_Q_R", "x05_daily_vs_B0_Q50_R", "r12T_daily_vs_B0_T_R"):
        v = chk[name]
        L.append(f"- {name}: 최대 차이 ${v['max_abs_diff_usd']:.4f}, $0.01 초과 {v['days_over_0.01']}일 → "
                 f"{'통과' if v['pass'] else '실패'}")
    s = chk["x1_shares_vs_s12"]
    L += [f"- x = 1 주식 수: 지정가 {s['limit']:,} (s12 {s['s12_limit']:,}), 시장가 {s['market']:,} (s12 {s['s12_market']:,}) → "
          f"{'통과' if s['pass'] else '실패'}",
          f"- 체결 행 비트 비교 (종목·날 중 다른 수): x=1↔Q {chk['symbol_days_fills_differ']['x1_eq_Q']}, "
          f"x=0.5↔Q50 {chk['symbol_days_fills_differ']['x05_eq_Q50']}",
          f"- 관측일이 dev_daily.csv와 같음: {chk['dates_equal_dev_daily']}, 2021년 이후 날짜 없음: {chk['no_date_after_2020']}"]
    h = chk["hand_trace"]
    L += ["", f"## 손 추적 (x = 0.5, {h['sym']} {h['date']})", "",
          f"- 그날 체결이 있는 주문 {h['orders_with_fills_traced']}개를 엔진과 별도로 주 단위로 다시 계산했다. "
          f"모든 체결 초와 수량이 엔진과 같음: {h['trace_equals_engine_all_orders']}",
          f"- 그날 큐 소진형(Q-3) 체결은 {h['q3_fills_that_day']}건이고, 그중 2건을 seed 16으로 뽑았다.", ""]
    for k, fl in enumerate(h["fills"], 1):
        side_k = "매수" if fl["side"] == "buy" else "매도"
        L += [f"**체결 {k}:** {side_k} {fl['size']}주 @ ${fl['price']:.2f}, {fl['dec_time']}에 주문. "
              f"그때 보이는 물량 {fl['displayed_at_dec']:,}주 → 줄 앞 {fl['queue_start']:,.1f}주에서 시작.", "",
              "| 초 | 경우 | 그 초 시장 호가 (최저–최고) | 그 가격 정규 단위 체결량 | 줄 앞 (전 → 후) | 우리 체결 |", "|---|---|---|---|---|---|"]
        for r in fl["rows"]:
            L.append(f"| {r['time']} | {r['case']} | ${r['mkt_lo'] / 1e4:.2f}–${r['mkt_hi'] / 1e4:.2f} | {r['credit']:,} | "
                     f"{r['queue_before']:,.1f} → {r['queue_after']:,.1f} | {r['fill']} |")
        L += ["", f"엔진 체결: {fl['engine_fill_bar']}에 {fl['engine_fill_qty']}주, 손 계산 {fl['trace_fill_qty']}주 → "
              f"{'일치' if fl['match'] else '불일치'}", ""]
    L += ["## 비고", "",
          "- 엔진은 연구 12 kernel·fill 함수를 queue16.py에 복사하고 줄 단위 K(1/10주)만 추가했다. 연구 12 파일은 바꾸지 않았다(DEVIATIONS16.md 2절).",
          "- R: 시장가 0.3¢/주, 지정가 0. R+: 같은 체결에 지정가 리베이트 0.2¢/주.",
          "- 주당 손익 = 순손익 / (지정가 + 시장가 주식 수). 샤프 = 일별 순손익 평균 / 표준편차 × √252 (505일, 거래 없는 날 0).",
          f"- 실행 {out['runtime_s']} s, {out['python']}"]
    with open(os.path.join(RES, "queue16_report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a.workers)
