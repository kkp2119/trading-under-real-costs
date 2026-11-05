"""Research 15 copy of research12/simulator.py (changed parts marked "R15"): per-side allow arrays in the kernel
(allow_b / allow_s, all True = research 12 B0) and taker_kernel for policy P3. Symbol-day kernel (SPEC 3, 4, 7; E1-05..E1-30, E1-50..E1-61). One pass over bars d0 .. d_f + 1 in the order of
E1-50: fills of bar u -> stale timer -> volatility halt -> decision (stale liquidation, quotes) or the d_f flatten.
Integer only; writes fill and order rows into preallocated arrays and counters into `st` (config S_* indices)."""
import numpy as np

from config15 import (FILL_COLS,                                                  # R15
                       MAX_ROWS, N_STATS, ORDER_COLS, Q, S_MAXQ, S_MKT_OVER, S_MKT_STALEQ, S_NDEC, S_NDEC_QB,
                    S_NINT_B, S_NINT_S, S_NINTF_B, S_NINTF_S, S_NMKT, S_NSTALE, S_QDF, S_SKIP, S_SUMQ, TICK)
from fillmodel15 import fill_buy, fill_sell, jit, market_price                     # R15

KERNEL_BARS = ("bid", "ask", "bsz", "asz", "bid_max", "bid_min", "ask_max", "ask_min", "vol_bid", "vol_ask", "vol_bid_odd",
               "vol_ask_odd", "px_min", "px_max")
KERNEL_FEATS = ("mid2", "qbase", "obi_bb", "obi_bs", "fi_bb", "fi_bs", "gcnt", "halt_trig", "stale_q")


@jit
def _push(fills, nf, u, side, kind, qty, price, ref, dec):
    fills[nf, 0] = u
    fills[nf, 1] = side
    fills[nf, 2] = kind
    fills[nf, 3] = qty
    fills[nf, 4] = price
    fills[nf, 5] = ref
    fills[nf, 6] = dec
    return nf + 1


@jit
def _order(orders, no, dec, side, price, size, a4):
    orders[no, 0] = dec
    orders[no, 1] = side
    orders[no, 2] = price
    orders[no, 3] = size
    orders[no, 4] = a4
    orders[no, 5] = -1
    return no + 1


@jit
def _close(orders, slot, u):
    """The order in row `slot` rests through bar u (cancel or replacement effective from u + 1)."""
    if slot >= 0:
        orders[slot, 5] = u
    return -1


@jit
def kernel(bid, ask, bsz, asz, bid_max, bid_min, ask_max, ask_min, vol_bid, vol_ask, vol_bid_odd, vol_ask_odd, px_min, px_max,
           mid2, qbase, obi_bb, obi_bs, fi_bb, fi_bs, gcnt, halt_trig, stale_q, p, fills, orders, st, lots, allow_b, allow_s,
           vopt):   # R15: allow_b, allow_s; vopt = [keep, improve, exit improve] (R1, all 0 = B0)
    for i in range(st.shape[0]):
        st[i] = 0
    mode, delta, L, stale_s, long_only = p[0], p[1], p[2], p[3], p[4]
    keep, impr, ex_impr = vopt[0], vopt[1], vopt[2]                  # R15 R1
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
            f, b_a4 = fill_buy(kind, b_px, b_rem, b_a4, u, ask_min, bid_max, bid_min, px_min, vol_bid, vol_bid_odd)
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
            f, a_a4 = fill_sell(kind, a_px, a_rem, a_a4, u, bid_max, ask_min, ask_max, px_max, vol_ask, vol_ask_odd)
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
        buy_ok = ok and not (use_obi == 1 and obi_bb[u]) and not (use_fi == 1 and fi_bb[u]) and allow_b[u]     # R15 allow_b
        sell_ok = ok and not (use_obi == 1 and obi_bs[u]) and not (use_fi == 1 and fi_bs[u]) and allow_s[u]    # R15 allow_s
        b_size = min(Q, LQ - q)
        if long_only == 1:
            s_size = min(Q, q)
        else:
            s_size = min(Q, LQ + q)
        pb = bid[u]
        pa = ask[u]
        sell_a4 = theta4 * asz[u]
        buy_a4 = theta4 * bsz[u]                                     # R15 R1 (B0: theta4 x bsz as before)
        if impr == 1 and ask[u] - bid[u] >= 3 * TICK:                # R15 R1 V2: one tick inside, alone at that price
            pb, buy_a4 = bid[u] + TICK, 0
            pa, sell_a4 = ask[u] - TICK, 0
        if ex_impr == 1 and ask[u] - bid[u] >= 2 * TICK:             # R15 R1 V4: exit side one tick inside
            if q > 0:
                pa, sell_a4 = ask[u] - TICK, 0
            elif q < 0:
                pb, buy_a4 = bid[u] + TICK, 0
        if naive == 1 and sum_q > 0:
            den = TICK * sum_q
            target = ((sum_pq + TICK * sum_q + den - 1) // den) * TICK
            if target > pa:
                pa = target
                sell_a4 = -1
        # buy side
        if (mode == 1 or keep == 1) and buy_ok and b_size > 0 and b_rem > 0 and b_px == pb:     # R15 R1 V1 keep
            if b_rem > b_size:
                b_rem = b_size
        else:
            b_rem = 0
            b_slot = _close(orders, b_slot, u)
            if buy_ok and b_size > 0:
                b_px, b_rem, b_a4, b_dec, b_hit, b_slot = pb, b_size, buy_a4, u, 0, no                 # R15 R1 buy_a4
                no = _order(orders, no, u, 1, b_px, b_rem, b_a4)
                st[S_NINT_B] += 1
        # sell side
        if (mode == 1 or keep == 1) and sell_ok and s_size > 0 and a_rem > 0 and a_px == pa:    # R15 R1 V1 keep
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


def simulate(b, f, params, allow_b=None, allow_s=None, vopt=None):
    """b: 21 bar columns (int64), f: features.compute(b), params: Config.kernel_params(...). Returns (fills, orders, st)
    as int64 arrays (fills columns FILL_COLS, orders ORDER_COLS, st indexed by config S_*). R15: allow_b / allow_s
    (bool per bar; None = all True, the research 12 kernel) gate the buy / sell quote of each decision."""
    n = len(b["bid"])                                                                                     # R15
    allow_b = np.ones(n, dtype=np.bool_) if allow_b is None else np.asarray(allow_b, dtype=np.bool_)
    allow_s = np.ones(n, dtype=np.bool_) if allow_s is None else np.asarray(allow_s, dtype=np.bool_)
    vopt = np.zeros(3, dtype=np.int64) if vopt is None else np.asarray(vopt, dtype=np.int64)          # R15 R1
    fills = np.zeros((MAX_ROWS, len(FILL_COLS)), dtype=np.int64)
    orders = np.zeros((MAX_ROWS, len(ORDER_COLS)), dtype=np.int64)
    st = np.zeros(N_STATS, dtype=np.int64)
    lots = np.zeros((MAX_ROWS, 2), dtype=np.int64)
    nf, no = kernel(*[b[k] for k in KERNEL_BARS], *[f[k] for k in KERNEL_FEATS], params, fills, orders, st, lots,
                    allow_b, allow_s, vopt)                                                               # R15
    return fills[:nf].copy(), orders[:no].copy(), st


@jit
def taker_kernel(bid, ask, bid_min, ask_max, qbase, mid2, p, h, tdir, fills, st):
    """R15 policy P3 (no limit orders). Decision bars as in kernel (d0 + k * delta < d_f) with the same day checks (gate,
    E1-26). At a decision bar u passing the B0 quote gate (qbase and spread >= min spread), tdir[u] = +1 / -1 enters 100
    shares at market when |q| stays <= L * Q; each entry leaves at market at the decision bar u + h. Exits, the stale
    liquidation (E1-52, 100 shares, oldest open entries reduced first) and the entry of one bar net into one market
    order, filled in the next bar at market_price; d_f flattens everything. Returns the number of fill rows."""
    for i in range(st.shape[0]):
        st[i] = 0
    delta, L, stale_s, min_spr = p[1], p[2], p[3], p[8] * TICK
    d0, df, gate = p[10], p[11], p[15]
    if gate == 0:
        st[S_SKIP] = 1
        return 0
    if not (bid[d0] > 0 and ask[d0] > 0 and ask[d0] >= bid[d0]):
        st[S_SKIP] = 2
        return 0
    LQ = L * Q
    e_bar = np.zeros(64, dtype=np.int64)                              # open entries, FIFO
    e_side = np.zeros(64, dtype=np.int64)
    e_qty = np.zeros(64, dtype=np.int64)
    head, tail = 0, 0
    q, hit, nf = 0, -1, 0
    m_qty, m_side, m_dec = 0, 0, -1
    for u in range(d0, df + 2):
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
        if aq == LQ:
            if hit < 0:
                hit = u
        else:
            hit = -1
        if u == df:
            st[S_QDF] = q
            if q != 0:
                m_qty = aq
                m_side = -1 if q > 0 else 1
                m_dec = u
                st[S_NMKT] += 1
                q = 0
            head = tail
            continue
        if (u - d0) % delta != 0:
            continue
        st[S_NDEC] += 1
        net = 0
        while head < tail and e_bar[head % 64] + h <= u:               # exits due at this bar
            net -= e_side[head % 64] * e_qty[head % 64]
            head += 1
        qn = q + net
        aqn = qn if qn >= 0 else -qn
        if stale_s >= 0 and hit >= 0 and u - hit >= stale_s and aqn == LQ:
            sgn = 1 if qn > 0 else -1
            net -= sgn * Q
            left = Q
            k = head
            while left > 0 and k < tail:                              # reduce the oldest entries on the held side
                j = k % 64
                if e_side[j] == sgn and e_qty[j] > 0:
                    take = min(left, e_qty[j])
                    e_qty[j] -= take
                    left -= take
                k += 1
            hit = -1
            st[S_NSTALE] += 1
        qn = q + net
        if tdir[u] != 0 and qbase[u] and ask[u] - bid[u] >= min_spr:
            nq = qn + tdir[u] * Q
            if -LQ <= nq <= LQ:
                net += tdir[u] * Q
                j = tail % 64
                e_bar[j], e_side[j], e_qty[j] = u, tdir[u], Q
                tail += 1
                if tdir[u] > 0:
                    st[S_NINT_B] += 1
                else:
                    st[S_NINT_S] += 1
        if net != 0:
            m_qty = net if net > 0 else -net
            m_side = 1 if net > 0 else -1
            m_dec = u
            st[S_NMKT] += 1
            q += net
    return nf


def simulate_taker(b, f, params, h, tdir):
    """R15: (fills, st) of taker_kernel; tdir int64 per bar (+1 buy, -1 sell, 0 none) read at decision bars only."""
    fills = np.zeros((MAX_ROWS, len(FILL_COLS)), dtype=np.int64)
    st = np.zeros(N_STATS, dtype=np.int64)
    nf = taker_kernel(b["bid"], b["ask"], b["bid_min"], b["ask_max"], f["qbase"], f["mid2"], params, h,
                      np.asarray(tdir, dtype=np.int64), fills, st)
    return fills[:nf].copy(), st


@jit
def replay(orders, n, kind, theta4, bsz, asz, ask_min, bid_max, bid_min, ask_max, px_min, px_max, vol_bid, vol_ask,
           vol_bid_odd, vol_ask_odd, out):
    """E1-60: each logged order with its own fixed size and life (dec + 1 .. end) through fill model (kind, theta4),
    independent of inventory; out[i] = shares filled. An order placed above the ask (a4 < 0) keeps an infinite queue."""
    for i in range(n):
        dec, side, price, size, a4m, end = orders[i, 0], orders[i, 1], orders[i, 2], orders[i, 3], orders[i, 4], orders[i, 5]
        a4 = -1 if a4m < 0 else theta4 * (bsz[dec] if side > 0 else asz[dec])
        rem = size
        for u in range(dec + 1, end + 1):
            if side > 0:
                f, a4 = fill_buy(kind, price, rem, a4, u, ask_min, bid_max, bid_min, px_min, vol_bid, vol_bid_odd)
            else:
                f, a4 = fill_sell(kind, price, rem, a4, u, bid_max, ask_min, ask_max, px_max, vol_ask, vol_ask_odd)
            rem -= f
            if rem == 0:
                break
        out[i] = size - rem


def replay_all(b, orders):
    """{model: shares filled per order} for every model of config.MODELS (G1.2 nesting check input)."""
    from config import MODEL_CODE
    res = {}
    for m, (kind, theta4) in MODEL_CODE.items():
        out = np.zeros(len(orders), dtype=np.int64)
        replay(orders, len(orders), kind, theta4, b["bsz"], b["asz"], b["ask_min"], b["bid_max"], b["bid_min"], b["ask_max"],
               b["px_min"], b["px_max"], b["vol_bid"], b["vol_ask"], b["vol_bid_odd"], b["vol_ask_odd"], out)
        res[m] = out
    return res


def nested(rep):
    """Per order S <= Q <= Q50 <= Q25 <= T (E0-6 (b))."""
    return bool(np.all(rep["S"] <= rep["Q"]) and np.all(rep["Q"] <= rep["Q50"]) and np.all(rep["Q50"] <= rep["Q25"])
                and np.all(rep["Q25"] <= rep["T"]))
