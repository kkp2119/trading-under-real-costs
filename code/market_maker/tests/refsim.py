"""Research 12 independent reference simulator (slow, literal, one bar at a time).

Written ONLY from research12/SPEC.md (sections 3, 4, 5 [A_naive], 7, 8) and research12/SPEC_AMENDMENTS_draft_E1.md
(Kyle decisions, E1-01..E1-49, implementation supplement E1-50..E1-61), plus the bar definitions in bars1s/bars.py.
It does not import or mirror the fast kernel; the two are compared bit for bit on synthetic symbol-days.

Units: prices in 1e-4 dollars (1 tick = 100), sizes in shares, Q (lot) = 100 shares. All decisions and fills use exact
Python integers (no floats). Queue ahead is kept in quarter shares A4 = 4*theta*size (E1-27, E1-56); None = infinite
(A_naive sell above the ask, E1-55), reported as -1 in the order log.

    simulate_ref(bars, close_bar, gate, cfg, model) -> dict   (see the interface at the bottom of simulate_ref)
"""
import numpy as np

COLS = ("bid", "ask", "bsz", "asz", "bid_max", "bid_min", "ask_max", "ask_min", "n_upd", "n_xl",
        "vol", "ntr", "px_last", "px_min", "px_max", "vol_bid", "vol_ask", "vol_in", "vol_unc",
        "vol_bid_odd", "vol_ask_odd")
QLOT = 100                       # SPEC 4: Q = 100 shares
TICK = 100                       # SPEC 4: 1 tick = 100 (1e-4 dollars)
MAX_SPREAD = 12 * TICK           # SPEC 4 / E1-53: ask - bid <= 12 ticks
HALT_FIRST_BAR = 1800            # E1-12: first possible halt evaluation at bar 1,800
HALT_LEN = 60                    # SPEC 4: 60 seconds, E1-51 / E1-15
THETA4 = {"S": 4, "T": 4, "Q": 4, "Q50": 2, "Q25": 1}   # 4*theta (SPEC 7; S/T do not use it, log with theta = 1)


class _Order:
    """One live limit order (SPEC 7). a4 = queue ahead in quarter shares, None = infinite (E1-55)."""
    __slots__ = ("dec", "side", "price", "rem", "a4", "filled")

    def __init__(self, dec, side, price, size, a4):
        self.dec, self.side, self.price, self.rem, self.a4, self.filled = dec, side, price, size, a4, False


def _session(window, close_bar):
    """SPEC 3 + E1-21 / E1-22: decision window [d0, df), flatten at df, filled at df + 1."""
    if window == 0:
        return 899, min(23099, close_bar - 301)          # 09:45:00 end .. 15:55:00 end
    return 1799, min(21599, close_bar - 301)             # 10:00:00 end .. 15:30:00 end


def simulate_ref(bars, close_bar, gate, cfg, model):
    b = {c: [int(x) for x in np.asarray(bars[c]).tolist()] for c in COLS}    # exact Python ints
    bid, ask, bsz, asz = b["bid"], b["ask"], b["bsz"], b["asz"]
    bid_max, bid_min, ask_max, ask_min = b["bid_max"], b["bid_min"], b["ask_max"], b["ask_min"]
    ntr, px_min, px_max = b["ntr"], b["px_min"], b["px_max"]
    vol_bid, vol_ask, vol_bid_odd, vol_ask_odd = b["vol_bid"], b["vol_ask"], b["vol_bid_odd"], b["vol_ask_odd"]
    nbar = len(bid)

    d0, df = _session(int(cfg["window"]), int(close_bar))
    out = {"skipped": 0, "d0": d0, "df": df, "fills": [], "orders": [], "q_df": 0,
           "n_dec": 0, "n_dec_qbase": 0, "n_int": (0, 0), "n_int_fill": (0, 0), "n_stale": 0}

    # E1-53: a symbol-day failing the liquidity gate is not simulated (skip code 1)
    if not gate:
        out["skipped"] = 1
        return out
    # SPEC 3 / E1-26: no valid state at this config's d0 (bid > 0, ask > 0, ask >= bid) -> skip code 2
    if not (bid[d0] > 0 and ask[d0] > 0 and ask[d0] >= bid[d0]):
        out["skipped"] = 2
        return out

    # ---- configuration (SPEC 5 grid dimensions; A_naive per SPEC 5 + E1-30) ----
    naive = bool(cfg.get("naive", False))
    mode = cfg["mode"]
    delta = int(cfg["delta"]) if mode == "reset" else 1            # event checks every bar (delta unused)
    L = 1 if naive else int(cfg["L"])                              # E1-30: A_naive L = 1
    long_only = True if naive else bool(cfg["long_only"])          # E1-30: A_naive long only
    stale_s = None if naive else cfg["stale_s"]                    # E1-30 / E1-52: A_naive has no stale liquidation
    use_obi, use_halt, use_fi = bool(cfg["obi"]), bool(cfg["halt"]), bool(cfg["fi"])
    min_spread = int(cfg["min_spread"])
    gcnt_m = int(cfg["gcnt_m"])
    th4 = THETA4[model]
    LQ = L * QLOT

    def is_decision(u):
        """E1-50: reset decides at d0 + k*delta, event at every bar of [d0, df) (caller ensures u < df)."""
        if mode == "reset":
            return (u - d0) % delta == 0
        return True

    # ---- volatility halt (SPEC 4, E1-12, E1-13): mid2 = bid + ask, valid 1-s changes only, integer test ----
    def mid_valid(u):
        return bid[u] > 0 and ask[u] > 0 and ask[u] >= bid[u]

    chg_ok = [0] * nbar       # 1 if both mid2[v] and mid2[v-1] are valid
    chg_sq = [0] * nbar       # (mid2[v] - mid2[v-1])^2 for a valid change, else 0
    if use_halt:
        for v in range(1, nbar):
            if mid_valid(v) and mid_valid(v - 1):
                dm = (bid[v] + ask[v]) - (bid[v - 1] + ask[v - 1])
                chg_ok[v] = 1
                chg_sq[v] = dm * dm

    def halt_trigger(u):
        if u < HALT_FIRST_BAR:                                     # E1-12: first 1,800 seconds not applied
            return False
        if not (mid_valid(u) and mid_valid(u - 15)):               # E1-13: both mids valid
            return False
        n = sum(chg_ok[u - 1799:u + 1])                            # E1-13: changes over [u - 1799, u]
        s2 = sum(chg_sq[u - 1799:u + 1])
        if n == 0:                                                 # E1-13: no valid change -> no trigger
            return False
        d15 = (bid[u] + ask[u]) - (bid[u - 15] + ask[u - 15])
        # |D15| >= 3 sigma1 sqrt(15)  <=>  D15^2 * n >= 135 * S2 ;  and >= 1 tick  <=> |D15 (in mid2)| >= 200
        return d15 * d15 * n >= 135 * s2 and abs(d15) >= 2 * TICK

    # ---- fill models (SPEC 7, E1-27, E1-28, E1-55, E1-61) ----
    def queue_credit(o, credit):
        """Q case 3: credit first reduces the queue ahead, the rest fills me (E1-27 quarter shares, floor)."""
        if o.a4 is None:                                           # E1-55: infinite queue ahead, no case-3 credit
            return 0
        if 4 * credit > o.a4:
            f = min(o.rem, (4 * credit - o.a4) // 4)
            o.a4 = 0
            return f
        o.a4 -= 4 * credit
        return 0

    def fill_qty(o, u):
        """Shares of order o filled in bar u (does not change o.rem; may change o.a4)."""
        P, r = o.price, o.rem
        if o.side == +1:                                           # buy at P_b
            cross = 0 < ask_min[u] <= P                            # E1-61: 0 < ask_min_u <= P_b
            if model == "S":
                return r if cross else 0                           # SPEC 7 model S
            if model == "T":
                touch = px_min[u] > 0 and px_min[u] <= P           # SPEC 7 model T
                return r if (cross or touch) else 0
            if cross:                                              # Q case 1: crossed, fill all remaining
                return r
            if 0 < bid_max[u] < P:                                 # Q case 2: market bid below me, I am best
                o.a4 = 0                                           # E1-28: queue ahead 0 and stays 0
                credit = max(0, vol_bid[u] - vol_bid_odd[u])       # odd lots excluded (SPEC 7, E1-28)
                return min(r, credit)
            if bid_min[u] == P and bid_max[u] == P:                # Q case 3: bid stayed at my price
                credit = max(0, vol_bid[u] - vol_bid_odd[u])
                return queue_credit(o, credit)
            return 0                                               # Q case 4: no credit
        else:                                                      # sell at P_a (mirror)
            cross = bid_max[u] >= P                                # E1-61: bid_max_u >= P_a (0 never, P_a > 0)
            if model == "S":
                return r if cross else 0
            if model == "T":
                touch = px_max[u] > 0 and px_max[u] >= P
                return r if (cross or touch) else 0
            if cross:                                              # Q case 1
                return r
            if ask_min[u] > P:                                     # Q case 2 (E1-61: ask_min_u > P_a)
                o.a4 = 0                                           # E1-28
                credit = max(0, vol_ask[u] - vol_ask_odd[u])
                return min(r, credit)
            if ask_min[u] == P and ask_max[u] == P:                # Q case 3
                credit = max(0, vol_ask[u] - vol_ask_odd[u])
                return queue_credit(o, credit)
            return 0                                               # Q case 4

    # ---- state ----
    q = 0                                  # inventory (shares)
    live = {+1: None, -1: None}            # at most one live limit order per side
    pending = None                         # market order waiting for the next bar: (dec, side, qty, q_already_moved)
    u_hit = None                           # E1-07 / E1-50: bar where |q| reached L*Q
    halt_end = None                        # E1-51: decisions d' <= halt_end do not quote
    lots = []                              # E1-55: A_naive FIFO lots [price, qty] of the remaining inventory
    fills, orders = [], []
    n_dec = n_dec_qbase = n_stale = 0
    n_int = {+1: 0, -1: 0}
    n_int_fill = {+1: 0, -1: 0}
    q_df = 0

    def naive_sell_price(d):
        """E1-55: max(ceil_to_tick(FIFO average buy price + 1 tick), ask_d)."""
        sum_pq = sum(p * s for p, s in lots)
        sum_q = sum(s for _, s in lots)
        num = sum_pq + TICK * sum_q
        den = TICK * sum_q
        ceil_ticks = -((-num) // den)
        return max(ceil_ticks * TICK, ask[d])

    # ---- E1-50: bars u = d0, d0 + 1, ..., df + 1 ----
    for u in range(d0, df + 2):
        # (1) fill live limit orders and the pending market order with bar u
        for side in (+1, -1):
            o = live[side]
            if o is None or o.rem <= 0:
                continue
            f = fill_qty(o, u)
            if f <= 0:
                continue
            # E1-24: reference mid; reset -> decision bar that placed the order, event -> previous check bar u - 1
            ref2 = (bid[o.dec] + ask[o.dec]) if mode == "reset" else (bid[u - 1] + ask[u - 1])
            fills.append((u, side, 0, f, o.price, ref2, o.dec))   # SPEC 7: fill at the limit price
            o.rem -= f
            if not o.filled:                                       # E1-54: order with at least one fill
                o.filled = True
                n_int_fill[side] += 1
            q += side * f
            if naive:                                              # E1-55: FIFO lots
                if side == +1:
                    lots.append([o.price, f])
                else:
                    left = f
                    while left > 0:
                        take = min(left, lots[0][1])
                        lots[0][1] -= take
                        left -= take
                        if lots[0][1] == 0:
                            lots.pop(0)
        if pending is not None:
            dm, side, qty, moved = pending
            if side == -1:                                         # SPEC 7 market sell: min(bid_d, bid_min_{d+1})
                px = bid[dm] if bid_min[u] == 0 else min(bid[dm], bid_min[u])    # E1-61: bid_min 0 -> bid_d
            else:                                                  # SPEC 7 market buy: max(ask_d, ask_max_{d+1})
                px = max(ask[dm], ask_max[u])
            fills.append((u, side, 1, qty, px, bid[dm] + ask[dm], dm))           # E1-58: ref mid2 = mid2[d]
            if not moved:                                          # flatten order: q -> 0 here
                q += side * qty
            pending = None
        if u == df + 1:                                            # E1-50: only the flatten fill at df + 1
            break

        # (2) stale timer (E1-50, E1-07, E1-08: exactly |q| = L*Q)
        if abs(q) == LQ:
            if u_hit is None:
                u_hit = u
        elif abs(q) < LQ:
            u_hit = None

        # (3) volatility halt (E1-14, E1-51, E1-15): cancel live limits now (effective u + 1), extend the end
        if use_halt and halt_trigger(u):
            live[+1] = None
            live[-1] = None
            halt_end = u + HALT_LEN if halt_end is None else max(halt_end, u + HALT_LEN)

        # (4) df: cancel all limits, market -q (filled at df + 1)  (SPEC 3, E1-06, E1-50)
        if u == df:
            q_df = q
            live[+1] = None
            live[-1] = None
            if q != 0:
                pending = (df, -1 if q > 0 else +1, abs(q), False)
            continue
        if not is_decision(u):
            continue

        # ---- decision at bar d = u (only information up to bar d; orders live from d + 1, SPEC 4) ----
        d = u
        n_dec += 1                                                 # E1-57 denominator
        base = (bid[d] > 0 and ask[d] > bid[d] and ask[d] - bid[d] <= MAX_SPREAD
                and bsz[d] > 0 and asz[d] > 0)                     # E1-53 basic quotable condition
        if base:
            n_dec_qbase += 1                                       # E1-57 numerator

        # stale liquidation (SPEC 4, E1-52): one lot by market, q moves now, fill at d + 1
        if stale_s is not None and u_hit is not None and d - u_hit >= int(stale_s) and pending is None:
            side = -1 if q > 0 else +1
            pending = (d, side, QLOT, True)
            q += side * QLOT                                       # E1-52 / E1-09: quotes below use the new q
            u_hit = None                                           # timer back to 0
            n_stale += 1

        # E1-53: common conditions (gate already passed for a simulated day)
        common = base and ask[d] - bid[d] >= min_spread * TICK
        if common and gcnt_m != 0:
            gcnt = sum(ntr[d - 299:d + 1])                         # E1-04: trades in bars [d - 299, d]
            common = gcnt >= gcnt_m
        if common and halt_end is not None and d <= halt_end:      # E1-51: no quotes at d' <= halt end
            common = False
        ok = {+1: common, -1: common}
        if common and use_obi:                                     # E1-17: OBI, theta_o = 0.6, integer test
            x = 5 * (bsz[d] - asz[d])
            y = 3 * (bsz[d] + asz[d])
            if x <= -y:
                ok[+1] = False                                     # OBI <= -0.6 -> no buy
            if x >= y:
                ok[-1] = False                                     # OBI >= +0.6 -> no sell
        if common and use_fi:                                      # E1-16: FI over bars [d - 9, d], non-odd lots
            A = sum(vol_ask[v] - vol_ask_odd[v] for v in range(d - 9, d + 1))
            B = sum(vol_bid[v] - vol_bid_odd[v] for v in range(d - 9, d + 1))
            if A + B >= 300:                                       # defined only with >= 300 shares
                if 5 * (A - B) >= 3 * (A + B):
                    ok[-1] = False                                 # FI >= +0.6 -> no sell
                if 5 * (A - B) <= -3 * (A + B):
                    ok[+1] = False                                 # FI <= -0.6 -> no buy

        # SPEC 4 sizes: buy min(Q, LQ - q), sell min(Q, LQ + q), long only sell min(Q, q)
        size = {+1: min(QLOT, LQ - q), -1: min(QLOT, q) if long_only else min(QLOT, LQ + q)}
        # SPEC 4 prices: join the best; A_naive sell per E1-55 (only needed when a sell can be placed)
        price = {+1: bid[d], -1: ask[d]}
        if naive and ok[-1] and size[-1] > 0:
            price[-1] = naive_sell_price(d)

        for side in (+1, -1):
            o = live[side]
            keep = False
            if mode == "event":
                # E1-18: keep while my price equals the current best (end of bar d) and the side may quote;
                # E1-11: no top-up, shrink to the allowed size (keeps priority); E1-19 cancels per side / both
                keep = (ok[side] and size[side] > 0 and o is not None and o.rem > 0 and o.price == price[side])
            if keep:
                o.rem = min(o.rem, size[side])
                continue
            live[side] = None                                      # reset: cancel and re-place every decision
            if ok[side] and size[side] > 0:
                P = price[side]
                if naive and side == -1 and P > ask[d]:
                    a4 = None                                      # E1-55: above the ask -> infinite queue ahead
                else:
                    a4 = th4 * (bsz[d] if side == +1 else asz[d])  # E1-56: A4 = 4 theta * displayed size at d
                no = _Order(d, side, P, size[side], a4)
                live[side] = no
                orders.append((d, side, P, size[side], -1 if a4 is None else a4))
                n_int[side] += 1

    out.update(fills=fills, orders=orders, q_df=q_df, n_dec=n_dec, n_dec_qbase=n_dec_qbase,
               n_int=(n_int[+1], n_int[-1]), n_int_fill=(n_int_fill[+1], n_int_fill[-1]), n_stale=n_stale)
    return out
