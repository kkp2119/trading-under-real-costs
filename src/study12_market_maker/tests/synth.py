"""Synthetic 1-second bars with the bars.py semantics (no real data): carried NBBO state, extremes over the start state
and the updates inside the second, 0 before the first state, trades classified against the state at the start of the
second (price <= bid -> bid side, checked first; >= ask -> ask side; else inside; no state -> unclassified), odd lots
(< 100 shares), vol identity. Random symbol-days mix quiet and fast seconds, jumps (crosses), locked and wide quotes."""
import dataclasses

import numpy as np

NSEC = 23400
COLS = ("bid", "ask", "bsz", "asz", "bid_max", "bid_min", "ask_max", "ask_min", "n_upd", "n_xl", "vol", "ntr", "px_last",
        "px_min", "px_max", "vol_bid", "vol_ask", "vol_in", "vol_unc", "vol_bid_odd", "vol_ask_odd")


def random_day(seed, close_bar=NSEC, first_state=0, until=None, p_upd=0.25, p_jump=0.01, p_trade=0.35, spread=(1, 6),
               start_tick=5000, p_lock=0.01, p_wide=0.01, size=(100, 1500), p_odd=0.15):
    """Bars (dict of int64 arrays) for one symbol-day. first_state: first bar with a quote; updates and trades stop after
    `until` (default close_bar - 1; later bars carry the state)."""
    r = np.random.default_rng(seed)
    out = {c: np.zeros(NSEC, dtype=np.int64) for c in COLS}
    until = close_bar - 1 if until is None else until
    ctr, sp = start_tick, int(r.integers(spread[0], spread[1] + 1))
    bid = ask = bsz = asz = 0
    for u in range(NSEC):
        live = first_state <= u <= until
        b0, a0 = bid, ask                                             # state at the start of the second
        ups = []
        if live and (bid == 0 or r.random() < p_upd):
            for _ in range(int(r.integers(1, 4))):
                if r.random() < p_jump:
                    ctr += int(r.choice([-1, 1])) * int(r.integers(3, 12))
                else:
                    ctr += int(r.choice([-1, 0, 1]))
                ctr = max(ctr, 200)
                if r.random() < 0.3:
                    sp = int(r.integers(spread[0], spread[1] + 1))
                s = 0 if r.random() < p_lock else (int(r.integers(13, 20)) if r.random() < p_wide else sp)
                ups.append((ctr * 100, (ctr + s) * 100, int(r.integers(*size)), int(r.integers(*size))))
        if ups:
            bid, ask, bsz, asz = ups[-1]
        out["bid"][u], out["ask"][u], out["bsz"][u], out["asz"][u] = bid, ask, bsz, asz
        bs = [x[0] for x in ups] + ([b0] if b0 > 0 else [])
        as_ = [x[1] for x in ups] + ([a0] if a0 > 0 else [])
        if bs:
            out["bid_max"][u], out["bid_min"][u] = max(bs), min(bs)
            out["ask_max"][u], out["ask_min"][u] = max(as_), min(as_)
        if u < close_bar:
            out["n_upd"][u] = len(ups)
            out["n_xl"][u] = sum(1 for x in ups if x[1] <= x[0])
        if live and u < close_bar and r.random() < p_trade:
            prices = []
            for _ in range(int(r.integers(1, 4))):
                z = int(r.integers(1, 100)) if r.random() < p_odd else 100 * int(r.integers(1, 6))
                if b0 <= 0:
                    px = ctr * 100
                    out["vol_unc"][u] += z
                else:
                    k = r.random()
                    px = (b0 - (100 if r.random() < 0.05 else 0) if k < 0.45 else
                          a0 + (100 if r.random() < 0.05 else 0) if k < 0.9 else b0 + 100)
                    side = "bid" if px <= b0 else ("ask" if px >= a0 else "in")
                    out[f"vol_{side}"][u] += z
                    if side != "in" and z < 100:
                        out[f"vol_{side}_odd"][u] += z
                prices.append(px)
                out["ntr"][u] += 1
            out["px_last"][u], out["px_min"][u], out["px_max"][u] = prices[-1], min(prices), max(prices)
    out["vol"] = out["vol_bid"] + out["vol_ask"] + out["vol_in"] + out["vol_unc"]
    return out


def settings():
    """[(name, Config, dict)] for every stage-1 setting and A_naive (dict = the refsim cfg interface)."""
    import grid
    from config import A_NAIVE
    return [(n, c, dataclasses.asdict(c)) for n, c in grid.stage1() + [("A_naive", A_NAIVE)]]
