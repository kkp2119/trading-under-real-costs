"""Hand-built bars for exact cases: a flat day (one NBBO state carried all day, no trades) edited bar by bar."""
import numpy as np

from synth import COLS, NSEC

P0, A0 = 500_000, 500_300            # $50.00 x $50.03 (3 ticks)


def flat_day(bid=P0, ask=A0, bsz=500, asz=500):
    b = {c: np.zeros(NSEC, dtype=np.int64) for c in COLS}
    b["bid"][:], b["ask"][:], b["bsz"][:], b["asz"][:] = bid, ask, bsz, asz
    b["bid_max"][:], b["bid_min"][:], b["ask_max"][:], b["ask_min"][:] = bid, bid, ask, ask
    return b


def move(b, u, bid=None, ask=None, bsz=None, asz=None):
    """New NBBO state from bar u on (one update inside bar u: the extremes of u span the old and new state)."""
    ob, oa = int(b["bid"][u - 1]), int(b["ask"][u - 1])
    nb = ob if bid is None else bid
    na = oa if ask is None else ask
    b["bid"][u:], b["ask"][u:] = nb, na
    b["bid_max"][u:], b["bid_min"][u:], b["ask_max"][u:], b["ask_min"][u:] = nb, nb, na, na
    b["bid_max"][u], b["bid_min"][u] = max(ob, nb), min(ob, nb)
    b["ask_max"][u], b["ask_min"][u] = max(oa, na), min(oa, na)
    if bsz is not None:
        b["bsz"][u:] = bsz
    if asz is not None:
        b["asz"][u:] = asz
    b["n_upd"][u] += 1


def trade(b, u, side, qty, price, odd=0):
    """A trade of qty shares (odd of them odd-lot) at price in bar u on 'bid', 'ask' or 'in'."""
    b[f"vol_{side}"][u] += qty
    if side in ("bid", "ask"):
        b[f"vol_{side}_odd"][u] += odd
    b["vol"][u] += qty
    b["ntr"][u] += 1
    b["px_last"][u] = price
    b["px_min"][u] = price if b["px_min"][u] == 0 else min(b["px_min"][u], price)
    b["px_max"][u] = max(b["px_max"][u], price)
