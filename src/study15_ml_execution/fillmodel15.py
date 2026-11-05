"""Research 15 copy of research12/fillmodel.py (unchanged code, R15: file name only). Fill models S, T, Q(theta) for one resting limit order in one bar (SPEC 7; E1-27, E1-28, E1-61). Integer only; the
queue ahead is in quarter shares (a4 = 4 theta x displayed size; a4 < 0 = infinite, E1-55). kind: 0 S, 1 T, 2 Q."""
try:
    from numba import njit as _njit
    HAVE_NUMBA = True
except ImportError:                                    # plain Python fallback: same code, slower
    HAVE_NUMBA = False


def jit(f):
    """numba-compiled f when numba is available (f.py_func is the plain function either way)."""
    if HAVE_NUMBA:
        return _njit(cache=False)(f)
    f.py_func = f
    return f


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
def market_price(side, dec, u, bid, ask, bid_min, ask_max):
    """SPEC 7 market order decided at bar dec, filled in bar u = dec + 1 (E1-61 for a 0 extreme)."""
    if side < 0:
        if bid_min[u] > 0 and bid_min[u] < bid[dec]:
            return bid_min[u]
        return bid[dec]
    if ask_max[u] > ask[dec]:
        return ask_max[u]
    return ask[dec]
