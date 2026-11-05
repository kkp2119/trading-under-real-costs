"""M7 event mode (SPEC 4, 7; E1-11/18/19/20): a kept order's queue keeps shrinking, a moved best cancels and re-places
(queue lost), reset re-sets the queue every decision, and decision counts / fill ownership by delta."""
import dataclasses

from config import B0
from hand import A0, P0, flat_day, move, trade
from kernelio import kern

EVENT = dataclasses.replace(B0, mode="event", delta=1)
RESET1 = dataclasses.replace(B0, delta=1)


def queue_day():
    b = flat_day(bsz=300)
    for u in range(900, 23000):
        trade(b, u, "bid", 100, P0)
    return b


def test_event_keeps_the_queue_reset_loses_it():
    b = queue_day()
    ev = [x for x in kern(b, EVENT, "Q")["fills"] if x[1] == 1 and x[2] == 0]
    assert ev[0] == (903, 1, 0, 100, P0, P0 + A0, 899)                # 1200 -> 800 -> 400 -> 0 -> fill; kept order of 899
    assert [o for o in kern(b, EVENT, "Q")["orders"] if o[1] == 1][0] == (899, 1, P0, 100, 1200)
    assert not [x for x in kern(b, RESET1, "Q")["fills"] if x[1] == 1 and x[2] == 0]      # queue re-set every second


def test_event_best_moves_cancel_and_replace_with_a_new_queue():
    b = flat_day(bsz=300)
    move(b, 907, bid=P0 + 100, bsz=400)
    buys = [o for o in kern(b, EVENT, "Q")["orders"] if o[1] == 1]
    assert buys[:2] == [(899, 1, P0, 100, 1200), (907, 1, P0 + 100, 100, 1600)]


def test_event_filter_cancels_only_the_blocked_side():
    b = flat_day(bsz=500, asz=500)
    move(b, 950, bsz=100, asz=900)                                   # OBI = -0.8: buy quote withdrawn, sell kept
    k = kern(b, dataclasses.replace(EVENT, obi=True), "Q")
    assert [o for o in k["orders"] if o[1] == -1] == [(899, -1, A0, 100, 2000)]   # the sell is never re-placed
    assert [o for o in k["orders"] if o[1] == 1] == [(899, 1, P0, 100, 2000)]     # buy cancelled at 950, not re-placed


def test_decision_counts_and_fill_owner_by_delta():
    b = flat_day()
    b["ask_min"][950] = P0
    for delta, n_dec, owner in ((1, 22200, 949), (5, 4440, 949), (15, 1480, 944), (60, 370, 899)):
        k = kern(b, dataclasses.replace(B0, delta=delta), "S")
        assert k["n_dec"] == n_dec
        buys = [x for x in k["fills"] if x[1] == 1 and x[2] == 0]
        assert buys[0][0] == 950 and buys[0][6] == owner, delta
