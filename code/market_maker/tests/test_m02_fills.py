"""M2 fill models S/T/Q (SPEC 7, E1-27/28/61): hand-computed cases on single orders and in the kernel (B0: reset 5 s, the
first decision at bar 899 places a buy at the bid 500,000 and a sell at the ask 500,300, 100 shares each)."""
import dataclasses

import numpy as np

from config import B0, MODEL_CODE
from fillmodel import fill_buy, fill_sell
from hand import A0, P0, flat_day, move, trade
from kernelio import kern


def fb(b, model, u, rem=100, a4=None, price=P0):
    kind, t4 = MODEL_CODE[model]
    a4 = t4 * int(b["bsz"][899]) if a4 is None else a4
    return fill_buy(kind, price, rem, a4, u, b["ask_min"], b["bid_max"], b["bid_min"], b["px_min"], b["vol_bid"],
                    b["vol_bid_odd"])


def fs(b, model, u, rem=100, a4=None, price=A0):
    kind, t4 = MODEL_CODE[model]
    a4 = t4 * int(b["asz"][899]) if a4 is None else a4
    return fill_sell(kind, price, rem, a4, u, b["bid_max"], b["ask_min"], b["ask_max"], b["px_max"], b["vol_ask"],
                     b["vol_ask_odd"])


def test_cross_fills_everything_in_every_model():
    b = flat_day()
    b["ask_min"][901] = P0                                           # the ask touched my bid inside the second
    assert all(fb(b, m, 901)[0] == 100 for m in MODEL_CODE)
    b["ask_min"][902] = 0                                            # E1-61: 0 = no quote, never crosses
    assert all(fb(b, m, 902)[0] == 0 for m in MODEL_CODE)


def test_touch_only_T_fills_and_Q_takes_queue_credit():
    b = flat_day(bsz=500)
    trade(b, 901, "bid", 100, P0)
    assert fb(b, "S", 901)[0] == 0 and fb(b, "T", 901)[0] == 100
    assert fb(b, "Q", 901) == (0, 4 * 500 - 4 * 100)                 # credit eats the queue ahead first


def test_queue_credit_fourth_bar_and_theta_scaling():
    b = flat_day(bsz=300)
    for u in (900, 901, 902, 903):
        trade(b, u, "bid", 100, P0)
    a4, got = 4 * 300, []
    for u in (900, 901, 902, 903):
        f, a4 = fb(b, "Q", u, 100 - sum(got), a4)
        got.append(f)
    assert got == [0, 0, 0, 100]                                     # 1200 -> 800 -> 400 -> 0 -> fill 100
    b2 = flat_day(bsz=150)
    trade(b2, 900, "bid", 100, P0)
    assert fb(b2, "Q25", 900) == (62, 0)                             # a4 = 150 quarter shares: floor((400 - 150) / 4)
    assert fb(b2, "Q50", 900) == (25, 0)                             # a4 = 300: (400 - 300) / 4
    assert fb(b2, "Q", 900) == (0, 200)                              # a4 = 600 - 400


def test_market_bid_below_my_price_case_2_and_odd_lots():
    b = flat_day()
    move(b, 900, bid=P0 - 100)                                       # bid moves down inside bar 900
    trade(b, 900, "bid", 70, P0 - 100)
    trade(b, 901, "bid", 150, P0 - 100, odd=50)
    assert fb(b, "Q", 900)[0] == 0                                    # bid changed inside the second: no credit (Q-4)
    assert fb(b, "Q", 901) == (100, 0)                               # Q-2: queue ahead 0, credit 150 - 50
    assert fb(b, "S", 901)[0] == 0 and fb(b, "T", 901)[0] == 100
    b["vol_bid_odd"][901] = 200                                      # more odd than total (bad data): credit floors at 0
    assert fb(b, "Q", 901) == (0, 0)


def test_bid_away_and_back_inside_the_second_gives_no_credit():
    b = flat_day()
    b["bid_max"][901] = P0 + 100                                     # bid went up and came back inside 901
    trade(b, 901, "bid", 300, P0)
    assert fb(b, "Q", 901) == (0, 4 * 500)


def test_sell_side_mirror():
    b = flat_day(asz=300)
    b["bid_max"][901] = A0
    assert all(fs(b, m, 901)[0] == 100 for m in MODEL_CODE)          # cross
    b = flat_day(asz=300)
    trade(b, 901, "ask", 100, A0)
    assert fs(b, "S", 901)[0] == 0 and fs(b, "T", 901)[0] == 100 and fs(b, "Q", 901) == (0, 800)
    b = flat_day(asz=150)
    trade(b, 900, "ask", 100, A0)
    assert fs(b, "Q25", 900) == (62, 0)
    b = flat_day()
    move(b, 900, ask=A0 + 100)
    trade(b, 901, "ask", 150, A0 + 100, odd=50)
    assert fs(b, "Q", 901) == (100, 0)                               # market ask above mine all second (Q-2)


def test_kernel_first_order_fills_with_reference_mid_and_both_sides_same_bar():
    b = flat_day()
    b["ask_min"][902] = P0                                           # buy crossed in 902
    b["bid_max"][902] = A0                                           # sell crossed in 902 too
    k = kern(b, B0, "S")
    assert k["fills"][:2] == [(902, -1, 0, 100, A0, P0 + A0, 899), (902, 1, 0, 100, P0, P0 + A0, 899)]
    assert (899, 1, P0, 100, 4 * 500) in k["orders"] and (899, -1, A0, 100, 4 * 500) in k["orders"]


def test_kernel_interval_end_bar_counts_and_next_bar_belongs_to_the_next_order():
    b = flat_day()
    b["ask_min"][904] = P0                                           # last bar of the order placed at 899 (lives 900..904)
    b["ask_min"][905] = P0                                           # first bar of the order placed at 904
    buys = [x for x in kern(b, B0, "S")["fills"] if x[1] == 1 and x[2] == 0]
    assert buys[:2] == [(904, 1, 0, 100, P0, P0 + A0, 899), (905, 1, 0, 100, P0, P0 + A0, 904)]


def test_kernel_locked_or_wide_quote_places_nothing():
    for ask in (P0, P0 + 1300):                                      # locked; 13 ticks > 12
        b = flat_day(ask=ask)
        k = kern(b, dataclasses.replace(B0, min_spread=1), "Q")
        assert k["orders"] == [] and k["n_dec"] > 0 and k["n_dec_qbase"] == 0
    k = kern(flat_day(ask=P0 + 100), B0, "Q")                        # 1 tick < minimum spread 2: data-quotable, no order
    assert k["orders"] == [] and k["n_dec_qbase"] == k["n_dec"]
