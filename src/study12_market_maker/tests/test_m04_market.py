"""M4 market orders (SPEC 7, 8; E1-61): price formula, R fee, the flatten decision bar (23,099 normal, 12,299 early close)
filled in the next bar."""
import accounting
import features
import simulator
from config import B0
from hand import A0, P0, flat_day
from kernelio import kern
from synth import random_day


def test_stale_liquidation_price_is_the_worse_of_decision_bid_and_next_bar_low():
    b = flat_day()
    b["ask_min"][900:] = P0
    b["bid_min"][975] = P0 - 200                                     # the bid dipped inside the fill bar
    mkt = [x for x in kern(b, B0, "S")["fills"] if x[2] == 1]
    assert mkt[0] == (975, -1, 1, 100, P0 - 200, P0 + A0, 974)
    b["bid_min"][975] = 0                                            # E1-61: no quote in the bar -> decision bid
    mkt = [x for x in kern(b, B0, "S")["fills"] if x[2] == 1]
    assert mkt[0][4] == P0


def test_market_buy_price_is_the_worse_of_decision_ask_and_next_bar_high():
    b = flat_day()
    b["bid_max"][900:] = A0                                          # every sell fills: short to the limit
    b["ask_max"][975] = A0 + 300
    mkt = [x for x in kern(b, B0, "S")["fills"] if x[2] == 1]
    assert mkt[0] == (975, 1, 1, 100, A0 + 300, P0 + A0, 974)


def test_flatten_bars_and_fee():
    for close_bar, df in ((23400, 23099), (12600, 12299)):
        seed = 31
        while kern(random_day(seed, close_bar=close_bar, p_trade=0.4), B0, "T", close_bar=close_bar)["q_df"] == 0:
            seed += 1                                                # a day that holds inventory at d_f
        b = random_day(seed, close_bar=close_bar, p_trade=0.4)
        k = kern(b, B0, "T", close_bar=close_bar)
        flat = [x for x in k["fills"] if x[2] == 1 and x[6] == df]
        assert k["q_df"] != 0 and len(flat) == 1 and flat[0][0] == df + 1 and flat[0][3] == abs(k["q_df"])
        assert max(x[0] for x in k["fills"]) == df + 1
    f = features.compute(b)
    fills, _, _ = simulator.simulate(b, f, B0.kernel_params("T", 12600, True))
    row = accounting.account(fills, f["mid2"], f["mid_ok"], 12600, *B0.d_bounds(12600))
    mkt_sh = row["sh_mkt_buy"] + row["sh_mkt_sell"]
    assert accounting.net(row, "R") == row["cash"] - 30 * mkt_sh                 # $0.003 per market share
