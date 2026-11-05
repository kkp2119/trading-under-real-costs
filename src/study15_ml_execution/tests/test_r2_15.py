"""Research 15 experiment R2 tests: every rule off equals exec15 S2i on real 2015 and 2019 days; hand cases for E1 (narrow
spread -> market), E2 (two ticks inside at a 6-tick spread), E3 (last 60 s, spread <= 2 -> market), E4 (mid 3 ticks against
us -> market). Run with python -m pytest."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import exec15 as E  # noqa: E402
import loader15 as L  # noqa: E402
import r2_15 as R  # noqa: E402

T_, F_ = np.ones(L.NSEC, np.bool_), np.zeros(L.NSEC, np.bool_)


def test_all_rules_off_equal_s2i_on_real_days():
    for y in (2015, 2019):
        man, cb = L.manifest(y), L.close_bars(y)
        for k in sorted(man)[:15]:
            b = L.load(y, man[k])
            dirs, starts = E.parents(k[1], cb[k[1]])
            for qty, hor in E.SIZES:
                a = E.run_method(b, 2, dirs, starts, hor, qty, T_, T_, F_, F_, 1)
                z = R.run_rules(b, dirs, starts, hor, qty, -1, 0, 0, -1)
                assert np.array_equal(a, z)


def day(spread_ticks):
    b = {c: np.zeros(L.NSEC, dtype=np.int64) for c in L.COLS}
    ask = 100_000 + 100 * spread_ticks
    b["bid"][:], b["ask"][:], b["bsz"][:], b["asz"][:] = 100_000, ask, 200, 300
    for c, v in (("bid_max", 100_000), ("bid_min", 100_000), ("ask_max", ask), ("ask_min", ask)):
        b[c][:] = v
    return b


D1, S1_ = np.array([1]), np.array([299])


def rules(b, k=-1, e2=0, e3=0, x=-1):
    return tuple(R.run_rules(b, D1, S1_, 300, 1000, k, e2, e3, x)[0])


def test_e1_narrow_spread_market_at_start():
    """Spread 1 tick: E1 k=1 sends the 1,000 at market at the first decision (ask 10.01); off: S2i posts at the bid."""
    b = day(1)
    assert rules(b, k=1) == (0, 0, 1000, 1000 * 100_100, 700)
    assert rules(b)[2] == 1000 and rules(b)[3] == 1000 * 100_100     # S2i: nothing trades, market at the deadline


def test_e2_two_ticks_inside_at_six():
    """Spread 6 ticks (10.00 x 10.06): E2 buys at 10.02 (S2i 10.01); 1,000 trade at the data bid in bar 301 -> Q-2 fill."""
    b = day(6)
    b["vol"][301] = b["vol_bid"][301] = 1000
    assert rules(b, e2=1)[:2] == (1000, 1000 * 100_200)
    assert rules(b)[:2] == (1000, 1000 * 100_100)


def test_e3_last_minute_market():
    """Spread 2 ticks, nothing trades: E3 sends the remainder at market at the first decision of the last 60 s (bar 539)."""
    b = day(2)
    b["ask"][539:] = 100_300                                         # the ask moves after 539: E3 pays 10.02, S2i 10.03 at 599
    b["ask"][539] = 100_200
    assert rules(b, e3=1)[2:4] == (1000, 1000 * 100_200)
    assert rules(b)[2:4] == (1000, 1000 * 100_300)


def test_e4_stop_on_adverse_mid():
    """Buy: from bar 330 the market is 10.03 x 10.05 (mid +3 ticks over 10.01). E4 X=3 buys at market at the next decision
    (334) at 10.05; X=6 waits to the deadline."""
    b = day(2)
    b["bid"][330:], b["ask"][330:] = 100_300, 100_500
    for c, v in (("bid_max", 100_300), ("bid_min", 100_300), ("ask_max", 100_500), ("ask_min", 100_500)):
        b[c][330:] = v
    assert rules(b, x=3)[2:4] == (1000, 1000 * 100_500)
    out6 = R.run_rules(b, D1, S1_, 300, 1000, -1, 0, 0, 6)
    assert out6[0, 2] == 1000                                        # market only at the deadline
