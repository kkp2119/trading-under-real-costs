"""M3 inventory rules (SPEC 4, E1-07/08/52): size limits, one side at the limit, stale liquidation with the timer back to
0, partial fills never above the limit, long only never short, inventory exactly 0 after the close."""
import dataclasses

import numpy as np
import pytest

from config import B0, MODELS, Q
from hand import P0, flat_day, trade
from kernelio import kern
from synth import random_day, settings


def bar_end_positions(fills):
    pos, out = 0, {}
    for u, side, _, qty, *_ in sorted(fills):
        pos += side * qty
        out[u] = pos
    return out


def crossing_buys():
    b = flat_day()
    b["ask_min"][900:] = P0                                          # every buy order fills in its first bar (S)
    return b


def test_limit_reached_then_only_sells_and_stale_liquidation_resets_timer():
    k = kern(crossing_buys(), B0, "S")
    buys = [o for o in k["orders"] if o[1] == 1]
    assert [o[0] for o in buys[:3]] == [899, 904, 909]               # q 0 -> 100 -> 200 -> 300 (limit at bar 910)
    assert all(o[0] >= 974 for o in buys[3:])                        # no buy quote while |q| = L*Q
    mkt = [x for x in k["fills"] if x[2] == 1]
    assert mkt[0][:2] == (975, -1) and mkt[0][6] == 974              # 974 - 910 = 64 >= 60 at the first decision; fills at 975
    assert mkt[1][6] == 1039                                         # limit again at 975 (re-bought), timer from 0: 1039 - 975 = 64
    assert k["n_stale"] == len([x for x in mkt if x[6] < 23099])


def test_stale_liquidation_at_exactly_s_seconds():
    b = flat_day()
    for u in (900, 905, 914):                                        # the order of 909 fills in its last bar: limit at 914
        b["ask_min"][u] = P0
    mkt = [x for x in kern(b, B0, "S")["fills"] if x[2] == 1]
    assert mkt[0][6] == 974 and mkt[0][0] == 975                     # 974 - 914 = 60 = S: liquidate at that decision


@pytest.mark.parametrize("L", [1, 3, 5])
def test_partial_fills_never_exceed_the_limit(L):
    b = flat_day(bid=P0)
    b["bid"][900:], b["bid_max"][900:], b["bid_min"][900:] = P0 - 100, P0 - 100, P0 - 100     # Q-2 every bar
    for u in range(900, 23000, 2):
        trade(b, u, "bid", 30, P0 - 100)
    k = kern(b, dataclasses.replace(B0, L=L), "Q")
    pos = bar_end_positions(k["fills"])
    assert max(pos.values()) <= L * Q and min(pos.values()) >= -L * Q
    assert any(x[3] % 100 for x in k["fills"])                       # partial fills happened


@pytest.mark.parametrize("seed", range(6))
def test_random_days_limits_long_only_and_flat_close(seed):
    b = random_day(1200 + seed, p_trade=0.4, p_jump=0.02)
    for name, cfg, _ in settings():
        for m in MODELS:
            k = kern(b, cfg, m)
            pos = bar_end_positions(k["fills"])
            if pos:
                assert max(pos.values()) <= cfg.L * Q and min(pos.values()) >= -cfg.L * Q, (name, m)
                if cfg.long_only or cfg.naive:
                    assert min(pos.values()) >= 0, (name, m)
            assert sum(x[1] * x[3] for x in k["fills"]) == 0, (name, m)               # flat after the close
            assert all(o[3] > 0 for o in k["orders"])
