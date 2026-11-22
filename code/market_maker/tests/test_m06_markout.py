"""M6 markouts (SPEC 8, E1-23): s x qty x (mid2_{u+k} - 2p) per limit fill; invalid mids and bars at or after the close
are excluded and counted."""
import numpy as np

import accounting
from hand import A0, P0, flat_day, move


def test_markout_by_hand_and_exclusions():
    b = flat_day()
    move(b, 903, bid=P0 - 200, ask=A0 - 200)                         # the mid moves exactly at u + 1
    b["ask"][932], b["bid"][932] = P0 - 300, P0 - 200                # invalid (crossed) mid at 902 + 30
    mid2 = b["bid"] + b["ask"]
    ok = (b["bid"] > 0) & (b["ask"] > 0) & (b["ask"] >= b["bid"])
    fills = np.array([[902, 1, 0, 100, P0, P0 + A0, 899]], dtype=np.int64)
    row = accounting.account(fills, mid2, ok, 23400, 899, 23099)
    assert row["mk1_sum2"] == 100 * (P0 + A0 - 400 - 2 * P0) and row["mk1_sh"] == 100      # bar 903, not 902
    assert row["mk5_sum2"] == 100 * (P0 + A0 - 400 - 2 * P0)
    assert (row["mk30_sum2"], row["mk30_sh"], row["mk30_excl"]) == (0, 0, 1)
    assert row["mk60_sum2"] == 100 * (P0 + A0 - 400 - 2 * P0)
    late = accounting.account(fills, mid2, ok, 950, 899, 23099)      # close at bar 950: 902 + 60 is after it
    assert late["mk60_excl"] == 1 and late["mk60_sh"] == 0 and late["mk30_excl"] == 1
    edge = accounting.account(fills, mid2, ok, 907, 899, 23099)      # 902 + 5 = the close bar itself: excluded
    assert (edge["mk5_sh"], edge["mk5_excl"], edge["mk1_sh"]) == (0, 1, 100)
    sell = np.array([[902, -1, 0, 100, A0, P0 + A0, 899]], dtype=np.int64)
    assert accounting.account(sell, mid2, ok, 23400, 899, 23099)["mk5_sum2"] == -100 * (P0 + A0 - 400 - 2 * A0)


def test_market_fills_have_no_markout_and_buckets():
    b = flat_day()
    mid2 = b["bid"] + b["ask"]
    ok = np.ones(23400, bool)
    fills = np.array([[950, -1, 1, 100, P0, P0 + A0, 949], [1000, 1, 0, 100, P0, P0 + A0, 999],
                      [8000, 1, 0, 200, P0, P0 + A0, 7999], [22000, -1, 0, 300, A0, P0 + A0, 21999]], dtype=np.int64)
    row = accounting.account(fills, mid2, ok, 23400, 899, 23099)
    assert row["mk1_sh"] == 600 and row["n_fill_ev"] == 3
    assert (row["open_sh"], row["noon_sh"], row["close_sh"]) == (100, 200, 300)
