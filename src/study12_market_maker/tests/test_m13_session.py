"""M13 sessions and missing data (SPEC 2.1, 3; E1-02, E1-03, E1-21, E1-26, E1-36)."""
import dataclasses

import numpy as np

import features
import session
from config import B0
from hand import A0, P0, flat_day
from kernelio import kern
from synth import random_day

CAL_DAYS = ["2019-01-02", "2019-01-03", "2019-01-04", "2019-01-07", "2019-01-08", "2019-01-09", "2019-01-10",
            "2019-01-11", "2019-07-03", "2019-07-05"]


def cal(tmp_path, early=("2019-07-03",)):
    p = tmp_path / "calendar.csv"
    p.write_text("date,close_time\n" + "".join(f"{d},{'13:00' if d in early else '16:00'}\n" for d in CAL_DAYS))
    return session.Calendar(str(p))


def test_close_bars_and_d_bounds(tmp_path):
    c = cal(tmp_path)
    assert c.close["20190703"] == 12600 and c.close["20190705"] == 23400
    assert B0.d_bounds(12600) == (899, 12299) and B0.d_bounds(23400) == (899, 23099)
    assert dataclasses.replace(B0, window=1).d_bounds(23400) == (1799, 21599)
    assert dataclasses.replace(B0, window=1).d_bounds(12600) == (1799, 12299)


def test_no_state_at_d0_skips_and_preopen_carry_counts(tmp_path):
    late = random_day(5, first_state=1000)                           # first quote after 09:45
    assert kern(late, B0, "Q")["skipped"] == 2
    assert kern(late, dataclasses.replace(B0, window=1), "Q")["skipped"] == 0  # the 10:00 window has a state at 1799
    early = random_day(5, first_state=0)
    assert kern(early, B0, "Q")["skipped"] == 0
    crossed = flat_day(ask=P0 - 100)                                 # an ask < bid state at d0
    assert kern(crossed, B0, "Q")["skipped"] == 2
    assert kern(flat_day(), B0, "Q", gate=False)["skipped"] == 1


def test_no_quote_seconds_and_bad_quotes_place_nothing():
    b = flat_day()
    b["bid"][2000:2100] = 0                                          # a stretch with no bid
    k = kern(b, B0, "Q")
    assert not [o for o in k["orders"] if 2000 <= o[0] < 2100]
    f = features.compute(b)
    assert not f["qbase"][2000:2100].any() and f["qbase"][2100:].all()


def test_zero_bar_day_and_absent_day_and_reappearance(tmp_path):
    c = cal(tmp_path)
    zero = {k: np.zeros(23400, dtype=np.int64) for k in random_day(1)}
    assert kern(zero, B0, "Q")["skipped"] == 2                       # in the master, no data: not traded
    assert features.day_stats(zero, 23400)["no_data"]
    avail = {("X", d) for d in ["20190102", "20190103", "20190104", "20190107", "20190110", "20190111"]}   # absent 08, 09
    dv = {k: 5_000_000 * 10_000 for k in avail}
    assert session.gate(c, "X", "20190104", dv, set(), avail)        # first 5 trading days of the year: exempt
    assert session.gate(c, "X", "20190110", dv, set(), avail)        # back after 2 absent days: median of available days
    zero_days = {("X", d) for d in ["20190103", "20190104", "20190107"]}
    assert not session.gate(c, "X", "20190110", dv, zero_days, avail)  # zero-bar days count as 0
    assert not session.gate(c, "Z", "20190110", dv, set(), avail)    # no file among the previous 5 days -> no trade
    assert session.observation_days(c, "dev", avail | {("Y", "20190709")}) == [d for d in c.split_days("dev")
                                                                             if d in {x for _, x in avail} | {"20190709"}]


def test_year_start_exemption_counts_trading_days(tmp_path):
    c = cal(tmp_path)
    assert [c.year_exempt(d) for d in ["20190102", "20190108", "20190109"]] == [True, True, False]
    assert c.previous("20190110") == ["20190103", "20190104", "20190107", "20190108", "20190109"]
