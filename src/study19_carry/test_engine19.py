"""Research 19: synthetic tests of engine19 (no data file is read). Writes results/engine19_tests.json.

Expected values are worked out by hand from SPEC19 / ANSWERS / ADDENDUM, not taken from the engine.
"""
import datetime as dt
import json
import math
import os
import sys
import tempfile
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine19 as G  # noqa: E402

TESTS = []
DIFF = {"AUD": 6, "CAD": 5, "CHF": 4, "EUR": 3, "GBP": 2, "JPY": 1, "NZD": 0, "NOK": -1, "SEK": -2}   # rate - US rate, percent
US = 2
C6 = G.CCY[:6]                                    # long AUD, CAD, CHF; short EUR, GBP, JPY; carry 3% a year
CARRY6 = 0.03 / 365


def test(f):
    TESTS.append(f)
    return f


def bdays(a, b, holidays=()):
    d, e, out = dt.date.fromisoformat(a), dt.date.fromisoformat(b), []
    while d <= e:
        if d.weekday() < 5 and d.isoformat() not in holidays:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def world(a="2021-01-01", b="2021-12-31", ccy=C6, fx=None, rates=None, vix=None, baa=None, holidays=(), usd=None):
    days = bdays(a, b, holidays)
    fxd = {c: {d: 1.0 for d in days} for c in ccy}
    for c, q in (fx or {}).items():
        fxd.setdefault(c, {}).update(q)
    ms = G.months(G.prv(G.ym(a)), G.ym(b))
    rate = {c: {m: US + DIFF[c] for m in ms} for c in ccy}
    rate["USD"] = {m: US for m in ms} if usd is None else usd
    for c, r in (rates or {}).items():
        rate[c] = r
    v = {d: 20 for d in days}
    v.update(vix or {})
    s = {d: 2 for d in days}
    s.update(baa or {})
    return G.make_data(fxd, rate, v, s), days


def mon(D, sim, a, b=None):
    ms, g, c = G.monthly(D, sim, (a, b or a))
    return {m: (float(g[k]), float(c[k])) for k, m in enumerate(ms)}


def run(D, name, cost=G.COST):
    return G.run_all(D, cost, None, (name,))[name]


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def idx(days, d):
    return days.index(d)


# ---------------------------------------------------------------- quote direction
@test
def direction_rows_usd_per_unit():
    q = G.fx_usd_per_unit([("2021-06-01", "0.7500"), ("2021-06-02", ""), ("2021-06-03", "0.7575")], True)
    assert q == {"2021-06-01": 0.75, "2021-06-03": 0.7575}


@test
def direction_rows_foreign_per_usd_is_inverted():
    q = G.fx_usd_per_unit([("2021-06-01", "100.00"), ("2021-06-02", "99.00")], False)
    assert close(q["2021-06-01"], 0.01) and close(q["2021-06-02"], 1 / 99) and q["2021-06-02"] > q["2021-06-01"]


@test
def direction_long_gains_when_foreign_rises_usd_per_unit():
    days = bdays("2021-01-01", "2021-12-31")
    path = {d: 0.75 * (1.01 if d >= "2021-07-01" else 1.0) for d in days}                 # AUD (long) +1% at the June exit
    D, _ = world(fx={"AUD": path})
    g, c = mon(D, run(D, "C0"), (2021, 6))[(2021, 6)]
    assert close(g, 0.01 / 3 + CARRY6 * 30) and g - CARRY6 * 30 > 0


@test
def direction_long_gains_when_foreign_rises_foreign_per_usd():
    days = bdays("2021-01-01", "2021-12-31")
    rows = [(d, "99.00" if d >= "2021-07-01" else "100.00") for d in days]                # fewer CAD per USD = CAD up
    D, _ = world(fx={"CAD": G.fx_usd_per_unit(rows, False)})
    g, _ = mon(D, run(D, "C0"), (2021, 6))[(2021, 6)]
    assert close(g, (100 / 99 - 1) / 3 + CARRY6 * 30) and g - CARRY6 * 30 > 0


@test
def direction_short_loses_when_foreign_rises():
    days = bdays("2021-01-01", "2021-12-31")
    rows = [(d, "99.00" if d >= "2021-07-01" else "100.00") for d in days]                # JPY is a short leg
    D, _ = world(fx={"JPY": G.fx_usd_per_unit(rows, False)})
    g, _ = mon(D, run(D, "C0"), (2021, 6))[(2021, 6)]
    assert close(g, -(100 / 99 - 1) / 3 + CARRY6 * 30)


# ---------------------------------------------------------------- interest
@test
def interest_30_days_3_percent():
    ms = G.months((2020, 12), (2021, 12))
    rates = {c: {m: US + (3 if c in C6[:3] else 0) for m in ms} for c in C6}             # longs +3%, shorts 0%
    D, days = world(rates=rates)
    assert (dt.date(2021, 7, 1) - dt.date(2021, 6, 1)).days == 30
    g, c = mon(D, run(D, "C0"), (2021, 6))[(2021, 6)]
    assert close(g, 0.03 * 30 / 365, 1e-15) and close(c, 0.0008, 1e-18)


@test
def interest_uses_calendar_days_not_twelfths():
    D, _ = world()
    m = mon(D, run(D, "C0"), (2021, 2), (2021, 3))
    assert close(m[(2021, 2)][0], CARRY6 * 28) and close(m[(2021, 3)][0], CARRY6 * 31)    # Feb 1 -> Mar 1 -> Apr 1


@test
def interest_short_leg_pays_the_differential():
    ms = G.months((2020, 12), (2021, 12))
    rates = {c: {m: US + (0 if c in C6[:3] else -3) for m in ms} for c in C6}            # shorts 3% below the US
    D, _ = world(rates=rates)
    g, _ = mon(D, run(D, "C0"), (2021, 6))[(2021, 6)]
    assert close(g, 0.03 * 30 / 365, 1e-15)                                              # short a low rate earns


# ---------------------------------------------------------------- execution at the next business day
def _spike(extra_fx=None, **kw):
    """VIX 30 from Wed 2021-06-09 to Wed 2021-06-16 (previous month end 20): DV20 is on for those closes."""
    days = bdays("2021-01-01", "2021-12-31", kw.get("holidays", ()))
    vix = {d: 30 for d in days if "2021-06-09" <= d <= "2021-06-16"}
    aud = {d: 1.0 + 0.001 * k for k, d in enumerate(days)}
    aud.update(extra_fx or {})
    return world(fx={"AUD": aud}, vix=vix, **kw), aud


@test
def delay_daily_exit_and_reentry_dates():
    (D, days), _ = _spike()
    s = run(D, "DV20-D")
    assert s["held"][idx(days, "2021-06-09")] and not s["held"][idx(days, "2021-06-10")]
    assert s["mid_exit"][idx(days, "2021-06-10")] and s["mid_exit"].sum() == 1
    assert not s["held"][idx(days, "2021-06-17")] and s["held"][idx(days, "2021-06-18")]
    assert s["mid_entry"][idx(days, "2021-06-18")] and s["mid_entry"].sum() == 1


@test
def delay_daily_month_total_by_hand():
    (D, days), aud = _spike()
    g, c = mon(D, run(D, "DV20-D"), (2021, 6))[(2021, 6)]
    spot = (aud["2021-06-10"] / aud["2021-06-01"] - 1) / 3 + (aud["2021-07-01"] / aud["2021-06-18"] - 1) / 3
    assert close(g, spot + CARRY6 * (9 + 13)) and close(c, 3 * 0.0008)


@test
def delay_signal_day_rate_does_not_matter():
    base = mon(*_first(_spike()), (2021, 6))
    for d in ("2021-06-09", "2021-06-17"):                                               # the two signal days
        alt = mon(*_first(_spike({d: 7.0})), (2021, 6))
        assert close(alt[(2021, 6)][0], base[(2021, 6)][0], 1e-12), d


@test
def delay_execution_day_rate_matters():
    base = mon(*_first(_spike()), (2021, 6))
    for d in ("2021-06-10", "2021-06-18"):                                               # the two execution days
        alt = mon(*_first(_spike({d: 1.5})), (2021, 6))
        assert abs(alt[(2021, 6)][0] - base[(2021, 6)][0]) > 0.01, d


def _first(x):
    (D, days), _ = x
    return D, run(D, "DV20-D")


@test
def delay_weekly_acts_on_monday():
    (D, days), aud = _spike()
    s = run(D, "DV20-W")
    assert s["mid_exit"][idx(days, "2021-06-14")] and s["mid_exit"].sum() == 1          # Friday 06-11 check, Monday exit
    assert s["mid_entry"][idx(days, "2021-06-21")] and s["mid_entry"].sum() == 1        # Friday 06-18 check, Monday entry
    g, c = mon(D, s, (2021, 6))[(2021, 6)]
    spot = (aud["2021-06-14"] / aud["2021-06-01"] - 1) / 3 + (aud["2021-07-01"] / aud["2021-06-21"] - 1) / 3
    assert close(g, spot + CARRY6 * (13 + 10)) and close(c, 3 * 0.0008)


@test
def delay_monthly_ignores_a_spike_inside_the_month():
    (D, days), aud = _spike()
    s, c0 = run(D, "DV20-M"), run(D, "C0")
    assert np.array_equal(s["pnl"], c0["pnl"]) and np.array_equal(s["cst"], c0["cst"]) and not s["exit"].any()


@test
def weekly_check_day_is_the_last_business_day_of_the_week():
    (D, days), _ = _spike(holidays=("2021-06-11",))                                      # Friday is a holiday
    chk = G.check_days(D, "W")
    assert chk[idx(days, "2021-06-10")] and not chk[idx(days, "2021-06-09")]
    s = run(D, "DV20-W")
    assert s["mid_exit"][idx(days, "2021-06-14")]                                        # Thursday check, Monday exit


@test
def check_days_by_frequency():
    D, days = world()
    m, w, d = (G.check_days(D, q) for q in "MWD")
    assert d.all() and m.sum() == 12 and m[idx(days, "2021-03-31")] and m[idx(days, "2021-05-31")]
    assert w.sum() == 53 and all(dt.date.fromisoformat(days[i]).weekday() == 4 for i in np.flatnonzero(w))


# ---------------------------------------------------------------- VIX percentile filters
def _levels(may=None, jun=None):
    days = bdays("2021-01-01", "2021-12-31")
    lvl = {"01": 10, "02": 20, "03": 30, "04": 40}
    vix = {d: lvl.get(d[5:7], 15) for d in days}
    vix.update({"2021-05-12": "31", "2021-05-13": "31.01", "2021-05-14": "34", "2021-05-17": "34.01", "2021-05-18": "37",
                "2021-05-19": "37.01", "2021-05-31": 100, "2021-06-09": "38", "2021-06-10": "38.01"})
    vix.update(may or {})
    vix.update(jun or {})
    return world(vix=vix)


@test
def percentile_value_by_hand():
    a = [Fraction(x) for x in (10, 20, 30, 40)]
    assert (G.pct(a, 70), G.pct(a, 80), G.pct(a, 90)) == (31, 34, 37)
    assert G.pct([Fraction(x) for x in (10, 20, 30, 40, 100)], 70) == 38


@test
def percentile_matches_numpy():
    rng = np.random.default_rng(1)
    for n in (1, 2, 3, 36, 37, 100, 275):
        v = sorted(int(x) for x in rng.integers(900, 8000, n))
        for q in (70, 80, 90):
            assert close(float(G.pct([Fraction(x, 100) for x in v], q)), float(np.percentile(np.array(v) / 100, q)), 1e-9)


@test
def percentile_equal_is_not_on():
    D, days = _levels()
    on = G.conditions(D)
    for f, eq, above in (("V70", "2021-05-12", "2021-05-13"), ("V80", "2021-05-14", "2021-05-17"), ("V90", "2021-05-18", "2021-05-19")):
        assert not on[f][idx(days, eq)] and on[f][idx(days, above)], f


@test
def percentile_reference_stops_at_the_previous_month_end():
    D, days = _levels()
    on = G.conditions(D)
    assert on["V70"][idx(days, "2021-05-31")]                       # 100 against Jan..Apr (31), its own month is not in
    assert not on["V70"][idx(days, "2021-06-09")] and on["V70"][idx(days, "2021-06-10")]   # June: 38 with May's 100 in


@test
def percentile_ignores_the_current_month_end_and_the_future():
    D, days = _levels()
    D2, _ = _levels(may={"2021-05-31": 5}, jun={d: 99 for d in bdays("2021-06-01", "2021-12-31")})
    a, b = G.conditions(D), G.conditions(D2)
    k = idx(days, "2021-05-28") + 1
    assert all(np.array_equal(a[f][:k], b[f][:k]) for f in ("V70", "V80", "V90"))


@test
def percentile_needs_a_reference():
    D, days = world(a="2021-01-01", b="2021-03-31", vix={d: 99 for d in bdays("2021-01-01", "2021-01-29")})
    on = G.conditions(D)
    assert not on["V70"][: idx(days, "2021-01-29") + 1].any()       # no month end before January


# ---------------------------------------------------------------- VIX jump and credit spread filters
@test
def dv_against_the_previous_month_end():
    vix = {"2021-05-12": "24", "2021-05-13": "24.01", "2021-05-14": "28", "2021-05-17": "28.01", "2021-05-18": "23.9", "2021-05-19": "24.5"}
    D, days = world(vix=vix)
    on = G.conditions(D)
    got = {d: (bool(on["DV20"][idx(days, d)]), bool(on["DV40"][idx(days, d)])) for d in vix}
    assert got == {"2021-05-12": (False, False), "2021-05-13": (True, False), "2021-05-14": (True, False),
                   "2021-05-17": (True, True), "2021-05-18": (False, False), "2021-05-19": (True, False)}


@test
def dv_reference_changes_with_the_month():
    vix = {d: 30 for d in bdays("2021-05-20", "2021-05-31")}
    vix.update({d: 30 for d in bdays("2021-06-01", "2021-06-30")})
    D, days = world(vix=vix)
    on = G.conditions(D)["DV20"]
    assert on[idx(days, "2021-05-31")] and not on[idx(days, "2021-06-01")]               # June compares with May's 30


@test
def cs_against_the_previous_month_end():
    baa = {d: "2.15" for d in bdays("2021-04-01", "2021-04-30")}
    baa.update({"2021-05-12": "2.35", "2021-05-13": "2.36", "2021-05-14": "2.55", "2021-05-17": "2.56"})
    D, days = world(baa=baa)
    on = G.conditions(D)
    got = {d: (bool(on["CS20"][idx(days, d)]), bool(on["CS40"][idx(days, d)])) for d in ("2021-05-12", "2021-05-13", "2021-05-14", "2021-05-17")}
    assert 2.35 - 2.15 > 0.20                                                            # the float trap the engine avoids
    assert got == {"2021-05-12": (False, False), "2021-05-13": (True, False), "2021-05-14": (True, False), "2021-05-17": (True, True)}


@test
def indicator_missing_on_a_check_day_uses_the_last_value():
    D, days = world()
    D2 = G.make_data({c: {d: 1.0 for d in days} for c in C6}, {k: dict(v) for k, v in D.rate.items()},
                     {d: (30 if d == "2021-06-10" else 20) for d in days if d != "2021-06-11"}, {d: 2 for d in days})
    on = G.conditions(D2)["DV20"]
    assert on[idx(days, "2021-06-10")] and on[idx(days, "2021-06-11")] and not on[idx(days, "2021-06-14")]


@test
def monthly_filter_empties_the_whole_next_month():
    days = bdays("2021-01-01", "2021-12-31")
    D, _ = world(vix={"2021-06-30": 30})                                                 # on at the June month end only
    s = run(D, "DV20-M")
    m = mon(D, s, (2021, 6), (2021, 8))
    assert close(m[(2021, 6)][0], CARRY6 * 30) and m[(2021, 7)] == (0.0, 0.0)            # June keeps its last day; July flat
    assert close(m[(2021, 8)][0], CARRY6 * 30) and close(m[(2021, 8)][1], 0.0008)        # July end: 20 against 30, off
    assert s["exit"][idx(days, "2021-07-01")] and s["exit"].sum() == 1 and not s["mid_exit"].any()


@test
def month_end_decision_is_the_same_at_every_frequency():
    D, days = world(vix={"2021-06-30": 30})                                              # Wednesday
    m = {q: mon(D, run(D, f"DV20-{q}"), (2021, 7))[(2021, 7)] for q in "MD"}
    assert m["M"] == (0.0, 0.0)
    # D: flat from 07-01; the 07-01 close is 20 against June's 30, so it re-enters on 07-02
    assert close(m["D"][0], CARRY6 * 31) and close(m["D"][1], 0.0008)


@test
def filters_wait_for_the_active_month():
    days = bdays("1992-11-02", "1993-04-30")
    vix = {d: 20 for d in days}
    vix.update({"1992-12-31": 30, "1993-01-29": 40, "1993-02-26": 20})          # on against the month end before, twice
    D = G.make_data({c: {d: 1.0 for d in days} for c in C6},
                    {**{c: {m: US + DIFF[c] for m in G.months((1992, 10), (1993, 4))} for c in C6},
                     "USD": {m: US for m in G.months((1992, 10), (1993, 4))}}, vix, {d: 2 for d in days})
    s = G.run_all(D, names=("DV20-M", "DV20-D"))
    for name in s:
        m = mon(D, s[name], (1993, 1), (1993, 2))
        assert m[(1993, 1)][1] == 0.0008, name                       # the 1992-12 month-end signal is before the start
        assert not s[name]["held"][days.index("1993-02-01")], name   # the 1993-01 month-end check is the first one
    assert mon(D, s["DV20-M"], (1993, 2))[(1993, 2)] == (0.0, 0.0)
    assert s["DV20-D"]["mid_entry"][days.index("1993-02-02")]        # daily: 02-01 closes at 20 against January's 40
    assert G.want(D, G.conditions(D)["DV20"], "D").tolist() == [d != "1993-01-29" for d in days]


# ---------------------------------------------------------------- cost
@test
def cost_exit_and_reentry_are_4bp_each():
    (D, days), _ = _spike()
    s = run(D, "DV20-D")
    assert close(s["cst"][idx(days, "2021-06-01")], 0.0004 * 2) and close(s["cst"][idx(days, "2021-06-10")], 0.0004 * 2)
    assert close(s["cst"][idx(days, "2021-06-18")], 0.0004 * 2) and close(s["cst"].sum(), (11 + 2) * 0.0008)   # Feb..Dec entries


@test
def cost_doubles_at_8bp():
    (D, days), _ = _spike()
    a, b = run(D, "DV20-D"), run(D, "DV20-D", 0.0008)
    assert np.array_equal(a["pnl"], b["pnl"]) and np.allclose(b["cst"], 2 * a["cst"], atol=1e-18)


@test
def cost_is_zero_in_a_flat_month_and_scheduled_exit_is_free():
    D, days = world(vix={"2021-06-30": 30})
    s = run(D, "DV20-M")
    assert s["cst"][idx(days, "2021-07-01")] == 0 and s["cst"][idx(days, "2021-06-30")] == 0
    assert close(mon(D, s, (2021, 7))[(2021, 7)][1], 0.0)


@test
def cost_of_c0_is_8bp_of_capital_a_month():
    D, _ = world()
    m = mon(D, run(D, "C0"), (2021, 2), (2021, 11))
    assert all(close(c, 0.0008) for _, c in m.values())


# ---------------------------------------------------------------- DEM -> EUR, fewer than six, fill
@test
def dem_to_eur_splice_in_euro_units():
    dem = [("1998-12-01", "1.6000"), ("1998-12-31", "1.6500"), ("1999-01-04", "9.9")]
    eur = [("1998-12-31", "9.9"), ("1999-01-04", "1.1800")]
    q = G.eur_slot(eur, dem)
    assert set(q) == {"1998-12-01", "1998-12-31", "1999-01-04"}
    assert close(q["1998-12-01"], 1.95583 / 1.6) and close(q["1998-12-31"], 1.95583 / 1.65) and q["1999-01-04"] == 1.18


@test
def dem_position_exits_at_the_euro_rate():
    days = bdays("1998-10-01", "1999-03-31", holidays=("1999-01-01",))
    dem = [(d, "1.6000") for d in days if d <= "1998-12-31"]
    eur = [(d, "1.2500") for d in days if d >= "1999-01-04"]
    ms = G.months((1998, 9), (1999, 3))
    rate = {c: {m: US + DIFF[c] for m in ms} for c in C6}
    rate["EUR"] = G.splice({m: 9 for m in ms}, {m: -5 for m in ms}, G.EUR_RATE_FROM)      # German rate high, euro rate low
    rate["USD"] = {m: US for m in ms}
    fx = {c: {d: 1.0 for d in days} for c in C6}
    fx["EUR"] = G.eur_slot(eur, dem)
    D = G.make_data(fx, rate, {d: 20 for d in days}, {d: 2 for d in days})
    p98, p99 = G.portfolio(D, (1998, 12)), G.portfolio(D, (1999, 1))
    j = G.CCY.index("EUR")
    assert p98[1][list(p98[0]).index(j)] > 0 and p99[1][list(p99[0]).index(j)] < 0        # long as the mark, short as the euro
    g, _ = mon(D, G.simulate(D), (1998, 12))[(1998, 12)]
    carry = ((7 + 6 + 5) - (4 + 2 + 1)) / 3 / 100 * 34 / 365      # mark 9 - 2 = 7: long EUR, AUD, CAD; short CHF, GBP, JPY
    assert close(g, (1.25 / (1.95583 / 1.6) - 1) / 3 + carry)                             # 1998-12-01 -> 1999-01-04, 34 days


@test
def dem_path_none_leaves_the_slot_empty_before_1999():
    days = bdays("1998-10-01", "1999-03-31", holidays=("1999-01-01",))
    ccy = ("AUD", "CAD", "EUR", "GBP", "JPY", "NZD", "NOK")
    fx = {c: {d: 1.0 for d in days} for c in ccy}
    fx["EUR"] = G.eur_slot([(d, "1.2500") for d in days if d >= "1999-01-04"], None)
    ms = G.months((1998, 9), (1999, 3))
    rate = {c: {m: US + DIFF[c] for m in ms} for c in ccy}
    rate["EUR"] = G.splice({}, {m: 9 for m in ms}, G.EUR_RATE_FROM)
    rate["USD"] = {m: US for m in ms}
    D = G.make_data(fx, rate, {d: 20 for d in days}, {d: 2 for d in days})
    j = G.CCY.index("EUR")
    assert j not in G.portfolio(D, (1998, 12))[0] and len(G.portfolio(D, (1998, 12))[0]) == 6
    assert j in G.portfolio(D, (1999, 1))[0]                                              # formed at the end of 1998-12


@test
def rate_splice_japan():
    cd, ib = {(2002, 3): 1, (2002, 4): 2}, {(2002, 3): 8, (2002, 4): 9, (2002, 5): 10}
    assert G.splice(cd, ib, G.JP_IB_FROM) == {(2002, 3): 1, (2002, 4): 9, (2002, 5): 10}


def _weights(D, m):
    p = G.portfolio(D, m)
    return None if p is None else {G.CCY[j]: round(float(w), 12) for j, w in zip(p[0], p[1])}


@test
def six_or_more_currencies_three_and_three():
    D, _ = world(ccy=G.CCY[:7])
    third = round(1 / 3, 12)
    assert _weights(D, (2021, 6)) == {"AUD": third, "CAD": third, "CHF": third, "GBP": -third, "JPY": -third, "NZD": -third}


@test
def fewer_than_six_currencies_two_and_two():
    D, _ = world(ccy=G.CCY[:5])
    assert _weights(D, (2021, 6)) == {"AUD": 0.5, "CAD": 0.5, "EUR": -0.5, "GBP": -0.5}
    D, _ = world(ccy=G.CCY[:4])
    assert _weights(D, (2021, 6)) == {"AUD": 0.5, "CAD": 0.5, "CHF": -0.5, "EUR": -0.5}


@test
def fewer_than_four_currencies_no_position():
    D, _ = world(ccy=G.CCY[:3])
    s = run(D, "C0")
    assert G.portfolio(D, (2021, 6)) is None and not s["pnl"].any() and not s["cst"].any() and not s["held"].any()


@test
def two_and_two_keeps_one_dollar_a_side():
    D, _ = world(ccy=G.CCY[:5])
    g, c = mon(D, run(D, "C0"), (2021, 6))[(2021, 6)]
    assert close(g, ((6 + 5) / 2 - (3 + 2) / 2) / 100 * 30 / 365) and close(c, 0.0008)


@test
def ties_go_in_currency_order():
    ms = G.months((2020, 12), (2021, 12))
    D, _ = world(ccy=G.CCY, rates={c: {m: 5 for m in ms} for c in G.CCY})
    third = round(1 / 3, 12)
    assert _weights(D, (2021, 6)) == {"AUD": third, "CAD": third, "CHF": third, "NZD": -third, "NOK": -third, "SEK": -third}


@test
def tie_is_exact_whatever_the_number_type():
    ms = G.months((2020, 12), (2021, 12))
    same = {"AUD": "5.250", "CAD": 5.25, "CHF": Fraction(21, 4), "EUR": "5.25", "GBP": 5.25, "JPY": Fraction(525, 100)}
    D, _ = world(rates={c: {m: v for m in ms} for c, v in same.items()})
    third = round(1 / 3, 12)
    assert _weights(D, (2021, 6)) == {"AUD": third, "CAD": third, "CHF": third, "EUR": -third, "GBP": -third, "JPY": -third}


@test
def fill_one_missing_month_with_the_month_before():
    ms = G.months((2020, 12), (2021, 12))
    aud = {m: US + (9 if m == (2021, 4) else 0.5) for m in ms if m != (2021, 5)}          # May missing; April says 9
    D, _ = world(rates={"AUD": aud})
    assert G.rate_at(D.rate["AUD"], (2021, 5)) == US + 9
    assert _weights(D, (2021, 6))["AUD"] > 0 and "AUD" not in {k for k, w in _weights(D, (2021, 7)).items() if w > 0}


@test
def fill_stops_after_one_month():
    ms = G.months((2020, 12), (2021, 12))
    aud = {m: US + 9 for m in ms if m not in ((2021, 5), (2021, 6))}
    D, _ = world(ccy=G.CCY[:7], rates={"AUD": aud})
    assert "AUD" in _weights(D, (2021, 6)) and "AUD" not in _weights(D, (2021, 7)) and "AUD" in _weights(D, (2021, 8))


@test
def fill_applies_to_the_us_rate():
    ms = G.months((2020, 12), (2021, 12))
    D, _ = world(usd={m: (0 if m == (2021, 4) else US) for m in ms if m not in ((2021, 5), (2021, 6))})
    g, _ = mon(D, run(D, "C0"), (2021, 6))[(2021, 6)]                                     # formed with April's US rate
    assert close(g, CARRY6 * 30)                                                          # the US rate cancels: long 1, short 1
    assert G.portfolio(D, (2021, 7)) is None and G.portfolio(D, (2021, 8)) is not None    # two months missing: no formation
    assert mon(D, run(D, "C0"), (2021, 7))[(2021, 7)] == (0.0, 0.0)


@test
def currency_needs_a_quote_on_the_entry_day():
    days = bdays("2021-01-01", "2021-12-31")
    D, _ = world(ccy=G.CCY[:7])
    fx = {c: {d: 1.0 for d in days} for c in G.CCY[:7]}
    del fx["AUD"]["2021-06-01"]
    D2 = G.make_data(fx, {k: dict(v) for k, v in D.rate.items()}, {d: 20 for d in days}, {d: 2 for d in days})
    assert "AUD" not in _weights(D2, (2021, 6)) and "AUD" in _weights(D2, (2021, 7)) and "AUD" not in _weights(D2, (2021, 5))


@test
def missing_quote_inside_the_month_is_carried_forward():
    days = bdays("2021-01-01", "2021-12-31")
    aud = {d: 1.0 + 0.001 * k for k, d in enumerate(days)}
    D, _ = world(fx={"AUD": aud})
    fx = {c: {d: 1.0 for d in days} for c in C6}
    fx["AUD"] = {d: v for d, v in aud.items() if d != "2021-06-15"}
    D2 = G.make_data(fx, {k: dict(v) for k, v in D.rate.items()}, {d: 20 for d in days}, {d: 2 for d in days})
    a, b = run(D, "C0"), run(D2, "C0")
    i = idx(days, "2021-06-15")
    assert close(b["pnl"][i], CARRY6) and close(b["pnl"][i] + b["pnl"][i + 1], a["pnl"][i] + a["pnl"][i + 1])
    assert close(mon(D, a, (2021, 6))[(2021, 6)][0], mon(D2, b, (2021, 6))[(2021, 6)][0])


# ---------------------------------------------------------------- holding and attribution
@test
def no_rebalancing_inside_the_month():
    aud = {"2021-06-01": 1.0, "2021-06-02": 1.1, "2021-06-03": 1.21}
    D, days = world(fx={"AUD": aud})
    s = run(D, "C0")
    i = idx(days, "2021-06-02")
    assert close(s["pnl"][i], 0.1 / 3 + CARRY6) and close(s["pnl"][i + 1], 0.11 / 3 + CARRY6)   # the same units both days


@test
def last_interval_of_the_month_belongs_to_the_month():
    aud = {d: 1.0 for d in bdays("2021-01-01", "2021-06-30")}
    aud.update({d: 1.3 for d in bdays("2021-07-01", "2021-12-31")})                       # the jump is on the July roll day
    D, days = world(fx={"AUD": aud})
    m = mon(D, run(D, "C0"), (2021, 6), (2021, 7))
    assert close(m[(2021, 6)][0], 0.3 / 3 + CARRY6 * 30) and close(m[(2021, 7)][0], CARRY6 * 32)
    assert close(m[(2021, 6)][1], 0.0008) and close(m[(2021, 7)][1], 0.0008)


@test
def reentry_uses_one_dollar_a_side_at_the_new_rate():
    (D, days), aud = _spike({d: 2.0 for d in bdays("2021-06-18", "2021-06-30")})          # AUD doubles while flat
    s = run(D, "DV20-D")
    i = idx(days, "2021-07-01")
    assert close(s["pnl"][i], (aud["2021-07-01"] / 2.0 - 1) / 3 + CARRY6)                 # 1/3 dollar bought at 2.0 on 06-18


@test
def data_end_before_the_exit_day_marks_at_the_last_day():
    D, days = world(b="2021-06-15", fx={"AUD": {"2021-06-15": 1.03}})
    g, c = mon(D, run(D, "C0"), (2021, 6))[(2021, 6)]
    assert close(g, 0.03 / 3 + CARRY6 * 14) and close(c, 0.0008)
    assert G.incomplete(D, ((2021, 5), (2021, 6))) == [(2021, 6)] and G.incomplete(D, ((2021, 4), (2021, 5))) == []


@test
def first_month_of_the_data_has_no_position():
    D, days = world()
    s = run(D, "C0")
    assert not s["held"][: idx(days, "2021-02-01")].any() and s["held"][idx(days, "2021-02-01")]


# ---------------------------------------------------------------- equivalence without a filter
def _random_world(seed=3):
    rng = np.random.default_rng(seed)
    days = bdays("2000-01-03", "2003-12-31", holidays=("2001-07-04", "2001-12-25", "2002-01-01", "2002-07-04", "2003-04-18"))
    fx = {}
    for c in G.CCY:
        steps = rng.normal(0, 0.007, len(days))
        fx[c] = dict(zip(days, (0.5 + rng.random()) * np.exp(np.cumsum(steps))))
    ms = G.months((1999, 12), (2003, 12))
    rate = {c: {m: round(float(rng.uniform(0, 9)), 4) for m in ms} for c in G.CCY}
    rate["USD"] = {m: round(float(rng.uniform(0, 6)), 4) for m in ms}
    vix = dict(zip(days, np.round(12 + 30 * rng.random(len(days)), 2)))
    baa = dict(zip(days, np.round(1.5 + np.cumsum(rng.normal(0, 0.05, len(days))) % 3, 2)))
    return G.make_data(fx, rate, vix, baa), days, fx, rate


@test
def equivalence_daily_engine_equals_the_monthly_formula():
    D, days, fx, rate = _random_world()
    ms, g, c = G.monthly(D, G.simulate(D), ((2000, 2), (2003, 11)))
    worst = 0.0
    for k, m in enumerate(ms):
        t, e, x = G.prv(m), days[D.first[m]], days[D.first[G.nxt(m)]]
        diff = sorted(((rate[cc][t] - rate["USD"][t], -j, cc) for j, cc in enumerate(G.CCY)), reverse=True)
        longs, shorts = [cc for _, _, cc in diff[:3]], [cc for _, _, cc in diff[-3:]]
        n = (dt.date.fromisoformat(x) - dt.date.fromisoformat(e)).days
        want = sum((fx[cc][x] / fx[cc][e] - 1) / 3 + (rate[cc][t] - rate["USD"][t]) / 100 * n / 365 / 3 for cc in longs)
        want -= sum((fx[cc][x] / fx[cc][e] - 1) / 3 + (rate[cc][t] - rate["USD"][t]) / 100 * n / 365 / 3 for cc in shorts)
        worst = max(worst, abs(g[k] - want))
        assert close(c[k], 0.0008, 1e-18)
    assert len(ms) == 46 and worst < 1e-13, worst


@test
def equivalence_filter_that_never_fires_is_c0():
    D, days, fx, rate = _random_world()
    D2 = G.make_data(fx, rate, {d: 20 for d in days}, {d: 2 for d in days})
    s = G.run_all(D2, active_from=None)
    for name in G.CANDS[1:]:
        assert np.array_equal(s[name]["pnl"], s["C0"]["pnl"]) and np.array_equal(s[name]["cst"], s["C0"]["cst"]), name


@test
def monthly_filter_equals_c0_with_flagged_months_zeroed():
    D, days, fx, rate = _random_world()
    on = G.conditions(D)["DV20"]
    ms, g, c = G.monthly(D, G.simulate(D), ((2000, 3), (2003, 11)))
    ms2, g2, c2 = G.monthly(D, run(D, "DV20-M"), ((2000, 3), (2003, 11)))
    flag = np.array([bool(on[D.last[G.prv(m)]]) for m in ms])
    assert 3 < flag.sum() < len(ms) - 3
    assert np.array_equal(g2[~flag], g[~flag]) and np.array_equal(c2[~flag], c[~flag]) and not g2[flag].any() and not c2[flag].any()


@test
def daily_filter_holds_exactly_when_the_previous_close_was_off():
    D, days, fx, rate = _random_world()
    for f in ("V80", "DV20", "CS20"):
        on = G.conditions(D)[f]
        s = run(D, f"{f}-D")
        i0 = D.first[(2000, 2)]
        assert np.array_equal(s["held"][i0:], ~on[i0 - 1:-1]), f
        n_exit = int((s["held"][i0:-1] & ~s["held"][i0 + 1:]).sum())
        assert n_exit == s["exit"].sum() and n_exit > 5, f


@test
def weekly_filter_changes_only_after_week_ends():
    D, days, fx, rate = _random_world()
    s = run(D, "V80-W")
    chk = G.check_days(D, "W")
    change = np.flatnonzero(s["held"][1:] != s["held"][:-1]) + 1
    first = D.first[(2000, 2)]
    assert len(change) > 5 and all(chk[i - 1] for i in change if i != first)


@test
def want_keeps_the_state_between_checks():
    D, days = world(vix={"2021-06-09": 30})                                               # on for one Wednesday close only
    cond = G.conditions(D)["DV20"]
    w = G.want(D, cond, "W", None)
    assert w.all()                                                                        # Friday sees 20 again
    d = G.want(D, cond, "D", None)
    assert not d[idx(days, "2021-06-09")] and d.sum() == len(days) - 1


# ---------------------------------------------------------------- statistics
@test
def sharpe_drawdown_worst12_by_hand():
    x = np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.00, 0.01, -0.03, 0.02, 0.01, 0.01, -0.01, 0.04])
    assert close(G.sharpe(x), x.mean() / x.std(ddof=1) * math.sqrt(12))
    assert close(G.max_drawdown(np.array([0.05, -0.02, -0.04, 0.03, -0.01])), 0.06)
    assert close(G.max_drawdown(np.array([-0.01, -0.02])), 0.03) and G.max_drawdown(np.array([0.01, 0.02])) == 0
    assert close(G.worst12(x), min(x[:12].sum(), x[1:].sum())) and math.isnan(G.worst12(x[:5]))
    assert G.sharpe(np.zeros(5)) == 0.0


@test
def stats_of_a_constant_carry_world():
    D, _ = world(a="2019-12-02", b="2021-12-31")
    r = G.stats(D, run(D, "C0"), ((2020, 1), (2021, 11)))
    assert r["months"] == 23 and r["flat_share"] == 0 and r["exits_per_year"] == 0 and close(r["entries_per_year"], 12)
    assert close(r["cost_sum"], 23 * 0.0008) and close(r["gross_sum"], 0.03 * 700 / 365)   # 2020-01-01 -> 2021-12-01
    assert close(r["cost_share"], 23 * 0.0008 / (0.03 * 700 / 365)) and close(r["net_sum"], r["gross_sum"] - r["cost_sum"])


@test
def stats_counts_exits_and_flat_days():
    (D, days), _ = _spike()
    r = G.stats(D, run(D, "DV20-D"), ((2021, 6), (2021, 6)))
    assert close(r["exits_per_year"], 12) and close(r["mid_exits_per_year"], 12) and close(r["mid_entries_per_year"], 12)
    assert close(r["entries_per_year"], 24) and close(r["flat_share"], 6 / 22)            # flat 06-10 .. 06-17 of 22 days
    r = G.stats(D, run(D, "DV20-M"), ((2021, 6), (2021, 6)))
    assert r["flat_share"] == 0 and r["exits_per_year"] == 0


@test
def stats_daily_drawdown_sees_inside_the_month():
    aud = {d: 1.0 for d in bdays("2021-01-01", "2021-12-31")}
    aud.update({d: 0.7 for d in bdays("2021-06-10", "2021-06-20")})                       # falls and recovers inside June
    D, days = world(fx={"AUD": aud})
    r = G.stats(D, run(D, "C0"), ((2021, 5), (2021, 7)))
    assert r["max_drawdown"] == 0 and r["max_drawdown_daily"] > 0.09


@test
def table_halves_and_doubled_cost():
    D, _ = world(a="2001-10-01", b="2002-04-30")
    s1, s2 = G.run_all(D, names=("C0",)), G.run_all(D, 0.0008, names=("C0",))
    t = G.table(D, s1, ((2001, 11), (2002, 3)), s2)["C0"]
    assert len(t["net"]) == 5 and close(t["half1_sum"], sum(t["net"][:2])) and close(t["half2_sum"], sum(t["net"][2:]))
    assert t["sharpe_cost2"] != t["sharpe"] and close(sum(t["net"]), t["net_sum"])


@test
def vix_change_and_correlation():
    D, _ = world(vix={"2021-05-31": 25, "2021-06-30": 22})
    assert G.vix_change(D, ((2021, 5), (2021, 7))) == [5.0, -3.0, -2.0]
    assert close(G.corr([1, 2, 3, None], [2, 4, 6, 9]), 1.0) and G.corr([1, 2], [1, 2]) is None and G.corr([1, 1, 1], [1, 2, 3]) is None


# ---------------------------------------------------------------- bootstrap, Holm, MDE
@test
def boot_index_blocks_are_circular_and_seeded():
    a, b = G.boot_index(158), G.boot_index(158)
    assert a.shape == (10_000, 158) and np.array_equal(a, b) and a.min() == 0 and a.max() == 157
    assert np.array_equal((a[:, 1:12] - a[:, :11]) % 158, np.ones((10_000, 11), dtype=int))
    assert not np.array_equal(G.boot_index(158, seed=18), a) and (a[:, 12] - a[:, 11] != 1).mean() > 0.9


@test
def boot_sharpe_matches_a_direct_loop():
    rng = np.random.default_rng(5)
    x = rng.normal(0.004, 0.02, 60)
    idx_ = G.boot_index(60, B=50)
    d = G.boot_sharpe(x, idx_)
    assert all(close(d[k], G.sharpe(x[idx_[k]]), 1e-12) for k in range(50))
    assert G.boot_sharpe(np.zeros(60), idx_).tolist() == [0.0] * 50


@test
def one_sided_p_and_bounds():
    dist = np.arange(-10, 990) / 100.0                                                    # 11 of 1000 are <= 0
    r = G.one_sided(4.9, dist, 0.05)
    assert close(r["p"], 0.011) and close(r["bound_at_level"], float(np.percentile(dist, 5))) and close(r["bound_97_5"], float(np.percentile(dist, 2.5)))


@test
def holm_levels_and_stopping():
    r = G.holm({"H1": 0.03, "H2": 0.01})
    assert r["H2"] == {"p": 0.01, "level": 0.025, "rejected": True} and r["H1"] == {"p": 0.03, "level": 0.05, "rejected": True}
    r = G.holm({"H1": 0.03, "H2": 0.04})
    assert not r["H1"]["rejected"] and not r["H2"]["rejected"] and r["H1"]["level"] == 0.025
    assert G.holm({"H1": 0.04})["H1"] == {"p": 0.04, "level": 0.05, "rejected": True}
    assert not G.holm({"H1": 0.05})["H1"]["rejected"]


@test
def mde_formula():
    assert close(G.mde(math.sqrt(12 / 158), 0.05), 0.685, 5e-4) and close(G.mde(math.sqrt(12 / 158), 0.025), 0.772, 5e-4)


# ---------------------------------------------------------------- A* selection
def _rows(**kw):
    rows = {c: {"sharpe": 0.30, "half1_sum": 1.0, "half2_sum": 1.0, "flat_share": 0.2} for c in G.CANDS}
    rows["C0"]["flat_share"] = 0.0
    for c, v in kw.items():
        rows[c.replace("_", "-")].update(v)
    return rows


@test
def select_c0_when_nothing_is_005_better():
    assert G.select_astar(_rows(V80_M={"sharpe": 0.349}))[0] == "C0"
    assert G.select_astar(_rows(V80_M={"sharpe": 0.35}))[0] == "V80-M"


@test
def select_skips_a_candidate_with_a_losing_half():
    a, rec = G.select_astar(_rows(V80_M={"sharpe": 0.60, "half2_sum": -0.1}, CS20_D={"sharpe": 0.40}))
    assert a == "CS20-D" and "V80-M" in rec["failed_halves"]
    assert G.select_astar(_rows(V80_M={"sharpe": 0.60, "half1_sum": 0.0}))[0] == "C0"


@test
def select_tie_prefers_the_lower_frequency():
    a, rec = G.select_astar(_rows(V80_D={"sharpe": 0.50}, DV20_W={"sharpe": 0.47}, CS20_M={"sharpe": 0.451}))
    assert a == "CS20-M" and set(rec["tie"]) == {"V80-D", "DV20-W", "CS20-M"}
    assert G.select_astar(_rows(V80_D={"sharpe": 0.50}, CS20_M={"sharpe": 0.449}))[0] == "V80-D"


@test
def select_tie_then_fewer_flat_days_then_order():
    a, _ = G.select_astar(_rows(V80_W={"sharpe": 0.50, "flat_share": 0.3}, DV20_W={"sharpe": 0.47, "flat_share": 0.1}))
    assert a == "DV20-W"
    assert G.select_astar(_rows(V90_W={"sharpe": 0.50}, V70_W={"sharpe": 0.50}))[0] == "V70-W"


@test
def select_tie_member_must_also_beat_c0_by_005():
    a, rec = G.select_astar(_rows(V80_D={"sharpe": 0.37}, CS20_M={"sharpe": 0.34}))       # M is within 0.05 of D but not of C0
    assert a == "V80-D" and rec["tie"] == ["V80-D"]


@test
def select_c0_when_no_filter_passes_the_halves():
    rows = _rows()
    for c in G.CANDS[1:]:
        rows[c].update(sharpe=0.9, half1_sum=-1.0)
    assert G.select_astar(rows)[0] == "C0"


# ---------------------------------------------------------------- files
@test
def read_rows_and_side_of_2013():
    with tempfile.TemporaryDirectory() as root:
        for part, rows in (("build", "2012-12-28,1.5\n2012-12-31,.\n"), ("sealed", "2013-01-01,\n2013-01-02,1.25\n")):
            os.makedirs(os.path.join(root, part))
            with open(os.path.join(root, part, "X.csv"), "w") as f:
                f.write("observation_date,X\n" + rows)
        read = []
        assert G.series_rows("X", ("build",), root, read) == [("2012-12-28", "1.5"), ("2012-12-31", "")]
        assert read == [os.path.join(root, "build", "X.csv")]
        assert G.series_rows("X", ("build", "sealed"), root)[-1] == ("2013-01-02", "1.25")
        with open(os.path.join(root, "build", "X.csv"), "a") as f:
            f.write("2013-01-02,9\n")
        try:
            G.series_rows("X", ("build",), root)
            raise AssertionError("a 2013 row in build/ must stop the load")
        except ValueError:
            pass


@test
def values_keep_their_decimals():
    D, days = world(vix={"2021-06-09": "24.005", "2021-06-10": "24.0"}, baa={"2021-06-09": "2.2001", "2021-06-10": "2.2"})
    on = G.conditions(D)
    assert on["DV20"][idx(days, "2021-06-09")] and not on["DV20"][idx(days, "2021-06-10")]
    assert on["CS20"][idx(days, "2021-06-09")] and not on["CS20"][idx(days, "2021-06-10")]


@test
def candidate_list():
    assert len(G.CANDS) == 22 and G.CANDS[0] == "C0" and G.CANDS[1:4] == ("V70-M", "V70-W", "V70-D") and G.CANDS[-1] == "CS40-D"
    assert G.kind("C0") == ("", "") and G.kind("DV40-W") == ("DV40", "W")
    assert G.months(*G.SEALED)[0] == (2013, 1) and len(G.months(*G.SEALED)) == 158 and len(G.months(*G.DEV)) == 276


# ---------------------------------------------------------------- loader on a made-up folder (build + sealed parts)
def make_tree(root, a="2010-01-01", b="2015-12-31", seed=7, quirks=True):
    """Writes a fake data folder (FRED-style files split at 2013-01-01, plus candidates/build) and a manifest folder.
    quirks: the things the real sealed files may hold (an empty US month, one currency without a quote on a day,
    indicator values with more decimals, a '.' for a missing value). Returns (here, data root, candidates root)."""
    rng = np.random.default_rng(seed)
    days = bdays(a, b, holidays=("2011-07-04", "2012-12-25", "2013-01-01", "2013-07-04", "2014-12-25", "2015-01-01"))
    ms = G.months(G.ym(a), G.ym(b))
    data, cand, here = os.path.join(root, "data"), os.path.join(root, "data", "candidates"), os.path.join(root, "here")
    for d in (os.path.join(data, "build"), os.path.join(data, "sealed"), os.path.join(cand, "build"), here):
        os.makedirs(d)
    man = {"SEALED_MANIFEST19.tsv": [], "SEALED_MANIFEST19_h10.tsv": []}

    def put(base, name, rows, which):
        for part in ("build", "sealed"):
            keep = [x for x in rows if (x[0] < "2013-01-01") == (part == "build")]
            if base == cand and part == "sealed":
                continue
            path = os.path.join(base, part, f"{name}.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"observation_date,{name}\n" + "".join(f"{d},{v}\n" for d, v in keep))
            man[which].append(f"{name}\t{part}\t{part}/{name}.csv\t{len(keep)}\t\t\t{G.sha(path)}\n")

    for c, (s, usd) in G.FX.items():
        lvl = (0.5 + rng.random()) * np.exp(np.cumsum(rng.normal(0, 0.006, len(days))))
        rows = [(d, f"{(v if usd else 1 / v):.4f}") for d, v in zip(days, lvl)]
        if quirks and c == "NOK":
            rows = [(d, "" if d == "2013-06-12" else v) for d, v in rows]                 # one currency, one day, sealed part
        if quirks and c == "SEK":
            rows = [(d, "." if d == "2011-03-09" else v) for d, v in rows]
        put(data, s, rows, "SEALED_MANIFEST19.tsv")
    for k, v in G.RATE.items():
        rows = [(f"{m[0]}-{m[1]:02d}-01", f"{rng.uniform(0, 8):.4f}") for m in ms]
        if quirks and k == "USD":
            rows = [(d, "" if d == "2014-04-01" else x) for d, x in rows]                 # an empty US month, sealed part
        put(data, f"IR3TIB01{v}M156N", rows, "SEALED_MANIFEST19.tsv")
    put(data, "IR3TIB01DEM156N", [(f"{m[0]}-{m[1]:02d}-01", "3.0000") for m in ms], "SEALED_MANIFEST19.tsv")
    vix = [(d, f"{12 + 25 * rng.random():.2f}") for d in days if d != "2013-03-29"]
    if quirks:
        vix = [(d, v + "5" if d >= "2014-01-01" else v) for d, v in vix]                  # three decimals in the sealed part
    put(data, "VIXCLS", vix, "SEALED_MANIFEST19.tsv")
    put(data, "BAA10Y", [(d, f"{2 + rng.random():.2f}") for d in days], "SEALED_MANIFEST19.tsv")
    put(cand, G.DEM_SERIES, [("1998-12-30", "1.6700"), ("1998-12-31", "1.6670")], "SEALED_MANIFEST19_h10.tsv")
    put(cand, G.JP_CD_SERIES, [("2002-02-01", "0.1000"), ("2002-03-01", "0.1000")], "SEALED_MANIFEST19_h10.tsv")
    for f, lines in man.items():
        with open(os.path.join(here, f), "w", encoding="utf-8") as fh:
            fh.write("series\tpart\tpath\trows\tfirst_date\tlast_date\tsha256\n" + "".join(lines))
    return here, data, cand


@test
def load_build_only_never_opens_the_sealed_part():
    with tempfile.TemporaryDirectory() as root:
        here, data, cand = make_tree(root)
        D, read = G.load(("build",), root=data, cand=cand)
        assert D.days[-1] == "2012-12-31" and len(read) == 24 and not any(os.sep + "sealed" + os.sep in p for p in read)
        assert G.verify(read, here, data, cand) == []


@test
def load_sealed_run_reads_21_sealed_files_and_joins_them():
    with tempfile.TemporaryDirectory() as root:
        here, data, cand = make_tree(root)
        D, read = G.load(("build", "sealed"), root=data, cand=cand)
        sealed = [p for p in read if os.sep + "sealed" + os.sep in p]
        assert len(sealed) == 21 and len(read) == 45 and not any(p.startswith(cand) and "sealed" in p for p in read)
        assert not any("IR3TIB01DEM156N" in p for p in sealed)                            # the German rate is not needed after 1998
        assert D.days[-1] == "2015-12-31" and D.days == sorted(set(D.days)) and "2013-01-01" not in D.days
        assert G.verify(read, here, data, cand) == []


@test
def verify_names_a_changed_file():
    with tempfile.TemporaryDirectory() as root:
        here, data, cand = make_tree(root)
        D, read = G.load(("build",), root=data, cand=cand)
        bad = os.path.join(data, "build", "DEXJPUS.csv")
        with open(bad, "a") as f:
            f.write("2012-12-31,99\n")
        assert G.verify(read, here, data, cand) == [bad] and G.verify([os.path.join(data, "build", "nope.csv")][:0], here, data, cand) == []


@test
def full_data_reproduces_the_build_only_run_before_the_last_month():
    with tempfile.TemporaryDirectory() as root:
        here, data, cand = make_tree(root)
        A, _ = G.load(("build",), root=data, cand=cand)
        B, _ = G.load(("build", "sealed"), root=data, cand=cand)
        n = len(A.days)
        assert B.days[:n] == A.days and np.array_equal(A.S, B.S[:n], equal_nan=True) and A.vix == B.vix[:n] and A.baa == B.baa[:n]
        ca, cb = G.conditions(A), G.conditions(B)
        assert all(np.array_equal(ca[f], cb[f][:n]) for f in G.FILTERS)
        sa, sb = G.run_all(A), G.run_all(B)
        w = ((2010, 2), (2012, 11))
        for name in G.CANDS:
            x, y = G.monthly(A, sa[name], w), G.monthly(B, sb[name], w)
            assert x[0] == y[0] and np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2]), name
        assert G.incomplete(A, ((2012, 12), (2012, 12))) == [(2012, 12)] and G.incomplete(B, ((2012, 12), (2012, 12))) == []
        last = ((2012, 12), (2012, 12))
        assert G.monthly(A, sa["C0"], last)[1][0] != G.monthly(B, sb["C0"], last)[1][0]   # one more interval with the full data


@test
def sealed_part_quirks_are_handled():
    with tempfile.TemporaryDirectory() as root:
        here, data, cand = make_tree(root)
        D, _ = G.load(("build", "sealed"), root=data, cand=cand)
        assert (2014, 4) not in D.rate["USD"] and G.rate_at(D.rate["USD"], (2014, 4)) == D.rate["USD"][(2014, 3)]
        assert G.portfolio(D, (2014, 5)) is not None and G.portfolio(D, (2014, 6)) is not None
        i, j = D.days.index("2013-06-12"), G.CCY.index("NOK")
        assert not D.fresh[i, j] and D.S[i, j] == D.S[i - 1, j] and D.fresh[i, :j].all()
        k = D.days.index("2013-03-29")
        assert D.vix[k] == D.vix[k - 1] and (D.vix[D.days.index("2014-01-02")] * 100).denominator != 1   # third decimal kept
        s = G.run_all(D)
        for name in G.CANDS:
            assert np.isfinite(s[name]["pnl"]).all() and np.isfinite(s[name]["cst"]).all(), name
        t = G.table(D, s, ((2013, 1), (2015, 11)))
        assert all(r["months"] == 35 and math.isfinite(r["sharpe"]) for r in t.values())
        assert any(t[c]["net"] != t["C0"]["net"] for c in G.CANDS[1:])


def main():
    res = []
    for f in TESTS:
        try:
            f()
            res.append({"test": f.__name__, "passed": True})
        except Exception as e:                                           # noqa: BLE001
            res.append({"test": f.__name__, "passed": False, "error": f"{type(e).__name__}: {e}"})
            print("FAIL", f.__name__, type(e).__name__, e)
    n = sum(r["passed"] for r in res)
    out = {"written": dt.datetime.now().isoformat(timespec="seconds"), "passed": n, "total": len(res), "tests": res}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "engine19_tests.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"{n}/{len(res)} passed")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
