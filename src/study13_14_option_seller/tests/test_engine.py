"""Engine tests on synthetic data only (memo 4b v2 9; SPEC 13 M1, M3-M12, M14, M15, M17-M19; A13-3 E13 tests and the
holdout rehearsal). No real data file is opened."""
import copy
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import accounting as A
import chain as ch
import config13 as C
import costs
import ladder
import market
import run_dev
import run_holdout
import select13
import signals
import stats13
import synth13

_W = {}


def world(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _W:
        _W[key] = synth13.World(**kw)
    return copy.deepcopy(_W[key])


def run_world(w, start, end, settings=C.SETTINGS, **kw):
    mkt = market.Market(w, "SPX", ("dev",))
    return mkt, ladder.run(mkt, settings, start, end, **kw)


def closes(mkt, days):
    return [mkt.close(d)[0] for d in days]


@pytest.fixture(scope="module")
def base():
    w = world(start=20110103, end=20110930, seed=3)
    w.drop_day(20110301)                                       # defect (i) and full fallback day
    w.invalidate(20110322, 20110416)
    w.invalidate(20110323, 20110416)
    w.invalidate(20110324, 20110416)
    w.opp.loc[w.opp.date == 20110405, "delta"] = np.nan        # Black-Scholes delta substitution
    m = (w.opp.date == 20110406) & (w.opp.exdate == 20110521)
    w.opp.loc[m, ["delta", "impl_volatility"]] = np.nan        # previous delta
    w.missing_close.add(20110510)
    mkt, eng = run_world(w, 20110103, 20110930)
    return w, mkt, eng


# ---------------------------------------------------------------- M1, M5, hedge
def test_m1_identity(base):
    _, _, eng = base
    for e in eng.values():
        assert e.ledger.identity(e.liability()) <= 1e-9


def test_m5_cost_monotone_every_day(base):
    _, _, eng = base
    for e in eng.values():
        df = e.ledger.frame()
        assert (df.P_stress <= df.P_base + 1e-9).all() and (df.P_base <= df.P_opt + 1e-9).all()
        for k in A.COMP:
            assert (df[f"{k}_stress"] <= df[f"{k}_base"] + 1e-9).all() and (df[f"{k}_base"] <= df[f"{k}_opt"] + 1e-9).all()


def test_spec_constants():
    assert C.KAPPA == {"base": 1.25, "stress": 1.50, "opt": 1.00} and C.FEE == {"base": 1.0, "stress": 2.0, "opt": 0.0}
    assert C.HEDGE_BP == {"base": 1e-4, "stress": 3e-4, "opt": 0.25e-4} and C.TAU == {"S": 0.25, "G": 0.50}
    assert (C.SEED, C.BLOCK, C.B_BOOT, C.SENS_BLOCKS, C.STATS_OFFSET) == (13, 21, 10_000, (10, 42), 43)
    assert [costs.fallback_buy(2.0, s) for s in C.SCEN] == [2.5, 3.0, 2.0]
    assert costs.sell_price(1.0, 1.2, "stress") == pytest.approx(1.1 - 1.5 * 0.1)


def test_hedge_pnl_attributed_next_day(base):
    _, mkt, eng = base
    for e in eng.values():
        df = e.ledger.frame()
        S = closes(mkt, df.date.tolist())
        exp = [0.0] + [df.Hd.iloc[i - 1] * (S[i] - S[i - 1]) for i in range(1, len(df))]
        assert np.array_equal(df.hedge_pnl.to_numpy(), np.array(exp))


def test_hedge_is_plus_sum_q_delta_on_entry_day():
    w = world(start=20110103, end=20110331, seed=5)
    mkt, eng = run_world(w, 20110103, 20110331, [C.setting("O0")])
    e = eng["O0"]
    legs = pd.DataFrame(e.ledger.legs)
    d0 = int(legs.date.min())
    first = legs[(legs.date == d0) & (legs.event == "open")]
    rows = w.opp[(w.opp.date == d0) & w.opp.optionid.isin(first.optionid)].set_index("optionid")
    exp = sum(q * 100.0 * rows.loc[o, "delta"] for o, q in zip(first.optionid, first.q))
    assert e.ledger.frame().set_index("date").loc[d0, "Hd"] == pytest.approx(exp, rel=1e-12)


def test_fees_per_contract_per_side(base):
    _, _, eng = base
    for e in eng.values():
        legs = pd.DataFrame(e.ledger.legs)
        df = e.ledger.frame()
        for sc in C.SCEN:
            assert df[f"fees_{sc}"].sum() == pytest.approx(-C.FEE[sc] * legs.q.sum(), rel=1e-12, abs=1e-9)


def test_forward_is_parity_on_tau025_chain(base):
    w, _, _ = base
    mkt = market.Market(copy.deepcopy(w), "SPX", ("dev",))
    v = mkt.day(20110105)
    for e in v.std_exdates[:3]:
        F, src, r, T = v.forward(int(e))
        F2, src2 = ch.forward(ch.dedupe(v.expiry_rows(int(e)), 0.25), r, T, v.S)
        assert (F, src) == (F2, src2) and src == "parity"


# ---------------------------------------------------------------- M3, M4
def _p0(df):
    return (df.mark + df.hedge_pnl).to_numpy()


def test_m3_fair_iv_mean_zero():
    means = []
    for s in range(40):
        w = world(start=20110103, end=20110630, seed=100 + s, sigma_real=0.2, sigma_imp=0.2, mu=0.0, r=0.0, q=0.0,
                  spread=0.0, min_tick=2e-4)
        _, eng = run_world(w, 20110103, 20110630, [C.setting("O0")])
        df = eng["O0"].ledger.frame()
        assert df.opened.sum() > 100                            # not vacuous: tranches were opened
        means.append(_p0(df).mean())
    means = np.array(means)
    se = means.std(ddof=1) / math.sqrt(len(means))
    assert abs(means.mean()) <= 2.5 * se + 1e-6


def test_m4_rich_iv_positive_and_near_theory():
    got, theo = [], []
    for s in range(20):
        w = world(start=20110103, end=20110630, seed=200 + s, sigma_real=0.12, sigma_imp=0.24, mu=0.0, r=0.0, q=0.0,
                  spread=0.0, min_tick=2e-4)
        mkt, eng = run_world(w, 20110103, 20110630, [C.setting("O0")])
        e = eng["O0"]
        df = e.ledger.frame()
        legs = pd.DataFrame(e.ledger.legs)
        op = legs[legs.event == "open"].set_index("optionid")
        cl = legs[legs.event != "open"].groupby("optionid").date.min()
        days = df.date.tolist()
        S = closes(mkt, days)
        th = np.zeros(len(days))
        for oid, r in op.iterrows():
            end = cl.get(oid, days[-1] + 1)
            for i in range(1, len(days)):
                if r.date < days[i] <= end:
                    T = (ch.ymd(mkt.cal.settlement(int(r.exdate))) - ch.ymd(days[i - 1])).days / 365.0
                    sd = 0.24 * math.sqrt(T)
                    d1 = (math.log(S[i - 1] / r.K) + 0.5 * sd * sd) / sd
                    gamma = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (S[i - 1] * sd)
                    th[i] += 0.5 * gamma * S[i - 1] ** 2 * (0.24 ** 2 - 0.12 ** 2) / 252 * r.q * 100
        got.append(_p0(df).mean())
        theo.append(th.mean())
    assert np.mean(got) > 0
    assert 0.5 <= np.mean(got) / np.mean(theo) <= 2.0


# ---------------------------------------------------------------- M6, M7, M8
def test_m6_no_lookahead_in_gates_and_sizing():
    d = 20110601
    w1 = world(start=20110103, end=20110801, seed=11, iv_path=lambda n: 0.15 + 0.1 * np.sin(np.arange(n) / 9))
    w2 = copy.deepcopy(w1)
    k = w2.days.index(d)
    w2.iv[k:] *= 1.7
    w2.iv91[k:] *= 0.6
    w2.S[k:] *= 1.05
    w2.closes = pd.Series(w2.S, index=w2.days)
    w2.opp.loc[w2.opp.date > d, ["best_bid", "best_offer"]] *= 1.3
    s = [C.setting(x) for x in ("O3", "O4", "O5", "O6", "O7", "O10")]
    _, e1 = run_world(w1, 20110103, 20110801, s)
    _, e2 = run_world(w2, 20110103, 20110801, s)
    for sid in e1:
        a, b = e1[sid].ledger.frame().set_index("date"), e2[sid].ledger.frame().set_index("date")
        assert a.loc[:d, "reason"].tolist() == b.loc[:d, "reason"].tolist()
        la, lb = pd.DataFrame(e1[sid].ledger.legs), pd.DataFrame(e2[sid].ledger.legs)
        oa = la[(la.event == "open") & (la.date <= d)][["date", "optionid", "q"]].reset_index(drop=True)
        ob = lb[(lb.event == "open") & (lb.date < d)][["date", "optionid", "q"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(oa[oa.date < d].reset_index(drop=True), ob)
        qa = la[(la.event == "open") & (la.date == d)].q.tolist()
        qb = lb[(lb.event == "open") & (lb.date == d)].q.tolist()
        if qa and qb:                                           # sizing on d uses the d-1 signal; S_d changed by 5%
            assert qa[0] * w1.closes[d] == pytest.approx(qb[0] * w2.closes[d], rel=1e-12)


def test_m7_tranche_count_and_notional():
    w = world(start=20110103, end=20111230, seed=21, spread=0.0, min_tick=2e-4, sigma_real=0.05, sigma_imp=0.20)
    _, eng = run_world(w, 20110103, 20111230, [C.setting("O0"), C.setting("O1")])
    for sid, e in eng.items():
        H = C.setting(sid).H
        df = e.ledger.frame()
        assert (df.n_tranches <= H + C.DEFER_MAX).all() and (df.notional_open <= C.N0 * (H + 3) / H + 1e-6).all()
        assert df.opened.sum() > 150 and df.kappa_exits.sum() == 0 and df.deferred.sum() == 0
        opened = df.opened.to_numpy()
        for i in range(H, len(df)):
            if opened[i - H + 1:i + 1].all() and df.n_tranches.iloc[i] == H:
                assert df.notional_open.iloc[i] == pytest.approx(C.N0, rel=1e-9)
        full = [i for i in range(H, len(df)) if opened[i - H + 1:i + 1].all()]
        assert full and all(df.n_tranches.iloc[i] <= H for i in full)
        assert any(df.n_tranches.iloc[i] == H for i in full)    # = H unless a DTE <= 5 exit came first


def kappa_day(cal_of, days, i_sched, exdate):
    """Expected kappa day of a leg invalid from its scheduled day on: walk the processed business days."""
    for j in range(i_sched, len(days)):
        if ladder.last_chance(cal_of(days[j]), days[j], exdate, j - i_sched):
            return days[j]
    return None


def test_m8_last_chance_rules():
    sat = ch.Calendar(synth13.weekdays(20110101, 20111231, {20110318}))   # third Friday holiday, Saturday era
    assert sat.settlement(20110319) == 20110317 and sat.last_trading(20110319) == 20110316
    d = synth13.weekdays(20110301, 20110331, {20110318})
    assert kappa_day(lambda x: sat, d, d.index(20110314), 20110319) == 20110316
    fri = ch.Calendar(synth13.weekdays(20150101, 20151231, {20150914}))   # Monday holiday, Friday exdate
    d = synth13.weekdays(20150901, 20150930, {20150914})
    assert kappa_day(lambda x: fri, d, d.index(20150915), 20150918) == 20150916   # DTE >= 2 and <= last trading day
    assert kappa_day(lambda x: fri, d, d.index(20150917), 20150918) == 20150917   # E13-03: past the limit -> that day
    plain = ch.Calendar(synth13.weekdays(20150101, 20151231))
    d = synth13.weekdays(20150901, 20150930)
    assert kappa_day(lambda x: plain, d, d.index(20150903), 20151016) == 20150908  # three processed business days
    wed = ch.Calendar(synth13.weekdays(20150101, 20151231, {20150916}))   # holiday on exdate - 2 (Wednesday)
    d = synth13.weekdays(20150901, 20150930, {20150916})
    assert kappa_day(lambda x: wed, d, d.index(20150914), 20150918) == 20150915   # next business day Thu has DTE 1


def test_m8_exit_timing(base):
    w, mkt, eng = base
    for sid in ("O0", "O1", "O2"):
        H = C.setting(sid).H
        legs = pd.DataFrame(eng[sid].ledger.legs)
        op = legs[legs.event == "open"].set_index(["tranche", "optionid"])
        ex = legs[legs.event != "open"].set_index(["tranche", "optionid"])
        days = mkt.days
        assert len(ex) > 50
        for key, r in ex.iterrows():
            o = op.loc[key]
            i0 = days.index(int(o.date))
            js = next(j for j in range(i0 + 1, len(days))
                      if j - i0 >= H or (ch.ymd(int(o.exdate)) - ch.ymd(days[j])).days <= C.DTE_EXIT)
            assert r.sched == days[js]
            kd = kappa_day(lambda x: mkt.cal, days, js, int(o.exdate)) or days[-1]
            assert days[js] <= r.date <= kd
            if r.event == "kappa":
                assert r.date == kd
            assert r.date <= mkt.cal.last_trading(int(o.exdate))


def test_n1_boundary_exit_on_the_actual_h_th_business_day():
    w = world(start=20120103, end=20121231, hold_end=20130329, seed=33, spread=0.0, min_tick=2e-4, sigma_real=0.05,
              sigma_imp=0.20, holidays=(20130121,))                  # Jan 1 and Jan 21 2013: weekday holidays after the
    mkt = market.Market(w, "SPX", ("dev",))                          # development calendar end
    s = [C.setting("O0"), C.setting("O1")]
    eng = ladder.run(mkt, s, 20120103, 20121231)
    w.enable_holdout()
    mkt.add_split("holdout")
    eng = ladder.run(mkt, s, 20130102, 20130329, forced_last_day=True, entry_on_last_day=False, engines=eng)
    days, n = mkt.days, 0
    for e in eng.values():
        H = e.s.H
        legs = pd.DataFrame(e.ledger.legs)
        op = legs[legs.event == "open"].set_index(["tranche", "optionid"])
        for key, r in legs[(legs.event == "normal") & (legs.date >= 20130102)].iterrows():
            o = op.loc[(r.tranche, r.optionid)]
            if o.date > 20121231:
                continue
            i0 = days.index(int(o.date))
            j = next(j for j in range(i0 + 1, len(days))
                     if j - i0 >= H or (ch.ymd(int(o.exdate)) - ch.ymd(days[j])).days <= C.DTE_EXIT)
            assert r.date == days[j]
            n += 1
    assert n > 5                                                     # tranches opened before the boundary, closed after


def test_n1_deferral_counted_in_processed_days_across_the_boundary():
    w = world(start=20120103, end=20121231, hold_end=20130329, seed=34)
    for d in synth13.weekdays(20121210, 20130115):
        if d != 20130101:
            w.invalidate(d, 20130119)
    mkt = market.Market(w, "SPX", ("dev",))
    s = [C.setting("O0"), C.setting("O1")]
    eng = ladder.run(mkt, s, 20120103, 20121231)
    w.enable_holdout()
    mkt.add_split("holdout")
    eng = ladder.run(mkt, s, 20130102, 20130329, forced_last_day=True, entry_on_last_day=False, engines=eng)
    days, n = mkt.days, 0
    for e in eng.values():
        legs = pd.DataFrame(e.ledger.legs)
        k = legs[(legs.event == "kappa") & (legs.exdate == 20130119) & (legs.date >= 20130102)]
        for _, r in k.iterrows():
            i = days.index(int(r.sched))
            if r.sched <= 20121231 and ch.ymd(20130119).toordinal() - ch.ymd(days[i + 3]).toordinal() >= 2:
                assert r.date == days[i + 3] and r.k == 3            # 3 processed days: Jan 1 is not counted
                n += 1
    assert n > 0


def test_m8_exit_exactly_on_schedule_without_deferrals():
    hol = (20110201, 20110302, 20110405, 20110503, 20110607, 20110705, 20110802, 20110906, 20111004, 20111101, 20111206)
    w = world(start=20110103, end=20111230, seed=21, spread=0.0, min_tick=2e-4, sigma_real=0.05, sigma_imp=0.20,
              holidays=hol)                                     # holidays make DTE <= 5 come before H days
    mkt, eng = run_world(w, 20110103, 20111230, [C.setting("O0"), C.setting("O1")])
    dte_first = 0
    for sid, e in eng.items():
        H = C.setting(sid).H
        legs = pd.DataFrame(e.ledger.legs)
        op = legs[legs.event == "open"].set_index(["tranche", "optionid"])
        ex = legs[legs.event != "open"].set_index(["tranche", "optionid"])
        assert len(ex) > 100 and (ex.event == "normal").all()
        for key, r in ex.iterrows():
            o = op.loc[key]
            i0 = mkt.days.index(int(o.date))
            j = next(j for j in range(i0 + 1, len(mkt.days))
                     if j - i0 >= H or (ch.ymd(int(o.exdate)) - ch.ymd(mkt.days[j])).days <= C.DTE_EXIT)
            assert r.date == mkt.days[j]
            dte_first += j - i0 < H
    assert dte_first > 0                                        # some exits came from DTE <= 5 before H days


def test_m8_holiday_week_deferral_to_wednesday():
    w = world(start=20110103, end=20110331, seed=7, holidays=(20110318,))
    for d in synth13.weekdays(20110310, 20110317, {20110318}):
        w.invalidate(d, 20110319)
    mkt, eng = run_world(w, 20110103, 20110331, [C.setting("O1")])
    legs = pd.DataFrame(eng["O1"].ledger.legs)
    mar = legs[(legs.exdate == 20110319) & (legs.event != "open")]
    assert len(mar) and (mar.date <= 20110316).all() and (mar.event.isin(["kappa", "normal"])).all()


# ---------------------------------------------------------------- M9, E13-04..06
def _manual(w, days, opens):
    """One engine for O0 stepped by hand; opens = set of days with an entry."""
    mkt = market.Market(w, "SPX", ("dev",))
    e = ladder.Engine(C.setting("O0"))
    views = []
    for d in days:
        v = mkt.day(d)
        e.step(v, (True, 1.0, "ok") if d in opens else (False, float("nan"), "gate"))
        views.append(v)
    return mkt, e, views


def test_m9_fallback_uses_last_valid_iv_and_u_and_jump():
    w = world(start=20110103, end=20110228, seed=9)
    days = synth13.weekdays(20110103, 20110110)
    d0, d1, d2 = days[0], days[1], days[2]
    mkt0, e0, _ = _manual(copy.deepcopy(w), [d0], {d0})
    oids = [l.oid for l in e0.legs]
    call = next(l for l in e0.legs if l.cp == 1)
    m = (w.opp.date == d1) & (w.opp.optionid == call.oid)
    w.opp.loc[m, "best_bid"] = 0.0
    w.opp.loc[m, "impl_volatility"] = 0.99                      # today's IV must not be used (E13: last valid IV)
    mkt, e, views = _manual(w, [d0, d1, d2], {d0})
    v1 = views[1]
    F, src, r, T = v1.forward(call.exdate)
    exp = float(ch.bs(F, call.K, T, call.last_iv, r, v1.S, 1)[0])
    rec1, rec2 = e.ledger.rows[1], e.ledger.rows[2]
    assert rec1["q_fallback"] == pytest.approx(call.q) and rec1["u"] == pytest.approx(0.5)
    leg = next(l for l in e.legs if l.oid == call.oid)
    row2 = w.opp[(w.opp.date == d2) & (w.opp.optionid == call.oid)].iloc[0]
    mid2 = (row2.best_bid + row2.best_offer) / 2
    assert rec2["jump_sum"] == pytest.approx(-call.q * 100 * (mid2 - exp), rel=1e-12)
    assert leg.mark == pytest.approx(mid2)
    assert set(oids) == {l.oid for l in e.legs}


def test_m9_theta_sign():
    p1 = ch.bs(1000.0, 1000.0, 30 / 365, 0.2, 0.0, 1000.0, 1)[0]
    p2 = ch.bs(1000.0, 1000.0, 29 / 365, 0.2, 0.0, 1000.0, 1)[0]
    assert p2 < p1


def test_e13_04_missing_row_is_invalid_and_e13_06_delta():
    w = world(start=20110103, end=20110228, seed=9)
    days = synth13.weekdays(20110103, 20110107)
    d0, d1, d2 = days[0], days[1], days[2]
    _, e0, _ = _manual(copy.deepcopy(w), [d0], {d0})
    put = next(l for l in e0.legs if l.cp == -1)
    call = next(l for l in e0.legs if l.cp == 1)
    w.opp = w.opp[~((w.opp.date == d1) & (w.opp.optionid == put.oid))].reset_index(drop=True)   # row missing
    w.opp.loc[(w.opp.date == d1) & (w.opp.optionid == call.oid), "delta"] = np.nan                # BS delta
    m2 = (w.opp.date == d2) & (w.opp.optionid == call.oid)
    w.opp.loc[m2, ["delta", "impl_volatility"]] = np.nan                                          # previous delta
    _, e, views = _manual(w, [d0, d1, d2], {d0})
    r1, r2 = e.ledger.rows[1], e.ledger.rows[2]
    assert r1["q_fallback"] == pytest.approx(put.q)                     # missing row -> invalid -> fallback
    assert r1["delta_bs"] == 1 and r1["delta_prev"] == 1                 # call: BS with today's IV; put: no row
    assert r2["delta_prev"] == 1
    v1 = views[1]
    F, _, r, T = v1.forward(call.exdate)
    iv1 = float(w.opp[(w.opp.date == d1) & (w.opp.optionid == call.oid)].impl_volatility.iloc[0])
    exp_dc = float(ch.bs(F, call.K, T, iv1, r, v1.S, 1)[1])
    put_prev = [x for x in e0.legs if x.oid == put.oid][0].delta
    assert r1["Hd"] == pytest.approx(call.q * 100 * exp_dc + put.q * 100 * put_prev, rel=1e-12)


def test_e13_05_u_weights_by_q_and_skips_new_legs():
    w = world(start=20110103, end=20110228, seed=9)
    d0, d1 = 20110103, 20110104
    mkt = market.Market(w, "SPX", ("dev",))
    e = ladder.Engine(C.setting("O0"))
    e.step(mkt.day(d0), (True, 1.0, "ok"))                      # tranche 1 (2 legs, q each)
    q1 = e.legs[0].q
    a, b = e.legs
    b.q = 3 * q1                                                 # unequal sizes: q-weighting differs from counting
    v = mkt.day(d1)
    v.rows = v.rows.copy()
    v.rows.loc[v.rows.optionid == b.oid, "best_bid"] = 0.0      # b falls back today
    e.step(v, (True, 1.0, "ok"))                                 # a new tranche opens today: not counted
    rec = e.ledger.rows[1]
    assert rec["opened"] == 1 and rec["u"] == pytest.approx(3 * q1 / (q1 + 3 * q1))


# ---------------------------------------------------------------- M10, M11
def test_m10_sizing():
    s = C.setting("O5")
    ms = [signals.decide(s, {"iv30": iv, "iv91": 1.0, "rv21": 0.1})[1] for iv in np.linspace(0.02, 2.0, 200)]
    assert min(ms) >= 0.25 and max(ms) <= 1.5 and all(a >= b for a, b in zip(ms, ms[1:]))
    assert signals.decide(s, {"iv30": float("nan"), "iv91": 1.0, "rv21": 0.1})[2] == "signal_missing"


def test_m11_bit_identical_rerun(base):
    w, _, eng = base
    w2 = copy.deepcopy(w)
    w2.access = []
    _, eng2 = run_world(w2, 20110103, 20110930)
    for sid in eng:
        pd.testing.assert_frame_equal(eng[sid].ledger.frame(), eng2[sid].ledger.frame(), check_exact=True)


# ---------------------------------------------------------------- M12, M14, M15, M17
def g(repo, *a):
    return subprocess.run(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True,
                          text=True, check=True)


def test_m12_clean_tree_and_lock_order(tmp_r13):
    repo = os.path.join(tmp_r13, "repo")
    study = os.path.join(repo, "research13")
    os.makedirs(os.path.join(study, "results"))
    subprocess.run(["git", "init", "-q", repo], check=True)
    open(os.path.join(study, ".gitignore"), "w").write("results/HOLDOUT_UNLOCKED\nlogs/\n")
    open(os.path.join(study, "a.py"), "w").write("x = 1\n")
    g(repo, "add", "research13/.gitignore", "research13/a.py")
    g(repo, "commit", "-q", "-m", "init")
    assert run_holdout.is_clean_at(study)
    open(os.path.join(study, "results", "HOLDOUT_UNLOCKED"), "w").close()
    assert run_holdout.is_clean_at(study)
    open(os.path.join(study, "new.tsv"), "w").close()
    assert not run_holdout.is_clean_at(study)
    called = []
    rd = os.path.join(tmp_r13, "res")
    os.makedirs(rd)
    why = run_holdout.guard(rd, lambda: True, lambda: True, lambda: called.append(1) or [])
    assert why == ["4: no HOLDOUT_UNLOCKED"] and not called          # data checks never run before the unlock


def test_m14_verdict_boundaries():
    J = stats13.judge
    assert J(0.5, 1e-9, 1.0)[0] == "a" and J(0.5, 0.0, 1.0)[0] == "c" and J(0.5 - 1e-12, 0.1, 1.0)[0] == "c"
    assert J(0.4, -0.1, 0.5 - 1e-12)[0] == "b" and J(0.4, -0.1, 0.5)[0] == "c"
    assert J(0.6, 0.1, 0.45) == ("c", "CI와 점추정 불일치")            # (a) and (b) together -> (c)
    assert stats13.downgrade("a", "", 0.05, 0.0) == ("a", "") and stats13.downgrade("a", "", 0.0501, 0.0)[0] == "c"
    assert stats13.downgrade("b", "", 0.12, 0.07)[0] == "b" and stats13.downgrade("b", "", 0.1201, 0.07)[0] == "c"
    assert stats13.tail_label("a", 1.5, 1.0) == "a-안정" and stats13.tail_label("a", 1.5000001, 1.0) == "a-꼬리주의"
    assert stats13.tail_label("c", 9, 1) == ""


def test_m15_bootstrap_coverage():
    rng = np.random.default_rng(15)
    T, rho, sr = 3190, 0.10, 0.5
    st = stats13.boot_starts(T, B=2000, block=21, seed=13)
    hit = 0
    for _ in range(1000):
        e = -(rng.gamma(1.0, 1.0, T + 200) - 1.0)
        y = np.empty_like(e)
        y[0] = e[0]
        for i in range(1, len(e)):
            y[i] = rho * y[i - 1] + e[i]
        y = y[200:] / math.sqrt(1 / (1 - rho ** 2)) + sr / math.sqrt(252)   # population Sharpe on the marginal sd
        sh = stats13.boot_sharpes(y, st, 21)
        lo, hi = np.percentile(sh, 2.5), np.percentile(sh, 97.5)
        hit += lo <= sr <= hi
    assert 0.92 <= hit / 1000 <= 0.98


def test_ci_uses_block_21_and_seed_13():
    x = np.random.default_rng(1).normal(0.001, 0.01, 800)
    assert C.BLOCK == 21 and C.SEED == 13
    assert stats13.ci(x) == stats13.ci(x, block=21, seed=13) != stats13.ci(x, block=10, seed=13)


def test_m17_each_refusal_blocks(tmp_r13):
    def setup(**over):
        d = os.path.join(tmp_r13, f"g{len(os.listdir(tmp_r13))}")
        os.makedirs(d)
        open(os.path.join(d, "HOLDOUT_UNLOCKED"), "w").close()
        json.dump({"candidate": {"setting": "O0"}}, open(os.path.join(d, "selected_config.json"), "w"))
        return d
    ok = dict(has_tag_fn=lambda: True, is_clean_fn=lambda: True, data_fn=lambda: [])
    d = setup()
    assert run_holdout.guard(d, **ok) == []
    d = setup()
    open(os.path.join(d, "holdout_summary.json"), "w").close()
    assert run_holdout.guard(d, **ok) == ["1: holdout results exist"]
    assert run_holdout.guard(setup(), **{**ok, "has_tag_fn": lambda: False}) == ["2: no dev-result-13 tag"]
    assert run_holdout.guard(setup(), **{**ok, "is_clean_fn": lambda: False}) == ["3: working tree not clean"]
    d = setup()
    os.remove(os.path.join(d, "HOLDOUT_UNLOCKED"))
    assert run_holdout.guard(d, **ok) == ["4: no HOLDOUT_UNLOCKED"]
    d = setup()
    json.dump({"candidate": None}, open(os.path.join(d, "selected_config.json"), "w"))
    assert run_holdout.guard(d, **ok) == ["5: no selected setting"]
    assert run_holdout.guard(setup(), **{**ok, "data_fn": lambda: ["6: MANIFEST.tsv differs"]}) == ["6: MANIFEST.tsv differs"]
    assert run_holdout.guard(setup(), **{**ok, "data_fn": lambda: ["7: holdout end date differs"]}) == \
        ["7: holdout end date differs"]


# ---------------------------------------------------------------- M18, E13-17
def test_m18_continuous_equals_dev_only_and_no_holdout_access():
    w = world(start=20120103, end=20121231, hold_end=20130329, seed=31)
    dev = copy.deepcopy(w)
    mkt_d = market.Market(dev, "SPX", ("dev",))
    e_d = ladder.run(mkt_d, C.SETTINGS, 20120103, 20121231)
    assert not dev.holdout_enabled and all(not (isinstance(a[2], int) and a[2] >= 2013) for a in dev.access)
    cont = copy.deepcopy(w)
    mkt_c = market.Market(cont, "SPX", ("dev",))
    e_c = ladder.run(mkt_c, C.SETTINGS, 20120103, 20121231)
    for sid in e_d:
        pd.testing.assert_frame_equal(e_d[sid].ledger.frame(), e_c[sid].ledger.frame(), check_exact=True)
    assert any(l.exdate > 20121231 for e in e_c.values() for l in e.legs)       # legs cross the boundary
    cont.enable_holdout()
    mkt_c.add_split("holdout")
    e_c = ladder.run(mkt_c, C.SETTINGS, 20130102, 20130329, forced_last_day=True, entry_on_last_day=False, engines=e_c)
    for e in e_c.values():
        assert not e.legs and e.ledger.rows[-1]["Hd"] == 0.0                     # E13-16
        assert e.ledger.identity(e.liability()) <= 1e-9


def test_market_refuses_later_year():
    w = world(start=20110103, end=20120330, seed=4)
    mkt = market.Market(w, "SPX", ("dev",))
    mkt.day(20110103)
    with pytest.raises(market.FileAccessError):
        mkt._sec_year(2012)


# ---------------------------------------------------------------- M19
def test_m19_defects(base):
    w, mkt, eng = base
    df = eng["O0"].ledger.frame().set_index("date")
    assert df.loc[20110301, "defect_i"] == 1 and df.loc[20110510, "defect_i"] == 1
    assert df.loc[20110301, "defect_ii"] == 1                          # no standard monthly at all that day
    gaps = df[df.sel_status == "gap"]
    assert len(gaps) and (gaps.defect_ii == 0).all()                    # structural gaps are not defects
    g3 = eng["O3"].ledger.frame()
    closed = g3[g3.reason.isin(["gate", "signal_missing"])]
    assert len(closed) and (closed.defect_ii == 0).all()
    w2 = world(start=20110103, end=20110228, seed=12)
    d = 20110120
    _, e0 = run_world(copy.deepcopy(w2), 20110103, d, [C.setting("O0")])
    ex = int(pd.DataFrame(e0["O0"].ledger.legs).query("date == @d and event == 'open'").exdate.iloc[0])
    w2.invalidate(d, ex)
    _, e = run_world(w2, 20110103, d, [C.setting("O0")])
    r = e["O0"].ledger.frame().set_index("date").loc[d]
    assert r.sel_status == "no_candidate" and r.defect_ii == 1
    rows = pd.DataFrame({"defect_i": [0, 0, 0, 0], "defect_ii": [0, 0, 0, 0], "u": [0.2, 0.2000001, 0.0, 0.1]})
    assert stats13.defect_ratio(rows) == 0.25


# ---------------------------------------------------------------- selection, statistics
def test_select_tie_rule_uses_the_best_only():
    m = lambda s: {"sharpe": s, "sharpe_h1": 1, "sharpe_h2": 1, "sharpe_stress": 1, "calmar": 1}   # noqa: E731
    met = {"O0": m(0.70), "O3": m(0.78), "O6": m(0.86), "O9": m(0.95)}
    assert select13.choose(met)[0] == "O6"                               # within 0.10 of 0.95: O6 (2), O9 (3)
    met = {"O1": m(0.60), "O2": m(0.65)}
    assert select13.choose(met)[0] == "O2"                               # same complexity: higher Sharpe
    met["O2"]["calmar"] = 0.329
    assert select13.choose(met)[0] == "O1"
    assert select13.choose({"O0": m(0.49)}) == (None, [])


def test_stats_start_is_plus_43():
    days = synth13.weekdays(20110103, 20111230)
    C_DEV = C.DEV_START
    try:
        C.DEV_START = days[2]
        assert run_dev.stats_start(days) == days[45]
    finally:
        C.DEV_START = C_DEV


def test_e13_12_drawdown_and_calmar_by_hand():
    x = np.array([0.01, -0.02, 0.005, -0.01, 0.03])
    assert stats13.max_drawdown(x) == pytest.approx(0.025)               # peak 0.01 -> trough -0.015
    t = stats13.tails(x)
    assert t["calmar"] == pytest.approx(x.mean() * 252 / 0.025) and t["worst_day"] == -0.02
    assert stats13.worst_window(x, 2) == pytest.approx(-0.015)


# ---------------------------------------------------------------- leg tracking by optionid (3.4)
def test_rule5_held_leg_keeps_its_expiry_chain_when_1_and_2_share_a_month():
    """A13-4 rule 5: the 3.4 (i) priority chooses entries only. A held leg on a Saturday (2) expiry whose month gets a
    third-Friday (1) expiry is still looked up and its forward still comes from parity on its own (ii)(iii) chain."""
    w = world(start=20110103, end=20110331, seed=9)
    d0, dd = 20110103, 20110110
    sat = 20110219
    _, e0, _ = _manual(copy.deepcopy(w), [d0], {d0})
    assert {l.exdate for l in e0.legs} == {sat}
    call = next(l for l in e0.legs if l.cp == 1)
    put = next(l for l in e0.legs if l.cp == -1)
    w.add_friday_monthly(sat, dd)
    m = (w.opp.date == dd) & (w.opp.optionid == call.oid)
    w.opp.loc[m, "best_bid"] = 0.0                                       # call falls back on dd (valuation uses F)
    w.opp.loc[(w.opp.date == dd) & (w.opp.optionid == put.oid), "delta"] = np.nan   # put: BS delta uses F
    days = synth13.weekdays(d0, dd)
    mkt, e, views = _manual(w, days, {d0})
    v = views[-1]
    assert sat not in set(v.std_exdates) and sat - 1 in set(v.std_exdates)          # the world does what it says
    assert sat in set(views[-2].std_exdates)
    F, src, r, T = v.forward(sat)
    allrows = w.opp[(w.opp.date == dd) & (w.opp.exdate == sat)].assign(K=lambda x: x.strike_price / 1000.0)
    F2, src2 = ch.forward(ch.dedupe(allrows, 0.25), r, T, v.S)
    assert src == "parity" and (F, src) == (F2, src2)
    rec = e.ledger.rows[-1]
    assert rec["fwd_spot"] == 0 and rec["q_fallback"] == pytest.approx(call.q) and rec["delta_bs"] == 1
    leg_c = next(l for l in e.legs if l.oid == call.oid)
    leg_p = next(l for l in e.legs if l.oid == put.oid)
    assert leg_c.mark == pytest.approx(float(ch.bs(F, call.K, T, call.last_iv, r, v.S, 1)[0]), rel=1e-12)
    rp = w.opp[(w.opp.date == dd) & (w.opp.optionid == put.oid)].iloc[0]
    assert leg_p.mark == pytest.approx((rp.best_bid + rp.best_offer) / 2, rel=1e-12)              # own row found
    assert leg_p.delta == pytest.approx(float(ch.bs(F, put.K, T, float(rp.impl_volatility), r, v.S, -1)[1]), rel=1e-12)
    sel = ladder.select(v, "S", 21)
    assert sel["status"] == "ok" and sel["exdate"] == sat - 1                                     # entries: (1) only


def test_8d_first_row_in_stored_order_nan_rates_dropped_and_counted():
    """A13-4 8d: rows sharing a key use the first row in stored order (opprcd (optionid, date), secprd date, vsurfd
    (date, days, delta, cp_flag), zerocd (date, days)); zerocd rows with NaN days or rate are dropped; counted."""
    w = world(start=20110103, end=20110331, seed=9)
    d0, d1, d2 = 20110103, 20110104, 20110105
    _, e0, _ = _manual(copy.deepcopy(w), [d0], {d0})
    oids = [l.oid for l in e0.legs]
    for d, first in ((d1, False), (d2, True)):
        dup = w.opp[(w.opp.date == d) & w.opp.optionid.isin(oids)].copy()
        dup[["best_bid", "best_offer"]] *= 1.1
        w.add_rows(dup, first=first)
    w.extra["secprd"] += [({"date": d1, "close": 1.0}, False), ({"date": d2, "close": 2000.0}, True)]
    vs = {"days": 30, "delta": 50, "cp_flag": 1}
    w.extra["vsurfd"] += [({"date": d1, **vs, "impl_volatility": 0.9}, False),
                          ({"date": d2, **vs, "impl_volatility": 0.3}, True)]
    w.extra["zerocd"] += [({"date": d1, "days": 180.0, "rate": np.nan}, False),
                          ({"date": d1, "days": np.nan, "rate": 9.0}, False),
                          ({"date": d2, "days": 30.0, "rate": 5.0}, True), ({"date": d2, "days": 365.0, "rate": 9.0}, False)]
    mkt, e, views = _manual(w, [d0, d1, d2], {d0})
    assert mkt.close(d1)[0] == pytest.approx(w.closes[d1]) and mkt.close(d2)[0] == 2000.0
    assert mkt.iv(d1, 30) == pytest.approx(w.iv[1]) and mkt.iv(d2, 30) == 0.3
    assert mkt.rates.rate(d1, 180.0) == pytest.approx(w.r)
    assert mkt.rates.rate(d2, 30.0) == pytest.approx(0.05) and mkt.rates.rate(d2, 365.0) == pytest.approx(w.r)
    for v in views[1:]:                                    # d1: duplicates stored after, d2: stored before
        first = w.opp[(w.opp.date == v.d) & w.opp.optionid.isin(oids)].drop_duplicates("optionid", keep="first")
        got = {int(r.optionid): float(r.best_bid) for r in v.leg_rows(oids)}
        assert got == {int(r.optionid): float(r.best_bid) for r in first.itertuples()}
    o2 = w.opp[(w.opp.date == d2) & (w.opp.optionid == oids[0])].best_bid.to_numpy()
    assert views[2].leg_rows(oids[:1])[0].best_bid == o2[0] and o2[0] == pytest.approx(o2[1] * 1.1)  # not vacuous
    for leg in e.legs:
        r = w.opp[(w.opp.date == d2) & (w.opp.optionid == leg.oid)].iloc[0]          # prepended row (first stored)
        assert leg.mark == pytest.approx((r.best_bid + r.best_offer) / 2, rel=1e-12)
    assert mkt.fixes["dev"] == {"opprcd_dup_rows": 2 * len(oids), "secprd_dup_rows": 2, "vsurfd_dup_rows": 2,
                                "zerocd_nan_rows": 2, "zerocd_dup_rows": 2}
    assert mkt.fixes["holdout"] == dict.fromkeys(market.FIX_KEYS, 0)


def test_8e_moved_exdate_uses_the_row_chain_and_the_entry_settlement():
    """A13-4 8e: a held leg whose row (found by optionid) shows another exdate builds its forward chain (8b) from the
    row's exdate; settlement and last trading day stay those computed at entry; such leg-days are counted."""
    w = world(start=20110103, end=20110331, seed=9)
    d0, dd, sat, new = 20110103, 20110110, 20110219, 20110211
    _, e0, _ = _manual(copy.deepcopy(w), [d0], {d0})
    call = next(l for l in e0.legs if l.cp == 1)
    assert call.exdate == sat and call.settle == 20110218 and call.ltd == e0.legs[0].ltd
    w.move_exdate(sat, new, dd)
    w.opp.loc[(w.opp.date == dd) & (w.opp.optionid == call.oid), "best_bid"] = 0.0      # call falls back on dd
    mkt, e, views = _manual(w, synth13.weekdays(d0, dd), {d0})
    v = views[-1]
    assert e.ledger.rows[-1]["exdate_moved"] == 2 and e.ledger.rows[-2]["exdate_moved"] == 0
    T0 = (ch.ymd(call.settle).toordinal() - ch.ymd(dd).toordinal()) / 365.0
    assert v.T1(new) != T0                                                               # the rule is not vacuous
    rows = w.opp[(w.opp.date == dd) & (w.opp.exdate == new)].assign(K=lambda x: x.strike_price / 1000.0)
    F, src = ch.forward(ch.dedupe(rows, 0.25), v.rate(T0), T0, v.S)
    assert src == "parity" and v.forward(new, call.settle) == (F, src, v.rate(T0), T0)
    leg = next(l for l in e.legs if l.oid == call.oid)
    assert leg.mark == pytest.approx(float(ch.bs(F, call.K, T0, call.last_iv, v.rate(T0), v.S, 1)[0]), rel=1e-12)
    assert leg.exdate == sat and leg.moved


def test_leg_tracked_by_optionid_not_by_contract_terms():
    w = world(start=20110103, end=20110228, seed=9)
    d0, d1 = 20110103, 20110104
    _, e0, _ = _manual(copy.deepcopy(w), [d0], {d0})
    call = next(l for l in e0.legs if l.cp == 1)
    dup = w.opp[(w.opp.optionid == call.oid) & w.opp.date.isin([d0, d1])].copy()
    dup["optionid"] = 99_999_999
    dup.loc[dup.date == d0, "best_bid"] = 0.0                           # duplicate invalid on d0: entry keeps oid
    w.add_rows(dup)
    w.opp.loc[(w.opp.date == d1) & (w.opp.optionid == call.oid), "best_bid"] = 0.0   # tracked contract invalid on d1
    _, e, _ = _manual(w, [d0, d1], {d0})
    assert {l.oid for l in e.legs} == {l.oid for l in e0.legs}
    assert e.ledger.rows[1]["q_fallback"] == pytest.approx(call.q)


# ---------------------------------------------------------------- holdout rehearsal (A13-3 tests)
@pytest.fixture
def rehearsal(tmp_r13, monkeypatch):
    w = world(start=20120103, end=20121231, hold_end=20130329, seed=41)
    for k, v in dict(DEV_START=20120103, DEV_END=20121231, HOLD_START=20130102, HOLD_END=20130329,
                     F2_SPLIT=20120629, ENTRY_ON_LAST_DAY=False).items():
        monkeypatch.setattr(C, k, v)
    rd, ext = os.path.join(tmp_r13, "results"), os.path.join(tmp_r13, "ext")
    os.makedirs(rd)
    run_dev.main(["--go"], source=copy.deepcopy(w), results_dir=rd, ext_dir=ext)
    sel = json.load(open(os.path.join(rd, "selected_config.json")))
    sel["candidate"] = {"setting": "O0", "sharpe": 0.0, "max_drawdown": 0.05, "calmar": 0.0, "d_dev": 0.0}
    json.dump(sel, open(os.path.join(rd, "selected_config.json"), "w"))
    open(os.path.join(rd, "HOLDOUT_UNLOCKED"), "w").close()
    kw = dict(results_dir=rd, has_tag_fn=lambda: True, is_clean_fn=lambda: True, data_fn=lambda: [], aux=())
    return w, rd, kw


def _whitelisted(text):
    import re
    return all(any(re.match(p, line) for p in run_holdout.WHITELIST) for line in text.strip().splitlines() if line)


def test_rehearsal_a_out_of_range_iv_does_not_stop(rehearsal, capsys):
    w, rd, kw = rehearsal
    k = w.days.index(20130115)
    w.iv[k:k + 3] = 5.0
    v = run_holdout.main(["--go"], source_factory=lambda: copy.deepcopy(w), **kw)
    assert v and v["setting"] == "O0"
    s = json.load(open(os.path.join(rd, "holdout_summary.json")))
    assert s["iv_out_of_range_days"] > 0 and "rv21_filled_signal_days" in s
    daily = pd.read_csv(os.path.join(rd, "holdout_daily.csv"))
    assert daily.date.iloc[-1] == C.HOLD_END
    assert _whitelisted(capsys.readouterr().out)


def test_rehearsal_d_holdout_duplicates_and_nan_rates_do_not_stop(rehearsal):
    """A13-4 8d with E13-18: duplicate keys and NaN zero-curve rows in holdout files do not stop the run; counts only
    go to holdout_summary.json (and the development counts to dev_checks.json)."""
    w, rd, kw = rehearsal
    h = 20130115
    n = int((w.opp.date == h).sum())
    w.add_rows(w.opp[w.opp.date == h].assign(best_bid=lambda x: x.best_bid * 1.1), first=True)
    w.extra["secprd"].append(({"date": h, "close": 1.0}, False))
    w.extra["vsurfd"].append(({"date": h, "days": 30, "delta": 50, "cp_flag": 1, "impl_volatility": 0.5}, False))
    w.extra["zerocd"] += [({"date": h, "days": 180.0, "rate": np.nan}, False), ({"date": h, "days": 30.0, "rate": 9.0}, False)]
    v = run_holdout.main(["--go"], source_factory=lambda: copy.deepcopy(w), **kw)
    assert v and v["setting"] == "O0"
    s = json.load(open(os.path.join(rd, "holdout_summary.json")))
    assert s["data_fixes"]["SPX"] == {"opprcd_dup_rows": n, "secprd_dup_rows": 1, "vsurfd_dup_rows": 1,
                                      "zerocd_nan_rows": 1, "zerocd_dup_rows": 1}
    assert set(s["exdate_moved_leg_days"]) == {x.id for x in C.SETTINGS}
    c = json.load(open(os.path.join(rd, "dev_checks.json")))
    assert c["data_fixes"] == dict.fromkeys(market.FIX_KEYS, 0)
    assert set(c["exdate_moved_leg_days"]) == {x.id for x in C.SETTINGS}


def test_n2_no_entry_on_the_data_end_day():
    assert C.ENTRY_ON_LAST_DAY is False and C.RV21_FILL_MISSING_CLOSE is True
    w = world(start=20120103, end=20121231, hold_end=20130329, seed=35)
    mkt = market.Market(w, "SPX", ("dev",))
    eng = ladder.run(mkt, C.SETTINGS, 20120103, 20121231)
    last_dev = {sid: e.ledger.rows[-1]["opened"] for sid, e in eng.items()}
    assert last_dev["O0"] == 1                                       # 2012-12-31 is not a data end: opens as usual
    w.enable_holdout()
    mkt.add_split("holdout")
    eng = ladder.run(mkt, C.SETTINGS, 20130102, 20130329, forced_last_day=True,
                     entry_on_last_day=C.ENTRY_ON_LAST_DAY, engines=eng)
    for e in eng.values():
        assert e.ledger.rows[-1]["opened"] == 0 and not e.legs


def test_rehearsal_b_invariant_stop_leaks_nothing(rehearsal, capsys, monkeypatch):
    w, rd, kw = rehearsal
    real = ladder.Engine.step

    def boom(self, v, dec, **k):
        if v.d >= 20130201:
            raise A.InvariantError("injected invariant violation")
        return real(self, v, dec, **k)
    monkeypatch.setattr(ladder.Engine, "step", boom)
    with pytest.raises(A.InvariantError):
        run_holdout.main(["--go"], source_factory=lambda: copy.deepcopy(w), **kw)
    assert not any(os.path.exists(os.path.join(rd, f)) for f in run_holdout.RESULT_FILES)
    out = capsys.readouterr().out
    assert _whitelisted(out) and "P_" not in out
    log = open(os.path.join(rd, "holdout_run.log")).read()
    assert _whitelisted(log)


def test_rehearsal_c_second_run_refused(rehearsal):
    w, rd, kw = rehearsal
    assert run_holdout.main(["--go"], source_factory=lambda: copy.deepcopy(w), **kw)
    assert run_holdout.main(["--go"], source_factory=lambda: copy.deepcopy(w), **kw) is None
    assert "refused: 1: holdout results exist" in open(os.path.join(rd, "holdout_run.log")).read()


def test_e13_02_dev_iv_out_of_range_stops(tmp_r13):
    w = world(start=20120103, end=20121231, seed=42)
    w.iv[10] = 5.0
    with pytest.raises(SystemExit, match="E13-02"):
        run_dev.main(["--go"], source=w, results_dir=os.path.join(tmp_r13, "r"), ext_dir=os.path.join(tmp_r13, "x"))


def test_run_dev_refuses_second_run(tmp_r13):
    rd = os.path.join(tmp_r13, "r2")
    os.makedirs(rd)
    open(os.path.join(rd, "dev_summary.json"), "w").close()
    with pytest.raises(SystemExit, match="one development run"):
        run_dev.main(["--go"], source=None, results_dir=rd)


# ---------------------------------------------------------------- held for Kyle: N2, N3
def test_n3_both_behaviours_and_refusal(monkeypatch, tmp_r13):
    w = world(start=20110103, end=20110331, seed=8)
    w.missing_close.add(20110215)
    mkt = market.Market(w, "SPX", ("dev",))
    for d in [x for x in w.days if x <= 20110301]:
        mkt.day(d)
    t = 20110301
    monkeypatch.setattr(C, "RV21_FILL_MISSING_CLOSE", False)
    assert math.isnan(mkt.rv21(t))
    monkeypatch.setattr(C, "RV21_FILL_MISSING_CLOSE", True)
    c = np.array([mkt.close(d)[0] for d in mkt.days[mkt._di[t] - 21:mkt._di[t] + 1]])
    assert mkt.rv21(t) == pytest.approx(np.std(np.diff(np.log(c)), ddof=1) * math.sqrt(252))
    monkeypatch.setattr(C, "RV21_FILL_MISSING_CLOSE", None)
    with pytest.raises(RuntimeError, match="N3"):
        mkt.rv21(t)
    with pytest.raises(SystemExit, match="N3"):
        run_dev.main(["--go"], source=w, results_dir=os.path.join(tmp_r13, "n3"))


def test_n2_forced_run_refuses_while_undecided():
    w = world(start=20110103, end=20110131, seed=8)
    mkt = market.Market(w, "SPX", ("dev",))
    with pytest.raises(RuntimeError, match="N2"):
        ladder.run(mkt, [C.setting("O0")], 20110103, 20110131, forced_last_day=True, entry_on_last_day=None)
