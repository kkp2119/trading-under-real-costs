"""chain.py tests: M16 (standard monthly identification and chain building), M20 (a)-(d) (A13-1), M2 (Black-Scholes
known values and put-call parity). Synthetic inputs only."""
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

import chain as ch


def weekdays(a, b, holidays=()):
    d, out = ch.ymd(a), []
    while ch.to_int(d) <= b:
        if d.weekday() < 5 and ch.to_int(d) not in holidays:
            out.append(ch.to_int(d))
        d += dt.timedelta(days=1)
    return out


def rows(date, exdate, K, cp="CP", am=1, ei=None, root="SPX", bid=10.0, offer=10.5, iv=0.2, oi=100.0, oid=None):
    out = []
    for c in cp:
        out.append({"date": date, "exdate": exdate, "K": float(K), "cp_flag": 1 if c == "C" else -1, "best_bid": bid,
                    "best_offer": offer, "impl_volatility": iv, "delta": 0.5 if c == "C" else -0.5, "open_interest": oi,
                    "optionid": oid if oid is not None else hash((date, exdate, K, c, root)) % 10**9,
                    "am_settlement": am, "expiry_indicator": ei, "root": root})
    return out


# ---------- M16 ----------

CAL = ch.Calendar(weekdays(20080101, 20081231, {20080321}) + weekdays(20150101, 20151231) +
                  weekdays(20190101, 20191231, {20190419}))


def test_m16_kinds_and_settlement():
    assert ch.standard_kind(20190517, CAL) == 1                       # third Friday
    assert ch.standard_kind(20080419, CAL) == 2                       # Saturday after third Friday (Saturday era)
    assert ch.standard_kind(20080322, CAL) == 2                       # Saturday era, third Friday is a holiday: (2)
    assert ch.standard_kind(20190418, CAL) == 3                       # Friday era, holiday third Friday -> Thursday
    assert ch.standard_kind(20080425, CAL) == 0 and ch.standard_kind(20190416, CAL) == 0
    assert (CAL.settlement(20080322), CAL.last_trading(20080322)) == (20080320, 20080319)
    assert (CAL.settlement(20080419), CAL.last_trading(20080419)) == (20080418, 20080417)
    assert (CAL.settlement(20190517), CAL.last_trading(20190517)) == (20190517, 20190516)
    assert (CAL.settlement(20190418), CAL.last_trading(20190418)) == (20190418, 20190417)
    assert (CAL.settlement(20190419), CAL.last_trading(20190419)) == (20190418, 20190417)   # holiday Friday recorded


def test_m16_calendar_end_rule():
    dev = ch.Calendar(weekdays(20121101, 20121231, {20121225}))
    assert ch.standard_kind(20130119, dev) == 2 and ch.standard_kind(20130216, dev) == 2
    assert (dev.settlement(20130119), dev.last_trading(20130119)) == (20130118, 20130117)
    assert (dev.settlement(20130216), dev.last_trading(20130216)) == (20130215, 20130214)
    assert ch.standard_kind(20130117, dev) == 0                       # third Friday after the end is a weekday
    assert not dev.is_bday(20121225) and dev.is_bday(20130101)        # beyond the end: weekday rule only


def test_m16_standard_mask_rules():
    d = 20150210
    df = pd.DataFrame(rows(d, 20150220, 2000, am=1) + rows(d, 20150221, 2000, am=1)              # (1) and (2) in Feb
                      + rows(d, 20150320, 2000, am=0)                                              # PM third Friday
                      + rows(d, 20150320, 2050, am=1, ei="w") + rows(d, 20150320, 2100, am=1, ei="z")  # listed / new
                      + rows(d, 20150227, 2000, am=1)                                              # not (i)
                      + rows(d, 20150417, 2000, am=1, ei=None))
    m = ch.standard_mask(df, CAL, nonstd=("w",))
    kept = df[m]
    assert set(kept.exdate) == {20150220, 20150320, 20150417}          # (1) beats (2) in February
    assert set(kept[kept.exdate == 20150320].K) == {2100.0}            # PM and listed code out, unseen code allowed
    assert not kept.exdate.eq(20150221).any() and not kept.exdate.eq(20150227).any()


def test_m16_roots_merge_and_dedupe_order():
    d, e = 20150210, 20150320
    a = rows(d, e, 2000, root="SPX", oid=5) + rows(d, e, 2025, root="SPQ", oid=6)          # strikes split by root
    dup = [dict(rows(d, e, 2050, cp="C", root="SPX", bid=0.0, offer=1.0, oi=500.0, oid=10)[0]),   # invalid quote
           dict(rows(d, e, 2050, cp="C", root="SPQ", oi=50.0, oid=11)[0]),
           dict(rows(d, e, 2050, cp="C", root="SPZ", oi=80.0, oid=12)[0]),
           dict(rows(d, e, 2075, cp="C", root="SPX", oi=10.0, oid=20)[0]),
           dict(rows(d, e, 2075, cp="C", root="SPQ", oi=10.0, oid=19)[0])]
    df = pd.DataFrame(a + dup)
    out = ch.dedupe(df, ch.TAU_S)
    assert set(out.K) == {2000.0, 2025.0, 2050.0, 2075.0}
    assert out[out.K == 2050.0].optionid.tolist() == [12]              # valid first, then larger OI
    assert out[out.K == 2075.0].optionid.tolist() == [19]              # then smaller optionid
    assert out.duplicated(["date", "exdate", "K", "cp_flag"]).sum() == 0


def test_valid_quote_and_iv():
    assert ch.valid_quote([1.0, 0.0, 2.0, 1.0], [1.2, 1.0, 1.0, 2.0], 0.25).tolist() == [True, False, False, False]
    assert ch.valid_quote([1.0], [1.5], 0.5).tolist() == [True] and ch.iv_valid([0.2, 0.0, np.nan, -99.99]).tolist() == \
        [True, False, False, False]


# ---------- M19 additions (A13-2) ----------

def test_m19_range_status_and_structural_gap_is_not_a_defect():
    d = 20080303
    assert ch.range_status(d, [], 35, 65) == "no_monthly"
    assert ch.range_status(d, [20080322, 20080517], 35, 65) == "gap"          # DTE 19 and 75: none in 35..65
    assert ch.range_status(d, [20080322, 20080419], 35, 65) == "in_range"     # DTE 47
    assert ch.range_status(d, [20080322], 19, 49) == "in_range"               # H=10 range, DTE 19
    assert not ch.defect_ii("gap", False)                                     # structural gap: not a defect
    assert ch.defect_ii("in_range", False)                                    # monthly in range, no candidate
    assert not ch.defect_ii("in_range", True)
    assert ch.defect_ii("no_monthly", False)                                  # no standard monthly at all
    assert not ch.defect_ii("no_monthly", False, gate_open=False)             # gate or signal closed: not counted


# ---------- M20 ----------

def forward_v1_reference(chain, r, T, S):
    """spec13-v1 3.7 as written, implemented independently of chain.py: (1) median forward_price of the expiry,
    (2) parity at the strike with the smallest |C_mid - P_mid| among strikes whose call and put are both V(0.25)
    (tie: lower strike), (3) S*exp(rT)."""
    fp = [x for x in chain.forward_price.astype(float) if not math.isnan(x)]
    if fp:
        return float(np.median(fp)), "forward_price"
    best = None
    for k in sorted(set(chain.K)):
        legs = {}
        for _, x in chain[chain.K == k].iterrows():
            b, o = float(x.best_bid), float(x.best_offer)
            if b > 0 and o > b and (o - b) / ((b + o) / 2) <= 0.25:
                legs[int(x.cp_flag)] = (b + o) / 2
        if 1 in legs and -1 in legs:
            d = legs[1] - legs[-1]
            if best is None or abs(d) < abs(best[1]):
                best = (k, d)
    if best is not None:
        return float(best[0] + np.exp(r * T) * best[1]), "parity"
    return float(S * np.exp(r * T)), "spot"


def random_chain(rng, n=9, forward_price=False):
    ks = 1000.0 + 25 * np.arange(n)
    out = []
    for k in ks:
        for cp in (1, -1):
            mid = max(0.05, rng.normal(30, 20))
            sp = abs(rng.normal(0.05, 0.1)) * mid
            out.append({"K": k, "cp_flag": cp, "best_bid": mid - sp / 2, "best_offer": mid + sp / 2,
                        "forward_price": rng.normal(1100, 5) if forward_price else np.nan})
    return pd.DataFrame(out)


def test_m20a_identical_to_v1_rule_when_forward_price_is_null():
    rng = np.random.default_rng(13)
    for _ in range(300):
        c = random_chain(rng)
        r, T, S = rng.uniform(0, 0.06), rng.uniform(5, 65) / 365, rng.uniform(900, 1200)
        assert ch.forward(c, r, T, S) == forward_v1_reference(c, r, T, S)
    c = random_chain(rng).assign(best_bid=0.0)                          # no valid pair: both fall back to S e^{rT}
    assert ch.forward(c, 0.03, 0.1, 1000.0) == forward_v1_reference(c, 0.03, 0.1, 1000.0) == (float(1000.0 * np.exp(0.003)), "spot")


def test_m20b_independent_of_forward_price():
    rng = np.random.default_rng(7)
    for _ in range(100):
        c = random_chain(rng, forward_price=True)
        f1 = ch.forward(c, 0.02, 0.1, 1000.0)
        f2 = ch.forward(c.assign(forward_price=c.forward_price * 3), 0.02, 0.1, 1000.0)
        assert f1 == f2 and f1[1] in ("parity", "spot")


def test_m20c_rates_previous_curve_flat_and_none():
    cal = ch.Calendar([20200102, 20200103, 20200106, 20200107])
    z = {"date": [20200102] * 2 + [20200104] * 2 + [20200107] * 2,     # 20200104 is not a calendar day
         "days": [30, 365] * 3, "rate": [1.0, 2.0, 9.0, 9.0, 3.0, 4.0]}
    rt = ch.Rates(z, cal)
    assert rt.curve_date(20200106) == 20200102 and rt.curve_date(20200103) == 20200102 and rt.curve_date(20200107) == 20200107
    assert rt.rate(20200106, 30) == pytest.approx(0.01) and rt.rate(20200106, 197.5) == pytest.approx(0.015)
    assert rt.rate(20200107, 5) == pytest.approx(0.03) and rt.rate(20200107, 800) == pytest.approx(0.04)   # flat outside
    with pytest.raises(RuntimeError, match="no zero curve"):
        rt.rate(20200101, 30)


def test_m20d_a3_prime_exclusions_and_ties():
    one = pd.DataFrame([{"K": 1000.0, "cp_flag": cp, "best_bid": 10.0, "best_offer": 10.4} for cp in (1, -1)])
    tie = pd.DataFrame([{"K": k, "cp_flag": cp, "best_bid": b, "best_offer": b + 0.2}
                        for k, bc, bp in [(990.0, 12.0, 10.0), (1000.0, 10.0, 10.0), (1010.0, 8.0, 10.0), (1020.0, 6.0, 12.0)]
                        for cp, b in ((1, bc), (-1, bp))])
    pts = ch.parity_points(tie)
    assert pts.K.tolist()[:3] == [1000.0, 990.0, 1010.0]               # |diff| 0, then 2 at 990 and 1010: lower first
    vals, no_m, lt2 = ch.a3_prime_summary([(1, None, 0.0, 0.1), (2, one, 0.0, 0.1), (3, tie, 0.0, 0.1)])
    assert (no_m, lt2, list(vals)) == (1, 1, [3])
    f = np.array([1000.0, 992.0, 1008.0])
    assert vals[3] == pytest.approx((f.max() - f.min()) / np.median(f))


# ---------- M2 ----------

def classic_bs(S, K, T, r, q, s, cp):
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))                  # noqa: E731
    d1 = (math.log(S / K) + (r - q + s * s / 2) * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    if cp > 0:
        return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2), math.exp(-q * T) * N(d1)
    return K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1), math.exp(-q * T) * (N(d1) - 1)


def test_m2_known_values_and_parity():
    S, K, T, r, s = 100.0, 100.0, 1.0, 0.05, 0.2
    F = S * math.exp(r * T)
    (c, p), (dc, dp) = ch.bs([F, F], [K, K], T, s, r, S, [1, -1])
    assert c == pytest.approx(10.4506, abs=1e-4) and p == pytest.approx(5.5735, abs=1e-4)      # textbook values
    assert dc == pytest.approx(0.6368, abs=1e-4) and dp == pytest.approx(-0.3632, abs=1e-4)
    assert c - p == pytest.approx(math.exp(-r * T) * (F - K), abs=1e-12)
    rng = np.random.default_rng(2)
    for _ in range(200):
        S, K, T, r, q, s = rng.uniform(500, 2000), rng.uniform(400, 2500), rng.uniform(5, 400) / 365, rng.uniform(0, .06), \
            rng.uniform(0, .04), rng.uniform(.05, .8)
        F = S * math.exp((r - q) * T)
        (c, p), (dc, dp) = ch.bs([F, F], [K, K], T, s, r, S, [1, -1])
        for cp, pr, de in ((1, c, dc), (-1, p, dp)):
            rp, rd = classic_bs(S, K, T, r, q, s, cp)
            assert pr == pytest.approx(rp, rel=1e-9, abs=1e-9) and de == pytest.approx(rd, abs=1e-10)
        assert c - p == pytest.approx(math.exp(-r * T) * (F - K), rel=1e-9, abs=1e-9)
