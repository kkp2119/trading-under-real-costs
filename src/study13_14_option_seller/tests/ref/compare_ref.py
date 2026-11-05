"""M13 comparison harness (memo 4b v2 10; A13-3 E13-15). Engine (ladder.run, window mode: empty ladder at the window
start) against tests/ref/refsim13.simulate on the same source. Output per window and setting: days passing, days
failing, the largest relative error, and for a failure only the first failing date. No P&L level, sum or statistic
is printed or written. A day passes when |a - b| <= 1e-6 max(|a|, |b|) + 1e-6 dollars, in each of the 3 scenarios."""
import copy
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(os.path.dirname(HERE)), os.path.dirname(HERE), HERE]
import config13 as C  # noqa: E402
import ladder  # noqa: E402
import market  # noqa: E402
import refsim13  # noqa: E402

SCEN = ("base", "stress", "opt")


def choose_windows(days, n=5, months=3, seed=13):
    """n non-overlapping windows of `months` calendar months inside the given business days, start months drawn with
    numpy default_rng(seed) (E13-15)."""
    ms = sorted({d // 100 for d in days})
    starts = [i for i in range(len(ms) - months + 1)]
    order = np.random.default_rng(seed).permutation(starts)
    taken, out = set(), []
    for i in order:
        span = set(range(i, i + months))
        if span & taken:
            continue
        taken |= span
        mm = ms[i:i + months]
        out.append((min(d for d in days if d // 100 == mm[0]), max(d for d in days if d // 100 == mm[-1])))
        if len(out) == n:
            break
    return sorted(out)


def compare(source_factory, setting_ids, windows):
    rows = []
    for a, b in windows:
        mkt = market.Market(source_factory(), "SPX", ("dev",))
        eng = ladder.run(mkt, [C.setting(s) for s in setting_ids], a, b)
        ref = refsim13.simulate(source_factory(), list(setting_ids), a, b)
        for sid in setting_ids:
            e = eng[sid].ledger.frame()
            r = ref[sid]
            ok_dates = list(e.date) == list(r.date)
            fail, worst, first = 0, 0.0, None
            for i in range(len(e)):
                bad = False
                for sc in SCEN:
                    x, y = float(e[f"P_{sc}"].iloc[i]), float(r[f"P_{sc}"].iloc[i])
                    rel = abs(x - y) / max(abs(x), abs(y), 1e-12)
                    worst = max(worst, rel if abs(x - y) > 1e-6 else 0.0)
                    bad |= abs(x - y) > 1e-6 * max(abs(x), abs(y)) + 1e-6
                if bad:
                    fail += 1
                    first = first or int(e.date.iloc[i])
            rows.append({"window": f"{a}-{b}", "setting": sid, "dates_equal": ok_dates, "pass_days": len(e) - fail,
                         "fail_days": fail, "max_rel_err": worst, "first_fail": first})
    return rows


WORLDS = ("clean", "injected", "missing_close", "rule5", "dup_nan", "exdate_change")
HOL = (20110318, 20110704, 20111124, 20120406)


def build_world(name):
    """Synthetic worlds of the M13 comparison. injected: whole-chain invalid days, missing deltas and IVs, a day
    without options, holidays. missing_close: injected + missing closes (A13-4 N3). rule5: injected + every Saturday
    monthly gets a third-Friday twin 30 days before expiry, and on every 37th day the quotes within 3% of the index
    are invalid, so held legs on a Saturday expiry that is no longer standard fall back with a parity forward
    (A13-4 rule 5). dup_nan: injected + duplicate keys stored after or before the original row (opprcd, secprd,
    vsurfd, zerocd) and zerocd rows with NaN days or rate (A13-4 8d). exdate_change: injected + from 30 days before
    expiry every Saturday monthly's contracts show exdate - 1 (odd months) or exdate - 8 (even months), with the same
    37th-day invalid quotes as rule5 (A13-4 8e)."""
    import datetime as dt
    import chain as ch
    import synth13
    kw = dict(start=20110103, end=20121231, seed=51) if name == "clean" else \
        dict(start=20110103, end=20121231, seed=52, holidays=HOL)
    base = synth13.World(**kw)
    if name != "clean":
        for d in (20110322, 20110323, 20110615, 20111017, 20120214):
            base.invalidate(d)
        base.opp.loc[base.opp.date.isin([20110405, 20110906]), "delta"] = np.nan
        m = base.opp.date.isin([20110406, 20120507])
        base.opp.loc[m & (base.opp.cp_flag == 1), ["delta", "impl_volatility"]] = np.nan
        base.drop_day(20110801)
    if name == "missing_close":
        base.missing_close |= {20110411, 20110412, 20111110, 20120410}
    if name == "rule5":
        for e in sorted(set(base.opp.exdate)):
            base.add_friday_monthly(int(e), ch.to_int(ch.ymd(int(e)) - dt.timedelta(days=30)))
    if name == "exdate_change":
        for e in sorted(set(base.opp.exdate)):
            new = ch.to_int(ch.ymd(int(e)) - dt.timedelta(days=1 if (e // 100) % 2 else 8))
            base.move_exdate(int(e), new, ch.to_int(ch.ymd(int(e)) - dt.timedelta(days=30)))
    if name in ("rule5", "exdate_change"):
        S = dict(zip(base.days, base.S))
        for d in base.days[5::37]:
            m = (base.opp.date == d) & ((base.opp.strike_price / 1000.0 / S[d] - 1).abs() < 0.03)
            base.opp.loc[m, "best_bid"] = 0.0
    if name == "dup_nan":
        for i, d in enumerate(base.days[3::29]):
            dup = base.opp[base.opp.date == d].copy()
            dup[["best_bid", "best_offer"]] *= 1.02
            dup["open_interest"] += 1.0
            base.add_rows(dup, first=bool(i % 2))
        for i, d in enumerate(base.days[7::31]):
            first = bool(i % 2)
            base.extra["secprd"].append(({"date": d, "close": base.closes[d] * (0.99 if first else 1.01)}, first))
            k = base.days.index(d)
            for n, iv in ((30, base.iv[k]), (91, base.iv91[k])):
                base.extra["vsurfd"].append(({"date": d, "days": n, "delta": 50, "cp_flag": 1,
                                              "impl_volatility": iv + (0.03 if first else 0.05)}, first))
        for i, d in enumerate(base.days[4::13]):
            first = bool(i % 2)
            base.extra["zerocd"] += [({"date": d, "days": 180.0, "rate": np.nan}, False),
                                     ({"date": d, "days": np.nan, "rate": 3.0}, False),
                                     ({"date": d, "days": 30.0, "rate": 3.5 if first else 9.0}, first)]
    return base


def activity(base, windows):
    """Non-vacuity counts over the compared windows: engine forward calls on an expiry that is not that day's
    standard monthly, by source (rule 5); held leg-days whose row shows another exdate (8e); rows fixed by 8d."""
    n = {"nonstd_fwd_parity": 0, "nonstd_fwd_spot": 0, "exdate_moved_leg_days": 0}
    orig = market.DayView.forward

    def fwd(self, exdate, settle=None):
        out = orig(self, exdate, settle)
        if exdate not in set(self.std_exdates.tolist()):
            n["nonstd_fwd_" + out[1]] += 1
        return out
    market.DayView.forward = fwd
    try:
        for a, b in windows:
            mkt = market.Market(copy.deepcopy(base), "SPX", ("dev",))
            for e in ladder.run(mkt, C.SETTINGS, a, b).values():
                n["exdate_moved_leg_days"] += int(sum(r["exdate_moved"] for r in e.ledger.rows))
            for k, x in mkt.fixes["dev"].items():
                n[k] = n.get(k, 0) + x
    finally:
        market.DayView.forward = orig
    return n


if __name__ == "__main__":
    import pandas as pd
    for name in (sys.argv[1:] or WORLDS):
        base = build_world(name)
        wins = choose_windows(base.days)
        res = pd.DataFrame(compare(lambda: copy.deepcopy(base), [s.id for s in C.SETTINGS], wins))
        print(f"== {name}: windows {wins}")
        g = res.groupby("setting").agg(pass_days=("pass_days", "sum"), fail_days=("fail_days", "sum"),
                                       max_rel_err=("max_rel_err", "max"), dates_equal=("dates_equal", "all"))
        print(g.to_string())
        f = res[res.fail_days > 0][["window", "setting", "fail_days", "first_fail"]]
        if len(f):
            print(f.to_string(index=False))
        if name in ("rule5", "dup_nan", "exdate_change"):
            print(f"{name} world activity:", activity(base, wins))
