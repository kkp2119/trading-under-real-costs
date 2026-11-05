"""Synthetic OptionMetrics-like world for engine tests (memo 4b 9.1). Same Source interface as market.RealSource:
calendar(split), zerocd(split), secprd(und, year), vsurfd(und, year), opprcd(und, year), enable_holdout(), access,
paths(). Holdout years (>= 2013) are refused until enable_holdout(). Prices are Black-Scholes from the forward
F = S exp((r - q) T1) with T1 = (settlement - date)/365; OptionMetrics delta = Black-Scholes spot delta."""
import datetime as dt
import math

import numpy as np
import pandas as pd

import chain as ch

UND_ID = {"SPX": 108105, "NDX": 102480, "RUT": 102434}


class Locked(RuntimeError):
    pass


def _insert(arrs, extra, keys):
    """Stored-order injection (A13-4 8d worlds): each extra (row, first) goes just before (first) or just after the
    first stored row with the same keys; a row without a match goes to the end."""
    cols = {k: list(v) for k, v in arrs.items()}
    for row, first in extra:
        n = len(cols[keys[0]])
        m = next((i for i in range(n) if all(cols[k][i] == row[k] for k in keys)), None)
        pos = n if m is None else (m if first else m + 1)
        for k in cols:
            cols[k].insert(pos, row.get(k, np.nan))
    return {k: np.asarray(v) for k, v in cols.items()}


def weekdays(a, b, holidays=()):
    d, out = ch.ymd(a), []
    while ch.to_int(d) <= b:
        if d.weekday() < 5 and ch.to_int(d) not in holidays:
            out.append(ch.to_int(d))
        d += dt.timedelta(days=1)
    return out


def monthly_exdates(y0, y1, era="sat", holidays=()):
    out = []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            tf = ch.third_friday(y, m)
            if era == "sat":
                out.append(ch.to_int(tf + dt.timedelta(days=1)))
            else:
                e = ch.to_int(tf)
                out.append(e if e not in holidays else ch.to_int(tf - dt.timedelta(days=1)))
    return out


class World:
    def __init__(self, start=20110103, end=20121231, hold_end=None, seed=1, S0=1000.0, sigma_real=0.18,
                 sigma_imp=0.20, mu=0.0, r=0.02, q=0.015, spread=0.04, min_tick=0.05, era="sat", holidays=(),
                 strike_step=10.0, iv_path=None, iv91_add=0.01, und="SPX", max_dte=130):
        self.und, self.r, self.q, self.era = und, r, q, era
        self.dev_days = weekdays(start, min(end, 20121231), holidays)
        self.hold_days = weekdays(20130102, hold_end, holidays) if hold_end else []
        days = self.dev_days + self.hold_days
        self.days = days
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(len(days))
        lr = (mu - 0.5 * sigma_real ** 2) / 252 + sigma_real / math.sqrt(252) * z
        lr[0] = 0.0
        self.S = S0 * np.exp(np.cumsum(lr))
        self.iv = np.asarray(iv_path(len(days)) if iv_path else np.full(len(days), sigma_imp), float)
        self.iv91 = self.iv + iv91_add
        cal = ch.Calendar(days)
        exds = monthly_exdates(start // 10000, (hold_end or end) // 10000 + 1, era, holidays)
        K = np.arange(S0 * 0.4 // strike_step * strike_step, S0 * 1.8, strike_step)
        oid_of, rows = {}, []
        for i, d in enumerate(days):
            S, sig = self.S[i], self.iv[i]
            for e in exds:
                dte = (ch.ymd(e) - ch.ymd(d)).days
                if dte < 0 or dte > max_dte:
                    continue
                T = (ch.ymd(cal.settlement(e)) - ch.ymd(d)).days / 365.0
                if T <= 0:
                    continue
                F = S * math.exp((r - q) * T)
                for cp in (1, -1):
                    px, de = ch.bs(F, K, T, sig, r, S, cp)
                    hs = np.maximum(spread * px, min_tick) / 2.0
                    bid = np.maximum(np.round(px - hs, 4), 0.0)
                    off = np.maximum(np.round(px + hs, 4), bid + 1e-4)          # offer > bid after rounding
                    for k, b, o, dl in zip(K, bid, off, de):
                        key = (e, float(k), cp)
                        if key not in oid_of:
                            oid_of[key] = 10_000_000 + len(oid_of)
                        rows.append((d, e, cp, float(k) * 1000.0, b, o, sig, dl, 100.0, oid_of[key], 1, -1))
        cols = ["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer", "impl_volatility", "delta",
                "open_interest", "optionid", "am_settlement", "expiry_indicator"]
        self.opp = pd.DataFrame(rows, columns=cols)
        self.oid_of = oid_of
        self.closes = pd.Series(self.S, index=days)
        self.missing_close = set()
        self.extra = {"secprd": [], "vsurfd": [], "zerocd": []}     # (row, first) injected in stored order
        self.access, self.read = [], []
        self.holdout_enabled = False

    # ---------------- injections
    def set_quote(self, date, exdate, K, cp, bid=None, offer=None, iv=None, delta=None, drop=False):
        m = (self.opp.date == date) & (self.opp.exdate == exdate) & (self.opp.strike_price == K * 1000.0) & \
            (self.opp.cp_flag == cp)
        if drop:
            self.opp = self.opp[~m].reset_index(drop=True)
            return
        for col, val in (("best_bid", bid), ("best_offer", offer), ("impl_volatility", iv), ("delta", delta)):
            if val is not None:
                self.opp.loc[m, col] = val

    def invalidate(self, date, exdate=None):
        m = self.opp.date == date
        if exdate is not None:
            m &= self.opp.exdate == exdate
        self.opp.loc[m, "best_bid"] = 0.0

    def drop_day(self, date):
        self.opp = self.opp[self.opp.date != date].reset_index(drop=True)

    def add_rows(self, df, first=False):
        """first: the rows come before equal-key rows in stored order (opprcd sorting is stable)."""
        self.opp = pd.concat([df, self.opp] if first else [self.opp, df], ignore_index=True)

    def move_exdate(self, exdate, new, from_date):
        """The same contracts (optionids) of exdate show exdate new from from_date on (A13-4 8e world). Rows that
        would have exdate < date are dropped."""
        m = (self.opp.exdate == exdate) & (self.opp.date >= from_date)
        self.opp.loc[m, "exdate"] = new
        self.opp = self.opp[self.opp.exdate >= self.opp.date].reset_index(drop=True)

    def add_friday_monthly(self, sat_exdate, from_date):
        """A Saturday (2) monthly gets a third-Friday (1) twin from from_date on (copied rows, new optionids), so the
        Saturday expiry stops being that month's standard monthly (A13-4 rule 5 world)."""
        df = self.opp[(self.opp.exdate == sat_exdate) & (self.opp.date >= from_date)].copy()
        df["exdate"] = ch.to_int(ch.ymd(sat_exdate) - dt.timedelta(days=1))
        df["optionid"] = df.optionid + 50_000_000
        self.add_rows(df)

    # ---------------- Source interface
    def enable_holdout(self):
        self.holdout_enabled = True

    def _chk(self, year):
        if year >= 2013 and not self.holdout_enabled:
            raise Locked(f"holdout year {year} before enable_holdout")

    def calendar(self, split):
        if split == "holdout" and not self.holdout_enabled:
            raise Locked("holdout calendar before enable_holdout")
        self.access.append(("calendar", None, split))
        return ch.Calendar(self.dev_days if split == "dev" else self.hold_days)

    def zerocd(self, split):
        if split == "holdout" and not self.holdout_enabled:
            raise Locked("holdout zerocd before enable_holdout")
        self.access.append(("zerocd", None, split))
        ds = self.dev_days if split == "dev" else self.hold_days
        z = {"date": np.repeat(np.array(ds, np.int64), 2), "days": np.tile([30.0, 365.0], len(ds)),
             "rate": np.full(2 * len(ds), self.r * 100.0)}
        return _insert(z, [x for x in self.extra["zerocd"] if x[0]["date"] in set(ds)], ("date", "days"))

    def secprd(self, und, year):
        self._chk(year)
        self.access.append(("secprd", und, year))
        ds = [d for d in self.days if d // 10000 == year]
        c = np.array([np.nan if d in self.missing_close else self.closes[d] for d in ds])
        a = {"date": np.array(ds, np.int64), "close": c, "return": np.r_[np.nan, c[1:] / c[:-1] - 1]}
        return _insert(a, [x for x in self.extra["secprd"] if x[0]["date"] // 10000 == year], ("date",))

    def vsurfd(self, und, year):
        self._chk(year)
        self.access.append(("vsurfd", und, year))
        idx = [i for i, d in enumerate(self.days) if d // 10000 == year]
        ds = np.array([self.days[i] for i in idx], np.int64)
        iv30, iv91 = self.iv[idx], self.iv91[idx]
        n = len(ds)
        a = {"date": np.r_[ds, ds], "days": np.r_[np.full(n, 30), np.full(n, 91)], "delta": np.full(2 * n, 50),
             "cp_flag": np.ones(2 * n, np.int8), "impl_volatility": np.r_[iv30, iv91],
             "impl_strike": np.r_[self.S[idx], self.S[idx]]}
        return _insert(a, [x for x in self.extra["vsurfd"] if x[0]["date"] // 10000 == year],
                       ("date", "days", "delta", "cp_flag"))

    def opprcd(self, und, year):
        self._chk(year)
        self.access.append(("opprcd", und, year))
        o = self.opp[self.opp.date // 10000 == year].sort_values(["date", "exdate", "cp_flag", "strike_price", "optionid"],
                                                                 kind="mergesort")
        a = {c: o[c].to_numpy() for c in o.columns if c != "expiry_indicator"}
        ei = o.expiry_indicator.to_numpy()
        codes = sorted({x for x in ei if isinstance(x, str)})
        a["expiry_indicator"] = np.array([codes.index(x) if isinstance(x, str) else -1 for x in ei], np.int16)
        a["expiry_indicator__codes"] = np.array(codes, dtype=str) if codes else np.zeros(0, dtype="<U1")
        return a

    def paths(self):
        return [f"synthetic:{t}:{u}:{y}" for t, u, y in self.access]
