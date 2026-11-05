#!/usr/bin/env python3
"""Research 13 market data for one run and one underlying (design memo 4b v2, 1.2, 2, 8.3; SPEC 2.3, 3.4, 3.7, 3.8;
A13-1, A13-3 E13-17).

A Source gives arrays in the 4a file layout: calendar(split), zerocd(split), secprd(und, year), vsurfd(und, year),
opprcd(und, year). RealSource reads $DATA_DIR/research13_data through omdata.Store; holdout files only after
enable_holdout(), which the holdout run calls after the E13-13 check. Every access is logged as (table, und, year).

File access (E13-17): processing day d loads opprcd of year(d) only, and secprd and vsurfd of year(d) and, when a
signal window reaches back, year(d) - 1. Nothing of a later year or of the next split is read ahead; a request for a
year after year(d) raises."""
import numpy as np
import pandas as pd

import chain as ch
import config13 as C
import omdata as om

OPP_COLS = ("date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer", "impl_volatility", "delta",
            "open_interest", "optionid", "am_settlement")


def split_of(year):
    return "dev" if year <= om.SPLITS["dev"][3] else "holdout"


class FileAccessError(RuntimeError):
    pass


class RealSource:
    def __init__(self, base=None, results_dir=None):
        self.base, self.results_dir = base, results_dir
        self.stores = {"dev": om.Store("dev", base=base)}
        self.access = []
        self.read_extra = []

    def enable_holdout(self):
        self.stores["holdout"] = om.Store("holdout", base=self.base, results_dir=self.results_dir)

    def _store(self, split):
        if split not in self.stores:
            raise FileAccessError(f"{split} store not enabled")
        return self.stores[split]

    def calendar(self, split):
        self._store(split)
        return ch.load_calendar(split, base=self.base, results_dir=self.results_dir, read=self.read_extra)

    def zerocd(self, split):
        self.access.append(("zerocd", None, split))
        return self._store(split).load(f"zerocd/zerocd_{split}.npz")

    def _yearly(self, table, und, year):
        self.access.append((table, und, year))
        return self._store(split_of(year)).load(f"{table}/{table}_{und}_{year}.npz")

    def secprd(self, und, year):
        return self._yearly("secprd", und, year)

    def vsurfd(self, und, year):
        return self._yearly("vsurfd", und, year)

    def opprcd(self, und, year):
        return self._yearly("opprcd", und, year)

    def paths(self):
        out = list(self.read_extra)
        for s in self.stores.values():
            out += s.read
        return out


def decode(a, col):
    codes = np.array(list(a[col + "__codes"]) + [None], dtype=object)
    c = np.asarray(a[col]).astype(np.int64)
    return codes[np.where(c < 0, len(codes) - 1, c)]


FIX_KEYS = ("opprcd_dup_rows", "secprd_dup_rows", "vsurfd_dup_rows", "zerocd_nan_rows", "zerocd_dup_rows")


def first_rows(df, keys):
    """A13-4 8d: of rows sharing keys keep the first in stored order. Returns (kept rows, number dropped)."""
    dup = df.duplicated(list(keys), keep="first").to_numpy()
    return df[~dup], int(dup.sum())


def clean_zerocd(z):
    """A13-4 8d: drop rows whose days or rate is NaN, then keep the first of rows sharing (date, days)."""
    d = pd.DataFrame({k: np.asarray(z[k]) for k in ("date", "days", "rate")})
    nan = (d.days.isna() | d.rate.isna()).to_numpy()
    d, ndup = first_rows(d[~nan], ("date", "days"))
    return {k: d[k].to_numpy() for k in ("date", "days", "rate")}, int(nan.sum()), ndup


class Market:
    """Day-by-day market view. splits: ("dev",) or ("dev", "holdout"). The calendar used for decisions on day d is the
    calendar of the splits active when d is processed (dev calendar for dev days; dev + holdout for holdout days)."""

    def __init__(self, source, und="SPX", splits=("dev",)):
        self.src, self.und = source, und
        self.cals = {}
        self.days = []
        self.rates = None
        self._z = {}
        self.splits = []
        self._opp_year, self._opp = None, None
        self._sec, self._vs = {}, {}
        self._cur_year = None
        self.iv_out_of_range = set()                     # E13-02: signal days with IV30/IV91 outside the range (counted)
        self.rv21_filled = set()                         # A13-4 N3: signal days whose RV21 window used a previous close
        self.fixes = {s: dict.fromkeys(FIX_KEYS, 0) for s in ("dev", "holdout")}   # A13-4 8d counts by split
        for s in splits:
            self.add_split(s)

    # ---------------------------------------------------------------- splits and calendars
    def add_split(self, split):
        if split in self.splits:
            return
        cal = self.src.calendar(split)
        self.splits.append(split)
        days = sorted(set(self.days) | set(int(x) for x in cal.days))
        self.days = days
        self._di = {x: i for i, x in enumerate(days)}
        self.cal = ch.Calendar(days)                      # active calendar (calendar-end rule beyond its last day)
        self.cals[split] = self.cal
        self._z[split], nnan, ndup = clean_zerocd(self.src.zerocd(split))
        self.fixes[split]["zerocd_nan_rows"] += nnan
        self.fixes[split]["zerocd_dup_rows"] += ndup
        z = {k: np.concatenate([np.asarray(self._z[s][k]) for s in self.splits]) for k in ("date", "days", "rate")}
        self.rates = ch.Rates(z, self.cal, scale=0.01)

    # ---------------------------------------------------------------- file access (E13-17)
    def _need(self, d):
        y = d // 10000
        if self._cur_year is not None and y < self._cur_year:
            raise FileAccessError("days must be processed in order")
        self._cur_year = y

    def _sec_year(self, y):
        if y > self._cur_year:
            raise FileAccessError(f"secprd {y} requested while processing {self._cur_year}")
        if y not in self._sec:
            if y < self._cur_year - 1:
                raise FileAccessError(f"secprd {y} is more than one year back")
            a = self.src.secprd(self.und, y)
            df, n = first_rows(pd.DataFrame({"date": np.asarray(a["date"], np.int64),
                                             "close": np.asarray(a["close"], float)}), ("date",))
            self.fixes[split_of(y)]["secprd_dup_rows"] += n
            self._sec[y] = pd.Series(df.close.to_numpy(), index=df.date.to_numpy())
        return self._sec[y]

    def _vs_year(self, y):
        if y > self._cur_year:
            raise FileAccessError(f"vsurfd {y} requested while processing {self._cur_year}")
        if y not in self._vs:
            a = self.src.vsurfd(self.und, y)
            df = pd.DataFrame({k: np.asarray(a[k]) for k in ("date", "days", "delta", "cp_flag", "impl_volatility")})
            df, n = first_rows(df, ("date", "days", "delta", "cp_flag"))
            self.fixes[split_of(y)]["vsurfd_dup_rows"] += n
            ok = (df.delta == 50) & (df.cp_flag == 1) & ch.iv_valid(df.impl_volatility)
            sub = df[ok & df.days.isin([30, 91])]
            self._vs[y] = {n: pd.Series(g.impl_volatility.to_numpy(float), index=g.date.to_numpy(np.int64))
                           for n, g in ((30, sub[sub.days == 30]), (91, sub[sub.days == 91]))}
        return self._vs[y]

    def _opp_load(self, y):
        if self._opp_year == y:
            return
        if y != self._cur_year:
            raise FileAccessError(f"opprcd {y} requested while processing {self._cur_year}")
        a = self.src.opprcd(self.und, y)
        df = pd.DataFrame({k: np.asarray(a[k]) for k in OPP_COLS})
        df["expiry_indicator"] = decode(a, "expiry_indicator")
        df, n = first_rows(df, ("date", "optionid"))                         # A13-4 8d: first row in stored order
        self.fixes[split_of(y)]["opprcd_dup_rows"] += n
        df = df[(df.am_settlement == 1) & (df.exdate >= df.date)]            # PM rows can never be held or used
        df = df[~df.expiry_indicator.isin(list(C.NONSTD_CODES))].copy()      # (ii) and (iii) only (A13-4 rule 5)
        df["K"] = df.strike_price / 1000.0
        df["is_std"] = ch.standard_mask(df, self.cal, C.NONSTD_CODES)        # (i) priority: entry candidates only
        df["dte"] = ch.dte(df.date, df.exdate)
        df = df.sort_values(["date", "exdate", "K", "cp_flag", "optionid"], kind="mergesort").reset_index(drop=True)
        self._opp, self._opp_year = df, y
        self._dates = df.date.to_numpy()

    # ---------------------------------------------------------------- per day
    def close(self, d):
        """Close of d; a calendar day without a close uses the previous close (3.4)."""
        y = d // 10000
        s = self._sec_year(y)
        if d in s.index and np.isfinite(s.loc[d]):
            return float(s.loc[d]), True
        i = self._di[d]
        for j in range(i - 1, -1, -1):
            dd = self.days[j]
            ss = self._sec_year(dd // 10000)
            if dd in ss.index and np.isfinite(ss.loc[dd]):
                return float(ss.loc[dd]), False
        return float("nan"), False

    def iv(self, t, n):
        """vsurfd IV n-day (30 or 91), delta 50 call on signal day t, NaN when missing."""
        s = self._vs_year(t // 10000)[n]
        v = float(s.loc[t]) if t in s.index else float("nan")
        if v == v and not (C.IV_RANGE[0] < v < C.IV_RANGE[1]):
            self.iv_out_of_range.add((t, n))
        return v

    def rv21(self, t):
        i = self._di[t]
        if i < C.RV_N:
            return float("nan")
        win = self.days[i - C.RV_N:i + 1]
        c = []
        for dd in win:
            s = self._sec_year(dd // 10000)
            c.append(float(s.loc[dd]) if dd in s.index else float("nan"))
        c = np.array(c)
        if not np.isfinite(c).all():
            if C.RV21_FILL_MISSING_CLOSE is None:
                raise RuntimeError("N3 undecided: missing close inside an RV21 window")
            if not C.RV21_FILL_MISSING_CLOSE:
                return float("nan")
            c = np.array([self.close(dd)[0] for dd in win])           # previous close (3.4), N3 = True
            self.rv21_filled.add(t)
            if not np.isfinite(c).all():
                return float("nan")
        return float(np.std(np.diff(np.log(c)), ddof=1) * np.sqrt(C.ANN))

    def day(self, d):
        """DayView for business day d (loads year(d) files; nothing later)."""
        self._need(d)
        self._sec_year(d // 10000)
        self._vs_year(d // 10000)
        self._opp_load(d // 10000)
        a, b = np.searchsorted(self._dates, d, "left"), np.searchsorted(self._dates, d, "right")
        return DayView(self, d, self._opp.iloc[a:b])


class DayView:
    """rows: the day's rows with (ii) AM settlement and (iii) no non-standard code. Held legs are looked up and forwards
    built from them (A13-4 rule 5); the standard monthly of 3.4 (i) (std, std_exdates) only chooses new entries."""

    def __init__(self, mkt, d, rows):
        self.m, self.d, self.rows = mkt, d, rows
        self.cal = mkt.cal
        self.S, self.has_close = mkt.close(d)
        self.std = rows[rows.is_std.to_numpy(bool)]
        self.has_options = len(self.std) > 0
        self._oid = pd.Index(rows.optionid.to_numpy()) if len(rows) else pd.Index([])
        self._fwd, self._sel = {}, {}
        self.std_exdates = np.unique(self.std.exdate.to_numpy()) if len(self.std) else np.array([], dtype=np.int64)
        self.curve_prev = mkt.rates.curve_date(d) != d

    # T1 and rates (switch table: T1; A13-1 change 4)
    def T1(self, exdate, settle=None):
        """(settlement - date)/365; settle given: a held leg's settlement computed at entry (A13-4 8e)."""
        s = self.cal.settlement(exdate) if settle is None else settle
        return (ch.ymd(s).toordinal() - ch.ymd(self.d).toordinal()) / 365.0

    def rate(self, T):
        return float(self.m.rates.rate(self.d, T * 365.0))

    def expiry_rows(self, exdate):
        """All (ii)(iii) rows of exdate today; for a standard exdate these are exactly its standard rows."""
        r = self.rows
        return r[r.exdate == exdate]

    def forward(self, exdate, settle=None):
        """A13-1 change 1 on the tau 0.25 deduped chain of expiry_rows(exdate) (A13-4 rule 5); T = T1(exdate, settle)."""
        key = (exdate, settle)
        if key not in self._fwd:
            T = self.T1(exdate, settle)
            r = self.rate(T)
            c = ch.dedupe(self.expiry_rows(exdate), C.TAU_FWD)
            self._fwd[key] = ch.forward(c, r, T, self.S) + (r, T)
        return self._fwd[key]

    def leg_rows(self, oids):
        """Rows of the given optionids today (None where absent)."""
        if not len(self._oid):
            return [None] * len(oids)
        idx = self._oid.get_indexer(list(oids))
        return [None if i < 0 else self.rows.iloc[i] for i in idx]
