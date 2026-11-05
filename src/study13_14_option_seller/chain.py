#!/usr/bin/env python3
"""Research 13 chain rules, shared by the G0 audit and the engine (4b). SPEC spec13-v1 (3d7dd482...) sections 3.4, 3.5
and 3.7 with SPEC_AMENDMENTS A13-1 (430fe471...) changes 1 and 4:
  Calendar      business days of one split; after its last day (or before its first) a weekday counts as a business
                day (3.4 calendar-end rule); settlement day and last trading day (3.5)
  standard_*    standard monthly expiries: (i) 1 = third Friday, 2 = the Saturday after it, 3 = the business day before
                a third Friday that is a holiday, priority 1 < 2 < 3 within a month on the same date; (ii) AM settled;
                (iii) expiry_indicator not in the non-standard list (missing and unseen codes are allowed)
  dedupe        one row per (date, exdate, strike, cp) of the merged roots: V(tau) and IV valid first, then larger
                open_interest, then smaller optionid (3.4)
  valid_quote   V: bid > 0, offer > bid, (offer - bid) / mid <= tau; iv_valid: impl_volatility present and > 0 (3.7)
  Rates         A13-1 change 4: curve of the latest calendar date <= d with a zerocd curve (zerocd dates outside the
                calendar unused), linear in days, flat outside the curve, none before -> error
  forward       A13-1 change 1: put-call parity on the tau 0.25 deduped chain at the strike with the smallest
                |C_mid - P_mid| (tie: lower strike), else S*exp(rT); forward_price is never used
  bs            Black-Scholes price and spot delta from the forward (G0 audit interpretation 2)
Strikes are in index points here (K = strike_price / 1000 after U1); rates are decimals (rate / 100 after U11)."""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
from scipy.special import ndtr

import omdata as om

TAU_S, TAU_G = 0.25, 0.50


def ymd(d):
    d = int(d)
    return dt.date(d // 10000, d // 100 % 100, d % 100)


def to_int(x):
    return x.year * 10000 + x.month * 100 + x.day


def dte(date, exdate):
    """Calendar days exdate - date (vectorised over YYYYMMDD ints)."""
    a = pd.to_datetime(pd.Series(np.asarray(date)).astype(str), format="%Y%m%d")
    b = pd.to_datetime(pd.Series(np.asarray(exdate)).astype(str), format="%Y%m%d")
    return (b - a).dt.days.to_numpy()


class Calendar:
    def __init__(self, days):
        self.days = np.asarray(sorted(int(x) for x in days), dtype=np.int64)
        self._set = set(self.days.tolist())
        self.first, self.last = int(self.days[0]), int(self.days[-1])

    def is_bday(self, d):
        d = int(d)
        if self.first <= d <= self.last:
            return d in self._set
        return ymd(d).weekday() < 5

    def prev_bday(self, d):
        x = ymd(d) - dt.timedelta(days=1)
        while not self.is_bday(to_int(x)):
            x -= dt.timedelta(days=1)
        return to_int(x)

    def settlement(self, exdate):
        return int(exdate) if self.is_bday(exdate) else self.prev_bday(exdate)

    def last_trading(self, exdate):
        return self.prev_bday(self.settlement(exdate))


def load_calendar(split, base=None, results_dir=None, read=None):
    """The split's calendar_<split>.csv, checked against SPLIT.json; holdout only with the unlock file."""
    om.check_unlock(split, results_dir)
    root = om.split_root(split, base)
    sp_path = os.path.join(root, "SPLIT.json")
    sp = json.load(open(sp_path))
    path = os.path.join(root, sp["calendar_file"])
    if read is not None:
        read += [sp_path, path]
    if om.sha256(path) != sp["calendar_sha256"]:
        raise om.Refused(f"{sp['calendar_file']}: sha256 differs from SPLIT.json")
    return Calendar(pd.read_csv(path).date.to_numpy())


def third_friday(y, m):
    first = dt.date(y, m, 1)
    return dt.date(y, m, 1 + (4 - first.weekday()) % 7 + 14)


def standard_kind(exdate, cal):
    e = ymd(exdate)
    tf = third_friday(e.year, e.month)
    if e == tf:
        return 1
    if e == tf + dt.timedelta(days=1):
        return 2
    if not cal.is_bday(to_int(tf)) and int(exdate) == cal.prev_bday(to_int(tf)):
        return 3
    return 0


def standard_mask(df, cal, nonstd=()):
    """Boolean mask of rows in a standard monthly (3.4 (i)-(iii)). df: date, exdate, am_settlement, expiry_indicator
    (str or None). Within a (date, month) the highest-priority kind present among qualifying rows is kept."""
    kinds = {e: standard_kind(e, cal) for e in np.unique(df.exdate)}
    k = df.exdate.map(kinds).to_numpy()
    ei = df.expiry_indicator.astype(object)
    ok = (k > 0) & (df.am_settlement.to_numpy() == 1) & ~ei.isin(list(nonstd)).to_numpy()
    tmp = pd.DataFrame({"date": df.date.to_numpy(), "month": df.exdate.to_numpy() // 100, "k": np.where(ok, k, 99)})
    best = tmp.groupby(["date", "month"]).k.transform("min").to_numpy()
    return ok & (k == best)


def range_status(date, std_exdates, lo, hi):
    """A13-2 change 1 for one day and one DTE range: 'no_monthly' when the day's chain has no standard monthly
    (3.4 (i)-(iii)) at all, 'gap' (structural gap) when it has some but none with lo <= DTE <= hi, else 'in_range'."""
    ex = np.asarray(std_exdates)
    if len(ex) == 0:
        return "no_monthly"
    d = dte(np.full(len(ex), int(date)), ex)
    return "in_range" if ((d >= lo) & (d <= hi)).any() else "gap"


def defect_ii(status, openable, gate_open=True):
    """SPEC 9 defect (ii) as amended by A13-2 change 3, for a day of the selected setting: (a) no standard monthly or
    (b) a monthly in range but no enterable candidate. Structural gaps and gate or signal closures are not defects."""
    if not gate_open:
        return False
    return status == "no_monthly" or (status == "in_range" and not openable)


def valid_quote(bid, offer, tau):
    bid, offer = np.asarray(bid, float), np.asarray(offer, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (bid > 0) & (offer > bid) & ((offer - bid) / ((bid + offer) / 2) <= tau)


def iv_valid(iv):
    iv = np.asarray(iv, float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(iv) & (iv > 0)


def dedupe(df, tau):
    good = valid_quote(df.best_bid, df.best_offer, tau) & iv_valid(df.impl_volatility)
    d = df.assign(_bad=~good, _noi=-df.open_interest.astype(float).fillna(-np.inf))
    d = d.sort_values(["date", "exdate", "K", "cp_flag", "_bad", "_noi", "optionid"], kind="mergesort")
    return d.drop_duplicates(["date", "exdate", "K", "cp_flag"], keep="first").drop(columns=["_bad", "_noi"])


class Rates:
    def __init__(self, z, cal, scale=0.01):
        d = pd.DataFrame({"date": np.asarray(z["date"]), "days": np.asarray(z["days"], float),
                          "rate": np.asarray(z["rate"], float)})
        d = d[d.date.isin(cal.days)].sort_values(["date", "days"])
        self.curves = {int(k): (g.days.to_numpy(), g.rate.to_numpy() * scale) for k, g in d.groupby("date")}
        self.dates = np.array(sorted(self.curves), dtype=np.int64)

    def curve_date(self, d):
        i = int(np.searchsorted(self.dates, int(d), side="right")) - 1
        if i < 0:
            raise RuntimeError(f"no zero curve on or before {d}")
        return int(self.dates[i])

    def rate(self, d, days):
        x, y = self.curves[self.curve_date(d)]
        return np.interp(np.asarray(days, float), x, y)


def parity_points(chain):
    """chain: one (date, exdate) deduped with tau 0.25 (columns K, cp_flag, best_bid, best_offer). Strikes where call
    and put are both V(0.25), ordered by |C_mid - P_mid| then strike: DataFrame K, diff (= C_mid - P_mid)."""
    v = valid_quote(chain.best_bid, chain.best_offer, TAU_S)
    q = chain[v].assign(mid=lambda x: (x.best_bid + x.best_offer) / 2)
    c = q[q.cp_flag == 1].set_index("K").mid
    p = q[q.cp_flag == -1].set_index("K").mid
    ks = c.index.intersection(p.index)
    out = pd.DataFrame({"K": ks.to_numpy(float), "diff": (c.loc[ks] - p.loc[ks]).to_numpy(float)})
    return out.assign(a=out["diff"].abs()).sort_values(["a", "K"], kind="mergesort").drop(columns="a").reset_index(drop=True)


def forward(chain, r, T, S):
    """A13-1 change 1. Returns (F, source) with source 'parity' or 'spot'."""
    pts = parity_points(chain)
    if len(pts):
        return float(pts.K.iloc[0] + np.exp(r * T) * pts["diff"].iloc[0]), "parity"
    return float(S * np.exp(r * T)), "spot"


def a3_prime(chain, r, T):
    """A13-1 change 2 for one day: (max - min) / median of parity forwards at up to 3 strikes, or None if < 2."""
    pts = parity_points(chain).head(3)
    if len(pts) < 2:
        return None
    f = pts.K.to_numpy() + np.exp(r * T) * pts["diff"].to_numpy()
    return float((f.max() - f.min()) / np.median(f))


def a3_prime_summary(days):
    """days: iterable of (date, chain or None when the day has no standard monthly in the H=21 range, r, T). Returns
    ({date: value}, days without a monthly, days with fewer than 2 usable strikes)."""
    vals, no_monthly, lt2 = {}, 0, 0
    for d, chain, r, T in days:
        if chain is None:
            no_monthly += 1
            continue
        v = a3_prime(chain, r, T)
        if v is None:
            lt2 += 1
        else:
            vals[d] = v
    return vals, no_monthly, lt2


def bs(F, K, T, sigma, r, S, cp):
    """Price and spot delta; cp = +1 call, -1 put (arrays broadcast)."""
    F, K, T, sigma, r, S, cp = (np.asarray(x, float) for x in (F, K, T, sigma, r, S, cp))
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = sigma * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / sd
        d2 = d1 - sd
        disc = np.exp(-r * T)
        price = np.where(cp > 0, disc * (F * ndtr(d1) - K * ndtr(d2)), disc * (K * ndtr(-d2) - F * ndtr(-d1)))
        delta = (F / S) * disc * np.where(cp > 0, ndtr(d1), ndtr(d1) - 1)
    return price, delta
