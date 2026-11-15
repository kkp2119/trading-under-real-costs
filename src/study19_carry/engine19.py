"""Research 19 engine: G10 carry with macro filters checked monthly, weekly or daily (SPEC19, ANSWERS, ADDENDUM).

Priority ADDENDUM > ANSWERS > SPEC19; the details the three leave open are listed in DEVIATIONS19.md 1.

Calendar: H.10 business days (a day on which an H.10 rate was published). On the last business day of month t the
currencies are ranked by (own 3-month rate - US 3-month rate), month t averages: long the top 3, short the bottom 3,
1/3 dollar each (2 and 2 with fewer than 6 currencies, nothing with fewer than 4). The position is opened at the H.10
rates of the first business day of month t + 1 and held to the first business day of month t + 2 with the foreign
amounts bought on the entry day. Interest = differential x calendar days held / 365.
Filters: the condition is evaluated at the close of a check day d (M: last business day of the month, W: of the week,
D: every business day) and acted on at the H.10 rates of the next business day: on -> flat, off -> the month's
portfolio again. Cost: 4 bp of the gross notional (2 dollars per dollar of capital) at every entry and at every exit
inside a month; the scheduled exit at the next month's first business day is free.
Monthly return of holding month m = P&L of the intervals that start in m - costs of the trades made in m.
"""
import datetime as dt
import hashlib
import math
import os
import statistics
from fractions import Fraction

import numpy as np

DATA = os.path.join(os.environ.get("DATA_DIR", "data"), "research19_data")
CAND = os.path.join(DATA, "candidates")
HERE = os.path.dirname(os.path.abspath(__file__))

CCY = ("AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "NOK", "SEK")      # ranking ties go in this order (QUESTIONS B6)
FX = {"AUD": ("DEXUSAL", True), "CAD": ("DEXCAUS", False), "CHF": ("DEXSZUS", False), "EUR": ("DEXUSEU", True),
      "GBP": ("DEXUSUK", True), "JPY": ("DEXJPUS", False), "NZD": ("DEXUSNZ", True), "NOK": ("DEXNOUS", False),
      "SEK": ("DEXSDUS", False)}                                           # (FRED series, quoted as USD per unit?)
RATE = {"AUD": "AU", "CAD": "CA", "CHF": "CH", "EUR": "EZ", "GBP": "GB", "JPY": "JP", "NZD": "NZ", "NOK": "NO", "SEK": "SE",
        "USD": "US"}
DEM_PATH = "h10"                    # ANSWERS A1, decided by a1_19.py (results/a1_check19.json): "h10" or "none"
DEM_SERIES, DEM_PER_EUR, DEM_LAST = "H10_DEM", Fraction("1.95583"), "1998-12-31"
EUR_RATE_FROM = (1998, 12)          # formation months from here use the euro-area rate, earlier ones the German rate
JP_CD_SERIES, JP_IB_FROM = "IR3TCD01JPM156N", (2002, 4)                   # ANSWERS A2
START = "1989-12-01"                # calendar start: the 1989-12 formation needs its last business day
ACTIVE_FROM = (1993, 1)             # filters act from the last business day of this month (36 month ends before it)
VIX_FROM = (1990, 1)
FILTERS = ("V70", "V80", "V90", "DV20", "DV40", "CS20", "CS40")
FREQS = ("M", "W", "D")
CANDS = ("C0",) + tuple(f"{f}-{q}" for f in FILTERS for q in FREQS)       # 22
COST, GROSS = 0.0004, 2.0           # 4 bp of the gross notional; long 1 + short 1 per dollar of capital
DEV = ((1990, 1), (2012, 12))       # holding months (ANSWERS A6: split by the start of the holding period)
SEALED = ((2013, 1), (2026, 2))     # ANSWERS A5 (b): last formation 2026-01, 158 holding months
HALVES = (((1990, 1), (2001, 12)), ((2002, 1), (2012, 12)))
MARGIN = 0.05
BOOT_B, BOOT_BLOCK, BOOT_SEED = 10_000, 12, 19


# ---------------------------------------------------------------- small helpers
def ym(d):
    return int(d[:4]), int(d[5:7])


def nxt(m):
    return (m[0], m[1] + 1) if m[1] < 12 else (m[0] + 1, 1)


def prv(m):
    return (m[0], m[1] - 1) if m[1] > 1 else (m[0] - 1, 12)


def months(a, b):
    out = []
    while a <= b:
        out.append(a)
        a = nxt(a)
    return out


def lab(m):
    return f"{m[0]}-{m[1]:02d}"


def frac(x):
    return x if isinstance(x, Fraction) else Fraction(str(x))


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def kind(name):
    """(filter, frequency) of a candidate; ('', '') for C0."""
    return ("", "") if name == "C0" else tuple(name.split("-"))


# ---------------------------------------------------------------- files -> series
def read_rows(path):
    """[(date, value text)] of a FRED-style CSV; '' and '.' mean no value."""
    with open(path, encoding="utf-8") as f:
        rows = [ln.rstrip("\n").split(",", 1) for ln in f.readlines()[1:] if ln.strip()]
    return [(d, "" if v.strip() in ("", ".") else v.strip()) for d, v in rows]


def series_rows(name, parts, root=None, read=None):
    """Rows of one series over the parts ('build', then 'sealed' only in the sealed run), in date order."""
    out, root = [], DATA if root is None else root
    for p in parts:
        path = os.path.join(root, p, f"{name}.csv")
        if read is not None:
            read.append(path)
        rows = read_rows(path)
        if any((d < "2013-01-01") != (p == "build") for d, _ in rows):
            raise ValueError(f"{path}: a row is on the wrong side of 2013-01-01")
        out += rows
    if [d for d, _ in out] != sorted({d for d, _ in out}):
        raise ValueError(f"{name}: dates are not strictly increasing")
    return out


def fx_usd_per_unit(rows, usd_per_unit):
    """{date: USD per unit of the foreign currency} from rows in the series' own quote."""
    return {d: (float(v) if usd_per_unit else 1.0 / float(v)) for d, v in rows if v}


def eur_slot(eur_rows, dem_rows):
    """USD per euro: the Fed's DM/US$ rate through 1998-12-31 as 1.95583 / x (the fixed conversion rate, so returns
    equal mark returns), DEXUSEU after it. dem_rows None = ANSWERS A1 path 'none'."""
    out = {d: float(DEM_PER_EUR) / float(v) for d, v in (dem_rows or []) if v and d <= DEM_LAST}
    out.update({d: float(v) for d, v in eur_rows if v and d > DEM_LAST})
    return out


def monthly_rate(rows):
    """{(y, m): Fraction} percent a year, from monthly rows dated on the first of the month."""
    return {ym(d): Fraction(v) for d, v in rows if v}


def splice(old, new, new_from):
    """Months before new_from from old, the rest from new."""
    out = {t: v for t, v in old.items() if t < new_from}
    out.update({t: v for t, v in new.items() if t >= new_from})
    return out


def load(parts=("build",), dem_path=DEM_PATH, end=None, root=None, cand=None):
    """(Data, files read). parts = ('build',) in development and rehearsal, ('build', 'sealed') in the sealed run.
    The series that end before 2013 in use (H10_DEM, Japan CD rate, German rate) are read from the build part only;
    the first two come from candidates/build. root, cand: other folders (synthetic tests)."""
    root, cand = DATA if root is None else root, CAND if cand is None else cand
    read = []
    fx = {}
    for c, (s, usd) in FX.items():
        rows = series_rows(s, parts, root, read)
        if c == "EUR":
            dem = series_rows(DEM_SERIES, ("build",), cand, read) if dem_path == "h10" else None
            fx[c] = eur_slot(rows, dem)
        else:
            fx[c] = fx_usd_per_unit(rows, usd)
    r = {k: monthly_rate(series_rows(f"IR3TIB01{v}M156N", parts, root, read)) for k, v in RATE.items()}
    de = monthly_rate(series_rows("IR3TIB01DEM156N", ("build",), root, read)) if dem_path == "h10" else {}
    r["EUR"] = splice(de, r["EUR"], EUR_RATE_FROM)
    r["JPY"] = splice(monthly_rate(series_rows(JP_CD_SERIES, ("build",), cand, read)), r["JPY"], JP_IB_FROM)
    vix = {d: Fraction(v) for d, v in series_rows("VIXCLS", parts, root, read) if v}
    baa = {d: Fraction(v) for d, v in series_rows("BAA10Y", parts, root, read) if v}
    return make_data(fx, r, vix, baa, START, end), read


def manifest(here=None, root=None, cand=None):
    """{absolute path: sha256} from every SEALED_MANIFEST19*.tsv (main series under DATA, the others under candidates)."""
    here, root, cand = HERE if here is None else here, DATA if root is None else root, CAND if cand is None else cand
    out = {}
    for f in sorted(os.listdir(here)):
        if f.startswith("SEALED_MANIFEST19") and f.endswith(".tsv"):
            base = root if f == "SEALED_MANIFEST19.tsv" else cand
            with open(os.path.join(here, f), encoding="utf-8") as fh:
                for ln in fh.readlines()[1:]:
                    x = ln.rstrip("\n").split("\t")
                    out[os.path.join(base, x[2])] = x[6]
    return out


def verify(read, here=None, root=None, cand=None):
    """Paths whose sha256 differs from the manifests (or that no manifest lists)."""
    man = manifest(here, root, cand)
    return [p for p in read if man.get(p) != sha(p)]


# ---------------------------------------------------------------- data on the H.10 calendar
class Data:
    """days: H.10 business days (ISO, sorted). S: N x 9 USD per unit, carried forward on a day without a fresh quote
    (nan before the first one); fresh: N x 9. rate: {currency or 'USD': {(y, m): Fraction}}. vix, baa: per day, the last
    observation at or before the day (Fraction or None). first, last: {(y, m): day index}."""


def asof(obs, days):
    out, keys, j, cur = [], sorted(obs), 0, None
    for d in days:
        while j < len(keys) and keys[j] <= d:
            cur = obs[keys[j]]
            j += 1
        out.append(cur)
    return out


def make_data(fx, rate, vix, baa, start=None, end=None):
    """fx: {currency: {date: USD per unit}}; rate: {currency or 'USD': {(y, m): percent}}; vix, baa: {date: value}."""
    D = Data()
    days = sorted({d for c in fx for d in fx[c] if (start is None or d >= start) and (end is None or d <= end)})
    D.days, N = days, len(days)
    D.date = [dt.date.fromisoformat(d) for d in days]
    D.ym = [ym(d) for d in days]
    D.S = np.full((N, len(CCY)), np.nan)
    D.fresh = np.zeros((N, len(CCY)), dtype=bool)
    for j, c in enumerate(CCY):
        q, cur = fx.get(c, {}), np.nan
        for i, d in enumerate(days):
            if d in q:
                cur = q[d]
                D.fresh[i, j] = True
            D.S[i, j] = cur
    D.rate = {k: {t: frac(v) for t, v in r.items()} for k, r in rate.items()}
    D.vix = asof({d: frac(v) for d, v in vix.items()}, days)
    D.baa = asof({d: frac(v) for d, v in baa.items()}, days)
    D.first, D.last = {}, {}
    for i, m in enumerate(D.ym):
        D.first.setdefault(m, i)
        D.last[m] = i
    D.gap = np.array([0] + [(D.date[i] - D.date[i - 1]).days for i in range(1, N)], dtype=float)
    return D


# ---------------------------------------------------------------- formation
def rate_at(r, t):
    """Month t, else month t - 1 (ANSWERS A4: a missing month is filled once with the month before), else None."""
    return r[t] if t in r else r.get(prv(t))


def portfolio(D, m):
    """Portfolio of holding month m, decided at the end of month m - 1: (column indexes, dollar weights, differentials
    in percent) or None. A currency needs its rate and the US rate for the formation month and an H.10 quote on the
    entry day and on the exit day (the last day of the data when the exit day is beyond it)."""
    t = prv(m)
    if m not in D.first or t not in D.first:
        return None
    us = rate_at(D.rate.get("USD", {}), t)
    if us is None:
        return None
    i0 = D.first[m]
    i1 = D.first.get(nxt(m), len(D.days) - 1)
    av = []
    for j, c in enumerate(CCY):
        r = rate_at(D.rate.get(c, {}), t)
        if r is not None and D.fresh[i0, j] and D.fresh[i1, j]:
            av.append((-(r - us), j))
    k = 3 if len(av) >= 6 else 2 if len(av) >= 4 else 0
    if k == 0:
        return None
    av.sort()
    pick = av[:k] + av[-k:]
    return (np.array([j for _, j in pick]), np.array([1.0 / k] * k + [-1.0 / k] * k), np.array([float(-d) for d, _ in pick]))


def portfolios(D):
    return {m: portfolio(D, m) for m in D.first}


# ---------------------------------------------------------------- filters
def pct(sorted_vals, q):
    """q-th percentile with linear interpolation (numpy's default), exact on Fractions."""
    n = len(sorted_vals)
    h = Fraction((n - 1) * q, 100)
    lo = h.numerator // h.denominator
    if lo + 1 >= n:
        return sorted_vals[-1]
    return sorted_vals[lo] + (h - lo) * (sorted_vals[lo + 1] - sorted_vals[lo])


def month_end(D, x):
    """{(y, m): value as of the last business day of the month} for months with a value."""
    return {m: x[i] for m, i in D.last.items() if x[i] is not None}


def conditions(D):
    """{filter: bool per day}: the condition at the close of each day, with the day's value against the month ends
    before the day's month (ADDENDUM 2). Strict '>'; False when a value or a reference is missing."""
    N = len(D.days)
    vme, bme = month_end(D, D.vix), month_end(D, D.baa)
    on = {f: np.zeros(N, dtype=bool) for f in FILTERS}
    thr = {}
    for m in D.first:
        ref = sorted(v for t, v in vme.items() if VIX_FROM <= t < m)
        thr[m] = {q: pct(ref, q) for q in (70, 80, 90)} if ref else None
    for i, m in enumerate(D.ym):
        v, b, p = D.vix[i], D.baa[i], prv(m)
        if v is not None and thr[m] is not None:
            for q in (70, 80, 90):
                on[f"V{q}"][i] = v > thr[m][q]
        if v is not None and p in vme:
            for x in (20, 40):
                on[f"DV{x}"][i] = v > vme[p] * Fraction(100 + x, 100)
        if b is not None and p in bme:
            for x in (20, 40):
                on[f"CS{x}"][i] = b - bme[p] > Fraction(x, 100)
    return on


def check_days(D, freq):
    """bool per day: M last business day of the month, W last business day of the ISO week, D every business day."""
    N = len(D.days)
    if freq == "D":
        return np.ones(N, dtype=bool)
    key = D.ym if freq == "M" else [d.isocalendar()[:2] for d in D.date]
    return np.array([i == N - 1 or key[i + 1] != key[i] for i in range(N)])


def want(D, cond, freq, active_from=ACTIVE_FROM):
    """bool per day: invested after the close of the day. The state changes on check days only and stays True (C0)
    before the last business day of active_from (None = active from the first day)."""
    N = len(D.days)
    chk = check_days(D, freq)
    a = 0 if active_from is None else D.last.get(active_from, 0 if D.ym and D.ym[0] > active_from else N)
    out, cur = np.ones(N, dtype=bool), True
    for i in range(N):
        if chk[i] and i >= a:
            cur = not cond[i]
        out[i] = cur
    return out


# ---------------------------------------------------------------- daily simulation
def simulate(D, wnt=None, cost=COST, ports=None):
    """Daily path of one candidate. wnt: output of want() (None = C0). Returns per day i:
    pnl (spot + interest of the interval that ends on day i), cst (cost of the trades on day i), held (a position is
    open after the trades of day i) and the event flags entry, mid_exit, mid_entry, exit (invested -> flat by the filter)."""
    N = len(D.days)
    ports = portfolios(D) if ports is None else ports
    out = {k: np.zeros(N) for k in ("pnl", "cst")}
    out.update({k: np.zeros(N, dtype=bool) for k in ("held", "entry", "mid_exit", "mid_entry", "exit")})
    cur = None                                                             # (columns, units, interest per day)
    for i in range(N):
        if cur is not None:
            idx, units, carry = cur
            out["pnl"][i] = float(units @ (D.S[i, idx] - D.S[i - 1, idx])) + carry * D.gap[i]
        m = D.ym[i]
        p = ports.get(m)
        desired = p is not None and (wnt is None or i == 0 or bool(wnt[i - 1]))
        trade = None
        if i == D.first[m]:                                                # scheduled roll: the old position ends free
            was, cur = cur is not None, None
            if desired:
                trade = "entry"
            elif was and p is not None:
                out["exit"][i] = True
        elif cur is not None and not desired:
            cur = None
            out["mid_exit"][i] = out["exit"][i] = True
            out["cst"][i] += cost * GROSS
        elif cur is None and desired:
            trade = "mid_entry"
        if trade:
            idx, w, dr = p
            cur = (idx, w / D.S[i, idx], float(w @ dr) / 100.0 / 365.0)
            out["entry"][i] = True
            out["mid_entry"][i] = trade == "mid_entry"
            out["cst"][i] += cost * GROSS
        out["held"][i] = cur is not None
    return out


def run_all(D, cost=COST, active_from=ACTIVE_FROM, names=CANDS):
    """{candidate: simulate(...)} for the 22 candidates."""
    ports, cond = portfolios(D), conditions(D)
    out = {}
    for name in names:
        f, q = kind(name)
        out[name] = simulate(D, None if name == "C0" else want(D, cond[f], q, active_from), cost, ports)
    return out


# ---------------------------------------------------------------- monthly returns and statistics
def monthly(D, sim, window):
    """(months, gross, cost) of the holding months in the window that have a business day in the data: gross = P&L of
    the intervals that start in the month, cost = cost of the trades made in the month."""
    ms = [m for m in months(*window) if m in D.first]
    pos = {m: k for k, m in enumerate(ms)}
    g, c = np.zeros(len(ms)), np.zeros(len(ms))
    for i in range(len(D.days)):
        if i > 0 and D.ym[i - 1] in pos:
            g[pos[D.ym[i - 1]]] += sim["pnl"][i]
        if D.ym[i] in pos:
            c[pos[D.ym[i]]] += sim["cst"][i]
    return ms, g, c


def incomplete(D, window):
    """Holding months of the window whose exit day (first business day of the next month) is not in the data."""
    return [m for m in months(*window) if m not in D.first or nxt(m) not in D.first]


def sharpe(x):
    x = np.asarray(x, dtype=float)
    s = x.std(ddof=1) if len(x) > 1 else 0.0
    return float(x.mean() / s * math.sqrt(12)) if s > 0 else 0.0


def max_drawdown(x):
    """Largest fall of the running sum from its earlier peak (no compounding; the sum starts at 0)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    return float(np.max(np.maximum.accumulate(c) - c))


def worst12(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 12:
        return float("nan")
    c = np.concatenate([[0.0], np.cumsum(x)])
    return float(np.min(c[12:] - c[:-12]))


def stats(D, sim, window, ports=None):
    """Statistics of one candidate over the holding months of a window (net of the cost inside sim)."""
    ms, g, c = monthly(D, sim, window)
    net = g - c
    ports = portfolios(D) if ports is None else ports
    inw = set(ms)
    start = np.array([m in inw for m in D.ym])                             # the interval or trade of day i is in the window
    live = start & np.array([ports.get(m) is not None for m in D.ym])
    live[-1] = False                                                       # no interval starts on the last day
    daily = -np.where(start, sim["cst"], 0.0)
    daily[1:] += np.where(start[:-1], sim["pnl"][1:], 0.0)
    years = len(ms) / 12.0
    gs, cs = float(g.sum()), float(c.sum())
    vol = float(net.std(ddof=1) * math.sqrt(12)) if len(net) > 1 else float("nan")
    return {"months": len(ms), "sharpe": sharpe(net), "ann_return": float(net.mean() * 12), "ann_vol": vol,
            "max_drawdown": max_drawdown(net), "max_drawdown_daily": max_drawdown(daily), "worst_12m": worst12(net),
            "gross_sum": gs, "cost_sum": cs, "net_sum": float(net.sum()), "cost_share": cs / gs if gs > 0 else None,
            "exits_per_year": float(sim["exit"][start].sum() / years), "mid_exits_per_year": float(sim["mid_exit"][start].sum() / years),
            "mid_entries_per_year": float(sim["mid_entry"][start].sum() / years), "entries_per_year": float(sim["entry"][start].sum() / years),
            "flat_share": float(1.0 - sim["held"][live].mean()) if live.any() else None,
            "flat_months": int(sum(1 for k, m in enumerate(ms) if c[k] == 0 and g[k] == 0))}


def table(D, sims, window, sims2=None):
    """{candidate: stats + monthly net list (+ sharpe at the doubled cost from sims2)}."""
    ports = portfolios(D)
    out = {}
    for name, sim in sims.items():
        ms, g, c = monthly(D, sim, window)
        r = stats(D, sim, window, ports)
        r["net"] = [float(x) for x in g - c]
        for k, h in enumerate(HALVES):
            sel = np.array([h[0] <= m <= h[1] for m in ms])
            r[f"half{k + 1}_sum"] = float((g - c)[sel].sum()) if sel.any() else None
        if sims2 is not None:
            r["sharpe_cost2"] = stats(D, sims2[name], window, ports)["sharpe"]
        out[name] = r
    return out


def vix_change(D, window):
    """Month-end VIX of each holding month minus the month end before it (None where either is missing)."""
    vme = month_end(D, D.vix)
    return [float(vme[m] - vme[prv(m)]) if m in vme and prv(m) in vme else None for m in months(*window) if m in D.first]


def corr(a, b):
    k = [i for i in range(len(a)) if a[i] is not None and b[i] is not None]
    if len(k) < 3:
        return None
    x, y = np.array([a[i] for i in k], dtype=float), np.array([b[i] for i in k], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------- bootstrap, Holm, selection
def boot_index(T, B=BOOT_B, block=BOOT_BLOCK, seed=BOOT_SEED):
    """Circular blocks of months as an index matrix B x T (same construction as engine18.boot_index)."""
    rng = np.random.default_rng(seed)
    k = math.ceil(T / block)
    return (rng.integers(0, T, size=(B, k))[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :T] % T


def boot_sharpe(x, idx):
    """Annualized Sharpe of each resample (0 where the resample has no variation)."""
    r = np.asarray(x, dtype=float)[idx]
    s = r.std(axis=1, ddof=1)
    return np.where(s > 0, r.mean(axis=1) / np.where(s > 0, s, 1.0), 0.0) * math.sqrt(12)


def one_sided(est, dist, level):
    """One-sided bootstrap test of 'statistic > 0': p = share of resamples <= 0; lower bounds at the level and at 2.5%."""
    return {"estimate": float(est), "p": float(np.mean(dist <= 0)), "bound_at_level": float(np.percentile(dist, 100 * level)),
            "bound_97_5": float(np.percentile(dist, 2.5)), "ci95": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
            "boot_se": float(dist.std(ddof=1))}


def holm(p, alpha=0.05):
    """{name: {p, level, rejected}}: the k-th smallest p against alpha / (m - k), stopping at the first failure
    (two hypotheses: 0.025 then 0.05; one: 0.05)."""
    order = sorted(p, key=lambda k: (p[k], k))
    res, go = {}, True
    for k, name in enumerate(order):
        lvl = alpha / (len(order) - k)
        go = go and p[name] < lvl
        res[name] = {"p": p[name], "level": lvl, "rejected": bool(go)}
    return res


def mde(se, alpha, power=0.80):
    z = statistics.NormalDist().inv_cdf
    return (z(1 - alpha) + z(power)) * se


def select_astar(rows):
    """ADDENDUM 4 on {candidate: {'sharpe', 'half1_sum', 'half2_sum', 'flat_share'}} (development, net of cost).
    1-2: the best filter candidate by Sharpe among those with a positive net sum in both halves. 3: A* = C0 unless it
    beats C0 by at least 0.05. 4: among the candidates within 0.05 of the best that also beat C0 by 0.05 and pass the
    halves rule: lower frequency (M, W, D), then the smaller flat share, then the order of CANDS."""
    eps = 1e-12
    ok = [c for c in rows if c != "C0" and rows[c]["half1_sum"] > 0 and rows[c]["half2_sum"] > 0]
    rec = {"c0_sharpe": rows["C0"]["sharpe"], "passed_halves": ok,
           "failed_halves": [c for c in rows if c != "C0" and c not in ok],
           "c0_halves": [rows["C0"]["half1_sum"], rows["C0"]["half2_sum"]]}
    if not ok:
        rec.update(best=None, reason="no filter candidate has a positive net sum in both halves")
        return "C0", rec
    best = max(ok, key=lambda c: (rows[c]["sharpe"], -list(rows).index(c)))
    rec.update(best=best, best_sharpe=rows[best]["sharpe"], best_minus_c0=rows[best]["sharpe"] - rows["C0"]["sharpe"])
    if rows[best]["sharpe"] - rows["C0"]["sharpe"] < MARGIN - eps:
        rec["reason"] = "the best filter candidate is not 0.05 above C0"
        return "C0", rec
    tie = [c for c in ok if rows[c]["sharpe"] >= rows[best]["sharpe"] - MARGIN - eps
           and rows[c]["sharpe"] - rows["C0"]["sharpe"] >= MARGIN - eps]
    pick = min(tie, key=lambda c: (FREQS.index(kind(c)[1]), rows[c]["flat_share"], list(rows).index(c)))
    rec.update(tie=tie, reason="lowest frequency, then smallest flat share, among the candidates within 0.05 of the best")
    return pick, rec
