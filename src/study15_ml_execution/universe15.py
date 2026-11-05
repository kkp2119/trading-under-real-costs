#!/usr/bin/env python3
"""Research 15 training universe (Kyle 2026-10-29): for each trading year Y = 2015..2018, screen year Y-1 and draw 10
symbols. Copy of research12/universe_rolling.py with the research-15 changes marked "R15". One WRDS connection
(PostgreSQL client credentials). Statistics go to files only; the log holds counts, timings and tickers.

Screen of year X = Y-1 (same rule as research12/universe_rolling.py):
  dates     the 15th of each month of X, or the next date with a ctm table
  universe  taqm_X.mastm_<last ctm date of X> sec_type 'A', no symbol suffix, minus EXCLUDE
  per day   ctm: n_trades, dollar volume, mean trade price; complete_nbbo: row-weighted mean spread in 0.01 ticks
            (research-12 trade allowlist and quote validity, 09:30:00 <= t < 16:00:00)
  pass      trade and quote data on >= 10 of 12 days; medians: price 10-150, dollar volume 5e6-1e8, trades >= 3,000,
            spread 3-12 ticks
Order (R15: seed s = 12 + Y for every year): rng = numpy default_rng(s); the first 10 are rng.choice(passing, 10) in draw
order, the rest rng.permutation of the remaining passing symbols.
Selection: the first 10 in that order with at least one valid session trade and one valid session quote on the first
trading day of Y (same filters). R15 (Kyle decision 1): for Y = 2016..2018, symbols in the tick size pilot test groups
G1-G3 (data_meta/tsp_test_group_symbols.tsv, tsp_pilot.py) are skipped before that; the control group is not skipped.
R15 common stock (Kyle 2026-10-29): the 2014 and 2015 masters have no sec_type column (it is filled from mastm_20161024).
For such X a symbol is common when its master CUSIP (first 8 characters) matches crsp.stocknames ncusip with shrcd 10,
11, 12 or 18 (rule R3, Kyle 2026-10-29 15:41 ET; R1 = 10/11 failed the gate) and namedt <= mdate <= nameenddt (point
in time, mdate = the master date). Gate: crsp_agree15.py, agreement with TAQ 'A' over all symbols >= 0.95.
R15 quote filter: natbbo_ind and qu_source are compared through coalesce(.., '') so a NULL indicator (UTP rows before
2016-10) is not dropped by SQL three-valued logic; this is the bars.py rule (fillna("")). The research-12 SQL dropped
every UTP quote row with a NULL natbbo_ind (filled in the samples checked from 2016-11-15 on).
R15 implementation: the per-day aggregates run over every no-suffix master symbol (minus EXCLUDE); the security type
is a column (taq_sec_type_A, crsp_common) and pass_common applies the rule of that year (per-symbol statistics are
independent, so the passing set equals a server-side type filter).
Files (research15/): universe15_screen_<X>.tsv, universe15_draw_order_<X>.tsv, universe15_<Y>.tsv;
logs/universe15_<X>.log (one log per screen year)."""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import wrds

HERE = os.path.dirname(os.path.abspath(__file__))
EXCLUDE = {"AAPL", "MSFT", "SH", "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "TLT", "IEF", "GLD", "UUP", "EEM", "HYG"}
DAY_LIMIT_SEC = 600
PILOT_YEARS = {2016, 2017, 2018}                                                    # R15
PILOT_FILE = os.path.join(HERE, "data_meta", "tsp_test_group_symbols.tsv")         # R15
TRADE_OK = """sym_suffix is null and time_m >= '09:30:00' and time_m < '16:00:00' and tr_corr = '00' and price > 0 and size > 0
  and (tr_scond is null or tr_scond ~ '^[@FI ]*$')"""
QUOTE_OK = """sym_suffix is null and time_m >= '09:30:00' and time_m < '16:00:00' and qu_cond = 'R'
  and not (coalesce(qu_source, '') = 'C' and coalesce(natbbo_ind, '') = 'O')
  and not (coalesce(qu_source, '') = 'N' and coalesce(natbbo_ind, '') = '1')
  and best_bid > 0 and best_ask >= best_bid and best_bidsizeshares > 0 and best_asksizeshares > 0"""
COMMON = "select symbol_15 from taqm_{x}.mastm_{m} where sec_type = 'A' and position(' ' in symbol_15) = 0"
ALL_NOSUFFIX = "select symbol_15 from taqm_{x}.mastm_{m} where position(' ' in symbol_15) = 0"            # R15
COMMON_CRSP = """select m.symbol_15 from taqm_{x}.mastm_{m} m where position(' ' in m.symbol_15) = 0 and exists (
  select 1 from crsp.stocknames s where s.ncusip = substr(m.cusip, 1, 8) and s.shrcd in (10, 11, 12, 18)
  and s.namedt <= date '{d}' and s.nameenddt >= date '{d}')"""                                                   # R15
TRD = """select sym_root as sym, count(*) as n_trades, sum(price * size) as dollar_volume, avg(price) as mean_price
from taqm_{x}.ctm_{d} where {ok} and sym_root in ({common}) group by 1"""
NBB = """select sym_root as sym, count(*) as n_quotes, avg((best_ask - best_bid) / 0.01) as spread_ticks
from taqm_{x}.complete_nbbo_{d} where {ok} and sym_root in ({common}) group by 1"""
CNT_T = "select sym_root as sym, count(*) as n from taqm_{y}.ctm_{d} where {ok} and sym_root in ({syms}) group by 1"
CNT_Q = "select sym_root as sym, count(*) as n from taqm_{y}.complete_nbbo_{d} where {ok} and sym_root in ({syms}) group by 1"


def table_days(db, year, kind):
    t = db.raw_sql(f"select table_name from information_schema.tables where table_schema = 'taqm_{year}' "
                   f"and table_name ~ '^{kind}_{year}[0-9]{{4}}$'")
    return sorted(x[-8:] for x in t.table_name)


def has_sec_type(db, x, mdate):                                                    # R15: column present and filled
    c = db.raw_sql(f"select 1 from information_schema.columns where table_schema = 'taqm_{x}' "
                   f"and table_name = 'mastm_{mdate}' and column_name = 'sec_type'")
    return len(c) > 0 and int(db.raw_sql(f"select count(sec_type) as n from taqm_{x}.mastm_{mdate}").n[0]) > 0


def screen(db, x, log):
    days = table_days(db, x, "ctm")
    nbb = set(table_days(db, x, "complete_nbbo"))
    pick = [min(d for d in days if d >= f"{x}{m:02d}15") for m in range(1, 13)]
    mdate = max(table_days(db, x, "mastm"))
    typed = has_sec_type(db, x, mdate)                                              # R15
    common_sql = ALL_NOSUFFIX.format(x=x, m=mdate)                                  # R15: type applied below
    common = set(db.raw_sql(common_sql).symbol_15) - EXCLUDE
    taq_a = set(db.raw_sql(COMMON.format(x=x, m=mdate)).symbol_15) if typed else set()
    crsp = set(db.raw_sql(COMMON_CRSP.format(x=x, m=mdate, d=f"{mdate[:4]}-{mdate[4:6]}-{mdate[6:]}")).symbol_15)
    rule = "taq_sec_type_A" if typed else "crsp_shrcd_10_11_12_18"
    log(f"screen {x}: dates {pick}; complete_nbbo missing on {[d for d in pick if d not in nbb]}; no-suffix symbols (mastm_{mdate}) "
        f"{len(common)}; common by {rule}: {len(common & (taq_a if typed else crsp))} (TAQ sec_type A {len(common & taq_a) if typed else 'n/a'}, "
        f"CRSP shrcd 10/11/12/18 {len(common & crsp)})")
    tr, nb = [], []
    for d in pick:
        t0 = time.time()
        tr.append(db.raw_sql(TRD.format(x=x, d=d, ok=TRADE_OK, common=common_sql)).assign(date=d))
        t1 = time.time()
        if d in nbb:
            nb.append(db.raw_sql(NBB.format(x=x, d=d, ok=QUOTE_OK, common=common_sql)).assign(date=d))
        t2 = time.time()
        log(f"  {d}: ctm {t1 - t0:.0f}s, complete_nbbo {t2 - t1:.0f}s")
        if t2 - t0 > DAY_LIMIT_SEC:
            raise SystemExit(f"STOP: {d} took {t2 - t0:.0f}s > {DAY_LIMIT_SEC}s")
    tr, nb = pd.concat(tr), pd.concat(nb)
    g = tr[tr.sym.isin(common)].groupby("sym").agg(n_days_trade=("date", "nunique"), med_price=("mean_price", "median"),
                                                   med_dollar_volume=("dollar_volume", "median"), med_trades=("n_trades", "median"))
    h = nb[nb.sym.isin(common)].groupby("sym").agg(n_days_quote=("date", "nunique"), med_spread_ticks=("spread_ticks", "median"))
    s = pd.DataFrame(index=sorted(common)).join(g).join(h)
    s[["n_days_trade", "n_days_quote"]] = s[["n_days_trade", "n_days_quote"]].fillna(0).astype(int)
    s["taq_sec_type_A"] = s.index.isin(taq_a) if typed else pd.NA                  # R15
    s["crsp_common"] = s.index.isin(crsp)
    s["common_rule"] = rule
    s["pass_common"] = s.index.isin(taq_a if typed else crsp)
    s["pass_days"] = (s.n_days_trade >= 10) & (s.n_days_quote >= 10)
    s["pass_price"] = s.med_price.between(10, 150)
    s["pass_dollar_volume"] = s.med_dollar_volume.between(5e6, 1e8)
    s["pass_trades"] = s.med_trades >= 3000
    s["pass_spread"] = s.med_spread_ticks.between(3, 12)
    s["pass_all"] = s[["pass_common", "pass_days", "pass_price", "pass_dollar_volume", "pass_trades", "pass_spread"]].all(axis=1)
    s.index.name = "sym"
    s.round({"med_price": 4, "med_dollar_volume": 0, "med_trades": 1, "med_spread_ticks": 3}).to_csv(
        os.path.join(HERE, f"universe15_screen_{x}.tsv"), sep="\t")
    log(f"  pass counts: " + ", ".join(f"{c} {int(s[c].sum())}" for c in
        ("pass_common", "pass_days", "pass_price", "pass_dollar_volume", "pass_trades", "pass_spread", "pass_all")))
    five = s[["pass_days", "pass_price", "pass_dollar_volume", "pass_trades", "pass_spread"]].all(axis=1)    # R15
    if typed:
        a, c = s.taq_sec_type_A[five].astype(bool), s.crsp_common[five]
        log(f"  type agreement among {int(five.sum())} symbols passing the 5 liquidity rules: TAQ A {int(a.sum())}, CRSP {int(c.sum())}, "
            f"both {int((a & c).sum())}, agree {float((a == c).mean()):.4f}")
    return sorted(s.index[s.pass_all])


def draw_order(passing, seed):
    rng = np.random.default_rng(seed)
    first = [str(v) for v in rng.choice(passing, 10, replace=False)]
    rest = [str(v) for v in rng.permutation(sorted(set(passing) - set(first)))]
    return first + rest


def select(db, x, y, order, seed, log):
    d = table_days(db, y, "ctm")[0]
    syms = ",".join(f"'{s}'" for s in order)
    t = db.raw_sql(CNT_T.format(y=y, d=d, ok=TRADE_OK, syms=syms)).set_index("sym").n
    q = db.raw_sql(CNT_Q.format(y=y, d=d, ok=QUOTE_OK, syms=syms)).set_index("sym").n if d in set(table_days(db, y, "complete_nbbo")) else pd.Series(dtype=int)
    o = pd.DataFrame({"order_rank": range(1, len(order) + 1), "sym": order})
    o["has_trades"] = o.sym.map(lambda s: bool(t.get(s, 0) > 0))
    o["has_quotes"] = o.sym.map(lambda s: bool(q.get(s, 0) > 0))
    pilot = set(pd.read_csv(PILOT_FILE, sep="\t", dtype=str).sym) if y in PILOT_YEARS else set()      # R15
    o["pilot_test_group"] = o.sym.isin(pilot)                                                          # R15
    ok = o.has_trades & o.has_quotes & ~o.pilot_test_group                                             # R15
    o["selected"] = ok & (ok.cumsum() <= 10)
    o.insert(0, "seed", seed)
    o.insert(1, "first_day", d)
    o.to_csv(os.path.join(HERE, f"universe15_draw_order_{x}.tsv"), sep="\t", index=False)
    u = o[o.selected][["sym", "order_rank"]].copy()
    u.insert(0, "year", y)
    u.insert(1, "slot", range(1, len(u) + 1))
    u.to_csv(os.path.join(HERE, f"universe15_{y}.tsv"), sep="\t", index=False)
    before = (o.order_rank <= int(u.order_rank.max())) & ~o.selected if len(u) else o.selected     # R15
    pil = o[before & o.pilot_test_group].sym.tolist()
    skipped = o[before & ~o.pilot_test_group].sym.tolist()
    log(f"universe {y} (first day {d}, seed {seed}): {u.sym.tolist()} | skipped as pilot test group: {pil} "
        f"| skipped for missing data on the first day: {skipped}")
    return u.sym.tolist()


def main():
    warnings.simplefilter("error")
    years = [int(a) for a in sys.argv[1:]] or list(range(2015, 2019))              # R15
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    db = wrds.Connection()
    try:
        for y in years:
            x = y - 1
            logf = open(os.path.join(HERE, "logs", f"universe15_{x}.log"), "a")     # R15: one log per screen year

            def log(m):
                print(m, flush=True)
                logf.write(m + "\n")
                logf.flush()
            try:
                passing = screen(db, x, log)
                seed = 12 + y                                                       # R15
                select(db, x, y, draw_order(passing, seed), seed, log)
            finally:
                logf.close()
    finally:
        db.close()


if __name__ == "__main__":
    main()
