#!/usr/bin/env python3
"""Research 12 rolling universe (Kyle 2026-10-27): for each trading year Y = 2019..2025, screen year Y-1 and draw 10 symbols.
One WRDS connection (PostgreSQL client credentials). Statistics go to files only; the log holds counts, timings and tickers.

Screen of year X = Y-1 (same rule as universe_screen_2018.py, criteria fixed before looking at the data):
  dates     the 15th of each month of X, or the next date with a ctm table
  universe  taqm_X.mastm_<last ctm date of X> sec_type 'A', no symbol suffix, minus EXCLUDE
  per day   ctm: n_trades, dollar volume, mean trade price; complete_nbbo: row-weighted mean spread in 0.01 ticks
            (research-12 trade allowlist and quote validity, 09:30:00 <= t < 16:00:00)
  pass      trade and quote data on >= 10 of 12 days; medians: price 10-150, dollar volume 5e6-1e8, trades >= 3,000,
            spread 3-12 ticks
Order (seed s = 12 for X = 2018, 12 + Y otherwise): rng = numpy default_rng(s); the first 10 are rng.choice(passing, 10)
in draw order (for 2018 these are the 10 reported on 2026-10-27), the rest rng.permutation of the remaining passing symbols.
Selection: the first 10 in that order with at least one valid session trade and one valid session quote on the first
trading day of Y (same filters). Symbols that later lose their data are not replaced (handled by the downloader).
Files (research12/): universe_screen_<X>.tsv (2018 already exists and is reused), universe_draw_order_<X>.tsv, universe_<Y>.tsv."""
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
TRADE_OK = """sym_suffix is null and time_m >= '09:30:00' and time_m < '16:00:00' and tr_corr = '00' and price > 0 and size > 0
  and (tr_scond is null or tr_scond ~ '^[@FI ]*$')"""
QUOTE_OK = """sym_suffix is null and time_m >= '09:30:00' and time_m < '16:00:00' and qu_cond = 'R'
  and not (qu_source = 'C' and natbbo_ind = 'O') and not (qu_source = 'N' and natbbo_ind = '1')
  and best_bid > 0 and best_ask >= best_bid and best_bidsizeshares > 0 and best_asksizeshares > 0"""
COMMON = "select symbol_15 from taqm_{x}.mastm_{m} where sec_type = 'A' and position(' ' in symbol_15) = 0"
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


def screen(db, x, log):
    days = table_days(db, x, "ctm")
    nbb = set(table_days(db, x, "complete_nbbo"))
    pick = [min(d for d in days if d >= f"{x}{m:02d}15") for m in range(1, 13)]
    mdate = max(table_days(db, x, "mastm"))
    common_sql = COMMON.format(x=x, m=mdate)
    common = set(db.raw_sql(common_sql).symbol_15) - EXCLUDE
    log(f"screen {x}: dates {pick}; complete_nbbo missing on {[d for d in pick if d not in nbb]}; universe (mastm_{mdate}) {len(common)}")
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
    s["pass_days"] = (s.n_days_trade >= 10) & (s.n_days_quote >= 10)
    s["pass_price"] = s.med_price.between(10, 150)
    s["pass_dollar_volume"] = s.med_dollar_volume.between(5e6, 1e8)
    s["pass_trades"] = s.med_trades >= 3000
    s["pass_spread"] = s.med_spread_ticks.between(3, 12)
    s["pass_all"] = s[["pass_days", "pass_price", "pass_dollar_volume", "pass_trades", "pass_spread"]].all(axis=1)
    s.index.name = "sym"
    s.round({"med_price": 4, "med_dollar_volume": 0, "med_trades": 1, "med_spread_ticks": 3}).to_csv(
        os.path.join(HERE, f"universe_screen_{x}.tsv"), sep="\t")
    log(f"  pass counts: " + ", ".join(f"{c} {int(s[c].sum())}" for c in
        ("pass_days", "pass_price", "pass_dollar_volume", "pass_trades", "pass_spread", "pass_all")))
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
    ok = o.has_trades & o.has_quotes
    o["selected"] = ok & (ok.cumsum() <= 10)
    o.insert(0, "seed", seed)
    o.insert(1, "first_day", d)
    o.to_csv(os.path.join(HERE, f"universe_draw_order_{x}.tsv"), sep="\t", index=False)
    u = o[o.selected][["sym", "order_rank"]].copy()
    u.insert(0, "year", y)
    u.insert(1, "slot", range(1, len(u) + 1))
    u.to_csv(os.path.join(HERE, f"universe_{y}.tsv"), sep="\t", index=False)
    skipped = o[(o.order_rank <= int(u.order_rank.max())) & ~o.selected].sym.tolist() if len(u) else []
    log(f"universe {y} (first day {d}, seed {seed}): {u.sym.tolist()} | skipped for missing data on the first day: {skipped}")
    return u.sym.tolist()


def main():
    warnings.simplefilter("error")
    logf = open(os.path.join(HERE, "universe_rolling.log"), "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()
    years = [int(a) for a in sys.argv[1:]] or list(range(2019, 2026))
    db = wrds.Connection()
    try:
        for y in years:
            x = y - 1
            if x == 2018:
                s = pd.read_csv(os.path.join(HERE, "universe_screen_2018.tsv"), sep="\t")
                passing = sorted(s.sym[s.pass_all])
                log(f"screen 2018: reused universe_screen_2018.tsv, pass_all {len(passing)}")
                seed = 12
            else:
                passing = screen(db, x, log)
                seed = 12 + y
            select(db, x, y, draw_order(passing, seed), seed, log)
    finally:
        db.close()
        logf.close()


if __name__ == "__main__":
    main()
