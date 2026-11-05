#!/usr/bin/env python3
"""Research 12 universe screen on 2018 data only (Kyle 2026-10-27; criteria fixed before looking at the data).
Server-side GROUP BY per date (one extra WRDS connection, PostgreSQL client credentials); no per-symbol download. Writes
research12/universe_screen_2018.tsv (per-symbol medians and pass flags) and prints only counts, timings and the medians of
the 10 drawn symbols.
Dates: the 15th of each 2018 month, or the next date with a ctm table.
Universe: taqm_2018.mastm_20181231 sec_type 'A' (Common Stock, Daily TAQ spec v3.3 Master File) with no symbol suffix
(symbol_15 without a space; the download pipeline uses sym_suffix is null), minus AAPL MSFT SH SPY QQQ and the 10 research-12 ETFs.
Per symbol-day, 09:30:00 <= time_m < 16:00:00, sym_suffix null:
  trades  (ctm): tr_corr '00', tr_scond NULL or only '@' 'F' 'I' ' ' (the research-12 allowlist), price > 0, size > 0
          -> n_trades, dollar volume sum(price * size), mean trade price avg(price)
  spread  (complete_nbbo): qu_cond 'R', not a no-NBBO indicator, best_bid > 0, best_ask >= best_bid, sizes > 0
          -> row-weighted mean of (best_ask - best_bid) / 0.01
Pass: trade data on >= 10 of 12 days and quote data on >= 10 days; medians over days with data: price 10-150, dollar
volume 5e6-1e8, trades >= 3,000, spread ticks 3-12. Draw: numpy default_rng(12).choice(sorted passing symbols, 10)."""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import wrds

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "universe_screen_2018.tsv")
EXCLUDE = {"AAPL", "MSFT", "SH", "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "TLT", "IEF", "GLD", "UUP", "EEM", "HYG"}
DAY_LIMIT_SEC = 300
COMMON = """select symbol_15 from taqm_2018.mastm_20181231 where sec_type = 'A' and position(' ' in symbol_15) = 0"""
TRD = """select sym_root as sym, count(*) as n_trades, sum(price * size) as dollar_volume, avg(price) as mean_price
from taqm_2018.ctm_{d}
where sym_suffix is null and time_m >= '09:30:00' and time_m < '16:00:00' and tr_corr = '00' and price > 0 and size > 0
  and (tr_scond is null or tr_scond ~ '^[@FI ]*$') and sym_root in ({common})
group by 1"""
NBB = """select sym_root as sym, count(*) as n_quotes, avg((best_ask - best_bid) / 0.01) as spread_ticks
from taqm_2018.complete_nbbo_{d}
where sym_suffix is null and time_m >= '09:30:00' and time_m < '16:00:00' and qu_cond = 'R'
  and not (qu_source = 'C' and natbbo_ind = 'O') and not (qu_source = 'N' and natbbo_ind = '1')
  and best_bid > 0 and best_ask >= best_bid and best_bidsizeshares > 0 and best_asksizeshares > 0 and sym_root in ({common})
group by 1"""


def main():
    warnings.simplefilter("error")
    db = wrds.Connection()
    try:
        t = db.raw_sql("select table_name from information_schema.tables where table_schema = 'taqm_2018' and table_name ~ '^ctm_2018[0-9]{4}$'")
        days = sorted(x[-8:] for x in t.table_name)
        pick = [min(d for d in days if d >= f"2018{m:02d}15") for m in range(1, 13)]
        common = set(db.raw_sql(COMMON).symbol_15) - EXCLUDE
        print(f"dates {pick}; common stocks (A, no suffix, at 2018-12-31) after exclusions: {len(common)}", flush=True)
        tr, nb = [], []
        for i, d in enumerate(pick):
            t0 = time.time()
            a = db.raw_sql(TRD.format(d=d, common=COMMON)).assign(date=d)
            t1 = time.time()
            b = db.raw_sql(NBB.format(d=d, common=COMMON)).assign(date=d)
            t2 = time.time()
            tr.append(a)
            nb.append(b)
            print(f"{d}: ctm {t1 - t0:.1f}s ({len(a)} symbols), complete_nbbo {t2 - t1:.1f}s ({len(b)} symbols)", flush=True)
            if i == 0 and t2 - t0 > DAY_LIMIT_SEC:
                print(f"STOP: first date took {t2 - t0:.0f}s > {DAY_LIMIT_SEC}s; report before running all 12 dates")
                return
    finally:
        db.close()
    tr, nb = pd.concat(tr), pd.concat(nb)
    tr = tr[tr.sym.isin(common)]
    nb = nb[nb.sym.isin(common)]
    g = tr.groupby("sym").agg(n_days_trade=("date", "nunique"), med_price=("mean_price", "median"),
                              med_dollar_volume=("dollar_volume", "median"), med_trades=("n_trades", "median"))
    h = nb.groupby("sym").agg(n_days_quote=("date", "nunique"), med_spread_ticks=("spread_ticks", "median"))
    s = pd.DataFrame(index=sorted(common)).join(g).join(h)
    s[["n_days_trade", "n_days_quote"]] = s[["n_days_trade", "n_days_quote"]].fillna(0).astype(int)
    s["pass_days"] = (s.n_days_trade >= 10) & (s.n_days_quote >= 10)
    s["pass_price"] = s.med_price.between(10, 150)
    s["pass_dollar_volume"] = s.med_dollar_volume.between(5e6, 1e8)
    s["pass_trades"] = s.med_trades >= 3000
    s["pass_spread"] = s.med_spread_ticks.between(3, 12)
    s["pass_all"] = s[["pass_days", "pass_price", "pass_dollar_volume", "pass_trades", "pass_spread"]].all(axis=1)
    passing = sorted(s.index[s.pass_all])
    drawn = list(np.random.default_rng(12).choice(passing, 10, replace=False)) if len(passing) >= 10 else []
    s["selected"] = s.index.isin(drawn)
    s.index.name = "sym"
    s.round({"med_price": 4, "med_dollar_volume": 0, "med_trades": 1, "med_spread_ticks": 3}).to_csv(OUT, sep="\t")
    print(f"screened {len(s)}; pass counts: " + ", ".join(f"{c} {int(s[c].sum())}" for c in
          ("pass_days", "pass_price", "pass_dollar_volume", "pass_trades", "pass_spread", "pass_all")))
    print("drawn (seed 12, in draw order):", drawn)
    print(s.loc[sorted(drawn), ["n_days_trade", "n_days_quote", "med_price", "med_dollar_volume", "med_trades", "med_spread_ticks"]]
          .round({"med_price": 2, "med_dollar_volume": 0, "med_trades": 0, "med_spread_ticks": 2}).to_string())


if __name__ == "__main__":
    main()
