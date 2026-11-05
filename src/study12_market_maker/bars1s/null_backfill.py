#!/usr/bin/env python3
"""NULL counts for symbol-days already downloaded (Kyle 2026-10-27, proposal 2): server-side counts only, one connection.
  python null_backfill.py --split dev --go        (run after the download of that split has finished)
For every (sym, date) in <split>/MANIFEST.tsv that is not yet in <split>/null_counts_backfill.tsv, one query per table and
date (all symbols of that date at once) counts, with the same definitions as download.py's null_counts.tsv:
  q_null_pre      complete_nbbo rows with any NULL best_bid / best_ask / best_bidsizeshares / best_asksizeshares among the
                  last 500 rows before 09:30:00 (order time_m desc, time_m_nano desc; ties beyond ns may differ from the
                  downloader's LIMIT 500)
  q_null_session  the same over 09:30:00 <= time_m < 16:00:00
  t_null_price, t_null_size, t_null_corr, t_null_scond   ctm session rows with that column NULL
Rows go to <split>/null_counts_backfill.tsv (a separate file, never the downloader's null_counts.tsv), source
'server_count', one fsync'ed append per date. Any error or warning stops the run (no retry). Only counts are printed."""
import argparse
import datetime as dt
import os
import sys
import warnings

import pandas as pd

import download

FILE = "null_counts_backfill.tsv"
Q = "(best_bid is null or best_ask is null or best_bidsizeshares is null or best_asksizeshares is null)"
PRE_SQL = """select sym_root, sum(case when {q} then 1 else 0 end) as q_null_pre
from (select sym_root, best_bid, best_ask, best_bidsizeshares, best_asksizeshares,
             row_number() over (partition by sym_root order by time_m desc, time_m_nano desc) as rn
      from {lib}.complete_nbbo_{d} where sym_root in ({syms}) and sym_suffix is null and time_m < '09:30:00') x
where rn <= {n} group by 1"""
SES_SQL = """select sym_root, sum(case when {q} then 1 else 0 end) as q_null_session
from {lib}.complete_nbbo_{d} where sym_root in ({syms}) and sym_suffix is null and time_m >= '09:30:00' and time_m < '16:00:00'
group by 1"""
TRD_SQL = """select sym_root, sum(case when price is null then 1 else 0 end) as t_null_price,
       sum(case when size is null then 1 else 0 end) as t_null_size, sum(case when tr_corr is null then 1 else 0 end) as t_null_corr,
       sum(case when tr_scond is null then 1 else 0 end) as t_null_scond
from {lib}.ctm_{d} where sym_root in ({syms}) and sym_suffix is null and time_m >= '09:30:00' and time_m < '16:00:00'
group by 1"""
CNT = ["q_null_pre", "q_null_session", "t_null_price", "t_null_size", "t_null_corr", "t_null_scond"]


def day_counts(db, day, syms):
    """DataFrame (sym, date, counts...) for one date; symbols without rows get 0."""
    kw = {"lib": f"taqm_{day[:4]}", "d": day, "syms": ",".join(f"'{s}'" for s in syms), "q": Q, "n": download.PREOPEN_ROWS}
    out = pd.DataFrame({"sym": sorted(syms)}).set_index("sym")
    for sql in (PRE_SQL, SES_SQL, TRD_SQL):
        r = db.raw_sql(sql.format(**kw)).set_index("sym_root")
        out = out.join(r, how="left")
    out = out.reindex(columns=CNT).fillna(0).astype(int).reset_index()
    out.insert(1, "date", day)
    return out


def backfill(db, root, log=print):
    man = pd.read_csv(os.path.join(root, "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False)
    path = os.path.join(root, FILE)
    have = set()
    if os.path.exists(path):
        h = pd.read_csv(path, sep="\t", dtype=str)
        have = set(zip(h.sym, h.date))
    else:
        with open(path, "w") as f:
            f.write("\t".join(download.NULL_COLS) + "\n")
    todo = sorted({(s, d) for s, d in zip(man.sym, man.date)} - have, key=lambda x: (x[1], x[0]))
    days = sorted({d for _, d in todo})
    n = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for day in days:
            syms = [s for s, d in todo if d == day]
            c = day_counts(db, day, syms)
            c["source"] = "server_count"
            c["written_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with open(path, "a") as f:
                f.write(c[download.NULL_COLS].to_csv(sep="\t", header=False, index=False))
                f.flush()
                os.fsync(f.fileno())
            n += len(c)
            nz = c[(c[CNT] > 0).any(axis=1)]
            log(f"{day}: {len(c)} symbol-days, with any NULL: {len(nz)}")
    return {"days": len(days), "rows": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=sorted(download.SPLITS))
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if not a.go:
        sys.exit("refused: needs --go (Kyle's permission)")
    import wrds
    db = wrds.Connection()
    try:
        r = backfill(db, os.path.join(download.BASE, f"bars_1s_{a.split}"))
        print(f"done: {r['days']} days, {r['rows']} symbol-day rows")
    finally:
        db.close()


if __name__ == "__main__":
    main()
