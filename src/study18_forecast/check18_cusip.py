#!/usr/bin/env python3
"""Research 18 step 1 (copy of study17_new_stocks/check17_cusip.py; changed lines marked "R18"): the same identity check on
new universe B. R18: the days are every calendar18.csv day x universe18_<Y>.tsv symbol (a superset of the SEALED_MANIFEST18
days, so the check does not wait for the download; a symbol-day absent from the master is "no master row"), and the
reference CUSIP comes from research 18's permno_map.csv. Research 17 docstring follows.
Research 17 step 1 identity check of the sealed new-universe bars: the downloader (research 15 code) checks only that the
ticker is in each day's TAQ master, so a ticker taken over by another issuer would pass. For every universe17_<Y>.tsv
symbol and every SEALED_MANIFEST.tsv day, the day's master CUSIP (taqm_Y.mastm_<day>.cusip, master tables only; no quote
or trade table is read) is compared with the selection CUSIP (X = Y - 1 last master, permno_map.csv from universe17.py).
Daily CUSIPs stay in $DATA_DIR/research17_data/selection/cusip_daily.csv; results/check17_cusip.json has counts and dates.
One WRDS connection (the WRDS credentials file), one query per year.
  python check17_cusip.py"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research18_data/selection")                                  # R18


def main():
    cal = pd.read_csv(os.path.join(HERE, "calendar18.csv"), dtype=str).date.str.replace("-", "")       # R18: expected days
    man = pd.DataFrame([(s, d) for d in cal for s in pd.read_csv(os.path.join(HERE, f"universe18_{d[:4]}.tsv"), sep="\t",
                                                                 dtype=str).sym], columns=["sym", "date"])
    pm = pd.read_csv(os.path.join(LOCAL, "permno_map.csv"), dtype={"sym": str, "cusip": str})
    ref = pm[pm.role == "candidate"].drop_duplicates(["year", "sym"]).set_index(["year", "sym"]).cusip
    import wrds
    db = wrds.Connection()
    rows = []
    try:
        t = db.raw_sql("select table_name from information_schema.tables where table_schema ~ '^taqm_20(2[1-6])$' "
                       "and table_name ~ '^mastm_[0-9]{8}$'")                                  # R18: days with a master
        have = {n[-8:] for n in t.table_name}
        no_table = sorted(set(man.date) - have)
        for y, g in man[man.date.isin(have)].groupby(man.date.str[:4]):
            syms = ",".join(repr(s) for s in sorted(set(g.sym)))
            q = " union all ".join(f"select '{d}' as date, symbol_15 as sym, cusip from taqm_{y}.mastm_{d} where symbol_15 in ({syms})"
                                   for d in sorted(set(g.date)))
            rows.append(db.raw_sql(q))
    finally:
        db.close()
    c = pd.concat(rows).drop_duplicates(["date", "sym", "cusip"])
    c.to_csv(os.path.join(LOCAL, "cusip_daily.csv"), index=False)
    m = man[["sym", "date"]].merge(c, on=["sym", "date"], how="left")
    m["ref"] = [ref.get((int(d[:4]), s)) for s, d in zip(m.sym, m.date)]
    m["status"] = ["no master table" if d in no_table else "no master row" if pd.isna(x) else "same" if x == r else "different"
                   for d, x, r in zip(m.date, m.cusip, m.ref)]                                       # R18: no master table
    multi = c.groupby(["date", "sym"]).size()
    out = {"symbol_days": int(len(m)), "status": m.status.value_counts().to_dict(),
           "symbol_days_with_two_master_rows": int((multi > 1).sum()), "days_without_master_table": no_table, "different": []}
    for (y, s), g in m[m.status != "same"].groupby([m.date.str[:4], "sym"]):
        out["different"].append({"year": y, "sym": s, "status": g.status.value_counts().to_dict(), "first": g.date.min(),
                                 "last": g.date.max(), "days_same": int(((m.sym == s) & (m.date.str[:4] == y) & (m.status == "same")).sum())})
    with open(os.path.join(HERE, "results", "check18_cusip.json"), "w") as fh:                       # R18
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
