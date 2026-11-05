#!/usr/bin/env python3
"""Research 17 step 1 identity check of the sealed new-universe bars: the downloader (research 15 code) checks only that the
ticker is in each day's TAQ master, so a ticker taken over by another issuer would pass. For every universe17_<Y>.tsv
symbol and every SEALED_MANIFEST.tsv day, the day's master CUSIP (taqm_Y.mastm_<day>.cusip, master tables only; no quote
or trade table is read) is compared with the selection CUSIP (X = Y - 1 last master, permno_map.csv from universe17.py).
Daily CUSIPs stay in $DATA_DIR/research17_data/selection/cusip_daily.csv; results/check17_cusip.json has counts and dates.
One WRDS connection (PostgreSQL client credentials), one query per year.
  python check17_cusip.py"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research17_data/selection")


def main():
    man = pd.read_csv(os.path.join(HERE, "SEALED_MANIFEST.tsv"), sep="\t", dtype=str)
    pm = pd.read_csv(os.path.join(LOCAL, "permno_map.csv"), dtype={"sym": str, "cusip": str})
    ref = pm[pm.role == "candidate"].drop_duplicates(["year", "sym"]).set_index(["year", "sym"]).cusip
    import wrds
    db = wrds.Connection()
    rows = []
    try:
        for y, g in man.groupby(man.date.str[:4]):
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
    m["status"] = ["no master row" if pd.isna(x) else "same" if x == r else "different" for x, r in zip(m.cusip, m.ref)]
    multi = c.groupby(["date", "sym"]).size()
    out = {"symbol_days": int(len(m)), "status": m.status.value_counts().to_dict(),
           "symbol_days_with_two_master_rows": int((multi > 1).sum()), "different": []}
    for (y, s), g in m[m.status != "same"].groupby([m.date.str[:4], "sym"]):
        out["different"].append({"year": y, "sym": s, "status": g.status.value_counts().to_dict(), "first": g.date.min(),
                                 "last": g.date.max(), "days_same": int(((m.sym == s) & (m.date.str[:4] == y) & (m.status == "same")).sum())})
    with open(os.path.join(HERE, "results", "check17_cusip.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
