#!/usr/bin/env python3
"""Research 21 step 1: trading calendar 2015-01-02 .. the latest TAQ day (date, close_time), calendar21.csv.

Days and closing times to 2026-11-09 are research 20's (study20_pairs/calendar20.csv); later days come from the WRDS table
names (a day with a ctm, complete_nbbo and mastm table) with a 16:00 close. The result is compared with the ctm table
names of 2015-2026: table names only, no data row is read. One WRDS connection (the WRDS credentials file).
  python calendar21.py"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SOURCE = "study20_pairs/calendar20.csv"
HALF_DAYS_AFTER = ()                                         # no early close between 2026-11-10 and the latest TAQ day (checked below)


def main():
    cal = pd.read_csv(os.path.join(PROJ, SOURCE), dtype=str)
    import wrds
    db = wrds.Connection()
    try:
        t = db.raw_sql("select table_name from information_schema.tables where table_schema ~ '^taqm_20(1[5-9]|2[0-6])$' "
                       "and table_name ~ '^(ctm|complete_nbbo|mastm)_[0-9]{8}$'")
    finally:
        db.close()
    kind = t.table_name.str.rsplit("_", n=1).str[0]
    day = t.table_name.str[-8:]
    have = {k: set(day[kind == k]) for k in ("ctm", "complete_nbbo", "mastm")}
    last = max(have["ctm"] & have["complete_nbbo"] & have["mastm"])
    new = sorted(d for d in have["ctm"] if d > cal.date.max().replace("-", "") and d <= last)
    if any(d in HALF_DAYS_AFTER for d in new):
        raise SystemExit("an early close among the new days needs its closing time")
    cal = pd.concat([cal, pd.DataFrame({"date": [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in new], "close_time": "16:00"})], ignore_index=True)
    days = set(cal.date.str.replace("-", ""))
    ctm = {d for d in have["ctm"] if "20150102" <= d <= last}
    print(f"calendar {cal.date.min()} .. {cal.date.max()}: {len(cal)} days, early closes {int((cal.close_time != '16:00').sum())}; "
          f"days added after research 20's calendar: {new}")
    print(f"against the ctm table names: calendar days without a table {sorted(days - ctm)}, tables outside the calendar {sorted(ctm - days)}")
    print(f"calendar days without a complete_nbbo table {sorted(days - have['complete_nbbo'])}, without a master {sorted(days - have['mastm'])}")
    if days != ctm or not cal.date.is_monotonic_increasing or cal.date.duplicated().any():
        raise SystemExit("the calendar and the ctm table names differ")
    cal.to_csv(os.path.join(HERE, "calendar21.csv"), index=False)
    print(pd.Series(cal.date.str[:4]).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
