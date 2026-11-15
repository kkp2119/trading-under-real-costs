#!/usr/bin/env python3
"""Research 20 step 1: trading calendar 2015-01-02 .. the latest TAQ day (date, close_time), calendar20.csv
(SPEC20_ANSWERS A0: development from 2015).

Days and closing times are taken from the calendars already in use (study15_ml_execution/calendar15.csv for 2015-2018,
study12_market_maker/calendar.csv for 2019-01-02..2026-10-23, study18_forecast/calendar18.csv to 2026-11-05); later days come from the
WRDS table names (a day with a ctm table) with a 16:00 close. The result is compared with the ctm table names of
2015-2026: table names only, no data row is read. One WRDS connection (the WRDS credentials file).
  python calendar20.py"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SOURCES = (("study15_ml_execution/calendar15.csv", "2015-01-01", "2018-12-31"), ("study12_market_maker/calendar.csv", "2019-01-01", "2026-10-23"),
           ("study18_forecast/calendar18.csv", "2026-10-24", "2026-11-05"))
HALF_DAYS_AFTER = ()                                         # no early close between 2026-11-06 and the latest TAQ day (checked below)


def main():
    cal = pd.concat([c[(c.date >= a) & (c.date <= b)] for p, a, b in SOURCES
                     for c in [pd.read_csv(os.path.join(PROJ, p), dtype=str)]], ignore_index=True)
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
          f"days added after {SOURCES[-1][2]}: {new}")
    print(f"against the ctm table names: calendar days without a table {sorted(days - ctm)}, tables outside the calendar {sorted(ctm - days)}")
    print(f"calendar days without a complete_nbbo table {sorted(days - have['complete_nbbo'])}, without a master {sorted(days - have['mastm'])}")
    if days != ctm or not cal.date.is_monotonic_increasing or cal.date.duplicated().any():
        raise SystemExit("the calendar and the ctm table names differ")
    cal.to_csv(os.path.join(HERE, "calendar20.csv"), index=False)
    print(pd.Series(cal.date.str[:4]).value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
