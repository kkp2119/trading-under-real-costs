#!/usr/bin/env python3
"""Research 15 calendar (same rule as research12/calendar.csv, SPEC U2): the days 2015-01-02 .. 2018-12-31 with a ctm
table (table names only, one WRDS connection, PostgreSQL client credentials), close_time 16:00 or 13:00 on the NYSE early closes below
(copied from NYSE holiday rules; checked later against the MANIFEST last trade second). Writes research15/calendar15.csv."""
import os

import pandas as pd
import wrds

HERE = os.path.dirname(os.path.abspath(__file__))
YEARS = range(2015, 2019)
EARLY = {"2015-11-27", "2015-12-24", "2016-11-25", "2017-07-03", "2017-11-24", "2018-07-03", "2018-11-23", "2018-12-24"}


def main():
    db = wrds.Connection()
    try:
        t = db.raw_sql("select table_name from information_schema.tables where table_schema in ("
                       + ",".join(f"'taqm_{y}'" for y in YEARS) + ") and table_name ~ '^ctm_[0-9]{8}$'")
    finally:
        db.close()
    days = sorted(f"{x[-8:-4]}-{x[-4:-2]}-{x[-2:]}" for x in t.table_name)
    c = pd.DataFrame({"date": days, "close_time": ["13:00" if d in EARLY else "16:00" for d in days]})
    assert EARLY <= set(days), sorted(EARLY - set(days))
    c.to_csv(os.path.join(HERE, "calendar15.csv"), index=False)
    print(f"calendar15.csv: {len(c)} days {days[0]} .. {days[-1]}, by year "
          f"{c.date.str[:4].value_counts().sort_index().to_dict()}, early closes {int((c.close_time == '13:00').sum())}")


if __name__ == "__main__":
    main()
