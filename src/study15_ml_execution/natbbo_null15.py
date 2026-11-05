#!/usr/bin/env python3
"""Research 15 (Kyle 2026-10-29 15:41 ET): rows of complete_nbbo with a NULL natbbo_ind on 2016-11-15 and 2018-01-16,
the rows the research-12 screen SQL dropped through three-valued logic. Counts only (no price or size), by qu_source:
all rows, and the rows the screen would otherwise keep (no suffix, 09:30:00 <= t < 16:00:00, qu_cond 'R').
One WRDS connection (PostgreSQL client credentials). Appends to logs/natbbo_null15.log."""
import os

import wrds

HERE = os.path.dirname(os.path.abspath(__file__))
DAYS = ("20161115", "20180116")
SQL = """select coalesce(qu_source, 'NULL') as qu_source, count(*) as n_all,
  sum(case when sym_suffix is null and time_m >= '09:30:00' and time_m < '16:00:00' and qu_cond = 'R' then 1 else 0 end) as n_screen
from taqm_{y}.complete_nbbo_{d} where natbbo_ind is null group by 1 order by 1"""


def main():
    db = wrds.Connection()
    lines = []
    try:
        for d in DAYS:
            t = db.raw_sql(SQL.format(y=d[:4], d=d))
            by = {r.qu_source: (int(r.n_all), int(r.n_screen)) for r in t.itertuples()}
            lines.append(f"{d}: NULL natbbo_ind rows all {int(t.n_all.sum()) if len(t) else 0}, "
                         f"screen-relevant {int(t.n_screen.sum()) if len(t) else 0}; by qu_source (all, screen-relevant) {by}")
    finally:
        db.close()
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "natbbo_null15.log"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
