#!/usr/bin/env python3
"""Research 20 step 1 (supplement to check20.py): how late the first quote state of the day arrives.

Time range only, from the downloader's MANIFEST: no bar file is opened and no price, size or spread is read.
  first_state_sec  the first second (0 = 09:30:00) at whose end a valid NBBO state exists (bars.py: the last valid update
                   before the end of the second, the last 500 rows before 09:30:00 included); -1 = no state all day
  init_state       a valid state already exists at 09:30:00.000
Counts per part, year and symbol. Writes results/check20_open_state.json.
  python check20_open.py"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.environ.get("DATA_DIR", "data"), "research20_data")
BINS = [-2, -1, 0, 5, 60, 300, 1800, 23400]
LABELS = ["none_all_day", "0", "1-5s", "6-60s", "61-300s", "301-1800s", "over_1800s"]


def main():
    out = {}
    for part in ("dev", "sealed"):
        m = pd.read_csv(os.path.join(BASE, part, "bars_1s", "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False)
        m = m.drop_duplicates(["sym", "date"], keep="last")
        m["year"] = m.date.str[:4]
        f = m.first_state_sec.astype(int)
        m["bucket"] = pd.cut(f, bins=BINS, labels=LABELS).astype(str)
        late = m[f != 0]
        out[part] = {"files": int(len(m)), "first_state_sec_not_0": int(len(late)),
                     "init_state_false": int((m.init_state != "True").sum()),
                     "preopen_rows_zero": int((m.n_quote_preopen.astype(int) == 0).sum()),
                     "max_first_state_sec": int(f.max()),
                     "by_year": {y: {b: int((g.bucket == b).sum()) for b in LABELS} for y, g in m.groupby("year")},
                     "late_by_year_and_symbol": {y: g.sym.value_counts().sort_index().to_dict() for y, g in late.groupby("year")},
                     "late_by_symbol": late.sym.value_counts().sort_index().to_dict()}
    with open(os.path.join(HERE, "results", "check20_open_state.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
