#!/usr/bin/env python3
"""Research 21 step 1 (study20_pairs/check20_open.py with the names of research 21; supplement to check21.py): how late the
first quote state of the day arrives.

Time range only, from the downloader's MANIFEST: no bar file is opened and no price, size or spread is read.
  first_state_sec  the first second (0 = 09:30:00) at whose end a valid NBBO state exists (bars.py: the last valid update
                   before the end of the second, the last 500 rows before 09:30:00 included); -1 = no state all day
  init_state       a valid state already exists at 09:30:00.000
Counts per year and symbol for one part. Writes results/check21_open_state_<part>.json (refuses to overwrite).
  python check21_open.py --part dev|sealed"""
import argparse
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.environ.get("DATA_DIR", "data"), "research21_data")
BINS = [-2, -1, 0, 5, 60, 300, 1800, 23400]
LABELS = ["none_all_day", "0", "1-5s", "6-60s", "61-300s", "301-1800s", "over_1800s"]


def main(only):
    path = os.path.join(HERE, "results", f"check21_open_state_{only}.json")
    if os.path.exists(path):
        raise SystemExit(f"{os.path.basename(path)} exists (nothing is overwritten)")
    out = {}
    for part in (only,):
        m = pd.read_csv(os.path.join(BASE, part, "bars_1s", "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False)
        m = m.drop_duplicates(["sym", "date"], keep="last")
        m["year"] = m.date.str[:4]
        f = m.first_state_sec.astype(int)
        m["bucket"] = pd.cut(f, bins=BINS, labels=LABELS).astype(str)
        late = m[f != 0]
        out[part] = {"files": int(len(m)), "first_state_sec_not_0": int(len(late)),
                     "init_state_false": int((m.init_state != "True").sum()),
                     "preopen_rows_zero": int((m.n_quote_preopen.astype(int) == 0).sum()),
                     "max_first_state_sec": int(f.max()), "none_all_day": int((f < 0).sum()),
                     "late_sec_counts": {b: int((m.bucket[f != 0] == b).sum()) for b in LABELS if b != "0"},
                     "by_year": {y: {b: int((g.bucket == b).sum()) for b in LABELS} for y, g in m.groupby("year")},
                     "late_by_year_and_symbol": {y: g.sym.value_counts().sort_index().to_dict() for y, g in late.groupby("year")},
                     "late_by_symbol": late.sym.value_counts().sort_index().to_dict()}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["dev", "sealed"])
    main(ap.parse_args().part)
