#!/usr/bin/env python3
"""Research 12 data step 1, item 1: value counts of the categorical code columns in the probe extracts
($DATA_DIR/research12_probe_raw, TLT/GLD/XLF x 2023-05-16, 2025-06-10, regular hours). Price and size columns are never
read (usecols). Also counts how many time_m strings match HH:MM:SS[.ffffff] and how many time_m_nano are non-zero
(timestamp format only). Output: logs/codes_count.json and the printed tables."""
import collections
import glob
import json
import os
import re

import pandas as pd

RAW = os.path.join(os.environ.get("DATA_DIR", "data"), "research12_probe_raw")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "codes_count.json")
COLS = {"cqm": ["ex", "qu_cond", "natbbo_ind"], "ctm": ["ex", "tr_scond", "tr_corr"],
        "complete_nbbo": ["qu_cond", "natbbo_ind", "qu_source", "nbbo_qu_cond"]}
TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}(\.\d{1,6})?$")


def main():
    res = {}
    for kind, cols in COLS.items():
        files = sorted(glob.glob(os.path.join(RAW, f"{kind}_*.csv.gz")))
        tot = {c: collections.Counter() for c in cols}
        chars = collections.Counter()
        per_file = {}
        tfmt = [0, 0]
        nano_nz = [0, 0]
        for f in files:
            use = cols + ["time_m"] + (["time_m_nano"] if kind == "complete_nbbo" else [])
            df = pd.read_csv(f, usecols=use, dtype=str, keep_default_na=False)
            per_file[os.path.basename(f)] = len(df)
            for c in cols:
                tot[c].update(df[c].map(lambda v: v if v != "" else "<empty>"))
            if kind == "ctm":
                for v in df["tr_scond"]:
                    chars.update(set(v.replace(" ", "")) or {"<empty>"})
            tfmt[0] += int(df["time_m"].str.match(TIME_RE).sum())
            tfmt[1] += len(df)
            if kind == "complete_nbbo":
                nano_nz[0] += int((df["time_m_nano"].astype(int) != 0).sum())
                nano_nz[1] += len(df)
        res[kind] = {"files": per_file, "counts": {c: dict(tot[c].most_common()) for c in cols},
                     "time_m_format_ok": tfmt}
        if kind == "ctm":
            res[kind]["tr_scond_chars"] = dict(chars.most_common())
        if kind == "complete_nbbo":
            res[kind]["time_m_nano_nonzero"] = nano_nz
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    for kind, r in res.items():
        print(f"== {kind}: rows {sum(r['files'].values())} in {len(r['files'])} files; time_m format ok {r['time_m_format_ok']}")
        for c, cnt in r["counts"].items():
            print(f"   {c}: {cnt}")
        if "tr_scond_chars" in r:
            print(f"   tr_scond characters (rows containing): {r['tr_scond_chars']}")
        if "time_m_nano_nonzero" in r:
            print(f"   time_m_nano non-zero: {r['time_m_nano_nonzero']}")


if __name__ == "__main__":
    main()
