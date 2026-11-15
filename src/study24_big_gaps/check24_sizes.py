#!/usr/bin/env python3
"""Research 24: the unit of the displayed sizes in the 1-second bars of one part (SPEC24 3; step-2 prompt 2 and 5: the
method of research 21's section 5, with the handling of empty files that research 22 and 23 used).
Only the two size columns (bsz, asz) of each bar file are read. No price is read and no size level is reported: the only
statistic is, per symbol-day,
  g        the greatest common divisor of all positive values of bsz and asz (0 when there is none)
  before   days to 2025-10-31;  after: days from 2025-11-03
The correction (divide by 100 from 2025-11-03) fits a symbol when
  before   every g is a multiple of 100 (sizes in shares, quoted in round lots of 100 shares)
  after    every g is a multiple of 100 x 10 (the round lot is 100, 40 or 10 shares, so sizes that are 100 times the shares
           are multiples of 10,000, 4,000 or 1,000), so that the corrected sizes are whole round lots
A file without any positive size all day (g = 0) is not counted as "another unit"; such files are listed on their own
(Kyle's decision of 2026-11-13 in research 22). The days on which a symbol's g changes after 2025-11-03 are listed: a
change between 10,000, 4,000 and 1,000 is a new round lot (first business day of May and November), a g that is not a
multiple of 1,000 would be another unit. A symbol or day that does not fit means: stop and report.
The development part (2015-2020) has only days before the change. The files are taken from the downloader's MANIFEST (a
file must exist; hashes are check24.py's job).
Writes results/check24_sizes_<part>.json (counts) and $DATA_DIR/research24_data/quality/sizes_gcd_<part>.csv (sym, date, g).
Nothing is overwritten.
  python check24_sizes.py --part dev|sealed [--workers 4]"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.environ.get("DATA_DIR", "data"), "research24_data")
CHANGE, LAST_BEFORE = "20251103", "20251031"
ROUND_LOTS = (100, 40, 10)


def gcd_of(path):
    with np.load(path) as z:
        v = np.concatenate([z["bsz"], z["asz"]]).astype(np.int64)
    v = v[v > 0]
    return int(np.gcd.reduce(v)) if len(v) else 0


def main(part, workers):
    root = os.path.join(BASE, part, "bars_1s")
    outs = (os.path.join(BASE, "quality", f"sizes_gcd_{part}.csv"), os.path.join(HERE, "results", f"check24_sizes_{part}.json"))
    if any(os.path.exists(o) for o in outs):
        raise SystemExit(f"an output exists (nothing is overwritten): {[os.path.basename(o) for o in outs if os.path.exists(o)]}")
    m = pd.read_csv(os.path.join(root, "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False).drop_duplicates(["sym", "date"], keep="last")
    t = pd.DataFrame({"sym": m.sym.to_numpy(), "date": m.date.to_numpy(), "path": [os.path.join(root, f) for f in m.file]})
    with mp.Pool(workers) as pool:
        t["g"] = pool.map(gcd_of, t.path, chunksize=64)
    t["period"] = np.where(t.date < CHANGE, "before", "after")
    os.makedirs(os.path.dirname(outs[0]), exist_ok=True)
    t[["sym", "date", "g"]].sort_values(["date", "sym"]).to_csv(outs[0], index=False)
    empty = t[t.g == 0]
    u = t[t.g > 0]                                                        # the unit is judged on the files that hold a size
    b, a = u[u.period == "before"], u[u.period == "after"]
    ok_b, ok_a = b.g % 100 == 0, a.g % 1000 == 0
    per = {}
    for s, g in u.groupby("sym"):
        gb, ga = g[g.period == "before"], g[g.period == "after"].sort_values("date")
        per[s] = {"days_before": int(len(gb)), "days_after": int(len(ga)),
                  "g_before": {str(k): int(v) for k, v in gb.g.value_counts().sort_index().items()},
                  "g_after": {str(k): int(v) for k, v in ga.g.value_counts().sort_index().items()},
                  "g_changes_after": [[d, int(x), int(y)] for d, x, y in zip(ga.date[1:], ga.g[:-1], ga.g[1:]) if x != y],
                  "days_after_not_multiple_of_1000": sorted(ga.date[ga.g % 1000 != 0]),
                  "days_before_not_multiple_of_100": sorted(gb.date[gb.g % 100 != 0]),
                  "fits": bool((gb.g % 100 == 0).all() and (ga.g % 1000 == 0).all())}
    chg = sorted({c[0] for v in per.values() for c in v["g_changes_after"]})
    out = {"part": part, "files": int(len(t)), "change_day": CHANGE, "last_day_before": LAST_BEFORE, "first_day": t.date.min(), "last_day": t.date.max(),
           "files_without_any_positive_size": {"count": int(len(empty)), "symbol_days": sorted([s, d] for s, d in zip(empty.sym, empty.date)),
                                               "note": "listed apart, not counted as another unit"},
           "days_with_a_change_of_g_after": {d: sorted(s for s, v in per.items() if any(c[0] == d for c in v["g_changes_after"])) for d in chg},
           "changes_after": sorted([s, *c] for s, v in per.items() for c in v["g_changes_after"]),
           "symbol_days_after_not_multiple_of_100": int((a.g % 100 != 0).sum()),
           "before": {"symbol_days": int(len(b)), "symbols": int(b.sym.nunique()), "g_counts": {str(k): int(v) for k, v in b.g.value_counts().sort_index().items()},
                      "multiple_of_100": int(ok_b.sum()), "not_multiple_of_100": int((~ok_b).sum())},
           "after": {"symbol_days": int(len(a)), "symbols": int(a.sym.nunique()), "g_counts": {str(k): int(v) for k, v in a.g.value_counts().sort_index().items()},
                     "multiple_of_1000": int(ok_a.sum()), "not_multiple_of_1000": int((~ok_a).sum()),
                     "exactly_100_times_a_round_lot": int(a.g.isin([100 * r for r in ROUND_LOTS]).sum())},
           "symbols_that_do_not_fit": sorted(s for s, v in per.items() if not v["fits"]),
           "correction_fits_every_symbol": bool(all(v["fits"] for v in per.values())), "by_symbol": per}
    with open(outs[1], "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "by_symbol"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["dev", "sealed"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    main(a.part, a.workers)
