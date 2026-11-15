#!/usr/bin/env python3
"""Research 23: the unit of the displayed sizes in the sealed universe D bars (SPEC23 1; step-2 prompt 4: the method of
research 22's section 4, study22_market_signal/check22_sizes.py with the names of research 23 and one change for empty files).
Only the two size columns (bsz, asz) of each bar file are read. No price is read and no size level is reported: the only
statistic is, per symbol-day,
  g        the greatest common divisor of all positive values of bsz and asz (0 when there is none)
  before   days to 2025-10-31;  after: days from 2025-11-03
The correction (divide by 100 from 2025-11-03) fits a symbol when
  before   every g is a multiple of 100 (sizes in shares, quoted in round lots of 100 shares)
  after    every g is a multiple of 100 x 10 (the round lot is 100, 40 or 10 shares, so sizes that are 100 times the shares
           are multiples of 10,000, 4,000 or 1,000), so that the corrected sizes are whole round lots
A file without any positive size all day (g = 0: no quote and no order all day) is not counted as "another unit"; such
files are listed on their own (step-2 prompt 4; Kyle's decision of 2026-11-13 in research 22). The days on which a symbol's
g changes after 2025-11-03 are listed: a change between 10,000, 4,000 and 1,000 is a new round lot (first business day of
May and November), a g that is not a multiple of 1,000 would be another unit. A symbol or day that does not fit means:
stop and report.
The files are taken from the downloader's MANIFEST (a file must exist; hashes are check23.py's job).
Writes results/check23_sizes.json (counts) and $DATA_DIR/research23_data/quality/sizes_gcd.csv (sym, date, g). Nothing is
overwritten.
  python check23_sizes.py [--workers 4]"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research23_data/sealed/bars_1s_new_stocks")
QUAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research23_data/quality")
CHANGE, LAST_BEFORE = "20251103", "20251031"
ROUND_LOTS = (100, 40, 10)
OUTS = (os.path.join(QUAL, "sizes_gcd.csv"), os.path.join(HERE, "results", "check23_sizes.json"))


def gcd_of(path):
    with np.load(path) as z:
        v = np.concatenate([z["bsz"], z["asz"]]).astype(np.int64)
    v = v[v > 0]
    return int(np.gcd.reduce(v)) if len(v) else 0


def main(workers):
    if any(os.path.exists(o) for o in OUTS):
        raise SystemExit(f"an output exists (nothing is overwritten): {[os.path.basename(o) for o in OUTS if os.path.exists(o)]}")
    m = pd.read_csv(os.path.join(ROOT, "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False).drop_duplicates(["sym", "date"], keep="last")
    t = pd.DataFrame({"sym": m.sym.to_numpy(), "date": m.date.to_numpy(), "path": [os.path.join(ROOT, f) for f in m.file]})
    with mp.Pool(workers) as pool:
        t["g"] = pool.map(gcd_of, t.path, chunksize=64)
    t["period"] = np.where(t.date < CHANGE, "before", "after")
    os.makedirs(QUAL, exist_ok=True)
    t[["sym", "date", "g"]].sort_values(["date", "sym"]).to_csv(OUTS[0], index=False)
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
                  "last_day_before": gb.date.max() if len(gb) else None, "first_day_after": ga.date.min() if len(ga) else None,
                  "g_changes_after": [[d, int(x), int(y)] for d, x, y in zip(ga.date[1:], ga.g[:-1], ga.g[1:]) if x != y],
                  "days_after_not_multiple_of_1000": sorted(ga.date[ga.g % 1000 != 0]),
                  "days_before_not_multiple_of_100": sorted(gb.date[gb.g % 100 != 0]),
                  "fits": bool((gb.g % 100 == 0).all() and (ga.g % 1000 == 0).all())}
    chg = sorted({c[0] for v in per.values() for c in v["g_changes_after"]})
    out = {"files": int(len(t)), "change_day": CHANGE, "last_day_before": LAST_BEFORE, "first_day": t.date.min(), "last_day": t.date.max(),
           "files_without_any_positive_size": {"count": int(len(empty)), "symbol_days": sorted([s, d] for s, d in zip(empty.sym, empty.date)),
                                               "note": "listed apart, not counted as another unit"},
           "days_with_a_change_of_g_after": {d: sorted(s for s, v in per.items() if any(c[0] == d for c in v["g_changes_after"])) for d in chg},
           "changes_after": sorted([s, *c] for s, v in per.items() for c in v["g_changes_after"]),
           "symbol_days_after_not_multiple_of_100": int((a.g % 100 != 0).sum()),
           "days_between": sorted(set(t.date[(t.date > LAST_BEFORE) & (t.date < CHANGE)])),
           "before": {"symbol_days": int(len(b)), "symbols": int(b.sym.nunique()), "g_counts": {str(k): int(v) for k, v in b.g.value_counts().sort_index().items()},
                      "multiple_of_100": int(ok_b.sum()), "not_multiple_of_100": int((~ok_b).sum())},
           "after": {"symbol_days": int(len(a)), "symbols": int(a.sym.nunique()), "g_counts": {str(k): int(v) for k, v in a.g.value_counts().sort_index().items()},
                     "multiple_of_1000": int(ok_a.sum()), "not_multiple_of_1000": int((~ok_a).sum()),
                     "exactly_100_times_a_round_lot": int(a.g.isin([100 * r for r in ROUND_LOTS]).sum())},
           "symbols_after": sorted(a.sym.unique()), "symbols_that_do_not_fit": sorted(s for s, v in per.items() if not v["fits"]),
           "correction_fits_every_symbol": bool(all(v["fits"] for v in per.values())), "by_symbol": per}
    with open(OUTS[1], "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "by_symbol"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args().workers)
