#!/usr/bin/env python3
"""Research 21 step 1: the unit of the displayed sizes in the 1-second bars (SPEC21 3: sizes from 2025-11-03 are to be
divided by 100; research 20's post hoc audit found them 100 times the true shares from that day).

Only the two size columns (bsz, asz) of each bar file are read. No price is read and no size level is reported: the
only statistic is, per symbol-day,
  g        the greatest common divisor of all positive values of bsz and asz (0 when there is none)
  before   days to 2025-10-31;  after: days from 2025-11-03
The correction fits a symbol when
  before   every g is a multiple of 100 (sizes in shares, quoted in round lots of 100 shares)
  after    every g is a multiple of 100 x 10 (the round lot is 100, 40 or 10 shares for a price up to 250, 1,000 or
           10,000 dollars, so sizes that are 100 times the shares are multiples of 10,000, 4,000 or 1,000), so that the
           corrected sizes are whole round lots
Symbols without a day after the change (not in the lists of 2025 or 2026) have nothing to check there.
The files are taken from the downloader's MANIFEST (a file must exist; hashes are check21.py's job).
Writes results/check21_sizes.json (counts) and $DATA_DIR/research21_data/quality/sizes_gcd.csv (sym, date, part, g).
Step 2 (prompt 5): the days on which a symbol's g changes after 2025-11-03 are listed (a change between 10,000 and 4,000
is a new round lot, e.g. on the first business day of May; a g that is not a multiple of 1,000 would be another unit).
With --part dev|sealed only that part is read; with --tag <name> both file names get the suffix _<name> (default: the
part's name). Refuses to overwrite.
  python check21_sizes.py --part dev|sealed [--workers 4]"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.environ.get("DATA_DIR", "data"), "research21_data")
CHANGE, LAST_BEFORE = "20251103", "20251031"
ROUND_LOTS = (100, 40, 10)


def gcd_of(path):
    with np.load(path) as z:
        v = np.concatenate([z["bsz"], z["asz"]]).astype(np.int64)
    v = v[v > 0]
    return int(np.gcd.reduce(v)) if len(v) else 0


def main(tag="", parts=("dev", "sealed"), workers=8):
    outs = [os.path.join(BASE, "quality", f"sizes_gcd{tag}.csv"), os.path.join(HERE, "results", f"check21_sizes{tag}.json")]
    if any(os.path.exists(o) for o in outs):
        raise SystemExit(f"an output exists (nothing is overwritten): {[os.path.basename(o) for o in outs if os.path.exists(o)]}")
    rows = []
    for part in parts:
        root = os.path.join(BASE, part, "bars_1s")
        m = pd.read_csv(os.path.join(root, "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False).drop_duplicates(["sym", "date"], keep="last")
        rows += [(r.sym, r.date, part, os.path.join(root, r.file)) for r in m.itertuples()]
    t = pd.DataFrame(rows, columns=["sym", "date", "part", "path"])
    with mp.Pool(workers) as pool:
        t["g"] = pool.map(gcd_of, t.path, chunksize=64)
    t["period"] = np.where(t.date < CHANGE, "before", "after")
    os.makedirs(os.path.join(BASE, "quality"), exist_ok=True)
    t[["sym", "date", "part", "g"]].sort_values(["date", "sym"]).to_csv(os.path.join(BASE, "quality", f"sizes_gcd{tag}.csv"), index=False)
    b, a = t[t.period == "before"], t[t.period == "after"]
    ok_b, ok_a = (b.g > 0) & (b.g % 100 == 0), (a.g > 0) & (a.g % 1000 == 0)
    per = {}
    for s, g in t.groupby("sym"):
        gb, ga = g[g.period == "before"], g[g.period == "after"]
        per[s] = {"days_before": int(len(gb)), "days_after": int(len(ga)),
                  "g_before": {str(k): int(v) for k, v in gb.g.value_counts().sort_index().items()},
                  "g_after": {str(k): int(v) for k, v in ga.g.value_counts().sort_index().items()},
                  "last_day_before": gb.date.max() if len(gb) else None, "first_day_after": ga.date.min() if len(ga) else None,
                  "g_changes_after": [[d, int(x), int(y)] for d, x, y in zip(ga.sort_values("date").date[1:], ga.sort_values("date").g[:-1], ga.sort_values("date").g[1:]) if x != y],
                  "days_after_not_multiple_of_1000": sorted(ga.date[~((ga.g > 0) & (ga.g % 1000 == 0))]),
                  "days_before_not_multiple_of_100": sorted(gb.date[~((gb.g > 0) & (gb.g % 100 == 0))]),
                  "fits": bool(((gb.g > 0) & (gb.g % 100 == 0)).all() and ((ga.g > 0) & (ga.g % 1000 == 0)).all())}
    chg = sorted({c[0] for v in per.values() for c in v["g_changes_after"]})
    out = {"files": int(len(t)), "parts": list(parts), "change_day": CHANGE, "last_day_before": LAST_BEFORE,
           "first_day": t.date.min(), "last_day": t.date.max(),
           "days_with_a_change_of_g_after": {d: sorted(s for s, v in per.items() if any(c[0] == d for c in v["g_changes_after"])) for d in chg},
           "symbol_days_after_not_multiple_of_100": int((~((a.g > 0) & (a.g % 100 == 0))).sum()),
           "days_between": sorted(set(t.date[(t.date > LAST_BEFORE) & (t.date < CHANGE)])),
           "before": {"symbol_days": int(len(b)), "symbols": int(b.sym.nunique()), "g_counts": {str(k): int(v) for k, v in b.g.value_counts().sort_index().items()},
                      "multiple_of_100": int(ok_b.sum()), "not_multiple_of_100": int((~ok_b).sum())},
           "after": {"symbol_days": int(len(a)), "symbols": int(a.sym.nunique()), "g_counts": {str(k): int(v) for k, v in a.g.value_counts().sort_index().items()},
                     "g_over_100_counts": {str(k): int(v) for k, v in (a.g // 100).value_counts().sort_index().items()},
                     "multiple_of_1000": int(ok_a.sum()), "not_multiple_of_1000": int((~ok_a).sum()),
                     "exactly_100_times_a_round_lot": int(a.g.isin([100 * r for r in ROUND_LOTS]).sum())},
           "symbols_after": sorted(a.sym.unique()), "symbols_that_do_not_fit": sorted(s for s, v in per.items() if not v["fits"]),
           "correction_fits_every_symbol": bool(all(v["fits"] for v in per.values())), "by_symbol": per}
    with open(os.path.join(HERE, "results", f"check21_sizes{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "by_symbol"}, indent=1))
    for s in out["symbols_after"]:
        print(s, "before", per[s]["g_before"], "after", per[s]["g_after"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--part", choices=["dev", "sealed"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tag = a.tag or (a.part or "")
    main(f"_{tag}" if tag else "", (a.part,) if a.part else ("dev", "sealed"), a.workers)
