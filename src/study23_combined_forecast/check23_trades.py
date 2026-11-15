#!/usr/bin/env python3
"""Research 23: the unit of the traded volumes in the sealed universe D bars (SPEC23_ANSWERS A4; SPEC23_QUESTIONS 19;
step-2 prompt 4). The displayed sizes are 100 x the shares from 2025-11-03; whether the traded volumes changed on that day
has never been checked. Only the trade-volume columns and the trade count of each bar file are read (vol, vol_bid, vol_ask,
vol_in, vol_unc, vol_bid_odd, vol_ask_odd, ntr). No price and no quote is read and no volume level is reported. Per
symbol-day, in memory only:
  g        the greatest common divisor of all positive per-second values of the seven volume columns (0: no trade)
  size     shares per trade of the day (sum of vol / sum of ntr)
Reported: counts of g, and per symbol with days on both sides of the change the ratio median(size after) / median(size
before) (a ratio, not a level). before: days to 2025-10-31; after: days from 2025-11-03.
A 100-fold difference would show in two ways, and either one means: stop and report (no correction is applied here):
  ratio    a symbol whose ratio is at least 30 or at most 1 / 30
  gcd      a symbol whose g is a multiple of 100 on every traded day after the change (at least 5 such days) while fewer
           than half of its traded days before the change are (or it has no day before): volumes that are 100 x the
           shares are all multiples of 100, volumes in shares are not as soon as one odd lot trades
Files are taken from the downloader's MANIFEST. Writes results/check23_trades.json (counts and ratios) and
$DATA_DIR/research23_data/quality/trade_units_gcd.csv (sym, date, g). Nothing is overwritten.
  python check23_trades.py [--workers 4]"""
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
VOLS = ("vol", "vol_bid", "vol_ask", "vol_in", "vol_unc", "vol_bid_odd", "vol_ask_odd")
RATIO, MIN_DAYS = 30.0, 5
OUTS = (os.path.join(QUAL, "trade_units_gcd.csv"), os.path.join(HERE, "results", "check23_trades.json"))


def unit_of(path):
    """(g, shares per trade or nan) of one bar file."""
    with np.load(path) as z:
        v = np.concatenate([z[c] for c in VOLS]).astype(np.int64)
        vol, ntr = int(z["vol"].astype(np.int64).sum()), int(z["ntr"].astype(np.int64).sum())
    v = v[v > 0]
    return (int(np.gcd.reduce(v)) if len(v) else 0), (vol / ntr if ntr > 0 else float("nan"))


def main(workers):
    if any(os.path.exists(o) for o in OUTS):
        raise SystemExit(f"an output exists (nothing is overwritten): {[os.path.basename(o) for o in OUTS if os.path.exists(o)]}")
    m = pd.read_csv(os.path.join(ROOT, "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False).drop_duplicates(["sym", "date"], keep="last")
    t = pd.DataFrame({"sym": m.sym.to_numpy(), "date": m.date.to_numpy(), "path": [os.path.join(ROOT, f) for f in m.file]})
    with mp.Pool(workers) as pool:
        r = pool.map(unit_of, t.path, chunksize=64)
    t["g"], t["size"] = [x[0] for x in r], [x[1] for x in r]
    t["period"] = np.where(t.date < CHANGE, "before", "after")
    os.makedirs(QUAL, exist_ok=True)
    t[["sym", "date", "g"]].sort_values(["date", "sym"]).to_csv(OUTS[0], index=False)
    tr = t[t.g > 0]
    per, flag_ratio, flag_gcd = {}, [], []
    for s, g in tr.groupby("sym"):
        gb, ga = g[g.period == "before"], g[g.period == "after"]
        ratio = float(ga["size"].median() / gb["size"].median()) if len(gb) >= MIN_DAYS and len(ga) >= MIN_DAYS else None
        fb = float((gb.g % 100 == 0).mean()) if len(gb) else None
        fa = float((ga.g % 100 == 0).mean()) if len(ga) else None
        per[s] = {"traded_days_before": int(len(gb)), "traded_days_after": int(len(ga)), "size_ratio_after_over_before": ratio,
                  "share_of_days_g_multiple_of_100_before": fb, "share_of_days_g_multiple_of_100_after": fa}
        if ratio is not None and (ratio >= RATIO or ratio <= 1 / RATIO):
            flag_ratio.append(s)
        if len(ga) >= MIN_DAYS and fa == 1.0 and (fb is None or fb < 0.5):
            flag_gcd.append(s)
    ratios = np.array([v["size_ratio_after_over_before"] for v in per.values() if v["size_ratio_after_over_before"] is not None])
    b, a = tr[tr.period == "before"], tr[tr.period == "after"]
    out = {"files": int(len(t)), "files_without_a_trade": int((t.g == 0).sum()), "change_day": CHANGE, "last_day_before": LAST_BEFORE,
           "before": {"traded_symbol_days": int(len(b)), "symbols": int(b.sym.nunique()), "g_multiple_of_100": int((b.g % 100 == 0).sum()), "g_equal_1": int((b.g == 1).sum())},
           "after": {"traded_symbol_days": int(len(a)), "symbols": int(a.sym.nunique()), "g_multiple_of_100": int((a.g % 100 == 0).sum()), "g_equal_1": int((a.g == 1).sum())},
           "symbols_with_days_on_both_sides": int(len(ratios)),
           "size_ratio_after_over_before": {"min": float(ratios.min()) if len(ratios) else None, "median": float(np.median(ratios)) if len(ratios) else None,
                                            "max": float(ratios.max()) if len(ratios) else None},
           "flagged_by_ratio": sorted(flag_ratio), "flagged_by_gcd": sorted(flag_gcd),
           "hundredfold_difference_seen": bool(flag_ratio or flag_gcd), "thresholds": {"ratio": RATIO, "min_days": MIN_DAYS}, "by_symbol": per}
    with open(OUTS[1], "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "by_symbol"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args().workers)
