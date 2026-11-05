#!/usr/bin/env python3
"""A2 recomputed under SPEC_AMENDMENTS A13-2 (dev only). Openable flags per day and combination come from
out/A2_daily_open.csv written by g0.py, whose rules equal the approved switch table (OptionMetrics delta with
Black-Scholes only when missing, T1, non-standard codes ['m', 'w']). Structural gaps come from chain.range_status on each
day's standard monthly exdates. Also: gap shares by combination and year, standard monthly spacing, a schedule-based
cross-check of the gap days, and defect (ii) counts under A13-2 per combination (for information)."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import chain as ch  # noqa: E402
import omdata as om  # noqa: E402

OUT = os.path.join(HERE, "out")
NONSTD = ("m", "w")                                   # approved switch table (2026-10-28 16:33 ET)
RANGE = {21: (35, 65), 10: (19, 49)}
COMBOS = ["S21", "S10", "G21", "G10"]
FIRST_DATA = 19960104

# the M19 cases of tests/test_chain.py, run again here before any computation (A13-2 change 5)
assert ch.range_status(20080303, [], 35, 65) == "no_monthly"
assert ch.range_status(20080303, [20080322, 20080517], 35, 65) == "gap"
assert ch.range_status(20080303, [20080322, 20080419], 35, 65) == "in_range"
assert not ch.defect_ii("gap", False) and ch.defect_ii("in_range", False) and ch.defect_ii("no_monthly", False)
assert not ch.defect_ii("no_monthly", False, gate_open=False)

read = []
st = om.Store("dev")
cal = ch.load_calendar("dev", read=read)
op_path = os.path.join(OUT, "A2_daily_open.csv")
read.append(op_path)
op = pd.read_csv(op_path)
op_sha = hashlib.sha256(open(op_path, "rb").read()).hexdigest()
days = [int(d) for d in cal.days if d >= FIRST_DATA]

rows, all_std = [], set()
for y in range(1996, 2013):
    a = st.load(f"opprcd/opprcd_SPX_{y}.npz")
    df = pd.DataFrame({k: a[k] for k in ("date", "exdate", "am_settlement")})
    codes = np.array(list(a["expiry_indicator__codes"]) + [None], dtype=object)
    c = a["expiry_indicator"].astype(int)
    df["expiry_indicator"] = codes[np.where(c < 0, len(codes) - 1, c)]
    u = df.drop_duplicates()
    s = u[ch.standard_mask(u, cal, NONSTD)][["date", "exdate"]].drop_duplicates()
    all_std |= set(s.exdate.astype(int))
    by = s.groupby("date").exdate.apply(lambda x: x.to_numpy())
    for d in [x for x in days if x // 10000 == y]:
        ex = by.get(d, np.array([], dtype=np.int64))
        rows.append({"date": d, **{f"st{H}": ch.range_status(d, ex, lo, hi) for H, (lo, hi) in RANGE.items()}})
R = pd.DataFrame(rows).merge(op, on="date", how="left")
assert R[COMBOS].notna().all().all() and len(R) == len(days)

# schedule-based cross-check: a gap day per chain should have no standard exdate of the whole sample in range
allx = np.array(sorted(all_std))
xc = {}
for H, (lo, hi) in RANGE.items():
    sched_gap = [not ((ch.dte(np.full(len(allx), d), allx) >= lo) & (ch.dte(np.full(len(allx), d), allx) <= hi)).any()
                 for d in R.date]
    xc[H] = int(((R[f"st{H}"] == "gap") != np.array(sched_gap)).sum())

R["month"] = R.date // 100
mon = []
for m, g in R.groupby("month"):
    row = {"month": int(m)}
    for cmb in COMBOS:
        H = int(cmb[1:])
        elig = g[g[f"st{H}"] != "gap"]
        if not len(elig):
            raise SystemExit(f"{m} {cmb}: no eligible day (rule undefined)")
        row[f"{cmb}_den"] = len(elig)
        row[f"{cmb}_ratio"] = round(float(elig[cmb].mean()), 5)
    row["all_pass"] = all(row[f"{c}_ratio"] >= 0.98 for c in COMBOS)
    mon.append(row)
MON = pd.DataFrame(mon)
months, passes = MON.month.tolist(), MON.all_pass.tolist()
start = next((months[i] for i in range(len(months) - 11) if all(passes[i:i + 12])), None)
dev_start = FIRST_DATA if start == months[0] else (min(d for d in days if d // 100 == start) if start else None)
i0 = int(np.searchsorted(cal.days, dev_start)) if dev_start else None
stats_start = int(cal.days[i0 + 43]) if i0 is not None else None
stats_days = int((cal.days >= stats_start).sum()) if stats_start else 0

R["year"] = R.date // 10000
gap = R.groupby("year").agg(days=("date", "size"), gap21=("st21", lambda s: int((s == "gap").sum())),
                            gap10=("st10", lambda s: int((s == "gap").sum())),
                            no_monthly=("st21", lambda s: int((s == "no_monthly").sum()))).reset_index()
gap["gap21_share"] = (gap.gap21 / gap.days).round(4)
gap["gap10_share"] = (gap.gap10 / gap.days).round(4)
spacing = pd.Series(np.diff([ch.ymd(x).toordinal() for x in allx])).value_counts().sort_index()
SP = pd.DataFrame({"gap_days": spacing.index, "count": spacing.to_numpy()})
SP["flag_not_28_35"] = ~SP.gap_days.isin([28, 35])
defects = {c: int(sum(ch.defect_ii(s, bool(o)) for s, o in zip(R[f"st{int(c[1:])}"], R[c]))) for c in COMBOS}

MON.to_csv(os.path.join(OUT, "A2_a13_2_monthly.csv"), index=False)
gap.to_csv(os.path.join(OUT, "A2_a13_2_gap_yearly.csv"), index=False)
SP.to_csv(os.path.join(OUT, "monthly_spacing.csv"), index=False)
paths = read + st.read
if any("data_holdout" in p for p in paths):
    raise SystemExit("holdout path read")
open(os.path.join(OUT, "read_paths_a2_a13_2.txt"), "w").write("\n".join(sorted(set(paths))) + "\n")
summary = {"openable_source": "out/A2_daily_open.csv", "openable_sha256": op_sha, "nonstd": list(NONSTD),
           "months": len(MON), "all_pass_months": int(MON.all_pass.sum()),
           "pass_months_by_combo": {c: int((MON[f"{c}_ratio"] >= 0.98).sum()) for c in COMBOS},
           "fail_months": MON[~MON.all_pass].month.tolist(), "first_12_month_run_start": start,
           "dev_start": dev_start, "stats_start": stats_start, "stats_days": stats_days,
           "stats_years": round(stats_days / 252, 2),
           "gap_days": {"H21": int((R.st21 == "gap").sum()), "H10": int((R.st10 == "gap").sum())},
           "gap_share": {"H21": round(float((R.st21 == "gap").mean()), 4), "H10": round(float((R.st10 == "gap").mean()), 4)},
           "no_monthly_days": int((R.st21 == "no_monthly").sum()), "schedule_crosscheck_disagreements": xc,
           "defect_ii_days_by_combo": defects, "spacing_counts": dict(zip(map(int, SP.gap_days), map(int, SP["count"]))),
           "read_paths": len(set(paths)), "holdout_paths": 0}
json.dump(summary, open(os.path.join(OUT, "a2_a13_2_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
