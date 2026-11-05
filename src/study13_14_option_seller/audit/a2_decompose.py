"""A2 decomposition (dev only): per day and combination, not-openable days split into (a) no SPX opprcd rows,
(b) no standard monthly in the DTE range, (c) a monthly in range but no enterable candidate. Uses g0 outputs and the
same chain rules; the proposed (iii) list from out/U10_code_proposal.csv."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import chain as ch, omdata as om
st = om.Store("dev"); read = []
cal = ch.load_calendar("dev", read=read)
prop = pd.read_csv(os.path.join(HERE, "out", "U10_code_proposal.csv"))
NONSTD = tuple(prop[prop["share_AM_and_(i)"] < 0.01].code)
op = pd.read_csv(os.path.join(HERE, "out", "A2_daily_open.csv"))
RANGE = {21: (35, 65), 10: (19, 49)}
rows = []
for y in range(1996, 2013):
    a = st.load(f"opprcd/opprcd_SPX_{y}.npz")
    df = pd.DataFrame({k: a[k] for k in ("date", "exdate", "am_settlement")})
    codes = np.array(list(a["expiry_indicator__codes"]) + [None], dtype=object); c = a["expiry_indicator"].astype(int)
    df["expiry_indicator"] = codes[np.where(c < 0, len(codes) - 1, c)]
    u = df.drop_duplicates(["date", "exdate", "am_settlement", "expiry_indicator"])
    s = u[ch.standard_mask(u, cal, NONSTD)][["date", "exdate"]].drop_duplicates()
    s["dte"] = ch.dte(s.date, s.exdate)
    have = set(np.unique(df.date))
    for H, (lo, hi) in RANGE.items():
        inr = set(s[(s.dte >= lo) & (s.dte <= hi)].date)
        for d in cal.days[(cal.days // 10000 == y) & (cal.days >= 19960104)]:
            rows.append({"date": int(d), "H": H, "opprcd": int(d) in have, "monthly_in_range": int(d) in inr})
R = pd.DataFrame(rows)
out = []
for comb in ("S21", "S10", "G21", "G10"):
    H = int(comb[1:])
    m = R[R.H == H].merge(op[["date", comb]], on="date")
    no = m[m[comb] == 0]
    out.append({"combo": comb, "days": len(m), "not_openable": len(no), "a_no_opprcd": int((~no.opprcd).sum()),
                "b_no_monthly_in_range": int((no.opprcd & ~no.monthly_in_range).sum()),
                "c_monthly_but_no_candidate": int((no.opprcd & no.monthly_in_range).sum()),
                "openable_share_given_monthly": round(float(m[m.monthly_in_range][comb].mean()), 5)})
D = pd.DataFrame(out); print(D.to_string(index=False))
D.to_csv(os.path.join(HERE, "out", "A2_decomposition.csv"), index=False)
mm = R[R.H == 21].assign(month=lambda x: x.date // 100).groupby("month").monthly_in_range.mean()
print("months where every day has an H=21 monthly in range:", int((mm == 1).sum()), "of", len(mm))
gap = R[(R.H == 21) & ~R.monthly_in_range]
print("H=21 no-monthly days by DTE gap example months:", sorted(set(gap.date // 100))[:8])
paths = read + st.read
assert not any("data_holdout" in p for p in paths)
open(os.path.join(HERE, "out", "read_paths_a2_decompose.txt"), "w").write("\n".join(sorted(set(paths))) + "\n")
print("read paths", len(set(paths)), "holdout 0")
