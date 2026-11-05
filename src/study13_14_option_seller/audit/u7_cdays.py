"""U7 (contract_size value counts, SPX dev) and the dates of A2 'monthly in range but no candidate' days (dev only)."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import chain as ch, omdata as om
st = om.Store("dev"); read = []
cal = ch.load_calendar("dev", read=read)
NONSTD = ("m", "w")
op = pd.read_csv(os.path.join(HERE, "out", "A2_daily_open.csv"))
u7, cdays = {}, []
for y in range(1996, 2013):
    a = st.load(f"opprcd/opprcd_SPX_{y}.npz")
    for k, v in pd.Series(a["contract_size"]).value_counts(dropna=False).items():
        u7[str(k)] = u7.get(str(k), 0) + int(v)
    df = pd.DataFrame({k: a[k] for k in ("date", "exdate", "am_settlement")})
    codes = np.array(list(a["expiry_indicator__codes"]) + [None], dtype=object); c = a["expiry_indicator"].astype(int)
    df["expiry_indicator"] = codes[np.where(c < 0, len(codes) - 1, c)]
    u = df.drop_duplicates()
    s = u[ch.standard_mask(u, cal, NONSTD)][["date", "exdate"]].drop_duplicates()
    s["dte"] = ch.dte(s.date, s.exdate)
    for H, (lo, hi) in {21: (35, 65), 10: (19, 49)}.items():
        inr = set(s[(s.dte >= lo) & (s.dte <= hi)].date)
        for comb in (f"S{H}", f"G{H}"):
            o = op[(op.date // 10000 == y) & (op[comb] == 0)]
            cdays += [{"combo": comb, "date": int(d)} for d in o.date if int(d) in inr]
pd.DataFrame([{"contract_size": k, "rows": v} for k, v in u7.items()]).to_csv(os.path.join(HERE, "out", "U7_contract_size.csv"), index=False)
C = pd.DataFrame(cdays); C.to_csv(os.path.join(HERE, "out", "A2_c_days.csv"), index=False)
print("U7:", u7); print(C.to_string(index=False))
paths = read + st.read; assert not any("data_holdout" in p for p in paths)
open(os.path.join(HERE, "out", "read_paths_u7_cdays.txt"), "w").write("\n".join(sorted(set(paths))) + "\n"); print("read", len(set(paths)), "holdout 0")
