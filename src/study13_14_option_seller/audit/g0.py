#!/usr/bin/env python3
"""Research 13 G0 audit on development data only (SPEC spec13-v1 9 and 2.5, SPEC_AMENDMENTS A13-1; interpretations
fixed in research13/AUDIT_G0.md section 0 before any computation). SPX (secid 108105) 1996-2012.

Pass 1: U1 U2 U5 U6 U7 U9 U10 A1 A5 A6 and the proposed (iii) non-standard code list.
Pass 2: A4 with T0 and T1 (U12).  Pass 3 (chosen T): A2 (four combinations), A3', U3, U4 (q), gate-signal missing
counts, spot-forward counts. No strategy P&L, return or Sharpe is computed; gate signals are only counted as present
or missing. Every path read is logged (out/read_paths.txt) and a holdout path stops the run."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R13 = os.path.dirname(HERE)
sys.path.insert(0, R13)
import chain as ch  # noqa: E402
import omdata as om  # noqa: E402

OUT = os.path.join(HERE, "out")
YEARS = range(1996, 2013)
FIRST_DATA = 19960104
COMBOS = [("S", 21), ("S", 10), ("G", 21), ("G", 10)]
RANGE = {21: (35, 65, 45), 10: (19, 49, 30)}
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
READ = []
st = om.Store("dev")


def guard():
    paths = READ + st.read
    if any("data_holdout" in p for p in paths):
        raise SystemExit("holdout path read: stop")
    return paths


def save(name, df):
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, name), index=False)


def decode(a, col):
    codes = np.array(list(a[col + "__codes"]) + [None], dtype=object)
    c = a[col].astype(np.int64)
    return codes[np.where(c < 0, len(codes) - 1, c)]


def load_year(y):
    a = st.load(f"opprcd/opprcd_SPX_{y}.npz")
    guard()
    df = pd.DataFrame({k: a[k] for k in ("date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer",
                                         "impl_volatility", "delta", "open_interest", "optionid", "am_settlement",
                                         "contract_size")})
    df["expiry_indicator"] = decode(a, "expiry_indicator")
    df["root"] = decode(a, "root")
    df["K"] = df.strike_price / 1000.0
    return df


def dte_table(df):
    u = df[["date", "exdate"]].drop_duplicates()
    u["dte"] = ch.dte(u.date, u.exdate)
    return u


# ---------------------------------------------------------------- setup
t_start = time.time()
cal = ch.load_calendar("dev", read=READ)
_sp = [st.load(f"secprd/secprd_SPX_{y}.npz") for y in YEARS]
closes = pd.concat([pd.Series(a["close"], index=a["date"]) for a in _sp])
z = st.load("zerocd/zerocd_dev.npz")
guard()
DAYS = cal.days[cal.days >= FIRST_DATA]

# U11: units from magnitude only
zr = z["rate"]
u11 = {"rows": int(len(zr)), "min": round(float(np.nanmin(zr)), 2), "median": round(float(np.nanmedian(zr)), 2),
       "max": round(float(np.nanmax(zr)), 2)}
u11["unit"] = "percent" if 0.01 < u11["median"] < 25 and u11["max"] < 25 else "UNDETERMINED"
if u11["unit"] != "percent":
    raise SystemExit(f"U11 unit undetermined: {u11}")
rates = ch.Rates(z, cal, scale=0.01)
prev_curve_days = [int(d) for d in cal.days if rates.curve_date(d) != d]

# ---------------------------------------------------------------- pass 1
u1, u2, u5, u6, u7, u9, u10, a1, a5, a6, mixed = [], [], [], [], {}, [], [], [], [], [], []
for y in YEARS:
    df = load_year(y)
    dt_ = dte_table(df)
    # U1: strike/1000 against the close at the parity strike of the expiry nearest 30 DTE (>= 5)
    near = dt_[dt_.dte >= 5].assign(gap=lambda x: (x.dte - 30).abs()).sort_values(["date", "gap", "exdate"]) \
        .groupby("date").head(1)[["date", "exdate"]]
    sub = df.merge(near, on=["date", "exdate"])
    v = sub[ch.valid_quote(sub.best_bid, sub.best_offer, ch.TAU_S)].assign(mid=lambda x: (x.best_bid + x.best_offer) / 2)
    m = v[v.cp_flag == 1][["date", "strike_price", "mid"]].merge(v[v.cp_flag == -1][["date", "strike_price", "mid"]],
                                                                 on=["date", "strike_price"], suffixes=("_c", "_p"))
    m = m.assign(a=(m.mid_c - m.mid_p).abs()).sort_values(["date", "a", "strike_price"]).groupby("date").head(1)
    ratio = m.strike_price.to_numpy() / 1000.0 / closes.reindex(m.date).to_numpy()
    u1.append({"year": y, "days": len(ratio), "median_ratio": round(float(np.median(ratio)), 4),
               "p01": round(float(np.percentile(ratio, 1)), 4), "p99": round(float(np.percentile(ratio, 99)), 4)})
    d = df.delta
    u2.append({"year": y, "call_nonnull": int((df.cp_flag == 1).sum() - d[df.cp_flag == 1].isna().sum()),
               "call_pos": int(((df.cp_flag == 1) & (d > 0)).sum()), "put_nonnull": int(d[df.cp_flag == -1].notna().sum()),
               "put_neg": int(((df.cp_flag == -1) & (d < 0)).sum())})
    rc = df.root.value_counts(dropna=False)
    u5.append({"year": y, "distinct_roots": int(df.root.nunique()), "rows_root_null": int(df.root.isna().sum()),
               "top_roots": "; ".join(f"{k}:{v}" for k, v in rc.head(5).items())})
    g = df.groupby([df.root.fillna("<NULL>"), df.am_settlement]).size().reset_index(name="rows")
    u6.append(g.assign(year=y))
    for k, v in df.contract_size.value_counts(dropna=False).items():
        u7[str(k)] = u7.get(str(k), 0) + int(v)
    ex = pd.Series(np.unique(df.exdate))
    wd = ex.map(lambda e: ch.ymd(e).weekday())
    kinds = ex.map(lambda e: ch.standard_kind(e, cal))
    row9 = {"year": y, "distinct_exdates": len(ex)}
    row9.update({f"exdates_{WD[i]}": int((wd == i).sum()) for i in range(7)})
    row9.update({"kind1": int((kinds == 1).sum()), "kind2": int((kinds == 2).sum()), "kind3": int((kinds == 3).sum())})
    u9.append(row9)
    kmap = dict(zip(ex, kinds))
    am = df[df.am_settlement == 1][["date", "exdate"]].drop_duplicates()
    am["k"] = am.exdate.map(kmap)
    am["month"] = am.exdate // 100
    ks = am[am.k > 0].groupby(["date", "month"]).k.agg(lambda s: tuple(sorted(set(s))))
    for (dd, mo), t in ks.items():
        if len(t) > 1:
            mixed.append({"date": dd, "month": mo, "kinds": "".join(map(str, t))})
    wmap = {e: WD[ch.ymd(e).weekday()] for e in ex}
    x = df.assign(wd=df.exdate.map(wmap), kind=df.exdate.map(kmap),
                  ei=df.expiry_indicator.fillna("<NULL>"))
    u10.append(x.groupby(["ei", "wd", "am_settlement", "kind"]).size().reset_index(name="rows").assign(year=y))
    days_y = cal.days[(cal.days // 10000) == y]
    have_opt = set(np.unique(df.date))
    have_close = set(closes.index[np.isfinite(closes.to_numpy())])
    a1.append({"year": y, "calendar_days": len(days_y), "opprcd_days": sum(int(d) in have_opt for d in days_y),
               "close_days": sum(int(d) in have_close for d in days_y),
               "both": sum(int(d) in have_opt and int(d) in have_close for d in days_y)})
    a5.append({"year": y, "rows": len(df), "distinct_roots": int(df.root.nunique()), "distinct_exdates": len(ex),
               "standard_exdates_kind>0": int((kinds > 0).sum())})
    a6.append({"year": y, "optionid_date_duplicates": int(df.duplicated(["optionid", "date"]).sum())})
    print(f"pass1 {y}: rows {len(df)}", flush=True)

U10 = pd.concat(u10)
codes = sorted(set(U10.ei) - {"<NULL>"})
prop = []
for c in codes:
    r = U10[U10.ei == c]
    std_am = int(r[(r.kind > 0) & (r.am_settlement == 1)].rows.sum())
    prop.append({"code": c, "rows": int(r.rows.sum()), "rows_AM_and_(i)": std_am,
                 "share_AM_and_(i)": round(std_am / max(int(r.rows.sum()), 1), 4),
                 "weekdays": ",".join(sorted(set(r.wd))), "am_values": ",".join(map(str, sorted(set(r.am_settlement))))})
PROP = pd.DataFrame(prop)
NONSTD = tuple(PROP[PROP["share_AM_and_(i)"] < 0.01].code) if len(PROP) else ()
save("U1_strike_scale.csv", pd.DataFrame(u1)); save("U2_delta_sign.csv", pd.DataFrame(u2))
save("U5_roots.csv", pd.DataFrame(u5)); save("U6_am_x_root.csv", pd.concat(u6))
save("U9_exdate_weekdays.csv", pd.DataFrame(u9)); save("U9_mixed_kinds_same_month.csv", pd.DataFrame(mixed, columns=["date", "month", "kinds"]))
save("U10_crosstab.csv", U10); save("U10_code_proposal.csv", PROP)
save("A1.csv", pd.DataFrame(a1)); save("A5.csv", pd.DataFrame(a5)); save("A6.csv", pd.DataFrame(a6))
U1med = float(pd.DataFrame(u1).median_ratio.median())
if not 0.98 <= U1med <= 1.02:
    raise SystemExit(f"U1 not confirmed (median ratio {U1med}); stop")


# ---------------------------------------------------------------- helpers for passes 2 and 3
def std_rows(df):
    s = df[ch.standard_mask(df, cal, NONSTD)]
    t = dte_table(s)
    return s.merge(t, on=["date", "exdate"]), t


def parity_table(c25):
    """Per (date, exdate): parity points ordered by |diff| then K (A13-1), all ranks kept."""
    v = c25[ch.valid_quote(c25.best_bid, c25.best_offer, ch.TAU_S)].assign(mid=lambda x: (x.best_bid + x.best_offer) / 2)
    m = v[v.cp_flag == 1][["date", "exdate", "K", "mid"]].merge(v[v.cp_flag == -1][["date", "exdate", "K", "mid"]],
                                                                on=["date", "exdate", "K"], suffixes=("_c", "_p"))
    m["diff"] = m.mid_c - m.mid_p
    m["a"] = m["diff"].abs()
    m = m.sort_values(["date", "exdate", "a", "K"], kind="mergesort")
    m["rank"] = m.groupby(["date", "exdate"]).cumcount()
    return m


def t_of(pairs, basis):
    if basis == "T0":
        return pairs.dte.to_numpy() / 365.0
    st_ = np.array([cal.settlement(e) for e in pairs.exdate])
    return ch.dte(pairs.date, st_) / 365.0


def forwards(pairs, ptab, basis):
    """pairs: (date, exdate, dte). Adds T, r, S, F, src."""
    p = pairs.copy()
    p["T"] = t_of(p, basis)
    p["r"] = [float(rates.rate(d, T * 365.0)) for d, T in zip(p.date, p["T"])]
    p["S"] = closes.reindex(p.date).to_numpy()
    first = ptab[ptab["rank"] == 0][["date", "exdate", "K", "diff"]]
    p = p.merge(first, on=["date", "exdate"], how="left")
    par = p.K.notna()
    p["F"] = np.where(par, p.K + np.exp(p.r * p["T"]) * p["diff"], p.S * np.exp(p.r * p["T"]))
    p["src"] = np.where(par, "parity", "spot")
    return p.drop(columns=["K", "diff"])


# ---------------------------------------------------------------- pass 2: A4 (T0 vs T1)
a4 = []
for y in YEARS:
    df = load_year(y)
    s, t = std_rows(df)
    s = s[(s.dte >= 5) & (s.dte <= 65)]
    c25 = ch.dedupe(s, ch.TAU_S)
    pt = parity_table(c25)
    pairs = c25[["date", "exdate", "dte"]].drop_duplicates()
    smp = c25[ch.valid_quote(c25.best_bid, c25.best_offer, ch.TAU_S) & ch.iv_valid(c25.impl_volatility) & c25.delta.notna()]
    out = smp[["date", "exdate", "K", "cp_flag", "impl_volatility", "delta"]].copy()
    for b in ("T0", "T1"):
        f = forwards(pairs, pt, b)
        mm = out.merge(f[["date", "exdate", "T", "r", "S", "F", "src"]], on=["date", "exdate"], how="left")
        assert len(mm) == len(out) and mm.F.notna().all()
        _, bd = ch.bs(mm.F, mm.K, mm["T"], mm.impl_volatility, mm.r, mm.S, mm.cp_flag)
        out[f"ad_{b}"] = np.abs(bd - mm.delta.to_numpy())
        out[f"src_{b}"] = mm.src.to_numpy()
    out["wd"] = out.exdate.map({e: WD[ch.ymd(e).weekday()] for e in np.unique(out.exdate)})
    out["year"] = y
    a4.append(out[["year", "wd", "cp_flag", "ad_T0", "ad_T1", "src_T0"]])
    print(f"pass2 {y}: A4 sample {len(out)}", flush=True)
A4 = pd.concat(a4)
sat = A4[A4.wd == "Sat"]
med_sat = {b: float(sat[f"ad_{b}"].median()) if len(sat) else float("nan") for b in ("T0", "T1")}
T_BASIS = "T1" if len(sat) and med_sat["T1"] < med_sat["T0"] else "T0"
a4_tab = []
for key, g in [("all", A4)] + [(f"wd={w}", g) for w, g in A4.groupby("wd")] + [(f"year={y}", g) for y, g in A4.groupby("year")]:
    a4_tab.append({"group": key, "n": len(g), "median_absdiff_T0": round(float(g.ad_T0.median()), 5),
                   "median_absdiff_T1": round(float(g.ad_T1.median()), 5),
                   "spot_forward_share": round(float((g.src_T0 == "spot").mean()), 4)})
A4T = pd.DataFrame(a4_tab)
save("A4.csv", A4T)
A4_med = float(A4[f"ad_{T_BASIS}"].median())

# ---------------------------------------------------------------- pass 3: A2, A3', U3, U4, counts
open_rows, a3_vals, q_rows, u3_rows, fwd_counts, gate_rows = [], [], [], [], [], []
a3_excl = {"no_monthly": {}, "lt2": {}}
for y in YEARS:
    df = load_year(y)
    s, t = std_rows(df)
    s = s[(s.dte >= 5) & (s.dte <= 65)]
    c25, c50 = ch.dedupe(s, ch.TAU_S), ch.dedupe(s, ch.TAU_G)
    pt = parity_table(c25)
    pairs = c25[["date", "exdate", "dte"]].drop_duplicates()
    F = forwards(pairs, pt, T_BASIS)
    days_y = [int(d) for d in DAYS if d // 10000 == y]
    chosen = {}
    for H, (lo, hi, tg) in RANGE.items():
        r = t[(t.dte >= lo) & (t.dte <= hi)].assign(g=lambda x: (x.dte - tg).abs()).sort_values(["date", "g", "exdate"])
        chosen[H] = r.groupby("date").head(1)[["date", "exdate"]]
        fc = F.merge(chosen[H], on=["date", "exdate"])
        fwd_counts.append({"year": y, "H": H, "chosen_pairs": len(fc), "spot_forward": int((fc.src == "spot").sum())})
    # S structure
    res = {d: {c: False for c in COMBOS} for d in days_y}
    for H in (21, 10):
        g = c25.merge(chosen[H], on=["date", "exdate"])
        ok = ch.valid_quote(g.best_bid, g.best_offer, ch.TAU_S) & ch.iv_valid(g.impl_volatility)
        g = g.assign(ok=ok)
        legs = g.pivot_table(index=["date", "exdate", "K"], columns="cp_flag", values="ok", aggfunc="max")
        legs = legs.dropna(subset=[c for c in (1, -1) if c in legs.columns]) if {1, -1} <= set(legs.columns) else legs.iloc[:0]
        legs = legs.reset_index().merge(F[["date", "exdate", "F"]], on=["date", "exdate"])
        legs["dist"] = (legs.K - legs.F).abs()
        legs = legs.sort_values(["date", "dist", "K"]).groupby("date").head(3)
        legs["both"] = legs[1].astype(bool) & legs[-1].astype(bool)
        for d, v in legs.groupby("date").both.any().items():
            if int(d) in res:
                res[int(d)][("S", H)] = bool(v)
        # G structure
        g5 = c50.merge(chosen[H], on=["date", "exdate"]).merge(F[["date", "exdate", "T", "r", "S", "F"]], on=["date", "exdate"])
        _, bd = ch.bs(g5.F, g5.K, g5["T"], g5.impl_volatility, g5.r, g5.S, g5.cp_flag)
        dl = np.where(g5.delta.notna(), g5.delta, np.where(ch.iv_valid(g5.impl_volatility), bd, np.nan))
        g5 = g5.assign(dl=dl, ok=ch.valid_quote(g5.best_bid, g5.best_offer, ch.TAU_G) & ch.iv_valid(g5.impl_volatility))
        g5 = g5[np.isfinite(g5.dl)]
        found = {}
        for cp, tgt in ((1, 0.16), (-1, -0.16)):
            leg = g5[g5.cp_flag == cp].assign(dist=lambda x: (x.dl - tgt).abs(), otm=lambda x: -x.K if cp == 1 else x.K)
            leg = leg.sort_values(["date", "dist", "otm"]).groupby("date").head(3)
            found[cp] = leg.groupby("date").ok.any()
        both = (found[1].reindex(found[1].index.union(found[-1].index), fill_value=False)
                & found[-1].reindex(found[1].index.union(found[-1].index), fill_value=False))
        for d, v in both.items():
            if int(d) in res:
                res[int(d)][("G", H)] = bool(v)
    for d, v in res.items():
        open_rows.append({"date": d, **{f"{a}{h}": int(v[(a, h)]) for a, h in COMBOS}})
    # A3' and q (H=21 chosen expiry)
    ch21 = chosen[21]
    pt21 = pt.merge(ch21, on=["date", "exdate"])
    has21 = set(ch21.date)
    a3_excl["no_monthly"][y] = sum(d not in has21 for d in days_y)
    f21 = F.merge(ch21, on=["date", "exdate"])
    top = pt21[pt21["rank"] < 3].merge(f21[["date", "exdate", "T", "r"]], on=["date", "exdate"])
    top["Fk"] = top.K + np.exp(top.r * top["T"]) * top["diff"]
    agg = top.groupby("date").Fk.agg(["count", "max", "min", "median"])
    a3_excl["lt2"][y] = int((agg["count"] < 2).sum()) + sum(d in has21 and d not in agg.index for d in days_y)
    ok3 = agg[agg["count"] >= 2]
    a3_vals.append(pd.DataFrame({"date": ok3.index, "year": y, "a3p": ((ok3["max"] - ok3["min"]) / ok3["median"]).to_numpy()}))
    par = f21[f21.src == "parity"]
    q_rows.append(pd.DataFrame({"year": y, "q": (par.r - np.log(par.F / par.S) / par["T"]).to_numpy()}))
    # U3: vsurfd 30-day delta-50 call impl_strike vs the parity forward of the monthly nearest 30 DTE
    vs = st.load(f"vsurfd/vsurfd_SPX_{y}.npz")
    guard()
    vd = pd.DataFrame({k: vs[k] for k in ("date", "days", "delta", "cp_flag", "impl_volatility", "impl_strike")})
    n30 = t[t.dte >= 5].assign(g=lambda x: (x.dte - 30).abs()).sort_values(["date", "g", "exdate"]).groupby("date").head(1)
    f30 = F.merge(n30[["date", "exdate"]], on=["date", "exdate"])
    k30 = vd[(vd.days == 30) & (vd.delta == 50) & (vd.cp_flag == 1)][["date", "impl_strike"]].merge(f30[["date", "F", "src"]], on="date")
    k30 = k30[k30.src == "parity"]
    u3_rows.append(pd.DataFrame({"year": y, "rel": (k30.impl_strike / k30.F - 1).to_numpy()}))
    # gate signals: present or missing only
    iv = vd[ch.iv_valid(vd.impl_volatility) & (vd.delta == 50) & (vd.cp_flag == 1)]
    has30, has91 = set(iv[iv.days == 30].date), set(iv[iv.days == 91].date)
    gate_rows.append({"year": y, "days": len(days_y), "iv30_missing": sum(d not in has30 for d in days_y),
                      "iv91_missing": sum(d not in has91 for d in days_y)})
    print(f"pass3 {y}: days {len(days_y)}", flush=True)

OPEN = pd.DataFrame(open_rows).sort_values("date")
OPEN["month"] = OPEN.date // 100
MON = OPEN.groupby("month")[[f"{a}{h}" for a, h in COMBOS]].mean().reset_index()
MON["all_pass"] = (MON[[f"{a}{h}" for a, h in COMBOS]] >= 0.98).all(axis=1)
YR = OPEN.assign(year=OPEN.date // 10000).groupby("year")[[f"{a}{h}" for a, h in COMBOS]].mean().reset_index()
save("A2_daily_open.csv", OPEN); save("A2_monthly.csv", MON); save("A2_yearly.csv", YR)
months = MON.month.tolist()
passes = MON.all_pass.tolist()
start_month = None
for i in range(len(months)):
    if all(passes[i:i + 12]) and len(passes[i:i + 12]) == 12:
        start_month = months[i]
        break
dev_start = FIRST_DATA if start_month == months[0] else (int(min(d for d in DAYS if d // 100 == start_month)) if start_month else None)
idx = int(np.searchsorted(cal.days, dev_start)) if dev_start else None
stats_start = int(cal.days[idx + 43]) if idx is not None else None
stats_days = int((cal.days >= stats_start).sum()) if stats_start else 0

A3P = pd.concat(a3_vals)
A3Y = A3P.groupby("year").a3p.agg(["count", "median"]).reset_index()
A3Y["no_monthly_days"] = A3Y.year.map(a3_excl["no_monthly"])
A3Y["lt2_days"] = A3Y.year.map(a3_excl["lt2"])
A3Y["flag_gt_0.1%"] = A3Y["median"] > 0.001
A3Y["gt_0.5%"] = A3Y["median"] > 0.005
save("A3prime_yearly.csv", A3Y)
Q = pd.concat(q_rows)
QY = Q.groupby("year").q.describe(percentiles=[0.1, 0.5, 0.9])[["count", "10%", "50%", "90%"]].reset_index()
save("U4_q_yearly.csv", QY)
U3Y = pd.concat(u3_rows).groupby("year").rel.describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]].reset_index()
save("U3_yearly.csv", U3Y)
# RV21 missing: t-21..t closes all present (22 closes), counted on DAYS
cl = closes.reindex(cal.days)
ok22 = pd.Series(np.isfinite(cl.to_numpy()).astype(int), index=cal.days).rolling(22).sum().eq(22)
GATE = pd.DataFrame(gate_rows)
GATE["rv21_missing"] = GATE.year.map(lambda y: int((~ok22[[d for d in DAYS if d // 10000 == y]]).sum()))
save("gate_signal_missing.csv", GATE)
FC = pd.DataFrame(fwd_counts)
save("forward_spot_counts.csv", FC)
excluded_iii = int(sum(((U10.kind > 0) & (U10.am_settlement == 1) & U10.ei.isin(list(NONSTD))) * U10.rows))

paths = guard()
with open(os.path.join(OUT, "read_paths.txt"), "w") as f:
    f.write("\n".join(sorted(set(paths))) + "\n")
summary = {
    "U11": u11, "prev_curve_days": prev_curve_days, "U1_median_ratio": U1med, "nonstd_proposal": list(NONSTD),
    "rows_excluded_by_iii_proposal_AM_(i)": excluded_iii, "A4_T_basis": T_BASIS, "A4_sat_median_T0_T1": med_sat,
    "A4_median_chosen": A4_med, "A4_pass": A4_med <= 0.01, "A1_ratio_all": None,
    "A2_fail_months": MON[~MON.all_pass].month.tolist(), "A2_start_month": start_month, "dev_start": dev_start,
    "stats_start": stats_start, "stats_days": stats_days, "stats_years": round(stats_days / 252, 2),
    "A3p_years_gt_0.1%": A3Y[A3Y["flag_gt_0.1%"]].year.tolist(), "A3p_years_gt_0.5%": A3Y[A3Y["gt_0.5%"]].year.tolist(),
    "A3p_stop": int(A3Y["gt_0.5%"].sum()) >= 3, "read_paths": len(set(paths)),
    "holdout_paths": sum("data_holdout" in p for p in paths), "seconds": round(time.time() - t_start, 1)}
A1 = pd.DataFrame(a1)
summary["A1_ratio_all"] = round(float(A1.both.sum() / A1.calendar_days.sum()), 5)
summary["A1_ratio_from_first_data"] = round(float((A1.both.sum()) / (A1.calendar_days.sum() - 2)), 5)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
print(json.dumps(summary, indent=1, default=str))
