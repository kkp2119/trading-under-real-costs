#!/usr/bin/env python3
"""Research 17: the check17_cusip.py identity check on the existing universes 2015-2026 (Kyle 2026-11-04). Counts only;
no research 12/15 result is recomputed and no research 12/15 bar, sealed or test folder is opened.
  lists        research15/universe15_<Y>.tsv 2015-2018, research12/universe_<Y>.tsv 2019-2026
  symbol-days  list x calendar (research15/calendar15.csv, research12/calendar.csv to 2026-10-23) on the days with a
               complete_nbbo, ctm and mastm table (the downloader's draft A5 rule, table names only)
  status       the day's master (taqm_Y.mastm_<day>; master tables only) has no row for the ticker -> absent (the
               downloader writes no file); master CUSIP = selection CUSIP (X = Y - 1 last master, permno_map.csv from
               universe17.py) -> same; else different (issuer = first 6 characters)
  cross-check  2015-2018 present symbol-days vs research15/bars1s/meta/MANIFEST.tsv (development metadata in the repo)
Splits: research 15 development 2015-2020 and test 2021-2026. Daily rows with sec_desc stay in
$DATA_DIR/research17_data/selection/cusip_daily_existing.csv; results/check17_cusip_existing.json has counts and dates.
One WRDS connection (PostgreSQL client credentials), three queries per year.
  python check17_cusip_existing.py"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
R12, R15 = os.path.join(PROJ, "study12_market_maker"), os.path.join(PROJ, "study15_ml_execution")
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research17_data/selection")
KINDS = ("complete_nbbo", "ctm", "mastm")
TABLES_SQL = ("select table_name from information_schema.tables where table_schema = 'taqm_{y}' "
              "and table_name ~ '^(complete_nbbo|ctm|mastm)_[0-9]{{8}}$'")
COLS_SQL = "select column_name from information_schema.columns where table_schema = 'taqm_{y}' and table_name = 'mastm_{d}'"
SPLITS = {"development 2015-2020": ("2015", "2020"), "test 2021-2026": ("2021", "2026")}


def lists():
    for y in range(2015, 2027):
        p = os.path.join(R15, f"universe15_{y}.tsv") if y <= 2018 else os.path.join(R12, f"universe_{y}.tsv")
        yield y, pd.read_csv(p, sep="\t", dtype=str).sym.tolist()


def calendar():
    c = pd.concat([pd.read_csv(os.path.join(R15, "calendar15.csv"), dtype=str), pd.read_csv(os.path.join(R12, "calendar.csv"), dtype=str)])
    return sorted(c.date.str.replace("-", ""))


def counts(g):
    n = g.status.value_counts().to_dict()
    present = n.get("same", 0) + n.get("different", 0)
    return {"planned": int(len(g)), "absent": int(n.get("absent", 0)), "present": int(present), "same": int(n.get("same", 0)),
            "different": int(n.get("different", 0)), "different_share_of_present": n.get("different", 0) / present if present else None}


def main():
    pm = pd.read_csv(os.path.join(LOCAL, "permno_map.csv"), dtype={"sym": str, "cusip": str})
    ref = pm[pm.role == "existing"].drop_duplicates(["year", "sym"]).set_index(["year", "sym"]).cusip
    days = calendar()
    import wrds
    db = wrds.Connection()
    parts, gaps = [], {}
    try:
        for y, syms in lists():
            have = {(n.rsplit("_", 1)[0], n[-8:]) for n in db.raw_sql(TABLES_SQL.format(y=y)).table_name}
            dy = [d for d in days if d[:4] == str(y)]
            ok = [d for d in dy if all((k, d) in have for k in KINDS)]
            gaps[y] = sorted(set(dy) - set(ok))
            desc = "sec_desc" if "sec_desc" in set(db.raw_sql(COLS_SQL.format(y=y, d=ok[-1])).column_name) else "null::text"
            s = ",".join(repr(x) for x in syms)
            q = " union all ".join(f"select '{d}' as date, symbol_15 as sym, cusip, {desc} as sec_desc from taqm_{y}.mastm_{d} "
                                   f"where symbol_15 in ({s})" for d in ok)
            r = db.raw_sql(q).drop_duplicates(["date", "sym", "cusip"])
            exp = pd.DataFrame([(d, x) for d in ok for x in syms], columns=["date", "sym"])
            parts.append(exp.merge(r, on=["date", "sym"], how="left").assign(year=y))
    finally:
        db.close()
    m = pd.concat(parts, ignore_index=True)
    m.to_csv(os.path.join(LOCAL, "cusip_daily_existing.csv"), index=False)
    m["ref"] = [ref.get((y, s)) for y, s in zip(m.year, m.sym)]
    two = m.groupby(["date", "sym"]).size()
    m["status"] = ["absent" if pd.isna(c) else "same" if c == r else "different" for c, r in zip(m.cusip, m.ref)]
    m = m.sort_values("status", key=lambda v: v.map({"same": 0, "different": 1, "absent": 2})).drop_duplicates(["date", "sym"])
    out = {"symbol_days_with_two_master_rows": int((two > 1).sum()), "table_gap_days": {str(y): v for y, v in gaps.items() if v},
           "total": counts(m), "splits": {}, "by_year": {}, "different": []}
    for k, (a, b) in SPLITS.items():
        out["splits"][k] = counts(m[(m.date.str[:4] >= a) & (m.date.str[:4] <= b)])
    for y, g in m.groupby("year"):
        out["by_year"][str(y)] = counts(g)
    for (y, s), g in m.groupby(["year", "sym"]):
        d = g[g.status == "different"]
        if len(d):
            out["different"].append({"year": int(y), "sym": s, "different": int(len(d)), "first": d.date.min(), "last": d.date.max(),
                                     "same": int((g.status == "same").sum()), "absent": int((g.status == "absent").sum()),
                                     "same_last": g[g.status == "same"].date.max() if (g.status == "same").any() else None,
                                     "issuer_prefix_same": sorted({bool(c[:6] == g.ref.iloc[0][:6]) for c in d.cusip})})
    mf = os.path.join(R15, "bars1s", "meta", "MANIFEST.tsv")
    if os.path.exists(mf):
        man = pd.read_csv(mf, sep="\t", dtype=str, usecols=["sym", "date"])
        p = m[(m.status != "absent") & (m.date <= "20181231")]
        a, b = set(zip(p.sym, p.date)), set(zip(man.sym, man.date))
        out["cross_check_2015_2018"] = {"present": len(a), "research15_manifest": len(b), "only_present": len(a - b), "only_manifest": len(b - a)}
    with open(os.path.join(HERE, "results", "check17_cusip_existing.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
