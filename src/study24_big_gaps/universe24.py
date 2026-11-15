#!/usr/bin/env python3
"""Research 24 step 1: yearly pairs of the two families (SPEC24 1; the points the plan leaves open are listed in
SPEC24_QUESTIONS.md). No price difference, signal or P&L is computed here.

For each trading year Y = 2015..2026 (screen year X = Y - 1, year-X information only), with the screens of the earlier
studies imported and only their dollar-volume floors changed:
  family E2  research 21's family E (universe21.screen, CIZ files): ETFs with at least 200 trading days in X, median close
             in [5, 2000] and a median daily dollar volume of at least 5,000,000 (research 21: 50,000,000); a pair has a
             correlation of the daily returns >= 0.995 and std(A) / std(B) in [0.95, 1.05]
  family D2  research 20's pairs (universe20.screen: two common-share securities of one issuer, correlation >= 0.95, at
             least 120 days, median close in [5, 2000]; legacy CRSP files to X = 2024, CIZ files for X = 2025) with a
             median daily dollar volume of at least 1,000,000 for both legs (research 20: 5,000,000) and no cut to 4 pairs
  both       leg A is the leg with the larger median dollar volume
             excluded  pairs with a PERMNO of research 12-23: research 12 (stocks and control ETFs), 15, 17, 18 and the
                       legs of research 20 (universe21.existing), the legs of research 21, and research 22's universe C
                       and research 23's universe D (their 20 slots a year, mapped on their own screen year); as a
                       safeguard, as before, also pairs with one of their tickers
             identity  a leg is a PERMNO; its CUSIPs in year Y come from the CRSP name history (universe20.cusip_sets); a
                       pair needs exactly one TAQ master symbol per leg on Y's first trading day
             order     the two families together, by the smaller of the two median dollar volumes, largest first
             selected  in that order, a security in one pair of the year only, the first 30; the lists for N_p = 30, 20
                       and 12 are the first N_p of that one order (n24.py writes universe24_<Y>.tsv for the N_p of the rule)
  counts     for the selected legs, the session rows of the quote and trade tables on Y's first trading day (counts only;
             n24.py turns them into an estimate of the collection time and size)
One WRDS connection (the WRDS credentials file). Counts, tickers and rounded statistics go to the log, universe24_order_<Y>.tsv and
results/universe24_summary.json; CRSP identifiers, CUSIPs and the full pair tables stay in
$DATA_DIR/research24_data/selection/. Nothing is overwritten.
  python universe24.py --go"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for r in ("study18_forecast", "study20_pairs", "study21_more_pairs"):
    sys.path.insert(0, os.path.join(PROJ, r))
import universe18 as U18  # noqa: E402  (permno_map and research 12's table_days; its main is never called)
import universe20 as U20  # noqa: E402  (screen of family D2, cusip_sets, TAQ_SQL, CTA_SQL; its main is never called)
import universe21 as U21  # noqa: E402  (screen of family E2, existing; its main is never called)

LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research24_data/selection")
LEGS21 = os.path.join(os.environ.get("DATA_DIR", "data"), "research21_data/selection/legs21.csv")
YEARS = range(2015, 2027)
N_MAX, NS = 30, (30, 20, 12)
E2_MIN_DV, D2_MIN_DV = 5_000_000.0, 1_000_000.0
LEGACY_END = 2024
CNT = ("select sym_root, sym_suffix, count(*) as n from taqm_{y}.{t}_{d} where time_m >= '09:30:00' and time_m < '16:00:00' "
       "and sym_root in ({r}) group by 1, 2")
COLS = ["family", "permno_a", "permno_b", "ticker_a", "ticker_b", "comnam_a", "comnam_b", "ncusip_a", "ncusip_b", "days_a", "days_b",
        "days_both", "corr", "sd_ratio", "med_prc_a", "med_prc_b", "med_dv_a", "med_dv_b"]


def outputs():
    return ([os.path.join(HERE, f"universe24_order_{y}.tsv") for y in YEARS]
            + [os.path.join(HERE, "results", "universe24_summary.json"), os.path.join(LOCAL, "pairs_all_years.csv"), os.path.join(LOCAL, "legs24.csv")])


def existing(db, crsp_end):
    """(rows, PERMNOs, tickers, counts by source) of every list of research 12-23."""
    ex, exm, permnos, _, _ = U21.existing(db, crsp_end)                       # research 12 (stocks, ETFs), 15, 17, 18, 20
    l21 = pd.read_csv(LEGS21, dtype=str)
    ex = pd.concat([ex, l21[["year", "sym"]].astype({"year": int}).assign(source="r21")], ignore_index=True)
    permnos = set(permnos) | set(l21.permno.astype(int))
    for k, src in ((22, "r22"), (23, "r23")):                                # universe C and universe D, 20 slots a year
        u = pd.DataFrame([(y, s, src) for y in range(2021, 2027)
                          for s in pd.read_csv(os.path.join(PROJ, f"research{k}", f"universe{k}_{y}.tsv"), sep="\t", dtype=str).sym],
                         columns=["year", "sym", "source"])
        m = pd.concat([U18.permno_map(db, y - 1, u.sym[u.year == y], crsp_end).assign(year=y) for y in range(2021, 2027)])
        if len(u) != 120 or (m.groupby(["year", "sym"]).permno.agg(lambda v: v.dropna().nunique()) != 1).any():
            raise SystemExit(f"research {k}: a symbol-year without exactly one PERMNO")
        ex = pd.concat([ex, u], ignore_index=True)
        permnos |= set(m.permno.dropna().astype(int))
        exm = pd.concat([exm, m])
    return ex, exm, permnos, set(ex.sym)


def candidates(db, x):
    """(passing pairs of both families with the common columns, info per family) of screen year x."""
    U21.MIN_DV = E2_MIN_DV                                                    # the floor of family E2 (the function reads it when called)
    pe, ie = U21.screen(db, "E", x, "v2")
    src = "legacy" if x <= LEGACY_END else "v2"
    U20.MIN_DV = D2_MIN_DV                                                    # the floor of family D2
    pdd, idd = U20.screen(db, x, src)
    e = pe[pe.passing].assign(family="E2")
    d = pdd[pdd.passing].assign(family="D2", comnam_a=lambda t: t.comnam, comnam_b=lambda t: t.comnam, sd_ratio=np.nan) if len(pdd) else pd.DataFrame(columns=COLS)
    c = pd.concat([e[COLS], d[COLS]], ignore_index=True)
    c["key"] = c.med_dv_b
    info = {"E2": {**ie, "min_dv": E2_MIN_DV, "source": "v2"},
            "D2": {**idd, "min_dv": D2_MIN_DV, "source": src, "passing": int(len(d)),
                   "not_passing": {k: int((~pdd[k]).sum()) for k in ("ok_days", "ok_corr", "ok_price", "ok_dv")} if len(pdd) else {}}}
    return c, info


def split_symbol(s):
    """(sym_root, sym_suffix) candidates of a master symbol: 'LEN B' -> (LEN, B); a five-letter symbol also as 4 + 1."""
    if " " in s:
        a, b = s.split(" ", 1)
        return [(a, b)]
    return [(s, "")] + ([(s[:4], s[4:])] if len(s) == 5 else [])


def first_day_counts(db, y, d, syms):
    """{master symbol: (quote rows, trade rows)} in the session of day d (counts only; None when the table is missing)."""
    roots = sorted({r for s in syms for r, _ in split_symbol(s)})
    res = {}
    for kind in ("complete_nbbo", "ctm"):
        if d not in set(U18.UR.table_days(db, y, kind)) or not roots:
            res[kind] = None
            continue
        t = db.raw_sql(CNT.format(y=y, t=kind, d=d, r=",".join(repr(r) for r in roots)))
        res[kind] = {(a, b or ""): int(n) for a, b, n in zip(t.sym_root, t.sym_suffix.fillna(""), t.n)}
    out = {}
    for s in syms:
        v = []
        for kind in ("complete_nbbo", "ctm"):
            if res[kind] is None:
                v.append(None)
            else:
                v.append(next((res[kind][k] for k in split_symbol(s) if k in res[kind]), 0))
        out[s] = tuple(v)
    return out


def main(log):
    done = [os.path.relpath(p, HERE) for p in outputs() if os.path.exists(p)]
    if done:
        raise SystemExit(f"outputs exist (nothing is overwritten): {done}")
    import wrds
    db = wrds.Connection()
    out = {"n_max": N_MAX, "min_dv": {"E2": E2_MIN_DV, "D2": D2_MIN_DV}, "years": {}}
    keep, legs = [], []
    try:
        crsp_end = str(db.raw_sql("select max(nameenddt) as e from crsp.stocknames").e.iloc[0])[:10]
        v2_end = str(db.raw_sql("select max(nameenddt) as e from crsp.stocknames_v2").e.iloc[0])[:10]
        out["crsp_end"] = {"legacy": crsp_end, "v2": v2_end}
        ex, exm, ex_permno, ex_tickers = existing(db, crsp_end)
        out["existing"] = {"rows": int(len(ex)), "tickers": int(ex.sym.nunique()), "permnos": len(ex_permno),
                           "tickers_by_source": {k: int(g.sym.nunique()) for k, g in ex.groupby("source")}}
        log(f"existing lists of research 12-23: {len(ex)} rows, {ex.sym.nunique()} tickers, {len(ex_permno)} PERMNOs; {out['existing']['tickers_by_source']}")
        for y in YEARS:
            x = y - 1
            d1 = U18.UR.table_days(db, y, "ctm")[0]
            c, info = candidates(db, x)
            perm = sorted(set(c.permno_a) | set(c.permno_b))
            cs = U20.cusip_sets(db, perm, y, {**dict(zip(c.permno_a, c.ncusip_a)), **dict(zip(c.permno_b, c.ncusip_b))}, v2_end)
            allc = sorted({v for s in cs.values() for v in s})
            has_cta = int(db.raw_sql(U20.CTA_SQL.format(y=y, d=d1)).n.iloc[0]) > 0
            m = (db.raw_sql(U20.TAQ_SQL.format(y=y, d=d1, c=",".join(repr(v) for v in allc), cta="cta_symbol" if has_cta else "null"))
                 if allc else pd.DataFrame(columns=["symbol_15", "cta_symbol", "cusip"]))
            m["c8"] = m.cusip.astype(str).str[:8]
            m["shown"] = [t if isinstance(t, str) and t else s.replace(" ", ".") for s, t in zip(m.symbol_15, m.cta_symbol)]
            sym_of = {}
            for pn in perm:
                hit = sorted(set(m.symbol_15[m.c8.isin(cs[int(pn)])]))
                if len(hit) == 1:
                    sym_of[int(pn)] = (hit[0], m.shown[m.symbol_15 == hit[0]].iloc[0])
            for leg in "ab":
                c[f"taq_{leg}"] = [sym_of.get(int(p), (None, None))[0] for p in c[f"permno_{leg}"]]
                c[f"sym_{leg}"] = [sym_of.get(int(p), (None, None))[1] for p in c[f"permno_{leg}"]]
            c["hit_permno"] = [a in ex_permno or b in ex_permno for a, b in zip(c.permno_a, c.permno_b)]
            c["hit_ticker"] = [bool({a, b, ta, tb} & ex_tickers) and not h for a, b, ta, tb, h in zip(c.sym_a, c.sym_b, c.ticker_a, c.ticker_b, c.hit_permno)]
            c["no_taq"] = c.taq_a.isna() | c.taq_b.isna()
            c = c.sort_values(["key", "permno_a", "permno_b"], ascending=[False, True, True]).reset_index(drop=True)
            used, pick = set(), []
            for i, r in c.iterrows():
                if not (r.hit_permno or r.hit_ticker or r.no_taq) and r.permno_a not in used and r.permno_b not in used and len(pick) < N_MAX:
                    pick.append(i)
                    used |= {r.permno_a, r.permno_b}
            c["rank"] = [pick.index(i) + 1 if i in pick else 0 for i in c.index]
            c["usable"] = ~(c.hit_permno | c.hit_ticker | c.no_taq)
            sel = c[c["rank"] > 0].sort_values("rank")
            cnt = first_day_counts(db, y, d1, sorted(set(sel.taq_a) | set(sel.taq_b)))
            o = sel[["rank", "family"]].copy()
            o.insert(0, "year", y)
            o["sym_a"], o["sym_b"], o["symbol_15_a"], o["symbol_15_b"] = sel.sym_a, sel.sym_b, sel.taq_a, sel.taq_b
            o["corr"], o["med_dv_musd_a"], o["med_dv_musd_b"] = sel["corr"].round(4), (sel.med_dv_a / 1e6).round(2), (sel.med_dv_b / 1e6).round(2)
            o["med_close_a"], o["med_close_b"] = sel.med_prc_a.round(2), sel.med_prc_b.round(2)
            for leg in "ab":
                o[f"quotes_first_day_{leg}"] = [cnt[s][0] for s in sel[f"taq_{leg}"]]
                o[f"trades_first_day_{leg}"] = [cnt[s][1] for s in sel[f"taq_{leg}"]]
            o.insert(3, "first_day", d1)
            o.to_csv(os.path.join(HERE, f"universe24_order_{y}.tsv"), sep="\t", index=False)
            for r in sel.itertuples():
                for leg, t, s, pn in (("A", r.taq_a, r.sym_a, r.permno_a), ("B", r.taq_b, r.sym_b, r.permno_b)):
                    cus = [k for k in cs[int(pn)] if len(k) == 8]
                    if not cus:
                        raise SystemExit(f"{y} {s}: no CUSIP")
                    legs.append({"year": y, "family": r.family, "pair": f"P{r.rank:02d}", "rank": r.rank, "leg": leg, "sym": s, "symbol_15": t,
                                 "permno": pn, "cusips": "|".join(cus)})
            syms = list(sel.sym_a) + list(sel.sym_b)
            if len(set(syms)) != len(syms) or any(s is None for s in syms):
                raise SystemExit(f"{y}: a symbol is missing or used twice")
            keep.append(c.assign(year=y))
            perm_sel = set(sel.permno_a) | set(sel.permno_b)
            fam = {}
            for f in ("E2", "D2"):
                g = c[c.family == f]
                fam[f] = {**info[f], "passing_pairs": int(len(g)), "excluded_existing_permno": int(g.hit_permno.sum()),
                          "excluded_existing_ticker_only": int(g.hit_ticker.sum()), "without_taq_symbol": int(g.no_taq.sum()),
                          "usable_pairs": int(g.usable.sum()), "selected": {str(n): int(((g["rank"] > 0) & (g["rank"] <= n)).sum()) for n in NS}}
            out["years"][y] = {"screen_year": x, "first_day": d1, "families": fam, "usable_pairs": int(c.usable.sum()), "selected": int(len(sel)),
                               "pairs": [{"rank": int(r["rank"]), "family": r.family, "pair": f"{r.sym_a}/{r.sym_b}", "corr": round(float(r["corr"]), 4),
                                          "med_dv_musd": [round(r.med_dv_a / 1e6, 2), round(r.med_dv_b / 1e6, 2)],
                                          "med_close": [round(r.med_prc_a, 2), round(r.med_prc_b, 2)]} for _, r in sel.iterrows()],
                               "overlap_permno": len(perm_sel & ex_permno), "overlap_ticker": len(set(syms) & ex_tickers),
                               "legs_with_two_cusips_in_the_year": sorted(s for r in sel.itertuples() for s, pn in ((r.sym_a, r.permno_a), (r.sym_b, r.permno_b))
                                                                          if len(cs[int(pn)]) > 1)}
            log(f"{y} (screen {x}, first TAQ day {d1}): E2 eligible ETFs {info['E2']['eligible']}, passing pairs {fam['E2']['passing_pairs']}, usable "
                f"{fam['E2']['usable_pairs']}; D2 issuers with two classes {info['D2']['issuers_with_two_classes']}, passing pairs {fam['D2']['passing_pairs']}, "
                f"usable {fam['D2']['usable_pairs']} -> selected {len(sel)} (E2 {fam['E2']['selected'][str(N_MAX)]}, D2 {fam['D2']['selected'][str(N_MAX)]}): "
                + ", ".join(f"{r.sym_a}/{r.sym_b}" for r in sel.itertuples()))
    finally:
        db.close()
    os.makedirs(LOCAL, exist_ok=True)
    pd.concat(keep).to_csv(os.path.join(LOCAL, "pairs_all_years.csv"), index=False)
    pd.DataFrame(legs).to_csv(os.path.join(LOCAL, "legs24.csv"), index=False)
    exm.to_csv(os.path.join(LOCAL, "permno_map_existing.csv"), index=False)
    out["overlap_total"] = {k: sum(v[f"overlap_{k}"] for v in out["years"].values()) for k in ("permno", "ticker")}
    out["years_with_fewer_pairs_than"] = {str(n): [y for y, v in out["years"].items() if v["selected"] < n] for n in NS}
    out["written"] = dt.datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "universe24_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    log(f"overlap with the existing lists: PERMNO {out['overlap_total']['permno']}, ticker {out['overlap_total']['ticker']}; years with fewer "
        f"pairs than N_p: {out['years_with_fewer_pairs_than']}")
    if out["overlap_total"]["permno"] or out["overlap_total"]["ticker"]:
        raise SystemExit("overlap with the existing lists is not 0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    if not ap.parse_args().go:
        sys.exit("refused: needs --go")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    lf = open(os.path.join(HERE, "logs", "universe24.log"), "a")

    def log(m):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {m}"
        print(line, flush=True)
        lf.write(line + "\n")
        lf.flush()
    main(log)
