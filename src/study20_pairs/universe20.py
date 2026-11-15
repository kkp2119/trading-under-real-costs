#!/usr/bin/env python3
"""Research 20 step 1: yearly pairs of two share classes of one issuer (SPEC20 1 with SPEC20_ANSWERS A0; the points the
plan leaves open are listed in SPEC20_QUESTIONS.md). No price difference, signal or P&L is computed here.

For each trading year Y = 2015..2026 (A0: development from 2015; screen year X = Y - 1, year-X information only):
  securities  CRSP common shares (shrcd 10 or 11) listed on NYSE, AMEX or Nasdaq (exchcd 1, 2, 3) on X's last CRSP
              trading day, with at least MIN_DAYS trading days in X
  pairs       two such securities (PERMNO) of one issuer (PERMCO)
  filters     correlation of the daily returns in X >= 0.95 (days on which both have a return, at least MIN_DAYS),
              median close in [5, 2000] for both, median daily dollar volume (|prc| x vol) >= 5,000,000 for both
  excluded    pairs with a PERMNO of the lists of research 12, 15, 17, 18 (mapped as research 18 did: TAQ master CUSIP =
              crsp.stocknames ncusip) and, as a safeguard, pairs with one of their tickers
  identity    a leg is a PERMNO. Its CUSIPs in year Y come from the CRSP name history (crsp.stocknames_v2, 8 characters;
              a CUSIP can change inside a year without a new security, e.g. Alphabet 2015-10, Liberty SiriusXM 2023-08).
              For 2026, beyond the end of CRSP, the last CUSIP is kept.
  TAQ symbol  the master row of Y's first trading day with one of the leg's CUSIPs; a pair without both is skipped
  selected    by the smaller of the two median dollar volumes, largest first; a security is used in one pair only; the
              first N_PAIRS. Leg A is the leg with the larger median dollar volume.
CRSP source: the legacy files (crsp.dsf, crsp.dsenames) end 2024-12-31, so X = 2025 uses the CIZ files (crsp.dsf_v2,
crsp.stocknames_v2) with CRSP's mapping of shrcd 10/11 and exchcd 1/2/3; X = 2024 is computed both ways and compared.
One WRDS connection (the WRDS credentials file). Counts, tickers and rounded statistics go to the log and results/universe20_summary.json;
CRSP identifiers, CUSIPs and the full pair tables stay in $DATA_DIR/research20_data/selection/.
  python universe20.py --go"""
import argparse
import datetime as dt
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
R18 = os.path.join(PROJ, "study18_forecast")
sys.path.insert(0, R18)
import universe18 as U18  # noqa: E402  (existing_lists, permno_map and research 12's table_days; its main is never called)

LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research20_data/selection")
YEARS = range(2015, 2027)            # SPEC20_ANSWERS A0: development 2015-2020, sealed 2021-
N_PAIRS = 4
MIN_DAYS = 120                       # trading days in X, and days with both returns (SPEC20_QUESTIONS A2)
MIN_CORR, PRICE_LO, PRICE_HI, MIN_DV = 0.95, 5.0, 2000.0, 5_000_000.0
LEGACY_END = 2024                    # last screen year with crsp.dsf / crsp.dsenames
NAMES = {"legacy": """select permno, permco, ticker, shrcls, ncusip, comnam from crsp.dsenames
where namedt <= date '{d}' and nameendt >= date '{d}' and shrcd in (10, 11) and exchcd in (1, 2, 3)""",
         "v2": """select permno, permco, ticker, shareclass as shrcls, cusip as ncusip, issuernm as comnam from crsp.stocknames_v2
where namedt <= date '{d}' and nameenddt >= date '{d}' and sharetype = 'NS' and securitytype = 'EQTY' and securitysubtype = 'COM'
  and usincflg = 'Y' and issuertype in ('ACOR', 'CORP') and primaryexch in ('N', 'A', 'Q') and conditionaltype = 'RW'
  and tradingstatusflg = 'A'"""}
DAILY = {"legacy": "select permno, date, abs(prc) as prc, vol, ret from crsp.dsf where date between '{x}-01-01' and '{x}-12-31' and permno in ({p})",
         "v2": "select permno, dlycaldt as date, abs(dlyprc) as prc, dlyvol as vol, dlyret as ret from crsp.dsf_v2 "
               "where dlycaldt between '{x}-01-01' and '{x}-12-31' and permno in ({p})"}
LAST = {"legacy": "select max(date) as d from crsp.dsf where date between '{x}-12-01' and '{x}-12-31'",
        "v2": "select max(dlycaldt) as d from crsp.dsf_v2 where dlycaldt between '{x}-12-01' and '{x}-12-31'"}
HIST_SQL = "select permno, namedt, nameenddt, cusip from crsp.stocknames_v2 where permno in ({p})"
TAQ_SQL = "select symbol_15, {cta} as cta_symbol, cusip from taqm_{y}.mastm_{d} where substr(cusip, 1, 8) in ({c})"
CTA_SQL = "select count(*) as n from information_schema.columns where table_schema = 'taqm_{y}' and table_name = 'mastm_{d}' and column_name = 'cta_symbol'"


def existing_lists():
    """(year, ticker, source) of every list used so far: research 15 (2015-2018), 12 (2019-2026), 17 and 18 (2021-2026)."""
    ex = U18.existing_lists()
    rows = [(y, s, "r18") for y in range(2021, 2027) for s in pd.read_csv(os.path.join(R18, f"universe18_{y}.tsv"), sep="\t", dtype=str).sym]
    return pd.concat([ex, pd.DataFrame(rows, columns=["year", "sym", "source"])], ignore_index=True)


def screen(db, x, source):
    """All pairs of eligible securities of one issuer in screen year x with their statistics and pass flags."""
    d = str(db.raw_sql(LAST[source].format(x=x)).d.iloc[0])[:10]
    n = db.raw_sql(NAMES[source].format(d=d)).drop_duplicates("permno")
    n = n[n.groupby("permco").permno.transform("nunique") >= 2]
    daily = db.raw_sql(DAILY[source].format(x=x, p=",".join(str(int(p)) for p in sorted(n.permno))))
    n["permno"], n["permco"], daily["permno"] = n.permno.astype(int), n.permco.astype(int), daily.permno.astype(int)
    daily["prc"], daily["vol"], daily["ret"] = (pd.to_numeric(daily[c], errors="coerce") for c in ("prc", "vol", "ret"))
    daily["dv"] = daily.prc * daily.vol
    st = daily.groupby("permno").agg(days=("prc", "count"), med_prc=("prc", "median"), med_dv=("dv", "median"))
    n = n.merge(st, left_on="permno", right_index=True, how="left")
    n = n[n.days.fillna(0) >= MIN_DAYS]
    ret = daily.dropna(subset=["ret"]).pivot(index="date", columns="permno", values="ret")
    rows = []
    for _, g in n.groupby("permco"):
        for a, b in itertools.combinations(sorted(g.permno.astype(int)), 2):
            both = ret[[a, b]].dropna() if a in ret and b in ret else pd.DataFrame()
            ra, rb = g[g.permno == a].iloc[0], g[g.permno == b].iloc[0]
            if ra.med_dv < rb.med_dv:                                   # leg A = the larger median dollar volume
                a, b, ra, rb = b, a, rb, ra
            rows.append({"x": x, "source": source, "last_day": d, "permco": int(ra.permco), "comnam": ra.comnam,
                         "permno_a": a, "permno_b": b, "ticker_a": ra.ticker, "ticker_b": rb.ticker, "shrcls_a": ra.shrcls,
                         "shrcls_b": rb.shrcls, "ncusip_a": ra.ncusip, "ncusip_b": rb.ncusip, "days_a": int(ra.days),
                         "days_b": int(rb.days), "days_both": int(len(both)),
                         "corr": float(both[a].corr(both[b])) if len(both) >= 2 else np.nan,
                         "med_prc_a": float(ra.med_prc), "med_prc_b": float(rb.med_prc), "med_dv_a": float(ra.med_dv),
                         "med_dv_b": float(rb.med_dv)})
    p = pd.DataFrame(rows)
    if len(p):
        p["ok_days"] = p.days_both >= MIN_DAYS
        p["ok_corr"] = p["corr"] >= MIN_CORR
        p["ok_price"] = p[["med_prc_a", "med_prc_b"]].min(axis=1).ge(PRICE_LO) & p[["med_prc_a", "med_prc_b"]].max(axis=1).le(PRICE_HI)
        p["ok_dv"] = p.med_dv_b >= MIN_DV
        p["passing"] = p.ok_days & p.ok_corr & p.ok_price & p.ok_dv
    return p, {"last_day": d, "issuers_with_two_classes": int(n.permco.nunique()), "securities": int(len(n)), "pairs": int(len(p))}


def cusip_sets(db, permnos, y, fallback, end):
    """{permno: sorted 8-character CUSIPs of its CRSP name records that overlap year y}; the last record is kept beyond
    the end of CRSP; `fallback` {permno: ncusip of the screen} is always included."""
    out = {int(p): {str(c)[:8]} for p, c in fallback.items()}
    if not permnos:
        return out
    h = db.raw_sql(HIST_SQL.format(p=",".join(str(int(p)) for p in sorted(permnos))))
    h["a"], h["b"] = h.namedt.astype(str).str[:10], h.nameenddt.astype(str).str[:10]
    for p, g in h.groupby(h.permno.astype(int)):
        keep = g[(g.b >= f"{y}-01-01") & (g.a <= f"{y}-12-31")]
        if f"{y}-01-01" > end:                                         # year beyond CRSP: the record open at its end
            keep = g[g.b == end]
        out.setdefault(p, set()).update(str(c)[:8] for c in keep.cusip if c)
    return {p: sorted(v) for p, v in out.items()}


def main(log):
    ex = existing_lists()
    import wrds
    db = wrds.Connection()
    out = {"n_pairs": N_PAIRS, "min_days": MIN_DAYS, "years": {}}
    keep, legs = [], []
    try:
        crsp_end = str(db.raw_sql("select max(nameenddt) as e from crsp.stocknames").e.iloc[0])[:10]
        v2_end = str(db.raw_sql("select max(nameenddt) as e from crsp.stocknames_v2").e.iloc[0])[:10]
        out["crsp_end"] = {"legacy": crsp_end, "v2": v2_end}
        exm = pd.concat([U18.permno_map(db, y - 1, ex.sym[ex.year == y], crsp_end).assign(year=y) for y in range(2015, 2027)])
        ex_permno, ex_tickers = set(exm.permno.dropna().astype(int)), set(ex.sym)
        out["existing"] = {"rows": int(len(ex)), "tickers": int(ex.sym.nunique()), "permnos": len(ex_permno),
                           "tickers_without_permno": sorted(set(exm.sym[exm.permno.isna()])),
                           "rows_by_source": ex.source.value_counts().to_dict()}
        log(f"existing lists (research 12, 15, 17, 18): {len(ex)} rows, {ex.sym.nunique()} tickers, {len(ex_permno)} PERMNOs; "
            f"tickers without a PERMNO: {out['existing']['tickers_without_permno']}")
        for y in YEARS:
            x = y - 1
            p, info = screen(db, x, "legacy" if x <= LEGACY_END else "v2")
            d1 = U18.UR.table_days(db, y, "ctm")[0]
            c = p[p.passing].copy()
            cs = cusip_sets(db, sorted(set(c.permno_a) | set(c.permno_b)), y,
                            {**dict(zip(c.permno_a, c.ncusip_a)), **dict(zip(c.permno_b, c.ncusip_b))}, v2_end)
            allc = sorted({v for s in cs.values() for v in s})
            has_cta = int(db.raw_sql(CTA_SQL.format(y=y, d=d1)).n.iloc[0]) > 0
            m = (db.raw_sql(TAQ_SQL.format(y=y, d=d1, c=",".join(repr(v) for v in allc), cta="cta_symbol" if has_cta else "null"))
                 if allc else pd.DataFrame(columns=["symbol_15", "cta_symbol", "cusip"]))
            m["c8"] = m.cusip.astype(str).str[:8]
            m["shown"] = [t if isinstance(t, str) and t else s.replace(" ", ".") for s, t in zip(m.symbol_15, m.cta_symbol)]
            shown = dict(zip(m.symbol_15, m.shown))
            for leg in "ab":
                hit = [sorted(set(m.symbol_15[m.c8.isin(cs[int(pn)])])) for pn in c[f"permno_{leg}"]]
                c[f"taq_{leg}"] = [h[0] if len(h) == 1 else None for h in hit]
                c[f"sym_{leg}"] = [shown.get(v) if v else None for v in c[f"taq_{leg}"]]
            c["hit_permno"] = [a in ex_permno or b in ex_permno for a, b in zip(c.permno_a, c.permno_b)]
            c["hit_ticker"] = [(a in ex_tickers or b in ex_tickers) and not h for a, b, h in zip(c.sym_a, c.sym_b, c.hit_permno)]
            c["no_taq"] = c.taq_a.isna() | c.taq_b.isna()
            c["min_dv"] = c.med_dv_b
            c = c.sort_values(["min_dv", "permno_a", "permno_b"], ascending=[False, True, True]).reset_index(drop=True)
            used, pick = set(), []
            for i, r in c.iterrows():
                ok = not (r.hit_permno or r.hit_ticker or r.no_taq) and r.permno_a not in used and r.permno_b not in used
                if ok and len(pick) < N_PAIRS:
                    pick.append(i)
                    used |= {r.permno_a, r.permno_b}
            c["selected"] = c.index.isin(pick)
            c["rank"] = [pick.index(i) + 1 if i in pick else 0 for i in c.index]
            keep.append(p.assign(year=y).merge(c[["permno_a", "permno_b", "taq_a", "taq_b", "sym_a", "sym_b", "hit_permno", "hit_ticker",
                                                 "no_taq", "selected", "rank"]], on=["permno_a", "permno_b"], how="left"))
            sel = c[c.selected].sort_values("rank")
            rows = []
            for r in sel.itertuples():
                for leg, t, s, pn in (("A", r.taq_a, r.sym_a, r.permno_a), ("B", r.taq_b, r.sym_b, r.permno_b)):
                    rows.append({"year": y, "pair": r.rank, "leg": leg, "sym": s, "symbol_15": t})
                    legs.append({"year": y, "pair": r.rank, "leg": leg, "sym": s, "symbol_15": t, "permno": pn, "permco": r.permco,
                                 "cusips": "|".join(cs[int(pn)])})
            u = pd.DataFrame(rows, columns=["year", "pair", "leg", "sym", "symbol_15"])
            u.to_csv(os.path.join(HERE, f"universe20_{y}.tsv"), sep="\t", index=False)
            perm = set(sel.permno_a) | set(sel.permno_b)
            out["years"][y] = {
                "screen_year": x, "crsp_source": "legacy" if x <= LEGACY_END else "v2", **info, "first_day": d1,
                "pairs_passing": int(len(c)), "failed": {k: int((~p[k]).sum()) for k in ("ok_days", "ok_corr", "ok_price", "ok_dv")} if len(p) else {},
                "passing_excluded_existing_permno": int(c.hit_permno.sum()), "passing_excluded_existing_ticker_only": int(c.hit_ticker.sum()),
                "passing_without_taq_symbol": int(c.no_taq.sum()), "selected": int(len(sel)),
                "overlap_permno": len(perm & ex_permno), "overlap_ticker": len(set(u.sym) & ex_tickers),
                "legs_with_two_cusips_in_the_year": sorted(s for r in sel.itertuples() for s, pn in ((r.sym_a, r.permno_a), (r.sym_b, r.permno_b))
                                                           if len(cs[int(pn)]) > 1),
                "passing_pairs": [{"sym_a": r.sym_a, "sym_b": r.sym_b, "issuer": r.comnam, "corr": round(r.corr, 4),
                                   "med_dv_musd": [round(r.med_dv_a / 1e6, 1), round(r.med_dv_b / 1e6, 1)],
                                   "med_close": [round(r.med_prc_a, 2), round(r.med_prc_b, 2)], "days_both": int(r.days_both),
                                   "rank": int(r.rank), "note": "existing PERMNO" if r.hit_permno else "existing ticker" if r.hit_ticker
                                   else "no TAQ symbol on the first day" if r.no_taq else ""} for r in c.itertuples()]}
            log(f"{y} (screen {x}, {out['years'][y]['crsp_source']}, last CRSP day {info['last_day']}, first TAQ day {d1}): issuers with two listed "
                f"common classes {info['issuers_with_two_classes']}, pairs {info['pairs']}, passing {len(c)}, existing PERMNO {int(c.hit_permno.sum())}, "
                f"no TAQ symbol {int(c.no_taq.sum())} -> selected " + ", ".join(f"{r.sym_a}/{r.sym_b} (corr {r.corr:.3f}, ${r.med_dv_b / 1e6:.0f}M)" for r in sel.itertuples()))
        # the two CRSP formats on the same screen year
        a, _ = screen(db, LEGACY_END, "legacy")
        b, _ = screen(db, LEGACY_END, "v2")
        ka, kb = (set(zip(t.permno_a[t.passing], t.permno_b[t.passing])) for t in (a, b))
        j = a[a.passing].merge(b[b.passing], on=["permno_a", "permno_b"], suffixes=("", "_v2"))
        out["legacy_vs_v2_on_2024"] = {
            "passing_legacy": len(ka), "passing_v2": len(kb), "in_both": len(ka & kb),
            "max_abs_diff_corr": float((j["corr"] - j["corr_v2"]).abs().max()) if len(j) else None,
            "max_rel_diff_med_dv": float(((j.med_dv_b - j.med_dv_b_v2).abs() / j.med_dv_b).max()) if len(j) else None,
            "same_order_of_min_dv": bool(list(j.sort_values("med_dv_b", ascending=False).permno_a) == list(j.sort_values("med_dv_b_v2", ascending=False).permno_a)) if len(j) else None}
        log(f"legacy and CIZ on screen year {LEGACY_END}: passing pairs {len(ka)} and {len(kb)}, in both {len(ka & kb)}, "
            f"largest correlation difference {out['legacy_vs_v2_on_2024']['max_abs_diff_corr']}")
    finally:
        db.close()
    os.makedirs(LOCAL, exist_ok=True)
    pd.concat(keep).to_csv(os.path.join(LOCAL, "pairs_all_years.csv"), index=False)
    pd.DataFrame(legs).to_csv(os.path.join(LOCAL, "legs20.csv"), index=False)        # CUSIPs of each selected leg (the downloader reads it)
    exm.to_csv(os.path.join(LOCAL, "permno_map_existing.csv"), index=False)
    out["overlap_total"] = {"permno": sum(v["overlap_permno"] for v in out["years"].values()),
                            "ticker": sum(v["overlap_ticker"] for v in out["years"].values())}
    out["years_with_fewer_than_two_pairs"] = [y for y, v in out["years"].items() if v["selected"] < 2]
    out["written"] = dt.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(HERE, "results", "universe20_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    log(f"overlap with the existing lists: PERMNO {out['overlap_total']['permno']}, ticker {out['overlap_total']['ticker']}; "
        f"years with fewer than two pairs: {out['years_with_fewer_than_two_pairs']}")
    if out["overlap_total"]["permno"] or out["overlap_total"]["ticker"]:
        raise SystemExit("overlap with the existing lists is not 0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    if not ap.parse_args().go:
        sys.exit("refused: needs --go")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    lf = open(os.path.join(HERE, "logs", "universe20.log"), "a")

    def log(m):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {m}"
        print(line, flush=True)
        lf.write(line + "\n")
        lf.flush()
    main(log)
