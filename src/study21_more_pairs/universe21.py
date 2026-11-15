#!/usr/bin/env python3
"""Research 21 step 1: yearly pairs of the two families (SPEC21 1; the points the plan leaves open are listed in
SPEC21_QUESTIONS.md). No price difference, signal or P&L is computed here.

For each trading year Y = 2015..2026 (screen year X = Y - 1, year-X information only; D = X's last CRSP trading day):
  family E   ETFs on D: CRSP security type FUND with subtype ETF, primary exchange NYSE, NYSE Arca, Nasdaq, NYSE American
             or Cboe BZX (N, R, Q, A, B), regular way and active
             eligible  at least MIN_DAYS trading days in X, median close in [5, 2000], median daily dollar volume
                       (|price| x volume) >= 50,000,000
             pair      correlation of the daily returns in X >= 0.995 and std(A) / std(B) in [0.95, 1.05], both over the
                       days on which both legs have a return
             order     the smaller of the two median dollar volumes, largest first
  family S   common shares on D (shrcd 10/11 on NYSE, NYSE American or Nasdaq; in the CIZ files CRSP's mapping, as in
             research 20), the S_TOP largest by the security's market capitalisation on D (|price| x shares outstanding)
             eligible  the same three conditions as E
             pair      two issuers (PERMCO differs), the same four-digit SIC code in the name records valid on D
                       (missing, 0 and 9999 are no code), correlation >= 0.80 over the days on which both have a return
             order     correlation, largest first
  both       leg A is the leg with the larger median dollar volume
             excluded  pairs with a PERMNO of an earlier list: research 12 (yearly stock lists and its ten control
                       ETFs), 15, 17, 18 (mapped as research 18 did: TAQ master CUSIP = crsp.stocknames ncusip) and 20
                       (PERMNOs of its legs); as a safeguard, as before, also pairs with one of their tickers
             identity  a leg is a PERMNO. Its CUSIPs in year Y come from the CRSP name history (research 20's
                       cusip_sets); a pair needs exactly one TAQ master symbol per leg on Y's first trading day
             selected  in the family's order, a security in one pair only, the first N_PAIRS
CRSP source: the CIZ files (crsp.dsf_v2, crsp.stocknames_v2) for every screen year, because only they flag ETFs. The
screens of X <= 2024 are repeated on the legacy files (crsp.dsf, crsp.dsenames: shrcd 73 for E, shrcd 10/11 with exchcd
1/2/3 for S) and the two selections are compared (results: legacy_check).
One WRDS connection (the WRDS credentials file). Counts, tickers and rounded statistics go to the log and
results/universe21_summary.json; CRSP identifiers, CUSIPs and the full pair tables stay in
$DATA_DIR/research21_data/selection/.
  python universe21.py --go"""
import argparse
import datetime as dt
import itertools
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
R12, R18, R20 = (os.path.join(PROJ, f"research{k}") for k in (12, 18, 20))
sys.path.insert(0, R18)
sys.path.insert(0, R20)
import universe18 as U18  # noqa: E402  (permno_map and research 12's table_days; its main is never called)
import universe20 as U20  # noqa: E402  (existing_lists, cusip_sets, TAQ_SQL, CTA_SQL; its main is never called)

LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research21_data/selection")
LEGS20 = os.path.join(os.environ.get("DATA_DIR", "data"), "research20_data/selection/legs20.csv")
YEARS = range(2015, 2027)            # SPEC21 1.3: development 2015-2020, sealed 2021-
N_PAIRS = 4
MIN_DAYS = 200
PRICE_LO, PRICE_HI, MIN_DV = 5.0, 2000.0, 50_000_000.0
E_MIN_CORR, E_SD_LO, E_SD_HI = 0.995, 0.95, 1.05
E_KEEP_CORR = 0.99                   # pairs of eligible ETFs below this correlation are counted, not tabulated
S_MIN_CORR, S_TOP = 0.80, 200
NO_SIC = (0, 9999)
LEGACY_END = 2024                    # last screen year with crsp.dsf / crsp.dsenames
LAST = {"v2": "select max(dlycaldt) as d from crsp.dsf_v2 where dlycaldt between '{x}-12-01' and '{x}-12-31'",
        "legacy": "select max(date) as d from crsp.dsf where date between '{x}-12-01' and '{x}-12-31'"}
NAMES = {
    ("E", "v2"): """select permno, permco, ticker, shareclass as shrcls, cusip as ncusip, issuernm as comnam, siccd, primaryexch as exch
from crsp.stocknames_v2 where namedt <= date '{d}' and nameenddt >= date '{d}' and securitytype = 'FUND' and securitysubtype = 'ETF'
  and primaryexch in ('N', 'R', 'Q', 'A', 'B') and conditionaltype = 'RW' and tradingstatusflg = 'A'""",
    ("E", "legacy"): """select permno, permco, ticker, shrcls, ncusip, comnam, siccd, exchcd as exch from crsp.dsenames
where namedt <= date '{d}' and nameendt >= date '{d}' and shrcd = 73 and exchcd in (1, 2, 3, 4, 5)""",
    ("S", "v2"): """select permno, permco, ticker, shareclass as shrcls, cusip as ncusip, issuernm as comnam, siccd, primaryexch as exch
from crsp.stocknames_v2 where namedt <= date '{d}' and nameenddt >= date '{d}' and sharetype = 'NS' and securitytype = 'EQTY'
  and securitysubtype = 'COM' and usincflg = 'Y' and issuertype in ('ACOR', 'CORP') and primaryexch in ('N', 'A', 'Q')
  and conditionaltype = 'RW' and tradingstatusflg = 'A'""",
    ("S", "legacy"): """select permno, permco, ticker, shrcls, ncusip, comnam, siccd, exchcd as exch from crsp.dsenames
where namedt <= date '{d}' and nameendt >= date '{d}' and shrcd in (10, 11) and exchcd in (1, 2, 3)"""}
DAILY = {"v2": "select permno, dlycaldt as date, abs(dlyprc) as prc, dlyvol as vol, dlyret as ret from crsp.dsf_v2 "
               "where dlycaldt between '{x}-01-01' and '{x}-12-31' and permno in ({p})",
         "legacy": "select permno, date, abs(prc) as prc, vol, ret from crsp.dsf where date between '{x}-01-01' and '{x}-12-31' and permno in ({p})"}
CAP = {"v2": "select permno, abs(dlyprc) * shrout as cap from crsp.dsf_v2 where dlycaldt = date '{d}'",
       "legacy": "select permno, abs(prc) * shrout as cap from crsp.dsf where date = date '{d}'"}
PAIR_COLS = ["permno_a", "permno_b", "permco_a", "permco_b", "ticker_a", "ticker_b", "comnam_a", "comnam_b", "ncusip_a", "ncusip_b",
             "sic_a", "sic_b", "exch_a", "exch_b", "days_a", "days_b", "days_both", "corr", "sd_ratio", "med_prc_a", "med_prc_b",
             "med_dv_a", "med_dv_b", "rank_a", "rank_b"]


def r12_etfs():
    """The ten control ETFs of research 12 (study12_market_maker/config.py ETFS)."""
    m = re.search(r"^ETFS = \(([^)]*)\)", open(os.path.join(R12, "config.py")).read(), re.M)
    etfs = re.findall(r'"([A-Z]+)"', m.group(1))
    if len(etfs) != 10:
        raise SystemExit("study12_market_maker/config.py: ETFS is not ten symbols")
    return etfs


def existing(db, crsp_end):
    """(ex, exm, permnos, tickers): the lists used so far and their PERMNOs. Research 15 (2015-2018), 12 (2019-2026, stocks
    and control ETFs), 17 and 18 (2021-2026) are mapped on their own screen year; research 20's legs carry their PERMNO."""
    ex = U20.existing_lists()
    ex = pd.concat([ex, pd.DataFrame([(y, s, "r12_etf") for y in range(2019, 2027) for s in r12_etfs()], columns=["year", "sym", "source"])],
                   ignore_index=True)
    exm = pd.concat([U18.permno_map(db, y - 1, ex.sym[ex.year == y], crsp_end).assign(year=y) for y in range(2015, 2027)])
    l20 = pd.read_csv(LEGS20, dtype=str)
    ex = pd.concat([ex, l20[["year", "sym"]].astype({"year": int}).assign(source="r20")], ignore_index=True)
    permnos = set(exm.permno.dropna().astype(int)) | set(l20.permno.astype(int))
    return ex, exm, permnos, set(ex.sym), l20


def frame(db, sql, key="permno"):
    t = db.raw_sql(sql)
    t[key] = t[key].astype(int)
    return t


def base(db, fam, x, src, d):
    """Securities of the family on D with their year-X statistics (index PERMNO) and the table of daily returns."""
    n = frame(db, NAMES[fam, src].format(d=d)).drop_duplicates("permno").set_index("permno")
    n["permco"], n["siccd"] = n.permco.astype(int), pd.to_numeric(n.siccd, errors="coerce")
    n["rank"] = 0
    if fam == "S":
        cap = frame(db, CAP[src].format(d=d)).drop_duplicates("permno").set_index("permno").cap
        n["cap"] = pd.to_numeric(cap, errors="coerce").reindex(n.index)
        n = n[n.cap > 0].reset_index().sort_values(["cap", "permno"], ascending=[False, True]).head(S_TOP).set_index("permno")
        n["rank"] = range(1, len(n) + 1)
    daily = frame(db, DAILY[src].format(x=x, p=",".join(str(p) for p in sorted(n.index))))
    for c in ("prc", "vol", "ret"):
        daily[c] = pd.to_numeric(daily[c], errors="coerce").astype(float)
    daily["dv"] = daily.prc * daily.vol
    st = daily.groupby("permno").agg(days=("prc", "count"), med_prc=("prc", "median"), med_dv=("dv", "median"))
    n = n.join(st)
    n["days"] = n.days.fillna(0).astype(int)
    n["ok_days"], n["ok_price"], n["ok_dv"] = n.days >= MIN_DAYS, n.med_prc.between(PRICE_LO, PRICE_HI), n.med_dv >= MIN_DV
    n["eligible"] = n.ok_days & n.ok_price & n.ok_dv
    ret = daily.dropna(subset=["ret"]).pivot(index="date", columns="permno", values="ret")
    return n, ret


def pair_row(n, ret, a, b):
    ra, rb = n.loc[a], n.loc[b]
    if ra.med_dv < rb.med_dv:                                           # leg A = the larger median dollar volume
        a, b, ra, rb = b, a, rb, ra
    both = ret[[a, b]].dropna() if a in ret and b in ret else pd.DataFrame(columns=[a, b], dtype=float)
    sa, sb = (float(both[k].std()) if len(both) >= 2 else np.nan for k in (a, b))
    out = {"permno_a": a, "permno_b": b, "days_both": int(len(both)),
           "corr": float(both[a].corr(both[b])) if len(both) >= 2 else np.nan, "sd_ratio": sa / sb if sb and sb > 0 else np.nan}
    for leg, r in (("a", ra), ("b", rb)):
        out.update({f"permco_{leg}": int(r.permco), f"ticker_{leg}": r.ticker, f"comnam_{leg}": r.comnam, f"ncusip_{leg}": r.ncusip,
                    f"sic_{leg}": None if pd.isna(r.siccd) else int(r.siccd), f"exch_{leg}": str(r.exch), f"days_{leg}": int(r.days),
                    f"med_prc_{leg}": float(r.med_prc), f"med_dv_{leg}": float(r.med_dv), f"rank_{leg}": int(r["rank"])})
    return out


def screen(db, fam, x, src):
    """(pairs, info) of one family in screen year x. E: the pairs of eligible ETFs with a correlation of at least
    E_KEEP_CORR; S: every pair of two issuers with one SIC code among the S_TOP largest."""
    d = str(db.raw_sql(LAST[src].format(x=x)).d.iloc[0])[:10]
    n, ret = base(db, fam, x, src, d)
    el = n[n.eligible]
    info = {"last_day": d, "securities": int(len(n)), "eligible": int(len(el)),
            "not_eligible": {k: int((~n[k]).sum()) for k in ("ok_days", "ok_price", "ok_dv")}}
    if fam == "E":
        cm = ret[sorted(el.index.intersection(ret.columns))].corr()
        rows = [pair_row(n, ret, a, b) for a, b in itertools.combinations(sorted(cm.columns), 2) if cm.at[a, b] >= E_KEEP_CORR]
        p = pd.DataFrame(rows, columns=PAIR_COLS)
        p["ok_corr"], p["ok_sd"] = p["corr"] >= E_MIN_CORR, p.sd_ratio.between(E_SD_LO, E_SD_HI)
        p["passing"] = p.ok_corr & p.ok_sd
        p["key"] = p.med_dv_b
        info.update({"pairs_of_eligible": int(len(el) * (len(el) - 1) // 2), "corr_ok": int(p.ok_corr.sum()),
                     "corr_ok_sd_outside": int((p.ok_corr & ~p.ok_sd).sum())})
    else:
        sic = n.siccd.where(~n.siccd.isin(NO_SIC))
        rows, same_issuer = [], 0
        for _, g in n[sic.notna()].groupby(sic[sic.notna()]):
            for a, b in itertools.combinations(sorted(g.index), 2):
                if g.permco[a] == g.permco[b]:
                    same_issuer += 1
                    continue
                rows.append({**pair_row(n, ret, a, b), "ok_elig": bool(n.eligible[a] and n.eligible[b])})
        p = pd.DataFrame(rows, columns=PAIR_COLS + ["ok_elig"])
        p["ok_corr"] = p["corr"] >= S_MIN_CORR
        p["passing"] = p.ok_elig.astype(bool) & p.ok_corr
        p["key"] = p["corr"]
        info.update({"without_sic_code": int(sic.isna().sum()), "pairs_same_sic_two_issuers": int(len(p)), "pairs_same_sic_one_issuer": same_issuer,
                     "pairs_both_eligible": int(p.ok_elig.astype(bool).sum()), "corr_ok": int(p.ok_corr.sum())})
    info["passing"] = int(p.passing.sum())
    return p.assign(x=x, source=src, family=fam, last_day=d), info


def choose(c, sym_of, ex_permno, ex_tickers):
    """The passing pairs in the family's order with TAQ symbols, exclusion flags and the selected N_PAIRS."""
    c = c[c.passing].copy()
    for leg in "ab":
        c[f"taq_{leg}"] = [sym_of.get(int(p), (None, None))[0] for p in c[f"permno_{leg}"]]
        c[f"sym_{leg}"] = [sym_of.get(int(p), (None, None))[1] for p in c[f"permno_{leg}"]]
    c["hit_permno"] = [a in ex_permno or b in ex_permno for a, b in zip(c.permno_a, c.permno_b)]
    c["hit_ticker"] = [bool({a, b, ta, tb} & ex_tickers) and not h for a, b, ta, tb, h in zip(c.sym_a, c.sym_b, c.ticker_a, c.ticker_b, c.hit_permno)]
    c["no_taq"] = c.taq_a.isna() | c.taq_b.isna()
    c = c.sort_values(["key", "permno_a", "permno_b"], ascending=[False, True, True]).reset_index(drop=True)
    used, pick = set(), []
    for i, r in c.iterrows():
        if not (r.hit_permno or r.hit_ticker or r.no_taq) and r.permno_a not in used and r.permno_b not in used and len(pick) < N_PAIRS:
            pick.append(i)
            used |= {r.permno_a, r.permno_b}
    c["rank"] = [pick.index(i) + 1 if i in pick else 0 for i in c.index]
    c["selected"] = c["rank"] > 0
    return c


def note(r):
    return ("existing PERMNO" if r.hit_permno else "existing ticker" if r.hit_ticker else "no TAQ symbol on the first day" if r.no_taq
            else "" if r.selected else "a leg is already in a pair, or four pairs are chosen")


def shown(r, fam):
    out = {"sym_a": r.sym_a or r.ticker_a, "sym_b": r.sym_b or r.ticker_b, "corr": round(r.corr, 4), "sd_ratio": round(r.sd_ratio, 3),
           "med_dv_musd": [round(r.med_dv_a / 1e6, 1), round(r.med_dv_b / 1e6, 1)], "med_close": [round(r.med_prc_a, 2), round(r.med_prc_b, 2)],
           "days": [int(r.days_a), int(r.days_b), int(r.days_both)], "rank": int(r.rank), "note": note(r)}
    if fam == "S":
        out.update({"sic": int(r.sic_a), "issuers": [r.comnam_a, r.comnam_b], "cap_rank": [int(r.rank_a), int(r.rank_b)]})
    else:
        out.update({"funds": [r.comnam_a, r.comnam_b]})
    return out


def main(log):
    import wrds
    db = wrds.Connection()
    out = {"n_pairs": N_PAIRS, "min_days": MIN_DAYS, "years": {}, "legacy_check": {}}
    keep, legs = [], []
    try:
        crsp_end = str(db.raw_sql("select max(nameenddt) as e from crsp.stocknames").e.iloc[0])[:10]
        v2_end = str(db.raw_sql("select max(nameenddt) as e from crsp.stocknames_v2").e.iloc[0])[:10]
        out["crsp_end"] = {"legacy": crsp_end, "v2": v2_end}
        ex, exm, ex_permno, ex_tickers, l20 = existing(db, crsp_end)
        nomap = sorted(set(exm.sym[exm.permno.isna()]) - set(exm.sym[exm.permno.notna()]))
        rows_nomap = int(exm.groupby(["year", "sym"]).permno.apply(lambda v: v.notna().sum() == 0).sum())
        out["existing"] = {"rows": int(len(ex)), "tickers": int(ex.sym.nunique()), "permnos": len(ex_permno),
                           "tickers_without_permno": nomap, "year_rows_without_permno": rows_nomap,
                           "rows_by_source": ex.source.value_counts().to_dict(),
                           "control_etfs_of_research12": r12_etfs()}
        log(f"existing lists (research 12 stocks and control ETFs, 15, 17, 18, 20): {len(ex)} rows, {ex.sym.nunique()} tickers, "
            f"{len(ex_permno)} PERMNOs; tickers without a PERMNO: {nomap}; year rows without a PERMNO: {rows_nomap}")
        for y in YEARS:
            x = y - 1
            d1 = U18.UR.table_days(db, y, "ctm")[0]
            tabs = {(f, s): screen(db, f, x, s) for s in (("v2", "legacy") if x <= LEGACY_END else ("v2",)) for f in "ES"}
            ps = pd.concat([t[t.passing] for t, _ in tabs.values()])
            perm = sorted(set(ps.permno_a) | set(ps.permno_b))
            cs = U20.cusip_sets(db, perm, y, {**dict(zip(ps.permno_a, ps.ncusip_a)), **dict(zip(ps.permno_b, ps.ncusip_b))}, v2_end)
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
            ch = {k: choose(t, sym_of, ex_permno, ex_tickers) for k, (t, _) in tabs.items()}
            out["years"][y], rows = {"screen_year": x, "first_day": d1}, []
            for f in "ES":
                c, (p, info) = ch[f, "v2"], tabs[f, "v2"]
                keep.append(p.assign(year=y).merge(c[["permno_a", "permno_b", "taq_a", "taq_b", "sym_a", "sym_b", "hit_permno", "hit_ticker", "no_taq",
                                                      "selected", "rank"]], on=["permno_a", "permno_b"], how="left"))
                sel = c[c.selected].sort_values("rank")
                for r in sel.itertuples():
                    for leg, t, s, pn, pc in (("A", r.taq_a, r.sym_a, r.permno_a, r.permco_a), ("B", r.taq_b, r.sym_b, r.permno_b, r.permco_b)):
                        rows.append({"year": y, "family": f, "pair": f"{f}{r.rank}", "leg": leg, "sym": s, "symbol_15": t})
                        cus = [k for k in cs[int(pn)] if len(k) == 8]
                        if not cus:
                            raise SystemExit(f"{y} {s}: no CUSIP")
                        legs.append({**rows[-1], "permno": pn, "permco": pc, "cusips": "|".join(cus)})
                perm_sel = set(sel.permno_a) | set(sel.permno_b)
                last = int(c.index[c.selected].max()) if len(sel) else -1
                out["years"][y][f] = {
                    **info, "excluded_existing_permno": int(c.hit_permno.sum()), "excluded_existing_ticker_only": int(c.hit_ticker.sum()),
                    "without_taq_symbol": int(c.no_taq.sum()), "usable": int((~(c.hit_permno | c.hit_ticker | c.no_taq)).sum()),
                    "selected": int(len(sel)), "overlap_permno": len(perm_sel & ex_permno),
                    "overlap_ticker": len((set(sel.sym_a) | set(sel.sym_b)) & ex_tickers),
                    "legs_with_two_cusips_in_the_year": sorted(s for r in sel.itertuples() for s, pn in ((r.sym_a, r.permno_a), (r.sym_b, r.permno_b))
                                                               if len(cs[int(pn)]) > 1),
                    "pairs": [shown(r, f) for r in c[c.index <= max(last, min(len(c), 12) - 1)].itertuples()]}
                log(f"{y} {f} (screen {x}, last CRSP day {info['last_day']}, first TAQ day {d1}): securities {info['securities']}, eligible "
                    f"{info['eligible']}, passing pairs {info['passing']}, existing PERMNO {int(c.hit_permno.sum())}, existing ticker only "
                    f"{int(c.hit_ticker.sum())}, no TAQ symbol {int(c.no_taq.sum())} -> selected "
                    + ", ".join(f"{r.sym_a}/{r.sym_b} (corr {r.corr:.4f}, sd {r.sd_ratio:.3f}, ${r.med_dv_b / 1e6:.0f}M"
                                + (f", SIC {int(r.sic_a)})" if f == "S" else ")") for r in sel.itertuples()))
                if (f, "legacy") in ch:
                    cl, (pl, il) = ch[f, "legacy"], tabs[f, "legacy"]
                    ka, kb = (set(zip(t.permno_a[t.passing], t.permno_b[t.passing])) for t in (p, pl))
                    j = p[p.passing].merge(pl[pl.passing], on=["permno_a", "permno_b"], suffixes=("", "_l"))
                    sa, sb = ([f"{r.sym_a}/{r.sym_b}" for r in t[t.selected].sort_values("rank").itertuples()] for t in (c, cl))
                    out["legacy_check"][f"{y} {f}"] = {
                        "securities": [info["securities"], il["securities"]], "eligible": [info["eligible"], il["eligible"]],
                        "passing": [len(ka), len(kb)], "in_both": len(ka & kb),
                        "max_abs_diff_corr": float((j["corr"] - j["corr_l"]).abs().max()) if len(j) else None,
                        "max_rel_diff_med_dv": float(((j.med_dv_b - j.med_dv_b_l).abs() / j.med_dv_b).max()) if len(j) else None,
                        "selected_v2": sa, "selected_legacy": sb, "same_selection": sa == sb}
                    if sa != sb:
                        log(f"  legacy files give another selection for {y} {f}: {sb}")
            u = pd.DataFrame(rows, columns=["year", "family", "pair", "leg", "sym", "symbol_15"])
            if u.sym.duplicated().any() or u.sym.isna().any():
                raise SystemExit(f"{y}: a symbol is missing or used twice")
            u.to_csv(os.path.join(HERE, f"universe21_{y}.tsv"), sep="\t", index=False)
    finally:
        db.close()
    os.makedirs(LOCAL, exist_ok=True)
    pd.concat(keep).to_csv(os.path.join(LOCAL, "pairs_all_years.csv"), index=False)
    pd.DataFrame(legs).to_csv(os.path.join(LOCAL, "legs21.csv"), index=False)        # CUSIPs of each selected leg (the downloader reads it)
    exm.to_csv(os.path.join(LOCAL, "permno_map_existing.csv"), index=False)
    out["overlap_total"] = {k: sum(v[f][f"overlap_{k}"] for v in out["years"].values() for f in "ES") for k in ("permno", "ticker")}
    out["years_with_fewer_than_two_pairs"] = {f: [y for y, v in out["years"].items() if v[f]["selected"] < 2] for f in "ES"}
    out["years_with_fewer_than_four_pairs"] = {f: [y for y, v in out["years"].items() if v[f]["selected"] < N_PAIRS] for f in "ES"}
    out["legacy_check_differences"] = sorted(k for k, v in out["legacy_check"].items() if not v["same_selection"])
    out["written"] = dt.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(HERE, "results", "universe21_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    log(f"overlap with the existing lists: PERMNO {out['overlap_total']['permno']}, ticker {out['overlap_total']['ticker']}; years with fewer "
        f"than two pairs: {out['years_with_fewer_than_two_pairs']}; fewer than four: {out['years_with_fewer_than_four_pairs']}; "
        f"legacy files differ in: {out['legacy_check_differences']}")
    if out["overlap_total"]["permno"] or out["overlap_total"]["ticker"]:
        raise SystemExit("overlap with the existing lists is not 0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    if not ap.parse_args().go:
        sys.exit("refused: needs --go")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    lf = open(os.path.join(HERE, "logs", "universe21.log"), "a")

    def log(m):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {m}"
        print(line, flush=True)
        lf.write(line + "\n")
        lf.flush()
    main(log)
