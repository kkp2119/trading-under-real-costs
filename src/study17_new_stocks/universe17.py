#!/usr/bin/env python3
"""Research 17 step 1: new-universe draw for 2021-2026 (SPEC17 1; choices recorded in SPEC17_QUESTIONS.md).
For each trading year Y = 2021..2026 (screen year X = Y - 1):
  passing   research 12's screen output for X (research12/universe_screen_<X>.tsv, pass_all; same code and filters,
            checked against the symbols of research12/universe_draw_order_<X>.tsv). Only year-X information.
  PERMNO    X's last TAQ master (taqm_X.mastm_<last date>) CUSIP[:8] = crsp.stocknames ncusip, namedt <= D <= nameenddt with
            D = min(master date, last CRSP name date) (the research 15 crsp_agree15 / universe15 join, any shrcd). CRSP ends
            2024-12-31 on WRDS (2026-11-04), so the X = 2025 symbols are matched on their name record of that day. The existing lists 2015-2026 (research15
            universe15_<Y>.tsv, research12 universe_<Y>.tsv) are mapped the same way on their own X.
  eligible  passing minus every existing PERMNO (and, as a safeguard, every existing ticker); candidates without a unique
            PERMNO are left out (their overlap cannot be checked) and counted.
  order     numpy default_rng(17).permutation(eligible sorted by symbol), per year.
  select    research 12's universe_rolling.select rule: the first N in that order with at least one valid session trade
            and one valid session quote on Y's first ctm day (research 12 TRADE_OK / QUOTE_OK; counts only).
One WRDS connection (PostgreSQL client credentials). Counts and tickers go to the log; CRSP fields stay in $DATA_DIR/research17_data/selection/.
  python universe17.py --n 20 --go"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
R12, R15 = os.path.join(PROJ, "study12_market_maker"), os.path.join(PROJ, "study15_ml_execution")
sys.path.insert(0, R12)
import universe_rolling as UR  # noqa: E402  (SQL constants and table_days only; its main is never called)

LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research17_data/selection")
YEARS = range(2021, 2027)
SEED = 17
MAP_SQL = """select m.symbol_15 as sym, m.cusip, s.permno, s.siccd, s.shrcd, s.exchcd
from taqm_{x}.mastm_{m} m left join crsp.stocknames s on s.ncusip = substr(m.cusip, 1, 8)
  and s.namedt <= date '{d}' and s.nameenddt >= date '{d}'
where m.symbol_15 in ({syms})"""
SIC_DIV = ((1, 9, "Agriculture"), (10, 14, "Mining"), (15, 17, "Construction"), (20, 39, "Manufacturing"),
           (40, 49, "Transport & utilities"), (50, 51, "Wholesale"), (52, 59, "Retail"), (60, 67, "Finance"),
           (70, 89, "Services"), (91, 97, "Public administration"), (99, 99, "Nonclassifiable"))
PRICE_BANDS = ((10, 20), (20, 50), (50, 100), (100, 150))


def sic_division(sic):
    try:
        k = int(sic) // 100
    except (TypeError, ValueError):
        return "unknown"
    return next((n for a, b, n in SIC_DIV if a <= k <= b), "unknown")


def existing_lists():
    rows = []
    for y in range(2015, 2027):
        p = os.path.join(R15, f"universe15_{y}.tsv") if y <= 2018 else os.path.join(R12, f"universe_{y}.tsv")
        rows += [(y, s) for s in pd.read_csv(p, sep="\t", dtype=str).sym]
    return pd.DataFrame(rows, columns=["year", "sym"])


def passing(x):
    s = pd.read_csv(os.path.join(R12, f"universe_screen_{x}.tsv"), sep="\t", dtype={"sym": str})
    p = sorted(s.sym[s.pass_all.astype(bool)])
    d = set(pd.read_csv(os.path.join(R12, f"universe_draw_order_{x}.tsv"), sep="\t", dtype={"sym": str}).sym)
    if set(p) != d:
        raise SystemExit(f"screen {x}: pass_all ({len(p)}) differs from universe_draw_order_{x}.tsv ({len(d)})")
    return p, s.set_index("sym").med_price


def last_master(db, x):
    t = db.raw_sql(f"select table_name from information_schema.tables where table_schema = 'taqm_{x}' "
                   f"and table_name ~ '^mastm_{x}[0-9]{{4}}$'")
    return sorted(n[-8:] for n in t.table_name)[-1]


def permno_map(db, x, syms, crsp_end):
    m = last_master(db, x)
    d = min(f"{m[:4]}-{m[4:6]}-{m[6:]}", crsp_end)
    q = ",".join(f"'{s}'" for s in sorted(set(syms)))
    t = db.raw_sql(MAP_SQL.format(x=x, m=m, d=d, syms=q))
    t["x"], t["master"], t["crsp_date"] = x, m, d
    return t


def main(n, log):
    ex = existing_lists()
    import wrds
    db = wrds.Connection()
    maps, out = [], {"seed": SEED, "n_per_year": n, "years": {}}
    try:
        crsp_end = str(db.raw_sql("select max(nameenddt) as e from crsp.stocknames").e.iloc[0])[:10]
        out["crsp_stocknames_end"] = crsp_end
        log(f"crsp.stocknames ends {crsp_end}")
        for y in range(2015, 2027):                                   # existing lists on their own screen year
            maps.append(permno_map(db, y - 1, ex.sym[ex.year == y], crsp_end).assign(role="existing", year=y))
        exm = pd.concat(maps)
        ex_permno = set(exm.permno.dropna().astype(int))
        ex_unmapped = sorted(set(exm.sym[exm.permno.isna()]))
        ex_tickers = set(ex.sym)
        log(f"existing lists 2015-2026: {len(ex)} rows, {ex.sym.nunique()} tickers, {len(ex_permno)} PERMNOs; "
            f"tickers without a PERMNO: {ex_unmapped}")
        out["existing"] = {"rows": int(len(ex)), "tickers": int(ex.sym.nunique()), "permnos": len(ex_permno),
                           "unmapped_tickers": ex_unmapped}
        for y in YEARS:
            x = y - 1
            p, med_price = passing(x)
            cm = permno_map(db, x, p, crsp_end).assign(role="candidate", year=y)
            maps.append(cm)
            k = cm.groupby("sym").permno.agg(lambda v: v.dropna().astype(int).unique().tolist())
            unique = {s: v[0] for s, v in k.items() if len(v) == 1}
            no_map = sorted(set(p) - set(unique))
            hit_permno = sorted(s for s, v in unique.items() if v in ex_permno)
            hit_ticker = sorted(s for s in unique if s in ex_tickers and s not in hit_permno)
            elig = sorted(s for s in unique if s not in set(hit_permno) | set(hit_ticker))
            order = [str(v) for v in np.random.default_rng(SEED).permutation(elig)]
            d = UR.table_days(db, y, "ctm")[0]
            syms = ",".join(f"'{s}'" for s in order)
            t = db.raw_sql(UR.CNT_T.format(y=y, d=d, ok=UR.TRADE_OK, syms=syms)).set_index("sym").n
            q = (db.raw_sql(UR.CNT_Q.format(y=y, d=d, ok=UR.QUOTE_OK, syms=syms)).set_index("sym").n
                 if d in set(UR.table_days(db, y, "complete_nbbo")) else pd.Series(dtype=int))
            o = pd.DataFrame({"order_rank": range(1, len(order) + 1), "sym": order})
            o["has_trades"] = o.sym.map(lambda s: bool(t.get(s, 0) > 0))
            o["has_quotes"] = o.sym.map(lambda s: bool(q.get(s, 0) > 0))
            ok = o.has_trades & o.has_quotes
            o["selected"] = ok & (ok.cumsum() <= n)
            o.insert(0, "seed", SEED)
            o.insert(1, "first_day", d)
            o.to_csv(os.path.join(HERE, f"universe17_draw_order_{y}.tsv"), sep="\t", index=False)
            u = o[o.selected][["sym", "order_rank"]].copy()
            u.insert(0, "year", y)
            u.insert(1, "slot", range(1, len(u) + 1))
            u.to_csv(os.path.join(HERE, f"universe17_{y}.tsv"), sep="\t", index=False)
            chosen = u.sym.tolist()
            skipped = o[(o.order_rank <= int(u.order_rank.max())) & ~o.selected].sym.tolist() if len(u) else []
            perm = [int(unique[s]) for s in chosen]
            sic = cm.drop_duplicates("sym").set_index("sym").siccd
            out["years"][y] = {
                "screen_year": x, "master": str(cm.master.iloc[0]), "crsp_date": str(cm.crsp_date.iloc[0]), "first_day": d, "passing": len(p),
                "no_unique_permno": len(no_map), "excluded_existing_permno": len(hit_permno),
                "excluded_existing_ticker_only": len(hit_ticker), "eligible": len(elig), "selected": chosen,
                "skipped_no_first_day_data": skipped, "overlap_permno": int(len(set(perm) & ex_permno)),
                "overlap_ticker": int(len(set(chosen) & ex_tickers)),
                "sic_division": pd.Series([sic_division(sic.get(s)) for s in chosen]).value_counts().to_dict(),
                "price_band_prior_year": {f"${a}-{b}": int(sum(a <= med_price[s] < b or (b == 150 and med_price[s] == 150)
                                                              for s in chosen)) for a, b in PRICE_BANDS}}
            log(f"universe {y} (screen {x}, master {cm.master.iloc[0]}, first day {d}, seed {SEED}): passing {len(p)}, "
                f"no unique PERMNO {len(no_map)}, existing PERMNO {len(hit_permno)}, existing ticker only {len(hit_ticker)}, "
                f"eligible {len(elig)} -> {chosen} | skipped for missing data on the first day: {skipped}")
    finally:
        db.close()
    os.makedirs(LOCAL, exist_ok=True)
    pd.concat(maps).to_csv(os.path.join(LOCAL, "permno_map.csv"), index=False)
    allsel = [(y, s) for y, v in out["years"].items() for s in v["selected"]]
    out["repeats_across_years"] = sorted({s for _, s in allsel if sum(1 for _, t in allsel if t == s) > 1})
    out["overlap_total"] = {"permno": sum(v["overlap_permno"] for v in out["years"].values()),
                            "ticker": sum(v["overlap_ticker"] for v in out["years"].values())}
    out["written"] = dt.datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "universe17_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    log(f"overlap with the existing lists: PERMNO {out['overlap_total']['permno']}, ticker {out['overlap_total']['ticker']}; "
        f"symbols drawn in more than one year: {out['repeats_across_years']}")
    if out["overlap_total"]["permno"] or out["overlap_total"]["ticker"]:
        raise SystemExit("overlap with the existing lists is not 0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if not a.go:
        sys.exit("refused: needs --go")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    lf = open(os.path.join(HERE, "logs", "universe17.log"), "a")

    def log(m):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {m}"
        print(line, flush=True)
        lf.write(line + "\n")
        lf.flush()
    main(a.n, log)
