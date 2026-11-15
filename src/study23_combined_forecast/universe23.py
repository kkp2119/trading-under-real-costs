#!/usr/bin/env python3
"""Research 23 step 1 (copy of study22_market_signal/universe22.py; changed lines marked "R23"): new universe D for 2021-2026
(SPEC23 1). R23: seed 23; the existing lists also hold research 22's universe C (study22_market_signal/universe22_<Y>.tsv, the 20
slots of every year, mapped on their own screen year like the other ticker lists); N is fixed at 20 (SPEC23 1), so
universe23_<Y>.tsv is written here. Research 22 docstring follows.
Research 22 step 1: new universe C for 2021-2026 (SPEC22 1), the procedure of research 17 and 18 with seed 22.
For each trading year Y = 2021..2026 (screen year X = Y - 1), as study18_forecast/universe18.py (whose functions are imported):
  passing   research 12's screen output for X (study12_market_maker/universe_screen_<X>.tsv, pass_all; same code and filters,
            checked against study12_market_maker/universe_draw_order_<X>.tsv). Only year-X information.
  PERMNO    X's last TAQ master CUSIP[:8] = crsp.stocknames ncusip on D = min(master date, last CRSP name date)
            (universe18.permno_map, unchanged).
  existing  every list of research 12-21, by PERMNO and, as a safeguard, by ticker: research 15 (2015-2018), research 12
            (2019-2026 stocks and its control ETFs), research 17 and 18 (2021-2026), mapped on their own screen year, and the
            legs of research 20 and 21 (2015-2026) with the PERMNO their selection recorded (universe21.existing plus
            $DATA_DIR/research21_data/selection/legs21.csv). Research 13 and 16 use the lists of research 12 and 15;
            research 19 uses no stock.
  eligible  passing minus every existing PERMNO and ticker; candidates without a unique PERMNO are left out and counted.
  order     numpy default_rng(22).permutation(eligible sorted by symbol), per year.
  slots     research 12's rule: the symbols of that order with at least one valid session trade and one valid session quote
            on Y's first ctm day (research 12 TRADE_OK / QUOTE_OK; counts only) take the slots 1, 2, ... in order. The lists
            for N = 20, 15, 10 are the first N slots (SPEC22 1: one draw order, cut from the front); n22.py writes
            universe22_<Y>.tsv for the N of the rule.
No signal, cost or return is computed. One WRDS connection (the WRDS credentials file): crsp.stocknames, TAQ master tables and the counts
of the first day; no bar is collected. Counts and tickers go to the log; CRSP fields stay in
$DATA_DIR/research22_data/selection/. Nothing is overwritten.
  python universe22.py --go
R23:  python universe23.py --go"""
import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for r in ("study12_market_maker", "study18_forecast", "study20_pairs", "study21_more_pairs"):
    sys.path.insert(0, os.path.join(PROJ, r))
import universe_rolling as UR  # noqa: E402  (SQL constants and table_days only; its main is never called)
import universe18 as U18  # noqa: E402  (passing, permno_map, sic_division, PRICE_BANDS; its main is never called)
import universe21 as U21  # noqa: E402  (existing: the lists of research 12, 15, 17, 18 and 20; its main is never called)

LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research23_data/selection")                             # R23
LEGS21 = os.path.join(os.environ.get("DATA_DIR", "data"), "research21_data/selection/legs21.csv")
YEARS = range(2021, 2027)
SEED = 23                                                                                    # R23
N_MAX, NS = 20, (20,)                                                                         # R23: N = 20 (SPEC23 1)
R22 = os.path.join(PROJ, "study22_market_signal")                                                        # R23


def outputs():
    return ([os.path.join(HERE, f"universe23_draw_order_{y}.tsv") for y in YEARS] + [os.path.join(HERE, f"universe23_{y}.tsv") for y in YEARS]
            + [os.path.join(HERE, "results", "universe23_summary.json"), os.path.join(LOCAL, "permno_map.csv")])        # R23


def main(log):
    done = [os.path.relpath(p, HERE) for p in outputs() if os.path.exists(p)]
    if done:
        raise SystemExit(f"outputs exist (nothing is overwritten): {done}")
    import wrds
    db = wrds.Connection()
    maps, out = [], {"seed": SEED, "n_max": N_MAX, "years": {}}
    try:
        crsp_end = str(db.raw_sql("select max(nameenddt) as e from crsp.stocknames").e.iloc[0])[:10]
        out["crsp_stocknames_end"] = crsp_end
        log(f"crsp.stocknames ends {crsp_end}")
        ex, exm, ex_permno, ex_tickers, _ = U21.existing(db, crsp_end)            # research 12 (stocks, ETFs), 15, 17, 18, 20
        l21 = pd.read_csv(LEGS21, dtype=str)
        ex = pd.concat([ex, l21[["year", "sym"]].astype({"year": int}).assign(source="r21")], ignore_index=True)
        ex_permno = set(ex_permno) | set(l21.permno.astype(int))
        c22 = pd.DataFrame([(y, s, "r22") for y in YEARS for s in pd.read_csv(os.path.join(R22, f"universe22_{y}.tsv"), sep="\t", dtype=str).sym],
                           columns=["year", "sym", "source"])                                 # R23: universe C, 20 slots a year
        m22 = pd.concat([U18.permno_map(db, y - 1, c22.sym[c22.year == y], crsp_end).assign(year=y) for y in YEARS])
        if len(c22) != 120 or (m22.groupby(["year", "sym"]).permno.agg(lambda v: v.dropna().nunique()) != 1).any():
            raise SystemExit("universe C: a symbol-year without exactly one PERMNO")
        ex = pd.concat([ex, c22], ignore_index=True)
        ex_permno |= set(m22.permno.dropna().astype(int))
        exm = pd.concat([exm, m22])
        ex_tickers = set(ex.sym)
        maps.append(exm.assign(role="existing"))
        mapped = set(exm.sym[exm.permno.notna()]) | set(ex.sym[ex.source.isin(["r20", "r21"])])
        nomap = sorted(set(ex.sym) - mapped)
        by_src = {k: int(g.sym.nunique()) for k, g in ex.groupby("source")}
        out["existing"] = {"rows": int(len(ex)), "tickers": int(ex.sym.nunique()), "permnos": len(ex_permno), "tickers_by_source": by_src,
                           "tickers_without_a_permno_in_any_year": nomap}
        log(f"existing lists (research 12 stocks and control ETFs, 15, 17, 18, 20, 21, 22): {len(ex)} rows, {ex.sym.nunique()} tickers, "
            f"{len(ex_permno)} PERMNOs; tickers by source {by_src}; tickers without a PERMNO: {nomap}")
        for y in YEARS:
            x = y - 1
            p, med_price = U18.passing(x)
            cm = U18.permno_map(db, x, p, crsp_end).assign(role="candidate", year=y)
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
            o["slot"] = np.where(ok & (ok.cumsum() <= N_MAX), ok.cumsum(), 0)            # 1..N_MAX, 0 = no slot
            o.insert(0, "seed", SEED)
            o.insert(1, "first_day", d)
            o.to_csv(os.path.join(HERE, f"universe23_draw_order_{y}.tsv"), sep="\t", index=False)        # R23
            u = o[o.slot > 0]
            if len(u) != N_MAX:                                                                   # R23
                raise SystemExit(f"{y}: {len(u)} slots, not {N_MAX}")
            v = u[["sym", "order_rank"]].copy()                                                   # R23: the list of the year
            v.insert(0, "year", y)
            v.insert(1, "slot", range(1, len(v) + 1))
            v.to_csv(os.path.join(HERE, f"universe23_{y}.tsv"), sep="\t", index=False)
            chosen = u.sym.tolist()
            skipped = o[(o.order_rank <= int(u.order_rank.max())) & (o.slot == 0)].sym.tolist() if len(u) else []
            perm = [int(unique[s]) for s in chosen]
            sic = cm.drop_duplicates("sym").set_index("sym").siccd
            out["years"][y] = {
                "screen_year": x, "master": str(cm.master.iloc[0]), "crsp_date": str(cm.crsp_date.iloc[0]), "first_day": d, "passing": len(p),
                "no_unique_permno": len(no_map), "excluded_existing_permno": len(hit_permno),
                "excluded_existing_ticker_only": len(hit_ticker), "eligible": len(elig),
                "selected": {str(n): chosen[:n] for n in NS}, "slots_filled": len(chosen),
                "skipped_no_first_day_data": skipped, "overlap_permno": int(len(set(perm) & ex_permno)),
                "overlap_ticker": int(len(set(chosen) & ex_tickers)),
                "sic_division": {str(n): pd.Series([U18.sic_division(sic.get(s)) for s in chosen[:n]]).value_counts().to_dict() for n in NS},
                "price_band_prior_year": {str(n): {f"${a}-{b}": int(sum(a <= med_price[s] < b or (b == 150 and med_price[s] == 150)
                                                                      for s in chosen[:n])) for a, b in U18.PRICE_BANDS} for n in NS}}
            log(f"universe D {y} (screen {x}, master {cm.master.iloc[0]}, first day {d}, seed {SEED}): passing {len(p)}, "
                f"no unique PERMNO {len(no_map)}, existing PERMNO {len(hit_permno)}, existing ticker only {len(hit_ticker)}, "
                f"eligible {len(elig)} -> {chosen} | skipped for missing data on the first day: {skipped}")
    finally:
        db.close()
    os.makedirs(LOCAL, exist_ok=True)
    pd.concat(maps).to_csv(os.path.join(LOCAL, "permno_map.csv"), index=False)
    allsel = [(y, s) for y, v in out["years"].items() for s in v["selected"][str(N_MAX)]]
    out["repeats_across_years"] = sorted({s for _, s in allsel if sum(1 for _, t in allsel if t == s) > 1})
    out["overlap_total"] = {"permno": sum(v["overlap_permno"] for v in out["years"].values()),
                            "ticker": sum(v["overlap_ticker"] for v in out["years"].values())}
    out["written"] = dt.datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "universe23_summary.json"), "w") as fh:           # R23
        json.dump(out, fh, indent=1, default=str)
    log(f"overlap with the existing lists: PERMNO {out['overlap_total']['permno']}, ticker {out['overlap_total']['ticker']}; "
        f"symbols drawn in more than one year: {out['repeats_across_years']}")
    if out["overlap_total"]["permno"] or out["overlap_total"]["ticker"]:
        raise SystemExit("overlap with the existing lists is not 0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if not a.go:
        sys.exit("refused: needs --go")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    lf = open(os.path.join(HERE, "logs", "universe23.log"), "a")                              # R23

    def log(m):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {m}"
        print(line, flush=True)
        lf.write(line + "\n")
        lf.flush()
    main(log)
