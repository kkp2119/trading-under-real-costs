#!/usr/bin/env python3
"""Research 26: research 12's Market Maker development run on 2015-2020 (SPEC26 2, 3, 5; DEVIATIONS26 1).
research 12's engine is imported and never edited: features, simulator, fill models, accounting, grid (stage 1 = B0 + 18,
adoption at B0 + 0.5, stage 2), selection filters (i)-(iv) and rule, gate G1, placebo permutations, result helpers.
What this file adds: the window, the two bar folders and the two kinds of stock list (data26), no ETF controls and no
mix step (no ETF bars for 2015-2018), a parallel run of the same placebo permutations (integer sums, so the result equals
research 12's serial loop), the trial count of this study, and the SPEC26 5 tables.
Filter (i) needs no new code: research 12's setting_metrics makes one yearly Sharpe per year of the observation days and
filter (i) requires every one of them to be > 0, so with 2015-2020 it is the six years.
  python run26.py --regress [--workers 4]   window 2019-2020 only, no placebo: must equal research 12's development
                                            results (grid, selection table, trial count 63, daily P&L); writes
                                            results/regress26/ and stops (exit 1) on any difference
  python run26.py --go [--workers 4]        the development run on 2015-2020, once (refuses when its results exist)
Results: results/ (aggregates); per symbol-day tables and fill logs: $DATA_DIR/research26_data/results_external/."""
import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data26 as D  # noqa: E402  (puts study12_market_maker on the path)

D.register()
import accounting  # noqa: E402
import grid  # noqa: E402
import results  # noqa: E402
import run_dev as RD  # noqa: E402  (research 12's driver: only its table helpers are used)
import runner  # noqa: E402
import select_candidate as sc  # noqa: E402
import session  # noqa: E402
from config import A_NAIVE, B0, MARKOUT_K, MODELS, SCENARIOS, SPLITS  # noqa: E402
from judge import gate_g1  # noqa: E402
from placebo import N_PLACEBO, PLACEBO_DIMS  # noqa: E402
from stats import sharpe  # noqa: E402

RES = os.path.join(HERE, "results")
EXT = os.path.join(D.DATA26, "results_external")
R12_RES = os.path.join(D.R12, "results")
TRIALS_BEFORE = 865
SENTENCES = {"none": "Market Maker는 2015–2020 개발 구간에서도 비용 뒤 돈을 버는 설정이 없었다. 봉인 데이터는 열지 않았다",
             "g1": "2015–2020에서 고른 설정 {name}이 G1을 통과했다. 4절의 봉인 시험으로 넘어간다"}
REPORT_MODELS = ("Q", "S", "T")


class Log(RD.Log):
    pass


# ---------------------------------------------------------------- data and gates (research 12's prepare, two folders)
def prepare(window, cal, results_dir, sources=None, list_dirs=(D.R15, D.R12)):
    sp = D.TwoFolders(SPLITS[window], sources=sources, results_dir=results_dir)
    avail = sp.available()
    tasks = D.tasks(cal, window, avail, list_dirs)
    _, days = runner.run_tasks(sp, cal, tasks, {k: True for k in tasks}, [], [], n_workers=1)
    dv = {(r.sym, r.date): int(r.dv) for r in days.itertuples()}
    zero = {(r.sym, r.date) for r in days.itertuples() if r.no_data}
    gates = {(s, d): session.gate(cal, s, d, dv, zero, avail) for s, d in tasks}
    return sp, tasks, gates, days, avail


# ---------------------------------------------------------------- placebo: research 12's B5 permutations, in parallel
def _placebo_work(a):
    sp, sym, day, close_bar, gate, dim, n_perm = a
    cfg = dataclasses.replace(B0, **{dim: True})
    tot, sh = [0] * n_perm, [0] * n_perm
    for row in runner.run_placebo_day(sp.load(sym, day), close_bar, gate, cfg, dim, sym, day, range(n_perm)):
        tot[row["r"]] += int(row["net_R"])
        sh[row["r"]] += int(row["lim_sh"])
    return tot, sh


def placebo_table(sp, cal, tasks, gates, n_perm, n_workers, log):
    """Same numbers as research 12's run_dev.placebo_table: per dimension and permutation the sum of net R and of limit
    shares over every symbol-day (exact integers), then net per filled share."""
    cols = {}
    for dim in PLACEBO_DIMS:
        jobs = [(sp, s, d, cal.close[d], gates[(s, d)], dim, n_perm) for s, d in tasks]
        if n_workers > 1:
            with mp.get_context("spawn").Pool(n_workers) as pool:
                res = pool.map(_placebo_work, jobs, chunksize=max(1, len(jobs) // (8 * n_workers)))
        else:
            res = [_placebo_work(j) for j in jobs]
        tot = np.array([sum(r[0][i] for r in res) for i in range(n_perm)], dtype=np.float64)
        sh = np.array([sum(r[1][i] for r in res) for i in range(n_perm)], dtype=np.float64)
        cols[dim] = np.where(sh > 0, tot / 1e4 / np.maximum(sh, 1), np.nan)
        log(f"placebo {dim}: {n_perm} permutations")
    return pd.DataFrame({"r": np.arange(n_perm), **{f"{k}_net_per_share": v for k, v in cols.items()}})


def record_rerun(S, cal, tasks, gates, runs, rec_settings, n_workers):
    """research 12's E1-66 check on the stocks (no ETF here): rerun in record mode, rows must be bit-identical."""
    names = [n for n, _ in rec_settings]
    rs, _, rec = runner.run_tasks_record(S, cal, tasks, gates, rec_settings, ["Q"], n_workers, names, names)
    ref = runs[(runs.model == "Q") & runs.setting.isin(names)].sort_values(["setting", "model", "sym", "date"], kind="stable")
    return rec, bool(rs.reset_index(drop=True).equals(ref[rs.columns].reset_index(drop=True)))


# ---------------------------------------------------------------- SPEC26 5 tables (no verdict)
def daily_net(runs, name, model, obs, scenario="R"):
    return sc.daily(runs, name, model, scenario, obs)


def yearly_table(runs, names, obs):
    """Per setting, fill model (Q, S, T), period (each year, 2015-2018, 2019-2020, all): net P&L (R, dollars), net Sharpe,
    shares traded and net per share (cents)."""
    rows = []
    years = sorted({d[:4] for d in obs})
    periods = [(y, {y}) for y in years] + [("2015-2018", {y for y in years if y <= "2018"}), ("2019-2020", {y for y in years if y >= "2019"}),
                                           ("all", set(years))]
    for name in names:
        for m in REPORT_MODELS:
            x = daily_net(runs, name, m, obs)
            r = runs[(runs.setting == name) & (runs.model == m)]
            yr = r.date.str[:4]
            for lab, ys in periods:
                if not ys:
                    continue
                mask = np.array([d[:4] in ys for d in obs])
                rr = r[yr.isin(ys)]
                sh = int((rr.sh_lim_buy + rr.sh_lim_sell + rr.sh_mkt_buy + rr.sh_mkt_sell).sum())
                net = float(x[mask].sum())
                rows.append({"setting": name, "model": m, "period": lab, "days": int(mask.sum()), "net_R_usd": net,
                             "sharpe_R": sharpe(x[mask]), "losing_days": int((x[mask] < 0).sum()), "shares": sh,
                             "limit_shares": int((rr.sh_lim_buy + rr.sh_lim_sell).sum()), "net_per_share_c": net / sh * 100 if sh else float("nan")})
    return pd.DataFrame(rows)


def cumulative_table(runs, best, obs):
    """Daily cumulative net P&L (R, dollars) for the presentation: B0 Q, best Q, B0 T (upper bound), best T."""
    out = {"date": obs}
    for name, m in (("B0", "Q"), (best, "Q"), ("B0", "T"), (best, "T")):
        out[f"{name}|{m}"] = np.cumsum(daily_net(runs, name, m, obs))
    return pd.DataFrame(out)


def markout_table(runs, names, obs):
    """research 12's markouts of limit fills (k = 1, 5, 30, 60 s, ticks; negative = adverse selection) per setting, fill
    model and year, and the 5-s markout by time of day (first hour, 11:30-12:30, last hour)."""
    rows = []
    years = sorted({d[:4] for d in obs})
    for name in names:
        for m in REPORT_MODELS:
            r = runs[(runs.setting == name) & (runs.model == m) & (runs.skip == 0)]
            for lab in years + ["all"]:
                rr = r if lab == "all" else r[r.date.str[:4] == lab]
                row = {"setting": name, "model": m, "period": lab, "limit_shares": int((rr.sh_lim_buy + rr.sh_lim_sell).sum())}
                for k in MARKOUT_K:
                    s = rr[f"mk{k}_sh"].sum()
                    row[f"mk{k}_ticks"] = float(rr[f"mk{k}_sum2"].sum() / 2 / s / 100) if s else float("nan")
                for b in ("open", "noon", "close"):
                    s = rr[f"{b}_mk5_sh"].sum()
                    row[f"{b}_mk5_ticks"] = float(rr[f"{b}_mk5_sum2"].sum() / 2 / s / 100) if s else float("nan")
                rows.append(row)
    return pd.DataFrame(rows)


def best_setting(metrics, names, best):
    """The selected setting, or the stock setting with the highest dev net Sharpe (Q, R) when there is none (ties: the
    first in run order)."""
    if best is not None:
        return names[best]
    ok = [(m["sharpe"], -i, names[c]) for i, (c, m) in enumerate(metrics.items()) if m["sharpe"] == m["sharpe"]]
    return max(ok)[2] if ok else "B0"


def compare_r12_days(daily, obs):
    """SPEC26 5: the 2019-2020 part of this run against research 12's development daily file (every column both hold)."""
    p = os.path.join(R12_RES, "dev_daily.csv")
    old = pd.read_csv(p, dtype={"date": str})
    new = daily[daily.date.isin(set(old.date))].reset_index(drop=True)
    cols = [c for c in new.columns if c != "date" and c in old.columns]
    old = old.set_index("date").loc[new.date, cols].reset_index(drop=True)
    diff = (new[cols] - old).abs().max().max() if cols else float("nan")
    return {"days": int(len(new)), "columns_compared": len(cols), "max_abs_diff_usd": float(diff),
            "columns_only_here": [c for c in daily.columns if c not in cols and c != "date"][:50]}


# ---------------------------------------------------------------- the run
def main(window="dev26", results_dir=RES, ext_dir=EXT, n_workers=4, n_perm=N_PLACEBO, calendar=None, sources=None,
         list_dirs=(D.R15, D.R12), run_placebo=True, compare_r12=True):
    prefix = window
    out_summary = os.path.join(results_dir, f"{prefix}_summary.json")
    if os.path.exists(out_summary):
        raise SystemExit(f"{out_summary} exists: the run is made once (nothing is overwritten)")
    t0 = time.time()
    log = Log(os.path.join(results_dir, f"{prefix}.log"))
    rid = results.run_id()
    log(f"start {rid} window {SPLITS[window]} workers {n_workers} {results.env_versions()}")
    cal = D.Calendar(calendar or D.build_calendar())
    S, tasks, gates, days, avail = prepare(window, cal, results_dir, sources, list_dirs)
    obs = session.observation_days(cal, window, avail)
    absent_listed = D.missing(cal, window, avail, list_dirs)
    s1 = grid.stage1()
    names1 = [n for n, _ in s1] + ["A_naive"]
    runs1, _ = runner.run_tasks(S, cal, tasks, gates, s1 + [("A_naive", A_NAIVE)], MODELS, n_workers, names1)
    m1 = {c: sc.setting_metrics(runs1, n, obs) for n, c in s1}
    adopt = grid.adopted({c: m["sharpe"] for c, m in m1.items()})
    s2 = grid.stage2(adopt, set(m1))
    log(f"stage 1 done; adopted {adopt}; stage 2 {[n for n, _ in s2]}")
    runs2 = runner.run_tasks(S, cal, tasks, gates, s2, MODELS, n_workers, [n for n, _ in s2])[0] if s2 else runs1.iloc[:0]
    runs = pd.concat([runs1, runs2], ignore_index=True)
    settings = s1 + s2
    metrics = {c: sc.setting_metrics(runs, n, obs) for n, c in settings}
    names = {c: n for n, c in settings}
    best, table = sc.select(metrics, names)
    stock_cfgs = [c for _, c in settings]
    out = {"window": list(SPLITS[window]), "adopted": {k: list(v) if isinstance(v, tuple) else v for k, v in adopt.items()},
           "filter_table": table, "candidate": None, "final": None, "g1": None, "env": results.env_versions(),
           "etf_and_mix": "not run (no ETF bars for 2015-2018; SPEC26 1)",
           "n_trials_research12_formula": grid.n_trials(stock_cfgs, [c for _, c in grid.etf_controls(best)], mix=best is not None),
           "trials": {"before": TRIALS_BEFORE, "added": len(set(stock_cfgs) - {A_NAIVE}) + 1,
                      "cumulative": TRIALS_BEFORE + len(set(stock_cfgs) - {A_NAIVE}) + 1,
                      "rule": "distinct stock settings run + A_naive (research 12's count without the ETF controls and the mix, which are not run)"}}
    if best is not None:
        bname = names[best]
        out["candidate"] = {"name": bname, "cid": best.cid(), "config": best.as_dict(), **metrics[best],
                            "neighbors_checked": next(t["neighbors_checked"] for t in table if t["name"] == bname)}
        out["final"] = {"type": "stock", "stock_setting": bname}
        fr = runs[runs.setting == bname]
        q = fr[(fr.model == "Q") & (fr.skip == 0)]
        flat = bool(((fr.sh_lim_buy + fr.sh_mkt_buy) == (fr.sh_lim_sell + fr.sh_mkt_sell)).all())
        syms = {s for y in sorted({d[:4] for d in obs}) for s in D.universe(int(y), list_dirs)}
        out["g1"] = gate_g1(metrics[best]["sharpe"], bool((q.nested != 0).all()), float(q.n_dec_qb.sum() / max(q.n_dec.sum(), 1)), flat,
                            set(q[q.n_fill_ev > 0].sym), syms)
        out["last_trade_days"] = RD.last_trade_days(runs, days, cal, bname)
        out["side_numbers"] = RD.side_numbers(fr, bname, days, obs)
    g1_pass = bool(out["g1"] and out["g1"]["pass"])
    out["result"] = "G1 통과" if g1_pass else "후보 없음 또는 G1 불통과"
    out["sentence"] = SENTENCES["g1"].format(name=names[best]) if g1_pass else SENTENCES["none"]
    rec_settings = [(names[best], best)] if best is not None else []
    rec_settings += [(n, c) for n, c in (("B0", B0), ("A_naive", A_NAIVE)) if all(c != x for _, x in rec_settings)]
    records, identical = record_rerun(S, cal, tasks, gates, runs, rec_settings, n_workers)
    log(f"record rerun of {[n for n, _ in rec_settings]}: identical to the grid = {identical}")
    if run_placebo:
        pl = placebo_table(S, cal, tasks, gates, n_perm, n_workers, log)
        out["placebo"] = pl.to_dict("list")
        out["placebo_actual"] = {}
        for dim in PLACEBO_DIMS:
            r = runs[(runs.setting == f"{dim}=True") & (runs.model == "Q")]
            sh = (r.sh_lim_buy + r.sh_lim_sell).sum()
            actual = float(accounting.net_arrays(r, "R").sum() / 1e4 / sh) if sh else float("nan")
            perm = np.asarray(out["placebo"][f"{dim}_net_per_share"], dtype=float)
            out["placebo_actual"][dim] = {"net_per_share": actual, "percentile": float(np.mean(perm[np.isfinite(perm)] < actual) * 100)
                                          if np.isfinite(perm).any() and actual == actual else float("nan")}
    else:
        out["placebo"] = None
        out["placebo_note"] = "not run (regression: SPEC26 step-1 prompt 3 allows leaving it out)"
    all_settings = settings + [("A_naive", A_NAIVE)]
    grid_df = RD.grid_table(runs, all_settings, obs).assign(universe="stocks")
    daily_df = RD.daily_table(runs, all_settings, obs)
    bname_rep = best_setting(metrics, names, best)
    out["report_setting"] = {"name": bname_rep, "why": "selected" if best is not None else "highest dev net Sharpe (Q, R); no candidate"}
    yearly = yearly_table(runs, ["B0", bname_rep] if bname_rep != "B0" else ["B0"], obs)
    marks = markout_table(runs, ["B0", bname_rep] if bname_rep != "B0" else ["B0"], obs)
    cum = cumulative_table(runs, bname_rep, obs)
    if compare_r12 and window == "dev26":
        out["compare_research12_2019_2020"] = compare_r12_days(daily_df, obs)
    b0 = yearly[(yearly.setting == "B0") & (yearly.model == "Q")].set_index("period")
    bs = yearly[(yearly.setting == "B0") & (yearly.model == "S")].set_index("period")
    ys = [p for p in b0.index if p.isdigit()]
    out["predictions_check"] = [
        {"item": "선택 후보가 G1을 통과함", "probability": 0.03, "outcome": g1_pass},
        {"item": "B0(Q, R)가 2015–2020의 여섯 해 모두 손실", "probability": 0.85, "outcome": bool(len(ys) == 6 and (b0.loc[ys, "net_R_usd"] < 0).all()),
         "by_year_usd": {y: float(b0.loc[y, "net_R_usd"]) for y in ys}},
        {"item": "2015–2018만 본 B0(Q, R)의 순 Sharpe < 0", "probability": 0.90,
         "outcome": bool(b0.loc["2015-2018", "sharpe_R"] < 0) if "2015-2018" in b0.index else None,
         "value": float(b0.loc["2015-2018", "sharpe_R"]) if "2015-2018" in b0.index else None},
        {"item": "줄 맨 앞(S) B0의 6년 순손익 < 0", "probability": 0.80, "outcome": bool(bs.loc["all", "net_R_usd"] < 0), "value": float(bs.loc["all", "net_R_usd"])}]
    keep = runs[(runs.model == "Q") | runs.setting.isin([bname_rep, "B0", "A_naive"])]
    out["external"] = {"fills": results.external_csv(keep, f"{prefix}_fills.csv", ext_dir, prefix, rid, results_dir),
                       **results.external_records(records, prefix, ext_dir, prefix, rid, results_dir)}
    out["checks"] = {"symbol_days": len(tasks), "skipped_symbol_days": runs[runs.model == "Q"].groupby("skip").size().to_dict(),
                     "gate_failed": int(sum(not g for g in gates.values())), "observation_days": len(obs),
                     "absent_listed": len(S.absent), "list_symbol_days_without_file": len(absent_listed),
                     "list_symbol_days_without_file_by_symbol": pd.Series([f"{d[:4]} {s}" for s, d in absent_listed]).value_counts().to_dict(),
                     "mkt_over_display": int(runs.mkt_over_disp.sum()), "mkt_on_stale_quote": int(runs.mkt_stale_quote.sum()),
                     "flat_after_close": bool(((runs.sh_lim_buy + runs.sh_mkt_buy) == (runs.sh_lim_sell + runs.sh_mkt_sell)).all()),
                     "record_rerun_identical": identical, "bar_folders": [r.replace(os.path.expanduser("~"), "~") for r in S.roots()]}
    out["runtime_s"] = round(time.time() - t0, 1)
    results.write_csv(grid_df, os.path.join(results_dir, f"{prefix}_grid.csv"))
    results.write_csv(daily_df, os.path.join(results_dir, f"{prefix}_daily.csv"))
    results.write_csv(yearly, os.path.join(results_dir, f"{prefix}_yearly.csv"))
    results.write_csv(marks, os.path.join(results_dir, f"{prefix}_markouts.csv"))
    results.write_csv(cum, os.path.join(results_dir, f"{prefix}_cumulative.csv"))
    if out["placebo"] is not None:
        results.write_csv(pd.DataFrame(out["placebo"]), os.path.join(results_dir, f"{prefix}_placebo.csv"))
    results.write_json(out, out_summary)
    log("files written")
    if not identical:
        raise SystemExit("record rerun differs from the grid rows: a defect to fix before anything else")
    print(f"{prefix}: candidate {out['candidate']['name'] if out['candidate'] else None}, G1 {g1_pass}, settings {len(settings)}, "
          f"trials {out['trials']['before']} -> {out['trials']['cumulative']}, {out['runtime_s']} s (see {results_dir})")
    return out


# ---------------------------------------------------------------- regression against research 12 (SPEC26 step-1 prompt 3)
def _json(x):
    return json.loads(json.dumps(results._clean(x), ensure_ascii=False, default=str))


def _max_abs_diff(a, b):
    """Largest absolute difference over the numeric (not bool) columns both frames hold, rows in the same order."""
    num = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and not pd.api.types.is_bool_dtype(a[c])
           and pd.api.types.is_numeric_dtype(b[c]) and not pd.api.types.is_bool_dtype(b[c])]
    if len(a) != len(b) or not num:
        return float("nan")
    return float((a[num].reset_index(drop=True) - b[num].reset_index(drop=True)).abs().max().max())


def regression(results_dir, n_workers):
    """2019-2020 through this file's path: grid, selection table, adopted values, candidate, trial count (research 12's
    formula) and the daily P&L must equal research 12's development results. Placebo left out.
    The run is made once; when its files exist (regress26_summary.json), only the comparison is made, from the files
    (both sides read back from what was written, as research 12's side is)."""
    rd = os.path.join(results_dir, "regress26")
    if not os.path.exists(os.path.join(rd, "regress26_summary.json")):
        main("regress26", results_dir=rd, ext_dir=os.path.join(D.DATA26, "regress26_external"), n_workers=n_workers,
             run_placebo=False, compare_r12=False)
    out = json.load(open(os.path.join(rd, "regress26_summary.json")))
    s12 = json.load(open(os.path.join(R12_RES, "dev_summary.json")))
    g12 = pd.read_csv(os.path.join(R12_RES, "dev_grid.csv"), dtype={"cid": str})
    g26 = pd.read_csv(os.path.join(rd, "regress26_grid.csv"), dtype={"cid": str})
    g12 = g12[g12.universe == "stocks"].reset_index(drop=True)
    d12 = pd.read_csv(os.path.join(R12_RES, "dev_daily.csv"), dtype={"date": str})
    d26 = pd.read_csv(os.path.join(rd, "regress26_daily.csv"), dtype={"date": str})
    gmax = _max_abs_diff(g12, g26)
    dmax = _max_abs_diff(d12, d26) if d12.date.equals(d26.date) else float("nan")
    chk = {
        "grid_rows": [len(g26), len(g12)], "grid_columns_equal": list(g12.columns) == list(g26.columns),
        "grid_equal": bool(g12.equals(g26)), "grid_max_abs_diff": gmax,
        "selection_table_equal": _json(out["filter_table"]) == s12["filter_table"], "settings_in_selection_table": len(out["filter_table"]),
        "adopted_equal": _json(out["adopted"]) == s12["adopted"], "candidate": out["candidate"], "candidate_equal": out["candidate"] == s12["candidate"],
        "n_trials_research12_formula": out["n_trials_research12_formula"], "n_trials_research12": s12["n_trials"],
        "daily_columns_equal": list(d12.columns) == list(d26.columns), "daily_equal": bool(d12.equals(d26)), "daily_max_abs_diff_usd": dmax,
        "b0_daily_equal": bool(all(d12[c].equals(d26[c]) for c in d12.columns if c.startswith("B0|"))),
        "record_rerun_identical": out["checks"]["record_rerun_identical"], "placebo": "left out (time); not compared"}
    chk["pass"] = bool(chk["grid_equal"] and chk["grid_columns_equal"] and chk["selection_table_equal"] and chk["adopted_equal"]
                       and chk["candidate_equal"] and chk["n_trials_research12_formula"] == chk["n_trials_research12"] == 63
                       and chk["daily_equal"] and chk["b0_daily_equal"] and chk["record_rerun_identical"])
    results.write_json(chk, os.path.join(results_dir, "regress26.json"))
    print(json.dumps({k: v for k, v in chk.items() if k != "candidate"}, indent=1, ensure_ascii=False))
    if not chk["pass"]:
        raise SystemExit("regression against research 12 failed: stop")
    return chk


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--go", action="store_true", help="the development run on 2015-2020, once, after Kyle's word")
    g.add_argument("--regress", action="store_true", help="2019-2020 against research 12's development results")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.regress:
        regression(RES, a.workers)
    else:
        main("dev26", n_workers=a.workers)
