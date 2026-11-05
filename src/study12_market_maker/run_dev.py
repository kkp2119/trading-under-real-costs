#!/usr/bin/env python3
"""Research 12 development run (SPEC 5, 6, 9, 12, 16; 17 step 3). Opens the dev folders only. Pass 0 day statistics ->
gates; stage 1 (B0 + 18, with A_naive) -> adoption -> stage 2 -> SPEC 6 selection -> ETF controls -> mix step -> G1 ->
filter placebo (B5) -> J-A count / J-B P&L / J-C stress on last-trade days (E1-25) -> result files, then a short print.
Not to be run on the real bars before code-freeze-12 and Kyle's separate permission."""
import argparse
import dataclasses
import datetime as dt
import os

import numpy as np
import pandas as pd

import accounting
import grid
import loader
import results
import runner
import select_candidate as sc
import session
from config import A_NAIVE, B0, B_ETF, CALENDAR, EXTERNAL, HERE, MARKOUT_K, MODELS, RESULTS, SCENARIOS
from judge import gate_g1
from placebo import N_PLACEBO, PLACEBO_DIMS
from stats import regress, sharpe


class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, msg):
        self.f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")
        self.f.flush()


def prepare(split, asset, base, results_dir, cal, universe_dir):
    """(Split, tasks, gates, day stats, availability)."""
    sp = loader.Split(split, asset, base=base, results_dir=results_dir)
    avail = sp.available()
    tasks = session.tasks(cal, split, asset, avail, universe_dir)
    _, days = runner.run_tasks(sp, cal, tasks, {k: True for k in tasks}, [], [], n_workers=1)
    dv = {(r.sym, r.date): int(r.dv) for r in days.itertuples()}
    zero = {(r.sym, r.date) for r in days.itertuples() if r.no_data}
    gates = {(s, d): session.gate(cal, s, d, dv, zero, avail) for s, d in tasks}
    return sp, tasks, gates, days, avail


def grid_table(runs, settings, obs):
    """dev_grid.csv: per setting x model x scenario net Sharpe, yearly Sharpe, fill rate, participation, markouts."""
    rows = []
    years = sorted({d[:4] for d in obs})
    for name, cfg in settings:
        for m in MODELS:
            r = runs[(runs.setting == name) & (runs.model == m)]
            if not len(r):
                continue
            row = {"setting": name, "cid": cfg.cid(), "model": m, **{k: v for k, v in cfg.as_dict().items()}}
            for scn in SCENARIOS:
                x = sc.daily(runs, name, m, scn, obs)
                row[f"sharpe_{scn}"] = sharpe(x)
                for y in years:
                    row[f"sharpe_{scn}_{y}"] = sharpe(x[[d[:4] == y for d in obs]])
            live = r[r.skip == 0]
            n_int = live.n_int_b.sum() + live.n_int_s.sum()
            row["interval_fill_rate"] = float((live.n_intf_b.sum() + live.n_intf_s.sum()) / n_int) if n_int else float("nan")
            row["fills_per_symday"] = float(live.n_fill_ev.mean()) if len(live) else float("nan")
            part = ((live.sh_lim_buy + live.sh_lim_sell + live.sh_mkt_buy + live.sh_mkt_sell) / live.mkt_vol)[live.mkt_vol > 0]
            row["participation"] = float(part.mean()) if len(part) else float("nan")
            for k in MARKOUT_K:
                sh = live[f"mk{k}_sh"].sum()
                row[f"markout{k}_ticks"] = float(live[f"mk{k}_sum2"].sum() / 2 / sh / 100) if sh else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def daily_table(runs, settings, obs):
    out = {"date": obs}
    for name, _ in settings:
        for m in MODELS:
            for scn in SCENARIOS:
                out[f"{name}|{m}|{scn}"] = sc.daily(runs, name, m, scn, obs)
    return pd.DataFrame(out)


def placebo_table(sp, cal, tasks, gates, n_perm, log):
    """B5 (E1-40): stage-1 OBI and FI settings on the dev stocks, Q and R; per permutation net per filled share."""
    cols = {}
    for dim in PLACEBO_DIMS:
        cfg = dataclasses.replace(B0, **{dim: True})
        tot = np.zeros(n_perm)
        sh = np.zeros(n_perm)
        for s, d in tasks:
            for row in runner.run_placebo_day(sp.load(s, d), cal.close[d], gates[(s, d)], cfg, dim, s, d, range(n_perm)):
                tot[row["r"]] += row["net_R"]
                sh[row["r"]] += row["lim_sh"]
        cols[dim] = np.where(sh > 0, tot / 1e4 / np.maximum(sh, 1), np.nan)
        log(f"placebo {dim}: {n_perm} permutations")
    return pd.DataFrame({"r": np.arange(n_perm), **{f"{k}_net_per_share": v for k, v in cols.items()}})


def last_trade_days(runs, days, cal, name):
    """E1-25: J-A (days the SPEC rule would skip), their J-B net R, and the J-C stress (flatten at the next day's first
    valid mid instead of the SPEC price), descriptive only."""
    r = runs[(runs.setting == name) & (runs.model == "Q") & (runs.ja_skip == 1) & (runs.skip == 0)]
    first = {(x.sym, x.date): x.first_mid2 for x in days.itertuples()}
    order = cal.order
    stress = 0.0
    for x in r.itertuples():
        if x.flat_qty:
            i = order.index(x.date)
            nxt = next((first[(x.sym, d)] for d in order[i + 1:] if (x.sym, d) in first and first[(x.sym, d)] > 0), None)
            if nxt is not None:
                stress += (-x.flat_side) * x.flat_qty * (nxt / 2 - x.flat_price) / 1e4
    return {"ja_days": int(len(r)), "jb_net_R": float(accounting.net_arrays(r, "R").sum() / 1e4) if len(r) else 0.0,
            "jc_stress_delta": float(stress)}


def record_rerun(S, E, cal, st_tasks, st_gates, et_tasks, et_gates, runs, runs_e, stock_settings, n_workers):
    """E1-66: once the candidate is known, rerun the recorded settings (model Q) in record mode; the run rows must be
    bit-identical to the grid's rows. Returns (records, identical)."""
    names = [n for n, _ in stock_settings]
    rs, _, rec = runner.run_tasks_record(S, cal, st_tasks, st_gates, stock_settings, ["Q"], n_workers, names, names)
    re_, _, rec_e = runner.run_tasks_record(E, cal, et_tasks, et_gates, [("B_etf", B_ETF)], ["Q"], n_workers, ["B_etf"],
                                            ["B_etf"])
    same = True
    for new, old, ns in ((rs, runs, names), (re_, runs_e, ["B_etf"])):
        ref = old[(old.model == "Q") & old.setting.isin(ns)].sort_values(["setting", "model", "sym", "date"], kind="stable")
        same &= new.reset_index(drop=True).equals(ref[new.columns].reset_index(drop=True))
    return {**rec, **rec_e}, bool(same)


def side_numbers(runs, setting, days, obs):
    """E1-69 (SPEC 10): market proxy regression, yearly and per-symbol net (Q, R) of one setting."""
    x = sc.daily(runs, setting, "Q", "R", obs)
    return {"regression": regress(x, results.market_proxy(days, obs)), "yearly_net_R": results.yearly(x, obs),
            "per_symbol_net_R": results.per_symbol(runs, setting)}


def main(base=None, results_dir=RESULTS, universe_dir=HERE, calendar=CALENDAR, n_workers=2, n_perm=N_PLACEBO,
         ext_dir=EXTERNAL):
    log = Log(os.path.join(results_dir, "run_dev.log"))
    rid = results.run_id()
    log(f"start {rid} {results.env_versions()}")
    cal = session.Calendar(calendar)
    S, st_tasks, st_gates, st_days, st_avail = prepare("dev", "stocks", base, results_dir, cal, universe_dir)
    E, et_tasks, et_gates, et_days, et_avail = prepare("dev", "etf", base, results_dir, cal, universe_dir)
    obs = session.observation_days(cal, "dev", st_avail)
    obs_etf = session.observation_days(cal, "dev", et_avail)
    s1 = grid.stage1()
    names1 = [n for n, _ in s1] + ["A_naive"]
    runs1, _ = runner.run_tasks(S, cal, st_tasks, st_gates, s1 + [("A_naive", A_NAIVE)], MODELS, n_workers, names1)
    m1 = {c: sc.setting_metrics(runs1, n, obs) for n, c in s1}
    adopt = grid.adopted({c: m["sharpe"] for c, m in m1.items()})
    s2 = grid.stage2(adopt, set(m1))
    log(f"stage 1 done; adopted {adopt}; stage 2 {[n for n, _ in s2]}")
    runs2 = runner.run_tasks(S, cal, st_tasks, st_gates, s2, MODELS, n_workers, [n for n, _ in s2])[0] if s2 else runs1.iloc[:0]
    runs = pd.concat([runs1, runs2], ignore_index=True)
    settings = s1 + s2
    metrics = {c: sc.setting_metrics(runs, n, obs) for n, c in settings}
    names = {c: n for n, c in settings}
    best, table = sc.select(metrics, names)
    out = {"adopted": {k: list(v) if isinstance(v, tuple) else v for k, v in adopt.items()}, "filter_table": table,
           "candidate": None, "final": None, "mix": None, "g1": None, "env": results.env_versions()}
    etf_settings = grid.etf_controls(best)
    runs_e, _ = runner.run_tasks(E, cal, et_tasks, et_gates, etf_settings + [("A_naive", A_NAIVE)], MODELS, n_workers,
                                 [n for n, _ in etf_settings])
    n_trials = grid.n_trials([c for _, c in settings], [c for _, c in etf_settings], mix=best is not None)
    out["n_trials"] = n_trials
    if best is not None:
        bname = names[best]
        out["candidate"] = {"name": bname, "cid": best.cid(), "config": best.as_dict(), **metrics[best],
                            "neighbors_checked": next(t["neighbors_checked"] for t in table if t["name"] == bname)}
        mix_runs = pd.concat([runs[runs.setting == bname].assign(setting="MIX"),
                              runs_e[runs_e.setting == "B_etf"].assign(setting="MIX")], ignore_index=True)
        mix_m = sc.setting_metrics(mix_runs, "MIX", sorted(set(obs) | set(obs_etf)))
        mixd = sc.mix_decision(metrics[best], mix_m)
        out["mix"] = {"metrics": mix_m, "decision": mixd, "etf_setting": B_ETF.as_dict()}
        final_runs = mix_runs if mixd["mix_wins"] else runs[runs.setting == bname]
        final_name = "MIX" if mixd["mix_wins"] else bname
        out["final"] = {"type": "mix" if mixd["mix_wins"] else "stock", "stock_setting": bname,
                        "other_side": "stock" if mixd["mix_wins"] else "mix"}
        q = final_runs[(final_runs.model == "Q") & (final_runs.skip == 0)]
        dev_sr = (mix_m if mixd["mix_wins"] else metrics[best])["sharpe"]
        flat = bool(((final_runs.sh_lim_buy + final_runs.sh_mkt_buy) == (final_runs.sh_lim_sell + final_runs.sh_mkt_sell)).all())
        universe_syms = {s for y in sorted({d[:4] for d in obs}) for s in session.universe(int(y), "stocks", universe_dir)}
        if mixd["mix_wins"]:
            universe_syms |= set(session.universe(0, "etf"))
        out["g1"] = gate_g1(dev_sr, bool((q.nested != 0).all()), float(q.n_dec_qb.sum() / max(q.n_dec.sum(), 1)), flat,
                            set(q[q.n_fill_ev > 0].sym), universe_syms)
        out["last_trade_days"] = last_trade_days(runs, st_days, cal, bname)
        out["side_numbers"] = side_numbers(final_runs, final_name, st_days, sorted(set(obs) | (set(obs_etf) if
                                                                                          mixd["mix_wins"] else set())))
    rec_settings = [(names[best], best)] if best is not None else []
    rec_settings += [(n, c) for n, c in (("B0", B0), ("A_naive", A_NAIVE)) if all(c != x for _, x in rec_settings)]
    records, identical = record_rerun(S, E, cal, st_tasks, st_gates, et_tasks, et_gates, runs, runs_e, rec_settings,
                                      n_workers)
    log(f"record rerun of {[n for n, _ in rec_settings] + ['B_etf']}: identical to the grid = {identical}")
    out["placebo"] = placebo_table(S, cal, st_tasks, st_gates, n_perm, log).to_dict("list")
    out["placebo_actual"] = {}
    for dim in PLACEBO_DIMS:                                         # B5: the real filter's value and its percentile
        r = runs[(runs.setting == f"{dim}=True") & (runs.model == "Q")]
        sh = (r.sh_lim_buy + r.sh_lim_sell).sum()
        actual = float(accounting.net_arrays(r, "R").sum() / 1e4 / sh) if sh else float("nan")
        perm = np.asarray(out["placebo"][f"{dim}_net_per_share"], dtype=float)
        out["placebo_actual"][dim] = {"net_per_share": actual,
                                      "percentile": float(np.mean(perm[np.isfinite(perm)] < actual) * 100)
                                      if np.isfinite(perm).any() and actual == actual else float("nan")}
    all_settings = settings + [("A_naive", A_NAIVE)]
    results.write_csv(grid_table(runs, all_settings, obs).assign(universe="stocks"), os.path.join(results_dir, "dev_grid.csv"))
    results.write_csv(grid_table(runs_e, etf_settings + [("A_naive", A_NAIVE)], obs_etf).assign(universe="etf"),
                      os.path.join(results_dir, "dev_grid_etf.csv"))
    results.write_csv(daily_table(runs, all_settings, obs), os.path.join(results_dir, "dev_daily.csv"))
    keep = runs[(runs.model == "Q") | runs.setting.isin([out["candidate"]["name"] if best is not None else "B0", "B0", "A_naive"])]
    out["external"] = {"dev_fills": results.external_csv(keep, "dev_fills.csv", ext_dir, "dev", rid, results_dir),  # E1-43, 67
                       **results.external_records(records, "dev", ext_dir, "dev", rid, results_dir)}
    results.write_csv(pd.DataFrame(out["placebo"]), os.path.join(results_dir, "dev_placebo.csv"))
    checks = {"skipped_symbol_days": runs[runs.model == "Q"].groupby("skip").size().to_dict(),
              "gate_failed": int(sum(not g for g in st_gates.values())), "observation_days": len(obs),
              "absent_listed": len(S.absent), "mkt_over_display": int(runs.mkt_over_disp.sum()),
              "mkt_on_stale_quote": int(runs.mkt_stale_quote.sum()),
              "flat_after_close": bool(((runs.sh_lim_buy + runs.sh_mkt_buy) == (runs.sh_lim_sell + runs.sh_mkt_sell)).all()),
              "record_rerun_identical": identical}
    results.write_json(checks, os.path.join(results_dir, "dev_checks.json"))
    results.write_json(out, os.path.join(results_dir, "dev_summary.json"))
    sc.write(out, os.path.join(results_dir, "selected_config.json"))
    log("files written")
    if not identical:
        raise SystemExit("record rerun differs from the grid rows (dev_checks.json): a defect to fix before anything else")
    print(f"dev run done: candidate {out['candidate']['name'] if out['candidate'] else None}, "
          f"G1 {out['g1']['pass'] if out['g1'] else None}, N {n_trials} (see {results_dir})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="only after code-freeze-12 and Kyle's permission")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if not a.go:
        raise SystemExit("refused: the dev run needs --go (after code-freeze-12 and Kyle's permission)")
    main(n_workers=a.workers)
