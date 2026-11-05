#!/usr/bin/env python3
"""Research 12 single holdout run (SPEC 11, 16, 17 step 4), then the reserve (descriptive). Every refusal condition of
SPEC 17 step 4 is checked before anything is computed or written:
  results already exist (holdout*, reserve*, holdout_COMPLETED) | no tag dev-selected-12 | work tree not clean |
  no HOLDOUT_UNLOCKED | G1 not passed (or no candidate) in selected_config.json | a bar file differs from its MANIFEST
  hash | the holdout data end before 2026-09-30.
Clean tree (Kyle 2026-10-27, as research 11): `git -C research12 status --porcelain -- .` prints nothing (logs, probe/,
HOLDOUT_UNLOCKED and large result files are in .gitignore; large files live outside the repository, E1-67)."""
import argparse
import glob
import json
import os
import subprocess

import pandas as pd

import loader
import results
import runner
import select_candidate as sc
import session
import stats
from config import (A_NAIVE, B0, B_ETF, CALENDAR, COMPLETED, Config, EXTERNAL, HERE, MODELS, RESULTS, SCENARIOS, SPLITS,
                    UNLOCK)
from judge import side_lines, verdict

TAG = "dev-selected-12"
RESULT_GLOBS = ("holdout*", "reserve*", COMPLETED)


def git(*a, path=HERE):
    return subprocess.run(["git", "-C", path, *a], capture_output=True, text=True)


def has_tag(tag=TAG):
    return bool(git("tag", "-l", tag).stdout.strip())


def is_clean(path=HERE):
    """No modified tracked file and no untracked file outside .gitignore under the study folder (research 11 rule)."""
    r = git("status", "--porcelain", "--", ".", path=path)
    return r.returncode == 0 and r.stdout.strip() == ""


def guard(results_dir=RESULTS, has_tag_fn=has_tag, is_clean_fn=is_clean, splits_fn=None):
    """List of refusal reasons (empty = may run). splits_fn() -> [loader.Split] is called only after the unlock check."""
    why = []
    if any(glob.glob(os.path.join(results_dir, g)) for g in RESULT_GLOBS):
        why.append("holdout results already exist")
    if not has_tag_fn():
        why.append(f"no tag {TAG}")
    if not is_clean_fn():
        why.append("work tree not clean")
    if not os.path.exists(os.path.join(results_dir, UNLOCK)):
        why.append(f"no {UNLOCK}")
        return why
    sel = os.path.join(results_dir, "selected_config.json")
    cfg = json.load(open(sel)) if os.path.exists(sel) else {}
    if not (cfg.get("g1") or {}).get("pass") or not cfg.get("candidate"):
        why.append("G1 not passed or no candidate in selected_config.json")
    if splits_fn is not None and not why:
        for sp in splits_fn():
            bad = sp.verify_all()
            if bad:
                why.append(f"{sp.root}: {len(bad)} files differ from MANIFEST")
            if sp.split == "holdout" and max(d for _, d in sp.available()) < SPLITS["holdout"][1]:
                why.append(f"{sp.root}: data end before {SPLITS['holdout'][1]}")
    return why


def run_split(split, cand, base, results_dir, cal, universe_dir, n_workers):
    """Final candidate (stock or mix), the other side, B0 and A_naive on the split, every recorded (E1-66, E1-68).
    Returns (runs, obs days, records, stock day stats)."""
    S = loader.Split(split, "stocks", base=base, results_dir=results_dir)
    E = loader.Split(split, "etf", base=base, results_dir=results_dir)
    out, obs_all, records, stock_days = [], set(), {}, None
    for sp, settings in ((S, [("cand", cand), ("B0", B0), ("A_naive", A_NAIVE)]), (E, [("B_etf", B_ETF)])):
        avail = sp.available()
        prev = set()
        if split == "reserve":                                       # gate history from the holdout days before
            prev = loader.Split("holdout", sp.asset, base=base, results_dir=results_dir).available()
        tasks = session.tasks(cal, split, sp.asset, avail, universe_dir)
        _, days = runner.run_tasks(sp, cal, tasks, {k: True for k in tasks}, [], [], n_workers=1)
        dv = {(r.sym, r.date): int(r.dv) for r in days.itertuples()}
        zero = {(r.sym, r.date) for r in days.itertuples() if r.no_data}
        gates = {(s, d): session.gate(cal, s, d, dv, zero, avail | prev) for s, d in tasks}
        names = [n for n, _ in settings]
        runs, _, rec = runner.run_tasks_record(sp, cal, tasks, gates, settings, MODELS, n_workers, record=names)
        records.update(rec)
        stock_days = days if sp.asset == "stocks" else stock_days
        out.append(runs.assign(universe=sp.asset))
        obs_all |= set(session.observation_days(cal, split, avail))
    runs = pd.concat(out, ignore_index=True)
    mix = pd.concat([runs[runs.setting == "cand"], runs[runs.setting == "B_etf"]]).assign(setting="MIX")
    runs = pd.concat([runs, mix], ignore_index=True)
    return runs, sorted(obs_all), records, stock_days


def main(base=None, results_dir=RESULTS, universe_dir=HERE, calendar=CALENDAR, n_workers=2, guard_kw=None, ext_dir=EXTERNAL):
    sel = json.load(open(os.path.join(results_dir, "selected_config.json"))) if os.path.exists(
        os.path.join(results_dir, "selected_config.json")) else {}
    why = guard(results_dir, splits_fn=lambda: [loader.Split("holdout", a, base=base, results_dir=results_dir)
                                                for a in ("stocks", "etf")], **(guard_kw or {}))
    if why:
        raise SystemExit("refused: " + "; ".join(why))
    cal = session.Calendar(calendar)
    cand = Config(**sel["candidate"]["config"])
    final = "MIX" if sel["final"]["type"] == "mix" else "cand"
    other = "cand" if final == "MIX" else "MIX"
    rid = results.run_id()
    runs, obs, records, sdays = run_split("holdout", cand, base, results_dir, cal, universe_dir, n_workers)
    x_daily = sc.daily(runs, final, "Q", "R", obs)
    x = stats.sharpe(x_daily)
    a, b = stats.ci(x_daily)
    mde = stats.mde(len(obs))
    code, sentence = verdict(x, a, b, sel["candidate"]["cid"], sel["n_trials"], mde, len(obs) / 252)
    model_sr = {f"{m}|{s}": stats.sharpe(sc.daily(runs, final, m, s, obs)) for m in MODELS for s in SCENARIOS}
    q = runs[(runs.setting == final) & (runs.model == "Q")]
    net_r = q.pipe(lambda r: (r.cash - 30 * (r.sh_mkt_buy + r.sh_mkt_sell)).sum())
    be = float(-net_r / 1e4 / max((q.sh_lim_buy + q.sh_lim_sell).sum(), 1))
    dev_sr = (sel["mix"]["metrics"] if final == "MIX" else sel["candidate"])["sharpe"]
    summary = {"verdict": code, "sentence": sentence, "sharpe": x, "ci95": [a, b], "mde": mde, "n_days": len(obs),
               "dsr": stats.deflated_sharpe(x_daily, sel["n_trials"]), "summary_R": stats.summary(x_daily),
               "model_scenario_sharpe": model_sr, "breakeven_rebate": be,
               "other_side_sharpe": stats.sharpe(sc.daily(runs, other, "Q", "R", obs)),
               "side_lines": side_lines(a, dev_sr, x, model_sr, be, stats.sharpe(sc.daily(runs, "B0", "Q", "R", obs)),
                                        stats.sharpe(sc.daily(runs, "A_naive", "Q", "R", obs))),
               "regression": stats.regress(x_daily, results.market_proxy(sdays, obs)),
               "yearly_net_R": results.yearly(x_daily, obs), "per_symbol_net_R": results.per_symbol(runs, final),
               "env": results.env_versions()}
    summary["external"] = {"holdout_fills": results.external_csv(runs, "holdout_fills.csv", ext_dir, "holdout", rid,
                                                                 results_dir),
                           **results.external_records(records, "holdout", ext_dir, "holdout", rid, results_dir)}
    results.write_csv(pd.DataFrame({"date": obs, **{f"{n}|{m}|{s}": sc.daily(runs, n, m, s, obs) for n in
                                                    (final, other, "B0", "A_naive") for m in MODELS for s in SCENARIOS}}),
                      os.path.join(results_dir, "holdout_daily.csv"))
    results.write_json(summary, os.path.join(results_dir, "holdout_summary.json"))
    rruns, robs, rrec, _ = run_split("reserve", cand, base, results_dir, cal, universe_dir, n_workers)
    results.write_json({"n_days": len(robs), "summary_R": stats.summary(sc.daily(rruns, final, "Q", "R", robs)),
                        "external": results.external_records(rrec, "reserve", ext_dir, "reserve", rid, results_dir)},
                       os.path.join(results_dir, "reserve_summary.json"))
    open(os.path.join(results_dir, COMPLETED), "w").close()
    print(sentence)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    main(n_workers=ap.parse_args().workers)
