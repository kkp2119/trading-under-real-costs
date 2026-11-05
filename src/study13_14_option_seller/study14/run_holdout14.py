#!/usr/bin/env python3
"""Research 14 holdout run (study14/SPEC14.md, spec14-v1, sha256 77497023...). Only after Kyle types
"UNLOCK HOLDOUT 14" (study14/results14/HOLDOUT_UNLOCKED) and with --go. One run.

The research 13 frozen modules (code-freeze-13, manifest 58267d6c...) are imported, never changed. Order: start checks
(4) -> refusal conditions (7) -> phase 1: development days only, daily P of all 11 settings must equal research13
results/dev_daily.csv bit for bit (no holdout file opened before this passes) -> phase 2: holdout files, forced close
on the last day -> NDX and RUT (O2, O8, O9, descriptive) -> holdout audit (SPEC14 3) -> statistics -> all result files
written at the end. Start-check failures and refusals go to the screen only; during the run the screen and the log
carry the research 13 whitelist lines only.

Judgement (SPEC14 5): O2, O8 and O9 each, (a)/(b)/(c) on the Bonferroni 98.33 % percentile CI of the frozen bootstrap
(same resample for every setting), defect downgrade and tail label with each setting's own development values."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R13 = os.path.dirname(HERE)
sys.path.insert(0, R13)
import chain as ch  # noqa: E402
import config13 as C  # noqa: E402
import market  # noqa: E402
import omdata as om  # noqa: E402
import results13 as R  # noqa: E402
import run_dev  # noqa: E402
import run_holdout  # noqa: E402
import stats13  # noqa: E402

SPEC14_SHA = "77497023f66da1b0137bbb23c19810a4fbe9af977cd5df8058b87e1079e8e4c2"
JUDGED = ("O2", "O8", "O9")
LEVEL = 1 - 0.05 / 3                                       # Bonferroni, 3 settings: 0.8333 and 99.1667 percentiles
D_DEV = {"O2": 0.254073199527745, "O8": 0.2639905548996458, "O9": 0.2675324675324675}
MDD_DEV = {"O2": 0.12206986737589562, "O8": 0.07828371970127149, "O9": 0.044234858519156345}
DEV_SHA = {"dev_daily.csv": "b6822a0a00428959c046f21726f29957260f49b7c885de3a7305ae9728b5c7f8",
           "dev_summary.json": "568663f8d8b777fe799062fd2f82e53675ee88380a3e111587ce903b5f434136",
           "dev_settings.csv": "51fccd2a713c313136f6e8127ac9e481aabdf8bf1f2938fd608ae4a2e0a272d6"}
MANIFEST13 = ("CODE_FREEZE_MANIFEST.tsv", "58267d6c727b76ac4fed982b1b08a674f7d1a67d201b3e727a73750b30f4b162")
FROZEN = ("config13.py", "market.py", "signals.py", "ladder.py", "costs.py", "accounting.py", "stats13.py",
          "select13.py", "results13.py", "run_dev.py", "run_holdout.py", "chain.py", "omdata.py")
TAGS = ("spec14-v1", "code-freeze-14")
RESULTS14 = os.path.join(HERE, "results14")
DEV_RESULTS = os.path.join(R13, "results")
RESULT_FILES = ("holdout_summary14.json", "holdout_daily14.csv", "holdout_verdict14.json")
LOG_FILE = "holdout_run14.log"
AUX = ("NDX", "RUT")


# ---------------------------------------------------------------- checks before any holdout path
def frozen_check(r13=R13):
    """Frozen modules against the code-freeze-13 manifest (start check 4, refusal 5). Empty list = all equal."""
    man = os.path.join(r13, MANIFEST13[0])
    if not os.path.exists(man) or R.sha256(man) != MANIFEST13[1]:
        return ["code-freeze-13 manifest sha256 differs"]
    rows = {l.split("\t")[0]: l.split("\t")[2] for l in open(man).read().splitlines()[1:] if l.startswith("research13/")}
    return [f"{f} differs from the code-freeze-13 manifest" for f in FROZEN
            if R.sha256(os.path.join(r13, f)) != rows.get(f"research13/{f}")]


def startup_checks(dev_dir, frozen_fn=frozen_check):
    """SPEC14 9: the three development result files and the frozen modules, before the refusal conditions."""
    why = []
    for i, f in enumerate(("dev_daily.csv", "dev_summary.json", "dev_settings.csv"), 1):
        p = os.path.join(dev_dir, f)
        if not os.path.exists(p) or R.sha256(p) != DEV_SHA[f]:
            why.append(f"start check {i}: {f} sha256 differs")
    why += [f"start check 4: {x}" for x in frozen_fn()]
    return why


def has_tags():
    return all(run_holdout.git("tag", "-l", t).stdout.strip() == t for t in TAGS)


def guard(results_dir, has_tag_fn=has_tags, is_clean_fn=None, frozen_fn=frozen_check, data_fn=None):
    """Refusal reasons in SPEC14 9 order (empty list = may run). Data checks (6, 7) only after the unlock file."""
    is_clean_fn = is_clean_fn or (lambda: run_holdout.is_clean_at(R13))
    data_fn = data_fn or (lambda: run_holdout.data_checks(results_dir=results_dir))
    why = []
    if any(os.path.exists(os.path.join(results_dir, f)) for f in RESULT_FILES):
        why.append("1: holdout results exist")
    if not has_tag_fn():
        why.append("2: no spec14-v1 or code-freeze-14 tag")
    if not is_clean_fn():
        why.append("3: research13 working tree not clean")
    if not os.path.exists(os.path.join(results_dir, om.UNLOCK)):
        why.append("4: no HOLDOUT_UNLOCKED in results14")
        return why
    why += [f"5: {x}" for x in frozen_fn()]
    dw = data_fn()
    why += [x for x in dw if x.startswith("6:")]
    why += [x for x in dw if x.startswith("7:")]
    return why


# ---------------------------------------------------------------- judgement and statistics
def verdict(sid, r, defect_ratio):
    """SPEC14 5 for one judged setting. r: daily C-base returns of the holdout window (last day included)."""
    point = stats13.sharpe(r)
    lo, hi = stats13.ci(r, level=LEVEL)
    v, why = stats13.judge(point, lo, hi)
    v, why = stats13.downgrade(v, why, defect_ratio, D_DEV[sid])
    mdd = stats13.max_drawdown(r[:-1])
    return {"setting": sid, "verdict": v, "reason": why, "label": stats13.tail_label(v, mdd, MDD_DEV[sid]),
            "sharpe": point, "ci_98_33": [lo, hi], "ci_95": list(stats13.ci(r)), "defect_ratio": defect_ratio,
            "downgrade_threshold": max(C.DEFECT_FLOOR, D_DEV[sid] + C.DEFECT_MARGIN), "max_drawdown": mdd,
            "tail_boundary": C.TAIL_RATIO * MDD_DEV[sid]}


def cost_share(df):
    po, pb = df.P_opt.sum(), df.P_base.sum()
    return float((po - pb) / po) if po != 0 else float("nan")


def setting_report(df, mret, dev_df=None):
    """SPEC14 6 for one setting's ledger frame: research 13 holdout metrics, DSR N = 11 and 180, F4 value, years,
    cost shares and the ledger audit items of SPEC14 3."""
    w = df[(df.date >= C.HOLD_START) & (df.date <= C.HOLD_END)]
    m = run_dev.setting_metrics(w, C.HOLD_START, C.HOLD_END, mret)
    r = w.P_base.to_numpy() / C.N0
    t = stats13.tails(r[:-1])                                     # tails without the data-end day
    m["tails"], m["calmar"], m["max_drawdown"] = t, t["calmar"], t["max_drawdown"]
    m["ci_95"], m["ci_98_33"] = list(stats13.ci(r)), list(stats13.ci(r, level=LEVEL))
    m["ci_sens"] = {b: {"95": list(stats13.ci(r, block=b)), "98_33": list(stats13.ci(r, level=LEVEL, block=b))}
                    for b in C.SENS_BLOCKS}
    m["dsr"] = {n: stats13.deflated_sharpe(r, n) for n in (11, 180)}
    m["F4_calmar_ge_0.33"] = bool(np.isfinite(t["calmar"]) and t["calmar"] >= C.CALMAR_MIN)
    m["last_day_return"] = float(r[-1])
    m["ann_net"] = float(r.mean() * C.ANN)
    m["years"] = {int(y): float(g.P_base.sum() / C.N0) for y, g in w.groupby(w.date // 10000)}
    m["cost_share"] = cost_share(w)
    if dev_df is not None:
        m["cost_share_dev"] = cost_share(dev_df)
    m["audit"] = {"defect_i_days": int(w.defect_i.sum()), "defect_ii_days": int(w.defect_ii.sum()),
                  "gap_days": int((w.sel_status == "gap").sum()), "no_monthly_days": int((w.sel_status == "no_monthly").sum()),
                  "no_candidate_days": int((w.sel_status == "no_candidate").sum()),
                  "u_gt_0.2_days": int((w.u > C.DEFECT_U).sum()), "fwd_spot_leg_days": int(w.fwd_spot.sum()),
                  "curve_prev_days": int(w.curve_prev.sum()), "exdate_moved_leg_days": int(w.exdate_moved.sum())}
    return m


def audit_pass(src, days):
    """SPEC14 3 / A13-3 change 1(a): chain-based structural gap days against gaps on the schedule of all standard
    exdates seen in the holdout, for both DTE ranges. Reads the holdout files again after the run; reports only."""
    src.enable_holdout()
    mkt = market.Market(src, "SPX", ("dev", "holdout"))
    exs = [mkt.day(d).std_exdates for d in days]
    allx = np.array(sorted({int(e) for ex in exs for e in ex}), dtype=np.int64)
    out = {}
    for H in sorted(C.RANGE):
        lo, hi, _ = C.RANGE[H]
        st = [ch.range_status(d, ex, lo, hi) for d, ex in zip(days, exs)]
        sched = []
        for d in days:
            x = ch.dte(np.full(len(allx), d), allx)
            sched.append(not ((x >= lo) & (x <= hi)).any())
        gap = np.array([s == "gap" for s in st])
        out[H] = {"chain_gap_days": int(gap.sum()), "no_monthly_days": int(sum(s == "no_monthly" for s in st)),
                  "schedule_gap_days": int(sum(sched)), "disagreements": int((gap != np.array(sched)).sum())}
    return {"days": len(days), "by_H": out, "standard_exdates": len(allx), "read_paths": len(set(src.paths())),
            "holdout_paths": sum("data_holdout" in p for p in src.paths())}


# ---------------------------------------------------------------- run
def main(argv=None, source_factory=None, results_dir=None, dev_results_dir=None, has_tag_fn=has_tags,
         is_clean_fn=None, frozen_fn=frozen_check, data_fn=None, startup_fn=startup_checks, aux=AUX, ext_dir=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    rdir, ddir = results_dir or RESULTS14, dev_results_dir or DEV_RESULTS
    screen = run_holdout.Log(None, strict=True)                   # start checks and refusals: screen only
    if not a.go:
        screen("refused: needs --go after Kyle types UNLOCK HOLDOUT 14")
        return None
    why = startup_fn(ddir, frozen_fn)
    if not why:
        why = guard(rdir, has_tag_fn, is_clean_fn, frozen_fn, data_fn)
    if why:
        for w in why:
            screen(f"refused: {w}")
        return None
    log = run_holdout.Log(os.path.join(rdir, LOG_FILE), strict=True)
    factory = source_factory or (lambda: market.RealSource(results_dir=rdir))
    log("phase 1 start")
    mkt, eng = run_holdout.run_und(factory(), "SPX", C.SETTINGS, log, ddir, check_dev=True)
    aux_runs = {u: run_holdout.run_und(factory(), u, [C.setting(s) for s in JUDGED], log) for u in aux}
    hold = [d for d in mkt.days if C.HOLD_START <= d <= C.HOLD_END]
    audit = audit_pass(factory(), hold)
    # ---- statistics (nothing below is printed; it goes to the result files only)
    mret = run_dev.market_returns(mkt, hold)
    dev = pd.read_csv(os.path.join(ddir, "dev_daily.csv"), float_precision="round_trip")
    s0 = run_dev.stats_start(dev.date.tolist())
    dev = dev[(dev.date >= s0) & (dev.date <= C.DEV_END)]
    frames = {sid: e.ledger.frame() for sid, e in eng.items()}
    out = {}
    for sid, df in frames.items():
        dsid = dev[["date"]].assign(P_base=dev[f"{sid}_base"], P_opt=dev[f"{sid}_opt"])
        out[sid] = setting_report(df, mret, dsid)
    verdicts = {}
    for sid in JUDGED:
        w = frames[sid][(frames[sid].date >= C.HOLD_START) & (frames[sid].date <= C.HOLD_END)]
        verdicts[sid] = verdict(sid, w.P_base.to_numpy() / C.N0, out[sid]["defect_ratio"])
    rets = pd.DataFrame({sid: frames[sid].set_index("date").loc[hold, "P_base"].to_numpy() for sid in JUDGED})
    aux_out = {}
    for u, (am, ae) in aux_runs.items():
        aux_out[u] = {"data_fixes": am.fixes["holdout"], "metrics": {}}
        for sid, e in ae.items():
            df = e.ledger.frame()
            r = df[(df.date >= C.HOLD_START) & (df.date <= C.HOLD_END)].P_base.to_numpy() / C.N0
            t = stats13.tails(r[:-1])
            aux_out[u]["metrics"][sid] = {"sharpe": stats13.sharpe(r), "ann_net": float(r.mean() * C.ANN),
                                          "max_drawdown": t["max_drawdown"], "calmar": t["calmar"]}
    daily = pd.DataFrame({"date": hold})
    for sid, df in frames.items():
        d = df.set_index("date")
        for sc in C.SCEN:
            daily[f"{sid}_{sc}"] = d.loc[hold, f"P_{sc}"].to_numpy()
    summary = {"spec14_sha256": SPEC14_SHA, "judged": list(JUDGED), "level": LEVEL,
               "bounds": {sid: {"d_dev": D_DEV[sid], "max_drawdown_dev": MDD_DEV[sid]} for sid in JUDGED},
               "dev_files_sha256": DEV_SHA, "code_freeze_13_manifest": MANIFEST13[1],
               "run_holdout14_sha256": R.sha256(os.path.abspath(__file__)), "metrics": out, "aux": aux_out,
               "audit": audit, "data_fixes": mkt.fixes["holdout"], "iv_out_of_range_days": len(mkt.iv_out_of_range),
               "rv21_filled_signal_days": len(mkt.rv21_filled),
               "judged_correlation": rets.corr().round(6).to_dict(),
               "spx_buy_hold_sharpe": stats13.sharpe(mret.dropna().to_numpy()), "env": R.env_versions()}
    vals = [x["verdict"] for x in verdicts.values()]
    verdict_doc = {"verdicts": verdicts, "level": LEVEL, "at_least_one_a": "a" in vals, "all_c": set(vals) == {"c"},
                   "at_least_one_b": "b" in vals}
    rid = R.run_id()
    R.write_csv(daily, os.path.join(rdir, "holdout_daily14.csv"))
    R.write_json(summary, os.path.join(rdir, "holdout_summary14.json"))
    R.write_json(verdict_doc, os.path.join(rdir, "holdout_verdict14.json"))
    for sid, e in eng.items():
        R.write_external(pd.DataFrame(e.ledger.legs), f"holdout14_legs_{sid}.csv", "holdout14", rid, rdir, ext_dir)
    log("results written")
    return verdict_doc


if __name__ == "__main__":
    main()
