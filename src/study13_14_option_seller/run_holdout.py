#!/usr/bin/env python3
"""Research 13 holdout run (SPEC 11; A13-3 E13-13, E13-17, E13-18, change 6; memo 4b v2 8). Only after Kyle types
"UNLOCK HOLDOUT 13" (results/HOLDOUT_UNLOCKED) and with --go. One run.

Order: refusal conditions (7) -> phase 1: development days only, daily P must equal results/dev_daily.csv bit for bit
(no holdout file opened before this passes) -> phase 2: holdout files, days 2013-01-02..end with the forced close on the
last day -> statistics -> all result files written at the end. During phase 2 the screen and the log carry progress
lines only (whitelist); no P&L, return, statistic or defect ratio appears before the result files are written."""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

import config13 as C
import ladder
import market
import omdata as om
import results13 as R
import run_dev
import stats13

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_FILES = ("holdout_summary.json", "holdout_daily.csv", "holdout_verdict.json")
WHITELIST = [r"^phase [12] (start|done)$", r"^phase 1 check: development days identical \(\d+ days, \d+ settings\)$",
             r"^holdout day \d{8} \(\d+/\d+\)$", r"^(SPX|NDX|RUT) (phase [12] )?(start|done)$",
             r"^refused: .+$", r"^stop: invariant .+$", r"^results written$"]


class Log:
    def __init__(self, path=None, strict=False):
        self.path, self.strict, self.lines = path, strict, []

    def __call__(self, msg):
        if self.strict and not any(re.match(p, msg) for p in WHITELIST):
            raise RuntimeError("log line outside the holdout whitelist (not written)")
        self.lines.append(msg)
        print(msg, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(msg + "\n")


def git(*a):
    return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True)


def has_tag(name="dev-result-13"):
    return git("tag", "-l", name).stdout.strip() == name


def is_clean_at(path):
    """Research 11 rule: `git -C <study> status --porcelain -- .` prints nothing."""
    r = subprocess.run(["git", "-C", path, "status", "--porcelain", "--", "."], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == ""


def is_clean():
    return is_clean_at(HERE)


def data_checks(base=None, results_dir=None):
    """Refusal conditions 6 and 7 (called only after the unlock check): repository manifest copy equal to the data
    folder MANIFEST, every holdout file equal to its MANIFEST sha256, end date 20250829 in SPLIT.json and its copy."""
    why = []
    root = om.split_root("holdout", base)
    rep = os.path.join(HERE, "data_manifests", "holdout")
    for f in ("MANIFEST.tsv", "SPLIT.json"):
        if R.sha256(os.path.join(root, f)) != R.sha256(os.path.join(rep, f)):
            why.append(f"6: {f} differs from the repository copy")
    st = om.Store("holdout", base=base, results_dir=results_dir)
    bad = [k for k, r in st.meta.items() if R.sha256(os.path.join(root, k)) != r["sha256"]]
    if bad:
        why.append(f"6: {len(bad)} holdout files differ from MANIFEST")
    sp = json.load(open(os.path.join(root, "SPLIT.json")))
    if int(sp.get("holdout_end_date", 0)) != C.HOLD_END:
        why.append("7: holdout end date differs")
    return why


def guard(results_dir, has_tag_fn=has_tag, is_clean_fn=is_clean, data_fn=data_checks):
    """Refusal reasons in SPEC 11 order (empty list = may run)."""
    why = []
    if any(os.path.exists(os.path.join(results_dir, f)) for f in RESULT_FILES):
        why.append("1: holdout results exist")
    if not has_tag_fn():
        why.append("2: no dev-result-13 tag")
    if not is_clean_fn():
        why.append("3: working tree not clean")
    if not os.path.exists(os.path.join(results_dir, om.UNLOCK)):
        why.append("4: no HOLDOUT_UNLOCKED")
        return why
    sel = os.path.join(results_dir, "selected_config.json")
    if not os.path.exists(sel) or not (json.load(open(sel)).get("candidate") or {}).get("setting"):
        why.append("5: no selected setting")
    why += data_fn()
    return why


def dev_identical(eng, results_dir, days):
    ref = pd.read_csv(os.path.join(results_dir, "dev_daily.csv"), float_precision="round_trip")   # exact floats
    if ref.date.tolist() != days:
        return False
    for sid, e in eng.items():
        df = e.ledger.frame()
        for sc in C.SCEN:
            if not np.array_equal(df[f"P_{sc}"].to_numpy(), ref[f"{sid}_{sc}"].to_numpy()):
                return False
    return True


def run_und(src, und, settings, log, results_dir=None, check_dev=False):
    mkt = market.Market(src, und, ("dev",))
    log(f"{und} phase 1 start")
    eng = ladder.run(mkt, settings, C.DEV_START, C.DEV_END)
    dev_days = [d for d in mkt.days if C.DEV_START <= d <= C.DEV_END]
    if check_dev:
        if any(isinstance(a[2], (int, np.integer)) and a[2] > C.DEV_END // 10000 for a in src.access) or \
                any(a[2] == "holdout" for a in src.access):
            raise RuntimeError("stop: invariant holdout file read in phase 1")
        if not dev_identical(eng, results_dir, dev_days):
            raise RuntimeError("stop: invariant development days differ from dev_daily.csv (E13-13)")
        log(f"phase 1 check: development days identical ({len(dev_days)} days, {len(settings)} settings)")
    src.enable_holdout()
    mkt.add_split("holdout")
    hold = [d for d in mkt.days if C.HOLD_START <= d <= C.HOLD_END]
    log(f"{und} phase 2 start")
    eng = ladder.run(mkt, settings, C.HOLD_START, C.HOLD_END, forced_last_day=True,
                     entry_on_last_day=C.ENTRY_ON_LAST_DAY, engines=eng,
                     on_day=lambda d: log(f"holdout day {d} ({hold.index(d) + 1}/{len(hold)})") if d == hold[-1] else None)
    for e in eng.values():
        e.ledger.check_identity(e.liability())
    log(f"{und} phase 2 done")
    return mkt, eng


def main(argv=None, source_factory=None, results_dir=None, has_tag_fn=has_tag, is_clean_fn=is_clean,
         data_fn=data_checks, aux=("NDX", "RUT")):
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    rdir = results_dir or R.RESULTS
    log = Log(os.path.join(rdir, "holdout_run.log") if os.path.isdir(rdir) else None, strict=True)
    if not a.go:
        log("refused: needs --go after Kyle types UNLOCK HOLDOUT 13")
        return None
    why = guard(rdir, has_tag_fn, is_clean_fn, data_fn)
    if C.ENTRY_ON_LAST_DAY is None:
        why.append("N2 undecided (entries on the data-end day)")
    if C.RV21_FILL_MISSING_CLOSE is None:
        why.append("N3 undecided (missing close inside the RV21 window)")
    if why:
        for w in why:
            log(f"refused: {w}")
        return None
    sel = json.load(open(os.path.join(rdir, "selected_config.json")))["candidate"]
    factory = source_factory or (lambda: market.RealSource(results_dir=rdir))
    log("phase 1 start")
    mkt, eng = run_und(factory(), "SPX", C.SETTINGS, log, rdir, check_dev=True)
    aux_eng, fixes = {}, {"SPX": mkt.fixes["holdout"]}                      # A13-4 8d counts (holdout files)
    for u in aux:
        am, e = run_und(factory(), u, [C.setting(sel["setting"])], log)
        aux_eng[u], fixes[u] = e, am.fixes["holdout"]
    # ---- statistics (nothing below is printed; it goes to the result files only)
    hold = [d for d in mkt.days if C.HOLD_START <= d <= C.HOLD_END]
    mret = run_dev.market_returns(mkt, hold)
    out = {}
    for sid, e in eng.items():
        df = e.ledger.frame()
        w = df[(df.date >= C.HOLD_START) & (df.date <= C.HOLD_END)]
        m = run_dev.setting_metrics(w, C.HOLD_START, C.HOLD_END, mret)
        r = w.P_base.to_numpy() / C.N0
        t = stats13.tails(r[:-1])                                     # 6, 11: tails without the data-end day
        m["tails"], m["calmar"], m["max_drawdown"] = t, t["calmar"], t["max_drawdown"]
        m["ci"] = stats13.ci(r)
        m["ci_sens"] = {b: stats13.ci(r, block=b) for b in C.SENS_BLOCKS}
        m["defect_ratio"] = stats13.defect_ratio(w)
        m["last_day_return"] = float(r[-1])
        out[sid] = m
    s = sel["setting"]
    v, why = stats13.judge(out[s]["sharpe"], *out[s]["ci"])
    v, why = stats13.downgrade(v, why, out[s]["defect_ratio"], sel["d_dev"])
    verdict = {"setting": s, "verdict": v, "reason": why,
               "label": stats13.tail_label(v, out[s]["max_drawdown"], sel["max_drawdown"])}
    daily = pd.DataFrame({"date": hold})
    for sid, e in eng.items():
        df = e.ledger.frame().set_index("date")
        for sc in C.SCEN:
            daily[f"{sid}_{sc}"] = df.loc[hold, f"P_{sc}"].to_numpy()
    aux_out = {}
    for u, e in aux_eng.items():
        df = e.ledger.frame()
        w = df[(df.date >= C.HOLD_START) & (df.date <= C.HOLD_END)]
        aux_out[u] = {"sharpe": stats13.sharpe(w.P_base.to_numpy() / C.N0)}
    R.write_csv(daily, os.path.join(rdir, "holdout_daily.csv"))
    R.write_json({"metrics": out, "aux": aux_out, "iv_out_of_range_days": len(mkt.iv_out_of_range),
                  "rv21_filled_signal_days": len(mkt.rv21_filled), "data_fixes": fixes,
                  "exdate_moved_leg_days": {sid: int(sum(r["exdate_moved"] for r in e.ledger.rows
                                                         if r["date"] >= C.HOLD_START)) for sid, e in eng.items()},
                  "env": R.env_versions()}, os.path.join(rdir, "holdout_summary.json"))
    R.write_json(verdict, os.path.join(rdir, "holdout_verdict.json"))
    log("results written")
    return verdict


if __name__ == "__main__":
    main()
