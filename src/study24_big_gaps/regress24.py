#!/usr/bin/env python3
"""Research 24 regression test (step-2 prompt 1; SPEC24_ANSWERS A3): research 24's code path (loader24.load ->
core24.simulate with pair24.trades -> core24.run_row) with 50,000 dollars per leg and delay 0 on the development data of
research 20 and of research 21 must reproduce their stored development numbers with a largest difference of exactly 0:
  research 20   results/dev20_summary.json: the 8 M candidates under L3D1 and L0D0 (research 20's development plan, its
                grid and its bootstrap seed 20)
  research 21   results/dev21_summary.json: families E and S, the 9 signal settings with market orders under L3D1 and L0D0
                (research 21's development plan and seed 21), and the delay check of each family's A* (every order one
                second late: the same delay argument research 24 uses for its verdict)
Development bars of research 20 and 21 only; no sealed file is read and no research 24 bar is read. Nothing of research 24's
own grid, amount or data is computed here.
Writes results/regress24.json (an earlier file is moved to results/superseded/); exits 1 when a check fails.
  python regress24.py [--workers 4]"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core24 as C24  # noqa: E402
import loader24 as L24  # noqa: E402
import stats24 as ST24  # noqa: E402
import loader20 as L20  # noqa: E402  (research 20's development plan)
import loader21 as L21  # noqa: E402  (research 21's development plan)
import stats21 as ST21  # noqa: E402

C, ST20 = C24.C, C24.ST20
RES = os.path.join(HERE, "results")
R20_RES = os.path.join(C24.PROJ, "study20_pairs", "results")
R21_RES = os.path.join(C24.PROJ, "study21_more_pairs", "results")
AMOUNT = 50_000


def study20_pairs(workers):
    dev = json.load(open(os.path.join(R20_RES, "dev20_summary.json")))
    if os.path.abspath(L20.DEV_ROOT) != os.path.abspath(L24.REGRESSION_ROOTS[0]):
        raise SystemExit("loader20.DEV_ROOT is not the regression root of loader24")
    tasks, days, skipped = L20.dev_plan()
    cfgs = ST20.configs()
    A, X, info, _ = C24.simulate(tasks, L24.REGRESSION_ROOTS[0], False, cfgs, amount=AMOUNT, delays=(0,), workers=workers)
    n_days = len(days)
    idx = ST24.boot_index(n_days, seed=ST20.BOOT_SEED)
    M = C.meta(tasks, days)
    worst, n_rows = {v: [] for v in C24.VARS}, 0
    for ci, c in enumerate(cfgs):
        for v in C24.VARS:
            old = next(r for r in dev["rows"] if r["name"] == f"{ST20.sig_name(c)}_M" and r["variant"] == v)
            worst[v] += C24.row_diffs(C24.run_row(A, X, M, ci, 0, v, n_days, idx, old["name"]), old)
            n_rows += 1
    mx = {f"M_{v}": float(max(x)) for v, x in worst.items()}
    chk = {"pair_days": len(tasks), "days": n_days, "settings": [ST20.sig_name(c) for c in cfgs], "rows_compared": n_rows,
           "numbers_compared": int(sum(len(x) for x in worst.values())), "max_abs_diff": mx,
           "bootstrap_index_equal_research20": bool(np.array_equal(idx, ST20.boot_index(n_days))),
           "m_trades_equal_signal_timetable": info["settings_with_m_trades_unlike_the_signal_timetable"] == 0,
           "sim_equal_research20": {k: info[k] == dev["sim"][k] for k in ("pair_days", "valid_grid_points", "grid_points")},
           "pairs_equal": M["names"] == dev["sim"]["pairs"], "skipped_equal": skipped == dev["sim"]["skipped_pair_days"]}
    chk["pass"] = bool(all(x == 0.0 for x in mx.values()) and chk["bootstrap_index_equal_research20"] and chk["m_trades_equal_signal_timetable"]
                       and all(chk["sim_equal_research20"].values()) and chk["pairs_equal"] and chk["skipped_equal"] and len(tasks) == 5831)
    return chk


def study21_more_pairs(workers):
    dev = json.load(open(os.path.join(R21_RES, "dev21_summary.json")))
    if os.path.abspath(L21.DEV_ROOT) != os.path.abspath(L24.REGRESSION_ROOTS[1]):
        raise SystemExit("loader21.DEV_ROOT is not the regression root of loader24")
    tasks, days, _ = L21.dev_plan()
    es = [t for t in tasks if t["family"] in ST21.FAMILIES]
    cfgs = ST21.settings()
    A, X, info, _ = C24.simulate(es, L24.REGRESSION_ROOTS[1], False, cfgs, amount=AMOUNT, delays=(0,), workers=workers)
    n_days = len(days)
    idx = ST21.boot_index(n_days)
    worst, n_rows, pairs_equal = {f"{f}_{v}": [] for f in ST21.FAMILIES for v in C24.VARS}, 0, True
    delay, n_delay = {f"{f}_{v}": [] for f in ST21.FAMILIES for v in C24.VARS}, 0
    for f in ST21.FAMILIES:
        sel = C24.subset(es, "family", f)
        tf = [es[i] for i in sel]
        M = C.meta(tf, days)
        old_f = dev["families"][f]
        pairs_equal &= M["names"] == old_f["pairs"] and len(tf) == old_f["pair_days"]
        for ci, c in enumerate(cfgs):
            for v in C24.VARS:
                old = old_f["settings"][ST21.sig_name(c)]["rows"][f"M_{v}"]
                worst[f"{f}_{v}"] += C24.row_diffs(C24.run_row(A[sel], X[sel], M, ci, 0, v, n_days, idx, old["name"]), old)
                n_rows += 1
        a_cfg = tuple(old_f["astar_signal"])
        A1, X1, _, _ = C24.simulate(tf, L24.REGRESSION_ROOTS[1], False, [a_cfg], amount=AMOUNT, delays=(1,), workers=workers)
        for v in C24.VARS:
            old = old_f["delay_check"][v]
            delay[f"{f}_{v}"] += C24.row_diffs(C24.run_row(A1, X1, M, 0, 0, v, n_days, idx, old["name"]), old)
            n_delay += 1
    mx = {k: float(max(x)) for k, x in worst.items()}
    mxd = {k: float(max(x)) for k, x in delay.items()}
    chk = {"pair_days": len(es), "days": n_days, "settings": [ST21.sig_name(c) for c in cfgs], "rows_compared": n_rows,
           "numbers_compared": int(sum(len(x) for x in worst.values())), "max_abs_diff": mx,
           "delay_1_rows_compared": n_delay, "delay_1_numbers_compared": int(sum(len(x) for x in delay.values())), "delay_1_max_abs_diff": mxd,
           "m_trades_equal_signal_timetable": info["settings_with_m_trades_unlike_the_signal_timetable"] == 0,
           "pair_days_equal_research21": len(es) == dev["sim"]["pair_days"], "pairs_equal": bool(pairs_equal)}
    chk["pass"] = bool(all(x == 0.0 for x in mx.values()) and all(x == 0.0 for x in mxd.values()) and chk["m_trades_equal_signal_timetable"]
                       and chk["pair_days_equal_research21"] and chk["pairs_equal"])
    return chk


def main(workers):
    t0 = time.time()
    r20 = study20_pairs(workers)
    r21 = study21_more_pairs(workers)
    out = {"pass": bool(r20["pass"] and r21["pass"] and not C24.frozen_differences()), "amount_usd": AMOUNT, "study20_pairs": r20, "study21_more_pairs": r21,
           "frozen_engine_files_equal_their_tags": not C24.frozen_differences(), "research24_bars_read": False, "workers": workers,
           "runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")}
    path = os.path.join(RES, "regress24.json")
    os.makedirs(RES, exist_ok=True)
    C24.keep_old(path)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args().workers)
