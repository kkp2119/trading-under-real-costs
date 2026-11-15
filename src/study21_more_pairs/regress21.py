#!/usr/bin/env python3
"""Research 21 regression test (step-2 prompt 1): research 21's code path (loader21.load -> core21.simulate ->
core21.family_tables) on the development data of research 20 must reproduce research 20's stored development numbers
with a largest difference of exactly 0:
  results/dev20_summary.json     the 8 M candidates under L3D1 and L0D0 and the 8 rows of the mid table
  results/delay20_summary.json   the delayed-mid table (8 settings x delays 0, 1, 5, 30 s) and its 5-second sentences
The pair-days are research 20's development plan (loader20.dev_plan), the signal settings its grid (kappa 0 and 1) and the
bootstrap index its seed 20, so that the intervals and p-values are compared too; everything else is the research 21
path. Development bars of research 20 only; no sealed file is read.
Writes results/regress21.json; exits 1 when a check fails (dev21.py refuses to run without a passing file).
  python regress21.py [--workers 4]"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core21 as C21  # noqa: E402
import loader21 as L21  # noqa: E402
import stats21 as ST21  # noqa: E402
import loader20 as L20  # noqa: E402  (research 20's development plan; core21 has put study20_pairs on the path)

S, ST20 = C21.S, C21.ST20
RES = os.path.join(HERE, "results")
R20_RES = os.path.join(C21.R20, "results")
row_diffs, delayed_diffs = C21.row_diffs, C21.delayed_diffs


def main(workers):
    t0 = time.time()
    dev = json.load(open(os.path.join(R20_RES, "dev20_summary.json")))
    dly = json.load(open(os.path.join(R20_RES, "delay20_summary.json")))
    if os.path.abspath(L20.DEV_ROOT) != os.path.abspath(L21.R20_DEV_ROOT):
        raise SystemExit("loader20.DEV_ROOT is not the regression root of loader21")
    tasks, days, skipped = L20.dev_plan()
    for t in tasks:
        t["family"] = "R20"
    cfgs = ST20.configs()
    A, info, ex = C21.simulate(tasks, L21.R20_DEV_ROOT, False, cfgs, delay=0, workers=workers, mid_delays=S.MID_DELAYS)
    n_days = len(days)
    idx = ST21.boot_index(n_days, seed=ST20.BOOT_SEED)
    M, blocks, _, _, _ = C21.family_tables(A, ex["mid_delayed"], tasks, days, cfgs, idx, "R20")
    worst = {"M_L3D1": [], "M_L0D0": [], "mid": [], "delayed": []}
    n_rows, sentences = 0, True
    for c in cfgs:
        name = ST20.sig_name(c)
        b = blocks[name]
        for v in C21.VARS:
            old = next(r for r in dev["rows"] if r["name"] == f"{name}_M" and r["variant"] == v)
            worst[f"M_{v}"] += row_diffs(b["rows"][f"M_{v}"], old)
            n_rows += 1
        old = next(r for r in dev["mid_rows"] if r["name"] == f"{name}_mid")
        worst["mid"] += row_diffs(b["rows"]["mid"], old)                      # the per-trade interval included
        n_rows += 1
        od = dly["settings"][name]
        worst["delayed"] += delayed_diffs(b["delayed"], od["rows"])
        sentences &= b["sentence_5s"] == od["sentence_5s"]
        n_rows += len(od["rows"])
    mx = {k: float(max(v)) for k, v in worst.items()}
    chk = {"pair_days": len(tasks), "days": n_days, "settings": [ST20.sig_name(c) for c in cfgs], "rows_compared": n_rows,
           "numbers_compared": int(sum(len(v) for v in worst.values())), "max_abs_diff": mx,
           "sentences_5s_equal": bool(sentences),
           "bootstrap_index_equal_research20": bool(np.array_equal(idx, ST20.boot_index(n_days))),
           "m_trades_equal_signal_timetable": info["settings_with_m_trades_unlike_the_signal_timetable"] == 0,
           "sim_equal_research20": {**{k: info[k] == dev["sim"][k] for k in ("pair_days", "valid_grid_points", "grid_points")},
                                    **{k: info[k] == dly["checks"][k] for k in ("pair_days_left_out_no_state_after_first_state",)}},
           "pairs_equal": M["names"] == dev["sim"]["pairs"], "skipped_equal": skipped == dev["sim"]["skipped_pair_days"],
           "frozen_engine_files_equal_code_freeze_20": not C21.frozen_differences(), "bars_root": "$DATA_DIR/research20_data/dev/bars_1s",
           "workers": workers}
    chk["pass"] = bool(all(v == 0.0 for v in mx.values()) and chk["sentences_5s_equal"] and chk["bootstrap_index_equal_research20"]
                       and chk["m_trades_equal_signal_timetable"] and all(chk["sim_equal_research20"].values()) and chk["pairs_equal"]
                       and chk["skipped_equal"] and chk["frozen_engine_files_equal_code_freeze_20"] and len(tasks) == 5831)
    chk["runtime_s"] = round(time.time() - t0, 1)
    chk["written"] = dt.datetime.now().isoformat(timespec="seconds")
    path = os.path.join(RES, "regress21.json")
    C21.keep_old(path)
    with open(path, "w") as fh:
        json.dump(chk, fh, indent=1)
    print(json.dumps(chk, indent=1))
    sys.exit(0 if chk["pass"] else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args().workers)
