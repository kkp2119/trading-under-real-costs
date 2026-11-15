#!/usr/bin/env python3
"""Research 20 regression test (step-2 prompt 1): the research 17 kernel, called through pair20's single-order path, on the
existing universe 2015-2020 must reproduce research 17's mean implementation shortfall at three decimals:
  L0D0  S0 3.785, S2i 2.674      L3D1  S0 4.579, S2i 3.112      (cents per share, 1,000-share parents, 300 s)
Bars through study15_ml_execution/loader15 (development bars of studies 15 and 12 only; no sealed or test folder is read). Also checked:
every per-parent output equals engine17.run. Writes results/regress20.json; exits 1 when a check fails (dev20.py refuses to
run without a passing file).
  python regress20.py [--workers 8]"""
import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(PROJ, "study15_ml_execution"))
import pair20 as PE  # noqa: E402
import loader15 as L15  # noqa: E402

G = PE.G
TARGETS = {"L0D0": {"S0": 3.785, "S2i": 2.674}, "L3D1": {"S0": 4.579, "S2i": 3.112}}
RUNS = (("S0", PE.M), ("S2i", PE.P))
YEARS = tuple(range(2015, 2021))


def task(t):
    year, row, close_bar = t
    b = L15.load(year, row)
    dirs, starts = G.parents(row["date"], close_bar)
    ok = G.quote_ok(b)[starts]
    dirs, starts = dirs[ok], starts[ok]
    X = PE.stack(b)
    outs = np.zeros((2, 2, len(dirs), G.NCOL), dtype=np.int64)
    same = True
    for li, lr in enumerate(G.LRULES):
        for ri, (name, meth) in enumerate(RUNS):
            PE.run_parents(X, meth, lr, dirs, starts, G.HOR, G.QTY, outs[li, ri])
            same &= bool(np.array_equal(outs[li, ri], G.run(b, name, lr, dirs, starts)))
    return dirs, (b["bid"] + b["ask"])[starts], outs, same, int((~ok).sum())


def main(workers):
    t0 = time.time()
    tasks = []
    for y in YEARS:
        man, cb = L15.manifest(y), L15.close_bars(y)
        tasks += [(y, r, cb[d]) for (s, d), r in sorted(man.items())]
    assert all(r["date"] <= "20201231" for _, r, _ in tasks), "date after 2020"
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(task, tasks, chunksize=16)
    dirs = np.concatenate([r[0] for r in res])
    mid2 = np.concatenate([r[1] for r in res])
    OUT = np.concatenate([r[2] for r in res], axis=2)
    chk = {"symbol_days": len(tasks), "parents": int(len(dirs)), "dropped_parents": int(sum(r[4] for r in res)),
           "per_parent_equal_engine17_run": bool(all(r[3] for r in res)), "mean_is_3dp": {}}
    for lab, lr, dep in G.VARIANTS:
        for ri, (name, _) in enumerate(RUNS):
            got = float(G.costs(OUT[G.LRULES.index(lr), ri], dirs, mid2, dep).mean())
            want = TARGETS[lab][name]
            chk["mean_is_3dp"][f"{lab} {name}"] = {"got": got, "want": want, "pass": round(got, 3) == want}
    chk["pass"] = bool(chk["per_parent_equal_engine17_run"] and all(v["pass"] for v in chk["mean_is_3dp"].values())
                       and chk["parents"] == 295_960)
    chk["engine17_sha256"] = PE.ENGINE17_SHA
    chk["runtime_s"] = round(time.time() - t0, 1)
    chk["written"] = dt.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(HERE, "results", "regress20.json"), "w") as fh:
        json.dump(chk, fh, indent=1)
    print(json.dumps(chk, indent=1))
    sys.exit(0 if chk["pass"] else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args().workers)
