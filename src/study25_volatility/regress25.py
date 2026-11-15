#!/usr/bin/env python3
"""Research 25 regression on the validation data (step-2 prompt 1, SPEC25 4, SPEC25_ANSWERS A3). It must pass before the
validation run. Reads the bars of the existing universe 2019-2020 and SPY of research 21's development bars (the same
parent orders as research 23's validation period) and research 23's own record of that period. No sealed path.
  runs     S0, S2i, R0 (E*) and the three regression runs with kappa = 0 read as "no volatility condition" (k0:B1_p60,
           k0:B2, k0:B3). None of the 14 candidates is run and no IS of a candidate is computed.
  checks   (dev25.regression; every difference must be exactly 0)
           the parents; the per-parent outputs of S0, S2i, R0 under both fill rules; the probability of research 23's
           frozen model, the target and the valid flag at every decision point; the mean IS of S0, S2i, C3, R0;
           k0:B1_p60 = research 23's R1_0.60, k0:B2 = R2_0.40, k0:B3 = R3_0.60_0.40 parent by parent
  counts   decision points with p (a valid decision quote) and without q, by cause (ANSWERS A3)
Writes results/regress25.json (an earlier file is moved to results/superseded/); exits 1 when a check fails.
  python regress25.py [--workers 4]"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core25 as C  # noqa: E402
import dev25 as DV  # noqa: E402

S = C.S


def keep_old(path):
    if os.path.exists(path):
        d = os.path.join(os.path.dirname(path), "superseded")
        os.makedirs(d, exist_ok=True)
        stem, ext = os.path.splitext(os.path.basename(path))
        q = os.path.join(d, f"{stem}_{dt.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y%m%d_%H%M%S')}{ext}")
        if os.path.exists(q):
            raise RuntimeError(f"{q} exists")
        os.rename(path, q)


def main(workers):
    t0 = time.time()
    model = C.model23()
    tasks, pinfo, ainfo = DV.plan(DV.VALID, workers)
    names = list(C.BASE + C.K0)
    if any(n in C.CANDS for n in names):
        raise SystemExit("a candidate among the regression runs")
    par, OUT, G, info = C.simulate(tasks, names, (), workers, model=model, grid=True)
    pairs = [(k, S.K0[k][1]) for k in C.K0]
    ok, chk = DV.regression(par, OUT, names, G, pairs)
    gc = info["grid_counts"]
    p_def = int(np.isfinite(G["P23"]).sum())
    out = {"pass": bool(ok), "runs": names, "pairs": [f"{a} = research 23's {b}" for a, b in pairs], "regression": chk,
           "points": {"grid": gc["points"], "p_defined": p_def, "quote_valid": gc["quote_valid"],
                      "p_defined_q_undefined": gc["quote_valid_q_undefined"], "of_which_window_not_full": gc["quote_valid_window_not_full"],
                      "of_which_spread_zero": gc["quote_valid_spread_zero"], "share_of_p_defined": gc["quote_valid_q_undefined"] / max(p_def, 1)},
           "sim": {**{k: v for k, v in info.items() if k != "grid_counts"}, **pinfo, "adv": ainfo, "days": int(par.date.nunique()),
                   "first_day": par.date.min(), "last_day": par.date.max()},
           "model_sha256": C.sha(C.MODEL23), "candidates_run": 0, "runtime_s": round(time.time() - t0, 1),
           "written": dt.datetime.now().isoformat(timespec="seconds")}
    path = os.path.join(HERE, "results", "regress25.json")
    keep_old(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=C.jdefault, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("pass", "pairs", "points", "runtime_s")}, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in chk.items() if k != "mean_is"}, indent=1, ensure_ascii=False))
    for k, v in chk.get("mean_is", {}).items():
        print(f"  {k:18s} {v['got']:.6f}  research 23 ({v['research23_run']}) {v['study23_combined_forecast']:.6f}  diff {v['abs_diff']:.1e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args().workers)
