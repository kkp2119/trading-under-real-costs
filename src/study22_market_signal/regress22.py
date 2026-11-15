#!/usr/bin/env python3
"""Research 22 regression test on the development data (step-2 prompt 1; every check must hold before the development run):
the existing universe 2015-2020 through loader22 / core22 with S0, S2i, G0 (E*) and the three rules on flags that are 0.5
everywhere (half_G1, half_G2, half_G3). No signal is computed and no candidate is run.
  parents     the parents (date, symbol, direction, start bar, start quote) and the per-parent outputs of S0, S2i and E*
              equal research 17's stored development outputs ($DATA_DIR/research17_data/dev/dev17_parents.*)
  means       mean IS of S0, S2i, G0 and C3 under L3D1 and L0D0 equal research 17 / 18's development values at 3 decimals
  flags 0.5   half_G1, half_G2 and half_G3 equal G0 in every output column of every parent
The SPY files of the development days are loaded and checked on the way (sha256, columns, volume identity); nothing of
them is used. Writes results/regress22.json (an earlier file is moved to results/superseded/); exits 1 when a check fails.
  python regress22.py [--workers 4]"""
import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core22 as C  # noqa: E402
import dev22 as DV  # noqa: E402
import loader22 as L22  # noqa: E402

E = C.E
DEV17 = os.path.join(os.environ.get("DATA_DIR", "data"), "research17_data/dev")
R17_NAME = {"S0": "S0", "S2i": "S2i", "G0": "Estar"}


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
    tasks, cal, pinfo = L22.dev_plan()
    names = list(C.BASE + C.HALF)
    par, OUT, sd, info = C.simulate(tasks, names, (), workers, bases=())
    ok, chk = DV.regression(par, OUT, names)
    z = np.load(os.path.join(DEV17, "dev17_parents.npz"))
    p17 = pd.read_csv(os.path.join(DEV17, "dev17_parents.csv.gz"), dtype={"date": str, "sym": str}, float_precision="round_trip")
    cols = ("date", "sym", "dir", "start", "mid2", "spread")
    chk["parents_equal_research17"] = bool(len(par) == len(p17) and all(np.array_equal(par[c].to_numpy(), p17[c].to_numpy()) for c in cols))
    r17, l17 = list(z["runs"]), list(z["lrules"])
    chk["per_parent_equal_research17"] = {f"L{lr}": {m: bool(chk["parents_equal_research17"] and np.array_equal(
        OUT[E.LRULES.index(lr), names.index(m), :, :9], z["out"][l17.index(lr), r17.index(R17_NAME[m])])) for m in C.BASE} for lr in E.LRULES}
    g0 = OUT[:, names.index("G0")]
    chk["flags_0.5_equal_G0"] = {h: bool(np.array_equal(OUT[:, names.index(h)], g0)) for h in C.HALF}
    chk["signals_computed"] = bool(sd["bases"])
    ok = bool(ok and chk["parents_equal_research17"] and all(all(v.values()) for v in chk["per_parent_equal_research17"].values())
              and all(chk["flags_0.5_equal_G0"].values()) and not chk["signals_computed"])
    out = {"pass": ok, "checks": chk, "sim": {**info, **pinfo}, "runtime_s": round(time.time() - t0, 1),
           "written": dt.datetime.now().isoformat(timespec="seconds")}
    path = os.path.join(HERE, "results", "regress22.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keep_old(path)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args().workers)
