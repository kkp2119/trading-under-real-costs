#!/usr/bin/env python3
"""Research 15 step 2 report (2015-2020 only): per year rows, no_quote share, NaN share per input, label baselines per side
(fill rate, partial share, mean val_h, mean mk_h of filled orders, mean tk_h, truncation shares) over quoted decisions,
and the trade-side shares vol_bid / vol_ask / vol_in / vol_unc from xy/MANIFEST.tsv. Every xy file is checked against
its MANIFEST sha256. Writes logs/xy_report15.txt and logs/xy_report15_nan.tsv."""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import xy15 as X  # noqa: E402

FEAT = list(X.FEATURES)


def main():
    man = pd.read_csv(os.path.join(X.OUT, "MANIFEST.tsv"), sep="\t", dtype={"date": str, "sym": str})
    man["y"] = man.date.str[:4].astype(int)
    col = {c: i for i, c in enumerate(X.COLUMNS)}
    out, nan_rows = [], []
    bad = 0
    for y, g in man.groupby("y"):
        n = nq = 0
        nan = np.zeros(len(FEAT))
        acc = {}

        def add(k, v):
            s, c = acc.get(k, (0.0, 0))
            acc[k] = (s + float(np.nansum(v)), c + int(np.sum(~np.isnan(v))))
        for r in g.itertuples():
            p = os.path.join(X.OUT, r.file)
            data = open(p, "rb").read()
            if hashlib.sha256(data).hexdigest() != r.sha256:
                bad += 1
            with np.load(p, allow_pickle=False) as z:
                M = z["xy"].astype(np.float64)
                assert tuple(z["columns"]) == X.COLUMNS
            n += len(M)
            q = M[:, col["no_quote"]] == 0
            nq += int((~q).sum())
            nan += np.isnan(M[:, [col[f] for f in FEAT]]).sum(axis=0)
            Q = M[q]
            for p_ in ("b", "s"):
                fill = Q[:, col[f"{p_}_fill"]]
                add(f"{p_}_fill", fill)
                add(f"{p_}_partial", np.where(fill == 1, (Q[:, col[f"{p_}_frac"]] < 1).astype(float), np.nan))
                for h in X.HS:
                    add(f"{p_}_val_{h}", Q[:, col[f"{p_}_val_{h}"]])
                    add(f"{p_}_mk_{h}", Q[:, col[f"{p_}_mk_{h}"]])
                    add(f"{p_}_tk_{h}", Q[:, col[f"{p_}_tk_{h}"]])
                    add(f"tr_mk{p_}_{h}", Q[:, col[f"tr_mk{p_}_{h}"]])
            for h in X.HS:
                add(f"tr_tk_{h}", Q[:, col[f"tr_tk_{h}"]])
        mean = {k: s / c if c else float("nan") for k, (s, c) in acc.items()}
        vs = g[["vol", "vol_bid", "vol_ask", "vol_in", "vol_unc"]].sum()
        out.append(f"{y}: symbol-days {len(g)}, rows {n}, no_quote {nq} ({nq / n:.4%}); decisions in the day files "
                   f"{int(g.decisions.sum())}; trade side shares bid {vs.vol_bid / vs.vol:.4f} ask {vs.vol_ask / vs.vol:.4f} "
                   f"in {vs.vol_in / vs.vol:.4f} unc {vs.vol_unc / vs.vol:.6f}")
        for p_, name in (("b", "buy"), ("s", "sell")):
            out.append(f"   {name}: fill rate {mean[f'{p_}_fill']:.4f}, partial among fills {mean[f'{p_}_partial']:.4f}; "
                       + "; ".join(f"h {h}: val {mean[f'{p_}_val_{h}']:+.4f}, mk(filled) {mean[f'{p_}_mk_{h}']:+.4f}, "
                                   f"tk {mean[f'{p_}_tk_{h}']:+.4f} c, trunc mk {mean[f'tr_mk{p_}_{h}']:.5f}" for h in X.HS))
        out.append("   taker truncation share: " + ", ".join(f"h {h} {mean[f'tr_tk_{h}']:.5f}" for h in X.HS))
        nan_rows.append(pd.Series(nan / n, index=FEAT, name=y))
    t = pd.DataFrame(nan_rows)
    t.index.name = "year"
    t.round(6).to_csv(os.path.join(HERE, "logs", "xy_report15_nan.tsv"), sep="\t")
    out.append(f"xy files with a sha256 different from MANIFEST: {bad}")
    out.append("NaN share per input (all stored rows), by year:")
    out.append(t.T.map(lambda v: f"{v:.4f}").to_string())
    txt = "\n".join(out)
    open(os.path.join(HERE, "logs", "xy_report15.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
