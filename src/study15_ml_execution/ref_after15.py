#!/usr/bin/env python3
"""Research 15 reference analysis after the step 3 results (Kyle 2026-10-29; not used for any selection; DEVIATIONS15 18).
Validation years only (2019, 2020); 2021+ data is never read. Run with python ref_after15.py.

1 B0 (09:30-16:00) is re-run on the validation symbol-days and its limit fills are kept one by one with their markouts
  (research 12 accounting rule: horizon bar < close and a valid mid there); the totals are checked against the B0 row of
  results15/val15_fold_<A|B>.csv.
2 Table: B0 and every setting with >= 1,000 limit fill events in both folds; per fold P&L per share (net R / all shares,
  cents), markouts 5/30/60 s (cents per limit share), limit and market shares, net Sharpe R, ratio of P&L per share to B0's.
3 Random baseline: for a setting with n limit fill events in a fold, n fills drawn without replacement from that fold's
  B0 limit fills (numpy default_rng(15), 100 draws); the per-share markout of each draw at 5/30/60 s gives a distribution.
  A random subset of fills is not a strategy, so the comparison is on the per-share markout of limit fills (the P&L of a
  limit fill held h seconds, no fee under R), not on the strategy P&L per share.
Writes results15/ref_after15.txt and results15/ref_after15.csv; B0 fills to $DATA_DIR/research15_data/val15/b0fills_<fold>.npz."""
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features15  # noqa: E402
import loader15 as L  # noqa: E402
import model15 as M15  # noqa: E402
import policy15 as P  # noqa: E402
import simulator15 as S  # noqa: E402

RES = os.path.join(HERE, "results15")
VAL = os.path.join(M15.DATA, "val15")
KS = (5, 30, 60)
N_DRAW, SEED, MIN_FILLS = 100, 15, 1000


def b0_fills(task):
    """Limit fills of B0 on one symbol-day: rows (qty, sum2_k, ok_k for k in KS) with research 12 markout rules."""
    year, brow, close_bar, gate = task
    b = L.load(year, brow)
    f = features15.compute(b)
    fills, _, _ = S.simulate(b, f, P.B0W2.kernel_params("Q", close_bar, gate))
    lim = fills[fills[:, 2] == 0] if len(fills) else fills
    out = np.zeros((len(lim), 1 + 2 * len(KS)), dtype=np.int64)
    if len(lim):
        u, s, qty, p = lim[:, 0], lim[:, 1], lim[:, 3], lim[:, 4]
        out[:, 0] = qty
        for i, k in enumerate(KS):
            t = u + k
            ok = t < close_bar
            ok[ok] &= f["mid_ok"][t[ok]]
            out[:, 1 + 2 * i] = np.where(ok, s * qty * (f["mid2"][np.minimum(t, len(b["bid"]) - 1)] - 2 * p), 0)
            out[:, 2 + 2 * i] = ok
    return out


def collect(fold):
    year = M15.FOLDS[fold][1]
    g = M15.read_gates().set_index(["sym", "date"])
    man, cb = L.manifest(year), L.close_bars(year)
    tasks = [(year, r, cb[d], bool(g.loc[(s, d)].gate)) for (s, d), r in sorted(man.items())]
    with mp.get_context("spawn").Pool(8) as pool:
        parts = pool.map(b0_fills, tasks, chunksize=8)
    F = np.concatenate(parts)
    np.savez_compressed(os.path.join(VAL, f"b0fills_{fold}.npz"), fills=F)
    return F


def markout(F, idx, i):
    """Per-share markout in cents at KS[i] over fills idx (research 12 rule: sum2 / 2 / valid shares / 100)."""
    sel = F[idx]
    sh = (sel[:, 0] * sel[:, 2 + 2 * i]).sum()
    return sel[:, 1 + 2 * i].sum() / 2 / sh / 100 if sh else float("nan")


def main():
    t = {f: pd.read_csv(os.path.join(RES, f"val15_fold_{f}.csv")).set_index("setting") for f in ("A", "B")}
    F = {}
    lines = ["Research 15 reference analysis after the step 3 results (validation 2019 = fold A, 2020 = fold B; not used "
             "for selection; test data not read)", ""]
    for f in ("A", "B"):
        F[f] = collect(f)
        b0 = t[f].loc["B0"]
        chk = (len(F[f]) == b0.n_fill_ev and F[f][:, 0].sum() == b0.sh_lim
               and all(abs(markout(F[f], np.arange(len(F[f])), i) - b0[f"markout{k}_cents"]) < 1e-9 for i, k in enumerate(KS)))
        lines.append(f"fold {f}: B0 limit fills re-run {len(F[f])}, shares {int(F[f][:, 0].sum())}; equal to the step 3 B0 row: {chk}")
        assert chk
    tgt = [s for s in t["A"].index if s != "B0" and t["A"].loc[s, "n_fill_ev"] >= MIN_FILLS and t["B"].loc[s, "n_fill_ev"] >= MIN_FILLS]
    rows = []
    for s in ["B0"] + tgt:
        r = {"setting": s}
        for f in ("A", "B"):
            x, b0 = t[f].loc[s], t[f].loc["B0"]
            r.update({f"pps_R_{f}": x.pnl_per_share_R_cents, **{f"mk{k}_{f}": x[f"markout{k}_cents"] for k in KS},
                      f"sh_lim_{f}": int(x.sh_lim), f"sh_mkt_{f}": int(x.sh_mkt), f"fills_{f}": int(x.n_fill_ev),
                      f"sharpe_R_{f}": x.sharpe_R, f"pps_ratio_to_B0_{f}": x.pnl_per_share_R_cents / b0.pnl_per_share_R_cents})
            if s != "B0":
                rng = np.random.default_rng(SEED)
                n = int(x.n_fill_ev)
                draws = np.array([[markout(F[f], idx, i) for i in range(len(KS))]
                                  for idx in (rng.choice(len(F[f]), n, replace=False) for _ in range(N_DRAW))])
                for i, k in enumerate(KS):
                    d = draws[:, i]
                    r[f"rnd{k}_p05_{f}"], r[f"rnd{k}_p50_{f}"], r[f"rnd{k}_p95_{f}"] = np.percentile(d, [5, 50, 95])
                    r[f"rnd{k}_rank_{f}"] = float((d < x[f"markout{k}_cents"]).mean())       # share of draws below the setting
        r["pps_R_mean"] = (r["pps_R_A"] + r["pps_R_B"]) / 2
        rows.append(r)
    T = pd.DataFrame(rows).set_index("setting")
    T.to_csv(os.path.join(RES, "ref_after15.csv"))
    S_ = T.drop(index="B0")
    b0a, b0b = T.loc["B0", "pps_R_A"], T.loc["B0", "pps_R_B"]
    better_a, better_b = S_.pps_R_A > b0a, S_.pps_R_B > b0b
    lines += ["", f"settings with >= {MIN_FILLS} limit fill events in both folds: {len(S_)}; P&L per share better than B0: "
                  f"fold A {int(better_a.sum())}, fold B {int(better_b.sum())}, both {int((better_a & better_b).sum())}",
              f"B0 P&L per share (R, cents): fold A {b0a:.3f}, fold B {b0b:.3f}; markout 5/30/60: "
              f"A {T.loc['B0', 'mk5_A']:.3f}/{T.loc['B0', 'mk30_A']:.3f}/{T.loc['B0', 'mk60_A']:.3f}, "
              f"B {T.loc['B0', 'mk5_B']:.3f}/{T.loc['B0', 'mk30_B']:.3f}/{T.loc['B0', 'mk60_B']:.3f}", "",
              "top 10 by mean P&L per share (R, cents) of the two folds:"]
    top = S_.sort_values("pps_R_mean", ascending=False).head(10)
    cols = ["pps_R_mean", "pps_R_A", "pps_R_B", "pps_ratio_to_B0_A", "pps_ratio_to_B0_B", "mk5_A", "mk30_A", "mk60_A",
            "mk5_B", "mk30_B", "mk60_B", "sh_lim_A", "sh_lim_B", "sharpe_R_A", "sharpe_R_B"]
    lines.append(pd.concat([T.loc[["B0"], cols], top[cols]]).to_string(float_format=lambda v: f"{v:.3f}"))
    lines += ["", f"random baseline ({N_DRAW} draws of the same number of fills from B0's fills, seed {SEED}): per-share "
                  "markout of the setting vs the draws (rank = share of draws below the setting)"]
    rc = [c for f in ("A", "B") for k in (5, 60) for c in (f"mk{k}_{f}", f"rnd{k}_p05_{f}", f"rnd{k}_p50_{f}", f"rnd{k}_p95_{f}",
                                                           f"rnd{k}_rank_{f}")]
    lines.append(top[rc].to_string(float_format=lambda v: f"{v:.3f}"))
    for k in KS:
        for f in ("A", "B"):
            rk = S_[f"rnd{k}_rank_{f}"]
            lines.append(f"all {len(S_)} settings, markout {k}s fold {f}: above the random median {int((rk > 0.5).sum())}, "
                         f"above the random 95th percentile {int((rk >= 0.95).sum())}, below the 5th {int((rk <= 0.05).sum())}")
    txt = "\n".join(lines)
    open(os.path.join(RES, "ref_after15.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
