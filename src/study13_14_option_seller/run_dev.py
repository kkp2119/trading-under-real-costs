#!/usr/bin/env python3
"""Research 13 development run (SPEC 8, 15 step 6; memo 4b 7). Run once, only with Kyle's permission:
    python run_dev.py --go
Refuses without --go or when development results exist. SPX, settings O0..O10, 1996-01-04..2012-12-31, no forced close
on the last day (11). Reads development files only (E13-17); every path read is logged."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import config13 as C
import ladder
import market
import results13 as R
import select13
import stats13


def stats_start(days):
    i = days.index(C.DEV_START)
    return days[i + C.STATS_OFFSET]


def setting_metrics(df, start, end, mkt_ret=None):
    w = df[(df.date >= start) & (df.date <= end)]
    rb, rs = w.P_base.to_numpy() / C.N0, w.P_stress.to_numpy() / C.N0
    h1 = w[w.date <= C.F2_SPLIT].P_base.to_numpy() / C.N0
    h2 = w[w.date > C.F2_SPLIT].P_base.to_numpy() / C.N0
    t = stats13.tails(rb)
    out = {"days": len(w), "sharpe": stats13.sharpe(rb), "sharpe_h1": stats13.sharpe(h1), "sharpe_h2": stats13.sharpe(h2),
           "sharpe_stress": stats13.sharpe(rs), "sharpe_opt": stats13.sharpe(w.P_opt.to_numpy() / C.N0),
           "calmar": t["calmar"], "max_drawdown": t["max_drawdown"], "tails": t,
           "lo_sharpe": stats13.lo_sharpe(rb), "psr_0": stats13.psr(rb, 0.0),
           "psr_star": stats13.psr(rb, C.SR_STAR / np.sqrt(C.ANN)),
           "dsr": {n: stats13.deflated_sharpe(rb, n) for n in C.N_TRIALS},
           "events": stats13.events(w.date.to_numpy(), rb), "defect_ratio": stats13.defect_ratio(w),
           "u_mean": float(w.u.mean()), "u_gt_0.2_days": int((w.u > C.DEFECT_U).sum()),
           "notional_mean_over_N0": float(w.notional_open.mean() / C.N0),
           "gap_days": int((w.sel_status == "gap").sum())}
    if mkt_ret is not None:
        mr = mkt_ret.reindex(w.date).to_numpy()
        out["betas"] = stats13.betas(rb, mr)
    return out


def market_returns(mkt, days):
    c = pd.Series([mkt.close(d)[0] for d in days], index=days)
    return c.pct_change()


def main(argv=None, source=None, results_dir=None, ext_dir=None, und="SPX"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        sys.exit("refused: the development run needs --go, given only after Kyle's permission")
    rdir = results_dir or R.RESULTS
    if C.RV21_FILL_MISSING_CLOSE is None:
        sys.exit("refused: N3 (missing close inside the RV21 window) is not decided")
    if os.path.exists(os.path.join(rdir, "dev_summary.json")):
        sys.exit("refused: development results exist (SPEC 15: one development run)")
    src = source or market.RealSource()
    mkt = market.Market(src, und, ("dev",))
    for y in range(C.DEV_START // 10000, C.DEV_END // 10000 + 1):       # E13-02 on development data before the run
        a = src.vsurfd(und, y)
        vs, _ = market.first_rows(pd.DataFrame({k: np.asarray(a[k]) for k in ("date", "days", "delta", "cp_flag",
                                                                            "impl_volatility")}),
                                  ("date", "days", "delta", "cp_flag"))                  # A13-4 8d
        iv = vs.impl_volatility.to_numpy(float)
        sel = ((vs.days == 30) & (vs.delta == 50) & (vs.cp_flag == 1)).to_numpy()
        x = iv[sel & np.isfinite(iv) & (iv > 0)]
        if len(x) and (x.min() <= C.IV_RANGE[0] or x.max() >= C.IV_RANGE[1]):
            sys.exit(f"stop: vsurfd IV30 outside {C.IV_RANGE} in {y} (E13-02)")
    eng = ladder.run(mkt, C.SETTINGS, C.DEV_START, C.DEV_END)
    checks = {sid: e.ledger.check_identity(e.liability()) for sid, e in eng.items()}
    days = [d for d in mkt.days if C.DEV_START <= d <= C.DEV_END]
    s0 = stats_start(mkt.days)
    mret = market_returns(mkt, days)
    frames = {sid: e.ledger.frame() for sid, e in eng.items()}
    metrics = {sid: setting_metrics(df, s0, C.DEV_END, mret) for sid, df in frames.items()}
    for sid in metrics:
        metrics[sid]["filters"] = select13.filters(metrics[sid])
    chosen, cands = select13.choose(metrics)
    daily = pd.DataFrame({"date": days})
    for sid, df in frames.items():
        for sc in C.SCEN:
            daily[f"{sid}_{sc}"] = df[f"P_{sc}"].to_numpy()
    rid = R.run_id()
    R.write_csv(daily, os.path.join(rdir, "dev_daily.csv"))
    R.write_csv(pd.DataFrame([{"setting": s, **{k: v for k, v in m.items() if not isinstance(v, dict)},
                               **{f"F_{k}": v for k, v in m["filters"].items()}} for s, m in metrics.items()]),
                os.path.join(rdir, "dev_settings.csv"))
    R.write_json({"identity_gap": checks, "stats_start": s0, "rv21_filled_signal_days": len(mkt.rv21_filled),
                  "iv_out_of_range_signal_days": len(mkt.iv_out_of_range), "data_fixes": mkt.fixes["dev"],
                  "exdate_moved_leg_days": {sid: int(df.exdate_moved.sum()) for sid, df in frames.items()},
                  "read_paths": len(set(src.paths())),
                  "holdout_paths": sum("data_holdout" in p for p in src.paths())}, os.path.join(rdir, "dev_checks.json"))
    R.write_json({"run_id": rid, "metrics": metrics, "env": R.env_versions()}, os.path.join(rdir, "dev_summary.json"))
    sel = None
    if chosen:
        df = frames[chosen]
        w = df[(df.date >= s0) & (df.date <= C.DEV_END)]
        sel = {"setting": chosen, "sharpe": metrics[chosen]["sharpe"], "max_drawdown": metrics[chosen]["max_drawdown"],
               "calmar": metrics[chosen]["calmar"], "d_dev": stats13.defect_ratio(w)}
    R.write_json({"candidate": sel, "candidates": cands, "stats_start": s0, "dev_days": len(days),
                  "switch_table": {"delta": C.DELTA_SOURCE, "T": C.T_BASIS, "nonstd": list(C.NONSTD_CODES)},
                  "code_sha256": R.code_hashes()}, os.path.join(rdir, "selected_config.json"))
    for sid, e in eng.items():
        R.write_external(pd.DataFrame(e.ledger.legs), f"dev_legs_{sid}.csv", "dev", rid, rdir, ext_dir)
    with open(os.path.join(rdir, "dev_read_paths.txt"), "w") as f:
        f.write("\n".join(sorted(set(src.paths()))) + "\n")
    return {"selected": chosen, "candidates": cands}


if __name__ == "__main__":
    main()
