#!/usr/bin/env python3
"""SPEC 6 automatic candidate selection (no human choice) and the SPEC 9 gate G1 (E1-32..E1-35, E1-37, E1-38, E1-49,
E1-59; Kyle 2026-10-27: `neighbors_checked` states when an event candidate has no delta neighbours).
Input: per-setting dev metrics (Q, R) built by run_dev from the run table; output: results/selected_config.json."""
import dataclasses
import json
import math
import os

import numpy as np

import accounting
import grid
from config import ADOPT_GAIN, B0, RESULTS
from stats import sharpe

DELTA_AXIS = (1, 5, 15, 60)
L_AXIS = (1, 3, 5)


def daily(runs, setting, model, scenario, obs_days):
    """Daily portfolio net P&L in dollars over obs_days (days with no row are 0)."""
    r = runs[(runs.setting == setting) & (runs.model == model)]
    s = r.assign(net=accounting.net_arrays(r, scenario)).groupby("date").net.sum()      # integer 1e-4 $ summed first
    return np.array([float(s.get(d, 0)) / 1e4 for d in obs_days])


def setting_metrics(runs, setting, obs_days):
    """Selection inputs for one setting: Sharpes (Q, R) overall and by year, Sharpe Q50, mean limit-fill events per
    traded symbol-day (Q) and mean participation (Q) (E1-32)."""
    x = daily(runs, setting, "Q", "R", obs_days)
    years = sorted({d[:4] for d in obs_days})
    q = runs[(runs.setting == setting) & (runs.model == "Q") & (runs.skip == 0)]
    part = ((q.sh_lim_buy + q.sh_lim_sell + q.sh_mkt_buy + q.sh_mkt_sell) / q.mkt_vol)[q.mkt_vol > 0]
    return {"sharpe": sharpe(x), **{f"sharpe_{y}": sharpe(x[[d[:4] == y for d in obs_days]]) for y in years},
            "sharpe_Q50": sharpe(daily(runs, setting, "Q50", "R", obs_days)),
            "fills_per_symday": float(q.n_fill_ev.mean()) if len(q) else 0.0,
            "participation": float(part.mean()) if len(part) else float("nan")}


def n_diff(cfg):
    return sum(grid.get_dim(cfg, d) != grid.get_dim(B0, d) for d in grid.DIMENSIONS)


def delta_of(cfg):
    return 1 if cfg.mode == "event" else cfg.delta                   # E1-59


def neighbours(cfg, tested):
    """Settings that differ from cfg only by one step of delta (reset only) or L and were tested (E1-33)."""
    out = []
    if cfg.mode == "reset" and cfg.delta in DELTA_AXIS:
        i = DELTA_AXIS.index(cfg.delta)
        out += [dataclasses.replace(cfg, delta=DELTA_AXIS[j]) for j in (i - 1, i + 1) if 0 <= j < len(DELTA_AXIS)]
    if cfg.L in L_AXIS:
        i = L_AXIS.index(cfg.L)
        out += [dataclasses.replace(cfg, L=L_AXIS[j]) for j in (i - 1, i + 1) if 0 <= j < len(L_AXIS)]
    return [c for c in out if c in tested]


def ok(v):
    return v == v and v > 0


def filters(m, cfg, tested_metrics):
    """(i)-(iv) of SPEC 6 for one setting. tested_metrics: {Config: metrics}."""
    years = sorted(k for k in m if k.startswith("sharpe_2"))
    nb = neighbours(cfg, tested_metrics)
    f = {"i_years": all(ok(m[y]) for y in years) and len(years) > 0,
         "ii_Q50": ok(m["sharpe_Q50"]),
         "iii_activity": bool(m["fills_per_symday"] >= 1 and m["participation"] == m["participation"]
                              and m["participation"] < 0.01),
         "iv_neighbours": all(tested_metrics[c]["sharpe"] == tested_metrics[c]["sharpe"]
                              and tested_metrics[c]["sharpe"] >= 0 for c in nb)}
    f["pass"] = all(f.values())
    checked = {"delta": [c.cid() for c in nb if c.delta != cfg.delta], "L": [c.cid() for c in nb if c.L != cfg.L],
               "note": "event: no delta neighbours (E1-33)" if cfg.mode == "event" else ""}
    return f, checked


def select(metrics_by_cfg, names):
    """SPEC 6 steps 1-3. metrics_by_cfg: {Config: metrics}; names: {Config: name}. Returns (chosen Config or None, table)."""
    table, passing = [], []
    for cfg, m in metrics_by_cfg.items():
        f, checked = filters(m, cfg, metrics_by_cfg)
        table.append({"name": names[cfg], "cid": cfg.cid(), **m, **f, "neighbors_checked": checked})
        if f["pass"] and m["sharpe"] == m["sharpe"]:
            passing.append(cfg)
    if not passing:
        return None, table
    best = max(passing, key=lambda c: (metrics_by_cfg[c]["sharpe"], -n_diff(c), delta_of(c)))
    return best, table


def mix_decision(stock_m, mix_m):
    """SPEC 6 step 4: the mix wins when its dev Sharpe is >= stock + 0.5 and it passes (i), (ii), (iii) (E1-49: pooled)."""
    years = sorted(k for k in mix_m if k.startswith("sharpe_2"))
    f = {"i_years": all(ok(mix_m[y]) for y in years), "ii_Q50": ok(mix_m["sharpe_Q50"]),
         "iii_activity": bool(mix_m["fills_per_symday"] >= 1 and mix_m["participation"] < 0.01),
         "gain": bool(mix_m["sharpe"] == mix_m["sharpe"] and stock_m["sharpe"] == stock_m["sharpe"]
                      and mix_m["sharpe"] >= stock_m["sharpe"] + ADOPT_GAIN)}
    f["mix_wins"] = all(f.values())
    return f


def write(out, path=None):
    path = path or os.path.join(RESULTS, "selected_config.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=lambda v: None if isinstance(v, float) and math.isnan(v) else str(v))
    os.replace(tmp, path)
    return path


if __name__ == "__main__":
    raise SystemExit("select_candidate is called by run_dev.py (it needs the dev run table); not run on its own")
