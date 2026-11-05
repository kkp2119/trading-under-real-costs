#!/usr/bin/env python3
"""Research 13 development selection (SPEC 8). Inputs per setting are C-base net figures on the development statistics
window (5.2) and the C-stress Sharpe:
  F1 Sharpe >= 0.50; F2 first half (window start..2004-12-31) and second half (2005-01-01..2012-12-31) Sharpe > 0;
  F3 C-stress Sharpe > 0; F4 Calmar >= 0.33.
Among candidates: those within 0.10 of the best Sharpe (always the best, never chained), then lowest complexity, then
higher Sharpe, then lower O number."""
import math

import config13 as C


def filters(m):
    """m: {'sharpe', 'sharpe_h1', 'sharpe_h2', 'sharpe_stress', 'calmar'} -> {'F1'..'F4', 'pass'}."""
    ok = lambda v: v is not None and math.isfinite(v)                    # noqa: E731
    f = {"F1": ok(m["sharpe"]) and m["sharpe"] >= C.F1_MIN,
         "F2": ok(m["sharpe_h1"]) and ok(m["sharpe_h2"]) and m["sharpe_h1"] > 0 and m["sharpe_h2"] > 0,
         "F3": ok(m["sharpe_stress"]) and m["sharpe_stress"] > 0,
         "F4": ok(m["calmar"]) and m["calmar"] >= C.CALMAR_MIN}
    f["pass"] = all(f.values())
    return f


def choose(metrics):
    """metrics: {sid: dict as in filters}. Returns (selected sid or None, candidate sids)."""
    cands = [sid for sid, m in metrics.items() if filters(m)["pass"]]
    if not cands:
        return None, []
    best = max(metrics[s]["sharpe"] for s in cands)
    band = [s for s in cands if metrics[s]["sharpe"] >= best - C.TIE_BAND]
    num = lambda s: int(s[1:])                                             # noqa: E731
    band.sort(key=lambda s: (C.setting(s).complexity, -metrics[s]["sharpe"], num(s)))
    return band[0], cands
