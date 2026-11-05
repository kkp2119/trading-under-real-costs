#!/usr/bin/env python3
"""Research 13 daily ledger (SPEC 5.1; memo 4b 5). One position path, three cost scenarios (E13-08).
Per day and scenario: P = mark + hedge_pnl + open_cost + close_cost + hedge_cost + fees (dollars). cash collects the
same day's cash flows (premium in, payout out, hedge P&L, hedge cost, fees) for the identity
sum(P) = sum(cash) - liability(open legs at their last mark)  (M1)."""
import numpy as np
import pandas as pd

import config13 as C

COMP = ("open_cost", "close_cost", "hedge_cost", "fees")


class InvariantError(RuntimeError):
    pass


class Ledger:
    def __init__(self, sid):
        self.sid = sid
        self.rows = []
        self.legs = []                      # contract-level log (results_external)

    def new_day(self, d):
        r = {"date": d, "mark": 0.0, "hedge_pnl": 0.0, "u": 0.0, "q_marked": 0.0, "q_fallback": 0.0,
             "opened": 0, "reason": "", "sel_status": "", "defect_ii": 0, "n_tranches": 0, "notional_open": 0.0,
             "Hd": 0.0, "fwd_spot": 0, "delta_bs": 0, "delta_prev": 0, "kappa_exits": 0, "deferred": 0,
             "jump_sum": 0.0, "jump_absmax": 0.0, "exdate_moved": 0}
        for sc in C.SCEN:
            for k in COMP + ("cash",):
                r[f"{k}_{sc}"] = 0.0
        self.rows.append(r)
        return r

    @staticmethod
    def close_day(r):
        for sc in C.SCEN:
            r[f"P_{sc}"] = r["mark"] + r["hedge_pnl"] + sum(r[f"{k}_{sc}"] for k in COMP)
        pb, ps, po = r["P_base"], r["P_stress"], r["P_opt"]
        tol = 1e-9 * max(1.0, abs(pb), abs(ps), abs(po))
        if not (ps <= pb + tol and pb <= po + tol):
            raise InvariantError(f"M5 cost monotonicity violated on {r['date']}")

    def frame(self):
        return pd.DataFrame(self.rows)

    def identity(self, liability):
        """M1: max relative gap over scenarios between sum(P) and sum(cash) - liability."""
        df = self.frame()
        gaps = []
        for sc in C.SCEN:
            a, b = df[f"P_{sc}"].sum(), df[f"cash_{sc}"].sum() - liability
            gaps.append(abs(a - b) / max(1.0, abs(a), abs(b)))
        return max(gaps) if gaps else 0.0

    def check_identity(self, liability, tol=1e-9):
        g = self.identity(liability)
        if not np.isfinite(g) or g > tol:
            raise InvariantError(f"M1 accounting identity gap {g:.3e} ({self.sid})")
        return g
