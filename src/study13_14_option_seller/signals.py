#!/usr/bin/env python3
"""Research 13 gates and sizing (SPEC 3.1, 3.8, 3.9; E13-01). Signals are read on the signal day t = the business day
before the trade day d. A setting whose signal is missing does not open (not a defect)."""
import math

import config13 as C


def signal(mkt, t):
    """{'iv30', 'iv91', 'rv21'} on signal day t (NaN when missing); t None (no earlier day) -> all NaN."""
    if t is None:
        return {"iv30": float("nan"), "iv91": float("nan"), "rv21": float("nan")}
    return {"iv30": mkt.iv(t, 30), "iv91": mkt.iv(t, 91), "rv21": mkt.rv21(t)}


def decide(setting, sig):
    """(open allowed, multiplier m, reason) for one setting; reason in {'ok', 'gate', 'signal_missing'}."""
    iv30, iv91, rv = sig["iv30"], sig["iv91"], sig["rv21"]
    need = set()
    if setting.gate == "G1":
        need |= {"iv30", "rv21"}
    if setting.gate == "G2":
        need |= {"iv30", "iv91"}
    if setting.sizing == "V":
        need |= {"iv30"}
    if any(not math.isfinite(sig[k]) for k in need):
        return False, float("nan"), "signal_missing"
    if setting.gate == "G1" and not (iv30 - rv > 0):
        return False, float("nan"), "gate"
    if setting.gate == "G2" and not (iv30 <= iv91):
        return False, float("nan"), "gate"
    m = 1.0 if setting.sizing == "K" else min(max(C.SIZING_REF / iv30, C.SIZING_LO), C.SIZING_HI)
    return True, m, "ok"
