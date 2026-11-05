#!/usr/bin/env python3
"""Research 13 engine core (SPEC 3.2-3.7, 5.1, 9, 11; A13-1, A13-2, A13-3; design memo 4b v2 2-5).
Order of day d: (1) mark open legs, (2) hedge P&L of yesterday's hedge, (3) exits, deferrals, kappa, (4) entries with
the signal of t = d-1, (5) hedge rebalance. Open legs are tracked by optionid (3.4)."""
import datetime as dt
import math

import numpy as np

import accounting as A
import chain as ch
import config13 as C
import costs
import signals


def next_bday(cal, d):
    x = ch.ymd(d) + dt.timedelta(days=1)
    while not cal.is_bday(ch.to_int(x)):
        x += dt.timedelta(days=1)
    return ch.to_int(x)


def defer_ok(cal, d, exdate, ltd=None):
    """3.5: d can be a deferral day: DTE >= 2 and d on or before the last trading day (calendar of the day; the
    calendar-end rule of 3.4 applies beyond its last day). ltd given: the value computed at entry (A13-4 8e)."""
    lt = cal.last_trading(exdate) if ltd is None else ltd
    return dte1(d, exdate) >= C.DTE_MIN_DEFER and d <= lt


def last_chance(cal, d, exdate, k, ltd=None):
    """3.5 with A13-4 N1: is d the last day an invalid leg may wait? k = deferral days already used, counted in
    processed business days (0 on the scheduled day). True when k reached 3, when d itself is not a deferral day
    (E13-03), or when the next business day (calendar-end rule) is not one."""
    return k >= C.DEFER_MAX or not defer_ok(cal, d, exdate, ltd) or not defer_ok(cal, next_bday(cal, d), exdate, ltd)


def dte1(d, exdate):
    return (ch.ymd(exdate) - ch.ymd(d)).days


def select(v, structure, H):
    """3.4, 3.7, A13-2 for one day: {'status': 'gap' | 'no_monthly' | 'no_candidate' | 'ok', 'exdate', 'legs'}."""
    key = (structure, H)
    if key in v._sel:
        return v._sel[key]
    lo, hi, tgt = C.RANGE[H]
    status = ch.range_status(v.d, v.std_exdates, lo, hi)
    if status != "in_range":
        v._sel[key] = {"status": status}
        return v._sel[key]
    ex = v.std_exdates
    dd = np.array([dte1(v.d, e) for e in ex])
    inr = [(abs(x - tgt), int(e)) for e, x in zip(ex, dd) if lo <= x <= hi]
    exdate = min(inr)[1]
    rows = v.expiry_rows(exdate)
    F, src, r, T = v.forward(exdate)
    legs = None
    if structure == "S":
        c = ch.dedupe(rows, C.TAU["S"])
        calls = {float(k): c.iloc[i] for i, k in enumerate(c.K) if c.cp_flag.iloc[i] == 1}
        puts = {float(k): c.iloc[i] for i, k in enumerate(c.K) if c.cp_flag.iloc[i] == -1}
        ks = sorted(set(calls) & set(puts), key=lambda k: (abs(k - F), k))[:3]
        for k in ks:
            cr, pr = calls[k], puts[k]
            if all(ch.valid_quote([x.best_bid], [x.best_offer], C.TAU["S"])[0] and ch.iv_valid([x.impl_volatility])[0]
                   for x in (cr, pr)):
                legs = [cr, pr]
                break
    else:
        c = ch.dedupe(rows, C.TAU["G"]).copy()
        _, bd = ch.bs(F, c.K, T, c.impl_volatility, r, v.S, c.cp_flag)
        om_ok = np.isfinite(c.delta.to_numpy(float))
        c["dl"] = np.where(om_ok, c.delta, np.where(ch.iv_valid(c.impl_volatility), bd, np.nan))
        c = c[np.isfinite(c.dl.to_numpy(float))]
        found = []
        for cp, tg in ((1, C.G_TARGET), (-1, -C.G_TARGET)):
            side = c[c.cp_flag == cp]
            order = sorted(range(len(side)), key=lambda i: (abs(side.dl.iloc[i] - tg),
                                                             -side.K.iloc[i] if cp == 1 else side.K.iloc[i]))[:3]
            pick = None
            for i in order:
                x = side.iloc[i]
                if ch.valid_quote([x.best_bid], [x.best_offer], C.TAU["G"])[0] and ch.iv_valid([x.impl_volatility])[0]:
                    pick = x
                    break
            found.append(pick)
        if all(x is not None for x in found):
            legs = found
    v._sel[key] = {"status": "ok" if legs else "no_candidate", "exdate": exdate, "legs": legs, "fwd_src": src}
    return v._sel[key]


class Leg:
    __slots__ = ("oid", "tranche", "exdate", "K", "cp", "q", "tau", "mark", "last_iv", "delta", "held", "deferring",
                 "sched", "k", "valid", "fallback_prev", "S_open", "opened", "settle", "ltd", "moved")

    def __init__(self, **kw):
        for k, val in kw.items():
            setattr(self, k, val)


class Engine:
    def __init__(self, setting):
        self.s = setting
        self.legs = []
        self.Hd_prev, self.S_prev = 0.0, None
        self.ledger = A.Ledger(setting.id)
        self.n_tranche = 0

    def liability(self):
        return sum(l.q * C.MULT * l.mark for l in self.legs)

    def _exit(self, rec, d, leg, row, kind):
        for sc in C.SCEN:
            if kind == "normal":
                bid, off = float(row.best_bid), float(row.best_offer)
                mid, px = (bid + off) / 2.0, costs.buy_price(bid, off, sc)
                rec[f"close_cost_{sc}"] -= leg.q * C.MULT * (px - mid)
            else:
                px = costs.fallback_buy(leg.mark, sc)
                rec[f"close_cost_{sc}"] -= leg.q * C.MULT * (px - leg.mark)
            rec[f"cash_{sc}"] -= leg.q * C.MULT * px
            rec[f"fees_{sc}"] -= costs.fee(leg.q, sc)
            rec[f"cash_{sc}"] -= costs.fee(leg.q, sc)
        if kind != "normal":
            rec["kappa_exits"] += 1
        self.ledger.legs.append({"date": d, "setting": self.s.id, "tranche": leg.tranche, "optionid": leg.oid,
                                 "event": kind, "exdate": leg.exdate, "K": leg.K, "cp": leg.cp, "q": leg.q,
                                 "mark": leg.mark, "sched": leg.sched, "k": leg.k})
        self.legs.remove(leg)

    def step(self, v, dec, last_day=False, entry_on_last_day=None):
        d, S = v.d, v.S
        rec = self.ledger.new_day(d)
        # (1) mark
        rows = v.leg_rows([l.oid for l in self.legs])
        for leg, row in zip(list(self.legs), rows):
            leg.held += 1
            leg.moved = row is not None and int(row.exdate) != leg.exdate          # A13-4 8e
            rec["exdate_moved"] += leg.moved
            fx, fs = (int(row.exdate), leg.settle) if leg.moved else (leg.exdate, None)
            valid = row is not None and bool(ch.valid_quote([row.best_bid], [row.best_offer], leg.tau)[0])
            if valid:
                new = (float(row.best_bid) + float(row.best_offer)) / 2.0
                if ch.iv_valid([row.impl_volatility])[0]:
                    leg.last_iv = float(row.impl_volatility)
                if leg.fallback_prev:
                    j = -leg.q * C.MULT * (new - leg.mark)
                    rec["jump_sum"] += j
                    rec["jump_absmax"] = max(rec["jump_absmax"], abs(j))
            else:
                F, src, r, T = v.forward(fx, fs)
                if not T > 0:
                    raise A.InvariantError(f"T1 <= 0 for a held leg on {d}")
                rec["fwd_spot"] += src == "spot"
                new = float(ch.bs(F, leg.K, T, leg.last_iv, r, S, leg.cp)[0])
            rec["mark"] -= leg.q * C.MULT * (new - leg.mark)
            leg.mark, leg.valid, leg.fallback_prev = new, valid, not valid
            rec["q_marked"] += leg.q
            rec["q_fallback"] += 0.0 if valid else leg.q
            if row is not None and math.isfinite(float(row.delta)):
                leg.delta = float(row.delta)
            elif row is not None and ch.iv_valid([row.impl_volatility])[0]:
                F, src, r, T = v.forward(fx, fs)
                leg.delta = float(ch.bs(F, leg.K, T, float(row.impl_volatility), r, S, leg.cp)[1])
                rec["delta_bs"] += 1
            else:
                rec["delta_prev"] += 1
        rec["u"] = rec["q_fallback"] / rec["q_marked"] if rec["q_marked"] > 0 else 0.0
        # (2) hedge P&L of yesterday's hedge
        if self.S_prev is not None:
            rec["hedge_pnl"] = self.Hd_prev * (S - self.S_prev)
            for sc in C.SCEN:
                rec[f"cash_{sc}"] += rec["hedge_pnl"]
        # (3) exits
        row_of = dict(zip([l.oid for l in self.legs], rows))
        for leg in list(self.legs):
            row = row_of.get(leg.oid)
            if last_day:
                self._exit(rec, d, leg, row, "normal" if leg.valid else "forced_kappa")
                continue
            if leg.deferring:
                leg.k += 1                                    # processed business days since the scheduled day
            elif leg.held >= self.s.H or dte1(d, leg.exdate) <= C.DTE_EXIT:
                leg.deferring, leg.sched, leg.k = True, d, 0
            if leg.deferring:
                if leg.valid:
                    self._exit(rec, d, leg, row, "normal")
                elif last_chance(v.cal, d, leg.exdate, leg.k, leg.ltd if leg.moved else None):
                    self._exit(rec, d, leg, row, "kappa")
                else:
                    rec["deferred"] += 1
        # (4) entries
        ok, m, reason = dec
        rec["reason"] = reason
        if ok and not (last_day and not entry_on_last_day):
            sel = select(v, self.s.structure, self.s.H)
            rec["sel_status"] = sel["status"]
            rec["defect_ii"] = int(ch.defect_ii("in_range" if sel["status"] in ("ok", "no_candidate") else sel["status"],
                                                sel["status"] == "ok"))
            if sel["status"] == "ok":
                self._open(rec, v, sel, m)
                if last_day:
                    for leg in [l for l in self.legs if l.tranche == self.n_tranche]:
                        self._exit(rec, d, leg, v.leg_rows([leg.oid])[0], "normal")
        # (5) hedge
        Hd = sum(l.q * C.MULT * l.delta for l in self.legs)
        rec["Hd"] = Hd
        for sc in C.SCEN:
            hc = costs.hedge_cost(Hd - self.Hd_prev, S, sc)
            rec[f"hedge_cost_{sc}"] -= hc
            rec[f"cash_{sc}"] -= hc
        self.Hd_prev, self.S_prev = Hd, S
        tr = {l.tranche: l.S_open * l.q * C.MULT for l in self.legs}
        rec["n_tranches"], rec["notional_open"] = len(tr), float(sum(tr.values()))
        if rec["n_tranches"] > self.s.H + C.DEFER_MAX:
            raise A.InvariantError(f"{rec['n_tranches']} open tranches > H + 3 on {d}")
        for l in self.legs:
            if d > (l.ltd if l.moved else v.cal.last_trading(l.exdate)):
                raise A.InvariantError(f"leg held after its last trading day on {d}")
        A.Ledger.close_day(rec)
        return rec

    def _open(self, rec, v, sel, m):
        S = v.S
        q = C.N0 / self.s.H * m / (S * C.MULT)
        self.n_tranche += 1
        F, src, r, T = v.forward(sel["exdate"])
        for row in sel["legs"]:
            bid, off = float(row.best_bid), float(row.best_offer)
            mid = (bid + off) / 2.0
            for sc in C.SCEN:
                px = costs.sell_price(bid, off, sc)
                rec[f"open_cost_{sc}"] -= q * C.MULT * (mid - px)
                rec[f"cash_{sc}"] += q * C.MULT * px
                rec[f"fees_{sc}"] -= costs.fee(q, sc)
                rec[f"cash_{sc}"] -= costs.fee(q, sc)
            iv = float(row.impl_volatility)
            dl = float(row.delta) if math.isfinite(float(row.delta)) else float(
                ch.bs(F, float(row.K), T, iv, r, S, int(row.cp_flag))[1])
            leg = Leg(oid=int(row.optionid), tranche=self.n_tranche, exdate=int(row.exdate), K=float(row.K),
                      cp=int(row.cp_flag), q=q, tau=C.TAU[self.s.structure], mark=mid, last_iv=iv, delta=dl, held=0,
                      deferring=False, sched=None, k=0, valid=True, fallback_prev=False, S_open=S, opened=v.d,
                      settle=v.cal.settlement(int(row.exdate)), ltd=v.cal.last_trading(int(row.exdate)), moved=False)
            self.legs.append(leg)
            self.ledger.legs.append({"date": v.d, "setting": self.s.id, "tranche": self.n_tranche, "optionid": leg.oid,
                                     "event": "open", "exdate": leg.exdate, "K": leg.K, "cp": leg.cp, "q": q, "mark": mid})
        rec["opened"] = 1


def run(mkt, settings, start, end, forced_last_day=False, entry_on_last_day=None, engines=None, on_day=None):
    """Process the market's business days start..end for the settings; returns {sid: Engine}. engines continues
    existing engines (continuous simulation). Day-level defect (i) is stored in each ledger row."""
    eng = engines or {s.id: Engine(s) for s in settings}
    days = [d for d in mkt.days if start <= d <= end]
    if forced_last_day and entry_on_last_day is None:
        raise RuntimeError("N2 (entries on the data-end day) is not decided; refusing a forced-close run")
    for i, d in enumerate(days):
        v = mkt.day(d)
        j = mkt._di[d]
        t = mkt.days[j - 1] if j > 0 else None
        sig = signals.signal(mkt, t)
        last = forced_last_day and i == len(days) - 1
        for s in settings:
            dec = signals.decide(s, sig)
            rec = eng[s.id].step(v, dec, last_day=last, entry_on_last_day=entry_on_last_day)
            rec["defect_i"] = int(not v.has_options or not v.has_close)
            rec["curve_prev"] = int(v.curve_prev)
        if on_day:
            on_day(d)
    return eng
