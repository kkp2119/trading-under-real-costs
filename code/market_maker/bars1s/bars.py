"""Research 12 data step 1: one symbol-day of TAQ complete_nbbo + ctm -> 1-second feature bars (regular hours).

Grid: second s = 0 .. 23,399 covers [09:30:00 + s, 09:30:00 + s + 1) (09:30:00 <= t < 16:00:00). Times are integer ns
since midnight (time_m + time_m_nano). Prices are stored as integer 1e-4 dollars, sizes as integer shares; every
column is int32 and every file has the same 21 columns of length 23,400 (an early close keeps the full grid; the last
trade second is reported separately).

Quotes (one NBBO update per row), Kyle 2026-10-27:
  code   = qu_cond 'R' and not a "no NBBO" indicator (CTA source 'C' with natbbo_ind 'O', UTP source 'N' with '1';
           Daily TAQ Client Spec v3.3 Appendix G; G, U, 2, 4 all mean a new NBBO and pass)
  valid  = code and bid > 0, ask > 0, bsz > 0, asz > 0, ask >= bid       (locked quotes are valid)
  xl     = bid > 0, ask > 0, ask <= bid                                   (locked or crossed; crossed are never valid)
  state at the end of second s = last valid update with t < end of s (pre-open updates count; carried forward)
  bid_max .. ask_min of second s = over {state at the start of s} + {valid updates inside s}
  n_upd = all session rows inside s (any validity), n_xl = xl rows inside s
  no state and no valid update -> 0 in the state and extreme columns
Trades, Kyle 2026-10-27:
  valid  = tr_corr '00' and every character of tr_scond in {'@', 'F', 'I', ' '} (empty allowed), price > 0, size > 0,
           inside the session; session trades failing the codes are counted by code (trade_exclusions)
  vol, ntr, px_last (last in time, input order on ties), px_min, px_max  (0 when no trade)
  side   = the last valid NBBO strictly before the trade time (a quote at the same ns is not used):
           price <= bid -> vol_bid (checked first, so a trade at a locked price is bid), price >= ask -> vol_ask,
           bid < price < ask -> vol_in, no earlier valid NBBO -> vol_unc;  vol = vol_bid + vol_ask + vol_in + vol_unc
  vol_bid_odd / vol_ask_odd = the part of vol_bid / vol_ask from trades whose tr_scond contains 'I' (odd lot)
NULL rows (Kyle 2026-10-27, SPEC_AMENDMENTS draft A3; applied in download.day_bars before build_bars):
  quote price or size NULL -> 0, so the row is invalid (it still counts in n_upd inside the session, never in n_xl)
  trade price or size NULL -> excluded, counted as 'NULL' in trade_exclusions; tr_scond NULL -> '' ; tr_corr NULL -> 'NULL'
  (not '00', so excluded and counted as 'corr=NULL'); time_m_nano NULL -> error (no rule yet)"""
import collections

import numpy as np

NS = 1_000_000_000
T0, T1 = 34_200, 57_600                    # 09:30:00, 16:00:00 in seconds since midnight
NSEC = T1 - T0                             # 23,400
SCALE = 10_000                             # 1e-4 dollars
DTYPE = np.int32
COLS = ("bid", "ask", "bsz", "asz", "bid_max", "bid_min", "ask_max", "ask_min", "n_upd", "n_xl",
        "vol", "ntr", "px_last", "px_min", "px_max", "vol_bid", "vol_ask", "vol_in", "vol_unc", "vol_bid_odd", "vol_ask_odd")
I32_MAX = np.iinfo(np.int32).max

VALID_QU_COND = frozenset({"R"})
NO_NBBO = {("C", "O"), ("N", "1")}         # (qu_source, natbbo_ind) meaning "no NBBO"
VALID_TR_CORR = frozenset({"00"})
ALLOWED_SCOND = frozenset({"@", "F", "I", " "})


def quote_ok(qu_cond, qu_source=None, natbbo_ind=None):
    ok = np.isin(np.asarray(qu_cond, dtype=object), list(VALID_QU_COND))
    if qu_source is not None:
        pairs = zip(np.asarray(qu_source, dtype=object), np.asarray(natbbo_ind, dtype=object))
        ok &= np.array([(s, i) not in NO_NBBO for s, i in pairs], dtype=bool)
    return ok


def trade_ok(tr_corr, tr_scond):
    corr = np.isin(np.asarray(tr_corr, dtype=object), list(VALID_TR_CORR))
    good = np.array([set(x or "") <= ALLOWED_SCOND for x in np.asarray(tr_scond, dtype=object)], dtype=bool)
    return corr & good


def odd_lot(tr_scond):
    return np.array(["I" in (x or "") for x in np.asarray(tr_scond, dtype=object)], dtype=bool)


def trade_exclusions(t_t, tr_corr, tr_scond, null=None):
    """Session trades failing the codes, counted by code: 'NULL' when price or size is NULL (mask `null`), else
    'corr=<tr_corr>' when tr_corr is not '00', else 'scond=<tr_scond>'."""
    t_t = np.asarray(t_t, dtype=np.int64)
    ins = (t_t >= T0 * NS) & (t_t < T1 * NS)
    corr, sc = np.asarray(tr_corr, dtype=object)[ins], np.asarray(tr_scond, dtype=object)[ins]
    nl = np.zeros(int(ins.sum()), bool) if null is None else np.asarray(null, dtype=bool)[ins]
    c = collections.Counter()
    for a, b, n in zip(corr, sc, nl):
        if n:
            c["NULL"] += 1
        elif a not in VALID_TR_CORR:
            c[f"corr={a}"] += 1
        elif not set(b or "") <= ALLOWED_SCOND:
            c[f"scond={b}"] += 1
    return dict(sorted(c.items()))


def ticks(px):
    """Dollars -> integer 1e-4 dollars, nearest (half up for positive prices)."""
    return np.floor(np.asarray(px, dtype=np.float64) * SCALE + 0.5).astype(np.int64)


def _subtick(px):
    x = np.asarray(px, dtype=np.float64) * SCALE
    return int(np.sum(np.abs(x - np.round(x)) > 1e-6))


def _i32(name, a):
    a = np.asarray(a)
    if a.size and (a.max() > I32_MAX or a.min() < -I32_MAX):
        raise OverflowError(f"column {name} does not fit int32")
    return a.astype(DTYPE)


def build_bars(q_t, q_bid, q_ask, q_bsz, q_asz, q_ok, t_t, t_px, t_sz, t_ok, t_odd=None):
    """Arrays for one symbol-day (any order; sorted stably by time here). q_ok / t_ok: code masks; t_odd: odd-lot mask.
    Returns (bars: dict of 21 int32 arrays of length NSEC, stats: dict of counts)."""
    q_t = np.asarray(q_t, dtype=np.int64)
    oq = np.argsort(q_t, kind="stable")
    q_t = q_t[oq]
    qbf, qaf = np.asarray(q_bid, dtype=np.float64)[oq], np.asarray(q_ask, dtype=np.float64)[oq]
    qb, qa = ticks(qbf), ticks(qaf)
    qbs, qas = np.asarray(q_bsz, dtype=np.int64)[oq], np.asarray(q_asz, dtype=np.int64)[oq]
    qok = np.asarray(q_ok, dtype=bool)[oq]
    valid = qok & (qb > 0) & (qa > 0) & (qbs > 0) & (qas > 0) & (qa >= qb)
    xl = (qb > 0) & (qa > 0) & (qa <= qb)
    vt, vb, va, vbs, vas = q_t[valid], qb[valid], qa[valid], qbs[valid], qas[valid]

    edges = (T0 + np.arange(NSEC + 1, dtype=np.int64)) * NS          # second boundaries
    last = np.searchsorted(vt, edges, "left") - 1                     # last valid update strictly before each boundary
    have = last >= 0
    li = np.maximum(last, 0)

    def state(v):
        return np.where(have, v[li] if len(v) else 0, 0)
    sb, sa, sbs, sas = state(vb), state(va), state(vbs), state(vas)    # length NSEC + 1: index s = state at start of s
    out = {"bid": sb[1:], "ask": sa[1:], "bsz": sbs[1:], "asz": sas[1:]}

    in_s = (q_t >= T0 * NS) & (q_t < T1 * NS)
    k_all = (q_t[in_s] // NS - T0).astype(np.int64)
    out["n_upd"] = np.bincount(k_all, minlength=NSEC)
    out["n_xl"] = np.bincount(k_all[xl[in_s]], minlength=NSEC)
    vin = (vt >= T0 * NS) & (vt < T1 * NS)
    kv = (vt[vin] // NS - T0).astype(np.int64)
    for name, v, start, fn, init in (("bid_max", vb, sb, np.maximum, -1), ("bid_min", vb, sb, np.minimum, I32_MAX),
                                     ("ask_max", va, sa, np.maximum, -1), ("ask_min", va, sa, np.minimum, I32_MAX)):
        acc = np.full(NSEC, init, dtype=np.int64)
        fn.at(acc, kv, v[vin])
        st = np.where(have[:-1], start[:-1], init)                     # state at the start of each second
        acc = fn(acc, st)
        out[name] = np.where(acc == init, 0, acc)

    t_t = np.asarray(t_t, dtype=np.int64)
    ot = np.argsort(t_t, kind="stable")
    t_t = t_t[ot]
    tpf = np.asarray(t_px, dtype=np.float64)[ot]
    tp, ts = ticks(tpf), np.asarray(t_sz, dtype=np.int64)[ot]
    tok = np.asarray(t_ok, dtype=bool)[ot]
    todd = (np.zeros(len(t_t), bool) if t_odd is None else np.asarray(t_odd, dtype=bool)[ot])
    tin = (t_t >= T0 * NS) & (t_t < T1 * NS)
    tv = tok & tin & (tp > 0) & (ts > 0)
    tt, tp, ts, tpf, todd = t_t[tv], tp[tv], ts[tv], tpf[tv], todd[tv]
    k = (tt // NS - T0).astype(np.int64)

    def vsum(m):
        return np.bincount(k[m], weights=ts[m], minlength=NSEC).round().astype(np.int64)
    out["vol"] = vsum(np.ones(len(k), bool))
    out["ntr"] = np.bincount(k, minlength=NSEC)
    pmin = np.full(NSEC, I32_MAX, dtype=np.int64)
    pmax = np.zeros(NSEC, dtype=np.int64)
    np.minimum.at(pmin, k, tp)
    np.maximum.at(pmax, k, tp)
    out["px_min"] = np.where(out["ntr"] > 0, pmin, 0)
    out["px_max"] = pmax
    lastk = np.zeros(NSEC, dtype=np.int64)
    if len(k):
        idx = np.searchsorted(k, np.arange(NSEC), "right") - 1        # k is sorted because tt is
        ok_idx = (idx >= 0) & (out["ntr"] > 0)
        lastk[ok_idx] = tp[idx[ok_idx]]
    out["px_last"] = lastk
    j = np.searchsorted(vt, tt, "left") - 1                            # strictly earlier valid NBBO
    hj = j >= 0
    jb = np.where(hj, vb[np.maximum(j, 0)] if len(vb) else 0, 0)
    ja = np.where(hj, va[np.maximum(j, 0)] if len(va) else 0, 0)
    on_bid = hj & (tp <= jb)
    on_ask = hj & ~on_bid & (tp >= ja)
    inside = hj & ~on_bid & ~on_ask
    out["vol_bid"], out["vol_ask"], out["vol_in"], out["vol_unc"] = vsum(on_bid), vsum(on_ask), vsum(inside), vsum(~hj)
    out["vol_bid_odd"], out["vol_ask_odd"] = vsum(on_bid & todd), vsum(on_ask & todd)

    bars = {c: _i32(c, out[c]) for c in COLS}
    have_end = have[1:]
    stats = {"n_quote_rows": int(len(q_t)), "n_quote_session": int(in_s.sum()), "n_quote_valid_session": int(vin.sum()),
             "n_quote_code_excluded": int((in_s & ~qok).sum()), "n_quote_crossed": int((in_s & xl & (qa < qb)).sum()),
             "n_quote_preopen": int((q_t < T0 * NS).sum()), "init_state": bool(have[0]),
             "first_state_sec": int(np.argmax(have_end)) if have_end.any() else -1,
             "n_trade_rows": int(len(t_t)), "n_trade_session": int(tin.sum()), "n_trade_valid": int(tv.sum()),
             "n_trade_excluded": int((tin & ~tok).sum()), "last_trade_sec": int(k.max()) if len(k) else -1,
             "n_subtick_trade": _subtick(tpf), "n_subtick_quote": _subtick(qbf[valid]) + _subtick(qaf[valid])}
    return bars, stats


def save(path_handle, bars):
    """np.savez_compressed into an open binary handle (so the caller controls the file name and the atomic rename)."""
    for c in COLS:
        if bars[c].dtype != DTYPE or bars[c].shape != (NSEC,):
            raise ValueError(f"column {c}: dtype {bars[c].dtype} shape {bars[c].shape}")
    np.savez_compressed(path_handle, **{c: bars[c] for c in COLS})


def load(path):
    with np.load(path) as z:
        return {c: z[c] for c in COLS}
