"""Synthetic tests of bars.build_bars with hand-computed expectations (no real data).
Prices in the expectations are 1e-4 dollars (10.01 -> 100100). Rules: Kyle 2026-10-27 (bars.py docstring)."""
import io
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bars  # noqa: E402

NS = bars.NS


def ns(h, m, s, frac=0):
    return ((h * 3600 + m * 60 + s) * NS) + frac


def run(quotes, trades):
    """quotes: (t, bid, ask, bsz, asz, ok); trades: (t, px, size, tr_corr, tr_scond)."""
    q = list(zip(*quotes)) if quotes else [[], [], [], [], [], []]
    t = list(zip(*trades)) if trades else [[], [], [], [], []]
    corr, sc = np.array(t[3], dtype=object), np.array(t[4], dtype=object)
    tok = bars.trade_ok(corr, sc) if trades else np.array([], bool)
    odd = bars.odd_lot(sc) if trades else np.array([], bool)
    return bars.build_bars(np.array(q[0], np.int64), np.array(q[1], float), np.array(q[2], float), np.array(q[3], np.int64),
                           np.array(q[4], np.int64), np.array(q[5], bool),
                           np.array(t[0], np.int64), np.array(t[1], float), np.array(t[2], np.int64), tok, odd)


QUOTES = [
    (ns(9, 29, 59, 500_000_000), 10.00, 10.02, 100, 200, True),   # Q1 pre-open: opening state
    (ns(9, 30, 0), 10.01, 10.03, 100, 100, True),                   # Q2 exactly 09:30:00 -> second 0
    (ns(9, 30, 2, 200_000_000), 9.99, 10.04, 50, 50, True),         # Q3 second 2
    (ns(9, 30, 2, 700_000_000), 10.00, 10.02, 60, 70, True),        # Q4 second 2 (last of the second)
    (ns(9, 30, 3, 500_000_000), 10.05, 10.04, 10, 10, True),        # Q5 crossed -> xl, excluded from state and extremes
    (ns(9, 30, 4, 100_000_000), 10.00, 10.03, 0, 10, True),         # Q6 zero size -> invalid, not xl
    (ns(9, 30, 5), 10.02, 10.02, 10, 10, True),                     # Q7 locked -> valid (state, extremes) and xl
    (ns(9, 30, 6, 300_000_000), 10.03, 10.05, 10, 10, False),       # Q8 code filter fails -> invalid, not xl
    (ns(15, 59, 59, 999_999_999), 10.10, 10.12, 5, 5, True),        # Q9 last ns of the session -> second 23,399
    (ns(16, 0, 0), 10.20, 10.22, 5, 5, True),                       # Q10 16:00:00 -> outside, not counted
]


def quad(b, cols, s):
    return tuple(int(b[c][s]) for c in cols)


ST, EX = ("bid", "ask", "bsz", "asz"), ("bid_max", "bid_min", "ask_max", "ask_min")


def test_state_carry_extremes_and_boundaries():
    b, st = run(QUOTES, [])
    # second 0: end state Q2; start state Q1 is included in the extremes
    assert quad(b, ST, 0) == (100100, 100300, 100, 100) and quad(b, EX, 0) == (100100, 100000, 100300, 100200)
    assert (b["n_upd"][0], b["n_xl"][0]) == (1, 0)
    # second 1: no update -> carried state; extremes = carried state only
    assert quad(b, ST, 1) == (100100, 100300, 100, 100) and quad(b, EX, 1) == (100100, 100100, 100300, 100300)
    assert b["n_upd"][1] == 0
    # second 2: start state 10.01/10.03 + Q3 + Q4
    assert quad(b, ST, 2) == (100000, 100200, 60, 70) and quad(b, EX, 2) == (100100, 99900, 100400, 100200)
    assert b["n_upd"][2] == 2
    # seconds 3, 4: crossed and zero-size quotes do not move the state or the extremes
    for s, xl in ((3, 1), (4, 0)):
        assert quad(b, ST, s) == (100000, 100200, 60, 70) and quad(b, EX, s) == (100000, 100000, 100200, 100200), s
        assert (b["n_upd"][s], b["n_xl"][s]) == (1, xl), s
    # second 5: the locked quote sets the state and enters the extremes, and counts in n_xl
    assert quad(b, ST, 5) == (100200, 100200, 10, 10) and quad(b, EX, 5) == (100200, 100000, 100200, 100200)
    assert (b["n_upd"][5], b["n_xl"][5]) == (1, 1)
    # second 6: code-filtered quote: counted in n_upd only
    assert quad(b, ST, 6) == (100200, 100200, 10, 10) and quad(b, EX, 6) == (100200, 100200, 100200, 100200)
    assert (b["n_upd"][6], b["n_xl"][6]) == (1, 0)
    # carried all day; last second gets Q9, 16:00:00 excluded
    assert quad(b, ST, 23398) == (100200, 100200, 10, 10)
    assert quad(b, ST, 23399) == (101000, 101200, 5, 5) and quad(b, EX, 23399) == (101000, 100200, 101200, 100200)
    assert int(b["n_upd"].sum()) == 8 and int(b["n_xl"].sum()) == 2
    assert st["init_state"] and st["first_state_sec"] == 0
    assert (st["n_quote_preopen"], st["n_quote_session"], st["n_quote_valid_session"]) == (1, 8, 5)
    assert (st["n_quote_code_excluded"], st["n_quote_crossed"]) == (1, 1)


def test_crossed_quote_never_enters_state_even_first():
    b, st = run([(ns(9, 30, 1), 10.05, 10.04, 5, 5, True), (ns(9, 30, 3), 10.00, 10.01, 5, 5, True)], [])
    assert quad(b, ST, 1) == (0, 0, 0, 0) and quad(b, EX, 1) == (0, 0, 0, 0) and b["n_xl"][1] == 1
    assert quad(b, ST, 3) == (100000, 100100, 5, 5) and st["first_state_sec"] == 3


def test_no_quote_before_first_update():
    b, st = run([(ns(9, 30, 5, 250_000_000), 10.00, 10.01, 10, 20, True)], [])
    for c in ST + EX:
        assert (b[c][:5] == 0).all(), c
    assert quad(b, EX, 5) == (100000, 100000, 100100, 100100)
    assert quad(b, ST, 6) == (100000, 100100, 10, 20)
    assert not st["init_state"] and st["first_state_sec"] == 5


TQ = [(ns(9, 30, 1), 10.00, 10.02, 100, 100, True),                 # A
      (ns(9, 30, 2, 500_000_000), 10.01, 10.03, 100, 100, True),    # B
      (ns(9, 30, 5, 500_000_000), 10.04, 10.04, 100, 100, True)]    # C locked
TRADES = [
    (ns(9, 30, 0, 500_000_000), 10.00, 100, "00", "@"),             # T1 no earlier NBBO -> unc
    (ns(9, 30, 1), 10.00, 50, "00", "@"),                           # T2 same ns as A -> A not used -> unc
    (ns(9, 30, 1, 200_000_000), 10.00, 30, "00", "@F"),             # T3 <= bid(A) -> bid
    (ns(9, 30, 1, 300_000_000), 10.02, 20, "00", ""),               # T4 >= ask(A) -> ask
    (ns(9, 30, 1, 400_000_000), 10.01, 10, "00", "@"),              # T5 inside -> in
    (ns(9, 30, 1, 500_000_000), 10.50, 1000, "08", "@"),            # T6 correction -> excluded (corr=08)
    (ns(9, 30, 1, 600_000_000), 10.00, 7, "00", "4 I"),             # T7 derivatively priced -> excluded (scond=4 I)
    (ns(9, 30, 2, 500_000_000), 10.02, 40, "00", "@"),              # T8 same ns as B -> uses A -> ask
    (ns(9, 30, 2, 600_000_000), 10.01, 5, "00", "F I"),             # T9 B -> <= bid -> bid, odd lot
    (ns(9, 30, 3), 10.00, 1, "00", "@"),                            # T10 same ns as T11 -> bid
    (ns(9, 30, 3), 10.03, 2, "00", "@"),                            # T11 later in input -> px_last; ask
    (ns(9, 30, 4), 10.05, 3, "00", "@  I"),                         # T12 odd lot -> ask
    (ns(9, 30, 4, 100_000_000), 10.02, 4, "00", "@ T"),             # T13 'T' not allowed -> excluded (scond=@ T)
    (ns(9, 30, 4, 200_000_000), 10.02, 6, "00", "@ TI"),            # T14 'T' not allowed -> excluded (scond=@ TI)
    (ns(9, 30, 6), 10.04, 8, "00", "I"),                            # T15 locked C: price <= bid -> bid, odd lot
    (ns(16, 0, 0), 10.00, 500, "00", "6"),                          # T16 16:00:00 -> outside, not counted anywhere
    (ns(15, 59, 59, 999_000_000), 10.04, 9, "00", "@"),             # T17 last second, locked C -> bid
]
COLS8 = ("vol", "ntr", "px_last", "px_min", "px_max", "vol_bid", "vol_ask", "vol_in", "vol_unc", "vol_bid_odd", "vol_ask_odd")
EXP = {0: (100, 1, 100000, 100000, 100000, 0, 0, 0, 100, 0, 0), 1: (110, 4, 100100, 100000, 100200, 30, 20, 10, 50, 0, 0),
       2: (45, 2, 100100, 100100, 100200, 5, 40, 0, 0, 5, 0), 3: (3, 2, 100300, 100000, 100300, 1, 2, 0, 0, 0, 0),
       4: (3, 1, 100500, 100500, 100500, 0, 3, 0, 0, 0, 3), 6: (8, 1, 100400, 100400, 100400, 8, 0, 0, 0, 8, 0),
       23399: (9, 1, 100400, 100400, 100400, 9, 0, 0, 0, 0, 0)}


def test_trade_fields_sides_and_odd_lots():
    b, st = run(TQ, TRADES)
    for s, e in EXP.items():
        assert quad(b, COLS8, s) == e, s
    others = np.ones(bars.NSEC, bool)
    others[list(EXP)] = False
    for c in COLS8:
        assert (b[c][others] == 0).all(), c
    assert (st["n_trade_rows"], st["n_trade_session"], st["n_trade_valid"], st["n_trade_excluded"], st["last_trade_sec"]) == (17, 16, 12, 4, 23399)


def test_volume_identity_and_odd_bounds():
    b, _ = run(TQ, TRADES)
    assert (b["vol"] == b["vol_bid"] + b["vol_ask"] + b["vol_in"] + b["vol_unc"]).all()
    assert (b["vol_bid_odd"] <= b["vol_bid"]).all() and (b["vol_ask_odd"] <= b["vol_ask"]).all()


def test_trade_exclusions_by_code():
    t = list(zip(*TRADES))
    ex = bars.trade_exclusions(np.array(t[0], np.int64), np.array(t[3], dtype=object), np.array(t[4], dtype=object))
    assert ex == {"corr=08": 1, "scond=4 I": 1, "scond=@ T": 1, "scond=@ TI": 1}       # T16 is outside the session


def test_trade_code_allowlist():
    ok = bars.trade_ok(np.array(["00"] * 9 + ["08", "10"], dtype=object),
                       np.array(["@F I", "", "I", "@", "F", "@  I", "@  W", "@O X", "@4", "@", "@"], dtype=object))
    assert list(ok) == [True] * 6 + [False] * 5
    assert list(bars.odd_lot(np.array(["@F I", "I", "@", "", "F"], dtype=object))) == [True, True, False, False, False]


def test_natbbo_no_nbbo_codes_filtered():
    ok = bars.quote_ok(np.array(["R"] * 6 + ["Y"], dtype=object), np.array(["C", "C", "N", "N", "C", "N", "C"], dtype=object),
                       np.array(["G", "U", "2", "4", "O", "1", "G"], dtype=object))
    assert list(ok) == [True, True, True, True, False, False, False]


def test_early_close_and_open_boundary():
    q = [(ns(9, 29, 0), 20.00, 20.01, 1, 1, True), (ns(12, 59, 59), 20.10, 20.11, 3, 4, True)]
    t = [(ns(9, 29, 59, 999_999_999), 20.00, 5, "00", "@"),           # before 09:30:00 -> excluded
         (ns(9, 30, 0), 20.01, 6, "00", "@"),                         # 09:30:00.000 -> second 0, ask side (pre-open state)
         (ns(12, 59, 59, 500_000_000), 20.10, 7, "00", "@")]          # last trade of an early-close day -> second 12,599
    b, st = run(q, t)
    assert b["vol"].shape == (23400,) and int(b["vol"][0]) == 6 and int(b["vol_ask"][0]) == 6
    assert st["last_trade_sec"] == 12599 and int(b["ntr"][12599]) == 1 and (b["ntr"][12600:] == 0).all()
    assert quad(b, ST, 23399) == (201000, 201100, 3, 4)                                      # carried to 16:00
    assert st["n_trade_session"] == 2


def test_dtypes_shape_and_roundtrip():
    b, _ = run(QUOTES, TRADES)
    assert tuple(b) == bars.COLS and len(bars.COLS) == 21
    assert all(b[c].dtype == np.int32 and b[c].shape == (23400,) for c in bars.COLS)
    buf = io.BytesIO()
    bars.save(buf, b)
    buf.seek(0)
    with np.load(buf) as z:
        assert sorted(z.files) == sorted(bars.COLS)
        assert all((z[c] == b[c]).all() and z[c].dtype == np.int32 for c in bars.COLS)


def test_ticks_subtick_and_overflow():
    assert list(bars.ticks([123.4567, 0.0001, 10.0])) == [1234567, 1, 100000]
    _, st = run(TQ, [(ns(9, 30, 5), 10.00004, 1, "00", "@")])
    assert st["n_subtick_trade"] == 1
    with pytest.raises(OverflowError):
        run(TQ, [(ns(9, 30, 5), 10.0, 3_000_000_000, "00", "@")])


# ---- NULL rows (SPEC_AMENDMENTS draft A3, 2026-10-27): handled in download.day_bars with nullable frames ----
import datetime as dt  # noqa: E402
import warnings  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download  # noqa: E402
import ref_v1_download  # noqa: E402  (frozen pre-change code)

NA = pd.NA


def tm(sec):
    return [(dt.datetime(2000, 1, 1) + dt.timedelta(microseconds=int(round(x * 1e6)))).time() for x in sec]


def qframe(rows):
    """rows: (sec, bid, ask, bsz, asz); nullable dtypes as wrds raw_sql returns them."""
    s, b, a, bs, as_ = zip(*rows)
    n = len(rows)
    return pd.DataFrame({"time_m": tm(s), "time_m_nano": pd.array([0] * n, dtype="Int16"),
                         "best_bid": pd.array(b, dtype="Float64"), "best_bidsizeshares": pd.array(bs, dtype="Int32"),
                         "best_ask": pd.array(a, dtype="Float64"), "best_asksizeshares": pd.array(as_, dtype="Int32"),
                         "qu_cond": pd.array(["R"] * n, dtype="string"), "qu_source": pd.array(["C"] * n, dtype="string"),
                         "natbbo_ind": pd.array(["G"] * n, dtype="string")})


def tframe(rows):
    """rows: (sec, price, size, tr_scond, tr_corr)."""
    s, p, z, sc, c = zip(*rows)
    return pd.DataFrame({"time_m": tm(s), "time_m_nano": pd.array([0] * len(rows), dtype="Int16"),
                         "price": pd.array(p, dtype="Float64"), "size": pd.array(z, dtype="Int32"),
                         "tr_scond": pd.array(sc, dtype="string"), "tr_corr": pd.array(c, dtype="string")})


PRE = qframe([(34140.0, 10.00, 10.02, 100, 100),              # P1 09:29:00 valid -> opening state
              (34170.0, NA, NA, NA, NA)])                       # P2 09:29:30 all NULL -> invalid (was the 2019-01-23 crash)
SES = qframe([(34201.0, 10.01, 10.03, 200, 200),               # S1 second 1
              (34202.0, NA, 10.04, 50, 50)])                    # S2 second 2: bid NULL -> invalid, n_upd only
TRD = tframe([(34201.5, 10.03, 10, NA, "00"),                  # T1 tr_scond NULL -> '' -> valid, ask side (S1)
              (34202.5, NA, 5, "@", "00"),                      # T2 price NULL -> excluded, 'NULL'
              (34203.0, 10.01, NA, "@", "00"),                  # T3 size NULL -> excluded, 'NULL'
              (34203.5, 10.01, 7, "@", NA),                     # T4 tr_corr NULL -> excluded, 'corr=NULL'
              (34204.0, 10.01, 3, "@", "00")])                  # T5 valid, bid side (S1 bid 10.01)


def test_null_quote_and_trade_rows():
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                         # the downloader treats warnings as failures
        b, st = download.day_bars(SES, PRE, TRD)
    assert quad(b, ST, 0) == (100000, 100200, 100, 100)                         # P2 ignored, P1 is the state at the open
    assert quad(b, ST, 1) == (100100, 100300, 200, 200)
    assert quad(b, ST, 2) == (100100, 100300, 200, 200) and (b["n_upd"][2], b["n_xl"][2]) == (1, 0)
    assert quad(b, EX, 2) == (100100, 100100, 100300, 100300)                   # S2 not in the extremes
    assert (int(b["vol_ask"][1]), int(b["vol"][1]), int(b["ntr"][1])) == (10, 10, 1)
    assert int(b["vol"][2]) == 0 and int(b["vol"][3]) == 0
    assert (int(b["vol_bid"][4]), int(b["vol"][4])) == (3, 3)
    assert (st["n_quote_null"], st["n_trade_null"], st["n_trade_valid"], st["n_trade_excluded"]) == (2, 2, 2, 3)
    assert json.loads(st["excl_by_code"]) == {"NULL": 2, "corr=NULL": 1}
    assert st["n_trade_excluded"] == sum(json.loads(st["excl_by_code"]).values())
    assert (b["vol"] == b["vol_bid"] + b["vol_ask"] + b["vol_in"] + b["vol_unc"]).all()
    # null_counts.tsv fields: P2 (pre-open), S2 (session); T2 price, T3 size, T4 tr_corr, T1 tr_scond
    assert st["null_counts"] == {"q_null_pre": 1, "q_null_session": 1, "t_null_price": 1, "t_null_size": 1,
                                 "t_null_corr": 1, "t_null_scond": 1}


def test_null_nano_is_an_error():
    pre = PRE.copy()
    pre["time_m_nano"] = pd.array([0, NA], dtype="Int16")
    with pytest.raises(ValueError, match="NULL time_m_nano"):
        download.day_bars(SES, pre, TRD)


def null_free(seed):
    r = np.random.default_rng(seed)
    n, m = 3000, 800
    qs = np.sort(r.uniform(34000, 57700, n))
    mid = 100 + np.cumsum(r.choice([-0.01, 0, 0.01], n))
    half = r.choice([0.0, 0.005, 0.01], n, p=[0.05, 0.75, 0.2])               # some locked quotes
    q = qframe(list(zip(qs, mid - half, mid + half, r.integers(0, 500, n), r.integers(1, 500, n))))
    q["qu_cond"] = pd.array(r.choice(["R", "R", "R", "Y"], n), dtype="string")
    pre, ses = q[q.time_m < dt.time(9, 30)].reset_index(drop=True), q[q.time_m >= dt.time(9, 30)].reset_index(drop=True)
    ts = np.sort(r.uniform(34100, 57700, m))
    j = np.clip(np.searchsorted(qs, ts) - 1, 0, n - 1)
    px = mid[j] + r.choice([-1.0, 0.0, 1.0], m) * half[j] + r.choice([0.0, 0.005], m, p=[0.9, 0.1])
    t = tframe(list(zip(ts, px, r.integers(1, 400, m), r.choice(["@", "", "F I", "I", "@4 W", "T", "@ Z"], m),
                        r.choice(["00", "00", "00", "08", "12"], m))))
    return ses, pre, t


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_null_free_input_matches_previous_code(seed):
    ses, pre, t = null_free(seed)
    b_new, st_new = download.day_bars(ses, pre, t)
    b_old, st_old = ref_v1_download.day_bars(ses, pre, t)
    assert all((b_new[c] == b_old[c]).all() for c in bars.COLS)
    assert {k: v for k, v in st_new.items() if k not in ("n_quote_null", "n_trade_null", "null_counts")} == st_old
    assert (st_new["n_quote_null"], st_new["n_trade_null"]) == (0, 0)
    assert set(st_new["null_counts"].values()) == {0}
