"""0-row query results (draft A4, Kyle 2026-10-27). raw_sql returns a 0-row result with object columns; the stock dev
download stopped on the pandas FutureWarning from concatenating one. With warnings as errors, day_bars must not warn and
must give, bit for bit, the bars computed without the 0-row frame: the frozen pre-change code (ref_v1_download) with a
0-row frame of the query's own dtypes in its place. The 0-row frames are read through SQLite with raw_sql's arguments
(tests/rawsql.py). Synthetic frames only; no WRDS, no files."""
import os
import sys
import warnings

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bars  # noqa: E402
import download  # noqa: E402
import ref_v1_download  # noqa: E402  (frozen pre-change code)
from rawsql import raw_frame  # noqa: E402
from test_bars import null_free  # noqa: E402  (nullable dtypes as raw_sql returns them)

QC = ["time_m", "time_m_nano", "best_bid", "best_bidsizeshares", "best_ask", "best_asksizeshares", "qu_cond", "qu_source",
      "natbbo_ind"]
TC = ["time_m", "time_m_nano", "price", "size", "tr_scond", "tr_corr"]
SEEDS = [1, 2, 3]
NULL_KEYS = ("n_quote_null", "n_trade_null", "null_counts")                    # added after ref_v1 (A3)


_SES, _PRE, _TRD = null_free(1)


def raw_empty(cols):
    """A 0-row raw_sql result (object columns), read through SQLite."""
    d = raw_frame(_TRD if cols == TC else _SES, rows=0)
    assert list(d.columns) == cols and len(d) == 0 and all(t == object for t in d.dtypes)
    return d


def strict(nbbo, pre, trades):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                         # the downloader treats warnings as failures
        return download.day_bars(nbbo, pre, trades)


def reference(nbbo, pre, trades):
    """Pre-change code; its inputs hold no 0-row object frame, so it does not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return ref_v1_download.day_bars(nbbo, pre, trades)


def assert_same(got, ref):
    (b, st), (b0, st0) = got, ref
    for c in bars.COLS:
        assert b[c].dtype == b0[c].dtype and b[c].shape == b0[c].shape and b[c].tobytes() == b0[c].tobytes(), c
    assert {k: v for k, v in st.items() if k not in NULL_KEYS} == st0
    assert (st["n_quote_null"], st["n_trade_null"]) == (0, 0) and set(st["null_counts"].values()) == {0}


def excluded_all(t):
    return t.assign(tr_corr=pd.array(["01"] * len(t), dtype="string"))       # tr_corr '01': every trade excluded


@pytest.mark.parametrize("seed", SEEDS)
def test_preopen_zero_rows(seed):
    ses, pre, t = null_free(seed)
    assert_same(strict(ses, raw_empty(QC), t), reference(ses, pre.iloc[:0], t))


@pytest.mark.parametrize("seed", SEEDS)
def test_session_zero_rows(seed):
    ses, pre, t = null_free(seed)
    assert_same(strict(raw_empty(QC), pre, t), reference(ses.iloc[:0], pre, t))


@pytest.mark.parametrize("seed", SEEDS)
def test_preopen_and_session_zero_rows(seed):
    ses, pre, t = null_free(seed)
    got = strict(raw_empty(QC), raw_empty(QC), t)
    assert_same(got, reference(ses.iloc[:0], pre.iloc[:0], t))
    b = got[0]
    assert all(int(abs(b[c]).sum()) == 0 for c in ("bid", "ask", "bsz", "asz", "n_upd"))
    assert (b["vol_unc"] == b["vol"]).all()                                   # no NBBO before any trade


@pytest.mark.parametrize("seed", SEEDS)
def test_trades_zero_rows(seed):
    ses, pre, t = null_free(seed)
    got = strict(ses, pre, raw_empty(TC))
    assert_same(got, reference(ses, pre, t.iloc[:0]))
    assert all(int(got[0][c].sum()) == 0 for c in ("vol", "ntr", "px_last"))


@pytest.mark.parametrize("seed", SEEDS)
def test_all_trades_excluded(seed):
    ses, pre, t = null_free(seed)
    got = strict(ses, pre, excluded_all(t))
    assert_same(got, reference(ses, pre, excluded_all(t)))
    b0 = reference(ses, pre, t.iloc[:0])[0]
    assert all(got[0][c].tobytes() == b0[c].tobytes() for c in bars.COLS)       # bars as with no trades
    assert got[1]["n_trade_valid"] == 0 and got[1]["n_trade_excluded"] == got[1]["n_trade_session"] > 0


@pytest.mark.parametrize("seed", SEEDS)
def test_preopen_zero_rows_and_all_trades_excluded(seed):
    ses, pre, t = null_free(seed)
    assert_same(strict(ses, raw_empty(QC), excluded_all(t)), reference(ses, pre.iloc[:0], excluded_all(t)))


def test_all_three_zero_rows():
    b, st = strict(raw_empty(QC), raw_empty(QC), raw_empty(TC))
    assert all(b[c].shape == (bars.NSEC,) and b[c].dtype == bars.DTYPE and int(abs(b[c]).sum()) == 0 for c in bars.COLS)
    assert (st["n_trade_valid"], st["n_quote_null"], st["n_trade_null"]) == (0, 0, 0)
