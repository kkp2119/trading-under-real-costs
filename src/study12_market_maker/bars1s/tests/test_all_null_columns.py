"""All-NULL columns (draft A4, second fix, Kyle 2026-10-27). raw_sql gives an all-NULL column a value-inferred dtype
(string, pd.NA); the stock dev download stopped twice on the pandas FutureWarning from concatenating such a pre-open or
session frame with the other. Inputs come through SQLite with raw_sql's arguments (tests/rawsql.py). With warnings as
errors, day_bars must give, bit for bit, the result for the same input with that column all-NA in its nullable dtype.
Supplementary: hand-made object columns (all None, all pd.NA) in case the WRDS driver returns object. No WRDS, no files."""
import os
import sys
import warnings

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bars  # noqa: E402
import download  # noqa: E402
from rawsql import raw_frame, typed_null  # noqa: E402
from test_bars import null_free  # noqa: E402

QCOLS = ["best_bid", "best_bidsizeshares", "best_ask", "best_asksizeshares", "qu_cond", "qu_source", "natbbo_ind"]
TCOLS = ["price", "size", "tr_scond", "tr_corr"]
CASES = [("pre", c) for c in QCOLS] + [("nbbo", c) for c in QCOLS] + [("trades", c) for c in TCOLS]      # 18
SEEDS = [1, 2]


def inputs(seed):
    ses, pre, t = null_free(seed)
    src = {"nbbo": ses, "pre": pre, "trades": t}
    return src, {k: raw_frame(v) for k, v in src.items()}


def strict(f):
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                         # the downloader treats warnings as failures
        return download.day_bars(f["nbbo"], f["pre"], f["trades"])


def assert_same(got, ref):
    (b, st), (b0, st0) = got, ref
    for c in bars.COLS:
        assert b[c].dtype == b0[c].dtype and b[c].shape == b0[c].shape and b[c].tobytes() == b0[c].tobytes(), c
    assert st == st0


def object_null(df, col, value):
    d = df.copy()
    d[col] = pd.Series([value] * len(d), dtype=object)
    return d


def test_raw_sql_format_is_reproduced():
    ses, pre, _ = null_free(1)
    full, nul, empty = raw_frame(pre), raw_frame(pre, ["best_bid"]), raw_frame(ses, rows=0)
    assert str(full.best_bid.dtype) == "Float64" and len(nul) == len(pre) > 0
    assert nul.best_bid.isna().all() and str(nul.best_bid.dtype) == "string"           # value-inferred, not Float64
    assert len(empty) == 0 and all(t == object for t in empty.dtypes)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pd.concat([nul, raw_frame(ses)], ignore_index=True)
    assert any(issubclass(x.category, FutureWarning) for x in w)                        # the hazard in this environment


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("frame,col", CASES)
def test_one_all_null_column(frame, col, seed):
    src, raw = inputs(seed)
    got = dict(raw, **{frame: raw_frame(src[frame], [col])})
    ref = dict(raw, **{frame: typed_null(got[frame], raw[frame], [col])})
    assert_same(strict(got), strict(ref))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("col", QCOLS)
def test_same_column_all_null_in_both_quote_frames(col, seed):
    src, raw = inputs(seed)
    got = dict(raw, pre=raw_frame(src["pre"], [col]), nbbo=raw_frame(src["nbbo"], [col]))
    ref = dict(raw, pre=typed_null(got["pre"], raw["pre"], [col]), nbbo=typed_null(got["nbbo"], raw["nbbo"], [col]))
    assert_same(strict(got), strict(ref))


@pytest.mark.parametrize("col", QCOLS)
def test_all_null_column_with_the_other_quote_frame_zero_rows(col):
    src, raw = inputs(1)
    for frame, other in (("pre", "nbbo"), ("nbbo", "pre")):
        got = dict(raw, **{frame: raw_frame(src[frame], [col]), other: raw_frame(src[other], rows=0)})
        ref = dict(got, **{frame: typed_null(got[frame], raw[frame], [col]), other: raw[other].iloc[:0]})
        assert_same(strict(got), strict(ref))


@pytest.mark.parametrize("value", [None, pd.NA], ids=["None", "NA"])
@pytest.mark.parametrize("frame,col", CASES)
def test_object_all_null_column_supplementary(frame, col, value):
    _, raw = inputs(1)
    got = dict(raw, **{frame: object_null(raw[frame], col, value)})
    ref = dict(raw, **{frame: typed_null(raw[frame], raw[frame], [col])})
    assert_same(strict(got), strict(ref))


def test_concat_raw_changes_only_value_free_columns():
    _, raw = inputs(1)
    nul = raw_frame(null_free(1)[1], ["best_bid", "natbbo_ind"])
    q = download.concat_raw([nul, raw["nbbo"]])
    assert dict(q.dtypes) == dict(raw["nbbo"].dtypes)
    assert q.iloc[:len(nul)].best_bid.isna().all() and q.iloc[:len(nul)].natbbo_ind.isna().all()
    assert q.iloc[len(nul):].reset_index(drop=True).equals(raw["nbbo"])
    assert download.concat_raw([raw["nbbo"].iloc[:0], raw["pre"].iloc[:0]]).shape == (0, len(QCOLS) + 2)
