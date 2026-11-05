"""Research 15 copy of download.py (adapted from research12/bars1s/tests/test_stocks.py): train split routing, universe15
files, year boundary, absent days, resume, and bars identical to the research-12 downloader for the same inputs. Fake
fetcher and fake master (no WRDS); outputs under $BARS1S_TEST_DIR (nothing deleted)."""
import datetime as dt
import importlib.util
import os
import sys
import uuid

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R12 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "study12_market_maker", "bars1s")
sys.path.insert(0, HERE)
import bars  # noqa: E402
import download  # noqa: E402

BASE = os.environ.get("BARS1S_TEST_DIR")
pytestmark = pytest.mark.skipif(not BASE, reason="set BARS1S_TEST_DIR (outside the repository)")
DAYS = ["2017-12-28", "2017-12-29", "2018-01-02", "2018-01-03", "2019-01-02"]           # the last one is outside train
MASTER = {"20171228": {"AAA", "BBB"}, "20171229": {"AAA"}, "20180102": {"CCC"}, "20180103": {"AAA", "CCC"},
          "20190102": {"AAA", "CCC"}}


def tm(sec):
    return [(dt.datetime(2000, 1, 1) + dt.timedelta(microseconds=int(x * 1e6))).time() for x in sec]


def frames(seed):
    r = np.random.default_rng(seed)
    qs = np.sort(r.uniform(34000, 57600, 300))
    mid = 50 + np.cumsum(r.choice([-0.01, 0, 0.01], 300))
    nb = pd.DataFrame({"time_m": tm(qs), "time_m_nano": 0, "best_bid": mid - 0.02, "best_bidsizeshares": 100,
                       "best_ask": mid + 0.02, "best_asksizeshares": 100, "qu_cond": "R", "qu_source": "N", "natbbo_ind": "4"})
    ts = np.sort(r.uniform(34200, 57600, 80))
    tr = pd.DataFrame({"time_m": tm(ts), "time_m_nano": 0, "price": mid[:80].round(2), "size": 100,
                       "tr_scond": r.choice(["@", "F", "I", "Z"], 80), "tr_corr": "00"})
    pre = nb[nb.time_m < dt.time(9, 30)]
    return nb[nb.time_m >= dt.time(9, 30)], pre, tr


def setup():
    d = os.path.join(BASE, uuid.uuid4().hex[:8])
    os.makedirs(d)
    pd.DataFrame({"year": 2017, "slot": [1, 2], "sym": ["AAA", "BBB"], "order_rank": [1, 2]}).to_csv(os.path.join(d, "universe15_2017.tsv"), sep="\t", index=False)
    pd.DataFrame({"year": 2018, "slot": [1, 2], "sym": ["AAA", "CCC"], "order_rank": [1, 3]}).to_csv(os.path.join(d, "universe15_2018.tsv"), sep="\t", index=False)
    pd.DataFrame({"date": DAYS, "close_time": "16:00"}).to_csv(os.path.join(d, "calendar15.csv"), index=False)
    return d


def present(day, syms):
    return MASTER[day] & set(syms)


def factory():
    return lambda sym, day: frames(int(day) % 997 + ord(sym[0]))


def test_constants_point_only_at_research15():
    assert download.SPLITS == {"train": ("20150102", "20181231")}
    assert download.BASE == os.path.join(os.environ.get("DATA_DIR", "data"), "research15_data")
    assert download.CALENDAR.endswith(os.path.join("study15_ml_execution", "calendar15.csv"))
    assert download.split_root(download.BASE, "20160104", download.STOCKS).endswith("research15_data/bars_1s_train_stocks")
    for bad in ("20141231", "20190102"):
        with pytest.raises(download.Stop, match="outside every split"):
            download.split_of(bad)


def test_tasks_files_absent_and_resume():
    d = setup()
    cal = os.path.join(d, "calendar15.csv")
    tasks, absent = download.stock_tasks("train", [2017, 2018], present, universe_dir=d, calendar=cal)
    assert tasks == [("AAA", "20171228"), ("BBB", "20171228"), ("AAA", "20171229"), ("CCC", "20180102"),
                     ("AAA", "20180103"), ("CCC", "20180103")]
    assert absent == [("BBB", "20171229"), ("AAA", "20180102")]
    out = os.path.join(d, "out")
    assert download.record_absent(out, absent, log=lambda m: None) == 2
    s = download.run(tasks, factory, base=out, free_fn=lambda p: 10**15, log=lambda m: None, suffix=download.STOCKS)
    assert (s["written"], s["failed"]) == (6, 0)
    root = os.path.join(out, "bars_1s_train_stocks")
    assert sorted(os.listdir(out)) == ["bars_1s_train_stocks"]
    files = sorted(os.path.relpath(os.path.join(a, f), root) for a, _, fs in os.walk(root) for f in fs if f.endswith(".npz"))
    assert files == ["2017/AAA_20171228.npz", "2017/AAA_20171229.npz", "2017/BBB_20171228.npz",
                     "2018/AAA_20180103.npz", "2018/CCC_20180102.npz", "2018/CCC_20180103.npz"]
    assert {f for f in os.listdir(root) if f.endswith(".tsv")} == {"MANIFEST.tsv", "null_counts.tsv", "skipped_absent.tsv"}
    s2 = download.run(tasks, factory, base=out, free_fn=lambda p: 10**15, log=lambda m: None, suffix=download.STOCKS)
    assert (s2["written"], s2["skipped"]) == (0, 6)


def test_bars_identical_to_research12_downloader():
    spec = importlib.util.spec_from_file_location("download12", os.path.join(R12, "download.py"))
    sys.path.insert(0, R12)
    d12 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d12)
    for k in range(5):
        nb, pre, tr = frames(k)
        b15, s15 = download.day_bars(nb, pre, tr)
        b12, s12 = d12.day_bars(nb, pre, tr)
        assert list(b15) == list(bars.COLS) and s15 == s12
        assert all(b15[c].dtype == np.int32 and b15[c].shape == (23_400,) and np.array_equal(b15[c], b12[c]) for c in bars.COLS)


def test_fetch_reads_time_m_nano_only_where_the_table_has_it():
    sqls = []

    class DB:
        def raw_sql(self, sql):
            sqls.append(sql)
            if "information_schema" in sql:
                return pd.DataFrame({"table_name": ["ctm_20170103", "complete_nbbo_20180102", "ctm_20180102"]})
            return pd.DataFrame({"x": [1]})
        close = None
    f = download.make_fetch(DB())
    for day in ("20150102", "20170103", "20180102"):
        sqls.clear()
        f("AAA", day)
        nb, pre, tr = [s for s in sqls if "information_schema" not in s]
        q_nano, t_nano = day >= "20180102", day >= "20170103"
        assert ("0 as time_m_nano" not in nb) == q_nano and ("0 as time_m_nano" not in pre) == q_nano
        assert ("time_m_nano desc" in pre) == q_nano
        assert ("0 as time_m_nano" not in tr) == t_nano
        if q_nano and t_nano:                                          # 2018: the research-12 queries unchanged
            assert (nb, pre, tr) == (download.NBBO_SQL.format(lib="taqm_2018", d=day, s="AAA"),
                                     download.PRE_SQL.format(lib="taqm_2018", d=day, s="AAA", n=download.PREOPEN_ROWS),
                                     download.TRADE_SQL.format(lib="taqm_2018", d=day, s="AAA"))
