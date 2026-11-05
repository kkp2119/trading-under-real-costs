"""Stock mode of download.py (Kyle 2026-10-27): yearly universe x calendar days, master presence, absent days, empty days,
year boundary, resume, re-appearance. Fake fetcher and fake master (no WRDS); outputs under $BARS1S_TEST_DIR (nothing deleted)."""
import datetime as dt
import os
import sys
import uuid

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import bars  # noqa: E402
import download  # noqa: E402

BASE = os.environ.get("BARS1S_TEST_DIR")
pytestmark = pytest.mark.skipif(not BASE, reason="set BARS1S_TEST_DIR (outside the repository)")
QC = ["time_m", "time_m_nano", "best_bid", "best_bidsizeshares", "best_ask", "best_asksizeshares", "qu_cond", "qu_source", "natbbo_ind"]
TC = ["time_m", "time_m_nano", "price", "size", "tr_scond", "tr_corr"]
DAYS = ["2019-12-30", "2019-12-31", "2020-01-02", "2020-01-03", "2021-01-04"]           # the last one is outside dev
MASTER = {"20191230": {"AAA", "BBB"}, "20191231": {"AAA"},                               # BBB absent on 12-31
          "20200102": {"CCC"}, "20200103": {"AAA", "CCC"}, "20210104": {"AAA", "CCC"}}    # AAA absent 01-02, back 01-03


def tm(sec):
    return [(dt.datetime(2000, 1, 1) + dt.timedelta(microseconds=int(x * 1e6))).time() for x in sec]


def frames(seed):
    r = np.random.default_rng(seed)
    qs = np.sort(r.uniform(34200, 57600, 300))
    mid = 50 + np.cumsum(r.choice([-0.01, 0, 0.01], 300))
    nb = pd.DataFrame({"time_m": tm(qs), "time_m_nano": 0, "best_bid": mid - 0.02, "best_bidsizeshares": 100,
                       "best_ask": mid + 0.02, "best_asksizeshares": 100, "qu_cond": "R", "qu_source": "N", "natbbo_ind": "4"})
    ts = np.sort(r.uniform(34200, 57600, 80))
    tr = pd.DataFrame({"time_m": tm(ts), "time_m_nano": 0, "price": 50.0, "size": 100, "tr_scond": "@", "tr_corr": "00"})
    return nb, nb.iloc[:0], tr


def empty():
    return (pd.DataFrame({c: pd.Series([], dtype=object) for c in QC}), pd.DataFrame({c: pd.Series([], dtype=object) for c in QC}),
            pd.DataFrame({c: pd.Series([], dtype=object) for c in TC}))


def setup():
    d = os.path.join(BASE, uuid.uuid4().hex[:8])
    os.makedirs(d)
    pd.DataFrame({"year": 2019, "slot": [1, 2], "sym": ["AAA", "BBB"], "order_rank": [1, 2]}).to_csv(os.path.join(d, "universe_2019.tsv"), sep="\t", index=False)
    pd.DataFrame({"year": 2020, "slot": [1, 2], "sym": ["AAA", "CCC"], "order_rank": [1, 3]}).to_csv(os.path.join(d, "universe_2020.tsv"), sep="\t", index=False)
    pd.DataFrame({"date": DAYS, "close_time": "16:00"}).to_csv(os.path.join(d, "calendar.csv"), index=False)
    return d


def present(day, syms):
    return MASTER[day] & set(syms)


def factory():
    def fetch(sym, day):
        return empty() if (sym, day) == ("AAA", "20191230") else frames(hash((sym, day)) % 1000)
    return fetch


def test_tasks_year_boundary_absent_and_reappearance():
    d = setup()
    tasks, absent = download.stock_tasks("dev", [2019, 2020], present, universe_dir=d, calendar=os.path.join(d, "calendar.csv"))
    assert tasks == [("AAA", "20191230"), ("BBB", "20191230"), ("AAA", "20191231"), ("CCC", "20200102"),
                     ("AAA", "20200103"), ("CCC", "20200103")]                     # 2019 days use universe_2019, 2020 days universe_2020
    assert absent == [("BBB", "20191231"), ("AAA", "20200102")]                   # 2021-01-04 is outside dev
    t19, _ = download.stock_tasks("dev", [2019], present, universe_dir=d, calendar=os.path.join(d, "calendar.csv"))
    assert {x for _, x in t19} == {"20191230", "20191231"}


def test_download_files_absent_log_empty_day_and_resume():
    d = setup()
    out = os.path.join(d, "out")
    tasks, absent = download.stock_tasks("dev", [2019, 2020], present, universe_dir=d, calendar=os.path.join(d, "calendar.csv"))
    assert download.record_absent(out, absent, log=lambda m: None) == 2
    s = download.run(tasks, factory, base=out, free_fn=lambda p: 10**15, log=lambda m: None, suffix=download.STOCKS)
    assert (s["written"], s["failed"]) == (6, 0)                                  # absent days are not errors
    root = os.path.join(out, "bars_1s_dev_stocks")
    assert sorted(os.listdir(out)) == ["bars_1s_dev_stocks"]
    files = sorted(os.path.relpath(os.path.join(a, f), root) for a, _, fs in os.walk(root) for f in fs if f.endswith(".npz"))
    assert files == ["2019/AAA_20191230.npz", "2019/AAA_20191231.npz", "2019/BBB_20191230.npz",
                     "2020/AAA_20200103.npz", "2020/CCC_20200102.npz", "2020/CCC_20200103.npz"]   # no file for absent days
    ab = pd.read_csv(os.path.join(root, "skipped_absent.tsv"), sep="\t", dtype=str)
    assert list(ab.columns) == download.ABSENT_COLS and set(zip(ab.sym, ab.date)) == {("BBB", "20191231"), ("AAA", "20200102")}
    assert set(ab.reason) == {"not in mastm"}
    man = pd.read_csv(os.path.join(root, "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False)
    assert list(man.columns) == download.MAN_COLS and len(man) == 6
    e = man.set_index(["sym", "date"]).loc[("AAA", "20191230")]                    # in the master, no data -> zero bars
    assert (e.n_quote_rows, e.n_trade_rows, e.last_trade_sec, e.init_state) == ("0", "0", "-1", "False")
    z = bars.load(os.path.join(root, "2019", "AAA_20191230.npz"))
    assert all(int(np.abs(z[c]).sum()) == 0 for c in bars.COLS)
    assert not os.path.exists(os.path.join(root, "errors.tsv"))
    # resume: everything skipped by hash, the absent list is not duplicated
    s2 = download.run(tasks, factory, base=out, free_fn=lambda p: 10**15, log=lambda m: None, suffix=download.STOCKS)
    assert (s2["written"], s2["skipped"]) == (0, 6)
    assert download.record_absent(out, absent, log=lambda m: None) == 0
    assert len(pd.read_csv(os.path.join(root, "skipped_absent.tsv"), sep="\t")) == 2


def test_missing_universe_file_stops():
    d = setup()
    os.rename(os.path.join(d, "universe_2020.tsv"), os.path.join(d, "universe_2020.moved"))
    with pytest.raises(download.Stop, match="universe_2020.tsv"):
        download.stock_tasks("dev", [2019, 2020], present, universe_dir=d, calendar=os.path.join(d, "calendar.csv"))
