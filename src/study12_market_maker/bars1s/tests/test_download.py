"""Downloader tests with a fake fetcher (no WRDS): split folders, atomic files, resume by hash, stop rules, connections,
deadline and the parallel-effect rule. Outputs go to $BARS1S_TEST_DIR/<unique>/ (outside the repository); nothing is deleted."""
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import download  # noqa: E402

BASE = os.environ.get("BARS1S_TEST_DIR")
pytestmark = pytest.mark.skipif(not BASE, reason="set BARS1S_TEST_DIR (outside the repository)")
FREE = {"free_fn": lambda p: 10**15, "log": lambda m: None}


def outdir():
    p = os.path.join(BASE, uuid.uuid4().hex[:8])
    os.makedirs(p)
    return p


def frames(seed):
    """Synthetic complete_nbbo session / pre-open / ctm frames with datetime.time values (as raw_sql returns).
    Trades include 3 with code 'T' and 2 with tr_corr '01' (excluded, counted by code)."""
    r = np.random.default_rng(seed)

    def tm(sec):
        us = (np.asarray(sec) * 1e6).astype(np.int64)
        return [(dt.datetime(2000, 1, 1) + dt.timedelta(microseconds=int(u))).time() for u in us]
    qs = np.sort(r.uniform(34200, 57600, 2000))
    mid = 100 + np.cumsum(r.choice([-0.01, 0, 0.01], 2000))
    nb = pd.DataFrame({"time_m": tm(qs), "time_m_nano": r.integers(0, 1000, 2000), "best_bid": mid - 0.01,
                       "best_bidsizeshares": r.integers(1, 500, 2000), "best_ask": mid + 0.01,
                       "best_asksizeshares": r.integers(1, 500, 2000), "qu_cond": "R", "qu_source": "C", "natbbo_ind": "G"})
    pre = nb.iloc[:1].assign(time_m=tm([34000.0]))
    ts = np.sort(r.uniform(34200, 57600, 500))
    sc = np.array(["@"] * 500, dtype=object)
    sc[:3] = "@ T"
    corr = np.array(["00"] * 500, dtype=object)
    corr[3:5] = "01"
    tr = pd.DataFrame({"time_m": tm(ts), "time_m_nano": 0, "price": 100.0, "size": r.integers(1, 300, 500),
                       "tr_scond": sc, "tr_corr": corr})
    return nb, pre, tr


class Fake:
    def __init__(self, fail=None, warn=None):
        """fail: (sym, day) -> RuntimeError('synthetic connection error'), or a dict {(sym, day): exception}."""
        self.calls, self.opened, self.warn = [], 0, warn
        self.fail = fail if isinstance(fail, dict) else ({fail: RuntimeError("synthetic connection error")} if fail else {})
        self.lock = threading.Lock()

    def factory(self):
        with self.lock:
            self.opened += 1

        def fetch(sym, day):
            with self.lock:
                self.calls.append((sym, day))
            if (sym, day) in self.fail:
                raise self.fail[(sym, day)]
            if (sym, day) == self.warn:
                warnings.warn("synthetic warning")
            return frames(hash((sym, day)) % 1000)
        return fetch


TASKS = [("AAA", "20190516"), ("BBB", "20190516"), ("AAA", "20200102")]


def test_write_dev_folders_manifest_and_resume():
    out, f = outdir(), Fake()
    s = download.run(TASKS, f.factory, base=out, **FREE)
    assert (s["written"], s["skipped"]) == (3, 0) and f.opened == 3
    root = os.path.join(out, "bars_1s_dev")
    for sym, day in TASKS:
        assert os.path.exists(os.path.join(root, day[:4], f"{sym}_{day}.npz"))
    assert sorted(os.listdir(out)) == ["bars_1s_dev"]
    assert not [x for _, _, fs in os.walk(out) for x in fs if x.endswith(".part")]
    man = pd.read_csv(os.path.join(root, "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False)
    assert len(man) == 3 and list(man.columns) == download.MAN_COLS
    for r in man.itertuples():
        assert download.sha256(os.path.join(root, r.file)) == r.sha256 and r.rows == "23400"
        assert json.loads(r.excl_by_code) == {"corr=01": 2, "scond=@ T": 3} and r.n_trade_excluded == "5"
    f2 = Fake()
    s2 = download.run(TASKS, f2.factory, base=out, **FREE)
    assert (s2["written"], s2["skipped"], f2.calls) == (0, 3, [])


def test_split_routing_by_date_only():
    out = outdir()
    tasks = [("AAA", "20201231"), ("AAA", "20210104"), ("AAA", "20260930"), ("AAA", "20261001"), ("AAA", "20261023")]
    download.run(tasks, Fake().factory, base=out, **FREE)
    got = sorted(os.path.relpath(os.path.join(d, x), out) for d, _, fs in os.walk(out) for x in fs if x.endswith(".npz"))
    assert got == ["bars_1s_dev/2020/AAA_20201231.npz", "bars_1s_holdout/2021/AAA_20210104.npz",
                   "bars_1s_holdout/2026/AAA_20260930.npz", "bars_1s_reserve/2026/AAA_20261001.npz",
                   "bars_1s_reserve/2026/AAA_20261023.npz"]
    for bad in ("20181231", "20190101", "20210101", "20261024"):
        with pytest.raises(download.Stop, match="outside every split"):
            download.split_of(bad)


def test_existing_file_without_matching_hash_stops_and_is_kept():
    out = outdir()
    download.run(TASKS[:1], Fake().factory, base=out, **FREE)
    root = os.path.join(out, "bars_1s_dev")
    p = os.path.join(root, "2019", "AAA_20190516.npz")
    before = download.sha256(p)
    shutil.copy(os.path.join(root, "MANIFEST.tsv"), os.path.join(root, "MANIFEST.bak"))
    with open(os.path.join(root, "MANIFEST.tsv"), "w") as fh:
        fh.write("\t".join(download.MAN_COLS) + "\n")                # manifest without the line
    with pytest.raises(download.Stop, match="exists without a matching MANIFEST hash"):
        download.run(TASKS[:1], Fake().factory, base=out, n_conn=1, **FREE)
    assert download.sha256(p) == before


TASKS36 = [(s, f"2019{m:02d}15") for m in range(1, 13) for s in ("AAA", "BBB", "CCC")]


def errors(out):
    p = os.path.join(out, "bars_1s_dev", "errors.tsv")
    return pd.read_csv(p, sep="\t", dtype=str) if os.path.exists(p) else pd.DataFrame(columns=download.ERR_COLS)


def test_failure_is_logged_not_retried_and_the_run_continues():
    out, f = outdir(), Fake(fail=("BBB", "20190215"))
    s = download.run(TASKS36, f.factory, base=out, **FREE)
    assert (s["written"], s["failed"], s["remaining"]) == (35, 1, 0)
    assert f.calls.count(("BBB", "20190215")) == 1                    # no retry
    e = errors(out)
    assert list(e.columns) == download.ERR_COLS and len(e) == 1 and (e.sym[0], e.date[0]) == ("BBB", "20190215")
    assert e.kind[0] == "RuntimeError: synthetic connection error"
    assert not os.path.exists(os.path.join(out, "bars_1s_dev", "2019", "BBB_20190215.npz"))


def test_warning_is_a_logged_failure():
    out, f = outdir(), Fake(warn=("AAA", "20190516"))
    s = download.run(TASKS, f.factory, base=out, n_conn=1, **FREE)
    assert (s["written"], s["failed"]) == (2, 1) and errors(out).kind[0].startswith("UserWarning: synthetic warning")


def test_more_than_20_failures_stops():
    fails = {t: RuntimeError(f"timeout after {i} s") for i, t in enumerate(TASKS36[:25])}   # one kind after digit masking
    out, f = outdir(), Fake(fail=fails)
    with pytest.raises(download.Stop, match=r"failures 21 \(limit 20\)"):
        download.run(TASKS36, f.factory, base=out, n_conn=1, **FREE)
    assert len(errors(out)) == 21 and len(f.calls) == 21              # stopped right after the 21st failure


def test_more_than_3_error_kinds_stops():
    kinds = [RuntimeError("a"), ValueError("b"), KeyError("c"), OSError("d")]
    fails = {t: kinds[i] for i, t in enumerate(TASKS36[:4])}
    out, f = outdir(), Fake(fail=fails)
    with pytest.raises(download.Stop, match="error kinds 4"):
        download.run(TASKS36, f.factory, base=out, n_conn=1, **FREE)
    assert len(errors(out)) == 4 and len(f.calls) == 4


def test_disk_floor_stops_before_fetching():
    f = Fake()
    with pytest.raises(download.Stop, match="free disk space below 15 GiB"):
        download.run(TASKS, f.factory, base=outdir(), free_fn=lambda p: 14 * 1024**3, log=lambda m: None)
    assert f.calls == []


def test_deadline_takes_no_new_task():
    f = Fake()
    s = download.run(TASKS, f.factory, base=outdir(), deadline=time.time() - 1, **FREE)
    assert (s["written"], s["remaining"], f.calls) == (0, 3, [])


def test_phased_calibration_and_connection_rule(monkeypatch):
    out, f = outdir(), Fake()
    tasks = [(s, f"2019{m:02d}15") for m in range(1, 5) for s in ("AAA", "BBB")]
    r = download.phased(tasks, f.factory, log=lambda m: None, base=out, calib=2, phase_sec=3600, free_fn=lambda p: 10**15)
    assert r["phase1"]["written"] == 2 and r["phase2"]["written"] == 6 and r["phase3"]["written"] == 0
    assert f.opened == 1 + 3 + r["n_final"]
    calls = []

    def fake_run(tasks, factory, n_conn=3, deadline=None, **kw):
        calls.append(n_conn)
        wall = {1: 10.0, 3: 8.0}.get(n_conn, 1.0)                         # 3 connections only 1.25x faster
        return {"written": 10, "skipped": 0, "remaining": 0, "wall": wall, "fetch_secs": [1.0]}
    monkeypatch.setattr(download, "run", fake_run)
    r = download.phased(tasks, None, log=lambda m: None, base=out)
    assert calls == [1, 3, 2] and r["n_final"] == 2 and abs(r["speedup"] - 1.25) < 1e-12


def test_trading_days_from_table_names_only():
    class DB:
        def raw_sql(self, sql):
            assert "information_schema.tables" in sql and "'taqm_2019'" in sql and "'taqm_2020'" in sql
            return pd.DataFrame({"table_schema": ["taqm_2019"] * 5 + ["taqm_2020"] * 2,
                                 "table_name": ["complete_nbbo_20190516", "ctm_20190516", "ctm_20190517",
                                                "complete_nbbo_20191231", "ctm_20191231", "complete_nbbo_20200102", "ctm_20200102"]})
    assert download.trading_days(DB(), "2019-05-01", "2020-01-01") == ["20190516", "20191231"]


def test_to_ns_time_objects_and_strings():
    t = [dt.time(9, 30, 0, 123456), "09:30:01.5"]
    assert list(download.to_ns(t, [7, 0])) == [34200 * 10**9 + 123456000 + 7, 34201 * 10**9 + 500000000]


def test_refuses_without_go():
    r = subprocess.run([sys.executable, os.path.join(HERE, "download.py"), "--split", "dev", "--syms", "AAA"],
                       capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode != 0 and "--go" in r.stderr


def null_frames(seed):
    """frames() with nullable dtypes and NULLs: 1 pre-open quote, 2 session quotes, trade price 1, size 1, tr_corr 1, tr_scond 4."""
    nb, pre, tr = frames(seed)
    nb = nb.astype({"best_bid": "Float64", "best_ask": "Float64", "best_bidsizeshares": "Int32", "best_asksizeshares": "Int32",
                    "time_m_nano": "Int16"})
    pre = pre.astype({"best_bid": "Float64", "best_ask": "Float64", "best_bidsizeshares": "Int32", "best_asksizeshares": "Int32",
                      "time_m_nano": "Int16"})
    tr = tr.astype({"price": "Float64", "size": "Int32", "tr_scond": "string", "tr_corr": "string", "time_m_nano": "Int16"})
    pre.loc[0, "best_ask"] = pd.NA
    nb.loc[10, "best_bid"] = pd.NA
    nb.loc[20, "best_asksizeshares"] = pd.NA
    tr.loc[100, "price"] = pd.NA
    tr.loc[101, "size"] = pd.NA
    tr.loc[102, "tr_corr"] = pd.NA
    tr.loc[[103, 104, 105, 106], "tr_scond"] = pd.NA
    return nb, pre, tr


def test_null_counts_file_one_line_per_written_day_and_manifest_unchanged():
    out = outdir()

    def factory():
        def fetch(sym, day):
            return null_frames(1) if sym == "BBB" else frames(2)
        return fetch
    download.run(TASKS, factory, base=out, **FREE)
    root = os.path.join(out, "bars_1s_dev")
    n = pd.read_csv(os.path.join(root, "null_counts.tsv"), sep="\t", dtype={"date": str})
    assert list(n.columns) == download.NULL_COLS and len(n) == 3 and set(n.source) == {"download"}
    got = n.set_index(["sym", "date"])
    assert got.loc[("BBB", "20190516"), ["q_null_pre", "q_null_session", "t_null_price", "t_null_size", "t_null_corr",
                                          "t_null_scond"]].tolist() == [1, 2, 1, 1, 1, 4]
    assert (got.loc[("AAA", "20190516"), ["q_null_pre", "q_null_session", "t_null_price", "t_null_size", "t_null_corr",
                                           "t_null_scond"]] == 0).all()
    man = pd.read_csv(os.path.join(root, "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False)
    assert list(man.columns) == download.MAN_COLS and len(man) == 3
    b = man.set_index("sym").loc["BBB"]
    assert json.loads(b.excl_by_code)["NULL"] == 2 and json.loads(b.excl_by_code)["corr=NULL"] == 1
