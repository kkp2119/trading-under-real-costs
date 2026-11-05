"""Draft A5 (Kyle 2026-10-27): stock mode drops days without a complete_nbbo, ctm or mastm table and records the reason;
a failed query rolls its connection back so later queries on it still work. A fake DB stands in for WRDS: a query on a
missing table raises and leaves the transaction aborted (every later query fails) until rollback(), as a SQLAlchemy
connection does when wrds raw_sql lets the error through. Outputs under $BARS1S_TEST_DIR (nothing deleted)."""
import os
import re
import sys
import uuid

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download  # noqa: E402
from test_download import frames  # noqa: E402

BASE = os.environ.get("BARS1S_TEST_DIR")
pytestmark = pytest.mark.skipif(not BASE, reason="set BARS1S_TEST_DIR (outside the repository)")
DAYS = ["20190516", "20190517", "20190520", "20190521", "20190522"]
MISSING = {"complete_nbbo_20190517", "mastm_20190520", "ctm_20190521", "mastm_20190521"}
GAPS = {"20190517": ["complete_nbbo"], "20190520": ["mastm"], "20190521": ["ctm", "mastm"]}


class UndefinedTable(Exception):
    pass


class InFailedSqlTransaction(Exception):
    pass


class FakeDB:
    def __init__(self, missing=MISSING, rollback_fails=False):
        self.tables = {f"{k}_{d}" for k in ("complete_nbbo", "ctm", "mastm") for d in DAYS} - set(missing)
        self.aborted, self.rollbacks, self.rollback_fails, self.queries = False, 0, rollback_fails, []
        self.connection = self

    def rollback(self):
        self.rollbacks += 1
        if self.rollback_fails:
            raise RuntimeError("synthetic rollback failure")
        self.aborted = False

    def close(self):
        pass

    def raw_sql(self, sql):
        self.queries.append(sql)
        if self.aborted:
            raise InFailedSqlTransaction("current transaction is aborted, commands ignored until end of transaction block")
        if "information_schema.tables" in sql:
            kinds = re.search(r"\^\(([a-z_|]+)\)", sql).group(1).split("|")
            names = sorted(t for t in self.tables if t.rsplit("_", 1)[0] in kinds)
            return pd.DataFrame({"table_schema": [f"taqm_{t[-8:-4]}" for t in names], "table_name": names})
        table = re.search(r"from taqm_\d{4}\.([a-z_]+_\d{8})", sql).group(1)
        if table not in self.tables:
            self.aborted = True
            raise UndefinedTable(f'relation "{table}" does not exist')
        if table.startswith("mastm_"):
            return pd.DataFrame({"symbol_15": ["AAA", "BBB"]})
        nb, pre, tr = frames(int(table[-8:]) % 1000)
        return {"complete_nbbo": pre if "order by" in sql else nb, "ctm": tr}[table.rsplit("_", 1)[0]]


def setup():
    d = os.path.join(BASE, uuid.uuid4().hex[:8])
    os.makedirs(d)
    pd.DataFrame({"year": 2019, "slot": [1, 2], "sym": ["AAA", "BBB"], "order_rank": [1, 2]}).to_csv(
        os.path.join(d, "universe_2019.tsv"), sep="\t", index=False)
    pd.DataFrame({"date": [f"{x[:4]}-{x[4:6]}-{x[6:]}" for x in DAYS], "close_time": "16:00"}).to_csv(
        os.path.join(d, "calendar.csv"), index=False)
    return d


def test_days_without_a_table_are_skipped_with_the_reason():
    d, db = setup(), FakeDB()
    days = download.split_days("dev", [2019], calendar=os.path.join(d, "calendar.csv"))
    gaps = download.table_gaps(db, days)
    assert gaps == GAPS
    assert sorted(set(days) - set(download.trading_days(db, "2019-05-16", "2019-05-22"))) == \
        sorted(x for x, m in gaps.items() if {"complete_nbbo", "ctm"} & set(m))              # trading_days' rule

    def present(day, syms):
        assert day not in GAPS, "a gap day must not be master-checked"
        return download.wrds_presence(db)(day, syms)
    tasks, absent = download.stock_tasks("dev", [2019], present, universe_dir=d, calendar=os.path.join(d, "calendar.csv"),
                                         gaps=gaps)
    assert tasks == [(s, x) for x in ("20190516", "20190522") for s in ("AAA", "BBB")]
    assert sorted(absent) == sorted((s, x, f"no {'/'.join(m)} table") for x, m in GAPS.items() for s in ("AAA", "BBB"))
    out = os.path.join(d, "out")
    assert download.record_absent(out, absent + [("AAA", "20190522")], log=lambda m: None) == 7
    ab = pd.read_csv(os.path.join(out, "bars_1s_dev_stocks", "skipped_absent.tsv"), sep="\t", dtype=str)
    assert list(ab.columns) == download.ABSENT_COLS
    assert dict(zip(zip(ab.sym, ab.date), ab.reason)) == {**{(s, x): f"no {'/'.join(m)} table" for x, m in GAPS.items()
                                                             for s in ("AAA", "BBB")}, ("AAA", "20190522"): "not in mastm"}
    assert download.record_absent(out, absent, log=lambda m: None) == 0                     # resume: no duplicates
    assert db.rollbacks == 0                                                               # no query failed


def test_failed_query_rolls_back_and_the_next_task_succeeds():
    out, dbs = os.path.join(setup(), "out"), []

    def factory():
        dbs.append(FakeDB())
        return download.make_fetch(dbs[-1])
    tasks = [("AAA", "20190516"), ("AAA", "20190517"), ("AAA", "20190520"), ("BBB", "20190516"), ("BBB", "20190522")]
    s = download.run(tasks, factory, base=out, n_conn=1, free_fn=lambda p: 10**15, log=lambda m: None,
                     suffix=download.STOCKS)
    assert (s["written"], s["failed"]) == (4, 1)                                           # only the missing table fails
    assert len(dbs) == 1 and dbs[0].rollbacks == 1 and not dbs[0].aborted
    e = pd.read_csv(os.path.join(out, "bars_1s_dev_stocks", "errors.tsv"), sep="\t", dtype=str)
    assert list(zip(e.sym, e.date)) == [("AAA", "20190517")] and e.kind[0].startswith("UndefinedTable")


def test_original_error_is_raised_even_when_rollback_fails():
    db = FakeDB(rollback_fails=True)
    with pytest.raises(UndefinedTable):
        download.make_fetch(db)("AAA", "20190517")
    assert db.rollbacks == 1
    with pytest.raises(UndefinedTable):
        download.query(FakeDB(), "select * from taqm_2019.ctm_20190521")
