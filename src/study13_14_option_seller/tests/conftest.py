"""Research 13 data-layer tests (S0, memo 4a): synthetic data in an in-memory SQLite stand-in for WRDS; no WRDS
connection and no real data file. Files go under $R13_TEST_DIR (outside the repository, nothing deleted).
Run with: R13_TEST_DIR=<scratch> python -m pytest -p no:cacheprovider research13/tests"""
import os
import sqlite3
import sys
import uuid

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT]

import omdata as om  # noqa: E402

UNDS = ("SPX", "NDX", "RUT")
OPP_COLS = ["secid", "date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer", "impl_volatility", "delta",
            "forward_price", "open_interest", "optionid", "am_settlement", "expiry_indicator", "root", "contract_size",
            "volume", "gamma"]
VS_COLS = ["secid", "date", "days", "delta", "cp_flag", "impl_volatility", "impl_strike", "impl_premium"]
SEC_COLS = ["secid", "date", "close", "return", "open"]
ZERO_COLS = ["date", "days", "rate"]


class _Conn:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


class FakeDB:
    """raw_sql over SQLite with schemas optionm and information_schema; .sql records every statement."""

    def __init__(self):
        self._con = sqlite3.connect(":memory:")
        for s in ("optionm", "information_schema"):
            self._con.execute(f"ATTACH DATABASE ':memory:' AS {s}")
        self._con.execute("CREATE TABLE information_schema.tables (table_schema TEXT, table_name TEXT)")
        self._con.execute("CREATE TABLE information_schema.columns (table_schema TEXT, table_name TEXT, column_name TEXT, "
                          "data_type TEXT, ordinal_position INT)")
        self.connection = _Conn()
        self.sql, self.fail_on, self.inject, self.closed = [], None, None, False

    def add(self, name, cols, rows):
        q = ", ".join(f'"{c}"' for c in cols)
        self._con.execute(f"CREATE TABLE optionm.{name} ({q})")
        self._con.executemany(f"INSERT INTO optionm.{name} VALUES ({','.join('?' * len(cols))})", rows)
        self._con.execute("INSERT INTO information_schema.tables VALUES ('optionm', ?)", (name,))
        self._con.executemany("INSERT INTO information_schema.columns VALUES ('optionm', ?, ?, ?, ?)",
                              [(name, c, "double precision", i + 1) for i, c in enumerate(cols)])

    def raw_sql(self, sql):
        self.sql.append(sql)
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError("synthetic query failure")
        df = pd.read_sql_query(sql, self._con)
        return self.inject(sql, df) if self.inject else df

    def close(self):
        self.closed = True

    def __del__(self):
        self._con.close()


def default_dates(year):
    return [f"{year}-01-05", f"{year}-03-10", f"{year}-12-28"] if year < 2025 else ["2025-01-06", "2025-03-10", "2025-08-28"]


def build_world(db, years, unds=UNDS, dates=None, close_null=(), zero_years=range(1996, 2026)):
    """Synthetic OptionMetrics tables. dates: {(table, und, year): [YYYY-MM-DD]} overrides; close_null: {(und, date)}.
    March opprcd rows have forward_price NULL throughout (an all-NULL column in that month's result)."""
    dates = dates or {}
    oid = 1000
    for y in years:
        opp, vs, sec = [], [], []
        for u in unds:
            sid = om.UND[u]
            for d in dates.get(("opprcd", u, y), default_dates(y)):
                for k, cp in enumerate("CP"):
                    oid += 1
                    opp.append((sid, d, f"{y}-12-31", cp, 1000000.0 + k, 10.0, 10.5, None if k else 0.2,
                                0.5 if cp == "C" else -0.5, None if d[5:7] == "03" else 1001.0, 100.0, oid,
                                None if k else 1, "w" if k else None, "SPX", 100.0, 7, 0.01))
            for d in dates.get(("vsurfd", u, y), default_dates(y)):
                vs += [(sid, d, 30, 50, "C", 0.2, 1000.0, 50.0), (sid, d, 91, -50, "P", 0.21, 1000.0, 60.0)]
            for d in dates.get(("secprd", u, y), default_dates(y)):
                sec.append((sid, d, None if (u, d) in close_null else 1000.0, 0.001, 999.0))
        db.add(f"opprcd{y}", OPP_COLS, opp)
        db.add(f"vsurfd{y}", VS_COLS, vs)
        db.add(f"secprd{y}", SEC_COLS, sec)
    db.add("zerocd", ZERO_COLS, [(f"{y}-0{m}-05", n, 1.5) for y in zero_years for m in (1, 6) for n in (30, 365)])
    return db


@pytest.fixture
def tmp_r13():
    base = os.environ.get("R13_TEST_DIR")
    if not base:
        pytest.skip("set R13_TEST_DIR (outside the repository)")
    d = os.path.join(base, uuid.uuid4().hex[:10])
    os.makedirs(d)
    return d
