"""null_backfill.py with a fake DB (no WRDS): per-date counts, zero fill, separate file, resume, stop on error."""
import os
import subprocess
import sys
import uuid

import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import download  # noqa: E402
import null_backfill  # noqa: E402

BASE = os.environ.get("BARS1S_TEST_DIR")
pytestmark = pytest.mark.skipif(not BASE, reason="set BARS1S_TEST_DIR (outside the repository)")


def root_with_manifest(pairs):
    r = os.path.join(BASE, uuid.uuid4().hex[:8], "bars_1s_dev")
    os.makedirs(r)
    m = pd.DataFrame([{c: "" for c in download.MAN_COLS} | {"sym": s, "date": d} for s, d in pairs])
    m.to_csv(os.path.join(r, "MANIFEST.tsv"), sep="\t", index=False)
    return r


class DB:
    def __init__(self, fail_day=None):
        self.sql, self.fail_day = [], fail_day

    def raw_sql(self, sql):
        self.sql.append(sql)
        if self.fail_day and self.fail_day in sql:
            raise RuntimeError("synthetic query error")
        if "row_number()" in sql:
            assert "rn <= 500" in sql and "time_m < '09:30:00'" in sql
            return pd.DataFrame({"sym_root": ["AAA"], "q_null_pre": [2]})
        if "q_null_session" in sql:
            assert "time_m >= '09:30:00' and time_m < '16:00:00'" in sql
            return pd.DataFrame({"sym_root": ["AAA", "BBB"], "q_null_session": [0, 1]})
        return pd.DataFrame({"sym_root": ["BBB"], "t_null_price": [0], "t_null_size": [3], "t_null_corr": [0], "t_null_scond": [7]})


def test_counts_zero_fill_separate_file_and_resume():
    r = root_with_manifest([("AAA", "20190102"), ("BBB", "20190102"), ("AAA", "20190103")])
    db = DB()
    s = null_backfill.backfill(db, r, log=lambda m: None)
    assert s == {"days": 2, "rows": 3} and len(db.sql) == 6                    # 3 queries per date
    assert not os.path.exists(os.path.join(r, "null_counts.tsv"))              # never the downloader's file
    t = pd.read_csv(os.path.join(r, "null_counts_backfill.tsv"), sep="\t", dtype={"date": str})
    assert list(t.columns) == download.NULL_COLS and set(t.source) == {"server_count"}
    g = t.set_index(["sym", "date"])[null_backfill.CNT]
    assert g.loc[("AAA", "20190102")].tolist() == [2, 0, 0, 0, 0, 0]
    assert g.loc[("BBB", "20190102")].tolist() == [0, 1, 0, 3, 0, 7]
    db2 = DB()
    assert null_backfill.backfill(db2, r, log=lambda m: None) == {"days": 0, "rows": 0} and db2.sql == []


def test_error_stops_and_the_failed_date_is_not_written():
    r = root_with_manifest([("AAA", "20190102"), ("AAA", "20190103")])
    with pytest.raises(RuntimeError, match="synthetic query error"):
        null_backfill.backfill(DB(fail_day="20190103"), r, log=lambda m: None)
    t = pd.read_csv(os.path.join(r, "null_counts_backfill.tsv"), sep="\t", dtype={"date": str})
    assert t.date.tolist() == ["20190102"]


def test_refuses_without_go():
    p = subprocess.run([sys.executable, os.path.join(HERE, "null_backfill.py"), "--split", "dev"], capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert p.returncode != 0 and "--go" in p.stderr
