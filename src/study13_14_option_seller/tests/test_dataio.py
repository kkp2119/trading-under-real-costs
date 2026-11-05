"""S0 offline tests of the research 13 data layer (memo 4a sections 1-5): storage format, manifest, date split,
holdout lock, resume path, failure policy, calendar and holdout end date, S1 statement whitelist."""
import datetime as dt
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import download as dl
import omdata as om
from conftest import FakeDB, build_world

DEV, HOLD = range(1996, 2013), range(2013, 2026)
BIG = lambda p: 100 * 1024**3                                             # noqa: E731


def lines(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


# ---------- split guard ----------

def test_split_guard_names_and_sql():
    with pytest.raises(om.SplitError):
        om.table_name("opprcd", "dev", 2013)
    with pytest.raises(om.SplitError):
        om.table_name("opprcd", "holdout", 2012)
    with pytest.raises(om.SplitError):
        om.table_name("opprcd", "reserve", 2013)
    for q in om.unit_sql("opprcd", "dev", "SPX", 2012):
        assert "date <= '2012-12-31'" in q and "secid = 108105" in q and "opprcd2012" in q
    assert len(om.unit_sql("opprcd", "dev", "SPX", 2000)) == 12 and len(om.unit_sql("secprd", "dev", "SPX", 2000)) == 1
    (z,) = om.unit_sql("zerocd", "holdout")
    assert "date >= '2013-01-01'" in z
    for st in ("S2", "S3"):
        split, units, _ = dl.stage_units(st)
        assert split == "dev" and all(y is None or y <= 2012 for _, _, y in units)
    for st in ("S4a", "S4b"):
        split, units, _ = dl.stage_units(st)
        assert split == "holdout" and all(y is None or y >= 2013 for _, _, y in units)
    _, u4b, _ = dl.stage_units("S4b")
    assert {s for _, s, _ in u4b} == {"NDX", "RUT"} and {t for t, _, _ in u4b} == {"vsurfd", "opprcd"}
    _, u4a, _ = dl.stage_units("S4a")
    assert {(t, s) for t, s, _ in u4a if t != "zerocd"} == {("secprd", "SPX"), ("secprd", "NDX"), ("secprd", "RUT"),
                                                              ("vsurfd", "SPX"), ("opprcd", "SPX")}
    assert [t for t, _, _ in u4a].index("opprcd") > [t for t, _, _ in u4a].index("vsurfd") > 0


def test_select_lists_only_memo_columns():
    for t in om.COLUMNS:
        sel = om.unit_sql(t, "dev", "SPX", 2000)[0] if t != "zerocd" else om.unit_sql(t, "dev")[0]
        cols = sel.split("SELECT ", 1)[1].split(" FROM", 1)[0].replace('"', "").split(", ")
        assert cols == [c for c, _ in om.COLUMNS[t]]
        for banned in ("gamma", "vega", "theta", "volume", "ss_flag", "impl_premium"):
            assert banned not in cols


# ---------- typed arrays ----------

def test_to_arrays_types_nulls_empty_and_all_null_frames():
    f1 = pd.DataFrame({"date": [dt.date(1996, 1, 4), dt.date(1996, 1, 4)], "exdate": ["1996-01-20", "1996-01-20"],
                       "cp_flag": ["C", "P"], "strike_price": [600000.0, 600000.0], "best_bid": [1.0, 2.0],
                       "best_offer": [1.5, 2.5], "impl_volatility": [0.2, None], "delta": [0.5, -0.5],
                       "forward_price": [None, None], "open_interest": [10.0, None], "optionid": [11.0, 12.0],
                       "am_settlement": [1.0, None], "expiry_indicator": [None, "w"], "root": ["SPX", "SPB"],
                       "contract_size": [100.0, 100.0]})
    f0 = f1.iloc[:0].astype(object)                                     # a month without rows
    f2 = f1.copy()
    f2["forward_price"] = [600.5, 601.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a, n = om.to_arrays("opprcd", [f0, f1, f2])
    assert a["date"].dtype == np.int32 and a["date"][0] == 19960104 and a["exdate"][0] == 19960120
    assert a["cp_flag"].tolist() == [1, -1, 1, -1] and a["cp_flag"].dtype == np.int8
    assert a["optionid"].dtype == np.int64 and a["am_settlement"].tolist() == [1, -1, 1, -1]
    assert np.isnan(a["impl_volatility"][1]) and np.isnan(a["forward_price"][:2]).all() and a["forward_price"][2] == 600.5
    assert a["expiry_indicator__codes"].tolist() == ["w"] and a["expiry_indicator"].tolist() == [-1, 0, -1, 0]
    assert a["root__codes"].tolist() == ["SPB", "SPX"] and a["root"].dtype == np.int16
    assert n["forward_price"] == 2 and n["impl_volatility"] == 2 and n["expiry_indicator"] == 2
    e, _ = om.to_arrays("opprcd", [f0])
    assert len(e["date"]) == 0 and e["expiry_indicator__codes"].shape == (0,)


@pytest.mark.parametrize("col,val", [("cp_flag", "X"), ("date", None), ("optionid", 1.5), ("strike_price", "abc")])
def test_to_arrays_rejects_values_outside_kind(col, val):
    f = pd.DataFrame({"date": ["1996-01-04"], "exdate": ["1996-01-20"], "cp_flag": ["C"], "strike_price": [1.0],
                      "best_bid": [1.0], "best_offer": [1.0], "impl_volatility": [0.2], "delta": [0.5],
                      "forward_price": [1.0], "open_interest": [1.0], "optionid": [1.0], "am_settlement": [1.0],
                      "expiry_indicator": [None], "root": ["SPX"], "contract_size": [100.0]}).astype(object)
    f.loc[0, col] = val
    with pytest.raises(om.DataError):
        om.to_arrays("opprcd", [f])


def test_check_dates_bounds():
    om.check_dates({"date": np.array([19960104, 19961231], np.int32)}, "dev", 1996)
    for bad, split, y in [([19970102], "dev", 1996), ([20130102], "dev", None), ([20121231], "holdout", None)]:
        with pytest.raises(om.DataError):
            om.check_dates({"date": np.array(bad, np.int32)}, split, y)


def test_content_hash_is_stable_across_rewrites(tmp_r13):
    a = {"date": np.array([19960104], np.int32), "x": np.array([1.5]), "c__codes": np.array(["w"])}
    p1, h1, _ = om.write_npz_part(os.path.join(tmp_r13, "a.npz"), a)
    p2, h2, _ = om.write_npz_part(os.path.join(tmp_r13, "b.npz"), {k: v.copy() for k, v in a.items()})
    with np.load(p1, allow_pickle=False) as z1, np.load(p2, allow_pickle=False) as z2:
        assert om.content_sha256({k: z1[k] for k in z1.files}) == om.content_sha256({k: z2[k] for k in z2.files}) \
            == om.content_sha256(a)


# ---------- S2: files, manifest, resume ----------

def test_s2_writes_typed_files_manifest_and_code_version(tmp_r13):
    db = build_world(FakeDB(), [1996])
    r = dl.run("S2", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)
    assert r == {"written": 4, "skipped": 0}
    root = os.path.join(tmp_r13, "data_dev")
    m = lines(os.path.join(root, "MANIFEST.tsv"))
    assert list(m.columns) == om.MAN_COLS and sorted(m.file) == sorted(
        ["secprd/secprd_SPX_1996.npz", "vsurfd/vsurfd_SPX_1996.npz", "opprcd/opprcd_SPX_1996.npz", "zerocd/zerocd_dev.npz"])
    st = om.Store("dev", base=tmp_r13)
    o = st.load("opprcd/opprcd_SPX_1996.npz")
    assert len(o["date"]) == 6 and set(np.unique(o["date"])) == {19960105, 19960310, 19961228}
    assert np.isnan(o["forward_price"][o["date"] == 19960310]).all()          # all-NULL March month
    assert set(o) == {c for c, _ in om.COLUMNS["opprcd"]} | {"expiry_indicator__codes", "root__codes"}
    z = st.load("zerocd/zerocd_dev.npz")
    assert z["date"].max() <= 20121231 and z["date"].min() >= 19960101 and len(z["date"]) == 17 * 4
    s = st.load("secprd/secprd_SPX_1996.npz")
    assert s["has_close"].all() and "open" not in s
    row = m.set_index("file").loc["opprcd/opprcd_SPX_1996.npz"]
    assert row.content_sha256 == om.content_sha256(o) and int(row.n_queries) == 12 and row.secid == "108105"
    cv = lines(os.path.join(root, "CODE_VERSION.tsv"))
    assert cv.memo_sha256.iloc[0] == om.MEMO_SHA and cv.stage.iloc[0] == "S2"
    nc = lines(os.path.join(root, "null_counts.tsv"))
    assert int(nc[(nc.table == "opprcd") & (nc.column == "forward_price")].n_null.iloc[0]) == 2
    assert not any(f.endswith(".part") for _, _, fs in os.walk(root) for f in fs)
    assert not any("gamma" in q or "volume" in q for q in db.sql)


def test_resume_skips_and_never_overwrites(tmp_r13):
    db = build_world(FakeDB(), [1996])
    dl.run("S2", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)
    n = len(db.sql)
    assert dl.run("S2", db, base=tmp_r13, log=lambda m: None, free_fn=BIG) == {"written": 0, "skipped": 4}
    assert all("information_schema" in q for q in db.sql[n:])               # nothing re-fetched
    p = os.path.join(tmp_r13, "data_dev", "opprcd", "opprcd_SPX_1996.npz")
    with open(p, "ab") as f:
        f.write(b"x")
    h = om.sha256(p)
    with pytest.raises(dl.Stop, match="without a matching MANIFEST"):
        dl.run("S2", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)
    assert om.sha256(p) == h


def test_leftover_part_is_rewritten(tmp_r13):
    db = build_world(FakeDB(), [1996])
    p = os.path.join(tmp_r13, "data_dev", "opprcd", "opprcd_SPX_1996.npz")
    os.makedirs(os.path.dirname(p))
    with open(p + ".part", "wb") as f:
        f.write(b"interrupted")
    dl.run("S2", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)
    assert not os.path.exists(p + ".part") and len(om.Store("dev", base=tmp_r13).load("opprcd/opprcd_SPX_1996.npz")["date"]) == 6


# ---------- failure policy ----------

def test_query_error_rolls_back_and_stops_at_first_failure(tmp_r13):
    db = build_world(FakeDB(), [1996])
    db.fail_on = "FROM optionm.opprcd1996"
    with pytest.raises(dl.Stop, match="RuntimeError"):
        dl.run("S2", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)
    root = os.path.join(tmp_r13, "data_dev")
    assert db.connection.rollbacks == 1
    assert len(lines(os.path.join(root, "errors.tsv"))) == 1
    assert sorted(lines(os.path.join(root, "MANIFEST.tsv")).file) == ["secprd/secprd_SPX_1996.npz", "vsurfd/vsurfd_SPX_1996.npz"]
    assert not os.path.exists(os.path.join(root, "zerocd", "zerocd_dev.npz"))


def test_table_check_error_is_a_stop(tmp_r13):
    db = build_world(FakeDB(), [1996])
    db.fail_on = "information_schema.tables"
    with pytest.raises(dl.Stop, match="table check"):
        dl.run("S2", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)
    assert db.connection.rollbacks == 1 and len(lines(os.path.join(tmp_r13, "data_dev", "errors.tsv"))) == 1


def test_out_of_range_row_stops_without_writing(tmp_r13):
    db = build_world(FakeDB(), [1996])

    def inject(sql, df):
        if "FROM optionm.opprcd1996" in sql and "BETWEEN '1996-01-01'" in sql:
            assert len(df)
            df = pd.concat([df, df.iloc[:1].assign(date="1997-01-02")], ignore_index=True)
        return df
    db.inject = inject
    with pytest.raises(dl.Stop, match="DataError"):
        dl.run("S2", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)
    root = os.path.join(tmp_r13, "data_dev")
    assert not os.path.exists(os.path.join(root, "opprcd", "opprcd_SPX_1996.npz"))
    assert "opprcd/opprcd_SPX_1996.npz" not in set(lines(os.path.join(root, "MANIFEST.tsv")).file)


def test_warning_is_a_failure(tmp_r13, monkeypatch):
    db = build_world(FakeDB(), [1996])
    real = om.to_arrays

    def warn(table, frames):
        warnings.warn("synthetic", FutureWarning)
        return real(table, frames)
    monkeypatch.setattr(om, "to_arrays", warn)
    with pytest.raises(dl.Stop, match="FutureWarning"):
        dl.run("S2", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)


def test_disk_floor_and_missing_tables_stop_before_writing(tmp_r13):
    db = build_world(FakeDB(), [1996])
    with pytest.raises(dl.Stop, match="free disk"):
        dl.run("S2", db, base=tmp_r13, log=lambda m: None, free_fn=lambda p: 14 * 1024**3)
    assert not os.path.exists(os.path.join(tmp_r13, "data_dev", "MANIFEST.tsv"))
    db2 = FakeDB()
    build_world(db2, [1996])
    db2._con.execute("DELETE FROM information_schema.tables WHERE table_name = 'vsurfd1996'")
    other = os.path.join(tmp_r13, "b")
    with pytest.raises(dl.Stop, match="missing tables: vsurfd1996"):
        dl.run("S2", db2, base=other, log=lambda m: None, free_fn=BIG)
    assert not os.path.exists(os.path.join(other, "data_dev", "CODE_VERSION.tsv"))


def test_cli_refuses_without_go(monkeypatch):
    monkeypatch.setattr(dl, "connect", lambda: (_ for _ in ()).throw(AssertionError("connected")))
    for argv in (["--stage", "S2"], ["--meta"], ["--stage", "S2", "--meta", "--go"], ["--go"]):
        with pytest.raises(SystemExit):
            dl.main(argv)


# ---------- lock ----------

def test_holdout_store_locked_until_unlock_file(tmp_r13, monkeypatch):
    res = os.path.join(tmp_r13, "results")
    os.makedirs(res)
    touched = []
    real_exists = os.path.exists
    monkeypatch.setattr(om.os.path, "exists", lambda p: (touched.append(p), real_exists(p))[1])
    with pytest.raises(om.Refused, match="locked"):
        om.Store("holdout", base=tmp_r13, results_dir=res)
    assert all("data_holdout" not in p for p in touched)
    monkeypatch.undo()
    db = build_world(FakeDB(), HOLD)
    dl.run("S4a", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)
    open(os.path.join(res, om.UNLOCK), "w").close()
    st = om.Store("holdout", base=tmp_r13, results_dir=res)
    assert len(st.load("secprd/secprd_SPX_2013.npz")["date"]) == 3
    p = os.path.join(tmp_r13, "data_holdout", "secprd", "secprd_NDX_2013.npz")
    with open(p, "ab") as f:
        f.write(b"x")
    with pytest.raises(om.Refused, match="sha256"):
        st.load("secprd/secprd_NDX_2013.npz")
    with pytest.raises(om.Refused, match="not in MANIFEST"):
        st.load("opprcd/opprcd_NDX_2013.npz")


# ---------- calendar and holdout end ----------

def test_calendar_dev_union_bits(tmp_r13):
    dates = {("secprd", "SPX", 2000): ["2000-01-05", "2000-03-10"], ("secprd", "NDX", 2000): ["2000-03-10", "2000-04-03"],
             ("secprd", "RUT", 2000): ["2000-03-10", "2000-05-01", "2000-06-01"]}
    db = build_world(FakeDB(), DEV, dates=dates, close_null={("RUT", "2000-06-01")})
    dl.run("S3", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)
    root = os.path.join(tmp_r13, "data_dev")
    cal = pd.read_csv(os.path.join(root, "calendar_dev.csv"))
    c = dict(zip(cal.date, cal.has))
    assert c[20000105] == 1 and c[20000310] == 7 and c[20000403] == 2 and c[20000501] == 4 and 20000601 not in c
    assert c[19960105] == 7 and cal.date.min() >= 19960101 and cal.date.max() <= 20121231
    sp = json.load(open(os.path.join(root, "SPLIT.json")))
    assert sp["calendar_sha256"] == om.sha256(os.path.join(root, "calendar_dev.csv")) and "holdout_end_date" not in sp
    assert sp["calendar_days"] == len(cal) and sp["memo_sha256"] == om.MEMO_SHA
    assert len(lines(os.path.join(root, "MANIFEST.tsv"))) == 3 * 3 * 17 + 1


def test_holdout_end_date_and_truncated_calendar_idempotent(tmp_r13):
    dates = {("secprd", "SPX", 2025): ["2025-01-06", "2025-08-28", "2025-08-29"],
             ("vsurfd", "SPX", 2025): ["2025-01-06", "2025-08-28"],
             ("opprcd", "SPX", 2025): ["2025-01-06", "2025-08-28", "2025-08-29"],
             ("secprd", "NDX", 2025): ["2025-01-06", "2025-08-28", "2025-09-05"]}
    db = build_world(FakeDB(), HOLD, dates=dates)
    logs = []
    dl.run("S4a", db, base=tmp_r13, log=logs.append, free_fn=BIG)
    root = os.path.join(tmp_r13, "data_holdout")
    sp = json.load(open(os.path.join(root, "SPLIT.json")))
    assert sp["holdout_end_date"] == 20250828
    cal = pd.read_csv(os.path.join(root, "calendar_holdout.csv"))
    assert cal.date.max() == 20250828 and 20250905 not in set(cal.date) and cal.date.min() >= 20130101
    h = om.sha256(os.path.join(root, "calendar_holdout.csv"))
    assert dl.run("S4a", db, base=tmp_r13, log=logs.append, free_fn=BIG)["written"] == 0
    assert om.sha256(os.path.join(root, "calendar_holdout.csv")) == h
    with open(os.path.join(root, "calendar_holdout.csv"), "a") as f:
        f.write("20250829,1\n")
    with pytest.raises(dl.Stop, match="different content"):
        dl.finalize_split("holdout", root, log=logs.append)
    pat = [r"^[a-z]+/[a-z]+_(SPX|NDX|RUT)_\d{4}\.npz written: rows \d+, bytes \d+, \d+\.\ds, sha [0-9a-f]{12}$",
           r"^zerocd/zerocd_holdout\.npz written: rows \d+, bytes \d+, \d+\.\ds, sha [0-9a-f]{12}$",
           r"^\S+\.npz skipped \(MANIFEST sha256 matches\)$",
           r"^calendar_holdout: \d+ days, sha [0-9a-f]{12}; holdout end date \d{8}$"]
    assert logs and all(any(re.match(p, m) for p in pat) for m in logs)        # nothing but counts, bytes, hashes


def test_finalize_reads_only_dates_and_has_close(tmp_r13, monkeypatch):
    db = build_world(FakeDB(), HOLD)
    dl.run("S4a", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)
    keys = []
    real = om.own_arrays
    monkeypatch.setattr(om, "own_arrays", lambda p, k: (keys.append(tuple(k)), real(p, k))[1])
    dl.finalize_split("holdout", os.path.join(tmp_r13, "data_holdout"), log=lambda m: None)
    assert keys and set(keys) <= {("date",), ("date", "has_close")}


def test_s4b_after_s4a(tmp_r13):
    db = build_world(FakeDB(), HOLD)
    dl.run("S4a", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)
    r = dl.run("S4b", db, base=tmp_r13, log=lambda m: None, free_fn=BIG)
    assert r["written"] == 2 * 2 * 13
    m = lines(os.path.join(tmp_r13, "data_holdout", "MANIFEST.tsv"))
    assert len(m) == 3 * 13 + 2 * 13 + 1 + 2 * 2 * 13 and m.date_min.astype(int).min() >= 20130101


# ---------- S1 ----------

def test_meta_statements_select_no_values():
    allowed = [r"^SELECT table_name FROM information_schema\.tables WHERE table_schema = 'optionm' AND table_name IN \(",
               r"^SELECT table_name, column_name, data_type, ordinal_position FROM information_schema\.columns WHERE ",
               r"^SELECT secid, count\(\*\) AS n, min\(date\) AS dmin, max\(date\) AS dmax FROM optionm\.(opprcd|vsurfd|secprd)"
               r"\d{4} WHERE secid IN \(108105,102480,102434\) GROUP BY secid$",
               r"^SELECT count\(\*\) AS n, min\(date\) AS dmin, max\(date\) AS dmax FROM optionm\.zerocd WHERE date BETWEEN "
               r"'\d{4}-01-01' AND '\d{4}-12-31'$"]
    st = dl.meta_sql()
    assert len(st) == 2 + 3 * 30 + 30
    for _, _, _, q in st:
        assert any(re.match(p, q) for p in allowed), q


def test_run_meta_on_fake(tmp_r13):
    db = build_world(FakeDB(), [1996, 2013])
    tabs, cols, counts = dl.run_meta(db, out_dir=os.path.join(tmp_r13, "meta"), log=lambda m: None)
    assert set(tabs.table_name) == {"opprcd1996", "vsurfd1996", "secprd1996", "opprcd2013", "vsurfd2013", "secprd2013", "zerocd"}
    c = counts[counts.exists]
    assert set(c[(c.table == "opprcd") & (c.year == 1996)].secid) == {108105, 102480, 102434}
    assert int(c[(c.table == "opprcd") & (c.year == 1996) & (c.secid == 108105)].rows.iloc[0]) == 6
    assert not any("FROM optionm.opprcd1997" in q for q in db.sql)        # absent tables are not queried
    assert len(counts[(counts.table == "zerocd") & counts.exists]) == 30
    for f in ("tables.csv", "columns.csv", "counts.csv"):
        assert os.path.exists(os.path.join(tmp_r13, "meta", f))
