"""Record mode (E1-66, E1-68), external files (E1-67), SPEC 10 side numbers (E1-69): recording leaves the run table
bit-identical; the fill log is the kernel's fills and equals the reference implementation's; the inventory path equals
positions computed from refsim's fills; external files carry the recorded sha256 and are never overwritten."""
import os

import numpy as np
import pandas as pd
import pytest

import loader
import results
import runner
import session
from config import MODELS, QPATH_BARS
from fakesplit import write_split
from kernelio import ref
from synth import settings

DAYS = [("AAA", "20190516"), ("BBB", "20190516"), ("AAA", "20190517"), ("BBB", "20190517")]
SETS = settings()
REC = ["B0", "event", "A_naive"]


@pytest.fixture
def world(tmpdir_r12):
    write_split(tmpdir_r12, "dev", "stocks", [(s, d, None) for s, d in DAYS], seed=70)
    cal_csv = os.path.join(tmpdir_r12, "calendar.csv")
    pd.DataFrame({"date": ["2019-05-16", "2019-05-17"], "close_time": "16:00"}).to_csv(cal_csv, index=False)
    split = loader.Split("dev", "stocks", base=tmpdir_r12, results_dir=tmpdir_r12)
    return tmpdir_r12, split, session.Calendar(cal_csv)


def test_recording_does_not_change_results(world):
    _, split, cal = world
    st = [(n, c) for n, c, _ in SETS if n in REC + ["obi=True"]]
    gates = {k: True for k in DAYS}
    off, d_off = runner.run_tasks(split, cal, DAYS, gates, st, MODELS, 1, ["B0"])
    on, d_on, rec = runner.run_tasks_record(split, cal, DAYS, gates, st, MODELS, 1, ["B0"], REC)
    pd.testing.assert_frame_equal(off, on)
    pd.testing.assert_frame_equal(d_off, d_on)
    on2, _, rec2 = runner.run_tasks_record(split, cal, DAYS, gates, st, MODELS, 2, ["B0"], REC)
    pd.testing.assert_frame_equal(on, on2)
    assert all(rec[n]["fills"].equals(rec2[n]["fills"]) for n in REC)


def test_fill_log_and_q_path_match_reference(world):
    _, split, cal = world
    st = [(n, c) for n, c, _ in SETS if n in REC]
    cds = {n: d for n, _, d in SETS}
    _, _, rec = runner.run_tasks_record(split, cal, DAYS, {k: True for k in DAYS}, st, ["Q"], 1, (), REC)
    for name in REC:
        f = rec[name]["fills"]
        idx, q = rec[name]["qpath"]
        assert q.shape == (len(DAYS), len(QPATH_BARS)) and len(idx) == len(DAYS)
        for i, (sym, day) in enumerate(idx.itertuples(index=False)):
            r = ref(split.load(sym, day), cds[name], "Q")
            mine = f[(f.sym == sym) & (f.date == day)][["u", "side", "kind", "qty", "price", "ref_mid2", "dec"]]
            assert sorted(map(tuple, mine.values.tolist())) == r["fills"], (name, sym, day)
            pos = np.array([sum(s * x for u, s, _, x, *_ in r["fills"] if u <= g) for g in QPATH_BARS])
            assert np.array_equal(q[i], pos), (name, sym, day)


def test_external_files_hash_and_no_overwrite(world):
    root, split, cal = world
    ext, res = os.path.join(root, "ext"), os.path.join(root, "res")
    df = pd.DataFrame({"a": [1, 2, 3]})
    rec = results.external_csv(df, "x.csv", ext, "dev", "RID1", res)
    assert rec["sha256"] == results.sha256(os.path.join(ext, rec["path"])) and rec["rows"] == 3
    reg = pd.read_csv(os.path.join(res, "EXTERNAL.tsv"), sep="\t", dtype=str)
    assert list(reg.columns) == results.EXTERNAL_COLS and reg.sha256.tolist() == [rec["sha256"]]
    with pytest.raises(FileExistsError):
        results.external_csv(df, "x.csv", ext, "dev", "RID1", res)


def test_side_numbers_by_hand():
    days = pd.DataFrame({"sym": ["A", "B", "A", "B"], "date": ["d1", "d1", "d2", "d2"],
                         "first_mid2": [1000, 2000, 1000, 0], "last_mid2": [1100, 1900, 900, 500]})
    assert np.allclose(results.market_proxy(days, ["d1", "d2", "d3"])[:2], [(0.1 - 0.05) / 2, -0.1], equal_nan=True)
    assert np.isnan(results.market_proxy(days, ["d3"])[0])
    assert results.yearly([1.0, 2.0, 4.0], ["20190102", "20191231", "20200102"]) == {"2019": 3.0, "2020": 4.0}
    runs = pd.DataFrame({"setting": "X", "model": "Q", "sym": ["A", "B", "A"], "cash": [10_000, 20_000, -5_000],
                         "sh_mkt_buy": 0, "sh_mkt_sell": [100, 0, 0], "sh_lim_buy": 0, "sh_lim_sell": 0})
    assert results.per_symbol(runs, "X") == {"A": (10_000 - 3_000 - 5_000) / 1e4, "B": 2.0}
