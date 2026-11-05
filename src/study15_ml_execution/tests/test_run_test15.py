"""run_test15 synthetic tests (Kyle 6단계 D): refusal without the unlock phrase (before any source path is touched), with a
wrong phrase, without --go and when results exist; bootstrap reproducibility; the same code on a 2020-01 development source
(temporary MANIFEST, temporary unlock file and output folders) reproduces the R2 per-parent IS numbers. The real
research15/UNLOCK15.txt is never created and no sealed folder is read. Run with python -m pytest."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R12 = os.path.join(os.path.dirname(HERE), "study12_market_maker")
sys.path.insert(0, HERE)
import r2_15 as R2  # noqa: E402
import run_test15 as T  # noqa: E402


def dirs(tmp):
    return {"out_dir": str(tmp / "out"), "ext_dir": str(tmp / "ext"), "unlock_path": str(tmp / "UNLOCK15.txt")}


def no_plan(*a, **k):
    raise AssertionError("a source was touched before the guard refused")


def test_refusals_happen_before_any_source(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "plan", no_plan)
    d = dirs(tmp_path)
    src = [(str(tmp_path / "sealed_not_here"), "x_{y}", "x.csv", "20210104")]
    with pytest.raises(T.Refused, match="no unlock phrase"):
        T.main(["--go"], sources=src, **d)
    open(d["unlock_path"], "w", encoding="utf-8").write("연구15 테스트를 연다\n")
    with pytest.raises(T.Refused, match="no unlock phrase"):
        T.main(["--go"], sources=src, **d)
    open(d["unlock_path"], "w", encoding="utf-8").write(T.PHRASE + "\n")
    with pytest.raises(T.Refused, match="--go"):
        T.main([], sources=src, **d)
    os.makedirs(d["out_dir"])
    open(os.path.join(d["out_dir"], "test15_summary.json"), "w").write("{}")
    with pytest.raises(T.Refused, match="already exist"):
        T.main(["--go"], sources=src, **d)
    assert not os.path.exists(os.path.join(HERE, "UNLOCK15.txt"))


def test_frozen_items_match():
    assert T.frozen_checks() == []


def test_bootstrap_reproducible():
    rng = np.random.default_rng(0)
    s, n = rng.normal(-1, 3, 300) * 20, np.full(300, 20)
    assert T.boot_ci(s, n) == T.boot_ci(s, n)
    est, lo, hi = T.boot_ci(s, n)
    assert lo < est < hi and np.isclose(est, s.sum() / n.sum())
    assert T.boot_ci(np.full(50, -40.0), np.full(50, 20)) == (-2.0, -2.0, -2.0)
    assert T.verdict(-0.3, -0.1) == "봉인 구간에서도 절감" and T.verdict(0.1, 0.3) == "더 비쌈" and T.verdict(-0.1, 0.1) == "판단 보류"


def test_2020_subset_reproduces_r2(tmp_path):
    root = os.path.join(os.environ.get("DATA_DIR", "data"), "research12_data/bars_1s_dev_stocks")
    m = pd.read_csv(os.path.join(root, "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False)
    m = m[m.date.str.startswith("202001")].copy()
    m["file"] = [os.path.join(root, f) for f in m.file]                  # absolute paths in a temporary MANIFEST
    src_root = tmp_path / "src2020"
    src_root.mkdir()
    m.to_csv(src_root / "MANIFEST.tsv", sep="\t", index=False)
    d = dirs(tmp_path)
    open(d["unlock_path"], "w", encoding="utf-8").write(T.PHRASE + "\n")
    src = [(str(src_root), os.path.join(R12, "universe_{y}.tsv"), os.path.join(R12, "calendar.csv"), "20200102")]
    s = T.main(["--go"], sources=src, workers=2, **d)
    assert s["n_symbol_days"] == len(m) and s["primary"]["verdict"] in ("봉인 구간에서도 절감", "더 비쌈", "판단 보류")
    assert all(os.path.exists(os.path.join(d["out_dir"], f)) for f in T.RESULT_FILES)
    read = open(os.path.join(d["out_dir"], "test15_read_paths.txt")).read().split()
    assert set(m.file) <= set(read)
    z = np.load(os.path.join(d["ext_dir"], "test15_parents.npz"))
    tp = pd.read_csv(os.path.join(d["ext_dir"], "test15_parents.csv.gz"), dtype={"date": str, "sym": str})
    rp, ra, rm = R2.load(2020)
    key = ["date", "sym", "start"]
    j = tp.merge(rp.reset_index()[key + ["index"]], on=key, how="left")
    assert j["index"].notna().all()
    ri = j["index"].to_numpy(int)
    for t_name, r_name in (("S0", "S0"), ("S1", "S1"), ("S2", "S2"), ("S2i", "S2i"), ("Estar", T.ESTAR_NAME)):
        a = z["is_r|1000_300"][:, list(T.METHODS).index(t_name)].astype(np.float32)
        b = ra["is_r"][ri, rm.index(r_name)]
        assert np.array_equal(a, b), t_name
