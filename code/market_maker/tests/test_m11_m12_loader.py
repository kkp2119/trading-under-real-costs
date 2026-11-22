"""M11 loader, lock and hashes; M12 units (LOT_FACTOR 1 and 100). Synthetic MANIFEST + npz in $R12_TEST_DIR only."""
import os

import numpy as np
import pandas as pd
import pytest

import loader
from config import B0, MODELS, UNLOCK
from fakesplit import write_split
from kernelio import kern


def test_dev_loads_and_checks(tmpdir_r12):
    write_split(tmpdir_r12, "dev", "stocks", [("AAA", "20190516", None), ("BBB", "20190517", None)])
    s = loader.Split("dev", "stocks", base=tmpdir_r12, results_dir=tmpdir_r12)
    b = s.load("AAA", "20190516")
    assert len(b) == 21 and all(v.dtype == np.int64 and v.shape == (23400,) for v in b.values())
    with pytest.raises(loader.Refused, match="not in MANIFEST"):
        s.load("AAA", "20190517")


def test_hash_mismatch_refused(tmpdir_r12):
    root = write_split(tmpdir_r12, "dev", "stocks", [("AAA", "20190516", None)])
    m = pd.read_csv(os.path.join(root, "MANIFEST.tsv"), sep="\t", dtype=str)
    m.loc[0, "sha256"] = "0" * 64
    m.to_csv(os.path.join(root, "MANIFEST.tsv"), sep="\t", index=False)
    with pytest.raises(loader.Refused, match="sha256"):
        loader.Split("dev", "stocks", base=tmpdir_r12, results_dir=tmpdir_r12).load("AAA", "20190516")


def test_holdout_and_reserve_locked_before_the_folder_is_touched(tmpdir_r12):
    for split in ("holdout", "reserve"):                             # the folders do not even exist: a lock error, not IO
        with pytest.raises(loader.Refused, match="locked"):
            loader.Split(split, "stocks", base=tmpdir_r12, results_dir=tmpdir_r12)
    write_split(tmpdir_r12, "holdout", "etf", [("IWM", "20210105", None)])
    with pytest.raises(loader.Refused, match="locked"):
        loader.Split("holdout", "etf", base=tmpdir_r12, results_dir=tmpdir_r12)
    open(os.path.join(tmpdir_r12, UNLOCK), "w").close()
    assert loader.Split("holdout", "etf", base=tmpdir_r12, results_dir=tmpdir_r12).load("IWM", "20210105")["bid"].any()


def test_date_range_violation_refused(tmpdir_r12):
    write_split(tmpdir_r12, "dev", "stocks", [("AAA", "20190516", None), ("AAA", "20210104", None)])
    with pytest.raises(loader.Refused, match="outside dev"):
        loader.Split("dev", "stocks", base=tmpdir_r12, results_dir=tmpdir_r12)


@pytest.mark.parametrize("bad", ["dtype", "missing", "rows", "identity"])
def test_bad_arrays_refused(tmpdir_r12, bad):
    def mutate(sym, day, a):
        if bad == "dtype":
            a["vol"] = a["vol"].astype(np.int64)
        elif bad == "missing":
            a.pop("n_xl")
        elif bad == "rows":
            a["bid"] = a["bid"][:-1]
        else:
            a["vol"] = a["vol"] + 1
        return a
    write_split(tmpdir_r12, "dev", "stocks", [("AAA", "20190516", None)], mutate=mutate)
    with pytest.raises(loader.Refused):
        loader.Split("dev", "stocks", base=tmpdir_r12, results_dir=tmpdir_r12).load("AAA", "20190516")


def test_units_lot_factor_1_and_100(tmpdir_r12):
    from synth import random_day
    b = random_day(77)
    b["bsz"], b["asz"] = (b["bsz"] // 100 + 1) * 100, (b["asz"] // 100 + 1) * 100      # whole lots
    sh_base, lot_base = os.path.join(tmpdir_r12, "sh"), os.path.join(tmpdir_r12, "lot")
    write_split(sh_base, "dev", "stocks", [("AAA", "20190516", b)])
    write_split(lot_base, "dev", "stocks", [("AAA", "20190516", b)], lot=100)
    b1 = loader.Split("dev", "stocks", base=sh_base, results_dir=tmpdir_r12, lot_factor=1).load("AAA", "20190516")
    b100 = loader.Split("dev", "stocks", base=lot_base, results_dir=tmpdir_r12, lot_factor=100).load("AAA", "20190516")
    assert all(np.array_equal(b1[c], b100[c]) for c in b1)
    for m in MODELS:
        assert kern(b1, B0, m) == kern(b100, B0, m)
