"""Runners on a small synthetic world (no real bars): run_dev end to end, then the SPEC 17 step 4 refusals of
run_holdout one by one, then one holdout + reserve run and its refusal to run again."""
import json
import os

import pandas as pd
import pytest

import run_dev
import run_holdout
from config import COMPLETED, UNLOCK
from fakesplit import write_split

DEV = ["2019-01-02", "2019-01-03", "2019-01-04", "2019-01-07", "2019-01-08", "2019-01-09", "2019-01-10",
       "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10"]
HOLD = ["2021-01-04", "2021-01-05", "2021-01-06", "2026-09-29", "2026-09-30"]
RES = ["2026-10-01", "2026-10-02"]
UNI = {2019: ["AAA", "BBB"], 2020: ["AAA", "CCC"], 2021: ["DDD", "EEE"], 2026: ["DDD", "FFF"]}


def world(root):
    pd.DataFrame({"date": DEV + HOLD + RES, "close_time": "16:00"}).to_csv(os.path.join(root, "calendar.csv"), index=False)
    for y, syms in UNI.items():
        pd.DataFrame({"year": y, "slot": range(1, len(syms) + 1), "sym": syms, "order_rank": range(1, len(syms) + 1)}).to_csv(
            os.path.join(root, f"universe_{y}.tsv"), sep="\t", index=False)
    base = os.path.join(root, "data")
    seed = 0
    for split, days in (("dev", DEV), ("holdout", HOLD), ("reserve", RES)):
        st = [(s, d.replace("-", ""), None) for d in days for s in UNI[int(d[:4])]]
        et = [(s, d.replace("-", ""), None) for d in days for s in ("IWM", "DIA")]
        write_split(base, split, "stocks", st, seed=seed, vol_scale=200)        # market volume >> our fills (filter iii)
        write_split(base, split, "etf", et, seed=seed + 500, vol_scale=200)
        seed += 1000
    res = os.path.join(root, "results")
    os.makedirs(res)
    return base, res


@pytest.fixture(scope="module")
def dev_world(tmp_path_factory):
    base_dir = os.environ.get("R12_TEST_DIR")
    if not base_dir:
        pytest.skip("set R12_TEST_DIR")
    root = os.path.join(base_dir, "runners_" + os.urandom(4).hex())
    os.makedirs(root)
    base, res = world(root)
    out = run_dev.main(base=base, results_dir=res, universe_dir=root, calendar=os.path.join(root, "calendar.csv"),
                       n_workers=1, n_perm=3, ext_dir=os.path.join(root, "ext"))
    return root, base, res, out


def test_dev_run_writes_every_file(dev_world):
    root, base, res, out = dev_world
    for f in ("dev_grid.csv", "dev_grid_etf.csv", "dev_daily.csv", "dev_placebo.csv", "dev_checks.json",
              "dev_summary.json", "selected_config.json", "run_dev.log", "EXTERNAL.tsv"):
        assert os.path.exists(os.path.join(res, f)), f
    assert not os.path.exists(os.path.join(res, "dev_fills.csv"))         # large files live outside (E1-67)
    reg = pd.read_csv(os.path.join(res, "EXTERNAL.tsv"), sep="\t", dtype=str)
    names = set(reg.name)
    assert "dev_fills.csv" in names and {f"dev_fill_log_{n}.csv" for n in ("B0", "A_naive", "B_etf")} <= names
    assert {f"dev_qpath_{n}.csv" for n in ("B0", "A_naive", "B_etf")} <= names
    import results
    assert all(results.sha256(os.path.join(root, "ext", p)) == h for p, h in zip(reg.path, reg.sha256))
    assert json.load(open(os.path.join(res, "dev_checks.json")))["record_rerun_identical"] is True
    if out["candidate"]:
        assert {"regression", "yearly_net_R", "per_symbol_net_R"} <= set(out["side_numbers"])
    g = pd.read_csv(os.path.join(res, "dev_grid.csv"))
    assert {"B0", "A_naive"} <= set(g.setting) and len(set(g.setting)) >= 20
    sel = json.load(open(os.path.join(res, "selected_config.json")))
    assert sel["n_trials"] <= 69 and len(sel["filter_table"]) >= 19
    assert all("neighbors_checked" in t for t in sel["filter_table"])
    assert len(pd.read_csv(os.path.join(res, "dev_placebo.csv"))) == 3
    assert json.load(open(os.path.join(res, "dev_checks.json")))["flat_after_close"]


def force_g1(res, ok=True):
    p = os.path.join(res, "selected_config.json")
    sel = json.load(open(p))
    if sel["candidate"] is None:
        pytest.skip("synthetic dev run selected no candidate")
    sel["g1"]["pass"] = ok
    json.dump(sel, open(p, "w"))


YES = {"has_tag_fn": lambda: True, "is_clean_fn": lambda: True}


def test_holdout_refusals_one_by_one(dev_world):
    root, base, res, _ = dev_world
    force_g1(res, True)
    assert run_holdout.guard(res, **YES) == [f"no {UNLOCK}"]
    open(os.path.join(res, UNLOCK), "w").close()
    assert run_holdout.guard(res, **YES) == []
    assert run_holdout.guard(res, has_tag_fn=lambda: False, is_clean_fn=lambda: True) == ["no tag dev-selected-12"]
    assert run_holdout.guard(res, has_tag_fn=lambda: True, is_clean_fn=lambda: False) == ["work tree not clean"]
    force_g1(res, False)
    assert run_holdout.guard(res, **YES) == ["G1 not passed or no candidate in selected_config.json"]
    force_g1(res, True)
    import loader
    sp = lambda: [loader.Split("holdout", "stocks", base=base, results_dir=res)]
    assert run_holdout.guard(res, splits_fn=sp, **YES) == []
    m = os.path.join(base, "bars_1s_holdout_stocks", "MANIFEST.tsv")
    man = pd.read_csv(m, sep="\t", dtype=str)
    short = man[man.date < "20260930"]
    short.to_csv(m, sep="\t", index=False)
    assert any("data end" in w for w in run_holdout.guard(res, splits_fn=sp, **YES))
    bad = man.copy()
    bad.loc[0, "sha256"] = "f" * 64
    bad.to_csv(m, sep="\t", index=False)
    assert any("differ from MANIFEST" in w for w in run_holdout.guard(res, splits_fn=sp, **YES))
    man.to_csv(m, sep="\t", index=False)
    open(os.path.join(res, "holdout_x.csv"), "w").close()
    assert "holdout results already exist" in run_holdout.guard(res, **YES)
    os.rename(os.path.join(res, "holdout_x.csv"), os.path.join(root, "moved_holdout_x.csv"))   # moved aside, not deleted


def test_holdout_runs_once_then_refuses(dev_world):
    root, base, res, _ = dev_world
    force_g1(res, True)
    if not os.path.exists(os.path.join(res, UNLOCK)):
        open(os.path.join(res, UNLOCK), "w").close()
    s = run_holdout.main(base=base, results_dir=res, universe_dir=root, calendar=os.path.join(root, "calendar.csv"),
                         n_workers=1, guard_kw=YES, ext_dir=os.path.join(root, "ext"))
    assert s["verdict"] in ("a", "b", "c") and s["sentence"].endswith("탐색적 증거)")
    for f in ("holdout_summary.json", "holdout_daily.csv", "reserve_summary.json", COMPLETED):
        assert os.path.exists(os.path.join(res, f)), f
    assert {"regression", "yearly_net_R", "per_symbol_net_R", "external"} <= set(s)
    assert {"holdout_fills", "holdout_fill_log_cand", "holdout_fill_log_B0", "holdout_fill_log_A_naive",
            "holdout_fill_log_B_etf", "holdout_qpath_cand"} <= set(s["external"])
    with pytest.raises(SystemExit, match="already exist"):
        run_holdout.main(base=base, results_dir=res, universe_dir=root, calendar=os.path.join(root, "calendar.csv"),
                         n_workers=1, guard_kw=YES, ext_dir=os.path.join(root, "ext"))
