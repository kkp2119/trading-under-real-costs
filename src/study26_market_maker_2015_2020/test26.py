#!/usr/bin/env python3
"""Research 26 synthetic tests (step-1 prompt 3). Synthetic bars only, written by research 12's own test helpers
(tests/fakesplit.py) under $R26_TEST_DIR, a folder outside the repository (nothing deleted); no real bar is read.
  folders    two bar folders and two kinds of stock list are read together; a MANIFEST row outside a folder's range or
             the run window, or a symbol-day in two folders, stops the run
  absent     symbol-days without a file are left out and never replaced
  filter i   with six years every yearly net Sharpe must be > 0 (one year <= 0 or undefined fails the setting)
  no ETF     the run needs no ETF folder and runs no ETF control or mix step
  placebo    the parallel placebo gives exactly research 12's serial numbers
  end to end the development path on a six-year synthetic world: every file, record rerun identical, trial count, the
             SPEC26 5 tables, a second run refused
  regression on a synthetic 2019-2020 world, research 12's run_dev and this file's path give the same stock grid,
             selection table, daily P&L and (with research 12's formula) the same trial count
research 12's own suite is run separately (pytest study12_market_maker/tests). Writes results/tests26.json; exit 1 on a failure.
  R26_TEST_DIR=<scratch folder> /opt/anaconda3/bin/python test26.py"""
import json
import os
import sys
import uuid

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run26 as R  # noqa: E402  (registers the windows, puts study12_market_maker on the path)

D = R.D
sys.path.insert(0, os.path.join(D.R12, "tests"))
import config as C12  # noqa: E402
import grid  # noqa: E402
import loader as L12  # noqa: E402
import run_dev as RD  # noqa: E402
import select_candidate as sc  # noqa: E402
from fakesplit import write_split  # noqa: E402  (research 12's test helper)

RESULTS = []
DAYS = {y: [f"{y}-01-{d:02d}" for d in (2, 5, 6, 7, 8, 9, 12)] for y in range(2015, 2021)}   # 7 weekdays a year (not all real)
UNI15 = {2015: ["AAA", "BBB"], 2016: ["AAA", "CCC"], 2017: ["BBB", "CCC"], 2018: ["AAA", "DDD"]}
UNI12 = {2019: ["EEE", "FFF"], 2020: ["EEE", "GGG"]}
ABSENT = {("BBB", "20150108"), ("DDD", "20180109")}                     # no file (like ANN, VC, CAVM)


def check(name, ok, **info):
    RESULTS.append({"test": name, "pass": bool(ok), **info})


def world(root, years=range(2015, 2021), absent=ABSENT, extra_rows=None):
    """root/r15 (universe15_<Y>.tsv, calendar15.csv), root/r12 (universe_<Y>.tsv, calendar.csv), root/b15/bars_1s_train_stocks,
    root/b12/bars_1s_dev_stocks (+ an ETF folder for research 12's own run_dev)."""
    d15, d12 = os.path.join(root, "r15"), os.path.join(root, "r12")
    os.makedirs(d15)
    os.makedirs(d12)
    pd.DataFrame({"date": [d for y in years if y <= 2018 for d in DAYS[y]], "close_time": "16:00"}).to_csv(os.path.join(d15, "calendar15.csv"), index=False)
    pd.DataFrame({"date": [d for y in years if y >= 2019 for d in DAYS[y]], "close_time": "16:00"}).to_csv(os.path.join(d12, "calendar.csv"), index=False)
    for y, syms in UNI15.items():
        pd.DataFrame({"year": y, "slot": range(1, 3), "sym": syms, "order_rank": range(1, 3)}).to_csv(os.path.join(d15, f"universe15_{y}.tsv"), sep="\t", index=False)
    for y, syms in UNI12.items():
        pd.DataFrame({"year": y, "slot": range(1, 3), "sym": syms, "order_rank": range(1, 3)}).to_csv(os.path.join(d12, f"universe_{y}.tsv"), sep="\t", index=False)
    b15, b12 = os.path.join(root, "b15"), os.path.join(root, "b12")
    st15 = [(s, d.replace("-", ""), None) for y in years if y <= 2018 for d in DAYS[y] for s in UNI15[y] if (s, d.replace("-", "")) not in absent]
    st12 = [(s, d.replace("-", ""), None) for y in years if y >= 2019 for d in DAYS[y] for s in UNI12[y] if (s, d.replace("-", "")) not in absent]
    st15 += (extra_rows or {}).get("train", [])
    st12 += (extra_rows or {}).get("dev", [])
    if st15:
        write_split(b15, "train", "stocks", st15, seed=0, vol_scale=200)
        pd.DataFrame([(s, d, "not in mastm", "x") for s, d in sorted(absent) if d < "2019"], columns=["sym", "date", "reason", "utc"]).to_csv(
            os.path.join(b15, "bars_1s_train_stocks", "skipped_absent.tsv"), sep="\t", index=False)
    if st12:
        write_split(b12, "dev", "stocks", st12, seed=5000, vol_scale=200)
        write_split(b12, "dev", "etf", [(s, d.replace("-", ""), None) for y in years if y >= 2019 for d in DAYS[y] for s in ("IWM", "DIA")],
                    seed=9000, vol_scale=200)
    src = {"train": (b15, D.TRAIN), "dev": (b12, C12.SPLITS["dev"])}
    cal = D.build_calendar(os.path.join(root, "calendar26.csv"), os.path.join(d15, "calendar15.csv"), os.path.join(d12, "calendar.csv"))
    return src, (d15, d12), cal, b12, d12


def folder_tests(tmp):
    src, lists, calp, _, _ = world(os.path.join(tmp, "f"))
    cal = D.Calendar(calp)
    S = D.TwoFolders(C12.SPLITS["dev26"], sources=src)
    av = S.available()
    t = D.tasks(cal, "dev26", av, lists)
    by_year = {y: sorted({s for s, d in t if d[:4] == str(y)}) for y in range(2015, 2021)}
    want = {**{y: sorted(set(v)) for y, v in UNI15.items()}, **{y: sorted(set(v)) for y, v in UNI12.items()}}
    p15 = L12.Split("train", "stocks", base=src["train"][0])
    p12 = L12.Split("dev", "stocks", base=src["dev"][0])
    same = all(np.array_equal(S.load(s, d)[c], (p15 if d < "2019" else p12).load(s, d)[c]) for s, d in [t[0], t[-1], t[len(t) // 2]] for c in ("bid", "ask", "bsz", "vol"))
    check("two folders and two kinds of list are read together: 2015-2018 from the train folder with universe15_<Y>.tsv, 2019-2020 from the "
          "dev folder with universe_<Y>.tsv; every symbol-day loads exactly what research 12's loader gives from its own folder",
          by_year == want and len(S.parts) == 2 and same and len(t) == 6 * 7 * 2 - len(ABSENT), tasks=len(t), by_year=by_year)
    R2 = D.TwoFolders(C12.SPLITS["regress26"], sources=src)
    check("the regression window 2019-2020 opens only the dev folder", len(R2.parts) == 1 and all(d >= "20190102" for _, d in R2.available()))
    cal_rows = pd.read_csv(calp, dtype=str)
    check("calendar26: the 2015-2018 rows of calendar15 and the 2019-2020 rows of research 12's calendar, in order, once",
          list(cal_rows.date) == [d for y in range(2015, 2021) for d in DAYS[y]] and cal_rows.date.is_unique)
    try:
        D.build_calendar(calp, os.path.join(tmp, "f", "r12", "calendar.csv"), os.path.join(tmp, "f", "r12", "calendar.csv"))
        cal_kept = False
    except D.Refused:
        cal_kept = True
    check("an existing calendar26 with other content is not overwritten", cal_kept)
    # out of range: a 2019 row in the train folder; a 2018 row in the dev folder; one symbol-day in two folders
    errs = {}
    for label, extra in (("train folder holds a 2019 row", {"train": [("AAA", "20190102", None)]}),
                         ("dev folder holds a 2018 row", {"dev": [("EEE", "20180102", None)]})):
        s2, _, _, _, _ = world(os.path.join(tmp, label.replace(" ", "_")), extra_rows=extra)
        try:
            D.TwoFolders(C12.SPLITS["dev26"], sources=s2)
            errs[label] = None
        except D.Refused as e:
            errs[label] = str(e)[:90]
    s3, _, _, _, _ = world(os.path.join(tmp, "dup"))
    C12.SPLITS["train_b"] = D.TRAIN
    write_split(os.path.join(tmp, "dup", "b15b"), "train_b", "stocks", [("AAA", "20150102", None)], seed=1)
    try:
        D.TwoFolders(C12.SPLITS["dev26"], sources={**s3, "train_b": (os.path.join(tmp, "dup", "b15b"), D.TRAIN)})
        errs["one symbol-day in two folders"] = None
    except D.Refused as e:
        errs["one symbol-day in two folders"] = str(e)[:90]
    try:
        D.TwoFolders(("20190102", "20190110"), sources=src)
        errs["a folder row outside a narrower run window"] = None
    except D.Refused as e:
        errs["a folder row outside a narrower run window"] = str(e)[:90]
    check("a MANIFEST row outside its folder's range or outside the run window, or a symbol-day in two folders, stops the run",
          all(v is not None for v in errs.values()) and len(errs) == 4, errors=errs)
    miss = D.missing(cal, "dev26", av, lists)
    per_day = pd.Series([d for _, d in t]).value_counts()
    check("absent symbol-days (no file) are left out of the tasks, listed as missing and not replaced by another symbol",
          set(miss) == ABSENT and not (set(t) & ABSENT) and per_day.max() == 2 and per_day["20150108"] == 1 and per_day["20180109"] == 1
          and S.absent == {k for k in ABSENT if k[1] < "2019"}, missing=sorted(miss))


def filter_tests():
    base = {"sharpe": 1.0, "sharpe_Q50": 0.5, "fills_per_symday": 3.0, "participation": 0.001}
    six = {f"sharpe_{y}": 0.4 for y in range(2015, 2021)}
    f_all, _ = sc.filters({**base, **six}, C12.B0, {})
    res = {}
    for y in range(2015, 2021):
        for bad in (0.0, -0.1, float("nan")):
            f, _ = sc.filters({**base, **six, f"sharpe_{y}": bad}, C12.B0, {})
            res[(y, bad)] = f["i_years"]
    check("filter (i) with six years: passes when all six yearly Sharpes are > 0 and fails when any one of them is 0, negative or "
          "undefined (research 12's filter, unchanged, reads every 'sharpe_<year>' key)", f_all["i_years"] and not any(res.values()),
          failing=sum(not v for v in res.values()))
    obs = [d.replace("-", "") for y in range(2015, 2021) for d in DAYS[y]]
    runs = pd.DataFrame({"setting": "X", "model": "Q", "date": obs * 2, "sym": ["A"] * len(obs) + ["B"] * len(obs), "skip": 0,
                         "cash": np.arange(2 * len(obs)) * 1000 - 30000, **{k: 0 for k in ("sh_mkt_buy", "sh_mkt_sell", "sh_lim_buy", "sh_lim_sell")},
                         "n_fill_ev": 2, "mkt_vol": 1})
    m = sc.setting_metrics(pd.concat([runs, runs.assign(model="Q50")]), "X", obs)
    check("setting_metrics makes one yearly Sharpe per year of the observation days (sharpe_2015 .. sharpe_2020)",
          sorted(k for k in m if k.startswith("sharpe_2")) == [f"sharpe_{y}" for y in range(2015, 2021)], keys=sorted(m))


def end_to_end(tmp):
    src, lists, calp, _, _ = world(os.path.join(tmp, "e"))
    res, ext = os.path.join(tmp, "e", "results"), os.path.join(tmp, "e", "ext")
    out = R.main("dev26", results_dir=res, ext_dir=ext, n_workers=1, n_perm=3, calendar=calp, sources=src, list_dirs=lists, compare_r12=False)
    files = ("dev26_grid.csv", "dev26_daily.csv", "dev26_yearly.csv", "dev26_markouts.csv", "dev26_cumulative.csv", "dev26_placebo.csv",
             "dev26_summary.json", "dev26.log", "EXTERNAL.tsv")
    g = pd.read_csv(os.path.join(res, "dev26_grid.csv"))
    yt = pd.read_csv(os.path.join(res, "dev26_yearly.csv"), dtype={"period": str})
    n_settings = len(out["filter_table"])
    check("end to end on a six-year synthetic world: every result file, the per symbol-day table and fill logs outside the results "
          "folder, record rerun identical, yearly Sharpes for 2015-2020 in the selection table",
          all(os.path.exists(os.path.join(res, f)) for f in files) and out["checks"]["record_rerun_identical"]
          and all(f"sharpe_{y}" in out["filter_table"][0] for y in range(2015, 2021)) and os.path.isdir(os.path.join(ext, "dev26"))
          and not any(f.endswith("_fills.csv") for f in os.listdir(res)), settings=n_settings, candidate=out["candidate"] and out["candidate"]["name"])
    check("no ETF step: no ETF folder is needed or opened, no ETF grid or mix is written, every grid row is a stock row",
          set(g.universe) == {"stocks"} and not any("etf" in f for f in os.listdir(res)) and out["etf_and_mix"].startswith("not run")
          and "mix" not in out and all("etf" not in r.lower() for r in out["checks"]["bar_folders"]))
    distinct = len({r["cid"] for r in out["filter_table"]})
    check("trial count: 865 + the distinct stock settings run + A_naive (no ETF control, no mix)",
          out["trials"]["before"] == 865 and out["trials"]["added"] == distinct + 1 and out["trials"]["cumulative"] == 865 + distinct + 1,
          trials=out["trials"])
    periods = set(yt.period)
    check("SPEC26 5 tables: yearly table for B0 and the reported setting under Q, S, T (each year, 2015-2018, 2019-2020, all); markouts "
          "k = 1, 5, 30, 60 s and by time of day; cumulative daily net for B0 Q, best Q, B0 T, best T; the prediction check rows",
          {str(y) for y in range(2015, 2021)} | {"2015-2018", "2019-2020", "all"} <= periods and set(yt.model) == {"Q", "S", "T"}
          and {"mk1_ticks", "mk60_ticks", "open_mk5_ticks"} <= set(pd.read_csv(os.path.join(res, "dev26_markouts.csv")).columns)
          and len(pd.read_csv(os.path.join(res, "dev26_cumulative.csv"))) == len(pd.read_csv(os.path.join(res, "dev26_daily.csv")))
          and len(out["predictions_check"]) == 4 and out["sentence"] in (R.SENTENCES["none"], R.SENTENCES["g1"].format(name=out["candidate"]["name"] if out["candidate"] else "")))
    try:
        R.main("dev26", results_dir=res, ext_dir=ext, n_workers=1, n_perm=3, calendar=calp, sources=src, list_dirs=lists, compare_r12=False)
        again = None
    except SystemExit as e:
        again = str(e)
    check("a second development run is refused (its results exist)", again is not None and "once" in again, message=again)
    # placebo: parallel = research 12's serial loop
    cal = D.Calendar(calp)
    S, tasks, gates, _, _ = R.prepare("dev26", cal, None, src, lists)
    log = lambda m: None   # noqa: E731
    par = R.placebo_table(S, cal, tasks, gates, 4, 2, log)
    ser = RD.placebo_table(S, cal, tasks, gates, 4, log)
    check("the parallel placebo (2 workers) gives exactly research 12's serial placebo numbers", par.equals(ser), parallel=par.round(6).to_dict("list"))


def regression_world(tmp):
    """research 12's run_dev on a synthetic 2019-2020 world against this file's path on the same stock folder."""
    root = os.path.join(tmp, "r")
    src, lists, calp, b12, d12 = world(root, years=(2019, 2020), absent=set())
    r12res, r26res = os.path.join(root, "res12"), os.path.join(root, "res26")
    o12 = RD.main(base=b12, results_dir=r12res, universe_dir=d12, calendar=os.path.join(d12, "calendar.csv"), n_workers=1, n_perm=2,
                  ext_dir=os.path.join(root, "ext12"))
    o26 = R.main("regress26", results_dir=r26res, ext_dir=os.path.join(root, "ext26"), n_workers=1, calendar=calp,
                 sources={"dev": src["dev"]}, list_dirs=lists, run_placebo=False, compare_r12=False)
    g12 = pd.read_csv(os.path.join(r12res, "dev_grid.csv"), dtype={"cid": str})
    g12 = g12[g12.universe == "stocks"].reset_index(drop=True)
    g26 = pd.read_csv(os.path.join(r26res, "regress26_grid.csv"), dtype={"cid": str})
    d12_ = pd.read_csv(os.path.join(r12res, "dev_daily.csv"), dtype={"date": str})
    d26_ = pd.read_csv(os.path.join(r26res, "regress26_daily.csv"), dtype={"date": str})
    check("regression path on a synthetic 2019-2020 world: research 12's run_dev and run26 give the same stock grid, selection table, adopted "
          "values, candidate, daily P&L, and research 12's trial count (with its ETF controls counted by its formula)",
          g12.equals(g26) and R._json(o26["filter_table"]) == R._json(o12["filter_table"]) and R._json(o26["adopted"]) == R._json(o12["adopted"])
          and (o26["candidate"] or {}).get("name") == (o12["candidate"] or {}).get("name") and d12_.equals(d26_)
          and o26["n_trials_research12_formula"] == o12["n_trials"], n_trials=[o26["n_trials_research12_formula"], o12["n_trials"]],
          candidate=[(o26["candidate"] or {}).get("name"), (o12["candidate"] or {}).get("name")])


def repo_tests():
    check("constants: windows dev26 2015-01-02..2020-12-31 and regress26 = research 12's dev window; trials 865 before; the fixed sentences "
          "of SPEC26 3; research 12's grid, adoption gain and G1 thresholds are the ones imported",
          C12.SPLITS["dev26"] == ("20150102", "20201231") and C12.SPLITS["regress26"] == ("20190102", "20201231") == C12.SPLITS["dev"]
          and R.TRIALS_BEFORE == 865 and R.SENTENCES["none"].startswith("Market Maker는 2015–2020 개발 구간에서도") and C12.ADOPT_GAIN == 0.5
          and C12.G1_SHARPE == 1.0 and C12.G1_QUOTABLE == 0.90 and len(grid.stage1()) == 19 and C12.COST["R"] == (30, 0)
          and C12.MODEL_CODE["Q"] == (2, 4) and not os.path.exists(os.path.join(HERE, "results", "dev26_summary.json")))


def main():
    base = os.environ.get("R26_TEST_DIR")
    if not base:
        raise SystemExit("set R26_TEST_DIR (a folder outside the repository; nothing is deleted)")
    tmp = os.path.join(base, "r26_tests_" + uuid.uuid4().hex[:8])
    os.makedirs(tmp)
    folder_tests(tmp)
    filter_tests()
    end_to_end(tmp)
    regression_world(tmp)
    repo_tests()
    n_ok = sum(r["pass"] for r in RESULTS)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", "tests26.json")
    if os.path.exists(path):
        os.rename(path, path.replace(".json", f"_{int(os.path.getmtime(path))}.json"))
    with open(path, "w") as fh:
        json.dump({"passed": n_ok, "total": len(RESULTS), "tests": RESULTS}, fh, indent=1, default=str, ensure_ascii=False)
    for r in RESULTS:
        print(("PASS " if r["pass"] else "FAIL ") + r["test"][:150])
        if not r["pass"]:
            print("     ", json.dumps(r, default=str, ensure_ascii=False)[:2000])
    print(f"{n_ok}/{len(RESULTS)} passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
