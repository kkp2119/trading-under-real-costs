"""Research 19: synthetic tests of run_test19 (guard, plan, sealed and rehearsal paths) on a made-up data folder.

No real data file is read: engine19.DATA / CAND / HERE and the paths of run_test19 point into a temporary folder
(test_engine19.make_tree). Writes results/run_test19_tests.json.
"""
import contextlib
import datetime as dt
import io
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine19 as G  # noqa: E402
import run_test19 as R  # noqa: E402
from test_engine19 import make_tree  # noqa: E402

TESTS = []
REAL_DATA, REAL_CAND = G.DATA, G.CAND
W_DEV = ((2010, 2), (2012, 12))
SEALED_CFG = {"window": ((2013, 1), (2015, 11)), "same_as_dev": ((2010, 2), (2012, 11)), "crisis": ((2013, 6), (2014, 10))}
REH_CFG = {"window": W_DEV, "same_as_dev": W_DEV, "crisis": ((2011, 8), (2012, 5))}


def test(f):
    TESTS.append(f)
    return f


@contextlib.contextmanager
def patched(obj, **kw):
    old = {k: getattr(obj, k) for k in kw}
    for k, v in kw.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


@contextlib.contextmanager
def fake(quirks=True):
    """A made-up data folder with its manifests, a results folder holding the 'development' files, everything patched in."""
    with tempfile.TemporaryDirectory() as root:
        here, data, cand = make_tree(root, quirks=quirks)
        res = os.path.join(root, "results")
        os.makedirs(res)
        halves = (((2010, 2), (2011, 6)), ((2011, 7), (2012, 12)))
        with patched(G, DATA=data, CAND=cand, HERE=here, HALVES=halves), \
                patched(R, RES=res, DEV_SUMMARY=os.path.join(res, "dev19_summary.json"), DEV_MONTHLY=os.path.join(res, "dev19_monthly.csv")):
            on_fake_data()
            D, _ = G.load(("build",))
            sims, sims2 = G.run_all(D), G.run_all(D, 2 * G.COST)
            tab = G.table(D, sims, W_DEV, sims2)
            ms = G.months(*W_DEV)
            with open(R.DEV_MONTHLY, "w", encoding="utf-8") as f:
                f.write("month," + ",".join(G.CANDS) + "\n")
                for k, m in enumerate(ms):
                    f.write(G.lab(m) + "," + ",".join(repr(tab[c]["net"][k]) for c in G.CANDS) + "\n")
            cols = list(R.CRISIS_COLS) + ["CS40-M"]
            with open(R.DEV_SUMMARY, "w", encoding="utf-8") as f:
                json.dump({"astar": G.select_astar(tab)[0], "candidates": {c: {k: tab[c][k] for k in R.KEYS} for c in G.CANDS},
                           "crisis": {"columns": cols, "rows": {G.lab(m): [tab[c]["net"][ms.index(m)] for c in cols] for m in REH_CFG["crisis"]}}}, f)
            yield {"root": root, "here": here, "data": data, "cand": cand, "res": res}


def quiet(f, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = f(*a, **kw)
    return rc, buf.getvalue()


def on_fake_data():
    """Stops a test that would otherwise touch the real data folder."""
    tmp = os.path.realpath(tempfile.gettempdir())
    for p in (G.DATA, G.CAND, G.HERE, R.RES):
        assert p not in (REAL_DATA, REAL_CAND, HERE, os.path.join(HERE, "results")) and os.path.realpath(p).startswith(tmp), p


def execute(*a, **kw):
    on_fake_data()
    return quiet(R.execute, *a, **kw)


def refresh_manifest(here, data):
    """Recompute the sha256 column of the main manifest after a test edits a data file."""
    p = os.path.join(here, "SEALED_MANIFEST19.tsv")
    with open(p, encoding="utf-8") as f:
        lines = f.read().splitlines()
    out = [lines[0]]
    for ln in lines[1:]:
        x = ln.split("\t")
        x[6] = G.sha(os.path.join(data, x[2]))
        out.append("\t".join(x))
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def blank(path, date):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert any(ln.startswith(date + ",") for ln in lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(date + "," if ln.startswith(date + ",") else ln for ln in lines) + "\n")


# ---------------------------------------------------------------- fixed texts
@test
def sentences_are_verbatim_from_spec19():
    with open(os.path.join(HERE, "SPEC19.md"), encoding="utf-8") as f:
        spec = f.read()
    for h in ("H1", "H2"):
        for ok in (True, False):
            assert f'"{R.SENTENCES[h][ok]}"' in spec, (h, ok)


@test
def windows_and_constants():
    assert R.MODES["sealed"]["window"] == ((2013, 1), (2026, 2)) and len(G.months(*R.MODES["sealed"]["window"])) == 158
    assert R.MODES["sealed"]["same_as_dev"] == ((1990, 1), (2012, 11)) and R.MODES["sealed"]["parts"] == ("build", "sealed")
    assert R.MODES["rehearsal"]["parts"] == ("build",) and R.MODES["rehearsal"]["allow_incomplete"] == ((2012, 12),)
    assert R.MODES["sealed"]["crisis"] == ((2015, 1), (2020, 3), (2024, 8)) and R.MODES["sealed"]["allow_incomplete"] == ()
    assert R.PHRASE == "연구19 테스트를 한 번 연다" and R.ASTAR in G.CANDS and R.BASELINE == 0.5
    assert (G.BOOT_B, G.BOOT_BLOCK, G.BOOT_SEED, G.COST, G.MARGIN) == (10_000, 12, 19, 0.0004, 0.05)



@test
def prediction_items_are_verbatim_from_the_documents():
    with open(os.path.join(HERE, "SPEC19.md"), encoding="utf-8") as f:
        spec = f.read()
    with open(os.path.join(HERE, "SPEC19_ADDENDUM.md"), encoding="utf-8") as f:
        add = f.read()
    for item, p in R.PRED[:5]:
        assert f"| {item} | 확률 {p} |" in spec, item
    for item, p in R.PRED[5:]:
        assert f"| {item} | 확률 {p} |" in add, item


# ---------------------------------------------------------------- guard
@contextlib.contextmanager
def guard_folder():
    with tempfile.TemporaryDirectory() as root:
        res = os.path.join(root, "results")
        os.makedirs(res)

        def put(rel, text):
            with open(os.path.join(root, rel), "w", encoding="utf-8") as f:
                f.write(text)

        for doc, path in R.DOCS:
            put(path, f"{doc} text\n")
            put(f"{doc}.sha256", f"{G.sha(os.path.join(root, path))}  {path}\n")
        put("engine19.py", "# engine\n")
        put("run_test19.py", "# runner\n")
        put("results/dev19_monthly.csv", "month\n")
        for p in R.CORE:
            if p.endswith(".tsv"):
                put(p, p + "\n")
        put("results/a1_check19.json", json.dumps({"dem_path": G.DEM_PATH}))
        put("results/dev19_summary.json", json.dumps({"astar": R.ASTAR, "engine19_sha256": G.sha(os.path.join(root, "engine19.py"))}))
        rows = [p for _, p in R.DOCS] + list(R.CORE)
        put("FREEZE19.md", "# freeze\n\n| 파일 | sha256 |\n|---|---|\n" + "".join(f"| `{p}` | `{G.sha(os.path.join(root, p))}` |\n" for p in rows))
        put("UNLOCK19.txt", R.PHRASE + "\n")
        with patched(R, HERE=root, RES=res, UNLOCK=os.path.join(root, "UNLOCK19.txt"), FREEZE=os.path.join(root, "FREEZE19.md"),
                     DEV_SUMMARY=os.path.join(res, "dev19_summary.json"), DEV_MONTHLY=os.path.join(res, "dev19_monthly.csv"),
                     A1=os.path.join(res, "a1_check19.json")):
            yield root, put


@test
def guard_passes_when_everything_is_in_place():
    with guard_folder() as (root, put):
        assert R.guard() == [] and len(R.frozen_files()) == len(R.DOCS) + len(R.CORE)


@test
def guard_needs_the_exact_unlock_phrase():
    with guard_folder() as (root, put):
        os.rename(os.path.join(root, "UNLOCK19.txt"), os.path.join(root, "UNLOCK19_away.txt"))
        assert R.guard() == ["no unlock phrase in UNLOCK19.txt"]
        for text, ok in ((R.PHRASE, True), (f"  {R.PHRASE} \n\n", True), ("연구19 테스트를 연다", False), (R.PHRASE + "!", False), ("", False),
                         (R.PHRASE.replace("19", "18"), False)):
            put("UNLOCK19.txt", text)
            assert (R.guard() == []) == ok, repr(text)


@test
def guard_stops_when_results_exist():
    with guard_folder() as (root, put):
        put("results/test19_report.md", "x")
        assert R.guard() == ["test results already exist"]


@test
def guard_names_a_changed_frozen_file():
    with guard_folder() as (root, put):
        put("run_test19.py", "# runner changed\n")
        assert R.guard() == ["frozen file differs: run_test19.py"]
        put("run_test19.py", "# runner\n")
        put("results/dev19_monthly.csv", "month,C0\n")
        assert R.guard() == ["frozen file differs: results/dev19_monthly.csv"]


@test
def guard_checks_the_three_documents():
    with guard_folder() as (root, put):
        put("SPEC19_ADDENDUM.md", "edited\n")
        assert R.guard() == ["frozen file differs: SPEC19_ADDENDUM.md", "SPEC19_ADDENDUM.md differs from SPEC19_ADDENDUM.sha256"]
    with guard_folder() as (root, put):
        os.rename(os.path.join(root, "SPEC19_ANSWERS.sha256"), os.path.join(root, "away.sha256"))
        assert R.guard() == ["SPEC19_ANSWERS.md differs from SPEC19_ANSWERS.sha256"]


@test
def guard_checks_astar_engine_and_dem_path():
    with guard_folder() as (root, put):
        with patched(R, ASTAR="C0"):
            assert R.guard() == ["ASTAR differs from the development selection"]
        put("results/a1_check19.json", json.dumps({"dem_path": "none"}))
        assert R.guard() == ["frozen file differs: results/a1_check19.json", "engine19.DEM_PATH differs from results/a1_check19.json"]
    with guard_folder() as (root, put):
        put("engine19.py", "# engine changed\n")
        assert R.guard() == ["frozen file differs: engine19.py", "engine19.py differs from the engine of the development run"]


@test
def guard_needs_the_core_rows_in_freeze():
    with guard_folder() as (root, put):
        with open(os.path.join(root, "FREEZE19.md"), encoding="utf-8") as f:
            text = f.read()
        put("FREEZE19.md", "".join(ln + "\n" for ln in text.splitlines() if "`engine19.py`" not in ln))
        assert R.guard() == ["FREEZE19.md missing or without the core files"]
        os.rename(os.path.join(root, "FREEZE19.md"), os.path.join(root, "FREEZE19_away.md"))
        assert R.guard() == ["FREEZE19.md missing or without the core files"]


@test
def go_with_a_failing_guard_reads_nothing():
    def boom(*a, **kw):
        raise AssertionError("data was loaded although the guard failed")

    with guard_folder() as (root, put):
        os.rename(os.path.join(root, "UNLOCK19.txt"), os.path.join(root, "UNLOCK19_away.txt"))
        with patched(G, load=boom), patched(sys, argv=["run_test19.py", "--go"]):
            rc, out = quiet(R.main)
        assert rc == 1 and "no unlock phrase" in out
        assert not any(f.startswith("test19") for f in os.listdir(os.path.join(root, "results")))


# ---------------------------------------------------------------- sealed path on made-up data
@test
def sealed_path_two_hypotheses_end_to_end():
    with fake() as F:
        rc, out = execute("sealed", "CS40-M", SEALED_CFG)
        assert rc == 0, out
        with open(os.path.join(F["res"], "test19_summary.json"), encoding="utf-8") as f:
            S = json.load(f)
        with open(os.path.join(F["res"], "test19_report.md"), encoding="utf-8") as f:
            rep = f.read()
        assert S["mode"] == "sealed" and S["astar"] == "CS40-M" and len(S["months"]) == 35 and S["months"][0] == "2013-01" and S["months"][-1] == "2015-11"
        assert S["hypotheses"] == 2 and S["holm_levels"] == [0.025, 0.05] and set(S["verdict"]) == {"H1", "H2"}
        idx = G.boot_index(35)
        d1 = G.boot_sharpe(S["monthly_net"]["C0"], idx)
        d2 = G.boot_sharpe(S["monthly_net"]["CS40-M"], idx) - d1
        assert S["verdict"]["H1"]["p"] == float(np.mean(d1 <= 0)) and S["verdict"]["H2"]["p"] == float(np.mean(d2 <= 0))
        assert abs(S["verdict"]["H1"]["estimate"] - G.sharpe(S["monthly_net"]["C0"])) < 1e-12
        assert abs(S["verdict"]["H2"]["estimate"] - (G.sharpe(S["monthly_net"]["CS40-M"]) - G.sharpe(S["monthly_net"]["C0"]))) < 1e-12
        h = G.holm({k: v["p"] for k, v in S["verdict"].items()})
        for k, v in S["verdict"].items():
            assert v["rejected"] == h[k]["rejected"] and v["level"] == h[k]["level"] and v["result"] == ("확인" if v["rejected"] else "확인 안 됨")
            assert v["sentence"] == R.SENTENCES[k][v["rejected"]] and f'"{v["sentence"]}"' in rep
        assert S["same_as_dev"] == {"months": 34, "numbers": 34 * 22, "max_abs_diff": 0.0, "equal": True}
        assert S["read"]["sealed"] == 21 and S["read"]["raw"] == 0 and S["read"]["files"] == 45
        assert S["plan"]["months"] == 35 and S["plan"]["incomplete_months"] == [] and S["plan"]["rate_fills_used"] == ["USD 2014-04"]
        assert S["plan"]["days_with_a_stale_quote"] == 1 and list(S["crisis"]["rows"]) == ["2013-06", "2014-10"]
        assert list(S["yearly"]) == ["2013", "2014", "2015"] and S["candidates"]["C0"]["half1_sum"] is None
        assert "## 판정\n" in rep and "0.025로" in rep and "수준 0.025, 0.05" in rep and "리허설" not in rep and "### 연도별 수익 합" in rep and "| 2013-06 |" in rep
        assert S["trials"] == {"before": 773, "added": 2, "after": 775} and "시행 수: 773 → 775 (봉인 판정 2개)" in rep
        pr = {x["item"]: x["outcome"] for x in S["predictions"]}
        t = S["candidates"]
        assert len(pr) == 7 and pr["H1 확인"] == S["verdict"]["H1"]["rejected"] and pr["H2 확인 (A*가 있을 때)"] == S["verdict"]["H2"]["rejected"]
        assert pr["A*가 C0이 아님"] is True and pr["A*의 빈도가 M"] is True and pr["봉인 C0 샤프가 −0.2 ~ 0.5 사이"] == (-0.2 <= t["C0"]["sharpe"] <= 0.5)
        assert pr["A*의 봉인 최대 낙폭이 C0보다 작음 (A*가 있을 때)"] == (t["CS40-M"]["max_drawdown"] < t["C0"]["max_drawdown"])
        assert pr["A* 필터의 D 버전이 봉인에서 M 버전보다 샤프가 높음"] == (t["CS40-D"]["sharpe"] > t["CS40-M"]["sharpe"])
        assert list(S["development"]) == ["C0", "CS40-M"] and "### 예측표 대조\n" in rep and "### 개발 구간과 비교" in rep and "청산일도 모두 있다" in rep
        assert out.strip() == rep.strip()


@test
def sealed_path_h1_alone_when_astar_is_c0():
    with fake() as F:
        rc, out = execute("sealed", "C0", SEALED_CFG)
        assert rc == 0, out
        with open(os.path.join(F["res"], "test19_summary.json"), encoding="utf-8") as f:
            S = json.load(f)
        assert S["hypotheses"] == 1 and S["holm_levels"] == [0.05] and S["verdict"]["H1"]["level"] == 0.05
        assert S["verdict"]["H2"] == {"result": "없음", "sentence": "개발에서 개선 없음 (A* = C0)"} and list(S["baseline_0.5"]) == ["C0"]
        assert "H1 하나이고 수준은 0.05" in out and "0.025로" not in out
        assert S["trials"]["after"] == 774 and [x["outcome"] for x in S["predictions"]][2:] == [False, None, None, None, None]


@test
def sealed_path_does_not_overwrite():
    with fake() as F:
        assert execute("sealed", "CS40-M", SEALED_CFG)[0] == 0
        before = G.sha(os.path.join(F["res"], "test19_summary.json"))
        rc, out = execute("sealed", "CS40-M", SEALED_CFG)
        assert rc == 1 and "already exist" in out and G.sha(os.path.join(F["res"], "test19_summary.json")) == before


def _nothing_written(F):
    return not any(f.startswith(("test19", "rehearsal19")) for f in os.listdir(F["res"]))


@test
def plan_stops_when_the_window_runs_past_the_data():
    with fake() as F:
        for window, word in ((((2013, 1), (2016, 2)), "without a business day"), (((2013, 1), (2015, 12)), "without an exit day")):
            rc, out = execute("sealed", "CS40-M", dict(SEALED_CFG, window=window))
            assert rc == 1 and "stop before any return is computed" in out and word in out and _nothing_written(F), window


@test
def plan_stops_when_a_month_has_no_portfolio():
    with fake() as F:
        blank(os.path.join(F["data"], "sealed", "IR3TIB01USM156N.csv"), "2014-05-01")       # April and May empty: June is not formed
        refresh_manifest(F["here"], F["data"])
        rc, out = execute("sealed", "CS40-M", SEALED_CFG)
        assert rc == 1 and "months without a portfolio: ['2014-06']" in out and _nothing_written(F)


@test
def changed_data_file_stops_before_anything():
    with fake() as F:
        with open(os.path.join(F["data"], "sealed", "DEXJPUS.csv"), "a") as f:
            f.write("2016-01-04,120.00\n")
        rc, out = execute("sealed", "CS40-M", SEALED_CFG)
        assert rc == 1 and "sha256 differs from the manifests" in out and _nothing_written(F)


@test
def development_months_must_reproduce():
    with fake() as F:
        with open(R.DEV_MONTHLY, encoding="utf-8") as f:
            lines = f.read().splitlines()
        x = lines[5].split(",")
        x[3] = repr(float(x[3]) + 1e-9)
        with open(R.DEV_MONTHLY, "w", encoding="utf-8") as f:
            f.write("\n".join(lines[:5] + [",".join(x)] + lines[6:]) + "\n")
        rc, out = execute("sealed", "CS40-M", SEALED_CFG)
        assert rc == 1 and "differ from results/dev19_monthly.csv" in out and _nothing_written(F)


@test
def sealed_statistics_do_not_depend_on_the_last_development_month_file_row():
    with fake() as F:
        on_fake_data()
        D, _ = G.load(("build", "sealed"))
        sims = G.run_all(D)
        s = R.same_as_dev(D, sims, W_DEV, R.DEV_MONTHLY)                                     # with 2012-12: one more interval
        assert not s["equal"] and s["max_abs_diff"] > 0 and R.same_as_dev(D, sims, SEALED_CFG["same_as_dev"], R.DEV_MONTHLY)["equal"]


# ---------------------------------------------------------------- rehearsal path on made-up data
@test
def rehearsal_reads_build_only_and_matches_the_development_files():
    with fake() as F:
        rc, out = execute("rehearsal", None, REH_CFG)
        assert rc == 0, out
        with open(os.path.join(F["res"], "rehearsal19_summary.json"), encoding="utf-8") as f:
            S = json.load(f)
        dev = json.load(open(R.DEV_SUMMARY, encoding="utf-8"))
        assert S["mode"] == "rehearsal" and S["read"] == dict(S["read"], files=24, sealed=0, raw=0) and len(S["months"]) == 35
        assert S["same_as_dev"]["equal"] and S["same_as_dev"]["numbers"] == 35 * 22 and S["same_as_dev"]["max_abs_diff"] == 0.0
        z = S["vs_dev"]
        assert z["max_abs_diff"] == 0.0 and z["not_comparable"] == [] and z["numbers"] > 22 * 15 and z["astar_from_rule"] == dev["astar"]
        assert S["plan"]["incomplete_months"] == ["2012-12"] and "리허설이다" in out and "## 판정 표 (리허설, 판정 아님)" in out
        assert S["trials"] is None and "시행 수" not in out and "리허설에서만 허용" in out and "예측표 대조 (리허설" in out
        assert not any(f.startswith("test19") for f in os.listdir(F["res"]))


@test
def rehearsal_flags_a_number_that_differs():
    with fake() as F:
        dev = json.load(open(R.DEV_SUMMARY, encoding="utf-8"))
        dev["candidates"]["V80-W"]["sharpe"] += 0.001
        with open(R.DEV_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(dev, f)
        rc, out = execute("rehearsal", None, REH_CFG)
        with open(os.path.join(F["res"], "rehearsal19_summary.json"), encoding="utf-8") as f:
            S = json.load(f)
        assert rc == 0 and abs(S["vs_dev"]["max_abs_diff"] - 0.001) < 1e-12


def main():
    res = []
    for f in TESTS:
        try:
            f()
            res.append({"test": f.__name__, "passed": True})
        except Exception as e:                                           # noqa: BLE001
            res.append({"test": f.__name__, "passed": False, "error": f"{type(e).__name__}: {e}"[:400]})
            print("FAIL", f.__name__, type(e).__name__, str(e)[:400])
    n = sum(r["passed"] for r in res)
    out = {"written": dt.datetime.now().isoformat(timespec="seconds"), "passed": n, "total": len(res), "tests": res}
    with open(os.path.join(HERE, "results", "run_test19_tests.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"{n}/{len(res)} passed")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
