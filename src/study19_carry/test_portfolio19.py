"""Research 19: synthetic tests of portfolio19 (PORTFOLIO19.md) on made-up inputs.

No real data file and no real result file is read: the data folder comes from test_engine19.make_tree, the option daily
files and the carry files are written into a temporary folder. Writes results/portfolio19_tests.json.
"""
import contextlib
import datetime as dt
import io
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine19 as G  # noqa: E402
import portfolio19 as P  # noqa: E402
from test_engine19 import bdays, make_tree  # noqa: E402

TESTS = []
W_DEV, W_SEALED = ((2010, 2), (2012, 12)), ((2013, 1), (2015, 11))
C_DEV, C_SEALED = ((2011, 8),), ((2013, 6), (2014, 10))


def test(f):
    TESTS.append(f)
    return f


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


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


def quiet(f, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = f(*a, **kw)
    return rc, buf.getvalue()


def option_file(path, a, b, seed):
    rng = np.random.default_rng(seed)
    with open(path, "w", encoding="utf-8") as f:
        f.write("date,O2_base,O2_stress,O2_opt\n")
        for d in bdays(a, b):
            x = rng.normal(300, 6000)
            f.write(f"{d.replace('-', '')},{x!r},{x - 50!r},{x + 80!r}\n")


@contextlib.contextmanager
def fake(test_results=True):
    """Made-up data folder, option files, development files and (optionally) the two sealed-test result files."""
    with tempfile.TemporaryDirectory() as root:
        here, data, cand = make_tree(root)
        res = os.path.join(root, "results")
        os.makedirs(res)
        opt = {"dev": os.path.join(root, "opt_dev.csv"), "sealed": os.path.join(root, "opt_sealed.csv")}
        option_file(opt["dev"], "2010-01-04", "2012-12-31", 1)
        option_file(opt["sealed"], "2013-01-02", "2015-11-30", 2)
        sha = {k: G.sha(p) for k, p in opt.items()}
        rng = np.random.default_rng(3)
        ms = G.months(*W_DEV)
        with open(os.path.join(res, "dev19_monthly.csv"), "w", encoding="utf-8") as f:
            f.write("month," + ",".join(G.CANDS) + "\n")
            for m in ms:
                f.write(G.lab(m) + "," + ",".join(repr(float(rng.normal(0.003, 0.02))) for _ in G.CANDS) + "\n")
        with open(os.path.join(res, "dev19_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"astar": "CS40-M"}, f)
        if test_results:
            sm = G.months((2013, 1), (2016, 2))
            with open(os.path.join(res, "test19_summary.json"), "w", encoding="utf-8") as f:
                json.dump({"mode": "sealed", "astar": "CS40-M", "months": [G.lab(m) for m in sm],
                           "monthly_net": {c: [float(rng.normal(0.002, 0.02)) for _ in sm] for c in G.CANDS}}, f)
            with open(os.path.join(res, "test19_report.md"), "w", encoding="utf-8") as f:
                f.write("x\n")
        with patched(G, DATA=data, CAND=cand, HERE=here):
            yield {"root": root, "res": res, "opt": opt, "sha": sha}


def dev(F):
    return quiet(P.run_dev, F["res"], F["opt"], W_DEV, C_DEV, F["sha"])


def sealed(F):
    return quiet(P.run_sealed, F["res"], F["opt"], W_SEALED, C_SEALED, F["sha"])


def load(F, name):
    with open(os.path.join(F["res"], name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- the option month
def _csv(root, rows, head="date,O2_base,O2_opt"):
    p = os.path.join(root, "o.csv")
    with open(p, "w", encoding="utf-8") as f:
        f.write(head + "\n" + "".join(r + "\n" for r in rows))
    return p


@test
def option_month_is_the_calendar_month_sum_over_one_million():
    with tempfile.TemporaryDirectory() as root:
        p = _csv(root, ["19960305,9999,1", "19960306,1000,1", "19960329,2000,1", "19960401,-500,1", "19960430,1500,1"])
        r, n = P.option_monthly(p, ((1996, 3), (1996, 4)))
        assert r == [0.003, 0.001] and n == [2, 2]                     # 03-05 is before the statistics start


@test
def option_month_missing_stops():
    with tempfile.TemporaryDirectory() as root:
        p = _csv(root, ["19960306,1000,1", "19960501,5,1"])
        for window, col in ((((1996, 3), (1996, 5)), "O2_base"), (((1996, 3), (1996, 3)), "O9_base")):
            try:
                P.option_monthly(p, window, col)
                raise AssertionError("must stop")
            except ValueError:
                pass


@test
def option_month_rejects_an_odd_date_or_value():
    with tempfile.TemporaryDirectory() as root:
        for rows in (["1996-03-06,1,1"], ["19960306,,1"]):
            try:
                P.option_monthly(_csv(root, rows), ((1996, 3), (1996, 3)))
                raise AssertionError("must stop")
            except ValueError:
                pass


# ---------------------------------------------------------------- weights and the bundle
@test
def weights_are_inverse_volatility():
    w = P.inverse_vol([0.02, -0.02, 0.02, -0.02], [0.04, -0.04, 0.04, -0.04])
    assert close(w["carry"], 2 / 3) and close(w["option"], 1 / 3) and close(w["sigma_option_monthly"], 2 * w["sigma_carry_monthly"])
    assert close(w["sigma_carry_monthly"], float(np.std([0.02, -0.02, 0.02, -0.02], ddof=1)))


def _thirty():
    rng = np.random.default_rng(11)
    c, o = rng.normal(0.004, 0.02, 30), rng.normal(0.002, 0.01, 30)
    w = {"carry": 0.25, "option": 0.75}
    return c, o, w, P.evaluate(((2000, 1), (2002, 6)), c, o, w, list(range(1, 31)), ((2000, 3), (2010, 1)))


@test
def bundle_is_the_weighted_sum_and_metrics_by_hand():
    c, o, w, E = _thirty()
    p = 0.25 * c + 0.75 * o
    assert np.allclose(E["bundle"], p, atol=0, rtol=0) and E["months"][0] == "2000-01" and E["months"][-1] == "2002-06"
    m = E["metrics"]["bundle"]
    assert close(m["sharpe"], p.mean() / p.std(ddof=1) * math.sqrt(12)) and close(m["ann_return"], p.mean() * 12)
    assert close(m["ann_vol"], p.std(ddof=1) * math.sqrt(12)) and close(m["max_drawdown"], G.max_drawdown(p)) and close(m["worst_12m"], G.worst12(p))
    assert close(E["corr_all"], float(np.corrcoef(c, o)[0, 1]))
    assert E["crisis"] == {"2000-03": {"carry": float(c[2]), "option": float(o[2]), "bundle": float(p[2])}}   # 2010-01 is outside


@test
def vix_jump_months_are_at_or_above_the_90th_percentile():
    c, o, w, E = _thirty()
    j = E["vix_jump"]
    assert close(j["threshold_vix_points"], 27.1) and j["months"] == ["2002-04", "2002-05", "2002-06"] and j["n"] == 3
    assert close(j["corr"], float(np.corrcoef(c[27:], o[27:])[0, 1])) and close(j["carry_sum"], float(c[27:].sum()))


@test
def vix_jump_handles_a_month_without_a_vix_change():
    rng = np.random.default_rng(12)
    c, o = rng.normal(0, 0.02, 30), rng.normal(0, 0.01, 30)
    dv = [None] + list(range(2, 31))
    E = P.evaluate(((2000, 1), (2002, 6)), c, o, {"carry": 0.5, "option": 0.5}, dv, ())
    assert E["vix_jump"]["months_without_vix_change"] == 1 and E["vix_jump"]["n"] == 3


@test
def bundle_sharpe_must_beat_both_sleeves_strictly():
    x = np.array([0.03, -0.01, 0.02, 0.00, 0.01, -0.02, 0.04, 0.01, -0.01, 0.02, 0.00, 0.03])
    y = x[::-1].copy()                                                 # same mean and volatility, different order
    w = {"carry": 0.5, "option": 0.5}
    dv = list(range(12))
    assert P.evaluate(((2000, 1), (2000, 12)), x, y, w, dv, ())["bundle_sharpe_above_both"]
    assert not P.evaluate(((2000, 1), (2000, 12)), x, x, w, dv, ())["bundle_sharpe_above_both"]      # identical sleeves: equal
    assert not P.evaluate(((2000, 1), (2000, 12)), x, -x, w, dv, ())["bundle_sharpe_above_both"]


# ---------------------------------------------------------------- the two runs on made-up inputs
@test
def dev_run_fixes_the_weights_and_reads_no_sealed_file():
    with fake(test_results=False) as F:
        rc, out = dev(F)
        assert rc == 0, out
        S = load(F, "portfolio19_dev.json")
        ms = G.months(*W_DEV)
        c = P.carry_from_csv(os.path.join(F["res"], "dev19_monthly.csv"), "CS40-M", W_DEV)
        o, n = P.option_monthly(F["opt"]["dev"], W_DEV)
        sc, so = np.std(c, ddof=1), np.std(o, ddof=1)
        assert close(S["weights"]["carry"], (1 / sc) / (1 / sc + 1 / so)) and close(S["weights"]["carry"] + S["weights"]["option"], 1)
        assert S["result"]["months"] == [G.lab(m) for m in ms] and S["inputs"]["sealed_files"] == 0 and S["inputs"]["raw_files"] == 0
        assert S["inputs"]["data_files"] == 24 and S["option_sha256"] == F["sha"] and all(ok for _, ok in S["checks"])
        assert list(S["result"]["crisis"]) == ["2011-08"] and os.path.isfile(os.path.join(F["res"], "portfolio19_dev_report.md"))


@test
def dev_run_does_not_overwrite():
    with fake(test_results=False) as F:
        assert dev(F)[0] == 0
        before = G.sha(os.path.join(F["res"], "portfolio19_dev.json"))
        rc, out = dev(F)
        assert rc == 1 and "already exist" in out and G.sha(os.path.join(F["res"], "portfolio19_dev.json")) == before


@test
def dev_run_stops_on_another_option_file():
    with fake(test_results=False) as F:
        with open(F["opt"]["sealed"], "a") as f:
            f.write("20151201,1.0,1.0,1.0\n")
        rc, out = dev(F)
        assert rc == 1 and "not the saved file" in out and not os.path.exists(os.path.join(F["res"], "portfolio19_dev.json"))


@test
def dev_run_stops_when_a_month_cannot_be_made():
    with fake(test_results=False) as F:
        option_file(F["opt"]["dev"], "2010-03-01", "2012-12-31", 1)        # no February 2010
        F["sha"]["dev"] = G.sha(F["opt"]["dev"])
        rc, out = dev(F)
        assert rc == 1 and "cannot be made" in out and not os.path.exists(os.path.join(F["res"], "portfolio19_dev.json"))


@test
def sealed_run_is_locked_until_the_test_results_exist():
    with fake(test_results=False) as F:
        assert dev(F)[0] == 0

        def boom(*a, **k):
            raise AssertionError("no data may be read before the test results exist")

        with patched(G, load=boom), patched(P, option_monthly=boom, carry_from_test=boom):
            rc, out = sealed(F)
        assert rc == 1 and "has not been run" in out
        assert not any(f.startswith("portfolio19_sealed") for f in os.listdir(F["res"]))
        with open(os.path.join(F["res"], "test19_summary.json"), "w") as f:      # one of the two files is not enough
            json.dump({"mode": "sealed", "astar": "CS40-M"}, f)
        with patched(G, load=boom):
            assert sealed(F)[0] == 1


@test
def sealed_run_end_to_end():
    with fake() as F:
        assert dev(F)[0] == 0
        rc, out = sealed(F)
        assert rc == 0, out
        D, S = load(F, "portfolio19_dev.json"), load(F, "portfolio19_sealed.json")
        T = load(F, "test19_summary.json")
        ms = G.months(*W_SEALED)
        c = np.array([dict(zip(T["months"], T["monthly_net"]["CS40-M"]))[G.lab(m)] for m in ms])
        o = np.array(P.option_monthly(F["opt"]["sealed"], W_SEALED)[0])
        w = D["weights"]
        assert S["weights"] == w and np.array_equal(np.array(S["result"]["bundle"]), w["carry"] * c + w["option"] * o)
        assert S["result"]["months"][0] == "2013-01" and S["result"]["months"][-1] == "2015-11" and len(S["result"]["months"]) == 35
        assert S["inputs"]["sealed_files"] == 21 and S["inputs"]["raw_files"] == 0 and list(S["result"]["crisis"]) == ["2013-06", "2014-10"]
        e = S["result"]
        assert S["sentences"][0] == P.SENTENCES["sharpe"][e["bundle_sharpe_above_both"]]
        assert S["sentences"][1] == P.SENTENCES["jump"][e["vix_jump"]["corr"] >= 0.5] and e["vix_jump"]["n"] == 4
        assert [x["outcome"] for x in S["predictions"]] == [e["bundle_sharpe_above_both"], e["vix_jump"]["corr"] >= 0.5]
        assert all(ok for _, ok in S["checks"]) and "미리 정한 문장" in out
        rc, out = sealed(F)
        assert rc == 1 and "already exist" in out


@test
def sealed_run_stops_on_changed_weights_or_another_test_file():
    with fake() as F:
        assert dev(F)[0] == 0
        p = os.path.join(F["res"], "portfolio19_dev.json")
        D = load(F, "portfolio19_dev.json")
        D["weights"]["carry"] += 0.01
        with open(p, "w", encoding="utf-8") as f:
            json.dump(D, f)
        rc, out = sealed(F)
        assert rc == 1 and "weights" in out and not os.path.exists(os.path.join(F["res"], "portfolio19_sealed.json"))
    with fake() as F:
        assert dev(F)[0] == 0
        T = load(F, "test19_summary.json")
        T["mode"] = "rehearsal"
        with open(os.path.join(F["res"], "test19_summary.json"), "w", encoding="utf-8") as f:
            json.dump(T, f)
        rc, out = sealed(F)
        assert rc == 1 and "not the sealed run" in out


@test
def sealed_run_stops_on_a_changed_option_file():
    with fake() as F:
        assert dev(F)[0] == 0
        with open(F["opt"]["sealed"], "a") as f:
            f.write("20151201,1.0,1.0,1.0\n")
        rc, out = sealed(F)
        assert rc == 1 and "differs from the one hashed" in out


# ---------------------------------------------------------------- constants against the document
@test
def constants_match_portfolio19_md():
    with open(os.path.join(HERE, "PORTFOLIO19.md"), encoding="utf-8") as f:
        doc = f.read()
    for s in list(P.SENTENCES["sharpe"].values()) + list(P.SENTENCES["jump"].values()) + [x for x, _ in P.PRED]:
        assert s in doc, s
    assert "| 봉인 구간에서 묶음 샤프가 두 칸 모두보다 높음 | 확률 0.35 |" in doc and "| VIX 급등 달 상관이 0.5 이상 | 확률 0.5 |" in doc
    assert len(G.months(*P.WINDOW["dev"])) == 202 and len(G.months(*P.WINDOW["sealed"])) == 152
    assert P.CRISIS["sealed"] == ((2015, 8), (2018, 2), (2020, 3), (2024, 8)) and P.OPT_COL == "O2_base" and P.OPT_N0 == 1e6
    assert P.OPT["dev"].endswith(os.path.join("study13_14_option_seller", "results", "dev_daily.csv"))
    assert P.OPT["sealed"].endswith(os.path.join("study13_14_option_seller", "study14", "results14", "holdout_daily14.csv"))
    assert P.TEST_FILES == ("test19_summary.json", "test19_report.md") and P.JUMP_PCT == 90


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
    with open(os.path.join(HERE, "results", "portfolio19_tests.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"{n}/{len(res)} passed")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
