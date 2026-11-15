"""Research 19B: synthetic tests of run_test19b on a made-up data folder (1978-2091 style files that end in 1991).

No real data file and no real result file is read: engine19.DATA / CAND / HERE point into a temporary folder and the
'stored values of research 19' are computed there with engine19.load, the loader research 19 used.
Writes results/run_test19b_tests.json.
"""
import contextlib
import datetime as dt
import io
import json
import os
import sys
import tempfile
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine19 as G  # noqa: E402
import run_test19b as R  # noqa: E402
from test_engine19 import bdays  # noqa: E402

TESTS = []
REAL = (G.DATA, G.CAND, G.HERE)
A, B = "1978-10-02", "1991-10-31"
HOLIDAYS = tuple(f"{y}-01-01" for y in range(1979, 1992)) + tuple(f"{y}-12-25" for y in range(1978, 1991))
RATE_FROM = {"AU": (1978, 9), "CA": (1978, 9), "DE": (1978, 9), "GB": (1978, 9), "NZ": (1978, 9), "US": (1978, 9),
             "NO": (1979, 1), "SE": (1982, 1), "CH": None, "EZ": None, "JP": None}       # like the real files before 1990
CFG = {"w_c": ((1979, 2), (1991, 8)), "decades": (("1979–89", (1979, 2), (1989, 12)), ("1990–91", (1990, 1), (1991, 8))),
       "reproduce": {"dev": ((1990, 1), (1990, 12)), "sealed": ((1991, 1), (1991, 8))},
       "reh_dev": ((1990, 1), (1990, 12)), "reh_full": ((1990, 1), (1991, 8))}


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


def make_tree(root, seed=19):
    """Fake FRED-style files split at 2013-01-01 (everything is before it), candidates/build and the manifest folder."""
    rng = np.random.default_rng(seed)
    days = bdays(A, B, holidays=HOLIDAYS)
    ms = G.months(G.ym(A), G.ym(B))
    data, cand, here = os.path.join(root, "data"), os.path.join(root, "data", "candidates"), os.path.join(root, "here")
    for d in (os.path.join(data, "build"), os.path.join(data, "sealed"), os.path.join(cand, "build"), here):
        os.makedirs(d)
    man = {"SEALED_MANIFEST19.tsv": [], "SEALED_MANIFEST19_h10.tsv": [], "SEALED_MANIFEST19B_h10.tsv": []}

    def put(base, name, rows, which):
        for part in ("build", "sealed"):
            if base == cand and part == "sealed":
                continue
            keep = rows if part == "build" else []
            path = os.path.join(base, part, f"{name}.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"observation_date,{name}\n" + "".join(f"{d},{v}\n" for d, v in keep))
            man[which].append(f"{name}\t{part}\t{part}/{name}.csv\t{len(keep)}\t\t\t{G.sha(path)}\n")

    for c, (s, usd) in G.FX.items():
        lvl = (0.5 + rng.random()) * np.exp(np.cumsum(rng.normal(0, 0.006, len(days))))
        rows = [(d, f"{(v if usd else 1 / v):.4f}") for d, v in zip(days, lvl)]
        if c == "EUR":
            rows = []                                                      # DEXUSEU starts in 1999
        if c == "NZD":
            rows = [(d, "" if d == "1985-03-01" else v) for d, v in rows]  # no quote on a month's first business day
        if c == "SEK":
            rows = [(d, "." if d == "1984-06-13" else v) for d, v in rows]
        put(data, s, rows, "SEALED_MANIFEST19.tsv")
    for k, v in G.RATE.items():
        first = RATE_FROM[v]
        rows = [(f"{m[0]}-{m[1]:02d}-01", f"{rng.uniform(1, 15):.4f}") for m in ms if first is not None and m >= first]
        put(data, f"IR3TIB01{v}M156N", rows, "SEALED_MANIFEST19.tsv")
    put(data, "IR3TIB01DEM156N", [(f"{m[0]}-{m[1]:02d}-01", f"{rng.uniform(1, 15):.4f}") for m in ms], "SEALED_MANIFEST19.tsv")
    put(data, "VIXCLS", [(d, f"{12 + 25 * rng.random():.2f}") for d in days if d >= "1990-01-02"], "SEALED_MANIFEST19.tsv")
    put(data, "BAA10Y", [(d, f"{2 + rng.random():.2f}") for d in days], "SEALED_MANIFEST19.tsv")
    dem = [(d, f"{v:.4f}") for d, v in zip(days, 1.5 + rng.random() + np.cumsum(rng.normal(0, 0.004, len(days))))]
    put(cand, R.DEM_OLD, [x for x in dem if x[0] <= "1989-12-31"], "SEALED_MANIFEST19B_h10.tsv")
    put(cand, G.DEM_SERIES, [x for x in dem if x[0] >= "1990-01-01"], "SEALED_MANIFEST19_h10.tsv")
    put(cand, G.JP_CD_SERIES, [(f"{m[0]}-{m[1]:02d}-01", f"{rng.uniform(1, 15):.4f}") for m in ms if m >= (1979, 5)], "SEALED_MANIFEST19_h10.tsv")
    last = {}
    for d, v in dem:
        last[G.ym(d)] = float(v)                                          # month-end DM per USD
    put(cand, R.OECD_DEM, [(f"{m[0]}-{m[1]:02d}-01", f"{1.95583 / last[m] * (1.002 if m[1] % 2 else 0.998)!r}") for m in ms],
        "SEALED_MANIFEST19_h10.tsv")                                      # USD per mark in euro units, 0.2% off either way
    for f, lines in man.items():
        with open(os.path.join(here, f), "w", encoding="utf-8") as fh:
            fh.write("series\tpart\tpath\trows\tfirst_date\tlast_date\tsha256\n" + "".join(lines))
    return here, data, cand


@contextlib.contextmanager
def fake(stored=True):
    """The fake tree patched into engine19, plus a results folder with the 'research 19' C0 values (engine19.load path)."""
    with tempfile.TemporaryDirectory() as root:
        here, data, cand = make_tree(root)
        res = os.path.join(root, "results")
        os.makedirs(res)
        with patched(G, DATA=data, CAND=cand, HERE=here):
            assert not any(p in REAL for p in (G.DATA, G.CAND, G.HERE))
            F = {"root": root, "res": res, "data": data, "cand": cand, "here": here,
                 "dev": os.path.join(res, "dev19_monthly.csv"), "test": os.path.join(res, "test19_summary.json")}
            if stored:
                D, _ = G.load(("build", "sealed"))                         # research 19's loader: calendar from 1989-12-01
                ms, g, c = G.monthly(D, G.run_all(D, names=("C0",))["C0"], ((1990, 1), (1991, 8)))
                net = {G.lab(m): float(x) for m, x in zip(ms, g - c)}
                F["net19"] = net
                with open(F["dev"], "w", encoding="utf-8") as f:
                    f.write("month,C0,V70-M\n" + "".join(f"{k},{v!r},0.0\n" for k, v in net.items() if k <= "1990-12"))
                with open(F["test"], "w", encoding="utf-8") as f:
                    k = [x for x in net if x >= "1991-01"]
                    json.dump({"months": k, "monthly_net": {"C0": [net[x] for x in k]}}, f)
            yield F


def sealed(F, **cfg):
    return quiet(R.run_sealed, dict(CFG, **cfg), F["res"], F["dev"], F["test"])


def rehearsal(F, **cfg):
    return quiet(R.run_rehearsal, dict(CFG, **cfg), F["res"], F["dev"], F["test"])


def load_json(F, name):
    with open(os.path.join(F["res"], name), encoding="utf-8") as f:
        return json.load(f)


def nothing_written(F):
    return not any(f.startswith(("test19b", "rehearsal19b")) for f in os.listdir(F["res"]))


def long_data():
    return R.load_b(("build", "sealed"), R.CAL_START)[0]


# ---------------------------------------------------------------- constants and Holm
@test
def constants_match_spec19b():
    with open(os.path.join(HERE, "SPEC19B.md"), encoding="utf-8") as f:
        doc = f.read()
    for s in [x for d in R.SENTENCES.values() for x in d.values()] + [x for x, _ in R.PRED]:
        assert s in doc, s
    assert "| 1979–89 샤프가 0보다 큼 | 확률 0.75 |" in doc and "| H1b 확인 | 확률 0.2 |" in doc
    assert "| 전체 기간 샤프가 0.15 ~ 0.40 사이 | 확률 0.6 |" in doc and "| H1c 확인 | 확률 0.3 |" in doc
    assert len(G.months(*R.W_B)) == 131 and len(G.months(*R.W_C)) == 565 and R.W_B[0] == G.nxt(R.START_FORMATION) == R.W_C[0]
    assert R.PHRASE == "연구19B 테스트를 한 번 연다" and R.BOOT_SEED == 192 and G.BOOT_B == 10_000 and G.BOOT_BLOCK == 12
    assert R.PRE_COST == 0.0008 and G.COST == 0.0004 and R.PRE_ROBUST == (0.0004, 0.0016) and R.SWITCH == (1990, 1)
    assert R.CAL_START == "1979-01-01" and R.REH_START == G.START == "1989-12-01" and R.TRIALS_BEFORE == 775
    assert [d[0] for d in R.DECADES] == ["1979–89", "1990–99", "2000–09", "2010–19", "2020–26"]
    assert sum(len(G.months(a, b)) for _, a, b in R.DECADES) == 565 and R.FLOAT_FROM == {"AUD": (1984, 1), "NZD": (1985, 4)}
    assert R.REPRODUCE == {"dev": ((1990, 1), (2012, 11)), "sealed": ((2013, 1), (2026, 2))}


@test
def holm_levels_are_0_0125_and_0_025():
    r = G.holm({"H1b": 0.012, "H1c": 0.02}, R.ALPHA)
    assert r["H1b"] == {"p": 0.012, "level": 0.0125, "rejected": True} and r["H1c"] == {"p": 0.02, "level": 0.025, "rejected": True}
    r = G.holm({"H1b": 0.013, "H1c": 0.001}, R.ALPHA)
    assert r["H1c"]["level"] == 0.0125 and r["H1c"]["rejected"] and r["H1b"]["level"] == 0.025 and r["H1b"]["rejected"]
    r = G.holm({"H1b": 0.02, "H1c": 0.03}, R.ALPHA)
    assert not r["H1b"]["rejected"] and not r["H1c"]["rejected"] and r["H1b"]["level"] == 0.0125      # stops at the first failure
    r = G.holm({"H1b": 0.3, "H1c": 0.0125}, R.ALPHA)
    assert not r["H1c"]["rejected"] and not r["H1b"]["rejected"]                                      # p must be below the level
    r = G.holm({"H1b": 0.03, "H1c": 0.001}, R.ALPHA)
    assert r["H1c"]["rejected"] and not r["H1b"]["rejected"]


# ---------------------------------------------------------------- universe and ranking before 1990
@test
def universe_grows_from_six_to_eight():
    with fake(stored=False):
        D = long_data()
        n = {m: R.available(D, m) for m in G.months((1979, 1), (1989, 12))}
        assert n[(1979, 1)] == []                                              # its formation month is before the calendar
        assert n[(1979, 2)] == ["AUD", "CAD", "EUR", "GBP", "NZD", "NOK"] and len(n[(1979, 5)]) == 6
        assert n[(1979, 6)] == ["AUD", "CAD", "EUR", "GBP", "JPY", "NZD", "NOK"]   # the CD rate starts 1979-05
        assert len(n[(1982, 1)]) == 7 and n[(1982, 2)] == ["AUD", "CAD", "EUR", "GBP", "JPY", "NZD", "NOK", "SEK"]
        assert "NZD" not in n[(1985, 2)] and "NZD" not in n[(1985, 3)] and "NZD" in n[(1985, 4)]   # no quote on 1985-03-01
        assert all("CHF" not in v for v in n.values()) and all(len(v) >= 6 for m, v in n.items() if m >= (1979, 2))
        prob, desc = R.plan(D, R.W_B)
        assert prob == [] and desc["months"] == 131 and desc["months_with_fewer_than_six"] == [] and desc["rate_fills_used"] == []
        assert desc["currencies_by_year"]["1979"] == "6–7" and desc["currencies_by_year"]["1985"] == "7–8" and desc["currencies_by_year"]["1989"] == "8"
        assert desc["days_with_a_stale_quote"] == 2 and desc["first_day"] == "1979-01-02"


def _hand_legs(F, D, m):
    """Top and bottom three of the usable currencies by the rate files, ranked here without the engine."""
    t = G.prv(m)
    key = f"{t[0]}-{t[1]:02d}-01"

    def rate(code, base=None):
        rows = dict(G.read_rows(os.path.join(base or F["data"], "build", f"{code}.csv")))
        return Fraction(rows[key])

    name = {"AUD": "IR3TIB01AUM156N", "CAD": "IR3TIB01CAM156N", "EUR": "IR3TIB01DEM156N", "GBP": "IR3TIB01GBM156N", "NZD": "IR3TIB01NZM156N",
            "NOK": "IR3TIB01NOM156N", "SEK": "IR3TIB01SEM156N"}
    us = rate("IR3TIB01USM156N")
    diff = {c: (rate(G.JP_CD_SERIES, F["cand"]) if c == "JPY" else rate(name[c])) - us for c in R.available(D, m)}
    order = sorted(diff, key=lambda c: (-diff[c], G.CCY.index(c)))
    return set(order[:3]), set(order[-3:]), {c: float(diff[c]) for c in order}


@test
def ranking_with_six_seven_and_eight_currencies():
    with fake(stored=False) as F:
        D = long_data()
        for m, n in (((1979, 3), 6), ((1980, 7), 7), ((1985, 3), 7), ((1987, 5), 8)):
            longs, shorts, diff = _hand_legs(F, D, m)
            assert len(diff) == n, (m, len(diff))
            idx, w, dr = G.portfolio(D, m)
            got_long = {G.CCY[j] for j, x in zip(idx, w) if x > 0}
            got_short = {G.CCY[j] for j, x in zip(idx, w) if x < 0}
            assert got_long == longs and got_short == shorts and len(idx) == 6, m
            assert all(close(abs(x), 1 / 3) for x in w) and all(close(d, diff[G.CCY[j]], 1e-9) for j, d in zip(idx, dr))
            assert not (got_long | got_short) & {"CHF"} and len(set(diff) - got_long - got_short) == n - 6


@test
def fewer_than_six_in_the_floating_only_universe_is_two_and_two():
    with fake(stored=False):
        D = long_data()
        D2 = R.without_pegged(D)
        assert "AUD" in R.available(D, (1983, 12)) and "AUD" not in R.available(D2, (1983, 12)) and "AUD" in R.available(D2, (1984, 1))
        assert "NZD" not in R.available(D2, (1985, 3)) and "NZD" in R.available(D2, (1985, 4))
        assert R.available(D2, (1979, 2)) == ["CAD", "EUR", "GBP", "NOK"] and R.available(D2, (1986, 1)) == R.available(D, (1986, 1))
        idx, w, _ = G.portfolio(D2, (1979, 2))
        assert len(idx) == 4 and sorted(w.tolist()) == [-0.5, -0.5, 0.5, 0.5]
        assert D2.rate["GBP"] is D.rate["GBP"] and len(D.rate["AUD"]) > len(D2.rate["AUD"]) and D2.S is D.S     # the original is untouched


# ---------------------------------------------------------------- cost switch and the calendar start
@test
def cost_is_8bp_before_1990_and_4bp_from_1990_01():
    with fake(stored=False):
        D = long_data()
        sims = R.simulate(D, (G.COST, R.PRE_COST) + R.PRE_ROBUST)
        ms, net, g, c = R.series(D, sims, ((1979, 2), (1991, 8)))
        by = dict(zip(ms, c))
        assert close(by[(1989, 12)], 0.0016, 1e-18) and close(by[(1990, 1)], 0.0008, 1e-18)
        assert all(close(by[m], 0.0016, 1e-18) for m in ms if m < (1990, 1)) and all(close(by[m], 0.0008, 1e-18) for m in ms if m >= (1990, 1))
        assert np.array_equal(net, g - c)
        for pre, want in ((0.0004, 0.0008), (0.0016, 0.0032)):
            ms2, n2, g2, c2 = R.series(D, sims, ((1979, 2), (1991, 8)), pre)
            assert np.array_equal(g2, g) and close(dict(zip(ms2, c2))[(1989, 12)], want, 1e-18) and close(dict(zip(ms2, c2))[(1990, 1)], 0.0008, 1e-18)


@test
def the_longer_calendar_does_not_change_1990_on():
    with fake(stored=False):
        long_, short = long_data(), R.load_b(("build", "sealed"), R.REH_START)[0]
        assert long_.days[0] == "1979-01-02" and short.days[0] == "1989-12-01" and G.portfolio(short, (1989, 12)) is None
        w = ((1990, 1), (1991, 8))
        a = R.series(long_, R.simulate(long_, (G.COST, R.PRE_COST)), w)
        b = R.series(short, R.simulate(short, (G.COST, R.PRE_COST)), w)
        assert a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]) and len(a[0]) == 20
        assert G.portfolio(long_, (1989, 12)) is not None and G.portfolio(long_, (1979, 1)) is None


@test
def load_b_gives_the_same_c0_as_the_loader_of_research_19():
    with fake() as F:
        D19, _ = G.load(("build", "sealed"))
        Db, read = R.load_b(("build", "sealed"), R.REH_START)
        assert D19.days == Db.days and np.array_equal(D19.fresh[:, [0, 1, 2, 4, 5, 6, 7, 8]], Db.fresh[:, [0, 1, 2, 4, 5, 6, 7, 8]])
        k = D19.days.index("1990-01-02")
        assert np.array_equal(D19.S[k:], Db.S[k:]) and D19.rate == Db.rate and G.verify(read) == []
        ms, net, _, _ = R.series(Db, R.simulate(Db, (G.COST, R.PRE_COST)), ((1990, 1), (1991, 8)))
        assert [float(x) for x in net] == [F["net19"][G.lab(m)] for m in ms]
        assert len(read) == 42 and sum("VIXCLS" in p or "BAA10Y" in p for p in read) == 0     # 45 of research 19 - 4 + the old DM file


@test
def the_two_fed_files_join_at_1990():
    with fake(stored=False) as F:
        D = long_data()
        old = dict(G.read_rows(os.path.join(F["cand"], "build", f"{R.DEM_OLD}.csv")))
        new = dict(G.read_rows(os.path.join(F["cand"], "build", f"{G.DEM_SERIES}.csv")))
        j = G.CCY.index("EUR")
        a, b = D.days.index("1989-12-29"), D.days.index("1990-01-02")
        assert b == a + 1 and close(D.S[a, j], 1.95583 / float(old["1989-12-29"]), 1e-15) and close(D.S[b, j], 1.95583 / float(new["1990-01-02"]), 1e-15)
        assert D.fresh[a, j] and D.fresh[b, j] and max(old) < min(new)
        with open(os.path.join(F["cand"], "build", f"{G.DEM_SERIES}.csv"), "a") as f:
            f.write("1990-01-03,1.7\n")
        try:
            R.load_b(("build",), R.CAL_START)
            raise AssertionError("dates out of order must stop the load")
        except ValueError:
            pass


@test
def overlapping_fed_files_stop_the_load():
    with fake(stored=False) as F:
        p = os.path.join(F["cand"], "build", f"{R.DEM_OLD}.csv")
        with open(p, "a") as f:
            f.write("1990-01-02,1.7\n")
        try:
            R.load_b(("build",), R.CAL_START)
            raise AssertionError("overlap must stop the load")
        except ValueError as e:
            assert "overlap" in str(e)


# ---------------------------------------------------------------- the sealed path on made-up data
@test
def sealed_path_end_to_end():
    with fake() as F:
        rc, out = sealed(F)
        assert rc == 0, out
        S = load_json(F, "test19b_summary.json")
        assert S["mode"] == "sealed" and S["months"][0] == "1979-02" and S["months"][-1] == "1991-08" and len(S["months"]) == 151
        net = np.array(S["monthly_net"])
        assert np.allclose(net, np.array(S["monthly_gross"]) - np.array(S["monthly_cost"]), atol=1e-18, rtol=0)
        assert S["monthly_cost"][S["months"].index("1989-12")] == 0.0016 and S["monthly_cost"][S["months"].index("1990-01")] == 0.0008
        v = S["verdict"]
        assert S["holm_levels"] == [0.0125, 0.025] and v["H1b"]["months"] == 131 and v["H1c"]["months"] == 151
        for h, x in (("H1b", net[:131]), ("H1c", net)):
            d = G.boot_sharpe(x, G.boot_index(len(x), seed=192))
            assert close(v[h]["estimate"], G.sharpe(x)) and close(v[h]["p"], float(np.mean(d <= 0))) and v[h]["sentence"] == R.SENTENCES[h][v[h]["rejected"]]
            assert close(v[h]["bound_at_level"], float(np.percentile(d, 100 * v[h]["level"])))
        small = min(v, key=lambda h: (v[h]["p"], h))
        assert v[small]["level"] == 0.0125 and v[small]["rejected"] == (v[small]["p"] < 0.0125)
        assert S["reproduced"]["dev"] == {"months": 12, "max_abs_diff": 0.0, "equal": True} and S["reproduced"]["sealed"]["months"] == 8
        assert list(S["decades"]) == ["1979–89", "1990–91"] and S["decades"]["1979–89"]["months"] == 131 and S["decades"]["1990–91"]["months"] == 20
        assert close(S["decades"]["1979–89"]["net_sum"] + S["decades"]["1990–91"]["net_sum"], S["windows"]["H1c"]["net_sum"])
        assert close(S["windows"]["H1b"]["sharpe"], G.sharpe(net[:131])) and close(S["windows"]["H1c"]["max_drawdown"], G.max_drawdown(net))
        r = S["cost_robustness_before_1990"]
        assert list(r) == ["4bp", "16bp"] and close(r["4bp"]["H1b"]["net_sum"] - S["windows"]["H1b"]["net_sum"], 131 * 0.0008)
        assert close(S["windows"]["H1b"]["net_sum"] - r["16bp"]["H1b"]["net_sum"], 131 * 0.0016)
        f = S["floating_only"]
        assert f["first_holding_month"] == {"AUD": "1984-01", "NZD": "1985-04"} and f["months_with_two_and_two"] > 30 and f["months_without_a_portfolio"] == 0
        assert [x["item"] for x in S["predictions"]] == [x for x, _ in R.PRED] and S["trials"] == {"before": 775, "added": 2, "after": 777}
        assert S["predictions"][0]["outcome"] == (S["windows"]["H1b"]["sharpe"] > 0) and S["predictions"][1]["outcome"] == v["H1b"]["rejected"]
        assert S["plan"]["currencies_by_year"]["1979"] == "6–7" and S["read"]["files"] == 43 and len(S["yearly"]) == 13
        q = S["dem_quality"]
        assert q["months"] == 132 and close(q["median_abs_rel_diff"], 0.002, 1e-4) and q["max_abs_rel_diff"] < 0.0021 and "OECD 월말 값과 비교(132개월" in out
        assert v["H1b"]["sentence"] in out and v["H1c"]["sentence"] in out and "수준 0.0125, 0.025" in out
        assert os.path.isfile(os.path.join(F["res"], "test19b_report.md"))
        rc, out = sealed(F)
        assert rc == 1 and "already exist" in out


@test
def sealed_path_stops_when_1990_on_differs_from_research_19():
    with fake() as F:
        with open(F["test"], encoding="utf-8") as f:
            t = json.load(f)
        t["monthly_net"]["C0"][3] += 1e-15
        with open(F["test"], "w", encoding="utf-8") as f:
            json.dump(t, f)
        rc, out = sealed(F)
        assert rc == 1 and "differ from the stored values" in out and nothing_written(F)


@test
def sealed_path_stops_on_a_changed_data_file():
    with fake() as F:
        with open(os.path.join(F["cand"], "build", f"{R.DEM_OLD}.csv"), "a") as f:
            f.write("1989-12-30,1.7\n")
        rc, out = sealed(F)
        assert rc == 1 and "sha256 differs" in out and nothing_written(F)


@test
def sealed_path_stops_before_returns_when_a_month_has_no_portfolio():
    with fake() as F:
        p = os.path.join(F["data"], "build", "IR3TIB01USM156N.csv")
        rows = [ln for ln in open(p, encoding="utf-8") if not ln.startswith(("1983-04", "1983-05"))]     # two US months gone
        with open(p, "w", encoding="utf-8") as f:
            f.writelines(rows)
        man = os.path.join(F["here"], "SEALED_MANIFEST19.tsv")
        lines = open(man, encoding="utf-8").read().splitlines()
        lines = [ln.rsplit("\t", 1)[0] + "\t" + G.sha(p) if ln.startswith("IR3TIB01USM156N\tbuild") else ln for ln in lines]
        with open(man, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        def boom(*a, **k):
            raise AssertionError("no return may be computed")

        with patched(G, simulate=boom, portfolios=boom):
            rc, out = sealed(F)
        assert rc == 1 and "fewer than four currencies" in out and "1983-06" in out and nothing_written(F)


@test
def sealed_path_stops_when_the_window_is_not_covered():
    with fake() as F:
        rc, out = sealed(F, w_c=((1979, 2), (1992, 3)))
        assert rc == 1 and "stop before any return" in out and nothing_written(F)


# ---------------------------------------------------------------- the rehearsal
@test
def rehearsal_reproduces_and_computes_nothing_before_1990():
    with fake() as F:
        seen = []
        real = G.simulate

        def spy(D, *a, **k):
            seen.append(D.days[0])
            return real(D, *a, **k)

        with patched(G, simulate=spy):
            rc, out = rehearsal(F)
        assert rc == 0, out
        S = load_json(F, "rehearsal19b_summary.json")
        assert S["equal"] and S["no_portfolio_before_1990"] and S["calendar_start"] == "1989-12-01" and set(seen) == {"1989-12-01"}
        assert S["pass_build"]["dev"] == {"months": 12, "max_abs_diff": 0.0, "equal": True}
        assert S["pass_full"]["sealed"] == {"months": 8, "max_abs_diff": 0.0, "equal": True} and S["pass_full"]["dev"]["equal"]
        assert S["read"] == {"build": 23, "full": 42} and "차이 0으로 재현됐다: 예" in out and "dem_quality" not in S
        rc, out = rehearsal(F)
        assert rc == 1 and "already exist" in out


@test
def rehearsal_reports_a_difference():
    with fake() as F:
        rows = open(F["dev"], encoding="utf-8").read().splitlines()
        a = rows[5].split(",")
        rows[5] = ",".join([a[0], repr(float(a[1]) + 1e-12), a[2]])
        with open(F["dev"], "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        rc, out = rehearsal(F)
        S = load_json(F, "rehearsal19b_summary.json")
        assert rc == 1 and not S["equal"] and not S["pass_build"]["dev"]["equal"] and S["pass_build"]["dev"]["max_abs_diff"] > 0


# ---------------------------------------------------------------- guard
@contextlib.contextmanager
def guard_folder():
    with tempfile.TemporaryDirectory() as root:
        res = os.path.join(root, "results")
        os.makedirs(res)

        def put(rel, text):
            with open(os.path.join(root, rel), "w", encoding="utf-8") as f:
                f.write(text)

        for c in R.CORE:
            put(c, f"content of {c}\n")
        put("results/check19b_quality.json", json.dumps({"start_formation_month": "1979-01"}))
        put("SPEC19B.sha256", f"{G.sha(os.path.join(root, 'SPEC19B.md'))}  SPEC19B.md\n")
        put("FREEZE19B.md", "| 파일 | sha256 | 역할 |\n|---|---|---|\n" + "".join(f"| `{c}` | `{G.sha(os.path.join(root, c))}` | x |\n" for c in R.CORE))
        put("UNLOCK19B.txt", R.PHRASE + "\n")
        with patched(R, HERE=root, RES=res, UNLOCK=os.path.join(root, "UNLOCK19B.txt"), FREEZE=os.path.join(root, "FREEZE19B.md"),
                     CHECK=os.path.join(res, "check19b_quality.json")):
            yield root, put


@test
def dem_quality_by_hand():
    with fake(stored=False) as F:
        D = long_data()
        rows = G.series_rows(R.OECD_DEM, ("build",), F["cand"])
        q = R.dem_quality(D, rows)
        dem = {}
        for d, v in G.read_rows(os.path.join(F["cand"], "build", f"{R.DEM_OLD}.csv")):
            if v:
                dem[G.ym(d)] = float(v)
        oecd = {G.ym(d): float(v) for d, v in rows}
        diffs = sorted(abs(1.95583 / dem[m] / oecd[m] - 1) for m in G.months((1979, 1), (1989, 12)))
        assert q["months"] == 132 and close(q["median_abs_rel_diff"], float(np.median(diffs)), 1e-12) and close(q["max_abs_rel_diff"], diffs[-1], 1e-12)
        assert R.dem_quality(D, []) is None


@test
def guard_passes_only_with_everything_in_place():
    with guard_folder() as (root, put):
        assert R.guard() == [] and len(R.frozen_files()) == len(R.CORE)


@test
def guard_needs_the_exact_unlock_phrase():
    with guard_folder() as (root, put):
        put("UNLOCK19B.txt", "연구19 테스트를 한 번 연다\n")                # the phrase of research 19 does not open 19B
        assert R.guard() == ["no unlock phrase in UNLOCK19B.txt"]
        os.remove(os.path.join(root, "UNLOCK19B.txt"))
        assert R.guard() == ["no unlock phrase in UNLOCK19B.txt"]
        put("UNLOCK19B.txt", "  " + R.PHRASE + "  \n\n")
        assert R.guard() == []


@test
def guard_stops_when_results_exist():
    with guard_folder() as (root, put):
        put("results/test19b_report.md", "x")
        assert R.guard() == ["test results already exist"]


@test
def guard_stops_on_a_changed_frozen_file_or_spec():
    with guard_folder() as (root, put):
        put("engine19.py", "changed\n")
        assert R.guard() == ["frozen file differs: engine19.py"]
    with guard_folder() as (root, put):
        put("SPEC19B.md", "changed\n")
        assert R.guard() == ["frozen file differs: SPEC19B.md", "SPEC19B.md differs from SPEC19B.sha256"]
    with guard_folder() as (root, put):
        os.remove(os.path.join(root, "results", "test19_summary.json"))
        assert R.guard() == ["frozen file differs: results/test19_summary.json"]


@test
def guard_stops_on_a_missing_core_row_or_another_start_month():
    with guard_folder() as (root, put):
        text = open(os.path.join(root, "FREEZE19B.md"), encoding="utf-8").read()
        put("FREEZE19B.md", "\n".join(ln for ln in text.splitlines() if "run_test19b.py" not in ln) + "\n")
        assert R.guard() == ["FREEZE19B.md missing or without the core files"]
    with guard_folder() as (root, put):
        put("results/check19b_quality.json", json.dumps({"start_formation_month": "1979-05"}))
        why = R.guard()
        assert "the start month differs from results/check19b_quality.json" in why and "frozen file differs: results/check19b_quality.json" in why
    with guard_folder() as (root, put):
        os.remove(os.path.join(root, "FREEZE19B.md"))
        assert "FREEZE19B.md missing or without the core files" in R.guard()


@test
def go_does_not_touch_data_when_the_guard_fails():
    with guard_folder() as (root, put):
        os.remove(os.path.join(root, "UNLOCK19B.txt"))

        def boom(*a, **k):
            raise AssertionError("nothing may be read when the guard fails")

        with patched(R, run_sealed=boom, load_b=boom), patched(sys, argv=["run_test19b.py", "--go"]):
            rc, out = quiet(R.main)
        assert rc == 1 and "no unlock phrase" in out


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
    with open(os.path.join(HERE, "results", "run_test19b_tests.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(f"{n}/{len(res)} passed")
    return 0 if n == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
