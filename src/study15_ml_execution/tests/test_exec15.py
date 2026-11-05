"""Research 15-E tests (declaration 7): hand cases S0, S1, S2 (queue kept), S3 (waits, deadline market); the fill functions are
simulator15's and match xy15.fill_path (= research 12 replay) on real 2019 orders; parent orders are reproducible; fold A
predictions of 2019 come from models trained on 2015-2018. Run with python -m pytest."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import exec15 as E  # noqa: E402
import loader15 as L  # noqa: E402
import model15 as M15  # noqa: E402
import simulator15 as S  # noqa: E402
import xy15 as X  # noqa: E402


def day():
    b = {c: np.zeros(L.NSEC, dtype=np.int64) for c in L.COLS}
    b["bid"][:], b["ask"][:], b["bsz"][:], b["asz"][:] = 100_000, 100_200, 200, 300
    for c, v in (("bid_max", 100_000), ("bid_min", 100_000), ("ask_max", 100_200), ("ask_min", 100_200)):
        b[c][:] = v
    return b


T_, F_ = np.ones(L.NSEC, np.bool_), np.zeros(L.NSEC, np.bool_)
D1, S1_ = np.array([1]), np.array([299])                    # one buy parent starting at bar 299 (09:35:00)


def run(b, method, post=F_, rush=F_, improve=0, qty=1000, horizon=300):
    out = E.run_method(b, method, D1, S1_, horizon, qty, post, post, rush, rush, improve)
    return out[0], E.costs(out, D1, (b["bid"] + b["ask"])[S1_], qty)


def test_s0_immediate():
    """1,000 at the ask 10.02, arrival mid 10.01: IS = 1 cent + 0.3 fee = 1.3 (R); 700 shares above the 300 displayed."""
    out, (ir, irp, lim, over) = run(day(), 0)
    assert tuple(out) == (0, 0, 1000, 1000 * 100_200, 700) and np.isclose(ir[0], 1.3) and np.isclose(irp[0], 1.3)


def test_s1_twap():
    """10 x 100 at the ask every 30 s: IS 1.3, nothing above the displayed 300."""
    out, (ir, _, _, over) = run(day(), 1)
    assert tuple(out) == (0, 0, 1000, 1000 * 100_200, 0) and np.isclose(ir[0], 1.3)


def test_s2_keeps_the_queue():
    """Buy 1,000 at the bid 10.00 behind 200 shares (a4 800). Bar 301: 150 trade at the bid -> a4 200. The 304 decision keeps
    the order (same price). Bar 306: 1,000 trade -> fill (4000 - 200) // 4 = 950 at 10.00; the last 50 go at market (ask
    10.02) at the deadline 599. Average 10.0010, mid 10.01: IS = -0.9 + 0.3 x 0.05 = -0.885 (R); R+ -0.885 - 0.2 x 0.95."""
    b = day()
    b["vol"][301] = b["vol_bid"][301] = 150
    b["vol"][306] = b["vol_bid"][306] = 1000
    out, (ir, irp, lim, over) = run(b, 2, post=T_)
    assert tuple(out) == (950, 950 * 100_000, 50, 50 * 100_200, 0)
    assert np.isclose(ir[0], -0.885) and np.isclose(irp[0], -0.885 - 0.19) and np.isclose(lim[0], 0.95)


def test_s3_waits_then_deadline_market():
    """S3 never posts (prediction below theta at every decision): the whole 1,000 goes at market at the deadline bar 599,
    where the ask is 10.03: IS = 2 + 0.3 = 2.3 cents; 700 above the displayed size."""
    b = day()
    b["ask"][599] = b["ask_max"][599] = b["ask_min"][599] = 100_300
    out, (ir, _, lim, over) = run(b, 2, post=F_)
    assert tuple(out) == (0, 0, 1000, 1000 * 100_300, 700) and np.isclose(ir[0], 2.3) and lim[0] == 0
    rush = F_.copy()
    rush[304] = True                                        # rush at the second decision: market at 10.02
    out, (ir, _, _, _) = run(b, 2, post=F_, rush=rush)
    assert tuple(out) == (0, 0, 1000, 1000 * 100_200, 700) and np.isclose(ir[0], 1.3)


def test_fill_functions_are_simulator15s_and_match_fill_path():
    assert E.fill_buy is S.fill_buy and E.fill_sell is S.fill_sell
    man, cb = L.manifest(2019), L.close_bars(2019)
    for k in sorted(man)[:20]:
        b = L.load(2019, man[k])
        us = X.decisions(cb[k[1]])
        us = us[X.quote_ok(b)[us] & (b["ask"][us] - b["bid"][us] >= 100)][:500]
        for side in (1, -1):
            out = E.run_method(b, 2, np.full(len(us), side), us, 5, 100, T_, T_, F_, F_, 0)
            assert np.array_equal(out[:, 0], X.fill_path(b, us, side).sum(axis=1))      # limit shares in bars u+1 .. u+5


def test_parents_reproducible():
    d1, s1 = E.parents("20190315", 23_400)
    d2, s2 = E.parents("20190315", 23_400)
    assert np.array_equal(d1, d2) and np.array_equal(s1, s2) and (d1 == 1).sum() == 10 and (d1 == -1).sum() == 10
    assert len(set(s1)) == 20 and s1.min() >= 299 and s1.max() <= 23_034 and np.all((s1 - 4) % 5 == 0)
    _, sh = E.parents("20191129", 12_600)
    assert sh.max() <= 12_539 - 305
    assert not np.array_equal(E.parents("20190318", 23_400)[1], s1)


def test_fold_models_do_not_see_the_validation_year():
    for fold, val in (("A", 2019), ("B", 2020)):
        j = json.load(open(os.path.join(M15.MODEL_DIR, fold, "fold.json")))
        assert max(j["train_years"]) < val and j["es_year"] < val and j["val_year"] == val
        assert all(int(y) < val for y in j["lgb_sample"]["years"])
    t = E.tasks("A")
    assert {x[1] for x in t} == {2019} and {x[5] for x in t} == {"A"}
