"""M16 candidate selection, M17 grid size and N, M18 mix step (SPEC 5, 6, 15; E1-32..E1-35, E1-49, E1-59)."""
import dataclasses
import itertools
import math

import numpy as np
import pandas as pd
import pytest

import grid
import select_candidate as sc
from config import A_NAIVE, B0, B_ETF, N_PRIOR

NAN = float("nan")


def m(sr, y19=1.0, y20=1.0, q50=1.0, fills=5.0, part=0.001):
    return {"sharpe": sr, "sharpe_2019": y19, "sharpe_2020": y20, "sharpe_Q50": q50, "fills_per_symday": fills,
            "participation": part}


def stage1_metrics(base=1.0, **over):
    out = {c: m(base) for _, c in grid.stage1()}
    for name, val in over.items():
        out[dict(grid.stage1())[name]] = val
    return out


def names():
    return {c: n for n, c in grid.stage1()}


def test_filters_each_fail_alone_and_selection_order():
    cfgs = dict(grid.stage1())
    mets = stage1_metrics(0.5, **{"reset1": m(3.0, y19=-0.1), "reset15": m(2.9, q50=-0.2), "reset60": m(2.8, fills=0.9),
                                  "event": m(2.7, part=0.02), "L=1": m(2.6)})
    mets[cfgs["B0"]] = m(0.5)
    mets[dataclasses.replace(B0, L=5)] = m(-0.1)                     # a neighbour (L axis) of B0 below 0
    best, table = sc.select(mets, names())
    t = {r["name"]: r for r in table}
    assert not t["reset1"]["i_years"] and not t["reset15"]["ii_Q50"] and not t["reset60"]["iii_activity"]
    assert not t["event"]["iii_activity"]
    assert best == cfgs["L=1"]                                       # highest passing
    assert not t["B0"]["iv_neighbours"]                              # L=5 neighbour < 0 -> B0 fails (iv)
    assert t["event"]["neighbors_checked"]["note"].startswith("event")


def test_ties_fewer_dimensions_then_longer_delta():
    cfgs = dict(grid.stage1())
    c_all = grid.set_dim(grid.set_dim(B0, "obi", True), "L", 5)
    mets = {cfgs["B0"]: m(1.0), cfgs["obi=True"]: m(2.0), c_all: m(2.0)}
    assert sc.select(mets, {**names(), c_all: "C_all"})[0] == cfgs["obi=True"]             # fewer dims differ from B0
    mets = {cfgs["reset15"]: m(2.0), cfgs["reset60"]: m(2.0), cfgs["event"]: m(2.0)}
    assert sc.select(mets, names())[0] == cfgs["reset60"]                                  # longer delta (event = 1)


def test_no_pass_returns_none_and_nan_never_selected():
    mets = {c: m(NAN) for _, c in grid.stage1()}
    assert sc.select(mets, names())[0] is None
    mets = {c: m(-1.0, y19=-1, y20=-1) for _, c in grid.stage1()}
    assert sc.select(mets, names())[0] is None


def test_adoption_threshold_and_stage2_generation():
    cfgs = dict(grid.stage1())
    sh = {c: 1.0 for c in cfgs.values()}
    sh[cfgs["obi=True"]] = 1.5                                       # exactly +0.5: adopted
    sh[cfgs["fi=True"]] = 1.4999                                     # below: not adopted
    sh[cfgs["reset15"]], sh[cfgs["reset60"]] = 1.7, 1.7              # tie: first listed (reset15)
    sh[cfgs["halt=True"]] = NAN
    ad = grid.adopted(sh)
    assert ad == {"obi": True, "requote": ("reset", 15)}
    s2 = grid.stage2(ad, set(cfgs.values()))
    assert [n for n, _ in s2] == ["C_all"]                          # two adopted: both reverts are stage-1 settings
    c_all = s2[0][1]
    assert (c_all.obi, c_all.delta) == (True, 15)
    assert grid.stage2({}, set()) == []


def test_grid_size_and_n_over_every_adoption_pattern():
    cfgs = dict(grid.stage1())
    alts = {d: v for d, v in ((d, vals[0]) for d, vals in grid.DIMENSIONS.items())}
    assert len(cfgs) == 19
    for k in range(len(alts) + 1):
        for dims in itertools.combinations(alts, k):
            s2 = grid.stage2({d: alts[d] for d in dims}, set(cfgs.values()))
            stock = list(cfgs.values()) + [c for _, c in s2]
            assert len(set(stock)) <= grid.MAX_STOCK and len(s2) <= 11
            etf = [c for _, c in grid.etf_controls(stock[-1])]
            assert len(etf) <= 3
            n = grid.n_trials(stock, etf)
            assert n == N_PRIOR + len(set(stock)) + 1 + len(set(etf)) + 1 <= 69
    assert len(grid.etf_controls(B0)) == 2 and len(grid.etf_controls(B_ETF)) == 2


def test_mix_threshold_and_filters():
    stock = m(1.0)
    assert sc.mix_decision(stock, m(1.5))["mix_wins"]                # exactly +0.5
    assert not sc.mix_decision(stock, m(1.4999))["mix_wins"]
    assert not sc.mix_decision(stock, m(2.0, y20=-0.1))["mix_wins"]
    assert not sc.mix_decision(stock, m(2.0, part=0.011))["mix_wins"]


def test_mix_daily_sum_by_day():
    days = ["20190102", "20190103", "20190104"]
    rows = pd.DataFrame({"setting": ["S1", "S1", "E1"], "model": "Q", "date": ["20190102", "20190104", "20190103"],
                         "cash": [10_000, -20_000, 5_000], "sh_mkt_buy": 0, "sh_mkt_sell": 0, "sh_lim_buy": 0,
                         "sh_lim_sell": 0})
    stock = sc.daily(rows, "S1", "Q", "R", days)
    etf = sc.daily(rows, "E1", "Q", "R", days)
    assert stock.tolist() == [1.0, 0.0, -2.0] and (stock + etf).tolist() == [1.0, 0.5, -2.0]
