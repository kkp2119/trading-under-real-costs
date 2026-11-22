"""M5 accounting (SPEC 8): cash = sum of edges + inventory P&L exactly (integer, so well inside 1e-9), scenario
arithmetic, break-even rebate (erratum E0-5 sign), symbol sum = portfolio."""
import numpy as np

import accounting
import features
import simulator
from config import MODELS
from synth import random_day, settings


def test_identity_on_random_runs():
    n = 0
    for seed in range(10):
        b = random_day(2000 + seed, p_trade=0.3, p_jump=0.02)
        f = features.compute(b)
        for name, cfg, _ in settings():
            for m in MODELS:
                fills, _, _ = simulator.simulate(b, f, cfg.kernel_params(m, 23400, True))
                row = accounting.account(fills, f["mid2"], f["mid_ok"], 23400, *cfg.d_bounds(23400))
                assert 2 * row["cash"] == row["edge2_lim"] + row["edge2_mkt"] + row["inv2"], (seed, name, m)
                n += 1
    assert n == 1000


def test_scenarios_by_hand():
    #        u   side kind qty  price    ref_mid2   dec
    fills = np.array([[901, 1, 0, 300, 500_000, 1_000_300, 899],
                      [905, -1, 0, 100, 500_300, 1_000_300, 904],
                      [950, -1, 1, 200, 499_900, 1_000_100, 949]], dtype=np.int64)
    mid2 = np.full(23400, 1_000_300, dtype=np.int64)
    row = accounting.account(fills, mid2, np.ones(23400, bool), 23400, 899, 23099)
    cash = -300 * 500_000 + 100 * 500_300 + 200 * 499_900
    assert row["cash"] == cash
    assert accounting.net(row, "R") == cash - 30 * 200
    assert accounting.net(row, "R+") == cash - 30 * 200 + 20 * 400
    assert accounting.net(row, "C") == cash - 100 * 200 + 50 * 400
    assert row["edge2_lim"] == 300 * 300 + 100 * 300 and row["edge2_mkt"] == -200 * (1_000_100 - 999_800)
    be = accounting.breakeven_rebate(accounting.net(row, "R"), 400)
    assert be == -(cash - 6000) / 1e4 / 400 and (be <= 0) == (accounting.net(row, "R") >= 0)


def test_symbol_sum_is_portfolio():
    rows = []
    for seed in range(3):
        b = random_day(2100 + seed)
        f = features.compute(b)
        fills, _, _ = simulator.simulate(b, f, settings()[0][1].kernel_params("Q", 23400, True))
        rows.append(accounting.account(fills, f["mid2"], f["mid_ok"], 23400, 899, 23099))
    tot = {k: sum(r[k] for r in rows) for k in accounting.ROW_KEYS}
    assert accounting.net(tot, "R") == sum(accounting.net(r, "R") for r in rows)
