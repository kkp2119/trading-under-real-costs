"""Research 15 step 3 tests (Kyle F): the copied B0 kernel with every side allowed equals research 12 B0 on 2019-01 (fills,
orders, counters, gates, daily net R and R+ against research12/results/dev_daily.csv), the 09:30-16:00 window grid, P1 with
theta -inf, the P3 taker kernel on hand cases, time buckets, and the leak rules of the folds. Run with python -m pytest.
Research 12 modules are imported read-only; only 2019 bars are read."""
import dataclasses
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R12 = os.path.join(os.path.dirname(HERE), "study12_market_maker")
sys.path.insert(0, HERE)
import accounting15  # noqa: E402
import config15  # noqa: E402
import features15  # noqa: E402
import loader15 as L  # noqa: E402
import model15 as M15  # noqa: E402
import policy15 as P  # noqa: E402
import session15  # noqa: E402
import simulator15 as S  # noqa: E402


def r12():
    sys.path.insert(0, R12)
    import accounting
    import config
    import features
    import session
    import simulator
    for m in (accounting, config, features, session, simulator):
        assert os.path.dirname(os.path.abspath(m.__file__)) == R12
    return accounting, config, features, session, simulator


def test_b0_equals_research12_on_2019_01():
    acc12, cfg12, feat12, ses12, sim12 = r12()
    man, cb = L.manifest(2019), L.close_bars(2019)
    g = M15.read_gates()
    g = g[g.year == 2019]
    gate = {(r.sym, r.date): bool(r.gate) for r in g.itertuples()}
    cal12 = ses12.Calendar()
    dv = {(r.sym, r.date): int(r.dv) for r in g.itertuples()}
    zero = {(r.sym, r.date) for r in g.itertuples() if r.no_data}
    avail = set(man)
    days = sorted({d for _, d in man if d.startswith("201901")})
    net = {d: [0, 0] for d in days}
    n_fill = 0
    for (s, d) in sorted(k for k in man if k[1] in net):
        assert gate[(s, d)] == ses12.gate(cal12, s, d, dv, zero, avail)          # research 12 gate, same inputs
        b = L.load(2019, man[(s, d)])
        assert features15.day_stats(b, cb[d])["dv"] == feat12.day_stats(b, cb[d])["dv"]
        f12, f15 = feat12.compute(b), features15.compute(b)
        assert all(np.array_equal(f12[k], f15[k]) for k in f12)
        p12 = cfg12.B0.kernel_params("Q", cb[d], gate[(s, d)])
        p15 = config15.B0.kernel_params("Q", cb[d], gate[(s, d)])
        assert np.array_equal(p12, p15)
        a = sim12.simulate(b, f12, p12)
        z = S.simulate(b, f15, p15)
        t = S.simulate(b, f15, p15, np.ones(len(b["bid"]), bool), np.ones(len(b["bid"]), bool))   # P1, theta = -inf
        for x, y, w in zip(a, z, t):
            assert np.array_equal(x, y) and np.array_equal(x, w)
        d0, df = config15.B0.d_bounds(cb[d])
        row = accounting15.account(z[0], f15["mid2"], f15["mid_ok"], cb[d], d0, df)
        assert row == acc12.account(a[0], f12["mid2"], f12["mid_ok"], cb[d], d0, df)
        net[d][0] += accounting15.net(row, "R")
        net[d][1] += accounting15.net(row, "R+")
        n_fill += len(z[0])
    dev = pd.read_csv(os.path.join(R12, "results", "dev_daily.csv"), dtype={"date": str}).set_index("date")
    for d in days:
        assert float(net[d][0]) / 1e4 == dev.loc[d, "B0|Q|R"] and float(net[d][1]) / 1e4 == dev.loc[d, "B0|Q|R+"], d
    print(f"2019-01: {len(days)} days, {sum(1 for k in man if k[1] in net)} symbol-days, {n_fill} fill rows, daily net R "
          f"and R+ equal to research12 dev_daily.csv")
    assert n_fill > 0


def flat_day():
    b = {c: np.zeros(L.NSEC, dtype=np.int64) for c in L.COLS}
    b["bid"][:], b["ask"][:], b["bsz"][:], b["asz"][:] = 100_000, 100_200, 200, 300
    for c, v in (("bid_max", 100_000), ("bid_min", 100_000), ("ask_max", 100_200), ("ask_min", 100_200)):
        b[c][:] = v
    return b


def test_window2_grid_and_flatten():
    b = flat_day()
    f = features15.compute(b)
    p = P.B0W2.kernel_params("Q", 23_400, True)
    assert P.B0W2.d_bounds(23_400) == (4, 23_339) and P.B0W2.d_bounds(12_600) == (4, 12_539)
    fills, _, st = S.simulate(b, f, p)
    assert st[config15.S_NDEC] == 4_667 and st[config15.S_SKIP] == 0                 # 09:30:05 .. 15:58:55


def test_taker_hand_cases():
    """h = 5: buy decided at 1004 fills at the ask (10.02) in bar 1005. At 1009 the exit of the first entry and the
    second entry net to no order (the position is held), and the second entry leaves at 1014 -> sells at the bid (10.00)
    in bar 1015. Cash = 100 x (10.00 - 10.02) = -2.00 $; R fees 0.003 x 200 = 0.60 $ -> net R -2.60 $."""
    b = flat_day()
    f = features15.compute(b)
    p = P.B0W2.kernel_params("Q", 23_400, True)
    td = np.zeros(L.NSEC, np.int64)
    td[1004] = td[1009] = 1
    fills, st = S.simulate_taker(b, f, p, 5, td)
    assert [tuple(r) for r in fills[:, [0, 1, 3, 4, 6]]] == [(1005, 1, 100, 100_200, 1004), (1015, -1, 100, 100_000, 1014)]
    row = P.summarize(fills, st, f, 23_400, 4, 23_339)
    assert row["net_R"] == -100 * 200 - 30 * 200 and row["sh_mkt"] == 200 and row["sh_lim"] == 0
    assert row["open_R"] + row["mid_R"] + row["close_R"] == row["net_R"]
    # inventory limit: 4 buys in a row with h = 60 -> the 4th is refused (|q| <= 300)
    td = np.zeros(L.NSEC, np.int64)
    td[[2004, 2009, 2014, 2019]] = 1
    fills, st = S.simulate_taker(b, f, p, 60, td)
    assert st[config15.S_NINT_B] == 3 and fills[:, 3].sum() == 600                   # 3 entries, 3 exits


def test_time_buckets_add_up_on_real_days():
    man, cb = L.manifest(2019), L.close_bars(2019)
    for (s, d) in sorted(man)[:5]:
        b = L.load(2019, man[(s, d)])
        f = features15.compute(b)
        fills, _, st = S.simulate(b, f, P.B0W2.kernel_params("Q", cb[d], True))
        row = P.summarize(fills, st, f, cb[d], *P.B0W2.d_bounds(cb[d]))
        assert row["open_R"] + row["mid_R"] + row["close_R"] == row["net_R"]
        assert row["open_R+"] + row["mid_R+"] + row["close_R+"] == row["net_Rp"]


def test_fold_leak_rules():
    for fold in ("A", "B", "A16", "B16"):
        tr, val = M15.FOLDS[fold]
        chk = M15.fold_check(tr, val, max(tr))
        assert val not in chk["train_years"] and chk["es_year"] == max(tr) < val
    with pytest.raises(AssertionError):
        M15.fold_check((2015, 2016, 2019), 2019, 2019)                                  # validation year in training
    with pytest.raises(AssertionError):
        M15.fold_check((2015, 2016, 2017, 2018), 2019, 2017)                            # early stopping not on the last year
    idx = M15.lgb_sample(10_000_000)
    assert len(idx) == 6_000_000 and idx.max() < 10_000_000 and np.array_equal(idx, np.sort(idx))
    assert np.array_equal(idx, M15.lgb_sample(10_000_000))                              # seed 15, reproducible


def test_ridge_scaler_uses_training_rows_only():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(size=(5000, len(M15.FEATS))).astype(np.float32)
    Xtr[::7, 3] = np.nan
    y = (Xtr[:, 0] * 2 + rng.normal(size=5000)).astype(np.float32)
    m = M15.ridge_fit(Xtr, y, [10.0])[10.0]
    D = M15.ridge_design(Xtr)
    assert np.allclose(m["mu"], D.mean(0)) and m["n"] == 5000
    Xall = np.vstack([Xtr, rng.normal(5, 1, size=(5000, len(M15.FEATS))).astype(np.float32)])
    assert not np.allclose(m["mu"], M15.ridge_design(Xall).mean(0))                    # validation rows would move it


def test_mirror_sell_rows():
    M = np.zeros((1, 66), np.float32)
    C = M15.C
    M[0, C["bsz"]], M[0, C["asz"]], M[0, C["imb"]], M[0, C["dmid_5"]], M[0, C["spread_tk"]] = 300, 100, 0.5, 2, 3
    xb, xs = M15.sides(M)
    F = list(M15.FEATS)
    assert xs[0, F.index("bsz")] == 100 and xs[0, F.index("asz")] == 300 and xs[0, F.index("imb")] == -0.5
    assert xs[0, F.index("dmid_5")] == -2 and xs[0, F.index("spread_tk")] == 3 and xs[0, F.index("side")] == -1
    assert xb[0, F.index("side")] == 1 and xb[0, F.index("imb")] == 0.5
