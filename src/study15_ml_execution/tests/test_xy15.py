"""Research 15 step 2 tests (Kyle 2026-10-29): no look-ahead in X, label range, equality with research 12 replay on
2019-01, three hand-computed cases, decision grid, sampling and no_quote. Real bars are read only from 2015 and 2019-2020
(loader15); research 12 modules are imported read-only. Nothing is written outside a temporary directory."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R12 = os.path.join(os.path.dirname(HERE), "study12_market_maker")
sys.path.insert(0, HERE)
import loader15 as L  # noqa: E402
import xy15 as X  # noqa: E402

RNG = np.random.default_rng(20261029)
COL = {c: i for i, c in enumerate(X.COLUMNS)}


def real_days(n_per_year=2, years=(2015, 2017, 2019, 2020)):
    out = []
    for y in years:
        man, cb = L.manifest(y), L.close_bars(y)
        keys = sorted(man)
        for i in RNG.choice(len(keys), n_per_year, replace=False):
            s, d = keys[i]
            out.append((L.load(y, man[(s, d)]), cb[d]))
    return out


def perturb(b, after):
    """Copy of b with every column replaced by random integers at bars > after."""
    c = {k: v.copy() for k, v in b.items()}
    n = len(c["bid"]) - after - 1
    if n > 0:
        for k in c:
            c[k][after + 1:] = RNG.integers(0, 2_000_000, n)
    return c


@pytest.fixture(scope="module")
def days():
    return real_days()


def test_no_lookahead_in_features(days):
    for b, cb in days:
        us = X.decisions(cb)
        for u in RNG.choice(us, 15, replace=False):
            p = perturb(b, int(u))
            keep = us <= u
            a, z = X.features(b, us, cb)[keep], X.features(p, us, cb)[keep]
            assert a.tobytes() == z.tobytes()                       # bit for bit, NaN included
            assert X.build(b, us, cb)[keep][:, :2 + len(X.FEATURES)].tobytes() == \
                X.build(p, us, cb)[keep][:, :2 + len(X.FEATURES)].tobytes()


def test_labels_use_bars_up_to_u_plus_5_plus_h_only(days):
    lab = {c: i for i, c in enumerate(X.LABELS)}
    for b, cb in days:
        us = X.decisions(cb)
        for u in RNG.choice(us, 10, replace=False):
            u = np.array([u])
            for h in X.HS:
                p = perturb(b, int(u[0]) + X.LIFE + h)
                a, z = X.labels(b, u, cb), X.labels(p, u, cb)
                cols = [lab[c] for c in X.LABELS if c.endswith(f"_{h}") or c[2:] in ("fill", "frac", "tf")]
                assert a[:, cols].tobytes() == z[:, cols].tobytes()


def _research12_replay():
    sys.path.insert(0, R12)
    import simulator                                                 # research12/simulator.py, read-only
    assert os.path.dirname(os.path.abspath(simulator.__file__)) == R12
    return simulator


def test_fill_equals_research12_replay_2019_01():
    sim = _research12_replay()
    man, cb = L.manifest(2019), L.close_bars(2019)
    keys = sorted(k for k in man if k[1].startswith("201901"))
    assert len(keys) >= 200
    n_orders = n_filled = n_partial = 0
    for s, d in keys:
        b = L.load(2019, man[(s, d)])
        us = X.decisions(cb[d])
        us = us[X.quote_ok(b)[us]]
        for side in (1, -1):
            path = X.fill_path(b, us, side)
            price = (b["bid"] if side > 0 else b["ask"])[us]
            cum = np.cumsum(path, axis=1)
            for k in range(1, X.LIFE + 1):
                orders = np.column_stack([us, np.full(len(us), side), price, np.full(len(us), X.QTY),
                                          np.zeros(len(us), np.int64), us + k]).astype(np.int64)
                if k < X.LIFE:                                       # earlier ends only where something filled
                    orders = orders[cum[:, -1] > 0]
                out = np.zeros(len(orders), dtype=np.int64)
                sim.replay(orders, len(orders), 2, 4, b["bsz"], b["asz"], b["ask_min"], b["bid_max"], b["bid_min"],
                           b["ask_max"], b["px_min"], b["px_max"], b["vol_bid"], b["vol_ask"], b["vol_bid_odd"],
                           b["vol_ask_odd"], out)
                mine = cum[:, k - 1] if k == X.LIFE else cum[cum[:, -1] > 0, k - 1]
                assert np.array_equal(out, mine), (s, d, side, k)
            n_orders += len(us)
            n_filled += int((cum[:, -1] > 0).sum())
            n_partial += int(((cum[:, -1] > 0) & (cum[:, -1] < X.QTY)).sum())
    print(f"2019-01: {len(keys)} symbol-days, {n_orders} orders, filled {n_filled}, partial {n_partial}")
    assert n_filled > 0 and n_partial > 0


def flat_day():
    b = {c: np.zeros(L.NSEC, dtype=np.int64) for c in L.COLS}
    b["bid"][:], b["ask"][:], b["bsz"][:], b["asz"][:] = 100_000, 100_200, 200, 300
    for c, v in (("bid_max", 100_000), ("bid_min", 100_000), ("ask_max", 100_200), ("ask_min", 100_200)):
        b[c][:] = v
    return b


def row(M, us, u):
    return dict(zip(X.COLUMNS, M[int(np.flatnonzero(us == u)[0])].astype(np.float64)))


def test_hand_case_fill():
    """Buy 100 at 10.00, queue 200 shares (a4 800). Bar 1006: 300 shares trade at the bid all second -> 4*300 > 800,
    fill min(100, (1200 - 800) // 4) = 100 in bar 1006 (tf 2). From bar 1009 the market is 10.01 x 10.03 (mid 10.02):
    mk_5 = mid(1011) - 10.00 = +2 cents, val_5 = 2; tk_5 = bid(1009) - ask(1004) = 10.01 - 10.02 = -1 cent.
    Sell 100 at 10.02 (a4 1200): bars 1005-1008 nothing trades at the ask (a4 stays), bar 1009 ask_min 10.03 > 10.02 with
    no ask volume -> 0; no fill: mk NaN, val 0, tf NaN."""
    b = flat_day()
    b["vol"][1006] = b["vol_bid"][1006] = b["ntr"][1006] = 300
    for c, v in (("bid", 100_100), ("ask", 100_300), ("bid_max", 100_100), ("bid_min", 100_100), ("ask_max", 100_300),
                 ("ask_min", 100_300)):
        b[c][1009:2000] = v
    us = X.decisions(23_400)
    r = row(X.build(b, us, 23_400), us, 1004)
    assert (r["no_quote"], r["b_fill"], r["b_frac"], r["b_tf"]) == (0, 1, 1, 2)
    assert (r["b_mk_5"], r["b_val_5"], r["b_mk_30"], r["b_mk_60"], r["b_tk_5"]) == (2, 2, 2, 2, -1)
    assert (r["s_fill"], r["s_frac"], r["s_val_5"]) == (0, 0, 0) and np.isnan(r["s_mk_5"]) and np.isnan(r["s_tf"])
    assert r["s_tk_5"] == (100_000 - 100_300) / 100                     # bid(1004) - ask(1009) = -3 cents
    assert (r["tr_tk_5"], r["tr_mkb_5"], r["tr_mkb_60"]) == (0, 0, 0) and np.isnan(r["tr_mks_5"])


def test_hand_case_no_fill():
    """Flat day, nothing trades: both orders sit in the queue (Q-3 with zero credit). fill 0, val 0, mk NaN;
    tk_5 = bid - ask = -2 cents on both sides."""
    b = flat_day()
    us = X.decisions(23_400)
    r = row(X.build(b, us, 23_400), us, 2004)
    for p in ("b", "s"):
        assert (r[f"{p}_fill"], r[f"{p}_frac"], r[f"{p}_val_5"], r[f"{p}_val_60"], r[f"{p}_tk_5"]) == (0, 0, 0, 0, -2)
        assert np.isnan(r[f"{p}_mk_5"]) and np.isnan(r[f"{p}_tf"])
    assert (r["spread_tk"], r["imb"], r["dmid_300"], r["sd_30"], r["hl_60"]) == (2, float(np.float32(-0.2)), 0, 0, 0)
    assert np.isnan(r["ofi_5"])                                           # no volume in the window -> NaN


def test_hand_case_truncated_at_the_flatten_bar():
    """Last decision u = F - 5 = 23,334 (F = 23,339, 15:59:00). Bar F: bid_max 10.02 >= my sell 10.02 -> the sell fills
    100 in bar F (tf 5); the market at the end of F is 10.02 x 10.04 (mid 10.03). Every markout is evaluated at F:
    mk_h = -(10.03 - 10.02) = -1 cent, truncated for h = 5, 30, 60. Taker: tk_5 = bid(u) - ask(F) = 10.00 - 10.04 = -4
    (u + 5 = F, not truncated); tk_30 = tk_60 = -4 evaluated at F, truncated. The buy at 10.00 does not fill."""
    b = flat_day()
    F = 23_339
    for c, v in (("bid", 100_200), ("ask", 100_400), ("bid_max", 100_200), ("bid_min", 100_000), ("ask_max", 100_400),
                 ("ask_min", 100_200)):
        b[c][F] = v
    us = X.decisions(23_400)
    assert us[-1] == F - 5
    r = row(X.build(b, us, 23_400), us, F - 5)
    assert (r["s_fill"], r["s_frac"], r["s_tf"]) == (1, 1, 5)
    for h in X.HS:
        assert (r[f"s_mk_{h}"], r[f"s_val_{h}"], r[f"tr_mks_{h}"], r[f"s_tk_{h}"]) == (-1, -1, 1, -4)
    assert (r["tr_tk_5"], r["tr_tk_30"], r["tr_tk_60"]) == (0, 1, 1)
    assert r["b_fill"] == 0 and np.isnan(r["tr_mkb_5"])


def test_decision_grid_sampling_and_no_quote():
    full, half = X.decisions(23_400), X.decisions(12_600)
    assert (full[0], full[-1], len(full)) == (4, 23_334, 4_667)              # 09:30:05 .. 15:58:55
    assert (half[0], half[-1]) == (4, 12_534)                                # early close: F = 12,539
    us, k = X.chosen("20170103", 23_400)
    assert 0 <= k < 6 and np.array_equal(us, full[k::6]) and X.chosen("20170103", 23_400)[1] == k
    us20, k20 = X.chosen("20200102", 23_400)
    assert k20 == -1 and np.array_equal(us20, full)                          # 2020: every decision
    b = flat_day()
    b["bid"][1004] = 0
    M = X.build(b, full, 23_400)
    r = row(M, full, 1004)
    assert r["no_quote"] == 1 and all(np.isnan(r[c]) for c in X.LABELS)
    assert M[:, COL["no_quote"]].sum() == 1
