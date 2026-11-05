"""Research 15 reference experiment R1 tests: with every option off the kernel equals B0 (2019-01, every symbol-day), and
hand cases for V1 (keep the queue), V2 (one tick inside, alone at the price) and V4 (exit side one tick inside).
Run with python -m pytest."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import features15  # noqa: E402
import loader15 as L  # noqa: E402
import model15 as M15  # noqa: E402
import policy15 as P  # noqa: E402
import simulator15 as S  # noqa: E402

OFF, V1, V2, V3, V4 = [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 1]


def test_options_off_equal_b0_on_2019_01():
    man, cb = L.manifest(2019), L.close_bars(2019)
    g = M15.read_gates().set_index(["sym", "date"])
    n = 0
    for (s, d) in sorted(k for k in man if k[1].startswith("201901")):
        b = L.load(2019, man[(s, d)])
        f = features15.compute(b)
        p = P.B0W2.kernel_params("Q", cb[d], bool(g.loc[(s, d)].gate))
        a, z = S.simulate(b, f, p), S.simulate(b, f, p, vopt=OFF)
        assert all(np.array_equal(x, y) for x, y in zip(a, z))
        n += len(a[0])
    assert n > 0


def day(spread_ticks):
    b = {c: np.zeros(L.NSEC, dtype=np.int64) for c in L.COLS}
    ask = 100_000 + 100 * spread_ticks
    b["bid"][:], b["ask"][:], b["bsz"][:], b["asz"][:] = 100_000, ask, 200, 300
    for c, v in (("bid_max", 100_000), ("bid_min", 100_000), ("ask_max", ask), ("ask_min", ask)):
        b[c][:] = v
    return b


def fills_of(b, vopt):
    f = features15.compute(b)
    fills, orders, st = S.simulate(b, f, P.B0W2.kernel_params("Q", 23_400, True), vopt=vopt)
    return [tuple(int(x) for x in r) for r in fills[:, [0, 1, 3, 4]]]          # (bar, side, qty, price)


def test_v1_keeps_the_queue():
    """Buy at 10.00 with 200 shares ahead (a4 800). Bar 1006: 150 shares trade at the bid -> a4 200. Bar 1011: 100 more.
    V1 keeps the order at the 1009 decision: 400 > 200 -> fills (400 - 200) // 4 = 50. B0 renews it at 1009 with a4 800:
    no fill."""
    b = day(2)
    b["vol"][1006] = b["vol_bid"][1006] = 150
    b["vol"][1011] = b["vol_bid"][1011] = 100
    assert (1011, 1, 50, 100_000) in fills_of(b, V1)
    assert not any(r[0] == 1011 for r in fills_of(b, OFF))


def test_v2_one_tick_inside_alone():
    """Spread 3 ticks (10.00 x 10.03): V2 buys at 10.01 with nothing ahead; 100 shares trade at the data bid 10.00 in
    bar 1006 -> Q-2, fills 100 at 10.01. B0 (10.00, a4 800): 400 < 800, no fill. Spread 2 ticks: V2 = B0."""
    b = day(3)
    b["vol"][1006] = b["vol_bid"][1006] = 100
    assert (1006, 1, 100, 100_100) in fills_of(b, V2)
    assert not any(r[0] == 1006 for r in fills_of(b, OFF))
    c = day(2)
    c["vol"][1006] = c["vol_bid"][1006] = 100
    assert fills_of(c, V2) == fills_of(c, OFF)


def test_v4_exit_one_tick_inside():
    """Spread 2 ticks. Bar 1006: ask_min 10.00 crosses the buy at 10.00 -> long 100. Decision 1009: V4 sells at 10.01
    (ask - 1 tick, nothing ahead); 100 shares trade at the data ask 10.02 in bar 1011 -> Q-2, sells 100 at 10.01. V3
    (spread < 3: sell at 10.02 behind 300 shares): no fill."""
    b = day(2)
    b["ask_min"][1006] = 100_000
    b["vol"][1011] = b["vol_ask"][1011] = 100
    assert (1006, 1, 100, 100_000) in fills_of(b, V4)
    assert (1011, -1, 100, 100_100) in fills_of(b, V4)
    assert not any(r[0] == 1011 for r in fills_of(b, V3))
