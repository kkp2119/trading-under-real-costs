"""M8 synthetic markets with known answers (SPEC 13; ENGINE_DESIGN_E1 4, M8). Price p_u on the tick grid moves at the
start of bar u by dP in {-1, 0, +1} tick (pi, 1 - 2 pi, pi); the move is an update inside bar u, so the bar's extremes
span the old and the new state; quotes bid = p - s/2, ask = p + s/2. Trades follow the update.
(i-T) uninformed trades (100 shares at the bid w.p. lam, at the ask w.p. lam): with reset 1 s, L = 3, no stale exit, a
buy fills under T iff a bid trade and dP <= 0, so E[dP | fill] = -pi/(1 - pi) tick and, p being a martingale,
  E[day net R] = Q (s/2 - tick pi/(1 - pi)) E[N_fill] - (s/2 + tick pi + 30) E|q_df|
with E[N_fill], E|q_df| from an exact forward recursion over q in {-300, ..., 300} (no simulation).
(i-Q) displayed size 50 (queue ahead 200 quarter shares) and 100-share trades: Q fills 50 shares only in a bar without
a move (case 3); no adverse selection, so E = 50 (s/2) E[N_fill] - (s/2 + tick pi + 30) E|q_df|.
(ii) trades are informed (the price moves 2 ticks against the filled side next second): Q markout and net P&L < 0.
(iii) OBI / FI carry the next move: the filter improves the 1 s markout and permutation placebos do not."""
import dataclasses

import numpy as np
import pytest

import accounting
import features
import placebo as plc
import simulator
from config import B0
from synth import COLS, NSEC

TICK, S, PI, LAM = 100, 400, 0.1, 0.3
CFG = dataclasses.replace(B0, delta=1, stale_s=None)
N_DAYS = 300


def market(seed, size=500, trade_sz=100, informed=0, signal=None, s=S):
    r = np.random.default_rng(seed)
    dp = r.choice([-1, 0, 1], p=[PI, 1 - 2 * PI, PI], size=NSEC)
    tb, ta = r.random(NSEC) < LAM, r.random(NSEC) < LAM
    sig = np.zeros(NSEC, dtype=np.int64)
    if informed:                                                     # (ii): next move against the traded side
        for u in range(NSEC - 1):
            if tb[u] != ta[u]:
                dp[u + 1] = -informed if tb[u] else informed
    if signal is not None:                                           # (iii): next move announced at bar u
        sig = r.choice([-1, 0, 1], p=[PI, 1 - 2 * PI, PI], size=NSEC)
        dp[1:] = sig[:-1]
        dp[0] = 0
    p = 500_000 + TICK * np.cumsum(dp)
    b = {c: np.zeros(NSEC, dtype=np.int64) for c in COLS}
    b["bid"], b["ask"] = p - s // 2, p + s // 2
    prev_b = np.concatenate(([b["bid"][0]], b["bid"][:-1]))
    prev_a = np.concatenate(([b["ask"][0]], b["ask"][:-1]))
    b["bid_max"], b["bid_min"] = np.maximum(prev_b, b["bid"]), np.minimum(prev_b, b["bid"])
    b["ask_max"], b["ask_min"] = np.maximum(prev_a, b["ask"]), np.minimum(prev_a, b["ask"])
    b["bsz"][:], b["asz"][:] = size, size
    if signal == "obi":
        b["bsz"] = np.where(sig < 0, 100, np.where(sig > 0, 900, 500))
        b["asz"] = np.where(sig < 0, 900, np.where(sig > 0, 100, 500))
    b["n_upd"] = (dp != 0).astype(np.int64)
    b["vol_bid"], b["vol_ask"] = trade_sz * tb, trade_sz * ta
    if signal == "fi":                                               # a 1,000-share burst on the side of the coming move
        b["vol_ask"] = b["vol_ask"] + 1000 * (sig > 0)
        b["vol_bid"] = b["vol_bid"] + 1000 * (sig < 0)
        tb, ta = b["vol_bid"] > 0, b["vol_ask"] > 0
    b["ntr"] = tb.astype(np.int64) + ta.astype(np.int64)
    b["px_min"] = np.where(tb, b["bid"], np.where(ta, b["ask"], 0))
    b["px_max"] = np.where(ta, b["ask"], np.where(tb, b["bid"], 0))
    b["px_last"] = b["px_max"]
    b["vol"] = b["vol_bid"] + b["vol_ask"]
    return b


def recursion(fill_sz, p_buy, p_sell, n_bars, LQ=300):
    """Exact distribution of q: per bar a resting buy (q < LQ) fills fill_sz w.p. p_buy(dP), a sell (q > -LQ) w.p.
    p_sell(dP), independent given dP. Returns (E[number of fill events], E|q| at the end)."""
    states = np.arange(-LQ, LQ + 1, fill_sz)
    idx = {q: i for i, q in enumerate(states)}
    dist = np.zeros(len(states))
    dist[idx[0]] = 1.0
    e_fill = 0.0
    for _ in range(n_bars):
        new = np.zeros_like(dist)
        for i, q in enumerate(states):
            if dist[i] == 0:
                continue
            for pdp, dp in ((PI, -1), (1 - 2 * PI, 0), (PI, 1)):
                pb = p_buy(dp) if q < LQ else 0.0
                ps = p_sell(dp) if q > -LQ else 0.0
                for bb, sb, pr in ((1, 1, pb * ps), (1, 0, pb * (1 - ps)), (0, 1, (1 - pb) * ps), (0, 0, (1 - pb) * (1 - ps))):
                    w = dist[i] * pdp * pr
                    if w:
                        new[idx[q + fill_sz * (bb - sb)]] += w
                        e_fill += w * (bb + sb)
        dist = new
    return e_fill, float(np.dot(dist, np.abs(states)))


def day_net(b, model):
    f = features.compute(b)
    fills, _, st = simulator.simulate(b, f, CFG.kernel_params(model, NSEC, True))
    row = accounting.account(fills, f["mid2"], f["mid_ok"], NSEC, 899, 23099)
    return accounting.net(row, "R"), row


def check(model, fill_sz, edge, p_buy, p_sell, **kw):
    n_bars = 23099 - 899
    e_fill, e_absq = recursion(fill_sz, p_buy, p_sell, n_bars)
    expect = fill_sz * edge * e_fill - (S / 2 + TICK * PI + 30) * e_absq
    nets = np.array([day_net(market(40_000 + i, **kw), model)[0] for i in range(N_DAYS)], dtype=float)
    se = nets.std(ddof=1) / np.sqrt(N_DAYS)
    assert abs(nets.mean() - expect) <= 2 * se, (nets.mean(), expect, se)
    return nets.mean(), expect, se


def test_i_T_uninformed_flow_matches_the_recursion():
    check("T", 100, S / 2 - TICK * PI / (1 - PI), lambda dp: LAM if dp <= 0 else 0.0, lambda dp: LAM if dp >= 0 else 0.0)


def test_i_Q_queue_with_small_display_matches_the_recursion():
    check("Q", 50, S / 2, lambda dp: LAM if dp == 0 else 0.0, lambda dp: LAM if dp == 0 else 0.0, size=50)


def test_ii_informed_flow_gives_negative_markout_and_net():
    cfg = dataclasses.replace(CFG, min_spread=2)
    mk, net = [], []
    for i in range(50):
        b = market(50_000 + i, size=50, informed=2, s=200)
        f = features.compute(b)
        fills, _, _ = simulator.simulate(b, f, cfg.kernel_params("Q", NSEC, True))
        row = accounting.account(fills, f["mid2"], f["mid_ok"], NSEC, 899, 23099)
        mk.append(row["mk5_sum2"] / 2 / max(row["mk5_sh"], 1))
        net.append(accounting.net(row, "R"))
    for x in (np.array(mk), np.array(net, dtype=float)):
        assert x.mean() + 2 * x.std(ddof=1) / np.sqrt(len(x)) < 0


def markout1(b, cfg, f, base_f, model="T"):
    """1 s markout (sum in 2x units, shares) with signals f; marks always use the real mids of base_f."""
    fills, _, _ = simulator.simulate(b, f, cfg.kernel_params(model, NSEC, True))
    row = accounting.account(fills, base_f["mid2"], base_f["mid_ok"], NSEC, 899, 23099)
    return row["mk1_sum2"], row["mk1_sh"]


@pytest.mark.parametrize("dim", ["obi", "fi"])
def test_iii_informative_filter_improves_markout_and_placebo_does_not(dim):
    days = [market(60_000 + i, signal=dim) for i in range(10)]
    fs = [features.compute(b) for b in days]
    on = dataclasses.replace(CFG, **{dim: True})
    base = [markout1(b, CFG, f, f) for b, f in zip(days, fs)]
    real = [markout1(b, on, f, f) for b, f in zip(days, fs)]
    per = lambda xs: sum(a for a, _ in xs) / max(sum(c for _, c in xs), 1)
    gain = per(real) - per(base)
    plac = []
    for r in range(200):
        xs = []
        for i, (b, f) in enumerate(zip(days, fs)):
            fp = plc.permuted(f, dim, r, "SYN", str(20190102 + i), 899, 23099)
            xs.append(markout1(b, on, fp, f))
        plac.append(per(xs) - per(base))
    plac = np.array(plac)
    assert gain > 0 and gain > plac.max()
    assert abs(plac.mean()) <= 2 * plac.std(ddof=1) / np.sqrt(len(plac)) + 1e-9 or abs(plac.mean()) < 0.1 * gain
