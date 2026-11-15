#!/usr/bin/env python3
"""Research 21 statistics and fixed values (SPEC21 4-6, SPEC21_ADDENDUM; readings in DEVIATIONS21.md 1).

The statistics are research 20's (stats20.py, unchanged: daily sums over every trading day, day-block bootstrap, ratio
intervals, Sharpe, the A* rule, MDE); only the seed, the grid and the hypotheses are new here.
  grid        8 candidates per family: W {10, 30} min x z_in {2, 3} x kappa {1, 2}, market orders (M)
  H1 setting  W 30, z_in 2, kappa 0 for both families (also the comparison run C2 and the control pairs C1)
  bootstrap   engine17.boot_index: 10,000 resamples, circular blocks of 21 days, seed 21
  A*          per family: the highest L3D1 mean daily P&L, within 5% the fewest trades, then the earlier candidate
              (stats20.select_astar); a mean <= 0 means "no profit in development" and that family has no H2
  Holm        over the hypotheses left (H1-E, H1-S and the H2 of each family that has one), overall alpha 0.05"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R20 = os.path.join(os.path.dirname(HERE), "study20_pairs")
if R20 not in sys.path:
    sys.path.insert(1, R20)
import pair20 as PE  # noqa: E402
import stats20 as ST20  # noqa: E402

G = PE.G
BOOT_B, BOOT_BLOCK, BOOT_SEED = 10_000, 21, 21
WS, ZINS, KAPPAS = (10, 30), (2.0, 3.0), (1.0, 2.0)
H1_SIGNAL = (30, 2.0, 0.0)
FAMILIES = ("E", "S")
FAMILY_X = {"E": "거의 같은 ETF 쌍", "S": "같은 업종 주식 쌍"}
ALPHA = 0.05
TRIALS_BEFORE, TRIALS_ADDED = 794, 16
SENTENCES = {"H1": {True: "{x}의 가격이 벌어지면 처음 보는 기간에도 다시 붙었다", False: "{x}의 되돌림은 처음 보는 기간에서 확인되지 않았다"},
             "H2": {True: "{x}의 차익은 실제 비용을 넣고도 수익이 났다", False: "{x}의 되돌림은 있었지만 비용을 넘는 수익은 확인되지 않았다"}}
NO_H2 = "{x}은 개발 구간에서 이미 비용을 넘지 못했다. 그래서 수익 가설은 봉인 데이터로 시험하지 않았다"    # "X는" of SPEC21 6 with the particle for "쌍"

sig_name = ST20.sig_name


def candidates():
    """The 8 candidate signal settings in order: W, z_in, kappa."""
    return [(W, zi, ka) for W in WS for zi in ZINS for ka in KAPPAS]


def settings():
    """The 9 signal settings of the development tables: the candidates, then the H1 setting."""
    return candidates() + [H1_SIGNAL]


def cand_name(c):
    return f"{sig_name(c)}_M"


def boot_index(T, seed=BOOT_SEED):
    return G.boot_index(T, B=BOOT_B, block=BOOT_BLOCK, seed=seed)


def select_astar(rows):
    """SPEC21 4 = SPEC20 4: rows [{"name", "mean", "trades"}] in candidate order -> (A*, record)."""
    return ST20.select_astar(rows)


def hypotheses(h2):
    """Names of the hypotheses left, in the fixed order H1-E, H1-S, H2-E, H2-S; h2 = {"E": bool, "S": bool}."""
    return [f"H1-{f}" for f in FAMILIES] + [f"H2-{f}" for f in FAMILIES if h2[f]]


def holm(p):
    return G.holm(p, alpha=ALPHA)


def strictest_level(n_hyp):
    """The first Holm level with n_hyp hypotheses."""
    return ALPHA / n_hyp


def sentence(h, family, confirmed):
    """SPEC21 6, as written; h = "H1" or "H2"."""
    return SENTENCES[h][bool(confirmed)].format(x=FAMILY_X[family])


def no_h2_sentence(family):
    return NO_H2.format(x=FAMILY_X[family])
