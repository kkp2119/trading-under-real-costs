#!/usr/bin/env python3
"""Research 24 fixed values and statistics (SPEC24 2-6, SPEC24_ANSWERS; readings in DEVIATIONS24.md 1).

The statistics are research 20's (stats20.py, unchanged: daily sums over every trading day of the window, day-block
bootstrap, the A* rule, MDE); the seed, the grid, the amount, the delays and the pair types are new here.
  grid        4 candidates: W {10, 30} min x kappa {2, 3}, z_in = 3, market orders, both families together
  amount      10,000 dollars per leg (ANSWERS A3)
  delays      1 second for the selection and the verdict; 0 and 5 seconds are secondary (ANSWERS A4)
  bootstrap   engine17.boot_index: 10,000 resamples, circular blocks of 21 days, seed 24
  A*          the highest mean daily P&L at delay 1 under L3D1; within 5% the fewest trades, then the earlier candidate
              (stats20.select_astar); a mean <= 0 means "no profit in development" and the sealed run is not made
  H1          mean daily P&L of A* > 0 at delay 1 under L3D1: one-sided p = share of bootstrap means <= 0, alpha 0.05
  pair types  ANSWERS A5 (secondary, no verdict): same company (family D2), leveraged or inverse index ETFs,
              single-stock leveraged and crypto ETFs, plain index ETFs (every other ETF pair); by the symbols below"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R20 = os.path.join(os.path.dirname(HERE), "study20_pairs")
if R20 not in sys.path:
    sys.path.insert(1, R20)
import pair20 as PE  # noqa: E402
import stats20 as ST20  # noqa: E402

G = PE.G
BOOT_B, BOOT_BLOCK, BOOT_SEED = 10_000, 21, 24
WS, Z_IN, KAPPAS = (10, 30), 3.0, (2.0, 3.0)
AMOUNT = 10_000
DELAYS, DELAY_MAIN = (0, 1, 5), 1
VARS, VAR_MAIN = ("L3D1", "L0D0"), "L3D1"
FAMILIES = ("E2", "D2")
ALPHA, TOP = 0.05, 5
TRIALS_BEFORE, TRIALS_ADDED = 847, 4                    # 846 after research 23's development run + its sealed verdict (DEVIATIONS24 1)
SENTENCES = {True: "크게 벌어진 차이만 골라 거래하면 1초 늦어도 비용을 넘는 수익이 났다",
             False: "크게 벌어진 차이만 골라도 1초 늦은 주문으로는 비용을 넘는 수익이 확인되지 않았다"}
NO_PROFIT = "1초 늦은 주문은 개발 구간에서 이미 비용을 넘지 못했다. 그래서 봉인 데이터로 시험하지 않았다"
# ANSWERS A5: pair types by symbol (fixed before any signal or profit was computed; both legs of a pair are of one type)
LEVERAGED_INDEX = frozenset("SPXS SPXU SRTY TZA TNA URTY SH SPDN".split())
SINGLE_OR_CRYPTO = frozenset("NVDL NVDX MSTU MSTX TSLL TSLT PLTU PTIR BITO GBTC FBTC IBIT ARKB BITB BITU BITX ETHA ETHE".split())
TYPES = ("index_etf", "leveraged_index_etf", "single_stock_or_crypto_etf", "same_company")
TYPE_KO = {"index_etf": "일반 지수 ETF", "leveraged_index_etf": "레버리지·인버스 ETF", "single_stock_or_crypto_etf": "단일 종목 레버리지와 가상자산 ETF",
           "same_company": "같은 회사 주식"}

sig_name = ST20.sig_name


def candidates():
    """The 4 candidate signal settings in grid order: (W, z_in, kappa)."""
    return [(W, Z_IN, ka) for W in WS for ka in KAPPAS]


def pair_type(family, a, b):
    """The type of a pair (ANSWERS A5); a pair with one leg in a list and the other outside it is an error."""
    if family == "D2":
        return "same_company"
    for name, group in (("leveraged_index_etf", LEVERAGED_INDEX), ("single_stock_or_crypto_etf", SINGLE_OR_CRYPTO)):
        if (a in group) != (b in group):
            raise ValueError(f"{a}/{b}: one leg only is in the {name} list")
        if a in group:
            return name
    return "index_etf"


def pair_key(a, b):
    """One name for the same two symbols whatever their order (a pair may be listed A/B in one year and B/A in another)."""
    return "/".join(sorted((a, b)))


def boot_index(T, seed=BOOT_SEED):
    return G.boot_index(T, B=BOOT_B, block=BOOT_BLOCK, seed=seed)


def select_astar(rows):
    """SPEC24 4 = SPEC20 4 on the delay-1 L3D1 rows: [{"name", "mean", "trades"}] in candidate order -> (A*, record)."""
    return ST20.select_astar(rows)
