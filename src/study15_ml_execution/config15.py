"""Research 15 copy of research12/config.py (changed lines marked "R15"). Research 12 engine constants and the setting
record (SPEC 4, 5, 7, 8; ENGINE_DESIGN_E1 1; SPEC_AMENDMENTS_draft_E1).
Units: price 1e-4 dollars (1 tick = 100), size shares, money 1e-4 dollars (int64) until reported in dollars."""
import dataclasses
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# R15: no research 12 data paths, holdout/reserve names or bars import here (bar columns from loader15)
from loader15 import COLS as _COLS, NSEC as _NSEC  # noqa: E402

ETFS = ("IWM", "DIA", "XLF", "XLE", "TLT", "IEF", "GLD", "UUP", "EEM", "HYG")
COLS = _COLS                                       # R15
NSEC = _NSEC                                       # 23,400 (R15)
TICK = 100
Q = 100                                            # shares per lot
LOT_FACTOR = 1                                     # bsz/asz are shares already (U1); M12 tests 100
MAX_SPREAD = 12 * TICK
WINDOWS = {0: (899, 23099), 1: (1799, 21599), 2: (4, 23339)}   # (d0, nominal d_f): 09:45-15:55, 10:00-15:30, R15 2: 09:30:05-15:59:00
CLOSE_GAP = 301                                    # d_f <= close bar - 301 (5 minutes before the close)
CLOSE_GAP_W = {2: 61}                              # R15: window 2 flattens 61 bars before the close (15:59:00, or 1 min before an early close)

# costs, 1e-4 dollars per share; a negative limit cost is a rebate (SPEC 8)
COST = {"R": (30, 0), "R+": (30, -20), "C": (100, -50)}
SCENARIOS = tuple(COST)
MODELS = ("S", "T", "Q", "Q50", "Q25")
MODEL_CODE = {"S": (0, 4), "T": (1, 4), "Q": (2, 4), "Q50": (2, 2), "Q25": (2, 1)}     # (kind, 4 theta); S, T log theta 1
MARKOUT_K = (1, 5, 30, 60)
QPATH_BARS = np.arange(899, 23100, 60)               # inventory path grid for B9 (E1-68): bars 899, 959, ..., 23,099

OBI_NUM, OBI_DEN = 3, 5                            # theta_o = 0.6 (E1-17)
FI_NUM, FI_DEN, FI_MIN, FI_WIN = 3, 5, 300, 10     # E1-16
GCNT_WIN = 300                                     # E1-04
HALT_LAG, HALT_WIN, HALT_K2, HALT_MIN2, HALT_LEN = 15, 1800, 135, 2 * TICK, 60     # E1-12/13/51
STALE_QUOTE_WIN = 300                              # E1-47 diagnostic
GATE_DV = 3_000_000 * 10_000                       # 3 million dollars in 1e-4 dollars (D6)
GATE_DAYS, YEAR_START_EXEMPT = 5, 5                # E1-02, E1-03

SEED, BLOCK, B_BOOT, ANN = 12, 10, 10_000, 252
SR_STAR, N_PRIOR = 0.5, 34
G1_SHARPE, G1_QUOTABLE = 1.0, 0.90
ADOPT_GAIN = 0.5                                   # SPEC 5 stage 2 and SPEC 6 step 4

# kernel parameter vector (int64); index names
P_MODE, P_DELTA, P_L, P_STALE, P_LONG, P_OBI, P_HALT, P_FI, P_MINSPR, P_GCNT, P_D0, P_DF, P_NAIVE, P_KIND, P_THETA4, \
    P_GATE = range(16)
N_PARAMS = 16
# kernel statistics vector (int64)
S_SKIP, S_NDEC, S_NDEC_QB, S_NINT_B, S_NINT_S, S_NINTF_B, S_NINTF_S, S_NSTALE, S_QDF, S_MAXQ, S_SUMQ, S_MKT_OVER, \
    S_MKT_STALEQ, S_NMKT = range(14)
N_STATS = 14
FILL_COLS = ("u", "side", "kind", "qty", "price", "ref_mid2", "dec")
ORDER_COLS = ("dec", "side", "price", "size", "a4", "end")      # end = last bar the order rests (E1-60)
MAX_ROWS = 3 * NSEC


@dataclasses.dataclass(frozen=True)
class Config:
    """One position rule (SPEC 4). stale_s None = no stale liquidation; naive = A_naive (SPEC 5, E1-30, E1-55)."""
    mode: str = "reset"
    delta: int = 5
    L: int = 3
    stale_s: object = 60
    long_only: bool = False
    obi: bool = False
    halt: bool = False
    fi: bool = False
    min_spread: int = 2
    gcnt_m: int = 0
    window: int = 0
    naive: bool = False

    def cid(self):
        rq = "event" if self.mode == "event" else f"reset{self.delta}"
        return (f"{rq}|L{self.L}|S{self.stale_s if self.stale_s is not None else 'off'}|{'long' if self.long_only else 'both'}"
                f"|obi{int(self.obi)}|halt{int(self.halt)}|fi{int(self.fi)}|ms{self.min_spread}|m{self.gcnt_m}"
                f"|w{self.window}{'|naive' if self.naive else ''}")

    def as_dict(self):
        return dataclasses.asdict(self)

    def d_bounds(self, close_bar):
        d0, df = WINDOWS[self.window]
        return d0, min(df, close_bar - CLOSE_GAP_W.get(self.window, CLOSE_GAP))      # R15

    def kernel_params(self, model, close_bar, gate):
        kind, theta4 = MODEL_CODE[model]
        d0, df = self.d_bounds(close_bar)
        p = np.zeros(N_PARAMS, dtype=np.int64)
        p[P_MODE] = 1 if self.mode == "event" else 0
        p[P_DELTA] = 1 if self.mode == "event" else self.delta
        p[P_L] = self.L
        p[P_STALE] = -1 if self.stale_s is None else self.stale_s
        p[P_LONG] = int(self.long_only or self.naive)
        p[P_OBI], p[P_HALT], p[P_FI] = int(self.obi), int(self.halt), int(self.fi)
        p[P_MINSPR], p[P_GCNT] = self.min_spread, self.gcnt_m
        p[P_D0], p[P_DF] = d0, df
        p[P_NAIVE] = int(self.naive)
        p[P_KIND], p[P_THETA4] = kind, theta4
        p[P_GATE] = int(bool(gate))
        return p


B0 = Config()
A_NAIVE = Config(L=1, stale_s=None, long_only=True, naive=True)
B_ETF = dataclasses.replace(B0, min_spread=1)
