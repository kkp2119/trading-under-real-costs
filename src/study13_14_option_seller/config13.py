#!/usr/bin/env python3
"""Research 13 engine constants (SPEC spec13-v1 3d7dd482...; SPEC_AMENDMENTS A13-1/2/3 695315de...; design memo 4b v2
e82942ef...; AUDIT_G0.md section 6 switch table approved 2026-10-28 16:33 ET)."""
from dataclasses import dataclass

SPEC_SHA = "3d7dd482126b949c053c9504ccc2e87e81d092953d3c09a96f935dd6d9e073f4"
AMEND_SHA = "695315de77041692cfc57b4cc2d08d2faa1f0685f6b1df5c1b1dc86be9a2290a"
MEMO_4B_SHA = "e82942ef4def60d55a0fff96f40769c698a56889f861424acbbfc4fba4b6d8d0"

# 3.2, 3.3, 3.5, 3.7
N0 = 1_000_000.0
MULT = 100.0
TAU = {"S": 0.25, "G": 0.50}
TAU_FWD = 0.25
G_TARGET = 0.16
RANGE = {21: (35, 65, 45), 10: (19, 49, 30)}
DTE_EXIT = 5
DEFER_MAX = 3
DTE_MIN_DEFER = 2

# 3.8, 3.9, E13-02
RV_N = 21
SIZING_REF, SIZING_LO, SIZING_HI = 0.16, 0.25, 1.5
IV_RANGE = (0.01, 3.0)

# 4 (scenario order fixed: base, stress, opt)
SCEN = ("base", "stress", "opt")
FEE = {"base": 1.0, "stress": 2.0, "opt": 0.0}
HEDGE_BP = {"base": 1e-4, "stress": 3e-4, "opt": 0.25e-4}
KAPPA = {"base": 1.25, "stress": 1.50, "opt": 1.00}
STRESS_HS = 1.5

# switch table (AUDIT_G0 6, Kyle 2026-10-28 16:33 ET)
DELTA_SOURCE = "optionmetrics_then_bs_then_previous"
T_BASIS = "T1"
NONSTD_CODES = ("m", "w")

# dates (D2, 5.2, A13-2, 2.3)
DEV_START = 19960104
DEV_END = 20121231
HOLD_START = 20130102
HOLD_END = 20250829
STATS_OFFSET = 43
F2_SPLIT = 20041231

# 6, 9, 10, 11
ANN = 252
SR_STAR = 0.5
SEED, BLOCK, B_BOOT = 13, 21, 10_000
SENS_BLOCKS = (10, 42)
N_TRIALS = (11, 177)
LO_LAGS = 21
DEFECT_U = 0.20
DEFECT_FLOOR, DEFECT_MARGIN = 0.05, 0.05
TAIL_RATIO = 1.5
EVENTS = {"2008-09..11": (20080901, 20081130), "2018-02": (20180201, 20180228),
          "2020-02..04": (20200201, 20200430), "2022": (20220101, 20221231)}

# 8
F1_MIN, CALMAR_MIN, TIE_BAND = 0.50, 0.33, 0.10

# items decided by Kyle 2026-10-28 (A13-4); a run refuses if either is None
ENTRY_ON_LAST_DAY = False         # N2 (Kyle 2026-10-28, A13-4): no new tranche on the data-end day
RV21_FILL_MISSING_CLOSE = True    # N3 (Kyle 2026-10-28, A13-4): a missing close in the RV21 window -> previous close (3.4)


@dataclass(frozen=True)
class Setting:
    id: str
    structure: str      # "S" straddle, "G" strangle
    H: int              # 21 or 10
    gate: str           # "G0", "G1", "G2"
    sizing: str         # "K", "V"
    complexity: int


SETTINGS = (
    Setting("O0", "S", 21, "G0", "K", 0), Setting("O1", "S", 10, "G0", "K", 1), Setting("O2", "G", 21, "G0", "K", 1),
    Setting("O3", "S", 21, "G1", "K", 1), Setting("O4", "S", 21, "G2", "K", 1), Setting("O5", "S", 21, "G0", "V", 1),
    Setting("O6", "S", 21, "G1", "V", 2), Setting("O7", "S", 21, "G2", "V", 2), Setting("O8", "G", 21, "G0", "V", 2),
    Setting("O9", "G", 21, "G1", "V", 3), Setting("O10", "S", 10, "G1", "V", 3))
BY_ID = {s.id: s for s in SETTINGS}


def setting(sid):
    if sid not in BY_ID:
        raise KeyError(f"{sid} is not one of the 11 settings of SPEC 7")
    return BY_ID[sid]
