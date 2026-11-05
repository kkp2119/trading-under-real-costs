"""Calendar, sessions, universes, task lists and the liquidity gate (SPEC 2.1-2.3, 3; E1-02, E1-03, E1-21, E1-22, E1-36)."""
import os

import numpy as np
import pandas as pd

from config import CALENDAR, ETFS, GATE_DAYS, GATE_DV, HERE, NSEC, SPLITS, YEAR_START_EXEMPT


def read_calendar(path=CALENDAR):
    """[(YYYYMMDD, close bar)]: close 16:00 -> 23,400, 13:00 -> 12,600."""
    c = pd.read_csv(path, dtype=str)
    out = []
    for d, t in zip(c.date, c.close_time):
        h, m = (int(x) for x in t.split(":"))
        out.append((d.replace("-", ""), NSEC - ((16 - h) * 3600 - m * 60)))
    return out


class Calendar:
    def __init__(self, path=CALENDAR):
        self.days = read_calendar(path)
        self.close = dict(self.days)
        self.order = [d for d, _ in self.days]
        self.pos = {d: i for i, d in enumerate(self.order)}

    def split_days(self, split):
        lo, hi = SPLITS[split]
        return [d for d in self.order if lo <= d <= hi]

    def year_exempt(self, day):
        """First YEAR_START_EXEMPT trading days of the calendar year (E1-03)."""
        same = [d for d in self.order if d[:4] == day[:4]]
        return same.index(day) < YEAR_START_EXEMPT

    def previous(self, day, n=GATE_DAYS):
        i = self.pos[day]
        return self.order[max(0, i - n):i]


def universe(year, asset, universe_dir=HERE):
    if asset == "etf":
        return list(ETFS)
    return pd.read_csv(os.path.join(universe_dir, f"universe_{year}.tsv"), sep="\t", dtype=str).sym.tolist()


def tasks(cal, split, asset, available, universe_dir=HERE):
    """(sym, day) for the year's universe x the split's calendar days with a bar file (`available`: set of (sym, day));
    symbol-days without a file are absent (not traded, E1-02)."""
    out = []
    for d in cal.split_days(split):
        for s in universe(int(d[:4]), asset, universe_dir):
            if (s, d) in available:
                out.append((s, d))
    return out


def observation_days(cal, split, available):
    """Calendar days of the split with at least one bar file (E1-36: a day without any file, e.g. 2026-02-24, is not an
    observation)."""
    have = {d for _, d in available}
    return [d for d in cal.split_days(split) if d in have]


def gate(cal, sym, day, dv, zero_days, available):
    """G-adv (SPEC 3, E1-02, E1-03): pass on the first 5 trading days of a year; otherwise the median daily dollar volume
    over the previous 5 calendar trading days with a file (absent days dropped, zero-bar days counted as 0) >= GATE_DV;
    no such day -> fail. dv: {(sym, day): 1e-4 $}; zero_days: set of (sym, day) with no data."""
    if cal.year_exempt(day):
        return True
    vals = [0 if (sym, d) in zero_days else dv[(sym, d)] for d in cal.previous(day) if (sym, d) in available]
    if not vals:
        return False
    return bool(np.median(np.asarray(vals, dtype=np.float64)) >= GATE_DV)
