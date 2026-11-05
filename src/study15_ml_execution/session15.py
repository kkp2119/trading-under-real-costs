"""Research 15 copy of research12/session.py (changed parts marked "R15"): calendar and the liquidity gate G-adv (SPEC 2.1-2.3,
3; E1-02, E1-03). R15: one calendar for 2015-2020 joined from research15/calendar15.csv and research12/calendar.csv;
the split, universe, task and ETF helpers are not copied (loader15 gives the symbol-days)."""
import numpy as np
import pandas as pd

from config15 import GATE_DAYS, GATE_DV, NSEC, YEAR_START_EXEMPT      # R15
import loader15                                                        # R15


def read_calendar(paths):
    """[(YYYYMMDD, close bar)]: close 16:00 -> 23,400, 13:00 -> 12,600. R15: several calendar files, 2015-2020 only."""
    out = []
    for path in paths:
        c = pd.read_csv(path, dtype=str)
        for d, t in zip(c.date, c.close_time):
            if d[:4] in {str(y) for y in loader15.YEARS}:
                h, m = (int(x) for x in t.split(":"))
                out.append((d.replace("-", ""), NSEC - ((16 - h) * 3600 - m * 60)))
    return sorted(out)


class Calendar:
    def __init__(self, paths=None):
        paths = paths or sorted({v[3] for v in loader15.SOURCES.values()})           # R15
        self.days = read_calendar(paths)
        self.close = dict(self.days)
        self.order = [d for d, _ in self.days]
        self.pos = {d: i for i, d in enumerate(self.order)}

    def year_exempt(self, day):
        """First YEAR_START_EXEMPT trading days of the calendar year (E1-03)."""
        same = [d for d in self.order if d[:4] == day[:4]]
        return same.index(day) < YEAR_START_EXEMPT

    def previous(self, day, n=GATE_DAYS):
        i = self.pos[day]
        return self.order[max(0, i - n):i]


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
