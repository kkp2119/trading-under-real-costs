"""Research 26 data wiring (SPEC26 1; DEVIATIONS26 1): two bar folders, two kinds of stock list and one calendar behind
research 12's engine, which is imported and never edited.
  bars       2015-2018: $DATA_DIR/research15_data/bars_1s_train_stocks (research 15's training stocks)
             2019-2020: $DATA_DIR/research12_data/bars_1s_dev_stocks   (research 12's development stocks)
             each folder is opened with research 12's loader.Split (MANIFEST sha256, 21 int32 columns of 23,400 rows, volume
             identity, its own date range); TwoFolders routes every symbol-day to the folder that holds it
  lists      2015-2018: study15_ml_execution/universe15_<Y>.tsv;  2019-2020: study12_market_maker/universe_<Y>.tsv (same columns)
  calendar   study15_ml_execution/calendar15.csv for 2015-2018 and study12_market_maker/calendar.csv for 2019-2020, joined (calendar26.csv)
  windows    dev26 2015-01-02..2020-12-31 (the development run); regress26 2019-01-02..2020-12-31 (the regression)
Symbol-days without a file (absent from the TAQ master) are left out and never replaced. No ETF folder is used."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
R12, R15 = os.path.join(PROJ, "study12_market_maker"), os.path.join(PROJ, "study15_ml_execution")
if R12 not in sys.path:
    sys.path.insert(0, R12)
import config as C12  # noqa: E402  (research 12, not edited)
import loader as L12  # noqa: E402
import session as S12  # noqa: E402

BACKUPS = os.environ.get("DATA_DIR", "data")
DATA26 = os.path.join(BACKUPS, "research26_data")
TRAIN = ("20150102", "20181231")                                    # research 15's training folder
WINDOWS = {"dev26": ("20150102", "20201231"), "regress26": C12.SPLITS["dev"]}
CALENDAR26 = os.path.join(HERE, "calendar26.csv")
SOURCES = {"train": (os.path.join(BACKUPS, "research15_data"), TRAIN), "dev": (os.path.join(BACKUPS, "research12_data"), C12.SPLITS["dev"])}
Refused = L12.Refused


def register():
    """The windows of this study in research 12's SPLITS table (in this process only; the engine file is not touched).
    'train' is the name research 12's folder rule needs for research 15's folder (bars_1s_train_stocks)."""
    for k, v in list(WINDOWS.items()) + [("train", TRAIN)]:
        if C12.SPLITS.get(k, v) != v:
            raise Refused(f"SPLITS[{k!r}] already holds another window")
        C12.SPLITS[k] = v


def universe(year, list_dirs=(R15, R12)):
    """The year's ten stocks: universe15_<Y>.tsv for 2015-2018 (research 15), universe_<Y>.tsv for 2019-2020 (research 12)."""
    d15, d12 = list_dirs
    path = os.path.join(d15, f"universe15_{year}.tsv") if year <= 2018 else os.path.join(d12, f"universe_{year}.tsv")
    return pd.read_csv(path, sep="\t", dtype=str).sym.tolist()


def build_calendar(path=CALENDAR26, cal15=os.path.join(R15, "calendar15.csv"), cal12=os.path.join(R12, "calendar.csv")):
    """calendar15 rows of 2015-2018 + calendar rows of 2019-2020, written once (an existing file must be identical)."""
    a = pd.read_csv(cal15, dtype=str)
    b = pd.read_csv(cal12, dtype=str)
    a = a[(a.date >= "2015-01-01") & (a.date <= "2018-12-31")]
    b = b[(b.date >= "2019-01-01") & (b.date <= "2020-12-31")]
    c = pd.concat([a, b], ignore_index=True)[["date", "close_time"]]
    if not c.date.is_monotonic_increasing or c.date.duplicated().any():
        raise Refused("calendar26: dates not strictly increasing")
    text = c.to_csv(index=False)
    if os.path.exists(path):
        if open(path).read() != text:
            raise Refused(f"{path} exists with other content (nothing is overwritten)")
    else:
        with open(path, "w") as f:
            f.write(text)
    return path


class TwoFolders:
    """research 12's Split interface (meta, absent, available, zero_days, load, verify_all) over two folders. Every row
    of both MANIFESTs must lie inside the run window, and no symbol-day may be in both folders."""

    def __init__(self, window, sources=None, results_dir=None):
        self.split, self.asset, self.window = "dev26", "stocks", window
        self.parts, self.owner, self.meta, self.absent = [], {}, {}, set()
        for name, (base, rng) in (sources or SOURCES).items():
            if rng[1] < window[0] or rng[0] > window[1]:
                continue                                              # a folder entirely outside the window is not opened
            p = L12.Split(name, "stocks", base=base, results_dir=results_dir)
            out = [k for k in p.meta if not (window[0] <= k[1] <= window[1])]
            if out:
                raise Refused(f"{len(out)} MANIFEST rows of {p.root} outside the run window {window[0]}..{window[1]}")
            both = set(p.meta) & set(self.meta)
            if both:
                raise Refused(f"{len(both)} symbol-days in two folders")
            i = len(self.parts)
            self.parts.append(p)
            for k, r in p.meta.items():
                self.owner[k] = i
                self.meta[k] = r
            self.absent |= p.absent
        if not self.parts:
            raise Refused("no bar folder inside the run window")

    def available(self):
        return set(self.meta)

    def zero_days(self):
        return {k for k, r in self.meta.items() if int(r["n_quote_rows"]) == 0 and int(r["n_trade_rows"]) == 0}

    def load(self, sym, day):
        i = self.owner.get((sym, day))
        if i is None:
            raise Refused(f"{sym} {day}: not in either MANIFEST")
        return self.parts[i].load(sym, day)

    def verify_all(self):
        return [k for p in self.parts for k in p.verify_all()]

    def roots(self):
        return [p.root for p in self.parts]


def tasks(cal, window_name, available, list_dirs=(R15, R12)):
    """session.tasks with the two kinds of list: (sym, day) for the year's ten stocks x the window's calendar days with a
    bar file. A symbol-day without a file is absent and is not replaced."""
    out = []
    for d in cal.split_days(window_name):
        for s in universe(int(d[:4]), list_dirs):
            if (s, d) in available:
                out.append((s, d))
    return out


def missing(cal, window_name, available, list_dirs=(R15, R12)):
    """Symbol-days of the lists and the calendar without a bar file (reported, never replaced)."""
    return [(s, d) for d in cal.split_days(window_name) for s in universe(int(d[:4]), list_dirs) if (s, d) not in available]


Calendar = S12.Calendar
