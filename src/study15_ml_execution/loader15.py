"""Research 15 bar loader (copy of research12/loader.py checks; changed parts marked "R15"). Two read-only sources, 2015-2020
only: research15 train bars (2015-2018) and research 12 development bars (2019-2020). No other folder of
$DATA_DIR/research12_data is listed or read (R15: the holdout/reserve split names do not exist here). Every file is checked
against its MANIFEST sha256, the 21 int32 columns of 23,400 rows and the volume identity, as in research 12."""
import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R12 = os.path.join(os.path.dirname(HERE), "study12_market_maker")
COLS = ("bid", "ask", "bsz", "asz", "bid_max", "bid_min", "ask_max", "ask_min", "n_upd", "n_xl",
        "vol", "ntr", "px_last", "px_min", "px_max", "vol_bid", "vol_ask", "vol_in", "vol_unc", "vol_bid_odd", "vol_ask_odd")
NSEC = 23_400
# R15: (root, years, universe file pattern, calendar) per source
SOURCES = {
    "train15": (os.path.join(os.environ.get("DATA_DIR", "data"), "research15_data/bars_1s_train_stocks"), range(2015, 2019),
                os.path.join(HERE, "universe15_{y}.tsv"), os.path.join(HERE, "calendar15.csv")),
    "dev12": (os.path.join(os.environ.get("DATA_DIR", "data"), "research12_data/bars_1s_dev_stocks"), range(2019, 2021),
              os.path.join(R12, "universe_{y}.tsv"), os.path.join(R12, "calendar.csv")),
}
YEARS = range(2015, 2021)


class Refused(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def source_of(year):
    for name, (root, years, _, _) in SOURCES.items():
        if year in years:
            return name
    raise Refused(f"{year} is outside 2015-2020")                     # R15: nothing after 2020 is reachable


def close_bars(year):
    """{YYYYMMDD: close bar} from the calendar of the year's source (16:00 -> 23,400, 13:00 -> 12,600)."""
    c = pd.read_csv(SOURCES[source_of(year)][3], dtype=str)
    out = {}
    for d, t in zip(c.date, c.close_time):
        if int(d[:4]) == year:
            h, m = (int(x) for x in t.split(":"))
            out[d.replace("-", "")] = NSEC - ((16 - h) * 3600 - m * 60)
    return out


def universe(year):
    return pd.read_csv(SOURCES[source_of(year)][2].format(y=year), sep="\t", dtype=str).sym.tolist()


def manifest(year):
    """MANIFEST rows (dict of str) of `year`, keyed (sym, day): only the year's universe symbols (R15)."""
    root = SOURCES[source_of(year)][0]
    m = pd.read_csv(os.path.join(root, "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False)
    m = m[(m.date.str[:4] == str(year)) & m.sym.isin(universe(year))].drop_duplicates(["sym", "date"], keep="last")
    return {(r["sym"], r["date"]): r for r in m.to_dict("records")}


def load(year, row):
    """21 columns as int64 for one MANIFEST row. Refuses on any check failure (research 12 Split.load checks)."""
    path = os.path.join(SOURCES[source_of(year)][0], row["file"])
    if sha256(path) != row["sha256"]:
        raise Refused(f"{row['sym']} {row['date']}: sha256 differs from MANIFEST")
    with np.load(path, allow_pickle=False) as z:
        if sorted(z.files) != sorted(COLS):
            raise Refused(f"{row['sym']} {row['date']}: columns {sorted(z.files)}")
        b = {}
        for c in COLS:
            a = z[c]
            if a.dtype != np.int32 or a.shape != (NSEC,):
                raise Refused(f"{row['sym']} {row['date']}: {c} {a.dtype} {a.shape}")
            b[c] = a.astype(np.int64)
    if not np.array_equal(b["vol"], b["vol_bid"] + b["vol_ask"] + b["vol_in"] + b["vol_unc"]):
        raise Refused(f"{row['sym']} {row['date']}: vol identity fails")
    return b
