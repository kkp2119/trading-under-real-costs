#!/usr/bin/env python3
"""Research 20 bar loader for the pair engine (after study15_ml_execution/loader15.py: the same per-file checks).

Development part only: $DATA_DIR/research20_data/dev/bars_1s, 2015-01-02..2020-12-31, the files of DATA_MANIFEST20.tsv
(sha256 pinned by tag spec20). The sealed part cannot be reached from this module: its folder and manifest are named only
in run_test20.py, behind its guard, and load() refuses any path under a folder named "sealed" unless the caller passes
allow_sealed=True (run_test20.py after the guard, nothing else).
A pair-day exists when both legs have a file (SPEC20_ANSWERS A1: a day with one leg only is not a trading day of the pair).
Every file is checked against its manifest sha256, the 21 int32 columns of 23,400 rows and the volume identity."""
import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research20_data/dev/bars_1s")
DEV_MANIFEST = os.path.join(HERE, "DATA_MANIFEST20.tsv")
DEV_WINDOW = ("20150102", "20201231")
CALENDAR = os.path.join(HERE, "calendar20.csv")
UNIVERSE = os.path.join(HERE, "universe20_{y}.tsv")
COLS = ("bid", "ask", "bsz", "asz", "bid_max", "bid_min", "ask_max", "ask_min", "n_upd", "n_xl",
        "vol", "ntr", "px_last", "px_min", "px_max", "vol_bid", "vol_ask", "vol_in", "vol_unc", "vol_bid_odd", "vol_ask_odd")
NSEC = 23_400


class Refused(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def close_bars():
    """{YYYYMMDD: close bar} from calendar20.csv (16:00 -> 23,400, 13:00 -> 12,600)."""
    c = pd.read_csv(CALENDAR, dtype=str)
    return {d.replace("-", ""): NSEC - ((16 - int(t[:2])) * 3600 - int(t[3:5]) * 60) for d, t in zip(c.date, c.close_time)}


def pairs(year):
    """[(pair number, leg A symbol, leg B symbol)] of the year, from universe20_<year>.tsv."""
    u = pd.read_csv(UNIVERSE.format(y=year), sep="\t", dtype=str)
    out = []
    for p, g in u.groupby("pair", sort=True):
        legs = dict(zip(g.leg, g.sym))
        if sorted(legs) != ["A", "B"]:
            raise Refused(f"{year} pair {p}: legs {sorted(legs)}")
        out.append((int(p), legs["A"], legs["B"]))
    return out


def manifest(path):
    """{(sym, date): row} of a manifest written by check20.py (sym, date, file, rows, bytes, sha256)."""
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if m.duplicated(["sym", "date"]).any():
        raise Refused(f"{os.path.basename(path)}: duplicated (sym, date)")
    return {(r["sym"], r["date"]): r for r in m.to_dict("records")}


def plan(man, window):
    """Pair-days inside the window, in (date, pair) order: dicts with date, year, pair, name, sym_a, sym_b, row_a, row_b,
    close_bar. Calendar days of the window on which one or both legs have no file are counted in the second result."""
    cb = close_bars()
    tasks, skipped, uni = [], {"one_leg": 0, "neither": 0}, {}
    for d in sorted(cb):
        if not window[0] <= d <= window[1]:
            continue
        y = int(d[:4])
        if y not in uni:
            uni[y] = pairs(y)
        for p, a, b in uni[y]:
            ra, rb = man.get((a, d)), man.get((b, d))
            if ra is None or rb is None:
                skipped["one_leg" if (ra is None) != (rb is None) else "neither"] += 1
                continue
            tasks.append({"date": d, "year": y, "pair": p, "name": f"{a}/{b}", "sym_a": a, "sym_b": b, "row_a": ra, "row_b": rb,
                          "close_bar": cb[d]})
    days = [d for d in sorted(cb) if window[0] <= d <= window[1]]
    return tasks, days, skipped


def dev_plan():
    tasks, days, skipped = plan(manifest(DEV_MANIFEST), DEV_WINDOW)
    if any(t["date"] > DEV_WINDOW[1] for t in tasks) or days[-1] > DEV_WINDOW[1]:
        raise Refused("a date after 2020 in the development plan")
    return tasks, days, skipped


def load(root, row, allow_sealed=False):
    """21 columns as int64 for one manifest row under `root`. Refuses on any check failure."""
    path = os.path.join(root, row["file"])
    if not allow_sealed and "sealed" in os.path.abspath(path).split(os.sep):
        raise Refused(f"{path}: a sealed path without the guard")
    if int(row["date"]) > int(DEV_WINDOW[1]) and not allow_sealed:
        raise Refused(f"{row['sym']} {row['date']}: after the development window")
    if sha256(path) != row["sha256"]:
        raise Refused(f"{row['sym']} {row['date']}: sha256 differs from the manifest")
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
