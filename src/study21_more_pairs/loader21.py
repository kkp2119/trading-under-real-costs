#!/usr/bin/env python3
"""Research 21 bar loader (after study20_pairs/loader20.py: the same per-file checks, plus the quote-size unit).

Development part only by default: $DATA_DIR/research21_data/dev/bars_1s, 2015-01-02..2020-12-31, the files of
DATA_MANIFEST21.tsv. load() refuses a path under a folder named "sealed", a date after the development window and a root
that is not a development folder, unless the caller passes allow_sealed=True (run_test21.py after its guard, nothing else).
The sealed folder and its manifest are named only in run_test21.py. The development bars of research 20 are a second
allowed root, for the regression test only (regress21.py).
Quote sizes (SPEC21 3, synthesis/size_unit_audit.md): from 2025-11-03 the bar columns bsz and asz hold 100 x the shares.
read() divides these two columns by 100 for a file dated 2025-11-03 or later and refuses a value that is not a multiple
of 100. Prices and trade volumes are not touched and no bar file is changed.
Pairs: universe21_<Y>.tsv (family E or S, pair E1..S4, legs A and B). Control pairs (SPEC21_ADDENDUM C1, family "C") are
built from the year's family S pairs and their SIC codes (results/universe21_pairs.tsv). A pair-day exists when both legs
have a file (SPEC21_ANSWERS A4: a day with one leg only is not a trading day of the pair). Every file is checked against
its manifest sha256, the 21 int32 columns of 23,400 rows and the volume identity."""
import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research21_data/dev/bars_1s")
R20_DEV_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research20_data/dev/bars_1s")        # regression test only
DEV_MANIFEST = os.path.join(HERE, "DATA_MANIFEST21.tsv")
DEV_WINDOW = ("20150102", "20201231")
CALENDAR = os.path.join(HERE, "calendar21.csv")
UNIVERSE = os.path.join(HERE, "universe21_{y}.tsv")
PAIR_STATS = os.path.join(HERE, "results", "universe21_pairs.tsv")
COLS = ("bid", "ask", "bsz", "asz", "bid_max", "bid_min", "ask_max", "ask_min", "n_upd", "n_xl",
        "vol", "ntr", "px_last", "px_min", "px_max", "vol_bid", "vol_ask", "vol_in", "vol_unc", "vol_bid_odd", "vol_ask_odd")
NSEC = 23_400
SIZE_COLS, SIZE_FROM, SIZE_DIV = ("bsz", "asz"), "20251103", 100
FAMILIES, CONTROL = ("E", "S"), "C"


class Refused(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def close_bars(calendar=None):
    """{YYYYMMDD: close bar} from the calendar (16:00 -> 23,400, 13:00 -> 12,600)."""
    c = pd.read_csv(calendar or CALENDAR, dtype=str)
    return {d.replace("-", ""): NSEC - ((16 - int(t[:2])) * 3600 - int(t[3:5]) * 60) for d, t in zip(c.date, c.close_time)}


def pairs(year):
    """[(family, pair label, leg A symbol, leg B symbol)] of the year in the order of universe21_<year>.tsv."""
    u = pd.read_csv(UNIVERSE.format(y=year), sep="\t", dtype=str, keep_default_na=False)
    out = []
    for (fam, p), g in u.groupby(["family", "pair"], sort=False):
        legs = dict(zip(g.leg, g.sym))
        if sorted(legs) != ["A", "B"] or len(g) != 2 or fam not in FAMILIES or not p.startswith(fam):
            raise Refused(f"{year} pair {p}: family {fam}, legs {sorted(legs)}")
        out.append((fam, p, legs["A"], legs["B"]))
    syms = [x for _, _, a, b in out for x in (a, b)]
    if len(set(syms)) != len(syms):
        raise Refused(f"{year}: a symbol in two pairs")
    return out


def sic_codes():
    """{(year, family S pair label): 4-digit SIC code of the pair (both legs have the same code, SPEC21 1.2)}."""
    t = pd.read_csv(PAIR_STATS, sep="\t", dtype=str, keep_default_na=False)
    t = t[t.family == "S"]
    if (t.sic.str.len() == 0).any() or not t.sic.str.isdigit().all():
        raise Refused("a family S pair without a SIC code")
    return {(int(y), p): s.zfill(4) for y, p, s in zip(t.year, t.pair, t.sic)}


def control_pairs(spairs, sic):
    """SPEC21_ADDENDUM C1. spairs: the year's family S pairs [(label, A, B)] in selection order; sic: their 4-digit codes.
    For i = 1..n in turn, A_i takes the first of B_(i+1), B_(i+2), ... (after n comes 1; its own B_i is no candidate)
    whose SIC first digit differs from A_i's and which is not yet in a control pair; without one, A_i has no control pair.
    Returns [("C<i>", A_i, B_j)]."""
    n, used, out = len(spairs), set(), []
    for i in range(n):
        for k in range(1, n):
            j = (i + k) % n
            if sic[j][0] == sic[i][0] or j in used:
                continue
            used.add(j)
            out.append((f"{CONTROL}{i + 1}", spairs[i][1], spairs[j][2]))
            break
    return out


def controls(year):
    sp = [(p, a, b) for fam, p, a, b in pairs(year) if fam == "S"]
    codes = sic_codes()
    return control_pairs(sp, [codes[(int(year), p)] for p, _, _ in sp])


def manifest(path):
    """{(sym, date): row} of a manifest written by check21.py (sym, date, file, rows, bytes, sha256)."""
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if m.duplicated(["sym", "date"]).any():
        raise Refused(f"{os.path.basename(path)}: duplicated (sym, date)")
    return {(r["sym"], r["date"]): r for r in m.to_dict("records")}


def plan(man, window, families=FAMILIES + (CONTROL,)):
    """Pair-days inside the window, in (date, family, pair) order: dicts with date, year, family, pair, name, sym_a, sym_b,
    row_a, row_b, close_bar; the calendar days of the window; and per family the pair-days without a file for one leg or
    for both (not trading days of the pair)."""
    cb = close_bars()
    tasks, uni = [], {}
    skipped = {f: {"one_leg": 0, "neither": 0} for f in families}
    days = [d for d in sorted(cb) if window[0] <= d <= window[1]]
    for d in days:
        y = int(d[:4])
        if y not in uni:
            uni[y] = pairs(y) + [(CONTROL, p, a, b) for p, a, b in controls(y)]
        for fam, p, a, b in uni[y]:
            if fam not in families:
                continue
            ra, rb = man.get((a, d)), man.get((b, d))
            if ra is None or rb is None:
                skipped[fam]["one_leg" if (ra is None) != (rb is None) else "neither"] += 1
                continue
            tasks.append({"date": d, "year": y, "family": fam, "pair": p, "name": f"{a}/{b}", "sym_a": a, "sym_b": b,
                          "row_a": ra, "row_b": rb, "close_bar": cb[d]})
    return tasks, days, skipped


def dev_plan():
    tasks, days, skipped = plan(manifest(DEV_MANIFEST), DEV_WINDOW)
    if any(t["date"] > DEV_WINDOW[1] for t in tasks) or days[-1] > DEV_WINDOW[1]:
        raise Refused("a date after 2020 in the development plan")
    return tasks, days, skipped


def fix_sizes(b, date):
    """Quote sizes in shares: bsz and asz divided by 100 from 2025-11-03 on (in place); everything else as stored."""
    if str(date) >= SIZE_FROM:
        for c in SIZE_COLS:
            if (b[c] % SIZE_DIV).any():
                raise Refused(f"{date}: {c} is not a multiple of {SIZE_DIV}")
            b[c] = b[c] // SIZE_DIV
    return b


def read(path, row):
    """21 columns as int64 for one manifest row at `path`, quote sizes in shares. Refuses on any check failure."""
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
    return fix_sizes(b, row["date"])


def load(root, row, allow_sealed=False):
    """read() behind the development-only guard."""
    path = os.path.join(root, row["file"])
    if not allow_sealed:
        if "sealed" in os.path.abspath(path).split(os.sep):
            raise Refused(f"{path}: a sealed path without the guard")
        if int(row["date"]) > int(DEV_WINDOW[1]):
            raise Refused(f"{row['sym']} {row['date']}: after the development window")
        if os.path.abspath(root) not in (os.path.abspath(DEV_ROOT), os.path.abspath(R20_DEV_ROOT)):
            raise Refused(f"{root}: not a development folder")
    return read(path, row)
