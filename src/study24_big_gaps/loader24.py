#!/usr/bin/env python3
"""Research 24 bar loader (after study21_more_pairs/loader21.py: its per-file checks and its quote-size correction are used as
they are, through loader21.read).

Development part only by default: $DATA_DIR/research24_data/dev/bars_1s, 2015-01-02..2020-12-31, the files of
DATA_MANIFEST24.tsv. load() refuses a path under a folder named "sealed", a date after the development window and a root
that is not a development folder, unless the caller passes allow_sealed=True (run_test24.py after its guard, nothing else).
The sealed folder and its manifest are named only in run_test24.py. The development bars of research 20 and 21 are two more
allowed roots, for the regression test only (regress24.py).
Quote sizes (SPEC24 3): loader21.read divides bsz and asz by 100 for a file dated 2025-11-03 or later and refuses a value
that is not a multiple of 100. Prices and trade volumes are not touched and no bar file is changed.
Pairs: universe24_<Y>.tsv (family E2 or D2, pair P01.., legs A and B; N_p = 20 by SPEC24_ANSWERS A2, fewer in a year
with fewer pairs, A7). A pair-day exists when both legs have a file (SPEC24_QUESTIONS 16). Every task carries the pair's
type (stats24.pair_type, ANSWERS A5) and a key that names the two symbols whatever their order."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for _p in (os.path.join(PROJ, "study21_more_pairs"), os.path.join(PROJ, "study20_pairs")):
    if _p not in sys.path:
        sys.path.insert(1, _p)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import loader21 as L21  # noqa: E402
import stats24 as ST24  # noqa: E402

DEV_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research24_data/dev/bars_1s")
REGRESSION_ROOTS = (L21.R20_DEV_ROOT, L21.DEV_ROOT)               # development bars of research 20 and 21, regression test only
DEV_MANIFEST = os.path.join(HERE, "DATA_MANIFEST24.tsv")
DEV_WINDOW = ("20150102", "20201231")
CALENDAR = os.path.join(HERE, "calendar24.csv")
UNIVERSE = os.path.join(HERE, "universe24_{y}.tsv")
FAMILIES = ST24.FAMILIES
COLS, NSEC, SIZE_FROM = L21.COLS, L21.NSEC, L21.SIZE_FROM
Refused, sha256, manifest, read = L21.Refused, L21.sha256, L21.manifest, L21.read


def close_bars():
    return L21.close_bars(CALENDAR)


def pairs(year):
    """[(family, pair label, leg A symbol, leg B symbol)] of the year in the order of universe24_<year>.tsv."""
    u = pd.read_csv(UNIVERSE.format(y=year), sep="\t", dtype=str, keep_default_na=False)
    out = []
    for (fam, p), g in u.groupby(["family", "pair"], sort=False):
        legs = dict(zip(g.leg, g.sym))
        if sorted(legs) != ["A", "B"] or len(g) != 2 or fam not in FAMILIES:
            raise Refused(f"{year} pair {p}: family {fam}, legs {sorted(legs)}")
        out.append((fam, p, legs["A"], legs["B"]))
    syms = [x for _, _, a, b in out for x in (a, b)]
    if len(set(syms)) != len(syms):
        raise Refused(f"{year}: a symbol in two pairs")
    return out


def plan(man, window):
    """Pair-days inside the window, in (date, pair) order: dicts with date, year, family, type, pair, name, key, sym_a,
    sym_b, row_a, row_b, close_bar; the calendar days of the window; and per family the pair-days without a file for
    one leg or for both (not trading days of the pair)."""
    cb = close_bars()
    tasks, uni = [], {}
    skipped = {f: {"one_leg": 0, "neither": 0} for f in FAMILIES}
    days = [d for d in sorted(cb) if window[0] <= d <= window[1]]
    for d in days:
        y = int(d[:4])
        if y not in uni:
            uni[y] = pairs(y)
        for fam, p, a, b in uni[y]:
            ra, rb = man.get((a, d)), man.get((b, d))
            if ra is None or rb is None:
                skipped[fam]["one_leg" if (ra is None) != (rb is None) else "neither"] += 1
                continue
            tasks.append({"date": d, "year": y, "family": fam, "type": ST24.pair_type(fam, a, b), "pair": p, "name": f"{a}/{b}",
                          "key": ST24.pair_key(a, b), "sym_a": a, "sym_b": b, "row_a": ra, "row_b": rb, "close_bar": cb[d]})
    return tasks, days, skipped


def dev_plan():
    tasks, days, skipped = plan(manifest(DEV_MANIFEST), DEV_WINDOW)
    if any(t["date"] > DEV_WINDOW[1] for t in tasks) or days[-1] > DEV_WINDOW[1]:
        raise Refused("a date after 2020 in the development plan")
    return tasks, days, skipped


def load(root, row, allow_sealed=False):
    """loader21.read behind the development-only guard of research 24."""
    path = os.path.join(root, row["file"])
    if not allow_sealed:
        if "sealed" in os.path.abspath(path).split(os.sep):
            raise Refused(f"{path}: a sealed path without the guard")
        if int(row["date"]) > int(DEV_WINDOW[1]):
            raise Refused(f"{row['sym']} {row['date']}: after the development window")
        if os.path.abspath(root) not in [os.path.abspath(r) for r in (DEV_ROOT,) + REGRESSION_ROOTS]:
            raise Refused(f"{root}: not a development folder")
    return read(path, row)
