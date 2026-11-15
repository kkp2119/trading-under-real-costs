#!/usr/bin/env python3
"""Research 22 bar loader (SPEC22 1, 3; step-2 prompt 1). By default only development sources are reachable:
  stocks   the existing universe 2015-2020 through study15_ml_execution/loader15 (its MANIFEST rows, universe lists and calendars;
           research 15 train bars 2015-2018, research 12 development bars 2019-2020)
  SPY      the reference leg 2015-2020 from research 21's development bars ($DATA_DIR/research21_data/dev/bars_1s, rows of
           study21_more_pairs/DATA_MANIFEST21.tsv)
The sealed folders (research 22's new universe C and research 21's sealed SPY bars) are named only in run_test22.py, which
hands their paths to plan_files().
Every file is checked as loader15.load does: sha256 against its manifest row, the 21 int32 columns of 23,400 rows and the
volume identity. Sizes: from 2025-11-03 the displayed sizes bsz and asz are divided by 100 (stocks and SPY; SPEC22 1,
research 21's loader21); a value that is not a multiple of 100 stops the run. Prices, traded volumes and the files stay
as they are. A task carries its own paths and hashes, so a worker process needs no module state."""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "study15_ml_execution"))
import loader15 as L15  # noqa: E402

COLS, NSEC = L15.COLS, L15.NSEC
SIZE_FROM = "20251103"                                             # first day whose displayed sizes are 100 x the shares
REF = "SPY"
DEV_SPY_ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research21_data/dev/bars_1s")
DEV_SPY_MANIFEST = os.path.join(PROJ, "study21_more_pairs", "DATA_MANIFEST21.tsv")
DEV_LAST = "20201231"


class Refused(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def read_npz(path, want, label):
    """21 columns as int64 (loader15.load checks: sha256, columns, dtype, shape, volume identity)."""
    if sha256(path) != want:
        raise Refused(f"{label}: sha256 differs from the manifest")
    with np.load(path, allow_pickle=False) as z:
        if sorted(z.files) != sorted(COLS):
            raise Refused(f"{label}: columns {sorted(z.files)}")
        b = {}
        for c in COLS:
            a = z[c]
            if a.dtype != np.int32 or a.shape != (NSEC,):
                raise Refused(f"{label}: {c} {a.dtype} {a.shape}")
            b[c] = a.astype(np.int64)
    if not np.array_equal(b["vol"], b["vol_bid"] + b["vol_ask"] + b["vol_in"] + b["vol_unc"]):
        raise Refused(f"{label}: vol identity fails")
    return b


def correct(b, date, label=""):
    """Displayed sizes in shares: bsz and asz / 100 for a file dated 2025-11-03 or later; nothing else changes."""
    if date < SIZE_FROM:
        return b
    for c in ("bsz", "asz"):
        if np.any(b[c] % 100 != 0):
            raise Refused(f"{label}: {c} holds a value that is not a multiple of 100 on {date}")
        b[c] = b[c] // 100
    return b


def close_bar(t):
    h, m = (int(x) for x in t.split(":"))
    return NSEC - ((16 - h) * 3600 - m * 60)


def ref_rows(manifest):
    """{YYYYMMDD: manifest row} of the reference leg in a research 21 manifest (sym, date, file, rows, bytes, sha256)."""
    m = pd.read_csv(manifest, sep="\t", dtype=str, keep_default_na=False)
    m = m[m.sym == REF].drop_duplicates("date", keep="last")
    return {r["date"]: r for r in m.to_dict("records")}


def _task(kind, year, sym, date, cb, path, sha, ref, ref_root):
    r = ref.get(date)
    return {"kind": kind, "year": int(year), "sym": sym, "date": date, "close_bar": int(cb), "path": path, "sha256": sha,
            "ref_path": os.path.join(ref_root, r["file"]) if r else None, "ref_sha256": r["sha256"] if r else None}


def dev_plan():
    """(tasks, {year: calendar days}, info): the existing universe 2015-2020 in loader15's order, each with its SPY file of
    the day (None when research 21 holds none). Nothing after 2020-12-31 is reachable."""
    ref = ref_rows(DEV_SPY_MANIFEST)
    tasks, cal = [], {}
    for y in L15.YEARS:
        man, cb = L15.manifest(y), L15.close_bars(y)
        root = L15.SOURCES[L15.source_of(y)][0]
        tasks += [_task("dev", y, s, d, cb[d], os.path.join(root, r["file"]), r["sha256"], ref, DEV_SPY_ROOT) for (s, d), r in sorted(man.items())]
        cal[int(y)] = sorted(cb)
    if any(t["date"] > DEV_LAST for t in tasks) or any(d > DEV_LAST for d in ref):
        raise Refused("a development source holds a date after 2020-12-31")
    no_ref = sorted({t["date"] for t in tasks if t["ref_path"] is None})
    info = {"tasks": len(tasks), "days": int(sum(len(v) for v in cal.values())), "ref_files": len(ref), "days_without_ref": no_ref,
            "symbol_days_without_ref": int(sum(t["ref_path"] is None for t in tasks))}
    return tasks, cal, info


def plan_files(manifest, exclude, calendar, universe, window, root, ref_manifest, ref_root):
    """(tasks, {year: calendar days}, info) for bars listed in a manifest written by check22.py (sym, date, file, rows,
    bytes, sha256): universe symbols of the year only, days of the calendar inside the window, minus the exclude list.
    Called by run_test22.py with the sealed paths (and by the tests with stand-ins)."""
    m = pd.read_csv(manifest, sep="\t", dtype=str, keep_default_na=False).drop_duplicates(["sym", "date"], keep="last")
    ex = pd.read_csv(exclude, sep="\t", dtype=str, keep_default_na=False)
    exk = set(zip(ex.sym, ex.date))
    c = pd.read_csv(calendar, dtype=str)
    cb = {d.replace("-", ""): close_bar(t) for d, t in zip(c.date, c.close_time)}
    days = sorted(d for d in cb if window[0] <= d <= window[1])
    uni = {y: set(pd.read_csv(universe.format(y=y), sep="\t", dtype=str).sym) for y in sorted({d[:4] for d in days})}
    ref = {d: r for d, r in ref_rows(ref_manifest).items() if window[0] <= d <= window[1]}
    rows = [r for r in m.to_dict("records") if r["date"] in cb and window[0] <= r["date"] <= window[1] and r["sym"] in uni[r["date"][:4]]]
    kept = [r for r in rows if (r["sym"], r["date"]) not in exk]
    tasks = [_task("file", r["date"][:4], r["sym"], r["date"], cb[r["date"]], os.path.join(root, r["file"]), r["sha256"], ref, ref_root)
             for r in sorted(kept, key=lambda r: (r["date"][:4], r["sym"], r["date"]))]
    cal = {}
    for d in days:
        cal.setdefault(int(d[:4]), []).append(d)
    info = {"tasks": len(tasks), "days": len(days), "manifest_rows": int(len(m)), "rows_in_window_and_universe": len(rows),
            "excluded_rows": len(rows) - len(kept), "ref_files": len(ref), "days_without_ref": sorted({t["date"] for t in tasks if t["ref_path"] is None}),
            "symbol_days_without_ref": int(sum(t["ref_path"] is None for t in tasks))}
    return tasks, cal, info


def load(t):
    """(stock bars, SPY bars or None): checked and with sizes in shares."""
    b = correct(read_npz(t["path"], t["sha256"], f"{t['sym']} {t['date']}"), t["date"], f"{t['sym']} {t['date']}")
    r = None
    if t["ref_path"] is not None:
        r = correct(read_npz(t["ref_path"], t["ref_sha256"], f"{REF} {t['date']}"), t["date"], f"{REF} {t['date']}")
    return b, r
