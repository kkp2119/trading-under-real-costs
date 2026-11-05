#!/usr/bin/env python3
"""Research 13 data layer (design memo 4a, research13/design/ENGINE_DESIGN_13A_DATA.md sha256 4f51396d..., Kyle approval
2026-10-28 13:27 ET; SPEC spec13-v1). OptionMetrics tables -> typed npz files, one per (table, underlying, year), split
by date into data_dev (<= 2012-12-31) and data_holdout (>= 2013-01-01) under $DATA_DIR/research13_data.

Per split folder: MANIFEST.tsv (one line per file, appended and fsynced; the latest line for a file wins),
null_counts.tsv, errors.tsv, CODE_VERSION.tsv, SPLIT.json and calendar_<split>.csv. Holdout files are read only when
research13/results/HOLDOUT_UNLOCKED exists, checked before any holdout path is touched (Store); the downloader's own
finalize step is the one exception (memo 3.3 item 6: it reads the date arrays, and secprd's has_close, of the files it
has just written). No price, quote, IV or Greek value is printed or logged by this module.

query() and concat_raw() are copied from research12/bars1s/download.py (sha256 4b7bcf73..., drafts A4 and A5) without
change; check_unlock follows research12/loader.py (sha256 f2c49933...)."""
import datetime as dt
import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("R13_DATA", os.path.join(os.environ.get("DATA_DIR", "data"), "research13_data"))
RESULTS = os.environ.get("R13_RESULTS", os.path.join(HERE, "results"))
UNLOCK = "HOLDOUT_UNLOCKED"
SCHEMA = "optionm"
MEMO_SHA = "4f51396d17c88b5cfd82f2977475c02763e0931f5c354dc8beccadaf6280e6a5"
SPLITS = {"dev": ("1996-01-01", "2012-12-31", 1996, 2012), "holdout": ("2013-01-01", "2025-12-31", 2013, 2025)}
UND = {"SPX": 108105, "NDX": 102480, "RUT": 102434}
CAL_BIT = {"SPX": 1, "NDX": 2, "RUT": 4}
# (column, kind): date -> int32 YYYYMMDD (no NULL); cp -> int8 C=+1 P=-1 (no NULL); f8 -> float64 (NULL -> NaN);
# id -> int64 (no NULL, integral); flag -> int8 (NULL -> -1); i2/i4 -> int16/int32 (no NULL, integral);
# code -> int16 codes into <column>__codes, sorted distinct values (NULL -> -1)
COLUMNS = {
    "opprcd": [("date", "date"), ("exdate", "date"), ("cp_flag", "cp"), ("strike_price", "f8"), ("best_bid", "f8"),
               ("best_offer", "f8"), ("impl_volatility", "f8"), ("delta", "f8"), ("forward_price", "f8"),
               ("open_interest", "f8"), ("optionid", "id"), ("am_settlement", "flag"), ("expiry_indicator", "code"),
               ("root", "code"), ("contract_size", "f8")],
    "vsurfd": [("date", "date"), ("days", "i2"), ("delta", "i2"), ("cp_flag", "cp"), ("impl_volatility", "f8"),
               ("impl_strike", "f8")],
    "secprd": [("date", "date"), ("close", "f8"), ("return", "f8")],
    "zerocd": [("date", "date"), ("days", "i4"), ("rate", "f8")],
}
ORDER = {"opprcd": "date, exdate, cp_flag, strike_price, optionid", "vsurfd": "date, days, delta, cp_flag",
         "secprd": "date", "zerocd": "date, days"}
MAN_COLS = ["table", "und", "secid", "year", "file", "rows", "n_dates", "date_min", "date_max", "bytes", "sha256",
            "content_sha256", "schema_sha256", "n_queries", "fetch_sec", "written_utc"]
NULL_COLS = ["table", "und", "year", "column", "n_null"]
ERR_COLS = ["table", "und", "year", "kind", "message", "utc"]


class SplitError(Exception):
    pass


class DataError(Exception):
    pass


class Refused(Exception):
    pass


def utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def split_root(split, base=None):
    if split not in SPLITS:
        raise SplitError(split)
    return os.path.join(base or BASE, f"data_{split}")


def table_name(table, split, year=None):
    """Schema-qualified table. Yearly tables only for the split's years (a dev run cannot name a 2013+ table)."""
    if split not in SPLITS:
        raise SplitError(split)
    if table == "zerocd":
        return f"{SCHEMA}.zerocd"
    lo, hi = SPLITS[split][2:]
    if table not in COLUMNS or not isinstance(year, int) or not lo <= year <= hi:
        raise SplitError(f"{table} {year} is outside split {split} ({lo}-{hi})")
    return f"{SCHEMA}.{table}{year}"


def rel_file(table, split, und=None, year=None):
    return f"zerocd/zerocd_{split}.npz" if table == "zerocd" else f"{table}/{table}_{und}_{year}.npz"


def _qcol(c):
    return f'"{c}"' if c == "return" else c


def unit_sql(table, split, und=None, year=None):
    """The SELECTs of one file: opprcd one per month, the others one. Every query also carries the split's bounds."""
    d0, d1 = SPLITS[split][:2]
    cols = ", ".join(_qcol(c) for c, _ in COLUMNS[table])
    src = table_name(table, split, year)
    if table == "zerocd":
        return [f"SELECT {cols} FROM {src} WHERE date >= '{d0}' AND date <= '{d1}' ORDER BY {ORDER[table]}"]
    if und not in UND:
        raise SplitError(f"unknown underlying {und}")
    if table == "opprcd":
        spans = [(f"{year}-{m:02d}-01", (dt.date(year + m // 12, m % 12 + 1, 1) - dt.timedelta(days=1)).isoformat())
                 for m in range(1, 13)]
    else:
        spans = [(f"{year}-01-01", f"{year}-12-31")]
    return [f"SELECT {cols} FROM {src} WHERE secid = {UND[und]} AND date BETWEEN '{a}' AND '{b}' "
            f"AND date >= '{d0}' AND date <= '{d1}' ORDER BY {ORDER[table]}" for a, b in spans]


def query(db, sql):
    """db.raw_sql(sql). On failure the connection is rolled back first (wrds raw_sql does not, and a failed statement
    leaves the transaction aborted for every later query on that connection), then the original error is raised."""
    try:
        return db.raw_sql(sql)
    except Exception:
        try:
            db.connection.rollback()
        except Exception:
            pass
        raise


def concat_raw(frames):
    """pd.concat for raw_sql results (draft A4); every concat of query results goes through here. raw_sql gives a 0-row
    result object columns and an all-NULL column a value-inferred dtype (string), and concatenating either with typed
    columns warns in pandas 2.2 (FutureWarning; warnings are failures here). 0-row frames are dropped and an all-NULL
    column takes the dtype the column has where it holds values. Only value-free columns change dtype, which is the
    result pandas 2.2 gives with the warning. All frames empty -> the last one."""
    fr = [f for f in frames if len(f)]
    if not fr:
        return pd.concat(frames[-1:], ignore_index=True)
    typed = {}
    for f in fr:
        for c in f.columns:
            if c not in typed and not f[c].isna().all():
                typed[c] = f[c].dtype
    fr = [f.astype({c: typed[c] for c in f.columns if c in typed and f[c].dtype != typed[c] and f[c].isna().all()})
          for f in fr]
    return pd.concat(fr, ignore_index=True)


def _numeric(col, name):
    try:
        return pd.to_numeric(col, errors="raise").astype("float64").to_numpy()
    except (ValueError, TypeError) as e:
        raise DataError(f"{name}: non-numeric value ({type(e).__name__})") from None


def _integral(v, name, allow_nan=False):
    bad = ~np.isfinite(v) if not allow_nan else np.isinf(v)
    if bad.any() or (np.nan_to_num(v) != np.round(np.nan_to_num(v))).any():
        raise DataError(f"{name}: NULL or non-integral value")
    return v


def to_arrays(table, frames):
    """Query results of one file -> ({column: typed array, code tables}, {column: NULL count}). Explicit dtype per
    column (memo 2.1); a value that does not fit the column's kind stops the file (DataError)."""
    df = concat_raw(frames)
    out, nulls = {}, {}
    for c, kind in COLUMNS[table]:
        if c not in df.columns:
            raise DataError(f"{table}: column {c} missing from the query result")
        col = df[c]
        isn = col.isna().to_numpy()
        nulls[c] = int(isn.sum())
        if kind == "date":
            if isn.any():
                raise DataError(f"{c}: NULL date")
            out[c] = (pd.to_datetime(col).dt.strftime("%Y%m%d").astype(np.int32).to_numpy() if len(col)
                      else np.zeros(0, np.int32))
        elif kind == "cp":
            m = col.map({"C": 1, "P": -1, "c": 1, "p": -1})
            if m.isna().any():
                raise DataError(f"{c}: value other than C/P")
            out[c] = m.to_numpy(np.int8)
        elif kind == "f8":
            out[c] = _numeric(col, c)
        elif kind in ("id", "i2", "i4"):
            v = _integral(_numeric(col, c), c)
            out[c] = v.astype({"id": np.int64, "i2": np.int16, "i4": np.int32}[kind])
        elif kind == "flag":
            v = _integral(_numeric(col, c), c, allow_nan=True)
            out[c] = np.where(np.isnan(v), -1, v).astype(np.int8)
        elif kind == "code":
            s = col.astype(object).where(~isn, None)
            uniq = sorted({str(x) for x in s if x is not None})
            idx = {u: i for i, u in enumerate(uniq)}
            out[c] = np.array([-1 if x is None else idx[str(x)] for x in s], dtype=np.int16)
            out[c + "__codes"] = np.array(uniq, dtype=str) if uniq else np.zeros(0, dtype="<U1")
    if table == "secprd":
        out["has_close"] = ~np.isnan(out["close"])        # calendar input, so finalize never reads close values
    return out, nulls


def check_dates(arrays, split, year=None):
    """Every date inside the split, and inside the file's year for yearly tables; else DataError (nothing written)."""
    d = arrays["date"]
    lo, hi = (int(x.replace("-", "")) for x in SPLITS[split][:2])
    if year is not None:
        lo, hi = max(lo, year * 10000 + 101), min(hi, year * 10000 + 1231)
    if len(d) and (d.min() < lo or d.max() > hi):
        raise DataError(f"date outside {lo}-{hi}")


def content_sha256(arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def schema_sha256(arrays):
    return hashlib.sha256(json.dumps(sorted((k, np.asarray(v).dtype.str) for k, v in arrays.items())).encode()).hexdigest()


def write_npz_part(path, arrays):
    """Write path + '.part' (overwriting an earlier .part), fsync, return (part path, sha256, bytes)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    with open(part, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    return part, sha256(part), os.path.getsize(part)


def append_tsv(path, cols, rows):
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]).replace("\t", " ").replace("\n", " ") for c in cols) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_manifest(root):
    p = os.path.join(root, "MANIFEST.tsv")
    if not os.path.exists(p):
        return {}
    m = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    return {r.file: r._asdict() for r in m.itertuples(index=False)}            # latest line wins


def check_unlock(split, results_dir=None):
    if split == "holdout" and not os.path.exists(os.path.join(results_dir or RESULTS, UNLOCK)):
        raise Refused(f"{split} is locked: no {UNLOCK}")


class Store:
    """Read side of one split. Holdout: refused unless the unlock file exists (checked before any holdout path is
    touched). Every file is checked against its MANIFEST sha256; every path read is logged in .read."""

    def __init__(self, split, base=None, results_dir=None):
        check_unlock(split, results_dir)
        self.split, self.root = split, split_root(split, base)
        self.meta = read_manifest(self.root)
        self.read = [os.path.join(self.root, "MANIFEST.tsv")]

    def load(self, rel):
        r = self.meta.get(rel)
        if r is None:
            raise Refused(f"{rel}: not in MANIFEST")
        path = os.path.join(self.root, rel)
        self.read.append(path)
        if sha256(path) != r["sha256"]:
            raise Refused(f"{rel}: sha256 differs from MANIFEST")
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


def own_arrays(path, keys):
    """Downloader finalize only (memo 3.3 item 6): the named arrays of a file this run's split folder holds."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def build_calendar(secprd, lo, hi):
    """secprd: {und: (dates int32, has_close bool)} -> (dates int32 sorted, has uint8 bitmask SPX=1 NDX=2 RUT=4) for the
    union of dates with a close, restricted to lo..hi (YYYYMMDD ints; holdout hi = the SPX holdout end date)."""
    has = {}
    for und, (d, ok) in secprd.items():
        for x in np.asarray(d)[np.asarray(ok, bool)]:
            if lo <= x <= hi:
                has[int(x)] = has.get(int(x), 0) | CAL_BIT[und]
    days = np.array(sorted(has), dtype=np.int32)
    return days, np.array([has[int(x)] for x in days], dtype=np.uint8)


def holdout_end(spx_dates):
    """SPEC 2.3: the last date on which SPX has opprcd, vsurfd and secprd rows. spx_dates: {table: array of dates}."""
    common = set.intersection(*(set(np.asarray(spx_dates[t]).tolist()) for t in ("opprcd", "vsurfd", "secprd")))
    if not common:
        raise DataError("no date with all three SPX tables")
    return max(common)


def calendar_text(days, has):
    return "date,has\n" + "".join(f"{int(d)},{int(b)}\n" for d, b in zip(days, has))


def write_text(path, text):
    """Atomic write (.part, fsync, rename); returns the file's sha256."""
    tmp = path + ".part"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return sha256(path)
