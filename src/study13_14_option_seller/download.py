#!/usr/bin/env python3
"""Research 13 data steps (design memo 4a section 5; each step only with Kyle's permission for that step):
  python download.py --meta --go           S1: metadata only (information_schema; COUNT(*), min(date), max(date) per
                                           table, year and secid). No market-data row or value is queried.
  python download.py --stage S2 --go       dev SPX 1996 (secprd, vsurfd, opprcd) + zerocd dev
  python download.py --stage S3 --go       dev 1996-2012, SPX/NDX/RUT, + zerocd dev + calendar_dev
  python download.py --stage S4a --go      holdout secprd (3) -> SPX vsurfd -> SPX opprcd + zerocd holdout -> finalize
  python download.py --stage S4b --go      holdout NDX/RUT vsurfd -> opprcd
Without --go nothing connects. One WRDS connection (credentials from the PostgreSQL client configuration).

Writing (memo 5): .part -> fsync -> sha256 -> MANIFEST line (append, fsync) -> atomic rename. Resume: a file whose
sha256 equals its latest MANIFEST line is skipped; a file without one stops the run (nothing is overwritten or
deleted). Required tables are checked by name first. The first error or warning stops the run (errors.tsv; the
connection is rolled back); so does free disk below 15 GiB. Logs and stdout carry file names, row counts, bytes,
seconds, sha256 prefixes, the holdout end date and calendar day counts only."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd

import omdata as om

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
META_DIR = os.path.join(HERE, "data_manifests", "s1_meta")
MIN_FREE = 15 * 1024**3
ALL = ("SPX", "NDX", "RUT")
CODE_COLS = ["utc", "stage", "split", "download_sha256", "omdata_sha256", "memo_sha256", "wrds", "pandas", "numpy",
             "python"]


class Stop(Exception):
    pass


def years(split):
    lo, hi = om.SPLITS[split][2:]
    return list(range(lo, hi + 1))


def stage_units(stage):
    """(split, [(table, und, year)], finalize). zerocd units carry und = year = None."""
    if stage == "S2":
        return "dev", [("secprd", "SPX", 1996), ("vsurfd", "SPX", 1996), ("opprcd", "SPX", 1996),
                       ("zerocd", None, None)], False
    if stage == "S3":
        u = [(t, s, y) for t in ("secprd", "vsurfd", "opprcd") for s in ALL for y in years("dev")]
        return "dev", u + [("zerocd", None, None)], True
    if stage == "S4a":
        u = [("secprd", s, y) for s in ALL for y in years("holdout")]
        u += [(t, "SPX", y) for t in ("vsurfd", "opprcd") for y in years("holdout")]
        return "holdout", u + [("zerocd", None, None)], True
    if stage == "S4b":
        return "holdout", [(t, s, y) for t in ("vsurfd", "opprcd") for s in ("NDX", "RUT") for y in years("holdout")], False
    raise Stop(f"unknown stage {stage}")


def missing_tables(db, split, units):
    need = sorted({om.table_name(t, split, y).split(".", 1)[1] for t, _, y in units})
    names = ",".join(f"'{n}'" for n in need)
    have = set(om.query(db, f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{om.SCHEMA}' "
                            f"AND table_name IN ({names})")["table_name"].astype(str))
    return [n for n in need if n not in have]


def code_version(root, stage, split):
    def ver(p):
        try:
            return importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            return "absent"
    om.append_tsv(os.path.join(root, "CODE_VERSION.tsv"), CODE_COLS, [{
        "utc": om.utc(), "stage": stage, "split": split, "download_sha256": om.sha256(os.path.abspath(__file__)),
        "omdata_sha256": om.sha256(os.path.abspath(om.__file__)), "memo_sha256": om.MEMO_SHA, "wrds": ver("wrds"),
        "pandas": pd.__version__, "numpy": np.__version__, "python": platform.python_version()}])


def fetch_unit(db, table, split, und, year):
    t0 = time.time()
    sqls = om.unit_sql(table, split, und, year)
    frames = [om.query(db, q) for q in sqls]
    arrays, nulls = om.to_arrays(table, frames)
    om.check_dates(arrays, split, year)
    return arrays, nulls, len(sqls), time.time() - t0


def run(stage, db, base=None, log=print, free_fn=None):
    """Download one stage. Returns {'written', 'skipped'}; raises Stop on the first problem."""
    split, units, finalize = stage_units(stage)
    root = om.split_root(split, base)
    os.makedirs(root, exist_ok=True)
    free_fn = free_fn or (lambda p: shutil.disk_usage(p).free)
    try:
        miss = missing_tables(db, split, units)
    except Exception as e:
        om.append_tsv(os.path.join(root, "errors.tsv"), om.ERR_COLS, [{
            "table": "information_schema", "und": "", "year": "", "kind": type(e).__name__, "message": str(e)[:300],
            "utc": om.utc()}])
        raise Stop(f"table check: {type(e).__name__}: {str(e)[:200]}") from e
    if miss:
        raise Stop(f"missing tables: {', '.join(miss)}")
    code_version(root, stage, split)
    done = om.read_manifest(root)
    written = skipped = 0
    for table, und, year in units:
        rel = om.rel_file(table, split, und, year)
        path = os.path.join(root, rel)
        if os.path.exists(path):
            if done.get(rel, {}).get("sha256") == om.sha256(path):
                skipped += 1
                log(f"{rel} skipped (MANIFEST sha256 matches)")
                continue
            raise Stop(f"{rel}: file exists without a matching MANIFEST line (not overwritten)")
        if free_fn(root) < MIN_FREE:
            raise Stop(f"free disk below {MIN_FREE // 1024**3} GiB")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                arrays, nulls, nq, sec = fetch_unit(db, table, split, und, year)
                part, h, nbytes = om.write_npz_part(path, arrays)
        except Exception as e:
            om.append_tsv(os.path.join(root, "errors.tsv"), om.ERR_COLS, [{
                "table": table, "und": und, "year": year, "kind": type(e).__name__, "message": str(e)[:300],
                "utc": om.utc()}])
            raise Stop(f"{rel}: {type(e).__name__}: {str(e)[:200]}") from e
        d = arrays["date"]
        row = {"table": table, "und": und or "", "secid": om.UND.get(und, ""), "year": year or "", "file": rel,
               "rows": len(d), "n_dates": len(np.unique(d)), "date_min": int(d.min()) if len(d) else "",
               "date_max": int(d.max()) if len(d) else "", "bytes": nbytes, "sha256": h,
               "content_sha256": om.content_sha256(arrays), "schema_sha256": om.schema_sha256(arrays),
               "n_queries": nq, "fetch_sec": round(sec, 2), "written_utc": om.utc()}
        om.append_tsv(os.path.join(root, "MANIFEST.tsv"), om.MAN_COLS, [row])
        om.append_tsv(os.path.join(root, "null_counts.tsv"), om.NULL_COLS,
                      [{"table": table, "und": und or "", "year": year or "", "column": c, "n_null": n}
                       for c, n in nulls.items()])
        os.replace(part, path)
        written += 1
        log(f"{rel} written: rows {len(d)}, bytes {nbytes}, {sec:.1f}s, sha {h[:12]}")
    if finalize:
        finalize_split(split, root, log)
    return {"written": written, "skipped": skipped}


def finalize_split(split, root, log=print):
    """Calendar (and for holdout the end date) from this split's own files: date arrays and secprd has_close only."""
    man = om.read_manifest(root)
    ys = years(split)
    need = [om.rel_file("secprd", split, s, y) for s in ALL for y in ys]
    if split == "holdout":
        need += [om.rel_file(t, split, "SPX", y) for t in ("vsurfd", "opprcd") for y in ys]
    absent = [r for r in need if r not in man]
    if absent:
        raise Stop(f"finalize: {len(absent)} files not in MANIFEST")
    for r in need:
        if om.sha256(os.path.join(root, r)) != man[r]["sha256"]:
            raise Stop(f"finalize: {r} sha256 differs from MANIFEST")
    sec = {}
    for s in ALL:
        parts = [om.own_arrays(os.path.join(root, om.rel_file("secprd", split, s, y)), ("date", "has_close")) for y in ys]
        sec[s] = (np.concatenate([p["date"] for p in parts]), np.concatenate([p["has_close"] for p in parts]))
    lo, hi = (int(x.replace("-", "")) for x in om.SPLITS[split][:2])
    info = {"split": split, "bounds": list(om.SPLITS[split][:2]), "memo_sha256": om.MEMO_SHA, "utc": om.utc()}
    if split == "holdout":
        spx = {t: np.concatenate([om.own_arrays(os.path.join(root, om.rel_file(t, split, "SPX", y)), ("date",))["date"]
                                  for y in ys]) for t in ("opprcd", "vsurfd")}
        spx["secprd"] = sec["SPX"][0]
        hi = om.holdout_end(spx)
        info["holdout_end_date"] = hi
    days, has = om.build_calendar(sec, lo, hi)
    text = om.calendar_text(days, has)
    h = hashlib.sha256(text.encode()).hexdigest()
    cal = os.path.join(root, f"calendar_{split}.csv")
    if os.path.exists(cal):
        if om.sha256(cal) != h:
            raise Stop(f"finalize: calendar_{split}.csv exists with different content (not overwritten)")
    else:
        om.write_text(cal, text)
    info.update({"calendar_file": os.path.basename(cal), "calendar_sha256": h, "calendar_days": int(len(days)),
                 "files_in_manifest": len(man)})
    om.write_text(os.path.join(root, "SPLIT.json"), json.dumps(info, indent=1) + "\n")
    log(f"calendar_{split}: {len(days)} days, sha {h[:12]}" + (f"; holdout end date {hi}" if split == "holdout" else ""))
    return info


def meta_sql():
    """S1 statements: information_schema names and types, and COUNT(*), min(date), max(date) per table-year-secid (zerocd
    per calendar year). Nothing else is selected."""
    names = [f"{t}{y}" for t in ("opprcd", "vsurfd", "secprd") for y in range(1996, 2026)] + ["zerocd"]
    lst = ",".join(f"'{n}'" for n in names)
    out = [("tables", None, None, f"SELECT table_name FROM information_schema.tables WHERE table_schema = 'optionm' "
                                  f"AND table_name IN ({lst})"),
           ("columns", None, None, f"SELECT table_name, column_name, data_type, ordinal_position FROM "
                                   f"information_schema.columns WHERE table_schema = 'optionm' AND table_name IN ({lst})")]
    ids = ",".join(str(v) for v in om.UND.values())
    for t in ("opprcd", "vsurfd", "secprd"):
        for y in range(1996, 2026):
            out.append(("count", t, y, f"SELECT secid, count(*) AS n, min(date) AS dmin, max(date) AS dmax FROM "
                                       f"optionm.{t}{y} WHERE secid IN ({ids}) GROUP BY secid"))
    for y in range(1996, 2026):
        out.append(("count", "zerocd", y, f"SELECT count(*) AS n, min(date) AS dmin, max(date) AS dmax FROM optionm.zerocd "
                                          f"WHERE date BETWEEN '{y}-01-01' AND '{y}-12-31'"))
    return out


def run_meta(db, out_dir=META_DIR, log=print):
    """S1. Tables that do not exist are not queried (their names come from information_schema first)."""
    os.makedirs(out_dir, exist_ok=True)
    rows, t0 = [], time.time()
    stmts = meta_sql()
    tabs = om.query(db, stmts[0][3])
    have = set(tabs["table_name"].astype(str))
    cols = om.query(db, stmts[1][3])
    tabs.to_csv(os.path.join(out_dir, "tables.csv"), index=False)
    cols.to_csv(os.path.join(out_dir, "columns.csv"), index=False)
    for kind, t, y, q in stmts[2:]:
        name = "zerocd" if t == "zerocd" else f"{t}{y}"
        if name not in have:
            rows.append({"table": t, "year": y, "secid": "", "rows": 0, "dmin": "", "dmax": "", "exists": False})
            continue
        r = om.query(db, q)
        if t == "zerocd":
            rows.append({"table": t, "year": y, "secid": "", "rows": int(r.n.iloc[0]), "dmin": r.dmin.iloc[0],
                         "dmax": r.dmax.iloc[0], "exists": True})
        else:
            for s, n, a, b in zip(r.secid, r.n, r.dmin, r.dmax):
                rows.append({"table": t, "year": y, "secid": int(s), "rows": int(n), "dmin": a, "dmax": b, "exists": True})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "counts.csv"), index=False)
    log(f"S1: {len(have)} tables present, {len(cols)} column rows, {len(rows)} count rows, {time.time() - t0:.1f}s")
    return tabs, cols, pd.DataFrame(rows)


def connect():
    import wrds
    return wrds.Connection()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["S2", "S3", "S4a", "S4b"])
    ap.add_argument("--meta", action="store_true", help="S1 metadata only")
    ap.add_argument("--go", action="store_true", help="required; only after Kyle's permission for this step")
    a = ap.parse_args(argv)
    if not a.go:
        sys.exit("refused: needs --go, given only after Kyle's permission for this step")
    if bool(a.stage) == a.meta:
        sys.exit("refused: give exactly one of --stage or --meta")
    os.makedirs(LOG_DIR, exist_ok=True)
    logf = open(os.path.join(LOG_DIR, f"{'S1' if a.meta else a.stage}_{time.strftime('%Y%m%d_%H%M%S')}.log"), "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    db = connect()
    try:
        if a.meta:
            run_meta(db, log=log)
        else:
            log(f"stage {a.stage} done: {run(a.stage, db, log=log)}")
    except Stop as e:
        log(f"STOP: {e}")
        sys.exit(1)
    finally:
        db.close()
        log("connection closed")


if __name__ == "__main__":
    main()
