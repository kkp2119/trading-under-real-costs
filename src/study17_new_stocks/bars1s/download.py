#!/usr/bin/env python3
"""Research 17 step 1 (copy of research15/bars1s/download.py; changed lines marked "R17"): new-universe sealed bars
2021-01-04 .. END into $DATA_DIR/research17_data/sealed/bars_1s_new_stocks/<YYYY>/ (universe research17/universe17_<Y>.tsv,
calendar research17/calendar17.csv). Quality checks only; no quote-based statistic is computed. Research 15 docstring follows.
Research 15 step 1 (copy of research12/bars1s/download.py; changed lines marked "R15"): WRDS TAQ -> 1-second bars,
one symbol-day per file, same bar definition (bars.py unchanged). R15: one split, train 2015-01-02 .. 2018-12-31, stock
mode only in use, base $DATA_DIR/research15_data, universe research15/universe15_<Y>.tsv, calendar research15/calendar15.csv.
Research 12 docstring follows. Run only with Kyle's permission per split.
  caffeinate -ims python download.py --split dev --syms IWM,DIA,... --go
Without --go it refuses and does not connect.

Splits by date only (Kyle 2026-10-27), each its own folder and MANIFEST under $DATA_DIR/research12_data/:
  bars_1s_dev      2019-01-02 .. 2020-12-31
  bars_1s_holdout  2021-01-04 .. 2026-09-30   (files never opened; values never printed)
  bars_1s_reserve  2026-10-01 .. 2026-10-23
  -> bars_1s_<split>/<YYYY>/<SYM>_<YYYYMMDD>.npz
Per symbol-day: complete_nbbo (session rows + the last 500 rows before 09:30:00 for the opening state) and ctm (session
rows) are read into memory, reduced to bars (bars.py) and only the bars are written. Writing: .part -> sha256 ->
MANIFEST line (append, fsync) -> atomic rename. Resume: an existing file whose sha256 equals its latest MANIFEST line is
skipped; an existing file without a matching line stops the run (nothing is overwritten or deleted). Trading days come
from table names only (complete_nbbo_YYYYMMDD and ctm_YYYYMMDD both present). A symbol-day that fails (error or warning)
is written to <split>/errors.tsv and not retried; the run continues and stops when failures exceed 20 or distinct error
kinds exceed 3 (Kyle 2026-10-27). It also stops when free disk space falls below 15 GiB. NULL rows: SPEC_AMENDMENTS
draft A3 (bars.py docstring); their counts per symbol-day go to <split>/null_counts.tsv (NULL_COLS, Kyle 2026-10-27):
quote rows with any NULL price/size among the pre-open rows read (last 500 before 09:30) and among the session rows,
and session trade rows with NULL price, size, tr_corr, tr_scond. The MANIFEST format is unchanged.
Connections: 3 (Kyle 2026-10-27; the earlier calibration measured about 2.5x for 3 connections, so `phased` is no longer
used by main). No price, size or return value is printed or logged: counts, seconds, bytes and hashes only.
Draft A5 (Kyle 2026-10-27): stock mode drops calendar days without a complete_nbbo, ctm or mastm table (table names only;
skipped_absent.tsv gives the reason), and a failed query rolls its connection back before the error is raised (wrds
raw_sql does not), so one failure does not fail the following queries on that connection."""
import argparse
import datetime as dt
import hashlib
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import warnings

import numpy as np
import pandas as pd

import bars

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.environ.get("DATA_DIR", "data"), "research17_data/sealed")               # R17
SPLITS = {"new": ("20210104", "20261231")}                                    # R17: calendar17.csv bounds the days
LOG_DIR = os.path.join(HERE, "logs")
N_CONN = 3
CALIB = 15
PHASE_SEC = 3600
MIN_SPEEDUP = 1.5
MIN_FREE = 15 * 1024**3
MAX_FAIL, MAX_KINDS = 20, 3
ERR_COLS = ["sym", "date", "kind", "message", "utc"]
NULL_COLS = ["sym", "date", "q_null_pre", "q_null_session", "t_null_price", "t_null_size", "t_null_corr", "t_null_scond",
             "source", "written_utc"]
PREOPEN_ROWS = 500
SYM_RE = re.compile(r"^[A-Z]{1,6}$")
MAN_COLS = ["sym", "date", "file", "rows", "bytes", "sha256", "fetch_sec", "build_sec", "n_quote_rows", "n_quote_session",
            "n_quote_valid_session", "n_quote_code_excluded", "n_quote_crossed", "n_quote_preopen", "init_state",
            "first_state_sec", "n_trade_rows", "n_trade_session", "n_trade_valid", "n_trade_excluded", "excl_by_code",
            "last_trade_sec", "n_subtick_trade", "n_subtick_quote", "written_utc"]

QCOLS = "time_m, time_m_nano, best_bid, best_bidsizeshares, best_ask, best_asksizeshares, qu_cond, qu_source, natbbo_ind"
NBBO_SQL = """select """ + QCOLS + """
from {lib}.complete_nbbo_{d}
where sym_root = '{s}' and sym_suffix is null and time_m >= '09:30:00' and time_m < '16:00:00'"""
PRE_SQL = """select """ + QCOLS + """
from {lib}.complete_nbbo_{d}
where sym_root = '{s}' and sym_suffix is null and time_m < '09:30:00'
order by time_m desc, time_m_nano desc limit {n}"""
TRADE_SQL = """select time_m, time_m_nano, price, size, tr_scond, tr_corr
from {lib}.ctm_{d}
where sym_root = '{s}' and sym_suffix is null and time_m >= '09:30:00' and time_m < '16:00:00'"""
DAYS_SQL = """select table_schema, table_name from information_schema.tables
where table_schema in ({schemas}) and table_name ~ '^(complete_nbbo|ctm)_[0-9]{{8}}$'"""
GAP_KINDS = ("complete_nbbo", "ctm", "mastm")
TABLES_SQL = """select table_schema, table_name from information_schema.tables
where table_schema in ({schemas}) and table_name ~ '^(complete_nbbo|ctm|mastm)_[0-9]{{8}}$'"""


class Stop(Exception):
    pass


def split_of(day):
    for name, (a, b) in SPLITS.items():
        if a <= day <= b:
            return name
    raise Stop(f"{day} is outside every split")


def split_root(base, day, suffix=""):
    return os.path.join(base, f"bars_1s_{split_of(day)}{suffix}")


# ---- stock mode (Kyle 2026-10-27): yearly universe x calendar days, master presence checked per date ----
STOCKS = "_stocks"
UNIVERSE_DIR = os.path.dirname(HERE)                          # research17/ (R17)
CALENDAR = os.path.join(UNIVERSE_DIR, "calendar17.csv")                       # R17
ABSENT_COLS = ["sym", "date", "reason", "utc"]


def read_universe(year, universe_dir=UNIVERSE_DIR):
    p = os.path.join(universe_dir, f"universe17_{year}.tsv")                   # R17
    if not os.path.exists(p):
        raise Stop(f"missing {p}")
    syms = pd.read_csv(p, sep="\t", dtype=str).sym.tolist()
    if not all(SYM_RE.match(x) for x in syms):
        raise Stop(f"{p}: symbols must match ^[A-Z]{{1,6}}$")
    return syms


def split_days(split, years, calendar=CALENDAR):
    """Calendar days (YYYYMMDD) of `split` in `years`."""
    lo, hi = SPLITS[split]
    days = [d.replace("-", "") for d in pd.read_csv(calendar, dtype=str).date]
    return [d for d in days if lo <= d <= hi and int(d[:4]) in set(years)]


def stock_tasks(split, years, present_fn, universe_dir=UNIVERSE_DIR, calendar=CALENDAR, gaps=None):
    """(tasks, absent): for every calendar day of `split` in `years`, the universe_<year> symbols present in that day's
    master (present_fn(day, syms) -> set) become tasks; the others are absent (no file; skipped_absent.tsv). Days in
    `gaps` ({day: [missing tables]}, table_gaps) are neither tasks nor master-checked: all their symbol-days are absent
    with the reason 'no <tables> table' (draft A5; SPEC 2.1 counts 2026-02-24 as a skipped day)."""
    days = split_days(split, years, calendar)
    uni = {y: read_universe(y, universe_dir) for y in sorted({int(d[:4]) for d in days})}
    gaps = gaps or {}
    tasks, absent = [], []
    for d in days:
        syms = uni[int(d[:4])]
        if d in gaps:
            absent.extend((x, d, f"no {'/'.join(gaps[d])} table") for x in syms)
            continue
        pres = present_fn(d, syms)
        for x in syms:
            (tasks if x in pres else absent).append((x, d))
    return tasks, absent


def record_absent(base, absent, suffix=STOCKS, log=print):
    """Append absent (sym, day[, reason]) to <split>/skipped_absent.tsv unless already listed (reason 'not in mastm'
    when not given). Not an error."""
    n = 0
    for root in sorted({split_root(base, t[1], suffix) for t in absent}):
        os.makedirs(root, exist_ok=True)
        p = os.path.join(root, "skipped_absent.tsv")
        have = set()
        if os.path.exists(p):
            h = pd.read_csv(p, sep="\t", dtype=str)
            have = set(zip(h.sym, h.date))
        new = [t for t in absent if split_root(base, t[1], suffix) == root and (t[0], t[1]) not in have]
        with open(p, "a") as f:
            if os.path.getsize(p) == 0:
                f.write("\t".join(ABSENT_COLS) + "\n")
            utc = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.writelines(f"{t[0]}\t{t[1]}\t{t[2] if len(t) > 2 else 'not in mastm'}\t{utc}\n" for t in new)
        n += len(new)
    log(f"absent from the master (no file, not an error): {len(absent)} symbol-days, newly listed {n}")
    return n


def wrds_presence(db):
    def present(day, syms):
        q = f"select symbol_15 from taqm_{day[:4]}.mastm_{day} where symbol_15 in ({','.join(repr(x) for x in syms)})"
        return set(query(db, q).symbol_15)
    return present


def table_gaps(db, days):
    """{day: [missing tables]} for the days (YYYYMMDD) without a complete_nbbo, ctm or mastm table, from table names
    only (draft A5). For complete_nbbo and ctm this is trading_days' rule."""
    if not days:
        return {}
    t = query(db, TABLES_SQL.format(schemas=",".join(f"'taqm_{y}'" for y in sorted({d[:4] for d in days}))))
    names = [str(x) for x in t["table_name"]]
    have = {(x.rsplit("_", 1)[0], x[-8:]) for x in names}
    return {d: miss for d in days if (miss := [k for k in GAP_KINDS if (k, d) not in have])}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def to_ns(time_m, nano):
    """time_m (datetime.time or 'HH:MM:SS[.ffffff]') + time_m_nano -> int64 ns since midnight."""
    s = pd.Series(time_m)
    s = s.map(lambda v: v.isoformat() if hasattr(v, "isoformat") else str(v))
    return pd.to_timedelta(s).to_numpy().astype("int64") + np.asarray(nano, dtype=np.int64)


def _num(col, dtype):
    """Nullable price / size column -> numpy, NULL -> 0 (the row then fails the price / size rules)."""
    return col.astype("Float64" if dtype is float else "Int64").fillna(0).to_numpy(dtype)


def _nano(col):
    if col.isna().any():
        raise ValueError(f"NULL time_m_nano in {int(col.isna().sum())} rows (no rule in A3)")
    return col.to_numpy(np.int64)


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


def day_bars(nbbo, pre, trades):
    """DataFrames from the three queries -> (bars, stats). No value is printed. NULL rules: SPEC_AMENDMENTS draft A3."""
    q = concat_raw([pre, nbbo])
    qt = to_ns(q["time_m"], _nano(q["time_m_nano"]))
    tt = to_ns(trades["time_m"], _nano(trades["time_m_nano"]))
    qnull = q[["best_bid", "best_ask", "best_bidsizeshares", "best_asksizeshares"]].isna().any(axis=1).to_numpy()
    tnull = (trades["price"].isna() | trades["size"].isna()).to_numpy()
    qpre, qses = qt < bars.T0 * bars.NS, (qt >= bars.T0 * bars.NS) & (qt < bars.T1 * bars.NS)
    tses = (tt >= bars.T0 * bars.NS) & (tt < bars.T1 * bars.NS)
    corr = trades["tr_corr"].map(lambda v: "NULL" if pd.isna(v) else str(v).zfill(2)).to_numpy(object)
    sc = trades["tr_scond"].fillna("").astype(str).to_numpy(object)
    b, st = bars.build_bars(qt, _num(q["best_bid"], float), _num(q["best_ask"], float),
                            _num(q["best_bidsizeshares"], np.int64), _num(q["best_asksizeshares"], np.int64),
                            bars.quote_ok(q["qu_cond"].fillna("").to_numpy(object), q["qu_source"].fillna("").to_numpy(object),
                                          q["natbbo_ind"].fillna("").astype(str).to_numpy(object)),
                            tt, _num(trades["price"], float), _num(trades["size"], np.int64),
                            bars.trade_ok(corr, sc) & ~tnull, bars.odd_lot(sc))
    st["excl_by_code"] = json.dumps(bars.trade_exclusions(tt, corr, sc, tnull), separators=(",", ":"))
    st["n_quote_null"], st["n_trade_null"] = int(qnull.sum()), int(tnull.sum())
    st["null_counts"] = {"q_null_pre": int((qnull & qpre).sum()), "q_null_session": int((qnull & qses).sum()),
                         **{f"t_null_{k}": int((trades[c].isna().to_numpy() & tses).sum())
                            for k, c in (("price", "price"), ("size", "size"), ("corr", "tr_corr"), ("scond", "tr_scond"))}}
    return b, st


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


NANO_SQL = """select table_name from information_schema.columns where table_schema = '{lib}' and column_name = 'time_m_nano'"""


def no_nano(sql):
    """R15: the query for a table without time_m_nano (complete_nbbo before 2018, ctm before 2016): time_m alone, the
    nanosecond part 0 (ns since midnight = time_m + 0, as in bars.py)."""
    return sql.replace("time_m, time_m_nano,", "time_m, 0 as time_m_nano,").replace(", time_m_nano desc", "")


def make_fetch(db):
    """fetch(sym, day) on one connection, returning the three DataFrames."""
    nano = {}                                                                     # R15: tables with time_m_nano, per year

    def fetch(sym, day):
        lib, d = f"taqm_{day[:4]}", day
        if lib not in nano:
            nano[lib] = set(query(db, NANO_SQL.format(lib=lib)).table_name)
        qn = (lambda s: s) if f"complete_nbbo_{d}" in nano[lib] else no_nano
        tn = (lambda s: s) if f"ctm_{d}" in nano[lib] else no_nano
        return (query(db, qn(NBBO_SQL).format(lib=lib, d=d, s=sym)), query(db, qn(PRE_SQL).format(lib=lib, d=d, s=sym, n=PREOPEN_ROWS)),
                query(db, tn(TRADE_SQL).format(lib=lib, d=d, s=sym)))
    fetch.close = db.close
    fetch.db = db
    return fetch


def wrds_fetcher():
    """One WRDS connection (credentials from the PostgreSQL client configuration) -> fetch(sym, day) returning the three DataFrames."""
    import wrds
    return make_fetch(wrds.Connection())


def trading_days(db, start, end):
    """Dates (YYYYMMDD) with both complete_nbbo and ctm tables, from table names only."""
    years = range(int(start[:4]), int(end[:4]) + 1)
    t = query(db, DAYS_SQL.format(schemas=",".join(f"'taqm_{y}'" for y in years)))
    by = t.assign(kind=t["table_name"].str.rsplit("_", n=1).str[0], d=t["table_name"].str[-8:]).groupby("d")["kind"].nunique()
    s, e = start.replace("-", ""), end.replace("-", "")
    return sorted(d for d, n in by.items() if n == 2 and s <= d <= e)


def read_manifest(path):
    if not os.path.exists(path):
        return {}
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {(r.sym, r.date): r.sha256 for r in m.itertuples()}               # latest line wins


def run(tasks, fetch_factory, base=BASE, n_conn=N_CONN, min_free=MIN_FREE, free_fn=None, deadline=None, log=print, suffix=""):
    """tasks: [(sym, YYYYMMDD)]. Workers take no new task after `deadline` (epoch seconds; not an error).
    Returns {'written', 'skipped', 'remaining', 'wall', 'fetch_secs'}; raises Stop on the first error, warning or stop rule."""
    free_fn = free_fn or (lambda p: shutil.disk_usage(p).free)
    roots = sorted({split_root(base, d, suffix) for _, d in tasks})
    done = {}
    for r in roots:
        os.makedirs(r, exist_ok=True)
        mp = os.path.join(r, "MANIFEST.tsv")
        if not os.path.exists(mp):
            with open(mp, "w") as f:
                f.write("\t".join(MAN_COLS) + "\n")
        done[r] = read_manifest(mp)
        npth = os.path.join(r, "null_counts.tsv")
        if not os.path.exists(npth):
            with open(npth, "w") as f:
                f.write("\t".join(NULL_COLS) + "\n")
    q = queue.Queue()
    for t in tasks:
        q.put(t)
    stop = threading.Event()
    lock = threading.Lock()
    summary = {"written": 0, "skipped": 0, "failed": 0, "kinds": set(), "errors": [], "fetch_secs": []}
    t_start = time.time()

    def fail(sym, day, e):
        kind = f"{type(e).__name__}: {re.sub(r'[0-9]+', '#', str(e))[:80]}"
        root = split_root(base, day, suffix)
        ep = os.path.join(root, "errors.tsv")
        with lock:
            new = not os.path.exists(ep)
            with open(ep, "a") as f:
                if new:
                    f.write("\t".join(ERR_COLS) + "\n")
                f.write("\t".join([sym, day, kind.replace("\t", " ").replace("\n", " "), str(e)[:300].replace("\t", " ").replace("\n", " "),   # R15: one line
                                   dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")]) + "\n")
            summary["failed"] += 1
            summary["kinds"].add(kind)
            if summary["failed"] > MAX_FAIL or len(summary["kinds"]) > MAX_KINDS:
                summary["errors"].append(f"failures {summary['failed']} (limit {MAX_FAIL}), error kinds {len(summary['kinds'])} "
                                         f"(limit {MAX_KINDS}): {sorted(summary['kinds'])}")
                stop.set()
        log(f"{sym} {day} FAILED (no retry, errors.tsv): {kind}")

    def one(fetch, sym, day):
        root = split_root(base, day, suffix)
        path = os.path.join(root, day[:4], f"{sym}_{day}.npz")
        if os.path.exists(path):
            if done[root].get((sym, day)) == sha256(path):
                with lock:
                    summary["skipped"] += 1
                return
            raise Stop(f"{sym} {day}: file exists without a matching MANIFEST hash (not overwritten)")
        if free_fn(base) < min_free:
            raise Stop(f"free disk space below {min_free / 1024**3:.0f} GiB")
        t0 = time.time()
        nb, pre, tr = fetch(sym, day)
        t1 = time.time()
        b, st = day_bars(nb, pre, tr)
        t2 = time.time()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        part = path + ".part"
        with open(part, "wb") as f:
            bars.save(f, b)
            f.flush()
            os.fsync(f.fileno())
        h = sha256(part)
        row = {"sym": sym, "date": day, "file": os.path.relpath(path, root), "rows": bars.NSEC, "bytes": os.path.getsize(part),
               "sha256": h, "fetch_sec": round(t1 - t0, 2), "build_sec": round(t2 - t1, 3), **st,
               "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        with lock:
            with open(os.path.join(root, "MANIFEST.tsv"), "a") as f:
                f.write("\t".join(str(row[c]) for c in MAN_COLS) + "\n")
                f.flush()
                os.fsync(f.fileno())
            nrow = {"sym": sym, "date": day, **st["null_counts"], "source": "download", "written_utc": row["written_utc"]}
            with open(os.path.join(root, "null_counts.tsv"), "a") as f:
                f.write("\t".join(str(nrow[c]) for c in NULL_COLS) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(part, path)
            summary["written"] += 1
            summary["fetch_secs"].append(row["fetch_sec"])
        log(f"{sym} {day} written: rows {bars.NSEC}, bytes {row['bytes']}, fetch {row['fetch_sec']}s, "
            f"quotes {st['n_quote_session']}, trades {st['n_trade_valid']}, excluded trades {st['n_trade_excluded']}, "
            f"null quote rows {st['n_quote_null']}, null trade rows {st['n_trade_null']}, last trade sec {st['last_trade_sec']}")

    def worker(i):
        fetch = None
        try:
            fetch = fetch_factory()
            while not stop.is_set() and not (deadline and time.time() >= deadline):
                try:
                    sym, day = q.get_nowait()
                except queue.Empty:
                    return
                try:
                    one(fetch, sym, day)
                except Stop:
                    raise
                except Exception as e:                                 # incl. warnings turned into errors
                    fail(sym, day, e)
        except BaseException as e:                                      # includes warnings turned into errors
            with lock:
                summary["errors"].append(f"worker {i}: {type(e).__name__}: {str(e)[:300]}")
            stop.set()
        finally:
            if fetch is not None and hasattr(fetch, "close"):
                try:
                    fetch.close()
                except Exception:
                    pass

    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # any warning -> exception -> stop
        th = [threading.Thread(target=worker, args=(i,)) for i in range(n_conn)]
        for x in th:
            x.start()
        for x in th:
            x.join()
    summary["remaining"] = q.qsize()
    summary["wall"] = time.time() - t_start
    summary["kinds"] = sorted(summary["kinds"])
    if summary["errors"]:
        raise Stop("; ".join(summary["errors"]) + f" | written {summary['written']}, skipped {summary['skipped']}, "
                   f"failed {summary['failed']}, not started {summary['remaining']} (no retry)")
    return summary


def phased(tasks, fetch_factory, log, base=BASE, calib=CALIB, phase_sec=PHASE_SEC, n_conn=N_CONN, **kw):
    """Calibration (1 connection, first `calib` tasks) -> n_conn connections until `phase_sec` after the start ->
    the rest with n_conn, or n_conn - 1 when the measured parallel effect is below MIN_SPEEDUP."""
    t0 = time.time()
    a = run(tasks[:calib], fetch_factory, base=base, n_conn=1, log=log, **kw)
    r1 = a["written"] / a["wall"] if a["written"] else None
    log(f"phase 1 (1 connection): written {a['written']}, skipped {a['skipped']}, wall {a['wall']:.1f}s, "
        f"rate {r1 and r1 * 3600:.0f}/h, mean fetch {np.mean(a['fetch_secs']) if a['fetch_secs'] else float('nan'):.2f}s")
    b = run(tasks, fetch_factory, base=base, n_conn=n_conn, deadline=t0 + phase_sec, log=log, **kw)
    r3 = b["written"] / b["wall"] if b["written"] else None
    speed = (r3 / r1) if (r1 and r3) else None
    n_final = n_conn if (speed is None or speed >= MIN_SPEEDUP) else n_conn - 1
    log(f"phase 2 ({n_conn} connections): written {b['written']}, skipped {b['skipped']}, wall {b['wall']:.1f}s, "
        f"rate {r3 and r3 * 3600:.0f}/h, mean fetch {np.mean(b['fetch_secs']) if b['fetch_secs'] else float('nan'):.2f}s, "
        f"parallel effect {speed and round(speed, 2)}x -> {n_final} connections for the rest")
    c = run(tasks, fetch_factory, base=base, n_conn=n_final, log=log, **kw)
    log(f"phase 3 ({n_final} connections): written {c['written']}, skipped {c['skipped']}, wall {c['wall']:.1f}s")
    return {"phase1": a, "phase2": b, "phase3": c, "speedup": speed, "n_final": n_final, "total_wall": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--syms", help="ETF mode: comma-separated symbols")
    ap.add_argument("--stocks", action="store_true", help="stock mode: universe_<Y>.tsv x calendar.csv days")
    ap.add_argument("--years", help="stock mode: comma-separated years (default: all years of the split)")
    ap.add_argument("--go", action="store_true", help="required; only after Kyle's permission for this split")
    a = ap.parse_args()
    if not a.go:
        sys.exit("refused: downloading needs --go, given only after Kyle's permission for this split")
    if a.stocks == bool(a.syms):
        sys.exit("refused: give exactly one of --syms (ETF mode) or --stocks")
    os.makedirs(LOG_DIR, exist_ok=True)
    tag = f"{a.split}{STOCKS if a.stocks else ''}"
    logf = open(os.path.join(LOG_DIR, f"download_{tag}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), "a")

    def log(msg):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    lo, hi = SPLITS[a.split]
    if a.stocks:
        years = [int(y) for y in a.years.split(",")] if a.years else list(range(int(lo[:4]), int(hi[:4]) + 1))
        first = wrds_fetcher()
        try:
            gaps = table_gaps(first.db, split_days(a.split, years))
            for d, miss in sorted(gaps.items()):
                log(f"no {'/'.join(miss)} table on {d}: not fetched (skipped_absent.tsv)")
            tasks, absent = stock_tasks(a.split, years, wrds_presence(first.db), gaps=gaps)
        except Stop as e:
            first.close()
            log(f"STOPPED: {e}")
            sys.exit(1)
        first.close()
        record_absent(BASE, absent, log=log)
        log(f"plan: split {a.split} stocks, years {years}, {len(tasks)} symbol-days present in the master "
            f"-> {os.path.join(BASE, 'bars_1s_' + tag)}")
        try:
            r = run(tasks, wrds_fetcher, n_conn=N_CONN, log=log, suffix=STOCKS)
            log(f"done: written {r['written']}, skipped {r['skipped']}, failed {r['failed']} {r['kinds']}, wall {r['wall']:.0f}s")
        except Stop as e:
            log(f"STOPPED: {e}")
            sys.exit(1)
        return
    syms = a.syms.split(",")
    if not all(SYM_RE.match(s) for s in syms):
        sys.exit("refused: symbols must match ^[A-Z]{1,6}$")
    first = wrds_fetcher()
    days = trading_days(first.db, f"{lo[:4]}-{lo[4:6]}-{lo[6:]}", f"{hi[:4]}-{hi[4:6]}-{hi[6:]}")
    first.close()
    tasks = [(s, d) for d in days for s in syms]
    log(f"plan: split {a.split} ({lo}..{hi}), {len(syms)} symbols x {len(days)} days = {len(tasks)} symbol-days "
        f"-> {os.path.join(BASE, 'bars_1s_' + a.split)}")
    try:
        r = run(tasks, wrds_fetcher, n_conn=N_CONN, log=log)
        log(f"done: written {r['written']}, skipped {r['skipped']}, failed {r['failed']} {r['kinds']}, wall {r['wall']:.0f}s")
    except Stop as e:
        log(f"STOPPED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
