# Frozen copy of download.py before the NULL handling change (2026-10-27, reference for the equivalence test).
#!/usr/bin/env python3
"""Research 12 data step 1: WRDS TAQ -> 1-second bars, one symbol-day per file. Run only with Kyle's permission per split.
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
from table names only (complete_nbbo_YYYYMMDD and ctm_YYYYMMDD both present). Any error or warning stops all workers
without retry; the run also stops when free disk space falls below 15 GiB.
Connections (Kyle 2026-10-27): 3. A calibration phase with 1 connection (first CALIB tasks) gives the single-connection
rate; the 3-connection rate over the rest of the first hour gives the parallel effect; below 1.5x the remaining tasks
use 2 connections. No price, size or return value is printed or logged: counts, seconds, bytes and hashes only."""
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

import ref_v1_bars as bars

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.environ.get("DATA_DIR", "data"), "research12_data")
SPLITS = {"dev": ("20190102", "20201231"), "holdout": ("20210104", "20260930"), "reserve": ("20261001", "20261023")}
LOG_DIR = os.path.join(HERE, "logs")
N_CONN = 3
CALIB = 15
PHASE_SEC = 3600
MIN_SPEEDUP = 1.5
MIN_FREE = 15 * 1024**3
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


class Stop(Exception):
    pass


def split_of(day):
    for name, (a, b) in SPLITS.items():
        if a <= day <= b:
            return name
    raise Stop(f"{day} is outside every split")


def split_root(base, day):
    return os.path.join(base, f"bars_1s_{split_of(day)}")


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


def day_bars(nbbo, pre, trades):
    """DataFrames from the three queries -> (bars, stats). No value is printed."""
    q = pd.concat([pre, nbbo], ignore_index=True)
    qt = to_ns(q["time_m"], q["time_m_nano"])
    tt = to_ns(trades["time_m"], trades["time_m_nano"])
    corr = trades["tr_corr"].fillna("").astype(str).str.zfill(2).to_numpy(object)
    sc = trades["tr_scond"].fillna("").astype(str).to_numpy(object)
    b, st = bars.build_bars(qt, q["best_bid"].to_numpy(float), q["best_ask"].to_numpy(float),
                            q["best_bidsizeshares"].to_numpy(np.int64), q["best_asksizeshares"].to_numpy(np.int64),
                            bars.quote_ok(q["qu_cond"].fillna("").to_numpy(object), q["qu_source"].fillna("").to_numpy(object),
                                          q["natbbo_ind"].fillna("").astype(str).to_numpy(object)),
                            tt, trades["price"].to_numpy(float), trades["size"].to_numpy(np.int64),
                            bars.trade_ok(corr, sc), bars.odd_lot(sc))
    st["excl_by_code"] = json.dumps(bars.trade_exclusions(tt, corr, sc), separators=(",", ":"))
    return b, st


def wrds_fetcher():
    """One WRDS connection (credentials from the PostgreSQL client configuration) -> fetch(sym, day) returning the three DataFrames."""
    import wrds
    db = wrds.Connection()

    def fetch(sym, day):
        lib, d = f"taqm_{day[:4]}", day
        return (db.raw_sql(NBBO_SQL.format(lib=lib, d=d, s=sym)), db.raw_sql(PRE_SQL.format(lib=lib, d=d, s=sym, n=PREOPEN_ROWS)),
                db.raw_sql(TRADE_SQL.format(lib=lib, d=d, s=sym)))
    fetch.close = db.close
    fetch.db = db
    return fetch


def trading_days(db, start, end):
    """Dates (YYYYMMDD) with both complete_nbbo and ctm tables, from table names only."""
    years = range(int(start[:4]), int(end[:4]) + 1)
    t = db.raw_sql(DAYS_SQL.format(schemas=",".join(f"'taqm_{y}'" for y in years)))
    by = t.assign(kind=t["table_name"].str.rsplit("_", n=1).str[0], d=t["table_name"].str[-8:]).groupby("d")["kind"].nunique()
    s, e = start.replace("-", ""), end.replace("-", "")
    return sorted(d for d, n in by.items() if n == 2 and s <= d <= e)


def read_manifest(path):
    if not os.path.exists(path):
        return {}
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return {(r.sym, r.date): r.sha256 for r in m.itertuples()}               # latest line wins


def run(tasks, fetch_factory, base=BASE, n_conn=N_CONN, min_free=MIN_FREE, free_fn=None, deadline=None, log=print):
    """tasks: [(sym, YYYYMMDD)]. Workers take no new task after `deadline` (epoch seconds; not an error).
    Returns {'written', 'skipped', 'remaining', 'wall', 'fetch_secs'}; raises Stop on the first error, warning or stop rule."""
    free_fn = free_fn or (lambda p: shutil.disk_usage(p).free)
    roots = sorted({split_root(base, d) for _, d in tasks})
    done = {}
    for r in roots:
        os.makedirs(r, exist_ok=True)
        mp = os.path.join(r, "MANIFEST.tsv")
        if not os.path.exists(mp):
            with open(mp, "w") as f:
                f.write("\t".join(MAN_COLS) + "\n")
        done[r] = read_manifest(mp)
    q = queue.Queue()
    for t in tasks:
        q.put(t)
    stop = threading.Event()
    lock = threading.Lock()
    summary = {"written": 0, "skipped": 0, "errors": [], "fetch_secs": []}
    t_start = time.time()

    def one(fetch, sym, day):
        root = split_root(base, day)
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
            os.replace(part, path)
            summary["written"] += 1
            summary["fetch_secs"].append(row["fetch_sec"])
        log(f"{sym} {day} written: rows {bars.NSEC}, bytes {row['bytes']}, fetch {row['fetch_sec']}s, "
            f"quotes {st['n_quote_session']}, trades {st['n_trade_valid']}, excluded trades {st['n_trade_excluded']}, "
            f"last trade sec {st['last_trade_sec']}")

    def worker(i):
        fetch = None
        try:
            fetch = fetch_factory()
            while not stop.is_set() and not (deadline and time.time() >= deadline):
                try:
                    sym, day = q.get_nowait()
                except queue.Empty:
                    return
                one(fetch, sym, day)
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
    if summary["errors"]:
        raise Stop("; ".join(summary["errors"]) + f" | written {summary['written']}, skipped {summary['skipped']}, "
                   f"not started {summary['remaining']} (no retry)")
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
    ap.add_argument("--syms", required=True)
    ap.add_argument("--go", action="store_true", help="required; only after Kyle's permission for this split")
    a = ap.parse_args()
    if not a.go:
        sys.exit("refused: downloading needs --go, given only after Kyle's permission for this split")
    syms = a.syms.split(",")
    if not all(SYM_RE.match(s) for s in syms):
        sys.exit("refused: symbols must match ^[A-Z]{1,6}$")
    os.makedirs(LOG_DIR, exist_ok=True)
    logf = open(os.path.join(LOG_DIR, f"download_{a.split}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), "a")

    def log(msg):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    lo, hi = SPLITS[a.split]
    first = wrds_fetcher()
    days = trading_days(first.db, f"{lo[:4]}-{lo[4:6]}-{lo[6:]}", f"{hi[:4]}-{hi[4:6]}-{hi[6:]}")
    first.close()
    tasks = [(s, d) for d in days for s in syms]
    log(f"plan: split {a.split} ({lo}..{hi}), {len(syms)} symbols x {len(days)} days = {len(tasks)} symbol-days "
        f"-> {os.path.join(BASE, 'bars_1s_' + a.split)}")
    try:
        r = phased(tasks, wrds_fetcher, log)
        log(f"done: total wall {r['total_wall']:.0f}s, parallel effect {r['speedup']}, final connections {r['n_final']}")
    except Stop as e:
        log(f"STOPPED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
