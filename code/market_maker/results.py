"""Result files of SPEC 16 (atomic writes: .part then rename), large files outside the repository with their sha256
recorded in results/EXTERNAL.tsv (E1-67), the environment record (Kyle 2026-10-27: python, numba, numpy versions and
code hashes with every run) and the SPEC 10 side numbers that need day statistics (E1-69)."""
import datetime as dt
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import accounting
from config import HERE, QPATH_BARS

ENGINE = ("config.py", "loader.py", "session.py", "features.py", "fillmodel.py", "simulator.py", "accounting.py", "stats.py",
          "grid.py", "select_candidate.py", "judge.py", "placebo.py", "runner.py", "results.py", "run_dev.py", "run_holdout.py")
EXTERNAL_COLS = ["name", "path", "sha256", "bytes", "rows", "written_utc", "engine_sha256"]


def _clean(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "w") as f:
        json.dump(_clean(obj), f, indent=2, ensure_ascii=False)
    os.replace(path + ".part", path)
    return path


def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path + ".part", index=False)
    os.replace(path + ".part", path)
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def code_hashes():
    return {f: sha256(os.path.join(HERE, f)) for f in ENGINE if os.path.exists(os.path.join(HERE, f))}


def engine_sha256():
    """One hash over the engine files' hashes (in ENGINE order)."""
    return hashlib.sha256("".join(code_hashes().values()).encode()).hexdigest()


def env_versions():
    out = {"python": sys.version.split()[0], "executable": sys.executable}
    for m in ("numpy", "numba", "pandas", "scipy", "statsmodels"):
        try:
            out[m] = __import__(m).__version__
        except ImportError:
            out[m] = None
    out["code_sha256"] = code_hashes()
    return out


def run_id():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_external(write_fn, name, rows, ext_dir, split, rid, results_dir):
    """write_fn(path) writes the file; it goes to <ext_dir>/<split>/<rid>/<name> (never overwritten) and one line
    (name, path relative to ext_dir, sha256, bytes, rows, utc, engine hash) is appended to results/EXTERNAL.tsv."""
    rel = os.path.join(split, rid, name)
    path = os.path.join(ext_dir, rel)
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists (external files are never overwritten)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_fn(path + ".part")
    os.replace(path + ".part", path)
    rec = {"name": name, "path": rel, "sha256": sha256(path), "bytes": os.path.getsize(path), "rows": int(rows),
           "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "engine_sha256": engine_sha256()}
    reg = os.path.join(results_dir, "EXTERNAL.tsv")
    os.makedirs(results_dir, exist_ok=True)
    new = not os.path.exists(reg)
    with open(reg, "a") as f:
        if new:
            f.write("\t".join(EXTERNAL_COLS) + "\n")
        f.write("\t".join(str(rec[c]) for c in EXTERNAL_COLS) + "\n")
    return rec


def external_csv(df, name, ext_dir, split, rid, results_dir):
    return write_external(lambda p: df.to_csv(p, index=False), name, len(df), ext_dir, split, rid, results_dir)


def external_records(records, prefix, ext_dir, split, rid, results_dir):
    """E1-66 fill logs (one row per Q fill) and E1-68 inventory paths, one pair of files per recorded setting."""
    out = {}
    for name, rec in records.items():
        f = rec["fills"]
        out[f"{prefix}_fill_log_{name}"] = external_csv(f, f"{prefix}_fill_log_{name}.csv", ext_dir, split, rid, results_dir)
        idx, q = rec["qpath"]
        tab = pd.concat([idx, pd.DataFrame(q, columns=[f"q_{u}" for u in QPATH_BARS])], axis=1)
        out[f"{prefix}_qpath_{name}"] = external_csv(tab, f"{prefix}_qpath_{name}.csv", ext_dir, split, rid, results_dir)
    return out


def market_proxy(days, obs):
    """E1-41: per observation day, the equal-weight first-to-last valid mid return of the symbol-days in `days`."""
    d = days[(days.first_mid2 > 0) & (days.last_mid2 > 0)]
    r = (d.last_mid2 / d.first_mid2 - 1.0).groupby(d.date).mean()
    return np.array([float(r.get(x, np.nan)) for x in obs])


def yearly(x, obs):
    s = pd.Series(np.asarray(x, dtype=float), index=[d[:4] for d in obs])
    return {y: float(v) for y, v in s.groupby(level=0).sum().items()}


def per_symbol(runs, setting, model="Q", scenario="R"):
    r = runs[(runs.setting == setting) & (runs.model == model)]
    s = pd.Series(accounting.net_arrays(r, scenario), index=r.sym.values)          # integer 1e-4 $ summed first
    return {k: float(v) / 1e4 for k, v in s.groupby(level=0).sum().items()}
