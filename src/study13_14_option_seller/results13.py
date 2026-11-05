#!/usr/bin/env python3
"""Research 13 result files (memo 4b 7, 11). Aggregates go to research13/results (repository); contract-level files go
to $DATA_DIR/research13_data/results_external/<split>/<run id>/ and are registered with sha256 in results/EXTERNAL.tsv
(research12 results.write_external, sha256 dc022174...). Writes are atomic (.part then rename)."""
import datetime as dt
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("R13_RESULTS", os.path.join(HERE, "results"))
EXTERNAL = os.environ.get("R13_EXTERNAL", os.path.join(os.environ.get("DATA_DIR", "data"), "research13_data/results_external"))
ENGINE = ("config13.py", "market.py", "signals.py", "ladder.py", "costs.py", "accounting.py", "stats13.py", "select13.py",
          "results13.py", "run_dev.py", "run_holdout.py", "chain.py", "omdata.py")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_hashes():
    return {f: sha256(os.path.join(HERE, f)) for f in ENGINE if os.path.exists(os.path.join(HERE, f))}


def _clean(o):
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        return _clean(o.item())
    return o


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


def env_versions():
    out = {"python": sys.version.split()[0], "executable": sys.executable}
    for m in ("numpy", "pandas", "scipy"):
        try:
            out[m] = __import__(m).__version__
        except ImportError:
            out[m] = None
    out["code_sha256"] = code_hashes()
    return out


def run_id():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_external(df, name, split, rid, results_dir=None, ext_dir=None):
    results_dir, ext_dir = results_dir or RESULTS, ext_dir or EXTERNAL
    rel = os.path.join(split, rid, name)
    path = os.path.join(ext_dir, rel)
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists (external files are never overwritten)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path + ".part", index=False)
    os.replace(path + ".part", path)
    reg = os.path.join(results_dir, "EXTERNAL.tsv")
    new = not os.path.exists(reg)
    os.makedirs(results_dir, exist_ok=True)
    with open(reg, "a") as f:
        if new:
            f.write("name\tpath\tsha256\tbytes\trows\twritten_utc\n")
        f.write(f"{name}\t{rel}\t{sha256(path)}\t{os.path.getsize(path)}\t{len(df)}\t"
                f"{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
    return path
