"""Research 19 step 1: download FRED public CSVs and split each one at 2013-01-01 right after the download.

Source: https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series> only (no API key, no login, generic User-Agent).
Each full file goes to raw/ and is not read by any later code; build/ holds rows dated before 2013-01-01 and
sealed/ the rest. The split works on text lines, so build + sealed hold every raw row unchanged (each file repeats
the header; line ends are written as \n).
Nothing here prints a value: only row counts, dates, sizes and sha256.

  python fetch19.py                    # main series (SPEC19 1): DATA_MANIFEST19.tsv, SEALED_MANIFEST19.tsv
  python fetch19.py --candidates A B   # replacement candidates into candidates/ (listed in SPEC19_QUESTIONS.md only)
"""
import argparse
import datetime as dt
import hashlib
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.environ.get("DATA_DIR", "data"), "research19_data")
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
UA = "study19_carry-fetch/1.0 (python urllib)"
SPLIT = "2013-01-01"
FX = ["DEXUSAL", "DEXCAUS", "DEXSZUS", "DEXUSEU", "DEXGEUS", "DEXUSUK", "DEXJPUS", "DEXUSNZ", "DEXNOUS", "DEXSDUS"]
RATES = [f"IR3TIB01{c}M156N" for c in ("AU", "CA", "CH", "DE", "EZ", "GB", "JP", "NZ", "NO", "SE", "US")]
MACRO = ["VIXCLS", "BAA10Y"]
GROUP = {**{s: "fx" for s in FX}, **{s: "rate" for s in RATES}, **{s: "macro" for s in MACRO}}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MAN_COLS = ["series", "group", "url", "downloaded_utc", "http_status", "bytes", "rows", "header", "sha256_raw", "note"]
SEAL_COLS = ["series", "part", "path", "rows", "first_date", "last_date", "sha256"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def get(series):
    req = urllib.request.Request(URL.format(series), headers={"User-Agent": UA, "Accept": "text/csv"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def split(text):
    """(header, build lines, sealed lines); every data line must start with an ISO date."""
    lines = text.splitlines()
    head, rows = lines[0], [x for x in lines[1:] if x != ""]
    bad = [i for i, x in enumerate(rows) if not DATE.match(x.split(",", 1)[0])]
    if bad:
        raise ValueError(f"{len(bad)} rows without an ISO date")
    return head, [x for x in rows if x[:10] < SPLIT], [x for x in rows if x[:10] >= SPLIT]


def write_new(path, b):
    if os.path.exists(path):
        raise FileExistsError(path)                                       # nothing is overwritten or deleted
    with open(path, "wb") as f:
        f.write(b)


def fetch(series, root, group):
    for d in ("raw", "build", "sealed"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    when = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status, b = get(series)
    m = {"series": series, "group": group, "url": URL.format(series), "downloaded_utc": when, "http_status": status,
         "bytes": len(b), "rows": "", "header": "", "sha256_raw": "", "note": ""}
    text = b.decode("utf-8", "replace")
    if status != 200 or not text.split("\n", 1)[0].rstrip().endswith("," + series):
        m["note"] = "not a FRED CSV for this id (missing or renamed)"
        return m, []
    head, build, sealed = split(text)
    write_new(os.path.join(root, "raw", f"{series}.csv"), b)
    m.update(rows=len(build) + len(sealed), header=head, sha256_raw=sha(b))
    seal = []
    for part, rows in (("build", build), ("sealed", sealed)):
        out = ("\n".join([head] + rows) + "\n").encode("utf-8")
        rel = os.path.join(part, f"{series}.csv")
        write_new(os.path.join(root, rel), out)
        seal.append({"series": series, "part": part, "path": rel, "rows": len(rows),
                     "first_date": rows[0][:10] if rows else "", "last_date": rows[-1][:10] if rows else "", "sha256": sha(out)})
    return m, seal


def tsv(path, cols, rows):
    if os.path.exists(path):
        raise FileExistsError(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+")
    a = ap.parse_args()
    if a.candidates:
        root, todo, tag = os.path.join(DATA, "candidates"), [(s, "candidate") for s in a.candidates], "_candidates"
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag += "_" + stamp
    else:
        root, todo, tag = DATA, [(s, GROUP[s]) for s in FX + RATES + MACRO], ""
    man, seal = [], []
    for s, g in todo:
        m, sl = fetch(s, root, g)
        man.append(m)
        seal += sl
        ok = "ok" if m["sha256_raw"] else "MISSING"
        print(f"{s:18s} {g:9s} http {m['http_status']} rows {m['rows'] or 0:>6} {ok}", flush=True)
        time.sleep(1.0)                                                   # polite pacing for the public endpoint
    tsv(os.path.join(HERE, f"DATA_MANIFEST19{tag}.tsv"), MAN_COLS, man)
    tsv(os.path.join(HERE, f"SEALED_MANIFEST19{tag}.tsv"), SEAL_COLS, seal)
    print(f"written: DATA_MANIFEST19{tag}.tsv, SEALED_MANIFEST19{tag}.tsv; data under {root}")
    return 0 if all(m["sha256_raw"] for m in man) else 1


if __name__ == "__main__":
    sys.exit(main())
