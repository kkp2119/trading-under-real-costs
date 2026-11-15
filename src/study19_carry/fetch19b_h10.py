"""Research 19B: daily German mark rate 1971-1989 from the Federal Reserve's H.10 historical files.

Source: https://www.federalreserve.gov/releases/h10/hist/dat89_ge.txt ("GERMANY -- SPOT EXCHANGE RATE, DM/US$", public
text file, no login, generic User-Agent): the older companion of the 1990-1998 file fetched by fetch19_h10.py, same
layout. The raw file goes to candidates/raw/ and is not read again; the rows are rewritten as a FRED-style CSV
(observation_date,H10_DEM_7189; ND -> empty) and split at 2013-01-01 like fetch19 (every row is before 2013).
Nothing here prints a value: rows, dates and sha256 only.

  python fetch19b_h10.py     # DATA_MANIFEST19B_h10.tsv, SEALED_MANIFEST19B_h10.tsv
"""
import datetime as dt
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch19 import DATA, MAN_COLS, SEAL_COLS, SPLIT, UA, sha, tsv, write_new  # noqa: E402
from fetch19_h10 import TITLE, parse  # noqa: E402

URL = "https://www.federalreserve.gov/releases/h10/hist/dat89_ge.txt"
NAME = "H10_DEM_7189"


def main():
    root = os.path.join(DATA, "candidates")
    when = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    req = urllib.request.Request(URL, headers={"User-Agent": UA, "Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=120) as r:
        status, b = r.status, r.read()
    rows = parse(b.decode("utf-8", "replace"))                            # two-digit years 71..89 are 19xx
    if not ("1971-01-01" <= rows[0][0] and rows[-1][0] <= "1989-12-31"):
        raise ValueError("the file is not the 1971-1989 file")
    write_new(os.path.join(root, "raw", "H10_dat89_ge.txt"), b)
    head = f"observation_date,{NAME}"
    seal = []
    for part, keep in (("build", [x for x in rows if x[0] < SPLIT]), ("sealed", [x for x in rows if x[0] >= SPLIT])):
        out = ("\n".join([head] + [f"{d},{v}" for d, v in keep]) + "\n").encode("utf-8")
        rel = os.path.join(part, f"{NAME}.csv")
        write_new(os.path.join(root, rel), out)
        seal.append({"series": NAME, "part": part, "path": rel, "rows": len(keep), "first_date": keep[0][0] if keep else "",
                     "last_date": keep[-1][0] if keep else "", "sha256": sha(out)})
    man = [{"series": NAME, "group": "fx_h10_hist", "url": URL, "downloaded_utc": when, "http_status": status, "bytes": len(b),
            "rows": len(rows), "header": TITLE, "sha256_raw": sha(b), "note": "DEM per USD; ND rows kept as empty"}]
    tsv(os.path.join(HERE, "DATA_MANIFEST19B_h10.tsv"), MAN_COLS, man)
    tsv(os.path.join(HERE, "SEALED_MANIFEST19B_h10.tsv"), SEAL_COLS, seal)
    for s in seal:
        print(s["part"], s["rows"], s["first_date"], s["last_date"])
    print("ND rows", sum(v == "" for _, v in rows))


if __name__ == "__main__":
    main()
