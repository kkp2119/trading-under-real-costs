"""Research 19 step 2, ANSWERS A1: daily German mark rate from the Federal Reserve's H.10 historical files.

Source: https://www.federalreserve.gov/releases/h10/hist/dat96_ge.txt ("GERMANY -- SPOT EXCHANGE RATE, DM/US$", public
text file, no login, generic User-Agent). The Data Download Program itself lists no German series (DEVIATIONS19 1).
The raw file goes to candidates/raw/ and is not read again; the rows are rewritten as a FRED-style CSV
(observation_date,H10_DEM; ND -> empty) and split at 2013-01-01 like fetch19 (every row is before 2013, so sealed/ gets
the header only). Nothing here prints a value.

  python fetch19_h10.py     # DATA_MANIFEST19_h10.tsv, SEALED_MANIFEST19_h10.tsv
"""
import datetime as dt
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch19 import DATA, MAN_COLS, SEAL_COLS, SPLIT, UA, sha, tsv, write_new  # noqa: E402

URL = "https://www.federalreserve.gov/releases/h10/hist/dat96_ge.txt"
NAME = "H10_DEM"
TITLE = "GERMANY -- SPOT EXCHANGE RATE, DM/US$"
ROW = re.compile(r"^\s*(\d{1,2})-([A-Za-z]{3})-(\d{2})\s+(ND|\d+\.\d+)\s*$")
MON = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}


def parse(text):
    """[(ISO date, value text or '')]; two-digit years are 19xx (the file covers 1990-1998)."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if TITLE not in lines[0]:
        raise ValueError("not the H.10 Germany file")
    out = []
    for x in lines[1:]:
        m = ROW.match(x)
        if not m:
            if set(x) <= set("- "):
                continue                                                  # the rule under the title
            raise ValueError("unexpected line")
        d, mon, y, v = m.groups()
        out.append((dt.date(1900 + int(y), MON[mon.lower()], int(d)).isoformat(), "" if v == "ND" else v))
    if [d for d, _ in out] != sorted({d for d, _ in out}):
        raise ValueError("dates are not strictly increasing")
    return out


def main():
    root = os.path.join(DATA, "candidates")
    when = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    req = urllib.request.Request(URL, headers={"User-Agent": UA, "Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=120) as r:
        status, b = r.status, r.read()
    rows = parse(b.decode("utf-8", "replace"))
    write_new(os.path.join(root, "raw", "H10_dat96_ge.txt"), b)
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
    tsv(os.path.join(HERE, "DATA_MANIFEST19_h10.tsv"), MAN_COLS, man)
    tsv(os.path.join(HERE, "SEALED_MANIFEST19_h10.tsv"), SEAL_COLS, seal)
    for s in seal:
        print(s["part"], s["rows"], s["first_date"], s["last_date"])
    print("ND rows", sum(v == "" for _, v in rows))


if __name__ == "__main__":
    main()
