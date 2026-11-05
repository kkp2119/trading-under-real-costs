#!/usr/bin/env python3
"""Research 17 step 1 quality check of the sealed new-universe bars (SPEC17 1): counts only. No bar file is opened for its
values (no price, size, spread or return is read); the files are hashed and compared with the downloader's MANIFEST.
  expected   calendar17.csv days x universe17_<Y>.tsv symbols
  written    MANIFEST lines whose file exists and whose sha256 matches; rows must be 23,400
  absent     skipped_absent.tsv (not in the master / no table), errors (errors.tsv), missing (none of these)
  coverage   per symbol-day from the MANIFEST: opening quote state present (first_state_sec == 0) and a trade in the
             last minute before the close (last_trade_sec >= close bar - 60); per symbol: first and last date with a file
Writes research17/SEALED_MANIFEST.tsv (sym, date, file, rows, bytes, sha256) and results/check17_quality.json.
  python check17.py"""
import hashlib
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research17_data/sealed/bars_1s_new_stocks")
NSEC = 23_400


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_tsv(name):
    p = os.path.join(ROOT, name)
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False) if os.path.exists(p) else pd.DataFrame(columns=["sym", "date"])


def main():
    cal = pd.read_csv(os.path.join(HERE, "calendar17.csv"), dtype=str)
    cal["d"] = cal.date.str.replace("-", "")
    close = {d: NSEC - ((16 - int(t[:2])) * 3600 - int(t[3:5]) * 60) for d, t in zip(cal.d, cal.close_time)}
    exp = [(s, d) for d in cal.d for s in pd.read_csv(os.path.join(HERE, f"universe17_{d[:4]}.tsv"), sep="\t", dtype=str).sym]
    exp = pd.DataFrame(exp, columns=["sym", "date"])
    man = read_tsv("MANIFEST.tsv").drop_duplicates(["sym", "date"], keep="last")
    man["exists"] = [os.path.exists(os.path.join(ROOT, f)) for f in man.file]
    man["sha_ok"] = [e and sha(os.path.join(ROOT, f)) == h for e, f, h in zip(man.exists, man.file, man.sha256)]
    absent, errors = read_tsv("skipped_absent.tsv"), read_tsv("errors.tsv")
    key = lambda df: set(zip(df.sym, df.date))                                          # noqa: E731
    w, a, e = key(man[man.sha_ok]), key(absent), key(errors)
    exp["status"] = ["written" if k in w else "absent" if k in a else "error" if k in e else "missing" for k in zip(exp.sym, exp.date)]
    exp["year"] = exp.date.str[:4]
    m = man[man.sha_ok].copy()
    m["close"] = m.date.map(close)
    m["open_state"] = m.first_state_sec.astype(int) == 0
    m["late_trade"] = m.last_trade_sec.astype(int) >= m.close - 60
    by_year = {}
    for y, g in exp.groupby("year"):
        mm = m[m.date.str[:4] == y]
        by_year[y] = {"expected": int(len(g)), **{s: int((g.status == s).sum()) for s in ("written", "absent", "error", "missing")},
                      "rows_not_23400": int((mm.rows.astype(int) != NSEC).sum()),
                      "no_opening_state": int((~mm.open_state).sum()), "no_trade_last_minute": int((~mm.late_trade).sum()),
                      "absent_reasons": absent[absent.date.str[:4] == y].reason.value_counts().to_dict() if len(absent) else {}}
    per_sym = m.groupby("sym").date.agg(["min", "max", "count"]).reset_index().to_dict("records")
    out = {"root": ROOT.replace(os.path.expanduser("~"), "~"), "calendar_days": int(len(cal)), "first_day": cal.d.min(),
           "last_day": cal.d.max(), "manifest_lines": int(len(man)), "files_missing": int((~man.exists).sum()),
           "sha_mismatch": int((man.exists & ~man.sha_ok).sum()), "by_year": by_year, "per_symbol": per_sym,
           "total": {s: int((exp.status == s).sum()) for s in ("written", "absent", "error", "missing")}}
    man[man.sha_ok][["sym", "date", "file", "rows", "bytes", "sha256"]].sort_values(["date", "sym"]).to_csv(
        os.path.join(HERE, "SEALED_MANIFEST.tsv"), sep="\t", index=False)
    with open(os.path.join(HERE, "results", "check17_quality.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_symbol"}, indent=1))


if __name__ == "__main__":
    main()
