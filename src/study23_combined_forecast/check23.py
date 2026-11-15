#!/usr/bin/env python3
"""Research 23 quality check of the sealed new-universe bars (SPEC23 1; step-2 prompt 4): counts only. A copy of
study22_market_signal/check22.py with the names of research 23 (study18_forecast/check18.py plus the opening-state table of
study21_more_pairs/check21_open.py and a count of files outside the expected list). No bar file is opened for its values (no price, size, spread or return is
read); the files are hashed and compared with the downloader's MANIFEST.
  expected   calendar23.csv days x universe23_<Y>.tsv symbols
  written    MANIFEST lines whose file exists and whose sha256 matches; rows must be 23,400
  absent     skipped_absent.tsv (not in the master / no table), errors (errors.tsv), missing (none of these)
  coverage   per symbol-day from the MANIFEST: the first second with a quote state (0 = 09:30:00; how late it is), a trade
             in the last minute before the close, session quotes and valid trades present; per symbol: first and last date
  excluded   the rows of EXCLUDE23.tsv among the written files (they stay in the manifest; the sealed run leaves them out)
Writes SEALED_MANIFEST23.tsv (sym, date, file, rows, bytes, sha256) and results/check23_quality.json. Nothing is
overwritten.
  python check23.py"""
import hashlib
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research23_data/sealed/bars_1s_new_stocks")
NSEC = 23_400
BINS = [-2, -1, 0, 5, 60, 300, 1800, 23400]
LABELS = ["none_all_day", "0", "1-5s", "6-60s", "61-300s", "301-1800s", "over_1800s"]
OUTS = (os.path.join(HERE, "SEALED_MANIFEST23.tsv"), os.path.join(HERE, "results", "check23_quality.json"))


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
    if any(os.path.exists(p) for p in OUTS):
        raise SystemExit("an output exists: nothing is overwritten")
    cal = pd.read_csv(os.path.join(HERE, "calendar23.csv"), dtype=str)
    cal["d"] = cal.date.str.replace("-", "")
    close = {d: NSEC - ((16 - int(t[:2])) * 3600 - int(t[3:5]) * 60) for d, t in zip(cal.d, cal.close_time)}
    exp = [(s, d) for d in cal.d for s in pd.read_csv(os.path.join(HERE, f"universe23_{d[:4]}.tsv"), sep="\t", dtype=str).sym]
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
    first = m.first_state_sec.astype(int)
    m["bucket"] = pd.cut(first, bins=BINS, labels=LABELS).astype(str)
    m["late_trade"] = m.last_trade_sec.astype(int) >= m.close - 60
    on_disk = {os.path.join(y, f) for y in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, y)) for f in os.listdir(os.path.join(ROOT, y))}
    ex = pd.read_csv(os.path.join(HERE, "EXCLUDE23.tsv"), sep="\t", dtype=str, keep_default_na=False)
    by_year = {}
    for y, g in exp.groupby("year"):
        mm = m[m.date.str[:4] == y]
        by_year[y] = {"expected": int(len(g)), **{s: int((g.status == s).sum()) for s in ("written", "absent", "error", "missing")},
                      "rows_not_23400": int((mm.rows.astype(int) != NSEC).sum()),
                      "no_opening_state": int((mm.first_state_sec.astype(int) != 0).sum()), "no_trade_last_minute": int((~mm.late_trade).sum()),
                      "no_session_quote": int((mm.n_quote_valid_session.astype(int) == 0).sum()), "no_valid_trade": int((mm.n_trade_valid.astype(int) == 0).sum()),
                      "absent_reasons": absent[absent.date.str[:4] == y].reason.value_counts().to_dict() if len(absent) else {}}
    late = m[first != 0]
    per_sym = m.groupby("sym").date.agg(["min", "max", "count"]).reset_index().to_dict("records")
    out = {"root": ROOT.replace(os.path.expanduser("~"), "~"), "calendar_days": int(len(cal)), "first_day": cal.d.min(),
           "last_day": cal.d.max(), "manifest_lines": int(len(man)), "files_missing": int((~man.exists).sum()),
           "sha_mismatch": int((man.exists & ~man.sha_ok).sum()), "bytes": int(m.bytes.astype(int).sum()),
           "files_on_disk": len(on_disk), "files_outside_the_manifest": len(on_disk - set(man.file)),
           "written_outside_the_expected_list": int(len(w - set(zip(exp.sym, exp.date)))),
           "first_day_with_a_file": m.date.min(), "last_day_with_a_file": m.date.max(),
           "days_with_a_file": int(m.date.nunique()), "calendar_days_without_any_file": sorted(set(cal.d) - set(m.date)),
           "by_year": by_year,
           "total": {s: int((exp.status == s).sum()) for s in ("written", "absent", "error", "missing")},
           "opening_state": {"first_state_sec_not_0": int(len(late)), "max_first_state_sec": int(first.max()), "none_all_day": int((first < 0).sum()),
                             "late_sec_counts": {b: int((late.bucket == b).sum()) for b in LABELS if b != "0"},
                             "late_by_year_and_symbol": {y: g.sym.value_counts().sort_index().to_dict() for y, g in late.groupby(late.date.str[:4])}},
           "excluded_by_EXCLUDE23": {"rows": int(len(ex)), "among_written": int(len(key(ex) & w)),
                                     "written_minus_excluded": int(len(w - key(ex)))},
           "per_symbol": per_sym}
    man[man.sha_ok][["sym", "date", "file", "rows", "bytes", "sha256"]].sort_values(["date", "sym"]).to_csv(OUTS[0], sep="\t", index=False)
    with open(OUTS[1], "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_symbol"}, indent=1))


if __name__ == "__main__":
    main()
