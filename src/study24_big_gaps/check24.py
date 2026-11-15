#!/usr/bin/env python3
"""Research 24 (study21_more_pairs/check21.py with the names of research 24; pairs are P01..P20 per year, families E2 and D2):
counts-only quality check of the 1-second bars of the pair legs.

No bar file is opened for its values (no price, size, spread or return is read): the files are hashed and compared with
the downloader's MANIFEST. Both parts are checked the same way; dev = 2015-01-02..2020-12-31 (SPEC24 1),
sealed = 2021-01-04..the last calendar day.
  expected   calendar24.csv days x universe24_<Y>.tsv legs
  written    MANIFEST lines whose file exists and whose sha256 matches; rows must be 23,400
  absent     skipped_absent.tsv (no single master symbol with a CUSIP of the leg / no table), error (errors.tsv),
             missing (none of these)
  coverage   from the MANIFEST: opening quote state present (first_state_sec == 0) and a trade in the last minute
             before the close; per leg and year: first and last day with a file
  pairs      a pair-day counts when both legs have a file; days with only one leg go to ONE_SIDED24.tsv
One part per run (step 2: the development part is checked when its collection is complete, the sealed part later).
Writes DATA_MANIFEST24.tsv (dev) or SEALED_MANIFEST24.tsv (sealed) (sym, date, file, rows, bytes, sha256),
ONE_SIDED24_<part>.tsv and results/check24_quality_<part>.json; refuses to overwrite any of them.
  python check24.py --part dev|sealed"""
import argparse
import hashlib
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.environ.get("DATA_DIR", "data"), "research24_data")
PARTS = {"dev": ("20150102", "20201231", "DATA_MANIFEST24.tsv"), "sealed": ("20210104", "20261231", "SEALED_MANIFEST24.tsv")}
NSEC = 23_400


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_tsv(root, name):
    p = os.path.join(root, name)
    return pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False) if os.path.exists(p) else pd.DataFrame(columns=["sym", "date"])


def main(only):
    outputs = [os.path.join(HERE, PARTS[only][2]), os.path.join(HERE, f"ONE_SIDED24_{only}.tsv"), os.path.join(HERE, "results", f"check24_quality_{only}.json")]
    if any(os.path.exists(o) for o in outputs):
        raise SystemExit(f"an output of part {only} exists (nothing is overwritten): {[os.path.basename(o) for o in outputs if os.path.exists(o)]}")
    cal = pd.read_csv(os.path.join(HERE, "calendar24.csv"), dtype=str)
    cal["d"] = cal.date.str.replace("-", "")
    close = {d: NSEC - ((16 - int(t[:2])) * 3600 - int(t[3:5]) * 60) for d, t in zip(cal.d, cal.close_time)}
    uni = {y: pd.read_csv(os.path.join(HERE, f"universe24_{y}.tsv"), sep="\t", dtype=str) for y in sorted(set(cal.d.str[:4]))}
    out, one_sided = {"calendar_days": int(len(cal)), "first_day": cal.d.min(), "last_day": cal.d.max(), "parts": {}}, []
    for part, (lo, hi, man_name) in ((only, PARTS[only]),):
        root = os.path.join(BASE, part, "bars_1s")
        days = [d for d in cal.d if lo <= d <= hi]
        exp = pd.DataFrame([(r.sym, d, r.pair, r.leg) for d in days for r in uni[d[:4]].itertuples()], columns=["sym", "date", "pair", "leg"])
        man = read_tsv(root, "MANIFEST.tsv").drop_duplicates(["sym", "date"], keep="last")
        man["exists"] = [os.path.exists(os.path.join(root, f)) for f in man.file]
        man["sha_ok"] = [e and sha(os.path.join(root, f)) == h for e, f, h in zip(man.exists, man.file, man.sha256)]
        absent, errors = read_tsv(root, "skipped_absent.tsv"), read_tsv(root, "errors.tsv")
        key = lambda df: set(zip(df.sym, df.date))                                          # noqa: E731
        w, a, e = key(man[man.sha_ok]), key(absent), key(errors)
        exp["status"] = ["written" if k in w else "absent" if k in a else "error" if k in e else "missing" for k in zip(exp.sym, exp.date)]
        exp["year"] = exp.date.str[:4]
        m = man[man.sha_ok].copy()
        m["close"] = m.date.map(close)
        m["open_state"] = m.first_state_sec.astype(int) == 0
        m["late_trade"] = m.last_trade_sec.astype(int) >= m.close - 60
        by_year, pairs, lost = {}, {}, []
        for y, g in exp.groupby("year"):
            mm = m[m.date.str[:4] == y]
            by_year[y] = {"expected": int(len(g)), **{s: int((g.status == s).sum()) for s in ("written", "absent", "error", "missing")},
                          "rows_not_23400": int((mm.rows.astype(int) != NSEC).sum()),
                          "no_opening_state": int((~mm.open_state).sum()), "no_trade_last_minute": int((~mm.late_trade).sum()),
                          "no_session_quote": int((mm.n_quote_session.astype(int) == 0).sum()), "no_valid_trade": int((mm.n_trade_valid.astype(int) == 0).sum()),
                          "absent_reasons": absent[absent.date.str[:4] == y].reason.value_counts().to_dict() if len(absent) else {}}
            for p, gp in g.groupby("pair"):
                t = gp.pivot(index="date", columns="leg", values="status")
                both = (t.A == "written") & (t.B == "written")
                one = (t.A == "written") ^ (t.B == "written")
                syms = dict(zip(gp.leg, gp.sym))
                pairs[f"{y} {p} {syms['A']}/{syms['B']}"] = {"days": int(len(t)), "both_legs": int(both.sum()), "one_leg_only": int(one.sum()),
                                                         "neither": int((~both & ~one).sum()),
                                                         "first_day_both": t.index[both].min() if both.any() else None,
                                                         "last_day_both": t.index[both].max() if both.any() else None}
                lost += [(d, p, f"{syms['A']}/{syms['B']}", "one_leg" if one[d] else "neither") for d in t.index[~both]]
                for d in t.index[one]:
                    have = "A" if t.A[d] == "written" else "B"
                    miss = "B" if have == "A" else "A"
                    one_sided.append({"part": part, "year": y, "pair": p, "date": d, "sym_present": syms[have], "sym_missing": syms[miss],
                                      "status_missing": t[miss][d]})
        out["parts"][part] = {"root": root.replace(os.path.expanduser("~"), "~"), "days": len(days), "manifest_lines": int(len(man)),
                              "files_missing": int((~man.exists).sum()), "sha_mismatch": int((man.exists & ~man.sha_ok).sum()),
                              "bytes": int(m.bytes.astype(int).sum()), "by_year": by_year, "pairs": pairs,
                              "total": {s: int((exp.status == s).sum()) for s in ("written", "absent", "error", "missing")},
                              "files_outside_the_expected_list": int(len(w - key(exp))),
                              "first_day_with_a_file": m.date.min() if len(m) else None, "last_day_with_a_file": m.date.max() if len(m) else None,
                              "pair_days_expected": int(sum(v["days"] for v in pairs.values())),
                              "pair_days_with_both_legs": int(sum(v["both_legs"] for v in pairs.values())),
                              "pair_days_not_trading_days": {"one_leg": sum(x[3] == "one_leg" for x in lost), "neither": sum(x[3] == "neither" for x in lost),
                                                             "list": [list(x) for x in sorted(lost)]}}
        m[["sym", "date", "file", "rows", "bytes", "sha256"]].sort_values(["date", "sym"]).to_csv(os.path.join(HERE, man_name), sep="\t", index=False)
    pd.DataFrame(one_sided, columns=["part", "year", "pair", "date", "sym_present", "sym_missing", "status_missing"]).to_csv(
        os.path.join(HERE, f"ONE_SIDED24_{only}.tsv"), sep="\t", index=False)
    out["one_sided_days"] = len(one_sided)
    with open(os.path.join(HERE, "results", f"check24_quality_{only}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({p: {k: v for k, v in x.items() if k not in ("by_year", "pairs")} for p, x in out["parts"].items()}, indent=1))
    for p, x in out["parts"].items():
        for k, v in x["pairs"].items():
            print(p, k, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    main(ap.parse_args().part)
