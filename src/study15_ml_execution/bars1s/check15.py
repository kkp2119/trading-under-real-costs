#!/usr/bin/env python3
"""Research 15 step 1 check of $DATA_DIR/research15_data/bars_1s_train_stocks (counts only; no price, size or return
value is printed or written). Args: years (default 2015..2018).
  1 per file: the 21 bars.COLS keys, int32, shape (23,400,); sha256 equals the latest MANIFEST line
  2 per date: files written vs universe symbols, absent (skipped_absent.tsv), errors (errors.tsv rows without a file; rows
    whose symbol-day was written by a later run are counted apart); calendar days with no file
  3 quote validity per year: seconds with a valid NBBO state (bid > 0 and ask >= bid), seconds before the first state,
    locked/crossed rows (n_xl), MANIFEST quote counts (session, valid, code-excluded, crossed), init_state
  4 format equal to the 2019 bars of research 12 (bars_1s_dev_stocks/2019, first MANIFEST file): keys, dtype, shape
  5 early closes: calendar15 close_time vs the MANIFEST last trade second (13:00 = second 12,600)
Writes logs/check15_<years>.txt and logs/check15_dates_<years>.tsv."""
import hashlib
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bars  # noqa: E402

ROOT = os.path.join(os.environ.get("DATA_DIR", "data"), "research15_data/bars_1s_train_stocks")
REF = os.path.join(os.environ.get("DATA_DIR", "data"), "research12_data/bars_1s_dev_stocks")
R15 = os.path.dirname(HERE)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt(path):
    with np.load(path) as z:
        return tuple((k, str(z[k].dtype), z[k].shape) for k in z.files)


def read_errors(path):
    """errors.tsv rows; a row whose kind held a newline (research-12 fail() before the R15 fix) spans several lines,
    so a line that does not start with SYM<tab>YYYYMMDD<tab> continues the previous row."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["sym", "date", "kind"])
    lines = open(path).read().splitlines()
    rows = []
    for ln in lines[1:]:
        if re.match(r"^[A-Z]{1,6}\t\d{8}\t", ln) or not rows:
            rows.append(ln)
        else:
            rows[-1] += " " + ln
    return pd.DataFrame([r.split("\t")[:5] for r in rows], columns=lines[0].split("\t"))


def main():
    years = [int(a) for a in sys.argv[1:]] or list(range(2015, 2019))
    tag = "_".join(map(str, years))
    out = []
    man = pd.read_csv(os.path.join(ROOT, "MANIFEST.tsv"), sep="\t", dtype=str, keep_default_na=False)
    man = man[man.date.str[:4].astype(int).isin(years)].drop_duplicates(["sym", "date"], keep="last")
    absf = os.path.join(ROOT, "skipped_absent.tsv")
    ab = pd.read_csv(absf, sep="\t", dtype=str) if os.path.exists(absf) else pd.DataFrame(columns=["sym", "date", "reason"])
    ab = ab[ab.date.str[:4].astype(int).isin(years)]
    er = read_errors(os.path.join(ROOT, "errors.tsv"))
    er = er[er.date.str[:4].astype(int).isin(years)]
    written = set(zip(man.sym, man.date))
    done = np.array([(a, b) in written for a, b in zip(er.sym, er.date)], dtype=bool)
    er_res, er = er.loc[done], er.loc[~done]                                   # er_res: failed once, written by a later run
    cal = pd.read_csv(os.path.join(R15, "calendar15.csv"), dtype=str)
    cal["d"] = cal.date.str.replace("-", "")
    cal = cal[cal.d.str[:4].astype(int).isin(years)]
    uni = {y: pd.read_csv(os.path.join(R15, f"universe15_{y}.tsv"), sep="\t", dtype=str).sym.tolist() for y in years}

    ref_row = pd.read_csv(os.path.join(REF, "MANIFEST.tsv"), sep="\t", dtype=str, nrows=1).iloc[0]
    ref_fmt = fmt(os.path.join(REF, ref_row.file))
    want = tuple((c, "int32", (bars.NSEC,)) for c in bars.COLS)
    out.append(f"4 format: 2019 reference {ref_row.file} keys/dtype/shape equal to bars.COLS int32 (23400,): {ref_fmt == want}")

    bad_fmt, bad_hash, stats = [], [], []
    for r in man.itertuples():
        p = os.path.join(ROOT, r.file)
        if not os.path.exists(p):
            bad_hash.append((r.sym, r.date, "missing"))
            continue
        if fmt(p) != ref_fmt:
            bad_fmt.append((r.sym, r.date))
        if sha256(p) != r.sha256:
            bad_hash.append((r.sym, r.date, "sha256"))
        z = bars.load(p)
        st = (z["bid"] > 0) & (z["ask"] >= z["bid"])
        stats.append((r.date, r.sym, int(st.sum()), int(np.argmax(st)) if st.any() else -1, int(z["n_xl"].sum()),
                      int(z["n_upd"].sum())))
    out.append(f"1 files: {len(man)} MANIFEST rows; format differs from 2019 in {len(bad_fmt)} {bad_fmt[:5]}; "
               f"missing or hash mismatch {len(bad_hash)} {bad_hash[:5]}")
    s = pd.DataFrame(stats, columns=["date", "sym", "valid_sec", "first_valid_sec", "n_xl", "n_upd"])

    rows = []
    for d in cal.d:
        y = int(d[:4])
        w = set(man.sym[man.date == d])
        a = set(ab.sym[ab.date == d])
        e = set(er.sym[er.date == d])
        rows.append((d, len(uni[y]), len(w), len(a), len(e), len(set(uni[y]) - w - a - e),
                     "|".join(sorted(set(ab.reason[ab.date == d])))))
    dd = pd.DataFrame(rows, columns=["date", "universe", "written", "absent", "errors", "unaccounted", "absent_reasons"])
    dd.to_csv(os.path.join(HERE, "logs", f"check15_dates_{tag}.tsv"), sep="\t", index=False)
    out.append(f"2 dates: {len(dd)} calendar days; days with no file at all {int((dd.written == 0).sum())} "
               f"{dd.date[dd.written == 0].tolist()[:10]}; symbol-days written {int(dd.written.sum())}, absent {int(dd.absent.sum())}, "
               f"errors {int(dd.errors.sum())}, unaccounted {int(dd.unaccounted.sum())}; written per day min {dd.written.min()} max {dd.written.max()}")
    out.append(f"   absent reasons: {ab.reason.value_counts().to_dict()}; absent by symbol: {ab.groupby(ab.date.str[:4]).sym.value_counts().to_dict()}")
    out.append(f"   errors still without a file by kind: {er.kind.value_counts().to_dict() if len(er) else {}}; "
               f"errors.tsv rows later written by a rerun: {len(er_res)} {er_res.kind.str[:60].value_counts().to_dict() if len(er_res) else {}}")
    m = man.assign(y=man.date.str[:4])
    for c in ("n_quote_session", "n_quote_valid_session", "n_quote_code_excluded", "n_quote_crossed", "n_trade_valid", "n_trade_excluded",
              "n_subtick_quote", "n_subtick_trade"):
        m[c] = m[c].astype(int)
    s["y"] = s.date.str[:4]
    for y, g in m.groupby("y"):
        sg = s[s.y == y]
        out.append(f"3 {y}: files {len(g)}; valid-state seconds {sg.valid_sec.sum() / (len(sg) * bars.NSEC):.4%}; "
                   f"files with no valid state {int((sg.valid_sec == 0).sum())}; first valid second median {int(sg.first_valid_sec.median())} "
                   f"max {int(sg.first_valid_sec.max())}; locked/crossed rows {int(sg.n_xl.sum())} of {int(sg.n_upd.sum())} session rows "
                   f"({sg.n_xl.sum() / max(sg.n_upd.sum(), 1):.4%}); MANIFEST quotes valid/session {g.n_quote_valid_session.sum() / g.n_quote_session.sum():.4%}, "
                   f"code-excluded {g.n_quote_code_excluded.sum()}, crossed {g.n_quote_crossed.sum()}, init_state False {int((g.init_state == 'False').sum())}; "
                   f"trades excluded/valid {g.n_trade_excluded.sum()}/{g.n_trade_valid.sum()}; subtick quote rows {g.n_subtick_quote.sum()}, "
                   f"subtick trades {g.n_subtick_trade.sum()}")
    lt = m.groupby("date").last_trade_sec.apply(lambda x: x.astype(int).max())
    early = set(cal.d[cal.close_time == "13:00"])
    late_early = sorted(d for d in early if d in lt.index and lt[d] >= 12_600 + 60)
    short = sorted(d for d in lt.index if d not in early and lt[d] < 12_600 + 60)
    out.append(f"5 early closes: calendar {sorted(early)}; early-close days with trades after 13:01 {late_early}; "
               f"other days whose last trade is before 13:01 {short}")
    txt = "\n".join(out)
    open(os.path.join(HERE, "logs", f"check15_{tag}.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
