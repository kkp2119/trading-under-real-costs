#!/usr/bin/env python3
"""Research 15 step 2 (Kyle 2026-10-29): inputs X and labels y per decision, 2015-2020 only. Definitions: FEATURES15.md.

Decisions: at the end of bar u = 4, 9, ..., F - 5 (every 5 s from 09:30:05), F = close bar - 61 (15:59:00 end, or one
minute before an early close) is the flatten bar. A decision whose bar has no valid NBBO (bid > 0, ask > 0, ask >= bid,
bsz > 0, asz > 0) is no_quote: X is kept, every label is NaN.
X(u) uses bars <= u only; a window reaching before bar 0 gives NaN.
y(u, side): a 100-share limit order at the bar-u best quote (buy at bid, sell at ask), resting bars u+1 .. u+5 (research 12
B0 reset5), queue ahead = 4 x the displayed size in quarter shares (research 12 Q model, theta 1). The fill rule is
research12/fillmodel.py fill_buy / fill_sell written over arrays (numpy; numba is not installed); tests/test_xy15.py
checks it against research12 simulator.replay. Labels: fill (any share filled), frac (shares / 100), tf (first fill bar
- u), mk_h = dir x (mid(t_f + h) - price) in cents, val_h = fill x mk_h (0 without a fill), tk_h = taker round trip in
cents (buy: bid(u+h) - ask(u), sell: bid(u) - ask(u+h)); a bar after F is replaced by F and flagged (tr_*). No fees.
Storage: $DATA_DIR/research15_data/xy/<Y>/<SYM>_<YYYYMMDD>.npz (xy float32 n x len(COLUMNS), columns), xy/MANIFEST.tsv.
2015-2019: one decision in six (30 s), start offset numpy default_rng(15 + int(YYYYMMDD)).integers(0, 6); 2020: every
decision. Usage: python xy15.py --estimate SCRATCH_DIR | --go [--years 2015,2016] [--jobs 4] [--all-decisions --out DIR]."""
import argparse
import datetime as dt
import hashlib
import io
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loader15 as L  # noqa: E402

OUT = os.path.join(os.environ.get("DATA_DIR", "data"), "research15_data/xy")
TICK = 100                                     # 1e-4 dollars per cent
QTY, THETA4 = 100, 4                           # 100 shares; queue = 4 x displayed size in quarter shares (Q, theta 1)
FIRST, DELTA, LIFE, FLAT_GAP = 4, 5, 5, 61
HS = (5, 30, 60)
SAMPLE_YEARS, FULL_YEARS = range(2015, 2020), (2020,)
SAMPLE_EVERY = 6
MAX_BYTES = 15 * 10**9
FEATURES = ("spread_tk", "spread_bp", "bsz", "asz", "imb", "mid_usd",
            "dmid_1", "dmid_5", "dmid_30", "dmid_60", "dmid_300",
            "ofi_5", "ofi_30", "ofi_60", "ofi_300", "vol_5", "vol_30", "vol_60", "vol_300",
            "ntr_5", "ntr_30", "ntr_60", "ntr_300", "sd_30", "sd_300", "hl_60", "nupd_5", "nupd_60",
            "t_open", "t_close", "dmid_day")
SIDE_LABELS = ("fill", "frac", "tf") + tuple(f"{k}_{h}" for k in ("mk", "val", "tk") for h in HS)
LABELS = tuple(f"b_{x}" for x in SIDE_LABELS) + tuple(f"s_{x}" for x in SIDE_LABELS) + \
    tuple(f"tr_{k}_{h}" for k in ("tk", "mkb", "mks") for h in HS)
COLUMNS = ("u", "no_quote") + FEATURES + LABELS
MAN_COLS = ("sym", "date", "file", "rows", "decisions", "no_quote", "bytes", "sha256", "source_sha256", "close_bar",
            "offset", "vol", "vol_bid", "vol_ask", "vol_in", "vol_unc", "written_utc")


def flatten_bar(close_bar):
    return close_bar - FLAT_GAP


def decisions(close_bar):
    return np.arange(FIRST, flatten_bar(close_bar) - LIFE + 1, DELTA, dtype=np.int64)


def offset(day):
    return int(np.random.default_rng(15 + int(day)).integers(0, SAMPLE_EVERY))


def chosen(day, close_bar, full=False):
    """(decision bars stored, offset); offset -1 = every decision (2020, or full=True: step 3 validation rows of 2019)."""
    d = decisions(close_bar)
    if full or int(day[:4]) in FULL_YEARS:
        return d, -1
    k = offset(day)
    return d[k::SAMPLE_EVERY], k


def quote_ok(b):
    return (b["bid"] > 0) & (b["ask"] > 0) & (b["ask"] >= b["bid"]) & (b["bsz"] > 0) & (b["asz"] > 0)


def mid_ok(b):
    return (b["bid"] > 0) & (b["ask"] > 0) & (b["ask"] >= b["bid"])


def _wsum(x, us, w):
    """sum of x[u - w + 1 .. u] (float64), NaN where u - w + 1 < 0."""
    c = np.concatenate(([0], np.cumsum(x, dtype=np.int64)))
    lo = us + 1 - w
    out = (c[us + 1] - c[np.maximum(lo, 0)]).astype(np.float64)
    out[lo < 0] = np.nan
    return out


def features(b, us, close_bar):
    """float64 array len(us) x len(FEATURES); row i uses bars <= us[i] only."""
    bid, ask, bsz, asz = b["bid"], b["ask"], b["bsz"], b["asz"]
    qok, mok = quote_ok(b), mid_ok(b)
    mid2 = bid + ask
    X = np.full((len(us), len(FEATURES)), np.nan)
    col = {f: i for i, f in enumerate(FEATURES)}
    q = qok[us]
    sp = (ask[us] - bid[us]).astype(np.float64)
    X[q, col["spread_tk"]] = sp[q] / TICK
    X[q, col["spread_bp"]] = 2e4 * sp[q] / mid2[us][q]
    X[q, col["bsz"]] = bsz[us][q]
    X[q, col["asz"]] = asz[us][q]
    X[q, col["imb"]] = (bsz[us][q] - asz[us][q]) / (bsz[us][q] + asz[us][q])
    m = mok[us]
    X[m, col["mid_usd"]] = mid2[us][m] / 2e4
    for w in (1, 5, 30, 60, 300):
        p = us - w
        ok = (p >= 0) & m & mok[np.maximum(p, 0)]
        X[ok, col[f"dmid_{w}"]] = (mid2[us][ok] - mid2[p[ok]]) / (2 * TICK)
    for w in (5, 30, 60, 300):
        va, vb, vt = _wsum(b["vol_ask"], us, w), _wsum(b["vol_bid"], us, w), _wsum(b["vol"], us, w)
        with np.errstate(invalid="ignore", divide="ignore"):
            X[:, col[f"ofi_{w}"]] = np.where(vt > 0, (va - vb) / vt, np.nan)
        X[:, col[f"vol_{w}"]] = vt
        X[:, col[f"ntr_{w}"]] = _wsum(b["ntr"], us, w)
    for w in (5, 60):
        X[:, col[f"nupd_{w}"]] = _wsum(b["n_upd"], us, w)
    pair = np.zeros(len(bid), dtype=bool)                              # 1-second mid change v-1 -> v, both valid
    pair[1:] = mok[1:] & mok[:-1]
    dm = np.zeros(len(bid), dtype=np.int64)
    dm[1:] = np.where(pair[1:], mid2[1:] - mid2[:-1], 0)              # 2x units
    for w in (30, 300):
        n, s1, s2 = _wsum(pair, us, w), _wsum(dm, us, w), _wsum(dm * dm, us, w)
        n[us - w + 1 < 1] = np.nan                                     # the first pair ends at bar 1
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.where(n >= 2, (n * s2 - s1 * s1) / (n * n), np.nan)
        X[:, col[f"sd_{w}"]] = np.sqrt(np.maximum(var, 0)) / (2 * TICK)
    hi = np.where(mok, mid2, np.iinfo(np.int64).min)
    lo = np.where(mok, mid2, np.iinfo(np.int64).max)
    ok = us >= 59
    if ok.any():
        win_hi = np.lib.stride_tricks.sliding_window_view(hi, 60)[us[ok] - 59].max(axis=1)
        win_lo = np.lib.stride_tricks.sliding_window_view(lo, 60)[us[ok] - 59].min(axis=1)
        have = win_hi > np.iinfo(np.int64).min
        v = np.full(int(ok.sum()), np.nan)
        v[have] = (win_hi[have] - win_lo[have]) / (2 * TICK)
        X[ok, col["hl_60"]] = v
    X[:, col["t_open"]] = us + 1
    X[:, col["t_close"]] = close_bar - (us + 1)
    first = np.flatnonzero(mok)
    if len(first):
        f0 = first[0]
        ok = m & (us >= f0)
        X[ok, col["dmid_day"]] = (mid2[us][ok] - mid2[f0]) / (2 * TICK)
    return X


def fill_path(b, us, side):
    """Shares filled in bars u+1 .. u+LIFE (int64 array len(us) x LIFE) for the 100-share order of `side` (+1 buy at bid,
    -1 sell at ask) placed at the end of bar u: research12 fillmodel fill_buy / fill_sell with kind Q, theta4 = 4."""
    n = len(us)
    price = (b["bid"] if side > 0 else b["ask"])[us]
    a4 = THETA4 * (b["bsz"] if side > 0 else b["asz"])[us]
    rem = np.full(n, QTY, dtype=np.int64)
    path = np.zeros((n, LIFE), dtype=np.int64)
    for k in range(1, LIFE + 1):
        v = us + k
        if side > 0:
            amin, bmax, bmin = b["ask_min"][v], b["bid_max"][v], b["bid_min"][v]
            credit = np.maximum(b["vol_bid"][v] - b["vol_bid_odd"][v], 0)
            crossed = (amin > 0) & (amin <= price)
            q2 = ~crossed & (bmax > 0) & (bmax < price)
            q3 = ~crossed & ~q2 & (bmin == price) & (bmax == price)
        else:
            bmax, amin, amax = b["bid_max"][v], b["ask_min"][v], b["ask_max"][v]
            credit = np.maximum(b["vol_ask"][v] - b["vol_ask_odd"][v], 0)
            crossed = (bmax > 0) & (bmax >= price)
            q2 = ~crossed & (amin > price)
            q3 = ~crossed & ~q2 & (amin == price) & (amax == price)
        f = np.zeros(n, dtype=np.int64)
        f[crossed] = rem[crossed]
        f[q2] = np.minimum(rem, credit)[q2]
        a4 = np.where(q2, 0, a4)
        q3a = q3 & (a4 >= 0) & (4 * credit > a4)
        f[q3a] = np.minimum(rem, (4 * credit - a4) // 4)[q3a]
        q3b = q3 & (a4 >= 0) & ~q3a
        a4 = np.where(q3a, 0, np.where(q3b, a4 - 4 * credit, a4))
        f = np.where(rem > 0, f, 0)
        rem = rem - f
        path[:, k - 1] = f
    return path


def labels(b, us, close_bar):
    """float64 array len(us) x len(LABELS); rows of no_quote decisions are NaN."""
    F = flatten_bar(close_bar)
    Y = np.full((len(us), len(LABELS)), np.nan)
    col = {x: i for i, x in enumerate(LABELS)}
    q = quote_ok(b)[us]
    uq = us[q]
    mok = mid_ok(b)
    mid2 = b["bid"] + b["ask"]
    for h in HS:
        Y[q, col[f"tr_tk_{h}"]] = (uq + h > F).astype(np.float64)
    for side, p in ((1, "b"), (-1, "s")):
        price = (b["bid"] if side > 0 else b["ask"])[uq]
        path = fill_path(b, uq, side)
        qty = path.sum(axis=1)
        filled = qty > 0
        k1 = np.argmax(path > 0, axis=1) + 1                           # first fill bar - u (valid where filled)
        Y[q, col[f"{p}_fill"]] = filled
        Y[q, col[f"{p}_frac"]] = qty / QTY
        Y[q, col[f"{p}_tf"]] = np.where(filled, k1, np.nan)
        tf = uq + k1
        for h in HS:
            v = np.minimum(tf + h, F)
            mk = np.where(filled & mok[v], side * (mid2[v] - 2 * price) / (2 * TICK), np.nan)
            Y[q, col[f"{p}_mk_{h}"]] = mk
            Y[q, col[f"{p}_val_{h}"]] = np.where(filled, mk, 0.0)
            Y[q, col[f"tr_mk{p}_{h}"]] = np.where(filled, (tf + h > F).astype(np.float64), np.nan)
            w = np.minimum(uq + h, F)
            if side > 0:
                tk = np.where(mok[w], (b["bid"][w] - b["ask"][uq]) / TICK, np.nan)
            else:
                tk = np.where(mok[w], (b["bid"][uq] - b["ask"][w]) / TICK, np.nan)
            Y[q, col[f"{p}_tk_{h}"]] = tk
    return Y


def build(b, us, close_bar):
    """float32 matrix len(us) x len(COLUMNS)."""
    no_q = ~quote_ok(b)[us]
    M = np.column_stack([us.astype(np.float64), no_q.astype(np.float64), features(b, us, close_bar), labels(b, us, close_bar)])
    return M.astype(np.float32)


def write(path, M):
    """Atomic npz (xy, columns); returns (bytes, sha256)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, xy=M, columns=np.array(COLUMNS))
    data = buf.getvalue()
    part = path + ".part"
    with open(part, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)
    return len(data), hashlib.sha256(data).hexdigest()


def one(task):
    """(year, MANIFEST row, close bar, out dir) -> MANIFEST dict for the xy file."""
    year, row, close_bar, out, full = task
    b = L.load(year, row)
    us, k = chosen(row["date"], close_bar, full)
    M = build(b, us, close_bar)
    path = os.path.join(out, str(year), f"{row['sym']}_{row['date']}.npz")
    nbytes, h = write(path, M)
    c = slice(0, close_bar)
    vs = {x: int(b[x][c].sum()) for x in ("vol", "vol_bid", "vol_ask", "vol_in", "vol_unc")}
    return {"sym": row["sym"], "date": row["date"], "file": os.path.relpath(path, out), "rows": len(us),
            "decisions": len(decisions(close_bar)), "no_quote": int(M[:, 1].sum()), "bytes": nbytes, "sha256": h,
            "source_sha256": row["sha256"], "close_bar": close_bar, "offset": k, **vs,
            "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


def tasks(years, out, full=False):
    t = []
    for y in years:
        cb = L.close_bars(y)
        for (s, d), r in sorted(L.manifest(y).items(), key=lambda kv: (kv[0][1], kv[0][0])):
            t.append((y, r, cb[d], out, full))
    return t


def estimate(tmp):
    """One day each of 2015, 2019 (sampled) and 2020 (every decision) into the scratch folder `tmp` -> projected total
    bytes (nothing is written in OUT)."""
    per = {}
    for y in (2015, 2019, 2020):
        t = tasks([y], tmp)
        day = t[0][1]["date"]
        rows = [one(x) for x in t if x[1]["date"] == day]                    # (task tuples carry full=False)
        per[y] = (sum(r["bytes"] for r in rows) / len(rows), sum(r["rows"] for r in rows) / len(rows))
    total = 0
    for y in L.YEARS:
        n = len(L.manifest(y))
        bpd = per[2020][0] if y in FULL_YEARS else per[2015 if y < 2019 else 2019][0]
        total += n * bpd
        print(f"{y}: {n} symbol-days x {bpd / 1e3:.0f} KB -> {n * bpd / 1e9:.2f} GB")
    print(f"per symbol-day: " + ", ".join(f"{y} {b / 1e3:.0f} KB {r:.0f} rows" for y, (b, r) in per.items()))
    print(f"projected total {total / 1e9:.2f} GB (stop rule {MAX_BYTES / 1e9:.0f} GB); estimate files in {tmp}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", metavar="SCRATCH_DIR", help="size estimate from one day, files written to SCRATCH_DIR")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--years", default=",".join(str(y) for y in L.YEARS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--all-decisions", action="store_true", help="every decision (step 3: 2019 validation rows)")
    ap.add_argument("--out", default=OUT, help="output folder (step 3: $DATA_DIR/research15_data/xy_full)")
    a = ap.parse_args()
    if a.estimate:
        estimate(a.estimate)
        return
    if not a.go:
        sys.exit("refused: give --estimate or --go")
    years = [int(y) for y in a.years.split(",")]
    if any(y not in L.YEARS for y in years):
        sys.exit("refused: years must be 2015-2020")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    logf = open(os.path.join(HERE, "logs", f"xy15_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), "a")

    def log(m):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    out = os.path.expanduser(a.out)
    mp_path = os.path.join(out, "MANIFEST.tsv")
    os.makedirs(out, exist_ok=True)
    have = set()
    if os.path.exists(mp_path):
        m = pd.read_csv(mp_path, sep="\t", dtype=str)
        have = set(zip(m.sym, m.date))
    else:
        with open(mp_path, "w") as f:
            f.write("\t".join(MAN_COLS) + "\n")
    t = [x for x in tasks(years, out, a.all_decisions) if (x[1]["sym"], x[1]["date"]) not in have]
    for x in t:
        if os.path.exists(os.path.join(out, str(x[0]), f"{x[1]['sym']}_{x[1]['date']}.npz")):
            sys.exit(f"refused: {x[1]['sym']} {x[1]['date']} file exists without a MANIFEST line (not overwritten)")
    log(f"plan: years {years}, {len(t)} symbol-days (already in MANIFEST: {len(have)}), jobs {a.jobs}, "
        f"all decisions {a.all_decisions} -> {out}")
    t0, done, nbytes = time.time(), 0, 0
    with mp.Pool(a.jobs) as pool, open(mp_path, "a") as f:
        for r in pool.imap(one, t, chunksize=4):
            f.write("\t".join(str(r[c]) for c in MAN_COLS) + "\n")
            f.flush()
            done += 1
            nbytes += r["bytes"]
            if done % 500 == 0 or done == len(t):
                log(f"{done}/{len(t)} symbol-days, {nbytes / 1e9:.2f} GB, {time.time() - t0:.0f}s")
            if shutil.disk_usage(out).free < 15 * 1024**3:
                log("STOP: free disk space below 15 GiB")
                pool.terminate()
                sys.exit(1)
    log(f"done: {done} symbol-days, {nbytes / 1e9:.2f} GB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
