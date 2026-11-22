#!/usr/bin/env python3
"""Research 31 validation and sealed runner (SPEC31 4-5; DEVIATIONS31 3). Written after the exploration (explore-run-31),
before any val bar is read. map31, run31, loader31 and research 30 / 30B code are imported unchanged.
  events  the base setting only (s15, k3, the three families together): map31.pairday's rows for that map, line for line,
          plus the market exit at delays 0 and 5 (leadlag30.events' d0_* and d5_* columns); the direction check as in map31
  g-hat   the frozen results/model31.json (s15_k3; sha256 checked before anything else); top = g-hat >= its 2015-2016 90th
          percentile threshold_bp from the same file; nothing is fitted again
  H1-31   mean before-cost value per event > 0;  H2-31  mean of the top events - mean of all events > 0
  test    pair-day cluster bootstrap (10,000, seed 33): per resample the pair-days are drawn with replacement and the
          per-event means are taken over the drawn pair-days' events; one-sided p = share of resampled statistics <= 0,
          alpha 0.05 (research 30G's boot)
  gate    validation: the sealed run is opened only when the H1-31 point estimate is > 0, otherwise the sentence
          "따라오는 현상은 검증 구간에서 이미 보이지 않았다. 봉인 데이터는 열지 않았다"; sealed: H1-31 and H2-31 at alpha 0.05
  H3      no candidate came out of the exploration: not judged; rule R of the 4 exploration picks is reported as an
          auxiliary (map31.run_rule with the frozen models)
  aux     after costs (L3D1, L0D0), family, year, 09:35-10:00, delays 0 and 5, g-hat deciles with the spread, J15, top
          events, direction check; every before-cost value is shown with the same events' after-cost value
Modes (each once; outputs never overwritten):
  ../research15/py15.sh val31.py --rehearsal   2015-2018 explore bars (seen data): the whole path; rehearse31_val.py
                                               compares it with the exploration
  ../research15/py15.sh val31.py --val         2019-2020: the tag pre-val-31 and the val collection complete
  ../research15/py15.sh val31.py --sealed      2021-01-04..2026-11-16: the tag code-freeze-31, Kyle's phrase in
                                               UNLOCK31.txt, the validation sent the study on, the sealed collection complete"""
import argparse
import datetime as dt
import functools
import glob
import hashlib
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import loader31 as LD  # noqa: E402
import map31 as M  # noqa: E402

LD.tag_ok = functools.lru_cache(maxsize=None)(LD.tag_ok)     # one git call per tag and process (DEVIATIONS31 3.9)
R31, LL, MD, C24 = M.R31, M.LL, M.MD, M.C24
RES = os.path.join(HERE, "results")
LOGS = os.path.join(HERE, "bars1s", "logs")
LOCAL = os.path.join(LD.ROOT, "runs31")
MODEL = os.path.join(RES, "model31.json")
MODEL_SHA = "14ef4eca178f4c880068f2bbe92f325dd33203fcb163a882608047a065be473c"
EXPLORE_SUMMARY = os.path.join(RES, "explore31_summary.json")
BASE_NAME = "s15_k3"
PICKS = ["s5_k3", "s15_k2", "s30_k3", "s5_k2"]                 # explore31_summary.json "picks" (2015-2016 rule R)
B, CHUNK, SEED, ALPHA = 10_000, 500, 33, 0.05
WINDOWS = {"rehearsal": LD.WINDOWS["explore"], "val": LD.WINDOWS["val"], "sealed": LD.WINDOWS["sealed"]}
SPLIT = {"rehearsal": "explore", "val": "val", "sealed": "sealed"}
YEARS = {"explore": list(range(2015, 2019)), "val": [2019, 2020], "sealed": list(range(2021, 2027))}
EXTRA = ("d0_mid", "d0_pnl", "d0_fee", "d5_mid", "d5_pnl", "d5_fee")
ROWV = M.ROW + EXTRA
VI = {c: i for i, c in enumerate(ROWV)}
TRIALS_BEFORE = 909
SENT = {"h1": "업종 ETF와 SPY가 먼저 움직이면, 대형주와 업종 ETF는 처음 보는 기간에도 따라왔다(비용 전)",
        "h2": "그리고 예측 모델이 고른 사건은 더 많이 따라왔다",
        "val_stop": "따라오는 현상은 검증 구간에서 이미 보이지 않았다. 봉인 데이터는 열지 않았다"}


class Refused(Exception):
    pass


def outputs(mode):
    return [os.path.join(RES, f"{mode}31_{x}") for x in ("summary.json", "report.md")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- the collection is complete
PLAN_RE = re.compile(r"plan: split (\w+) stocks, years \[([0-9, ]+)\], (\d+) symbol-days present in the master")
DONE_RE = re.compile(r"done: written (\d+), skipped (\d+), failed (\d+)")


def collection_log(split, logs=LOGS):
    """{years: (planned, written, skipped, failed, log file)} from the downloader logs' own lines: each done line belongs to
    the plan line before it in the same log; the latest (log name, then line) wins for a years group."""
    out = {}
    for p in sorted(glob.glob(os.path.join(logs, f"download_{split}_stocks_*.log"))):
        plan = None
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = PLAN_RE.search(line)
                if m:
                    plan = (tuple(int(y) for y in m.group(2).split(",")), int(m.group(3))) if m.group(1) == split else None
                d = DONE_RE.search(line)
                if d and plan:
                    out[plan[0]] = (plan[1], *(int(x) for x in d.groups()), os.path.basename(p))
                    plan = None
    return out


def downloader_running(split):
    r = subprocess.run(["pgrep", "-f", f"download.py --split {split}"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def plan_groups(split, logs=LOGS):
    """{years: (planned, failed of its done line or None, log file, order)} from each years group's NEWEST plan line in the
    downloader logs (log name, then line): a plan without a later done line (a stopped run) gives None
    (DEVIATIONS31 4.10)."""
    out, k = {}, 0
    for p in sorted(glob.glob(os.path.join(logs, f"download_{split}_stocks_*.log"))):
        cur = None
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = PLAN_RE.search(line)
                if m:
                    cur = tuple(int(y) for y in m.group(2).split(",")) if m.group(1) == split else None
                    if cur is not None:
                        k += 1
                        out[cur] = (int(m.group(3)), None, os.path.basename(p), k)
                d = DONE_RE.search(line)
                if d and cur is not None:
                    n, _, name, order = out[cur]
                    out[cur] = (n, int(d.group(3)), name, order)
                    cur = None
    return out


def plan_cover(split, groups):
    """The groups whose plans make up the split: the all-years group when it is newer than every part (the one-downloader
    fallback), otherwise the parts, which must cover the split's years exactly once. Returns (cover, problems)."""
    allys = tuple(YEARS[split])
    parts = {g: v for g, v in groups.items() if g != allys}
    if allys in groups and all(groups[allys][3] > v[3] for v in parts.values()):
        return {allys: groups[allys]}, []
    ys = sorted(y for g in parts for y in g)
    if ys != list(allys):
        return parts, [f"the plan groups' years {ys} do not cover {list(allys)} exactly once"]
    return parts, []


def collection_check(split, logs=LOGS, root=None, running=downloader_running):
    """Problems that keep a split from being read (empty = complete) (DEVIATIONS31 4.10: judged by the MANIFEST, not by
    adding up the logs' written + skipped, which counted a plan twice after a one-downloader fallback):
      no downloader still running for the split;
      planned = the plans of the groups that make up the split (plan_cover), each group's newest run ending with a done
        line with failed 0 (the logs are read first; the MANIFEST and the folder, behind loader31's lock, only after);
      the MANIFEST's distinct (symbol, day) = planned; every symbol-day in errors.tsv is in the MANIFEST (failed 0 in
        substance); no .part file in the folder; the MANIFEST sha256 recorded."""
    why = []
    if running(split):
        why.append(f"a downloader for {split} is still running")
    groups = plan_groups(split, logs)
    cover, problems = plan_cover(split, groups)
    why += problems
    for g, (n, f, name, _) in sorted(cover.items()):
        if f is None:
            why.append(f"{name}: the newest run for years {list(g)} has no done line")
        elif f != 0:
            why.append(f"{name}: failed {f}")
    planned = sum(v[0] for v in cover.values())
    info = {"planned": planned, "groups": {",".join(map(str, g)): list(v[:3]) for g, v in sorted(cover.items())}}
    if why:
        return why, info
    man = LD.manifest(split, root)
    if len(man) != planned:
        why.append(f"MANIFEST has {len(man)} distinct symbol-days, the plans count {planned}")
    base = LD.folder(split, root)
    ep = os.path.join(base, "errors.tsv")
    unresolved = 0
    if os.path.exists(ep):
        e = pd.read_csv(ep, sep="\t", dtype=str, keep_default_na=False, usecols=["sym", "date"])
        unresolved = len({(s, d) for s, d in zip(e["sym"], e["date"])} - set(man))
        if unresolved:
            why.append(f"{unresolved} symbol-days in errors.tsv are not in the MANIFEST")
    parts = [f for _, _, fs in os.walk(base) for f in fs if f.endswith(".part")]
    if parts:
        why.append(f"{len(parts)} .part files in the folder")
    return why, dict(info, manifest_rows=len(man), errors_unresolved=unresolved,
                     manifest_sha256=sha256(os.path.join(base, "MANIFEST.tsv")))


# ---------------------------------------------------------------- guard
def guard(mode, root=None):
    """Refuse a run that is not allowed; no val or sealed path is named before the checks in front of it pass."""
    if mode not in WINDOWS:
        raise Refused(f"unknown mode {mode}")
    if any(os.path.exists(p) for p in outputs(mode)):
        raise Refused(f"{mode}: the outputs exist, the run has been made")
    if sha256(MODEL) != MODEL_SHA:
        raise Refused("model31.json differs from its frozen sha256")
    if json.load(open(EXPLORE_SUMMARY, encoding="utf-8"))["picks"] != PICKS:
        raise Refused("the picks differ from explore31_summary.json")
    if mode == "val" and not LD.tag_ok(LD.VAL_TAG):
        raise Refused(f"validation needs the tag {LD.VAL_TAG}")
    if mode == "sealed":
        why = []
        if not LD.tag_ok(LD.FREEZE_TAG):
            why.append(f"no tag {LD.FREEZE_TAG}")
        if not LD.unlock_ok():
            why.append("no unlock phrase in UNLOCK31.txt")
        v = os.path.join(RES, "val31_summary.json")
        if not os.path.exists(v) or not json.load(open(v, encoding="utf-8"))["decision"]["go_sealed"]:
            why.append("the validation did not send the study to the sealed run")
        if why:
            raise Refused("; ".join(why))
    try:
        why, info = collection_check(SPLIT[mode], root=root)
    except LD.Locked as e:
        raise Refused(str(e))
    if why:
        raise Refused(f"{SPLIT[mode]} collection not complete: " + "; ".join(why))
    return info


def model():
    """{map name: frozen model} of model31.json (beta, bounds, threshold_bp, lambda)."""
    return json.load(open(MODEL, encoding="utf-8"))


def key(name):
    s, k = name[1:].split("_k")
    return int(s), float(k)


# ---------------------------------------------------------------- one pair-day
def pairday(i, t, a, b, spy, pa, pb):
    """Rows (ROWV) of the base map and the direction-check sums of one pair-day: map31.pairday for (s15, k3) line for line,
    with the delay 0 and 5 market-exit columns of the same events appended."""
    cb = t["close_bar"]
    beta = R31.beta_prev(pa, pb, t["prev_close_bar"])
    Q = LL.prepare(a, b, cb)
    fam = M.FAM[t["family"]]
    s, k = M.BASE
    rows = []
    if Q is not None and np.isfinite(beta):
        d = R31.d_pair(a, b, s, beta)
        sd, _ = LL.sigma(d)
        ev = R31.events(Q, d, k)
        BI = MD.bar_inputs(a, b, spy, s, d, sd, False)
        for r in ev:
            u, dirn, ad, q = int(r[LL.EVI["u"]]), int(r[LL.EVI["dir"]]), float(r[LL.EVI["ad"]]), int(r[LL.EVI["qty"]])
            o = M.EN.trade_ji(Q, u, dirn, ad, q, M.E_J, False)
            jp, jm, _, _, _ = M.EN.money_ji(o, dirn, 1)
            x, y = MD.event_inputs(BI, u, dirn, cb)
            craw, _ = R31.c_raw(Q, u, dirn, q)
            bid, ask = Q["X"][0, u], Q["X"][1, u]
            rows.append([i, u, dirn, fam, q, Q["m2"][u] / 2e4, r[LL.EVI["spread_ticks"]], 2e4 * (ask - bid) / (ask + bid), r[LL.EVI["size"]],
                         r[LL.EVI["vol1m_bp"]], r[LL.EVI["bucket"]], r[LL.EVI["ratio"]], craw, r[LL.EVI["mkt_mid"]], r[LL.EVI["mkt_pnl"]],
                         r[LL.EVI["mkt_pnl_L0D0"]], r[LL.EVI["mkt_fee"]], o[2], jp, jm, y, *x, float(fam == 0), float(fam == 2),
                         *[r[LL.EVI[c]] for c in EXTRA]])
    E = np.asarray(rows, dtype=np.float64).reshape(-1, len(ROWV))
    rev = np.zeros(3)
    if Q is not None:
        br = R31.beta_prev(pb, pa, t["prev_close_bar"])
        Qr = LL.prepare(b, a, cb)
        if Qr is not None and np.isfinite(br):
            er = R31.events(Qr, R31.d_pair(b, a, s, br), k)
            rev = np.array([len(er), er[:, LL.EVI["mkt_mid"]].sum() if len(er) else 0.0, er[:, LL.EVI["mkt_pnl"]].sum() if len(er) else 0.0])
    return E, rev


def work(args):
    i, t = args
    E, rev = pairday(i, t, *M.bars_of(t))
    return i, E.astype(np.float32), rev


def collect(T, path, workers):
    """Event rows of every pair-day into path (float32, written once) and the direction-check sums."""
    rev = np.zeros((len(T), 3))
    pool = mp.get_context("spawn").Pool(workers) if workers else None
    try:
        with open(path, "xb") as fh:
            it = pool.imap(work, list(enumerate(T)), chunksize=8) if pool else map(work, list(enumerate(T)))
            for i, E, rv in it:
                E.tofile(fh)
                rev[i] = rv
    finally:
        if pool:
            pool.close()
            pool.join()
    return np.fromfile(path, dtype=np.float32).reshape(-1, len(ROWV)).astype(np.float64), rev


# ---------------------------------------------------------------- tests and tables
def g_hat(R, m):
    return MD.predict(m["beta"], R[:, [VI[c] for c in M.XCOLS]], m["bounds"]) if len(R) else np.zeros(0)


def boot(R, top, seed=SEED):
    """H1-31 and H2-31 with the pair-day cluster bootstrap (research 30G's boot with this study's seed)."""
    t, inv = np.unique(R[:, VI["task"]].astype(np.int64), return_inverse=True)
    v = R[:, VI["mkt_mid"]]
    s_all, n_all = np.bincount(inv, weights=v), np.bincount(inv).astype(np.float64)
    s_top, n_top = np.bincount(inv, weights=v * top), np.bincount(inv, weights=top.astype(np.float64))
    G = len(t)
    rng = np.random.default_rng(seed)
    h1, h2 = [], []
    for _ in range(B // CHUNK):
        ix = rng.integers(0, G, size=(CHUNK, G))
        ma = s_all[ix].sum(axis=1) / n_all[ix].sum(axis=1)
        nt = n_top[ix].sum(axis=1)
        mt = np.where(nt > 0, s_top[ix].sum(axis=1) / np.where(nt > 0, nt, 1), np.nan)
        h1.append(ma)
        h2.append(mt - ma)
    h1, h2 = np.concatenate(h1), np.concatenate(h2)
    h2 = h2[np.isfinite(h2)]                       # a resample without a top event has no H2 statistic (DEVIATIONS31 3.6)
    m_all = float(v.mean())
    m_top = float(v[top].mean()) if top.any() else float("nan")
    a = R[:, VI["mkt_pnl"]]
    return ({"mean_before_per_event": m_all, "ci95": [float(x) for x in np.percentile(h1, [2.5, 97.5])], "p_one_sided": float(np.mean(h1 <= 0)),
             "after_same_events": float(a.mean()), "events": int(len(v)), "pair_days": int(G)},
            {"mean_top": m_top, "mean_all": m_all, "difference": m_top - m_all,
             "ci95": [float(x) for x in np.percentile(h2, [2.5, 97.5])] if len(h2) else [float("nan")] * 2,
             "p_one_sided": float(np.mean(h2 <= 0)) if len(h2) else float("nan"), "resamples_h2": int(len(h2)),
             "top_events": int(top.sum()), "top_share": float(top.mean()),
             "top_after": float(a[top].mean()) if top.any() else float("nan")})


def summ(R, m):
    if not m.any():
        return {"events": 0}
    return {"events": int(m.sum()), "before": float(R[m, VI["mkt_mid"]].mean()), "after_L3D1": float(R[m, VI["mkt_pnl"]].mean()),
            "after_L0D0": float(R[m, VI["mkt_pnl_L0D0"]].mean())}


def aux(R, g, top, T, bounds):
    """Auxiliary tables (not judged)."""
    if not len(R):
        return {"all": {"events": 0}}
    ti = R[:, VI["task"]].astype(np.int64)
    yr = np.array([int(t["date"][:4]) for t in T])[ti]
    every = np.ones(len(R), bool)
    jf = R[:, VI["j_filled"]] > 0
    out = {"all": summ(R, every),
           "delays": {f"d{dl}": {"before": float(R[:, VI[f'd{dl}_mid']].mean()), "after_L3D1": float(R[:, VI[f'd{dl}_pnl']].mean())} for dl in (0, 5)},
           "j15": {"after": float(R[:, VI["j_pnl"]].mean()), "before": float(R[:, VI["j_mid"]].mean()), "fill_rate": float(jf.mean()),
                   "after_filled": float(R[jf, VI["j_pnl"]].mean()) if jf.any() else None},
           "top": summ(R, top),
           "by_family": {f: summ(R, R[:, VI["family"]] == v) for f, v in M.FAM.items()},
           "by_year": {str(y): summ(R, yr == y) for y in sorted(set(yr.tolist()))},
           "bucket_0935_1000": summ(R, R[:, VI["bucket"]] == 0)}
    y = R[:, VI["y"]]
    ok = np.isfinite(y)
    yc = np.clip(y, *bounds["y"])
    out["g_corr_clipped"] = float(np.corrcoef(g[ok], yc[ok])[0, 1]) if ok.sum() > 2 else None
    q = np.quantile(g, np.linspace(0, 1, 11))
    dec = []
    for j in range(10):
        m = (g >= q[j]) & ((g < q[j + 1]) if j < 9 else (g <= q[j + 1]))
        dec.append({"decile": j + 1, **summ(R, m), "g_mean": float(g[m].mean()) if m.any() else None,
                    "y_clipped_mean": float(yc[m & ok].mean()) if (m & ok).any() else None,
                    "spread_bp": float(R[m, VI["spread_bp"]].mean()) if m.any() else None,
                    "spread_ticks": float(R[m, VI["spread_ticks"]].mean()) if m.any() else None})
    out["g_deciles"] = dec
    return out


def decide(mode, h1, h2):
    """SPEC31 4-5: validation -> the sealed run only when the H1-31 point estimate is > 0; sealed -> H1-31 and H2-31 at
    one-sided alpha 0.05 (the H2 clause is added to the H1 sentence only)."""
    after = h1["after_same_events"]
    if mode == "sealed":
        ok1, ok2 = h1["p_one_sided"] < ALPHA, h2["p_one_sided"] < ALPHA
        sent = (SENT["h1"] + (", " + SENT["h2"] if ok2 else "")) if ok1 else None
        return {"H1_31_confirmed": bool(ok1), "H2_31_confirmed": bool(ok2), "H3_31": "not judged (no candidate)", "sentence": sent,
                "after_cost_per_event_same_events": after, "trials_added": 2, "trials_total": TRIALS_BEFORE + 2}
    go = h1["mean_before_per_event"] > 0
    return {"go_sealed": bool(go), "sentence": None if go else SENT["val_stop"], "after_cost_per_event_same_events": after,
            "H1_p_below_alpha": bool(h1["p_one_sided"] < ALPHA), "H2_p_below_alpha": bool(h2["p_one_sided"] < ALPHA),
            "H3_31": "not judged (no candidate)"}


# ---------------------------------------------------------------- the run
def run(mode, workers):
    info = guard(mode)
    t0 = time.time()
    w = WINDOWS[mode]
    T, days, miss = M.tasks(w)
    if any(not (w[0] <= t["date"] <= w[1]) for t in T):
        raise Refused("a date outside the window")
    os.makedirs(LOCAL, exist_ok=True)
    R, rev = collect(T, os.path.join(LOCAL, f"{mode}31_events.f32"), workers)
    t1 = time.time()
    models = model()
    mb = models[BASE_NAME]
    g = g_hat(R, mb)
    top = g >= mb["threshold_bp"]
    h1, h2 = boot(R, top)
    side = aux(R, g, top, T, mb["bounds"])
    rv = rev.sum(axis=0)
    side["direction_check"] = {"events": int(rv[0]), "before": float(rv[1] / rv[0]) if rv[0] else None, "after": float(rv[2] / rv[0]) if rv[0] else None}
    t2 = time.time()
    idx = np.arange(len(T))
    rr = M.run_rule(T, idx, {key(n): models[n] for n in PICKS}, [key(n) for n in PICKS], workers)
    rng = np.random.default_rng(SEED)
    rule = {}
    for n in PICKS:
        rows = rr[key(n)]
        np.save(os.path.join(LOCAL, f"{mode}31_rule_{n}.npy"), np.asarray(rows, dtype=np.float64).reshape(-1, 5))
        rule[n] = M.rule_stats(rows, len(T), idx, days, rng)
    with open(os.path.join(LOCAL, f"{mode}31_tasks.tsv"), "x", encoding="utf-8") as fh:
        fh.write("task\tdate\tfamily\tleader\tfollower\n" + "".join(f"{i}\t{t['date']}\t{t['family']}\t{t['leader']}\t{t['follower']}\n" for i, t in enumerate(T)))
    out = {"mode": mode, "window": list(w), "pair_days": len(T), "days": len(days), "not_run": miss,
           "pair_days_by_family": {f: int(sum(t["family"] == f for t in T)) for f in M.FAM},
           "H1_31": h1, "H2_31": h2, "aux": side, "rule_R_aux": rule, "threshold_bp": mb["threshold_bp"], "model_sha256": MODEL_SHA,
           "collection": info, "decision": decide(mode, h1, h2),
           "runtime_s": {"events": round(t1 - t0, 1), "tests": round(t2 - t1, 1), "total": round(time.time() - t0, 1)},
           "disk": {"free_gib_after": shutil.disk_usage(LD.ROOT).free / 1024 ** 3}, "written": dt.datetime.now().isoformat(timespec="seconds")}
    p_sum, p_rep = outputs(mode)
    with open(p_sum, "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=C24.jdefault)
    L = report(out)
    with open(p_rep, "x", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return out


def f3(x):
    return "없음" if x is None else f"{x:+.3f}"


def report(o):
    h1, h2, a, d = o["H1_31"], o["H2_31"], o["aux"], o["decision"]
    name = {"rehearsal": "리허설 (2015–2018, 이미 본 자료)", "val": "검증 (2019–2020, 한 번)", "sealed": "봉인 (2021-01-04 ~ 2026-11-16, 한 번)"}[o["mode"]]
    L = [f"# 연구 31 · {name}", "",
         f"- 쌍·일 {o['pair_days']:,} (X {o['pair_days_by_family']['X']:,}, P {o['pair_days_by_family']['P']:,}, M {o['pair_days_by_family']['M']:,}), 거래일 {o['days']}, 돌리지 않은 쌍·일 {o['not_run']}.",
         "- 기본 설정 s15 k3, 세 가족 합침. 사건 한 번 = 후행 1만 달러, 시장가 진입 1초 지연, 기준 청산, 값은 달러/사건. 모델은 동결본(재적합 없음).", "",
         f"- **H1-31 (비용 전 사건당 평균 > 0):** {h1['mean_before_per_event']:+.3f} [{h1['ci95'][0]:+.3f}, {h1['ci95'][1]:+.3f}], p {h1['p_one_sided']:.4f}, "
         f"사건 {h1['events']:,}; 같은 사건의 비용 뒤 {h1['after_same_events']:+.3f}",
         f"- **H2-31 (상위 − 전체 > 0):** {h2['mean_top']:+.3f} − {h2['mean_all']:+.3f} = {h2['difference']:+.3f} [{h2['ci95'][0]:+.3f}, {h2['ci95'][1]:+.3f}], "
         f"p {h2['p_one_sided']:.4f} (상위 사건 {h2['top_events']:,}, {100 * h2['top_share']:.1f}%, 비용 뒤 {h2['top_after']:+.3f})", ""]
    if o["mode"] == "sealed":
        sent = f"\"{d['sentence']}\"" if d["sentence"] else "없음 (SPEC31 5절에 이 결과의 문장이 없다)"
        L += [f"- **판정:** H1-31 {'확인' if d['H1_31_confirmed'] else '확인 안 됨'}, H2-31 {'확인' if d['H2_31_confirmed'] else '확인 안 됨'}, H3-31 판정 안 함(후보 없음)",
              f"- **문장:** {sent} (같은 사건의 비용 뒤 {d['after_cost_per_event_same_events']:+.3f}달러/사건)",
              f"- 시행 수 {d['trials_total']} (+{d['trials_added']})"]
    else:
        L += [f"- **게이트:** H1-31 점추정 {h1['mean_before_per_event']:+.3f} → " + ("봉인으로 간다" if d["go_sealed"] else f"\"{d['sentence']}\"")
              + f" (같은 사건의 비용 뒤 {d['after_cost_per_event_same_events']:+.3f}달러/사건)",
              f"- 참고(판정 아님): H1 p < 0.05 {'예' if d['H1_p_below_alpha'] else '아니오'}, H2 p < 0.05 {'예' if d['H2_p_below_alpha'] else '아니오'}. H3는 후보가 없어 판정하지 않는다."]
    L += ["", "| 보조 (판정 아님) | 사건 | 비용 전 | 비용 뒤 L3D1 | 비용 뒤 L0D0 |", "|---|---|---|---|---|"]

    def line(lab, s):
        if s.get("events"):
            L.append(f"| {lab} | {s['events']:,} | {s['before']:+.3f} | {s['after_L3D1']:+.3f} | {s['after_L0D0']:+.3f} |")
        else:
            L.append(f"| {lab} | 0 | | | |")
    line("전체", a["all"])
    for f, s in a["by_family"].items():
        line(f"가족 {f}", s)
    for y, s in a["by_year"].items():
        line(y, s)
    line("09:35–10:00", a["bucket_0935_1000"])
    line("ĝ 상위", a["top"])
    L += ["", "| 지연 | 비용 전 | 비용 뒤 L3D1 |", "|---|---|---|", f"| 1초 (기본) | {a['all']['before']:+.3f} | {a['all']['after_L3D1']:+.3f} |"]
    for k_, v in a["delays"].items():
        L.append(f"| {k_[1:]}초 | {v['before']:+.3f} | {v['after_L3D1']:+.3f} |")
    j = a["j15"]
    dc = a["direction_check"]
    L += ["", f"- J15 (합류 지정가 15초): 비용 전 {j['before']:+.3f}, 비용 뒤 {j['after']:+.3f}, 체결률 {100 * j['fill_rate']:.1f}%",
          f"- 방향 진단(선행·후행 뒤집음): 사건 {dc['events']:,}, 비용 전 {f3(dc['before'])}, 비용 뒤 {f3(dc['after'])}",
          f"- ĝ 상관(잘린 y): {f3(a.get('g_corr_clipped'))}", "",
          "| ĝ 10분위 | 사건 | ĝ | 잘린 y | 비용 전 | 비용 뒤 | 스프레드 bp | 스프레드 틱 |", "|---|---|---|---|---|---|---|---|"]
    for r in a.get("g_deciles", []):
        if r.get("events"):
            L.append(f"| {r['decile']} | {r['events']:,} | {r['g_mean']:+.3f} | {f3(r['y_clipped_mean'])} | {r['before']:+.3f} | {r['after_L3D1']:+.3f} | {r['spread_bp']:.2f} | {r['spread_ticks']:.2f} |")
    L += ["", "| 판단 규칙 R (보조, 탐색에서 고른 4개) | 거래 | 거래당 비용 전 $ | 거래당 비용 뒤 $ | 하루 손익 $ | 95% CI |", "|---|---|---|---|---|---|"]
    for n, r in o["rule_R_aux"].items():
        L.append(f"| {n} | {r['trades']:,} | {r['mid_per_trade']:+.2f} | {r['pnl_per_trade']:+.2f} | {r['mean_daily_usd']:+.2f} | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] |")
    c = o["collection"]
    L += ["", f"- 수집 확인: 계획 {c['planned']:,}, MANIFEST {c['manifest_rows']:,}, MANIFEST sha256 {c['manifest_sha256'][:16]}…",
          f"- 모델 sha256 {o['model_sha256'][:16]}…, 상위 경계 {o['threshold_bp']:.4f} bp, 시간 {o['runtime_s']['total']:.0f}초, 남은 공간 {o['disk']['free_gib_after']:.1f} GiB."]
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g_ = ap.add_mutually_exclusive_group(required=True)
    for m_ in WINDOWS:
        g_.add_argument(f"--{m_}", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a_ = ap.parse_args()
    mode_ = [m_ for m_ in WINDOWS if getattr(a_, m_)][0]
    try:
        run(mode_, a_.workers)
    except Refused as e:
        print(f"REFUSED: {e}")
        sys.exit(2)
