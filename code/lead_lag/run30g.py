#!/usr/bin/env python3
"""Research 30G (SPEC30G; DEVIATIONS30G 1): does the follower's before-cost move hold on unseen periods?
  events   research 30's events for s = 15, k = 3 (leadlag30.events, unchanged): market entry 1 s after the decision, the
           base exit (phi 1.0, T 120 s, stop, 15:55, market exit), 10,000 dollars; value before costs = map30's "before"
           column (mkt_mid), after costs = mkt_pnl (L3D1 + fee)
  g-hat    research 30B's frozen model (results/model30b.json, s15_k3 g, sha256 checked), input 7 (SPY) = 0 in every period
           (DEVIATIONS30G 1); top 10% = g-hat >= threshold30g.json (2015-2016 90th percentile)
  H1-G     mean before-cost value per event > 0;  H2-G  mean of the top-10% events - mean of all events > 0
  test     pair-day cluster bootstrap (10,000, seed 32), one-sided p = share of resampled statistics <= 0, alpha 0.05
Modes (each once; outputs never overwritten):
  ../research15/py15.sh run30g.py --rehearsal   2015-2018 (seen data): the whole judgment path
  ../research15/py15.sh run30g.py --val         2019-2020 (needs the tag pre-val-30g)
  ../research15/py15.sh run30g.py --sealed      research 24's sealed bars 2021-01-04..2026-11-10 (needs the tag
                                                code-freeze-30g and Kyle's phrase in UNLOCK30G.txt)"""
import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for _p in (os.path.join(PROJ, "research30b"), os.path.join(PROJ, "research30")):
    if _p not in sys.path:
        sys.path.insert(1, _p)
import leadlag30 as LL  # noqa: E402
import model30b as MD  # noqa: E402

L24, C24 = LL.L24, LL.C24
RES = os.path.join(HERE, "results")
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research30g_data")
MODEL = os.path.join(PROJ, "research30b", "results", "model30b.json")
MODEL_SHA = "4f51096fb5d098d8ea5b3521196d62bf7e1252c69ac1d17ed86f85a5cbac7a06"
THRESH = os.path.join(HERE, "threshold30g.json")
UNLOCK = os.path.join(HERE, "UNLOCK30G.txt")
PHRASE = "연구30G 테스트를 한 번 연다"
VAL_TAG, FREEZE_TAG = "pre-val-30g", "code-freeze-30g"
SEALED_BARS = os.path.join(os.environ.get("DATA_DIR", "data"), "research24_data/sealed/bars_1s")
SEALED_MANIFEST = os.path.join(PROJ, "research24", "SEALED_MANIFEST24.tsv")
S, K = 15, 3.0
WINDOWS = {"rehearsal": ("20150102", "20181231"), "val": ("20190102", "20201231"), "sealed": ("20210104", "20261110")}
B, CHUNK, SEED, ALPHA = 10_000, 500, 32, 0.05
COLS = ("task", "u", "mkt_mid", "mkt_pnl", "mkt_pnl_L0D0", "d0_mid", "d0_pnl", "d5_mid", "d5_pnl", "g")
CI_ = {c: i for i, c in enumerate(COLS)}
SENT = {"h1": "먼저 움직인 종목을 보면, 따라올 종목은 처음 보는 기간에도 실제로 따라왔다(비용 전)",
        "h2": "그리고 예측 모델이 고른 사건은 더 많이 따라왔다",
        "h1_no": "따라오는 현상은 처음 보는 기간에 확인되지 않았다",
        "val_stop": "따라오는 현상은 검증 구간에서 이미 보이지 않았다. 봉인 데이터는 열지 않았다"}


class Refused(Exception):
    pass


def tag_ok(tag):
    r = subprocess.run(["git", "-C", PROJ, "merge-base", "--is-ancestor", tag, "HEAD"], capture_output=True)
    return r.returncode == 0


def unlock_ok():
    return os.path.isfile(UNLOCK) and open(UNLOCK, encoding="utf-8").read().strip() == PHRASE


def outputs(mode):
    return [os.path.join(RES, f"{mode}30g_{x}") for x in ("summary.json", "report.md")]


def guard(mode):
    """Refuse a run that is not allowed; nothing sealed is named before this passes."""
    if any(os.path.exists(p) for p in outputs(mode)):
        raise Refused(f"{mode}: the outputs exist, the run has been made")
    if hashlib.sha256(open(MODEL, "rb").read()).hexdigest() != MODEL_SHA:
        raise Refused("model30b.json differs from its frozen sha256")
    if mode == "val" and not tag_ok(VAL_TAG):
        raise Refused(f"validation needs the tag {VAL_TAG}")
    if mode == "sealed":
        why = []
        if not tag_ok(FREEZE_TAG):
            why.append(f"no tag {FREEZE_TAG}")
        if not unlock_ok():
            why.append("no unlock phrase in UNLOCK30G.txt")
        v = os.path.join(RES, "val30g_summary.json")
        if not os.path.exists(v) or not json.load(open(v, encoding="utf-8"))["decision"]["go_sealed"]:
            why.append("the validation did not send the study to the sealed run")
        if why:
            raise Refused("; ".join(why))


def plan(mode):
    w = WINDOWS[mode]
    if mode == "sealed":
        tasks, days, skipped = L24.plan(L24.manifest(SEALED_MANIFEST), w)
        return tasks, days, SEALED_BARS, True
    tasks, days, _ = L24.dev_plan()
    return [t for t in tasks if w[0] <= t["date"] <= w[1]], [d for d in days if w[0] <= d <= w[1]], L24.DEV_ROOT, False


def model():
    m = json.load(open(MODEL, encoding="utf-8"))["s15_k3"]
    return m["g"]["beta"], m["bounds"]


def pairday(i, t, a, b, beta, bounds):
    """Event rows (COLS) of one pair-day."""
    Q = LL.prepare(a, b, t["close_bar"])
    if Q is None:
        return np.zeros((0, len(COLS)))
    ev = LL.events(Q, S, K)
    if not len(ev):
        return np.zeros((0, len(COLS)))
    d = LL.dseries(a, b, S)
    sd, _ = LL.sigma(d)
    BI = MD.bar_inputs(a, b, None, S, d, sd, t["family"] == "D2")              # no SPY: input 7 = 0
    X = np.array([MD.event_inputs(BI, int(u), int(dr), t["close_bar"])[0] for u, dr in zip(ev[:, LL.EVI["u"]], ev[:, LL.EVI["dir"]])])
    g = MD.predict(beta, X, bounds)
    E = LL.EVI
    return np.column_stack([np.full(len(ev), i), ev[:, E["u"]], ev[:, E["mkt_mid"]], ev[:, E["mkt_pnl"]], ev[:, E["mkt_pnl_L0D0"]],
                            ev[:, E["d0_mid"]], ev[:, E["d0_pnl"]], ev[:, E["d5_mid"]], ev[:, E["d5_pnl"]], g])


def work(args):
    i, t, root, allow_sealed, beta, bounds = args
    a, b = L24.load(root, t["row_a"], allow_sealed), L24.load(root, t["row_b"], allow_sealed)
    return pairday(i, t, a, b, beta, bounds)


def collect(tasks, root, allow_sealed, workers):
    beta, bounds = model()
    args = [(i, t, root, allow_sealed, beta, bounds) for i, t in enumerate(tasks)]
    if workers:
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(work, args, chunksize=16)
    else:
        res = [work(x) for x in args]
    return np.concatenate(res) if res else np.zeros((0, len(COLS)))


# ---------------------------------------------------------------- statistics
def boot(R, thr, seed=SEED):
    """H1-G and H2-G with the pair-day cluster bootstrap: per resample the pair-days are drawn with replacement and the
    per-event means are taken over the drawn pair-days' events."""
    t, inv = np.unique(R[:, CI_["task"]].astype(np.int64), return_inverse=True)
    v = R[:, CI_["mkt_mid"]]
    top = R[:, CI_["g"]] >= thr
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
    m_all = float(v.mean())
    m_top = float(v[top].mean()) if top.any() else float("nan")
    return ({"mean_before_per_event": m_all, "ci95": [float(x) for x in np.percentile(h1, [2.5, 97.5])], "p_one_sided": float(np.mean(h1 <= 0)),
             "events": int(len(v)), "pair_days": int(G)},
            {"mean_top": m_top, "mean_all": m_all, "difference": m_top - m_all, "ci95": [float(x) for x in np.nanpercentile(h2, [2.5, 97.5])],
             "p_one_sided": float(np.nanmean(h2 <= 0)), "top_events": int(top.sum()), "top_share": float(top.mean()) if len(top) else float("nan")})


def side_tables(R, tasks, thr):
    """Auxiliary (not judged): after costs, L0D0, delays 0 and 5, family, year, the 09:35-10:00 bucket, g-hat deciles."""
    fam = np.array([tasks[int(i)]["family"] for i in R[:, CI_["task"]]]) if len(R) else np.zeros(0, dtype=str)
    yr = np.array([int(tasks[int(i)]["date"][:4]) for i in R[:, CI_["task"]]]) if len(R) else np.zeros(0, dtype=int)

    def summ(m):
        if not m.any():
            return {"events": 0}
        return {"events": int(m.sum()), "before": float(R[m, CI_["mkt_mid"]].mean()), "after_L3D1": float(R[m, CI_["mkt_pnl"]].mean()),
                "after_L0D0": float(R[m, CI_["mkt_pnl_L0D0"]].mean())}
    every = np.ones(len(R), bool)
    out = {"all": {**summ(every), "delay0_before": float(R[:, CI_["d0_mid"]].mean()) if len(R) else None,
                   "delay0_after": float(R[:, CI_["d0_pnl"]].mean()) if len(R) else None,
                   "delay5_before": float(R[:, CI_["d5_mid"]].mean()) if len(R) else None,
                   "delay5_after": float(R[:, CI_["d5_pnl"]].mean()) if len(R) else None},
           "top10": summ(R[:, CI_["g"]] >= thr) if len(R) else {"events": 0},
           "by_family": {f: summ(fam == f) for f in ST_FAM},
           "by_year": {str(y): summ(yr == y) for y in sorted(set(yr.tolist()))},
           "bucket_0935_1000": summ(R[:, CI_["u"]] < 1799) if len(R) else {"events": 0}}
    if len(R):
        g = R[:, CI_["g"]]
        q = np.quantile(g, np.linspace(0, 1, 11))
        dec = []
        for k in range(10):
            m = (g >= q[k]) & ((g < q[k + 1]) if k < 9 else (g <= q[k + 1]))
            dec.append({"decile": k + 1, **summ(m), "g_mean": float(g[m].mean()) if m.any() else None})
        out["g_deciles"] = dec
    return out


ST_FAM = LL.ST24.FAMILIES


def decide(mode, h1, h2, after, judge_h2=None):
    """SPEC30G 2-3: validation -> go to the sealed run when the H1-G point estimate is > 0 (H2-G judged there only when its
    validation point estimate is > 0 too); sealed -> H1-G (and H2-G when judged) at one-sided alpha 0.05."""
    if mode == "sealed":
        ok1 = h1["p_one_sided"] < ALPHA
        ok2 = bool(judge_h2 and ok1 and h2["p_one_sided"] < ALPHA)
        sent = (SENT["h1"] + (", " + SENT["h2"] if ok2 else "")) if ok1 else SENT["h1_no"]
        return {"H1_G_confirmed": bool(ok1), "H2_G_judged": bool(judge_h2), "H2_G_confirmed": ok2, "sentence": sent,
                "after_cost_per_event_same_events": after, "trials_added": 2 if judge_h2 else 1}
    go = h1["mean_before_per_event"] > 0
    return {"go_sealed": bool(go), "judge_H2_in_sealed": bool(go and h2["difference"] > 0), "sentence": None if go else SENT["val_stop"],
            "after_cost_per_event_same_events": after}


def run(mode, workers):
    guard(mode)
    t0 = time.time()
    thr = json.load(open(THRESH, encoding="utf-8"))["threshold_bp"]
    tasks, days, root, allow_sealed = plan(mode)
    if mode != "sealed" and any(t["date"] > WINDOWS[mode][1] for t in tasks):
        raise Refused("a date outside the window")
    R = collect(tasks, root, allow_sealed, workers)
    os.makedirs(LOCAL, exist_ok=True)
    np.save(os.path.join(LOCAL, f"{mode}30g_events.npy"), R.astype(np.float32))
    h1, h2 = boot(R, thr)
    side = side_tables(R, tasks, thr)
    out = {"mode": mode, "window": list(WINDOWS[mode]), "pair_days": len(tasks), "days": len(days), "H1_G": h1, "H2_G": h2, "aux": side,
           "threshold_bp": thr, "model_sha256": MODEL_SHA, "runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")}
    judge_h2 = (json.load(open(os.path.join(RES, "val30g_summary.json"), encoding="utf-8"))["decision"]["judge_H2_in_sealed"]
                if mode == "sealed" else None)
    out["decision"] = decide(mode, h1, h2, side["all"].get("after_L3D1"), judge_h2)
    os.makedirs(RES, exist_ok=True)
    p_sum, p_rep = outputs(mode)
    with open(p_sum, "x", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=C24.jdefault)
    L = report(out)
    with open(p_rep, "x", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    return out


def report(o):
    h1, h2, a, d = o["H1_G"], o["H2_G"], o["aux"], o["decision"]
    title = {"rehearsal": "리허설 (2015–2018, 본 자료)", "val": "검증 (2019–2020, 한 번)", "sealed": "봉인 (2021-01-04 ~ 2026-11-10, 한 번)"}[o["mode"]]
    L = [f"# 연구 30G · {title}", "", f"- 쌍·일 {o['pair_days']:,}, 사건 {h1['events']:,} (s 15, k 3, 시장가 진입 1초 지연, 기준 청산). 값은 1만 달러 사건당 달러.", "",
         f"- **H1-G (비용 전 사건당 평균 > 0):** {h1['mean_before_per_event']:+.3f} [{h1['ci95'][0]:+.3f}, {h1['ci95'][1]:+.3f}], p {h1['p_one_sided']:.4f}",
         f"- **H2-G (ĝ 상위 10% − 전체 > 0):** {h2['mean_top']:+.3f} − {h2['mean_all']:+.3f} = {h2['difference']:+.3f} [{h2['ci95'][0]:+.3f}, {h2['ci95'][1]:+.3f}], "
         f"p {h2['p_one_sided']:.4f} (상위 사건 {h2['top_events']:,}, {100 * h2['top_share']:.1f}%)",
         f"- 같은 사건의 비용 뒤(L3D1): 전체 {a['all'].get('after_L3D1', float('nan')):+.3f}, 상위 10% {a['top10'].get('after_L3D1', float('nan')):+.3f}. "
         f"L0D0 {a['all'].get('after_L0D0', float('nan')):+.3f}. 지연 0초 비용 전 {a['all'].get('delay0_before', float('nan')):+.3f}, 5초 {a['all'].get('delay5_before', float('nan')):+.3f}"]
    if o["mode"] == "sealed":
        L.append(f"- **판정:** \"{d['sentence']}\" (같은 사건의 비용 뒤 {d['after_cost_per_event_same_events']:+.3f}달러/사건)")
    else:
        L.append(f"- **판단:** " + ("H1-G 점추정 > 0 → 봉인으로 간다" + (" (H2-G도 판정)" if d["judge_H2_in_sealed"] else " (H2-G는 0 이하라 봉인에서는 H1-G만 판정)")
                                  if d["go_sealed"] else f"\"{d['sentence']}\"") + f" (같은 사건의 비용 뒤 {d['after_cost_per_event_same_events']:+.3f}달러/사건)")
    L += ["", "| 구분 | 사건 | 비용 전 | 비용 뒤 (L3D1) |", "|---|---|---|---|"]
    for k, v in list(a["by_family"].items()) + list(a["by_year"].items()) + [("09:35–10:00", a["bucket_0935_1000"])]:
        if v.get("events"):
            L.append(f"| {k} | {v['events']:,} | {v['before']:+.3f} | {v['after_L3D1']:+.3f} |")
    if "g_deciles" in a:
        L += ["", "| ĝ 분위 | 평균 ĝ bp | 사건 | 비용 전 | 비용 뒤 |", "|---|---|---|---|---|"]
        for x in a["g_deciles"]:
            L.append(f"| {x['decile']} | {x['g_mean']:+.2f} | {x['events']:,} | {x['before']:+.3f} | {x['after_L3D1']:+.3f} |")
    L.append(f"- 시간 {o['runtime_s']:.0f}초. 경계 {o['threshold_bp']:.4f}bp, 모델 sha256 {o['model_sha256'][:16]}…")
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for m in WINDOWS:
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    run(next(m for m in WINDOWS if getattr(a, m)), a.workers)
