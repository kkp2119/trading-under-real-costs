#!/usr/bin/env python3
"""Research 16 stage 0 gate (GATE16.md; implementation details DEVIATIONS16.md 1): short-term reversal on CRSP daily quote
midpoints, development period only (CRSP rows 2014-11-01 .. 2020-12-31; nothing later is queried).
  python gate16.py fetch     raw CRSP rows -> $DATA_DIR/research16_data/ (licensed data, local only)
  python gate16.py run       G, bootstrap CI, verdict, secondary statistics, checks -> research16/results/
WRDS through PostgreSQL client credentials only."""
import datetime as dt
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(os.environ.get("DATA_DIR", "data"), "research16_data")
RES = os.path.join(HERE, "results")
Q_LO, Q_HI = "2014-11-01", "2020-12-31"
SIG_LO, SIG_HI = 20150102, 20201230
MID_LO, MID_HI, SPR_LO, SPR_HI, DV_MIN, DV_WIN, MIN_N, FRAC = 10.0, 150.0, 3.0, 12.0, 5e6, 20, 40, 0.2
COST_ESTAR, COST_S0 = 5.2, 7.6
BOOT_B, BOOT_BLOCK, BOOT_SEED = 10_000, 21, 16
NAMES_SQL = f"""select permno, namedt, nameendt, shrcd, exchcd from crsp.dsenames
where permno in (select distinct permno from crsp.dsenames where shrcd in (10, 11) and exchcd in (1, 2, 3)
                 and nameendt >= date '{Q_LO}' and namedt <= date '{Q_HI}')
  and nameendt >= date '{Q_LO}' and namedt <= date '{Q_HI}'"""
DSF_SQL = """select permno, date, bid, ask, prc, vol, cfacpr from crsp.dsf
where date between date '{lo}' and date '{hi}' and permno in ({permnos})"""


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch():
    import wrds
    os.makedirs(RAW, exist_ok=True)
    db = wrds.Connection()
    try:
        names = db.raw_sql(NAMES_SQL)
        perm = sorted(names.permno.astype(int).unique())
        parts = []
        for y in range(2014, 2021):
            lo = Q_LO if y == 2014 else f"{y}-01-01"
            hi = f"{y}-12-31"
            for i in range(0, len(perm), 2000):
                ps = ",".join(str(p) for p in perm[i:i + 2000])
                parts.append(db.raw_sql(DSF_SQL.format(lo=lo, hi=hi, permnos=ps)))
            print(f"fetched {y}: {sum(len(p) for p in parts)} rows so far", flush=True)
    finally:
        db.close()
    d = pd.concat(parts, ignore_index=True)
    date = pd.to_datetime(d.date)
    assert date.max() <= pd.Timestamp(Q_HI) and date.min() >= pd.Timestamp(Q_LO)
    np.savez_compressed(os.path.join(RAW, "crsp_dsf.npz"), permno=d.permno.to_numpy(np.int32),
                        date=date.dt.strftime("%Y%m%d").astype(np.int32).to_numpy(),
                        **{c: pd.to_numeric(d[c], errors="coerce").to_numpy(np.float64) for c in ("bid", "ask", "prc", "vol", "cfacpr")})
    names.assign(namedt=pd.to_datetime(names.namedt).dt.strftime("%Y%m%d").astype(int),
                 nameendt=pd.to_datetime(names.nameendt).dt.strftime("%Y%m%d").astype(int)).to_csv(
        os.path.join(RAW, "crsp_dsenames.csv"), index=False)
    man = {"queried_utc": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}", "range": [Q_LO, Q_HI],
           "names_sql": NAMES_SQL, "dsf_sql": DSF_SQL, "rows": int(len(d)), "permnos": len(perm),
           "files": {f: sha(os.path.join(RAW, f)) for f in ("crsp_dsf.npz", "crsp_dsenames.csv")}}
    open(os.path.join(RAW, "MANIFEST.json"), "w").write(json.dumps(man, indent=1))
    print(f"fetch: {len(d)} rows, {len(perm)} permnos, dates {date.min():%Y-%m-%d} .. {date.max():%Y-%m-%d}")


def load():
    z = np.load(os.path.join(RAW, "crsp_dsf.npz"))
    d = pd.DataFrame({k: z[k] for k in z.files})
    names = pd.read_csv(os.path.join(RAW, "crsp_dsenames.csv"))
    assert (d.date < 20210101).all() and (names.namedt < 20210101).all()            # no date after 2020 in the data
    return d, names


def panel(d, names):
    """Date x permno matrices (NaN where no row) and the point-in-time common-stock mask."""
    dates = np.sort(d.date.unique())
    perms = np.sort(d.permno.unique())
    di, pi = np.searchsorted(dates, d.date), np.searchsorted(perms, d.permno)
    M = {}
    for c in ("bid", "ask", "prc", "vol", "cfacpr"):
        a = np.full((len(dates), len(perms)), np.nan)
        a[di, pi] = d[c].to_numpy()
        M[c] = a
    common = np.zeros((len(dates), len(perms)), bool)
    ok = names.shrcd.isin([10, 11]) & names.exchcd.isin([1, 2, 3])
    for r in names[ok & names.permno.isin(perms)].itertuples():
        j = np.searchsorted(perms, r.permno)
        common[(dates >= r.namedt) & (dates <= r.nameendt), j] = True
    return dates, perms, M, common


def build(d, names):
    """Positions table and daily universe counts (GATE16 2-3, DEVIATIONS16 1)."""
    dates, perms, M, common = panel(d, names)
    bid, ask, cfac = M["bid"], M["ask"], M["cfacpr"]
    with np.errstate(invalid="ignore"):
        valid = (bid > 0) & (ask > bid) & (cfac > 0)
    mid = np.where(valid, (bid + ask) / 2, np.nan)
    adj = mid / np.where(valid, cfac, np.nan)
    spr_c = np.round((ask - bid) * 100, 6)
    dvol = np.nan_to_num(np.abs(M["prc"]) * M["vol"], nan=0.0)
    cs = np.vstack([np.zeros((1, dvol.shape[1])), np.cumsum(dvol, axis=0)])
    dv20 = np.full(dvol.shape, np.nan)
    dv20[DV_WIN:] = (cs[DV_WIN:-1] - cs[:-DV_WIN - 1]) / DV_WIN                     # rows t-20 .. t-1
    s = np.full(adj.shape, np.nan)
    s[1:] = adj[1:] / adj[:-1] - 1
    with np.errstate(invalid="ignore"):
        U = (common & valid & (mid >= MID_LO) & (mid <= MID_HI) & (spr_c >= SPR_LO) & (spr_c <= SPR_HI)
             & (dv20 >= DV_MIN) & np.isfinite(s))
    rows, days = [], []
    for t in range(1, len(dates) - 1):
        if not (SIG_LO <= dates[t] <= SIG_HI):
            continue
        idx = np.flatnonzero(U[t])
        n = len(idx)
        days.append({"date": int(dates[t]), "n_universe": n, "traded": n >= MIN_N})
        if n < MIN_N:
            continue
        sd = s[t, idx] - s[t, idx].mean()
        order = idx[np.lexsort((perms[idx], sd))]                                    # ascending s, ties by permno
        k = int(math.floor(FRAC * n))
        for side, sel in ((1, order[:k]), (-1, order[n - k:])):
            r1 = adj[t + 1, sel] / adj[t, sel] - 1
            r2 = adj[t + 2, sel] / adj[t, sel] - 1 if t + 2 < len(dates) else np.full(len(sel), np.nan)
            rows.append(pd.DataFrame({"date": dates[t], "permno": perms[sel], "side": side, "mid": mid[t, sel],
                                      "spread_c": spr_c[t, sel], "s_dm": s[t, sel] - s[t, idx].mean(), "r1": r1, "r2": r2}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(days), (dates, perms, adj, mid, spr_c, dv20, common, valid, s)


def G(pos, r="r1"):
    p = pos[np.isfinite(pos[r])]
    return float((p.side * p[r]).sum() / (1 / p.mid).sum() * 100)


def G_loop(pos):
    num = den = 0.0
    for _, g in pos.groupby("date"):
        for side, r, m in zip(g.side.to_numpy(), g.r1.to_numpy(), g.mid.to_numpy()):
            if np.isfinite(r):
                num += side * r
                den += 1 / m
    return num / den * 100


def boot(pos, r="r1"):
    p = pos[np.isfinite(pos[r])]
    g = p.assign(a=p.side * p[r], b=1 / p.mid).groupby("date")[["a", "b"]].sum()
    a, b = g.a.to_numpy(), g.b.to_numpy()
    T = len(a)
    rng = np.random.default_rng(BOOT_SEED)
    k = math.ceil(T / BOOT_BLOCK)
    idx = (rng.integers(0, T, size=(BOOT_B, k))[:, :, None] + np.arange(BOOT_BLOCK)[None, None, :]).reshape(BOOT_B, -1)[:, :T] % T
    stat = a[idx].sum(axis=1) / b[idx].sum(axis=1) * 100
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi), int(T)


def hand_check(d, names, pos, n_dates=3):
    """Recompute the long/short lists and r1 on 3 random signal dates straight from the raw rows (long table, no panel)."""
    tr = np.sort(d.date.unique())
    sig = [x for x in tr if SIG_LO <= x <= SIG_HI and x in set(pos.date)]
    pick = sorted(np.random.default_rng(BOOT_SEED).choice(sig, n_dates, replace=False))
    out = []
    for t in pick:
        i = int(np.searchsorted(tr, t))
        prev, nxt, win = tr[i - 1], tr[i + 1], tr[i - DV_WIN:i]
        rows = d[d.date.isin([prev, t, nxt])].copy()
        rows["valid"] = (rows.bid > 0) & (rows.ask > rows.bid) & (rows.cfacpr > 0)
        rows["mid"] = (rows.bid + rows.ask) / 2
        rows["adj"] = rows.mid / rows.cfacpr
        cur = rows[(rows.date == t) & rows.valid].set_index("permno")
        prv = rows[(rows.date == prev) & rows.valid].set_index("permno")
        nx = rows[(rows.date == nxt) & rows.valid].set_index("permno")
        nm = names[(names.namedt <= t) & (names.nameendt >= t) & names.shrcd.isin([10, 11]) & names.exchcd.isin([1, 2, 3])]
        w = d[d.date.isin(win)].assign(dv=lambda x: (x.prc.abs() * x.vol).fillna(0)).groupby("permno").dv.sum() / DV_WIN
        c = cur[cur.index.isin(set(nm.permno)) & cur.index.isin(prv.index)].copy()
        c["spr"] = np.round((c.ask - c.bid) * 100, 6)
        c["dv20"] = w.reindex(c.index).fillna(0)
        c = c[(c.mid >= MID_LO) & (c.mid <= MID_HI) & (c.spr >= SPR_LO) & (c.spr <= SPR_HI) & (c.dv20 >= DV_MIN)]
        c["s"] = c.adj / prv.adj.reindex(c.index) - 1
        c["s"] -= c.s.mean()
        c = c.reset_index().sort_values(["s", "permno"])
        k = int(math.floor(FRAC * len(c)))
        longs, shorts = list(c.permno[:k]), list(c.permno[len(c) - k:])
        r1 = (nx.adj.reindex(longs + shorts) / c.set_index("permno").adj.reindex(longs + shorts) - 1).to_numpy()
        p = pos[pos.date == t]
        mine_l, mine_s = sorted(p[p.side == 1].permno), sorted(p[p.side == -1].permno)
        rr = p.set_index("permno").r1.reindex(longs + shorts).to_numpy()
        ok = sorted(longs) == mine_l and sorted(shorts) == mine_s and np.allclose(r1, rr, equal_nan=True, rtol=0, atol=1e-12)
        out.append({"date": int(t), "n_universe": int(len(c)), "k": k, "match": bool(ok)})
    return out


def verdict(g):
    if g < COST_ESTAR:
        return "멈춤", "G < 5.2: E*로도 못 살린다"
    if g < COST_S0:
        return "진행", "5.2 <= G < 7.6: E*가 손익을 뒤집을 수 있는 범위 (1단계 설계로)"
    return "진행", "G >= 7.6: 시장가로도 남는다; E*의 역할은 뒤집기가 아니라 더하기"


def run():
    t0 = dt.datetime.now()
    d, names = load()
    pos, days, _ = build(d, names)
    assert (pos.date < 20210101).all() and pos.date.max() <= SIG_HI
    g1 = G(pos)
    gl = G_loop(pos)
    assert round(g1, 6) == round(gl, 6), (g1, gl)
    lo, hi, T = boot(pos)
    v, why = verdict(g1)
    hc = hand_check(d, names, pos)
    assert all(h["match"] for h in hc), hc
    p1 = pos[np.isfinite(pos.r1)]
    spr_b = {}
    rc = p1.spread_c.round()
    for lab, a, b in (("3-5c", 3, 5), ("6-8c", 6, 8), ("9-12c", 9, 12)):
        q = p1[(rc >= a) & (rc <= b)]
        spr_b[lab] = {"G_cents": G(q) if len(q) else None, "positions": int(len(q))}
    yr = {str(y): {"G_cents": G(g), "positions": int(len(g))} for y, g in p1.groupby(p1.date // 10000)}
    p2 = pos[np.isfinite(pos.r2)]
    tr = days[days.traded]
    res = {"written": f"{dt.datetime.now():%Y-%m-%d %H:%M} ET", "gate_file_sha256": sha(os.path.join(HERE, "GATE16.md")),
           "trials_cumulative": 692, "raw_manifest": json.load(open(os.path.join(RAW, "MANIFEST.json")))["files"],
           "G_cents_round_trip": g1, "G_loop_check": gl, "ci95_cents": [lo, hi], "bootstrap": {"block": BOOT_BLOCK, "B": BOOT_B,
           "seed": BOOT_SEED, "days": T}, "costs_round_trip_cents": {"Estar": COST_ESTAR, "S0": COST_S0},
           "verdict": v, "verdict_reason": why,
           "G_bp": float((p1.side * p1.r1).sum() / len(p1) * 1e4),
           "secondary": {"by_spread": spr_b, "by_year": yr, "hold_2day_G_cents": G(pos, "r2"),
                         "hold_2day_positions": int(len(p2)), "hold_2day_excluded": int((~np.isfinite(pos.r2)).sum())},
           "signal_days": int(len(days)), "traded_days": int(len(tr)), "days_below_40": int((~days.traded).sum()),
           "universe_mean": float(tr.n_universe.mean()), "universe_min": int(tr.n_universe.min()), "universe_max": int(tr.n_universe.max()),
           "long_per_day_mean": float(pos[pos.side == 1].groupby("date").size().mean()),
           "short_per_day_mean": float(pos[pos.side == -1].groupby("date").size().mean()),
           "positions": int(len(pos)), "positions_used": int(len(p1)), "excluded_no_mid_t1": int((~np.isfinite(pos.r1)).sum()),
           "first_signal": int(pos.date.min()), "last_signal": int(pos.date.max()),
           "checks": {"G_vectorized_equals_loop_6dp": True, "hand_check_3_dates": hc, "no_date_after_2020": True},
           "dividends": "ignored (adjusted quote midpoints only)", "seconds": (dt.datetime.now() - t0).seconds}
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, "gate16_result.json"), "w").write(json.dumps(res, indent=1, ensure_ascii=False))
    rep = [f"**판정: {v}** — G = {g1:.3f}¢ (왕복, 주당, 비용 전), 95% 구간 [{lo:.3f}, {hi:.3f}]; {why}", "",
           f"- 작성 {res['written']}; GATE16.md sha256 `{res['gate_file_sha256']}`; 누적 시행 수 692",
           f"- 신호일 {res['signal_days']}일(2015-01-02 ~ 2020-12-30), 거래일 {res['traded_days']}일, 유니버스 40개 미만으로 쉰 날 "
           f"{res['days_below_40']}일; 하루 평균 유니버스 {res['universe_mean']:.1f}종목(최소 {res['universe_min']}, 최대 "
           f"{res['universe_max']}), 롱 {res['long_per_day_mean']:.1f} · 숏 {res['short_per_day_mean']:.1f}종목",
           f"- 포지션 {res['positions']}개, t+1 중간값이 없어 제외 {res['excluded_no_mid_t1']}개; G (bp) = {res['G_bp']:.3f}",
           f"- 비용 범위(연구 15 개발 구간 실측 왕복): E* 5.2¢, 시장가 S0 7.6¢", "- 배당은 무시했다(조정 호가 중간값만 사용).",
           "", "보조 통계 (판정에 쓰지 않음):", "", "| 구분 | G (¢) | 포지션 |", "|---|---|---|"]
    rep += [f"| 스프레드 {k} | {x['G_cents']:.3f} | {x['positions']} |" for k, x in spr_b.items()]
    rep += [f"| {y}년 | {x['G_cents']:.3f} | {x['positions']} |" for y, x in yr.items()]
    rep += [f"| 2일 보유 | {res['secondary']['hold_2day_G_cents']:.3f} | {res['secondary']['hold_2day_positions']} "
            f"(제외 {res['secondary']['hold_2day_excluded']}) |", "",
            f"대조 점검: G 벡터화 = 반복문(소수 6자리), 무작위 3일 손 계산 일치 {[h['date'] for h in hc]}, 2021년 이후 날짜 없음."]
    open(os.path.join(RES, "gate16_report.md"), "w").write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    {"fetch": fetch, "run": run}[sys.argv[1]]()
