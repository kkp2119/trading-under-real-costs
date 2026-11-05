#!/usr/bin/env python3
"""Research 17 step 2 development run (SPEC17 3; step-2 prompt 1-2): candidates C0-C5 on the existing universe 2015-2020
through engine17, read through research15/loader15 (development bars only; no research 17 sealed file and no research
12/15/16 sealed or test folder is read). Regression first (stop when any check fails, no summary written):
  parents equal to the 16-D table; S0 / S1 / S2i / E* per-parent integer outputs equal to the stored 16-D outputs (L0 and L3);
  mean IS (R) at 3 decimals equal to 16-C / 16-D (L0D0 and L3D1); vol_pre equal to the 16-E development variables.
Then per candidate and variant (L3D1 primary, L0D0): mean and p95 IS, candidate - C0 and candidate - S0 with 95% CIs (day
blocks 21, 10,000, seed 17), yearly candidate - C0, limit / stop / S2i-branch shares; A* by SPEC17 3; the H2 power flag.
  python dev17.py run [--workers 8]
Per-parent outputs -> $DATA_DIR/research17_data/dev/, aggregates -> results/dev17_*."""
import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, "study15_ml_execution"))
import engine17 as G  # noqa: E402
import loader15 as L  # noqa: E402

RES = os.path.join(HERE, "results")
LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research17_data/dev")
R16LOCAL = os.path.join(os.environ.get("DATA_DIR", "data"), "research16_data")
YEARS = tuple(range(2015, 2021))
BASE = ("S0", "S1", "S2i")
TARGETS = {"L0D0": {"S0": 3.785, "S1": 3.737, "S2i": 2.674, "Estar": 2.605},
           "L3D1": {"S0": 4.579, "S1": 3.737, "S2i": 3.112, "Estar": 3.148}}
QTY16_METHODS = ("S0", "S1", "S2", "S2i", "Estar")
TRIALS_BEFORE, TRIALS_ADDED = 722, 6
NAMES = {"C0": "E*", "C1": "S2i (단순 대기)", "C2": "변동성 전환", "C3": "스프레드 전환", "C4": "E* + 손절", "C5": "변동성 전환 + 손절"}


# ---------------------------------------------------------------- simulation
def task(t):
    year, row, close_bar = t
    b = L.load(year, row)
    meta, outs = G.symday(b, row["date"], close_bar)
    return row["sym"], row["date"], meta, outs


def collect(results):
    """(par, OUT[len(LRULES), len(RUN_NAMES), N, NCOL], {(sym, date): day volume}, dropped parents) in task order."""
    par = pd.DataFrame({"date": np.concatenate([[d] * len(m["dir"]) for _, d, m, _ in results]),
                        "sym": np.concatenate([[s] * len(m["dir"]) for s, _, m, _ in results]),
                        **{k: np.concatenate([m[k] for _, _, m, _ in results]) for k in ("dir", "start", "mid2", "spread", "vol_pre")}})
    OUT = np.concatenate([o for _, _, _, o in results], axis=2)
    dayvol = {(s, d): m["day_vol"] for s, d, m, _ in results}
    return par, OUT, dayvol, int(sum(m["n_dropped"] for _, _, m, _ in results))


def simulate(workers):
    tasks = []
    for y in YEARS:
        man, cb = L.manifest(y), L.close_bars(y)
        tasks += [(y, r, cb[d]) for (s, d), r in sorted(man.items())]
    assert all(r["date"] <= "20201231" for _, r, _ in tasks), "date after 2020"
    with mp.get_context("spawn").Pool(workers) as pool:
        res = list(pool.imap(task, tasks, chunksize=8))
    par, OUT, dayvol, dropped = collect(res)
    os.makedirs(LOCAL, exist_ok=True)
    np.savez_compressed(os.path.join(LOCAL, "dev17_parents.npz"), out=OUT, runs=np.array(G.RUN_NAMES), lrules=np.array(G.LRULES))
    par.to_csv(os.path.join(LOCAL, "dev17_parents.csv.gz"), index=False)
    pd.Series(dayvol).rename_axis(["sym", "date"]).rename("day_vol").to_csv(os.path.join(LOCAL, "dev17_dayvol.csv.gz"))
    return par, OUT, {"symbol_days": len(tasks), "dropped_parents": dropped}


# ---------------------------------------------------------------- regression (16-C / 16-D / 16-E)
def regression(par, OUT):
    z = np.load(os.path.join(R16LOCAL, "qty16_parents.npz"))
    p16 = pd.read_csv(os.path.join(R16LOCAL, "qty16_parents.csv.gz"), dtype={"date": str, "sym": str})
    chk = {"parents_equal_16d": bool(len(par) == len(p16) and all(np.array_equal(par[c].to_numpy(), p16[c].to_numpy())
                                                                  for c in p16.columns))}
    lr16 = list(z["lrules"])
    chk["per_parent_equal_16d"] = {f"L{lr}": {m: bool(np.array_equal(OUT[G.LRULES.index(lr), G.RUN_NAMES.index(m), :, :7],
                                                                     z["out"][lr16.index(lr), QTY16_METHODS.index(m)]))
                                              for m in ("S0", "S1", "S2i", "Estar")} for lr in G.LRULES}
    dirs, mid2 = par.dir.to_numpy(), par.mid2.to_numpy()
    chk["mean_is_3dp"] = {}
    for lab, lr, dep in G.VARIANTS:
        for m, want in TARGETS[lab].items():
            got = float(G.costs(OUT[G.LRULES.index(lr), G.RUN_NAMES.index(m)], dirs, mid2, dep).mean())
            chk["mean_is_3dp"][f"{lab} {m}"] = {"got": got, "want": want, "pass": round(got, 3) == want}
    v16 = pd.read_csv(os.path.join(R16LOCAL, "decomp16_parents.csv.gz"), usecols=["period", "date", "sym", "start", "vol_pre_bp"],
                      dtype={"date": str, "sym": str})
    v16 = v16[v16.period == "dev"].reset_index(drop=True)
    same_rows = len(v16) == len(par) and all(np.array_equal(v16[c].to_numpy(), par[c].to_numpy()) for c in ("date", "sym", "start"))
    a, b = par.vol_pre.to_numpy(), v16.vol_pre_bp.to_numpy()
    chk["vol_pre_equal_16e"] = {"rows_aligned": bool(same_rows),
                                "pass": bool(same_rows and np.array_equal(np.isnan(a), np.isnan(b)) and
                                             np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-12, atol=1e-12)),
                                "undefined": int(np.isnan(a).sum())}
    shares = all(bool(np.all(OUT[li, ri, :, 0] + OUT[li, ri, :, 2] == G.QTY)) for li in range(len(G.LRULES))
                 for ri in range(len(G.RUN_NAMES)))
    nostop = all(bool(np.all(OUT[li, G.RUN_NAMES.index(m), :, 7] == 0)) for li in range(len(G.LRULES))
                 for m in ("S0", "S1", "S2i", "Estar"))
    chk["shares_sum_to_qty"], chk["no_stop_in_runs_without_stop"] = shares, nostop
    chk["stop_tests_skipped_invalid_quote"] = {f"L{lr}": int(OUT[G.LRULES.index(lr), :, :, 8].sum()) for lr in G.LRULES}
    chk["no_date_after_2020"] = bool(par.date.max() <= "20201231")
    ok = (chk["parents_equal_16d"] and all(all(v.values()) for v in chk["per_parent_equal_16d"].values())
          and all(v["pass"] for v in chk["mean_is_3dp"].values()) and chk["vol_pre_equal_16e"]["pass"] and shares and nostop
          and chk["no_date_after_2020"])
    return ok, chk


# ---------------------------------------------------------------- metrics
def variant_tables(par, OUT, lab, lr, dep, codes, n_days, idx):
    """{method: IS}, {method: out} for the baselines and the six candidates of one variant, and the S2i-branch flags."""
    o = OUT[G.LRULES.index(lr)]
    cand, flags = G.compose(o, par.vol_pre.to_numpy(), par.spread.to_numpy())
    outs = {**{m: o[G.RUN_NAMES.index(m)] for m in BASE}, **cand}
    IS = {m: G.costs(v, par.dir.to_numpy(), par.mid2.to_numpy(), dep) for m, v in outs.items()}
    return IS, outs, flags


def rows_for(par, IS, outs, flags, lab, codes, n_days, idx):
    yr = par.date.str[:4].to_numpy()
    rows = []
    for m in BASE + G.CANDS:
        r = {"variant": lab, "method": m, "n_parents": int(len(IS[m])), "mean_is_c": float(IS[m].mean()),
             "p95_is_c": float(np.percentile(IS[m], 95))}
        for c in ("C0", "S0"):
            est, lo, hi = G.diff_ci(IS[m] - IS[c], codes, n_days, idx)
            r.update({f"minus_{c}_c": est, f"minus_{c}_lo": lo, f"minus_{c}_hi": hi})
        o = outs[m]
        r.update({"limit_frac": float((o[:, 0] / G.QTY).mean()), "stop_parent_frac": float((o[:, 7] > 0).mean()),
                  "stop_share_frac": float((o[:, 7] / G.QTY).mean()), "deadline_parent_frac": float((o[:, 6] > 0).mean()),
                  "s2i_branch_frac": float(flags[m].mean()) if m in flags else float(m == "S2i"),
                  "mean_done_s": float(o[:, 5].mean())})
        r.update({f"minus_C0_{y}": float((IS[m] - IS["C0"])[yr == str(y)].mean()) for y in YEARS})
        rows.append(r)
    return rows


def h2_power(par, d_astar, power):
    """SPEC17 4 / step-2 prompt: |A* - C0| (development, L3D1) against the step-1 MDE (Holm 0.025); the same variance model
    (power17) on A* - C0 itself is given for reference."""
    import power17 as PW
    from scipy.stats import norm
    mde1 = power["h"]["H2 proxy: S2i - E*"]["mde"]["0.025"]["new_universe_c"]
    eff = float(d_astar.mean())
    sd, day, W = PW.series(d_astar, par)
    n_dev = float(sd.groupby(level=0).size().mean())
    V = float(day.var(ddof=1))
    vN = W / power["n_new_per_year"] + (V - W / n_dev)
    f = PW.vif(day)
    se = float(np.sqrt(vN * f / power["t_new_days"]))
    ref = {a: (norm.ppf(1 - a) + norm.ppf(0.8)) * se for a in (0.05, 0.025)}
    return {"effect_astar_minus_c0_c": eff, "mde_step1_holm_0.025_c": mde1, "underpowered": bool(abs(eff) < mde1),
            "reference_same_model_on_astar_minus_c0": {"se_new_c": se, "VIF": f, "mde_0.05_c": ref[0.05], "mde_0.025_c": ref[0.025],
                                                       "underpowered_at_0.025": bool(abs(eff) < ref[0.025])}}


def analyse(par, OUT, info, t0):
    ok, chk = regression(par, OUT)
    os.makedirs(RES, exist_ok=True)
    if not ok:
        with open(os.path.join(RES, "dev17_checks_FAILED.json"), "w") as fh:
            json.dump(chk, fh, indent=1, default=str)
        raise SystemExit("a regression check failed: results/dev17_checks_FAILED.json (no summary written)")
    days = sorted(par.date.unique())
    codes = pd.Categorical(par.date, categories=days).codes
    idx = G.boot_index(len(days))
    rows, IS_all = [], {}
    for lab, lr, dep in G.VARIANTS:
        IS, outs, flags = variant_tables(par, OUT, lab, lr, dep, codes, len(days), idx)
        IS_all[lab] = IS
        rows += rows_for(par, IS, outs, flags, lab, codes, len(days), idx)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "dev17_rows.csv"), index=False, float_format="%.6f")
    prim = {r["method"]: {"mean": r["mean_is_c"], "p95": r["p95_is_c"]} for r in rows if r["variant"] == "L3D1" and r["method"] in G.CANDS}
    astar, sel = G.select_astar(prim)
    power = json.load(open(os.path.join(RES, "power17.json")))
    h2 = None if astar == "C0" else h2_power(par, IS_all["L3D1"][astar] - IS_all["L3D1"]["C0"], power)
    out = {"astar": astar, "astar_name": NAMES[astar], "selection": sel,
           "q2": "개발에서 개선 없음 (H2 없음)" if astar == "C0" else f"A* = {astar} ({NAMES[astar]})",
           "h2_power": h2, "trials_before": TRIALS_BEFORE, "trials_added": TRIALS_ADDED,
           "trials_cumulative": TRIALS_BEFORE + TRIALS_ADDED, "rows": rows, "checks": chk, "sim": info,
           "n_parents": int(len(par)), "n_days": len(days), "boot": {"B": G.BOOT_B, "block": G.BOOT_BLOCK, "seed": G.BOOT_SEED},
           "runtime_s": round(time.time() - t0, 1), "written": dt.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(RES, "dev17_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str, ensure_ascii=False)
    report(out)
    print(open(os.path.join(RES, "dev17_report.md")).read())


# ---------------------------------------------------------------- report
def report(out):
    R = pd.DataFrame(out["rows"])
    L_ = [f"# 연구 17 · 2단계 개발 실행 (기존 유니버스 2015–2020, C0–C5)", "",
          f"- 주문 {out['n_parents']:,}건, {out['n_days']}일, 종목·일 {out['sim']['symbol_days']:,}개. 봉인 데이터는 읽지 않았다.",
          f"- CI: 일 블록 21, 10,000회, seed 17. 시행 수 +{out['trials_added']}, 누적 {out['trials_cumulative']}.", "",
          f"**A* = {out['astar']} ({out['astar_name']})**", ""]
    for lab in ("L3D1", "L0D0"):
        L_ += [f"## {lab}: 평균 IS, p95, 후보 − C0, 후보 − S0 (¢/주)", "",
               "| 방법 | 평균 IS | p95 | − C0 [95% CI] | − S0 [95% CI] | 지정가 | 손절 | S2i 분기 |", "|---|---|---|---|---|---|---|---|"]
        for _, r in R[R.variant == lab].iterrows():
            L_.append(f"| {r.method} | {r.mean_is_c:.3f} | {r.p95_is_c:.2f} | {r.minus_C0_c:+.3f} [{r.minus_C0_lo:+.3f}, {r.minus_C0_hi:+.3f}] | "
                      f"{r.minus_S0_c:+.3f} [{r.minus_S0_lo:+.3f}, {r.minus_S0_hi:+.3f}] | {r.limit_frac:.3f} | {r.stop_parent_frac:.3f} | "
                      f"{r.s2i_branch_frac:.3f} |")
        L_ += ["", f"### {lab} 연도별 후보 − C0", "", "| 방법 | " + " | ".join(str(y) for y in YEARS) + " |", "|---" * (len(YEARS) + 1) + "|"]
        for _, r in R[(R.variant == lab) & R.method.isin(G.CANDS)].iterrows():
            L_.append(f"| {r.method} | " + " | ".join(f"{r[f'minus_C0_{y}']:+.3f}" for y in YEARS) + " |")
        L_.append("")
    s = out["selection"]
    L_ += ["## A* 선택 (SPEC17 3절, L3D1)", "", f"- 평균 IS 최저 {s['lowest_mean']} ({s['lowest_mean_c']:.3f}¢), 0.02¢ 안 동률 {s['tied_within_0.02']}, "
           f"그중 p95 최저 {s['lowest_p95_among_tied']}, 규칙 수 {s['rule_count']} → **{s['astar']}**", ""]
    h = out["h2_power"]
    if h:
        rf = h["reference_same_model_on_astar_minus_c0"]
        L_ += ["## H2 검정력", "", f"- 개발 효과 A* − C0 = {h['effect_astar_minus_c0_c']:+.4f}¢, 1단계 MDE(Holm 0.025) {h['mde_step1_holm_0.025_c']:.4f}¢ → "
               f"**{'검정력 부족' if h['underpowered'] else '검정력 충분'}**",
               f"- 참고: 같은 분산 모형을 A* − C0에 적용한 MDE {rf['mde_0.025_c']:.4f}¢ (α 0.025), {rf['mde_0.05_c']:.4f}¢ (α 0.05), VIF {rf['VIF']:.2f}", ""]
    else:
        L_ += ["## H2", "", "- A* = C0: Q2는 개발에서 개선 없음, H2 없음", ""]
    c = out["checks"]
    L_ += ["## 회귀 점검", "", f"- 주문표 = 16-D: {c['parents_equal_16d']}, 주문별 출력 = 16-D: {c['per_parent_equal_16d']}",
           "- 평균 IS 3자리: " + ", ".join(f"{k} {v['got']:.3f}" for k, v in c["mean_is_3dp"].items()),
           f"- vol_pre = 16-E: {c['vol_pre_equal_16e']}, 1,000주 합 {c['shares_sum_to_qty']}, 손절 없는 실행의 손절 0 {c['no_stop_in_runs_without_stop']}",
           f"- 호가 0으로 손절 확인을 건너뛴 결정 시점: {c['stop_tests_skipped_invalid_quote']}", f"- 실행 {out['runtime_s']} s"]
    with open(os.path.join(RES, "dev17_report.md"), "w") as fh:
        fh.write("\n".join(L_) + "\n")


def main(workers):
    t0 = time.time()
    par, OUT, info = simulate(workers)
    print(f"simulated {len(par):,} parents, {info['symbol_days']:,} symbol-days, {time.time() - t0:.0f} s", flush=True)
    analyse(par, OUT, info, t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.workers)
