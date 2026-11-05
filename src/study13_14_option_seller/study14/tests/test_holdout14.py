"""Research 14 shell tests (SPEC14 9) on synthetic data only. The development results the shell checks against are
made by the frozen run_dev on the synthetic world; the start-check sha256 constants are pointed at those files."""
import contextlib
import copy
import io
import json
import math
import os
import re
import subprocess

import numpy as np
import pandas as pd
import pytest

import accounting as A
import config13 as C
import ladder
import omdata as om
import results13 as R
import run_dev
import run_holdout
import run_holdout14 as H
import stats13
import synth13

# SPEC14 5 table (exact values) and 9 start checks
SPEC_D_DEV = {"O2": 0.254073199527745, "O8": 0.2639905548996458, "O9": 0.2675324675324675}
SPEC_THR = {"O2": 0.304073199527745, "O8": 0.3139905548996458, "O9": 0.3175324675324675}
SPEC_MDD = {"O2": 0.12206986737589562, "O8": 0.07828371970127149, "O9": 0.044234858519156345}
SPEC_TAIL = {"O2": 0.18310480106384341, "O8": 0.11742557955190723, "O9": 0.06635228777873452}
SPEC_DEV_SHA = {"dev_daily.csv": "b6822a0a00428959c046f21726f29957260f49b7c885de3a7305ae9728b5c7f8",
                "dev_summary.json": "568663f8d8b777fe799062fd2f82e53675ee88380a3e111587ce903b5f434136",
                "dev_settings.csv": "51fccd2a713c313136f6e8127ac9e481aabdf8bf1f2938fd608ae4a2e0a272d6"}
DATES = dict(DEV_START=20120103, DEV_END=20121231, HOLD_START=20130102, HOLD_END=20130329, F2_SPLIT=20120629,
             ENTRY_ON_LAST_DAY=False)


def new_dir():
    base = os.environ.get("R13_TEST_DIR")
    if not base:
        pytest.skip("set R13_TEST_DIR (outside the repository)")
    d = os.path.join(base, "s14_" + os.urandom(5).hex())
    os.makedirs(d)
    return d


def whitelisted(text):
    return all(any(re.match(p, l) for p in run_holdout.WHITELIST) for l in text.strip().splitlines() if l)


def run(**kw):
    """main() with the screen captured: (return value, screen text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        v = H.main(["--go"], **kw)
    return v, buf.getvalue()


def series(sr, T=3185, seed=0, scale=0.005):
    """Daily returns whose sample annualised Sharpe is exactly sr."""
    z = np.random.default_rng(seed).standard_normal(T)
    z = (z - z.mean()) / z.std(ddof=1)
    return scale * (z + sr / math.sqrt(C.ANN))


@pytest.fixture(scope="module")
def env():
    mp = pytest.MonkeyPatch()
    for k, v in DATES.items():
        mp.setattr(C, k, v)
    d = new_dir()
    w = synth13.World(start=20120103, end=20121231, hold_end=20130329, seed=41)
    dev = os.path.join(d, "dev_results")
    run_dev.main(["--go"], source=copy.deepcopy(w), results_dir=dev, ext_dir=os.path.join(d, "ext_dev"))
    yield w, dev, {f: R.sha256(os.path.join(dev, f)) for f in H.DEV_SHA}
    mp.undo()


def results_dir(base):
    rd = os.path.join(base, "results14")
    os.makedirs(rd)
    open(os.path.join(rd, om.UNLOCK), "w").close()
    return rd


@pytest.fixture
def h14(env, tmp14, monkeypatch):
    w, dev, shas = env
    monkeypatch.setattr(H, "DEV_SHA", dict(shas))
    rd = results_dir(tmp14)
    kw = dict(results_dir=rd, dev_results_dir=dev, has_tag_fn=lambda: True, is_clean_fn=lambda: True,
              data_fn=lambda: [], aux=(), ext_dir=os.path.join(tmp14, "ext"))
    return w, rd, kw


@pytest.fixture(scope="module")
def ran(env):
    """One full run (rehearsal a: out-of-range IV in the holdout; NDX descriptive run)."""
    w, dev, shas = env
    mp = pytest.MonkeyPatch()
    mp.setattr(H, "DEV_SHA", dict(shas))
    w2 = copy.deepcopy(w)
    k = w2.days.index(20130115)
    w2.iv[k:k + 3] = 5.0
    base = new_dir()
    rd = results_dir(base)
    kw = dict(results_dir=rd, dev_results_dir=dev, has_tag_fn=lambda: True, is_clean_fn=lambda: True,
              data_fn=lambda: [], aux=("NDX",), ext_dir=os.path.join(base, "ext"))
    v, out = run(source_factory=lambda: copy.deepcopy(w2), **kw)
    yield w2, rd, kw, v, out
    mp.undo()


# ---------------------------------------------------------------- constants (SPEC14 5, 9)
def test_constants_judged_level_bounds_and_hashes():
    assert H.JUDGED == ("O2", "O8", "O9") and H.LEVEL == 1 - 0.05 / 3
    assert H.D_DEV == SPEC_D_DEV and H.MDD_DEV == SPEC_MDD
    assert {s: max(C.DEFECT_FLOOR, H.D_DEV[s] + C.DEFECT_MARGIN) for s in H.JUDGED} == SPEC_THR
    assert {s: C.TAIL_RATIO * H.MDD_DEV[s] for s in H.JUDGED} == SPEC_TAIL
    assert H.DEV_SHA == SPEC_DEV_SHA
    assert H.MANIFEST13[1] == "58267d6c727b76ac4fed982b1b08a674f7d1a67d201b3e727a73750b30f4b162"
    assert H.SPEC14_SHA == R.sha256(os.path.join(H.HERE, "SPEC14.md"))
    assert H.TAGS == ("spec14-v1", "code-freeze-14") and H.RESULTS14 == os.path.join(H.HERE, "results14")


def test_bounds_equal_the_research13_development_files():
    sm = json.load(open(os.path.join(H.DEV_RESULTS, "dev_summary.json")))["metrics"]
    ds = pd.read_csv(os.path.join(H.DEV_RESULTS, "dev_settings.csv"), float_precision="round_trip").set_index("setting")
    for s in H.JUDGED:
        assert H.D_DEV[s] == sm[s]["defect_ratio"] == ds.loc[s, "defect_ratio"]
        assert H.MDD_DEV[s] == sm[s]["max_drawdown"] == ds.loc[s, "max_drawdown"]
    assert H.startup_checks(H.DEV_RESULTS) == []                      # committed files and frozen modules


def test_frozen_modules_equal_the_code_freeze_13_manifest():
    assert H.frozen_check() == []


def test_gitignore_keeps_the_unlock_file_out_of_the_tree():
    gi = open(os.path.join(H.R13, ".gitignore")).read().splitlines()
    assert "study14/results14/HOLDOUT_UNLOCKED" in gi
    if subprocess.run(["git", "-C", H.R13, "rev-parse"], capture_output=True).returncode == 0:
        p = os.path.join(H.RESULTS14, om.UNLOCK)
        assert subprocess.run(["git", "-C", H.R13, "check-ignore", "-q", p]).returncode == 0
        assert subprocess.run(["git", "-C", H.R13, "check-ignore", "-q",
                               os.path.join(H.RESULTS14, H.RESULT_FILES[0])]).returncode == 1


# ---------------------------------------------------------------- Bonferroni CI and verdicts (SPEC14 5)
def test_98_33_percentiles_come_from_the_same_resample_and_are_wider():
    r = series(0.5, seed=3)
    sh = stats13.boot_sharpes(r, stats13.boot_starts(len(r)))
    lo, hi = stats13.ci(r, level=H.LEVEL)
    a = (1 - H.LEVEL) / 2
    assert abs(100 * a - 100 / 120) < 1e-12                              # 0.8333 and 99.1667 percentiles
    assert (lo, hi) == (float(np.nanpercentile(sh, 100 * a)), float(np.nanpercentile(sh, 100 * (1 - a))))
    lo95, hi95 = stats13.ci(r)
    assert lo < lo95 and hi > hi95
    half = (hi - lo) / 2 / (2.394 * math.sqrt(C.ANN / len(r)))           # iid normal: z(0.99167) x SE
    assert 0.85 < half < 1.15


def test_verdict_judges_on_the_98_33_ci_not_the_95_ci():
    r = series(0.62, seed=5)
    lo95, _ = stats13.ci(r)
    lo98, _ = stats13.ci(r, level=H.LEVEL)
    assert lo95 > 0 >= lo98                                              # the two levels disagree here
    v = H.verdict("O2", r, 0.0)
    assert v["verdict"] == "c" and stats13.judge(stats13.sharpe(r), *stats13.ci(r))[0] == "a"
    assert v["ci_98_33"] == [lo98, stats13.ci(r, level=H.LEVEL)[1]]


def test_each_setting_uses_its_own_downgrade_and_tail_boundary():
    r = series(1.2, seed=7)
    r = r * (0.10 / stats13.max_drawdown(r[:-1]))                       # holdout MaxDD 0.10, Sharpe unchanged
    v = {s: H.verdict(s, r, 0.31) for s in H.JUDGED}                     # 0.3041 < 0.31 < 0.3140 < 0.3175
    assert v["O2"]["verdict"] == "c" and "결함일" in v["O2"]["reason"]
    assert v["O8"]["verdict"] == "a" and v["O8"]["label"] == "a-안정"       # 0.10 <= 0.1174
    assert v["O9"]["verdict"] == "a" and v["O9"]["label"] == "a-꼬리주의"   # 0.10 > 0.0664
    assert {s: v[s]["downgrade_threshold"] for s in H.JUDGED} == SPEC_THR
    assert {s: v[s]["tail_boundary"] for s in H.JUDGED} == SPEC_TAIL


# ---------------------------------------------------------------- start checks and refusals (screen only)
@pytest.mark.parametrize("i,f", [(1, "dev_daily.csv"), (2, "dev_summary.json"), (3, "dev_settings.csv")])
def test_each_start_check_refuses(h14, monkeypatch, i, f):
    w, rd, kw = h14
    monkeypatch.setitem(H.DEV_SHA, f, "0" * 64)
    v, out = run(source_factory=lambda: copy.deepcopy(w), **kw)
    assert v is None and f"refused: start check {i}: {f} sha256 differs" in out
    assert os.listdir(rd) == [om.UNLOCK] and whitelisted(out)


def test_start_check_4_frozen_modules(h14):
    w, rd, kw = h14
    v, out = run(source_factory=lambda: copy.deepcopy(w), frozen_fn=lambda: ["ladder.py differs"], **kw)
    assert v is None and "refused: start check 4: ladder.py differs" in out and os.listdir(rd) == [om.UNLOCK]


def test_each_refusal_condition_blocks_and_leaves_no_file(h14):
    w, rd, kw = h14
    kw = {**kw, "startup_fn": lambda d, f: []}
    ok = H.guard(rd, kw["has_tag_fn"], kw["is_clean_fn"], H.frozen_check, kw["data_fn"])
    assert ok == []
    cases = [({"has_tag_fn": lambda: False}, "refused: 2: no spec14-v1 or code-freeze-14 tag"),
             ({"is_clean_fn": lambda: False}, "refused: 3: research13 working tree not clean"),
             ({"frozen_fn": lambda: ["chain.py differs"]}, "refused: 5: chain.py differs"),
             ({"data_fn": lambda: ["6: MANIFEST.tsv differs from the repository copy"]},
              "refused: 6: MANIFEST.tsv differs from the repository copy"),
             ({"data_fn": lambda: ["7: holdout end date differs"]}, "refused: 7: holdout end date differs")]
    for over, msg in cases:
        v, out = run(source_factory=lambda: copy.deepcopy(w), **{**kw, **over})
        assert v is None and msg in out and whitelisted(out), msg
        assert os.listdir(rd) == [om.UNLOCK], msg
    open(os.path.join(rd, "holdout_summary14.json"), "w").close()
    v, out = run(source_factory=lambda: copy.deepcopy(w), **kw)
    assert v is None and "refused: 1: holdout results exist" in out
    assert sorted(os.listdir(rd)) == sorted([om.UNLOCK, "holdout_summary14.json"])
    os.remove(os.path.join(rd, "holdout_summary14.json"))
    os.remove(os.path.join(rd, om.UNLOCK))
    called = []
    v, out = run(source_factory=lambda: copy.deepcopy(w), **{**kw, "data_fn": lambda: called.append(1) or []})
    assert v is None and "refused: 4: no HOLDOUT_UNLOCKED in results14" in out and not called
    assert os.listdir(rd) == []


def test_needs_go(h14):
    w, rd, kw = h14
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert H.main([], source_factory=lambda: copy.deepcopy(w), **kw) is None
    assert "refused: needs --go" in buf.getvalue() and os.listdir(rd) == [om.UNLOCK]


# ---------------------------------------------------------------- development identity before any holdout file
def test_dev_mismatch_stops_before_any_holdout_file(env, tmp14, monkeypatch):
    w, dev, shas = env
    bad = os.path.join(tmp14, "dev_bad")
    os.makedirs(bad)
    for f in H.DEV_SHA:
        open(os.path.join(bad, f), "wb").write(open(os.path.join(dev, f), "rb").read())
    d = pd.read_csv(os.path.join(bad, "dev_daily.csv"), float_precision="round_trip")
    d.loc[d.index[-5], "O2_base"] += 1e-6
    d.to_csv(os.path.join(bad, "dev_daily.csv"), index=False)
    monkeypatch.setattr(H, "DEV_SHA", {**shas, "dev_daily.csv": R.sha256(os.path.join(bad, "dev_daily.csv"))})
    rd = results_dir(tmp14)
    src = copy.deepcopy(w)
    with pytest.raises(RuntimeError, match="development days differ"):
        run(source_factory=lambda: src, results_dir=rd, dev_results_dir=bad, has_tag_fn=lambda: True,
            is_clean_fn=lambda: True, data_fn=lambda: [], aux=(), ext_dir=os.path.join(tmp14, "ext"))
    assert not src.holdout_enabled
    assert not any(a[2] == "holdout" or (isinstance(a[2], int) and a[2] >= 2013) for a in src.access)
    assert not any(os.path.exists(os.path.join(rd, f)) for f in H.RESULT_FILES)


# ---------------------------------------------------------------- full run (rehearsal a) and results
def test_rehearsal_a_out_of_range_iv_does_not_stop(ran):
    w, rd, kw, v, out = ran
    assert v and set(v["verdicts"]) == set(H.JUDGED) and v["level"] == H.LEVEL
    s = json.load(open(os.path.join(rd, "holdout_summary14.json")))
    assert s["iv_out_of_range_days"] > 0 and "rv21_filled_signal_days" in s
    daily = pd.read_csv(os.path.join(rd, "holdout_daily14.csv"))
    assert daily.date.iloc[0] == C.HOLD_START and daily.date.iloc[-1] == C.HOLD_END and daily.shape[1] == 1 + 33
    assert whitelisted(out) and whitelisted(open(os.path.join(rd, H.LOG_FILE)).read())
    assert "phase 1 check: development days identical" in out and "P_" not in out


def test_results_written_once_atomically_with_required_items(ran):
    w, rd, kw, v, out = ran
    files = os.listdir(rd)
    assert all(f in files for f in H.RESULT_FILES) and not any(f.endswith(".part") for f in files)
    s = json.load(open(os.path.join(rd, "holdout_summary14.json")))
    assert s["spec14_sha256"] == H.SPEC14_SHA and s["judged"] == list(H.JUDGED) and len(s["metrics"]) == 11
    for sid in H.JUDGED:
        m = s["metrics"][sid]
        assert set(m["dsr"]) == {"11", "180"} and "F4_calmar_ge_0.33" in m and "cost_share_dev" in m
        assert set(m["audit"]) >= {"defect_i_days", "gap_days", "no_candidate_days", "fwd_spot_leg_days",
                                   "exdate_moved_leg_days"}
        assert set(m["ci_sens"]) == {"10", "42"} and len(m["ci_98_33"]) == 2
    assert set(s["audit"]["by_H"]) == {"10", "21"} and s["audit"]["by_H"]["21"]["disagreements"] == 0
    assert set(s["aux"]["NDX"]["metrics"]) == set(H.JUDGED) and set(s["data_fixes"]) == set(market_fix_keys())
    assert set(s["judged_correlation"]) == set(H.JUDGED) and "spx_buy_hold_sharpe" in s
    vd = json.load(open(os.path.join(rd, "holdout_verdict14.json")))
    assert {k: vd["verdicts"][k]["downgrade_threshold"] for k in H.JUDGED} == SPEC_THR
    assert {k: vd["verdicts"][k]["tail_boundary"] for k in H.JUDGED} == SPEC_TAIL
    ext = pd.read_csv(os.path.join(rd, "EXTERNAL.tsv"), sep="\t")
    assert len(ext) == 11 and all(p.startswith("holdout14/") for p in ext.path)


def market_fix_keys():
    import market
    return market.FIX_KEYS


def test_rehearsal_c_second_run_refused_on_the_screen_only(ran):
    w, rd, kw, v, out = ran
    log_before = open(os.path.join(rd, H.LOG_FILE)).read()
    v2, out2 = run(source_factory=lambda: copy.deepcopy(w), **kw)
    assert v2 is None and "refused: 1: holdout results exist" in out2
    assert open(os.path.join(rd, H.LOG_FILE)).read() == log_before


def test_rehearsal_b_invariant_stop_leaks_nothing(h14, monkeypatch):
    w, rd, kw = h14
    real = ladder.Engine.step

    def boom(self, v, dec, **k):
        if v.d >= 20130201:
            raise A.InvariantError("injected invariant violation")
        return real(self, v, dec, **k)
    monkeypatch.setattr(ladder.Engine, "step", boom)
    buf = io.StringIO()
    with pytest.raises(A.InvariantError), contextlib.redirect_stdout(buf):
        H.main(["--go"], source_factory=lambda: copy.deepcopy(w), **kw)
    assert not any(os.path.exists(os.path.join(rd, f)) for f in H.RESULT_FILES + ("EXTERNAL.tsv",))
    assert whitelisted(buf.getvalue()) and whitelisted(open(os.path.join(rd, H.LOG_FILE)).read())


def test_nothing_is_written_before_the_statistics_finish(h14, monkeypatch):
    w, rd, kw = h14

    def fail(*a, **k):
        raise RuntimeError("statistics failed")
    monkeypatch.setattr(stats13, "ci", fail)
    with pytest.raises(RuntimeError, match="statistics failed"):
        run(source_factory=lambda: copy.deepcopy(w), **kw)
    assert sorted(os.listdir(rd)) == sorted([om.UNLOCK, H.LOG_FILE])
