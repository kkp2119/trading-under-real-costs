"""M14 reproducibility: same input -> bit-identical results; 1 worker = 2 workers; numba = plain Python (NUMBA_DISABLE_JIT
in a subprocess); bar files written by numpy 2.3 (the downloader's environment) load identically under this numpy."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import loader
import runner
import session
from config import B0, MODELS
from fakesplit import write_split
from kernelio import kern
from synth import random_day, settings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MINICONDA = "/opt/miniconda3/bin/python"
SCRIPT = """
import json, sys
sys.path[:0] = [{root!r}, {tests!r}]
from kernelio import kern
from synth import random_day, settings
out = {{}}
for seed in (1, 2):
    b = random_day(3000 + seed, p_trade=0.3)
    for name, cfg, _ in settings()[::4]:
        for m in ("S", "T", "Q", "Q25"):
            out[f"{{seed}}|{{name}}|{{m}}"] = kern(b, cfg, m)
print(json.dumps(out))
"""


def _canon(d):
    return json.loads(json.dumps(d))


def test_same_input_twice_is_identical():
    b = random_day(3100)
    for name, cfg, _ in settings():
        assert kern(b, cfg, "Q") == kern(b, cfg, "Q"), name


def test_numba_equals_plain_python():
    env = dict(os.environ, NUMBA_DISABLE_JIT="1", PYTHONDONTWRITEBYTECODE="1")
    res = subprocess.run([sys.executable, "-c", SCRIPT.format(root=ROOT, tests=HERE)], env=env, capture_output=True,
                         text=True, check=True, timeout=3000)
    plain = json.loads(res.stdout.strip().splitlines()[-1])
    jitted = {}
    for seed in (1, 2):
        b = random_day(3000 + seed, p_trade=0.3)
        for name, cfg, _ in settings()[::4]:
            for m in ("S", "T", "Q", "Q25"):
                jitted[f"{seed}|{name}|{m}"] = kern(b, cfg, m)
    assert plain == _canon(jitted)


def test_workers_1_and_2_give_identical_tables(tmpdir_r12):
    days = [("AAA", "20190516"), ("BBB", "20190516"), ("AAA", "20190517"), ("BBB", "20190517")]
    write_split(tmpdir_r12, "dev", "stocks", [(s, d, None) for s, d in days], seed=40)
    cal_csv = os.path.join(tmpdir_r12, "calendar.csv")
    pd.DataFrame({"date": ["2019-05-16", "2019-05-17"], "close_time": "16:00"}).to_csv(cal_csv, index=False)
    cal = session.Calendar(cal_csv)
    split = loader.Split("dev", "stocks", base=tmpdir_r12, results_dir=tmpdir_r12)
    gates = {k: True for k in days}
    st = [(n, c) for n, c, _ in settings()[:3]]
    r1, d1 = runner.run_tasks(split, cal, days, gates, st, MODELS, n_workers=1, nesting_for=("B0",))
    r2, d2 = runner.run_tasks(split, cal, days, gates, st, MODELS, n_workers=2, nesting_for=("B0",))
    r1b, _ = runner.run_tasks(split, cal, days, gates, st, MODELS, n_workers=1, nesting_for=("B0",))
    pd.testing.assert_frame_equal(r1, r2)
    pd.testing.assert_frame_equal(d1, d2)
    pd.testing.assert_frame_equal(r1, r1b)
    q = r1[r1.model == "Q"]
    assert len(r1) == 4 * 3 * 5 and set(q[q.setting == "B0"].nested) == {1} and set(q[q.setting != "B0"].nested) == {-1}


@pytest.mark.skipif(not os.path.exists(MINICONDA), reason="downloader environment not present")
def test_npz_from_numpy_2_3_loads_identically(tmpdir_r12):
    code = ("import numpy as np, sys; r = np.random.default_rng(9); "
            "a = {c: r.integers(-2**31, 2**31 - 1, 23400, dtype=np.int64).astype(np.int32) for c in sys.argv[2:]}; "
            "f = open(sys.argv[1], 'wb'); np.savez_compressed(f, **a); f.close(); print(np.__version__)")
    path = os.path.join(tmpdir_r12, "x.npz")
    cols = list(random_day(1))
    v = subprocess.run([MINICONDA, "-I", "-c", code, path, *cols], capture_output=True, text=True, check=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1")).stdout.strip()
    assert v.startswith("2.")
    r = np.random.default_rng(9)
    expect = {c: r.integers(-2**31, 2**31 - 1, 23400, dtype=np.int64).astype(np.int32) for c in cols}
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == sorted(cols)
        assert all(z[c].dtype == np.int32 and z[c].tobytes() == expect[c].tobytes() for c in cols)
    assert np.__version__.startswith("1.26")
