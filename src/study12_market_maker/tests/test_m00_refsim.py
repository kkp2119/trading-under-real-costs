"""Kernel vs the independent reference implementation (tests/refsim.py, written from SPEC + E1 only, before the kernel
and without seeing it): bit-identical fills, orders and counters on random synthetic symbol-days."""
import numpy as np
import pytest

import features
from config import MODELS
from kernelio import kern, ref
from synth import random_day, settings

SETTINGS = settings()
N_DAYS, PER_DAY = 300, 20


def day_kwargs(seed):
    """Varied synthetic conditions: early close, late first quote, quotes freezing early (halt-like), thin trading,
    wide/locked quotes, many jumps."""
    r = np.random.default_rng(10_000 + seed)
    kw = {"p_trade": float(r.choice([0.02, 0.1, 0.35])), "p_jump": float(r.choice([0.002, 0.01, 0.05])),
          "spread": [(1, 2), (1, 6), (3, 12)][int(r.integers(3))], "p_wide": float(r.choice([0.0, 0.01, 0.05])),
          "p_lock": float(r.choice([0.0, 0.01, 0.05])), "p_odd": float(r.choice([0.0, 0.15, 0.5]))}
    close_bar = 12600 if r.random() < 0.15 else 23400
    if r.random() < 0.15:
        kw["first_state"] = int(r.integers(0, 2000))
    if r.random() < 0.15:
        kw["until"] = int(r.integers(5000, close_bar - 1))
    return kw, close_bar, bool(r.random() > 0.05)


@pytest.mark.parametrize("chunk", range(10))
def test_random_days_match_reference(chunk):
    bad = []
    for seed in range(chunk * N_DAYS // 10, (chunk + 1) * N_DAYS // 10):
        kw, close_bar, gate = day_kwargs(seed)
        b = random_day(seed, close_bar=close_bar, **kw)
        f = features.compute(b)
        r = np.random.default_rng(20_000 + seed)
        for i in r.choice(len(SETTINGS) * len(MODELS), PER_DAY, replace=False):
            name, cfg, cd = SETTINGS[i // len(MODELS)]
            m = MODELS[i % len(MODELS)]
            if kern(b, cfg, m, close_bar, gate, f) != ref(b, cd, m, close_bar, gate):
                bad.append((seed, name, m))
    assert not bad, bad[:10]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_full_grid_matches_reference(seed):
    b = random_day(500 + seed, p_trade=[0.35, 0.1, 0.02][seed - 1])
    f = features.compute(b)
    bad = [(n, m) for n, c, d in SETTINGS for m in MODELS if kern(b, c, m, 23400, True, f) != ref(b, d, m)]
    assert not bad, bad
