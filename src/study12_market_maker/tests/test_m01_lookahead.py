"""M1 no lookahead: changing or blanking every bar after d leaves the orders of decisions <= d, the fills of bars <= d,
the counters' prefix and every feature value at bars <= d bit-identical; a day's own bars do not change its gate."""
import numpy as np
import pytest

import features
import session
from config import MODELS
from kernelio import kern
from synth import random_day, settings

SETTINGS = settings()
FEATS = features.FEATURES


def cut(k, d):
    return [x for x in k["orders"] if x[0] <= d], [x for x in k["fills"] if x[0] <= d]


def perturbed(b, d, how, seed):
    out = {c: v.copy() for c, v in b.items()}
    other = random_day(seed, p_trade=0.5, p_jump=0.05)
    for c in out:
        out[c][d + 1:] = other[c][d + 1:] if how == "change" else 0
    return out


@pytest.mark.parametrize("how", ["change", "blank"])
@pytest.mark.parametrize("seed", range(4))
def test_orders_fills_features_ignore_later_bars(seed, how):
    b = random_day(700 + seed, p_trade=0.2)
    d = int(np.random.default_rng(seed).integers(1900, 22000))
    b2 = perturbed(b, d, how, 900 + seed)
    f, f2 = features.compute(b), features.compute(b2)
    for k in FEATS:
        assert np.array_equal(f[k][:d + 1], f2[k][:d + 1]), k
    for name, cfg, _ in SETTINGS:
        for m in MODELS:
            assert cut(kern(b, cfg, m, f=f), d) == cut(kern(b2, cfg, m, f=f2), d), (name, m)


def test_gate_uses_previous_days_only(tmp_path):
    cal_csv = tmp_path / "calendar.csv"
    days = ["2019-03-01", "2019-03-04", "2019-03-05", "2019-03-06", "2019-03-07", "2019-03-08", "2019-03-11"]
    cal_csv.write_text("date,close_time\n" + "".join(f"{d},16:00\n" for d in ["2019-01-02"] + days))
    cal = session.Calendar(str(cal_csv))
    keys = [("X", d.replace("-", "")) for d in days]
    dv = {k: 4_000_000 * 10_000 for k in keys}
    g1 = session.gate(cal, "X", "20190311", dv, set(), set(keys))
    dv[("X", "20190311")] = 0                                        # the day itself
    assert session.gate(cal, "X", "20190311", dv, set(), set(keys)) == g1 is True
