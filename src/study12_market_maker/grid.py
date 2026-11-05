"""The closed setting grid (SPEC 5, 15; E1-34). Stage 1 = B0 + 18 one-dimension alternatives; stage 2 = C_all + reverts
from the adopted values (dev net Sharpe >= B0 + 0.5, best per dimension, ties -> first listed); duplicates of tested
settings are neither run again nor counted. N = 34 + distinct (setting, universe) pairs tested."""
import dataclasses
import math

from config import ADOPT_GAIN, A_NAIVE, B0, B_ETF, N_PRIOR

DIMENSIONS = {                               # dimension -> stage-1 alternative values, SPEC 5 table order
    "requote": [("reset", 1), ("reset", 15), ("reset", 60), ("event", 1)],
    "L": [1, 5],
    "stale_s": [30, 120],
    "long_only": [True],
    "obi": [True],
    "halt": [True],
    "fi": [True],
    "min_spread": [1, 3, 4],
    "gcnt_m": [5, 20],
    "window": [1],
}
MAX_STOCK = 30


def set_dim(cfg, dim, value):
    if dim == "requote":
        return dataclasses.replace(cfg, mode=value[0], delta=value[1])
    return dataclasses.replace(cfg, **{dim: value})


def get_dim(cfg, dim):
    return (cfg.mode, cfg.delta if cfg.mode == "reset" else 1) if dim == "requote" else getattr(cfg, dim)


def label(dim, v):
    if dim == "requote":
        return "event" if v[0] == "event" else f"reset{v[1]}"
    return f"{dim}={v}"


def stage1():
    """[(name, Config)]: B0 then the 18 alternatives."""
    return [("B0", B0)] + [(label(dim, v), set_dim(B0, dim, v)) for dim, vals in DIMENSIONS.items() for v in vals]


def adopted(sharpes):
    """sharpes: {Config: dev net Sharpe (Q, R)} for stage 1. Returns {dim: value} adopted (SPEC 5 stage 2)."""
    base = sharpes[B0]
    out = {}
    for dim, vals in DIMENSIONS.items():
        best, best_sr = None, -math.inf
        for v in vals:
            sr = sharpes.get(set_dim(B0, dim, v), float("nan"))
            if sr == sr and sr >= base + ADOPT_GAIN and sr > best_sr:          # NaN never adopted; ties keep the first
                best, best_sr = v, sr
        if best is not None:
            out[dim] = best
    return out


def stage2(adopt, tested):
    """[(name, Config)] new settings: C_all and the one-dimension reverts, minus settings already in `tested`."""
    if not adopt:
        return []
    c_all = B0
    for dim, v in adopt.items():
        c_all = set_dim(c_all, dim, v)
    cand = [("C_all", c_all)] + [(f"C_all-{dim}", set_dim(c_all, dim, get_dim(B0, dim))) for dim in adopt]
    seen, out = set(tested), []
    for name, c in cand:
        if c not in seen:
            out.append((name, c))
            seen.add(c)
    return out


def etf_controls(candidate=None):
    """Distinct ETF control settings (SPEC 5): B_etf, B0, the stock candidate (when there is one)."""
    out = []
    for name, c in [("B_etf", B_ETF), ("B0_etf", B0)] + ([("cand_etf", candidate)] if candidate is not None else []):
        if all(c != x for _, x in out):
            out.append((name, c))
    return out


def n_trials(stock_settings, etf_settings, mix=True, naive=True):
    """N = 34 + distinct stock settings + A_naive (stocks) + distinct ETF controls + the mix (SPEC 15)."""
    return N_PRIOR + len(set(stock_settings) - {A_NAIVE}) + int(naive) + len(set(etf_settings)) + int(mix)
