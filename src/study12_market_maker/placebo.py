"""Filter placebo B5 (SPEC 12; E1-40): within one symbol-day the OBI (or FI) block signals of bars d0 .. d_f are permuted
together (buy and sell blocks move as a pair, undefined FI bars included), with rng = default_rng([12, r, sym_id, date]).
Run with model Q and cost R; the metric is net P&L per filled share."""
import hashlib

import numpy as np

from config import SEED

PLACEBO_DIMS = {"obi": ("obi_bb", "obi_bs"), "fi": ("fi_bb", "fi_bs")}
N_PLACEBO = 200


def sym_id(sym):
    return int(hashlib.sha256(sym.encode()).hexdigest()[:8], 16)


def permuted(feats, dim, r, sym, day, d0, df):
    """A copy of feats with the dimension's block pair permuted over bars d0 .. d_f (inclusive)."""
    rng = np.random.default_rng([SEED, r, sym_id(sym), int(day)])
    perm = d0 + rng.permutation(df - d0 + 1)
    out = dict(feats)
    for k in PLACEBO_DIMS[dim]:
        a = feats[k].copy()
        a[d0:df + 1] = feats[k][perm]
        out[k] = a
    return out
