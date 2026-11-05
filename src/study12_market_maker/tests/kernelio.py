"""Kernel and refsim outputs in one comparable form (sorted fill and order tuples, counters)."""
import features
import simulator
from config import S_NDEC, S_NDEC_QB, S_NINT_B, S_NINT_S, S_NINTF_B, S_NINTF_S, S_NSTALE, S_QDF, S_SKIP
from refsim import simulate_ref


def kern(b, cfg, model, close_bar=23400, gate=True, f=None):
    f = features.compute(b) if f is None else f
    fills, orders, st = simulator.simulate(b, f, cfg.kernel_params(model, close_bar, gate))
    return {"skipped": int(st[S_SKIP]), "fills": sorted(map(tuple, fills.tolist())),
            "orders": sorted(map(tuple, orders[:, :5].tolist())), "q_df": int(st[S_QDF]), "n_dec": int(st[S_NDEC]),
            "n_dec_qbase": int(st[S_NDEC_QB]), "n_int": (int(st[S_NINT_B]), int(st[S_NINT_S])),
            "n_int_fill": (int(st[S_NINTF_B]), int(st[S_NINTF_S])), "n_stale": int(st[S_NSTALE])}


def ref(b, cfg_dict, model, close_bar=23400, gate=True):
    r = simulate_ref(b, close_bar, gate, cfg_dict, model)
    return {"skipped": int(r["skipped"]), "fills": sorted(tuple(int(v) for v in x) for x in r["fills"]),
            "orders": sorted(tuple(int(v) for v in x) for x in r["orders"]), "q_df": int(r["q_df"]),
            "n_dec": int(r["n_dec"]), "n_dec_qbase": int(r["n_dec_qbase"]), "n_int": tuple(int(v) for v in r["n_int"]),
            "n_int_fill": tuple(int(v) for v in r["n_int_fill"]), "n_stale": int(r["n_stale"])}
