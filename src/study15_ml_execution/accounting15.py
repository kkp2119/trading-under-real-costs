"""Research 15 copy of research12/accounting.py (only the import changed, R15). Costs and P&L of one symbol-day run from its fill rows (SPEC 8; E1-23, E1-24, E1-42). Decisions and fills do not depend
on costs, so scenarios R, R+, C are applied afterwards. Money in 1e-4 dollars (int); edges and inventory P&L in 2x units
so that 2 * cash = edge2_lim + edge2_mkt + inv2 holds exactly (M5)."""
import numpy as np

from config15 import COST, MARKOUT_K                                                # R15

ROW_KEYS = (["cash", "edge2_lim", "edge2_mkt", "inv2", "sh_lim_buy", "sh_lim_sell", "sh_mkt_buy", "sh_mkt_sell", "n_fill_ev",
             "flat_qty", "flat_side", "flat_price"]
            + [f"mk{k}_sum2" for k in MARKOUT_K] + [f"mk{k}_sh" for k in MARKOUT_K] + [f"mk{k}_excl" for k in MARKOUT_K]
            + [f"{b}_{x}" for b in ("open", "noon", "close") for x in ("sh", "mk5_sum2", "mk5_sh")])


def buckets(d0, df):
    """B4 time buckets as [lo, hi] bar ranges (E1-42): first hour, 11:30-12:30, last hour before d_f."""
    return {"open": (d0, 3599), "noon": (7200, 10799), "close": (df - 3599, df)}


def account(fills, mid2, mid_ok, close_bar, d0, df):
    """fills: int64 rows (u, side, kind, qty, price, ref_mid2, dec). Returns {ROW_KEYS: int}."""
    out = dict.fromkeys(ROW_KEYS, 0)
    if len(fills) == 0:
        return out
    u, s, kind, qty, p, ref = (fills[:, i] for i in range(6))
    lim, mkt = kind == 0, kind == 1
    out["cash"] = int(np.sum(-s * qty * p))
    out["edge2_lim"] = int(np.sum((s * qty * (ref - 2 * p))[lim]))
    out["edge2_mkt"] = int(np.sum((s * qty * (ref - 2 * p))[mkt]))
    out["inv2"] = int(np.sum(-s * qty * ref))
    out["sh_lim_buy"], out["sh_lim_sell"] = int(qty[lim & (s > 0)].sum()), int(qty[lim & (s < 0)].sum())
    out["sh_mkt_buy"], out["sh_mkt_sell"] = int(qty[mkt & (s > 0)].sum()), int(qty[mkt & (s < 0)].sum())
    out["n_fill_ev"] = int(lim.sum())
    flat = mkt & (fills[:, 6] == df)
    if flat.any():
        i = int(np.flatnonzero(flat)[0])
        out["flat_qty"], out["flat_side"], out["flat_price"] = int(qty[i]), int(s[i]), int(p[i])
    li = np.flatnonzero(lim)
    for k in MARKOUT_K:
        t = u[li] + k
        good = t < close_bar
        good[good] &= mid_ok[t[good]]
        j = li[good]
        out[f"mk{k}_sum2"] = int(np.sum(s[j] * qty[j] * (mid2[u[j] + k] - 2 * p[j])))
        out[f"mk{k}_sh"] = int(qty[j].sum())
        out[f"mk{k}_excl"] = int((~good).sum())
    for name, (lo, hi) in buckets(d0, df).items():
        jb = li[(u[li] >= lo) & (u[li] <= hi)]
        out[f"{name}_sh"] = int(qty[jb].sum())
        t = u[jb] + 5
        good = t < close_bar
        good[good] &= mid_ok[t[good]]
        jj = jb[good]
        out[f"{name}_mk5_sum2"] = int(np.sum(s[jj] * qty[jj] * (mid2[u[jj] + 5] - 2 * p[jj])))
        out[f"{name}_mk5_sh"] = int(qty[jj].sum())
    return out


def net(row, scenario):
    """Net P&L in 1e-4 dollars: cash - market fee x market shares - limit cost x limit shares (rebate when negative)."""
    mkt_fee, lim_cost = COST[scenario]
    return row["cash"] - mkt_fee * (row["sh_mkt_buy"] + row["sh_mkt_sell"]) - lim_cost * (row["sh_lim_buy"] + row["sh_lim_sell"])


def net_arrays(table, scenario):
    """Vectorized net over a DataFrame or dict of arrays with the ROW_KEYS columns."""
    mkt_fee, lim_cost = COST[scenario]
    return (np.asarray(table["cash"]) - mkt_fee * (np.asarray(table["sh_mkt_buy"]) + np.asarray(table["sh_mkt_sell"]))
            - lim_cost * (np.asarray(table["sh_lim_buy"]) + np.asarray(table["sh_lim_sell"])))


def breakeven_rebate(net_r_total, lim_shares_total):
    """Dollars per limit share needed to break even under R (SPEC 8 with erratum E0-5: <= 0 means none needed)."""
    if lim_shares_total <= 0:
        return float("nan")
    return float(-net_r_total / 1e4 / lim_shares_total)
