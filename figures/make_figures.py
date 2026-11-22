#!/usr/bin/env python3
"""Draw the README figures from the aggregated files in results/ only (no market data).
  python figures/make_figures.py"""
import csv
import datetime as dt
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
NAVY, BROWN, GREY = "#1F3B63", "#6E4630", "#9AA3AE"
D = r"\$"   # a literal dollar sign (matplotlib reads $...$ as math)


def usd(v, nd=2, unit="", sign=True):
    s = "+" if v > 0 and sign else ("\u2212" if v < 0 else "")
    return f"{s}{D}{abs(v):,.{nd}f}{unit}"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 11, "figure.dpi": 150, "savefig.bbox": "tight"})


def daily(name, cols):
    rows = list(csv.DictReader(open(os.path.join(RES, name))))
    dates = [dt.datetime.strptime(r["date"].replace("-", ""), "%Y%m%d") for r in rows]
    out = {}
    for c in cols:
        s, acc = 0.0, []
        for r in rows:
            s += float(r[c])
            acc.append(s)
        out[c] = acc
    return dates, out


def end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=color, fontsize=9)


def money_axis(ax, unit=1e6, suffix="M"):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: usd(v / unit, 0, suffix) if v else D + "0"))
    ax.axhline(0, color="#444444", lw=0.6)


def lead_lag_cumulative():
    d, c = daily("lead_lag_etf_twins_daily.csv", ["before_costs_usd", "after_costs_usd"])
    ll = json.load(open(os.path.join(RES, "lead_lag.json")))["etf_twins_and_share_classes"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(d, c["before_costs_usd"], color=NAVY, lw=1.8)
    ax.plot(d, c["after_costs_usd"], color=GREY, lw=1.4)
    end_label(ax, d[-1], c["before_costs_usd"][-1], f"before costs\n{usd(ll['H1_follows_before_costs']['usd_per_trade'])} a trade", NAVY)
    end_label(ax, d[-1], c["after_costs_usd"][-1], f"after costs\n{usd(ll['after_costs']['all'])} a trade", GREY)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: usd(v / 1e6, 1, "M") if v else D + "0"))
    ax.axhline(0, color="#444444", lw=0.6)
    ax.set_title("Lead-lag, ETF twins and share classes: the follower followed, costs were 5 times the gain")
    ax.set_ylabel(f"cumulative P&L, {D}10,000 a trade")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.text(0.01, 0.03, f"Sealed data 2021-01-04 to 2026-11-10, opened once. {ll['events']:,} trades.", transform=ax.transAxes,
            fontsize=8, color="#555555")
    fig.savefig(os.path.join(HERE, "lead_lag_cumulative.png"))
    plt.close(fig)


def stat_arb_cumulative():
    d, c = daily("stat_arb_daily.csv", ["before_costs_usd"])
    sa = json.load(open(os.path.join(RES, "stat_arb.json")))
    g, a = sa["gap_closes_before_costs"], sa["after_costs_market_orders"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(d, c["before_costs_usd"], color=NAVY, lw=1.8)
    end_label(ax, d[-1], c["before_costs_usd"][-1], f"{usd(g['usd_a_day'], 0)} a day\nbefore costs", NAVY)
    money_axis(ax)
    ax.set_title("Share classes of one company: the price gap closed on unseen data")
    ax.set_ylabel("cumulative P&L at mid prices")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    pt = a["per_trade_usd"]
    ax.text(0.01, 0.80, f"Per trade: gap {usd(pt['gap_at_mid'], 0)}, spread {usd(pt['spread'], 0)}, fees {usd(pt['fees'], 0)}"
            f" = {usd(pt['net'], 0)}\nAfter costs: {usd(a['usd_a_day'], 0)} a day", transform=ax.transAxes, fontsize=9, color=BROWN)
    ax.text(0.99, 0.03, f"Sealed data 2021-01-04 to 2026-11-09, opened once. 7 pairs, {D}50,000 a side.", transform=ax.transAxes,
            ha="right", fontsize=8, color="#555555")
    fig.savefig(os.path.join(HERE, "stat_arb_cumulative.png"))
    plt.close(fig)


def lead_lag_deciles():
    t = json.load(open(os.path.join(RES, "lead_lag.json")))["large_caps_and_sector_etfs"]["forecast_deciles"]
    x = [r["decile"] for r in t]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, [r["forecast_bp"] for r in t], color=NAVY, lw=1.4, ls="--", marker="o", ms=4, label="forecast move")
    ax.plot(x, [r["realised_bp_clipped"] for r in t], color=NAVY, lw=1.8, marker="o", ms=4, label="realised move (before costs)")
    ax.plot(x, [r["spread_bp"] for r in t], color=BROWN, lw=1.8, marker="s", ms=4, label="follower's bid-ask spread")
    ax.axhline(0, color="#444444", lw=0.6)
    ax.set_xticks(x)
    ax.set_xlabel("forecast decile (1 = weakest, 10 = strongest)")
    ax.set_ylabel("basis points")
    ax.set_title("Lead-lag, large caps: the stronger the forecast, the wider the spread")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.99, 0.03, "Sealed data 2021-2026, 4.4 million trades, 441,552 per decile.", transform=ax.transAxes, ha="right",
            fontsize=8, color="#555555")
    fig.savefig(os.path.join(HERE, "lead_lag_deciles.png"))
    plt.close(fig)


def lead_lag_costs():
    ll = json.load(open(os.path.join(RES, "lead_lag.json")))
    sets = [("ETF twins, share classes", ll["etf_twins_and_share_classes"]["cost_breakdown"], NAVY),
            ("Large caps, sector ETFs", ll["large_caps_and_sector_etfs"]["cost_breakdown"], BROWN)]
    keys = [("follow_before_costs", "Follow\n(before costs)"), ("spread_and_fees", "Spread\n+ fees"),
            ("size_beyond_displayed_depth", "Size beyond\ndisplayed quote"), ("after_costs_L3D1", "After\ncosts")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.38
    for i, (name, b, col) in enumerate(sets):
        xs = [k + (i - 0.5) * w for k in range(len(keys))]
        vals = [b[k] for k, _ in keys]
        ax.bar(xs, vals, width=w, color=col, label=name)
        for xx, v in zip(xs, vals):
            ax.annotate(usd(v), (xx, v), xytext=(0, 3 if v >= 0 else -11), textcoords="offset points", ha="center", fontsize=8)
    ax.axhline(0, color="#444444", lw=0.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([lab for _, lab in keys])
    ax.set_ylabel(f"{D} per {D}10,000 trade")
    ax.set_title("Lead-lag: where each trade's money went")
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.625, 0.42), fontsize=8)
    d = [b for _, b, _ in sets]
    ax.text(0.99, 0.97, "Entering 1 s late is the base case. After costs if 0 s / 5 s late:\n"
            f"ETF twins {usd(d[0]['after_costs_if_entered_0s_late'])} / {usd(d[0]['after_costs_if_entered_5s_late'])}, "
            f"large caps {usd(d[1]['after_costs_if_entered_0s_late'])} / {usd(d[1]['after_costs_if_entered_5s_late'])}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555555")
    fig.savefig(os.path.join(HERE, "lead_lag_costs.png"))
    plt.close(fig)


def market_maker():
    d, c = daily("market_maker_daily.csv", ["as_designed_usd", "best_tuned_usd", "upper_bound_usd"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, col, lab in (("upper_bound_usd", GREY, "upper bound:\nfilled on any touch"), ("best_tuned_usd", BROWN, "best tuned"),
                        ("as_designed_usd", NAVY, "as designed,\nreal queue")):
        ax.plot(d, c[k], color=col, lw=1.8)
        end_label(ax, d[-1], c[k][-1], f"{lab}\n{usd(c[k][-1] / 1e6, 2, 'M')}", col)
    money_axis(ax)
    ax.set_ylim(-6.2e6, 1.4e6)
    ax.set_title("Market maker: waiting in the real queue lost money on all 1,511 days")
    ax.set_ylabel("cumulative P&L after costs")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.text(0.01, 0.03, f"Build data 2015-2020, 10 stocks a year. Market orders pay the spread + {D}0.003 a share.",
            transform=ax.transAxes, fontsize=8, color="#555555")
    fig.savefig(os.path.join(HERE, "market_maker.png"))
    plt.close(fig)


if __name__ == "__main__":
    for f in (lead_lag_cumulative, stat_arb_cumulative, lead_lag_deciles, lead_lag_costs, market_maker):
        f()
        print("drew", f.__name__)
