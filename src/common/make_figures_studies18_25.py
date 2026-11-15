"""Draw the three figures added with Studies 18-25 from the aggregated files in results/ only (no licensed data needed).
  figures/patient_trader_cost_lines.png   results/study15/s15_exec_daily.csv
  figures/stat_arb_by_year.png            results/study20/study20_daily.csv
  figures/stat_arb_latency.png            results/study20/study20_summary.json
  python src/common/make_figures_studies18_25.py"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
NAVY, BROWN, GREY = "#1F3B63", "#6E4630", "#7A838F"
SEALED_BG = "#EEF1F5"
CUT = pd.Timestamp("2025-11-03")                 # displayed sizes are 100x too big from this day (docs/data_unit_correction.md)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlesize": 11})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


def patient_trader_cost_lines():
    d = pd.read_csv(os.path.join(RES, "study15", "s15_exec_daily.csv"), dtype={"date": str})
    d["date"] = pd.to_datetime(d.date, format="%Y%m%d")
    d = d.sort_values("date")
    # dollars a day: cost in cents a share x 1,000 shares x orders / 100
    s0 = (d.IS_S0 * d.n_orders * 10).cumsum() / 1e6
    es = (d.IS_Estar * d.n_orders * 10).cumsum() / 1e6
    before, after = d.date < CUT, d.date >= CUT
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.axvspan(pd.Timestamp("2021-01-01"), d.date.max(), color=SEALED_BG, zorder=0)
    ax.text(pd.Timestamp("2021-02-01"), 0.97, "sealed data, opened once", transform=ax.get_xaxis_transform(), va="top", color=GREY, fontsize=9)
    ax.fill_between(d.date[before], es[before], s0[before], color=NAVY, alpha=0.12, lw=0)
    ax.plot(d.date[before], s0[before], color=GREY, lw=1.8, label="Market order at once")
    ax.plot(d.date[before], es[before], color=NAVY, lw=2.2, label="Patient Trader")
    k = int(before.sum()) - 1                                      # join the dashed part to the last solid point
    ax.plot(d.date.iloc[k:], s0.iloc[k:], color=GREY, lw=1.6, ls="--")
    ax.plot(d.date.iloc[k:], es.iloc[k:], color=NAVY, lw=1.8, ls="--")
    mid = d.date[before].iloc[int(before.sum() * 0.62)]
    i = d.index[d.date == mid][0]
    ax.annotate("What waiting saved", xy=(mid, (s0[i] + es[i]) / 2), xytext=(mid - pd.Timedelta(days=900), s0[i] * 0.98),
                color=NAVY, fontsize=10, arrowprops={"arrowstyle": "-", "color": NAVY, "lw": 0.8})
    ax.annotate("after the data unit change,\nnot counted", xy=(d.date.iloc[-1] - pd.Timedelta(days=120), es.iloc[-1] * 0.985),
                xytext=(pd.Timestamp("2024-02-01"), es.iloc[-1] * 0.62), color=GREY, fontsize=8.5, ha="left",
                arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.8})
    ax.set_title("Patient Trader vs. market orders: cumulative cost of 1,000-share orders, 2015–2026")
    ax.set_ylabel("Cumulative execution cost ($ million)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, s0.max() * 1.08)
    save(fig, "patient_trader_cost_lines.png")
    return {"orders_before": int(d.n_orders[before].sum()), "market_usd_m": float(s0[before].iloc[-1]), "patient_usd_m": float(es[before].iloc[-1])}


def stat_arb_by_year():
    """Before costs: yearly mean of the daily file. After costs: the yearly means stored in the summary (the sealed result
    files hold no daily series of the default signal's market-order run, so the daily file leaves it empty)."""
    d = pd.read_csv(os.path.join(RES, "study20", "study20_daily.csv"), dtype={"date": str})
    d["year"] = d.date.str[:4].astype(int)
    y = d.groupby("year")[["h1_mid_d0"]].mean()
    s = json.load(open(os.path.join(RES, "study20", "study20_summary.json"), encoding="utf-8"))
    after = {int(k): v for p in ("dev", "sealed") for k, v in s["periods"][p]["runs"]["W30_z2_k0"]["M_L0D0"]["by_year"].items()}
    y["default_M_L0D0"] = [after[k] for k in y.index]
    for p in ("dev", "sealed"):                                   # the daily file and the summary agree on the before-cost means
        for k, v in s["periods"][p]["runs"]["W30_z2_k0"]["mid"]["by_year"].items():
            assert abs(y.h1_mid_d0[int(k)] - v) < 1e-6, (k, v)
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.axvspan(2020.5, y.index.max() + 0.5, color=SEALED_BG, zorder=0)
    ax.text(2020.6, 0.97, "sealed", transform=ax.get_xaxis_transform(), va="top", color=GREY, fontsize=9)
    w = 0.38
    ax.bar(y.index - w / 2, y.h1_mid_d0, w, color=NAVY, label="Before costs (gap closes, at mid prices)")
    ax.bar(y.index + w / 2, y.default_M_L0D0, w, color=BROWN, label="After costs (market orders)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(y.index)
    ax.set_title("Share classes of one company: mean P&L a day, by year")
    ax.set_ylabel("\\$ a day (\\$50,000 a leg)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(frameon=False, loc="lower left")
    save(fig, "stat_arb_by_year.png")
    return y.round(1).to_dict()


def stat_arb_latency():
    s = json.load(open(os.path.join(RES, "study20", "study20_summary.json"), encoding="utf-8"))
    delays = ["0", "1", "5", "30"]
    vals = {p: [s["periods"][p]["delayed_mid"]["rows"][k]["ratio_to_delay0"] * 100 for k in delays] for p in ("dev", "sealed")}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = range(len(delays))
    w = 0.38
    for off, p, c, lab in ((-w / 2, "dev", GREY, "Build 2015–2020"), (w / 2, "sealed", NAVY, "Sealed 2021–2026")):
        bars = ax.bar([i + off for i in x], vals[p], w, color=c, label=lab)
        for b, v in zip(bars, vals[p]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8.5, color=c)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{k} s" for k in delays])
    ax.set_xlabel("Delay between the signal and both trades")
    ax.set_ylabel("Gain kept (% of no delay)")
    ax.set_ylim(0, 112)
    ax.set_title("Speed decides who can arbitrage: share classes of one company")
    ax.legend(frameon=False)
    save(fig, "stat_arb_latency.png")
    return vals


if __name__ == "__main__":
    print(patient_trader_cost_lines())
    print(stat_arb_by_year())
    print(stat_arb_latency())
