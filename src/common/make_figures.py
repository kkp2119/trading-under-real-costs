"""Draw the six figures in figures/ from the aggregated files in results/ only (no licensed data needed).
  python src/common/make_figures.py"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
NAVY, ORANGE, GREY = "#1F3B63", "#B4561F", "#8A8A8A"
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titleweight": "bold", "axes.titlesize": 11})


def dates(s):
    return pd.to_datetime(s.astype(str), format="%Y%m%d") if s.dtype != object or s.str.len().max() == 8 else pd.to_datetime(s)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


def shade(ax, start, end, label):
    ax.axvspan(start, end, color=ORANGE, alpha=0.10, lw=0, label=label)


def mm_cumulative_pnl():
    d = pd.read_csv(os.path.join(RES, "study12", "s12_daily_pnl.csv"))
    t = dates(d.date)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, d.B0_Q.cumsum() / 1e6, color=NAVY, lw=2, label="As designed (B0, queue fill model)")
    ax.plot(t, d.best_tuned.cumsum() / 1e6, color=NAVY, lw=1.5, ls="--", label="Best tuned setting (picked after the fact)")
    ax.plot(t, d.B0_T.cumsum() / 1e6, color=ORANGE, lw=1.5, ls=":", label="Upper bound (fill whenever a trade touches our price)")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_title("Market Maker, 2019–2020: cumulative net P&L after costs")
    ax.set_ylabel("Cumulative net P&L ($ millions)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(frameon=False, loc="lower left")
    save(fig, "mm_cumulative_pnl.png")


def mm_queue_position():
    d = pd.read_csv(os.path.join(RES, "study16", "queue16_sweep.csv"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for sc, ls, lab in (("R", "-", "Costs R: $0.003/share on market orders"),
                        ("R+", "--", "Costs R+: R plus a $0.002/share rebate on limit fills")):
        s = d[d.scenario == sc].sort_values("x")
        ax.plot(s.x, s.pnl_per_share_c, color=NAVY, ls=ls, marker="o", ms=4, lw=1.8, label=lab)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_title("Market Maker: P&L per share by place in the queue")
    ax.set_xlabel("Queue ahead of us = x × displayed size  (0 = first in line, 1 = back of the line)")
    ax.set_ylabel("Net P&L (¢ per share)")
    ax.set_ylim(min(d.pnl_per_share_c.min() * 1.1, -0.1), 0.2)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "mm_queue_position.png")


def ml_vs_no_ml():
    d = pd.read_csv(os.path.join(RES, "study15", "s15_ml_daily_pnl.csv"))
    t = dates(d.date)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    a = ax.plot(t, d.B0.cumsum() / 1e6, color=ORANGE, lw=1.8, label="Without ML (B0), left axis")
    ax.set_ylabel("Without ML: cumulative net P&L ($ millions)", color=ORANGE)
    ax.tick_params(axis="y", colors=ORANGE)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    b = ax2.plot(t, d.ML_best.cumsum() / 1e3, color=NAVY, lw=1.8, label="With ML (best setting, picked after the fact), right axis")
    ax2.set_ylabel("With ML: cumulative net P&L ($ thousands)", color=NAVY)
    ax2.tick_params(axis="y", colors=NAVY)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_title("Market Maker with and without ML, 2019–2020 (different axes)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(a + b, [x.get_label() for x in a + b], frameon=False, loc="lower left")
    save(fig, "ml_vs_no_ml.png")


def option_seller_cumulative():
    d = pd.read_csv(os.path.join(RES, "study13_14", "option_seller_O2_daily.csv"))
    d = d[d.date >= 19960306].reset_index(drop=True)          # statistics window (the first 43 days fill the ladder)
    t = dates(d.date)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = t[d.period == "sealed"]
    shade(ax, s.min(), s.max(), "Sealed period (opened once)")
    ax.plot(t, d.O2_opt.cumsum() / 1e3, color=NAVY, lw=1.3, ls="--", label="Before costs (mid prices, no fees)")
    ax.plot(t, d.O2_base.cumsum() / 1e3, color=NAVY, lw=2, label="After costs (bid/ask, $1/contract, 1 bp hedge cost)")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_title("Option Seller (setting O2, $1M notional), 1996–2025: cumulative P&L")
    ax.set_ylabel("Cumulative P&L ($ thousands)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "option_seller_cumulative.png")


def patient_trader_savings():
    d = pd.read_csv(os.path.join(RES, "study15", "s15_exec_daily.csv"))
    t = dates(d.date)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = t[d.period == "test"]
    shade(ax, s.min(), s.max(), "Sealed period 2021–Oct 2026 (opened once)")
    ax.plot(t, d.saved_Estar.cumsum() / 1e6, color=NAVY, lw=2, label="Patient Trader (E*) vs. market orders")
    ax.set_title("Patient Trader, 2015–2026: cumulative savings vs. market orders")
    ax.set_ylabel("Cumulative savings ($ millions, 1,000-share orders)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "patient_trader_savings.png")


def patient_trader_stress():
    r = pd.read_csv(os.path.join(RES, "study16", "robust16_summary.csv"))
    q = pd.read_csv(os.path.join(RES, "study16", "qty16_summary.csv"))
    e = pd.concat([r[(r.period == "2015-2020") & (r.method == "Estar")], q[q.variant.isin(["L3D0", "L3D1"]) & (q.method == "Estar")],
                   r[(r.period != "2015-2020") & (r.method == "Estar")]], ignore_index=True)
    order = [("L0D0", "Base engine"), ("L0D1", "+ depth penalty"), ("L2D0", "Only trades\nthrough our price"),
             ("L2D1", "Through-only\n+ depth"), ("L3D0", "Through-only, fills\ncapped at shown size"), ("L3D1", "Size cap\n+ depth"),
             ("sealed", "Sealed 2021–26\n(base engine)")]
    rows = [e[(e.period == "2015-2020") & (e.variant == v)].iloc[0] if v != "sealed" else e[e.period != "2015-2020"].iloc[0]
            for v, _ in order]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=False)
    for ax, (c, title) in zip(axes, (("S0", "E* − market order"), ("S1", "E* − TWAP"), ("S2i", "E* − passive-only (S2i)"))):
        for i, row in enumerate(rows):
            col = ORANGE if order[i][0] == "sealed" else NAVY
            m, lo, hi = row[f"minus_{c}_c"], row[f"minus_{c}_lo"], row[f"minus_{c}_hi"]
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", color=col, ms=5, capsize=3, lw=1.5)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([lab for _, lab in order], rotation=90, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("¢ per share (below 0 = E* cheaper)")
    fig.suptitle("Patient Trader stress tests: difference and 95% CI (2015–2020 unless marked sealed)", fontweight="bold")
    save(fig, "patient_trader_stress.png")


if __name__ == "__main__":
    for f in (mm_cumulative_pnl, mm_queue_position, ml_vs_no_ml, option_seller_cumulative, patient_trader_savings,
              patient_trader_stress):
        f()
        print("wrote", f.__name__)
