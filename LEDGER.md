# Ledger: every trial, one row per study

Every configuration whose result was computed counts as a trial. Rows are in the order the trials were counted; **Total** is the running sum.

**Verdict** has three values:
- **Confirmed** or **Not confirmed**: the pre-registered test on data the study had not used for tuning.
- **Stopped**: the study ended on build or validation data, so its sealed data were never opened.

The four results in the README are in **bold**. **Record** is the study's identifier in the research repository's tags. Counting rules: [METHOD.md](METHOD.md#6-how-trials-are-counted).

| Record | Strategy | Question | Verdict | Trials | Total |
|---|---|---|---|---|---|
| 1 | Order-book imbalance on 1-second quotes | Does quote imbalance predict the next 1–10 s after costs? | Stopped | 1 | 1 |
| 2 | Intraday momentum, first half hour to last half hour (SPY) | Does it pay after costs on unseen years? | Not confirmed | 1 | 2 |
| 3 | Closing-auction gap, next-open reversal (index ETFs) | Do auction gaps revert overnight? | Not confirmed | 1 | 3 |
| 4 | Sector ETF momentum rotation | Do last month's winning sectors keep winning? | Not confirmed | 1 | 4 |
| 5 | Earnings-season premium, sector-hedged | Do stocks earn a premium around earnings? | Not confirmed | 1 | 5 |
| 6 | Volatility-targeted risk-parity ETF portfolio | Does it beat its benchmarks after costs? | Not confirmed | 1 | 6 |
| 7 | Scheduled execution against one market order | Do trading schedules cut implementation shortfall? | Not confirmed | 1 | 7 |
| 8 | Distance-method pairs trading | Does the classic pairs edge survive costs? | Not confirmed | 5 | 12 |
| 9 | Multi-factor long-short equity, monthly | A meaningful Sharpe ratio after costs? | Not confirmed | 9 | 21 |
| 10 | Time-series momentum in 41 futures | A meaningful Sharpe ratio after costs, 2000–2026? | Confirmed (Sharpe 0.54; weak after 2010) | 8 | 29 |
| 11 | Short-horizon cointegrated pairs, cost-aware | A meaningful Sharpe ratio after costs? | Not confirmed | 5 | 34 |
| 12 | Market maker on 1-second quotes | Is any setting profitable after costs on 2019–20 build data? | Stopped | 29 | 63 |
| 12-T1 | Market-maker screens: deeper quotes, toxic-flow states | Is there any source of edge in the build data? | Stopped | 103 | 166 |
| 13 | S&P 500 option seller, delta-hedged | Sharpe ≥ 0.5 on 1996–2012 build data? | Stopped | 11 | 177 |
| 14 | Option seller, three settings on sealed 2013–2025 | Sharpe ≥ 0.5 on sealed data? | Not confirmed | 3 | 180 |
| 15-S3 | Machine-learning market maker | Can a learned quoting policy make market making pay? | Stopped | 252 | 432 |
| 15-R1 | Market maker with simple queue-front rules | Do simple queue rules fix it? | Stopped | 4 | 436 |
| 15-E | Machine-learning timing for patient execution | Does it beat plain waiting? | Stopped | 216 | 652 |
| 15 | **Patient Trader** | Is it cheaper than a market order on sealed 2021–2026? | **Confirmed** | 36 | 688 |
| 16-G | Short-term reversal with cheap execution | Can a low execution cost rescue it? | Stopped | 4 | 692 |
| 16-Q | Market-maker P&L by queue position | Where in the queue does it break even? | Stopped (not even first in line) | 22 | 714 |
| 16-R | Patient Trader under stricter fill rules | Does the saving survive? | Confirmed (build-data check) | 6 | 720 |
| 16-D | Patient Trader with fill-size caps | Does the saving survive? | Confirmed (build-data check) | 2 | 722 |
| 16-E | Patient Trader savings by volatility and volume | Where is the saving largest? | Descriptive, no test | 0 | 722 |
| 17 | Patient Trader on 120 never-used stocks, plus adaptive patience | Does it replicate? Is the adaptive switch cheaper? | Both confirmed | 8 | 730 |
| 18 | Patient Trader with an order-book forecast | Does the forecast predict? Does it cut cost? | Forecast confirmed (AUC 0.591); cost gain stopped | 21 | 751 |
| 19 | G10 currency carry with a macro filter | Does carry pay? Does the filter help? | Not confirmed | 24 | 775 |
| 19B | Currency carry, 1979–2026 | Is the Sharpe ratio above 0 over 47 years? | Not confirmed | 2 | 777 |
| 20 | **Share-class arbitrage** | Do the gaps close? Does trading them pay? | **Gap confirmed**; profit stopped | 17 | 794 |
| 21 | Same-industry and near-identical ETF pairs | Do the gaps close? Does trading them pay? | Gaps confirmed; profit not confirmed or stopped | 19 | 813 |
| 22 | Patient Trader with a reference-price signal, third new stock set | Does it replicate? Does the signal help? | Replication confirmed; signal stopped | 13 | 826 |
| 23 | Patient Trader with a combined forecast, fourth new stock set | Does the forecast predict? Does it cut cost? | Forecast confirmed (AUC 0.601); cost gain stopped | 21 | 847 |
| 24 | Rare wide gaps across many pairs, orders 1 s late | Profit after costs? | Stopped | 4 | 851 |
| 25 | Patient Trader with last-minute volatility and volume | Cheaper than the plain Patient Trader? | Stopped | 14 | 865 |
| 26 | **Market maker, rebuilt on 2015–2020** | Is any setting profitable over six years of build data? | **Stopped** | 27 | 892 |
| 27 | Three cost-aware variants: arbitrage sized by profit minus cost; Patient Trader taking only the displayed size; regression hedge ratios | Does any beat its base strategy on validation data? | Two stopped; one sent to sealed (27B) | 9 | 901 |
| 28 | The class simulator's market maker on real quotes | Is there any condition where it pays? | Stopped | 1 | 902 |
| 27B | Patient Trader taking only the displayed size | Is it cheaper than the Patient Trader on sealed data? | Confirmed under the strictest fill rules only (0.11¢) | 1 | 903 |
| 29 | Auction-to-auction overnight long-short | Profit after costs? | Stopped | 4 | 907 |
| 30 | Lead-lag, trading the follower only | Does the follower's move exceed one-way costs? | Stopped | 0 | 907 |
| 30B | Lead-lag with limit entry and a forecast gate | Does the follower's move exceed one-way costs? | Stopped | 0 | 907 |
| 30C | Lead-lag on large caps | Does the follower's move exceed one-way costs? | Stopped | 0 | 907 |
| 30G | **Lead-lag, ETF twins and share classes** | Does the follower follow on unseen data, before costs? Do the forecast's top trades follow more? | **Both confirmed** (after costs −$4.00 a trade) | 2 | 909 |
| 31 | **Lead-lag, large caps after sector ETFs and SPY** | Does the follower follow on unseen data, before costs? Do the forecast's top trades follow more? | **Both confirmed** (after costs −$4.69 a trade) | 2 | 911 |

**Total: 911 trials.**

**Notes**
- **Studies 1–7.** These count one trial each, as every later record does, even where a study compared several rules. Study 6's own spec counted itself as 2, which would make the total 912.
- **Sealed runs.** Sealed and holdout runs added trials only from study 17 on, one per sealed verdict. Study 14's three holdout settings are the one earlier exception.
- **Lead-lag screens (30, 30B, 30C).** Their screening maps were not counted. No cell was positive after costs, so no candidate came out.
- **Study 27.** Its three variants were counted together, 892 to 901, on validation data. 27B's sealed verdict came after study 28.
