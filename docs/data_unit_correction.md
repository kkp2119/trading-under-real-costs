# Data error: displayed quote sizes are 100× too big from 3 Nov 2025

**What went wrong.** On 3 Nov 2025 the US consolidated quote feeds changed how displayed sizes are reported. In our 1-second bars (built from WRDS TAQ), every displayed bid and ask size from that day on is exactly **100 times** the true number of shares. Sizes up to 31 Oct 2025 are correct. Our code never multiplied by 100; the unit changed in the data.

**How we found it.** In the sealed results of the share-class pairs, the depth penalty of the market-order run with the selected signal was **exactly zero on all 256 trading days from 3 Nov 2025**. Before that date it was zero only on days without a trade. We then read the two size columns (and nothing else) of those bars:
- up to 31 Oct 2025, every stock-day moved in steps of 100 shares (21,230 stock-days);
- from 3 Nov 2025, every stock-day moved in steps of 10,000 or 4,000 shares (2,040 stock-days): 100 × the round lot of 100 or 40 shares.

**What it affects.** Only the fill rules that use displayed size:
- the depth penalty (D1, +1¢ beyond the displayed size) and the size cap of the strictest fill rule (L3) become **looser**;
- the queue in front of a resting limit order (L0) becomes **stricter**.

Mid prices, spreads, traded volumes, and the order-book imbalance ratio used by the forecasts are not affected.

**What we did.**
- No sealed test was rerun and no verdict was changed.
- For the tests already run (below), we split the stored daily or yearly results at the date. The numbers below are post-hoc.
- Every later test divides displayed sizes by 100 from 3 Nov 2025 when it loads the bars, and checked the unit before its sealed run. Traded volumes were checked too: they did not change.

| Strategy | Result | Whole sealed period | Before the change |
|---|---|---|---|
| Patient Trader, original stocks | E\* − market order, ¢ a share | −1.50 [−1.67, −1.33] | **−1.30 [−1.46, −1.14]** (to 31 Oct 2025) |
| | Saving as a share of the market-order cost | 31% | **29%** |
| Patient Trader, 120 new stocks | E\* − market order, strictest fill rules | −1.25 [−1.37, −1.14] | **−1.19** (2021–2024; only yearly values are stored) |
| | Spread switch C3 − E\* | −0.10 | **−0.13** (2021–2024) |
| Order-book forecast | AUC on new stocks | 0.591 | 0.596 (2021–2024, mean of yearly AUCs; sizes enter only as a ratio) |
| Share-class pairs | Gap closes, $ a day | +4,174 | **+4,252** (to 31 Oct 2025) |
| | Selected variant after costs (strictest fill rules), $ a day | −777 | −871 (to 31 Oct 2025) |

**Conclusions.** Every verdict holds on the data before the change. The README quotes the Patient Trader before the change (1.30¢ instead of 1.50¢).
