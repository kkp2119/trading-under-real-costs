# Making the Patient Trader smarter

**Question.** The Patient Trader (E\*) waits 1¢ inside the spread and crosses only when the spread narrows to 1–2¢. If we can forecast the next price move, should it cross early when the price is about to run away, or keep waiting when the price is about to come back?

**Design (pre-registered).**
- **Same orders as the [replication on new stocks](patient_trader_new_stocks.md):** 20 parent orders of 1,000 shares per stock and day, 5-minute deadline, one decision every 5 seconds. Costs: real spreads, $0.003 a share on market orders. Main fill rules: the strictest (L3D1).
- **Four kinds of information, one at a time.** Each was turned into rules on build data only. A rule had to beat plain E\* by **more than 0.02¢ a share** to earn a sealed test; otherwise only the forecast itself, and E\* against a market order, were tested on stocks never used before (20 a year, 2021–2026).

**Results.**

| Information added | Did the forecast hold on new stocks? | Best rule vs. E\* on build data | Acting early: best early-fill rule vs. E\* |
|---|---|---|---|
| **Order book:** size imbalance, spread, recent volatility, last 5-s move (logistic model, trained 2015–2018) | Yes: AUC **0.591** [0.589, 0.594] (validation 0.569) | −0.001¢ (keep waiting when p ≤ 0.40) | +0.009¢ (higher cost) |
| **The market's move:** a reference-price signal on SPY and on the stock's own price | Yes for SPY: after a "price is running away" signal the next 60 s moved a further 0.16 bp against the order [0.10, 0.21]; after a "price is coming back" signal, 0.10 bp in its favour | −0.006¢ (keep waiting on the SPY signal) | +0.115¢ |
| **All signals in one model:** the four order-book inputs, both reference signals, trade imbalance and trade intensity | Yes: AUC **0.601** [0.599, 0.603]; the order-book model on the same points: 0.594 | −0.006¢ (keep waiting when p ≤ 0.40) | +0.004¢ |
| **Last-minute volatility and volume** (the last minute's move vs. the spread and vs. pre-order volatility), as conditions on the all-signals forecast | Not tested (no rule passed the bar; the new stocks were not collected) | −0.004¢ (keep waiting when p ≤ 0.40 and the last minute moved at least half a spread) | +0.013¢ |

- Build data: the 2019–2020 validation years (99,080 orders), except the market's move, built on 2015–2020.
- **No rule reached the 0.02¢ bar**, so no rule was ever tested on sealed data. All eight early-fill rules with volatility or volume conditions cost more than E\*.
- **E\* itself held again on two more sets of new stocks:** −1.63¢ a share vs. a market order (95% CI −1.84 to −1.43) and −1.29¢ (−1.41 to −1.17), strictest fill rules, 2021 – Nov 2026.

**Volatility predicts the size of the next move, not its direction** (validation 2019–2020, every 5-s decision point; not used to pick any rule).

| Last-minute move ÷ spread (q) | Share of points | Mean absolute mid move in the next 60 s | Forecast AUC |
|---|---|---|---|
| below 0.25 | 30% | 2.5¢ | 0.579 |
| 0.25 – 0.5 | 16% | 3.1¢ | 0.577 |
| 0.5 – 1 | 27% | 3.4¢ | 0.576 |
| 1 – 2 | 19% | 4.1¢ | 0.569 |
| 2 or more | 7% | 6.9¢ | 0.565 |

**What it means.** Order-book and market information forecast the next move on stocks the model had never seen. But the forecast moves are smaller than the spread, so paying the spread to act early always cost more. Waiting longer on a forecast barely changed anything (at most 0.006¢), because E\* already crosses only when the spread is 1–2¢.

**Caveats.**
- Fills are simulated, and our own orders' market impact is not modeled.
- The 0.02¢ bar is an economic threshold set in advance, not a statistical test.
- The volatility rules were picked on the same 2019–2020 orders that the order-book and all-signals rules had used.

**Records.** Pre-registration files in [docs/preregistration](preregistration/) (study18, study22, study23, study25; tags `spec…`, `code-freeze…`, `test-run…`, `dev-run-25`).
