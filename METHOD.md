# Method

## 1. Pre-registration flow

Each study followed the same steps, and each step left a git commit or tag in the research repository:

1. **Spec.** The question, data, settings grid, costs, pass/fail test, the sentence to write for each outcome, and a table of predictions with probabilities. Committed and tagged before any result.
2. **Build (exploration).** Settings are chosen on build years only, by a rule fixed in the spec.
3. **Validation (some studies).** One run on later, still unseen years. A gate decides whether the study goes on. For lead-lag, it went on only if the point estimate before costs was above 0.
4. **Freeze.** The code, model files and data manifests are hashed and recorded, then tagged `code-freeze-N`.
5. **Sealed run, once.** The loader refuses the sealed folder unless the freeze tag is an ancestor of HEAD and an unlock file holds a phrase I typed myself. The runner refuses a second run once its outputs exist. Results are committed and tagged `test-run-N`.
6. **Deviations.** Anything changed after the spec goes into a `DEVIATIONS` file with the reason. No rule was changed after its results were seen.

If a study failed on build or validation data, its sealed data were never opened.

## 2. Data and splits

- **Source.** WRDS TAQ millisecond quotes and trades, turned into **1-second bars**. Each bar holds the national best bid and offer (from the WRDS NBBO table), the displayed sizes, and the trades in that second.
- **Session.** 09:30–16:00. Strategies trade between 09:35 and 15:55, and every position is closed by 15:55.
- **Stocks.** Chosen each year from CRSP and the bars, by liquidity and spread filters fixed in the spec.

| Result | Build (tuning) | Validation | Sealed, opened once |
|---|---|---|---|
| Market maker | 2015–2020 | — | 2021–2026, never opened (failed on build data) |
| Share-class arbitrage | 2015–2020 | — | 2021-01-04 to 2026-11-09 |
| Lead-lag, ETF twins and share classes | 2015–2018 | 2019–2020 | 2021-01-04 to 2026-11-10 |
| Lead-lag, large caps and sector ETFs | 2015–2018 | 2019–2020 | 2021-01-04 to 2026-11-16 |
| Patient Trader | 2015–2020 | — | 2021-01-04 to 2026-10-23 |

Each sealed period ends on the last day of data its study had, so the end dates differ. The README's [Data periods](README.md#data-periods) table adds the replication sets and the shorter windows behind two Patient Trader numbers.

## 3. Fill simulator

**Market orders** take the opposite best quote at the decision bar, plus the delay: 1 second in the base case, and 0 and 5 seconds in checks.

**Limit orders** rest at their price, and the fill rule decides when they fill.

| Rule | What fills a resting limit order |
|---|---|
| **Q** (queue; the verdict model) | It waits behind the shares already displayed at its price. It fills only after that much volume has traded there. |
| **T** (upper bound, not achievable) | Any trade that touches its price fills it in full. |
| **L0** (as built) | Trades at its price fill it once the queue ahead is gone (Q). Trades through its price, or the opposite quote reaching it, also fill it. |
| **L2** | Only trades *through* its price fill it, never trades *at* it. |
| **L3** (strictest) | L2, plus a fill from the opposite quote reaching its price is capped at the displayed size. |

**Depth penalty for market orders:**
- **D0:** the whole order fills at the best quote.
- **D1:** every share beyond the displayed size costs 1¢ more.

**L3D1 is the strictest combination.** Each after-cost number names its rule. The arbitrage figures in the README use L0D0, the original rules; L3D1 is worse. The lead-lag figures use L3D1.

The engines are tested on synthetic markets with known answers. Each extended engine must also reproduce its predecessor's stored results, to 3 decimal places or better, before it is used: see `code/`.

## 4. Costs

- **Spread:** the real NBBO in each 1-second bar.
- **Market orders:** pay $0.003 a share (0.3¢) on top of the spread.
- **Limit orders:** pay no fee and get no rebate.
  - The class simulator's schedule (a 1¢ market fee, a 0.5¢ limit rebate) appears only as an aside, for the market maker.
- **"Before costs"** means:
  - **Arbitrage and lead-lag:** profit at the mid prices of the decision bars, so no spread or fee is paid.
  - **Patient Trader:** "before costs" is not used. The measure is implementation shortfall against the mid when the order arrived, in cents a share, compared with buying at market at once. The spread and fees are included.

## 5. Statistics

- **Daily series** (Patient Trader, arbitrage): a circular block bootstrap by day, with 21-day blocks and 10,000 draws. The test is one-sided. Where a study has two hypotheses, a Holm correction is applied.
- **Lead-lag events:** a pair-day cluster bootstrap, 10,000 resamples. Each draw resamples pair-days and averages the events inside them. The test is one-sided at α 0.05.
  - The second hypothesis asks whether the forecast's top trades beat the average. It is reported separately. Its clause joins the result sentence only if the first hypothesis holds.
- **Forecasts:** AUC on stocks or years the model never saw.
- **Seeds:** every study fixes its random seed in the spec.

## 6. How trials are counted

A **trial** is any configuration whose result I computed and could have picked: a grid cell, a candidate rule, a model, or a sealed verdict.

- **Studies 1–7** count one trial each. From study 8 on, every configuration counts.
- **Sealed verdicts:** from the Patient Trader replication onward, each sealed verdict adds one trial.
- **Not counted:** the screening maps of three closed lead-lag studies. No cell was positive after costs (for example 0 of 2,214 and 0 of 2,893), so no candidate came out and they added 0. The fixed baselines (buy at market at once, the upper bound T) are not counted either.

The running total sits in each study's code (`TRIALS_BEFORE`) and records. The chain adds up to **911**: see [LEDGER.md](LEDGER.md).
