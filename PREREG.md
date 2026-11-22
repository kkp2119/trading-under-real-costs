# Pre-registration: the two lead-lag tests

Both tests ask one question first: **after a leader moves, does the follower follow, on data never seen, before costs?** Neither test claims a profit.

For each test, the plan was committed and tagged before its data were read. The code was frozen before the sealed data were opened. The sealed run happened once, and the runner refuses a second run. Tags and hashes refer to the private research repository, where the records are in Korean.

## Shared design

- **Event.** The trigger is the leader's 15-second log move net of the follower's, at 3 standard deviations of its recent history (k = 3). Events fire from 09:35 to 15:50.
- **Trade.** The strategy buys or sells the follower at market 1 second later, $10,000 a trade.
  - It exits at a target, a stop, 120 seconds, or 15:55, whichever comes first.
- **Before-cost value.** P&L at the mid prices of the decision bars. Next to every verdict, the after-cost value of the same trades is reported: strictest fills (L3D1) plus $0.003 a share.
- **Forecast ĝ.** An OLS model of each event's follow-through, fitted once on 2015–2016 events and then frozen.
  - "Top" trades are those with ĝ at or above its 2015–2016 90th percentile.
- **Test.** A pair-day cluster bootstrap, 10,000 resamples, one-sided at α 0.05.
  - The H2 clause is added to the result sentence only if H1 is confirmed.

## Test 1: ETF twins and share classes (study 30G)

| | |
|---|---|
| Pairs | ETF twins and share classes of one company, from an earlier arbitrage study. The leader is the leg with the larger dollar volume the year before. |
| H1-G | Mean before-cost value per trade > 0 |
| H2-G | Mean of the forecast's top trades − mean of all trades > 0 (top: ĝ ≥ 2.40 bp) |
| Gate | If the H1 point estimate in validation was ≤ 0, stop and never open the sealed data. H2 is judged on sealed data only if its validation estimate is > 0. |
| Exploration, 2015–2018 | Done before this plan, in two closed studies: tags `explore-run-30` (9abbedf) and `explore-run-30b` (23f8488), both 2026-11-17. The plan declares them as already seen. |
| Validation, 2019–2020 | Run once on 2026-11-18 at 00:17, code at `pre-val-30g` (02b0694). H1 +1.234 [+1.180, +1.293], H2 +4.882 [+4.392, +5.404] → sealed, both hypotheses |
| Sealed, 2021-01-04 to 2026-11-10 | Run once on 2026-11-18 at 00:36. **H1 +0.978 [+0.954, +1.006], confirmed. H2 +4.633 [+4.269, +5.006], confirmed.** After costs: −3.999 a trade |
| Tags | `spec30g` 5cc090e (2026-11-18 00:13) · `code-freeze-30g` 7d02472 (00:18) · `test-run-30g` c66a8bd (00:37) |
| Plan sha256 | SPEC30G.md `ef7902d6a1e1a1f6c2e21a3725bb4edbccce5b13bc9887501a55681fd1efc730` |
| | PREDICTIONS30G.md `1bc843e9c89c6e9219729ace4a683d258ad7d96c1e20fd78a172e96d6d6919ce` |
| Trials | 907 → 909 (+2) |
| Predictions | Validation H1 > 0: 0.85, it happened. Sealed H1 confirmed: 0.80, it happened. Sealed H2 confirmed: 0.60, it happened. After-cost value > 0: 0.03, it did not happen. |

## Test 2: large caps after their sector ETF or SPY (study 31)

| | |
|---|---|
| Pairs | Three families: sector SPDR → large cap (40 pairs), SPY → large cap (40), SPY → sector SPDR (9). The 40 large caps were chosen from the S&P 500 of end-2014, and none had been used in an earlier study. |
| H1-31 | Mean before-cost value per trade > 0 (base setting, three families pooled) |
| H2-31 | Mean of the forecast's top trades − mean of all trades > 0 (top: ĝ ≥ 1.17 bp) |
| H3-31 | A cost-gated rule (enter only when the forecast beats the expected cost) is profitable after costs. To be judged only if exploration produced a candidate; none did, so H3 was not judged. |
| Gates | Exploration: the base setting's 95% CI lower bound before costs, 2015–2018, must be > 0. Validation: the H1 point estimate must be > 0. Otherwise the study stops and the sealed data are never opened. |
| Exploration, 2015–2018 | Run once on 2026-11-18, 19:16–20:16, code at `pre-explore-31` (0aa9477). +0.406 [+0.396, +0.416] → validation. No H3 candidate. |
| Validation, 2019–2020 | Run once on 2026-11-19, 06:09–06:24, code at `pre-val-31` (565312a). H1 +0.628 [+0.604, +0.652], H2 +1.206 [+1.122, +1.289] → sealed |
| Sealed, 2021-01-04 to 2026-11-16 | Run once on 2026-11-21, 19:35–20:17. **H1 +0.378 [+0.356, +0.402], confirmed. H2 +1.002 [+0.867, +1.156], confirmed.** After costs: −4.688 a trade |
| Tags | `spec31` e79c391 (2026-11-18 01:51) · `universe31` 2e9c178 · `explore-run-31` d5f7c79 · `val-run-31` ccd64e4 · `code-freeze-31` 2b4f9c4 (2026-11-21 03:31) · `test-run-31` 0c8ca4e (2026-11-21 20:18) |
| Plan sha256 | SPEC31.md `071ff01e983c8b87caa583022309af2cb71f62c5b0bc9591b0819b45d16f4551` |
| | PREDICTIONS31.md `c5981c891b6b5814aedee04a55a9ba51a0c08424dc3792610d2aed2e667bb8b5` |
| Trials | 909 → 911 (+2) |
| Predictions | Sealed H1 confirmed: 0.75, it happened. Sealed H2 confirmed: 0.55, it happened. H3 confirmed: 0.35, not judged. Exploration items: CI lower bound > 0: 0.75, it happened. Large-cap cost at most half of test 1's: 0.80, it did not happen. Smaller follow-through than test 1 in bp: 0.70, it happened. An H3 candidate: 0.12, it did not happen. |

Values are $ per $10,000 trade, with 95% CIs in brackets. The two plans' sha256 were checked against the files at their spec tags. The quote sizes from 3 Nov 2025 were read corrected (÷100) in both tests.
