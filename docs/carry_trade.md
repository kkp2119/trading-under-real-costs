# Currency carry trade

**Question.** Each month, buy the 3 G10 currencies with the highest short-term interest rates and sell the 3 with the lowest. Does this classic carry trade earn money after costs in years it was not built on? Can a "danger" filter that steps aside in stressed markets help?

**Design (pre-registered).**
- **Data (public, FRED):** month-end exchange rates (H.10) and 3-month interbank rates (OECD) for the G10 currencies; VIX and the BAA–10-year credit spread for the filters. Cost: 4 bp of gross notional at each rebalance (8 bp before 1990).
- **Build 1990–2012, sealed 2013 – Feb 2026,** opened once.
- **22 candidates:** C0 (no filter) and 7 filters, each checked monthly (-M), weekly (-W) or daily (-D). A filter steps out of the trade when VIX is in its top 30/20/10% so far (V70, V80, V90), when VIX jumps more than 20%/40% (DV20, DV40), or when the credit spread widens more than 0.20/0.40 points (CS20, CS40).
- **Selection:** the filter with the best build Sharpe within the tie rules, A\* = CS40-M (credit spread, checked monthly).
- **Tests:** H1, Sharpe of C0 > 0 (level 0.025); H2, Sharpe(A\*) − Sharpe(C0) > 0 (level 0.05).
- **47-year extension:** C0 was also tested back to 1979 (H1b: 1979–1989 alone; H1c: all 47 years, level 0.0125).

**Results.**

| | Build 1990–2012 | Sealed 2013 – Feb 2026 |
|---|---|---|
| C0, carry without a filter: Sharpe | 0.35 | **0.04** [−0.40, 0.51], p = 0.42: not confirmed |
| A\* = CS40-M, credit-spread filter: Sharpe | 0.51 | **0.04**; A\* − C0 = −0.005: not confirmed |
| C0: costs as a share of the gross profit | 24% | **80%** |

- **Checking more often did worse.** Mean Sharpe of the 7 filters by check frequency: monthly 0.41 → −0.01, weekly 0.35 → −0.12, daily 0.24 → −0.16 (build → sealed). The faster filters traded more and lost more to costs.
- **Back to 1979:** 1979–1989 alone, Sharpe 0.35 (p = 0.09); all 47 years, Sharpe 0.29 (p = 0.028, above the 0.0125 level). Not confirmed.
- **Mixed with the Option Seller** (weights fixed on 1996–2012: 26% carry, 74% option seller): build Sharpe 0.79 (carry 0.73, options 0.47); sealed 2013 – Aug 2025, 0.04 (carry −0.03, options 0.09). Monthly correlation 0.41; in Aug 2015, Feb 2018 and Mar 2020 both lost at the same time, because both sell crash insurance.

**What it means.** The carry premium that looked solid in 1990–2012 almost vanished after 2012, and costs took most of what was left. Filters that looked good on history did not help, and combining two strategies that both sell insurance did not diversify it.

**Caveats.** Monthly data and a fixed cost in basis points per rebalance.

**Records.** Pre-registration files in [docs/preregistration](preregistration/) (study19; tags `spec19`, `code-freeze-19`, `test-run-19`, `code-freeze-19b`, `test-run-19b`).
