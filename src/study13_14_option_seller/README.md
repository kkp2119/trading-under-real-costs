# Studies 13–14 · Option Seller (code)

- **What it does:** builds the daily SPX option ladder (O0–O10), marks it to market and delta-hedges it, then applies the cost scenarios (base, stress, mid/no-fee) and the build-period filters. `study14/run_holdout14.py` is the single sealed run (2013–Aug 2025).
- **Run order:**
  1. `download.py` (OptionMetrics into build/sealed folders)
  2. `run_dev.py` (1996–2012)
  3. `select13.py` (filters; no setting passed)
  4. `study14/run_holdout14.py` (once)
  - `audit/` holds the pre-holdout audits.
  - `run_holdout.py` (Study 13's own sealed run) was never run.
- **Data:** WRDS OptionMetrics (SPX option prices, volatility surface, index closes) under `$DATA_DIR/research13_data/`. `tests/` uses synthetic chains.
