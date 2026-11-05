# Study 12 · Market Maker (code)

- **What it does:** turns TAQ quotes and trades into 1-second NBBO bars, then simulates the B0 market maker and its one-change-at-a-time grid under the S/T/Q fill models and the R/R+/C cost scenarios. It also applies the pre-registered filters.
- **Run order:**
  1. `bars1s/download.py` (bars)
  2. `universe_screen_2018.py` and `universe_rolling.py` (stock draws)
  3. `run_dev.py` (build period 2019–2020)
  4. `select_candidate.py` / `judge.py` (verdict)
  - `run_holdout.py` was never run, because no setting qualified.
- **Data:** WRDS TAQ (consolidated quotes and trades) and CRSP for the universe, under `$DATA_DIR/research12_data/`. `tests/` uses synthetic markets only (`python -m pytest tests`).
