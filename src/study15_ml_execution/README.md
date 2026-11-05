# Study 15 · ML market maker and Patient Trader (code)

- **What it does:**
  - Builds the 2015–2018 1-second bars and the 31 past-only features.
  - Trains ridge and LightGBM quote filters on expanding folds and simulates the ML market maker.
  - Simulates the execution methods S0, S1, S2, S2i and E*.
- **Run order:**
  1. `universe15.py` and `calendar15.py`, then `bars1s/download.py`
  2. `xy15.py` (features), then `model15.py` (training)
  3. `run15.py` (ML market maker)
  4. `exec15.py`, then `r2_15.py` (execution methods; E* rule search)
  5. `run_test15.py` (the sealed 2021–2026 test, run once)
- **Data:** WRDS TAQ and CRSP under `$DATA_DIR/research15_data/`. It also reuses the Study 12 bars (`$DATA_DIR/research12_data/`) and Study 12 code (`../study12_market_maker`). `fillmodel15.py` is a copy of the Study 12 fill model (code unchanged).
