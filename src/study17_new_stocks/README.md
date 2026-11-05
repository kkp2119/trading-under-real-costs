# Study 17 · Replication on new stocks (code)

- **What it does:**
  - `universe17.py` draws 20 new stocks a year for 2021–2026 (seed 17, no stock from any earlier study).
  - `bars1s/` downloads their 1-second bars. `check17*.py` run counts-only quality and CUSIP checks, and `power17.py` the power calculation.
  - `engine17.py` is the Study 16-D engine with a stop-loss and candidates C0–C5. `test_engine17.py` holds the synthetic tests, `dev17.py` the development run that picked C3, and `run_test17.py` the single sealed run.
- **Run order:** `universe17.py --n 20 --go` → `bars1s/download.py` → `check17.py`, `check17_cusip.py` → `power17.py` → `test_engine17.py` → `dev17.py run` → `run_test17.py --rehearsal` → `run_test17.py --go` (once).
- **Data:** WRDS TAQ and CRSP under `$DATA_DIR/`, plus the Study 12, 15 and 16 folders next to this one. The per-run files (`calendar17.csv`, `universe17_<Y>.tsv`, `SEALED_MANIFEST.tsv`, `EXCLUDE17.tsv`) live in the private research folder. The ticker lists are in `results/study17/new_universe_tickers.csv`.
