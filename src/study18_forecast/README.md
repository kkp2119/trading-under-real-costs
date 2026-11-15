# Study 18 · Order-book forecast for the Patient Trader (code)

- **What it does:** `universe18.py` draws 20 new stocks a year (2021–2026) and `bars1s/` downloads their 1-second bars; `check18*.py` run counts-only quality and identity checks, `power18.py` the power calculation.
- `engine18.py` holds the four-input logistic forecast and the rules R1 (cross early), R2 (keep waiting), R3 (both) on top of E\*; `test_engine18.py` the synthetic tests; `dev18.py` fits the model on 2015–2018 and picks a rule on 2019–2020; `run_test18.py` is the single sealed run.
- **Run order:** `universe18.py` → `bars1s/download.py` → `check18.py`, `check18_cusip.py` → `test_engine18.py` → `dev18.py` → `run_test18.py --rehearsal` → `run_test18.py --go` (once).
- **Data:** WRDS TAQ and CRSP under `$DATA_DIR/`, and the Study 15 and 17 folders next to this one.
- **Note:** the integrity checks in this code (frozen-file hashes, git tags, the unlock phrase) refer to the private research repository, so in this copy they refuse to run by design. The code is here to be read; paths and credentials were generalised, the logic was not changed.
