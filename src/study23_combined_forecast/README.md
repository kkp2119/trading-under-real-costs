# Study 23 · All signals in one forecast (code)

- **What it does:** `universe23.py` draws 20 new stocks a year; `bars1s/` downloads their bars; `check23*.py` run quality, identity, size-unit and traded-volume checks.
- `feat23.py` builds the nine model inputs, `model23.py` fits and scores the logistic model (fitted once; coefficients in `results/study23/model23.json`), `core23.py` runs the rules; `test23.py`, `regress23.py` are the tests; `dev23.py` the fit and validation run; `run_test23.py` the single sealed run.
- **Run order:** `universe23.py` → `bars1s/download.py` → `check23*.py` → `test23.py`, `regress23.py` → `dev23.py run` → `run_test23.py --go` (once).
- **Data:** WRDS TAQ and CRSP under `$DATA_DIR/`; SPY bars from Study 21; the Study 18 and 22 folders next to this one.
- **Note:** the integrity checks in this code (frozen-file hashes, git tags, the unlock phrase) refer to the private research repository, so in this copy they refuse to run by design. The code is here to be read; paths and credentials were generalised, the logic was not changed.
