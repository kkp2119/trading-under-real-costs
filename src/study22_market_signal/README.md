# Study 22 · The market's move as a signal (code)

- **What it does:** `universe22.py` draws 20 new stocks a year; `bars1s/` downloads their bars; `check22*.py` run quality, identity and size-unit checks.
- `signal22.py` computes the reference-price signal (SPY and the stock's own price); `loader22.py` reads the bars with the size correction; `core22.py` runs E\* and the candidate rules through the Study 18 engine; `test22.py`, `regress22.py` are the tests; `dev22.py` the build run; `run_test22.py` the single sealed run.
- **Run order:** `universe22.py` → `bars1s/download.py` → `check22*.py` → `test22.py`, `regress22.py` → `dev22.py` → `run_test22.py --go` (once).
- **Data:** WRDS TAQ and CRSP under `$DATA_DIR/`; SPY bars from Study 21; the Study 15–20 folders next to this one.
- **Note:** the integrity checks in this code (frozen-file hashes, git tags, the unlock phrase) refer to the private research repository, so in this copy they refuse to run by design. The code is here to be read; paths and credentials were generalised, the logic was not changed.
