# Study 20 · Share-class pairs (code)

- **What it does:** `universe20.py` picks share-class pairs a year from CRSP; `calendar20.py` and `bars1s/` build the calendar and download 1-second bars; `check20*.py` run quality checks.
- `signal20.py` (z-score of the price gap), `pair20.py` (market and patient execution of both legs), `core20.py`, `stats20.py` and `loader20.py` are the engine; `test20.py` and `regress20.py` the synthetic and regression tests; `dev20.py` the build run, `delay20.py` the delay analysis, `run_test20.py` the single sealed run.
- **Run order:** `universe20.py` → `calendar20.py` → `bars1s/download.py` → `check20.py` → `test20.py`, `regress20.py` → `dev20.py`, `delay20.py` → `run_test20.py --go` (once).
- **Data:** WRDS TAQ and CRSP under `$DATA_DIR/`, and the Study 17 engine next to this one.
- **Note:** the integrity checks in this code (frozen-file hashes, git tags, the unlock phrase) refer to the private research repository, so in this copy they refuse to run by design. The code is here to be read; paths and credentials were generalised, the logic was not changed.
