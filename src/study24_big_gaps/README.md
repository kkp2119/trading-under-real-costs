# Study 24 · Biggest gaps in a wide pair universe (code)

- **What it does:** `universe24.py` ranks near-identical ETF pairs and share classes a year; `bars1s/` (with `collect24.sh`) collects their bars; `check24*.py` run quality and size-unit checks.
- `pair24.py` is the Study 20 pair engine with the dollar amount as an argument, `loader24.py`, `core24.py` and `stats24.py` the rest; `test24.py`, `regress24.py` are the tests; `dev24.py` the single build run. `run_test24.py` was never used: it refuses because no setting made money on build data.
- **Run order:** `universe24.py` → `bars1s/collect24.sh` → `check24*.py` → `test24.py`, `regress24.py` → `dev24.py run`.
- **Data:** WRDS TAQ and CRSP under `$DATA_DIR/`, and the Study 20 and 21 folders next to this one.
- **Note:** the integrity checks in this code (frozen-file hashes, git tags, the unlock phrase) refer to the private research repository, so in this copy they refuse to run by design. The code is here to be read; paths and credentials were generalised, the logic was not changed.
