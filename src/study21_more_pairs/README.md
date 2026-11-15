# Study 21 · Same-industry and ETF pairs (code)

- **What it does:** `universe21.py` picks same-industry stock pairs, near-identical ETF pairs and unrelated control pairs a year; `calendar21.py` and `bars1s/` (with `collect21.sh`) collect 1-second bars; `check21*.py` run quality and size-unit checks.
- `loader21.py` (sizes divided by 100 from 3 Nov 2025), `core21.py` and `stats21.py` reuse the Study 20 engine unchanged; `test21.py`, `regress21.py` are the tests; `dev21.py` the build run; `run_test21.py` the single sealed run.
- **Run order:** `universe21.py` → `calendar21.py` → `bars1s/collect21.sh` → `check21*.py` → `test21.py`, `regress21.py` → `dev21.py` → `run_test21.py --go` (once).
- **Data:** WRDS TAQ and CRSP under `$DATA_DIR/`, and the Study 20 folder next to this one.
- **Note:** the integrity checks in this code (frozen-file hashes, git tags, the unlock phrase) refer to the private research repository, so in this copy they refuse to run by design. The code is here to be read; paths and credentials were generalised, the logic was not changed.
