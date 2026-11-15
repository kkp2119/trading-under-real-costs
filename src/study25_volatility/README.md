# Study 25 · Last-minute volatility and volume (code)

- **What it does:** `universe25.py` drew 20 new stocks a year and `check25_cusip.py` checked their identity (the bars were never collected).
- `signal25.py` computes the last-minute move against the spread and against pre-order volatility, and what the engine gets for each of the 14 rules; `core25.py` runs them with Study 23's frozen forecast; `test25.py`, `regress25.py` are the tests; `dev25.py` the single validation run. `run_test25.py` was never used: no rule beat E\* by the 0.02¢ bar.
- **Run order:** `universe25.py` → `check25_cusip.py` → `test25.py`, `regress25.py` → `dev25.py run`.
- **Data:** the Study 15, 18, 22 and 23 folders next to this one, with their bars under `$DATA_DIR/`.
- **Note:** the integrity checks in this code (frozen-file hashes, git tags, the unlock phrase) refer to the private research repository, so in this copy they refuse to run by design. The code is here to be read; paths and credentials were generalised, the logic was not changed.
