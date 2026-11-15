# Study 19 and 19B · Currency carry trade (code)

- **What it does:** `fetch19*.py` download the public FRED series (exchange rates, 3-month rates, VIX, credit spread); `check19*.py` check them; `engine19.py` builds the monthly carry portfolio and the 22 filter candidates; `dev19.py` is the build run (1990–2012); `run_test19.py` the sealed run (2013 – Feb 2026); `run_test19b.py` the 1979–1989 extension; `portfolio19.py` the mix with the Option Seller.
- **Tests:** `test_engine19.py`, `test_run19.py`, `test_run19b.py`, `test_portfolio19.py`.
- **Run order:** `fetch19.py`, `fetch19_h10.py`, `fetch19b_h10.py` → `check19.py`, `check19b.py` → tests → `dev19.py` → `run_test19.py --go` → `run_test19b.py --go` → `portfolio19.py`.
- **Data:** FRED only (no WRDS), saved under `$DATA_DIR/research19_data/`; the Option Seller's daily P&L from Studies 13–14.
- **Note:** the integrity checks in this code (frozen-file hashes, git tags, the unlock phrase) refer to the private research repository, so in this copy they refuse to run by design. The code is here to be read; paths and credentials were generalised, the logic was not changed.
