# Study 26 · Market Maker rebuilt on 2015–2020 (code)

- **What it does:** re-runs the Market Maker build step of `../study12_market_maker/` on six years instead of two. The grid, the pass/fail filters and the sealed-run bar stay the same. The engine files there are imported, not edited.
- `data26.py` joins the two bar folders (2015–2018 from the Patient Trader study, 2019–2020 from the original Market Maker study), the two kinds of yearly stock list and the two calendars. `run26.py` runs the build step without the ETF steps, because there are no ETF bars before 2019. `test26.py` holds 16 synthetic tests.
- **Run order:** `python test26.py` (with `R26_TEST_DIR` set) → `python run26.py --regress` (2019–2020 must equal the original build results) → `python run26.py --go` (once).
- **Data:** bars under `$DATA_DIR/research15_data/` and `$DATA_DIR/research12_data/`. Results: `results/study26/`.
- **Note:** this copy is here to be read. Its checks refer to the private research repository, and the yearly stock lists and calendars are not included. Paths were generalised; the logic was not changed.
