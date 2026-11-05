# Study 16 · Follow-ups (code)

- **What it does:**
  - `gate16.py`: the daily-reversal gate (CRSP).
  - `queue16.py`: the market maker with the queue ahead set to x × displayed size (Study 12 engine).
  - `robust16.py` and `qty16.py`: E* under stricter fill and depth rules (Study 15 engine).
- **Run order:** each script is independent and runs as `python <script> run`. Run `gate16.py fetch` first for the CRSP rows. `robust16.py` and `qty16.py` re-check their output against the stored Study 15 results.
- **Data:** CRSP daily (16-A) and the 2015–2020 1-second bars of Studies 12 and 15 under `$DATA_DIR/`. Per-order tables are written to `$DATA_DIR/research16_data/`; only aggregates go to `results/`.
