# Common code

- **`make_figures.py`** draws every figure in `figures/` from `results/` only: `python src/common/make_figures.py`.
- **Shared engine pieces stay inside each study folder.** Each study froze its code with SHA-256 manifests (see `docs/preregistration/`), so the pieces are not moved here:
  - **Queue fill model (S/T/Q):** `study12_market_maker/fillmodel.py`. `study15_ml_execution/fillmodel15.py` is a copy with the code unchanged.
  - **Cost models:** `study12_market_maker/accounting.py` (R, R+, C), `study13_14_option_seller/costs.py` (option bid/ask, fees, hedge) and `study15_ml_execution/exec15.py` → `costs` (implementation shortfall).
  - **Bootstrap and Sharpe:** `study12_market_maker/stats.py`, `study13_14_option_seller/stats13.py`, `study15_ml_execution/run_test15.py` → `boot_ci` and `study16_followups/robust16.py` → `boot_index`, `diff_ci`.
