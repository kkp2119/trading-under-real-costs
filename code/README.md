# Code: engines only

These are the parts that decide the results: the fill simulators, cost accounting, lead-lag event definition, and the bootstrap and verdict code, with their tests. The files are copied as they ran. The only edits replace local data paths with `$DATA_DIR`. The data download, universe selection and run scripts are left out, and so are all data.

## market_maker/: the queue-position fill simulator and market-maker accounting

| File | Role |
|---|---|
| `fillmodel.py` | Fill models for one resting limit order in one 1-second bar: **S**, **T** (filled on any touch) and **Q** (waits behind θ × the displayed queue). Integer arithmetic, numba if available. |
| `simulator.py` | The quoting loop: quote every 5 s, inventory limit, forced market exit, flat at the close. |
| `accounting.py` | Cash, spread earned, inventory P&L, markouts. Cost settings: **R** (market 0.3¢ a share, limit free), **R+** (0.2¢ limit rebate) and **C** (the class simulator's schedule: market 1¢, limit rebate 0.5¢). |
| `stats.py`, `judge.py` | Block bootstrap, Sharpe ratios with confidence bounds, and the pre-registered pass/fail filters. |
| `config.py`, `session.py`, `features.py`, `placebo.py`, `loader.py`, `runner.py`, `grid.py`, `results.py`, `bars1s/bars.py` | Settings, trading session, signals, placebo test, bar loading, per-day runner, grid and output tables. |
| `tests/` | Synthetic markets only. Includes an independent reference simulator (`refsim.py`), look-ahead, fill, inventory and accounting tests, known-market cases, and a reproduction check. |

Run the tests with `cd code/market_maker && python -m pytest -p no:cacheprovider tests`. On 2026-11-21: 76 passed and 14 skipped; the skipped tests need `R12_TEST_DIR`, a scratch folder outside the repository.

## execution/: the execution engine (Patient Trader, pairs, lead-lag)

| File | Role |
|---|---|
| `engine17.py` | The kernel shared by the Patient Trader and later studies. It holds the limit-order fill rules L0 and L3 (with the queue model), market orders with the depth penalty D1, the $0.003 fee, the stop-loss, candidate routing, and the one-sided bootstrap and Holm helpers. The pairs and lead-lag code import it unchanged. |
| `test_engine17.py` | Synthetic tests of every rule. `python test_engine17.py` gave 40/40 on 2026-11-21. |

## lead_lag/: event definition, forecast, and the sealed runners

| File | Role |
|---|---|
| `leadlag30.py` | **The event.** It defines the leader's move net of the follower's over s = 15 s against its rolling SD, the trigger at k = 3, follower-only market entry 1 s later, the exits (target, stop, 120 s, 15:55), and the money split (mid, spread, fee). |
| `model30b.py` | The forecast ĝ of each event's follow-through. Its inputs are the move size, leader volume, both spreads, imbalance, SPY's move and volatility. It is an OLS fit frozen after 2015–2016, with per-spread cost tables. |
| `run30g.py` | Validation and sealed runner for ETF twins and share classes. It checks the locks, applies the top-forecast threshold frozen from 2015–2016, runs the pair-day cluster bootstrap (10,000, seed 32), and applies the pre-written verdict sentences. |
| `val31.py` | Validation and sealed runner for large caps and sector ETFs. It checks the locks (freeze tag, unlock phrase, validation gate, collection completeness), runs the bootstrap (10,000, seed 33), and applies the verdict. |

The lead-lag files import the frozen pair engine and data loaders of the private research repository by its folder names (`research17` … `research31`). They are here so the logic can be read, and they will not run from this folder. The same code's synthetic and regression tests ran in the research repository before every run.
