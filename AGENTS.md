# AGENTS.md

This is the operating guide for agents and contributors working on the EQ
Backtest repository. Prefer current code over historical performance notes.

## Scope and source of truth

This repository backtests forecast-ranked A-share portfolios on daily, 15-minute,
and 5-minute bars. It is not a standalone data package: market and prediction
paths are local external paths, while benchmark returns come from a local market
cache selected by symbol.

Read sources in this order:

1. `doc/README.md` for the current user-facing contract.
2. `config.py` for actual defaults and configuration loading.
3. `data_loader.py`, `portfolio.py`, and `run.py` for execution semantics.
4. `tests/test_plan_behaviors.py` for covered behavior.

`doc/README_CN.md` is useful background but may describe configuration fields or
implementation details no longer present in the code. `doc/PERFORMANCE_OPTIMIZATION.md`
contains historical profiling and obsolete API references; do not use it as an
implementation specification.

## Quick operating loop

```bash
uv sync
uv run python -m unittest discover -s tests -v
uv run python run.py runs/<name>
```

Run commands from the repository root. `run.py` requires an existing dedicated
run directory containing its own `config.yml`; relative configuration paths,
pool cache, and results resolve inside it. Use one directory per experiment and
never commit `runs/`, `run_*/`, `backtest_runs/`, local data, cache files, or
generated charts.

There is no configured formatter, linter, or type checker. Preserve the existing
typed, module-oriented Python style, add focused tests for behavioral changes,
and always inspect `git diff` plus `git diff --check` before handoff.

## Architecture and ownership

| File | Responsibility |
| --- | --- |
| `config.py` | Defaults, deep merge of `config.yml`, path conversion, and derived runtime constants. |
| `data_loader.py` | Schema validation, market/prediction normalization, daily-rule broadcast, pool cache, and `BacktestDataset` encoding. |
| `market_cache.py` | Local per-symbol benchmark cache loading and explicit RQData refresh command. |
| `portfolio.py` | Numba portfolio state machine plus materialization of positions and target weights. |
| `run.py` | Entry point, daily aggregation, benchmark/excess evaluation, file writing, and plots. |
| `metrics.py` | Portfolio and holding-period metrics. |
| `plotting.py` | Matplotlib output only. |
| `research.py` | Signal diagnostics and parameter sweeps; do not treat results as production backtests without verifying this entry point. |

Keep hot-path logic in the compact NumPy/Numba representation. Use Polars for
large parquet loading, joins, and CSV/Parquet writes; use pandas at the reporting
and plotting boundary. Do not put pandas objects, Python dictionaries, or file
I/O inside `_simulate_portfolio_core`.

## Non-negotiable semantic invariants

1. `target` (ideal rank-band allocation) and `held` (actual executable
   allocation) are distinct. Do not collapse them: price-limit and T+1 effects
   make their divergence meaningful.
2. New purchases are frozen until the first bar of the next calendar day. Held
   count may temporarily exceed `port_size`.
3. Signals and execution may be separated by one bar through
   `trade_on_next_bar_for(frequency)`. Preserve timestamp alignment when
   changing either loading or simulator code.
4. Daily accounting follows previous close -> execution VWAP -> bar close.
   Do not derive a previous close across a missing symbol/trading-date gap.
5. For intraday data, eligibility and size rank come from the daily dataset and
   are broadcast by `(date, symbol)`; limit flags are recomputed from execution
   price and daily limit prices.
6. `size_rank` is recomputed inside the basic eligible universe. Ineligible
   names receive `999999` and must not consume candidate-pool ranks.
7. Costs are deducted in the share-based simulator. Evaluation must not deduct
   them a second time. Long excess is strategy minus benchmark; short excess is
   benchmark minus strategy.
8. `exclude_period` is an evaluation filter only. It must not change the data,
   positions, or trading timeline.

## Data, cache, and configuration discipline

- `config.py` deep-merges only dictionaries. Add a default for every new public
  configuration key and update `doc/README.md` in the same change.
- `RUN_DIR` is the directory holding the active `config.yml`. Relative market,
  prediction, and cache paths must remain relative to that directory;
  `OUTPUT_DIR` must remain inside it.
- Backtests must read benchmark returns only through `market_cache.py` using
  `market_cache_dir` and `benchmark_symbol`. Keep RQData imports inside the
  explicit refresh path; never make normal backtest execution fetch data.
- Prediction files are only read from the direct `preds_dir` directory; do not
  recursively mix neighboring experiment outputs. Each file must have unique
  `(datetime, symbol)` keys.
- For rolling daily production predictions, `prediction_sources` is the exact
  user-supplied list of files and optional inclusive date intervals. It overrides
  `horizons`; loaded intervals must remain non-overlapping. Never silently blend
  model vintages with `prediction_merge_mode: mean`.
- `concat_disjoint` is intentionally safe: overlapping prediction dates must
  fail unless the user explicitly selects `mean`.
- The pool cache stores an encoded frame, including `row_idx` and `symbol_id`.
  Bump `POOL_CACHE_VERSION` whenever its serialized schema or decoding semantics
  change.
- If a new setting changes pool contents, add it to `_pool_cache_path`'s payload.
  Cache identity includes the selected prediction-file signatures, their source
  date bounds, and `nosuspend_days`; preserve that coverage when changing
  eligibility, ranking, or prediction selection semantics.
- Treat external datasets as read-only. Do not add sample copies, rewrite
  source parquets, or delete a user's cache to force a result.

## Safe implementation and verification

For a behavior change, first identify whether it belongs to data normalization,
the ideal target layer, executable trades, share accounting, or evaluation. Test
the smallest relevant boundary with synthetic Polars/pandas data; do not require
the unavailable production datasets for unit tests.

Suggested checks:

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q .
git diff --check
git status --short
```

The first Numba invocation can compile and populate local cache files; report
whether a timing result is cold or warm. Avoid presenting historical benchmark
numbers as current performance.

## Known integration issue

`research.py` currently calls `build_pool(..., exclude_period=...)`, but
`build_pool` has no `exclude_period` parameter. Its CLI fails before diagnostics
or parameter sweeps run. Repair and test that call before relying on `research.py`
outputs; do not hide the failure by broadening `build_pool` with an unused
argument.
