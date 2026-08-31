# EQ Backtest

A forecast-driven A-share equity backtesting engine. It supports daily, 15-minute,
and 5-minute bars; long and short books; multiple portfolio sizes in one run; and
execution constraints such as price limits, ST status, listing age, and suspension
filters. The simulator maintains share-level positions and writes both tabular
results and diagnostic charts.

## How it fits together

```text
config.py defaults + local config.yml
                |
                v
      data_loader.build_pool()
  market bars + predictions + daily rules
                |
                v
    BacktestDataset (Polars + NumPy)
                |
                v
portfolio.generate_portfolio()  [Numba state machine]
                |
                v
 run.evaluate_portfolio_result() --> CSV / Parquet / charts
```

The data loader and portfolio simulator are frequency-independent. Frequency
changes affect input normalization and the aggregation of intraday returns to
daily returns, while the trading engine remains the same.

## Installation and first run

The project requires Python 3.11 or later and uses [uv](https://docs.astral.sh/uv/)
for dependency management.

```bash
uv sync
```

Each backtest must use a dedicated run directory. It contains that run's own
`config.yml`, pool cache, CSV/Parquet results, and charts; use a distinct
directory for every experiment. Run directories are ignored by Git.

```yaml
# runs/v6-baseline/config.yml
frequency: "daily"
start: "2024-01-01"
end: "2024-12-31"
market_cache_dir: "../../cache/market_data"
benchmark_symbol: "000852"
freq_config:
  daily:
    market_data: "/absolute/path/to/daily_market.parquet"
    preds_dir: "/absolute/path/to/predictions"
    horizons: ["5d"]
```

Create the directory, put the configuration above in `config.yml`, and launch it
from the repository root:

```bash
mkdir -p runs/v6-baseline
uv run python run.py runs/v6-baseline
```

The positional run directory must already exist and contain `config.yml`. Relative
paths in that configuration are resolved from the run directory, not the
repository root. The first run builds `data/.cache/` there; output is written to
`output_{long|short}_{pool_size}_{frequency}/` there, for example
`runs/v6-baseline/output_long_4400_daily/`.

To diagnose one instrument at one bar, set these local overrides and rerun:

```yaml
debug: true
debug_symbol: "300169.XSHE"
debug_datetime: "2019-01-03 09:46:00"
```

The simulator will print the signal, eligibility, cash, T+1, and execution
decision used for that instrument.

## Configuration

`config.py` owns the complete set of defaults. At import time it deeply merges
the optional `config.yml` over those defaults; omitted keys retain their default
values. Path-valued configuration is converted to `pathlib.Path`.

### Time, frequency, and inputs

| Key | Purpose |
| --- | --- |
| `start`, `end` | Inclusive backtest date range. |
| `frequency` | One of `daily`, `15min`, or `5min`. |
| `freq_config` | Per-frequency market path, prediction directory, prediction-file selectors, price columns, and optional execution lag. |
| `freq_config.<frequency>.prediction_sources` | Optional exact list of selected prediction parquets, each with optional inclusive date bounds. It overrides `horizons`. |
| `prediction_merge_mode` | `concat_disjoint` rejects overlapping prediction date ranges; `mean` averages overlaps by `(datetime, symbol)`. |
| `use_pool_cache`, `pool_cache_dir` | Enable and locate the encoded-pool cache. |
| `market_cache_dir` | Directory containing normalized, per-symbol benchmark caches. |
| `benchmark_symbol` | Cached benchmark symbol selected for this run. |

`freq_config.<frequency>.horizons` is the legacy discovery mechanism: it is
matched against prediction file stems. An empty selector (`[""]`) matches every
filename; use it only when the directory contains one intended prediction file.

For rolling daily production predictions, set `prediction_sources` instead. It
is a user-owned, ordered list; entries may be a direct filename or a mapping
with a filename plus inclusive `start` and `end` bounds. Relative filenames must
be direct children of `preds_dir`. The bounds are applied before concatenation,
and the resulting source ranges must not overlap when
`prediction_merge_mode: concat_disjoint` is used.

```yaml
frequency: "daily"
start: "2020-01-02"
end: "2026-08-28"
freq_config:
  daily:
    # From runs/<name>, this reaches ../daily/prod_cache/preds from the repository root.
    preds_dir: "../../../daily/prod_cache/preds"
    prediction_sources:
      - {file: "predictions_v6_2020.parquet", start: "2020-01-02", end: "2020-12-31"}
      - {file: "predictions_v6_2021.parquet", start: "2021-01-04", end: "2021-12-31"}
      # Continue with the exact model-vintage and date windows chosen for this run.
```

Do not select overlapping rolling-model files and switch to `mean`: that would
average scores from different model vintages. The source path signatures and
selected date bounds form part of the pool-cache key, so a changed selection
rebuilds the cache automatically.

### Benchmark

The backtest reads benchmark returns only from `market_cache_dir`; it does not
read a NAV parquet or a one-off benchmark CSV. Set `benchmark_symbol` to one
cached instrument, for example `000852`. Each cache file has `date`, `close`,
and `pct_change` columns; `pct_change` is the close-to-close daily return used
for excess-return evaluation. A `manifest.json` may map either an
`instrument_id` or an RQData symbol (such as `000852.XSHG`) to its cache file,
which permits many benchmarks in the same directory.

The default local cache is `cache/market_data/`.
From a run directory under `runs/<name>`, configure it as
`market_cache_dir: "../../cache/market_data"`. To create or update
cache files with RQData, run an explicit refresh; RQData is not imported by a
backtest. The refresh reads every `instrument_id`, `rq_symbol`, and
`first_date` from `cache/market_data/manifest.json`:

```bash
uv run python market_cache.py
```

The refresh command uses the manifest's RQData mapping when present. It needs a
locally installed and authenticated `rqdatac` package (and optionally
`RQDATAC_USERNAME` / `RQDATAC_PASSWORD`).

### Strategy and execution

| Key | Purpose |
| --- | --- |
| `universe` | Optional list of index identifiers allowed for new positions; `null` means all eligible names. |
| `allow_st_open`, `nosuspend_days` | Controls basic opening eligibility. |
| `pool_size` | Maximum eligible size rank for the candidate pool. |
| `port_sizes` | List of actual portfolio sizes to simulate. |
| `thresh_out_buffer` | A held target may remain until rank exceeds `port_size + thresh_out_buffer`. |
| `is_short` | Select low predictions and compute excess as benchmark minus strategy return. |
| `trade_on_next_bar` | Use the preceding bar's signal for execution. The daily configuration enables this by default. |
| `strict_first_bar_top_n` | Restrict initial execution to strict top-N candidates. |
| `close_on_size_drop` | Force a close when a holding leaves the size pool. |
| `weight_mode` | `equal`, `rank_linear`, or `rank_square`. |
| `max_weight_multiple` | Caps an individual non-equal weight relative to equal weight. |
| `cost_per_turnover` | One-way transaction cost; default is 4.5 bp. |
| `portfolio_initial_value` | Initial notional value for the share-based accounting. |
| `agg_mode` | Intraday-to-daily return aggregation: `simple` or `compound`. |
| `compounding` | Use geometric rather than arithmetic annualization for metrics. |
| `exclude_period` | Optional `[start, end]` interval excluded from final evaluation metrics only. |

## Input contracts

All internal inputs are normalized to a pool keyed by `(datetime, symbol)`.
Symbols may be supplied as `symbol` or legacy `stock_code`.

### Daily market data

The daily parquet requires:

```text
date, symbol, turnover, log_size, size_rank, industry, index, ret,
is_limit_up, is_limit_down, listed_Satisfied, is_ST, normal_days
```

Daily execution also requires the configured execution VWAP and bar-close
columns (normally `vwap30` and `close_ex`). `prev_close` may be supplied; when
it is not, it is derived from the previous global trading date and is not
bridged across a missing symbol date. The daily price path is:

```text
previous close -> execution VWAP -> bar close
```

### Intraday market data

The 15-minute and 5-minute parquets require:

```text
datetime, symbol (or stock_code), turnover, vwap_ret,
configured execution-VWAP column, configured bar-close column
```

Intraday data is joined to the daily parquet configured at
`freq_config.daily.market_data`. That supplies size rank, ST/listing/suspension
eligibility, and limit prices. A bar is buyable or sellable only when turnover
is positive and it is not at the corresponding one-sided price limit.

### Prediction data

Prediction files are parquet files directly inside the configured `preds_dir`.
They may be any of these forms:

1. Long: `datetime, symbol, pred`
2. Long legacy form: `datetime, symbol, prediction` or
   `trade_date, stock_code, prediction`
3. Daily production long form: `date, symbol, prediction, label`; `label` is
   excluded and `date` is normalized to `datetime`
4. A wide pandas-parquet matrix with `__index_level_0__` as its datetime index

Each input file must contain unique `(datetime, symbol)` rows. By default,
multiple prediction files must cover non-overlapping date ranges; choose
`prediction_merge_mode: mean` to combine overlapping forecasts deliberately.

## Trading and return semantics

The simulator separates an ideal `target` layer from the executable `held`
layer. A target is built from eligible signal ranks, but live execution still
obeys liquidity and price-limit constraints. Consequently, the held book can
temporarily differ from the target and can exceed `port_size` while a newly
bought position is T+1 frozen.

For each signal bar, the engine:

1. ranks finite forecasts (descending for long, ascending for short);
2. retains target names inside the exit threshold and fills remaining target
   slots from eligible candidates;
3. sells executable overweight holdings, respecting T+1 freezes;
4. buys executable target shortfalls, prioritizing existing holdings and then
   higher-ranked candidates; and
5. marks shares through the execution price and bar close, deducting trading
   costs from portfolio value.

For daily bars, a bar return is already a daily return. Intraday bar returns are
grouped by calendar day with either a sum (`simple`) or compounded return. Long
excess return is `portfolio - benchmark`; short excess return is
`benchmark - portfolio`. Costs are already deducted by the simulator and are
not deducted again during evaluation.

Metrics use 242 trading days per year. By default annual return is arithmetic
mean daily return times 242; `compounding: true` selects geometric annualization.
Maximum drawdown is calculated from `1 + cumulative_sum(daily_return)` and the
0.1% drawdown quantile.

## Outputs

For a run with pool size `P`, benchmark symbol `B`, and portfolio size `N`, the
output directory contains:

| Path | Contents |
| --- | --- |
| `metrics_{P}_{B}.csv` | One row per portfolio size: annual return, volatility, Sharpe, max drawdown, Calmar, and annual turnover. |
| `returns_{P}_{B}.csv` | Daily excess returns, one column per portfolio size. |
| `cumrets_{P}_{B}.csv` | Cumulative daily excess returns. |
| `portfolio_pnl_{N}.csv` | Daily strategy, benchmark, alpha, turnover, close-count, and cumulative PnL series. |
| `positions_{N}.csv` | Actual position snapshots and their market, eligibility, forecast, and weight fields. |
| `target_weights_{N}.csv` and `.parquet` | Ideal target-weight snapshots for intraday runs. |
| `plots_long/` or `plots_short/` | Portfolio overview, metrics table, holding-period, and position-size charts. |

## Development

Run the focused regression suite:

```bash
uv run python -m unittest discover -s tests -v
```

Tests use small synthetic frames and cover daily price-path derivation,
prediction merge validation, and signal-validation layers. A complete run needs
the local market, prediction, and benchmark data described above. See the root
[`AGENTS.md`](../AGENTS.md) for the repository's maintenance guide.
