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
`output_{long|short|long_short}_{pool_size}_{frequency}/` there, for example
`runs/v6-baseline/output_long_4400_daily/`.

## Docker

For an Ubuntu host, build the image once and run a dedicated run directory:

```bash
./docker/build.sh
./docker/run.sh runs/v6-baseline
```

The runner mounts the repository source at `/app` read-only, while mounting
`runs/` and `cache/` read-write. Therefore Python and configuration changes are
used by the next `./docker/run.sh` invocation without an image rebuild. Rebuild
only after changing `Dockerfile`, `pyproject.toml`, or `uv.lock`.

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
| `use_pool_cache`, `pool_cache_dir` | Enable and locate the encoded-pool cache and normalized SBL cache. When false, both caches are bypassed and SBL workbooks are streamed without a cache write. |
| `market_cache_dir` | Directory containing normalized, per-symbol benchmark caches. |
| `benchmark_symbol` | Cached benchmark symbol selected for this run. |
| `benchmark_missing_return_policy` | `error` (default) rejects a missing benchmark return for any backtest date; `zero` is an explicit opt-in for dates intentionally treated as zero. |

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

`start` and `end` bound the requested period, but market bars define the
simulation timeline. In particular, a growing local market file may end before
`end`; the run intentionally stops at its last available market bar rather than
inventing missing bars or zero returns. Prediction and benchmark observations
beyond that bar are not evaluated. Read the dates in the generated output as
the effective backtest period.

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
`first_date` from `cache/manifest.json`:

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
| `short_port_size` | Optional short-sleeve name count; `null` uses the paired `port_sizes` entry. |
| `short_exit_rank` | Deprecated compatibility setting. Short sleeves always rebalance against the `short_port_size` rank frontier. |
| `strategy_modes` | Explicit ordered list of `long_only`, `short_only`, and/or `long_short`. |
| `short_borrow_sources` | Explicit SBL source mappings with `provider`, `adapter`, and run-directory-relative `path`; required by short modes. |
| `borrow_selection` | Available-borrow selection policy; currently `min_available_rate`. |
| `is_short` | Legacy single-mode compatibility switch when `strategy_modes` is omitted. |
| `trade_on_next_bar` | Use the preceding bar's signal for execution. The daily configuration enables this by default. |
| `strict_first_bar_top_n` | Restrict initial execution to strict top-N candidates. |
| `close_on_size_drop` | Force a close when a holding leaves the size pool. |
| `weight_mode` | `equal`, `rank_linear`, or `rank_square`. |
| `max_weight_multiple` | Caps an individual non-equal weight relative to equal weight. |
| `cost_per_turnover` | One-way transaction cost; default is 4.5 bp. |
| `portfolio_initial_value` | Initial notional value for the share-based accounting. |
| `agg_mode` | Intraday-to-daily return aggregation: `simple` or `compound`. |
| `compounding` | Use geometric rather than arithmetic annualization for the annual-return metric; cumulative reporting remains arithmetic. |
| `exclude_period` | Optional `[start, end]` interval excluded from final evaluation metrics only. |

### Optional optimizer service (daily long-only)

The optimizer path is off by default and is currently restricted to `daily`,
`long_only`, `weight_mode: equal`, and next-day execution. When enabled, EQ
Backtest still selects the rank-band target names, but obtains their final
weights from the independently running `equal_weight/1` service. The client does
not recalculate or normalize a successful response. Intraday and short modes
continue to use the local path.

The v0.2 capital policy preserves the existing denominator: if `M` valid target
names are selected for a configured size `N`, the request budget is `M/N`, each
service weight is `1/N`, and the unallocated amount stays in cash. A valid signal
that explicitly produces no target is a local liquidation event; absence of a
ranked signal is `no_signal` and does not call the service or reset the book.

Start the sibling optimizer checkout first, using an absolute socket path inside
the run directory:

```bash
cd /path/to/optimizer
uv run --python 3.11 python -m service \
  --socket /path/to/eq-backtest/runs/v6-optimizer/ipc/optimizer.sock
```

Then enable the fixed shm/1 contract in that run's `config.yml`:

```yaml
frequency: daily
strategy_modes: [long_only]
weight_mode: equal
freq_config:
  daily:
    trade_on_next_bar: true
optimizer:
  enabled: true
  transport: shared_memory
  protocol_version: "shm/1"
  control: unix_domain_socket
  socket_path: ipc/optimizer.sock
  backend: memfd
  zero_copy: required
  schema_version: "1.1"
  model: {type: equal_weight, version: "1", config: {}}
  timeout_ms: 5000
  request_timeout_ms: 6000
  max_inflight_per_session: 1
  max_control_bytes: 1048576
  max_shared_bytes: 268435456
  failure_policy: fail
  max_retries: 0
```

Relative `optimizer.socket_path` values resolve from the dedicated run directory.
Startup performs HELLO, HEALTH, and CAPABILITIES checks. Unreachable service,
timeout, version/model/policy mismatch, malformed handles, invalid weights, and
late-session identity all fail the experiment with zero retry and no local or
HTTP fallback.

Successful service runs additionally write `optimizer_calls.jsonl` and
`decision_targets.parquet` in the mode output directory. Calls include skipped
events, request/session/epoch/buffer identity, decision and planned execution
timestamps, budget, byte-copy counters, and timing. `target_weights_*.parquet`
remains the execution-time ideal target, while positions remain actual holdings;
the two are intentionally not collapsed.
The run directory also receives `optimizer_run_status.json`; service/config/data
failures leave it as `failed` with `complete: false`, and no successful summary
is fabricated from partially written outputs.

With the service running, a synthetic latency/copy audit can be repeated without
production data:

```bash
uv run python scripts/benchmark_optimizer.py --socket /absolute/run/ipc/optimizer.sock
```

The first request is treated as warm-up. The report separates control wait from
service model time and reports shared payload and transport-copy bytes.
The checked-in synthetic acceptance record is
[OPTIMIZER_VERIFICATION_V0.2.md](OPTIMIZER_VERIFICATION_V0.2.md).

### Securities borrowing (SBL)

`strategy_modes` explicitly selects any of `long_only`, `short_only`, and
`long_short`. When it is omitted, legacy `is_short` still selects one mode.
`short_only` is a long-benchmark / short-stock spread; `long_short` has one
unit of each sleeve (2.0 gross). Each selected mode writes its own
`output_{long|short|long_short}_{pool_size}_{frequency}/` directory.

A short mode requires an explicit source list. The source path is resolved from
the dedicated run directory and must be an `.xlsx` Yading workbook for now:

```yaml
strategy_modes: [long_only, short_only, long_short]
short_borrow_sources:
  - provider: yading
    adapter: yading
    path: ../../cache/SBL/Yading/20250424-20260709券单.xlsx
borrow_selection: min_available_rate
```

The Yading adaptor streams every worksheet and validates its six fields and
composite `市场` values (`SH.QFII`, `SH.HK`, `SZ.QFII`, `SZ.HK`). It parses the
exchange into the A-share suffix and stores the remaining value as the channel.
For duplicate date-symbol rows, a positive-quantity row with the lowest annual
rate is selected; ties use configured source order, channel, then source row.
Quantity establishes availability only and does not size the equal-weight book.

Fresh short sales require availability on both the signal and execution dates.
Before covering an existing short, the simulator qualifies replacement short
sales and limits full covers to retain the configured short-sleeve count.
Those full covers are taken from the worst-ranked current short positions first.
An opened short remains eligible to hold or cover after later list disappearance;
its selected borrow rate is locked (with share-weighted blending for later
increases). Fees are deducted in the short simulator once at each calendar-day
boundary using current pre-trade marked notional and Act/Act fractions, including
weekends and holidays. Positions include the selected provider/channel, current
availability, selected rate, locked rate, and sleeve for audit.

Evaluation returns are `long - benchmark` for `long_only`, `benchmark + signed
short P&L` for `short_only`, and `long P&L + signed short P&L` for `long_short`.
Combined turnover is the sum of the 1.0-NAV sleeve turnovers; it is not divided
by the 2.0 gross exposure.

The benchmark-missing policy is applied identically to performance metrics and
the PnL CSV. `research.py` supports exactly one sleeve (`long_only` or
`short_only`); a short sweep uses this same SBL configuration and rejects a
missing source list rather than silently producing an empty book.

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

With `trade_on_next_bar: true`, the signal is available at the previous close.
The simulator deliberately determines the share quantity from that previous
close, then executes that quantity at the next bar's execution VWAP and marks
it to the bar close. This matches an order workflow in which quantities are
computed after the prior close and before the following open. It is therefore
intentional that a target weight is a prior-close sizing instruction, rather
than an exact execution-VWAP notional weight.

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
The loader enforces timestamp alignment, not the provenance of a forecast or
its features. Prediction production remains responsible for ensuring each
signal was available by its recorded timestamp.

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

The current execution model is intentionally a binary-feasibility model: it
uses turnover and limit flags to allow or block trades, then applies the flat
configured transaction cost. It does not model participation limits, market
impact, order-queue priority, partial fills, or capacity. Treat it as a
selection backtest under this stated execution convention, not as a capacity
estimate, until a richer execution model is added.

Metrics use 242 trading days per year. Reporting is intentionally arithmetic:
by default annual return is mean daily return times 242, cumulative series are
simple sums of daily returns, and drawdown is calculated from
`1 + cumulative_sum(daily_return)` using the 0.1% drawdown quantile. These are
the project's reference metrics rather than compounded wealth, CAGR, or the
single worst peak-to-trough drawdown. `compounding: true` changes the
annual-return metric to geometric annualization only.

## Outputs

For a run with pool size `P`, benchmark symbol `B`, and portfolio size `N`, the
output directory contains:

| Path | Contents |
| --- | --- |
| `metrics_{P}_{B}.csv` | One row per portfolio size: annual return, volatility, Sharpe, max drawdown, Calmar, and annual turnover. |
| `returns_{P}_{B}.csv` | Daily excess returns, one column per portfolio size. |
| `cumrets_{P}_{B}.csv` | Arithmetic cumulative daily excess returns. |
| `portfolio_pnl_{N}.csv` | Daily strategy, benchmark, mode-aware evaluation return, turnover, borrow cost, close-count, and arithmetic cumulative PnL series. |
| `positions_{N}.csv` | Actual position snapshots with market, eligibility, forecast, weight, sleeve, and SBL audit fields when applicable. |
| `target_weights_{N}.csv` and `.parquet` | Ideal target-weight snapshots for intraday runs. |
| `plots_long/`, `plots_short/`, or `plots_long_short/` | Portfolio overview, metrics table, holding-period, and position-size charts. |

## Development

Run the focused regression suite:

```bash
uv run python -m unittest discover -s tests -v
```

Tests use small synthetic frames and cover daily price-path derivation,
prediction merge validation, and signal-validation layers. A complete run needs
the local market, prediction, and benchmark data described above. See the root
[`AGENTS.md`](../AGENTS.md) for the repository's maintenance guide.
