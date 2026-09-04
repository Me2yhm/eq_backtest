# Simple 5-Minute `open_1d` Backtest Refinement Plan

## Status

Planned. A dedicated configuration has been added at
`runs/5min_open_1d_long-only/config.yml`; this plan intentionally keeps the
first intraday run close to the established daily long-only pattern.

## Objective

Run a long-only, equal-weight 5-minute backtest against benchmark `000852`
using one user-selected `open_1d` model checkpoint. The implementation must
use the cached 5-minute bars and the cached daily rules without adding a second
portfolio engine or changing daily semantics.

## Deliberate scope

This first pass does not add model validation, prediction-coverage checks,
capacity modelling, alternative cost models, checkpoint optimisation, or a
streaming/multi-process simulator. Those concerns remain separate from a
simple intraday backtest and should be introduced only when their requirements
are agreed.

## Input and execution contract

The run configuration uses:

- `cache/ddb_data/5min_bar_full_left_close.parquet` for `datetime`, `symbol`,
  `turnover`, `vwap_ret`, `vwap5`, and `close`.
- `cache/ddb_data/daily_with_limit_prevcap.pqt` for daily `log_size`, listing,
  ST, suspension, and limit-price rules broadcast by `(date, symbol)`. The
  simulator recomputes size rank over the basic eligible scope; it does not use
  the file's stored `size_rank` directly.
- Exactly three files for one checkpoint from
  `5m/res_nd/output/open_1d/predictions/`, one per configured calendar range.

The selected checkpoint is an explicit user-owned choice. The initial config
selects epoch 15. Selecting epoch 11 through 14 means changing all three
`prediction_sources` filenames to the same epoch. Do not mix checkpoint
epochs, discover every parquet in the directory, or use `mean` merging.

The 5-minute execution convention is explicit:

```text
signal on completed bar t -> size shares at t close
-> execute those shares at bar t+1 VWAP5 -> mark at bar t+1 close
```

This is the intraday equivalent of the daily run's separated signal and
execution timing. With next-bar execution, the share count uses the signal
bar's close rather than the execution bar's VWAP5; target weights are therefore
prior-close sizing instructions, not exact VWAP5 notional weights. VWAP5 is the
intermediate marking point for the previous-close-to-VWAP5 and VWAP5-to-close
P&L legs.

The immediately preceding bar supplies the signal even across a session
boundary: the first 5-minute bar of a trading day uses the final bar of the
previous trading day. T+1 remains calendar-day based: an intraday purchase
cannot be sold until the first bar of the following trading day.

## Necessary adjustments

### 1. Use an isolated, explicit run configuration

Use `runs/5min_open_1d_long-only/config.yml` rather than the generic 5-minute
defaults. It pins the cached market paths, daily rules, benchmark cache,
long-only mode, VWAP5/close columns, simple daily aggregation, and next-bar
execution. It also sets exact prediction source files, which avoids the
overlapping checkpoint files in the prediction directory.

No new public configuration key is necessary: the existing
`prediction_sources` contract already makes checkpoint selection explicit and
includes the selected source signatures in the pool-cache identity.

### 2. Reuse the existing intraday path unchanged

No portfolio-core change is planned for the simple run. The existing pipeline
already:

1. normalizes the 5-minute market frame;
2. broadcasts daily eligibility and recomputes size rank from `log_size` over
   the basic eligible scope;
3. recomputes intraday limit eligibility from the execution VWAP5 and daily
   limit prices, using the existing `0.0005` price tolerance;
4. joins predictions by exact `(datetime, symbol)`;
5. applies target/held separation, price limits, and calendar-day T+1 in the
   common share-based simulator; and
6. aggregates bar returns and additive turnover to daily reporting series.

Keeping this common path avoids diverging from the daily backtest's portfolio,
cost, benchmark, and arithmetic-reporting conventions.

### 3. Add only two focused regression tests

Keep the loader and configuration contracts independently testable:

1. Add a loader test with three disjoint files for one selected checkpoint and
   decoy files from another. It must prove that the explicit
   `prediction_sources` load only the selected checkpoint and preserve source
   ordering and date bounds.
2. Add a run-configuration resolution test following the existing dedicated
   run-directory pattern. It must assert that the 5-minute configuration
   resolves `trade_on_next_bar_for("5min")` to `True`.

Do not add a model-quality or missing-data test in this phase.

## Execution sequence

1. Confirm the desired epoch in the run config; epoch 15 is only the initial
   selection, not an automatic recommendation.
2. Run the focused regression suite.
3. Launch the isolated run from the repository root:

   ```bash
   uv run python run.py runs/5min_open_1d_long-only
   ```

4. Inspect the run-local pool-cache and `output_long_4400_5min/` artifacts.
   The positions and target-weight files provide the signal/execution audit
   trail; the daily PnL, returns, and metrics retain the project's arithmetic
   reporting convention.

## Resource note

The configured multi-year 5-minute input is large. In particular, the current
simulator allocates position-record buffers at `n_bars * n_symbols`, not at the
nominal portfolio size. At roughly 27,000 bars and 5,300 symbols, that is about
143 million record slots. The nine position-record arrays alone consume roughly
57 bytes per slot, or about 8 GB before target-record buffers, the encoded
pool, and other NumPy state.

The first run should therefore measure peak RSS as well as cold/warm cache
timings. If the available machine cannot hold the encoded pool and simulator
records, decide on a separately scoped streaming or date-chunking design; do
not change portfolio semantics opportunistically to reduce memory use.

## Acceptance criteria

- The selected three prediction files are the only prediction inputs.
- The pool is built from the cached 5-minute and daily market paths.
- Signal bar and execution bar differ by exactly one 5-minute bar, including
  from the previous trading day's final bar to the next day's first bar.
- Daily eligibility/limit/T+1 constraints continue to match the common engine.
- Results are isolated in the new run directory and do not overwrite daily
  outputs.
- The existing regression suite and both focused checkpoint/configuration tests
  pass.
