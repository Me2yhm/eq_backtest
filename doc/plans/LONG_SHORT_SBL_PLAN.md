# SBL-Constrained Long-Short Backtest Plan

## Status

Implemented. This document records the delivered contract and design decisions.

## Objective

Add an SBL-aware short side and support the following independently reported
strategy modes:

- `long_only`: the existing equal-weight long portfolio, evaluated against the
  configured benchmark.
- `short_only`: an equal-weight short-stock book paired with a long benchmark.
- `long_short`: an equal-weight 1.0-long / 1.0-short portfolio (2.0 gross).

The short side may open positions only in securities listed as available in an
SBL source for the corresponding date. The long side remains unaffected by SBL
data.

## SBL source adaptor layer

Create an adaptor registry in a dedicated SBL loader module. Configuration must
list the exact source files and their adaptor names; it must not recursively
discover files under `cache/SBL` or silently combine unrelated files.

Every adaptor normalizes its source into these canonical fields:

```text
date, symbol, provider, channel, available_shares, annual_borrow_rate, source_row
```

`borrow_available` is true when the source reports a positive available amount.
The initial strategy ignores `available_shares` for sizing, but retains it for
auditing and possible future capacity limits.

### Initial Yading adaptor

The current source is:

```text
cache/SBL/Yading/20250424-20260709券单.xlsx
```

Its fields are `生效日期`, `股票代码`, `股票名称`, `市场`, `利率`, and `总数`.
`Sheet1` contains the header, while `Sheet2` is a headerless continuation at a
date boundary. The adaptor reads all worksheets, accepts the known continuation
layout, defensively zero-pads stock codes, and maps `SH.*` to `.XSHG` and `SZ.*` to `.XSHE`.

The workbook is large, so the adaptor should use streaming Excel reads and
write a normalized Parquet cache under the run cache. The cache key includes
the adaptor version/configuration and the raw source-file signatures. External
SBL files remain read-only.

Each future source format requires its own adaptor plus a fixture and
schema-validation test; its format must be supplied rather than guessed.

## Availability and trade semantics

SBL availability is joined by exact `(date, symbol)`. It is never forward-filled:
a missing date or security is unavailable for new short entries. For intraday
data, the exact daily availability is broadcast to every bar for that calendar
day.

For a new short position:

1. It must be in the SBL list at the signal bar when the short candidate set is
   formed.
2. It must again be available at the execution bar before the short sale can
   occur. This preserves the existing signal/execution lag without look-ahead.
3. It must meet the current base-universe, ranking, liquidity, price-limit, and
   T+1 constraints.

After a short opens, its borrowed security is considered locked. Subsequent
absence from an SBL list does not remove its target or force a cover. A cover is
always permitted subject only to the ordinary short-cover execution constraints;
it does not require fresh availability.

The hot path receives a compact availability boolean. Pool schema and cache
version must be updated, and pool-cache identity must include all selected SBL
source signatures and adaptor settings whenever a short mode is active.

## Borrow cost

The rate reported by the SBL list is locked when a short is opened. It does not
reset if later daily lists report another rate or omit the security.

Borrow cost is accrued on the actual calendar-time interval for each open short,
including weekends and holidays:

```text
borrow_cost = short_notional * locked_annual_rate
              * elapsed_calendar_days / days_in_that_calendar_year
```

The calculation uses 365 or 366 as appropriate for each calendar-year portion
of an interval. It is deducted once inside the share-based short simulator and
is reported separately from transaction costs. There is no borrow fee before a
short executes, and accrual stops when it is covered.

## Portfolio modes and evaluation

All modes use the current equal-weight portfolio construction for now. The
existing `port_sizes` list applies symmetrically to both sleeves in
`long_short` mode.

### Long-only

No semantic change:

```text
daily excess = long-book return - benchmark return
```

### Short-only

This is a long-benchmark / short-selected-stocks spread, not an unrestricted
standalone short return:

```text
daily strategy return = benchmark return + short-book P&L return
```

The short simulator already produces signed short P&L, so it is added to the
long benchmark leg. The resulting spread is the primary evaluated return; it
must not subtract the benchmark a second time.

### Long-short

The combined portfolio has one unit of long gross exposure and one unit of short
gross exposure. Its daily return is:

```text
daily strategy return = long-book return + short-book P&L return
```

The benchmark is retained as a reference series but is not mechanically
subtracted from the combined strategy return. Each sleeve deducts its own
transaction costs, and the short sleeve also deducts locked-rate borrow costs.
Combined turnover is the sum of the two sleeve turnovers, relative to initial
NAV.

## Configuration and outputs

Introduce an explicit multi-mode configuration, retaining compatibility with
existing long-only configurations. SBL sources are resolved relative to the
dedicated run directory, as are other configured paths. A representative shape
is:

```yaml
strategy_modes: [long_only, short_only, long_short]
short_borrow_sources:
  - provider: yading
    adapter: yading
    path: ../../cache/SBL/Yading/20250424-20260709券单.xlsx
```

Each mode writes to its own output directory, preventing results from overwriting
one another. Combined positions carry a `sleeve` column. Short-position audit
fields include current availability, selected provider/channel/rate, and locked borrow rate.

## Verification

Add focused synthetic tests for:

- Yading multi-sheet/headerless-continuation parsing and SH/SZ symbol mapping.
- Exact-date availability, missing-date rejection, and intraday broadcasting.
- No short opening without availability at both signal and execution bars.
- Retention of an opened short after later list removal, with normal covering.
- Locked-rate Act/Act accrual across weekends, holidays, leap years, and covers.
- Correct short-only benchmark-plus-short and long-short return/cost aggregation.
- SBL cache invalidation when source files or adaptor settings change.

Before handoff, run the repository regression suite, compile check, and Git
diff hygiene checks specified in `AGENTS.md`.

## Authoritative design corrections

This section supersedes any conflicting wording above and is the implementation
contract for the SBL work.

### Yading provider/channel records

The Yading `市场` value is composite, with exactly these valid values:
`SH.QFII`, `SH.HK`, `SZ.QFII`, and `SZ.HK`. The adaptor must validate this
set, split the exchange prefix from the channel suffix, and map `SH` to
`.XSHG` and `SZ` to `.XSHE`. It must not infer that `QFII` or `HK` is a
counterparty without a separate business definition.

The configured source supplies the provider name (for example, `yading`).
Before selection, the canonical raw SBL schema is:

```text
date, symbol, provider, channel, available_shares, annual_borrow_rate, source_row
```

The current workbook has a six-column header in `Sheet1`, ending on
2026-04-15, and a headerless six-column `Sheet2` beginning on 2026-04-16.
This is a date-boundary continuation, not an Excel row-limit split. Leading
zero padding of codes is defensive because the current source preserves it.

### Selection from duplicate availability rows

A symbol may have several provider/channel records on the same date. The
strategy therefore selects exactly one usable borrow record per `(date,
symbol)`, rather than treating the join key as unique. `borrow_available` is
true if any row has positive availability. The initial public selection policy
is `min_available_rate`: select the positive-availability row with the lowest
annual rate, breaking ties by configured provider order, then channel, then
source row. Record the selected provider and channel with the position.

If an account cannot choose among all listed provider/channels, configuration
must filter the allowed rows before this selection. Quantity remains ignored
for sizing in this phase.

### Locked rates, increments, and Act/Act accrual

The rate is locked for each borrowed increment. Represent the outstanding
symbol borrow as borrowed shares plus a share-weighted locked annual rate:
an increase blends the existing shares/rate with new shares at the selected
execution-date rate, while a partial cover removes shares pro rata. This is
fee-equivalent to individual rate lots for a single stock and avoids dynamic
lot collections in the Numba core.

At the first bar of each calendar date, before rebalancing or covering, deduct
the fee for the carried short shares since their last accrual date. Use the
pre-trade marked notional `abs(current_shares * execution_vwap)` and split
each elapsed calendar interval at year boundaries:

```text
borrow_cost = marked_pretrade_short_notional * locked_annual_rate
              * elapsed_days_in_year_portion / 365_or_366
```

This charges non-trading calendar days, including weekends and holidays, once
per date even for intraday runs. A new short has no day-zero fee; a cover pays
the accrued fee before it executes and then stops future accrual.

The implementation precomputes one Act/Act fraction per first bar of a date and
maintains per-symbol locked-rate and borrowed-share state.

### Mode-aware evaluation and turnover

Replace boolean `is_short` evaluation branching with explicit strategy modes.
The short simulator already emits signed short P&L. Consequently:

```text
long_only  evaluated return = long_book_return - benchmark_return
short_only evaluated return = benchmark_return + short_book_pnl_return
long_short evaluated return = long_book_return + short_book_pnl_return
```

The current `compute_returns` short branch (`benchmark - short_pnl`) conflicts
with `_build_portfolio_pnl_frame` (`benchmark + short_pnl`) and must not be
reused for the new `short_only` mode. The benchmark remains a reference, not
a subtraction, for `long_short`.

Combined turnover is `long_sleeve_turnover + short_sleeve_turnover`, measured
per 1.0 base initial NAV. It is deliberately not divided by the 2.0 gross
exposure and may exceed 1.0.

### Configuration, cache, dependencies, and tests

Add defaults for every new public configuration key, including
`strategy_modes`, `borrow_selection`, and `short_borrow_sources`, and extend
nested path resolution for `short_borrow_sources[].path`. Preserve existing
long-only configuration behavior; resolve legacy `is_short` only for backward
compatibility and document its deprecation.

Use `openpyxl.load_workbook(..., read_only=True, data_only=True)` for streaming
workbook ingestion. Add `openpyxl` to `pyproject.toml` and refresh `uv.lock`.
Bump the pool-cache schema version and include exact SBL source signatures,
adaptor settings, and selection policy in the cache identity.

Tests must cover exact Yading market-value validation; its headerless second
sheet; duplicate-row selection and provenance; signal- and execution-date
availability; held-short retention after list removal; locked-rate additions
and partial covers; calendar, weekend, and leap-year accrual; and the three
mode-specific evaluation formulas.
