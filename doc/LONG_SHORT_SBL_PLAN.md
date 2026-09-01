# SBL-Constrained Long-Short Backtest Plan

## Status

Approved design only. This document does not change the current backtest
implementation.

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
date, symbol, counterparty, available_shares, annual_borrow_rate, borrow_available
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
`Sheet1` contains the header, while `Sheet2` is a headerless continuation after
Excel's row limit. The adaptor must read all worksheets, accept the known
continuation layout, zero-pad six-digit stock codes, and map `SH.*` to `.XSHG`
and `SZ.*` to `.XSHE`.

The workbook is large, so the adaptor should use streaming Excel reads and
write a normalized Parquet cache under the run cache. The cache key includes
the adaptor version/configuration and the raw source-file signatures. External
SBL files remain read-only.

Each future counterparty requires its own small adaptor plus a fixture and
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
  - adapter: yading
    path: ../../cache/SBL/Yading/20250424-20260709券单.xlsx
```

Each mode writes to its own output directory, preventing long-only, short-only,
and long-short results from overwriting one another. Combined outputs include
side-specific returns, costs, turnover, held counts, and a signed position file
with a `side` column. Short-position audit fields include availability at entry,
locked borrow rate, borrow-cost accrual, and counterparty provenance.

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
