# Sequential max_score/1 integration

`max_score/1` uses Schema 1.2 and maximizes the original prediction values. The
existing default, `equal_weight/1`, Schema 1.1, and its batch target preparation
remain unchanged. The new backtest path currently accepts **daily, long-only,
next-bar execution** only. The mathematical core is frequency-independent;
intraday execution through this adapter has not been accepted.

The model is opt-in. Set `optimizer.model` to
`{type: max_score, version: "1", config: {}}`, `schema_version: "1.2"`, and
explicitly populate `optimizer.max_score`. `port_sizes` becomes an output and
independent-account label on this path: it never truncates candidates, changes
the stock budget, or imposes a count constraint. Increase `optimizer.max_assets`
and the service's `--max-assets` to cover the full candidate/held union, for
example 6000. Over-limit input fails; it is never truncated.

## Explicit settings

The complete template is [examples/max_score_config.yml](../examples/max_score_config.yml).
Every new public key has a default in `config.py`; unresolved investment choices
default to `null`, and enabling max_score without them reports an error.

| Setting | Meaning |
| --- | --- |
| `purpose` | `engineering_acceptance` or `strategy_backtest`; test values are not investment recommendations. |
| `target_budget` | Required stock allocation B in total NAV, from 0 through 1; cash target is 1-B. |
| `max_weight` | Optional hard cap: scalar or complete symbol-to-cap mapping. `null` disables the cap. A mapping must cover the entire current union including old holdings. |
| `size_constraint` | `null`, or explicit `lower`, `upper`, and `source` with matching exposure identity/units. Bounds are absolute total-NAV exposures. |
| `industry_constraint` | `null`, or explicit industry-ID `bounds` and one classification source. Other valid industry IDs can remain unconstrained. |
| `turnover_constraint` | `null`, or explicit `definition` and `limit`; no initial-build or liquidation exemption. |
| `candidate_policy` | `finite_score_eligible` excludes new assets without finite scores; `eligible` instead rejects their missing scores. Both require basic opening eligibility and configured `pool_size` size eligibility, never score Top-N. Exclusion counts are audited. |
| `missing_valuation_policy` | `error`, or explicit `carry_last` closing mark for an absent/invalid close. A previously unknown held mark is always an error. |
| `no_signal_policy` | `error`, or `hold` only when **both the candidate set and actual book are empty**. A held asset lacking a score always reports `MISSING_SCORE_FOR_HELD`, including when all forecasts disappear. HOLD is logged as no new solution. |
| `source_point_in_time_status` | `synthetic`, `unverified`, or `verified`; real source publication history must be established independently before claiming verified status. |
| `price_convention` | `consistent_share_prices`, or `engineering_unverified` restricted to engineering acceptance. The latter does not certify a real execution benchmark. |

The existing `weight_mode: equal` configuration value remains the compatibility
gate for enabling an optimizer; the selected model supplies the actual weights.
The max_score path does not call the local equal/rank allocation formula.

## Account state and timing

For each signal date, the engine first executes the previous decision, deducts
actual fees, and marks the resulting shares to that day's close. The next
request uses the same close valuation, actual remaining cash, and the complete
held book. NAV is `cash + sum(shares * valuation_price)`; this simple synthetic
account has no other assets, liabilities, dividends, or corporate-action model.
Sources must therefore provide a consistent share/price convention. Adjusted
and raw prices cannot be mixed without an explicit account transformation.

The universe is `allowed candidates UNION all actual held assets`. An excluded
old holding has an upper target bound equal to its actual weight, allowing a
reduction while forbidding a new increase. Its score and any enabled exposure
remain required. No past target or reporting weight is used as actual state.
The request does not pre-read the next execution date's limits or liquidity.

Targets are sized as `weight * decision_close_NAV / decision_close_price`.
Those quantities execute at the next configured VWAP, sells before buys.
Execution checks that bar's buy/sell flags, opening eligibility, available cash,
and calendar-day T+1 freezes. A failed order leaves actual shares unchanged;
cash can partially fund a buy. Costs equal executed notional times the existing
one-way `cost_per_turnover`; they are charged once in account cash. This path's
reported execution turnover is gross executed stock notional divided by the
prior decision NAV. It is separate from the solver's planned turnover metric.
No integer lot, participation, impact, queue, or corporate-action model is added.

Allowed planned turnover formulas are `stock_l1 = sum(abs(w-w0))`,
`stock_half_l1 = 0.5*sum(abs(w-w0))`, and `cash_inclusive_half_l1 =
0.5*(sum(abs(w-w0))+abs((1-B)-cash0))`. The core returns independently recomputed
target measurements. Execution audits separately report actual post-execution
exposure using the decision's exposure snapshot. Price moves, fees and blocked
trades can cause that actual book to violate a target bound; the audit does not
claim that a target bound guarantees a later fill or a later price weight.

## Exposure provider and benchmark conversion

Each enabled constraint declares exact long-form Parquet `path` or `paths`,
`date_column`, `symbol_column`, `value_column`, optional
`available_at_column`, and metadata. Relative paths resolve from the run folder.
There is no directory discovery, source substitution, factor transform, or
automatic lag. Source dates are observation dates; the optional publication
timestamp must include a timezone and must be no later than the decision.
Without that column, source publication history remains unknown.

Required metadata is `exposure_id`, `source`, `version`, `units`,
`transformation: identity`, and `semantic_status`. Industry additionally
requires `classification` and `level`. The raw source field is recorded in the
request. This supports explicitly selecting raw SIZE or `market_cap_3` without
claiming they have the same meaning. To use log or normalized exposures, prepare
and document that transformed source explicitly; the adapter never guesses it.

Duplicate observation-date/symbol keys, null keys, missing required values,
boolean pseudo-numbers and unknown industry zero fail. Numeric industry codes
must be finite positive integers; explicit nonempty category strings are also
supported and receive a deterministic positive integer encoding per snapshot.
The request carries the complete code-to-industry-ID mapping. Industry bounds
remain keyed by the declared IDs, including a lower bound on an absent industry.

`providers.benchmark_bounds.relative_to_absolute_bounds` accepts a complete
independently declared benchmark membership list, stock-sleeve weights summing
to one, complete aligned exposures, and matching definition/version/units.
It never accepts the optimizer candidate universe or renormalizes a subset.
For benchmark exposure `e`, stock-sleeve deviation `d` gives `B*(e +/- d)`;
total-NAV deviation gives `B*e +/- d`. It returns absolute bounds for this
configuration. Industry relative bounds use one 0/1 exposure vector per explicit
industry ID. Missing real benchmark metadata remains a blocked real relative
example; synthetic conversion tests do not authenticate a real benchmark.

## Outputs and reproducible local execution

Existing reporting files remain available. Max_score adds
`account_state_<label>.parquet` (cash, stock value, NAV, actual fees, carried
mark counts) and `executed_trades_<label>.parquet` (requested/filled shares,
notional, fees, failure reason). `decision_targets.parquet` includes actual
shares/weights, decision marks and raw signal scores alongside target weights.
`optimizer_calls.jsonl` distinguishes target, explicit liquidation, execution,
data error, solver error and empty-book hold. Failures stop the run without
fallback/retry and leave `optimizer_run_status.json` incomplete.

The formal dependency is `optimizer[lp]` at the merged Git SHA
`3ffc0e128799379aa37c61bbb7ada900eac6952f`, with SciPy included in `uv.lock`.
Use a new environment and `uv sync --frozen`, then verify the installed Git
identity from outside both repositories as described in
[MAX_SCORE_LOCAL_VALIDATION.md](MAX_SCORE_LOCAL_VALIDATION.md). With
`TASK_ROOT/venv` created by that procedure, run from the EQ checkout:

```bash
PYTHONPATH= "$TASK_ROOT/venv/bin/python" -m unittest discover -s tests -p test_max_score_integration.py -v
PYTHONPATH= "$TASK_ROOT/venv/bin/python" scripts/verify_max_score_modes.py --work-dir /tmp/new-max-score-verification
```

The second command creates fixed synthetic inputs, 25 consecutive decisions
per account, all four constraints, two labels, a real separate Linux SHM
service, two CLI runs, account reconciliation, actual/target divergence checks,
hard infeasibility runs, logs and `comparison.json`. The output folder must be
new; local Unix socket permission is required. In-process and SHM consume the
same Schema 1.2 snapshot and mathematical core. Only one lease is live at a
time; static provider inputs may be cached while audit output grows with time.
Solver internal matrix preparation is distinct from SHM transport zero-copy.

Original data's VWAP/adjustment semantics and historical publication revisions
remain independent acceptance conditions. Real-source single-snapshot checks
and synthetic execution checks must not be described as a certified historical
strategy backtest. Engineering performance or returns do not select strategy
parameters or establish economic score direction.

See [local installation and validation](MAX_SCORE_LOCAL_VALIDATION.md) for source,
base-package, wheel, legacy pressure, full-market and real-snapshot commands.
