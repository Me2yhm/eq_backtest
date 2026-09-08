# Optimizer integration verification v0.2

Date: 2026-09-07

This is a reproducible synthetic verification record, not a production-market
performance claim. Server and client ran as independent same-UID processes on
Linux 4.18 x86_64; the server used Python 3.14.3 and the EQ client Python 3.11.9.

## Frozen decisions

The optimizer repository owns ADR-0001 and Schema 1.1. Scope is daily long-only;
selection stays in EQ Backtest; `M < N` retains cash and returns `1/N`; transport
is memfd + AF_UNIX/SCM_RIGHTS with zero retry and no HTTP; day `t` close decisions
execute on the next trading day.
Acceptance tolerances are absolute `1e-10` for published weights and `1e-8` for
daily-return baseline comparisons; identifiers and timestamps must match exactly.

## Evidence

- Model/contract: 1, 2, 800, and 5,000 assets; partial budget; empty, duplicate,
  unknown, and unsupported inputs.
- IPC: independent process HELLO/HEALTH/CAPABILITIES, both-direction shared-view
  sentinel, success and error responses, descriptor identity, `CONSUMED`, and
  `RELEASE`.
- Consumption: a test-only `0.4/0.1` target reaches the common Numba execution
  core unchanged; the local equal-weight branch does not overwrite it.
- Causality: changing the next execution day's opening flag does not change the
  prior close's selected request assets.
- Failure: invalid shared weights fail before publication; disabled configuration
  constructs no client; enabled failures write `complete: false`.
- Compatibility: the complete EQ test suite passes with the service disabled.

Output leases for all decisions remain live through the single common-core run,
then views are destroyed and buffers are consumed/released. Equal-weight declares
no prediction, current-weight, or risk dependency, so those fields are not
invented or sent. Provider metadata/dependency checks are present for later
models, but production capabilities still advertise only equal weight.

## Synthetic latency

Command: `scripts/benchmark_optimizer.py --repeats 30`; one warm-up request per
size was excluded. Values are milliseconds.

| Assets | Wait P50 | Wait P95 | Model P50 | Model P95 | Shared bytes | Transport copy bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.260 | 0.345 | 0.021 | 0.026 | 8 | 0 |
| 800 | 0.701 | 0.713 | 0.026 | 0.036 | 6,400 | 0 |
| 5,000 | 1.847 | 1.883 | 0.035 | 0.036 | 40,000 | 0 |

`transport copy bytes` covers the published shared output through model write and
client/core reads. Asset-ID JSON, audit scalar materialization, input preparation,
page faults, and execution state updates are outside that boundary and are
reported separately by runtime audit fields where applicable.

## Known delivery boundary

No production covariance, mean-variance solver, or risk provider is included in
v0.2. Stateful future models must add their declared provider and decision-state
snapshot before capability enablement. The existing equal-weight service can be
rolled back only by starting a new run with `optimizer.enabled: false`; results
from local and service paths are never mixed in one run.
