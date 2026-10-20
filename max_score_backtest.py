"""Sequential daily max_score integration with genuine account feedback.

This path deliberately leaves the legacy equal-weight state machine and batch
target preparation untouched. One portfolio owns one account and at most one
active optimizer lease, independent of the length of a backtest.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from data_loader import BacktestDataset
from max_score_config import validate_max_score_settings
from max_score_state import ShareAccount, execute_next_bar
from optimizer_adapter import _append_audit, _daily_timestamp
from portfolio import PortfolioResult
from providers.max_score import LongExposureProvider, SnapshotDataError, canonical_symbol


def build_max_score_request(pool: BacktestDataset, bar_idx: int, account: ShareAccount,
                            settings: dict, model: dict, timeout_ms: int,
                            providers: dict) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """Align candidates UNION all actual holdings at the decision close."""
    start, end = map(int, pool.bar_offsets[bar_idx:bar_idx + 2])
    rows = np.arange(start, end)
    row_map = {int(pool.row_symbol_ids[r]): int(r) for r in rows}
    eligible = pool.can_open_base[rows] & (pool.size_rank[rows] <= settings["size_cut"])
    nonfinite_excluded = 0
    if settings["candidate_policy"] == "finite_score_eligible":
        nonfinite_excluded = int((eligible & ~np.isfinite(pool.pred[rows])).sum())
        eligible &= np.isfinite(pool.pred[rows])
    candidates = pool.row_symbol_ids[rows[eligible]]
    held = np.flatnonzero(account.shares > 0)
    symbol_ids = np.union1d(candidates, held).astype(np.int32)
    scores = np.asarray([pool.pred[row_map[int(s)]] if int(s) in row_map else np.nan
                         for s in symbol_ids], dtype=np.float64)
    missing = ~np.isfinite(scores)
    missing_held = missing & (account.shares[symbol_ids] > 0)
    if missing_held.any():
        raise SnapshotDataError("MISSING_SCORE_FOR_HELD", f"{int(missing_held.sum())} actual holdings")
    if missing.any():
        raise SnapshotDataError("MISSING_SCORE", f"{int(missing.sum())} candidate scores")
    if not len(symbol_ids) and settings["target_budget"] > 0:
        raise SnapshotDataError("NO_SIGNAL", "empty candidate universe and no holdings")
    symbols = [canonical_symbol(pool.symbols[s]) for s in symbol_ids]
    if len(set(symbols)) != len(symbols):
        raise SnapshotDataError("DUPLICATE_ASSET_ID", "symbol conversion collides")
    bars = pd.to_datetime(pool.bars)
    as_of = _daily_timestamp(bars[bar_idx], 15)
    planned_at = _daily_timestamp(bars[bar_idx + 1], 9, 30)
    actual = account.weights(symbol_ids)
    cash_weight = account.cash / account.nav
    upper = np.ones(len(symbol_ids), dtype=np.float64)
    candidate_set = set(map(int, candidates))
    for i, sid in enumerate(symbol_ids):
        if int(sid) not in candidate_set:
            upper[i] = actual[i]  # excluded old holdings remain sellable, never new buys
    inputs = {"scores": scores, "current_weights": actual,
              "current_cash": np.asarray([cash_weight], dtype=np.float64), "upper_weights": upper}
    constraints = [{"type": "trade_bounds", "upper_input": "upper_weights"}]
    snapshot = {
        "valuation_at": as_of.isoformat(), "weight_denominator": "total_nav",
        "asset_order_hash": hashlib.sha256(json.dumps(symbols, ensure_ascii=False,
                                          separators=(",", ":")).encode()).hexdigest(),
        "source_point_in_time_status": settings["source_point_in_time_status"],
        "sources": {"purpose": settings["purpose"], "price_convention": settings["price_convention"],
                    "prediction": {"field": "pred", "transform": "none", "availability": "unknown"},
                    "valuation": {"field": "bar_close", "cash": "actual_remaining_cash",
                                  "missing_policy": settings["missing_valuation_policy"]}},
    }
    cap = settings.get("max_weight")
    if isinstance(cap, dict):
        mapping = {canonical_symbol(s): v for s, v in cap.items()}
        if len(mapping) != len(cap) or set(symbols).difference(mapping):
            raise SnapshotDataError("MISSING_MAX_WEIGHT", "explicit cap mapping must cover the complete universe")
        inputs["max_weights"] = np.asarray([mapping[s] for s in symbols], dtype=np.float64)
        constraints.append({"type": "max_weight", "upper_input": "max_weights"})
    elif cap is not None:
        constraints.append({"type": "max_weight", "upper": cap})
    if "size" in providers:
        source = settings["size_constraint"]["source"]
        meta = source["metadata"]
        inputs["size_exposure"] = providers["size"].snapshot(bars[bar_idx], symbols, as_of=as_of)
        snapshot["size"] = {"exposure_id": meta["exposure_id"], "field": source["value_column"],
                            "source": meta["source"], "version": meta["version"], "unit": meta["units"],
                            "transform": meta["transformation"], "semantics_status": meta["semantic_status"]}
        constraints.append({"type": "size", "exposure_id": meta["exposure_id"], "unit": meta["units"],
                            "lower": settings["size_constraint"]["lower"],
                            "upper": settings["size_constraint"]["upper"]})
    if "industry" in providers:
        meta = settings["industry_constraint"]["source"]["metadata"]
        industry_ids = providers["industry"].snapshot(bars[bar_idx], symbols, as_of=as_of)
        # Absent constrained industries remain explicit zero-exposure rows.
        ids = sorted(set(industry_ids.tolist()) |
                     {bound["industry_id"] for bound in settings["industry_constraint"]["bounds"]})
        code_by_id = {identity: i + 1 for i, identity in enumerate(ids)}
        inputs["industry_codes"] = np.asarray([code_by_id[i] for i in industry_ids], dtype=np.int64)
        snapshot["industry"] = {"classification_id": meta["classification"], "source": meta["source"],
                                "version": meta["version"], "level": meta["level"],
                                "code_to_id": {str(code): identity for identity, code in code_by_id.items()}}
        constraints.append({"type": "industry", "classification_id": meta["classification"],
                            "bounds": settings["industry_constraint"]["bounds"]})
    if settings.get("turnover_constraint") is not None:
        constraints.append({"type": "turnover", **settings["turnover_constraint"]})
    request = {
        "schema_version": "1.2", "request_id": str(uuid.uuid4()),
        "data_context": {"as_of": as_of.isoformat(), "planned_execution_at": planned_at.isoformat(),
                         "frequency": "daily"},
        "portfolio_policy": {"long_only": True, "fully_invested": False, "allow_cash": True,
                             "allow_leverage": False, "cash_policy": "retain",
                             "target_budget": settings["target_budget"]},
        "model": model, "universe": {"asset_ids": symbols}, "inputs": inputs,
        "constraints": constraints, "snapshot": snapshot,
        "options": {"timeout_ms": timeout_ms, "deterministic": True},
    }
    counts = {"candidate_count": len(candidates), "actual_held_count": len(held),
              "universe_count": len(symbol_ids), "nonfinite_new_scores_excluded": nonfinite_excluded,
              "basic_or_size_ineligible_rows": int((~(pool.can_open_base[rows] &
                                                        (pool.size_rank[rows] <= settings["size_cut"]))).sum())}
    return request, symbol_ids, scores, counts


def _actual_exposures(request: dict, weights: np.ndarray) -> dict:
    result = {"stock_weight": float(weights.sum()), "max_weight": float(weights.max()) if len(weights) else 0.0}
    if "size_exposure" in request["inputs"]:
        result["size"] = float(request["inputs"]["size_exposure"] @ weights)
    if "industry_codes" in request["inputs"]:
        codes = request["inputs"]["industry_codes"]
        result["industry"] = {identity: float(weights[codes == int(code)].sum())
                              for code, identity in request["snapshot"]["industry"]["code_to_id"].items()}
    return result


def generate_max_score_portfolio(pool: BacktestDataset, client, *, settings: dict, model: dict,
                                 portfolio_id: str, portfolio_label: int, size_cut: int,
                                 timeout_ms: int, initial_value: float, cost_per_turnover: float,
                                 output_dir: str | None = None) -> PortfolioResult:
    """Execute each prior solution before constructing the next actual snapshot."""
    validate_max_score_settings(settings)
    settings = {**settings, "size_cut": size_cut}
    bars = pd.to_datetime(pool.bars)
    if len(bars) < 2 or not (bars == bars.normalize()).all() or not bars.is_monotonic_increasing or bars.has_duplicates:
        raise ValueError("max_score integration requires at least two unique ordered daily bars")
    if client.schema_version != "1.2":
        raise ValueError("max_score requires optimizer schema 1.2")
    providers = {}
    for identity in ("size", "industry"):
        constraint = settings.get(identity + "_constraint")
        if constraint is not None:
            providers[identity] = LongExposureProvider(constraint["source"], start=bars[0], end=bars[-1],
                                                      industry=identity == "industry")
    account = ShareAccount.empty(len(pool.symbols), initial_value)
    previous_nav = initial_value
    pending = None
    calls, decision_rows, position_rows, target_rows, account_rows, trade_rows = [], [], [], [], [], []
    returns, turnovers, held_counts, closes = [], [], [], []
    output = Path(output_dir) if output_dir else None
    audit = output / "optimizer_calls.jsonl" if output else None
    for bar_idx, bar in enumerate(bars):
        execution = {"turnover": 0.0, "fees": 0.0, "closed": 0, "trades": []}
        if pending is not None:
            request, symbol_ids, desired, priority = pending
            execution = execute_next_bar(account, pool, bar_idx, symbol_ids, desired,
                                         cost_rate=cost_per_turnover, priority=priority)
            for trade in execution["trades"]:
                trade_rows.append({"date": bar, "request_id": request["request_id"],
                                   "symbol": pool.symbols[trade["symbol_id"]], **trade})
        carried = account.mark_close(pool, bar_idx, missing_policy=settings["missing_valuation_policy"])
        nav = account.nav
        returns.append(nav / previous_nav - 1.0)
        previous_nav = nav
        turnovers.append(execution["turnover"])
        held = np.flatnonzero(account.shares > 0)
        held_counts.append(len(held))
        closes.append(execution["closed"])
        account_rows.append({"date": bar, "cash": account.cash, "stock_value": nav - account.cash,
                             "nav": nav, "fees": execution["fees"], "carried_valuation_count": carried,
                             "valuation_at": _daily_timestamp(bar, 15).isoformat()})
        if pending is not None:
            actual_record = {"portfolio_id": portfolio_id, "request_id": request["request_id"],
                             "event": "execution", "actual_execution_at": _daily_timestamp(bar, 9, 30).isoformat(),
                             "actual_valuation_at": _daily_timestamp(bar, 15).isoformat(),
                             "cash": account.cash, "nav": nav, "fees": execution["fees"],
                             "actual_post_execution": _actual_exposures(request, account.weights(symbol_ids)),
                             "status": "executed", "carried_valuation_count": carried}
            calls.append(actual_record)
            _append_audit(audit, actual_record)
        start, end = map(int, pool.bar_offsets[bar_idx:bar_idx + 2])
        row_map = {int(pool.row_symbol_ids[r]): r for r in range(start, end)}
        for sid, actual_weight in zip(held, account.weights(held)):
            row = row_map.get(int(sid))
            position_rows.append({"date": bar, "symbol": pool.symbols[sid], "shares": account.shares[sid],
                                  "valuation_price": account.prices[sid], "weight_actual": actual_weight,
                                  "weight": actual_weight, "size_rank": int(pool.size_rank[row]) if row is not None else 999999,
                                  "pred": float(pool.pred[row]) if row is not None else np.nan,
                                  "pred_rank": np.nan, "tradable": bool(pool.tradable[row]) if row is not None else False,
                                  "can_open": bool(pool.can_open_base[row]) if row is not None else False,
                                  "valuation_carried": row is None or not np.isfinite(pool.bar_close[row])
                                                       or pool.bar_close[row] <= 0})
        pending = None
        if bar_idx == len(bars) - 1:
            continue
        prepare_started = time.perf_counter()
        try:
            request, symbol_ids, scores, counts = build_max_score_request(
                pool, bar_idx, account, settings, model, timeout_ms, providers)
        except SnapshotDataError as exc:
            if exc.code == "NO_SIGNAL" and settings["no_signal_policy"] == "hold":
                row = {"portfolio_id": portfolio_id, "decision_as_of": _daily_timestamp(bar, 15).isoformat(),
                       "event": "no_signal", "status": "hold_without_new_solution"}
                calls.append(row)
                _append_audit(audit, row)
                continue
            _append_audit(audit, {"portfolio_id": portfolio_id, "event": "data_error", "status": "failed",
                                  "decision_as_of": _daily_timestamp(bar, 15).isoformat(), "code": exc.code,
                                  "error": str(exc)})
            raise
        client.validate_capacity(len(symbol_ids))
        prepare_ms = (time.perf_counter() - prepare_started) * 1000
        lease = weights = None
        primary = None
        try:
            lease, wait_ms = client.optimize(request)
            weights = lease.weights
            response = lease.response
            if response.get("status") != "feasible" or response.get("diagnostics", {}).get("optimality_verified") is not True:
                raise ValueError("max_score did not return a verified optimal solution")
            if (weights is None or len(weights) != len(symbol_ids) or not np.isfinite(weights).all()
                    or (weights < -1e-8).any() or abs(float(weights.sum()) - settings["target_budget"]) > 1e-8):
                raise ValueError("max_score returned invalid weights")
            positive = weights > 0
            prices = account.prices[symbol_ids]
            if np.any(positive & (~np.isfinite(prices) | (prices <= 0))):
                raise SnapshotDataError("MISSING_SIZING_PRICE", "positive target lacks decision-close price")
            current_close_known = np.asarray([
                int(sid) in row_map and np.isfinite(pool.bar_close[row_map[int(sid)]])
                and pool.bar_close[row_map[int(sid)]] > 0 for sid in symbol_ids], dtype=bool)
            if np.any(positive & ~current_close_known & (account.shares[symbol_ids] == 0)):
                raise SnapshotDataError("MISSING_SIZING_PRICE", "new purchase cannot use a stale cached close")
            desired = np.zeros(len(symbol_ids), dtype=np.float64)
            desired[positive] = weights[positive] * nav / prices[positive]
            # Raw signal scores, with stable symbol-ID ties, determine fills.
            priority = np.lexsort((symbol_ids, -scores))
            pending = (request, symbol_ids, desired, priority)
            row = {"portfolio_id": portfolio_id, "request_id": request["request_id"],
                   "decision_as_of": request["data_context"]["as_of"],
                   "planned_execution_at": request["data_context"]["planned_execution_at"],
                   "event": "target" if settings["target_budget"] else "liquidate", "status": response["status"],
                   "schema_version": "1.2", "model_type": "max_score", "model_version": "1",
                   "portfolio_size_label": portfolio_label, "target_budget": settings["target_budget"],
                   "cash": account.cash, "nav": nav, "actual_cash_weight": account.cash / nav,
                   "prepare_ms": prepare_ms, "wait_ms": wait_ms, "shared_bytes": lease.shared_bytes,
                   "copy_bytes": lease.copy_bytes, "diagnostics": response.get("diagnostics", {}), **counts}
            calls.append(row)
            _append_audit(audit, row)
            for i, (sid, weight) in enumerate(zip(symbol_ids, weights)):
                decision_rows.append({"portfolio_id": portfolio_id, "request_id": request["request_id"],
                                      "decision_as_of": request["data_context"]["as_of"],
                                      "planned_execution_at": request["data_context"]["planned_execution_at"],
                                      "symbol": pool.symbols[sid], "weight_target": float(weight),
                                      "weight_actual": float(request["inputs"]["current_weights"][i]),
                                      "actual_shares": account.shares[sid], "valuation_price": prices[i],
                                      "score": scores[i], "target_budget": settings["target_budget"],
                                      "portfolio_size": portfolio_label, "target_count": int(positive.sum()),
                                      "cash_policy": "retain"})
                target_rows.append({"date": bars[bar_idx + 1], "symbol": pool.symbols[sid],
                                    "weight_target": float(weight), "request_id": request["request_id"],
                                    "decision_as_of": request["data_context"]["as_of"]})
        except BaseException as exc:
            primary = exc
            _append_audit(audit, {"portfolio_id": portfolio_id, "request_id": request["request_id"],
                                  "decision_as_of": request["data_context"]["as_of"],
                                  "event": "solver_error", "status": "failed", "error": str(exc),
                                  "response": getattr(exc, "response", None)})
            raise
        finally:
            weights = None
            if lease is not None:
                try:
                    lease.release()
                except Exception as cleanup:
                    if primary is None:
                        raise
                    primary.cleanup_errors = list(getattr(primary, "cleanup_errors", [])) + [
                        {"error_type": type(cleanup).__name__, "error": str(cleanup)}]
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        pl.from_pandas(pd.DataFrame(account_rows)).write_parquet(output / f"account_state_{portfolio_label}.parquet")
        pl.from_pandas(pd.DataFrame(trade_rows, columns=["date", "request_id", "symbol", "symbol_id", "side", "shares",
                                                       "notional", "fee", "status", "requested_shares"])).write_parquet(
            output / f"executed_trades_{portfolio_label}.parquet")
    position_columns = ["date", "symbol", "shares", "valuation_price", "weight_actual", "weight", "size_rank",
                        "pred", "pred_rank", "tradable", "can_open", "valuation_carried"]
    positions = pd.DataFrame(position_rows, columns=position_columns).set_index(["date", "symbol"]).sort_index()
    target_frame = pd.DataFrame(target_rows, columns=["date", "symbol", "weight_target", "request_id", "decision_as_of"])
    target_frame = target_frame.set_index(["date", "symbol"]).sort_index()
    turnover = pd.Series(turnovers, index=bars, name="turnover")
    return PortfolioResult(positions=positions, close_counts=pd.DataFrame({"n_closed": closes}, index=bars),
                           portfolio_returns=pd.Series(returns, index=bars, name="portfolio_return"),
                           cost_turnover=turnover.copy().rename("cost_turnover"),
                           borrow_cost=pd.Series(0.0, index=bars, name="borrow_cost"), turnover=turnover,
                           held_counts=pd.Series(held_counts, index=bars, name="held_count"),
                           target_weights=target_frame, optimizer_calls=calls,
                           decision_targets=pd.DataFrame(decision_rows))
