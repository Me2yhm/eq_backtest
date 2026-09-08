"""Causal daily decision selection and optimizer request adaptation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import uuid

from numba import types
from numba.typed import List
import numpy as np
import pandas as pd

from data_loader import BacktestDataset
from optimizer_client import OptimizerClient, OptimizerClientError, OptimizerLease
from providers import required_inputs


NO_SIGNAL = np.int8(0)
TARGET = np.int8(1)
LIQUIDATE = np.int8(2)
_DECISION_TARGET_COLUMNS = [
    "portfolio_id", "request_id", "decision_as_of", "planned_execution_at", "symbol",
    "weight_target", "portfolio_size", "target_count", "target_budget", "cash_policy",
]


@dataclass(frozen=True, slots=True)
class DecisionContext:
    portfolio_id: str
    as_of: pd.Timestamp
    planned_execution_at: pd.Timestamp
    symbol_ids: np.ndarray
    # equal_weight/1 declares no state dependency. These remain explicit and
    # absent rather than being populated with invented values.
    actual_shares: np.ndarray | None = None
    valuation_prices: np.ndarray | None = None


@dataclass(slots=True)
class OptimizerTargetPlan:
    events: np.ndarray
    offsets: np.ndarray
    symbol_ids: np.ndarray
    weights_by_bar: object
    leases: list[OptimizerLease]
    calls: list[dict]
    decision_targets: pd.DataFrame

    def close(self) -> None:
        self.weights_by_bar = None
        errors: list[Exception] = []
        for lease in self.leases:
            try:
                lease.release()
            except Exception as exc:
                errors.append(exc)
        self.leases.clear()
        if errors:
            raise OptimizerClientError(f"failed to release {len(errors)} optimizer lease(s)") from errors[0]


def _iso(timestamp: pd.Timestamp) -> str:
    value = timestamp
    if value.tzinfo is None:
        value = value.tz_localize("Asia/Shanghai")
    return value.isoformat()


def _daily_timestamp(timestamp: pd.Timestamp, hour: int, minute: int = 0) -> pd.Timestamp:
    value = pd.Timestamp(timestamp).normalize() + pd.Timedelta(hours=hour, minutes=minute)
    return value.tz_localize("Asia/Shanghai") if value.tzinfo is None else value


def _contract_symbol(symbol: object) -> str:
    value = str(symbol)
    if value.endswith(".XSHE"):
        return value[:-5] + ".SZ"
    if value.endswith(".XSHG"):
        return value[:-5] + ".SH"
    return value


def _append_audit(path: Path | None, row: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def _decision_target_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_DECISION_TARGET_COLUMNS)


def _select_targets(
    pool: BacktestDataset,
    *,
    port_size: int,
    thresh_out: int,
    size_cut: int,
    close_on_size_drop: bool,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Select on signal-bar information only; execution-bar flags are excluded."""
    n_bars = len(pool.bars)
    events = np.full(n_bars, NO_SIGNAL, dtype=np.int8)
    selections: list[np.ndarray] = [np.empty(0, dtype=np.int32) for _ in range(n_bars)]
    previous: list[int] = []
    previous_set: set[int] = set()

    for bar_idx in range(max(0, n_bars - 1)):
        start = int(pool.bar_offsets[bar_idx])
        end = int(pool.bar_offsets[bar_idx + 1])
        rows = np.arange(start, end, dtype=np.int64)
        valid = rows[np.isfinite(pool.pred[start:end])]
        if valid.size == 0:
            continue
        local_order = np.arange(valid.size, dtype=np.int64)
        sorter = np.lexsort((local_order, -pool.pred[valid]))
        ranked_rows = valid[sorter]
        row_for_symbol = {int(pool.row_symbol_ids[row]): int(row) for row in ranked_rows}
        rank_by_symbol: dict[int, int] = {}
        in_size: set[int] = set()
        for row in ranked_rows:
            symbol_id = int(pool.row_symbol_ids[row])
            if pool.size_rank[row] <= size_cut and pool.can_open_base[row]:
                in_size.add(symbol_id)
            if symbol_id in in_size or symbol_id in previous_set:
                rank_by_symbol[symbol_id] = len(rank_by_symbol) + 1

        retained: list[int] = []
        for symbol_id in previous:
            row = row_for_symbol.get(symbol_id)
            rank = rank_by_symbol.get(symbol_id, 0)
            eligible = row is not None and bool(pool.can_open_base[row])
            leaves_size = close_on_size_drop and symbol_id not in in_size
            if eligible and rank > 0 and rank <= thresh_out and not leaves_size:
                retained.append(symbol_id)
        selected = retained
        selected_set = set(selected)
        for row in ranked_rows:
            if len(selected) == port_size:
                break
            symbol_id = int(pool.row_symbol_ids[row])
            rank = rank_by_symbol.get(symbol_id, 0)
            if rank == 0:
                continue
            if rank > port_size:
                break
            if symbol_id in in_size and symbol_id not in selected_set:
                selected.append(symbol_id)
                selected_set.add(symbol_id)
        previous = selected
        previous_set = selected_set
        if selected:
            events[bar_idx] = TARGET
            selections[bar_idx] = np.asarray(selected, dtype=np.int32)
        else:
            events[bar_idx] = LIQUIDATE
    return events, selections


def prepare_optimizer_targets(
    pool: BacktestDataset,
    client: OptimizerClient,
    *,
    portfolio_id: str,
    port_size: int,
    thresh_out_buffer: int,
    size_cut: int,
    close_on_size_drop: bool,
    model: dict,
    timeout_ms: int,
    audit_path: Path | None = None,
) -> OptimizerTargetPlan:
    if isinstance(port_size, bool) or not isinstance(port_size, int) or port_size <= 0:
        raise ValueError("optimizer portfolio size must be a positive integer")
    dependencies = required_inputs(client.capabilities, model)
    if dependencies:
        raise ValueError(
            "configured optimizer model requires data providers that are not enabled: "
            + ", ".join(dependencies)
        )
    events, selections = _select_targets(
        pool, port_size=port_size, thresh_out=port_size + thresh_out_buffer,
        size_cut=size_cut, close_on_size_drop=close_on_size_drop,
    )
    required_shared_bytes = sum(len(selection) * 8 for selection in selections)
    service_limit = client.capabilities.get("limits", {}).get("max_shared_bytes_per_session")
    if not isinstance(service_limit, int) or required_shared_bytes > min(service_limit, client.max_shared_bytes):
        raise OptimizerClientError(
            f"optimizer target leases require {required_shared_bytes} shared bytes, above the session limit",
            client.capabilities,
        )
    offsets = np.zeros(len(pool.bars) + 1, dtype=np.int64)
    flat_parts: list[np.ndarray] = []
    leases: list[OptimizerLease] = []
    calls: list[dict] = []
    target_rows: list[dict] = []
    weights_by_bar = List.empty_list(types.Array(types.float64, 1, "C", readonly=True))
    bars = pd.to_datetime(pool.bars)
    config_signature = hashlib.sha256(json.dumps(
        {"model": model, "port_size": port_size, "thresh_out_buffer": thresh_out_buffer,
         "size_cut": size_cut, "close_on_size_drop": close_on_size_drop,
         "cash_policy": "retain", "schema_version": client.schema_version,
         "protocol_version": client.protocol_version},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()

    try:
        for bar_idx, event in enumerate(events):
            selected = selections[bar_idx]
            offsets[bar_idx + 1] = offsets[bar_idx] + len(selected)
            if selected.size:
                flat_parts.append(selected)
            if event == NO_SIGNAL:
                empty = np.empty(0, dtype=np.float64)
                empty.setflags(write=False)
                weights_by_bar.append(empty)
                if bar_idx < len(bars) - 1:
                    row = {"portfolio_id": portfolio_id,
                           "decision_as_of": _iso(_daily_timestamp(bars[bar_idx], 15)),
                           "planned_execution_at": _iso(_daily_timestamp(bars[bar_idx + 1], 9, 30)),
                           "actual_execution_at": _iso(_daily_timestamp(bars[bar_idx + 1], 9, 30)),
                           "event": "no_signal",
                           "status": "skipped", "request_id": None,
                           "config_signature": config_signature}
                    calls.append(row)
                    _append_audit(audit_path, row)
                continue
            if event == LIQUIDATE:
                empty = np.empty(0, dtype=np.float64)
                empty.setflags(write=False)
                weights_by_bar.append(empty)
                row = {"portfolio_id": portfolio_id,
                       "decision_as_of": _iso(_daily_timestamp(bars[bar_idx], 15)),
                       "planned_execution_at": _iso(_daily_timestamp(bars[bar_idx + 1], 9, 30)),
                       "actual_execution_at": _iso(_daily_timestamp(bars[bar_idx + 1], 9, 30)),
                       "event": "liquidate",
                       "status": "accepted", "request_id": None, "target_count": 0, "target_budget": 0.0}
                row["config_signature"] = config_signature
                calls.append(row)
                _append_audit(audit_path, row)
                continue

            prepare_started = time.perf_counter_ns()
            context = DecisionContext(
                portfolio_id,
                _daily_timestamp(bars[bar_idx], 15),
                _daily_timestamp(bars[bar_idx + 1], 9, 30),
                selected,
            )
            request_id = str(uuid.uuid4())
            target_budget = len(selected) / port_size
            request = {
                "schema_version": client.schema_version,
                "request_id": request_id,
                "data_context": {"as_of": _iso(context.as_of),
                                 "planned_execution_at": _iso(context.planned_execution_at),
                                 "frequency": "daily"},
                "portfolio_policy": {"long_only": True, "fully_invested": False, "allow_cash": True,
                                     "allow_leverage": False, "cash_policy": "retain",
                                     "target_budget": target_budget},
                "model": model,
                "universe": {"asset_ids": [_contract_symbol(pool.symbols[index]) for index in selected]},
                "constraints": [],
                "options": {"timeout_ms": timeout_ms, "deterministic": True},
            }
            prepare_ms = (time.perf_counter_ns() - prepare_started) / 1_000_000
            try:
                lease, wait_ms = client.optimize(request)
            except OptimizerClientError as exc:
                row = {"portfolio_id": portfolio_id, "request_id": request_id,
                       "decision_as_of": _iso(context.as_of),
                       "planned_execution_at": _iso(context.planned_execution_at),
                       "actual_execution_at": _iso(context.planned_execution_at),
                       "event": "target", "status": "failed", "error": str(exc)}
                row["config_signature"] = config_signature
                if exc.response:
                    row["service_status"] = exc.response.get("status")
                    row["errors"] = exc.response.get("errors", [])
                calls.append(row)
                _append_audit(audit_path, row)
                raise
            leases.append(lease)
            assert lease.weights is not None
            weights_by_bar.append(lease.weights)
            response = lease.response
            descriptor = lease.descriptor
            row = {
                "portfolio_id": portfolio_id, "request_id": request_id,
                "decision_as_of": _iso(context.as_of), "planned_execution_at": _iso(context.planned_execution_at),
                "actual_execution_at": _iso(context.planned_execution_at),
                "event": "target", "status": response["status"], "schema_version": client.schema_version,
                "protocol_version": client.protocol_version, "model_type": model["type"],
                "model_version": model["version"], "session_id": client.session_id,
                "service_epoch": client.service_epoch, "buffer_id": descriptor["buffer_id"],
                "generation": descriptor["generation"], "shared_bytes": descriptor["nbytes"],
                "staging_bytes": 0, "copy_bytes": 0, "wait_ms": wait_ms,
                "audit_materialization_bytes": descriptor["nbytes"],
                "prepare_ms": prepare_ms,
                "mapping_validation_ms": lease.mapping_validation_ms,
                "model_time_ms": response.get("diagnostics", {}).get("model_time_ms"),
                "target_count": len(selected), "portfolio_size": port_size,
                "target_budget": target_budget, "cash_policy": "retain",
                "config_signature": config_signature,
            }
            calls.append(row)
            _append_audit(audit_path, row)
            for symbol_id, weight in zip(selected, lease.weights):
                target_rows.append({
                    "portfolio_id": portfolio_id, "request_id": request_id,
                    "decision_as_of": context.as_of, "planned_execution_at": context.planned_execution_at,
                    "symbol": pool.symbols[symbol_id], "weight_target": float(weight),
                    "portfolio_size": port_size, "target_count": len(selected),
                    "target_budget": target_budget, "cash_policy": "retain",
                })
    except Exception:
        plan = OptimizerTargetPlan(events, offsets,
                                   np.concatenate(flat_parts) if flat_parts else np.empty(0, dtype=np.int32),
                                   weights_by_bar, leases, calls, _decision_target_frame(target_rows))
        plan.close()
        raise

    return OptimizerTargetPlan(
        events, offsets, np.concatenate(flat_parts) if flat_parts else np.empty(0, dtype=np.int32),
        weights_by_bar, leases, calls, _decision_target_frame(target_rows),
    )
