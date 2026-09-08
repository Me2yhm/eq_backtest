from __future__ import annotations

import os
import importlib.util
from array import array
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
import sys
import threading
from unittest.mock import MagicMock, patch

from numba import types
from numba.typed import List
import numpy as np
import pandas as pd
import polars as pl

from data_loader import BacktestDataset
from optimizer_adapter import OptimizerTargetPlan, _select_targets, prepare_optimizer_targets
from optimizer_client import OptimizerClient, OptimizerClientError
from portfolio import generate_portfolio
from providers import ProviderSnapshot, required_inputs, resolve_snapshots
import run as run_module


def _pool(*, second_day_can_open: bool = True) -> BacktestDataset:
    bars = np.asarray(pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"]), dtype="datetime64[ns]")
    symbols = np.asarray(["A", "B"], dtype=object)
    datetimes = np.repeat(bars, 2)
    symbol_values = np.tile(symbols, 3)
    row_symbol_ids = np.tile(np.asarray([0, 1], dtype=np.int32), 3)
    can_open_base = np.asarray([True, True, second_day_can_open, True, True, True])
    frame = pl.DataFrame({
        "datetime": datetimes,
        "date": [pd.Timestamp(value).date() for value in datetimes],
        "symbol": symbol_values,
        "pred": np.tile([2.0, 1.0], 3),
        "turnover": np.ones(6),
        "log_size": np.ones(6),
        "size_rank": np.tile([1, 2], 3),
        "industry": ["I"] * 6,
        "index": ["IDX"] * 6,
        "listed_Satisfied": [True] * 6,
        "is_ST": [False] * 6,
        "normal_days": [10] * 6,
        "is_limit_up": [False] * 6,
        "is_limit_down": [False] * 6,
        "can_open": [True] * 6,
        "borrow_available": [False] * 6,
        "borrow_rate": np.zeros(6),
        "borrow_provider": [None] * 6,
        "borrow_channel": [None] * 6,
    })
    return BacktestDataset(
        pool_frame=frame,
        daily_snapshot_frame=frame,
        bars=bars,
        symbols=symbols,
        bar_offsets=np.asarray([0, 2, 4, 6], dtype=np.int64),
        bar_session_index=np.asarray([0, 1, 2], dtype=np.int32),
        row_symbol_ids=row_symbol_ids,
        vwap_ret=np.zeros(6),
        prev_close=np.full(6, 10.0),
        bar_close=np.full(6, 10.0),
        execution_vwap=np.full(6, 10.0),
        pred=np.tile([2.0, 1.0], 3),
        size_rank=np.tile([1, 2], 3).astype(np.int32),
        tradable=np.ones(6, dtype=np.bool_),
        can_open=np.ones(6, dtype=np.bool_),
        can_open_base=can_open_base,
        can_trade_buy=np.ones(6, dtype=np.bool_),
        can_trade_sell=np.ones(6, dtype=np.bool_),
        borrow_available=np.zeros(6, dtype=np.bool_),
        borrow_rate=np.zeros(6),
    )


class OptimizerIntegrationTests(unittest.TestCase):
    def test_disabled_run_never_constructs_optimizer_client(self) -> None:
        with (
            patch.object(run_module.cfg, "OPTIMIZER", {"enabled": False}),
            patch.object(run_module, "_run") as execute,
            patch.object(run_module, "OptimizerClient") as client_type,
        ):
            run_module.run()
        execute.assert_called_once_with(None)
        client_type.assert_not_called()

    def test_enabled_run_records_incomplete_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = MagicMock()
            client.__enter__.return_value = client
            with (
                patch.object(run_module.cfg, "RUN_DIR", Path(tmp)),
                patch.object(run_module.cfg, "PORT_SIZES", [2]),
                patch.object(run_module.cfg, "OPTIMIZER", {
                    "enabled": True, "socket_path": Path(tmp) / "service.sock",
                    "protocol_version": "shm/1", "schema_version": "1.1",
                    "timeout_ms": 5, "request_timeout_ms": 6,
                    "max_control_bytes": 1024, "max_shared_bytes": 1024,
                }),
                patch.object(run_module, "OptimizerClient", return_value=client),
                patch.object(run_module, "_run", side_effect=RuntimeError("boom")),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    run_module.run()
            status = json.loads((Path(tmp) / "optimizer_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")
            self.assertFalse(status["complete"])

    @unittest.skipUnless(os.environ.get("RUN_OPTIMIZER_IPC_TESTS") == "1", "requires UDS permission")
    def test_client_rejects_invalid_shared_weight_sum_before_publication(self) -> None:
        client_socket, server_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        client = OptimizerClient("unused")
        client.socket = client_socket
        client.session_id = "session"
        client.service_epoch = "epoch"
        request = {
            "schema_version": "1.1", "request_id": "bad-weights",
            "data_context": {"as_of": "2026-09-01T15:00:00+08:00",
                             "planned_execution_at": "2026-09-02T09:30:00+08:00", "frequency": "daily"},
            "portfolio_policy": {"target_budget": 0.5},
            "model": {"type": "equal_weight", "version": "1", "config": {}},
            "universe": {"asset_ids": ["A", "B"]},
        }

        def respond() -> None:
            server_socket.recv(1_048_576)
            fd = os.memfd_create("invalid-optimizer-test", flags=os.MFD_CLOEXEC)
            os.ftruncate(fd, 16)
            with os.fdopen(os.dup(fd), "r+b", closefd=True) as handle:
                handle.write(np.asarray([0.3, 0.3], dtype="<f8").tobytes())
            descriptor = {"buffer_id": "bad", "generation": 1, "offset": 0, "nbytes": 16,
                          "shape": [2], "dtype": "<f8", "order": "C", "role": "output"}
            response = {"type": "OPTIMIZE_RESULT", "protocol_version": "shm/1",
                        "service_epoch": "epoch", "session_id": "session", "schema_version": "1.1",
                        "request_id": "bad-weights", "status": "feasible", "warnings": [], "errors": [],
                        "solution": {"asset_ids": ["A", "B"], "weights": descriptor},
                        "diagnostics": {"model_type": "equal_weight", "model_version": "1"}}
            payload = json.dumps(response, separators=(",", ":")).encode()
            server_socket.sendmsg([payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array("i", [fd]))])
            os.close(fd)

        worker = threading.Thread(target=respond)
        worker.start()
        try:
            with self.assertRaisesRegex(OptimizerClientError, "target_budget"):
                client.optimize(request)
        finally:
            client.close()
            server_socket.close()
            worker.join(timeout=2)

    def test_enabled_config_resolves_socket_from_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "config.yml").write_text(
                """
frequency: daily
strategy_modes: [long_only]
weight_mode: equal
freq_config:
  daily:
    trade_on_next_bar: true
optimizer:
  enabled: true
  socket_path: ipc/service.sock
""".lstrip(), encoding="utf-8")
            name = "_optimizer_enabled_config_test"
            spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "config.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            old = os.environ.get("EQ_BACKTEST_RUN_DIR")
            os.environ["EQ_BACKTEST_RUN_DIR"] = str(run_dir)
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(name, None)
                if old is None:
                    os.environ.pop("EQ_BACKTEST_RUN_DIR", None)
                else:
                    os.environ["EQ_BACKTEST_RUN_DIR"] = old
            self.assertTrue(module.OPTIMIZER["enabled"])
            self.assertEqual(module.OPTIMIZER["socket_path"], run_dir / "ipc/service.sock")

    def test_provider_dependencies_fail_before_request_and_preserve_metadata(self) -> None:
        capabilities = {"models": [{"type": "test_risk", "versions": ["1"],
                                     "required_inputs": ["risk_model"]}]}
        dependencies = required_inputs(capabilities, {"type": "test_risk", "version": "1"})
        with self.assertRaisesRegex(ValueError, "missing optimizer data provider"):
            resolve_snapshots(dependencies, {}, as_of="2026-09-01T15:00:00+08:00",
                              asset_ids=("A",), horizon="1d")

        class RiskProvider:
            def snapshot(self, *, as_of, asset_ids, horizon):
                return ProviderSnapshot(as_of, asset_ids, horizon, "synthetic", "1", np.eye(1))

        snapshots = resolve_snapshots(
            dependencies, {"risk_model": RiskProvider()}, as_of="2026-09-01T15:00:00+08:00",
            asset_ids=("A",), horizon="1d",
        )
        self.assertEqual(snapshots["risk_model"].source, "synthetic")

    def test_decision_selection_does_not_read_next_execution_day_flags(self) -> None:
        baseline = _select_targets(_pool(second_day_can_open=True), port_size=2, thresh_out=2,
                                   size_cut=10, close_on_size_drop=False)
        changed = _select_targets(_pool(second_day_can_open=False), port_size=2, thresh_out=2,
                                  size_cut=10, close_on_size_drop=False)
        np.testing.assert_array_equal(baseline[1][0], changed[1][0])

    def test_execution_core_consumes_external_weights_without_equal_recalculation(self) -> None:
        arrays = List.empty_list(types.Array(types.float64, 1, "C", readonly=True))
        for values in ([0.4, 0.1], [], []):
            array = np.asarray(values, dtype=np.float64)
            array.setflags(write=False)
            arrays.append(array)
        plan = OptimizerTargetPlan(
            events=np.asarray([1, 0, 0], dtype=np.int8),
            offsets=np.asarray([0, 2, 2, 2], dtype=np.int64),
            symbol_ids=np.asarray([0, 1], dtype=np.int32),
            weights_by_bar=arrays,
            leases=[],
            calls=[],
            decision_targets=pd.DataFrame(),
        )
        result = generate_portfolio(
            _pool(), port_size=2, thresh_out_buffer=0, size_cut=10,
            close_on_size_drop=False, trade_on_next_bar=True, strict_first_bar_top_n=False,
            is_short=False, plot_heatmap=False, record_target_weights=True,
            cost_per_turnover=0.0, optimizer_plan=plan,
        )
        execution_day = result.target_weights.loc[pd.Timestamp("2026-09-02")]
        self.assertEqual(execution_day.loc["A", "weight_target"], 0.4)
        self.assertEqual(execution_day.loc["B", "weight_target"], 0.1)
        no_signal_day = result.target_weights.loc[pd.Timestamp("2026-09-03")]
        self.assertEqual(no_signal_day.loc["A", "weight_target"], 0.4)
        self.assertEqual(no_signal_day.loc["B", "weight_target"], 0.1)

    def test_equal_service_target_sequence_matches_local_accounting_baseline(self) -> None:
        arrays = List.empty_list(types.Array(types.float64, 1, "C", readonly=True))
        for values in ([0.5, 0.5], [0.5, 0.5], []):
            value = np.asarray(values, dtype=np.float64)
            value.setflags(write=False)
            arrays.append(value)
        plan = OptimizerTargetPlan(
            events=np.asarray([1, 1, 0], dtype=np.int8),
            offsets=np.asarray([0, 2, 4, 4], dtype=np.int64),
            symbol_ids=np.asarray([0, 1, 0, 1], dtype=np.int32),
            weights_by_bar=arrays, leases=[], calls=[], decision_targets=pd.DataFrame(),
        )
        common = dict(
            pool=_pool(), port_size=2, thresh_out_buffer=0, size_cut=10,
            close_on_size_drop=False, trade_on_next_bar=True, strict_first_bar_top_n=False,
            is_short=False, plot_heatmap=False, record_target_weights=True,
            cost_per_turnover=0.0,
        )
        local = generate_portfolio(**common)
        service = generate_portfolio(**common, optimizer_plan=plan)
        np.testing.assert_allclose(service.portfolio_returns, local.portfolio_returns, atol=1e-8, rtol=0.0)
        np.testing.assert_allclose(
            service.target_weights["weight_target"], local.target_weights["weight_target"],
            atol=1e-10, rtol=0.0,
        )

    @unittest.skipUnless(os.environ.get("RUN_OPTIMIZER_IPC_TESTS") == "1", "requires UDS permission")
    def test_real_optimizer_process_is_callable_from_eq_client(self) -> None:
        optimizer_repo = Path(__file__).parents[2] / "optimizer"
        if not (optimizer_repo / "service.py").is_file():
            self.skipTest("sibling optimizer checkout is not available")
        with tempfile.TemporaryDirectory() as tmp:
            socket_path = Path(tmp) / "optimizer.sock"
            process = subprocess.Popen(
                [os.fspath(Path(__file__).parents[1] / ".venv/bin/python"), "-m", "service",
                 "--socket", str(socket_path)], cwd=optimizer_repo,
            )
            try:
                deadline = time.monotonic() + 5
                while not socket_path.exists():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        self.fail("optimizer service did not start")
                    time.sleep(0.01)
                with OptimizerClient(socket_path) as client:
                    request = {
                        "schema_version": "1.1", "request_id": "eq-e2e",
                        "data_context": {"as_of": "2026-09-01T15:00:00+08:00",
                                         "planned_execution_at": "2026-09-02T09:30:00+08:00",
                                         "frequency": "daily"},
                        "portfolio_policy": {"long_only": True, "fully_invested": False,
                                             "allow_cash": True, "allow_leverage": False,
                                             "cash_policy": "retain", "target_budget": 0.5},
                        "model": {"type": "equal_weight", "version": "1", "config": {}},
                        "universe": {"asset_ids": ["A", "B"]}, "constraints": [],
                        "options": {"timeout_ms": 5000, "deterministic": True},
                    }
                    lease, _ = client.optimize(request)
                    self.assertIsNotNone(lease.weights)
                    self.assertFalse(lease.weights.flags.owndata)
                    np.testing.assert_array_equal(lease.weights, [0.25, 0.25])
                    lease.release()

                    audit_path = Path(tmp) / "optimizer_calls.jsonl"
                    plan = prepare_optimizer_targets(
                        _pool(), client, portfolio_id="long:2", port_size=2,
                        thresh_out_buffer=0, size_cut=10, close_on_size_drop=False,
                        model={"type": "equal_weight", "version": "1", "config": {}},
                        timeout_ms=5000, audit_path=audit_path,
                    )
                    try:
                        result = generate_portfolio(
                            _pool(), port_size=2, thresh_out_buffer=0, size_cut=10,
                            close_on_size_drop=False, trade_on_next_bar=True,
                            strict_first_bar_top_n=False, is_short=False, plot_heatmap=False,
                            record_target_weights=True, cost_per_turnover=0.0,
                            optimizer_plan=plan,
                        )
                        day = result.target_weights.loc[pd.Timestamp("2026-09-02")]
                        np.testing.assert_array_equal(day["weight_target"].to_numpy(), [0.5, 0.5])
                        self.assertTrue(day["request_id"].notna().all())
                        self.assertIn("request_id", result.positions.columns)
                        self.assertTrue(audit_path.read_text(encoding="utf-8").strip())
                    finally:
                        plan.close()
            finally:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
