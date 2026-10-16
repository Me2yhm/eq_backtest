"""Real UDS sessions belong to one portfolio, after input loading finishes."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import pandas as pd
from optimizer.transport.shm.server import OptimizerServer

from optimizer_client import OptimizerClient, OptimizerClientError
import run as runner
from test_optimizer_integration import _pool


class OptimizerSessionLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.socket_path = self.directory / "optimizer.sock"
        self.server = OptimizerServer(self.socket_path, handshake_timeout_s=0.15)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.clients = []
        self.timeline = []
        self.batches = []
        self.exit_error = None
        self.request_timeout_ms = 6000
        self.addCleanup(self._stop)
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.socket_path.exists():
            if not self.thread.is_alive() or time.monotonic() >= deadline:
                self.fail("real optimizer service did not start")
            time.sleep(0.01)

    def _stop(self) -> None:
        for client in self.clients:
            client.close()
        self.server.shutdown()
        self.thread.join(timeout=3)
        self.assertFalse(self.thread.is_alive())
        self.assertFalse(self.socket_path.exists())

    def _record(self, event: str, **details) -> None:
        self.timeline.append({"event": event, "at": time.monotonic(), **details})

    def _factory(self, _config):
        owner = self

        class TrackedClient(OptimizerClient):
            def __init__(self):
                super().__init__(owner.socket_path, request_timeout_ms=owner.request_timeout_ms)
                self.owned_batches = []
                owner._record("constructed")

            def connect(self):
                super().connect()
                owner._record("connected", session_id=self.session_id)

            def validate_capacity(self, n_assets):
                super().validate_capacity(n_assets)
                owner._record("capacity", session_id=self.session_id, port_size=n_assets)

            def optimize_many(self, requests):
                batch = super().optimize_many(requests)
                self.owned_batches.append(batch)
                owner.batches.append(batch)
                session_id = self.session_id
                release = batch._release_callback

                def release_output():
                    release()
                    owner._record("released", session_id=session_id)

                batch._release_callback = release_output
                owner._record("finalized", session_id=session_id, requests=len(requests))
                return batch

            def close(self):
                if self.socket is not None:
                    owner._record("closed", session_id=self.session_id,
                                  released=all(batch.released for batch in self.owned_batches))
                super().close()

            def __exit__(self, *exc_info):
                super().__exit__(*exc_info)
                if owner.exit_error is not None:
                    raise owner.exit_error

        client = TrackedClient()
        self.clients.append(client)
        return client

    def _execute(self, port_sizes, *, load_delay=0.0, report_delay=0.0):
        run_dir = self.directory / "run"
        mode_dir = run_dir / "results"
        run_dir.mkdir()
        pool = _pool()

        def load_pool(**kwargs):
            self._record("load_started", connected=sum(client.socket is not None for client in self.clients))
            time.sleep(load_delay)
            self._record("load_finished", connected=sum(client.socket is not None for client in self.clients))
            return pool

        def plot_holding_periods(_stats, port_size, **kwargs):
            self._record("report_started", port_size=port_size,
                         connected=sum(client.socket is not None for client in self.clients))
            time.sleep(report_delay)
            self._record("report_finished", port_size=port_size,
                         connected=sum(client.socket is not None for client in self.clients))

        with ExitStack() as stack:
            stack.enter_context(patch.multiple(
                runner.cfg, RUN_DIR=run_dir, PORT_SIZES=port_sizes, FREQUENCY="daily", POOL_SIZE=10,
                STRATEGY_MODES=("long_only",), CLOSE_ON_SIZE_DROP=False, THRESH_OUT_BUFFER=0,
                WEIGHT_MODE="equal", DEBUG=False, STRICT_FIRST_BAR_TOP_N=False,
                OPTIMIZER={**runner.cfg.OPTIMIZER, "enabled": True, "mode": "shared_memory",
                           "socket_path": self.socket_path},
            ))
            stack.enter_context(patch.object(runner.cfg, "trade_on_next_bar_for", return_value=True))
            stack.enter_context(patch.object(runner.cfg, "output_dir_for_mode", return_value=mode_dir))
            stack.enter_context(patch.object(runner, "load_benchmark_returns", return_value=pd.Series(
                0.0, index=pd.to_datetime(pool.bars), name="benchmark")))
            stack.enter_context(patch.object(runner, "build_pool", side_effect=load_pool))
            stack.enter_context(patch.object(runner, "create_optimizer_client", side_effect=self._factory))
            stack.enter_context(patch.object(runner, "plot_holding_periods", side_effect=plot_holding_periods))
            stack.enter_context(patch.object(runner, "plot_portfolio_results"))
            stack.enter_context(patch.object(runner, "plot_metrics_table"))
            runner.run()
        self.assertEqual(json.loads((run_dir / "optimizer_run_status.json").read_text()),
                         {"status": "success", "complete": True})

    def test_slow_pool_loading_finishes_before_client_construction_or_connection(self):
        self._execute([2], load_delay=0.3)
        loaded = next(row for row in self.timeline if row["event"] == "load_finished")
        constructed = next(row for row in self.timeline if row["event"] == "constructed")
        connected = next(row for row in self.timeline if row["event"] == "connected")
        self.assertLess(loaded["at"], constructed["at"], self.timeline)
        self.assertLess(loaded["at"], connected["at"], self.timeline)
        self.assertEqual(loaded["connected"], 0)
        self.assertEqual(len(self.batches), 1)
        self.assertTrue(self.batches[0].released)
        self.assertEqual(self.batches[0].copy_bytes, 0)

    def test_each_portfolio_releases_and_closes_before_reporting_and_next_session(self):
        port_sizes = [1, 2, 4, 8]
        self._execute(port_sizes, report_delay=0.3)
        connected = [row for row in self.timeline if row["event"] == "connected"]
        self.assertEqual(len(connected), len(port_sizes), self.timeline)
        self.assertEqual(len({row["session_id"] for row in connected}), len(port_sizes))
        capacities = [row["port_size"] for row in self.timeline if row["event"] == "capacity"]
        self.assertEqual(capacities, port_sizes)
        for index, connection in enumerate(connected):
            session = connection["session_id"]
            released = next(row for row in self.timeline if row["event"] == "released" and row["session_id"] == session)
            closed = next(row for row in self.timeline if row["event"] == "closed" and row["session_id"] == session)
            report = next(row for row in self.timeline if row["event"] == "report_started" and row["port_size"] == port_sizes[index])
            self.assertLess(released["at"], closed["at"])
            self.assertTrue(closed["released"])
            self.assertLess(closed["at"], report["at"])
            self.assertEqual(report["connected"], 0)
            if index + 1 < len(connected):
                self.assertLess(closed["at"], connected[index + 1]["at"])
        self.assertEqual(len(self.batches), len(port_sizes))
        self.assertTrue(all(batch.released and batch.copy_bytes == 0 for batch in self.batches))

    def _failed_status(self):
        status = json.loads((self.directory / "run/optimizer_run_status.json").read_text())
        self.assertFalse(status["complete"])
        self.assertEqual(status["status"], "failed")
        self.assertFalse(list((self.directory / "run").glob("**/metrics_*.csv")))
        return status

    def test_finalized_arena_remains_valid_during_slow_simulation(self):
        generate = runner.generate_portfolio

        def slow(*args, **kwargs):
            result = generate(*args, **kwargs)
            self.assertEqual(len(self.batches), 1)
            self.assertFalse(self.batches[0].released)
            time.sleep(0.45)
            self.assertFalse(self.batches[0].weights.flags.writeable)
            return result

        with patch.object(runner, "generate_portfolio", side_effect=slow):
            self._execute([2])
        self.assertTrue(self.batches[0].released)

    def test_client_exit_error_keeps_simulation_primary_and_separate_cleanup(self):
        primary = ValueError("original simulation failure")
        self.exit_error = RuntimeError("client context exit failure")
        with patch.object(runner, "generate_portfolio", side_effect=primary):
            with self.assertRaises(ValueError) as caught:
                self._execute([2])
        self.assertIs(caught.exception, primary)
        status = self._failed_status()
        self.assertEqual(status["error"], str(primary))
        self.assertEqual(status["cleanup_errors"], [{
            "phase": "client_context", "error_type": "RuntimeError", "error": str(self.exit_error),
        }])
        self.assertTrue(self.batches[0].released)
        self.assertIsNone(self.clients[0].socket)

    def test_client_exit_failure_without_primary_fails_the_run(self):
        self.exit_error = RuntimeError("client context exit failure")
        with self.assertRaises(RuntimeError) as caught:
            self._execute([2])
        self.assertIs(caught.exception, self.exit_error)
        status = self._failed_status()
        self.assertEqual(status["cleanup_status"], "failed")
        self.assertEqual(len(status["cleanup_errors"]), 1)
        self.assertTrue(self.batches[0].released)

    def test_capacity_and_configuration_errors_fail_after_loading_without_reports(self):
        for kind in ("capacity", "configuration"):
            with self.subTest(kind=kind):
                # Each case owns its own dedicated run directory.
                if (self.directory / "run").exists():
                    self.directory = self.directory / "next"
                    self.directory.mkdir()
                if kind == "capacity":
                    self.server.max_assets = self.server.optimizer.max_assets = 1
                    with self.assertRaisesRegex(OptimizerClientError, "cannot accept 2"):
                        self._execute([2])
                else:
                    with patch.object(self, "_factory", side_effect=ValueError("invalid optimizer config")):
                        with self.assertRaisesRegex(ValueError, "invalid optimizer config"):
                            self._execute([2])
                self._failed_status()
                self.assertTrue(any(row["event"] == "load_finished" for row in self.timeline))
                self.assertTrue(all(client.socket is None for client in self.clients))
                self.assertEqual(self.batches, [])

    def test_request_timeout_and_client_exit_failure_preserve_timeout_without_retries(self):
        self.request_timeout_ms = 100
        self.exit_error = RuntimeError("client context exit failure")
        calls = []
        proceed = threading.Event()
        optimize = self.server.optimizer.optimize

        def delayed(*args, **kwargs):
            calls.append(args[0].request_id)
            proceed.wait(timeout=2)
            return optimize(*args, **kwargs)

        with patch.object(self.server.optimizer, "optimize", side_effect=delayed):
            try:
                with self.assertRaisesRegex(OptimizerClientError, "timed out") as caught:
                    self._execute([1, 2])
            finally:
                proceed.set()
        status = self._failed_status()
        self.assertEqual(status["error"], str(caught.exception))
        self.assertEqual(status["cleanup_errors"][0]["phase"], "client_context")
        self.assertEqual(len(self.clients), 1)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(self.clients[0].socket)


if __name__ == "__main__":
    unittest.main()
