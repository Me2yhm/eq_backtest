"""Real SHM ownership and primary/cleanup failure priority."""
from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import traceback
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numba import njit
from optimizer.transport.shm import server as service_module
from optimizer.transport.shm.server import OptimizerServer

import optimizer_adapter as adapter
from optimizer_client import OptimizerClient, OptimizerClientError
import run as runner
from test_optimizer_integration import _pool


def _buffer_fds() -> int:
    count = 0
    for fd in Path("/proc/self/fd").iterdir():
        try:
            count += "memfd:optimizer-buffer" in os.readlink(fd)
        except FileNotFoundError:
            pass
    return count


@njit
def _raise_numba_failure() -> None:
    raise ValueError("original Numba simulation failure")


class ShmCleanupPriorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.socket_path = Path(self.tmp.name) / "optimizer.sock"
        self.server = OptimizerServer(self.socket_path)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self._stop)
        deadline = time.monotonic() + 5
        while not self.socket_path.exists():
            if not self.thread.is_alive() or time.monotonic() >= deadline:
                self.fail("real optimizer service did not start")
            time.sleep(0.01)

    def _stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=3)
        self.assertFalse(self.thread.is_alive())
        self.assertFalse(self.socket_path.exists())

    def _client(self, **kwargs):
        client = OptimizerClient(self.socket_path, **kwargs)
        self.addCleanup(client.close)
        client.connect()
        return client

    def _prepare(self, client):
        return adapter.prepare_optimizer_targets(
            _pool(), client, portfolio_id="long:2", port_size=2,
            thresh_out_buffer=0, size_cut=10, close_on_size_drop=False,
            model={"type": "equal_weight", "version": "1", "config": {}}, timeout_ms=5000,
        )

    def _capture_batches(self, client, captured):
        original = client.optimize_many

        def optimize(requests):
            batch = original(requests)
            captured.append(batch)
            return batch

        return patch.object(client, "optimize_many", new=optimize)

    def _bad_ack(self):
        original = service_module.send_packet

        def send(sock, payload, fds=()):
            if payload.get("type") == "CONSUMED_OK":
                payload = {**payload, "buffer_id": "incorrect-ACK-buffer"}
            return original(sock, payload, fds)

        return patch.object(service_module, "send_packet", new=send)

    def _wait_clean(self, baseline: int) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with self.server._connections_lock:
                active = len(self.server._connections)
            if active == 0 and _buffer_fds() == baseline:
                return
            time.sleep(0.01)
        self.fail(f"SHM resources retained: active={active}, buffer_fds={_buffer_fds()}, baseline={baseline}")

    def _run_case(self, *, operation: str = "success", cleanup_failure: bool = False):
        baseline = _buffer_fds()
        client = self._client()
        batches, results, original_errors = [], [], []
        run_dir = Path(self.tmp.name) / f"run-{operation}-{cleanup_failure}"
        run_dir.mkdir()

        def execute(factory):
            with runner._optimizer_session(factory, 2) as active_client:
                results.append(runner._mode_result(
                    _pool(), "long_only", port_size=2, output_dir=str(run_dir), optimizer_client=active_client,
                ))

        def fail_audit(*args, **kwargs):
            error = ValueError("original audit materialization failure")
            original_errors.append(error)
            raise error

        def fail_simulation(*args, **kwargs):
            # A real compiled Numba exception at the simulation boundary; the
            # target plan and its shared output were successfully constructed.
            try:
                _raise_numba_failure()
            except ValueError as error:
                original_errors.append(error)
                raise

        failure = None
        frames = []
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.multiple(
                    runner.cfg, RUN_DIR=run_dir, PORT_SIZES=[2], FREQUENCY="daily", POOL_SIZE=10,
                    CLOSE_ON_SIZE_DROP=False, THRESH_OUT_BUFFER=0, WEIGHT_MODE="equal", DEBUG=False,
                    OPTIMIZER={"enabled": True, "model": {"type": "equal_weight", "version": "1", "config": {}},
                               "timeout_ms": 5000},
                ))
                stack.enter_context(patch.object(runner.cfg, "trade_on_next_bar_for", return_value=True))
                stack.enter_context(patch.object(runner, "create_optimizer_client", new=lambda config: client))
                stack.enter_context(patch.object(runner, "_run", new=execute))
                stack.enter_context(self._capture_batches(client, batches))
                if cleanup_failure:
                    stack.enter_context(self._bad_ack())
                if operation == "audit":
                    stack.enter_context(patch.object(adapter, "_append_audit", new=fail_audit))
                elif operation == "simulation":
                    stack.enter_context(patch.object(runner, "generate_portfolio", new=fail_simulation))
                if cleanup_failure:
                    logs = stack.enter_context(self.assertLogs("optimizer_adapter", level="ERROR"))
                else:
                    logs = None
                try:
                    runner.run()
                except Exception as error:
                    failure = error
                    frames = [frame.name for frame in traceback.extract_tb(error.__traceback__)]
            status = json.loads((run_dir / "optimizer_run_status.json").read_text())
            self.assertEqual(len(batches), 1, "each case must own a real finalized SHM arena")
            self.assertEqual(batches[0].transport, "shared_memory")
            self.assertEqual(batches[0].copy_bytes, 0)
            self.assertTrue(batches[0].released)
            if original_errors:
                self.assertIs(failure, original_errors[0], "cleanup replaced the original exception object")
            return failure, status, frames, results, logs.output if logs else []
        finally:
            if failure is not None:
                traceback.clear_frames(failure.__traceback__)
            for batch in batches:
                if not batch.released:
                    try:
                        batch.release()
                    except OptimizerClientError:
                        pass
            client.close()
            self._wait_clean(baseline)

    def test_real_shm_simulation_and_cleanup_four_case_matrix(self) -> None:
        for primary_failure in (False, True):
            for cleanup_failure in (False, True):
                with self.subTest(primary_failure=primary_failure, cleanup_failure=cleanup_failure):
                    error, status, frames, results, logs = self._run_case(
                        operation="simulation" if primary_failure else "success", cleanup_failure=cleanup_failure,
                    )
                    if not primary_failure and not cleanup_failure:
                        self.assertIsNone(error)
                        self.assertEqual(status, {"status": "success", "complete": True})
                        self.assertEqual(len(results), 1)
                        continue
                    self.assertFalse(status["complete"])
                    self.assertEqual(status["status"], "failed")
                    self.assertEqual(status["error_type"], type(error).__name__)
                    self.assertEqual(status["error"], str(error))
                    if primary_failure:
                        self.assertIsInstance(error, ValueError)
                        self.assertEqual(str(error), "original Numba simulation failure")
                        self.assertIn("_raise_numba_failure", frames)
                    else:
                        self.assertIsInstance(error, OptimizerClientError)
                    if cleanup_failure:
                        self.assertEqual(status["cleanup_status"], "failed")
                        self.assertEqual(len(status["cleanup_errors"]), 1)
                        self.assertIn("output consumption", json.dumps(status["cleanup_errors"]))
                        self.assertFalse(status["cleanup_errors"][0]["retryable"])
                        self.assertIn("output consumption", "\n".join(logs))
                    else:
                        self.assertEqual(status["cleanup_status"], "complete")
                        self.assertEqual(status["cleanup_errors"], [])

    def test_real_batch_audit_and_release_failures_keep_original_status_and_traceback(self) -> None:
        error, status, frames, results, logs = self._run_case(operation="audit", cleanup_failure=True)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), "original audit materialization failure")
        self.assertIn("fail_audit", frames)
        self.assertFalse(status["complete"])
        self.assertEqual(status["error_type"], "ValueError")
        self.assertEqual(status["error"], str(error))
        self.assertEqual(status["cleanup_status"], "failed")
        self.assertIn("output consumption", json.dumps(status["cleanup_errors"]))
        self.assertFalse(results, "audit failure must not enter simulation")
        self.assertIn("output consumption", "\n".join(logs))

    def test_published_legacy_lease_then_timeout_keeps_primary_and_safe_cleanup(self) -> None:
        baseline = _buffer_fds()
        client = self._client(request_timeout_ms=40)
        from datetime import datetime

        context = adapter.DecisionContext(
            "long:2", pd.Timestamp(datetime(2026, 9, 1, 15)), pd.Timestamp(datetime(2026, 9, 2, 9, 30)),
            np.asarray([0, 1], dtype=np.int32),
        )
        payload = adapter._request(
            context, _pool(), schema_version="1.1", port_size=2,
            model={"type": "equal_weight", "version": "1", "config": {}}, timeout_ms=5000,
        )
        lease, _ = client.optimize(payload)
        np.testing.assert_array_equal(lease.weights, [0.5, 0.5])
        plan = adapter.OptimizerTargetPlan(
            np.empty(0, dtype=np.int8), np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.int32),
            [], [lease], [], pd.DataFrame(),
        )
        original = self.server.optimizer.optimize
        entered, finish = threading.Event(), threading.Event()

        def stall(*args, **kwargs):
            entered.set()
            finish.wait(timeout=2)
            return original(*args, **kwargs)

        second = dict(payload, request_id="later-request-timeout")
        primary = None
        try:
            with patch.object(self.server.optimizer, "optimize", new=stall):
                try:
                    client.optimize(second)
                except OptimizerClientError as error:
                    primary = error
                    original_traceback = error.__traceback__
                    with self.assertLogs("optimizer_adapter", level="ERROR") as logs:
                        adapter.close_optimizer_plan(plan, primary)
                    self.assertIs(error.__traceback__, original_traceback)
                finally:
                    finish.set()
                self.assertTrue(entered.is_set())
                self.assertIsNotNone(primary)
                self.assertIn("timed out", str(primary))
                self.assertIn("abandoned session", json.dumps(primary.cleanup_errors))
                self.assertIn("abandoned session", "\n".join(logs.output))
                self.assertFalse(primary.cleanup_errors[0]["retryable"])
                self.assertTrue(lease.released)
                self.assertEqual(plan.leases, [])
                fd = os.open("/dev/null", os.O_RDONLY)
                try:
                    adapter.close_optimizer_plan(plan)
                    lease.release()
                    os.fstat(fd)
                finally:
                    os.close(fd)
        finally:
            finish.set()
            client.close()
            if not lease.released:
                lease.release()
            self._wait_clean(baseline)

    def test_final_target_frame_failure_cleans_real_batch_without_repeating_materialization(self) -> None:
        baseline = _buffer_fds()
        client = self._client()
        batches = []
        primary = ValueError("original final target-frame materialization failure")
        materializations = []

        def materialize(rows):
            materializations.append(len(rows))
            raise primary

        try:
            with (self._capture_batches(client, batches), self._bad_ack(),
                  patch.object(adapter, "_decision_target_frame", new=materialize)):
                with self.assertRaises(ValueError) as caught:
                    self._prepare(client)
            self.assertIs(caught.exception, primary)
            self.assertEqual(len(batches), 1, "failure must occur after one published real SHM arena")
            self.assertTrue(batches[0].released, "target-frame failure leaked its published arena")
            self.assertEqual(materializations, [4], "cleanup must not repeat business materialization")
            self.assertIn("output consumption", json.dumps(primary.cleanup_errors))
            self.assertFalse(primary.cleanup_errors[0]["retryable"])
        finally:
            # Also make a failing pre-fix regression release its actual output.
            traceback.clear_frames(primary.__traceback__)
            for batch in batches:
                if not batch.released:
                    try:
                        batch.release()
                    except OptimizerClientError:
                        pass
            client.close()
            self._wait_clean(baseline)


if __name__ == "__main__":
    unittest.main()
