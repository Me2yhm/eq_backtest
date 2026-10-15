"""Primary execution failures must survive independent cleanup failures."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import optimizer_adapter as adapter
from optimizer_client import InProcessOptimizerClient, OptimizerClientError
import run as runner
from test_optimizer_integration import _pool


class CleanupPriorityTests(unittest.TestCase):
    def test_target_preparation_keeps_primary_error_and_cleanup_diagnostic(self):
        primary = OptimizerClientError("original request timeout")
        with InProcessOptimizerClient() as client:
            optimize_many = client.optimize_many

            def invoke(requests):
                batch = optimize_many(requests)
                batch._release_callback = MagicMock(side_effect=RuntimeError("release ACK lost"))
                return batch

            with (patch.object(client, "optimize_many", side_effect=invoke),
                  patch.object(adapter, "_append_audit", side_effect=primary)):
                with self.assertRaises(OptimizerClientError) as caught:
                    adapter.prepare_optimizer_targets(
                        _pool(), client, portfolio_id="long:2", port_size=2,
                        thresh_out_buffer=0, size_cut=10, close_on_size_drop=False,
                        model={"type": "equal_weight", "version": "1", "config": {}}, timeout_ms=5000,
                    )
        self.assertIs(caught.exception, primary)
        self.assertIn("release ACK lost", json.dumps(primary.cleanup_errors))

    def test_simulation_failure_and_cleanup_failure_keep_original_status(self):
        primary = ValueError("original simulation failure")
        plan = MagicMock()
        plan.close.side_effect = OptimizerClientError("release ACK lost")
        client = MagicMock()
        client.__enter__.return_value = client
        def execute(factory):
            with runner._optimizer_session(factory, 2) as active_client:
                return runner._mode_result(_pool(), "long_only", port_size=2,
                                           output_dir=tmp, optimizer_client=active_client)
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(runner.cfg, "RUN_DIR", Path(tmp)),
                patch.object(runner.cfg, "PORT_SIZES", [2]),
                patch.object(runner.cfg, "OPTIMIZER", {"enabled": True, "model": {}, "timeout_ms": 5000}),
                patch.object(runner, "create_optimizer_client", return_value=client),
                patch.object(runner, "prepare_optimizer_targets", return_value=plan),
                patch.object(runner, "generate_portfolio", side_effect=primary),
                patch.object(runner, "_run", side_effect=execute),
            ):
                with self.assertRaises(ValueError) as caught:
                    runner.run()
            self.assertIs(caught.exception, primary)
            status = json.loads((Path(tmp) / "optimizer_run_status.json").read_text())
            self.assertFalse(status["complete"])
            self.assertEqual(status["error"], str(primary))
            self.assertIn("release ACK lost", json.dumps(status["cleanup_errors"]))


if __name__ == "__main__":
    unittest.main()
