from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from scripts.verify_optimizer_modes import verify


@unittest.skipUnless(os.environ.get("RUN_OPTIMIZER_IPC_TESTS") == "1", "requires Linux UDS permission")
class OptimizerEndToEndTests(unittest.TestCase):
    def test_real_cli_backends_match_and_failures_remain_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = verify(Path(tmp) / "runs")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["max_weight_abs_error"], 0)
        self.assertEqual(report["max_daily_return_abs_error"], 0)


if __name__ == "__main__":
    unittest.main()
