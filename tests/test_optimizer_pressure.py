"""The long-horizon acceptance must bound peak resources, not only final FDs."""

from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from verify_optimizer_pressure import verify_long_horizon


class OptimizerPressureTests(unittest.TestCase):
    def test_two_thousand_target_days_under_256_file_descriptors(self):
        with tempfile.TemporaryDirectory(prefix="eq-pressure-test-") as temporary:
            root = Path(temporary) / "verification"
            report = verify_long_horizon(root, target_days=2000, nofile=256)
            self.assertEqual(report["status"], "passed", report)
            self.assertEqual(report["comparison"]["events"]["target"], 2000)
            self.assertFalse(report["socket_residual"])
            for mode in ("inprocess", "shared_memory"):
                self.assertTrue(report["runs"][mode]["run_status"]["complete"])


if __name__ == "__main__":
    unittest.main()
