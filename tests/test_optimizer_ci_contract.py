from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

import yaml


class OptimizerCIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).resolve().parents[1] / ".github/workflows/optimizer-integration.yml"
        # BaseLoader preserves GitHub's `on` key instead of YAML 1.1 booleans.
        cls.workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)

    def test_push_and_pr_base_events_match_supported_branches(self) -> None:
        triggers = self.workflow["on"]
        cases = [
            ("push", "refs/heads/main", None, True),
            ("push", "refs/heads/5min-old", None, True),
            ("push", "refs/heads/fix/optimizer-session-lifecycle", None, False),
            ("pull_request", "refs/pull/3/merge", "main", True),
            ("pull_request", "refs/pull/3/merge", "5min-old", True),
            ("pull_request", "refs/pull/3/merge", "other", False),
        ]
        for event, ref, base, expected in cases:
            with self.subTest(event=event, ref=ref, base=base):
                # GitHub applies PR branch filters to the destination branch.
                branch = base if event == "pull_request" else ref.removeprefix("refs/heads/")
                self.assertEqual(branch in triggers[event]["branches"], expected)
        self.assertIn("workflow_dispatch", triggers)

    def test_missing_private_dependency_token_fails_before_installation(self) -> None:
        steps = self.workflow["jobs"]["linux"]["steps"]
        install = next(step for step in steps if step.get("name") == "Install frozen private Git dependency")
        environment = os.environ.copy()
        environment.pop("OPTIMIZER_READ_TOKEN", None)
        environment.pop("RUNNER_TEMP", None)
        result = subprocess.run(
            [sys.executable, "-c", install["run"]], env=environment,
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("integration has not run", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})


if __name__ == "__main__":
    unittest.main()
