from __future__ import annotations

import unittest

import pandas as pd

from test_prediction_causality import _cash_competition_pool, _simulate


class PredictionFieldSemanticsTests(unittest.TestCase):
    def test_position_pred_is_execution_observation_while_buys_use_signal_score(self) -> None:
        execution_scores = {"A": -100.0, "B": 100.0, "C": 2.0, "D": 1.0}
        for external in (False, True):
            with self.subTest(external=external):
                baseline = _simulate(_cash_competition_pool(), external=external)
                pool = _cash_competition_pool(execution_scores)
                changed = _simulate(pool, external=external)
                execution_at = pd.Timestamp(pool.bars[-1])
                positions = changed.positions.xs(execution_at, level="date")
                # Only half the cash is free. Signal-day A=4 beats B=3, so A
                # buys first even though the execution-day scores reverse them.
                self.assertEqual(positions["weight_actual"].to_dict(), {"A": 0.5, "C": 0.5})
                self.assertEqual(positions["pred"].to_dict(), {"A": -100.0, "C": 2.0})
                self.assertEqual(baseline.positions.xs(execution_at, level="date").loc["A", "pred"], 4.0)
                self.assertAlmostEqual(changed.portfolio_returns.iloc[-1], 0.05)
                pd.testing.assert_frame_equal(
                    baseline.positions.drop(columns="pred"), changed.positions.drop(columns="pred"),
                )
                pd.testing.assert_frame_equal(baseline.target_weights, changed.target_weights)
                if external:
                    decisions = changed.decision_targets
                    chosen = decisions[decisions["planned_execution_at"] == execution_at]
                    self.assertEqual(chosen["symbol"].tolist(), ["A", "B"])
                    self.assertTrue((chosen["decision_as_of"] == pd.Timestamp(pool.bars[-2])).all())


if __name__ == "__main__":
    unittest.main()
