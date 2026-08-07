from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import polars as pl

from data_loader import (
    _filter_exclude_period,
    _recompute_intraday_vwap_ret,
    _validate_prediction_frames,
)
from research import signal_validation


class PlanBehaviorTests(unittest.TestCase):
    def test_exclude_period_is_inclusive_and_recomputes_cross_gap_return(self) -> None:
        frame = pl.DataFrame({
            "datetime": [
                "2024-03-30 09:31:00",
                "2024-03-31 09:31:00",
                "2024-04-01 09:31:00",
                "2024-04-01 09:36:00",
            ],
            "symbol": ["A", "A", "A", "A"],
            "execution_vwap": [100.0, 105.0, 110.0, 111.0],
            "vwap_ret": [0.0, 0.0, 0.0, 0.0],
        }).with_columns(pl.col("datetime").str.to_datetime())
        filtered = _filter_exclude_period(frame, "datetime", ("2024-03-31", "2024-03-31"))
        result = _recompute_intraday_vwap_ret(filtered)

        self.assertEqual(result["datetime"].dt.date().to_list(), [
            pd.Timestamp("2024-03-30").date(),
            pd.Timestamp("2024-04-01").date(),
            pd.Timestamp("2024-04-01").date(),
        ])
        self.assertAlmostEqual(result["vwap_ret"][0], 0.10)
        self.assertAlmostEqual(result["vwap_ret"][1], 111.0 / 110.0 - 1.0)
        self.assertTrue(result["vwap_ret"][2] is None)

    def test_prediction_files_must_be_disjoint_by_default(self) -> None:
        first = pl.DataFrame({
            "datetime": [pd.Timestamp("2024-01-01")],
            "symbol": ["A"],
            "pred": [1.0],
        })
        second = pl.DataFrame({
            "datetime": [pd.Timestamp("2024-01-02")],
            "symbol": ["A"],
            "pred": [2.0],
        })
        _validate_prediction_frames([first, second], [Path("a.parquet"), Path("b.parquet")], "concat_disjoint")
        with self.assertRaises(ValueError):
            _validate_prediction_frames([first, first], [Path("a.parquet"), Path("b.parquet")], "concat_disjoint")
        _validate_prediction_frames([first, first], [Path("a.parquet"), Path("b.parquet")], "mean")

    def test_signal_validation_uses_same_timestamp_and_three_layers(self) -> None:
        rows = []
        for timestamp in ("2025-01-01 09:31:00", "2025-01-01 09:36:00"):
            for symbol_id in range(5):
                rows.append({
                    "datetime": timestamp,
                    "symbol": str(symbol_id),
                    "pred": float(5 - symbol_id),
                    "vwap_ret": float(symbol_id) / 1000.0,
                    "size_rank": symbol_id + 1,
                    "can_open_base": symbol_id < 4,
                    "can_open": symbol_id < 3,
                })
        pool = SimpleNamespace(pool_frame=pl.DataFrame(rows))
        per_bar, summary = signal_validation(pool, pool_size=4, top_n=2, bottom_n=2)

        self.assertEqual(set(per_bar["layer"].unique().to_list()), {"all", "pool", "executable"})
        self.assertEqual(summary.height, 3)
        executable = per_bar.filter(pl.col("layer") == "executable")
        self.assertEqual(executable["n_obs"].to_list(), [3, 3])
        self.assertLess(executable["ic"].mean(), 0.0)


if __name__ == "__main__":
    unittest.main()
