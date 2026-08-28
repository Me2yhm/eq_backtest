from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import polars as pl

from data_loader import _attach_daily_price_fields, _validate_prediction_frames
from research import signal_validation


class PlanBehaviorTests(unittest.TestCase):
    def test_daily_vwap30_price_path_derives_previous_close(self) -> None:
        frame = pl.DataFrame(
            {
                "date": ["2018-12-28", "2019-01-02", "2018-12-28", "2019-01-02"],
                "symbol": ["A", "A", "B", "B"],
                "ret": [0.0, 0.1, 0.0, 0.2],
                "close_ex": [10.0, 11.0, 20.0, 24.0],
                "vwap30": [10.0, 10.5, 20.0, 22.0],
            }
        ).with_columns(pl.col("date").str.strptime(pl.Date))

        result = _attach_daily_price_fields(
            frame,
            {"market_columns": {"execution_vwap": "vwap30", "bar_close": "close_ex", "prev_close": None}},
        )
        result = result.sort(["date", "symbol"])

        self.assertEqual(result["prev_close"].to_list(), [None, None, 10.0, 20.0])
        self.assertEqual(result["execution_vwap"].to_list(), [10.0, 20.0, 10.5, 22.0])
        self.assertEqual(result["bar_close"].to_list(), [10.0, 20.0, 11.0, 24.0])

    def test_daily_vwap30_mode_rejects_missing_price_column(self) -> None:
        frame = pl.DataFrame(
            {"date": ["2019-01-02"], "symbol": ["A"], "ret": [0.1], "close_ex": [11.0]}
        ).with_columns(pl.col("date").str.strptime(pl.Date))

        with self.assertRaisesRegex(ValueError, "execution_vwap"):
            _attach_daily_price_fields(
                frame,
                {"market_columns": {"execution_vwap": "vwap30", "bar_close": "close_ex"}},
            )

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
