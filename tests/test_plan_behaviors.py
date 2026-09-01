from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl

from data_loader import (
    _attach_daily_price_fields,
    _pool_cache_path,
    _prediction_sources,
    _validate_prediction_frames,
    load_predictions,
)
from market_cache import _parse_args, _refresh_instruments, load_benchmark_returns
from plotting import plot_position_heatmap
from research import signal_validation


class PlanBehaviorTests(unittest.TestCase):
    def test_position_heatmap_uses_fixed_6000_size_rank_range(self) -> None:
        positions = pd.DataFrame(
            {"size_rank": [100, 999_999]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-03"), "B")],
                names=["date", "symbol"],
            ),
        )
        figure = MagicMock()
        axis = MagicMock()

        with (
            TemporaryDirectory() as tmp,
            patch("plotting.plt.subplots", return_value=(figure, axis)),
            patch("plotting.plt.colorbar"),
            patch("plotting.plt.tight_layout"),
            patch("plotting.plt.close"),
        ):
            plot_position_heatmap(positions, output_path=tmp)

        heatmap = axis.imshow.call_args.args[0]
        self.assertEqual(heatmap.shape, (20, 2))
        self.assertEqual(heatmap.iloc[:, 1].sum(), 0)
        self.assertEqual(axis.set_yticklabels.call_args.args[0][-1], "5700–6000")

    def test_run_directory_config_owns_relative_paths_and_output(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp).resolve()
            (run_dir / "config.yml").write_text(
                """
frequency: daily
market_cache_dir: market_cache
benchmark_symbol: 000852
pool_cache_dir: cache
freq_config:
  daily:
    market_data: market.parquet
    preds_dir: predictions
""".lstrip(),
                encoding="utf-8",
            )
            module_name = "_isolated_run_config_test"
            spec = importlib.util.spec_from_file_location(module_name, Path(__file__).parents[1] / "config.py")
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                with patch.dict(os.environ, {"EQ_BACKTEST_RUN_DIR": str(run_dir)}):
                    spec.loader.exec_module(module)
            finally:
                sys.modules.pop(module_name, None)

            self.assertEqual(module.RUN_DIR, run_dir)
            self.assertEqual(module.CONFIG_PATH, run_dir / "config.yml")
            self.assertEqual(module.MARKET_CACHE_DIR, run_dir / "market_cache")
            self.assertEqual(module.BENCHMARK_SYMBOL, "000852")
            self.assertFalse(hasattr(module, "BM_PATH"))
            self.assertFalse(hasattr(module, "EXTERNAL_NAV_PATH"))
            self.assertEqual(module.POOL_CACHE_DIR, run_dir / "cache")
            self.assertEqual(module.FREQ_CONFIG["daily"]["market_data"], run_dir / "market.parquet")
            self.assertEqual(module.FREQ_CONFIG["daily"]["preds_dir"], run_dir / "predictions")
            self.assertEqual(module.OUTPUT_DIR, run_dir / "output_long_4400_daily")

    def test_market_cache_loads_manifest_mapped_benchmarks(self) -> None:
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "market_data"
            cache_dir.mkdir()
            (cache_dir.parent / "manifest.json").write_text(
                """
{"instruments": [
  {"instrument_id": "000852", "rq_symbol": "000852.XSHG", "cache_file": "000852.csv", "first_date": "2010-01-01"},
  {"instrument_id": "512890", "rq_symbol": "512890.XSHG", "cache_file": "512890.csv", "first_date": "2019-01-18"}
]}
""".lstrip(),
                encoding="utf-8",
            )
            (cache_dir / "000852.csv").write_text(
                "date,close,pct_change\n2024-01-02,100,0\n2024-01-03,101,0.01\n2024-01-04,99,-0.019801980198\n",
                encoding="utf-8",
            )
            (cache_dir / "512890.csv").write_text(
                "date,close,pct_change\n2024-01-02,10,0\n2024-01-03,11,0.1\n",
                encoding="utf-8",
            )

            returns = load_benchmark_returns(
                cache_dir, "000852.XSHG", start="2024-01-03", end="2024-01-04"
            )
            alternate = load_benchmark_returns(cache_dir, "512890")

            self.assertEqual(
                returns.index.tolist(), [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
            )
            self.assertEqual(returns.tolist(), [0.01, -0.019801980198])
            self.assertEqual(alternate.tolist(), [0.0, 0.1])
            refresh_sources = _refresh_instruments(cache_dir)
            self.assertEqual([source.symbol for source, _ in refresh_sources], ["000852", "512890"])
            self.assertEqual([seed.isoformat() for _, seed in refresh_sources], ["2010-01-01", "2019-01-18"])

    def test_market_cache_refresh_is_the_default_command(self) -> None:
        args = _parse_args([])
        self.assertEqual(args.command, "refresh")
        self.assertEqual(args.cache_dir, Path("cache/market_data"))

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

    def test_daily_production_sources_normalize_and_clip_before_concat(self) -> None:
        with TemporaryDirectory() as tmp:
            preds_dir = Path(tmp)
            first_path = preds_dir / "predictions_v6_2024.parquet"
            second_path = preds_dir / "predictions_v6_2025.parquet"
            pl.DataFrame({
                "date": ["2024-01-01", "2024-01-02"],
                "symbol": ["A", "A"],
                "prediction": [1.0, 2.0],
                "label": [0.1, 0.2],
            }).write_parquet(first_path)
            pl.DataFrame({
                "date": ["2024-01-02", "2024-01-03"],
                "symbol": ["A", "A"],
                "prediction": [3.0, 4.0],
                "label": [0.3, 0.4],
            }).write_parquet(second_path)
            freq_cfg = {
                "market_data": "daily_market.parquet",
                "preds_dir": str(preds_dir),
                "horizons": [],
                "prediction_sources": [
                    {"file": first_path.name, "end": "2024-01-01"},
                    {"file": second_path.name, "start": "2024-01-02"},
                ],
            }

            sources = _prediction_sources(freq_cfg)
            result = load_predictions(freq_cfg, "2024-01-01", "2024-01-03", sources=sources)

            self.assertEqual(result.columns, ["datetime", "symbol", "pred"])
            self.assertEqual(result["datetime"].dt.strftime("%Y-%m-%d").to_list(), [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ])
            self.assertEqual(result["pred"].to_list(), [1.0, 3.0, 4.0])

            market_path = preds_dir / "daily_market.parquet"
            pl.DataFrame({"date": ["2024-01-01"]}).write_parquet(market_path)
            cache_freq_cfg = {**freq_cfg, "market_data": str(market_path)}
            cache_a = _pool_cache_path(
                preds_dir / "cache",
                cache_freq_cfg,
                market_path,
                sources,
                "2024-01-01",
                "2024-01-03",
                None,
                False,
                10,
                "concat_disjoint",
            )
            changed_sources = _prediction_sources({
                **freq_cfg,
                "prediction_sources": [
                    {"file": first_path.name, "end": "2024-01-02"},
                    {"file": second_path.name, "start": "2024-01-03"},
                ],
            })
            cache_b = _pool_cache_path(
                preds_dir / "cache",
                cache_freq_cfg,
                market_path,
                changed_sources,
                "2024-01-01",
                "2024-01-03",
                None,
                False,
                10,
                "concat_disjoint",
            )
            self.assertNotEqual(cache_a, cache_b)

    def test_daily_production_sources_reject_overlapping_selection(self) -> None:
        with TemporaryDirectory() as tmp:
            preds_dir = Path(tmp)
            first_path = preds_dir / "predictions_v6_2024.parquet"
            second_path = preds_dir / "predictions_v6_2025.parquet"
            for path, prediction in ((first_path, 1.0), (second_path, 2.0)):
                pl.DataFrame({
                    "date": ["2024-01-02"],
                    "symbol": ["A"],
                    "prediction": [prediction],
                    "label": [0.0],
                }).write_parquet(path)
            freq_cfg = {
                "market_data": "daily_market.parquet",
                "preds_dir": str(preds_dir),
                "horizons": [],
                "prediction_sources": [first_path.name, second_path.name],
            }

            with self.assertRaisesRegex(ValueError, "overlap"):
                load_predictions(freq_cfg, "2024-01-01", "2024-01-03")

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
