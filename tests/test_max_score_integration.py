from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import polars as pl

from data_loader import _encode_dataset
from max_score_backtest import build_max_score_request, generate_max_score_portfolio
from max_score_config import validate_max_score_settings
from max_score_state import ShareAccount, execute_next_bar
from optimizer_client import InProcessOptimizerClient, OptimizerClientError
from providers.benchmark_bounds import relative_to_absolute_bounds
from providers.max_score import LongExposureProvider, SnapshotDataError


def settings(**overrides) -> dict:
    return {"purpose": "engineering_acceptance", "target_budget": 0.8, "max_weight": 0.6,
            "size_constraint": None, "industry_constraint": None,
            "turnover_constraint": {"definition": "stock_half_l1", "limit": 1.0},
            "candidate_policy": "finite_score_eligible", "missing_valuation_policy": "carry_last",
            "no_signal_policy": "error", "source_point_in_time_status": "synthetic",
            "price_convention": "consistent_share_prices", **overrides}


def synthetic_pool(n_days=5):
    records = []
    for day, date in enumerate(pd.bdate_range("2026-09-01", periods=n_days)):
        for sid in range(3):
            records.append({"datetime": date.to_pydatetime(), "date": date.date(),
                            "symbol": f"{sid+1:06d}.XSHE", "turnover": 1e6, "log_size": 3.0-sid,
                            "size_rank": sid+1, "industry": str(sid+1), "index": "TEST", "ret": 0.0,
                            "is_limit_up": False, "is_limit_down": False, "listed_Satisfied": True,
                            "is_ST": False, "normal_days": 20, "pred": float(3-sid),
                            "tradable": True, "can_open": True, "can_open_base": True,
                            "can_trade_buy": True, "can_trade_sell": True,
                            "borrow_available": False, "borrow_rate": 0.0,
                            "borrow_provider": None, "borrow_channel": None,
                            "vwap_ret": 0.0, "execution_vwap": 10.0, "prev_close": 10.0,
                            "bar_close": 10.0})
    return _encode_dataset(pl.DataFrame(records))


def source(path, *, industry=False, available=False):
    metadata = {"exposure_id": "fixture-size", "source": "synthetic", "version": "1",
                "units": "raw", "transformation": "identity", "semantic_status": "explicit"}
    if industry:
        metadata.update(classification="fixture-v1", level="1")
    return {"path": str(path), "date_column": "date", "symbol_column": "symbol", "value_column": "value",
            "available_at_column": "available_at" if available else None, "metadata": metadata}


class RecordingClient:
    schema_version = "1.2"

    def __init__(self):
        self.requests = []
        self.client = InProcessOptimizerClient(schema_version="1.2", max_assets=6000)

    def validate_capacity(self, size):
        self.client.validate_capacity(size)

    def optimize(self, request):
        self.requests.append(deepcopy(request))
        return self.client.optimize(request)

    def close(self):
        self.client.close()


class MaxScoreIntegrationTests(unittest.TestCase):
    def run_pool(self, pool, config=None, *, label=1, costs=0.0):
        client = RecordingClient()
        self.addCleanup(client.close)
        result = generate_max_score_portfolio(pool, client, settings=config or settings(),
                                              model={"type": "max_score", "version": "1", "config": {}},
                                              portfolio_id=f"long:{label}", portfolio_label=label,
                                              size_cut=10, timeout_ms=5000, initial_value=100.0,
                                              cost_per_turnover=costs)
        return result, client.requests

    def test_account_conservation_price_movement_and_cost_partial_fill(self):
        pool = synthetic_pool()
        account = ShareAccount.empty(3, 100)
        account.mark_close(pool, 0, missing_policy="error")
        outcome = execute_next_bar(account, pool, 1, np.array([0], np.int32), np.array([10.]),
                                   cost_rate=0.01, priority=np.array([0]))
        self.assertEqual(outcome["trades"][0]["status"], "partial_cash")
        self.assertAlmostEqual(account.shares[0], 100 / 10.1)
        self.assertAlmostEqual(account.cash, 0)
        account.mark_close(replace(pool, bar_close=np.full(len(pool.pred), 12.0)), 1, missing_policy="error")
        self.assertAlmostEqual(account.nav, 120 / 1.01)
        self.assertAlmostEqual(account.nav, account.cash + float(account.shares @ account.prices))
        self.assertAlmostEqual(outcome["fees"], 100 / 101)

    def test_full_budget_fee_sizing_never_creates_negative_cash(self):
        pool = synthetic_pool()
        for price in (0.37, 1.03, 10.1, 123.45):
            account = ShareAccount.empty(3, 100)
            account.mark_close(pool, 0, missing_policy="error")
            execute_next_bar(account, replace(pool, execution_vwap=np.full(len(pool.pred), price)), 1,
                             np.array([0], np.int32), np.array([1000.]), cost_rate=0.0037,
                             priority=np.array([0]))
            self.assertGreaterEqual(account.cash, 0.0)

    def test_sequential_uses_actual_cash_shares_not_prior_target(self):
        pool = synthetic_pool()
        buy_flags = pool.can_trade_buy.copy()
        buy_flags[3] = False  # highest score purchase fails on first execution
        result, requests = self.run_pool(replace(pool, can_trade_buy=buy_flags))
        first = requests[1]
        np.testing.assert_allclose(first["inputs"]["current_weights"], [0, 0.2, 0], atol=1e-10)
        np.testing.assert_allclose(first["inputs"]["current_cash"], [0.8], atol=1e-10)
        self.assertEqual(len(requests[0]["universe"]["asset_ids"]), 3)
        self.assertAlmostEqual(result.decision_targets.iloc[0]["weight_target"], 0.6)
        execution = [c for c in result.optimizer_calls if c["event"] == "execution"][0]
        self.assertAlmostEqual(execution["actual_post_execution"]["stock_weight"], 0.2)

    def test_decisions_independent_of_portfolio_label_and_old_rank_budget(self):
        left, req_left = self.run_pool(synthetic_pool(), label=1)
        right, req_right = self.run_pool(synthetic_pool(), label=800)
        np.testing.assert_array_equal(left.decision_targets.weight_target, right.decision_targets.weight_target)
        self.assertEqual(len(req_left[0]["universe"]["asset_ids"]), 3)
        self.assertEqual(req_right[0]["portfolio_policy"]["target_budget"], 0.8)

    def test_old_holding_remains_when_not_candidate_and_cannot_increase(self):
        pool = synthetic_pool()
        account = ShareAccount.empty(3, 40.0)
        account.shares[2] = 6
        account.mark_close(pool, 0, missing_policy="error")
        open_flags = pool.can_open_base.copy()
        open_flags[2] = False
        cfg = {**settings(), "size_cut": 10}
        request, ids, _, counts = build_max_score_request(
            replace(pool, can_open_base=open_flags), 0, account, cfg,
            {"type": "max_score", "version": "1", "config": {}}, 5000, {})
        np.testing.assert_array_equal(ids, [0, 1, 2])
        self.assertEqual(counts["candidate_count"], 2)
        self.assertEqual(request["inputs"]["upper_weights"][2], 0.6)
        self.assertAlmostEqual(request["inputs"]["current_cash"][0], 0.4)

    def test_missing_score_for_held_is_explicit_error(self):
        pool = synthetic_pool()
        pred = pool.pred.copy()
        pred[3] = np.nan
        with self.assertRaisesRegex(SnapshotDataError, "MISSING_SCORE_FOR_HELD"):
            self.run_pool(replace(pool, pred=pred))

    def test_missing_row_held_is_preserved_and_rejected_not_dropped(self):
        pool = synthetic_pool()
        # Remove A after its first purchase has actually completed.
        frame = pool.pool_frame.filter(~((pl.col("date") == pd.Timestamp("2026-09-03").date()) &
                                         (pl.col("symbol") == "000001.XSHE"))).drop(["row_idx", "symbol_id"])
        with self.assertRaisesRegex(SnapshotDataError, "MISSING_SCORE_FOR_HELD"):
            self.run_pool(_encode_dataset(frame))

    def test_future_execution_and_prediction_changes_do_not_change_prior_decision(self):
        pool = synthetic_pool()
        _, before = self.run_pool(pool)
        pred, prices, flags = pool.pred.copy(), pool.bar_close.copy(), pool.can_trade_buy.copy()
        pred[3:] = np.tile([1., 2., 3.], len(pool.bars)-1)
        prices[3:] *= 2
        flags[3:] = False
        _, after = self.run_pool(replace(pool, pred=pred, bar_close=prices, can_trade_buy=flags))
        for key in before[0]["inputs"]:
            np.testing.assert_array_equal(before[0]["inputs"][key], after[0]["inputs"][key])
        self.assertEqual(before[0]["portfolio_policy"], after[0]["portfolio_policy"])
        self.assertNotEqual(before[1]["inputs"]["current_weights"].tolist(), after[1]["inputs"]["current_weights"].tolist())

    def test_no_signal_hold_is_explicit_and_zero_budget_is_valid(self):
        pool = synthetic_pool()
        empty = replace(pool, pred=np.full_like(pool.pred, np.nan))
        with self.assertRaisesRegex(SnapshotDataError, "NO_SIGNAL"):
            self.run_pool(empty)
        result, requests = self.run_pool(empty, settings(no_signal_policy="hold"))
        self.assertFalse(requests)
        self.assertTrue(all(c["status"] == "hold_without_new_solution" for c in result.optimizer_calls))
        result, requests = self.run_pool(empty, settings(target_budget=0.0))
        self.assertEqual(len(requests), len(pool.bars)-1)
        self.assertTrue(result.positions.empty)
        self.assertTrue(all(c["target_budget"] == 0 for c in result.optimizer_calls if c["event"] == "liquidate"))

    def test_valuation_carry_and_error_are_explicit(self):
        pool = synthetic_pool()
        account = ShareAccount.empty(3, 50)
        account.shares[0] = 5
        account.mark_close(pool, 0, missing_policy="error")
        close = pool.bar_close.copy()
        close[3] = 0
        changed = replace(pool, bar_close=close)
        self.assertEqual(account.mark_close(changed, 1, missing_policy="carry_last"), 1)
        self.assertEqual(account.nav, 100)
        with self.assertRaisesRegex(SnapshotDataError, "MISSING_VALUATION"):
            account.mark_close(changed, 1, missing_policy="error")

    def test_new_purchase_cannot_use_stale_cached_sizing_price(self):
        pool = synthetic_pool()
        pred, close = pool.pred.copy(), pool.bar_close.copy()
        # C was observed on day zero but not held; day one's score promotes it
        # while that day's close is invalid. Its cached old close is not sizing data.
        pred[3:6] = [1., 2., 10.]
        close[5] = 0.0
        with self.assertRaisesRegex(SnapshotDataError, "MISSING_SIZING_PRICE"):
            self.run_pool(replace(pool, pred=pred, bar_close=close))

    def test_sell_block_and_t_plus_one_preserve_cash_and_shares(self):
        pool = synthetic_pool()
        account = ShareAccount.empty(3, 0.01)
        account.shares[0], account.prices[0] = 10, 10
        day = int(np.datetime64(pool.bars[1], "D").astype(np.int64))
        account.bought_day[0] = day
        frozen = execute_next_bar(account, pool, 1, np.array([0], np.int32), np.array([0.]),
                                  cost_rate=0.0, priority=np.array([0]))
        self.assertEqual(frozen["trades"][0]["status"], "sell_blocked")
        self.assertEqual(account.cash, 0.01)
        flags = pool.can_trade_sell.copy()
        flags[6] = False
        blocked = execute_next_bar(account, replace(pool, can_trade_sell=flags), 2,
                                   np.array([0], np.int32), np.array([0.]), cost_rate=0.0, priority=np.array([0]))
        self.assertEqual(blocked["trades"][0]["status"], "sell_blocked")
        done = execute_next_bar(account, pool, 3, np.array([0], np.int32), np.array([0.]),
                                cost_rate=0.01, priority=np.array([0]))
        self.assertEqual(done["closed"], 1)
        self.assertEqual(account.shares[0], 0)
        self.assertAlmostEqual(account.cash, 99.01)

    def test_source_publication_cutoff_duplicate_and_invalid_industry(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "exposure.parquet"
            frame = pl.DataFrame({"date": [pd.Timestamp("2026-09-01").date()], "symbol": ["000001.XSHE"],
                                  "value": [1.0], "available_at": ["2026-09-02T12:00:00+08:00"]})
            frame.write_parquet(path)
            date = pd.Timestamp("2026-09-01")
            provider = LongExposureProvider(source(path, available=True), start=date, end=date)
            with self.assertRaisesRegex(SnapshotDataError, "MISSING_SIZE_EXPOSURE"):
                provider.snapshot(date, ["000001.SZ"], as_of=date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15))
            for value in (0.0, 1.2, -1.0, "NaN", "inf", "-1", "0", True):
                frame.with_columns(pl.lit(value).alias("value")).write_parquet(path)
                provider = LongExposureProvider(source(path, industry=True), start=date, end=date, industry=True)
                with self.assertRaisesRegex(SnapshotDataError, "INVALID_INDUSTRY"):
                    provider.snapshot(date, ["000001.SZ"], as_of=date)
            frame.with_columns(pl.lit(True).alias("value")).write_parquet(path)
            provider = LongExposureProvider(source(path), start=date, end=date)
            with self.assertRaisesRegex(SnapshotDataError, "INVALID_SIZE_EXPOSURE"):
                provider.snapshot(date, ["000001.SZ"], as_of=date)
            frame.with_columns(pl.lit("2026-09-01T12:00:00").alias("available_at")).write_parquet(path)
            provider = LongExposureProvider(source(path, available=True), start=date, end=date)
            with self.assertRaisesRegex(SnapshotDataError, "INVALID_AVAILABLE_AT"):
                provider.snapshot(date, ["000001.SZ"], as_of=date)
            for key in ("date", "symbol"):
                frame.with_columns(pl.lit(None).cast(frame.schema[key]).alias(key)).write_parquet(path)
                with self.assertRaisesRegex(SnapshotDataError, "INVALID_EXPOSURE_KEY"):
                    LongExposureProvider(source(path), start=date, end=date)
            pl.concat([frame, frame]).write_parquet(path)
            with self.assertRaisesRegex(SnapshotDataError, "DUPLICATE_EXPOSURE_KEY"):
                LongExposureProvider(source(path), start=date, end=date)

    def test_config_does_not_supply_investment_choices_or_allow_unverified_strategy(self):
        with self.assertRaisesRegex(ValueError, "target_budget"):
            validate_max_score_settings(settings(target_budget=None))
        with self.assertRaisesRegex(ValueError, "engineering_acceptance"):
            validate_max_score_settings(settings(purpose="strategy_backtest", price_convention="engineering_unverified"))
        with self.assertRaisesRegex(ValueError, "definition"):
            validate_max_score_settings(settings(turnover_constraint={"limit": 0.3}))

    def test_relative_benchmark_bounds_preserve_full_membership_and_denominators(self):
        meta = {"exposure_id": "SIZE", "field": "factor_value", "version": "v3_0", "unit": "raw", "transform": "identity"}
        arguments = dict(benchmark_asset_ids=["A", "B", "OUTSIDE_CANDIDATES"],
                         benchmark_weights=np.asarray([0.2, 0.3, 0.5], dtype=np.float64),
                         exposure_asset_ids=["OUTSIDE_CANDIDATES", "B", "A"],
                         exposure_values=np.asarray([2., 1., -1.], dtype=np.float64),
                         expected_benchmark_asset_ids=["A", "B", "OUTSIDE_CANDIDATES"],
                         benchmark_complete=True, benchmark_weight_denominator="stock_sleeve",
                         exposure_metadata=meta, benchmark_exposure_metadata=meta,
                         target_budget=0.8, lower_deviation=0.1, upper_deviation=0.2)
        sleeve = relative_to_absolute_bounds(**arguments, deviation_denominator="stock_sleeve")
        nav = relative_to_absolute_bounds(**arguments, deviation_denominator="total_nav")
        self.assertAlmostEqual(sleeve["benchmark_exposure"], 1.1)
        self.assertAlmostEqual(sleeve["lower"], 0.8)
        self.assertAlmostEqual(nav["lower"], 0.78)
        self.assertAlmostEqual(sleeve["upper"], 1.04)
        self.assertAlmostEqual(nav["upper"], 1.08)
        for overrides, code in (({"benchmark_complete": False}, "INVALID_BENCHMARK"),
                                ({"benchmark_weight_denominator": "total_nav"}, "INVALID_BENCHMARK"),
                                ({"benchmark_exposure_metadata": {**meta, "unit": "log_cap"}}, "EXPOSURE_METADATA_MISMATCH"),
                                ({"expected_benchmark_asset_ids": ["A", "B"]}, "INCOMPLETE_BENCHMARK"),
                                ({"benchmark_weights": np.array([0.2, 0.3, 0.0])}, "INVALID_BENCHMARK_WEIGHTS")):
            with self.assertRaisesRegex(SnapshotDataError, code):
                relative_to_absolute_bounds(**{**arguments, **overrides}, deviation_denominator="stock_sleeve")

    def test_all_four_constraints_and_absent_industry_row(self):
        pool = synthetic_pool()
        with TemporaryDirectory() as tmp:
            size_path, industry_path = Path(tmp)/"size.parquet", Path(tmp)/"industry.parquet"
            keys = pool.pool_frame.select(["date", "symbol"])
            keys.with_columns(pl.Series("value", np.tile([-1., 0., 1.], len(pool.bars)))).write_parquet(size_path)
            keys.with_columns(pl.Series("value", np.tile([1, 2, 1], len(pool.bars)))).write_parquet(industry_path)
            bounds = [{"industry_id": "1", "lower": 0.2, "upper": 0.6},
                      {"industry_id": "2", "lower": 0.2, "upper": 0.6},
                      {"industry_id": "absent", "lower": 0.0, "upper": 0.1}]
            cfg = settings(size_constraint={"lower": -0.2, "upper": 0.2, "source": source(size_path)},
                           industry_constraint={"bounds": bounds, "source": source(industry_path, industry=True)})
            result, requests = self.run_pool(pool, cfg)
            self.assertEqual({c["type"] for c in requests[0]["constraints"]},
                             {"trade_bounds", "max_weight", "size", "industry", "turnover"})
            self.assertIn("absent", requests[0]["snapshot"]["industry"]["code_to_id"].values())
            self.assertFalse(result.decision_targets.empty)
            impossible = deepcopy(cfg)
            impossible["industry_constraint"]["bounds"][-1]["lower"] = 0.1
            with self.assertRaises(OptimizerClientError) as context:
                self.run_pool(pool, impossible)
            self.assertEqual(context.exception.response["status"], "infeasible")

    def test_industry_large_integer_and_string_ids_are_exact(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"industry.parquet"
            date = pd.Timestamp("2026-09-01")
            for raw in (9007199254740993, "9007199254740993", "bank", "0001"):
                pl.DataFrame({"date": [date.date()], "symbol": ["000001.XSHE"], "value": [raw]}).write_parquet(path)
                provider = LongExposureProvider(source(path, industry=True), start=date, end=date, industry=True)
                self.assertEqual(provider.snapshot(date, ["000001.SZ"], as_of=date).tolist(), [str(raw)])


if __name__ == "__main__":
    unittest.main()
