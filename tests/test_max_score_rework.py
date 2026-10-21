"""Regression of optimizer R1 through EQ's full state and transport boundary."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import unittest

import numpy as np
import polars as pl

from data_loader import _encode_dataset
from max_score_backtest import build_max_score_request, generate_max_score_portfolio
from max_score_state import ShareAccount, execute_next_bar
from optimizer_client import InProcessOptimizerClient, ShmOptimizerClient
from test_max_score_integration import settings, source, synthetic_pool


class _RecordingClient:
    schema_version = "1.2"

    def __init__(self, backend):
        self.backend = backend
        self.requests = []
        self.responses = []

    def validate_capacity(self, size):
        self.backend.validate_capacity(size)

    def optimize(self, request):
        self.requests.append(deepcopy(request))
        lease, wait = self.backend.optimize(request)
        self.responses.append(deepcopy(lease.response))
        return lease, wait


def _four_asset_pool():
    frame = synthetic_pool(5).pool_frame.drop(["row_idx", "symbol_id"])
    fourth = frame.filter(pl.col("symbol") == "000003.XSHE").with_columns(
        pl.lit("000004.XSHE").alias("symbol"), pl.lit(4).alias("size_rank"),
        pl.lit(-1e12).alias("pred"))
    return _encode_dataset(pl.concat([frame, fourth], how="vertical_relaxed").sort(["datetime", "symbol"]))


class MaxScoreReworkStateTests(unittest.TestCase):
    def _settings(self, pool, directory):
        path = Path(directory) / "size.parquet"
        pool.pool_frame.select("date", "symbol").with_columns(
            pl.col("symbol").replace_strict({"000001.XSHE": 1., "000002.XSHE": -1.,
                                            "000003.XSHE": 0., "000004.XSHE": 0.}).alias("value")
        ).write_parquet(path)
        return settings(target_budget=1.,
            max_weight={"000001.XSHE": 1., "000002.XSHE": 1., "000003.XSHE": 1., "000004.XSHE": 0.},
            size_constraint={"lower": -.1, "upper": .1, "source": source(path)})

    def _assert_sequential(self, backend, directory):
        pool = _four_asset_pool()
        client = _RecordingClient(backend)
        result = generate_max_score_portfolio(pool, client, settings=self._settings(pool, directory),
            model={"type": "max_score", "version": "1", "config": {}},
            portfolio_id="long:rework", portfolio_label=800, size_cut=10,
            timeout_ms=5000, initial_value=100., cost_per_turnover=0., output_dir=directory)
        expected_assets = ["000001.XSHE", "000002.XSHE", "000003.XSHE", "000004.XSHE"]
        self.assertEqual(len(client.requests), 4)
        for request, response in zip(client.requests, client.responses):
            self.assertEqual(request["universe"]["asset_ids"], [s.replace(".XSHE", ".SZ") for s in expected_assets])
            self.assertEqual(response["solution"]["asset_ids"], request["universe"]["asset_ids"])
            self.assertTrue(response["diagnostics"]["optimality_verified"])
            self.assertAlmostEqual(response["diagnostics"]["objective_value"], 2.55, places=10)
        # Every fixed-zero asset remains a row in original order, including
        # decision/target audit output after internal solver elimination.
        for _, group in result.decision_targets.groupby("decision_as_of", sort=False):
            self.assertEqual(group.symbol.tolist(), expected_assets)
            np.testing.assert_allclose(group.weight_target, [.55, .45, 0., 0.], rtol=0, atol=1e-10)
        np.testing.assert_allclose(client.requests[0]["inputs"]["current_weights"], [0., 0., 0., 0.], atol=0)
        for request in client.requests[1:]:
            np.testing.assert_allclose(request["inputs"]["current_weights"], [.55, .45, 0., 0.], rtol=0, atol=1e-10)
            self.assertAlmostEqual(request["inputs"]["current_cash"][0], 0., places=10)
        states = pl.read_parquet(Path(directory) / "account_state_800.parquet").to_pandas()
        np.testing.assert_allclose(states.nav, 100., rtol=0, atol=1e-10)
        np.testing.assert_allclose(states.cash.iloc[1:], 0., rtol=0, atol=1e-10)
        np.testing.assert_allclose(result.turnover.iloc[:2], [0., 1.], rtol=0, atol=1e-10)
        actual = result.positions.reset_index()
        np.testing.assert_allclose(actual[actual.symbol == expected_assets[0]].shares, 5.5, rtol=0, atol=1e-10)
        np.testing.assert_allclose(actual[actual.symbol == expected_assets[1]].shares, 4.5, rtol=0, atol=1e-10)

    def test_r1_optimum_full_output_and_actual_shares_cash_feedback(self):
        with TemporaryDirectory() as directory, InProcessOptimizerClient(schema_version="1.2") as backend:
            self._assert_sequential(backend, directory)

    def test_zero_cap_excluded_old_holding_survives_blocked_execution_and_feedback(self):
        pool = _four_asset_pool()
        account = ShareAccount.empty(4, 80.)
        account.shares[3] = 2.  # genuine old holding: 20% of NAV
        account.mark_close(pool, 0, missing_policy="error")
        opening = pool.can_open_base.copy()
        opening[pool.row_symbol_ids == 3] = False
        selling = pool.can_trade_sell.copy()
        selling[pool.row_symbol_ids == 3] = False
        pool = replace(pool, can_open_base=opening, can_trade_sell=selling)
        config = settings(target_budget=1., max_weight={str(s): 0. if i == 3 else 1.
                                                      for i, s in enumerate(pool.symbols)})
        config["size_cut"] = 10
        with InProcessOptimizerClient(schema_version="1.2") as client:
            request, ids, scores, counts = build_max_score_request(pool, 0, account, config,
                {"type": "max_score", "version": "1", "config": {}}, 5000, {})
            self.assertEqual(counts["candidate_count"], 3)
            self.assertEqual(counts["actual_held_count"], 1)
            np.testing.assert_array_equal(ids, [0, 1, 2, 3])
            lease, _ = client.optimize(request)
            try:
                np.testing.assert_allclose(lease.weights, [1., 0., 0., 0.], rtol=0, atol=1e-10)
                desired = np.array(lease.weights) * account.nav / account.prices[ids]
            finally:
                lease.release()
            execution = execute_next_bar(account, pool, 1, ids, desired, cost_rate=0.,
                                         priority=np.lexsort((ids, -scores)))
            account.mark_close(pool, 1, missing_policy="error")
            old_trade = next(trade for trade in execution["trades"] if trade["symbol_id"] == 3)
            self.assertEqual(old_trade["status"], "sell_blocked")
            self.assertEqual(account.shares[3], 2.)
            feedback, next_ids, _, next_counts = build_max_score_request(pool, 1, account, config,
                {"type": "max_score", "version": "1", "config": {}}, 5000, {})
            np.testing.assert_array_equal(next_ids, [0, 1, 2, 3])
            self.assertEqual(next_counts["actual_held_count"], 2)
            np.testing.assert_allclose(feedback["inputs"]["current_weights"], [.8, 0., 0., .2], rtol=0, atol=1e-10)
            self.assertEqual(feedback["inputs"]["max_weights"][3], 0.)
            self.assertAlmostEqual(feedback["inputs"]["current_cash"][0], 0., places=10)
            self.assertAlmostEqual(account.nav, 100.)
            self.assertAlmostEqual(execution["turnover"], .8)
            # The subsequent solve consumes the real blocked-sale snapshot;
            # target-zero does not erase the still-held asset's input row.
            following, _ = client.optimize(feedback)
            try:
                self.assertEqual(len(following.weights), 4)
                np.testing.assert_allclose(following.weights, [1., 0., 0., 0.], rtol=0, atol=1e-10)
                self.assertAlmostEqual(following.response["diagnostics"]["constraint_checks"]["turnover"]["value"], .2)
            finally:
                following.release()

    @unittest.skipUnless(os.environ.get("RUN_OPTIMIZER_IPC_TESTS") == "1", "requires Linux UDS permission")
    def test_r1_native_subprocess_full_output_and_actual_feedback(self):
        with TemporaryDirectory() as directory:
            socket_path = Path(directory) / "optimizer.sock"
            with (Path(directory) / "service.log").open("w") as log:
                service = subprocess.Popen([sys.executable, "-m", "optimizer.transport.shm.service",
                    "--socket", str(socket_path)], stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 10
                    while not socket_path.exists():
                        if service.poll() is not None or time.monotonic() > deadline:
                            self.fail("independent optimizer service failed to start")
                        time.sleep(.01)
                    with ShmOptimizerClient(socket_path, schema_version="1.2") as backend:
                        self._assert_sequential(backend, directory)
                finally:
                    service.terminate()
                    try:
                        service.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        service.kill()
                        service.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
