from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
from numba import types
from numba.typed import List

from data_loader import BacktestDataset, _encode_dataset
from optimizer_adapter import _select_targets
from portfolio import PortfolioResult, generate_portfolio


def _cash_competition_pool(
    execution_predictions: dict[str, float] | None = None,
    *,
    signal_tie: bool = False,
    block_a: bool = False,
    intraday_minutes: int | None = None,
    is_short: bool = False,
) -> BacktestDataset:
    """C cannot exit, leaving only half the capital for two half-weight targets."""
    if intraday_minutes is None:
        bars = pd.date_range("2026-09-01", periods=3, freq="B")
    else:
        bars = pd.date_range("2026-09-01 09:31", periods=3, freq=f"{intraday_minutes}min")
    scores = [
        {"A": 2.0, "B": 1.0, "C": 4.0, "D": 3.0},
        {"A": 4.0, "B": 4.0 if signal_tie else 3.0, "C": 2.0, "D": 1.0},
        execution_predictions or {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
    ]
    market_rows = []
    prediction_rows = []
    for bar_idx, bar in enumerate(bars):
        # Deliberately reverse input symbols; encoding gives deterministic IDs.
        for symbol in ("D", "C", "B", "A"):
            limit_up = bar_idx == 2 and ((symbol == "A" and block_a) or (symbol == "C" and is_short))
            limit_down = bar_idx == 2 and symbol == "C" and not is_short
            can_buy = not limit_up
            can_sell = not limit_down
            if intraday_minutes is not None and bar_idx == 1 and symbol == "D":
                # C is T+1 frozen within this session; leave 0.5 cash initially.
                can_buy = False
            close = 11.0 if bar_idx == 2 and symbol == "A" else 9.0 if bar_idx == 2 and symbol == "B" else 10.0
            market_rows.append({
                "datetime": bar, "date": bar.date(), "symbol": symbol,
                "turnover": 1.0, "log_size": 1.0, "size_rank": 1,
                "industry": "I", "index": "IDX", "ret": 0.0,
                "is_limit_up": limit_up, "is_limit_down": limit_down,
                "listed_Satisfied": True, "is_ST": False, "normal_days": 100,
                "tradable": can_buy and can_sell, "can_open": can_buy and can_sell,
                "can_open_base": True, "can_trade_buy": can_buy, "can_trade_sell": can_sell,
                "borrow_available": is_short, "borrow_rate": 0.0,
                "borrow_provider": "synthetic", "borrow_channel": "synthetic",
                "vwap_ret": 0.0, "execution_vwap": 10.0,
                "prev_close": 10.0, "bar_close": close,
            })
            if symbol in scores[bar_idx]:
                score = scores[bar_idx][symbol]
                prediction_rows.append({"datetime": bar, "symbol": symbol, "pred": -score if is_short else score})
    # A missing prediction row is a null after the same market-left-join used
    # by build_pool; it does not remove the symbol's execution market row.
    frame = pl.DataFrame(market_rows).join(
        pl.DataFrame(prediction_rows), on=["datetime", "symbol"], how="left",
    ).sort(["datetime", "symbol"])
    return _encode_dataset(frame)


def _external_plan(pool: BacktestDataset) -> SimpleNamespace:
    arrays = List.empty_list(types.Array(types.float64, 1, "C", readonly=True))
    for weights in ([0.5, 0.5], [0.5, 0.5], []):
        array = np.asarray(weights, dtype=np.float64)
        array.setflags(write=False)
        arrays.append(array)
    bars = pd.to_datetime(pool.bars)
    decisions = pd.DataFrame([
        {"request_id": f"causality-{index}", "decision_as_of": bars[index],
         "planned_execution_at": bars[index + 1], "symbol": symbol, "weight_target": 0.5}
        for index, symbols in enumerate((("C", "D"), ("A", "B")))
        for symbol in symbols
    ])
    return SimpleNamespace(
        events=np.asarray([1, 1, 0], dtype=np.int8),
        offsets=np.asarray([0, 2, 4, 4], dtype=np.int64),
        symbol_ids=np.asarray([2, 3, 0, 1], dtype=np.int32),
        weights_by_bar=arrays, calls=[], decision_targets=decisions,
    )


def _simulate(pool: BacktestDataset, *, external: bool = False, is_short: bool = False,
              trade_on_next_bar: bool = True) -> PortfolioResult:
    return generate_portfolio(
        pool, port_size=2, thresh_out_buffer=0, size_cut=4,
        close_on_size_drop=False, trade_on_next_bar=trade_on_next_bar,
        is_short=is_short, plot_heatmap=False, record_target_weights=True,
        cost_per_turnover=0.0, portfolio_initial_value=100.0,
        optimizer_plan=_external_plan(pool) if external else None,
    )


class PredictionCausalityTests(unittest.TestCase):
    mutations = {
        "reversed": {"A": -100.0, "B": 100.0, "C": 2.0, "D": 1.0},
        "one_nan": {"A": float("nan"), "B": 3.0, "C": 2.0, "D": 1.0},
        "all_nan": {symbol: float("nan") for symbol in "ABCD"},
        "missing_prediction_row": {"B": 3.0, "C": 2.0, "D": 1.0},
        "tied": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0},
    }

    def assert_same_execution(self, baseline: PortfolioResult, changed: PortfolioResult) -> None:
        # Reported contemporaneous raw pred is deliberately different; every
        # other position field, including signal pred_rank, must be invariant.
        pd.testing.assert_frame_equal(baseline.positions.drop(columns="pred"), changed.positions.drop(columns="pred"))
        pd.testing.assert_frame_equal(baseline.target_weights, changed.target_weights)
        pd.testing.assert_frame_equal(baseline.close_counts, changed.close_counts)
        for field in ("portfolio_returns", "turnover", "cost_turnover", "held_counts"):
            pd.testing.assert_series_equal(getattr(baseline, field), getattr(changed, field))

    def assert_cash_competition(self, result: PortfolioResult, pool: BacktestDataset) -> None:
        day = pd.Timestamp(pool.bars[-1])
        actual = result.positions.xs(day, level="date")["weight_actual"]
        target = result.target_weights.xs(day, level="date")["weight_target"]
        self.assertEqual(actual.to_dict(), {"A": 0.5, "C": 0.5})
        self.assertEqual(target.to_dict(), {"A": 0.5, "B": 0.5})
        # The retained C consumes half the capital; just one of the two new
        # targets can be bought. Their different price moves expose P&L bias.
        self.assertAlmostEqual(result.portfolio_returns.iloc[-1], 0.05)

    def test_local_next_day_buy_priority_ignores_execution_day_predictions(self) -> None:
        pool = _cash_competition_pool()
        baseline = _simulate(pool)
        self.assert_cash_competition(baseline, pool)
        for name, predictions in self.mutations.items():
            with self.subTest(mutation=name):
                self.assert_same_execution(baseline, _simulate(_cash_competition_pool(predictions)))

    def test_external_targets_keep_actual_cash_constrained_buys_causal(self) -> None:
        pool = _cash_competition_pool()
        baseline = _simulate(pool, external=True)
        self.assert_cash_competition(baseline, pool)
        for name, predictions in self.mutations.items():
            with self.subTest(mutation=name):
                changed = _simulate(_cash_competition_pool(predictions), external=True)
                self.assert_same_execution(baseline, changed)
                pd.testing.assert_frame_equal(baseline.decision_targets, changed.decision_targets)

    def test_signal_tie_break_uses_canonical_symbol_order_deterministically(self) -> None:
        for external in (False, True):
            baseline = _simulate(_cash_competition_pool(signal_tie=True), external=external)
            for repetition in range(3):
                with self.subTest(external=external, repetition=repetition):
                    changed_pool = _cash_competition_pool(self.mutations["reversed"], signal_tie=True)
                    changed = _simulate(changed_pool, external=external)
                    self.assert_cash_competition(changed, changed_pool)
                    self.assert_same_execution(baseline, changed)

    def test_execution_flags_change_fills_but_preserve_decisions_and_targets(self) -> None:
        baseline_pool = _cash_competition_pool()
        changed_pool = _cash_competition_pool(block_a=True)
        baseline = _simulate(baseline_pool, external=True)
        changed = _simulate(changed_pool, external=True)
        baseline_events, baseline_selections = _select_targets(
            baseline_pool, port_size=2, thresh_out=2, size_cut=4, close_on_size_drop=False,
        )
        changed_events, changed_selections = _select_targets(
            changed_pool, port_size=2, thresh_out=2, size_cut=4, close_on_size_drop=False,
        )
        np.testing.assert_array_equal(baseline_events, changed_events)
        for selected, changed_selected in zip(baseline_selections, changed_selections):
            np.testing.assert_array_equal(selected, changed_selected)
        pd.testing.assert_frame_equal(baseline.decision_targets, changed.decision_targets)
        pd.testing.assert_frame_equal(baseline.target_weights, changed.target_weights)
        actual = changed.positions.xs(pd.Timestamp(changed_pool.bars[-1]), level="date")["weight_actual"]
        self.assertEqual(actual.to_dict(), {"B": 0.5, "C": 0.5})
        self.assertAlmostEqual(changed.portfolio_returns.iloc[-1], -0.05)

    def test_intraday_next_bar_buy_priority_uses_signal_and_preserves_t1(self) -> None:
        for minutes in (5, 15):
            with self.subTest(minutes=minutes):
                pool = _cash_competition_pool(intraday_minutes=minutes)
                baseline = _simulate(pool)
                self.assert_cash_competition(baseline, pool)
                changed = _simulate(_cash_competition_pool(self.mutations["reversed"], intraday_minutes=minutes))
                self.assert_same_execution(baseline, changed)

    def test_intraday_same_bar_still_uses_current_signal_and_t1_freezes(self) -> None:
        for minutes in (5, 15):
            with self.subTest(minutes=minutes):
                baseline_pool = _cash_competition_pool(intraday_minutes=minutes)
                changed_pool = _cash_competition_pool(self.mutations["reversed"], intraday_minutes=minutes)
                baseline = _simulate(baseline_pool, trade_on_next_bar=False)
                changed = _simulate(changed_pool, trade_on_next_bar=False)
                day = pd.Timestamp(baseline_pool.bars[-1])
                self.assertEqual(baseline.target_weights.xs(day, level="date").index.tolist(), ["A", "B"])
                self.assertEqual(changed.target_weights.xs(day, level="date").index.tolist(), ["B", "C"])
                # C/D were bought on the first bar of this same calendar day;
                # neither can be sold even when the current target changes.
                self.assertEqual(changed.positions.xs(day, level="date")["weight_actual"].to_dict(),
                                 {"C": 0.5, "D": 0.5})

    def test_short_next_bar_priority_remains_signal_ranked(self) -> None:
        baseline = _simulate(_cash_competition_pool(is_short=True), is_short=True)
        changed = _simulate(_cash_competition_pool(self.mutations["reversed"], is_short=True), is_short=True)
        self.assert_same_execution(baseline, changed)
        self.assertAlmostEqual(baseline.portfolio_returns.iloc[-1], -0.05)


if __name__ == "__main__":
    unittest.main()
